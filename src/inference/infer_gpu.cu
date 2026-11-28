// Warpmate's CUDA inference kernels. See infer_gpu.h for the variants.
//
// Shapes: n boards; 769 inputs -> 256 -> 64 -> 3. All weights are copied to
// the GPU once, when the engine is created, and stay there.
#include "inference/infer_gpu.h"

#include <stdexcept>
#include <string>
#include <vector>

#include "gpu/cuda_check.h"

namespace wm::infer {

namespace {

constexpr int kThreads = 256;  // threads per block for the simple kernels

// Number of blocks needed to give `total` threads one item each.
unsigned blocks_for(long long total) {
    return static_cast<unsigned>((total + kThreads - 1) / kThreads);
}

// ---------------------------------------------------------------------------
// Variant kNaive: one kernel per step, results passed through GPU memory.
// Each kernel gives every thread one output number to compute.

// Step 1: turn the compact boards into the network's 769 inputs (the same as
// features() in ml/model.py). One thread per input number: n x 769 threads.
__global__ void expand_kernel(const int16_t* codes, int slots, const uint8_t* stm, int n,
                              float* x) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<long long>(n) * kInputs) return;
    const int b = static_cast<int>(i / kInputs);
    const int k = static_cast<int>(i % kInputs);
    float v = 0.0f;
    if (k == kStmInput) {
        v = stm[b] == 0 ? 1.0f : 0.0f;
    } else {
        for (int s = 0; s < slots; ++s)
            if (codes[b * slots + s] == k) v = 1.0f;
    }
    x[i] = v;
}

// Steps 2-4: one fully connected layer, out = act(in . W + bias). One thread
// per output number: n x n_out threads. Thread (b, j) reads all of board b's
// inputs and column j of the weights.
//
// The weights are input-major (wt[k * n_out + j]), so the 32 threads of a warp,
// which have neighbouring j, read 32 neighbouring floats at each step k: the
// GPU serves that with a single wide memory read ("coalesced"). All threads
// of the warp read the same in[b * n_in + k], which the GPU sends to all of
// them at once ("broadcast").
__global__ void dense_layer_kernel(const float* in, int n_in, const float* wt, const float* bias,
                                   int n_out, bool relu, int n, float* out) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<long long>(n) * n_out) return;
    const int b = static_cast<int>(i / n_out);
    const int j = static_cast<int>(i % n_out);
    const float* x = in + static_cast<long long>(b) * n_in;
    float acc = bias[j];
    for (int k = 0; k < n_in; ++k) acc += x[k] * wt[k * n_out + j];
    out[i] = relu ? fmaxf(acc, 0.0f) : acc;
}

// The same softmax as softmax3 on the CPU. Inlined into every kernel that needs it.
__device__ inline void softmax3_device(const float* logits, float* probs) {
    const float m = fmaxf(logits[0], fmaxf(logits[1], logits[2]));
    const float e0 = expf(logits[0] - m), e1 = expf(logits[1] - m), e2 = expf(logits[2] - m);
    const float sum = e0 + e1 + e2;
    probs[0] = e0 / sum;
    probs[1] = e1 / sum;
    probs[2] = e2 / sum;
}

// Step 5: softmax, one thread per board.
__global__ void softmax_kernel(const float* logits, int n, float* probs) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n) return;
    softmax3_device(logits + 3 * b, probs + 3 * b);
}

// ---------------------------------------------------------------------------
// Variants kFused and kSparse: the whole network in ONE kernel.
//
// Each block of 256 threads handles kBoards boards from start to finish. The
// in-between results (the 256 and 64 hidden numbers per board) never go to
// GPU memory: they stay in the block's shared memory, a small, fast scratch
// area on the chip that all threads of a block can read and write. That
// removes 4 of the naive version's 5 kernel launches and all its trips
// through GPU memory.
//
//   kSparse = false ("fused"):  layer 1 multiplies all 769 inputs, exactly
//                               like the naive version (and PyTorch)
//   kSparse = true  ("sparse"): layer 1 only adds up the weight rows of the
//                               3-6 inputs that are on (see network.h)
//
// Several boards per block means each weight read from memory is used for
// all of them, which matters most for layer 2 (64 KB of weights).

constexpr int kBlock = 256;  // threads per block: one per layer-1 neuron
static_assert(kBlock == kH1, "layer 1 uses one thread per neuron");
static_assert(kBlock % kH2 == 0 && kH2 == 64, "layer 2 splits the block into groups of 64");

__device__ inline bool is_piece_device(int code) { return code >= 0 && code < kStmInput; }

// Last layer + softmax for one board, done by one warp (32 threads). Lane l
// multiplies h2[l] and h2[l + 32]; the 32 partial sums are then added up
// with warp shuffles: each step, lane i adds the value held by lane
// i + offset, halving the number of partial sums (16, 8, 4, 2, 1).
__device__ inline void output_layer_warp(const float* h2, const float* __restrict__ w3,
                                         const float* __restrict__ b3, int lane, long long board,
                                         bool active, float* probs, float* logits) {
    float p[kOutputs];
#pragma unroll
    for (int o = 0; o < kOutputs; ++o)
        p[o] = h2[lane] * w3[o * kH2 + lane] + h2[lane + 32] * w3[o * kH2 + lane + 32];
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
#pragma unroll
        for (int o = 0; o < kOutputs; ++o) p[o] += __shfl_down_sync(0xffffffffu, p[o], offset);
    if (lane == 0 && active) {
        float out[kOutputs];
#pragma unroll
        for (int o = 0; o < kOutputs; ++o) out[o] = p[o] + b3[o];
        softmax3_device(out, probs + 3 * board);
        if (logits)
#pragma unroll
            for (int o = 0; o < kOutputs; ++o) logits[3 * board + o] = out[o];
    }
}

template <bool kSparse, int kBoards>
__global__ void __launch_bounds__(kBlock)
    fused_kernel(const int16_t* __restrict__ codes, int slots, const uint8_t* __restrict__ stm,
                 int n, const float* __restrict__ w1t, const float* __restrict__ b1,
                 const float* __restrict__ w2t, const float* __restrict__ b2,
                 const float* __restrict__ w3, const float* __restrict__ b3, float* probs,
                 float* logits) {
    static_assert(kBoards % (kBlock / kH2) == 0 && kBoards % (kBlock / 32) == 0,
                  "boards must split evenly between layer-2 groups and warps");
    // Shared memory: one copy per block, visible to all its threads.
    __shared__ int16_t s_codes[kBoards * kMaxSlots];
    __shared__ uint8_t s_stm[kBoards];
    __shared__ float h1[kBoards][kH1];
    __shared__ float h2[kBoards][kH2];
    __shared__ float x[kSparse ? 1 : kBoards * kInputs];  // dense inputs (fused only)

    const int t = threadIdx.x;
    const long long first = static_cast<long long>(blockIdx.x) * kBoards;
    const int count = static_cast<int>(min(static_cast<long long>(kBoards), n - first));

    // Copy this block's boards into shared memory. Missing boards (past the
    // end of the batch) become empty boards; their results are never written.
    for (int i = t; i < kBoards * slots; i += kBlock)
        s_codes[i] = i / slots < count ? codes[first * slots + i] : kPad;
    if (t < kBoards) s_stm[t] = t < count ? stm[first + t] : 1;
    // A barrier: no thread continues until every thread of the block has
    // reached it, so the copies above are finished before anyone reads them.
    __syncthreads();

    // ---- Layer 1: thread j computes neuron j for every board of the block.
    const int j = t;
    float acc[kBoards];
    if constexpr (kSparse) {
        // Add the weight row (256 numbers, contiguous) of each piece on the
        // board. The 32 threads of a warp read 32 neighbouring floats of the
        // row: one coalesced read.
#pragma unroll
        for (int b = 0; b < kBoards; ++b) {
            float a = b1[j];
            for (int s = 0; s < slots; ++s) {
                const int c = s_codes[b * slots + s];
                if (is_piece_device(c)) a += w1t[c * kH1 + j];
            }
            if (s_stm[b] == 0) a += w1t[kStmInput * kH1 + j];
            acc[b] = a;
        }
    } else {
        // Build the 769 inputs of each board in shared memory (zeros, then a
        // 1 per piece), then multiply all of them, as PyTorch does.
        for (int i = t; i < kBoards * kInputs; i += kBlock) x[i] = 0.0f;
        __syncthreads();
        if (t < kBoards) {
            for (int s = 0; s < slots; ++s) {
                const int c = s_codes[t * slots + s];
                if (is_piece_device(c)) x[t * kInputs + c] = 1.0f;
            }
            x[t * kInputs + kStmInput] = s_stm[t] == 0 ? 1.0f : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int b = 0; b < kBoards; ++b) acc[b] = b1[j];
        for (int k = 0; k < kInputs; ++k) {
            const float w = w1t[k * kH1 + j];  // read once, used for every board
#pragma unroll
            for (int b = 0; b < kBoards; ++b) acc[b] += x[b * kInputs + k] * w;
        }
    }
#pragma unroll
    for (int b = 0; b < kBoards; ++b) h1[b][j] = fmaxf(acc[b], 0.0f);
    __syncthreads();

    // ---- Layer 2: 4 groups of 64 threads. Thread (g, j2) computes neuron j2
    // for boards g, g + 4, g + 8, ... Each weight is read once and used for
    // all of that thread's boards; the h1 values are the same for the whole
    // warp, so shared memory broadcasts them.
    {
        constexpr int kGroups = kBlock / kH2;
        constexpr int kPer = kBoards / kGroups;
        const int j2 = t % kH2, g = t / kH2;
        float a[kPer];
#pragma unroll
        for (int i = 0; i < kPer; ++i) a[i] = b2[j2];
        for (int k = 0; k < kH1; ++k) {
            const float w = w2t[k * kH2 + j2];
#pragma unroll
            for (int i = 0; i < kPer; ++i) a[i] += h1[g + i * kGroups][k] * w;
        }
#pragma unroll
        for (int i = 0; i < kPer; ++i) h2[g + i * kGroups][j2] = fmaxf(a[i], 0.0f);
    }
    __syncthreads();

    // ---- Layer 3 + softmax: warp w handles boards w, w + 8, ...
    {
        constexpr int kWarps = kBlock / 32;
        const int warp = t / 32, lane = t % 32;
        for (int b = warp; b < kBoards; b += kWarps)
            output_layer_warp(h2[b], w3, b3, lane, first + b, b < count, probs, logits);
    }
}

// Boards per block. The dense version keeps 769 inputs per board in shared
// memory (3 KB each), so it fits fewer boards in the 48 KB a block can use.
constexpr int kFusedBoards = 8;
constexpr int kSparseBoards = 16;

// ---------------------------------------------------------------------------

// A GPU buffer that grows when a bigger batch arrives (and never shrinks).
template <class T>
struct DeviceBuffer {
    T* ptr = nullptr;
    size_t capacity = 0;

    T* get(size_t count) {
        if (count > capacity) {
            if (ptr) CUDA_CHECK(cudaFree(ptr));
            ptr = nullptr;
            CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
            capacity = count;
        }
        return ptr;
    }
    void release() {
        if (ptr) cudaFree(ptr);
        ptr = nullptr;
        capacity = 0;
    }
};

template <class T>
T* upload(const std::vector<T>& host) {
    T* dev = nullptr;
    CUDA_CHECK(cudaMalloc(&dev, host.size() * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(dev, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    return dev;
}

}  // namespace

const char* variant_name(Variant v) {
    switch (v) {
        case Variant::kNaive: return "naive";
        case Variant::kFused: return "fused";
        case Variant::kSparse: return "sparse";
    }
    return "?";
}

struct GpuEngine::Impl {
    // Float weights (see FloatWeights for the layouts).
    float *w1t = nullptr, *b1 = nullptr, *w2t = nullptr, *b2 = nullptr, *w3t = nullptr,
          *w3 = nullptr, *b3 = nullptr;

    // Scratch memory for the naive variant's in-between results.
    DeviceBuffer<float> x, h1, h2, logits;
    // Buffers for run_host.
    DeviceBuffer<int16_t> codes;
    DeviceBuffer<uint8_t> stm;
    DeviceBuffer<float> probs, host_logits;

    ~Impl() {
        for (float* p : {w1t, b1, w2t, b2, w3t, w3, b3})
            if (p) cudaFree(p);
        for (auto* buf : {&x, &h1, &h2, &logits, &probs, &host_logits}) buf->release();
        codes.release();
        stm.release();
    }
};

GpuEngine::GpuEngine(const FloatWeights& w) : impl_(new Impl) {
    try {
        impl_->w1t = upload(w.w1t);
        impl_->b1 = upload(w.b1);
        impl_->w2t = upload(w.w2t);
        impl_->b2 = upload(w.b2);
        // The naive layer kernel wants every weight matrix input-major, so the
        // last layer (stored one row per output) is transposed here.
        std::vector<float> w3t(kH2 * kOutputs);
        for (int o = 0; o < kOutputs; ++o)
            for (int k = 0; k < kH2; ++k) w3t[k * kOutputs + o] = w.w3[o * kH2 + k];
        impl_->w3t = upload(w3t);
        impl_->w3 = upload(w.w3);
        impl_->b3 = upload(w.b3);
    } catch (...) {
        delete impl_;
        throw;
    }
}

GpuEngine::~GpuEngine() { delete impl_; }

void GpuEngine::run(Variant v, const int16_t* codes, int slots, const uint8_t* stm, int n,
                    float* probs, float* logits, void* stream_ptr) {
    if (n <= 0) return;
    if (slots < 1 || slots > kMaxSlots)
        throw std::invalid_argument("slots must be between 1 and " + std::to_string(kMaxSlots));
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    Impl& m = *impl_;

    switch (v) {
        case Variant::kNaive: {
            float* x = m.x.get(static_cast<size_t>(n) * kInputs);
            float* h1 = m.h1.get(static_cast<size_t>(n) * kH1);
            float* h2 = m.h2.get(static_cast<size_t>(n) * kH2);
            float* out = logits ? logits : m.logits.get(static_cast<size_t>(n) * kOutputs);
            // <<<blocks, threads, shared memory bytes, stream>>>
            expand_kernel<<<blocks_for(static_cast<long long>(n) * kInputs), kThreads, 0,
                            stream>>>(codes, slots, stm, n, x);
            dense_layer_kernel<<<blocks_for(static_cast<long long>(n) * kH1), kThreads, 0,
                                 stream>>>(x, kInputs, m.w1t, m.b1, kH1, true, n, h1);
            dense_layer_kernel<<<blocks_for(static_cast<long long>(n) * kH2), kThreads, 0,
                                 stream>>>(h1, kH1, m.w2t, m.b2, kH2, true, n, h2);
            dense_layer_kernel<<<blocks_for(static_cast<long long>(n) * kOutputs), kThreads, 0,
                                 stream>>>(h2, kH2, m.w3t, m.b3, kOutputs, false, n, out);
            softmax_kernel<<<blocks_for(n), kThreads, 0, stream>>>(out, n, probs);
            break;
        }
        case Variant::kFused:
            fused_kernel<false, kFusedBoards>
                <<<static_cast<unsigned>((n + kFusedBoards - 1) / kFusedBoards), kBlock, 0,
                   stream>>>(codes, slots, stm, n, m.w1t, m.b1, m.w2t, m.b2, m.w3, m.b3, probs,
                             logits);
            break;
        case Variant::kSparse:
            fused_kernel<true, kSparseBoards>
                <<<static_cast<unsigned>((n + kSparseBoards - 1) / kSparseBoards), kBlock, 0,
                   stream>>>(codes, slots, stm, n, m.w1t, m.b1, m.w2t, m.b2, m.w3, m.b3, probs,
                             logits);
            break;
        default:
            throw std::invalid_argument("unknown inference variant");
    }
    // Kernel launches don't return errors directly; this picks up a bad launch
    // (e.g. too many threads) right away instead of at some later call.
    CUDA_CHECK(cudaGetLastError());
}

void GpuEngine::run_host(Variant v, const int16_t* codes, int slots, const uint8_t* stm, int n,
                         float* probs, float* logits) {
    if (n <= 0) return;
    Impl& m = *impl_;
    const size_t nc = static_cast<size_t>(n) * slots, no = static_cast<size_t>(n) * kOutputs;
    int16_t* d_codes = m.codes.get(nc);
    uint8_t* d_stm = m.stm.get(n);
    float* d_probs = m.probs.get(no);
    float* d_logits = logits ? m.host_logits.get(no) : nullptr;
    CUDA_CHECK(cudaMemcpy(d_codes, codes, nc * sizeof(int16_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_stm, stm, n, cudaMemcpyHostToDevice));
    run(v, d_codes, slots, d_stm, n, d_probs, d_logits, nullptr);
    // cudaMemcpy waits for the kernels (same default stream) before copying.
    CUDA_CHECK(cudaMemcpy(probs, d_probs, no * sizeof(float), cudaMemcpyDeviceToHost));
    if (logits)
        CUDA_CHECK(cudaMemcpy(logits, d_logits, no * sizeof(float), cudaMemcpyDeviceToHost));
}

std::string GpuEngine::device_name() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        cudaGetLastError();  // clear the error so later calls aren't affected
        return "";
    }
    int dev = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&prop, dev) != cudaSuccess)
        return "";
    return prop.name;
}

}  // namespace wm::infer
