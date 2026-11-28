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
    }
    return "?";
}

struct GpuEngine::Impl {
    // Float weights (see FloatWeights for the layouts).
    float *w1t = nullptr, *b1 = nullptr, *w2t = nullptr, *b2 = nullptr, *w3t = nullptr,
          *b3 = nullptr;

    // Scratch memory for the naive variant's in-between results.
    DeviceBuffer<float> x, h1, h2, logits;
    // Buffers for run_host.
    DeviceBuffer<int16_t> codes;
    DeviceBuffer<uint8_t> stm;
    DeviceBuffer<float> probs, host_logits;

    ~Impl() {
        for (float* p : {w1t, b1, w2t, b2, w3t, b3})
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
