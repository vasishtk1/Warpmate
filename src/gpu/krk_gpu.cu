// GPU (CUDA) version of the KRvK solver: one GPU thread per position.
//
// The chess logic and the per-position solver step are NOT rewritten here.
// They live in core/krk.h and solver/krk_retro.h, marked WM_HD, which nvcc
// turns into __host__ __device__. This file only adds the GPU plumbing:
// allocating GPU memory, launching kernels, and copying results back.
#include "gpu/krk_gpu.h"

#include <cuda_runtime.h>

#include <chrono>
#include <stdexcept>
#include <string>

#include "solver/krk_retro.h"

namespace wm::krk {

namespace {

// Every CUDA API call returns an error code. Ignoring them is the most common
// CUDA bug: a failed call silently leaves garbage behind and the program
// crashes (or gives wrong answers) somewhere else. CUDA_CHECK turns any
// failure into an exception that names the call and the line.
#define CUDA_CHECK(call)                                                               \
    do {                                                                               \
        const cudaError_t err_ = (call);                                               \
        if (err_ != cudaSuccess) {                                                     \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err_) + \
                                     " at " __FILE__ ":" + std::to_string(__LINE__) +  \
                                     " (" #call ")");                                  \
        }                                                                              \
    } while (0)

// Threads per block. 256 is a common default: a multiple of 32 (the warp
// size) and enough threads per block to keep the GPU busy.
constexpr int kThreadsPerBlock = 256;

// Enough blocks to give every position its own thread.
constexpr int kNumBlocks = (kNumPositions + kThreadsPerBlock - 1) / kThreadsPerBlock;

// ----------------------------------------------------------------------------
// Kernels. A __global__ function runs on the GPU, launched from the CPU with
// kernel<<<blocks, threads>>>(...). Every thread runs the same code; each one
// works out which position it owns from its block and thread numbers.
// ----------------------------------------------------------------------------

// Round 0: mark invalid positions, checkmates and stalemates.
__global__ void init_kernel(Value* table, unsigned int* checkmates, unsigned int* stalemates) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    // The last block can have spare threads past the end of the table.
    if (i >= kNumPositions) return;
    const Value v = init_value(i);
    table[i] = v;
    // atomicAdd lets many threads safely add to the same counter. A plain
    // "++*checkmates" would lose updates when threads collide.
    if (v == 0) atomicAdd(checkmates, 1u);
    if (v == kDraw) atomicAdd(stalemates, 1u);
}

// One round of the backward solve. This is the CPU solver's loop body, with
// the loop replaced by "one thread per position".
//
// The table is updated in place while other threads read it. That is safe
// here because of the parity rule (see krk_retro.h): this round only writes
// values equal to `ply` and never makes decisions based on them, so a
// neighbour's write landing early or late can't change any thread's answer.
__global__ void step_kernel(Value* table, int ply, unsigned int* changed) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kNumPositions) return;
    if (table[i] != kUnknown) return;  // already solved (most threads exit here)
    const Value v = retro_step(table, i, ply);
    if (v != kUnknown) {
        table[i] = v;
        atomicAdd(changed, 1u);
    }
}

// Whatever is still unknown after the last round is a draw.
__global__ void finalize_kernel(Value* table) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kNumPositions) return;
    if (table[i] == kUnknown) table[i] = kDraw;
}

// Frees GPU memory and events even if an exception is thrown halfway.
struct DeviceBuffers {
    Value* table = nullptr;
    unsigned int* counters = nullptr;  // [0] changed, [1] checkmates, [2] stalemates
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    ~DeviceBuffers() {
        if (table) cudaFree(table);
        if (counters) cudaFree(counters);
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
    }
};

}  // namespace

std::string gpu_name() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    return props.name;
}

SolveResult solve_gpu(const RoundCallback& on_round, GpuTiming* timing) {
    SolveResult result;
    const auto wall_start = std::chrono::steady_clock::now();

    // GPU memory is separate from CPU memory: cudaMalloc reserves space on
    // the GPU, and data moves between the two with cudaMemcpy.
    DeviceBuffers d;
    CUDA_CHECK(cudaMalloc(&d.table, kNumPositions * sizeof(Value)));
    CUDA_CHECK(cudaMalloc(&d.counters, 3 * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d.counters, 0, 3 * sizeof(unsigned int)));
    unsigned int* d_changed = d.counters;

    // CUDA events time work on the GPU itself, which is more accurate than a
    // CPU clock because kernel launches return before the GPU finishes.
    CUDA_CHECK(cudaEventCreate(&d.start));
    CUDA_CHECK(cudaEventCreate(&d.stop));
    CUDA_CHECK(cudaEventRecord(d.start));

    init_kernel<<<kNumBlocks, kThreadsPerBlock>>>(d.table, d.counters + 1, d.counters + 2);
    // Launch errors (bad configuration, etc.) are reported by cudaGetLastError.
    CUDA_CHECK(cudaGetLastError());

    // The CPU drives the rounds: launch one step kernel per round, then read
    // back how many positions changed to decide whether to keep going.
    for (int ply = 1; ply < kDraw; ++ply) {
        CUDA_CHECK(cudaMemset(d_changed, 0, sizeof(unsigned int)));
        step_kernel<<<kNumBlocks, kThreadsPerBlock>>>(d.table, ply, d_changed);
        CUDA_CHECK(cudaGetLastError());

        // This copy waits for the kernel to finish before reading the count.
        unsigned int changed = 0;
        CUDA_CHECK(cudaMemcpy(&changed, d_changed, sizeof changed, cudaMemcpyDeviceToHost));

        const RoundStats stats{ply, changed};
        result.rounds.push_back(stats);
        if (on_round) on_round(stats);
        if (changed == 0) break;
    }

    finalize_kernel<<<kNumBlocks, kThreadsPerBlock>>>(d.table);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(d.stop));
    CUDA_CHECK(cudaEventSynchronize(d.stop));

    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, d.start, d.stop));

    // Copy the finished table and the counters back to the CPU.
    const auto copy_start = std::chrono::steady_clock::now();
    result.table.resize(kNumPositions);
    CUDA_CHECK(cudaMemcpy(result.table.data(), d.table, kNumPositions * sizeof(Value),
                          cudaMemcpyDeviceToHost));
    unsigned int counters[3];
    CUDA_CHECK(cudaMemcpy(counters, d.counters, sizeof counters, cudaMemcpyDeviceToHost));
    const auto end = std::chrono::steady_clock::now();

    result.checkmates = counters[1];
    result.stalemates = counters[2];
    result.seconds = std::chrono::duration<double>(end - wall_start).count();
    if (timing) {
        timing->kernel_ms = kernel_ms;
        timing->copy_ms = std::chrono::duration<double, std::milli>(end - copy_start).count();
    }
    return result;
}

}  // namespace wm::krk
