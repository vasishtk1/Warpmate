// CUDA basics: the smallest useful GPU programs.
//
//   1. Print which GPU we have.
//   2. Add two lists of numbers on the GPU (the "hello world" of CUDA).
//   3. Square a list of numbers on the GPU.
//
// Each one follows the same five steps every CUDA program uses:
//   (a) allocate memory on the GPU          cudaMalloc
//   (b) copy input from CPU to GPU          cudaMemcpy(..., HostToDevice)
//   (c) launch a kernel                     kernel<<<blocks, threads>>>(...)
//   (d) copy the result back                cudaMemcpy(..., DeviceToHost)
//   (e) free the GPU memory                 cudaFree
// and checks the GPU's answer against the same calculation on the CPU.
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Stop with a clear message if any CUDA call fails. Without this, errors are
// silent and show up later as wrong answers.
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error: %s at %s:%d (%s)\n",             \
                         cudaGetErrorString(err_), __FILE__, __LINE__, #call); \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Kernels: __global__ functions run on the GPU. When launched, the GPU starts
// many threads, all running this same function. Threads are grouped into
// blocks; each thread finds "its" element from its block and thread number:
//
//   index = blockIdx.x * blockDim.x + threadIdx.x
//
//   block 0: threads 0..255   -> elements 0..255
//   block 1: threads 0..255   -> elements 256..511   ...and so on.
// ---------------------------------------------------------------------------

__global__ void add_kernel(const float* a, const float* b, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];  // the last block may have spare threads
}

__global__ void square_kernel(float* data, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = data[i] * data[i];
}

// Times GPU work with CUDA events (recorded on the GPU's own timeline).
struct GpuTimer {
    cudaEvent_t start, stop;
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }
    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void begin() { CUDA_CHECK(cudaEventRecord(start)); }
    float end_ms() {
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));  // wait for the GPU to finish
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

double cpu_ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void print_gpu_info() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, device));
    std::printf("== 1. Your GPU ==\n");
    std::printf("  Name:                      %s\n", p.name);
    std::printf("  Streaming multiprocessors: %d  (the GPU's 'cores', each runs many threads)\n",
                p.multiProcessorCount);
    std::printf("  Max threads per block:     %d\n", p.maxThreadsPerBlock);
    std::printf("  Warp size:                 %d  (threads that run in lockstep)\n", p.warpSize);
    std::printf("  GPU memory:                %.1f GB\n", p.totalGlobalMem / 1e9);
    std::printf("  Compute capability:        %d.%d\n\n", p.major, p.minor);
}

void vector_add(int n) {
    std::printf("== 2. Add two lists of %d numbers ==\n", n);
    const size_t bytes = n * sizeof(float);
    std::vector<float> a(n), b(n), out(n), expected(n);
    for (int i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(2 * i);
    }

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i) expected[i] = a[i] + b[i];
    const double cpu_ms = cpu_ms_since(t0);

    // (a) allocate GPU memory
    float *d_a, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    GpuTimer copy_in, kernel, copy_out;

    // (b) copy inputs to the GPU
    copy_in.begin();
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));
    const float in_ms = copy_in.end_ms();

    // (c) launch: enough blocks of 256 threads to cover all n elements
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    kernel.begin();
    add_kernel<<<blocks, threads>>>(d_a, d_b, d_out, n);
    CUDA_CHECK(cudaGetLastError());  // catches launch problems
    const float kernel_ms = kernel.end_ms();

    // (d) copy the result back
    copy_out.begin();
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    const float out_ms = copy_out.end_ms();

    // (e) free GPU memory
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));

    int wrong = 0;
    for (int i = 0; i < n; ++i) wrong += out[i] != expected[i];

    std::printf("  Launched %d blocks x %d threads = %d threads\n", blocks, threads, blocks * threads);
    std::printf("  CPU loop:             %8.3f ms\n", cpu_ms);
    std::printf("  GPU copy in:          %8.3f ms\n", in_ms);
    std::printf("  GPU kernel:           %8.3f ms\n", kernel_ms);
    std::printf("  GPU copy out:         %8.3f ms\n", out_ms);
    std::printf("  Results correct:      %s\n", wrong == 0 ? "yes" : "NO");
    std::printf("  Lesson: the math is fast; moving data to and from the GPU often costs more.\n\n");
}

void square(int n) {
    std::printf("== 3. Square %d numbers ==\n", n);
    const size_t bytes = n * sizeof(float);
    std::vector<float> data(n), expected(n);
    for (int i = 0; i < n; ++i) data[i] = static_cast<float>(i % 1000) * 0.5f;

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i) expected[i] = data[i] * data[i];
    const double cpu_ms = cpu_ms_since(t0);

    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMemcpy(d_data, data.data(), bytes, cudaMemcpyHostToDevice));

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    GpuTimer kernel;
    kernel.begin();
    square_kernel<<<blocks, threads>>>(d_data, n);
    CUDA_CHECK(cudaGetLastError());
    const float kernel_ms = kernel.end_ms();

    CUDA_CHECK(cudaMemcpy(data.data(), d_data, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_data));

    int wrong = 0;
    for (int i = 0; i < n; ++i) wrong += std::fabs(data[i] - expected[i]) > 1e-3f;

    std::printf("  CPU loop:             %8.3f ms\n", cpu_ms);
    std::printf("  GPU kernel:           %8.3f ms\n", kernel_ms);
    std::printf("  Results correct:      %s\n\n", wrong == 0 ? "yes" : "NO");
}

int main() {
    print_gpu_info();
    // The first CUDA call sets up the GPU context, which is slow (tens of ms).
    // Do it before timing anything so it doesn't distort the numbers.
    CUDA_CHECK(cudaFree(nullptr));
    vector_add(1 << 20);   // about 1 million
    square(1 << 24);       // about 16 million
    return 0;
}
