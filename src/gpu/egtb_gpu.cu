// GPU (CUDA) version of the general endgame solver: one thread per position.
//
// The chess rules and the per-position step come unchanged from egtb/ (all
// WM_HD). What's new compared with the KRvK GPU solver:
//   * sub-tables: captures and promotions look up other, already-solved
//     tables, so those tables live in GPU memory, and the kernels receive a
//     GPU-side list of them
//   * 64-bit position numbers, since 4-piece tables have 33 million rows
//     (and 5-piece tables would have 2 billion)
#include "gpu/egtb_gpu.h"

#include <algorithm>
#include <chrono>
#include <vector>

#include "egtb/retro.h"
#include "gpu/cuda_check.h"

namespace wm::egtb {

namespace {

constexpr int kThreadsPerBlock = 256;

// Each thread's position number. blockIdx.x * blockDim.x is computed in 64
// bits so large tables don't overflow.
__device__ inline uint64_t thread_index() {
    return static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
}

// The Context (material + GPU pointer to the sub-table list) is passed by
// value: kernel arguments are copied to the GPU at launch.
__global__ void init_kernel(Context ctx, Value* table, uint64_t size) {
    const uint64_t i = thread_index();
    if (i >= size) return;
    table[i] = init_value(ctx, i);
}

// One round, updating the table in place. As in the KRvK solver this is safe:
// a round only writes values equal to `ply`, and the round's decisions never
// depend on values equal to `ply` (see egtb/retro.h).
__global__ void step_kernel(Context ctx, Value* table, uint64_t size, int ply,
                            unsigned long long* changed) {
    const uint64_t i = thread_index();
    if (i >= size) return;
    if (table[i] != kUnknown) return;
    const Value v = retro_step(ctx, table, i, ply);
    if (v != kUnknown) {
        table[i] = v;
        atomicAdd(changed, 1ULL);
    }
}

__global__ void finalize_kernel(Value* table, uint64_t size) {
    const uint64_t i = thread_index();
    if (i >= size) return;
    if (table[i] == kUnknown) table[i] = kDraw;
}

unsigned int blocks_for(uint64_t size) {
    return static_cast<unsigned int>((size + kThreadsPerBlock - 1) / kThreadsPerBlock);
}

}  // namespace

GpuSolver::GpuSolver() {
    // Move generation and sub-table lookups use a fair amount of per-thread
    // scratch memory; raise the per-thread stack limit so deeper call chains
    // never run out.
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 8192));
}

GpuSolver::~GpuSolver() {
    for (auto& [name, ptr] : device_tables_) cudaFree(ptr);
}

std::string GpuSolver::device_name() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    return props.name;
}

const TableResult& GpuSolver::solve(const Material& requested) {
    const Material m = canonical(requested);
    const std::string name = material_name(m);
    auto it = tables_.find(name);
    if (it != tables_.end()) return it->second;

    for (const Material& d : direct_dependencies(m)) solve(d);

    TableResult result = solve_one(m);
    auto& stored = tables_[name] = std::move(result);
    if (on_table) on_table(stored);
    return stored;
}

TableResult GpuSolver::solve_one(const Material& m) {
    const auto wall_start = std::chrono::steady_clock::now();
    const std::string name = material_name(m);
    const uint64_t size = table_size(m);

    // Sub-tables: GPU pointers to the already-solved tables this one needs.
    // The list itself must also be in GPU memory for the kernels to read it.
    std::vector<SubTable> subs;
    int max_sub = 0;
    for (const Material& d : direct_dependencies(m)) {
        const std::string dn = material_name(d);
        subs.push_back(SubTable{d, device_tables_.at(dn)});
        for (Value v : tables_.at(dn).values) {
            if (is_decided(v)) max_sub = std::max<int>(max_sub, v);
        }
    }
    SubTable* d_subs = nullptr;
    if (!subs.empty()) {
        CUDA_CHECK(cudaMalloc(&d_subs, subs.size() * sizeof(SubTable)));
        CUDA_CHECK(cudaMemcpy(d_subs, subs.data(), subs.size() * sizeof(SubTable),
                              cudaMemcpyHostToDevice));
    }
    const Context ctx{m, d_subs, static_cast<int>(subs.size())};

    Value* d_table = nullptr;
    unsigned long long* d_changed = nullptr;
    CUDA_CHECK(cudaMalloc(&d_table, size * sizeof(Value)));
    CUDA_CHECK(cudaMalloc(&d_changed, sizeof(unsigned long long)));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    const unsigned int blocks = blocks_for(size);
    init_kernel<<<blocks, kThreadsPerBlock>>>(ctx, d_table, size);
    CUDA_CHECK(cudaGetLastError());

    // Keep going until a round changes nothing AND we're past the longest
    // distance in any sub-table (a late capture can still decide a position).
    int ply = 1;
    for (; ply < kDraw; ++ply) {
        CUDA_CHECK(cudaMemset(d_changed, 0, sizeof(unsigned long long)));
        step_kernel<<<blocks, kThreadsPerBlock>>>(ctx, d_table, size, ply, d_changed);
        CUDA_CHECK(cudaGetLastError());
        unsigned long long changed = 0;
        CUDA_CHECK(cudaMemcpy(&changed, d_changed, sizeof changed, cudaMemcpyDeviceToHost));
        if (changed == 0 && ply > max_sub + 1) break;
    }

    finalize_kernel<<<blocks, kThreadsPerBlock>>>(d_table, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    TableResult result;
    result.m = m;
    result.rounds = ply;
    result.values.resize(size);
    CUDA_CHECK(cudaMemcpy(result.values.data(), d_table, size * sizeof(Value),
                          cudaMemcpyDeviceToHost));
    cudaFree(d_changed);
    if (d_subs) cudaFree(d_subs);

    device_tables_[name] = d_table;  // keep it on the GPU for larger endgames
    kernel_ms_[name] = ms;
    result.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();
    return result;
}

}  // namespace wm::egtb
