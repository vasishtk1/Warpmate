// GPU (CUDA) version of the general endgame solver: one thread per position.
//
// The chess rules, un-moves and the per-position step come unchanged from
// egtb/ (all WM_HD), shared with the CPU solver. Two algorithms:
//   * full scan: every round checks every unsolved position (the original)
//   * frontier:  every round checks only positions that can change, found by
//                walking back (un-moves) from last round's solved positions
// GPU-specific details:
//   * sub-tables: captures and promotions look up other, already-solved
//     tables, so those tables live in GPU memory, and the kernels receive a
//     GPU-side list of them
//   * 64-bit position numbers, since 4-piece tables have 33 million rows
//     (and 5-piece tables would have 2 billion)
#include "gpu/egtb_gpu.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "egtb/retro.h"
#include "egtb/unmove.h"
#include "gpu/cuda_check.h"

namespace wm::egtb {

namespace {

constexpr int kThreadsPerBlock = 256;

// Each thread's position number. blockIdx.x * blockDim.x is computed in 64
// bits so large tables don't overflow.
__device__ inline uint64_t thread_index() {
    return static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
}

// Adds up a true/false flag across the whole block and makes ONE atomicAdd
// per block instead of one per thread. __syncthreads_count is a built-in
// barrier that also returns how many threads in the block passed `flag`, so
// every thread of the block must reach this call (no early returns before it).
__device__ inline void count_block(bool flag, unsigned long long* total) {
    const int n = __syncthreads_count(flag);
    if (threadIdx.x == 0 && n > 0) atomicAdd(total, static_cast<unsigned long long>(n));
}

// Round 0. Also counts the positions left unsolved (used for statistics).
// The Context (material + GPU pointer to the sub-table list) is passed by
// value: kernel arguments are copied to the GPU at launch.
__global__ void init_kernel(Context ctx, Value* table, uint64_t size, unsigned long long* unknown) {
    const uint64_t i = thread_index();
    bool is_unknown = false;
    if (i < size) {
        const Value v = init_value(ctx, i);
        table[i] = v;
        is_unknown = v == kUnknown;
    }
    count_block(is_unknown, unknown);
}

// ----- Full scan (the original algorithm) ----------------------------------

// One round, updating the table in place. This is safe: a round only writes
// values equal to `ply`, and the round's decisions never depend on values
// equal to `ply` (see egtb/retro.h).
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

// ----- Frontier (only check positions that can change) ---------------------
//
// Each round has two kernels:
//   1. mark_parents_kernel: one thread per position solved last round. It
//      walks back through un-moves and marks each unsolved parent with the
//      round number ("check me this round").
//   2. frontier_step_kernel: one thread per position, but only marked
//      positions (or ones a capture/promotion can decide this round) run the
//      expensive per-position step. Everything else exits after reading a
//      few bytes. Newly solved positions are appended to the frontier list
//      for the next round.

// Before round 1: record which rounds captures and promotions can decide
// each position (see "Frontier solving" in egtb/retro.h), and list the
// checkmates found in round 0 as the first frontier.
__global__ void prepare_kernel(Context ctx, const Value* table, uint64_t size, uint8_t* win_round,
                               uint8_t* loss_round, uint32_t* frontier,
                               unsigned long long* frontier_size) {
    const uint64_t i = thread_index();
    if (i >= size) return;
    const Value v = table[i];
    if (v == kUnknown && ctx.num_subs > 0) {
        sub_triggers(ctx, table, i, &win_round[i], &loss_round[i]);
    } else if (v == 0) {
        // atomicAdd returns the old value, which is a free slot in the list.
        const unsigned long long slot = atomicAdd(frontier_size, 1ULL);
        frontier[slot] = static_cast<uint32_t>(i);
    }
}

// Visitor for un-moves: marks a parent if it's still unsolved. Many threads
// may mark the same parent; they all write the same byte, so no atomic is
// needed.
struct MarkParents {
    const Material* m;
    const Value* table;
    uint8_t* mark;
    uint8_t round;
    __device__ bool operator()(const Position& p) {
        const uint64_t index = encode(*m, p);
        if (table[index] == kUnknown) mark[index] = round;
        return true;
    }
};

__global__ void mark_parents_kernel(Context ctx, const Value* table, const uint32_t* frontier,
                                    unsigned long long frontier_size, uint8_t* mark,
                                    uint8_t round) {
    const uint64_t k = thread_index();
    if (k >= frontier_size) return;
    MarkParents visit{&ctx.m, table, mark, round};
    for_each_unmove(ctx.m, decode(ctx.m, frontier[k]), visit);
}

// counters[0]: positions solved this round (also the frontier list length)
// counters[1]: positions checked this round (statistics only)
__global__ void frontier_step_kernel(Context ctx, Value* table, uint64_t size, int ply,
                                     const uint8_t* mark, const uint8_t* win_round,
                                     const uint8_t* loss_round, uint32_t* frontier,
                                     unsigned long long* counters) {
    const uint64_t i = thread_index();
    const auto round = static_cast<uint8_t>(ply);
    bool check = false;
    if (i < size && table[i] == kUnknown) {
        check = mark[i] == round || win_round[i] == round || loss_round[i] == round;
    }
    if (check) {
        // In-place update, safe for the same reason as step_kernel.
        const Value v = retro_step(ctx, table, i, ply);
        if (v != kUnknown) {
            table[i] = v;
            const unsigned long long slot = atomicAdd(&counters[0], 1ULL);
            frontier[slot] = static_cast<uint32_t>(i);
        }
    }
    count_block(check, &counters[1]);  // every thread reaches this
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

GpuSolver::GpuSolver(Algorithm algorithm) : algorithm_(algorithm) {
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
    if (size > UINT32_MAX) throw std::runtime_error("table too large for 32-bit position lists");

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

    // counters[0..1] are per-round counts; counters[2] is the unknown count
    // after round 0.
    Value* d_table = nullptr;
    unsigned long long* d_counters = nullptr;
    CUDA_CHECK(cudaMalloc(&d_table, size * sizeof(Value)));
    CUDA_CHECK(cudaMalloc(&d_counters, 3 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_counters, 0, 3 * sizeof(unsigned long long)));

    // Frontier-only buffers.
    uint8_t* d_marks = nullptr;  // mark, win_round, loss_round: size bytes each
    uint32_t* d_frontier = nullptr;
    if (algorithm_ == Algorithm::kFrontier) {
        CUDA_CHECK(cudaMalloc(&d_marks, 3 * size));
        CUDA_CHECK(cudaMemset(d_marks, 0, 3 * size));
        CUDA_CHECK(cudaMalloc(&d_frontier, size * sizeof(uint32_t)));
    }
    uint8_t* d_mark = d_marks;
    uint8_t* d_win_round = d_marks ? d_marks + size : nullptr;
    uint8_t* d_loss_round = d_marks ? d_marks + 2 * size : nullptr;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    const unsigned int blocks = blocks_for(size);
    init_kernel<<<blocks, kThreadsPerBlock>>>(ctx, d_table, size, &d_counters[2]);
    CUDA_CHECK(cudaGetLastError());

    TableResult result;
    unsigned long long frontier_size = 0;
    if (algorithm_ == Algorithm::kFrontier) {
        prepare_kernel<<<blocks, kThreadsPerBlock>>>(ctx, d_table, size, d_win_round, d_loss_round,
                                                     d_frontier, &d_counters[0]);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(&frontier_size, &d_counters[0], sizeof frontier_size,
                              cudaMemcpyDeviceToHost));
    }

    // Keep going until a round changes nothing AND we're past the longest
    // distance in any sub-table (a late capture can still decide a position).
    int ply = 1;
    for (; ply < kDraw; ++ply) {
        CUDA_CHECK(cudaMemset(d_counters, 0, 2 * sizeof(unsigned long long)));
        if (algorithm_ == Algorithm::kFullScan) {
            step_kernel<<<blocks, kThreadsPerBlock>>>(ctx, d_table, size, ply, &d_counters[0]);
        } else {
            if (frontier_size > 0) {
                mark_parents_kernel<<<blocks_for(frontier_size), kThreadsPerBlock>>>(
                    ctx, d_table, d_frontier, frontier_size, d_mark, static_cast<uint8_t>(ply));
                CUDA_CHECK(cudaGetLastError());
            }
            frontier_step_kernel<<<blocks, kThreadsPerBlock>>>(ctx, d_table, size, ply, d_mark,
                                                               d_win_round, d_loss_round,
                                                               d_frontier, d_counters);
        }
        CUDA_CHECK(cudaGetLastError());
        // This copy also waits for the round's kernels to finish.
        unsigned long long counts[2] = {0, 0};
        CUDA_CHECK(cudaMemcpy(counts, d_counters, sizeof counts, cudaMemcpyDeviceToHost));
        frontier_size = counts[0];
        RoundStats stats;
        stats.solved = counts[0];
        stats.checked = counts[1];  // full scan fills this in below
        result.round_stats.push_back(stats);
        if (counts[0] == 0 && ply > max_sub + 1) break;
    }

    finalize_kernel<<<blocks, kThreadsPerBlock>>>(d_table, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (algorithm_ == Algorithm::kFullScan) {
        // A full-scan round checks every position still unsolved, so the
        // count follows from round 0's unknown count without extra atomics.
        unsigned long long unknown = 0;
        CUDA_CHECK(cudaMemcpy(&unknown, &d_counters[2], sizeof unknown, cudaMemcpyDeviceToHost));
        for (RoundStats& s : result.round_stats) {
            s.checked = unknown;
            unknown -= s.solved;
        }
    }

    result.m = m;
    result.rounds = ply;
    result.values.resize(size);
    CUDA_CHECK(cudaMemcpy(result.values.data(), d_table, size * sizeof(Value),
                          cudaMemcpyDeviceToHost));
    cudaFree(d_counters);
    if (d_marks) cudaFree(d_marks);
    if (d_frontier) cudaFree(d_frontier);
    if (d_subs) cudaFree(d_subs);

    device_tables_[name] = d_table;  // keep it on the GPU for larger endgames
    kernel_ms_[name] = ms;
    result.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();
    return result;
}

}  // namespace wm::egtb
