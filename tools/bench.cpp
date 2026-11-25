// warpmate-bench: times every solver on this machine and checks they agree.
//
//   warpmate-bench [--trials 5] [--threads N]
//       KRvK with the specialised solvers (1 thread, N threads, GPU)
//   warpmate-bench --egtb KQvKR [--cpu-trials 1] [--gpu-trials 3] [--threads N]
//       any endgame with the general solver (N threads vs GPU). Only the
//       requested table is timed; the smaller tables it depends on are
//       solved first, untimed.
//
// Each solver runs several times; the median and best times are reported.
// The GPU gets one untimed warm-up run first, because the very first CUDA
// call pays a one-off start-up cost that isn't part of solving.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <thread>
#include <vector>

#include "egtb/solver.h"
#include "solver/krk_solver.h"
#ifdef WARPMATE_HAS_CUDA
#include "gpu/egtb_gpu.h"
#include "gpu/krk_gpu.h"
#endif

using namespace wm::krk;

namespace {

struct Row {
    std::string name;
    std::vector<double> ms;  // one entry per trial
    std::string note;
};

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

double best(const std::vector<double>& v) { return *std::min_element(v.begin(), v.end()); }

// Solves everything `m` depends on, then times solving `m` itself.
template <typename S>
const wm::egtb::TableResult& solve_timed(S& solver, const wm::egtb::Material& m) {
    for (const auto& d : wm::egtb::direct_dependencies(m)) solver.solve(d);
    return solver.solve(m);
}

int bench_egtb(const std::string& name, int cpu_trials, int gpu_trials, int threads) {
    using namespace wm::egtb;
    Material m;
    std::string error;
    if (!parse_material(name, &m, &error)) {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 2;
    }
    m = canonical(m);
    std::printf("Warpmate benchmark: %s (%llu positions), general solver\n",
                material_name(m).c_str(), static_cast<unsigned long long>(table_size(m)));
    std::printf("CPU threads used: %d\n", threads);

    std::vector<double> cpu_ms;
    std::vector<wm::Value> reference;
    for (int t = 0; t < cpu_trials; ++t) {
        Solver cpu(threads);
        const TableResult& r = solve_timed(cpu, m);
        cpu_ms.push_back(r.seconds * 1000.0);
        std::printf("  CPU trial %d: %.1f ms (%d rounds)\n", t + 1, r.seconds * 1000.0, r.rounds);
        if (t == 0) reference = r.values;
    }
    std::printf("\n| Solver | Median (ms) | Best (ms) | Speedup vs CPU |\n|---|---|---|---|\n");
    const double base = cpu_ms.empty() ? 0.0 : median(cpu_ms);
    if (!cpu_ms.empty()) {
        std::printf("| CPU, %d threads | %.1f | %.1f | 1.0x |\n", threads, base, best(cpu_ms));
    }
    bool match = true;
    bool gpu_ran = false;
#ifdef WARPMATE_HAS_CUDA
    try {
        { GpuSolver warm; solve_timed(warm, m); }  // warm-up, not timed
        gpu_ran = true;
        std::vector<double> total, kernels;
        for (int t = 0; t < gpu_trials; ++t) {
            GpuSolver gpu;
            const TableResult& r = solve_timed(gpu, m);
            total.push_back(r.seconds * 1000.0);
            kernels.push_back(gpu.kernel_ms(material_name(m)));
            if (!reference.empty() && r.values != reference) match = false;
        }
        const auto speedup = [&](double ms) { return base > 0 ? base / ms : 0.0; };
        std::printf("| GPU (total) | %.1f | %.1f | %.1fx |\n", median(total), best(total),
                    speedup(median(total)));
        std::printf("| GPU (kernels only) | %.1f | %.1f | %.1fx |\n", median(kernels),
                    best(kernels), speedup(median(kernels)));
        std::printf("GPU: %s\n", GpuSolver::device_name().c_str());
    } catch (const std::exception& e) {
        std::printf("GPU: unavailable (%s)\n", e.what());
    }
#else
    (void)gpu_trials;
    std::printf("GPU: not built (no CUDA compiler found)\n");
#endif
    if (gpu_ran && !reference.empty()) {
        std::printf("\nCPU and GPU tables identical: %s\n", match ? "yes" : "NO");
    }
    return match ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    int trials = 5;
    int cpu_trials = 1;
    int gpu_trials = 3;
    std::string egtb;
    int threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--trials") && i + 1 < argc) {
            trials = std::max(1, std::atoi(argv[++i]));
        } else if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) {
            threads = std::max(1, std::atoi(argv[++i]));
        } else if (!std::strcmp(argv[i], "--egtb") && i + 1 < argc) {
            egtb = argv[++i];
        } else if (!std::strcmp(argv[i], "--cpu-trials") && i + 1 < argc) {
            cpu_trials = std::max(0, std::atoi(argv[++i]));
        } else if (!std::strcmp(argv[i], "--gpu-trials") && i + 1 < argc) {
            gpu_trials = std::max(1, std::atoi(argv[++i]));
        } else {
            std::fprintf(stderr,
                         "usage: %s [--trials N] [--threads N]\n"
                         "       %s --egtb MATERIAL [--cpu-trials N] [--gpu-trials N] [--threads N]\n",
                         argv[0], argv[0]);
            return 2;
        }
    }
    if (!egtb.empty()) return bench_egtb(egtb, cpu_trials, gpu_trials, threads);

    std::printf("Warpmate benchmark: KRvK, %d trial(s) per solver\n", trials);
    std::printf("CPU threads available: %u\n", std::thread::hardware_concurrency());

    std::vector<Row> rows;
    bool all_match = true;

    // Reference: single-threaded CPU.
    Row single{"CPU, 1 thread", {}, ""};
    SolveResult reference;
    for (int t = 0; t < trials; ++t) {
        reference = solve();
        single.ms.push_back(reference.seconds * 1000.0);
    }
    rows.push_back(single);

    Row parallel{"CPU, " + std::to_string(threads) + " threads", {}, ""};
    for (int t = 0; t < trials; ++t) {
        const SolveResult r = solve_parallel(threads);
        parallel.ms.push_back(r.seconds * 1000.0);
        if (r.table != reference.table) all_match = false;
    }
    rows.push_back(parallel);

#ifdef WARPMATE_HAS_CUDA
    try {
        std::printf("GPU: %s\n", gpu_name().c_str());
        solve_gpu();  // warm-up, not timed

        Row gpu_total{"GPU (total)", {}, "includes GPU memory setup and copying the table back"};
        Row gpu_kernels{"GPU (kernels only)", {}, "time spent running on the GPU"};
        for (int t = 0; t < trials; ++t) {
            GpuTiming timing;
            const SolveResult r = solve_gpu(nullptr, &timing);
            gpu_total.ms.push_back(r.seconds * 1000.0);
            gpu_kernels.ms.push_back(timing.kernel_ms);
            if (r.table != reference.table) all_match = false;
        }
        rows.push_back(gpu_total);
        rows.push_back(gpu_kernels);
    } catch (const std::exception& e) {
        std::printf("GPU: unavailable (%s)\n", e.what());
    }
#else
    std::printf("GPU: not built (no CUDA compiler found)\n");
#endif

    const double base = median(rows[0].ms);
    std::printf("\n| Solver | Median (ms) | Best (ms) | Speedup vs 1 thread |\n");
    std::printf("|---|---|---|---|\n");
    for (const Row& row : rows) {
        const double m = median(row.ms);
        std::printf("| %s | %.2f | %.2f | %.1fx |\n", row.name.c_str(), m, best(row.ms), base / m);
    }
    for (const Row& row : rows) {
        if (!row.note.empty()) std::printf("  %s: %s\n", row.name.c_str(), row.note.c_str());
    }

    std::printf("\nAll solvers produced identical tables: %s\n", all_match ? "yes" : "NO");
    return all_match ? 0 : 1;
}
