// warpmate-bench: times the solvers on one endgame and checks that they all
// produce identical tables.
//
//   warpmate-bench                   KRvK with default settings
//   warpmate-bench KQvKR [--cpu-trials 1] [--gpu-trials 3] [--threads N]
//                        [--cpu-algo frontier|full] [--rounds]
//
// Solvers timed: the CPU solver (frontier algorithm unless --cpu-algo full),
// and on CUDA builds both GPU algorithms, full scan (the original) and
// frontier. --cpu-trials 0 skips the CPU. --rounds prints, for every round,
// how many positions each GPU algorithm checked and how many were solved.
//
// Only the requested table is timed; the smaller tables it depends on are
// solved first, untimed. Each solver runs several times and the median and
// best times are reported. The GPU gets one untimed warm-up run first,
// because the very first CUDA call pays a one-off start-up cost that isn't
// part of solving.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <thread>
#include <vector>

#include "egtb/solver.h"
#ifdef WARPMATE_HAS_CUDA
#include "gpu/egtb_gpu.h"
#endif

using namespace wm;
using namespace wm::egtb;

namespace {

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

double best(const std::vector<double>& v) { return *std::min_element(v.begin(), v.end()); }

// Solves everything `m` depends on, then times solving `m` itself.
template <typename S>
const TableResult& solve_timed(S& solver, const Material& m) {
    for (const auto& d : direct_dependencies(m)) solver.solve(d);
    return solver.solve(m);
}

struct Row {
    std::string name;
    std::vector<double> ms;
};

#ifdef WARPMATE_HAS_CUDA
uint64_t total_checked(const TableResult& r) {
    uint64_t n = 0;
    for (const RoundStats& s : r.round_stats) n += s.checked;
    return n;
}
#endif

}  // namespace

int main(int argc, char** argv) {
    std::string name = "KRvK";
    int cpu_trials = 1;
    int gpu_trials = 3;
    int threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    Algorithm cpu_algo = Algorithm::kFrontier;
    bool show_rounds = false;
    bool named = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) {
            threads = std::max(1, std::atoi(argv[++i]));
        } else if (!std::strcmp(argv[i], "--cpu-trials") && i + 1 < argc) {
            cpu_trials = std::max(0, std::atoi(argv[++i]));
        } else if (!std::strcmp(argv[i], "--gpu-trials") && i + 1 < argc) {
            gpu_trials = std::max(1, std::atoi(argv[++i]));
        } else if (!std::strcmp(argv[i], "--cpu-algo") && i + 1 < argc) {
            const std::string a = argv[++i];
            if (a != "full" && a != "frontier") {
                std::fprintf(stderr, "error: --cpu-algo must be full or frontier\n");
                return 2;
            }
            cpu_algo = a == "full" ? Algorithm::kFullScan : Algorithm::kFrontier;
        } else if (!std::strcmp(argv[i], "--rounds")) {
            show_rounds = true;
        } else if (argv[i][0] != '-' && !named) {
            name = argv[i];
            named = true;
        } else {
            std::fprintf(stderr,
                         "usage: %s [MATERIAL] [--cpu-trials N] [--gpu-trials N] [--threads N]\n"
                         "          [--cpu-algo frontier|full] [--rounds]\n",
                         argv[0]);
            return 2;
        }
    }

    Material m;
    std::string error;
    if (!parse_material(name, &m, &error)) {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 2;
    }
    m = canonical(m);
    std::printf("Warpmate benchmark: %s (%llu positions)\n", material_name(m).c_str(),
                static_cast<unsigned long long>(table_size(m)));

    std::vector<Row> rows;
    std::vector<Value> reference;  // the first table solved; all others must match
    bool match = true;
    const auto compare = [&](const std::vector<Value>& values) {
        if (reference.empty()) reference = values;
        else if (values != reference) match = false;
    };

    if (cpu_trials > 0) {
        Row cpu{"CPU " + std::string(algorithm_name(cpu_algo)) + ", " + std::to_string(threads) +
                    (threads == 1 ? " thread" : " threads"),
                {}};
        for (int t = 0; t < cpu_trials; ++t) {
            Solver solver(threads, cpu_algo);
            const TableResult& r = solve_timed(solver, m);
            cpu.ms.push_back(r.seconds * 1000.0);
            std::printf("  %s, trial %d: %.1f ms (%d rounds)\n", cpu.name.c_str(), t + 1,
                        r.seconds * 1000.0, r.rounds);
            compare(r.values);
        }
        rows.push_back(cpu);
    }

    bool gpu_ran = false;
    double full_total = 0, full_kernels = 0, frontier_total = 0, frontier_kernels = 0;
#ifdef WARPMATE_HAS_CUDA
    try {
        { GpuSolver warm; solve_timed(warm, m); }  // warm-up, not timed
        gpu_ran = true;
        TableResult last[2];
        for (int a = 0; a < 2; ++a) {
            const Algorithm algo = a == 0 ? Algorithm::kFullScan : Algorithm::kFrontier;
            Row total{std::string("GPU ") + algorithm_name(algo) + " (total)", {}};
            Row kernels{std::string("GPU ") + algorithm_name(algo) + " (kernels only)", {}};
            for (int t = 0; t < gpu_trials; ++t) {
                GpuSolver gpu(algo);
                const TableResult& r = solve_timed(gpu, m);
                total.ms.push_back(r.seconds * 1000.0);
                kernels.ms.push_back(gpu.kernel_ms(material_name(m)));
                compare(r.values);
                last[a] = r;
            }
            (a == 0 ? full_total : frontier_total) = median(total.ms);
            (a == 0 ? full_kernels : frontier_kernels) = median(kernels.ms);
            rows.push_back(total);
            rows.push_back(kernels);
        }
        std::printf("GPU: %s\n", GpuSolver::device_name().c_str());

        std::printf("\nPositions checked (the per-position step was run) over all rounds:\n");
        std::printf("  full scan: %llu\n  frontier:  %llu\n",
                    static_cast<unsigned long long>(total_checked(last[0])),
                    static_cast<unsigned long long>(total_checked(last[1])));
        if (show_rounds) {
            std::printf("\n| Round | Checked (full scan) | Checked (frontier) | Solved |\n");
            std::printf("|---|---|---|---|\n");
            for (size_t r = 0; r < last[0].round_stats.size(); ++r) {
                const RoundStats& f = last[0].round_stats[r];
                const RoundStats& g = r < last[1].round_stats.size() ? last[1].round_stats[r]
                                                                     : RoundStats{};
                std::printf("| %zu | %llu | %llu | %llu |\n", r + 1,
                            static_cast<unsigned long long>(f.checked),
                            static_cast<unsigned long long>(g.checked),
                            static_cast<unsigned long long>(f.solved));
            }
        }
    } catch (const std::exception& e) {
        std::printf("GPU: unavailable (%s)\n", e.what());
    }
#else
    (void)gpu_trials;
    (void)show_rounds;
    std::printf("GPU: not built (no CUDA compiler found)\n");
#endif

    if (!rows.empty()) {
        const double base = median(rows[0].ms);
        std::printf("\n| Solver | Median (ms) | Best (ms) | Speedup vs %s |\n|---|---|---|---|\n",
                    rows[0].name.c_str());
        for (const Row& row : rows) {
            const double med = median(row.ms);
            std::printf("| %s | %.1f | %.1f | %.1fx |\n", row.name.c_str(), med, best(row.ms),
                        base / med);
        }
    }
    if (gpu_ran) {
        std::printf("\nGPU frontier vs GPU full scan: %.1fx faster (total), %.1fx (kernels only)\n",
                    full_total / frontier_total, full_kernels / frontier_kernels);
        std::printf("GPU (total) includes GPU memory setup and copying the table back.\n");
    }
    if (rows.size() > 1) {
        std::printf("\nAll tables identical: %s\n", match ? "yes" : "NO");
    }
    return match ? 0 : 1;
}
