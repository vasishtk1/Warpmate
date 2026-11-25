// warpmate-bench: times every solver on this machine and checks they agree.
//
//   warpmate-bench [--trials 5] [--threads N]
//
// Each solver runs `trials` times; the median and best times are reported.
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

#include "solver/krk_solver.h"
#ifdef WARPMATE_HAS_CUDA
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

}  // namespace

int main(int argc, char** argv) {
    int trials = 5;
    int threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--trials") && i + 1 < argc) {
            trials = std::max(1, std::atoi(argv[++i]));
        } else if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) {
            threads = std::max(1, std::atoi(argv[++i]));
        } else {
            std::fprintf(stderr, "usage: %s [--trials N] [--threads N]\n", argv[0]);
            return 2;
        }
    }

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
