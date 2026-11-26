// warpmate-bench: times the CPU and GPU solvers on one endgame and checks
// that they produce identical tables.
//
//   warpmate-bench                   KRvK with default settings
//   warpmate-bench KQvKR [--cpu-trials 1] [--gpu-trials 3] [--threads N]
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

}  // namespace

int main(int argc, char** argv) {
    std::string name = "KRvK";
    int cpu_trials = 1;
    int gpu_trials = 3;
    int threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    bool named = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) {
            threads = std::max(1, std::atoi(argv[++i]));
        } else if (!std::strcmp(argv[i], "--cpu-trials") && i + 1 < argc) {
            cpu_trials = std::max(0, std::atoi(argv[++i]));
        } else if (!std::strcmp(argv[i], "--gpu-trials") && i + 1 < argc) {
            gpu_trials = std::max(1, std::atoi(argv[++i]));
        } else if (argv[i][0] != '-' && !named) {
            name = argv[i];
            named = true;
        } else {
            std::fprintf(stderr,
                         "usage: %s [MATERIAL] [--cpu-trials N] [--gpu-trials N] [--threads N]\n",
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
    std::printf("CPU threads used: %d\n", threads);

    // CPU runs. The first table is the reference the GPU must match.
    std::vector<double> cpu_ms;
    std::vector<Value> reference;
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
        std::printf("| CPU, %d thread%s | %.1f | %.1f | 1.0x |\n", threads, threads == 1 ? "" : "s",
                    base, best(cpu_ms));
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
        std::printf("  GPU (total) includes GPU memory setup and copying the table back.\n");
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
