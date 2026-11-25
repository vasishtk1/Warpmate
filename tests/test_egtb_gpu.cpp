// The general GPU solver must produce exactly the same tables as the CPU
// solver. Covers all five 3-piece endgames (fast enough for a unit test);
// 4-piece endgames are compared by warpmate-bench --egtb.
// Built only when CUDA is available.
#include <cstdio>
#include <exception>

#include "check.h"
#include "egtb/solver.h"
#include "gpu/egtb_gpu.h"

using namespace wm;
using namespace wm::egtb;

int main() {
    try {
        std::printf("GPU: %s\n", GpuSolver::device_name().c_str());
        Solver cpu;
        GpuSolver gpu;
        for (const Material& m : all_three_piece()) {
            cpu.solve(m);
            gpu.solve(m);
        }
        for (const auto& [name, t] : cpu.tables()) {
            const auto& g = gpu.tables().at(name);
            uint64_t diffs = 0;
            for (uint64_t i = 0; i < t.values.size(); ++i) diffs += t.values[i] != g.values[i];
            std::printf("  %-5s CPU vs GPU differences: %llu\n", name.c_str(),
                        static_cast<unsigned long long>(diffs));
            CHECK(diffs == 0);
            CHECK(t.rounds == g.rounds);
        }
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
    return wm::test::finish("egtb-gpu");
}
