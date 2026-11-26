// Both GPU algorithms (full scan and frontier) must produce exactly the same
// tables as the CPU solver. Covers all five 3-piece endgames (fast enough for
// a unit test); 4-piece endgames are compared by warpmate-bench.
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
        GpuSolver full(Algorithm::kFullScan);
        GpuSolver frontier(Algorithm::kFrontier);
        for (const Material& m : all_three_piece()) {
            cpu.solve(m);
            full.solve(m);
            frontier.solve(m);
        }
        for (const auto& [name, t] : cpu.tables()) {
            for (const GpuSolver* gpu : {&full, &frontier}) {
                const TableResult& g = gpu->tables().at(name);
                uint64_t diffs = 0;
                for (uint64_t i = 0; i < t.values.size(); ++i) diffs += t.values[i] != g.values[i];
                std::printf("  %-5s CPU vs GPU (%s) differences: %llu\n", name.c_str(),
                            gpu == &full ? "full scan" : "frontier",
                            static_cast<unsigned long long>(diffs));
                CHECK(diffs == 0);
                CHECK(t.rounds == g.rounds);
                // Same positions solved in every round.
                CHECK(t.round_stats.size() == g.round_stats.size());
                for (size_t r = 0; r < t.round_stats.size() && r < g.round_stats.size(); ++r) {
                    CHECK(t.round_stats[r].solved == g.round_stats[r].solved);
                }
            }
            // The GPU frontier checks exactly the positions the CPU frontier does.
            const TableResult& g = frontier.tables().at(name);
            for (size_t r = 0; r < t.round_stats.size() && r < g.round_stats.size(); ++r) {
                CHECK(t.round_stats[r].checked == g.round_stats[r].checked);
            }
        }
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
    return wm::test::finish("egtb-gpu");
}
