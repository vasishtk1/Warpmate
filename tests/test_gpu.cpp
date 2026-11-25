// The GPU solver must produce exactly the same table as the CPU solver.
// Built only when CUDA is available.
#include <cstdio>
#include <exception>

#include "check.h"
#include "core/notation.h"
#include "gpu/krk_gpu.h"
#include "solver/krk_solver.h"

using namespace wm;
using namespace wm::krk;

int main() {
    try {
        std::printf("GPU: %s\n", gpu_name().c_str());
        const SolveResult cpu = solve();
        const SolveResult gpu = solve_gpu();

        CHECK(gpu.table.size() == cpu.table.size());
        CHECK(gpu.rounds.size() == cpu.rounds.size());
        CHECK(gpu.checkmates == cpu.checkmates);
        CHECK(gpu.stalemates == cpu.stalemates);
        for (size_t r = 0; r < cpu.rounds.size() && r < gpu.rounds.size(); ++r) {
            CHECK(gpu.rounds[r].newly_solved == cpu.rounds[r].newly_solved);
        }

        // Byte-for-byte comparison, reporting the first few differences.
        uint32_t diffs = 0;
        for (uint32_t i = 0; i < cpu.table.size() && i < gpu.table.size(); ++i) {
            if (cpu.table[i] != gpu.table[i]) {
                if (diffs < 5) {
                    std::printf("  mismatch at index %u (%s): cpu %d, gpu %d\n", i,
                                to_fen(decode(i)).c_str(), cpu.table[i], gpu.table[i]);
                }
                ++diffs;
            }
        }
        std::printf("Positions differing between CPU and GPU: %u\n", diffs);
        CHECK(diffs == 0);
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
    return wm::test::finish("gpu");
}
