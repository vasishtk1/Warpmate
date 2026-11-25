// warpmate-solve: solves King + Rook vs King on the CPU and saves the table.
//
//   warpmate-solve [-o tables/krvk.wmtb]
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>

#include "core/notation.h"
#include "format.h"
#include "io/tablebase.h"
#include "solver/krk_solver.h"

using namespace wm;
using namespace wm::krk;

int main(int argc, char** argv) {
    std::string out_path = "tables/krvk.wmtb";
    for (int i = 1; i < argc; ++i) {
        if ((!std::strcmp(argv[i], "-o") || !std::strcmp(argv[i], "--out")) && i + 1 < argc) {
            out_path = argv[++i];
        } else {
            std::fprintf(stderr, "usage: %s [-o OUTPUT]\n", argv[0]);
            return 2;
        }
    }

    std::printf("Solving KRvK (%s positions) on the CPU...\n",
                with_commas(kNumPositions).c_str());

    const SolveResult result = solve([](const RoundStats& r) {
        std::printf("  round %2d: %9s positions solved\n", r.ply,
                    with_commas(r.newly_solved).c_str());
    });

    const TableSummary s = summarize(result.table);
    std::printf("\nDone in %.3f s (%zu rounds)\n", result.seconds, result.rounds.size());
    std::printf("  invalid positions: %s\n", with_commas(s.invalid).c_str());
    std::printf("  checkmates: %s, stalemates: %s\n", with_commas(result.checkmates).c_str(),
                with_commas(result.stalemates).c_str());
    const char* names[2] = {"White to move", "Black to move"};
    for (int side = 0; side < 2; ++side) {
        std::printf("  %s: %s wins, %s draws, %s losses\n", names[side],
                    with_commas(s.side[side].wins).c_str(),
                    with_commas(s.side[side].draws).c_str(),
                    with_commas(s.side[side].losses).c_str());
    }
    const Position longest = decode(s.longest_win_index);
    std::printf("  longest forced win: mate in %d (e.g. %s)\n", moves_to_mate(s.longest_win),
                to_fen(longest).c_str());

    const auto parent = std::filesystem::path(out_path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    std::string error;
    if (!save_tablebase(out_path, Tablebase{"KRvK", result.table}, &error)) {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 1;
    }
    std::printf("\nSaved %s\n", out_path.c_str());
    return 0;
}
