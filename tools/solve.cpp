// warpmate-solve: solves King + Rook vs King and saves the table.
//
//   warpmate-solve [-o tables/krvk.wmtb] [--threads N | --gpu]
//
// Default is one CPU thread. --threads uses several CPU threads; --gpu uses
// CUDA (only in builds with GPU support). All produce the same table.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <string>

#include "core/notation.h"
#include "format.h"
#include "io/tablebase.h"
#include "solver/krk_solver.h"
#ifdef WARPMATE_HAS_CUDA
#include "gpu/krk_gpu.h"
#endif

using namespace wm;
using namespace wm::krk;

int main(int argc, char** argv) {
    std::string out_path = "tables/krvk.wmtb";
    int threads = 1;
    bool use_gpu = false;
    for (int i = 1; i < argc; ++i) {
        if ((!std::strcmp(argv[i], "-o") || !std::strcmp(argv[i], "--out")) && i + 1 < argc) {
            out_path = argv[++i];
        } else if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) {
            threads = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--gpu")) {
            use_gpu = true;
        } else {
            std::fprintf(stderr, "usage: %s [-o OUTPUT] [--threads N | --gpu]\n", argv[0]);
            return 2;
        }
    }
#ifndef WARPMATE_HAS_CUDA
    if (use_gpu) {
        std::fprintf(stderr, "error: this build has no GPU support (no CUDA compiler found)\n");
        return 2;
    }
#endif

    const auto print_round = [](const RoundStats& r) {
        std::printf("  round %2d: %9s positions solved\n", r.ply,
                    with_commas(r.newly_solved).c_str());
    };

    SolveResult result;
    try {
        if (use_gpu) {
#ifdef WARPMATE_HAS_CUDA
            std::printf("Solving KRvK (%s positions) on the GPU (%s)...\n",
                        with_commas(kNumPositions).c_str(), gpu_name().c_str());
            result = solve_gpu(print_round);
#endif
        } else if (threads == 1) {
            std::printf("Solving KRvK (%s positions) on 1 CPU thread...\n",
                        with_commas(kNumPositions).c_str());
            result = solve(print_round);
        } else {
            std::printf("Solving KRvK (%s positions) on %s CPU threads...\n",
                        with_commas(kNumPositions).c_str(),
                        threads > 0 ? std::to_string(threads).c_str() : "all");
            result = solve_parallel(threads, print_round);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

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
