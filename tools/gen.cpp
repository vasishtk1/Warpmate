// warpmate-gen: solves endgames and saves them.
//
//   warpmate-gen KQvKR KRvKP        solve these (and anything they depend on)
//   warpmate-gen --all3             solve every 3-piece endgame
//   warpmate-gen --all4             solve every 4-piece endgame except KPvKP
//   options: -d DIR (default tables), --threads N (CPU, default: all cores),
//            --gpu (solve on the GPU; builds with CUDA only)
//
// Each table is written to DIR/<material>.wmtb, e.g. tables/KQvKR.wmtb.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <string>
#include <vector>

#include "egtb/notation.h"
#include "egtb/solver.h"
#include "format.h"
#include "io/tablebase.h"
#ifdef WARPMATE_HAS_CUDA
#include "gpu/egtb_gpu.h"
#endif

using namespace wm;
using namespace wm::egtb;

namespace {

// Solves every requested material with `solver` (CPU or GPU: both have the
// same interface), printing and saving each table as it finishes.
template <typename S>
bool run(S& solver, const std::vector<Material>& requested, const std::string& dir) {
    bool ok = true;
    solver.on_table = [&](const TableResult& t) {
        const std::string name = material_name(t.m);
        const TableSummary s = summarize(t.m, t.values);
        std::printf("%-6s %11s positions  %3d rounds  %8.2f s   White to move: %s W / %s D / %s L\n",
                    name.c_str(), with_commas(t.values.size()).c_str(), t.rounds, t.seconds,
                    with_commas(s.wins[0]).c_str(), with_commas(s.draws[0]).c_str(),
                    with_commas(s.losses[0]).c_str());
        if (s.longest_win > 0) {
            const Board b = to_board(t.m, decode(t.m, s.longest_win_index));
            std::printf("       longest win: mate in %d, e.g. %s\n", moves_to_mate(s.longest_win),
                        to_fen(b).c_str());
        }
        std::string error;
        const std::string path = dir + "/" + name + ".wmtb";
        if (!save_tablebase(path, Tablebase{name, t.values}, &error)) {
            std::fprintf(stderr, "error: %s\n", error.c_str());
            ok = false;
        }
    };
    for (const Material& m : requested) solver.solve(m);
    std::printf("Saved %zu table(s) to %s/\n", solver.tables().size(), dir.c_str());
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    std::string dir = "tables";
    int threads = 0;
    bool use_gpu = false;
    std::vector<Material> requested;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-d" && i + 1 < argc) {
            dir = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            threads = std::atoi(argv[++i]);
        } else if (arg == "--gpu") {
            use_gpu = true;
        } else if (arg == "--all3") {
            for (const Material& m : all_three_piece()) requested.push_back(m);
        } else if (arg == "--all4") {
            for (const Material& m : all_four_piece()) requested.push_back(m);
        } else {
            Material m;
            std::string error;
            if (!parse_material(arg, &m, &error)) {
                std::fprintf(stderr, "error: %s: %s\n", arg.c_str(), error.c_str());
                return 2;
            }
            if (m.n > 4) {
                std::fprintf(stderr, "error: %s: up to 4 pieces for now (5 needs ~2 GB per table)\n",
                             arg.c_str());
                return 2;
            }
            requested.push_back(m);
        }
    }
    if (requested.empty()) {
        std::fprintf(stderr, "usage: %s [-d DIR] [--threads N | --gpu] (--all3 | --all4 | MATERIAL...)\n",
                     argv[0]);
        return 2;
    }
#ifndef WARPMATE_HAS_CUDA
    if (use_gpu) {
        std::fprintf(stderr, "error: this build has no GPU support (no CUDA compiler found)\n");
        return 2;
    }
#endif

    std::filesystem::create_directories(dir);
    try {
        if (use_gpu) {
#ifdef WARPMATE_HAS_CUDA
            std::printf("Solving on the GPU (%s)\n", GpuSolver::device_name().c_str());
            GpuSolver solver;
            return run(solver, requested, dir) ? 0 : 1;
#endif
        }
        Solver solver(threads);
        return run(solver, requested, dir) ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
