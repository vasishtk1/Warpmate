// warpmate-gen: solves endgames with the general CPU solver and saves them.
//
//   warpmate-gen KQvK KPvK          solve these (and anything they depend on)
//   warpmate-gen --all3             solve every 3-piece endgame
//   options: -d DIR (default tables), --threads N (default: all cores)
//
// Each table is written to DIR/<material>.wmtb, e.g. tables/KQvK.wmtb.
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

using namespace wm;
using namespace wm::egtb;

int main(int argc, char** argv) {
    std::string dir = "tables";
    int threads = 0;
    std::vector<Material> requested;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-d" && i + 1 < argc) {
            dir = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            threads = std::atoi(argv[++i]);
        } else if (arg == "--all3") {
            for (const Material& m : all_three_piece()) requested.push_back(m);
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
        std::fprintf(stderr, "usage: %s [-d DIR] [--threads N] (--all3 | MATERIAL...)\n", argv[0]);
        return 2;
    }

    std::filesystem::create_directories(dir);
    Solver solver(threads);
    bool ok = true;
    solver.on_table = [&](const TableResult& t) {
        const std::string name = material_name(t.m);
        const TableSummary s = summarize(t.m, t.values);
        std::printf("%-6s %11s positions  %3d rounds  %7.2f s   White to move: %s W / %s D / %s L",
                    name.c_str(), with_commas(t.values.size()).c_str(), t.rounds, t.seconds,
                    with_commas(s.wins[0]).c_str(), with_commas(s.draws[0]).c_str(),
                    with_commas(s.losses[0]).c_str());
        if (s.longest_win > 0) {
            std::printf("   longest win: mate in %d", moves_to_mate(s.longest_win));
        }
        std::printf("\n");
        std::string error;
        const std::string path = dir + "/" + name + ".wmtb";
        if (!save_tablebase(path, Tablebase{name, t.values}, &error)) {
            std::fprintf(stderr, "error: %s\n", error.c_str());
            ok = false;
        }
    };

    try {
        for (const Material& m : requested) solver.solve(m);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    std::printf("Saved %zu table(s) to %s/\n", solver.tables().size(), dir.c_str());
    return ok ? 0 : 1;
}
