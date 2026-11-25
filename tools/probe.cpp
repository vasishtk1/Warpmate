// warpmate-probe: looks up a position in a solved table.
//
//   warpmate-probe "7k/8/6K1/8/8/8/8/R7 w - - 0 1"
//   warpmate-probe -t tables/krvk.wmtb "<fen>"
//
// Prints the verdict, every legal move with its result, and the line of
// perfect play to the end.
#include <cstdio>
#include <cstring>
#include <string>

#include "core/notation.h"
#include "io/tablebase.h"
#include "solver/krk_solver.h"

using namespace wm;
using namespace wm::krk;

namespace {

// Text for the result after a move, from the mover's point of view.
std::string outcome_after(const Position& p, const Move& m, const Value* table) {
    const Value c = child_value(table, p, m);
    if (m.capture || c == kDraw) return "draw";
    const int n = moves_to_mate(c);
    if (c == 0) return "checkmate";
    // The child is the opponent's position: their loss is our win.
    return is_loss(c) ? "wins, mate in " + std::to_string(n)
                      : "loses, mated in " + std::to_string(n);
}

void print_line(Position p, const Value* table) {
    std::printf("Perfect play:");
    for (int ply = 0; ply < 64; ++ply) {
        const Value v = table[encode(p)];
        Move m;
        if (!best_move(table, p, &m)) break;
        if (p.stm == kWhite) std::printf(" %d.", ply / 2 + 1);
        else if (ply == 0) std::printf(" 1...");
        std::printf(" %s", move_to_string(p, m).c_str());
        if (m.capture || v == kDraw) break;  // draws don't end in mate
        p = make_move(p, m);
    }
    std::printf("\n");
}

}  // namespace

int main(int argc, char** argv) {
    std::string table_path = "tables/krvk.wmtb";
    std::string fen;
    for (int i = 1; i < argc; ++i) {
        if ((!std::strcmp(argv[i], "-t") || !std::strcmp(argv[i], "--table")) && i + 1 < argc) {
            table_path = argv[++i];
        } else if (fen.empty()) {
            fen = argv[i];
        } else {
            fen.clear();
            break;
        }
    }
    if (fen.empty()) {
        std::fprintf(stderr, "usage: %s [-t TABLE] \"<FEN>\"\n", argv[0]);
        return 2;
    }

    std::string error;
    Tablebase tb;
    if (!load_tablebase(table_path, &tb, &error)) {
        std::fprintf(stderr, "error: %s (run warpmate-solve first)\n", error.c_str());
        return 1;
    }
    if (tb.name != "KRvK" || tb.values.size() != kNumPositions) {
        std::fprintf(stderr, "error: %s is not a KRvK table\n", table_path.c_str());
        return 1;
    }
    auto pos = parse_fen(fen, &error);
    if (!pos) {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 1;
    }

    const Position p = *pos;
    const Value v = tb.values[encode(p)];
    std::printf("%s\n", to_fen(p).c_str());
    std::printf("Result: %s\n", describe(p, v).c_str());
    if (v == kInvalid) return 0;

    Move moves[kMaxMoves];
    const int n = generate_moves(p, moves);
    Move best;
    if (best_move(tb.values.data(), p, &best)) {
        std::printf("Best move: %s\n\n", move_to_string(p, best).c_str());
        std::printf("All moves:\n");
        for (int i = 0; i < n; ++i) {
            std::printf("  %-6s %s\n", move_to_string(p, moves[i]).c_str(),
                        outcome_after(p, moves[i], tb.values.data()).c_str());
        }
        std::printf("\n");
        print_line(p, tb.values.data());
    }
    return 0;
}
