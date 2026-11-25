// warpmate-probe: looks up a position in the solved tables.
//
//   warpmate-probe "8/8/8/8/8/2k5/1R6/K7 w - - 0 1"
//   warpmate-probe -d tables "<fen>"
//
// Works for any endgame that warpmate-gen has solved. Loads the table for
// the position's material plus every table it can turn into (for captures
// and promotions), then prints the verdict, every legal move's outcome, and
// the line of perfect play.
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "egtb/notation.h"
#include "egtb/solver.h"
#include "io/tablebase.h"

using namespace wm;
using namespace wm::egtb;

namespace {

// Loads `m` and, recursively, everything it depends on.
bool load_family(const std::string& dir, const Material& m, std::map<std::string, Tablebase>* out,
                 std::string* error) {
    const std::string name = material_name(m);
    if (out->count(name)) return true;
    Tablebase tb;
    if (!load_tablebase(dir + "/" + name + ".wmtb", &tb, error)) {
        *error += " (solve it with: warpmate-gen " + name + ")";
        return false;
    }
    if (tb.name != name || tb.values.size() != table_size(m)) {
        *error = dir + "/" + name + ".wmtb doesn't match " + name;
        return false;
    }
    (*out)[name] = std::move(tb);
    for (const Material& d : direct_dependencies(m)) {
        if (!load_family(dir, d, out, error)) return false;
    }
    return true;
}

// Ranks a child value from the mover's point of view (higher is better).
int score_for_mover(Value child) {
    if (is_loss(child)) return 1000 - child;  // opponent lost: sooner is better
    if (is_win(child)) return -1000 + child;  // opponent wins: later is better
    return 0;
}

struct Choice {
    Move move;
    Value child;
};

std::vector<Choice> all_moves(const std::vector<SubTable>& subs, const Board& b) {
    Material m;
    Position p;
    to_position(b, &m, &p);
    std::vector<Choice> out;
    struct Collect {
        std::vector<Choice>* out;
        const std::vector<SubTable>* subs;
        const Material* m;
        const Position* p;
        bool operator()(const Move& mv) {
            const Board after = apply_move(*m, *p, mv);
            out->push_back({mv, lookup(subs->data(), static_cast<int>(subs->size()), after)});
            return true;
        }
    } collect{&out, &subs, &m, &p};
    for_each_move(m, p, collect);
    return out;
}

std::string outcome_text(const Move& mv, Value c) {
    (void)mv;
    if (c == kDraw) return "draw";
    if (c == 0) return "checkmate";
    const int n = moves_to_mate(c);
    return is_loss(c) ? "wins, mate in " + std::to_string(n) : "loses, mated in " + std::to_string(n);
}

}  // namespace

int main(int argc, char** argv) {
    std::string dir = "tables";
    std::string fen;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if ((arg == "-d" || arg == "-t") && i + 1 < argc) {
            dir = argv[++i];
        } else if (fen.empty()) {
            fen = arg;
        } else {
            fen.clear();
            break;
        }
    }
    if (fen.empty()) {
        std::fprintf(stderr, "usage: %s [-d DIR] \"<FEN>\"\n", argv[0]);
        return 2;
    }

    std::string error;
    auto board = parse_fen(fen, &error);
    if (!board) {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 1;
    }
    Material actual;
    Position pos;
    to_position(*board, &actual, &pos);

    std::map<std::string, Tablebase> loaded;
    if (actual.n > 2 && !load_family(dir, canonical(actual), &loaded, &error)) {
        std::fprintf(stderr, "error: %s\n", error.c_str());
        return 1;
    }
    std::vector<SubTable> subs;
    for (const auto& [name, tb] : loaded) {
        Material m;
        parse_material(name, &m);
        subs.push_back(SubTable{m, tb.values.data()});
    }

    std::printf("%s\n", to_fen(*board).c_str());
    if (!is_valid(actual, pos)) {
        std::printf("Result: Invalid position\n");
        return 0;
    }
    const Value v = lookup(subs.data(), static_cast<int>(subs.size()), *board);
    std::printf("Result: %s\n", describe(board->stm, v).c_str());

    std::vector<Choice> moves = all_moves(subs, *board);
    if (moves.empty()) return 0;
    size_t best = 0;
    for (size_t i = 1; i < moves.size(); ++i) {
        if (score_for_mover(moves[i].child) > score_for_mover(moves[best].child)) best = i;
    }
    std::printf("Best move: %s\n\nAll moves:\n", move_to_string(actual, pos, moves[best].move).c_str());
    for (const Choice& c : moves) {
        std::printf("  %-7s %s\n", move_to_string(actual, pos, c.move).c_str(),
                    outcome_text(c.move, c.child).c_str());
    }

    // Follow perfect play until mate (or a draw is reached).
    std::printf("\nPerfect play:");
    Board b = *board;
    for (int ply = 0; ply < 300; ++ply) {
        Material m;
        Position p;
        to_position(b, &m, &p);
        const Value here = lookup(subs.data(), static_cast<int>(subs.size()), b);
        std::vector<Choice> options = all_moves(subs, b);
        if (options.empty()) break;
        size_t pick = 0;
        for (size_t i = 1; i < options.size(); ++i) {
            if (score_for_mover(options[i].child) > score_for_mover(options[pick].child)) pick = i;
        }
        // Move numbers count full moves; if Black starts, White's first
        // reply is move 2.
        const int move_no = (ply + (board->stm == kBlack ? 1 : 0)) / 2 + 1;
        if (b.stm == kWhite) std::printf(" %d.", move_no);
        else if (ply == 0) std::printf(" %d...", move_no);
        std::printf(" %s", move_to_string(m, p, options[pick].move).c_str());
        if (here == kDraw) break;  // draws don't end in mate: show one move
        b = apply_move(m, p, options[pick].move);
        if (b.n == 2) break;
    }
    std::printf("\n");
    return 0;
}
