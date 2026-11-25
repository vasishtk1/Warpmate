// Tests for the KRK solver: known positions, the known longest mate, and a
// full consistency check of every entry in the table.
#include <string>

#include "check.h"
#include "core/notation.h"
#include "io/tablebase.h"
#include "solver/krk_solver.h"

using namespace wm;
using namespace wm::krk;

namespace {

Value lookup(const std::vector<Value>& table, const std::string& fen) {
    auto p = parse_fen(fen);
    CHECK(p.has_value());
    return p ? table[encode(*p)] : kInvalid;
}

void test_known_positions(const std::vector<Value>& t) {
    // Already checkmated.
    CHECK(lookup(t, "R6k/8/6K1/8/8/8/8/8 b - - 0 1") == 0);
    // Mate in 1 with Ra8#.
    CHECK(lookup(t, "7k/8/6K1/8/8/8/8/R7 w - - 0 1") == 1);
    // Stalemate is a draw.
    CHECK(lookup(t, "8/8/8/8/8/1K6/1R6/k7 b - - 0 1") == kDraw);
    // Black's only move captures the undefended rook: a draw.
    CHECK(lookup(t, "7K/8/8/8/8/8/1R6/k7 b - - 0 1") == kDraw);
    // Kings touching: invalid.
    CHECK(lookup(t, "8/8/8/8/8/8/8/RKk5 w - - 0 1") == kInvalid);
}

void test_best_move(const std::vector<Value>& t) {
    Position p = *parse_fen("7k/8/6K1/8/8/8/8/R7 w - - 0 1");
    Move m;
    CHECK(best_move(t.data(), p, &m));
    CHECK(move_to_string(p, m) == "Ra8#");
}

void test_longest_mate(const std::vector<Value>& t) {
    // Known result for this ending: the longest forced mate is 16 moves,
    // i.e. 31 plies with the winning side to move.
    const TableSummary s = summarize(t);
    CHECK(s.longest_win == 31);
    CHECK(moves_to_mate(s.longest_win) == 16);
    // Black has no way to win with a lone king.
    CHECK(s.side[kBlack].wins == 0);
    CHECK(s.side[kWhite].losses == 0);
}

// Every stored value must agree with the values of its children:
//   win in v  -> some child is lost in exactly v-1, none lost faster
//   loss in v -> every child is won, the slowest in exactly v-1
//   draw      -> no child is lost, and not every child is won
void test_consistency(const std::vector<Value>& t) {
    int bad = 0;
    for (uint32_t i = 0; i < kNumPositions && bad < 5; ++i) {
        const Value v = t[i];
        if (v == kInvalid) continue;
        const Position p = decode(i);
        Move moves[kMaxMoves];
        const int n = generate_moves(p, moves);

        int min_loss = 1000, max_win = -1, wins = 0, losses = 0;
        for (int k = 0; k < n; ++k) {
            const Value c = child_value(t.data(), p, moves[k]);
            CHECK(c != kInvalid && c != kUnknown);
            if (is_loss(c)) { ++losses; if (c < min_loss) min_loss = c; }
            if (is_win(c)) { ++wins; if (c > max_win) max_win = c; }
        }

        bool ok;
        if (n == 0) ok = v == 0 ? side_to_move_in_check(p) : (v == kDraw && !side_to_move_in_check(p));
        else if (is_win(v)) ok = losses > 0 && min_loss == v - 1;
        else if (is_loss(v)) ok = wins == n && max_win == v - 1;
        else ok = v == kDraw && losses == 0 && wins < n;

        if (!ok) {
            ++bad;
            std::printf("  inconsistent: %s value %d\n", to_fen(p).c_str(), v);
        }
    }
    CHECK(bad == 0);
}

// The multithreaded solver must produce exactly the same table and rounds.
void test_parallel_matches(const SolveResult& single) {
    for (int threads : {1, 3, 8}) {
        const SolveResult par = solve_parallel(threads);
        CHECK(par.table == single.table);
        CHECK(par.rounds.size() == single.rounds.size());
        CHECK(par.checkmates == single.checkmates);
        CHECK(par.stalemates == single.stalemates);
    }
}

void test_save_load(const std::vector<Value>& t) {
    const std::string path = "warpmate_test_table.wmtb";
    std::string error;
    CHECK(save_tablebase(path, Tablebase{"KRvK", t}, &error));
    Tablebase loaded;
    CHECK(load_tablebase(path, &loaded, &error));
    CHECK(loaded.name == "KRvK");
    CHECK(loaded.values == t);
    std::remove(path.c_str());
}

}  // namespace

int main() {
    const SolveResult result = solve();
    test_known_positions(result.table);
    test_best_move(result.table);
    test_longest_mate(result.table);
    test_consistency(result.table);
    test_parallel_matches(result);
    test_save_load(result.table);
    return wm::test::finish("solver");
}
