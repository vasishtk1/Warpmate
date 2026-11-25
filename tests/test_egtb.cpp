// Tests for the general solver: piece attacks, material handling, colour
// flipping, agreement with the KRvK solver, known results for every 3-piece
// endgame, and a full consistency check of every table entry.
#include <cstdio>
#include <string>

#include "check.h"
#include "egtb/notation.h"
#include "egtb/solver.h"
#include "solver/krk_solver.h"

using namespace wm;
using namespace wm::egtb;

namespace {

int popcount(Bitboard b) { return __builtin_popcountll(b); }

Material mat(const char* name) {
    Material m;
    CHECK(parse_material(name, &m));
    return m;
}

void test_attacks() {
    CHECK(popcount(knight_attacks(0)) == 2);    // a1
    CHECK(popcount(knight_attacks(27)) == 8);   // d4
    CHECK(popcount(knight_attacks(7)) == 2);    // h1, no wrap-around
    CHECK(popcount(bishop_attacks(27, 0)) == 13);  // d4 on an empty board
    CHECK(popcount(bishop_attacks(0, 0)) == 7);    // a1
    CHECK(popcount(queen_attacks(27, 0)) == 27);
    CHECK(pawn_attacks(12, true) == (bb(19) | bb(21)));   // e2 -> d3, f3
    CHECK(pawn_attacks(52, false) == (bb(43) | bb(45)));  // e7 -> d6, f6
    CHECK(pawn_attacks(8, true) == bb(17));                // a2 -> b3 only
}

void test_material() {
    CHECK(material_name(mat("KQvKR")) == "KQvKR");
    CHECK(material_name(canonical(mat("KvKQ"))) == "KQvK");
    CHECK(material_name(canonical(mat("KRvKQ"))) == "KQvKR");
    CHECK(material_name(canonical(mat("KPvKR"))) == "KRvKP");
    Material bad;
    CHECK(!parse_material("KQKR", &bad));
    CHECK(!parse_material("KQvR", &bad));
    CHECK(!parse_material("KXvK", &bad));

    // KPvK can become KQvK, KRvK, KBvK or KNvK by promotion.
    const auto deps = direct_dependencies(mat("KPvK"));
    CHECK(deps.size() == 4);
    // KRvKP: captures give KRvK and KPvK; promotions give KRvKQ (stored as
    // KQvKR), KRvKR, KRvKB, KRvKN; promoting with capture gives KQvK etc.
    const auto deps2 = direct_dependencies(mat("KRvKP"));
    bool has_kqvkr = false, has_kpvk = false;
    for (const Material& d : deps2) {
        has_kqvkr |= material_name(d) == "KQvKR";
        has_kpvk |= material_name(d) == "KPvK";
    }
    CHECK(has_kqvkr && has_kpvk);
}

void test_move_generation() {
    // White pawn on e2 with an empty path: e3 and e4.
    auto b = parse_fen("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1");
    CHECK(b.has_value());
    Material m;
    Position p;
    to_position(*b, &m, &p);
    // King on e1: d1, f1, d2, f2 (e2 is its own pawn) = 4, pawn: 2.
    CHECK(count_moves(m, p) == 6);

    // Promotion gives four moves (queen, rook, bishop, knight).
    b = parse_fen("k7/4P3/8/8/8/8/8/4K3 w - - 0 1");
    to_position(*b, &m, &p);
    CHECK(count_moves(m, p) == 5 + 4);

    // A pinned piece can't move off the pin line: White Ke1, Ne2, black Qe8.
    b = parse_fen("4q2k/8/8/8/8/8/4N3/4K3 w - - 0 1");
    to_position(*b, &m, &p);
    // Knight is pinned (0 moves). King: d1, f1, d2, f2 = 4.
    CHECK(count_moves(m, p) == 4);
}

void test_matches_krk_solver(const Solver& s) {
    // The general solver uses the same row layout as the KRvK-only solver,
    // so the two tables must be byte-for-byte identical.
    const auto& general = s.tables().at("KRvK").values;
    const auto specific = krk::solve().table;
    CHECK(general == specific);
}

void test_known_results(const Solver& s) {
    const auto summary = [&](const char* name) {
        return summarize(mat(name), s.tables().at(name).values);
    };
    // Known longest forced mates: KQvK 10 moves, KRvK 16 moves.
    CHECK(moves_to_mate(summary("KQvK").longest_win) == 10);
    CHECK(moves_to_mate(summary("KRvK").longest_win) == 16);
    // A lone bishop or knight can never force mate: everything is a draw.
    for (const char* name : {"KBvK", "KNvK"}) {
        const TableSummary t = summary(name);
        CHECK(t.wins[0] == 0 && t.wins[1] == 0 && t.losses[0] == 0 && t.losses[1] == 0);
    }
    // KPvK has both wins and draws for White, and Black never wins.
    const TableSummary kp = summary("KPvK");
    CHECK(kp.wins[0] > 0 && kp.draws[0] > 0);
    CHECK(kp.wins[1] == 0);
}

void test_lookups(const Solver& s) {
    std::vector<SubTable> subs;
    for (const auto& [name, t] : s.tables()) subs.push_back(SubTable{t.m, t.values.data()});
    const auto value = [&](const char* fen) {
        auto b = parse_fen(fen);
        CHECK(b.has_value());
        return b ? lookup(subs.data(), static_cast<int>(subs.size()), *b) : kInvalid;
    };
    // Mate in 1 with the rook, and the same position with colours swapped
    // and the board mirrored (Black has the rook): must give the same value.
    CHECK(value("7k/8/6K1/8/8/8/8/R7 w - - 0 1") == 1);
    CHECK(value("r7/8/8/8/8/6k1/8/7K b - - 0 1") == 1);
    // King + pawn: White Kd6, Pd5, Black Kd8, White to move wins
    // (the classic "king in front of the pawn" case).
    CHECK(is_win(value("3k4/8/3K4/3P4/8/8/8/8 w - - 0 1")));
    // Same with Black to move and the black king in front: a draw.
    CHECK(value("8/8/3k4/8/3P4/3K4/8/8 b - - 0 1") == kDraw);
}

// Every entry must agree with its children (including children in other
// tables reached by captures and promotions).
void test_consistency(const Solver& s, const char* name) {
    const TableResult& t = s.tables().at(name);
    const std::vector<SubTable> subs = s.sub_tables(t.m);
    const Context ctx{t.m, subs.data(), static_cast<int>(subs.size())};

    struct Gather {
        const Context* ctx;
        const Value* table;
        const Position* p;
        int n = 0, wins = 0, losses = 0, min_loss = 1000, max_win = -1, bad = 0;
        bool operator()(const Move& mv) {
            const Value c = child_value(*ctx, table, *p, mv);
            ++n;
            if (c == kInvalid || c == kUnknown) ++bad;
            if (is_loss(c)) { ++losses; if (c < min_loss) min_loss = c; }
            if (is_win(c)) { ++wins; if (c > max_win) max_win = c; }
            return true;
        }
    };

    int bad = 0;
    for (uint64_t i = 0; i < t.values.size() && bad < 5; ++i) {
        const Value v = t.values[i];
        if (v == kInvalid) continue;
        const Position p = decode(t.m, i);
        Gather g{&ctx, t.values.data(), &p};
        for_each_move(t.m, p, g);
        const bool check = in_check(t.m, p, p.stm);
        bool ok;
        if (g.bad) ok = false;
        else if (g.n == 0) ok = v == 0 ? check : (v == kDraw && !check);
        else if (is_win(v)) ok = g.losses > 0 && g.min_loss == v - 1;
        else if (is_loss(v)) ok = g.wins == g.n && g.max_win == v - 1;
        else ok = v == kDraw && g.losses == 0 && g.wins < g.n;
        if (!ok) {
            ++bad;
            std::printf("  %s inconsistent: %s value %d\n", name, to_fen(to_board(t.m, p)).c_str(), v);
        }
    }
    CHECK(bad == 0);
}

}  // namespace

int main() {
    test_attacks();
    test_material();
    test_move_generation();

    Solver solver;
    for (const Material& m : all_three_piece()) solver.solve(m);

    test_matches_krk_solver(solver);
    test_known_results(solver);
    test_lookups(solver);
    for (const char* name : {"KQvK", "KRvK", "KBvK", "KNvK", "KPvK"}) {
        test_consistency(solver, name);
    }
    return wm::test::finish("egtb");
}
