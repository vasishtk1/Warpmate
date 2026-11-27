// Tests for the solver: piece attacks, indexing, move generation, un-moves,
// notation, material handling, colour flipping, known results for every
// 3-piece endgame, the frontier and multithreaded solvers, saving tables,
// and a full consistency check of every table entry.
#include <cstdio>
#include <set>
#include <string>

#include "check.h"
#include "egtb/notation.h"
#include "egtb/solver.h"
#include "egtb/unmove.h"
#include "io/tablebase.h"

using namespace wm;
using namespace wm::egtb;

namespace {

int popcount(Bitboard b) { return __builtin_popcountll(b); }

Material mat(const char* name) {
    Material m;
    CHECK(parse_material(name, &m));
    return m;
}

// Builds a Material + Position from a FEN.
void from_fen(const char* fen, Material* m, Position* p) {
    auto b = parse_fen(fen);
    CHECK(b.has_value());
    if (b) to_position(*b, m, p);
}

void test_attacks() {
    CHECK(popcount(king_attacks(0)) == 3);   // a1 corner
    CHECK(popcount(king_attacks(7)) == 3);   // h1 corner, no wrap to a2
    CHECK(popcount(king_attacks(4)) == 5);   // e1 edge
    CHECK(popcount(king_attacks(28)) == 8);  // e4 centre
    CHECK(!(king_attacks(7) & bb(8)));       // h1 must not "see" a2
    CHECK(popcount(rook_attacks(0, 0)) == 14);      // empty board: always 14
    CHECK(popcount(rook_attacks(0, bb(24))) == 10); // a1, blocker on a4
    CHECK(rook_attacks(0, bb(24)) & bb(24));        // blocker square included
    CHECK(!(rook_attacks(0, bb(24)) & bb(32)));     // nothing beyond it
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

void test_encode_decode() {
    const Material m = mat("KRvK");
    for (uint64_t i = 0; i < table_size(m); ++i) {
        if (encode(m, decode(m, i)) != i) {
            CHECK(encode(m, decode(m, i)) == i);
            return;
        }
    }
}

void test_validity() {
    Material m;
    Position p;
    from_fen("7k/8/6K1/8/8/8/8/R7 w - - 0 1", &m, &p);
    CHECK(is_valid(m, p));
    from_fen("8/8/8/8/8/8/8/RKk5 w - - 0 1", &m, &p);   // kings touching
    CHECK(!is_valid(m, p));
    from_fen("R6k/8/6K1/8/8/8/8/8 w - - 0 1", &m, &p);  // Black in check, White to move
    CHECK(!is_valid(m, p));
    from_fen("R6k/8/6K1/8/8/8/8/8 b - - 0 1", &m, &p);  // same, Black to move: fine
    CHECK(is_valid(m, p));
    // Double checks no legal move can produce: two checkers on one line
    // through the king (queens b1 and f1 around Kd1; bishop b7 and pawn d5
    // around Kc6). Syzygy treats these as illegal.
    from_fen("8/8/8/8/8/8/8/KQ1k1Q2 b - - 0 1", &m, &p);
    CHECK(!is_valid(m, p));
    from_fen("8/1B6/2k5/3P4/8/8/8/2K5 b - - 0 1", &m, &p);
    CHECK(!is_valid(m, p));
    // A real double check (knight from e4 to d6 uncovers the rook on e1).
    from_fen("4k3/8/3N4/8/8/8/8/K3R3 b - - 0 1", &m, &p);
    CHECK(is_valid(m, p));
    // Two white pieces on one line through the king, but the queen blocks
    // the rook, so only the queen gives check: fine.
    from_fen("8/8/8/8/8/8/8/KRQk4 b - - 0 1", &m, &p);
    CHECK(is_valid(m, p));
}

void test_move_generation() {
    Material m;
    Position p;
    // Checkmate: no moves, in check. Stalemate: no moves, not in check.
    from_fen("R6k/8/6K1/8/8/8/8/8 b - - 0 1", &m, &p);
    CHECK(count_moves(m, p) == 0 && in_check(m, p, p.stm));
    from_fen("8/8/8/8/8/1K6/1R6/k7 b - - 0 1", &m, &p);
    CHECK(count_moves(m, p) == 0 && !in_check(m, p, p.stm));
    // Undefended rook next to the black king: capturing it is the only move.
    from_fen("7K/8/8/8/8/8/1R6/k7 b - - 0 1", &m, &p);
    CHECK(count_moves(m, p) == 1);
    // White Ra1, Ke1: rook a2..a8 (7) + b1..d1 (3) = 10, king 5. Total 15.
    from_fen("7k/8/8/8/8/8/8/R3K3 w - - 0 1", &m, &p);
    CHECK(count_moves(m, p) == 15);

    // White pawn on e2 with an empty path: e3 and e4.
    auto b = parse_fen("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1");
    CHECK(b.has_value());
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

void test_notation() {
    const char* fen = "7k/8/6K1/8/8/8/8/R7 w - - 0 1";
    Material m;
    Position p;
    from_fen(fen, &m, &p);
    CHECK(to_fen(to_board(m, p)) == fen);
    // Piece 1 is the rook (canonical order is K, R, k): a1 to a8.
    const Move ra8{1, 0, 56, kNone, kNone};
    CHECK(move_to_string(m, p, ra8) == "Ra8#");
    std::string err;
    CHECK(!parse_fen("8/8/8/8/8/8/8/8 w - - 0 1", &err).has_value());  // no kings
}

// KRvK counts, checked against the original KRvK-only solver (byte-identical
// tables) and Syzygy before that solver was removed.
void test_krvk_counts(const Solver& s) {
    const TableSummary t = summarize(mat("KRvK"), s.tables().at("KRvK").values);
    CHECK(t.invalid == 125176);
    CHECK(t.wins[kWhite] == 175168 && t.draws[kWhite] == 0 && t.losses[kWhite] == 0);
    CHECK(t.wins[kBlack] == 0 && t.draws[kBlack] == 22244 && t.losses[kBlack] == 201700);
    CHECK(s.tables().at("KRvK").rounds == 33);
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
    CHECK(value("R6k/8/6K1/8/8/8/8/8 b - - 0 1") == 0);       // checkmated
    CHECK(value("8/8/8/8/8/1K6/1R6/k7 b - - 0 1") == kDraw);  // stalemate
    CHECK(value("7K/8/8/8/8/8/1R6/k7 b - - 0 1") == kDraw);   // rook falls
    CHECK(value("8/8/8/8/8/8/8/RKk5 w - - 0 1") == kInvalid); // kings touching
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

// Un-moves must be the exact reverse of moves that stay in the table:
//   forward:  every such move P -> Q appears among Q's un-moves
//   backward: every un-move Q <- P is a real move of P
// Checks every `stride`-th position (stride 1 = all of them).
struct HasParent {
    const Material* m;
    uint64_t want;
    bool found = false;
    bool operator()(const Position& p) {
        if (encode(*m, p) == want) found = true;
        return !found;
    }
};

struct HasChild {
    const Material* m;
    const Position* p;
    uint64_t want;
    bool found = false;
    bool operator()(const Move& mv) {
        if (mv.captured != kNone || mv.promo != kNone) return true;
        Position q = *p;
        q.sq[mv.piece] = mv.to;
        q.stm ^= 1;
        if (encode(*m, q) == want) found = true;
        return !found;
    }
};

void test_unmoves(const char* name, uint64_t stride) {
    const Material m = mat(name);
    uint64_t moves = 0, unmoves = 0, bad = 0;
    for (uint64_t i = 0; i < table_size(m) && bad < 5; i += stride) {
        const Position p = decode(m, i);
        if (!is_valid(m, p)) continue;
        // Forward: each quiet move from p must be undone back to p.
        struct Forward {
            const Material* m;
            const Position* p;
            uint64_t* moves;
            uint64_t* bad;
            bool operator()(const Move& mv) {
                if (mv.captured != kNone || mv.promo != kNone) return true;
                Position q = *p;
                q.sq[mv.piece] = mv.to;
                q.stm ^= 1;
                HasParent h{m, encode(*m, *p)};
                for_each_unmove(*m, q, h);
                ++*moves;
                if (!h.found) ++*bad;
                return true;
            }
        } fwd{&m, &p, &moves, &bad};
        for_each_move(m, p, fwd);
        // Backward: each parent of p must have a move to p.
        struct Backward {
            const Material* m;
            const Position* q;
            uint64_t* unmoves;
            uint64_t* bad;
            bool operator()(const Position& parent) {
                HasChild h{m, &parent, encode(*m, *q)};
                for_each_move(*m, parent, h);
                ++*unmoves;
                if (!h.found) ++*bad;
                return true;
            }
        } back{&m, &p, &unmoves, &bad};
        for_each_unmove(m, p, back);
    }
    if (bad) std::printf("  %s: %llu un-move mismatches\n", name, static_cast<unsigned long long>(bad));
    CHECK(bad == 0);
    CHECK(moves > 0 && unmoves > 0);
    // Over a whole table, every quiet move is counted once in each direction.
    if (stride == 1) CHECK(moves == unmoves);
}

// The frontier solver must give exactly the same tables, rounds, and number
// of positions solved per round as the original full scan, while checking
// fewer positions.
void test_frontier_matches(const Solver& frontier) {
    Solver full(0, Algorithm::kFullScan);
    for (const Material& m : all_three_piece()) {
        const std::string name = material_name(m);
        const TableResult& a = full.solve(m);
        const TableResult& b = frontier.tables().at(name);
        CHECK(a.values == b.values);
        CHECK(a.rounds == b.rounds);
        CHECK(a.round_stats.size() == b.round_stats.size());
        uint64_t checked_full = 0, checked_frontier = 0;
        for (size_t r = 0; r < a.round_stats.size() && r < b.round_stats.size(); ++r) {
            CHECK(a.round_stats[r].solved == b.round_stats[r].solved);
            checked_full += a.round_stats[r].checked;
            checked_frontier += b.round_stats[r].checked;
        }
        CHECK(checked_frontier <= checked_full);
        std::printf("  %-5s positions checked: full scan %llu, frontier %llu\n", name.c_str(),
                    static_cast<unsigned long long>(checked_full),
                    static_cast<unsigned long long>(checked_frontier));
    }
}

// The list of 4-piece endgames: 29, all different, all in canonical form,
// and every one's dependencies are 3- or 4-piece endgames in the list.
void test_all_four_piece() {
    const auto all = all_four_piece();
    CHECK(all.size() == 29);
    std::set<std::string> names;
    for (const Material& m : all) {
        names.insert(material_name(m));
        CHECK(material_name(canonical(m)) == material_name(m));
    }
    CHECK(names.size() == 29);
    CHECK(names.count("KQQvK") && names.count("KBNvK") && names.count("KQvKQ") &&
          names.count("KNvKP") && !names.count("KPvKP"));
    for (const Material& m : all) {
        for (const Material& d : direct_dependencies(m)) {
            CHECK(d.n == 3 || names.count(material_name(d)));
        }
    }
}

// Any number of threads must give exactly the same tables. KPvK also
// exercises lookups into the four tables a promotion can reach.
void test_threads_match(const Solver& reference) {
    for (int threads : {1, 3, 8}) {
        Solver s(threads);
        for (const char* name : {"KRvK", "KPvK"}) {
            const TableResult& t = s.solve(mat(name));
            const TableResult& r = reference.tables().at(name);
            CHECK(t.values == r.values);
            CHECK(t.rounds == r.rounds);
        }
    }
}

void test_save_load(const Solver& s) {
    const auto& values = s.tables().at("KRvK").values;
    const std::string path = "warpmate_test_table.wmtb";
    std::string error;
    CHECK(save_tablebase(path, Tablebase{"KRvK", values}, &error));
    Tablebase loaded;
    CHECK(load_tablebase(path, &loaded, &error));
    CHECK(loaded.name == "KRvK");
    CHECK(loaded.values == values);
    std::remove(path.c_str());
}

}  // namespace

int main() {
    test_attacks();
    test_encode_decode();
    test_validity();
    test_material();
    test_move_generation();
    test_notation();
    for (const char* name : {"KRvK", "KQvK", "KBvK", "KNvK", "KPvK"}) test_unmoves(name, 1);
    // 4-piece: every 97th position (about 346,000 per table), including
    // repeated pieces (two queens, two bishops, two pawns on one side).
    for (const char* name : {"KQvKR", "KRvKP", "KRvKN", "KPvKP", "KBvKN", "KQQvK", "KBBvK",
                             "KPPvK"}) {
        test_unmoves(name, 97);
    }
    test_all_four_piece();

    Solver solver;
    for (const Material& m : all_three_piece()) solver.solve(m);

    test_krvk_counts(solver);
    test_known_results(solver);
    test_lookups(solver);
    test_frontier_matches(solver);
    test_threads_match(solver);
    test_save_load(solver);
    for (const char* name : {"KQvK", "KRvK", "KBvK", "KNvK", "KPvK"}) {
        test_consistency(solver, name);
    }
    return wm::test::finish("egtb");
}
