// Tests for board indexing, attacks, move generation and notation.
#include <string>

#include "check.h"
#include "core/krk.h"
#include "core/notation.h"

using namespace wm;
using namespace wm::krk;

namespace {

Position fen(const std::string& s) {
    auto p = parse_fen(s);
    CHECK(p.has_value());
    return p.value_or(Position{0, 0, 0, 0});
}

int popcount(Bitboard b) { return __builtin_popcountll(b); }

void test_encode_decode_round_trip() {
    for (uint32_t i = 0; i < kNumPositions; ++i) {
        if (encode(decode(i)) != i) {
            CHECK(encode(decode(i)) == i);
            return;
        }
    }
}

void test_king_attacks() {
    CHECK(popcount(king_attacks(0)) == 3);   // a1 corner
    CHECK(popcount(king_attacks(7)) == 3);   // h1 corner, no wrap to a2
    CHECK(popcount(king_attacks(4)) == 5);   // e1 edge
    CHECK(popcount(king_attacks(28)) == 8);  // e4 centre
    CHECK(!(king_attacks(7) & bb(8)));       // h1 must not "see" a2
}

void test_rook_attacks() {
    CHECK(popcount(rook_attacks(0, 0)) == 14);  // empty board: always 14
    // Rook a1, blocker on a4: a2, a3, a4 upward plus 7 along rank 1.
    CHECK(popcount(rook_attacks(0, bb(24))) == 10);
    CHECK(rook_attacks(0, bb(24)) & bb(24));    // blocker square included
    CHECK(!(rook_attacks(0, bb(24)) & bb(32))); // nothing beyond it
}

void test_validity() {
    CHECK(is_valid(fen("7k/8/6K1/8/8/8/8/R7 w - - 0 1")));
    CHECK(!is_valid(fen("8/8/8/8/8/8/8/RKk5 w - - 0 1")));   // kings touching
    CHECK(!is_valid(fen("R6k/8/6K1/8/8/8/8/8 w - - 0 1")));  // Black in check, White to move
    CHECK(is_valid(fen("R6k/8/6K1/8/8/8/8/8 b - - 0 1")));   // same, Black to move: fine
}

void test_move_generation() {
    Move moves[kMaxMoves];

    // Checkmate: Black Kh8, White Kg6 + Ra8. Black has no moves.
    Position mate = fen("R6k/8/6K1/8/8/8/8/8 b - - 0 1");
    CHECK(black_in_check(mate));
    CHECK(generate_moves(mate, moves) == 0);

    // Stalemate: Black Ka1, White Kb3 + Rb2 (protected). Not in check, no moves.
    Position stale = fen("8/8/8/8/8/1K6/1R6/k7 b - - 0 1");
    CHECK(!black_in_check(stale));
    CHECK(generate_moves(stale, moves) == 0);

    // Undefended rook next to the black king: capturing it is the only move.
    Position grab = fen("7K/8/8/8/8/8/1R6/k7 b - - 0 1");
    CHECK(generate_moves(grab, moves) == 1);
    CHECK(moves[0].capture == 1);

    // The black king can't step "behind" itself along the rook's line:
    // Black Ke5, Rook e1, White Ka8. e6 stays attacked once the king leaves e5.
    Position xray = fen("K7/8/8/4k3/8/8/8/4R3 b - - 0 1");
    int n = generate_moves(xray, moves);
    bool steps_to_e6 = false;
    for (int i = 0; i < n; ++i) steps_to_e6 |= moves[i].to == make_square(4, 5);
    CHECK(!steps_to_e6);

    // White: Ra1, Ke1, Black Kh8. Rook: a2..a8 (7) + b1..d1 (3, blocked by
    // its own king on e1) = 10. King: d1, f1, d2, e2, f2 = 5. Total 15.
    Position open = fen("7k/8/8/8/8/8/8/R3K3 w - - 0 1");
    CHECK(generate_moves(open, moves) == 15);
}

void test_notation() {
    Position p = fen("7k/8/6K1/8/8/8/8/R7 w - - 0 1");
    CHECK(to_fen(p) == "7k/8/6K1/8/8/8/8/R7 w - - 0 1");
    Move m{kWhiteRook, 0, 56, 0};  // Ra1-a8
    CHECK(move_to_string(p, m) == "Ra8#");
    CHECK(square_name(0) == "a1");
    CHECK(square_name(63) == "h8");

    std::string err;
    CHECK(!parse_fen("8/8/8/8/8/8/8/8 w - - 0 1", &err).has_value());
    CHECK(!parse_fen("7k/8/6K1/8/8/8/8/Q7 w - - 0 1", &err).has_value());
}

}  // namespace

int main() {
    test_encode_decode_round_trip();
    test_king_attacks();
    test_rook_attacks();
    test_validity();
    test_move_generation();
    test_notation();
    return wm::test::finish("core");
}
