// Un-moves: the positions that could have led to a given position.
//
// A move goes forward: from position P, the side to move plays, giving Q.
// An un-move goes backward: given Q, which positions P could have been on the
// board one move earlier? The faster ("frontier") solver needs this: when Q
// is solved, only its parents P can be affected next round, so only they are
// re-checked instead of every position in the table.
//
// Only moves that stay in the same table are reversed: ordinary moves and
// pawn pushes. Captures and promotions change the material, so their parents
// are in a different (larger) table and are never needed here.
//
// Everything is WM_HD (runs on CPU and GPU), like position.h.
#pragma once

#include "core/attacks.h"
#include "core/common.h"
#include "egtb/position.h"

namespace wm::egtb {

// Calls visit(parent) for every valid position that reaches `q` with one
// legal, non-capturing, non-promoting move. visit returns false to stop.
//
// Why this is exactly the reverse of for_each_move (tested in test_egtb.cpp):
//   * The side that just moved is the one NOT to move in q.
//   * A non-pawn piece now on square t came from a square f it attacks from
//     t. Sliding moves are symmetric: the squares between f and t are the
//     same in both positions, since nothing was captured.
//   * f must be empty in q (the piece left it, nothing moved in).
//   * A pawn came from one square behind, or two from its starting rank with
//     both squares behind it empty. It never came from a capture, because
//     captures leave the table.
//   * The forward move was legal because q is valid (the mover's king isn't
//     in check in q), so only the parent's own validity needs checking.
template <typename Visitor>
WM_HD void for_each_unmove(const Material& m, const Position& q, Visitor& visit) {
    const uint8_t mover = q.stm ^ 1;  // the side that just moved
    const Bitboard occ = occupancy(m, q);

    for (int i = 0; i < m.n; ++i) {
        if (m.color[i] != mover) continue;
        const int to = q.sq[i];
        Bitboard from;
        if (m.type[i] == kPawn) {
            const int dir = mover == kWhite ? 8 : -8;
            const int start_rank = mover == kWhite ? 1 : 6;
            from = 0;
            const int one = to - dir;  // one square behind
            if (one >= 0 && one < 64 && !(occ & bb(one))) {
                from |= bb(one);
                const int two = one - dir;
                if (two >= 0 && two < 64 && rank_of(two) == start_rank && !(occ & bb(two))) {
                    from |= bb(two);
                }
            }
        } else {
            // Squares this piece attacks from `to` are where it could have
            // come from (minus occupied ones).
            from = piece_attacks(m.type[i], mover, to, occ) & ~occ;
        }

        while (from) {
            const int f = lsb(from);
            from &= from - 1;
            Position p = q;
            p.sq[i] = static_cast<uint8_t>(f);
            p.stm = mover;
            // is_valid rejects pawns on the first/last rank and positions
            // where the side not to move is in check.
            if (!is_valid(m, p)) continue;
            if (!visit(p)) return;
        }
    }
}

}  // namespace wm::egtb
