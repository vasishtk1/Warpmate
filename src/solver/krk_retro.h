// The per-position work of the retrograde (backward) solve.
//
// These two functions are the heart of the solver, and they are written to
// look at ONE position only. The CPU solver calls them in a loop over every
// position; the CUDA solver will call them from a kernel where each GPU
// thread handles one position. No position's update depends on another
// position's update in the same round, which is what makes this parallel.
#pragma once

#include "core/krk.h"
#include "core/values.h"

namespace wm::krk {

// Starting value for a position before any rounds run.
WM_HD inline Value init_value(uint32_t index) {
    const Position p = decode(index);
    if (!is_valid(p)) return kInvalid;
    Move moves[kMaxMoves];
    if (generate_moves(p, moves) == 0) {
        // No legal moves: checkmate if in check, otherwise stalemate.
        return side_to_move_in_check(p) ? Value(0) : kDraw;
    }
    return kUnknown;
}

// One round of the backward solve for one position.
//
// Round `ply` only creates values equal to `ply`:
//   odd ply  -> look for WINS:   any move reaching a lost position
//   even ply -> look for LOSSES: every move reaching a won position
//
// Returns the new value, or kUnknown if the position still can't be decided.
//
// Safe to run with the table being updated in place (even by many GPU threads
// at once): odd rounds only write odd values and only care about even ones,
// and even rounds only write even values and only care about odd ones, so a
// write in this round never changes another position's decision this round.
WM_HD inline Value retro_step(const Value* table, uint32_t index, int ply) {
    const Value current = table[index];
    if (current != kUnknown) return current;

    const Position p = decode(index);
    Move moves[kMaxMoves];
    const int n = generate_moves(p, moves);

    if (ply & 1) {
        // Win if ANY move leaves the opponent lost. Because rounds run in
        // order, the fastest such loss was created in the previous round,
        // so the distance is exactly `ply`.
        for (int i = 0; i < n; ++i) {
            if (is_loss(child_value(table, p, moves[i]))) return static_cast<Value>(ply);
        }
        return kUnknown;
    }

    // Loss only if EVERY move leaves the opponent winning. One draw or
    // unsolved reply means we can't call it lost (yet).
    for (int i = 0; i < n; ++i) {
        if (!is_win(child_value(table, p, moves[i]))) return kUnknown;
    }
    return static_cast<Value>(ply);
}

}  // namespace wm::krk
