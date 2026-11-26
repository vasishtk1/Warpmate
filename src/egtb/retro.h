// Per-position work of the general backward solve (any material).
//
// Each round looks at every unsolved position and its children. Captures and
// promotions lead into other, already-solved tables, whose distances to mate
// can be larger than the current round, so a position is only decided when
// the relevant child value is within reach of this round:
//   odd ply  -> WIN  if some child is lost in at most ply-1
//   even ply -> LOSS if every child is won, all in at most ply-1
#pragma once

#include "core/values.h"
#include "egtb/position.h"

namespace wm::egtb {

WM_HD inline Value init_value(const Context& ctx, uint64_t index) {
    const Position p = decode(ctx.m, index);
    if (!is_valid(ctx.m, p)) return kInvalid;
    if (count_moves(ctx.m, p) == 0) {
        return in_check(ctx.m, p, p.stm) ? Value(0) : kDraw;  // mate or stalemate
    }
    return kUnknown;
}

// Visitor for odd rounds: stop at the first child lost soon enough.
struct FindWin {
    const Context* ctx;
    const Value* table;
    const Position* p;
    int limit;  // ply - 1
    bool found;
    WM_HD bool operator()(const Move& mv) {
        const Value c = child_value(*ctx, table, *p, mv);
        if (is_loss(c) && c <= limit) {
            found = true;
            return false;  // stop: one winning move is enough
        }
        return true;
    }
};

// Visitor for even rounds: stop at the first child that isn't won soon enough.
struct AllLose {
    const Context* ctx;
    const Value* table;
    const Position* p;
    int limit;  // ply - 1
    bool all;
    WM_HD bool operator()(const Move& mv) {
        const Value c = child_value(*ctx, table, *p, mv);
        if (!(is_win(c) && c <= limit)) {
            all = false;
            return false;  // stop: an escape exists (for now)
        }
        return true;
    }
};

WM_HD inline Value retro_step(const Context& ctx, const Value* table, uint64_t index, int ply) {
    const Value current = table[index];
    if (current != kUnknown) return current;
    const Position p = decode(ctx.m, index);
    if (ply & 1) {
        FindWin v{&ctx, table, &p, ply - 1, false};
        for_each_move(ctx.m, p, v);
        return v.found ? static_cast<Value>(ply) : kUnknown;
    }
    AllLose v{&ctx, table, &p, ply - 1, true};
    for_each_move(ctx.m, p, v);
    return v.all ? static_cast<Value>(ply) : kUnknown;
}

}  // namespace wm::egtb
