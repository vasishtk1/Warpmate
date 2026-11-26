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

// ---------------------------------------------------------------------------
// Frontier solving: which positions can change in a given round?
//
// A position still unsolved at round `ply` can only be decided then if one
// of its children has value exactly ply-1:
//   * odd ply (win): it needs a child lost in at most ply-1. Any child lost
//     sooner would already have made it a win two rounds ago.
//   * even ply (loss): every child must be won in at most ply-1. If the
//     slowest were faster than ply-1, it would have been lost two rounds ago.
// A child in this table gets value ply-1 by being solved in round ply-1, so
// the solver marks its parents (via un-moves). A child in a sub-table
// (reached by a capture or promotion) has a fixed value, so each position
// records up front the rounds at which those children matter:
//   win_round  = (smallest lost sub-child) + 1: the round a capture or
//                promotion first wins
//   loss_round = (largest won sub-child) + 1, only if every sub-child is
//                won: the round the last escape through a capture runs out
// 0 means "never". Other sub-child values can't change the result.

struct SubTriggers {
    const Context* ctx;
    const Value* table;
    const Position* p;
    int min_loss = 1000;
    int max_win = -1;
    bool any_sub = false;
    bool all_won = true;
    WM_HD bool operator()(const Move& mv) {
        if (mv.captured == kNone && mv.promo == kNone) return true;  // same table
        const Value c = child_value(*ctx, table, *p, mv);
        any_sub = true;
        if (is_loss(c) && c < min_loss) min_loss = c;
        if (is_win(c)) {
            if (c > max_win) max_win = c;
        } else {
            all_won = false;
        }
        return true;
    }
};

WM_HD inline void sub_triggers(const Context& ctx, const Value* table, uint64_t index,
                               uint8_t* win_round, uint8_t* loss_round) {
    const Position p = decode(ctx.m, index);
    SubTriggers t{&ctx, table, &p};
    for_each_move(ctx.m, p, t);
    *win_round = t.min_loss < 1000 ? static_cast<uint8_t>(t.min_loss + 1) : 0;
    *loss_round = (t.any_sub && t.all_won) ? static_cast<uint8_t>(t.max_win + 1) : 0;
}

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
