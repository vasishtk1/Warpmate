#include "solver/krk_solver.h"

#include <chrono>

#include "solver/krk_retro.h"

namespace wm::krk {

SolveResult solve(const std::function<void(const RoundStats&)>& on_round) {
    SolveResult result;
    const auto start = std::chrono::steady_clock::now();

    // Round 0: mark invalid positions, checkmates and stalemates.
    result.table.resize(kNumPositions);
    for (uint32_t i = 0; i < kNumPositions; ++i) {
        const Value v = init_value(i);
        result.table[i] = v;
        if (v == 0) ++result.checkmates;
        if (v == kDraw) ++result.stalemates;
    }

    // Rounds 1, 2, 3, ...: sweep every position until a round changes nothing.
    // (Once a round finds nothing, the next one can't either: it would be
    // asking the same question with the same information.)
    //
    // This "check everything every round" sweep is deliberately simple; it is
    // the same shape the first CUDA version will have, with this loop body
    // becoming one GPU thread.
    Value* table = result.table.data();
    for (int ply = 1; ply < kDraw; ++ply) {
        uint32_t changed = 0;
        for (uint32_t i = 0; i < kNumPositions; ++i) {
            if (table[i] != kUnknown) continue;
            const Value v = retro_step(table, i, ply);
            if (v != kUnknown) {
                table[i] = v;
                ++changed;
            }
        }
        const RoundStats stats{ply, changed};
        result.rounds.push_back(stats);
        if (on_round) on_round(stats);
        if (changed == 0) break;
    }

    // Anything never decided can't be forced either way: a draw.
    for (Value& v : result.table) {
        if (v == kUnknown) v = kDraw;
    }

    result.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return result;
}

TableSummary summarize(const std::vector<Value>& table) {
    TableSummary s;
    for (uint32_t i = 0; i < table.size(); ++i) {
        const Value v = table[i];
        if (v == kInvalid) {
            ++s.invalid;
            continue;
        }
        SideSummary& side = s.side[decode(i).stm];
        if (is_win(v)) {
            ++side.wins;
            if (v > s.longest_win) {
                s.longest_win = v;
                s.longest_win_index = i;
            }
        } else if (is_loss(v)) {
            ++side.losses;
        } else {
            ++side.draws;
        }
    }
    return s;
}

namespace {

// Ranks a child value from the mover's point of view (higher is better).
// The child is the opponent's position, so the opponent losing is good.
int score_for_mover(Value child) {
    if (is_loss(child)) return 1000 - child;  // opponent lost: sooner is better
    if (is_win(child)) return -1000 + child;  // opponent wins: later is better
    return 0;                                 // draw
}

}  // namespace

bool best_move(const Value* table, const Position& p, Move* out) {
    Move moves[kMaxMoves];
    const int n = generate_moves(p, moves);
    if (n == 0) return false;
    int best = 0;
    int best_score = score_for_mover(child_value(table, p, moves[0]));
    for (int i = 1; i < n; ++i) {
        const int score = score_for_mover(child_value(table, p, moves[i]));
        if (score > best_score) {
            best_score = score;
            best = i;
        }
    }
    *out = moves[best];
    return true;
}

}  // namespace wm::krk
