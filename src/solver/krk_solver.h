// CPU driver for the King + Rook vs King solve, plus table statistics.
#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "core/krk.h"
#include "core/values.h"

namespace wm::krk {

struct RoundStats {
    int ply;
    uint32_t newly_solved;
};

struct SolveResult {
    std::vector<Value> table;  // one Value per position index
    std::vector<RoundStats> rounds;
    uint32_t checkmates = 0;
    uint32_t stalemates = 0;
    double seconds = 0.0;
};

// Solves every position. `on_round` (optional) is called after each round,
// e.g. to print progress.
SolveResult solve(const std::function<void(const RoundStats&)>& on_round = nullptr);

struct SideSummary {
    uint64_t wins = 0;
    uint64_t draws = 0;
    uint64_t losses = 0;
};

struct TableSummary {
    SideSummary side[2];  // indexed by kWhite / kBlack
    uint64_t invalid = 0;
    Value longest_win = 0;            // largest odd value in the table
    uint32_t longest_win_index = 0;   // one position achieving it
};

TableSummary summarize(const std::vector<Value>& table);

// The best move by perfect play: fastest mate when winning, longest
// resistance when losing, and a non-losing move when drawing.
// Returns false if the position has no legal moves.
bool best_move(const Value* table, const Position& p, Move* out);

}  // namespace wm::krk
