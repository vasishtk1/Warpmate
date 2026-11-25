// How a solved position is stored: one byte per position.
//
// The byte holds the distance to mate in plies (half-moves), from the point
// of view of the side to move:
//   0        side to move is checkmated right now
//   1        side to move mates in 1 move
//   2        side to move is mated in 1 move (every move allows mate in 1)
//   3        side to move mates in 2
//   ...      odd  = side to move WINS,  in (value + 1) / 2 moves
//            even = side to move LOSES, in  value / 2 moves
//   253      draw
//   254      not solved yet (only during solving; becomes a draw at the end)
//   255      invalid position (can't occur in a real game)
//
// One byte per position keeps tables small, which matters on the GPU where
// memory traffic, not arithmetic, usually limits speed.
#pragma once

#include <cstdint>

#include "core/common.h"

namespace wm {

using Value = uint8_t;

constexpr Value kDraw = 253;
constexpr Value kUnknown = 254;
constexpr Value kInvalid = 255;

WM_HD inline bool is_decided(Value v) { return v < kDraw; }
WM_HD inline bool is_win(Value v) { return v < kDraw && (v & 1) != 0; }
WM_HD inline bool is_loss(Value v) { return v < kDraw && (v & 1) == 0; }

// Moves to mate for a decided value (0 for "already checkmated").
WM_HD inline int moves_to_mate(Value v) { return (v + 1) / 2; }

}  // namespace wm
