// Human-readable chess notation (CPU only): square names, FEN, move text.
#pragma once

#include <optional>
#include <string>

#include "core/krk.h"

namespace wm {

std::string square_name(int sq);  // 0 -> "a1"

namespace krk {

// Parses a FEN containing exactly a white king, white rook and black king,
// e.g. "7k/8/6K1/8/8/8/8/R7 w - - 0 1". Only the board and side-to-move
// fields are used. On failure returns nullopt and fills `error`.
std::optional<Position> parse_fen(const std::string& fen, std::string* error = nullptr);

std::string to_fen(const Position& p);

// Short algebraic notation, e.g. "Ra8#", "Kd6", "Kxb2", with "+" for check
// and "#" for checkmate.
std::string move_to_string(const Position& p, const Move& m);

// Plain-English verdict for a stored value, e.g. "White mates in 7".
std::string describe(const Position& p, Value v);

}  // namespace krk
}  // namespace wm
