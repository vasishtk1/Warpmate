// Human-readable input/output for the general solver (CPU only).
#pragma once

#include <optional>
#include <string>

#include "egtb/material.h"
#include "egtb/position.h"

namespace wm::egtb {

// A position as a plain list of pieces, in any order, either side stronger.
struct Board {
    int n = 0;
    uint8_t type[kMaxPieces];
    uint8_t color[kMaxPieces];
    uint8_t sq[kMaxPieces];
    uint8_t stm = kWhite;
};

// Parses the board and side-to-move fields of a FEN (castling and en passant
// fields are ignored). Each side needs exactly one king.
std::optional<Board> parse_fen(const std::string& fen, std::string* error = nullptr);
std::string to_fen(const Board& b);

// The board as a (non-colour-flipped) Material + Position, pieces sorted.
void to_position(const Board& b, Material* m, Position* p);
Board to_board(const Material& m, const Position& p);

// The board after `mv`, which may have one piece fewer or a promoted piece.
Board apply_move(const Material& m, const Position& p, const Move& mv);

// Looks a board up among `subs` (tables for any material), colour-flipping
// if needed. Returns kInvalid if no table matches.
Value lookup(const SubTable* subs, int num_subs, Board b);

// Short algebraic notation, e.g. "Qh7#", "Kxd4", "exd5", "e8=Q+".
std::string move_to_string(const Material& m, const Position& p, const Move& mv);

// "White mates in 7", "Draw", ...
std::string describe(uint8_t stm, Value v);

}  // namespace wm::egtb
