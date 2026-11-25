// King + Rook vs King: positions, indexing and move generation.
//
// White has a king and a rook; Black has only a king. Every function here is
// marked WM_HD so it can later run inside a CUDA kernel unchanged.
#pragma once

#include "core/attacks.h"
#include "core/common.h"
#include "core/values.h"

namespace wm::krk {

enum Side : int { kWhite = 0, kBlack = 1 };
enum PieceId : uint8_t { kWhiteKing = 0, kWhiteRook = 1, kBlackKing = 2 };

struct Position {
    int wk;   // white king square
    int wr;   // white rook square
    int bk;   // black king square
    int stm;  // side to move: kWhite or kBlack
};

// Every arrangement of the three pieces, times two for the side to move:
// 2 * 64 * 64 * 64 = 524,288. Each one gets a row in the table.
constexpr uint32_t kNumPositions = 2u * 64 * 64 * 64;

// Position -> table row. The layout is simply [stm][wk][wr][bk], six bits per
// square, which makes encode/decode a few shifts and masks. Many rows are
// impossible positions (pieces overlapping, kings touching); they are marked
// invalid. Later versions can squeeze those out using board symmetry.
WM_HD inline uint32_t encode(const Position& p) {
    return (static_cast<uint32_t>(p.stm) << 18) | (static_cast<uint32_t>(p.wk) << 12) |
           (static_cast<uint32_t>(p.wr) << 6) | static_cast<uint32_t>(p.bk);
}

WM_HD inline Position decode(uint32_t index) {
    Position p;
    p.bk = static_cast<int>(index & 63);
    p.wr = static_cast<int>((index >> 6) & 63);
    p.wk = static_cast<int>((index >> 12) & 63);
    p.stm = static_cast<int>((index >> 18) & 1);
    return p;
}

// The rook's attacks are blocked only by the white king. The black king never
// blocks an attack on its own square, and when it moves it leaves its old
// square, so it is left out of the blockers on purpose.
WM_HD inline Bitboard rook_attacks_on_black(const Position& p) {
    return rook_attacks(p.wr, bb(p.wk));
}

WM_HD inline bool black_in_check(const Position& p) {
    return (rook_attacks_on_black(p) & bb(p.bk)) != 0;
}

// White can never be in check here: the only black piece is the king, and
// kings may never stand next to each other.
WM_HD inline bool side_to_move_in_check(const Position& p) {
    return p.stm == kBlack && black_in_check(p);
}

WM_HD inline bool is_valid(const Position& p) {
    if (p.wk == p.wr || p.wk == p.bk || p.wr == p.bk) return false;  // overlap
    if (king_attacks(p.wk) & bb(p.bk)) return false;                 // kings touching
    // The side NOT to move can't be in check: that would mean the previous
    // move left its own king attacked, which is illegal.
    if (p.stm == kWhite && black_in_check(p)) return false;
    return true;
}

struct Move {
    uint8_t piece;    // PieceId
    uint8_t from;
    uint8_t to;
    uint8_t capture;  // 1 if Black's king captures the rook
};

// White: up to 8 king moves + 14 rook moves. Black: up to 8 king moves.
constexpr int kMaxMoves = 24;

// Writes all legal moves into `out` and returns how many there are.
// Assumes `p` is valid.
WM_HD inline int generate_moves(const Position& p, Move* out) {
    int n = 0;
    if (p.stm == kWhite) {
        // King: any neighbouring square that isn't the rook's and isn't next
        // to the black king.
        Bitboard targets = king_attacks(p.wk) & ~bb(p.wr) & ~king_attacks(p.bk) & ~bb(p.bk);
        while (targets) {
            const int to = lsb(targets);
            targets &= targets - 1;  // clear the lowest set bit
            out[n++] = Move{kWhiteKing, static_cast<uint8_t>(p.wk), static_cast<uint8_t>(to), 0};
        }
        // Rook: slide until blocked by either king, without landing on one.
        targets = rook_attacks(p.wr, bb(p.wk) | bb(p.bk)) & ~bb(p.wk) & ~bb(p.bk);
        while (targets) {
            const int to = lsb(targets);
            targets &= targets - 1;
            out[n++] = Move{kWhiteRook, static_cast<uint8_t>(p.wr), static_cast<uint8_t>(to), 0};
        }
    } else {
        // Black king: can't step next to the white king or onto a square the
        // rook attacks. It may capture the rook if the white king doesn't
        // protect it (protected rook squares are next to the white king, so
        // the first mask already removes them).
        const Bitboard rook_danger = rook_attacks_on_black(p);
        Bitboard targets = king_attacks(p.bk) & ~king_attacks(p.wk);
        while (targets) {
            const int to = lsb(targets);
            targets &= targets - 1;
            if (to == p.wr) {
                out[n++] = Move{kBlackKing, static_cast<uint8_t>(p.bk), static_cast<uint8_t>(to), 1};
            } else if (!(rook_danger & bb(to))) {
                out[n++] = Move{kBlackKing, static_cast<uint8_t>(p.bk), static_cast<uint8_t>(to), 0};
            }
        }
    }
    return n;
}

// The position after `m` (side to move flips). Not meaningful for captures,
// which leave only two kings and are handled as draws by the caller.
WM_HD inline Position make_move(Position p, const Move& m) {
    if (m.piece == kWhiteKing) {
        p.wk = m.to;
    } else if (m.piece == kWhiteRook) {
        p.wr = m.to;
    } else {
        p.bk = m.to;
    }
    p.stm ^= 1;
    return p;
}

// Value of the position reached by `m`, looked up in `table`.
// Capturing the rook leaves king vs king, which is always a draw.
WM_HD inline Value child_value(const Value* table, const Position& p, const Move& m) {
    if (m.capture) return kDraw;
    return table[encode(make_move(p, m))];
}

}  // namespace wm::krk
