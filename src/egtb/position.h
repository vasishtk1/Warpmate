// General positions: indexing, legality, move generation and child lookups
// for any Material with up to kMaxPieces pieces.
//
// Everything is WM_HD (runs on CPU and GPU) and avoids heap memory, so the
// same code can later run inside CUDA kernels. Moves are produced through a
// visitor (a small struct with an operator()) rather than written into a big
// array, which keeps per-thread memory small on the GPU and lets callers stop
// early (e.g. as soon as one winning move is found).
#pragma once

#include "core/attacks.h"
#include "core/common.h"
#include "core/values.h"
#include "egtb/material.h"

namespace wm::egtb {

constexpr uint8_t kNoSquare = 64;   // marks a piece removed by a capture
constexpr uint8_t kNone = 0xFF;

struct Position {
    uint8_t sq[kMaxPieces];  // one square per piece, in the Material's order
    uint8_t stm;             // side to move: kWhite or kBlack
};

// Row number: [stm][sq0][sq1]...[sqN-1], six bits per square.
WM_HD inline uint64_t encode(const Material& m, const Position& p) {
    uint64_t index = p.stm;
    for (int i = 0; i < m.n; ++i) index = index * 64 + p.sq[i];
    return index;
}

WM_HD inline Position decode(const Material& m, uint64_t index) {
    Position p;
    for (int i = m.n - 1; i >= 0; --i) {
        p.sq[i] = static_cast<uint8_t>(index & 63);
        index >>= 6;
    }
    p.stm = static_cast<uint8_t>(index & 1);
    return p;
}

WM_HD inline Bitboard occupancy(const Material& m, const Position& p) {
    Bitboard occ = 0;
    for (int i = 0; i < m.n; ++i) {
        if (p.sq[i] != kNoSquare) occ |= bb(p.sq[i]);
    }
    return occ;
}

WM_HD inline Bitboard piece_attacks(uint8_t type, uint8_t color, int sq, Bitboard occ) {
    switch (type) {
        case kKing: return king_attacks(sq);
        case kQueen: return queen_attacks(sq, occ);
        case kRook: return rook_attacks(sq, occ);
        case kBishop: return bishop_attacks(sq, occ);
        case kKnight: return knight_attacks(sq);
        default: return pawn_attacks(sq, color == kWhite);
    }
}

// Is `target` attacked by any piece of colour `by`?
WM_HD inline bool is_attacked(const Material& m, const Position& p, int target, uint8_t by,
                              Bitboard occ) {
    for (int i = 0; i < m.n; ++i) {
        if (m.color[i] != by || p.sq[i] == kNoSquare) continue;
        if (piece_attacks(m.type[i], m.color[i], p.sq[i], occ) & bb(target)) return true;
    }
    return false;
}

WM_HD inline int king_index(const Material& m, uint8_t color) {
    for (int i = 0; i < m.n; ++i) {
        if (m.type[i] == kKing && m.color[i] == color) return i;
    }
    return -1;
}

WM_HD inline bool in_check(const Material& m, const Position& p, uint8_t color) {
    return is_attacked(m, p, p.sq[king_index(m, color)], color ^ 1, occupancy(m, p));
}

WM_HD inline bool is_valid(const Material& m, const Position& p) {
    Bitboard seen = 0;
    for (int i = 0; i < m.n; ++i) {
        if (seen & bb(p.sq[i])) return false;  // two pieces on one square
        seen |= bb(p.sq[i]);
        if (m.type[i] == kPawn && (rank_of(p.sq[i]) == 0 || rank_of(p.sq[i]) == 7)) {
            return false;  // pawns never stand on the first or last rank
        }
    }
    // The side that just moved can't have left its own king in check.
    // (This also rules out the two kings standing next to each other.)
    return !in_check(m, p, p.stm ^ 1);
}

struct Move {
    uint8_t piece;     // index of the moving piece
    uint8_t from;
    uint8_t to;
    uint8_t captured;  // index of the captured piece, or kNone
    uint8_t promo;     // PieceType promoted to, or kNone
};

// Calls visit(move) for every legal move of the side to move.
// visit returns false to stop early. Returns the number of moves visited.
//
// Moves are generated "pseudo-legally" (following piece rules) and then
// checked: a move is legal only if it doesn't leave the mover's king attacked.
template <typename Visitor>
WM_HD int for_each_move(const Material& m, const Position& p, Visitor& visit) {
    const uint8_t us = p.stm;
    const Bitboard occ = occupancy(m, p);
    Bitboard own = 0;
    for (int i = 0; i < m.n; ++i) {
        if (m.color[i] == us) own |= bb(p.sq[i]);
    }
    const int our_king = king_index(m, us);
    int count = 0;

    for (int i = 0; i < m.n; ++i) {
        if (m.color[i] != us) continue;
        const int from = p.sq[i];
        Bitboard targets;
        if (m.type[i] == kPawn) {
            // Pawns capture diagonally but move straight, and only onto
            // empty squares (two squares from their starting rank).
            const int dir = us == kWhite ? 8 : -8;
            const int start_rank = us == kWhite ? 1 : 6;
            targets = pawn_attacks(from, us == kWhite) & occ & ~own;
            const int one = from + dir;
            if (!(occ & bb(one))) {
                targets |= bb(one);
                const int two = one + dir;
                if (rank_of(from) == start_rank && !(occ & bb(two))) targets |= bb(two);
            }
        } else {
            targets = piece_attacks(m.type[i], us, from, occ) & ~own;
        }

        while (targets) {
            const int to = lsb(targets);
            targets &= targets - 1;

            // Which enemy piece (if any) is on the target square.
            uint8_t captured = kNone;
            for (int j = 0; j < m.n; ++j) {
                if (j != i && p.sq[j] == to) captured = static_cast<uint8_t>(j);
            }
            // Kings are never captured in a legal position.
            if (captured != kNone && m.type[captured] == kKing) continue;

            // Legality: make the move on a copy and check our king.
            Position next = p;
            next.sq[i] = static_cast<uint8_t>(to);
            if (captured != kNone) next.sq[captured] = kNoSquare;
            const Bitboard next_occ = occupancy(m, next);
            if (is_attacked(m, next, next.sq[our_king], us ^ 1, next_occ)) continue;

            const bool promotes =
                m.type[i] == kPawn && (rank_of(to) == 7 || rank_of(to) == 0);
            if (promotes) {
                for (uint8_t promo = kQueen; promo <= kKnight; ++promo) {
                    const Move mv{static_cast<uint8_t>(i), static_cast<uint8_t>(from),
                                  static_cast<uint8_t>(to), captured, promo};
                    ++count;
                    if (!visit(mv)) return count;
                }
            } else {
                const Move mv{static_cast<uint8_t>(i), static_cast<uint8_t>(from),
                              static_cast<uint8_t>(to), captured, kNone};
                ++count;
                if (!visit(mv)) return count;
            }
        }
    }
    return count;
}

struct CountMoves {
    WM_HD bool operator()(const Move&) { return true; }
};

WM_HD inline int count_moves(const Material& m, const Position& p) {
    CountMoves counter;
    return for_each_move(m, p, counter);
}

// A solved table for another material, used for captures and promotions,
// which lead out of the current endgame into a smaller or different one.
struct SubTable {
    Material m;
    const Value* values;
};

// Where a position lives after a move: either the current table (same
// material) or another one.
struct Context {
    Material m;              // this table's material
    const SubTable* subs;    // tables reachable by captures/promotions
    int num_subs;
};

// Looks up a position given as a free list of pieces (any order, either
// colour stronger), by putting it into canonical form first.
WM_HD inline Value lookup_pieces(const SubTable* subs, int num_subs, int n, uint8_t* type,
                                 uint8_t* color, uint8_t* sq, uint8_t stm) {
    if (n == 2) return kDraw;  // king vs king
    Material cm;
    cm.n = n;
    sort_pieces(n, type, color, sq);
    for (int i = 0; i < n; ++i) {
        cm.type[i] = type[i];
        cm.color[i] = color[i];
    }
    if (black_is_stronger(cm)) {
        // Swap colours and mirror the board top-to-bottom (a1 <-> a8).
        for (int i = 0; i < n; ++i) {
            color[i] ^= 1;
            sq[i] ^= 56;
        }
        stm ^= 1;
        sort_pieces(n, type, color, sq);
        for (int i = 0; i < n; ++i) {
            cm.type[i] = type[i];
            cm.color[i] = color[i];
        }
    }
    for (int s = 0; s < num_subs; ++s) {
        if (same_material(subs[s].m, cm)) {
            Position cp;
            for (int i = 0; i < n; ++i) cp.sq[i] = sq[i];
            cp.stm = stm;
            return subs[s].values[encode(cm, cp)];
        }
    }
    return kInvalid;  // missing sub-table (a bug: dependencies weren't solved)
}

// Value of the position reached by `mv`. Ordinary moves stay in this table;
// captures and promotions change the material and look up a sub-table.
WM_HD inline Value child_value(const Context& ctx, const Value* table, const Position& p,
                               const Move& mv) {
    if (mv.captured == kNone && mv.promo == kNone) {
        Position next = p;
        next.sq[mv.piece] = mv.to;
        next.stm ^= 1;
        return table[encode(ctx.m, next)];
    }
    uint8_t type[kMaxPieces], color[kMaxPieces], sq[kMaxPieces];
    int n = 0;
    for (int i = 0; i < ctx.m.n; ++i) {
        if (i == mv.captured) continue;
        type[n] = (i == mv.piece && mv.promo != kNone) ? mv.promo : ctx.m.type[i];
        color[n] = ctx.m.color[i];
        sq[n] = i == mv.piece ? mv.to : p.sq[i];
        ++n;
    }
    return lookup_pieces(ctx.subs, ctx.num_subs, n, type, color, sq, p.stm ^ 1);
}

}  // namespace wm::egtb
