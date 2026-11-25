// Which squares each piece attacks.
//
// These are computed on the fly instead of read from big lookup tables.
// On a GPU, every thread reading scattered table entries is slow (memory is
// the bottleneck), while a few bit operations per thread are nearly free.
// Lookup tables can be revisited later as an optimization experiment.
#pragma once

#include "core/common.h"

namespace wm {

// The (up to) 8 squares around a king.
WM_HD inline Bitboard king_attacks(int sq) {
    const Bitboard not_a_file = 0xfefefefefefefefeULL;
    const Bitboard not_h_file = 0x7f7f7f7f7f7f7f7fULL;
    const Bitboard b = bb(sq);
    // Shifting left/right by 1 moves one file; masks stop pieces on the edge
    // from wrapping around to the other side of the board.
    const Bitboard sideways = ((b << 1) & not_a_file) | ((b >> 1) & not_h_file);
    const Bitboard row = sideways | b;
    // Shifting by 8 moves one rank up or down; bits pushed off the board vanish.
    return (sideways | (row << 8) | (row >> 8));
}

// Squares a rook on `sq` attacks, given the occupied squares `occ`.
// The rook slides in each direction until it hits a piece (that square is
// included, since it could be captured or is defended) or the board edge.
WM_HD inline Bitboard rook_attacks(int sq, Bitboard occ) {
    Bitboard att = 0;
    const int r = rank_of(sq);
    const int f = file_of(sq);
    for (int rr = r + 1; rr < 8; ++rr) {  // up
        const int s = make_square(f, rr);
        att |= bb(s);
        if (occ & bb(s)) break;
    }
    for (int rr = r - 1; rr >= 0; --rr) {  // down
        const int s = make_square(f, rr);
        att |= bb(s);
        if (occ & bb(s)) break;
    }
    for (int ff = f + 1; ff < 8; ++ff) {  // right
        const int s = make_square(ff, r);
        att |= bb(s);
        if (occ & bb(s)) break;
    }
    for (int ff = f - 1; ff >= 0; --ff) {  // left
        const int s = make_square(ff, r);
        att |= bb(s);
        if (occ & bb(s)) break;
    }
    return att;
}

// Squares a bishop on `sq` attacks: slides diagonally until blocked.
WM_HD inline Bitboard bishop_attacks(int sq, Bitboard occ) {
    Bitboard att = 0;
    const int r = rank_of(sq);
    const int f = file_of(sq);
    const int dr[4] = {1, 1, -1, -1};
    const int df[4] = {1, -1, 1, -1};
    for (int d = 0; d < 4; ++d) {
        int rr = r + dr[d];
        int ff = f + df[d];
        while (rr >= 0 && rr < 8 && ff >= 0 && ff < 8) {
            const int s = make_square(ff, rr);
            att |= bb(s);
            if (occ & bb(s)) break;
            rr += dr[d];
            ff += df[d];
        }
    }
    return att;
}

WM_HD inline Bitboard queen_attacks(int sq, Bitboard occ) {
    return rook_attacks(sq, occ) | bishop_attacks(sq, occ);
}

// The (up to) 8 squares a knight jumps to. Each shift moves the knight by
// one of its L-shapes; the masks drop jumps that would wrap around an edge.
WM_HD inline Bitboard knight_attacks(int sq) {
    const Bitboard b = bb(sq);
    const Bitboard not_a = 0xfefefefefefefefeULL;
    const Bitboard not_ab = 0xfcfcfcfcfcfcfcfcULL;
    const Bitboard not_h = 0x7f7f7f7f7f7f7f7fULL;
    const Bitboard not_gh = 0x3f3f3f3f3f3f3f3fULL;
    return ((b << 17) & not_a) | ((b << 15) & not_h) | ((b << 10) & not_ab) |
           ((b << 6) & not_gh) | ((b >> 17) & not_h) | ((b >> 15) & not_a) |
           ((b >> 10) & not_gh) | ((b >> 6) & not_ab);
}

// Squares a pawn attacks (diagonally forward). White pawns move up the
// board (towards rank 8), black pawns move down.
WM_HD inline Bitboard pawn_attacks(int sq, bool white) {
    const Bitboard b = bb(sq);
    const Bitboard not_a = 0xfefefefefefefefeULL;
    const Bitboard not_h = 0x7f7f7f7f7f7f7f7fULL;
    if (white) return ((b << 9) & not_a) | ((b << 7) & not_h);
    return ((b >> 7) & not_a) | ((b >> 9) & not_h);
}

}  // namespace wm
