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

}  // namespace wm
