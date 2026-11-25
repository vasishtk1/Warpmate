// Material: which pieces an endgame has, e.g. "KQvKR".
//
// The general solver works for any small set of pieces. A Material lists them
// in a fixed canonical order (white king, other white pieces strongest first,
// black king, other black pieces), and positions store one square per entry
// in that order.
//
// Each endgame is stored once, with the stronger side as White. A position
// where Black has the extra material (say K vs KR) is looked up in the KRvK
// table by swapping colours and mirroring the board top-to-bottom; the
// result, measured from the side to move, is unchanged.
#pragma once

#include "core/common.h"

namespace wm::egtb {

// Order matters: lower = stronger, and kings sort first.
enum PieceType : uint8_t { kKing = 0, kQueen, kRook, kBishop, kKnight, kPawn };
enum Color : uint8_t { kWhite = 0, kBlack = 1 };

constexpr int kMaxPieces = 5;

struct Material {
    int n;                         // number of pieces, including both kings
    uint8_t type[kMaxPieces];      // PieceType
    uint8_t color[kMaxPieces];     // Color
};

// Sort key for canonical order: by colour, then by piece type.
WM_HD inline int piece_key(uint8_t color, uint8_t type) { return color * 8 + type; }

// Sorts a list of pieces (and their squares, if given) into canonical order.
// Insertion sort: lists have at most kMaxPieces entries.
WM_HD inline void sort_pieces(int n, uint8_t* type, uint8_t* color, uint8_t* sq) {
    for (int i = 1; i < n; ++i) {
        const uint8_t t = type[i], c = color[i];
        const uint8_t s = sq ? sq[i] : 0;
        int j = i - 1;
        while (j >= 0 && piece_key(color[j], type[j]) > piece_key(c, t)) {
            type[j + 1] = type[j];
            color[j + 1] = color[j];
            if (sq) sq[j + 1] = sq[j];
            --j;
        }
        type[j + 1] = t;
        color[j + 1] = c;
        if (sq) sq[j + 1] = s;
    }
}

// True if Black's pieces are stronger than White's, meaning this material is
// stored colour-flipped. More pieces is stronger; with equal counts, compare
// strongest piece first (queen > rook > bishop > knight > pawn).
// Assumes canonical order.
WM_HD inline bool black_is_stronger(const Material& m) {
    uint8_t w[kMaxPieces], b[kMaxPieces];
    int nw = 0, nb = 0;
    for (int i = 0; i < m.n; ++i) {
        if (m.type[i] == kKing) continue;
        if (m.color[i] == kWhite) w[nw++] = m.type[i];
        else b[nb++] = m.type[i];
    }
    if (nw != nb) return nb > nw;
    for (int i = 0; i < nw; ++i) {
        if (w[i] != b[i]) return b[i] < w[i];
    }
    return false;  // identical sides: either colour works
}

WM_HD inline bool same_material(const Material& a, const Material& b) {
    if (a.n != b.n) return false;
    for (int i = 0; i < a.n; ++i) {
        if (a.type[i] != b.type[i] || a.color[i] != b.color[i]) return false;
    }
    return true;
}

// 2 sides to move x 64 squares per piece: 2 * 64^n rows.
WM_HD inline uint64_t table_size(const Material& m) {
    uint64_t size = 2;
    for (int i = 0; i < m.n; ++i) size *= 64;
    return size;
}

}  // namespace wm::egtb
