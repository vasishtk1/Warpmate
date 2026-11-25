// Basic types and helpers shared by every part of Warpmate.
//
// Squares are numbered 0..63: a1 = 0, b1 = 1, ..., h1 = 7, a2 = 8, ..., h8 = 63.
// A "bitboard" is a 64-bit integer with one bit per square, so a set of squares
// fits in a single register and set operations become single instructions
// (AND = intersection, OR = union). This is cheap on both CPUs and GPUs.
#pragma once

#include <cstdint>

// WM_HD marks functions that must run on both the CPU and the GPU.
// When compiled by nvcc (the CUDA compiler) it expands to __host__ __device__,
// meaning "compile this for the CPU (host) and for the GPU (device)".
// When compiled by a normal C++ compiler it expands to nothing.
// Writing the chess logic this way lets the exact same code power the CPU
// solver and, later, the CUDA kernels. (__CUDA__ covers clang's CUDA mode.)
#if defined(__CUDACC__) || defined(__CUDA__)
#define WM_HD __host__ __device__
#else
#define WM_HD
#endif

namespace wm {

using Bitboard = uint64_t;

WM_HD inline Bitboard bb(int sq) { return Bitboard(1) << sq; }
WM_HD inline int file_of(int sq) { return sq & 7; }
WM_HD inline int rank_of(int sq) { return sq >> 3; }
WM_HD inline int make_square(int file, int rank) { return rank * 8 + file; }

// Index of the lowest set bit. CPUs and GPUs expose this through different
// intrinsics, so pick the right one for whichever side is being compiled.
// __CUDA_ARCH__ is defined only while nvcc compiles the GPU version.
WM_HD inline int lsb(Bitboard b) {
#if defined(__CUDA_ARCH__)
    return __ffsll(static_cast<long long>(b)) - 1;
#else
    return __builtin_ctzll(b);
#endif
}

}  // namespace wm
