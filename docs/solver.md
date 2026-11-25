# Solver design

How Warpmate solves an endgame, using king + rook vs king (KRvK) as the first case.
The CPU solver described here is the reference implementation that the CUDA solver will be checked against.

## Board representation

Squares are numbered 0–63 (`a1` = 0, `h1` = 7, `a8` = 56, `h8` = 63). Sets of squares are stored as **bitboards**: 64-bit integers with one bit per square, so set operations such as "squares the king could move to, minus squares the rook attacks" are single AND/OR instructions.

King and rook attacks are computed with a few bit operations and short loops rather than large lookup tables. On a GPU, scattered table reads are expensive while arithmetic is cheap, so this keeps per-thread work in registers.

## Indexing positions

A KRvK position is three squares plus the side to move. Each position maps to a table row:

```
index = side_to_move << 18 | white_king << 12 | white_rook << 6 | black_king
```

That gives 2 × 64³ = **524,288** rows. Encoding and decoding are a few shifts and masks. Rows for impossible positions (overlapping pieces, kings touching, the side not to move in check) are marked invalid. A later version will remove most of them using board symmetry.

## Stored values

Each position takes **one byte**, holding the distance to mate in plies (half-moves) from the side to move's point of view:

| Value | Meaning |
|---|---|
| 0 | side to move is checkmated |
| odd *n* | side to move wins, mate in (*n* + 1) / 2 moves |
| even *n* > 0 | side to move loses, mated in *n* / 2 moves |
| 253 | draw |
| 254 | not yet solved (only during solving) |
| 255 | invalid position |

Parity encodes win vs loss, so no extra bits are needed.

## Retrograde analysis

1. **Initialize:** mark invalid positions, checkmates (value 0) and stalemates (draw).
2. **Round `ply` = 1, 2, 3, …**, over every unsolved position:
   - *odd ply:* if **any** move reaches a lost position, the position is won in `ply`.
   - *even ply:* if **every** move reaches a won position, the position is lost in `ply`.
3. **Stop** after a round that solves nothing. Remaining positions are draws.

Because rounds run in increasing order, the first round in which a position qualifies gives its exact distance to mate.

### Why this parallelizes

Each position's update reads only its children's values and writes only its own entry. Updates can even be written in place while other positions are being processed: odd rounds only write odd values and only read even ones, and even rounds the reverse, so no write in a round can change another position's decision in that same round. This makes every position in a round independent, which maps directly to one GPU thread per position.

The per-position logic (`init_value`, `retro_step` in `src/solver/krk_retro.h`) and all chess logic are marked `WM_HD`, which expands to `__host__ __device__` under nvcc. The same code is shared by the CPU solver and the CUDA kernels.

## Implementations

All three share `init_value` and `retro_step`; they differ only in how positions are distributed.

| Solver | Where | How a round runs |
|---|---|---|
| `solve` | `src/solver/krk_solver.cpp` | one CPU thread loops over all positions, updating the table in place |
| `solve_parallel` | `src/solver/krk_solver.cpp` | positions split into contiguous ranges across CPU threads; reads from one buffer and writes to a copy, since concurrent read/write of the same bytes is undefined behaviour in standard C++ |
| `solve_gpu` | `src/gpu/krk_gpu.cu` | one CUDA thread per position (2,048 blocks × 256 threads); the table is updated in place on the GPU, safe by the parity argument above |

### GPU solver (v1)

- **Kernels:** `init_kernel` (round 0), `step_kernel` (one round), `finalize_kernel` (unsolved → draw).
- **Host loop:** for each round, reset a device counter, launch `step_kernel`, then copy the counter back. That copy also synchronizes with the kernel, and the loop stops when a round solves nothing.
- **Counters** (positions changed, checkmates, stalemates) are updated with `atomicAdd`.
- **Data movement:** the table stays on the GPU for the whole solve; only a 4-byte counter crosses each round, and the 0.5 MB table is copied back once.
- **Error handling:** every CUDA API call is checked, and kernel launches are checked with `cudaGetLastError`.

Profiling shows `step_kernel` accounts for ~98% of GPU time, mainly because every round revisits all 524,288 positions even though most are already solved (see [benchmarks.md](benchmarks.md)).

## General solver (any material)

`src/egtb/` generalizes everything above to any set of up to 4 pieces (5 supported by the data structures), with all six piece types including pawns and promotion.

### Material and canonical form

A **material** such as `KQvKR` lists the pieces in canonical order: white king, other white pieces strongest first (Q, R, B, N, P), then the same for Black. A position stores one square per piece in that order, and the row index is `[stm][sq0][sq1]…`, the same layout as the KRvK solver. The general KRvK table is therefore byte-for-byte identical to the specialized one, which the tests check.

Each endgame is stored once, **with the stronger side as White**. A position where Black has the extra material is looked up by swapping colours, mirroring the board top-to-bottom (`square ^ 56`) and flipping the side to move. Values are measured from the side to move, so they carry over unchanged.

### Linked tables

Captures and promotions leave the current endgame. In KPvK, promoting the pawn reaches KQvK, KRvK, KBvK or KNvK. In KRvKP, capturing the pawn reaches KRvK, and promoting it reaches KRvKQ (stored as KQvKR). The solver computes these **dependencies**, solves them first, and passes them to the per-position step as read-only sub-tables. A child position reached by a capture or promotion is put into canonical form and looked up in the matching sub-table.

### Rounds with sub-tables

Sub-table children can have distances larger than the current round, so the rules become:

- *odd ply:* **win** if some child is lost in at most `ply − 1`
- *even ply:* **loss** if every child is won, all in at most `ply − 1`

Solving can't stop at the first empty round either. It continues until a round finds nothing **and** the round number has passed the largest distance in any sub-table.

### Move generation

Moves are produced through a **visitor** (a small struct whose `operator()` receives each legal move) instead of being written into an array. This keeps per-thread memory small on the GPU, and it lets the solver stop early: an odd round stops at the first winning move, an even round at the first escape. All of it is `WM_HD`, so it compiles for CUDA unchanged. Legality is checked by making each move on a copy and testing whether the mover's king is attacked.

**Not handled yet:** en passant, which only matters with pawns on both sides (e.g. KPvKP). Castling never arises in these endgames.

### 3-piece results

From `warpmate-gen --all3` (Apple M1, all cores):

| Endgame | Rounds | Time | White to move | Longest forced mate |
|---|---|---|---|---|
| KQvK | 21 | 0.25 s | 144,508 wins | 10 moves |
| KRvK | 33 | 0.33 s | 175,168 wins | 16 moves |
| KBvK | 2 | 0.04 s | 193,284 draws | none |
| KNvK | 2 | 0.04 s | 205,496 draws | none |
| KPvK | 57 | 0.26 s | 124,960 wins, 38,368 draws | 28 moves |

All five tables match Syzygy on every legal position ([verification.md](verification.md)).

## Correctness checks

The test suite (`tests/test_solver.cpp`) checks:

- known positions: checkmate, mate in 1, stalemate, a forced rook capture, an invalid position
- the longest forced mate is **16 moves** (31 plies), the known result for KRvK
- **every entry is consistent with its children**: a win in *n* has a child lost in exactly *n* − 1, a loss in *n* has only won children with the slowest at *n* − 1, and a draw has no lost child and at least one non-won child
- tables survive a save/load round trip
- the multithreaded solver (1, 3 and 8 threads) produces the identical table
- the GPU solver produces a table byte-for-byte identical to the CPU's (`tests/test_gpu.cpp`, built with CUDA)

The general solver's tests (`tests/test_egtb.cpp`) check:

- knight, bishop, queen and pawn attacks, including board edges
- material parsing, canonical form and dependency lists
- move generation: pawn pushes, promotions (four choices) and pinned pieces
- the general KRvK table is byte-for-byte identical to the specialized solver's
- known results: KQvK longest mate 10 moves, KRvK 16, KBvK and KNvK all draws
- colour-flipped lookups give the same value as the original position
- every entry of all five 3-piece tables agrees with its children, including children in other tables

An independent check against the Syzygy tablebases is provided by `scripts/verify_syzygy.py`: all 399,112 legal KRvK positions match, with 0 mismatches ([verification.md](verification.md)).

## KRvK results

Output of `warpmate-solve`:

| | Count |
|---|---|
| Table rows | 524,288 |
| Invalid | 125,176 |
| Checkmates | 216 |
| Stalemates | 68 |
| White to move | 175,168 wins, 0 draws, 0 losses |
| Black to move | 0 wins, 22,244 draws, 201,700 losses |
| Rounds | 33 (32 that solve positions + 1 empty) |
| Longest forced mate | 16 moves, e.g. `8/8/8/8/8/2k5/1R6/K7 w` |

Every valid white-to-move position is a win. Black's only drawing chances are stalemate or capturing an undefended rook.
