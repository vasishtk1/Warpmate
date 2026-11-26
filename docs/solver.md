# Solver design

How Warpmate solves an endgame, using king + rook vs king (KRvK) as the first case.
The CPU solver described here is the reference implementation that the CUDA solver is checked against.

## Board representation

Squares are numbered 0 to 63 (`a1` = 0, `h1` = 7, `a8` = 56, `h8` = 63). Sets of squares are stored as **bitboards**: 64-bit integers with one bit per square, so set operations such as "squares the king could move to, minus squares the rook attacks" are single AND/OR instructions.

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
2. **Round `ply` = 1, 2, 3, ...**, over every unsolved position:
   - *odd ply:* if **any** move reaches a lost position, the position is won in `ply`.
   - *even ply:* if **every** move reaches a won position, the position is lost in `ply`.
3. **Stop** after a round that solves nothing. Remaining positions are draws.

Because rounds run in increasing order, the first round in which a position qualifies gives its exact distance to mate.

### Why this parallelizes

Each position's update reads only its children's values and writes only its own entry. Updates can even be written in place while other positions are being processed: odd rounds only write odd values and only read even ones, and even rounds the reverse, so no write in a round can change another position's decision in that same round. This makes every position in a round independent, which maps directly to one GPU thread per position.

The per-position logic (`init_value`, `retro_step` in `src/egtb/retro.h`) and all chess logic are marked `WM_HD`, which expands to `__host__ __device__` under nvcc. The same code is shared by the CPU solver and the CUDA kernels.

## Implementations

Both share `init_value` and `retro_step`, and both offer two algorithms: **full scan** (described here: every round checks every unsolved position) and **frontier** (the default: every round checks only positions that can change, see [Frontier solver](#frontier-solver)).

| Solver | Where | How a round runs |
|---|---|---|
| `Solver` | `src/egtb/solver.cpp` | positions split into contiguous ranges across CPU threads (1 thread is also supported); reads from one buffer and writes to a copy, since concurrent read/write of the same bytes is undefined behaviour in standard C++ |
| `GpuSolver` | `src/gpu/egtb_gpu.cu` | one CUDA thread per position (256 threads per block); the table is updated in place on the GPU, safe by the parity argument above |

The project started with separate solvers written only for KRvK. They were removed once the general solver below produced byte-for-byte identical KRvK tables.

### GPU solver (v1)

- **Kernels:** `init_kernel` (round 0), `step_kernel` (one round), `finalize_kernel` (turns unsolved positions into draws).
- **Host loop:** for each round, reset a device counter, launch `step_kernel`, then copy the counter back. That copy also synchronizes with the kernel, and the loop stops when a round solves nothing.
- **Counters** (positions changed, checkmates, stalemates) are updated with `atomicAdd`.
- **Data movement:** the table stays on the GPU for the whole solve; only a small counter crosses each round, and the finished table is copied back once.
- **Error handling:** every CUDA API call is checked, and kernel launches are checked with `cudaGetLastError`.

Profiling the first (KRvK-only) GPU version showed `step_kernel` accounts for ~98% of GPU time, mainly because every round revisits all 524,288 positions even though most are already solved (see [benchmarks.md](benchmarks.md)). The frontier solver below fixes most of that waste.

## General solver (any material)

`src/egtb/` generalizes everything above to any set of up to 4 pieces (5 supported by the data structures), with all six piece types including pawns and promotion.

### Material and canonical form

A **material** such as `KQvKR` lists the pieces in canonical order: white king, other white pieces strongest first (Q, R, B, N, P), then the same for Black. A position stores one square per piece in that order, and the row index is `[stm][sq0][sq1]...`, the same layout as the KRvK solver. The general KRvK table was therefore byte-for-byte identical to the original KRvK-only solver's.

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

### 4-piece results

From `warpmate-gen --gpu KRvKP` on a Tesla T4 (Google Colab), which solves all five. GPU times are for the frontier solver; the full scan took 2.26 to 3.68 s per table ([benchmarks.md](benchmarks.md)).

| Endgame | Rounds | GPU time | White to move (wins / draws / losses) | Longest forced mate |
|---|---|---|---|---|
| KQvKR | 71 | 1.42 s | 8,863,768 / 71,704 / 17,136 | 35 moves |
| KRvKR | 39 | 0.61 s | 3,139,232 / 7,569,032 / 72,464 | 19 moves |
| KRvKB | 59 | 0.56 s | 3,787,160 / 6,993,568 / 0 | 29 moves |
| KRvKN | 81 | 0.69 s | 5,210,920 / 5,569,800 / 8 | 40 moves |
| KRvKP | 86 | 0.92 s | 7,406,204 / 680,382 / 13,454 | 43 moves |

- The longest mates for KQvKR (35), KRvKR (19), KRvKB (29) and KRvKN (40) match the known results for these endgames.
- KRvKP's longest forced mate is a win for the side with the pawn (Black to move): the pawn promotes, and the queen then beats the rook.
- White's losses with the extra material are positions where Black wins material at once, for example by a fork or a skewer.
- A random sample of 1,000,000 rows from each table matches Syzygy with 0 mismatches ([verification.md](verification.md)).

## Frontier solver

The full scan wastes most of its work: in KQvKR it runs the per-position step 681 million times over 71 rounds to solve about 33 million positions, because every round re-checks every unsolved position, and most of them can't change. The frontier solver only checks positions that can.

### Which positions can change

A position still unsolved at round `ply` can only be decided then if one of its children has value exactly `ply − 1`:

- *odd ply (win):* it needs a child lost in at most `ply − 1`. A child lost sooner would already have made it a win two rounds earlier.
- *even ply (loss):* every child must be won in at most `ply − 1`. If the slowest child were faster, it would already have been lost two rounds earlier.

So each round only needs to check:

1. **Parents of last round's solved positions.** A child in the same table gets value `ply − 1` by being solved in round `ply − 1`. The solver keeps a list of those positions (the *frontier*) and walks backward from each one to its parents with **un-moves**.
2. **Positions whose captures or promotions matter this round.** Children in sub-tables have fixed values, so before round 1 each position records the round where they first count: one plus its fastest lost sub-table child (a win), and one plus its slowest won sub-table child if all of them are won (a loss). Other sub-table values can't change the result.

This gives exactly the same table as the full scan, with the same number of positions solved in every round; only the number of positions *checked* changes.

### Un-moves

`for_each_unmove` (`src/egtb/unmove.h`) lists every position one move before a given one: for each piece of the side that just moved, every square it could have come from. Sliding pieces use the same attack functions as forward moves (a rook could have come from any empty square it attacks now, because nothing between the two squares changed). Pawns un-push one square, or two from their starting rank. Captures and promotions are never reversed, since their parents belong to a different, larger table. Like everything else it is `WM_HD`.

### On the CPU and GPU

Each round has two steps:

| Step | CPU (`Solver`) | GPU (`GpuSolver`) |
|---|---|---|
| Mark parents of last round's solved positions | threads split the frontier list; marks are relaxed atomic bytes | `mark_parents_kernel`: one thread per frontier position; threads that mark the same parent write the same byte, so no atomic is needed |
| Check marked positions | threads scan the table; results are collected first and written after, so no thread reads a value written this round | `frontier_step_kernel`: one thread per position, but only marked ones run the per-position step; newly solved positions get a slot in the next frontier list from `atomicAdd` |

GPU details:

- **`prepare_kernel`** runs once before round 1: it records the capture/promotion rounds for every position and lists the checkmates as the first frontier.
- **Counting with one atomic per block.** The number of positions checked per round (a statistic) is counted with `__syncthreads_count`, a barrier that also returns how many threads in the block passed a condition, so only one thread per block does an `atomicAdd`.
- **Memory:** the frontier list (4 bytes per position) and three bytes of marks per position stay on the GPU; about 235 MB for a 4-piece table.

### Results

On a Tesla T4, the GPU frontier solver is 2.7× to 6.1× faster than the full scan on the 4-piece endgames, and solves all five in 4.2 s instead of 15.7 s, with identical tables ([benchmarks.md](benchmarks.md)). The remaining cost is mostly the round kernel, which still launches one thread per position to find the marked ones.

## Correctness checks

The test suite (`tests/test_egtb.cpp`) checks:

- king, rook, knight, bishop, queen and pawn attacks, including board edges
- position encoding round trips, and which positions count as invalid
- move generation: checkmate, stalemate, a forced rook capture, pawn pushes, promotions (four choices) and pinned pieces
- **un-moves are the exact reverse of moves**: every non-capturing move appears among the destination's un-moves, and every un-move is a real move, for every position of all five 3-piece endgames and every 97th position of five 4-piece endgames (including pawns on both sides)
- FEN and move notation, material parsing, canonical form and dependency lists
- known results: KQvK longest mate 10 moves, KRvK 16, KBvK and KNvK all draws, and the exact KRvK win/draw/loss counts below
- known positions: checkmate, mate in 1, stalemate, an undefended rook, kings touching, and colour-flipped lookups giving the same value
- the frontier solver produces the same tables, rounds and per-round solved counts as the full scan, for all five 3-piece endgames
- 1, 3 and 8 CPU threads produce identical tables
- tables survive a save/load round trip
- **every entry is consistent with its children** in all five 3-piece tables, including children in other tables: a win in *n* has a child lost in exactly *n* − 1, a loss in *n* has only won children with the slowest at *n* − 1, and a draw has no lost child and at least one non-won child

With CUDA, `tests/test_egtb_gpu.cpp` checks that both GPU algorithms produce tables byte-for-byte identical to the CPU's for all five 3-piece endgames, with the same positions solved in every round, and that the GPU frontier checks exactly the same positions as the CPU frontier. `warpmate-bench` compares every algorithm's table on the 4-piece endgames.

An independent check against the Syzygy tablebases is provided by `scripts/verify_syzygy.py` ([verification.md](verification.md)).

## KRvK results

The solved KRvK table:

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
