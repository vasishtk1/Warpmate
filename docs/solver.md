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

## Correctness checks

The test suite (`tests/test_solver.cpp`) checks:

- known positions: checkmate, mate in 1, stalemate, a forced rook capture, an invalid position
- the longest forced mate is **16 moves** (31 plies), the known result for KRvK
- **every entry is consistent with its children**: a win in *n* has a child lost in exactly *n* − 1, a loss in *n* has only won children with the slowest at *n* − 1, and a draw has no lost child and at least one non-won child
- tables survive a save/load round trip

An independent check against the Syzygy tablebases is provided by `scripts/verify_syzygy.py`.

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
