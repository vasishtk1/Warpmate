#!/usr/bin/env python3
"""Builds the training dataset from solved Warpmate tables.

Each legal position becomes one example:
  * board: up to 5 "piece codes", one per piece: (colour*6 + type)*64 + square,
    padded with PAD. This is a compact form of the network's 769 inputs
    (12 piece kinds x 64 squares, plus side to move), which are built from
    the codes on the GPU while training (see model.py).
  * label: 0 = side to move loses, 1 = draw, 2 = side to move wins
  * value: the stored distance-to-mate byte, kept for later analysis

Positions with two identical pieces (e.g. KQQvK) appear twice in a table,
once per order of the two pieces; only one copy is kept, so the same
position can't land in both the training and the test set.

Usage:
    python ml/dataset.py --tables tables --out data/dataset.npz
    python ml/dataset.py --tables tables --out data/dataset.npz --per-table 1000000 \
        --syzygy syzygy --check 2000

--check N looks up N random examples per table in the Syzygy tablebases and
confirms every label (needs python-chess and the .rtbw files).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tables import DRAW, INVALID, load_table, pieces_of  # noqa: E402

PAD = 769          # piece-code slot for "no piece" (see model.py)
MAX_PIECES = 5
TEST_FRACTION = 0.1


def table_examples(name: str, values: np.ndarray, per_table: int, rng: np.random.Generator):
    """Legal, de-duplicated positions of one table, sampled. Returns a dict."""
    pieces = pieces_of(name)
    n = len(pieces)
    if len(values) != 2 * 64 ** n:
        raise ValueError(f"{name}: table size doesn't match the material")

    rows = np.flatnonzero(np.asarray(values) != INVALID)  # legal positions
    legal = len(rows)
    squares = [((rows >> (6 * (n - 1 - k))) & 63).astype(np.uint8) for k in range(n)]

    # Identical pieces sit next to each other in canonical order; keep only
    # the row where their squares are in increasing order.
    keep = np.ones(len(rows), dtype=bool)
    duplicated = False
    for k in range(n - 1):
        if pieces[k] == pieces[k + 1]:
            keep &= squares[k] < squares[k + 1]
            duplicated = True
    rows = rows[keep]
    squares = [s[keep] for s in squares]
    unique = len(rows)
    if duplicated and unique * 2 != legal:
        raise AssertionError(f"{name}: expected every position exactly twice")

    # Sample without replacement (or take everything for small tables).
    if per_table and unique > per_table:
        pick = np.sort(rng.choice(unique, size=per_table, replace=False))
        rows = rows[pick]
        squares = [s[pick] for s in squares]

    count = len(rows)
    codes = np.full((count, MAX_PIECES), PAD, dtype=np.int16)
    for k, (t, c) in enumerate(pieces):
        codes[:, k] = (c * 6 + t) * 64 + squares[k].astype(np.int16)
    stm = (rows >> (6 * n)).astype(np.uint8)
    value = np.asarray(values)[rows]
    label = np.where(value == DRAW, 1, np.where(value % 2 == 1, 2, 0)).astype(np.uint8)
    return {"codes": codes, "stm": stm, "value": value, "label": label,
            "legal": legal, "unique": unique}


def syzygy_check(name: str, ex: dict, syzygy: Path, count: int, rng: np.random.Generator) -> int:
    """Confirms `count` random labels against Syzygy. Returns the mismatch count."""
    import chess
    import chess.syzygy
    from tables import CHESS_TYPE

    picks = rng.choice(len(ex["label"]), size=min(count, len(ex["label"])), replace=False)
    bad = 0
    with chess.syzygy.open_tablebase(str(syzygy)) as tb:
        for i in picks:
            board = chess.Board(None)
            for code in ex["codes"][i]:
                if code == PAD:
                    continue
                kind, sq = divmod(int(code), 64)
                colour, t = divmod(kind, 6)
                board.set_piece_at(sq, chess.Piece(CHESS_TYPE[t], colour == 0))
            board.turn = chess.WHITE if ex["stm"][i] == 0 else chess.BLACK
            expected = int(np.sign(tb.probe_wdl(board))) + 1  # -1/0/1 -> 0/1/2
            if expected != ex["label"][i]:
                bad += 1
                if bad <= 3:
                    print(f"    mismatch: {board.fen()} Syzygy {expected}, label {ex['label'][i]}")
    return bad


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tables", type=Path, required=True, help="directory of .wmtb files")
    parser.add_argument("--out", type=Path, required=True, help="output .npz file")
    parser.add_argument("--per-table", type=int, default=1_000_000,
                        help="examples sampled per table (0 = all)")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--syzygy", type=Path, help="Syzygy directory for --check")
    parser.add_argument("--check", type=int, default=0,
                        help="confirm this many labels per table against Syzygy")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    paths = sorted(args.tables.glob("*.wmtb"))
    if not paths:
        sys.exit(f"no .wmtb files in {args.tables}")

    parts, names = [], []
    total_bad = 0
    print(f"{'Table':7} {'legal':>12} {'unique':>12} {'sampled':>10}   loss / draw / win")
    for path in paths:
        name, values = load_table(path)
        start = time.time()
        ex = table_examples(name, values, args.per_table, rng)
        ex["table_id"] = np.full(len(ex["label"]), len(names), dtype=np.uint8)
        names.append(name)
        counts = np.bincount(ex["label"], minlength=3) / len(ex["label"]) * 100
        print(f"{name:7} {ex['legal']:12,} {ex['unique']:12,} {len(ex['label']):10,}   "
              f"{counts[0]:4.1f}% / {counts[1]:4.1f}% / {counts[2]:4.1f}%  "
              f"({time.time() - start:.1f} s)")
        if args.check and args.syzygy:
            bad = syzygy_check(name, ex, args.syzygy, args.check, rng)
            total_bad += bad
            print(f"        Syzygy check of {min(args.check, len(ex['label'])):,} labels: "
                  f"{bad} mismatches")
        parts.append(ex)

    data = {key: np.concatenate([p[key] for p in parts])
            for key in ("codes", "stm", "value", "label", "table_id")}
    # Random 90/10 split. Every example is a different position, so no
    # position is in both sets.
    is_test = rng.random(len(data["label"])) < TEST_FRACTION
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **data, is_test=is_test, names=np.array(names))

    total = len(data["label"])
    print(f"\n{total:,} examples from {len(names)} tables: "
          f"{int((~is_test).sum()):,} train, {int(is_test.sum()):,} test")
    counts = np.bincount(data["label"], minlength=3) / total * 100
    print(f"Overall: {counts[0]:.1f}% loss, {counts[1]:.1f}% draw, {counts[2]:.1f}% win "
          f"(for the side to move)")
    if args.check and args.syzygy:
        print(f"Syzygy label check: {'PASS' if total_bad == 0 else 'FAIL'} "
              f"({total_bad} mismatches)")
    print(f"Saved {args.out}")
    return 0 if total_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
