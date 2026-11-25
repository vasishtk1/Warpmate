#!/usr/bin/env python3
"""Check Warpmate tables against the official Syzygy tablebases.

For every row of each table, this compares:
  * validity: rows Warpmate marks invalid must be impossible positions
  * result:   Warpmate's win/draw/loss must equal Syzygy's WDL value

Syzygy stores win/draw/loss (WDL) and distance-to-zeroing (DTZ), not
distance-to-mate, so mate distances are checked separately by the C++ tests.
Syzygy's "cursed win" / "blessed loss" (decided only by the 50-move rule)
count as wins / losses here, since Warpmate ignores the 50-move rule.

Setup:
    pip install -r scripts/requirements.txt
    mkdir -p syzygy
    # download the matching .rtbw files into syzygy/, e.g. KRvK.rtbw from
    # https://tablebase.lichess.ovh/tables/standard/3-4-5-wdl/

Usage:
    python scripts/verify_syzygy.py --syzygy syzygy tables/KRvK.wmtb
    python scripts/verify_syzygy.py --syzygy syzygy tables/*.wmtb --report docs/verification.md
    python scripts/verify_syzygy.py --syzygy syzygy tables/KQvKR.wmtb --sample 1000000

--sample N checks N randomly chosen rows per table (fixed seed, so runs are
repeatable) instead of every row. Python checks roughly 30,000 rows per
second, so a full 4-piece table (33.5 million rows) takes 15+ minutes.
"""

from __future__ import annotations

import argparse
import datetime
import random
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import chess
import chess.syzygy

# Must match src/core/values.h
DRAW, UNKNOWN, INVALID = 253, 254, 255

# Must match src/io/tablebase.h
HEADER = struct.Struct("<4sI16sQ")
MAGIC = b"WMTB"

PIECE_TYPES = {"K": chess.KING, "Q": chess.QUEEN, "R": chess.ROOK,
               "B": chess.BISHOP, "N": chess.KNIGHT, "P": chess.PAWN}


def load_table(path: Path) -> tuple[str, bytes]:
    data = path.read_bytes()
    magic, version, name, count = HEADER.unpack_from(data)
    if magic != MAGIC:
        sys.exit(f"{path} is not a Warpmate tablebase")
    if version != 1:
        sys.exit(f"unsupported tablebase version {version}")
    name = name.rstrip(b"\0").decode()
    values = data[HEADER.size:]
    if len(values) != count:
        sys.exit(f"{path} is truncated")
    return name, values


def pieces_of(name: str) -> list[chess.Piece]:
    """'KQvKR' -> pieces in the table's square order (matches egtb/material.h)."""
    white, black = name.split("v")
    return ([chess.Piece(PIECE_TYPES[c], chess.WHITE) for c in white] +
            [chess.Piece(PIECE_TYPES[c], chess.BLACK) for c in black])


def decode(index: int, n: int) -> tuple[int, list[int]]:
    """Row number -> (side to move, squares). Mirrors egtb/position.h."""
    squares = []
    for _ in range(n):
        squares.append(index & 63)
        index >>= 6
    return index & 1, squares[::-1]


def warpmate_wdl(value: int) -> int:
    """Warpmate value -> -1 (loss), 0 (draw), 1 (win) for the side to move."""
    if value >= DRAW:
        return 0
    return 1 if value % 2 == 1 else -1


def sign(x: int) -> int:
    return (x > 0) - (x < 0)


@dataclass
class Result:
    name: str
    rows: int = 0
    checked_rows: int = 0
    sampled: bool = False
    compared: int = 0
    invalid_ok: int = 0
    overlapping: int = 0
    mismatches: list[str] = field(default_factory=list)
    seconds: float = 0.0


def verify(path: Path, tablebase: chess.syzygy.Tablebase, sample: int | None) -> Result:
    name, values = load_table(path)
    pieces = pieces_of(name)
    n = len(pieces)
    if len(values) != 2 * 64 ** n:
        sys.exit(f"{path}: size doesn't match {name}")
    result = Result(name=name, rows=len(values))
    start = time.time()

    if sample is not None and sample < len(values):
        indices = sorted(random.Random(12345).sample(range(len(values)), sample))
        result.sampled = True
    else:
        indices = range(len(values))
    result.checked_rows = len(indices)

    for index in indices:
        value = values[index]
        stm, squares = decode(index, n)
        if len(set(squares)) < n:
            result.overlapping += 1
            if value != INVALID:
                result.mismatches.append(f"index {index}: overlapping pieces not marked invalid")
            continue

        board = chess.Board(None)
        for piece, square in zip(pieces, squares):
            board.set_piece_at(square, piece)
        board.turn = chess.WHITE if stm == 0 else chess.BLACK

        if not board.is_valid():
            if value == INVALID:
                result.invalid_ok += 1
            else:
                result.mismatches.append(f"{board.fen()}: impossible position, "
                                         f"Warpmate value {value}")
            continue
        if value == INVALID:
            result.mismatches.append(f"{board.fen()}: legal position marked invalid")
            continue

        expected = sign(tablebase.probe_wdl(board))
        got = warpmate_wdl(value)
        result.compared += 1
        if got != expected:
            result.mismatches.append(f"{board.fen()}: Syzygy {expected:+d}, "
                                     f"Warpmate {got:+d} (value {value})")

    result.seconds = time.time() - start
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tables", type=Path, nargs="+", help="Warpmate .wmtb files")
    parser.add_argument("--syzygy", type=Path, required=True,
                        help="directory containing the matching .rtbw files")
    parser.add_argument("--report", type=Path,
                        help="write a Markdown summary here (e.g. docs/verification.md)")
    parser.add_argument("--sample", type=int,
                        help="check this many random rows per table instead of all rows")
    args = parser.parse_args()

    results = []
    with chess.syzygy.open_tablebase(str(args.syzygy)) as tablebase:
        for path in args.tables:
            r = verify(path, tablebase, args.sample)
            results.append(r)
            status = "PASS" if not r.mismatches else "FAIL"
            scope = f"random sample of {r.checked_rows:,} rows" if r.sampled else "all rows"
            print(f"{r.name:7} {status}  {scope}: compared {r.compared:,} legal positions, "
                  f"{len(r.mismatches):,} mismatches  ({r.seconds:.1f} s)")
            for line in r.mismatches[:10]:
                print(f"    {line}")

    passed = all(not r.mismatches for r in results)
    total = sum(r.compared for r in results)
    print(f"\nTotal legal positions compared: {total:,}")
    print("PASS" if passed else "FAIL")

    if args.report:
        lines = [
            "# Verification against Syzygy",
            "",
            f"Generated by `scripts/verify_syzygy.py` on {datetime.date.today().isoformat()}.",
            "",
            "For legal positions, Warpmate's win/draw/loss result was compared with",
            "the Syzygy WDL tables (via python-chess); every row Warpmate marks",
            "invalid was confirmed to be an impossible position. \"All\" means every",
            "row of the table was checked; otherwise a random sample of rows (fixed",
            "seed) was checked.",
            "",
            "| Endgame | Rows checked | Legal positions compared | Invalid rows confirmed | Overlapping arrangements | Mismatches |",
            "|---|---|---|---|---|---|",
        ]
        for r in results:
            scope = f"sample of {r.checked_rows:,}" if r.sampled else f"all {r.rows:,}"
            lines.append(f"| {r.name} | {scope} | {r.compared:,} | {r.invalid_ok:,} | "
                         f"{r.overlapping:,} | **{len(r.mismatches):,}** |")
        lines += [
            f"| **Total** | | **{total:,}** | | | **{sum(len(r.mismatches) for r in results):,}** |",
            "",
            f"**Result: {'PASS' if passed else 'FAIL'}**",
        ]
        for r in results:
            if r.mismatches:
                lines += ["", f"First mismatches in {r.name}:", ""]
                lines += [f"- `{m}`" for m in r.mismatches[:20]]
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text("\n".join(lines) + "\n")
        print(f"Report written to {args.report}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
