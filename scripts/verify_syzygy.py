#!/usr/bin/env python3
"""Check a Warpmate KRvK table against the official Syzygy tablebases.

For every arrangement of the pieces, this compares:
  * validity: positions Warpmate marks invalid must be invalid in python-chess
  * result:   Warpmate's win/draw/loss must equal Syzygy's WDL value

Syzygy stores win/draw/loss (WDL) and distance-to-zeroing (DTZ), not
distance-to-mate, so mate distances are checked separately by the C++ tests
(longest forced mate = 16 moves).

Setup:
    pip install -r scripts/requirements.txt
    mkdir -p syzygy
    # download KRvK.rtbw into syzygy/, e.g. from
    # https://tablebase.lichess.ovh/tables/standard/3-4-5-wdl/

Usage:
    python scripts/verify_syzygy.py --syzygy syzygy
    python scripts/verify_syzygy.py --syzygy syzygy --report docs/verification.md
"""

from __future__ import annotations

import argparse
import datetime
import struct
import sys
import time
from pathlib import Path

import chess
import chess.syzygy

# Must match src/core/values.h
DRAW, UNKNOWN, INVALID = 253, 254, 255

# Must match src/io/tablebase.h
HEADER = struct.Struct("<4sI16sQ")
MAGIC = b"WMTB"
NUM_POSITIONS = 2 * 64 * 64 * 64


def load_table(path: Path) -> bytes:
    data = path.read_bytes()
    magic, version, name, count = HEADER.unpack_from(data)
    if magic != MAGIC:
        sys.exit(f"{path} is not a Warpmate tablebase")
    if version != 1:
        sys.exit(f"unsupported tablebase version {version}")
    name = name.rstrip(b"\0").decode()
    if name != "KRvK" or count != NUM_POSITIONS:
        sys.exit(f"{path} is not a KRvK table (name={name!r}, count={count})")
    values = data[HEADER.size:]
    if len(values) != count:
        sys.exit(f"{path} is truncated")
    return values


def decode(index: int) -> tuple[int, int, int, int]:
    """Table index -> (stm, white king, white rook, black king). Mirrors krk.h."""
    return (index >> 18) & 1, (index >> 12) & 63, (index >> 6) & 63, index & 63


def to_board(stm: int, wk: int, wr: int, bk: int) -> chess.Board:
    board = chess.Board(None)
    board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(wr, chess.Piece(chess.ROOK, chess.WHITE))
    board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
    board.turn = chess.WHITE if stm == 0 else chess.BLACK
    return board


def warpmate_wdl(value: int) -> int:
    """Warpmate value -> -1 (loss), 0 (draw), 1 (win) for the side to move."""
    if value >= DRAW:
        return 0
    return 1 if value % 2 == 1 else -1


def sign(x: int) -> int:
    return (x > 0) - (x < 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--table", type=Path, default=Path("tables/krvk.wmtb"))
    parser.add_argument("--syzygy", type=Path, required=True,
                        help="directory containing KRvK.rtbw")
    parser.add_argument("--report", type=Path,
                        help="write a Markdown summary here (e.g. docs/verification.md)")
    args = parser.parse_args()

    values = load_table(args.table)
    if not (args.syzygy / "KRvK.rtbw").exists():
        sys.exit(f"KRvK.rtbw not found in {args.syzygy}")

    checked = skipped_overlap = invalid_ok = 0
    mismatches: list[str] = []
    start = time.time()

    with chess.syzygy.open_tablebase(str(args.syzygy)) as tablebase:
        for index in range(NUM_POSITIONS):
            stm, wk, wr, bk = decode(index)
            value = values[index]

            if len({wk, wr, bk}) < 3:
                # Pieces on the same square can't be put on a board at all.
                skipped_overlap += 1
                if value != INVALID:
                    mismatches.append(f"index {index}: overlapping pieces not marked invalid")
                continue

            board = to_board(stm, wk, wr, bk)
            if not board.is_valid():
                if value == INVALID:
                    invalid_ok += 1
                else:
                    mismatches.append(f"{board.fen()}: invalid in python-chess, "
                                      f"Warpmate value {value}")
                continue
            if value == INVALID:
                mismatches.append(f"{board.fen()}: valid in python-chess, Warpmate says invalid")
                continue

            expected = sign(tablebase.probe_wdl(board))
            got = warpmate_wdl(value)
            checked += 1
            if got != expected:
                mismatches.append(f"{board.fen()}: Syzygy WDL {expected:+d}, Warpmate {got:+d} "
                                  f"(value {value})")

            if index % 65536 == 0:
                print(f"  {index / NUM_POSITIONS:5.1%} checked", file=sys.stderr)

    elapsed = time.time() - start
    passed = not mismatches

    print(f"Valid positions compared with Syzygy: {checked:,}")
    print(f"Invalid positions confirmed by python-chess: {invalid_ok:,}")
    print(f"Overlapping arrangements skipped: {skipped_overlap:,}")
    print(f"Mismatches: {len(mismatches):,}")
    for line in mismatches[:20]:
        print(f"  {line}")
    print(f"Time: {elapsed:.1f} s")
    print("PASS" if passed else "FAIL")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Verification against Syzygy",
            "",
            f"Generated by `scripts/verify_syzygy.py` on {datetime.date.today().isoformat()}.",
            "",
            "Every arrangement of the KRvK pieces was checked: Warpmate's win/draw/loss",
            "result was compared with the Syzygy WDL tables (via python-chess), and every",
            "position Warpmate marks invalid was confirmed invalid by python-chess.",
            "",
            "| Check | Count |",
            "|---|---|",
            f"| Valid positions compared with Syzygy | {checked:,} |",
            f"| Invalid positions confirmed | {invalid_ok:,} |",
            f"| Overlapping arrangements (impossible) | {skipped_overlap:,} |",
            f"| **Mismatches** | **{len(mismatches):,}** |",
            "",
            f"**Result: {'PASS' if passed else 'FAIL'}**",
        ]
        if mismatches:
            lines += ["", "First mismatches:", ""] + [f"- `{m}`" for m in mismatches[:20]]
        args.report.write_text("\n".join(lines) + "\n")
        print(f"Report written to {args.report}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
