"""Reading Warpmate tables from Python, and looking positions up in them.

Mirrors the C++ code exactly (src/egtb/material.h, position.h, io/tablebase.h):
  * a table stores one byte per row; row = [side to move][sq0][sq1]...,
    six bits per square, pieces in canonical order
  * canonical order: White's pieces then Black's, each king first, then
    queen, rook, bishop, knight, pawn
  * each endgame is stored with the stronger side as White; a position where
    Black is stronger is looked up with colours swapped and the board
    mirrored top to bottom (square ^ 56), which keeps the value unchanged
    because values are from the side to move's point of view
"""

from __future__ import annotations

import struct
from pathlib import Path

import chess
import numpy as np

# Must match src/core/values.h
DRAW, UNKNOWN, INVALID = 253, 254, 255

# Must match src/io/tablebase.h
HEADER = struct.Struct("<4sI16sQ")
MAGIC = b"WMTB"

# Must match PieceType in src/egtb/material.h (king = 0 ... pawn = 5).
LETTERS = "KQRBNP"
CHESS_TYPE = [chess.KING, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]
TYPE_OF = {t: i for i, t in enumerate(CHESS_TYPE)}  # python-chess type -> ours
WHITE, BLACK = 0, 1


def load_table(path: Path) -> tuple[str, np.ndarray]:
    """Returns (material name, values) without reading the file into memory."""
    with open(path, "rb") as f:
        magic, version, name, count = HEADER.unpack(f.read(HEADER.size))
    if magic != MAGIC or version != 1:
        raise ValueError(f"{path} is not a Warpmate tablebase (version 1)")
    values = np.memmap(path, dtype=np.uint8, mode="r", offset=HEADER.size, shape=(count,))
    return name.rstrip(b"\0").decode(), values


def pieces_of(name: str) -> list[tuple[int, int]]:
    """'KQvKR' -> [(type, colour), ...] in the order the name lists them."""
    white, black = name.split("v")
    return ([(LETTERS.index(c), WHITE) for c in white] +
            [(LETTERS.index(c), BLACK) for c in black])


def piece_key(piece: tuple[int, int]) -> int:
    """Canonical sort key, as in material.h: colour first, then type."""
    t, c = piece
    return c * 8 + t


def black_is_stronger(pieces: list[tuple[int, int]]) -> bool:
    """Same rule as black_is_stronger in material.h (pieces in canonical order)."""
    w = [t for t, c in pieces if c == WHITE and t != 0]
    b = [t for t, c in pieces if c == BLACK and t != 0]
    if len(w) != len(b):
        return len(b) > len(w)
    for tw, tb in zip(w, b):
        if tw != tb:
            return tb < tw
    return False


def material_name(pieces: list[tuple[int, int]]) -> str:
    white = "".join(LETTERS[t] for t, c in pieces if c == WHITE)
    black = "".join(LETTERS[t] for t, c in pieces if c == BLACK)
    return f"{white}v{black}"


def canonical(board: chess.Board) -> tuple[str, list[int], list[tuple[int, int]], int]:
    """A python-chess board in the form the tables use.

    Returns (material name, squares, pieces, side to move) with the stronger
    side as White, exactly like lookup_pieces in position.h. Sorting is stable,
    so two identical pieces keep their relative order (both orders are stored
    in the tables anyway).
    """
    entries = []
    for sq, piece in board.piece_map().items():
        colour = WHITE if piece.color == chess.WHITE else BLACK
        entries.append((TYPE_OF[piece.piece_type], colour, sq))
    stm = WHITE if board.turn == chess.WHITE else BLACK
    entries.sort(key=lambda e: piece_key((e[0], e[1])))
    pieces = [(t, c) for t, c, _ in entries]
    if black_is_stronger(pieces):
        entries = [(t, c ^ 1, sq ^ 56) for t, c, sq in entries]
        stm ^= 1
        entries.sort(key=lambda e: piece_key((e[0], e[1])))
        pieces = [(t, c) for t, c, _ in entries]
    return material_name(pieces), [sq for _, _, sq in entries], pieces, stm


def encode(squares: list[int], stm: int) -> int:
    index = stm
    for sq in squares:
        index = index * 64 + sq
    return index


def wdl(value: int) -> int:
    """Stored value -> -1 (side to move loses), 0 (draw), +1 (wins)."""
    if value >= DRAW:
        return 0
    return 1 if value % 2 == 1 else -1


class TableSet:
    """All solved tables in a directory, for looking up any position."""

    def __init__(self, directory: Path):
        self.tables: dict[str, np.ndarray] = {}
        for path in sorted(Path(directory).glob("*.wmtb")):
            name, values = load_table(path)
            self.tables[name] = values

    def value(self, board: chess.Board) -> int:
        """Stored value of the position (from the side to move's view)."""
        if len(board.piece_map()) == 2:
            return DRAW  # king vs king
        name, squares, _, stm = canonical(board)
        if name not in self.tables:
            raise KeyError(f"no table for {name}")
        return int(self.tables[name][encode(squares, stm)])
