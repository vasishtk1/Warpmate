"""Position analysis for the dashboard: the network's guess, the perfect
answer from the solved tables, and every legal move ranked.

Kept separate from the web server (app.py) so it can be tested directly.
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import chess
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))
from tables import DRAW, INVALID, TableSet, black_is_stronger, canonical, wdl  # noqa: E402

PAD = 769  # must match ml/model.py


# --------------------------------------------------------------------------
# Network


class Network:
    """The trained network, or a clearly marked placeholder if none is found.

    TEMPORARY (see notes/DASHBOARD_TODO.md): runs in PyTorch on the CPU.
    Phase 4 replaces this with the custom CUDA inference kernels.
    """

    def __init__(self, path: Path):
        self.path = path
        self.model = None
        self.placeholder = True
        if path.exists():
            import torch
            from model import load
            self.torch = torch
            self.model = load(str(path), "cpu")
            self.placeholder = False

    def predict(self, boards: list[chess.Board]) -> np.ndarray:
        """(N, 3) probabilities of loss / draw / win for each side to move."""
        if self.placeholder:
            # TEMPORARY placeholder: no trained weights found, so every
            # position gets the same made-up split. The UI labels it.
            return np.tile(np.array([1 / 3, 1 / 3, 1 / 3]), (len(boards), 1))
        width = max(len(b.piece_map()) for b in boards)
        codes = np.full((len(boards), width), PAD, dtype=np.int64)
        stm = np.zeros(len(boards), dtype=np.int64)
        for i, board in enumerate(boards):
            c, s = encode(board)
            codes[i, :len(c)] = c
            stm[i] = s
        from model import features
        with self.torch.no_grad():
            x = features(self.torch.from_numpy(codes), self.torch.from_numpy(stm))
            return self.torch.softmax(self.model(x), 1).numpy()


def encode(board: chess.Board) -> tuple[list[int], int]:
    """Board -> (piece codes, side to move) in the form the network was
    trained on: stronger side as White. Works for any number of pieces."""
    _, squares, pieces, stm = canonical(board)
    return [(c * 6 + t) * 64 + sq for (t, c), sq in zip(pieces, squares)], stm


# --------------------------------------------------------------------------
# Perfect answers


@dataclass
class Perfect:
    result: int     # +1 win, 0 draw, -1 loss, for the side to move
    mate_in: int    # moves to mate (0 for a draw)

    def text(self, side: str) -> str:
        other = "Black" if side == "White" else "White"
        if self.result == 0:
            return "Draw"
        if self.result > 0:
            return f"{side} mates in {self.mate_in}"
        return f"{other} mates in {self.mate_in}" if self.mate_in else f"{side} is checkmated"


def perfect_from_value(value: int) -> Perfect | None:
    if value == INVALID:
        return None
    if value >= DRAW:
        return Perfect(0, 0)
    return Perfect(wdl(value), (value + 1) // 2)


class Analyzer:
    def __init__(self, tables_dir: Path, model_path: Path):
        self.tables = TableSet(tables_dir)
        self.network = Network(model_path)

    # ---- perfect answers ----

    def perfect(self, board: chess.Board) -> tuple[Perfect | None, str]:
        """Perfect answer, or None with the reason it isn't available."""
        if board.ep_square is not None and board.has_legal_en_passant():
            return None, "en passant is possible here, which the tables don't cover"
        if board.castling_rights:
            return None, "castling is possible here, which the tables don't cover"
        if len(board.piece_map()) == 2:
            return Perfect(0, 0), ""
        name, _, _, _ = canonical(board)
        if len(board.piece_map()) > 4:
            return None, f"{len(board.piece_map())} pieces: tables cover up to 4"
        if name not in self.tables.tables:
            if name == "KPvKP":
                return None, "KPvKP isn't solved yet (needs en passant)"
            return None, f"the {name} table isn't on this computer"
        result = perfect_from_value(self.tables.value(board))
        if result is None:
            return None, "the table marks this position as impossible"
        return result, ""

    def move_perfect(self, board: chess.Board, move: chess.Move) -> Perfect | None:
        """Perfect result of `move` for the player making it."""
        board.push(move)
        try:
            child, _ = self.perfect(board)
        finally:
            board.pop()
        if child is None:
            return None
        if child.result == 0:
            return Perfect(0, 0)
        # The child's value is from the opponent's point of view: if they're
        # lost in n moves, the mover mates in n + 1 (it took this move).
        if child.result < 0:
            return Perfect(1, child.mate_in + 1)
        return Perfect(-1, child.mate_in)

    # ---- the full analysis the page shows ----

    def analyze(self, fen: str) -> dict:
        try:
            board = chess.Board(fen)
        except ValueError as e:
            return {"valid": False, "reason": f"bad position: {e}"}
        reason = invalid_reason(board)
        if reason:
            return {"valid": False, "reason": reason}

        side = "White" if board.turn == chess.WHITE else "Black"
        pieces = len(board.piece_map())
        moves = list(board.legal_moves)
        children = []
        for mv in moves:
            board.push(mv)
            children.append(board.copy(stack=False))
            board.pop()
        probs = self.network.predict([board] + children)

        net = to_colors(probs[0], board.turn)
        perfect, why = self.perfect(board)

        move_rows = []
        for k, mv in enumerate(moves):
            p = self.move_perfect(board, mv) if perfect is not None else None
            child_probs = probs[k + 1]  # from the opponent's point of view
            mover = {"win": float(child_probs[0]), "draw": float(child_probs[1]),
                     "loss": float(child_probs[2])}
            move_rows.append({
                "uci": mv.uci(),
                "san": board.san(mv),
                "perfect": None if p is None else {
                    "result": p.result, "mate_in": p.mate_in, "text": move_text(p)},
                "network": mover,
            })
        move_rows.sort(key=lambda r: rank_key(r), reverse=True)
        if move_rows:
            top = rank_key(move_rows[0])
            for r in move_rows:
                r["best"] = rank_key(r) == top

        return {
            "valid": True,
            "fen": board.fen(),
            "turn": "w" if board.turn == chess.WHITE else "b",
            "pieces": pieces,
            "network": {**net, "placeholder": self.network.placeholder,
                        "trained_range": pieces <= 4},
            "perfect": ({"available": True, "result": perfect.result, "mate_in": perfect.mate_in,
                         "text": perfect.text(side)}
                        if perfect is not None else {"available": False, "why": why}),
            "ranked_by": "perfect" if perfect is not None else "network",
            "moves": move_rows,
            "game_over": game_over(board),
        }

    def model_info(self, fen: str | None = None) -> dict:
        """The network's architecture, its training record (if the files from
        the Phase 3 notebook are next to the weights), and which of the 769
        inputs are switched on for `fen`."""
        net = self.network
        hidden = list(net.model.hidden) if net.model is not None else [256, 64]
        sizes = [769, *hidden, 3]
        layers = []
        for k in range(len(sizes) - 1):
            layers.append({"in": sizes[k], "out": sizes[k + 1],
                           "params": sizes[k] * sizes[k + 1] + sizes[k + 1],
                           "activation": "ReLU" if k < len(sizes) - 2 else "softmax"})
        info = {
            "loaded": not net.placeholder,
            "sizes": sizes,
            "layers": layers,
            "total_params": sum(l["params"] for l in layers),
            "training": read_json(net.path.parent / "training_log.json"),
            "results": read_json(net.path.parent / "results.json"),
            "input": None,
        }
        if fen:
            try:
                board = chess.Board(fen)
            except ValueError:
                return info
            if invalid_reason(board) is None:
                codes, stm = encode(board)
                # Canonical order (colour, then type), as black_is_stronger expects.
                pieces = sorted((piece_key_of(p) for p in board.piece_map().values()),
                                key=lambda tc: tc[1] * 8 + tc[0])
                flipped = black_is_stronger(pieces)
                info["input"] = {"active": sorted(codes + ([768] if stm == 0 else [])),
                                 "flipped": flipped}
        return info

    def random_position(self, material: str | None, rng: random.Random) -> str:
        """A random legal position for `material` (e.g. 'KQvKR'), or for a
        random endgame we have a table for."""
        names = sorted(n for n in self.tables.tables)
        if material is None:
            material = rng.choice(names) if names else "KRvK"
        white, black = material.split("v")
        letters = [(c, chess.WHITE) for c in white] + [(c, chess.BLACK) for c in black]
        for _ in range(10_000):
            board = chess.Board(None)
            squares = rng.sample(range(64), len(letters))
            for (letter, colour), sq in zip(letters, squares):
                board.set_piece_at(sq, chess.Piece(chess.PIECE_SYMBOLS.index(letter.lower()),
                                                   colour))
            board.turn = rng.random() < 0.5
            if board.is_valid() and not board.is_game_over():
                return board.fen()
        raise ValueError(f"couldn't place {material}")


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def piece_key_of(piece: chess.Piece) -> tuple[int, int]:
    """python-chess piece -> (type, colour) in the tables' numbering."""
    from tables import TYPE_OF
    return (TYPE_OF[piece.piece_type], 0 if piece.color == chess.WHITE else 1)


def to_colors(p: np.ndarray, turn: bool) -> dict:
    """loss/draw/win for the side to move -> white/draw/black."""
    loss, draw, win = (float(x) for x in p)
    if turn == chess.WHITE:
        return {"white": win, "draw": draw, "black": loss}
    return {"white": loss, "draw": draw, "black": win}


def move_text(p: Perfect) -> str:
    if p.result == 0:
        return "draw"
    if p.result > 0:
        return f"mate in {p.mate_in}"
    return f"mated in {p.mate_in}" if p.mate_in else "checkmated"


def rank_key(row: dict) -> float:
    """Higher is better for the player to move."""
    p = row["perfect"]
    if p is not None:
        if p["result"] > 0:
            return 1000 - p["mate_in"]      # win: faster is better
        if p["result"] < 0:
            return -1000 + p["mate_in"]     # loss: slower is better
        return 0
    n = row["network"]
    return n["win"] - n["loss"]


def game_over(board: chess.Board) -> str | None:
    if board.is_checkmate():
        return "checkmate"
    if board.is_stalemate():
        return "stalemate"
    if board.is_insufficient_material():
        return "insufficient material"
    return None


def invalid_reason(board: chess.Board) -> str | None:
    """Plain-English reason a position can't happen, or None if it's fine."""
    status = board.status()
    if status == chess.STATUS_VALID:
        return None
    reasons = [
        (chess.STATUS_NO_WHITE_KING, "White needs a king"),
        (chess.STATUS_NO_BLACK_KING, "Black needs a king"),
        (chess.STATUS_TOO_MANY_KINGS, "each side can have only one king"),
        (chess.STATUS_PAWNS_ON_BACKRANK, "pawns can't stand on the first or last rank"),
        (chess.STATUS_OPPOSITE_CHECK, "the side not to move is in check"),
        (chess.STATUS_TOO_MANY_WHITE_PAWNS, "White has more than 8 pawns"),
        (chess.STATUS_TOO_MANY_BLACK_PAWNS, "Black has more than 8 pawns"),
        (chess.STATUS_TOO_MANY_WHITE_PIECES, "White has more than 16 pieces"),
        (chess.STATUS_TOO_MANY_BLACK_PIECES, "Black has more than 16 pieces"),
        (chess.STATUS_TOO_MANY_CHECKERS, "too many pieces are giving check"),
        (chess.STATUS_IMPOSSIBLE_CHECK, "this double check can't happen in a real game"),
    ]
    for flag, text in reasons:
        if status & flag:
            return text
    return "this position can't happen in a real game"
