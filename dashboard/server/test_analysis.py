"""Tests for the dashboard's analysis. Needs the 3-piece tables in <repo>/tables
(./build/warpmate-gen --all3). Run from dashboard/server:

    python -m unittest -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import chess
import numpy as np

from analysis import ROOT, Analyzer, invalid_reason

TABLES = ROOT / "tables"


def moves_by_san(result: dict) -> dict:
    return {m["san"]: m for m in result["moves"]}


@unittest.skipUnless((TABLES / "KRvK.wmtb").exists(), "3-piece tables not found")
class PerfectAnswers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a = Analyzer(TABLES, Path("/nonexistent/model.pt"))  # placeholder network

    def test_longest_krvk_mate(self):
        r = self.a.analyze("8/8/8/8/8/2k5/1R6/K7 w - - 0 1")
        self.assertTrue(r["valid"])
        self.assertEqual(r["perfect"]["text"], "White mates in 16")
        self.assertEqual(r["ranked_by"], "perfect")
        best = [m["san"] for m in r["moves"] if m["best"]]
        self.assertIn("Kb1", best)
        for m in r["moves"]:
            if m["best"]:
                self.assertEqual(m["perfect"]["text"], "mate in 16")

    def test_mate_in_one(self):
        r = self.a.analyze("7k/8/6K1/8/8/8/8/R7 w - - 0 1")
        self.assertEqual(r["perfect"]["text"], "White mates in 1")
        self.assertEqual(r["moves"][0]["san"], "Ra8#")
        self.assertEqual(r["moves"][0]["perfect"]["text"], "mate in 1")

    def test_checkmated(self):
        r = self.a.analyze("R6k/8/6K1/8/8/8/8/8 b - - 0 1")
        self.assertEqual(r["game_over"], "checkmate")
        self.assertEqual(r["perfect"]["text"], "Black is checkmated")
        self.assertEqual(r["moves"], [])

    def test_colour_flipped(self):
        # Black has the rook: looked up through the KRvK table, colours swapped.
        r = self.a.analyze("r7/8/8/8/8/6k1/8/7K b - - 0 1")
        self.assertEqual(r["perfect"]["text"], "Black mates in 1")

    def test_pawn_promotes_into_another_table(self):
        r = self.a.analyze("3k4/8/3K4/3P4/8/8/8/8 w - - 0 1")
        self.assertEqual(r["perfect"]["text"], "White mates in 11")

    def test_draw_and_blunder_ranking(self):
        # Rook next to the black king and undefended: Black to move takes it.
        r = self.a.analyze("7K/8/8/8/8/8/1R6/k7 b - - 0 1")
        self.assertEqual(r["perfect"]["text"], "Draw")
        self.assertEqual(r["moves"][0]["san"], "Kxb2")

    def test_ranked_best_first(self):
        r = self.a.analyze("8/8/8/4k3/8/8/8/R3K3 w - - 0 1")
        results = [m["perfect"] for m in r["moves"]]
        wins = [p["mate_in"] for p in results if p["result"] > 0]
        self.assertEqual(wins, sorted(wins))  # faster mates first

    def test_more_pieces_than_tables(self):
        r = self.a.analyze("4k3/8/8/8/8/8/3PPP2/4K3 w - - 0 1")
        self.assertTrue(r["valid"])
        self.assertFalse(r["perfect"]["available"])
        self.assertEqual(r["ranked_by"], "network")
        self.assertFalse(r["network"]["trained_range"])

    def test_placeholder_network_is_labelled(self):
        r = self.a.analyze("8/8/8/8/8/2k5/1R6/K7 w - - 0 1")
        self.assertTrue(r["network"]["placeholder"])

    def test_random_position(self):
        import random
        fen = self.a.random_position("KRvK", random.Random(1))
        board = chess.Board(fen)
        self.assertTrue(board.is_valid())
        self.assertEqual(len(board.piece_map()), 3)


class Validity(unittest.TestCase):
    def test_reasons(self):
        self.assertIsNone(invalid_reason(chess.Board("7k/8/6K1/8/8/8/8/R7 w - - 0 1")))
        self.assertIn("king", invalid_reason(chess.Board("8/8/8/8/8/8/8/RK6 w - - 0 1")))
        self.assertIn("not to move",
                      invalid_reason(chess.Board("R6k/8/6K1/8/8/8/8/8 w - - 0 1")))
        self.assertIn("double check",
                      invalid_reason(chess.Board("8/8/8/8/8/8/8/KQ1k1Q2 b - - 0 1")))


class NetworkEncoding(unittest.TestCase):
    """The dashboard must feed the network exactly what evaluate.py does."""

    def test_matches_evaluate(self):
        import torch
        sys.path.insert(0, str(ROOT / "ml"))
        from evaluate import codes_of, predict
        from model import WarpmateNet, save

        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "net.pt"
            save(WarpmateNet(), str(path))  # random weights are enough here
            a = Analyzer(Path(tmp), path)
            self.assertFalse(a.network.placeholder)
            fens = ["8/8/8/8/8/2k5/1R6/K7 w - - 0 1", "r7/8/8/8/8/6k1/8/7K b - - 0 1",
                    "3k4/8/3K4/3P4/8/8/8/8 w - - 0 1", "8/8/1q6/8/3k4/8/2R5/K7 b - - 0 1"]
            boards = [chess.Board(f) for f in fens]
            ours = a.network.predict(boards)
            codes, stm = zip(*(codes_of(b) for b in boards))
            theirs = predict(a.network.model, "cpu", np.array(codes, dtype=np.int16),
                             np.array(stm, dtype=np.uint8))
            np.testing.assert_allclose(ours, theirs, atol=1e-6)
            np.testing.assert_allclose(ours.sum(1), 1.0, atol=1e-5)
            # Works for a full board too (more than 5 pieces).
            full = a.network.predict([chess.Board()])
            self.assertEqual(full.shape, (1, 3))


if __name__ == "__main__":
    unittest.main()
