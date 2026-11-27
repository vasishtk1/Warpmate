#!/usr/bin/env python3
"""Measures the trained network against perfect answers.

1. Test-set accuracy: overall, per endgame, and per true result (a confusion
   matrix), each next to a "dumb" baseline that always guesses the most
   common answer.
2. Move choice: in positions from the test set, the network scores every
   legal move (by evaluating the position after it) and picks its favourite.
   The solved tables say whether that move keeps the best possible result
   (keeps a win, or keeps a draw). Losing positions are skipped, since every
   move there is equally lost.
3. Generalization to 5 pieces: random legal positions from 5-piece endgames
   the network never saw, labelled by the Syzygy tablebases.

Usage:
    python ml/evaluate.py --data data/dataset.npz --model model/warpmate_net.pt \
        --tables tables [--syzygy syzygy --five KRPvKR KQRvKR] [--out model/results.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chess
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import CLASSES, PAD, features, load  # noqa: E402
from tables import CHESS_TYPE, TableSet, canonical, wdl  # noqa: E402


def codes_of(board: chess.Board) -> tuple[list[int], int]:
    """Any position -> (piece codes, side to move), stronger side as White,
    the same form the network was trained on."""
    _, squares, pieces, stm = canonical(board)
    codes = [(c * 6 + t) * 64 + sq for (t, c), sq in zip(pieces, squares)]
    return codes + [PAD] * (5 - len(codes)), stm


def board_of(codes, stm) -> chess.Board:
    board = chess.Board(None)
    for code in codes:
        if code == PAD:
            continue
        kind, sq = divmod(int(code), 64)
        colour, t = divmod(kind, 6)
        board.set_piece_at(sq, chess.Piece(CHESS_TYPE[t], colour == 0))
    board.turn = chess.WHITE if stm == 0 else chess.BLACK
    return board


def predict(model, device, codes: np.ndarray, stm: np.ndarray, batch: int = 65536) -> np.ndarray:
    """Win/draw/loss probabilities (N, 3) for the side to move."""
    out = []
    with torch.no_grad():
        for i in range(0, len(codes), batch):
            c = torch.from_numpy(np.ascontiguousarray(codes[i:i + batch])).to(device)
            s = torch.from_numpy(np.ascontiguousarray(stm[i:i + batch])).to(device)
            out.append(torch.softmax(model(features(c, s)), 1).cpu().numpy())
    return np.concatenate(out)


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def test_accuracy(model, device, data) -> dict:
    is_test = data["is_test"]
    names = list(data["names"])
    label = data["label"][is_test]
    table_id = data["table_id"][is_test]
    probs = predict(model, device, data["codes"][is_test], data["stm"][is_test])
    guess = probs.argmax(1)
    majority = int(np.bincount(data["label"][~is_test], minlength=3).argmax())

    print("== 1. Test-set accuracy (positions never used in training)")
    overall = float((guess == label).mean())
    baseline = float((label == majority).mean())
    print(f"Overall: {pct(overall)} on {len(label):,} positions "
          f"(baseline, always '{CLASSES[majority]}': {pct(baseline)})")

    confusion = np.zeros((3, 3), dtype=np.int64)
    np.add.at(confusion, (label, guess), 1)
    print("\nTrue result (rows) vs network's guess (columns), for the side to move:")
    print(f"{'':>8}" + "".join(f"{c:>12}" for c in CLASSES) + f"{'accuracy':>12}")
    for t in range(3):
        row_total = confusion[t].sum()
        acc = confusion[t, t] / row_total if row_total else 0.0
        print(f"{CLASSES[t]:>8}" + "".join(f"{v:12,}" for v in confusion[t]) + f"{pct(acc):>12}")

    per_table = []
    print(f"\n{'Endgame':8} {'test positions':>15} {'accuracy':>10} {'baseline':>10}")
    for k, name in enumerate(names):
        mask = table_id == k
        if not mask.any():
            continue
        acc = float((guess[mask] == label[mask]).mean())
        base = float((label[mask] == np.bincount(label[mask], minlength=3).argmax()).mean())
        per_table.append({"endgame": str(name), "positions": int(mask.sum()),
                          "accuracy": acc, "baseline": base})
    for row in sorted(per_table, key=lambda r: r["accuracy"]):
        print(f"{row['endgame']:8} {row['positions']:15,} {pct(row['accuracy']):>10} "
              f"{pct(row['baseline']):>10}")
    print("(baseline here = always guess that endgame's most common answer)")
    return {"accuracy": overall, "baseline": baseline, "positions": int(len(label)),
            "confusion": confusion.tolist(), "per_endgame": per_table}


def move_choice(model, device, data, tables: TableSet, samples: int, rng) -> dict:
    print(f"\n== 2. Move choice (up to {samples:,} winning or drawn test positions)")
    test_rows = np.flatnonzero(data["is_test"] & (data["label"] != 0))
    picks = rng.choice(test_rows, size=min(samples, len(test_rows)), replace=False)

    positions = []  # (true result of the position, [true result of each move for the mover])
    child_codes, child_stm, owner = [], [], []
    for i in picks:
        board = board_of(data["codes"][i], data["stm"][i])
        moves = list(board.legal_moves)
        if not moves:
            continue
        truths = []
        for mv in moves:
            board.push(mv)
            truths.append(-wdl(tables.value(board)))  # child is from the opponent's view
            c, s = codes_of(board)
            board.pop()
            child_codes.append(c)
            child_stm.append(s)
            owner.append(len(positions))
        positions.append((int(data["label"][i]) - 1, truths))

    probs = predict(model, device, np.array(child_codes, dtype=np.int16),
                    np.array(child_stm, dtype=np.uint8))
    # The mover wants the opponent (to move after the move) to lose:
    # score = P(opponent loses) - P(opponent wins).
    scores = probs[:, 0] - probs[:, 2]

    kept = {1: [0, 0], 0: [0, 0]}  # position result -> [best move chosen, positions]
    start = 0
    for p, (result, truths) in enumerate(positions):
        end = start + len(truths)
        chosen = int(np.argmax(scores[start:end]))
        kept[result][1] += 1
        kept[result][0] += truths[chosen] == max(truths)
        start = end

    total_ok = kept[1][0] + kept[0][0]
    total = kept[1][1] + kept[0][1]
    print(f"Winning positions: chose a move that keeps the win in "
          f"{pct(kept[1][0] / max(1, kept[1][1]))} of {kept[1][1]:,}")
    print(f"Drawn positions:   chose a move that keeps the draw in "
          f"{pct(kept[0][0] / max(1, kept[0][1]))} of {kept[0][1]:,}")
    print(f"Overall: the network's favourite move keeps the best result in "
          f"{pct(total_ok / max(1, total))} of {total:,} positions")
    return {"winning": {"kept": kept[1][0], "positions": kept[1][1]},
            "drawn": {"kept": kept[0][0], "positions": kept[0][1]},
            "overall": total_ok / max(1, total)}


def random_board(name: str, rng) -> chess.Board:
    white, black = name.split("v")
    while True:
        board = chess.Board(None)
        squares = rng.choice(64, size=len(white) + len(black), replace=False)
        pieces = ([(c, chess.WHITE) for c in white] + [(c, chess.BLACK) for c in black])
        for (letter, colour), sq in zip(pieces, squares):
            board.set_piece_at(int(sq), chess.Piece(CHESS_TYPE["KQRBNP".index(letter)], colour))
        board.turn = bool(rng.integers(2))
        if board.is_valid():
            return board


def generalization(model, device, syzygy: Path, names: list[str], samples: int, rng) -> list:
    import chess.syzygy
    print(f"\n== 3. Generalization: 5-piece endgames never seen in training "
          f"({samples:,} random legal positions each, labelled by Syzygy)")
    print(f"{'Endgame':8} {'accuracy':>10} {'baseline':>10}   true loss / draw / win")
    results = []
    with chess.syzygy.open_tablebase(str(syzygy)) as tb:
        for name in names:
            codes, stm, label = [], [], []
            while len(label) < samples:
                board = random_board(name, rng)
                try:
                    truth = int(np.sign(tb.probe_wdl(board))) + 1
                except (chess.syzygy.MissingTableError, KeyError):
                    print(f"{name:8} skipped: Syzygy file missing")
                    break
                c, s = codes_of(board)
                codes.append(c)
                stm.append(s)
                label.append(truth)
            if len(label) < samples:
                continue
            label = np.array(label)
            guess = predict(model, device, np.array(codes, dtype=np.int16),
                            np.array(stm, dtype=np.uint8)).argmax(1)
            acc = float((guess == label).mean())
            dist = np.bincount(label, minlength=3) / len(label)
            base = float(dist.max())
            print(f"{name:8} {pct(acc):>10} {pct(base):>10}   "
                  f"{dist[0] * 100:4.1f}% / {dist[1] * 100:4.1f}% / {dist[2] * 100:4.1f}%")
            results.append({"endgame": name, "positions": samples, "accuracy": acc,
                            "baseline": base})
    print("(baseline = always guess that endgame's most common answer)")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tables", type=Path, required=True,
                        help="solved tables, for checking move choices")
    parser.add_argument("--move-samples", type=int, default=20_000)
    parser.add_argument("--syzygy", type=Path, help="Syzygy directory with 5-piece files")
    parser.add_argument("--five", nargs="*", default=[], help="5-piece endgames to test")
    parser.add_argument("--five-samples", type=int, default=20_000)
    parser.add_argument("--out", type=Path, help="write results as JSON")
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    model = load(str(args.model), device)
    data = np.load(args.data)

    results = {"test": test_accuracy(model, device, data)}
    results["moves"] = move_choice(model, device, data, TableSet(args.tables),
                                   args.move_samples, rng)
    if args.syzygy and args.five:
        results["five_piece"] = generalization(model, device, args.syzygy, args.five,
                                               args.five_samples, rng)
    if args.out:
        args.out.write_text(json.dumps(results, indent=2))
        print(f"\nResults written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
