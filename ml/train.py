#!/usr/bin/env python3
"""Trains the Warpmate network on a dataset built by dataset.py.

The whole dataset is copied to the GPU once (it's compact: about 12 bytes
per example), and every batch's 769 inputs are built there by
model.features, so the GPU never waits for data.

Each step: the network guesses win/draw/loss for a batch of positions, the
cross-entropy loss measures how far the guesses are from the true answers,
loss.backward() works out how to nudge every weight, and the Adam optimizer
applies the nudges. After each epoch (one pass over the training set) the
network is scored on the test set, which it never trains on.

Usage:
    python ml/train.py --data data/dataset.npz --out model
    python ml/train.py --data data/dataset.npz --out model --epochs 10 --batch 4096

Writes to --out:
    warpmate_net.pt     the trained weights (PyTorch)
    warpmate_net.npz    the same weights as plain arrays (w0, b0, w1, b1, ...),
                        for the CUDA inference code in Phase 4
    training_log.json   loss and accuracy per epoch
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import WarpmateNet, features, save  # noqa: E402


def evaluate(model, codes, stm, label, batch: int) -> tuple[float, float]:
    """Average loss and accuracy over a whole set, in batches."""
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(label), batch):
            logits = model(features(codes[i:i + batch], stm[i:i + batch]))
            y = label[i:i + batch]
            total_loss += F.cross_entropy(logits, y, reduction="sum").item()
            correct += (logits.argmax(1) == y).sum().item()
    model.train()
    return total_loss / len(label), correct / len(label)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=str, default="256,64",
                        help="hidden layer sizes, e.g. 256,64")
    parser.add_argument("--max-train", type=int, default=0,
                        help="use only this many training examples (for quick tests)")
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")

    data = np.load(args.data)
    is_test = data["is_test"]
    to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)  # noqa: E731
    train_idx = np.flatnonzero(~is_test)
    if args.max_train:
        train_idx = train_idx[:args.max_train]
    test_idx = np.flatnonzero(is_test)
    tr_codes, tr_stm = to(data["codes"][train_idx]), to(data["stm"][train_idx])
    tr_label = to(data["label"][train_idx].astype(np.int64))
    te_codes, te_stm = to(data["codes"][test_idx]), to(data["stm"][test_idx])
    te_label = to(data["label"][test_idx].astype(np.int64))
    print(f"Training examples: {len(train_idx):,}   test examples: {len(test_idx):,}")

    # Baseline: always guess the most common answer. The network has to beat it.
    majority = int(np.bincount(data["label"][train_idx], minlength=3).argmax())
    baseline = float((data["label"][test_idx] == majority).mean())
    print(f"Baseline (always guess the most common answer): {baseline * 100:.2f}% test accuracy")

    model = WarpmateNet(tuple(int(h) for h in args.hidden.split(","))).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Network: {model.layers}\nWeights: {params:,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    log = {"baseline_accuracy": baseline, "weights": params, "hidden": model.hidden,
           "train_examples": len(train_idx), "test_examples": len(test_idx),
           "batch": args.batch, "lr": args.lr, "epochs": []}
    n = len(train_idx)
    print(f"\n{'epoch':>5} {'train loss':>11} {'test loss':>10} {'test acc':>9} {'time':>7}")
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        order = torch.randperm(n, device=device)  # new shuffle each epoch
        running = 0.0
        for i in range(0, n, args.batch):
            idx = order[i:i + args.batch]
            logits = model(features(tr_codes[idx], tr_stm[idx]))   # 1. guess
            loss = F.cross_entropy(logits, tr_label[idx])          # 2. how wrong
            optimizer.zero_grad()
            loss.backward()                                        # 3. work out nudges
            optimizer.step()                                       # 4. apply them
            running += loss.item() * len(idx)
        if device == "cuda":
            torch.cuda.synchronize()
        seconds = time.time() - start
        test_loss, test_acc = evaluate(model, te_codes, te_stm, te_label, 65536)
        print(f"{epoch:5d} {running / n:11.4f} {test_loss:10.4f} {test_acc * 100:8.2f}% "
              f"{seconds:6.1f}s")
        log["epochs"].append({"epoch": epoch, "train_loss": running / n, "test_loss": test_loss,
                              "test_accuracy": test_acc, "seconds": seconds})

    args.out.mkdir(parents=True, exist_ok=True)
    save(model, str(args.out / "warpmate_net.pt"))
    arrays = {}
    linears = [m for m in model.layers if isinstance(m, torch.nn.Linear)]
    for k, layer in enumerate(linears):
        arrays[f"w{k}"] = layer.weight.detach().cpu().numpy()  # shape (out, in)
        arrays[f"b{k}"] = layer.bias.detach().cpu().numpy()
    np.savez(args.out / "warpmate_net.npz", **arrays)
    (args.out / "training_log.json").write_text(json.dumps(log, indent=2))
    print(f"\nFinal test accuracy: {log['epochs'][-1]['test_accuracy'] * 100:.2f}% "
          f"(baseline {baseline * 100:.2f}%)")
    print(f"Saved weights to {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
