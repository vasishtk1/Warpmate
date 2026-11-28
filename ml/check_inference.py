#!/usr/bin/env python3
"""Checks Warpmate's own inference code against PyTorch on real positions.

For every implementation available (the C++ CPU reference, and each CUDA
variant when there is a GPU) it runs the test set and compares with PyTorch:

  * the largest difference in any probability and any logit
  * whether it picks the same result (loss / draw / win) on every board.
    Boards where PyTorch's top two answers are within 1e-4 of each other
    are counted separately: there a rounding difference in the last digit
    can legitimately flip the pick.
  * accuracy against the perfect answers (the labels)

It also re-runs a sample with extra padding slots and with awkward batch sizes
(1, 7, 9, ...) to check those give exactly the same answers.

Usage:
    python ml/check_inference.py --data data/dataset.npz --model model [--limit N]

`--model` is the directory holding warpmate_net.pt and warpmate_net.npz.
Exits with status 1 if any check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference import Engine  # noqa: E402
from model import PAD, features, load  # noqa: E402

PROB_TOLERANCE = 1e-4   # 32-bit versions: max allowed probability difference
TIE_MARGIN = 1e-4       # picks within this of a tie may flip on rounding


def torch_reference(model, device, codes, stm, batch=65536):
    probs, logits = [], []
    with torch.no_grad():
        for i in range(0, len(codes), batch):
            c = torch.from_numpy(np.ascontiguousarray(codes[i:i + batch])).to(device)
            s = torch.from_numpy(np.ascontiguousarray(stm[i:i + batch])).to(device)
            out = model(features(c, s))
            logits.append(out.cpu().numpy())
            probs.append(torch.softmax(out, 1).cpu().numpy())
    return np.concatenate(probs), np.concatenate(logits)


def compare(name, probs, logits, ref_probs, ref_logits, labels, tolerance):
    """Prints and returns the comparison of one implementation with PyTorch."""
    n = len(probs)
    ref_probs, ref_logits, labels = ref_probs[:n], ref_logits[:n], labels[:n]
    prob_diff = float(np.abs(probs - ref_probs).max())
    logit_diff = float(np.abs(logits - ref_logits).max()) if logits is not None else None
    pick, ref_pick = probs.argmax(1), ref_probs.argmax(1)
    top2 = np.sort(ref_probs, 1)
    near_tie = (top2[:, 2] - top2[:, 1]) < TIE_MARGIN
    differ = pick != ref_pick
    real_differ = int((differ & ~near_tie).sum())
    accuracy = float((pick == labels).mean())
    ok = prob_diff < tolerance and real_differ == 0 if tolerance is not None else True
    logit_text = f"{logit_diff:9.2e}" if logit_diff is not None else "      n/a"
    print(f"  {name:12} {n:>10,} {prob_diff:10.2e} {logit_text} {int(differ.sum()):>8} "
          f"{real_differ:>8} {accuracy * 100:8.3f}%  {'PASS' if ok else 'FAIL'}")
    return {"boards": n, "max_prob_diff": prob_diff, "max_logit_diff": logit_diff,
            "picks_differ": int(differ.sum()), "picks_differ_not_near_tie": real_differ,
            "accuracy": accuracy, "pass": bool(ok)}


def header():
    print(f"  {'version':12} {'boards':>10} {'prob diff':>10} {'logit diff':>9} "
          f"{'pick !=':>8} {'non-tie':>8} {'accuracy':>9}")


def shape_checks(engine, variants, codes, stm, rng) -> bool:
    """Same boards with extra padding and in awkward batch sizes must give
    exactly the same answers as one big batch."""
    print("\n== Padding and batch-size checks (must match exactly)")
    sample = rng.choice(len(codes), size=min(5000, len(codes)), replace=False)
    c, s = codes[sample], stm[sample]
    ok = True
    for v in variants:
        base = engine.gpu(v, c, s) if v != "cpu" else engine.cpu(c, s)
        run = (lambda cc, ss: engine.cpu(cc, ss)) if v == "cpu" else \
              (lambda cc, ss, v=v: engine.gpu(v, cc, ss))
        worst = 0.0
        # 1. extra padding slots (8 and 32 slots instead of 5)
        for slots in (8, 32):
            padded = np.full((len(c), slots), PAD, dtype=np.int16)
            padded[:, :c.shape[1]] = c
            worst = max(worst, float(np.abs(run(padded, s) - base).max()))
        # 2. pieces in a different slot order
        shuffled = c[:, ::-1].copy()
        worst_order = float(np.abs(run(shuffled, s) - base).max())
        # 3. awkward batch sizes
        start = 0
        for size in (1, 7, 9, 31, 33, 255, 257, 1000):
            part = run(c[start:start + size], s[start:start + size])
            worst = max(worst, float(np.abs(part - base[start:start + size]).max()))
            start += size
        # Padding and batch size must not change a single bit. A different slot
        # order adds the same numbers in a different order, so the last digit
        # may change; it must stay far inside the tolerance.
        good = worst == 0.0 and worst_order < 1e-5
        ok &= good
        print(f"  {v:12} padding/batch sizes: max diff {worst:.1e}; slot order: "
              f"{worst_order:.1e}  {'PASS' if good else 'FAIL'}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="dataset .npz (ml/dataset.py)")
    parser.add_argument("--model", type=Path, default=Path("model"),
                        help="directory with warpmate_net.pt and warpmate_net.npz")
    parser.add_argument("--limit", type=int, default=0, help="use only this many test boards")
    parser.add_argument("--cpu-limit", type=int, default=300_000,
                        help="boards for the (slower) CPU reference")
    parser.add_argument("--out", type=Path, help="write the results as JSON")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False  # plain 32-bit math in PyTorch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = np.load(args.data)
    is_test = data["is_test"]
    codes, stm, labels = data["codes"][is_test], data["stm"][is_test], data["label"][is_test]
    if args.limit:
        codes, stm, labels = codes[:args.limit], stm[:args.limit], labels[:args.limit]
    print(f"{len(codes):,} test boards; PyTorch reference on {device}")

    model = load(str(args.model / "warpmate_net.pt"), device)
    start = time.time()
    ref_probs, ref_logits = torch_reference(model, device, codes, stm)
    print(f"PyTorch accuracy {(ref_probs.argmax(1) == labels).mean() * 100:.3f}% "
          f"({time.time() - start:.1f} s)")

    engine = Engine(args.model / "warpmate_net.npz")
    print(f"Inference library: GPU {'yes, ' + engine.device_name if engine.has_gpu else 'no'}; "
          f"variants {engine.variants if engine.has_gpu else []}")

    results = {"boards": len(codes), "device": engine.device_name,
               "pytorch_accuracy": float((ref_probs.argmax(1) == labels).mean()),
               "versions": {}}
    ok = True
    print("\n== Compared with PyTorch")
    header()

    n_cpu = min(args.cpu_limit, len(codes))
    probs, logits = engine.cpu(codes[:n_cpu], stm[:n_cpu], logits=True)
    r = compare("cpu", probs, logits, ref_probs, ref_logits, labels, PROB_TOLERANCE)
    results["versions"]["cpu"] = r
    ok &= r["pass"]

    gpu_variants = engine.variants if engine.has_gpu else []
    for v in gpu_variants:
        start = time.time()
        probs, logits = engine.gpu(v, codes, stm, logits=True)
        r = compare(v, probs, logits, ref_probs, ref_logits, labels, PROB_TOLERANCE)
        r["seconds_including_copies"] = time.time() - start
        results["versions"][v] = r
        ok &= r["pass"]

    rng = np.random.default_rng(0)
    shapes_ok = shape_checks(engine, ["cpu"] + gpu_variants, codes, stm, rng)
    results["shape_checks_pass"] = bool(shapes_ok)
    ok &= shapes_ok

    if not engine.has_gpu:
        print("\nNo GPU here: only the CPU reference was checked.")
    print(f"\nOverall: {'PASS' if ok else 'FAIL'}")
    if args.out:
        args.out.write_text(json.dumps(results, indent=2))
        print(f"Saved {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
