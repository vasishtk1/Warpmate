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

The 8-bit versions can't match PyTorch's 32-bit answers exactly. They are
checked against the NumPy implementation of the same 8-bit math
(ml/quantize.py), which they must match to within rounding, and their
accuracy drop compared with PyTorch is reported (and must stay under 1
percentage point).

It also re-runs a sample with extra padding slots and with awkward batch sizes
(1, 7, 9, ...) to check those give exactly the same answers.

Usage:
    python ml/check_inference.py --data data/dataset.npz --model model [--limit N]

`--model` is the directory holding warpmate_net.pt, warpmate_net.npz and
(optionally) warpmate_net_int8.npz from ml/quantize.py.
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
from quantize import int8_forward, load_int8  # noqa: E402

PROB_TOLERANCE = 1e-4   # 32-bit versions: max allowed probability difference
TIE_MARGIN = 1e-4       # picks within this of a tie may flip on rounding
INT8_TOLERANCE = 1e-5   # 8-bit versions vs the NumPy 8-bit reference
INT8_MAX_DROP = 0.01    # max accuracy drop of the 8-bit versions (1 point)


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
          f"{real_differ:>8} {accuracy * 100:8.3f}%  "
          f"{'(info)' if tolerance is None else 'PASS' if ok else 'FAIL'}")
    return {"boards": n, "max_prob_diff": prob_diff, "max_logit_diff": logit_diff,
            "picks_differ": int(differ.sum()), "picks_differ_not_near_tie": real_differ,
            "accuracy": accuracy, "pass": bool(ok)}


def header():
    print(f"  {'version':12} {'boards':>10} {'prob diff':>10} {'logit diff':>9} "
          f"{'pick !=':>8} {'non-tie':>8} {'accuracy':>9}")


def check_int8(engine, int8_path, gpu_variants, codes, stm, labels, ref_probs, ref_logits,
               n_cpu, results) -> bool:
    """8-bit versions: same answers as the NumPy 8-bit math, small accuracy drop."""
    q = load_int8(int8_path)
    ok = True
    print("\n== 8-bit versions compared with PyTorch's 32-bit answers (for information)")
    header()
    outputs = {"cpu-int8": engine.cpu(codes[:n_cpu], stm[:n_cpu], int8=True, logits=True)}
    if "int8" in gpu_variants:
        outputs["int8"] = engine.gpu("int8", codes, stm, logits=True)
    for name, (probs, logits) in outputs.items():
        results["versions"][name] = compare(name, probs, logits, ref_probs, ref_logits, labels,
                                            None)

    sample = min(200_000, len(codes))
    ref8 = int8_forward(q, codes[:sample], stm[:sample])
    ref8_probs = np.exp(ref8 - ref8.max(1, keepdims=True))
    ref8_probs /= ref8_probs.sum(1, keepdims=True)
    torch_acc = float((ref_probs.argmax(1) == labels).mean())
    print(f"\n== 8-bit versions compared with the NumPy 8-bit reference ({sample:,} boards)")
    for name, (probs, logits) in outputs.items():
        m = min(sample, len(probs))
        diff = float(np.abs(probs[:m] - ref8_probs[:m]).max())
        logit_diff = float(np.abs(logits[:m] - ref8[:m]).max())
        n = len(probs)
        acc = float((probs.argmax(1) == labels[:n]).mean())
        torch_acc_n = float((ref_probs[:n].argmax(1) == labels[:n]).mean())
        drop = torch_acc_n - acc
        good = diff < INT8_TOLERANCE and drop < INT8_MAX_DROP
        ok &= good
        print(f"  {name:12} max prob diff {diff:.2e}, max logit diff {logit_diff:.2e}; "
              f"accuracy {acc * 100:.3f}% vs PyTorch {torch_acc_n * 100:.3f}% on the same "
              f"{n:,} boards (drop {drop * 100:+.3f} points)  {'PASS' if good else 'FAIL'}")
        results["versions"][name].update({"max_prob_diff_vs_numpy_int8": diff,
                                          "accuracy_drop": drop, "pass": bool(good)})
    results["pytorch_accuracy"] = torch_acc
    return ok


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

    int8_path = args.model / "warpmate_net_int8.npz"
    engine = Engine(args.model / "warpmate_net.npz", int8_path if int8_path.exists() else None)
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
    for v in [v for v in gpu_variants if v != "int8"]:
        start = time.time()
        probs, logits = engine.gpu(v, codes, stm, logits=True)
        r = compare(v, probs, logits, ref_probs, ref_logits, labels, PROB_TOLERANCE)
        r["seconds_including_copies"] = time.time() - start
        results["versions"][v] = r
        ok &= r["pass"]

    if engine.has_int8:
        ok &= check_int8(engine, int8_path, gpu_variants, codes, stm, labels, ref_probs,
                         ref_logits, n_cpu, results)
    else:
        print(f"\n(no {int8_path}: 8-bit checks skipped; make it with ml/quantize.py)")

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
