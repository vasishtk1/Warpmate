#!/usr/bin/env python3
"""Makes the 8-bit (INT8) version of the trained network.

8-bit numbers are 4x smaller than 32-bit floats and the GPU can multiply
four pairs of them in one instruction (__dp4a on the T4). The price is
precision: every weight is rounded to one of 255 levels.

How the numbers are made (the CUDA kernel and the C++ reference in
src/inference do exactly the same math):

  * Weights, per neuron ("per output channel"): scale s = max |weight| / 127,
    q = round(weight / s), so q is in -127..127 and weight ~ s * q.
  * Layer 1: h1 = relu(s1 * (sum of the q1 rows of the board's inputs) + b1).
    The sum is exact integer arithmetic.
  * h1 is then stored in 8 bits too: h1q = min(127, round(h1 / a1)). The
    scale a1 is "calibrated": chosen from h1 values on training positions
    (never the test set). A small a1 keeps more precision but cuts off
    (clips) the largest values. Several candidates are tried and the one
    whose answers agree most often with the 32-bit network is kept.
  * Layer 2: h2 = relu(m2 * dot(q2, h1q) + b2) with m2 = s2 * a1, again an
    exact integer dot product followed by one multiply per neuron.
  * Layer 3 (64 x 3, tiny) stays 32-bit.

Usage:
    python ml/quantize.py --model model --data data/dataset.npz
Writes model/warpmate_net_int8.npz.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PAD = 769
STM = 768
CHUNK = 20_000


# --------------------------------------------------------------------------
# NumPy implementations (references for the C++/CUDA code)


def _rows(codes: np.ndarray) -> np.ndarray:
    """Piece codes -> row indices into a weight table with a zero row at 769
    (anything that isn't a piece points there)."""
    c = codes.astype(np.int64)
    return np.where((c >= 0) & (c < STM), c, PAD)


def float_forward(w: dict, codes: np.ndarray, stm: np.ndarray) -> np.ndarray:
    """32-bit network on compact boards -> logits (n, 3)."""
    w1t = np.vstack([w["w0"].T, np.zeros((1, w["w0"].shape[0]), np.float32)])  # 770 x 256
    out = []
    for i in range(0, len(codes), CHUNK):
        c, s = _rows(codes[i:i + CHUNK]), stm[i:i + CHUNK]
        h1 = w1t[c].sum(1) + (s == 0)[:, None] * w1t[STM] + w["b0"]
        h1 = np.maximum(h1, 0)
        h2 = np.maximum(h1 @ w["w1"].T + w["b1"], 0)
        out.append(h2 @ w["w2"].T + w["b2"])
    return np.concatenate(out).astype(np.float32)


def layer1_int8(q: dict, codes: np.ndarray, stm: np.ndarray) -> np.ndarray:
    """h1 (32-bit, before the 8-bit rounding) for the 8-bit network."""
    q1t = np.vstack([q["q1"].T.astype(np.int32), np.zeros((1, q["q1"].shape[0]), np.int32)])
    acc = q1t[_rows(codes)].sum(1) + (stm == 0)[:, None].astype(np.int32) * q1t[STM]
    return np.maximum(q["s1"] * acc.astype(np.float32) + q["b1"], np.float32(0))


def int8_forward(q: dict, codes: np.ndarray, stm: np.ndarray) -> np.ndarray:
    """8-bit network on compact boards -> logits (n, 3). Same float32
    operations in the same order as the C++ and CUDA code."""
    out = []
    q2 = q["q2"].astype(np.int32)
    for i in range(0, len(codes), CHUNK):
        h1 = layer1_int8(q, codes[i:i + CHUNK], stm[i:i + CHUNK])
        h1q = np.minimum(np.float32(127), np.rint(h1 / q["a1"])).astype(np.int32)
        acc = h1q @ q2.T                                       # exact integers
        h2 = np.maximum(q["m2"] * acc.astype(np.float32) + q["b2"], np.float32(0))
        out.append(h2 @ q["w3"].T + q["b3"])
    return np.concatenate(out).astype(np.float32)


def load_int8(path: str | Path) -> dict:
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


# --------------------------------------------------------------------------
# Making the 8-bit weights


def quantize_rows(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row scale and 8-bit values: w ~ scale[:, None] * q."""
    scale = (np.abs(w).max(1) / 127).astype(np.float32)
    scale[scale == 0] = 1.0
    q = np.clip(np.rint(w / scale[:, None]), -127, 127).astype(np.int8)
    return q, scale


def build(w: dict, a1: float) -> dict:
    q1, s1 = quantize_rows(w["w0"])
    q2, s2 = quantize_rows(w["w1"])
    a1 = np.float32(a1)
    return {"q1": q1, "s1": s1, "b1": w["b0"].astype(np.float32), "a1": a1,
            "q2": q2, "m2": (s2 * a1).astype(np.float32), "b2": w["b1"].astype(np.float32),
            "w3": w["w2"].astype(np.float32), "b3": w["b2"].astype(np.float32)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, default=Path("model"),
                        help="directory with warpmate_net.npz")
    parser.add_argument("--data", type=Path, required=True, help="dataset .npz (ml/dataset.py)")
    parser.add_argument("--calibration", type=int, default=200_000,
                        help="training positions used to choose a1")
    parser.add_argument("--validation", type=int, default=200_000,
                        help="other training positions used to compare the candidates")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    with np.load(args.model / "warpmate_net.npz") as f:
        w = {k: f[k].astype(np.float32) for k in f.files}
    data = np.load(args.data)
    train = np.flatnonzero(~data["is_test"])
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(train), size=min(len(train), args.calibration + args.validation),
                      replace=False)
    cal, val = train[pick[:args.calibration]], train[pick[args.calibration:]]
    codes, stm = data["codes"], data["stm"]
    print(f"Calibration: {len(cal):,} training positions; validation: {len(val):,} "
          f"(the test set is not used)")

    # The h1 values the 8-bit network will actually see (8-bit layer 1).
    probe = build(w, 1.0)
    h1 = np.concatenate([layer1_int8(probe, codes[cal[i:i + CHUNK]], stm[cal[i:i + CHUNK]])
                         for i in range(0, len(cal), CHUNK)])
    positive = h1[h1 > 0]
    print(f"h1: {h1.size:,} values, {positive.size / h1.size * 100:.1f}% above zero, "
          f"largest {h1.max():.3f}")

    vc, vs = codes[val], stm[val]
    ref_pick = float_forward(w, vc, vs).argmax(1)
    labels = data["label"][val]
    print(f"\n{'h1 range from':>18} {'a1':>9} {'clipped':>9} {'agrees w/ 32-bit':>17} "
          f"{'accuracy':>9}")
    print(f"{'(32-bit network)':>18} {'':>9} {'':>9} {'':>17} "
          f"{(ref_pick == labels).mean() * 100:8.3f}%")
    best = None
    for pct in (99.9, 99.99, 99.999, 100.0):
        top = float(np.percentile(positive, pct)) if pct < 100 else float(positive.max())
        q = build(w, top / 127)
        pick8 = int8_forward(q, vc, vs).argmax(1)
        agree = float((pick8 == ref_pick).mean())
        clipped = float((positive > top).mean())
        print(f"{f'{pct}th percentile':>18} {q['a1']:9.5f} {clipped * 100:8.4f}% "
              f"{agree * 100:16.3f}% {(pick8 == labels).mean() * 100:8.3f}%")
        if best is None or agree > best[0]:
            best = (agree, pct, q)

    agree, pct, q = best
    out = args.model / "warpmate_net_int8.npz"
    np.savez(out, **q)
    print(f"\nChose the {pct}th percentile (a1 = {q['a1']:.5f}): "
          f"{agree * 100:.3f}% of validation picks match the 32-bit network")
    size32 = sum(w[k].nbytes for k in w)
    size8 = sum(np.asarray(v).nbytes for v in q.values())
    print(f"Weights: {size32 / 1e3:.0f} KB (32-bit) -> {size8 / 1e3:.0f} KB (8-bit)")
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    start = time.time()
    code = main()
    print(f"({time.time() - start:.1f} s)")
    sys.exit(code)
