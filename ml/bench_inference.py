#!/usr/bin/env python3
"""Benchmarks Warpmate's CUDA inference kernels against PyTorch and TensorRT.

Every method starts from the same place, the compact boards already in GPU
memory (codes (n, 5) int16, stm (n,) uint8), and ends with (n, 3)
probabilities in GPU memory. So input preparation (turning codes into 769
numbers, or the type conversions TensorRT needs) is counted for everyone.

Methods:
  ours: naive, fused, sparse, int8   (src/inference, through ml/inference.py)
  pytorch          the Phase 3 code: features() + model + softmax, 32-bit
  pytorch-fp16     the same in 16-bit floats (uses the T4's tensor cores)
  torch.compile    PyTorch's compiler (generates fused kernels)
  trt-dense / trt-dense-fp16    TensorRT on the 769-number input
  trt-sparse / trt-sparse-fp16  TensorRT given the same "add up the weight
                   rows" trick as our sparse kernel (Gather + ReduceSum), so
                   the comparison isn't just "sparse vs dense"
TensorRT engines are built separately for every batch size (min = opt = max),
which lets TensorRT tune for each one.

Each method is timed two ways, with CUDA events, after warm-up:
  eager:  calling it from Python in a loop. Includes Python and kernel-launch
          overhead, which dominates for small batches. This is what a Python
          program (like the dashboard) actually pays.
  graph:  the same work recorded once as a CUDA graph and replayed. Launch
          overhead mostly disappears, leaving close to pure GPU time.

Usage:
    python ml/bench_inference.py --data data/dataset.npz --model model \
        [--batches 1 32 256 4096 65536] [--no-trt] [--no-compile] [--profile] \
        [--out model/inference_bench.json]
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference import Engine  # noqa: E402
from model import N_INPUTS, features, load  # noqa: E402

OURS = ("naive", "fused", "sparse", "int8")


# --------------------------------------------------------------------------
# Timing


def time_eager(fn, seconds: float = 0.3) -> float:
    """Average milliseconds per call of fn(), called from Python in a loop."""
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(10):
        fn()
    end.record()
    end.synchronize()
    per = start.elapsed_time(end) / 10
    reps = int(min(5000, max(20, seconds * 1000 / max(per, 1e-3))))
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / reps


def time_graph(fn, seconds: float = 0.3) -> float | None:
    """Average milliseconds per replay of fn() recorded as a CUDA graph, or
    None if this method can't be recorded."""
    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):  # warm up on a side stream, as PyTorch advises
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
    except Exception as e:  # noqa: BLE001 - report and carry on
        print(f"      (CUDA graph capture failed: {str(e).splitlines()[0][:100]})")
        torch.cuda.synchronize()
        return None
    return time_eager(graph.replay, seconds)


# --------------------------------------------------------------------------
# Methods. Each make_* returns fn() writing into / returning a probs tensor.


def make_ours(engine: Engine, variant: str, codes, stm):
    probs = torch.empty((len(codes), 3), device="cuda", dtype=torch.float32)

    def fn():
        engine.gpu_device(variant, codes, stm, probs)
        return probs
    return fn, lambda: probs


def make_pytorch(model, codes, stm, half: bool = False):
    net = model.half() if half else model
    out = {}

    def fn():
        with torch.no_grad():
            x = features(codes, stm)
            if half:
                x = x.half()
            out["p"] = torch.softmax(net(x).float(), 1)
        return out["p"]
    return fn, lambda: out["p"]


def make_compiled(compiled, codes, stm):
    out = {}

    def fn():
        out["p"] = compiled(codes, stm)
        return out["p"]
    return fn, lambda: out["p"]


class SparseModule(nn.Module):
    """The network with layer 1 written as "gather the weight rows of the
    inputs that are on, add them up", for exporting to TensorRT.
    Inputs: codes (n, slots) int32 (769 = empty), white (n, 1) float."""

    def __init__(self, model):
        super().__init__()
        first, second, third = [m for m in model.layers if isinstance(m, nn.Linear)]
        w1t = first.weight.detach().t()                                # 769 x 256
        self.table = nn.Parameter(torch.cat([w1t, torch.zeros(1, w1t.shape[1],
                                                             device=w1t.device)]),
                                  requires_grad=False)                # 770 x 256
        self.stm_row = nn.Parameter(w1t[768].clone(), requires_grad=False)
        self.b1 = nn.Parameter(first.bias.detach().clone(), requires_grad=False)
        self.second, self.third = second, third

    def forward(self, codes, white):
        h1 = torch.relu(self.table[codes].sum(1) + white * self.stm_row + self.b1)
        h2 = torch.relu(self.second(h1))
        return torch.softmax(self.third(h2), 1)


class DenseModule(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return torch.softmax(self.model(x), 1)


def onnx_bytes(module, args, names, dynamic) -> bytes:
    buf = io.BytesIO()
    kwargs = dict(input_names=names, output_names=["probs"], opset_version=17,
                  dynamic_axes={n: {0: "batch"} for n in names + ["probs"]} if dynamic else None)
    try:
        torch.onnx.export(module, args, buf, dynamo=False, **kwargs)
    except TypeError:  # older PyTorch without the dynamo argument
        torch.onnx.export(module, args, buf, **kwargs)
    return buf.getvalue()


class TrtRunner:
    """A TensorRT engine for one batch size, run on torch's current stream."""

    def __init__(self, trt, onnx: bytes, inputs: dict, fp16: bool):
        logger = trt.Logger(trt.Logger.ERROR)
        builder = trt.Builder(logger)
        flags = 0
        if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH") and \
                int(trt.__version__.split(".")[0]) < 10:
            flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flags)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse(onnx):
            raise RuntimeError("; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        profile = builder.create_optimization_profile()
        for name, t in inputs.items():
            profile.set_shape(name, tuple(t.shape), tuple(t.shape), tuple(t.shape))
        config.add_optimization_profile(profile)
        plan = builder.build_serialized_network(network, config)
        if plan is None:
            raise RuntimeError("TensorRT build failed")
        self.engine = trt.Runtime(logger).deserialize_cuda_engine(plan)
        self.context = self.engine.create_execution_context()
        self.inputs = inputs
        n = next(iter(inputs.values())).shape[0]
        self.probs = torch.empty((n, 3), device="cuda", dtype=torch.float32)
        for name, t in inputs.items():
            self.context.set_input_shape(name, tuple(t.shape))

    def run(self) -> torch.Tensor:
        for name, t in self.inputs.items():
            self.context.set_tensor_address(name, t.data_ptr())
        self.context.set_tensor_address("probs", self.probs.data_ptr())
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        return self.probs


def make_trt(trt, model, kind: str, fp16: bool, codes, stm):
    """TensorRT method. The input preparation runs in PyTorch inside fn()."""
    n = len(codes)
    if kind == "dense":
        x = torch.empty((n, N_INPUTS), device="cuda")
        runner = TrtRunner(trt, onnx_bytes(DenseModule(model), (x,), ["x"], False),
                           {"x": x}, fp16)

        def fn():
            x.copy_(features(codes, stm))
            return runner.run()
    else:
        c32 = torch.empty(codes.shape, device="cuda", dtype=torch.int32)
        white = torch.empty((n, 1), device="cuda")
        runner = TrtRunner(trt, onnx_bytes(SparseModule(model), (c32, white),
                                           ["codes", "white"], False),
                           {"codes": c32, "white": white}, fp16)

        def fn():
            c32.copy_(codes)
            white.copy_((stm == 0).float().unsqueeze(1))
            return runner.run()
    return fn, lambda: runner.probs


# --------------------------------------------------------------------------


def profile(engine, model, codes, stm, batch: int) -> str:
    """Per-kernel GPU time for each method at one batch size (PyTorch's
    profiler sees every kernel in the process, including ours)."""
    from torch.profiler import ProfilerActivity
    lines = [f"\n== Profile at batch {batch:,} (GPU time per kernel, 20 calls each)"]
    c, s = codes[:batch], stm[:batch]
    methods = [(v, make_ours(engine, v, c, s)[0]) for v in engine.variants]
    methods.append(("pytorch", make_pytorch(model, c, s)[0]))
    for name, fn in methods:
        fn()
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(20):
                fn()
            torch.cuda.synchronize()
        try:
            table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=8)
        except Exception:  # noqa: BLE001 - newer PyTorch renamed the column
            table = prof.key_averages().table(sort_by="device_time_total", row_limit=8)
        lines += [f"\n--- {name}", table]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("model"))
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 32, 256, 4096, 65536])
    parser.add_argument("--no-trt", action="store_true", help="skip TensorRT")
    parser.add_argument("--no-compile", action="store_true", help="skip torch.compile")
    parser.add_argument("--profile", action="store_true", help="also print per-kernel times")
    parser.add_argument("--seconds", type=float, default=0.3, help="timing length per method")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("needs an NVIDIA GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    data = np.load(args.data)
    test = np.flatnonzero(data["is_test"])
    rng = np.random.default_rng(0)
    pick = rng.choice(test, size=max(args.batches), replace=len(test) < max(args.batches))
    all_codes = torch.from_numpy(data["codes"][pick]).cuda().contiguous()
    all_stm = torch.from_numpy(data["stm"][pick]).cuda().contiguous()

    int8_path = args.model / "warpmate_net_int8.npz"
    engine = Engine(args.model / "warpmate_net.npz", int8_path if int8_path.exists() else None)
    if not engine.has_gpu:
        sys.exit("the inference library has no GPU support (was it built with CUDA?)")
    model = load(str(args.model / "warpmate_net.pt"), "cuda")

    trt = None
    if not args.no_trt:
        try:
            import tensorrt as trt  # noqa: F811
        except ImportError:
            print("TensorRT not installed (pip install tensorrt): skipping it")
    compiled = None
    if not args.no_compile:
        def pipeline(c, s):
            with torch.no_grad():
                return torch.softmax(model(features(c, s)), 1)
        compiled = torch.compile(pipeline, dynamic=False)

    info = {"device": torch.cuda.get_device_name(), "torch": torch.__version__,
            "tensorrt": getattr(trt, "__version__", None) if trt else None,
            "batches": {}}
    print(f"GPU: {info['device']}; PyTorch {info['torch']}; TensorRT {info['tensorrt']}")

    for batch in args.batches:
        codes, stm = all_codes[:batch].contiguous(), all_stm[:batch].contiguous()
        methods = {}
        for v in engine.variants:
            methods[v] = lambda v=v: make_ours(engine, v, codes, stm)
        methods["pytorch"] = lambda: make_pytorch(model, codes, stm)
        methods["pytorch-fp16"] = lambda: make_pytorch(
            load(str(args.model / "warpmate_net.pt"), "cuda"), codes, stm, half=True)
        if compiled is not None:
            methods["torch.compile"] = lambda: make_compiled(compiled, codes, stm)
        if trt is not None:
            for kind in ("dense", "sparse"):
                for fp16 in (False, True):
                    name = f"trt-{kind}" + ("-fp16" if fp16 else "")
                    methods[name] = lambda k=kind, h=fp16: make_trt(trt, model, k, h, codes, stm)

        print(f"\n== Batch {batch:,}")
        print(f"  {'method':16} {'eager us':>10} {'graph us':>10} {'boards/s (graph)':>17} "
              f"{'vs pytorch':>11} {'max diff':>9}")
        reference = make_pytorch(model, codes, stm)[0]().clone()
        rows = {}
        for name, make in methods.items():
            try:
                start = time.time()
                fn, result = make()
                fn()
                torch.cuda.synchronize()
                diff = float((result() - reference).abs().max())
                eager = time_eager(fn, args.seconds)
                graph = time_graph(fn, args.seconds)
            except Exception as e:  # noqa: BLE001 - one broken method shouldn't stop the run
                print(f"  {name:16} failed: {str(e).splitlines()[0][:120]}")
                torch.cuda.synchronize()
                continue
            rows[name] = {"eager_ms": eager, "graph_ms": graph, "max_prob_diff": diff,
                          "setup_s": time.time() - start}
        base = rows.get("pytorch", {}).get("eager_ms")
        for name, r in rows.items():
            best = r["graph_ms"] if r["graph_ms"] is not None else r["eager_ms"]
            g = f"{r['graph_ms'] * 1000:10.1f}" if r["graph_ms"] is not None else f"{'n/a':>10}"
            speed = f"{base / r['eager_ms']:10.2f}x" if base else f"{'':>11}"
            print(f"  {name:16} {r['eager_ms'] * 1000:10.1f} {g} {batch / best * 1000:17,.0f} "
                  f"{speed} {r['max_prob_diff']:9.1e}")
        info["batches"][str(batch)] = rows

    print("\n'vs pytorch' compares eager times (both called from Python). 'max diff' is the "
          "largest probability difference from 32-bit PyTorch.")

    if args.profile:
        for batch in (32, max(args.batches)):
            text = profile(engine, model, all_codes, all_stm, batch)
            print(text)
            info.setdefault("profile", {})[str(batch)] = text

    if args.out:
        args.out.write_text(json.dumps(info, indent=2))
        print(f"\nSaved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
