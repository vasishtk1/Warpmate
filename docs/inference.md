# Custom CUDA inference

Warpmate runs its trained network ([network.md](network.md)) with its own CUDA kernels instead of PyTorch. The code is in `src/inference/`. One Colab notebook ([`notebooks/phase4_inference.ipynb`](../notebooks/phase4_inference.ipynb)) builds it, checks it against PyTorch, and benchmarks it against PyTorch, `torch.compile` and TensorRT.

> **Status:** the code and checks are written. Correctness and speed results will be filled in from the Colab run; no numbers are published here until they are measured.

## The network, from the GPU's point of view

```
769 inputs -> 256 (ReLU) -> 64 (ReLU) -> 3 logits -> softmax (loss / draw / win)
```

Boards arrive in the dataset's compact form instead of 769 numbers: up to 32 16-bit **piece codes** per board, where code = (colour × 6 + type) × 64 + square is the index of that piece's input, plus the **side to move** (input 768). A 4-piece position switches on only 4 or 5 of the 769 inputs.

Where the work is, per board:

| Layer | Dense multiply-adds | What the sparse version does |
|---|---|---|
| 1 (769 → 256) | 196,864 | adds 3 to 6 rows of 256 weights: ~1,300 additions |
| 2 (256 → 64) | 16,384 | same (dense) |
| 3 (64 → 3) | 192 | same (dense) |

So done densely, layer 1 is 92% of the work and almost all of it is multiplying by zero.

## The versions

Each version keeps everything the previous one does and adds one idea. They all take the same input (codes and side to move, already on the GPU) and write the same output (probabilities, and optionally logits).

| Version | Kernels per batch | Idea |
|---|---|---|
| `naive` | 5 | One kernel per step: expand codes to 769 floats, layer 1, layer 2, layer 3, softmax. One thread per output number; results pass through GPU memory. |
| `fused` | 1 | The whole network in one kernel. A block of 256 threads handles 8 boards from start to finish, keeping the hidden layers in shared memory (on-chip). Layer 1 still multiplies all 769 inputs. |
| `sparse` | 1 | As `fused`, but layer 1 adds up the weight rows of the inputs that are on. 16 boards per block. |
| `int8` | 1 | As `sparse`, with 8-bit weights and 8-bit integer math for layers 1 and 2 (`__dp4a`). |

### Memory layout

- **Layer 1 weights are stored input-major** (`w1t[input][neuron]`, 769 × 256). The sparse layer reads one contiguous 1 KB row per piece, and the 32 threads of a warp (neighbouring neurons) read 32 neighbouring floats: one coalesced memory transaction.
- **Layer 2 weights are also input-major** (256 × 64), for the same reason.
- At 0.79 MB (32-bit) or 0.2 MB (8-bit), the layer 1 table fits in the T4's 4 MB L2 cache, so the rows that pieces keep hitting are served from cache.

### Work split inside a block (`fused`, `sparse`, `int8`)

- **Codes** for the block's boards are copied into shared memory first.
- **Layer 1:** thread *j* computes neuron *j* for every board in the block, keeping one total per board in registers. Each weight is read from memory once and used for all of the boards.
- **Layer 2:** the 256 threads form 4 groups of 64. Thread (*g*, *j*) computes neuron *j* for boards *g*, *g* + 4, *g* + 8, … The h1 values come from shared memory, and since a whole warp reads the same one, it is broadcast.
- **Layer 3 and softmax:** one warp per board. Each lane multiplies 2 of the 64 inputs, and 5 rounds of warp shuffles (`__shfl_down_sync`) add up the 32 partial sums without touching memory.
- **Barriers:** `__syncthreads()` separates the layers. No thread returns early, so boards past the end of the batch are computed as empty boards and simply not written.

### 8-bit version

`ml/quantize.py` makes it from the trained weights:

- **Weights:** each neuron gets its own scale, *s* = max |weight| / 127, and weights are stored as *q* = round(weight / *s*), in −127…127.
- **Layer 1:** *h1* = ReLU(*s1* × Σ *q1* rows + *b1*). The sum is in exact integer arithmetic.
- ***h1* in 8 bits:** *h1q* = min(127, round(*h1* / *a1*)). The scale *a1* is calibrated on 200,000 training positions (never the test set). Candidates (the 99.9th, 99.99th and 99.999th percentiles and the maximum of *h1*) are compared on another 200,000 training positions, and the one whose picks agree most with the 32-bit network is kept.
- **Layer 2:** *h2* = ReLU(*m2* × (*q2* · *h1q*) + *b2*) with *m2* = *s2* × *a1*. Here `__dp4a` multiplies four pairs of 8-bit numbers and adds them to a 32-bit total in one instruction.
- **Layer 3** stays 32-bit, since it is only 64 × 3.

The scaling steps use `__fmul_rn` / `__fadd_rn`, which stops the compiler from fusing them into one FMA instruction (one rounding instead of two). The 8-bit kernel therefore rounds exactly like the NumPy reference and the C++ CPU reference (which is built with `-ffp-contract=off`), and its *h1q* values can be checked bit for bit.

## Correctness checks

1. **`ctest` (`tests/test_inference.cpp`):** needs no trained model; it uses made-up weights.
   - The CPU's sparse layer 1 matches a plain dense double-precision calculation, for both 32-bit and 8-bit weights.
   - Every GPU version matches the CPU reference across awkward batch sizes (1, 7, 8, 9, 100, 5,000) and slot counts (1, 5, 8, 32), including padding with out-of-range codes.
2. **`ml/check_inference.py`:** compares against PyTorch on the 3,093,133 test positions.
   - **32-bit versions** (`cpu`, `naive`, `fused`, `sparse`): every probability within 1e-4 of PyTorch's, and the same win/draw/loss pick on every board except near-ties (top two probabilities within 1e-4).
   - **8-bit versions:** within 1e-5 of the NumPy implementation of the same 8-bit math, with an accuracy drop of less than 1 point compared with PyTorch. The exact drop is reported.
   - **Padding and batch size** (extra padding slots, batches of 1, 7, 9, 31, …) must not change a single bit. A different slot order must stay within 1e-5, since the same numbers are added in a different order.
3. **Dashboard test:** boards encoded by the dashboard give the same answers through the inference library as through PyTorch.

Results: *pending the Colab run.*

## Benchmark method

`ml/bench_inference.py`, all on one GPU:

- **Same start and finish for every method:** compact boards already in GPU memory in, probabilities in GPU memory out. Input preparation (expanding to 769 numbers, or the conversions TensorRT needs) is counted for every method that needs it.
- **Methods:**
  - our 4 versions
  - PyTorch, 32-bit and 16-bit (the Phase 3 code)
  - `torch.compile`
  - TensorRT on the dense 769-number input, 32-bit and 16-bit
  - TensorRT given the same sparse trick (Gather + ReduceSum on the weight rows), 32-bit and 16-bit, so the comparison isn't only sparse vs dense
- **Batch sizes:** 1, 32 (one position's legal moves, what the dashboard sends), 256, 4,096 and 65,536. TensorRT engines are built separately for each batch size, so TensorRT can tune for every one.
- **Two timings**, with CUDA events after warm-up:
  - **eager:** called from Python in a loop. This includes Python and kernel-launch overhead, which dominates small batches, and is what a Python program actually pays.
  - **graph:** recorded once as a CUDA graph and replayed, which is close to pure GPU time.
- **Output check:** every method's output is compared with 32-bit PyTorch, so nothing is timed that gives wrong answers.
- **Per-kernel profile:** PyTorch's profiler, which sees every kernel in the process including ours.

Results: *pending the Colab run.*

## Files

| File | What it is |
|---|---|
| `src/inference/npz.{h,cpp}` | Reads NumPy `.npy` / `.npz` files (zip64 included) |
| `src/inference/network.{h,cpp}` | Weight layouts, loaders, CPU reference (32-bit and 8-bit) |
| `src/inference/infer_gpu.{h,cu}` | The CUDA kernels and the `GpuEngine` that owns the GPU memory |
| `src/inference/c_api.{h,cpp}` | Plain C interface; built as `libwarpmate_infer` |
| `ml/inference.py` | Python access through ctypes: NumPy arrays or torch CUDA tensors |
| `ml/quantize.py` | Makes `warpmate_net_int8.npz`; NumPy reference of the 8-bit math |
| `ml/check_inference.py` | Correctness against PyTorch |
| `ml/bench_inference.py` | Benchmarks and profile |

The dashboard server uses the `sparse` version when the library was built with CUDA and a GPU is present, and falls back to PyTorch on the CPU otherwise (`WARPMATE_BACKEND=cuda|torch` forces one).
