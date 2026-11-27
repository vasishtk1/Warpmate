# Warpmate: CUDA Chess Endgame Solver

Warpmate is a GPU pipeline built around small chess endgames:

1. **Solve** every position of an endgame (say king + rook vs king) with CUDA, producing perfect win/draw/loss labels and the exact distance to mate.
2. **Train** a small neural network on that perfectly labeled data to predict who wins from any position.
3. **Run** the network with custom CUDA inference kernels, benchmarked against PyTorch and TensorRT.
4. **Play** on a two-player dashboard: two people share one board while live win/draw/loss bars update after every move, next to the perfect answer.

> **Status:** early development. Features marked *planned* are not built yet.

## How it works

### Solving endgames (retrograde analysis)

The solver works backward from checkmate:

1. List every legal position for the chosen pieces.
2. Mark each checkmate as a loss for the side to move.
3. A position is a **win** if *any* move reaches a position lost for the opponent, and a **loss** if *every* move reaches a position won for the opponent.
4. Repeat until nothing changes. Every unmarked position is a **draw**.

Each round applies the same check to every position independently, which maps naturally onto the GPU: one CUDA thread per position, millions of positions per round.

### Learning from perfect data

Because the solver labels every position exactly, the network's accuracy can be measured against ground truth rather than estimated. During play, every legal next move is evaluated in a single batched GPU call, so the dashboard can rank moves and flag blunders in real time.

```mermaid
flowchart LR
    A[CUDA solver:<br/>label every position] --> B[(Tablebase)]
    B --> C[Train network<br/>PyTorch]
    B --> F
    C --> D[Custom CUDA<br/>inference kernels]
    D --> F[Two-player dashboard:<br/>live win/draw/loss bars,<br/>ranked moves, perfect answer]
```

## Features

- **CPU solver** in C++ (multithreaded), the reference the GPU is checked against. First case, king + rook vs king: solves all 524,288 positions and finds the known 16-move longest mate *(done)*
- **General solver** for any endgame of up to 4 pieces (all piece types, pawns and promotion), with linked tables for captures and promotions *(done)*
- **Probe tool** that looks up any position and prints the result, every move's outcome, and the line of perfect play *(done)*
- **CUDA solver** (one GPU thread per position): produces tables byte-for-byte identical to the CPU's *(done)*
- **4-piece endgames on the GPU**: KQvKR, KRvKR, KRvKB, KRvKN and KRvKP (33.5 million positions each), including the known 35-move longest mate in KQvKR *(done)*
- **Frontier solver**: instead of re-checking every position every round, walks backward (un-moves) from newly solved positions and checks only what can change. 2.7× to 6.1× faster on the GPU, all five 4-piece endgames in 4.2 s instead of 15.7 s on a T4, with identical tables. KQvKR is 88× faster on the GPU than on the CPU, both using the frontier algorithm *(done)*
- **Board-symmetry compression** (4 to 8× smaller tables) *(planned)*
- **Every 3- and 4-piece endgame** (34 tables; all but KPvKP, which needs en passant) solved on the GPU *(done)*
- **Verified against Syzygy**: all 1,945,584 legal positions of the 3-piece endgames, and random samples totalling 4,358,127 legal positions across all 34 tables, match the official tablebases with 0 mismatches *(done)*
- **Neural network** (769 → 256 → 64 → 3, 213,763 weights) trained on 27.9 million solver-labelled positions: **98.06%** win/draw/loss accuracy on 3.09 million held-out positions (baseline 35.97%), and its favourite move keeps the best result in 99.18% of positions. On 5-piece endgames it never saw, it ranges from 87.6% (KQRvKR) to worse than guessing "draw" (KRBvKR, KRNvKR) ([details](docs/network.md)) *(done)*
- **Custom CUDA inference**: batched evaluation, fused layers, and 8-bit quantization, compared with PyTorch and TensorRT *(planned)*
- **Two-player dashboard** (React + FastAPI, runs locally): set up any position or play one out with drag-and-drop moves, the network's win/draw/loss bars, the perfect answer, ranked moves, and blunder warnings *(in progress)*
- **Benchmarks** comparing CPU and GPU, full scan and frontier, with Nsight profiling *(done)*

## Tech stack

| Area | Tools |
|---|---|
| Solver | C++17, CUDA |
| Build | CMake |
| Verification | Python, [python-chess](https://python-chess.readthedocs.io/) with Syzygy tablebases |
| Machine learning | Python, PyTorch |
| Inference | CUDA C++, compared with PyTorch and TensorRT |
| Profiling | NVIDIA Nsight Systems / Nsight Compute |
| Dashboard | HTML, CSS, JavaScript |

## Project structure

```
Warpmate/
├── CMakeLists.txt         Build configuration
├── src/
│   ├── core/              Bitboards, piece attacks and stored values (CPU + GPU)
│   ├── egtb/              Solver: positions, moves, per-position step, linked tables, CPU driver
│   ├── gpu/               CUDA solver: kernels and GPU memory management
│   └── io/                Tablebase file format
├── tools/
│   ├── gen.cpp            warpmate-gen: solve any endgame (and its dependencies)
│   ├── probe.cpp          warpmate-probe: look up any solved position
│   └── bench.cpp          warpmate-bench: time CPU and GPU solvers and check they agree
├── tests/                 Unit, consistency, and CPU-vs-GPU tests
├── examples/
│   └── cuda_basics/       Introductory CUDA programs
├── ml/                    Neural network: dataset builder, model, training, evaluation (PyTorch)
├── dashboard/             Two-player analysis app: React page (web/) and FastAPI server (server/)
├── notebooks/             Colab notebooks to build and run on a free GPU
├── scripts/
│   └── verify_syzygy.py   Check a table against Syzygy
└── docs/                  Technical documentation
```

## Getting started

Requires CMake 3.20+ and a C++17 compiler. GPU support is built automatically when the CUDA toolkit (`nvcc`) is found; without it, everything else still builds and runs on the CPU.

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build          # run the tests

./build/warpmate-gen --all3     # solve every 3-piece endgame into tables/
./build/warpmate-probe "8/8/8/8/8/2k5/1R6/K7 w - - 0 1"

./build/warpmate-gen --gpu KQvKR   # solve on the GPU (CUDA builds only)
./build/warpmate-bench KQvKR       # time CPU and GPU, full scan vs frontier
```

No NVIDIA GPU? Open [`notebooks/step6c_frontier_gpu.ipynb`](notebooks/step6c_frontier_gpu.ipynb) in Google Colab with a free T4 GPU.

Example probe output (abridged):

```
Result: White mates in 16
Best move: Kb1
Perfect play: 1. Kb1 Kd4 2. Kc2 Kc4 3. Rb1 Kd4 ... 15. Kc7 Ka7 16. Ra1#
```

Probing a king + pawn ending shows perfect play crossing into another table when the pawn promotes:

```
$ ./build/warpmate-probe "3k4/8/3K4/3P4/8/8/8/8 w - - 0 1"
Result: White mates in 11
Perfect play: 1. Ke6 Kc7 2. d6+ Kb6 3. d7 Kc7 4. Ke7 Kb6 5. d8=Q+ Kc5 ... 11. Qa2#
```

### Dashboard

A local web app to set up or play out positions, with the network's guess next to the perfect answer. It needs the solved tables in `tables/` and, for real network output, trained weights in `model/warpmate_net.pt` (from the Phase 3 notebook); without them the bars are a labelled placeholder.

```bash
python3 -m venv .venv && .venv/bin/pip install -r dashboard/server/requirements.txt
(cd dashboard/web && npm install)
dashboard/run.sh                 # then open http://localhost:5173
(cd dashboard/server && ../../.venv/bin/python -m unittest)   # server tests
```

To verify against Syzygy, install the Python requirements and download the matching `.rtbw` files (see the instructions at the top of `scripts/verify_syzygy.py`), then:

```bash
pip install -r scripts/requirements.txt
python scripts/verify_syzygy.py --syzygy syzygy tables/*.wmtb
```

See [docs/solver.md](docs/solver.md) for how the solver works.

## Results

**Correctness:** all five 3-piece endgames (KQvK, KRvK, KBvK, KNvK, KPvK) match the Syzygy tablebases on every one of 1,945,584 legal positions, with 0 mismatches. Longest forced mates match known values (KQvK 10 moves, KRvK 16). See [docs/verification.md](docs/verification.md).

**Speed (KRvK, Colab Tesla T4):**

| Solver | Median time | Speedup vs 1 CPU thread |
|---|---|---|
| CPU, 1 thread | 616 ms | 1.0× |
| CPU, 2 threads | 536 ms | 1.1× |
| **GPU v1** | **8.1 ms** | **76×** |

The GPU table is byte-for-byte identical to the CPU table. Colab's CPU is weak (2 threads on one core), so the gap would be smaller against a strong multi-core CPU; see [docs/benchmarks.md](docs/benchmarks.md) for methodology, caveats, and the Nsight profile.

Model accuracy will be published here once measured.
