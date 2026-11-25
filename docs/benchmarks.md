# Benchmarks

Measured results only. Every solver must produce an identical table; the benchmark tool checks this on every run.

## KRvK solver: CPU vs GPU (GPU v1)

Measured with `warpmate-bench --trials 5` (median of 5 runs). The GPU gets one untimed warm-up run first, so its one-off start-up cost (about 180 ms, mostly creating the CUDA context) isn't counted as solving time.

**Machine:** Google Colab, NVIDIA Tesla T4 (40 SMs, 15 GB), Intel Xeon @ 2.00 GHz (2 threads), CUDA 13.0

| Solver | Median (ms) | Best (ms) | Speedup vs 1 CPU thread |
|---|---|---|---|
| CPU, 1 thread | 615.9 | 597.9 | 1.0× |
| CPU, 2 threads | 536.3 | 524.6 | 1.1× |
| **GPU, total** | **8.1** | **8.0** | **76×** |
| GPU, kernels only | 7.7 | 7.7 | 80× |

- *GPU, total* includes GPU memory allocation and copying the finished table back to the CPU.
- *GPU, kernels only* is time on the GPU, measured with CUDA events.
- All solvers produced identical tables.

**Against the best CPU result on the same machine, the GPU is about 66× faster** (8.1 ms vs 536 ms).

### Caveats

- Colab's CPU is weak: its 2 threads share one physical core, which is why 2 threads are only 1.1× faster than 1. The speedup is real for this machine but would be smaller against a strong multi-core CPU.
- For reference, on a different machine (Apple M1, 4 performance + 4 efficiency cores) the CPU solver takes about 410 ms on 1 thread and 110 ms on 8 threads (3 trials). Different hardware, so not directly comparable with the GPU numbers above.
- Final benchmarks are planned on a rented cloud GPU paired with a strong CPU.

### Where the GPU time goes (Nsight Systems)

Profile of one `warpmate-solve --gpu` run on the T4:

| GPU activity | Count | Total time | Share |
|---|---|---|---|
| `step_kernel` (one solving round) | 33 | 6.60 ms | 97.9% |
| `init_kernel` | 1 | 0.13 ms | 1.9% |
| `finalize_kernel` | 1 | 0.01 ms | 0.2% |
| Memory copies, device → host | 35 | 0.10 ms | (outside kernels) |

- **The solving rounds are the whole cost.** Each round checks all 524,288 positions even though most are already solved, so most threads exit immediately. Only revisiting positions near newly solved ones is the planned next optimization.
- **Data transfer is negligible:** the table stays on the GPU for the whole solve; each round copies back only a 4-byte counter, and the 0.5 MB table is copied once at the end.

## CUDA basics (`examples/cuda_basics`)

Same Colab T4 machine, single runs:

| Task | CPU loop | GPU kernel | GPU copies (in + out) |
|---|---|---|---|
| Add two arrays of 1M floats | 0.90 ms | 0.25 ms | 3.01 + 1.17 ms |
| Square 16M floats | 11.20 ms | 0.54 ms | not measured |

For simple element-wise work the kernel is fast, but copying data to and from the GPU can cost far more than the computation. This is why the solver keeps its table on the GPU throughout.
