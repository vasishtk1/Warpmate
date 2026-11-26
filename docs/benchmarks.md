# Benchmarks

Measured results only. Every solver must produce an identical table; the benchmark tool checks this on every run.

All measurements below are on the same machine unless stated otherwise.

**Machine:** Google Colab, NVIDIA Tesla T4 (40 SMs, 15 GB), Intel Xeon @ 2.00 GHz (2 threads), CUDA 13.0

## Summary

| Comparison | Result |
|---|---|
| GPU frontier vs GPU full scan, 4-piece endgames | **2.7× to 6.1× faster** |
| All five 4-piece endgames on the GPU | **15.7 s → 4.2 s** |
| Best GPU vs best CPU, KQvKR (frontier on both) | **88× faster** (1.49 s vs 131 s) |
| CPU frontier vs CPU full scan, KQvKR | 4.5× faster (131 s vs 588 s) |

The two algorithms are described in [solver.md](solver.md#frontier-solver): **full scan** checks every unsolved position in every round; **frontier** only checks positions that can change in that round.

## GPU: frontier vs full scan

Measured with `warpmate-bench <endgame> --cpu-trials 0 --gpu-trials 3`: one untimed warm-up, then 3 timed runs of each algorithm (median shown). Only the named table is timed; the smaller tables it depends on are solved first, untimed. *Total* includes GPU memory setup and copying the finished table back; *kernels* is GPU time measured with CUDA events.

| Endgame | Full scan, total | Frontier, total | Speedup (total) | Speedup (kernels) | Positions checked: full scan | Positions checked: frontier |
|---|---|---|---|---|---|---|
| KQvKR | 3,819.7 ms | 1,436.7 ms | **2.7×** | 2.7× | 680,952,568 | 79,235,144 |
| KRvKR | 2,422.0 ms | 639.1 ms | **3.8×** | 4.0× | 741,743,776 | 42,991,352 |
| KRvKB | 3,634.3 ms | 599.9 ms | **6.1×** | 6.3× | 1,190,943,592 | 31,120,568 |
| KRvKN | 4,043.5 ms | 746.1 ms | **5.4×** | 5.6× | 1,548,952,752 | 36,882,784 |
| KRvKP | 3,151.0 ms | 1,010.6 ms | **3.1×** | 3.6× | 647,691,796 | 40,413,892 |

- Every frontier table was identical to the full-scan table.
- *Positions checked* counts how many times the expensive per-position step ran, over all rounds.
- The speedup is smaller than the drop in positions checked, for two reasons (see the profile below): every round still launches one thread per position to find the marked ones, and in even ("loss") rounds many marked positions still have an escape, so they're checked and stay unsolved. In KQvKR's round 18, for example, the frontier checked 7,104,820 positions and solved 184,740.

### Every 4-piece endgame, one run each

From `warpmate-gen --gpu KRvKP`, which solves all five 4-piece endgames plus the 3-piece ones they need. Each time includes copying the table back to the CPU.

| Endgame | Rounds | Full scan (step 6b) | Frontier (step 6c) |
|---|---|---|---|
| KQvKR | 71 | 3.68 s | 1.42 s |
| KRvKR | 39 | 2.26 s | 0.61 s |
| KRvKB | 59 | 3.33 s | 0.56 s |
| KRvKN | 81 | 3.62 s | 0.69 s |
| KRvKP | 86 | 2.78 s | 0.92 s |
| **Total** | | **15.7 s** | **4.2 s** |

## CPU vs GPU: KQvKR

`warpmate-bench KQvKR --cpu-trials 1 --gpu-trials 3`. The CPU ran once per algorithm because a run takes minutes; the GPU had a warm-up and 3 timed runs.

| Algorithm | CPU, 2 threads | GPU, total (median) | GPU speedup |
|---|---|---|---|
| Full scan (step 6b) | 588,046.9 ms | 3,667.9 ms | 160× |
| Frontier (step 6c) | 131,432.7 ms | 1,489.5 ms | **88×** |

- The fair comparison is the best version on each: **88×**. The CPU gains more from the frontier algorithm (4.5×) than the GPU (2.7×, measured in the same run as the frontier CPU), because the GPU still launches a thread per position every round.
- Separate runs of the GPU full scan on the same machine measured 3,667.9 ms, 3,819.7 ms and 3,970.5 ms (median of 3 each), so run-to-run variation on Colab is a few percent.

## KRvK with the current solver

`warpmate-bench KRvK --cpu-trials 5` with each CPU setting; GPU numbers from the same session (median of 5 runs, after a warm-up).

| Solver | Median (ms) | Best (ms) |
|---|---|---|
| CPU full scan, 1 thread | 2,153.4 | 2,145.3 |
| CPU full scan, 2 threads | 2,277.5 | 2,218.0 |
| CPU frontier, 1 thread | 578.7 | 548.8 |
| CPU frontier, 2 threads | 556.2 | 548.2 |
| GPU full scan, total | 22.3 | 22.0 |
| **GPU frontier, total** | **14.0** | **13.9** |

- Best GPU vs best CPU: **about 40×** (14.0 ms vs 556.2 ms). KRvK has only 524,288 positions, too few to keep the GPU busy, so the gap is smaller than for 4-piece endgames.
- The current solver is slower on KRvK than the project's first, KRvK-only solver (below: 616 ms on 1 CPU thread, 8.1 ms on the GPU). It handles any material, so each position costs more work. The first solver was removed because the general one produces the identical table.

## Where the GPU time goes now (Nsight Systems)

Profile of `warpmate-gen --gpu KQvKR` with the frontier solver. It also solves KQvK and KRvK first, so the counts include those two small tables.

| GPU activity | Launches | Total time | Share of kernel time |
|---|---|---|---|
| `frontier_step_kernel` (one round) | 125 | 1,180.4 ms | 80.1% |
| `prepare_kernel` (capture/promotion rounds, first frontier) | 3 | 112.7 ms | 7.6% |
| `mark_parents_kernel` (walk back from newly solved positions) | 125 | 98.2 ms | 6.7% |
| `init_kernel` (round 0) | 3 | 82.1 ms | 5.6% |
| `finalize_kernel` | 3 | 0.6 ms | 0.0% |
| Memory copies, GPU to CPU | 131 | 6.7 ms | (outside kernels) |

- **The round kernel is still most of the cost.** It launches one thread per position to find the marked ones; the next step would be to launch threads only for marked positions, and to avoid re-checking every move of positions that still have an escape.
- `cudaMemcpy` shows up as 99% of CUDA API time (1.48 s), but that's the CPU waiting for each round's kernels to finish before reading the round's counters. The data actually copied took 6.7 ms.

## Caveats

- Colab's CPU is weak: its 2 threads share one physical core, so 2 threads are barely faster than 1 (sometimes slower). Speedups over the CPU are real for this machine but would be smaller against a strong multi-core CPU.
- Final benchmarks are planned on a rented cloud GPU paired with a strong CPU.

## History: the first KRvK-only solver (GPU v1)

Measured with the project's first solvers, which were written only for KRvK (median of 5 runs). They have since been replaced by the general solver (re-measured above). The GPU gets one untimed warm-up run first, so its one-off start-up cost (about 180 ms, mostly creating the CUDA context) isn't counted as solving time.

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

For reference, on a different machine (Apple M1, 4 performance + 4 efficiency cores) that first CPU solver took about 410 ms on 1 thread and 110 ms on 8 threads (3 trials).

### Where the GPU time went (Nsight Systems)

Profile of one KRvK solve with the first GPU solver on the T4:

| GPU activity | Count | Total time | Share |
|---|---|---|---|
| `step_kernel` (one solving round) | 33 | 6.60 ms | 97.9% |
| `init_kernel` | 1 | 0.13 ms | 1.9% |
| `finalize_kernel` | 1 | 0.01 ms | 0.2% |
| Memory copies, GPU to CPU | 35 | 0.10 ms | (outside kernels) |

- **The solving rounds were the whole cost.** Each round checked all 524,288 positions even though most were already solved. This led to the frontier solver above.
- **Data transfer was negligible:** the table stays on the GPU for the whole solve; each round copies back only a small counter, and the table is copied once at the end.

## CUDA basics (`examples/cuda_basics`)

Same Colab T4 machine, single runs:

| Task | CPU loop | GPU kernel | GPU copies (in + out) |
|---|---|---|---|
| Add two arrays of 1M floats | 0.90 ms | 0.25 ms | 3.01 + 1.17 ms |
| Square 16M floats | 11.20 ms | 0.54 ms | not measured |

For simple element-wise work the kernel is fast, but copying data to and from the GPU can cost far more than the computation. This is why the solver keeps its table on the GPU throughout.
