# Warpmate docs

Technical documentation for Warpmate. For the project overview, see the [main README](../README.md).

| Doc | What it covers |
|---|---|
| [solver.md](solver.md) | Board representation, position indexing, value encoding, the retrograde algorithm, why it parallelizes, the frontier solver, and correctness checks |
| [verification.md](verification.md) | Results of checking every table (all 34 3- and 4-piece endgames) against the Syzygy tablebases |
| [network.md](network.md) | The neural network: training data, model, training curve, accuracy (overall, per endgame, move choice) and 5-piece generalization |
| [inference.md](inference.md) | Custom CUDA inference: the four kernel versions (naive, fused, sparse, INT8), memory layout, 8-bit quantization, correctness checks and benchmark method (results pending the Colab run) |
| [benchmarks.md](benchmarks.md) | CPU vs GPU and full scan vs frontier timings, methodology, caveats, and Nsight profiles |

New docs (GPU design, benchmarks, verification results, model accuracy) are added to this table as the project develops.
