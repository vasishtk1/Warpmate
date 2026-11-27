# The neural network

A small network trained on Warpmate's solved tables to predict win / draw / loss for the side to move, measured against the tables' perfect answers. Code in `ml/`; one Colab notebook ([`notebooks/phase3_network.ipynb`](../notebooks/phase3_network.ipynb)) runs everything below.

All numbers are from one run on Google Colab (Tesla T4), 2026-11-27.

## Training data

1. **Solve every 3- and 4-piece endgame** on the GPU: the 5 three-piece endgames and 29 of the 30 four-piece ones (`warpmate-gen --gpu --all3 --all4`). KPvKP is left out because pawns on both sides need en passant, which the move generator doesn't handle yet. The 29 four-piece tables solve in 18.0 s in total.
2. **Check every table against Syzygy**: a random sample of 200,000 rows from each of the 34 tables, 4,358,127 legal positions in total, **0 mismatches** ([verification.md](verification.md)).
3. **Build the dataset** (`ml/dataset.py`): up to 1,000,000 legal positions per table (all of them for 3-piece tables), sampled at random.
   - In endgames with two identical pieces (KQQvK, KBBvK, KNNvK, KRRvK, KPPvK), every position appears twice in the table, once per order of the two pieces; only one copy is kept, so no position can be in both the training and the test set.
   - **30,945,584 examples**: 27,852,451 for training, 3,093,133 (10%) held out for testing.
   - Labels across the dataset: 32.8% loss, 36.0% draw, 31.2% win for the side to move.
   - 2,000 labels per table (68,000 in total) were re-checked against Syzygy: 0 mismatches.

## Model

```
769 inputs -> 256 (ReLU) -> 64 (ReLU) -> 3 outputs (softmax: loss / draw / win)
```

| Layer | Shape | Weights |
|---|---|---|
| 1 | 769 → 256 | 197,120 |
| 2 | 256 → 64 | 16,448 |
| 3 | 64 → 3 | 195 |
| **Total** | | **213,763** (0.86 MB as 32-bit floats) |

**Input encoding:** 12 boards of 64 squares, one per piece kind (white king, white queen, ..., black pawn), with a 1 where that piece stands, plus one input for the side to move. A 4-piece position switches on 4 or 5 of the 769 inputs. Positions are always shown to the network with the stronger side as White, exactly as the tables store them (a position where Black is stronger is colour-flipped and mirrored first). Inputs are stored compactly (one 16-bit code per piece) and expanded to 769 numbers on the GPU during training.

## Training

PyTorch on the T4: cross-entropy loss, Adam (learning rate 0.001), batches of 4,096, 10 passes (epochs) over the training set. The whole dataset sits in GPU memory. Each epoch took about 12 seconds; training took about 2 minutes in total.

| Epoch | Train loss | Test loss | Test accuracy |
|---|---|---|---|
| 1 | 0.1515 | 0.0944 | 96.65% |
| 2 | 0.0819 | 0.0753 | 97.29% |
| 3 | 0.0696 | 0.0674 | 97.59% |
| 4 | 0.0632 | 0.0634 | 97.71% |
| 5 | 0.0591 | 0.0604 | 97.80% |
| 6 | 0.0562 | 0.0582 | 97.89% |
| 7 | 0.0539 | 0.0570 | 97.94% |
| 8 | 0.0521 | 0.0563 | 97.94% |
| 9 | 0.0506 | 0.0539 | 98.04% |
| 10 | 0.0494 | 0.0531 | 98.06% |

Test loss falls along with training loss throughout, so the network is learning patterns rather than memorizing positions.

## Results

### Accuracy on held-out positions

**98.06%** on 3,093,133 test positions. Always guessing the most common answer ("draw") scores 35.97%.

| True result ↓ / guess → | Loss | Draw | Win | Accuracy |
|---|---|---|---|---|
| **Loss** | 1,001,554 | 13,252 | 1,203 | 98.58% |
| **Draw** | 16,610 | 1,079,136 | 16,921 | 96.99% |
| **Win** | 859 | 11,182 | 952,416 | 98.75% |

Almost all mistakes involve draws; a win is mistaken for a loss (or the reverse) in only 2,062 of 3.09 million positions.

### Per endgame

The baseline here is "always guess this endgame's most common answer". Hardest first:

| Endgame | Accuracy | Baseline | | Endgame | Accuracy | Baseline |
|---|---|---|---|---|---|---|
| KRvKN | 91.15% | 72.00% | | KQvKB | 99.07% | 43.96% |
| KRvKP | 94.97% | 49.81% | | KQBvK | 99.27% | 57.99% |
| KQvKQ | 94.98% | 57.81% | | KQPvK | 99.40% | 59.32% |
| KNvKP | 95.29% | 76.85% | | KQRvK | 99.44% | 65.70% |
| KRvKB | 95.44% | 81.56% | | KQNvK | 99.44% | 56.34% |
| KRvKR | 96.19% | 70.35% | | KRBvK | 99.47% | 54.25% |
| KQvKR | 96.54% | 60.50% | | KRPvK | 99.59% | 55.07% |
| KNPvK | 96.68% | 45.37% | | KBBvK | 99.64% | 55.10% |
| KPvK | 96.83% | 37.71% | | KRNvK | 99.65% | 52.87% |
| KQvKP | 97.02% | 47.73% | | KRRvK | 99.86% | 61.70% |
| KBvKP | 97.13% | 85.15% | | KQvK | 99.87% | 54.62% |
| KBPvK | 97.62% | 45.18% | | KRvK | 99.94% | 50.62% |
| KBNvK | 98.02% | 45.60% | | KNNvK | 99.99% | 99.99% |
| KQvKN | 98.08% | 47.16% | | KBvK | 99.99% | 100.00% |
| KPPvK | 98.39% | 47.72% | | KBvKB | 99.99% | 100.00% |
| KQQvK | 98.79% | 68.92% | | KNvKN | 100.00% | 100.00% |
| | | | | KBvKN | 100.00% | 100.00% |
| | | | | KNvK | 100.00% | 100.00% |

- The hardest endgames are the balanced ones with many drawn-looking positions that are actually won (or the reverse): rook against knight, rook against pawn, queen against queen.
- In endgames that are (almost) always drawn (KNNvK, KBvK, KBvKB, KNvKN, KBvKN, KNvK), the baseline is already 99.99 to 100%, so these say nothing about the network; it matches the baseline but doesn't beat it.

### Move choice

For 19,950 winning or drawn test positions, the network scored every legal move (by evaluating the position after it) and picked its favourite. The tables then said whether that move keeps the best possible result. Losing positions are left out, since every move there loses.

| Positions | Favourite move keeps the best result |
|---|---|
| Winning (9,272) | 98.88% keep the win |
| Drawn (10,678) | 99.44% keep the draw |
| **All (19,950)** | **99.18%** |

This only measures whether the result is kept, not whether the move is the fastest win.

### Generalization to 5 pieces

The network never saw a 5-piece position. For 5 endgames, 20,000 random legal positions each were labelled with the Syzygy 5-piece tables:

| Endgame | Accuracy | Baseline | True loss / draw / win |
|---|---|---|---|
| KQRvKR | **87.59%** | 49.86% | 49.9% / 10.5% / 39.6% |
| KRPvKR | 53.65% | 43.73% | 12.9% / 43.7% / 43.4% |
| KQPvKQ | 53.59% | 51.33% | 7.2% / 41.5% / 51.3% |
| KRNvKR | 66.84% | **78.98%** | 1.8% / 79.0% / 19.2% |
| KRBvKR | 65.36% | **78.05%** | 2.4% / 78.0% / 19.6% |

Generalization is limited, and mixed:

- It transfers well where an extra piece decides the game (**KQRvKR: 87.6%** against a 49.9% baseline).
- It is only slightly better than the baseline in pawn endgames (KRPvKR, KQPvKQ), where the result depends on exact pawn and king placement.
- It is **worse than always guessing "draw"** in KRBvKR and KRNvKR. These are mostly draws (about 79%), but an extra minor piece looks like a winning advantage to a network that learned from 4-piece endgames such as KRBvK, where the stronger side nearly always wins.

So the network has learned the endgames it was trained on very well (98.06%), but it has not learned general chess principles that carry over to bigger endgames.

## Files

`ml/train.py` writes, and the dashboard and the planned CUDA inference code read:

| File | Contents |
|---|---|
| `warpmate_net.pt` | the weights, for PyTorch |
| `warpmate_net.npz` | the same weights as plain arrays (`w0`, `b0`, `w1`, `b1`, `w2`, `b2`; each `w` is (outputs, inputs)) |
| `training_log.json` | loss and accuracy per epoch |
| `results.json` | everything in the Results section above |

These are generated files and aren't committed; the notebook rebuilds them.
