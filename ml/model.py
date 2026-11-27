"""The Warpmate network: a small multilayer perceptron that predicts
win / draw / loss for the side to move.

Input (769 numbers): 12 boards of 64 squares, one per piece kind (white king,
white queen, ..., black pawn), with a 1 where that piece stands and 0
elsewhere, plus one number for the side to move (1 = White).

    769 -> Linear -> 256 -> ReLU -> Linear -> 64 -> ReLU -> Linear -> 3

The 3 outputs are raw scores ("logits") for loss, draw, win. Softmax turns
them into percentages; training uses cross-entropy, which works on the
logits directly.
"""

from __future__ import annotations

import torch
from torch import nn

N_INPUTS = 769       # 12 * 64 piece squares + side to move
PAD = 769            # piece code meaning "no piece" (dropped below)
CLASSES = ("loss", "draw", "win")  # for the side to move


class WarpmateNet(nn.Module):
    def __init__(self, hidden: tuple[int, ...] = (256, 64)):
        super().__init__()
        self.hidden = tuple(hidden)
        layers: list[nn.Module] = []
        width = N_INPUTS
        for h in self.hidden:
            layers += [nn.Linear(width, h), nn.ReLU()]
            width = h
        layers.append(nn.Linear(width, len(CLASSES)))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def features(codes: torch.Tensor, stm: torch.Tensor) -> torch.Tensor:
    """Piece codes (B, 5) and side to move (B,) -> network input (B, 769).

    Each code is (colour*6 + type)*64 + square, i.e. the input position of a 1.
    `scatter_` writes all the 1s for a whole batch in one GPU operation. Empty
    slots hold PAD, which lands in a spare column that is then dropped.
    """
    batch = codes.shape[0]
    x = torch.zeros(batch, N_INPUTS + 1, device=codes.device)
    x.scatter_(1, codes.long(), 1.0)
    x[:, 768] = (stm == 0).float()  # 1 if White is to move
    return x[:, :N_INPUTS]


def save(model: WarpmateNet, path: str) -> None:
    torch.save({"hidden": model.hidden, "state_dict": model.state_dict()}, path)


def load(path: str, device: str = "cpu") -> WarpmateNet:
    checkpoint = torch.load(path, map_location=device)
    model = WarpmateNet(tuple(checkpoint["hidden"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()
