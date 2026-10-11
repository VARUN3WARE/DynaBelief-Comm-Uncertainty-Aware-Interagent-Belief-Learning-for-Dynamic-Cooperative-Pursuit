"""Belief heads on the actor's post-message hidden state (losses: ``beliefs/losses.py``)."""

from __future__ import annotations

import torch
from torch import nn

from dynabelief.models.encoder import init_layer


class PointBeliefHead(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            init_layer(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            init_layer(nn.Linear(hidden_dim, 2), gain=0.1),
            nn.Sigmoid(),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


class MapBeliefHead(nn.Module):
    """``hidden [..., H]`` -> occupancy logits ``[..., Y, X]``.

    The output bias starts at the prior logit, so an untrained map already reports the
    expected density instead of 0.5 everywhere.
    """

    def __init__(self, hidden_dim: int, grid_yx: tuple[int, int], prior_logit: float) -> None:
        super().__init__()
        self.grid_yx = tuple(grid_yx)
        cells = grid_yx[0] * grid_yx[1]
        self.hidden = nn.Sequential(init_layer(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU())
        self.out = init_layer(nn.Linear(hidden_dim, cells), gain=0.1)
        nn.init.constant_(self.out.bias, prior_logit)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.out(self.hidden(hidden)).reshape(*hidden.shape[:-1], *self.grid_yx)
