"""Running normalization of value targets (as in MAPPO)."""

from __future__ import annotations

import torch
from torch import nn


class ValueNorm(nn.Module):
    """Running mean/variance (Chan et al. parallel update) of return targets.

    The critic predicts normalized values; ``denormalize`` maps them back to
    return units. Stats live in buffers so they are saved in checkpoints.
    """

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", torch.zeros((), dtype=torch.float64))
        self.register_buffer("var", torch.ones((), dtype=torch.float64))
        self.register_buffer("count", torch.zeros((), dtype=torch.float64))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.detach().to(torch.float64).reshape(-1)
        if x.numel() == 0:
            return
        if not torch.isfinite(x).all():
            raise ValueError("non-finite value target")
        batch_mean, batch_var, n = x.mean(), x.var(unbiased=False), float(x.numel())
        total = self.count + n
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * n / total
        m2 = self.var * self.count + batch_var * n + delta.pow(2) * self.count * n / total
        self.mean.copy_(new_mean)
        self.var.copy_(m2 / total)
        self.count.copy_(total)

    def _std(self) -> torch.Tensor:
        return torch.sqrt(self.var + self.eps)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mean.to(x.dtype)) / self._std().to(x.dtype)).to(x.dtype)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self._std().to(x.dtype) + self.mean.to(x.dtype)).to(x.dtype)
