"""Convolutional encoders for local views (actor) and the global state (critic)."""

from __future__ import annotations

import math

import torch
from torch import nn


def init_layer(layer: nn.Module, gain: float = math.sqrt(2.0)) -> nn.Module:
    """Orthogonal weights, zero bias (standard for PPO)."""
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    return layer


def scale_counts(x: torch.Tensor) -> torch.Tensor:
    """Observation channels are agent counts (0..n); log1p keeps them O(1)."""
    return torch.log1p(x)


class LocalObsEncoder(nn.Module):
    """Shared CNN for one pursuer's local view.

    Input ``[..., R, R, C]`` (channels-last, as the env emits it), output
    ``[..., out_dim]``. Leading dims are flattened for the convolution and restored.
    """

    def __init__(self, obs_shape: tuple[int, int, int], conv_channels: int, out_dim: int) -> None:
        super().__init__()
        rows, cols, channels = obs_shape
        self.obs_shape = tuple(obs_shape)
        self.conv = nn.Sequential(
            init_layer(nn.Conv2d(channels, conv_channels, 3, padding=1)),
            nn.ReLU(),
            init_layer(nn.Conv2d(conv_channels, conv_channels, 3, padding=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.proj = nn.Sequential(
            init_layer(nn.Linear(conv_channels * rows * cols, out_dim)),
            nn.ReLU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if tuple(obs.shape[-3:]) != self.obs_shape:
            raise ValueError(f"expected [..., {self.obs_shape}], got {tuple(obs.shape)}")
        lead = obs.shape[:-3]
        x = scale_counts(obs.reshape(-1, *self.obs_shape)).permute(0, 3, 1, 2)
        return self.proj(self.conv(x)).reshape(*lead, -1)


class GlobalStateEncoder(nn.Module):
    """CNN for the centralized critic's ``[..., C, Y, X]`` global state (channels-first)."""

    def __init__(self, state_shape: tuple[int, int, int], conv_channels: int, out_dim: int) -> None:
        super().__init__()
        channels, rows, cols = state_shape
        self.state_shape = tuple(state_shape)
        self.conv = nn.Sequential(
            init_layer(nn.Conv2d(channels, conv_channels, 3, padding=1)),
            nn.ReLU(),
            init_layer(nn.Conv2d(conv_channels, conv_channels, 3, stride=2, padding=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        flat = conv_channels * math.ceil(rows / 2) * math.ceil(cols / 2)
        self.proj = nn.Sequential(init_layer(nn.Linear(flat, out_dim)), nn.ReLU())

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if tuple(state.shape[-3:]) != self.state_shape:
            raise ValueError(f"expected [..., {self.state_shape}], got {tuple(state.shape)}")
        lead = state.shape[:-3]
        x = scale_counts(state.reshape(-1, *self.state_shape))
        return self.proj(self.conv(x)).reshape(*lead, -1)
