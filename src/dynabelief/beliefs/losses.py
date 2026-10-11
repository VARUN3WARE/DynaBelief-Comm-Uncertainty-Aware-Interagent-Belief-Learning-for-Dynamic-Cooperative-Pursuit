"""Belief losses shared by the PPO update and InfER replay (heads: ``models/belief.py``).

``point`` (M4): normalized (x, y) of the nearest team-visible evader, squared Euclidean
error, masked by the "found" rule. ``map`` (M5, D-DIABL): one logit per grid cell,
BCE-with-logits against the true occupancy, averaged over free (non-obstacle) cells.
Sigmoids are applied only for metrics and visualization.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(x.dtype)
    return (x * mask).sum() / mask.sum().clamp(min=1.0)


def point_loss(pred, target, valid, agent_mask, grid_scale: float):
    """Returns ``(loss, metrics)``; metrics are detached scalars."""
    mask = valid & agent_mask
    squared = (pred - target).pow(2).sum(-1)
    loss = _masked_mean(squared, mask)
    with torch.no_grad():
        metrics = {
            "belief_error_cells": _masked_mean(squared.sqrt(), mask) * grid_scale,
            "belief_valid_fraction": _masked_mean(mask.float(), agent_mask),
        }
    return loss, metrics


def map_loss(logits, occupancy, free, agent_mask):
    """``logits [L, B, N, Y, X]``, ``occupancy / free [L, B, Y, X]``, ``agent_mask [L, B, N]``."""
    target = occupancy.unsqueeze(2).expand_as(logits)
    mask = free.unsqueeze(2) & agent_mask[..., None, None]
    per_cell = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    loss = _masked_mean(per_cell, mask)
    with torch.no_grad():
        metrics = {"belief_brier": _masked_mean((torch.sigmoid(logits) - target).pow(2), mask)}
    return loss, metrics


def belief_loss(actor, hiddens: torch.Tensor, batch, grid_scale: float):
    """Dispatch on the actor's belief type. ``batch``: a PPO ``Minibatch`` or a replay batch."""
    pred = actor.belief_head(hiddens)
    if actor.belief_type == "point":
        return point_loss(pred, batch.belief_target, batch.belief_valid, batch.agent_mask,
                          grid_scale)  # fmt: skip
    if actor.belief_type == "map":
        return map_loss(pred, batch.occupancy, batch.free, batch.agent_mask)
    raise ValueError(f"unknown belief type {actor.belief_type!r}")
