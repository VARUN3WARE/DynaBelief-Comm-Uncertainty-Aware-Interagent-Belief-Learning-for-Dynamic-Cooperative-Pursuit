"""Generalized advantage estimation with explicit termination vs truncation."""

from __future__ import annotations

import torch


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advantages and returns for a ``[T, E, N]`` rollout.

    Args:
        rewards, values: ``[T, E, N]``; ``values[t]`` = V(s_t).
        next_values: ``[T, E, N]``, the value of the state reached by step t:
            V(s_{t+1}) normally, V(terminal obs) when the step truncated, and
            anything (ignored) when it terminated.
        terminated, truncated: ``[T, E]`` bool flags of step t.

    Termination cuts the bootstrap (absorbing state, value 0). Truncation keeps
    it (the episode was merely cut off). Both stop the advantage recursion,
    because step t+1 belongs to a new episode.
    """
    if not (rewards.shape == values.shape == next_values.shape):
        raise ValueError(f"shape mismatch: {rewards.shape}, {values.shape}, {next_values.shape}")
    if terminated.shape != rewards.shape[:2] or truncated.shape != rewards.shape[:2]:
        raise ValueError(f"flags must be [T, E] = {tuple(rewards.shape[:2])}")
    if (terminated & truncated).any():
        raise ValueError("a step cannot be both terminated and truncated")

    not_terminal = (~terminated).to(values.dtype).unsqueeze(-1)
    continues = (~(terminated | truncated)).to(values.dtype).unsqueeze(-1)
    deltas = rewards + gamma * next_values * not_terminal - values
    advantages = torch.zeros_like(values)
    running = torch.zeros_like(values[0])
    for t in reversed(range(rewards.shape[0])):
        running = deltas[t] + gamma * gae_lambda * continues[t] * running
        advantages[t] = running
    return advantages, advantages + values


def build_next_values(
    values: torch.Tensor,
    last_values: torch.Tensor,
    truncated: torch.Tensor,
    final_values: torch.Tensor,
) -> torch.Tensor:
    """``next_values`` for ``compute_gae``.

    Args:
        values: ``[T, E, N]`` V(s_t) as collected.
        last_values: ``[E, N]`` V(s_T) for the observation after the rollout.
        truncated: ``[T, E]``.
        final_values: ``[T, E, N]`` V(terminal obs of step t), only read where truncated.
    """
    shifted = torch.cat([values[1:], last_values.unsqueeze(0)], dim=0)
    return torch.where(truncated.unsqueeze(-1), final_values, shifted)
