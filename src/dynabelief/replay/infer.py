"""Informative Event Replay (InfER, Puc et al. 2026, Algorithm 1), uniform variant (M4).

Events: the first time each evader becomes visible to the team in an episode (the
paper's "target discovered for the first time"). For each event at step ``t_r`` the
window ``t_r : t_r + T_r`` of that environment, with ALL agents, is stored. A window
that would cross an episode end, or the end of the rollout, is skipped (never cut), so
stored windows never span two episodes.

Replay updates ONLY the belief loss (no policy or value loss, hence no off-policy
correction is needed). As in the paper, messages are fully delivered during replay so
that an informative event is never wasted by packet loss. Windows are sampled
uniformly; SP-InfER (M6) will add priorities.

Storage is bounded (ring buffer of ``capacity`` windows) and compact: observations are
agent counts and are stored as ``uint8`` (exact for counts <= 255).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def first_sightings(team_visible: np.ndarray, seen_before: np.ndarray) -> np.ndarray:
    """``[E, M]`` evaders seen now but never before in their episode (updates nothing)."""
    return team_visible & ~seen_before


@dataclass
class ReplayBatch:
    """Time-major replay windows: ``[T_r, B, N, ...]``."""

    obs: torch.Tensor
    own_pos: torch.Tensor
    episode_start: torch.Tensor  # [T_r, B]
    agent_mask: torch.Tensor
    belief_target: torch.Tensor
    belief_valid: torch.Tensor
    hidden0: torch.Tensor  # [B, N, H]
    delivery: torch.Tensor  # [T_r, B, N, N] full delivery among alive agents


class EventReplayBuffer:
    def __init__(
        self,
        capacity: int,
        window: int,
        n_agents: int,
        obs_shape: tuple[int, ...],
        hidden_dim: int,
        generator: torch.Generator,
    ) -> None:
        if capacity < 1 or window < 1:
            raise ValueError("capacity and window must be >= 1")
        self.capacity, self.window, self.n = capacity, window, n_agents
        self.generator = generator
        W, N = window, n_agents
        self.obs = torch.zeros(capacity, W, N, *obs_shape, dtype=torch.uint8)
        self.own_pos = torch.zeros(capacity, W, N, 2)
        self.episode_start = torch.zeros(capacity, W, dtype=torch.bool)
        self.agent_mask = torch.zeros(capacity, W, N, dtype=torch.bool)
        self.belief_target = torch.zeros(capacity, W, N, 2)
        self.belief_valid = torch.zeros(capacity, W, N, dtype=torch.bool)
        self.hidden0 = torch.zeros(capacity, N, hidden_dim)
        self.size = 0
        self.next = 0
        self.added = 0
        self.skipped = 0

    def __len__(self) -> int:
        return self.size

    def add_from_rollout(self, rollout, event: torch.Tensor) -> int:
        """Store a window for every event ``event[t, e]`` of a full ``RolloutBuffer``.

        Returns the number of windows stored; skipped events are counted in ``skipped``.
        """
        T, W = rollout.T, self.window
        done = (rollout.terminated | rollout.truncated).cpu()
        stored = 0
        for t, e in torch.nonzero(event.cpu(), as_tuple=False).tolist():
            # The window ends at t + W - 1; an episode end strictly before that, or the
            # rollout end, would make it span two episodes or run out of data.
            if t + W > T or done[t : t + W - 1, e].any():
                self.skipped += 1
                continue
            obs = rollout.obs[t : t + W, e].cpu()
            if obs.max() > 255 or obs.min() < 0 or (obs != obs.round()).any():
                raise ValueError("observations are not small integer counts; cannot store as uint8")
            i = self.next
            self.obs[i] = obs.to(torch.uint8)
            self.own_pos[i] = rollout.own_pos[t : t + W, e].cpu()
            self.episode_start[i] = rollout.episode_start[t : t + W, e].cpu()
            self.agent_mask[i] = rollout.agent_mask[t : t + W, e].cpu()
            self.belief_target[i] = rollout.belief_target[t : t + W, e].cpu()
            self.belief_valid[i] = rollout.belief_valid[t : t + W, e].cpu()
            self.hidden0[i] = rollout.actor_hidden[t, e].cpu()
            self.next = (i + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
            stored += 1
        self.added += stored
        return stored

    def sample(self, batch: int, device: torch.device | str) -> ReplayBatch:
        """``batch`` windows uniformly at random (with replacement)."""
        if self.size == 0:
            raise RuntimeError("replay buffer is empty")
        idx = torch.randint(0, self.size, (batch,), generator=self.generator)

        def take(x: torch.Tensor, time_major: bool = True) -> torch.Tensor:
            out = x[idx].to(device)
            return out.transpose(0, 1) if time_major else out

        mask = take(self.agent_mask)
        eye = torch.eye(self.n, dtype=torch.bool, device=mask.device)
        delivery = mask[..., :, None] & mask[..., None, :] & ~eye
        return ReplayBatch(
            obs=take(self.obs).float(),
            own_pos=take(self.own_pos),
            episode_start=take(self.episode_start),
            agent_mask=mask,
            belief_target=take(self.belief_target),
            belief_valid=take(self.belief_valid),
            hidden0=take(self.hidden0, time_major=False),
            delivery=delivery,
        )
