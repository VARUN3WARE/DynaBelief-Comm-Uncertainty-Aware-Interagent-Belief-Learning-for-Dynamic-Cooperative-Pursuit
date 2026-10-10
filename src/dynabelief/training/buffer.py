"""On-policy rollout storage and recurrent minibatching for MAPPO.

Layout is ``[T, E, N, ...]`` (time, env, agent). Recurrent minibatches are
built from *(env, chunk)* units: a chunk is ``chunk_length`` consecutive steps
of one environment with ALL its agents. Keeping agents together is required
once agents communicate (M3), and harmless now.

Hidden-state contract: ``actor_hidden[t]`` is the actor's hidden state BEFORE
it processed ``obs[t]`` (and before the episode-start reset is applied). The
actor re-applies the reset from ``episode_start``, so unrolling a chunk from
``actor_hidden[chunk_start]`` reproduces the rollout's logits exactly.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch


@dataclass
class Minibatch:
    """Time-major sequences: ``[L, B, ...]`` with ``B`` = (env, chunk) units."""

    obs: torch.Tensor  # [L, B, N, R, R, C]
    episode_start: torch.Tensor  # [L, B]
    actor_hidden0: torch.Tensor  # [B, N, H]
    actions: torch.Tensor  # [L, B, N]
    old_log_probs: torch.Tensor  # [L, B, N]
    old_values: torch.Tensor  # [L, B, N]  (return units)
    advantages: torch.Tensor  # [L, B, N]  (normalized)
    returns: torch.Tensor  # [L, B, N]  (return units)
    agent_mask: torch.Tensor  # [L, B, N]
    delivery: torch.Tensor  # [L, B, N_recv, N_send] messages readable at each step
    own_pos: torch.Tensor  # [L, B, N, 2] each agent's own normalized position
    belief_target: torch.Tensor  # [L, B, N, 2] TRAINING-ONLY label (never an actor input)
    belief_valid: torch.Tensor  # [L, B, N] label exists (target found by the team)
    global_state: torch.Tensor  # [L, B, C, Y, X]
    pursuer_pos: torch.Tensor  # [L, B, N, 2]
    step: torch.Tensor  # [L, B]


class RolloutBuffer:
    def __init__(
        self,
        rollout_length: int,
        num_envs: int,
        n_agents: int,
        obs_shape: tuple[int, ...],
        state_shape: tuple[int, ...],
        hidden_dim: int,
        device: torch.device | str = "cpu",
    ) -> None:
        T, E, N = rollout_length, num_envs, n_agents
        self.T, self.E, self.N = T, E, N
        self.device = torch.device(device)

        def zeros(*shape, dtype=torch.float32):
            return torch.zeros(*shape, dtype=dtype, device=self.device)

        self.obs = zeros(T, E, N, *obs_shape)
        self.episode_start = zeros(T, E, dtype=torch.bool)
        self.actor_hidden = zeros(T, E, N, hidden_dim)
        self.actions = zeros(T, E, N, dtype=torch.long)
        self.log_probs = zeros(T, E, N)
        self.values = zeros(T, E, N)
        self.rewards = zeros(T, E, N)
        self.agent_mask = zeros(T, E, N, dtype=torch.bool)
        self.delivery = zeros(T, E, N, N, dtype=torch.bool)
        self.own_pos = zeros(T, E, N, 2)
        self.belief_target = zeros(T, E, N, 2)
        self.belief_valid = zeros(T, E, N, dtype=torch.bool)
        self.terminated = zeros(T, E, dtype=torch.bool)
        self.truncated = zeros(T, E, dtype=torch.bool)
        self.final_values = zeros(T, E, N)
        self.global_state = zeros(T, E, *state_shape)
        self.pursuer_pos = zeros(T, E, N, 2, dtype=torch.long)
        self.step = zeros(T, E, dtype=torch.long)
        self.advantages = zeros(T, E, N)
        self.returns = zeros(T, E, N)
        self.pos = 0

    @property
    def full(self) -> bool:
        return self.pos == self.T

    def add(self, **fields: torch.Tensor) -> None:
        """Store one time step; every per-step field must be given exactly once."""
        if self.full:
            raise RuntimeError("buffer is full; call reset() after the update")
        required = {
            "obs", "episode_start", "actor_hidden", "actions", "log_probs", "values",
            "rewards", "agent_mask", "delivery", "own_pos", "belief_target", "belief_valid",
            "terminated", "truncated", "final_values",
            "global_state", "pursuer_pos", "step",
        }  # fmt: skip
        if set(fields) != required:
            missing, unexpected = sorted(required - set(fields)), sorted(set(fields) - required)
            raise ValueError(f"missing {missing}, unexpected {unexpected}")
        for name, value in fields.items():
            target = getattr(self, name)
            if tuple(value.shape) != tuple(target.shape[1:]):
                raise ValueError(
                    f"{name}: expected {tuple(target.shape[1:])}, got {tuple(value.shape)}"
                )
            target[self.pos] = value.to(device=self.device, dtype=target.dtype)
        self.pos += 1

    def reset(self) -> None:
        self.pos = 0

    def minibatches(
        self,
        num_minibatches: int,
        chunk_length: int,
        generator: torch.Generator,
        advantages: torch.Tensor,
    ) -> Iterator[Minibatch]:
        """Shuffled minibatches of (env, chunk) sequences covering the rollout once."""
        if not self.full:
            raise RuntimeError("minibatches() needs a full buffer")
        if self.T % chunk_length:
            raise ValueError(
                f"rollout_length {self.T} not divisible by chunk_length {chunk_length}"
            )
        n_chunks = self.T // chunk_length
        n_units = self.E * n_chunks
        if not 1 <= num_minibatches <= n_units:
            raise ValueError(f"num_minibatches must be in [1, {n_units}], got {num_minibatches}")

        order = torch.randperm(n_units, generator=generator).to(self.device)
        offsets = torch.arange(chunk_length, device=self.device)
        for unit_ids in torch.tensor_split(order, num_minibatches):
            env_idx = unit_ids % self.E  # [B]
            start = (unit_ids // self.E) * chunk_length  # [B]
            time_idx = start[None, :] + offsets[:, None]  # [L, B]
            env_grid = env_idx[None, :].expand_as(time_idx)
            index = (time_idx, env_grid)

            def take(x: torch.Tensor, index: tuple[torch.Tensor, torch.Tensor] = index):
                return x[index]

            yield Minibatch(
                obs=take(self.obs),
                episode_start=take(self.episode_start),
                actor_hidden0=self.actor_hidden[start, env_idx],
                actions=take(self.actions),
                old_log_probs=take(self.log_probs),
                old_values=take(self.values),
                advantages=take(advantages),
                returns=take(self.returns),
                agent_mask=take(self.agent_mask),
                delivery=take(self.delivery),
                own_pos=take(self.own_pos),
                belief_target=take(self.belief_target),
                belief_valid=take(self.belief_valid),
                global_state=take(self.global_state),
                pursuer_pos=take(self.pursuer_pos),
                step=take(self.step),
            )
