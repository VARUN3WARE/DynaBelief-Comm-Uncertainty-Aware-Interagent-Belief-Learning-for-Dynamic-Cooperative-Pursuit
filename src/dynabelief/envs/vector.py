"""Synchronous vector of ``PursuitEnv`` with auto-reset and per-episode seeds.

Episode ``k`` of sub-environment ``i`` is reset with
``derive_seed(base_seed, stream, i, k)`` (64-bit), so every episode is
reproducible from the experiment seed alone. Training and evaluation use
different stream names, so their episode seeds coincide only with negligible
(~1e-8) probability.

Subprocess workers can replace this class later; it is the deterministic
reference implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dynabelief.config import EnvConfig
from dynabelief.envs.pursuit import (
    ActorObs,
    EpisodeStats,
    PrivilegedInfo,
    PursuitEnv,
    stack_actor,
    stack_privileged,
)
from dynabelief.utils.seeding import derive_seed


@dataclass(frozen=True)
class VecStepResult:
    """Batched step output; arrays have a leading ``num_envs`` dim.

    For sub-environments whose episode ended this step, ``actor``/``privileged``
    already hold the FIRST observation of the next episode (auto-reset). The
    terminal observations are kept in ``final_actor``/``final_privileged``,
    keyed by env index, for value bootstrapping on truncation.
    """

    actor: ActorObs
    privileged: PrivilegedInfo
    rewards: np.ndarray  # [E, N] float32
    terminated: np.ndarray  # [E] bool
    truncated: np.ndarray  # [E] bool
    captures: np.ndarray  # [E] int64
    final_actor: dict[int, ActorObs]
    final_privileged: dict[int, PrivilegedInfo]
    completed: list[EpisodeStats]

    @property
    def done(self) -> np.ndarray:
        return self.terminated | self.truncated


class PursuitVecEnv:
    def __init__(self, config: EnvConfig, num_envs: int, seed: int, stream: str = "train") -> None:
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        self.config = config
        self.num_envs = num_envs
        self.seed = seed
        self.stream = stream
        self.envs = [PursuitEnv(config) for _ in range(num_envs)]
        ref = self.envs[0]
        self.n_agents = ref.n_agents
        self.n_actions = ref.n_actions
        self.obs_shape = ref.obs_shape
        self.state_shape = ref.state_shape
        self.grid_shape = ref.grid_shape
        self.agent_ids = ref.agent_ids
        self._episode_index = [0] * num_envs

    def episode_seed(self, env_index: int, episode_index: int) -> int:
        return derive_seed(self.seed, self.stream, env_index, episode_index)

    def _reset_one(self, i: int) -> tuple[ActorObs, PrivilegedInfo]:
        seed = self.episode_seed(i, self._episode_index[i])
        self._episode_index[i] += 1
        return self.envs[i].reset(seed)

    def reset(self) -> tuple[ActorObs, PrivilegedInfo]:
        """Start a fresh episode in every sub-environment."""
        outs = [self._reset_one(i) for i in range(self.num_envs)]
        return stack_actor([o[0] for o in outs]), stack_privileged([o[1] for o in outs])

    def step(self, actions: np.ndarray) -> VecStepResult:
        actions = np.asarray(actions)
        expected = (self.num_envs, self.n_agents)
        if actions.shape != expected:
            raise ValueError(f"actions must have shape {expected}, got {actions.shape}")

        actors, privs, completed = [], [], []
        final_actor: dict[int, ActorObs] = {}
        final_priv: dict[int, PrivilegedInfo] = {}
        rewards = np.zeros(expected, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        captures = np.zeros(self.num_envs, dtype=np.int64)

        for i, env in enumerate(self.envs):
            result = env.step(actions[i])
            rewards[i] = result.rewards
            terminated[i], truncated[i] = result.terminated, result.truncated
            captures[i] = result.captures
            if result.done:
                stats = env.episode_stats
                assert stats is not None
                completed.append(stats)
                final_actor[i], final_priv[i] = result.actor, result.privileged
                actor, priv = self._reset_one(i)
            else:
                actor, priv = result.actor, result.privileged
            actors.append(actor)
            privs.append(priv)

        return VecStepResult(
            actor=stack_actor(actors),
            privileged=stack_privileged(privs),
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            captures=captures,
            final_actor=final_actor,
            final_privileged=final_priv,
            completed=completed,
        )

    def close(self) -> None:
        for env in self.envs:
            env.close()
