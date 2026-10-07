"""PettingZoo ``pursuit_v4`` wrapper with an explicit actor / training-only split.

Information boundary
--------------------
Every reset/step returns two separate objects:

* ``ActorObs``: the ONLY data a decentralized actor may consume: each
  pursuer's own local view plus an agent mask. Delivered messages are added
  by the communication layer, never by this module.
* ``PrivilegedInfo``: training-only ground truth (global state, evader
  occupancy, positions). It may feed the centralized critic, auxiliary belief
  labels, evaluation metrics, and diagnostics. It must never reach the actor.

Grid layout conventions
-----------------------
The simulator indexes its internal maps as ``[x, y]`` and positions as
``(x, y)``. PettingZoo returns local observations transposed to
``[y, x, channel]``. To keep one orientation everywhere, every grid this
package emits uses image layout ``row = y, col = x``:

* local obs        ``[n_agents, obs_range, obs_range, 3]`` (channels-last, as PettingZoo returns it)
* global state     ``[3, y_size, x_size]`` (channels-first, for a CNN critic)
* occupancy        ``[y_size, x_size]``
* positions        ``(x, y)`` integer pairs, as in the simulator

Observation channels: 0 = wall / obstacle / out-of-bounds, 1 = pursuer count
(including the observer itself), 2 = evader count.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

# PettingZoo's Pursuit calls pygame.init(), and SDL then installs a C-level SIGTERM
# handler that swallows the signal. Worker processes would ignore terminate() and
# a crashing or killed training run would hang at exit. This SDL hint keeps the
# default signal handlers; it must be set before pygame initializes.
os.environ.setdefault("SDL_NO_SIGNAL_HANDLERS", "1")

import numpy as np  # noqa: E402
from pettingzoo.sisl import pursuit_v4  # noqa: E402

from dynabelief.config import EnvConfig

N_OBS_CHANNELS = 3
N_STATE_CHANNELS = 3
# Action index -> (dx, dy) in simulator coordinates (pettingzoo DiscreteAgent.motion_range).
ACTION_DELTAS = ((-1, 0), (1, 0), (0, 1), (0, -1), (0, 0))


@dataclass(frozen=True)
class ActorObs:
    """Actor-visible inputs. Leading batch dims (e.g. ``num_envs``) are allowed.

    Attributes:
        obs: ``[..., n_agents, obs_range, obs_range, 3]`` float32 local views.
        agent_mask: ``[..., n_agents]`` bool, True for agents that act this step.
    """

    obs: np.ndarray
    agent_mask: np.ndarray


@dataclass(frozen=True)
class PrivilegedInfo:
    """TRAINING-ONLY ground truth. Never pass any field to the actor.

    Attributes:
        global_state: ``[..., 3, y_size, x_size]`` float32 (obstacles, pursuers, evaders).
        occupancy: ``[..., y_size, x_size]`` float32 in {0, 1}; 1 where >= 1 evader stands.
        pursuer_pos: ``[..., n_pursuers, 2]`` int64 ``(x, y)``.
        evader_pos: ``[..., n_evaders, 2]`` int64 ``(x, y)`` by original id; ``-1`` once caught.
        evader_alive: ``[..., n_evaders]`` bool by original evader id.
        step: ``[...]`` int64 steps taken so far in the episode (0 right after reset).
            Lets the centralized critic value the time left before truncation.
    """

    global_state: np.ndarray
    occupancy: np.ndarray
    pursuer_pos: np.ndarray
    evader_pos: np.ndarray
    evader_alive: np.ndarray
    step: np.ndarray


@dataclass
class EpisodeStats:
    """Per-episode summary. ``capture_steps`` are 1-based step indices of each capture."""

    seed: int
    length: int = 0
    team_return: float = 0.0
    agent_returns: list[float] = field(default_factory=list)
    captures: int = 0
    capture_steps: list[int] = field(default_factory=list)
    n_evaders: int = 0
    terminated: bool = False
    truncated: bool = False

    @property
    def capture_rate(self) -> float:
        return self.captures / self.n_evaders if self.n_evaders else 0.0

    @property
    def mean_time_to_capture(self) -> float | None:
        return float(np.mean(self.capture_steps)) if self.capture_steps else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "length": self.length,
            "team_return": self.team_return,
            "agent_returns": self.agent_returns,
            "captures": self.captures,
            "n_evaders": self.n_evaders,
            "capture_rate": self.capture_rate,
            "capture_steps": self.capture_steps,
            "mean_time_to_capture": self.mean_time_to_capture,
            "all_captured": self.captures == self.n_evaders,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }


@dataclass(frozen=True)
class StepResult:
    actor: ActorObs
    privileged: PrivilegedInfo
    rewards: np.ndarray  # [n_agents] float32
    terminated: bool  # all evaders caught
    truncated: bool  # max_cycles reached
    captures: int  # evaders caught during this step

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


class PursuitEnv:
    """Single Pursuit environment with array I/O in a fixed agent order.

    ``reset`` requires an explicit seed: PettingZoo's Pursuit draws OS entropy
    when reset without one, which would silently break reproducibility.
    """

    def __init__(self, config: EnvConfig) -> None:
        config.validate()
        self.config = config
        self._env = pursuit_v4.parallel_env(**config.pettingzoo_kwargs())
        self.agent_ids: tuple[str, ...] = tuple(self._env.possible_agents)
        self.n_agents = len(self.agent_ids)
        self.n_evaders = config.n_evaders
        first = self.agent_ids[0]
        self.obs_shape: tuple[int, ...] = tuple(self._env.observation_space(first).shape)
        self.obs_dtype = np.dtype(self._env.observation_space(first).dtype)
        self.n_actions = int(self._env.action_space(first).n)
        self.grid_shape = (config.y_size, config.x_size)  # (rows=y, cols=x)
        self.state_shape = (N_STATE_CHANNELS, *self.grid_shape)
        self._episode: EpisodeStats | None = None
        self._done = True

    # Simulator internals ------------------------------------------------
    @property
    def _sim(self) -> Any:
        """The underlying ``pursuit_base.Pursuit`` object (read-only use)."""
        return self._env.unwrapped.env

    def _privileged(self) -> PrivilegedInfo:
        sim = self._sim
        obstacles = (sim.map_matrix == -1).astype(np.float32)
        pursuer_counts = sim.pursuer_layer.get_state_matrix().astype(np.float32)
        evader_counts = sim.evader_layer.get_state_matrix().astype(np.float32)
        # Simulator maps are [x, y]; transpose to image layout [y, x].
        global_state = np.stack([obstacles.T, pursuer_counts.T, evader_counts.T])
        occupancy = (evader_counts.T > 0).astype(np.float32)

        pursuer_pos = np.array(
            [sim.pursuer_layer.get_position(i) for i in range(sim.pursuer_layer.n_agents())],
            dtype=np.int64,
        ).reshape(-1, 2)

        # The simulator pops caught evaders from its list; remaining entries keep
        # their original order, so they map onto the not-yet-gone original ids.
        alive = ~np.asarray(sim.evaders_gone, dtype=bool)
        alive_ids = np.flatnonzero(alive)
        n_live = sim.evader_layer.n_agents()
        if len(alive_ids) != n_live:
            raise RuntimeError(
                f"evader bookkeeping mismatch: {len(alive_ids)} ids vs {n_live} live"
            )
        evader_pos = np.full((self.n_evaders, 2), -1, dtype=np.int64)
        for row, original_id in enumerate(alive_ids):
            evader_pos[original_id] = sim.evader_layer.get_position(row)
        return PrivilegedInfo(
            global_state=global_state,
            occupancy=occupancy,
            pursuer_pos=pursuer_pos,
            evader_pos=evader_pos,
            evader_alive=alive.copy(),
            step=np.asarray(self._episode.length if self._episode else 0, dtype=np.int64),
        )

    def _actor(self, observations: dict[str, np.ndarray]) -> ActorObs:
        missing = [a for a in self.agent_ids if a not in observations]
        if missing:
            raise RuntimeError(f"PettingZoo returned no observation for {missing}")
        obs = np.stack([np.asarray(observations[a], dtype=np.float32) for a in self.agent_ids])
        if obs.shape[1:] != self.obs_shape:
            raise RuntimeError(f"observation shape {obs.shape[1:]} != declared {self.obs_shape}")
        if not np.isfinite(obs).all():
            raise RuntimeError("non-finite value in observation")
        return ActorObs(obs=obs, agent_mask=np.ones(self.n_agents, dtype=bool))

    # Public API ---------------------------------------------------------
    def reset(self, seed: int) -> tuple[ActorObs, PrivilegedInfo]:
        if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
            raise TypeError(f"reset requires an integer seed, got {seed!r}")
        observations, _ = self._env.reset(seed=int(seed))
        self._episode = EpisodeStats(
            seed=int(seed), agent_returns=[0.0] * self.n_agents, n_evaders=self.n_evaders
        )
        self._done = False
        return self._actor(observations), self._privileged()

    def step(self, actions: np.ndarray) -> StepResult:
        if self._done or self._episode is None:
            raise RuntimeError("step() called on a finished episode; call reset(seed) first")
        actions = np.asarray(actions)
        if actions.shape != (self.n_agents,):
            raise ValueError(f"actions must have shape ({self.n_agents},), got {actions.shape}")
        if not np.issubdtype(actions.dtype, np.integer):
            raise TypeError(f"actions must be integers, got dtype {actions.dtype}")
        if actions.min() < 0 or actions.max() >= self.n_actions:
            raise ValueError(f"actions must be in [0, {self.n_actions}), got {actions}")

        gone_before = int(np.sum(self._sim.evaders_gone))
        action_dict = {a: int(act) for a, act in zip(self.agent_ids, actions, strict=True)}
        observations, rewards, terminations, truncations, _ = self._env.step(action_dict)
        captures = int(np.sum(self._sim.evaders_gone)) - gone_before

        reward_arr = np.array([rewards[a] for a in self.agent_ids], dtype=np.float32)
        if not np.isfinite(reward_arr).all():
            raise RuntimeError(f"non-finite reward {reward_arr}")
        # PettingZoo only sets truncation on the max_cycles step, even if that step caught
        # the last evader. Catching everyone is an absorbing terminal state (no value
        # bootstrap), so it takes precedence over the time limit.
        all_caught = self._sim.evader_layer.n_agents() == 0
        terminated = bool(any(terminations.values())) or all_caught
        truncated = bool(any(truncations.values())) and not terminated

        ep = self._episode
        ep.length += 1
        ep.team_return += float(reward_arr.mean())
        ep.agent_returns = [r + float(x) for r, x in zip(ep.agent_returns, reward_arr, strict=True)]
        ep.captures += captures
        ep.capture_steps.extend([ep.length] * captures)
        ep.terminated, ep.truncated = terminated, truncated
        self._done = terminated or truncated

        return StepResult(
            actor=self._actor(observations),
            privileged=self._privileged(),
            rewards=reward_arr,
            terminated=terminated,
            truncated=truncated,
            captures=captures,
        )

    @property
    def episode_stats(self) -> EpisodeStats | None:
        return self._episode

    def close(self) -> None:
        self._env.close()


def stack_actor(items: list[ActorObs]) -> ActorObs:
    return ActorObs(
        obs=np.stack([x.obs for x in items]), agent_mask=np.stack([x.agent_mask for x in items])
    )


def stack_privileged(items: list[PrivilegedInfo]) -> PrivilegedInfo:
    return PrivilegedInfo(
        global_state=np.stack([x.global_state for x in items]),
        occupancy=np.stack([x.occupancy for x in items]),
        pursuer_pos=np.stack([x.pursuer_pos for x in items]),
        evader_pos=np.stack([x.evader_pos for x in items]),
        evader_alive=np.stack([x.evader_alive for x in items]),
        step=np.stack([x.step for x in items]),
    )
