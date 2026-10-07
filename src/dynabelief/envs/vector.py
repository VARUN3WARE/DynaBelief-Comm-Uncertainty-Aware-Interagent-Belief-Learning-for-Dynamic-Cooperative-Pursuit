"""Vector of ``PursuitEnv`` with auto-reset, per-episode seeds, and optional subprocesses.

Episode ``k`` of sub-environment ``i`` is reset with
``derive_seed(base_seed, stream, i, k)`` (64-bit), so every episode is
reproducible from the experiment seed alone. Training and evaluation use
different stream names, so their episode seeds coincide only with negligible
(~1e-8) probability.

Backends
--------
``num_workers=0`` steps every sub-environment in this process. ``num_workers=W``
splits the sub-environments into ``W`` contiguous groups, each stepped by a
spawned worker process. Both backends run the same ``EnvGroup`` code with the
same global env indices, so they produce bit-identical results; the subprocess
backend is only faster (the PettingZoo simulator is pure Python and CPU-bound).
"""

from __future__ import annotations

import dataclasses
import multiprocessing as mp
import traceback
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Any

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


@dataclass(frozen=True)
class _EnvStep:
    """One sub-environment's step outcome after auto-reset (picklable)."""

    actor: ActorObs
    privileged: PrivilegedInfo
    rewards: np.ndarray
    terminated: bool
    truncated: bool
    captures: int
    final: tuple[ActorObs, PrivilegedInfo] | None
    stats: EpisodeStats | None


class EnvGroup:
    """Sub-environments with fixed GLOBAL indices; shared by both backends."""

    def __init__(self, config: EnvConfig, env_indices: list[int], seed: int, stream: str) -> None:
        self.env_indices = list(env_indices)
        self.seed = seed
        self.stream = stream
        self.envs = [PursuitEnv(config) for _ in self.env_indices]
        self.episode_index = [0] * len(self.env_indices)

    def _reset_one(self, local: int) -> tuple[ActorObs, PrivilegedInfo]:
        global_index = self.env_indices[local]
        seed = derive_seed(self.seed, self.stream, global_index, self.episode_index[local])
        self.episode_index[local] += 1
        return self.envs[local].reset(seed)

    def reset(self) -> list[tuple[ActorObs, PrivilegedInfo]]:
        return [self._reset_one(local) for local in range(len(self.envs))]

    def step(self, actions: np.ndarray) -> list[_EnvStep]:
        out = []
        for local, env in enumerate(self.envs):
            result = env.step(actions[local])
            final, stats = None, None
            if result.done:
                stats = dataclasses.replace(env.episode_stats)  # snapshot before reset
                final = (result.actor, result.privileged)
                actor, priv = self._reset_one(local)
            else:
                actor, priv = result.actor, result.privileged
            out.append(
                _EnvStep(
                    actor=actor,
                    privileged=priv,
                    rewards=result.rewards,
                    terminated=result.terminated,
                    truncated=result.truncated,
                    captures=result.captures,
                    final=final,
                    stats=stats,
                )
            )
        return out

    def close(self) -> None:
        for env in self.envs:
            env.close()


def _worker(
    remote: Connection,
    parent_remote: Connection,
    config: EnvConfig,
    env_indices: list[int],
    seed: int,
    stream: str,
) -> None:
    parent_remote.close()
    group = None
    try:
        group = EnvGroup(config, env_indices, seed, stream)
        remote.send(("ok", None))
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                remote.send(("ok", group.reset()))
            elif cmd == "step":
                remote.send(("ok", group.step(data)))
            elif cmd == "close":
                break
            else:
                raise ValueError(f"unknown command {cmd!r}")
    except (EOFError, KeyboardInterrupt):
        pass
    except Exception:  # forwarded to the parent, which re-raises it
        try:
            remote.send(("error", traceback.format_exc()))
        except (BrokenPipeError, OSError):
            pass
    finally:
        if group is not None:
            group.close()
        remote.close()


class WorkerError(RuntimeError):
    """An exception raised inside an environment worker process."""


class PursuitVecEnv:
    def __init__(
        self,
        config: EnvConfig,
        num_envs: int,
        seed: int,
        stream: str = "train",
        num_workers: int = 0,
    ) -> None:
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        if not 0 <= num_workers <= num_envs:
            raise ValueError(f"num_workers must be in [0, num_envs={num_envs}], got {num_workers}")
        config.validate()
        self.config = config
        self.num_envs = num_envs
        self.seed = seed
        self.stream = stream
        self.num_workers = num_workers
        self._closed = False

        probe = PursuitEnv(config)
        self.n_agents = probe.n_agents
        self.n_actions = probe.n_actions
        self.obs_shape = probe.obs_shape
        self.state_shape = probe.state_shape
        self.grid_shape = probe.grid_shape
        self.agent_ids = probe.agent_ids
        probe.close()

        if num_workers == 0:
            self._local: EnvGroup | None = EnvGroup(config, list(range(num_envs)), seed, stream)
            self._remotes: list[Connection] = []
            self._processes: list[Any] = []
            self._chunks = [list(range(num_envs))]
        else:
            self._local = None
            self._chunks = [c.tolist() for c in np.array_split(np.arange(num_envs), num_workers)]
            ctx = mp.get_context("spawn")  # safe with torch/CUDA in the parent
            self._remotes, self._processes = [], []
            for chunk in self._chunks:
                parent, child = ctx.Pipe()
                proc = ctx.Process(
                    target=_worker,
                    args=(child, parent, config, chunk, seed, stream),
                    daemon=True,
                )
                proc.start()
                child.close()
                self._remotes.append(parent)
                self._processes.append(proc)
            for remote in self._remotes:
                self._receive(remote)

    def episode_seed(self, env_index: int, episode_index: int) -> int:
        return derive_seed(self.seed, self.stream, env_index, episode_index)

    def _receive(self, remote: Connection) -> Any:
        try:
            status, payload = remote.recv()
        except EOFError as exc:
            self.close()
            raise WorkerError("environment worker exited unexpectedly") from exc
        if status == "error":
            self.close()
            raise WorkerError(f"environment worker failed:\n{payload}")
        return payload

    def _gather(self, cmd: str, per_chunk: list[Any]) -> list[Any]:
        if self._closed:
            raise RuntimeError("PursuitVecEnv is closed")
        if self._local is not None:
            group_call = self._local.reset if cmd == "reset" else self._local.step
            return group_call() if cmd == "reset" else group_call(per_chunk[0])
        for remote, data in zip(self._remotes, per_chunk, strict=True):
            remote.send((cmd, data))
        out: list[Any] = []
        for remote in self._remotes:
            out.extend(self._receive(remote))
        return out

    def reset(self) -> tuple[ActorObs, PrivilegedInfo]:
        """Start a fresh episode in every sub-environment."""
        outs = self._gather("reset", [None] * len(self._chunks))
        return stack_actor([o[0] for o in outs]), stack_privileged([o[1] for o in outs])

    def step(self, actions: np.ndarray) -> VecStepResult:
        actions = np.asarray(actions)
        expected = (self.num_envs, self.n_agents)
        if actions.shape != expected:
            raise ValueError(f"actions must have shape {expected}, got {actions.shape}")
        steps: list[_EnvStep] = self._gather("step", [actions[c] for c in self._chunks])

        final_actor = {i: s.final[0] for i, s in enumerate(steps) if s.final is not None}
        final_priv = {i: s.final[1] for i, s in enumerate(steps) if s.final is not None}
        return VecStepResult(
            actor=stack_actor([s.actor for s in steps]),
            privileged=stack_privileged([s.privileged for s in steps]),
            rewards=np.stack([s.rewards for s in steps]).astype(np.float32, copy=False),
            terminated=np.array([s.terminated for s in steps], dtype=bool),
            truncated=np.array([s.truncated for s in steps], dtype=bool),
            captures=np.array([s.captures for s in steps], dtype=np.int64),
            final_actor=final_actor,
            final_privileged=final_priv,
            completed=[s.stats for s in steps if s.stats is not None],
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._local is not None:
            self._local.close()
        for remote in self._remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for proc in self._processes:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
        for remote in self._remotes:
            remote.close()

    def __enter__(self) -> PursuitVecEnv:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # interpreter shutdown: best effort only
            pass
