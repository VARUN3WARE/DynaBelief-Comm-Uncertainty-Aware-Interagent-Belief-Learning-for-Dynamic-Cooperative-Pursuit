"""Strictly validated experiment configuration.

Configs are YAML files mapped onto frozen dataclasses. Unknown keys and invalid
values raise ``ConfigError`` so that a typo can never silently fall back to a
default and change an experiment.

Only the sections needed by the current milestone exist. Later milestones add
sections (model, algorithm, belief, replay) instead of accepting free-form keys.
"""

from __future__ import annotations

import dataclasses
import math
import re
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised for unknown keys, wrong types, or out-of-range values."""


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "debug"
    seed: int = 0
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "cuda:N"
    output_dir: str = "runs"
    # Deterministic CUDA kernels: bit-reproducible GPU runs at some speed cost. CPU runs
    # are reproducible either way.
    deterministic: bool = False

    def validate(self) -> None:
        _require(
            _NAME_RE.fullmatch(self.name) is not None,
            f"experiment.name must match {_NAME_RE.pattern} (it becomes a directory name), "
            f"got {self.name!r}",
        )
        _require(0 <= self.seed < 2**63, "experiment.seed must be in [0, 2**63)")
        _require(
            _DEVICE_RE.fullmatch(self.device) is not None,
            f"experiment.device must be auto/cpu/cuda/cuda:N, got {self.device!r}",
        )


_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_DEVICE_RE = re.compile(r"auto|cpu|cuda(:[0-9]+)?")


@dataclass(frozen=True)
class EnvConfig:
    """Parameters forwarded to ``pettingzoo.sisl.pursuit_v4.parallel_env``.

    Defaults follow the project study (16x16 grid, 8 pursuers, 7x7 views,
    500-step episodes); PettingZoo's own defaults differ for some fields.
    """

    x_size: int = 16
    y_size: int = 16
    n_pursuers: int = 8
    n_evaders: int = 30
    obs_range: int = 7
    max_cycles: int = 500
    n_catch: int = 2
    freeze_evaders: bool = False
    shared_reward: bool = True
    surround: bool = True
    tag_reward: float = 0.01
    catch_reward: float = 5.0
    urgency_reward: float = -0.1
    constraint_window: float = 1.0

    def validate(self) -> None:
        _require(self.x_size >= 3 and self.y_size >= 3, "env grid must be at least 3x3")
        _require(self.n_pursuers >= 1, "env.n_pursuers must be >= 1")
        _require(self.n_evaders >= 1, "env.n_evaders must be >= 1")
        _require(
            self.obs_range >= 3 and self.obs_range % 2 == 1,
            "env.obs_range must be an odd integer >= 3",
        )
        _require(self.max_cycles >= 1, "env.max_cycles must be >= 1")
        _require(self.n_catch >= 1, "env.n_catch must be >= 1")
        _require(0.0 < self.constraint_window <= 1.0, "env.constraint_window must be in (0, 1]")
        self._validate_spawnable()

    def _validate_spawnable(self) -> None:
        """Reject layouts where PettingZoo's spawn sampler could loop forever.

        PettingZoo places each group (pursuers, evaders) by rejection sampling, without
        a retry limit, inside a random window of relative size ``constraint_window``. A
        cell is rejected if it is an obstacle or is on/4-adjacent to an already placed
        member of the same group, so each placement blocks at most 5 cells. Requiring
        ``5 * (n - 1) < free cells`` in the smallest reachable window guarantees that
        every placement can succeed. The bound is conservative (~52 agents per group on
        an open 16x16 grid).
        """
        from pettingzoo.sisl.pursuit.utils.two_d_maps import rectangle_map

        free = rectangle_map(self.x_size, self.y_size) != -1
        x_windows = _spawn_windows(self.x_size, self.constraint_window)
        y_windows = _spawn_windows(self.y_size, self.constraint_window)
        _require(
            all(hi > lo for lo, hi in x_windows | y_windows),
            f"env.constraint_window={self.constraint_window} gives an empty spawn window",
        )
        min_free = min(
            int(free[xl:xh, yl:yh].sum()) for xl, xh in x_windows for yl, yh in y_windows
        )
        largest_group = max(self.n_pursuers, self.n_evaders)
        _require(
            5 * (largest_group - 1) < min_free,
            f"env: {largest_group} agents in one group may not fit the smallest spawn window "
            f"({min_free} free cells, need > {5 * (largest_group - 1)}); PettingZoo would hang. "
            "Use a larger grid/constraint_window or fewer agents.",
        )

    def pettingzoo_kwargs(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class CommConfig:
    """Communication channel settings.

    ``delay_steps=1`` is the project's canonical timing: messages produced at
    step ``t`` are delivered (or dropped) and become readable at ``t+1``.
    With ``train.policy: mappo``, ``enabled: true`` means TarMAC-MAPPO (M3): a message
    is a key (``key_dim``) plus a value (``message_dim``) floats.
    """

    enabled: bool = True
    message_dim: int = 16  # value (content) dimension
    key_dim: int = 16  # attention signature dimension
    packet_loss: float = 0.0
    delay_steps: int = 1
    bytes_per_element: int = 4  # float32 on the wire

    @property
    def elements_per_message(self) -> int:
        """Floats on the wire per directed message (key + value; the query stays local)."""
        return self.key_dim + self.message_dim

    def validate(self) -> None:
        _require(self.message_dim >= 1, "comm.message_dim must be >= 1")
        _require(self.key_dim >= 1, "comm.key_dim must be >= 1")
        _require(0.0 <= self.packet_loss <= 1.0, "comm.packet_loss must be in [0, 1]")
        _require(self.delay_steps == 1, "comm.delay_steps: only 1-step delay is supported")
        _require(self.bytes_per_element in (1, 2, 4, 8), "comm.bytes_per_element must be 1/2/4/8")


@dataclass(frozen=True)
class TrainConfig:
    """Rollout and run-length settings.

    ``total_env_steps`` counts sub-environment steps (vector steps x num_envs).
    ``policy: random`` runs the M1 smoke rollout (no learning); ``mappo`` trains.
    """

    policy: str = "random"
    num_envs: int = 4
    num_workers: int = 0  # 0 = step envs in-process; W > 0 = W subprocess workers
    rollout_length: int = 32
    total_env_steps: int = 256
    log_every_steps: int = 0  # random policy only: 0 = log once per rollout
    checkpoint_every_updates: int = 10  # 0 = only the final checkpoint
    # PyTorch CPU threads in the trainer process (0 = PyTorch default). Keep small when
    # env workers share the CPU: measured thrashing at the default on a shared CPU.
    torch_threads: int = 0

    def validate(self) -> None:
        _require(
            self.policy in SUPPORTED_POLICIES, f"train.policy must be one of {SUPPORTED_POLICIES}"
        )
        _require(self.num_envs >= 1, "train.num_envs must be >= 1")
        _require(
            0 <= self.num_workers <= self.num_envs,
            "train.num_workers must be in [0, train.num_envs]",
        )
        _require(self.rollout_length >= 1, "train.rollout_length must be >= 1")
        _require(self.total_env_steps >= 1, "train.total_env_steps must be >= 1")
        _require(self.log_every_steps >= 0, "train.log_every_steps must be >= 0")
        _require(self.checkpoint_every_updates >= 0, "train.checkpoint_every_updates must be >= 0")
        _require(self.torch_threads >= 0, "train.torch_threads must be >= 0")


SUPPORTED_POLICIES = ("random", "mappo")


@dataclass(frozen=True)
class ModelConfig:
    """Shared actor/critic network sizes."""

    hidden_dim: int = 128  # actor CNN output and GRU size
    conv_channels: int = 32
    critic_hidden_dim: int = 128
    # Feed the agent's own normalized (x, y) to the actor (D31). Needed for beliefs in
    # absolute grid coordinates (M4/M5); off keeps the M2/M3 architecture unchanged.
    use_own_position: bool = False

    def validate(self) -> None:
        _require(self.hidden_dim >= 1, "model.hidden_dim must be >= 1")
        _require(self.conv_channels >= 1, "model.conv_channels must be >= 1")
        _require(self.critic_hidden_dim >= 1, "model.critic_hidden_dim must be >= 1")


@dataclass(frozen=True)
class PPOConfig:
    """MAPPO hyperparameters. Every loss coefficient is explicit and logged."""

    lr_actor: float = 5e-4
    lr_critic: float = 5e-4
    adam_eps: float = 1e-5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_clip_eps: float = 0.2  # 0 disables value clipping
    epochs: int = 5
    num_minibatches: int = 2
    # Recurrent sequence length for updates (truncated BPTT). It bounds how far back
    # memory can be learned: gradients never cross a chunk boundary.
    chunk_length: int = 16
    entropy_coef: float = 0.01
    value_coef: float = 1.0
    max_grad_norm: float = 10.0
    huber_delta: float = 10.0  # 0 = squared error
    use_value_norm: bool = True
    normalize_advantages: bool = True
    target_kl: float = 0.0  # 0 disables early stopping on KL

    def validate(self) -> None:
        for name in ("lr_actor", "lr_critic", "adam_eps"):
            _require(getattr(self, name) > 0, f"ppo.{name} must be > 0")
        _require(0.0 < self.gamma <= 1.0, "ppo.gamma must be in (0, 1]")
        _require(0.0 <= self.gae_lambda <= 1.0, "ppo.gae_lambda must be in [0, 1]")
        _require(0.0 < self.clip_eps < 1.0, "ppo.clip_eps must be in (0, 1)")
        _require(self.value_clip_eps >= 0.0, "ppo.value_clip_eps must be >= 0")
        _require(self.epochs >= 1, "ppo.epochs must be >= 1")
        _require(self.num_minibatches >= 1, "ppo.num_minibatches must be >= 1")
        _require(self.chunk_length >= 1, "ppo.chunk_length must be >= 1")
        _require(self.entropy_coef >= 0.0, "ppo.entropy_coef must be >= 0")
        _require(self.value_coef > 0.0, "ppo.value_coef must be > 0")
        _require(self.max_grad_norm > 0.0, "ppo.max_grad_norm must be > 0")
        _require(self.huber_delta >= 0.0, "ppo.huber_delta must be >= 0")
        _require(self.target_kl >= 0.0, "ppo.target_kl must be >= 0")


BELIEF_TYPES = ("none", "point")


@dataclass(frozen=True)
class BeliefConfig:
    """Interagent belief learning (DIABL family). ``point`` (M4): each agent regresses the
    absolute position of the nearest team-visible evader from its post-message state."""

    type: str = "none"
    coef: float = 1.0  # weight of the belief loss in the actor loss

    def validate(self) -> None:
        _require(self.type in BELIEF_TYPES, f"belief.type must be one of {BELIEF_TYPES}")
        _require(self.coef >= 0.0, "belief.coef must be >= 0")


@dataclass(frozen=True)
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    comm: CommConfig = field(default_factory=CommConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    belief: BeliefConfig = field(default_factory=BeliefConfig)

    def validate(self) -> None:
        for section in dataclasses.fields(self):
            getattr(self, section.name).validate()
        if self.belief.type != "none":
            _require(
                self.comm.enabled and self.model.use_own_position,
                "belief learning needs comm.enabled (beliefs are learned THROUGH the channel) "
                "and model.use_own_position (targets are absolute coordinates; D31)",
            )
        if self.train.policy == "mappo":
            train, ppo = self.train, self.ppo
            _require(
                train.rollout_length % ppo.chunk_length == 0,
                f"train.rollout_length ({train.rollout_length}) must be divisible by "
                f"ppo.chunk_length ({ppo.chunk_length})",
            )
            units = train.num_envs * (train.rollout_length // ppo.chunk_length)
            _require(
                ppo.num_minibatches <= units,
                f"ppo.num_minibatches ({ppo.num_minibatches}) exceeds the {units} "
                "(env, chunk) sequences per rollout",
            )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _spawn_windows(size: int, window: float) -> set[tuple[int, int]]:
    """All ``[lo, hi)`` index ranges PettingZoo's reset can draw along one axis.

    Mirrors ``pursuit_base.Pursuit.reset``: ``u ~ U(0, 1 - window)``,
    ``lo = int(size * u)``, ``hi = int(size * (u + window))``. The pair only changes
    where ``size * u`` or ``size * (u + window)`` crosses an integer, so checking
    those breakpoints and the midpoints between them covers every case.
    """
    span = 1.0 - window
    points = {0.0, span}
    for k in range(size + 1):
        for u in (k / size, k / size - window):
            if 0.0 <= u <= span:
                points.add(u)
    ordered = sorted(points)
    ordered += [(a + b) / 2 for a, b in zip(ordered, ordered[1:], strict=False)]
    return {(int(size * u), int(size * (u + window))) for u in ordered}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _coerce(value: Any, annotation: Any, path: str) -> Any:
    """Check ``value`` against a simple type annotation; ints are accepted for floats."""
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        for option in typing.get_args(annotation):
            try:
                return _coerce(value, option, path)
            except ConfigError:
                continue
        raise ConfigError(f"{path}: {value!r} does not match {annotation}")
    if annotation is type(None):
        if value is None:
            return None
        raise ConfigError(f"{path}: expected null, got {value!r}")
    if annotation is bool:
        if isinstance(value, bool):
            return value
        raise ConfigError(f"{path}: expected bool, got {type(value).__name__} {value!r}")
    if annotation is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        raise ConfigError(f"{path}: expected int, got {type(value).__name__} {value!r}")
    if annotation is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not math.isfinite(value):
                raise ConfigError(f"{path}: expected a finite float, got {value!r}")
            return float(value)
        raise ConfigError(f"{path}: expected float, got {type(value).__name__} {value!r}")
    if annotation is str:
        if isinstance(value, str):
            return value
        raise ConfigError(f"{path}: expected str, got {type(value).__name__} {value!r}")
    raise ConfigError(f"{path}: unsupported annotation {annotation}")


def _build_section(cls: type, raw: Any, path: str) -> Any:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a mapping, got {type(raw).__name__}")
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ConfigError(f"{path}: unknown key(s) {unknown}; allowed: {sorted(known)}")
    values = {key: _coerce(val, hints[key], f"{path}.{key}") for key, val in raw.items()}
    return cls(**values)


def config_from_dict(raw: dict[str, Any] | None) -> Config:
    """Build and validate a ``Config`` from a nested mapping."""
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"config root must be a mapping, got {type(raw).__name__}")
    hints = typing.get_type_hints(Config)
    known = {f.name for f in dataclasses.fields(Config)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ConfigError(f"unknown top-level section(s) {unknown}; allowed: {sorted(known)}")
    sections = {name: _build_section(hints[name], raw.get(name), name) for name in known}
    config = Config(**sections)
    config.validate()
    return config


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys instead of keeping the last one."""


def _construct_unique_mapping(loader: yaml.SafeLoader, node: yaml.MappingNode) -> dict[Any, Any]:
    seen: set[Any] = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=True)
        if key in seen:
            raise ConfigError(f"duplicate key {key!r} at line {key_node.start_mark.line + 1}")
        seen.add(key)
    return loader.construct_mapping(node, deep=True)


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Load a YAML config, apply dotted overrides (e.g. ``{"experiment.seed": 3}``), validate."""
    with open(path, encoding="utf-8") as handle:
        raw = yaml.load(handle, Loader=_UniqueKeyLoader) or {}  # noqa: S506 - SafeLoader subclass
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: config root must be a mapping")
    return config_with_overrides(raw, overrides)


def config_with_overrides(raw: dict[str, Any], overrides: dict[str, Any] | None = None) -> Config:
    """Apply dotted overrides to a nested mapping (copied), then build and validate."""
    raw = {k: dict(v) if isinstance(v, dict) else v for k, v in raw.items()}
    for dotted, value in (overrides or {}).items():
        section, _, key = dotted.partition(".")
        if not key:
            raise ConfigError(f"override {dotted!r} must look like section.key")
        if raw.get(section) is None:
            raw[section] = {}
        if not isinstance(raw[section], dict):
            raise ConfigError(f"override {dotted!r}: section {section!r} is not a mapping")
        raw[section][key] = value
    return config_from_dict(raw)


def parse_override(text: str) -> tuple[str, Any]:
    """Parse a CLI override ``section.key=value``; the value is parsed as YAML."""
    if "=" not in text:
        raise ConfigError(f"override {text!r} must look like section.key=value")
    key, _, value = text.partition("=")
    return key.strip(), yaml.safe_load(value)
