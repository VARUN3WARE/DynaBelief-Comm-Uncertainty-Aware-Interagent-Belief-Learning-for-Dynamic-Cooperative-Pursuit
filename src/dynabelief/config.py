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
    """

    enabled: bool = True
    message_dim: int = 16
    packet_loss: float = 0.0
    delay_steps: int = 1
    bytes_per_element: int = 4  # float32 on the wire

    def validate(self) -> None:
        _require(self.message_dim >= 1, "comm.message_dim must be >= 1")
        _require(0.0 <= self.packet_loss <= 1.0, "comm.packet_loss must be in [0, 1]")
        _require(self.delay_steps == 1, "comm.delay_steps: only 1-step delay is supported")
        _require(self.bytes_per_element in (1, 2, 4, 8), "comm.bytes_per_element must be 1/2/4/8")


@dataclass(frozen=True)
class TrainConfig:
    """Rollout settings. ``policy`` is ``random`` until MAPPO lands (M2)."""

    policy: str = "random"
    num_envs: int = 4
    num_workers: int = 0  # 0 = step envs in-process; W > 0 = W subprocess workers
    rollout_length: int = 32
    total_env_steps: int = 256
    log_every_steps: int = 0  # 0 = log once per rollout

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


SUPPORTED_POLICIES = ("random",)


@dataclass(frozen=True)
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    comm: CommConfig = field(default_factory=CommConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def validate(self) -> None:
        for section in dataclasses.fields(self):
            getattr(self, section.name).validate()

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
