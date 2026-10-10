"""Checkpoints: model, optimizers, value normalizer, counters, config, and RNG state.

Resume semantics: everything the learner owns is restored exactly (weights,
Adam moments, ValueNorm stats, Python/NumPy/Torch/minibatch RNG streams, step
counters). PettingZoo's simulator state cannot be serialized, so a resumed run
starts every environment on its NEXT episode seed (``episode_index``). Episodes
that were in progress at save time are dropped.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

FORMAT_VERSION = 1


class CheckpointError(RuntimeError):
    """Missing, corrupt, or incompatible checkpoint."""


def rng_state(minibatch_generator: torch.Generator | None = None) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),  # noqa: NPY002 - legacy global RNG is part of state
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    if minibatch_generator is not None:
        state["minibatch"] = minibatch_generator.get_state()
    return state


def restore_rng_state(
    state: dict[str, Any], minibatch_generator: torch.Generator | None = None
) -> None:
    # RNG states must be CPU ByteTensors even if the checkpoint was mapped to a GPU.
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])  # noqa: NPY002
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["torch_cuda"]])
    if minibatch_generator is not None:
        if "minibatch" not in state:
            raise CheckpointError("checkpoint has no minibatch generator state")
        minibatch_generator.set_state(state["minibatch"].cpu())


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write atomically (temp file + rename) so a crash never leaves a torn file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    torch.save({"format_version": FORMAT_VERSION, **payload}, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise CheckpointError(f"checkpoint not found: {path}")
    try:
        # weights_only=False: the payload holds RNG tuples and plain config dicts we wrote.
        data = torch.load(path, map_location=map_location, weights_only=False)
    except Exception as exc:  # torch raises many types for corrupt files
        raise CheckpointError(f"cannot read checkpoint {path}: {exc}") from exc
    version = data.get("format_version") if isinstance(data, dict) else type(data).__name__
    if version != FORMAT_VERSION:
        raise CheckpointError(f"{path}: unsupported checkpoint format {version!r}")
    return data


# Fields that change the network's tensor shapes or meaning.
_SHAPE_FIELDS = {
    "env": ("x_size", "y_size", "obs_range"),
    "comm": ("enabled", "key_dim", "message_dim"),  # architecture; packet_loss may change
}


def check_compatible(saved: dict[str, Any], current: dict[str, Any], resume: bool) -> None:
    """Raise ``CheckpointError`` if ``current`` cannot use a checkpoint saved with ``saved``.

    Evaluation may change agent counts, evader counts, packet loss, episode
    length, etc. (generalization / robustness tests) but not the model, the
    communication architecture, or grid/view shapes.
    Resuming training requires the same experiment definition (only run-length
    and logging fields may differ).
    """
    problems = []
    if saved.get("model") != current.get("model"):
        problems.append(f"model: {saved.get('model')} != {current.get('model')}")
    for section, keys in _SHAPE_FIELDS.items():
        for key in keys:
            a, b = saved.get(section, {}).get(key), current.get(section, {}).get(key)
            if a != b:
                problems.append(f"{section}.{key}: {a} != {b}")
    if resume:
        for section in ("env", "comm", "ppo"):
            if saved.get(section) != current.get(section):
                problems.append(f"{section} differs (resume needs the same experiment)")
        mutable = {
            "total_env_steps", "checkpoint_every_updates", "log_every_steps", "num_workers",
            "torch_threads",
        }  # fmt: skip
        saved_train = {k: v for k, v in saved.get("train", {}).items() if k not in mutable}
        current_train = {k: v for k, v in current.get("train", {}).items() if k not in mutable}
        if saved_train != current_train:
            problems.append(f"train: {saved_train} != {current_train}")
        if saved.get("experiment", {}).get("seed") != current.get("experiment", {}).get("seed"):
            problems.append("experiment.seed differs")
    if problems:
        raise CheckpointError("incompatible checkpoint:\n  " + "\n  ".join(problems))
