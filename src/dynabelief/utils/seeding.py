"""Deterministic seeding.

Every stochastic component gets its own generator derived from the experiment
seed via ``numpy.random.SeedSequence``. Streams are keyed by name so adding a
new consumer never shifts the random numbers seen by an existing one.
"""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic_torch: bool = False) -> None:
    """Seed Python, NumPy's legacy global RNG, and PyTorch (CPU and CUDA)."""
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")
    random.seed(seed)
    np.random.seed(seed % 2**32)  # noqa: NPY002 - seeds legacy global RNG used by third-party code
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False


def _stream_key(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:4], "little")


def _sequence(seed: int, stream: str, indices: tuple[int, ...]) -> np.random.SeedSequence:
    if seed < 0 or any(i < 0 for i in indices):
        raise ValueError(f"seed and indices must be >= 0, got {seed}, {indices}")
    return np.random.SeedSequence([seed, _stream_key(stream), len(indices), *indices])


def derive_seed(seed: int, stream: str, *indices: int) -> int:
    """A 64-bit seed for a named stream, e.g. ``derive_seed(s, "train", env_i, episode_k)``.

    Distinct streams/indices give distinct seeds except with probability ~N^2 / 2^65
    (about 1e-8 for a million episodes), which is negligible but not zero.
    """
    return int(_sequence(seed, stream, indices).generate_state(1, dtype=np.uint64)[0])


def make_rng(seed: int, stream: str, *indices: int) -> np.random.Generator:
    """An independent NumPy generator for a named stream."""
    return np.random.default_rng(_sequence(seed, stream, indices))
