"""Run directories, reproducibility metadata, and JSONL metric logs."""

from __future__ import annotations

import json
import logging
import math
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import yaml

TRACKED_PACKAGES = ("dynabelief", "torch", "numpy", "pettingzoo", "gymnasium", "pyyaml")


def get_logger(name: str = "dynabelief") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


GIT_COMMIT_ENV = "DYNABELIEF_GIT_COMMIT"


def git_state(cwd: Path | None = None) -> dict[str, Any]:
    """Commit hash and dirty flag of the repository containing ``cwd``.

    Inside a SageMaker job the code is a ``git archive`` of a clean commit with no
    ``.git`` directory; the launcher passes that commit in ``DYNABELIEF_GIT_COMMIT``.
    """
    cwd = cwd or Path(__file__).resolve().parent
    commit = _git(["rev-parse", "HEAD"], cwd)
    if commit is None and os.environ.get(GIT_COMMIT_ENV):
        return {"commit": os.environ[GIT_COMMIT_ENV], "dirty": False, "source": "git archive"}
    status = _git(["status", "--porcelain"], cwd)
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def device_info(device: str) -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {"device": device, "cuda_available": torch.cuda.is_available()}
    if device.startswith("cuda") and torch.cuda.is_available():
        index = torch.device(device).index or 0
        info["gpu_name"] = torch.cuda.get_device_name(index)
        info["gpu_total_memory_gb"] = round(
            torch.cuda.get_device_properties(index).total_memory / 1024**3, 2
        )
    return info


def run_metadata(device: str, argv: list[str] | None = None) -> dict[str, Any]:
    """``argv`` should be the arguments actually parsed (defaults to ``sys.argv[1:]``)."""
    return {
        "created_utc": datetime.now(UTC).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "entry_point": sys.argv[0],
        "argv": list(sys.argv[1:] if argv is None else argv),
        "git": git_state(),
        "packages": package_versions(),
        "hardware": device_info(device),
    }


def create_run_dir(output_dir: str | Path, name: str, seed: int) -> Path:
    """``<output_dir>/<name>_s<seed>_<UTC timestamp>``; never reuses an existing directory."""
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    base = Path(output_dir) / f"{name}_s{seed}_{stamp}"
    base.parent.mkdir(parents=True, exist_ok=True)
    run_dir, suffix = base, 1
    while True:
        try:
            run_dir.mkdir()  # atomic: concurrent runs can never share a directory
            return run_dir
        except FileExistsError:
            run_dir = base.with_name(f"{base.name}-{suffix}")
            suffix += 1


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"refusing to log non-finite value {value!r}")
    return value


class JsonlWriter:
    """Append-only JSON-lines log. Non-finite floats raise instead of being written."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle = open(path, "a", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        self._handle.write(json.dumps(_jsonable(record), allow_nan=False) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> JsonlWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resource_usage() -> dict[str, float]:
    """RSS of this process and of its worker processes, and peak CUDA memory (GB)."""
    import psutil
    import torch

    process = psutil.Process()
    children_rss = 0
    for child in process.children(recursive=True):
        try:
            children_rss += child.memory_info().rss
        except psutil.Error:  # child exited between listing and reading
            pass
    usage = {
        "ram_rss_gb": process.memory_info().rss / 1024**3,
        "ram_workers_rss_gb": children_rss / 1024**3,
    }
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        usage["cuda_peak_allocated_gb"] = torch.cuda.max_memory_allocated() / 1024**3
    return usage
