"""Evaluation grid: (checkpoint x condition) cells on one CPU, one torch thread each.

Each cell writes ``<out>/<arm>/s<seed>/<condition>/{summary.json, episodes.jsonl}``; a cell
with a ``summary.json`` is never recomputed. Also holds the pre-registered threshold rules
used by ``scripts/calibrate_thresholds.py``.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any

CORE = ["iid-p0", "iid-p0.2", "iid-p0.5", "iid-p0.8", "iid-p1"]
CHANNEL = CORE + ["ge-p0.5-b5", "ge-p0.8-b5", "ge-p0.8-b20", "ge-p0.8-b50", "iid-p0.2-n6",
                  "iid-p0.2-n10", "$random", "$periodic", "$trigger"]  # fmt: skip
PLANS = {"core": CORE, "channel": CHANNEL}
RANDOM_RATES = (0.5, 0.2, 0.1, 0.05, 0.02, 0.01)


def plan_conditions(plan: str, thresholds: dict[str, Any] | None = None) -> list[str]:
    """A named plan or a comma list; ``$key`` labels come from the thresholds record and are
    dropped when no thresholds are given."""
    labels = PLANS.get(plan, plan.split(","))
    out = []
    for label in labels:
        if label.startswith("$"):
            if thresholds is None:
                continue
            label = thresholds[label[1:]]
        if label not in out:
            out.append(label)
    return out


def pick_bstar(full: float, captures_by_rate: dict[float, float]) -> float:
    """Going down the random send rates, the first whose capture < 0.95 x full (else 1 %)."""
    for rate in RANDOM_RATES:
        if captures_by_rate[rate] < 0.95 * full:
            return rate
    return RANDOM_RATES[-1]


def pick_hard(p0: float, p1: float, captures: dict[str, float]) -> str:
    """The condition whose capture is closest to the midpoint of 0 % and 100 % loss."""
    mid = (p0 + p1) / 2
    return min(captures, key=lambda label: (abs(captures[label] - mid), label))


@dataclass(frozen=True)
class Job:
    arm: str
    checkpoint: Path
    condition: str
    stream: str
    episodes: int
    out_dir: Path


def cpu_model() -> str:
    try:
        if platform.system() == "Darwin":
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True, check=True)  # fmt: skip
            return out.stdout.strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or "unknown"


def _seed_of(checkpoint: Path) -> int:
    import torch

    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return int(data["config"]["experiment"]["seed"])


def run_job(job: Job, seed: int) -> dict[str, Any]:
    """Evaluate one cell (in a worker process). Returns its summary."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch

    from dynabelief.config import config_from_dict
    from dynabelief.evaluation.conditions import EVAL_BASE_SEED, Condition
    from dynabelief.evaluation.evaluate import MAPPOAgent, evaluate, load_policy_for_eval
    from dynabelief.utils.checkpointing import load_checkpoint
    from dynabelief.utils.logging import git_state

    torch.set_num_threads(1)
    cell = Path(job.out_dir) / job.arm / f"s{seed}" / job.condition
    summary_path = cell / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    cell.mkdir(parents=True, exist_ok=True)
    (cell / "episodes.jsonl").unlink(missing_ok=True)  # a crashed cell starts over

    cond = Condition.parse(job.condition)
    raw = load_checkpoint(job.checkpoint, map_location="cpu")["config"]
    raw = json.loads(json.dumps(raw))
    raw["experiment"]["device"] = "cpu"
    if cond.n_pursuers is not None:
        raw["env"]["n_pursuers"] = cond.n_pursuers
    config = config_from_dict(raw)
    policy, data = load_policy_for_eval(job.checkpoint, config, "cpu")
    agent = MAPPOAgent(policy, EVAL_BASE_SEED, "cpu", deterministic=cond.greedy)
    extra = {"arm": job.arm, "seed": seed, "checkpoint": str(job.checkpoint)}
    with torch.no_grad():
        summary = evaluate(config, agent, job.episodes, cell, extra=extra, condition=cond,
                           stream=job.stream)  # fmt: skip
    summary.update(
        arm=job.arm,
        seed=seed,
        checkpoint=str(job.checkpoint),
        checkpoint_env_steps=int(data["env_steps"]),
        commit=git_state()["commit"],
        cpu=cpu_model(),
        torch=torch.__version__,
        threads=torch.get_num_threads(),
    )
    tmp = summary_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(summary, indent=1))
    tmp.replace(summary_path)
    return summary


def run_jobs(jobs: list[Job], workers: int) -> list[dict[str, Any]]:
    """Run cells in parallel (spawned processes, 1 thread each), in input order."""
    seeds = {ckpt: _seed_of(ckpt) for ckpt in {job.checkpoint for job in jobs}}
    if workers <= 1:
        return [run_job(job, seeds[job.checkpoint]) for job in jobs]
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool:
        futures = [pool.submit(run_job, job, seeds[job.checkpoint]) for job in jobs]
        return [f.result() for f in futures]
