"""Entry point that runs INSIDE a SageMaker training job (AWS PyTorch DLC image).

Launched by ``cloud/launch.py``; never run this locally. Steps:

1. ``pip install .`` the repo snapshot (a ``git archive`` of one clean commit).
2. Train with ``scripts/train.py``. Runs are written under ``/opt/ml/checkpoints``,
   which SageMaker syncs to S3 continuously, so metrics and checkpoints are visible
   while the job runs. If the job restarted (e.g. a spot interruption), it resumes
   from the newest ``latest.pt`` found there.
3. Evaluate ``final.pt`` on the held-out eval seeds (sampled and greedy actions) and
   plot the learning curves.
4. Copy the small result files to ``/opt/ml/model`` (exported as ``model.tar.gz``).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
CHECKPOINTS = Path(os.environ.get("SM_CHECKPOINT_DIR", "/opt/ml/checkpoints"))
MODEL = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=CODE)


def pick_workers(num_envs: int, vcpus: int, reserve: int = 2) -> int:
    """Largest divisor of ``num_envs`` that leaves ``reserve`` vCPUs for the trainer.

    A divisor gives every worker the same number of envs: each vector step waits for
    the slowest worker, so an uneven split makes stragglers.
    """
    budget = max(1, vcpus - reserve)
    return max(d for d in range(1, num_envs + 1) if num_envs % d == 0 and d <= budget)


def find_resume(runs_dir: Path) -> Path | None:
    candidates = sorted(runs_dir.glob("*/checkpoints/latest.pt"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def newest_final(runs_dir: Path) -> Path:
    finals = sorted(runs_dir.glob("*/checkpoints/final.pt"), key=lambda p: p.stat().st_mtime)
    if not finals:
        raise SystemExit(f"training finished without a final.pt under {runs_dir}")
    return finals[-1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("train", "tests"), default="train")
    parser.add_argument("--config", help="repo-relative config (train mode)")
    # nargs="?": the toolkit passes a bare flag when a value is empty. Values arrive
    # unquoted on a shell command line, hence base64 JSON for free-form overrides.
    parser.add_argument("--overrides_b64", nargs="?", const="", default="",
                        help="base64 JSON list of section.key=value")  # fmt: skip
    parser.add_argument("--eval_episodes", type=int, default=50)
    parser.add_argument("--eval_packet_loss", nargs="?", const="", default="",
                        help="comma-separated loss sweep for sampled evals")  # fmt: skip
    parser.add_argument("--workers", type=int, default=0, help="0 = derive from vCPUs")
    parser.add_argument("--torch_threads", type=int, default=2)
    parser.add_argument("--skip_install", action="store_true", help="local testing only")
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print(f"ignoring unknown args {unknown}", flush=True)
    if args.mode == "train" and not args.config:
        parser.error("--config is required in train mode")
    return args


def decode_list(encoded: str) -> list[str]:
    import base64
    import json

    return json.loads(base64.b64decode(encoded)) if encoded else []


def eval_plan(comm_enabled: bool, train_loss: float, sweep: str) -> list[tuple[str, list[str]]]:
    """(label, extra evaluate.py args): sampled actions at every swept packet loss (or at
    the training loss) plus greedy actions at the training loss."""
    losses = [float(x) for x in sweep.split(",") if x.strip()] if comm_enabled else []
    plan = [(f"sampled_loss{loss:g}", ["--packet-loss", str(loss)]) for loss in losses]
    if not plan:
        plan = [("sampled", [])]
    plan.append(("greedy", ["--deterministic", "--packet-loss", str(train_loss)]))
    return plan


def run_tests() -> int:
    run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--quiet", ".[dev]"])
    MODEL.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         f"--junitxml={MODEL / 'pytest.xml'}"])  # fmt: skip
    print("done", flush=True)
    return 0


def main() -> int:
    args = parse_args()
    if args.mode == "tests":
        return run_tests()
    if not args.skip_install:
        run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--quiet", "."])
    from dynabelief.config import load_config, parse_override

    overrides = decode_list(args.overrides_b64)
    config = load_config(CODE / args.config, dict(parse_override(o) for o in overrides))
    workers = args.workers or pick_workers(config.train.num_envs, os.cpu_count() or 2)
    overrides += [f"train.num_workers={workers}", f"train.torch_threads={args.torch_threads}"]
    print(f"vCPUs={os.cpu_count()} -> {workers} env workers", flush=True)

    runs_dir, evals_dir = CHECKPOINTS / "runs", CHECKPOINTS / "evals"
    resume = find_resume(runs_dir)
    cmd = [sys.executable, "scripts/train.py", "--config", args.config,
           "--output-dir", str(runs_dir)]  # fmt: skip
    for override in overrides:
        cmd += ["--set", override]
    if resume is not None:
        cmd += ["--resume", str(resume)]
        print(f"resuming from {resume}", flush=True)
    run(cmd)

    final = newest_final(runs_dir)
    run_dir = final.parents[1]
    plan = []
    if args.eval_episodes > 0:
        plan = eval_plan(config.comm.enabled, config.comm.packet_loss, args.eval_packet_loss)
        # Each evaluation is a single-process episode loop: run them all concurrently.
        procs = []
        for label, extra in plan:
            eval_cmd = [sys.executable, "scripts/evaluate.py", "--checkpoint", str(final),
                        "--episodes", str(args.eval_episodes),
                        "--output-dir", str(evals_dir / label), *extra]  # fmt: skip
            print("+", " ".join(eval_cmd), flush=True)
            procs.append(subprocess.Popen(eval_cmd, cwd=CODE))
        failed = [label for (label, _), p in zip(plan, procs, strict=True) if p.wait() != 0]
        if failed:
            raise SystemExit(f"evaluations failed: {failed}")
    run([sys.executable, "scripts/plot_metrics.py", str(run_dir)])

    MODEL.mkdir(parents=True, exist_ok=True)
    for name in ("config.yaml", "metadata.yaml", "summary.json", "curves.png"):
        if (run_dir / name).exists():
            shutil.copy2(run_dir / name, MODEL / name)
    for label, _ in plan:
        for summary in (evals_dir / label).glob("*/summary.json"):
            shutil.copy2(summary, MODEL / f"eval_{label}.json")
    shutil.copy2(final, MODEL / "final.pt")
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
