"""Entry point that runs INSIDE a Kaggle kernel (launched by ``cloud/kaggle_launch.py``).

The generated kernel script clones the GitHub repo at one pushed commit into scratch
space and runs this file with the job's arguments, so every job is reproducible from
its commit hash (the Kaggle analogue of D12). The package is not pip-installed: the
Kaggle image's Python may be newer than ``requires-python`` allows, so ``src`` goes on
``PYTHONPATH`` and only the missing dependencies are installed.

Modes:

* ``bench``: time a few PPO updates per env-worker count (collection vs update time)
  and project the wall time of a 5M-step run, to size runs against the 12 h session
  limit and the weekly GPU quota.
* ``tests``: the full test suite on the Kaggle software stack.
* ``train``: one training run. It stops itself before the session limit; a run that
  hits the limit is incomplete and gets rerun from scratch, never resumed (a resume
  drops in-progress episodes, so it is not bit-identical).

Only ``/kaggle/working`` survives the session (it becomes the version's output). The
script always writes ``status.json`` there and exits 0, so outputs and logs are kept
even when a step fails; ``kaggle_launch.py download`` reports the real status.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
WORKING = Path(os.environ.get("DYNABELIEF_KAGGLE_WORKING", "/kaggle/working"))
DEPS = ["pettingzoo==1.25.0", "gymnasium>=1.0,<2", "pygame>=2.3"]
STEPS_PER_UPDATE = 32 * 128  # num_envs x rollout_length of every study config


def child_env() -> dict[str, str]:
    paths = [str(CODE / "src"), str(CODE)]
    if os.environ.get("PYTHONPATH"):
        paths.append(os.environ["PYTHONPATH"])
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(paths),
        "PYTHONUNBUFFERED": "1",
        "CUDA_VISIBLE_DEVICES": "0",  # T4 x2: one run uses one GPU
        "SDL_VIDEODRIVER": "dummy",
    }


def run(cmd: list[str], timeout: float | None = None, log: Path | None = None) -> int:
    """Run ``cmd`` in its own process group; on timeout, SIGTERM the group, then SIGKILL."""
    print("+", " ".join(cmd), flush=True)
    out = open(log, "a") if log else None  # noqa: SIM115
    proc = subprocess.Popen(cmd, cwd=CODE, env=child_env(), start_new_session=True,
                            stdout=out, stderr=subprocess.STDOUT if out else None)  # fmt: skip
    try:
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"timeout after {timeout:.0f} s: stopping {cmd[1]}", flush=True)
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        return -signal.SIGTERM
    finally:
        if out:
            out.close()


def install_command(dev: bool, version: tuple[int, int] = tuple(sys.version_info[:2])) -> list:
    pkgs = DEPS + (["pytest>=8"] if dev else [])
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir"]
    if version >= (3, 13):
        # pettingzoo 1.25 declares Requires-Python <3.13 but is pure Python and runs on the
        # Kaggle image's 3.13 (verified: env, filter and training tests pass on 3.13).
        cmd.append("--ignore-requires-python")
    return cmd + pkgs


def install(dev: bool) -> int:
    return run(install_command(dev))


def _version(module: str) -> str | None:
    try:
        return __import__(module).__version__
    except Exception:  # noqa: BLE001 - report, never fail
        return None


def environment_report() -> dict:
    """Software and hardware of this session, written next to every job's outputs."""
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "vcpus": os.cpu_count(),
        "vcpus_usable": (
            len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
        ),
        "kaggle": {
            k: os.environ[k]
            for k in ("KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_DOCKER_IMAGE", "KAGGLE_URL_BASE")
            if k in os.environ
        },
    }
    for module in ("torch", "numpy", "pettingzoo", "gymnasium", "pygame", "scipy"):
        report[module] = _version(module)
    try:
        cpu = next(line for line in Path("/proc/cpuinfo").read_text().splitlines()
                   if line.startswith("model name"))  # fmt: skip
        report["cpu_model"] = cpu.split(":", 1)[1].strip()
    except (OSError, StopIteration):
        pass
    try:
        import torch

        report["cuda"] = torch.cuda.is_available()
        if report["cuda"]:
            report["gpu"] = torch.cuda.get_device_name(0)
            report["cuda_version"] = torch.version.cuda
    except Exception:  # noqa: BLE001
        report["cuda"] = False
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=CODE, capture_output=True,
                            text=True, check=False).stdout.strip()  # fmt: skip
    report["commit"] = commit or None
    return report


def bench_summary(metrics: list[dict], total_env_steps: int = 5_000_000) -> dict:
    """Median collection and update time per PPO update (first update skipped as warm-up)
    and the projected training hours for ``total_env_steps``."""
    rows = metrics[1:] if len(metrics) > 1 else metrics
    collect = sorted(STEPS_PER_UPDATE / r["collect_fps"] for r in rows if r.get("collect_fps"))
    update = sorted(r["update_seconds"] for r in rows if r.get("update_seconds") is not None)
    if not collect or not update:
        return {"updates": len(metrics), "error": "no timing fields in metrics.jsonl"}
    c, u = collect[len(collect) // 2], update[len(update) // 2]
    updates = -(-total_env_steps // STEPS_PER_UPDATE)
    return {
        "updates": len(metrics),
        "collect_seconds_per_update": round(c, 3),
        "update_seconds": round(u, 3),
        "seconds_per_update": round(c + u, 3),
        "projected_hours_5M": round(updates * (c + u) / 3600, 2),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, default=str))
    print(f"wrote {path}", flush=True)


def train_cmd(config: str, output_dir: Path, overrides: list[str]) -> list[str]:
    cmd = [sys.executable, "scripts/train.py", "--config", config, "--output-dir", str(output_dir)]
    for override in overrides:
        cmd += ["--set", override]
    return cmd


def mode_bench(args: argparse.Namespace, status: dict) -> None:
    from dynabelief.utils.logging import read_jsonl

    results = {}
    for workers in [int(w) for w in args.bench_workers.split(",")]:
        out = WORKING / "bench" / f"workers{workers}"
        overrides = [*args.set, f"train.num_workers={workers}",
                     f"train.torch_threads={args.torch_threads}",
                     f"train.total_env_steps={args.bench_updates * STEPS_PER_UPDATE}"]  # fmt: skip
        start = time.time()
        code = run(train_cmd(args.config, out, overrides), timeout=3600)
        metrics_files = sorted(out.glob("*/metrics.jsonl"))
        summary = bench_summary(read_jsonl(metrics_files[-1])) if metrics_files else {}
        results[f"workers{workers}"] = {**summary, "returncode": code,
                                        "wall_seconds": round(time.time() - start, 1)}  # fmt: skip
        print(json.dumps(results[f"workers{workers}"]), flush=True)
    write_json(WORKING / "bench.json", {"config": args.config, "overrides": args.set, **results})
    status["ok"] = all(r["returncode"] == 0 for r in results.values())


def mode_tests(args: argparse.Namespace, status: dict) -> None:
    code = run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                f"--junitxml={WORKING / 'pytest.xml'}", *args.pytest_args],
               timeout=args.time_limit_hours * 3600, log=WORKING / "pytest.txt")  # fmt: skip
    print((WORKING / "pytest.txt").read_text()[-3000:], flush=True)
    status.update(ok=code == 0, returncode=code)


def mode_train(args: argparse.Namespace, status: dict) -> None:
    runs_dir = WORKING / "runs"
    overrides = [*args.set, f"train.num_workers={args.workers}",
                 f"train.torch_threads={args.torch_threads}"]  # fmt: skip
    start = time.time()
    code = run(train_cmd(args.config, runs_dir, overrides), timeout=args.time_limit_hours * 3600)
    finals = sorted(runs_dir.glob("*/checkpoints/final.pt"))
    status.update(returncode=code, wall_hours=round((time.time() - start) / 3600, 3),
                  complete=bool(finals), ok=code == 0 and bool(finals))  # fmt: skip
    for run_dir in sorted(p.parent for p in runs_dir.glob("*/checkpoints")):
        run([sys.executable, "scripts/plot_metrics.py", str(run_dir)], timeout=600)
        status["run_dir"] = str(run_dir.relative_to(WORKING))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("bench", "tests", "train"), required=True)
    parser.add_argument("--config", default="configs/point_diabl.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    parser.add_argument("--workers", type=int, default=4, help="env workers (train mode)")
    parser.add_argument("--torch_threads", type=int, default=2)
    parser.add_argument("--bench_workers", default="2,4")
    parser.add_argument("--bench_updates", type=int, default=15)
    parser.add_argument("--time_limit_hours", type=float, default=11.25)
    parser.add_argument("--pytest_args", nargs="*", default=[])
    parser.add_argument("--skip_install", action="store_true", help="local dry runs only")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    WORKING.mkdir(parents=True, exist_ok=True)
    status: dict = {"mode": args.mode, "config": args.config, "overrides": args.set, "ok": False}
    try:
        if not args.skip_install and install(dev=args.mode == "tests") != 0:
            status["error"] = "pip install failed"
            return 0
        write_json(WORKING / "environment.json", environment_report())
        {"bench": mode_bench, "tests": mode_tests, "train": mode_train}[args.mode](args, status)
    except Exception as exc:  # noqa: BLE001 - recorded in status.json, outputs kept
        status["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        write_json(WORKING / "status.json", status)
    return 0


if __name__ == "__main__":
    sys.exit(main())
