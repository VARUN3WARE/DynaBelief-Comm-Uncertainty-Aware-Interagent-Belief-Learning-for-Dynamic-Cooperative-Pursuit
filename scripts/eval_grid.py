"""Evaluate checkpoints under a plan of conditions (CPU, 1 thread per cell, resumable).

Usage::

    PYTHONPATH=src:. .venv/bin/python scripts/eval_grid.py \
        --checkpoint point_diabl=runs/.../final.pt --checkpoint tarmac_pos=runs/.../final.pt \
        --plan channel --thresholds docs/results/thresholds.json \
        --stream test --episodes 100 --workers 6 --out runs/evals/final

Cells go to ``<out>/<arm>/s<seed>/<condition>/``; the training seed is read from the
checkpoint. Plans: ``core``, ``channel`` or a comma list of labels (``$random``,
``$periodic``, ``$trigger``, ``$hard`` resolve from the thresholds file).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dynabelief.evaluation.grid import Job, plan_conditions, run_jobs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", action="append", required=True, help="ARM=PATH (repeatable)")
    p.add_argument("--plan", default="core")
    p.add_argument("--thresholds", type=Path)
    p.add_argument("--stream", default="test", choices=("calib", "test"))
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", type=Path, default=Path("runs/evals/final"))
    args = p.parse_args(argv)

    thresholds = json.loads(args.thresholds.read_text()) if args.thresholds else None
    labels = plan_conditions(args.plan, thresholds)
    jobs = []
    for spec in args.checkpoint:
        arm, _, path = spec.partition("=")
        if not path or not Path(path).exists():
            raise SystemExit(f"--checkpoint {spec!r}: expected ARM=PATH to an existing file")
        jobs += [Job(arm, Path(path), label, args.stream, args.episodes, args.out)
                 for label in labels]  # fmt: skip
    print(f"{len(jobs)} cells ({len(args.checkpoint)} checkpoints x {len(labels)} conditions)")
    for job, s in zip(jobs, run_jobs(jobs, args.workers), strict=True):
        print(f"{job.arm:24s} s{s['seed']} {job.condition:28s} capture {s['mean_capture_rate']:.3f}"
              + (f"  send {s['send_rate']:.3f}" if "send_rate" in s else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
