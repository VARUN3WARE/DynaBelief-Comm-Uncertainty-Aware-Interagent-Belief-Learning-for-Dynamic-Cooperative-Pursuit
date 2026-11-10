"""Fix the send budget b*, trigger threshold tau and hard channel H* on the calib stream.

Rules (pre-registration section 4), applied to one checkpoint (pilot A, seed 0, final.pt):

* b*: going down the random send rates 50, 20, 10, 5, 2, 1 %, the first rate where capture
  at 0 % loss falls below 0.95 x capture with full communication (1 % if none does).
* tau: the belief-change threshold (grid cells) whose trigger send rate is within 10 % of
  b*: a coarse grid, then bisection, at most 12 trigger evaluations; closest rate wins.
* H*: among the non-zero i.i.d. and Gilbert-Elliott conditions, the one whose capture is
  closest to the midpoint of capture at 0 % and at 100 % loss.

Usage::

    PYTHONPATH=src:. .venv/bin/python scripts/calibrate_thresholds.py \
        --checkpoint runs/sagemaker/.../point_diabl_s0_.../checkpoints/final.pt \
        --output docs/results/thresholds.json --episodes 100 --workers 6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dynabelief.evaluation.grid import (
    RANDOM_RATES,
    Job,
    cpu_model,
    pick_bstar,
    pick_hard,
    run_jobs,
)

HARD_CANDIDATES = [
    "iid-p0.2", "iid-p0.5", "iid-p0.8", "iid-p0.9", "iid-p0.95",
    "ge-p0.5-b5", "ge-p0.5-b20", "ge-p0.5-b50", "ge-p0.8-b5", "ge-p0.8-b20", "ge-p0.8-b50",
]  # fmt: skip
TAU_GRID = (1.0, 2.0, 4.0, 8.0)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cells", type=Path, default=Path("runs/evals/calibration"))
    args = p.parse_args(argv)

    def cells(labels: list[str]) -> dict[str, dict]:
        jobs = [Job("calibration", args.checkpoint, label, "calib", args.episodes, args.cells)
                for label in labels]  # fmt: skip
        return {j.condition: s for j, s in zip(jobs, run_jobs(jobs, args.workers), strict=True)}

    cap = lambda s: s["mean_capture_rate"]  # noqa: E731
    first = cells(["iid-p0", "iid-p1", *[f"iid-p0-random-r{r:g}" for r in RANDOM_RATES],
                   *HARD_CANDIDATES])  # fmt: skip
    full, none = cap(first["iid-p0"]), cap(first["iid-p1"])
    random_caps = {r: cap(first[f"iid-p0-random-r{r:g}"]) for r in RANDOM_RATES}
    bstar = pick_bstar(full, random_caps)
    hard = pick_hard(full, none, {label: cap(first[label]) for label in HARD_CANDIDATES})
    periodic = f"iid-p0-periodic-r{bstar:g}"
    cells([periodic])

    # tau: send rate falls as tau grows; grid, then bisection on the bracketing pair.
    tried: dict[float, dict] = {}

    def trigger(tau: float) -> dict:
        tau = round(tau, 2)
        if tau not in tried:
            tried[tau] = cells([f"iid-p0-trigger-t{tau:g}"])[f"iid-p0-trigger-t{tau:g}"]
        return tried[tau]

    for tau in TAU_GRID:
        trigger(tau)
    lo = max([t for t in tried if tried[t]["send_rate"] >= bstar], default=min(tried))
    hi = min([t for t in tried if tried[t]["send_rate"] < bstar], default=max(tried))
    while len(tried) < 12 and lo < hi:
        rates = {t: tried[t]["send_rate"] for t in tried}
        if any(abs(r - bstar) <= 0.1 * bstar for r in rates.values()):
            break
        mid = round((lo + hi) / 2, 2)
        if mid in tried:
            break
        if trigger(mid)["send_rate"] >= bstar:
            lo = mid
        else:
            hi = mid
    tau = min(tried, key=lambda t: (abs(tried[t]["send_rate"] - bstar), t))

    record = {
        "checkpoint": str(args.checkpoint),
        "stream": "calib",
        "episodes": args.episodes,
        "cpu": cpu_model(),
        "commit": first["iid-p0"]["commit"],
        "torch": first["iid-p0"]["torch"],
        "full_capture": full,
        "no_message_capture": none,
        "bar": 0.95 * full,
        "random_captures": {f"{r:g}": c for r, c in random_caps.items()},
        "bstar": bstar,
        "random": f"iid-p0-random-r{bstar:g}",
        "periodic": periodic,
        "tau": tau,
        "trigger": f"iid-p0-trigger-t{tau:g}",
        "trigger_send_rates": {f"{t:g}": tried[t]["send_rate"] for t in sorted(tried)},
        "trigger_captures": {f"{t:g}": cap(tried[t]) for t in sorted(tried)},
        "hard": hard,
        "hard_captures": {label: cap(first[label]) for label in HARD_CANDIDATES},
        "midpoint": (full + none) / 2,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: record[k] for k in ("full_capture", "bstar", "tau", "hard")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
