"""Compare finished runs: eval table (with paired differences), learning curves, loss sweep.

Inputs are job folders downloaded with ``cloud/launch.py download`` (they contain
``runs/<run>/metrics.jsonl`` and ``evals/<label>/<eval>/episodes.jsonl``). All methods
are evaluated on the SAME episode seeds, so per-episode differences against a reference
method are paired, which gives much tighter confidence intervals than unpaired means.

Example:
    python scripts/compare_runs.py --ref no_comm \\
        no_comm=runs/sagemaker/<job1> tarmac=runs/sagemaker/<job2> --out docs/results/m3
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # fixed order


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_run(folder: Path, eval_folder: Path | None = None) -> dict:
    """``eval_folder``: an eval-only job's folder whose evals replace the training job's."""
    metrics = sorted(folder.glob("runs/*/metrics.jsonl"))
    if not metrics:
        raise SystemExit(f"{folder}: no runs/*/metrics.jsonl")
    folder = eval_folder or folder
    evals = {}
    for episodes in sorted(folder.glob("evals/*/*/episodes.jsonl")):
        evals[episodes.parent.parent.name] = read_jsonl(episodes)
    for episodes in sorted(folder.glob("evals/*/episodes.jsonl")):  # pre-sweep job layout
        records = read_jsonl(episodes)
        evals["greedy" if records[0].get("deterministic") else "sampled"] = records
    return {"metrics": read_jsonl(metrics[-1]), "evals": evals}


def mean_ci(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if len(arr) < 2:
        return float(arr.mean()), float("nan")
    return float(arr.mean()), float(1.96 * arr.std(ddof=1) / math.sqrt(len(arr)))


def eval_label_for_loss(evals: dict, loss: float) -> str | None:
    """Sampled-action eval at a packet loss; runs without comm have one 'sampled' eval."""
    for label in (f"sampled_loss{loss:g}", "sampled"):
        if label in evals:
            return label
    return None


def belief_cell(point_errors: list, map_records: list) -> str:
    """Point beliefs: mean error in cells. Map beliefs: BCE (nats/cell) vs the prior map."""
    if map_records:
        model = np.mean([r["map_bce_model"] for r in map_records])
        prior = np.mean([r["map_bce_prior"] for r in map_records])
        return f"BCE {model:.3f} (prior {prior:.3f})"
    return f"{np.mean(point_errors):.2f} cells" if point_errors else "–"


def remote_cell(point_remote: list, map_records: list) -> str:
    if map_records:
        model = np.mean([r["map_bce_remote_model"] for r in map_records])
        prior = np.mean([r["map_bce_remote_prior"] for r in map_records])
        return f"BCE {model:.3f} (prior {prior:.3f})"
    return f"{np.mean(point_remote):.2f} cells" if point_remote else "–"


def table(runs: dict[str, dict], ref: str, losses: list[float]) -> str:
    header = (f"| Method | Packet loss | Capture rate | Δ vs {ref} (paired) | Evaders / episode "
              "| Time to capture | Belief quality | Remote-only belief quality |")  # fmt: skip
    lines = [header, "|---|---|---|---|---|---|---|---|"]
    for name, run in runs.items():
        for loss in losses:
            label = eval_label_for_loss(run["evals"], loss)
            ref_label = eval_label_for_loss(runs[ref]["evals"], loss)
            if label is None or ref_label is None:
                continue
            recs, ref_recs = run["evals"][label], runs[ref]["evals"][ref_label]
            if label == "sampled" and loss != losses[0]:
                continue  # no comm: identical at every loss, show once
            cap = mean_ci([r["capture_rate"] for r in recs])
            seeds = [r["seed"] for r in recs]
            paired = "–"
            if name != ref and seeds == [r["seed"] for r in ref_recs]:
                d = mean_ci([a["capture_rate"] - b["capture_rate"]
                             for a, b in zip(recs, ref_recs, strict=True)])  # fmt: skip
                paired = f"{d[0]:+.3f} ± {d[1]:.3f}"
            ttc = [r["mean_time_to_capture"] for r in recs if r["mean_time_to_capture"]]
            belief = [r["belief_error_cells"] for r in recs if "belief_error_cells" in r]
            maps_ = [r for r in recs if "map_bce_model" in r]
            remote = [r["belief_remote_error_cells"] for r in recs
                      if "belief_remote_error_cells" in r]  # fmt: skip
            loss_txt = "– (no comm)" if label == "sampled" else f"{loss:g}"
            lines.append(
                f"| {name} | {loss_txt} | {cap[0]:.3f} ± {cap[1]:.3f} | {paired} | "
                f"{np.mean([r['captures'] for r in recs]):.1f} | "
                f"{(f'{np.mean(ttc):.0f}' if ttc else '–')} | "
                f"{belief_cell(belief, maps_)} | {remote_cell(remote, maps_)} |"
            )
    return "\n".join(lines)


def style(ax, title: str, xlabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=9.5, color=TEXT)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color=TEXT_2)
    ax.tick_params(colors=TEXT_2)


def plot(runs: dict[str, dict], losses: list[float], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), facecolor=SURFACE)
    curve, sweep = axes
    style(curve, "Capture rate during training (last 100 episodes)", "env steps (millions)")
    style(sweep, "Held-out capture rate vs packet loss (sampled actions)", "packet loss")
    for color, (name, run) in zip(SERIES, runs.items(), strict=False):
        pts = [(m["env_steps"] / 1e6, m["window_mean_capture_rate"]) for m in run["metrics"]
               if "window_mean_capture_rate" in m]  # fmt: skip
        xs, ys = zip(*pts, strict=True)
        curve.plot(xs, ys, color=color, linewidth=2, label=name)
        if "sampled" in run["evals"]:  # no communication: one eval, independent of loss
            cap = mean_ci([r["capture_rate"] for r in run["evals"]["sampled"]])[0]
            sweep.axhline(cap, color=color, linewidth=2, linestyle=(0, (4, 3)),
                          label=f"{name} (no messages)")  # fmt: skip
            continue
        values = []
        for loss in losses:
            label = eval_label_for_loss(run["evals"], loss)
            if label:
                values.append((loss, mean_ci([r["capture_rate"] for r in run["evals"][label]])))
        if values:
            lx = [v[0] for v in values]
            ly = [v[1][0] for v in values]
            err = [v[1][1] for v in values]
            sweep.errorbar(lx, ly, yerr=err, color=color, linewidth=2, marker="o", markersize=5,
                           capsize=3, label=name)  # fmt: skip
    curve.legend(frameon=False, fontsize=8, labelcolor=TEXT, loc="lower right")
    sweep.legend(frameon=False, fontsize=8, labelcolor=TEXT, loc="center left")
    sweep.set_xticks(losses)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("runs", nargs="+", metavar="NAME=TRAIN_FOLDER[:EVAL_FOLDER]")
    parser.add_argument("--ref", required=True, help="reference method for paired differences")
    parser.add_argument("--losses", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.5, 1.0])
    parser.add_argument("--out", type=Path, required=True, help="output prefix (no extension)")
    args = parser.parse_args(argv)
    runs = {}
    for item in args.runs:
        name, _, spec = item.partition("=")
        train, _, evals = spec.partition(":")
        runs[name] = load_run(Path(train), Path(evals) if evals else None)
    if args.ref not in runs:
        raise SystemExit(f"--ref {args.ref} is not one of {list(runs)}")
    text = table(runs, args.ref, args.losses)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".md").write_text(text + "\n", encoding="utf-8")
    plot(runs, args.losses, args.out.with_suffix(".png"))
    print(text)
    print(f"wrote {args.out.with_suffix('.md')} and {args.out.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
