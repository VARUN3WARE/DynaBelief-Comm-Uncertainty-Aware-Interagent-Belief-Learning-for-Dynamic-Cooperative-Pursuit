"""Plot training curves from a run's ``metrics.jsonl`` as small multiples (one PNG).

Every panel is generated from the saved raw records; nothing is typed in by hand.

Example:
    python scripts/plot_metrics.py runs/<run> --reference window_mean_capture_rate=0.387
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dynabelief.utils.logging import read_jsonl  # noqa: E402

# Reference palette (dataviz skill): surface, text, one series hue, recessive grid.
SURFACE, TEXT, TEXT_2, GRID, SERIES = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"

PANELS = [
    ("window_mean_capture_rate", "Capture rate (last 100 episodes)"),
    ("window_mean_length", "Episode length (steps)"),
    ("window_mean_team_return", "Team return per episode"),
    ("entropy", "Policy entropy (nats)"),
    ("explained_variance", "Critic explained variance"),
    ("approx_kl", "Approx. KL per update"),
]


def parse_reference(items: list[str]) -> dict[str, float]:
    refs = {}
    for item in items:
        key, _, value = item.partition("=")
        if not value:
            raise SystemExit(f"--reference expects KEY=VALUE, got {item!r}")
        refs[key] = float(value)
    return refs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot learning curves from metrics.jsonl.")
    parser.add_argument("run_dir", type=Path, help="run directory containing metrics.jsonl")
    parser.add_argument("--output", type=Path, help="PNG path (default: <run_dir>/curves.png)")
    parser.add_argument(
        "--reference",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="dashed reference line, e.g. a random-policy baseline",
    )
    parser.add_argument("--reference-label", default="random policy")
    args = parser.parse_args(argv)

    records = read_jsonl(args.run_dir / "metrics.jsonl")
    if not records:
        raise SystemExit(f"no records in {args.run_dir / 'metrics.jsonl'}")
    refs = parse_reference(args.reference)
    steps = [r["env_steps"] / 1e3 for r in records]

    plt.rcParams.update({"font.size": 9, "text.color": TEXT, "axes.labelcolor": TEXT_2,
                         "xtick.color": TEXT_2, "ytick.color": TEXT_2})  # fmt: skip
    cols = 3
    rows = math.ceil(len(PANELS) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 2.8 * rows), facecolor=SURFACE)
    for ax, (key, title) in zip(axes.flat, PANELS, strict=False):
        pts = [(s, r[key]) for s, r in zip(steps, records, strict=True) if key in r]
        ax.set_facecolor(SURFACE)
        ax.set_title(title, loc="left", fontsize=9.5, color=TEXT)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("env steps (thousands)")
        if pts:
            xs, ys = zip(*pts, strict=True)
            ax.plot(xs, ys, color=SERIES, linewidth=2, solid_capstyle="round")
            ax.annotate(f"{ys[-1]:.3g}", (xs[-1], ys[-1]), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=8,
                        color=TEXT)  # fmt: skip
        if key in refs:
            ax.axhline(refs[key], color=TEXT_2, linewidth=1.2, linestyle=(0, (4, 3)))
            ax.annotate(args.reference_label, (0.99, refs[key]), xycoords=("axes fraction", "data"),
                        xytext=(0, 3), textcoords="offset points", ha="right", fontsize=8,
                        color=TEXT_2)  # fmt: skip
    for ax in list(axes.flat)[len(PANELS) :]:
        ax.set_visible(False)
    fig.suptitle(args.run_dir.name, x=0.01, ha="left", fontsize=10.5, color=TEXT)
    fig.tight_layout()
    output = args.output or args.run_dir / "curves.png"
    fig.savefig(output, dpi=150, facecolor=SURFACE)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
