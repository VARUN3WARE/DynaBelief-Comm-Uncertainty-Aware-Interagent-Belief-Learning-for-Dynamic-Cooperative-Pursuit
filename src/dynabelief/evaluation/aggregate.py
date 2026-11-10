"""Hypothesis tests on evaluation cells (``configs/analysis/hypotheses.yaml``).

For arms (or conditions) X and Y and training seed s, ``D_s`` is the mean over paired
episodes (same ``episode_seed``) of the per-episode capture-rate difference X - Y. The
estimate is the mean of ``D_s`` over seeds; its 95 % CI comes from a seed-stratified
paired bootstrap (episodes resampled within each seed, 10,000 resamples, fixed RNG).

Verdicts: ``confirmed`` (threshold met, CI excludes the null, sign agreement in >= 2/3 of
the seeds or in all seeds where ``all_seeds_positive``), ``not detected``, ``incomplete``
(fewer than 2 seeds), ``not run`` (an arm has no evaluations at all), ``not comparable``
(budget conditions whose send rates differ by more than ``rate_match``).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

Key = tuple[str, int, str]  # (arm, training seed, condition label)


class UnpairedError(ValueError):
    """Two cells meant to be compared do not share their episode seeds."""


def load_cells(root: Path) -> dict[Key, dict[str, Any]]:
    cells: dict[Key, dict[str, Any]] = {}
    for summary_path in sorted(Path(root).glob("*/s*/*/summary.json")):
        summary = json.loads(summary_path.read_text())
        episodes_path = summary_path.parent / "episodes.jsonl"
        rows = [json.loads(x) for x in episodes_path.read_text().splitlines() if x.strip()]
        key = (summary["arm"], int(summary["seed"]), summary["condition"])
        cells[key] = {"summary": summary, "episodes": rows}
    return cells


def bootstrap(
    per_seed: Sequence[Callable[[np.ndarray], Any]],
    sizes: Sequence[int],
    combine: Callable[[list], float] = lambda values: float(np.mean(values)),
    n: int = 10_000,
    rng_seed: int = 0,
) -> tuple[float, float, float]:
    """Estimate and 95 % percentile CI; episodes are resampled within each seed."""
    estimate = combine([f(np.arange(size)) for f, size in zip(per_seed, sizes, strict=True)])
    rng = np.random.default_rng(rng_seed)
    draws = np.empty(n)
    for b in range(n):
        draws[b] = combine([f(rng.integers(0, size, size))
                            for f, size in zip(per_seed, sizes, strict=True)])  # fmt: skip
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(estimate), float(lo), float(hi)


def _paired(cells, x: Key, y: Key) -> tuple[np.ndarray, np.ndarray]:
    xs = {r["episode_seed"]: r["capture_rate"] for r in cells[x]["episodes"]}
    ys = {r["episode_seed"]: r["capture_rate"] for r in cells[y]["episodes"]}
    if set(xs) != set(ys):
        raise UnpairedError(f"{x} and {y} were evaluated on different episodes")
    order = sorted(xs)
    return np.array([xs[k] for k in order]), np.array([ys[k] for k in order])


def _resolve(label: str, thresholds: dict[str, Any] | None) -> str | None:
    if not label.startswith("$"):
        return label
    return None if thresholds is None else thresholds.get(label[1:])


def evaluate_hypothesis(
    h: dict[str, Any], cells: dict[Key, dict[str, Any]], thresholds: dict[str, Any] | None = None
) -> dict[str, Any]:
    kind = h["kind"]
    out: dict[str, Any] = {"id": h["id"], "kind": kind, "threshold": h.get("threshold"),
                           "secondary": bool(h.get("secondary")), "seeds": [], "per_seed": {},
                           "estimate": None, "ci": None, "note": ""}  # fmt: skip
    if "arm" in h:
        xa = ya = h["arm"]
        xc, yc = _resolve(h["x_condition"], thresholds), _resolve(h["y_condition"], thresholds)
    else:
        xa, ya = h["x"], h.get("y", h["x"])
        conditions = h.get("conditions", [h.get("condition")])
        xc = yc = _resolve(conditions[0], thresholds)
    if xc is None or yc is None:
        out.update(verdict="not run", note="condition needs the thresholds record")
        return out
    out.update(x=f"{xa}@{xc}", y=f"{ya}@{yc}" if kind != "metric" else None)
    arms_present = {k[0] for k in cells}
    if xa not in arms_present or (kind != "metric" and ya not in arms_present):
        out.update(verdict="not run", note="an arm has no evaluations")
        return out

    seeds = sorted({s for (a, s, c) in cells if a == xa and c == xc}
                   & ({s for (a, s, c) in cells if a == ya and c == yc}
                      if kind != "metric" else {s for (a, s, c) in cells}))  # fmt: skip
    out["seeds"] = seeds
    if not seeds:
        out.update(verdict="incomplete", note="no seed has both cells")
        return out

    if kind == "metric":
        if h["metric"] != "remote_bce_explained":
            raise ValueError(f"unknown metric {h['metric']!r}")
        fns, sizes, values = [], [], {}
        for s in seeds:
            rows = cells[(xa, s, xc)]["episodes"]
            model = np.array([r["map_bce_remote_model"] for r in rows])
            prior = np.array([r["map_bce_remote_prior"] for r in rows])
            fns.append(lambda idx, m=model, p=prior: 1.0 - m[idx].mean() / p[idx].mean())
            sizes.append(len(rows))
            values[s] = fns[-1](np.arange(len(rows)))
        est, lo, hi = bootstrap(fns, sizes)
        positive = [values[s] >= h["threshold"] for s in seeds]
        out.update(estimate=est, ci=[lo, hi], per_seed=values)
        null = 0.0
    else:
        if h.get("rate_match") is not None:
            for s in seeds:
                rx = cells[(xa, s, xc)]["summary"].get("send_rate")
                ry = cells[(ya, s, yc)]["summary"].get("send_rate")
                if rx is None or ry is None or abs(rx - ry) > h["rate_match"] * ry:
                    out.update(verdict="not comparable", note=f"seed {s}: send rates {rx} vs {ry}")
                    return out
        pairs = [_paired(cells, (xa, s, xc), (ya, s, yc)) for s in seeds]
        sizes = [len(x) for x, _ in pairs]
        diffs = {s: float((x - y).mean()) for s, (x, y) in zip(seeds, pairs, strict=True)}
        if kind == "ratio":
            fns = [lambda idx, x=x, y=y: np.array([x[idx].mean(), y[idx].mean()]) for x, y in pairs]
            est, lo, hi = bootstrap(fns, sizes, combine=lambda v: float(
                np.sum([a[0] for a in v]) / max(np.sum([a[1] for a in v]), 1e-12)))  # fmt: skip
            null = 1.0
        else:
            fns = [lambda idx, d=x - y: d[idx].mean() for x, y in pairs]
            est, lo, hi = bootstrap(fns, sizes)
            null = 0.0
        positive = [diffs[s] > 0 for s in seeds]
        out.update(estimate=est, ci=[lo, hi], per_seed=diffs)

    if len(seeds) < 2:
        out.update(verdict="incomplete", note=f"{len(seeds)} seed(s); confirmation needs >= 2")
        return out
    need = len(seeds) if h.get("all_seeds_positive") else math.ceil(2 * len(seeds) / 3)
    confirmed = est >= h["threshold"] and lo > null and sum(positive) >= need
    out["verdict"] = "confirmed" if confirmed else "not detected"
    return out
