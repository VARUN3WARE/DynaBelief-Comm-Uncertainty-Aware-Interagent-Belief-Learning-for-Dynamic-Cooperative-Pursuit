"""Aggregation: paired per-seed differences, seed-stratified bootstrap, verdicts."""

from __future__ import annotations

import json

import numpy as np
import pytest

from dynabelief.evaluation.aggregate import (
    UnpairedError,
    bootstrap,
    evaluate_hypothesis,
    load_cells,
)


def write_cell(root, arm, seed, cond, captures, seeds=None, **extra):
    cell = root / arm / f"s{seed}" / cond
    cell.mkdir(parents=True)
    seeds = seeds if seeds is not None else list(range(len(captures)))
    rows = [{"capture_rate": c, "episode_seed": s, **{k: v[i] for k, v in extra.items()}}
            for i, (c, s) in enumerate(zip(captures, seeds, strict=True))]  # fmt: skip
    (cell / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    summary = {"arm": arm, "seed": seed, "condition": cond, "stream": "test",
               "mean_capture_rate": float(np.mean(captures))}  # fmt: skip
    if "send_rate" in extra:
        summary["send_rate"] = float(np.mean(extra["send_rate"]))
    (cell / "summary.json").write_text(json.dumps(summary))


def test_bootstrap_ci_covers_known_mean():
    rng = np.random.default_rng(1)
    diffs = [rng.normal(0.1, 0.2, 200) for _ in range(3)]
    est, lo, hi = bootstrap([lambda idx, d=d: d[idx].mean() for d in diffs],
                            [len(d) for d in diffs], n=2000)  # fmt: skip
    assert lo < 0.1 < hi and lo < est < hi and hi - lo < 0.1


def test_diff_hypothesis_confirmed_and_not_detected(tmp_path):
    rng = np.random.default_rng(0)
    for seed in (1, 2, 3):
        y = rng.uniform(0.4, 0.6, 100)
        write_cell(tmp_path, "x_arm", seed, "iid-p0", np.clip(y + 0.2, 0, 1))
        write_cell(tmp_path, "y_arm", seed, "iid-p0", y)
        write_cell(tmp_path, "z_arm", seed, "iid-p0", y + rng.normal(0, 0.01, 100))
    cells = load_cells(tmp_path)
    h = {"id": "H", "x": "x_arm", "y": "y_arm", "condition": "iid-p0", "kind": "diff",
         "threshold": 0.1, "all_seeds_positive": True}  # fmt: skip
    r = evaluate_hypothesis(h, cells)
    assert r["verdict"] == "confirmed" and r["seeds"] == [1, 2, 3]
    assert r["estimate"] == pytest.approx(0.2, abs=0.01) and r["ci"][0] > 0
    r = evaluate_hypothesis({**h, "x": "z_arm"}, cells)
    assert r["verdict"] == "not detected"


def test_ratio_hypothesis(tmp_path):
    for seed in (0, 1):
        write_cell(tmp_path, "A", seed, "iid-p0.2", [0.95] * 50 + [1.0] * 50)
        write_cell(tmp_path, "ctrl", seed, "iid-p0.2", [0.5] * 50 + [0.6] * 50)
    h = {"id": "R1", "x": "A", "y": "ctrl", "condition": "iid-p0.2", "kind": "ratio",
         "threshold": 1.10, "all_seeds_positive": True}  # fmt: skip
    r = evaluate_hypothesis(h, load_cells(tmp_path))
    assert r["verdict"] == "confirmed" and r["estimate"] == pytest.approx(0.975 / 0.55)


def test_within_arm_conditions_and_thresholds(tmp_path):
    for seed in (0, 1, 2):
        write_cell(tmp_path, "A", seed, "iid-p0-trigger-t3", [0.8] * 100, send_rate=[0.1] * 100)
        write_cell(tmp_path, "A", seed, "iid-p0-random-r0.1", [0.9] * 100, send_rate=[0.1] * 100)
    h = {"id": "C2a", "arm": "A", "x_condition": "$trigger", "y_condition": "$random",
         "kind": "diff", "threshold": 0.03, "rate_match": 0.1}  # fmt: skip
    th = {"trigger": "iid-p0-trigger-t3", "random": "iid-p0-random-r0.1"}
    r = evaluate_hypothesis(h, load_cells(tmp_path), th)
    assert r["verdict"] == "not detected" and r["estimate"] == pytest.approx(-0.1)


def test_rate_mismatch_is_not_comparable(tmp_path):
    for seed in (0, 1):
        write_cell(tmp_path, "A", seed, "iid-p0-trigger-t3", [0.9] * 10, send_rate=[0.3] * 10)
        write_cell(tmp_path, "A", seed, "iid-p0-random-r0.1", [0.8] * 10, send_rate=[0.1] * 10)
    h = {"id": "C2a", "arm": "A", "x_condition": "iid-p0-trigger-t3",
         "y_condition": "iid-p0-random-r0.1", "kind": "diff", "threshold": 0.03,
         "rate_match": 0.1}  # fmt: skip
    assert evaluate_hypothesis(h, load_cells(tmp_path))["verdict"] == "not comparable"


def test_unpaired_cells_error(tmp_path):
    write_cell(tmp_path, "x", 1, "iid-p0", [0.5, 0.6], seeds=[10, 11])
    write_cell(tmp_path, "y", 1, "iid-p0", [0.5, 0.6], seeds=[10, 12])
    write_cell(tmp_path, "x", 2, "iid-p0", [0.5, 0.6])
    write_cell(tmp_path, "y", 2, "iid-p0", [0.5, 0.6])
    h = {"id": "H", "x": "x", "y": "y", "condition": "iid-p0", "kind": "diff", "threshold": 0}
    with pytest.raises(UnpairedError):
        evaluate_hypothesis(h, load_cells(tmp_path))


def test_incomplete_when_seed_missing(tmp_path):
    write_cell(tmp_path, "x", 1, "iid-p0", [0.9] * 20)
    write_cell(tmp_path, "y", 1, "iid-p0", [0.1] * 20)
    write_cell(tmp_path, "x", 2, "iid-p0", [0.9] * 20)  # y has no seed 2
    h = {"id": "H", "x": "x", "y": "y", "condition": "iid-p0", "kind": "diff", "threshold": 0}
    r = evaluate_hypothesis(h, load_cells(tmp_path))
    assert r["verdict"] == "incomplete" and r["seeds"] == [1]
    assert r["estimate"] == pytest.approx(0.8)
    r = evaluate_hypothesis({**h, "x": "missing_arm"}, load_cells(tmp_path))
    assert r["verdict"] == "not run" and r["seeds"] == []


def test_remote_bce_metric(tmp_path):
    for seed in (0, 1):
        write_cell(tmp_path, "C", seed, "iid-p0", [0.6] * 10,
                   map_bce_remote_model=[0.08] * 10, map_bce_remote_prior=[0.10] * 10)  # fmt: skip
    h = {"id": "C1a", "x": "C", "condition": "iid-p0", "kind": "metric",
         "metric": "remote_bce_explained", "threshold": 0.15,
         "all_seeds_positive": True}  # fmt: skip
    r = evaluate_hypothesis(h, load_cells(tmp_path))
    assert r["estimate"] == pytest.approx(0.2) and r["verdict"] == "confirmed"
