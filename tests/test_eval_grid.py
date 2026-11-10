"""Evaluation grid runner and the pre-registered threshold rules (b*, H*)."""

from __future__ import annotations

import json

import pytest

from dynabelief.evaluation.grid import Job, pick_bstar, pick_hard, plan_conditions, run_jobs
from dynabelief.training.trainer import MAPPOTrainer
from tests.test_posterior_target import small

THRESHOLDS = {
    "random": "iid-p0-random-r0.1",
    "periodic": "iid-p0-periodic-r0.1",
    "trigger": "iid-p0-trigger-t3.16",
    "hard": "ge-p0.8-b50",
}


def test_plan_resolves_budgets():
    assert plan_conditions("core") == ["iid-p0", "iid-p0.2", "iid-p0.5", "iid-p0.8", "iid-p1"]
    channel = plan_conditions("channel", THRESHOLDS)
    for label in ("ge-p0.8-b20", "iid-p0.2-n6", "iid-p0.2-n10", *THRESHOLDS.values()):
        assert label in channel
    assert len(channel) == len(set(channel))
    assert not any("trigger" in c for c in plan_conditions("channel"))  # none without thresholds
    assert plan_conditions("iid-p0.2,$trigger", THRESHOLDS) == ["iid-p0.2", "iid-p0-trigger-t3.16"]


def test_pick_bstar_first_rate_below_bar():
    caps = {0.5: 0.970, 0.2: 0.961, 0.1: 0.924, 0.05: 0.788, 0.02: 0.658, 0.01: 0.615}
    assert pick_bstar(0.981, caps) == 0.1  # bar 0.932
    assert pick_bstar(0.5, caps) == 0.01  # never below the bar -> 1 %


def test_pick_hard_closest_to_midpoint():
    caps = {"iid-p0.5": 0.95, "ge-p0.8-b50": 0.758, "ge-p0.8-b20": 0.736}
    assert pick_hard(0.981, 0.537, caps) == "ge-p0.8-b50"


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    run = tmp_path_factory.mktemp("train")
    MAPPOTrainer(small("point_diabl"), "cpu", run_dir=run).train()
    return run / "checkpoints" / "final.pt"


def test_grid_writes_summary(tiny_checkpoint, tmp_path):
    jobs = [Job("point_diabl", tiny_checkpoint, label, "calib", 2, tmp_path)
            for label in ("iid-p0.2", "iid-p0-trigger-t3", "iid-p0.2-n6")]  # fmt: skip
    summaries = run_jobs(jobs, workers=2)
    assert len(summaries) == 3
    for job in jobs:
        path = tmp_path / "point_diabl" / "s0" / job.condition / "summary.json"
        s = json.loads(path.read_text())
        assert s["arm"] == "point_diabl" and s["seed"] == 0 and s["episodes"] == 2
        assert s["condition"] == job.condition and s["stream"] == "calib"
        assert 0.0 <= s["mean_capture_rate"] <= 1.0
        assert {"commit", "cpu", "torch"} <= set(s)
        assert (path.parent / "episodes.jsonl").exists()
    trig = json.loads((tmp_path / "point_diabl/s0/iid-p0-trigger-t3/summary.json").read_text())
    assert 0.0 < trig["send_rate"] <= 1.0
    # finished cells are not recomputed
    again = run_jobs(jobs, workers=1)
    assert [a["mean_capture_rate"] for a in again] == [s["mean_capture_rate"] for s in summaries]
