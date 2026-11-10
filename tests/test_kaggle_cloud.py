"""Kaggle launcher/entry helpers (pure logic plus one local dry run; no Kaggle API calls)."""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from tests.test_cloud import load

launch = load("kaggle_launch")
entry = load("kaggle_entry")


@pytest.mark.parametrize("label", ["point_diabl-s1", "d_diabl_coef10-s3", "X" * 80, "tests"])
def test_job_name_fits_kaggle_title_and_is_a_slug(label):
    name = launch.job_name(label, datetime(2026, 11, 3, 9, 5))
    title = launch.kernel_title(name)
    assert len(title) <= launch.MAX_TITLE
    assert title == title.lower() and all(c.isalnum() or c == "-" for c in title)
    assert name.endswith("-1103-0905")


def test_kernel_metadata_gpu_and_cpu():
    gpu = launch.kernel_metadata("someone", "point_diabl-s1-1103-0905", gpu=True)
    assert gpu["id"] == "someone/dynabelief-point_diabl-s1-1103-0905"
    assert gpu["id"].split("/")[1] == gpu["title"]
    assert gpu["is_private"] and gpu["enable_internet"] and gpu["enable_gpu"]
    assert gpu["machine_shape"] == launch.GPU_SHAPE and gpu["code_file"] == "run.py"
    cpu = launch.kernel_metadata("someone", "tests-1103-0905", gpu=False)
    assert not cpu["enable_gpu"] and "machine_shape" not in cpu


def test_run_py_is_valid_python_pinned_to_the_commit():
    args = ["--mode", "train", "--config", "configs/point_diabl.yaml", "--set", "experiment.seed=1"]
    src = launch.run_py("job", "https://github.com/o/r.git", "abc123", args)
    compile(src, "run.py", "exec")
    assert "COMMIT = 'abc123'" in src and repr(args) in src
    assert launch.ENTRY in src


def test_https_url_for_anonymous_clone():
    assert launch.https_url("git@github.com:o/r.git") == "https://github.com/o/r.git"
    assert launch.https_url("https://github.com/o/r.git") == "https://github.com/o/r.git"


def test_bench_summary_skips_warmup_and_projects_5m_steps():
    rows = [{"collect_fps": 100.0, "update_seconds": 50.0}]  # warm-up, ignored
    rows += [{"collect_fps": 4096 / 6.0, "update_seconds": 4.0} for _ in range(5)]
    summary = entry.bench_summary(rows)
    assert summary["collect_seconds_per_update"] == pytest.approx(6.0)
    assert summary["seconds_per_update"] == pytest.approx(10.0)
    assert summary["projected_hours_5M"] == pytest.approx(1221 * 10 / 3600, abs=0.01)
    assert "error" in entry.bench_summary([{"env_steps": 1}])


def test_train_cmd_passes_overrides():
    cmd = entry.train_cmd("configs/x.yaml", entry.Path("/out"), ["a.b=1", "c.d=2"])
    assert cmd[-4:] == ["--set", "a.b=1", "--set", "c.d=2"]


def test_entry_train_dry_run_writes_status_and_outputs(tmp_path, monkeypatch):
    """The whole train path locally: tiny smoke run, plots, status.json, environment.json."""
    monkeypatch.setattr(entry, "WORKING", tmp_path)
    argv = ["--mode", "train", "--config", "configs/smoke_mappo.yaml", "--skip_install",
            "--workers", "0", "--set", "train.total_env_steps=256"]  # fmt: skip
    assert entry.main(argv) == 0
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["ok"] and status["complete"] and status["returncode"] == 0
    env = json.loads((tmp_path / "environment.json").read_text())
    assert env["torch"] and env["commit"]
    run_dir = tmp_path / status["run_dir"]
    for name in ("config.yaml", "metrics.jsonl", "summary.json", "curves.png"):
        assert (run_dir / name).exists(), name
    assert (run_dir / "checkpoints" / "final.pt").exists()


def test_train_command_keeps_snapshots_and_seed(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(launch, "launch", lambda label, entry, **kw: captured.update(
        label=label, entry=entry, **kw) or 0)  # fmt: skip
    args = launch.build_parser().parse_args(
        ["train", "--config", "configs/point_diabl.yaml", "--seed", "2", "--set", "a.b=1"]
    )
    assert launch.cmd_train(args) == 0
    sets = [captured["entry"][i + 1] for i, x in enumerate(captured["entry"]) if x == "--set"]
    assert sets == ["experiment.seed=2", "train.snapshot_every_updates=100", "a.b=1"]
    assert captured["label"] == "point_diabl-s2" and captured["gpu"]


def test_kaggle_config_dir_is_project_local(tmp_path, monkeypatch):
    """A repo-local .kaggle/ (this project's own account) wins over the user's ~/.kaggle."""
    monkeypatch.delenv("KAGGLE_CONFIG_DIR", raising=False)
    assert "KAGGLE_CONFIG_DIR" not in launch.kaggle_env(tmp_path)  # no local config: untouched
    (tmp_path / ".kaggle").mkdir()
    (tmp_path / ".kaggle" / "kaggle.json").write_text('{"username": "u", "key": "k"}')
    assert launch.kaggle_env(tmp_path)["KAGGLE_CONFIG_DIR"] == str(tmp_path / ".kaggle")


def test_kaggle_cli_runs_with_project_config(tmp_path, monkeypatch):
    (tmp_path / ".kaggle").mkdir()
    (tmp_path / ".kaggle" / "kaggle.json").write_text("{}")
    monkeypatch.setattr(launch, "REPO", tmp_path)
    seen = {}

    def fake_run(cmd, **kw):
        seen.update(kw)
        return launch.subprocess.CompletedProcess(cmd, 0, "username: someone\n", "")

    monkeypatch.setattr(launch.subprocess, "run", fake_run)
    assert launch.kaggle_user() == "someone"
    assert seen["env"]["KAGGLE_CONFIG_DIR"] == str(tmp_path / ".kaggle")
