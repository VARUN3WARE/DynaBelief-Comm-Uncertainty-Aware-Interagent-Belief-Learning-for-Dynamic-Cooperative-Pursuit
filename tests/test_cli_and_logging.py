from __future__ import annotations

import json
import subprocess
import sys

import pytest
import yaml

from dynabelief.utils.logging import JsonlWriter, create_run_dir, read_jsonl
from tests.helpers import REPO_ROOT

SCRIPTS = ["train.py", "evaluate.py", "benchmark_env.py"]


@pytest.mark.parametrize("script", SCRIPTS)
def test_cli_help(script):
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout.lower()


def write_tiny_config(tmp_path):
    path = tmp_path / "tiny.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "experiment": {
                    "name": "tiny",
                    "device": "cpu",
                    "output_dir": str(tmp_path / "runs"),
                },
                "env": {"max_cycles": 8},
                "train": {"num_envs": 2, "rollout_length": 4, "total_env_steps": 32},
            }
        )
    )
    return path


def test_train_cli_writes_reproducibility_records(tmp_path):
    from scripts import train

    assert train.main(["--config", str(write_tiny_config(tmp_path))]) == 0
    (run_dir,) = (tmp_path / "runs").iterdir()
    for name in ["config.yaml", "metadata.yaml", "episodes.jsonl", "metrics.jsonl", "summary.json"]:
        assert (run_dir / name).exists(), name
    metadata = yaml.safe_load((run_dir / "metadata.yaml").read_text())
    assert {"git", "packages", "python", "hardware"} <= set(metadata)
    assert "commit" in metadata["git"] and "dirty" in metadata["git"]
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["learning"] is False
    assert len(read_jsonl(run_dir / "episodes.jsonl")) == summary["episodes"] == 4


def test_evaluate_cli(tmp_path):
    from scripts import evaluate

    config = write_tiny_config(tmp_path)
    assert evaluate.main(["--config", str(config), "--episodes", "2", "--packet-loss", "0.3"]) == 0
    (run_dir,) = (tmp_path / "runs").iterdir()
    records = read_jsonl(run_dir / "episodes.jsonl")
    assert len(records) == 2 and all(r["packet_loss"] == 0.3 for r in records)


def test_evaluate_rejects_checkpoint_until_m2(tmp_path):
    from scripts import evaluate

    with pytest.raises(SystemExit):
        evaluate.main(["--config", str(write_tiny_config(tmp_path)), "--checkpoint", "x.pt"])


def test_jsonl_refuses_non_finite(tmp_path):
    with JsonlWriter(tmp_path / "m.jsonl") as writer:
        writer.write({"a": 1.0})
        with pytest.raises(ValueError):
            writer.write({"a": float("nan")})
    assert read_jsonl(tmp_path / "m.jsonl") == [{"a": 1.0}]


def test_run_dirs_are_never_reused(tmp_path):
    a = create_run_dir(tmp_path, "x", 0)
    b = create_run_dir(tmp_path, "x", 0)
    assert a != b and a.exists() and b.exists()


def test_metadata_records_parsed_argv(tmp_path):
    from scripts import train

    args = ["--config", str(write_tiny_config(tmp_path)), "--set", "experiment.seed=2"]
    assert train.main(args) == 0
    (run_dir,) = (tmp_path / "runs").iterdir()
    metadata = yaml.safe_load((run_dir / "metadata.yaml").read_text())
    assert metadata["argv"] == args
    assert yaml.safe_load((run_dir / "config.yaml").read_text())["experiment"]["seed"] == 2


def test_benchmark_rejects_non_positive_steps(tmp_path):
    from scripts import benchmark_env

    with pytest.raises(SystemExit):
        benchmark_env.main(["--config", str(write_tiny_config(tmp_path)), "--steps", "0"])
