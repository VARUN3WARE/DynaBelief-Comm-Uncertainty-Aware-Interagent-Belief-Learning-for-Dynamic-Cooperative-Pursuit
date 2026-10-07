"""SageMaker launcher/entry helpers (pure logic only; no AWS calls)."""

from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime

import pytest

from tests.helpers import REPO_ROOT


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "cloud" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launch = load("launch")
entry = load("sagemaker_entry")


@pytest.mark.parametrize(
    "num_envs, vcpus, expected",
    [(32, 32, 16), (32, 16, 8), (32, 64, 32), (16, 4, 2), (32, 1, 1), (12, 8, 6)],
)
def test_pick_workers_divides_envs_and_reserves_cpus(num_envs, vcpus, expected):
    workers = entry.pick_workers(num_envs, vcpus)
    assert workers == expected and num_envs % workers == 0 and workers <= max(1, vcpus - 2)


def test_image_uri_matches_instance_family():
    assert launch.image_uri("us-east-1", "ml.g4dn.8xlarge").endswith(
        "pytorch-training:2.5.1-gpu-py311-cu124-ubuntu22.04-sagemaker"
    )
    assert launch.image_uri("us-east-1", "ml.c6i.8xlarge").endswith(
        "pytorch-training:2.5.1-cpu-py311-ubuntu22.04-sagemaker"
    )


def test_job_name_is_valid_for_sagemaker():
    name = launch.job_name("no_comm", 3, datetime(2026, 10, 7, 12, 0, 1, tzinfo=UTC))
    assert name == "dynabelief-no-comm-s3-20261007-120001"
    long = launch.job_name("x" * 100, 0, datetime(2026, 1, 1, tzinfo=UTC))
    assert len(long) <= 63 and not long.endswith("-")


def request(**over):
    base = dict(name="dynabelief-t-s0-1", region="us-east-1", role_arn="arn:role", bucket="b",
                source_uri="s3://b/src.tar.gz", commit="abc123", config="configs/no_comm.yaml",
                overrides=["experiment.seed=1", "train.total_env_steps=100"],
                instance_type="ml.g4dn.8xlarge", spot=False, max_hours=2,
                eval_episodes=50)  # fmt: skip
    base.update(over)
    return launch.training_job_request(**base)


def test_training_job_request_shape():
    r = request()
    hyper = {k: json.loads(v) for k, v in r["HyperParameters"].items()}  # all JSON strings
    assert hyper["sagemaker_program"] == "cloud/sagemaker_entry.py"
    assert hyper["sagemaker_submit_directory"] == "s3://b/src.tar.gz"
    assert hyper["overrides"] == "experiment.seed=1;train.total_env_steps=100"
    # The toolkit passes this to logging.basicConfig(level=...): it must decode to an int.
    assert hyper["sagemaker_container_log_level"] == 20
    assert hyper["eval_episodes"] == 50
    assert r["Environment"]["DYNABELIEF_GIT_COMMIT"] == "abc123"
    assert r["CheckpointConfig"] == {"S3Uri": "s3://b/dynabelief/checkpoints/dynabelief-t-s0-1",
                                     "LocalPath": "/opt/ml/checkpoints"}  # fmt: skip
    assert r["StoppingCondition"] == {"MaxRuntimeInSeconds": 7200}
    assert {"Key": "git_commit", "Value": "abc123"} in r["Tags"]
    spot = request(spot=True)
    assert spot["EnableManagedSpotTraining"] is True
    assert spot["StoppingCondition"]["MaxWaitTimeInSeconds"] == 14400


def test_entry_finds_newest_resume_and_final(tmp_path):
    import os
    import time

    assert entry.find_resume(tmp_path) is None
    older = tmp_path / "a" / "checkpoints"
    newer = tmp_path / "b" / "checkpoints"
    for d in (older, newer):
        d.mkdir(parents=True)
    (older / "latest.pt").write_bytes(b"0")
    (newer / "latest.pt").write_bytes(b"1")
    (newer / "final.pt").write_bytes(b"1")
    past = time.time() - 100
    os.utime(older / "latest.pt", (past, past))
    assert entry.find_resume(tmp_path) == newer / "latest.pt"
    assert entry.newest_final(tmp_path) == newer / "final.pt"
    with pytest.raises(SystemExit):
        entry.newest_final(tmp_path / "missing")


def test_git_state_falls_back_to_launcher_commit(monkeypatch, tmp_path):
    from dynabelief.utils import logging as dlog

    monkeypatch.setenv(dlog.GIT_COMMIT_ENV, "deadbeef")
    state = dlog.git_state(tmp_path)  # tmp_path is not a git repo
    assert state == {"commit": "deadbeef", "dirty": False, "source": "git archive"}
