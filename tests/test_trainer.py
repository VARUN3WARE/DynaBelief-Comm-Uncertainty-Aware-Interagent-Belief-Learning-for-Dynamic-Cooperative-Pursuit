from __future__ import annotations

import ctypes
import json
import math
import multiprocessing as mp
import signal
import sys

import numpy as np
import pytest
import torch
import yaml

from dynabelief.config import ConfigError, EnvConfig, config_from_dict
from dynabelief.envs.pursuit import PursuitEnv
from dynabelief.envs.vector import PursuitVecEnv
from dynabelief.training.trainer import MAPPOTrainer
from dynabelief.utils.checkpointing import CheckpointError, load_checkpoint
from dynabelief.utils.logging import read_jsonl

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def tiny(tmp_path=None, **train):
    raw = {
        "experiment": {"name": "t", "device": "cpu"},
        "env": {"max_cycles": 12},
        "comm": {"enabled": False},
        "train": {"policy": "mappo", "num_envs": 2, "rollout_length": 8,
                  "total_env_steps": 32, "checkpoint_every_updates": 1, **train},
        "model": {"hidden_dim": 16, "conv_channels": 4, "critic_hidden_dim": 16},
        "ppo": {"epochs": 1, "num_minibatches": 2, "chunk_length": 4},
    }  # fmt: skip
    if tmp_path is not None:
        raw["experiment"]["output_dir"] = str(tmp_path)
    return config_from_dict(raw)


# ------------------------------------------------------------ signal safety
@pytest.mark.skipif(sys.platform != "linux", reason="sigaction layout is Linux-specific")
def test_pursuit_does_not_hijack_sigterm():
    """pygame/SDL would install a C SIGTERM handler that swallows terminate()."""
    PursuitEnv(EnvConfig())

    class SigAction(ctypes.Structure):
        _fields_ = [("handler", ctypes.c_void_p), ("mask", ctypes.c_ulong * 16),
                    ("flags", ctypes.c_int), ("restorer", ctypes.c_void_p)]  # fmt: skip

    action = SigAction()
    ctypes.CDLL(None).sigaction(signal.SIGTERM, None, ctypes.byref(action))
    assert not action.handler  # SIG_DFL


def test_workers_die_on_terminate():
    vec = PursuitVecEnv(EnvConfig(max_cycles=5), num_envs=2, seed=0, num_workers=2)
    vec.reset()
    for proc in vec._processes:
        proc.terminate()
        proc.join(timeout=10)
        assert not proc.is_alive(), "worker ignored SIGTERM"
    vec._closed = True


def test_trainer_init_failure_leaves_no_workers(monkeypatch):
    import dynabelief.training.trainer as trainer_mod

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(trainer_mod, "build_policy", boom)
    with pytest.raises(RuntimeError, match="boom"):
        MAPPOTrainer(tiny(num_workers=2), "cpu")
    for child in mp.active_children():
        child.join(timeout=10)
    assert not mp.active_children()


# ----------------------------------------------------------------- training
@pytest.mark.parametrize("workers", [0, 2])
def test_training_run_writes_finite_metrics_and_checkpoints(tmp_path, workers):
    trainer = MAPPOTrainer(tiny(num_workers=workers), "cpu", run_dir=tmp_path)
    summary = trainer.train()
    assert summary["updates"] == 2 and summary["env_steps"] == 32 and summary["learning"]
    metrics = read_jsonl(tmp_path / "metrics.jsonl")
    assert [m["update"] for m in metrics] == [1, 2]
    for m in metrics:
        for key in (
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "clip_fraction",
            "actor_grad_norm",
            "critic_grad_norm",
            "collect_fps",
        ):
            assert math.isfinite(m[key]), key
    episodes = read_jsonl(tmp_path / "episodes.jsonl")
    assert len(episodes) == 2  # 16 steps per env, 12-step episodes
    assert {p.name for p in (tmp_path / "checkpoints").iterdir()} >= {"final.pt", "latest.pt"}
    ckpt = load_checkpoint(tmp_path / "checkpoints" / "final.pt")
    assert ckpt["env_steps"] == 32 and ckpt["update"] == 2
    assert ckpt["episode_index"] == [2, 2]


@pytest.mark.parametrize("device", DEVICES)
def test_resume_continues_counters_on_every_device(tmp_path, device):
    first = MAPPOTrainer(tiny(), device, run_dir=tmp_path / "a")
    first.train()
    final = tmp_path / "a" / "checkpoints" / "final.pt"
    longer = tiny(total_env_steps=64)
    resumed = MAPPOTrainer(longer, device, run_dir=tmp_path / "b", resume_from=final)
    assert resumed.update_index == 2 and resumed.env_steps == 32
    saved = load_checkpoint(final)["learner"]["policy"]
    for name, value in resumed.policy.state_dict().items():
        assert torch.equal(value.cpu(), saved[name].cpu()), name
    summary = resumed.train()
    assert summary["updates"] == 4 and summary["env_steps"] == 64
    assert load_checkpoint(tmp_path / "b" / "checkpoints" / "final.pt")["episode_index"] == [4, 4]


def test_resume_rejects_changed_experiment(tmp_path):
    MAPPOTrainer(tiny(), "cpu", run_dir=tmp_path).train()
    changed = config_from_dict(
        {**tiny().to_dict(), "ppo": {**tiny().to_dict()["ppo"], "lr_actor": 1e-3}}
    )
    with pytest.raises(CheckpointError, match="ppo differs"):
        MAPPOTrainer(changed, "cpu", resume_from=tmp_path / "checkpoints" / "final.pt")


def test_mappo_requires_comm_disabled_until_m3():
    raw = tiny().to_dict()
    raw["comm"]["enabled"] = True
    with pytest.raises(ConfigError, match="comm.enabled"):
        config_from_dict(raw)


# ---------------------------------------------------------------------- CLI
def test_train_and_evaluate_cli_with_checkpoint(tmp_path):
    from scripts import evaluate, train

    cfg_path = tmp_path / "tiny.yaml"
    cfg_path.write_text(yaml.safe_dump(tiny(tmp_path / "runs").to_dict()))
    assert train.main(["--config", str(cfg_path)]) == 0
    (run_dir,) = (tmp_path / "runs").iterdir()
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["policy"] == "mappo" and summary["learning"] is True
    final = run_dir / "checkpoints" / "final.pt"

    out = tmp_path / "eval"
    assert evaluate.main(["--checkpoint", str(final), "--episodes", "2", "--output-dir", str(out),
                          "--set", "env.n_pursuers=5"]) == 0  # fmt: skip
    (eval_dir,) = out.iterdir()
    records = read_jsonl(eval_dir / "episodes.jsonl")
    assert len(records) == 2 and all(r["policy"] == "mappo" for r in records)
    assert all(len(r["agent_returns"]) == 5 for r in records)  # generalization eval


def test_checkpoint_evaluation_is_reproducible(tmp_path):
    from dynabelief.evaluation.evaluate import evaluate_checkpoint

    config = tiny()
    MAPPOTrainer(config, "cpu", run_dir=tmp_path).train()
    final = tmp_path / "checkpoints" / "final.pt"
    a = evaluate_checkpoint(config, final, episodes=2, device="cpu")
    b = evaluate_checkpoint(config, final, episodes=2, device="cpu")
    assert a == b
    np.testing.assert_(a["episodes"] == 2)


def test_observations_are_contiguous_for_both_backends():
    for workers in (0, 2):
        vec = PursuitVecEnv(EnvConfig(max_cycles=5), num_envs=2, seed=0, num_workers=workers)
        actor, _ = vec.reset()
        assert actor.obs.flags["C_CONTIGUOUS"]
        assert vec.step(np.zeros((2, 8), dtype=np.int64)).actor.obs.flags["C_CONTIGUOUS"]
        vec.close()


def test_training_is_identical_with_and_without_workers(tmp_path):
    """Same seed -> bit-identical weights whether envs run in-process or in workers."""
    states = []
    for workers in (0, 2):
        trainer = MAPPOTrainer(tiny(num_workers=workers), "cpu", run_dir=tmp_path / str(workers))
        trainer.train()
        states.append(trainer.policy.state_dict())
    for name in states[0]:
        assert torch.equal(states[0][name], states[1][name]), name


def test_worker_dying_while_idle_raises_worker_error():
    from dynabelief.envs.vector import WorkerError

    vec = PursuitVecEnv(EnvConfig(max_cycles=5), num_envs=2, seed=0, num_workers=2)
    vec.reset()
    vec._processes[1].kill()
    vec._processes[1].join(timeout=10)
    with pytest.raises(WorkerError):
        for _ in range(3):  # the first send may still succeed into the pipe buffer
            vec.step(np.zeros((2, 8), dtype=np.int64))
    assert vec._closed
