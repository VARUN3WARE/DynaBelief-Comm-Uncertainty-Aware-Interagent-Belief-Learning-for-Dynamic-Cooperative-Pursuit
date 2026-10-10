"""M3 gate tests: TarMAC through the trainer, checkpoints and evaluation."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch.distributions import Categorical

from dynabelief.config import config_from_dict
from dynabelief.training.trainer import MAPPOTrainer
from dynabelief.utils.checkpointing import CheckpointError, load_checkpoint
from dynabelief.utils.logging import read_jsonl
from tests.test_trainer import tiny


def tarmac(packet_loss: float = 0.0, **train):
    raw = tiny(**train).to_dict()
    raw["comm"].update(enabled=True, key_dim=4, message_dim=4, packet_loss=packet_loss)
    return config_from_dict(raw)


def collect_once(config):
    trainer = MAPPOTrainer(config, "cpu")
    cfg = config.train
    actor, priv = trainer.vec.reset()
    hidden = trainer.policy.actor.initial_state(cfg.num_envs, trainer.vec.n_agents)
    start = torch.ones(cfg.num_envs, dtype=torch.bool)
    trainer._collect(actor, priv, hidden, start, None)
    trainer.vec.close()
    return trainer


def test_delivery_masks_follow_channel_and_episode_starts():
    trainer = collect_once(tarmac(packet_loss=0.0, rollout_length=16))
    buf = trainer.buffer
    n = trainer.vec.n_agents
    off_diag = ~torch.eye(n, dtype=torch.bool)
    assert not buf.delivery[buf.episode_start].any()  # nothing readable at a start
    mid = ~buf.episode_start
    assert (buf.delivery[mid] == off_diag).all()  # loss 0: every other agent delivered
    assert not buf.delivery.diagonal(dim1=-2, dim2=-1).any()
    metrics = trainer._comm_metrics
    expected_sent = int(mid.sum()) * n * (n - 1)
    assert metrics["comm_messages_sent"] == expected_sent == metrics["comm_messages_delivered"]
    assert (
        math.isfinite(metrics["comm_attention_entropy"]) and metrics["comm_attention_entropy"] > 0
    )
    assert metrics["comm_bytes_sent"] == expected_sent * 8 * 4  # (key 4 + value 4) float32


def test_packet_loss_drops_the_configured_fraction():
    trainer = collect_once(tarmac(packet_loss=0.5, rollout_length=16))
    rate = trainer._comm_metrics["comm_drop_rate"]
    assert 0.4 < rate < 0.6


def test_recomputed_log_probs_match_with_communication():
    """Ratio == 1 before any update: stored delivery masks and hidden states are aligned."""
    trainer = collect_once(tarmac(packet_loss=0.3, rollout_length=16))
    buf, policy = trainer.buffer, trainer.policy
    for mb in buf.minibatches(2, 4, torch.Generator().manual_seed(0), buf.advantages):
        logits = policy.actor.unroll(mb.obs, mb.actor_hidden0, mb.episode_start, mb.delivery)
        new_lp = Categorical(logits=logits).log_prob(mb.actions)
        torch.testing.assert_close(new_lp, mb.old_log_probs, rtol=1e-5, atol=1e-5)
    # The same check must FAIL if the messages are ignored: communication is really used.
    mb = next(iter(buf.minibatches(1, 16, torch.Generator().manual_seed(0), buf.advantages)))
    wrong = policy.actor.unroll(mb.obs, mb.actor_hidden0, mb.episode_start,
                                torch.zeros_like(mb.delivery))  # fmt: skip
    assert not torch.allclose(Categorical(logits=wrong).log_prob(mb.actions), mb.old_log_probs)


def test_tarmac_training_runs_and_logs_comm_metrics(tmp_path):
    trainer = MAPPOTrainer(tarmac(packet_loss=0.2), "cpu", run_dir=tmp_path)
    summary = trainer.train()
    assert summary["updates"] == 2
    metrics = read_jsonl(tmp_path / "metrics.jsonl")
    for m in metrics:
        for key in ("policy_loss", "value_loss", "comm_drop_rate", "comm_attention_entropy",
                    "comm_messages_sent_per_step", "actor_grad_norm"):  # fmt: skip
            assert math.isfinite(m[key]), key
    comm_params = [p for name, p in trainer.policy.actor.named_parameters() if "comm." in name]
    assert comm_params and all(torch.isfinite(p).all() for p in comm_params)


def test_resume_restores_the_channel_stream(tmp_path):
    MAPPOTrainer(tarmac(packet_loss=0.3), "cpu", run_dir=tmp_path / "a").train()
    final = tmp_path / "a" / "checkpoints" / "final.pt"
    saved = load_checkpoint(final)["comm_rng"]
    resumed = MAPPOTrainer(tarmac(packet_loss=0.3, total_env_steps=64), "cpu", resume_from=final)
    assert resumed.channel.rng.bit_generator.state == saved
    resumed.vec.close()


def test_eval_feeds_delivered_messages_and_checks_comm_architecture(tmp_path):
    from dynabelief.evaluation.evaluate import evaluate_checkpoint

    config = tarmac()
    MAPPOTrainer(config, "cpu", run_dir=tmp_path).train()
    final = tmp_path / "checkpoints" / "final.pt"
    full = evaluate_checkpoint(config, final, episodes=1, device="cpu", run_dir=tmp_path / "e0")
    lost = config_from_dict({**config.to_dict(), "comm": {**config.to_dict()["comm"],
                                                          "packet_loss": 1.0}})  # fmt: skip
    evaluate_checkpoint(lost, final, episodes=1, device="cpu", run_dir=tmp_path / "e1")
    r0 = read_jsonl(tmp_path / "e0" / "episodes.jsonl")[0]
    r1 = read_jsonl(tmp_path / "e1" / "episodes.jsonl")[0]
    n = config.env.n_pursuers
    assert r0["messages_sent"] == (r0["length"] - 1) * n * (n - 1)  # none at the first step
    assert r0["messages_delivered"] == r0["messages_sent"] and r1["messages_delivered"] == 0
    assert full["episodes"] == 1

    no_comm = config_from_dict({**config.to_dict(), "comm": {**config.to_dict()["comm"],
                                                             "enabled": False}})  # fmt: skip
    with pytest.raises(CheckpointError, match="comm.enabled"):
        evaluate_checkpoint(no_comm, final, episodes=1, device="cpu")


def test_eval_agent_requires_delivery_for_communicating_policy(tmp_path):
    from dynabelief.envs.pursuit import PursuitEnv
    from dynabelief.evaluation.evaluate import MAPPOAgent, load_policy_for_eval

    config = tarmac()
    MAPPOTrainer(config, "cpu", run_dir=tmp_path).train()
    policy, _ = load_policy_for_eval(tmp_path / "checkpoints" / "final.pt", config, "cpu")
    agent = MAPPOAgent(policy, seed=0, device="cpu", deterministic=True)
    actor, _ = PursuitEnv(config.env).reset(0)
    agent.reset(0)
    with pytest.raises(ValueError, match="delivery"):
        agent.act(actor)
    assert agent.act(actor, np.zeros((8, 8), dtype=bool)).shape == (8,)
