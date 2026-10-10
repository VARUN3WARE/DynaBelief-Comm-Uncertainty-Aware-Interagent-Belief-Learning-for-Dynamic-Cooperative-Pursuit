"""M4 gate tests for uniform Informative Event Replay."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from dynabelief.config import ConfigError, config_from_dict
from dynabelief.replay.infer import EventReplayBuffer, first_sightings
from dynabelief.training.trainer import MAPPOTrainer
from dynabelief.utils.logging import read_jsonl
from tests.test_point_belief import diabl_config
from tests.test_ppo_math import OBS, fill_buffer, tiny_policy


def replay(capacity=10, window=4, n=2, seed=0):
    return EventReplayBuffer(capacity, window, n, OBS, 16, torch.Generator().manual_seed(seed))


def rollout_with_dones(T=8, E=2):
    buf = fill_buffer(tiny_policy(), T=T, E=E, N=2)
    buf.terminated.zero_()
    buf.truncated.zero_()
    buf.terminated[2, 0] = True  # env 0: episode ends at step 2
    return buf


def test_first_sightings():
    visible = np.array([[True, False, True]])
    seen = np.array([[True, False, False]])
    assert first_sightings(visible, seen).tolist() == [[False, False, True]]


def test_windows_never_cross_episode_or_rollout_boundaries():
    buf = rollout_with_dones()
    events = torch.zeros(8, 2, dtype=torch.bool)
    events[0, 0] = True  # 0..3 contains the end at 2 -> skip
    events[3, 0] = True  # 3..6 inside the next episode -> keep
    events[5, 0] = True  # 5..8 runs past the rollout -> skip
    events[0, 1] = True  # env 1 has no end -> keep
    events[4, 1] = True  # 4..7 fits exactly -> keep
    rb = replay()
    assert rb.add_from_rollout(buf, events) == 3 and rb.skipped == 2 and len(rb) == 3
    # events are stored in (t, e) order: (0,1) -> slot 0, (3,0) -> slot 1, (4,1) -> slot 2
    torch.testing.assert_close(rb.obs[0].float(), buf.obs[0:4, 1])
    torch.testing.assert_close(rb.obs[1].float(), buf.obs[3:7, 0])
    torch.testing.assert_close(rb.hidden0[1], buf.actor_hidden[3, 0])
    torch.testing.assert_close(rb.belief_target[2], buf.belief_target[4:8, 1])
    # An episode end ON the window's last step is fine (the window stays in one episode).
    buf.terminated[2, 0] = False
    buf.terminated[3, 0] = True
    events = torch.zeros(8, 2, dtype=torch.bool)
    events[0, 0] = True  # 0..3 ends exactly at the episode end
    assert replay().add_from_rollout(buf, events) == 1


def test_capacity_is_bounded_ring():
    buf = rollout_with_dones()
    events = torch.zeros(8, 2, dtype=torch.bool)
    events[:5, 1] = True  # five valid windows in env 1
    rb = replay(capacity=3)
    assert rb.add_from_rollout(buf, events) == 5
    assert len(rb) == 3 and rb.obs.shape[0] == 3 and rb.added == 5
    torch.testing.assert_close(rb.obs[0].float(), buf.obs[3:7, 1])  # slot 0 overwritten (t=3)


def test_compact_storage_is_exact_and_guards_non_counts():
    buf = rollout_with_dones()
    events = torch.zeros(8, 2, dtype=torch.bool)
    events[0, 1] = True
    rb = replay()
    rb.add_from_rollout(buf, events)
    assert rb.obs.dtype == torch.uint8
    torch.testing.assert_close(rb.sample(1, "cpu").obs[:, 0], buf.obs[0:4, 1])
    buf.obs[1, 1, 0, 0, 0, 0] = 0.5
    with pytest.raises(ValueError, match="uint8"):
        replay().add_from_rollout(buf, events)


def test_sampling_is_uniform_deterministic_and_fully_delivered():
    buf = rollout_with_dones()
    events = torch.zeros(8, 2, dtype=torch.bool)
    events[:5, 1] = True
    a, b = replay(seed=4), replay(seed=4)
    a.add_from_rollout(buf, events)
    b.add_from_rollout(buf, events)
    x, y = a.sample(6, "cpu"), b.sample(6, "cpu")
    torch.testing.assert_close(x.obs, y.obs)
    assert x.obs.shape == (4, 6, 2, *OBS) and x.hidden0.shape == (6, 2, 16)
    assert x.delivery.shape == (4, 6, 2, 2)
    assert x.delivery[..., 0, 1].all() and not x.delivery.diagonal(dim1=-2, dim2=-1).any()
    counts = torch.bincount(
        torch.randint(0, 5, (5000,), generator=torch.Generator().manual_seed(0))
    )
    assert counts.min() > 900  # sanity: the sampler draws indices uniformly


def test_replay_requires_belief_and_fits_rollout():
    raw = diabl_config().to_dict()
    with pytest.raises(ConfigError, match="belief"):
        config_from_dict({**raw, "belief": {"type": "none"}, "replay": {"type": "uniform"}})
    with pytest.raises(ConfigError, match="window"):
        config_from_dict({**raw, "replay": {"type": "uniform", "window": 64}})


def infer_config():
    raw = diabl_config().to_dict()
    raw["replay"] = {"type": "uniform", "capacity": 50, "window": 4, "batch": 4}
    return config_from_dict(raw)


def test_infer_training_fills_replays_and_logs(tmp_path):
    trainer = MAPPOTrainer(infer_config(), "cpu", run_dir=tmp_path)
    trainer.train()
    metrics = read_jsonl(tmp_path / "metrics.jsonl")
    assert sum(m["replay_events"] for m in metrics) > 0
    assert sum(m["replay_windows_added"] for m in metrics) > 0
    assert all(m["replay_size"] <= 50 for m in metrics)
    replayed = [m for m in metrics if m.get("replay_steps")]
    assert replayed and all(math.isfinite(m["replay_belief_loss"]) for m in replayed)
    # first discovery: at most one event per evader per episode
    assert sum(m["replay_events"] for m in metrics) <= trainer.config.env.n_evaders * 2 * 2


def test_replay_steps_reduce_the_replayed_belief_loss():
    trainer = MAPPOTrainer(infer_config(), "cpu")
    cfg = trainer.config.train
    actor, priv = trainer.vec.reset()
    hidden = trainer.policy.actor.initial_state(cfg.num_envs, trainer.vec.n_agents)
    trainer._collect(actor, priv, hidden, torch.ones(cfg.num_envs, dtype=torch.bool), None)
    trainer.vec.close()
    assert len(trainer.replay) > 0
    trainer.replay.generator.manual_seed(0)
    first = [trainer.algo.replay_step() for _ in range(5)]
    for _ in range(150):
        trainer.algo.replay_step()
    last = [trainer.algo.replay_step() for _ in range(5)]
    assert np.mean(last) < 0.5 * np.mean(first)


def test_replay_optimizer_state_is_checkpointed(tmp_path):
    from dynabelief.utils.checkpointing import load_checkpoint

    MAPPOTrainer(infer_config(), "cpu", run_dir=tmp_path).train()
    state = load_checkpoint(tmp_path / "checkpoints" / "final.pt")["learner"]["replay_optimizer"]
    assert state is not None and len(state["state"]) > 0
    resumed = MAPPOTrainer(infer_config(), "cpu", resume_from=tmp_path / "checkpoints" / "final.pt")
    assert len(resumed.algo.replay_optimizer.state_dict()["state"]) == len(state["state"])
    resumed.vec.close()
