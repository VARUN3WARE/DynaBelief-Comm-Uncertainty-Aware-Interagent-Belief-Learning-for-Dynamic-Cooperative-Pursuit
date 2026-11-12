"""Arm AR: the channel-randomized training channel (per-episode clean / bursty / low send rate)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from dynabelief.comm.channel import RandomizedTrainChannel
from dynabelief.config import ConfigError, config_from_dict, load_config
from tests.test_posterior_target import REPO, small

E, N = 64, 8
EYE = np.eye(N, dtype=bool)


def _channel(seed=0):
    ch = RandomizedTrainChannel(N, E, np.random.default_rng(seed))
    ch.new_episodes(np.ones(E, dtype=bool))
    return ch


def test_modes_are_drawn_in_the_configured_shares():
    ch = RandomizedTrainChannel(N, 4000, np.random.default_rng(0))
    ch.new_episodes(np.ones(4000, dtype=bool))
    share = {m: float((ch.mode == i).mean()) for i, m in enumerate(ch.MODES)}
    assert abs(share["clean"] - 0.4) < 0.03 and abs(share["bursty"] - 0.3) < 0.03
    assert abs(share["budget"] - 0.3) < 0.03


def test_clean_envs_deliver_everything_and_never_to_self():
    ch = _channel()
    alive = np.ones((E, N), dtype=bool)
    alive[:, 3] = False
    d = ch.sample_delivery(alive)
    assert d.shape == (E, N, N) and not d[:, EYE].any()
    assert not d[:, 3].any() and not d[:, :, 3].any()
    clean = ch.mode == 0
    expect = ~EYE & alive[0][:, None] & alive[0][None, :]
    assert (d[clean] == expect).all()


def test_bursty_envs_drop_whole_steps_at_their_loss_rate():
    ch = _channel(1)
    alive = np.ones((E, N), dtype=bool)
    steps = np.stack([ch.sample_delivery(alive) for _ in range(3000)])  # [T, E, N, N]
    bursty = np.flatnonzero(ch.mode == 1)
    assert len(bursty) > 5
    links = steps[:, bursty][:, :, ~EYE]  # [T, e, links]
    assert ((links.all(-1)) | (~links.any(-1))).all()  # team-wide: all or nothing per step
    lost = 1 - links.any(-1).mean(0)
    err = lost - ch.loss[bursty]  # burst-50 envs see ~60 bursts in 3000 steps: noisy per env
    assert abs(err.mean()) < 0.03 and np.abs(err).max() < 0.2


def test_budget_envs_send_per_sender_at_their_rate():
    ch = _channel(2)
    alive = np.ones((E, N), dtype=bool)
    steps = np.stack([ch.sample_delivery(alive) for _ in range(2000)])
    budget = np.flatnonzero(ch.mode == 2)
    assert len(budget) > 5
    sent = steps[:, budget].any(axis=2)  # [T, e, N_send]: a sender reaches someone
    assert np.abs(sent.mean((0, 2)) - ch.rate[budget]).max() < 0.03
    per_sender = steps[:, budget].sum(axis=2)  # receivers reached by each sender
    assert set(np.unique(per_sender)) <= {0, N - 1}


def test_new_episodes_resample_only_those_envs():
    ch = _channel(3)
    before = (ch.mode.copy(), ch.loss.copy(), ch.rate.copy())
    start = np.zeros(E, dtype=bool)
    start[:32] = True
    ch.new_episodes(start)
    assert (ch.mode[32:] == before[0][32:]).all() and (ch.loss[32:] == before[1][32:]).all()
    assert (ch.mode[:32] != before[0][:32]).any()


def test_config_and_training_run(tmp_path):
    from dynabelief.training.trainer import MAPPOTrainer

    assert (
        load_config(REPO / "configs" / "point_diabl_robust.yaml").comm.train_channel == "randomized"
    )
    raw = small("point_diabl").to_dict()
    raw["comm"]["train_channel"] = "bogus"
    with pytest.raises(ConfigError, match="train_channel"):
        config_from_dict(raw)
    raw["comm"]["train_channel"] = "randomized"
    trainer = MAPPOTrainer(config_from_dict(raw), "cpu", run_dir=tmp_path)
    trainer.train()
    assert isinstance(trainer.channel, RandomizedTrainChannel)
    rows = [json.loads(x) for x in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert all(np.isfinite(r["belief_loss"]) for r in rows)
    assert 0.0 < rows[-1]["comm_drop_rate"] < 1.0  # some messages are lost in training
