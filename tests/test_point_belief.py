"""M4 gate tests for point-belief DIABL."""

from __future__ import annotations

import inspect
import math

import pytest
import torch

from dynabelief.config import ConfigError, config_from_dict
from dynabelief.models.actor_critic import MAPPOPolicy, RecurrentActor
from dynabelief.training.trainer import MAPPOTrainer
from dynabelief.utils.logging import read_jsonl
from tests.test_trainer import tiny

OBS = (7, 7, 3)


def diabl_config(**train):
    raw = tiny(rollout_length=16, **train).to_dict()
    raw["comm"].update(enabled=True, key_dim=4, message_dim=4)
    raw["model"]["use_own_position"] = True
    raw["belief"] = {"type": "point", "coef": 1.0}
    return config_from_dict(raw)


def belief_actor(seed=0):
    torch.manual_seed(seed)
    return RecurrentActor(OBS, 5, hidden_dim=16, conv_channels=4, comm_key_dim=4,
                          comm_value_dim=4, use_own_position=True, belief_type="point")  # fmt: skip


def test_belief_requires_comm_and_own_position():
    raw = diabl_config().to_dict()
    for section, key in (("comm", "enabled"), ("model", "use_own_position")):
        broken = {k: dict(v) if isinstance(v, dict) else v for k, v in raw.items()}
        broken[section][key] = False
        with pytest.raises(ConfigError, match="belief learning needs"):
            config_from_dict(broken)
    with pytest.raises(ConfigError):
        config_from_dict({**raw, "belief": {"type": "bogus"}})


def test_belief_loss_trains_the_sender_through_the_channel_only_if_delivered():
    actor = belief_actor()
    n = 3
    obs = torch.randint(0, 3, (2, 1, n, *OBS)).float().requires_grad_(True)
    pos = torch.rand(2, 1, n, 2)
    starts = torch.tensor([[True], [False]])
    for delivered in (True, False):
        obs.grad = None
        delivery = torch.zeros(2, 1, n, n, dtype=torch.bool)
        delivery[1, 0, 0, 1] = delivered  # message 1 -> 0 read at t=1
        _, hiddens = actor.unroll(obs, actor.initial_state(1, n), starts, delivery, pos,
                                  return_hidden=True)  # fmt: skip
        belief = actor.belief_head(hiddens)
        assert belief.shape == (2, 1, n, 2) and (belief >= 0).all() and (belief <= 1).all()
        belief[1, 0, 0].sum().backward()  # receiver 0's belief at t=1
        sender_grad = obs.grad[0, 0, 1].abs().sum()
        assert (sender_grad > 0) == delivered


def test_belief_head_can_be_fit_supervised():
    """Synthetic overfit: learn to report a fixed function of the own-position input."""
    actor = belief_actor()
    opt = torch.optim.Adam(actor.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(0)
    obs = torch.randint(0, 3, (4, 8, 3, *OBS), generator=g).float()
    pos = torch.rand(4, 8, 3, 2, generator=g)
    target = 1.0 - pos  # mirrored position
    starts = torch.zeros(4, 8, dtype=torch.bool)
    starts[0] = True
    delivery = torch.ones(4, 8, 3, 3, dtype=torch.bool)

    def loss():
        _, h = actor.unroll(obs, actor.initial_state(8, 3), starts, delivery, pos,
                            return_hidden=True)  # fmt: skip
        return (actor.belief_head(h) - target).pow(2).sum(-1).mean()

    first = loss().item()
    for _ in range(300):
        opt.zero_grad()
        value = loss()
        value.backward()
        opt.step()
    assert loss().item() < 0.05 * first


def test_actor_interfaces_take_no_belief_labels():
    for method in (RecurrentActor.step, RecurrentActor.unroll, MAPPOPolicy.act):
        params = set(inspect.signature(method).parameters)
        assert not params & {"belief_target", "belief_valid", "target", "occupancy"}, method


def test_diabl_training_collects_labels_and_logs_sender_gradient(tmp_path):
    trainer = MAPPOTrainer(diabl_config(), "cpu", run_dir=tmp_path)
    trainer.train()
    buf = trainer.buffer
    assert buf.belief_valid.any()
    assert (buf.belief_target[~buf.belief_valid] == 0).all()
    assert ((buf.belief_target >= 0) & (buf.belief_target <= 1)).all()
    metrics = read_jsonl(tmp_path / "metrics.jsonl")
    for m in metrics:
        for key in ("belief_loss", "belief_error_cells", "belief_valid_fraction",
                    "belief_sender_grad_norm", "weighted_belief_loss"):  # fmt: skip
            assert math.isfinite(m[key]), key
    assert any(m["belief_sender_grad_norm"] > 0 for m in metrics)
    assert all(0 < m["belief_valid_fraction"] <= 1 for m in metrics)


def test_ppo_update_optimizes_the_belief_loss():
    """The belief term really enters the actor's gradient step (not only the logs)."""
    raw = diabl_config().to_dict()
    raw["belief"]["coef"] = 20.0
    trainer = MAPPOTrainer(config_from_dict(raw), "cpu")
    cfg = trainer.config.train
    actor, priv = trainer.vec.reset()
    hidden = trainer.policy.actor.initial_state(cfg.num_envs, trainer.vec.n_agents)
    trainer._collect(actor, priv, hidden, torch.ones(cfg.num_envs, dtype=torch.bool), None)
    trainer.vec.close()
    algo, buf = trainer.algo, trainer.buffer

    def belief_loss():
        with torch.no_grad():
            mb = next(iter(buf.minibatches(1, 4, torch.Generator().manual_seed(0), buf.advantages)))
            return algo._minibatch_terms(mb, None).belief_loss.item()

    before = belief_loss()
    for _ in range(50):  # tiny config: 1 epoch x 2 minibatches per update (measured 0.21 -> 0.14)
        algo.update(buf)
    assert belief_loss() < 0.75 * before
