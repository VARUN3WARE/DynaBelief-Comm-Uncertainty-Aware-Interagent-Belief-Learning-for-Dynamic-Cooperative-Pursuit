"""Stop-gradient arm F and the evaluation-only message hooks of TarMACComm."""

from __future__ import annotations

import pytest
import torch

from dynabelief.config import ConfigError, config_from_dict, load_config
from dynabelief.models.comm import TarMACComm
from dynabelief.training.trainer import MAPPOTrainer
from dynabelief.utils.logging import read_jsonl
from tests.test_point_belief import OBS, belief_actor, diabl_config


def comm_inputs(seed=0, b=2, n=3, h=8):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(b, n, h, generator=g, requires_grad=True)
    delivery = torch.rand(b, n, n, generator=g) > 0.3
    delivery &= ~torch.eye(n, dtype=torch.bool)
    return hidden, delivery


def test_hooks_off_leave_the_computation_unchanged():
    torch.manual_seed(0)
    comm = TarMACComm(8, 4, 4)
    hidden, delivery = comm_inputs()
    base = comm(hidden, delivery)
    same = comm(hidden, delivery, sender_hidden=hidden, message_transform=lambda k, v: (k, v))
    for a, b in zip(base, same, strict=True):
        assert torch.equal(a, b)


def test_detached_messages_same_values_no_message_gradient():
    torch.manual_seed(0)
    comm = TarMACComm(8, 4, 4)
    hidden, delivery = comm_inputs()
    ctx, _ = comm(hidden, delivery)
    ctx_d, _ = comm(hidden, delivery, detach_messages=True)
    assert torch.equal(ctx, ctx_d)
    ctx_d.sum().backward()
    assert comm.key.weight.grad is None and comm.value.weight.grad is None
    assert comm.query.weight.grad.abs().sum() > 0  # the receiver's own query still learns


def test_sender_hidden_changes_only_the_messages():
    torch.manual_seed(0)
    comm = TarMACComm(8, 4, 4)
    hidden, delivery = comm_inputs()
    older = hidden.detach() * 0.5
    ctx, _ = comm(hidden, delivery, sender_hidden=older)
    manual = comm.value(older)
    attention = comm(hidden, delivery, sender_hidden=older)[1]
    torch.testing.assert_close(ctx, attention @ manual)


def test_stop_gradient_cuts_the_belief_path_to_senders_only():
    actor = belief_actor()
    n = 3
    obs = torch.randint(0, 3, (2, 1, n, *OBS)).float().requires_grad_(True)
    pos = torch.rand(2, 1, n, 2)
    starts = torch.tensor([[True], [False]])
    delivery = torch.zeros(2, 1, n, n, dtype=torch.bool)
    delivery[1, 0, 0, 1] = True  # message 1 -> 0, read at t=1
    for detach in (False, True):
        obs.grad = None
        actor.zero_grad()
        _, hiddens = actor.unroll(obs, actor.initial_state(1, n), starts, delivery, pos,
                                  return_hidden=True, detach_messages=detach)  # fmt: skip
        actor.belief_head(hiddens)[1, 0, 0].sum().backward()  # receiver 0 at t=1
        assert (obs.grad[0, 0, 1].abs().sum() > 0) == (not detach)  # sender 1
        assert obs.grad[1, 0, 0].abs().sum() > 0  # the receiver's own observation
        # value, not key: with one delivered message the softmax weight is 1 for any key
        value_grad = actor.comm.value.weight.grad
        assert (value_grad is not None and value_grad.abs().sum() > 0) == (not detach)


def test_channel_gradient_flag_needs_a_belief():
    raw = diabl_config().to_dict()
    raw["belief"] = {"type": "none", "channel_gradient": False}
    with pytest.raises(ConfigError, match="channel_gradient"):
        config_from_dict(raw)


def test_arm_f_trains_without_belief_gradient_on_senders(tmp_path):
    raw = diabl_config().to_dict()
    raw["belief"]["channel_gradient"] = False
    trainer = MAPPOTrainer(config_from_dict(raw), "cpu", run_dir=tmp_path)
    before = trainer.policy.actor.comm.value.weight.detach().clone()
    trainer.train()
    metrics = read_jsonl(tmp_path / "metrics.jsonl")
    assert all(m["belief_sender_grad_norm"] == 0.0 for m in metrics)
    assert all(m["belief_loss"] > 0 for m in metrics)
    # PPO still trains the message layers through the normal forward pass.
    assert not torch.equal(before, trainer.policy.actor.comm.value.weight)


def test_arm_f_config_differs_from_point_diabl_only_in_the_flag():
    a = load_config("configs/point_diabl.yaml").to_dict()
    f = load_config("configs/point_diabl_stopgrad.yaml").to_dict()
    assert f["belief"]["channel_gradient"] is False and a["belief"]["channel_gradient"] is True
    a["belief"].pop("channel_gradient"), f["belief"].pop("channel_gradient")
    a["experiment"].pop("name"), f["experiment"].pop("name")
    assert a == f
