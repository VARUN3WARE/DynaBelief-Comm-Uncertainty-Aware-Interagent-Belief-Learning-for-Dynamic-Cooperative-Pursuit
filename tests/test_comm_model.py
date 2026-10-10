"""M3 gate tests for TarMAC communication in the actor."""

from __future__ import annotations

import pytest
import torch

from dynabelief.models.actor_critic import MAPPOPolicy, RecurrentActor
from dynabelief.models.comm import TarMACComm

OBS = (7, 7, 3)


def comm_actor(seed: int = 0, hidden: int = 16) -> RecurrentActor:
    torch.manual_seed(seed)
    return RecurrentActor(OBS, 5, hidden_dim=hidden, conv_channels=4, comm_key_dim=8,
                          comm_value_dim=8)  # fmt: skip


def obs(*lead):
    return torch.randint(0, 3, (*lead, *OBS)).float()


def full_delivery(batch, n):
    return ~torch.eye(n, dtype=torch.bool).expand(batch, n, n).clone()


def test_attention_rows_are_distributions_or_zero():
    torch.manual_seed(0)
    comm = TarMACComm(hidden_dim=16, key_dim=8, value_dim=8)
    delivery = torch.rand(3, 5, 5) < 0.5
    delivery[0, 2] = False  # receiver 2 of env 0 gets nothing
    context, attention = comm(torch.randn(3, 5, 16), delivery)
    sums = attention.sum(-1)
    torch.testing.assert_close(sums[delivery.any(-1)], torch.ones(int(delivery.any(-1).sum())))
    assert (sums[~delivery.any(-1)] == 0).all()
    assert (attention[~delivery] == 0).all()
    assert torch.isfinite(context).all() and (context[0, 2] == 0).all()


@pytest.mark.parametrize("n_agents", [2, 4, 8])
def test_shapes_for_variable_agent_counts(n_agents):
    actor = comm_actor()
    h = actor.initial_state(3, n_agents)
    logits, h2 = actor.step(obs(3, n_agents), h, torch.zeros(3, dtype=torch.bool),
                            full_delivery(3, n_agents))  # fmt: skip
    assert logits.shape == (3, n_agents, 5) and h2.shape == (3, n_agents, 16)
    assert actor.last_attention.shape == (3, n_agents, n_agents)


def test_dropped_message_has_no_effect_and_no_gradient():
    actor = comm_actor()
    n = 4
    hidden = torch.randn(1, n, 16, requires_grad=True)
    o = obs(1, n)
    start = torch.zeros(1, dtype=torch.bool)
    delivery = full_delivery(1, n)
    delivery[0, 0, 3] = False  # message 3 -> 0 dropped
    logits, _ = actor.step(o, hidden, start, delivery)
    logits[0, 0].sum().backward()
    assert hidden.grad[0, 3].abs().sum() == 0  # sender 3 cannot affect receiver 0
    assert hidden.grad[0, 1].abs().sum() > 0  # delivered senders do

    changed = hidden.detach().clone()
    changed[0, 3] += 5.0
    a, _ = actor.step(o, hidden.detach(), start, delivery)
    b, _ = actor.step(o, changed, start, delivery)
    torch.testing.assert_close(a[0, 0], b[0, 0])
    assert not torch.allclose(a[0, 1], b[0, 1])  # 3 -> 1 was delivered


def test_all_messages_dropped_equals_no_communication():
    """With nothing delivered every agent is processed independently (the No-Comm case)."""
    actor = comm_actor()
    n = 3
    o = obs(1, n)
    none = torch.zeros(1, n, n, dtype=torch.bool)
    start = torch.zeros(1, dtype=torch.bool)
    base, _ = actor.step(o, torch.randn(1, n, 16), start, none)
    hidden = torch.randn(1, n, 16)
    a, _ = actor.step(o, hidden, start, none)
    hidden_other = hidden.clone()
    hidden_other[0, 1:] += 3.0
    o_other = o.clone()
    o_other[0, 1:] += 1.0
    b, _ = actor.step(o_other, hidden_other, start, none)
    torch.testing.assert_close(a[0, 0], b[0, 0])
    assert base.shape == a.shape


def test_no_message_crosses_an_episode_boundary():
    actor = comm_actor()
    with torch.no_grad():  # trained biases are non-zero; init_layer starts them at 0
        for layer in (actor.comm.query, actor.comm.key, actor.comm.value):
            layer.bias.normal_()
    n = 3
    o = obs(1, n)
    start = torch.ones(1, dtype=torch.bool)
    a, _ = actor.step(o, torch.randn(1, n, 16), start, full_delivery(1, n))
    b, _ = actor.step(o, torch.randn(1, n, 16), start, full_delivery(1, n))
    torch.testing.assert_close(a, b)  # previous-episode hidden states are irrelevant
    # All agents of an env reset together, so the masking matters for "phantom" messages
    # built from zeroed hidden states (bias only): at a start nothing is read at all.
    none = torch.zeros(1, n, n, dtype=torch.bool)
    c, _ = actor.step(o, torch.randn(1, n, 16), start, none)
    torch.testing.assert_close(a, c)


def test_unroll_equals_stepwise_with_messages_and_resets():
    actor = comm_actor()
    L, B, n = 6, 2, 4
    o = obs(L, B, n)
    starts = torch.zeros(L, B, dtype=torch.bool)
    starts[0] = True
    starts[3, 1] = True
    delivery = torch.rand(L, B, n, n) < 0.7
    h0 = torch.randn(B, n, 16)
    unrolled = actor.unroll(o, h0, starts, delivery)
    h, stepped = h0, []
    for t in range(L):
        logits, h = actor.step(o[t], h, starts[t], delivery[t])
        stepped.append(logits)
    torch.testing.assert_close(unrolled, torch.stack(stepped))


def test_receiver_loss_reaches_sender_encoder_through_the_channel():
    """The differentiable channel: a loss on agent 0 at t=1 trains agent 1's encoder at t=0."""
    actor = comm_actor()
    n = 3
    o = obs(2, 1, n).requires_grad_(True)
    starts = torch.tensor([[True], [False]])
    delivery = torch.zeros(2, 1, n, n, dtype=torch.bool)
    # 1 -> 0 and 2 -> 0, read at t=1. Two messages, so the attention weights (and hence
    # key/query) matter: with a single delivered message softmax is exactly 1 and only
    # the value path carries gradient.
    delivery[1, 0, 0, 1] = delivery[1, 0, 0, 2] = True
    logits = actor.unroll(o, actor.initial_state(1, n), starts, delivery)
    logits[1, 0, 0].sum().backward()
    assert o.grad[0, 0, 1].abs().sum() > 0  # sender's t=0 observation influences receiver
    assert o.grad[0, 0, 2].abs().sum() > 0
    for name in ("key", "value", "query"):
        grad = getattr(actor.comm, name).weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name


def test_comm_actor_requires_delivery_and_policy_threads_it():
    actor = comm_actor()
    with pytest.raises(ValueError, match="delivery"):
        actor.step(obs(1, 2), actor.initial_state(1, 2), torch.zeros(1, dtype=torch.bool))
    policy = MAPPOPolicy(OBS, (3, 16, 16), 5, 50, hidden_dim=16, conv_channels=4,
                         critic_hidden_dim=16, comm_key_dim=8, comm_value_dim=8)  # fmt: skip
    actions, logp, h = policy.act(obs(2, 3), policy.actor.initial_state(2, 3),
                                  torch.ones(2, dtype=torch.bool),
                                  delivery=full_delivery(2, 3))  # fmt: skip
    assert actions.shape == (2, 3) and h.shape == (2, 3, 16)


def test_attention_entropy_diagnostic():
    attention = torch.tensor([[[0.0, 0.5, 0.5], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    delivery = torch.tensor([[[False, True, True], [False, False, False], [True, False, False]]])
    entropy = TarMACComm.attention_entropy(attention, delivery)
    assert entropy.item() == pytest.approx((torch.log(torch.tensor(2.0)).item() + 0.0) / 2)
