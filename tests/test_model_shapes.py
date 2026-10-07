from __future__ import annotations

import inspect

import pytest
import torch

from dynabelief.models.actor_critic import CentralCritic, MAPPOPolicy, RecurrentActor
from dynabelief.models.encoder import LocalObsEncoder

OBS = (7, 7, 3)
STATE = (3, 16, 16)


def random_obs(*lead, obs_shape=OBS):
    return torch.randint(0, 3, (*lead, *obs_shape)).float()


def random_positions(batch, n_agents, x_size=16, y_size=16):
    x = torch.randint(0, x_size, (batch, n_agents, 1))
    y = torch.randint(0, y_size, (batch, n_agents, 1))
    return torch.cat([x, y], dim=-1)


@pytest.mark.parametrize("obs_shape", [(7, 7, 3), (5, 5, 3)])
def test_encoder_shapes(obs_shape):
    enc = LocalObsEncoder(obs_shape, conv_channels=8, out_dim=32)
    assert enc(random_obs(4, 3, obs_shape=obs_shape)).shape == (4, 3, 32)
    with pytest.raises(ValueError):
        enc(torch.zeros(2, 9, 9, 3))


def test_actor_step_shapes_and_parameter_sharing_across_agent_counts():
    actor = RecurrentActor(OBS, n_actions=5, hidden_dim=32, conv_channels=8)
    for n_agents in (1, 4, 8):
        h = actor.initial_state(3, n_agents)
        logits, h2 = actor.step(random_obs(3, n_agents), h, torch.zeros(3, dtype=torch.bool))
        assert logits.shape == (3, n_agents, 5) and h2.shape == (3, n_agents, 32)


def test_unroll_equals_stepwise_including_resets():
    torch.manual_seed(0)
    actor = RecurrentActor(OBS, n_actions=5, hidden_dim=16, conv_channels=8)
    length, batch, n_agents = 6, 3, 4
    obs = random_obs(length, batch, n_agents)
    starts = torch.zeros(length, batch, dtype=torch.bool)
    starts[0, :] = True
    starts[3, 1] = True  # env 1 starts a new episode mid-sequence
    h0 = torch.randn(batch, n_agents, 16)
    unrolled = actor.unroll(obs, h0, starts)
    h, stepped = h0, []
    for t in range(length):
        logits, h = actor.step(obs[t], h, starts[t])
        stepped.append(logits)
    torch.testing.assert_close(unrolled, torch.stack(stepped))


def test_episode_start_discards_previous_hidden_state():
    torch.manual_seed(0)
    actor = RecurrentActor(OBS, n_actions=5, hidden_dim=16, conv_channels=8)
    obs = random_obs(2, 4)
    start = torch.ones(2, dtype=torch.bool)
    a, _ = actor.step(obs, torch.randn(2, 4, 16), start)
    b, _ = actor.step(obs, torch.zeros(2, 4, 16), start)
    torch.testing.assert_close(a, b)
    c, _ = actor.step(obs, torch.randn(2, 4, 16), ~start)
    assert not torch.allclose(a, c)


def test_agents_are_processed_independently_in_m2_actor():
    """Without communication, agent j's input must not affect agent i's output."""
    torch.manual_seed(0)
    actor = RecurrentActor(OBS, n_actions=5, hidden_dim=16, conv_channels=8)
    obs = random_obs(1, 3)
    start = torch.ones(1, dtype=torch.bool)
    base, _ = actor.step(obs, actor.initial_state(1, 3), start)
    changed = obs.clone()
    changed[0, 2] += 1.0
    other, _ = actor.step(changed, actor.initial_state(1, 3), start)
    torch.testing.assert_close(base[0, :2], other[0, :2])
    assert not torch.allclose(base[0, 2], other[0, 2])


def test_actor_interface_has_no_privileged_inputs():
    privileged = {"global_state", "occupancy", "pursuer_pos", "evader_pos", "evader_alive", "state"}
    for method in (RecurrentActor.step, RecurrentActor.unroll, MAPPOPolicy.act):
        assert not privileged & set(inspect.signature(method).parameters), method.__name__


def test_critic_shapes_and_agent_specific_values():
    torch.manual_seed(0)
    critic = CentralCritic(STATE, max_cycles=500, hidden_dim=32, conv_channels=8)
    state = torch.randint(0, 3, (2, *STATE)).float()
    pos = random_positions(2, 8)
    values = critic(state, pos, torch.tensor([0, 250]))
    assert values.shape == (2, 8)
    pos[:, 1] = pos[:, 0]  # two agents on the same cell get the same value
    same = critic(state, pos, torch.tensor([0, 250]))
    torch.testing.assert_close(same[:, 0], same[:, 1])
    later = critic(state, pos, torch.tensor([400, 250]))
    assert not torch.allclose(later[0], same[0])  # time matters
    torch.testing.assert_close(later[1], same[1])


def test_critic_works_for_other_agent_counts_and_grid_sizes():
    critic = CentralCritic((3, 10, 12), max_cycles=100, hidden_dim=16, conv_channels=4)
    out = critic(
        torch.zeros(3, 3, 10, 12),
        random_positions(3, 5, x_size=12, y_size=10),
        torch.zeros(3, dtype=torch.long),
    )
    assert out.shape == (3, 5)


def test_policy_act_and_finite_gradients():
    torch.manual_seed(0)
    policy = MAPPOPolicy(
        OBS,
        STATE,
        n_actions=5,
        max_cycles=500,
        hidden_dim=32,
        conv_channels=8,
        critic_hidden_dim=32,
    )
    h = policy.actor.initial_state(4, 8)
    actions, logp, h2 = policy.act(random_obs(4, 8), h, torch.ones(4, dtype=torch.bool))
    assert actions.shape == (4, 8) and logp.shape == (4, 8) and h2.shape == (4, 8, 32)
    assert ((actions >= 0) & (actions < 5)).all() and (logp <= 0).all()
    det, _, _ = policy.act(random_obs(4, 8), h, torch.ones(4, dtype=torch.bool), deterministic=True)
    assert det.shape == (4, 8)

    logits = policy.actor.unroll(random_obs(5, 4, 8), h, torch.zeros(5, 4, dtype=torch.bool))
    values = policy.critic(torch.rand(4, *STATE), random_positions(4, 8), torch.zeros(4))
    (logits.logsumexp(-1).mean() + values.pow(2).mean()).backward()
    for name, param in policy.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
