from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.distributions import Categorical

from dynabelief.algorithms.gae import build_next_values, compute_gae
from dynabelief.algorithms.mappo import MAPPO, explained_variance, masked_mean, ppo_loss_terms
from dynabelief.config import PPOConfig
from dynabelief.models.actor_critic import MAPPOPolicy
from dynabelief.training.buffer import RolloutBuffer
from dynabelief.utils.value_norm import ValueNorm

OBS, STATE = (7, 7, 3), (3, 16, 16)


# ---------------------------------------------------------------- GAE
def brute_force_gae(rewards, values, next_values, terminated, truncated, gamma, lam):
    """Independent reference: explicit per-episode lambda-return sums."""
    T, E, N = rewards.shape
    adv = np.zeros((T, E, N))
    for e in range(E):
        for n in range(N):
            for t in range(T):
                total, discount = 0.0, 1.0
                for k in range(t, T):
                    boot = 0.0 if terminated[k, e] else next_values[k, e, n]
                    delta = rewards[k, e, n] + gamma * boot - values[k, e, n]
                    total += discount * delta
                    if terminated[k, e] or truncated[k, e]:
                        break
                    discount *= gamma * lam
                adv[t, e, n] = total
    return adv


def random_flags(rng, T, E):
    term = rng.random((T, E)) < 0.15
    trunc = (rng.random((T, E)) < 0.15) & ~term
    return term, trunc


@pytest.mark.parametrize("gamma, lam", [(0.99, 0.95), (1.0, 1.0), (0.9, 0.0)])
def test_gae_matches_brute_force(gamma, lam):
    rng = np.random.default_rng(0)
    T, E, N = 12, 3, 2
    rewards, values, next_values = (rng.normal(size=(T, E, N)) for _ in range(3))
    term, trunc = random_flags(rng, T, E)
    expected = brute_force_gae(rewards, values, next_values, term, trunc, gamma, lam)
    adv, ret = compute_gae(
        *(torch.tensor(x, dtype=torch.float64) for x in (rewards, values, next_values)),
        torch.tensor(term), torch.tensor(trunc), gamma, lam,
    )  # fmt: skip
    np.testing.assert_allclose(adv.numpy(), expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(ret.numpy(), expected + values, rtol=1e-10, atol=1e-10)


def one_step(reward, value, next_value, terminated, truncated, gamma=0.9):
    t = torch.tensor([[[reward]]])
    adv, _ = compute_gae(
        t, torch.tensor([[[value]]]), torch.tensor([[[next_value]]]),
        torch.tensor([[terminated]]), torch.tensor([[truncated]]), gamma, 0.95,
    )  # fmt: skip
    return adv.item()


def test_termination_does_not_bootstrap_but_truncation_does():
    assert one_step(1.0, 0.5, 100.0, terminated=True, truncated=False) == pytest.approx(0.5)
    assert one_step(1.0, 0.5, 100.0, terminated=False, truncated=True) == pytest.approx(
        1.0 + 0.9 * 100.0 - 0.5
    )


def test_gae_rejects_inconsistent_inputs():
    z = torch.zeros(2, 1, 1)
    flags = torch.zeros(2, 1, dtype=torch.bool)
    with pytest.raises(ValueError):
        compute_gae(
            z,
            z,
            z,
            torch.ones(2, 1, dtype=torch.bool),
            torch.ones(2, 1, dtype=torch.bool),
            0.9,
            0.9,
        )
    with pytest.raises(ValueError):
        compute_gae(z, z, torch.zeros(3, 1, 1), flags, flags, 0.9, 0.9)


def test_build_next_values_uses_final_values_only_on_truncation():
    values = torch.arange(6.0).reshape(3, 2, 1)
    last = torch.tensor([[10.0], [20.0]])
    final = torch.full((3, 2, 1), -1.0)
    truncated = torch.tensor([[False, True], [False, False], [True, False]])
    nxt = build_next_values(values, last, truncated, final)
    expected = torch.tensor([[[2.0], [-1.0]], [[4.0], [5.0]], [[-1.0], [20.0]]])
    torch.testing.assert_close(nxt, expected)


# ---------------------------------------------------------------- losses
def loss(new_lp, old_lp, adv, mask=None, value_clip=0.0, **kw):
    n = torch.tensor(new_lp)
    defaults = dict(
        entropy=torch.zeros_like(n), new_values=torch.zeros_like(n), old_values=torch.zeros_like(n),
        value_targets=torch.zeros_like(n), clip_eps=0.2, value_clip_eps=value_clip, huber_delta=0.0,
    )  # fmt: skip
    defaults.update(kw)
    return ppo_loss_terms(
        n, torch.tensor(old_lp), torch.tensor(adv),
        mask=torch.ones_like(n, dtype=torch.bool) if mask is None else torch.tensor(mask),
        **defaults,
    )  # fmt: skip


def test_clipped_surrogate_values():
    import math

    up = math.log(1.5)  # ratio 1.5
    # positive advantage: gain is capped at (1 + eps) * A
    assert loss([up], [0.0], [2.0]).policy_loss.item() == pytest.approx(-1.2 * 2.0)
    # negative advantage: the pessimistic (unclipped) term is used
    assert loss([up], [0.0], [-2.0]).policy_loss.item() == pytest.approx(1.5 * 2.0)
    # inside the trust region: plain ratio * A
    small = math.log(1.1)
    assert loss([small], [0.0], [1.0]).policy_loss.item() == pytest.approx(-1.1)
    terms = loss([up, 0.0], [0.0, 0.0], [1.0, 1.0])
    assert terms.clip_fraction.item() == pytest.approx(0.5)
    assert terms.approx_kl.item() >= 0


def test_clipped_ratio_has_no_policy_gradient():
    new = torch.tensor([0.5], requires_grad=True)  # ratio e^0.5 > 1.2
    terms = ppo_loss_terms(
        new, torch.zeros(1), torch.ones(1), torch.zeros(1), torch.zeros(1), torch.zeros(1),
        torch.zeros(1), torch.ones(1, dtype=torch.bool), 0.2, 0.0, 0.0,
    )  # fmt: skip
    terms.policy_loss.backward()
    assert new.grad.abs().item() == 0.0


def test_value_clipping_takes_pessimistic_error():
    # old 0, new 1.0, target 1.0, clip 0.2 -> clipped prediction 0.2 has error 0.32
    terms = loss(
        [0.0],
        [0.0],
        [0.0],
        value_clip=0.2,
        new_values=torch.tensor([1.0]),
        old_values=torch.tensor([0.0]),
        value_targets=torch.tensor([1.0]),
    )
    assert terms.value_loss.item() == pytest.approx(0.5 * 0.8**2)
    unclipped = loss(
        [0.0], [0.0], [0.0], new_values=torch.tensor([1.0]), value_targets=torch.tensor([1.0])
    )
    assert unclipped.value_loss.item() == pytest.approx(0.0)


def test_masked_entries_do_not_affect_losses():
    a = loss([0.1, 5.0], [0.0, 0.0], [1.0, 100.0], mask=[True, False])
    b = loss([0.1], [0.0], [1.0])
    assert a.policy_loss.item() == pytest.approx(b.policy_loss.item())
    assert masked_mean(torch.tensor([1.0, 9.0]), torch.tensor([True, False])).item() == 1.0


def test_explained_variance():
    target = torch.tensor([1.0, 2.0, 3.0, 4.0])
    mask = torch.ones(4, dtype=torch.bool)
    assert explained_variance(target, target, mask) == pytest.approx(1.0)
    assert explained_variance(torch.full((4,), 2.5), target, mask) == pytest.approx(0.0)


# ---------------------------------------------------------------- value norm
def test_value_norm_matches_numpy_and_inverts():
    rng = np.random.default_rng(0)
    batches = [rng.normal(3.0, 2.0, size=n) for n in (10, 37, 5)]
    norm = ValueNorm()
    for b in batches:
        norm.update(torch.tensor(b))
    allv = np.concatenate(batches)
    assert norm.mean.item() == pytest.approx(allv.mean())
    assert norm.var.item() == pytest.approx(allv.var())
    x = torch.randn(20)
    torch.testing.assert_close(norm.denormalize(norm.normalize(x)), x, rtol=1e-5, atol=1e-5)
    with pytest.raises(ValueError):
        norm.update(torch.tensor([float("nan")]))


# ---------------------------------------------------------------- buffer
def tiny_policy(seed=0):
    torch.manual_seed(seed)
    return MAPPOPolicy(
        OBS, STATE, n_actions=5, max_cycles=50, hidden_dim=16, conv_channels=4, critic_hidden_dim=16
    )


def fill_buffer(policy, T=8, E=3, N=2, seed=0):
    """Roll the real actor over synthetic observations with episode resets."""
    g = torch.Generator().manual_seed(seed)
    buf = RolloutBuffer(T, E, N, OBS, STATE, hidden_dim=16)
    hidden = policy.actor.initial_state(E, N)
    start = torch.ones(E, dtype=torch.bool)
    for t in range(T):
        obs = torch.randint(0, 3, (E, N, *OBS), generator=g).float()
        actions, logp, new_hidden = policy.act(obs, hidden, start)
        terminated = torch.rand(E, generator=g) < 0.2
        truncated = (torch.rand(E, generator=g) < 0.2) & ~terminated
        buf.add(
            obs=obs, episode_start=start, actor_hidden=hidden, actions=actions, log_probs=logp,
            values=torch.randn(E, N, generator=g), rewards=torch.randn(E, N, generator=g),
            agent_mask=torch.ones(E, N, dtype=torch.bool), terminated=terminated,
            truncated=truncated, final_values=torch.randn(E, N, generator=g),
            global_state=torch.rand(E, *STATE, generator=g),
            pursuer_pos=torch.randint(0, 16, (E, N, 2), generator=g),
            step=torch.full((E,), t),
        )  # fmt: skip
        hidden, start = new_hidden, terminated | truncated
    return buf


def test_minibatches_cover_every_step_exactly_once():
    buf = fill_buffer(tiny_policy())
    buf.rewards.copy_(
        torch.arange(buf.rewards.numel(), dtype=torch.float32).reshape(buf.rewards.shape)
    )
    seen = []
    for mb in buf.minibatches(
        3, chunk_length=4, generator=torch.Generator().manual_seed(0), advantages=buf.rewards
    ):
        assert mb.obs.shape[0] == 4 and mb.obs.shape[2:] == (2, *OBS)
        assert mb.global_state.shape[2:] == STATE
        seen.append(mb.advantages.reshape(-1))
    assert sorted(torch.cat(seen).tolist()) == buf.rewards.reshape(-1).tolist()


def test_minibatch_hidden_is_chunk_start_hidden():
    buf = fill_buffer(tiny_policy())
    for mb in buf.minibatches(6, 4, torch.Generator().manual_seed(1), buf.advantages):
        for b in range(mb.actor_hidden0.shape[0]):
            matches = [
                torch.equal(mb.actor_hidden0[b], buf.actor_hidden[t, e])
                and torch.equal(mb.obs[0, b], buf.obs[t, e])
                for t in (0, 4)
                for e in range(3)
            ]
            assert any(matches)


@pytest.mark.parametrize("chunk_length", [1, 2, 4, 8])
def test_ratio_is_one_before_first_gradient_step(chunk_length):
    """Recomputed log-probs equal stored ones: buffer, chunking and unroll are aligned,
    including chunks that start mid-episode from a stored hidden state."""
    policy = tiny_policy()
    buf = fill_buffer(policy)
    assert (~buf.episode_start[::chunk_length]).any() or chunk_length == 8
    for mb in buf.minibatches(2, chunk_length, torch.Generator().manual_seed(0), buf.advantages):
        logits = policy.actor.unroll(mb.obs, mb.actor_hidden0, mb.episode_start)
        new_lp = Categorical(logits=logits).log_prob(mb.actions)
        torch.testing.assert_close(new_lp, mb.old_log_probs, rtol=1e-5, atol=1e-5)


def test_buffer_validation():
    buf = RolloutBuffer(4, 2, 2, OBS, STATE, 16)
    with pytest.raises(ValueError, match="missing"):
        buf.add(obs=torch.zeros(2, 2, *OBS))
    with pytest.raises(RuntimeError):
        list(buf.minibatches(1, 2, torch.Generator(), buf.advantages))
    full = fill_buffer(tiny_policy(), T=8)
    with pytest.raises(ValueError, match="divisible"):
        list(full.minibatches(1, 3, torch.Generator(), full.advantages))
    with pytest.raises(RuntimeError, match="full"):
        full.add()


# ---------------------------------------------------------------- update
def test_update_changes_parameters_with_finite_diagnostics():
    policy = tiny_policy()
    buf = fill_buffer(policy)
    algo = MAPPO(
        policy,
        PPOConfig(epochs=2, num_minibatches=2, chunk_length=4),
        "cpu",
        torch.Generator().manual_seed(0),
    )
    algo.compute_returns(buf, last_values=torch.zeros(3, 2))
    before = [p.detach().clone() for p in policy.parameters()]
    metrics = algo.update(buf)
    assert metrics["gradient_steps"] == 4
    for key in (
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "actor_grad_norm",
        "critic_grad_norm",
        "explained_variance",
    ):
        assert np.isfinite(metrics[key]), key
    assert any(not torch.equal(a, b) for a, b in zip(before, policy.parameters(), strict=True))
    assert algo.value_norm.count.item() == buf.returns.numel()


def test_compute_returns_consistent_with_gae():
    policy = tiny_policy()
    buf = fill_buffer(policy)
    algo = MAPPO(policy, PPOConfig(chunk_length=4), "cpu", torch.Generator())
    last = torch.randn(3, 2)
    algo.compute_returns(buf, last)
    nxt = build_next_values(buf.values, last, buf.truncated, buf.final_values)
    adv, ret = compute_gae(buf.rewards, buf.values, nxt, buf.terminated, buf.truncated, 0.99, 0.95)
    torch.testing.assert_close(buf.advantages, adv)
    torch.testing.assert_close(buf.returns, ret)


def test_value_clip_centre_is_critics_own_prediction():
    """Regression: the clip centre must be the critic's output at collection time, so
    re-normalizing it with ValueNorm stats updated on the new returns is wrong."""
    import copy

    policy = tiny_policy()
    algo = MAPPO(policy, PPOConfig(chunk_length=4), "cpu", torch.Generator().manual_seed(0))
    algo.value_norm.update(torch.randn(500) * 3 + 2)  # non-trivial stats from "earlier"
    buf = fill_buffer(policy)
    E, N = buf.E, buf.N
    for t in range(buf.T):  # store genuine critic predictions, in return units
        buf.values[t] = algo.values(buf.global_state[t], buf.pursuer_pos[t], buf.step[t])
    buf.returns.copy_(torch.randn_like(buf.returns) * 10 + 20)  # shifts the stats a lot
    old_norm = copy.deepcopy(algo.value_norm)
    algo.value_norm.update(buf.returns.reshape(-1))
    for mb in buf.minibatches(2, 4, torch.Generator().manual_seed(0), buf.advantages):
        raw = policy.critic(mb.global_state.flatten(0, 1), mb.pursuer_pos.flatten(0, 1),
                            mb.step.flatten(0, 1)).reshape(mb.old_values.shape)  # fmt: skip
        torch.testing.assert_close(old_norm.normalize(mb.old_values), raw, rtol=1e-4, atol=1e-4)
        assert not torch.allclose(algo.value_norm.normalize(mb.old_values), raw, atol=1e-2)
        terms = algo._minibatch_terms(mb, old_norm)
        unclipped = ppo_loss_terms(
            torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1), raw,
            old_norm.normalize(mb.old_values), algo.value_norm.normalize(mb.returns),
            mb.agent_mask, 0.2, 0.0, algo.config.huber_delta,
        )  # fmt: skip
        # Before any critic step, prediction == clip centre, so clipping changes nothing.
        torch.testing.assert_close(terms.value_loss, unclipped.value_loss, rtol=1e-4, atol=1e-5)
    assert E and N
