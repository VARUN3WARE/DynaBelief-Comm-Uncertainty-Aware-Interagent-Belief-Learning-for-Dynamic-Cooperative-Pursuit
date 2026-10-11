"""M5 gate tests for D-DIABL (occupancy-map belief)."""

from __future__ import annotations

import inspect
import math

import pytest
import torch

from dynabelief.beliefs.losses import map_loss
from dynabelief.config import ConfigError, config_from_dict
from dynabelief.models.actor_critic import MAPPOPolicy, RecurrentActor
from dynabelief.models.belief import MapBeliefHead
from dynabelief.training.trainer import MAPPOTrainer, map_prior_logit
from dynabelief.utils.logging import read_jsonl
from tests.test_point_belief import diabl_config

OBS, GRID = (7, 7, 3), (16, 16)


def map_config(replay: bool = False):
    raw = diabl_config().to_dict()
    raw["belief"]["type"] = "map"
    if replay:
        raw["replay"] = {"type": "uniform", "capacity": 50, "window": 4, "batch": 4}
    return config_from_dict(raw)


def map_actor(seed=0, prior=-1.7):
    torch.manual_seed(seed)
    return RecurrentActor(OBS, 5, hidden_dim=16, conv_channels=4, comm_key_dim=4,
                          comm_value_dim=4, use_own_position=True, belief_type="map",
                          grid_yx=GRID, belief_prior_logit=prior)  # fmt: skip


def test_map_belief_requires_comm_and_position():
    raw = map_config().to_dict()
    raw["model"]["use_own_position"] = False
    with pytest.raises(ConfigError, match="belief learning needs"):
        config_from_dict(raw)


def test_head_shapes_and_prior_initialization():
    head = MapBeliefHead(16, GRID, prior_logit=-1.7)
    logits = head(torch.zeros(3, 4, 2, 16))  # [L, B, N, H]
    assert logits.shape == (3, 4, 2, *GRID)
    torch.testing.assert_close(logits, torch.full_like(logits, -1.7))  # zero input -> bias
    p = 30 / 193
    assert map_prior_logit(map_config()) == pytest.approx(math.log(p / (1 - p)))


def test_map_loss_masks_obstacles_and_agents():
    L, B, N = 2, 1, 2
    occupancy = torch.zeros(L, B, *GRID)
    occupancy[0, 0, 3, 4] = 1.0
    free = torch.ones(L, B, *GRID, dtype=torch.bool)
    free[..., 5:9, 5:9] = False
    agent_mask = torch.ones(L, B, N, dtype=torch.bool)
    logits = torch.zeros(L, B, N, *GRID)
    base, metrics = map_loss(logits, occupancy, free, agent_mask)
    assert base.item() == pytest.approx(math.log(2))  # p = 0.5 everywhere
    assert metrics["belief_brier"].item() == pytest.approx(0.25)
    poisoned = logits.clone()
    poisoned[..., 5:9, 5:9] = 50.0  # obstacle cells: ignored
    poisoned[:, :, 1] = -30.0  # agent 1: masked out below
    mask = agent_mask.clone()
    mask[:, :, 1] = False
    a, _ = map_loss(poisoned, occupancy, free, mask)
    b, _ = map_loss(logits[:, :, :1], occupancy, free, mask[:, :, :1])
    assert a.item() == pytest.approx(b.item())


def test_receiver_map_trains_sender_only_through_delivered_messages():
    actor = map_actor()
    n = 3
    obs = torch.randint(0, 3, (2, 1, n, *OBS)).float().requires_grad_(True)
    pos = torch.rand(2, 1, n, 2)
    starts = torch.tensor([[True], [False]])
    occupancy = torch.zeros(2, 1, *GRID)
    occupancy[:, 0, 2, 2] = 1.0
    free = torch.ones(2, 1, *GRID, dtype=torch.bool)
    for delivered in (True, False):
        obs.grad = None
        actor.zero_grad()
        delivery = torch.zeros(2, 1, n, n, dtype=torch.bool)
        delivery[1, 0, 0, 1] = delivery[1, 0, 0, 2] = delivered
        _, hiddens = actor.unroll(obs, actor.initial_state(1, n), starts, delivery, pos,
                                  return_hidden=True)  # fmt: skip
        mask = torch.zeros(2, 1, n, dtype=torch.bool)
        mask[1, 0, 0] = True  # receiver 0 at t = 1 only
        loss, _ = map_loss(actor.belief_head(hiddens), occupancy, free, mask)
        loss.backward()
        assert (obs.grad[0, 0, 1].abs().sum() > 0) == delivered
        comm_grad = actor.comm.value.weight.grad
        assert (comm_grad is not None and comm_grad.abs().sum() > 0) == delivered


def test_map_head_can_be_fit_supervised():
    """Synthetic overfit: map a constant observation to a fixed occupancy pattern."""
    actor = map_actor(prior=0.0)
    opt = torch.optim.Adam(actor.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(0)
    obs = torch.randint(0, 3, (3, 4, 2, *OBS), generator=g).float()
    pos = torch.rand(3, 4, 2, 2, generator=g)
    occupancy = (torch.rand(3, 4, *GRID, generator=g) < 0.15).float()
    occupancy[:] = occupancy[0]  # same map at every step: learnable
    free = torch.ones(3, 4, *GRID, dtype=torch.bool)
    starts = torch.zeros(3, 4, dtype=torch.bool)
    starts[0] = True
    delivery = torch.ones(3, 4, 2, 2, dtype=torch.bool)
    mask = torch.ones(3, 4, 2, dtype=torch.bool)

    def loss():
        _, h = actor.unroll(obs, actor.initial_state(4, 2), starts, delivery, pos,
                            return_hidden=True)  # fmt: skip
        return map_loss(actor.belief_head(h), occupancy, free, mask)[0]

    first = loss().item()
    for _ in range(300):
        opt.zero_grad()
        value = loss()
        value.backward()
        opt.step()
    assert loss().item() < 0.1 * first


def test_actor_interfaces_take_no_occupancy():
    for method in (RecurrentActor.step, RecurrentActor.unroll, MAPPOPolicy.act):
        params = set(inspect.signature(method).parameters)
        assert not params & {"occupancy", "global_state", "belief_target", "free"}, method


@pytest.mark.parametrize("replay", [False, True])
def test_d_diabl_training_logs_map_metrics(tmp_path, replay):
    trainer = MAPPOTrainer(map_config(replay), "cpu", run_dir=tmp_path)
    trainer.train()
    assert trainer.policy.actor.belief_type == "map"
    metrics = read_jsonl(tmp_path / "metrics.jsonl")
    for m in metrics:
        for key in ("belief_loss", "belief_brier", "belief_sender_grad_norm"):
            assert math.isfinite(m[key]), key
        assert 0.0 < m["belief_loss"] < 1.0  # BCE per free cell starts near the prior
    if replay:
        assert any(m.get("replay_steps") for m in metrics)
        assert trainer.replay.occupancy.dtype == torch.uint8
        assert trainer.replay.occupancy[: len(trainer.replay)].sum() > 0


def test_ppo_update_optimizes_the_map_loss():
    raw = map_config().to_dict()
    raw["belief"]["coef"] = 20.0
    trainer = MAPPOTrainer(config_from_dict(raw), "cpu")
    cfg = trainer.config.train
    actor, priv = trainer.vec.reset()
    hidden = trainer.policy.actor.initial_state(cfg.num_envs, trainer.vec.n_agents)
    trainer._collect(actor, priv, hidden, torch.ones(cfg.num_envs, dtype=torch.bool), None)
    trainer.vec.close()
    algo, buf = trainer.algo, trainer.buffer

    def current():
        with torch.no_grad():
            mb = next(iter(buf.minibatches(1, 4, torch.Generator().manual_seed(0), buf.advantages)))
            return algo._minibatch_terms(mb, None).belief_loss.item()

    before = current()
    density = buf.global_state[:, :, 2][buf.global_state[:, :, 0] == 0].gt(0).float().mean().item()
    best_constant = -(density * math.log(density) + (1 - density) * math.log(1 - density))
    assert before == pytest.approx(best_constant, abs=0.01)  # prior-initialized bias
    for _ in range(40):
        algo.update(buf)
    # Evaders move randomly, so BCE is bounded away from 0: require a clear drop below the
    # best constant prediction (measured: 0.421 -> 0.390 after 40 updates).
    assert current() < before - 0.02
