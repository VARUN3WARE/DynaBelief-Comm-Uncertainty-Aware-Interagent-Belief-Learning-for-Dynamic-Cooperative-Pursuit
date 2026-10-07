from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from dynabelief.algorithms.mappo import MAPPO
from dynabelief.config import Config, EnvConfig, ModelConfig, PPOConfig
from dynabelief.envs.vector import PursuitVecEnv
from dynabelief.utils.checkpointing import (
    CheckpointError,
    check_compatible,
    load_checkpoint,
    restore_rng_state,
    rng_state,
    save_checkpoint,
)
from tests.test_ppo_math import fill_buffer, tiny_policy


def trained_algo(seed=0):
    policy = tiny_policy(seed)
    algo = MAPPO(
        policy,
        PPOConfig(epochs=1, num_minibatches=2, chunk_length=4),
        "cpu",
        torch.Generator().manual_seed(seed),
    )
    buf = fill_buffer(policy)
    algo.compute_returns(buf, torch.zeros(3, 2))
    algo.update(buf)  # populates Adam moments and ValueNorm stats
    return algo


def test_roundtrip_restores_weights_optimizers_and_value_norm(tmp_path):
    algo = trained_algo()
    path = save_checkpoint(tmp_path / "ckpt.pt", {"learner": algo.state_dict(), "env_steps": 42})
    fresh = MAPPO(tiny_policy(seed=99), algo.config, "cpu", torch.Generator())
    loaded = load_checkpoint(path)
    fresh.load_state_dict(loaded["learner"])
    assert loaded["env_steps"] == 42
    for a, b in zip(
        algo.policy.state_dict().values(), fresh.policy.state_dict().values(), strict=True
    ):
        assert torch.equal(a, b)
    assert fresh.value_norm.count.item() == algo.value_norm.count.item() > 0
    a_state = algo.actor_optimizer.state_dict()["state"]
    b_state = fresh.actor_optimizer.state_dict()["state"]
    assert a_state.keys() == b_state.keys() and len(a_state) > 0
    for k in a_state:
        assert torch.equal(a_state[k]["exp_avg"], b_state[k]["exp_avg"])


def test_resumed_update_matches_uninterrupted_update(tmp_path):
    """Save after one update, reload into a fresh learner: the next update is identical."""
    algo = trained_algo()
    gen_state = algo.generator.get_state()
    save_checkpoint(
        tmp_path / "c.pt", {"learner": algo.state_dict(), "rng": rng_state(algo.generator)}
    )

    buf = fill_buffer(algo.policy, seed=5)
    algo.compute_returns(buf, torch.zeros(3, 2))
    expected = algo.update(buf)

    fresh = MAPPO(tiny_policy(seed=7), algo.config, "cpu", torch.Generator())
    data = load_checkpoint(tmp_path / "c.pt")
    fresh.load_state_dict(data["learner"])
    restore_rng_state(data["rng"], fresh.generator)
    assert torch.equal(fresh.generator.get_state(), gen_state)
    buf2 = fill_buffer(fresh.policy, seed=5)
    fresh.compute_returns(buf2, torch.zeros(3, 2))
    got = fresh.update(buf2)
    assert got == pytest.approx(expected)
    for a, b in zip(algo.policy.parameters(), fresh.policy.parameters(), strict=True):
        assert torch.equal(a, b)


def test_rng_state_roundtrip():
    gen = torch.Generator().manual_seed(3)
    state = rng_state(gen)
    expected = (
        random.random(),
        np.random.rand(),
        torch.rand(1).item(),  # noqa: NPY002
        torch.rand(1, generator=gen).item(),
    )
    restore_rng_state(state, gen)
    got = (
        random.random(),
        np.random.rand(),
        torch.rand(1).item(),  # noqa: NPY002
        torch.rand(1, generator=gen).item(),
    )
    assert got == expected


def test_atomic_save_leaves_no_temp_file(tmp_path):
    save_checkpoint(tmp_path / "a.pt", {"x": 1})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.pt"]


def test_missing_corrupt_and_wrong_version(tmp_path):
    with pytest.raises(CheckpointError, match="not found"):
        load_checkpoint(tmp_path / "nope.pt")
    (tmp_path / "bad.pt").write_bytes(b"not a checkpoint")
    with pytest.raises(CheckpointError, match="cannot read"):
        load_checkpoint(tmp_path / "bad.pt")
    torch.save({"format_version": 999}, tmp_path / "v.pt")
    with pytest.raises(CheckpointError, match="unsupported"):
        load_checkpoint(tmp_path / "v.pt")


def test_value_norm_mismatch_rejected():
    algo = trained_algo()
    other = MAPPO(
        tiny_policy(), PPOConfig(use_value_norm=False, chunk_length=4), "cpu", torch.Generator()
    )
    with pytest.raises(ValueError, match="use_value_norm"):
        other.load_state_dict(algo.state_dict())


def test_compatibility_rules():
    base = Config().to_dict()
    check_compatible(base, base, resume=True)

    evaluation = Config(env=EnvConfig(n_pursuers=4, max_cycles=100)).to_dict()
    check_compatible(base, evaluation, resume=False)  # generalization eval is allowed
    with pytest.raises(CheckpointError, match="env differs"):
        check_compatible(base, evaluation, resume=True)

    with pytest.raises(CheckpointError, match="model"):
        check_compatible(base, Config(model=ModelConfig(hidden_dim=64)).to_dict(), resume=False)
    with pytest.raises(CheckpointError, match="obs_range"):
        check_compatible(base, Config(env=EnvConfig(obs_range=5)).to_dict(), resume=False)


def test_vec_env_episode_index_resume():
    from dynabelief.envs.pursuit import PursuitEnv

    cfg = EnvConfig(max_cycles=2)
    zeros = np.zeros((2, 8), dtype=np.int64)
    for workers in (0, 2):
        vec = PursuitVecEnv(cfg, num_envs=2, seed=1, num_workers=workers)
        vec.reset()
        for _ in range(4):
            vec.step(zeros)
        index = vec.episode_index()
        assert index == [3, 3]  # initial reset + two auto-resets each
        vec.close()

        resumed = PursuitVecEnv(cfg, num_envs=2, seed=1, num_workers=workers, episode_index=index)
        actor, _ = resumed.reset()
        for env_i in range(2):
            first, _ = PursuitEnv(cfg).reset(resumed.episode_seed(env_i, 3))
            np.testing.assert_array_equal(actor.obs[env_i], first.obs)
        assert resumed.episode_index() == [4, 4]
        resumed.close()
    with pytest.raises(ValueError):
        PursuitVecEnv(cfg, num_envs=2, seed=1, episode_index=[0])
