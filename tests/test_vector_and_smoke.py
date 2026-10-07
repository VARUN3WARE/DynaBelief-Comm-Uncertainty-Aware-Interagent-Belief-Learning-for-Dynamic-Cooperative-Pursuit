from __future__ import annotations

import numpy as np
import pytest

from dynabelief.config import EnvConfig, config_from_dict
from dynabelief.envs.pursuit import PursuitEnv
from dynabelief.envs.vector import PursuitVecEnv
from dynabelief.training.rollout import run_random_smoke


def test_vector_shapes():
    vec = PursuitVecEnv(EnvConfig(max_cycles=5), num_envs=3, seed=0)
    actor, priv = vec.reset()
    assert actor.obs.shape == (3, 8, 7, 7, 3)
    assert priv.global_state.shape == (3, 3, 16, 16)
    result = vec.step(np.zeros((3, 8), dtype=np.int64))
    assert result.rewards.shape == (3, 8)
    assert result.done.shape == (3,)
    with pytest.raises(ValueError):
        vec.step(np.zeros((2, 8), dtype=np.int64))


def test_auto_reset_uses_next_episode_seed():
    config = EnvConfig(max_cycles=3)
    vec = PursuitVecEnv(config, num_envs=2, seed=4)
    vec.reset()
    zeros = np.zeros((2, 8), dtype=np.int64)
    for _ in range(2):
        assert not vec.step(zeros).done.any()
    result = vec.step(zeros)
    assert result.truncated.all()
    assert len(result.completed) == 2
    assert [s.seed for s in result.completed] == [vec.episode_seed(0, 0), vec.episode_seed(1, 0)]
    assert set(result.final_actor) == {0, 1}

    # The returned observation is the first observation of episode 1.
    reference = PursuitEnv(config)
    first_obs, _ = reference.reset(vec.episode_seed(0, 1))
    np.testing.assert_array_equal(result.actor.obs[0], first_obs.obs)
    assert not np.array_equal(result.final_actor[0].obs, result.actor.obs[0])


def test_sub_environments_get_different_episodes():
    vec = PursuitVecEnv(EnvConfig(max_cycles=5), num_envs=2, seed=0)
    _, priv = vec.reset()
    assert not np.array_equal(priv.occupancy[0], priv.occupancy[1])


def smoke_config(seed: int):
    return config_from_dict(
        {
            "experiment": {"seed": seed},
            "env": {"max_cycles": 10},
            "comm": {"packet_loss": 0.2},
            "train": {"num_envs": 2, "rollout_length": 8, "total_env_steps": 48},
        }
    )


def test_smoke_rollout_is_deterministic_and_seed_sensitive():
    a = run_random_smoke(smoke_config(0))
    b = run_random_smoke(smoke_config(0))
    c = run_random_smoke(smoke_config(1))
    assert a["trajectory_digest"] == b["trajectory_digest"]
    assert a["trajectory_digest"] != c["trajectory_digest"]
    assert a["comm"] == b["comm"]


def test_smoke_summary_reports_no_learning():
    summary = run_random_smoke(smoke_config(0))
    assert summary["learning"] is False and summary["policy"] == "random"
    assert summary["env_steps"] == 48
    assert summary["episodes"] == 4  # 2 envs x 24 steps / 10-step episodes
    assert summary["comm"]["messages_sent"] == 24 * 2 * 8 * 7
    assert 0.0 < summary["comm"]["drop_rate"] < 0.5
