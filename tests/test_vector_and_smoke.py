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


def run_vec(num_workers: int, steps: int = 25):
    vec = PursuitVecEnv(EnvConfig(max_cycles=7), num_envs=5, seed=3, num_workers=num_workers)
    rng = np.random.default_rng(0)
    actor, priv = vec.reset()
    trace = [actor.obs.copy(), priv.global_state.copy()]
    completed = []
    for _ in range(steps):
        result = vec.step(rng.integers(0, 5, size=(5, 8)))
        trace += [result.actor.obs, result.rewards, result.done, result.privileged.step]
        trace += [result.final_actor[i].obs for i in sorted(result.final_actor)]
        completed += [s.as_dict() for s in result.completed]
    vec.close()
    return trace, completed


def test_subprocess_backend_matches_in_process_exactly():
    reference, ref_completed = run_vec(num_workers=0)
    for workers in (1, 2, 5):
        trace, completed = run_vec(num_workers=workers)
        assert len(trace) == len(reference)
        for a, b in zip(reference, trace, strict=True):
            np.testing.assert_array_equal(a, b)
        assert completed == ref_completed


def test_worker_errors_propagate_and_close_is_idempotent():
    from dynabelief.envs.vector import WorkerError

    vec = PursuitVecEnv(EnvConfig(max_cycles=5), num_envs=2, seed=0, num_workers=2)
    vec.reset()
    with pytest.raises(WorkerError, match="actions must be in"):
        vec.step(np.full((2, 8), 9))
    vec.close()
    vec.close()
    with pytest.raises(RuntimeError):
        vec.reset()


def test_privileged_step_counts_episode_steps():
    vec = PursuitVecEnv(EnvConfig(max_cycles=3), num_envs=1, seed=0)
    _, priv = vec.reset()
    assert priv.step.tolist() == [0]
    zeros = np.zeros((1, 8), dtype=np.int64)
    assert vec.step(zeros).privileged.step.tolist() == [1]
    assert vec.step(zeros).privileged.step.tolist() == [2]
    result = vec.step(zeros)  # truncates, auto-resets
    assert result.final_privileged[0].step.tolist() == 3
    assert result.privileged.step.tolist() == [0]


def test_invalid_worker_count():
    with pytest.raises(ValueError):
        PursuitVecEnv(EnvConfig(), num_envs=2, seed=0, num_workers=3)
