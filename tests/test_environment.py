from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from pettingzoo.sisl import pursuit_v4
from pettingzoo.test import parallel_api_test

from dynabelief.config import EnvConfig
from dynabelief.envs.pursuit import PursuitEnv
from tests.helpers import bfs_action

SHORT = EnvConfig(max_cycles=20)


def rollout(env: PursuitEnv, seed: int, actions: np.ndarray):
    """Run a fixed action sequence; returns stacked observations/rewards/privileged occupancy."""
    actor, priv = env.reset(seed)
    obs, rewards, occ, pos = [actor.obs], [], [priv.occupancy], [priv.evader_pos]
    for a in actions:
        result = env.step(a)
        obs.append(result.actor.obs)
        rewards.append(result.rewards)
        occ.append(result.privileged.occupancy)
        pos.append(result.privileged.evader_pos)
        if result.done:
            break
    return np.stack(obs), np.stack(rewards), np.stack(occ), np.stack(pos)


def test_pettingzoo_parallel_api():
    parallel_api_test(pursuit_v4.parallel_env(**SHORT.pettingzoo_kwargs()), num_cycles=30)


def test_shapes_and_dtypes_match_pettingzoo_spaces():
    env = PursuitEnv(EnvConfig())
    raw = pursuit_v4.parallel_env(**EnvConfig().pettingzoo_kwargs())
    space = raw.observation_space(raw.possible_agents[0])
    actor, priv = env.reset(0)
    assert env.n_agents == 8 and env.n_actions == 5
    assert env.obs_shape == (7, 7, 3) == space.shape
    assert actor.obs.shape == (8, 7, 7, 3) and actor.obs.dtype == space.dtype == np.float32
    assert actor.agent_mask.shape == (8,) and actor.agent_mask.dtype == bool
    assert priv.global_state.shape == (3, 16, 16) and priv.global_state.dtype == np.float32
    assert priv.occupancy.shape == (16, 16)
    assert priv.pursuer_pos.shape == (8, 2) and priv.evader_pos.shape == (30, 2)
    assert priv.evader_alive.shape == (30,) and priv.evader_alive.all()
    result = env.step(np.zeros(8, dtype=np.int64))
    assert result.rewards.shape == (8,) and result.rewards.dtype == np.float32
    env.close()


def test_matches_raw_pettingzoo_observations_and_rewards():
    """The wrapper must not alter what PettingZoo produces."""
    rng = np.random.default_rng(0)
    actions = rng.integers(0, 5, size=(15, 8))
    env = PursuitEnv(SHORT)
    obs, rewards, _, _ = rollout(env, 11, actions)
    raw = pursuit_v4.parallel_env(**SHORT.pettingzoo_kwargs())
    raw_obs, _ = raw.reset(seed=11)
    agents = raw.possible_agents
    np.testing.assert_array_equal(obs[0], np.stack([raw_obs[a] for a in agents]))
    for t, a in enumerate(actions[: len(rewards)]):
        raw_obs, raw_rew, *_ = raw.step({ag: int(x) for ag, x in zip(agents, a, strict=True)})
        np.testing.assert_array_equal(obs[t + 1], np.stack([raw_obs[ag] for ag in agents]))
        np.testing.assert_allclose(rewards[t], [raw_rew[ag] for ag in agents], rtol=0, atol=1e-6)


def test_same_seed_and_actions_reproduce_trajectory():
    actions = np.random.default_rng(1).integers(0, 5, size=(20, 8))
    a = rollout(PursuitEnv(SHORT), 42, actions)
    b = rollout(PursuitEnv(SHORT), 42, actions)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)


def test_reset_with_same_seed_reproduces_on_same_instance():
    env = PursuitEnv(SHORT)
    actions = np.random.default_rng(2).integers(0, 5, size=(20, 8))
    a = rollout(env, 7, actions)
    rollout(env, 8, actions)  # consume RNG in between
    b = rollout(env, 7, actions)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)


def test_different_seeds_change_trajectory():
    actions = np.random.default_rng(3).integers(0, 5, size=(20, 8))
    a_obs, _, a_occ, _ = rollout(PursuitEnv(SHORT), 1, actions)
    b_obs, _, b_occ, _ = rollout(PursuitEnv(SHORT), 2, actions)
    assert not np.array_equal(a_occ[0], b_occ[0])  # different spawns
    assert not np.array_equal(a_obs, b_obs)


def test_evaders_move_when_not_frozen():
    actions = np.full((10, 8), 4)  # pursuers stay
    _, _, _, pos = rollout(PursuitEnv(SHORT), 5, actions)
    assert not np.array_equal(pos[0], pos[-1])


def test_random_policy_completes_multiple_episodes_with_truncation():
    env = PursuitEnv(EnvConfig(max_cycles=15))
    rng = np.random.default_rng(0)
    for episode in range(3):
        env.reset(100 + episode)
        steps = 0
        while True:
            result = env.step(rng.integers(0, 5, size=8))
            steps += 1
            if result.done:
                break
        stats = env.episode_stats
        assert stats is not None and stats.length == steps
        if not result.terminated:
            assert result.truncated and steps == 15
        assert len(stats.agent_returns) == 8
        assert stats.captures == len(stats.capture_steps)


def test_step_after_done_raises():
    env = PursuitEnv(EnvConfig(max_cycles=2))
    env.reset(0)
    env.step(np.zeros(8, dtype=np.int64))
    assert env.step(np.zeros(8, dtype=np.int64)).done
    with pytest.raises(RuntimeError):
        env.step(np.zeros(8, dtype=np.int64))


@pytest.mark.parametrize(
    "actions, error",
    [
        (np.zeros(7, dtype=np.int64), ValueError),
        (np.full(8, 5), ValueError),
        (np.full(8, -1), ValueError),
        (np.zeros(8, dtype=np.float32), TypeError),
    ],
)
def test_invalid_actions_raise(actions, error):
    env = PursuitEnv(SHORT)
    env.reset(0)
    with pytest.raises(error):
        env.step(actions)


def test_reset_requires_integer_seed():
    env = PursuitEnv(SHORT)
    with pytest.raises(TypeError):
        env.reset(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        env.reset(1.5)  # type: ignore[arg-type]


def test_coordinate_alignment_between_positions_grids_and_local_views():
    """(x, y) positions, [y, x] grids, and local [y, x, c] views must all agree."""
    config = EnvConfig()
    env = PursuitEnv(config)
    off = config.obs_range // 2
    rng = np.random.default_rng(4)
    actor, priv = env.reset(3)
    for _ in range(10):
        assert priv.occupancy.sum() == len({tuple(p) for p in priv.evader_pos[priv.evader_alive]})
        np.testing.assert_array_equal(priv.occupancy, (priv.global_state[2] > 0).astype(np.float32))
        for ex, ey in priv.evader_pos[priv.evader_alive]:
            assert priv.occupancy[ey, ex] == 1.0
        for agent, (px, py) in enumerate(priv.pursuer_pos):
            assert priv.global_state[1, py, px] >= 1
            assert actor.obs[agent, off, off, 1] >= 1  # observer sees itself at the centre
            for ex, ey in priv.evader_pos[priv.evader_alive]:
                dy, dx = ey - py, ex - px
                if abs(dx) <= off and abs(dy) <= off:
                    assert actor.obs[agent, dy + off, dx + off, 2] >= 1
        result = env.step(rng.integers(0, 5, size=8))
        actor, priv = result.actor, result.privileged


def place(env: PursuitEnv, pursuers: list[tuple[int, int]], evaders: list[tuple[int, int]]) -> None:
    """Test-only: teleport agents in the simulator and refresh its state layers."""
    sim = env._sim
    for i, (x, y) in enumerate(pursuers):
        sim.pursuer_layer.set_position(i, x, y)
    for i, (x, y) in enumerate(evaders):
        sim.evader_layer.set_position(i, x, y)
    sim.model_state[1] = sim.pursuer_layer.get_state_matrix()
    sim.model_state[2] = sim.evader_layer.get_state_matrix()


def test_action_semantics_match_documented_deltas():
    """0: x-1, 1: x+1, 2: y+1, 3: y-1, 4: stay, checked from a known open cell."""
    env = PursuitEnv(EnvConfig(n_pursuers=1, n_evaders=1, freeze_evaders=True, max_cycles=50))
    env.reset(0)
    place(env, pursuers=[(1, 1)], evaders=[(15, 15)])
    expected = [(1, (2, 1)), (2, (2, 2)), (0, (1, 2)), (3, (1, 1)), (4, (1, 1))]
    for action, position in expected:
        result = env.step(np.array([action]))
        assert not result.done
        assert tuple(result.privileged.pursuer_pos[0]) == position, (action, position)
    # Walls: moving out of bounds leaves the pursuer in place.
    place(env, pursuers=[(0, 0)], evaders=[(15, 15)])
    for action in (0, 3):
        result = env.step(np.array([action]))
        assert tuple(result.privileged.pursuer_pos[0]) == (0, 0)


@pytest.mark.parametrize("max_cycles", [1, 3])
def test_capture_on_final_step_is_termination_not_truncation(max_cycles):
    """PettingZoo flags only truncation on the max_cycles step; catching the last evader
    then must still count as termination (absorbing state, no value bootstrap)."""
    config = EnvConfig(n_pursuers=2, n_evaders=1, freeze_evaders=True, max_cycles=max_cycles)
    env = PursuitEnv(config)
    env.reset(0)
    place(env, pursuers=[(5, 15), (6, 15)], evaders=[(15, 0)])  # far apart, no capture
    for _ in range(max_cycles - 1):
        assert not env.step(np.array([4, 4])).done
    place(env, pursuers=[(1, 0), (0, 1)], evaders=[(0, 0)])  # corner needs 2 to surround
    result = env.step(np.array([4, 4]))
    assert result.captures == 1 and not result.privileged.evader_alive.any()
    assert result.terminated and not result.truncated
    stats = env.episode_stats.as_dict()
    assert stats["all_captured"] and stats["terminated"] and not stats["truncated"]


def test_capture_before_final_step_terminates():
    env = PursuitEnv(EnvConfig(n_pursuers=2, n_evaders=1, freeze_evaders=True, max_cycles=5))
    env.reset(0)
    place(env, pursuers=[(1, 0), (0, 1)], evaders=[(0, 0)])
    result = env.step(np.array([4, 4]))
    assert result.terminated and not result.truncated and result.captures == 1


def test_capture_bookkeeping_and_termination():
    """Frozen evaders, n_catch=1 without surround: walking onto an evader captures it."""
    config = EnvConfig(
        n_pursuers=1, n_evaders=3, freeze_evaders=True, surround=False, n_catch=1, max_cycles=200
    )
    env = PursuitEnv(config)
    _, priv = env.reset(5)
    sim_map = env._sim.map_matrix
    total_captures = 0
    for _ in range(200):
        alive_ids = np.flatnonzero(priv.evader_alive)
        target = priv.evader_pos[alive_ids[0]]
        action = bfs_action(sim_map, tuple(priv.pursuer_pos[0]), tuple(target))
        result = env.step(np.array([action]))
        total_captures += result.captures
        priv = result.privileged
        caught = ~priv.evader_alive
        assert (priv.evader_pos[caught] == -1).all()
        assert (priv.evader_pos[~caught] >= 0).all()
        assert caught.sum() == total_captures
        if result.done:
            break
    assert result.terminated and not result.truncated
    stats = env.episode_stats
    assert stats is not None
    assert stats.captures == 3 and stats.capture_rate == 1.0
    assert stats.capture_steps == sorted(stats.capture_steps) and len(stats.capture_steps) == 3
    assert stats.as_dict()["all_captured"]
    assert priv.occupancy.sum() == 0


def test_episode_stats_serialisable():
    env = PursuitEnv(EnvConfig(max_cycles=3))
    env.reset(0)
    for _ in range(3):
        env.step(np.zeros(8, dtype=np.int64))
    record = env.episode_stats.as_dict()
    assert record["length"] == 3 and record["truncated"]
    assert set(record) >= {
        "seed",
        "team_return",
        "captures",
        "capture_rate",
        "mean_time_to_capture",
    }


def test_config_fields_reach_pettingzoo():
    config = EnvConfig(x_size=12, y_size=10, n_pursuers=4, n_evaders=5, obs_range=5)
    env = PursuitEnv(config)
    actor, priv = env.reset(0)
    assert actor.obs.shape == (4, 5, 5, 3)
    assert priv.global_state.shape == (3, 10, 12)
    assert priv.evader_pos.shape == (5, 2)
    assert set(dataclasses.asdict(config)) == set(config.pettingzoo_kwargs())
