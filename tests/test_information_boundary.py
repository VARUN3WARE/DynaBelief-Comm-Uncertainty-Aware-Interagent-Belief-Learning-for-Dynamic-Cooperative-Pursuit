"""The decentralized actor may see only its own local view (plus delivered messages, from M3).

These tests pin the boundary between ``ActorObs`` and training-only
``PrivilegedInfo`` so later milestones cannot leak global state by accident.
"""

from __future__ import annotations

import dataclasses
import inspect

import numpy as np

from dynabelief.config import EnvConfig
from dynabelief.envs.pursuit import ActorObs, PrivilegedInfo, PursuitEnv
from dynabelief.envs.vector import PursuitVecEnv
from dynabelief.policies.random_policy import RandomPolicy
from tests.helpers import bfs_action

ACTOR_FIELDS = {"obs", "agent_mask"}


def expected_local_view(global_state: np.ndarray, x: int, y: int, obs_range: int) -> np.ndarray:
    """Rebuild one pursuer's view from global state: window around (x, y), walls outside the map."""
    off = obs_range // 2
    channels, rows, cols = global_state.shape
    padded = np.zeros((channels, rows + 2 * off, cols + 2 * off), dtype=np.float32)
    padded[0] = 1.0  # out-of-bounds is wall
    padded[:, off : off + rows, off : off + cols] = global_state
    window = padded[:, y : y + obs_range, x : x + obs_range]
    return np.transpose(window, (1, 2, 0))  # [y, x, c]


def test_actor_obs_has_only_whitelisted_fields():
    assert {f.name for f in dataclasses.fields(ActorObs)} == ACTOR_FIELDS
    overlap = ACTOR_FIELDS & {f.name for f in dataclasses.fields(PrivilegedInfo)}
    assert not overlap


def test_actor_view_is_exactly_the_local_window():
    """Every actor observation equals the local crop of the global state, nothing more."""
    config = EnvConfig(max_cycles=30)
    env = PursuitEnv(config)
    rng = np.random.default_rng(0)
    actor, priv = env.reset(9)
    for _ in range(25):
        for agent, (px, py) in enumerate(priv.pursuer_pos):
            expected = expected_local_view(priv.global_state, int(px), int(py), config.obs_range)
            np.testing.assert_array_equal(actor.obs[agent], expected)
        result = env.step(rng.integers(0, 5, size=8))
        if result.done:
            break
        actor, priv = result.actor, result.privileged


def test_actor_view_is_local_window_through_captures():
    """Same check across capture steps, where evader bookkeeping changes (random play rarely
    captures, so a scripted pursuer walks onto frozen evaders)."""
    config = EnvConfig(
        n_pursuers=2, n_evaders=4, freeze_evaders=True, surround=False, n_catch=1, max_cycles=300
    )
    env = PursuitEnv(config)
    actor, priv = env.reset(2)
    sim_map = env._sim.map_matrix
    captures = 0
    for _ in range(300):
        for agent, (px, py) in enumerate(priv.pursuer_pos):
            expected = expected_local_view(priv.global_state, int(px), int(py), config.obs_range)
            np.testing.assert_array_equal(actor.obs[agent], expected)
        target = priv.evader_pos[np.flatnonzero(priv.evader_alive)[0]]
        chase = bfs_action(sim_map, tuple(priv.pursuer_pos[0]), tuple(target))
        result = env.step(np.array([chase, 4]))
        captures += result.captures
        actor, priv = result.actor, result.privileged
        if result.done:
            break
    assert captures == 4 and result.terminated
    for agent, (px, py) in enumerate(priv.pursuer_pos):
        expected = expected_local_view(priv.global_state, int(px), int(py), config.obs_range)
        np.testing.assert_array_equal(actor.obs[agent], expected)


def _observe(env: PursuitEnv) -> ActorObs:
    """Re-read actor observations through the real PettingZoo observe path."""
    return env._actor({a: env._env.aec_env.observe(a) for a in env.agent_ids})


def test_evaders_outside_view_do_not_affect_actor():
    """Teleporting an evader between two cells outside pursuer 0's view changes the
    privileged occupancy but not pursuer 0's actor input; moving it INTO view does."""
    config = EnvConfig(n_pursuers=2, n_evaders=1, freeze_evaders=True)
    env = PursuitEnv(config)
    env.reset(0)
    sim = env._sim
    off = config.obs_range // 2
    sim.pursuer_layer.set_position(0, 1, 1)
    sim.pursuer_layer.set_position(1, 14, 1)

    def put_evader(x, y):
        sim.evader_layer.set_position(0, x, y)
        sim.model_state[1] = sim.pursuer_layer.get_state_matrix()
        sim.model_state[2] = sim.evader_layer.get_state_matrix()
        return _observe(env), env._privileged()

    far_a, priv_a = put_evader(15, 15)
    far_b, priv_b = put_evader(15, 14)
    assert not np.array_equal(priv_a.occupancy, priv_b.occupancy)
    np.testing.assert_array_equal(far_a.obs[0], far_b.obs[0])

    near, _ = put_evader(1 + off, 1)  # inside pursuer 0's view: must be visible
    assert not np.array_equal(near.obs[0], far_a.obs[0])
    assert near.obs[0, off, off + off, 2] == 1  # [y, x, evader channel]


def test_actor_and_privileged_arrays_do_not_share_memory():
    vec = PursuitVecEnv(EnvConfig(max_cycles=5), num_envs=2, seed=0)
    actor, priv = vec.reset()
    single = PursuitEnv(EnvConfig(max_cycles=5))
    s_actor, s_priv = single.reset(0)
    for a_obj, p_obj in ((actor, priv), (s_actor, s_priv)):
        for a_field in dataclasses.fields(a_obj):
            for p_field in dataclasses.fields(p_obj):
                a, p = getattr(a_obj, a_field.name), getattr(p_obj, p_field.name)
                assert not np.shares_memory(a, p), (a_field.name, p_field.name)
    # Privileged arrays must not alias the simulator's internal buffers either, or a
    # stored PrivilegedInfo would silently change on the next step.
    sim = single._sim
    for p_field in dataclasses.fields(s_priv):
        p = getattr(s_priv, p_field.name)
        for internal in (
            sim.pursuer_layer.global_state,
            sim.evader_layer.global_state,
            sim.model_state,
            sim.map_matrix,
            sim.evaders_gone,
        ):
            assert not np.shares_memory(p, internal), p_field.name


def test_policy_interface_accepts_only_actor_obs():
    params = list(inspect.signature(RandomPolicy.act).parameters.values())[1:]
    assert len(params) == 1
    assert params[0].annotation in (ActorObs, "ActorObs")
