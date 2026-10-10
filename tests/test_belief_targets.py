"""Point-belief labels: alignment with simulator coordinates and the 'found' rule."""

from __future__ import annotations

import numpy as np

from dynabelief.beliefs.targets import point_targets, visible_evaders
from dynabelief.config import EnvConfig
from dynabelief.envs.pursuit import PursuitEnv


def test_visibility_window_and_dead_evaders():
    pursuers = np.array([[5, 5]])
    evaders = np.array([[8, 5], [9, 5], [5, 2], [-1, -1]])
    alive = np.array([True, True, True, False])
    vis = visible_evaders(pursuers, evaders, alive, obs_range=7)  # half-width 3
    assert vis.tolist() == [[True, False, True, False]]


def test_nearest_team_visible_evader_and_found_rule():
    # 5x5 views (half-width 2). Agent 0 at (1,1) sees evader 0 at (2,2); agent 1 at
    # (14,14) sees evader 1 at (14,13); evader 2 at (4,1) is 3 cells from agent 0: unseen.
    pursuers = np.array([[1, 1], [14, 14]])
    evaders = np.array([[2, 2], [14, 13], [4, 1]])
    alive = np.array([True, True, True])
    seen = visible_evaders(pursuers, evaders, alive, 5).any(axis=0)
    assert seen.tolist() == [True, True, False]
    target, valid = point_targets(pursuers, evaders, alive, obs_range=5, grid_xy=(16, 16))
    np.testing.assert_allclose(target[0], np.array([2, 2]) / 15)
    np.testing.assert_allclose(target[1], np.array([14, 13]) / 15)
    assert valid.all()

    # Only evader 0 is seen (by agent 0), so agent 1 must target it although the unseen
    # evader 1 is much closer to agent 1: beliefs are supervised only on found targets.
    evaders = np.array([[2, 2], [10, 14]])
    target, valid = point_targets(pursuers, evaders, np.array([True, True]), 3, (16, 16))
    np.testing.assert_allclose(target[1], np.array([2, 2]) / 15)
    assert valid.all()


def test_no_target_when_team_sees_nothing():
    target, valid = point_targets(np.array([[0, 0], [1, 0]]), np.array([[15, 15]]),
                                  np.array([True]), 7, (16, 16))  # fmt: skip
    assert not valid.any() and (target == 0).all()


def test_batched_shapes_and_ties_take_lowest_id():
    pursuers = np.array([[[5, 5]], [[0, 0]]])  # [E=2, N=1, 2]
    evaders = np.array([[[6, 5], [4, 5]], [[1, 0], [0, 1]]])  # equidistant pairs
    alive = np.ones((2, 2), dtype=bool)
    target, valid = point_targets(pursuers, evaders, alive, 7, (16, 16))
    assert target.shape == (2, 1, 2) and valid.shape == (2, 1)
    np.testing.assert_allclose(target[:, 0], np.array([[6, 5], [1, 0]]) / 15)


def test_targets_align_with_the_simulator_and_local_views():
    """The labelled evader really sits where the simulator says, and it shows up in the
    observer's local view at the matching [y, x] offset."""
    config = EnvConfig()
    env = PursuitEnv(config)
    rng = np.random.default_rng(0)
    actor, priv = env.reset(1)
    off = config.obs_range // 2
    checked = 0
    for _ in range(30):
        target, valid = point_targets(priv.pursuer_pos, priv.evader_pos, priv.evader_alive,
                                      config.obs_range, (config.x_size, config.y_size))  # fmt: skip
        for i in np.flatnonzero(valid):
            tx, ty = np.rint(target[i] * 15).astype(int)
            assert priv.occupancy[ty, tx] == 1.0  # an evader stands on the target cell
            observers = [
                j for j, (px, py) in enumerate(priv.pursuer_pos)
                if abs(tx - px) <= off and abs(ty - py) <= off
            ]  # fmt: skip
            assert observers, "target must be in some pursuer's view"
            px, py = priv.pursuer_pos[observers[0]]
            assert actor.obs[observers[0], ty - py + off, tx - px + off, 2] >= 1
            checked += 1
        result = env.step(rng.integers(0, 5, size=8))
        actor, priv = result.actor, result.privileged
    assert checked > 50
