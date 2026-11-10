"""Training-only evader Bayes filter: the knowable-target teacher of arms C′ and FD."""

from __future__ import annotations

import time

import numpy as np

from dynabelief.beliefs.filter import EvaderFilter
from dynabelief.config import EnvConfig
from dynabelief.envs.pursuit import PursuitEnv

Y = X = 8


def _filter(num_envs=1, n_evaders=1, obs_range=3):
    return EvaderFilter(num_envs, n_evaders, (Y, X), obs_range)


def _free(num_envs=1):
    return np.ones((num_envs, Y, X), dtype=bool)


def _far_pursuer(num_envs=1):
    # one pursuer in the corner opposite the cells under test, so nothing near them is viewed
    return np.full((num_envs, 1, 2), 7, dtype=np.int64)


def test_predict_matches_pettingzoo_kernel():
    f = _filter()
    f.reset(np.array([True]), _free())
    f.P[:] = 0.0
    f.P[0, 0, 3, 3] = 1.0  # interior cell [y=3, x=3]
    f.fresh[:] = False
    f.predict(_free())
    p = f.P[0, 0]
    for y, x in [(3, 3), (2, 3), (4, 3), (3, 2), (3, 4)]:
        assert np.isclose(p[y, x], 0.2)
    assert np.isclose(p.sum(), 1.0)

    f.P[:] = 0.0
    f.P[0, 0, 0, 0] = 1.0  # corner: two of the four moves leave the grid and stay
    f.predict(_free())
    p = f.P[0, 0]
    assert np.isclose(p[0, 0], 0.6) and np.isclose(p[1, 0], 0.2) and np.isclose(p[0, 1], 0.2)

    free = _free()
    free[0, 3, 4] = False  # obstacle to the right: that move also stays
    f.P[:] = 0.0
    f.P[0, 0, 3, 3] = 1.0
    f.predict(free)
    assert np.isclose(f.P[0, 0, 3, 3], 0.4) and f.P[0, 0, 3, 4] == 0.0


def test_visible_evader_is_delta():
    f = _filter(obs_range=3)
    f.reset(np.array([True]), _free())
    pursuers = np.array([[[2, 2]]])
    evaders = np.array([[[3, 1]]])  # (x, y) inside the 3x3 view of (2, 2)
    f.update(pursuers, evaders, np.array([[True]]), _free())
    assert f.P[0, 0, 1, 3] == 1.0 and np.isclose(f.P[0, 0].sum(), 1.0)
    assert f.tracked().tolist() == [[True]]


def test_viewed_cells_zero_for_unseen_evader():
    f = _filter(obs_range=3)
    f.reset(np.array([True]), _free())
    pursuers = np.array([[[2, 2]]])
    evaders = np.array([[[7, 7]]])  # far away, not visible
    f.update(pursuers, evaders, np.array([[True]]), _free())
    p = f.P[0, 0]
    assert p[1:4, 1:4].sum() == 0.0  # the viewed 3x3 block
    assert np.isclose(p.sum(), 1.0) and np.isclose(p[7, 7], 1.0 / (Y * X - 9))
    assert f.tracked().tolist() == [[False]]


def test_dead_evader_has_no_mass():
    f = _filter()
    f.reset(np.array([True]), _free())
    f.update(_far_pursuer(), np.array([[[-1, -1]]]), np.array([[False]]), _free())
    assert f.P[0, 0].sum() == 0.0 and f.occupancy()[0].max() == 0.0


def test_filter_resets_on_episode_start():
    f = _filter(num_envs=2, obs_range=3)
    f.reset(np.array([True, True]), _free(2))
    pursuers = np.array([[[2, 2]], [[2, 2]]])
    evaders = np.array([[[3, 1]], [[3, 1]]])
    f.update(pursuers, evaders, np.array([[True], [True]]), _free(2))
    before = f.P[1].copy()
    f.reset(np.array([True, False]), _free(2))
    assert np.allclose(f.P[0, 0], 1.0 / (Y * X)) and not f.tracked()[0].any()
    assert np.array_equal(f.P[1], before) and f.tracked()[1].all()
    # the reset env skips the motion step on its first update (no motion before step 0)
    f.update(_far_pursuer(2), np.array([[[0, 0]], [[0, 0]]]), np.array([[True], [True]]),
             _free(2))  # fmt: skip
    assert not f.fresh.any()


def test_occupancy_bounds_and_obstacles():
    f = _filter(n_evaders=3)
    free = _free()
    free[0, 0, :] = False
    f.reset(np.array([True]), free)
    f.update(_far_pursuer(), np.full((1, 3, 2), -1), np.array([[True, True, True]]), free)
    occ = f.occupancy()
    assert occ.min() >= 0.0 and occ.max() <= 1.0 and occ[0, 0, :].max() == 0.0
    # independent evaders: occupancy = 1 - prod(1 - P_m)
    expected = 1 - np.prod(1 - f.P[0], axis=0)
    assert np.allclose(occ[0], expected)


def test_true_cell_never_zero_on_real_rollouts():
    """The kernel must match PettingZoo's evader dynamics exactly, or mass vanishes."""
    cfg = EnvConfig(x_size=16, y_size=16, n_pursuers=8, n_evaders=30, obs_range=7,
                    max_cycles=200)  # fmt: skip
    env = PursuitEnv(cfg)
    rng = np.random.default_rng(0)
    try:
        for episode in range(3):
            actor, priv = env.reset(seed=1000 + episode)
            free = (priv.global_state[0] == 0)[None]
            f = EvaderFilter(1, cfg.n_evaders, (cfg.y_size, cfg.x_size), cfg.obs_range)
            f.reset(np.array([True]), free)
            for _ in range(200):
                f.update(priv.pursuer_pos[None], priv.evader_pos[None],
                         priv.evader_alive[None], free)  # fmt: skip
                alive = np.flatnonzero(priv.evader_alive)
                xs, ys = priv.evader_pos[alive, 0], priv.evader_pos[alive, 1]
                assert (f.P[0, alive, ys, xs] > 0).all()
                result = env.step(rng.integers(env.n_actions, size=cfg.n_pursuers))
                if result.done:
                    break
                priv = result.privileged
            assert f.resets == 0
    finally:
        env.close()


def test_update_is_fast_for_training_batch():
    f = EvaderFilter(32, 30, (16, 16), 7)
    free = np.ones((32, 16, 16), dtype=bool)
    f.reset(np.ones(32, dtype=bool), free)
    rng = np.random.default_rng(0)
    pursuers = rng.integers(16, size=(32, 8, 2))
    evaders = rng.integers(16, size=(32, 30, 2))
    alive = np.ones((32, 30), dtype=bool)
    f.update(pursuers, evaders, alive, free)
    start = time.perf_counter()
    for _ in range(20):
        f.update(pursuers, evaders, alive, free)
        f.occupancy()
    assert (time.perf_counter() - start) / 20 < 0.01  # well below one vector-env step
