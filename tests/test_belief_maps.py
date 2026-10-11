"""M5: occupancy-map targets and calibration metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from dynabelief.beliefs.maps import (
    N_BINS,
    bce,
    brier,
    calibration_bins,
    expected_calibration_error,
    map_targets,
    own_view_mask,
    prior_logit,
)
from dynabelief.config import EnvConfig
from dynabelief.envs.pursuit import PursuitEnv


def test_coordinates_to_grid_targets():
    """Every alive evader at simulator (x, y) marks target[y, x]; obstacles are not free."""
    env = PursuitEnv(EnvConfig())
    _, priv = env.reset(4)
    occupancy, free = map_targets(priv.global_state)
    assert occupancy.shape == (16, 16) and free.shape == (16, 16)
    for x, y in priv.evader_pos[priv.evader_alive]:
        assert occupancy[y, x] == 1.0 and free[y, x]
    assert occupancy.sum() == len({tuple(p) for p in priv.evader_pos[priv.evader_alive]})
    assert not free[priv.global_state[0] > 0].any()
    assert (occupancy[~free] == 0).all()  # no evader stands on an obstacle


def test_batched_targets_keep_leading_dims():
    state = np.zeros((4, 2, 3, 5, 6), dtype=np.float32)  # [T, E, C, Y, X]
    state[1, 0, 2, 3, 4] = 2.0  # two evaders on (x=4, y=3)
    state[..., 0, 0, 0] = 1.0  # obstacle at (0, 0)
    occupancy, free = map_targets(state)
    assert occupancy.shape == (4, 2, 5, 6)
    assert occupancy[1, 0, 3, 4] == 1.0 and occupancy.sum() == 1
    assert not free[..., 0, 0].any() and free[..., 1, 1].all()


def test_prior_logit_matches_density():
    assert 1 / (1 + math.exp(-prior_logit(30, 193))) == pytest.approx(30 / 193)


def test_own_view_mask():
    mask = own_view_mask(np.array([[1, 1], [10, 12]]), (16, 16), obs_range=7)
    assert mask.shape == (2, 16, 16)
    assert mask[0].sum() == 5 * 5  # clipped at the grid border (x, y in 0..4)
    assert mask[1].sum() == 7 * 7
    assert mask[1, 12, 10] and mask[1, 15, 13] and not mask[1, 8, 10]


def test_metrics_on_toy_cases():
    y = np.array([1.0, 0.0, 0.0, 1.0])
    m = np.ones(4, dtype=bool)
    assert brier(y, y, m) == 0.0 and bce(y, y, m) < 1e-5
    assert brier(np.full(4, 0.5), y, m) == pytest.approx(0.25)
    assert bce(np.full(4, 0.5), y, m) == pytest.approx(math.log(2))
    masked = np.array([True, True, False, False])
    assert brier(np.array([1.0, 0.0, 9.0, 9.0]), y, masked) == 0.0


def test_ece_is_zero_when_calibrated_and_large_when_overconfident():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 200_000)
    y = (rng.uniform(0, 1, p.shape) < p).astype(float)
    m = np.ones_like(p, dtype=bool)
    bins = calibration_bins(p, y, m)
    assert bins.shape == (N_BINS, 3) and bins[:, 0].sum() == len(p)
    assert expected_calibration_error(bins) < 0.01
    overconfident = np.where(p > 0.5, 0.99, 0.01)
    assert expected_calibration_error(calibration_bins(overconfident, y, m)) > 0.2
    assert math.isnan(expected_calibration_error(np.zeros((N_BINS, 3))))
