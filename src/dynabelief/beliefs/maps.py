"""Occupancy-map beliefs (D-DIABL, M5): training-only targets and evaluation metrics.

Each agent predicts, for every grid cell, the probability that at least one evader stands
there (independent sigmoid per cell). The target is the TRUE occupancy of all evaders at
that step, not only the seen ones: a calibrated map should report the base rate where
nothing is known and spread probability around a last sighting, which is exactly the
uncertainty a single point cannot express. Obstacle cells never hold evaders and are
masked out of every loss and metric.

Grids use the package's image layout ``[y, x]`` (``occupancy[y, x]``).
"""

from __future__ import annotations

import numpy as np

N_BINS = 10  # calibration bins over [0, 1]


def map_targets(global_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``global_state [..., 3, Y, X]`` -> ``occupancy [..., Y, X]`` float32, ``free`` bool."""
    occupancy = (global_state[..., 2, :, :] > 0).astype(np.float32)
    free = global_state[..., 0, :, :] == 0
    return occupancy, free


def prior_logit(n_evaders: int, free_cells: int) -> float:
    """Logit of the expected occupancy of a free cell at the start of an episode."""
    p = min(max(n_evaders / max(free_cells, 1), 1e-4), 1 - 1e-4)
    return float(np.log(p / (1 - p)))


def own_view_mask(pursuer_pos: np.ndarray, grid_yx: tuple[int, int], obs_range: int) -> np.ndarray:
    """``[N, Y, X]`` bool: cells inside each pursuer's own square view."""
    Y, X = grid_yx
    off = obs_range // 2
    ys, xs = np.mgrid[0:Y, 0:X]
    px, py = pursuer_pos[:, 0, None, None], pursuer_pos[:, 1, None, None]
    return (np.abs(xs - px) <= off) & (np.abs(ys - py) <= off)


def bce(p: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    """Mean binary cross-entropy (nats per cell) over masked cells."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    values = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(values[mask].mean())


def brier(p: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    return float(((p - y) ** 2)[mask].mean())


def calibration_bins(p: np.ndarray, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """``[N_BINS, 3]``: per probability bin, (count, sum of predictions, sum of outcomes)."""
    pv, yv = p[mask].ravel(), y[mask].ravel()
    idx = np.minimum((pv * N_BINS).astype(int), N_BINS - 1)
    out = np.zeros((N_BINS, 3))
    np.add.at(out[:, 0], idx, 1.0)
    np.add.at(out[:, 1], idx, pv)
    np.add.at(out[:, 2], idx, yv)
    return out


def expected_calibration_error(bins: np.ndarray) -> float:
    """ECE = sum_b (n_b / n) |mean prediction_b - observed frequency_b|."""
    count = bins[:, 0]
    used = count > 0
    if not used.any():
        return float("nan")
    gap = np.abs(bins[used, 1] / count[used] - bins[used, 2] / count[used])
    return float((count[used] / count.sum() * gap).sum())
