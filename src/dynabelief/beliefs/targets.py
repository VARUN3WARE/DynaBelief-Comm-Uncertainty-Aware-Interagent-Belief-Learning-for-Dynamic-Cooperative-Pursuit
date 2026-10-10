"""Training-only belief labels computed from ``PrivilegedInfo`` (never actor inputs).

Point belief (M4, adapting DIABL's target-coordinate belief to moving evaders):
agent i should know the position of the nearest evader that the TEAM currently sees.
The paper penalizes a belief only once its target has been found (it would otherwise
be guesswork); here "found" means: inside at least one pursuer's view this step. If
the team sees no evader, agent i has no target and the belief is not penalized.

Coordinates are absolute and normalized: ``(x / (X-1), y / (Y-1))`` in [0, 1], the
same frame as ``ActorObs.own_pos``. Ties in distance go to the lowest evader id.
"""

from __future__ import annotations

import numpy as np


def visible_evaders(
    pursuer_pos: np.ndarray, evader_pos: np.ndarray, evader_alive: np.ndarray, obs_range: int
) -> np.ndarray:
    """``[..., N, M]`` bool: evader m is alive and inside pursuer n's square view."""
    off = obs_range // 2
    delta = np.abs(evader_pos[..., None, :, :] - pursuer_pos[..., :, None, :])  # [..., N, M, 2]
    return (delta <= off).all(axis=-1) & evader_alive[..., None, :]


def point_targets(
    pursuer_pos: np.ndarray,
    evader_pos: np.ndarray,
    evader_alive: np.ndarray,
    obs_range: int,
    grid_xy: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest team-visible evader per agent.

    Args:
        pursuer_pos: ``[..., N, 2]`` int (x, y); evader_pos: ``[..., M, 2]`` int (x, y),
            ``-1`` for caught evaders; evader_alive: ``[..., M]`` bool.
        grid_xy: ``(x_size, y_size)``.

    Returns:
        ``target [..., N, 2]`` float32 in [0, 1] (0 where invalid) and
        ``valid [..., N]`` bool.
    """
    team_seen = visible_evaders(pursuer_pos, evader_pos, evader_alive, obs_range).any(axis=-2)
    dist = np.abs(evader_pos[..., None, :, :] - pursuer_pos[..., :, None, :]).sum(-1)  # [.., N, M]
    dist = np.where(team_seen[..., None, :], dist, np.iinfo(np.int64).max)
    nearest = dist.argmin(axis=-1)  # [..., N]; argmin takes the lowest id on ties
    valid = np.broadcast_to(team_seen.any(axis=-1, keepdims=True), nearest.shape)
    chosen = np.take_along_axis(evader_pos, nearest[..., None].repeat(2, axis=-1), axis=-2)
    scale = np.array([grid_xy[0] - 1, grid_xy[1] - 1], dtype=np.float32)
    target = np.where(valid[..., None], chosen / scale, 0.0).astype(np.float32)
    return target, valid.copy()
