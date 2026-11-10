"""Training-only Bayes filter over evader positions: what the TEAM could know.

Arms C′ and FD supervise beliefs with this filter's posterior instead of the true positions
of evaders nobody has seen (D-DIABL's target). The filter uses only:

* the union of the pursuers' square views at every step so far (cells the team saw),
* PettingZoo's evader dynamics: each step an evader picks one of {left, right, up, down,
  stay} uniformly at random; a move off the grid or into an obstacle leaves it in place,
* which evaders are still alive (captures happen on pursuer cells, so the team knows).

Per evader m it keeps ``P_m``, a distribution over cells. Visible -> delta at its cell;
not visible -> one motion step, then zero on the cells the team views, renormalized; dead
-> zero. Evader identity is used (the simulator knows which evader is which); this is a
training-time teacher, never an actor input. Grids use the package's ``[y, x]`` layout and
positions are ``(x, y)``, as in ``PrivilegedInfo``.
"""

from __future__ import annotations

import numpy as np

from dynabelief.beliefs.targets import visible_evaders

_MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))  # (dy, dx); the fifth action is "stay"


def team_view_mask(pursuer_pos: np.ndarray, grid_yx: tuple[int, int], obs_range: int):
    """``pursuer_pos [E, N, 2] (x, y)`` -> ``[E, Y, X]`` bool: cells inside any pursuer's view."""
    Y, X = grid_yx
    off = obs_range // 2
    ys, xs = np.arange(Y), np.arange(X)
    in_y = np.abs(ys[None, None, :] - pursuer_pos[..., 1:2]) <= off  # [E, N, Y]
    in_x = np.abs(xs[None, None, :] - pursuer_pos[..., 0:1]) <= off  # [E, N, X]
    return (in_y[..., :, None] & in_x[..., None, :]).any(axis=1)


def _shift(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """``out[..., y, x] = a[..., y - dy, x - dx]`` with zeros shifted in."""
    out = np.zeros_like(a)
    Y, X = a.shape[-2:]
    ys_dst, ys_src = slice(max(dy, 0), Y + min(dy, 0)), slice(max(-dy, 0), Y + min(-dy, 0))
    xs_dst, xs_src = slice(max(dx, 0), X + min(dx, 0)), slice(max(-dx, 0), X + min(-dx, 0))
    out[..., ys_dst, xs_dst] = a[..., ys_src, xs_src]
    return out


class EvaderFilter:
    def __init__(self, num_envs: int, n_evaders: int, grid_yx: tuple[int, int], obs_range: int):
        self.grid_yx = tuple(grid_yx)
        self.obs_range = obs_range
        self.P = np.zeros((num_envs, n_evaders, *grid_yx), dtype=np.float32)
        self.fresh = np.ones(num_envs, dtype=bool)  # skip motion before an episode's 1st update
        self._tracked = np.zeros((num_envs, n_evaders), dtype=bool)
        self.resets = 0  # times an unseen evader's mass vanished (0 if the model is exact)

    def reset(self, envs: np.ndarray, free: np.ndarray) -> None:
        """Start new episodes in ``envs [E]``: uniform over free cells, nothing tracked."""
        envs = np.asarray(envs, dtype=bool)
        if not envs.any():
            return
        uniform = free[envs].astype(np.float32)
        uniform /= uniform.sum(axis=(-2, -1), keepdims=True)
        self.P[envs] = uniform[:, None]
        self._tracked[envs] = False
        self.fresh[envs] = True

    def predict(self, free: np.ndarray) -> None:
        """One evader motion step for every env that is not at an episode start."""
        envs = ~self.fresh
        if not envs.any():
            return
        P, free_e = self.P[envs], free[envs][:, None]  # [e, M, Y, X], [e, 1, Y, X]
        out = 0.2 * P  # "stay"
        for dy, dx in _MOVES:
            # a cell can move by (dy, dx) when the destination is on the grid and free
            ok = _shift(free_e, -dy, -dx)
            out += 0.2 * _shift(P * ok, dy, dx) + 0.2 * P * ~ok
        self.P[envs] = out

    def update(self, pursuer_pos, evader_pos, evader_alive, free) -> None:
        """Advance one step: motion (unless fresh), then condition on the team's views."""
        self.predict(free)
        self.fresh[:] = False
        E, M = self.P.shape[:2]
        view = team_view_mask(pursuer_pos, self.grid_yx, self.obs_range)  # [E, Y, X]
        visible = visible_evaders(pursuer_pos, evader_pos, evader_alive, self.obs_range).any(-2)
        alive = np.asarray(evader_alive, dtype=bool)

        P = self.P
        P *= ~view[:, None]  # an unseen evader is not on a viewed cell
        mass = P.sum(axis=(-2, -1))  # [E, M]
        lost = alive & ~visible & (mass <= 0)
        if lost.any():
            self.resets += int(lost.sum())
            fallback = (free & ~view).astype(np.float32)
            for e, m in zip(*np.nonzero(lost), strict=True):
                P[e, m] = fallback[e]
            mass = P.sum(axis=(-2, -1))
        P /= np.where(mass > 0, mass, 1.0)[..., None, None]

        e_idx, m_idx = np.nonzero(visible)
        P[e_idx, m_idx] = 0.0
        xs, ys = evader_pos[e_idx, m_idx, 0], evader_pos[e_idx, m_idx, 1]
        P[e_idx, m_idx, ys, xs] = 1.0
        P[~alive] = 0.0

        self._tracked |= visible
        self._tracked &= alive

    def per_evader(self) -> np.ndarray:
        """``[E, M, Y, X]``: each alive evader's posterior (sums to 1); dead evaders are 0."""
        return self.P

    def occupancy(self) -> np.ndarray:
        """``[E, Y, X]`` P(at least one evader on the cell) = 1 - prod_m (1 - P_m)."""
        return (1.0 - np.prod(1.0 - self.P, axis=1)).astype(np.float32)

    def tracked(self) -> np.ndarray:
        """``[E, M]`` alive evaders the team has seen at least once this episode."""
        return self._tracked.copy()
