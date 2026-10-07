from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

from dynabelief.envs.pursuit import ACTION_DELTAS

REPO_ROOT = Path(__file__).resolve().parents[1]


def bfs_action(map_matrix: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> int:
    """First action of a shortest obstacle-free path on a simulator ``[x, y]`` map (test helper)."""
    xs, ys = map_matrix.shape
    start, goal = tuple(map(int, start)), tuple(map(int, goal))
    if start == goal:
        return 4
    parent: dict[tuple[int, int], tuple[tuple[int, int], int]] = {}
    queue = deque([start])
    seen = {start}
    while queue:
        cell = queue.popleft()
        for action, (dx, dy) in enumerate(ACTION_DELTAS[:4]):
            nxt = (cell[0] + dx, cell[1] + dy)
            if not (0 <= nxt[0] < xs and 0 <= nxt[1] < ys) or nxt in seen:
                continue
            if map_matrix[nxt] == -1:
                continue
            seen.add(nxt)
            parent[nxt] = (cell, action)
            if nxt == goal:
                while parent[nxt][0] != start:
                    nxt = parent[nxt][0]
                return parent[nxt][1]
            queue.append(nxt)
    raise AssertionError(f"no path from {start} to {goal}")
