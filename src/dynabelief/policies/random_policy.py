"""Uniform random policy: the M1 smoke-test baseline. It learns nothing."""

from __future__ import annotations

import numpy as np

from dynabelief.envs.pursuit import ActorObs


class RandomPolicy:
    """Samples actions uniformly from a dedicated seeded generator.

    Like every actor, it accepts only ``ActorObs``. It never sees training-only
    ``PrivilegedInfo``.
    """

    def __init__(self, n_actions: int, rng: np.random.Generator) -> None:
        if n_actions < 1:
            raise ValueError(f"n_actions must be >= 1, got {n_actions}")
        self.n_actions = n_actions
        self.rng = rng

    def act(self, actor: ActorObs) -> np.ndarray:
        """Actions with the shape of ``actor.agent_mask``; masked-out agents get action 0."""
        actions = self.rng.integers(0, self.n_actions, size=actor.agent_mask.shape, dtype=np.int64)
        return np.where(actor.agent_mask, actions, 0)
