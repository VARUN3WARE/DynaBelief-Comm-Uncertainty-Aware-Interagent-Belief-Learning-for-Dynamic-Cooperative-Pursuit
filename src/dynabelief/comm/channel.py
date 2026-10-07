"""Deterministic packet-loss channel and message accounting.

Delivery masks use receiver-major layout ``mask[..., receiver, sender]`` so a
receiver's row lines up with an attention row over senders (TarMAC, M3).

Timing contract (1-step delay): messages produced at step ``t`` pass through
``sample_delivery`` once, and the delivered subset is what receivers may read
at step ``t+1``. A dropped message must have no effect on the receiver; the
model layer enforces that by masking, and this module supplies the mask.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MessageStats:
    """Cumulative counts of directed messages (one sender -> one receiver)."""

    sent: int = 0
    delivered: int = 0
    dropped: int = 0
    steps: int = 0
    message_dim: int = 0
    bytes_per_element: int = 4

    @property
    def bytes_sent(self) -> int:
        return self.sent * self.message_dim * self.bytes_per_element

    @property
    def bytes_delivered(self) -> int:
        return self.delivered * self.message_dim * self.bytes_per_element

    @property
    def drop_rate(self) -> float:
        return self.dropped / self.sent if self.sent else 0.0

    def as_dict(self) -> dict[str, float]:
        steps = max(self.steps, 1)
        return {
            "messages_sent": self.sent,
            "messages_delivered": self.delivered,
            "messages_dropped": self.dropped,
            "messages_sent_per_step": self.sent / steps,
            "messages_delivered_per_step": self.delivered / steps,
            "drop_rate": self.drop_rate,
            "bytes_sent": self.bytes_sent,
            "bytes_delivered": self.bytes_delivered,
        }


class PacketLossChannel:
    """Independent Bernoulli loss per directed message, from a dedicated seeded generator.

    Each step every alive agent broadcasts one message to every other alive
    agent (no self-messages). A message is dropped with probability
    ``packet_loss``. The generator is consumed identically whatever the alive
    mask, so the loss pattern for a given seed does not depend on agent deaths.
    """

    def __init__(
        self,
        n_agents: int,
        packet_loss: float,
        rng: np.random.Generator,
        message_dim: int = 16,
        bytes_per_element: int = 4,
    ) -> None:
        if n_agents < 1:
            raise ValueError(f"n_agents must be >= 1, got {n_agents}")
        if not 0.0 <= packet_loss <= 1.0:
            raise ValueError(f"packet_loss must be in [0, 1], got {packet_loss}")
        self.n_agents = n_agents
        self.packet_loss = float(packet_loss)
        self.rng = rng
        self.stats = MessageStats(message_dim=message_dim, bytes_per_element=bytes_per_element)
        self._off_diagonal = ~np.eye(n_agents, dtype=bool)

    def sample_delivery(self, alive: np.ndarray | None = None) -> np.ndarray:
        """Boolean delivery mask.

        Args:
            alive: ``[..., n_agents]`` bool mask of agents able to send/receive
                (leading dims, e.g. ``num_envs``, are batch dims). ``None``
                means all agents alive and no batch dims.

        Returns:
            ``[..., n_agents(receiver), n_agents(sender)]`` bool, True where the
            message was attempted *and* delivered.
        """
        if alive is None:
            alive = np.ones(self.n_agents, dtype=bool)
        alive = np.asarray(alive, dtype=bool)
        if alive.shape[-1] != self.n_agents:
            raise ValueError(f"alive has {alive.shape[-1]} agents, channel has {self.n_agents}")

        attempted = self._off_diagonal & alive[..., :, None] & alive[..., None, :]
        survives = self.rng.random(attempted.shape) >= self.packet_loss
        delivered = attempted & survives

        n_sent = int(attempted.sum())
        n_delivered = int(delivered.sum())
        self.stats.sent += n_sent
        self.stats.delivered += n_delivered
        self.stats.dropped += n_sent - n_delivered
        self.stats.steps += int(np.prod(alive.shape[:-1], dtype=np.int64))
        return delivered

    def reset_stats(self) -> MessageStats:
        """Return the accumulated stats and start a fresh counter."""
        old = self.stats
        self.stats = MessageStats(
            message_dim=old.message_dim, bytes_per_element=old.bytes_per_element
        )
        return old
