"""TarMAC-style targeted communication (Das et al., 2019) over a lossy, delayed channel.

Timing (``comm.delay_steps = 1``): every agent j broadcasts a message derived from its
hidden state ``h_j(t-1)``: a key ``k_j`` (signature) and a value ``v_j`` (content).
At step t, receiver i forms a query ``q_i`` from its own ``h_i(t-1)`` and attends over the
messages actually DELIVERED to it::

    alpha_ij = softmax_j( q_i . k_j / sqrt(d_k) )   over delivered j only
    c_i(t)   = sum_j alpha_ij v_j                   (0 if nothing was delivered)

``c_i(t)`` is an extra GRU input next to the local observation features. The query
never leaves the receiver; only ``(k_j, v_j)`` count as transmitted.

Guarantees (tested): an undelivered message has no effect on the receiver's output and
no gradient flows through it; a receiver with no deliveries gets ``c = 0``; gradients
from a receiver's loss reach the sender's encoder through ``k`` and ``v``.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from dynabelief.models.encoder import init_layer


class TarMACComm(nn.Module):
    def __init__(self, hidden_dim: int, key_dim: int, value_dim: int) -> None:
        super().__init__()
        self.key_dim, self.value_dim = key_dim, value_dim
        self.query = init_layer(nn.Linear(hidden_dim, key_dim), gain=1.0)
        self.key = init_layer(nn.Linear(hidden_dim, key_dim), gain=1.0)
        self.value = init_layer(nn.Linear(hidden_dim, value_dim), gain=1.0)

    def forward(
        self, hidden: torch.Tensor, delivery: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``hidden [B, N, H]`` (previous step), ``delivery [B, N_recv, N_send]`` bool.

        Returns ``(context [B, N, value_dim], attention [B, N_recv, N_send])``; attention
        rows sum to 1 when at least one message was delivered, else they are all 0.
        """
        if delivery.shape != hidden.shape[:2] + hidden.shape[1:2]:
            raise ValueError(
                f"delivery {tuple(delivery.shape)} does not match {tuple(hidden.shape)}"
            )
        q, k, v = self.query(hidden), self.key(hidden), self.value(hidden)
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.key_dim)  # [B, recv, send]
        any_delivered = delivery.any(dim=-1, keepdim=True)
        # Masked entries get -inf (zero weight, zero gradient); rows with no delivery
        # are set to 0 before the softmax so they stay finite, then zeroed by the mask.
        scores = scores.masked_fill(~delivery, float("-inf"))
        scores = torch.where(any_delivered, scores, torch.zeros_like(scores))
        attention = torch.softmax(scores, dim=-1) * delivery.to(scores.dtype)
        return attention @ v, attention

    @staticmethod
    def attention_entropy(attention: torch.Tensor, delivery: torch.Tensor) -> torch.Tensor:
        """Mean entropy (nats) of the attention rows that received at least one message."""
        rows = delivery.any(dim=-1)
        if not rows.any():
            return attention.new_zeros(())
        p = attention[rows]
        return -(p * torch.log(p.clamp_min(1e-12))).sum(-1).mean()
