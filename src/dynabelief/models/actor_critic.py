"""Decentralized recurrent actor and centralized critic (CTDE) for MAPPO.

Tensor conventions
------------------
``B`` = environments, ``N`` = agents, ``L`` = time steps, ``H`` = hidden size.

* Actor inputs are ONLY local views ``[B, N, R, R, 3]`` and the actor's own
  recurrent state ``[B, N, H]``. M3 adds delivered messages here.
* Critic inputs are training-only: global state ``[B, 3, Y, X]``, pursuer
  positions ``[B, N, 2]`` (x, y) and the episode step ``[B]``.

Parameters are shared by all agents, so one network controls any ``N``.
Agent dims are kept intact (minibatches are taken over environments), so a
later communication layer can mix information across agents of one env.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical

from dynabelief.models.encoder import GlobalStateEncoder, LocalObsEncoder, init_layer


class RecurrentActor(nn.Module):
    """pi(a_i | o_i history): local CNN -> GRU -> action logits."""

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        n_actions: int,
        hidden_dim: int = 128,
        conv_channels: int = 32,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions
        self.encoder = LocalObsEncoder(obs_shape, conv_channels, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.policy_head = init_layer(nn.Linear(hidden_dim, n_actions), gain=0.01)
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            else:
                nn.init.zeros_(param)

    def initial_state(self, batch: int, n_agents: int, device: torch.device | str = "cpu"):
        return torch.zeros(batch, n_agents, self.hidden_dim, device=device)

    def _recur(self, features: torch.Tensor, hidden: torch.Tensor, start: torch.Tensor):
        """One GRU step; ``start`` [B] bool zeroes the hidden state of new episodes."""
        batch, n_agents, dim = features.shape
        hidden = hidden * (~start).to(hidden.dtype)[:, None, None]
        new = self.gru(features.reshape(-1, dim), hidden.reshape(-1, self.hidden_dim))
        return new.reshape(batch, n_agents, self.hidden_dim)

    def step(self, obs: torch.Tensor, hidden: torch.Tensor, episode_start: torch.Tensor):
        """Single step. Returns ``(logits [B, N, A], new_hidden [B, N, H])``."""
        new_hidden = self._recur(self.encoder(obs), hidden, episode_start)
        return self.policy_head(new_hidden), new_hidden

    def unroll(self, obs: torch.Tensor, hidden0: torch.Tensor, episode_start: torch.Tensor):
        """Sequence ``obs [L, B, N, ...]``, ``episode_start [L, B]`` -> logits ``[L, B, N, A]``.

        Identical to calling ``step`` L times; the CNN runs once over all steps.
        """
        features = self.encoder(obs)
        hidden, outputs = hidden0, []
        for t in range(obs.shape[0]):
            hidden = self._recur(features[t], hidden, episode_start[t])
            outputs.append(hidden)
        return self.policy_head(torch.stack(outputs))


class CentralCritic(nn.Module):
    """V(s, i): global state + a one-hot map of agent i's cell + time fraction -> value.

    TRAINING ONLY. Each agent gets its own value estimate from the shared state,
    distinguished by the extra position channel.
    """

    def __init__(
        self,
        state_shape: tuple[int, int, int],
        max_cycles: int,
        hidden_dim: int = 128,
        conv_channels: int = 32,
    ) -> None:
        super().__init__()
        channels, self.rows, self.cols = state_shape
        self.max_cycles = max_cycles
        self.encoder = GlobalStateEncoder(
            (channels + 1, self.rows, self.cols), conv_channels, hidden_dim
        )
        self.head = nn.Sequential(
            init_layer(nn.Linear(hidden_dim + 1, hidden_dim)),
            nn.ReLU(),
            init_layer(nn.Linear(hidden_dim, 1), gain=1.0),
        )

    def forward(
        self, global_state: torch.Tensor, pursuer_pos: torch.Tensor, step: torch.Tensor
    ) -> torch.Tensor:
        """``[B, C, Y, X]``, ``[B, N, 2]``, ``[B]`` -> values ``[B, N]``."""
        batch, n_agents = pursuer_pos.shape[:2]
        x, y = pursuer_pos[..., 0].long(), pursuer_pos[..., 1].long()
        own = torch.zeros(batch, n_agents, self.rows * self.cols, device=global_state.device)
        own.scatter_(2, (y * self.cols + x).unsqueeze(-1), 1.0)
        own = own.reshape(batch, n_agents, 1, self.rows, self.cols)
        per_agent = torch.cat(
            [global_state.unsqueeze(1).expand(-1, n_agents, -1, -1, -1), own], dim=2
        )
        features = self.encoder(per_agent)  # [B, N, H]
        time_frac = (step.to(features.dtype) / self.max_cycles)[:, None, None]
        features = torch.cat([features, time_frac.expand(-1, n_agents, 1)], dim=-1)
        return self.head(features).squeeze(-1)


class MAPPOPolicy(nn.Module):
    """Actor + critic container with sampling helpers."""

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        state_shape: tuple[int, int, int],
        n_actions: int,
        max_cycles: int,
        hidden_dim: int = 128,
        conv_channels: int = 32,
        critic_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.actor = RecurrentActor(obs_shape, n_actions, hidden_dim, conv_channels)
        self.critic = CentralCritic(state_shape, max_cycles, critic_hidden_dim, conv_channels)

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor,
        episode_start: torch.Tensor,
        deterministic: bool = False,
    ):
        """Returns ``(actions [B, N], log_probs [B, N], new_hidden [B, N, H])``."""
        logits, new_hidden = self.actor.step(obs, hidden, episode_start)
        dist = Categorical(logits=logits)
        actions = logits.argmax(-1) if deterministic else dist.sample()
        return actions, dist.log_prob(actions), new_hidden
