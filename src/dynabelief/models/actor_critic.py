"""Decentralized recurrent actor and centralized critic (CTDE) for MAPPO.

Tensor conventions
------------------
``B`` = environments, ``N`` = agents, ``L`` = time steps, ``H`` = hidden size.

* Actor inputs are ONLY local views ``[B, N, R, R, 3]``, the actor's own recurrent
  state ``[B, N, H]`` and, with communication (M3), the channel's delivery mask
  ``[B, N_recv, N_send]``: receivers read only messages that were actually delivered.
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

from dynabelief.models.comm import TarMACComm
from dynabelief.models.encoder import GlobalStateEncoder, LocalObsEncoder, init_layer


class RecurrentActor(nn.Module):
    """pi(a_i | o_i history, delivered messages): local CNN [+ TarMAC context] -> GRU -> logits.

    With ``comm_key_dim``/``comm_value_dim`` set, each step attends over the messages
    delivered from the previous step (see ``models/comm.py``). Without them the actor
    is the M2 No-Comm actor and agents are processed independently.
    """

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        n_actions: int,
        hidden_dim: int = 128,
        conv_channels: int = 32,
        comm_key_dim: int | None = None,
        comm_value_dim: int | None = None,
        use_own_position: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions
        self.use_own_position = use_own_position
        self.encoder = LocalObsEncoder(obs_shape, conv_channels, hidden_dim)
        self.comm = (
            TarMACComm(hidden_dim, comm_key_dim, comm_value_dim)
            if comm_key_dim and comm_value_dim
            else None
        )
        gru_input = hidden_dim + (comm_value_dim if self.comm is not None else 0)
        gru_input += 2 if use_own_position else 0
        self.gru = nn.GRUCell(gru_input, hidden_dim)
        self.last_attention: torch.Tensor | None = None  # diagnostics of the latest step
        self.policy_head = init_layer(nn.Linear(hidden_dim, n_actions), gain=0.01)
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            else:
                nn.init.zeros_(param)

    def initial_state(self, batch: int, n_agents: int, device: torch.device | str = "cpu"):
        return torch.zeros(batch, n_agents, self.hidden_dim, device=device)

    def _recur(
        self,
        features: torch.Tensor,
        hidden: torch.Tensor,
        start: torch.Tensor,
        delivery: torch.Tensor | None,
        own_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One GRU step; ``start`` [B] bool zeroes the hidden state of new episodes.

        Messages are built from the (reset) previous hidden state. At an episode start
        no message from the previous episode may be read, so delivery is masked there.
        """
        batch, n_agents, dim = features.shape
        hidden = hidden * (~start).to(hidden.dtype)[:, None, None]
        if self.comm is not None:
            if delivery is None:
                raise ValueError("this actor communicates: pass the delivery mask")
            delivery = delivery & ~start[:, None, None]
            context, self.last_attention = self.comm(hidden, delivery)
            features = torch.cat([features, context], dim=-1)
        if self.use_own_position:
            if own_pos is None:
                raise ValueError("this actor uses its own position: pass own_pos")
            features = torch.cat([features, own_pos.to(features.dtype)], dim=-1)
        dim = features.shape[-1]
        new = self.gru(features.reshape(-1, dim), hidden.reshape(-1, self.hidden_dim))
        return new.reshape(batch, n_agents, self.hidden_dim)

    def step(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor,
        episode_start: torch.Tensor,
        delivery: torch.Tensor | None = None,
        own_pos: torch.Tensor | None = None,
    ):
        """Single step. Returns ``(logits [B, N, A], new_hidden [B, N, H])``."""
        new_hidden = self._recur(self.encoder(obs), hidden, episode_start, delivery, own_pos)
        return self.policy_head(new_hidden), new_hidden

    def unroll(
        self,
        obs: torch.Tensor,
        hidden0: torch.Tensor,
        episode_start: torch.Tensor,
        delivery: torch.Tensor | None = None,
        own_pos: torch.Tensor | None = None,
        return_hidden: bool = False,
    ):
        """Sequence ``obs [L, B, N, ...]``, ``episode_start [L, B]``, ``delivery
        [L, B, N, N]``, ``own_pos [L, B, N, 2]`` -> logits ``[L, B, N, A]`` (and the
        hidden states ``[L, B, N, H]`` with ``return_hidden``, e.g. for belief heads).

        Identical to calling ``step`` L times; the CNN runs once over all steps. Because
        messages come from the previous hidden state, gradients flow across agents and
        time within the sequence (receiver -> message -> sender).
        """
        features = self.encoder(obs)
        hidden, outputs = hidden0, []
        for t in range(obs.shape[0]):
            step_delivery = None if delivery is None else delivery[t]
            step_pos = None if own_pos is None else own_pos[t]
            hidden = self._recur(features[t], hidden, episode_start[t], step_delivery, step_pos)
            outputs.append(hidden)
        hiddens = torch.stack(outputs)
        logits = self.policy_head(hiddens)
        return (logits, hiddens) if return_hidden else logits


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
        comm_key_dim: int | None = None,
        comm_value_dim: int | None = None,
        use_own_position: bool = False,
    ) -> None:
        super().__init__()
        self.actor = RecurrentActor(
            obs_shape, n_actions, hidden_dim, conv_channels, comm_key_dim, comm_value_dim,
            use_own_position,
        )  # fmt: skip
        self.critic = CentralCritic(state_shape, max_cycles, critic_hidden_dim, conv_channels)

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor,
        episode_start: torch.Tensor,
        deterministic: bool = False,
        *,
        delivery: torch.Tensor | None = None,
        own_pos: torch.Tensor | None = None,
    ):
        """Returns ``(actions [B, N], log_probs [B, N], new_hidden [B, N, H])``."""
        logits, new_hidden = self.actor.step(obs, hidden, episode_start, delivery, own_pos)
        dist = Categorical(logits=logits)
        actions = logits.argmax(-1) if deterministic else dist.sample()
        return actions, dist.log_prob(actions), new_hidden
