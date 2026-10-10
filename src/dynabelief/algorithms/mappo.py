"""MAPPO: clipped PPO with a centralized critic and a recurrent decentralized actor.

Loss (all terms logged weighted and unweighted)::

    L_actor  = L_policy - entropy_coef * H[pi]
    L_critic = value_coef * L_value

Actor and critic have separate parameters and optimizers, so the two losses
never share gradients. Means are taken over valid (agent_mask) entries only.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from dynabelief.algorithms.gae import build_next_values, compute_gae
from dynabelief.config import PPOConfig
from dynabelief.models.actor_critic import MAPPOPolicy
from dynabelief.training.buffer import Minibatch, RolloutBuffer
from dynabelief.utils.value_norm import ValueNorm


@dataclass
class LossTerms:
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor
    approx_kl: torch.Tensor
    clip_fraction: torch.Tensor


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(x.dtype)
    return (x * mask).sum() / mask.sum().clamp(min=1.0)


def ppo_loss_terms(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    entropy: torch.Tensor,
    new_values: torch.Tensor,
    old_values: torch.Tensor,
    value_targets: torch.Tensor,
    mask: torch.Tensor,
    clip_eps: float,
    value_clip_eps: float,
    huber_delta: float,
) -> LossTerms:
    """Clipped surrogate + (optionally clipped) value loss. Values in one consistent scale."""
    log_ratio = new_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    surrogate = torch.min(
        ratio * advantages, torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    )

    def error(pred: torch.Tensor) -> torch.Tensor:
        if huber_delta > 0:
            return F.huber_loss(pred, value_targets, reduction="none", delta=huber_delta)
        return 0.5 * (pred - value_targets).pow(2)

    value_error = error(new_values)
    if value_clip_eps > 0:
        clipped = old_values + torch.clamp(new_values - old_values, -value_clip_eps, value_clip_eps)
        value_error = torch.max(value_error, error(clipped))

    with torch.no_grad():
        approx_kl = masked_mean((ratio - 1.0) - log_ratio, mask)
        clip_fraction = masked_mean(((ratio - 1.0).abs() > clip_eps).float(), mask)
    return LossTerms(
        policy_loss=-masked_mean(surrogate, mask),
        value_loss=masked_mean(value_error, mask),
        entropy=masked_mean(entropy, mask),
        approx_kl=approx_kl,
        clip_fraction=clip_fraction,
    )


def normalize_advantages(advantages: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Standardize with statistics of valid entries only; invalid entries are left as-is
    (they are masked out of every loss)."""
    valid = advantages[mask]
    return (advantages - valid.mean()) / (valid.std() + 1e-8)


def explained_variance(predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    pred, tgt = predicted[mask], target[mask]
    var = tgt.var(unbiased=False)
    if tgt.numel() < 2 or var <= 0:
        return float("nan")
    return float(1.0 - (tgt - pred).var(unbiased=False) / var)


class MAPPO:
    def __init__(
        self,
        policy: MAPPOPolicy,
        config: PPOConfig,
        device: torch.device | str,
        generator: torch.Generator,
    ) -> None:
        self.policy = policy
        self.config = config
        self.device = torch.device(device)
        self.generator = generator  # CPU generator for minibatch shuffling
        self.actor_optimizer = torch.optim.Adam(
            policy.actor.parameters(), lr=config.lr_actor, eps=config.adam_eps
        )
        self.critic_optimizer = torch.optim.Adam(
            policy.critic.parameters(), lr=config.lr_critic, eps=config.adam_eps
        )
        self.value_norm = ValueNorm().to(self.device) if config.use_value_norm else None

    # ------------------------------------------------------------ state io
    def state_dict(self) -> dict:
        return {
            "policy": self.policy.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "value_norm": self.value_norm.state_dict() if self.value_norm is not None else None,
        }

    def load_state_dict(self, state: dict) -> None:
        self.policy.load_state_dict(state["policy"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        if (state["value_norm"] is None) != (self.value_norm is None):
            raise ValueError("checkpoint and config disagree on ppo.use_value_norm")
        if self.value_norm is not None:
            self.value_norm.load_state_dict(state["value_norm"])

    # ---------------------------------------------------------------- values
    def _to_return_units(self, raw: torch.Tensor) -> torch.Tensor:
        return self.value_norm.denormalize(raw) if self.value_norm is not None else raw

    def _to_critic_units(self, values: torch.Tensor) -> torch.Tensor:
        return self.value_norm.normalize(values) if self.value_norm is not None else values

    @torch.no_grad()
    def values(
        self, global_state: torch.Tensor, pursuer_pos: torch.Tensor, step: torch.Tensor
    ) -> torch.Tensor:
        """Critic values in return units, ``[B, N]``."""
        return self._to_return_units(self.policy.critic(global_state, pursuer_pos, step))

    # -------------------------------------------------------------- returns
    @torch.no_grad()
    def compute_returns(self, buffer: RolloutBuffer, last_values: torch.Tensor) -> None:
        """Fill ``buffer.advantages`` and ``buffer.returns`` (return units)."""
        next_values = build_next_values(
            buffer.values, last_values, buffer.truncated, buffer.final_values
        )
        advantages, returns = compute_gae(
            buffer.rewards,
            buffer.values,
            next_values,
            buffer.terminated,
            buffer.truncated,
            self.config.gamma,
            self.config.gae_lambda,
        )
        buffer.advantages.copy_(advantages)
        buffer.returns.copy_(returns)

    # --------------------------------------------------------------- update
    def _minibatch_terms(self, mb: Minibatch, old_norm: ValueNorm | None) -> LossTerms:
        """``old_norm``: ValueNorm as it was when the rollout's values were predicted.

        The value-clip centre must be the critic's own earlier output, i.e. the stored
        values normalized with the OLD stats. Targets use the updated stats.
        """
        length, batch, n_agents = mb.actions.shape
        actor = self.policy.actor
        delivery = mb.delivery if actor.comm is not None else None
        own_pos = mb.own_pos if actor.use_own_position else None
        logits = actor.unroll(mb.obs, mb.actor_hidden0, mb.episode_start, delivery, own_pos)
        dist = Categorical(logits=logits)
        raw_values = self.policy.critic(
            mb.global_state.flatten(0, 1), mb.pursuer_pos.flatten(0, 1), mb.step.flatten(0, 1)
        ).reshape(length, batch, n_agents)
        cfg = self.config
        return ppo_loss_terms(
            new_log_probs=dist.log_prob(mb.actions),
            old_log_probs=mb.old_log_probs,
            advantages=mb.advantages,
            entropy=dist.entropy(),
            new_values=raw_values,
            old_values=old_norm.normalize(mb.old_values) if old_norm is not None else mb.old_values,
            value_targets=self._to_critic_units(mb.returns),
            mask=mb.agent_mask,
            clip_eps=cfg.clip_eps,
            value_clip_eps=cfg.value_clip_eps,
            huber_delta=cfg.huber_delta,
        )

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        """Run ``epochs`` x ``num_minibatches`` PPO steps on a full buffer."""
        cfg = self.config
        mask = buffer.agent_mask
        diagnostics = {
            "explained_variance": explained_variance(buffer.values, buffer.returns, mask),
            "advantage_mean": float(buffer.advantages[mask].mean()),
            "advantage_std": float(buffer.advantages[mask].std()),
            "return_mean": float(buffer.returns[mask].mean()),
            "value_mean": float(buffer.values[mask].mean()),
        }
        old_norm = copy.deepcopy(self.value_norm) if self.value_norm is not None else None
        if self.value_norm is not None:
            self.value_norm.update(buffer.returns[mask])

        advantages = buffer.advantages
        if cfg.normalize_advantages:
            advantages = normalize_advantages(advantages, mask)

        sums: dict[str, float] = {}
        n_steps = 0
        stopped_early = False
        self.policy.train()
        for _epoch in range(cfg.epochs):
            epoch_kl = []
            for mb in buffer.minibatches(
                cfg.num_minibatches, cfg.chunk_length, self.generator, advantages
            ):
                terms = self._minibatch_terms(mb, old_norm)
                actor_loss = terms.policy_loss - cfg.entropy_coef * terms.entropy
                critic_loss = cfg.value_coef * terms.value_loss
                total = actor_loss + critic_loss
                if not torch.isfinite(total):
                    raise RuntimeError(
                        f"non-finite loss: policy={terms.policy_loss.item()} "
                        f"value={terms.value_loss.item()} entropy={terms.entropy.item()}"
                    )
                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                total.backward()
                actor_grad = torch.nn.utils.clip_grad_norm_(
                    self.policy.actor.parameters(), cfg.max_grad_norm
                )
                critic_grad = torch.nn.utils.clip_grad_norm_(
                    self.policy.critic.parameters(), cfg.max_grad_norm
                )
                if not (torch.isfinite(actor_grad) and torch.isfinite(critic_grad)):
                    raise RuntimeError("non-finite gradient norm")
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                record = {
                    "policy_loss": terms.policy_loss.item(),
                    "value_loss": terms.value_loss.item(),
                    "entropy": terms.entropy.item(),
                    "weighted_entropy_bonus": cfg.entropy_coef * terms.entropy.item(),
                    "weighted_value_loss": critic_loss.item(),
                    "approx_kl": terms.approx_kl.item(),
                    "clip_fraction": terms.clip_fraction.item(),
                    "actor_grad_norm": float(actor_grad),
                    "critic_grad_norm": float(critic_grad),
                }
                for key, value in record.items():
                    sums[key] = sums.get(key, 0.0) + value
                n_steps += 1
                epoch_kl.append(record["approx_kl"])
            if cfg.target_kl > 0 and sum(epoch_kl) / len(epoch_kl) > 1.5 * cfg.target_kl:
                stopped_early = True
                break

        metrics = {key: value / n_steps for key, value in sums.items()}
        metrics.update(diagnostics)
        metrics["gradient_steps"] = n_steps
        metrics["kl_early_stop"] = float(stopped_early)
        return {k: v for k, v in metrics.items() if not (isinstance(v, float) and math.isnan(v))}
