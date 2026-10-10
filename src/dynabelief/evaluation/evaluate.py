"""Evaluation on a fixed, ordered list of episode seeds.

Evaluation seeds come from the ``"eval"`` stream, disjoint from the training
stream, so paired comparisons between methods see identical episodes. Every
method runs through the same ``evaluate`` loop; only the agent differs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from dynabelief.comm.channel import PacketLossChannel
from dynabelief.config import Config, config_from_dict
from dynabelief.envs.pursuit import ActorObs, PursuitEnv
from dynabelief.policies.random_policy import RandomPolicy
from dynabelief.training.rollout import summarize_episodes
from dynabelief.utils.checkpointing import check_compatible, load_checkpoint
from dynabelief.utils.logging import JsonlWriter
from dynabelief.utils.seeding import derive_seed, make_rng


def eval_seeds(base_seed: int, episodes: int) -> list[int]:
    return [derive_seed(base_seed, "eval", 0, k) for k in range(episodes)]


class EvalAgent(Protocol):
    """Decentralized controller for evaluation: sees only ``ActorObs`` and the channel's
    delivery mask ``[N_recv, N_send]`` (``None`` when communication is disabled)."""

    def reset(self, episode: int) -> None: ...

    def act(self, actor: ActorObs, delivery: np.ndarray | None = None) -> np.ndarray: ...


class RandomAgent:
    def __init__(self, n_actions: int, seed: int) -> None:
        self.n_actions, self.seed = n_actions, seed
        self.policy: RandomPolicy | None = None

    def reset(self, episode: int) -> None:
        self.policy = RandomPolicy(self.n_actions, make_rng(self.seed, "eval_policy", episode))

    def act(self, actor: ActorObs, delivery: np.ndarray | None = None) -> np.ndarray:
        assert self.policy is not None
        return self.policy.act(actor)


class MAPPOAgent:
    """Wraps a trained actor with its recurrent state; per-episode seeded sampling."""

    def __init__(self, policy, seed: int, device: str, deterministic: bool) -> None:
        self.policy = policy.eval()
        self.seed, self.device, self.deterministic = seed, torch.device(device), deterministic
        self.hidden: torch.Tensor | None = None
        self.start = True

    def reset(self, episode: int) -> None:
        torch.manual_seed(derive_seed(self.seed, "eval_policy", episode))
        self.hidden, self.start = None, True

    def act(self, actor: ActorObs, delivery: np.ndarray | None = None) -> np.ndarray:
        obs = torch.as_tensor(actor.obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if self.hidden is None:
            self.hidden = self.policy.actor.initial_state(1, obs.shape[1], self.device)
        start = torch.tensor([self.start], device=self.device)
        mask = None
        if self.policy.actor.comm is not None:
            if delivery is None:
                raise ValueError("communicating policy evaluated without a delivery mask")
            mask = torch.as_tensor(delivery, dtype=torch.bool, device=self.device).unsqueeze(0)
        own_pos = None
        if self.policy.actor.use_own_position:
            own_pos = torch.as_tensor(actor.own_pos, device=self.device).unsqueeze(0)
        actions, _, self.hidden = self.policy.act(
            obs, self.hidden, start, self.deterministic, delivery=mask, own_pos=own_pos
        )
        self.start = False
        return np.where(actor.agent_mask, actions[0].cpu().numpy(), 0)


def evaluate(
    config: Config,
    agent: EvalAgent,
    episodes: int,
    run_dir: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run ``agent`` on the fixed eval seeds; per-episode records go to ``episodes.jsonl``."""
    seed = config.experiment.seed
    env = PursuitEnv(config.env)
    records: list[dict[str, Any]] = []
    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    writer = JsonlWriter(run_dir / "episodes.jsonl") if run_dir else None
    try:
        for k, episode_seed in enumerate(eval_seeds(seed, episodes)):
            actor, _ = env.reset(episode_seed)
            agent.reset(k)
            # Fresh per-episode stream: episode k gets the same drop pattern for every
            # method, independent of how long earlier episodes lasted.
            channel = PacketLossChannel(
                env.n_agents,
                config.comm.packet_loss,
                make_rng(seed, "eval_comm", k),
                message_dim=config.comm.elements_per_message,
                bytes_per_element=config.comm.bytes_per_element,
            )
            first_step = True
            while True:
                delivery = None
                if config.comm.enabled:
                    # Messages readable now were sent at t-1; none exist at the first step.
                    alive = actor.agent_mask & (not first_step)
                    delivery = channel.sample_delivery(alive)
                result = env.step(agent.act(actor, delivery))
                first_step = False
                actor = result.actor
                if result.done:
                    break
            stats = env.episode_stats
            assert stats is not None
            record = {
                **stats.as_dict(),
                "episode": k,
                "packet_loss": config.comm.packet_loss,
                **(extra or {}),
                **(channel.stats.as_dict() if config.comm.enabled else {}),
            }
            records.append(record)
            if writer:
                writer.write(record)
    finally:
        env.close()
        if writer:
            writer.close()
    summary = summarize_episodes(records)
    if config.comm.enabled and records:
        sent = sum(r["messages_sent"] for r in records)
        delivered = sum(r["messages_delivered"] for r in records)
        steps = sum(r["length"] for r in records)
        summary.update(
            packet_loss=config.comm.packet_loss,
            messages_delivered_per_step=delivered / steps,
            drop_rate=1.0 - delivered / sent if sent else 0.0,
            bytes_per_capture=(
                sum(r["bytes_sent"] for r in records) / max(1, sum(r["captures"] for r in records))
            ),
        )
    return summary


def evaluate_random(config: Config, episodes: int, run_dir: Path | None = None) -> dict[str, Any]:
    env = PursuitEnv(config.env)
    n_actions = env.n_actions
    env.close()
    agent = RandomAgent(n_actions, config.experiment.seed)
    summary = evaluate(config, agent, episodes, run_dir, extra={"policy": "random"})
    return {"policy": "random", "learning": False, **summary}


def load_policy_for_eval(checkpoint_path: str | Path, config: Config, device: str):
    """Build the actor-critic from the checkpoint's model config for ``config``'s env."""
    from dynabelief.training.trainer import build_policy

    data = load_checkpoint(checkpoint_path, map_location=device)
    check_compatible(data["config"], config.to_dict(), resume=False)
    saved = config_from_dict(data["config"])
    env = PursuitEnv(config.env)
    policy = build_policy(saved, env.obs_shape, env.state_shape, env.n_actions)
    env.close()
    policy.load_state_dict(data["learner"]["policy"])
    return policy.to(device), data


def evaluate_checkpoint(
    config: Config,
    checkpoint_path: str | Path,
    episodes: int,
    device: str,
    deterministic: bool = False,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    policy, data = load_policy_for_eval(checkpoint_path, config, device)
    agent = MAPPOAgent(policy, config.experiment.seed, device, deterministic)
    extra = {
        "policy": "mappo",
        "checkpoint": str(checkpoint_path),
        "checkpoint_env_steps": data["env_steps"],
        "deterministic": deterministic,
    }
    with torch.no_grad():
        summary = evaluate(config, agent, episodes, run_dir, extra=extra)
    return {**extra, "learning": True, **summary}
