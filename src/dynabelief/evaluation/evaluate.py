"""Evaluation on a fixed, ordered list of episode seeds.

Evaluation seeds come from the ``"eval"`` stream, disjoint from the training
stream, so paired comparisons between methods see identical episodes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dynabelief.comm.channel import PacketLossChannel
from dynabelief.config import Config
from dynabelief.envs.pursuit import PursuitEnv
from dynabelief.policies.random_policy import RandomPolicy
from dynabelief.training.rollout import summarize_episodes
from dynabelief.utils.logging import JsonlWriter
from dynabelief.utils.seeding import derive_seed, make_rng


def eval_seeds(base_seed: int, episodes: int) -> list[int]:
    return [derive_seed(base_seed, "eval", 0, k) for k in range(episodes)]


def evaluate_random(config: Config, episodes: int, run_dir: Path | None = None) -> dict[str, Any]:
    """Evaluate the random policy; per-episode records are written to ``episodes.jsonl``."""
    seed = config.experiment.seed
    env = PursuitEnv(config.env)
    policy = RandomPolicy(env.n_actions, make_rng(seed, "eval_policy"))
    records: list[dict[str, Any]] = []
    writer = JsonlWriter(run_dir / "episodes.jsonl") if run_dir else None
    try:
        for k, episode_seed in enumerate(eval_seeds(seed, episodes)):
            actor, _ = env.reset(episode_seed)
            # Fresh per-episode stream: episode k gets the same drop pattern for every
            # method, independent of how long earlier episodes lasted.
            channel = PacketLossChannel(
                env.n_agents,
                config.comm.packet_loss,
                make_rng(seed, "eval_comm", k),
                message_dim=config.comm.message_dim,
                bytes_per_element=config.comm.bytes_per_element,
            )
            while True:
                if config.comm.enabled:
                    channel.sample_delivery(actor.agent_mask)
                result = env.step(policy.act(actor))
                actor = result.actor
                if result.done:
                    break
            stats = env.episode_stats
            assert stats is not None
            record = {
                **stats.as_dict(),
                "packet_loss": config.comm.packet_loss,
                **(channel.stats.as_dict() if config.comm.enabled else {}),
            }
            records.append(record)
            if writer:
                writer.write(record)
    finally:
        env.close()
        if writer:
            writer.close()
    return {"policy": "random", "learning": False, **summarize_episodes(records)}
