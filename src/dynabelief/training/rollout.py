"""M1 random-policy smoke rollout.

This exercises the full data path (vector env, auto-reset, packet-loss
accounting, logging, run metadata) WITHOUT any learning. Its summary is marked
``"learning": false`` so it can never be mistaken for a trained result.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from dynabelief.comm.channel import PacketLossChannel
from dynabelief.config import Config
from dynabelief.envs.vector import PursuitVecEnv
from dynabelief.policies.random_policy import RandomPolicy
from dynabelief.utils.logging import JsonlWriter, get_logger, resource_usage
from dynabelief.utils.seeding import make_rng, seed_everything


def describe_arrays(**arrays: np.ndarray) -> dict[str, str]:
    return {name: f"shape={tuple(a.shape)} dtype={a.dtype}" for name, a in arrays.items()}


def summarize_episodes(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"episodes": 0}
    ttc = [r["mean_time_to_capture"] for r in records if r["mean_time_to_capture"] is not None]
    return {
        "episodes": len(records),
        "mean_team_return": float(np.mean([r["team_return"] for r in records])),
        "mean_length": float(np.mean([r["length"] for r in records])),
        "mean_captures": float(np.mean([r["captures"] for r in records])),
        "mean_capture_rate": float(np.mean([r["capture_rate"] for r in records])),
        "mean_time_to_capture": float(np.mean(ttc)) if ttc else None,
        "all_captured_fraction": float(np.mean([r["all_captured"] for r in records])),
    }


def run_random_smoke(config: Config, run_dir: Path | None = None) -> dict[str, Any]:
    """Roll out the random policy for ``train.total_env_steps`` sub-environment steps.

    Returns a summary dict containing a ``trajectory_digest``: a SHA-256 over every
    actor observation (including terminal ones), action, reward, and done flag,
    used to verify determinism.
    """
    if config.train.policy != "random":
        raise NotImplementedError(f"policy {config.train.policy!r} is not implemented yet (M2+)")
    log = get_logger()
    seed = config.experiment.seed
    seed_everything(seed)

    vec = PursuitVecEnv(config.env, config.train.num_envs, seed, stream="train")
    policy = RandomPolicy(vec.n_actions, make_rng(seed, "policy"))
    channel = (
        PacketLossChannel(
            vec.n_agents,
            config.comm.packet_loss,
            make_rng(seed, "comm"),
            message_dim=config.comm.message_dim,
            bytes_per_element=config.comm.bytes_per_element,
        )
        if config.comm.enabled
        else None
    )

    episodes_log = JsonlWriter(run_dir / "episodes.jsonl") if run_dir else None
    metrics_log = JsonlWriter(run_dir / "metrics.jsonl") if run_dir else None
    digest = hashlib.sha256()
    episode_records: list[dict[str, Any]] = []
    rollout_rewards: list[float] = []

    num_envs = config.train.num_envs
    vector_steps = math.ceil(config.train.total_env_steps / num_envs)
    log_every = config.train.log_every_steps or config.train.rollout_length * num_envs

    try:
        actor, privileged = vec.reset()
        shapes = describe_arrays(
            actor_obs=actor.obs,
            agent_mask=actor.agent_mask,
            global_state=privileged.global_state,
            occupancy=privileged.occupancy,
            pursuer_pos=privileged.pursuer_pos,
            evader_pos=privileged.evader_pos,
        )
        log.info("tensor shapes (batch = num_envs=%d):", num_envs)
        for name, desc in shapes.items():
            log.info("  %-13s %s", name, desc)

        start = time.perf_counter()
        env_steps = 0
        next_log = log_every
        for _ in range(vector_steps):
            actions = policy.act(actor)
            if channel is not None:
                channel.sample_delivery(actor.agent_mask)
            result = vec.step(actions)

            digest.update(actor.obs.tobytes())
            digest.update(actions.tobytes())
            digest.update(result.rewards.tobytes())
            digest.update(result.terminated.tobytes() + result.truncated.tobytes())
            for i in sorted(result.final_actor):  # terminal obs are replaced by auto-reset
                digest.update(result.final_actor[i].obs.tobytes())

            for stats in result.completed:
                record = stats.as_dict()
                episode_records.append(record)
                if episodes_log:
                    episodes_log.write(record)
            rollout_rewards.append(float(result.rewards.mean()))
            env_steps += num_envs
            actor = result.actor

            if env_steps >= next_log or env_steps >= vector_steps * num_envs:
                next_log += log_every
                elapsed = time.perf_counter() - start
                metrics = {
                    "env_steps": env_steps,
                    "fps": env_steps / elapsed if elapsed > 0 else 0.0,
                    "mean_step_reward": float(np.mean(rollout_rewards)),
                    "episodes_completed": len(episode_records),
                    **(channel.stats.as_dict() if channel else {}),
                    **resource_usage(),
                }
                rollout_rewards.clear()
                if metrics_log:
                    metrics_log.write(metrics)
                log.info(
                    "steps=%d fps=%.0f episodes=%d mean_step_reward=%.4f",
                    env_steps,
                    metrics["fps"],
                    len(episode_records),
                    metrics["mean_step_reward"],
                )
        elapsed = time.perf_counter() - start
    finally:
        vec.close()
        if episodes_log:
            episodes_log.close()
        if metrics_log:
            metrics_log.close()

    summary = {
        "milestone": "M1",
        "policy": "random",
        "learning": False,
        "env_steps": env_steps,
        "wall_time_s": elapsed,
        "fps": env_steps / elapsed if elapsed > 0 else 0.0,
        "trajectory_digest": digest.hexdigest(),
        "shapes": shapes,
        **summarize_episodes(episode_records),
        **({"comm": channel.stats.as_dict()} if channel else {}),
        **resource_usage(),
    }
    if run_dir:
        with open(run_dir / "summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
    return summary
