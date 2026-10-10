"""MAPPO training loop: collect -> GAE -> PPO update -> log -> checkpoint.

Information boundary in the loop: the actor is called with ``ActorObs`` fields
only (local views); ``PrivilegedInfo`` feeds the critic and nothing else.
"""

from __future__ import annotations

import math
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dynabelief.algorithms.mappo import MAPPO
from dynabelief.beliefs.targets import point_targets, visible_evaders
from dynabelief.comm.channel import PacketLossChannel
from dynabelief.config import Config
from dynabelief.envs.pursuit import PrivilegedInfo, stack_privileged
from dynabelief.envs.vector import PursuitVecEnv
from dynabelief.models.actor_critic import MAPPOPolicy
from dynabelief.models.comm import TarMACComm
from dynabelief.replay.infer import EventReplayBuffer, first_sightings
from dynabelief.training.buffer import RolloutBuffer
from dynabelief.training.rollout import summarize_episodes
from dynabelief.utils.checkpointing import (
    check_compatible,
    load_checkpoint,
    restore_rng_state,
    rng_state,
    save_checkpoint,
)
from dynabelief.utils.logging import JsonlWriter, get_logger, resource_usage
from dynabelief.utils.seeding import derive_seed, make_rng, seed_everything


def build_policy(config: Config, obs_shape, state_shape, n_actions: int) -> MAPPOPolicy:
    m = config.model
    return MAPPOPolicy(
        obs_shape=tuple(obs_shape),
        state_shape=tuple(state_shape),
        n_actions=n_actions,
        max_cycles=config.env.max_cycles,
        hidden_dim=m.hidden_dim,
        conv_channels=m.conv_channels,
        critic_hidden_dim=m.critic_hidden_dim,
        comm_key_dim=config.comm.key_dim if config.comm.enabled else None,
        comm_value_dim=config.comm.message_dim if config.comm.enabled else None,
        use_own_position=m.use_own_position,
        belief_type=config.belief.type,
    )


class MAPPOTrainer:
    def __init__(
        self,
        config: Config,
        device: str,
        run_dir: Path | None = None,
        resume_from: str | Path | None = None,
    ) -> None:
        if config.train.policy != "mappo":
            raise ValueError("MAPPOTrainer requires train.policy: mappo")
        self.config = config
        self.device = torch.device(device)
        self.run_dir = Path(run_dir) if run_dir is not None else None
        if self.run_dir is not None:
            self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log = get_logger()
        seed = config.experiment.seed
        seed_everything(seed, deterministic_torch=config.experiment.deterministic)
        if config.train.torch_threads:
            torch.set_num_threads(config.train.torch_threads)

        checkpoint = None
        if resume_from is not None:
            # Load on CPU: load_state_dict copies tensors to each parameter's device.
            checkpoint = load_checkpoint(resume_from, map_location="cpu")
            check_compatible(checkpoint["config"], config.to_dict(), resume=True)

        train = config.train
        self.vec = PursuitVecEnv(
            config.env,
            train.num_envs,
            seed,
            stream="train",
            num_workers=train.num_workers,
            episode_index=checkpoint["episode_index"] if checkpoint else None,
        )
        try:
            self.policy = build_policy(
                config, self.vec.obs_shape, self.vec.state_shape, self.vec.n_actions
            )
            self.policy.to(self.device)
            generator = torch.Generator().manual_seed(derive_seed(seed, "minibatch"))
            # InfER buffer (not checkpointed: a resumed run refills it from new episodes).
            self.replay = (
                EventReplayBuffer(
                    config.replay.capacity,
                    config.replay.window,
                    self.vec.n_agents,
                    self.vec.obs_shape,
                    config.model.hidden_dim,
                    torch.Generator().manual_seed(derive_seed(seed, "replay")),
                )
                if config.replay.type != "none"
                else None
            )
            self._seen = np.zeros((train.num_envs, config.env.n_evaders), dtype=bool)
            self.algo = MAPPO(
                self.policy,
                config.ppo,
                self.device,
                generator,
                belief_coef=config.belief.coef if config.belief.type != "none" else 0.0,
                grid_scale=float(config.env.x_size - 1),
                replay=self.replay,
                replay_config=config.replay if self.replay is not None else None,
            )
            self.buffer = RolloutBuffer(
                train.rollout_length,
                train.num_envs,
                self.vec.n_agents,
                self.vec.obs_shape,
                self.vec.state_shape,
                config.model.hidden_dim,
                self.device,
            )
            # Seeded lossy channel (M3). Messages are sampled per directed link per step.
            comm = config.comm
            self.channel = (
                PacketLossChannel(
                    self.vec.n_agents,
                    comm.packet_loss,
                    make_rng(seed, "comm"),
                    message_dim=comm.elements_per_message,
                    bytes_per_element=comm.bytes_per_element,
                )
                if comm.enabled
                else None
            )
            self.env_steps = 0
            self.update_index = 0
            self.best_score = -math.inf
            if checkpoint is not None:
                self.algo.load_state_dict(checkpoint["learner"])
                restore_rng_state(checkpoint["rng"], self.algo.generator)
                if self.channel is not None:
                    self.channel.rng.bit_generator.state = checkpoint["comm_rng"]
                self.env_steps = checkpoint["env_steps"]
                self.update_index = checkpoint["update"]
                # best.pt is per run directory: a resumed run starts its own best tracking
                # (the recent-episode window is not checkpointed either).
                self.log.info(
                    "resumed from %s at update %d (%d env steps); environments start new episodes",
                    resume_from,
                    self.update_index,
                    self.env_steps,
                )
        except BaseException:
            self.vec.close()  # never leave worker processes behind
            raise
        self._recent_episodes: deque[dict[str, Any]] = deque(maxlen=100)

    # ------------------------------------------------------------- helpers
    def _t(self, array: np.ndarray, dtype: torch.dtype | None = None) -> torch.Tensor:
        return torch.as_tensor(array, device=self.device, dtype=dtype)

    def _critic_values(self, priv: PrivilegedInfo) -> torch.Tensor:
        return self.algo.values(
            self._t(priv.global_state, torch.float32),
            self._t(priv.pursuer_pos, torch.long),
            self._t(priv.step, torch.long),
        )

    # ------------------------------------------------------------- collect
    def _collect(self, actor, priv, hidden, start, episodes_log) -> tuple:
        buf = self.buffer
        buf.reset()
        completed: list[dict[str, Any]] = []
        entropy_sum, entropy_n = 0.0, 0
        n = self.vec.n_agents
        uses_pos = self.policy.actor.use_own_position
        env_cfg = self.config.env
        events = torch.zeros(buf.T, self.vec.num_envs, dtype=torch.bool)
        for t in range(buf.T):
            obs = self._t(actor.obs, torch.float32)
            own_pos = self._t(actor.own_pos, torch.float32)
            if self.channel is not None:
                # Messages readable now were sent at t-1; none exist at an episode start.
                alive = actor.agent_mask & ~start.cpu().numpy()[:, None]
                delivery = self._t(self.channel.sample_delivery(alive), torch.bool)
            else:
                delivery = torch.zeros(self.vec.num_envs, n, n, dtype=torch.bool,
                                       device=self.device)  # fmt: skip
            # Actor sees only its local views, its own recurrent state and delivered messages.
            actions, log_probs, new_hidden = self.policy.act(
                obs,
                hidden,
                start,
                delivery=delivery if self.channel is not None else None,
                own_pos=own_pos if uses_pos else None,
            )
            if self.channel is not None and delivery.any():
                attention = self.policy.actor.last_attention
                entropy_sum += float(TarMACComm.attention_entropy(attention, delivery))
                entropy_n += 1
            values = self._critic_values(priv)
            # InfER events: an evader seen by the team for the first time this episode.
            if self.replay is not None:
                self._seen[start.cpu().numpy()] = False
                visible = visible_evaders(priv.pursuer_pos, priv.evader_pos, priv.evader_alive,
                                          env_cfg.obs_range).any(axis=-2)  # fmt: skip
                events[t] = torch.from_numpy(first_sightings(visible, self._seen).any(axis=-1))
                self._seen |= visible
            # TRAINING-ONLY labels for the state the actor just acted in (loss input only).
            if self.config.belief.type == "point":
                target, valid = point_targets(priv.pursuer_pos, priv.evader_pos,
                                              priv.evader_alive, env_cfg.obs_range,
                                              (env_cfg.x_size, env_cfg.y_size))  # fmt: skip
            else:
                target = np.zeros((self.vec.num_envs, n, 2), dtype=np.float32)
                valid = np.zeros((self.vec.num_envs, n), dtype=bool)
            result = self.vec.step(actions.cpu().numpy())

            final_values = torch.zeros_like(values)
            truncated_envs = [i for i in result.final_privileged if result.truncated[i]]
            if truncated_envs:
                finals = stack_privileged([result.final_privileged[i] for i in truncated_envs])
                final_values[truncated_envs] = self._critic_values(finals)

            buf.add(
                obs=obs,
                episode_start=start,
                actor_hidden=hidden,
                actions=actions,
                log_probs=log_probs,
                values=values,
                rewards=self._t(result.rewards, torch.float32),
                agent_mask=self._t(actor.agent_mask, torch.bool),
                delivery=delivery,
                own_pos=own_pos,
                belief_target=self._t(target, torch.float32),
                belief_valid=self._t(valid, torch.bool),
                terminated=self._t(result.terminated, torch.bool),
                truncated=self._t(result.truncated, torch.bool),
                final_values=final_values,
                global_state=self._t(priv.global_state, torch.float32),
                pursuer_pos=self._t(priv.pursuer_pos, torch.long),
                step=self._t(priv.step, torch.long),
            )
            for stats in result.completed:
                record = {**stats.as_dict(), "env_steps": self.env_steps}
                completed.append(record)
                if episodes_log:
                    episodes_log.write(record)
            self.env_steps += self.vec.num_envs
            actor, priv = result.actor, result.privileged
            hidden, start = new_hidden, self._t(result.done, torch.bool)

        self.algo.compute_returns(buf, self._critic_values(priv))
        self._replay_metrics = {}
        if self.replay is not None:
            skipped_before = self.replay.skipped
            added = self.replay.add_from_rollout(buf, events)
            self._replay_metrics = {
                "replay_events": int(events.sum()),
                "replay_windows_added": added,
                "replay_windows_skipped": self.replay.skipped - skipped_before,
            }
        self._comm_metrics = {}
        if self.channel is not None:
            self._comm_metrics = {
                **{f"comm_{k}": v for k, v in self.channel.reset_stats().as_dict().items()},
                "comm_attention_entropy": entropy_sum / entropy_n if entropy_n else 0.0,
            }
        return actor, priv, hidden, start, completed

    # ---------------------------------------------------------- checkpoint
    def _checkpoint_payload(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "learner": self.algo.state_dict(),
            "rng": rng_state(self.algo.generator),
            "env_steps": self.env_steps,
            "update": self.update_index,
            "episode_index": self.vec.episode_index(),
            "best_score": self.best_score,
            "comm_rng": self.channel.rng.bit_generator.state if self.channel else None,
        }

    def save(self, name: str) -> Path | None:
        if self.run_dir is None:
            return None
        return save_checkpoint(self.run_dir / "checkpoints" / name, self._checkpoint_payload())

    # ---------------------------------------------------------------- train
    def train(self) -> dict[str, Any]:
        cfg = self.config.train
        steps_per_update = cfg.rollout_length * cfg.num_envs
        total_updates = math.ceil(cfg.total_env_steps / steps_per_update)
        metrics_log = JsonlWriter(self.run_dir / "metrics.jsonl") if self.run_dir else None
        episodes_log = JsonlWriter(self.run_dir / "episodes.jsonl") if self.run_dir else None
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        start_time = time.perf_counter()
        start_steps = self.env_steps
        last: dict[str, Any] = {}
        try:
            actor, priv = self.vec.reset()
            hidden = self.policy.actor.initial_state(cfg.num_envs, self.vec.n_agents, self.device)
            start = torch.ones(cfg.num_envs, dtype=torch.bool, device=self.device)
            self.log.info(
                "MAPPO: %d updates x %d env steps (%d envs x %d steps), device=%s, workers=%d",
                total_updates - self.update_index,
                steps_per_update,
                cfg.num_envs,
                cfg.rollout_length,
                self.device,
                cfg.num_workers,
            )
            while self.update_index < total_updates:
                t0 = time.perf_counter()
                self.policy.eval()
                actor, priv, hidden, start, completed = self._collect(
                    actor, priv, hidden, start, episodes_log
                )
                t1 = time.perf_counter()
                update_metrics = self.algo.update(self.buffer)
                t2 = time.perf_counter()
                self.update_index += 1
                self._recent_episodes.extend(completed)

                window = summarize_episodes(list(self._recent_episodes))
                record = {
                    "update": self.update_index,
                    "env_steps": self.env_steps,
                    "collect_fps": steps_per_update / (t1 - t0),
                    "update_seconds": t2 - t1,
                    "overall_fps": (self.env_steps - start_steps) / (t2 - start_time),
                    "episodes_this_update": len(completed),
                    "rollout_mean_reward": float(self.buffer.rewards.mean()),
                    **{f"window_{k}": v for k, v in window.items() if v is not None},
                    **update_metrics,
                    **self._comm_metrics,
                    **self._replay_metrics,
                    **resource_usage(),
                }
                if metrics_log:
                    metrics_log.write(record)
                last = record
                self.log.info(
                    "upd %d/%d steps=%d fps=%.0f | ret=%.2f cap=%.3f len=%.0f | "
                    "pi=%.3f v=%.3f ent=%.3f kl=%.4f clip=%.3f ev=%.2f",
                    self.update_index,
                    total_updates,
                    self.env_steps,
                    record["overall_fps"],
                    window.get("mean_team_return", float("nan")),
                    window.get("mean_capture_rate", float("nan")),
                    window.get("mean_length", float("nan")),
                    update_metrics["policy_loss"],
                    update_metrics["value_loss"],
                    update_metrics["entropy"],
                    update_metrics["approx_kl"],
                    update_metrics["clip_fraction"],
                    update_metrics.get("explained_variance", float("nan")),
                )

                score = window.get("mean_capture_rate")
                if score is not None and window["episodes"] >= 10 and score > self.best_score:
                    self.best_score = score
                    self.save("best.pt")
                every = cfg.checkpoint_every_updates
                if every and self.update_index % every == 0:
                    self.save("latest.pt")
            final_path = self.save("final.pt")
            self.save("latest.pt")
        finally:
            self.vec.close()
            if metrics_log:
                metrics_log.close()
            if episodes_log:
                episodes_log.close()

        elapsed = time.perf_counter() - start_time
        return {
            "milestone": "M2",
            "policy": "mappo",
            "learning": True,
            "updates": self.update_index,
            "env_steps": self.env_steps,
            "wall_time_s": elapsed,
            "fps": (self.env_steps - start_steps) / elapsed if elapsed > 0 else 0.0,
            "final_checkpoint": str(final_path) if final_path else None,
            "best_window_capture_rate": None if self.best_score == -math.inf else self.best_score,
            "last_update": last,
            **resource_usage(),
        }
