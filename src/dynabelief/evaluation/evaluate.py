"""Evaluation on a fixed, ordered list of episode seeds.

Evaluation seeds come from the ``"eval"`` stream, disjoint from the training
stream, so paired comparisons between methods see identical episodes. Every
method runs through the same ``evaluate`` loop; only the agent differs.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from dynabelief.beliefs import maps
from dynabelief.beliefs.targets import point_targets
from dynabelief.comm.channel import PacketLossChannel
from dynabelief.config import Config, config_from_dict
from dynabelief.envs.pursuit import ActorObs, PursuitEnv
from dynabelief.policies.random_policy import RandomPolicy
from dynabelief.training.rollout import summarize_episodes
from dynabelief.utils.checkpointing import check_compatible, load_checkpoint
from dynabelief.utils.logging import JsonlWriter
from dynabelief.utils.seeding import derive_seed, make_rng


class MapScores:
    """Per-episode accumulator for occupancy-map beliefs and their naive references."""

    KEYS = ("model", "prior", "own_view", "remote_model", "remote_prior")

    def __init__(self) -> None:
        self.bce = {k: [0.0, 0] for k in self.KEYS}
        self.brier = [0.0, 0]
        self.bins = np.zeros((maps.N_BINS, 3))
        self.remote_bins = np.zeros((maps.N_BINS, 3))

    def _add(self, key: str, p: np.ndarray, y: np.ndarray, mask: np.ndarray) -> None:
        if mask.any():
            n = int(mask.sum())
            self.bce[key][0] += maps.bce(p, y, mask) * n
            self.bce[key][1] += n

    def update(self, belief, priv, actor, obs_range: int) -> None:
        occupancy, free = maps.map_targets(priv.global_state)
        y = np.broadcast_to(occupancy, belief.shape)
        mask = np.broadcast_to(free, belief.shape) & actor.agent_mask[:, None, None]
        own = maps.own_view_mask(priv.pursuer_pos, occupancy.shape, obs_range)
        density = occupancy[free].mean() if free.any() else 0.0
        prior = np.full(belief.shape, density)
        own_view = np.where(own, y, prior)  # exact inside one's own view, prior elsewhere
        remote = mask & ~own
        self._add("model", belief, y, mask)
        self._add("prior", prior, y, mask)
        self._add("own_view", own_view, y, mask)
        self._add("remote_model", belief, y, remote)
        self._add("remote_prior", prior, y, remote)
        self.brier[0] += maps.brier(belief, y, mask) * int(mask.sum())
        self.brier[1] += int(mask.sum())
        self.bins += maps.calibration_bins(belief, y, mask)
        self.remote_bins += maps.calibration_bins(belief, y, remote)

    def record(self) -> dict[str, Any]:
        out = {f"map_bce_{k}": v[0] / v[1] for k, v in self.bce.items() if v[1]}
        if self.brier[1]:
            out["map_brier"] = self.brier[0] / self.brier[1]
        out["map_calibration_bins"] = self.bins.tolist()
        out["map_remote_calibration_bins"] = self.remote_bins.tolist()
        return out


def _cell_errors(guess: np.ndarray, target: np.ndarray, valid: np.ndarray, scale: np.ndarray):
    """Euclidean errors in grid cells for the labelled agents."""
    return np.linalg.norm((guess - target)[valid] * scale, axis=-1).tolist()


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
        # Point belief [N, 2] (normalized) after the latest act(), if the actor has one.
        self.last_belief: np.ndarray | None = None

    def reset(self, episode: int) -> None:
        torch.manual_seed(derive_seed(self.seed, "eval_policy", episode))
        self.hidden, self.start, self.last_belief = None, True, None

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
        if self.policy.actor.belief_head is not None:
            with torch.no_grad():
                out = self.policy.actor.belief_head(self.hidden)[0]
                if self.policy.actor.belief_type == "map":
                    out = torch.sigmoid(out)  # probabilities only for metrics / plots
                self.last_belief = out.cpu().numpy()
        self.start = False
        return np.where(actor.agent_mask, actions[0].cpu().numpy(), 0)


def evaluate(
    config: Config,
    agent: EvalAgent,
    episodes: int,
    run_dir: Path | None = None,
    extra: dict[str, Any] | None = None,
    on_step: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run ``agent`` on the fixed eval seeds; per-episode records go to ``episodes.jsonl``.

    For agents with a point belief, each step scores it against the training-only label
    of the same state (in grid cells) together with two naive references: guessing one's
    own position and guessing the grid centre. ``on_step`` receives a per-step dict
    (episode, t, actor, privileged, belief, target, valid) for visualization.
    """
    env_cfg = config.env
    scale = np.array([env_cfg.x_size - 1, env_cfg.y_size - 1], dtype=np.float32)
    seed = config.experiment.seed
    env = PursuitEnv(config.env)
    records: list[dict[str, Any]] = []
    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    writer = JsonlWriter(run_dir / "episodes.jsonl") if run_dir else None
    try:
        for k, episode_seed in enumerate(eval_seeds(seed, episodes)):
            actor, priv = env.reset(episode_seed)
            agent.reset(k)
            errors: dict[str, list[float]] = {
                "model": [], "own": [], "centre": [], "remote_model": [], "remote_own": [],
            }  # fmt: skip
            labelled, remote, t = 0, 0, 0
            map_scores = MapScores()
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
                actions = agent.act(actor, delivery)
                belief = getattr(agent, "last_belief", None)
                if belief is not None and belief.ndim == 3:  # occupancy map [N, Y, X]
                    map_scores.update(belief, priv, actor, env_cfg.obs_range)
                    if on_step is not None:
                        on_step({"episode": k, "t": t, "actor": actor, "privileged": priv,
                                 "belief": belief, "target": None, "valid": None})  # fmt: skip
                elif belief is not None or on_step is not None:
                    target, valid = point_targets(priv.pursuer_pos, priv.evader_pos,
                                                  priv.evader_alive, env_cfg.obs_range,
                                                  (env_cfg.x_size, env_cfg.y_size))  # fmt: skip
                    if belief is not None and valid.any():
                        errors["model"] += _cell_errors(belief, target, valid, scale)
                        errors["own"] += _cell_errors(actor.own_pos, target, valid, scale)
                        centre = np.full_like(target, 0.5)
                        errors["centre"] += _cell_errors(centre, target, valid, scale)
                        labelled += int(valid.sum())
                        # Remote-only labels: the target is outside the agent's own view, so
                        # only teammates' messages can tell it where the evader is.
                        offset = np.abs(np.rint(target * scale) - priv.pursuer_pos)
                        in_own_view = (offset <= env_cfg.obs_range // 2).all(axis=-1)
                        far = valid & ~in_own_view
                        if far.any():
                            errors["remote_model"] += _cell_errors(belief, target, far, scale)
                            errors["remote_own"] += _cell_errors(actor.own_pos, target, far, scale)
                            remote += int(far.sum())
                    if on_step is not None:
                        on_step({"episode": k, "t": t, "actor": actor, "privileged": priv,
                                 "belief": belief, "target": target, "valid": valid})  # fmt: skip
                result = env.step(actions)
                first_step = False
                actor, priv = result.actor, result.privileged
                t += 1
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
            if map_scores.brier[1]:
                record.update(map_scores.record())
            if errors["model"]:
                record.update(
                    belief_error_cells=float(np.mean(errors["model"])),
                    belief_ref_own_position_cells=float(np.mean(errors["own"])),
                    belief_ref_grid_centre_cells=float(np.mean(errors["centre"])),
                    belief_valid_fraction=labelled / (stats.length * env.n_agents),
                    belief_remote_fraction=remote / max(1, labelled),
                )
                if errors["remote_model"]:
                    record.update(
                        belief_remote_error_cells=float(np.mean(errors["remote_model"])),
                        belief_remote_ref_own_position_cells=float(np.mean(errors["remote_own"])),
                    )
            records.append(record)
            if writer:
                writer.write(record)
    finally:
        env.close()
        if writer:
            writer.close()
    summary = summarize_episodes(records)
    belief_keys = ("belief_error_cells", "belief_ref_own_position_cells",
                   "belief_ref_grid_centre_cells", "belief_valid_fraction")  # fmt: skip
    if records and all(key in records[0] for key in belief_keys):
        summary.update({key: float(np.mean([r[key] for r in records])) for key in belief_keys})
        summary["belief_remote_fraction"] = float(
            np.mean([r["belief_remote_fraction"] for r in records])
        )
        for key in ("belief_remote_error_cells", "belief_remote_ref_own_position_cells"):
            values = [r[key] for r in records if key in r]
            if values:
                summary[key] = float(np.mean(values))
    if records and "map_brier" in records[0]:
        for key in [k for k in records[0] if k.startswith("map_bce_")] + ["map_brier"]:
            summary[key] = float(np.mean([r[key] for r in records]))
        for key, out in (("map_calibration_bins", "map_ece"),
                         ("map_remote_calibration_bins", "map_remote_ece")):  # fmt: skip
            summary[out] = maps.expected_calibration_error(
                np.sum([np.asarray(r[key]) for r in records], axis=0)
            )
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
