"""Arm C′: the D-DIABL map trained on the team posterior (training-only Bayes filter).

Also guards the frozen arms: adding the C′ path must not change how ctrl / A / D / F train.
The reference ``tests/data/frozen_first_update.json`` was recorded from the code before the
C′ change (commit ee332f2, whose training code equals 50de5d5) with ``python -m
tests.test_posterior_target``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from dynabelief.config import ConfigError, config_from_dict, load_config
from dynabelief.training.trainer import MAPPOTrainer

REPO = Path(__file__).resolve().parents[1]
FROZEN = ("tarmac_pos", "point_diabl", "d_diabl_coef10", "point_diabl_stopgrad")
REFERENCE = REPO / "tests" / "data" / "frozen_first_update.json"
_VOLATILE = ("time", "fps", "seconds", "ram", "vram", "gb", "wall", "elapsed")


def small(arm: str, **belief):
    """The arm's own config, shrunk to two quick CPU updates."""
    raw = load_config(REPO / "configs" / f"{arm}.yaml").to_dict()
    raw["experiment"].update(device="cpu", output_dir="runs")
    raw["env"]["max_cycles"] = 40
    raw["train"].update(num_envs=4, num_workers=0, rollout_length=16, total_env_steps=128,
                        checkpoint_every_updates=1000)  # fmt: skip
    raw["ppo"].update(epochs=1, num_minibatches=2, chunk_length=8)
    if raw["replay"]["type"] != "none":
        raw["replay"].update(window=8, batch=4, capacity=64)
    raw["belief"].update(belief)
    return config_from_dict(raw)


def first_updates(config, run_dir: Path) -> list[dict]:
    torch.set_num_threads(2)
    MAPPOTrainer(config, "cpu", run_dir=run_dir).train()
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]

    def stable(key: str, value) -> bool:
        return isinstance(value, int | float) and not any(s in key for s in _VOLATILE)

    return [{k: v for k, v in row.items() if stable(k, v)} for row in rows]


@pytest.mark.parametrize("arm", FROZEN)
def test_frozen_configs_unchanged_loss(arm, tmp_path):
    reference = json.loads(REFERENCE.read_text())[arm]
    rows = first_updates(small(arm), tmp_path)
    assert len(rows) == len(reference)
    for got, want in zip(rows, reference, strict=True):
        assert set(got) == set(want)
        for key, value in want.items():
            assert got[key] == pytest.approx(value, rel=1e-4, abs=1e-7), key


def test_posterior_config_requires_map():
    raw = small("point_diabl").to_dict()
    raw["belief"]["map_target"] = "posterior"
    with pytest.raises(ConfigError, match="map_target"):
        config_from_dict(raw)
    raw = small("d_diabl_coef10").to_dict()
    raw["belief"]["map_target"] = "seen"
    with pytest.raises(ConfigError, match="map_target"):
        config_from_dict(raw)
    assert load_config(REPO / "configs" / "d_diabl_posterior.yaml").belief.map_target == "posterior"


def test_posterior_target_reaches_loss(tmp_path):
    trainer = MAPPOTrainer(small("d_diabl_coef10", map_target="posterior"), "cpu",
                           run_dir=tmp_path)  # fmt: skip
    trainer.train()
    buf = trainer.buffer
    target = buf.map_target
    truth = (buf.global_state[:, :, 2] > 0).float()
    free = buf.global_state[:, :, 0] == 0
    assert target.min() >= 0 and target.max() <= 1
    soft = (target > 0.01) & (target < 0.99) & free
    assert soft.any()  # a posterior, not the truth
    assert not torch.equal(target, truth)
    # cells the team sees right now are exact
    assert torch.all(target[truth.bool()] > 0)
    rows = [json.loads(x) for x in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert all(np.isfinite(r["belief_loss"]) for r in rows)
    # the loss really uses the posterior: it differs from the truth-target run
    truth_rows = first_updates(small("d_diabl_coef10"), tmp_path / "truth")
    assert rows[-1]["belief_loss"] != pytest.approx(truth_rows[-1]["belief_loss"], rel=1e-3)


def test_replay_stores_soft_target(tmp_path):
    trainer = MAPPOTrainer(small("d_diabl_coef10", map_target="posterior"), "cpu",
                           run_dir=tmp_path)  # fmt: skip
    trainer.train()
    replay = trainer.replay
    assert len(replay) > 0 and replay.map_target.dtype == torch.float16
    batch = replay.sample(2, "cpu")
    values = batch.map_target
    assert values is not None and ((values > 0.01) & (values < 0.99)).any()


def test_eval_scores_filter_reference():
    from dynabelief.envs.pursuit import PursuitEnv
    from dynabelief.evaluation.evaluate import MapScores

    config = small("d_diabl_coef10")
    env = PursuitEnv(config.env)
    scores = MapScores()
    rng = np.random.default_rng(0)
    try:
        actor, priv = env.reset(seed=7)
        for _ in range(30):
            free = priv.global_state[0] == 0
            belief = np.where(free, 30 / free.sum(), 0.0)[None].repeat(env.n_agents, 0)
            scores.update(belief, priv, actor, config.env.obs_range)
            result = env.step(rng.integers(env.n_actions, size=env.n_agents))
            actor, priv = result.actor, result.privileged
    finally:
        env.close()
    record = scores.record()
    assert "map_bce_remote_filter" in record and "map_bce_filter" in record
    # the teacher knows more than the prior about cells outside one's own view
    assert record["map_bce_remote_filter"] < record["map_bce_remote_prior"]


if __name__ == "__main__":  # record the reference (only from the pre-C′ code)
    import tempfile

    out = {}
    for arm in FROZEN:
        with tempfile.TemporaryDirectory() as tmp:
            out[arm] = first_updates(small(arm), Path(tmp))
    REFERENCE.write_text(json.dumps(out, indent=1, sort_keys=True))
    print("wrote", REFERENCE, {k: len(v) for k, v in out.items()})
