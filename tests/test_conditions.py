"""Evaluation conditions: channel models, send budgets, streams (evaluation only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dynabelief.evaluation.conditions import Condition, EvalChannel, stream_episode_seeds

N = 8
ALIVE = np.ones(N, dtype=bool)
EYE = np.eye(N, dtype=bool)


@pytest.mark.parametrize(
    "label",
    ["iid-p0", "iid-p0.2", "iid-p1", "ge-p0.8-b20", "ge-p0.5-b5", "iid-p0-random-r0.1",
     "iid-p0-periodic-r0.1", "iid-p0-trigger-t3.16", "iid-p0.2-n6", "iid-p0.2-n10",
     "iid-p0-greedy", "ge-p0.8-b50-random-r0.05-n6"],
)  # fmt: skip
def test_label_roundtrip(label):
    assert Condition.parse(label).label == label


def test_parse_values():
    c = Condition.parse("ge-p0.8-b20")
    assert c.loss == 0.8 and c.burst == 20 and c.budget is None
    c = Condition.parse("iid-p0-trigger-t3.16")
    assert c.budget == "trigger" and c.tau == 3.16
    with pytest.raises(ValueError):
        Condition.parse("iid-p1.5")
    with pytest.raises(ValueError):
        Condition.parse("bogus")


def _run(label, steps, beliefs=None, seed=0):
    ch = EvalChannel(N, Condition.parse(label), np.random.default_rng(seed))
    masks = []
    for t in range(steps):
        b = None if beliefs is None else beliefs(t)
        masks.append(ch.sample_delivery(ALIVE, b))
    return ch, np.stack(masks)


def test_no_self_messages_and_dead_agents():
    ch = EvalChannel(N, Condition.parse("iid-p0"), np.random.default_rng(0))
    alive = ALIVE.copy()
    alive[3] = False
    d = ch.sample_delivery(alive, None)
    assert not d[EYE].any() and not d[3].any() and not d[:, 3].any()
    assert d[~EYE & alive[:, None] & alive[None, :]].all()


def test_iid_drop_rate():
    _, m = _run("iid-p0.2", 2000)
    links = m[:, ~EYE]
    assert abs(1 - links.mean() - 0.2) < 0.01


def test_ge_stationary_and_burst():
    _, m = _run("ge-p0.8-b20", 40000)
    bad = ~m[:, ~EYE].any(axis=1)  # team-wide: a bad step drops every link
    assert abs(bad.mean() - 0.8) < 0.03
    runs, length = [], 0
    for b in bad:
        if b:
            length += 1
        elif length:
            runs.append(length)
            length = 0
    assert abs(np.mean(runs) - 20) / 20 < 0.1


def test_random_budget_rate():
    ch, m = _run("iid-p0-random-r0.1", 5000)
    assert abs(ch.send_rate - 0.1) < 0.01
    # a sender either reaches every receiver or none: sending is per sender
    sent = m.any(axis=1)  # [T, N_send]
    assert np.array_equal(m.sum(axis=1), sent * (N - 1))


def test_periodic_exact_rate():
    ch, m = _run("iid-p0-periodic-r0.1", 100)
    assert ch.send_rate == pytest.approx(0.1)
    assert m.any(axis=1).sum(axis=0).tolist() == [10] * N  # every sender exactly 10 times


def test_trigger_sends_on_change():
    still = np.full((N, 2), 0.5, dtype=np.float32)
    ch, m = _run("iid-p0-trigger-t3", 20, beliefs=lambda t: still)
    sends = m.any(axis=1).sum(axis=0)
    assert sends.tolist() == [1] * N  # only the first message, then nothing changes

    def jump(t):
        b = still.copy()
        b[0] = 0.5 + (0.3 if t >= 10 else 0.0)  # 0.3 * 15 cells = 4.5 > 3
        return b

    ch, m = _run("iid-p0-trigger-t3", 20, beliefs=jump)
    sends = m.any(axis=1).sum(axis=0)
    assert sends[0] == 2 and sends[1:].tolist() == [1] * (N - 1)


def test_trigger_needs_beliefs():
    ch = EvalChannel(N, Condition.parse("iid-p0-trigger-t3"), np.random.default_rng(0))
    with pytest.raises(ValueError, match="belief"):
        ch.sample_delivery(ALIVE, None)


def test_same_drops_across_policies():
    """Common random numbers: the drop pattern must not depend on what the policy believes."""
    a = lambda t: np.full((N, 2), 0.1 * (t % 3), dtype=np.float32)  # noqa: E731
    b = lambda t: np.zeros((N, 2), dtype=np.float32)  # noqa: E731
    for label in ("iid-p0.5", "ge-p0.8-b5", "iid-p0.2-random-r0.2"):
        _, m1 = _run(label, 200, beliefs=a, seed=3)
        _, m2 = _run(label, 200, beliefs=b, seed=3)
        assert np.array_equal(m1, m2), label


def test_streams_disjoint():
    seeds = {s: set(stream_episode_seeds(s, 200)) for s in ("eval", "calib", "test")}
    assert not seeds["calib"] & seeds["test"]
    assert not seeds["eval"] & (seeds["calib"] | seeds["test"])
    assert stream_episode_seeds("test", 5) == stream_episode_seeds("test", 5)


PILOT_A = Path("runs/sagemaker/dynabelief-point-diabl-s0-20261010-172552/runs/"
               "point_diabl_s0_20261010-173035/checkpoints/final.pt")  # fmt: skip


@pytest.mark.skipif(not PILOT_A.exists(), reason="pilot checkpoint not present")
def test_load_pilot_point_checkpoint():
    from dynabelief.config import load_config
    from dynabelief.evaluation.evaluate import load_policy_for_eval

    config = load_config("configs/point_diabl.yaml")
    policy, data = load_policy_for_eval(PILOT_A, config, "cpu")
    assert policy.actor.belief_head is not None and data["env_steps"] > 4_000_000


def _comm_config():
    from dynabelief.config import config_from_dict

    return config_from_dict({
        "experiment": {"name": "t", "device": "cpu", "seed": 5},
        "env": {"max_cycles": 15},
        "comm": {"enabled": True},
        "train": {"policy": "mappo"},
    })  # fmt: skip


def test_evaluate_with_condition_uses_stream_and_records_condition(tmp_path):
    import json

    from dynabelief.evaluation.evaluate import RandomAgent, evaluate

    config = _comm_config()
    cond = Condition.parse("ge-p0.5-b5-random-r0.5")
    summary = evaluate(config, RandomAgent(5, 0), 3, tmp_path, condition=cond, stream="calib")
    rows = [json.loads(x) for x in (tmp_path / "episodes.jsonl").read_text().splitlines()]
    assert [r["episode_seed"] for r in rows] == stream_episode_seeds("calib", 3)
    assert all(r["condition"] == cond.label and r["stream"] == "calib" for r in rows)
    assert summary["condition"] == cond.label and summary["episodes"] == 3
    assert 0.3 < summary["send_rate"] < 0.7
    # the training seed does not change the evaluation episodes of a stream
    other = _comm_config().to_dict()
    other["experiment"]["seed"] = 6
    from dynabelief.config import config_from_dict

    evaluate(config_from_dict(other), RandomAgent(5, 0), 3, tmp_path / "b", condition=cond,
             stream="calib")  # fmt: skip
    rows_b = [json.loads(x) for x in (tmp_path / "b" / "episodes.jsonl").read_text().splitlines()]
    assert [r["episode_seed"] for r in rows_b] == [r["episode_seed"] for r in rows]
    assert [r["messages_delivered"] for r in rows_b] == [r["messages_delivered"] for r in rows]
