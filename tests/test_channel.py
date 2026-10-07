from __future__ import annotations

import numpy as np
import pytest

from dynabelief.comm.channel import PacketLossChannel


def make(n=8, loss=0.2, seed=0, dim=16):
    return PacketLossChannel(n, loss, np.random.default_rng(seed), message_dim=dim)


def test_no_self_messages():
    mask = make(loss=0.0).sample_delivery()
    assert mask.shape == (8, 8)
    assert not mask.diagonal().any()
    assert mask.sum() == 8 * 7


def test_full_loss_delivers_nothing():
    channel = make(loss=1.0)
    assert not channel.sample_delivery().any()
    assert channel.stats.dropped == channel.stats.sent == 56


def test_drop_rate_matches_probability():
    channel = make(loss=0.2, seed=1)
    for _ in range(2000):
        channel.sample_delivery()
    assert channel.stats.drop_rate == pytest.approx(0.2, abs=0.01)
    assert channel.stats.sent == channel.stats.delivered + channel.stats.dropped


def test_same_seed_same_pattern():
    a, b = make(seed=3), make(seed=3)
    for _ in range(10):
        np.testing.assert_array_equal(a.sample_delivery(), b.sample_delivery())


def test_dead_agents_neither_send_nor_receive():
    alive = np.ones(8, dtype=bool)
    alive[2] = False
    mask = make(loss=0.0).sample_delivery(alive)
    assert not mask[2].any() and not mask[:, 2].any()
    assert mask.sum() == 7 * 6


def test_loss_pattern_independent_of_alive_mask():
    """Agent deaths must not shift the random stream seen by the surviving links."""
    alive = np.ones(8, dtype=bool)
    alive[5] = False
    full, partial = make(seed=9), make(seed=9)
    for _ in range(5):
        m_full = full.sample_delivery()
        m_partial = partial.sample_delivery(alive)
        keep = alive[:, None] & alive[None, :]
        np.testing.assert_array_equal(m_full & keep, m_partial)


def test_batched_masks_and_step_count():
    channel = make(loss=0.0)
    mask = channel.sample_delivery(np.ones((4, 8), dtype=bool))
    assert mask.shape == (4, 8, 8)
    assert channel.stats.steps == 4
    assert channel.stats.sent == 4 * 56


def test_bytes_accounting():
    channel = make(loss=0.0, dim=16)
    channel.sample_delivery()
    assert channel.stats.bytes_sent == 56 * 16 * 4
    stats = channel.stats.as_dict()
    assert stats["messages_sent_per_step"] == 56
    old = channel.reset_stats()
    assert old.sent == 56 and channel.stats.sent == 0


def test_invalid_arguments():
    with pytest.raises(ValueError):
        make(loss=1.1)
    with pytest.raises(ValueError):
        make(n=0)
    with pytest.raises(ValueError):
        make().sample_delivery(np.ones(5, dtype=bool))
