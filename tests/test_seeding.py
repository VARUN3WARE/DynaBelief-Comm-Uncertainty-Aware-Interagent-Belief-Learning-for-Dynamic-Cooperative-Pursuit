from __future__ import annotations

import numpy as np
import pytest
import torch

from dynabelief.utils.seeding import derive_seed, make_rng, seed_everything


def test_derive_seed_is_deterministic_and_keyed():
    assert derive_seed(0, "train", 1, 2) == derive_seed(0, "train", 1, 2)
    distinct = {
        derive_seed(0, "train", 1, 2),
        derive_seed(1, "train", 1, 2),
        derive_seed(0, "eval", 1, 2),
        derive_seed(0, "train", 2, 1),
        derive_seed(0, "train", 1),
    }
    assert len(distinct) == 5


def test_train_and_eval_episode_seeds_do_not_overlap():
    train = {derive_seed(0, "train", e, k) for e in range(16) for k in range(500)}
    evaluation = {derive_seed(0, "eval", 0, k) for k in range(1000)}
    assert not train & evaluation


def test_make_rng_streams():
    a1 = make_rng(5, "policy").random(8)
    a2 = make_rng(5, "policy").random(8)
    b = make_rng(5, "comm").random(8)
    np.testing.assert_array_equal(a1, a2)
    assert not np.array_equal(a1, b)


def test_negative_seed_rejected():
    with pytest.raises(ValueError):
        derive_seed(-1, "train")
    with pytest.raises(ValueError):
        seed_everything(-1)


def test_seed_everything_reproduces_torch_and_numpy():
    seed_everything(123)
    t1, n1 = torch.rand(4), np.random.rand(4)
    seed_everything(123)
    t2, n2 = torch.rand(4), np.random.rand(4)
    assert torch.equal(t1, t2)
    np.testing.assert_array_equal(n1, n2)


def test_derived_seeds_use_64_bits_and_reset_accepts_them():
    from dynabelief.config import EnvConfig
    from dynabelief.envs.pursuit import PursuitEnv

    seeds = [derive_seed(0, "train", 0, k) for k in range(64)]
    assert max(seeds) >= 2**32 and all(0 <= s < 2**64 for s in seeds)
    env = PursuitEnv(EnvConfig(max_cycles=2))
    a, _ = env.reset(max(seeds))
    b, _ = env.reset(max(seeds))
    np.testing.assert_array_equal(a.obs, b.obs)
