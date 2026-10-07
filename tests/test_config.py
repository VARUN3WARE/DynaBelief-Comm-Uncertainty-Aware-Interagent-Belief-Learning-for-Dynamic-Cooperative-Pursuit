from __future__ import annotations

import pytest

from dynabelief.config import (
    Config,
    ConfigError,
    config_from_dict,
    load_config,
    parse_override,
)
from tests.helpers import REPO_ROOT


def test_defaults_are_valid():
    config = config_from_dict({})
    assert config == Config()
    assert config.env.x_size == 16 and config.env.n_pursuers == 8
    assert config.env.obs_range == 7 and config.env.max_cycles == 500


@pytest.mark.parametrize("name", ["smoke.yaml", "default.yaml"])
def test_shipped_configs_load(name):
    config = load_config(REPO_ROOT / "configs" / name)
    config.validate()


def test_unknown_section_rejected():
    with pytest.raises(ConfigError, match="unknown top-level"):
        config_from_dict({"modle": {}})


def test_unknown_key_rejected():
    with pytest.raises(ConfigError, match="unknown key"):
        config_from_dict({"env": {"n_pursuer": 8}})


@pytest.mark.parametrize(
    "raw",
    [
        {"env": {"x_size": "16"}},
        {"env": {"x_size": 16.0}},
        {"env": {"x_size": True}},
        {"env": {"surround": 1}},
        {"comm": {"packet_loss": "0.1"}},
        {"experiment": {"name": 3}},
        {"env": []},
    ],
)
def test_wrong_types_rejected(raw):
    with pytest.raises(ConfigError):
        config_from_dict(raw)


def test_int_accepted_for_float():
    config = config_from_dict({"comm": {"packet_loss": 0}})
    assert isinstance(config.comm.packet_loss, float)


@pytest.mark.parametrize(
    "raw",
    [
        {"comm": {"packet_loss": 1.5}},
        {"comm": {"packet_loss": -0.1}},
        {"comm": {"delay_steps": 0}},
        {"env": {"obs_range": 6}},
        {"env": {"n_pursuers": 0}},
        {"env": {"x_size": 4, "y_size": 4, "n_pursuers": 8, "n_evaders": 30}},
        {"train": {"num_envs": 0}},
        {"train": {"policy": "mappo"}},
        {"experiment": {"seed": -1}},
        {"experiment": {"device": "gpu"}},
    ],
)
def test_invalid_values_rejected(raw):
    with pytest.raises(ConfigError):
        config_from_dict(raw)


def test_overrides_applied(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("experiment:\n  seed: 1\n")
    config = load_config(path, {"experiment.seed": 7, "comm.packet_loss": 0.3})
    assert config.experiment.seed == 7
    assert config.comm.packet_loss == pytest.approx(0.3)


def test_override_unknown_key_rejected(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("{}\n")
    with pytest.raises(ConfigError):
        load_config(path, {"env.not_a_key": 1})


def test_parse_override():
    assert parse_override("experiment.seed=3") == ("experiment.seed", 3)
    assert parse_override("comm.enabled=false") == ("comm.enabled", False)
    assert parse_override("experiment.name=abc") == ("experiment.name", "abc")
    with pytest.raises(ConfigError):
        parse_override("experiment.seed")


def test_config_is_frozen():
    config = Config()
    with pytest.raises(AttributeError):
        config.env.x_size = 3  # type: ignore[misc]


def test_duplicate_yaml_keys_rejected(tmp_path):
    path = tmp_path / "dup.yaml"
    path.write_text("env:\n  x_size: 16\n  x_size: 12\n")
    with pytest.raises(ConfigError, match="duplicate key"):
        load_config(path)
    path.write_text("env:\n  x_size: 12\nexperiment:\n  seed: 1\nenv:\n  y_size: 12\n")
    with pytest.raises(ConfigError, match="duplicate key"):
        load_config(path)


def test_null_section_accepts_override(tmp_path):
    path = tmp_path / "null.yaml"
    path.write_text("experiment:\nenv:\n")
    assert load_config(path, {"experiment.seed": 4}).experiment.seed == 4


@pytest.mark.parametrize("value", [".inf", "-.inf", ".nan"])
def test_non_finite_floats_rejected(tmp_path, value):
    path = tmp_path / "inf.yaml"
    path.write_text(f"env:\n  catch_reward: {value}\n")
    with pytest.raises(ConfigError, match="finite"):
        load_config(path)


@pytest.mark.parametrize(
    "raw",
    [
        {"experiment": {"name": "../escape"}},
        {"experiment": {"name": "a/b"}},
        {"experiment": {"name": " "}},
        {"experiment": {"device": "cuda:x"}},
        {"experiment": {"device": "cuda:"}},
        {"experiment": {"seed": 2**63}},
    ],
)
def test_experiment_identity_values_rejected(raw):
    with pytest.raises(ConfigError):
        config_from_dict(raw)


@pytest.mark.parametrize("device", ["auto", "cpu", "cuda", "cuda:0", "cuda:12"])
def test_valid_devices_accepted(device):
    assert config_from_dict({"experiment": {"device": device}}).experiment.device == device


@pytest.mark.parametrize(
    "env",
    [
        {"x_size": 8, "y_size": 8},  # 30 evaders cannot be spread on 49 free cells
        {"n_evaders": 100},
        {"constraint_window": 0.5},  # worst-case window is mostly the obstacle
        {"constraint_window": 0.1},
        {"constraint_window": 0.0},  # PettingZoo crashes with low >= high
    ],
)
def test_unspawnable_layouts_rejected(env):
    """These configs make PettingZoo's rejection sampler hang or crash at construction."""
    with pytest.raises(ConfigError):
        config_from_dict({"env": env})


def test_spawn_windows_match_pettingzoo_reset_arithmetic():
    import numpy as np

    from dynabelief.config import _spawn_windows

    rng = np.random.default_rng(0)
    for size, window in [(16, 1.0), (16, 0.75), (13, 0.4), (20, 0.33)]:
        windows = _spawn_windows(size, window)
        for u in rng.uniform(0.0, 1.0 - window, size=2000):
            assert (int(size * u), int(size * (u + window))) in windows
