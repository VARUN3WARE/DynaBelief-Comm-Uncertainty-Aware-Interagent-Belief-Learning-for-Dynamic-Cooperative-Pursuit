"""Fingerprint a short training run, to prove that new code leaves frozen arms unchanged.

Trains ``--updates`` PPO updates of a config in-process (env workers do not change
results, D9) on one device and hashes the final weights, optimizer states and the
logged losses. Run it at the freeze commit and at any later commit: equal hashes mean
the later code trains that config bit-identically, so confirmatory runs launched from
different commits stay comparable (pre-registration, section 4).

    python scripts/regression_hash.py --config configs/point_diabl.yaml --updates 2
    python scripts/regression_hash.py --all --output tests/data/regression_hashes.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import torch

FROZEN = ("configs/tarmac_pos.yaml", "configs/point_diabl.yaml", "configs/d_diabl_coef10.yaml")
LOSS_KEYS = (
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "belief_loss",
    "replay_belief_loss",
)


def _update_tensor(h, obj) -> None:
    if isinstance(obj, torch.Tensor):
        h.update(obj.detach().cpu().contiguous().numpy().tobytes())
    elif isinstance(obj, dict):
        for key in sorted(obj, key=str):
            h.update(str(key).encode())
            _update_tensor(h, obj[key])
    elif isinstance(obj, list | tuple):
        for item in obj:
            _update_tensor(h, item)
    else:
        h.update(repr(obj).encode())


def fingerprint(config_path: str, updates: int, device: str, threads: int) -> dict:
    from dynabelief.config import load_config
    from dynabelief.training.trainer import MAPPOTrainer
    from dynabelief.utils.logging import read_jsonl

    torch.set_num_threads(threads)
    probe = load_config(config_path)
    steps = updates * probe.train.num_envs * probe.train.rollout_length
    overrides = {"train.total_env_steps": steps, "train.num_workers": 0,
                 "train.torch_threads": threads, "train.snapshot_every_updates": 0}  # fmt: skip
    config = load_config(config_path, overrides)
    with tempfile.TemporaryDirectory() as tmp:
        trainer = MAPPOTrainer(config, device, run_dir=Path(tmp))
        trainer.train()
        h = hashlib.sha256()
        _update_tensor(h, trainer.algo.state_dict())
        losses = [[m.get(k) for k in LOSS_KEYS] for m in read_jsonl(Path(tmp) / "metrics.jsonl")]
        h.update(json.dumps(losses).encode())
    return {"config": config_path, "updates": updates, "device": device, "threads": threads,
            "torch": torch.__version__, "cpu_model": cpu_model(), "sha256": h.hexdigest(),
            "losses": losses}  # fmt: skip


def cpu_model() -> str:
    """Floating-point results can differ across CPU models, so hashes are per machine."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    import platform

    return platform.processor() or "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--all", action="store_true", help=f"the frozen arms {FROZEN}")
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", help="write results as JSON")
    args = parser.parse_args(argv)
    configs = list(FROZEN) if args.all else args.config
    if not configs:
        parser.error("give --config or --all")
    results = [fingerprint(c, args.updates, args.device, args.threads) for c in configs]
    for r in results:
        print(f"{r['config']:32s} {r['sha256'][:16]}  ({r['updates']} updates, {r['device']})")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(results, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
