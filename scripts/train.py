"""Train entry point.

``train.policy: random`` runs the M1 smoke rollout (data path only, NO learning).
``train.policy: mappo`` trains the No-Communication MAPPO baseline (M2).

Examples:
    python scripts/train.py --config configs/smoke.yaml
    python scripts/train.py --config configs/smoke_mappo.yaml
    python scripts/train.py --config configs/no_comm.yaml --set experiment.seed=1
    python scripts/train.py --config configs/no_comm.yaml \\
        --resume runs/<run>/checkpoints/latest.pt --set train.total_env_steps=20000000
"""

from __future__ import annotations

import argparse
import json
import sys

from dynabelief.config import load_config, parse_override
from dynabelief.training.rollout import run_random_smoke
from dynabelief.utils.device import resolve_device
from dynabelief.utils.logging import create_run_dir, get_logger, run_metadata, write_yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DynaBelief-Comm training (random smoke rollout or No-Comm MAPPO).",
    )
    parser.add_argument(
        "--config", required=True, help="path to a YAML config, e.g. configs/no_comm.yaml"
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override a config value (repeatable), e.g. --set experiment.seed=3",
    )
    parser.add_argument("--output-dir", help="override experiment.output_dir")
    parser.add_argument(
        "--resume", metavar="CHECKPOINT", help="continue a MAPPO run from a checkpoint"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    args = parser.parse_args(argv)
    overrides = dict(parse_override(text) for text in args.overrides)
    if args.output_dir:
        overrides["experiment.output_dir"] = args.output_dir
    config = load_config(args.config, overrides)
    if args.resume and config.train.policy != "mappo":
        parser.error("--resume requires train.policy: mappo")
    device = resolve_device(config.experiment.device)

    log = get_logger()
    run_dir = create_run_dir(
        config.experiment.output_dir, config.experiment.name, config.experiment.seed
    )
    write_yaml(run_dir / "config.yaml", config.to_dict())
    write_yaml(
        run_dir / "metadata.yaml", {**run_metadata(device, argv), "resumed_from": args.resume}
    )
    log.info("run directory: %s", run_dir)

    if config.train.policy == "random":
        log.info("policy=random device=%s (random policy: no learning happens)", device)
        summary = run_random_smoke(config, run_dir)
        log.info(
            "done: %d env steps, %d episodes, %.0f steps/s, mean capture rate %.3f, digest %s",
            summary["env_steps"],
            summary["episodes"],
            summary["fps"],
            summary.get("mean_capture_rate", float("nan")),
            summary["trajectory_digest"][:16],
        )
        return 0

    from dynabelief.training.trainer import MAPPOTrainer

    trainer = MAPPOTrainer(config, device, run_dir, resume_from=args.resume)
    summary = trainer.train()
    with open(run_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    log.info(
        "done: %d updates, %d env steps, %.0f steps/s, final checkpoint %s",
        summary["updates"],
        summary["env_steps"],
        summary["fps"],
        summary["final_checkpoint"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
