"""Train entry point.

M1 status: only ``train.policy: random`` exists, so this runs a smoke rollout
that checks the data path and does NOT learn. MAPPO arrives in M2.

Example:
    python scripts/train.py --config configs/smoke.yaml
    python scripts/train.py --config configs/smoke.yaml --set experiment.seed=3
"""

from __future__ import annotations

import argparse
import sys

from dynabelief.config import load_config, parse_override
from dynabelief.training.rollout import run_random_smoke
from dynabelief.utils.device import resolve_device
from dynabelief.utils.logging import create_run_dir, get_logger, run_metadata, write_yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DynaBelief-Comm training (M1: random-policy smoke rollout only, no learning).",
    )
    parser.add_argument(
        "--config", required=True, help="path to a YAML config, e.g. configs/smoke.yaml"
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
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args = build_parser().parse_args(argv)
    overrides = dict(parse_override(text) for text in args.overrides)
    if args.output_dir:
        overrides["experiment.output_dir"] = args.output_dir
    config = load_config(args.config, overrides)
    device = resolve_device(config.experiment.device)

    log = get_logger()
    run_dir = create_run_dir(
        config.experiment.output_dir, config.experiment.name, config.experiment.seed
    )
    write_yaml(run_dir / "config.yaml", config.to_dict())
    write_yaml(run_dir / "metadata.yaml", run_metadata(device, argv))
    log.info("run directory: %s", run_dir)
    log.info(
        "policy=%s device=%s (random policy: no learning happens)", config.train.policy, device
    )

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


if __name__ == "__main__":
    sys.exit(main())
