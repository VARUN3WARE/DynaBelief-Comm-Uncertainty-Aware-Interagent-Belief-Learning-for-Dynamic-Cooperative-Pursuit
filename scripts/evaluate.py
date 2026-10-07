"""Evaluation entry point.

M1 status: only the random policy can be evaluated. ``--checkpoint`` is
reserved for M2+ and is rejected for now instead of silently ignored.

Example:
    python scripts/evaluate.py --config configs/smoke.yaml --episodes 5 --packet-loss 0.2
"""

from __future__ import annotations

import argparse
import json
import sys

from dynabelief.config import load_config, parse_override
from dynabelief.evaluation.evaluate import evaluate_random
from dynabelief.utils.device import resolve_device
from dynabelief.utils.logging import create_run_dir, get_logger, run_metadata, write_yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DynaBelief-Comm evaluation on a fixed seed list (M1: random policy only).",
    )
    parser.add_argument("--config", required=True, help="path to a YAML config")
    parser.add_argument("--checkpoint", help="trained checkpoint (available from M2)")
    parser.add_argument("--episodes", type=int, default=10, help="number of evaluation episodes")
    parser.add_argument("--packet-loss", type=float, help="override comm.packet_loss")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override a config value (repeatable)",
    )
    parser.add_argument("--output-dir", help="override experiment.output_dir")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.checkpoint:
        parser.error("--checkpoint is not supported until MAPPO lands (M2)")
    if args.episodes < 1:
        parser.error("--episodes must be >= 1")
    overrides = dict(parse_override(text) for text in args.overrides)
    if args.packet_loss is not None:
        overrides["comm.packet_loss"] = args.packet_loss
    if args.output_dir:
        overrides["experiment.output_dir"] = args.output_dir
    config = load_config(args.config, overrides)
    device = resolve_device(config.experiment.device)

    log = get_logger()
    run_dir = create_run_dir(
        config.experiment.output_dir, f"eval_{config.experiment.name}", config.experiment.seed
    )
    write_yaml(run_dir / "config.yaml", config.to_dict())
    write_yaml(run_dir / "metadata.yaml", run_metadata(device, argv))
    summary = evaluate_random(config, args.episodes, run_dir)
    with open(run_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    log.info("run directory: %s", run_dir)
    log.info("summary: %s", json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
