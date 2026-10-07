"""Evaluation entry point (fixed evaluation seed list, raw per-episode records).

Examples:
    # random-policy reference
    python scripts/evaluate.py --config configs/smoke.yaml --episodes 5 --packet-loss 0.2
    # trained checkpoint, using the config stored inside it
    python scripts/evaluate.py --checkpoint runs/<run>/checkpoints/final.pt --episodes 20
    # generalization: same checkpoint, different pursuer count
    python scripts/evaluate.py --checkpoint runs/<run>/checkpoints/final.pt --set env.n_pursuers=6
"""

from __future__ import annotations

import argparse
import json
import sys

from dynabelief.config import config_with_overrides, load_config, parse_override
from dynabelief.evaluation.evaluate import evaluate_checkpoint, evaluate_random
from dynabelief.utils.checkpointing import load_checkpoint
from dynabelief.utils.device import resolve_device
from dynabelief.utils.logging import create_run_dir, get_logger, run_metadata, write_yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DynaBelief-Comm evaluation on a fixed seed list.",
    )
    parser.add_argument(
        "--config", help="YAML config (optional with --checkpoint: its stored config is used)"
    )
    parser.add_argument("--checkpoint", help="trained MAPPO checkpoint (.pt)")
    parser.add_argument("--episodes", type=int, default=10, help="number of evaluation episodes")
    parser.add_argument("--packet-loss", type=float, help="override comm.packet_loss")
    parser.add_argument(
        "--deterministic", action="store_true", help="greedy actions instead of sampling"
    )
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
    if args.episodes < 1:
        parser.error("--episodes must be >= 1")
    if not args.config and not args.checkpoint:
        parser.error("give --config, --checkpoint, or both")
    overrides = dict(parse_override(text) for text in args.overrides)
    if args.packet_loss is not None:
        overrides["comm.packet_loss"] = args.packet_loss
    if args.output_dir:
        overrides["experiment.output_dir"] = args.output_dir
    if args.config:
        config = load_config(args.config, overrides)
    else:
        config = config_with_overrides(load_checkpoint(args.checkpoint)["config"], overrides)
    device = resolve_device(config.experiment.device)

    log = get_logger()
    run_dir = create_run_dir(
        config.experiment.output_dir, f"eval_{config.experiment.name}", config.experiment.seed
    )
    write_yaml(run_dir / "config.yaml", config.to_dict())
    write_yaml(run_dir / "metadata.yaml", run_metadata(device, argv))
    if args.checkpoint:
        summary = evaluate_checkpoint(
            config, args.checkpoint, args.episodes, device, args.deterministic, run_dir
        )
    else:
        summary = evaluate_random(config, args.episodes, run_dir)
    with open(run_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    log.info("run directory: %s", run_dir)
    log.info("summary: %s", json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
