"""Measure raw Pursuit simulator throughput with a random policy.

Use this to size training runs: total env steps / steps-per-second gives a
lower bound on wall time before any network cost is added.

Example:
    python scripts/benchmark_env.py --config configs/default.yaml --steps 5000
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from dynabelief.config import load_config
from dynabelief.envs.pursuit import PursuitEnv
from dynabelief.utils.seeding import derive_seed, make_rng


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark single-process Pursuit steps/second.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=5000, help="environment steps to time")
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("--steps must be >= 1")

    config = load_config(args.config)
    env = PursuitEnv(config.env)
    rng = make_rng(config.experiment.seed, "benchmark")
    episode = 0
    env.reset(derive_seed(config.experiment.seed, "benchmark", episode))
    start = time.perf_counter()
    for _ in range(args.steps):
        result = env.step(rng.integers(0, env.n_actions, size=env.n_agents))
        if result.done:
            episode += 1
            env.reset(derive_seed(config.experiment.seed, "benchmark", episode))
    elapsed = time.perf_counter() - start
    env.close()

    sps = args.steps / elapsed
    agent_sps = sps * env.n_agents
    print(
        f"env: {config.env.x_size}x{config.env.y_size}, {env.n_agents} pursuers, "
        f"{config.env.n_evaders} evaders, max_cycles={config.env.max_cycles}"
    )
    print(
        f"{args.steps} steps in {elapsed:.2f}s -> {sps:.0f} env steps/s "
        f"({agent_sps:.0f} agent steps/s) single process; episodes finished: {episode}"
    )
    for target in (1e6, 5e6, 10e6):
        hours = target / sps / 3600
        print(f"  {target / 1e6:>4.0f}M env steps ~ {hours:6.2f} h simulator time (1 process)")
    np.testing.assert_(sps > 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
