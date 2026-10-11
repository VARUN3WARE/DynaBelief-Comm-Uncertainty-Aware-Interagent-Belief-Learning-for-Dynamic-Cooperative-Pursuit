"""Visualize point beliefs on the grid for one held-out evaluation episode.

Each panel is one time step: obstacles, pursuers, evaders (team-visible ones
highlighted), every agent's predicted evader position, its label (the nearest
team-visible evader), and a thin line from prediction to label (the error).
Coordinates are the simulator's (x, y); row y is drawn top-down, as in the local views.

Example:
    python scripts/plot_beliefs.py --checkpoint runs/<run>/checkpoints/final.pt --steps 10 100 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.patches

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Reference palette (dataviz skill): surface, text, two categorical slots, neutral greys.
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
PREDICTION, LABEL, EVADER, PURSUER, OBSTACLE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#d9d8d3"


def plot_maps(panels: list, config, args, summary: dict) -> int:
    """One agent's occupancy belief per panel, with the true evaders and its own view."""
    X, Y = config.env.x_size, config.env.y_size
    off = config.env.obs_range // 2
    agent = args.agent
    fig, axes = plt.subplots(1, len(panels), figsize=(3.7 * len(panels), 4.1), facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    image = None
    for ax, frame in zip(axes, panels, strict=True):
        priv = frame["privileged"]
        belief = np.where(priv.global_state[0] > 0, np.nan, frame["belief"][agent])
        ax.set_facecolor(OBSTACLE)  # NaN (obstacle) cells show the surface behind
        image = ax.imshow(belief, cmap="Blues", vmin=0, vmax=1, origin="upper",
                          extent=(-0.5, X - 0.5, Y - 0.5, -0.5))  # fmt: skip
        alive = priv.evader_alive
        ax.scatter(priv.evader_pos[alive, 0], priv.evader_pos[alive, 1], s=22, marker="o",
                   facecolors="none", edgecolors=LABEL, linewidths=1.5,
                   label="true evader")  # fmt: skip
        others = np.arange(len(priv.pursuer_pos)) != agent
        ax.scatter(priv.pursuer_pos[others, 0], priv.pursuer_pos[others, 1], s=36, marker="^",
                   color=PURSUER, linewidths=0, label="other pursuers")  # fmt: skip
        px, py = priv.pursuer_pos[agent]
        ax.scatter([px], [py], s=70, marker="^", color=TEXT, linewidths=0, label=f"agent {agent}")
        ax.add_patch(matplotlib.patches.Rectangle((px - off - 0.5, py - off - 0.5), 2 * off + 1,
                     2 * off + 1, fill=False, edgecolor=TEXT, linewidth=1, linestyle=(0, (3, 2)),
                     label="own view"))  # fmt: skip
        ax.set_title(f"t = {frame['t']}", loc="left", fontsize=9, color=TEXT)
        ax.set_xlim(-0.5, X - 0.5)
        ax.set_ylim(Y - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.tick_params(colors=TEXT_2, labelsize=7)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.08), ncol=4, fontsize=7.5,
                   frameon=False, labelcolor=TEXT)  # fmt: skip
    bar = fig.colorbar(image, ax=list(axes), fraction=0.015, pad=0.01)
    bar.set_label("P(evader in cell)", color=TEXT_2, fontsize=8)
    bar.ax.tick_params(colors=TEXT_2, labelsize=7)
    run_name = args.title or config.experiment.name
    fig.suptitle(f"Occupancy belief of agent {agent}, held-out episode 0 ({run_name}, packet loss "
                 f"{args.packet_loss:g}); episode BCE "
                 f"{summary.get('map_bce_model', float('nan')):.3f} vs prior "
                 f"{summary.get('map_bce_prior', float('nan')):.3f} nats/cell",
                 x=0.01, ha="left", fontsize=10, color=TEXT)  # fmt: skip
    output = args.output or Path(args.checkpoint).parent.parent / "beliefs.png"
    fig.savefig(output, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    from dynabelief.config import config_with_overrides, parse_override
    from dynabelief.evaluation.evaluate import MAPPOAgent, evaluate, load_policy_for_eval
    from dynabelief.utils.checkpointing import load_checkpoint
    from dynabelief.utils.device import resolve_device

    parser = argparse.ArgumentParser(description="Plot point beliefs for one eval episode.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[5, 100, 250, 450])
    parser.add_argument("--packet-loss", type=float, default=0.0)
    parser.add_argument("--output", type=Path, help="PNG (default: next to the checkpoint)")
    parser.add_argument("--title", help="run name for the figure title")
    parser.add_argument("--agent", type=int, default=0, help="agent shown (map beliefs)")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args(argv)

    overrides = dict(parse_override(o) for o in args.overrides)
    overrides["comm.packet_loss"] = args.packet_loss
    config = config_with_overrides(load_checkpoint(args.checkpoint)["config"], overrides)
    device = resolve_device(config.experiment.device)
    policy, _ = load_policy_for_eval(args.checkpoint, config, device)
    if policy.actor.belief_head is None:
        raise SystemExit("this checkpoint has no belief head")

    wanted, frames = set(args.steps), {}

    def capture(step: dict) -> None:
        if step["t"] in wanted:
            frames[step["t"]] = step

    agent = MAPPOAgent(policy, config.experiment.seed, device, deterministic=False)
    summary = evaluate(config, agent, episodes=1, on_step=capture)
    if not frames:
        raise SystemExit(f"episode ended before steps {sorted(wanted)}")

    X, Y = config.env.x_size, config.env.y_size
    scale = np.array([X - 1, Y - 1])
    panels = [frames[t] for t in sorted(frames)]
    if policy.actor.belief_type == "map":
        return plot_maps(panels, config, args, summary)
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 4.0), facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, frame in zip(axes, panels, strict=True):
        priv = frame["privileged"]
        ax.set_facecolor(SURFACE)
        cmap = matplotlib.colors.ListedColormap([SURFACE, OBSTACLE])
        ax.imshow(priv.global_state[0], cmap=cmap, origin="upper",
                  extent=(-0.5, X - 0.5, Y - 0.5, -0.5), vmin=0, vmax=1)  # fmt: skip
        ax.set_xticks(range(0, X, 5))
        ax.set_yticks(range(0, Y, 5))
        ax.tick_params(colors=TEXT_2, labelsize=7)
        for side in ax.spines.values():
            side.set_color(GRID)
        alive = priv.evader_alive
        ex, ey = priv.evader_pos[alive, 0], priv.evader_pos[alive, 1]
        ax.scatter(ex, ey, s=14, color=EVADER, alpha=0.35, linewidths=0, label="evader")
        px, py = priv.pursuer_pos[:, 0], priv.pursuer_pos[:, 1]
        ax.scatter(px, py, s=46, marker="^", color=PURSUER, linewidths=0, label="pursuer")
        valid = frame["valid"]
        pred = frame["belief"] * scale
        target = frame["target"] * scale
        for i in np.flatnonzero(valid):
            ax.plot([pred[i, 0], target[i, 0]], [pred[i, 1], target[i, 1]], color=TEXT_2,
                    linewidth=0.8, alpha=0.7)  # fmt: skip
        ax.scatter(target[valid, 0], target[valid, 1], s=70, facecolors="none",
                   edgecolors=LABEL, linewidths=2, label="label (nearest seen evader)")  # fmt: skip
        ax.scatter(pred[valid, 0], pred[valid, 1], s=40, marker="x", color=PREDICTION,
                   linewidths=2, label="belief")  # fmt: skip
        err = np.linalg.norm(pred[valid] - target[valid], axis=-1)
        title = f"t = {frame['t']}" + (f"   mean error {err.mean():.1f} cells" if len(err) else "")
        ax.set_title(title, loc="left", fontsize=9, color=TEXT)
        ax.set_xlim(-0.5, X - 0.5)
        ax.set_ylim(Y - 0.5, -0.5)
        ax.set_aspect("equal")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.08), ncol=4, fontsize=7.5,
                   frameon=False, labelcolor=TEXT)  # fmt: skip
    run_name = args.title or load_checkpoint(args.checkpoint)["config"]["experiment"]["name"]
    fig.suptitle(f"Point beliefs, held-out episode 0 ({run_name}, "
                 f"packet loss {args.packet_loss:g}); episode mean error "
                 f"{summary.get('belief_error_cells', float('nan')):.1f} cells",
                 x=0.01, ha="left", fontsize=10, color=TEXT)  # fmt: skip
    fig.tight_layout()
    output = args.output or Path(args.checkpoint).parent.parent / "beliefs.png"
    fig.savefig(output, dpi=150, facecolor=SURFACE)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
