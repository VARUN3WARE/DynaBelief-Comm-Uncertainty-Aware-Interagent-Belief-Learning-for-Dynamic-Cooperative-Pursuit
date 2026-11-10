"""Render one evaluation episode as a video: pursuers, views, evaders, messages, beliefs.

Usage::

    PYTHONPATH=src:. .venv/bin/python scripts/record_demo.py \
        --checkpoint "A: point DIABL=runs/.../point_diabl.../final.pt" \
        --checkpoint "ctrl: TarMAC=runs/.../tarmac_pos.../final.pt" \
        --condition iid-p0.2 --episode 0 --out runs/demo/a_vs_ctrl.mp4

Several checkpoints are drawn side by side on the SAME episode (same evader spawns and
drop pattern: test stream, common random numbers). Solid lines = delivered messages,
dashed = dropped; crosses = each pursuer's point belief; a map belief is shown as the
heat map of pursuer 0. Needs ``imageio-ffmpeg`` for mp4 (falls back to GIF).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def collect(checkpoint: Path, condition: str, episode: int, max_steps: int) -> dict:
    import torch

    from dynabelief.config import config_from_dict
    from dynabelief.evaluation.conditions import EVAL_BASE_SEED, Condition, stream_episode_seeds
    from dynabelief.evaluation.evaluate import MAPPOAgent, evaluate, load_policy_for_eval
    from dynabelief.utils.checkpointing import load_checkpoint

    torch.set_num_threads(1)
    cond = Condition.parse(condition)
    raw = load_checkpoint(checkpoint, map_location="cpu")["config"]
    raw["experiment"]["device"] = "cpu"
    if cond.n_pursuers:
        raw["env"]["n_pursuers"] = cond.n_pursuers
    config = config_from_dict(raw)
    policy, _ = load_policy_for_eval(checkpoint, config, "cpu")
    agent = MAPPOAgent(policy, EVAL_BASE_SEED, "cpu", deterministic=cond.greedy)
    frames: list[dict] = []
    captured = 0

    def on_step(s: dict) -> None:
        nonlocal captured
        if s["episode"] != episode or s["t"] >= max_steps:
            return
        priv = s["privileged"]
        frames.append({
            "t": s["t"],
            "obstacles": priv.global_state[0] > 0,
            "pursuers": priv.pursuer_pos.copy(),
            "evaders": priv.evader_pos[priv.evader_alive].copy(),
            "n_alive": int(priv.evader_alive.sum()),
            "belief": None if s["belief"] is None else np.array(s["belief"]),
            "delivery": None if s["delivery"] is None else s["delivery"].copy(),
        })  # fmt: skip

    # Episodes before `episode` must run so the stream's later episode is reproduced exactly.
    with torch.no_grad():
        summary = evaluate(config, agent, episode + 1, None, on_step=on_step, condition=cond,
                           stream="test")  # fmt: skip
    del captured
    seed = stream_episode_seeds("test", episode + 1)[episode]
    return {"frames": frames, "summary": summary, "obs_range": config.env.obs_range,
            "n_evaders": config.env.n_evaders, "episode_seed": seed,
            "grid": (config.env.y_size, config.env.x_size)}  # fmt: skip


def render(panels: list[tuple[str, dict]], condition: str, out: Path, fps: int) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.patches import Rectangle

    try:
        import imageio_ffmpeg

        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        writer, suffix = animation.FFMpegWriter(fps=fps, bitrate=2400), ".mp4"
    except ImportError:
        writer, suffix = animation.PillowWriter(fps=fps), ".gif"
    out = out.with_suffix(suffix)
    n_frames = max(len(p["frames"]) for _, p in panels)
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 5.6), dpi=120)
    axes = np.atleast_1d(axes)
    Y, X = panels[0][1]["grid"]

    def draw(i: int) -> None:
        for ax, (title, data) in zip(axes, panels, strict=True):
            ax.clear()
            frames = data["frames"]
            f = frames[min(i, len(frames) - 1)]
            done = i >= len(frames)
            belief = f["belief"]
            if belief is not None and belief.ndim == 3:  # map belief of pursuer 0
                ax.imshow(1 / (1 + np.exp(-belief[0])) if belief.max() > 1 or belief.min() < 0
                          else belief[0], cmap="Oranges", vmin=0, vmax=0.6,
                          extent=(-0.5, X - 0.5, Y - 0.5, -0.5))  # fmt: skip
            ax.imshow(np.where(f["obstacles"], 1.0, np.nan), cmap="Greys", vmin=0, vmax=1.4,
                      extent=(-0.5, X - 0.5, Y - 0.5, -0.5))  # fmt: skip
            off = data["obs_range"] // 2
            P = f["pursuers"]
            for x, y in P:
                ax.add_patch(Rectangle((x - off - 0.5, y - off - 0.5), 2 * off + 1, 2 * off + 1,
                                       fill=True, alpha=0.06, color="#2a78d6", lw=0))  # fmt: skip
            d = f["delivery"]
            if d is not None:
                for r in range(len(P)):
                    for s in range(len(P)):
                        if r <= s:
                            continue
                        both = d[r, s] or d[s, r]
                        ax.plot([P[r, 0], P[s, 0]], [P[r, 1], P[s, 1]],
                                color="#2a78d6" if both else "#b4b2a9",
                                lw=0.6 if both else 0.5, ls="-" if both else (0, (2, 2)),
                                alpha=0.55 if both else 0.5)  # fmt: skip
            if belief is not None and belief.ndim == 2:  # point beliefs, normalized (x, y)
                bx, by = belief[:, 0] * (X - 1), belief[:, 1] * (Y - 1)
                for (px, py), qx, qy in zip(P, bx, by, strict=True):
                    ax.plot([px, qx], [py, qy], color="#eb6834", lw=0.6, alpha=0.6)
                ax.scatter(bx, by, marker="x", s=36, color="#eb6834", lw=1.4, zorder=4)
            E = f["evaders"]
            if len(E):
                ax.scatter(E[:, 0], E[:, 1], s=34, color="#e34948", edgecolor="white", lw=0.8,
                           zorder=5)  # fmt: skip
            ax.scatter(P[:, 0], P[:, 1], s=70, marker="s", color="#2a78d6", edgecolor="white",
                       lw=1.0, zorder=6)  # fmt: skip
            caught = data["n_evaders"] - f["n_alive"]
            status = "  (episode over)" if done else ""
            ax.set_title(f"{title}\nstep {f['t']:3d}   captured {caught}/{data['n_evaders']}"
                         f"{status}", fontsize=10, loc="left")  # fmt: skip
            ax.set_xlim(-0.5, X - 0.5)
            ax.set_ylim(Y - 0.5, -0.5)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"PettingZoo Pursuit 16x16, 8 pursuers (blue), 30 evaders (red); "
                     f"channel {condition}; same episode in every panel", fontsize=10)  # fmt: skip

    anim = animation.FuncAnimation(fig, draw, frames=n_frames, interval=1000 / fps)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=writer)
    draw(min(60, n_frames - 1))
    fig.savefig(out.with_name(out.stem + "_frame60.png"))
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", action="append", required=True, help="TITLE=PATH")
    p.add_argument("--condition", default="iid-p0.2")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--out", type=Path, default=Path("runs/demo/demo.mp4"))
    args = p.parse_args(argv)
    panels = []
    for spec in args.checkpoint:
        title, _, path = spec.rpartition("=")
        data = collect(Path(path), args.condition, args.episode, args.max_steps)
        print(
            f"{title}: {len(data['frames'])} steps, capture "
            f"{data['summary']['mean_capture_rate']:.3f} (mean of {args.episode + 1} episodes)"
        )
        panels.append((title or Path(path).parent.parent.name, data))
    out = render(panels, args.condition, args.out, args.fps)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
