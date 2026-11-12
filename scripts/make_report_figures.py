"""Report figures with plain method names, from the evaluation cells (no number is typed by hand).

    scripts/make_report_figures.py runs/evals/final --out ../report/figures  (PYTHONPATH=src:.)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from dynabelief.beliefs.filter import EvaderFilter, team_view_mask
from dynabelief.config import load_config
from dynabelief.envs.pursuit import PursuitEnv
from dynabelief.evaluation.aggregate import load_cells

NAMES = {
    "tarmac_pos": "TarMAC baseline\n(no belief)",
    "d_diabl_posterior": "Map belief\n(team knowledge)",
    "d_diabl_coef10": "Map belief\n(true positions)",
    "point_diabl": "Point belief",
    "point_diabl_stopgrad": "Point belief,\nno channel gradient",
    "point_diabl_noreplay": "Point belief,\nno replay",
    "point_diabl_robust": "Point belief,\nrobust training (ours)",
}
COLORS = {
    "tarmac_pos": "#2a78d6", "point_diabl": "#eb6834", "d_diabl_coef10": "#1baf7a",
    "d_diabl_posterior": "#eda100", "point_diabl_stopgrad": "#e87ba4",
    "point_diabl_noreplay": "#008300", "point_diabl_robust": "#6250d6",
}  # fmt: skip
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"


def seeds_of(cells, arm, cond):
    return {s: c["summary"]["mean_capture_rate"] for (a, s, k), c in cells.items()
            if a == arm and k == cond}  # fmt: skip


def style(ax, axis="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def bars(ax, cells, arms, cond, width=0.62):
    for i, arm in enumerate(arms):
        per = seeds_of(cells, arm, cond)
        if not per:
            continue
        m = float(np.mean(list(per.values())))
        ax.bar(i, m, width, color=COLORS[arm], edgecolor="white", linewidth=2)
        ax.scatter(np.full(len(per), i) + np.linspace(-0.12, 0.12, len(per)), list(per.values()),
                   s=16, color=INK, zorder=3)  # fmt: skip
        ax.text(i, max(max(per.values()), m) + 0.03, f"{m:.2f}", ha="center", fontsize=9,
                color=INK)  # fmt: skip
    ax.set_xticks(range(len(arms)), [NAMES[a] for a in arms], fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("capture rate", color=INK)
    style(ax)


def fig_methods(cells, out):
    arms = ["tarmac_pos", "d_diabl_posterior", "d_diabl_coef10", "point_diabl"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=220)
    bars(ax, cells, arms, "iid-p0")
    ax.set_title("Evaders caught on a clean channel (bars: mean over seeds; dots: seeds)",
                 fontsize=9, loc="left", color=MUTED)  # fmt: skip
    fig.tight_layout()
    fig.savefig(out / "fig_methods.png")
    plt.close(fig)


def fig_mechanism(cells, out):
    arms = ["point_diabl", "point_diabl_stopgrad", "point_diabl_noreplay", "tarmac_pos"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=220)
    bars(ax, cells, arms, "iid-p0")
    ax.set_title("Which ingredient gives the gain? (clean channel)", fontsize=9, loc="left",
                 color=MUTED)  # fmt: skip
    fig.tight_layout()
    fig.savefig(out / "fig_mechanism.png")
    plt.close(fig)


def fig_robust(cells, out):
    conds = [("iid-p0", "clean\nchannel"), ("iid-p0.8", "80 % loss,\nindependent"),
             ("ge-p0.8-b20", "80 % loss,\nin bursts"), ("iid-p0-random-r0.1", "send only\n10 % of the time"),  # noqa: E501
             ("iid-p1", "no messages\nat all")]  # fmt: skip
    fig, ax = plt.subplots(figsize=(6.8, 3.6), dpi=220)
    w = 0.36
    for j, (arm, label) in enumerate([("point_diabl", "Point belief (trained on a clean channel)"),
                                      ("point_diabl_robust", "Point belief, robust training (ours)")]):  # fmt: skip  # noqa: E501
        for i, (cond, _) in enumerate(conds):
            per = seeds_of(cells, arm, cond)
            if not per:
                continue
            m = float(np.mean(list(per.values())))
            x = i + (j - 0.5) * (w + 0.02)
            ax.bar(x, m, w, color=COLORS[arm], edgecolor="white", linewidth=2,
                   hatch="//" if j else None, label=label if i == 0 else None)  # fmt: skip
            ax.scatter(np.full(len(per), x) + np.linspace(-0.08, 0.08, len(per)),
                       list(per.values()), s=12, color=INK, zorder=3)  # fmt: skip
            ax.text(x, m + 0.03, f"{m:.2f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(conds)), [c[1] for c in conds], fontsize=8)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("capture rate", color=INK)
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    ax.set_title("Same test episodes, increasingly bad channels (dots: 3 training seeds)",
                 fontsize=9, loc="left", color=MUTED)  # fmt: skip
    fig.tight_layout()
    fig.savefig(out / "fig_robust.png")
    plt.close(fig)


def fig_loss_curves(cells, out):
    conds = ["iid-p0", "iid-p0.2", "iid-p0.5", "iid-p0.8", "iid-p1"]
    xs = [0, 20, 50, 80, 100]
    marks = {"tarmac_pos": "o", "point_diabl": "s", "d_diabl_coef10": "^", "d_diabl_posterior": "D",
             "point_diabl_robust": "X"}  # fmt: skip
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=220)
    for arm, marker in marks.items():
        per = [seeds_of(cells, arm, c) for c in conds]
        if not all(per):
            continue
        mean = [np.mean(list(p.values())) for p in per]
        ax.fill_between(xs, [min(p.values()) for p in per], [max(p.values()) for p in per],
                        color=COLORS[arm], alpha=0.12, linewidth=0)  # fmt: skip
        ax.plot(xs, mean, color=COLORS[arm], linewidth=2, marker=marker, markersize=6,
                markeredgecolor="white", markeredgewidth=1,
                label=NAMES[arm].replace("\n", " "))  # fmt: skip
    ax.set_xlabel("messages lost independently (%)", color=INK)
    ax.set_ylabel("capture rate", color=INK)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(xs)
    style(ax)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "fig_loss_curves.png")
    plt.close(fig)


def fig_map_quality(cells, out):
    """How much of the prior's error on cells outside the agent's own view each map removes."""
    gain = {"d_diabl_coef10": [], "d_diabl_posterior": [], "teacher": []}
    for (a, _s, c), cell in cells.items():
        if a in gain and c == "iid-p0":
            s = cell["summary"]
            gain[a].append(100 * (1 - s["map_bce_remote_model"] / s["map_bce_remote_prior"]))
            gain["teacher"].append(
                100 * (1 - s["map_bce_remote_filter"] / s["map_bce_remote_prior"])
            )
    if not gain["teacher"]:
        return
    items = [(NAMES[a].replace("\n", " "), gain[a], COLORS[a])
             for a in ("d_diabl_coef10", "d_diabl_posterior")]  # fmt: skip
    items.append(("The teacher itself\n(best any learner could do)", gain["teacher"], "#6b6a64"))
    fig, ax = plt.subplots(figsize=(6.4, 2.6), dpi=220)
    for i, (_label, vals, col) in enumerate(items[::-1]):
        v = float(np.mean(vals))
        ax.barh(i, v, 0.6, color=col, edgecolor="white", linewidth=2)
        ax.scatter(vals, np.full(len(vals), i), s=14, color=INK, zorder=3)
        ax.text(max(vals) + 0.4, i, f"{v:.1f} %", va="center", fontsize=9, color=INK)
    ax.axvline(15, color=INK, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(15.2, len(items) - 0.55, "our bar: 15 %", fontsize=8, color=INK)
    ax.set_yticks(range(len(items)), [i[0] for i in items[::-1]], fontsize=8)
    ax.set_xlim(0, 22)
    ax.set_xlabel("uncertainty about unseen cells removed, vs. knowing only the base rate (%)",
                  color=INK, fontsize=8)  # fmt: skip
    style(ax, "x")
    fig.tight_layout()
    fig.savefig(out / "fig_map_quality.png")
    plt.close(fig)


def fig_targets(out, seed=11, steps=40):
    """One real game state: what each kind of belief is asked to predict."""
    cfg = load_config("configs/d_diabl_posterior.yaml").env
    env = PursuitEnv(cfg)
    rng = np.random.default_rng(0)
    _, priv = env.reset(seed=seed)
    free = (priv.global_state[0] == 0)[None]
    f = EvaderFilter(1, cfg.n_evaders, (cfg.y_size, cfg.x_size), cfg.obs_range)
    f.reset(np.array([True]), free)
    for _ in range(steps):
        f.update(priv.pursuer_pos[None], priv.evader_pos[None], priv.evader_alive[None], free)
        priv = env.step(rng.integers(env.n_actions, size=cfg.n_pursuers)).privileged
    f.update(priv.pursuer_pos[None], priv.evader_pos[None], priv.evader_alive[None], free)
    env.close()
    view = team_view_mask(priv.pursuer_pos[None], (cfg.y_size, cfg.x_size), cfg.obs_range)[0]
    truth = priv.global_state[2] > 0
    obstacles = priv.global_state[0] > 0
    post = f.occupancy()[0]
    alive = priv.evader_pos[priv.evader_alive]
    P = priv.pursuer_pos
    seen = alive[view[alive[:, 1], alive[:, 0]]]
    d = np.abs(seen - P[0]).sum(1) if len(seen) else np.array([])
    titles = [
        "(a) The game: what the team sees",
        "(b) Point target: one seen evader",
        "(c) Map target: every true position",
        "(d) Map target (ours): what the team could know",
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.6), dpi=200)
    ext = (-0.5, cfg.x_size - 0.5, cfg.y_size - 0.5, -0.5)
    for k, ax in enumerate(axes):
        if k == 2:
            ax.imshow(truth.astype(float), cmap="Oranges", vmin=0, vmax=1.3, extent=ext)
        elif k == 3:
            ax.imshow(post, cmap="Oranges", vmin=0, vmax=0.6, extent=ext)
        else:
            ax.imshow(np.zeros_like(post), cmap="Greys", vmin=0, vmax=1, extent=ext)
        ax.imshow(np.where(obstacles, 1.0, np.nan), cmap="Greys", vmin=0, vmax=1.5, extent=ext)
        if k in (0, 3):
            off = cfg.obs_range // 2
            for x, y in P:
                ax.add_patch(Rectangle((x - off - 0.5, y - off - 0.5), 2 * off + 1, 2 * off + 1,
                                       fill=k == 0, alpha=0.10 if k == 0 else 1, lw=0.7,
                                       edgecolor="#2a78d6", facecolor="#2a78d6"))  # fmt: skip
        if k == 0:
            ax.scatter(alive[:, 0], alive[:, 1], s=22, color="#e34948", edgecolor="white", lw=0.6,
                       zorder=4)  # fmt: skip
        ax.scatter(P[:, 0], P[:, 1], s=42, marker="s", color="#2a78d6", edgecolor="white", lw=0.8,
                   zorder=5)  # fmt: skip
        if k == 1 and len(seen):
            t = seen[int(d.argmin())]
            ax.scatter([t[0]], [t[1]], s=160, marker="x", color="#eb6834", lw=2.5, zorder=6)
            ax.plot([P[0, 0], t[0]], [P[0, 1], t[1]], color="#eb6834", lw=1)
            ax.scatter([P[0, 0]], [P[0, 1]], s=110, marker="s", facecolor="none",
                       edgecolor="#eb6834", lw=1.6, zorder=6)  # fmt: skip
        ax.set_xlim(-0.5, cfg.x_size - 0.5)
        ax.set_ylim(cfg.y_size - 0.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(titles[k], fontsize=9.5, loc="left")
    fig.tight_layout()
    fig.savefig(out / "fig_targets.png")
    plt.close(fig)
    return {"evaders_alive": int(len(alive)), "evaders_seen_now": int(len(seen))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cells = load_cells(a.root)
    fig_methods(cells, a.out)
    fig_mechanism(cells, a.out)
    fig_robust(cells, a.out)
    fig_loss_curves(cells, a.out)
    fig_map_quality(cells, a.out)
    info = fig_targets(a.out)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
