# experiments/025-solid-growth/scripts/graph_reg_sweep_plots.py
# [[experiments.025-solid-growth.scripts.graph_reg_sweep_plots]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/025-solid-growth/scripts/graph_reg_sweep_plots
"""The figure of the graph-regularization sweep, from the readout's CSVs.

Reads experiments/025-solid-growth/results/graph_reg_sweep_runs.csv,
graph_reg_sweep_history.csv and graph_reg_sweep_summary.json (written by
graph_reg_sweep_readout.py) and writes ONE full-width 3 x 3 figure, graph_reg_sweep, to
$ASSET_IMAGES_DIR/025-solid-growth/ (true-size SVG plus PNG; a timestamped SVG copy for
the iteration history):

  a  held-out Pearson across the ladder at three readings per seed: epoch 29 (circle),
     max over epochs (triangle), at the epoch of minimum validation point loss (square);
     stars are the paired t of the epoch-29 reading against no penalty
  b  held-out point loss (z-scored MSE) at epoch 29 and at its minimum
  c  edge recall at degree at epoch 29 against each arm's own target graphs
  d  divergence of layer-1 attention from the target graphs at epoch 29 (validation pass)
  e  gradient-norm ratio, penalty over point loss, on the probe batch across the probe
     epochs 0, 1, 2, 5, 10 and 20
  f  the random-graph control at lambda 1e-3 against the biological graphs and no penalty
  g  validation Pearson by epoch, mean over complete seeds with a +- sd band
  h  training Pearson by epoch
  i  validation point loss by epoch

Marks follow the repo's dot plots: every seed point is filled with its arm's color and
edged in black; arm means are short black bars; a quantity the model did not compute is
an x at the bottom of the axis. One palette color per arm that is drawn as a curve (no
penalty gray, hard mask red, KL 1e-3 orange, KL 1e-1 yellow, KL 1 blue, random graphs
purple); the rest of the ladder is orange. Curves use complete (30-epoch) runs only.
Every panel carries the same light horizontal grid. Panel titles state the finding the
panel shows; each is checked against the summary numbers in the document's Section 2.

Repo figure standards: Arial 6 pt, boxed axes, palette from torchcell.utils, panel letters
outside the axes, no bbox_inches="tight". Prose
labels spell lambda in decimals because Arial has no superscript minus or nabla glyph; the
ladder axis is log10 lambda so no text falls under 5 pt.
"""

from __future__ import annotations

import json
import os
import os.path as osp
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator, NullLocator
from numpy.typing import NDArray

from torchcell.timestamp import timestamp
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    panel_label,
    savefig_true_size_svg,
)

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "025-solid-growth", "results")
IMG_DIR = osp.join(ASSET_IMAGES_DIR, "025-solid-growth")

ORANGE, RED, PURPLE, YELLOW, BLUE, GRAY = PLOT_PALETTE[:6]
TERRACOTTA, SAND, SLATE = PLOT_PALETTE[12], PLOT_PALETTE[15], PLOT_PALETTE[16]
# Palette index of each arm's color.
ARM_INDEX = {
    "kl_0": 5,  # gray
    "kl_1e-05": 15,  # sand
    "kl_0.0001": 16,  # slate
    "kl_0.001": 0,  # amber
    "kl_0.01": 12,  # terracotta
    "kl_0.1": 3,  # wheat
    "kl_1": 4,  # steel blue
    "mask": 1,  # brick
    "random_0.001": 2,  # lilac
}


GRID_COLOR, GRID_LW = "#E3E3E3", 0.3
plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 6,
        "axes.labelsize": 6,
        "axes.titlesize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "svg.fonttype": "none",
        "axes.linewidth": 0.5,
    }
)
TS = timestamp()

RANDOM = "random_0.001"
LADDER = [
    "kl_0",
    "kl_1e-05",
    "kl_0.0001",
    "kl_0.001",
    "kl_0.01",
    "kl_0.1",
    "kl_1",
    "mask",
    RANDOM,
]
# Exponents of lambda at full size: mathtext superscripts render at 0.7 of the font
# (4.2 pt at 6 pt), under Nature's 5 pt floor, so the axis carries log10 lambda instead.
LADDER_TICKS = ["none", "−5", "−4", "−3", "−2", "−1", "0", "mask", "rand"]
XPOS = {arm: i for i, arm in enumerate(LADDER)}
ARM_COLOR = {arm: PLOT_PALETTE[k] for arm, k in ARM_INDEX.items()}
ARM_SHORT = {
    "kl_1e-05": "1e-5",
    "kl_0.0001": "1e-4",
    "kl_0.001": "1e-3",
    "kl_0.01": "1e-2",
    "kl_0.1": "0.1",
    "kl_1": "1",
    RANDOM: "random",
}
# Drawn back to front, so no penalty (gray) is on top and never hidden.
CURVE_ARMS = [
    (RANDOM, "KL λ = 0.001, random graphs", ARM_COLOR[RANDOM], "--"),
    ("kl_1", "KL λ = 1", ARM_COLOR["kl_1"], "-"),
    ("kl_0.1", "KL λ = 0.1", ARM_COLOR["kl_0.1"], "-"),
    ("kl_0.001", "KL λ = 0.001", ARM_COLOR["kl_0.001"], "-"),
    ("mask", "hard mask", ARM_COLOR["mask"], "-"),
    ("kl_0", "no penalty", ARM_COLOR["kl_0"], "-"),
]
# Reading -> (marker, x offset inside the arm's column, runs-table column, star row).
# Shades of one color did not separate the readings on review (2026-09-27), so the
# reading is the marker shape and the arm is the color.
READINGS = {
    "fixed": ("o", -0.27, "val_pearson_fixed", 0.4665),
    "max": ("^", 0.0, "val_pearson_max", 0.4635),
    "min_loss": ("s", 0.27, "val_pearson_at_min_loss", 0.4605),
}
PROBE_EPOCHS = [0, 1, 2, 5, 10, 20]
X_LABEL = "log10 λ (graph prior weight)"


def _box(ax: Axes) -> None:
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)
    ax.tick_params(width=0.5, length=2)
    ax.grid(axis="y", which="major", color=GRID_COLOR, linewidth=GRID_LW)
    ax.set_axisbelow(True)


def _pearson_grid(ax: Axes) -> None:
    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.yaxis.set_minor_locator(MultipleLocator(0.01))
    ax.grid(axis="y", which="minor", color=GRID_COLOR, linewidth=GRID_LW)
    ax.tick_params(axis="y", which="minor", length=0)


def _save(fig: Figure, name: str) -> None:
    os.makedirs(IMG_DIR, exist_ok=True)
    savefig_true_size_svg(fig, osp.join(IMG_DIR, f"{name}.svg"))
    fig.savefig(osp.join(IMG_DIR, f"{name}.png"), dpi=300)
    savefig_true_size_svg(fig, osp.join(IMG_DIR, f"{name}_{TS}.svg"))
    plt.close(fig)


def _points(
    ax: Axes,
    x: float,
    vals: NDArray[Any],
    color: str,
    marker: str = "o",
    jitter: float = 0.07,
    size: float = 9,
) -> None:
    """Seed points: arm color face, black edge; the arm mean as a short black bar."""
    xs = x + np.linspace(-jitter, jitter, len(vals)) if len(vals) > 1 else np.array([x])
    ax.scatter(
        xs,
        vals,
        s=size,
        marker=marker,
        facecolor=color,
        edgecolor="black",
        linewidth=0.4,
        zorder=4,
    )
    ax.plot([x - 0.14, x + 0.14], [vals.mean()] * 2, color="black", lw=0.7, zorder=3)


def _not_logged(ax: Axes, x: float, y: float, color: str) -> None:
    ax.scatter([x], [y], s=12, marker="x", color=color, linewidth=0.6, zorder=4)


def _ladder_axis(ax: Axes, xlabel: str = X_LABEL) -> None:
    ax.set_xticks(range(len(LADDER)))
    ax.set_xticklabels(LADDER_TICKS)
    ax.set_xlim(-0.6, len(LADDER) - 0.4)
    ax.set_xlabel(xlabel)
    _box(ax)


def _plain_log_ticks(ax: Axes, ticks: list[float]) -> None:
    """Log axis with decimal tick labels at full font size: the default 10^k mathtext puts
    the exponent at 0.7 of the font, under the 5 pt floor."""
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def _stars(p: float | None) -> str:
    if p is None:
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def _curve(
    ax: Axes,
    hist: pd.DataFrame,
    arm: str,
    key: str,
    color: str,
    ls: str,
    label: str,
    z: int,
) -> None:
    sub = hist[(hist.arm == arm) & (hist.key == key)]
    if sub.empty:
        return
    g = sub.groupby("epoch")["value"]
    mean, sd, n = g.mean(), g.std(), g.count()
    ax.plot(
        mean.index, mean.to_numpy(), color=color, lw=0.9, ls=ls, label=label, zorder=z
    )
    if (n.max() > 1) and sd.notna().any():
        ax.fill_between(
            mean.index,
            (mean - sd).to_numpy(),
            (mean + sd).to_numpy(),
            color=color,
            alpha=0.15,
            lw=0,
            zorder=z - 10,
        )


def figure(runs: pd.DataFrame, hist: pd.DataFrame, summary: dict[str, Any]) -> None:
    """The 3 x 3 figure."""
    fig, axes = plt.subplots(
        3, 3, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(165))
    )
    fig.subplots_adjust(
        left=0.055, right=0.99, bottom=0.055, top=0.955, wspace=0.3, hspace=0.55
    )
    ax = axes.ravel()
    done = runs[runs.complete]
    pvals = {
        (p["arm"], p["reading"]): p["p_two_sided"]
        for p in summary["paired_vs_no_penalty"]
    }

    # a: three readings of held-out Pearson per seed, as three shades of the arm color
    a = ax[0]
    for arm in LADDER:
        for reading, (marker, dx, col, star_y) in READINGS.items():
            pool = runs if reading == "max" else done
            vals = pool[pool.arm == arm][col].dropna().to_numpy()
            if len(vals):
                _points(
                    a,
                    XPOS[arm] + dx,
                    vals,
                    ARM_COLOR[arm],
                    marker=marker,
                    jitter=0.05,
                    size=6,
                )
            star = _stars(pvals.get((arm, reading)))
            if star:
                a.text(
                    XPOS[arm] + dx, star_y, star, ha="center", va="center", fontsize=5.5
                )
    a.set_ylim(0.372, 0.4685)
    a.set_ylabel("Held-out Pearson, gene interaction")
    a.set_title("Prior gains at epoch 29, little at the peak")
    _pearson_grid(a)
    _ladder_axis(a)
    for reading, label in (
        ("fixed", "epoch 29"),
        ("max", "max over epochs"),
        ("min_loss", "at min validation loss"),
    ):
        a.scatter(
            [],
            [],
            s=9,
            marker=READINGS[reading][0],
            facecolor="white",
            edgecolor="black",
            linewidth=0.4,
            label=label,
        )
    a.plot([], [], color="black", lw=0.7, label="arm mean")
    a.scatter([], [], s=0, label="* vs no penalty, paired t")
    a.legend(
        loc="lower left",
        frameon=False,
        fontsize=5,
        handletextpad=0.2,
        borderpad=0.1,
        labelspacing=0.15,
        handlelength=1.2,
        markerscale=0.8,
    )

    # b: held-out point loss, at epoch 29 and at its minimum
    b = ax[1]
    for arm in LADDER:
        sub = done[done.arm == arm]
        if not len(sub):
            continue
        _points(
            b,
            XPOS[arm] - 0.18,
            sub.val_point_loss_fixed.to_numpy(),
            ARM_COLOR[arm],
            jitter=0.05,
        )
        _points(
            b,
            XPOS[arm] + 0.18,
            sub.val_point_loss_min.to_numpy(),
            ARM_COLOR[arm],
            marker="s",
            jitter=0.05,
        )
    b.set_ylabel("Held-out point loss (z-scored MSE)")
    b.set_title("Held-out loss falls with λ from 0.001 up")
    b.yaxis.set_major_locator(MultipleLocator(0.02))
    _ladder_axis(b)
    b.scatter(
        [], [], s=9, facecolor=GRAY, edgecolor="black", linewidth=0.4, label="epoch 29"
    )
    b.scatter(
        [],
        [],
        s=9,
        marker="s",
        facecolor="white",
        edgecolor="black",
        linewidth=0.4,
        label="at min validation loss",
    )
    b.legend(loc="upper right", frameon=False, handletextpad=0.3, borderpad=0.2)

    # c: edge recall at degree against each arm's own target
    c = ax[2]
    for arm in LADDER:
        vals = done[done.arm == arm].edge_recall_fixed.dropna().to_numpy()
        if len(vals):
            _points(c, XPOS[arm], vals, ARM_COLOR[arm])
    c.scatter(
        [XPOS["mask"]],
        [1.0],
        s=9,
        marker="o",
        facecolor=RED,
        edgecolor="black",
        linewidth=0.4,
        zorder=4,
    )
    c.text(
        XPOS["mask"],
        0.94,
        "by\nconstruction",
        ha="center",
        va="top",
        fontsize=5,
        color=RED,
    )
    c.text(
        XPOS[RANDOM],
        0.75,
        "vs its\nrewired\ntarget",
        ha="center",
        va="top",
        fontsize=5,
        color=PURPLE,
    )
    c.set_ylim(0, 1.04)
    c.yaxis.set_major_locator(MultipleLocator(0.2))
    c.set_ylabel("Edge recall at degree, mean of nine heads")
    c.set_title("Any λ tested moves attention onto its target")
    _ladder_axis(c)

    # d: divergence to the target graphs
    d = ax[3]
    for arm in LADDER:
        vals = done[done.arm == arm].val_divergence_fixed.dropna().to_numpy()
        if len(vals):
            _points(d, XPOS[arm], vals, ARM_COLOR[arm])
    d.set_yscale("log")
    d.set_ylim(2e2, 3e3)
    _plain_log_ticks(d, [300, 1000, 3000])
    _not_logged(d, XPOS["kl_0"], 2.3e2, ARM_COLOR["kl_0"])
    _not_logged(d, XPOS["mask"], 2.3e2, ARM_COLOR["mask"])
    d.text(
        XPOS[RANDOM],
        1.25e3,
        "vs its\nrewired\ntarget",
        ha="center",
        va="bottom",
        fontsize=5,
        color=PURPLE,
    )
    d.set_ylabel("Divergence to the target graphs, epoch 29")
    d.set_title("Divergence floors near 300 from λ = 0.01")
    _ladder_axis(d)
    d.scatter(
        [],
        [],
        s=12,
        marker="x",
        color="black",
        linewidth=0.6,
        label="not computed by the model",
    )
    d.legend(loc="upper right", frameon=False, handletextpad=0.3, borderpad=0.2)

    # e: gradient budget across the probe epochs
    e = ax[4]
    xs = np.arange(len(PROBE_EPOCHS))
    for arm in LADDER:
        if arm in ("kl_0", "mask"):
            continue
        sub = hist[
            (hist.arm == arm)
            & (hist.key == "probe/grad_ratio/graph_reg_to_point")
            & hist.run_id.isin(done.run_id)
        ]
        if sub.empty:
            continue
        m = sub.groupby("epoch")["value"].mean().reindex(PROBE_EPOCHS)
        ls = "--" if arm == RANDOM else "-"
        e.plot(
            xs,
            m.to_numpy(),
            color=ARM_COLOR[arm],
            lw=0.9,
            ls=ls,
            marker="o",
            ms=2.2,
            mec="black",
            mew=0.3,
            zorder=3,
        )
        e.text(
            xs[-1] + 0.12,
            m.to_numpy()[-1],
            ARM_SHORT[arm],
            va="center",
            ha="left",
            fontsize=5,
            color=ARM_COLOR[arm],
        )
    e.axhline(1.0, color="black", lw=0.5, ls="--")
    _not_logged(e, 0, 1.4e-3, ARM_COLOR["kl_0"])
    _not_logged(e, 0, 1.4e-3, ARM_COLOR["mask"])
    e.text(0.35, 1.4e-3, "none, mask: 0", va="center", fontsize=5, color="black")
    e.set_yscale("log")
    e.set_ylim(1e-3, 5e3)
    _plain_log_ticks(e, [0.001, 0.01, 0.1, 1, 10, 100, 1000])
    e.set_xticks(xs)
    e.set_xticklabels([str(p) for p in PROBE_EPOCHS])
    e.set_xlim(-0.4, len(PROBE_EPOCHS) - 0.2)
    e.set_xlabel("Probe epoch")
    e.set_ylabel("Gradient norm ratio, penalty over point loss")
    e.set_title("Penalty dominates the gradient from λ = 0.01")
    _box(e)

    # f: the random-graph control at lambda 1e-3
    f = ax[5]
    arms = [
        ("kl_0.001", "biological\ngraphs", ORANGE),
        (RANDOM, "random,\ndegree-matched", PURPLE),
        ("kl_0", "none\n(λ = 0)", GRAY),
    ]
    for i, (arm, _, color) in enumerate(arms):
        sub = done[done.arm == arm]
        part = runs[(runs.arm == arm) & ~runs.complete]
        if len(sub):
            f.bar(
                i,
                sub.val_pearson_fixed.mean(),
                width=0.62,
                facecolor=color,
                edgecolor="black",
                linewidth=0.5,
                zorder=2,
            )
            f.scatter(
                i + np.linspace(-0.12, 0.12, len(sub)),
                sub.val_pearson_fixed.to_numpy(),
                s=9,
                facecolor="black",
                edgecolor="black",
                linewidth=0.4,
                zorder=4,
            )
            f.text(
                i,
                0.4015,
                f"n = {len(sub)}",
                ha="center",
                va="bottom",
                fontsize=5,
                color="white",
            )
        for r in part.to_dict("records"):
            f.scatter(
                [i + 0.42],
                [float(r["val_pearson_max"])],
                s=12,
                marker="x",
                color=color,
                linewidth=0.6,
                zorder=4,
            )
            f.text(
                i + 0.42,
                float(r["val_pearson_max"]) + 0.003,
                f"seed {int(r['seed'])}\nmax, ep {int(r['epochs_logged']) - 1}\n(running)",
                ha="center",
                va="bottom",
                fontsize=5,
                color=color,
            )
    f.set_xticks(range(len(arms)))
    f.set_xticklabels([t[1] for t in arms])
    f.set_xlim(-0.6, len(arms) - 0.3)
    f.set_ylim(0.40, 0.46)
    f.set_ylabel("Held-out Pearson, epoch 29")
    f.set_title("λ = 0.001: random target equals no penalty")
    _pearson_grid(f)
    _box(f)

    # g, h, i: curves, complete runs only
    h_ = hist[hist.run_id.isin(set(done.run_id))]
    panels = [
        (
            "val/gene_interaction/Pearson",
            "Validation Pearson, gene interaction",
            "Strong prior removes the decline after epoch 16",
        ),
        (
            "train/gene_interaction/Pearson",
            "Training Pearson, gene interaction",
            "Strong priors fit training like no penalty",
        ),
        (
            "val/point_loss",
            "Validation point loss (z-scored MSE)",
            "Held-out loss rises less under a strong prior",
        ),
    ]
    for k, (key, ylabel, title) in enumerate(panels):
        p_ = ax[6 + k]
        for z, (arm, label, color, ls) in enumerate(CURVE_ARMS):
            _curve(p_, h_, arm, key, color, ls, label, z=20 + z)
        p_.set_xlabel("Epoch")
        p_.set_ylabel(ylabel)
        p_.set_title(title)
        p_.set_xlim(0, 29)
        _box(p_)
    _pearson_grid(ax[6])
    handles, labels = ax[6].get_legend_handles_labels()
    ax[6].legend(
        handles[::-1],
        labels[::-1],
        loc="lower right",
        frameon=False,
        handlelength=1.6,
        handletextpad=0.4,
        borderpad=0.2,
    )
    ax[7].yaxis.set_major_locator(MultipleLocator(0.1))
    ax[8].yaxis.set_major_locator(MultipleLocator(0.02))

    for p_, letter in zip(ax, "abcdefghi"):
        panel_label(p_, letter)
    _save(fig, "graph_reg_sweep")


def main() -> None:
    """The figure from the CSVs and the summary."""
    runs = pd.read_csv(osp.join(RESULTS_DIR, "graph_reg_sweep_runs.csv"))
    hist = pd.read_csv(osp.join(RESULTS_DIR, "graph_reg_sweep_history.csv"))
    summary = json.load(open(osp.join(RESULTS_DIR, "graph_reg_sweep_summary.json")))
    figure(runs, hist, summary)
    print("wrote", osp.join(IMG_DIR, "graph_reg_sweep.svg"))


if __name__ == "__main__":
    main()
