# experiments/025-solid-growth/scripts/graph_reg_sweep_plots.py
# [[experiments.025-solid-growth.scripts.graph_reg_sweep_plots]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/025-solid-growth/scripts/graph_reg_sweep_plots
"""The figure of the graph-regularization sweep, from the readout's CSVs.

Reads experiments/025-solid-growth/results/graph_reg_sweep_runs.csv and
graph_reg_sweep_history.csv (written by graph_reg_sweep_readout.py) and writes ONE
full-width 3 x 3 figure, graph_reg_sweep, to $ASSET_IMAGES_DIR/025-solid-growth/ (true-size
SVG plus PNG; a timestamped SVG copy for the iteration history):

  a  held-out Pearson at epoch 29 across the ladder (no penalty, the KL ladder, the hard
     mask, the random-graph control), seeds as points and the arm mean as a bar
  b  the same at each run's max over epochs
  c  edge recall at degree at epoch 29, mean over the nine regularized heads
  d  divergence of layer-1 attention from the nine graphs at epoch 29 (validation pass)
  e  gradient-norm ratio, penalty over point loss, on the probe batch at epochs 0 and 20
  f  the random-graph control at lambda 1e-3 against the biological graphs and no penalty
  g  validation Pearson by epoch, mean over complete seeds with a +- sd band
  h  training Pearson by epoch
  i  validation point loss by epoch

Colors: one palette color per arm that is drawn as a curve (no penalty gray, hard mask
red, KL 1e-3 orange, KL 1e-1 yellow, KL 1 blue, random graphs purple); the rest of the
ladder is orange. Curves use complete (30-epoch) runs only, so no arm ends early because
one seed is still running. Tick labels are standalone mathtext; prose labels spell lambda
in decimals because Arial has no superscript minus or nabla glyph.

Repo figure standards: Arial 6 pt, boxed axes, palette from torchcell.utils, tenth
gridlines on the Pearson axes, panel letters outside the axes, no bbox_inches="tight".
"""

from __future__ import annotations

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
from matplotlib.ticker import MultipleLocator
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
LADDER_TICKS = [
    "0",
    "$10^{-5}$",
    "$10^{-4}$",
    "$10^{-3}$",
    "$10^{-2}$",
    "$10^{-1}$",
    "1",
    "mask",
    "rand",
]
ARM_COLOR = {
    "kl_0": GRAY,
    "kl_1e-05": ORANGE,
    "kl_0.0001": ORANGE,
    "kl_0.001": ORANGE,
    "kl_0.01": ORANGE,
    "kl_0.1": YELLOW,
    "kl_1": BLUE,
    "mask": RED,
    RANDOM: PURPLE,
}
# Drawn back to front, so no penalty (gray) is on top and never hidden.
CURVE_ARMS = [
    (RANDOM, "KL λ = 0.001, random graphs", PURPLE, "--"),
    ("kl_1", "KL λ = 1", BLUE, "-"),
    ("kl_0.1", "KL λ = 0.1", YELLOW, "-"),
    ("kl_0.001", "KL λ = 0.001", ORANGE, "-"),
    ("mask", "hard mask", RED, "-"),
    ("kl_0", "no penalty", GRAY, "-"),
]
X_LABEL = "λ (graph prior weight)"


def _box(ax: Axes) -> None:
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)
    ax.tick_params(width=0.5, length=2)


def _tenths(ax: Axes) -> None:
    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.yaxis.set_minor_locator(MultipleLocator(0.01))
    ax.grid(axis="y", which="both", color="#DDDDDD", linewidth=0.3)
    ax.tick_params(axis="y", which="minor", length=0)


def _save(fig: Figure, name: str) -> None:
    os.makedirs(IMG_DIR, exist_ok=True)
    savefig_true_size_svg(fig, osp.join(IMG_DIR, f"{name}.svg"))
    fig.savefig(osp.join(IMG_DIR, f"{name}.png"), dpi=300)
    savefig_true_size_svg(fig, osp.join(IMG_DIR, f"{name}_{TS}.svg"))
    plt.close(fig)


def _seed_points(
    ax: Axes,
    x: float,
    vals: NDArray[Any],
    color: str,
    filled: bool,
    jitter: float = 0.1,
) -> None:
    xs = x + np.linspace(-jitter, jitter, len(vals)) if len(vals) > 1 else np.array([x])
    ax.scatter(
        xs,
        vals,
        s=7,
        facecolor=color if filled else "white",
        edgecolor=color,
        linewidth=0.5,
        zorder=3,
    )


def _ladder_axis(ax: Axes, xlabel: str) -> None:
    ax.set_xticks(range(len(LADDER)))
    ax.set_xticklabels(LADDER_TICKS)
    ax.set_xlim(-0.6, len(LADDER) - 0.4)
    ax.set_xlabel(xlabel)
    _box(ax)


def _ladder_points(ax: Axes, runs: pd.DataFrame, col: str, filled: bool = True) -> None:
    """Seed points, arm-mean bars, and a line through the KL ladder's means."""
    xpos = {arm: i for i, arm in enumerate(LADDER)}
    means = []
    for arm in LADDER:
        vals = runs[runs.arm == arm][col].dropna().to_numpy()
        if not len(vals):
            continue
        c = ARM_COLOR[arm]
        _seed_points(ax, xpos[arm], vals, c, filled)
        ax.plot(
            [xpos[arm] - 0.3, xpos[arm] + 0.3],
            [vals.mean()] * 2,
            color=c,
            lw=1.0,
            zorder=2,
        )
        means.append((xpos[arm], vals.mean(), arm))
    kl = [(x, m) for x, m, arm in means if arm.startswith("kl_") and arm != "kl_0"]
    ax.plot([x for x, _ in kl], [m for _, m in kl], color=ORANGE, lw=0.8, zorder=1)


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


def figure(runs: pd.DataFrame, hist: pd.DataFrame) -> None:
    """The 3 x 3 figure."""
    fig, axes = plt.subplots(
        3, 3, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(160))
    )
    fig.subplots_adjust(
        left=0.055, right=0.99, bottom=0.05, top=0.955, wspace=0.28, hspace=0.5
    )
    ax = axes.ravel()
    done = runs[runs.complete]
    xpos = {arm: i for i, arm in enumerate(LADDER)}

    # a: fixed-epoch reading; b: max over epochs
    _ladder_points(ax[0], done, "val_pearson_fixed")
    ax[0].set_ylabel("Held-out Pearson, epoch 29")
    ax[0].set_title("Accuracy at the fixed epoch")
    ax[0].set_ylim(0.405, 0.465)
    _tenths(ax[0])
    _ladder_axis(ax[0], X_LABEL)
    _ladder_points(ax[1], runs, "val_pearson_max", filled=False)
    ax[1].set_ylabel("Held-out Pearson, max over epochs")
    ax[1].set_title("Accuracy at each run's best epoch")
    ax[1].set_ylim(0.405, 0.465)
    _tenths(ax[1])
    _ladder_axis(ax[1], X_LABEL)

    # c: edge recall at degree
    _ladder_points(ax[2], done, "edge_recall_fixed")
    ax[2].text(
        xpos["mask"],
        0.985,
        "1 by construction",
        ha="center",
        va="top",
        fontsize=6,
        color=RED,
    )
    ax[2].set_ylim(0, 1.0)
    ax[2].yaxis.set_major_locator(MultipleLocator(0.2))
    ax[2].set_ylabel("Edge recall at degree, mean of nine heads")
    ax[2].set_title("Attention sits on the graph")
    _ladder_axis(ax[2], "λ (graph prior weight)")

    # d: divergence
    _ladder_points(ax[3], done, "val_divergence_fixed")
    ax[3].set_yscale("log")
    ax[3].set_ylim(2e2, 3e3)
    ax[3].text(xpos["kl_0"], 2.2e2, "not logged", ha="center", fontsize=6, color=GRAY)
    ax[3].text(xpos["mask"], 2.2e2, "not logged", ha="center", fontsize=6, color=RED)
    ax[3].set_ylabel("Divergence to the graphs, epoch 29")
    ax[3].set_title("How close the attention gets")
    _ladder_axis(ax[3], "λ (graph prior weight)")

    # e: gradient budget
    for arm in LADDER:
        sub = runs[runs.arm == arm]
        c = ARM_COLOR[arm]
        r0 = sub.probe_ratio_epoch0.dropna().to_numpy()
        r20 = sub.probe_ratio_epoch20.dropna().to_numpy()
        if len(r0) and r0.max() > 0:
            _seed_points(ax[4], xpos[arm] - 0.15, r0, c, True, jitter=0.06)
        if len(r20) and r20.max() > 0:
            _seed_points(ax[4], xpos[arm] + 0.15, r20, c, False, jitter=0.06)
    ax[4].axhline(1.0, color="black", lw=0.5, ls="--")
    ax[4].text(xpos["kl_0"], 1.3e-3, "0", ha="center", fontsize=6, color=GRAY)
    ax[4].text(xpos["mask"], 1.3e-3, "0", ha="center", fontsize=6, color=RED)
    ax[4].set_yscale("log")
    ax[4].set_ylim(1e-3, 3e3)
    ax[4].set_ylabel("Gradient norm ratio, penalty over point loss")
    ax[4].set_title("Where the gradient comes from")
    ax[4].scatter(
        [], [], s=7, facecolor=ORANGE, edgecolor=ORANGE, linewidth=0.5, label="epoch 0"
    )
    ax[4].scatter(
        [],
        [],
        s=7,
        facecolor="white",
        edgecolor=ORANGE,
        linewidth=0.5,
        label="epoch 20",
    )
    ax[4].legend(loc="upper left", frameon=False, handletextpad=0.3, borderpad=0.2)
    _ladder_axis(ax[4], "λ (graph prior weight)")

    # f: the random-graph control at lambda 1e-3
    arms = [
        ("kl_0.001", "biological", ORANGE),
        (RANDOM, "random,\ndegree-matched", PURPLE),
        ("kl_0", "none (λ = 0)", GRAY),
    ]
    for i, (arm, _, color) in enumerate(arms):
        sub = done[done.arm == arm]
        part = runs[(runs.arm == arm) & ~runs.complete]
        if len(sub):
            ax[5].bar(
                i,
                sub.val_pearson_fixed.mean(),
                width=0.6,
                facecolor="white",
                edgecolor=color,
                linewidth=0.8,
            )
            _seed_points(
                ax[5], i, sub.val_pearson_fixed.to_numpy(), color, True, jitter=0.12
            )
        for r in part.to_dict("records"):
            vmax, ep = float(r["val_pearson_max"]), int(r["epochs_logged"]) - 1
            ax[5].scatter(
                [i + 0.22], [vmax], s=9, marker="x", color=color, linewidth=0.6
            )
            ax[5].text(
                i + 0.22,
                vmax + 0.003,
                f"seed {int(r['seed'])}, max at\nepoch {ep}, partial",
                ha="center",
                fontsize=5,
                color=color,
            )
    ax[5].set_xticks(range(len(arms)))
    ax[5].set_xticklabels([a[1] for a in arms])
    ax[5].set_xlim(-0.6, len(arms) - 0.4)
    ax[5].set_ylim(0.40, 0.46)
    ax[5].set_ylabel("Held-out Pearson, epoch 29")
    ax[5].set_title("Graph target of the KL at λ = 0.001")
    _tenths(ax[5])
    _box(ax[5])

    # g, h, i: curves, complete runs only
    complete_ids = set(done.run_id)
    h = hist[hist.run_id.isin(complete_ids)]
    panels = [
        (
            "val/gene_interaction/Pearson",
            "Validation Pearson, gene interaction",
            "Held-out accuracy by epoch",
        ),
        (
            "train/gene_interaction/Pearson",
            "Training Pearson, gene interaction",
            "Fit to the training triples",
        ),
        (
            "val/point_loss",
            "Validation point loss (z-scored MSE)",
            "Held-out loss by epoch",
        ),
    ]
    for k, (key, ylabel, title) in enumerate(panels):
        a = ax[6 + k]
        for z, (arm, label, color, ls) in enumerate(CURVE_ARMS):
            _curve(a, h, arm, key, color, ls, label, z=20 + z)
        a.set_xlabel("Epoch")
        a.set_ylabel(ylabel)
        a.set_title(title)
        a.set_xlim(0, 29)
        _box(a)
    _tenths(ax[6])
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
    ax[7].grid(axis="y", color="#DDDDDD", linewidth=0.3)

    for a, letter in zip(ax, "abcdefghi"):
        panel_label(a, letter)
    _save(fig, "graph_reg_sweep")


def main() -> None:
    """The figure from the CSVs."""
    runs = pd.read_csv(osp.join(RESULTS_DIR, "graph_reg_sweep_runs.csv"))
    hist = pd.read_csv(osp.join(RESULTS_DIR, "graph_reg_sweep_history.csv"))
    figure(runs, hist)
    print("wrote", osp.join(IMG_DIR, "graph_reg_sweep.svg"))


if __name__ == "__main__":
    main()
