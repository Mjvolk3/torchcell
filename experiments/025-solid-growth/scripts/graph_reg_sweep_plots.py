# experiments/025-solid-growth/scripts/graph_reg_sweep_plots.py
# [[experiments.025-solid-growth.scripts.graph_reg_sweep_plots]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/025-solid-growth/scripts/graph_reg_sweep_plots
"""Figures of the graph-regularization sweep, from the readout's CSVs.

Reads experiments/025-solid-growth/results/graph_reg_sweep_runs.csv and
graph_reg_sweep_history.csv (written by graph_reg_sweep_readout.py) and writes three
figures to $ASSET_IMAGES_DIR/025-solid-growth/, each a full-width row of panels at Nature
print size (true-size SVG plus PNG; timestamped copies for the iteration history):

- graph_reg_sweep_ladder    (a) held-out Pearson across the ladder, no penalty and the
                            hard mask at the ends, the random-graph control beside the
                            control's lambda; (b) how much of each regularized head's
                            attention sits on its graph (edge recall at degree); (c) the
                            divergence of the attention from the graph; (d) the gradient
                            budget: penalty gradient over point-loss gradient on the
                            probe batch at epochs 0 and 20
- graph_reg_sweep_curves    validation Pearson, training Pearson and validation point
                            loss by epoch, mean over seeds, for the arms that separate
- graph_reg_sweep_control   the random-graph control against the biological graphs and
                            no penalty at the control's lambda, seeds as points

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
DARK_ORANGE, TERRACOTTA = PLOT_PALETTE[6], PLOT_PALETTE[12]
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

# The ladder as drawn: no penalty at the left, the mask at the right, KL between.
RANDOM = "random_0.001"
# Tick labels are standalone mathtext (nothing adjoins them, so the PDF conversion cannot
# drop a space); prose labels spell lambda in decimals because Arial has no superscript minus.
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
    "random",
]
ARM_COLOR = {
    "kl_0": GRAY,
    "kl_1e-05": ORANGE,
    "kl_0.0001": ORANGE,
    "kl_0.001": ORANGE,
    "kl_0.01": ORANGE,
    "kl_0.1": DARK_ORANGE,
    "kl_1": TERRACOTTA,
    "mask": RED,
    RANDOM: PURPLE,
}
CURVE_ARMS = [
    ("kl_0", "no penalty", GRAY, "-"),
    ("mask", "hard mask", RED, "-"),
    ("kl_0.001", "KL λ = 0.001", ORANGE, "-"),
    ("kl_0.1", "KL λ = 0.1", DARK_ORANGE, "-"),
    ("kl_1", "KL λ = 1", TERRACOTTA, "-"),
    (RANDOM, "KL λ = 0.001, random graphs", PURPLE, "--"),
]


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
    jitter: float = 0.08,
) -> None:
    xs = x + np.linspace(-jitter, jitter, len(vals)) if len(vals) > 1 else np.array([x])
    ax.scatter(
        xs,
        vals,
        s=6,
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


def fig_ladder(runs: pd.DataFrame) -> None:
    """(a) accuracy, (b) edge recall, (c) divergence, (d) gradient budget, across the ladder."""
    fig, axes = plt.subplots(
        2, 2, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(105))
    )
    fig.subplots_adjust(
        left=0.06, right=0.99, bottom=0.09, top=0.93, wspace=0.22, hspace=0.42
    )
    axes = axes.ravel()
    xpos = {arm: i for i, arm in enumerate(LADDER)}
    done = runs[runs.complete]

    # (a) held-out Pearson: epoch 29 (filled, line through the KL means) and max over
    # epochs (hollow), seeds as points.
    ax = axes[0]
    means_fixed = []
    for arm in LADDER:
        sub = done[done.arm == arm]
        allruns = runs[runs.arm == arm]
        c = ARM_COLOR[arm]
        if len(sub):
            _seed_points(
                ax, xpos[arm] - 0.15, sub.val_pearson_fixed.to_numpy(), c, True
            )
            means_fixed.append((xpos[arm], sub.val_pearson_fixed.mean()))
        if len(allruns):
            _seed_points(
                ax, xpos[arm] + 0.15, allruns.val_pearson_max.to_numpy(), c, False
            )
    kl = [
        (x, m)
        for x, m in means_fixed
        if LADDER[x].startswith("kl_") and LADDER[x] != "kl_0"
    ]
    ax.plot([x for x, _ in kl], [m for _, m in kl], color=ORANGE, lw=0.8, zorder=2)
    for x, m in means_fixed:
        ax.plot(
            [x - 0.3, x + 0.3], [m, m], color=ARM_COLOR[LADDER[x]], lw=1.0, zorder=2
        )
    ax.set_ylabel("Held-out Pearson, gene interaction")
    ax.set_title("Accuracy across the ladder")
    ax.set_ylim(0.405, 0.474)
    _tenths(ax)
    _ladder_axis(
        ax,
        "λ (graph prior weight); mask = hard mask, layer 1; random = KL at λ = 0.001 toward rewired graphs",
    )
    ax.scatter(
        [],
        [],
        s=6,
        facecolor=ORANGE,
        edgecolor=ORANGE,
        linewidth=0.5,
        label="epoch 29, per seed",
    )
    ax.scatter(
        [],
        [],
        s=6,
        facecolor="white",
        edgecolor=ORANGE,
        linewidth=0.5,
        label="max over epochs, per seed",
    )
    ax.legend(loc="upper left", frameon=False, handletextpad=0.3, borderpad=0.2, ncol=2)

    # (b) support contraction: edge recall at degree, mean over the nine heads, epoch 29.
    ax = axes[1]
    for arm in LADDER:
        sub = done[done.arm == arm]
        rec = sub.edge_recall_fixed.dropna().to_numpy()
        if len(rec):
            _seed_points(ax, xpos[arm], rec, ARM_COLOR[arm], True)
            ax.plot(
                [xpos[arm] - 0.3, xpos[arm] + 0.3],
                [rec.mean()] * 2,
                color=ARM_COLOR[arm],
                lw=1.0,
            )
    ax.text(
        xpos["mask"],
        0.985,
        "1 by construction",
        ha="center",
        va="top",
        fontsize=6,
        color=RED,
    )
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.set_ylabel("Edge recall at degree, mean of nine heads")
    ax.set_title("Attention sits on the graph")
    _ladder_axis(ax, "λ (graph prior weight)")

    # (c) divergence of layer-1 attention from the nine normalized adjacencies, epoch 29,
    # validation batch; the penalty divided by lambda, so comparable across the ladder.
    ax = axes[2]
    for arm in LADDER:
        sub = done[done.arm == arm]
        div = sub.val_divergence_fixed.dropna().to_numpy()
        if len(div):
            _seed_points(ax, xpos[arm], div, ARM_COLOR[arm], True)
            ax.plot(
                [xpos[arm] - 0.3, xpos[arm] + 0.3],
                [div.mean()] * 2,
                color=ARM_COLOR[arm],
                lw=1.0,
            )
    ax.set_yscale("log")
    ax.set_ylim(2e2, 3e3)
    ax.text(xpos["kl_0"], 2.2e2, "not logged", ha="center", fontsize=6, color=GRAY)
    ax.text(xpos["mask"], 2.2e2, "not logged", ha="center", fontsize=6, color=RED)
    ax.set_ylabel("Divergence to the graphs, epoch 29")
    ax.set_title("How close the attention gets")
    _ladder_axis(ax, "λ (graph prior weight)")

    # (d) gradient budget on the probe batch: penalty over point-loss gradient norm.
    ax = axes[3]
    for arm in LADDER:
        sub = runs[runs.arm == arm]
        c = ARM_COLOR[arm]
        r0 = sub.probe_ratio_epoch0.dropna().to_numpy()
        r20 = sub.probe_ratio_epoch20.dropna().to_numpy()
        if len(r0) and r0.max() > 0:
            _seed_points(ax, xpos[arm] - 0.15, r0, c, True)
        if len(r20) and r20.max() > 0:
            _seed_points(ax, xpos[arm] + 0.15, r20, c, False)
    ax.axhline(1.0, color="black", lw=0.5, ls="--")
    ax.text(xpos["kl_0"], 1.3e-3, "0", ha="center", fontsize=6, color=GRAY)
    ax.text(xpos["mask"], 1.3e-3, "0", ha="center", fontsize=6, color=RED)
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 3e3)
    ax.set_ylabel("Gradient norm ratio, penalty over point loss")
    ax.set_title("Where the gradient comes from")
    ax.scatter(
        [], [], s=6, facecolor=ORANGE, edgecolor=ORANGE, linewidth=0.5, label="epoch 0"
    )
    ax.scatter(
        [],
        [],
        s=6,
        facecolor="white",
        edgecolor=ORANGE,
        linewidth=0.5,
        label="epoch 20",
    )
    ax.legend(loc="upper left", frameon=False, handletextpad=0.3, borderpad=0.2)
    _ladder_axis(ax, "λ (graph prior weight)")

    for ax, letter in zip(axes, "abcd"):
        panel_label(ax, letter)
    _save(fig, "graph_reg_sweep_ladder")


def _curve(
    ax: Axes, hist: pd.DataFrame, arm: str, key: str, color: str, ls: str, label: str
) -> None:
    sub = hist[(hist.arm == arm) & (hist.key == key)]
    if sub.empty:
        return
    g = sub.groupby("epoch")["value"]
    n = g.count()
    mean = g.mean()[n == n.max()]  # epochs every seed reached
    sd = g.std()[n == n.max()]
    ax.plot(mean.index, mean.to_numpy(), color=color, lw=0.9, ls=ls, label=label)
    if (n.max() > 1) and sd.notna().any():
        ax.fill_between(
            mean.index,
            (mean - sd).to_numpy(),
            (mean + sd).to_numpy(),
            color=color,
            alpha=0.15,
            lw=0,
        )


def fig_curves(hist: pd.DataFrame) -> None:
    """Validation Pearson, training Pearson and validation point loss by epoch."""
    fig, axes = plt.subplots(
        1, 3, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(55))
    )
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.17, top=0.86, wspace=0.3)
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
    for ax, (key, ylabel, title) in zip(axes, panels):
        for arm, label, color, ls in CURVE_ARMS:
            _curve(ax, hist, arm, key, color, ls, label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xlim(0, 29)
        _box(ax)
    _tenths(axes[0])
    axes[0].legend(
        loc="lower right",
        frameon=False,
        handlelength=1.6,
        handletextpad=0.4,
        borderpad=0.2,
    )
    axes[1].yaxis.set_major_locator(MultipleLocator(0.1))
    axes[1].grid(axis="y", color="#DDDDDD", linewidth=0.3)
    for ax, letter in zip(axes, "abc"):
        panel_label(ax, letter)
    _save(fig, "graph_reg_sweep_curves")


def fig_control(runs: pd.DataFrame, hist: pd.DataFrame) -> None:
    """The random-graph control: is it the biology or any target, at lambda 1e-3."""
    fig, axes = plt.subplots(
        1, 2, figsize=(mm_to_in(PANEL_WIDTHS_MM["half_plus"]), mm_to_in(55))
    )
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.2, top=0.86, wspace=0.4)
    arms = [
        ("kl_0.001", "biological", ORANGE),
        (RANDOM, "random", PURPLE),
        ("kl_0", "none", GRAY),
    ]
    ax = axes[0]
    for i, (arm, label, color) in enumerate(arms):
        sub = runs[(runs.arm == arm) & runs.complete]
        part = runs[(runs.arm == arm) & ~runs.complete]
        if len(sub):
            m = sub.val_pearson_fixed.mean()
            ax.bar(i, m, width=0.6, facecolor="white", edgecolor=color, linewidth=0.8)
            _seed_points(
                ax, i, sub.val_pearson_fixed.to_numpy(), color, True, jitter=0.12
            )
        for r in part.to_dict("records"):
            vmax, ep = float(r["val_pearson_max"]), int(r["epochs_logged"]) - 1
            ax.scatter([i + 0.22], [vmax], s=8, marker="x", color=color, linewidth=0.6)
            ax.text(
                i + 0.22,
                vmax + 0.004,
                f"ep {ep}\npartial",
                ha="center",
                fontsize=5,
                color=color,
            )
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([a[1] for a in arms])
    ax.set_ylim(0.40, 0.46)
    ax.set_ylabel("Held-out Pearson, epoch 29")
    ax.set_title("Graph target of the KL at λ = 0.001")
    _tenths(ax)
    _box(ax)
    ax = axes[1]
    for arm, label, color in arms:
        _curve(
            ax,
            hist,
            arm,
            "val/gene_interaction/Pearson",
            color,
            "-" if arm != RANDOM else "--",
            {
                "biological": "biological graphs",
                "random": "random, degree-matched",
                "none": "none (λ = 0)",
            }[label],
        )
    ax.set_xlim(0, 29)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Pearson, gene interaction")
    ax.set_title("Mean over seeds, ± sd")
    ax.legend(
        loc="lower right",
        frameon=False,
        handlelength=1.6,
        handletextpad=0.4,
        borderpad=0.2,
    )
    _tenths(ax)
    _box(ax)
    for ax, letter in zip(axes, "ab"):
        panel_label(ax, letter)
    _save(fig, "graph_reg_sweep_control")


def main() -> None:
    """All three figures from the CSVs."""
    runs = pd.read_csv(osp.join(RESULTS_DIR, "graph_reg_sweep_runs.csv"))
    hist = pd.read_csv(osp.join(RESULTS_DIR, "graph_reg_sweep_history.csv"))
    fig_ladder(runs)
    fig_curves(hist)
    fig_control(runs, hist)
    print("wrote", IMG_DIR)


if __name__ == "__main__":
    main()
