# experiments/048-pputida-isoprenol-cgt/scripts/plot_low_data_trial.py
# [[experiments.048-pputida-isoprenol-cgt.scripts.plot_low_data_trial]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/048-pputida-isoprenol-cgt/scripts/plot_low_data_trial
"""The low-data trial's result figure: what the sweep bought, and what it cost.

Three panels, reading the sweep's own output files rather than recomputing
anything:

a
    Held-out Spearman against the graph-regularization lambda, one line per
    metabolic-module setting, with the boosted-trees score drawn as the line to
    beat. The objective of the trial on one axis.
b
    Spearman against Pearson for every cell, because the trial's first finding
    was that the two disagree about which model is better: the transformer wins
    on magnitude and the trees win on rank. A point above the diagonal ranks
    better than it calibrates.
c
    The selected cell's training curve, train and validation on one axis, which
    is the only way a 375-record fit shows its overfitting as it happens.

Reads ``results/low_data_trial/sweep_summary.csv`` and the per-cell reports
beside it. Writes the figure to ``ASSET_IMAGES_DIR`` as PNG and true-size SVG.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from matplotlib.ticker import MultipleLocator

from torchcell.timestamp import timestamp
from torchcell.utils import (
    PLOT_PALETTE,
    apply_paper_style,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()

ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
TRIAL = osp.join(
    EXPERIMENT_ROOT, "048-pputida-isoprenol-cgt", "results", "low_data_trial"
)

FIG_WIDTH_MM = 179.0
FIG_HEIGHT_MM = 62.0
FS_LABEL = 6.0
FS_TITLE = 6.0
FS_LEGEND = 5.0
FS_PANEL_LETTER = 8.0


def lambda_ticks(frame: pd.DataFrame) -> list[float]:
    """The lambda ladder, in order, as categorical positions.

    Lambda 0 has no place on a log axis, and the ladder is a ladder rather than a
    continuum, so panel a's x axis is categorical: evenly spaced ticks in the
    ladder's own order. Placing 0 at some arbitrary small number would draw a
    value the sweep never ran.
    """
    return sorted(frame.graph_reg_lambda.unique())


def panel_objective(ax: plt.Axes, frame: pd.DataFrame) -> None:
    """Held-out Spearman against lambda, one line per metabolic-module setting."""
    ladder = lambda_ticks(frame)
    positions = {lam: index for index, lam in enumerate(ladder)}
    for color_index, (flux, label) in enumerate(
        ((False, "pooled head"), (True, "flux metabolism"))
    ):
        subset = frame[frame.flux == flux]
        if subset.empty:
            continue
        grouped = subset.groupby("graph_reg_lambda").test_spearman.agg(["mean", "std"])
        x = [positions[lam] for lam in grouped.index]
        ax.errorbar(
            x,
            grouped["mean"],
            yerr=grouped["std"].fillna(0.0),
            marker="o",
            markersize=2.5,
            linewidth=0.8,
            capsize=1.5,
            color=PLOT_PALETTE[color_index],
            label=label,
        )
    trees = float(frame.trees_test_spearman.mean())
    ax.axhline(
        trees,
        color=PLOT_PALETTE[5],
        linewidth=0.8,
        linestyle="--",
        label=f"boosted trees ({trees:.3f})",
    )
    ax.set_xticks(list(positions.values()))
    ax.set_xticklabels([("0" if lam == 0 else f"{lam:g}") for lam in ladder])
    ax.set_xlabel("graph regularization lambda", fontsize=FS_LABEL)
    ax.set_ylabel("test Spearman", fontsize=FS_LABEL)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which="minor", length=0)
    ax.legend(fontsize=FS_LEGEND, frameon=False, loc="lower left")


def panel_rank_versus_scale(ax: plt.Axes, frame: pd.DataFrame) -> None:
    """Spearman against Pearson per cell, with the identity line for reference."""
    for color_index, (flux, label) in enumerate(
        ((False, "pooled head"), (True, "flux metabolism"))
    ):
        subset = frame[frame.flux == flux]
        if subset.empty:
            continue
        ax.scatter(
            subset.test_pearson,
            subset.test_spearman,
            s=8.0,
            color=PLOT_PALETTE[color_index],
            edgecolor="black",
            linewidth=0.3,
            label=label,
            zorder=3,
        )
    low = float(min(frame.test_pearson.min(), frame.test_spearman.min())) - 0.05
    high = float(max(frame.test_pearson.max(), frame.test_spearman.max())) + 0.05
    ax.plot([low, high], [low, high], color="#999999", linewidth=0.5, zorder=1)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("test Pearson", fontsize=FS_LABEL)
    ax.set_ylabel("test Spearman", fontsize=FS_LABEL)
    ax.legend(fontsize=FS_LEGEND, frameon=False, loc="upper left")


def panel_curve(ax: plt.Axes, report: dict) -> None:
    """The selected cell's train and validation curves on one axis."""
    history = pd.DataFrame(report["history"])
    ax.plot(
        history.epoch,
        history.train_spearman,
        color=PLOT_PALETTE[0],
        linewidth=0.8,
        label="train",
    )
    ax.plot(
        history.epoch,
        history.val_spearman,
        color=PLOT_PALETTE[4],
        linewidth=0.8,
        label="validation",
    )
    ax.axvline(
        report["best_epoch"],
        color="#666666",
        linewidth=0.5,
        linestyle=":",
        label=f"selected (epoch {report['best_epoch']})",
    )
    ax.set_xlabel("epoch", fontsize=FS_LABEL)
    ax.set_ylabel("Spearman", fontsize=FS_LABEL)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which="minor", length=0)
    ax.legend(fontsize=FS_LEGEND, frameon=False, loc="lower right")
    ax.set_title(report["args"]["tag"], fontsize=FS_TITLE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-timestamp", action="store_true")
    args = parser.parse_args()

    apply_paper_style()
    summary = osp.join(TRIAL, "sweep_summary.csv")
    frame = pd.read_csv(summary)
    design = frame[frame.split == "design"]
    if design.empty:
        raise ValueError(
            f"{summary} holds no design-split cells, so there is no sweep to "
            f"plot; the trial's configuration grid runs on the design split"
        )

    best = design.sort_values("test_spearman", ascending=False).iloc[0]
    report_path = osp.join(
        TRIAL, f"train_{best.split}_{best.tag}_seed{int(best.seed)}.json"
    )
    with open(report_path) as handle:
        report = json.load(handle)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(mm_to_in(FIG_WIDTH_MM), mm_to_in(FIG_HEIGHT_MM)),
        gridspec_kw={"width_ratios": [1.05, 0.95, 1.0], "wspace": 0.3},
    )
    panel_objective(axes[0], design)
    panel_rank_versus_scale(axes[1], design)
    panel_curve(axes[2], report)
    for index, ax in enumerate(axes):
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_visible(True)
        ax.tick_params(labelsize=FS_LABEL)
        ax.text(
            -0.18,
            1.04,
            "abc"[index],
            transform=ax.transAxes,
            fontsize=FS_PANEL_LETTER,
            fontweight="bold",
            va="bottom",
        )
    fig.subplots_adjust(left=0.06, right=0.99, top=0.92, bottom=0.17)

    out_dir = osp.join(ASSET_IMAGES_DIR, "048-pputida-isoprenol-cgt")
    os.makedirs(out_dir, exist_ok=True)
    stem = "low_data_trial"
    if not args.no_timestamp:
        stem = f"{stem}_{timestamp()}"
    png = osp.join(out_dir, f"{stem}.png")
    svg = osp.join(out_dir, f"{stem}.svg")
    fig.savefig(png, dpi=400)
    savefig_true_size_svg(fig, svg)
    plt.close(fig)

    print(f"cells plotted {len(design)}")
    print(f"best {best.tag} seed {int(best.seed)} test rho {best.test_spearman:.3f}")
    print(png)
    print(svg)


if __name__ == "__main__":
    main()
