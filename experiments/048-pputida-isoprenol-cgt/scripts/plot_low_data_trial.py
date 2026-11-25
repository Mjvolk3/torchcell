# experiments/048-pputida-isoprenol-cgt/scripts/plot_low_data_trial.py
# [[experiments.048-pputida-isoprenol-cgt.scripts.plot_low_data_trial]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/048-pputida-isoprenol-cgt/scripts/plot_low_data_trial
"""The low-data trial in three panels, at Nature Biotechnology print size.

Reads ``results/low_data_trial/trial_cells.csv``, written by
``summarize_low_data_trial.py``, so the figure and the note's tables cannot
disagree.

a
    The objective on the split that can carry it. Held-out Spearman against the
    graph-regularization lambda, one line per readout head, with the
    boosted-trees score as the line to beat. The trees see exactly what the
    transformer sees about the genotype, minus the graph and the sequence.
b
    Rank against scale, every cell of both splits. Spearman on x, R-squared on
    y, and the y axis crosses zero because negative R-squared is the
    interesting region: a model there ranks strains usefully and predicts
    titers worse than the test split's own mean. Every flux cell on the cycle
    split lands in it.
c
    What the design split's rank score is actually measuring. Ranking the design
    test set by DBTL cycle number alone beats every model and every null,
    because titer rose across the campaign and that split mixes cycles across
    train and test. The cycle split has no such escape, every one of its test
    records being cycle 6.

Nature Biotechnology standards, from the repo's figure section: 179 mm full
width, height well under the 170 mm cap, Arial through ``apply_paper_style``,
all text inside the 5 to 7 pt band with 8 pt bold lowercase panel letters, all
four spines at 0.5 pt, gridlines every 0.1 on the correlation axes labelled
every 0.2, colors from ``PLOT_PALETTE`` line colors rather than the pale object
fills, and a true-size SVG beside the PNG so the panel imports at exact
millimetres.
"""

from __future__ import annotations

import argparse
import os
import os.path as osp

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from matplotlib.lines import Line2D
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
FIG_HEIGHT_MM = 58.0
FS_AXIS = 6.0
FS_TICK = 6.0
FS_LEGEND = 5.0
FS_PANEL_LETTER = 8.0
FS_ANNOTATION = 5.0

#: Measured by cycle_confound_audit.py on the 44-record design-split test set.
CHRONOLOGY_RHO = 0.807
CHRONOLOGY_WITHIN_CYCLE = 0.426

POOLED_COLOR = PLOT_PALETTE[0]
FLUX_COLOR = PLOT_PALETTE[2]
NULL_COLOR = PLOT_PALETTE[5]


def box(ax: plt.Axes) -> None:
    """All four spines at 0.5 pt, which is the house look for a panel."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
    ax.tick_params(labelsize=FS_TICK, width=0.5)


def ticks(
    ax: plt.Axes, axis: str = "y", major: float = 0.2, minor: float = 0.1
) -> None:
    """Gridlines at ``minor``, labels at ``major``, minor tick marks hidden.

    The house default is the tenth ladder, 0.1 gridlines labelled every 0.2,
    which assumes an axis spanning most of 0 to 1. Two axes here do not: panel
    a is zoomed into a 0.45 band where that ladder would print two labels, and
    panel b's R-squared runs negative. Both pass their own spacing rather than
    keeping a ladder that makes the panel unreadable.
    """
    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_locator(MultipleLocator(major))
    target.set_minor_locator(MultipleLocator(minor))
    ax.grid(which="both", axis=axis, linewidth=0.3, color="#DDDDDD", zorder=0)
    ax.tick_params(which="minor", length=0)


def panel_objective(ax: plt.Axes, cells: pd.DataFrame) -> None:
    """Cycle-split Spearman against lambda, one line per head."""
    cycle = cells[cells.split == "cycle"]
    ladder = sorted(cycle.graph_reg_lambda.unique())
    position = {value: index for index, value in enumerate(ladder)}
    for flux, color, label in (
        (False, POOLED_COLOR, "pooled head"),
        (True, FLUX_COLOR, "flux metabolism"),
    ):
        subset = cycle[cycle.flux == flux]
        grouped = subset.groupby("graph_reg_lambda").test_spearman.agg(
            ["mean", "std", "count"]
        )
        ax.errorbar(
            [position[v] for v in grouped.index],
            grouped["mean"],
            yerr=grouped["std"].fillna(0.0),
            marker="o",
            markersize=2.5,
            linewidth=0.8,
            elinewidth=0.5,
            capsize=1.5,
            color=color,
            label=label,
            zorder=3,
        )
    trees = float(cycle.trees_test_spearman.mean())
    ax.axhline(
        trees,
        color=NULL_COLOR,
        linewidth=0.8,
        linestyle="--",
        zorder=2,
        label=f"boosted trees ({trees:.2f})",
    )
    ax.set_xticks(list(position.values()))
    ax.set_xticklabels(["0" if v == 0 else f"{v:g}" for v in ladder])
    ax.set_xlabel("graph regularization $\\lambda$", fontsize=FS_AXIS)
    ax.set_ylabel("held-out Spearman $\\rho$ (cycle 6)", fontsize=FS_AXIS)
    ax.set_xlim(-0.35, len(ladder) - 0.65)
    ax.set_ylim(0.30, 0.78)
    ticks(ax, major=0.1, minor=0.05)
    ax.legend(fontsize=FS_LEGEND, frameon=False, loc="lower left", handlelength=1.6)


def panel_rank_versus_scale(ax: plt.Axes, cells: pd.DataFrame) -> None:
    """Spearman against R-squared, every cell, split by head and by split."""
    # R-squared reaches about -8 on the worst flux cells, and an axis that
    # spans it compresses every informative point into the top eighth of the
    # panel. The axis is clipped at -1.5 and the cells below are drawn on the
    # floor as open markers with their count stated, so they are visible as
    # "off the scale" rather than silently dropped.
    floor = -1.5
    clipped = 0
    for flux, color in ((False, POOLED_COLOR), (True, FLUX_COLOR)):
        for split, marker in (("cycle", "o"), ("design", "^")):
            subset = cells[(cells.flux == flux) & (cells.split == split)]
            inside = subset[subset.test_r2 >= floor]
            outside = subset[subset.test_r2 < floor]
            clipped += len(outside)
            ax.scatter(
                inside.test_spearman,
                inside.test_r2,
                s=7.0,
                marker=marker,
                facecolor=color,
                edgecolor="black",
                linewidth=0.25,
                zorder=3,
            )
            ax.scatter(
                outside.test_spearman,
                [floor + 0.04] * len(outside),
                s=7.0,
                marker=marker,
                facecolor="none",
                edgecolor=color,
                linewidth=0.5,
                zorder=3,
            )
    ax.axhline(0.0, color="black", linewidth=0.5, zorder=2)
    ax.set_xlabel("held-out Spearman $\\rho$", fontsize=FS_AXIS)
    ax.set_ylabel("held-out $R^2$", fontsize=FS_AXIS)
    ax.set_ylim(floor, 1.0)
    ax.set_xlim(0.28, 0.88)
    ticks(ax, major=0.5, minor=0.25)
    ticks(ax, axis="x", major=0.2, minor=0.1)
    if clipped:
        ax.annotate(
            f"{clipped} cells below $R^2$ = {floor:g} (open)",
            xy=(0.03, floor + 0.08),
            xycoords=("axes fraction", "data"),
            fontsize=FS_ANNOTATION,
            ha="left",
            va="bottom",
        )
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=2.6,
            color=POOLED_COLOR,
            markeredgecolor="black",
            markeredgewidth=0.25,
            label="pooled",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=2.6,
            color=FLUX_COLOR,
            markeredgecolor="black",
            markeredgewidth=0.25,
            label="flux",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=2.6,
            color="#888888",
            markeredgecolor="black",
            markeredgewidth=0.25,
            label="cycle split",
        ),
        Line2D(
            [],
            [],
            marker="^",
            linestyle="",
            markersize=2.6,
            color="#888888",
            markeredgecolor="black",
            markeredgewidth=0.25,
            label="design split",
        ),
    ]
    ax.legend(
        handles=handles,
        fontsize=FS_LEGEND,
        frameon=False,
        loc="upper left",
        handlelength=1.0,
        ncol=2,
        columnspacing=0.8,
    )


def panel_confound(ax: plt.Axes, cells: pd.DataFrame) -> None:
    """The chronology null beside the models, on both splits."""
    design = cells[cells.split == "design"]
    cycle = cells[cells.split == "cycle"]
    bars = [
        ("cycle\nno.", CHRONOLOGY_RHO, NULL_COLOR, "design"),
        ("within\ncycle", CHRONOLOGY_WITHIN_CYCLE, NULL_COLOR, "design"),
        ("trees", float(design.trees_test_spearman.mean()), PLOT_PALETTE[3], "design"),
        ("CGT", float(design.test_spearman.max()), POOLED_COLOR, "design"),
        ("trees", float(cycle.trees_test_spearman.mean()), PLOT_PALETTE[3], "cycle"),
        ("CGT", float(cycle.test_spearman.max()), POOLED_COLOR, "cycle"),
    ]
    for index, (label, value, color, _) in enumerate(bars):
        ax.bar(
            index,
            value,
            width=0.68,
            facecolor=color,
            edgecolor="black",
            linewidth=0.4,
            zorder=3,
        )
        ax.text(
            index,
            value + 0.02,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=FS_ANNOTATION,
        )
    ax.axvline(3.5, color="black", linewidth=0.5, linestyle=":", zorder=2)
    ax.text(1.5, 0.95, "design split", ha="center", fontsize=FS_ANNOTATION)
    ax.text(4.5, 0.95, "cycle split", ha="center", fontsize=FS_ANNOTATION)
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([b[0] for b in bars], fontsize=FS_ANNOTATION)
    ax.set_ylabel("held-out Spearman $\\rho$", fontsize=FS_AXIS)
    ax.set_ylim(0.0, 1.05)
    ticks(ax, major=0.2, minor=0.1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-timestamp", action="store_true")
    args = parser.parse_args()

    apply_paper_style()
    path = osp.join(TRIAL, "trial_cells.csv")
    cells = pd.read_csv(path)
    if cells.empty:
        raise ValueError(
            f"{path} is empty; run summarize_low_data_trial.py after the trial "
            f"so the figure and the note's tables read the same numbers"
        )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(mm_to_in(FIG_WIDTH_MM), mm_to_in(FIG_HEIGHT_MM)),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.25], "wspace": 0.32},
    )
    panel_objective(axes[0], cells)
    panel_rank_versus_scale(axes[1], cells)
    panel_confound(axes[2], cells)
    for index, ax in enumerate(axes):
        box(ax)
        ax.set_axisbelow(True)
        ax.text(
            -0.16,
            1.03,
            "abc"[index],
            transform=ax.transAxes,
            fontsize=FS_PANEL_LETTER,
            fontweight="bold",
            va="bottom",
        )
    fig.subplots_adjust(left=0.055, right=0.995, top=0.94, bottom=0.21)

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

    print(f"cells plotted {len(cells)}")
    print(png)
    print(svg)


if __name__ == "__main__":
    main()
