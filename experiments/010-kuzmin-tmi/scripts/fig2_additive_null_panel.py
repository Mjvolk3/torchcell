# experiments/010-kuzmin-tmi/scripts/fig2_additive_null_panel.py
# [[experiments.010-kuzmin-tmi.scripts.fig2_additive_null_panel]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/fig2_additive_null_panel
"""Figure 2 panel: does the model learn interaction, or only what an additive null learns?

Reads the two result tables written by additive_baseline_gene_interaction.py and
query_pair_disjoint_cv.py (experiments/010-kuzmin-tmi/results/) and draws ONE third-width
panel, fig2_additive_null_panel, to $ASSET_IMAGES_DIR/010-kuzmin-tmi/: test Pearson on the
376,732 Kuzmin trigenic interaction scores for four learners, on the random split over
triples (left group) and on the five query-pair-disjoint folds (right group).

  additive ridge        4,352 gene indicators, no capacity for interaction (B1)
  additive + pair       B1 plus the 420 recurring query pairs (B2)
  gene-embedding MLP    the same indicators through a nonlinear map, three seeds (B5)
  CGT                   the 010 cell graph transformer, three checkpoints; not refit on
                        the disjoint folds, which the panel says

Bar is the mean, whisker the sample sd, every replicate an open circle (one bar with one
value has no whisker). Repo figure standards: Arial 6 pt, boxed axes, palette from
torchcell.utils, tenth gridlines on the 0 to 1 axis, true-size SVG, no bbox_inches="tight".
"""

from __future__ import annotations

import os
import os.path as osp

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.ticker import MultipleLocator

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
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi", "results")
IMG_DIR = osp.join(ASSET_IMAGES_DIR, "010-kuzmin-tmi")

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

# (label, prefix in the results tables, palette index). CGT is brick red, the color the
# state-of-the-art panel of Figure 2 gives TorchCell; the nulls are the cool neutrals.
LEARNERS = [
    ("additive ridge", "B1_additive_gene", 16),
    ("additive + pair", "B2_additive_plus_pair", 17),
    ("gene-embedding MLP", "B5_gene_embedding_mlp", 10),
    ("CGT", "CGT_", 1),
]
GROUPS = [("random split\nover triples", "random"), ("held-out\nquery pairs", "disjoint")]
GRID_COLOR, GRID_LW = "#E3E3E3", 0.3


def values(group: str, prefix: str, random: pd.DataFrame, cv: pd.DataFrame) -> np.ndarray:
    """Every held-out Pearson of one learner in one split."""
    if group == "random":
        sub = random[(random.split == "test") & random.model.str.startswith(prefix)]
        return sub.pearson.to_numpy(dtype=float)
    sub = cv[cv.model.str.startswith(prefix)]
    return sub.pearson.to_numpy(dtype=float)


def main() -> None:
    random = pd.read_csv(osp.join(RESULTS_DIR, "additive_baseline_gene_interaction.csv"))
    cv = pd.read_csv(osp.join(RESULTS_DIR, "query_pair_disjoint_cv.csv"))

    fig, ax = plt.subplots(
        figsize=(mm_to_in(PANEL_WIDTHS_MM["third"]), mm_to_in(52))
    )
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.2, top=0.84)
    width = 0.19
    for g, (glabel, group) in enumerate(GROUPS):
        for k, (label, prefix, color) in enumerate(LEARNERS):
            x = g + (k - 1.5) * width
            vals = values(group, prefix, random, cv)
            if len(vals) == 0:
                ax.scatter(
                    [x], [0.012], s=12, marker="x", color=PLOT_PALETTE[color], linewidth=0.6
                )
                ax.text(
                    x, 0.03, "not\nrefit", ha="center", va="bottom", fontsize=5,
                    color=PLOT_PALETTE[color],
                )
                continue
            sd = vals.std(ddof=1) if len(vals) > 1 else 0.0
            ax.bar(
                x,
                vals.mean(),
                width=width * 0.9,
                color=PLOT_PALETTE[color],
                edgecolor="black",
                linewidth=0.4,
                yerr=sd if len(vals) > 1 else None,
                error_kw={"elinewidth": 0.5, "capsize": 1.2, "capthick": 0.5},
                zorder=3,
                label=label if g == 0 else None,
            )
            jitter = np.linspace(-0.3, 0.3, len(vals)) * width if len(vals) > 1 else [0.0]
            ax.scatter(
                x + np.asarray(jitter),
                vals,
                s=6,
                facecolor="white",
                edgecolor="black",
                linewidth=0.4,
                zorder=4,
            )
            print(f"{group:8s} {label:20s} n={len(vals)} mean {vals.mean():.3f} sd {sd:.3f}")
    ax.set_xticks(range(len(GROUPS)))
    ax.set_xticklabels([g for g, _ in GROUPS])
    ax.set_xlim(-0.55, len(GROUPS) - 0.45)
    ax.set_ylim(0, 0.5)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which="minor", length=0)
    ax.grid(axis="y", which="both", color=GRID_COLOR, linewidth=GRID_LW)
    ax.set_axisbelow(True)
    ax.set_ylabel("Test Pearson, trigenic interaction")
    ax.set_title("A no-interaction null reaches 0.40 of CGT's 0.45")
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)
    ax.tick_params(width=0.5, length=2)
    ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=5,
        handlelength=1.0,
        handletextpad=0.4,
        borderpad=0.2,
        labelspacing=0.25,
    )
    panel_label(ax, "d")
    os.makedirs(IMG_DIR, exist_ok=True)
    stem = osp.join(IMG_DIR, "fig2_additive_null_panel")
    savefig_true_size_svg(fig, stem + ".svg")
    fig.savefig(stem + ".png", dpi=300)
    savefig_true_size_svg(fig, f"{stem}_{timestamp()}.svg")
    print("wrote", stem + ".svg")


if __name__ == "__main__":
    main()
