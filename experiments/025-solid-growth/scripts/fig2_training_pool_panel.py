# experiments/025-solid-growth/scripts/fig2_training_pool_panel.py
# [[experiments.025-solid-growth.scripts.fig2_training_pool_panel]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/025-solid-growth/scripts/fig2_training_pool_panel
"""Figure 2 panel: does training on the lower orders too lift trigenic accuracy?

Reads s3_closure_readout.csv (written by s3_closure_readout.py, which scores every run as
the mean validation Pearson over the fixed epoch window 10 to 29) and draws ONE
third-width panel, fig2_training_pool_panel, to $ASSET_IMAGES_DIR/025-solid-growth/:
the same cell graph transformer read on the same 37,673 validation triples, trained on
three pools of the 025 solid-growth build.

  S0, interaction only     the 376,732 Kuzmin triples of experiment 010, interaction head
  S0, + fitness head       the same triples, interaction and fitness heads (joint, 1.0)
  S3 closure, + fitness    S0 plus every single and every double whose genes lie inside
                           some triple (5,694 singles, 739,219 doubles), joint heads

The 030 per-entry build (13.5 M measurement entries, every source entry kept) is the next
pool and has not been trained; the panel marks it as not run rather than leaving it out.
Bar is the seed mean, whisker the sample sd, every seed an open circle. Repo figure
standards: Arial 6 pt, boxed axes, primaries of torchcell.utils, true-size SVG.
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
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "025-solid-growth", "results")
IMG_DIR = osp.join(ASSET_IMAGES_DIR, "025-solid-growth")

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

# (arm in the readout, two-line label with the pool size, palette index). The pool that
# lifts the score is brick red, the color Figure 2 gives TorchCell; the two S0 pools
# are gray and yellow.
POOLS = [
    ("s0_ctrl_013", "S0\n377 k", 5),
    ("s0_fit_014", "S0 + fitness\nhead, 377 k", 3),
    ("s3_fit_031", "S3 closure\n1.12 M", 1),
]
NEXT = "030 per entry\n13.5 M"
SCORE = "val_gi_pearson_mean_ep10_29"
GRID_COLOR, GRID_LW = "#E3E3E3", 0.3


def main() -> None:
    d = pd.read_csv(osp.join(RESULTS_DIR, "s3_closure_readout.csv"))
    fig, ax = plt.subplots(
        figsize=(mm_to_in(PANEL_WIDTHS_MM["third"]), mm_to_in(52))
    )
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.2, top=0.84)
    for x, (arm, label, color) in enumerate(POOLS):
        vals = d[d.arm == arm][SCORE].dropna().to_numpy(dtype=float)
        sd = vals.std(ddof=1)
        ax.bar(
            x,
            vals.mean(),
            width=0.62,
            color=PLOT_PALETTE[color],
            edgecolor="black",
            linewidth=0.4,
            yerr=sd,
            error_kw={"elinewidth": 0.5, "capsize": 1.5, "capthick": 0.5},
            zorder=3,
        )
        ax.scatter(
            x + np.linspace(-0.16, 0.16, len(vals)),
            vals,
            s=7,
            facecolor="white",
            edgecolor="black",
            linewidth=0.4,
            zorder=4,
        )
        print(f"{arm:12s} n={len(vals)} mean {vals.mean():.4f} sd {sd:.4f} {vals.round(4).tolist()}")
    x = len(POOLS)
    ax.scatter([x], [0.383], s=12, marker="x", color=PLOT_PALETTE[1], linewidth=0.6)
    ax.text(x, 0.387, "not\nrun", ha="center", va="bottom", fontsize=5, color=PLOT_PALETTE[1])
    ax.set_xticks(range(len(POOLS) + 1))
    ax.set_xticklabels([label for _, label, _ in POOLS] + [NEXT], fontsize=5)
    ax.set_xlim(-0.6, len(POOLS) + 0.6)
    ax.set_ylim(0.38, 0.49)
    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.yaxis.set_minor_locator(MultipleLocator(0.01))
    ax.tick_params(which="minor", length=0)
    ax.grid(axis="y", which="both", color=GRID_COLOR, linewidth=GRID_LW)
    ax.set_axisbelow(True)
    ax.set_ylabel("Validation Pearson, epochs 10 to 29")
    ax.set_xlabel("training pool (records), same validation triples")
    ax.set_title("Lower-order data lifts trigenic accuracy by 0.04")
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)
    ax.tick_params(width=0.5, length=2)
    # No panel letter here: the draw.io figure places its own letters in the slot
    # frames (Fig2-gene-interactions.compose.py).
    os.makedirs(IMG_DIR, exist_ok=True)
    stem = osp.join(IMG_DIR, "fig2_training_pool_panel")
    savefig_true_size_svg(fig, stem + ".svg")
    fig.savefig(stem + ".png", dpi=300)
    savefig_true_size_svg(fig, f"{stem}_{timestamp()}.svg")
    print("wrote", stem + ".svg")


if __name__ == "__main__":
    main()
