# experiments/025-solid-growth/scripts/fig2_graphs_help_panel.py
# [[experiments.025-solid-growth.scripts.fig2_graphs_help_panel]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/025-solid-growth/scripts/fig2_graphs_help_panel
"""Figure 2 panel: do the nine gene-gene graphs help the cell graph transformer?

Reads graph_reg_sweep_runs.csv and graph_reg_sweep_summary.json (written by
graph_reg_sweep_readout.py) and draws ONE third-width panel, fig2_graphs_help_panel, to
$ASSET_IMAGES_DIR/025-solid-growth/: held-out validation Pearson at epoch 29, the
protocol's last epoch, for the four arms that answer the question under one protocol
(the 010 triples on their pinned split, 30 epochs, three seeds; five for the random arm):

  no penalty            the graphs never enter
  hard mask             layer-1 attention restricted to each graph's edges
  soft prior, lambda 1  a divergence penalty pulling layer-1 attention toward each graph
  soft prior, random    the same penalty toward degree-matched rewirings of the graphs

Bar is the seed mean, whisker the sample sd, every seed an open circle. Stars above a bar
are the paired t of that arm against no penalty on the same seeds (* p < 0.05, ** 0.01,
*** 0.001), read from the summary; the bracket is the paired t of the prior against its
random control on the three seeds they share, computed here from the runs table.
Repo figure standards: Arial 6 pt, boxed axes, palette from torchcell.utils, tenth
gridlines, true-size SVG, no bbox_inches="tight".
"""

from __future__ import annotations

import json
import os
import os.path as osp

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.ticker import MultipleLocator
from scipy import stats

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

# (arm in the runs table, label, palette index). The prior is brick red, the color
# Figure 2 gives TorchCell; the control arms are neutrals.
ARMS = [
    ("kl_0", "no\npenalty", 5),
    ("mask", "hard\nmask", 11),
    ("kl_1", "soft prior\nλ = 1", 1),
    ("random_1", "prior,\nrandom graphs", 17),
]
GRID_COLOR, GRID_LW = "#E3E3E3", 0.3


def stars(p: float | None) -> str:
    if p is None:
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def main() -> None:
    runs = pd.read_csv(osp.join(RESULTS_DIR, "graph_reg_sweep_runs.csv"))
    summary = json.load(open(osp.join(RESULTS_DIR, "graph_reg_sweep_summary.json")))
    pvals = {
        p["arm"]: p["p_two_sided"]
        for p in summary["paired_vs_no_penalty"]
        if p["reading"] == "fixed"
    }
    done = runs[runs.complete]

    fig, ax = plt.subplots(
        figsize=(mm_to_in(PANEL_WIDTHS_MM["third"]), mm_to_in(52))
    )
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.2, top=0.84)
    by_seed: dict[str, pd.Series] = {}
    for x, (arm, label, color) in enumerate(ARMS):
        sub = done[done.arm == arm].set_index("seed").val_pearson_fixed.dropna()
        by_seed[arm] = sub
        vals = sub.to_numpy(dtype=float)
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
        star = stars(pvals.get(arm))
        if star:
            ax.text(x, vals.max() + 0.012, star, ha="center", va="bottom", fontsize=6)
        print(f"{arm:10s} n={len(vals)} mean {vals.mean():.4f} sd {sd:.4f} p vs none {pvals.get(arm)}")

    # The prior against its random control, paired on the seeds both arms share.
    shared = sorted(set(by_seed["kl_1"].index) & set(by_seed["random_1"].index))
    diffs = np.array([by_seed["kl_1"][s] - by_seed["random_1"][s] for s in shared])
    t = diffs.mean() / (diffs.std(ddof=1) / np.sqrt(len(diffs)))
    p = float(2 * stats.t.sf(abs(t), df=len(diffs) - 1))
    print(f"prior minus random on seeds {shared}: {diffs.round(4).tolist()} mean {diffs.mean():+.4f} t {t:.1f} p {p:.3f}")
    y = 0.474
    ax.plot([2, 2, 3, 3], [y - 0.004, y, y, y - 0.004], color="black", lw=0.5)
    ax.text(
        2.5, y + 0.002, f"+{diffs.mean():.3f}, every seed{' ' + stars(p) if stars(p) else ''}",
        ha="center", va="bottom", fontsize=5,
    )

    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([label for _, label, _ in ARMS])
    ax.set_xlim(-0.6, len(ARMS) - 0.4)
    ax.set_ylim(0.38, 0.49)
    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.yaxis.set_minor_locator(MultipleLocator(0.01))
    ax.tick_params(which="minor", length=0)
    ax.grid(axis="y", which="both", color=GRID_COLOR, linewidth=GRID_LW)
    ax.set_axisbelow(True)
    ax.set_ylabel("Validation Pearson at epoch 29")
    ax.set_title("Graphs help as a soft prior toward the real graphs")
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)
    ax.tick_params(width=0.5, length=2)
    ax.text(
        0.02, 0.97,
        "stars: paired t vs no penalty\n* p < 0.05, ** 0.01, *** 0.001",
        transform=ax.transAxes, ha="left", va="top", fontsize=5,
    )
    panel_label(ax, "e")
    os.makedirs(IMG_DIR, exist_ok=True)
    stem = osp.join(IMG_DIR, "fig2_graphs_help_panel")
    savefig_true_size_svg(fig, stem + ".svg")
    fig.savefig(stem + ".png", dpi=300)
    savefig_true_size_svg(fig, f"{stem}_{timestamp()}.svg")
    print("wrote", stem + ".svg")


if __name__ == "__main__":
    main()
