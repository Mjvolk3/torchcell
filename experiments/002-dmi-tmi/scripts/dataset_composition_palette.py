# experiments/002-dmi-tmi/scripts/dataset_composition_palette
# [[experiments.002-dmi-tmi.scripts.dataset_composition_palette]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/002-dmi-tmi/scripts/dataset_composition_palette
#
# Fig 7 dataset-description panel: perturbation-order composition (single/double/
# triple mutant) of the fitness (smf-dmf-tmf-001) and interaction (002-dmi-tmi)
# traditional-ML datasets. Reads each experiment's one-hot X.npy (row-sum = number
# of genes perturbed) directly from DATA_ROOT -- both experiments live under it.

import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import torchcell
from torchcell.utils import savefig_true_size_svg, mm_to_in, PANEL_WIDTHS_MM, PLOT_PALETTE

load_dotenv()
plt.style.use(osp.join(osp.dirname(torchcell.__file__), "torchcell.mplstyle"))
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6,
        "axes.titlesize": 6,
        "axes.labelsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "legend.title_fontsize": 6,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.5,
        "lines.linewidth": 0.7,
        "patch.linewidth": 0.4,
        "savefig.bbox": "standard",
        "savefig.pad_inches": 0.01,
    }
)

DATA_ROOT = os.getenv("DATA_ROOT")
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")

# Largest subset with an "all" split on disk; composition is stable across the
# 1e3-1e5 subsets (verified), so one size represents the dataset.
SIZE = "1e04"
EXPERIMENTS = [
    ("Fitness (001)", "data/torchcell/experiments/smf-dmf-tmf-traditional-ml/one_hot_gene"),
    ("Interactions (002)", "data/torchcell/experiments/002-dmi-tmi/traditional-ml/one_hot_gene"),
]
# First three of the ordered plot palette (orange / red / purple) for
# single / double / triple.
ORDER_COLORS = {1: PLOT_PALETTE[0], 2: PLOT_PALETTE[1], 3: PLOT_PALETTE[2]}
ORDER_LABELS = {1: "Single (1)", 2: "Double (2)", 3: "Triple (3)"}


def load_perturbation_counts(one_hot_root, size):
    """Number of genes perturbed per sample = row-sum of the one-hot X."""
    base = osp.join(DATA_ROOT, one_hot_root, f"sum_pert_{size}")
    all_x = osp.join(base, "all", "X.npy")
    if osp.exists(all_x):
        X = np.load(all_x)
    else:
        X = np.concatenate(
            [
                np.load(osp.join(base, s, "X.npy"))
                for s in ("train", "val", "test")
                if osp.exists(osp.join(base, s, "X.npy"))
            ],
            axis=0,
        )
    return X.sum(axis=1).astype(int)


def main():
    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_WIDTHS_MM["half"]), 1.7))
    ypos = list(range(len(EXPERIMENTS)))[::-1]  # first experiment on top
    for y, (label, root) in zip(ypos, EXPERIMENTS):
        counts = load_perturbation_counts(root, SIZE)
        n = len(counts)
        left = 0.0
        for order in (1, 2, 3):
            pct = 100.0 * np.sum(counts == order) / n
            ax.barh(
                y,
                pct,
                left=left,
                height=0.6,
                color=ORDER_COLORS[order],
                edgecolor="black",
                linewidth=0.4,
            )
            if pct >= 3:
                ax.text(
                    left + pct / 2,
                    y,
                    f"{pct:.0f}%",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=6,
                )
            left += pct
        # No per-bar N: the composition is a size-invariant proportion (stratified /
        # balanced), identical across the 10^3-10^5 subsets, so a single N would
        # misleadingly imply one dataset size.

    ax.set_yticks(ypos)
    ax.set_yticklabels([e[0] for e in EXPERIMENTS], fontsize=6)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(EXPERIMENTS) - 0.4)
    ax.set_xlabel("Percent of dataset", fontsize=6)

    handles = [plt.Rectangle((0, 0), 1, 1, color=ORDER_COLORS[o]) for o in (1, 2, 3)]
    ax.legend(
        handles,
        [ORDER_LABELS[o] for o in (1, 2, 3)],
        title="Genes perturbed",
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        handlelength=1.0,
        handletextpad=0.4,
        columnspacing=1.0,
        borderpad=0.3,
        framealpha=0.5,
        fontsize=6,
    )
    # Full black box border (repo plotting standard).
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_edgecolor("black")
    ax.tick_params(width=0.5, length=2)

    plt.subplots_adjust(left=0.24, right=0.95, top=0.70, bottom=0.22)
    savefig_true_size_svg(
        fig,
        osp.join(ASSET_IMAGES_DIR, "traditional-ml_dataset-composition_palette.svg"),
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
