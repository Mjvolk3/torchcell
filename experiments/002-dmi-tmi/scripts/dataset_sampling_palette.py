# experiments/002-dmi-tmi/scripts/dataset_sampling_palette
# [[experiments.002-dmi-tmi.scripts.dataset_sampling_palette]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/002-dmi-tmi/scripts/dataset_sampling_palette
#
# Fig 7 sampling panel: how the 10^3 / 10^4 / 10^5 subsets are split into
# train/val/test (80/10/10, stratified, seed 42 -- identical scheme for the
# fitness-001 and interaction-002 experiments). Counts are read from the on-disk
# split X.npy so the bars are the real split sizes, not nominal ratios.

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

# 002 has all splits on disk at every size; the 80/10/10 stratified scheme is
# identical for 001 (seed 42), so one experiment's real counts represent both.
ONE_HOT = "data/torchcell/experiments/002-dmi-tmi/traditional-ml/one_hot_gene"
SIZES = ["1e03", "1e04", "1e05"]
SIZE_LABELS = {"1e03": "$10^3$", "1e04": "$10^4$", "1e05": "$10^5$"}
SPLITS = ["train", "val", "test"]
# First three of the ordered plot palette (orange / red / purple); blue/gray are
# only used after the four primaries, so a 3-way split takes the first three.
SPLIT_COLORS = {
    "train": PLOT_PALETTE[0],
    "val": PLOT_PALETTE[1],
    "test": PLOT_PALETTE[2],
}


def split_count(size, split):
    p = osp.join(DATA_ROOT, ONE_HOT, f"sum_pert_{size}", split, "X.npy")
    return int(np.load(p).shape[0]) if osp.exists(p) else 0


def main():
    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_WIDTHS_MM["half"]), 2.0))
    x = np.arange(len(SIZES))
    width = 0.26
    for i, split in enumerate(SPLITS):
        counts = [split_count(s, split) for s in SIZES]
        bars = ax.bar(
            x + (i - 1) * width,
            counts,
            width,
            color=SPLIT_COLORS[split],
            label=split.capitalize(),
            edgecolor="black",
            linewidth=0.3,
        )
        for rect, c in zip(bars, counts):
            if c:
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    c,
                    f"{c:,}",
                    ha="center",
                    va="bottom",
                    fontsize=5,
                    rotation=90,
                )

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([SIZE_LABELS[s] for s in SIZES])
    ax.set_xlabel("Dataset size", fontsize=6)
    ax.set_ylabel("Samples", fontsize=6)
    ax.set_ylim(top=ax.get_ylim()[1] * 3)  # headroom for rotated count labels
    ax.legend(
        title="Split (80/10/10, stratified)",
        loc="upper left",
        ncol=3,
        handlelength=1.0,
        handletextpad=0.4,
        columnspacing=0.8,
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
    ax.grid(axis="y", which="major", ls="--", alpha=0.4)

    plt.subplots_adjust(left=0.14, right=0.97, top=0.95, bottom=0.16)
    savefig_true_size_svg(
        fig, osp.join(ASSET_IMAGES_DIR, "traditional-ml_dataset-sampling_palette.svg")
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
