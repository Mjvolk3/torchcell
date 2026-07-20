# experiments/010-kuzmin-tmi/scripts/topk_triples_from_constructed_10.py
# [[experiments.010-kuzmin-tmi.scripts.topk_triples_from_constructed_10]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/topk_triples_from_constructed_10
"""How many top-k triples are buildable from the 10 properly-constructed genes.

The wet-lab plate (exp-019) kept 10 of the inference_3 panel-12 genes and swapped
out two (YIL174W and LCL2/YLR104W) for SPH1/YLR313C and LCL1/YPL056C. This asks:
of the ranked constructible triples, how many can still be built from just those
10 "properly constructed" genes (RED) vs how many need one of the two dropped
genes (GRAY)?

Mirrors the reference figure `plot_predictions` in
investigate_YLR313C_smf_and_interactions.py (rank-vs-prediction scatter +
histogram), recolored by 10-gene membership using the repo PLOT_PALETTE.

Input : results/inference_3/triples_table_panel12_k200.csv (122 ranked constructible)
        results/inference_3/top_k_constructible_panel12_k200.csv (top-52 subset)
Output: notes/assets/images/010-kuzmin-tmi/topk_triples_from_constructed_10.{png,svg}

Run from repo root:
  ~/miniconda3/envs/torchcell/bin/python \
    experiments/010-kuzmin-tmi/scripts/topk_triples_from_constructed_10.py
"""
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.ticker import MultipleLocator

from torchcell.utils import (
    PLOT_PALETTE,
    PANEL_WIDTHS_MM,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]

RESULTS = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi", "results", "inference_3")
OUT_DIR = osp.join(ASSET_IMAGES_DIR, "010-kuzmin-tmi")

# The inference_3 panel-12 and the wet-lab plate share exactly these 10 genes;
# YIL174W and YLR104W (LCL2) were dropped from the plate.
TEN = {
    "YBR203W", "YDR057W", "YER079W", "YGL087C", "YJR060W",
    "YKL033W-A", "YLL012W", "YLR312C-B", "YPL046C", "YPL081W",
}
DROPPED = {"YIL174W", "YLR104W"}

COLOR_RED = PLOT_PALETTE[1]    # #B85450 — buildable from the 10 (properly constructed)
COLOR_GRAY = PLOT_PALETTE[5]   # #666666 — needs a dropped gene

# Repo figure standards: Arial 6 pt, editable SVG text.
plt.rcParams.update({
    "font.family": "Arial", "font.size": 6,
    "svg.fonttype": "none",
    "axes.linewidth": 0.5,
})


def within_ten(df: pd.DataFrame) -> np.ndarray:
    return df.apply(
        lambda r: {r.gene1, r.gene2, r.gene3}.issubset(TEN), axis=1
    ).to_numpy()


def box_axes(ax) -> None:
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(0.5)


def tenth_gridlines(ax, axis: str) -> None:
    loc = ax.yaxis if axis == "y" else ax.xaxis
    loc.set_major_locator(MultipleLocator(0.2))
    loc.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which="minor", length=0)
    ax.grid(which="both", axis=axis, lw=0.3, color="0.85", zorder=0)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(osp.join(RESULTS, "triples_table_panel12_k200.csv"))
    df = df.sort_values("prediction", ascending=False).reset_index(drop=True)
    red = within_ten(df)
    preds = df["prediction"].to_numpy()
    ranks = np.arange(1, len(df) + 1)
    n_red, n_gray = int(red.sum()), int((~red).sum())

    # top-52 subset counts (for annotation)
    topk = pd.read_csv(osp.join(RESULTS, "top_k_constructible_panel12_k200.csv"))
    topk_red = int(within_ten(topk).sum())

    fig, axes = plt.subplots(
        1, 2, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(72))
    )

    ax = axes[0]
    ax.scatter(ranks[~red], preds[~red], s=10, c=COLOR_GRAY, zorder=2,
               label=f"Needs a dropped gene (n={n_gray})")
    ax.scatter(ranks[red], preds[red], s=14, c=COLOR_RED, zorder=3,
               edgecolors="black", linewidths=0.3,
               label=f"Buildable from the 10 (n={n_red})")
    ax.axhline(0.0, color="black", lw=0.5, ls="--", alpha=0.6)
    ax.set_xlabel("Triple rank (by predicted interaction, desc)")
    ax.set_ylabel("Predicted gene interaction")
    ax.set_title("Constructible triples of the plate's 10 genes")
    tenth_gridlines(ax, "y")
    box_axes(ax)
    ax.legend(loc="upper right", frameon=True, fontsize=6)

    ax = axes[1]
    bins = np.linspace(preds.min(), preds.max(), 24)
    ax.hist(preds[~red], bins=bins, color=COLOR_GRAY, alpha=0.85,
            label=f"Needs dropped gene (μ={preds[~red].mean():.3f})")
    ax.hist(preds[red], bins=bins, color=COLOR_RED, alpha=0.85,
            label=f"From the 10 (μ={preds[red].mean():.3f})")
    ax.set_xlabel("Predicted gene interaction")
    ax.set_ylabel("Triple count")
    ax.set_title(f"top-52 subset: {topk_red}/{len(topk)} from the 10")
    tenth_gridlines(ax, "x")
    box_axes(ax)
    ax.legend(loc="upper right", frameon=True, fontsize=6)

    fig.suptitle(
        f"Panel-12 inference-3 triples buildable from the 10 properly-constructed "
        f"genes — {n_red}/{len(df)} (red)",
        fontsize=7,
    )
    fig.tight_layout()

    png = osp.join(OUT_DIR, "topk_triples_from_constructed_10.png")
    svg = osp.join(OUT_DIR, "topk_triples_from_constructed_10.svg")
    fig.savefig(png, dpi=200)
    savefig_true_size_svg(fig, svg)
    plt.close(fig)

    print(f"122 ranked constructible: {n_red} buildable from the 10, {n_gray} need a dropped gene")
    print(f"  contain YIL174W: {int(df.apply(lambda r: 'YIL174W' in {r.gene1,r.gene2,r.gene3}, axis=1).sum())}")
    print(f"  contain YLR104W: {int(df.apply(lambda r: 'YLR104W' in {r.gene1,r.gene2,r.gene3}, axis=1).sum())}")
    print(f"top-52 subset: {topk_red}/{len(topk)} buildable from the 10")
    print(f"saved: {png}\n       {svg}")


if __name__ == "__main__":
    main()
