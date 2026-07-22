# experiments/019-echo-crispr-array/scripts/cellpose_error_analysis.py
# [[experiments.019-echo-crispr-array.scripts.cellpose_error_analysis]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-echo-crispr-array/scripts/cellpose_error_analysis
"""Per-genotype standard deviation and standard error for the Cellpose measurements.

For the tuned Cellpose method (CLAHE + cellprob -4 + the colony-validity invalidation),
each strain's fitness is the mean over its replicate colonies (invalidated `M`/`N`/`C`
wells excluded). We report both the replicate **SD** and, since p-values pair with the
standard error, **SE = SD / sqrt(n_used)** -- the reference-comparable uncertainty, on
the same footing as Costanzo's double-mutant SE (sample SD over 4 colonies; we have
~8-11). Reads `run2_cellpose_fitness_by_condition.csv` (which now carries `fitness_se`).

Outputs:
  results/run2_cellpose_error_by_strain.csv   (fitness, SD, SE, n per strain x condition)
  ASSET_IMAGES_DIR/019-echo-crispr-array/cellpose/run2_cellpose_fitness_se.{png,svg}

Run from repo root:
    ~/miniconda3/envs/torchcell/bin/python \
        experiments/019-echo-crispr-array/scripts/cellpose_error_analysis.py
"""

from __future__ import annotations

import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
EXP_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
RESULTS_DIR = osp.join(EXP_DIR, "results")
IMG_DIR = osp.join(ASSET_IMAGES_DIR, "019-echo-crispr-array", "cellpose")
os.makedirs(IMG_DIR, exist_ok=True)

GROUPS = ["P1_t44", "P2_t44", "P1_t50", "P2_t50", "P1_t72", "P2_t72"]
LABEL = {
    "P1_t44": "2.5 nL, 44 h", "P2_t44": "5 nL, 44 h",
    "P1_t50": "2.5 nL, 50 h", "P2_t50": "5 nL, 50 h",
    "P1_t72": "2.5 nL, 72 h", "P2_t72": "5 nL, 72 h",
}


def main() -> None:
    fit = pd.read_csv(osp.join(RESULTS_DIR, "run2_cellpose_fitness_by_condition.csv"))
    if "fitness_se" not in fit.columns:
        fit["fitness_se"] = fit["fitness_sd"] / np.sqrt(fit["n_used"].clip(lower=1))
    fit.to_csv(osp.join(RESULTS_DIR, "run2_cellpose_error_by_strain.csv"), index=False)

    print("Mean per-strain SD and SE by condition (Cellpose, tuned):")
    summ = (
        fit.groupby("group")[["fitness_sd", "fitness_se", "n_used"]]
        .mean()
        .reindex(GROUPS)
        .round(3)
    )
    print(summ.to_string())

    plt.rcParams.update(
        {"font.family": "Arial", "font.size": 6, "svg.fonttype": "none", "axes.linewidth": 0.5}
    )
    fig, axes = plt.subplots(2, 3, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(110)))
    for ax, g in zip(axes.ravel(), GROUPS):
        d = fit[fit["group"] == g].sort_values("fitness")
        x = np.arange(len(d))
        ax.bar(x, d["fitness"], color=PLOT_PALETTE[0], edgecolor="black", linewidth=0.4)
        ax.errorbar(
            x, d["fitness"], yerr=d["fitness_se"], fmt="none",
            ecolor="black", elinewidth=0.5, capsize=1.5,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(d["strain"], rotation=90, fontsize=4)
        ax.set_ylim(0, 1.6)
        ax.axhline(1.0, color=PLOT_PALETTE[5], lw=0.4, ls="--")
        ax.set_title(LABEL[g], fontsize=6)
        for s in ax.spines.values():
            s.set_visible(True)
        if g in ("P1_t44", "P2_t50"):
            ax.set_ylabel("fitness (mut/WT) ± SE")
    fig.suptitle(
        "Cellpose per-genotype fitness with standard error (SE = SD/√n over replicate colonies)",
        fontsize=7,
    )
    fig.tight_layout()
    fig.savefig(osp.join(IMG_DIR, "run2_cellpose_fitness_se.png"), dpi=300)
    savefig_true_size_svg(fig, osp.join(IMG_DIR, "run2_cellpose_fitness_se.svg"))
    plt.close(fig)
    print(f"\nwrote -> {osp.join(RESULTS_DIR, 'run2_cellpose_error_by_strain.csv')}")
    print(f"wrote -> {osp.join(IMG_DIR, 'run2_cellpose_fitness_se.png/.svg')}")


if __name__ == "__main__":
    main()
