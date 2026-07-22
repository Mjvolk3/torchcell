# experiments/019-echo-crispr-array/scripts/compare_cellpose_vs_previous_scoring.py
# [[experiments.019-echo-crispr-array.scripts.compare_cellpose_vs_previous_scoring]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-echo-crispr-array/scripts/compare_cellpose_vs_previous_scoring
"""Does the Cellpose sizing SCORE better than the previous (classical) sizing?

The previous pipeline judged per-strain fitness against the published Costanzo
single-mutant fitness (``reference_smf_12panel.csv``); the metric of record is the
per-condition Pearson/Spearman to Costanzo (``run2_vs_reference_stats.csv``, best
classical was P1_t50 r=0.76). This recomputes that SAME benchmark for Cellpose and
places the two side by side, plus the direct Cellpose-vs-classical fitness
agreement. Reads the fitness tables both segmenters already wrote; no re-sizing.

Inputs (results/):
  run2_cellpose_fitness_by_condition.csv  (Cellpose per-strain fitness)
  run2_fitness_by_condition.csv           (classical / previous per-strain fitness)
  reference_smf_12panel.csv               (Costanzo ground truth)
Outputs:
  results/run2_cellpose_scoring_vs_previous.csv
  ASSET_IMAGES_DIR/019-echo-crispr-array/cellpose/run2_scoring_vs_reference.{png,svg}

Run from repo root:
    ~/miniconda3/envs/torchcell/bin/python \
        experiments/019-echo-crispr-array/scripts/compare_cellpose_vs_previous_scoring.py
"""

from __future__ import annotations

import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.ticker import MultipleLocator
from scipy.stats import pearsonr, spearmanr

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


def _vs_reference(fit: pd.DataFrame, ref: pd.DataFrame, group: str) -> dict[str, float]:
    """Per-condition agreement of a segmenter's per-strain fitness with the Costanzo
    reference: Pearson r, Spearman rho, RMSE, median bias over the shared strains.
    """
    d = (
        fit[fit["group"] == group][["strain", "fitness"]]
        .merge(ref[["strain", "costanzo_smf"]], on="strain", how="inner")
        .dropna(subset=["fitness", "costanzo_smf"])
    )
    x, y = d["fitness"].to_numpy(float), d["costanzo_smf"].to_numpy(float)
    if len(x) < 3:
        return dict(n=len(x), pearson_r=np.nan, spearman_rho=np.nan, rmse=np.nan, median_bias=np.nan)
    return dict(
        n=len(x),
        pearson_r=float(pearsonr(x, y)[0]),
        spearman_rho=float(spearmanr(x, y)[0]),
        rmse=float(np.sqrt(np.mean((x - y) ** 2))),
        median_bias=float(np.median(x - y)),
    )


def main() -> None:
    cp = pd.read_csv(osp.join(RESULTS_DIR, "run2_cellpose_fitness_by_condition.csv"))
    cl = pd.read_csv(osp.join(RESULTS_DIR, "run2_fitness_by_condition.csv"))
    ref = pd.read_csv(osp.join(RESULTS_DIR, "reference_smf_12panel.csv"))

    rows = []
    for g in GROUPS:
        c = _vs_reference(cp, ref, g)
        p = _vs_reference(cl, ref, g)
        # direct cellpose-vs-classical per-strain fitness agreement
        m = (
            cp[cp["group"] == g][["strain", "fitness"]]
            .merge(cl[cl["group"] == g][["strain", "fitness"]], on="strain", suffixes=("_cp", "_cl"))
            .dropna()
        )
        cp_cl_r = float(pearsonr(m["fitness_cp"], m["fitness_cl"])[0]) if len(m) > 2 else np.nan
        rows.append(
            dict(
                group=g,
                n=c["n"],
                cellpose_r=c["pearson_r"],
                classical_r=p["pearson_r"],
                d_r=c["pearson_r"] - p["pearson_r"],
                cellpose_rho=c["spearman_rho"],
                classical_rho=p["spearman_rho"],
                cellpose_rmse=c["rmse"],
                classical_rmse=p["rmse"],
                cellpose_bias=c["median_bias"],
                classical_bias=p["median_bias"],
                cp_vs_cl_r=cp_cl_r,
            )
        )
    out = pd.DataFrame(rows)
    out.to_csv(osp.join(RESULTS_DIR, "run2_cellpose_scoring_vs_previous.csv"), index=False)
    print("Per-condition fitness-vs-Costanzo (Pearson r) and Cellpose-vs-classical:")
    print(
        out[
            ["group", "n", "cellpose_r", "classical_r", "d_r", "cellpose_rho", "classical_rho", "cp_vs_cl_r"]
        ].round(3).to_string(index=False)
    )
    win = (out["d_r"] > 0).sum()
    print(f"\nCellpose beats classical on Pearson-r vs Costanzo in {win}/{len(out)} conditions "
          f"(mean dr = {out['d_r'].mean():+.3f}).")

    # ---- figure: per-condition Pearson r to Costanzo, Cellpose vs classical ----
    plt.rcParams.update(
        {"font.family": "Arial", "font.size": 6, "svg.fonttype": "none", "axes.linewidth": 0.5}
    )
    w_in = mm_to_in(PANEL_WIDTHS_MM["half"])
    fig, ax = plt.subplots(figsize=(w_in, mm_to_in(55)))
    x = np.arange(len(GROUPS))
    bw = 0.38
    ax.bar(x - bw / 2, out["cellpose_r"], bw, color=PLOT_PALETTE[0], edgecolor="black",
           linewidth=0.5, label="Cellpose")
    ax.bar(x + bw / 2, out["classical_r"], bw, color=PLOT_PALETTE[4], edgecolor="black",
           linewidth=0.5, label="Classical")
    ax.set_xticks(x)
    ax.set_xticklabels(GROUPS, rotation=45, ha="right")
    ax.set_ylabel("Pearson r vs Costanzo SMF")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which="minor", length=0)
    ax.grid(axis="y", which="both", linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    for s in ax.spines.values():
        s.set_visible(True)
    fig.tight_layout()
    fig.savefig(osp.join(IMG_DIR, "run2_scoring_vs_reference.png"), dpi=300)
    savefig_true_size_svg(fig, osp.join(IMG_DIR, "run2_scoring_vs_reference.svg"))
    plt.close(fig)
    print(f"\nwrote results -> {osp.join(RESULTS_DIR, 'run2_cellpose_scoring_vs_previous.csv')}")
    print(f"wrote figure  -> {osp.join(IMG_DIR, 'run2_scoring_vs_reference.png/.svg')}")


if __name__ == "__main__":
    main()
