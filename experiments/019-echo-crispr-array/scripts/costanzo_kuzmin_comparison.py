# experiments/019-echo-crispr-array/scripts/costanzo_kuzmin_comparison.py
# [[experiments.019-echo-crispr-array.scripts.costanzo_kuzmin_comparison]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-echo-crispr-array/scripts/costanzo_kuzmin_comparison
"""Benchmark the CRISPR fitness assay against published single-mutant fitness (SMF).

Two deliverables, both on the tightened Cellpose measurements (halo-corrected sizes):

  (1) CORRELATION -- our per-genotype fitness vs Costanzo 2016 SMF, with BOTH the
      Pearson r (agreement in value) and Spearman rho (agreement in rank). Kuzmin
      2018 SMF is overlaid for the 4 genes it covers.
  (2) ERROR / SD COMPARISON -- our within-genotype replicate SD (the spread of the
      ~21-27 colony fitness measurements) vs Costanzo's reported SMF SD, per gene.
      Kuzmin's released SMF carries NO per-strain SD (only 4 point fitness values),
      so a Kuzmin-SD comparison is not possible from the released data -- flagged,
      not fabricated.

We benchmark the SINGLE capture closest to the planned assay -- 5 nL / 50 h (P2_t50),
the engineering target being 5 nL / 48 h -- not an average over settings we will not
run. Per-genotype fitness, replicate-colony SD, and SE all come from that one plate.
NOTE on a fair error comparison: Costanzo's SD column is a bootstrap SE across 17/350
control screens (already SE-like), whereas our SD is a raw sample SD across ~27 same-
plate colonies -- different kinds of number. The correlation uses SE-vs-SE error bars;
the SD figure shows both SDs with that caveat. Full derivation in the note section
"How Costanzo computes SD vs how we do" ([[torchcell.datasets.scerevisiae.costanzo2016.noise-computation]]).

Inputs (regenerate first, tightened recipe):
  results/run2_cellpose_error_by_strain.csv   (fitness, fitness_sd, fitness_se, n)
  results/reference_smf_12panel.csv           (costanzo/kuzmin SMF + SD)
Outputs:
  results/run2_reference_comparison.csv        (per-strain aggregate + reference)
  results/run2_reference_comparison_stats.csv  (Pearson/Spearman, mean-SD summary)
  ASSET_IMAGES_DIR/019-echo-crispr-array/cellpose/run2_fitness_correlation_reference.{png,svg}
  ASSET_IMAGES_DIR/019-echo-crispr-array/cellpose/run2_sd_vs_reference.{png,svg}

Run from repo root:
    ~/miniconda3/envs/torchcell/bin/python \
        experiments/019-echo-crispr-array/scripts/costanzo_kuzmin_comparison.py
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

WT_NAME = "BY4741"
# The panel is engineered at 5 nL, and the planned assay images at 48 h; of the six
# captures, 5 nL / 50 h (P2_t50) is the closest to that target, so we benchmark THAT
# single condition -- the one we will actually run -- not an average over settings we
# will not use.
CONDITION = "P2_t50"
CONDITION_LABEL = "5 nL, 50 h"
C_ORANGE, C_RED, C_PURPLE, C_GRAY = (
    PLOT_PALETTE[0],
    PLOT_PALETTE[1],
    PLOT_PALETTE[2],
    PLOT_PALETTE[5],
)


def _select(fit: pd.DataFrame) -> pd.DataFrame:
    """Per-strain measurements for the single engineering-target capture (CONDITION):
    fitness, the within-plate replicate-colony SD (raw sample SD over ~27 colonies),
    and SE = SD / sqrt(n_used).
    """
    d = fit[(fit["group"] == CONDITION) & (fit["strain"] != WT_NAME)]
    return pd.DataFrame(
        {
            "strain": d["strain"].to_numpy(),
            "our_fitness": d["fitness"].to_numpy(),
            "our_sd": d["fitness_sd"].to_numpy(),
            "our_se": d["fitness_se"].to_numpy(),
        }
    )


def _corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float, int]:
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 3:
        return (np.nan, np.nan, np.nan, np.nan, n)
    pr, pp = pearsonr(x[m], y[m])
    sr, sp = spearmanr(x[m], y[m])
    return (float(pr), float(pp), float(sr), float(sp), n)


def _fig_correlation(df: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "svg.fonttype": "none",
            "axes.linewidth": 0.5,
        }
    )
    fig, ax = plt.subplots(
        figsize=(mm_to_in(PANEL_WIDTHS_MM["half_plus"]), mm_to_in(88))
    )
    lim = (0.4, 1.3)
    ax.plot(lim, lim, ls="--", lw=0.5, color=C_GRAY, zorder=1)  # identity

    # Costanzo (all 12); x error = Costanzo SD column (a bootstrap SE), y error = our SE
    # -- SE-vs-SE, the like-for-like uncertainty (see the SD-method note).
    cst = df.dropna(subset=["costanzo_smf"])
    ax.errorbar(
        cst["costanzo_smf"],
        cst["our_fitness"],
        yerr=cst["our_se"],
        xerr=cst["costanzo_sd"],
        fmt="o",
        ms=3,
        mfc=C_ORANGE,
        mec="black",
        mew=0.4,
        ecolor="black",
        elinewidth=0.4,
        capsize=1.2,
        zorder=3,
        label="Costanzo 2016",
    )
    # Kuzmin (4 genes) overlaid as red circles
    kuz = df.dropna(subset=["kuzmin_smf"])
    ax.scatter(
        kuz["kuzmin_smf"],
        kuz["our_fitness"],
        s=16,
        marker="o",
        facecolor=C_RED,
        edgecolor="black",
        linewidth=0.4,
        zorder=4,
        label="Kuzmin 2018",
    )
    for _, r in cst.iterrows():
        ax.annotate(
            r["label"],
            (r["costanzo_smf"], r["our_fitness"]),
            fontsize=3.5,
            xytext=(3, 2),
            textcoords="offset points",
        )

    pr, pp, sr, sp, n = _corr(
        cst["costanzo_smf"].to_numpy(), cst["our_fitness"].to_numpy()
    )
    kpr, _, ksr, _, kn = _corr(
        kuz["kuzmin_smf"].to_numpy(), kuz["our_fitness"].to_numpy()
    )
    ax.text(
        0.03,
        0.97,
        f"Costanzo (n={n}): Pearson r={pr:.2f} (p={pp:.3f})\n"
        f"                 Spearman ρ={sr:.2f} (p={sp:.3f})\n"
        f"Kuzmin (n={kn}): Pearson r={kpr:.2f}, Spearman ρ={ksr:.2f}",
        transform=ax.transAxes,
        fontsize=4.5,
        va="top",
        ha="left",
    )
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("published single-mutant fitness")
    ax.set_ylabel(f"CRISPR assay fitness ({CONDITION_LABEL})")
    ax.set_title(f"Assay fitness vs published SMF ({CONDITION_LABEL})", fontsize=6)
    ax.legend(fontsize=4.5, loc="lower right", frameon=False)
    for s in ax.spines.values():
        s.set_visible(True)
    fig.tight_layout()
    fig.savefig(osp.join(IMG_DIR, "run2_fitness_correlation_reference.png"), dpi=300)
    savefig_true_size_svg(
        fig, osp.join(IMG_DIR, "run2_fitness_correlation_reference.svg")
    )
    plt.close(fig)


def _fig_sd(df: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "svg.fonttype": "none",
            "axes.linewidth": 0.5,
        }
    )
    d = df.dropna(subset=["costanzo_sd"]).sort_values("costanzo_sd")
    x = np.arange(len(d))
    w = 0.38
    fig, ax = plt.subplots(
        figsize=(mm_to_in(PANEL_WIDTHS_MM["half_plus"]), mm_to_in(70))
    )
    ax.bar(
        x - w / 2,
        d["our_sd"],
        w,
        color=C_ORANGE,
        edgecolor="black",
        linewidth=0.4,
        label=f"CRISPR assay replicate SD ({CONDITION_LABEL})",
    )
    ax.bar(
        x + w / 2,
        d["costanzo_sd"],
        w,
        color=C_RED,
        edgecolor="black",
        linewidth=0.4,
        label="Costanzo 2016 SMF SD (bootstrap SE)",
    )
    ax.axhline(d["our_sd"].mean(), color=C_ORANGE, lw=0.5, ls="--")
    ax.axhline(d["costanzo_sd"].mean(), color=C_RED, lw=0.5, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(d["label"], rotation=90, fontsize=4)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.set_ylim(0, float(max(d["our_sd"].max(), d["costanzo_sd"].max())) * 1.10)
    ax.set_ylabel("fitness standard deviation")
    # legend ABOVE the axes so it never occludes the bars
    ax.legend(
        fontsize=4.5,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=1,
        frameon=False,
        handlelength=1.2,
    )
    for s in ax.spines.values():
        s.set_visible(True)
    fig.tight_layout()
    fig.savefig(osp.join(IMG_DIR, "run2_sd_vs_reference.png"), dpi=300)
    savefig_true_size_svg(fig, osp.join(IMG_DIR, "run2_sd_vs_reference.svg"))
    plt.close(fig)


def main() -> None:
    fit = pd.read_csv(osp.join(RESULTS_DIR, "run2_cellpose_error_by_strain.csv"))
    ref = pd.read_csv(osp.join(RESULTS_DIR, "reference_smf_12panel.csv"))
    sel = _select(fit)
    df = sel.merge(ref, on="strain", how="inner")
    df["label"] = df["orf"]  # systematic ORF -> every gene is labelled
    df.to_csv(osp.join(RESULTS_DIR, "run2_reference_comparison.csv"), index=False)

    pr, pp, sr, sp, n = _corr(
        df["costanzo_smf"].to_numpy(), df["our_fitness"].to_numpy()
    )
    kpr, kpp, ksr, ksp, kn = _corr(
        df["kuzmin_smf"].to_numpy(), df["our_fitness"].to_numpy()
    )
    sd = df.dropna(subset=["costanzo_sd"])
    stats = pd.DataFrame(
        [
            dict(
                reference="Costanzo2016",
                n=n,
                pearson_r=pr,
                pearson_p=pp,
                spearman_rho=sr,
                spearman_p=sp,
                our_mean_sd=float(sd["our_sd"].mean()),
                our_mean_se=float(sd["our_se"].mean()),
                ref_mean_sd=float(sd["costanzo_sd"].mean()),
            ),
            dict(
                reference="Kuzmin2018",
                n=kn,
                pearson_r=kpr,
                pearson_p=kpp,
                spearman_rho=ksr,
                spearman_p=ksp,
                our_mean_sd=np.nan,
                ref_mean_sd=np.nan,
            ),
        ]
    )
    stats.to_csv(
        osp.join(RESULTS_DIR, "run2_reference_comparison_stats.csv"), index=False
    )

    _fig_correlation(df)
    _fig_sd(df)

    print(f"strains compared: {len(df)}  (Costanzo n={n}, Kuzmin n={kn})")
    print(
        f"Costanzo  Pearson r={pr:.3f} (p={pp:.3f})  Spearman rho={sr:.3f} (p={sp:.3f})"
    )
    print(f"Kuzmin    Pearson r={kpr:.3f}  Spearman rho={ksr:.3f}  (n={kn})")
    print(
        f"[{CONDITION_LABEL}] mean replicate SD ours={sd['our_sd'].mean():.3f}  "
        f"SE ours={sd['our_se'].mean():.3f}  Costanzo SMF SD={sd['costanzo_sd'].mean():.3f}"
    )
    print(f"wrote -> {RESULTS_DIR}/run2_reference_comparison.csv (+ _stats.csv)")
    print(f"wrote -> {IMG_DIR}/run2_fitness_correlation_reference.{{png,svg}}")
    print(f"wrote -> {IMG_DIR}/run2_sd_vs_reference.{{png,svg}}")


if __name__ == "__main__":
    main()
