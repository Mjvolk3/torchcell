# experiments/019-echo-crispr-array/scripts/bootstrap_across_plates.py
# [[experiments.019-echo-crispr-array.scripts.bootstrap_across_plates]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-echo-crispr-array/scripts/bootstrap_across_plates
"""Demonstrate bootstrap-across-plates SE vs the naive pooled SD/sqrt(n) SE.

Question (from the 019 fitness-assay discussion): if we treat the plates we have
collected as replicate "screens" (Costanzo-style) and bootstrap each strain's mean
fitness across plates, how does that standard error compare to just pooling every
colony across the plates and taking SD/sqrt(n)?

The two are NOT the same, and the gap is the whole point:

  * BOOTSTRAP ACROSS PLATES resamples the plate-level fitness values (one per plate)
    with replacement; the SD of the bootstrap means is the SE, and it CAPTURES
    between-plate variation (volume/day/position) -- the real reproducibility.
  * POOLED SD/sqrt(n) pools all colonies across plates and divides by the total colony
    count. It treats spatially-correlated same-plate colonies as independent, IGNORES
    the between-plate component, and so reports a much smaller (over-confident) SE.

We report both the bootstrapped MEAN and MEDIAN (Costanzo/Kuzmin bootstrap means;
Baryshnikova bootstrapped medians), so the mean-vs-median choice is visible too.

Caveat the caller should weigh: pooling 2.5 nL and 5 nL plates mixes volumes. Fitness
is WT-normalised per plate so normalised values are broadly comparable, but any
systematic volume effect enters the between-plate variance -- which makes the
across-plate SE conservative (larger), not wrong. Restrict --conditions to one volume
to remove that.

Reads results/run2_cellpose_error_by_strain.csv (per strain x plate: fitness,
fitness_sd, fitness_se, n_used).

Run from repo root:
    ~/miniconda3/envs/torchcell/bin/python \
        experiments/019-echo-crispr-array/scripts/bootstrap_across_plates.py \
        [--conditions P1_t50,P2_t50,P1_t44,P2_t44,P1_t72,P2_t72] [--n_boot 4000]
"""

from __future__ import annotations

import argparse
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.ticker import MultipleLocator

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
C_ORANGE, C_RED, C_GRAY = PLOT_PALETTE[0], PLOT_PALETTE[1], PLOT_PALETTE[5]

# NOTE: Math.random / np.random default seeding is fine here (script is run by hand,
# not inside a resumable workflow). Fixed seed for reproducibility.
SEED = 1234


def _bootstrap(
    vals: np.ndarray, n_boot: int, rng: np.random.Generator
) -> dict[str, float]:
    """Bootstrap the mean and median of the per-plate fitness values (resample plates
    with replacement). Returns point estimates + the SD of the bootstrap distribution
    (= the bootstrap SE) for each.
    """
    k = len(vals)
    idx = rng.integers(0, k, size=(n_boot, k))
    samp = vals[idx]
    boot_mean = samp.mean(axis=1)
    boot_med = np.median(samp, axis=1)
    return {
        "boot_mean": float(boot_mean.mean()),
        "boot_mean_se": float(boot_mean.std(ddof=1)),
        "boot_median": float(boot_med.mean()),
        "boot_median_se": float(boot_med.std(ddof=1)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--conditions",
        default="P1_t44,P2_t44,P1_t50,P2_t50,P1_t72,P2_t72",
        help="comma list of plate groups to treat as replicate screens",
    )
    ap.add_argument("--n_boot", type=int, default=4000)
    args = ap.parse_args()
    conds = args.conditions.split(",")
    rng = np.random.default_rng(SEED)

    fit = pd.read_csv(osp.join(RESULTS_DIR, "run2_cellpose_error_by_strain.csv"))
    fit = fit[fit["group"].isin(conds) & (fit["strain"] != WT_NAME)]

    rows = []
    for strain, d in fit.groupby("strain"):
        vals = d["fitness"].to_numpy(dtype=float)
        if len(vals) < 2:
            continue
        n_plates = len(vals)
        # (1) bootstrap across plates (plate = the resampling unit)
        bs = _bootstrap(vals, args.n_boot, rng)
        # between-plate SD (the component the pooled SE ignores)
        between_sd = float(vals.std(ddof=1))
        # (2) naive pooled SD/sqrt(n): pool all colonies across plates
        ni = d["n_used"].to_numpy(dtype=float)
        sdi = d["fitness_sd"].to_numpy(dtype=float)
        dof = np.clip(ni - 1, 0, None)
        pooled_sd = float(np.sqrt((dof * sdi**2).sum() / max(dof.sum(), 1)))
        total_n = float(ni.sum())
        pooled_se = pooled_sd / np.sqrt(max(total_n, 1))
        rows.append(
            dict(
                strain=strain,
                n_plates=n_plates,
                total_colonies=int(total_n),
                fitness_mean=float(vals.mean()),
                boot_mean=bs["boot_mean"],
                boot_mean_se=bs["boot_mean_se"],
                boot_median=bs["boot_median"],
                boot_median_se=bs["boot_median_se"],
                between_plate_sd=between_sd,
                pooled_colony_sd=pooled_sd,
                pooled_se=pooled_se,
                se_ratio=bs["boot_mean_se"] / pooled_se if pooled_se else np.nan,
            )
        )
    res = pd.DataFrame(rows).sort_values("strain")
    res.to_csv(osp.join(RESULTS_DIR, "run2_bootstrap_vs_pooled_se.csv"), index=False)

    # figure: per strain, bootstrap-across-plates SE vs pooled SD/sqrt(n) SE
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "svg.fonttype": "none",
            "axes.linewidth": 0.5,
        }
    )
    x = np.arange(len(res))
    w = 0.38
    fig, ax = plt.subplots(
        figsize=(mm_to_in(PANEL_WIDTHS_MM["half_plus"]), mm_to_in(72))
    )
    ax.bar(
        x - w / 2,
        res["boot_mean_se"],
        w,
        color=C_ORANGE,
        edgecolor="black",
        linewidth=0.4,
        label=f"bootstrap-across-plates SE (n={len(conds)} plates)",
    )
    ax.bar(
        x + w / 2,
        res["pooled_se"],
        w,
        color=C_RED,
        edgecolor="black",
        linewidth=0.4,
        label="pooled SE (colony SD $/\\sqrt{n}$)",
    )
    ax.axhline(res["boot_mean_se"].mean(), color=C_ORANGE, lw=0.5, ls="--")
    ax.axhline(res["pooled_se"].mean(), color=C_RED, lw=0.5, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(res["strain"], rotation=90, fontsize=4)
    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.set_ylim(0, float(max(res["boot_mean_se"].max(), res["pooled_se"].max())) * 1.28)
    ax.set_ylabel("standard error of the mean fitness")
    # legend inside, top-right (headroom above the right-hand bars)
    ax.legend(fontsize=4.5, loc="upper right", frameon=False)
    for s in ax.spines.values():
        s.set_visible(True)
    fig.tight_layout()
    fig.savefig(osp.join(IMG_DIR, "run2_bootstrap_vs_pooled_se.png"), dpi=300)
    savefig_true_size_svg(fig, osp.join(IMG_DIR, "run2_bootstrap_vs_pooled_se.svg"))
    plt.close(fig)

    print(f"plates (screens) used: {conds}")
    print(res.round(4).to_string(index=False))
    print(
        f"\nMEAN over strains: bootstrap-across-plates SE={res['boot_mean_se'].mean():.4f}  "
        f"pooled SD/sqrt(n) SE={res['pooled_se'].mean():.4f}  "
        f"ratio={res['se_ratio'].mean():.1f}x"
    )
    print(
        f"bootstrap MEAN vs MEDIAN SE (mean over strains): "
        f"{res['boot_mean_se'].mean():.4f} vs {res['boot_median_se'].mean():.4f}"
    )
    print(f"wrote -> {RESULTS_DIR}/run2_bootstrap_vs_pooled_se.csv")
    print(f"wrote -> {IMG_DIR}/run2_bootstrap_vs_pooled_se.{{png,svg}}")


if __name__ == "__main__":
    main()
