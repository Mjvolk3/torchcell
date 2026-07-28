# experiments/019-echo-crispr-array/scripts/assay_precision_benchmark.py
# [[experiments.019-echo-crispr-array.scripts.assay_precision_benchmark]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-echo-crispr-array/scripts/assay_precision_benchmark
"""How does our assay's fitness error compare to the published SGA screens?

"Is our assay good enough?" cannot be answered by comparing error bars at face value,
because the published uncertainties are NOT the same estimand. Each is grouped here by the
replication level it actually spans:

  HONEST  (spans independent experiment batches -- comparable to ours)
    ours, bootstrap over 3 re-randomized plates
    Costanzo 2016 SMF, bootstrap SE over 17 screens
  OPTIMISTIC (within a single batch -- omits the batch/plate term)
    ours, within-plate colony SE (shown to make the gap visible)
    Costanzo 2016 DMF, sample SD / sqrt(4 colonies), one screen
    Kuzmin 2018 TMF, sample SD / sqrt(4 colonies), one screen
  PARTIAL (clustered colonies bootstrapped i.i.d.)
    Kuzmin 2018 Data File S4 query standard, 12-24 colonies spanning 6 screens but
    resampled as if independent, so the between-screen term is under-represented

Comparing our HONEST number against someone else's OPTIMISTIC number is the mistake that
makes a good assay look bad.

Run from repo root:
    ~/miniconda3/envs/torchcell/bin/python \
        experiments/019-echo-crispr-array/scripts/assay_precision_benchmark.py
"""

from __future__ import annotations

import math
import os
import os.path as osp
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator

from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    savefig_true_size_svg,
)

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from ladder_feasibility import deletions_only, read_fitness_lmdb  # noqa: E402

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
EXP_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
RESULTS = osp.join(EXP_DIR, "results")
IMG_DIR = osp.join(ASSET_IMAGES_DIR, "019-echo-crispr-array")

KUZMIN_S4 = osp.join(
    DATA_ROOT,
    "torchcell-library/kuzminSystematicAnalysisComplex2018/si/si_data",
    "Data File S4_Fitness standard for single and double mutant query strains.xlsx",
)

HONEST, PARTIAL, OPTIMISTIC = "honest", "partial", "optimistic"


def collect() -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Gather every comparable uncertainty distribution with its error-model label."""
    boot = pd.read_csv(osp.join(RESULTS, "run3_strain_bootstrap.csv"))
    smf = deletions_only(read_fitness_lmdb("smf_costanzo2016")).dropna(
        subset=["fitness_se"]
    )
    smf = smf.drop_duplicates(subset=["gene0", "ptype0", "fitness"])
    dmf = deletions_only(read_fitness_lmdb("dmf_costanzo2016_5e5")).dropna(
        subset=["fitness_se"]
    )
    tmf = deletions_only(read_fitness_lmdb("tmf_kuzmin2018")).dropna(
        subset=["fitness_se"]
    )
    s4 = pd.read_excel(KUZMIN_S4).dropna(subset=["St.dev."])

    # A bootstrap of the mean over n observations converges to sqrt((n-1)/n) * s/sqrt(n),
    # i.e. it is biased LOW by 18.4% at n = 3. Undo that before comparing to a 17-screen
    # bootstrap, otherwise we flatter ourselves.
    n_plates_run3 = int(boot["n_plates"].iloc[0])
    bias = math.sqrt((n_plates_run3 - 1) / n_plates_run3)

    series: dict[str, np.ndarray] = {
        "ours: bootstrap over 3 plates": boot["boot_se"].to_numpy(float),
        f"ours: bias-corrected (/{bias:.3f})": boot["boot_se"].to_numpy(float) / bias,
        "Costanzo SMF: bootstrap over 17 screens": smf["fitness_se"].to_numpy(float),
        "Kuzmin S4 query standard: 12-24 colonies / 6 screens": s4["St.dev."].to_numpy(float),
        "ours: within-plate colonies only": boot["mean_within_plate_se"].to_numpy(float),
        "Costanzo DMF: 4 colonies, one screen": dmf["fitness_se"].to_numpy(float),
        "Kuzmin TMF: 4 colonies, one screen": tmf["fitness_se"].to_numpy(float),
    }
    model = {
        "ours: bootstrap over 3 plates": HONEST,
        f"ours: bias-corrected (/{bias:.3f})": HONEST,
        "Costanzo SMF: bootstrap over 17 screens": HONEST,
        "Kuzmin S4 query standard: 12-24 colonies / 6 screens": PARTIAL,
        "ours: within-plate colonies only": OPTIMISTIC,
        "Costanzo DMF: 4 colonies, one screen": OPTIMISTIC,
        "Kuzmin TMF: 4 colonies, one screen": OPTIMISTIC,
    }
    table = pd.DataFrame(
        [
            {
                "source": k,
                "error_model": model[k],
                "n": int(len(v)),
                "median_se": float(np.median(v)),
                "q25": float(np.quantile(v, 0.25)),
                "q75": float(np.quantile(v, 0.75)),
            }
            for k, v in series.items()
        ]
    )
    return series, table


def main() -> None:
    """Print the like-for-like comparison and project where more plates put us."""
    os.makedirs(IMG_DIR, exist_ok=True)
    series, table = collect()
    table.to_csv(osp.join(RESULTS, "run3_precision_benchmark.csv"), index=False)
    print(table.to_string(index=False))

    ours = table.loc[table.source == "ours: bootstrap over 3 plates", "median_se"].iloc[0]
    ours_bc = table.loc[table.source.str.startswith("ours: bias-corrected"), "median_se"].iloc[0]
    cost = table.loc[
        table.source == "Costanzo SMF: bootstrap over 17 screens", "median_se"
    ].iloc[0]
    print("\nLIKE FOR LIKE (both span independent batches):")
    print(f"    ours, 3 plates, as reported   median SE = {ours:.4f}  ({ours / cost:.2f}x)")
    print(f"    ours, 3 plates, bias-corrected median SE = {ours_bc:.4f}  ({ours_bc / cost:.2f}x)")
    print(f"    Costanzo SMF, 17 screens      median SE = {cost:.4f}")

    fit = pd.read_csv(osp.join(RESULTS, "run3_fitness_by_condition.csv"))
    boot = pd.read_csv(osp.join(RESULTS, "run3_strain_bootstrap.csv"))
    var_p = float(boot["across_plate_sd"].mean()) ** 2
    var_c = float(fit["fitness_sd"].mean()) ** 2
    print("\nprojection at c = 13 colonies/strain/plate:")
    for n_plates in (3, 4, 6, 8, 12, 17):
        se = math.sqrt(var_p / n_plates + var_c / (n_plates * 13))
        print(f"    P={n_plates:>3}: SE = {se:.4f}   ({se / cost:.2f}x Costanzo SMF)")

    make_figure(series, table, var_p, var_c, cost, ours_bc,
                osp.join(IMG_DIR, "assay_precision_benchmark"))
    print("\nfigure:", osp.join(IMG_DIR, "assay_precision_benchmark.svg"))


def make_figure(
    series: dict[str, np.ndarray],
    table: pd.DataFrame,
    var_p: float,
    var_c: float,
    costanzo_median: float,
    measured_p3: float,
    out_stem: str,
) -> None:
    """Two panels: SE distributions grouped by error model, and our plate projection."""
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "axes.linewidth": 0.5,
            "svg.fonttype": "none",
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "axes.labelsize": 6,
            "axes.titlesize": 6,
            "legend.fontsize": 5,
        }
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(58.0)),
        gridspec_kw={"width_ratios": [1.55, 1.0]},
    )
    colour = {HONEST: PLOT_PALETTE[0], PARTIAL: PLOT_PALETTE[3], OPTIMISTIC: PLOT_PALETTE[5]}

    # (a) SE distributions, ordered so the two comparable ones sit together
    ax = axes[0]
    names = list(series)
    pos = np.arange(len(names))[::-1]
    bp = ax.boxplot(
        [series[n] for n in names], positions=pos, vert=False, widths=0.62,
        showfliers=False, patch_artist=True,
        medianprops={"color": "black", "lw": 0.8},
        whiskerprops={"lw": 0.5}, capprops={"lw": 0.5}, boxprops={"lw": 0.5},
    )
    for patch, name in zip(bp["boxes"], names):
        model = table.loc[table.source == name, "error_model"].iloc[0]
        patch.set_facecolor(colour[model])
    ax.set_yticks(pos)
    ax.set_yticklabels(names, fontsize=5)
    ax.set_xlabel("standard error of a strain's fitness")
    ax.set_xlim(0, 0.22)
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.xaxis.set_minor_locator(MultipleLocator(0.025))
    ax.set_title("a  like-for-like only within a colour", fontsize=6, loc="left")
    handles = [
        Rectangle((0, 0), 1, 1, fc=colour[m], ec="black", lw=0.4)
        for m in (HONEST, PARTIAL, OPTIMISTIC)
    ]
    ax.legend(handles, ["spans batches (honest)", "clustered i.i.d. (partial)",
                        "one batch (optimistic)"], frameon=False, loc="lower right")

    # (b) where more plates put us, against the Costanzo SMF benchmark
    ax = axes[1]
    plates = np.arange(2, 19)
    se = [math.sqrt(var_p / int(p) + var_c / (int(p) * 13)) for p in plates]
    ax.plot(plates, se, "o-", ms=2.5, lw=0.9, color=PLOT_PALETTE[0], label="ours, c = 13")
    ax.axhline(costanzo_median, color=PLOT_PALETTE[1], lw=0.8, ls="--",
               label=f"Costanzo SMF median {costanzo_median:.3f}")
    ax.scatter([3], [measured_p3], s=14, marker="D", color=PLOT_PALETTE[2], zorder=5,
               label=f"measured at P = 3 ({measured_p3:.3f})")
    cross = next((int(p) for p, s in zip(plates, se) if s <= costanzo_median), None)
    if cross is not None:
        ax.annotate(f"parity at P = {cross}", (cross, costanzo_median),
                    textcoords="offset points", xytext=(4, 6), fontsize=5,
                    arrowprops={"arrowstyle": "->", "lw": 0.5})
    ax.set_xlabel("plates")
    ax.set_ylabel("SE of a strain mean")
    ax.set_title("b  plates needed for parity", fontsize=6, loc="left")
    ax.legend(frameon=False, loc="upper right")
    ax.set_ylim(0, 0.13)
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.04))
    ax.yaxis.set_minor_locator(MultipleLocator(0.02))

    for ax in axes:
        for s in ax.spines.values():
            s.set_visible(True)
        ax.tick_params(width=0.5, length=2)
        ax.tick_params(which="minor", length=0)

    fig.tight_layout(pad=0.4)
    fig.savefig(out_stem + ".png", dpi=300)
    savefig_true_size_svg(fig, out_stem + ".svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
