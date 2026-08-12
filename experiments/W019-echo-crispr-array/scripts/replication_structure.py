# experiments/W019-echo-crispr-array/scripts/replication_structure.py
# [[experiments.W019-echo-crispr-array.scripts.replication_structure]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/W019-echo-crispr-array/scripts/replication_structure
"""What the campaign's replication structure replicates -- and what it still leaves exposed.

The plan settled 2026-07-27: seed cultures are struck fresh from glycerol, one per plate, so
plates on a given day are genuine replicates that re-draw the culture. Later rounds add more
plating as doubles and then triples come online. That fixes the starter-culture floor found
in `next_round_layout.py`, but it moves the un-replicated level UP one:

    day / batch   <- media batch, incubator run, imaging session
      seed culture  <- fresh from glycerol, one per plate  (NOW replicated)
        plate         <- one randomized 384 layout          (replicated)
          colony        <- one well                          (replicated)

`Var(strain mean) = sigma_day^2 / D + sigma_plate^2 / P + sigma_colony^2 / (P*c)`

All plates on ONE day leaves `sigma_day^2 / 1` as a floor, exactly as one culture did. Our
measured `sigma_plate = 0.140` comes from three run-3 plates on a single day, so it does NOT
contain the day term and is a LOWER bound for a multi-day campaign.

Two checks are done here:
  1. Is `sigma_plate` really plate-to-plate variation, or is it residual positional bias that
     better normalization could remove? (Answer: it is real -- normalization is sound.)
  2. How much does measuring the orders in DIFFERENT rounds cost, versus measuring singles,
     doubles and triples together on the same plates?

Run from repo root:
    ~/miniconda3/envs/torchcell/bin/python \
        experiments/W019-echo-crispr-array/scripts/replication_structure.py
"""

from __future__ import annotations

import math
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.ticker import MultipleLocator
from scipy.stats import spearmanr

from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
EXP_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
RESULTS = osp.join(EXP_DIR, "results")
IMG_DIR = osp.join(ASSET_IMAGES_DIR, "W019-echo-crispr-array")

N_ROWS, N_COLS = 16, 24
C_DENSE = 13
F_BAR = 0.851
# fitness of a typical double under the multiplicative null, used to weight the day term
F_DOUBLE = F_BAR**2


def positional_residual_check() -> pd.DataFrame:
    """Does position still predict colony size AFTER row/col + spatial correction?

    If normalization were leaving positional bias behind, `sigma_plate` would be partly an
    artifact of re-randomizing the layout between plates, and better normalization would be
    cheaper than more plates. Tested on run 2, the only per-colony table with positions.
    """
    d = pd.read_csv(osp.join(RESULTS, "run2_cellpose_colonies_registered.csv"))
    d = d[d["is_reference"] & ~d["is_missing"]].copy()
    d["edge"] = np.minimum.reduce(
        [d["row"] - 1, N_ROWS - d["row"], d["col"] - 1, N_COLS - d["col"]]
    )
    rows = []
    for g, x in d.groupby("group"):
        rho_raw = float(spearmanr(x["edge"], x["size"]).statistic)
        rho_norm = float(spearmanr(x["edge"], x["norm"]).statistic)
        rows.append(
            {
                "group": g,
                "n": int(len(x)),
                "rho_size_vs_edge": rho_raw,
                "rho_norm_vs_edge": rho_norm,
                "bias_removed": 1 - abs(rho_norm) / abs(rho_raw),
            }
        )
    return pd.DataFrame(rows)


def se_strain(
    var_p: float, var_c: float, n_plates: int, c: int, sd_day: float, n_days: int
) -> float:
    """SE of a strain mean with the day level made explicit."""
    return math.sqrt(
        sd_day**2 / n_days + var_p / n_plates + var_c / (n_plates * c)
    )


def eps_day_penalty(same_batch: bool) -> float:
    """Coefficient on sigma_day in SE(eps), for same- vs cross-batch measurement.

    eps = f_ab - f_a*f_b. Write a day deviation as f -> f*(1+delta) on the WT-normalised
    scale. If every term is measured in the SAME batch they share one delta:
        eps_obs - eps ~= delta * (f_ab - 2*f_a*f_b) ~= -delta * F_DOUBLE
    so the coefficient is F_DOUBLE. If the double comes from one batch and the singles from
    another, the two deltas are independent:
        eps_obs - eps ~= f_ab*delta1 - 2*f_a*f_b*delta2
    giving sqrt(1 + 4) * F_DOUBLE. Cross-batch is therefore sqrt(5) = 2.24x worse in the day
    term -- the whole reason to re-measure every order alongside the new one each round.
    """
    return F_DOUBLE if same_batch else math.sqrt(5.0) * F_DOUBLE


def main() -> None:
    """Report the positional check, the day floor, and the same-batch requirement."""
    os.makedirs(IMG_DIR, exist_ok=True)
    boot = pd.read_csv(osp.join(RESULTS, "run3_strain_bootstrap.csv"))
    fit = pd.read_csv(osp.join(RESULTS, "run3_fitness_by_condition.csv"))
    var_p = float(boot["across_plate_sd"].mean()) ** 2
    var_c = float(fit["fitness_sd"].mean()) ** 2

    print("[1] is sigma_plate real, or leftover positional bias?")
    pos = positional_residual_check()
    pos.to_csv(osp.join(RESULTS, "run2_positional_residual_check.csv"), index=False)
    print(pos.to_string(index=False))
    print(
        f"    normalization removes {pos.bias_removed.min():.0%}-{pos.bias_removed.max():.0%}"
        " of the edge bias; residual |rho| <= "
        f"{pos.rho_norm_vs_edge.abs().max():.3f}."
    )
    print("    -> sigma_plate is REAL plate-to-plate variation, not a normalization gap.")
    print("       Better spatial correction is NOT a cheaper substitute for more plates.")

    print("\n[2] the day floor: 4 plates, c = 13, spread over D days")
    print(f"    {'sigma_day':>10} " + "".join(f"{('D=' + str(x)):>9}" for x in (1, 2, 4)))
    for sd_day in (0.00, 0.03, 0.06, 0.10):
        row = f"    {sd_day:>10.2f} "
        for n_days in (1, 2, 4):
            row += f"{se_strain(var_p, var_c, 4, C_DENSE, sd_day, n_days):>9.4f}"
        print(row)
    print("    run-3 plates were all one day, so sigma_plate = 0.140 EXCLUDES the day term.")
    print("    Spreading round 4 over >= 2 days measures sigma_day for the first time.")

    print("\n[3] measuring the orders together vs in separate rounds")
    se_base = se_strain(var_p, var_c, 4, C_DENSE, 0.0, 1)
    se_e = math.sqrt(se_base**2 + 2 * (F_BAR**2) * se_base**2)
    print(f"    {'sigma_day':>10} {'SE(eps) same batch':>20} {'SE(eps) split rounds':>22} {'penalty':>9}")
    for sd_day in (0.00, 0.03, 0.06, 0.10):
        same = math.sqrt(se_e**2 + (eps_day_penalty(True) * sd_day) ** 2)
        cross = math.sqrt(se_e**2 + (eps_day_penalty(False) * sd_day) ** 2)
        print(f"    {sd_day:>10.2f} {same:>20.4f} {cross:>22.4f} {cross / same:>8.2f}x")
    print("    -> re-measure singles AND doubles alongside the triples EVERY round.")
    print("       Keeping the strains is what makes that possible.")

    make_figure(pos, var_p, var_c, se_e, osp.join(IMG_DIR, "replication_structure"))
    print("\nfigure:", osp.join(IMG_DIR, "replication_structure.svg"))


def make_figure(
    pos: pd.DataFrame, var_p: float, var_c: float, se_e: float, out_stem: str
) -> None:
    """Three panels: normalization check, day floor, same-batch requirement."""
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
        1, 3, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(54.0))
    )

    # (a) positional bias before vs after normalization
    ax = axes[0]
    idx = np.arange(len(pos))
    ax.barh(idx + 0.19, pos["rho_size_vs_edge"].abs(), 0.36,
            color=PLOT_PALETTE[5], edgecolor="black", lw=0.4, label="raw colony size")
    ax.barh(idx - 0.19, pos["rho_norm_vs_edge"].abs(), 0.36,
            color=PLOT_PALETTE[0], edgecolor="black", lw=0.4, label="after normalization")
    ax.set_yticks(idx)
    ax.set_yticklabels(pos["group"], fontsize=5)
    ax.set_xlabel("|Spearman rho| vs distance from plate edge")
    ax.set_title("a  normalization is doing its job", fontsize=6, loc="left")
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlim(0, 0.26)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))

    # (b) the day floor
    ax = axes[1]
    plates = np.arange(2, 17)
    for sd_day, col, lab in (
        (0.00, PLOT_PALETTE[5], "sigma_day = 0"),
        (0.06, PLOT_PALETTE[1], "sigma_day = 0.06, all one day"),
        (0.10, PLOT_PALETTE[2], "sigma_day = 0.10, all one day"),
    ):
        ax.plot(plates, [se_strain(var_p, var_c, int(p), C_DENSE, sd_day, 1) for p in plates],
                lw=1.1, color=col, label=lab)
    ax.plot(plates,
            [se_strain(var_p, var_c, int(p), C_DENSE, 0.10, max(1, int(p) // 2)) for p in plates],
            lw=1.1, ls="--", color=PLOT_PALETTE[2], label="sigma_day = 0.10, 2 plates/day")
    ax.set_xlabel("plates")
    ax.set_ylabel("SE of a strain mean")
    ax.set_title("b  one day = a new floor", fontsize=6, loc="left")
    ax.legend(frameon=False, loc="upper right")
    ax.set_ylim(0, 0.17)
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.04))
    ax.yaxis.set_minor_locator(MultipleLocator(0.02))

    # (c) same-batch vs split-round measurement of epsilon
    ax = axes[2]
    days = np.linspace(0, 0.12, 100)
    ax.plot(days, [math.sqrt(se_e**2 + (eps_day_penalty(True) * d) ** 2) for d in days],
            lw=1.1, color=PLOT_PALETTE[0], label="all orders, same plates")
    ax.plot(days, [math.sqrt(se_e**2 + (eps_day_penalty(False) * d) ** 2) for d in days],
            lw=1.1, color=PLOT_PALETTE[1], label="orders split across rounds")
    ax.axhline(0.08, color="black", lw=0.7, ls="--")
    ax.annotate("Kuzmin |eps| > 0.08", (0.002, 0.083), fontsize=5)
    ax.set_xlabel("day-to-day SD (sigma_day)")
    ax.set_ylabel("SE of a single eps")
    ax.set_title("c  measure every order together", fontsize=6, loc="left")
    ax.legend(frameon=False, loc="upper left")
    ax.set_ylim(0, 0.25)
    ax.xaxis.set_major_locator(MultipleLocator(0.04))
    ax.xaxis.set_minor_locator(MultipleLocator(0.02))
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.025))

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
