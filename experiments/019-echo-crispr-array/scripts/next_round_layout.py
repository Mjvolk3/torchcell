# experiments/019-echo-crispr-array/scripts/next_round_layout.py
# [[experiments.019-echo-crispr-array.scripts.next_round_layout]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-echo-crispr-array/scripts/next_round_layout
"""Dense-plate layout for the next round, scored on EPISTASIS rather than fitness gain.

Supersedes the sampling arithmetic in `next_round_design.py` for three reasons settled on
2026-07-27:

  1. The target is epistasis (eps = f_ab - f_a*f_b), not a fitness ladder -- the upward
     ladder was ruled out in `ladder_feasibility.py`.
  2. The plate is packed: 26 strains (12 singles + 14 doubles) + wild type, no cushion.
  3. Every destination replicate of a strain is dispensed from ONE source well, so it
     descends from ONE liquid starter culture. That adds a variance term which NO amount
     of within-experiment replication removes, and which run 3 could not measure because
     it is perfectly confounded with the strain effect.

Variance model for a strain mean over P plates x c colonies, with K independent starter
cultures per strain spread across those plates:

    Var(mean) = sigma_culture^2 / K  +  sigma_plate^2 / P  +  sigma_colony^2 / (P*c)

K = 1 (one starter for everything) leaves sigma_culture as an SE FLOOR that more plates
cannot cross. Kuzmin avoids this by re-inoculating from frozen stock for each of the three
screen repeats, so their replication re-draws the culture -- see the note.

Run from repo root:
    ~/miniconda3/envs/torchcell/bin/python \
        experiments/019-echo-crispr-array/scripts/next_round_layout.py
"""

from __future__ import annotations

import json
import math
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
RESULTS = osp.join(EXP_DIR, "results")
IMG_DIR = osp.join(ASSET_IMAGES_DIR, "019-echo-crispr-array")

N_SINGLES, N_DOUBLES = 12, 14
N_STRAINS = N_SINGLES + N_DOUBLES
WELLS = 384
# Kuzmin 2018 called a digenic interaction at |eps| > 0.08 (SI, "interaction magnitude
# cut-off for digenic interactions (p < 0.05, |eps| > 0.08)")
KUZMIN_EPS_THRESHOLD = 0.08
# Kuzmin 2018 Data File S4 released St.dev. for its 546 single/double query strains
KUZMIN_QUERY_SD_MEDIAN = 0.0130


def strain_se(
    var_plate: float,
    var_colony: float,
    n_plates: int,
    c: int,
    sd_culture: float = 0.0,
    n_cultures: int = 1,
) -> float:
    """SE of one strain's mean fitness under the three-level variance model."""
    return math.sqrt(
        sd_culture**2 / n_cultures + var_plate / n_plates + var_colony / (n_plates * c)
    )


def epsilon_se(se_double: float, se_single: float, f_bar: float) -> float:
    """Delta-method SE of eps = f_ab - f_a*f_b.

    d(eps)/d(f_a) = -f_b and d(eps)/d(f_b) = -f_a, so each single contributes its SE scaled
    by the OTHER single's fitness. Assumes the three estimates are independent: residual
    plate effects here are strain x plate interaction (the common plate factor is already
    removed by normalising to on-plate wild type), so they do not cancel in the difference.
    """
    return math.sqrt(se_double**2 + 2 * (f_bar**2) * se_single**2)


def attenuation(sd_signal: float, se_meas: float) -> float:
    """Factor by which measurement error shrinks an observed correlation.

    observed_r = true_r * sqrt(reliability), reliability = var_signal / (var_signal + var_meas).
    """
    return math.sqrt(sd_signal**2 / (sd_signal**2 + se_meas**2))


def main() -> None:
    """Print the layout, the epistasis precision it buys, and the culture-confounding risk."""
    os.makedirs(IMG_DIR, exist_ok=True)
    boot = pd.read_csv(osp.join(RESULTS, "run3_strain_bootstrap.csv"))
    fit = pd.read_csv(osp.join(RESULTS, "run3_fitness_by_condition.csv"))
    with open(osp.join(RESULTS, "run3_ladder_feasibility_summary.json")) as fh:
        feas = json.load(fh)

    sd_plate = float(boot["across_plate_sd"].mean())
    sd_colony = float(fit["fitness_sd"].mean())
    var_p, var_c = sd_plate**2, sd_colony**2
    f_bar = float(boot["boot_fitness"].mean())
    sd_eps_random = float(feas["epsilon_sd_random_pairs"])

    print("[1] measured noise (run 3) and the epistasis signal to resolve")
    print(f"    sigma_plate  = {sd_plate:.3f}   sigma_colony = {sd_colony:.3f}")
    print(f"    mean single-mutant fitness f_bar = {f_bar:.3f}")
    print(f"    SD of eps across RANDOM Costanzo deletion pairs = {sd_eps_random:.4f}")
    print(f"    Kuzmin digenic call threshold |eps| > {KUZMIN_EPS_THRESHOLD}")
    print(f"    Kuzmin query-strain released St.dev. (median)   = {KUZMIN_QUERY_SD_MEDIAN}")

    print("\n[2] dense layout: 384 wells, 26 strains + WT, no cushion")
    rows = []
    for c in (12, 13, 14):
        wt = WELLS - N_STRAINS * c
        rows.append({"c_per_strain": c, "strain_wells": N_STRAINS * c, "wt_wells": wt})
        print(f"    c={c:>3} -> {N_STRAINS * c:>3} strain wells + {wt:>3} WT wells")
    print("    WT deserves the remainder: every strain is normalised to it, so WT error")
    print("    propagates into EVERY strain, not just one.")
    pd.DataFrame(rows).to_csv(osp.join(RESULTS, "run4_layout_options.csv"), index=False)

    c_dense, c_run3 = 13, 27
    print(f"\n[3] cost of packing the plate (c={c_run3} -> c={c_dense}), SE of a strain mean")
    for n_plates in (3, 4, 6):
        a = strain_se(var_p, var_c, n_plates, c_run3)
        b = strain_se(var_p, var_c, n_plates, c_dense)
        print(f"    P={n_plates}: {a:.4f} -> {b:.4f}   ({100 * (b - a) / a:+.1f}%)")
    print("    -> packing twice as many strains costs ~3%. Density is nearly free.")

    print(f"\n[4] epistasis precision at c={c_dense} (what the model is actually scored on)")
    print(f"    {'P':>3} {'SE(strain)':>11} {'SE(eps)':>9} {'vs |eps|>0.08':>14}")
    for n_plates in (3, 4, 6, 8, 12):
        se_s = strain_se(var_p, var_c, n_plates, c_dense)
        se_e = epsilon_se(se_s, se_s, f_bar)
        print(f"    {n_plates:>3} {se_s:>11.4f} {se_e:>9.4f} {se_e / KUZMIN_EPS_THRESHOLD:>13.1f}x")
    print("    -> at P=3-4 the SE on a single eps EXCEEDS Kuzmin's entire calling threshold.")
    print("       Per-strain significance on eps is out of reach; the CORRELATION is not.")

    print("\n[5] correlation attenuation: observed_r = true_r * factor")
    spans = [
        ("random pairs", sd_eps_random),
        ("2x random spread", 2 * sd_eps_random),
        ("3x random spread", 3 * sd_eps_random),
        ("4x random spread", 4 * sd_eps_random),
    ]
    print(f"    {'design spread of true eps':>26} " + "".join(f"{('P=' + str(p)):>8}" for p in (3, 4, 6, 8)))
    for label, sd_sig in spans:
        row = f"    {label + f' (SD={sd_sig:.3f})':>26} "
        for n_plates in (3, 4, 6, 8):
            se_s = strain_se(var_p, var_c, n_plates, c_dense)
            row += f"{attenuation(sd_sig, epsilon_se(se_s, se_s, f_bar)):>8.2f}"
        print(row)
    print("    -> CHOOSING the 14 pairs to SPAN a wide predicted eps beats adding plates.")
    print("       Picking representative pairs throws the experiment away.")

    print("\n[6] the starter-culture floor (K = independent cultures per strain)")
    print(f"    {'sigma_culture':>14} " + "".join(f"{('K=' + str(k)):>9}" for k in (1, 2, 3, 4)))
    for sd_cult in (0.00, 0.02, 0.05, 0.10):
        row = f"    {sd_cult:>14.2f} "
        for k in (1, 2, 3, 4):
            row += f"{strain_se(var_p, var_c, 4, c_dense, sd_cult, k):>9.4f}"
        print(row)
    print("    (SE of a strain mean at P=4; K>P is impossible -- one culture per plate max)")
    print("    sigma_culture is UNMEASURED here: run 3 used one culture per strain, so it is")
    print("    perfectly confounded with the strain effect. K=3 both shrinks it AND makes it")
    print("    measurable for the first time.")

    print("\n[7] panel growth: CRISPR strains are kept, so the panel accumulates")
    print(f"    {'round':>22} {'strains':>8} {'c':>4} {'WT':>4} {'SE @ P=4':>10} {'vs 26':>7}")
    base = strain_se(var_p, var_c, 4, c_dense)
    for label, n_str in (
        ("now: 12 S + 14 D", 26),
        ("+14 triples", 40),
        ("+28 triples", 54),
        ("+14 more of each", 68),
    ):
        c = (WELLS - 20) // n_str  # keep >= 20 WT wells
        wt = WELLS - n_str * c
        se = strain_se(var_p, var_c, 4, c)
        print(f"    {label:>22} {n_str:>8} {c:>4} {wt:>4} {se:>10.4f} {se / base:>6.2f}x")
    print("    -> a panel 2.6x larger costs <10% SE. Accumulating strains is nearly free,")
    print("       because colonies only ever divide the SMALLER variance term.")

    make_figure(var_p, var_c, f_bar, sd_eps_random, c_dense, osp.join(IMG_DIR, "next_round_layout"))
    print("\nfigure:", osp.join(IMG_DIR, "next_round_layout.svg"))


def make_figure(
    var_p: float,
    var_c: float,
    f_bar: float,
    sd_eps_random: float,
    c_dense: int,
    out_stem: str,
) -> None:
    """Three panels: plates vs SE(eps), attenuation vs design spread, culture floor."""
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
        1, 3, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(52.0))
    )
    plates = np.arange(2, 17)

    # (a) SE on epsilon vs plates, against Kuzmin's calling threshold
    ax = axes[0]
    se_eps = [
        epsilon_se(strain_se(var_p, var_c, int(p), c_dense),
                   strain_se(var_p, var_c, int(p), c_dense), f_bar)
        for p in plates
    ]
    ax.plot(plates, se_eps, "o-", ms=2.5, lw=0.9, color=PLOT_PALETTE[0], label="SE(eps)")
    ax.axhline(KUZMIN_EPS_THRESHOLD, color=PLOT_PALETTE[1], lw=0.8, ls="--",
               label="Kuzmin |eps| > 0.08")
    ax.axhline(KUZMIN_QUERY_SD_MEDIAN, color="black", lw=0.8, ls=":",
               label="Kuzmin query SD 0.013")
    ax.set_xlabel("plates")
    ax.set_ylabel("SE of a single eps")
    ax.set_title("a  per-eps precision", fontsize=6, loc="left")
    ax.legend(frameon=False, loc="upper right")
    ax.set_ylim(0, 0.20)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.025))

    # (b) attenuation vs how widely the chosen pairs span true epsilon
    ax = axes[1]
    spread = np.linspace(0.02, 0.25, 120)
    for n_plates, col in ((3, PLOT_PALETTE[0]), (4, PLOT_PALETTE[1]), (8, PLOT_PALETTE[2])):
        se_e = epsilon_se(strain_se(var_p, var_c, n_plates, c_dense),
                          strain_se(var_p, var_c, n_plates, c_dense), f_bar)
        ax.plot(spread, [attenuation(s, se_e) for s in spread], lw=1.0, color=col,
                label=f"P = {n_plates}")
    ax.axvline(sd_eps_random, color="black", lw=0.8, ls="--")
    ax.annotate("random\npairs", (sd_eps_random, 0.86), xytext=(4, 0),
                textcoords="offset points", fontsize=5)
    ax.set_xlabel("SD of true eps among the 14 chosen pairs")
    ax.set_ylabel("observed r / true r")
    ax.set_title("b  pick pairs that SPAN eps", fontsize=6, loc="left")
    ax.legend(frameon=False, loc="lower right")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    # (c) the starter-culture floor
    ax = axes[2]
    for sd_cult, col in ((0.0, PLOT_PALETTE[5]), (0.02, PLOT_PALETTE[3]),
                         (0.05, PLOT_PALETTE[1]), (0.10, PLOT_PALETTE[2])):
        ax.plot(plates, [strain_se(var_p, var_c, int(p), c_dense, sd_cult, 1) for p in plates],
                lw=1.0, color=col, label=f"K = 1, sigma_cult = {sd_cult:.2f}")
    ax.plot(plates, [strain_se(var_p, var_c, int(p), c_dense, 0.05, min(3, int(p))) for p in plates],
            lw=1.0, ls="--", color=PLOT_PALETTE[1], label="K = 3, sigma_cult = 0.05")
    ax.set_xlabel("plates")
    ax.set_ylabel("SE of a strain mean")
    ax.set_title("c  one starter culture = an SE floor", fontsize=6, loc="left")
    ax.legend(frameon=False, loc="upper right", handlelength=1.4, borderaxespad=0.2)
    ax.set_ylim(0, 0.19)
    ax.yaxis.set_major_locator(MultipleLocator(0.04))
    ax.yaxis.set_minor_locator(MultipleLocator(0.02))

    # panels a and c are indexed by plates; panel b is indexed by eps spread
    for ax in (axes[0], axes[2]):
        ax.xaxis.set_major_locator(MultipleLocator(4))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
    axes[1].xaxis.set_major_locator(MultipleLocator(0.05))
    axes[1].xaxis.set_minor_locator(MultipleLocator(0.025))

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
