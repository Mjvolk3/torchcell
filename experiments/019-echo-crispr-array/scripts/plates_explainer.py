# experiments/019-echo-crispr-array/scripts/plates_explainer.py
# [[experiments.019-echo-crispr-array.scripts.plates_explainer]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-echo-crispr-array/scripts/plates_explainer
"""What does each extra plate actually buy, and why does adding strains cost so little?

Four plain-language panels, all from run-3 measured noise:

  a. Error bar on one strain vs plates -- and where the returns die.
  b. What you can CALL: the smallest fitness gap between two strains you can declare real.
  c. What it does to the model score: observed vs true correlation on epistasis.
  d. Why adding strains is cheap: splitting a plate more ways costs almost nothing, because
     colonies were already averaging out the small noise term; the plate-to-plate term does
     not care how many strains share the plate.

Run from repo root:
    ~/miniconda3/envs/torchcell/bin/python \
        experiments/019-echo-crispr-array/scripts/plates_explainer.py
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

WELLS = 384
C_DENSE = 13
F_BAR = 0.851
# a design that deliberately spans predicted epsilon, ~3x the random-pair spread
EPS_DESIGN_SD = 0.140


def se_strain(var_p: float, var_c: float, n_plates: float, c: int) -> float:
    """SE of one strain's mean fitness over `n_plates` plates x `c` colonies."""
    return math.sqrt(var_p / n_plates + var_c / (n_plates * c))


def se_epsilon(se: float) -> float:
    """SE of eps = f_ab - f_a*f_b when all three estimates share the same SE."""
    return math.sqrt(se**2 + 2 * (F_BAR**2) * se**2)


def main() -> None:
    """Emit the explainer figure and the plain-numbers table behind it."""
    os.makedirs(IMG_DIR, exist_ok=True)
    boot = pd.read_csv(osp.join(RESULTS, "run3_strain_bootstrap.csv"))
    fit = pd.read_csv(osp.join(RESULTS, "run3_fitness_by_condition.csv"))
    var_p = float(boot["across_plate_sd"].mean()) ** 2
    var_c = float(fit["fitness_sd"].mean()) ** 2

    rows = []
    for n_plates in (2, 3, 4, 5, 6, 8, 10, 12, 16):
        se = se_strain(var_p, var_c, n_plates, C_DENSE)
        se_e = se_epsilon(se)
        rows.append(
            {
                "plates": n_plates,
                "se_strain": se,
                "callable_gap": 2.8 * se,
                "se_epsilon": se_e,
                "r_attenuation": math.sqrt(
                    EPS_DESIGN_SD**2 / (EPS_DESIGN_SD**2 + se_e**2)
                ),
                "gain_vs_prev_plate": np.nan,
            }
        )
    tab = pd.DataFrame(rows)
    se_vals = np.asarray(tab["se_strain"], dtype=float)
    p_vals = np.asarray(tab["plates"], dtype=float)
    tab["gain_vs_prev_plate"] = np.append(
        [np.nan], -np.diff(se_vals) / np.diff(p_vals)
    )
    tab.to_csv(osp.join(RESULTS, "run4_plates_explainer.csv"), index=False)
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(tab.to_string(index=False))

    print("\nplain reading:")
    for n_plates in (3, 4, 6, 8):
        r = tab.loc[tab.plates == n_plates].iloc[0]
        print(
            f"  {n_plates} plates: error bar +/-{r.se_strain:.3f}, "
            f"can call a gap of {r.callable_gap:.2f}, "
            f"epistasis correlation retains {r.r_attenuation:.0%} of the truth"
        )

    make_figure(var_p, var_c, osp.join(IMG_DIR, "plates_explainer"))
    print("\nfigure:", osp.join(IMG_DIR, "plates_explainer.svg"))


def make_figure(var_p: float, var_c: float, out_stem: str) -> None:
    """Four panels answering 'what does another plate buy me?'."""
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
        2, 2, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(104.0))
    )
    plates = np.arange(2, 17)
    se = np.array([se_strain(var_p, var_c, int(p), C_DENSE) for p in plates])

    # (a) the error bar itself, with the marginal gain per added plate
    ax = axes[0, 0]
    ax.plot(plates, se, "o-", ms=2.5, lw=0.9, color=PLOT_PALETTE[0])
    for p_mark in (3, 4, 8):
        v = se_strain(var_p, var_c, p_mark, C_DENSE)
        ax.annotate(f"P={p_mark}\n{v:.3f}", (p_mark, v), textcoords="offset points",
                    xytext=(5, 5), fontsize=5)
    ax.set_xlabel("plates")
    ax.set_ylabel("error bar on one strain (SE)")
    ax.set_title("a  what a plate buys: SE falls as 1/sqrt(P)", fontsize=6, loc="left")
    ax.set_ylim(0, 0.12)
    ax.yaxis.set_major_locator(MultipleLocator(0.04))
    ax.yaxis.set_minor_locator(MultipleLocator(0.02))

    ax2 = ax.twinx()
    gain = -np.diff(se)
    ax2.bar(plates[1:], gain, width=0.6, color=PLOT_PALETTE[5], alpha=0.35, zorder=0)
    ax2.set_ylabel("SE removed by that plate", fontsize=6, labelpad=1)
    ax2.set_ylim(0, 0.035)
    ax2.tick_params(labelsize=6, width=0.5, length=2)

    # (b) what you can actually call between two strains
    ax = axes[0, 1]
    ax.plot(plates, 2.8 * se, "o-", ms=2.5, lw=0.9, color=PLOT_PALETTE[1])
    ax.axhline(0.10, color="black", lw=0.7, ls="--")
    ax.annotate("a 10% fitness difference", (10.5, 0.105), fontsize=5)
    ax.set_xlabel("plates")
    ax.set_ylabel("smallest callable gap", labelpad=1)
    ax.set_title("b  what you can CALL between two strains", fontsize=6, loc="left")
    ax.set_ylim(0, 0.35)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))

    # (c) what it does to the model score
    ax = axes[1, 0]
    for sd_sig, col, lab in (
        (0.047, PLOT_PALETTE[5], "random pairs (SD 0.047)"),
        (0.140, PLOT_PALETTE[0], "spanning design (SD 0.140)"),
    ):
        att = [
            math.sqrt(sd_sig**2 / (sd_sig**2 + se_epsilon(s) ** 2)) for s in se
        ]
        ax.plot(plates, att, "-", lw=1.1, color=col, label=lab)
    ax.set_xlabel("plates")
    ax.set_ylabel("fraction of the true correlation kept")
    ax.set_title("c  what it does to the model score (epistasis)", fontsize=6, loc="left")
    ax.legend(frameon=False, loc="lower right")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    # (d) why adding strains is cheap -- and what a plate does instead
    ax = axes[1, 1]
    n_strains = np.arange(10, 90, 2)
    for n_plates, col in ((3, PLOT_PALETTE[5]), (4, PLOT_PALETTE[0]), (8, PLOT_PALETTE[2])):
        vals = [
            se_strain(var_p, var_c, n_plates, max(1, (WELLS - 20) // int(n)))
            for n in n_strains
        ]
        ax.plot(n_strains, vals, lw=1.1, color=col, label=f"{n_plates} plates")
    ax.axvline(26, color="black", lw=0.7, ls=":")
    ax.annotate("now: 26", (26, 0.115), fontsize=5, xytext=(3, 0),
                textcoords="offset points")
    ax.set_xlabel("strains sharing the plate")
    ax.set_ylabel("SE of a strain mean")
    ax.set_title("d  strains are cheap, plates are the lever", fontsize=6, loc="left")
    ax.legend(frameon=False, loc="center right")
    ax.set_ylim(0, 0.13)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(0.04))
    ax.yaxis.set_minor_locator(MultipleLocator(0.02))

    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.xaxis.set_major_locator(MultipleLocator(4))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
    for ax in axes.ravel():
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
