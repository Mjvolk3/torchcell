# experiments/026-metabolism-flux/scripts/plot_arm_comparison.py
# [[experiments.026-metabolism-flux.scripts.plot_arm_comparison]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/plot_arm_comparison.py

r"""Does any flux arm beat its own null? Drawn so the answer cannot be misread.

The 026 arm ordering looks like the ladder the design predicted, and quoting it as one
would be wrong. Two facts have to be drawn alongside every arm mean or the picture
inverts:

* the score is a **maximum over epochs** of validation Pearson, an upward-biased order
  statistic whose bias grows with the number of epochs searched;
* the comparison is not against zero, it is against a **label-permutation null** trained on
  permuted targets and validated on real ones.

Panel a therefore draws the null band behind the arms rather than beside them, panel b
draws what is left of each arm at the end of training, and panel c puts the whole sweep in
the context of the two architecture families, which is the effect that dwarfs every arm
difference here.
"""

import argparse
import json
import os
import os.path as osp
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

from torchcell.timestamp import timestamp
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    apply_paper_style,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()
ASSET_IMAGES_DIR = cast(str, os.getenv("ASSET_IMAGES_DIR"))
EXPERIMENT_ROOT = cast(str, os.getenv("EXPERIMENT_ROOT"))
RESULTS = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
OUT_DIR = osp.join(ASSET_IMAGES_DIR, "026-metabolism-flux")

SWEEP = "sweep_summary_2026-09-03-13-32-33.json"

# Display order is the ablation ladder, weakest first, so a reader tracks what each arm adds
# rather than a ranking.
ORDER = [
    ("arms:arms-flux_nullspace", "nullspace"),
    ("arms:arms-pooled", "pooled"),
    ("arms:arms-flux_off", "flux_off"),
    ("arms:arms-flux_free", "flux_free"),
    ("arms:arms-flux_anchored", "flux_anchored"),
]

# Measured betaxanthin validation Pearson by backbone family. Sources named in the caption;
# every value is a peak val Pearson from a completed run, not an estimate.
WEAK_FAMILY = [0.1233, 0.0832, 0.1057, 0.0572, 0.1209, 0.1314, 0.1151, 0.0837, 0.0416]
STRONG_FAMILY = [0.3227, 0.4135, 0.4340, 0.4301, 0.4050]


def box(ax: plt.Axes) -> None:
    """All four spines visible, the repo's boxed look."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)


def main() -> None:
    """Draw the three panels and echo the numbers they rest on."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-timestamp", action="store_true")
    args = parser.parse_args()

    apply_paper_style()
    plt.rcParams.update({"xtick.major.width": 0.5, "ytick.major.width": 0.5})
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = "" if args.no_timestamp else f"_{timestamp()}"

    with open(osp.join(RESULTS, SWEEP)) as handle:
        sweep = json.load(handle)["summary"]
    rows = {r["label"]: r for r in sweep["rows"]}
    null = sweep["null"]

    fig, axes = plt.subplots(
        1, 3, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(52)), dpi=300
    )

    # --- a. every arm against the null band ------------------------------------
    ax = axes[0]
    labels = [name for key, name in ORDER if key in rows]
    means = [rows[key]["peak_mean"] for key, _ in ORDER if key in rows]
    sems = [rows[key]["peak_sem"] for key, _ in ORDER if key in rows]
    positions = np.arange(len(labels))

    # The null band goes down FIRST, so the arms are read against it rather than against 0.
    ax.axhspan(
        null["peak_mean"] - null["peak_sd"],
        null["peak_p95"],
        color=PLOT_PALETTE[5],
        alpha=0.20,
        linewidth=0,
        zorder=0,
    )
    ax.axhline(null["peak_mean"], color=PLOT_PALETTE[5], linewidth=0.8, zorder=1,
               label=f"permuted-label null, mean {null['peak_mean']:.3f}")
    ax.axhline(null["peak_p95"], color=PLOT_PALETTE[5], linewidth=0.8, linestyle="--",
               zorder=1, label=f"null p95 = {null['peak_p95']:.3f}")
    ax.errorbar(
        positions, means, yerr=sems, fmt="o", markersize=3.2,
        color=PLOT_PALETTE[1], ecolor="black", elinewidth=0.6, capsize=2,
        markeredgecolor="black", markeredgewidth=0.3, zorder=3,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("peak val Pearson (mean $\\pm$ s.e.m.)")
    ax.set_ylim(0, 0.20)
    ax.legend(loc="upper left", frameon=False, fontsize=4.6, handlelength=1.3,
              handletextpad=0.4, borderpad=0.1, labelspacing=0.25)
    box(ax)

    # --- b. what survives to the end of training -------------------------------
    ax = axes[1]
    last5 = [rows[key]["last5_mean"] for key, _ in ORDER if key in rows]
    width = 0.38
    ax.bar(positions - width / 2, means, width, color=PLOT_PALETTE[1],
           edgecolor="black", linewidth=0.4, label="peak (max over epochs)")
    ax.bar(positions + width / 2, last5, width, color=PLOT_PALETTE[3],
           edgecolor="black", linewidth=0.4, label="mean of final 5 epochs")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("val Pearson")
    ax.legend(loc="upper left", frameon=False, fontsize=4.8, handlelength=1.2,
              handletextpad=0.4, borderpad=0.1)
    box(ax)

    # --- c. the effect that dwarfs every arm -----------------------------------
    ax = axes[2]
    for i, (values, name, color) in enumerate(
        [(WEAK_FAMILY, "weak backbone", PLOT_PALETTE[5]),
         (STRONG_FAMILY, "strong backbone", PLOT_PALETTE[0])]
    ):
        jitter = (np.random.default_rng(0).random(len(values)) - 0.5) * 0.22
        ax.scatter(np.full(len(values), i) + jitter, values, s=11, color=color,
                   edgecolors="black", linewidths=0.3, zorder=3, label=name)
        ax.plot([i - 0.24, i + 0.24], [np.median(values)] * 2, color="black",
                linewidth=1.0, zorder=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["2 graphs\nlearnable emb\nMSE", "9 graphs\nprot_T5\nCRPS"])
    ax.set_ylabel("peak val Pearson")
    ax.set_ylim(0, 0.5)
    ax.set_xlim(-0.5, 1.5)
    box(ax)

    for letter, ax in zip("abc", axes):
        ax.text(-0.20, 1.07, letter, transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="top")

    fig.tight_layout()
    base = f"arm_comparison{stamp}"
    savefig_true_size_svg(fig, osp.join(OUT_DIR, f"{base}.svg"))
    fig.savefig(osp.join(OUT_DIR, f"{base}.png"), dpi=300)
    plt.close(fig)

    report = {
        "null": null,
        "arms": {name: rows[key] for key, name in ORDER if key in rows},
        "n_arms_above_null_p95": sum(m > null["peak_p95"] for m in means),
        "weak_family_median": float(np.median(WEAK_FAMILY)),
        "strong_family_median": float(np.median(STRONG_FAMILY)),
    }
    with open(osp.join(RESULTS, "arm_comparison_summary.json"), "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report["arms"], indent=2)[:900])
    print(f"\narms above null p95: {report['n_arms_above_null_p95']} of {len(means)}")
    print(f"figure -> {osp.join(OUT_DIR, base)}.svg")


if __name__ == "__main__":
    main()
