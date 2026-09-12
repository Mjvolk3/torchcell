# experiments/026-metabolism-flux/scripts/plot_ec_class_variance.py
# [[experiments.026-metabolism-flux.scripts.plot_ec_class_variance]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/plot_ec_class_variance.py

r"""Six panels on what a predictor's correlation is actually made of.

Reads ``results/ec_class_variance.json`` and the per-gene and accuracy reports, and draws
the comparison that decides how to read a Pearson computed across six decades: the
between-class range, the within-class biological spread, the measurement floor, and the
disagreement between predictors, all in the same unit of decades.

Legends sit outside the axes and no value is written into a data area, per CLAUDE.md.
Run from the repo root with the torchcell env, after ``ec_class_variance.py``.
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
import pandas as pd
from dotenv import load_dotenv
from matplotlib.ticker import MultipleLocator

from torchcell.timestamp import timestamp
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    PLOT_PALETTE_FILL,
    apply_paper_style,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()
ASSET_IMAGES_DIR = cast(str, os.getenv("ASSET_IMAGES_DIR"))
EXPERIMENT_ROOT = cast(str, os.getenv("EXPERIMENT_ROOT"))
RESULTS = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
OUT_DIR = osp.join(ASSET_IMAGES_DIR, "026-metabolism-flux")

DISPLAY = {
    "dlkcat": "DLKcat",
    "unikp": "UniKP",
    "eitlem": "EITLEM",
    "turnup": "TurNuP",
    "deepenzyme": "DeepEnzyme",
    "ezkinetics": "EZKinetics",
    "catpred": "CatPred",
}
#: Same map as the agreement figure, so a predictor keeps one color across the document.
#: CatPred takes the muted tier's red, slot 13: the vivid six are spent, and the muted tier
#: separates a new category by chroma rather than by the lightness the palette reserves for
#: two levels of one series.
PREDICTOR_COLOR_INDEX = {
    "dlkcat": 0, "unikp": 1, "eitlem": 2, "turnup": 3, "deepenzyme": 4, "ezkinetics": 5,
    "catpred": 13,
}


def color_of(name: str) -> str:
    return PLOT_PALETTE[PREDICTOR_COLOR_INDEX[name] % len(PLOT_PALETTE)]


def box(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)


def legend_above(ax: plt.Axes, ncol: int, handles=None, names=None) -> None:
    if handles is None:
        handles, names = ax.get_legend_handles_labels()
    ax.legend(
        handles, names, loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=ncol,
        frameon=False, fontsize=6, handlelength=1.2, handletextpad=0.4, borderpad=0.0,
        columnspacing=1.0,
    )


def swatch(color: str) -> plt.Rectangle:
    return plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="black", linewidth=0.4)


def tenths(ax: plt.Axes, low: float = 0.0) -> None:
    """A gridline every 0.1 on a metric axis, labeled every 0.2."""
    ax.set_ylim(low, 1.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(axis="y", which="minor", length=0)
    ax.grid(axis="y", which="both", linewidth=0.3, color="0.85")
    ax.set_axisbelow(True)


def panel_a(ax: plt.Axes, report: dict) -> None:
    """How far apart enzymes of the same EC class are, as the class is read to more digits.

    A class defined on fewer digits pools unlike chemistry, so the spread has to rise as
    digits are dropped. What the panel settles is how much of the four-digit spread is left
    once that pooling is removed, which is the part attributable to enzymes catalyzing the
    same reaction.
    """
    levels = [1, 2, 3, 4]
    width = 0.36
    positions = np.arange(len(levels))
    for offset, (parameter, color) in enumerate(
        (("k_cat", PLOT_PALETTE[0]), ("K_M", PLOT_PALETTE[2]))
    ):
        blocks = {
            b["ec_level"]: b for b in report["parameters"][parameter]["within_class_by_level"]
        }
        medians = [blocks[level]["median_within_class_sd_decades"] for level in levels]
        p90 = [blocks[level]["p90_within_class_sd_decades"] for level in levels]
        ax.bar(
            positions + (offset - 0.5) * width, medians, width, color=color,
            edgecolor="black", linewidth=0.4,
            label=r"$k_{cat}$" if parameter == "k_cat" else r"$K_M$",
        )
        # Upper whisker only: the 90th percentile of the class distribution, so the bar is
        # a median and the whisker says how heavy the tail above it is.
        ax.vlines(positions + (offset - 0.5) * width, medians, p90, color="black",
                  linewidth=0.6)
        # The spread with no class labels at all, which is what a class definition has to
        # beat to carry any information about the value.
        total = {
            b["level"]: b for b in report["parameters"][parameter]["variance_decomposition"]
        }[4]["total_sd_decades"]
        ax.axhline(total, color=color, linewidth=0.7, linestyle="--")
    ax.set_xticks(positions)
    ax.set_xticklabels([f"EC{level}" for level in levels])
    ax.set_xlabel("EC digits used to define a class")
    ax.set_ylabel("within-class s.d. of " r"$\log_{10}$" " value\n(orders of magnitude)")
    handles, names = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0], color="black", linewidth=0.7, linestyle="--"))
    names.append("no classes (total s.d.)")
    legend_above(ax, ncol=3, handles=handles, names=names)
    box(ax)


def panel_b(ax: plt.Axes, report: dict) -> None:
    """Between-class variance fraction as the EC number is read to more digits."""
    levels = [1, 2, 3, 4]
    width = 0.36
    positions = np.arange(len(levels))
    for offset, (parameter, color) in enumerate(
        (("k_cat", PLOT_PALETTE[0]), ("K_M", PLOT_PALETTE[2]))
    ):
        blocks = {b["level"]: b for b in report["parameters"][parameter]["variance_decomposition"]}
        ax.bar(
            positions + (offset - 0.5) * width,
            [blocks[level]["icc"] for level in levels],
            width, color=color, edgecolor="black", linewidth=0.4,
            label=r"$k_{cat}$" if parameter == "k_cat" else r"$K_M$",
        )
    ax.set_xticks(positions)
    ax.set_xticklabels([f"EC{level}" for level in levels])
    ax.set_xlabel("EC digits used to define a class")
    ax.set_ylabel("between-class variance\nfraction (ICC)")
    tenths(ax)
    legend_above(ax, ncol=2)
    box(ax)


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(values)
    return ordered, np.arange(1, len(ordered) + 1) / len(ordered)


def panel_c(
    ax: plt.Axes, condition_pairs: pd.DataFrame, classes: pd.DataFrame,
    predictor_spread: np.ndarray,
) -> None:
    """The three spreads as DISTRIBUTIONS on one axis, which is where they separate.

    Three summary bars could be read as three numbers that happen to differ. Three
    cumulative distributions show that the measurement distribution is essentially
    disjoint from the biological one: the question is not whether the medians differ but
    whether the populations overlap, and they barely do.
    """
    # Three series, so they take the first three palette slots IN ORDER. They used to run
    # blue, orange, red, which spends the blue slot before two of the warm primaries.
    for values, name, color in (
        (condition_pairs["range"].to_numpy(),
         "measurement floor\n(same enzyme and substrate)", PLOT_PALETTE[0]),
        (classes["range_decades"].to_numpy(),
         "biology\n(within one EC4 class)", PLOT_PALETTE[1]),
        (predictor_spread, "models\n(across predictors)", PLOT_PALETTE[2]),
    ):
        x, y = ecdf(values)
        ax.step(x, y, where="post", color=color, linewidth=1.0, label=name)
    ax.set_xscale("symlog", linthresh=0.05)
    ax.set_xlim(0, 6)
    ax.set_xlabel("spread of " r"$\log_{10}\ k_{cat}$" "\n(orders of magnitude)")
    ax.set_ylabel("cumulative proportion")
    ax.set_ylim(0, 1.02)
    legend_above(ax, ncol=3)
    box(ax)


def panel_d(ax: plt.Axes, report: dict, accuracy: dict) -> None:
    """Published predictors against a lookup table of the EC class median."""
    scored = {
        name: block for name, block in accuracy["parameters"]["k_cat"]["predictors"].items()
        if "spearman" in block
    }
    ordered = sorted(scored, key=lambda name: -scored[name]["spearman"])
    baseline = report["ec_class_baseline_k_cat"]
    positions = np.arange(len(ordered) + 1)
    values = [scored[name]["spearman"] for name in ordered] + [baseline["spearman"]]
    # White with a black hatch, not a palette color: the baseline is not a seventh
    # predictor, and gray is already spent on EZKinetics.
    colors = [color_of(name) for name in ordered] + ["white"]
    hatches = [""] * len(ordered) + ["///"]
    bars = ax.bar(positions, values, 0.7, color=colors, edgecolor="black", linewidth=0.4)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax.axhline(baseline["spearman"], color="black", linewidth=0.7, linestyle="--")
    ax.set_xticks(positions)
    ax.set_xticklabels([DISPLAY.get(name, name) for name in ordered] + ["EC median"],
                       rotation=35, ha="right")
    ax.set_ylabel("Spearman against measured")
    tenths(ax)
    legend_above(
        ax, ncol=2,
        handles=[
            plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black",
                          linewidth=0.4, hatch="///"),
            plt.Line2D([0], [0], color="black", linewidth=0.7, linestyle="--"),
        ],
        names=["EC-class median lookup", "its score"],
    )
    box(ax)


def panel_e(ax: plt.Axes, report: dict) -> None:
    """The same correlations with the between-class range removed by centering."""
    block = report["within_class_accuracy_k_cat"]["predictors"]
    ordered = sorted(block, key=lambda name: -block[name]["pearson_raw"])
    positions = np.arange(len(ordered))
    width = 0.38
    ax.bar(positions - width / 2, [block[name]["pearson_raw"] for name in ordered], width,
           color=[color_of(name) for name in ordered], edgecolor="black", linewidth=0.4,
           label="all pairs")
    lower = [
        block[name]["pearson_within_class"] - block[name]["pearson_within_class_ci95"][0]
        for name in ordered
    ]
    upper = [
        block[name]["pearson_within_class_ci95"][1] - block[name]["pearson_within_class"]
        for name in ordered
    ]
    ax.bar(positions + width / 2,
           [block[name]["pearson_within_class"] for name in ordered], width,
           color=[PLOT_PALETTE_FILL[PREDICTOR_COLOR_INDEX[name] % len(PLOT_PALETTE_FILL)]
                  for name in ordered],
           edgecolor="black", linewidth=0.4, label="within EC4 class",
           yerr=[lower, upper], error_kw={"elinewidth": 0.5, "capsize": 1.2,
                                          "capthick": 0.5, "ecolor": "black"})
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels([DISPLAY.get(name, name) for name in ordered], rotation=35,
                       ha="right")
    ax.set_ylabel("Pearson, " r"$\log_{10}\ k_{cat}$")
    tenths(ax, low=-0.8)
    legend_above(ax, ncol=2)
    box(ax)


def panel_f(ax: plt.Axes, report: dict) -> None:
    """The measurement floor, all conditions against the physiological window."""
    condition = report["parameters"]["k_cat"]["condition_spread"]
    groups = ["all reported\nconditions", "physiological\nwindow"]
    keys = ["all_conditions", "physiological_window"]
    positions = np.arange(len(groups))
    width = 0.36
    ax.bar(positions - width / 2,
           [condition[key]["median_range_decades"] for key in keys], width,
           color=PLOT_PALETTE_FILL[0], edgecolor="black", linewidth=0.4,
           label="all repeated pairs")
    ax.bar(positions + width / 2,
           [condition[key]["median_range_multi_condition_decades"] for key in keys], width,
           color=PLOT_PALETTE[0], edgecolor="black", linewidth=0.4,
           label="pairs measured at 2+ conditions")
    ax.set_xticks(positions)
    ax.set_xticklabels(groups)
    ax.set_ylabel("median within-pair range\n(orders of magnitude)")
    legend_above(ax, ncol=1)
    box(ax)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-timestamp", action="store_true")
    args = parser.parse_args()

    apply_paper_style()
    plt.rcParams.update({"xtick.major.width": 0.5, "ytick.major.width": 0.5})
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = "" if args.no_timestamp else f"_{timestamp()}"

    with open(osp.join(RESULTS, "ec_class_variance.json")) as handle:
        report = json.load(handle)
    with open(osp.join(RESULTS, "predictor_vs_measured.json")) as handle:
        accuracy = json.load(handle)
    with open(osp.join(RESULTS, "per_gene_agreement_k_cat.json")) as handle:
        report["across_predictor_spread_decades"] = json.load(handle)[
            "spread_median_decades"
        ]
    kcat_classes = pd.read_csv(osp.join(RESULTS, "ec_class_spread_k_cat.csv"))
    condition_pairs = pd.read_csv(osp.join(RESULTS, "condition_spread_pairs_k_cat.csv"))
    # The models' spread comes from the consensus table, which is per catalytic unit and is
    # the same quantity fig 7e draws, so the two figures cannot disagree.
    predictor_spread = pd.read_parquet(
        osp.join(cast(str, os.getenv("DATA_ROOT")), "data", "torchcell", "kinetics",
                 "_consensus", "k_cat_consensus.parquet")
    )["spread_decades"].to_numpy()

    fig, axes = plt.subplots(
        3, 2, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(148)), dpi=300
    )
    panel_a(axes[0, 0], report)
    panel_b(axes[0, 1], report)
    panel_c(axes[1, 0], condition_pairs, kcat_classes, predictor_spread)
    panel_d(axes[1, 1], report, accuracy)
    panel_e(axes[2, 0], report)
    panel_f(axes[2, 1], report)

    for letter, ax in zip("abcdef", axes.ravel()):
        ax.text(-0.14, 1.15, letter, transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="top")

    fig.tight_layout(w_pad=2.6, h_pad=2.6)
    base = f"ec_class_variance{stamp}"
    savefig_true_size_svg(fig, osp.join(OUT_DIR, f"{base}.svg"))
    fig.savefig(osp.join(OUT_DIR, f"{base}.png"), dpi=300)
    plt.close(fig)
    print(f"figure -> {osp.join(OUT_DIR, base)}.svg")


if __name__ == "__main__":
    main()
