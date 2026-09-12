# experiments/026-metabolism-flux/scripts/plot_catpred_admission.py
# [[experiments.026-metabolism-flux.scripts.plot_catpred_admission]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/plot_catpred_admission.py

r"""Two figures: whether CatPred is admitted, and what its variance is worth.

FIGURE 1, ``catpred_admission``
    **a**, **b** put CatPred beside the seven incumbents on the statistic the resolver
    actually decides with: the fraction of measured yeast pairs landing inside the standard
    deviation of the pair's OWN EC4 class, with a Wilson 95 percent interval, against the
    EC-median lookup's point fraction as a dashed line. A predictor is admitted when its
    lower whisker clears that line, so the verdict is legible from the panel rather than
    from a table. **c** is the reason the verdict has to be read carefully: how much of the
    scorable set CatPred was trained on. **d** is raw accuracy on the same pairs.

FIGURE 2, ``catpred_uncertainty``
    **a**, **b** ask whether the predicted standard deviation is a standard deviation:
    empirical coverage at one and two of them against the Gaussian rates. **c** asks the
    weaker and more useful question, whether it RANKS: median absolute error inside each
    quartile of the predicted total standard deviation. **d** asks whether a resolver could
    spend it, comparing four ways of combining the admitted predictors.

COLOR
-----
One meaning per color, held across both figures. In figure 1 a bar is a PREDICTOR and
takes that predictor's color from the shared map, CatPred the muted red no other predictor
uses; the two-level bars of **c** are one series whose lighter member is the wider
denominator. In figure 2 no predictor needs distinguishing, so the hues carry the
uncertainty COMPONENT instead: orange aleatoric, red epistemic, purple total, and the
fourth primary, yellow, carries the combination rules of **d**.

Run from the worktree root after ``within_class_plausibility.py``,
``compare_predictors_to_measured.py``, ``catpred_training_overlap.py`` and
``catpred_uncertainty.py``::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/plot_catpred_admission.py
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
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
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
OUT_DIR = osp.join(ASSET_IMAGES_DIR, "026-metabolism-flux")

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from plot_per_gene_predictor_agreement import DISPLAY, predictor_color  # noqa: E402

#: Orange aleatoric, red epistemic, purple total. Three of the four primaries, and the
#: fourth carries the combination rules, so no hue dominates either figure.
COMPONENT_COLOR = {"aleatoric": 0, "epistemic": 1, "total": 2}
RULE_SLOT = 3
SYMBOL = {"k_cat": r"$k_{cat}$", "K_M": r"$K_M$"}
RULE_LABEL = {
    "incumbent_median_without_catpred": "median,\nno CatPred",
    "plain_median": "plain\nmedian",
    "inverse_variance": "inverse\nvariance",
    "sd_gated": "s.d.\ngated",
}
RULE_ORDER = [
    "incumbent_median_without_catpred",
    "plain_median",
    "inverse_variance",
    "sd_gated",
]


def box(axis: Axes) -> None:
    """All four spines, which is the house look for a panel."""
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)


def tenths(axis: Axes) -> None:
    """A gridline every 0.1 on a 0 to 1 axis, labeled every 0.2."""
    axis.set_ylim(0, 1.0)
    axis.yaxis.set_major_locator(MultipleLocator(0.2))
    axis.yaxis.set_minor_locator(MultipleLocator(0.1))
    axis.tick_params(axis="y", which="minor", length=0)
    axis.grid(axis="y", which="both", linewidth=0.3, color="0.85")
    axis.set_axisbelow(True)


def legend_above(axis: Axes, ncol: int, handles: Any = None, names: Any = None) -> None:
    """Legends sit OUTSIDE the axes; nothing is drawn inside but data."""
    if handles is None:
        handles, names = axis.get_legend_handles_labels()
    axis.legend(
        handles,
        names,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=ncol,
        frameon=False,
        fontsize=6,
        handlelength=1.2,
        handletextpad=0.4,
        borderpad=0.0,
        columnspacing=1.0,
    )


def panel_own_class(axis: Axes, block: dict[str, Any], parameter: str) -> None:
    """Own-class EC4 fraction per predictor, with the lookup line the rule compares to."""
    entries = block["own_class_band"]["predictors"]
    ordered = sorted(entries, key=lambda p: -entries[p]["frac"])
    positions = np.arange(len(ordered))
    fractions = [entries[p]["frac"] for p in ordered]
    lower = [f - entries[p]["ci95"][0] for f, p in zip(fractions, ordered)]
    upper = [entries[p]["ci95"][1] - f for f, p in zip(fractions, ordered)]
    axis.bar(
        positions,
        fractions,
        0.7,
        color=[predictor_color(p) for p in ordered],
        edgecolor="black",
        linewidth=0.4,
    )
    axis.errorbar(
        positions,
        fractions,
        yerr=[lower, upper],
        fmt="none",
        ecolor="black",
        elinewidth=0.5,
        capsize=1.2,
    )
    axis.axhline(
        block["ec_median_lookup"]["own_class"]["frac"],
        color="black",
        linewidth=0.7,
        linestyle="--",
    )
    axis.set_xticks(positions)
    axis.set_xticklabels([DISPLAY[p] for p in ordered], rotation=35, ha="right")
    axis.set_ylabel(f"{SYMBOL[parameter]} pairs inside\ntheir own EC4 class band")
    tenths(axis)
    legend_above(
        axis,
        ncol=1,
        handles=[Line2D([0], [0], color="black", linewidth=0.7, linestyle="--")],
        names=["EC-median lookup"],
    )
    box(axis)


def panel_overlap(axis: Axes, overlap: dict[str, Any]) -> None:
    """How much of the scorable set, and of the GEM, CatPred was trained on."""
    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []
    line = predictor_color("catpred")
    fill = PLOT_PALETTE_FILL[13]
    n_measured = overlap["parameters"][0]["n_measured_pairs"]
    for block in overlap["parameters"]:
        short = "k$_{cat}$" if block["parameter"] == "k_cat" else "K$_M$"
        for level in block["levels"]:
            labels.append(f"{short}\n{level['level']}")
            values.append(level["frac"])
            colors.append(line)
        labels.append(f"{short}\nGEM")
        values.append(block["n_gem_accessions_in_db"] / block["n_gem_accessions"])
        colors.append(fill)
    positions = np.arange(len(labels))
    axis.bar(positions, values, 0.7, color=colors, edgecolor="black", linewidth=0.4)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=0)
    axis.set_ylabel("fraction in CatPred-DB")
    tenths(axis)
    legend_above(
        axis,
        ncol=2,
        handles=[
            Rectangle((0, 0), 1, 1, facecolor=line, edgecolor="black", linewidth=0.4),
            Rectangle((0, 0), 1, 1, facecolor=fill, edgecolor="black", linewidth=0.4),
        ],
        names=[f"of the {n_measured} measured pairs", "of the GEM's proteins"],
    )
    box(axis)


def panel_accuracy(axis: Axes, accuracy: dict[str, Any], parameter: str) -> None:
    """Median absolute error in decades on the measured pairs, per predictor."""
    entries = {
        name: block
        for name, block in accuracy["parameters"][parameter]["predictors"].items()
        if "median_abs_error_decades" in block
    }
    ordered = sorted(entries, key=lambda p: entries[p]["median_abs_error_decades"])
    positions = np.arange(len(ordered))
    axis.bar(
        positions,
        [entries[p]["median_abs_error_decades"] for p in ordered],
        0.7,
        color=[predictor_color(p) for p in ordered],
        edgecolor="black",
        linewidth=0.4,
    )
    axis.set_xticks(positions)
    axis.set_xticklabels([DISPLAY[p] for p in ordered], rotation=35, ha="right")
    axis.set_ylabel(
        f"median |error| on the measured\n{SYMBOL[parameter]} pairs (decades)"
    )
    axis.grid(axis="y", linewidth=0.3, color="0.85")
    axis.set_axisbelow(True)
    box(axis)


def panel_coverage(axis: Axes, block: dict[str, Any], parameter: str) -> None:
    """Empirical coverage at one and two predicted standard deviations."""
    records = {record["component"]: record for record in block["coverage_all"]}
    groups = ["1_sigma", "2_sigma"]
    positions = np.arange(len(groups))
    width = 0.26
    for index, component in enumerate(("aleatoric", "epistemic", "total")):
        values = [records[f"{component}_{g}"]["empirical"] for g in groups]
        lower = [
            v - records[f"{component}_{g}"]["ci95"][0] for v, g in zip(values, groups)
        ]
        upper = [
            records[f"{component}_{g}"]["ci95"][1] - v for v, g in zip(values, groups)
        ]
        axis.bar(
            positions + (index - 1) * width,
            values,
            width,
            color=PLOT_PALETTE[COMPONENT_COLOR[component]],
            edgecolor="black",
            linewidth=0.4,
            label=component,
        )
        axis.errorbar(
            positions + (index - 1) * width,
            values,
            yerr=[lower, upper],
            fmt="none",
            ecolor="black",
            elinewidth=0.5,
            capsize=1.0,
        )
    for record in (records["total_1_sigma"], records["total_2_sigma"]):
        axis.axhline(record["nominal"], color="black", linewidth=0.7, linestyle="--")
    axis.set_xticks(positions)
    axis.set_xticklabels(["±1 s.d.", "±2 s.d."])
    axis.set_ylabel(f"{SYMBOL[parameter]} pairs inside\nthe predicted interval")
    tenths(axis)
    handles, names = axis.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="black", linewidth=0.7, linestyle="--"))
    names.append("Gaussian rate")
    legend_above(axis, ncol=4, handles=handles, names=names)
    box(axis)


def panel_quartiles(axis: Axes, report: dict[str, Any]) -> None:
    """Median absolute error inside each quartile of the predicted total s.d."""
    labels: list[str] = []
    values: list[float] = []
    for block in report["parameters"]:
        short = "k$_{cat}$" if block["parameter"] == "k_cat" else "K$_M$"
        for index, value in enumerate(block["mae_by_sd_quartile"]):
            labels.append(f"{short}\nQ{index + 1}")
            values.append(value)
    positions = np.arange(len(labels))
    axis.bar(
        positions,
        values,
        0.7,
        color=PLOT_PALETTE[COMPONENT_COLOR["total"]],
        edgecolor="black",
        linewidth=0.4,
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_ylabel("median |error| (decades)")
    axis.set_xlabel("quartile of the predicted total standard deviation")
    axis.grid(axis="y", linewidth=0.3, color="0.85")
    axis.set_axisbelow(True)
    box(axis)


def panel_rules(axis: Axes, report: dict[str, Any]) -> None:
    """The four ways of combining the admitted predictors, scored on the same pairs."""
    labels: list[str] = []
    values: list[float] = []
    for block in report["parameters"]:
        short = "k$_{cat}$" if block["parameter"] == "k_cat" else "K$_M$"
        scored = {rule["rule"]: rule for rule in block["combinations"]}
        for rule in RULE_ORDER:
            labels.append(f"{short}\n{RULE_LABEL[rule]}")
            values.append(scored[rule]["median_abs_error_decades"])
    positions = np.arange(len(labels))
    axis.bar(
        positions,
        values,
        0.7,
        color=PLOT_PALETTE[RULE_SLOT],
        edgecolor="black",
        linewidth=0.4,
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, fontsize=5)
    axis.set_ylabel("median |error| on the\nmeasured pairs (decades)")
    axis.grid(axis="y", linewidth=0.3, color="0.85")
    axis.set_axisbelow(True)
    box(axis)


def letters(figure: Figure, axes: list[Axes], dx: float = -0.13) -> None:
    """Panel letters at each panel's OUTER top-left, above the axes."""
    for letter, axis in zip("abcdef", axes):
        axis.text(
            dx,
            1.22,
            letter,
            transform=axis.transAxes,
            fontsize=8,
            fontweight="bold",
            va="top",
            ha="left",
        )


def main() -> None:
    """Draw both figures from the result files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-timestamp", action="store_true")
    # The plausibility file that HAS CatPred in it. The shipped name is what a running
    # training cell resolves against and is deliberately not the default input here.
    parser.add_argument("--suffix", default="_with_catpred")
    args = parser.parse_args()

    apply_paper_style()
    plt.rcParams.update({"xtick.major.width": 0.5, "ytick.major.width": 0.5})
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = "" if args.no_timestamp else f"_{timestamp()}"

    with open(
        osp.join(RESULTS, f"within_class_plausibility{args.suffix}.json")
    ) as handle:
        plausibility = json.load(handle)
    with open(osp.join(RESULTS, "predictor_vs_measured.json")) as handle:
        accuracy = json.load(handle)
    with open(osp.join(RESULTS, "catpred_training_overlap.json")) as handle:
        overlap = json.load(handle)
    with open(osp.join(RESULTS, "catpred_uncertainty.json")) as handle:
        uncertainty = json.load(handle)

    width = mm_to_in(PANEL_WIDTHS_MM["full"])
    figure, grid = plt.subplots(2, 2, figsize=(width, mm_to_in(96.0)))
    figure.subplots_adjust(
        left=0.10, right=0.995, top=0.86, bottom=0.16, wspace=0.30, hspace=0.72
    )
    panel_own_class(grid[0][0], plausibility["parameters"]["k_cat"], "k_cat")
    panel_own_class(grid[0][1], plausibility["parameters"]["K_M"], "K_M")
    panel_overlap(grid[1][0], overlap)
    panel_accuracy(grid[1][1], accuracy, "k_cat")
    letters(figure, list(grid.flat))
    base = osp.join(OUT_DIR, f"catpred_admission{stamp}")
    figure.savefig(f"{base}.png", dpi=300)
    savefig_true_size_svg(figure, f"{base}.svg")
    plt.close(figure)
    print(f"-> {base}.svg")

    figure, grid = plt.subplots(2, 2, figsize=(width, mm_to_in(94.0)))
    figure.subplots_adjust(
        left=0.10, right=0.995, top=0.86, bottom=0.18, wspace=0.30, hspace=0.62
    )
    blocks = {block["parameter"]: block for block in uncertainty["parameters"]}
    panel_coverage(grid[0][0], blocks["k_cat"], "k_cat")
    panel_coverage(grid[0][1], blocks["K_M"], "K_M")
    panel_quartiles(grid[1][0], uncertainty)
    panel_rules(grid[1][1], uncertainty)
    letters(figure, list(grid.flat))
    base = osp.join(OUT_DIR, f"catpred_uncertainty{stamp}")
    figure.savefig(f"{base}.png", dpi=300)
    savefig_true_size_svg(figure, f"{base}.svg")
    plt.close(figure)
    print(f"-> {base}.svg")


if __name__ == "__main__":
    main()
