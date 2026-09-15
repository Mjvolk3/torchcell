# experiments/026-metabolism-flux/scripts/plot_per_gene_predictor_agreement.py
# [[experiments.026-metabolism-flux.scripts.plot_per_gene_predictor_agreement]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/plot_per_gene_predictor_agreement.py

r"""How far apart the predictors are for ONE GENE, which is the unit a model consumes.

The agreement table in ``compare_kinetic_predictors.py`` is keyed by
(enzyme, substrate) PAIR. That is the right key for asking whether two models predict the
same number for the same measurement, but it is not the key the flux layer reads: a
catalytic unit draws one :math:`k_{cat}` per gene, so the question that decides whether a
predicted table is usable is how much the choice of predictor moves THAT gene's value.

A gene can carry several units and several substrates, so the per-gene value is the median
over its pairs in :math:`\log_{10}` space, taken separately for each predictor. Median
rather than mean because the pair distribution is heavy-tailed and one lipid substrate
should not set a gene's turnover number.

SIX PANELS, three rows of two
------------------------------
a. The across-predictor spread per gene, in orders of magnitude. This is the error bar on a
   single gene's :math:`k_{cat}` when the predictor is chosen arbitrarily. (The JSON keys
   still say ``decades``; they are read by the plotting code and renaming them would break
   the figures for a word that no reader sees.)
b. The same spread laid against the dynamic range of :math:`k_{cat}` itself, so the
   spread can be compared to the signal rather than read in isolation.
c. Gene-level rank agreement between predictors. Pair-level Spearman is already reported;
   this asks the different question of whether two predictors order the GENES alike. The
   value is written into each cell: on a heatmap the number IS the datum, so this is the one
   place the no-text-inside-the-axes rule does not apply, and the layout is three rows of two
   rather than two of three so that the cells are large enough to carry a 6 pt number
   without shrinking it.
d. The genes that feed the betaxanthin cassette. Two of the four cassette members are
   heterologous and have no reaction in yeast-GEM, so no predictor emits a value for them;
   that absence is drawn rather than left out.
e. Predicted against measured, on the Open Enzyme Database pairs.
f. Accuracy against measured, ranked.
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
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.ticker import MultipleLocator

from torchcell.timestamp import timestamp
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    PLOT_PALETTE_FILL,
    PLOT_REGION_FILL,
    PLOT_REGION_STROKE,
    apply_paper_style,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()
DATA_ROOT = cast(str, os.getenv("DATA_ROOT"))
ASSET_IMAGES_DIR = cast(str, os.getenv("ASSET_IMAGES_DIR"))
EXPERIMENT_ROOT = cast(str, os.getenv("EXPERIMENT_ROOT"))
KINETICS = osp.join(DATA_ROOT, "data", "torchcell", "kinetics")
RESULTS = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
OUT_DIR = osp.join(ASSET_IMAGES_DIR, "026-metabolism-flux")

#: One predictor -> one color, EVERYWHERE. Panels d, e and f each drew their own series in
#: whatever order they happened to iterate, so UniKP was blue in one panel and orange in the
#: next. A reader comparing panels then reads two different enzymes. The map is keyed on the
#: predictor name so the color cannot depend on sort order.
PREDICTOR_COLOR_INDEX = {
    "dlkcat": 0,
    "unikp": 1,
    "eitlem": 2,
    "turnup": 3,
    "deepenzyme": 4,
    # EZKinetics takes slot 5 and Boost_KM slot 6, not the other way round. Slots 7-12 are
    # a 0.73 SHADE of slots 1-6, so slot 6 is dark orange and collides with DLKcat's orange
    # in the k_cat panels: the palette reserves lightness for validation-vs-test, never for
    # category. Boost_KM is K_M-only and never shares a panel with DLKcat, so it is the one
    # that can safely take a shade. This keeps all six k_cat predictors on distinct hues.
    "ezkinetics": 5,
    "boost_km": 6,
    # CatPred is the seventh k_cat series and also a K_M series, so the union of slots it
    # must avoid is 0 to 6 and there is no vivid hue left. It takes the MUTED tier's red,
    # slot 13, rather than a dark shade: slots 7 to 12 differ from the vivid tier in
    # LIGHTNESS, which the palette reserves for validation against test within one series,
    # while slots 13 to 18 differ in CHROMA, which is the tier the palette provides for a
    # new category. Slot 12, the muted orange, is the band's.
    "catpred": 13,
}

# A DELIBERATE DEVIATION from the primary-before-secondary rule, recorded so it is not
# "fixed" later. Six k_cat predictors need six slots, so blue (DeepEnzyme, fifth) and gray
# (EZKinetics, sixth) are correct there: the four primaries are spent first, which is
# exactly what the rule asks. The K_M figures then carry gray for EZKinetics with only four
# series, which the rule alone would forbid. One predictor, one color across every figure is
# the stronger constraint: a reader comparing the k_cat and K_M figures is comparing the
# same models, and a predictor that changed color between them would be read as two. Moving
# EZKinetics to a primary would put it on yellow beside Boost_KM's dark yellow, separating
# two categories by lightness, which is the encoding the palette reserves for validation
# against test.


def predictor_color(name: str) -> str:
    """The one color this predictor is drawn in, in every panel of every figure."""
    return PLOT_PALETTE[PREDICTOR_COLOR_INDEX[name] % len(PLOT_PALETTE)]


#: ONE color for the min-max envelope, whichever parameter the figure is drawn for. It used
#: to follow the parameter, orange for k_cat and purple for K_M, which made the two figures
#: of the same script look like two different figures; a reader compares them panel by
#: panel, so they have to be identical in styling and differ only in the data
#: (user, 2026-09-11). The parameter is named in the axis label and the caption.
PARAMETER_COLOR = {"k_cat": 0, "K_M": 0}

#: The within-class band of panel e is a reference REGION, not a series, so it must not take
#: a color any predictor in the figure already has. It is the muted tier's ORANGE,
#: terracotta (line #C48161, fill #F9D0BE): no predictor reaches past slot 7, its low chroma
#: keeps it behind the points, and it is the hue this figure family reads as the reference
#: envelope. It was dusty mauve until 2026-09-11, when the user asked for the light-purple
#: backgrounds of these four figures to be orange.
#: Zero-based, so 12 is the thirteenth color.
BAND_SLOT = 12


DISPLAY = {
    "dlkcat": "DLKcat",
    "unikp": "UniKP",
    "eitlem": "EITLEM",
    "turnup": "TurNuP",
    "deepenzyme": "DeepEnzyme",
    "boost_km": "Boost_KM",
    "ezkinetics": "EZKinetics",
    "catpred": "CatPred",
}

# The Btx-cassette as torchcell.datasets.scerevisiae.cachera2023 declares it, plus the
# aromatic-amino-acid genes immediately upstream that supply its tyrosine precursor.
# CYP76AD1 and DOD are the heterologous members: they are named here so their ABSENCE
# from every predictor table is visible in panel d rather than silently dropped.
CASSETTE: list[tuple[str, str]] = [
    ("CYP76AD1", "CYP76AD1"),
    ("DOD", "DOD"),
    ("YBR249C", "ARO4"),
    ("YPR060C", "ARO7"),
]
UPSTREAM: list[tuple[str, str]] = [
    ("YDR035W", "ARO3"),
    ("YDR127W", "ARO1"),
    ("YGL148W", "ARO2"),
    ("YBR166C", "TYR1"),
    ("YGL202W", "ARO8"),
    ("YHR137W", "ARO9"),
    ("YNL316C", "PHA2"),
]


def per_gene_log10(parameter: str) -> pd.DataFrame:
    """Gene x predictor table of median log10 values, from every built parquet."""
    columns: dict[str, pd.Series] = {}
    for predictor in sorted(os.listdir(KINETICS)):
        path = osp.join(KINETICS, predictor, "processed", f"{parameter}.parquet")
        if not osp.exists(path):
            continue
        frame = pd.read_parquet(path)
        values = frame[frame[parameter] > 0].copy()
        values["log10"] = np.log10(values[parameter])
        columns[predictor] = values.groupby("gene_id")["log10"].median()
    return pd.DataFrame(columns)


def box(ax: plt.Axes) -> None:
    """All four spines visible, the repo's boxed look."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)


def tenths(ax: plt.Axes) -> None:
    """A gridline every 0.1 on a 0 to 1 metric axis, labeled every 0.2."""
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(axis="y", which="minor", length=0)
    ax.grid(axis="y", which="both", linewidth=0.3, color="0.85")
    ax.set_axisbelow(True)


def legend_above(ax: plt.Axes, handles, names, ncol: int) -> None:
    """Put a legend OUTSIDE the axes, above it.

    Repo rule: nothing but data is drawn inside the axes. A legend placed with `loc=` sits
    over the data, gets shrunk to fit whatever space is left, and silently collides with any
    annotation that moves. Anchoring above the axes costs a little height and cannot collide.
    """
    ax.legend(
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


def main() -> None:
    """Draw the six panels and record every number they show."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameter", default="k_cat", choices=["k_cat", "K_M"])
    parser.add_argument("--no-timestamp", action="store_true")
    args = parser.parse_args()

    apply_paper_style()
    plt.rcParams.update({"xtick.major.width": 0.5, "ytick.major.width": 0.5})
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = "" if args.no_timestamp else f"_{timestamp()}"
    parameter = args.parameter
    symbol = r"$k_{cat}$" if parameter == "k_cat" else r"$K_M$"

    table = per_gene_log10(parameter)
    predictors = list(table.columns)
    complete = table.dropna()
    summary: dict[str, object] = {
        "parameter": parameter,
        "predictors": predictors,
        "n_genes_any": int(len(table)),
        "n_genes_all_predictors": int(len(complete)),
    }

    # Three rows of two, not two of three. At two-of-three every panel was 55 mm wide and
    # the heatmap in c came out with cells under 5 mm; at 88 mm a panel the cells are 7 mm
    # and carry their value at 6 pt. Height stays under the 170 mm ceiling.
    fig, axes = plt.subplots(
        3, 2, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(152)), dpi=300
    )

    # --- a. the across-predictor spread for one gene ---------------------------
    ax = axes[0, 0]
    spread = (complete.max(axis=1) - complete.min(axis=1)).to_numpy()
    median_spread = float(np.median(spread))
    ax.hist(spread, bins=45, color=PLOT_PALETTE[0], edgecolor="black", linewidth=0.3)
    ax.axvline(median_spread, color="black", linewidth=0.8, linestyle="--")
    # The median is NOT annotated on the panel; it is stated in the caption. See the
    # no-labels-inside-the-axes rule in CLAUDE.md.
    ax.set_xlabel(
        f"across-predictor spread of {symbol},  "
        r"$\log_{10}(\mathrm{max}/\mathrm{min})$"
        "\n(orders of magnitude)"
    )
    ax.set_ylabel("genes")
    box(ax)
    summary["spread_median_decades"] = median_spread
    summary["spread_p90_decades"] = float(np.percentile(spread, 90))
    summary["spread_frac_above_1_decade"] = float((spread > 1.0).mean())

    # --- b. spread against the dynamic range of the quantity itself ------------
    #
    # WHY THIS PANEL WAS REDRAWN. The band carries the whole message and it was drawn in
    # PLOT_PALETTE_FILL[0] with linewidth 0: a 92-percent-luminance wash with no edge, under
    # a 0.7 pt pure-black median line. The palest color in the palette was doing the primary
    # work while the darkest ink went to a curve that is monotone by construction, since the
    # genes are SORTED by it. Three changes, none of them cosmetic:
    #   1. the band keeps the pale fill but gains its own line color as a real edge, drawn as
    #      the min and max curves, so the envelope has a boundary the eye can find;
    #   2. the color follows the PARAMETER, so K_M is purple here and purple in the coverage
    #      figure rather than orange in one and purple in the other;
    #   3. x is the percentile of genes rather than an ordinal count, so the axis reads
    #      without the reader converting "gene 800 of 980" into a fraction.
    # The median line stays black and thin: it is the summary, and it should not outweigh
    # the quantity the panel exists to show.
    ax = axes[0, 1]
    slot = PARAMETER_COLOR[parameter]
    center = complete.median(axis=1).to_numpy()
    order = np.argsort(center)
    low = complete.min(axis=1).to_numpy()[order]
    high = complete.max(axis=1).to_numpy()[order]
    percentile = np.linspace(0, 100, len(order))
    ax.fill_between(
        percentile,
        low,
        high,
        color=PLOT_PALETTE_FILL[slot],
        linewidth=0,
        label="predictor min-max",
    )
    for edge in (low, high):
        ax.plot(percentile, edge, color=PLOT_PALETTE[slot], linewidth=0.6)
    ax.plot(
        percentile,
        center[order],
        color="black",
        linewidth=0.7,
        label="median over predictors",
    )
    ax.set_xlim(0, 100)
    ax.set_xlabel("gene percentile, ranked by median across predictors")
    ax.set_ylabel(f"log$_{{10}}$ {symbol}")
    handles, names = ax.get_legend_handles_labels()
    legend_above(ax, handles, names, ncol=2)
    box(ax)
    signal = float(np.percentile(center, 90) - np.percentile(center, 10))
    summary["central_p10_p90_range_decades"] = signal
    summary["spread_over_signal"] = float(median_spread / signal)

    # --- c. gene-level rank agreement -----------------------------------------
    ax = axes[1, 0]
    n = len(predictors)
    grid = np.full((n, n), np.nan)
    for i, left in enumerate(predictors):
        for j, right in enumerate(predictors):
            if i == j:
                continue
            both = table[[left, right]].dropna()
            if len(both) >= 3:
                grid[i, j] = both[left].corr(both[right], method="spearman")
    agreement_cmap = LinearSegmentedColormap.from_list(
        "tc_orange", ["#FFFFFF", PLOT_PALETTE_FILL[0], PLOT_PALETTE[0], PLOT_PALETTE[6]]
    )
    agreement_cmap.set_bad("#FFFFFF")
    # aspect="auto" lets the matrix fill the panel it is given. With the default equal
    # aspect the height bounds it, the panel sits as a square in the middle of an 88 mm
    # slot and the letter floats 20 mm from its own panel.
    image = ax.imshow(
        np.ma.masked_invalid(grid), cmap=agreement_cmap, vmin=0, vmax=1, aspect="auto"
    )
    ax.set_xticks(range(n))
    ax.set_xticklabels([DISPLAY.get(p, p) for p in predictors], rotation=35, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels([DISPLAY.get(p, p) for p in predictors])
    # The value goes in the cell. This is data, not a label: a heatmap cell's number is the
    # datum the color encodes, and the color alone cannot be read to two decimals. Black
    # throughout, because the scale tops out at 0.55 where the orange is still mid-tone.
    for i in range(n):
        for j in range(n):
            if np.isfinite(grid[i, j]):
                ax.text(
                    j,
                    i,
                    f"{grid[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="black",
                )
    # The colorbar takes its own strip to the right rather than the gap below, which is
    # what had squeezed the heatmap to a third of the panel height.
    bar = fig.colorbar(
        image, ax=ax, location="right", fraction=0.05, pad=0.04, aspect=22
    )
    bar.set_label("gene-level Spearman", fontsize=6)
    bar.ax.tick_params(labelsize=6, width=0.5)
    bar.outline.set_linewidth(0.5)
    box(ax)
    summary["gene_level_spearman"] = {
        f"{predictors[i]}|{predictors[j]}": (
            None if not np.isfinite(grid[i, j]) else float(grid[i, j])
        )
        for i in range(n)
        for j in range(n)
        if i < j
    }

    # --- d. the genes the betaxanthin case study runs on -----------------------
    ax = axes[1, 1]
    wanted = CASSETTE + UPSTREAM
    labels, offsets = [], []
    absent_rows: list[int] = []
    absent: list[str] = []
    for row, (orf, common) in enumerate(wanted):
        labels.append(common)
        offsets.append(row)
        if orf not in table.index:
            absent.append(common)
            absent_rows.append(row)
            continue
        for k, predictor in enumerate(predictors):
            value = table.loc[orf, predictor]
            if np.isfinite(value):
                ax.scatter(
                    value,
                    row,
                    s=9,
                    color=predictor_color(predictor),
                    edgecolors="black",
                    linewidths=0.3,
                    zorder=3,
                    label=DISPLAY.get(predictor, predictor) if row == 2 else None,
                )
    if absent_rows:
        # The shaded band marks the heterologous genes; what it means is said in the
        # caption, not written across the panel.
        ax.axhspan(
            min(absent_rows) - 0.45,
            max(absent_rows) + 0.45,
            color=PLOT_PALETTE_FILL[5],
            linewidth=0,
            zorder=0,
        )
    ax.set_yticks(offsets)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axhline(len(CASSETTE) - 0.5, color="black", linewidth=0.5, linestyle=":")
    ax.set_xlabel(f"log$_{{10}}$ {symbol}")
    ax.set_ylabel("cassette / tyrosine supply")
    handles, names = ax.get_legend_handles_labels()
    if handles:
        legend_above(ax, handles, names, ncol=3)
    box(ax)
    summary["cassette_genes_absent_from_all_predictors"] = absent

    # --- e and f. accuracy against measured values -----------------------------
    accuracy_path = osp.join(RESULTS, "predictor_vs_measured.json")
    pairs_path = osp.join(RESULTS, "predictor_vs_measured.csv")
    with open(accuracy_path) as handle:
        accuracy = json.load(handle)
    pairs = pd.read_csv(pairs_path)
    block = accuracy["parameters"][parameter]["predictors"]
    scored = {k: v for k, v in block.items() if "spearman" in v}

    # --- e. predicted against measured ----------------------------------------
    ax = axes[2, 0]
    subset = pairs[pairs["parameter"] == parameter]
    for k, predictor in enumerate(sorted(scored, key=lambda p: -scored[p]["spearman"])):
        rows = subset[subset["predictor"] == predictor]
        ax.scatter(
            np.log10(rows["measured"]),
            np.log10(rows["predicted"]),
            s=6,
            color=to_rgba(predictor_color(predictor), 0.85),
            edgecolors="black",
            linewidths=0.2,
            zorder=3,
            label=DISPLAY.get(predictor, predictor),
        )
    lo = float(np.log10(subset["measured"].min())) - 0.4
    hi = float(np.log10(subset["measured"].max())) + 0.4
    # The band of biological plausibility, not a tolerance on the prediction. Its half
    # width is the median standard deviation of log10 kcat across the distinct enzymes of
    # one EC4 class, measured over the whole Open Enzyme Database. A point inside it is no
    # further from the truth than two enzymes doing the SAME job are from each other, which
    # is what a predictor that had learned only the enzyme's function would achieve.
    with open(osp.join(RESULTS, "ec_class_variance.json")) as handle:
        within_class_sd = json.load(handle)["parameters"][parameter]["within_class"][
            "median_within_class_sd_decades"
        ]
    ax.fill_between(
        [lo, hi],
        [lo - within_class_sd, hi - within_class_sd],
        [lo + within_class_sd, hi + within_class_sd],
        color=PLOT_REGION_FILL,
        edgecolor=PLOT_REGION_STROKE,
        linewidth=0.4,
        zorder=1,
        label=f"within-class spread (±{within_class_sd:.2f})",
    )
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.6, linestyle="--", zorder=2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"measured log$_{{10}}$ {symbol}, Open Enzyme Database")
    ax.set_ylabel(f"predicted log$_{{10}}$ {symbol}")
    handles, names = ax.get_legend_handles_labels()
    legend_above(ax, handles, names, ncol=4)
    summary["within_class_band_decades"] = within_class_sd
    summary["frac_predictions_inside_within_class_band"] = {
        predictor: float(
            (
                np.abs(
                    np.log10(subset.loc[subset["predictor"] == predictor, "predicted"])
                    - np.log10(subset.loc[subset["predictor"] == predictor, "measured"])
                )
                < within_class_sd
            ).mean()
        )
        for predictor in sorted(scored)
    }
    box(ax)

    # --- f. accuracy, ranked --------------------------------------------------
    ax = axes[2, 1]
    ordered = sorted(scored, key=lambda p: -scored[p]["spearman"])
    positions = np.arange(len(ordered))
    width = 0.38
    ax.bar(
        positions - width / 2,
        [scored[p]["spearman"] for p in ordered],
        width,
        color=PLOT_PALETTE[0],
        edgecolor="black",
        linewidth=0.4,
        label="Spearman against measured",
    )
    # Two series, so the second takes the SECOND palette slot rather than the third.
    ax.bar(
        positions + width / 2,
        [scored[p]["median_abs_error_decades"] for p in ordered],
        width,
        color=PLOT_PALETTE[1],
        edgecolor="black",
        linewidth=0.4,
        label="median |error| (orders of magnitude)",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([DISPLAY.get(p, p) for p in ordered], rotation=35, ha="right")
    # Two different quantities share the axis, so the label names both rather than calling
    # them one thing. Higher is better for the first, lower for the second. The unit of
    # the second is in the legend entry; repeating it here ran the label into the panel
    # letter.
    ax.set_ylabel("Spearman  /  median |error|")
    tenths(ax)
    handles, names = ax.get_legend_handles_labels()
    legend_above(ax, handles, names, ncol=1)
    box(ax)
    summary["accuracy_vs_measured"] = scored
    summary["accuracy_n_pairs"] = (
        int(next(iter(scored.values()))["n_pairs"]) if scored else 0
    )
    summary["selection_effect"] = accuracy["parameters"][parameter].get(
        "selection_effect"
    )

    for letter, ax in zip("abcdef", axes.flatten()):
        ax.text(
            -0.14,
            1.14,
            letter,
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            va="top",
        )

    fig.tight_layout(w_pad=2.6, h_pad=2.8)
    base = f"per_gene_agreement_{parameter}{stamp}"
    savefig_true_size_svg(fig, osp.join(OUT_DIR, f"{base}.svg"))
    fig.savefig(osp.join(OUT_DIR, f"{base}.png"), dpi=300)
    plt.close(fig)

    os.makedirs(RESULTS, exist_ok=True)
    with open(osp.join(RESULTS, f"per_gene_agreement_{parameter}.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2)[:1200])
    print(f"figure written to {osp.join(OUT_DIR, base)}.svg")


if __name__ == "__main__":
    main()
