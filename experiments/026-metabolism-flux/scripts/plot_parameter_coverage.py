# experiments/026-metabolism-flux/scripts/plot_parameter_coverage.py
# [[experiments.026-metabolism-flux.scripts.plot_parameter_coverage]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/plot_parameter_coverage.py

r"""Eight panels on how far the three parameter tables cover yeast-GEM, and what covers means.

Drawn so the two results that invert the naive reading cannot be missed:

* raw reaction coverage understates the model, because most unpriced reactions cannot carry
  flux under the wild-type medium (panel f);
* composition-matched surrogate pricing would raise coverage to 93.5 percent on numbers that
  do not determine a sign, so it is drawn as the negative result it is (panel b).

Every legend sits OUTSIDE its axes and no value is written into the data area; the numbers
live in the caption and in ``results/parameter_coverage.json``. That is the repo rule in
CLAUDE.md, and it is also what keeps this figure's text at 6 pt rather than shrinking
annotations to fit around bars.

THE PREDICTOR SUBSET, and why panels e, g and h use one
-------------------------------------------------------
Panel d enumerates every predictor, because the thing it shows is the 95.3 percent input
ceiling and the gap to the measured sources: dropping predictors there would delete the
result. Panels e, g and h ask a different question, what a table built from the predictors
that actually fit yeast enzymes would carry, so they run on a SELECTED subset.

The selection is computed, not chosen: :func:`select_predictors` ranks predictors by the
mean of the two leakage-robust within-class plausibility fractions in
``within_class_plausibility.json`` (the shared EC4 band and each pair's own class band),
keeps at most three, and drops any that does not beat the EC-median lookup on that same
statistic. Accuracy against the 57 measured pairs is NOT in the rule, because every
predictor trained on the corpus those pairs come from; it is reported alongside so the two
readings can be compared. The rule is what the caption states.

Reads ``results/parameter_coverage.json``, ``results/unresolved_metabolites.json`` and
``results/within_class_plausibility.json``, plus the surrogate tables and the per-predictor
parquets under ``$DATA_ROOT``. Run from the repo root with the torchcell env.
"""

import argparse
import json
import os
import os.path as osp
import sys
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.ticker import MultipleLocator

from torchcell.metabolism.parameters import (
    KINETICS_VERSION,
    PLAUSIBILITY_RESULT,
    PredictorSelection,
)
from torchcell.timestamp import timestamp
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    apply_paper_style,
    mm_to_in,
    savefig_true_size_svg,
)

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from plot_per_gene_predictor_agreement import DISPLAY  # noqa: E402

load_dotenv()
ASSET_IMAGES_DIR = cast(str, os.getenv("ASSET_IMAGES_DIR"))
EXPERIMENT_ROOT = cast(str, os.getenv("EXPERIMENT_ROOT"))
DATA_ROOT = cast(str, os.getenv("DATA_ROOT"))
RESULTS = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
OUT_DIR = osp.join(ASSET_IMAGES_DIR, "026-metabolism-flux")
THERMO_DIR = osp.join(DATA_ROOT, "data", "torchcell", "thermo_equilibrator")
KINETICS_DIR = osp.join(DATA_ROOT, "data", "torchcell", "kinetics")
CONSENSUS_DIR = osp.join(KINETICS_DIR, "_consensus")

#: How many predictors panels e, g and h may keep. Three is the most a grouped bar panel
#: carries at 88 mm without the x labels rotating past legibility.
MAX_SELECTED = 3

#: Per-predictor coverage of the 3,728 catalytic units, from the per-predictor summaries.
KCAT_COVERAGE = {
    "DLKcat": 0.9528,
    "UniKP": 0.9528,
    "EITLEM": 0.9528,
    "EZKinetics": 0.9528,
    "DeepEnzyme": 0.9512,
    "TurNuP": 0.5515,
}
KM_COVERAGE = {
    "UniKP": 0.9528,
    "EITLEM": 0.9528,
    "EZKinetics": 0.9528,
    "Boost_KM": 0.9303,
}

LADDER_LABELS = {
    "shipped_only": "GEM shipped",
    "recomputed_only": "eQuilibrator",
    "union_shipped_recomputed": "union",
    "union_plus_tight_surrogate": "union + tight\nsurrogate",
}


def box(ax: plt.Axes) -> None:
    """All four spines visible at 0.5 pt, the repo's boxed look."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)


def legend_above(ax: plt.Axes, ncol: int, handles=None, names=None) -> None:
    """Legend outside the axes, above it. Repo rule: only data goes inside the axes."""
    if handles is None:
        handles, names = ax.get_legend_handles_labels()
    ax.legend(
        handles, names, loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=ncol,
        frameon=False, fontsize=6, handlelength=1.2, handletextpad=0.4, borderpad=0.0,
        columnspacing=1.0,
    )


def swatch(color: str, hatch: str = "") -> plt.Rectangle:
    return plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="black", linewidth=0.4,
                         hatch=hatch)


HATCH = "////"


def tenths(ax: plt.Axes) -> None:
    """A gridline every 0.1 on a 0 to 1 metric axis, labeled every 0.2."""
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(axis="y", which="minor", length=0)
    ax.grid(axis="y", which="both", linewidth=0.3, color="0.85")
    ax.set_axisbelow(True)


# ONE SELECTION RULE, AND IT LIVES IN THE CODE THE FLUX LAYER RUNS.
# ------------------------------------------------------------------
# This figure used to rank predictors by its own statistic: the mean of the shared-EC4-band
# and own-class-band fractions, top three, drop anything at or below the EC-median lookup.
# `PredictorSelection.from_study` in `torchcell/metabolism/parameters.py` used a different
# one: the own-class band alone, keeping a predictor when its WILSON 95 PERCENT LOWER BOUND
# clears the lookup's point fraction. Both were in the repo and they disagreed on K_M, where
# the figure's rule admitted EZKinetics and EITLEM and the code's admitted EZKinetics alone.
#
# The code's rule wins and this file now calls it, for two reasons. It is the rule the flux
# layer actually applies, so a figure using any other rule describes a table nobody builds.
# And it is the stricter of the two on the evidence available: with 44 to 47 scorable pairs,
# a point estimate above the baseline is not evidence of being above the baseline, and
# EITLEM's K_M lower bound (0.372) sits below the lookup (0.489). Admitting it would need a
# rule that states an error tolerance, which is a decision, not a default.
def plausibility(block: dict, name: str) -> float:
    """The selection statistic, kept for the figure's ranking axis: the own-class fraction."""
    return float(block["own_class_band"]["predictors"][name]["frac"])


def lookup_plausibility(block: dict) -> float:
    """The EC-median lookup's own-class fraction, which is the bar to clear."""
    return float(block["ec_median_lookup"]["own_class"]["frac"])


def select_predictors(parameter: str) -> list[str]:
    """The admitted set, read from the one rule the flux layer applies.

    Delegated rather than reimplemented: two implementations of one rule is how the repo
    came to hold two answers for K_M.
    """
    return PredictorSelection.from_study(
        parameter, osp.join(EXPERIMENT_ROOT, PLAUSIBILITY_RESULT)
    ).selected


def unit_spread(parameter: str, predictors: list[str]) -> np.ndarray:
    r"""Across-predictor spread per catalytic unit, over the given predictors only.

    The consensus parquet carries a ``spread_decades`` column, but it is taken over EVERY
    predictor that emitted a value, so it cannot answer what a table restricted to the
    selected ones would carry. Recomputed here from the per-predictor parquets: one median
    :math:`\\log_{10}` value per unit per predictor, then max minus min over the units all
    of the selected predictors cover.
    """
    columns: dict[str, pd.Series] = {}
    for predictor in predictors:
        # The VERIFIED tables: the ones built on substrate structures whose composition
        # was checked against the GEM's formula, which is what the resolver reads.
        path = osp.join(KINETICS_DIR, predictor, KINETICS_VERSION, f"{parameter}.parquet")
        frame = pd.read_parquet(path)
        values = frame[frame[parameter] > 0].copy()
        values["log10"] = np.log10(values[parameter])
        columns[predictor] = values.groupby("unit_id")["log10"].median()
    table = pd.DataFrame(columns).dropna()
    return (table.max(axis=1) - table.min(axis=1)).to_numpy()


def panel_a(ax: plt.Axes, report: dict) -> None:
    """The DrG coverage ladder at the level the flux layer consumes."""
    steps = [s for s in report["delta_g_ladder"] if s["step"] in LADDER_LABELS]
    labels = [LADDER_LABELS[s["step"]] for s in steps]
    fracs = [s["frac_reactions"] for s in steps]
    # A four-step ladder is ONE ordered series, so it takes the first four palette slots in
    # order. It used to run gray, blue, orange, red, which reads as four unrelated
    # categories and spends blue and gray before the warm primaries.
    colors = PLOT_PALETTE[:4]
    positions = np.arange(len(labels))
    ax.bar(positions, fracs, 0.66, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_xticks(positions)
    # Horizontal: at 88 mm a panel the four names have 22 mm each, which is enough. At the
    # old 55 mm they had to be rotated or they printed as "GEMeQuilibratorunion".
    ax.set_xticklabels(labels)
    ax.set_ylabel("reactions with all\nparticipants priced")
    tenths(ax)
    box(ax)


def panel_b(ax: plt.Axes, surrogates: pd.DataFrame, threshold: float) -> None:
    """Why composition matching cannot buy coverage: the spread it leaves behind."""
    spread = surrogates["surrogate_spread_kj_per_mol"].to_numpy()
    bins = np.logspace(-2, np.log10(max(spread.max(), 10.0)), 34)
    ax.hist(spread, bins=bins, color=PLOT_PALETTE[1], edgecolor="black", linewidth=0.3,
            label="same-composition spread")
    ax.axvline(threshold, color="black", linewidth=0.9, linestyle="--",
               label=f"{threshold:.0f} kJ/mol threshold")
    ax.set_xscale("log")
    ax.set_xlabel("spread across same-composition\ncache compounds (kJ/mol)")
    ax.set_ylabel("metabolites")
    legend_above(ax, ncol=1)
    box(ax)


#: Short forms for panel c's y labels, so nine rows fit a 37 mm panel without colliding.
#: The caption gives the full name of every one of them.
CLASS_ABBREV = {
    "phosphatidylinositol_PI": "PI",
    "cardiolipin_CL": "CL",
    "triacylglycerol_TAG": "TAG",
    "phosphatidate_PA": "PA",
    "phosphatidylethanolamine_PE": "PE",
    "phosphatidylglycerol_PG": "PG",
    "phosphatidylcholine_PC": "PC",
    "acyl_CoA": "acyl-CoA",
    "fatty_acid_acyl_chain": "acyl chain",
    "glycan": "glycan",
    "sphingolipid": "sphingolipid",
}


def panel_c(ax: plt.Axes, unresolved: dict) -> None:
    """What is actually missing, by chemical class."""
    by_class = unresolved["baseline"]["unmatched_by_class"]
    items = sorted(by_class.items(), key=lambda kv: kv[1])[-9:]
    # ABBREVIATE. Nine full chemical-class names in a 37 mm row overlap each other, and the
    # names already carry their own abbreviation; the caption spells them out.
    labels = [CLASS_ABBREV.get(k, k.replace("_", " ")) for k, _ in items]
    counts = [v for _, v in items]
    positions = np.arange(len(labels))
    lipid = ("phosphatidyl", "cardiolipin", "triacyl", "phosphatidate", "sphingo", "acyl")
    # Two categories take TWO PRIMARIES, orange and red. The second used to be the gray
    # slot, which is a secondary: blue and gray enter a series at the fifth and sixth color,
    # never as the second.
    colors = [
        PLOT_PALETTE[0] if any(t in k for t in lipid) else PLOT_PALETTE[1]
        for k, _ in items
    ]
    ax.barh(positions, counts, 0.7, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("metabolites with no accession match")
    ax.set_xlim(0, max(counts) * 1.04)
    legend_above(
        ax, ncol=2,
        handles=[swatch(PLOT_PALETTE[0]), swatch(PLOT_PALETTE[1])],
        names=["lipid species", "other"],
    )
    box(ax)


def panel_d(ax: plt.Axes, measured: dict) -> None:
    """Per-predictor coverage of the 3,728 catalytic units, both parameters, and what the
    measured sources supply, which is the gap that motivates the predictors.

    The Open Enzyme Database prices a unit when every gene of the unit has a wild-type
    measurement for a yeast enzyme. EnzEngDB sits at zero because it holds no kcat or KM
    field, no EC, no organism field and no S. cerevisiae content; it is drawn rather than
    dropped so the zero is visible.
    """
    entries = [(k, v, "k_cat") for k, v in KCAT_COVERAGE.items()]
    entries += [(k, v, "K_M") for k, v in KM_COVERAGE.items()]
    sources = measured["sources"]
    entries += [
        ("OED", sources["open_enzyme_database"]["k_cat"]["frac"], "measured"),
        ("EnzEngDB", sources["enzengdb"]["k_cat"]["frac"], "measured"),
    ]
    labels = [e[0] for e in entries]
    values = [e[1] for e in entries]
    # Three categories, three PRIMARIES: orange, purple, red. "measured" used to take the
    # gray slot, spending a secondary as the third color of a three-series panel.
    color_of = {
        "k_cat": PLOT_PALETTE[0],
        "K_M": PLOT_PALETTE[2],
        "measured": PLOT_PALETTE[1],
    }
    colors = [color_of[e[2]] for e in entries]
    positions = np.arange(len(labels))
    ax.bar(positions, values, 0.7, color=colors, edgecolor="black", linewidth=0.4)
    ax.axhline(0.9528, color="black", linewidth=0.7, linestyle="--")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("catalytic units covered")
    tenths(ax)
    legend_above(
        ax, ncol=2,
        handles=[
            swatch(PLOT_PALETTE[0]),
            swatch(PLOT_PALETTE[2]),
            swatch(PLOT_PALETTE[1]),
            plt.Line2D([0], [0], color="black", linewidth=0.7, linestyle="--"),
        ],
        names=[r"$k_{cat}$ predicted", r"$K_M$ predicted", "measured, either parameter",
               "95.3% ceiling"],
    )
    box(ax)


def panel_e(
    ax: plt.Axes, spreads: dict[str, np.ndarray], classes: pd.DataFrame
) -> None:
    """Across-predictor disagreement per unit over the SELECTED predictors, against how far
    real enzymes of the same function actually sit apart.

    Restricted to the predictors panel g and h show clearing the lookup, because that is
    the table anyone would actually build. Over all six the spread also carries the
    disagreement of predictors nobody would use, which answers a question about the
    literature rather than about a usable table.

    The biological distribution is the reference that makes the predictor one readable:
    disagreement wider than it means the choice of predictor moves a turnover number
    further than swapping the enzyme for another one catalyzing the same reaction.
    """
    # ONLY k_cat. The admission rule keeps a single K_M predictor, so the K_M
    # across-predictor spread is identically zero and drawing it puts a spike at the origin
    # that flattens the two distributions this panel exists to compare. The K_M column of
    # results/parameter_coverage_selected.json records the zero; the caption states it.
    bins = np.linspace(0, 3.2, 40)
    ax.hist(spreads["k_cat"], bins=bins, histtype="step", linewidth=1.0,
            color=PLOT_PALETTE[0], label=r"$k_{cat}$")
    # Densities differ by an order of magnitude (3,552 units against 523 classes), so the
    # reference is rescaled to the panel; its SHAPE is the comparison. Drawn as a third
    # STEP OUTLINE rather than as a pale wash: the palette's light fills are the lighter
    # member of a two-level bar and never a mark in their own right, and a translucent area
    # behind two outlines reads as a fourth, softer series. Red rather than the gray slot,
    # since this is the third color of a three-series panel and gray is a secondary.
    counts, edges = np.histogram(classes["range_decades"].to_numpy(), bins=bins)
    scale = max(np.histogram(spreads["k_cat"], bins=bins)[0].max(), 1) / max(
        counts.max(), 1
    )
    ax.step(
        0.5 * (edges[:-1] + edges[1:]), counts * scale, where="mid",
        color=PLOT_PALETTE[1], linewidth=1.0, zorder=0,
        label="within one EC4 class (scaled)",
    )
    ax.axvline(1.0, color="black", linewidth=0.7, linestyle=":",
               label="one order of magnitude")
    ax.set_xlabel("spread across selected predictors,\nper catalytic unit (orders of magnitude)")
    ax.set_ylabel("catalytic units")
    legend_above(ax, ncol=2)
    box(ax)


def panel_within_class(ax: plt.Axes, block: dict, parameter: str) -> dict:
    """Within-class plausibility for the selected predictors, against the EC-median lookup.

    Two readings of the same question, since neither alone is conclusive: the shared band
    scores every pair against one median class width, the own-class band scores each pair
    against its own class and can only be computed where the class holds three measured
    enzymes. The lookup is drawn as a hatched group rather than as a rule, so it is read on
    the same axis and by the same two bars as the predictors it is being compared with.
    """
    selected = select_predictors(parameter)
    groups = [*selected, "ec_median_lookup"]
    shared = [
        block["shared_band"][p]["ec4"] if p in block["shared_band"]
        else block["ec_median_lookup"]["shared"]["ec4"]
        for p in groups
    ]
    own = [
        block["own_class_band"]["predictors"][p]
        if p in block["own_class_band"]["predictors"]
        else block["ec_median_lookup"]["own_class"]
        for p in groups
    ]
    positions = np.arange(len(groups))
    width = 0.38

    def whisker(cells: list[dict]) -> np.ndarray:
        """Wilson interval as a matplotlib yerr pair, which is data and may sit inside."""
        low = [c["frac"] - c["ci95"][0] for c in cells]
        high = [c["ci95"][1] - c["frac"] for c in cells]
        return np.array([low, high])

    # SEMANTIC COLOR: the two band types are purple and red here and nowhere else in this
    # figure, which keeps orange for k_cat in panels d and e and stops the page from
    # reading as one hue.
    for offset, cells, color, name in [
        (-width / 2, shared, PLOT_PALETTE[2], "inside the shared EC4 band"),
        (+width / 2, own, PLOT_PALETTE[1], "inside its own class band"),
    ]:
        hatches = ["" if g != "ec_median_lookup" else HATCH for g in groups]
        for k, (cell, hatch) in enumerate(zip(cells, hatches)):
            ax.bar(positions[k] + offset, cell["frac"], width, color=color,
                   edgecolor="black", linewidth=0.4, hatch=hatch,
                   label=name if k == 0 else None)
        ax.errorbar(positions + offset, [c["frac"] for c in cells],
                    yerr=whisker(cells), fmt="none", ecolor="black", elinewidth=0.5,
                    capsize=1.2, capthick=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [DISPLAY.get(g, "EC-median\nlookup") for g in groups], rotation=25, ha="right"
    )
    symbol = r"$k_{cat}$" if parameter == "k_cat" else r"$K_M$"
    ax.set_ylabel(f"{symbol} predictions inside\nthe class spread")
    tenths(ax)
    handles, names = ax.get_legend_handles_labels()
    handles.append(swatch(PLOT_PALETTE[0], HATCH))
    names.append("lookup, knows only the EC")
    legend_above(ax, ncol=1, handles=handles, names=names)
    box(ax)
    return {
        "selected": selected,
        "lookup_statistic": lookup_plausibility(block),
        "statistic": {p: plausibility(block, p) for p in block["shared_band"]},
    }


def panel_f(ax: plt.Axes, report: dict) -> None:
    """Raw against flux-weighted coverage, the reframing panel."""
    flux = report["flux_weighted"]
    ladder = {s["step"]: s for s in report["delta_g_ladder"]}
    groups = ["GEM shipped", "recommended union"]
    raw = [
        ladder["shipped_only"]["frac_reactions"],
        ladder["union_plus_tight_surrogate"]["frac_reactions"],
    ]
    weighted = [
        flux["frac_flux_carrying_priced_shipped"],
        flux["frac_flux_carrying_priced_recommended"],
    ]
    positions = np.arange(len(groups))
    width = 0.36
    # One quantity at two scopes, so both bars take the SAME palette slot and the narrower
    # scope is a solid black hatch on it. The pale fill used to carry the first bar, which
    # inverts the palette rule: the light fills are the lighter member of a two-level bar,
    # never the primary mark.
    # SEMANTIC COLOR: a bar takes the color panel a gives its SOURCE, shipped orange and
    # the recommended union purple, and the narrower flux-carrying scope is a solid black
    # hatch on the same color. Drawing both sources in one hue made this panel a block of
    # orange and encoded nothing.
    source_colors = [PLOT_PALETTE[0], PLOT_PALETTE[2]]
    ax.bar(positions - width / 2, raw, width, color=source_colors,
           edgecolor="black", linewidth=0.4)
    ax.bar(positions + width / 2, weighted, width, color=source_colors,
           edgecolor="black", linewidth=0.4, hatch=HATCH)
    legend_above(
        ax, ncol=1,
        handles=[swatch("#FFFFFF"), swatch("#FFFFFF", HATCH)],
        names=["all 4,131 reactions",
               f"{flux['n_reactions_with_nonzero_fva_width']:,} that carry flux"],
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(groups)
    ax.set_ylabel("reactions priced")
    tenths(ax)
    box(ax)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-timestamp", action="store_true")
    args = parser.parse_args()

    apply_paper_style()
    plt.rcParams.update({"xtick.major.width": 0.5, "ytick.major.width": 0.5,
                         "hatch.linewidth": 0.4})
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = "" if args.no_timestamp else f"_{timestamp()}"

    with open(osp.join(RESULTS, "parameter_coverage.json")) as handle:
        report = json.load(handle)
    with open(osp.join(RESULTS, "unresolved_metabolites.json")) as handle:
        unresolved = json.load(handle)
    with open(osp.join(RESULTS, "within_class_plausibility.json")) as handle:
        plausible = json.load(handle)
    measured = plausible["measured_source_coverage"]
    surrogates = pd.read_parquet(osp.join(THERMO_DIR, "metabolite_dgf_surrogate.parquet"))

    selected = {
        parameter: select_predictors(parameter)
        for parameter in ("k_cat", "K_M")
    }
    spreads = {p: unit_spread(p, selected[p]) for p in ("k_cat", "K_M")}
    # The all-predictor reference the caption compares against, read from the consensus
    # tables rather than recomputed, so it is the same quantity panel e used to draw.
    consensus = {
        "k_cat": pd.read_parquet(osp.join(CONSENSUS_DIR, "k_cat_consensus.parquet")),
        "K_M": pd.read_parquet(osp.join(CONSENSUS_DIR, "K_M_consensus.parquet")),
    }

    # Four rows of two. At two rows of three each panel was 55 by 40 mm and the figure read
    # as six thumbnails; at 88 mm wide the bars and histograms are legible at the same 6 pt.
    #
    # 148 mm, not the 170 mm ceiling. The ceiling is what a figure may occupy on its own;
    # this one is placed with [H] above a caption that runs 20 lines, and at 168 mm the two
    # together overran the text block by 27 mm. Nothing errors when that happens, the
    # content is simply pushed past the bottom margin, so the height is set against the
    # figure PLUS its caption rather than against the ceiling alone. 37 mm a row still
    # leaves the 6 pt type and the rotated tick labels legible; the last few mm were measured
    # off against the overfull-vbox warning, which is the only signal this has a budget.
    fig, axes = plt.subplots(
        4, 2, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(156)), dpi=300
    )
    panel_a(axes[0, 0], report)
    panel_b(axes[0, 1], surrogates, report["surrogate"]["tight_spread_threshold_kj"])
    panel_c(axes[1, 0], unresolved)
    panel_d(axes[1, 1], measured)
    panel_e(axes[2, 0], spreads, pd.read_csv(osp.join(RESULTS, "ec_class_spread_k_cat.csv")))
    panel_f(axes[2, 1], report)
    chosen = {
        "k_cat": panel_within_class(axes[3, 0], plausible["parameters"]["k_cat"], "k_cat"),
        "K_M": panel_within_class(axes[3, 1], plausible["parameters"]["K_M"], "K_M"),
    }

    fig.tight_layout(w_pad=2.6, h_pad=2.2)

    # The letter goes at the OUTER top-left of each panel, measured from the panel's own
    # tight bounding box rather than placed at a fixed offset in axes coordinates. The fixed
    # offset was (-0.14, 1.15), which is 12 mm left of an 88 mm axes: fine at 47 mm a row and
    # a collision at 37 mm, where a two-line y-label reaches the top of the axes and the
    # letter lands on the end of it. Nothing errors when that happens. Measuring instead
    # puts the letter clear of the y-label and above the legend at any row height.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inverse = fig.transFigure.inverted()
    for letter, ax in zip("abcdefgh", axes.ravel()):
        extent = ax.get_tightbbox(renderer).transformed(inverse)
        fig.text(extent.x0, extent.y1, letter, fontsize=8, fontweight="bold",
                 ha="left", va="bottom")
    base = f"parameter_coverage{stamp}"
    savefig_true_size_svg(fig, osp.join(OUT_DIR, f"{base}.svg"))
    fig.savefig(osp.join(OUT_DIR, f"{base}.png"), dpi=300)
    plt.close(fig)

    # Every number panels e, g and h show, written back so the caption can be checked
    # against a file rather than against the picture.
    report_out = {
        "selection_rule": (
            "own-class EC4 band, keeping a predictor whose Wilson 95 percent lower bound "
            "exceeds the EC-median lookup's point fraction. One rule, applied by "
            "PredictorSelection.from_study in torchcell/metabolism/parameters.py, which is "
            "the rule the flux layer runs"
        ),
        "predictor_table_version": KINETICS_VERSION,
        "parameters": {
            parameter: {
                "selected": chosen[parameter]["selected"],
                "lookup_statistic": chosen[parameter]["lookup_statistic"],
                "statistic": chosen[parameter]["statistic"],
                "n_units_all_selected": int(spreads[parameter].size),
                "spread_median_decades": float(np.median(spreads[parameter])),
                "spread_p90_decades": float(np.percentile(spreads[parameter], 90)),
                "spread_frac_above_1_decade": float((spreads[parameter] > 1.0).mean()),
                "all_predictors": {
                    "n_units": int(len(consensus[parameter])),
                    "spread_median_decades": float(
                        consensus[parameter]["spread_decades"].median()
                    ),
                    "spread_frac_above_1_decade": float(
                        (consensus[parameter]["spread_decades"] > 1.0).mean()
                    ),
                },
            }
            for parameter in ("k_cat", "K_M")
        },
    }
    with open(osp.join(RESULTS, "parameter_coverage_selected.json"), "w") as handle:
        json.dump(report_out, handle, indent=2)
    print(json.dumps(report_out, indent=2))
    print(f"figure -> {osp.join(OUT_DIR, base)}.svg")


if __name__ == "__main__":
    main()
