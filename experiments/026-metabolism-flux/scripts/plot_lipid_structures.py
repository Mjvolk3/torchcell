# experiments/026-metabolism-flux/scripts/plot_lipid_structures.py
# [[experiments.026-metabolism-flux.lipid-structure-writer]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/plot_lipid_structures.py

r"""Three panels on what writing the tier-b lipid structures bought.

Panel a is the funnel, per yeast-GEM chemical class: of the distinct tier-b names, how many
the parser decomposes, how many produce a structure whose formula matches yeast-GEM's, how
many the eQuilibrator group decomposer accepts, and how many come out with an energy. The
five segments partition each class's name count, so a bar's full height is that class's
distinct-name total and every loss is visible as the segment that carries it.

Panel b is reaction coverage before and after, for the three route families
``lipid_coverage.json`` reports plus the flux-carrying restriction.

Panel c is the formation-energy uncertainty of the newly priced lipids against the lipids
the compound cache could already price, which is the question of whether a written
structure buys a number of the same quality as a looked-up one.

Reads ``results/lipid_names_parsed.json``, ``results/lipid_structures.json``,
``results/lipid_priced.json`` and ``results/lipid_coverage.json``. Writes
``$ASSET_IMAGES_DIR/026-metabolism-flux/lipid_structure_writer.svg`` and ``.png``.

Run from the repo root with the torchcell env::

    ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/plot_lipid_structures.py
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
from matplotlib.ticker import MultipleLocator

from torchcell.timestamp import timestamp
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    apply_paper_style,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()
ASSET_IMAGES_DIR = cast(str, os.environ["ASSET_IMAGES_DIR"])
EXPERIMENT_ROOT = cast(str, os.environ["EXPERIMENT_ROOT"])
RESULTS = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
OUT_DIR = osp.join(ASSET_IMAGES_DIR, "026-metabolism-flux")

#: yeast-GEM's class strings are long enough to rotate the x labels past legibility, so
#: they are shortened here and only here; the data keys stay as the audit wrote them.
CLASS_LABEL = {
    "phosphatidylinositol_PI": "PI",
    "cardiolipin_CL": "CL",
    "triacylglycerol_TAG": "TAG",
    "phosphatidylethanolamine_PE": "PE",
    "phosphatidate_PA": "PA",
    "phosphatidylcholine_PC": "PC",
    "phosphatidylglycerol_PG": "PG",
    "acyl_CoA": "acyl-CoA",
    "fatty_acid_acyl_chain": "acyl chain",
    "sphingolipid": "sphingolipid",
    "glycan": "glycan",
    "protein_linked": "protein",
    "sterol_ester": "sterol ester",
    "generic_pseudo_metabolite": "pseudo",
    "other": "other",
}

#: The funnel, priced first so the stack reads bottom-up as the pipeline succeeds. The five
#: keys partition each class's distinct-name count.
FUNNEL = [
    ("priced_cache_lookup", "priced, cache lookup"),
    ("priced_external_pka", "priced, external pKa"),
    ("written_not_priced", "structure written, not priced"),
    ("parsed_not_formula_pass", "parsed, formula mismatch"),
    ("unparsed", "name not parsed"),
]


def load(name: str) -> dict:
    with open(osp.join(RESULTS, name)) as handle:
        return json.load(handle)


def funnel_by_class(parsed: dict, structures: dict, priced: dict) -> dict[str, dict[str, int]]:
    """Each class's distinct names split into the five mutually exclusive funnel stages."""
    counts: dict[str, dict[str, int]] = {}
    for record in parsed["names"]:
        block = counts.setdefault(
            record["chemical_class"], {key: 0 for key, _ in FUNNEL}
        )
        if not record["parsed"]:
            block["unparsed"] += 1
    for record in structures["structures"]:
        if not record["formula_match"]:
            counts[record["chemical_class"]]["parsed_not_formula_pass"] += 1
    route = {r["name"]: (r["outcome"], r["route"]) for r in priced["records"]}
    for record in structures["structures"]:
        if not record["formula_match"]:
            continue
        block = counts[record["chemical_class"]]
        outcome, taken = route.get(record["name"], ("not_attempted", "blocked"))
        if outcome != "priced":
            block["written_not_priced"] += 1
        elif taken == "cache_lookup":
            block["priced_cache_lookup"] += 1
        else:
            block["priced_external_pka"] += 1
    return counts


def panel_a(ax, counts: dict[str, dict[str, int]]) -> None:
    order = sorted(
        counts, key=lambda k: (-sum(counts[k].values()), CLASS_LABEL.get(k, k))
    )
    x = np.arange(len(order))
    bottom = np.zeros(len(order))
    for index, (key, label) in enumerate(FUNNEL):
        heights = np.array([counts[c][key] for c in order], dtype=float)
        ax.bar(
            x,
            heights,
            bottom=bottom,
            width=0.72,
            color=PLOT_PALETTE[index],
            edgecolor="black",
            linewidth=0.4,
            label=label,
        )
        bottom += heights
    ax.set_xticks(x)
    ax.set_xticklabels([CLASS_LABEL.get(c, c) for c in order], rotation=45, ha="right")
    ax.set_ylabel("distinct yeast-GEM names")
    ax.set_xlim(-0.7, len(order) - 0.3)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.02),
        ncol=3,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        handlelength=1.2,
        columnspacing=1.0,
    )


def panel_b(ax, coverage: dict) -> None:
    reactions = coverage["reactions"]
    flux = coverage["flux_carrying"]
    rows = [
        (
            "eQuilibrator\nrecomputation",
            reactions["recomputation"]["frac_reactions"],
            reactions["recomputation_plus_lipids_cache_route"]["frac_reactions"],
            reactions["recomputation_plus_lipids"]["frac_reactions"],
        ),
        (
            "union with\nshipped table",
            reactions["union_shipped_recomputed"]["frac_reactions"],
            reactions["union_plus_lipids_cache_route"]["frac_reactions"],
            reactions["union_plus_lipids"]["frac_reactions"],
        ),
        (
            "recommended\nset",
            reactions["recommended_union_plus_tight_surrogate"]["frac_reactions"],
            reactions["recommended_plus_lipids_cache_route"]["frac_reactions"],
            reactions["recommended_plus_lipids"]["frac_reactions"],
        ),
        (
            "recommended,\nflux-carrying",
            flux["frac_recommended"],
            flux["frac_recommended_plus_lipids_cache_route"],
            flux["frac_recommended_plus_lipids"],
        ),
    ]
    x = np.arange(len(rows))
    width = 0.27
    series = [
        (1, "before", -width),
        (2, "plus cache route", 0.0),
        (3, "plus both routes", width),
    ]
    for column, label, offset in series:
        ax.bar(
            x + offset,
            [r[column] for r in rows],
            width,
            color=PLOT_PALETTE[column - 1],
            edgecolor="black",
            linewidth=0.4,
            label=label,
        )
    # The shipped metabolite column is the number the report is measured against, so it is
    # a reference line rather than a fourth bar.
    ax.axhline(
        reactions["shipped_metabolite_sum"]["frac_reactions"],
        color=PLOT_PALETTE[3],
        linewidth=0.7,
        linestyle="--",
        label="shipped column",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel("reactions with every participant priced")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(axis="y", which="minor", length=0)
    ax.grid(axis="y", which="both", linewidth=0.3, color="0.85")
    ax.set_axisbelow(True)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.02),
        ncol=2,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        handlelength=1.2,
        columnspacing=1.0,
    )


def panel_c(ax, coverage: dict) -> None:
    cached = np.array(coverage["sigma_cache_priced_lipids"], dtype=float)
    by_cache = np.array(coverage["sigma_newly_priced_cache_route"], dtype=float)
    by_external = np.array(coverage["sigma_newly_priced_external_pka"], dtype=float)
    everything = np.concatenate([cached, by_cache, by_external])
    bins = np.linspace(0.0, float(np.ceil(everything.max())), 25)
    ax.hist(
        cached,
        bins=bins,
        density=True,
        color=PLOT_PALETTE[0],
        edgecolor="black",
        linewidth=0.4,
        label=f"lipids the table already prices (n = {cached.size})",
    )
    for values, color, label in (
        (by_cache, PLOT_PALETTE[1], f"written, cache route (n = {by_cache.size})"),
        (by_external, PLOT_PALETTE[2], f"written, external pKa (n = {by_external.size})"),
    ):
        if values.size:
            ax.hist(
                values,
                bins=bins,
                density=True,
                histtype="step",
                color=color,
                linewidth=1.0,
                label=label,
            )
    ax.set_xlabel(r"$\sigma$ on $\Delta_f G'^{\circ}$ (kJ mol$^{-1}$)")
    ax.set_ylabel("density")
    ax.set_xlim(bins[0], bins[-1])
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.02),
        ncol=1,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        handlelength=1.2,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-timestamp", action="store_true", help="write the stable file name"
    )
    args = parser.parse_args()
    stamp = "" if args.no_timestamp else f"_{timestamp()}"

    apply_paper_style()
    os.makedirs(OUT_DIR, exist_ok=True)

    parsed = load("lipid_names_parsed.json")
    structures = load("lipid_structures.json")
    priced = load("lipid_priced.json")
    coverage = load("lipid_coverage.json")

    fig = plt.figure(
        figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(126)), dpi=300
    )
    grid = fig.add_gridspec(
        2, 2, height_ratios=[1.0, 0.92], hspace=0.55, wspace=0.30,
        left=0.070, right=0.985, top=0.905, bottom=0.085,
    )
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])

    panel_a(ax_a, funnel_by_class(parsed, structures, priced))
    panel_b(ax_b, coverage)
    panel_c(ax_c, coverage)

    for letter, ax in zip("abc", (ax_a, ax_b, ax_c)):
        ax.text(
            -0.055 if ax is ax_a else -0.16,
            1.30,
            letter,
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            va="top",
        )

    base = f"lipid_structure_writer{stamp}"
    savefig_true_size_svg(fig, osp.join(OUT_DIR, f"{base}.svg"))
    fig.savefig(osp.join(OUT_DIR, f"{base}.png"), dpi=300)
    plt.close(fig)
    print(f"figure -> {osp.join(OUT_DIR, base)}.svg")
    print(f"figure -> {osp.join(OUT_DIR, base)}.png")


if __name__ == "__main__":
    main()
