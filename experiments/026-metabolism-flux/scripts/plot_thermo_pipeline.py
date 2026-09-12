# experiments/026-metabolism-flux/scripts/plot_thermo_pipeline.py
# [[experiments.026-metabolism-flux.scripts.plot_thermo_pipeline]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/plot_thermo_pipeline.py

r"""eQuilibrator on yeast-GEM: what the pipeline is, what it covers, and where the rest is.

One figure for the whole coverage story, so that the resolution bar chart, the pipeline
funnel and the blocker breakdown are read together rather than three pages apart.

EIGHT PANELS, four rows of two
------------------------------
a. **Resolution by identifier namespace.** Which namespace resolved each metabolite to an
   eQuilibrator compound, in the order the resolver tries them, so the bars are disjoint.
b. **The pipeline as a funnel.** A metabolite becomes a formation energy in stages, each
   losing some: an identifier resolves to a compound-cache entry; the entry carries a group
   decomposition so component contribution runs; the Legendre transform to (pH, I, T, pMg)
   gives :math:`\Delta_f G'^\circ`. A reaction survives only if EVERY participant does.
   Counts are after the three recovery routes (extra namespaces, MetaNetX cross-references,
   InChIKey structure lookup), which is the table the layer reads.
c. **Three routes to a reaction energy.** yeast-GEM ships a reaction column and a
   metabolite column built from different sources; the recomputation is a third. What each
   covers, and what only it covers.
d. **The proton convention was worth 986 reactions.** Drawn as the before-and-after it is,
   because it is the largest single coverage change in this work and came from a bug rather
   than from chemistry.
e. **Why the rest cannot be recovered with a short list.** Reactions freed by resolving the
   top-k blocking metabolites. A flat curve means the gap is diffuse.
f. **What still blocks each unpriced metabolite, and what kind of metabolite it is.** The
   lipid structure writer (``lipid_structure_writer.py``) has since priced 347 of the 409
   in the "no structure anywhere" row, so that share is drawn hatched rather than left as
   a stale bar. Those values are not in the built metabolite table yet: they are all at
   pH 7.2 and need re-transforming per compartment first.
g. **Method ceiling against curation gap.** A placeholder formula (R, X, *) is a ceiling.
   A definite formula is not, and 347 of the 480 definite-formula entries now carry a
   written structure and a price.
h. **The heterologous test case.** The betaxanthin pathway intermediates against the GEM's
   own metabolites: the fraction eQuilibrator prices today. All five pathway compounds
   resolve from the cache by InChIKey prefix; an earlier exact-InChI lookup missed three
   of them because the cache stores the major-microspecies InChI.

Palette rule: plot marks use the line colors; the pale fill colors are never a primary mark.
Where a bar has two levels (covers, and covers exclusively; before, and after), the second
level is a black hatch on the same color, not a lighter fill.

Reads ``results/reaction_thermo_coverage.json``, ``results/unresolved_metabolites.json``,
``results/pathway_thermo_betaxanthin.json``, ``results/lipid_priced.json`` and the
metabolite table under ``$DATA_ROOT``.
Run from the repo root with the torchcell env, after ``audit_reaction_thermo_coverage.py``.
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
    apply_paper_style,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()
ASSET_IMAGES_DIR = cast(str, os.getenv("ASSET_IMAGES_DIR"))
EXPERIMENT_ROOT = cast(str, os.getenv("EXPERIMENT_ROOT"))
DATA_ROOT = cast(str, os.getenv("DATA_ROOT"))
RESULTS = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
OUT_DIR = osp.join(ASSET_IMAGES_DIR, "026-metabolism-flux")
THERMO_DIR = osp.join(DATA_ROOT, "data", "torchcell", "thermo_equilibrator")

#: Reaction coverage before the proton convention was fixed, from the audit run of
#: 2026-09-06 recorded in the section text. Kept as a constant with its provenance rather
#: than recomputed, because reproducing it means reverting the fix.
RECOMPUTED_REACTIONS_BEFORE_PROTON_FIX = 1755

HATCH = "////"


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


def swatch(color: str, hatch: str = "") -> plt.Rectangle:
    return plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="black", linewidth=0.4,
                         hatch=hatch)


def tenths(ax: plt.Axes) -> None:
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(axis="y", which="minor", length=0)
    ax.grid(axis="y", which="both", linewidth=0.3, color="0.85")
    ax.set_axisbelow(True)


def panel_a(ax: plt.Axes, metabolites: pd.DataFrame) -> dict:
    """Which namespace resolved each metabolite, in resolver order."""
    tiers = metabolites["resolution_tier"].fillna("unresolved").value_counts()
    order = [t for t in ["metanetx.chemical", "chebi", "kegg.compound", "bigg.metabolite",
                         "inchi", "unresolved"] if t in tiers.index]
    counts = [int(tiers[t]) for t in order]
    labels = [t.replace(".chemical", "").replace(".compound", "").replace(".metabolite", "")
              for t in order]
    colors = [PLOT_PALETTE[i] for i in range(len(order) - 1)] + [PLOT_PALETTE[5]]
    positions = np.arange(len(order))
    ax.bar(positions, counts, 0.66, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("metabolites")
    box(ax)
    return dict(zip(labels, counts))


def panel_b(ax: plt.Axes, unresolved: dict, report: dict) -> dict:
    """The pipeline as a funnel, metabolites in and reactions out.

    Counts are the BUILT table, the one the layer reads and that panels c and d are
    computed on. The three recovery routes in the audit add 28 metabolites and 46 reactions
    on top of it, and are reported in the summary for the caption rather than drawn, since
    they are not yet built into the table.
    """
    final = unresolved["coverage_by_step"][0]
    recovered = unresolved["coverage_by_step"][-1]
    stages = [
        ("GEM metabolites", unresolved["n_metabolites"], PLOT_PALETTE[0]),
        ("identifier resolved", final["n_metabolites_with_compound"], PLOT_PALETTE[0]),
        ("energy returned", final["n_metabolites_priceable"], PLOT_PALETTE[0]),
        ("GEM reactions", report["n_reactions"], PLOT_PALETTE[1]),
        ("all participants priced",
         final["n_reactions_all_participants_priceable"], PLOT_PALETTE[1]),
    ]
    positions = np.arange(len(stages))
    ax.barh(positions, [s[1] for s in stages], 0.66, color=[s[2] for s in stages],
            edgecolor="black", linewidth=0.4)
    ax.set_yticks(positions)
    ax.set_yticklabels([s[0] for s in stages])
    ax.invert_yaxis()
    ax.set_xlabel("count")
    legend_above(ax, ncol=2, handles=[swatch(PLOT_PALETTE[0]), swatch(PLOT_PALETTE[1])],
                 names=["metabolites", "reactions"])
    box(ax)
    return {
        "drawn_step": final["step"],
        **{s[0].replace("\n", " "): s[1] for s in stages},
        "after_recovery_routes": {
            "step": recovered["step"],
            "n_metabolites_priceable": recovered["n_metabolites_priceable"],
            "n_reactions_all_participants_priceable":
                recovered["n_reactions_all_participants_priceable"],
        },
    }


def panel_c(ax: plt.Axes, report: dict) -> None:
    """Three routes, what each covers, and what only it covers."""
    names = {
        "shipped_metabolite_sum": "GEM metabolite\ncolumn, summed",
        "shipped_reaction_column": "GEM reaction\ncolumn",
        "equilibrator_recomputation": "eQuilibrator\nrecomputation",
    }
    order = list(names)
    exclusive = [report["route_overlap"].get(k, 0) for k in order]
    positions = np.arange(len(order))
    width = 0.38
    ax.bar(positions - width / 2, [report["routes"][k]["n"] for k in order], width,
           color=PLOT_PALETTE[0], edgecolor="black", linewidth=0.4, label="covers")
    ax.bar(positions + width / 2, exclusive, width, color=PLOT_PALETTE[0],
           edgecolor="black", linewidth=0.4, hatch=HATCH,
           label="covers and no other route does")
    ax.axhline(report["n_reactions"], color="black", linewidth=0.7, linestyle="--")
    ax.set_xticks(positions)
    ax.set_xticklabels([names[k] for k in order])
    ax.set_ylabel("reactions")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0], color="black", linewidth=0.7, linestyle="--"))
    labels.append(f"all {report['n_reactions']:,}")
    legend_above(ax, ncol=2, handles=handles, names=labels)
    box(ax)


def panel_d(ax: plt.Axes, report: dict) -> None:
    """One convention, 986 reactions."""
    after = report["routes"]["equilibrator_recomputation"]["n"]
    positions = np.arange(2)
    bars = ax.bar(positions, [RECOMPUTED_REACTIONS_BEFORE_PROTON_FIX, after], 0.6,
                  color=PLOT_PALETTE[1], edgecolor="black", linewidth=0.4)
    bars[0].set_hatch(HATCH)
    ax.axhline(report["n_reactions"], color="black", linewidth=0.7, linestyle="--")
    ax.set_xticks(positions)
    ax.set_xticklabels([r"H$^+$ left unpriced", r"H$^+$ priced at zero"])
    ax.set_ylabel("reactions the\nrecomputation can price")
    ax.set_ylim(0, report["n_reactions"] * 1.02)
    legend_above(
        ax, ncol=3,
        handles=[swatch(PLOT_PALETTE[1], HATCH), swatch(PLOT_PALETTE[1]),
                 plt.Line2D([0], [0], color="black", linewidth=0.7, linestyle="--")],
        names=["before", "after", f"all {report['n_reactions']:,}"],
    )
    box(ax)


def panel_e(ax: plt.Axes, report: dict) -> None:
    """Is there a short list of metabolites worth resolving? Measured, not assumed."""
    gap = report["recovery_gap"]
    ks = [0, 10, 25, 50]
    freed = [0, gap["reactions_freed_by_top_10_metabolites"],
             gap["reactions_freed_by_top_25_metabolites"],
             gap["reactions_freed_by_top_50_metabolites"]]
    ax.plot(ks, freed, marker="o", markersize=3, color=PLOT_PALETTE[0], linewidth=1.0,
            markeredgecolor="black", markeredgewidth=0.3, label="reactions recovered")
    ax.axhline(gap["n_reactions_shipped_can_price_and_we_cannot"], color="black",
               linewidth=0.7, linestyle="--", label="the whole gap")
    ax.set_xlabel("blocking metabolites resolved, most-blocking first")
    ax.set_ylabel("reactions recovered")
    ax.set_xlim(0, 52)
    ax.set_ylim(0, gap["n_reactions_shipped_can_price_and_we_cannot"] * 1.05)
    legend_above(ax, ncol=2)
    box(ax)


#: yeast-GEM's chemical classes, as the unresolved-metabolite audit labels them, folded to
#: the four kinds a reader needs to tell apart: a lipid species with a definite acyl
#: composition (a structure can be written from its name), a macromolecule-linked or
#: polymeric entry (no single structure exists), a generic pseudo-metabolite (a stoichiometric
#: placeholder such as "sulfur donor"), and everything else.
COARSE_CLASS = {
    "phosphatidylinositol_PI": "lipid, defined acyl chains",
    "phosphatidylethanolamine_PE": "lipid, defined acyl chains",
    "phosphatidylcholine_PC": "lipid, defined acyl chains",
    "phosphatidylglycerol_PG": "lipid, defined acyl chains",
    "phosphatidate_PA": "lipid, defined acyl chains",
    "cardiolipin_CL": "lipid, defined acyl chains",
    "triacylglycerol_TAG": "lipid, defined acyl chains",
    "sphingolipid": "lipid, defined acyl chains",
    "sterol_ester": "lipid, defined acyl chains",
    "acyl_CoA": "lipid, defined acyl chains",
    "fatty_acid_acyl_chain": "lipid, defined acyl chains",
    "tRNA_linked": "tRNA-, protein-linked, glycan",
    "protein_linked": "tRNA-, protein-linked, glycan",
    "glycan": "tRNA-, protein-linked, glycan",
    "generic_pseudo_metabolite": "generic pseudo-metabolite",
    "other": "other (ions, cofactors, small)",
}
COARSE_ORDER = [
    "lipid, defined acyl chains",
    "tRNA-, protein-linked, glycan",
    "generic pseudo-metabolite",
    "other (ions, cofactors, small)",
]
#: One LINE each. Five two-line labels in a 33 mm axis overlap, and the full wording of
#: each blocker is in the caption.
BLOCKER_NAMES = {
    "no_usable_structure_available": "no structure anywhere",
    "placeholder_formula_no_definite_structure": "placeholder formula",
    "structure_known_but_no_group_decomposition": "no group decomposition",
    "structure_absent_from_cache": "structure not in cache",
    "only_composition_mismatched_candidates": "composition mismatch",
}


def lipid_priced_routes(priced: dict) -> dict[str, str]:
    """Metabolite id to the route that priced it, from the lipid structure writer.

    One written structure serves every compartment copy of the same chemical, so the
    record's ``met_ids`` list is what maps onto the audit's per-metabolite rows.
    """
    return {
        met_id: record["route"]
        for record in priced["records"]
        for met_id in record["met_ids"]
    }


def remaining_unpriced(unresolved: dict, lipid_routes: dict[str, str]) -> pd.DataFrame:
    """The metabolites still unpriced after every recovery route, one row each.

    ``lipid_priced`` marks the ones the lipid structure writer has since priced. They are
    still rows of this table: the written values sit in ``results/lipid_priced.json`` at a
    single pH and are not in the built metabolite table the other panels are computed on.
    """
    rows = [m for m in unresolved["metabolites_unpriced_at_baseline"]
            if not m.get("recovered_priceable")]
    frame = pd.DataFrame(rows)
    frame["coarse"] = frame["chemical_class"].map(COARSE_CLASS)
    if frame["coarse"].isna().any():
        missing = sorted(frame.loc[frame["coarse"].isna(), "chemical_class"].unique())
        raise ValueError(f"chemical classes without a coarse label: {missing}")
    frame["lipid_priced"] = frame["met_id"].isin(lipid_routes)
    covered = set(frame["met_id"]) & set(lipid_routes)
    if len(covered) != len(lipid_routes):
        raise ValueError(
            f"{len(lipid_routes) - len(covered)} priced lipid rows are not in the "
            "unpriced set this figure draws"
        )
    return frame


def panel_f(ax: plt.Axes, remaining: pd.DataFrame, n_metabolites: int) -> dict:
    """What still blocks each unpriced metabolite, and what kind of metabolite it is.

    Each class segment is drawn in two passes, the hatched one being the share the lipid
    structure writer has since priced. Drawing the classes at their audit totals and
    hatching inside them keeps the blocker taxonomy the audit measured while showing what
    is no longer open, rather than silently leaving a bar that is mostly closed.
    """
    order = list(BLOCKER_NAMES)
    grouped = remaining.groupby(["final_blocker", "coarse", "lipid_priced"]).size()
    positions = np.arange(len(order))
    left = np.zeros(len(order))
    colors = [PLOT_PALETTE[0], PLOT_PALETTE[1], PLOT_PALETTE[2], PLOT_PALETTE[5]]
    for coarse, color in zip(COARSE_ORDER, colors):
        for is_priced, hatch in ((True, HATCH), (False, "")):
            values = np.array([
                int(grouped.get((blocker, coarse, is_priced), 0)) for blocker in order
            ], dtype=float)
            ax.barh(positions, values, 0.66, left=left, color=color, edgecolor="black",
                    linewidth=0.4, hatch=hatch)
            left += values
    ax.set_yticks(positions)
    ax.set_yticklabels([BLOCKER_NAMES[k] for k in order])
    ax.invert_yaxis()
    ax.set_xlabel(f"metabolites unpriced after the audit "
                  f"({len(remaining):,} of {n_metabolites:,})")
    legend_above(
        ax, ncol=2,
        handles=[swatch(c) for c in colors] + [swatch(PLOT_PALETTE[0], HATCH)],
        names=COARSE_ORDER + ["priced since, from a written structure"],
    )
    box(ax)
    return {
        "n_unpriced_after_audit": int(len(remaining)),
        "n_priced_by_written_lipid_structures": int(remaining["lipid_priced"].sum()),
        "by_blocker": {
            blocker: {
                "n": int((remaining["final_blocker"] == blocker).sum()),
                "n_priced_since": int(
                    ((remaining["final_blocker"] == blocker)
                     & remaining["lipid_priced"]).sum()
                ),
            }
            for blocker in order
        },
    }


def panel_g(ax: plt.Axes, remaining: pd.DataFrame, unresolved: dict) -> dict:
    """Ceiling against curation: how much of the gap is the method, how much is us.

    The curation half is no longer a projection. It is split at the line the lipid
    structure writer actually reached, so the two counts are exact rather than the
    composition probe's proportional estimate; that estimate is kept in the returned
    summary for the caption.
    """
    probe = unresolved["composition_probe_for_unpriced"]
    placeholder = int(remaining["has_placeholder_formula"].sum())
    definite = int(len(remaining) - placeholder)
    written = int((remaining["lipid_priced"] & ~remaining["has_placeholder_formula"]).sum())
    bars = [
        ("placeholder\nformula", placeholder, PLOT_PALETTE[1]),
        ("definite formula,\nstructure written\nand priced", written, PLOT_PALETTE[0]),
        # A REAL palette color, not the pale fill. PLOT_PALETTE_FILL is the lighter member
        # of a two-level bar and never a mark in its own right, and drawn as one this bar
        # read as no color at all. Purple also keeps the panel off two shades of orange.
        ("definite formula,\nnot yet written", definite - written, PLOT_PALETTE[2]),
    ]
    positions = np.arange(len(bars))
    ax.bar(positions, [b[1] for b in bars], 0.6, color=[b[2] for b in bars],
           edgecolor="black", linewidth=0.4)
    ax.set_xticks(positions)
    ax.set_xticklabels([b[0] for b in bars])
    ax.set_ylabel("metabolites unpriced\nafter the audit")
    legend_above(
        ax, ncol=2,
        handles=[swatch(PLOT_PALETTE[1]), swatch(PLOT_PALETTE[0]),
                 swatch(PLOT_PALETTE[2])],
        names=["method ceiling", "curation gap, closed", "curation gap, open"],
    )
    box(ax)
    return {
        "placeholder": placeholder,
        "definite": definite,
        "definite_written_and_priced": written,
        "definite_not_yet_written": definite - written,
        "composition_probe": {
            "n_unpriced_with_definite_formula":
                int(probe["n_unpriced_with_definite_formula"]),
            "n_composition_decomposable_in_cache":
                int(probe["n_metabolites_whose_composition_is_decomposable_in_cache"]),
        },
    }


def panel_h(ax: plt.Axes, unresolved: dict, pathway: dict) -> dict:
    """The heterologous pathway against the GEM's own metabolites."""
    final = unresolved["coverage_by_step"][0]
    gem_priced = final["n_metabolites_priceable"]
    gem_total = unresolved["n_metabolites"]
    path_priced = sum(1 for c in pathway["compounds"] if c["dgf_prime_kj_per_mol"] is not None)
    path_total = pathway["n_compounds"]
    groups = [(f"yeast-GEM\n({gem_total:,} metabolites)", gem_priced / gem_total),
              (f"betaxanthin pathway\n({path_total} intermediates)", path_priced / path_total)]
    positions = np.arange(len(groups))
    ax.bar(positions, [g[1] for g in groups], 0.55, color=PLOT_PALETTE[0], edgecolor="black",
           linewidth=0.4, label="priced")
    ax.bar(positions, [1 - g[1] for g in groups], 0.55, bottom=[g[1] for g in groups],
           color=PLOT_PALETTE[0], edgecolor="black", linewidth=0.4, hatch=HATCH,
           label="not priced")
    ax.set_xticks(positions)
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("fraction of metabolites\nwith a formation energy")
    tenths(ax)
    legend_above(ax, ncol=2)
    box(ax)
    return {
        "gem_priced": gem_priced, "gem_total": gem_total,
        "pathway_priced": path_priced, "pathway_total": path_total,
        "pathway_unpriced": [
            {"name": c["name"], "blocker": c["blocker"]}
            for c in pathway["compounds"] if c["dgf_prime_kj_per_mol"] is None
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-timestamp", action="store_true")
    args = parser.parse_args()

    apply_paper_style()
    plt.rcParams.update({"xtick.major.width": 0.5, "ytick.major.width": 0.5,
                         "hatch.linewidth": 0.4})
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = "" if args.no_timestamp else f"_{timestamp()}"

    with open(osp.join(RESULTS, "reaction_thermo_coverage.json")) as handle:
        report = json.load(handle)
    with open(osp.join(RESULTS, "unresolved_metabolites.json")) as handle:
        unresolved = json.load(handle)
    with open(osp.join(RESULTS, "pathway_thermo_betaxanthin.json")) as handle:
        pathway = json.load(handle)
    with open(osp.join(RESULTS, "lipid_priced.json")) as handle:
        lipid_priced = json.load(handle)
    metabolites = pd.read_parquet(osp.join(THERMO_DIR, "metabolite_dgf_prime.parquet"))
    lipid_routes = lipid_priced_routes(lipid_priced)
    remaining = remaining_unpriced(unresolved, lipid_routes)

    fig, axes = plt.subplots(
        4, 2, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(158)), dpi=300
    )
    summary: dict = {}
    summary["resolution"] = panel_a(axes[0, 0], metabolites)
    summary["funnel"] = panel_b(axes[0, 1], unresolved, report)
    panel_c(axes[1, 0], report)
    panel_d(axes[1, 1], report)
    panel_e(axes[2, 0], report)
    summary["blockers"] = panel_f(axes[2, 1], remaining, unresolved["n_metabolites"])
    summary["ceiling"] = panel_g(axes[3, 0], remaining, unresolved)
    summary["pathway"] = panel_h(axes[3, 1], unresolved, pathway)
    summary["lipid_structure_writer"] = {
        "n_metabolite_rows_priced": len(lipid_routes),
        "n_by_route": {
            route: sum(1 for r in lipid_routes.values() if r == route)
            for route in sorted(set(lipid_routes.values()))
        },
        "note": "values sit in results/lipid_priced.json at pH 7.2 and are not in the "
                "built metabolite table panels a to e are computed on",
    }

    # Panels d, g and h carry a TWO-LINE rotated y-label, which reaches further left than
    # the one-line labels do, so the shared -0.14 offset put the letter on top of the label.
    # Nothing errors when that happens; it only shows in the rendered page.
    # Panel f's legend runs to three rows, so its letter has to clear one more row than
    # the two-row legends leave space for.
    letter_dx = {"d": -0.21, "g": -0.21, "h": -0.21}
    letter_dy = {"f": 1.30}
    for letter, ax in zip("abcdefgh", axes.ravel()):
        ax.text(letter_dx.get(letter, -0.14), letter_dy.get(letter, 1.16), letter,
                transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")

    fig.tight_layout(w_pad=2.6, h_pad=2.4)
    base = f"thermo_pipeline{stamp}"
    savefig_true_size_svg(fig, osp.join(OUT_DIR, f"{base}.svg"))
    fig.savefig(osp.join(OUT_DIR, f"{base}.png"), dpi=300)
    plt.close(fig)
    with open(osp.join(RESULTS, "thermo_pipeline_figure.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"figure -> {osp.join(OUT_DIR, base)}.svg")


if __name__ == "__main__":
    main()
