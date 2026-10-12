# experiments/026-metabolism-flux/scripts/plot_thermo_recovery_routes.py
# [[experiments.026-metabolism-flux.scripts.plot_thermo_recovery_routes]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/plot_thermo_recovery_routes.py

r"""Route attribution for every formation and reaction energy in the built thermo table.

Six panels answering one question: for each of yeast-GEM 9.0.2's 2,806 metabolites and
4,131 reactions, WHICH route produced the number, and where a number is absent, which
failure class holds it.

``a``
    Metabolites by the route that priced them, one bar per :class:`ThermoRoute`.
``b``
    Metabolites with no value, by :class:`UnpricedReason`.
``c``
    Reactions by route. ``mixed`` is a reaction whose participants came from more than
    one route, so its uncertainty belongs to no single one.
``d``
    Unpriced reactions by the single reason class that blocks them.
``e``
    The ceiling ladder. Each row carries its own denominator in the tick label, because
    the before/after-curation pair exists only over the whole model and the ladder is
    over the 3,847 chemical reactions.
``f``
    Coverage under four denominators, including the flux-carrying restriction and its
    complement, the reactions no medium lets carry flux.

Every count is read from ``results/reaction_thermo_coverage_final.json`` and
``results/thermo_curation_closure.json``; nothing is recomputed here. Numbers the panels
show are written back to ``results/thermo_recovery_routes_figure.json``.

Run::

    PYTHONPATH=<repo> ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/plot_thermo_recovery_routes.py
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
from datetime import UTC, datetime
from typing import cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from pydantic import BaseModel, ConfigDict

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
IMAGE_DIR = osp.join(ASSET_IMAGES_DIR, "026-metabolism-flux")

COVERAGE_JSON = osp.join(RESULTS, "reaction_thermo_coverage_final.json")
CLOSURE_JSON = osp.join(RESULTS, "thermo_curation_closure.json")
FIGURE_JSON = osp.join(RESULTS, "thermo_recovery_routes_figure.json")

#: Evidence class per route. Four classes, in the order the legend draws them, and the
#: same four the LaTeX route tables carry. A surrogate is filed under the estimate it is
#: made of: the number comes from component contribution on another molecule.
EVIDENCE_CLASS: dict[str, str] = {
    "carrier_potential": "measured",
    "literature_table": "tabulated",
    "shipped_table": "tabulated",
    "shipped_reaction_column": "tabulated",
    "cache_accession": "estimated",
    "cache_structure": "estimated",
    "external_pka": "estimated",
    "composition_surrogate": "estimated",
    "repeat_unit_surrogate": "estimated",
    "conserved_moiety": "estimated",
    "cancelled_participants": "estimated",
    "mixed": "estimated",
    "curated_structure": "curated",
}

EVIDENCE_ORDER = ["measured", "tabulated", "estimated", "curated"]
EVIDENCE_COLOR = dict(zip(EVIDENCE_ORDER, PLOT_PALETTE[:4]))
EVIDENCE_LABEL = {
    "measured": "measured",
    "tabulated": "tabulated",
    "estimated": "estimated (component contribution)",
    "curated": "curated",
}

#: Display name per route, short enough for a 6 pt tick label.
ROUTE_LABEL: dict[str, str] = {
    "cache_accession": "cache, by accession",
    "cache_structure": "cache, by written structure",
    "external_pka": "external pKa cascade",
    "shipped_table": "shipped metabolite column",
    "shipped_reaction_column": "shipped reaction column",
    "literature_table": "CODATA ion table",
    "composition_surrogate": "composition surrogate",
    "repeat_unit_surrogate": "repeat-unit surrogate",
    "curated_structure": "curated structure",
    "carrier_potential": "carrier potential",
    "conserved_moiety": "conserved moiety",
    "cancelled_participants": "cancellation",
    "mixed": "mixed routes",
}

#: Display name per failure class.
REASON_LABEL: dict[str, str] = {
    "placeholder_formula": "placeholder formula",
    "pseudo_metabolite": "pseudo-metabolite",
    "group_decomposition_failed": "decomposition refused",
    "no_structure_available": "no structure",
    "placeholder_structure": "placeholder structure",
    "recorded_not_admitted": "recorded, not admitted",
    "group_in_null_space": "group in the null space",
    "mixed_reasons": "more than one reason",
    "non_chemical": "non-chemical reaction",
}


class PanelCounts(BaseModel):
    """One panel's bars, as label to count."""

    model_config = ConfigDict(extra="forbid")

    panel: str
    what: str
    counts: dict[str, int]


class LadderRow(BaseModel):
    """One row of panel e: a count against the denominator it is a fraction of."""

    model_config = ConfigDict(extra="forbid")

    label: str
    n_priced: int
    denominator: int
    fraction: float


class FigureNumbers(BaseModel):
    """Every number the figure draws, written beside it."""

    model_config = ConfigDict(extra="forbid")

    built_at: str
    coverage_json: str
    closure_json: str
    figure_svg: str
    figure_png: str
    n_metabolites: int
    n_metabolites_priced: int
    n_reactions: int
    n_reactions_priced: int
    n_reactions_chemical: int
    n_reactions_chemical_priced: int
    evidence_class_by_route: dict[str, str]
    metabolites_by_evidence_class: dict[str, int]
    reactions_by_evidence_class: dict[str, int]
    panels: list[PanelCounts]
    ladder: list[LadderRow]


def _read_json(path: str) -> dict[str, object]:
    import json

    with open(path) as handle:
        return cast(dict[str, object], json.load(handle))


def _barh(
    ax: Axes,
    counts: dict[str, int],
    label_map: dict[str, str],
    colored_by_route: bool,
    xlabel: str,
) -> None:
    """One horizontal log-axis bar panel, largest at the top."""
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    labels = [label_map.get(key, key) for key, _ in items]
    values = [value for _, value in items]
    if colored_by_route:
        colors = [EVIDENCE_COLOR[EVIDENCE_CLASS[key]] for key, _ in items]
    else:
        colors = [PLOT_PALETTE[5]] * len(items)
    positions = np.arange(len(items))[::-1]
    ax.barh(
        positions, values, height=0.72, color=colors, edgecolor="black", linewidth=0.4
    )
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xscale("log")
    ax.set_xlim(0.8, max(values) * 3.0)
    ax.set_xlabel(xlabel)
    ax.set_ylim(-0.7, len(items) - 0.3)
    ax.grid(axis="x", which="major", linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-timestamp", action="store_true")
    args = parser.parse_args()
    stamp = "" if args.no_timestamp else f"_{timestamp()}"

    apply_paper_style()
    coverage = _read_json(COVERAGE_JSON)
    closure = _read_json(CLOSURE_JSON)

    met_routes = cast(dict[str, int], coverage["metabolite_routes"])
    met_reasons = cast(dict[str, int], coverage["metabolite_reasons"])
    rxn_routes = cast(dict[str, int], coverage["reaction_routes"])
    rxn_blocking = cast(dict[str, int], coverage["reactions_by_sole_blocking_reason"])
    ladder = cast(dict[str, object], coverage["ceiling_ladder"])
    flux = cast(dict[str, float], coverage["flux_carrying"])

    met_priced = {k: v for k, v in met_routes.items() if k != "none"}
    rxn_priced = {k: v for k, v in rxn_routes.items() if k != "none"}

    n_metabolites = int(cast(int, coverage["n_metabolites"]))
    n_reactions = int(cast(int, coverage["n_reactions"]))
    n_chemical = int(cast(int, coverage["n_reactions_chemical"]))
    n_chemical_priced = int(cast(int, coverage["n_reactions_chemical_priced"]))
    n_flux = int(flux["n_flux_carrying"])
    n_flux_priced = int(flux["n_priced"])

    width = PANEL_WIDTHS_MM["full"]
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(mm_to_in(width), mm_to_in(132.0)),
        gridspec_kw={"wspace": 0.62, "hspace": 0.72, "width_ratios": [1.0, 1.0]},
    )
    fig.subplots_adjust(left=0.20, right=0.985, top=0.90, bottom=0.075)

    _barh(axes[0, 0], met_priced, ROUTE_LABEL, True, "metabolites")
    _barh(axes[0, 1], met_reasons, REASON_LABEL, False, "metabolites with no value")
    _barh(axes[1, 0], rxn_priced, ROUTE_LABEL, True, "reactions")
    _barh(axes[1, 1], rxn_blocking, REASON_LABEL, False, "unpriced reactions")

    # e, the ceiling ladder. Each row against its own denominator.
    rungs = cast(list[dict[str, object]], ladder["rungs"])
    denom = int(cast(int, ladder["denominator"]))
    rung_label = {
        "now": "as built (3,847)",
        "structures_and_pka_ladders_resolved": "structures resolved (3,847)",
        "everything_chemically_possible_resolved": "chemically possible (3,847)",
    }
    rows: list[LadderRow] = [
        LadderRow(
            label="before curation (4,131)",
            n_priced=int(cast(int, closure["n_reactions_priced_before"])),
            denominator=n_reactions,
            fraction=int(cast(int, closure["n_reactions_priced_before"])) / n_reactions,
        ),
        LadderRow(
            label="after curation (4,131)",
            n_priced=int(cast(int, closure["n_reactions_priced_after"])),
            denominator=n_reactions,
            fraction=int(cast(int, closure["n_reactions_priced_after"])) / n_reactions,
        ),
    ]
    for rung in rungs:
        name = cast(str, rung["name"])
        rung_priced = int(cast(int, rung["n_reactions_priced"]))
        rows.append(
            LadderRow(
                label=rung_label[name],
                n_priced=rung_priced,
                denominator=denom,
                fraction=rung_priced / denom,
            )
        )

    # A dot plot rather than bars: the interesting spread is 0.914 to 0.925, and a bar
    # drawn on a truncated axis would read as a value four times what it is.
    ax = axes[2, 0]
    positions = np.arange(len(rows))[::-1]
    fractions = [row.fraction for row in rows]
    left = 0.9095
    ax.hlines(positions, left, fractions, color=PLOT_PALETTE[4], linewidth=0.8)
    ax.plot(
        fractions,
        positions,
        linestyle="none",
        marker="o",
        markersize=3.2,
        markerfacecolor=PLOT_PALETTE[4],
        markeredgecolor="black",
        markeredgewidth=0.4,
    )
    ax.set_yticks(positions)
    ax.set_yticklabels([row.label for row in rows])
    ax.set_xlim(left, 0.9285)
    ax.set_xlabel("fraction of that row's denominator priced")
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.grid(axis="x", linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)

    # f, coverage under four denominators.
    ax = axes[2, 1]
    groups = [
        ("all reactions", n_reactions, int(cast(int, coverage["n_reactions_priced"]))),
        ("chemical only", n_chemical, n_chemical_priced),
        ("flux-carrying", n_flux, n_flux_priced),
        (
            "no flux possible",
            n_reactions - n_flux,
            int(cast(int, coverage["n_reactions_priced"])) - n_flux_priced,
        ),
    ]
    xs = np.arange(len(groups))
    priced = np.array([g[2] for g in groups], dtype=float)
    total = np.array([g[1] for g in groups], dtype=float)
    ax.bar(
        xs, priced, width=0.66, color=PLOT_PALETTE[4], edgecolor="black", linewidth=0.4
    )
    ax.bar(
        xs,
        total - priced,
        width=0.66,
        bottom=priced,
        color=PLOT_PALETTE[5],
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_xticks(xs)
    ax.set_xticklabels([g[0] for g in groups], rotation=20, ha="right")
    ax.set_ylabel("reactions")
    ax.grid(axis="y", linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)

    for ax in axes.ravel():
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_visible(True)

    handles = [
        Patch(
            facecolor=EVIDENCE_COLOR[name],
            edgecolor="black",
            linewidth=0.4,
            label=EVIDENCE_LABEL[name],
        )
        for name in EVIDENCE_ORDER
    ] + [
        Patch(
            facecolor=PLOT_PALETTE[4],
            edgecolor="black",
            linewidth=0.4,
            label="priced (e, f)",
        ),
        Patch(
            facecolor=PLOT_PALETTE[5],
            edgecolor="black",
            linewidth=0.4,
            label="no value",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.995),
        ncol=6,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        borderpad=0.35,
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=1.1,
    )
    letters = [
        (0.005, 0.905),
        (0.52, 0.905),
        (0.005, 0.615),
        (0.52, 0.615),
        (0.005, 0.325),
        (0.52, 0.325),
    ]
    for letter, (x, y) in zip("abcdef", letters):
        fig.text(x, y, letter, fontsize=8, fontweight="bold", ha="left", va="bottom")

    base = osp.join(IMAGE_DIR, f"thermo_recovery_routes{stamp}")
    fig.savefig(f"{base}.png", dpi=300)
    savefig_true_size_svg(fig, f"{base}.svg")
    plt.close(fig)

    by_class_met: dict[str, int] = {name: 0 for name in EVIDENCE_ORDER}
    for route, count in met_priced.items():
        by_class_met[EVIDENCE_CLASS[route]] += count
    by_class_rxn: dict[str, int] = {name: 0 for name in EVIDENCE_ORDER}
    for route, count in rxn_priced.items():
        by_class_rxn[EVIDENCE_CLASS[route]] += count

    numbers = FigureNumbers(
        built_at=datetime.now(UTC).isoformat(),
        coverage_json=COVERAGE_JSON,
        closure_json=CLOSURE_JSON,
        figure_svg=f"{base}.svg",
        figure_png=f"{base}.png",
        n_metabolites=n_metabolites,
        n_metabolites_priced=int(cast(int, coverage["n_metabolites_priced"])),
        n_reactions=n_reactions,
        n_reactions_priced=int(cast(int, coverage["n_reactions_priced"])),
        n_reactions_chemical=n_chemical,
        n_reactions_chemical_priced=n_chemical_priced,
        evidence_class_by_route=EVIDENCE_CLASS,
        metabolites_by_evidence_class=by_class_met,
        reactions_by_evidence_class=by_class_rxn,
        panels=[
            PanelCounts(
                panel="a", what="metabolites by pricing route", counts=met_priced
            ),
            PanelCounts(
                panel="b", what="unpriced metabolites by reason", counts=met_reasons
            ),
            PanelCounts(
                panel="c", what="reactions by pricing route", counts=rxn_priced
            ),
            PanelCounts(
                panel="d",
                what="unpriced reactions by their sole blocking reason class",
                counts=rxn_blocking,
            ),
            PanelCounts(
                panel="f",
                what="reactions priced under four denominators",
                counts={g[0]: g[2] for g in groups}
                | {f"{g[0]} total": g[1] for g in groups},
            ),
        ],
        ladder=rows,
    )
    with open(FIGURE_JSON, "w") as handle:
        handle.write(numbers.model_dump_json(indent=1))
    print(f"wrote {base}.png\nwrote {base}.svg\nwrote {FIGURE_JSON}")


if __name__ == "__main__":
    main()
