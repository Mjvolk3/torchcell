# experiments/026-metabolism-flux/scripts/plot_molecule_mapping.py
# [[experiments.026-metabolism-flux.scripts.plot_molecule_mapping]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/plot_molecule_mapping.py

r"""How a yeast-GEM species becomes a molecule an estimator can price.

Between the model and eQuilibrator sit four mappings, and each of them can be silently
wrong. This draws all four and what each cost.

``a``
    Which accession namespace resolved each metabolite, in the order the resolver tries
    them, so the bars are disjoint: a metabolite counted under ChEBI is one whose
    MetaNetX identifier missed.
``b``
    The composition check over every accession-priced species: does the compound the
    accession resolved carry the model's own heavy-atom composition.
``c``
    The lipid structure writer, one bar per stage, counted in distinct names.
``d``
    The same writer counted in metabolite rows, which is the count coverage moves by.
``e``
    The polymer repeat-unit surrogate over its 97 candidates, and the acyl-transfer
    writer over its 3.
``f``
    The 16 curation records by admission, split by whether the record closed anything.

Reads ``results/thermo_pipeline_figure.json``, ``results/thermo_curation_closure.json``,
``results/lipid_names_parsed.json``, ``results/lipid_coverage.json``,
``results/thermo_polymer_repeat_unit.json`` and ``results/thermo_acyl_transfer.json``.
Numbers are written to ``results/molecule_mapping_figure.json``.

Run::

    PYTHONPATH=<repo> ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/plot_molecule_mapping.py
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
from collections import Counter
from datetime import UTC, datetime
from typing import cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from matplotlib.ticker import MaxNLocator
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

PIPELINE_FIGURE_JSON = osp.join(RESULTS, "thermo_pipeline_figure.json")
CLOSURE_JSON = osp.join(RESULTS, "thermo_curation_closure.json")
PARSED_JSON = osp.join(RESULTS, "lipid_names_parsed.json")
LIPID_COVERAGE_JSON = osp.join(RESULTS, "lipid_coverage.json")
POLYMER_JSON = osp.join(RESULTS, "thermo_polymer_repeat_unit.json")
ACYL_JSON = osp.join(RESULTS, "thermo_acyl_transfer.json")
FIGURE_JSON = osp.join(RESULTS, "molecule_mapping_figure.json")

#: The order build_equilibrator_thermo.py tries namespaces in, then the structure route.
NAMESPACE_ORDER = ["metanetx", "chebi", "kegg", "bigg", "inchi", "unresolved"]
NAMESPACE_LABEL = {
    "metanetx": "MetaNetX",
    "chebi": "ChEBI",
    "kegg": "KEGG",
    "bigg": "BiGG",
    "inchi": "InChI structure",
    "unresolved": "none resolved",
}
COMPOSITION_LABEL = {
    "n_match": "composition matches",
    "n_repeat_unit_multiple": "one repeat unit apart",
    "n_other": "a different molecule",
}
ADMISSION_ORDER = ["measured", "tabulated", "provisional", "none"]


class FigureNumbers(BaseModel):
    """Every number the figure draws."""

    model_config = ConfigDict(extra="forbid")

    built_at: str
    sources: dict[str, str]
    figure_svg: str
    figure_png: str
    namespace_resolution: dict[str, int]
    namespace_order_tried: list[str]
    composition_check: dict[str, int]
    lipid_writer: dict[str, int]
    polymer_surrogate: dict[str, int]
    acyl_transfer: dict[str, int]
    curation_by_admission: dict[str, dict[str, int]]


def _read(path: str) -> dict[str, object]:
    with open(path) as handle:
        return cast(dict[str, object], json.load(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-timestamp", action="store_true")
    args = parser.parse_args()
    stamp = "" if args.no_timestamp else f"_{timestamp()}"

    apply_paper_style()
    pipeline_figure = _read(PIPELINE_FIGURE_JSON)
    closure = _read(CLOSURE_JSON)
    parsed = _read(PARSED_JSON)
    lipid_coverage = _read(LIPID_COVERAGE_JSON)
    polymer = _read(POLYMER_JSON)
    acyl = _read(ACYL_JSON)

    resolution = cast(dict[str, int], pipeline_figure["resolution"])
    sweep = cast(dict[str, object], closure["accession_sweep"])
    composition = {key: int(cast(int, sweep[key])) for key in COMPOSITION_LABEL}

    by_route = cast(dict[str, int], lipid_coverage["newly_priced_by_route"])
    lipid = {
        "names read": int(cast(int, parsed["n_distinct_names"])),
        "parsed": int(cast(int, parsed["n_parsed"])),
        "formula ok": int(cast(int, parsed["n_parsed"])),
        "cache compound": len(
            cast(list[float], lipid_coverage["sigma_newly_priced_cache_route"])
        ),
        "created compound": len(
            cast(list[float], lipid_coverage["sigma_newly_priced_external_pka"])
        ),
    }
    lipid_rows = {
        "in scope": int(cast(int, parsed["n_metabolite_rows"])),
        "names parse": int(cast(int, parsed["parsed_rows"])),
        "priced": int(cast(int, lipid_coverage["n_newly_priced_metabolites"])),
        "cache compound": int(by_route["cache_lookup"]),
        "created compound": int(by_route["external_pka"]),
    }
    polymer_outcomes = {
        key: int(value)
        for key, value in cast(dict[str, int], polymer["outcomes"]).items()
    }
    acyl_outcomes = {
        key: int(value) for key, value in cast(dict[str, int], acyl["outcomes"]).items()
    }

    rows = cast(list[dict[str, object]], closure["rows"])
    curation: dict[str, dict[str, int]] = {
        admission: {"prices a species": 0, "reported only": 0}
        for admission in ADMISSION_ORDER
    }
    for row in rows:
        admission = cast(str, row["admission"])
        key = (
            "prices a species"
            if int(cast(int, row["n_metabolites_priced"])) > 0
            or int(cast(int, row["n_reactions_closed"])) > 0
            else "reported only"
        )
        curation[admission][key] += 1

    width = PANEL_WIDTHS_MM["full"]
    fig = plt.figure(figsize=(mm_to_in(width), mm_to_in(104.0)))
    grid = fig.add_gridspec(
        2, 3, wspace=0.60, hspace=1.05, left=0.08, right=0.985, top=0.94, bottom=0.20
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2])
    ax_d = fig.add_subplot(grid[1, 0])
    ax_e = fig.add_subplot(grid[1, 1])
    ax_f = fig.add_subplot(grid[1, 2])

    # a, namespace resolution in the order tried.
    xs = np.arange(len(NAMESPACE_ORDER))
    colors = [PLOT_PALETTE[0]] * (len(NAMESPACE_ORDER) - 1) + [PLOT_PALETTE[5]]
    ax_a.bar(
        xs,
        [resolution[key] for key in NAMESPACE_ORDER],
        width=0.66,
        color=colors,
        edgecolor="black",
        linewidth=0.4,
    )
    ax_a.set_xticks(xs)
    ax_a.set_xticklabels(
        [NAMESPACE_LABEL[key] for key in NAMESPACE_ORDER], rotation=35, ha="right"
    )
    ax_a.set_yscale("log")
    ax_a.set_ylabel("metabolites")
    ax_a.grid(axis="y", linewidth=0.3, alpha=0.4)
    ax_a.set_axisbelow(True)

    # b, the composition check.
    keys = list(COMPOSITION_LABEL)
    xs = np.arange(len(keys))
    ax_b.bar(
        xs,
        [composition[key] for key in keys],
        width=0.6,
        color=[PLOT_PALETTE[0], PLOT_PALETTE[1], PLOT_PALETTE[2]],
        edgecolor="black",
        linewidth=0.4,
    )
    ax_b.set_xticks(xs)
    ax_b.set_xticklabels(
        [COMPOSITION_LABEL[key] for key in keys], rotation=35, ha="right"
    )
    ax_b.set_yscale("log")
    ax_b.set_ylabel("accession-priced species")
    ax_b.grid(axis="y", linewidth=0.3, alpha=0.4)
    ax_b.set_axisbelow(True)

    # c, the lipid writer by stage, on distinct names.
    keys = list(lipid)
    xs = np.arange(len(keys))
    ax_c.bar(
        xs,
        [lipid[key] for key in keys],
        width=0.66,
        color=PLOT_PALETTE[0],
        edgecolor="black",
        linewidth=0.4,
    )
    ax_c.set_xticks(xs)
    ax_c.set_xticklabels(keys, rotation=35, ha="right")
    ax_c.set_ylabel("distinct names")
    ax_c.margins(y=0.14)

    # d, the same writer counted in metabolite rows.
    keys = list(lipid_rows)
    xs = np.arange(len(keys))
    ax_d.bar(
        xs,
        [lipid_rows[key] for key in keys],
        width=0.66,
        color=PLOT_PALETTE[1],
        edgecolor="black",
        linewidth=0.4,
    )
    ax_d.set_xticks(xs)
    ax_d.set_xticklabels(keys, rotation=35, ha="right")
    ax_d.set_ylabel("metabolite rows")
    ax_d.margins(y=0.14)

    # e, the two writers that are not the lipid one.
    labels = [
        "polymer residue\npriced",
        "polymer residue\nrefused",
        "acyl product\ncomposed",
        "acyl product\nrefused",
    ]
    counts = [
        polymer_outcomes.get("repeat_unit_surrogate", 0),
        polymer_outcomes.get("no_hydrolysis_reaction_identifies_a_monomer", 0)
        + polymer_outcomes.get("stoichiometry_not_per_repeat_unit", 0),
        acyl_outcomes.get("composed", 0),
        acyl_outcomes.get("ambiguous_hydroxyl", 0),
    ]
    xs = np.arange(len(labels))
    ax_e.bar(
        xs,
        counts,
        width=0.66,
        color=[PLOT_PALETTE[0], PLOT_PALETTE[5], PLOT_PALETTE[0], PLOT_PALETTE[5]],
        edgecolor="black",
        linewidth=0.4,
    )
    ax_e.set_xticks(xs)
    ax_e.set_xticklabels(labels, rotation=35, ha="right")
    ax_e.set_ylabel("candidates")
    ax_e.margins(y=0.14)

    # f, curation records by admission.
    xs = np.arange(len(ADMISSION_ORDER))
    prices = np.array(
        [curation[a]["prices a species"] for a in ADMISSION_ORDER], dtype=float
    )
    reported = np.array(
        [curation[a]["reported only"] for a in ADMISSION_ORDER], dtype=float
    )
    ax_f.bar(
        xs,
        prices,
        width=0.66,
        color=PLOT_PALETTE[0],
        edgecolor="black",
        linewidth=0.4,
        label="closes something",
    )
    ax_f.bar(
        xs,
        reported,
        bottom=prices,
        width=0.66,
        color=PLOT_PALETTE[5],
        edgecolor="black",
        linewidth=0.4,
        label="reported only",
    )
    ax_f.set_xticks(xs)
    ax_f.set_xticklabels(ADMISSION_ORDER, rotation=35, ha="right")
    ax_f.set_ylabel("curation records")
    ax_f.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_f.margins(y=0.18)
    ax_f.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        frameon=True,
        fancybox=False,
        edgecolor="black",
        borderpad=0.3,
        handlelength=1.1,
        handletextpad=0.4,
    )

    for ax in (ax_a, ax_b, ax_c, ax_d, ax_e, ax_f):
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_visible(True)

    for letter, (x, y) in zip(
        "abcdef",
        [
            (0.005, 0.965),
            (0.345, 0.965),
            (0.675, 0.965),
            (0.005, 0.475),
            (0.345, 0.475),
            (0.675, 0.475),
        ],
    ):
        fig.text(x, y, letter, fontsize=8, fontweight="bold", ha="left", va="bottom")

    base = osp.join(IMAGE_DIR, f"molecule_mapping{stamp}")
    fig.savefig(f"{base}.png", dpi=300)
    savefig_true_size_svg(fig, f"{base}.svg")
    plt.close(fig)

    numbers = FigureNumbers(
        built_at=datetime.now(UTC).isoformat(),
        sources={
            "namespaces": PIPELINE_FIGURE_JSON,
            "composition": CLOSURE_JSON,
            "lipid_names": PARSED_JSON,
            "lipid_coverage": LIPID_COVERAGE_JSON,
            "polymer": POLYMER_JSON,
            "acyl": ACYL_JSON,
        },
        figure_svg=f"{base}.svg",
        figure_png=f"{base}.png",
        namespace_resolution={key: resolution[key] for key in NAMESPACE_ORDER},
        namespace_order_tried=[
            "metanetx.chemical",
            "chebi",
            "kegg.compound",
            "bigg.metabolite",
            "inchi structure",
        ],
        composition_check=composition
        | {"n_checked": int(cast(int, sweep["n_checked"]))},
        lipid_writer=lipid | lipid_rows,
        polymer_surrogate=polymer_outcomes
        | {"n_candidates": int(cast(int, polymer["n_candidates"]))},
        acyl_transfer=acyl_outcomes
        | {
            "n_candidate_reactions": int(cast(int, acyl["n_candidate_reactions"])),
            "n_priced": int(cast(int, acyl["n_priced"])),
        },
        curation_by_admission=curation,
    )
    with open(FIGURE_JSON, "w") as handle:
        handle.write(numbers.model_dump_json(indent=1))
    print(f"wrote {base}.png\nwrote {base}.svg\nwrote {FIGURE_JSON}")
    print(Counter(cast(str, r["admission"]) for r in rows))


if __name__ == "__main__":
    main()
