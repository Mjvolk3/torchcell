# experiments/026-metabolism-flux/scripts/plot_pka_outside_equilibrator.py
# [[experiments.026-metabolism-flux.scripts.plot_pka_outside_equilibrator]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/plot_pka_outside_equilibrator.py

r"""What the external pKa route was actually spent on, and what a ladder is worth.

``plot_pka_method_selection.py`` draws the benchmark that CHOSE the method. This draws
what the chosen method was then asked to do on yeast-GEM 9.0.2: which species needed a
pKa ladder eQuilibrator could not supply, which member of the cascade answered for each,
how long the ladders are, and how large the two energy shifts a ladder controls are.

``a``
    The 55 targets by where they came from, partitioned by outcome. A target is one
    distinct structure; several yeast-GEM species can share one.
``b``
    Which cascade member answered. The cascade is Uni-pKa on the simple enumeration
    template, then MolGpKa, then pKasolver-lite, and it degenerates to its first member
    on this population.
``c``
    Ladder length, the number of constants inside (0, 14) the cascade returned.
``d``
    The Legendre-transform error a ladder carries, as the p90 of
    ``|dGf'0(predicted) - dGf'0(cache)|`` at each of the eight working compartment pHs,
    over the 2,100 held-out benchmark compounds. This is the quantity the selection rule
    reads, and the reason it reads the worst pH rather than the mean.
``e``
    The reference-microspecies shift, which is larger than the ladder error by three
    orders of magnitude: the signed change in each target's transformed formation energy
    when the pH 7 major microspecies replaces the neutral parent the writer emitted.

Reads ``results/external_pka_applied.json``, ``results/pka_method_selection.json`` and
``results/pka_method_selection_figure.json``. Numbers are written back to
``results/pka_outside_equilibrator_figure.json``.

Run::

    PYTHONPATH=<repo> ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/plot_pka_outside_equilibrator.py
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

APPLIED_JSON = osp.join(RESULTS, "external_pka_applied.json")
DECISION_JSON = osp.join(RESULTS, "pka_method_selection.json")
SELECTION_FIGURE_JSON = osp.join(RESULTS, "pka_method_selection_figure.json")
PIPELINE_JSON = osp.join(RESULTS, "thermo_pipeline_yeast9.json")
FIGURE_JSON = osp.join(RESULTS, "pka_outside_equilibrator_figure.json")

SOURCE_LABEL = {
    "lipid_external_pka": "written lipid",
    "tier_a_pilot": "pKa-only target",
    "acyl_transfer_writer": "acyl transfer product",
}
OUTCOME_LABEL = {
    "priced": "priced",
    "priced_no_ionizable_center": "priced, no ionizable center",
    "compound_creation_failed": "compound creation failed",
}
MEMBER_LABEL = {
    "unipka-simple-template": "Uni-pKa, simple template",
    "molgpka": "MolGpKa",
    "pkasolver": "pKasolver-lite",
    "": "no member needed",
}


class FigureNumbers(BaseModel):
    """Every number the figure draws."""

    model_config = ConfigDict(extra="forbid")

    built_at: str
    applied_json: str
    decision_json: str
    selection_figure_json: str
    figure_svg: str
    figure_png: str
    method: str
    cascade_members: list[str]
    n_targets: int
    n_priced: int
    n_metabolite_rows_priced: int
    n_pipeline_metabolites_on_this_route: int
    targets_by_source_and_outcome: dict[str, dict[str, int]]
    served_by_member: dict[str, int]
    ladder_lengths: dict[str, int]
    working_ph: list[float]
    cascade_p90_by_ph: list[float]
    convention_shift_kj_per_mol: dict[str, float]


def _read(path: str) -> dict[str, object]:
    with open(path) as handle:
        return cast(dict[str, object], json.load(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-timestamp", action="store_true")
    args = parser.parse_args()
    stamp = "" if args.no_timestamp else f"_{timestamp()}"

    apply_paper_style()
    applied = _read(APPLIED_JSON)
    decision = _read(DECISION_JSON)
    selection = _read(SELECTION_FIGURE_JSON)
    pipeline = _read(PIPELINE_JSON)

    records = cast(list[dict[str, object]], applied["records"])
    outcome_order = ["priced", "priced_no_ionizable_center", "compound_creation_failed"]
    present = {cast(str, r["outcome"]) for r in records}
    outcomes = [o for o in outcome_order if o in present]
    sources = sorted({cast(str, r["source"]) for r in records})
    by_source_outcome = {
        source: {
            outcome: sum(
                1 for r in records if r["source"] == source and r["outcome"] == outcome
            )
            for outcome in outcomes
        }
        for source in sources
    }
    served = Counter((cast(list[str], r["members_used"]) or [""])[0] for r in records)
    lengths = Counter(len(cast(list[float], r["pkas"])) for r in records)
    shifts = np.array(
        [
            float(cast(float, r["change_kj_per_mol"]))
            for r in records
            if r["change_kj_per_mol"] is not None
        ]
    )

    panels = cast(list[dict[str, object]], selection["panels"])
    per_ph = next(p for p in panels if p["panel"] == "b")
    values = cast(dict[str, object], per_ph["values"])
    working_ph = [float(v) for v in cast(list[float], values["working_ph"])]
    cascade_p90 = [
        float(v)
        for v in cast(dict[str, list[float]], values["p90_by_method"])["cascade_top3"]
    ]

    route_counts = cast(
        dict[str, int],
        cast(dict[str, object], pipeline["metabolite_counts"])["by_route"],
    )

    width = PANEL_WIDTHS_MM["full"]
    fig = plt.figure(figsize=(mm_to_in(width), mm_to_in(100.0)))
    grid = fig.add_gridspec(
        2, 3, wspace=0.55, hspace=0.85, left=0.075, right=0.985, top=0.88, bottom=0.185
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2])
    ax_d = fig.add_subplot(grid[1, 0:2])
    ax_e = fig.add_subplot(grid[1, 2])

    # a, targets by source, stacked by outcome.
    xs = np.arange(len(sources))
    bottom = np.zeros(len(sources))
    for index, outcome in enumerate(outcomes):
        heights = np.array(
            [by_source_outcome[s][outcome] for s in sources], dtype=float
        )
        ax_a.bar(
            xs,
            heights,
            bottom=bottom,
            width=0.66,
            color=PLOT_PALETTE[index],
            edgecolor="black",
            linewidth=0.4,
            label=OUTCOME_LABEL[outcome],
        )
        bottom += heights
    ax_a.set_xticks(xs)
    ax_a.set_xticklabels([SOURCE_LABEL[s] for s in sources], rotation=25, ha="right")
    ax_a.set_ylabel("targets")
    ax_a.margins(y=0.14)
    ax_a.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        frameon=True,
        fancybox=False,
        edgecolor="black",
        borderpad=0.3,
        handlelength=1.1,
        handletextpad=0.4,
    )

    # b, which cascade member answered.
    member_order = [
        m
        for m in ["unipka-simple-template", "molgpka", "pkasolver", ""]
        if served.get(m, 0) > 0
    ]
    ax_b.bar(
        np.arange(len(member_order)),
        [served[m] for m in member_order],
        width=0.6,
        color=PLOT_PALETTE[0],
        edgecolor="black",
        linewidth=0.4,
    )
    ax_b.set_xticks(np.arange(len(member_order)))
    ax_b.set_xticklabels(
        [MEMBER_LABEL[m] for m in member_order], rotation=25, ha="right"
    )
    ax_b.set_ylabel("targets")
    ax_b.margins(y=0.14)

    # c, ladder length.
    keys = sorted(lengths)
    ax_c.bar(
        keys,
        [lengths[k] for k in keys],
        width=0.6,
        color=PLOT_PALETTE[0],
        edgecolor="black",
        linewidth=0.4,
    )
    ax_c.set_xticks(keys)
    ax_c.set_xlabel("constants in (0, 14)")
    ax_c.set_ylabel("targets")
    ax_c.margins(y=0.14)

    # d, the transform error per working pH.
    ax_d.plot(
        working_ph,
        cascade_p90,
        marker="o",
        markersize=3.0,
        linewidth=0.9,
        color=PLOT_PALETTE[0],
        markerfacecolor=PLOT_PALETTE[0],
        markeredgecolor="black",
        markeredgewidth=0.4,
    )
    ax_d.set_xlabel("compartment pH the metabolite is priced at")
    ax_d.set_ylabel("p90 $|\\Delta\\Delta_f G'^{\\circ}|$ (kJ mol$^{-1}$)")
    ax_d.set_ylim(0, max(cascade_p90) * 1.15)
    ax_d.grid(linewidth=0.3, alpha=0.4)
    ax_d.set_axisbelow(True)

    # e, the reference-microspecies shift.
    ax_e.hist(
        shifts / 1000.0,
        bins=14,
        color=PLOT_PALETTE[1],
        edgecolor="black",
        linewidth=0.4,
    )
    ax_e.set_xlabel("shift on submitting the pH 7\nmicrospecies (MJ mol$^{-1}$)")
    ax_e.set_ylabel("targets")
    ax_e.grid(axis="y", linewidth=0.3, alpha=0.4)
    ax_e.set_axisbelow(True)

    for ax in (ax_a, ax_b, ax_c, ax_d, ax_e):
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_visible(True)

    for letter, (x, y) in zip(
        "abcde",
        [
            (0.005, 0.945),
            (0.345, 0.945),
            (0.675, 0.945),
            (0.005, 0.435),
            (0.675, 0.435),
        ],
    ):
        fig.text(x, y, letter, fontsize=8, fontweight="bold", ha="left", va="bottom")

    base = osp.join(IMAGE_DIR, f"pka_outside_equilibrator{stamp}")
    fig.savefig(f"{base}.png", dpi=300)
    savefig_true_size_svg(fig, f"{base}.svg")
    plt.close(fig)

    numbers = FigureNumbers(
        built_at=datetime.now(UTC).isoformat(),
        applied_json=APPLIED_JSON,
        decision_json=DECISION_JSON,
        selection_figure_json=SELECTION_FIGURE_JSON,
        figure_svg=f"{base}.svg",
        figure_png=f"{base}.png",
        method=cast(str, decision["selected"]),
        cascade_members=cast(list[str], decision["selected_members"]),
        n_targets=int(cast(int, applied["n_targets"])),
        n_priced=int(cast(int, applied["n_priced"])),
        n_metabolite_rows_priced=int(cast(int, applied["n_metabolite_rows_priced"])),
        n_pipeline_metabolites_on_this_route=int(route_counts["external_pka"]),
        targets_by_source_and_outcome=by_source_outcome,
        served_by_member={MEMBER_LABEL[m]: int(served[m]) for m in member_order},
        ladder_lengths={str(k): int(lengths[k]) for k in keys},
        working_ph=working_ph,
        cascade_p90_by_ph=cascade_p90,
        convention_shift_kj_per_mol={
            "n": float(shifts.size),
            "median": float(np.median(shifts)),
            "min": float(shifts.min()),
            "max": float(shifts.max()),
            "median_absolute": float(np.median(np.abs(shifts))),
        },
    )
    with open(FIGURE_JSON, "w") as handle:
        handle.write(numbers.model_dump_json(indent=1))
    print(f"wrote {base}.png\nwrote {base}.svg\nwrote {FIGURE_JSON}")


if __name__ == "__main__":
    main()
