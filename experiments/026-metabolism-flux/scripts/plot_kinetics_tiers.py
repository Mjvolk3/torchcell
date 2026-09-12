# experiments/026-metabolism-flux/scripts/plot_kinetics_tiers.py
# [[experiments.026-metabolism-flux.scripts.plot_kinetics_tiers]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/plot_kinetics_tiers.py

r"""Where every turnover number and Michaelis constant in the shipped table came from.

``plot_parameter_coverage.py`` draws what each PREDICTOR covers. This draws what the
resolved table is made of after the four tiers have been applied in order, which is a
different question and the one a reader of the flux layer asks.

``a``
    The four tiers over the 3,728 catalytic units of yeast-GEM 9.0.2, for the table the
    flux layer reads.
``b``
    The same counts under each table version, so the effect of keying the measured tier
    on a substrate match and of verifying every substrate structure is visible rather
    than asserted.
``c``
    The predictor admission rule: the own-class EC4 band fraction per predictor with its
    Wilson 95 percent interval, against the EC-median lookup's point fraction. A
    predictor is admitted only when its lower bound clears the lookup.
``d``
    The 28 substrate rows behind the 18 units still on the organism default, by why no
    open-source tool reaches them.

Reads ``results/kinetic_coverage_closure.json`` and
``results/parameter_coverage_selected.json``, plus
``results/within_class_plausibility.json`` for the interval widths. Numbers are written
to ``results/kinetics_tiers_figure.json``.

Run::

    PYTHONPATH=<repo> ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/plot_kinetics_tiers.py
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
from matplotlib.ticker import MaxNLocator, MultipleLocator
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

CLOSURE_JSON = osp.join(RESULTS, "kinetic_coverage_closure.json")
SELECTED_BASE = "parameter_coverage_selected"
PLAUSIBILITY_BASE = "within_class_plausibility"
FIGURE_JSON = osp.join(RESULTS, "kinetics_tiers_figure.json")

#: The variant the flux layer reads, and the order the others are drawn in.
SHIPPED_VARIANT = "verified_substrate_key_with_homolog"
VARIANT_ORDER = [
    "unverified_accession_key",
    "verified_accession_key",
    "verified_substrate_key",
    SHIPPED_VARIANT,
]
VARIANT_LABEL = {
    "unverified_accession_key": "accession key,\nunverified structures",
    "verified_accession_key": "accession key,\nverified structures",
    "verified_substrate_key": "substrate-matched,\nno homolog tier",
    SHIPPED_VARIANT: "substrate-matched,\nhomolog tier",
}
TIER_ORDER = ["measured", "homolog", "predicted", "default"]
TIER_LABEL = {
    "measured": "measured, this organism",
    "homolog": "measured, same EC, another organism",
    "predicted": "predicted, median of admitted predictors",
    "default": "organism default",
}
PREDICTOR_LABEL = {
    "ezkinetics": "EZKinetics",
    "eitlem": "EITLEM",
    "unikp": "UniKP",
    "dlkcat": "DLKcat",
    "turnup": "TurNuP",
    "deepenzyme": "DeepEnzyme",
    "boost_km": "Boost\\_KM",
    "catpred": "CatPred",
}
REASON_SHORT = {
    "R-group placeholder, the GEM defines the species only up to a substituent": "R-group\nplaceholder",
    "macromolecular species with no formula in the GEM": "macromolecular,\nno formula",
    "no candidate structure in any mirrored catalog for this compound": "no candidate\nstructure",
    "candidate structures exist but none matches the GEM's composition, so the GEM's "
    "formula is a partial or protein-bound species": "composition\nmismatch",
}


class FigureNumbers(BaseModel):
    """Every number the figure draws."""

    model_config = ConfigDict(extra="forbid")

    built_at: str
    closure_json: str
    selected_json: str
    figure_svg: str
    figure_png: str
    n_units: int
    shipped_variant: str
    tier_counts: dict[str, int]
    tier_counts_by_variant: dict[str, dict[str, int]]
    admission: dict[str, dict[str, list[float]]]
    lookup_statistic: dict[str, float]
    admitted: dict[str, list[str]]
    default_rows_by_reason: dict[str, int]
    n_default_units: int
    n_reachable_with_open_source_tools: int


def _read(path: str) -> dict[str, object]:
    with open(path) as handle:
        return cast(dict[str, object], json.load(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-timestamp", action="store_true")
    # Which plausibility study to read. The default name is the live one the resolver
    # and every spawned DataLoader worker resolve against, so a study that scores a new
    # predictor is written as a versioned sibling and selected here by name.
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    stamp = "" if args.no_timestamp else f"_{timestamp()}"

    apply_paper_style()
    closure = _read(CLOSURE_JSON)
    selected = _read(osp.join(RESULTS, f"{SELECTED_BASE}{args.suffix}.json"))
    plausibility = _read(
        osp.join(RESULTS, f"{PLAUSIBILITY_BASE}{args.suffix}.json")
    )

    kcat = cast(
        dict[str, object], cast(dict[str, object], closure["parameters"])["k_cat"]
    )
    n_units = int(cast(int, kcat["n_units"]))
    variants = cast(dict[str, dict[str, object]], kcat["variants"])
    tier_by_variant = {
        name: cast(dict[str, int], variants[name]["tier_counts"])
        for name in VARIANT_ORDER
    }
    shipped = tier_by_variant[SHIPPED_VARIANT]

    parameters = cast(dict[str, dict[str, object]], selected["parameters"])
    lookup = {
        parameter: float(cast(float, parameters[parameter]["lookup_statistic"]))
        for parameter in ("k_cat", "K_M")
    }
    admitted = {
        parameter: cast(list[str], parameters[parameter]["selected"])
        for parameter in ("k_cat", "K_M")
    }
    #: The Wilson interval is the study's own, not recomputed here.
    bands = {
        parameter: cast(
            dict[str, dict[str, object]],
            cast(
                dict[str, dict[str, object]],
                cast(dict[str, dict[str, object]], plausibility["parameters"])[
                    parameter
                ],
            )["own_class_band"]["predictors"],
        )
        for parameter in ("k_cat", "K_M")
    }

    ceiling = cast(dict[str, object], closure["ceiling"])
    remaining = cast(list[dict[str, object]], ceiling["remaining"])
    reasons = Counter(cast(str, row["reason"]) for row in remaining)

    width = PANEL_WIDTHS_MM["full"]
    fig = plt.figure(figsize=(mm_to_in(width), mm_to_in(108.0)))
    grid = fig.add_gridspec(
        2, 2, wspace=0.30, hspace=1.15, left=0.085, right=0.985, top=0.83, bottom=0.20
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    # a, the four tiers of the shipped table.
    xs = np.arange(len(TIER_ORDER))
    ax_a.bar(
        xs,
        [shipped[tier] for tier in TIER_ORDER],
        width=0.66,
        color=[PLOT_PALETTE[i] for i in range(len(TIER_ORDER))],
        edgecolor="black",
        linewidth=0.4,
    )
    ax_a.set_xticks(xs)
    ax_a.set_xticklabels(TIER_ORDER, rotation=25, ha="right")
    ax_a.set_yscale("log")
    ax_a.set_ylabel("catalytic units")
    ax_a.grid(axis="y", linewidth=0.3, alpha=0.4)
    ax_a.set_axisbelow(True)

    # b, the same counts across table versions, stacked.
    xs = np.arange(len(VARIANT_ORDER))
    bottom = np.zeros(len(VARIANT_ORDER))
    for index, tier in enumerate(TIER_ORDER):
        heights = np.array(
            [tier_by_variant[name][tier] for name in VARIANT_ORDER], dtype=float
        )
        ax_b.bar(
            xs,
            heights,
            bottom=bottom,
            width=0.66,
            color=PLOT_PALETTE[index],
            edgecolor="black",
            linewidth=0.4,
            label=TIER_LABEL[tier],
        )
        bottom += heights
    ax_b.set_xticks(xs)
    ax_b.set_xticklabels(
        [VARIANT_LABEL[name] for name in VARIANT_ORDER], rotation=25, ha="right"
    )
    ax_b.set_ylabel("catalytic units")

    # c, the admission rule.
    offsets = {"k_cat": -0.18, "K_M": 0.18}
    admission_out: dict[str, dict[str, list[float]]] = {}
    names: list[str] = []
    for parameter in ("k_cat", "K_M"):
        statistic = cast(dict[str, float], parameters[parameter]["statistic"])
        for name in statistic:
            if name not in names:
                names.append(name)
    names.sort(
        key=lambda n: (
            -max(
                float(cast(dict[str, float], parameters[p]["statistic"]).get(n, 0.0))
                for p in ("k_cat", "K_M")
            )
        )
    )
    positions = {name: index for index, name in enumerate(names)}
    for index, parameter in enumerate(("k_cat", "K_M")):
        statistic = cast(dict[str, float], parameters[parameter]["statistic"])
        xs_p: list[float] = []
        ys_p: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        admission_out[parameter] = {}
        for name, value in statistic.items():
            low, high = (
                float(v) for v in cast(list[float], bands[parameter][name]["ci95"])
            )
            xs_p.append(positions[name] + offsets[parameter])
            ys_p.append(value)
            lows.append(value - low)
            highs.append(high - value)
            admission_out[parameter][name] = [value, low, high]
        ax_c.errorbar(
            xs_p,
            ys_p,
            yerr=[lows, highs],
            fmt="o",
            markersize=3.0,
            elinewidth=0.7,
            capsize=1.6,
            capthick=0.7,
            color=PLOT_PALETTE[index],
            markerfacecolor=PLOT_PALETTE[index],
            markeredgecolor="black",
            markeredgewidth=0.4,
            label="$k_{cat}$" if parameter == "k_cat" else "$K_M$",
        )
        ax_c.axhline(
            lookup[parameter], color=PLOT_PALETTE[index], linestyle="--", linewidth=0.7
        )
    ax_c.set_xticks(list(positions.values()))
    ax_c.set_xticklabels(
        [PREDICTOR_LABEL[name].replace("\\_", "_") for name in positions],
        rotation=25,
        ha="right",
    )
    ax_c.set_ylabel("own-class EC4 band fraction")
    ax_c.set_ylim(0.2, 1.0)
    ax_c.yaxis.set_major_locator(MultipleLocator(0.2))
    ax_c.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax_c.tick_params(axis="y", which="minor", length=0)
    ax_c.grid(axis="y", which="both", linewidth=0.3, alpha=0.4)
    ax_c.set_axisbelow(True)

    # d, why the defaults are defaults.
    keys = [key for key, _ in reasons.most_common()]
    xs = np.arange(len(keys))
    # Semantic color: these rows are the reason a unit falls to the organism default, so
    # they carry the default tier's color from panels a and b rather than a neutral gray.
    ax_d.bar(
        xs,
        [reasons[key] for key in keys],
        width=0.6,
        color=PLOT_PALETTE[TIER_ORDER.index("default")],
        edgecolor="black",
        linewidth=0.4,
    )
    ax_d.set_xticks(xs)
    ax_d.set_xticklabels([REASON_SHORT[key] for key in keys], rotation=25, ha="right")
    ax_d.set_ylabel("substrate rows")
    ax_d.margins(y=0.14)
    ax_d.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_d.grid(axis="y", linewidth=0.3, alpha=0.4)
    ax_d.set_axisbelow(True)

    for ax in (ax_a, ax_b, ax_c, ax_d):
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_visible(True)

    handles, labels = ax_b.get_legend_handles_labels()
    fig.legend(
        handles=handles,
        labels=labels,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.995),
        ncol=2,
        fontsize=6,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        borderpad=0.45,
        labelspacing=0.45,
        handlelength=1.8,
        handleheight=1.0,
        handletextpad=0.5,
        columnspacing=1.4,
    )
    ax_c.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=2,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        borderpad=0.3,
        handlelength=1.1,
        handletextpad=0.4,
    )
    for letter, (x, y) in zip(
        "abcd", [(0.005, 0.865), (0.51, 0.865), (0.005, 0.395), (0.51, 0.395)]
    ):
        fig.text(x, y, letter, fontsize=8, fontweight="bold", ha="left", va="bottom")

    base = osp.join(IMAGE_DIR, f"kinetics_tiers{stamp}")
    fig.savefig(f"{base}.png", dpi=300)
    savefig_true_size_svg(fig, f"{base}.svg")
    plt.close(fig)

    numbers = FigureNumbers(
        built_at=datetime.now(UTC).isoformat(),
        closure_json=CLOSURE_JSON,
        selected_json=osp.join(RESULTS, f"{SELECTED_BASE}{args.suffix}.json"),
        figure_svg=f"{base}.svg",
        figure_png=f"{base}.png",
        n_units=n_units,
        shipped_variant=SHIPPED_VARIANT,
        tier_counts=dict(shipped),
        tier_counts_by_variant={k: dict(v) for k, v in tier_by_variant.items()},
        admission=admission_out,
        lookup_statistic=lookup,
        admitted=admitted,
        default_rows_by_reason={key: int(value) for key, value in reasons.items()},
        n_default_units=int(cast(int, ceiling["n_default_units"])),
        n_reachable_with_open_source_tools=int(
            cast(int, ceiling["n_reachable_with_open_source_tools"])
        ),
    )
    with open(FIGURE_JSON, "w") as handle:
        handle.write(numbers.model_dump_json(indent=1))
    print(f"wrote {base}.png\nwrote {base}.svg\nwrote {FIGURE_JSON}")


if __name__ == "__main__":
    main()
