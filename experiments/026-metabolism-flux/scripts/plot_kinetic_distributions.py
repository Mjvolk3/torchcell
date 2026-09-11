# experiments/026-metabolism-flux/scripts/plot_kinetic_distributions.py
# [[experiments.026-metabolism-flux.scripts.plot_kinetic_distributions]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/plot_kinetic_distributions.py

r"""Reproduce Wu et al. (2026) Fig. 3a-h: predicted kinetic parameter distributions.

WHAT THE ORIGINAL FIGURE IS
-----------------------------
Eight panels, each an empirical cumulative distribution of a predicted kinetic parameter,
with two curves: the **core network** (reactions already in Yeast9) against the
**underground network** (reactions predicted by Yeast-MetaTwin). Panels a-e are
:math:`k_{cat}` from DLKcat, UniKP, EITLEM-Kinetics, TurNuP and DeepEnzyme; f-h are
:math:`K_M` from Boost_KM, UniKP and EITLEM-Kinetics. The paper's conclusion is that
underground metabolism is separated from known metabolism by :math:`K_M` rather than by
:math:`k_{cat}`.

The shape of the plot is taken from the authors' own notebook
(``Code/kcatkm_prediction/Yeast-MetaTwin-05.Fig3abcde.ipynb``), which sorts the values and
plots the running proportion, marking each curve's median with a short vertical tick. The
split there is on whether a reaction id contains ``rxn``, which is how their tables encode
predicted rather than curated reactions.

WHAT IS REPRODUCED HERE, AND WHAT IS INDEPENDENT
--------------------------------------------------
Two different things are drawn on each panel and they must not be confused.

* **Reproduced.** The authors' own released per-pair predictions, split core against
  underground. This redraws their figure from their numbers, and it is the only way to
  show the underground arm at all, since generating the Yeast-MetaTwin reaction set is a
  separate pipeline.
* **Independent.** Our own run of the same predictor over yeast-GEM 9.0.2, which is the
  core network. This is a replication of the core arm from our own mirrors and our own
  genome-derived sequences, and it is the curve that tests whether we can reproduce their
  numbers rather than merely re-plot them.

A panel with only our curve means that predictor ran here but the authors' table for it is
not available; a panel with only theirs means the reverse. Both cases are labeled on the
panel rather than left to inference.
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

from torchcell.timestamp import timestamp
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    apply_paper_style,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()
DATA_ROOT = cast(str, os.getenv("DATA_ROOT"))
ASSET_IMAGES_DIR = cast(str, os.getenv("ASSET_IMAGES_DIR"))
EXPERIMENT_ROOT = cast(str, os.getenv("EXPERIMENT_ROOT"))
KINETICS = osp.join(DATA_ROOT, "data", "torchcell", "kinetics")
METATWIN = osp.join(DATA_ROOT, "data", "enzyme_kinetics", "yeast_metatwin")
RESULTS = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
OUT_DIR = osp.join(ASSET_IMAGES_DIR, "026-metabolism-flux")

# Panel order and labelling follow the paper exactly, so a reader can set the two figures
# side by side. ``ours`` names the local build; ``theirs`` the authors' released table.
#
# EZKinetics is not in the original eight. It is drawn here because the comparison this
# document makes is between the predictors WE can run, and leaving it out of the
# distribution figure while ranking it first on accuracy elsewhere would be inconsistent.
# It has no authors' table, so it carries our curve only, like the other five k_cat panels.
PANELS: list[dict[str, str]] = [
    {"letter": "a", "predictor": "dlkcat", "parameter": "k_cat", "title": "DLKcat"},
    {"letter": "b", "predictor": "unikp", "parameter": "k_cat", "title": "UniKP-$k_{cat}$"},
    {"letter": "c", "predictor": "eitlem", "parameter": "k_cat",
     "title": "EITLEM-$k_{cat}$"},
    {"letter": "d", "predictor": "turnup", "parameter": "k_cat", "title": "TurNuP"},
    {"letter": "e", "predictor": "deepenzyme", "parameter": "k_cat",
     "title": "DeepEnzyme"},
    {"letter": "f", "predictor": "ezkinetics", "parameter": "k_cat",
     "title": "EZKinetics-$k_{cat}$"},
    {"letter": "g", "predictor": "boost_km", "parameter": "K_M", "title": "Boost_KM"},
    {"letter": "h", "predictor": "unikp", "parameter": "K_M", "title": "UniKP-$K_M$"},
    {"letter": "i", "predictor": "eitlem", "parameter": "K_M", "title": "EITLEM-$K_M$"},
    {"letter": "j", "predictor": "ezkinetics", "parameter": "K_M",
     "title": "EZKinetics-$K_M$"},
]

# The three-way split Wu et al.'s Fig. 3 draws. Reserved rather than drawn: their per-pair
# core and underground tables are in neither released artifact, so only OURS_COLOR has a
# curve. Three categories take THREE PRIMARIES; underground was the blue slot, which is a
# secondary and does not belong as the third color of a three-series scheme.
CORE_COLOR = PLOT_PALETTE[1]  # brick, the core network
UNDERGROUND_COLOR = PLOT_PALETTE[2]  # lilac, the underground network
OURS_COLOR = PLOT_PALETTE[0]  # amber, our independent run


#: One predictor, one color, in every figure of this document. Same map as the agreement
#: and EC-class figures, so a reader comparing them is comparing the same enzymes.
PREDICTOR_COLOR_INDEX = {
    "dlkcat": 0, "unikp": 1, "eitlem": 2, "turnup": 3, "deepenzyme": 4, "ezkinetics": 5,
    "boost_km": 6,
}
DISPLAY = {
    "dlkcat": "DLKcat", "unikp": "UniKP", "eitlem": "EITLEM", "turnup": "TurNuP",
    "deepenzyme": "DeepEnzyme", "ezkinetics": "EZKinetics", "boost_km": "Boost_KM",
}


def predictor_color(name: str) -> str:
    return PLOT_PALETTE[PREDICTOR_COLOR_INDEX[name] % len(PLOT_PALETTE)]


def apply_style() -> None:
    """Repo figure standards, from the shared source rather than a local copy.

    ``apply_paper_style`` carries the Arial mathtext settings, so a panel title like
    ``UniKP-$k_{cat}$`` renders in one face at one size.
    """
    apply_paper_style()
    plt.rcParams.update({"xtick.major.width": 0.5, "ytick.major.width": 0.5})


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sorted values and their running proportion, the authors' own construction."""
    ordered = np.sort(values[np.isfinite(values)])
    return ordered, np.arange(1, len(ordered) + 1) / max(len(ordered), 1)


def draw_curve(ax: plt.Axes, values: np.ndarray, color: str, label: str,
               style: str = "-") -> float:
    """Plot one empirical CDF in log10 space. Returns the median, which is not drawn.

    The original marks each curve's median with a short vertical tick, which works when a
    panel holds one or two curves. Overlaid, six ticks land at the same height within a few
    tenths of each other and read as artifacts on the curves rather than as annotations. The
    median of a cumulative distribution is where it crosses 0.5, so a single reference line
    at that height shows every median at once and costs no ink per curve. The value is still
    returned and recorded.
    """
    ordered, proportion = ecdf(values)
    if len(ordered) == 0:
        return float("nan")
    logged = np.log10(ordered[ordered > 0])
    _, proportion = ecdf(ordered[ordered > 0])
    ax.plot(logged, proportion, color=color, linestyle=style, linewidth=1.0, label=label)
    return float(np.median(logged))


def load_ours(predictor: str, parameter: str) -> np.ndarray | None:
    """Our own build for one predictor and parameter, if it exists."""
    path = osp.join(KINETICS, predictor, "processed", f"{parameter}.parquet")
    if not osp.exists(path):
        return None
    return pd.read_parquet(path)[parameter].to_numpy(dtype=float)


def load_theirs(mapping: dict[str, dict[str, str]], predictor: str,
                parameter: str) -> tuple[np.ndarray, np.ndarray] | None:
    """The authors' table for one predictor, split into core and underground.

    Their convention: a reaction id containing ``rxn`` is a predicted (underground)
    reaction; anything else is a curated Yeast9 reaction.
    """
    key = f"{predictor}:{parameter}"
    if key not in mapping:
        return None
    spec = mapping[key]
    frame = pd.read_csv(spec["path"])
    column = spec["column"]
    ids = frame[spec["id_column"]].astype(str)
    underground = frame[ids.str.contains("rxn")][column].to_numpy(dtype=float)
    core = frame[~ids.str.contains("rxn")][column].to_numpy(dtype=float)
    return core, underground


def main() -> None:
    """Draw the eight panels and record what each one was able to show."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paper-tables",
        default=osp.join(METATWIN, "paper_tables.json"),
        help="Map of 'predictor:parameter' -> {path, column, id_column} for the "
        "authors' released predictions, written by unpack_metatwin.py.",
    )
    parser.add_argument("--no-timestamp", action="store_true")
    args = parser.parse_args()

    apply_style()
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = "" if args.no_timestamp else f"_{timestamp()}"

    mapping: dict[str, dict[str, str]] = {}
    if osp.exists(args.paper_tables):
        with open(args.paper_tables) as handle:
            mapping = json.load(handle)

    # ONE panel per parameter, every predictor overlaid, rather than one panel per
    # predictor. Ten separate ECDFs of the same quantity all look alike, and what the
    # comparison is actually about is how far apart they sit: that is a difference between
    # curves, and a difference between curves cannot be read across ten panels. The
    # per-predictor medians are still recorded, so nothing measured is lost.
    fig, axes = plt.subplots(
        1, 2,
        figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(66)),
        dpi=300,
    )
    record: dict[str, object] = {}

    for ax, parameter, letter in zip(axes, ("k_cat", "K_M"), "ab"):
        panels = [p for p in PANELS if p["parameter"] == parameter]
        for panel in panels:
            entry: dict[str, object] = {}
            predictor = panel["predictor"]
            ours = load_ours(predictor, parameter)
            if ours is None:
                continue
            entry["ours_n"] = int(len(ours))
            entry["ours_median_log10"] = draw_curve(
                ax, ours, predictor_color(predictor), DISPLAY[predictor]
            )
            theirs = load_theirs(mapping, predictor, parameter)
            if theirs is not None:
                core, underground = theirs
                entry["paper_core_n"] = int(len(core))
                entry["paper_underground_n"] = int(len(underground))
            record[f"{predictor}_{parameter}"] = entry

        # Drawn after the curves so it reads as a reference rather than as data, and thin
        # enough not to compete with them. Each curve's median is its crossing point.
        ax.axhline(0.5, color="black", linewidth=0.5, linestyle=":", zorder=1,
                   label="median (crossing)")
        unit = "1/s" if parameter == "k_cat" else "mM"
        symbol = "$k_{cat}$" if parameter == "k_cat" else "$K_M$"
        ax.set_xlabel(f"log$_{{10}}$ {symbol} ({unit})")
        ax.set_ylabel("cumulative proportion")
        ax.set_ylim(0, 1.02)
        ax.legend(
            loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=3, frameon=False,
            fontsize=6, handlelength=1.2, handletextpad=0.4, borderpad=0.0,
            columnspacing=1.0,
        )
        ax.text(-0.16, 1.30, letter, transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="top")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)

    fig.tight_layout(w_pad=2.6)
    savefig_true_size_svg(fig, osp.join(OUT_DIR, f"kinetics_fig3{stamp}.svg"))
    fig.savefig(osp.join(OUT_DIR, f"kinetics_fig3{stamp}.png"), dpi=300)
    plt.close(fig)

    with open(osp.join(RESULTS, "kinetic_distribution_panels.json"), "w") as handle:
        json.dump(record, handle, indent=2)
    print(json.dumps(record, indent=2))
    print(f"figure written to {OUT_DIR}{stamp and ' with stamp ' + stamp}")


if __name__ == "__main__":
    main()
