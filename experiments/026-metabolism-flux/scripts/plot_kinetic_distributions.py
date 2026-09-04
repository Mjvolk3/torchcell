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
PANELS: list[dict[str, str]] = [
    {"letter": "a", "predictor": "dlkcat", "parameter": "k_cat", "title": "DLKcat"},
    {"letter": "b", "predictor": "unikp", "parameter": "k_cat", "title": "UniKP-$k_{cat}$"},
    {"letter": "c", "predictor": "eitlem", "parameter": "k_cat",
     "title": "EITLEM-$k_{cat}$"},
    {"letter": "d", "predictor": "turnup", "parameter": "k_cat", "title": "TurNuP"},
    {"letter": "e", "predictor": "deepenzyme", "parameter": "k_cat",
     "title": "DeepEnzyme"},
    {"letter": "f", "predictor": "boost_km", "parameter": "K_M", "title": "Boost_KM"},
    {"letter": "g", "predictor": "unikp", "parameter": "K_M", "title": "UniKP-$K_M$"},
    {"letter": "h", "predictor": "eitlem", "parameter": "K_M", "title": "EITLEM-$K_M$"},
]

CORE_COLOR = PLOT_PALETTE[1]  # brick, the core network
UNDERGROUND_COLOR = PLOT_PALETTE[4]  # steel blue, the underground network
OURS_COLOR = PLOT_PALETTE[0]  # amber, our independent run


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
    """Plot one empirical CDF in log10 space and mark its median. Returns the median."""
    ordered, proportion = ecdf(values)
    if len(ordered) == 0:
        return float("nan")
    logged = np.log10(ordered[ordered > 0])
    _, proportion = ecdf(ordered[ordered > 0])
    ax.plot(logged, proportion, color=color, linestyle=style, linewidth=1.0, label=label)
    median = float(np.median(logged))
    # The short vertical tick at the median is how the original marks central tendency.
    ax.plot([median, median], [0.5, 0.62], color=color, linewidth=0.8)
    return median


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

    fig, axes = plt.subplots(
        2, 4,
        figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(88)),
        dpi=300,
    )
    record: dict[str, object] = {}

    for panel, ax in zip(PANELS, axes.flatten()):
        parameter = panel["parameter"]
        entry: dict[str, object] = {}

        theirs = load_theirs(mapping, panel["predictor"], parameter)
        if theirs is not None:
            core, underground = theirs
            entry["paper_core_n"] = int(len(core))
            entry["paper_underground_n"] = int(len(underground))
            entry["paper_core_median_log10"] = draw_curve(
                ax, core, CORE_COLOR, "core (paper)"
            )
            entry["paper_underground_median_log10"] = draw_curve(
                ax, underground, UNDERGROUND_COLOR, "underground (paper)"
            )

        ours = load_ours(panel["predictor"], parameter)
        if ours is not None:
            entry["ours_n"] = int(len(ours))
            entry["ours_median_log10"] = draw_curve(
                ax, ours, OURS_COLOR, "core (ours)", style="--"
            )

        if theirs is None and ours is None:
            ax.text(0.5, 0.5, "not run", transform=ax.transAxes, ha="center",
                    va="center", fontsize=6, color="#666666")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            unit = "1/s" if parameter == "k_cat" else "mM"
            symbol = "$k_{cat}$" if parameter == "k_cat" else "$K_M$"
            ax.set_xlabel(f"log$_{{10}}$ {symbol} ({unit})")
            ax.set_ylabel("cumulative proportion")
            ax.set_ylim(0, 1.02)
            ax.legend(loc="lower right", frameon=False, handlelength=1.4,
                      handletextpad=0.4, borderpad=0.1)

        ax.set_title(panel["title"], fontsize=6)
        ax.text(-0.24, 1.06, panel["letter"], transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="top")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
        record[f"{panel['letter']}_{panel['predictor']}_{parameter}"] = entry

    fig.tight_layout()
    savefig_true_size_svg(fig, osp.join(OUT_DIR, f"kinetics_fig3{stamp}.svg"))
    fig.savefig(osp.join(OUT_DIR, f"kinetics_fig3{stamp}.png"), dpi=300)
    plt.close(fig)

    with open(osp.join(RESULTS, "kinetic_distribution_panels.json"), "w") as handle:
        json.dump(record, handle, indent=2)
    print(json.dumps(record, indent=2))
    print(f"figure written to {OUT_DIR}{stamp and ' with stamp ' + stamp}")


if __name__ == "__main__":
    main()
