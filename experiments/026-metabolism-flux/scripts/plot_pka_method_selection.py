# experiments/026-metabolism-flux/scripts/plot_pka_method_selection.py
# [[experiments.026-metabolism-flux.scripts.plot_pka_method_selection]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/plot_pka_method_selection.py

r"""Four panels of the pKa method-selection benchmark, drawn from the per-compound scores.

``external_pka_method_selection.py score`` writes one row per compound per method to
``results/pka_method_selection_per_compound.csv``, carrying the change in ``dGf'0`` that
swapping the method's pKa ladder for the cache's ChemAxon list produces, at pH 7.2 and at
every working compartment pH. The tables in the note report summaries of that file; the
panels report the file.

Every panel is drawn on the HELD-OUT compounds only, the ones outside the pinned seed-17
300, because the pinned subset is where the ensemble membership and the per-group rule were
fitted.

Panels

``a``
    The distribution of ``|ddGf'0|`` at pH 7.2 per method, as a box over the held-out
    compounds on a log axis. The tail is what the selection rule reads, so the whiskers are
    drawn at the 10th and 90th percentiles and the box at the quartiles.
``b``
    The p90 of ``|ddGf'0|`` at each working compartment pH, one line per method. This is
    the panel that answers whether a method chosen at pH 7.2 still holds in the vacuole.
``c``
    The p90 per method split by the lipid-like stratum, which is the regime the yeast-GEM
    fill-in actually runs in and the regime the earlier 300-compound set could not reach.
``d``
    Failure rate against p90, one point per method: the two quantities the selection rule
    trades off. A method in the lower left answers for everything and answers well.

Every number a panel shows is written to ``results/pka_method_selection_figure.json``.

Run::

    PYTHONPATH=<repo> ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/plot_pka_method_selection.py
"""

from __future__ import annotations

import json
import os
import os.path as osp
from datetime import UTC, datetime
from typing import cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.lines import Line2D
from pydantic import BaseModel

from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    apply_paper_style,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()

EXPERIMENT_ROOT = cast(str, os.environ["EXPERIMENT_ROOT"])
ASSET_IMAGES_DIR = cast(str, os.environ["ASSET_IMAGES_DIR"])
RESULTS = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
PER_COMPOUND_CSV = osp.join(RESULTS, "pka_method_selection_per_compound.csv")
SCORES_JSON = osp.join(RESULTS, "pka_method_selection_scores.json")
FIGURE_JSON = osp.join(RESULTS, "pka_method_selection_figure.json")
IMAGE_DIR = osp.join(ASSET_IMAGES_DIR, "026-metabolism-flux")
FIGURE_STEM = osp.join(IMAGE_DIR, "pka_method_selection")

FIG_W_MM = PANEL_WIDTHS_MM["full"]
FIG_H_MM = 96.0
P_H = 7.2

#: The method key is an identifier: it names a CSV column value and a JSON key. Every place
#: a reader sees a method named uses the published capitalization instead.
METHOD_DISPLAY = {
    "molgpka": "MolGpKa",
    "pkasolver-lite": "pKasolver-lite",
    "pkasolver": "pKasolver-lite",
    "unipka-simple-template": "Uni-pKa simple",
    "unipka": "Uni-pKa full",
    "qupkake-truncated": "QupKake trunc.",
    "ensemble_top2": "Median top 2",
    "ensemble_top3": "Median top 3",
    "cascade_top2": "Cascade top 2",
    "cascade_top3": "Cascade top 3",
    "per_group_best": "Per-group best",
}

#: A floor for the log axis. A transform difference of exactly zero happens when a method
#: reproduces the cache ladder, and log(0) is not plottable; clipping is stated rather than
#: silently dropping those compounds.
LOG_FLOOR_KJ = 1e-3


def display_method(key: str) -> str:
    return METHOD_DISPLAY.get(key, key)


class PanelNumbers(BaseModel):
    """Everything one panel shows, so the caption never has to be read off the pixels."""
    panel: str
    what: str
    values: dict[str, object]


class FigureNumbers(BaseModel):
    """The figure as data."""
    built_at: str
    source_csv: str
    scores_json: str
    figure_svg: str
    figure_png: str
    n_heldout_compounds: int
    working_ph: list[float]
    methods_drawn: list[str]
    panels: list[PanelNumbers]


def _style_axes(ax) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
    ax.tick_params(labelsize=6, width=0.5, length=2)
    ax.grid(True, which="major", linewidth=0.3, color="0.85")
    ax.set_axisbelow(True)


def main() -> None:
    apply_paper_style()
    os.makedirs(IMAGE_DIR, exist_ok=True)

    with open(SCORES_JSON) as handle:
        scores = json.load(handle)
    working_ph = [float(p) for p in scores["working_ph"]]
    frame = pd.read_csv(PER_COMPOUND_CSV)
    heldout = frame[~frame["pinned"].astype(bool)]

    order = [
        m["method"]
        for m in sorted(
            scores["methods_heldout"],
            key=lambda s: s["worst_working_p90"]
            if s["worst_working_p90"] is not None
            else 1e9,
        )
    ]
    colors = {name: PLOT_PALETTE[i % len(PLOT_PALETTE)] for i, name in enumerate(order)}
    panels: list[PanelNumbers] = []

    fig = plt.figure(figsize=(mm_to_in(FIG_W_MM), mm_to_in(FIG_H_MM)))
    grid = fig.add_gridspec(
        2, 2, left=0.075, right=0.995, bottom=0.235, top=0.925, wspace=0.30, hspace=0.85
    )

    # ------------------------------------------------------------------ a: distribution
    ax = fig.add_subplot(grid[0, 0])
    column = f"ddgf_ph{P_H:g}"
    data, labels, stats = [], [], {}
    for name in order:
        values = heldout.loc[
            (heldout["method"] == name) & heldout[column].notna(), column
        ].abs()
        values = values.clip(lower=LOG_FLOOR_KJ)
        if values.empty:
            continue
        data.append(values.to_numpy())
        labels.append(display_method(name))
        stats[name] = {
            "n": int(values.size),
            "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)),
            "max": float(values.max()),
        }
    box = ax.boxplot(
        data,
        whis=(10, 90),
        showfliers=False,
        patch_artist=True,
        widths=0.6,
        medianprops={"color": "black", "linewidth": 0.8},
        boxprops={"linewidth": 0.5},
        whiskerprops={"linewidth": 0.5},
        capprops={"linewidth": 0.5},
    )
    for patch, name in zip(box["boxes"], order):
        patch.set_facecolor(colors[name])
        patch.set_edgecolor("black")
    ax.set_yscale("log")
    ax.set_ylabel(r"$|\Delta \Delta_f G'^\circ|$ at pH 7.2 (kJ mol$^{-1}$)", fontsize=6)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    _style_axes(ax)
    panels.append(
        PanelNumbers(
            panel="a",
            what=(
                "box of |ddGf'0| at pH 7.2 over the held-out compounds; whiskers at the "
                "10th and 90th percentiles, values clipped up to 1e-3 for the log axis"
            ),
            values=stats,
        )
    )

    # -------------------------------------------------------------------- b: across pH
    ax = fig.add_subplot(grid[0, 1])
    by_ph: dict[str, list[float | None]] = {}
    for i, name in enumerate(order):
        block = next(
            m for m in scores["methods_heldout"] if m["method"] == name
        )["by_ph"]
        values = [block.get(f"{p:g}", {}).get("p90") for p in working_ph]
        by_ph[name] = values
        ax.plot(
            working_ph,
            values,
            marker="o",
            markersize=2.0,
            linewidth=0.8,
            color=colors[name],
            label=display_method(name),
        )
    ax.set_xlabel("compartment pH", fontsize=6)
    ax.set_ylabel(r"p90 $|\Delta \Delta_f G'^\circ|$ (kJ mol$^{-1}$)", fontsize=6)
    _style_axes(ax)
    panels.append(
        PanelNumbers(
            panel="b",
            what="p90 of |ddGf'0| at each working compartment pH, held out",
            values={"working_ph": working_ph, "p90_by_method": by_ph},
        )
    )

    # ---------------------------------------------------------------------- c: lipids
    ax = fig.add_subplot(grid[1, 0])
    strata = ["not_lipid_like", "lipid_like"]
    width = 0.8 / max(1, len(order))
    lipid_stats: dict[str, dict[str, float | None]] = {}
    for i, name in enumerate(order):
        heights: list[float] = []
        for stratum in strata:
            subset = heldout[
                (heldout["method"] == name)
                & (heldout["lipid"] == stratum)
                & heldout[column].notna()
            ][column].abs()
            heights.append(
                float(np.percentile(subset, 90)) if len(subset) else float("nan")
            )
        lipid_stats[name] = dict(zip(strata, heights))
        ax.bar(
            np.arange(len(strata)) + i * width - 0.4 + width / 2,
            heights,
            width=width,
            color=colors[name],
            edgecolor="black",
            linewidth=0.4,
        )
    ax.set_xticks(range(len(strata)))
    ax.set_xticklabels(["not lipid-like", "lipid-like"], fontsize=6)
    ax.set_ylabel(r"p90 $|\Delta \Delta_f G'^\circ|$ (kJ mol$^{-1}$)", fontsize=6)
    _style_axes(ax)
    panels.append(
        PanelNumbers(
            panel="c",
            what="p90 of |ddGf'0| at pH 7.2 by lipid-like stratum, held out",
            values=lipid_stats,
        )
    )

    # ------------------------------------------------------------- d: failure vs error
    ax = fig.add_subplot(grid[1, 1])
    trade: dict[str, dict[str, float | None]] = {}
    for name in order:
        block = next(m for m in scores["methods_heldout"] if m["method"] == name)
        ax.plot(
            [block["failure_rate"]],
            [block["worst_working_p90"]],
            marker="o",
            markersize=3.0,
            linestyle="none",
            color=colors[name],
        )
        trade[name] = {
            "failure_rate": block["failure_rate"],
            "worst_working_p90": block["worst_working_p90"],
            "frac_count_match": block["frac_count_match"],
        }
    ax.axvline(0.02, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("failure rate", fontsize=6)
    ax.set_ylabel(
        r"worst working-pH p90 $|\Delta \Delta_f G'^\circ|$ (kJ mol$^{-1}$)", fontsize=6
    )
    _style_axes(ax)
    panels.append(
        PanelNumbers(
            panel="d",
            what=(
                "failure rate against the worst working-pH p90, held out; the dashed line "
                "is the selection rule's failure-rate bound"
            ),
            values=trade,
        )
    )

    handles = [
        Line2D(
            [],
            [],
            marker="s",
            linestyle="none",
            markersize=3,
            markerfacecolor=colors[name],
            markeredgecolor="black",
            markeredgewidth=0.4,
            label=display_method(name),
        )
        for name in order
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.004),
        ncol=4,
        frameon=True,
        fontsize=6,
        handletextpad=0.4,
        columnspacing=1.0,
    )
    for letter, (x, y) in zip("abcd", [(0.005, 0.955), (0.52, 0.955), (0.005, 0.505), (0.52, 0.505)]):
        fig.text(x, y, letter, fontsize=8, fontweight="bold", ha="left", va="bottom")

    png = f"{FIGURE_STEM}.png"
    svg = f"{FIGURE_STEM}.svg"
    fig.savefig(png, dpi=300)
    savefig_true_size_svg(fig, svg)
    plt.close(fig)

    numbers = FigureNumbers(
        built_at=datetime.now(UTC).isoformat(),
        source_csv=PER_COMPOUND_CSV,
        scores_json=SCORES_JSON,
        figure_svg=svg,
        figure_png=png,
        n_heldout_compounds=int(heldout["compound_id"].nunique()),
        working_ph=working_ph,
        methods_drawn=order,
        panels=panels,
    )
    with open(FIGURE_JSON, "w") as handle:
        handle.write(numbers.model_dump_json(indent=1))
    print(f"wrote {png}\nwrote {svg}\nwrote {FIGURE_JSON}")


if __name__ == "__main__":
    main()
