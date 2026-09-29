# experiments/033-env-chemgen-pooled/scripts/post_query_figures.py
# [[experiments.033-env-chemgen-pooled.scripts.post_query_figures]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/033-env-chemgen-pooled/scripts/post_query_figures
"""Figures of the post-query analysis, drawn from the tables ``store_against_plan.py`` wrote.

Three figures, each at the full panel width:

``store_units``          what a row is: served measurements, store cells and the plan's
                         (gene, compound) pairs per source; the spread at each unit; and
                         how many measurements a cell folds.
``standardized_target``  the pooled target after orienting and standardizing per source:
                         the four densities on one axis, each source's share of the cells
                         against its share of the raw squared error, and how much of each
                         source's squared error sits beyond three standard deviations.
``store_reliability``    repeat agreement inside the store per screen pair, the Vanacloig
                         fold ceilings from the store against the plan's, and the two
                         Hoepfner ploidy arms against each other on paired cells.

Reads the result CSVs, plus the cell table for the two panels that need every cell.
``--stable`` writes names without a timestamp, which is what the notes-tex document's
``make plots`` looks for.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator

from torchcell.timestamp import timestamp
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    PLOT_PALETTE_FILL,
    apply_paper_style,
    mm_to_in,
    panel_label,
    savefig_true_size_svg,
)

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]

BUILD_ROOT = "/db/experiments/033-env-chemgen-pooled-001-pooled-build"
LABEL_TABLE = osp.join(BUILD_ROOT, "processed", "label_df.parquet")
DEFAULT_CELLS = osp.join(
    DATA_ROOT,
    "experiments",
    "033-env-chemgen-pooled",
    "cell_table",
    "cell_table.parquet",
)
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "033-env-chemgen-pooled", "results")
IMAGE_DIR = osp.join(ASSET_IMAGES_DIR, "033-env-chemgen-pooled")

DISPLAY: dict[str, str] = {
    "EnvChemgenVanacloig2022Dataset": "Vanacloig 2022",
    "HetHillenmeyer2008Dataset": "Hillenmeyer HET",
    "EnvChemgenHoepfner2014Dataset": "Hoepfner 2014",
    "EnvChemgenWildenhain2015Dataset": "Wildenhain 2015",
}
SHORT: dict[str, str] = {
    "Vanacloig 2022": "Vanacloig",
    "Hillenmeyer HET": "HET",
    "Hoepfner 2014": "Hoepfner",
    "Wildenhain 2015": "Wildenhain",
}
SICK_SIGN: dict[str, int] = {
    "EnvChemgenVanacloig2022Dataset": -1,
    "HetHillenmeyer2008Dataset": +1,
    "EnvChemgenHoepfner2014Dataset": -1,
    "EnvChemgenWildenhain2015Dataset": -1,
}
COLOR: dict[str, str] = dict(zip(DISPLAY.values(), PLOT_PALETTE[:4], strict=True))
FILL: dict[str, str] = dict(zip(DISPLAY.values(), PLOT_PALETTE_FILL[:4], strict=True))
#: The plan's reference reliability, the median Pearson over the 71 cross-screen pairs
#: of 031 ``hoepfner_cross_screen_reliability.csv``, drawn as a line in the reliability
#: panel. The same statistic as the points, so the comparison is like for like.
PLAN_HOEPFNER_RELIABILITY = 0.611


#: Panel letters sit 12 pt above the axes, so the layout leaves the top 8 percent free.
LETTER_RECT = (0.0, 0.0, 1.0, 0.92)


def tenths(ax: Axes) -> None:
    """A gridline every 0.1 on a 0 to 1 axis, labeled every 0.2."""
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.grid(axis="y", which="both", color="0.85", linewidth=0.3)
    ax.tick_params(axis="y", which="minor", length=0)
    ax.set_axisbelow(True)


def box(ax: Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.5)
    ax.tick_params(width=0.5, length=2)


def save(fig: Figure, name: str, stable: bool) -> None:
    os.makedirs(IMAGE_DIR, exist_ok=True)
    stem = osp.join(IMAGE_DIR, name if stable else f"{name}_{timestamp()}")
    fig.savefig(f"{stem}.png", dpi=300)
    savefig_true_size_svg(fig, f"{stem}.svg")
    plt.close(fig)
    print(f"wrote {stem}.svg")


def grouped_bars(
    ax: Axes, frame: pd.DataFrame, columns: dict[str, str], hatches: list[str]
) -> None:
    """One group per source, one bar per column, the series told apart by hatch."""
    width = 0.8 / len(columns)
    for j, (column, label) in enumerate(columns.items()):
        x = np.arange(len(frame)) + (j - (len(columns) - 1) / 2) * width
        ax.bar(
            x,
            frame[column],
            width=width,
            color=[COLOR[d] for d in frame["dataset"]],
            edgecolor="black",
            linewidth=0.5,
            hatch=hatches[j],
            label=label,
        )
    ax.set_xticks(np.arange(len(frame)))
    ax.set_xticklabels([SHORT[d] for d in frame["dataset"]])
    legend = ax.legend(loc="upper left")
    for handle in legend.legend_handles:
        handle.set_facecolor("white")


def fig_store_units(results: str, stable: bool) -> None:
    axes_table = pd.read_csv(osp.join(results, "store_axes.csv"))
    units = pd.read_csv(osp.join(results, "unit_of_analysis.csv"))
    with open(osp.join(results, "dataset_index_summary.json")) as f:
        multiplicity = json.load(f)["measurements_per_entry"]

    fig, (a, b, c) = plt.subplots(
        1, 3, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(62))
    )
    counts = axes_table[["dataset", "measurements", "cells"]].merge(
        units[["dataset", "pair_cells"]], on="dataset", validate="one_to_one"
    )
    grouped_bars(
        a,
        counts,
        {
            "measurements": "served measurements",
            "cells": "store cells",
            "pair_cells": "(gene, compound) pairs",
        },
        ["", "///", "..."],
    )
    a.set_yscale("log")
    a.set_ylim(1e4, 3e7)
    a.set_ylabel("count")
    a.set_title("What a row is", loc="left", pad=3)

    grouped_bars(
        b,
        units,
        {"store_sd": "store cell", "pair_sd": "(gene, compound) pair"},
        ["", "..."],
    )
    b.set_yscale("log")
    b.set_ylabel("standard deviation of the response")
    b.set_title("Spread at each unit", loc="left", pad=3)

    sizes = sorted(int(k) for k in multiplicity)
    c.bar(
        [str(s) for s in sizes],
        [multiplicity[str(s)] for s in sizes],
        color=PLOT_PALETTE[5],
        edgecolor="black",
        linewidth=0.5,
    )
    c.set_yscale("log")
    c.set_xlabel("measurements folded into one cell")
    c.set_ylabel("cells")
    c.set_title("Repeated cells", loc="left", pad=3)

    for ax in (a, b, c):
        box(ax)
    fig.tight_layout(w_pad=1.5, rect=LETTER_RECT)
    for ax, letter in zip((a, b, c), "abc", strict=True):
        panel_label(ax, letter)
    save(fig, "store_units", stable)


def fig_standardized_target(table: pd.DataFrame, results: str, stable: bool) -> None:
    target = pd.read_csv(osp.join(results, "standardized_target.csv"))
    fig, (a, b, c) = plt.subplots(
        1,
        3,
        figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(62)),
        gridspec_kw={"width_ratios": [1.5, 1, 1]},
    )
    bins = np.linspace(-15, 8, 231)
    for name, display in DISPLAY.items():
        y = -SICK_SIGN[name] * table.loc[table["dataset"] == name, "label"].to_numpy()
        z = (y - y.mean()) / y.std(ddof=1)
        density, _ = np.histogram(z, bins=bins, density=True)
        a.step(bins[:-1], density, where="post", color=COLOR[display], linewidth=0.7)
        a.plot([], [], color=COLOR[display], linewidth=0.7, label=SHORT[display])
    normal = np.exp(-0.5 * bins**2) / np.sqrt(2 * np.pi)
    a.plot(bins, normal, color="black", linewidth=0.5, linestyle=":", label="normal")
    a.set_yscale("log")
    a.set_ylim(1e-6, 2)
    a.set_xlim(-15, 8)
    a.set_xlabel("oriented response, standard deviations of its own source")
    a.set_ylabel("density")
    a.set_title("One axis after the transform", loc="left", pad=3)
    a.legend(loc="upper left")

    x = np.arange(len(target))
    b.bar(
        x - 0.2,
        target["share_of_cells"],
        width=0.4,
        color=[COLOR[d] for d in target["dataset"]],
        edgecolor="black",
        linewidth=0.5,
        label="share of cells",
    )
    b.bar(
        x + 0.2,
        target["share_of_raw_squared_error"],
        width=0.4,
        color=[FILL[d] for d in target["dataset"]],
        edgecolor="black",
        linewidth=0.5,
        hatch="///",
        label="share of raw squared error",
    )
    b.set_xticks(x)
    b.set_xticklabels([SHORT[d] for d in target["dataset"]], rotation=30, ha="right")
    b.set_ylim(0, 1)
    tenths(b)
    b.set_ylabel("share of the pool")
    b.set_title("Unstandardized weight", loc="left", pad=3)
    legend = b.legend(loc="upper left")
    legend.legend_handles[0].set_facecolor("white")
    legend.legend_handles[1].set_facecolor("white")

    c.bar(
        x - 0.2,
        target["frac_cells_beyond_3sd"],
        width=0.4,
        color=[COLOR[d] for d in target["dataset"]],
        edgecolor="black",
        linewidth=0.5,
        label="cells beyond 3 sd",
    )
    c.bar(
        x + 0.2,
        target["share_of_own_squared_error_beyond_3sd"],
        width=0.4,
        color=[FILL[d] for d in target["dataset"]],
        edgecolor="black",
        linewidth=0.5,
        hatch="///",
        label="their share of squared error",
    )
    c.set_xticks(x)
    c.set_xticklabels([SHORT[d] for d in target["dataset"]], rotation=30, ha="right")
    c.set_ylim(0, 1)
    tenths(c)
    c.set_ylabel("share within the source")
    c.set_title("Where the loss sits", loc="left", pad=3)
    legend = c.legend(loc="upper left")
    legend.legend_handles[0].set_facecolor("white")
    legend.legend_handles[1].set_facecolor("white")

    for ax in (a, b, c):
        box(ax)
    fig.tight_layout(w_pad=1.5, rect=LETTER_RECT)
    for ax, letter in zip((a, b, c), "abc", strict=True):
        panel_label(ax, letter)
    save(fig, "standardized_target", stable)


def fig_store_reliability(table: pd.DataFrame, results: str, stable: bool) -> None:
    pairs = pd.read_csv(osp.join(results, "within_cell_pairs.csv"))
    folds = pd.read_csv(osp.join(results, "vanacloig_folds.csv"))

    fig, (a, b, c) = plt.subplots(
        1, 3, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(62))
    )
    groups = {
        "HET": pairs[pairs["dataset"] == "Hillenmeyer HET"],
        "Hoepfner\nheterozygous": pairs[
            (pairs["dataset"] == "Hoepfner 2014") & (pairs["functional_dose"] == 0.5)
        ],
        "Hoepfner\nhomozygous": pairs[
            (pairs["dataset"] == "Hoepfner 2014") & (pairs["functional_dose"] == 0.0)
        ],
    }
    colors = [COLOR["Hillenmeyer HET"], COLOR["Hoepfner 2014"], COLOR["Hoepfner 2014"]]
    rng = np.random.default_rng(0)
    for i, ((_, g), color) in enumerate(zip(groups.items(), colors, strict=True)):
        jitter = rng.uniform(-0.25, 0.25, len(g))
        a.scatter(i + jitter, g["pearson"], s=2, color=color, linewidths=0)
        a.plot([i - 0.35, i + 0.35], [g["pearson"].median()] * 2, color="black", lw=0.8)
    a.axhline(
        PLAN_HOEPFNER_RELIABILITY,
        color="black",
        lw=0.5,
        ls=":",
        label=f"plan, Hoepfner cross-screen median {PLAN_HOEPFNER_RELIABILITY}",
    )
    a.legend(loc="lower right")
    a.set_xticks(range(len(groups)))
    a.set_xticklabels(
        [f"{label}\nn={len(g)}" for label, g in groups.items()], fontsize=6
    )
    a.set_ylim(-0.3, 1.0)
    a.set_ylabel("Pearson between two screens, over genes")
    a.set_title("Repeat agreement in the store", loc="left", pad=3)

    b.plot([-0.2, 1], [-0.2, 1], color="black", lw=0.5, ls=":")
    b.scatter(
        folds["plan_reliability"],
        folds["reliability"],
        s=6,
        color=COLOR["Vanacloig 2022"],
        edgecolor="black",
        linewidths=0.3,
    )
    b.set_xlim(-0.2, 1)
    b.set_ylim(-0.2, 1)
    b.set_xlabel("reliability index, plan (served records)")
    b.set_ylabel("reliability index, built store")
    b.set_title("Vanacloig folds, 41 compounds", loc="left", pad=3)

    h = table[
        (table["dataset"] == "EnvChemgenHoepfner2014Dataset")
        & (table["n_compounds"] == 1)
    ]
    arm = (
        h.groupby(["query_gene", "inchikeys", "conc_values", "functional_dose"])[
            "mean_response"
        ]
        .mean()
        .unstack("functional_dose")
        .dropna()
    )
    lo = float(np.quantile(arm.to_numpy(), 0.005))
    hi = float(np.quantile(arm.to_numpy(), 0.995))
    c.hexbin(
        arm[0.0],
        arm[0.5],
        gridsize=60,
        bins="log",
        cmap="Greys",
        extent=(lo, hi, lo, hi),
        linewidths=0,
        mincnt=1,
    )
    c.axhline(0, color="black", lw=0.3)
    c.axvline(0, color="black", lw=0.3)
    c.set_xlim(lo, hi)
    c.set_ylim(lo, hi)
    c.set_xlabel("homozygous deletion response")
    c.set_ylabel("heterozygous deletion response")
    c.set_title(f"Hoepfner paired cells, n={len(arm):,}", loc="left", pad=3)

    for ax in (a, b, c):
        box(ax)
    fig.tight_layout(w_pad=1.5, rect=LETTER_RECT)
    for ax, letter in zip((a, b, c), "abc", strict=True):
        panel_label(ax, letter)
    save(fig, "store_reliability", stable)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-table", default=DEFAULT_CELLS)
    parser.add_argument("--results", default=RESULTS_DIR)
    parser.add_argument("--stable", action="store_true")
    args = parser.parse_args()

    # after every import: importing torchcell can reset the global style
    apply_paper_style()
    table = pd.read_parquet(
        args.cell_table,
        columns=[
            "index",
            "dataset",
            "query_gene",
            "functional_dose",
            "inchikeys",
            "conc_values",
            "n_compounds",
            "responses",
        ],
    )
    labels = pd.read_parquet(LABEL_TABLE).rename(
        columns={"environment_response": "label"}
    )
    table = table.merge(labels, on="index", how="left", validate="one_to_one")
    table["mean_response"] = table["responses"].map(np.mean)

    fig_store_units(args.results, args.stable)
    fig_standardized_target(table, args.results, args.stable)
    fig_store_reliability(table, args.results, args.stable)


if __name__ == "__main__":
    main()
