# experiments/028-gene-essentiality/scripts/module_portability_figure.py
# [[experiments.028-gene-essentiality.scripts.module_portability_figure]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/module_portability_figure

"""The second goal on one figure: a metabolic module that fits one record per experiment.

A record in the database is one genotype in one environment, so a portable module reads
one medium, and at database scale it solves nothing inside the training loop. Three
panels share one list of module forms, read from ``results/simplification_readout.json``
(written by ``simplification_readout.py``):

- **a** the released split: the seed-mean held-out AUROC of each form (bar, one point per
  seed) and its five-fold ensemble (diamond);
- **b** the ten repeated outer splits: each form minus the same cell with no metabolism,
  one point per split and the mean over splits (bar);
- **c** the share of (split, seed) pairs on which the form scores above the cell with no
  metabolism, 0.5 being a module that helps on half the runs.

Color follows the summary figure: orange a form that reads the offline solve, yellow a
learned form that reads no solve. The form that meets both constraints is the first row.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD \
        python experiments/028-gene-essentiality/scripts/module_portability_figure.py
"""

from __future__ import annotations

import json
import os
import os.path as osp

import matplotlib
import numpy as np
from dotenv import load_dotenv

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

from torchcell.timestamp import timestamp  # noqa: E402
from torchcell.utils import (  # noqa: E402
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
RESULTS = osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "results")
IMAGES = osp.join(ASSET_IMAGES_DIR, "028-gene-essentiality")

NO_SOLVE = PLOT_PALETTE[3]
SOLVE = PLOT_PALETTE[0]
WIDTH_MM = PANEL_WIDTHS_MM["full"]
HEIGHT_MM = 62.0

#: Row label -> (ladder key, name of the repeated-split pair against no metabolism).
FORMS: list[tuple[str, str, str | None]] = [
    (
        "1 medium: ratio, flux vector, rule counts",
        "one_solve_counts",
        "one medium, solve and counts, over no metabolism",
    ),
    (
        "4 media: ratios, flux vector, rule counts",
        "four_solve_counts",
        "four media, solve and counts, over no metabolism",
    ),
    (
        "4 media: ratios, flux vector",
        "four_solve",
        "four media and the solve, over no metabolism",
    ),
    (
        "4 media: ratios, rule counts",
        "four_counts",
        "four media and the counts, over no metabolism",
    ),
    (
        "2 media: ratios, rule counts",
        "two_counts",
        "two media and the counts, over no metabolism",
    ),
    (
        "1 medium: ratio, rule counts",
        "one_counts",
        "one medium and the counts, over no metabolism",
    ),
    ("4 media: ratios alone", "four_only", "four ratios alone, over no metabolism"),
    ("no metabolism", "none", None),
]


def place(fig: Figure, x: float, top: float, w: float, h: float) -> Axes:
    """An axes whose box is given in millimeters from the figure's top-left corner."""
    ax = fig.add_axes(
        (x / WIDTH_MM, 1.0 - (top + h) / HEIGHT_MM, w / WIDTH_MM, h / HEIGHT_MM)
    )
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    ax.tick_params(width=0.5, length=2)
    return ax


def letter(fig: Figure, x: float, top: float, text: str) -> None:
    fig.text(
        x / WIDTH_MM,
        1.0 - (top - 1.0) / HEIGHT_MM,
        text,
        fontsize=8,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "svg.fonttype": "none",
            "axes.linewidth": 0.5,
            "axes.titlesize": 6,
        }
    )
    with open(osp.join(RESULTS, "simplification_readout.json")) as fh:
        report = json.load(fh)
    ladder = {r["key"]: r for r in report["ladder"]}
    pairs = {r["name"]: r for r in report["resplit"]}

    fig = plt.figure(figsize=(mm_to_in(WIDTH_MM), mm_to_in(HEIGHT_MM)))
    y = np.arange(len(FORMS))[::-1]
    colors = [SOLVE if ladder[key]["solver"] else NO_SOLVE for _, key, _ in FORMS]

    handles = [
        Patch(
            facecolor=SOLVE, edgecolor="black", lw=0.4, label="reads the offline solve"
        ),
        Patch(facecolor=NO_SOLVE, edgecolor="black", lw=0.4, label="reads no solve"),
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            ms=2.2,
            mfc="white",
            mec="black",
            mew=0.4,
            label="one seed (a) or one split (b)",
        ),
        Line2D(
            [],
            [],
            marker="D",
            ls="",
            ms=2.8,
            mfc="black",
            mec="black",
            label="five-fold ensemble",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=4,
        frameon=False,
        handlelength=1.2,
        columnspacing=1.6,
        borderaxespad=0.2,
    )

    top, height = 12.0, 40.0
    # a: the released split.
    ax = place(fig, 50.0, top, 42.0, height)
    for yi, color, (_, key, _) in zip(y, colors, FORMS, strict=True):
        row = ladder[key]
        ax.barh(
            yi,
            row["cell_mean"],
            height=0.62,
            color=color,
            edgecolor="black",
            lw=0.4,
            zorder=2,
        )
        ax.plot(
            row["cell_values"],
            np.full(len(row["cell_values"]), yi),
            "o",
            ms=2.2,
            mfc="white",
            mec="black",
            mew=0.4,
            zorder=3,
        )
        fold = row["fold_ensemble_three_seeds"] or row["fold_ensemble_one_seed"]
        if fold is not None:
            ax.plot(fold, yi, "D", ms=2.8, color="black", zorder=4)
    ax.set_xlim(0.8, 0.96)
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.grid(axis="x", lw=0.3, color="0.85", zorder=0)
    ax.set_axisbelow(True)
    ax.set_yticks(y)
    ax.set_yticklabels([label for label, _, _ in FORMS])
    ax.set_ylim(-0.6, len(FORMS) - 0.4)
    ax.set_xlabel("held-out AUROC, released split")
    ax.set_title("195 released held-out genes", pad=3, loc="left")
    letter(fig, 0.0, top, "a")

    # b: the repeated splits, each form minus no metabolism.
    ax = place(fig, 99.0, top, 42.0, height)
    for yi, color, (_, _, name) in zip(y, colors, FORMS, strict=True):
        if name is None or name not in pairs:
            continue
        row = pairs[name]
        ax.barh(
            yi,
            row["mean_difference"],
            height=0.62,
            color=color,
            edgecolor="black",
            lw=0.4,
            zorder=2,
        )
        ax.plot(
            row["per_unit_difference"],
            np.full(row["n"], yi),
            "o",
            ms=2.2,
            mfc="white",
            mec="black",
            mew=0.4,
            zorder=3,
        )
    ax.axvline(0.0, color="black", lw=0.5, zorder=1)
    ax.set_xlim(-0.03, 0.13)
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.grid(axis="x", lw=0.3, color="0.85", zorder=0)
    ax.set_axisbelow(True)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_ylim(-0.6, len(FORMS) - 0.4)
    ax.set_xlabel("AUROC minus no metabolism, ten repeated splits")
    ax.set_title("gain over the same cell with no metabolism", pad=3, loc="left")
    letter(fig, 95.0, top, "b")

    # c: the share of (split, seed) pairs the form wins.
    ax = place(fig, 147.0, top, 28.0, height)
    for yi, color, (_, _, name) in zip(y, colors, FORMS, strict=True):
        if name is None or name not in pairs:
            continue
        row = pairs[name]
        ax.barh(
            yi,
            row["n_pairs_positive"] / row["n_pairs"],
            height=0.62,
            color=color,
            edgecolor="black",
            lw=0.4,
            zorder=2,
        )
    ax.axvline(0.5, color="black", lw=0.5, ls="--", zorder=1)
    ax.set_xlim(0.0, 1.0)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(axis="x", which="minor", length=0)
    ax.grid(axis="x", which="both", lw=0.3, color="0.85", zorder=0)
    ax.set_axisbelow(True)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_ylim(-0.6, len(FORMS) - 0.4)
    ax.set_xlabel("share of split-and-seed pairs won")
    ax.set_title("how often the module helps", pad=3, loc="left")
    letter(fig, 143.0, top, "c")

    os.makedirs(IMAGES, exist_ok=True)
    stem = osp.join(IMAGES, f"module_portability_{timestamp()}")
    savefig_true_size_svg(fig, stem + ".svg")
    fig.savefig(stem + ".png", dpi=300)
    print(stem + ".svg")


if __name__ == "__main__":
    main()
