# experiments/010-kuzmin-tmi/scripts/plate_prediction_figures.py
# [[experiments.010-kuzmin-tmi.scripts.plate_prediction_figures]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/plate_prediction_figures
"""Figures for the corrected predictions over the wet-lab plate.

Reads ``results/wetlab_plate_rescored.csv``, written by
``rescore_wetlab_plate.py``, which scores every combination over the 11 distinct
loci on the plate with all three checkpoints under the correct 6,607-gene index
space, and also under the shifted 6,579-gene space the original run used.

Four figures, each answering one question a person actually has in front of a
plate:

    A  what changed        corrected against as-run, per triple
    B  which pair drives   mean over the third gene, as an 11 x 11 matrix
    C  which gene drives   per-gene distribution of the triples containing it
    D  how sure are we     the three checkpoints shown separately, top and bottom

Panel A is the one that says whether the old ranking can be reused. Panel D is
the one that says how far down the ranking is worth reading.
"""

import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.colors import TwoSlopeNorm

from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    apply_paper_style,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()

EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi", "results")
IMAGE_DIR = osp.join(ASSET_IMAGES_DIR, "010-kuzmin-tmi")

CHECKPOINTS = ["M01_lzs9pcj3", "M02_yv4r30bi", "M03_c7671wgj"]
# The label-to-locus collapse the plate needs, stated where a reader will see it.
DUPLICATE_NOTE = "SPH1 and YLR312C-B are one locus (YLR313C)"


def style() -> None:
    apply_paper_style()


def save(fig, stem: str) -> None:
    os.makedirs(IMAGE_DIR, exist_ok=True)
    path = osp.join(IMAGE_DIR, stem)
    fig.savefig(path + ".png", dpi=300)
    savefig_true_size_svg(fig, path + ".svg")
    print(f"wrote {path}.svg")
    plt.close(fig)


def box(ax) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.grid(which="both", linewidth=0.3, color="0.85")
    ax.set_axisbelow(True)


def panel_a_what_changed(tri: pd.DataFrame) -> None:
    """Corrected against what inference_3 recorded, so the old list can be judged."""
    have = tri["stored_inference_3"].notna()
    fig, ax = plt.subplots(
        figsize=(mm_to_in(PANEL_WIDTHS_MM["half_plus"]), mm_to_in(72.0))
    )
    ax.axhline(0, color="0.6", linewidth=0.4, linestyle=":")
    ax.axvline(0, color="0.6", linewidth=0.4, linestyle=":")
    ax.errorbar(
        tri.loc[have, "stored_inference_3"],
        tri.loc[have, "mean"],
        yerr=[
            tri.loc[have, "mean"] - tri.loc[have, "min"],
            tri.loc[have, "max"] - tri.loc[have, "mean"],
        ],
        fmt="o",
        ms=2.2,
        mfc=PLOT_PALETTE[0],
        mec="black",
        mew=0.3,
        ecolor=PLOT_PALETTE[5],
        elinewidth=0.4,
        capsize=0,
        linestyle="none",
        label=f"recorded by inference_3 ({int(have.sum())})",
    )
    missing = (~have).sum()
    if missing:
        ax.scatter(
            np.full(int(missing), np.nan),
            np.full(int(missing), np.nan),
            s=5,
            color=PLOT_PALETTE[1],
            label=f"no inference_3 record ({int(missing)})",
        )
    r = float(tri.loc[have, ["stored_inference_3", "mean"]].corr().iloc[0, 1])
    ax.set_xlabel(r"What inference_3 recorded (invalid, shifted index)")
    ax.set_ylabel(r"Corrected (predicted $\tau$, mean of 3 checkpoints)")
    ax.set_title(
        f"A  Corrected against what the panel was chosen on\n"
        f"Pearson {r:+.3f} over {int(have.sum())} shared combinations",
        fontsize=6,
    )
    ax.legend(frameon=False, loc="upper left", fontsize=5)
    box(ax)
    fig.tight_layout()
    save(fig, "plate_fig_a_what_changed")


def panel_b_pair_matrix(tri: pd.DataFrame, genes: list[str]) -> None:
    """Mean predicted tau over the third gene, for every pair on the plate."""
    n = len(genes)
    pos = {g: i for i, g in enumerate(genes)}
    total = np.zeros((n, n))
    count = np.zeros((n, n))
    for row in tri.itertuples():
        parts = row.plate_label.split("+")
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = pos[parts[i]], pos[parts[j]]
                total[a, b] += row.mean_pred
                total[b, a] += row.mean_pred
                count[a, b] += 1
                count[b, a] += 1
    with np.errstate(invalid="ignore"):
        mat = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    np.fill_diagonal(mat, np.nan)

    lim = float(np.nanmax(np.abs(mat)))
    fig, ax = plt.subplots(
        figsize=(mm_to_in(PANEL_WIDTHS_MM["half_plus"]), mm_to_in(80.0))
    )
    im = ax.imshow(
        mat, cmap="RdYlBu_r", norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    )
    ax.set_xticks(range(n))
    ax.set_xticklabels(genes, rotation=90, fontsize=5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(genes, fontsize=5)
    ax.set_title(
        "B  Mean predicted $\\tau$ over the third gene\n" + DUPLICATE_NOTE, fontsize=6
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=5)
    cb.outline.set_linewidth(0.5)
    for spine in ax.spines.values():
        spine.set_visible(True)
    fig.tight_layout()
    save(fig, "plate_fig_b_pair_matrix")


def panel_c_per_gene(tri: pd.DataFrame, genes: list[str]) -> None:
    """Every triple containing each gene, so a single driver is visible."""
    groups = {
        g: tri[tri["plate_label"].str.split("+").apply(lambda p: g in p)][
            "mean_pred"
        ].to_numpy()
        for g in genes
    }
    order = sorted(genes, key=lambda g: float(np.median(groups[g])))
    fig, ax = plt.subplots(
        figsize=(mm_to_in(PANEL_WIDTHS_MM["half_plus"]), mm_to_in(66.0))
    )
    ax.axvline(0, color="black", linewidth=0.5)
    for i, g in enumerate(order):
        vals = groups[g]
        jitter = (np.random.default_rng(i).random(vals.size) - 0.5) * 0.5
        ax.scatter(
            vals,
            np.full(vals.size, i) + jitter,
            s=3,
            color=PLOT_PALETTE[0],
            edgecolor="black",
            linewidth=0.2,
            alpha=0.85,
        )
        ax.plot(
            [np.median(vals)], [i], marker="|", ms=9, color="black", markeredgewidth=1.0
        )
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=5)
    ax.set_xlabel(r"Predicted $\tau$ of every triple containing this gene")
    ax.set_title(
        "C  Per-gene distribution, ordered by median\nvertical bar is the median",
        fontsize=6,
    )
    box(ax)
    fig.tight_layout()
    save(fig, "plate_fig_c_per_gene")


def panel_d_checkpoint_agreement(tri: pd.DataFrame) -> None:
    """The three checkpoints separately, which is what says how far to trust the order."""
    top = pd.concat([tri.head(12), tri.tail(12)])
    fig, ax = plt.subplots(
        figsize=(mm_to_in(PANEL_WIDTHS_MM["half_plus"]), mm_to_in(88.0))
    )
    y = np.arange(len(top))[::-1]
    marks = ["o", "s", "^"]
    for k, tag in enumerate(CHECKPOINTS):
        ax.scatter(
            top[tag],
            y,
            s=8,
            marker=marks[k],
            facecolor=PLOT_PALETTE[k],
            edgecolor="black",
            linewidth=0.3,
            label=tag.split("_")[0],
            zorder=3,
        )
    for yy, (_, row) in zip(y, top.iterrows()):
        ax.plot(
            [row["min"], row["max"]], [yy, yy], color="0.6", linewidth=0.6, zorder=1
        )
    ax.axvline(0, color="black", linewidth=0.5)
    ax.axhline(len(top) - 12.5, color="0.4", linewidth=0.5, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(top["plate_label"], fontsize=5)
    ax.set_xlabel(r"Predicted $\tau$, each checkpoint separately")
    agree = int(tri["sign_agree"].sum())
    ax.set_title(
        f"D  Top 12 and bottom 12, three checkpoints\n"
        f"all three agree on sign for {agree} of {len(tri)} triples",
        fontsize=6,
    )
    ax.legend(frameon=False, loc="lower right", fontsize=5)
    box(ax)
    fig.tight_layout()
    save(fig, "plate_fig_d_checkpoint_agreement")


def main() -> None:
    df = pd.read_csv(osp.join(RESULTS_DIR, "wetlab_plate_rescored.csv"))
    tri = df[df["order"] == 3].copy()
    # `mean` collides with the DataFrame method under itertuples, so rename once.
    tri = tri.rename(columns={"mean": "mean_pred"})
    tri["mean"] = tri["mean_pred"]
    tri = tri.sort_values("mean_pred", ascending=False).reset_index(drop=True)

    genes = sorted({g for label in tri["plate_label"] for g in label.split("+")})
    print(f"{len(tri)} triples over {len(genes)} loci: {', '.join(genes)}")
    print(
        f"predicted tau range {tri['mean_pred'].min():+.4f} to "
        f"{tri['mean_pred'].max():+.4f}"
    )

    style()
    panel_a_what_changed(tri)
    panel_b_pair_matrix(tri, genes)
    panel_c_per_gene(tri, genes)
    panel_d_checkpoint_agreement(tri)


if __name__ == "__main__":
    main()
