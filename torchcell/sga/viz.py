# torchcell/sga/viz.py
# [[torchcell.sga.viz]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/sga/viz
"""SGAtools-style data-analysis views: plate heatmap, value histogram, and a
per-strain fitness plot.

Heatmaps use a perceptually-uniform sequential colormap (position -> value is a
magnitude, not a category). The per-strain fitness plot is categorical and uses
the repo palette (``torchcell.utils.PLOT_PALETTE``). Type is Arial 6 pt per the
repo figure standard; heatmaps are exploratory PNGs (timestamped by the caller).
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Colormap
from matplotlib.figure import Figure

from torchcell.sga.models import NormalizationConfig, ScoreReport
from torchcell.utils import PLOT_PALETTE

# Sequential colormap for all heatmaps: matplotlib "magma" -- a perceptually
# uniform ramp (dark low -> bright high), green-free, so magnitude reads
# unambiguously. Chosen over a custom warm ramp (whose light-to-dark direction was
# read as white-hot/ambiguous) per the repo figure standard.
SEQUENTIAL_CMAP = plt.get_cmap("magma")

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 6,
        "svg.fonttype": "none",
        "axes.linewidth": 0.5,
    }
)


def _grid(df: pd.DataFrame, value_col: str) -> np.ndarray:
    n_rows, n_cols = int(df["row"].max()), int(df["col"].max())
    grid = np.full((n_rows, n_cols), np.nan)
    grid[df["row"].to_numpy() - 1, df["col"].to_numpy() - 1] = df[value_col].to_numpy()
    return grid


def plate_heatmap(
    df: pd.DataFrame,
    value_col: str = "norm",
    title: str = "",
    cmap: str | Colormap = SEQUENTIAL_CMAP,
    vmin: float | None = None,
    vmax: float | None = None,
    divider_after_col: int | None = None,
    half_labels: tuple[str, str] | None = None,
) -> Figure:
    """Colony values laid out in true plate geometry (SGAtools heatmap view).

    ``vmin``/``vmax`` fix the colorbar range (e.g. ``vmin=0`` to anchor at zero).
    ``divider_after_col`` draws a vertical line just after that plate column (to
    mark a block boundary, e.g. a volume split); ``half_labels`` labels the two
    sides above the plate.
    """
    grid = _grid(df, value_col)
    n_rows, n_cols = grid.shape
    fig, ax = plt.subplots(figsize=(max(4.0, n_cols * 0.3), max(2.5, n_rows * 0.3)))
    im = ax.imshow(
        grid, cmap=cmap, aspect="equal", origin="upper", vmin=vmin, vmax=vmax
    )
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([str(i) for i in range(1, n_cols + 1)])
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([chr(ord("A") + i) for i in range(n_rows)])
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    ax.set_title(title)
    if divider_after_col is not None:
        xline = divider_after_col - 0.5  # plate col c -> x index c-1; line after c
        ax.axvline(xline, color="#1a1a1a", linewidth=1.6)
        if half_labels is not None:
            ax.text(
                (xline) / 2,
                -0.9,
                half_labels[0],
                ha="center",
                va="bottom",
                fontsize=6,
                fontweight="bold",
            )
            ax.text(
                (xline + n_cols) / 2,
                -0.9,
                half_labels[1],
                ha="center",
                va="bottom",
                fontsize=6,
                fontweight="bold",
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(value_col)
    fig.tight_layout()
    return fig


def layout_heatmap(df: pd.DataFrame, title: str = "strain layout") -> Figure:
    """Categorical map of which strain sits at each well (decoded picklist)."""
    strains = sorted(df["strain"].dropna().unique())
    code = {s: i for i, s in enumerate(strains)}
    n_rows, n_cols = int(df["row"].max()), int(df["col"].max())
    grid = np.full((n_rows, n_cols), np.nan)
    for _, r in df.iterrows():
        if pd.notna(r["strain"]):
            grid[int(r["row"]) - 1, int(r["col"]) - 1] = code[r["strain"]]
    colors = [PLOT_PALETTE[i % len(PLOT_PALETTE)] for i in range(len(strains))]
    cmap = matplotlib.colors.ListedColormap(colors)
    fig, ax = plt.subplots(figsize=(max(4.0, n_cols * 0.3), max(2.5, n_rows * 0.3)))
    ax.imshow(
        grid, cmap=cmap, aspect="equal", origin="upper", vmin=0, vmax=len(strains) - 1
    )
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([str(i) for i in range(1, n_cols + 1)])
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([chr(ord("A") + i) for i in range(n_rows)])
    ax.set_title(title)
    handles = [
        matplotlib.patches.Patch(
            facecolor=colors[i], edgecolor="black", lw=0.4, label=s
        )
        for i, s in enumerate(strains)
    ]
    ax.legend(
        handles=handles,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
        fontsize=5,
    )
    fig.tight_layout()
    return fig


def value_histogram(
    df: pd.DataFrame, value_col: str = "norm", title: str = ""
) -> Figure:
    """Distribution of a value across colonies (SGAtools histogram view)."""
    vals = df[value_col].dropna()
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.hist(vals, bins=30, color=PLOT_PALETTE[4], edgecolor="black", linewidth=0.4)
    ax.set_xlabel(value_col)
    ax.set_ylabel("colonies")
    ax.set_title(title)
    for spine in ax.spines.values():
        spine.set_visible(True)
    fig.tight_layout()
    return fig


def colony_shape_by_volume(
    df: pd.DataFrame, cfg: NormalizationConfig | None = None
) -> Figure:
    """Two panels: (left) circularity distribution per volume, (right) circularity
    vs colony size per volume -- the scatter exposes whether low circularity is
    just small colonies (measurement bias) or genuine spreading at high volume.
    """
    from torchcell.sga.models import NormalizationConfig

    cfg = cfg or NormalizationConfig()
    plated = df[~df["is_blank"] & ~df["is_missing"]]
    nogash = plated[~plated["flags"].fillna("").str.contains("S")]
    vols = sorted(nogash["volume_nl"].dropna().unique())
    fig, (axb, axs) = plt.subplots(1, 2, figsize=(5.4, 2.6))

    data = [
        nogash[nogash["volume_nl"] == v]["circularity"].dropna().to_numpy()
        for v in vols
    ]
    bp = axb.boxplot(
        data, labels=[f"{v} nL" for v in vols], patch_artist=True, widths=0.6
    )
    for i, box in enumerate(bp["boxes"]):
        box.set(facecolor=PLOT_PALETTE[i], edgecolor="black", linewidth=0.5)
    for med in bp["medians"]:
        med.set(color="black", linewidth=0.8)
    axb.set_ylabel("circularity (1 = round)")
    axb.set_title("shape by volume")

    for i, v in enumerate(vols):
        s = nogash[nogash["volume_nl"] == v]
        axs.scatter(
            s["size"],
            s["circularity"],
            s=8,
            color=PLOT_PALETTE[i],
            edgecolor="black",
            linewidth=0.2,
            alpha=0.8,
            label=f"{v} nL",
        )
    axs.set_xlabel("colony size (px)")
    axs.set_ylabel("circularity")
    axs.set_title("circularity vs size")
    axs.legend(frameon=False, fontsize=5)
    for ax in (axb, axs):
        for sp in ax.spines.values():
            sp.set_visible(True)
    fig.tight_layout()
    return fig


def strain_fitness_plot(report: ScoreReport, alpha: float = 0.05) -> Figure:
    """Per-strain relative fitness vs the on-plate wild-type, sorted, with
    significance (filled = MWU p < alpha vs WT).
    """
    rows = [
        s
        for s in report.strains
        if s.strain != report.blank_name and s.relative_fitness is not None
    ]
    rows.sort(
        key=lambda s: s.relative_fitness if s.relative_fitness is not None else 0.0
    )
    names = [s.strain for s in rows]
    fig, ax = plt.subplots(figsize=(max(3.4, len(rows) * 0.28), 2.6))
    for i, s in enumerate(rows):
        val = s.relative_fitness
        assert val is not None  # rows filtered to non-None relative_fitness above
        is_wt = s.strain == report.wt_name
        sig = s.pvalue is not None and s.pvalue < alpha
        color = PLOT_PALETTE[5] if is_wt else PLOT_PALETTE[1]
        ax.bar(
            i,
            val,
            color=color if sig or is_wt else "white",
            edgecolor=color,
            linewidth=0.8,
        )
    ax.axhline(1.0, color=PLOT_PALETTE[5], linestyle="--", linewidth=0.6)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90)
    ax.set_ylabel("relative fitness (vs BY4741)")
    ax.set_title(f"{report.plate_id}: single-KO fitness")
    for spine in ax.spines.values():
        spine.set_visible(True)
    fig.tight_layout()
    return fig
