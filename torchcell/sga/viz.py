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
from PIL import Image, ImageDraw, ImageFont

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


# --- plate-address overlay labelling (shared by the run scripts + artifact builder) ---


def _overlay_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Bold TrueType at `size`, falling back to PIL's bitmap default."""
    import os.path as osp

    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ):
        if osp.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def plate_labels(op: str, n_rows: int, n_cols: int) -> tuple[list[str], list[str]]:
    """Per-image-index PLATE labels (rows A-P, cols 1-24) under the resolved orientation.

    The overlay is drawn in IMAGE order, but the well a colony belongs to is its PLATE
    address -- ``r2.apply_orientation`` maps image (row, col) -> plate (row, col). All four
    ops keep rows and columns separable (each either preserved or reversed), so the mapping
    collapses to a per-axis label list. Without this the headers would silently show image
    coordinates on any plate that resolves to rot180/flip_v/flip_h.
    """
    rows_rev = op in ("rot180", "flip_v")  # nr = n_rows + 1 - r
    cols_rev = op in ("rot180", "flip_h")  # nc = n_cols + 1 - c
    rows = [
        chr(ord("A") + (n_rows - 1 - ri if rows_rev else ri)) for ri in range(n_rows)
    ]
    cols = [str(n_cols - ci if cols_rev else ci + 1) for ci in range(n_cols)]
    return rows, cols


def label_plate_overlay(
    overlay_path: str, nodes: np.ndarray, op: str = "identity"
) -> None:
    """Add 384-well axis labels to a detection overlay, matplotlib-style.

    The plate image is matted into a white margin and the headers are drawn OUTSIDE it --
    rows A-P down the left and right, columns 1-24 across the top and bottom, each aligned
    to its fitted grid line and joined to the image edge by a short tick. Keeping them off
    the plate is the point: drawn in-image they collide with edge-row colonies, and no
    amount of halo makes small glyphs legible over agar, plastic and the dark lid at once.

    Headers are PLATE addresses under the resolved orientation ``op`` (see ``plate_labels``),
    not raw image indices. Grid nodes are marked on the plate itself (cyan crosses), so a
    mis-fit node -- sitting off its colony -- stays visible.
    """
    im = Image.open(overlay_path).convert("RGB")
    n_rows, n_cols, _ = nodes.shape
    row_lab, col_lab = plate_labels(op, n_rows, n_cols)
    w, h = im.width, im.height

    # grid nodes go on the plate image itself
    dp = ImageDraw.Draw(im)
    for ri in range(n_rows):
        for ci in range(n_cols):
            y, x = float(nodes[ri, ci, 0]), float(nodes[ri, ci, 1])
            dp.line([(x - 5, y), (x + 5, y)], fill=(0, 255, 255), width=1)
            dp.line([(x, y - 5), (x, y + 5)], fill=(0, 255, 255), width=1)

    fs = max(30, int(w / 45))
    fnt = _overlay_font(fs)
    pad = int(2.2 * fs)  # margin band, sized to hold the glyphs + tick
    tick = max(3, fs // 4)
    canvas = Image.new("RGB", (w + 2 * pad, h + 2 * pad), (255, 255, 255))
    canvas.paste(im, (pad, pad))
    d = ImageDraw.Draw(canvas)
    blk = (0, 0, 0)

    def ctext(cx: float, cy: float, s: str) -> None:
        x0, y0, x1, y1 = d.textbbox((0, 0), s, font=fnt)
        d.text(
            (cx - (x1 - x0) / 2 - x0, cy - (y1 - y0) / 2 - y0), s, fill=blk, font=fnt
        )

    # columns 1..24 across the TOP and BOTTOM margins
    for ci in range(n_cols):
        xt = pad + float(nodes[0, ci, 1])
        xb = pad + float(nodes[-1, ci, 1])
        d.line([(xt, pad - tick), (xt, pad)], fill=blk, width=2)
        d.line([(xb, pad + h), (xb, pad + h + tick)], fill=blk, width=2)
        ctext(xt, pad - tick - fs * 0.7, col_lab[ci])
        ctext(xb, pad + h + tick + fs * 0.7, col_lab[ci])
    # rows A..P down the LEFT and RIGHT margins
    for ri in range(n_rows):
        yl = pad + float(nodes[ri, 0, 0])
        yr = pad + float(nodes[ri, -1, 0])
        d.line([(pad - tick, yl), (pad, yl)], fill=blk, width=2)
        d.line([(pad + w, yr), (pad + w + tick, yr)], fill=blk, width=2)
        ctext(pad - tick - fs * 0.7, yl, row_lab[ri])
        ctext(pad + w + tick + fs * 0.7, yr, row_lab[ri])
    canvas.save(overlay_path)
