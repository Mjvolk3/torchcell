# experiments/012-sameith-kemmeren/scripts/single_mutant_expression_distributions
# [[experiments.012-sameith-kemmeren.scripts.single_mutant_expression_distributions]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/012-sameith-kemmeren/scripts/single_mutant_expression_distributions
# Test file: experiments/012-sameith-kemmeren/scripts/test_single_mutant_expression_distributions.py

"""Per-knockout genome-wide expression distributions (Option A, Nature-sized).

For every single-gene deletion we look at the distribution of log2 fold-changes
across the whole transcriptome (~6.1K measured genes). Two panels, each a
full-width strip (179 mm x 35.7 mm -- the canonical wide-strip cell of
``notes/assets/drawio/figure-sizing-template.drawio.svg``), strains ranked by how
much their transcriptome moves (IQR, quiet -> disruptive):

**Both panels are the same *sorted spread band*** (per-strain quantiles vs rank: the
IQR band Q1-Q3 inside a 5-95% band, black median line) on **one shared y-scale**, so
their spread is directly comparable when the strips are stacked. Kemmeren 2014 has
1,484 deletions (box-per-strain would be 0.12 mm/box, invisible); Sameith 2015 has 82
GSTF deletions. Drawing Sameith as the matching band rather than boxes keeps the pair a
consistent chart type / coloration on the shared scale.

Repo figure standards (CLAUDE.md "Figure & Plotting Standards" +
[[paper.nature-biotech.figures]]): palette red (``PLOT_PALETTE[1]``) as the 012
lead color, Arial 6 pt, boxed axes, ``full`` panel width, true-size SVG for
draw.io + a 300-dpi PNG. Titles/panel letters are added at draw.io composition,
so the panels stay title-free; the threshold breakdown lives in the note caption
and the results CSV.
"""

import logging
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from torchcell.datasets.scerevisiae.kemmeren2014 import MicroarrayKemmeren2014Dataset
from torchcell.datasets.scerevisiae.sameith2015 import SmMicroarraySameith2015Dataset
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    PLOT_PALETTE_FILL,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")

DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "50"))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Panel geometry -- from notes/assets/drawio/figure-sizing-template.drawio.svg:
# the full-width wide strip is 179.4 mm x 35.7 mm (~5:1), the canonical single-row
# panel cell (width is the strict PANEL_WIDTHS_MM["full"]; 35.7 mm is the height unit).
PANEL_H_MM = 35.7
RED = PLOT_PALETTE[1]  # #B85450 -- the 012 lead color
RED_FILL = PLOT_PALETTE_FILL[1]  # #F8CECC -- lighter member for the outer band
INK = "#000000"
GRID = "#4A4A4A"
THRESHOLDS = (0.25, 0.5, 0.75, 1.0)


def _apply_rc() -> None:
    """Arial 6 pt, editable SVG text, thin axes -- the repo figure standard."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6,
            "axes.titlesize": 6,
            "axes.labelsize": 6,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "svg.fonttype": "none",
            "axes.linewidth": 0.5,
            # importing torchcell.datasets flips savefig.bbox -> "tight" globally, which
            # re-crops at save time and defeats the strict full-width template (panels
            # came out ~182 mm instead of 179 mm). Pin it back so the true-size SVG is
            # exactly PANEL_WIDTHS_MM["full"] x PANEL_H_MM.
            "savefig.bbox": None,
        }
    )


def extract_expression_data(dataset, dataset_name, sample_range=None):
    """Extract genome-wide log2 ratios per deletion.

    Returns ``{systematic_gene_name: np.ndarray of log2_ratios}`` (NaNs dropped).
    """
    logger.info(f"Extracting expression data from {dataset_name}")
    data_dict = {}
    iter_range = range(
        len(dataset) if sample_range is None else min(sample_range, len(dataset))
    )

    for i in tqdm(iter_range, desc=f"Processing {dataset_name}"):
        data = dataset[i]
        perturbations = data["experiment"]["genotype"]["perturbations"]
        if len(perturbations) != 1:
            continue
        gene_deleted = perturbations[0]["systematic_gene_name"]
        values = np.asarray(
            list(data["experiment"]["phenotype"]["expression_log2_ratio"].values()),
            dtype=float,
        )
        values = values[~np.isnan(values)]
        data_dict[gene_deleted] = values

    logger.info(f"Extracted data for {len(data_dict)} unique genes")
    return data_dict


def _strain_quantiles(data_dict):
    """Per-strain quantile table, sorted by IQR (quiet -> disruptive).

    Returns a DataFrame indexed by rank with columns p5, q1, median, q3, p95, iqr,
    n_genes and the fraction of genes beyond each THRESHOLD.
    """
    rows = []
    for gene, values in data_dict.items():
        p5, q1, med, q3, p95 = np.percentile(values, [5, 25, 50, 75, 95])
        row = {
            "gene": gene,
            "p5": p5,
            "q1": q1,
            "median": med,
            "q3": q3,
            "p95": p95,
            "iqr": q3 - q1,
            "n_genes": values.size,
        }
        for t in THRESHOLDS:
            row[f"frac_gt_{t}"] = float(np.mean(np.abs(values) > t))
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("iqr").reset_index(drop=True)
    return df


def _threshold_caption(data_dict):
    """Pooled % of gene measurements beyond each threshold (for the note caption)."""
    allv = np.concatenate([v for v in data_dict.values()])
    parts = [
        f"|log2FC|>{t}: {100 * np.mean(np.abs(allv) > t):.1f}%" for t in THRESHOLDS
    ]
    return "  ".join(parts)


def _style_axes(ax, ymax):
    """Box all four spines, symmetric y-limits, zero line, light y-grid."""
    ax.set_ylim(-ymax, ymax)
    ax.axhline(0, color=GRID, linestyle=":", linewidth=0.5, zorder=1)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_color(INK)
        ax.spines[s].set_linewidth(0.5)
    ax.tick_params(colors=INK, width=0.5, length=2)
    ax.grid(True, axis="y", alpha=0.15, linewidth=0.4, color=GRID)
    ax.set_axisbelow(True)


def _new_strip():
    """A full-width x 35.7 mm strip figure + axes (true-size template cell)."""
    fig, ax = plt.subplots(
        figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(PANEL_H_MM)),
        constrained_layout=True,
    )
    return fig, ax


def _save(fig, output_prefix):
    output_dir = osp.join(ASSET_IMAGES_DIR, "012-sameith-kemmeren-expression")
    os.makedirs(output_dir, exist_ok=True)
    png_path = osp.join(output_dir, f"{output_prefix}.png")
    svg_path = osp.join(output_dir, f"{output_prefix}.svg")
    fig.savefig(png_path, dpi=300, facecolor="white")
    savefig_true_size_svg(fig, svg_path, facecolor="white")
    plt.close(fig)
    logger.info(f"✓ Saved: {png_path} + {osp.basename(svg_path)}")


def _band_ymax(df):
    """Symmetric y-limit that contains a strain-quantile frame's 5-95% band."""
    return float(np.ceil(max(df["p95"].abs().max(), df["p5"].abs().max()) * 10) / 10)


def plot_spread_band(df, xlabel, output_prefix, ymax):
    """Per-strain quantile bands vs rank, sorted by IQR (quiet -> disruptive).

    Used for BOTH panels so they are the same chart type and coloration; ``ymax`` is
    passed in (shared across the two datasets) so the spread is directly comparable when
    the panels are stacked. 1,484 box-per-strain would be 0.12 mm/box, and at 82 boxes
    Sameith's spread reads better as the matching band than as boxes on the shared scale.
    """
    logger.info(f"Plotting sorted spread band: {output_prefix}")
    _apply_rc()
    fig, ax = _new_strip()
    x = np.arange(len(df))

    # two-level red band: 5-95% (lighter fill) with the IQR (line color) inside it,
    # then the median line in black -- the sanctioned light/dark pairing.
    ax.fill_between(x, df["p5"], df["p95"], facecolor=RED_FILL, linewidth=0, zorder=2)
    ax.fill_between(x, df["q1"], df["q3"], facecolor=RED, linewidth=0, zorder=3)
    ax.plot(x, df["median"], color=INK, linewidth=0.5, zorder=4)

    _style_axes(ax, ymax)
    ax.set_xlim(0, len(df) - 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("log2 fold-change")
    _save(fig, output_prefix)


def main():
    logger.info("=" * 80)
    logger.info("PER-KNOCKOUT GENOME-WIDE EXPRESSION DISTRIBUTIONS (Option A)")
    logger.info("=" * 80)
    sample_range = SAMPLE_SIZE if DEBUG_MODE else None

    kemmeren = MicroarrayKemmeren2014Dataset(
        root=osp.join(DATA_ROOT, "data/torchcell/microarray_kemmeren2014"), io_workers=0
    )
    logger.info(f"Loaded Kemmeren2014: {len(kemmeren)} experiments")
    sm_sameith = SmMicroarraySameith2015Dataset(
        root=osp.join(DATA_ROOT, "data/torchcell/sm_microarray_sameith2015"),
        io_workers=0,
    )
    logger.info(f"Loaded SmSameith2015: {len(sm_sameith)} experiments")

    kemmeren_data = extract_expression_data(kemmeren, "Kemmeren2014", sample_range)
    sameith_data = extract_expression_data(sm_sameith, "SmSameith2015", sample_range)

    kemmeren_df = _strain_quantiles(kemmeren_data)
    sameith_df = _strain_quantiles(sameith_data)

    logger.info(f"Kemmeren pooled thresholds: {_threshold_caption(kemmeren_data)}")
    logger.info(f"Sameith  pooled thresholds: {_threshold_caption(sameith_data)}")

    # one shared y-scale across both panels so the spread is directly comparable when
    # the two strips are stacked (Kemmeren's disruptive tail sets the ceiling).
    shared_ymax = max(_band_ymax(kemmeren_df), _band_ymax(sameith_df))
    logger.info(f"Shared y-scale: +/-{shared_ymax}")
    plot_spread_band(
        kemmeren_df,
        "Kemmeren deletion strains, ranked by transcriptome spread (n = "
        f"{len(kemmeren_df)})",
        "single_mutant_kemmeren",
        shared_ymax,
    )
    plot_spread_band(
        sameith_df,
        f"Sameith GSTF deletion strains, ranked by transcriptome spread (n = {len(sameith_df)})",
        "single_mutant_sameith",
        shared_ymax,
    )

    results_dir = osp.join(EXPERIMENT_ROOT, "012-sameith-kemmeren/results")
    os.makedirs(results_dir, exist_ok=True)
    kemmeren_df.to_csv(
        osp.join(results_dir, "single_mutant_kemmeren_spread.csv"), index=False
    )
    sameith_df.to_csv(
        osp.join(results_dir, "single_mutant_sameith_spread.csv"), index=False
    )

    logger.info("=" * 80)
    logger.info("✓ COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
