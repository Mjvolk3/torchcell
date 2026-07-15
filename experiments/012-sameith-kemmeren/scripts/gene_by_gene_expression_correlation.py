# experiments/012-sameith-kemmeren/scripts/gene_by_gene_expression_correlation
# [[experiments.012-sameith-kemmeren.scripts.gene_by_gene_expression_correlation]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/012-sameith-kemmeren/scripts/gene_by_gene_expression_correlation
# Test file: experiments/012-sameith-kemmeren/scripts/test_gene_by_gene_expression_correlation.py

"""
Gene-by-Gene Expression Correlation Analysis

For overlapping genes between Kemmeren and Sameith, compares the full
expression profiles (~6K genes) to see if deleting the same gene produces
correlated genome-wide expression changes across the two studies.

This is different from comparing mean expression - we're asking:
"When we delete gene X, do we see correlated expression changes across
the entire genome in both studies?"
"""

import logging
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy import stats
from tqdm import tqdm

# Removed timestamp import - using stable filenames instead
from torchcell.datasets.scerevisiae.kemmeren2014 import MicroarrayKemmeren2014Dataset
from torchcell.datasets.scerevisiae.sameith2015 import SmMicroarraySameith2015Dataset
from torchcell.utils import (
    MAX_HEIGHT_MM,
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    savefig_true_size_svg,
)

# Load environment variables
load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")

# Configuration
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "50"))

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_expression_profiles(dataset, dataset_name, sample_range=None):
    """
    Extract full expression profiles for each gene deletion.

    Returns:
        dict: {gene_name: expression_dict}
              where expression_dict = {measured_gene: log2_ratio}
    """
    logger.info(f"Extracting expression profiles from {dataset_name}")

    data_dict = {}

    if sample_range is None:
        iter_range = range(len(dataset))
    else:
        iter_range = range(min(sample_range, len(dataset)))

    for i in tqdm(iter_range, desc=f"Processing {dataset_name}"):
        data = dataset[i]

        perturbations = data["experiment"]["genotype"]["perturbations"]
        if len(perturbations) != 1:
            continue

        gene_deleted = perturbations[0]["systematic_gene_name"]

        # Get full expression profile (dict of {gene: log2_ratio})
        expression_profile = data["experiment"]["phenotype"]["expression_log2_ratio"]

        # Remove NaN values
        expression_profile = {
            g: v for g, v in expression_profile.items() if not np.isnan(v)
        }

        if gene_deleted in data_dict:
            logger.warning(f"{dataset_name}: Duplicate gene {gene_deleted}")

        data_dict[gene_deleted] = expression_profile

    logger.info(f"Extracted profiles for {len(data_dict)} unique genes")
    return data_dict


def compute_gene_correlations(kemmeren_profiles, sameith_profiles):
    """
    For each overlapping gene, compute correlation between full expression
    profiles.

    Returns:
        pd.DataFrame with columns: gene, pearson_r, pearson_p, spearman_r,
                                    spearman_p, n_common_genes
    """
    logger.info("Computing gene-by-gene expression correlations")

    # Find overlapping genes
    kemmeren_genes = set(kemmeren_profiles.keys())
    sameith_genes = set(sameith_profiles.keys())
    overlap_genes = sorted(list(kemmeren_genes & sameith_genes))

    logger.info(f"Found {len(overlap_genes)} overlapping genes")

    results = []

    for gene in tqdm(overlap_genes, desc="Computing correlations"):
        kemmeren_profile = kemmeren_profiles[gene]
        sameith_profile = sameith_profiles[gene]

        # Find common measured genes
        kemmeren_measured = set(kemmeren_profile.keys())
        sameith_measured = set(sameith_profile.keys())
        common_measured = kemmeren_measured & sameith_measured

        if len(common_measured) < 10:  # Need minimum genes for correlation
            logger.warning(f"{gene}: Only {len(common_measured)} common measured genes")
            continue

        # Extract values for common genes
        common_genes_list = sorted(list(common_measured))
        kemmeren_values = [kemmeren_profile[g] for g in common_genes_list]
        sameith_values = [sameith_profile[g] for g in common_genes_list]

        # Compute correlations
        pearson_r, pearson_p = stats.pearsonr(kemmeren_values, sameith_values)
        spearman_r, spearman_p = stats.spearmanr(kemmeren_values, sameith_values)

        results.append(
            {
                "gene": gene,
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
                "n_common_genes": len(common_measured),
            }
        )

    return pd.DataFrame(results)


def plot_correlation_distribution(df_corr, corr_type, output_prefix):
    """Histogram of the per-deletion cross-study correlation, repo palette + standards.

    One distribution (the 82 per-deletion correlations), so a single hue -- ``PLOT_PALETTE``
    slot 1 (red). Green-free (the old red->white->green gradient violated the palette).
    Boxed axes, Arial 6 pt, ``half`` panel width, true-size SVG for draw.io + a 300-dpi PNG.
    """
    logger.info(f"Creating {corr_type} correlation distribution plot")
    r_col = f"{corr_type}_r"

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
        }
    )
    color = PLOT_PALETTE[1]  # red
    ink = "#000000"

    w = mm_to_in(PANEL_WIDTHS_MM["half"])
    fig, ax = plt.subplots(
        figsize=(w, min(w * 0.72, mm_to_in(MAX_HEIGHT_MM))), constrained_layout=True
    )

    # correlations are all positive here (median ~0.74); a single-hue histogram is honest.
    ax.hist(
        df_corr[r_col],
        bins=np.linspace(-1, 1, 41),
        color=color,
        edgecolor=ink,
        linewidth=0.25,
    )
    median_r = df_corr[r_col].median()
    ax.axvline(median_r, color=ink, linestyle="--", linewidth=1.0)
    ax.axvline(0, color="#4A4A4A", linestyle=":", linewidth=0.6)

    # headroom + median label. Anchor the label on the side of the median line
    # that has room: median r is strongly positive here, so hang its RIGHT edge
    # just left of the line (ha="right", offset left) -- otherwise a right-side
    # median (~0.74) pushes "median r = 0.74" off the right spine.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 1.25)
    on_right = median_r > 0
    ax.annotate(
        f"median r = {median_r:.2f}",
        xy=(median_r, ax.get_ylim()[1] * 0.92),
        xytext=(-4 if on_right else 4, 0),
        textcoords="offset points",
        ha="right" if on_right else "left",
        va="top",
        color=ink,
        fontsize=6,
    )

    ax.set_xlabel(f"{corr_type.capitalize()} r  (Kemmeren vs Sameith, per deletion)")
    ax.set_ylabel("gene deletions")
    ax.set_xlim(-1, 1)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_color(ink)
        ax.spines[s].set_linewidth(0.5)
    ax.tick_params(colors=ink, width=0.5, length=2)
    ax.grid(True, alpha=0.15, linewidth=0.4, color="#4A4A4A")
    ax.set_axisbelow(True)

    n = len(df_corr)
    n5 = int((df_corr[r_col] > 0.5).sum())
    n7 = int((df_corr[r_col] > 0.7).sum())
    ax.annotate(
        f"n = {n} deletions\nmean r = {df_corr[r_col].mean():.2f}\n"
        f"r > 0.5: {100 * n5 / n:.0f}%\nr > 0.7: {100 * n7 / n:.0f}%",
        xy=(0.03, 0.97),
        xycoords="axes fraction",
        va="top",
        ha="left",
        fontsize=6,
        color=ink,
    )
    fig.suptitle(
        "Two labs' expression profiles for the same deletion agree",
        x=0.005,
        ha="left",
        fontsize=7,
        color=ink,
    )

    output_dir = osp.join(ASSET_IMAGES_DIR, "012-sameith-kemmeren-expression")
    os.makedirs(output_dir, exist_ok=True)
    png_path = osp.join(output_dir, f"{output_prefix}_{corr_type}.png")
    svg_path = osp.join(output_dir, f"{output_prefix}_{corr_type}.svg")
    fig.savefig(png_path, dpi=300, facecolor="white")
    savefig_true_size_svg(fig, svg_path, facecolor="white")
    logger.info(f"✓ Saved: {png_path}")
    plt.close()


def main():
    logger.info("=" * 80)
    logger.info("GENE-BY-GENE EXPRESSION CORRELATION ANALYSIS")
    logger.info(f"Debug mode: {DEBUG_MODE}")
    if DEBUG_MODE:
        logger.info(f"Sample size: {SAMPLE_SIZE}")
    logger.info("=" * 80)

    sample_range = SAMPLE_SIZE if DEBUG_MODE else None

    # Load datasets
    logger.info("\n--- Loading Datasets ---")

    kemmeren = MicroarrayKemmeren2014Dataset(
        root=osp.join(DATA_ROOT, "data/torchcell/microarray_kemmeren2014"), io_workers=0
    )
    logger.info(f"Loaded Kemmeren2014: {len(kemmeren)} experiments")

    sm_sameith = SmMicroarraySameith2015Dataset(
        root=osp.join(DATA_ROOT, "data/torchcell/sm_microarray_sameith2015"),
        io_workers=0,
    )
    logger.info(f"Loaded SmSameith2015: {len(sm_sameith)} experiments")

    # Extract expression profiles
    kemmeren_profiles = extract_expression_profiles(
        kemmeren, "Kemmeren2014", sample_range
    )
    sameith_profiles = extract_expression_profiles(
        sm_sameith, "SmSameith2015", sample_range
    )

    # Compute correlations
    df_corr = compute_gene_correlations(kemmeren_profiles, sameith_profiles)

    if len(df_corr) == 0:
        logger.error("No correlations computed!")
        return

    # Print summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("CORRELATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total genes analyzed: {len(df_corr)}")
    logger.info("\nPearson Correlation:")
    logger.info(f"  Median r: {df_corr['pearson_r'].median():.3f}")
    logger.info(f"  Mean r: {df_corr['pearson_r'].mean():.3f}")
    logger.info(f"  Std r: {df_corr['pearson_r'].std():.3f}")
    logger.info(
        f"  r > 0.5: {(df_corr['pearson_r'] > 0.5).sum()} "
        f"({(df_corr['pearson_r'] > 0.5).sum() / len(df_corr) * 100:.1f}%)"
    )
    logger.info(
        f"  r > 0.7: {(df_corr['pearson_r'] > 0.7).sum()} "
        f"({(df_corr['pearson_r'] > 0.7).sum() / len(df_corr) * 100:.1f}%)"
    )
    logger.info("\nSpearman Correlation:")
    logger.info(f"  Median r: {df_corr['spearman_r'].median():.3f}")
    logger.info(f"  Mean r: {df_corr['spearman_r'].mean():.3f}")
    logger.info(f"  Std r: {df_corr['spearman_r'].std():.3f}")
    logger.info(
        f"  r > 0.5: {(df_corr['spearman_r'] > 0.5).sum()} "
        f"({(df_corr['spearman_r'] > 0.5).sum() / len(df_corr) * 100:.1f}%)"
    )
    logger.info(
        f"  r > 0.7: {(df_corr['spearman_r'] > 0.7).sum()} "
        f"({(df_corr['spearman_r'] > 0.7).sum() / len(df_corr) * 100:.1f}%)"
    )

    # Create visualizations
    logger.info("\n--- Creating Visualizations ---")
    plot_correlation_distribution(
        df_corr, "pearson", "gene_expression_correlation_dist"
    )
    plot_correlation_distribution(
        df_corr, "spearman", "gene_expression_correlation_dist"
    )

    # Save correlation data (no timestamp - stable filenames for documentation)
    output_dir = osp.join(os.getenv("EXPERIMENT_ROOT"), "012-sameith-kemmeren/results")
    os.makedirs(output_dir, exist_ok=True)

    csv_path = osp.join(output_dir, "gene_expression_correlations.csv")
    df_corr.to_csv(csv_path, index=False)
    logger.info(f"✓ Saved correlations to: {csv_path}")

    logger.info("\n" + "=" * 80)
    logger.info("✓ GENE-BY-GENE CORRELATION ANALYSIS COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
