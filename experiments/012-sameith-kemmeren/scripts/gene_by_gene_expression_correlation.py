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

import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from dotenv import load_dotenv
from tqdm import tqdm
import logging
import pandas as pd

# Removed timestamp import - using stable filenames instead
from torchcell.datasets.scerevisiae.kemmeren2014 import (
    MicroarrayKemmeren2014Dataset,
)
from torchcell.datasets.scerevisiae.sameith2015 import (
    SmMicroarraySameith2015Dataset,
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
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Color schemes
KEMMEREN_COLOR = "#D0838E"
SAMEITH_COLOR = "#53777A"


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
        expression_profile = data["experiment"]["phenotype"][
            "expression_log2_ratio"
        ]

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
            logger.warning(
                f"{gene}: Only {len(common_measured)} common measured genes"
            )
            continue

        # Extract values for common genes
        common_genes_list = sorted(list(common_measured))
        kemmeren_values = [kemmeren_profile[g] for g in common_genes_list]
        sameith_values = [sameith_profile[g] for g in common_genes_list]

        # Compute correlations
        pearson_r, pearson_p = stats.pearsonr(kemmeren_values, sameith_values)
        spearman_r, spearman_p = stats.spearmanr(
            kemmeren_values, sameith_values
        )

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
    """
    Publication-quality histogram with color-coded bars.

    Args:
        df_corr: DataFrame with correlation results
        corr_type: "pearson" or "spearman"
        output_prefix: Prefix for output file

    Displays:
    - Histogram with bars colored by correlation value
    - Summary statistics in top left corner
    """
    logger.info(f"Creating {corr_type} correlation distribution plot")

    # Select appropriate column
    r_col = f"{corr_type}_r"

    # Use torchcell style
    plt.style.use('torchcell/torchcell.mplstyle')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create histogram with color gradient based on correlation value
    bins = np.linspace(-1, 1, 31)  # 30 bins
    counts, edges = np.histogram(df_corr[r_col], bins=bins)

    # Color map: negative (red) to positive (green) with white at 0
    from matplotlib.colors import to_rgb

    # Use torchcell colors for negative/positive
    neg_color = np.array(to_rgb('#B73C39'))  # Red from torchcell palette
    pos_color = np.array(to_rgb('#6B8D3A'))  # Green from torchcell palette
    white = np.array([1.0, 1.0, 1.0])

    # Create color for each bin by interpolating RGB values
    colors = []
    for i in range(len(edges) - 1):
        bin_center = (edges[i] + edges[i+1]) / 2
        if bin_center < 0:
            # Interpolate from white (at 0) to red (at -1)
            t = abs(bin_center)  # 0 to 1
            color = (1 - t) * white + t * neg_color
            colors.append(tuple(color))
        else:
            # Interpolate from white (at 0) to green (at +1)
            t = bin_center  # 0 to 1
            color = (1 - t) * white + t * pos_color
            colors.append(tuple(color))

    # Plot bars with individual colors
    ax.bar(edges[:-1], counts, width=np.diff(edges), align='edge',
           edgecolor='black', linewidth=0.8, color=colors)

    # Add vertical line at r=0
    ax.axvline(x=0, color='#000000', linestyle='--', linewidth=1.5, alpha=0.5)

    ax.set_xlabel(f"{corr_type.capitalize()} Correlation Coefficient (r)", fontsize=14, weight='bold')
    ax.set_ylabel("Number of Genes", fontsize=14, weight='bold')
    ax.set_title(
        f"Cross-Study Expression Profile Correlation\n"
        f"Kemmeren 2014 vs Sameith 2015",
        fontsize=16,
        weight='bold',
        pad=20
    )
    ax.set_xlim(-1, 1)
    ax.grid(axis="y", alpha=0.3, linestyle=':', linewidth=0.8)

    # Summary stats - TOP LEFT
    median_r = df_corr[r_col].median()
    mean_r = df_corr[r_col].mean()
    std_r = df_corr[r_col].std()
    n_total = len(df_corr)
    n_strong = (df_corr[r_col].abs() > 0.7).sum()
    n_moderate = ((df_corr[r_col].abs() > 0.5) & (df_corr[r_col].abs() <= 0.7)).sum()

    stats_text = (
        f"n = {n_total} gene deletions\n"
        f"Median r = {median_r:.3f}\n"
        f"Mean r = {mean_r:.3f} ± {std_r:.3f}\n"
        f"|r| > 0.7: {n_strong} ({100*n_strong/n_total:.1f}%)\n"
        f"|r| > 0.5: {n_strong + n_moderate} ({100*(n_strong + n_moderate)/n_total:.1f}%)"
    )

    # Place in top left with box
    ax.text(
        0.02,  # Top left corner
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='#666666', linewidth=1.2),
        fontfamily='sans-serif',
        fontweight='normal'
    )

    plt.tight_layout()

    # Save (no timestamp - stable filenames for documentation)
    output_dir = osp.join(ASSET_IMAGES_DIR, "012-sameith-kemmeren-expression")
    os.makedirs(output_dir, exist_ok=True)

    png_path = osp.join(output_dir, f"{output_prefix}_{corr_type}.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor='white')
    logger.info(f"✓ Saved PNG: {png_path}")

    plt.close()
    plt.style.use('default')  # Reset style


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
        root=osp.join(DATA_ROOT, "data/torchcell/microarray_kemmeren2014"),
        io_workers=0,
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
    plot_correlation_distribution(df_corr, "pearson", "gene_expression_correlation_dist")
    plot_correlation_distribution(df_corr, "spearman", "gene_expression_correlation_dist")

    # Save correlation data (no timestamp - stable filenames for documentation)
    output_dir = osp.join(
        os.getenv("EXPERIMENT_ROOT"), "012-sameith-kemmeren/results"
    )
    os.makedirs(output_dir, exist_ok=True)

    csv_path = osp.join(output_dir, "gene_expression_correlations.csv")
    df_corr.to_csv(csv_path, index=False)
    logger.info(f"✓ Saved correlations to: {csv_path}")

    logger.info("\n" + "=" * 80)
    logger.info("✓ GENE-BY-GENE CORRELATION ANALYSIS COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
