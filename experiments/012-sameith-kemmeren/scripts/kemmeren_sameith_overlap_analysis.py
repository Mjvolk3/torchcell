# experiments/012-sameith-kemmeren/scripts/kemmeren_sameith_overlap_analysis
# [[experiments.012-sameith-kemmeren.scripts.kemmeren_sameith_overlap_analysis]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/012-sameith-kemmeren/scripts/kemmeren_sameith_overlap_analysis
# Test file: experiments/012-sameith-kemmeren/scripts/test_kemmeren_sameith_overlap_analysis.py

"""
Cross-Dataset Overlap Analysis: Kemmeren vs Sameith

Analyzes overlapping genes between Kemmeren 2014 and Sameith 2015 single mutant datasets.

Outputs:
1. Scatter plot: Mean expression changes correlation
2. Distribution comparison: Side-by-side box plots for top genes
"""

import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from dotenv import load_dotenv
from tqdm import tqdm
import logging

from torchcell.datasets.scerevisiae.kemmeren2014 import MicroarrayKemmeren2014Dataset
from torchcell.datasets.scerevisiae.sameith2015 import SmMicroarraySameith2015Dataset

# Load environment variables
load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")

# Configuration
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "50"))
TOP_N_GENES = 20  # Number of top genes to show in distribution comparison

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Color schemes
KEMMEREN_COLOR = "#D0838E"  # Pinkish/rose
SAMEITH_COLOR = "#53777A"  # Teal/blue-green


def extract_expression_data(dataset, dataset_name, sample_range=None):
    """Extract log2 expression ratios for each gene deletion."""
    logger.info(f"Extracting expression data from {dataset_name}")

    data_dict = {}

    # Determine iteration range
    if sample_range is None:
        iter_range = range(len(dataset))
    else:
        iter_range = range(min(sample_range, len(dataset)))

    for i in tqdm(iter_range, desc=f"Processing {dataset_name}"):
        data = dataset[i]

        # Extract deleted gene
        perturbations = data["experiment"]["genotype"]["perturbations"]
        if len(perturbations) != 1:
            continue

        gene_deleted = perturbations[0]["systematic_gene_name"]

        # Extract expression log2 ratios
        log2_ratios = list(
            data["experiment"]["phenotype"]["expression_log2_ratio"].values()
        )

        # Filter out NaN values
        log2_ratios = [r for r in log2_ratios if not np.isnan(r)]

        if gene_deleted in data_dict:
            logger.warning(f"{dataset_name}: Duplicate gene {gene_deleted}")

        data_dict[gene_deleted] = log2_ratios

    logger.info(f"Extracted data for {len(data_dict)} unique genes")
    return data_dict


def create_scatter_plot(kemmeren_data, sameith_data, overlap_genes):
    """
    Create scatter plot of mean expression changes.

    X-axis: Mean log2 FC in Kemmeren
    Y-axis: Mean log2 FC in Sameith
    """
    logger.info("Creating scatter plot of mean expression changes")

    # Calculate means for each gene
    kemmeren_means = [np.mean(kemmeren_data[gene]) for gene in overlap_genes]
    sameith_means = [np.mean(sameith_data[gene]) for gene in overlap_genes]

    # Calculate correlations
    pearson_r, pearson_p = stats.pearsonr(kemmeren_means, sameith_means)
    spearman_r, spearman_p = stats.spearmanr(kemmeren_means, sameith_means)

    # OLS regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        kemmeren_means, sameith_means
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Scatter plot
    ax.scatter(
        kemmeren_means,
        sameith_means,
        alpha=0.6,
        s=100,
        color="#4A90E2",
        edgecolors="black",
        linewidths=0.5,
    )

    # Regression line
    x_range = np.array([min(kemmeren_means), max(kemmeren_means)])
    y_pred = slope * x_range + intercept
    ax.plot(
        x_range,
        y_pred,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"OLS: y = {slope:.3f}x + {intercept:.3f}",
    )

    # Diagonal reference line (perfect correlation)
    ax.plot(
        x_range,
        x_range,
        color="gray",
        linestyle=":",
        linewidth=1,
        alpha=0.5,
        label="y = x",
    )

    # Labels and title
    ax.set_xlabel("Mean Log2 FC (Kemmeren 2014)", fontsize=12)
    ax.set_ylabel("Mean Log2 FC (Sameith 2015)", fontsize=12)
    ax.set_title(
        f"Expression Change Correlation Across {len(overlap_genes)} Overlapping Genes",
        fontsize=14,
        pad=20,
    )

    # Add statistics annotation with publication-quality styling
    stats_text = (
        f"Pearson r = {pearson_r:.3f} (p = {pearson_p:.2e})\n"
        f"Spearman ρ = {spearman_r:.3f} (p = {spearman_p:.2e})\n"
        f"R² = {r_value**2:.3f}\n"
        f"n = {len(overlap_genes)}"
    )
    ax.text(
        0.05,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="black", linewidth=1),
    )

    # Grid
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8, alpha=0.3)
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8, alpha=0.3)

    # Legend
    ax.legend(loc="lower right", fontsize=10)

    # Equal aspect ratio
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()

    # Save output
    output_dir = osp.join(ASSET_IMAGES_DIR, "012-sameith-kemmeren-expression")
    os.makedirs(output_dir, exist_ok=True)

    png_path = osp.join(output_dir, "kemmeren_sameith_overlap_scatter.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    logger.info(f"✓ Saved PNG: {png_path}")

    plt.close()

    return pearson_r, spearman_r


def create_distribution_comparison(kemmeren_data, sameith_data, top_genes):
    """
    Create side-by-side box plots comparing distributions for top genes.

    Top genes ranked by absolute difference in mean expression.
    """
    logger.info(f"Creating distribution comparison for {len(top_genes)} genes")

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))

    # Prepare data for plotting
    plot_data = []
    plot_positions = []
    plot_colors = []
    plot_labels = []

    for i, gene in enumerate(top_genes):
        # Kemmeren data
        kemmeren_values = kemmeren_data[gene]
        sameith_values = sameith_data[gene]

        # Position: each gene gets 2 positions (Kemmeren, Sameith) with gap
        pos_k = i * 3  # Kemmeren position
        pos_s = i * 3 + 1  # Sameith position

        plot_data.append(kemmeren_values)
        plot_data.append(sameith_values)
        plot_positions.append(pos_k)
        plot_positions.append(pos_s)
        plot_colors.append(KEMMEREN_COLOR)
        plot_colors.append(SAMEITH_COLOR)

    # Create box plots
    bp = ax.boxplot(
        plot_data,
        positions=plot_positions,
        widths=0.8,
        showcaps=False,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.5},
        whiskerprops={"linewidth": 0.8},
        boxprops={"linewidth": 0.8},
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.5},
    )

    # Color boxes
    for patch, color in zip(bp["boxes"], plot_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # X-axis: gene names centered between pairs
    gene_positions = [i * 3 + 0.5 for i in range(len(top_genes))]
    ax.set_xticks(gene_positions)
    ax.set_xticklabels(top_genes, rotation=45, ha="right", fontsize=10)

    # Labels and title
    ax.set_xlabel("Gene Systematic Name", fontsize=12)
    ax.set_ylabel("Log2 Fold Change", fontsize=12)
    ax.set_title(
        f"Expression Distribution Comparison: Top {len(top_genes)} Genes\n"
        f"(Ranked by Absolute Difference in Mean Expression)",
        fontsize=14,
        pad=20,
    )

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=KEMMEREN_COLOR, alpha=0.7, label="Kemmeren 2014"),
        Patch(facecolor=SAMEITH_COLOR, alpha=0.7, label="Sameith 2015"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=11)

    # Grid
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)

    plt.tight_layout()

    # Save output
    output_dir = osp.join(ASSET_IMAGES_DIR, "012-sameith-kemmeren-expression")
    os.makedirs(output_dir, exist_ok=True)

    png_path = osp.join(output_dir, "kemmeren_sameith_overlap_distributions.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    logger.info(f"✓ Saved PNG: {png_path}")

    plt.close()


def main():
    logger.info("=" * 80)
    logger.info("CROSS-DATASET OVERLAP ANALYSIS: KEMMEREN VS SAMEITH")
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

    # Extract expression data
    kemmeren_data = extract_expression_data(kemmeren, "Kemmeren2014", sample_range)
    sameith_data = extract_expression_data(sm_sameith, "SmSameith2015", sample_range)

    # Find overlapping genes
    kemmeren_genes = set(kemmeren_data.keys())
    sameith_genes = set(sameith_data.keys())
    overlap_genes = sorted(list(kemmeren_genes & sameith_genes))

    logger.info(f"\n--- Overlap Analysis ---")
    logger.info(f"Kemmeren genes: {len(kemmeren_genes)}")
    logger.info(f"Sameith genes: {len(sameith_genes)}")
    logger.info(f"Overlapping genes: {len(overlap_genes)}")

    if len(overlap_genes) == 0:
        logger.error("No overlapping genes found!")
        return

    # Calculate mean differences for ranking
    gene_rankings = {}
    for gene in overlap_genes:
        kemmeren_mean = np.mean(kemmeren_data[gene])
        sameith_mean = np.mean(sameith_data[gene])
        abs_diff = abs(kemmeren_mean - sameith_mean)
        gene_rankings[gene] = abs_diff

    # Get top N genes by absolute mean difference
    top_genes = sorted(gene_rankings, key=gene_rankings.get, reverse=True)[
        :TOP_N_GENES
    ]
    logger.info(f"\nTop {len(top_genes)} genes by absolute mean difference:")
    for i, gene in enumerate(top_genes[:10], 1):
        logger.info(f"  {i}. {gene}: Δ = {gene_rankings[gene]:.4f}")

    # Create visualizations
    logger.info("\n--- Creating Visualizations ---")

    # Scatter plot
    pearson_r, spearman_r = create_scatter_plot(
        kemmeren_data, sameith_data, overlap_genes
    )

    # Distribution comparison
    create_distribution_comparison(kemmeren_data, sameith_data, top_genes)

    logger.info("\n" + "=" * 80)
    logger.info("✓ OVERLAP ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nKey findings:")
    logger.info(f"  - Overlapping genes: {len(overlap_genes)}")
    logger.info(f"  - Pearson correlation: {pearson_r:.3f}")
    logger.info(f"  - Spearman correlation: {spearman_r:.3f}")
    logger.info(f"\nOutputs saved to: {osp.join(ASSET_IMAGES_DIR, '012-sameith-kemmeren-expression')}")


if __name__ == "__main__":
    main()
