# experiments/012-sameith-kemmeren/scripts/noise_comparison_analysis
# [[experiments.012-sameith-kemmeren.scripts.noise_comparison_analysis]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/012-sameith-kemmeren/scripts/noise_comparison_analysis
# Test file: experiments/012-sameith-kemmeren/scripts/test_noise_comparison_analysis.py

"""
Cross-Dataset Noise/Standard Deviation Comparison

Compares measurement noise between Kemmeren 2014 and Sameith 2015 datasets
for the 82 overlapping GSTF genes.

For each gene deletion, we compute the median std across all ~6,000 measured genes.
This single value represents the "typical noise" for that deletion experiment.

We then compare these noise levels between datasets to determine if noise is:
- Gene-intrinsic (high correlation): Certain genes are always noisy
- Study-specific (low correlation): Noise is driven by technical differences
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
EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")

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
KEMMEREN_COLOR = "#D0838E"  # Pinkish/rose
SAMEITH_COLOR = "#53777A"  # Teal/blue-green


def extract_gene_std_data(dataset, dataset_name, std_field, sample_range=None):
    """
    Extract std values for each gene deletion (NO COMPUTATION - already stored).

    For each gene deletion:
    - Access dataset[i]['experiment']['phenotype'][std_field]
    - This returns {measured_gene: std_value} dict with ~6K genes
    - Compute median std across all measured genes
    - Map to deleted gene name

    Args:
        dataset: Dataset object
        dataset_name: Name for logging
        std_field: "expression_technical_std" or "expression_log2_ratio_std"
        sample_range: Optional limit for debug mode

    Returns:
        dict: {deleted_gene: median_std_across_measured_genes}
    """
    logger.info(f"Extracting std data from {dataset_name} using field '{std_field}'")

    std_data = {}

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
            logger.warning(
                f"{dataset_name}[{i}]: Expected 1 perturbation, got {len(perturbations)}"
            )
            continue

        gene_deleted = perturbations[0]["systematic_gene_name"]

        # Extract std dict (already computed and stored)
        std_dict = data["experiment"]["phenotype"].get(std_field)

        if std_dict is None:
            logger.warning(f"{dataset_name}[{i}]: {std_field} is None for {gene_deleted}")
            continue

        # Filter out NaN values
        std_values = [v for v in std_dict.values() if not np.isnan(v)]

        if len(std_values) == 0:
            logger.warning(f"{dataset_name}[{i}]: No valid std values for {gene_deleted}")
            continue

        # Compute median std across all measured genes
        median_std = np.median(std_values)

        if gene_deleted in std_data:
            logger.warning(f"{dataset_name}: Duplicate gene {gene_deleted}")

        std_data[gene_deleted] = median_std

    logger.info(f"Extracted std data for {len(std_data)} unique genes")
    return std_data


def compare_overlapping_std_scatter(kemmeren_std, sameith_std, output_prefix):
    """
    Scatter plot: Kemmeren std vs Sameith std for overlapping genes.

    X-axis: Kemmeren median std (per gene deletion)
    Y-axis: Sameith median std (per gene deletion)
    Each point: One overlapping GSTF gene

    Adds:
    - Diagonal reference line (y=x)
    - Pearson and Spearman correlations
    - Annotated outliers (genes far from diagonal)
    """
    logger.info("Creating scatter plot comparing std values")

    # Get overlapping genes
    overlap_genes = sorted(set(kemmeren_std.keys()) & set(sameith_std.keys()))

    # Extract values
    kemmeren_values = [kemmeren_std[g] for g in overlap_genes]
    sameith_values = [sameith_std[g] for g in overlap_genes]

    # Compute correlations
    pearson_r, pearson_p = stats.pearsonr(kemmeren_values, sameith_values)
    spearman_r, spearman_p = stats.spearmanr(kemmeren_values, sameith_values)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Scatter plot
    ax.scatter(kemmeren_values, sameith_values, alpha=0.6, s=100, color=SAMEITH_COLOR, edgecolors="black", linewidths=0.5)

    # Diagonal reference line (y=x)
    max_val = max(max(kemmeren_values), max(sameith_values))
    min_val = min(min(kemmeren_values), min(sameith_values))
    ax.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1, alpha=0.5, label="y=x")

    # Labels
    ax.set_xlabel("Kemmeren Median Std (log2_ratio_std)", fontsize=12)
    ax.set_ylabel("Sameith Median Std (log2_ratio_std)", fontsize=12)
    ax.set_title(
        f"Cross-Dataset Noise Comparison\n"
        f"Pearson r = {pearson_r:.3f} (p={pearson_p:.2e}) | "
        f"Spearman r = {spearman_r:.3f} (p={spearman_p:.2e})",
        fontsize=14,
    )

    # Grid
    ax.grid(True, alpha=0.3)

    # Identify and annotate outliers (far from diagonal)
    residuals = np.array(sameith_values) - np.array(kemmeren_values)
    outlier_threshold = np.percentile(np.abs(residuals), 95)

    for i, gene in enumerate(overlap_genes):
        if np.abs(residuals[i]) > outlier_threshold:
            ax.annotate(
                gene,
                (kemmeren_values[i], sameith_values[i]),
                fontsize=8,
                alpha=0.7,
                xytext=(5, 5),
                textcoords="offset points",
            )

    plt.tight_layout()

    # Save
    output_dir = osp.join(ASSET_IMAGES_DIR, "012-sameith-kemmeren-expression")
    os.makedirs(output_dir, exist_ok=True)

    png_path = osp.join(output_dir, f"{output_prefix}.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    logger.info(f"✓ Saved PNG: {png_path}")

    plt.close()


def compare_std_distributions_boxplot(kemmeren_std, sameith_std, output_prefix):
    """
    Side-by-side box plots comparing std distributions.

    Left: Kemmeren std for overlapping genes
    Right: Sameith std for overlapping genes

    Shows median, quartiles, outliers
    Includes Wilcoxon signed-rank test p-value
    """
    logger.info("Creating box plot comparing std distributions")

    # Get overlapping genes
    overlap_genes = sorted(set(kemmeren_std.keys()) & set(sameith_std.keys()))

    # Extract values
    kemmeren_values = [kemmeren_std[g] for g in overlap_genes]
    sameith_values = [sameith_std[g] for g in overlap_genes]

    # Wilcoxon signed-rank test (paired test)
    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(kemmeren_values, sameith_values)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Box plots
    bp = ax.boxplot(
        [kemmeren_values, sameith_values],
        labels=["Kemmeren\n(log2_ratio_std)", "Sameith\n(log2_ratio_std)"],
        patch_artist=True,
        widths=0.6,
    )

    # Color boxes
    bp["boxes"][0].set_facecolor(KEMMEREN_COLOR)
    bp["boxes"][1].set_facecolor(SAMEITH_COLOR)

    for element in ["whiskers", "fliers", "means", "medians", "caps"]:
        plt.setp(bp[element], color="black", linewidth=1.5)

    # Labels
    ax.set_ylabel("Median Std (across ~6K measured genes)", fontsize=12)
    ax.set_title(
        f"Noise Distribution Comparison ({len(overlap_genes)} overlapping genes)\n"
        f"Wilcoxon test: p = {wilcoxon_p:.2e}",
        fontsize=14,
    )

    # Grid
    ax.grid(axis="y", alpha=0.3)

    # Summary stats
    stats_text = (
        f"Kemmeren median: {np.median(kemmeren_values):.2f}\n"
        f"Sameith median: {np.median(sameith_values):.2f}\n"
        f"Kemmeren IQR: {np.percentile(kemmeren_values, 75) - np.percentile(kemmeren_values, 25):.2f}\n"
        f"Sameith IQR: {np.percentile(sameith_values, 75) - np.percentile(sameith_values, 25):.2f}"
    )

    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        color='black',
        weight='bold',
    )

    plt.tight_layout()

    # Save
    output_dir = osp.join(ASSET_IMAGES_DIR, "012-sameith-kemmeren-expression")
    os.makedirs(output_dir, exist_ok=True)

    png_path = osp.join(output_dir, f"{output_prefix}.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    logger.info(f"✓ Saved PNG: {png_path}")

    plt.close()


def compare_std_distributions_histogram(kemmeren_std, sameith_std, output_prefix):
    """
    Overlaid histograms of std distributions.

    Kemmeren (pinkish), Sameith (teal)
    With median lines
    """
    logger.info("Creating histogram comparing std distributions")

    # Get overlapping genes
    overlap_genes = sorted(set(kemmeren_std.keys()) & set(sameith_std.keys()))

    # Extract values
    kemmeren_values = [kemmeren_std[g] for g in overlap_genes]
    sameith_values = [sameith_std[g] for g in overlap_genes]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Histograms
    ax.hist(
        kemmeren_values,
        bins=20,
        alpha=0.6,
        color=KEMMEREN_COLOR,
        edgecolor="black",
        label="Kemmeren (log2_ratio_std)",
    )
    ax.hist(
        sameith_values,
        bins=20,
        alpha=0.6,
        color=SAMEITH_COLOR,
        edgecolor="black",
        label="Sameith (log2_ratio_std)",
    )

    # Median lines
    ax.axvline(
        np.median(kemmeren_values),
        color=KEMMEREN_COLOR,
        linestyle="--",
        linewidth=2,
        label=f"Kemmeren median: {np.median(kemmeren_values):.2f}",
    )
    ax.axvline(
        np.median(sameith_values),
        color=SAMEITH_COLOR,
        linestyle="--",
        linewidth=2,
        label=f"Sameith median: {np.median(sameith_values):.2f}",
    )

    # Labels
    ax.set_xlabel("Median Std (across ~6K measured genes)", fontsize=12)
    ax.set_ylabel("Number of Genes", fontsize=12)
    ax.set_title(
        f"Noise Distribution Comparison ({len(overlap_genes)} overlapping genes)",
        fontsize=14,
    )

    # Legend
    ax.legend(loc="upper right", fontsize=10)

    # Grid
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    # Save
    output_dir = osp.join(ASSET_IMAGES_DIR, "012-sameith-kemmeren-expression")
    os.makedirs(output_dir, exist_ok=True)

    png_path = osp.join(output_dir, f"{output_prefix}.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    logger.info(f"✓ Saved PNG: {png_path}")

    plt.close()


def main():
    logger.info("=" * 80)
    logger.info("CROSS-DATASET NOISE COMPARISON ANALYSIS")
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

    # Extract median std per gene deletion
    # Both datasets use expression_log2_ratio_std for fair comparison
    logger.info("\n--- Extracting Std Data ---")
    kemmeren_std = extract_gene_std_data(
        kemmeren, "Kemmeren2014", "expression_log2_ratio_std", sample_range
    )
    sameith_std = extract_gene_std_data(
        sm_sameith, "SmSameith2015", "expression_log2_ratio_std", sample_range
    )

    # Find overlapping genes
    overlap_genes = sorted(set(kemmeren_std.keys()) & set(sameith_std.keys()))
    logger.info(f"\nFound {len(overlap_genes)} overlapping genes")

    if len(overlap_genes) == 0:
        logger.error("No overlapping genes found!")
        return

    # Filter to overlapping genes only
    kemmeren_overlap = {g: kemmeren_std[g] for g in overlap_genes}
    sameith_overlap = {g: sameith_std[g] for g in overlap_genes}

    # Print summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("NOISE COMPARISON SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Overlapping genes: {len(overlap_genes)}")
    logger.info(f"\nKemmeren (log2_ratio_std):")
    logger.info(f"  Median: {np.median(list(kemmeren_overlap.values())):.3f}")
    logger.info(f"  Mean: {np.mean(list(kemmeren_overlap.values())):.3f}")
    logger.info(f"  Std: {np.std(list(kemmeren_overlap.values())):.3f}")
    logger.info(f"\nSameith (log2_ratio_std):")
    logger.info(f"  Median: {np.median(list(sameith_overlap.values())):.3f}")
    logger.info(f"  Mean: {np.mean(list(sameith_overlap.values())):.3f}")
    logger.info(f"  Std: {np.std(list(sameith_overlap.values())):.3f}")

    # Compute correlations
    kemmeren_values = [kemmeren_overlap[g] for g in overlap_genes]
    sameith_values = [sameith_overlap[g] for g in overlap_genes]
    pearson_r, pearson_p = stats.pearsonr(kemmeren_values, sameith_values)
    spearman_r, spearman_p = stats.spearmanr(kemmeren_values, sameith_values)

    logger.info(f"\nCorrelations:")
    logger.info(f"  Pearson r: {pearson_r:.3f} (p={pearson_p:.2e})")
    logger.info(f"  Spearman r: {spearman_r:.3f} (p={spearman_p:.2e})")

    # Create visualizations
    logger.info("\n--- Creating Visualizations ---")
    compare_overlapping_std_scatter(kemmeren_overlap, sameith_overlap, "noise_scatter")
    compare_std_distributions_boxplot(kemmeren_overlap, sameith_overlap, "noise_boxplot")
    compare_std_distributions_histogram(kemmeren_overlap, sameith_overlap, "noise_histogram")

    # Save CSV
    logger.info("\n--- Saving Results ---")
    output_dir = osp.join(EXPERIMENT_ROOT, "012-sameith-kemmeren/results")
    os.makedirs(output_dir, exist_ok=True)

    df_std = pd.DataFrame({
        "gene": overlap_genes,
        "kemmeren_log2_ratio_std": [kemmeren_overlap[g] for g in overlap_genes],
        "sameith_log2_ratio_std": [sameith_overlap[g] for g in overlap_genes],
    })
    csv_path = osp.join(output_dir, "noise_comparison.csv")
    df_std.to_csv(csv_path, index=False)
    logger.info(f"✓ Saved comparison to: {csv_path}")

    logger.info("\n" + "=" * 80)
    logger.info("✓ CROSS-DATASET NOISE COMPARISON COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nOutputs saved to:")
    logger.info(f"  Images: {osp.join(ASSET_IMAGES_DIR, '012-sameith-kemmeren-expression')}")
    logger.info(f"  CSV: {csv_path}")


if __name__ == "__main__":
    main()
