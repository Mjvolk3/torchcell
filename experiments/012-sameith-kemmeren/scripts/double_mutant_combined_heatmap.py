# experiments/012-sameith-kemmeren/scripts/double_mutant_combined_heatmap
# [[experiments.012-sameith-kemmeren.scripts.double_mutant_combined_heatmap]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/012-sameith-kemmeren/scripts/double_mutant_combined_heatmap
# Test file: experiments/012-sameith-kemmeren/scripts/test_double_mutant_combined_heatmap.py

"""
Double Mutant Combined Heatmap Visualization

Creates a single combined heatmap showing both SD and Mean in one plot:
- Upper triangle: Standard Deviation (variability of expression changes)
- Lower triangle: Mean Expression Change (signed log2 fold changes)
- Diagonal: Black (not meaningful for self-pairs)
- Unconstructed pairs: Black (experiments not performed)

This is more efficient than two separate plots since the matrices are symmetric.
"""

import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.cm import ScalarMappable
from dotenv import load_dotenv
from tqdm import tqdm
import logging

# Removed timestamp import - using stable filenames instead
from torchcell.datasets.scerevisiae.sameith2015 import DmMicroarraySameith2015Dataset

# Load environment variables
load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")

# Configuration
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "20"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Colors
UNCONSTRUCTED_COLOR = "#000000"  # Black for unconstructed pairs


def extract_double_mutant_data(dataset, sample_range=None):
    """
    Extract double mutant expression statistics.

    Returns:
        tuple: (gene_set, sd_matrix, mean_matrix, gene_to_idx)
    """
    logger.info("Extracting double mutant expression data")

    # Get all unique genes
    all_genes = sorted(list(dataset.gene_set))
    n_genes = len(all_genes)
    logger.info(f"Found {n_genes} unique genes in double mutant dataset")

    # Create gene to index mapping (O(1) lookup)
    gene_to_idx = {gene: idx for idx, gene in enumerate(all_genes)}

    # Initialize matrices with NaN (for unconstructed pairs)
    sd_matrix = np.full((n_genes, n_genes), np.nan)
    mean_matrix = np.full((n_genes, n_genes), np.nan)

    # Determine iteration range
    if sample_range is None:
        iter_range = range(len(dataset))
    else:
        iter_range = range(min(sample_range, len(dataset)))

    # Process each double mutant
    for i in tqdm(iter_range, desc="Processing double mutants"):
        data = dataset[i]

        # Extract perturbed genes
        perturbations = data["experiment"]["genotype"]["perturbations"]
        if len(perturbations) != 2:
            logger.warning(
                f"Index {i}: Expected 2 perturbations, "
                f"got {len(perturbations)}"
            )
            continue

        gene1 = perturbations[0]["systematic_gene_name"]
        gene2 = perturbations[1]["systematic_gene_name"]

        # Skip if genes not in gene set (shouldn't happen)
        if gene1 not in gene_to_idx or gene2 not in gene_to_idx:
            logger.warning(
                f"Index {i}: Genes {gene1} or {gene2} not in gene set"
            )
            continue

        # Extract expression log2 ratios
        log2_ratios = list(
            data["experiment"]["phenotype"]["expression_log2_ratio"].values()
        )

        # Filter out NaN values
        log2_ratios = [r for r in log2_ratios if not np.isnan(r)]

        if len(log2_ratios) == 0:
            logger.warning(
                f"Index {i}: No valid expression data for {gene1}-{gene2}"
            )
            continue

        # Calculate statistics
        sd_val = np.std(log2_ratios, ddof=1)  # Sample standard deviation
        mean_val = np.mean(log2_ratios)

        # Get indices
        idx1 = gene_to_idx[gene1]
        idx2 = gene_to_idx[gene2]

        # Fill symmetric positions
        sd_matrix[idx1, idx2] = sd_val
        sd_matrix[idx2, idx1] = sd_val
        mean_matrix[idx1, idx2] = mean_val
        mean_matrix[idx2, idx1] = mean_val

    # Count constructed pairs
    n_constructed = np.sum(~np.isnan(sd_matrix)) // 2
    logger.info(f"Constructed {n_constructed} double mutant pairs")

    return all_genes, sd_matrix, mean_matrix, gene_to_idx


def create_combined_heatmap(genes, sd_matrix, mean_matrix, output_prefix):
    """
    Create combined heatmap with SD in upper triangle, Mean in lower triangle.

    Upper triangle: Standard deviation (Greens colormap)
    Lower triangle: Mean expression change (RdBu_r diverging colormap)
    Diagonal: Black (not meaningful)
    Unconstructed pairs: Black
    """
    logger.info("Creating combined SD/Mean heatmap")

    n_genes = len(genes)

    # Create SQUARE figure - absolutely square!
    fig, ax = plt.subplots(figsize=(18, 18))

    # Get value ranges (excluding NaN and diagonal)
    sd_valid = sd_matrix[~np.isnan(sd_matrix)]
    mean_valid = mean_matrix[~np.isnan(mean_matrix)]

    sd_vmin = 0
    sd_vmax = np.percentile(sd_valid, 99)

    mean_abs_max = np.percentile(np.abs(mean_valid), 99)
    mean_vmin = -mean_abs_max
    mean_vmax = mean_abs_max

    # Create combined matrix manually
    # We'll create two separate colormapped arrays and combine them

    # Normalize SD values
    norm_sd = Normalize(vmin=sd_vmin, vmax=sd_vmax)
    cmap_sd = plt.cm.Greens  # Changed from Blues to avoid conflict with RdBu

    # Normalize Mean values
    norm_mean = TwoSlopeNorm(vmin=mean_vmin, vcenter=0, vmax=mean_vmax)
    cmap_mean = plt.cm.RdBu_r

    # Create RGBA image manually
    combined_img = np.ones((n_genes, n_genes, 4))  # RGBA

    # Fill in upper triangle with SD colors
    for i in range(n_genes):
        for j in range(i + 1, n_genes):  # Upper triangle (j > i)
            if not np.isnan(sd_matrix[i, j]):
                # Get color from SD colormap
                rgba = cmap_sd(norm_sd(sd_matrix[i, j]))
                combined_img[i, j] = rgba
            else:
                # Unconstructed - black
                combined_img[i, j] = [0.0, 0.0, 0.0, 1.0]  # Black

    # Fill in lower triangle with Mean colors
    for i in range(n_genes):
        for j in range(i):  # Lower triangle (j < i)
            if not np.isnan(mean_matrix[i, j]):
                # Get color from Mean colormap
                rgba = cmap_mean(norm_mean(mean_matrix[i, j]))
                combined_img[i, j] = rgba
            else:
                # Unconstructed - black
                combined_img[i, j] = [0.0, 0.0, 0.0, 1.0]  # Black

    # Fill diagonal with black
    for i in range(n_genes):
        combined_img[i, i] = [0.0, 0.0, 0.0, 1.0]  # Black

    # Plot the combined image with EQUAL aspect ratio (ensures square pixels)
    im = ax.imshow(
        combined_img,
        aspect="equal",
        interpolation="nearest",
        origin="upper",
    )

    # Create dummy mappables for colorbars
    sm_sd = ScalarMappable(cmap=cmap_sd, norm=norm_sd)
    sm_sd.set_array([])

    sm_mean = ScalarMappable(cmap=cmap_mean, norm=norm_mean)
    sm_mean.set_array([])

    # Create two colorbars - REDUCED SIZE
    # SD colorbar (right side)
    cbar_sd = plt.colorbar(
        sm_sd, ax=ax, fraction=0.03, pad=0.04, location="right"
    )
    cbar_sd.set_label("SD of Log2 FC (Upper Triangle)", fontsize=12)
    cbar_sd.ax.tick_params(labelsize=10)

    # Mean colorbar (left side)
    cbar_mean = plt.colorbar(
        sm_mean, ax=ax, fraction=0.03, pad=0.08, location="left"
    )
    cbar_mean.set_label("Mean Log2 FC (Lower Triangle)", fontsize=12)
    cbar_mean.ax.tick_params(labelsize=10)

    # Axis labels
    ax.set_xticks(range(len(genes)))
    ax.set_yticks(range(len(genes)))
    ax.set_xticklabels(genes, rotation=90, fontsize=6, ha="center")
    ax.set_yticklabels(genes, fontsize=6)

    # Remove spines to make labels flush with heatmap
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Adjust tick parameters to position labels right at edge
    ax.tick_params(axis='x', which='both', length=0, pad=2)
    ax.tick_params(axis='y', which='both', length=0, pad=2)

    # Title
    ax.set_title(
        "Double Mutant Expression Analysis\n"
        "Upper: Variability (SD) | Lower: Mean Change | Black: Unconstructed",
        fontsize=14,
        pad=20,
    )

    # Add white diagonal line to separate triangles (adjusted to image bounds)
    ax.plot([-0.5, n_genes - 0.5], [-0.5, n_genes - 0.5], "w-", linewidth=2, alpha=1.0)

    # Tight layout
    plt.tight_layout()

    # Save output (no timestamp - stable filenames for documentation)
    output_dir = osp.join(ASSET_IMAGES_DIR, "012-sameith-kemmeren-expression")
    os.makedirs(output_dir, exist_ok=True)

    png_path = osp.join(output_dir, f"{output_prefix}.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    logger.info(f"✓ Saved PNG: {png_path}")

    plt.close()

    return png_path


def print_matrix_stats(matrix, matrix_name):
    """Print statistics about the matrix."""
    valid_values = matrix[~np.isnan(matrix)]
    n_total = matrix.size
    n_valid = len(valid_values)
    n_nan = n_total - n_valid

    logger.info(f"\n{matrix_name} Statistics:")
    logger.info(f"  Total cells: {n_total}")
    logger.info(f"  Constructed pairs: {n_valid // 2}")
    logger.info(f"  Unconstructed cells: {n_nan}")
    logger.info(f"  Min: {np.min(valid_values):.4f}")
    logger.info(f"  Max: {np.max(valid_values):.4f}")
    logger.info(f"  Mean: {np.mean(valid_values):.4f}")
    logger.info(f"  Median: {np.median(valid_values):.4f}")
    logger.info(f"  Std: {np.std(valid_values, ddof=1):.4f}")


def main():
    logger.info("=" * 80)
    logger.info("DOUBLE MUTANT COMBINED HEATMAP VISUALIZATION")
    logger.info(f"Debug mode: {DEBUG_MODE}")
    if DEBUG_MODE:
        logger.info(f"Sample size: {SAMPLE_SIZE}")
    logger.info("=" * 80)

    sample_range = SAMPLE_SIZE if DEBUG_MODE else None

    # Load dataset
    logger.info("\n--- Loading Double Mutant Dataset ---")
    dm_sameith = DmMicroarraySameith2015Dataset(
        root=osp.join(DATA_ROOT, "data/torchcell/dm_microarray_sameith2015"),
        io_workers=0,
    )
    logger.info(f"Loaded {len(dm_sameith)} double mutant experiments")

    # Extract data
    genes, sd_matrix, mean_matrix, gene_to_idx = extract_double_mutant_data(
        dm_sameith, sample_range
    )

    # Print statistics
    print_matrix_stats(sd_matrix, "SD Matrix")
    print_matrix_stats(mean_matrix, "Mean Matrix")

    # Create combined heatmap
    logger.info("\n--- Creating Combined Heatmap ---")
    create_combined_heatmap(
        genes, sd_matrix, mean_matrix, "double_mutant_combined_heatmap"
    )

    logger.info("\n" + "=" * 80)
    logger.info("✓ COMBINED HEATMAP VISUALIZATION COMPLETE")
    logger.info("=" * 80)
    logger.info(
        f"\nOutput saved to: {osp.join(ASSET_IMAGES_DIR, '012-sameith-kemmeren-expression')}"
    )


if __name__ == "__main__":
    main()
