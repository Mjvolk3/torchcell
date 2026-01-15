# experiments/012-sameith-kemmeren/scripts/single_mutant_expression_distributions
# [[experiments.012-sameith-kemmeren.scripts.single_mutant_expression_distributions]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/012-sameith-kemmeren/scripts/single_mutant_expression_distributions
# Test file: experiments/012-sameith-kemmeren/scripts/test_single_mutant_expression_distributions.py

"""
Single Mutant Expression Distribution Visualization

Creates single wide box plot visualizations showing log2 expression changes across all
measured genes for single deletion mutants in:
- Kemmeren 2014: ~1,484 single gene deletions
- Sameith 2015: 82 GSTF deletions

Matches the style of deprecated smf_ge_box_plot.py with:
- 100×20 inch figure size
- No individual gene labels (shows distribution only)
- Bold styling (10pt spines, #E84A26 whiskers)
- Large fonts (80-100pt)
- Percentage statistics for values outside ranges
"""

import os
import os.path as osp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from tqdm import tqdm
import logging

# Removed timestamp import - using stable filenames instead
from torchcell.datasets.scerevisiae.kemmeren2014 import MicroarrayKemmeren2014Dataset
from torchcell.datasets.scerevisiae.sameith2015 import SmMicroarraySameith2015Dataset

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
KEMMEREN_COLOR = "#D0838E"  # Pinkish/rose
SAMEITH_COLOR = "#53777A"  # Teal/blue-green
MEDIAN_COLOR = "#FF6B35"  # Orange


def extract_expression_data(dataset, dataset_name, sample_range=None):
    """
    Extract log2 expression ratios for each gene deletion.

    Returns:
        dict: {gene_systematic_name: [log2_ratios_across_all_measured_genes]}
    """
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
            logger.warning(
                f"{dataset_name}[{i}]: Expected 1 perturbation, got {len(perturbations)}"
            )
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


def calc_percentage_outside(data_dict, threshold):
    """
    Calculate percentage of expression values outside [-threshold, +threshold].

    Returns average per deletion strain statistics rather than pooled counts.

    Args:
        data_dict: {gene_name: [log2_ratios]}
        threshold: Absolute threshold value

    Returns:
        tuple: (avg_count_outside, avg_total_genes, percentage)
            - avg_count_outside: Average number of genes outside threshold per deletion strain
            - avg_total_genes: Average number of genes measured per deletion strain
            - percentage: (total_outside / total_genes) * 100
    """
    num_strains = len(data_dict)

    # Count total genes outside threshold (across all strains)
    total_count_outside = sum(
        sum(1 for v in gene_values if abs(v) > threshold)
        for gene_values in data_dict.values()
    )

    # Count total genes measured (across all strains)
    total_genes = sum(len(gene_values) for gene_values in data_dict.values())

    # Calculate averages per strain
    avg_count_outside = total_count_outside / num_strains
    avg_genes_measured = total_genes / num_strains
    percentage = (total_count_outside / total_genes) * 100

    return int(round(avg_count_outside)), int(round(avg_genes_measured)), percentage


def create_single_wide_boxplot(
    data_dict,
    title,
    output_prefix,
    xlabel="Single Gene Deletion Strains",
):
    """
    Create single wide box plot matching deprecated script style.

    Styling:
    - Figure size: 100×20 inches (very wide)
    - No individual gene labels (shows distribution only)
    - Bold spines (10pt linewidth)
    - Orange-red whiskers (#E84A26)
    - Large fonts (80-100pt)
    - Percentage statistics for values outside ±0.25, ±0.50, ±0.75, ±1.00 log2 FC
    """
    logger.info(f"Creating single wide box plot: {title}")

    # Sort genes alphabetically for consistent ordering
    genes_sorted = sorted(data_dict.keys())

    # Melt data to long format for seaborn
    records = []
    for gene in genes_sorted:
        for value in data_dict[gene]:
            records.append({"Gene": gene, "Expression": value})

    melted_df = pd.DataFrame(records)

    # Calculate percentage statistics (values OUTSIDE ranges)
    count_25, total_25, perc_25 = calc_percentage_outside(data_dict, 0.25)
    count_50, total_50, perc_50 = calc_percentage_outside(data_dict, 0.50)
    count_75, total_75, perc_75 = calc_percentage_outside(data_dict, 0.75)
    count_100, total_100, perc_100 = calc_percentage_outside(data_dict, 1.00)

    # Create figure
    plt.figure(figsize=(100, 20))
    ax = sns.boxplot(
        x="Gene",
        y="Expression",
        data=melted_df,
        order=genes_sorted,  # Ensure alphabetical order
        showfliers=False,  # No outliers
        showcaps=False,  # No caps
        whiskerprops={"color": "#E84A26", "linewidth": 3},  # Orange-red whiskers, thicker
    )

    # Change the thickness of the axis border (spines)
    for spine in ax.spines.values():
        spine.set_linewidth(10)

    # X-axis customization (no labels)
    ax.set(xticklabels=[])
    plt.xlabel(xlabel, fontsize=80)
    plt.ylabel("Log2 Fold Change", fontsize=80)

    # Y-axis customization
    plt.yticks(fontsize=80)
    ax.tick_params(axis="y", length=20, width=10)

    # Title
    plt.title(title, fontsize=100)

    # Add text annotations for percentages with absolute counts (no timestamp needed - stable output)
    text_str = (
        f"±0.25 log2 FC: ({count_25}/{total_25}) {perc_25:.2f}%\n"
        f"±0.50 log2 FC: ({count_50}/{total_50}) {perc_50:.2f}%\n"
        f"±0.75 log2 FC: ({count_75}/{total_75}) {perc_75:.2f}%\n"
        f"±1.00 log2 FC: ({count_100}/{total_100}) {perc_100:.2f}%"
    )
    ax.text(
        0.5,
        0.95,
        text_str,
        transform=ax.transAxes,
        fontsize=80,
        verticalalignment="top",
        horizontalalignment="center",
        color='black',
        weight='bold',
    )

    # Save output (no timestamp - stable filenames for documentation)
    output_dir = osp.join(ASSET_IMAGES_DIR, "012-sameith-kemmeren-expression")
    os.makedirs(output_dir, exist_ok=True)

    png_path = osp.join(output_dir, f"{output_prefix}.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    logger.info(f"✓ Saved PNG: {png_path}")

    plt.close()

    return png_path


def main():
    logger.info("=" * 80)
    logger.info("SINGLE MUTANT EXPRESSION DISTRIBUTION VISUALIZATION")
    logger.info(f"Debug mode: {DEBUG_MODE}")
    if DEBUG_MODE:
        logger.info(f"Sample size: {SAMPLE_SIZE}")
    logger.info("=" * 80)

    sample_range = SAMPLE_SIZE if DEBUG_MODE else None

    # Process Kemmeren dataset
    logger.info("\n--- Processing Kemmeren 2014 ---")
    kemmeren = MicroarrayKemmeren2014Dataset(
        root=osp.join(DATA_ROOT, "data/torchcell/microarray_kemmeren2014"),
        io_workers=0,
    )
    logger.info(f"Loaded {len(kemmeren)} experiments")

    kemmeren_data = extract_expression_data(kemmeren, "Kemmeren2014", sample_range)

    # Create Kemmeren plot
    logger.info("\n--- Creating Kemmeren Visualization ---")
    create_single_wide_boxplot(
        kemmeren_data,
        title=f"Kemmeren 2014: Log2 Expression Changes ({len(kemmeren_data)} Genes)",
        output_prefix="single_mutant_kemmeren",
        xlabel="Single Gene Deletion Strains",
    )

    # Process Sameith dataset
    logger.info("\n--- Processing Sameith 2015 (Single Mutants) ---")
    sm_sameith = SmMicroarraySameith2015Dataset(
        root=osp.join(DATA_ROOT, "data/torchcell/sm_microarray_sameith2015"),
        io_workers=0,
    )
    logger.info(f"Loaded {len(sm_sameith)} experiments")

    sameith_data = extract_expression_data(sm_sameith, "SmSameith2015", sample_range)

    # Create Sameith plot
    logger.info("\n--- Creating Sameith Visualization ---")
    create_single_wide_boxplot(
        sameith_data,
        title=f"Sameith 2015: Log2 Expression Changes ({len(sameith_data)} GSTF Genes)",
        output_prefix="single_mutant_sameith",
        xlabel="GSTF Gene Deletion Strains",
    )

    logger.info("\n" + "=" * 80)
    logger.info("✓ VISUALIZATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nOutputs saved to: {osp.join(ASSET_IMAGES_DIR, '012-sameith-kemmeren-expression')}")


if __name__ == "__main__":
    main()
