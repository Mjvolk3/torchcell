# experiments/010-kuzmin-tmi/scripts/inferred_triple_interaction_histogram_from_gene_selection
# [[experiments.010-kuzmin-tmi.scripts.inferred_triple_interaction_histogram_from_gene_selection]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/inferred_triple_interaction_histogram_from_gene_selection
# Test file: experiments/010-kuzmin-tmi/scripts/test_inferred_triple_interaction_histogram_from_gene_selection.py

"""
Plot histograms of inferred gene interaction scores for selected gene panels.

For each selected gene panel (12 and 24 genes), generates ALL possible triples
and looks up their inferred interaction scores from the inference data.

12-gene panel: C(12,3) = 220 possible triples
24-gene panel: C(24,3) = 2024 possible triples
"""

import math
import os
import os.path as osp
from glob import glob
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from torchcell.timestamp import timestamp

load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")

# Validate environment variables
assert DATA_ROOT is not None, "DATA_ROOT environment variable not set"
assert ASSET_IMAGES_DIR is not None, "ASSET_IMAGES_DIR environment variable not set"
assert EXPERIMENT_ROOT is not None, "EXPERIMENT_ROOT environment variable not set"


def load_inference_data() -> pd.DataFrame:
    """Load the inference parquet file for the best model (Pearson=0.4619)."""
    inference_dir = osp.join(
        DATA_ROOT, "data/torchcell/experiments/010-kuzmin-tmi/inference_1/inferred"
    )

    # Find the Pearson=0.4619 parquet file
    pattern = osp.join(inference_dir, "*Pearson=0.4619*.parquet")
    files = glob(pattern)

    if not files:
        raise FileNotFoundError(f"No parquet file matching pattern: {pattern}")

    parquet_path = files[0]
    print(f"Loading inference data from: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df):,} inferred triples")

    return df


def load_gene_selection_results() -> pd.DataFrame:
    """Load gene selection results to get selected gene panels."""
    results_path = osp.join(
        EXPERIMENT_ROOT, "010-kuzmin-tmi/results/gene_selection_results.csv"
    )
    df = pd.read_csv(results_path)
    print(f"Loaded gene selection results: {len(df)} rows")
    return df


def create_triple_lookup(df: pd.DataFrame) -> dict[frozenset, float]:
    """
    Create a lookup dictionary from triples to their prediction values.

    Args:
        df: DataFrame with gene1, gene2, gene3, prediction columns

    Returns:
        Dict mapping frozenset({gene1, gene2, gene3}) -> prediction
    """
    print("Creating triple lookup dictionary...")
    lookup = {}
    for _, row in df.iterrows():
        triple = frozenset([row["gene1"], row["gene2"], row["gene3"]])
        lookup[triple] = row["prediction"]
    print(f"Created lookup with {len(lookup):,} triples")
    return lookup


def get_all_panel_triples(
    genes: list[str], triple_lookup: dict[frozenset, float]
) -> list[float]:
    """
    Get prediction values for all possible triples from a gene panel.

    Args:
        genes: List of genes in the panel
        triple_lookup: Dict mapping triple frozensets to prediction values

    Returns:
        List of prediction values for all constructible triples
    """
    predictions = []
    missing = 0

    for triple in combinations(sorted(genes), 3):
        triple_set = frozenset(triple)
        if triple_set in triple_lookup:
            predictions.append(triple_lookup[triple_set])
        else:
            missing += 1

    if missing > 0:
        print(f"  Warning: {missing} triples not found in inference data")

    return predictions


def parse_selected_genes(genes_str: str) -> list[str]:
    """Parse comma-separated gene string into list."""
    return [g.strip() for g in genes_str.split(",")]


def plot_panel_histogram(
    predictions_by_k: dict[int, list[float]],
    panel_size: int,
    output_dir: str,
    bins: np.ndarray,
) -> None:
    """
    Plot histograms of inferred predictions for a gene panel across k values.

    Args:
        predictions_by_k: Dict mapping k value to list of inferred predictions
        panel_size: Number of genes (12 or 24)
        output_dir: Directory to save plot
        bins: Bin edges for histogram
    """
    n_plots = len(predictions_by_k)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False
    )

    design_space = math.comb(panel_size, 3)

    for idx, (k, inferred) in enumerate(sorted(predictions_by_k.items())):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        n_inferred = len(inferred)

        if inferred:
            ax.hist(
                inferred,
                bins=bins,
                color="#2166ac",
                edgecolor="black",
                alpha=0.7,
                label=f"inferred: {n_inferred}/{design_space}",
            )

            # Add gene interaction = 0 reference line
            ax.axvline(
                x=0, color="red", linestyle="--", alpha=0.5, label="gene interaction = 0"
            )

            # Add mean line
            mean_inferred = np.mean(inferred)
            ax.axvline(
                x=mean_inferred,
                color="green",
                linestyle="-",
                alpha=0.8,
                linewidth=2,
                label=f"mean={mean_inferred:.3f}",
            )

            ax.set_title(f"k={k}")
            ax.set_xlabel("Predicted Gene Interaction")
            ax.set_ylabel("Frequency")
            ax.legend(fontsize=7, loc="upper right")
        else:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title(f"k={k}")

    # Hide empty subplots
    for idx in range(n_plots, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle(
        f"Inferred Gene Interaction Distribution: {panel_size}-Gene Panel\n"
        f"(Design space = {design_space} triples)",
        fontsize=14,
    )
    plt.tight_layout()

    output_path = osp.join(
        output_dir, f"inferred_interactions_panel{panel_size}_{timestamp()}.png"
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    """Main execution function."""
    print("=" * 60)
    print("Inferred Triple Interaction Histogram from Gene Selection")
    print("=" * 60)

    # Create output directory
    plots_dir = osp.join(ASSET_IMAGES_DIR, "010-kuzmin-tmi")
    os.makedirs(plots_dir, exist_ok=True)

    # Load data
    print("\n" + "=" * 60)
    print("Loading data...")
    print("=" * 60)
    inference_df = load_inference_data()
    selection_df = load_gene_selection_results()

    # Create triple lookup for inferred predictions
    triple_lookup = create_triple_lookup(inference_df)

    # Define bins based on data distribution
    bins = np.linspace(-0.2, 0.15, 36)

    # Process each panel size
    for panel_size in [12, 24]:
        print(f"\n{'=' * 60}")
        print(f"Processing {panel_size}-gene panel")
        print(f"Design space: C({panel_size},3) = {math.comb(panel_size, 3)} triples")
        print("=" * 60)

        panel_df = selection_df[selection_df["panel_size"] == panel_size]

        if panel_df.empty:
            print(f"No data for panel_size={panel_size}")
            continue

        predictions_by_k = {}

        for _, row in panel_df.iterrows():
            k = int(row["k"])
            genes = parse_selected_genes(str(row["selected_genes"]))

            print(f"\n  k={k}: {len(genes)} genes selected")

            # Get inferred predictions
            predictions = get_all_panel_triples(genes, triple_lookup)
            predictions_by_k[k] = predictions

            print(f"    Inferred: {len(predictions)} triples")
            if predictions:
                print(
                    f"    Mean: {np.mean(predictions):.4f}, "
                    f"Std: {np.std(predictions):.4f}"
                )

        # Plot histograms for this panel
        plot_panel_histogram(predictions_by_k, panel_size, plots_dir, bins)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
