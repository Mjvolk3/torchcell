# experiments/013-uncharacterized-genes/scripts/triple_interaction_enrichment_of_uncharacterized_genes.py
# [[experiments.013-uncharacterized-genes.scripts.triple_interaction_enrichment_of_uncharacterized_genes]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/013-uncharacterized-genes/scripts/triple_interaction_enrichment_of_uncharacterized_genes

import json
import os
import os.path as osp
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from tqdm import tqdm

from torchcell.data import GenotypeAggregator, MeanExperimentDeduplicator
from torchcell.data.graph_processor import SubgraphRepresentation
from torchcell.data.neo4j_cell import Neo4jCellDataset
from torchcell.datasets import CodonFrequencyDataset
from torchcell.datasets.fungal_up_down_transformer import (
    FungalUpDownTransformerDataset,
)
from torchcell.metabolism.yeast_GEM import YeastGEM
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")

if DATA_ROOT is None:
    raise ValueError("DATA_ROOT environment variable not set")
if EXPERIMENT_ROOT is None:
    raise ValueError("EXPERIMENT_ROOT environment variable not set")
if ASSET_IMAGES_DIR is None:
    raise ValueError("ASSET_IMAGES_DIR environment variable not set")

# Custom colors from torchcell.mplstyle
CUSTOM_BLUE = "#7191A9"
CUSTOM_GREEN = "#6B8D3A"


def load_uncharacterized_genes() -> set[str]:
    """Load uncharacterized genes from JSON file."""
    assert EXPERIMENT_ROOT is not None, "EXPERIMENT_ROOT must be set"
    results_dir = osp.join(
        EXPERIMENT_ROOT, "013-uncharacterized-genes", "results"
    )
    with open(osp.join(results_dir, "uncharacterized_genes.json"), "r") as f:
        uncharacterized_data = json.load(f)
    return set(uncharacterized_data.keys())


def load_existing_dataset():
    """
    Load the existing Neo4j dataset from experiment 010.

    Returns:
        Neo4jCellDataset instance
    """
    assert DATA_ROOT is not None, "DATA_ROOT must be set"
    assert EXPERIMENT_ROOT is not None, "EXPERIMENT_ROOT must be set"

    # Use the existing dataset from experiment 010
    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/010-kuzmin-tmi/001-small-build"
    )

    # Load query (needed for dataset initialization)
    query_path = osp.join(
        EXPERIMENT_ROOT, "010-kuzmin-tmi", "queries", "001_small_build.cql"
    )
    with open(query_path, "r") as f:
        query = f.read()

    # Initialize genome
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )

    print(f"Genome gene set size: {len(genome.gene_set)}")

    # Initialize embeddings (needed for dataset initialization)
    fudt_3prime_dataset = FungalUpDownTransformerDataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/fudt_embedding"),
        genome=genome,
        model_name="species_downstream",
    )
    fudt_5prime_dataset = FungalUpDownTransformerDataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/fudt_embedding"),
        genome=genome,
        model_name="species_upstream",
    )
    codon_frequency = CodonFrequencyDataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/codon_frequency_embedding"),
        genome=genome,
    )

    # Load existing dataset (will NOT reprocess if data exists)
    print("Loading existing trigenic interaction dataset from experiment 010...")
    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        uri="bolt://torchcell-database.ncsa.illinois.edu:7687",
        username="readonly",
        password="ReadOnly",
        graphs=None,
        incidence_graphs={"metabolism_bipartite": YeastGEM().bipartite_graph},
        node_embeddings={
            "codon_frequency": codon_frequency,
            "fudt_3prime": fudt_3prime_dataset,
            "fudt_5prime": fudt_5prime_dataset,
        },
        converter=None,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=GenotypeAggregator,
        graph_processor=SubgraphRepresentation(),
    )

    print(f"Dataset size: {len(dataset)}")
    return dataset


def analyze_dataset(dataset, uncharacterized_genes: set[str]):
    """
    Analyze dataset for uncharacterized gene involvement using index.

    Args:
        dataset: Neo4jCellDataset instance with is_any_perturbed_gene_index
        uncharacterized_genes: Set of uncharacterized gene IDs

    Returns:
        List of interaction dictionaries
    """
    interactions = []
    seen_indices = set()  # Track which interactions we've already processed

    print(f"\nFinding interactions involving uncharacterized genes...")
    print(f"Using is_any_perturbed_gene_index for efficient lookup...")

    # Iterate through uncharacterized genes
    for gene_id in tqdm(uncharacterized_genes, desc="Processing genes"):
        # Check if this gene has any interactions
        if gene_id not in dataset.is_any_perturbed_gene_index:
            continue

        # Get all interaction indices involving this gene
        gene_indices = dataset.is_any_perturbed_gene_index[gene_id]

        # Process each interaction (but avoid duplicates)
        for idx in gene_indices:
            if idx in seen_indices:
                continue  # Already processed this interaction
            seen_indices.add(idx)

            # Get the interaction data
            data = dataset[idx]

            # Get perturbed genes from HeteroData
            perturbed_genes = data["gene"].ids_pert

            if len(perturbed_genes) != 3:
                continue  # Skip non-trigenic

            # Get interaction data from HeteroData
            gene_interaction = float(data["gene"].phenotype_values[0])

            # Get p-value (check if it exists)
            p_value = (
                float(data["gene"].phenotype_stat_values[0])
                if len(data["gene"].phenotype_stat_values) > 0
                else np.nan
            )

            # Count uncharacterized genes in this interaction
            unchar_genes = [
                g for g in perturbed_genes if g in uncharacterized_genes
            ]
            num_uncharacterized = len(unchar_genes)

            interactions.append(
                {
                    "idx": idx,
                    "gene_1": perturbed_genes[0],
                    "gene_2": perturbed_genes[1],
                    "gene_3": perturbed_genes[2],
                    "gene_interaction": gene_interaction,
                    "p_value": p_value,
                    "num_uncharacterized": num_uncharacterized,
                    "uncharacterized_genes": ",".join(sorted(unchar_genes)),
                }
            )

    print(f"Found {len(interactions)} unique interactions involving uncharacterized genes")
    return interactions


def create_visualizations(df: pd.DataFrame, output_dir: str):
    """
    Create visualizations for interaction analysis.

    Args:
        df: DataFrame with interaction data
        output_dir: Base directory to save plots (will create subdirectory)
    """
    # Create experiment-specific subdirectory
    plot_output_dir = osp.join(output_dir, "013-uncharacterized-genes")
    os.makedirs(plot_output_dir, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 300

    # 1. Distribution of interaction scores
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(hspace=0.35, wspace=0.3)  # Add spacing between subplots

    # All interactions
    ax = axes[0, 0]
    ax.hist(
        df["gene_interaction"].dropna(),
        bins=50,
        color=CUSTOM_BLUE,
        edgecolor="black",
        alpha=0.7,
    )
    ax.axvline(0, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel(r"Triple Mutant Interaction Score $(\tau)$")
    ax.set_ylabel("Frequency")
    ax.set_title("All Interaction Scores\n(n=668 uncharacterized genes)", fontsize=11)
    ax.grid(True, alpha=0.3)

    # By number of uncharacterized genes
    ax = axes[0, 1]
    # Use consistent color palette - shades of blue for different counts
    colors = ["#5A7A8C", "#A97171", "#7A8C5A"]  # blue-gray, muted red, muted green
    for i, num_unchar in enumerate(sorted(df["num_uncharacterized"].unique())):
        subset = df[df["num_uncharacterized"] == num_unchar][
            "gene_interaction"
        ].dropna()
        ax.hist(
            subset,
            bins=50,
            alpha=0.6,
            label=f"{num_unchar} uncharacterized",
            edgecolor="black",
            color=colors[i % len(colors)],
        )
    ax.axvline(0, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel(r"Triple Mutant Interaction Score $(\tau)$")
    ax.set_ylabel("Frequency")
    ax.set_title("By # Uncharacterized Genes\n(n=668 total)", fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # High-confidence interactions (p < 0.01)
    ax = axes[1, 0]
    high_conf = df[df["p_value"] < 0.01]["gene_interaction"].dropna()
    if len(high_conf) > 0:
        ax.hist(
            high_conf, bins=30, color=CUSTOM_BLUE, edgecolor="black", alpha=0.7
        )
        ax.axvline(0, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel(r"Triple Mutant Interaction Score $(\tau)$")
    ax.set_ylabel("Frequency")
    ax.set_title(
        f"High-Confidence Interactions\n(p < 0.01, n={len(high_conf)})", fontsize=11
    )
    ax.grid(True, alpha=0.3)

    # P-value distribution
    ax = axes[1, 1]
    p_values = df["p_value"].dropna()
    if len(p_values) > 0:
        ax.hist(
            p_values,
            bins=50,
            color=CUSTOM_BLUE,
            edgecolor="black",
            alpha=0.7,
        )
        ax.axvline(0.01, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("P-value")
    ax.set_ylabel("Frequency")
    ax.set_title("P-value Distribution\n(n=668 uncharacterized genes)", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Don't use tight_layout since we set spacing manually with subplots_adjust
    fig1_path = osp.join(
        plot_output_dir,
        "interaction_score_distributions.png",
    )
    plt.savefig(fig1_path)
    plt.close()
    print(f"  Saved: {fig1_path}")

    # 2. Histogram of interaction counts per gene
    gene_interaction_counts = Counter()
    for _, row in df.iterrows():
        unchar_genes = row["uncharacterized_genes"].split(",")
        for gene in unchar_genes:
            if gene:  # Skip empty strings
                gene_interaction_counts[gene] += 1

    # Create DataFrame for plotting
    gene_counts_df = pd.DataFrame(
        [
            {"gene": gene, "count": count}
            for gene, count in gene_interaction_counts.items()
        ]
    )
    gene_counts_df = gene_counts_df.sort_values("count", ascending=False)

    _, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Histogram of interaction counts
    ax = axes[0]
    ax.hist(
        gene_counts_df["count"],
        bins=50,
        color=CUSTOM_BLUE,
        edgecolor="black",
        alpha=0.7,
    )
    ax.set_xlabel("Number of Interactions per Gene")
    ax.set_ylabel("Number of Genes")
    ax.set_title("Interaction Counts per Gene\n(n=668 uncharacterized)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.text(
        0.95,
        0.95,
        f"Total genes: {len(gene_counts_df)}\n"
        f"Mean: {gene_counts_df['count'].mean():.1f}\n"
        f"Median: {gene_counts_df['count'].median():.0f}\n"
        f"Max: {gene_counts_df['count'].max():.0f}",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            edgecolor="#666666",
            alpha=0.95,
            linewidth=1.2,
        ),
        fontfamily="sans-serif",
        fontweight="normal",
    )

    # Top 20 most connected genes
    ax = axes[1]
    top_20 = gene_counts_df.head(20)
    ax.barh(range(len(top_20)), top_20["count"], color=CUSTOM_BLUE, alpha=0.7)
    ax.set_yticks(range(len(top_20)))
    ax.set_yticklabels(top_20["gene"], fontsize=8)
    ax.set_xlabel("Number of Interactions")
    ax.set_title("Top 20 Most Connected Genes\n(of 668 uncharacterized)", fontsize=11)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    fig2_path = osp.join(
        plot_output_dir,
        "gene_interaction_counts.png",
    )
    plt.savefig(fig2_path)
    plt.close()
    print(f"  Saved: {fig2_path}")

    return gene_counts_df


def main():
    """Main analysis function."""
    # Type assertions
    assert DATA_ROOT is not None, "DATA_ROOT must be set"
    assert EXPERIMENT_ROOT is not None, "EXPERIMENT_ROOT must be set"
    assert ASSET_IMAGES_DIR is not None, "ASSET_IMAGES_DIR must be set"

    # Load uncharacterized genes
    print("Loading uncharacterized genes...")
    uncharacterized_genes = load_uncharacterized_genes()
    print(f"Loaded {len(uncharacterized_genes)} uncharacterized genes")

    # Load existing dataset from experiment 010
    dataset = load_existing_dataset()

    # Analyze interactions
    interactions = analyze_dataset(dataset, uncharacterized_genes)
    df = pd.DataFrame(interactions)

    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"Total interactions with uncharacterized genes: {len(df)}")
    print(f"\nBreakdown by # uncharacterized genes:")
    print(df["num_uncharacterized"].value_counts().sort_index())

    # Statistics
    print(f"\nInteraction score statistics:")
    print(df["gene_interaction"].describe())

    # Filter for valid p-values
    valid_p_values = df["p_value"].notna()
    print(
        f"\nInteractions with valid p-values: {valid_p_values.sum()} / {len(df)}"
    )
    if valid_p_values.sum() > 0:
        print(
            f"High-confidence interactions (p < 0.01): "
            f"{len(df[(df['p_value'] < 0.01) & valid_p_values])}"
        )

    print(
        f"Interactions with 2+ uncharacterized: "
        f"{len(df[df['num_uncharacterized'] >= 2])}"
    )
    print(
        f"Interactions with all 3 uncharacterized: "
        f"{len(df[df['num_uncharacterized'] == 3])}"
    )

    # Top interactions
    print(f"\nTop 10 strongest positive interactions:")
    for _, row in df.nlargest(10, "gene_interaction").iterrows():
        p_str = f"{row['p_value']:.4e}" if not np.isnan(row["p_value"]) else "N/A"
        print(
            f"  {row['gene_1']} × {row['gene_2']} × {row['gene_3']}: "
            f"score={row['gene_interaction']:.4f}, p={p_str}"
        )
        print(f"    Uncharacterized: {row['uncharacterized_genes']}")

    print(f"\nTop 10 strongest negative interactions:")
    for _, row in df.nsmallest(10, "gene_interaction").iterrows():
        p_str = f"{row['p_value']:.4e}" if not np.isnan(row["p_value"]) else "N/A"
        print(
            f"  {row['gene_1']} × {row['gene_2']} × {row['gene_3']}: "
            f"score={row['gene_interaction']:.4f}, p={p_str}"
        )
        print(f"    Uncharacterized: {row['uncharacterized_genes']}")

    # Save results
    output_dir = osp.join(
        EXPERIMENT_ROOT, "013-uncharacterized-genes", "results"
    )
    os.makedirs(output_dir, exist_ok=True)

    csv_file = osp.join(output_dir, "uncharacterized_triple_interactions.csv")
    df.to_csv(csv_file, index=False)
    print(f"\nSaved interaction data to: {csv_file}")

    # Create visualizations
    print("\nCreating visualizations:")
    gene_counts_df = create_visualizations(df, ASSET_IMAGES_DIR)

    # Save gene interaction counts
    gene_counts_file = osp.join(output_dir, "gene_interaction_counts.csv")
    gene_counts_df.to_csv(gene_counts_file, index=False)
    print(f"Saved gene interaction counts to: {gene_counts_file}")

    return df, gene_counts_df


if __name__ == "__main__":
    df, gene_counts_df = main()
