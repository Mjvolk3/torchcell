# experiments/014-genes-enriched-at-extreme-tmi/scripts/analyze_extreme_interactions.py
# [[experiments.014-genes-enriched-at-extreme-tmi.scripts.analyze_extreme_interactions]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/014-genes-enriched-at-extreme-tmi/scripts/analyze_extreme_interactions

"""
Data processing script for extreme TMI analysis.
Loads dataset, analyzes interactions, and saves results to CSV files.
This is the slow step - only needs to be run once.
"""

import json
import os
import os.path as osp
from collections import Counter

import numpy as np
import pandas as pd
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

if DATA_ROOT is None:
    raise ValueError("DATA_ROOT environment variable not set")
if EXPERIMENT_ROOT is None:
    raise ValueError("EXPERIMENT_ROOT environment variable not set")


def load_uncharacterized_genes() -> set[str]:
    """Load uncharacterized genes from experiment 013."""
    assert EXPERIMENT_ROOT is not None, "EXPERIMENT_ROOT must be set"
    results_dir = osp.join(
        EXPERIMENT_ROOT, "013-uncharacterized-genes", "results"
    )
    with open(osp.join(results_dir, "uncharacterized_genes.json"), "r") as f:
        uncharacterized_data = json.load(f)
    return set(uncharacterized_data.keys())


def load_existing_dataset():
    """
    Load the existing Neo4j dataset from experiment 009/010.

    Returns:
        Neo4jCellDataset instance
    """
    assert DATA_ROOT is not None, "DATA_ROOT must be set"
    assert EXPERIMENT_ROOT is not None, "EXPERIMENT_ROOT must be set"

    # Use the existing dataset from experiment 009 (uses same data as 010)
    # Note: experiment 009's query.py uses 010's dataset/query paths
    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/010-kuzmin-tmi/001-small-build"
    )

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
    # Using experiment 009's query (which is actually stored in 010's query folder)
    print("Loading existing TMI dataset from experiment 009/010...")

    query_path = osp.join(
        EXPERIMENT_ROOT, "009-kuzmin-tmi", "queries", "001_small_build.cql"
    )

    if osp.exists(query_path):
        with open(query_path, "r") as f:
            query = f.read()
        print(f"Loaded query from: {query_path}")
    else:
        raise FileNotFoundError(f"Query file not found at {query_path}")

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


def analyze_extreme_interactions(
    dataset,
    uncharacterized_genes: set[str],
    score_threshold: float = 0.1,
    p_value_threshold: float = 0.05,
):
    """
    Analyze dataset for extreme interactions involving uncharacterized genes.
    Uses is_any_perturbed_gene_index for efficient lookup.

    Args:
        dataset: Neo4jCellDataset instance with is_any_perturbed_gene_index
        uncharacterized_genes: Set of uncharacterized gene IDs
        score_threshold: Minimum absolute score for extreme interactions
        p_value_threshold: Maximum p-value for significance filtering

    Returns:
        Tuple of (all_interactions_df, high_conf_interactions_df)
    """
    all_interactions = []  # |τ| > threshold with uncharacterized genes
    high_conf_interactions = []  # |τ| > threshold AND p < threshold with uncharacterized genes
    seen_indices = set()  # Track which interactions we've already processed

    print(f"\nAnalyzing extreme interactions (|τ| > {score_threshold}) involving uncharacterized genes...")
    print(f"Also tracking high-confidence subset with p < {p_value_threshold}")
    print(f"Using is_any_perturbed_gene_index for efficient lookup...")

    # Iterate through uncharacterized genes using index (MUCH faster!)
    for gene_id in tqdm(uncharacterized_genes, desc="Processing uncharacterized genes"):
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

            # Count uncharacterized genes in this interaction
            unchar_genes = [g for g in perturbed_genes if g in uncharacterized_genes]

            # Get interaction score and p-value
            gene_interaction = float(data["gene"].phenotype_values[0])

            # Check if p-value exists
            p_value = (
                float(data["gene"].phenotype_stat_values[0])
                if len(data["gene"].phenotype_stat_values) > 0
                else np.nan
            )

            # Filter by absolute score threshold
            if abs(gene_interaction) > score_threshold:
                interaction_dict = {
                    "idx": idx,
                    "num_genes": len(perturbed_genes),
                    "genes": ",".join(sorted(perturbed_genes)),
                    "num_uncharacterized": len(unchar_genes),
                    "uncharacterized_genes": ",".join(sorted(unchar_genes)),
                    "gene_interaction": gene_interaction,
                    "p_value": p_value,
                    "abs_score": abs(gene_interaction),
                }

                all_interactions.append(interaction_dict)

                # Also check if it meets high-confidence criteria
                if not np.isnan(p_value) and p_value < p_value_threshold:
                    high_conf_interactions.append(interaction_dict)

    print(f"\nFound {len(all_interactions)} extreme interactions with uncharacterized genes "
          f"(|τ| > {score_threshold})")
    print(f"Found {len(high_conf_interactions)} high-confidence extreme interactions "
          f"(|τ| > {score_threshold} AND p < {p_value_threshold})")

    return pd.DataFrame(all_interactions), pd.DataFrame(high_conf_interactions)


def count_gene_participation(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Count how many extreme interactions each uncharacterized gene participates in.

    Args:
        df: DataFrame with interaction data (must have 'uncharacterized_genes' column)
        label: Label for this analysis (e.g., "all" or "high_conf")

    Returns:
        DataFrame with uncharacterized gene interaction counts
    """
    gene_interaction_counts = Counter()

    for _, row in df.iterrows():
        # Only count uncharacterized genes
        unchar_genes = row["uncharacterized_genes"].split(",")
        for gene in unchar_genes:
            if gene:  # Skip empty strings
                gene_interaction_counts[gene] += 1

    # Create DataFrame for output
    gene_counts_df = pd.DataFrame(
        [
            {"gene": gene, "count": count}
            for gene, count in gene_interaction_counts.items()
        ]
    )
    gene_counts_df = gene_counts_df.sort_values("count", ascending=False)

    print(f"\n{label} Gene Participation Statistics (Uncharacterized Genes Only):")
    print(f"  Total unique uncharacterized genes: {len(gene_counts_df)}")
    print(f"  Mean extreme interactions per gene: {gene_counts_df['count'].mean():.2f}")
    print(f"  Median extreme interactions per gene: {gene_counts_df['count'].median():.0f}")
    print(f"  Max extreme interactions for a gene: {gene_counts_df['count'].max():.0f}")

    return gene_counts_df


def calculate_enrichment_fractions(
    dataset,
    extreme_gene_counts: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate the fraction of extreme interactions for each gene.

    Enrichment fraction = (# extreme interactions) / (# total interactions)

    Args:
        dataset: Neo4jCellDataset with is_any_perturbed_gene_index
        extreme_gene_counts: DataFrame with extreme interaction counts per gene

    Returns:
        DataFrame with genes, total counts, extreme counts, and enrichment fractions
    """
    print(f"\nCalculating enrichment fractions (extreme / total interactions)...")

    enrichment_data = []

    # Get genes that have extreme interactions
    genes_with_extremes = set(extreme_gene_counts["gene"])

    for gene_id in tqdm(genes_with_extremes, desc="Computing enrichment fractions"):
        # Get total number of interactions for this gene
        if gene_id not in dataset.is_any_perturbed_gene_index:
            continue

        total_interactions = len(dataset.is_any_perturbed_gene_index[gene_id])

        # Get extreme count from the counts DataFrame
        extreme_count = extreme_gene_counts[extreme_gene_counts["gene"] == gene_id]["count"].values[0]

        # Calculate enrichment fraction
        enrichment_fraction = extreme_count / total_interactions if total_interactions > 0 else 0.0

        enrichment_data.append({
            "gene": gene_id,
            "total_interactions": total_interactions,
            "extreme_interactions": extreme_count,
            "enrichment_fraction": enrichment_fraction,
        })

    # Create DataFrame and sort by enrichment fraction
    enrichment_df = pd.DataFrame(enrichment_data)
    enrichment_df = enrichment_df.sort_values("enrichment_fraction", ascending=False)

    print(f"\nEnrichment Fraction Statistics:")
    print(f"  Mean enrichment: {enrichment_df['enrichment_fraction'].mean():.3f}")
    print(f"  Median enrichment: {enrichment_df['enrichment_fraction'].median():.3f}")
    print(f"  Max enrichment: {enrichment_df['enrichment_fraction'].max():.3f}")

    # Show top 10 by enrichment fraction
    print(f"\nTop 10 genes by enrichment fraction:")
    for i, (_, row) in enumerate(enrichment_df.head(10).iterrows(), 1):
        print(f"  {i:2d}. {row['gene']:12s}: {row['enrichment_fraction']:.3f} "
              f"({row['extreme_interactions']:.0f}/{row['total_interactions']:.0f})")

    return enrichment_df


def main():
    """Main data processing function."""
    # Type assertions
    assert DATA_ROOT is not None, "DATA_ROOT must be set"
    assert EXPERIMENT_ROOT is not None, "EXPERIMENT_ROOT must be set"

    # Analysis parameters
    SCORE_THRESHOLD = 0.1
    P_VALUE_THRESHOLD = 0.05

    print("=" * 60)
    print("UNCHARACTERIZED GENES IN EXTREME TMI ANALYSIS")
    print("DATA PROCESSING STEP")
    print("=" * 60)
    print(f"Score threshold: |τ| > {SCORE_THRESHOLD}")
    print(f"P-value threshold: p < {P_VALUE_THRESHOLD}")
    print("=" * 60)

    # Load uncharacterized genes from experiment 013
    print("\nLoading uncharacterized genes from experiment 013...")
    uncharacterized_genes = load_uncharacterized_genes()
    print(f"Loaded {len(uncharacterized_genes)} uncharacterized genes")

    # Load existing dataset from experiment 009/010
    dataset = load_existing_dataset()

    # Analyze extreme interactions involving uncharacterized genes
    all_df, high_conf_df = analyze_extreme_interactions(
        dataset,
        uncharacterized_genes=uncharacterized_genes,
        score_threshold=SCORE_THRESHOLD,
        p_value_threshold=P_VALUE_THRESHOLD,
    )

    # Count gene participation for both criteria
    print("\n" + "=" * 60)
    print("GENE PARTICIPATION ANALYSIS")
    print("=" * 60)

    all_gene_counts = count_gene_participation(
        all_df, f"All Extreme (|τ| > {SCORE_THRESHOLD})"
    )

    high_conf_gene_counts = count_gene_participation(
        high_conf_df, f"High-Confidence (|τ| > {SCORE_THRESHOLD}, p < {P_VALUE_THRESHOLD})"
    )

    # Calculate enrichment fractions
    print("\n" + "=" * 60)
    print("ENRICHMENT FRACTION ANALYSIS")
    print("=" * 60)

    all_enrichment = calculate_enrichment_fractions(
        dataset,
        all_gene_counts,
    )

    high_conf_enrichment = None
    if len(high_conf_gene_counts) > 0:
        print("\nFor high-confidence criterion:")
        high_conf_enrichment = calculate_enrichment_fractions(
            dataset,
            high_conf_gene_counts,
        )

    # Show top genes for each criterion
    print("\n" + "=" * 60)
    print("TOP GENES BY ABSOLUTE COUNT")
    print("=" * 60)

    print(f"\nTop 20 genes - All criterion (|τ| > {SCORE_THRESHOLD}):")
    for i, (_, row) in enumerate(all_gene_counts.head(20).iterrows(), 1):
        print(f"  {i:2d}. {row['gene']:12s}: {row['count']:5d} interactions")

    if len(high_conf_gene_counts) > 0:
        print(f"\nTop 20 genes - High-confidence criterion (|τ| > {SCORE_THRESHOLD}, p < {P_VALUE_THRESHOLD}):")
        for i, (_, row) in enumerate(high_conf_gene_counts.head(20).iterrows(), 1):
            print(f"  {i:2d}. {row['gene']:12s}: {row['count']:5d} interactions")

    # Save results
    output_dir = osp.join(
        EXPERIMENT_ROOT, "014-genes-enriched-at-extreme-tmi", "results"
    )
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    # Save interaction data
    all_interactions_file = osp.join(output_dir, "uncharacterized_extreme_interactions_all.csv")
    all_df.to_csv(all_interactions_file, index=False)
    print(f"All extreme interactions:     {all_interactions_file}")

    high_conf_file = osp.join(output_dir, "uncharacterized_extreme_interactions_high_conf.csv")
    high_conf_df.to_csv(high_conf_file, index=False)
    print(f"High-confidence interactions: {high_conf_file}")

    # Save gene counts (similar to experiment 013 format)
    all_gene_counts_file = osp.join(output_dir, "uncharacterized_gene_counts_all.csv")
    all_gene_counts.to_csv(all_gene_counts_file, index=False)
    print(f"Gene counts (all):            {all_gene_counts_file}")

    high_conf_gene_counts_file = osp.join(output_dir, "uncharacterized_gene_counts_high_conf.csv")
    high_conf_gene_counts.to_csv(high_conf_gene_counts_file, index=False)
    print(f"Gene counts (high-conf):      {high_conf_gene_counts_file}")

    # Save enrichment fraction data
    all_enrichment_file = osp.join(output_dir, "uncharacterized_enrichment_fractions_all.csv")
    all_enrichment.to_csv(all_enrichment_file, index=False)
    print(f"Enrichment fractions (all):   {all_enrichment_file}")

    if high_conf_enrichment is not None:
        high_conf_enrichment_file = osp.join(output_dir, "uncharacterized_enrichment_fractions_high_conf.csv")
        high_conf_enrichment.to_csv(high_conf_enrichment_file, index=False)
        print(f"Enrichment fractions (high):  {high_conf_enrichment_file}")

    # Save summary statistics
    summary_file = osp.join(output_dir, "uncharacterized_extreme_tmi_summary.txt")
    with open(summary_file, "w") as f:
        f.write("UNCHARACTERIZED GENES IN EXTREME TMI ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        f.write("Parameters:\n")
        f.write(f"  Score threshold: |τ| > {SCORE_THRESHOLD}\n")
        f.write(f"  P-value threshold: p < {P_VALUE_THRESHOLD}\n")
        f.write(f"  Uncharacterized genes analyzed: {len(uncharacterized_genes)}\n\n")
        f.write("Results:\n")
        f.write(f"  Total interactions in dataset: {len(dataset)}\n")
        f.write(f"  Extreme interactions with uncharacterized genes (all): {len(all_df)}\n")
        f.write(f"  High-confidence extreme interactions: {len(high_conf_df)}\n")
        f.write(f"  Unique uncharacterized genes in extreme interactions (all): {len(all_gene_counts)}\n")
        f.write(f"  Unique uncharacterized genes in high-conf interactions: {len(high_conf_gene_counts)}\n\n")
        f.write("Top 10 uncharacterized genes (all criterion):\n")
        for i, (_, row) in enumerate(all_gene_counts.head(10).iterrows(), 1):
            f.write(f"  {i:2d}. {row['gene']:12s}: {row['count']:5d} extreme interactions\n")
        if len(high_conf_gene_counts) > 0:
            f.write("\nTop 10 uncharacterized genes (high-confidence criterion):\n")
            for i, (_, row) in enumerate(high_conf_gene_counts.head(10).iterrows(), 1):
                f.write(f"  {i:2d}. {row['gene']:12s}: {row['count']:5d} extreme interactions\n")

    print(f"Summary:                      {summary_file}")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("DATA PROCESSING COMPLETE")
    print("=" * 60)
    print("\nYou can now run visualize_extreme_interactions.py to create plots.")

    return all_df, high_conf_df, all_gene_counts, high_conf_gene_counts, all_enrichment


if __name__ == "__main__":
    results = main()
