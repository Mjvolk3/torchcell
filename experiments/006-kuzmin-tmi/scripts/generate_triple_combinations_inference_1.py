#!/usr/bin/env python
"""
Generate triple gene combinations for inference_1 using adjacency graph filtering.

Key differences from original:
- FITNESS_THRESHOLD = 1.0 (was 0.5) - maximum stringency
- Adjacency graph approach - only generates valid triples
- Missing DMF data = invalid pair (aggressive strategy)
- Uses expanded gene list (~2261 genes from 4 sources)
- TMI filtering against 009-kuzmin-tmi (deletion-only dataset)

Strategy:
1. Build graph of valid pairs (fitness = 1.0 in DMF datasets)
2. Use mutual neighbors to generate only feasible triples
3. Filter out triples existing in 009-kuzmin-tmi TMI dataset
"""

import os
import os.path as osp
from itertools import combinations
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from torchcell.timestamp import timestamp
from torchcell.datasets.scerevisiae.kuzmin2018 import DmfKuzmin2018Dataset
from torchcell.datasets.scerevisiae.costanzo2016 import DmfCostanzo2016Dataset
from torchcell.datasets.scerevisiae.kuzmin2020 import DmfKuzmin2020Dataset
import matplotlib.pyplot as plt
import numpy as np

# Import for TMI filtering
from torchcell.data import Neo4jCellDataset, MeanExperimentDeduplicator, GenotypeAggregator
from torchcell.data.graph_processor import SubgraphRepresentation
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

# Load environment
load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR", "notes/assets/images")

FITNESS_THRESHOLD = 1.0  # Maximum stringency: only pairs with perfect fitness (1.0) are valid


def load_selected_genes(gene_list_file):
    """Load the selected genes from the text file."""
    print(f"Loading selected genes from {gene_list_file}")
    with open(gene_list_file, 'r') as f:
        selected_genes = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(selected_genes)} selected genes")
    return selected_genes


def build_valid_pairs_from_dmf(dataset_name, dataset_class, selected_genes):
    """
    Build set of valid pairs (fitness >= threshold) from DMF dataset.
    Returns dict mapping pairs to their fitness values.
    """
    print(f"\nProcessing {dataset_name}...")

    dataset = dataset_class(root=osp.join(DATA_ROOT, f"data/torchcell/{dataset_name}"))

    # Access the preprocessed dataframe if available
    preprocess_path = osp.join(dataset.preprocess_dir, "data.csv")
    if not osp.exists(preprocess_path):
        print(f"Warning: No preprocessed dataframe found for {dataset_name}")
        return {}

    print(f"  Loading preprocessed dataframe from {preprocess_path}")
    df = pd.read_csv(preprocess_path)

    # Extract systematic names based on dataset structure
    if 'Query Systematic Name' in df.columns and 'Array Systematic Name' in df.columns:
        # Costanzo2016 format
        df['gene1'] = df['Query Systematic Name']
        df['gene2'] = df['Array Systematic Name']
        fitness_col = 'Double mutant fitness'
    elif 'Query systematic name no ho' in df.columns and 'Array systematic name' in df.columns:
        # Kuzmin2018/2020 DMF format
        df['gene1'] = df['Query systematic name no ho']
        df['gene2'] = df['Array systematic name']
        # Check which fitness column exists
        if 'Combined mutant fitness' in df.columns:
            fitness_col = 'Combined mutant fitness'  # Kuzmin2018
        elif 'fitness' in df.columns:
            fitness_col = 'fitness'  # Kuzmin2020
        else:
            print(f"  Warning: No fitness column found in {dataset_name}")
            return {}
    else:
        print(f"  Warning: Unknown dataframe format for {dataset_name}")
        return {}

    # Filter to only include pairs where both genes are in selected set
    df_filtered = df[(df['gene1'].isin(selected_genes)) & (df['gene2'].isin(selected_genes))]
    print(f"  Filtered from {len(df)} to {len(df_filtered)} pairs with selected genes")

    # Build pair fitness dictionary
    pair_fitness = {}

    for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc=f"  Building {dataset_name} pairs"):
        gene1, gene2 = row['gene1'], row['gene2']
        fitness = row[fitness_col]

        # Store as sorted tuple for consistency
        pair = tuple(sorted([gene1, gene2]))

        # Track maximum fitness across datasets (most optimistic)
        if pair not in pair_fitness or fitness > pair_fitness[pair]:
            pair_fitness[pair] = fitness

    print(f"  Found {len(pair_fitness)} unique pairs in {dataset_name}")

    return pair_fitness


def build_valid_pairs_graph(selected_genes, dmf_datasets):
    """
    Build adjacency graph of valid pairs.
    Only includes pairs with EXPLICIT fitness >= FITNESS_THRESHOLD in DMF datasets.
    Missing data = invalid pair (aggressive strategy).
    """
    print("\n" + "="*80)
    print(f"Building valid pairs graph (fitness >= {FITNESS_THRESHOLD})")
    print("="*80)

    # Combine all DMF data to get fitness for each pair
    all_pair_fitness = {}

    for dataset_name, pair_fitness in dmf_datasets.items():
        for pair, fitness in pair_fitness.items():
            # Track MAXIMUM fitness across datasets (most optimistic)
            if pair not in all_pair_fitness or fitness > all_pair_fitness[pair]:
                all_pair_fitness[pair] = fitness

    # Build valid pairs: only those with fitness >= FITNESS_THRESHOLD
    valid_pairs = {pair for pair, fitness in all_pair_fitness.items()
                   if fitness >= FITNESS_THRESHOLD}

    print(f"\nPair statistics:")
    print(f"  Pairs with DMF data: {len(all_pair_fitness):,}")
    print(f"  Valid pairs (fitness >= {FITNESS_THRESHOLD}): {len(valid_pairs):,}")
    print(f"  Invalid pairs (fitness < {FITNESS_THRESHOLD}): {len(all_pair_fitness) - len(valid_pairs):,}")
    print(f"  Reduction: {(1 - len(valid_pairs)/len(all_pair_fitness))*100:.1f}%")

    # Build adjacency list
    adjacency = defaultdict(set)
    for (g1, g2) in valid_pairs:
        adjacency[g1].add(g2)
        adjacency[g2].add(g1)

    # Report genes with poor connectivity
    connectivity_stats = {gene: len(neighbors)
                         for gene, neighbors in adjacency.items()}

    print(f"\nGene connectivity statistics:")
    print(f"  Genes with valid pairs: {len(connectivity_stats)}")
    print(f"  Genes with NO valid pairs: {len(selected_genes) - len(connectivity_stats)}")

    if connectivity_stats:
        conn_values = list(connectivity_stats.values())
        print(f"  Avg valid pairs per gene: {np.mean(conn_values):.1f}")
        print(f"  Median: {np.median(conn_values):.1f}")
        print(f"  Min: {min(conn_values)}")
        print(f"  Max: {max(conn_values)}")

        # Show genes with low connectivity
        low_connectivity = {gene: count for gene, count in connectivity_stats.items() if count < 10}
        if low_connectivity:
            print(f"  Genes with < 10 valid pairs: {len(low_connectivity)}")

    return valid_pairs, adjacency, all_pair_fitness


def generate_triples_from_adjacency(selected_genes, adjacency):
    """
    Generate triples using adjacency graph.
    Only generates triples where all 3 constituent pairs are valid.

    Uses mutual neighbors approach:
    - For each gene g1, iterate through its neighbors g2
    - Find genes g3 that are neighbors of BOTH g1 AND g2
    - This ensures all three pairs (g1,g2), (g1,g3), (g2,g3) are valid
    """
    print("\n" + "="*80)
    print("Generating triples from valid pair graph...")
    print("="*80)

    triples = []
    genes_sorted = sorted(selected_genes)  # For consistent ordering

    for g1 in tqdm(genes_sorted, desc="Processing genes"):
        if g1 not in adjacency:
            continue  # Gene has no valid pairs

        neighbors_g1 = adjacency[g1]
        neighbors_g1_sorted = sorted(neighbors_g1)

        for i, g2 in enumerate(neighbors_g1_sorted):
            if g2 <= g1:  # Maintain ordering to avoid duplicates
                continue

            # Find mutual neighbors (genes connected to BOTH g1 AND g2)
            mutual_neighbors = neighbors_g1 & adjacency[g2]

            for g3 in mutual_neighbors:
                if g3 > g2:  # Maintain ordering (g1 < g2 < g3)
                    triples.append((g1, g2, g3))

    print(f"\nGenerated {len(triples):,} valid triples")
    print(f"(All constituent pairs have fitness >= {FITNESS_THRESHOLD})")

    return triples


def check_triple_exists_in_tmi(triple, is_any_perturbed_gene_index):
    """
    Check if a triple exists in TMI datasets by looking for common indices.

    Args:
        triple: Tuple of three gene names
        is_any_perturbed_gene_index: Dict mapping gene names to lists of dataset indices

    Returns:
        bool: True if all three genes share at least one common index (triple exists)
    """
    # Get indices for each gene
    indices_gene1 = set(is_any_perturbed_gene_index.get(triple[0], []))
    indices_gene2 = set(is_any_perturbed_gene_index.get(triple[1], []))
    indices_gene3 = set(is_any_perturbed_gene_index.get(triple[2], []))

    # Check if there's a common index where all three genes are perturbed
    common_indices = indices_gene1 & indices_gene2 & indices_gene3

    return len(common_indices) > 0


def filter_existing_tmi_triples(triples, is_any_perturbed_gene_index):
    """Filter out triples that already exist in TMI datasets."""
    print("\n" + "="*80)
    print("Filtering triples to remove those that exist in TMI datasets...")
    print("="*80)

    filtered_triples = []
    tmi_removed_triples = []
    rejected_count = 0

    for triple in tqdm(triples, desc="Checking TMI existence"):
        if not check_triple_exists_in_tmi(triple, is_any_perturbed_gene_index):
            filtered_triples.append(triple)
        else:
            tmi_removed_triples.append(triple)
            rejected_count += 1

    print(f"\nTMI filtering complete:")
    print(f"  Kept {len(filtered_triples):,} triples (not in TMI datasets)")
    print(f"  Rejected {rejected_count:,} triples (already exist in TMI datasets)")
    if len(triples) > 0:
        print(f"  Rejection rate: {rejected_count/len(triples)*100:.2f}%")

    return filtered_triples, tmi_removed_triples


def create_visualizations(selected_genes, dmf_filtered_triples, final_triples,
                         valid_pairs, all_pair_fitness, adjacency, ts):
    """Create comprehensive visualizations of the filtering process."""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Triple filtering flow
    max_possible = len(list(combinations(selected_genes, 3)))
    filtering_stages = [
        ('Max Possible\n(C(n,3))', max_possible),
        ('After DMF\nGraph Filter', len(dmf_filtered_triples)),
        ('After TMI\nFilter', len(final_triples))
    ]
    stages, counts = zip(*filtering_stages)
    colors = ['#d62728', '#ff7f0e', '#2ca02c']
    bars = ax1.bar(stages, counts, color=colors)

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}',
                ha='center', va='bottom', fontsize=10)

    ax1.set_ylabel('Number of Triples', fontsize=12)
    ax1.set_title(f'Triple Filtering Pipeline (Threshold: {FITNESS_THRESHOLD})', fontsize=14)
    ax1.set_yscale('log')

    # 2. Pair fitness distribution
    fitness_values = list(all_pair_fitness.values())
    ax2.hist(fitness_values, bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(FITNESS_THRESHOLD, color='red', linestyle='--', linewidth=2,
                label=f'Threshold: {FITNESS_THRESHOLD}')
    ax2.set_xlabel('DMF Pair Fitness', fontsize=12)
    ax2.set_ylabel('Number of Pairs', fontsize=12)
    ax2.set_title('Distribution of DMF Pair Fitness', fontsize=14)
    ax2.legend()

    # Add statistics
    valid_count = sum(1 for f in fitness_values if f >= FITNESS_THRESHOLD)
    ax2.text(0.05, 0.95,
             f'Total pairs: {len(fitness_values):,}\n'
             f'Valid (>={FITNESS_THRESHOLD}): {valid_count:,}\n'
             f'Invalid (<{FITNESS_THRESHOLD}): {len(fitness_values)-valid_count:,}',
             transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 3. Gene connectivity distribution
    if adjacency:
        connectivity_values = [len(neighbors) for neighbors in adjacency.values()]
        ax3.hist(connectivity_values, bins=50, edgecolor='black', alpha=0.7)
        ax3.set_xlabel('Number of Valid Pairs per Gene', fontsize=12)
        ax3.set_ylabel('Number of Genes', fontsize=12)
        ax3.set_title('Gene Connectivity Distribution', fontsize=14)
        ax3.axvline(np.mean(connectivity_values), color='red', linestyle='--',
                   label=f'Mean: {np.mean(connectivity_values):.1f}')
        ax3.axvline(np.median(connectivity_values), color='green', linestyle='--',
                   label=f'Median: {np.median(connectivity_values):.1f}')
        ax3.legend()

    # 4. Filtering breakdown pie chart
    dmf_removed = max_possible - len(dmf_filtered_triples)
    tmi_removed = len(dmf_filtered_triples) - len(final_triples)
    kept = len(final_triples)

    sizes = [kept, dmf_removed, tmi_removed]
    labels = [f'Kept\n({kept:,})',
              f'DMF Filtered\n({dmf_removed:,})',
              f'TMI Filtered\n({tmi_removed:,})']
    colors_pie = ['#2ca02c', '#ff7f0e', '#d62728']
    explode = (0.1, 0, 0)

    ax4.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
            startangle=90, explode=explode)
    ax4.set_title('Triple Filtering Breakdown', fontsize=14)

    plt.tight_layout()
    plt.savefig(osp.join(ASSET_IMAGES_DIR, f"triple_filtering_adjacency_{ts}.png"), dpi=300)
    plt.close()

    print(f"\n  Saved visualization to triple_filtering_adjacency_{ts}.png")


def main():
    ts = timestamp()
    print(f"Starting triple combination generation (adjacency graph) at {ts}")
    print(f"FITNESS_THRESHOLD: {FITNESS_THRESHOLD}")

    # Find the most recent gene selection file
    results_dir = "/Users/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/results/inference_preprocessing_expansion"
    gene_list_files = [f for f in os.listdir(results_dir)
                       if f.startswith("expanded_genes_inference_1_") and f.endswith(".txt")]

    if not gene_list_files:
        raise FileNotFoundError(f"No expanded gene list found in {results_dir}")

    latest_file = sorted(gene_list_files)[-1]
    gene_list_path = osp.join(results_dir, latest_file)
    print(f"Using gene list: {latest_file}")

    # Setup output directory
    inference_dir = osp.join(DATA_ROOT, "data/torchcell/experiments/006-kuzmin-tmi/inference_1")
    os.makedirs(inference_dir, exist_ok=True)

    # Load selected genes
    selected_genes = load_selected_genes(gene_list_path)

    # Build DMF pair fitness from all datasets
    print("\n" + "="*80)
    print("Loading DMF datasets...")
    print("="*80)

    dmf_datasets = {}

    # Kuzmin2018 DMF
    dmf_datasets['dmf_kuzmin2018'] = build_valid_pairs_from_dmf(
        'dmf_kuzmin2018', DmfKuzmin2018Dataset, selected_genes
    )

    # Costanzo2016 DMF
    dmf_datasets['dmf_costanzo2016'] = build_valid_pairs_from_dmf(
        'dmf_costanzo2016', DmfCostanzo2016Dataset, selected_genes
    )

    # Kuzmin2020 DMF
    dmf_datasets['dmf_kuzmin2020'] = build_valid_pairs_from_dmf(
        'dmf_kuzmin2020', DmfKuzmin2020Dataset, selected_genes
    )

    # Build valid pairs graph
    valid_pairs, adjacency, all_pair_fitness = build_valid_pairs_graph(selected_genes, dmf_datasets)

    # Generate triples using adjacency graph
    dmf_filtered_triples = generate_triples_from_adjacency(selected_genes, adjacency)

    # Load Neo4j dataset to check for existing TMI triples
    print("\n" + "="*80)
    print("Loading Neo4j dataset to check for existing TMI triples...")
    print("="*80)

    # Load query (using 009-kuzmin-tmi deletion-only query)
    with open("experiments/009-kuzmin-tmi/queries/001_small_build.cql", "r") as f:
        query = f.read()

    # Initialize genome
    genome = SCerevisiaeGenome(osp.join(DATA_ROOT, "data/sgd/genome"))
    genome.drop_chrmt()
    genome.drop_empty_go()

    # Create dataset (using 009-kuzmin-tmi deletion-only dataset)
    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/009-kuzmin-tmi/001-small-build"
    )

    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        graphs=None,
        node_embeddings=None,
        converter=None,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=GenotypeAggregator,
        graph_processor=SubgraphRepresentation(),
    )

    # Get the is_any_perturbed_gene_index
    is_any_perturbed_gene_index = dataset.is_any_perturbed_gene_index

    # Filter out existing TMI triples
    final_triples, tmi_removed_triples = filter_existing_tmi_triples(
        dmf_filtered_triples, is_any_perturbed_gene_index
    )

    # Clean up dataset
    dataset.close_lmdb()

    # Save results summary
    summary_file = osp.join(results_dir, f"triple_combinations_summary_{ts}.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Triple Combination Generation Summary (Adjacency Graph)\n")
        f.write(f"Timestamp: {ts}\n")
        f.write(f"Fitness threshold: {FITNESS_THRESHOLD}\n")
        f.write(f"Strategy: Adjacency graph + missing data = invalid\n\n")

        f.write(f"Selected genes: {len(selected_genes)}\n")
        f.write(f"Maximum possible triples C(n,3): {len(list(combinations(selected_genes, 3))):,}\n\n")

        f.write(f"DMF pair statistics:\n")
        f.write(f"  Total pairs with data: {len(all_pair_fitness):,}\n")
        f.write(f"  Valid pairs (fitness >= {FITNESS_THRESHOLD}): {len(valid_pairs):,}\n")
        f.write(f"  Invalid pairs (fitness < {FITNESS_THRESHOLD}): {len(all_pair_fitness) - len(valid_pairs):,}\n\n")

        f.write(f"Triple generation:\n")
        f.write(f"  After DMF graph filtering: {len(dmf_filtered_triples):,}\n")
        f.write(f"  After TMI filtering: {len(final_triples):,}\n")
        f.write(f"  Total reduction: {(1 - len(final_triples)/len(list(combinations(selected_genes, 3))))*100:.2f}%\n\n")

        f.write(f"DMF dataset contributions:\n")
        for dataset_name, pairs in dmf_datasets.items():
            f.write(f"  {dataset_name}: {len(pairs):,} pairs\n")

    print(f"\nSaved summary to {summary_file}")

    # Save triple list to BOTH locations
    # 1. Results directory for reference
    triple_list_file = osp.join(results_dir, f"triple_combinations_list_{ts}.txt")
    with open(triple_list_file, 'w') as f:
        for triple in final_triples:
            f.write(f"{triple[0]},{triple[1]},{triple[2]}\n")
    print(f"Saved triple list to {triple_list_file}")

    # 2. inference_1/raw directory for dataset creation
    inference_raw_dir = osp.join(inference_dir, "raw")
    os.makedirs(inference_raw_dir, exist_ok=True)
    inference_triple_file = osp.join(inference_raw_dir, f"triple_combinations_list_{ts}.txt")
    with open(inference_triple_file, 'w') as f:
        for triple in final_triples:
            f.write(f"{triple[0]},{triple[1]},{triple[2]}\n")
    print(f"Also saved triple list to inference raw directory: {inference_triple_file}")

    # Save TMI-removed triples
    tmi_removed_file = osp.join(results_dir, f"tmi_removed_triples_{ts}.txt")
    with open(tmi_removed_file, 'w') as f:
        for triple in tmi_removed_triples:
            f.write(f"{triple[0]},{triple[1]},{triple[2]}\n")
    print(f"Saved TMI-removed triples to {tmi_removed_file}")

    # Create visualizations
    create_visualizations(
        selected_genes, dmf_filtered_triples, final_triples,
        valid_pairs, all_pair_fitness, adjacency, ts
    )

    # Print final summary
    print("\n" + "="*80)
    print("TRIPLE COMBINATION SUMMARY")
    print("="*80)
    print(f"Strategy: Adjacency graph with aggressive filtering")
    print(f"Fitness threshold: {FITNESS_THRESHOLD}")
    print(f"Missing data handling: Invalid (only explicit fitness >= {FITNESS_THRESHOLD})")
    print(f"\nInput genes: {len(selected_genes)}")
    print(f"Maximum possible triples: {len(list(combinations(selected_genes, 3))):,}")

    print(f"\nPair filtering:")
    print(f"  Pairs with DMF data: {len(all_pair_fitness):,}")
    print(f"  Valid pairs (>= {FITNESS_THRESHOLD}): {len(valid_pairs):,}")
    print(f"  Reduction: {(1 - len(valid_pairs)/len(all_pair_fitness))*100:.1f}%")

    print(f"\nTriple generation:")
    print(f"  After DMF graph filtering: {len(dmf_filtered_triples):,}")
    print(f"  After TMI filtering: {len(final_triples):,}")
    print(f"  Total reduction: {(1 - len(final_triples)/len(list(combinations(selected_genes, 3))))*100:.2f}%")

    # Sample triples
    if final_triples:
        print(f"\nSample of final filtered triples:")
        for i, triple in enumerate(final_triples[:5]):
            print(f"  {i+1}. {triple[0]}, {triple[1]}, {triple[2]}")
        if len(final_triples) > 5:
            print(f"  ... and {len(final_triples)-5:,} more")
    else:
        print(f"\n⚠️  WARNING: No triples passed filtering!")
        print(f"   Consider reducing FITNESS_THRESHOLD or checking data coverage")

    # Show TMI removed samples
    if tmi_removed_triples:
        print(f"\nSample of TMI-removed triples (already exist in datasets):")
        for i, triple in enumerate(tmi_removed_triples[:5]):
            print(f"  {i+1}. {triple[0]}, {triple[1]}, {triple[2]}")
        if len(tmi_removed_triples) > 5:
            print(f"  ... and {len(tmi_removed_triples)-5:,} more")


if __name__ == "__main__":
    main()
