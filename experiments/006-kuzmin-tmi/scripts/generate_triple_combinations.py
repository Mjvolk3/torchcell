# experiments/006-kuzmin-tmi/scripts/generate_triple_combinations
# [[experiments.006-kuzmin-tmi.scripts.generate_triple_combinations]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/006-kuzmin-tmi/scripts/generate_triple_combinations
# Test file: experiments/006-kuzmin-tmi/scripts/test_generate_triple_combinations.py

"""
Generate triple gene combinations from selected genes.
Filters out triples where any pair has fitness < 0.5 in DMF datasets.
Also filters out triples that already exist in TMI datasets.
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

FITNESS_THRESHOLD = 0.5  # Pairs with fitness below this are excluded


def load_selected_genes(gene_list_file):
    """Load the selected genes from the text file."""
    print(f"Loading selected genes from {gene_list_file}")
    with open(gene_list_file, 'r') as f:
        selected_genes = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(selected_genes)} selected genes")
    return selected_genes


def build_dmf_lookup_dataframe(dataset_name, dataset_class, selected_genes):
    """Build lookup of gene pairs with fitness < threshold to exclude from triples."""
    print(f"\nBuilding lookup for {dataset_name}...")
    
    dataset = dataset_class(root=osp.join(DATA_ROOT, f"data/torchcell/{dataset_name}"))
    
    # Access the preprocessed dataframe if available
    preprocess_path = osp.join(dataset.preprocess_dir, "data.csv")
    if osp.exists(preprocess_path):
        print(f"Loading preprocessed dataframe from {preprocess_path}")
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
                print(f"Warning: No fitness column found in {dataset_name}")
                return {}
        else:
            print(f"Warning: Unknown dataframe format for {dataset_name}")
            return {}
        
        # Filter to only include pairs where both genes are in selected set
        df_filtered = df[(df['gene1'].isin(selected_genes)) & (df['gene2'].isin(selected_genes))]
        print(f"Filtered from {len(df)} to {len(df_filtered)} pairs with selected genes")
        
        # Build lookup dictionary
        pair_fitness = {}
        low_fitness_count = 0
        
        for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Building lookup"):
            gene1, gene2 = row['gene1'], row['gene2']
            fitness = row[fitness_col]
            
            # Store both orderings for easy lookup
            pair = tuple(sorted([gene1, gene2]))
            if fitness < FITNESS_THRESHOLD:
                pair_fitness[pair] = fitness
                low_fitness_count += 1
        
        print(f"Found {low_fitness_count} gene pairs with fitness < {FITNESS_THRESHOLD}")
        return pair_fitness
    
    else:
        print(f"No preprocessed dataframe found, using dataset directly (slower)")
        return build_dmf_lookup_direct(dataset, selected_genes)


def build_dmf_lookup_direct(dataset, selected_genes):
    """Build lookup by iterating through dataset directly (fallback method)."""
    pair_fitness = {}
    low_fitness_count = 0
    
    for i in tqdm(range(len(dataset)), desc="Processing dataset"):
        data = dataset[i]
        experiment = data['experiment']
        fitness = experiment['phenotype']['fitness']
        
        if fitness < FITNESS_THRESHOLD:
            perturbations = experiment['genotype']['perturbations']
            if len(perturbations) == 2:
                genes = [p['systematic_gene_name'] for p in perturbations]
                if all(g in selected_genes for g in genes):
                    pair = tuple(sorted(genes))
                    pair_fitness[pair] = fitness
                    low_fitness_count += 1
    
    print(f"Found {low_fitness_count} gene pairs with fitness < {FITNESS_THRESHOLD}")
    return pair_fitness


def generate_triples(selected_genes):
    """Generate all possible triple combinations."""
    print(f"\nGenerating triple combinations from {len(selected_genes)} genes...")
    triples = list(combinations(selected_genes, 3))
    print(f"Generated {len(triples):,} triple combinations")
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
    print("\nFiltering triples to remove those that exist in TMI datasets...")
    
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
    print(f"  - Kept {len(filtered_triples):,} triples (not in TMI datasets)")
    print(f"  - Rejected {rejected_count:,} triples (already exist in TMI datasets)")
    print(f"  - Rejection rate: {rejected_count/len(triples)*100:.2f}%")
    
    return filtered_triples, tmi_removed_triples


def filter_triples(triples, dmf_lookups):
    """Filter out triples where any pair has fitness < threshold."""
    print("\nFiltering triples to remove those with low fitness pairs...")
    
    # Combine all DMF lookups - these are pairs to EXCLUDE
    all_low_fitness_pairs = set()
    for dataset_name, lookup in dmf_lookups.items():
        all_low_fitness_pairs.update(lookup.keys())
    
    print(f"Total unique low fitness pairs to exclude: {len(all_low_fitness_pairs)}")
    
    filtered_triples = []
    rejected_count = 0
    rejection_reasons = defaultdict(int)
    
    for triple in tqdm(triples, desc="Filtering triples"):
        # Check all three pairs in the triple
        pairs = [
            tuple(sorted([triple[0], triple[1]])),
            tuple(sorted([triple[0], triple[2]])),
            tuple(sorted([triple[1], triple[2]]))
        ]
        
        # Check if any pair has low fitness (< 0.5) - if so, reject the triple
        reject = False
        for pair in pairs:
            if pair in all_low_fitness_pairs:
                reject = True
                # Track which dataset caused rejection
                for dataset_name, lookup in dmf_lookups.items():
                    if pair in lookup:
                        rejection_reasons[dataset_name] += 1
                break
        
        if not reject:
            # Keep this triple - all pairs have fitness >= 0.5 (or no data)
            filtered_triples.append(triple)
        else:
            rejected_count += 1
    
    print(f"\nFiltering complete:")
    print(f"  - Kept {len(filtered_triples):,} triples (all pairs have fitness >= {FITNESS_THRESHOLD} or no data)")
    print(f"  - Rejected {rejected_count:,} triples (contain at least one pair with fitness < {FITNESS_THRESHOLD})")
    print(f"  - Rejection rate: {rejected_count/len(triples)*100:.2f}%")
    
    print("\nRejection reasons by dataset:")
    for dataset, count in rejection_reasons.items():
        print(f"  - {dataset}: {count:,} rejections")
    
    return filtered_triples


def create_visualizations(selected_genes, triples, dmf_filtered_triples, final_filtered_triples, dmf_lookups, ts):
    """Create visualizations of the filtering process."""
    
    # 1. Triple filtering summary - now with two stages
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
    
    # Overall filtering flow
    filtering_stages = [
        ('Total\nGenerated', len(triples)),
        ('After DMF\nFilter', len(dmf_filtered_triples)),
        ('After TMI\nFilter', len(final_filtered_triples))
    ]
    stages, counts = zip(*filtering_stages)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    bars = ax1.bar(stages, counts, color=colors)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}',
                ha='center', va='bottom')
    
    ax1.set_ylabel('Number of Triples')
    ax1.set_title('Triple Filtering Pipeline')
    
    # Pie chart of final results
    sizes = [len(final_filtered_triples), len(triples) - len(final_filtered_triples)]
    labels = ['Kept', 'Rejected']
    colors = ['#2ca02c', '#d62728']
    ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Final Triple Filtering Results')
    
    # Bar chart of low fitness pairs by dataset
    datasets = list(dmf_lookups.keys())
    pair_counts = [len(lookup) for lookup in dmf_lookups.values()]
    ax3.bar(datasets, pair_counts)
    ax3.set_ylabel(f'Number of Pairs with Fitness < {FITNESS_THRESHOLD}')
    ax3.set_title(f'Low Fitness Pairs (< {FITNESS_THRESHOLD}) by Dataset')
    ax3.tick_params(axis='x', rotation=45)
    
    # Rejection breakdown
    dmf_rejected = len(triples) - len(dmf_filtered_triples)
    tmi_rejected = len(dmf_filtered_triples) - len(final_filtered_triples)
    rejection_data = [dmf_rejected, tmi_rejected]
    rejection_labels = ['DMF Rejection', 'TMI Rejection']
    ax4.bar(rejection_labels, rejection_data, color=['#ff7f0e', '#d62728'])
    ax4.set_ylabel('Number of Triples Rejected')
    ax4.set_title('Rejection Breakdown by Filter Type')
    
    # Add percentage labels
    for i, (label, count) in enumerate(zip(rejection_labels, rejection_data)):
        if len(triples) > 0:
            percentage = count / len(triples) * 100
            ax4.text(i, count, f'{count:,}\n({percentage:.1f}%)', 
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(osp.join(ASSET_IMAGES_DIR, f"triple_filtering_summary_{ts}.png"), dpi=300)
    plt.close()
    
    # 2. Distribution of pairs per gene
    gene_pair_counts = defaultdict(int)
    for dmf_lookup in dmf_lookups.values():
        for pair in dmf_lookup:
            gene_pair_counts[pair[0]] += 1
            gene_pair_counts[pair[1]] += 1
    
    plt.figure(figsize=(10, 6))
    counts = list(gene_pair_counts.values())
    plt.hist(counts, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Number of Low Fitness Pairs per Gene')
    plt.ylabel('Number of Genes')
    plt.title('Distribution of Low Fitness Pairs per Gene')
    plt.axvline(np.mean(counts), color='red', linestyle='--', label=f'Mean: {np.mean(counts):.1f}')
    plt.axvline(np.median(counts), color='green', linestyle='--', label=f'Median: {np.median(counts):.1f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(osp.join(ASSET_IMAGES_DIR, f"low_fitness_pairs_distribution_{ts}.png"), dpi=300)
    plt.close()


def main():
    ts = timestamp()
    print(f"Starting triple combination generation at {ts}")
    
    # Find the most recent gene selection file
    results_dir = "/Users/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/results/inference_preprocessing"
    gene_list_files = [f for f in os.listdir(results_dir) if f.startswith("selected_genes_list_") and f.endswith(".txt")]
    if not gene_list_files:
        raise FileNotFoundError("No gene selection data found. Run rank_metabolic_genes.py first.")
    
    latest_file = sorted(gene_list_files)[-1]
    gene_list_path = osp.join(results_dir, latest_file)
    
    # Setup inference directory
    inference_dir = osp.join(DATA_ROOT, "data/torchcell/experiments/006-kuzmin-tmi/inference_0")
    os.makedirs(inference_dir, exist_ok=True)
    
    # Load selected genes
    selected_genes = load_selected_genes(gene_list_path)
    
    # Build DMF lookups
    dmf_lookups = {}
    
    # Kuzmin2018 DMF
    dmf_lookups['dmf_kuzmin2018'] = build_dmf_lookup_dataframe(
        'dmf_kuzmin2018', DmfKuzmin2018Dataset, selected_genes
    )
    
    # Costanzo2016 DMF
    dmf_lookups['dmf_costanzo2016'] = build_dmf_lookup_dataframe(
        'dmf_costanzo2016', DmfCostanzo2016Dataset, selected_genes
    )
    
    # Kuzmin2020 DMF
    dmf_lookups['dmf_kuzmin2020'] = build_dmf_lookup_dataframe(
        'dmf_kuzmin2020', DmfKuzmin2020Dataset, selected_genes
    )
    
    # Generate triples
    triples = generate_triples(selected_genes)
    
    # Filter triples based on DMF data
    dmf_filtered_triples = filter_triples(triples, dmf_lookups)
    
    # Load Neo4j dataset to check for existing TMI triples
    print("\nLoading Neo4j dataset to check for existing TMI triples...")
    
    # Load query
    with open("experiments/006-kuzmin-tmi/queries/001_small_build.cql", "r") as f:
        query = f.read()
    
    # Initialize genome
    genome = SCerevisiaeGenome(osp.join(DATA_ROOT, "data/sgd/genome"))
    genome.drop_chrmt()
    genome.drop_empty_go()
    
    # Create dataset
    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/006-kuzmin-tmi/001-small-build"
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
    final_filtered_triples, tmi_removed_triples = filter_existing_tmi_triples(dmf_filtered_triples, is_any_perturbed_gene_index)
    
    # Clean up dataset
    dataset.close_lmdb()
    
    # Save results information to a summary file
    summary_file = osp.join(results_dir, f"triple_combinations_summary_{ts}.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Triple Combination Generation Summary\n")
        f.write(f"Timestamp: {ts}\n")
        f.write(f"Fitness threshold: {FITNESS_THRESHOLD}\n")
        f.write(f"Selected genes: {len(selected_genes)}\n")
        f.write(f"Total triples generated: {len(triples)}\n")
        f.write(f"After DMF filtering: {len(dmf_filtered_triples)}\n")
        f.write(f"After TMI filtering: {len(final_filtered_triples)}\n")
        f.write(f"\nDMF lookups summary:\n")
        for dataset_name, lookup in dmf_lookups.items():
            f.write(f"  {dataset_name}: {len(lookup)} low fitness pairs\n")
    print(f"\nSaved summary to {summary_file}")
    
    # Save just the triple list as text in BOTH locations
    # Save in results directory for reference
    triple_list_file = osp.join(results_dir, f"triple_combinations_list_{ts}.txt")
    with open(triple_list_file, 'w') as f:
        for triple in final_filtered_triples:
            f.write(f"{triple[0]},{triple[1]},{triple[2]}\n")
    print(f"Saved triple list to {triple_list_file}")
    
    # Also save to inference_0/raw directory for easy transfer to supercomputer
    inference_raw_dir = osp.join(DATA_ROOT, "data/torchcell/experiments/006-kuzmin-tmi/inference_0/raw")
    os.makedirs(inference_raw_dir, exist_ok=True)
    inference_triple_file = osp.join(inference_raw_dir, f"triple_combinations_list_{ts}.txt")
    with open(inference_triple_file, 'w') as f:
        for triple in final_filtered_triples:
            f.write(f"{triple[0]},{triple[1]},{triple[2]}\n")
    print(f"Also saved triple list to inference raw directory: {inference_triple_file}")
    
    # Save TMI-removed triples as text
    tmi_removed_file = osp.join(results_dir, f"tmi_removed_triples_{ts}.txt")
    with open(tmi_removed_file, 'w') as f:
        for triple in tmi_removed_triples:
            f.write(f"{triple[0]},{triple[1]},{triple[2]}\n")
    print(f"Saved TMI-removed triples to {tmi_removed_file}")
    
    # Create visualizations
    create_visualizations(selected_genes, triples, dmf_filtered_triples, final_filtered_triples, dmf_lookups, ts)
    
    # Print summary statistics
    print("\n" + "="*80)
    print("TRIPLE COMBINATION SUMMARY")
    print("="*80)
    print(f"Input genes: {len(selected_genes)}")
    print(f"Total possible triples: {len(triples):,}")
    print(f"After DMF filtering: {len(dmf_filtered_triples):,}")
    print(f"After TMI filtering: {len(final_filtered_triples):,}")
    print(f"Total reduction: {(1 - len(final_filtered_triples)/len(triples))*100:.2f}%")
    
    # Breakdown of filtering
    print("\nFiltering breakdown:")
    dmf_removed = len(triples) - len(dmf_filtered_triples)
    tmi_removed = len(dmf_filtered_triples) - len(final_filtered_triples)
    print(f"  - DMF filtering removed: {dmf_removed:,} ({dmf_removed/len(triples)*100:.2f}%)")
    print(f"  - TMI filtering removed: {tmi_removed:,} ({tmi_removed/len(triples)*100:.2f}%)")
    
    # Sample some triples
    print("\nSample of final filtered triples:")
    for i, triple in enumerate(final_filtered_triples[:5]):
        print(f"  {i+1}. {triple[0]}, {triple[1]}, {triple[2]}")
    if len(final_filtered_triples) > 5:
        print(f"  ... and {len(final_filtered_triples)-5:,} more")
    
    # Show sample of TMI-removed triples
    if tmi_removed_triples:
        print(f"\nSample of TMI-removed triples (already exist in datasets):")
        for i, triple in enumerate(tmi_removed_triples[:5]):
            print(f"  {i+1}. {triple[0]}, {triple[1]}, {triple[2]}")
        if len(tmi_removed_triples) > 5:
            print(f"  ... and {len(tmi_removed_triples)-5:,} more")


if __name__ == "__main__":
    main()