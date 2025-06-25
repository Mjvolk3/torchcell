#!/usr/bin/env python
"""
Generate triple gene combinations from selected genes.
Filters out triples where any pair has fitness < 0.5 in DMF datasets.
"""

import os
import os.path as osp
import pickle
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

# Load environment
load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR", "notes/assets/images")

FITNESS_THRESHOLD = 0.5  # Pairs with fitness below this are excluded


def load_selected_genes(gene_selection_file):
    """Load the selected genes from the previous analysis."""
    print(f"Loading selected genes from {gene_selection_file}")
    with open(gene_selection_file, 'rb') as f:
        data = pickle.load(f)
    
    selected_genes = data['selected_genes']
    print(f"Loaded {len(selected_genes)} selected genes")
    return selected_genes, data


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


def create_visualizations(selected_genes, triples, filtered_triples, dmf_lookups, ts):
    """Create visualizations of the filtering process."""
    
    # 1. Triple filtering summary
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie chart of kept vs rejected
    sizes = [len(filtered_triples), len(triples) - len(filtered_triples)]
    labels = ['Kept', 'Rejected']
    colors = ['#2ca02c', '#d62728']
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Triple Filtering Results')
    
    # Bar chart of low fitness pairs by dataset
    datasets = list(dmf_lookups.keys())
    pair_counts = [len(lookup) for lookup in dmf_lookups.values()]
    ax2.bar(datasets, pair_counts)
    ax2.set_ylabel(f'Number of Pairs with Fitness < {FITNESS_THRESHOLD}')
    ax2.set_title(f'Low Fitness Pairs (< {FITNESS_THRESHOLD}) by Dataset')
    ax2.tick_params(axis='x', rotation=45)
    
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
    results_dir = "/Users/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/results"
    gene_selection_files = [f for f in os.listdir(results_dir) if f.startswith("gene_selection_data_") and f.endswith(".pkl")]
    if not gene_selection_files:
        raise FileNotFoundError("No gene selection data found. Run rank_metabolic_genes.py first.")
    
    latest_file = sorted(gene_selection_files)[-1]
    gene_selection_path = osp.join(results_dir, latest_file)
    
    # Load selected genes
    selected_genes, selection_data = load_selected_genes(gene_selection_path)
    
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
    
    # Filter triples
    filtered_triples = filter_triples(triples, dmf_lookups)
    
    # Save results
    output_data = {
        'selected_genes': selected_genes,
        'all_triples': triples,
        'filtered_triples': filtered_triples,
        'dmf_lookups': dmf_lookups,
        'fitness_threshold': FITNESS_THRESHOLD,
        'timestamp': ts
    }
    
    output_file = osp.join(results_dir, f"triple_combinations_{ts}.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump(output_data, f)
    print(f"\nSaved results to {output_file}")
    
    # Save just the triple list as text
    triple_list_file = osp.join(results_dir, f"triple_combinations_list_{ts}.txt")
    with open(triple_list_file, 'w') as f:
        for triple in filtered_triples:
            f.write(f"{triple[0]},{triple[1]},{triple[2]}\n")
    print(f"Saved triple list to {triple_list_file}")
    
    # Create visualizations
    create_visualizations(selected_genes, triples, filtered_triples, dmf_lookups, ts)
    
    # Print summary statistics
    print("\n" + "="*80)
    print("TRIPLE COMBINATION SUMMARY")
    print("="*80)
    print(f"Input genes: {len(selected_genes)}")
    print(f"Total possible triples: {len(triples):,}")
    print(f"Filtered triples: {len(filtered_triples):,}")
    print(f"Reduction: {(1 - len(filtered_triples)/len(triples))*100:.2f}%")
    
    # Sample some triples
    print("\nSample of filtered triples:")
    for i, triple in enumerate(filtered_triples[:5]):
        print(f"  {i+1}. {triple[0]}, {triple[1]}, {triple[2]}")
    if len(filtered_triples) > 5:
        print(f"  ... and {len(filtered_triples)-5:,} more")


if __name__ == "__main__":
    main()