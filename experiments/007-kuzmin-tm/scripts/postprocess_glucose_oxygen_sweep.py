#!/usr/bin/env python3
"""
Post-processing script for glucose/oxygen sensitivity analysis.
Loads experimental data from Neo4j and matches to FBA predictions for all sweep conditions.
Saves matched results as parquet files.
"""

import os
import os.path as osp
import sys
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm

# Add torchcell to path
sys.path.append('/home/michaelvolk/Documents/projects/torchcell')
from dotenv import load_dotenv

# For experimental data
from torchcell.data import Neo4jCellDataset, MeanExperimentDeduplicator, GenotypeAggregator
from torchcell.data.graph_processor import SubgraphRepresentation
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.metabolism.yeast_GEM import YeastGEM


def load_experimental_data():
    """Load experimental data from Neo4j dataset."""
    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")
    EXPERIMENT_ROOT = osp.join(DATA_ROOT, "data/torchcell/experiments/007-kuzmin-tm")

    dataset_root = osp.join(EXPERIMENT_ROOT, "001-small-build")

    # Load query
    with open("experiments/007-kuzmin-tm/queries/001_small_build.cql", "r") as f:
        query = f.read()

    # Load genome
    genome = SCerevisiaeGenome(osp.join(DATA_ROOT, "data/sgd/genome"))
    genome.drop_chrmt()
    genome.drop_empty_go()

    # Load dataset
    print("Loading dataset...")
    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        graphs=None,
        incidence_graphs={"metabolism_bipartite": YeastGEM().bipartite_graph},
        node_embeddings=None,
        converter=None,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=GenotypeAggregator,
        graph_processor=SubgraphRepresentation(),
    )

    # Extract genotypes and experimental values
    print("Extracting genotypes and experimental data...")
    genotypes = []
    fitness_values = []
    interaction_values = []

    for i in tqdm(range(len(dataset)), desc="Processing"):
        data = dataset[i]
        gene_ids = sorted(data['gene'].ids_pert)
        genotype = '+'.join(gene_ids)
        genotypes.append(genotype)

        # Get fitness and interaction values from label_df
        fitness_values.append(dataset.label_df.iloc[i]['fitness'])
        interaction_values.append(dataset.label_df.iloc[i]['gene_interaction'])

    # Create experimental dataframe
    experimental_df = pd.DataFrame({
        'genotype': genotypes,
        'exp_fitness': fitness_values,
        'exp_interaction': interaction_values
    })

    return experimental_df


def get_double_fitness(doubles_fitness, gene1, gene2):
    """Get double mutant fitness, trying both possible orderings."""
    key1 = f'{gene1}+{gene2}'
    key2 = f'{gene2}+{gene1}'

    if key1 in doubles_fitness:
        return doubles_fitness[key1]
    elif key2 in doubles_fitness:
        return doubles_fitness[key2]
    else:
        return np.nan


def calculate_fba_triple_interactions(singles_df, doubles_df, triples_df):
    """Calculate FBA triple interactions using the full formula.

    τ_ijk = f_ijk - f_i*f_j*f_k - ε_ij*f_k - ε_ik*f_j - ε_jk*f_i
    where ε_ij = f_ij - f_i*f_j

    Args:
        singles_df: DataFrame with columns ['genotype', 'fitness']
        doubles_df: DataFrame with columns ['genotype', 'fitness']
        triples_df: DataFrame with columns ['genotype', 'fitness']

    Returns:
        List of FBA triple interactions matching triples_df order
    """
    # Create lookup dictionaries for fast access
    singles_fitness = dict(zip(singles_df['genotype'], singles_df['fitness']))
    doubles_fitness = dict(zip(doubles_df['genotype'], doubles_df['fitness']))

    fba_triple_interactions = []

    for _, row in triples_df.iterrows():
        genotype = row['genotype']
        genes = genotype.split('+')

        if len(genes) != 3:
            fba_triple_interactions.append(np.nan)
            continue

        # Get triple mutant fitness
        f_ijk = row['fitness']

        # Get single mutant fitnesses
        f_i = singles_fitness.get(genes[0], np.nan)
        f_j = singles_fitness.get(genes[1], np.nan)
        f_k = singles_fitness.get(genes[2], np.nan)

        # Get double mutant fitnesses (try both orderings)
        f_ij = get_double_fitness(doubles_fitness, genes[0], genes[1])
        f_ik = get_double_fitness(doubles_fitness, genes[0], genes[2])
        f_jk = get_double_fitness(doubles_fitness, genes[1], genes[2])

        # Check if we have all required values
        if np.isnan(f_i) or np.isnan(f_j) or np.isnan(f_k):
            fba_triple_interactions.append(np.nan)
            continue

        if np.isnan(f_ij) or np.isnan(f_ik) or np.isnan(f_jk):
            fba_triple_interactions.append(np.nan)
            continue

        # Calculate digenic interactions
        epsilon_ij = f_ij - f_i * f_j
        epsilon_ik = f_ik - f_i * f_k
        epsilon_jk = f_jk - f_j * f_k

        # Calculate trigenic interaction
        tau_ijk = f_ijk - f_i*f_j*f_k - epsilon_ij*f_k - epsilon_ik*f_j - epsilon_jk*f_i

        fba_triple_interactions.append(tau_ijk)

    return fba_triple_interactions


def match_condition_results(results_dir, media, glucose, oxygen, experimental_df):
    """Match FBA results to experimental data for a specific condition."""
    # Load singles
    singles_file = f'{results_dir}/singles_deletions_{media}_glc{glucose}_o2{oxygen}.parquet'
    if not osp.exists(singles_file):
        return None

    singles = pd.read_parquet(singles_file)
    singles['perturbation_type'] = 'single'
    singles['genotype'] = singles['gene']

    # Load doubles
    doubles_file = f'{results_dir}/doubles_deletions_{media}_glc{glucose}_o2{oxygen}.parquet'
    if osp.exists(doubles_file):
        doubles = pd.read_parquet(doubles_file)
        doubles['perturbation_type'] = 'double'
        doubles['genotype'] = doubles['gene1'] + '+' + doubles['gene2']
    else:
        doubles = pd.DataFrame()

    # Load triples
    triples_file = f'{results_dir}/triples_deletions_{media}_glc{glucose}_o2{oxygen}.parquet'
    if osp.exists(triples_file):
        triples = pd.read_parquet(triples_file)
        triples['perturbation_type'] = 'triple'
        triples['genotype'] = triples['gene1'] + '+' + triples['gene2'] + '+' + triples['gene3']
    else:
        triples = pd.DataFrame()

    # Calculate FBA triple interactions
    if len(triples) > 0 and len(singles) > 0 and len(doubles) > 0:
        fba_triple_interactions = calculate_fba_triple_interactions(singles, doubles, triples)
        triples['fba_triple_interaction'] = fba_triple_interactions

    # Combine all FBA results
    fba_results = pd.concat([singles, doubles, triples], ignore_index=True)

    # Merge with experimental data
    matched = pd.merge(
        fba_results,
        experimental_df,
        on='genotype',
        how='inner'
    )

    # Rename columns for clarity
    matched['fba_fitness'] = matched['fitness']
    matched['fba_growth'] = matched['growth']
    matched['experimental_fitness'] = matched['exp_fitness']
    matched['experimental_interaction'] = matched['exp_interaction']

    # Add condition metadata
    matched['media'] = media
    matched['glucose'] = glucose
    matched['oxygen'] = oxygen

    return matched


def main():
    """Main execution."""

    BASE_DIR = "experiments/007-kuzmin-tm"
    RESULTS_DIR = f"{BASE_DIR}/results/cobra-fba-growth"

    print("="*70)
    print("Glucose/O2 Sensitivity Analysis - Post-Processing")
    print("="*70)
    print()

    # Load experimental data once
    experimental_df = load_experimental_data()
    print(f"Loaded {len(experimental_df)} experimental measurements")

    # Find all condition files
    pattern = f'{RESULTS_DIR}/singles_deletions_*_glc*_o2*.parquet'
    singles_files = glob.glob(pattern)

    print(f"\nFound {len(singles_files)} conditions to process")

    # Extract condition info from filenames
    conditions = []
    for file in singles_files:
        # Parse filename: singles_deletions_minimal_glc10_o2100.parquet
        basename = osp.basename(file)
        parts = basename.replace('singles_deletions_', '').replace('.parquet', '').split('_')
        media = parts[0]
        glucose = float(parts[1].replace('glc', ''))
        oxygen = float(parts[2].replace('o2', ''))
        conditions.append((media, glucose, oxygen))

    # Match each condition
    print("\n=== Matching FBA Results to Experimental Data ===")
    matched_count = 0

    for media, glucose, oxygen in tqdm(conditions, desc="Matching conditions"):
        matched = match_condition_results(RESULTS_DIR, media, glucose, oxygen, experimental_df)

        if matched is not None and len(matched) > 0:
            # Save matched results
            output_file = f'{RESULTS_DIR}/matched_{media}_glc{glucose}_o2{oxygen}.parquet'
            matched.to_parquet(output_file)
            matched_count += 1

    print(f"\nMatched {matched_count}/{len(conditions)} conditions")
    print("\n" + "="*70)
    print("Post-processing complete!")
    print("="*70)


if __name__ == "__main__":
    main()
