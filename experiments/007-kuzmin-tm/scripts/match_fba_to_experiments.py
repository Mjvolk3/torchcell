"""
Match FBA predictions to experimental data using label_df directly.
This version uses the dataset.label_df which has the correct fitness values.
"""

import os
import os.path as osp
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from scipy import stats
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from datetime import datetime
from tqdm import tqdm

# Neo4j dataset imports
from torchcell.data import Neo4jCellDataset, MeanExperimentDeduplicator, GenotypeAggregator
from torchcell.data.graph_processor import SubgraphRepresentation
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.metabolism.yeast_GEM import YeastGEM


def load_experimental_data_from_label_df(dataset_root: str, experiment_root: str) -> pd.DataFrame:
    """Load experimental data directly from dataset.label_df."""
    
    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")
    
    # Load query
    with open("experiments/007-kuzmin-tm/queries/001_small_build.cql", "r") as f:
        query = f.read()
    
    # Initialize genome
    genome = SCerevisiaeGenome(osp.join(DATA_ROOT, "data/sgd/genome"))
    genome.drop_chrmt()
    genome.drop_empty_go()
    
    # Create dataset
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
    
    print(f"Loaded dataset with {len(dataset)} data points")
    
    # Get gene perturbations for each item
    print("Extracting gene perturbations...")
    perturbations = []
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        gene_ids = sorted(data['gene'].ids_pert)
        perturbations.append(gene_ids)
    
    # Create experimental dataframe
    experimental_data = []
    
    # Add fitness measurements
    for i, genes in enumerate(perturbations):
        fitness_value = dataset.label_df.iloc[i]['fitness']
        gene_interaction_value = dataset.label_df.iloc[i]['gene_interaction']
        
        # Create fitness record
        record_fitness = {
            'perturbation': tuple(genes),
            'n_genes': len(genes),
            'phenotype_type': 'fitness',
            'experimental_value': fitness_value,
        }
        
        # Create gene interaction record
        record_gi = {
            'perturbation': tuple(genes),
            'n_genes': len(genes),
            'phenotype_type': 'gene_interaction',
            'experimental_value': gene_interaction_value,
        }
        
        # Add gene columns for matching
        if len(genes) == 1:
            record_fitness['gene'] = genes[0]
            record_gi['gene'] = genes[0]
        elif len(genes) == 2:
            record_fitness['gene1'] = genes[0]
            record_fitness['gene2'] = genes[1]
            record_gi['gene1'] = genes[0]
            record_gi['gene2'] = genes[1]
        elif len(genes) == 3:
            record_fitness['gene1'] = genes[0]
            record_fitness['gene2'] = genes[1]
            record_fitness['gene3'] = genes[2]
            record_gi['gene1'] = genes[0]
            record_gi['gene2'] = genes[1]
            record_gi['gene3'] = genes[2]
        
        experimental_data.append(record_fitness)
        experimental_data.append(record_gi)
    
    exp_df = pd.DataFrame(experimental_data)
    print(f"Extracted {len(exp_df)} experimental measurements")
    
    # Verify the data
    fitness_exp = exp_df[exp_df['phenotype_type'] == 'fitness']
    gi_exp = exp_df[exp_df['phenotype_type'] == 'gene_interaction']
    
    print(f"\nFitness measurements: {len(fitness_exp)}")
    print(f"  Range: {fitness_exp['experimental_value'].min():.4f} to {fitness_exp['experimental_value'].max():.4f}")
    print(f"  Negative values: {(fitness_exp['experimental_value'] < 0).sum()}")
    
    print(f"\nGene interaction measurements: {len(gi_exp)}")
    print(f"  Range: {gi_exp['experimental_value'].min():.4f} to {gi_exp['experimental_value'].max():.4f}")
    
    return exp_df


def match_fba_to_experiments(
    exp_df: pd.DataFrame,
    single_df: pd.DataFrame,
    double_df: pd.DataFrame,
    triple_df: pd.DataFrame,
    digenic_df: pd.DataFrame,
    trigenic_df: pd.DataFrame,
) -> pd.DataFrame:
    """Match FBA predictions to experimental measurements using efficient merges."""
    
    matched_dfs = []
    
    # Process fitness measurements
    fitness_exp = exp_df[exp_df['phenotype_type'] == 'fitness'].copy()
    
    # Match singles using merge
    singles_fitness = fitness_exp[fitness_exp['n_genes'] == 1].copy()
    if len(singles_fitness) > 0:
        print(f"  Matching {len(singles_fitness)} single deletions...")
        singles_merged = singles_fitness.merge(
            single_df[['gene', 'fitness']],
            on='gene',
            how='inner'
        )
        singles_merged['perturbation_type'] = 'single'
        singles_merged['genes'] = singles_merged['gene']
        singles_merged['fba_predicted'] = singles_merged['fitness']
        singles_merged['experimental'] = singles_merged['experimental_value']
        matched_dfs.append(singles_merged[['perturbation_type', 'genes', 'phenotype_type', 'experimental', 'fba_predicted']])
    
    # Match doubles using merge
    doubles_fitness = fitness_exp[fitness_exp['n_genes'] == 2].copy()
    if len(doubles_fitness) > 0:
        print(f"  Matching {len(doubles_fitness)} double deletions...")
        doubles_merged = doubles_fitness.merge(
            double_df[['gene1', 'gene2', 'fitness']],
            on=['gene1', 'gene2'],
            how='inner'
        )
        doubles_merged['perturbation_type'] = 'double'
        doubles_merged['genes'] = doubles_merged['gene1'] + ',' + doubles_merged['gene2']
        doubles_merged['fba_predicted'] = doubles_merged['fitness']
        doubles_merged['experimental'] = doubles_merged['experimental_value']
        matched_dfs.append(doubles_merged[['perturbation_type', 'genes', 'phenotype_type', 'experimental', 'fba_predicted']])
    
    # Match triples using merge
    triples_fitness = fitness_exp[fitness_exp['n_genes'] == 3].copy()
    if len(triples_fitness) > 0:
        print(f"  Matching {len(triples_fitness)} triple deletions...")
        triples_merged = triples_fitness.merge(
            triple_df[['gene1', 'gene2', 'gene3', 'fitness']],
            on=['gene1', 'gene2', 'gene3'],
            how='inner'
        )
        triples_merged['perturbation_type'] = 'triple'
        triples_merged['genes'] = triples_merged['gene1'] + ',' + triples_merged['gene2'] + ',' + triples_merged['gene3']
        triples_merged['fba_predicted'] = triples_merged['fitness']
        triples_merged['experimental'] = triples_merged['experimental_value']
        matched_dfs.append(triples_merged[['perturbation_type', 'genes', 'phenotype_type', 'experimental', 'fba_predicted']])
    
    # Process gene interaction measurements
    gi_exp = exp_df[exp_df['phenotype_type'] == 'gene_interaction'].copy()
    
    # Match digenic interactions using merge
    doubles_gi = gi_exp[gi_exp['n_genes'] == 2].copy()
    if len(doubles_gi) > 0:
        print(f"  Matching {len(doubles_gi)} digenic interactions...")
        digenic_merged = doubles_gi.merge(
            digenic_df[['gene1', 'gene2', 'epsilon']],
            on=['gene1', 'gene2'],
            how='inner'
        )
        digenic_merged['perturbation_type'] = 'double'
        digenic_merged['genes'] = digenic_merged['gene1'] + ',' + digenic_merged['gene2']
        digenic_merged['fba_predicted'] = digenic_merged['epsilon']
        digenic_merged['experimental'] = digenic_merged['experimental_value']
        matched_dfs.append(digenic_merged[['perturbation_type', 'genes', 'phenotype_type', 'experimental', 'fba_predicted']])
    
    # Match trigenic interactions using merge
    triples_gi = gi_exp[gi_exp['n_genes'] == 3].copy()
    if len(triples_gi) > 0:
        print(f"  Matching {len(triples_gi)} trigenic interactions...")
        trigenic_merged = triples_gi.merge(
            trigenic_df[['gene1', 'gene2', 'gene3', 'tau']],
            on=['gene1', 'gene2', 'gene3'],
            how='inner'
        )
        trigenic_merged['perturbation_type'] = 'triple'
        trigenic_merged['genes'] = trigenic_merged['gene1'] + ',' + trigenic_merged['gene2'] + ',' + trigenic_merged['gene3']
        trigenic_merged['fba_predicted'] = trigenic_merged['tau']
        trigenic_merged['experimental'] = trigenic_merged['experimental_value']
        matched_dfs.append(trigenic_merged[['perturbation_type', 'genes', 'phenotype_type', 'experimental', 'fba_predicted']])
    
    # Combine all matched data
    if matched_dfs:
        matched_df = pd.concat(matched_dfs, ignore_index=True)
    else:
        matched_df = pd.DataFrame(columns=['perturbation_type', 'genes', 'phenotype_type', 'experimental', 'fba_predicted'])
    
    print(f"\nMatched {len(matched_df)} measurements to FBA predictions")
    
    return matched_df


def main():
    """Main execution function."""
    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")
    EXPERIMENT_ROOT = osp.join(DATA_ROOT, "data/torchcell/experiments/007-kuzmin-tm")
    
    # Set up paths
    dataset_root = osp.join(EXPERIMENT_ROOT, "001-small-build")
    results_dir = "experiments/007-kuzmin-tm/results/cobra-fba-growth"
    
    print("=== Loading FBA Results ===")
    
    # Check if FBA results exist
    required_files = [
        "singles_deletions.parquet",
        "doubles_deletions.parquet", 
        "triples_deletions.parquet",
        "digenic_interactions.parquet",
        "trigenic_interactions.parquet"
    ]
    
    for file in required_files:
        if not osp.exists(osp.join(results_dir, file)):
            print(f"Error: Missing {file}. Please run targeted_fba_growth.py first.")
            return
    
    # Load FBA results
    single_df = pd.read_parquet(osp.join(results_dir, "singles_deletions.parquet"))
    double_df = pd.read_parquet(osp.join(results_dir, "doubles_deletions.parquet"))
    triple_df = pd.read_parquet(osp.join(results_dir, "triples_deletions.parquet"))
    digenic_df = pd.read_parquet(osp.join(results_dir, "digenic_interactions.parquet"))
    trigenic_df = pd.read_parquet(osp.join(results_dir, "trigenic_interactions.parquet"))
    
    print(f"Loaded FBA results:")
    print(f"  Singles: {len(single_df)}")
    print(f"  Doubles: {len(double_df)}")
    print(f"  Triples: {len(triple_df)}")
    print(f"  Digenic interactions: {len(digenic_df)}")
    print(f"  Trigenic interactions: {len(trigenic_df)}")
    
    print("\n=== Loading Experimental Data from label_df ===")
    exp_df = load_experimental_data_from_label_df(dataset_root, EXPERIMENT_ROOT)
    
    print("\n=== Matching FBA to Experimental Data ===")
    matched_df = match_fba_to_experiments(
        exp_df, single_df, double_df, triple_df, digenic_df, trigenic_df
    )
    
    # Save matched data
    matched_df.to_csv(osp.join(results_dir, "matched_fba_experimental_fixed.csv"), index=False)
    matched_df.to_parquet(osp.join(results_dir, "matched_fba_experimental_fixed.parquet"), index=False)
    
    # Print summary
    print("\n=== Summary ===")
    for phenotype in ['fitness', 'gene_interaction']:
        subset = matched_df[matched_df['phenotype_type'] == phenotype]
        if len(subset) > 0:
            print(f"\n{phenotype}:")
            print(f"  Count: {len(subset)}")
            print(f"  Experimental range: {subset['experimental'].min():.4f} to {subset['experimental'].max():.4f}")
            print(f"  FBA predicted range: {subset['fba_predicted'].min():.4f} to {subset['fba_predicted'].max():.4f}")
            if phenotype == 'fitness':
                neg_count = (subset['experimental'] < 0).sum()
                print(f"  Negative experimental values: {neg_count}")
                if neg_count > 0:
                    print("  ⚠️ WARNING: Found negative fitness values!")
    
    print(f"\nResults saved to {results_dir}")
    print("Files created:")
    print("  - matched_fba_experimental_fixed.csv")
    print("  - matched_fba_experimental_fixed.parquet")


if __name__ == "__main__":
    main()