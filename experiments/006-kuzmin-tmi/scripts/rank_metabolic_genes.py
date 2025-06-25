#!/usr/bin/env python
"""
Rank genes for metabolic importance targeting ~220 genes.
Combines metabolic genes from YeastGEM with kinase activity genes,
filters out essential and synthetic lethal genes.
"""

import os
import os.path as osp
from collections import defaultdict
import pandas as pd
from torchcell.metabolism.yeast_GEM import YeastGEM
from torchcell.graph import SCerevisiaeGraph
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.datasets.scerevisiae.sgd import GeneEssentialitySgdDataset
from torchcell.datasets.scerevisiae.kuzmin2018 import SmfKuzmin2018Dataset
from torchcell.datasets.scerevisiae.costanzo2016 import SmfCostanzo2016Dataset
from torchcell.datasets.scerevisiae.kuzmin2020 import SmfKuzmin2020Dataset
from dotenv import load_dotenv
from torchcell.timestamp import timestamp
import matplotlib.pyplot as plt
import seaborn as sns
from torchcell.sequence import GeneSet
import pickle
from tqdm import tqdm
# Load environment
load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR", "notes/assets/images")


def get_essential_genes(genome, graph):
    """Extract essential genes from SGD dataset."""
    print("Loading essential genes from SGD...")
    essential_dataset = GeneEssentialitySgdDataset(
        root=osp.join(DATA_ROOT, "data/torchcell/gene_essentiality_sgd"),
        scerevisiae_graph=graph
    )
    
    essential_genes = set()
    for i in range(len(essential_dataset)):
        data = essential_dataset[i]
        experiment = data['experiment']
        if experiment['phenotype']['is_essential']:
            for perturbation in experiment['genotype']['perturbations']:
                essential_genes.add(perturbation['systematic_gene_name'])
    
    print(f"Found {len(essential_genes)} essential genes")
    return essential_genes


def get_low_fitness_genes_kuzmin2018(fitness_threshold=0.5):
    """Extract genes with fitness < threshold from Kuzmin2018 dataset."""
    print(f"Loading genes with fitness < {fitness_threshold} from Kuzmin2018...")
    dataset = SmfKuzmin2018Dataset(
        root=osp.join(DATA_ROOT, "data/torchcell/smf_kuzmin2018")
    )
    
    low_fitness_genes = set()
    gene_fitness_map = {}  # Store fitness values for reference
    
    for i in tqdm(range(len(dataset)), desc="Processing Kuzmin2018"):
        data = dataset[i]
        experiment = data['experiment']
        fitness = experiment['phenotype']['fitness']
        
        if fitness < fitness_threshold:
            for perturbation in experiment['genotype']['perturbations']:
                gene = perturbation['systematic_gene_name']
                low_fitness_genes.add(gene)
                gene_fitness_map[gene] = fitness
    
    print(f"Found {len(low_fitness_genes)} genes with fitness < {fitness_threshold} in Kuzmin2018")
    return low_fitness_genes, gene_fitness_map


def get_low_fitness_genes_costanzo2016(fitness_threshold=0.5):
    """Extract genes with fitness < threshold from Costanzo2016 dataset."""
    print(f"Loading genes with fitness < {fitness_threshold} from Costanzo2016...")
    dataset = SmfCostanzo2016Dataset(
        root=osp.join(DATA_ROOT, "data/torchcell/smf_costanzo2016")
    )
    
    low_fitness_genes = set()
    gene_fitness_map = {}  # Store fitness values for reference
    
    for i in tqdm(range(len(dataset)), desc="Processing Costanzo2016"):
        data = dataset[i]
        experiment = data['experiment']
        fitness = experiment['phenotype']['fitness']
        
        if fitness < fitness_threshold:
            for perturbation in experiment['genotype']['perturbations']:
                gene = perturbation['systematic_gene_name']
                low_fitness_genes.add(gene)
                # Store minimum fitness if gene appears multiple times
                if gene not in gene_fitness_map or fitness < gene_fitness_map[gene]:
                    gene_fitness_map[gene] = fitness
    
    print(f"Found {len(low_fitness_genes)} genes with fitness < {fitness_threshold} in Costanzo2016")
    return low_fitness_genes, gene_fitness_map


def get_low_fitness_genes_kuzmin2020(fitness_threshold=0.5):
    """Extract genes with fitness < threshold from Kuzmin2020 dataset."""
    print(f"Loading genes with fitness < {fitness_threshold} from Kuzmin2020...")
    dataset = SmfKuzmin2020Dataset(
        root=osp.join(DATA_ROOT, "data/torchcell/smf_kuzmin2020")
    )
    
    low_fitness_genes = set()
    gene_fitness_map = {}  # Store fitness values for reference
    
    for i in tqdm(range(len(dataset)), desc="Processing Kuzmin2020"):
        data = dataset[i]
        experiment = data['experiment']
        fitness = experiment['phenotype']['fitness']
        
        if fitness < fitness_threshold:
            for perturbation in experiment['genotype']['perturbations']:
                gene = perturbation['systematic_gene_name']
                low_fitness_genes.add(gene)
                # Store minimum fitness if gene appears multiple times
                if gene not in gene_fitness_map or fitness < gene_fitness_map[gene]:
                    gene_fitness_map[gene] = fitness
    
    print(f"Found {len(low_fitness_genes)} genes with fitness < {fitness_threshold} in Kuzmin2020")
    return low_fitness_genes, gene_fitness_map




def analyze_metabolic_genes(yeast_gem):
    """Analyze genes from YeastGEM bipartite graph."""
    print("\nAnalyzing YeastGEM metabolic genes...")
    B = yeast_gem.bipartite_graph
    
    # Separate node types
    reaction_nodes = [n for n, d in B.nodes(data=True) if d["node_type"] == "reaction"]
    
    # Analyze gene participation
    gene_to_reactions = defaultdict(set)
    gene_to_subsystems = defaultdict(set)
    gene_to_metabolites = defaultdict(set)
    
    for node in reaction_nodes:
        node_data = B.nodes[node]
        genes = node_data.get('genes', set())
        reaction_id = node_data['reaction_id']
        subsystem = node_data.get('subsystem', 'Unknown')
        
        # Get connected metabolites
        metabolites = list(B.neighbors(node))
        
        for gene in genes:
            gene_to_reactions[gene].add(reaction_id)
            gene_to_subsystems[gene].add(subsystem)
            gene_to_metabolites[gene].update(metabolites)
    
    # Find bottleneck reactions
    reaction_gene_combinations = defaultdict(list)
    for node in reaction_nodes:
        node_data = B.nodes[node]
        reaction_id = node_data['reaction_id']
        genes = node_data.get('genes', set())
        if genes:
            reaction_gene_combinations[reaction_id].append(genes)
    
    # Identify genes in bottleneck reactions
    bottleneck_genes = set()
    for reaction_id, gene_combinations in reaction_gene_combinations.items():
        if len(gene_combinations) == 1:  # Only one gene combination
            bottleneck_genes.update(gene_combinations[0])
    
    return gene_to_reactions, gene_to_subsystems, gene_to_metabolites, bottleneck_genes


def identify_central_metabolism_genes(yeast_gem):
    """Identify genes in central metabolic pathways."""
    B = yeast_gem.bipartite_graph
    reaction_nodes = [n for n, d in B.nodes(data=True) if d["node_type"] == "reaction"]
    
    central_pathways = [
        'Glycolysis', 'Gluconeogenesis', 'TCA cycle', 'Citric acid cycle',
        'Pentose phosphate', 'Oxidative phosphorylation', 'ATP synthesis',
        'Pyruvate metabolism', 'Electron transport', 'Fatty acid',
        'Amino acid metabolism', 'Nucleotide metabolism'
    ]
    
    central_genes = set()
    for node in reaction_nodes:
        node_data = B.nodes[node]
        subsystem = node_data.get('subsystem', '')
        genes = node_data.get('genes', set())
        
        for pathway in central_pathways:
            if pathway.lower() in subsystem.lower():
                central_genes.update(genes)
    
    return central_genes


def create_gene_ranking_dataframe(
    genome, graph, yeast_gem, gene_to_reactions, gene_to_subsystems, 
    gene_to_metabolites, bottleneck_genes, central_genes, 
    metabolic_go_genes, kinase_genes, essential_genes,
    low_fitness_genes_kuzmin, low_fitness_genes_costanzo, low_fitness_genes_kuzmin2020,
    fitness_map_kuzmin, fitness_map_costanzo, fitness_map_kuzmin2020
):
    """Create comprehensive gene ranking dataframe."""
    
    # Combine all metabolic and kinase genes
    all_candidate_genes = set()
    all_candidate_genes.update(gene_to_reactions.keys())  # YeastGEM genes
    all_candidate_genes.update(metabolic_go_genes)        # GO metabolic process
    all_candidate_genes.update(kinase_genes)              # GO kinase activity
    
    # Filter to only include genes in genome
    all_candidate_genes = all_candidate_genes.intersection(genome.gene_set)
    
    print(f"\nTotal candidate genes before filtering: {len(all_candidate_genes)}")
    
    # Create dataframe
    gene_data = []
    for gene in all_candidate_genes:
        gene_info = {
            'gene': gene,
            'num_reactions': len(gene_to_reactions.get(gene, set())),
            'num_subsystems': len(gene_to_subsystems.get(gene, set())),
            'num_metabolites': len(gene_to_metabolites.get(gene, set())),
            'is_bottleneck': gene in bottleneck_genes,
            'is_central_metabolism': gene in central_genes,
            'is_metabolic_go': gene in metabolic_go_genes,
            'is_kinase': gene in kinase_genes,
            'is_essential': gene in essential_genes,
            'is_low_fitness_kuzmin': gene in low_fitness_genes_kuzmin,
            'is_low_fitness_costanzo': gene in low_fitness_genes_costanzo,
            'is_low_fitness_kuzmin2020': gene in low_fitness_genes_kuzmin2020,
            'is_low_fitness_any': gene in (low_fitness_genes_kuzmin | low_fitness_genes_costanzo | low_fitness_genes_kuzmin2020),
            'min_fitness_kuzmin': fitness_map_kuzmin.get(gene, None),
            'min_fitness_costanzo': fitness_map_costanzo.get(gene, None),
            'min_fitness_kuzmin2020': fitness_map_kuzmin2020.get(gene, None),
            'subsystems': '; '.join(sorted(gene_to_subsystems.get(gene, [])))
        }
        gene_data.append(gene_info)
    
    df = pd.DataFrame(gene_data)
    
    # Calculate overlap statistics
    print("\nGene set overlaps:")
    yeastgem_genes = set(gene_to_reactions.keys())
    print(f"YeastGEM genes: {len(yeastgem_genes)}")
    print(f"GO metabolic process genes: {len(metabolic_go_genes)}")
    print(f"GO kinase activity genes: {len(kinase_genes)}")
    print(f"YeastGEM ∩ Metabolic GO: {len(yeastgem_genes.intersection(metabolic_go_genes))}")
    print(f"YeastGEM ∩ Kinase: {len(yeastgem_genes.intersection(kinase_genes))}")
    print(f"Metabolic GO ∩ Kinase: {len(metabolic_go_genes.intersection(kinase_genes))}")
    print(f"All three: {len(yeastgem_genes.intersection(metabolic_go_genes).intersection(kinase_genes))}")
    
    return df


def calculate_gene_scores(df):
    """Calculate composite scores for gene ranking."""
    # Normalize numerical features
    df['norm_reactions'] = df['num_reactions'] / (df['num_reactions'].max() + 1e-6)
    df['norm_metabolites'] = df['num_metabolites'] / (df['num_metabolites'].max() + 1e-6)
    df['norm_subsystems'] = df['num_subsystems'] / (df['num_subsystems'].max() + 1e-6)
    
    # Calculate base metabolic importance score
    df['metabolic_score'] = (
        df['norm_reactions'] * 0.4 +
        df['norm_metabolites'] * 0.3 +
        df['norm_subsystems'] * 0.3
    )
    
    # Bonus for bottleneck and central metabolism
    df['metabolic_score'] += df['is_bottleneck'].astype(int) * 0.2
    df['metabolic_score'] += df['is_central_metabolism'].astype(int) * 0.1
    
    # Combined category score (prefer genes in multiple categories)
    df['category_score'] = (
        (df['num_reactions'] > 0).astype(int) +  # In YeastGEM
        df['is_metabolic_go'].astype(int) +
        df['is_kinase'].astype(int)
    )
    
    # Final composite score
    df['composite_score'] = df['metabolic_score'] * 0.7 + (df['category_score'] / 3) * 0.3
    
    return df


def filter_and_rank_genes(df, target_count=220):
    """Filter out essential genes and low fitness genes, then rank."""
    # Filter out essential genes and low fitness genes
    df_filtered = df[(~df['is_essential']) & (~df['is_low_fitness_any'])].copy()
    
    print(f"\nFiltering results:")
    print(f"Removed {df['is_essential'].sum()} essential genes")
    print(f"Removed {df['is_low_fitness_kuzmin'].sum()} low fitness genes from Kuzmin2018")
    print(f"Removed {df['is_low_fitness_costanzo'].sum()} low fitness genes from Costanzo2016")
    print(f"Removed {df['is_low_fitness_kuzmin2020'].sum()} low fitness genes from Kuzmin2020")
    print(f"Removed {df['is_low_fitness_any'].sum()} low fitness genes total (union)")
    print(f"Genes after filtering: {len(df_filtered)}")
    
    # Sort by composite score
    df_filtered = df_filtered.sort_values('composite_score', ascending=False)
    
    # Select top genes
    if len(df_filtered) > target_count:
        df_top = df_filtered.head(target_count).copy()
        print(f"\nSelected top {target_count} genes")
    else:
        df_top = df_filtered.copy()
        print(f"\nAll {len(df_top)} filtered genes selected (less than target {target_count})")
    
    # Add rank
    df_top['rank'] = range(1, len(df_top) + 1)
    
    return df_filtered, df_top


def create_visualizations(df_all, df_filtered, df_top, ts):
    """Create analysis visualizations."""
    # 1. Gene category overlap Venn diagram approximation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Category distribution
    categories = []
    if 'num_reactions' in df_all.columns:
        yeastgem_mask = df_all['num_reactions'] > 0
        categories.append(('YeastGEM', yeastgem_mask.sum()))
    categories.extend([
        ('Metabolic GO', df_all['is_metabolic_go'].sum()),
        ('Kinase GO', df_all['is_kinase'].sum()),
        ('Essential', df_all['is_essential'].sum()),
        ('Low Fitness (Kuzmin18)', df_all['is_low_fitness_kuzmin'].sum()),
        ('Low Fitness (Costanzo16)', df_all['is_low_fitness_costanzo'].sum()),
        ('Low Fitness (Kuzmin20)', df_all['is_low_fitness_kuzmin2020'].sum())
    ])
    
    cats, counts = zip(*categories)
    ax1.bar(cats, counts)
    ax1.set_ylabel('Number of Genes')
    ax1.set_title('Gene Categories Distribution')
    ax1.tick_params(axis='x', rotation=45)
    
    # Score distribution
    ax2.hist(df_filtered['composite_score'], bins=50, alpha=0.7, label='All filtered')
    ax2.hist(df_top['composite_score'], bins=30, alpha=0.7, label=f'Top {len(df_top)}')
    ax2.set_xlabel('Composite Score')
    ax2.set_ylabel('Count')
    ax2.set_title('Gene Score Distribution')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(osp.join(ASSET_IMAGES_DIR, f"gene_ranking_overview_{ts}.png"), dpi=300)
    plt.close()
    
    # 2. Top genes visualization
    plt.figure(figsize=(14, 10))
    top_viz = df_top.head(30).copy()
    
    # Create a heatmap of gene properties
    properties = ['is_bottleneck', 'is_central_metabolism', 'is_metabolic_go', 'is_kinase']
    heatmap_data = top_viz[properties].astype(int)
    
    sns.heatmap(heatmap_data.T, 
                xticklabels=top_viz['gene'], 
                yticklabels=['Bottleneck', 'Central Met.', 'Metabolic GO', 'Kinase'],
                cmap='YlOrRd', cbar_kws={'label': 'Property Present'})
    plt.title('Top 30 Genes Properties')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(osp.join(ASSET_IMAGES_DIR, f"top_genes_properties_{ts}.png"), dpi=300)
    plt.close()
    
    # 3. Filtering summary visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    filtering_stats = [
        ('Total candidates', len(df_all)),
        ('After essential\ngene filter', len(df_all[~df_all['is_essential']])),
        ('After low fitness\nfilter (Kuzmin18)', len(df_all[~df_all['is_low_fitness_kuzmin']])),
        ('After low fitness\nfilter (Costanzo16)', len(df_all[~df_all['is_low_fitness_costanzo']])),
        ('After low fitness\nfilter (Kuzmin20)', len(df_all[~df_all['is_low_fitness_kuzmin2020']])),
        ('After all filters', len(df_filtered)),
        ('Final selected', len(df_top))
    ]
    
    stages, counts = zip(*filtering_stats)
    bars = ax.bar(stages, counts, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2'])
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}',
                ha='center', va='bottom')
    
    ax.set_ylabel('Number of Genes')
    ax.set_title('Gene Filtering Pipeline')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(osp.join(ASSET_IMAGES_DIR, f"gene_filtering_pipeline_{ts}.png"), dpi=300)
    plt.close()


def main():
    ts = timestamp()
    print(f"Starting metabolic gene ranking analysis at {ts}")
    
    # Initialize genome and graph
    print("\nInitializing genome and graph...")
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go")
    )
    
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        genome=genome
    )
    
    # Initialize YeastGEM
    print("\nInitializing YeastGEM...")
    yeast_gem = YeastGEM()
    
    # Get GO gene sets
    print("\nExtracting GO gene sets...")
    metabolic_go_genes = set(graph.go_to_genes.get('GO:0008152', GeneSet()))
    kinase_genes = set(graph.go_to_genes.get('GO:0016301', GeneSet()))
    
    print(f"GO:0008152 (metabolic process): {len(metabolic_go_genes)} genes")
    print(f"GO:0016301 (kinase activity): {len(kinase_genes)} genes")
    
    # Get essential genes
    essential_genes = get_essential_genes(genome, graph)
    
    # Get low fitness genes from all datasets
    low_fitness_genes_kuzmin, fitness_map_kuzmin = get_low_fitness_genes_kuzmin2018()
    low_fitness_genes_costanzo, fitness_map_costanzo = get_low_fitness_genes_costanzo2016()
    low_fitness_genes_kuzmin2020, fitness_map_kuzmin2020 = get_low_fitness_genes_kuzmin2020()
    
    # Analyze metabolic genes from YeastGEM
    gene_to_reactions, gene_to_subsystems, gene_to_metabolites, bottleneck_genes = analyze_metabolic_genes(yeast_gem)
    
    # Identify central metabolism genes
    central_genes = identify_central_metabolism_genes(yeast_gem)
    print(f"\nCentral metabolism genes: {len(central_genes)}")
    print(f"Bottleneck genes: {len(bottleneck_genes)}")
    
    # Create ranking dataframe
    df = create_gene_ranking_dataframe(
        genome, graph, yeast_gem, gene_to_reactions, gene_to_subsystems,
        gene_to_metabolites, bottleneck_genes, central_genes,
        metabolic_go_genes, kinase_genes, essential_genes,
        low_fitness_genes_kuzmin, low_fitness_genes_costanzo, low_fitness_genes_kuzmin2020,
        fitness_map_kuzmin, fitness_map_costanzo, fitness_map_kuzmin2020
    )
    
    # Calculate scores
    df = calculate_gene_scores(df)
    
    # Filter and rank
    df_filtered, df_top = filter_and_rank_genes(df, target_count=220)
    
    # Save results
    results_dir = "/Users/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save full analysis
    df.to_csv(osp.join(results_dir, f"all_metabolic_genes_analysis_{ts}.csv"), index=False)
    df_filtered.to_csv(osp.join(results_dir, f"filtered_metabolic_genes_{ts}.csv"), index=False)
    df_top.to_csv(osp.join(results_dir, f"top_metabolic_genes_ranked_{ts}.csv"), index=False)
    
    # Save just the gene list
    with open(osp.join(results_dir, f"selected_genes_list_{ts}.txt"), 'w') as f:
        for gene in df_top['gene']:
            f.write(f"{gene}\n")
    
    # Save as pickle for easy loading
    with open(osp.join(results_dir, f"gene_selection_data_{ts}.pkl"), 'wb') as f:
        pickle.dump({
            'selected_genes': list(df_top['gene']),
            'df_all': df,
            'df_filtered': df_filtered,
            'df_top': df_top,
            'essential_genes': essential_genes,
            'metabolic_go_genes': metabolic_go_genes,
            'kinase_genes': kinase_genes,
            'low_fitness_genes_kuzmin': low_fitness_genes_kuzmin,
            'low_fitness_genes_costanzo': low_fitness_genes_costanzo,
            'low_fitness_genes_kuzmin2020': low_fitness_genes_kuzmin2020,
            'fitness_map_kuzmin': fitness_map_kuzmin,
            'fitness_map_costanzo': fitness_map_costanzo,
            'fitness_map_kuzmin2020': fitness_map_kuzmin2020
        }, f)
    
    # Create visualizations
    create_visualizations(df, df_filtered, df_top, ts)
    
    # Print summary
    print("\n" + "="*80)
    print("GENE SELECTION SUMMARY")
    print("="*80)
    print(f"Total candidate genes analyzed: {len(df)}")
    print(f"After filtering (non-essential, non-low-fitness): {len(df_filtered)}")
    print(f"Final selected genes: {len(df_top)}")
    print(f"\nDetailed filtering breakdown:")
    print(f"  - Essential genes removed: {df['is_essential'].sum()}")
    print(f"  - Low fitness genes (Kuzmin2018) removed: {df['is_low_fitness_kuzmin'].sum()}")
    print(f"  - Low fitness genes (Costanzo2016) removed: {df['is_low_fitness_costanzo'].sum()}")
    print(f"  - Low fitness genes (Kuzmin2020) removed: {df['is_low_fitness_kuzmin2020'].sum()}")
    print(f"  - Low fitness genes (any dataset) removed: {df['is_low_fitness_any'].sum()}")
    overlap_count = ((df['is_essential']) & (df['is_low_fitness_any'])).sum()
    print(f"  - Overlap (essential AND low fitness): {overlap_count}")
    print(f"\nTop 10 genes by composite score:")
    print(df_top[['rank', 'gene', 'composite_score', 'num_reactions', 'is_bottleneck', 'is_central_metabolism', 'is_kinase']].head(10))
    
    print(f"\nResults saved to: {results_dir}")
    print(f"Timestamp: {ts}")


if __name__ == "__main__":
    main()