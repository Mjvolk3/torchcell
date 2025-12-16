#!/usr/bin/env python
"""
Expand gene selection for inference_1 from ~220 to ~3000+ genes.
Combines genes from 4 sources:
1. Ohya morphology (top 2000 by most stringent significance)
2. Kemmeren responsive mutants
3. Sameith doubles
4. Expanded metabolic genes (GO terms + betaxanthin + original YeastGEM/kinase)

Filters out essential genes and low fitness genes (< 0.9).
"""

import os
import os.path as osp
import pandas as pd
from torchcell.metabolism.yeast_GEM import YeastGEM
from torchcell.graph import SCerevisiaeGraph
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.datasets.scerevisiae.sgd import GeneEssentialitySgdDataset
from torchcell.datasets.scerevisiae.kuzmin2018 import SmfKuzmin2018Dataset
from torchcell.datasets.scerevisiae.costanzo2016 import SmfCostanzo2016Dataset
from torchcell.datasets.scerevisiae.kuzmin2020 import SmfKuzmin2020Dataset
from torchcell.datasets.scerevisiae.kemmeren2014 import MicroarrayKemmeren2014Dataset
from torchcell.datasets.scerevisiae.ohya2005 import ScmdOhya2005Dataset
from torchcell.datasets.scerevisiae.sameith2015 import DmMicroarraySameith2015Dataset
from dotenv import load_dotenv
from torchcell.timestamp import timestamp
from torchcell.sequence import GeneSet
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load environment
load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR", "notes/assets/images")

# Global parameters
SINGLE_GENE_FITNESS_THRESHOLD = 0.9  # Remove genes with fitness < 0.9
DOUBLE_GENE_FITNESS_THRESHOLD = 0.9  # For triple filtering (Phase 2)
OHYA_TOP_N_GENES = 2000  # Top morphology genes to include


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


def get_low_fitness_genes(fitness_threshold=0.9):
    """Extract genes with fitness < threshold from all fitness datasets."""
    print(f"\nLoading genes with fitness < {fitness_threshold} from fitness datasets...")

    low_fitness_genes = set()
    gene_fitness_map = {}

    # Load from all three datasets
    datasets = [
        ('Kuzmin2018', SmfKuzmin2018Dataset, 'smf_kuzmin2018'),
        ('Costanzo2016', SmfCostanzo2016Dataset, 'smf_costanzo2016'),
        ('Kuzmin2020', SmfKuzmin2020Dataset, 'smf_kuzmin2020'),
    ]

    for dataset_name, dataset_class, dataset_path in datasets:
        print(f"  Processing {dataset_name}...")
        dataset = dataset_class(root=osp.join(DATA_ROOT, f"data/torchcell/{dataset_path}"))

        for i in tqdm(range(len(dataset)), desc=f"  {dataset_name}"):
            data = dataset[i]
            experiment = data['experiment']
            fitness = experiment['phenotype']['fitness']

            if fitness < fitness_threshold:
                for perturbation in experiment['genotype']['perturbations']:
                    gene = perturbation['systematic_gene_name']
                    low_fitness_genes.add(gene)
                    # Store minimum fitness
                    if gene not in gene_fitness_map or fitness < gene_fitness_map[gene]:
                        gene_fitness_map[gene] = fitness

    print(f"Found {len(low_fitness_genes)} genes with fitness < {fitness_threshold}")
    return low_fitness_genes, gene_fitness_map


def get_ohya_morphology_genes(genome, top_n=2000):
    """Extract top N genes by morphological significance from Ohya 2005.

    Uses the 0.000001 column (most stringent threshold) from the Excel file.
    """
    print(f"\nExtracting top {top_n} morphology genes from Ohya 2005...")

    excel_path = osp.join(
        DATA_ROOT,
        "data/torchcell/experiments/006-kuzmin-tmi/inference_1",
        "Ohya - SI - table3 ORF Statisitcs.xlsx"
    )

    if not osp.exists(excel_path):
        raise FileNotFoundError(
            f"Ohya Excel file not found at {excel_path}\n"
            f"Please ensure the file exists at this location."
        )

    # Read Excel file - skip first row which is descriptive header
    # Row 0: Long description
    # Row 1: Column headers (ORF name, gene name, 0.001, 0.0001, 0.00001, 0.000001)
    # Row 2+: Data
    df = pd.read_excel(excel_path, header=1)  # Use row 1 as header

    # Now columns should be properly named
    # First column: ORF name, Last column: 0.000001 threshold (1.00E-06)
    orf_col = 'ORF name'  # Explicit column name
    most_stringent_col = df.columns[-1]  # Last column (0.000001)

    print(f"  Excel columns: {list(df.columns)}")
    print(f"  Using ORF column: {orf_col}")
    print(f"  Using significance column: {most_stringent_col}")

    # Data starts immediately, no need to skip rows
    df_data = df

    # Sort by most stringent column (descending)
    df_sorted = df_data.sort_values(by=most_stringent_col, ascending=False)

    # Take top N
    df_top = df_sorted.head(top_n)

    # Extract gene names and verify they exist in genome
    ohya_genes = set()
    skipped = 0

    for _, row in df_top.iterrows():
        gene_name = str(row[orf_col]).strip().upper()

        # Verify gene exists in genome.gene_set
        if gene_name in genome.gene_set:
            ohya_genes.add(gene_name)
        else:
            skipped += 1

    print(f"  Extracted {len(ohya_genes)} genes from top {top_n}")
    print(f"  Skipped {skipped} genes not in genome.gene_set")

    return ohya_genes


def get_kemmeren_responsive_genes():
    """Extract all responsive mutant genes from Kemmeren 2014."""
    print("\nExtracting responsive mutant genes from Kemmeren 2014...")

    dataset = MicroarrayKemmeren2014Dataset(
        root=osp.join(DATA_ROOT, "data/torchcell/microarray_kemmeren2014")
    )

    # Filter for responsive mutants using the new column
    responsive_df = dataset.df[dataset.df['is_responsive_mutant'] == True]

    # Extract unique systematic gene names (excluding wildtype samples)
    kemmeren_genes = set(
        responsive_df[responsive_df['is_deletion'] == True]['systematic_gene_name'].dropna().unique()
    )

    print(f"  Found {len(kemmeren_genes)} responsive mutant genes")
    print(f"  (from {len(responsive_df)} responsive samples)")

    return kemmeren_genes


def get_sameith_doubles_genes():
    """Extract all unique genes from Sameith 2015 double mutants."""
    print("\nExtracting genes from Sameith 2015 double mutants...")

    dataset = DmMicroarraySameith2015Dataset(
        root=osp.join(DATA_ROOT, "data/torchcell/dm_microarray_sameith2015")
    )

    sameith_genes = set(dataset.gene_set)

    print(f"  Found {len(sameith_genes)} unique genes in double mutants")

    return sameith_genes


def get_expanded_metabolic_genes(genome, graph, yeast_gem):
    """Get expanded metabolic genes from GO terms + YeastGEM + betaxanthin."""
    print("\nExtracting expanded metabolic genes...")

    metabolic_genes = set()

    # 1. Original GO terms from rank_metabolic_genes.py
    print("  Getting GO:0008152 (metabolic process)...")
    go_metabolic = set(graph.go_to_genes.get('GO:0008152', GeneSet()))
    metabolic_genes.update(go_metabolic)
    print(f"    Added {len(go_metabolic)} genes")

    print("  Getting GO:0016301 (kinase activity)...")
    go_kinase = set(graph.go_to_genes.get('GO:0016301', GeneSet()))
    metabolic_genes.update(go_kinase)
    print(f"    Added {len(go_kinase)} genes")

    # 2. New GO terms for beta carotene and amino acids
    print("  Getting GO:0016117 (carotenoid biosynthetic process)...")
    go_carotenoid = set(graph.go_to_genes.get('GO:0016117', GeneSet()))
    metabolic_genes.update(go_carotenoid)
    print(f"    Added {len(go_carotenoid)} genes")

    print("  Getting GO:0008652 (amino acid biosynthetic process)...")
    go_amino_acid = set(graph.go_to_genes.get('GO:0008652', GeneSet()))
    metabolic_genes.update(go_amino_acid)
    print(f"    Added {len(go_amino_acid)} genes")

    # 3. YeastGEM genes
    print("  Getting YeastGEM metabolic genes...")
    B = yeast_gem.bipartite_graph
    reaction_nodes = [n for n, d in B.nodes(data=True) if d["node_type"] == "reaction"]
    yeastgem_genes = set()
    for node in reaction_nodes:
        node_data = B.nodes[node]
        genes = node_data.get('genes', set())
        yeastgem_genes.update(genes)

    # Filter out mitochondrial genes (Q-prefix)
    yeastgem_genes_filtered = {g for g in yeastgem_genes if not g.startswith('Q')}
    mitochondrial_count = len(yeastgem_genes) - len(yeastgem_genes_filtered)

    metabolic_genes.update(yeastgem_genes_filtered)
    print(f"    Added {len(yeastgem_genes_filtered)} genes ({mitochondrial_count} mitochondrial genes excluded)")

    # 4. Betaxanthin precursor genes (ARO4, ARO7)
    print("  Adding betaxanthin precursor genes...")
    betaxanthin_genes = {'YBR249C', 'YPR060C'}  # ARO4, ARO7

    # Verify these exist in genome
    verified_betaxanthin = set()
    for gene in betaxanthin_genes:
        if gene in genome.gene_set:
            verified_betaxanthin.add(gene)
            print(f"    ✓ {gene} exists in genome")
        else:
            print(f"    ✗ {gene} NOT in genome.gene_set")

    metabolic_genes.update(verified_betaxanthin)
    print(f"    Added {len(verified_betaxanthin)} betaxanthin genes")

    print(f"  Total expanded metabolic genes: {len(metabolic_genes)}")

    return metabolic_genes, verified_betaxanthin


def save_gene_list(genes, filename, results_dir):
    """Save gene list to text file."""
    filepath = osp.join(results_dir, filename)
    with open(filepath, 'w') as f:
        for gene in sorted(genes):
            f.write(f"{gene}\n")
    print(f"  Saved {len(genes)} genes to {filename}")
    return filepath


def create_analysis_dataframe(
    all_gene_sources,
    essential_genes,
    low_fitness_genes,
    gene_fitness_map
):
    """Create comprehensive analysis dataframe."""
    print("\nCreating analysis dataframe...")

    # Combine all genes
    all_genes = set()
    for source_name, genes in all_gene_sources.items():
        all_genes.update(genes)

    # Create dataframe
    gene_data = []
    for gene in sorted(all_genes):
        gene_info = {
            'gene': gene,
            'in_ohya_morphology': gene in all_gene_sources.get('ohya_morphology', set()),
            'in_kemmeren_responsive': gene in all_gene_sources.get('kemmeren_responsive', set()),
            'in_sameith_doubles': gene in all_gene_sources.get('sameith_doubles', set()),
            'in_expanded_metabolic': gene in all_gene_sources.get('expanded_metabolic', set()),
            'is_essential': gene in essential_genes,
            'is_low_fitness': gene in low_fitness_genes,
            'min_fitness': gene_fitness_map.get(gene, None),
            'num_sources': sum([
                gene in all_gene_sources.get('ohya_morphology', set()),
                gene in all_gene_sources.get('kemmeren_responsive', set()),
                gene in all_gene_sources.get('sameith_doubles', set()),
                gene in all_gene_sources.get('expanded_metabolic', set()),
            ])
        }
        gene_data.append(gene_info)

    df = pd.DataFrame(gene_data)

    print(f"  Created dataframe with {len(df)} genes")
    print(f"\nGene overlap statistics:")
    for source_name, genes in all_gene_sources.items():
        print(f"  {source_name}: {len(genes)} genes")

    # Source overlap counts
    overlap_counts = df['num_sources'].value_counts().sort_index()
    print(f"\nGenes by number of sources:")
    for num_sources, count in overlap_counts.items():
        print(f"  {num_sources} sources: {count} genes")

    return df


def create_visualizations(df, all_gene_sources, ts, results_dir):
    """Create analysis visualizations."""
    print("\nCreating visualizations...")

    # 1. Source distribution
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Source sizes
    source_names = list(all_gene_sources.keys())
    source_sizes = [len(genes) for genes in all_gene_sources.values()]

    ax1.bar(source_names, source_sizes)
    ax1.set_ylabel('Number of Genes')
    ax1.set_title('Gene Count by Source')
    ax1.tick_params(axis='x', rotation=45)

    # Add count labels on bars
    for i, (name, size) in enumerate(zip(source_names, source_sizes)):
        ax1.text(i, size, f'{size:,}', ha='center', va='bottom')

    # Genes by number of sources
    overlap_counts = df['num_sources'].value_counts().sort_index()
    ax2.bar(overlap_counts.index, overlap_counts.values)
    ax2.set_xlabel('Number of Sources')
    ax2.set_ylabel('Number of Genes')
    ax2.set_title('Gene Distribution by Source Overlap')

    # Add count labels
    for num, count in overlap_counts.items():
        ax2.text(num, count, f'{count:,}', ha='center', va='bottom')

    # Filtering summary
    filtering_stats = [
        ('Total\ncandidates', len(df)),
        ('After essential\nfilter', len(df[~df['is_essential']])),
        ('After low fitness\nfilter', len(df[~df['is_essential'] & ~df['is_low_fitness']])),
    ]

    stages, counts = zip(*filtering_stats)
    bars = ax3.bar(stages, counts, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax3.set_ylabel('Number of Genes')
    ax3.set_title('Gene Filtering Pipeline')

    # Add count labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}',
                ha='center', va='bottom')

    # Source overlap heatmap
    sources = list(all_gene_sources.keys())
    overlap_matrix = np.zeros((len(sources), len(sources)))

    for i, source1 in enumerate(sources):
        for j, source2 in enumerate(sources):
            if i <= j:
                overlap = len(all_gene_sources[source1] & all_gene_sources[source2])
                overlap_matrix[i, j] = overlap
                overlap_matrix[j, i] = overlap

    sns.heatmap(overlap_matrix, annot=True, fmt='.0f',
                xticklabels=[s.replace('_', '\n') for s in sources],
                yticklabels=[s.replace('_', '\n') for s in sources],
                cmap='YlOrRd', ax=ax4)
    ax4.set_title('Source Overlap Matrix')

    plt.tight_layout()
    plt.savefig(osp.join(ASSET_IMAGES_DIR, f"gene_selection_overview_{ts}.png"), dpi=300)
    plt.close()

    print(f"  Saved visualization to gene_selection_overview_{ts}.png")


def main():
    ts = timestamp()
    print(f"Starting expanded gene selection at {ts}")
    print(f"Parameters:")
    print(f"  SINGLE_GENE_FITNESS_THRESHOLD: {SINGLE_GENE_FITNESS_THRESHOLD}")
    print(f"  OHYA_TOP_N_GENES: {OHYA_TOP_N_GENES}")

    # Initialize genome and graph
    print("\n" + "="*80)
    print("Initializing genome and graph...")
    print("="*80)
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go")
    )

    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        genome=genome
    )

    # Initialize YeastGEM
    yeast_gem = YeastGEM()

    print(f"Genome gene set size: {len(genome.gene_set)}")

    # Setup results directory
    results_dir = "/Users/michaelvolk/Documents/projects/torchcell/experiments/006-kuzmin-tmi/results/inference_preprocessing_expansion"
    os.makedirs(results_dir, exist_ok=True)

    # Get essential genes and low fitness genes
    print("\n" + "="*80)
    print("Getting filtering gene sets...")
    print("="*80)
    essential_genes = get_essential_genes(genome, graph)
    low_fitness_genes, gene_fitness_map = get_low_fitness_genes(SINGLE_GENE_FITNESS_THRESHOLD)

    # Collect genes from all 4 sources
    print("\n" + "="*80)
    print("Collecting genes from all sources...")
    print("="*80)

    all_gene_sources = {}

    # Source 1: Ohya morphology
    ohya_genes = get_ohya_morphology_genes(genome, top_n=OHYA_TOP_N_GENES)
    all_gene_sources['ohya_morphology'] = ohya_genes
    save_gene_list(ohya_genes, f"ohya_morphology_genes_list_{ts}.txt", results_dir)

    # Source 2: Kemmeren responsive
    kemmeren_genes = get_kemmeren_responsive_genes()
    all_gene_sources['kemmeren_responsive'] = kemmeren_genes
    save_gene_list(kemmeren_genes, f"kemmeren_responsive_genes_list_{ts}.txt", results_dir)

    # Source 3: Sameith doubles
    sameith_genes = get_sameith_doubles_genes()
    all_gene_sources['sameith_doubles'] = sameith_genes
    save_gene_list(sameith_genes, f"sameith_doubles_genes_list_{ts}.txt", results_dir)

    # Source 4: Expanded metabolic
    metabolic_genes, betaxanthin_genes = get_expanded_metabolic_genes(genome, graph, yeast_gem)
    all_gene_sources['expanded_metabolic'] = metabolic_genes
    save_gene_list(metabolic_genes, f"expanded_metabolic_genes_list_{ts}.txt", results_dir)
    save_gene_list(betaxanthin_genes, f"betaxanthin_genes_list_{ts}.txt", results_dir)

    # Take union of all sources
    print("\n" + "="*80)
    print("Combining all gene sources...")
    print("="*80)
    all_genes_union = set()
    for source_name, genes in all_gene_sources.items():
        all_genes_union.update(genes)

    # Filter out mitochondrial genes from all sources
    all_genes_before_mito_filter = len(all_genes_union)
    all_genes_union = {g for g in all_genes_union if not g.startswith('Q')}
    mito_filtered_count = all_genes_before_mito_filter - len(all_genes_union)

    print(f"Total unique genes before filtering: {all_genes_before_mito_filter}")
    if mito_filtered_count > 0:
        print(f"  Filtered {mito_filtered_count} mitochondrial genes (Q-prefix)")
    print(f"Total unique genes after mitochondrial filter: {len(all_genes_union)}")

    # Apply filtering
    print("\nApplying filters...")
    filtered_genes = all_genes_union - essential_genes - low_fitness_genes

    print(f"  Removed {len(all_genes_union & essential_genes)} essential genes")
    print(f"  Removed {len(all_genes_union & low_fitness_genes)} low fitness genes")
    print(f"  Final gene count: {len(filtered_genes)}")

    # Save final expanded gene list
    print("\n" + "="*80)
    print("Saving final expanded gene list...")
    print("="*80)
    final_filepath = save_gene_list(
        filtered_genes,
        f"expanded_genes_inference_1_{ts}.txt",
        results_dir
    )

    # Create analysis dataframe
    df = create_analysis_dataframe(
        all_gene_sources,
        essential_genes,
        low_fitness_genes,
        gene_fitness_map
    )

    # Save analysis CSV
    analysis_filepath = osp.join(results_dir, f"expanded_genes_analysis_{ts}.csv")
    df.to_csv(analysis_filepath, index=False)
    print(f"\n  Saved analysis to {analysis_filepath}")

    # Create visualizations
    create_visualizations(df, all_gene_sources, ts, results_dir)

    # Print final summary
    print("\n" + "="*80)
    print("GENE SELECTION SUMMARY")
    print("="*80)
    print(f"Source contributions:")
    for source_name, genes in all_gene_sources.items():
        print(f"  {source_name}: {len(genes)} genes")

    print(f"\nTotal unique genes (union): {len(all_genes_union)}")
    print(f"After filtering:")
    print(f"  Essential genes removed: {len(all_genes_union & essential_genes)}")
    print(f"  Low fitness genes removed: {len(all_genes_union & low_fitness_genes)}")
    print(f"  Final selected genes: {len(filtered_genes)}")

    print(f"\nFiles saved:")
    print(f"  Final gene list: {final_filepath}")
    print(f"  Analysis CSV: {analysis_filepath}")
    print(f"  Timestamp: {ts}")


if __name__ == "__main__":
    main()
