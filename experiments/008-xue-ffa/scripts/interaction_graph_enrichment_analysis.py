# experiments/008-xue-ffa/scripts/interaction_graph_enrichment_analysis
# [[experiments.008-xue-ffa.scripts.interaction_graph_enrichment_analysis]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/008-xue-ffa/scripts/interaction_graph_enrichment_analysis
# Test file: experiments/008-xue-ffa/scripts/test_interaction_graph_enrichment_analysis.py

import os
import os.path as osp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dotenv import load_dotenv
from torchcell.timestamp import timestamp
from torchcell.graph.graph import SCerevisiaeGraph
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
import glob
import warnings
from scipy.stats import fisher_exact, chi2_contingency
from typing import Dict, List, Tuple, Set
import networkx as nx
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
if not ASSET_IMAGES_DIR:
    PROJECT_ROOT = os.getenv("PROJECT_ROOT", os.getcwd())
    ASSET_IMAGES_DIR = osp.join(PROJECT_ROOT, "notes/assets/images")
os.makedirs(ASSET_IMAGES_DIR, exist_ok=True)

# Results directory
RESULTS_DIR = Path("/Users/michaelvolk/Documents/projects/torchcell/experiments/008-xue-ffa/results")
GRAPH_ENRICHMENT_DIR = RESULTS_DIR / "graph_enrichment"
os.makedirs(GRAPH_ENRICHMENT_DIR, exist_ok=True)

# Apply torchcell style
STYLE_PATH = "/Users/michaelvolk/Documents/projects/torchcell/torchcell/torchcell.mplstyle"
if osp.exists(STYLE_PATH):
    plt.style.use(STYLE_PATH)


def load_latest_csv(pattern):
    """Load CSV file from pattern (no wildcards)."""
    # Remove wildcard from pattern if present
    filename = pattern.replace('_*.csv', '.csv')
    filepath = RESULTS_DIR / filename
    if not filepath.exists():
        return None
    return pd.read_csv(filepath)


def normalize_ffa_names(df):
    """Normalize FFA names to standard format."""
    if df is None:
        return None

    ffa_map = {
        'C140': 'C14:0',
        'C160': 'C16:0',
        'C180': 'C18:0',
        'C161': 'C16:1',
        'C181': 'C18:1',
        'Total Titer': 'Total Titer'
    }

    df = df.copy()
    if 'ffa_type' in df.columns:
        df['ffa_type'] = df['ffa_type'].replace(ffa_map)

    return df


def check_digenic_edge(gene1: str, gene2: str, graph: nx.Graph, genome) -> bool:
    """Check if there's an edge between two genes in a graph.

    Converts standard gene names to systematic names if needed.
    """
    # Try to convert to systematic names if genome has alias mapping
    if hasattr(genome, 'alias_to_systematic'):
        sys_names1 = genome.alias_to_systematic.get(gene1, [gene1])
        sys_names2 = genome.alias_to_systematic.get(gene2, [gene2])

        # If mapping returns a list, use first element
        if isinstance(sys_names1, list) and len(sys_names1) > 0:
            gene1_sys = sys_names1[0]
        else:
            gene1_sys = gene1

        if isinstance(sys_names2, list) and len(sys_names2) > 0:
            gene2_sys = sys_names2[0]
        else:
            gene2_sys = gene2
    else:
        gene1_sys = gene1
        gene2_sys = gene2

    if gene1_sys not in graph.nodes() or gene2_sys not in graph.nodes():
        return False
    return graph.has_edge(gene1_sys, gene2_sys) or graph.has_edge(gene2_sys, gene1_sys)


def check_trigenic_triangle(gene1: str, gene2: str, gene3: str, graph: nx.Graph, genome) -> bool:
    """Check if three genes form a triangle (cycle) in the graph."""
    # Convert to systematic names
    genes = [gene1, gene2, gene3]
    sys_genes = []

    if hasattr(genome, 'alias_to_systematic'):
        for gene in genes:
            sys_names = genome.alias_to_systematic.get(gene, [gene])
            if isinstance(sys_names, list) and len(sys_names) > 0:
                sys_genes.append(sys_names[0])
            else:
                sys_genes.append(gene)
    else:
        sys_genes = genes

    # Check all genes are in graph
    if not all(g in graph.nodes() for g in sys_genes):
        return False

    # Check if all three edges exist (triangle)
    edges_exist = [
        graph.has_edge(sys_genes[0], sys_genes[1]) or graph.has_edge(sys_genes[1], sys_genes[0]),
        graph.has_edge(sys_genes[1], sys_genes[2]) or graph.has_edge(sys_genes[2], sys_genes[1]),
        graph.has_edge(sys_genes[2], sys_genes[0]) or graph.has_edge(sys_genes[0], sys_genes[2])
    ]

    return all(edges_exist)


def check_trigenic_connected(gene1: str, gene2: str, gene3: str, graph: nx.Graph, genome) -> bool:
    """Check if three genes are connected (at least 2 edges forming a path)."""
    # Convert to systematic names
    genes = [gene1, gene2, gene3]
    sys_genes = []

    if hasattr(genome, 'alias_to_systematic'):
        for gene in genes:
            sys_names = genome.alias_to_systematic.get(gene, [gene])
            if isinstance(sys_names, list) and len(sys_names) > 0:
                sys_genes.append(sys_names[0])
            else:
                sys_genes.append(gene)
    else:
        sys_genes = genes

    # Check all genes are in graph
    if not all(g in graph.nodes() for g in sys_genes):
        return False

    # Check all possible edge combinations
    edges = [
        (sys_genes[0], sys_genes[1]), (sys_genes[1], sys_genes[0]),
        (sys_genes[1], sys_genes[2]), (sys_genes[2], sys_genes[1]),
        (sys_genes[2], sys_genes[0]), (sys_genes[0], sys_genes[2])
    ]

    edge_count = sum(graph.has_edge(e[0], e[1]) for e in edges)

    # Connected if at least 2 edges exist
    return edge_count >= 2


def parse_gene_set(gene_set_str: str) -> List[str]:
    """Parse gene set string to list of genes."""
    # Handle both ':' and '_' separators
    return gene_set_str.replace(':', '_').split('_')


def analyze_graph_overlap(interactions_df: pd.DataFrame, gene_graphs: Dict,
                          interaction_type: str = 'digenic', genome=None,
                          interaction_sign: str = 'all') -> pd.DataFrame:
    """
    Analyze graph overlap for interactions.

    Args:
        interactions_df: DataFrame with interaction data
        gene_graphs: Dictionary of gene graphs
        interaction_type: 'digenic' or 'trigenic'
        genome: Genome object for gene name conversion
        interaction_sign: 'all', 'positive', or 'negative' - filter by interaction score sign

    Returns:
        DataFrame with graph overlap results
    """
    # Filter by interaction sign if specified
    if interaction_sign == 'positive':
        interactions_df = interactions_df[interactions_df['interaction_score'] > 0].copy()
    elif interaction_sign == 'negative':
        interactions_df = interactions_df[interactions_df['interaction_score'] < 0].copy()
    # else: 'all' - no filtering

    results = []

    for idx, row in tqdm(interactions_df.iterrows(),
                        total=len(interactions_df),
                        desc=f"Analyzing {interaction_type} {interaction_sign} interactions"):
        genes = parse_gene_set(row['gene_set'])

        if interaction_type == 'digenic' and len(genes) != 2:
            continue
        if interaction_type == 'trigenic' and len(genes) != 3:
            continue

        overlap_data = {
            'gene_set': row['gene_set'],
            'ffa_type': row['ffa_type'],
            'p_value': row['p_value'],
            'significant_p05': row['significant_p05'],
            'interaction_score': row['interaction_score']
        }

        # Check overlap in each graph
        for graph_name, gene_graph in gene_graphs.items():
            graph = gene_graph.graph  # Get underlying networkx graph

            if interaction_type == 'digenic':
                overlap_data[f'{graph_name}_edge'] = check_digenic_edge(genes[0], genes[1], graph, genome)
            else:  # trigenic
                overlap_data[f'{graph_name}_triangle'] = check_trigenic_triangle(
                    genes[0], genes[1], genes[2], graph, genome)
                overlap_data[f'{graph_name}_connected'] = check_trigenic_connected(
                    genes[0], genes[1], genes[2], graph, genome)

        results.append(overlap_data)

    return pd.DataFrame(results)


def calculate_enrichment(overlap_df: pd.DataFrame, graph_columns: List[str]) -> pd.DataFrame:
    """
    Calculate enrichment of graph overlap in significant vs non-significant interactions.

    Uses Fisher's exact test for each graph type.
    """
    enrichment_results = []

    for col in graph_columns:
        # Create contingency table
        sig_with_edge = len(overlap_df[(overlap_df['significant_p05']) & (overlap_df[col])])
        sig_without_edge = len(overlap_df[(overlap_df['significant_p05']) & (~overlap_df[col])])
        nonsig_with_edge = len(overlap_df[(~overlap_df['significant_p05']) & (overlap_df[col])])
        nonsig_without_edge = len(overlap_df[(~overlap_df['significant_p05']) & (~overlap_df[col])])

        # Fisher's exact test
        contingency_table = [
            [sig_with_edge, sig_without_edge],
            [nonsig_with_edge, nonsig_without_edge]
        ]

        try:
            odds_ratio, p_value = fisher_exact(contingency_table)
        except:
            odds_ratio, p_value = np.nan, np.nan

        # Calculate percentages
        sig_pct = 100 * sig_with_edge / (sig_with_edge + sig_without_edge) if (sig_with_edge + sig_without_edge) > 0 else 0
        nonsig_pct = 100 * nonsig_with_edge / (nonsig_with_edge + nonsig_without_edge) if (nonsig_with_edge + nonsig_without_edge) > 0 else 0

        # Calculate fold enrichment - handle edge cases
        if sig_pct == 0 and nonsig_pct == 0:
            fold_enrichment = np.nan  # Both 0 - undefined
        elif nonsig_pct == 0:
            fold_enrichment = np.inf  # Only nonsig is 0 - infinite enrichment
        else:
            fold_enrichment = sig_pct / nonsig_pct

        enrichment_results.append({
            'graph_type': col,
            'sig_with_overlap': sig_with_edge,
            'sig_without_overlap': sig_without_edge,
            'nonsig_with_overlap': nonsig_with_edge,
            'nonsig_without_overlap': nonsig_without_edge,
            'sig_pct': sig_pct,
            'nonsig_pct': nonsig_pct,
            'fold_enrichment': fold_enrichment,
            'odds_ratio': odds_ratio,
            'p_value': p_value
        })

    return pd.DataFrame(enrichment_results)


def main():
    """Main analysis function."""
    print("="*80)
    print("GENETIC INTERACTION GRAPH ENRICHMENT ANALYSIS")
    print("="*80)

    # Initialize genome and graphs
    print("\nInitializing genome and gene graphs...")
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False
    )

    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome
    )

    # Select graph types (STRING 12.0 only)
    graph_types = [
        'physical',
        'regulatory',
        'genetic',
        'tflink',
        'string12_0_neighborhood',
        'string12_0_fusion',
        'string12_0_cooccurence',
        'string12_0_coexpression',
        'string12_0_experimental',
        'string12_0_database'
    ]

    print("\nLoading gene graphs...")
    gene_graphs = {}
    for graph_type in tqdm(graph_types):
        if graph_type == 'physical':
            gene_graphs[graph_type] = graph.G_physical
        elif graph_type == 'regulatory':
            gene_graphs[graph_type] = graph.G_regulatory
        elif graph_type == 'genetic':
            gene_graphs[graph_type] = graph.G_genetic
        elif graph_type == 'tflink':
            gene_graphs[graph_type] = graph.G_tflink
        elif graph_type == 'string12_0_neighborhood':
            gene_graphs[graph_type] = graph.G_string12_0_neighborhood
        elif graph_type == 'string12_0_fusion':
            gene_graphs[graph_type] = graph.G_string12_0_fusion
        elif graph_type == 'string12_0_cooccurence':
            gene_graphs[graph_type] = graph.G_string12_0_cooccurence
        elif graph_type == 'string12_0_coexpression':
            gene_graphs[graph_type] = graph.G_string12_0_coexpression
        elif graph_type == 'string12_0_experimental':
            gene_graphs[graph_type] = graph.G_string12_0_experimental
        elif graph_type == 'string12_0_database':
            gene_graphs[graph_type] = graph.G_string12_0_database

        print(f"  {graph_type}: {gene_graphs[graph_type].graph.number_of_nodes()} nodes, "
              f"{gene_graphs[graph_type].graph.number_of_edges()} edges")

    # Model types
    models = {
        'multiplicative': {
            'digenic': 'multiplicative_digenic_interactions_3_delta_normalized.csv',
            'trigenic': 'multiplicative_trigenic_interactions_3_delta_normalized.csv'
        },
        'additive': {
            'digenic': 'additive_digenic_interactions_3_delta_normalized.csv',
            'trigenic': 'additive_trigenic_interactions_3_delta_normalized.csv'
        },
        'log_ols': {
            'digenic': 'glm_models/log_ols_digenic_interactions.csv',
            'trigenic': 'glm_models/log_ols_trigenic_interactions.csv'
        },
        'glm_log_link': {
            'digenic': 'glm_log_link/glm_log_link_digenic_interactions.csv',
            'trigenic': 'glm_log_link/glm_log_link_trigenic_interactions.csv'
        }
    }

    # Analyze each model
    for model_name, patterns in models.items():
        print(f"\n{'='*80}")
        print(f"Analyzing {model_name.upper()} model")
        print(f"{'='*80}")

        # Analyze digenic interactions - for all, positive, and negative
        print(f"\n  Loading digenic interactions...")
        digenic_df = normalize_ffa_names(load_latest_csv(patterns['digenic']))

        if digenic_df is not None:
            print(f"  Found {len(digenic_df)} digenic interactions")

            for sign in ['all', 'positive', 'negative']:
                print(f"\n  Analyzing {sign} digenic interactions...")

                digenic_overlap = analyze_graph_overlap(digenic_df, gene_graphs, 'digenic', genome, interaction_sign=sign)

                if len(digenic_overlap) == 0:
                    print(f"    No {sign} interactions found, skipping...")
                    continue

                # Get edge columns
                edge_cols = [col for col in digenic_overlap.columns if col.endswith('_edge')]

                # Calculate enrichment
                print(f"    Calculating enrichment statistics...")
                digenic_enrichment = calculate_enrichment(digenic_overlap, edge_cols)

                # Save results
                sign_suffix = f"_{sign}" if sign != 'all' else ""
                overlap_path = GRAPH_ENRICHMENT_DIR / f"{model_name}_digenic{sign_suffix}_graph_overlap.csv"
                enrichment_path = GRAPH_ENRICHMENT_DIR / f"{model_name}_digenic{sign_suffix}_enrichment.csv"

                digenic_overlap.to_csv(overlap_path, index=False)
                digenic_enrichment.to_csv(enrichment_path, index=False)

                print(f"    Saved {sign} overlap results to: {overlap_path}")
                print(f"    Saved {sign} enrichment results to: {enrichment_path}")

        # Analyze trigenic interactions - for all, positive, and negative
        print(f"\n  Loading trigenic interactions...")
        trigenic_df = normalize_ffa_names(load_latest_csv(patterns['trigenic']))

        if trigenic_df is not None:
            print(f"  Found {len(trigenic_df)} trigenic interactions")

            for sign in ['all', 'positive', 'negative']:
                print(f"\n  Analyzing {sign} trigenic interactions...")

                trigenic_overlap = analyze_graph_overlap(trigenic_df, gene_graphs, 'trigenic', genome, interaction_sign=sign)

                if len(trigenic_overlap) == 0:
                    print(f"    No {sign} interactions found, skipping...")
                    continue

                # Get triangle and connected columns
                triangle_cols = [col for col in trigenic_overlap.columns if col.endswith('_triangle')]
                connected_cols = [col for col in trigenic_overlap.columns if col.endswith('_connected')]

                # Calculate enrichment for triangles
                print(f"    Calculating triangle enrichment...")
                triangle_enrichment = calculate_enrichment(trigenic_overlap, triangle_cols)

                # Calculate enrichment for connectedness
                print(f"    Calculating connectedness enrichment...")
                connected_enrichment = calculate_enrichment(trigenic_overlap, connected_cols)

                # Save results
                sign_suffix = f"_{sign}" if sign != 'all' else ""
                overlap_path = GRAPH_ENRICHMENT_DIR / f"{model_name}_trigenic{sign_suffix}_graph_overlap.csv"
                triangle_path = GRAPH_ENRICHMENT_DIR / f"{model_name}_trigenic{sign_suffix}_triangle_enrichment.csv"
                connected_path = GRAPH_ENRICHMENT_DIR / f"{model_name}_trigenic{sign_suffix}_connected_enrichment.csv"

                trigenic_overlap.to_csv(overlap_path, index=False)
                triangle_enrichment.to_csv(triangle_path, index=False)
                connected_enrichment.to_csv(connected_path, index=False)

                print(f"    Saved {sign} overlap results to: {overlap_path}")
                print(f"    Saved {sign} triangle enrichment to: {triangle_path}")
                print(f"    Saved {sign} connectedness enrichment to: {connected_path}")

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"\nAll results saved to: {GRAPH_ENRICHMENT_DIR}")


if __name__ == "__main__":
    main()
