

import os
import os.path as osp
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv
from torchcell.graph.graph import SCerevisiaeGraph
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.timestamp import timestamp
import matplotlib
import json
import numpy as np
matplotlib.use('Agg')  # Non-interactive backend

# Load environment
load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")

# Results directory
RESULTS_DIR = Path("/Users/michaelvolk/Documents/projects/torchcell/experiments/008-xue-ffa/results")
FFA_REACTIONS_DIR = RESULTS_DIR / "ffa_reactions"
GRAPH_ENRICHMENT_DIR = RESULTS_DIR / "graph_enrichment"

# V21: Updated color scheme - lighter core_gene color for better text readability
# V11: Using colors from torchcell.mplstyle
COLORS = {
    'gene': '#4A9C60',  # Green
    'reaction': '#E6A65D',  # Orange (changed from red)
    'metabolite': '#6D666F',  # Grey
    'core_gene': '#3D796E',  # Teal-green from mplstyle (readable with black text)
    'target_ffa': '#3978B5',  # Blue (changed from dark red)
    'tf_gene': '#7A6DBF',  # Purple for TFs
    'positive_interaction': '#4A9C60',  # Green for positive
    'negative_interaction': '#B73C39',  # Red for negative
    'induced_edge': '#6D666F',  # Grey for baseline connections
}

# TF genes from experiment (for consistent circle ordering)
TF_GENES = ['FKH1', 'GCN5', 'MED4', 'OPI1', 'RFX1', 'RGR1', 'RPD3', 'SPT3', 'YAP6', 'TFC7']

# V13: Define FFA ordering by chain length
# V14: Will reverse this in layout so C14:0 is at top
FFA_ORDER = ['C14:0', 'C16:0', 'C16:1', 'C18:0', 'C18:1']


def extract_all_ffa_types(metabolite_name):
    """
    Extract all FFA types from a metabolite name.

    For complex lipids like "phosphatidate (1-16:0, 2-18:1)",
    this extracts both 16:0 and 18:1, returning them as sorted list.

    Also handles simple fatty acid names like "myristate", "palmitate", etc.

    Returns:
        List of FFA types sorted by chain length (e.g., ['C16:0', 'C18:1'])
    """
    import re

    ffa_types = set()

    # First check for simple fatty acid names and map to FFA types
    name_lower = metabolite_name.lower()
    simple_ffa_map = {
        'myristate': 'C14:0',
        'myristic': 'C14:0',
        'palmitate': 'C16:0',
        'palmitic': 'C16:0',
        'palmitoleate': 'C16:1',
        'palmitoleic': 'C16:1',
        'stearate': 'C18:0',
        'stearic': 'C18:0',
        'oleate': 'C18:1',
        'oleic': 'C18:1'
    }

    # Check for simple fatty acid names
    for ffa_name, ffa_type in simple_ffa_map.items():
        if ffa_name in name_lower:
            ffa_types.add(ffa_type)

    # Also check for numeric patterns like 16:0, 18:1 for complex lipids
    pattern = r'\b(\d+):(\d+)\b'
    matches = re.findall(pattern, metabolite_name)

    for chain_length, saturation in matches:
        ffa_type = f"C{chain_length}:{saturation}"
        ffa_types.add(ffa_type)

    # Sort by chain length then saturation
    # Order: C14:0 < C16:0 < C16:1 < C18:0 < C18:1
    ffa_order = {
        'C14:0': 0, 'C16:0': 1, 'C16:1': 2,
        'C18:0': 3, 'C18:1': 4
    }

    sorted_ffas = sorted(ffa_types, key=lambda x: ffa_order.get(x, 99))
    return sorted_ffas


def load_ffa_network_and_layout():
    """Load the FFA bipartite network and fixed layout from Step 3."""
    print("=" * 80)
    print("LOADING FFA NETWORK AND LAYOUT")
    print("=" * 80)

    network_path = RESULTS_DIR / "ffa_bipartite_network.graphml"
    layout_path = RESULTS_DIR / "ffa_network_layout.json"

    # Load network
    G = nx.read_graphml(network_path)
    print(f"\nLoaded network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Load layout
    with open(layout_path, 'r') as f:
        pos_serialized = json.load(f)

    # Convert back to tuples
    pos = {node: tuple(coords) for node, coords in pos_serialized.items()}
    print(f"Loaded layout for {len(pos)} nodes")

    return G, pos


def improve_pathway_layout_with_interleaving(pos, G, vertical_spread_multiplier=4.0, met_x_shift=0.8, met_to_ffa=None):
    """
    Improve layout with INTERLEAVED metabolites to prevent target FFA label stacking.

    V14: Put grey nodes at END, reverse FFA order so C14:0 is at TOP.
    V13: Better interleaving - 3 grey nodes between each target FFA, ordered by chain length.
    V11: More aggressive metabolite shift (0.8 instead of 0.4) for label space.
    V11: Only respread CORE genes (exclude non-core that aren't displayed).
    V10: Compressed gene vertical spacing (4.0 instead of 6.0) to eliminate gaps.
    V9: Shift metabolites RIGHT (positive x) to give more horizontal space for labels.
    """
    print(f"  Improving pathway layout (v_spread={vertical_spread_multiplier}x, met_x_shift=+{met_x_shift})...")

    # Helper function to handle string/bool conversion
    def is_true(val):
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() in ('true', '1', 'yes')
        return False

    new_pos = pos.copy()

    # V11: Only respread CORE genes (exclude non-core genes that aren't displayed)
    gene_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'gene']
    core_gene_nodes = [n for n in gene_nodes if is_true(G.nodes[n].get('is_core'))]

    if core_gene_nodes:
        gene_positions = [(n, pos[n]) for n in core_gene_nodes if n in pos]
        gene_positions.sort(key=lambda x: x[1][1])
        n_genes = len(gene_positions)
        if n_genes > 1:
            y_positions = np.linspace(-vertical_spread_multiplier, vertical_spread_multiplier, n_genes)
            for (node, old_pos), new_y in zip(gene_positions, y_positions):
                new_pos[node] = (old_pos[0], new_y)
        print(f"    Respread {n_genes} CORE genes equally (excludes non-core)")

    # Spread reactions vertically
    reaction_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'reaction']
    if reaction_nodes:
        rxn_positions = [(n, pos[n]) for n in reaction_nodes if n in pos]
        rxn_positions.sort(key=lambda x: x[1][1])
        n_rxns = len(rxn_positions)
        if n_rxns > 1:
            y_positions = np.linspace(-vertical_spread_multiplier, vertical_spread_multiplier, n_rxns)
            for (node, old_pos), new_y in zip(rxn_positions, y_positions):
                new_pos[node] = (old_pos[0], new_y)
        print(f"    Spread {n_rxns} reactions vertically")

    # V14: BETTER INTERLEAVING with reversed ordered FFAs (C14:0 at TOP)
    metabolite_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'metabolite']
    if metabolite_nodes and met_to_ffa:
        # Separate target FFAs and others
        target_ffas = [n for n in metabolite_nodes if is_true(G.nodes[n].get('is_target_ffa'))]
        other_mets = [n for n in metabolite_nodes if not is_true(G.nodes[n].get('is_target_ffa'))]

        # V13: Sort target FFAs by chain length order
        def get_ffa_order(node):
            ffa_type = met_to_ffa.get(node, 'ZZZ')  # Default to end if not found
            try:
                return FFA_ORDER.index(ffa_type)
            except ValueError:
                return 999  # Put unknown at end

        target_ffas_sorted = sorted(target_ffas, key=get_ffa_order)

        # Sort other metabolites by current y position
        other_positions = sorted([(n, pos[n]) for n in other_mets if n in pos], key=lambda x: x[1][1])
        other_mets_sorted = [n for n, _ in other_positions]

        n_targets = len(target_ffas_sorted)
        n_others = len(other_mets_sorted)
        n_total = n_targets + n_others

        if n_total > 1:
            # V14: Create interleaved pattern with targets first, then grey nodes at END
            # V14: Also REVERSE y positions so C14:0 is at TOP
            all_y_positions = np.linspace(vertical_spread_multiplier, -vertical_spread_multiplier, n_total)  # REVERSED!

            # Build interleaved list - targets with spacing, then remaining grey at end
            interleaved_nodes = []
            other_idx = 0
            spacing = 3  # Number of grey nodes between target FFAs

            for i, target in enumerate(target_ffas_sorted):
                # Add the target FFA
                interleaved_nodes.append(target)

                # Add spacing grey nodes between targets (not after the last one)
                if i < n_targets - 1:
                    for _ in range(spacing):
                        if other_idx < n_others:
                            interleaved_nodes.append(other_mets_sorted[other_idx])
                            other_idx += 1

            # V14: Add any remaining grey nodes at the END
            while other_idx < n_others:
                interleaved_nodes.append(other_mets_sorted[other_idx])
                other_idx += 1

            # Assign positions
            for node, y_pos in zip(interleaved_nodes, all_y_positions):
                old_x = pos[node][0] if node in pos else 0
                new_pos[node] = (old_x + met_x_shift, y_pos)

            print(f"    Interleaved {n_targets} target FFAs (C14:0 at TOP) with {n_others} other metabolites")
            print(f"    Using {spacing} grey nodes between targets, extras at END")

    elif metabolite_nodes:
        # Fallback if no met_to_ffa mapping
        met_positions = [(n, pos[n]) for n in metabolite_nodes if n in pos]
        met_positions.sort(key=lambda x: x[1][1])
        n_total = len(met_positions)
        y_positions = np.linspace(-vertical_spread_multiplier, vertical_spread_multiplier, n_total)
        for (node, old_pos), new_y in zip(met_positions, y_positions):
            new_pos[node] = (old_pos[0] + met_x_shift, new_y)
        print(f"    Spread {n_total} metabolites vertically, shifted RIGHT by +{met_x_shift}")

    return new_pos


def compute_tf_circle_layout(tf_nodes, radius=1.0, center_x=-8.0):
    """
    Position TFs in a fixed circle for consistent comparison across plots.
    V34: Moved to center_x=-6.25 (1.5x previous adjustment) to eliminate overlap with core genes.
    V33: Moved to center_x=-5.5 to avoid overlap with core genes.
    V32: Default radius 2.5, center at -5.0 for better visibility and less overlap.
    """
    print(f"    Computing TF circle layout (radius={radius}, center_x={center_x})...")

    tf_pos = {}
    n_tfs = len(tf_nodes)

    if n_tfs == 0:
        return tf_pos

    # Sort TF nodes alphabetically for consistent positioning
    sorted_tfs = sorted(tf_nodes)

    # Position TFs evenly around circle
    for i, tf in enumerate(sorted_tfs):
        # Start from top and go clockwise
        angle = -np.pi/2 + (2 * np.pi * i / n_tfs)  # Start at top (-90 degrees)
        x = center_x + radius * np.cos(angle)
        y = radius * np.sin(angle)
        tf_pos[tf] = (x, y)

    print(f"      Positioned {len(tf_pos)} TFs in compressed fixed circle")
    return tf_pos


def load_graph_enrichment_data(model='multiplicative', interaction_type='digenic',
                                 sign='positive', topology='edge'):
    """Load graph enrichment data for a specific model and interaction type."""
    print(f"\n  Loading {model} {interaction_type} {sign} {topology} interactions...")

    # Build filename (no timestamp, no wildcard)
    if sign == 'all' or sign == 'both':
        # V15: For 'both', load file without sign suffix (contains ALL interactions)
        filename = f"{model}_{interaction_type}_graph_overlap.csv"
    else:
        filename = f"{model}_{interaction_type}_{sign}_graph_overlap.csv"

    filepath = GRAPH_ENRICHMENT_DIR / filename

    if not filepath.exists():
        print(f"    File not found: {filename}")
        return None

    df = pd.read_csv(filepath)
    print(f"    Loaded {len(df)} interactions from {filename}")

    # V15: Debug - show interaction breakdown
    if 'interaction_score' in df.columns and 'significant_p05' in df.columns:
        sig_df = df[df['significant_p05'] == True]
        pos_count = len(sig_df[sig_df['interaction_score'] > 0])
        neg_count = len(sig_df[sig_df['interaction_score'] < 0])
        print(f"    Significant interactions in file: {pos_count} positive, {neg_count} negative")

    return df


def load_enrichment_significance(model='multiplicative', interaction_type='digenic',
                                   sign='positive', topology='edge'):
    """
    Load enrichment significance data to filter for significantly enriched graphs.

    V24: Fixed to only return True for actual ENRICHMENT (not depletion).
    Requires: p < 0.05 AND sig_with_overlap > 0 AND fold_enrichment > 0
    """
    print(f"\n  Loading enrichment significance for {model} {interaction_type} {sign}...")

    # Build filename for enrichment CSV (no timestamp, no wildcard)
    if topology == 'edge':
        # Digenic
        if sign == 'all' or sign == 'both':
            filename = f"{model}_{interaction_type}_enrichment.csv"
        else:
            filename = f"{model}_{interaction_type}_{sign}_enrichment.csv"
    else:
        # Trigenic with topology
        if sign == 'all' or sign == 'both':
            filename = f"{model}_{interaction_type}_{topology}_enrichment.csv"
        else:
            filename = f"{model}_{interaction_type}_{sign}_{topology}_enrichment.csv"

    filepath = GRAPH_ENRICHMENT_DIR / filename

    if not filepath.exists():
        print(f"    Enrichment file not found: {filename}")
        return {}

    df = pd.read_csv(filepath)
    print(f"    Loaded enrichment from {filename}")

    # V24: Create dict of graph_type -> is_enriched (not just significant)
    # Only mark as enriched if:
    # 1. p < 0.05 (statistically significant)
    # 2. sig_with_overlap > 0 (there are actually interactions to plot)
    # 3. fold_enrichment > 0 (not depletion - fold_enrichment=0 means depletion)
    enrichment_sig = {}
    for _, row in df.iterrows():
        graph_type = row['graph_type']
        p_value = row['p_value']
        sig_with_overlap = row['sig_with_overlap']
        fold_enrichment = row['fold_enrichment']

        # V24: Check for actual enrichment (not depletion)
        is_enriched = (
            p_value < 0.05 and
            sig_with_overlap > 0 and
            fold_enrichment > 0
        )

        enrichment_sig[graph_type] = is_enriched

        # Print detailed info
        if is_enriched:
            print(f"      {graph_type}: ENRICHED (p={p_value:.4f}, n={sig_with_overlap}, fold={fold_enrichment:.2f})")
        elif p_value < 0.05 and sig_with_overlap == 0:
            print(f"      {graph_type}: DEPLETED (p={p_value:.4f}, no significant interactions with topology)")
        elif p_value < 0.05:
            print(f"      {graph_type}: DEPLETED (p={p_value:.4f}, fold={fold_enrichment:.2f})")
        else:
            print(f"      {graph_type}: not significant (p={p_value:.4f})")

    return enrichment_sig


def convert_gene_names_to_systematic(genes, genome):
    """Convert standard gene names to systematic IDs."""
    systematic_genes = []

    for gene in genes:
        if hasattr(genome, 'alias_to_systematic'):
            sys_names = genome.alias_to_systematic.get(gene, [gene])
            if isinstance(sys_names, list) and len(sys_names) > 0:
                systematic_genes.append(sys_names[0])
            else:
                systematic_genes.append(gene)
        else:
            systematic_genes.append(gene)

    return systematic_genes


def add_tf_nodes_to_network(G, pos, genome):
    """Add TF nodes to the FFA network if not already present."""
    print("\n  Adding TF nodes to network...")

    # Convert TF names to systematic IDs
    tf_systematic = convert_gene_names_to_systematic(TF_GENES, genome)

    added_tfs = []
    existing_tfs = []

    for tf_std, tf_sys in zip(TF_GENES, tf_systematic):
        if tf_sys not in G.nodes():
            G.add_node(
                tf_sys,
                node_type='tf_gene',
                bipartite=0,
                is_core=False,
                is_tf=True,
                name=tf_std,
                label=tf_std,
                systematic_name=tf_sys  # V14: Store systematic name
            )
            added_tfs.append(tf_sys)
        else:
            existing_tfs.append(tf_sys)
            G.nodes[tf_sys]['is_tf'] = True
            G.nodes[tf_sys]['systematic_name'] = tf_sys  # V14: Store systematic name

    print(f"    Added {len(added_tfs)} new TF nodes")
    print(f"    {len(existing_tfs)} TFs already in network")

    return list(set(tf_systematic))


def extract_significant_interactions(overlap_df, graph_type, interaction_type='digenic',
                                       topology='edge', genome=None, sign_filter='both'):
    """
    Extract significant interactions that have edges/topology in the specified graph type.

    V17: Return interaction counts for accurate reporting
    V16: More debugging for edge count mismatches
    V15: Added debugging to track missing interactions
    V13: Added sign_filter parameter to support 'both', 'positive', 'negative'
    """
    # Filter for significant interactions (p < 0.05)
    sig_df = overlap_df[overlap_df['significant_p05'] == True].copy()

    if len(sig_df) == 0:
        return [], []

    # Determine the column name for this graph type and topology
    if interaction_type == 'digenic':
        col_name = f'{graph_type}_edge'
    else:  # trigenic
        col_name = f'{graph_type}_{topology}'

    # Filter for interactions with this topology in this graph
    if col_name not in sig_df.columns:
        print(f"    Warning: Column {col_name} not found")
        return [], []

    sig_with_topology = sig_df[sig_df[col_name] == True].copy()

    if len(sig_with_topology) == 0:
        print(f"    No significant interactions found with {topology} in {graph_type}")
        return [], []

    print(f"    Found {len(sig_with_topology)} total interactions with {topology}")

    # Separate by interaction sign
    positive_interactions = []
    negative_interactions = []

    for _, row in sig_with_topology.iterrows():
        gene_set = row['gene_set']
        interaction_score = row['interaction_score']

        # Parse gene set
        genes = gene_set.replace(':', '_').split('_')

        if interaction_score > 0:
            positive_interactions.append(genes)
        else:
            negative_interactions.append(genes)

    print(f"    Before filtering - Positive: {len(positive_interactions)}, Negative: {len(negative_interactions)}")

    # V16: Show examples of interactions for debugging
    if positive_interactions and len(positive_interactions) <= 10:
        print(f"    Positive interactions: {positive_interactions[:5]}")
    if negative_interactions and len(negative_interactions) <= 10:
        print(f"    Negative interactions: {negative_interactions[:5]}")

    # V13: Apply sign filter
    if sign_filter == 'positive':
        negative_interactions = []
    elif sign_filter == 'negative':
        positive_interactions = []
    # if sign_filter == 'both', keep both

    print(f"    After filtering ({sign_filter}) - Positive: {len(positive_interactions)}, Negative: {len(negative_interactions)}")

    return positive_interactions, negative_interactions


def add_tf_to_gene_connections(G, tf_nodes, pathway_gene_nodes, regulatory_graph, tflink_graph, genome):
    """
    Add edges from TFs to pathway genes based on regulatory or tflink graphs.

    V18: Only connect TFs to core pathway genes (not hidden non-core genes).
    """
    print("    Adding TF→gene regulatory connections...")

    # V18: Helper to check if gene is core
    def is_true(val):
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() in ('true', '1', 'yes')
        return False

    # V18: Filter to only core pathway genes
    core_pathway_genes = [g for g in pathway_gene_nodes if is_true(G.nodes[g].get('is_core'))]
    print(f"      Filtering to {len(core_pathway_genes)} core genes (out of {len(pathway_gene_nodes)} total)")

    tf_gene_edges = []

    for graph_name, gene_graph in [('regulatory', regulatory_graph), ('tflink', tflink_graph)]:
        if gene_graph is None:
            continue

        graph = gene_graph.graph

        for tf in tf_nodes:
            for gene in core_pathway_genes:  # V18: Use core_pathway_genes instead of pathway_gene_nodes
                if tf in graph.nodes() and gene in graph.nodes():
                    if graph.has_edge(tf, gene) or graph.has_edge(gene, tf):
                        if not G.has_edge(tf, gene) and not G.has_edge(gene, tf):
                            G.add_edge(tf, gene, edge_type='tf_regulates_gene', source=graph_name)
                            tf_gene_edges.append((tf, gene))

    print(f"      Added {len(tf_gene_edges)} TF→gene regulatory edges")
    return tf_gene_edges


def add_induced_subgraph_edges(G, tf_nodes, gene_graph, genome):
    """Add edges from induced subgraph of TF nodes in the gene_graph."""
    print("    Adding TF-TF baseline edges...")

    graph = gene_graph.graph
    induced_edges = []

    for i, tf1 in enumerate(tf_nodes):
        for tf2 in tf_nodes[i+1:]:
            if tf1 in graph.nodes() and tf2 in graph.nodes():
                if graph.has_edge(tf1, tf2) or graph.has_edge(tf2, tf1):
                    if not G.has_edge(tf1, tf2):
                        G.add_edge(tf1, tf2, edge_type='induced_tf_tf')
                        induced_edges.append((tf1, tf2))

    print(f"      Added {len(induced_edges)} TF-TF baseline edges")
    return induced_edges


def load_ffa_metabolite_mapping():
    """Load FFA metabolite mapping to get FFA types for labeling."""
    mapping_path = FFA_REACTIONS_DIR / "ffa_metabolite_mapping.csv"
    df = pd.read_csv(mapping_path)

    met_to_ffa = {}
    for _, row in df.iterrows():
        met_id = row['metabolite_id']
        ffa_type = row['ffa_type']
        met_to_ffa[met_id] = ffa_type

    return met_to_ffa


def create_interaction_edges_with_deduplication(positive_interactions, negative_interactions, G):
    """
    Create interaction edges with proper deduplication and debugging.

    V17: Returns both interaction counts and edge lists for accurate reporting.
    V16: More detailed debugging for edge count mismatches.
    """
    positive_edges = set()
    negative_edges = set()

    # V17: Track original interaction counts
    n_positive_interactions = len(positive_interactions)
    n_negative_interactions = len(negative_interactions)

    # V15: Debug - track original interaction counts
    print(f"    Creating edges from {n_positive_interactions} positive and {n_negative_interactions} negative interactions")

    # Process positive interactions
    for i, genes in enumerate(positive_interactions):
        if len(genes) >= 2:
            if len(genes) == 2:
                # Digenic
                if genes[0] in G.nodes() and genes[1] in G.nodes():
                    edge = tuple(sorted([genes[0], genes[1]]))
                    positive_edges.add(edge)
            elif len(genes) == 3:
                # Trigenic - add all pairwise edges
                edges_from_this = []
                for j in range(len(genes)):
                    for k in range(j+1, len(genes)):
                        if genes[j] in G.nodes() and genes[k] in G.nodes():
                            edge = tuple(sorted([genes[j], genes[k]]))
                            edges_from_this.append(edge)
                            positive_edges.add(edge)
                # V16: Debug trigenic edge creation
                if len(edges_from_this) < 3:
                    print(f"      Trigenic interaction {i+1} ({genes}) created only {len(edges_from_this)} edges: {edges_from_this}")

    # Process negative interactions
    for i, genes in enumerate(negative_interactions):
        if len(genes) >= 2:
            if len(genes) == 2:
                # Digenic
                if genes[0] in G.nodes() and genes[1] in G.nodes():
                    edge = tuple(sorted([genes[0], genes[1]]))
                    negative_edges.add(edge)
            elif len(genes) == 3:
                # Trigenic - add all pairwise edges
                edges_from_this = []
                for j in range(len(genes)):
                    for k in range(j+1, len(genes)):
                        if genes[j] in G.nodes() and genes[k] in G.nodes():
                            edge = tuple(sorted([genes[j], genes[k]]))
                            edges_from_this.append(edge)
                            negative_edges.add(edge)
                # V16: Debug trigenic edge creation
                if len(edges_from_this) < 3:
                    print(f"      Trigenic interaction {i+1} ({genes}) created only {len(edges_from_this)} edges: {edges_from_this}")

    # Convert back to list of tuples
    positive_edges = list(positive_edges)
    negative_edges = list(negative_edges)

    print(f"    Edge counts after deduplication: {len(positive_edges)} positive, {len(negative_edges)} negative")

    # V17: Report when there's a mismatch between interactions and edges
    if n_positive_interactions > len(positive_edges) and n_positive_interactions > 0:
        print(f"    NOTE: {n_positive_interactions} positive interactions → {len(positive_edges)} unique edges (trigenic overlap)")
    if n_negative_interactions > len(negative_edges) and n_negative_interactions > 0:
        print(f"    NOTE: {n_negative_interactions} negative interactions → {len(negative_edges)} unique edges (trigenic overlap)")

    # V17: Return interaction counts along with edges
    return positive_edges, negative_edges, n_positive_interactions, n_negative_interactions


def create_multigraph_overlay(G_base, pos_base, tf_nodes, positive_interactions, negative_interactions,
                               induced_edges, tf_gene_edges, graph_type, graph_type_name, tf_pos, met_to_ffa,
                               interaction_type='digenic', sign='both', topology='edge', genome=None, batch_suffix='', filter_enrichment=False, model='multiplicative'):
    """
    Create publication-quality visualization of FFA network with TF interaction overlays.

    V26: Added graph_type parameter to conditionally show arrows (only regulatory/tflink are directed).
    V23: Added model parameter to display model type in title.
    V22: Added filter_enrichment parameter to display filtering mode in title.
    V19: Added batch_suffix parameter to distinguish batches (_all vs _enriched).
    """
    print(f"\n  Creating visualization for {graph_type_name}...")

    # Create a copy
    G = G_base.copy()

    # V14: Improve layout with better interleaving and reversed ordered FFAs
    # V32: Increased vertical spread from 4.0 to 7.0 for better readability
    pos = improve_pathway_layout_with_interleaving(pos_base, G, vertical_spread_multiplier=7.0,
                                                   met_x_shift=0.8, met_to_ffa=met_to_ffa)

    # Merge TF positions
    pos.update(tf_pos)

    # Helper function
    def is_true(val):
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() in ('true', '1', 'yes')
        return False

    # Separate node types
    gene_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'gene' and n not in tf_nodes]
    reaction_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'reaction']
    metabolite_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'metabolite']
    tf_nodes_in_graph = [n for n in tf_nodes if n in G.nodes()]

    core_genes = [n for n in gene_nodes if is_true(G.nodes[n].get('is_core'))]
    other_genes = [n for n in gene_nodes if not is_true(G.nodes[n].get('is_core'))]
    target_ffas = [n for n in metabolite_nodes if is_true(G.nodes[n].get('is_target_ffa'))]
    other_metabolites = [n for n in metabolite_nodes if not is_true(G.nodes[n].get('is_target_ffa'))]

    # V18: Create set of visible nodes (nodes we're actually drawing)
    visible_nodes = set(core_genes + reaction_nodes + metabolite_nodes + tf_nodes_in_graph)

    # PUBLICATION SIZE: 14×18 inches (fits on one page)
    fig, ax = plt.subplots(figsize=(14, 18))

    # Separate edge types
    gene_rxn_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'catalyzes']
    met_rxn_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'consumed_by']
    rxn_met_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'produces']

    # V18: Filter metabolic edges to only include edges between visible nodes
    gene_rxn_edges = [(u, v) for u, v in gene_rxn_edges if u in visible_nodes and v in visible_nodes]
    met_rxn_edges = [(u, v) for u, v in met_rxn_edges if u in visible_nodes and v in visible_nodes]
    rxn_met_edges = [(u, v) for u, v in rxn_met_edges if u in visible_nodes and v in visible_nodes]

    print(f"    Filtered edges: {len(gene_rxn_edges)} gene-rxn, {len(met_rxn_edges)} met-rxn, {len(rxn_met_edges)} rxn-met")

    # Draw metabolic edges (all grey/black)
    if gene_rxn_edges:
        nx.draw_networkx_edges(G, pos, edgelist=gene_rxn_edges, edge_color='#404040',
                              width=0.8, alpha=0.3, arrows=False, ax=ax)

    if met_rxn_edges:
        nx.draw_networkx_edges(G, pos, edgelist=met_rxn_edges, edge_color='#404040',
                              width=0.6, alpha=0.25, arrows=True, arrowsize=8,
                              arrowstyle='->', ax=ax)

    if rxn_met_edges:
        nx.draw_networkx_edges(G, pos, edgelist=rxn_met_edges, edge_color='#404040',
                              width=0.6, alpha=0.25, arrows=True, arrowsize=8,
                              arrowstyle='->', ax=ax)

    # V18: Filter TF→gene regulatory connections to only visible nodes
    tf_gene_edges_filtered = [(u, v) for u, v in tf_gene_edges if u in visible_nodes and v in visible_nodes]
    print(f"    Filtered TF→gene edges: {len(tf_gene_edges_filtered)} (was {len(tf_gene_edges)})")

    # V26: Determine if this graph type should show directed edges (arrows)
    # Only regulatory and tflink should be directed
    show_arrows = graph_type in ['regulatory', 'tflink']
    print(f"    Graph type '{graph_type}': arrows={'enabled' if show_arrows else 'disabled'}")

    # V26: Draw TF→gene regulatory connections - always directed (arrows=True) since they come from regulatory/tflink
    if tf_gene_edges_filtered:
        nx.draw_networkx_edges(G, pos, edgelist=tf_gene_edges_filtered, edge_color='#606060',
                              width=1.2, alpha=0.4, arrows=True, arrowsize=10, style='dotted', ax=ax)

    # V26: Draw TF-TF baseline edges - only show arrows for regulatory/tflink graphs
    if induced_edges:
        nx.draw_networkx_edges(G, pos, edgelist=induced_edges, edge_color='#808080',
                              width=1.2, alpha=0.5, style='dashed', arrows=show_arrows, arrowsize=10, ax=ax)

    # V25: Create interaction edges and get both edge counts and interaction counts
    positive_edges, negative_edges, n_pos_interactions, n_neg_interactions = create_interaction_edges_with_deduplication(
        positive_interactions, negative_interactions, G
    )

    # V25: Always draw both - negative first (extra thick) then positive (thinner) so they layer correctly
    if negative_edges:
        nx.draw_networkx_edges(G, pos, edgelist=negative_edges, edge_color=COLORS['negative_interaction'],
                              width=6.0, alpha=0.95, arrows=False, ax=ax)
    if positive_edges:
        nx.draw_networkx_edges(G, pos, edgelist=positive_edges, edge_color=COLORS['positive_interaction'],
                              width=2.5, alpha=0.95, arrows=False, ax=ax)

    # Draw nodes (adjusted sizes for publication)
    # V11: Make all metabolite nodes same size (100 for both)
    if other_metabolites:
        nx.draw_networkx_nodes(G, pos, nodelist=other_metabolites, node_color=COLORS['metabolite'],
                              node_size=100, ax=ax, alpha=0.4)

    if target_ffas:
        nx.draw_networkx_nodes(G, pos, nodelist=target_ffas, node_color=COLORS['target_ffa'],
                              node_size=100, ax=ax, alpha=0.85)

    # V11: Updated color for reaction nodes
    if reaction_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=reaction_nodes, node_color=COLORS['reaction'],
                              node_size=80, node_shape='s', ax=ax, alpha=0.6)

    # V6: HIDE non-core pathway genes (other_genes) to reduce clutter
    # These were the light green dots that didn't connect to FFA metabolites
    # if other_genes:
    #     nx.draw_networkx_nodes(G, pos, nodelist=other_genes, node_color=COLORS['gene'],
    #                           node_size=150, ax=ax, alpha=0.75)

    if core_genes:
        nx.draw_networkx_nodes(G, pos, nodelist=core_genes, node_color=COLORS['core_gene'],
                              node_size=200, ax=ax, alpha=0.9)

    # LARGER TF nodes
    if tf_nodes_in_graph:
        nx.draw_networkx_nodes(G, pos, nodelist=tf_nodes_in_graph, node_color=COLORS['tf_gene'],
                              node_size=400, ax=ax, alpha=0.95)

    # V16: BIGGER TF labels (12 instead of 10)
    tf_labels = {}
    for node in tf_nodes_in_graph:
        short_name = G.nodes[node].get('label', node)
        systematic_name = G.nodes[node].get('systematic_name', node)
        # Format: "systematic_name\nshort_name"
        if systematic_name != short_name:
            tf_labels[node] = f"{systematic_name}\n{short_name}"
        else:
            tf_labels[node] = short_name

    if tf_labels:
        nx.draw_networkx_labels(G, pos, labels=tf_labels, font_size=12,
                               font_weight='bold', ax=ax)  # V16: 12 instead of 10

    # V16: BIGGER gene names for core pathway genes (12 instead of 10)
    gene_labels = {}
    for node in core_genes:
        gene_name = G.nodes[node].get('name', node)
        if gene_name and gene_name != node:
            gene_labels[node] = gene_name
        else:
            # If no name, use the node ID (systematic name)
            gene_labels[node] = node

    if gene_labels:
        nx.draw_networkx_labels(G, pos, labels=gene_labels, font_size=12,
                               font_weight='bold', ax=ax)  # V16: 12 instead of 10

    # V30: Updated metabolite labels - include compartment for ALL metabolites
    # Format: "[FFA_type] metabolite [compartment]" for target FFAs
    #         "metabolite [compartment]" for other metabolites
    met_labels = {}
    met_label_pos = {}

    # Compartment code mapping
    comp_map = {
        'c': 'cytoplasm',
        'erm': 'ER membrane',
        'lp': 'lipid particle',
        'e': 'extracellular',
        'm': 'mitochondria',
        'n': 'nucleus',
        'p': 'peroxisome',
        'v': 'vacuole',
        'g': 'Golgi'
    }

    # V30: Process only target FFAs (blue nodes) for labeling
    for node in target_ffas:
        met_name = G.nodes[node].get('name', node)

        # V30: Get compartment from node attributes (not from node ID)
        compartment = G.nodes[node].get('compartment', '')

        # Extract ALL FFA types from the metabolite name (for complex lipids)
        ffa_types = extract_all_ffa_types(met_name)

        # Build label: [FFA_type1, FFA_type2] metabolite_name [compartment]
        comp_name = comp_map.get(compartment, compartment) if compartment else ''

        if ffa_types and comp_name:
            ffa_str = ', '.join(ffa_types)
            met_labels[node] = f"[{ffa_str}] {met_name} [{comp_name}]"
        elif ffa_types:
            ffa_str = ', '.join(ffa_types)
            met_labels[node] = f"[{ffa_str}] {met_name}"
        elif comp_name:
            met_labels[node] = f"{met_name} [{comp_name}]"
        else:
            met_labels[node] = met_name

        # V12: Keep label at same position as node OR shift slightly LEFT
        node_pos = pos[node]
        met_label_pos[node] = (node_pos[0] - 0.05, node_pos[1])  # Shift label 0.05 units LEFT

    if met_labels:
        # V12: Use RIGHT alignment so text extends to the LEFT of the position
        # V16: Even larger font size (13 instead of 11)
        nx.draw_networkx_labels(G, met_label_pos, labels=met_labels, font_size=13,
                               font_weight='bold', horizontalalignment='right', ax=ax)

    # V25: Updated title - removed sign since we always show both positive + negative
    # Determine interaction type string
    if interaction_type == 'digenic':
        interaction_type_str = 'Digenic'
    else:  # trigenic
        interaction_type_str = 'Trigenic'

    # Add topology if present
    topology_str = ''
    if topology and topology != 'edge':
        topology_str = f" ({topology})"

    # Build counts string with pipe separator
    counts_parts = []

    if len(positive_edges) > 0:
        if n_pos_interactions != len(positive_edges):
            count_str = f"{len(positive_edges)} positive edges ({n_pos_interactions} interactions)"
        else:
            count_str = f"{len(positive_edges)} positive edges"
        counts_parts.append(count_str)

    if len(negative_edges) > 0:
        if n_neg_interactions != len(negative_edges):
            count_str = f"{len(negative_edges)} negative edges ({n_neg_interactions} interactions)"
        else:
            count_str = f"{len(negative_edges)} negative edges"
        counts_parts.append(count_str)

    counts_str = " | ".join(counts_parts) if counts_parts else "no edges"

    # V25: Build 4-line title with model type and filtering mode (removed sign)
    # Line 1: FFA Metabolic Network + TF Epistatic Interactions (Model Type)
    # Line 2: Filtering mode (Significant vs Significant and Enriched)
    # Line 3: Graph Name — Interaction Type (no sign since we always show both)
    # Line 4: Edge/interaction counts
    model_str = model.replace('_', ' ').title()
    title_line1 = f"FFA Metabolic Network + TF Epistatic Interactions ({model_str} Model)"
    title_line2 = "Significant and Enriched Interactions" if filter_enrichment else "Significant Interactions"
    title_line3 = f"{graph_type_name} — {interaction_type_str}{topology_str}"
    title_line4 = counts_str

    ax.set_title(f"{title_line1}\n{title_line2}\n{title_line3}\n{title_line4}",
                fontsize=14, pad=15, fontweight='bold')

    # V25: Simplified legend - always show both positive and negative since sign is always 'both'
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['tf_gene'],
               markersize=12, label='Transcription Factors'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['core_gene'],
               markersize=10, label='Core FFA Pathway Genes'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['reaction'],
               markersize=8, label='Reactions'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['target_ffa'],
               markersize=10, label='Target FFAs'),
        Line2D([0], [0], color=COLORS['positive_interaction'], linewidth=2.5,
               label=f'Positive Epistatic Interactions ({n_pos_interactions})'),
        Line2D([0], [0], color=COLORS['negative_interaction'], linewidth=6,
               label=f'Negative Epistatic Interactions ({n_neg_interactions})'),
        Line2D([0], [0], color='#808080', linewidth=2, linestyle='--',
               label='TF-TF Baseline'),
        Line2D([0], [0], color='#606060', linewidth=2, linestyle=':',
               label='TF→Gene Regulation'),
        Line2D([0], [0], color='#404040', linewidth=1,
               label='Metabolic Reactions'),
    ]

    # V14: LARGER LEGEND TEXT (fontsize 14)
    ax.legend(handles=legend_elements, loc='upper left', fontsize=14, framealpha=0.95)

    # V30: Set equal aspect ratio to keep TF circle perfectly circular (not oval)
    ax.set_aspect('equal', adjustable='datalim')
    ax.axis('off')

    plt.tight_layout()

    # V25: Save to nested directory structure with model prepended to filename
    safe_name = graph_type_name.replace(' ', '_').replace('/', '_').replace('.', '_')
    safe_topology = topology if topology != 'edge' else ''
    if safe_topology:
        safe_name += f"_{safe_topology}"

    # Create nested directory: notes/assets/images/008-xue-ffa/ffa_multigraph_overlays/<model_name>/all_ffa
    model_dir = osp.join(ASSET_IMAGES_DIR, "008-xue-ffa", "ffa_multigraph_overlays", model, "all_ffa")
    os.makedirs(model_dir, exist_ok=True)

    # Prepend model to filename and add batch suffix
    filename = f"{model}_ffa_multigraph_{safe_name}{batch_suffix}.png"
    output_path = osp.join(model_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"    Saved to: {output_path}")
    plt.close()

    return output_path


def create_ffa_multigraph_overlays_comprehensive(model='multiplicative', filter_enrichment=False):
    """
    Create comprehensive FFA bipartite network visualizations with multigraph overlays.

    V25: Only creates 'both' (positive + negative) plots. Nested directory structure with model prepended to filenames.
    V19: Always creates visualizations (even with 0 interactions) and adds batch suffix.

    Args:
        model: Model type ('multiplicative', 'additive', 'log_ols', 'glm_log_link')
        filter_enrichment: If True, only create visualizations for significantly enriched graphs (green bars)
    """
    print("=" * 80)
    print(f"FFA MULTIGRAPH OVERLAYS COMPREHENSIVE (V25): {model.upper()}")
    if filter_enrichment:
        print("FILTERING FOR SIGNIFICANTLY ENRICHED GRAPHS ONLY")
    print("=" * 80)

    # V26: Updated batch suffix naming - _unenriched instead of _all
    batch_suffix = '_enriched' if filter_enrichment else '_unenriched'

    # Load FFA network and layout
    G_base, pos = load_ffa_network_and_layout()

    # Initialize genome and gene graphs
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

    # Add TF nodes to network
    tf_systematic = add_tf_nodes_to_network(G_base, pos, genome)

    # Load FFA metabolite mapping for labeling
    print("\nLoading FFA metabolite mapping...")
    met_to_ffa = load_ffa_metabolite_mapping()

    # Get pathway gene nodes (excluding TFs)
    pathway_gene_nodes = [n for n in G_base.nodes()
                         if G_base.nodes[n].get('node_type') == 'gene'
                         and n not in tf_systematic]
    print(f"  Found {len(pathway_gene_nodes)} pathway genes")

    # Compute COMPRESSED TF circle layout (same for all visualizations)
    # V34: Moved further left to -6.25 (1.5x previous adjustment) to eliminate overlap
    # V33: Moved slightly left to -5.5 to avoid overlap with core genes
    # V32: Increased radius to 2.5 and moved right to -5.0 for better visibility
    print("\nComputing compressed TF circle layout...")
    tf_pos = compute_tf_circle_layout(tf_systematic, radius=2.5, center_x=-6.25)

    # Graph types to visualize
    graph_configs = [
        ('physical', graph.G_physical, 'Physical Interactions'),
        ('regulatory', graph.G_regulatory, 'Regulatory Interactions'),
        ('genetic', graph.G_genetic, 'Genetic Interactions'),
        ('tflink', graph.G_tflink, 'TFLink'),
        ('string12_0_coexpression', graph.G_string12_0_coexpression, 'STRING 12.0 Coexpression'),
        ('string12_0_experimental', graph.G_string12_0_experimental, 'STRING 12.0 Experimental'),
        ('string12_0_database', graph.G_string12_0_database, 'STRING 12.0 Database'),
    ]

    # V25: Only create 'both' (positive + negative) plots
    interaction_configs = [
        ('digenic', 'both', 'edge'),
        ('trigenic', 'both', 'triangle'),
        ('trigenic', 'both', 'connected'),
    ]

    output_paths = []

    for interaction_type, sign, topology in interaction_configs:
        print(f"\n{'='*80}")
        print(f"Processing {interaction_type.upper()} {sign.upper()} {topology.upper()}")
        print(f"{'='*80}")

        # V15: FIX - Load ALL data for 'both' sign
        overlap_df = load_graph_enrichment_data(model, interaction_type, sign, topology)

        if overlap_df is None:
            print(f"  No enrichment data found for {model} {interaction_type} {sign} {topology}")
            continue

        # Load enrichment significance if filtering
        enrichment_sig = {}
        if filter_enrichment:
            if sign == 'both':
                # For 'both', we use positive enrichment data (or could use a combined approach)
                enrichment_sig = load_enrichment_significance(model, interaction_type, 'positive', topology)
            else:
                enrichment_sig = load_enrichment_significance(model, interaction_type, sign, topology)

        for graph_type, gene_graph, display_name in graph_configs:
            print(f"\n{'-'*80}")
            print(f"  {display_name}")
            print(f"{'-'*80}")

            # Check if this graph type is significantly enriched (if filtering)
            if filter_enrichment:
                # Map display name to graph_type in enrichment data
                if interaction_type == 'digenic':
                    enrich_key = f"{graph_type}_edge"
                else:
                    enrich_key = f"{graph_type}_{topology}"

                if enrich_key not in enrichment_sig:
                    print(f"    No enrichment data for {enrich_key}, skipping...")
                    continue

                if not enrichment_sig[enrich_key]:
                    print(f"    Not significantly enriched (p >= 0.05), skipping...")
                    continue

                print(f"    SIGNIFICANTLY ENRICHED (p < 0.05) - creating visualization")

            # Create a fresh copy of the base network
            G = G_base.copy()

            # Extract significant interactions
            # V15: Pass correct sign_filter for 'both'
            positive_interactions, negative_interactions = extract_significant_interactions(
                overlap_df, graph_type, interaction_type, topology, genome, sign_filter=sign
            )

            # V19: REMOVED zero-interaction skip - always create visualization (shows base network)
            if len(positive_interactions) == 0 and len(negative_interactions) == 0:
                print(f"    No significant interactions found - will show base network only")

            # Convert gene names to systematic IDs
            positive_interactions_sys = []
            for genes in positive_interactions:
                sys_genes = convert_gene_names_to_systematic(genes, genome)
                positive_interactions_sys.append(sys_genes)

            negative_interactions_sys = []
            for genes in negative_interactions:
                sys_genes = convert_gene_names_to_systematic(genes, genome)
                negative_interactions_sys.append(sys_genes)

            print(f"    Converted to systematic IDs:")
            print(f"      Positive: {len(positive_interactions_sys)} interactions")
            print(f"      Negative: {len(negative_interactions_sys)} interactions")

            # Add TF→gene regulatory connections (V18: now filters to core genes only)
            tf_gene_edges = add_tf_to_gene_connections(G, tf_systematic, pathway_gene_nodes,
                                                        graph.G_regulatory, graph.G_tflink, genome)

            # Add TF-TF baseline edges from the current gene_graph
            # NOTE: These edges SHOULD vary per graph type - they show TF relationships in that specific network
            induced_edges = add_induced_subgraph_edges(G, tf_systematic, gene_graph, genome)

            # V26: Create visualization with graph_type, batch_suffix, filter_enrichment flag, and model type
            output_path = create_multigraph_overlay(
                G, pos, tf_systematic,
                positive_interactions_sys, negative_interactions_sys,
                induced_edges, tf_gene_edges, graph_type, display_name, tf_pos, met_to_ffa,
                interaction_type, sign, topology, genome, batch_suffix, filter_enrichment, model
            )
            output_paths.append(output_path)

    print("\n" + "=" * 80)
    print("COMPREHENSIVE MULTIGRAPH OVERLAYS COMPLETE (V25)!")
    print("=" * 80)
    print(f"\nCreated {len(output_paths)} visualizations")
    print(f"Saved to: notes/assets/images/008-xue-ffa/ffa_multigraph_overlays/{model}/all_ffa/")
    print("\nNEW in V25:")
    print("  - SIMPLIFIED: Only creates 'both' (positive + negative) plots")
    print("  - REMOVED sign from titles (always showing both is now implicit)")
    print("  - ADDED STRING 12.0 Database to graph types")
    print("  - NESTED directory structure: /008-xue-ffa/ffa_multigraph_overlays/<model>/all_ffa/")
    print(f"  - PREPENDED model name to filenames: {model}_ffa_multigraph_...")
    print("\nV19-V24 improvements retained:")
    print("  - Model type in title and filtering mode (Significant vs Significant and Enriched)")
    print("  - Always creates visualization (shows base network even with 0 interactions)")
    print("  - Batch suffix in filenames: _all vs _enriched")
    print("  - Accurate interaction count reporting in legend")
    print("  - Thicker negative lines (6.0 width) for visibility")
    print("  - Larger text (TF: 12pt, genes: 12pt, metabolites: 13pt)")
    if filter_enrichment:
        print("  - ENRICHMENT FILTERING: Only significantly enriched graphs visualized")
    print("\nNOTE: TF-TF baseline edges (dashed) vary per graph type - this is intentional!")
    print("      Each plot shows TF relationships in that specific network (STRING Exp vs Coexp vs Database, etc.)")

    return output_paths


if __name__ == "__main__":
    # Loop through all four models
    for model in ['multiplicative', 'additive', 'log_ols', 'glm_log_link']:
        print("\n" + "="*80)
        print(f"PROCESSING {model.upper()} MODEL")
        print("="*80)

        # Set filter_enrichment=False for all interactions (original)
        print("\n" + "="*80)
        print(f"CREATING {model.upper()} VISUALIZATIONS WITHOUT ENRICHMENT FILTERING")
        print("="*80)
        output_paths_all = create_ffa_multigraph_overlays_comprehensive(
            model=model,
            filter_enrichment=False
        )

        # Set filter_enrichment=True for only significantly enriched interactions (stringent)
        print("\n" + "="*80)
        print(f"CREATING {model.upper()} VISUALIZATIONS WITH ENRICHMENT FILTERING (STRINGENT)")
        print("="*80)
        output_paths_enriched = create_ffa_multigraph_overlays_comprehensive(
            model=model,
            filter_enrichment=True
        )

        print("\n" + "="*80)
        print(f"{model.upper()} MODEL DONE!")
        print("="*80)
        print(f"Created {len(output_paths_all)} visualizations (all significant interactions)")
        print(f"Created {len(output_paths_enriched)} visualizations (enriched only)")

    print("\n" + "="*80)
    print("ALL MODELS COMPLETE!")
    print("="*80)
