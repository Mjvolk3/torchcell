# experiments/008-xue-ffa/scripts/create_ffa_bipartite_network.py
# [[experiments.008-xue-ffa.scripts.create_ffa_bipartite_network]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/008-xue-ffa/scripts/create_ffa_bipartite_network
# Test file: experiments/008-xue-ffa/scripts/test_create_ffa_bipartite_network.py
# V2: Updated colors to match multigraph overlays - lighter core_gene, grey edges, publication quality

import os
import os.path as osp
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv
from torchcell.metabolism.yeast_GEM import YeastGEM
from torchcell.timestamp import timestamp
import matplotlib
import json
matplotlib.use('Agg')  # Non-interactive backend

# Load environment
load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")

# Results directory
RESULTS_DIR = Path("/Users/michaelvolk/Documents/projects/torchcell/experiments/008-xue-ffa/results")
FFA_REACTIONS_DIR = RESULTS_DIR / "ffa_reactions"

# Color scheme aligned with multigraph overlays (v19)
# Uses colors from torchcell.mplstyle for consistency
COLORS = {
    'gene': '#4A9C60',  # Green (light enough for black text)
    'reaction': '#E6A65D',  # Orange (changed from red to match multigraph)
    'metabolite': '#6D666F',  # Grey
    'core_gene': '#3D796E',  # Teal-green from mplstyle (readable with black text)
    'target_ffa': '#3978B5',  # Blue (changed from dark red to match multigraph)
}


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


def load_ffa_data():
    """Load the FFA reactions, genes, and metabolites from Step 1."""
    print("=" * 80)
    print("LOADING FFA DATA")
    print("=" * 80)

    reactions_df = pd.read_csv(FFA_REACTIONS_DIR / "ffa_reactions_list.csv")
    genes_df = pd.read_csv(FFA_REACTIONS_DIR / "ffa_genes_list.csv")
    metabolites_df = pd.read_csv(FFA_REACTIONS_DIR / "ffa_metabolites_list.csv")
    ffa_mapping_df = pd.read_csv(FFA_REACTIONS_DIR / "ffa_metabolite_mapping.csv")

    print(f"\nLoaded:")
    print(f"  {len(reactions_df)} reactions")
    print(f"  {len(genes_df)} genes ({sum(genes_df['is_core_pathway'])} core)")
    print(f"  {len(metabolites_df)} metabolites ({sum(metabolites_df['is_target_ffa'])} target FFAs)")

    return reactions_df, genes_df, metabolites_df, ffa_mapping_df


def extract_ffa_subnetwork(model, reactions_df, genes_df, metabolites_df):
    """
    Extract FFA subnetwork from the full Yeast GEM bipartite graph.

    Creates a bipartite graph with:
    - Gene nodes (systematic IDs from genes_df)
    - Reaction nodes (reaction IDs from reactions_df)
    - Metabolite nodes (metabolite IDs from metabolites_df)

    Edges:
    - gene → reaction (gene catalyzes reaction)
    - reaction → metabolite (metabolite is product)
    - metabolite → reaction (metabolite is reactant)
    """
    print("\n" + "=" * 80)
    print("EXTRACTING FFA SUBNETWORK FROM BIPARTITE GRAPH")
    print("=" * 80)

    # Create bipartite graph
    G = nx.DiGraph()

    # Track node types for bipartite structure
    gene_nodes = set()
    reaction_nodes = set()
    metabolite_nodes = set()

    # Add gene nodes
    print("\nAdding gene nodes...")
    for _, row in genes_df.iterrows():
        gene_id = row['gene_id']
        G.add_node(
            gene_id,
            node_type='gene',
            bipartite=0,
            is_core=row['is_core_pathway'],
            name=row['gene_name'],
            label=row['standard_name']
        )
        gene_nodes.add(gene_id)

    # Add reaction nodes and gene→reaction edges
    print("Adding reaction nodes and gene→reaction edges...")
    for _, row in reactions_df.iterrows():
        rxn_id = row['reaction_id']
        G.add_node(
            rxn_id,
            node_type='reaction',
            bipartite=1,
            equation=row['equation'],
            subsystem=row['subsystem'],
            reversible=row['reversible']
        )
        reaction_nodes.add(rxn_id)

        # Add edges from genes to reactions
        if pd.notna(row['gene_ids']) and row['gene_ids'] != 'No genes':
            gene_ids = row['gene_ids'].split(', ')
            for gene_id in gene_ids:
                if gene_id in gene_nodes:
                    G.add_edge(gene_id, rxn_id, edge_type='catalyzes')

    # Add metabolite nodes and reaction↔metabolite edges
    print("Adding metabolite nodes and reaction↔metabolite edges...")
    for _, row in metabolites_df.iterrows():
        met_id = row['metabolite_id']
        G.add_node(
            met_id,
            node_type='metabolite',
            bipartite=2,
            name=row['metabolite_name'],
            formula=row['formula'],
            compartment=row['compartment'],
            is_target_ffa=row['is_target_ffa']
        )
        metabolite_nodes.add(met_id)

    # Add edges between reactions and metabolites
    print("Connecting reactions to metabolites...")
    for rxn_id in reaction_nodes:
        rxn = model.reactions.get_by_id(rxn_id)
        for met, coef in rxn.metabolites.items():
            if met.id in metabolite_nodes:
                if coef < 0:  # Reactant
                    G.add_edge(met.id, rxn_id, edge_type='consumed_by', stoich=abs(coef))
                else:  # Product
                    G.add_edge(rxn_id, met.id, edge_type='produces', stoich=coef)

    print(f"\nFFA Subnetwork Statistics:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"    Genes: {len(gene_nodes)}")
    print(f"    Reactions: {len(reaction_nodes)}")
    print(f"    Metabolites: {len(metabolite_nodes)}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"    Gene→Reaction: {sum(1 for u, v, d in G.edges(data=True) if d.get('edge_type') == 'catalyzes')}")
    print(f"    Metabolite→Reaction: {sum(1 for u, v, d in G.edges(data=True) if d.get('edge_type') == 'consumed_by')}")
    print(f"    Reaction→Metabolite: {sum(1 for u, v, d in G.edges(data=True) if d.get('edge_type') == 'produces')}")

    return G, gene_nodes, reaction_nodes, metabolite_nodes


def compute_layout(G, gene_nodes, reaction_nodes, metabolite_nodes):
    """
    Compute hierarchical layout for bipartite graph with better metabolite spacing.
    Layout: genes (left) → reactions (center) → metabolites (right)
    """
    print("\n" + "=" * 80)
    print("COMPUTING GRAPH LAYOUT")
    print("=" * 80)

    pos = {}

    # Filter to only core genes for visualization
    core_genes = [n for n in gene_nodes if G.nodes[n].get('is_core')]

    # Sort nodes for better visualization
    genes_sorted = sorted(core_genes)  # Only core genes
    reactions_sorted = sorted(reaction_nodes)

    # Sort metabolites by FFA type and compartment
    def get_ffa_type(met_id):
        # Priority order for FFA types (C14:0 first)
        ffa_order = {'C14:0': 0, 'C16:0': 1, 'C16:1': 2, 'C18:0': 3, 'C18:1': 4}
        met_name = G.nodes[met_id].get('name', '')
        for ffa_type in ffa_order:
            if any(name in met_name.lower() for name in ['myristate', 'palmitate', 'palmitoleate', 'stearate', 'oleate']):
                if 'myristate' in met_name.lower():
                    return 0
                elif 'palmitate' in met_name.lower() and 'palmitoleate' not in met_name.lower():
                    return 1
                elif 'palmitoleate' in met_name.lower():
                    return 2
                elif 'stearate' in met_name.lower():
                    return 3
                elif 'oleate' in met_name.lower():
                    return 4
        return 99  # Other metabolites at the end

    # Separate target FFAs and other metabolites
    target_ffas = [n for n in metabolite_nodes if G.nodes[n].get('is_target_ffa')]
    other_metabolites = [n for n in metabolite_nodes if not G.nodes[n].get('is_target_ffa')]

    # Sort target FFAs by type and compartment
    target_ffas_sorted = sorted(target_ffas, key=lambda x: (get_ffa_type(x), G.nodes[x].get('compartment', '')))
    other_metabolites_sorted = sorted(other_metabolites)

    # Improved vertical spacing
    vertical_span = 8.0  # Larger span for metabolites

    # Genes on the left (x=-3) - only core genes
    n_genes = len(genes_sorted)
    if n_genes > 0:
        y_positions = np.linspace(vertical_span/2, -vertical_span/2, n_genes)
        for gene, y in zip(genes_sorted, y_positions):
            pos[gene] = (-3, y)

    # Reactions in the middle (x=0)
    n_reactions = len(reactions_sorted)
    if n_reactions > 0:
        y_positions = np.linspace(vertical_span/2, -vertical_span/2, n_reactions)
        for rxn, y in zip(reactions_sorted, y_positions):
            pos[rxn] = (0, y)

    # Metabolites on the right (x=3) with interleaving
    # Interleave target FFAs with other metabolites for spacing
    interleaved_metabolites = []
    other_idx = 0

    for i, target in enumerate(target_ffas_sorted):
        interleaved_metabolites.append(target)
        # Add 2-3 other metabolites between target FFAs for spacing
        for _ in range(3):
            if other_idx < len(other_metabolites_sorted):
                interleaved_metabolites.append(other_metabolites_sorted[other_idx])
                other_idx += 1

    # Add remaining other metabolites at the end
    while other_idx < len(other_metabolites_sorted):
        interleaved_metabolites.append(other_metabolites_sorted[other_idx])
        other_idx += 1

    n_metabolites = len(interleaved_metabolites)
    if n_metabolites > 0:
        y_positions = np.linspace(vertical_span*1.5, -vertical_span*1.5, n_metabolites)
        for met, y in zip(interleaved_metabolites, y_positions):
            pos[met] = (3, y)  # Right-shifted for label space

    print(f"Computed layout for {len(pos)} nodes ({len(genes_sorted)} core genes shown)")
    return pos


def visualize_ffa_network(G, pos, gene_nodes, reaction_nodes, metabolite_nodes):
    """
    Create visualization of FFA bipartite network.
    Updated to match multigraph overlay style (publication quality).
    """
    print("\n" + "=" * 80)
    print("CREATING NETWORK VISUALIZATION")
    print("=" * 80)

    # Publication size matching multigraph overlays
    fig, ax = plt.subplots(figsize=(20, 16))

    # Separate nodes by type and properties
    core_genes = [n for n in gene_nodes if G.nodes[n].get('is_core')]
    other_genes = [n for n in gene_nodes if not G.nodes[n].get('is_core')]
    target_ffas = [n for n in metabolite_nodes if G.nodes[n].get('is_target_ffa')]
    other_metabolites = [n for n in metabolite_nodes if not G.nodes[n].get('is_target_ffa')]

    # Draw edges FIRST (so they appear behind nodes)
    gene_rxn_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'catalyzes']
    met_rxn_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'consumed_by']
    rxn_met_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'produces']

    # Grey metabolic edges matching multigraph style
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

    # Draw nodes with sizes matching multigraph overlays
    if other_metabolites:
        nx.draw_networkx_nodes(G, pos, nodelist=other_metabolites, node_color=COLORS['metabolite'],
                              node_size=100, ax=ax, alpha=0.4)
    if target_ffas:
        nx.draw_networkx_nodes(G, pos, nodelist=target_ffas, node_color=COLORS['target_ffa'],
                              node_size=100, ax=ax, alpha=0.85)
    if reaction_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=reaction_nodes, node_color=COLORS['reaction'],
                              node_size=80, node_shape='s', ax=ax, alpha=0.6)
    # REMOVED non-core genes (other_genes) - like overlay plots
    # if other_genes:
    #     nx.draw_networkx_nodes(G, pos, nodelist=other_genes, node_color=COLORS['gene'],
    #                           node_size=150, ax=ax, alpha=0.75)
    if core_genes:
        nx.draw_networkx_nodes(G, pos, nodelist=core_genes, node_color=COLORS['core_gene'],
                              node_size=200, ax=ax, alpha=0.9)

    # Gene labels
    gene_labels = {}
    for node in core_genes:
        gene_labels[node] = G.nodes[node].get('label', node)

    if gene_labels:
        nx.draw_networkx_labels(G, pos, labels=gene_labels, font_size=11,
                               font_weight='bold', ax=ax)

    # Metabolite labels with FFA type and compartment - LEFT SHIFTED
    # V4: Label ALL metabolites with their FFA type, including complex lipids
    met_labels = {}
    met_label_pos = {}

    # Load FFA mapping to get FFA type for ALL metabolites
    ffa_mapping_path = FFA_REACTIONS_DIR / "ffa_metabolite_mapping.csv"
    ffa_mapping = pd.read_csv(ffa_mapping_path)
    met_to_ffa = dict(zip(ffa_mapping['metabolite_id'], ffa_mapping['ffa_type']))

    for node in target_ffas:
        met_name = G.nodes[node].get('name', node)
        compartment = G.nodes[node].get('compartment', '')

        # Map compartment codes to readable names
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
        comp_name = comp_map.get(compartment, compartment) if compartment else ''

        # Extract ALL FFA types from the metabolite name (for complex lipids)
        ffa_types = extract_all_ffa_types(met_name)

        # Format label: "[C16:0, C18:1] phosphatidate (1-16:0, 2-18:1) [ER membrane]"
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

        # Left-shift label position for readability
        node_pos = pos[node]
        met_label_pos[node] = (node_pos[0] - 0.2, node_pos[1])

    if met_labels:
        # Use right alignment so text extends to the left
        nx.draw_networkx_labels(G, met_label_pos, labels=met_labels, font_size=10,
                               font_weight='bold', horizontalalignment='right', ax=ax)

    # Title matching multigraph style
    ax.set_title("FFA Metabolic Network: Genes → Reactions → Metabolites",
                fontsize=14, pad=15, fontweight='bold')

    # Legend with larger font matching multigraph style
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['core_gene'],
               markersize=12, label='Core FFA Pathway Genes'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['gene'],
               markersize=10, label='Associated Genes'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['reaction'],
               markersize=8, label='Reactions'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['target_ffa'],
               markersize=10, label='Target FFAs'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['metabolite'],
               markersize=8, label='Other Metabolites'),
        Line2D([0], [0], color='#404040', linewidth=1, label='Metabolic Reactions'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=12, framealpha=0.95)
    ax.axis('off')

    plt.tight_layout()

    # Save figure
    filename = "ffa_bipartite_network.png"
    ffa_dir = osp.join(ASSET_IMAGES_DIR, "008-xue-ffa")
    os.makedirs(ffa_dir, exist_ok=True)
    output_path = osp.join(ffa_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved network visualization to:\n  {output_path}")
    plt.close()

    return output_path


def compute_subsystem_layout(H, gene_nodes, reaction_nodes, metabolite_nodes):
    """
    Compute compact layout for subsystem subgraph with better metabolite spacing.
    """
    pos = {}

    # Filter to only core genes
    core_genes = [n for n in gene_nodes if H.nodes[n].get('is_core')]

    # Sort nodes for consistent ordering
    genes_sorted = sorted(core_genes)  # Only core genes
    reactions_sorted = sorted(reaction_nodes)

    # Separate and sort metabolites
    target_ffas = [n for n in metabolite_nodes if H.nodes[n].get('is_target_ffa')]
    other_metabolites = [n for n in metabolite_nodes if not H.nodes[n].get('is_target_ffa')]

    # Sort by compartment for grouping
    def get_compartment_order(met_id):
        comp = H.nodes[met_id].get('compartment', 'z')
        # Order: cytoplasm first, then others
        comp_order = {'c': 0, 'erm': 1, 'lp': 2, 'e': 3, 'm': 4, 'n': 5, 'p': 6, 'v': 7, 'g': 8}
        return comp_order.get(comp, 99)

    target_ffas_sorted = sorted(target_ffas, key=lambda x: (get_compartment_order(x), x))
    other_metabolites_sorted = sorted(other_metabolites)

    # Compact vertical spacing but with more room for metabolites
    vertical_span = 4.0

    # Genes on the left (x=-2.5) - only core genes
    n_genes = len(genes_sorted)
    if n_genes > 0:
        y_positions = np.linspace(vertical_span/2, -vertical_span/2, n_genes)
        for gene, y in zip(genes_sorted, y_positions):
            pos[gene] = (-2.5, y)

    # Reactions in the middle (x=0)
    n_reactions = len(reactions_sorted)
    if n_reactions > 0:
        y_positions = np.linspace(vertical_span/2, -vertical_span/2, n_reactions)
        for rxn, y in zip(reactions_sorted, y_positions):
            pos[rxn] = (0, y)

    # Metabolites on the right (x=2.5) with interleaving for spacing
    interleaved_metabolites = []
    other_idx = 0

    # Interleave target FFAs with other metabolites
    for i, target in enumerate(target_ffas_sorted):
        interleaved_metabolites.append(target)
        # Add 2 other metabolites between target FFAs for spacing
        for _ in range(2):
            if other_idx < len(other_metabolites_sorted):
                interleaved_metabolites.append(other_metabolites_sorted[other_idx])
                other_idx += 1

    # Add remaining other metabolites at the end
    while other_idx < len(other_metabolites_sorted):
        interleaved_metabolites.append(other_metabolites_sorted[other_idx])
        other_idx += 1

    n_metabolites = len(interleaved_metabolites)
    if n_metabolites > 0:
        y_positions = np.linspace(vertical_span * 1.5, -vertical_span * 1.5, n_metabolites)
        for met, y in zip(interleaved_metabolites, y_positions):
            pos[met] = (2.5, y)  # Right-shifted for label space

    return pos


def create_subsystem_subgraphs(G, pos, reactions_df):
    """
    Create separate visualizations for each subsystem with more compact layout.
    """
    print("\n" + "=" * 80)
    print("CREATING SUBSYSTEM SUBGRAPHS")
    print("=" * 80)

    subsystems = reactions_df['subsystem'].unique()
    output_paths = []

    for subsystem in subsystems:
        print(f"\nCreating subgraph for: {subsystem}")

        # Get reactions in this subsystem
        subsys_rxns = set(reactions_df[reactions_df['subsystem'] == subsystem]['reaction_id'])

        # Get connected nodes
        connected_nodes = set(subsys_rxns)
        for rxn in subsys_rxns:
            # Add genes catalyzing these reactions
            for pred in G.predecessors(rxn):
                if G.nodes[pred]['node_type'] == 'gene':
                    connected_nodes.add(pred)
            # Add metabolites involved in these reactions
            for succ in G.successors(rxn):
                if G.nodes[succ]['node_type'] == 'metabolite':
                    connected_nodes.add(succ)
            for pred in G.predecessors(rxn):
                if G.nodes[pred]['node_type'] == 'metabolite':
                    connected_nodes.add(pred)

        # Create subgraph
        H = G.subgraph(connected_nodes).copy()

        # Separate nodes by type
        gene_nodes_sub = [n for n in H.nodes() if H.nodes[n]['node_type'] == 'gene']
        reaction_nodes_sub = [n for n in H.nodes() if H.nodes[n]['node_type'] == 'reaction']
        metabolite_nodes_sub = [n for n in H.nodes() if H.nodes[n]['node_type'] == 'metabolite']

        # Create custom compact layout for subsystem
        subgraph_pos = compute_subsystem_layout(H, gene_nodes_sub, reaction_nodes_sub, metabolite_nodes_sub)

        # More compact figure size based on node count
        n_nodes = max(len(gene_nodes_sub), len(reaction_nodes_sub), len(metabolite_nodes_sub))
        fig_height = min(10, max(6, n_nodes * 0.3))  # More compact height
        fig, ax = plt.subplots(figsize=(14, fig_height))

        # Separate nodes by type
        core_genes_sub = [n for n in gene_nodes_sub if H.nodes[n].get('is_core')]
        other_genes_sub = [n for n in gene_nodes_sub if not H.nodes[n].get('is_core')]
        target_ffas_sub = [n for n in metabolite_nodes_sub if H.nodes[n].get('is_target_ffa')]
        other_mets_sub = [n for n in metabolite_nodes_sub if not H.nodes[n].get('is_target_ffa')]

        # Draw edges FIRST (so they appear behind nodes)
        gene_rxn_edges = [(u, v) for u, v, d in H.edges(data=True) if d.get('edge_type') == 'catalyzes']
        met_rxn_edges = [(u, v) for u, v, d in H.edges(data=True) if d.get('edge_type') == 'consumed_by']
        rxn_met_edges = [(u, v) for u, v, d in H.edges(data=True) if d.get('edge_type') == 'produces']

        # Grey edges matching multigraph style
        if gene_rxn_edges:
            nx.draw_networkx_edges(H, subgraph_pos, edgelist=gene_rxn_edges,
                                  edge_color='#404040', width=0.8, alpha=0.3, arrows=False, ax=ax)
        if met_rxn_edges:
            nx.draw_networkx_edges(H, subgraph_pos, edgelist=met_rxn_edges,
                                  edge_color='#404040', width=0.6, alpha=0.25, arrows=True, arrowsize=8,
                                  arrowstyle='->', ax=ax)
        if rxn_met_edges:
            nx.draw_networkx_edges(H, subgraph_pos, edgelist=rxn_met_edges,
                                  edge_color='#404040', width=0.6, alpha=0.25, arrows=True, arrowsize=8,
                                  arrowstyle='->', ax=ax)

        # Draw nodes with consistent sizes matching individual FFA plots
        if other_mets_sub:
            nx.draw_networkx_nodes(H, subgraph_pos, nodelist=other_mets_sub,
                                  node_color=COLORS['metabolite'], node_size=100, ax=ax, alpha=0.4)
        if target_ffas_sub:
            nx.draw_networkx_nodes(H, subgraph_pos, nodelist=target_ffas_sub,
                                  node_color=COLORS['target_ffa'], node_size=100, ax=ax, alpha=0.85)
        if reaction_nodes_sub:
            nx.draw_networkx_nodes(H, subgraph_pos, nodelist=reaction_nodes_sub,
                                  node_color=COLORS['reaction'], node_size=80, node_shape='s', ax=ax, alpha=0.6)
        # REMOVED non-core genes (other_genes_sub) - like overlay plots
        # if other_genes_sub:
        #     nx.draw_networkx_nodes(H, subgraph_pos, nodelist=other_genes_sub,
        #                           node_color=COLORS['gene'], node_size=150, ax=ax, alpha=0.75)
        if core_genes_sub:
            nx.draw_networkx_nodes(H, subgraph_pos, nodelist=core_genes_sub,
                                  node_color=COLORS['core_gene'], node_size=200, ax=ax, alpha=0.9)

        # Add labels for subsystem view (all black text)
        # Gene labels (only core genes)
        gene_labels = {}
        for node in core_genes_sub:
            gene_labels[node] = H.nodes[node].get('label', node)

        if gene_labels:
            nx.draw_networkx_labels(H, subgraph_pos, labels=gene_labels, font_size=11,
                                   font_weight='bold', ax=ax)

        # Metabolite labels with FFA type and compartment - LEFT SHIFTED
        # V4: Label ALL metabolites with their FFA type, including complex lipids
        met_labels = {}
        met_label_pos = {}

        # Load FFA mapping to get FFA type for ALL metabolites
        ffa_mapping_path = FFA_REACTIONS_DIR / "ffa_metabolite_mapping.csv"
        ffa_mapping = pd.read_csv(ffa_mapping_path)
        met_to_ffa = dict(zip(ffa_mapping['metabolite_id'], ffa_mapping['ffa_type']))

        # Process target FFAs for labeling
        for node in target_ffas_sub:
            met_name = H.nodes[node].get('name', node)
            compartment = H.nodes[node].get('compartment', '')

            # Map compartment codes to readable names
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
            comp_name = comp_map.get(compartment, compartment) if compartment else ''

            # Extract ALL FFA types from the metabolite name (for complex lipids)
            ffa_types = extract_all_ffa_types(met_name)

            # Format label: "[C16:0, C18:1] phosphatidate (1-16:0, 2-18:1) [ER membrane]"
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

            # Left-shift label position for readability
            node_pos = subgraph_pos[node]
            met_label_pos[node] = (node_pos[0] - 0.1, node_pos[1])

        if met_labels:
            # Use right alignment so text extends to the left
            nx.draw_networkx_labels(H, met_label_pos, labels=met_labels, font_size=10,
                                   font_weight='bold', horizontalalignment='right', ax=ax)

        # Title matching multigraph style
        ax.set_title(f"FFA Subsystem: {subsystem}\n({len(reaction_nodes_sub)} reactions, "
                    f"{len(gene_nodes_sub)} genes, {len(metabolite_nodes_sub)} metabolites)",
                    fontsize=14, pad=15, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()

        # Save
        safe_name = subsystem.replace(' ', '_').replace('/', '_')
        filename = f"ffa_subsystem_{safe_name}.png"
        ffa_dir = osp.join(ASSET_IMAGES_DIR, "008-xue-ffa")
        os.makedirs(ffa_dir, exist_ok=True)
        output_path = osp.join(ffa_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved to: {output_path}")
        output_paths.append(output_path)
        plt.close()

    return output_paths


def save_network_data(G, pos):
    """Save the network and layout for use in Step 4."""
    print("\n" + "=" * 80)
    print("SAVING NETWORK DATA")
    print("=" * 80)

    network_path = RESULTS_DIR / "ffa_bipartite_network.graphml"
    pos_path = RESULTS_DIR / "ffa_network_layout.json"

    # Save network
    nx.write_graphml(G, network_path)
    print(f"Saved network to: {network_path}")

    # Save layout
    pos_serializable = {node: list(coords) for node, coords in pos.items()}
    with open(pos_path, 'w') as f:
        json.dump(pos_serializable, f, indent=2)
    print(f"Saved layout to: {pos_path}")


def create_individual_ffa_plots(G, model, ffa_mapping_df, metabolites_df, genes_df):
    """
    Create individual plots for each FFA type with proper spacing.
    Each plot shows only the subnetwork for that specific FFA.
    FFA ordering: C14:0 at top, longer chains below.
    """
    print("\n" + "=" * 80)
    print("CREATING INDIVIDUAL FFA PLOTS")
    print("=" * 80)

    # FFA types in display order (C14:0 first/at top)
    # This matches the overlay plot ordering
    ffa_types = ['C14:0', 'C16:0', 'C16:1', 'C18:0', 'C18:1']
    output_paths = []

    for ffa_type in ffa_types:
        print(f"\n{'-'*80}")
        print(f"Creating plot for {ffa_type}")
        print(f"{'-'*80}")

        # Get metabolites for this FFA type
        # Filter to only include direct FFA metabolites (not complex lipids)
        ffa_df = ffa_mapping_df[ffa_mapping_df['ffa_type'] == ffa_type]

        # Define the direct FFA names for each type (keywords to match)
        direct_ffa_names = {
            'C14:0': ['myristate'],
            'C16:0': ['palmitate'],
            'C16:1': ['palmitoleate'],
            'C18:0': ['stearate'],
            'C18:1': ['oleate']
        }

        # Filter to only the direct FFA forms (not diglycerides, triglycerides, etc.)
        if ffa_type in direct_ffa_names:
            # Use partial matching for metabolite names
            keywords = direct_ffa_names[ffa_type]
            # Exclude complex lipids (those containing position notation like "1-16:0")
            mask = ffa_df['metabolite_name'].apply(
                lambda x: any(kw in x.lower() for kw in keywords) and
                         not any(pattern in x for pattern in ['1-', '2-', '3-', 'phosphatidate', 'diglyceride', 'triglyceride'])
            )
            ffa_metabolites = set(ffa_df[mask]['metabolite_id'])
        else:
            # Fallback if FFA type not in our mapping
            ffa_metabolites = set(ffa_df['metabolite_id'])

        if len(ffa_metabolites) == 0:
            print(f"  No direct FFA metabolites found for {ffa_type}")
            continue

        # Build subgraph for this FFA
        # Start with FFA metabolites for this specific type
        subgraph_nodes = set(ffa_metabolites)

        # Add reactions that produce or consume these FFA metabolites
        connected_reactions = set()
        for met_id in ffa_metabolites:
            if met_id in G.nodes():
                # Get reactions that produce this metabolite
                for pred in G.predecessors(met_id):
                    if G.nodes[pred].get('node_type') == 'reaction':
                        connected_reactions.add(pred)
                # Get reactions that consume this metabolite
                for succ in G.successors(met_id):
                    if G.nodes[succ].get('node_type') == 'reaction':
                        connected_reactions.add(succ)

        subgraph_nodes.update(connected_reactions)

        # Add ONLY genes that catalyze these reactions (core genes only)
        connected_genes = set()
        for rxn in connected_reactions:
            for pred in G.predecessors(rxn):
                if G.nodes[pred].get('node_type') == 'gene':
                    # Only include core genes
                    if G.nodes[pred].get('is_core'):
                        connected_genes.add(pred)

        subgraph_nodes.update(connected_genes)

        # V3: Include ALL metabolites connected to reactions to show complete reaction context
        # This includes substrates/cofactors/products, not just the target FFA
        all_connected_metabolites = set()
        for rxn in connected_reactions:
            # Get all metabolites involved in this reaction (both inputs and outputs)
            for pred in G.predecessors(rxn):
                if G.nodes[pred].get('node_type') == 'metabolite':
                    all_connected_metabolites.add(pred)
            for succ in G.successors(rxn):
                if G.nodes[succ].get('node_type') == 'metabolite':
                    all_connected_metabolites.add(succ)

        subgraph_nodes.update(all_connected_metabolites)

        # Keep track of which metabolites are target FFAs vs others for coloring
        metabolite_nodes_filtered = all_connected_metabolites

        # Create subgraph
        H = G.subgraph(subgraph_nodes).copy()

        print(f"  Subgraph: {len(connected_genes)} genes, {len(connected_reactions)} reactions, {len(metabolite_nodes_filtered)} metabolites ({len(ffa_metabolites)} target FFAs)")

        # Create custom layout for this FFA
        pos_ffa = compute_ffa_specific_layout(H, connected_genes, connected_reactions, metabolite_nodes_filtered, ffa_type)

        # Visualize
        output_path = visualize_individual_ffa(H, pos_ffa, connected_genes, connected_reactions,
                                              metabolite_nodes_filtered, ffa_type, ffa_mapping_df, metabolites_df)
        output_paths.append(output_path)

    return output_paths


def compute_ffa_specific_layout(H, gene_nodes, reaction_nodes, metabolite_nodes, ffa_type):
    """
    Compute layout for individual FFA plot with better vertical spacing.
    Implements interleaving and compartment awareness from overlay plots.
    V3: Adaptive vertical spacing based on node count to prevent overlap.
    """
    pos = {}

    # Convert sets to sorted lists for consistent ordering
    genes_sorted = sorted(gene_nodes)
    reactions_sorted = sorted(reaction_nodes)

    # Sort metabolites by compartment for better grouping
    # Extract compartment from node attributes (not from node ID)
    def get_compartment(met_id):
        if met_id in H.nodes():
            return H.nodes[met_id].get('compartment', 'z')
        return 'z'  # Default for no compartment

    metabolites_sorted = sorted(metabolite_nodes, key=lambda x: (get_compartment(x), x))

    # V3: Adaptive vertical spacing based on node count to prevent overlap
    n_reactions = len(reactions_sorted)
    n_metabolites = len(metabolites_sorted)
    # Calculate adaptive spacing - more nodes require larger vertical span
    vertical_span = max(4.0, max(n_reactions, n_metabolites) * 0.8)

    # Genes on the left (x=-2.5) - farther left for more space
    n_genes = len(genes_sorted)
    if n_genes > 0:
        y_positions = np.linspace(vertical_span/2, -vertical_span/2, n_genes)
        for gene, y in zip(genes_sorted, y_positions):
            pos[gene] = (-2.5, y)

    # Reactions in the middle (x=0)
    n_reactions = len(reactions_sorted)
    if n_reactions > 0:
        y_positions = np.linspace(vertical_span/2, -vertical_span/2, n_reactions)
        for rxn, y in zip(reactions_sorted, y_positions):
            pos[rxn] = (0, y)

    # Metabolites on the right (x=2.5) with proper spacing
    n_metabolites = len(metabolites_sorted)
    if n_metabolites > 0:
        # Use interleaving approach from overlay plots
        # Space metabolites evenly across slightly larger range
        y_positions = np.linspace(vertical_span * 1.2, -vertical_span * 1.2, n_metabolites)
        for met, y in zip(metabolites_sorted, y_positions):
            pos[met] = (2.5, y)  # Shifted right for label space

    return pos


def visualize_individual_ffa(H, pos, gene_nodes, reaction_nodes, metabolite_nodes,
                            ffa_type, ffa_mapping_df, metabolites_df):
    """
    Create visualization for a single FFA type with proper labels.
    Improved with compartment suffixes and better aspect ratios.
    V3: Distinguish target FFAs (blue) from other metabolites (grey).
    """
    # Better aspect ratio - wider and less tall
    n_nodes = max(len(gene_nodes), len(reaction_nodes), len(metabolite_nodes))
    fig_height = min(10, max(6, n_nodes * 0.4))  # Capped height for compactness
    fig, ax = plt.subplots(figsize=(14, fig_height))

    # Draw edges FIRST (grey, behind nodes)
    gene_rxn_edges = [(u, v) for u, v, d in H.edges(data=True) if d.get('edge_type') == 'catalyzes']
    met_rxn_edges = [(u, v) for u, v, d in H.edges(data=True) if d.get('edge_type') == 'consumed_by']
    rxn_met_edges = [(u, v) for u, v, d in H.edges(data=True) if d.get('edge_type') == 'produces']

    # Grey edges matching multigraph style
    if gene_rxn_edges:
        nx.draw_networkx_edges(H, pos, edgelist=gene_rxn_edges, edge_color='#404040',
                              width=0.8, alpha=0.3, arrows=False, ax=ax)
    if met_rxn_edges:
        nx.draw_networkx_edges(H, pos, edgelist=met_rxn_edges, edge_color='#404040',
                              width=0.6, alpha=0.25, arrows=True, arrowsize=8,
                              arrowstyle='->', ax=ax)
    if rxn_met_edges:
        nx.draw_networkx_edges(H, pos, edgelist=rxn_met_edges, edge_color='#404040',
                              width=0.6, alpha=0.25, arrows=True, arrowsize=8,
                              arrowstyle='->', ax=ax)

    # V3: Separate target FFAs from other metabolites for proper coloring
    # Get target FFAs for this FFA type (direct FFAs only, not complex lipids)
    direct_ffa_names = {
        'C14:0': ['myristate'],
        'C16:0': ['palmitate'],
        'C16:1': ['palmitoleate'],
        'C18:0': ['stearate'],
        'C18:1': ['oleate']
    }

    target_ffa_metabolites = set()
    other_metabolites = set()

    for met_id in metabolite_nodes:
        if met_id in H.nodes():
            met_name = H.nodes[met_id].get('name', '')
            # Check if this is a target FFA for this specific type
            is_target = False
            if ffa_type in direct_ffa_names:
                keywords = direct_ffa_names[ffa_type]
                # Check if metabolite name contains the FFA keyword AND is not a complex lipid
                if (any(kw in met_name.lower() for kw in keywords) and
                    not any(pattern in met_name for pattern in ['1-', '2-', '3-', 'phosphatidate', 'diglyceride', 'triglyceride'])):
                    is_target = True

            if is_target:
                target_ffa_metabolites.add(met_id)
            else:
                other_metabolites.add(met_id)

    # Draw nodes with consistent sizes from overlay plots
    if gene_nodes:
        nx.draw_networkx_nodes(H, pos, nodelist=list(gene_nodes),
                              node_color=COLORS['core_gene'],
                              node_size=200, ax=ax, alpha=0.9)
    if reaction_nodes:
        nx.draw_networkx_nodes(H, pos, nodelist=list(reaction_nodes),
                              node_color=COLORS['reaction'],
                              node_size=80, node_shape='s', ax=ax, alpha=0.6)

    # V3: Draw other metabolites in grey (substrates, cofactors, products)
    if other_metabolites:
        nx.draw_networkx_nodes(H, pos, nodelist=list(other_metabolites),
                              node_color=COLORS['metabolite'],
                              node_size=100, ax=ax, alpha=0.4)

    # V3: Draw target FFAs in blue
    if target_ffa_metabolites:
        nx.draw_networkx_nodes(H, pos, nodelist=list(target_ffa_metabolites),
                              node_color=COLORS['target_ffa'],
                              node_size=100, ax=ax, alpha=0.85)

    # Gene labels
    gene_labels = {}
    gene_label_pos = {}
    for node in gene_nodes:
        if node in H.nodes():
            label = H.nodes[node].get('label', node)
            gene_labels[node] = label
            # Keep gene labels at node position
            gene_label_pos[node] = pos[node]

    if gene_labels:
        nx.draw_networkx_labels(H, gene_label_pos, labels=gene_labels, font_size=12,
                               font_weight='bold', ax=ax)

    # V3: Metabolite labels - apply FFA type only to target FFAs
    met_labels = {}
    met_label_pos = {}

    # Map compartment codes to readable names
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

    for node in metabolite_nodes:
        if node in H.nodes():
            met_name = H.nodes[node].get('name', node)
            compartment = H.nodes[node].get('compartment', '')
            comp_name = comp_map.get(compartment, compartment) if compartment else ''

            # Extract ALL FFA types from the metabolite name (for complex lipids)
            ffa_types = extract_all_ffa_types(met_name)

            # Format: "[C16:0, C18:1] phosphatidate (1-16:0, 2-18:1) [cytoplasm]" for target FFAs
            # Format: "ATP [cytoplasm]" for other metabolites
            if node in target_ffa_metabolites:
                # This is a target FFA - add FFA type prefix
                if ffa_types and comp_name:
                    ffa_str = ', '.join(ffa_types)
                    met_labels[node] = f"[{ffa_str}] {met_name} [{comp_name}]"
                elif ffa_types:
                    ffa_str = ', '.join(ffa_types)
                    met_labels[node] = f"[{ffa_str}] {met_name}"
                elif comp_name:
                    # Fallback if no FFA types extracted
                    met_labels[node] = f"[{ffa_type}] {met_name} [{comp_name}]"
                else:
                    met_labels[node] = f"[{ffa_type}] {met_name}"
            else:
                # This is another metabolite (substrate/cofactor/product) - no FFA type
                if comp_name:
                    met_labels[node] = f"{met_name} [{comp_name}]"
                else:
                    met_labels[node] = met_name

            # Left-shift label position for better readability
            node_pos = pos[node]
            met_label_pos[node] = (node_pos[0] - 0.1, node_pos[1])

    if met_labels:
        # Use right alignment so text extends to the left
        nx.draw_networkx_labels(H, met_label_pos, labels=met_labels, font_size=11,
                               font_weight='bold', horizontalalignment='right', ax=ax)

    # Title with updated counts
    ax.set_title(f"FFA Metabolic Network: {ffa_type}\n"
                f"({len(gene_nodes)} core genes, {len(reaction_nodes)} reactions, "
                f"{len(target_ffa_metabolites)} target FFAs)",
                fontsize=14, pad=15, fontweight='bold')

    # V3: Updated legend to distinguish target FFAs from other metabolites
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['core_gene'],
               markersize=10, label='Core FFA Pathway Genes'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['reaction'],
               markersize=8, label='Reactions'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['target_ffa'],
               markersize=10, label=f'{ffa_type} Metabolites'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['metabolite'],
               markersize=8, label='Other Metabolites'),
        Line2D([0], [0], color='#404040', linewidth=1, label='Metabolic Reactions'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.95)
    ax.axis('off')

    plt.tight_layout()

    # Save figure
    safe_name = ffa_type.replace(':', '_')
    filename = f"ffa_bipartite_{safe_name}.png"
    ffa_dir = osp.join(ASSET_IMAGES_DIR, "008-xue-ffa")
    os.makedirs(ffa_dir, exist_ok=True)
    output_path = osp.join(ffa_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved to: {output_path}")
    plt.close()

    return output_path


def create_ffa_bipartite_network():
    """
    Create bipartite network visualization of FFA metabolism.

    Extracts the FFA subnetwork from Yeast GEM and creates:
    1. Individual FFA-specific plots (C14:0, C16:0, C16:1, C18:0, C18:1)
    2. Overall bipartite network (genes → reactions → metabolites)
    3. Subsystem-specific subgraphs

    Saves network and layout for use in Step 4 (multigraph overlay).
    """
    print("=" * 80)
    print("FFA BIPARTITE NETWORK CREATION")
    print("=" * 80)

    # Load Yeast GEM
    print("\nLoading Yeast GEM model...")
    yeast_gem = YeastGEM(root=osp.join(DATA_ROOT, "data/torchcell/yeast_gem"))
    model = yeast_gem.model

    # Load FFA data from Step 1
    reactions_df, genes_df, metabolites_df, ffa_mapping_df = load_ffa_data()

    # Extract FFA subnetwork
    G, gene_nodes, reaction_nodes, metabolite_nodes = extract_ffa_subnetwork(
        model, reactions_df, genes_df, metabolites_df
    )

    # NEW: Create individual FFA plots with proper spacing
    individual_paths = create_individual_ffa_plots(G, model, ffa_mapping_df, metabolites_df, genes_df)

    # Compute layout (fixed for Step 4)
    pos = compute_layout(G, gene_nodes, reaction_nodes, metabolite_nodes)

    # Create visualizations
    main_viz_path = visualize_ffa_network(G, pos, gene_nodes, reaction_nodes, metabolite_nodes)

    # Create subsystem subgraphs
    subsystem_paths = create_subsystem_subgraphs(G, pos, reactions_df)

    # Save for Step 4
    save_network_data(G, pos)

    print("\n" + "=" * 80)
    print("FFA BIPARTITE NETWORK CREATION COMPLETE!")
    print("=" * 80)
    print(f"\nCreated visualizations:")
    print(f"  Individual FFA plots: {len(individual_paths)}")
    print(f"  Main network: {main_viz_path}")
    print(f"  Subsystem graphs: {len(subsystem_paths)}")
    print("\nNetwork and layout saved for Step 4 (multigraph overlay).")

    return G, pos


if __name__ == "__main__":
    G, pos = create_ffa_bipartite_network()
