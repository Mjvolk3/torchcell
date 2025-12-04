#!/usr/bin/env python3
"""
Create a NetworkX/matplotlib visualization of FFA metabolism.
Since Escher isn't working with our yeast IDs, let's create our own visualization.
"""

import os
import os.path as osp
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from torchcell.metabolism.yeast_GEM import YeastGEM
from torchcell.timestamp import timestamp
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Load environment
load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")

RESULTS_DIR = Path("/Users/michaelvolk/Documents/projects/torchcell/experiments/008-xue-ffa/results")
FFA_REACTIONS_DIR = RESULTS_DIR / "ffa_reactions"


def create_networkx_ffa_visualization():
    """
    Create a metabolic pathway visualization using NetworkX and matplotlib.
    """
    print("=" * 80)
    print("CREATING FFA VISUALIZATION WITH NETWORKX")
    print("=" * 80)

    # Load Yeast GEM
    print("\nLoading Yeast GEM model...")
    yeast_gem = YeastGEM(root=osp.join(DATA_ROOT, "data/torchcell/yeast_gem"))
    model = yeast_gem.model
    print(f"Loaded model: {len(model.reactions)} reactions")

    # Load FFA reactions
    print("\nLoading FFA reactions...")
    reactions_df = pd.read_csv(FFA_REACTIONS_DIR / "ffa_reactions_list.csv")
    metabolites_df = pd.read_csv(FFA_REACTIONS_DIR / "ffa_metabolites_list.csv")
    print(f"Loaded {len(reactions_df)} reactions")
    print(f"Loaded {len(metabolites_df)} metabolites")

    # Create bipartite graph
    print("\nCreating bipartite graph...")
    G = nx.DiGraph()  # Directed graph for metabolic pathways

    ffa_reaction_ids = set(reactions_df['reaction_id'].unique())

    # Track target FFAs and subsystems
    target_ffas = set(metabolites_df[metabolites_df['is_target_ffa']]['metabolite_id'])
    subsystem_colors = {
        'Fatty acid biosynthesis': '#2E7D32',  # Green
        'Fatty acid degradation': '#C62828',   # Red
        'Biosynthesis of unsaturated fatty acids': '#1565C0',  # Blue
        'Glycerolipid metabolism': '#F57C00',  # Orange
        'Fatty acid ester pathway': '#7B1FA2'  # Purple
    }

    # Add reactions and metabolites
    for rxn_id in ffa_reaction_ids:
        if rxn_id not in model.reactions:
            continue

        rxn = model.reactions.get_by_id(rxn_id)
        rxn_info = reactions_df[reactions_df['reaction_id'] == rxn_id].iloc[0]
        subsystem = rxn_info['subsystem']

        # Add reaction node
        G.add_node(rxn_id,
                   node_type='reaction',
                   name=rxn.name,
                   subsystem=subsystem,
                   color=subsystem_colors.get(subsystem, '#666666'))

        # Add metabolites and edges
        for met, coef in rxn.metabolites.items():
            met_id = met.id

            # Add metabolite node if not present
            if met_id not in G:
                is_target = met_id in target_ffas
                G.add_node(met_id,
                          node_type='metabolite',
                          name=met.name,
                          is_target_ffa=is_target,
                          color='#FFD700' if is_target else '#E0E0E0')  # Gold for targets

            # Add directed edges
            if coef < 0:  # Reactant
                G.add_edge(met_id, rxn_id, weight=abs(coef))
            else:  # Product
                G.add_edge(rxn_id, met_id, weight=coef)

    print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # Compute layout
    print("\nComputing layout...")
    # Try different layouts to find the best one
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Create visualization
    print("\nCreating visualization...")
    fig, ax = plt.subplots(1, 1, figsize=(20, 16))

    # Separate nodes by type
    reaction_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'reaction']
    metabolite_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'metabolite']
    target_ffa_nodes = [n for n, d in G.nodes(data=True)
                        if d.get('node_type') == 'metabolite' and d.get('is_target_ffa', False)]

    # Draw edges with transparency
    nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='gray',
                          arrows=True, arrowsize=10, arrowstyle='->')

    # Draw reaction nodes (squares)
    for node in reaction_nodes:
        x, y = pos[node]
        color = G.nodes[node]['color']
        box = FancyBboxPatch((x - 0.02, y - 0.02), 0.04, 0.04,
                             boxstyle="round,pad=0.005",
                             facecolor=color, edgecolor='black',
                             transform=ax.transData)
        ax.add_patch(box)

    # Draw metabolite nodes (circles)
    metabolite_colors = [G.nodes[n]['color'] for n in metabolite_nodes]
    nx.draw_networkx_nodes(G, pos, nodelist=metabolite_nodes,
                          node_color=metabolite_colors,
                          node_shape='o', node_size=300, alpha=0.8)

    # Highlight target FFAs with border
    if target_ffa_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=target_ffa_nodes,
                              node_color='#FFD700',
                              node_shape='o', node_size=500,
                              edgecolors='red', linewidths=2)

    # Add labels for important nodes
    print("\nAdding labels...")
    # Label all target FFAs
    target_labels = {n: G.nodes[n]['name'].split('[')[0].strip()[:20]
                    for n in target_ffa_nodes}
    nx.draw_networkx_labels(G, pos, target_labels, font_size=8, font_weight='bold')

    # Label key reactions (sample from each subsystem)
    key_reactions = []
    for subsystem in subsystem_colors.keys():
        subsystem_rxns = [n for n in reaction_nodes
                         if G.nodes[n].get('subsystem') == subsystem]
        if subsystem_rxns:
            key_reactions.extend(subsystem_rxns[:2])  # Take first 2 from each

    reaction_labels = {n: n for n in key_reactions[:10]}  # Show up to 10 reaction IDs
    nx.draw_networkx_labels(G, pos, reaction_labels, font_size=6, font_color='blue')

    # Create legend
    legend_elements = []
    for subsystem, color in subsystem_colors.items():
        count = sum(1 for n in reaction_nodes if G.nodes[n].get('subsystem') == subsystem)
        if count > 0:
            legend_elements.append(mpatches.Patch(color=color,
                                                 label=f'{subsystem} ({count})'))

    legend_elements.append(mpatches.Patch(color='#FFD700', label='Target FFAs'))
    legend_elements.append(mpatches.Patch(color='#E0E0E0', label='Other metabolites'))

    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    # Set title and clean up
    ax.set_title('Free Fatty Acid (FFA) Metabolic Network in S. cerevisiae',
                fontsize=16, fontweight='bold')
    ax.axis('off')

    # Save the figure
    filename = "ffa_networkx_visualization.png"
    ffa_dir = osp.join(ASSET_IMAGES_DIR, "008-xue-ffa")
    os.makedirs(ffa_dir, exist_ok=True)
    output_path = osp.join(ffa_dir, filename)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved visualization to: {output_path}")

    # Also save a version with hierarchical layout
    print("\nCreating hierarchical layout version...")
    fig2, ax2 = plt.subplots(1, 1, figsize=(24, 16))

    # Try hierarchical layout for better organization
    # Create a simplified version for hierarchical layout
    G_simple = nx.DiGraph()
    for edge in G.edges():
        G_simple.add_edge(edge[0], edge[1])

    try:
        # Try to create layers
        pos2 = nx.multipartite_layout(G, subset_key='node_type')
    except:
        # Fall back to spring layout if multipartite doesn't work
        pos2 = nx.kamada_kawai_layout(G)

    # Draw with same style as before
    nx.draw_networkx_edges(G, pos2, alpha=0.2, edge_color='gray',
                          arrows=True, arrowsize=10, arrowstyle='->', ax=ax2)

    # Draw reaction nodes
    for node in reaction_nodes:
        x, y = pos2[node]
        color = G.nodes[node]['color']
        box = FancyBboxPatch((x - 0.02, y - 0.02), 0.04, 0.04,
                             boxstyle="round,pad=0.005",
                             facecolor=color, edgecolor='black',
                             transform=ax2.transData)
        ax2.add_patch(box)

    # Draw metabolites
    nx.draw_networkx_nodes(G, pos2, nodelist=metabolite_nodes,
                          node_color=metabolite_colors,
                          node_shape='o', node_size=300, alpha=0.8, ax=ax2)

    if target_ffa_nodes:
        nx.draw_networkx_nodes(G, pos2, nodelist=target_ffa_nodes,
                              node_color='#FFD700',
                              node_shape='o', node_size=500,
                              edgecolors='red', linewidths=2, ax=ax2)

    nx.draw_networkx_labels(G, pos2, target_labels, font_size=8, font_weight='bold', ax=ax2)

    ax2.legend(handles=legend_elements, loc='upper left', fontsize=10)
    ax2.set_title('FFA Metabolic Network (Alternative Layout)', fontsize=16, fontweight='bold')
    ax2.axis('off')

    filename2 = "ffa_networkx_hierarchical.png"
    output_path2 = osp.join(ffa_dir, filename2)
    plt.tight_layout()
    plt.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved hierarchical version to: {output_path2}")

    plt.show()

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE!")
    print("=" * 80)
    print(f"\nCreated custom NetworkX visualizations of the FFA pathway")
    print(f"Files saved to: {ASSET_IMAGES_DIR}")

    return output_path, output_path2


if __name__ == "__main__":
    path1, path2 = create_networkx_ffa_visualization()