# torchcell/metabolism/yeast_GEM
# [[torchcell.metabolism.yeast_GEM]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/metabolism/yeast_GEM
# Test file: tests/torchcell/metabolism/test_yeast_GEM.py

import os.path as osp
import os
import zipfile
import requests
import cobra
import networkx as nx
from typing import Optional
from attrs import define, field
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import re
import hypernetx as hnx
from torchcell.sequence import GeneSet


@define
class YeastGEM:
    root: str = field(default="data/torchcell/yeast-GEM")
    version: str = field(default="9.0.2")
    induced_gene_set: Optional[GeneSet] = field(default=None)
    _model: Optional[cobra.Model] = field(default=None, init=False)
    _compound_graph: Optional[nx.DiGraph] = field(default=None, init=False)
    _gene_set: Optional[GeneSet] = field(default=None, init=False)
    _bipartite_graph: Optional[nx.Graph] = field(default=None, init=False)
    model_dir: str = field(init=False)

    def __attrs_post_init__(self):
        self.model_dir = osp.join(self.root, f"yeast-GEM-{self.version}")
        self._download()

    def _download(self) -> None:
        if osp.exists(self.model_dir):
            return

        os.makedirs(self.root, exist_ok=True)
        url = f"https://github.com/SysBioChalmers/yeast-GEM/archive/refs/tags/v{self.version}.zip"
        response = requests.get(url)
        response.raise_for_status()

        zip_path = osp.join(self.root, f"yeast-GEM-{self.version}.zip")
        with open(zip_path, "wb") as f:
            f.write(response.content)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(self.root)

        os.remove(zip_path)

    @property
    def model(self) -> cobra.Model:
        if self._model is None:
            model_path = osp.join(self.model_dir, "model", "yeast-GEM.xml")
            self._model = cobra.io.read_sbml_model(model_path)
        return self._model

    def _parse_gene_combinations(self, rule: str) -> list[set[str]]:
        """Parse gene rule into list of AND-connected gene sets."""
        # Might be able to do this more elegantly using cobra
        if not rule or rule == "":
            return [set()]

        # Remove outer parentheses and split by 'or'
        terms = re.split(" or ", rule.replace("(", "").replace(")", ""))

        # For each OR term, create a set of AND-connected genes
        combinations = []
        for term in terms:
            genes = set(gene.strip() for gene in term.split(" and "))
            combinations.append(genes)

        return combinations

    @property
    def reaction_map(self) -> hnx.Hypergraph:
        """Create a hypergraph where edges represent reactions with gene combinations."""
        # Dictionary to store edges for the hypergraph
        edge_dict = {}
        edge_props = {}

        # Process each reaction
        for reaction in self.model.reactions:
            # Get metabolites and their roles
            metabolites = {}
            reactants = []
            products = []

            for m, coef in reaction.metabolites.items():
                if coef < 0:
                    reactants.append(m.id)
                else:
                    products.append(m.id)
                metabolites[m.id] = coef

            # Parse gene combinations
            gene_combinations = self._parse_gene_combinations(
                reaction.gene_reaction_rule
            )

            # Use empty set for no-gene reactions instead of skipping them
            if gene_combinations == [set()]:
                # Create the no-gene edge
                base_edge_id = f"{reaction.id}_noGene"

                # Create forward edge
                fwd_edge_id = f"{base_edge_id}_fwd"
                edge_dict[fwd_edge_id] = list(metabolites.keys())
                stoichiometry = {
                    f"stoich_coefficient-{k}": v for k, v in metabolites.items()
                }
                edge_props[fwd_edge_id] = {
                    "genes": set(),  # Empty set for no genes
                    "reaction_id": reaction.id,
                    "direction": "forward",
                    "reactants": reactants,
                    "products": products,
                    "equation": reaction.reaction,
                    "reversibility": reaction.reversibility,
                    "stoichiometry": list(stoichiometry.values()),
                    **stoichiometry,
                }

                # Create reverse edge if reaction is reversible
                if reaction.reversibility:
                    rev_edge_id = f"{base_edge_id}_rev"
                    edge_dict[rev_edge_id] = list(metabolites.keys())
                    stoichiometry = {
                        f"stoich_coefficient-{k}": -v for k, v in metabolites.items()
                    }
                    edge_props[rev_edge_id] = {
                        "genes": set(),  # Empty set for no genes
                        "reaction_id": reaction.id,
                        "direction": "reverse",
                        "reactants": products,
                        "products": reactants,
                        "equation": reaction.reaction,
                        "reversibility": reaction.reversibility,
                        "stoichiometry": list(stoichiometry.values()),
                        **stoichiometry,
                    }
            else:
                # Regular case with gene associations
                for idx, genes in enumerate(gene_combinations):
                    if not genes:
                        continue

                    # Check against the induced_gene_set if provided
                    if self.induced_gene_set is not None:
                        genes_not_in_set = genes - self.induced_gene_set

                        if genes_not_in_set == genes:
                            # All genes are outside the induced_gene_set, skip the edge
                            continue
                        elif genes_not_in_set:
                            # Convert to a standard set for proper string formatting
                            print(
                                f"Warning: Partial gene set overlap for edge {reaction.id}_comb{idx}. "
                                f"Genes not in set: {set(genes_not_in_set)}. "
                                f"Full gene set for this reaction: {set(genes)}"
                            )

                    # Create base edge ID for this gene combination
                    base_edge_id = f"{reaction.id}_comb{idx}"

                    # Always create forward edge
                    fwd_edge_id = f"{base_edge_id}_fwd"
                    edge_dict[fwd_edge_id] = list(metabolites.keys())
                    stoichiometry = {
                        f"stoich_coefficient-{k}": v for k, v in metabolites.items()
                    }
                    edge_props[fwd_edge_id] = {
                        "genes": genes,
                        "reaction_id": reaction.id,
                        "direction": "forward",
                        "reactants": reactants,
                        "products": products,
                        "equation": reaction.reaction,
                        "reversibility": reaction.reversibility,
                        "stoichiometry": list(stoichiometry.values()),
                        **stoichiometry,
                    }

                    # Create reverse edge if reaction is reversible
                    if reaction.reversibility:
                        rev_edge_id = f"{base_edge_id}_rev"
                        edge_dict[rev_edge_id] = list(metabolites.keys())
                        stoichiometry = {
                            f"stoich_coefficient-{k}": -v
                            for k, v in metabolites.items()
                        }
                        edge_props[rev_edge_id] = {
                            "genes": genes,
                            "reaction_id": reaction.id,
                            "direction": "reverse",
                            "reactants": products,
                            "products": reactants,
                            "equation": reaction.reaction,
                            "reversibility": reaction.reversibility,
                            "stoichiometry": list(stoichiometry.values()),
                            **stoichiometry,
                        }

        # Create and return the hypergraph
        return hnx.Hypergraph(edge_dict, edge_properties=edge_props)

    @property
    def bipartite_graph(self) -> nx.DiGraph:
        """Returns a directed bipartite graph representation of the metabolic network.

        The graph has two types of nodes:
        - Reactions (node_type="reaction")
        - Metabolites (node_type="metabolite")

        Edges connect reactions to metabolites with a directed structure:
        - Reaction → Metabolite with edge_type="product" (metabolite is produced)
        - Reaction → Metabolite with edge_type="reactant" (metabolite is consumed)
        
        The edge direction is always from reaction to metabolite, with the 
        edge_type indicating whether the metabolite is a reactant or product.

        Returns:
            nx.DiGraph: A directed bipartite graph
        """
        if getattr(self, "_bipartite_graph", None) is None:
            # Create directed bipartite graph
            B = nx.DiGraph()

            # Process all reactions in the model
            for reaction in self.model.reactions:
                # Parse gene combinations
                gene_combinations = self._parse_gene_combinations(
                    reaction.gene_reaction_rule
                )

                # Extract reactants and products
                reactants = []
                products = []
                for metabolite, coefficient in reaction.metabolites.items():
                    if coefficient < 0:
                        reactants.append(metabolite.id)
                    else:
                        products.append(metabolite.id)

                # For reactions with gene associations
                if gene_combinations and gene_combinations != [set()]:
                    # For each gene combination, create a separate reaction node
                    for idx, genes in enumerate(gene_combinations):
                        # Check against induced_gene_set if provided
                        if self.induced_gene_set is not None:
                            genes_not_in_set = genes - self.induced_gene_set
                            if genes_not_in_set == genes:
                                # All genes outside the set, skip
                                continue

                        # Create base node ID for this gene combination
                        base_node_id = f"{reaction.id}_comb{idx}"
                        
                        # Add forward reaction node
                        fwd_node_id = f"{base_node_id}_fwd"
                        B.add_node(
                            fwd_node_id,
                            node_type="reaction",
                            reaction_id=reaction.id,
                            direction="forward",
                            genes=genes,
                            equation=reaction.reaction,
                            reversibility=reaction.reversibility,
                            reactants=reactants,
                            products=products,
                        )
                        
                        # Add metabolite nodes and connect in unified format
                        self._add_metabolite_edges_unified(B, reaction, fwd_node_id, "forward")
                        
                        # If reversible, add reverse reaction and connect
                        if reaction.reversibility:
                            rev_node_id = f"{base_node_id}_rev"
                            B.add_node(
                                rev_node_id,
                                node_type="reaction",
                                reaction_id=reaction.id,
                                direction="reverse",
                                genes=genes,
                                equation=reaction.reaction,
                                reversibility=reaction.reversibility,
                                reactants=products,  # Swapped for reverse direction
                                products=reactants,  # Swapped for reverse direction
                            )
                            
                            # Connect in unified format
                            self._add_metabolite_edges_unified(B, reaction, rev_node_id, "reverse")
                else:
                    # For reactions without gene associations
                    fwd_node_id = f"{reaction.id}_noGene_fwd"
                    B.add_node(
                        fwd_node_id,
                        node_type="reaction",
                        reaction_id=reaction.id,
                        direction="forward",
                        genes=set(),  # Empty set for no genes
                        equation=reaction.reaction,
                        reversibility=reaction.reversibility,
                        reactants=reactants,
                        products=products,
                    )
                    
                    # Connect in unified format
                    self._add_metabolite_edges_unified(B, reaction, fwd_node_id, "forward")
                    
                    # If reversible, add reverse reaction
                    if reaction.reversibility:
                        rev_node_id = f"{reaction.id}_noGene_rev"
                        B.add_node(
                            rev_node_id,
                            node_type="reaction",
                            reaction_id=reaction.id,
                            direction="reverse",
                            genes=set(),
                            equation=reaction.reaction,
                            reversibility=reaction.reversibility,
                            reactants=products,  # Swapped for reverse direction
                            products=reactants,  # Swapped for reverse direction
                        )
                        
                        # Connect in unified format
                        self._add_metabolite_edges_unified(B, reaction, rev_node_id, "reverse")
            
            self._bipartite_graph = B

        return self._bipartite_graph

    def _add_metabolite_edges_unified(self, graph, reaction, reaction_node_id, direction):
        """Helper function to add unified directed edges between reaction and metabolites."""
        # Add metabolite nodes if they don't exist
        for metabolite in reaction.metabolites:
            if not graph.has_node(metabolite.id):
                graph.add_node(metabolite.id, node_type="metabolite")
        
        # Process based on direction (forward or reverse)
        for metabolite, coefficient in reaction.metabolites.items():
            if direction == "forward":
                # Always add edge from reaction to metabolite
                if coefficient < 0:
                    # Reactant (reaction consumes this metabolite)
                    graph.add_edge(
                        reaction_node_id,
                        metabolite.id,
                        edge_type="reactant",
                        direction="forward",
                        stoichiometry=abs(coefficient),
                        reaction_id=reaction.id,
                    )
                else:
                    # Product (reaction produces this metabolite)
                    graph.add_edge(
                        reaction_node_id,
                        metabolite.id,
                        edge_type="product",
                        direction="forward",
                        stoichiometry=coefficient,
                        reaction_id=reaction.id,
                    )
            else:
                # Reverse direction: swap reactants and products
                if coefficient < 0:
                    # Original reactant becomes product in reverse direction
                    graph.add_edge(
                        reaction_node_id,
                        metabolite.id,
                        edge_type="product",
                        direction="reverse",
                        stoichiometry=abs(coefficient),
                        reaction_id=reaction.id,
                    )
                else:
                    # Original product becomes reactant in reverse direction
                    graph.add_edge(
                        reaction_node_id,
                        metabolite.id,
                        edge_type="reactant",
                        direction="reverse",
                        stoichiometry=coefficient,
                        reaction_id=reaction.id,
                    )

    @property
    def gene_set(self) -> GeneSet:
        """Returns a GeneSet containing all genes in the model."""
        if self._gene_set is None:
            # Use cobra's built-in genes property which properly parses GPRs
            all_genes = {gene.id for gene in self.model.genes}
            self._gene_set = GeneSet(all_genes)
        return self._gene_set


def plot_reaction_map(
    yeast_gem: YeastGEM, reaction_id: str, output_path: str = "reaction_map.png"
):
    """Plot hypergraph for a specific reaction."""
    H = yeast_gem.reaction_map

    # Filter edges for this reaction and maintain properties
    edges_dict = {}
    edge_properties = {}

    for eid, data in H.edges.elements.items():
        if H.edges[eid].properties["reaction_id"] == reaction_id:
            edges_dict[eid] = data
            edge_properties[eid] = H.edges[eid].properties

    if not edges_dict:
        print(f"No edges found for reaction {reaction_id}")
        return

    # Create subhypergraph with properties
    H_sub = hnx.Hypergraph(edges_dict, edge_properties=edge_properties)

    fig = plt.figure(figsize=(15, 10))
    ax = plt.gca()

    # Function to determine color based on direction
    def get_color(e):
        return (
            "orange"
            if H_sub.edges[e].properties["direction"] == "forward"
            else "lightblue"
        )

    # Draw hypergraph
    hnx.draw(
        H_sub,
        with_node_labels=True,
        with_edge_labels=True,
        node_labels_kwargs={"fontsize": 8},
        edge_labels_kwargs={"fontsize": 6},
        edges_kwargs={
            "linewidths": 2,
            "edgecolors": lambda e: get_color(e),  # Edge color now matches face color
            "facecolors": lambda e: get_color(e),
            "alpha": 0.2,
        },
        layout_kwargs={"seed": 42},
        ax=ax,
    )

    # Add detailed edge information
    y_pos = 0.8
    for eid in H_sub.edges:
        edge_props = H_sub.edges[eid].properties
        genes_str = " and ".join(sorted(edge_props["genes"]))
        direction = edge_props["direction"]
        reactants = ", ".join(edge_props["reactants"])
        products = ", ".join(edge_props["products"])

        edge_text = (
            f"Edge: {eid}\n"
            f"Direction: {direction}\n"
            f"Genes: {genes_str}\n"
            f"Reactants: {reactants}\n"
            f"Products: {products}"
        )
        plt.text(
            1.2,
            y_pos,
            edge_text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        y_pos -= 0.1

    # Add reaction information
    reaction = yeast_gem.model.reactions.get_by_id(reaction_id)
    info_text = (
        f"Reaction: {reaction_id}\n"
        f"Equation: {reaction.reaction}\n"
        f"Gene Rule: {reaction.gene_reaction_rule}\n"
        f"Reversible: {reaction.reversibility}"
    )
    plt.text(
        0.95,
        0.95,
        info_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        fontsize=8,
    )

    # Add legend with explicit patches
    legend_elements = [
        Patch(facecolor="orange", edgecolor="orange", alpha=0.2, label="Forward"),
        Patch(facecolor="lightblue", edgecolor="lightblue", alpha=0.2, label="Reverse"),
    ]
    # Place legend outside the main plot
    # ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.1, 0.5))
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.1, 0.5))
    # ax.legend(handles=legend_elements, loc="upper right")

    plt.title(f"Reaction Map for {reaction_id}", pad=20)
    plt.axis("off")
    plt.subplots_adjust(right=0.7)

    # Ensure the figure is rendered with the legend
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Reaction map visualization saved to {output_path}")


def plot_full_network(yeast_gem: YeastGEM, output_path: str = "full_network.png"):
    """Plot the entire metabolic network structure without labels."""
    H = yeast_gem.reaction_map

    # Print network statistics
    n_nodes = len(H.nodes)
    n_edges = len(H.edges)
    n_reversible = sum(1 for e in H.edges if H.edges[e].properties["reversibility"])

    print(f"\nNetwork Statistics:")
    print(f"Number of metabolites (nodes): {n_nodes}")
    print(f"Number of reactions (edges): {n_edges}")
    print(f"Number of reversible reactions: {n_reversible}")

    # Create visualization
    fig = plt.figure(figsize=(40, 40))
    ax = plt.gca()

    # Draw hypergraph with minimal decoration
    hnx.draw(
        H,
        with_node_labels=False,
        with_edge_labels=False,
        nodes_kwargs={
            "facecolors": "black",
            "alpha": 0.6,
            "linewidths": 0,
            "sizes": [20 for _ in range(len(H.nodes))],
        },
        edges_kwargs={
            "linewidths": 0.5,
            "edgecolors": "black",
            "facecolors": lambda e: (
                "lightblue"
                if H.edges[e].properties["direction"] == "forward"
                else "lightgreen"
            ),
            "alpha": 0.1,
        },
        layout_kwargs={
            "seed": 42,
            "k": 50,  # Dramatically increased spacing
            "iterations": 200,  # More iterations for better convergence
            "weight": None,  # Don't use edge weights in layout
            "scale": 20,  # Scale up the final layout
        },
        ax=ax,
    )

    # Simple legend
    legend_elements = [
        Patch(facecolor="lightblue", alpha=0.3, label="Forward"),
        Patch(facecolor="lightgreen", alpha=0.3, label="Reverse"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1, 1))

    plt.title("Yeast Metabolic Network Structure", pad=20, fontsize=16)
    plt.axis("off")

    # Save with high quality
    plt.savefig(output_path, dpi=300, bbox_inches="tight", format="png")
    plt.close()

    print(f"\nNetwork visualization saved to {output_path}")


def plot_random_network(
    yeast_gem: YeastGEM,
    n_edges: int = 10,
    output_path: str = "random_network.png",
    layout="spring",
):
    import random
    from networkx.drawing.layout import (
        spring_layout,
        spectral_layout,
        kamada_kawai_layout,
    )

    H = yeast_gem.reaction_map

    sampled_edges = random.sample(list(H.edges.elements.keys()), n_edges)
    edges_dict = {eid: H.edges.elements[eid] for eid in sampled_edges}
    edge_properties = {eid: H.edges[eid].properties for eid in sampled_edges}

    H_sub = hnx.Hypergraph(edges_dict, edge_properties=edge_properties)

    plt.figure(figsize=(20, 20))

    if layout == "spring":
        layout_kwargs = {"k": 10, "iterations": 1000}
        layout = spring_layout
    elif layout == "spectral":
        layout_kwargs = {}
        layout = spectral_layout
    elif layout == "kamada_kawai":
        layout_kwargs = {}
        layout = kamada_kawai_layout

    hnx.draw(
        H_sub,
        with_node_labels=False,
        with_edge_labels=False,
        nodes_kwargs={"alpha": 0.4},
        layout=layout,
        layout_kwargs=layout_kwargs,
    )

    plt.axis("off")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_bipartite_network(
    yeast_gem: YeastGEM,
    reaction_id: str = None,
    output_path: str = "bipartite_network.png",
    figsize=(20, 15),
    show_labels: bool = False,
):
    """
    Plot a bipartite network visualization of genes to metabolites.
    If reaction_id is provided, only plot that specific reaction's network.
    """
    import networkx as nx

    # Get the bipartite graph
    B = yeast_gem.bipartite_graph

    # Filter graph if reaction_id is provided
    if reaction_id:
        # Create a subgraph with only edges for the specified reaction
        edges_to_keep = [
            (u, v) for u, v, d in B.edges(data=True) if d["reaction_id"] == reaction_id
        ]
        # Create the subgraph
        B = nx.Graph(B.edge_subgraph(edges_to_keep))
        # Need to preserve node attributes in the subgraph
        for node in B.nodes():
            if node in yeast_gem.bipartite_graph.nodes():
                B.nodes[node].update(yeast_gem.bipartite_graph.nodes[node])

    # Create layout
    pos = nx.spring_layout(B, k=2.0)

    # Draw the network
    plt.figure(figsize=figsize)

    # Draw nodes
    gene_nodes = [n for n, d in B.nodes(data=True) if d["node_type"] == "gene"]
    metabolite_nodes = [
        n for n, d in B.nodes(data=True) if d["node_type"] == "metabolite"
    ]

    nx.draw_networkx_nodes(
        B,
        pos,
        nodelist=gene_nodes,
        node_color="#1f77b4",
        node_size=10,
        alpha=0.7,
        label="Genes",
    )

    nx.draw_networkx_nodes(
        B,
        pos,
        nodelist=metabolite_nodes,
        node_color="#2ca02c",
        node_size=10,
        alpha=0.7,
        label="Metabolites",
    )

    # Draw edges with different colors based on type
    edges_reactant_fwd = [
        (u, v)
        for (u, v, d) in B.edges(data=True)
        if d["edge_type"] == "reactant" and d["direction"] == "forward"
    ]
    edges_product_fwd = [
        (u, v)
        for (u, v, d) in B.edges(data=True)
        if d["edge_type"] == "product" and d["direction"] == "forward"
    ]
    edges_reactant_rev = [
        (u, v)
        for (u, v, d) in B.edges(data=True)
        if d["edge_type"] == "reactant" and d["direction"] == "reverse"
    ]
    edges_product_rev = [
        (u, v)
        for (u, v, d) in B.edges(data=True)
        if d["edge_type"] == "product" and d["direction"] == "reverse"
    ]

    nx.draw_networkx_edges(
        B,
        pos,
        edgelist=edges_reactant_fwd,
        edge_color="red",
        alpha=0.4,
        label="Reactant (Forward)",
    )
    nx.draw_networkx_edges(
        B,
        pos,
        edgelist=edges_product_fwd,
        edge_color="blue",
        alpha=0.4,
        label="Product (Forward)",
    )
    nx.draw_networkx_edges(
        B,
        pos,
        edgelist=edges_reactant_rev,
        edge_color="orange",
        alpha=0.4,
        label="Reactant (Reverse)",
    )
    nx.draw_networkx_edges(
        B,
        pos,
        edgelist=edges_product_rev,
        edge_color="purple",
        alpha=0.4,
        label="Product (Reverse)",
    )

    # Add labels only if requested
    if show_labels:
        labels = {node: node for node in B.nodes()}
        nx.draw_networkx_labels(B, pos, labels, font_size=8)

    # Title
    if not show_labels:
        plt.title("")
    else:
        plt.title(
            "Bipartite Network: Genes to Metabolites"
            + (f" for {reaction_id}" if reaction_id else ""),
            pad=20,
        )

    # Enhanced legend with larger font size and marker size
    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=16,
        markerscale=3,
        frameon=True,
        fancybox=True,
        shadow=True,
        borderpad=1,
    )

    plt.axis("off")

    # Save with high quality
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Print some statistics
    print(f"\nNetwork Statistics:")
    print(f"Number of genes: {len(gene_nodes)}")
    print(f"Number of metabolites: {len(metabolite_nodes)}")
    print(f"Number of edges: {B.number_of_edges()}")


def main():
    from dotenv import load_dotenv

    load_dotenv()

    ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
    yeast_gem = YeastGEM()

    # # PLOTTING
    # # Plot first 4 reactions
    # for i in range(5):
    #     reaction = list(yeast_gem.model.reactions)[i]
    #     print(f"Reaction ID: {reaction.id}")
    #     print(f"Reaction: {reaction.reaction}")
    #     print(f"Gene rule: {reaction.gene_reaction_rule}")

    #     # Plot using standalone function
    #     plot_reaction_map(
    #         yeast_gem,
    #         reaction.id,
    #         osp.join(ASSET_IMAGES_DIR, f"reaction_map_{reaction.id}.png"),
    #     )

    # # Plot full network
    # plot_full_network(
    #     yeast_gem, osp.join(ASSET_IMAGES_DIR, "yeast_metabolic_network.png")
    # )

    # Plot random networks
    import random
    plot_full_network(
        yeast_gem, osp.join(ASSET_IMAGES_DIR, "yeast_metabolic_network.png")
    )
    random.seed(42)  # For reproducibility
    layout = "spring"
    for n_edges in [5, 10, 20, 50, 100, 1000, 4881]:
        plot_random_network(
            yeast_gem,
            n_edges=n_edges,
            layout="spring",
            output_path=osp.join(
                ASSET_IMAGES_DIR, f"yeast_metabolic_random_nc_{layout}_{n_edges}.png"
            ),
        )


def main_with_gene_set():
    from dotenv import load_dotenv
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

    # Setup dataset (unchanged)
    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")

    # Without edge drop
    yeast_gem = YeastGEM()
    H = yeast_gem.reaction_map
    print(f"H num edges without gene_set edge drop: {len(H.edges)}")

    # Dataset setup
    genome = SCerevisiaeGenome(osp.join(DATA_ROOT, "data/sgd/genome"))
    genome.drop_chrmt()

    yeast_gem = YeastGEM(gene_set=genome.gene_set)
    H = yeast_gem.reaction_map
    print(f"H num edges with gene_set edge drop: {len(H.edges)}")


def main_bipartite():
    from dotenv import load_dotenv

    load_dotenv()

    ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
    yeast_gem = YeastGEM()

    # Plot full bipartite network
    plot_bipartite_network(
        yeast_gem, output_path=osp.join(ASSET_IMAGES_DIR, "full_bipartite_network.png")
    )


def sanity_check_metabolic_networks(yeast_gem: YeastGEM, num_reactions: int = 3):
    """
    Print detailed information about a few random reactions and their metabolites
    as a sanity check for the reaction_map and bipartite_graph representations.

    Args:
        yeast_gem: The YeastGEM instance
        num_reactions: Number of random reactions to check
    """
    import random

    # Get random reactions
    all_reactions = list(yeast_gem.model.reactions)
    sample_reactions = random.sample(
        all_reactions, min(num_reactions, len(all_reactions))
    )

    # Get both network representations
    H = yeast_gem.reaction_map
    B = yeast_gem.bipartite_graph

    print("\n===== YeastGEM Metabolic Network Sanity Check =====")

    for rxn in sample_reactions:
        print(f"\n\n{'='*50}")
        print(f"REACTION: {rxn.id}")
        print(f"{'='*50}")

        # Print COBRA model information
        print(f"\n--- COBRA Model Information ---")
        print(f"Equation: {rxn.reaction}")
        print(f"Reversible: {rxn.reversibility}")
        print(f"Gene rule: {rxn.gene_reaction_rule}")
        print(f"Genes involved: {', '.join(gene.id for gene in rxn.genes)}")

        # Print metabolite details
        print(f"\nMetabolites:")
        for metabolite, coefficient in rxn.metabolites.items():
            role = "Reactant" if coefficient < 0 else "Product"
            print(f"  - {metabolite.id} ({role}, coefficient: {coefficient})")
            print(f"    Name: {metabolite.name}")
            print(f"    Formula: {metabolite.formula}")
            print(f"    Compartment: {metabolite.compartment}")

        # Check reaction_map (hypergraph)
        print(f"\n--- Reaction Map (Hypergraph) Information ---")
        reaction_edges = []

        # Correct way to iterate through edges and check properties
        for eid in H.edges:
            edge_props = H.edges[eid].properties
            if edge_props.get("reaction_id") == rxn.id:
                reaction_edges.append(eid)

        print(f"Number of hyperedges for this reaction: {len(reaction_edges)}")

        for edge_id in reaction_edges:
            edge_props = H.edges[edge_id].properties
            print(f"\nEdge ID: {edge_id}")
            print(f"Direction: {edge_props['direction']}")
            print(f"Genes: {', '.join(sorted(edge_props['genes']))}")
            print(f"Reactants: {', '.join(edge_props['reactants'])}")
            print(f"Products: {', '.join(edge_props['products'])}")

        # Check bipartite_graph with corrected implementation
        print(f"\n--- Bipartite Graph Information ---")
        rxn_nodes = [
            node
            for node in B.nodes()
            if B.nodes[node].get("node_type") == "reaction"
            and B.nodes[node].get("reaction_id") == rxn.id
        ]

        print(f"Number of reaction nodes for this reaction: {len(rxn_nodes)}")

        # Show reaction node details
        for i, node in enumerate(rxn_nodes):
            print(f"\nReaction node {i+1}: {node}")
            node_data = B.nodes[node]
            print(f"  Direction: {node_data.get('direction')}")
            genes = node_data.get("genes", set())
            if genes:
                print(f"  Genes: {', '.join(sorted(genes))}")
            else:
                print(f"  Genes: None")

            # Get connected metabolites
            connected_metabolites = list(B.neighbors(node))
            reactants = [
                m
                for m in connected_metabolites
                if B.edges[node, m].get("edge_type") == "reactant"
            ]
            products = [
                m
                for m in connected_metabolites
                if B.edges[node, m].get("edge_type") == "product"
            ]

            print(f"  Reactants: {', '.join(reactants)}")
            print(f"  Products: {', '.join(products)}")

            # Print a few sample edges
            print(f"\n  Sample edges:")
            for j, metabolite in enumerate(
                connected_metabolites[:3]
            ):  # Show at most 3 edges
                print(f"    {j+1}. {node} -- {metabolite}")
                print(f"       Edge type: {B.edges[node, metabolite].get('edge_type')}")
                print(
                    f"       Stoichiometry: {B.edges[node, metabolite].get('stoichiometry')}"
                )

            if len(connected_metabolites) > 3:
                print(f"    ... and {len(connected_metabolites) - 3} more metabolites")

    # Print overall statistics
    print("\n===== Overall Statistics =====")
    print(f"Total reactions in model: {len(yeast_gem.model.reactions)}")
    print(f"Total metabolites in model: {len(yeast_gem.model.metabolites)}")
    print(f"Total genes in model: {len(yeast_gem.model.genes)}")
    print(f"Total edges in hypergraph: {len(H.edges)}")
    print(f"Total nodes in hypergraph: {len(H.nodes)}")

    # Count bipartite nodes correctly
    reaction_nodes = [n for n, d in B.nodes(data=True) if d.get("bipartite") == 0]
    metabolite_nodes = [n for n, d in B.nodes(data=True) if d.get("bipartite") == 1]

    print(f"Total edges in bipartite graph: {B.number_of_edges()}")
    print(f"Total nodes in bipartite graph: {B.number_of_nodes()}")
    print(f"Reaction nodes: {len(reaction_nodes)}")
    print(f"Metabolite nodes: {len(metabolite_nodes)}")

    # Verify bipartite property
    rxn_set = set(reaction_nodes)
    met_set = set(metabolite_nodes)

    # Sanity check: every edge should connect a reaction to a metabolite
    mixed_edges = [
        (u, v)
        for u, v in B.edges()
        if (u in rxn_set and v in rxn_set) or (u in met_set and v in met_set)
    ]

    if mixed_edges:
        print(f"WARNING: {len(mixed_edges)} edges connect nodes of the same type!")
        # Print a few examples
        for i, (u, v) in enumerate(mixed_edges[:3]):
            print(f"  {i+1}. {u} -- {v}")
            print(
                f"     Node types: {B.nodes[u].get('node_type')} -- {B.nodes[v].get('node_type')}"
            )

        if len(mixed_edges) > 3:
            print(f"  ... and {len(mixed_edges) - 3} more problematic edges")
    else:
        print(
            "✓ All edges connect reactions to metabolites (bipartite property verified)"
        )

    # Check for isolated nodes
    isolated = list(nx.isolates(B))
    if isolated:
        print(f"WARNING: {len(isolated)} isolated nodes found!")
        # Print node types of a few examples
        for i, node in enumerate(isolated[:3]):
            print(f"  {i+1}. {node} (Type: {B.nodes[node].get('node_type')})")

        if len(isolated) > 3:
            print(f"  ... and {len(isolated) - 3} more isolated nodes")
    else:
        print("✓ No isolated nodes found")


def analyze_reactions_without_genes(yeast_gem: YeastGEM):
    """
    Analyze reactions without gene associations in the YeastGEM model.

    Args:
        yeast_gem: YeastGEM instance

    Returns:
        dict: Statistics and lists of reactions without genes
    """
    # Track reactions without genes
    no_gene_rule = []  # Empty gene_reaction_rule
    no_genes_obj = []  # Empty genes list
    empty_gene_comb = []  # _parse_gene_combinations returns [set()]

    # Analyze all reactions
    for reaction in yeast_gem.model.reactions:
        # Method 1: Check gene_reaction_rule string
        if not reaction.gene_reaction_rule or reaction.gene_reaction_rule == "":
            no_gene_rule.append(reaction.id)

        # Method 2: Check genes attribute
        if len(reaction.genes) == 0:
            no_genes_obj.append(reaction.id)

        # Method 3: Check parsed gene combinations
        gene_combinations = yeast_gem._parse_gene_combinations(
            reaction.gene_reaction_rule
        )
        if gene_combinations == [set()]:
            empty_gene_comb.append(reaction.id)

    # Check consistency between methods
    methods_consistent = no_gene_rule == no_genes_obj == empty_gene_comb

    # Get metabolite stats for reactions without genes
    rxn_stats = {}
    compartments = {}

    for rxn_id in no_gene_rule:
        rxn = yeast_gem.model.reactions.get_by_id(rxn_id)
        n_mets = len(rxn.metabolites)

        # Track reaction types by number of metabolites
        rxn_stats[n_mets] = rxn_stats.get(n_mets, 0) + 1

        # Track compartments involved
        for met in rxn.metabolites:
            comp = met.compartment
            compartments[comp] = compartments.get(comp, 0) + 1

    # Classify reactions
    exchange_rxns = [
        r
        for r in no_gene_rule
        if len(yeast_gem.model.reactions.get_by_id(r).metabolites) == 1
    ]
    transport_rxns = [
        r
        for r in no_gene_rule
        if any(
            m1.id[:-1] == m2.id[:-1] and m1.id[-1] != m2.id[-1]
            for m1 in yeast_gem.model.reactions.get_by_id(r).metabolites
            for m2 in yeast_gem.model.reactions.get_by_id(r).metabolites
            if m1 != m2
        )
    ]
    other_rxns = [
        r for r in no_gene_rule if r not in exchange_rxns and r not in transport_rxns
    ]

    # Print summary statistics
    print("\n===== Analysis of Reactions Without Gene Associations =====")
    print(f"Total reactions in model: {len(yeast_gem.model.reactions)}")
    print(
        f"Reactions without gene rules: {len(no_gene_rule)} ({len(no_gene_rule)/len(yeast_gem.model.reactions):.1%})"
    )

    if methods_consistent:
        print("✓ All detection methods give consistent results")
    else:
        print("⚠ Inconsistency in detection methods:")
        print(f"  - Empty gene_reaction_rule: {len(no_gene_rule)}")
        print(f"  - No genes attribute: {len(no_genes_obj)}")
        print(f"  - Empty gene combinations: {len(empty_gene_comb)}")

    print("\nReaction classification:")
    print(
        f"  - Exchange reactions: {len(exchange_rxns)} ({len(exchange_rxns)/len(no_gene_rule):.1%})"
    )
    print(
        f"  - Transport reactions: {len(transport_rxns)} ({len(transport_rxns)/len(no_gene_rule):.1%})"
    )
    print(
        f"  - Other reactions: {len(other_rxns)} ({len(other_rxns)/len(no_gene_rule):.1%})"
    )

    print("\nReactions by number of metabolites:")
    for n_mets, count in sorted(rxn_stats.items()):
        print(f"  - {n_mets} metabolites: {count} reactions")

    print("\nCompartments involved:")
    for comp, count in sorted(compartments.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {comp}: {count} occurrences")

    # Print some examples of each type
    def print_examples(rxn_list, category, n=3):
        print(
            f"\nExample {category} (showing {min(n, len(rxn_list))} of {len(rxn_list)}):"
        )
        for i, rxn_id in enumerate(rxn_list[:n]):
            rxn = yeast_gem.model.reactions.get_by_id(rxn_id)
            print(f"  {i+1}. {rxn_id}: {rxn.reaction}")

    print_examples(exchange_rxns, "exchange reactions")
    print_examples(transport_rxns, "transport reactions")
    print_examples(other_rxns, "other reactions")

    # Return detailed data for further analysis if needed
    return {
        "total_reactions": len(yeast_gem.model.reactions),
        "no_gene_reactions": no_gene_rule,
        "exchange_reactions": exchange_rxns,
        "transport_reactions": transport_rxns,
        "other_reactions": other_rxns,
        "methods_consistent": methods_consistent,
    }


if __name__ == "__main__":
    # # main_bipartite()
    # # main_with_gene_set()
    # from dotenv import load_dotenv

    # load_dotenv()

    # ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
    # yeast_gem = YeastGEM()
    # yeast_gem.reaction_map
    # # sanity_check_metabolic_networks(yeast_gem)
    # # analyze_reactions_without_genes(yeast_gem)

    main()