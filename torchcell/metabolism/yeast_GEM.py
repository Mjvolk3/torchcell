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
    gene_set: Optional[GeneSet] = field(default=None)
    _model: Optional[cobra.Model] = field(default=None, init=False)
    _compound_graph: Optional[nx.DiGraph] = field(default=None, init=False)
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

            # Create edge for each gene combination
            for idx, genes in enumerate(gene_combinations):
                if not genes:
                    continue

                # Check against the gene_set if provided
                if self.gene_set is not None:
                    genes_not_in_set = genes - self.gene_set

                    if genes_not_in_set == genes:
                        # All genes are outside the gene_set, skip the edge
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
                        f"stoich_coefficient-{k}": -v for k, v in metabolites.items()
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
        # edges_kwargs={
        #     "edgecolors": "black",
        #     "facecolors": lambda e: (
        #         "lightblue"
        #         if H_sub.edges[e].properties["direction"] == "forward"
        #         else "lightgreen"
        #     ),
        #     "alpha": 0.1,
        #     "linewidths": 2.0,
        # },
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
):
    """
    Plot a bipartite network visualization of genes to metabolites.
    If reaction_id is provided, only plot that specific reaction's network.
    """
    import networkx as nx

    # Create bipartite graph
    B = nx.Graph()

    # Get the hypergraph
    H = yeast_gem.reaction_map

    # Filter edges if reaction_id is provided
    edges_to_process = {}
    if reaction_id:
        for eid, props in H.edges.properties.items():
            if props["reaction_id"] == reaction_id:
                edges_to_process[eid] = H.edges.elements[eid]
    else:
        edges_to_process = H.edges.elements

    # Add nodes and edges
    genes_set = set()
    metabolites_set = set()

    # First pass: collect all nodes
    for eid, edge_data in edges_to_process.items():
        edge_props = H.edges[eid].properties
        genes = edge_props["genes"]
        metabolites = edge_data

        genes_set.update(genes)
        metabolites_set.update(metabolites)

    # Add nodes with proper bipartite attribute
    B.add_nodes_from(genes_set, bipartite=0, node_type="gene")
    B.add_nodes_from(metabolites_set, bipartite=1, node_type="metabolite")

    # Second pass: add edges
    for eid, edge_data in edges_to_process.items():
        edge_props = H.edges[eid].properties
        genes = edge_props["genes"]
        metabolites = edge_data
        direction = edge_props["direction"]

        # Add edges between each gene and each metabolite
        for gene in genes:
            for metabolite in metabolites:
                # Edge color based on whether metabolite is reactant or product
                if metabolite in edge_props["reactants"]:
                    edge_type = "reactant"
                else:
                    edge_type = "product"

                B.add_edge(gene, metabolite, edge_type=edge_type, direction=direction)

    # Create layout
    # Separate layout for genes and metabolites
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
        node_color="lightblue",
        node_size=1000,
        alpha=0.7,
        label="Genes",
    )

    nx.draw_networkx_nodes(
        B,
        pos,
        nodelist=metabolite_nodes,
        node_color="lightgreen",
        node_size=1000,
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

    # Add labels
    labels = {node: node for node in B.nodes()}
    nx.draw_networkx_labels(B, pos, labels, font_size=8)

    plt.title(
        "Bipartite Network: Genes to Metabolites"
        + (f" for {reaction_id}" if reaction_id else ""),
        pad=20,
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
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
    
    # Plot bipartite network for specific reactions
    for i in range(5):
        reaction = list(yeast_gem.model.reactions)[i]
        plot_bipartite_network(
            yeast_gem,
            reaction_id=reaction.id,
            output_path=osp.join(ASSET_IMAGES_DIR, f"bipartite_network_{reaction.id}.png")
        )
    
    # Plot full bipartite network
    plot_bipartite_network(
        yeast_gem,
        output_path=osp.join(ASSET_IMAGES_DIR, "full_bipartite_network.png")
    )


if __name__ == "__main__":
    main_bipartite()
    # main_with_gene_set()
