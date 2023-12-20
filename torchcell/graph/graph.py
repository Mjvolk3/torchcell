# torchcell/graph/graph.py
# [[torchcell.graph.graph]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/graph/graph.py
# Test file: torchcell/graph/test_graph.py

import glob
import gzip
import json
import logging
import os
import os.path as osp
import pickle
import shutil
import tarfile
import time
from collections import defaultdict
from datetime import datetime
from itertools import product
from typing import Set

import gffutils
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import networkx as nx
import pandas as pd
from attrs import define, field
from Bio import Seq, SeqIO
from Bio.SeqRecord import SeqRecord
from gffutils import FeatureDB
from gffutils.feature import Feature
from matplotlib.patches import Patch
from sortedcontainers import SortedDict, SortedSet
from torch_geometric.data import download_url
from torch_geometric.utils import from_networkx
from tqdm import tqdm

from torchcell.sequence import GeneSet, Genome, ParsedGenome
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
import torchcell
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
style_file_path = osp.join(osp.dirname(torchcell.__file__), 'torchcell.mplstyle')
plt.style.use(style_file_path)

# BUG switching to genome for GO, this will create training issues with sql,
# but need genome for created graphs.
@define
class SCerevisiaeGraph:
    data_root: str = field(init=True, repr=False, default="data/sgd/genome")
    # genome: Genome = field(init=True, repr=False, default=None)
    genome: GeneSet = field(init=True, repr=False, default=None)
    json_files: list[str] = field(init=False, repr=False, default=None)
    G_raw: nx.Graph = field(init=False, repr=False, default=None)
    G_gene: nx.Graph = field(init=False, repr=False, default=None)
    G_physical: nx.Graph = field(init=False, repr=False, default=None)
    G_genetic: nx.Graph = field(init=False, repr=False, default=None)
    G_regulatory: nx.DiGraph = field(init=False, repr=False, default=None)
    G_go: nx.DiGraph = field(init=False, repr=False, default=None)
    _all_go_terms = field(init=False, repr=False, default=None)
    _go_to_genes = field(init=False, repr=False, default=None)
    # parsed_genome: ParsedGenome = field(init=True, repr=False, default=None)

    def __attrs_post_init__(self) -> None:
        self.json_files = [f"{gene}.json" for gene in self.genome.gene_set]

        graph_dir = osp.join(self.data_root, "graph")
        if not osp.exists(graph_dir):
            os.makedirs(graph_dir)
        # GO
        go_graph_path = osp.join(graph_dir, "G_go.pkl")
        if osp.exists(go_graph_path):
            self.G_go = self.load_graph("G_go")
        else:
            self.read_raw()
            self.G_go = self.create_G_go()
            # HACK save and unsave
            self.save_graph(self.G_go, "G_go")
        # # Gene Graph
        # gene_graph_path = osp.join(graph_dir, "G_gene.pkl")
        # if osp.exists(gene_graph_path):
        #     self.G_gene = self.load_graph("G_gene")
        # else:
        #     self.G_gene = self.add_gene_protein_overview(
        #         G_raw=self.G_raw, G_gene=nx.Graph()
        #     )
        #     self.save_graph(self.G_gene, "G_gene")

        # Physical Graph
        physical_graph_path = osp.join(graph_dir, "G_physical.pkl")
        if osp.exists(physical_graph_path):
            self.G_physical = self.load_graph("G_physical")
        else:
            self.read_raw()
            self.G_physical = self.add_physical_edges(
                G_raw=self.G_raw, G_physical=nx.Graph()
            )
            self.save_graph(self.G_physical, "G_physical")

        # # Genetic Graph
        # genetic_graph_path = osp.join(graph_dir, "G_genetic.pkl")
        # if osp.exists(genetic_graph_path):
        #     self.G_genetic = self.load_graph("G_genetic")
        # else:
        #     self.G_genetic = self.add_genetic_edges(
        #         G_raw=self.G_raw, G_genetic=nx.Graph()
        #     )
        #     self.save_graph(self.G_genetic, "G_genetic")

        # Regulatory Graph
        regulatory_graph_path = osp.join(graph_dir, "G_regulatory.pkl")
        if osp.exists(regulatory_graph_path):
            self.G_regulatory = self.load_graph("G_regulatory")
        else:
            self.read_raw()
            self.G_regulatory = self.add_regulatory_edges(G_raw=self.G_raw)
            self.save_graph(self.G_regulatory, "G_regulatory")

        # Clean up sql connections
        self.genome = self.parse_genome(self.genome)

    @staticmethod
    def parse_genome(genome) -> ParsedGenome:
        data = {}
        data["gene_set"] = genome.gene_set
        return ParsedGenome(**data)

    def read_raw(self):
        # Raw Graph
        # HACK Loading the raw graph is what takes a lot of time, so this should only
        # be called if the other small graphs don't exist
        # keeping for now for quick dev
        graph_dir = osp.join(self.data_root, "graph")
        if not osp.exists(graph_dir):
            os.makedirs(graph_dir)
        raw_graph_path = osp.join(graph_dir, "G_raw.pkl")
        if osp.exists(raw_graph_path):
            self.G_raw = self.load_graph("G_raw")
        else:
            self.G_raw = self.add_json_data_to_graph(
                data_root=self.data_root, json_files=self.json_files
            )
            self.save_graph(self.G_raw, "G_raw")

    def save_graph(self, graph, graph_name):
        dir = osp.join(self.data_root, "graph")
        if not osp.exists(dir):
            os.makedirs(dir)
        path = osp.join(dir, f"{graph_name}.pkl")

        with open(path, "wb") as f:
            pickle.dump(graph, f)

    def load_graph(self, graph_name):
        dir = osp.join(self.data_root, "graph")
        path = osp.join(dir, f"{graph_name}.pkl")

        if not os.path.exists(path):
            print(f"Graph file {path} not found!")
            return None

        with open(path, "rb") as f:
            graph = pickle.load(f)
        return graph

    @staticmethod
    def add_json_data_to_graph(data_root: str, json_files: list[str]) -> nx.Graph:
        G = nx.Graph()  # This is the node graph

        for i, json_file in tqdm(enumerate(json_files)):
            with open(osp.join(data_root, "genes", json_file)) as file:
                data = json.load(file)
                node_name = json_file.split(".")[0]
                G.add_node(node_name, **data)
                if i > 1000:
                    # break
                    pass

        return G

    # GO
    @property
    def all_go_terms(self) -> GeneSet:
        """Collects a SortedSet of all GO terms for all genes in genome.gene_set."""
        if self._all_go_terms is None:
            all_go_terms = GeneSet()
            missing_go_terms = set()
            go_dag = self.genome.go_dag
            for gene in self.genome.gene_set:
                go_details = self.G_raw.nodes[gene].get("go_details", [])
                for detail in go_details:
                    go_id = detail["go"]["go_id"]
                    if go_id in go_dag and not go_dag[go_id].is_obsolete:
                        all_go_terms.add(go_id)
                    else:
                        log.warning(
                            f"GO term {go_id} not found in go_dag for gene {gene}. "
                            "Most likely deprecated."
                        )
                        print(f"deprecated: {go_id}")
                        missing_go_terms.add(go_id)
            log.warning(f"Missing GO terms: {missing_go_terms}")
            self._all_go_terms = all_go_terms
            return self._all_go_terms
        else:  # already computed
            return self._all_go_terms

    @property
    def go_to_genes(self) -> SortedDict[str, GeneSet]:
        if self._go_to_genes is None:
            go_to_genes_dict = SortedDict()

            # Iterate through all genes in genome.gene_set
            for gene in self.genome.gene_set:
                go_details = self.G_raw.nodes[gene].get("go_details", [])

                # Iterate through all GO details for the current gene
                for detail in go_details:
                    go_id = detail["go"]["go_id"]
                    if go_id not in go_to_genes_dict:
                        go_to_genes_dict[go_id] = GeneSet()
                    go_to_genes_dict[go_id].add(gene)

            self._go_to_genes = go_to_genes_dict
            return go_to_genes_dict
        else:
            return self._go_to_genes

    def create_go_subgraph(self, go_terms, go_dag):
        G = nx.DiGraph()  # Using a directed graph for GO hierarchy

        for go_id in go_terms:
            node_data = {
                "id": go_dag[go_id].id,
                "item_id": go_dag[go_id].item_id,
                "name": go_dag[go_id].name,
                "namespace": go_dag[go_id].namespace,
                "level": go_dag[go_id].level,
                "depth": go_dag[go_id].depth,
                "is_obsolete": go_dag[go_id].is_obsolete,
                "alt_ids": go_dag[go_id].alt_ids,
                "gene_set": self.go_to_genes.get(go_id, None),
                "genes": {},
            }
            # Add data from raw
            for gene in node_data["gene_set"]:
                for go_details in self.G_raw.nodes[gene]["go_details"]:
                    if go_details["go"]["go_id"] == go_id:
                        node_data["genes"][gene] = {}
                        node_data["genes"][gene]["go_details"] = go_details
            G.add_node(go_id, **node_data)

            # Add edges
            for parent in go_dag[go_id].parents:
                G.add_edge(go_id, parent.id)

        # Remove nodes that have no data. These are redundant nodes.
        nodes_to_remove = [
            node
            for node, data in G.nodes(data=True)
            if all(value is None for value in data.values())
        ]
        for node in nodes_to_remove:
            G.remove_node(node)

        # Assert there's only one node at level 0
        num_level_0_nodes = len(
            [node for node in G.nodes(data=True) if node[1]["level"] == 0]
        )
        assert (
            num_level_0_nodes == 1
        ), "There should be only one root node for a GO subgraph"
        root_node = [node for node, data in G.nodes(data=True) if data["level"] == 0][0]

        # Check for each level starting from level 0, if all nodes have a path to root
        # If not, make a direct edge from the node to root
        max_level = max(data["level"] for _, data in G.nodes(data=True))
        for level in range(0, max_level + 1):
            nodes_at_level = [
                node for node, data in G.nodes(data=True) if data["level"] == level
            ]
            for node in nodes_at_level:
                if not nx.has_path(G, node, root_node):
                    G.add_edge(node, root_node)

        return G

    @staticmethod
    def combine_with_super_node(graphs: list[nx.Graph] = None) -> nx.Graph:
        G_combined = nx.DiGraph()
        super_node = "GO:ROOT"  # A fictitious super node
        G_combined.add_node(
            super_node, name="GO Super Node", namespace="super_root", level=-1
        )

        for G in graphs:
            # Merge nodes and edges from G into G_combined
            G_combined = nx.compose(G_combined, G)

            # Identify the level 0 nodes of G using the "level" attribute
            level_0_nodes = [
                node for node, data in G.nodes(data=True) if data.get("level") == 0
            ]

            # Link the identified level 0 nodes to the super node
            for node in level_0_nodes:
                G_combined.add_edge(node, super_node)

        return G_combined

    def create_G_go(self) -> nx.DiGraph:
        bp_terms = [
            go_id
            for go_id, term in self.genome.go_dag.items()
            if term.namespace == "biological_process" and go_id in self.all_go_terms
        ]
        mf_terms = [
            go_id
            for go_id, term in self.genome.go_dag.items()
            if term.namespace == "molecular_function" and go_id in self.all_go_terms
        ]
        cc_terms = [
            go_id
            for go_id, term in self.genome.go_dag.items()
            if term.namespace == "cellular_component" and go_id in self.all_go_terms
        ]

        # construct individual subgraphs
        bp_subgraph = self.create_go_subgraph(bp_terms, self.genome.go_dag)
        mf_subgraph = self.create_go_subgraph(mf_terms, self.genome.go_dag)
        cc_subgraph = self.create_go_subgraph(cc_terms, self.genome.go_dag)

        # construct whole GO with super node
        G_go = self.combine_with_super_node([bp_subgraph, mf_subgraph, cc_subgraph])
        return G_go

    # Edge Features
    @staticmethod
    def add_physical_edges(G_raw: nx.Graph, G_physical: nx.Graph) -> nx.Graph:
        for node_name, node_data in G_raw.nodes(data=True):
            if "interaction_details" in node_data and isinstance(
                node_data["interaction_details"], list
            ):
                for interaction in node_data["interaction_details"]:
                    if interaction.get("interaction_type") == "Physical":
                        node1 = interaction["locus1"]["format_name"]
                        node2 = interaction["locus2"]["format_name"]
                        if node1 in G_raw and node2 in G_raw:
                            # TODO Need a mechanism to process the interaction data.
                            # Commented out to work with cell data.
                            G_physical.add_edge(node1, node2, **interaction)
                            G_physical.add_edge(node1, node2)
        return G_physical

    @staticmethod
    def add_genetic_edges(G_raw: nx.Graph, G_genetic: nx.Graph) -> nx.Graph:
        for node_name, node_data in G_raw.nodes(data=True):
            if "interaction_details" in node_data and isinstance(
                node_data["interaction_details"], list
            ):
                for interaction in node_data["interaction_details"]:
                    if interaction.get("interaction_type") == "Genetic":
                        node1 = interaction["locus1"]["format_name"]
                        node2 = interaction["locus2"]["format_name"]
                        if node1 in G_raw and node2 in G_raw:
                            G_genetic.add_edge(node1, node2, **interaction)
        return G_genetic

    @staticmethod
    def add_regulatory_edges(
        G_raw: nx.Graph, G_regulatory: nx.DiGraph = None
    ) -> nx.DiGraph:
        """
        Add regulatory edges from the raw graph to the regulatory graph.
        """
        if G_regulatory is None:
            G_regulatory = nx.DiGraph()

        for node_name, node_data in G_raw.nodes(data=True):
            if "regulation_details" in node_data and isinstance(
                node_data["regulation_details"], list
            ):
                for regulation in node_data["regulation_details"]:
                    # Extract details
                    locus1_name = regulation["locus1"]["format_name"]
                    locus2_name = regulation["locus2"]["format_name"]

                    # Add nodes
                    G_regulatory.add_node(locus1_name, **regulation["locus1"])
                    G_regulatory.add_node(locus2_name, **regulation["locus2"])

                    # Add directed edge
                    G_regulatory.add_edge(locus1_name, locus2_name, **regulation)

        return G_regulatory

    # Node Features
    @staticmethod
    def add_gene_protein_overview(G_raw: nx.Graph, G_gene: nx.Graph) -> nx.Graph:
        for node_name, node_data in G_raw.nodes(data=True):
            protein_overview_template = {
                "length": None,
                "molecular_weight": None,
                "pi": None,
                "median_value": None,
                "median_abs_dev_value": None,
            }
            if "protein_overview" in node_data["locus"]:
                protein_overview = protein_overview_template.copy()
                protein_overview["length"] = node_data["locus"]["protein_overview"].get(
                    "length"
                )
                protein_overview["molecular_weight"] = node_data["locus"][
                    "protein_overview"
                ].get("molecular_weight")
                protein_overview["pi"] = node_data["locus"]["protein_overview"].get(
                    "pi"
                )
                protein_overview["median_value"] = node_data["locus"][
                    "protein_overview"
                ].get("median_value")
                protein_overview["median_abs_dev_value"] = node_data["locus"][
                    "protein_overview"
                ].get("median_abs_dev_value")
                G_gene.add_node(node_name, **protein_overview)
            else:
                G_gene.add_node(node_name, **protein_overview_template)
        return G_gene


def plot_go_graph(G):
    # Define a color map for namespaces.
    namespace_colors = {
        "super_root": "#122A4B",
        "biological_process": "#FF552E",
        "molecular_function": "#8286C2",
        "cellular_component": "#D23943",
    }

    # Identify nodes without 'namespace'
    nodes_missing_namespace = [
        node for node, attrs in G.nodes(data=True) if "namespace" not in attrs
    ]
    print(f"num nodes_missing_namespace: {len(nodes_missing_namespace)}")

    # Generate the color list for nodes based on their namespace
    node_colors = [
        namespace_colors[G.nodes[node].get("namespace", "missing")]
        for node in G.nodes()
    ]

    max_level = max([data["level"] for node, data in G.nodes(data=True)])
    min_level = min([data["level"] for node, data in G.nodes(data=True)])

    # Calculate node positions based on levels
    level_spacing = 10.0
    horizontal_spacing = 20.0

    new_pos = {}
    for node, data in G.nodes(data=True):
        level = data.get("level")
        nodes_in_level = [n for n, d in G.nodes(data=True) if d.get("level") == level]

        # Calculate horizontal position
        idx_in_level = nodes_in_level.index(node)
        total_nodes_in_level = len(nodes_in_level)
        new_x = (idx_in_level - total_nodes_in_level / 2) * horizontal_spacing

        # Adjust vertical position
        new_y = (max_level - level) * level_spacing - min_level * level_spacing

        new_pos[node] = (new_x, new_y)

    # Plotting
    plt.figure(figsize=(20, 10))
    nx.draw(
        G,
        new_pos,
        with_labels=False,
        node_size=20,
        node_color=node_colors,
        alpha=0.6,
        linewidths=0.5,
        width=0.25,
        edge_color="lightgray",
        arrowsize=5,
    )

    # Add legend
    legend_elements = [
        Patch(
            facecolor=namespace_colors[key],
            edgecolor="black",
            label=key.replace("_", " ").title(),
        )
        for key in namespace_colors
    ]
    plt.legend(
        handles=legend_elements,
        loc="upper right",
        title="Namespaces",
        fontsize=28,
        title_fontsize=28,
    )

    # Annotate levels on the left side
    unique_levels = sorted(list({data["level"] for node, data in G.nodes(data=True)}))

    # Determine the minimum x-coordinate and set the margin
    min_x_coord = min([x for x, y in new_pos.values()])
    margin = 120.0 * horizontal_spacing  # Adjust the multiplier as needed

    for level in unique_levels:
        y_pos = (max_level - level) * level_spacing - min_level * level_spacing
        plt.text(
            x=min_x_coord - margin,  # Use the adjusted x-coordinate
            y=y_pos,
            s=f"level: {level}",
            ha="left",
            va="center",
            fontsize=28,
        )

    plt.title("GO DAG with Forwarded Isolated Nodes and Super Node", fontsize=28)
    plt.tight_layout()
    plt.savefig(
        "./notes/assets/images/dcell-gene-ontology-dag-no-isoloated-with-super-node.png",
        dpi=300,
    )
    plt.close()
    # plt.show()


# TODO add a function that will remove genes not in dmf costanzo and smf costanzo


def filter_go_IGI(G_go: nx.DiGraph) -> nx.DiGraph:
    # Create a copy of the original graph - avoid inplace
    G = G_go.copy()
    nodes_to_remove = []

    for go_term in G.nodes():
        if "gene_set" in G.nodes[go_term] and "genes" in G.nodes[go_term]:
            # Identify genes with "IGI" experiment
            remove_genes = [
                gene
                for gene in G.nodes[go_term]["gene_set"]
                if gene in G.nodes[go_term]["genes"]
                and G.nodes[go_term]["genes"][gene]["go_details"]["experiment"][
                    "display_name"
                ]
                == "IGI"
            ]

            # Remove genes from gene_set and genes attribute
            for gene in remove_genes:
                G.nodes[go_term]["gene_set"].remove(gene)
                del G.nodes[go_term]["genes"][gene]

            # Check if gene_set is now empty
            if len(G.nodes[go_term]["gene_set"]) == 0:
                in_nodes = list(G.predecessors(go_term))
                out_nodes = list(G.successors(go_term))

                # Forward the in edges of go_term to each of its out nodes
                for in_node in in_nodes:
                    for out_node in out_nodes:
                        G.add_edge(in_node, out_node)
                nodes_to_remove.append(go_term)

    # Remove nodes marked for deletion
    log.info(f"IGI nodes removed: {len(nodes_to_remove)}")
    for node in nodes_to_remove:
        G.remove_node(node)

    return G


def filter_by_date(G_go: nx.DiGraph, cutoff_date: str) -> nx.DiGraph:
    # Create a copy of the original graph - avoid inplace
    G = G_go.copy()
    nodes_to_remove = []

    for go_term in G.nodes():
        if "gene_set" in G.nodes[go_term] and "genes" in G.nodes[go_term]:
            # Identify genes whose annotations are after the cutoff date
            remove_genes = [
                gene
                for gene in G.nodes[go_term]["gene_set"]
                if gene in G.nodes[go_term]["genes"]
                and G.nodes[go_term]["genes"][gene]["go_details"]["date_created"]
                > cutoff_date
            ]

            # Remove genes from gene_set and genes attribute
            for gene in remove_genes:
                G.nodes[go_term]["gene_set"].remove(gene)
                del G.nodes[go_term]["genes"][gene]

            # Check if gene_set is now empty
            if len(G.nodes[go_term]["gene_set"]) == 0:
                in_nodes = list(G.predecessors(go_term))
                out_nodes = list(G.successors(go_term))

                # Forward the in edges of go_term to each of its out nodes
                for in_node in in_nodes:
                    for out_node in out_nodes:
                        G.add_edge(in_node, out_node)
                nodes_to_remove.append(go_term)

    # Remove nodes marked for deletion
    log.info(f"Nodes annotated after {cutoff_date} removed: {len(nodes_to_remove)}")
    for node in nodes_to_remove:
        G.remove_node(node)

    return G


def filter_redundant_terms(G_go: nx.DiGraph) -> nx.DiGraph:
    # Create a copy of the original graph - avoid inplace
    G = G_go.copy()

    # Find leaf nodes - nodes with no successors
    leaf_nodes = [node for node in G if len(list(G.successors(node))) == 0]

    # Nodes to check for redundancy
    nodes_to_check = set(leaf_nodes)
    nodes_to_remove = set()

    while nodes_to_check:
        node = nodes_to_check.pop()
        parent_nodes = list(G.predecessors(node))

        for parent in parent_nodes:
            if (
                "gene_set" in G.nodes[parent]
                and "gene_set" in G.nodes[node]
                and set(G.nodes[parent]["gene_set"]) == set(G.nodes[node]["gene_set"])
            ):
                # Add the redundant node to the list for removal
                nodes_to_remove.add(node)

                # Forward the in edges of node to each of its out nodes
                in_nodes = list(G.predecessors(node))
                out_nodes = list(G.successors(node))
                for in_node in in_nodes:
                    for out_node in out_nodes:
                        G.add_edge(in_node, out_node)

            # Add the parent to the nodes to check, only if it's not already marked for removal
            if parent not in nodes_to_remove:
                nodes_to_check.add(parent)

    # Remove nodes marked for deletion
    log.info(f"Redundant nodes removed: {len(nodes_to_remove)}")
    for node in nodes_to_remove:
        if node in G:
            G.remove_node(node)

    return G


def filter_by_contained_genes(G_go: nx.DiGraph, n: int, gene_set: set) -> nx.DiGraph:
    G = G_go.copy()

    def compute_containment(go_term) -> set[str]:
        """Function to compute containment of a given term with its subsequent terms."""

        # Reverse the graph to travel in the opposite direction
        G_reverse = G_go.reverse(copy=False)

        # Find all reachable nodes from the current term in the reversed graph
        reachable_nodes = nx.single_source_shortest_path_length(
            G_reverse, go_term
        ).keys()

        # Get gene sets of all these nodes
        gene_sets = [
            set(G_go.nodes[node].get("gene_set", [])) for node in reachable_nodes
        ]

        # Compute the union of all gene sets
        contained_genes = set.union(*gene_sets)
        if gene_set is not None:
            contained_genes = contained_genes.intersection(gene_set)
        return contained_genes

    # Construct the containment_dictionary
    containment_dictionary = {node: compute_containment(node) for node in G.nodes()}

    # Mark nodes for removal based on containment criteria
    nodes_to_remove = [
        node for node, genes in containment_dictionary.items() if len(genes) < n
    ]

    # Forward the connections of nodes marked for removal
    for node in nodes_to_remove:
        in_nodes = list(G.predecessors(node))
        out_nodes = list(G.successors(node))
        for in_node in in_nodes:
            for out_node in out_nodes:
                if in_node != out_node and not G.has_edge(in_node, out_node):
                    G.add_edge(in_node, out_node)

    # Remove the marked nodes
    log.info(f"Nodes with < {n} contained genes removed: {len(nodes_to_remove)}")
    for node in nodes_to_remove:
        G.remove_node(node)

    return G


def plot_annotation_dates_by_month(G_go: nx.DiGraph):
    # Function to convert date strings to datetime objects
    def convert_to_datetime(date_str):
        return datetime.strptime(date_str, "%Y-%m-%d")

    # Collect all the annotation dates and group by month and year
    dates = defaultdict(int)
    for go_term in G_go.nodes():
        if "genes" in G_go.nodes[go_term]:
            for gene, details in G_go.nodes[go_term]["genes"].items():
                date_obj = convert_to_datetime(details["go_details"]["date_created"])
                month_year = date_obj.strftime("%Y-%m")  # Format as "YYYY-MM"
                dates[month_year] += 1

    # Sort the dates chronologically
    sorted_dates = sorted(dates.items(), key=lambda x: datetime.strptime(x[0], "%Y-%m"))

    # Extract x and y values
    x_vals, y_vals = zip(*sorted_dates)

    # Extract unique years and their first month occurrence for labeling purposes
    year_ticks = [date for idx, date in enumerate(x_vals) if date.endswith("-01")]

    # Plot the histogram
    plt.bar(x_vals, y_vals, edgecolor="k", alpha=0.7)
    plt.title("Gene Ontology Annotation Dates (Grouped by Month)", fontsize=24)
    plt.xlabel("Date (Year)", labelpad=10, fontsize=24)
    plt.ylabel("Number of Annotations", labelpad=10, fontsize=24)

    # Display only the labels for the years using their first month
    plt.xticks(year_ticks, rotation=45, fontsize=16)
    plt.yticks(fontsize=16)

    # Make year ticks longer for distinction
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        if tick.label1.get_text() in year_ticks:
            tick.tick1line.set_markersize(10)
            tick.label1.set_fontweight("bold")
    plt.tight_layout()
    plt.savefig(
        "./notes/assets/images/gene-ontology-annotation-dates-grouped-by-month.png",
        dpi=300,
    )
    # plt.show()
    plt.close()


def plot_histogram_of_gene_counts(G_go: nx.DiGraph):
    # Extracting the gene sets for each GO term
    gene_sets = [G_go.nodes[node].get("gene_set", []) for node in G_go.nodes()]

    # Getting the counts of genes for each GO term
    gene_counts = [len(gene_set) for gene_set in gene_sets]

    # Plotting the histogram
    plt.hist(gene_counts, bins=100, alpha=0.7)
    plt.title("Histogram of Gene Counts per GO Term")
    plt.xlabel("Number of Genes")
    plt.ylabel("Number of GO Terms")
    plt.grid(axis="y", linestyle="--")
    plt.tight_layout()
    plt.show()


def plot_histogram_of_contained_gene_counts(
    G_go: nx.DiGraph, gene_set: set = None, show_min_max: bool = False
):
    def compute_containment(go_term):
        """Function to compute containment of a given term with its subsequent terms."""

        # Reverse the graph to travel in the opposite direction
        G_reverse = G_go.reverse(copy=False)

        # Find all reachable nodes from the current term in the reversed graph
        reachable_nodes = nx.single_source_shortest_path_length(
            G_reverse, go_term
        ).keys()

        # Get gene sets of all these nodes
        gene_sets = [
            set(G_go.nodes[node].get("gene_set", [])) for node in reachable_nodes
        ]

        # Compute the union of all gene sets
        contained_genes = set.union(*gene_sets)
        if gene_set is not None:
            contained_genes = contained_genes.intersection(gene_set)
        return contained_genes

    # Construct containment_dictionary
    containment_dictionary = {node: compute_containment(node) for node in G_go.nodes()}

    # Get counts for histogram
    contained_gene_counts = [len(genes) for genes in containment_dictionary.values()]

    # Plotting the histogram
    plt.hist(contained_gene_counts, bins=100, alpha=0.7)
    plt.title("Histogram of Contained Gene Counts per GO Term")
    plt.xlabel("Number of Contained Genes")
    plt.ylabel("Number of GO Terms")

    # Optional vertical dashed lines for min and max values
    if show_min_max:
        plt.axvline(min(contained_gene_counts), color="red", linestyle="--", alpha=0.7)
        plt.axvline(max(contained_gene_counts), color="red", linestyle="--", alpha=0.7)

    plt.grid(axis="y", linestyle="--")
    plt.tight_layout()
    plt.show()


def compare_go_terms(G_raw, G):
    matching_nodes = 0
    total_nodes = len(G_raw.nodes())
    total_new_terms = (
        0  # This will accumulate the total new terms added to G_raw from G
    )

    for node in G_raw.nodes():
        # Extract GO terms from the complex graph G_raw
        go_details_G_raw = G_raw.nodes[node]["go_details"]
        go_terms_G_raw = {
            detail["go"]["go_id"] for detail in go_details_G_raw if "go" in detail
        }

        # Check if the node exists in the simpler graph G
        if node in G.nodes():
            # Extract GO terms from the simple graph G
            go_terms_G = {rec["GO_ID"] for rec in G.nodes[node]["annotations"]}

            # Check if the terms in G are exactly the same as in G_raw
            if go_terms_G == go_terms_G_raw:
                matching_nodes += 1
            else:
                # Calculate the new terms from G that aren't in G_raw
                new_terms = go_terms_G - go_terms_G_raw
                total_new_terms += len(new_terms)
                # Optionally, report the node and the number of new terms
                print(
                    f"Node {node}: {len(new_terms)} new terms from G would be added to G_raw."
                )
        else:
            # Handle the case where a node in G_raw does not exist in G
            print(f"Node {node} from G_raw does not exist in G.")

    print(f"{matching_nodes} out of {total_nodes} nodes have matching GO term sets.")
    print(f"A total of {total_new_terms} new GO terms would be added to G_raw from G.")


def go_gaf_investigation():
    import os
    import random

    import matplotlib.pyplot as plt
    import pandas as pd
    from Bio.UniProt.GOA import gafiterator
    from dotenv import load_dotenv

    from torchcell.datasets.scerevisiae import (
        DmfCostanzo2016Dataset,
        SmfCostanzo2016Dataset,
    )

    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")

    genome = SCerevisiaeGenome(
        data_root=osp.join(DATA_ROOT, "data/sgd/genome"), overwrite=True
    )
    genome.drop_chrmt()
    genome.drop_empty_go()
    print(f"Gene Set Length: {len(genome.gene_set)}")
    # Initialize the graph
    G = nx.Graph()

    # Add nodes to the graph
    for gene_name in genome.gene_set:
        G.add_node(gene_name, annotations=[])

    # Define your filters here
    synonym_filter = {"Synonym": genome.gene_set}

    # Function to check if a record matches the given filters
    def record_has(rec, filters):
        for field, values in filters.items():
            if field in rec and not set(rec[field]).isdisjoint(values):
                return True
        return False

    # Read the GAF file and apply filters
    gaf_path = osp.join(DATA_ROOT, "data/sgd.gaf")
    with open(gaf_path) as handle:
        for record in gafiterator(handle):
            # Apply filters
            for synonym in record.get("Synonym", []):
                if synonym in genome.gene_set:
                    G.nodes[synonym]["annotations"].append(record)
                    break  # Once matched, no need to check other synonyms

    # use this len(G.nodes[[node for node in G.nodes][100]]["annotations"])
    graph = SCerevisiaeGraph(
        data_root=osp.join(DATA_ROOT, "data/sgd/genome"), genome=genome
    )
    graph.G_go
    print()
    compare_go_terms(graph.G_raw, G)


import networkx as nx
import plotly.graph_objects as go


def plotly_go_graph(G):
    # Define color map for namespaces
    namespace_colors = {
        "super_root": "#122A4B",
        "biological_process": "#FF552E",
        "molecular_function": "#8286C2",
        "cellular_component": "#D23943",
        "missing": "#AAAAAA",  # Color for missing namespace
    }

    # Determine levels and layout
    max_level = max(data.get("level", 0) for node, data in G.nodes(data=True))
    min_level = min(data.get("level", 0) for node, data in G.nodes(data=True))
    level_spacing = 2.0  # Increased spacing for levels
    horizontal_spacing = 2.0  # Increased spacing horizontally

    new_pos = {}
    for node, data in G.nodes(data=True):
        level = data.get("level", 0)
        nodes_in_level = [n for n, d in G.nodes(data=True) if d.get("level") == level]
        idx_in_level = nodes_in_level.index(node)
        total_nodes_in_level = len(nodes_in_level)
        new_x = (idx_in_level - total_nodes_in_level / 2) * horizontal_spacing
        new_y = (max_level - level) * level_spacing - min_level * level_spacing
        new_pos[node] = (new_x, new_y)

    # Creating edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = new_pos[edge[0]]
        x1, y1 = new_pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    # Creating node traces
    node_x = []
    node_y = []
    node_info = []
    node_color = []
    for node in G.nodes():
        x, y = new_pos[node]
        node_x.append(x)
        node_y.append(y)
        node_data = G.nodes[node]
        name = node_data.get("name", "none")
        level = node_data.get("level", "none")
        namespace = node_data.get("namespace", "missing")
        info_str = f"{node}: name={name}, level={level}"
        node_info.append(info_str)
        node_color.append(namespace_colors.get(namespace, "missing"))

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        text=node_info,
        marker=dict(showscale=False, size=7, color=node_color, line=None),
    )
    # Create the figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="Gene Ontology Graph",
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="rgba(0,0,0,0)",
        ),
    )

    fig.show()


def main() -> None:
    import os
    import random

    import matplotlib.pyplot as plt
    import pandas as pd
    from dotenv import load_dotenv

    from torchcell.datasets.scerevisiae import (
        DmfCostanzo2016Dataset,
        SmfCostanzo2016Dataset,
    )

    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")

    genome = SCerevisiaeGenome(
        data_root=osp.join(DATA_ROOT, "data/sgd/genome"), overwrite=True
    )
    graph = SCerevisiaeGraph(
        data_root=osp.join(DATA_ROOT, "data/sgd/genome"), genome=genome
    )
    #
    dmf_dataset = DmfCostanzo2016Dataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/costanzo2016_1e5"),
        preprocess={"duplicate_resolution": "low_dmf_std"},
        # subset_n=100,
    )
    smf_dataset = SmfCostanzo2016Dataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/costanzo2016_smf"),
        preprocess={"duplicate_resolution": "low_std_both"},
        skip_process_file_exist_check=True,
    )
    gene_set = smf_dataset.gene_set.union(dmf_dataset.gene_set)
    #
    print(graph.G_go.number_of_nodes())
    G = graph.G_go.copy()

    ##### Filtering
    # plot_histogram_of_gene_counts(G)
    # plot_histogram_of_contained_gene_counts(G, gene_set=gene_set, show_min_max=True)
    # G = filter_by_date(G, "2018-02-01")
    G = filter_by_date(G, "2017-07-19")
    print(f"After date filter: {G.number_of_nodes()}")
    G = filter_go_IGI(G)
    print(f"After IGI filter: {G.number_of_nodes()}")
    G = filter_redundant_terms(G)
    print(f"After redundant filter: {G.number_of_nodes()}")
    # (filter_by_contained_genes(G, 1, gene_set=gene_set)).number_of_nodes()
    G = filter_by_contained_genes(G, n=6, gene_set=gene_set)
    print(f"After containment filter: {G.number_of_nodes()}")
    # print()
    # plot_go_graph(G)
    print()

    # plot_annotation_dates_by_month(graph.G_go)
    # plot_go_graph(graph.G_go)
    plotly_go_graph(G)


if __name__ == "__main__":
    main()
    # main_go()

    # # TODO maybe remove
    # @staticmethod
    # def parse_genome(genome) -> ParsedGenome:
    #     # BUG we have to do this black magic because when you merge datasets with +
    #     # the genome is None
    #     if genome is None:
    #         return None
    #     else:
    #         data = {}
    #         data["gene_set"] = genome.gene_set
    #         return ParsedGenome(**data)
    #         data["gene_set"] = genome.gene_set
    #         return ParsedGenome(**data)
