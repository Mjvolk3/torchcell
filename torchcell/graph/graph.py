# torchcell/graph/graph.py
# [[torchcell.graph.graph]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/graph/graph.py
# Test file: torchcell/graph/test_graph.py

import glob
import requests
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
from typing import Optional
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
from torchcell.datamodels.pydant import ModelStrictArbitrary
from torchcell.sequence import GeneSet, Genome, ParsedGenome
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
import torchcell
from pydantic import field_validator


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
style_file_path = osp.join(osp.dirname(torchcell.__file__), "torchcell.mplstyle")
plt.style.use(style_file_path)


class GeneGraph(ModelStrictArbitrary):
    """
    A graph representation with genes as nodes, wrapping an nx.Graph.
    """

    name: str
    graph: nx.Graph
    max_gene_set: GeneSet

    @field_validator("graph")
    @classmethod
    def validate_genes_in_graph(cls, graph, info):
        """Validate that all nodes in the graph are in the max_gene_set"""
        values = info.data
        if "max_gene_set" not in values:
            return graph

        max_gene_set = values["max_gene_set"]
        graph_nodes = set(graph.nodes())
        invalid_nodes = graph_nodes - set(max_gene_set)

        if invalid_nodes:
            import logging

            log = logging.getLogger(__name__)
            log.warning(
                f"Graph contains {len(invalid_nodes)} nodes not in max_gene_set"
            )

        return graph

    def __getattr__(self, name):
        """Forward attribute access to the underlying graph"""
        return getattr(self.graph, name)

    def __repr__(self):
        """Informative representation showing graph name, nodes and edges"""
        return f"GeneGraph(name='{self.name}', nodes={self.graph.number_of_nodes()}, edges={self.graph.number_of_edges()})"


class GeneMultiGraph(ModelStrictArbitrary):
    """
    A collection of GeneGraph objects stored in a SortedDict.
    """

    graphs: SortedDict[str, GeneGraph]

    def __getitem__(self, key):
        """Allow dictionary-like access to graphs by name"""
        return self.graphs[key]

    def __iter__(self):
        """Iterate over the graph names"""
        return iter(self.graphs)

    def __contains__(self, key):
        """Check if a graph name exists"""
        return key in self.graphs

    def __len__(self):
        """Get number of graphs"""
        return len(self.graphs)

    def items(self):
        """Get the (name, graph) pairs"""
        return self.graphs.items()

    def keys(self):
        """Get graph names"""
        return self.graphs.keys()

    def values(self):
        """Get graph objects"""
        return self.graphs.values()

    def __repr__(self):
        """Informative representation showing all contained graphs"""
        graph_reprs = []
        for name, graph in self.graphs.items():
            graph_reprs.append(
                f"  {name}: {graph.graph.number_of_nodes()} nodes, {graph.graph.number_of_edges()} edges"
            )

        graphs_str = "\n".join(graph_reprs)
        return f"GeneMultiGraph(\n{graphs_str}\n)"


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


@define
class SCerevisiaeGraph:
    sgd_root: str = field(init=True, repr=False, default="data/sgd/genome")
    string_root: str = field(init=True, repr=False, default="data/string")
    tflink_root: str = field(init=True, repr=False, default="data/tflink")
    genome: GeneSet = field(init=True, repr=False, default=None)
    json_files: list[str] = field(init=False, repr=False, default=None)

    # Using private attributes for storage
    _G_raw: nx.Graph = field(init=False, repr=False, default=None)
    _G_gene: GeneGraph = field(init=False, repr=False, default=None)
    _G_physical: GeneGraph = field(init=False, repr=False, default=None)
    _G_genetic: GeneGraph = field(init=False, repr=False, default=None)
    _G_regulatory: GeneGraph = field(init=False, repr=False, default=None)
    _G_go: nx.DiGraph = field(init=False, repr=False, default=None)

    # TFlink database
    _G_tflink: GeneGraph = field(init=False, repr=False, default=None)

    # STRING v9.1 graph attributes
    _G_string9_1_neighborhood: GeneGraph = field(init=False, repr=False, default=None)
    _G_string9_1_fusion: GeneGraph = field(init=False, repr=False, default=None)
    _G_string9_1_cooccurence: GeneGraph = field(init=False, repr=False, default=None)
    _G_string9_1_coexpression: GeneGraph = field(init=False, repr=False, default=None)
    _G_string9_1_experimental: GeneGraph = field(init=False, repr=False, default=None)
    _G_string9_1_database: GeneGraph = field(init=False, repr=False, default=None)

    # STRING v12.0 graph attributes
    _G_string12_0_neighborhood: GeneGraph = field(init=False, repr=False, default=None)
    _G_string12_0_fusion: GeneGraph = field(init=False, repr=False, default=None)
    _G_string12_0_cooccurence: GeneGraph = field(init=False, repr=False, default=None)
    _G_string12_0_coexpression: GeneGraph = field(init=False, repr=False, default=None)
    _G_string12_0_experimental: GeneGraph = field(init=False, repr=False, default=None)
    _G_string12_0_database: GeneGraph = field(init=False, repr=False, default=None)

    _all_go_terms = field(init=False, repr=False, default=None)
    _go_to_genes = field(init=False, repr=False, default=None)

    def __attrs_post_init__(self) -> None:
        self.json_files = [f"{gene}.json" for gene in self.genome.gene_set]

        # Create SGD graph directory
        sgd_graph_dir = osp.join(self.sgd_root, "graph")
        os.makedirs(sgd_graph_dir, exist_ok=True)

        # Create STRING graph directory
        string_graph_dir = osp.join(self.string_root, "graph")
        os.makedirs(string_graph_dir, exist_ok=True)

        # Create TFLink graph directory
        tflink_graph_dir = osp.join(self.tflink_root, "graph")
        os.makedirs(tflink_graph_dir, exist_ok=True)

        # Clean up sql connections
        self.genome = self.parse_genome(self.genome)

    @staticmethod
    def parse_genome(genome) -> ParsedGenome:
        data = {}
        data["gene_set"] = genome.gene_set
        # Store the alias_to_systematic dictionary if it exists
        if hasattr(genome, "alias_to_systematic"):
            # We need to save this as a regular dict since ParsedGenome may not accept it directly
            setattr(ParsedGenome, "alias_to_systematic", genome.alias_to_systematic)
        return ParsedGenome(**data)

    # Property definitions for lazy loading
    @property
    def G_raw(self) -> nx.Graph:
        if self._G_raw is None:
            graph_dir = osp.join(self.sgd_root, "graph")
            raw_graph_path = osp.join(graph_dir, "G_raw.pkl")
            if osp.exists(raw_graph_path):
                self._G_raw = self.load_graph("G_raw", root_type="sgd")
            else:
                self._G_raw = self.add_json_data_to_graph(
                    data_root=self.sgd_root, json_files=self.json_files
                )
                self.save_graph(self._G_raw, "G_raw", root_type="sgd")
        return self._G_raw

    @property
    def G_gene(self) -> GeneGraph:
        if self._G_gene is None:
            gene_graph_path = osp.join(self.sgd_root, "graph", "G_gene.pkl")
            if osp.exists(gene_graph_path):
                self._G_gene = self.load_graph("G_gene", root_type="sgd")
            else:
                nx_graph = self.add_gene_protein_overview(
                    G_raw=self.G_raw, G_gene=nx.Graph()
                )
                nx_graph = self.add_loci_information(G_raw=self.G_raw, G_gene=nx_graph)
                nx_graph = self.add_pathway_annotation(
                    G_raw=self.G_raw, G_gene=nx_graph
                )
                self._G_gene = GeneGraph(
                    name="gene", graph=nx_graph, max_gene_set=self.genome.gene_set
                )
                self.save_graph(self._G_gene, "G_gene", root_type="sgd")
        return self._G_gene

    @property
    def G_physical(self) -> GeneGraph:
        if self._G_physical is None:
            physical_graph_path = osp.join(self.sgd_root, "graph", "G_physical.pkl")
            if osp.exists(physical_graph_path):
                self._G_physical = self.load_graph("G_physical", root_type="sgd")
            else:
                nx_graph = self.add_physical_edges(
                    G_raw=self.G_raw, G_physical=nx.Graph()
                )
                self._G_physical = GeneGraph(
                    name="physical", graph=nx_graph, max_gene_set=self.genome.gene_set
                )
                self.save_graph(self._G_physical, "G_physical", root_type="sgd")
        return self._G_physical

    @property
    def G_genetic(self) -> GeneGraph:
        if self._G_genetic is None:
            genetic_graph_path = osp.join(self.sgd_root, "graph", "G_genetic.pkl")
            if osp.exists(genetic_graph_path):
                self._G_genetic = self.load_graph("G_genetic", root_type="sgd")
            else:
                nx_graph = self.add_genetic_edges(
                    G_raw=self.G_raw, G_genetic=nx.Graph()
                )
                self._G_genetic = GeneGraph(
                    name="genetic", graph=nx_graph, max_gene_set=self.genome.gene_set
                )
                self.save_graph(self._G_genetic, "G_genetic", root_type="sgd")
        return self._G_genetic

    @property
    def G_regulatory(self) -> GeneGraph:
        if self._G_regulatory is None:
            regulatory_graph_path = osp.join(self.sgd_root, "graph", "G_regulatory.pkl")
            if osp.exists(regulatory_graph_path):
                self._G_regulatory = self.load_graph("G_regulatory", root_type="sgd")
            else:
                nx_graph = self.add_regulatory_edges(G_raw=self.G_raw)
                self._G_regulatory = GeneGraph(
                    name="regulatory", graph=nx_graph, max_gene_set=self.genome.gene_set
                )
                self.save_graph(self._G_regulatory, "G_regulatory", root_type="sgd")
        return self._G_regulatory

    @property
    def G_go(self) -> nx.DiGraph:
        if self._G_go is None:
            go_graph_path = osp.join(self.sgd_root, "graph", "G_go.pkl")
            if osp.exists(go_graph_path):
                self._G_go = self.load_graph("G_go", root_type="sgd")
            else:
                self._G_go = self.create_G_go()
                self.save_graph(self._G_go, "G_go", root_type="sgd")
        return self._G_go

    @property
    def G_tflink(self) -> GeneGraph:
        if self._G_tflink is None:
            tf_graph_path = osp.join(self.tflink_root, "graph", "G_tflink.pkl")
            if osp.exists(tf_graph_path):
                self._G_tflink = self.load_graph("G_tflink", root_type="tflink")
            else:
                nx_graph = self.create_G_tflink()
                self._G_tflink = GeneGraph(
                    name="tflink", graph=nx_graph, max_gene_set=self.genome.gene_set
                )
                self.save_graph(self._G_tflink, "G_tflink", root_type="tflink")
        return self._G_tflink

    # STRING v9.1 properties
    @property
    def G_string9_1_neighborhood(self) -> GeneGraph:
        if self._G_string9_1_neighborhood is None:
            self._initialize_string_graph("neighborhood", "9.1")
        return self._G_string9_1_neighborhood

    @property
    def G_string9_1_fusion(self) -> GeneGraph:
        if self._G_string9_1_fusion is None:
            self._initialize_string_graph("fusion", "9.1")
        return self._G_string9_1_fusion

    @property
    def G_string9_1_cooccurence(self) -> GeneGraph:
        if self._G_string9_1_cooccurence is None:
            self._initialize_string_graph("cooccurence", "9.1")
        return self._G_string9_1_cooccurence

    @property
    def G_string9_1_coexpression(self) -> GeneGraph:
        if self._G_string9_1_coexpression is None:
            self._initialize_string_graph("coexpression", "9.1")
        return self._G_string9_1_coexpression

    @property
    def G_string9_1_experimental(self) -> GeneGraph:
        if self._G_string9_1_experimental is None:
            self._initialize_string_graph("experimental", "9.1")
        return self._G_string9_1_experimental

    @property
    def G_string9_1_database(self) -> GeneGraph:
        if self._G_string9_1_database is None:
            self._initialize_string_graph("database", "9.1")
        return self._G_string9_1_database

    # STRING v12.0 properties
    @property
    def G_string12_0_neighborhood(self) -> GeneGraph:
        if self._G_string12_0_neighborhood is None:
            self._initialize_string_graph("neighborhood", "12.0")
        return self._G_string12_0_neighborhood

    @property
    def G_string12_0_fusion(self) -> GeneGraph:
        if self._G_string12_0_fusion is None:
            self._initialize_string_graph("fusion", "12.0")
        return self._G_string12_0_fusion

    @property
    def G_string12_0_cooccurence(self) -> GeneGraph:
        if self._G_string12_0_cooccurence is None:
            self._initialize_string_graph("cooccurence", "12.0")
        return self._G_string12_0_cooccurence

    @property
    def G_string12_0_coexpression(self) -> GeneGraph:
        if self._G_string12_0_coexpression is None:
            self._initialize_string_graph("coexpression", "12.0")
        return self._G_string12_0_coexpression

    @property
    def G_string12_0_experimental(self) -> GeneGraph:
        if self._G_string12_0_experimental is None:
            self._initialize_string_graph("experimental", "12.0")
        return self._G_string12_0_experimental

    @property
    def G_string12_0_database(self) -> GeneGraph:
        if self._G_string12_0_database is None:
            self._initialize_string_graph("database", "12.0")
        return self._G_string12_0_database


    def _initialize_string_graph(self, network_type: str, version: str) -> None:
        """Initialize a specific STRING graph if it's not already loaded"""
        version_str = version.replace(".", "_")
        attr_name = f"_G_string{version_str}_{network_type}"

        graph_path = osp.join(
            self.string_root, "graph", f"G_string{version_str}_{network_type}.pkl"
        )

        if osp.exists(graph_path):
            setattr(
                self,
                attr_name,
                self.load_graph(
                    f"G_string{version_str}_{network_type}", root_type="string"
                ),
            )
        else:
            # If one graph doesn't exist, we need to create all of them
            version_dir = osp.join(self.string_root, f"v{version}")
            os.makedirs(version_dir, exist_ok=True)

            string_file = osp.join(
                version_dir, f"4932.protein.links.detailed.v{version}.txt.gz"
            )
            if not osp.exists(string_file):
                self.download_string_data(string_file, version=version)

            self.create_string_graphs(string_file, version)

    def download_tflink_data(self, output_path: str) -> None:
        """
        Download the TFLink interaction file for S. cerevisiae.

        Args:
            output_path: Path to save the downloaded file
        """
        url = "https://cdn.netbiol.org/tflink/download_files/TFLink_Saccharomyces_cerevisiae_interactions_All_simpleFormat_v1.0.tsv"

        if os.path.exists(output_path):
            log.info(f"TFLink data already exists at {output_path}, skipping download")
            return

        log.info(f"Downloading TFLink data to {output_path}...")
        response = requests.get(url, stream=True)
        with open(output_path, "wb") as f:
            for chunk in tqdm(response.iter_content(chunk_size=8192)):
                f.write(chunk)
        log.info("TFLink download complete!")

    def download_string_data(self, output_path: str, version: str = "9.1") -> None:
        """
        Download the STRING protein interaction file for S. cerevisiae.

        Args:
            output_path: Path to save the downloaded file
            version: STRING database version ("9.1" or "12.0")
        """
        if version == "9.1":
            url = "http://string91.embl.de/newstring_download/protein.links.detailed.v9.1/4932.protein.links.detailed.v9.1.txt.gz"
        elif version == "12.0":
            url = "https://stringdb-downloads.org/download/protein.links.detailed.v12.0/4932.protein.links.detailed.v12.0.txt.gz"
        else:
            raise ValueError(f"Unsupported STRING version: {version}")

        if os.path.exists(output_path):
            log.info(
                f"STRING v{version} data already exists at {output_path}, skipping download"
            )
            return

        log.info(f"Downloading STRING v{version} data to {output_path}...")
        response = requests.get(url, stream=True)
        with open(output_path, "wb") as f:
            for chunk in tqdm(response.iter_content(chunk_size=8192)):
                f.write(chunk)
        log.info(f"STRING v{version} download complete!")

    def create_string_graphs(self, file_path: str, version: str) -> None:
        """
        Create GeneGraph objects from STRING data and save them to attributes.
        Strip the '4932.' prefix from protein IDs for consistency.
        Only include nodes that exist in genome.gene_set.
        """
        # Load the data
        df = pd.read_csv(file_path, sep=" ", compression="gzip")

        # Network types we're interested in - combined is removed
        network_types = [
            "neighborhood",
            "fusion",
            "cooccurence",
            "coexpression",
            "experimental",
            "database",
        ]

        # Format the version string for attribute names
        version_str = version.replace(".", "_")

        # Process each individual network type
        for network_type in network_types:
            # Create a graph for each network type
            attr_name = f"_G_string{version_str}_{network_type}"
            nx_graph = nx.Graph()

            # Filter interactions where the network type score > 0
            type_df = df[df[network_type] > 0][["protein1", "protein2", network_type]]

            # Add edges to the individual graph with the weight attribute
            for _, row in type_df.iterrows():
                # Strip the prefixes from both protein IDs
                protein1 = self.strip_string_prefix(row["protein1"])
                protein2 = self.strip_string_prefix(row["protein2"])

                # Only add edges if both nodes exist in genome.gene_set
                if (protein1 in self.genome.gene_set and 
                    protein2 in self.genome.gene_set):
                    nx_graph.add_edge(
                        protein1, protein2, weight=row[network_type], version=version
                    )

            # Create a GeneGraph and save it
            gene_graph = GeneGraph(
                name=f"string{version_str}_{network_type}",
                graph=nx_graph,
                max_gene_set=self.genome.gene_set,
            )
            setattr(self, attr_name, gene_graph)

            # Save the graph
            graph_name = f"G_string{version_str}_{network_type}"
            self.save_graph(gene_graph, graph_name, root_type="string")
            log.info(
                f"STRING v{version} {network_type} network: {gene_graph.graph.number_of_nodes()} nodes, {gene_graph.graph.number_of_edges()} edges"
            )
    def strip_string_prefix(self, protein_id: str) -> str:
        """Strip the '4932.' prefix from STRING protein IDs"""
        if protein_id.startswith("4932."):
            return protein_id[5:]
        return protein_id

    def create_G_tflink(self) -> nx.DiGraph:
        """
        Create a directed graph of transcription factor interactions from TFLink data.
        Only include interactions where both TF and target are in genome.gene_set.
        Convert alias gene names to systematic names if possible.

        Returns:
            nx.DiGraph: Directed graph of TF interactions
        """
        G_tflink = nx.DiGraph()

        # Check if we have alias mapping
        has_alias_mapping = hasattr(self.genome, "alias_to_systematic")
        if not has_alias_mapping:
            log.warning(
                "Missing alias_to_systematic mapping in genome. TFLink functionality will be limited."
            )

        # Path to TFLink TSV file
        tflink_file = osp.join(
            self.tflink_root,
            "TFLink_Saccharomyces_cerevisiae_interactions_All_simpleFormat_v1.0.tsv",
        )

        # If the file doesn't exist, download it
        if not osp.exists(tflink_file):
            log.info(f"TFLink file not found at {tflink_file}, downloading...")
            self.download_tflink_data(tflink_file)

        if not osp.exists(tflink_file):
            log.warning(
                f"TFLink file {tflink_file} still not found after download attempt! Cannot create G_tflink."
            )
            return G_tflink

        # Load the TFLink data
        df = pd.read_csv(tflink_file, sep="\t")
        log.info(f"Loaded TFLink data with {len(df)} interactions")

        # Add counters for diagnostic purposes
        tf_not_found = 0
        target_not_found = 0
        not_in_gene_set = 0
        edges_added = 0

        # If we don't have alias mapping, create a simple mapping function
        if not has_alias_mapping:
            # Create a placeholder to map known systematic names
            # (this assumes TF names in dataset might already be systematic names in some cases)
            def get_systematic_names(name):
                if name in self.genome.gene_set:
                    return [name]  # If it's already a systematic name in gene_set
                return []  # Otherwise we can't map it

        else:
            # Use the full mapping
            def get_systematic_names(name):
                return self.genome.alias_to_systematic.get(name, [])

        # Process each interaction
        for _, row in tqdm(
            df.iterrows(), total=len(df), desc="Processing TF interactions"
        ):
            tf_name = row["Name.TF"]
            target_name = row["Name.Target"]

            # Convert alias to systematic names using our mapping function
            tf_systematic_names = get_systematic_names(tf_name)
            target_systematic_names = get_systematic_names(target_name)

            if not tf_systematic_names:
                tf_not_found += 1
                continue

            if not target_systematic_names:
                target_not_found += 1
                continue

            # For each combination of TF and target systematic names, add an edge
            edge_added_for_pair = False
            for tf_sys in tf_systematic_names:
                for target_sys in target_systematic_names:
                    # Only add edge if both genes are in the gene set
                    if (
                        tf_sys in self.genome.gene_set
                        and target_sys in self.genome.gene_set
                    ):
                        # Add edge with all metadata from the row
                        edge_data = row.to_dict()
                        # Add systematic names to edge data
                        edge_data["TF_systematic"] = tf_sys
                        edge_data["Target_systematic"] = target_sys

                        G_tflink.add_edge(tf_sys, target_sys, **edge_data)
                        edges_added += 1
                        edge_added_for_pair = True
                    else:
                        not_in_gene_set += 1

            if not edge_added_for_pair:
                log.debug(f"No valid edges added for TF {tf_name} -> {target_name}")

        log.info(
            f"TF interaction network: {G_tflink.number_of_nodes()} nodes, {G_tflink.number_of_edges()} edges"
        )
        log.info(
            f"TF name not found: {tf_not_found}, Target name not found: {target_not_found}"
        )
        log.info(
            f"Gene pairs not in gene_set: {not_in_gene_set}, Edges added: {edges_added}"
        )

        return G_tflink

    def save_graph(self, graph, graph_name, root_type="sgd"):
        """
        Save graph to a pickle file. Handles both NetworkX graphs and GeneGraph objects.

        Args:
            graph: NetworkX graph or GeneGraph to save
            graph_name: Name of the graph
            root_type: Type of root directory ('sgd', 'string', or 'tflink')
        """
        if root_type == "sgd":
            dir = osp.join(self.sgd_root, "graph")
        elif root_type == "string":
            dir = osp.join(self.string_root, "graph")
        else:  # tflink
            dir = osp.join(self.tflink_root, "graph")

        os.makedirs(dir, exist_ok=True)
        path = osp.join(dir, f"{graph_name}.pkl")

        with open(path, "wb") as f:
            pickle.dump(graph, f)

    def load_graph(self, graph_name, root_type="sgd"):
        """
        Load graph from a pickle file. Could be a NetworkX graph or a GeneGraph.

        Args:
            graph_name: Name of the graph
            root_type: Type of root directory ('sgd', 'string', or 'tflink')

        Returns:
            NetworkX graph or GeneGraph object
        """
        if root_type == "sgd":
            dir = osp.join(self.sgd_root, "graph")
        elif root_type == "string":
            dir = osp.join(self.string_root, "graph")
        else:  # tflink
            dir = osp.join(self.tflink_root, "graph")

        path = osp.join(dir, f"{graph_name}.pkl")

        if not os.path.exists(path):
            log.warning(f"Graph file {path} not found!")
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

        return G

    # GO properties and methods
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

    # Edge Features methods
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

    # Node Features methods
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

    @staticmethod
    def add_loci_information(G_raw: nx.Graph, G_gene: nx.Graph) -> nx.Graph:
        loci_information_template = {"start": None, "end": None, "chromosome": None}
        for node_name, node_data in G_raw.nodes(data=True):
            loci_information = loci_information_template.copy()
            for i in node_data["sequence_details"]["genomic_dna"]:
                if i["strain"]["display_name"] == "S288C":
                    loci_information["start"] = i.get("start")
                    loci_information["end"] = i.get("end")
                    loci_information["chromosome"] = i["contig"]["display_name"]
                    G_gene.add_node(node_name, **loci_information)
        return G_gene

    @staticmethod
    def add_pathway_annotation(G_raw: nx.Graph, G_gene: nx.Graph) -> nx.Graph:
        pathway_annotation_template = {"pathways": None}
        for node_name, node_data in G_raw.nodes(data=True):
            pathway_annotation = pathway_annotation_template.copy()
            pathways = node_data["locus"].get("pathways")
            if pathways != []:
                pathway_annotation["pathways"] = []
                for pathway in pathways:
                    pathway_annotation["pathways"].append(
                        pathway["pathway"]["display_name"]
                    )
                G_gene.add_node(node_name, **pathway_annotation)
            else:
                G_gene.add_node(node_name, **pathway_annotation)
        return G_gene


SCEREVISIAE_GENE_GRAPH_MAP = {
    "physical": lambda graph: graph.G_physical,
    "regulatory": lambda graph: graph.G_regulatory,
    "genetic": lambda graph: graph.G_genetic,
    "tflink": lambda graph: graph.G_tflink,
    "string9_1_neighborhood": lambda graph: graph.G_string9_1_neighborhood,
    "string9_1_fusion": lambda graph: graph.G_string9_1_fusion,
    "string9_1_cooccurence": lambda graph: graph.G_string9_1_cooccurence,
    "string9_1_coexpression": lambda graph: graph.G_string9_1_coexpression,
    "string9_1_experimental": lambda graph: graph.G_string9_1_experimental,
    "string9_1_database": lambda graph: graph.G_string9_1_database,
    "string12_0_neighborhood": lambda graph: graph.G_string12_0_neighborhood,
    "string12_0_fusion": lambda graph: graph.G_string12_0_fusion,
    "string12_0_cooccurence": lambda graph: graph.G_string12_0_cooccurence,
    "string12_0_coexpression": lambda graph: graph.G_string12_0_coexpression,
    "string12_0_experimental": lambda graph: graph.G_string12_0_experimental,
    "string12_0_database": lambda graph: graph.G_string12_0_database,
}

# List of all valid graph types
SCEREVISIAE_GENE_GRAPH_VALID_NAMES = list(SCEREVISIAE_GENE_GRAPH_MAP.keys())


def build_gene_multigraph(
    graph: "SCerevisiaeGraph", graph_names: Optional[list[str]] = None
) -> Optional[GeneMultiGraph]:
    """
    Build a GeneMultiGraph containing GeneGraph objects based on the provided graph names.
    Only loads the specific graphs requested, avoiding unnecessary computation.

    Args:
        graph: An SCerevisiaeGraph instance containing various graph types
        graph_names: List of graph names to include in the multigraph.
                     If None, returns None.

    Returns:
        A GeneMultiGraph containing the specified graphs,
        or None if graph_names is None.

    Raises:
        ValueError: If any graph name in graph_names is not a valid graph type.
    """
    if graph_names is None:
        return None

    # Validate all graph names before proceeding
    invalid_graph_names = [
        name for name in graph_names if name not in SCEREVISIAE_GENE_GRAPH_MAP
    ]
    if invalid_graph_names:
        
        raise ValueError(
            f"Invalid graph type(s): {', '.join(invalid_graph_names)}. "
            f"Valid types are: {', '.join(SCEREVISIAE_GENE_GRAPH_VALID_NAMES)}"
        )

    graphs_dict = SortedDict()

    # Only load the requested graphs
    for name in graph_names:
        # Access the property (which loads the graph) if requested
        graph_obj = SCEREVISIAE_GENE_GRAPH_MAP[name](graph)
        if graph_obj is not None:
            graphs_dict[name] = graph_obj
        else:
            log.warning(f"Graph '{name}' is None and will not be included")

    # Create and return the GeneMultiGraph
    return GeneMultiGraph(graphs=graphs_dict)


def check_regulatory_nodes_have_edges() -> None:
    """
    Check if all nodes in the Regulatory graph have at least one edge that is not a self-edge.
    Creates a new SCerevisiaeGraph instance using environment variables.
    """
    import os
    import os.path as osp
    from dotenv import load_dotenv
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

    # Load environment variables
    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")

    # Create genome and graph
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,  # Use existing data to avoid rebuilding
    )
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )

    # Get the underlying NetworkX graph
    G_regulatory = graph.G_regulatory.graph

    nodes_without_proper_edges = []
    for node in G_regulatory.nodes():
        # Get all neighbors of the node
        neighbors = set(G_regulatory.predecessors(node)).union(
            set(G_regulatory.successors(node))
        )

        # Check if all edges are self-loops
        if len(neighbors) == 0 or (len(neighbors) == 1 and node in neighbors):
            nodes_without_proper_edges.append(node)

    if nodes_without_proper_edges:
        print(
            f"Found {len(nodes_without_proper_edges)} nodes without non-self edges in the Regulatory graph:"
        )
        # Print a sample of problematic nodes (up to 10)
        sample = nodes_without_proper_edges[:10]
        print(f"Sample: {sample}")
    else:
        print(
            "All nodes in the Regulatory graph have at least one connection to another node."
        )


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
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=True,
    )
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )
    # graph.read_raw()

    print("-----------")
    print(
        f"G_string9_1_neighborhood: {set(graph.G_string9_1_neighborhood.nodes()) - genome.gene_set}"
    )
    print(
        f"G_string9_1_fusion: {set(graph.G_string9_1_fusion.nodes()) - genome.gene_set}"
    )
    print(
        f"G_string9_1_cooccurence: {set(graph.G_string9_1_cooccurence.nodes()) - genome.gene_set}"
    )
    print(
        f"G_string9_1_coexpression: {set(graph.G_string9_1_coexpression.nodes()) - genome.gene_set}"
    )
    print(
        f"G_string9_1_experimental: {set(graph.G_string9_1_experimental.nodes()) - genome.gene_set}"
    )
    print(
        f"G_string9_1_database: {set(graph.G_string9_1_database.nodes()) - genome.gene_set}"
    )
    print(
        f"G_string9_1_combined: {set(graph.G_string9_1_combined.nodes()) - genome.gene_set}"
    )
    print("-----------")
    print(
        f"G_string12_0_neighborhood: {set(graph.G_string12_0_neighborhood.nodes()) - genome.gene_set}"
    )
    print(
        f"G_string12_0_fusion: {set(graph.G_string12_0_fusion.nodes()) - genome.gene_set}"
    )
    print(
        f"G_string12_0_cooccurence: {set(graph.G_string12_0_cooccurence.nodes()) - genome.gene_set}"
    )
    print(
        f"G_string12_0_coexpression: {set(graph.G_string12_0_coexpression.nodes()) - genome.gene_set}"
    )
    print(
        f"G_string12_0_experimental: {set(graph.G_string12_0_experimental.nodes()) - genome.gene_set}"
    )
    print(
        f"G_string12_0_database: {set(graph.G_string12_0_database.nodes()) - genome.gene_set}"
    )
    print(
        f"G_string12_0_combined: {set(graph.G_string12_0_combined.nodes()) - genome.gene_set}"
    )
    print("-----------")
    print(graph.G_tflink.number_of_nodes())
    print(graph.G_tflink.number_of_edges())
    print("-----------")


if __name__ == "__main__":
    # main()
    check_regulatory_nodes_have_edges()
