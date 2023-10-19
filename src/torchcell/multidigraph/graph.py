# src/torchcell/multidigraph/graph.py
# [[src.torchcell.multidigraph.graph]]
# https://github.com/Mjvolk3/torchcell/tree/main/src/torchcell/multidigraph/graph.py
# Test file: src/torchcell/multidigraph/test_graph.py

import glob
import gzip
import json
import logging
import os
import os.path as osp
import shutil
import tarfile
from itertools import product
from typing import Set

import gffutils
import networkx as nx
import pandas as pd
from attrs import define, field
from Bio import Seq, SeqIO
from Bio.SeqRecord import SeqRecord
from gffutils import FeatureDB
from gffutils.feature import Feature
from goatools.obo_parser import GODag
from sortedcontainers import SortedDict, SortedSet
from torch_geometric.data import download_url
from torch_geometric.utils import from_networkx
from tqdm import tqdm

from torchcell.sequence import GeneSet, Genome, ParsedGenome
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome


@define
class SCerevisiaeGraph:
    data_root: str = field(init=True, repr=False, default="data/sgd/genes")
    # genome: Genome = field(init=True, repr=False, default=None)
    gene_set: GeneSet = field(init=True, repr=False, default=None)
    G_raw: nx.Graph = field(init=False, repr=False, default=None)
    G_gene: nx.Graph = field(init=False, repr=False, default=None)
    G_physical: nx.Graph = field(init=False, repr=False, default=None)
    G_genetic: nx.Graph = field(init=False, repr=False, default=None)
    G_regulatory: nx.DiGraph = field(init=False, repr=False, default=None)
    json_files: list[str] = field(init=False, repr=False, default=None)
    # parsed_genome: ParsedGenome = field(init=True, repr=False, default=None)

    def __attrs_post_init__(self) -> None:
        # self.genome = self.parse_genome(self.genome)
        self.json_files = [f"{gene}.json" for gene in self.gene_set]
        self.G_raw = nx.Graph()
        self.G_raw = self.add_json_data_to_graph(
            data_root=self.data_root, json_files=self.json_files
        )
        # node
        self.G_gene = nx.Graph()
        self.G_gene = self.add_gene_protein_overview(
            G_raw=self.G_raw, G_gene=self.G_gene
        )
        # physical
        self.G_physical = nx.Graph()
        self.G_physical = self.add_physical_edges(
            G_raw=self.G_raw, G_physical=self.G_physical
        )

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

    @staticmethod
    def add_json_data_to_graph(data_root: str, json_files: list[str]) -> nx.Graph:
        G = nx.Graph()  # This is the node graph

        for i, json_file in tqdm(enumerate(json_files)):
            with open(osp.join(data_root, json_file)) as file:
                data = json.load(file)
                node_name = json_file.split(".")[0]
                G.add_node(node_name, **data)
                if i > 100:
                    # break
                    pass
        return G

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
                            # Need a mechanism to process the interaction data
                            # G_physical.add_edge(node1, node2, **interaction)
                            # from_networkx works
                            G_physical.add_edge(node1, node2)
        return G_physical

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


def main() -> None:
    import os
    import random

    import matplotlib.pyplot as plt
    import pandas as pd
    from dotenv import load_dotenv

    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")

    genome = SCerevisiaeGenome(
        data_root=osp.join(DATA_ROOT, "data/sgd/genome"), overwrite=True
    )
    graph = SCerevisiaeGraph(
        data_root=osp.join(DATA_ROOT, "data/sgd/genes"), gene_set=genome.gene_set
    )

    print()


if __name__ == "__main__":
    main()
