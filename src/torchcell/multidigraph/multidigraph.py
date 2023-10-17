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

from torchcell.sequence import Genome
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome


@define
class SCerevisiaeMultiDiGraph:
    data_root: str = field(init=True, repr=False, default="data/sgd/genes")
    genome: Genome = field(init=True, repr=True, default=None)

    def __attrs_post_init__(self) -> None:
        # Probably specify root dir.
        # Then call sgd.py api if all the genes don't exist to download data
        G = self.add_json_data_to_graph(self.data_root)
        print()

    @staticmethod
    def add_json_data_to_graph(data_root: str) -> nx.MultiDiGraph:
        # Create an empty MultiDiGraph
        G = nx.MultiDiGraph()

        # List all .json files in the directory
        json_files = [f for f in os.listdir(data_root) if f.endswith(".json")]

        # For each JSON file, read its content and add it to the graph
        for json_file in json_files:
            with open(osp.join(data_root, json_file)) as file:
                data = json.load(file)
                # The file name (without the .json extension) is used as the node name
                node_name = json_file[:-5]
                # Add node to the graph with attributes from the JSON data
                G.add_node(node_name, **data)

        return G


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
    SCerevisiaeMultiDiGraph(
        data_root=osp.join(DATA_ROOT, "data/sgd/genome"), genome=genome
    )


if __name__ == "__main__":
    pass
