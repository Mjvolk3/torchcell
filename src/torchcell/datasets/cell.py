# src/torchcell/datasets/cell.py
# [[src.torchcell.datasets.cell]]
# https://github.com/Mjvolk3/torchcell/tree/main/src/torchcell/datasets/cell.py
# Test file: src/torchcell/datasets/test_cell.py
import copy
import json
import logging
import os
import os.path as osp
import pickle
import re
import shutil
import threading
import zipfile
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from os import environ
from typing import List, Optional, Tuple, Union

import lmdb
import pandas as pd
import torch
from attrs import define
from sklearn import experimental
from sortedcontainers import SortedDict, SortedSet
from torch_geometric.data import Batch, Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.data.separate import separate
from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph
from tqdm import tqdm

from torchcell.data import Dataset
from torchcell.datasets.fungal_utr_transformer import FungalUtrTransformerDataset
from torchcell.datasets.nucleotide_embedding import BaseEmbeddingDataset
from torchcell.datasets.nucleotide_transformer import NucleotideTransformerDataset
from torchcell.datasets.scerevisiae import (
    DMFCostanzo2016Dataset,
    SMFCostanzo2016Dataset,
)
from torchcell.models import FungalUtrTransformer, NucleotideTransformer
from torchcell.models.llm import NucleotideModel
from torchcell.models.nucleotide_transformer import NucleotideTransformer
from torchcell.prof import prof, prof_input
from torchcell.sequence import Genome
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

log = logging.getLogger(__name__)


class DiMultiGraph:
    pass


class Ontology:
    pass


class CellDataset(Dataset):
    """
    Represents a dataset for cellular data.
    """

    def __init__(
        self,
        root: str = "data/scerevisiae/cell",
        genome: Genome = None,
        seq_embeddings: BaseEmbeddingDataset | None = None,
        experiments: list[InMemoryDataset] | InMemoryDataset = None,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
    ):
        self._gene_set = None
        self.genome = genome
        self.seq_embeddings = seq_embeddings
        self.experiments = experiments
        self.dimultigraph: DiMultiGraph | None = None
        self.experiment_datasets: list[InMemoryDataset] | None = None
        self.ontology: Ontology | None = None

        # TODO consider moving to Dataset
        self.preprocess_dir = osp.join(root, "preprocess")

        # This is here because we can't run process without getting gene_set
        super().__init__(root, transform, pre_transform, pre_filter)

        # Create the seq graph
        if self.seq_embeddings:
            self.seq_graph = self.create_seq_graph(self.seq_embeddings)

        # Handle LMDB database
        self.env = lmdb.open(self.processed_paths[0], readonly=True, lock=False)

    @property
    def raw_file_names(self) -> list[str]:
        # TODO consider return the processed of the experiments, etc.
        return None  # Specify raw files if needed

    @property
    def processed_file_names(self) -> list[str]:
        return "data.lmdb"

    # TODO think more on this method
    def create_seq_graph(self, seq_embeddings: BaseEmbeddingDataset) -> Data:
        """
        Create a graph from seq_embeddings.
        """
        # Extract and concatenate embeddings for all items in seq_embeddings
        embeddings = []
        ids = []
        for item in seq_embeddings:
            keys = item["embeddings"].keys()
            if item.id in self.genome.gene_set:
                # TODO using self.genome.gene_set since this is the super set of genes.
                ids.append(item.id)
                item_embeddings = [item["embeddings"][k].squeeze(0) for k in keys]
                embeddings.append(torch.cat(item_embeddings))

        # Stack the embeddings to get a 2D tensor of shape [num_nodes, num_features]
        embeddings = torch.stack(embeddings, dim=0)

        # Create a dummy edge_index (no edges)
        edge_index = torch.empty((2, 0), dtype=torch.long)

        # Create a Data object with embeddings as node features
        # ids as an attribute, and the dummy edge_index
        data = Data(x=embeddings, ids=SortedSet(ids), edge_index=edge_index)

        return data

    @prof
    def process(self):
        self.gene_set = self.compute_gene_set()
        gene_set = self.gene_set

        # Function to filter experiments
        def filter_experiment(item):
            result = any(i["id"] in gene_set for i in item.genotype)
            with lock:
                pbar.update(1)
            return result

        # Initialize a thread-safe lock and a tqdm progress bar
        lock = threading.Lock()
        pbar = tqdm(total=len(self.experiments))

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor() as executor:
            # Use map to efficiently filter experiments
            filtered_experiments = list(
                filter(None, executor.map(filter_experiment, self.experiments))
            )

        pbar.close()

        # Your existing code to write to LMDB
        env = lmdb.open(osp.join(self.processed_dir, "data.lmdb"), map_size=int(1e12))
        with env.begin(write=True) as txn:
            for idx, data in enumerate(filtered_experiments):
                serialized_data = pickle.dumps(
                    data
                )  # Consider using a faster serialization method
                txn.put(f"{idx}".encode(), serialized_data)

    # def process(self):
    #     combined_data = []
    #     self.gene_set = self.compute_gene_set()

    #     # Precompute gene_set for faster lookup
    #     gene_set = self.gene_set

    #     # # Use list comprehension and any() for fast filtering
    #     combined_data = [
    #         item
    #         for item in tqdm(self.experiments)
    #         if any(i["id"] in gene_set for i in item.genotype)
    #     ]

    #     # TODO remove dev code
    #     # combined_data = []
    #     # for item in tqdm(self.experiments):
    #     #     if any(i["id"] in gene_set for i in item.genotype):
    #     #         combined_data.append(item)
    #     #     if len(combined_data) >= 100:
    #     #         break

    #     log.info("creating lmdb database")
    #     # Initialize LMDB environment
    #     env = lmdb.open(osp.join(self.processed_dir, "data.lmdb"), map_size=int(1e12))

    #     with env.begin(write=True) as txn:
    #         for idx, item in tqdm(enumerate(combined_data)):
    #             data = Data()
    #             data.genotype = item["genotype"]
    #             data.phenotype = item["phenotype"]

    #             # Serialize the data object using pickle
    #             serialized_data = pickle.dumps(data)

    #             # Save the serialized data in the LMDB environment
    #             txn.put(f"{idx}".encode(), serialized_data)

    # def process(self):
    #     self.gene_set = self.compute_gene_set()
    #     gene_set = self.gene_set

    #     # Use ThreadPoolExecutor for parallel processing
    #     with ThreadPoolExecutor() as executor:
    #         # Use filter and map to efficiently filter and process experiments
    #         filtered_experiments = filter(
    #             lambda item: any(i["id"] in gene_set for i in item.genotype),
    #             self.experiments,
    #         )
    #         processed_data = list(
    #             executor.map(self.process_experiment, filtered_experiments)
    #         )

    #     # Write to LMDB in one go (or in larger batches)
    #     env = lmdb.open(osp.join(self.processed_dir, "data.lmdb"), map_size=int(1e12))
    #     with env.begin(write=True) as txn:
    #         for idx, data in enumerate(processed_data):
    #             serialized_data = pickle.dumps(
    #                 data
    #             )  # Consider using a faster serialization method
    #             txn.put(f"{idx}".encode(), serialized_data)

    # def process_experiment(self, experiment):
    #     data = Data()
    #     data.genotype = experiment["genotype"]
    #     data.phenotype = experiment["phenotype"]
    #     return data

    @property
    def gene_set(self):
        try:
            if osp.exists(osp.join(self.preprocess_dir, "gene_set.json")):
                with open(osp.join(self.preprocess_dir, "gene_set.json")) as f:
                    self._gene_set = set(json.load(f))
            elif self._gene_set is None:
                raise ValueError(
                    "gene_set not written during process. "
                    "Please call compute_gene_set in process."
                )
            return self._gene_set
        except json.JSONDecodeError:
            raise ValueError("Invalid or empty JSON file found.")

    @gene_set.setter
    def gene_set(self, value):
        if not value:
            raise ValueError("Cannot set an empty or None value for gene_set")
        if not osp.exists(self.preprocess_dir):
            os.makedirs(self.preprocess_dir)
        with open(osp.join(self.preprocess_dir, "gene_set.json"), "w") as f:
            json.dump(list(sorted(value)), f, indent=0)
        self._gene_set = value

    def compute_gene_set(self):
        if not self._gene_set:
            if isinstance(self.experiments, Dataset):
                experiment_gene_set = self.experiments.gene_set
            else:
                # TODO: handle other data types for experiments, if necessary
                raise NotImplementedError(
                    "Expected 'experiments' to be of type InMemoryDataset"
                )
            # Not sure we shoudl take the intersection here...
            # Could use gene_set from genome instead, since this is base
            # In case of gene addition would need to update the gene_set
            # then cell_dataset should be max possible.
            cell_gene_set = set(self.genome.gene_set).intersection(experiment_gene_set)
        return cell_gene_set

    def _subset_graph(self, data: Data) -> Data:
        """
        Subset the reference graph based on the genes in data.genotype.
        """
        # Nodes to remove based on the genes in data.genotype
        nodes_to_remove = [
            self.seq_graph.ids.index(gene["id"])
            for gene in data.genotype
            if gene["id"] in self.seq_graph.ids
        ]
        perturbed_nodes = torch.tensor(nodes_to_remove, dtype=torch.long)

        # Compute the nodes to keep
        all_nodes = torch.arange(self.seq_graph.num_nodes, dtype=torch.long)
        nodes_to_keep = torch.tensor(
            [node for node in all_nodes if node not in perturbed_nodes],
            dtype=torch.long,
        )

        # Get the induced subgraph using the nodes to keep
        subset_graph = self.seq_graph.subgraph(nodes_to_keep)
        subset_graph.perturbed_nodes = perturbed_nodes
        return subset_graph

    def _add_label(self, data: Data, original_data: Data) -> Data:
        """
        Adds the dmf_fitness label to the data object if it exists in the original data's phenotype["observation"].

        Args:
            data (Data): The Data object to which the label should be added.
            original_data (Data): The original Data object from which the label should be extracted.

        Returns:
            Data: The modified Data object with the added label.
        """
        if "dmf" in original_data.phenotype["observation"]:
            data.dmf = original_data.phenotype["observation"]["dmf"]
        return data

    def get(self, idx: int) -> Data:
        with self.env.begin() as txn:
            serialized_data = txn.get(f"{idx}".encode())
            if serialized_data is None:
                return None
            data = pickle.loads(serialized_data)
            if self.transform:
                data = self.transform(data)

            # Get the subset data using the separate method
            subset_data = self._subset_graph(data)

            # Add the dmf_fitness label to the subset_data
            subset_data = self._add_label(subset_data, data)

            return subset_data

    def len(self):
        lmdb_path = os.path.join(self.processed_dir, "data.lmdb")
        if not os.path.exists(lmdb_path):
            raise FileNotFoundError(f"LMDB directory does not exist: {lmdb_path}")

        env = lmdb.open(lmdb_path, readonly=True)
        with env.begin() as txn:
            return txn.stat()["entries"]


def main():
    # genome
    genome = SCerevisiaeGenome()
    genome.drop_chrmt()
    genome.drop_empty_go()

    # nucleotide transformer
    # nt_dataset = NucleotideTransformerDataset(
    #     root="data/scerevisiae/nucleotide_transformer_embed",
    #     genome=genome,
    #     transformer_model_name="nt_window_5979",
    # )
    fut3_dataset = FungalUtrTransformerDataset(
        root="data/scerevisiae/fungal_utr_embed",
        genome=genome,
        transformer_model_name="fut_species_window_3utr_300_undersize",
    )
    # fut5_dataset = FungalUtrTransformerDataset(
    #     root="data/scerevisiae/fungal_utr_embed",
    #     genome=genome,
    #     transformer_model_name="fut_species_window_5utr_1000_undersize",
    # )
    # seq_embeddings = nt_dataset + fut3_dataset + fut5_dataset
    seq_embeddings = fut3_dataset

    cell_dataset = CellDataset(
        root="data/scerevisiae/cell_dl",
        genome=genome,
        seq_embeddings=seq_embeddings,
        experiments=DMFCostanzo2016Dataset(root="data/scerevisiae/costanzo2016"),
    )

    print(cell_dataset)
    print(cell_dataset.gene_set)
    print(cell_dataset[0])
    print()


if __name__ == "__main__":
    main()
