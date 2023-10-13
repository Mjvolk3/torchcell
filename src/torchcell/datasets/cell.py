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
from pydantic import BaseModel, Extra, Field, ValidationError, validator
from sklearn import experimental
from torch_geometric.data import Batch, Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.data.separate import separate
from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph
from tqdm import tqdm

from torchcell.data import Dataset
from torchcell.datamodels import ModelStrictArbitrary
from torchcell.datasets.codon_frequency import CodonFrequencyDataset
from torchcell.datasets.embedding import BaseEmbeddingDataset
from torchcell.datasets.fungal_up_down_transformer import FungalUpDownTransformerDataset
from torchcell.datasets.nucleotide_transformer import NucleotideTransformerDataset
from torchcell.datasets.scerevisiae import (
    DmfCostanzo2016Dataset,
    SmfCostanzo2016Dataset,
)
from torchcell.models.llm import NucleotideModel
from torchcell.models.nucleotide_transformer import NucleotideTransformer
from torchcell.prof import prof, prof_input
from torchcell.sequence import GeneSet, Genome
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

log = logging.getLogger(__name__)


class DiMultiGraph:
    pass


class ParsedGenome(ModelStrictArbitrary):
    gene_set: GeneSet

    @validator("gene_set")
    def validate_gene_set(cls, value):
        if not isinstance(value, GeneSet):
            raise ValueError(f"gene_set must be a GeneSet, got {type(value).__name__}")
        return value


class CellDataset(Dataset):
    """
    Represents a dataset for cellular data.
    """

    def __init__(
        self,
        root: str = "data/scerevisiae/cell",
        # genome: Genome = None,
        genome: Genome = None,
        seq_embeddings: BaseEmbeddingDataset | None = None,
        experiments: list[InMemoryDataset] | InMemoryDataset = None,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
    ):
        self._gene_set = None
        # HACK start
        # Extract data from genome object, then remove for pickling with data loader
        self.genome = self.parse_genome(genome)
        del genome
        # self.genome = None
        # HACK end
        self.seq_embeddings = seq_embeddings
        self.experiments = experiments
        self.dimultigraph: DiMultiGraph | None = None
        self.experiment_datasets: list[InMemoryDataset] | None = None

        # TODO consider moving to Dataset
        self.preprocess_dir = osp.join(root, "preprocess")

        # This is here because we can't run process without getting gene_set
        super().__init__(root, transform, pre_transform, pre_filter)

        # Create WT

        # Create the seq graph
        if self.seq_embeddings:
            self.seq_graph = self.create_seq_graph(self.seq_embeddings)
        # LMDB env
        self.env = None

    @staticmethod
    def parse_genome(genome) -> ParsedGenome:
        data = {}
        data["gene_set"] = genome.gene_set
        return ParsedGenome(**data)

    @property
    def raw_file_names(self) -> list[str]:
        # TODO consider return the processed of the experiments, etc.
        # This might cause an issue because there is expected behavior for raw,# and this is not it.
        return None  # Specify raw files if needed

    @property
    def processed_file_names(self) -> list[str]:
        return "data.lmdb"

    @property
    def wt(self):
        # Need to be able to combine WTs into one WT
        # wts = [experiment.wt for experiment in self.experiments]
        # TODO aggregate WTS. For now just return the first one.
        wt = self.experiments.wt
        subset_data = self._subset_graph(wt)
        data = self._add_label(subset_data, wt)
        return data

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
                ids.append(item.id)
                item_embeddings = [item["embeddings"][k].squeeze(0) for k in keys]
                embeddings.append(torch.cat(item_embeddings))

        # Stack the embeddings to get a 2D tensor of shape [num_nodes, num_features]
        embeddings = torch.stack(embeddings, dim=0)

        # Create a dummy edge_index (no edges)
        edge_index = torch.empty((2, 0), dtype=torch.long)

        # Create a Data object with embeddings as node features
        # ids as an attribute, and the dummy edge_index
        # TODO cannot add SortedSet to ids since need to do for nt dataset
        data = Data(x=embeddings, ids=ids, edge_index=edge_index)

        return data

    def process(self):
        combined_data = []
        self.gene_set = self.compute_gene_set()

        # Precompute gene_set for faster lookup
        gene_set = self.gene_set

        # # Use list comprehension and any() for fast filtering
        combined_data = [
            item
            for item in tqdm(self.experiments)
            if all(i["id"] in gene_set for i in item.genotype)
        ]

        log.info("creating lmdb database")
        # Initialize LMDB environment
        env = lmdb.open(osp.join(self.processed_dir, "data.lmdb"), map_size=int(1e12))

        with env.begin(write=True) as txn:
            for idx, item in tqdm(enumerate(combined_data)):
                data = Data()
                data.genotype = item["genotype"]
                data.phenotype = item["phenotype"]

                # Serialize the data object using pickle
                serialized_data = pickle.dumps(data)

                # Save the serialized data in the LMDB environment
                txn.put(f"{idx}".encode(), serialized_data)

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
            return GeneSet(self._gene_set)
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
            # Not sure we should take the intersection here...
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
        nodes_to_remove = torch.tensor(
            [
                self.seq_graph.ids.index(gene["id"])
                for gene in data.genotype
                if gene["id"] in self.seq_graph.ids
            ],
            dtype=torch.long,
        )

        perturbed_nodes = nodes_to_remove.clone().detach()

        # Compute the nodes to keep
        all_nodes = torch.arange(self.seq_graph.num_nodes, dtype=torch.long)
        nodes_to_keep = torch.tensor(
            [node for node in all_nodes if node not in perturbed_nodes],
            dtype=torch.long,
        )

        # Get the induced subgraph using the nodes to keep
        subset_graph = self.seq_graph.subgraph(nodes_to_keep)
        subset_remove_graph = self.seq_graph.subgraph(nodes_to_remove)
        subset_graph.x_pert = subset_remove_graph.x
        subset_graph.ids_pert = subset_remove_graph.ids
        subset_graph.x_pert_idx = perturbed_nodes
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
            # TODO change dmf in costanzo to be fitness - need to standardize
            data.fitness = original_data.phenotype["observation"]["dmf"]
        if "fitness" in original_data.phenotype["observation"]:
            # TODO change dmf in costanzo to be fitness - need to standardize
            data.fitness = original_data.phenotype["observation"]["fitness"]
        if "genetic_interaction_score" in original_data.phenotype["observation"]:
            data.genetic_interaction_score = original_data.phenotype["observation"][
                "genetic_interaction_score"
            ]
        return data

    # def get(self, idx: int) -> Data:
    #     env = lmdb.open(self.processed_paths[0], readonly=True, lock=False)
    #     with env.begin() as txn:
    #         serialized_data = txn.get(f"{idx}".encode())
    #         if serialized_data is None:
    #             return None
    #         data = pickle.loads(serialized_data)
    #         if self.transform:
    #             data = self.transform(data)

    #         # Get the subset data using the separate method
    #         subset_data = self._subset_graph(data)

    #         # Add the dmf_fitness label to the subset_data
    #         subset_data = self._add_label(subset_data, data)

    #         return subset_data

    def get(self, idx):
        """Initialize LMDB if it hasn't been initialized yet."""
        if self.env is None:
            self._init_db()

        with self.env.begin() as txn:
            serialized_data = txn.get(f"{idx}".encode())
            if serialized_data is None:
                return None
            data = pickle.loads(serialized_data)
            if self.transform:
                data = self.transform(data)

            subset_data = self._subset_graph(data)
            subset_data = self._add_label(subset_data, data)
            return subset_data
            return subset_data

    def _init_db(self):
        """Initialize the LMDB environment."""
        self.env = lmdb.open(
            osp.join(self.processed_dir, "data.lmdb"),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    def len(self) -> int:
        if self.env is None:
            self._init_db()

        with self.env.begin() as txn:
            length = txn.stat()["entries"]

        # Must be closed for dataloader num_workers > 0
        self.close_lmdb()

        return length

    def close_lmdb(self):
        if self.env is not None:
            self.env.close()
            self.env = None


def main():
    # genome
    import os.path as osp

    from dotenv import load_dotenv

    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")
    genome = SCerevisiaeGenome(osp.join(DATA_ROOT, "data/sgd/genome"))
    # genome.drop_chrmt()
    # genome.drop_empty_go()

    # nucleotide transformer
    nt_dataset = NucleotideTransformerDataset(
        root="data/scerevisiae/nucleotide_transformer_embed",
        genome=genome,
        model_name="nt_window_5979",
    )

    fud3_dataset = FungalUpDownTransformerDataset(
        root="data/scerevisiae/fungal_up_down_embed",
        genome=genome,
        model_name="species_downstream",
    )
    fud5_dataset = FungalUpDownTransformerDataset(
        root="data/scerevisiae/fungal_up_down_embed",
        genome=genome,
        model_name="species_upstream",
    )
    codon_frequency_dataset = CodonFrequencyDataset(
        root="data/scerevisiae/codon_frequency", genome=genome
    )

    seq_embeddings = nt_dataset + fud3_dataset + fud5_dataset + codon_frequency_dataset

    # Experiments
    experiments = DmfCostanzo2016Dataset(
        preprocess={"duplicate_resolution": "low_dmf_std"},
        root=osp.join(DATA_ROOT, "data/scerevisiae/costanzo2016_1e3"),
        subset_n=1000,
    )
    # experiments = experiments[:2]
    cell_dataset = CellDataset(
        root="data/scerevisiae/cell_1e3",
        genome=genome,
        seq_embeddings=seq_embeddings,
        experiments=experiments,
    )

    print(cell_dataset)
    print(cell_dataset.gene_set)
    print(cell_dataset[0])
    print(cell_dataset.wt)
    print()


if __name__ == "__main__":
    main()
