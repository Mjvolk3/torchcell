# src/torchcell/datasets/cell.py
# [[src.torchcell.datasets.cell]]
# https://github.com/Mjvolk3/torchcell/tree/main/src/torchcell/datasets/cell.py
# Test file: src/torchcell/datasets/test_cell.py

import copy
import os
import shutil
import zipfile
from abc import ABC, abstractmethod
from collections.abc import Callable
from os import environ
from typing import List, Optional, Tuple, Union

import pandas as pd
import torch
from attrs import define
from sklearn import experimental
from torch_geometric.data import Batch, Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.data.separate import separate
from torch_geometric.utils import subgraph
from tqdm import tqdm

from torchcell.datasets.fungal_utr_transformer import FungalUtrTransformerDataset
from torchcell.datasets.nucleotide_embedding import BaseEmbeddingDataset
from torchcell.datasets.nucleotide_transformer import NucleotideTransformerDataset
from torchcell.datasets.scerevisiae import (  # DMFCostanzo2016Dataset,
    DMFCostanzo2016SmallDataset,
    SMFCostanzo2016Dataset,
)
from torchcell.models import FungalUtrTransformer, NucleotideTransformer
from torchcell.models.llm import NucleotideModel
from torchcell.models.nucleotide_transformer import NucleotideTransformer
from torchcell.sequence import Genome
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome


class DiMultiGraph:
    pass


class Ontology:
    pass


class CellDataset(InMemoryDataset):
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

        # Create the seq graph
        if self.seq_embeddings:
            self.seq_graph = self.create_seq_graph(self.seq_embeddings)

        # This is here because we can't run process without getting gene_set
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self) -> list[str]:
        return ["cell.pt"]

    # TODO think more on this method
    def create_seq_graph(self, seq_embeddings: BaseEmbeddingDataset) -> Data:
        """
        Create a graph from seq_embeddings.
        """
        # Extract and concatenate embeddings for all items in seq_embeddings
        embeddings = []
        for item in seq_embeddings:
            keys = item["embeddings"].keys()
            item_embeddings = [item["embeddings"][k].squeeze(0) for k in keys]
            embeddings.append(torch.cat(item_embeddings))

        # Stack the embeddings to get a 2D tensor of shape [num_nodes, num_features]
        embeddings = torch.stack(embeddings, dim=0)

        # Extract ids for all items in seq_embeddings
        ids = [item["id"] for item in seq_embeddings]

        # Create a dummy edge_index (no edges)
        edge_index = torch.empty((2, 0), dtype=torch.long)

        # Create a Data object with embeddings as node features, ids as an attribute, and the dummy edge_index
        data = Data(x=embeddings, id=ids, edge_index=edge_index)

        return data

    def process(self):
        # Start with an empty list for filtered data
        filtered_data_list = []
        self.gene_set
        for data_item in tqdm(
            self.experiments
        ):  # assuming experiments contains the data
            # Check if data_item's ID is in the gene_set
            item_id_set = {i["id"] for i in data_item.genotype}
            if len(item_id_set.intersection(self.gene_set)) > 0:
                filtered_data_list.append(data_item)
        #######
        # TODO remove this safety check
        # for item in tqdm(filtered_data_list):
        #     if (
        #         len(item.genotype) != 1
        #         or "smf_fitness" not in item.phenotype["observation"]
        #     ):
        #         print(item)
        #########
        # Save this filtered data to a processed file
        torch.save(
            self.collate(filtered_data_list),
            os.path.join(self.processed_dir, "cell.pt"),
        )

    @property
    def gene_set(self):
        if self._gene_set is None:
            self._gene_set = self.compute_gene_set()
        return self._gene_set

    @gene_set.setter
    def gene_set(self, value):
        self._gene_set = value

    def compute_gene_set(self):
        if not self._gene_set:
            if isinstance(self.experiments, InMemoryDataset):
                experiment_gene_set = self.experiments.gene_set
            else:
                # TODO: handle other data types for experiments, if necessary
                raise NotImplementedError(
                    "Expected 'experiments' to be of type InMemoryDataset"
                )
            cell_gene_set = set(self.genome.gene_set).intersection(experiment_gene_set)
        return cell_gene_set

    def _subset_graph(self, data: Data) -> Data:
        """
        Subset the reference graph based on the genes in data.genotype.
        """
        # Nodes to remove based on the genes in data.genotype
        nodes_to_remove = [
            self.seq_graph.id.index(gene["id"])
            for gene in data.genotype
            if gene["id"] in self.seq_graph.id
        ]
        nodes_to_remove_tensor = torch.tensor(nodes_to_remove, dtype=torch.long)

        # Compute the nodes to keep
        all_nodes = torch.arange(self.seq_graph.num_nodes, dtype=torch.long)
        nodes_to_keep = torch.tensor(
            [node for node in all_nodes if node not in nodes_to_remove_tensor],
            dtype=torch.long,
        )

        # Get the induced subgraph using the nodes to keep
        return self.seq_graph.subgraph(nodes_to_keep)

    def _add_label(self, data: Data, original_data: Data) -> Data:
        """
        Adds the dmf_fitness label to the data object if it exists in the original data's phenotype["observation"].

        Args:
            data (Data): The Data object to which the label should be added.
            original_data (Data): The original Data object from which the label should be extracted.

        Returns:
            Data: The modified Data object with the added label.
        """
        if "dmf_fitness" in original_data.phenotype["observation"]:
            data.dmf_fitness = original_data.phenotype["observation"]["dmf_fitness"]
        return data

    def get(self, idx: int) -> Data:
        # If there's only one item in the dataset, return the entire reference graph
        if self.len() == 1:
            return copy.copy(self._data)

        # Check if the data list has been initialized or if the item at the current index is already cached
        if not hasattr(self, "_data_list") or self._data_list is None:
            self._data_list = self.len() * [None]
        elif self._data_list[idx] is not None:
            return copy.copy(self._data_list[idx])

        # Get the data object for the current index
        data = separate(
            cls=self._data.__class__,
            batch=self._data,
            idx=idx,
            slice_dict=self.slices,
            decrement=False,
        )

        # Get the subset data using the separate method
        subset_data = self._subset_graph(data)

        # Add the dmf_fitness label to the subset_data
        subset_data = self._add_label(subset_data, data)

        # Cache the subset data for future accesses
        # breakpoint here, investigating why size of x is same.
        # TODO if the gene is not in genome, then the node is not removed... this leads to some misrepresentation.
        # BUG
        # assert len(self.seq_graph.x) - len(data.genotype) == len(
        #     subset_data.x
        # ), "nodes not removed"
        self._data_list[idx] = copy.copy(subset_data)

        return subset_data


def main():
    genome = SCerevisiaeGenome()
    # nucleotide transformer
    nt_dataset = NucleotideTransformerDataset(
        root="data/scerevisiae/nucleotide_transformer_embed",
        genome=genome,
        transformer_model_name="nt_window_5979",
    )
    fut3_dataset = FungalUtrTransformerDataset(
        root="data/scerevisiae/fungal_utr_embed",
        genome=genome,
        transformer_model_name="fut_species_window_3utr_300_undersize",
    )
    fut5_dataset = FungalUtrTransformerDataset(
        root="data/scerevisiae/fungal_utr_embed",
        genome=genome,
        transformer_model_name="fut_species_window_5utr_1000_undersize",
    )
    seq_embeddings = nt_dataset + fut3_dataset + fut5_dataset

    cell_dataset = CellDataset(
        root="data/scerevisiae/cell",
        genome=SCerevisiaeGenome(),
        seq_embeddings=seq_embeddings,
        experiments=DMFCostanzo2016SmallDataset(),
    )

    print(cell_dataset)
    print(cell_dataset[0])
    print()


if __name__ == "__main__":
    main()
