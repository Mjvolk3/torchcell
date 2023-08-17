# src/torchcell/datasets/cell.py
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
from sympy import sequence
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from tqdm import tqdm

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
        self.experiments = experiments  # After experiments runs I get that self.gene_set = {"YAL001C", "YBL007C"} which comes from the exerimental.gene_set property
        self.dimultigraph: DiMultiGraph | None = None
        self.experiment_datasets: list[InMemoryDataset] | None = None
        self.ontology: Ontology | None = None

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        # Define embeddings file path
        self.embeddings_path = os.path.join(self.processed_dir, "gene_embeddings.pt")

        if os.path.exists(self.embeddings_path):
            self.gene_embeddings = torch.load(self.embeddings_path)
        else:
            self.gene_embeddings = self._process_genome_embeddings()
            torch.save(self.gene_embeddings, self.embeddings_path)

    @property
    def processed_file_names(self) -> list[str]:
        return ["cell.pt"]

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

        for item in tqdm(filtered_data_list):
            if (
                len(item.genotype) != 1
                or "smf_fitness" not in item.phenotype["observation"]
            ):
                print(item)
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
        experiments=SMFCostanzo2016Dataset(),
    )

    print(cell_dataset)
    print(cell_dataset[0])
    print()


if __name__ == "__main__":
    main()
