# src/torchcell/datasets/cell.py
import os
import shutil
import zipfile
from abc import ABC, abstractmethod
from os import environ
from typing import Callable, List, Optional, Tuple, Union

import pandas as pd
import torch
from attrs import define
from sklearn import experimental
from torch_geometric.data import (Data, InMemoryDataset, download_url,
                                  extract_zip)
from tqdm import tqdm

from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.datasets.scerevisiae import (DMFCostanzo2016Dataset,
                                            SMFCostanzo2016Dataset)
from torchcell.models import FungalUtrTransformer, NucleotideTransformer
from torchcell.models.llm import NucleotideModel
from torchcell.sequence import Genome


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
        nucleotide_model: Optional[NucleotideModel] = None,
        experiments: Union[List[InMemoryDataset], InMemoryDataset] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        self._gene_set = None
        self.genome = genome
        self.nucleotide_model = nucleotide_model
        self.experiments = experiments  # After experiments runs I get that self.gene_set = {"YAL001C", "YBL007C"} which comes from the exerimental.gene_set property
        self.dimultigraph: Optional[DiMultiGraph] = None
        self.experiment_datasets: Optional[List[InMemoryDataset]] = None
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

    def _process_genome_embeddings(self):
        """
        Process the reference genome and create embeddings for each gene.
        Returns a dictionary with gene IDs as keys and their embeddings as values.
        """
        embeddings = {}

        # TODO chunk 1 should be cpu setting
        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i : i + n]

        print("getting embeddings...")

        gene_ids = list(self.gene_set)
        chunk_size = 8

        for chunked_ids in tqdm(chunks(gene_ids, chunk_size)):
            # Generate sequences for the current chunk
            seqs = [
                self.genome[gene_id]
                .window(self.nucleotide_model.max_sequence_size, is_max_size=True)
                .seq
                for gene_id in chunked_ids
            ]

            nucleotide_model_embeddings = self.nucleotide_model.embed(
                seqs, mean_embedding=True
            )

            # Update the embeddings dictionary with the current chunk's results
            for gene_id, embedding in zip(chunked_ids, nucleotide_model_embeddings):
                embeddings[gene_id] = embedding

        return embeddings

    def get_embedding(self, gene_id: str) -> torch.Tensor:
        """
        Returns the embedding for the given gene_id.
        """
        return self.gene_embeddings.get(gene_id, None)

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
            item_id_set = set([i["id"] for i in data_item.genotype])
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
    print("Loading cell data...")

    cell_dataset = CellDataset(
        root="data/scerevisiae/cell",
        genome=SCerevisiaeGenome(),
        nucleotide_model=FungalUtrTransformer("downstream_300"),
        experiments=SMFCostanzo2016Dataset(),
    )

    print(cell_dataset)
    print(cell_dataset[0])
    print()


if __name__ == "__main__":
    main()
