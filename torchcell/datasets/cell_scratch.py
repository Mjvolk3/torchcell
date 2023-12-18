# torchcell/cell/cell.py
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
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip

from torchcell.data_prior.sequence import SCerevisiaeGenome
from torchcell.datasets.scerevisiae import (
    DmfCostanzo2016Dataset,
    SmfCostanzo2016Dataset,
)
from torchcell.sequence import Genome


class DiMultiGraph:
    pass


class Ontology:
    """
    Represents the biological ontology guiding the data join.
    """

    def __init__(self):
        # TODO: Initialize ontology
        pass

    def join(self, other: "Ontology") -> "Ontology":
        # TODO: Implement ontology join logic
        pass


class CellDataset(InMemoryDataset):
    """
    Represents a dataset for cellular data.
    """

    def __init__(
        self,
        root: str,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
    ):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.genome: Genome
        self.dimultigraph: DiMultiGraph | None = None
        self.experiment_datasets: list[InMemoryDataset] | None = None
        self.ontology: Ontology | None = None

    @property
    def processed_file_names(self) -> list[str]:
        return ["cell.pt"]

    def process():
        pass

    def __and__(self, other: "CellDataset") -> "IntersectionDataset":
        return IntersectionDataset(self, other)


class IntersectionDataset(CellDataset):
    """
    Represents a dataset formed by joining two CellDatasets.
    """

    def __init__(self, ds1: CellDataset, ds2: CellDataset):
        # TODO: Logic to initialize the IntersectionDataset using ds1 and ds2
        pass

    def _merge_dataset_indices(self):
        # TODO: Logic to merge indices from the two datasets
        pass

    def __getitem__(self, idx: int) -> Data:
        # TODO: Logic to get item given an index
        pass


# Convenience functions as described in the notes:


def visualize_ontology(ontology: Ontology):
    """
    Visualizes the given ontology.
    """
    # TODO: Implement visualization
    pass


def compare_ontologies(ont1: Ontology, ont2: Ontology):
    """
    Compares two ontologies and returns overlapping and conflicting parts.
    """
    # TODO: Implement comparison
    pass


def recall_dropped_data(dataset: IntersectionDataset):
    """
    Recalls any data that was dropped during the join.
    """
    # TODO: Implement recall
    pass


def recall_transformed_data(dataset: IntersectionDataset):
    """
    Recalls any data that was transformed during the join.
    """
    # TODO: Implement recall
    pass


# The main function and entry point:


def main():
    print("cell loading")
    genome = SCerevisiaeGenome()
    cell = CellDataset(root="data/scerevisiae")
    cell.genome = genome
    cell.experiment_datasets = [DmfCostanzo2016Dataset()]


if __name__ == "__main__":
    main()
