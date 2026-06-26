# torchcell/cell/cell.py
"""Scratch sketch of cell dataset joining via ontology intersection."""

from collections.abc import Callable

from torch_geometric.data import Data, InMemoryDataset

from torchcell.data_prior.sequence import SCerevisiaeGenome
from torchcell.datasets.scerevisiae import DmfCostanzo2016Dataset
from torchcell.sequence import Genome


class DiMultiGraph:
    """Placeholder for a directed multigraph representation."""

    pass


class Ontology:
    """Represents the biological ontology guiding the data join."""

    def __init__(self):
        """Initialize an empty ontology."""
        # TODO: Initialize ontology
        pass

    def join(self, other: "Ontology") -> "Ontology":
        """Join this ontology with another and return the result."""
        # TODO: Implement ontology join logic
        pass


class CellDataset(InMemoryDataset):
    """Represents a dataset for cellular data."""

    def __init__(
        self,
        root: str,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
    ):
        """Set up the cell dataset and its optional join components."""
        super().__init__(root, transform, pre_transform, pre_filter)
        self.genome: Genome
        self.dimultigraph: DiMultiGraph | None = None
        self.experiment_datasets: list[InMemoryDataset] | None = None
        self.ontology: Ontology | None = None

    @property
    def processed_file_names(self) -> list[str]:
        """Return the processed file names for this dataset."""
        return ["cell.pt"]

    def process():
        """Process raw data into the cached dataset."""
        pass

    def __and__(self, other: "CellDataset") -> "IntersectionDataset":
        """Return the intersection of this dataset with another."""
        return IntersectionDataset(self, other)


class IntersectionDataset(CellDataset):
    """Represents a dataset formed by joining two CellDatasets."""

    def __init__(self, ds1: CellDataset, ds2: CellDataset):
        """Set up the intersection dataset from two cell datasets."""
        # TODO: Logic to initialize the IntersectionDataset using ds1 and ds2
        pass

    def _merge_dataset_indices(self):
        # TODO: Logic to merge indices from the two datasets
        pass

    def __getitem__(self, idx: int) -> Data:
        """Return the data item at the given index."""
        # TODO: Logic to get item given an index
        pass


# Convenience functions as described in the notes:


def visualize_ontology(ontology: Ontology):
    """Visualize the given ontology."""
    # TODO: Implement visualization
    pass


def compare_ontologies(ont1: Ontology, ont2: Ontology):
    """Compare two ontologies and return overlapping and conflicting parts."""
    # TODO: Implement comparison
    pass


def recall_dropped_data(dataset: IntersectionDataset):
    """Recall any data that was dropped during the join."""
    # TODO: Implement recall
    pass


def recall_transformed_data(dataset: IntersectionDataset):
    """Recall any data that was transformed during the join."""
    # TODO: Implement recall
    pass


# The main function and entry point:


def main():
    """Load the genome and build a cell dataset for S. cerevisiae."""
    print("cell loading")
    genome = SCerevisiaeGenome()
    cell = CellDataset(root="data/scerevisiae")
    cell.genome = genome
    cell.experiment_datasets = [DmfCostanzo2016Dataset()]


if __name__ == "__main__":
    main()
