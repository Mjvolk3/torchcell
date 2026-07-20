"""Base genome dataset abstractions for in-memory genome graphs."""

# torchcell/datasets/genome.py
# [[torchcell.datasets.genome]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/genome.py
# Test file: torchcell/datasets/test_genome.py

from abc import ABC

from torch_geometric.data import InMemoryDataset


class GenomeDataset(InMemoryDataset, ABC):  # type: ignore[misc]  # PyG InMemoryDataset is untyped (Any) base
    """Abstract base for in-memory genome datasets."""

    pass

    def __add__(  # type: ignore[empty-body]  # stub, body intentionally empty
        self, other: "GenomeDataset"
    ) -> "GenomeDataset":
        """Combine two genome datasets (not yet implemented)."""
        pass


class SCerevisiaeS88C(GenomeDataset):
    """Genome dataset for the S. cerevisiae S288C reference strain."""

    pass


if __name__ == "__main__":
    pass
