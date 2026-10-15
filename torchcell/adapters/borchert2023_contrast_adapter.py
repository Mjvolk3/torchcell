# torchcell/adapters/borchert2023_contrast_adapter.py
# [[torchcell.adapters.borchert2023_contrast_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/borchert2023_contrast_adapter.py
# Test file: tests/torchcell/adapters/test_borchert2023_contrast_adapter.py
"""BioCypher adapter exposing Borchert 2023's released two-arm fitness contrasts as
knowledge-graph nodes and edges.

One record per released (gene, comparison sheet) row: a single
``TransposonInsertionPerturbation`` served as a ``perturbation`` node, and
the two-sample t test of the gene's fitness between the arms as a
``FitnessContrastPhenotype`` (``fitness contrast phenotype``). The condition arm is the
record's environment and the reference arm the reference's.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.pputida.borchert2023_contrast import (
    FitnessContrastBorchert2023Dataset,
)


class FitnessContrastBorchert2023Adapter(CellAdapter):
    """Cell adapter that serves FitnessContrastBorchert2023Dataset to BioCypher."""

    def __init__(
        self,
        dataset: FitnessContrastBorchert2023Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "fitness_contrast_borchert2023_adapter.yaml"
        )
        if not osp.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path) as file:
            yaml_config = yaml.safe_load(file)
        config = OmegaConf.create(yaml_config)
        super().__init__(
            config, dataset, process_workers, io_workers, chunk_size, loader_batch_size
        )
        self.dataset = dataset
        self.process_workers = process_workers
        self.io_workers = io_workers
        self.chunk_size = chunk_size
        self.loader_batch_size = loader_batch_size
