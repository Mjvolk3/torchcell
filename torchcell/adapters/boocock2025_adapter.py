# torchcell/adapters/boocock2025_adapter.py
# [[torchcell.adapters.boocock2025_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/boocock2025_adapter.py
# Test file: tests/torchcell/adapters/test_boocock2025_adapter.py
"""BioCypher adapter exposing the Boocock 2025 single-cell segregant panels as
knowledge-graph nodes and edges.

Each record is one segregant: a de novo segregant cell (crosses A, B and 3004) or a
pooled, previously genotyped BY x RM segregant. The genotype is a ``segregant genotype``
node (a haplotype mosaic carrying the released posterior; no ``perturbation`` nodes, since
a segregant carries no gene-keyed perturbation), the culture is an ``environment`` with
its media and temperature, and the readout is a ``single cell expression phenotype``.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.scerevisiae.boocock2025 import Boocock2025Dataset


class Boocock2025Adapter(CellAdapter):
    """Cell adapter that serves the Boocock 2025 single-cell segregant panels to BioCypher."""

    def __init__(
        self,
        dataset: Boocock2025Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(current_dir, "conf", "boocock2025_adapter.yaml")
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
