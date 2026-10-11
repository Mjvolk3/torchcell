# torchcell/adapters/nguessan2025_adapter.py
# [[torchcell.adapters.nguessan2025_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/nguessan2025_adapter.py
# Test file: tests/torchcell/adapters/test_nguessan2025_adapter.py
"""BioCypher adapter exposing the N'Guessan 2025 pooled BY x RM segregants as
knowledge-graph nodes and edges.

Each record is one batch-1 strain of the Nguyen Ba 2022 BB-QTL panel: its significantly
assigned cells summed. The genotype is a ``segregant genotype`` node (the panel's
released posterior as a haplotype mosaic; no ``perturbation`` nodes), the culture is an
``environment`` (YPD; its temperature is a typed gap, so no temperature node is
emitted), and the readout is a ``single cell expression phenotype`` whose cell-cycle
partition is a typed gap.

Not in the dataset-to-adapter map until the walled Dryad genotypes are deposited and
the store builds (#1014).
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.scerevisiae.nguessan2025 import Nguessan2025Dataset


class Nguessan2025Adapter(CellAdapter):
    """Cell adapter that serves the N'Guessan 2025 pooled segregants to BioCypher."""

    def __init__(
        self,
        dataset: Nguessan2025Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(current_dir, "conf", "nguessan2025_adapter.yaml")
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
