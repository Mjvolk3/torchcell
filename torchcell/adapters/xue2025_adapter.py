# torchcell/adapters/xue2025_adapter.py
# [[torchcell.adapters.xue2025_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/xue2025_adapter.py
# Test file: tests/torchcell/adapters/test_xue2025_adapter.py
"""BioCypher adapter exposing the Xue 2025 free-fatty-acid titer dataset.

FFA titers over a combinatorial-deletion chassis are modeled as
``MetabolitePhenotype``. NOTE: in-house/unpublished data whose ``Publication`` uses
a methodology-anchor identifier -- whether it enters a shared graph is a per-build
decision, made at dataset selection, not by the adapter's existence.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.scerevisiae.xue2025 import FattyAcidXue2025Dataset


class FattyAcidXue2025Adapter(CellAdapter):
    """Cell adapter serving the Xue 2025 free-fatty-acid titer dataset to BioCypher."""

    def __init__(
        self,
        dataset: FattyAcidXue2025Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(current_dir, "conf", "ffa_xue2025_adapter.yaml")
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
