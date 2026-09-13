# torchcell/adapters/auesukaree2009_adapter.py
# [[torchcell.adapters.auesukaree2009_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/auesukaree2009_adapter.py
# Test file: tests/torchcell/adapters/test_auesukaree2009_adapter.py
"""BioCypher adapter exposing the Auesukaree 2009 six-stress sensitivity screen.

Each record is one (KanMX deletion strain, stress) pair. The genotype is gene-keyed, so it
serves as a ``genotype`` node plus a ``perturbation`` node and the edge between them. The
condition is an ``environment`` with its media and temperature as first-class nodes; the
five chemical stresses additionally emit an ``environment perturbation`` node, while the
heat stress deliberately emits none (its edit is the raised temperature, M2). The readout
is an ``environment response phenotype`` whose ``category`` / ``category_label`` carry the
typed call and the paper's own word.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.scerevisiae.auesukaree2009 import (
    EnvChemgenAuesukaree2009Dataset,
)


class EnvChemgenAuesukaree2009Adapter(CellAdapter):
    """Cell adapter serving the Auesukaree 2009 stress-sensitivity dataset."""

    def __init__(
        self,
        dataset: EnvChemgenAuesukaree2009Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(current_dir, "conf", "auesukaree2009_adapter.yaml")
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
