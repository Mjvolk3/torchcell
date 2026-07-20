# torchcell/adapters/sameith2015_adapter.py
# [[torchcell.adapters.sameith2015_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/sameith2015_adapter.py
# Test file: tests/torchcell/adapters/test_sameith2015_adapter.py
"""BioCypher adapter(s) exposing Sameith 2015 double-mutant microarray expression; Sameith 2015 single-mutant microarray expression as knowledge-graph nodes and edges."""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.scerevisiae.sameith2015 import (
    DmMicroarraySameith2015Dataset,
    SmMicroarraySameith2015Dataset,
)


class SmMicroarraySameith2015Adapter(CellAdapter):
    """Cell adapter that serves the Sameith 2015 single-mutant microarray expression dataset to BioCypher."""

    def __init__(
        self,
        dataset: SmMicroarraySameith2015Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "sm_microarray_sameith2015_adapter.yaml"
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


class DmMicroarraySameith2015Adapter(CellAdapter):
    """Cell adapter that serves the Sameith 2015 double-mutant microarray expression dataset to BioCypher."""

    def __init__(
        self,
        dataset: DmMicroarraySameith2015Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "dm_microarray_sameith2015_adapter.yaml"
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
