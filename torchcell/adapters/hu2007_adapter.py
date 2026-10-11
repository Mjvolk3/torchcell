# torchcell/adapters/hu2007_adapter.py
# [[torchcell.adapters.hu2007_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/hu2007_adapter.py
# Test file: tests/torchcell/adapters/test_hu2007_adapter.py
"""BioCypher adapter exposing Hu 2007 TF-deletion error-model scores as graph nodes."""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.scerevisiae.hu2007 import MicroarrayScoreHu2007Dataset


class MicroarrayScoreHu2007Adapter(CellAdapter):
    """Cell adapter that serves the Hu 2007 microarray score dataset to BioCypher."""

    def __init__(
        self,
        dataset: MicroarrayScoreHu2007Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "microarray_score_hu2007_adapter.yaml"
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
