# torchcell/adapters/cachera2023_adapter.py
# [[torchcell.adapters.cachera2023_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/cachera2023_adapter.py
# Test file: tests/torchcell/adapters/test_cachera2023_adapter.py
"""BioCypher adapter(s) exposing Cachera 2023 betaxanthin metabolite screen as knowledge-graph nodes and edges."""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.scerevisiae.cachera2023 import BetaxanthinCachera2023Dataset


class BetaxanthinCachera2023Adapter(CellAdapter):
    """Cell adapter that serves the Cachera 2023 betaxanthin metabolite screen dataset to BioCypher."""

    def __init__(
        self,
        dataset: BetaxanthinCachera2023Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "betaxanthin_cachera2023_adapter.yaml"
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
