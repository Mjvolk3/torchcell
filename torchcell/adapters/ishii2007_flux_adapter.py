# torchcell/adapters/ishii2007_flux_adapter.py
# [[torchcell.adapters.ishii2007_flux_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/ishii2007_flux_adapter.py
# Test file: tests/torchcell/adapters/test_bacterial_adapters.py
"""BioCypher adapter exposing the Ishii 2007 13C net fluxes as knowledge-graph nodes
and edges.

The first adapter to serve ``flux phenotype``, a graph class that was declared and
given ``CellAdapter`` methods before any dataset used it. One record per BW25113
Keio disruptant: a fitted net-flux map over the release's own 43 central-carbon
reactions, as a percentage of the specific glucose uptake rate, with no interval
because the release publishes none.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.ecoli.ishii2007 import FluxIshii2007Dataset


class FluxIshii2007Adapter(CellAdapter):
    """Cell adapter that serves FluxIshii2007Dataset to BioCypher."""

    def __init__(
        self,
        dataset: FluxIshii2007Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(current_dir, "conf", "flux_ishii2007_adapter.yaml")
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
