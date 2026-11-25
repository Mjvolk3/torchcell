# torchcell/adapters/lim2025_tale_growth_rate_adapter.py
# [[torchcell.adapters.lim2025_tale_growth_rate_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/lim2025_tale_growth_rate_adapter.py
# Test file: tests/torchcell/adapters/test_lim2025_tale_growth_rate_adapter.py
"""BioCypher adapter exposing the Lim 2025 TALE lineages' initial growth rates as
knowledge-graph nodes and edges.

One record per Supplementary Table 3 lineage (KT2440, IPL300 and IPL400 at 4 g/L
isoprenol, KT2440 at 1 mM formaldehyde): the strain's designed deletions and its founder
clone's called variants served as ``perturbation`` nodes, the absolute initial growth
rate as an ``EnvironmentResponsePhenotype``, and the stressor as an environment
perturbation.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.pputida.lim2025 import TaleGrowthRateLim2025Dataset


class TaleGrowthRateLim2025Adapter(CellAdapter):
    """Cell adapter that serves TaleGrowthRateLim2025Dataset to BioCypher."""

    def __init__(
        self,
        dataset: TaleGrowthRateLim2025Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "tale_growth_rate_lim2025_adapter.yaml"
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
