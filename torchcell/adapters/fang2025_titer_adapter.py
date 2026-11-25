# torchcell/adapters/fang2025_titer_adapter.py
# [[torchcell.adapters.fang2025_titer_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/fang2025_titer_adapter.py
# Test file: tests/torchcell/adapters/test_fang2025_titer_adapter.py
"""BioCypher adapter exposing the Fang 2025 per-strain FFA titers as knowledge-graph
nodes and edges.

One record per released biological replicate culture: the knockdown, gene-addition and
deletion leaves of the strain served as ``perturbation`` nodes (each knockdown with its
``CrisprConstruct``), the released mg/L as a ``ProductTiterPhenotype`` carrying the
culture's own ``replicate_id``, and the tube or flask fermentation's modified-M9
medium, 30 C temperature, 1 mM IPTG and, for the P_BAD fadR strain, 10 mM arabinose as
the environment.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.ecoli.fang2025 import FfaTiterFang2025Dataset


class FfaTiterFang2025Adapter(CellAdapter):
    """Cell adapter that serves FfaTiterFang2025Dataset to BioCypher."""

    def __init__(
        self,
        dataset: FfaTiterFang2025Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(current_dir, "conf", "ffa_titer_fang2025_adapter.yaml")
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
