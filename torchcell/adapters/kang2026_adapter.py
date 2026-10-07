# torchcell/adapters/kang2026_adapter.py
# [[torchcell.adapters.kang2026_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/kang2026_adapter.py
# Test file: tests/torchcell/adapters/test_kang2026_adapter.py
"""BioCypher adapter exposing the Kang 2026 isoprenyl acetate titers as
knowledge-graph nodes and edges.

One record per strain and titer column on the PIPA chassis: pathway and deletion leaves
served as ``bacterial perturbation`` nodes, and the titer as a ``ProductTiterPhenotype``.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.pputida.kang2026 import IsoprenylAcetateTiterKang2026Dataset


class IsoprenylAcetateTiterKang2026Adapter(CellAdapter):
    """Cell adapter that serves IsoprenylAcetateTiterKang2026Dataset to BioCypher."""

    def __init__(
        self,
        dataset: IsoprenylAcetateTiterKang2026Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "isoprenyl_acetate_titer_kang2026_adapter.yaml"
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
