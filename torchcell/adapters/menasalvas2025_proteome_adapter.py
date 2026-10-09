# torchcell/adapters/menasalvas2025_proteome_adapter.py
# [[torchcell.adapters.menasalvas2025_proteome_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/menasalvas2025_proteome_adapter.py
# Test file: tests/torchcell/adapters/test_menasalvas2025_proteome_adapter.py
"""BioCypher adapter exposing the Menasalvas 2025 deposited growth-phase and production-phase proteome as knowledge-graph nodes and edges.

One record per released sample of Supplementary Data 2 sheet 4: the pathway,
biosensor and deletion leaves served as ``bacterial perturbation`` nodes, the
TEAM-2595 records' 75 called variants of data S1-5 served as ``bacterial sequence
variant perturbation`` nodes, and the per-sample mean Counts_sum profile as a
``ProteinAbundancePhenotype``.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.pputida.menasalvas2025 import ProteomeMenasalvas2025Dataset


class ProteomeMenasalvas2025Adapter(CellAdapter):
    """Cell adapter that serves ProteomeMenasalvas2025Dataset to BioCypher."""

    def __init__(
        self,
        dataset: ProteomeMenasalvas2025Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "proteome_menasalvas2025_adapter.yaml"
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
