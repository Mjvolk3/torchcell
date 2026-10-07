# torchcell/adapters/desiqueira2025_proteome_adapter.py
# [[torchcell.adapters.desiqueira2025_proteome_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/desiqueira2025_proteome_adapter.py
# Test file: tests/torchcell/adapters/test_desiqueira2025_proteome_adapter.py
"""BioCypher adapter exposing the de Siqueira 2025 Top3 proteome as
knowledge-graph nodes and edges.

One record per writable strain (KT2440 and the pre-tolerized PT parent): the PT
strain's deletion is a ``BacterialDeletionPerturbation`` served as a ``bacterial
perturbation`` node, and the abundances are a ``ProteinAbundancePhenotype``.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.pputida.desiqueira2025 import ProteomeDeSiqueira2025Dataset


class ProteomeDeSiqueira2025Adapter(CellAdapter):
    """Cell adapter that serves ProteomeDeSiqueira2025Dataset to BioCypher."""

    def __init__(
        self,
        dataset: ProteomeDeSiqueira2025Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "proteome_desiqueira2025_adapter.yaml"
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
