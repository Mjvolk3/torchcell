# torchcell/adapters/tong2020_adapter.py
# [[torchcell.adapters.tong2020_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/tong2020_adapter.py
# Test file: tests/torchcell/adapters/test_tong2020_adapter.py
"""BioCypher adapter exposing the Tong 2020 carbon-source growth screen as
knowledge-graph nodes and edges.

One record per (deletion strain, carbon source): a single ``BacterialDeletionPerturbation``
served as a ``bacterial perturbation`` node, the growth as a ``FitnessPhenotype``, and
the varied carbon source as an environment perturbation.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.ecoli.tong2020 import CarbonSourceTong2020Dataset


class CarbonSourceTong2020Adapter(CellAdapter):
    """Cell adapter that serves CarbonSourceTong2020Dataset to BioCypher."""

    def __init__(
        self,
        dataset: CarbonSourceTong2020Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "ecoli_carbon_source_tong2020_adapter.yaml"
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
