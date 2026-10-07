# torchcell/adapters/fuhrer2017_adapter.py
# [[torchcell.adapters.fuhrer2017_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/fuhrer2017_adapter.py
# Test file: tests/torchcell/adapters/test_fuhrer2017_adapter.py
"""BioCypher adapter exposing the Fuhrer 2017 Keio deletion metabolome as
knowledge-graph nodes and edges.

One record per BW25113 Keio strain: a single ``BacterialDeletionPerturbation`` served as
a ``bacterial perturbation`` node, and its FIA-TOF-MS ion z-scores as a
``MetabolitePhenotype``. The environment is one fixed medium with no perturbation.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.ecoli.fuhrer2017 import MetabolomeFuhrer2017Dataset


class MetabolomeFuhrer2017Adapter(CellAdapter):
    """Cell adapter that serves MetabolomeFuhrer2017Dataset to BioCypher."""

    def __init__(
        self,
        dataset: MetabolomeFuhrer2017Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "metabolome_fuhrer2017_adapter.yaml"
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
