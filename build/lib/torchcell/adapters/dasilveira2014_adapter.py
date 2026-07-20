# torchcell/adapters/dasilveira2014_adapter.py
# [[torchcell.adapters.dasilveira2014_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/dasilveira2014_adapter.py
# Test file: tests/torchcell/adapters/test_dasilveira2014_adapter.py
"""BioCypher adapter exposing the da Silveira 2014 lipidome (MS lipidomics) dataset.

Lipid species are modeled as ``MetabolitePhenotype`` (lipids are metabolites), with
``measurement_type="lipidomics_ms_relative_abundance_au"``.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.scerevisiae.dasilveira2014 import (
    MetaboliteDaSilveira2014Dataset,
)


class MetaboliteDaSilveira2014Adapter(CellAdapter):
    """Cell adapter serving the da Silveira 2014 lipidome (MS lipidomics) dataset to BioCypher."""

    def __init__(
        self,
        dataset: MetaboliteDaSilveira2014Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "metabolite_dasilveira2014_adapter.yaml"
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
