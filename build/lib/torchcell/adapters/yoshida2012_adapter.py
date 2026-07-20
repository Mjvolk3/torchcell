# torchcell/adapters/yoshida2012_adapter.py
# [[torchcell.adapters.yoshida2012_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/yoshida2012_adapter.py
# Test file: tests/torchcell/adapters/test_yoshida2012_adapter.py
"""BioCypher adapter exposing the Yoshida 2012 organic-acid titer dataset.

Organic-acid titers (succinate/malate/... in mM) are modeled as
``MetabolitePhenotype``, with ``measurement_type`` set by the loader.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.scerevisiae.yoshida2012 import OrganicAcidYoshida2012Dataset


class OrganicAcidYoshida2012Adapter(CellAdapter):
    """Cell adapter serving the Yoshida 2012 organic-acid titer dataset to BioCypher."""

    def __init__(
        self,
        dataset: OrganicAcidYoshida2012Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "organic_acid_yoshida2012_adapter.yaml"
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
