# torchcell/adapters/mota2024_adapter.py
# [[torchcell.adapters.mota2024_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/mota2024_adapter.py
# Test file: tests/torchcell/adapters/test_mota2024_adapter.py
"""BioCypher adapter exposing the Mota 2024 weak-acid susceptibility screen.

Each record is one (KanMX deletion strain, acid) pair. The genotype is gene-keyed, so it
serves as a ``genotype`` node plus a ``perturbation`` node and the edge between them. The
condition is an ``environment`` on the shared solid-YPD medium at 30 C, carrying TWO
``environment perturbation`` nodes: the weak acid at its molar dose, and the typed pH 4.5
edit whose agent is HCl. Both ride on the reference environment too, because the parental
control sat in empty wells of the same plates. The readout is an ``environment response
phenotype`` holding the ordinal grade and its typed call.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.scerevisiae.mota2024 import EnvChemgenMota2024Dataset


class EnvChemgenMota2024Adapter(CellAdapter):
    """Cell adapter serving the Mota 2024 weak-acid chemogenomic dataset."""

    def __init__(
        self,
        dataset: EnvChemgenMota2024Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(current_dir, "conf", "mota2024_adapter.yaml")
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
