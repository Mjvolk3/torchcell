# torchcell/adapters/wildenhain2015_adapter.py
# [[torchcell.adapters.wildenhain2015_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/wildenhain2015_adapter.py
# Test file: tests/torchcell/adapters/test_wildenhain2015_adapter.py
"""BioCypher adapter exposing the Wildenhain 2015 chemical-genetic matrix as
knowledge-graph nodes and edges.

Each record is one (Euroscarf haploid deletion, compound) cell of the CGM. The genotype is
gene-keyed, so it serves ``genotype``/``perturbation`` nodes; the screened compound at
20 uM is an ``environment perturbation`` node, which is what lets a CID join across the
chemogenomic datasets instead of hiding inside a serialized environment.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.scerevisiae.wildenhain2015 import (
    EnvChemgenWildenhain2015Dataset,
)


class EnvChemgenWildenhain2015Adapter(CellAdapter):
    """Cell adapter that serves the Wildenhain 2015 chemical-genetic matrix."""

    def __init__(
        self,
        dataset: EnvChemgenWildenhain2015Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "env_chemgen_wildenhain2015_adapter.yaml"
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
