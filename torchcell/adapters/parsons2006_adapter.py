# torchcell/adapters/parsons2006_adapter.py
# [[torchcell.adapters.parsons2006_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/parsons2006_adapter.py
# Test file: tests/torchcell/adapters/test_parsons2006_adapter.py
"""BioCypher adapter exposing the Parsons 2006 chemical-genetic compendium as
knowledge-graph nodes and edges.

Each record is one (Research Genetics MATa haploid deletion, condition) cell of Table S7.
The genotype is gene-keyed, so it serves ``genotype``/``perturbation`` nodes; the compound
at its Table S1 screening dose is an ``environment perturbation`` node, which is what lets
its CAS-resolved PubChem identity join across the chemogenomic datasets.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.scerevisiae.parsons2006 import EnvChemgenParsons2006Dataset


class EnvChemgenParsons2006Adapter(CellAdapter):
    """Cell adapter that serves the Parsons 2006 chemical-genetic compendium."""

    def __init__(
        self,
        dataset: EnvChemgenParsons2006Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "env_chemgen_parsons2006_adapter.yaml"
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
