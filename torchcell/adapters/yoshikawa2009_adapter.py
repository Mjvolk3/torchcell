# torchcell/adapters/yoshikawa2009_adapter.py
# [[torchcell.adapters.yoshikawa2009_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/yoshikawa2009_adapter.py
# Test file: tests/torchcell/adapters/test_yoshikawa2009_adapter.py
"""BioCypher adapter exposing Yoshikawa 2009's deletion-collection growth rates under
ethanol and NaCl stress as knowledge-graph nodes and edges.

Each record is one culture of one (Open Biosystems MATalpha deletion, condition) cell of
Table S1. The genotype is gene-keyed, so it serves ``genotype``/``perturbation`` nodes;
ethanol at 8% (v/v) and sodium chloride at 1 M are ``environment perturbation`` nodes,
which is what lets the two stresses join the other datasets that add the same compounds.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.scerevisiae.yoshikawa2009 import EnvChemgenYoshikawa2009Dataset


class EnvChemgenYoshikawa2009Adapter(CellAdapter):
    """Cell adapter that serves the Yoshikawa 2009 stress growth rates."""

    def __init__(
        self,
        dataset: EnvChemgenYoshikawa2009Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "env_chemgen_yoshikawa2009_adapter.yaml"
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
