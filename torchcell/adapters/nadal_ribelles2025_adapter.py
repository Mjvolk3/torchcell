# torchcell/adapters/nadal_ribelles2025_adapter.py
# [[torchcell.adapters.nadal_ribelles2025_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/nadal_ribelles2025_adapter.py
# Test file: tests/torchcell/adapters/test_nadal_ribelles2025_adapter.py
"""BioCypher adapter exposing Nadal-Ribelles 2025 pseudobulk Perturb-seq expression as
knowledge-graph nodes and edges.

Each record is one (single-gene URA3-marker deletion, condition) pair; the condition is
either base YPD or YPD + 0.4 M NaCl for 15 min, and the salt is emitted as an
``environment perturbation`` node attached to its environment.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.scerevisiae.nadal_ribelles2025 import (
    NadalRibellesPerturbSeq2025Dataset,
)


class NadalRibellesPerturbSeq2025Adapter(CellAdapter):
    """Cell adapter that serves the Nadal-Ribelles 2025 pseudobulk Perturb-seq dataset to BioCypher."""

    def __init__(
        self,
        dataset: NadalRibellesPerturbSeq2025Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "nadal_ribelles_perturbseq2025_adapter.yaml"
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
