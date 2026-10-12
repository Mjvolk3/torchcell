# torchcell/adapters/royet2025_graded_adapter.py
# [[torchcell.adapters.royet2025_graded_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/royet2025_graded_adapter.py
# Test file: tests/torchcell/adapters/test_royet2025_graded_adapter.py
"""BioCypher adapter exposing Royet 2025 Table S4, the TRANSIT HMM state of every
KT2440 gene on LB agar and after outgrowth in LB, as knowledge-graph nodes and edges
(#909).

One record per (gene, stage): a single gene-level ``TransposonInsertionPerturbation``
served as a ``bacterial perturbation`` node, and the released state as a
``GradedGeneEssentialityPhenotype``. Neither stage carries an environment perturbation,
so that pair is off; the two media and the one temperature are served.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.pputida.royet2025 import GradedGeneEssentialityRoyet2025Dataset


class GradedGeneEssentialityRoyet2025Adapter(CellAdapter):
    """Cell adapter that serves GradedGeneEssentialityRoyet2025Dataset to BioCypher."""

    def __init__(
        self,
        dataset: GradedGeneEssentialityRoyet2025Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "graded_gene_essentiality_royet2025_adapter.yaml"
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
