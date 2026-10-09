# torchcell/adapters/caglar2017_protein_fold_change_adapter.py
# [[torchcell.adapters.caglar2017_protein_fold_change_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/caglar2017_protein_fold_change_adapter.py
# Test file: tests/torchcell/adapters/test_caglar2017_protein_fold_change_adapter.py
"""BioCypher adapter exposing the Caglar 2017 REL606 protein fold changes as
knowledge-graph nodes and edges.

One record per Table S8 protein-level contrast under the paper's primary design. Every
strain is the wild type, so the genotype carries no perturbation and no
``bacterial perturbation`` node is served; the contrast itself is an environment edit,
so the environment-perturbation pair is enabled.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.ecoli.caglar2017 import ProteinFoldChangeCaglar2017Dataset


class ProteinFoldChangeCaglar2017Adapter(CellAdapter):
    """Cell adapter that serves ProteinFoldChangeCaglar2017Dataset to BioCypher."""

    def __init__(
        self,
        dataset: ProteinFoldChangeCaglar2017Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "protein_fold_change_caglar2017_adapter.yaml"
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
