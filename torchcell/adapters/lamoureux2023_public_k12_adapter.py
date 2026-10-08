# torchcell/adapters/lamoureux2023_public_k12_adapter.py
# [[torchcell.adapters.lamoureux2023_public_k12_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/lamoureux2023_public_k12_adapter.py
# Test file: tests/torchcell/adapters/test_lamoureux2023_public_k12_adapter.py
"""BioCypher adapter exposing Public K-12 (Lamoureux 2023) as knowledge-graph
nodes and edges.

One record per reprocessed public MG1655 RNA-seq library: the wild type or up to two
``BacterialDeletionPerturbation`` leaves, served as ``bacterial perturbation`` nodes, and
the medium's supplements and physical settings as environment perturbations.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.ecoli.lamoureux2023_public_k12 import (
    RnaseqPublicK12Lamoureux2023Dataset,
)


class RnaseqPublicK12Lamoureux2023Adapter(CellAdapter):
    """Cell adapter that serves RnaseqPublicK12Lamoureux2023Dataset to BioCypher."""

    def __init__(
        self,
        dataset: RnaseqPublicK12Lamoureux2023Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "rnaseq_public_k12_lamoureux2023_adapter.yaml"
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
