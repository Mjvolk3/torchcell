# torchcell/adapters/lamoureux2023_growth_adapter.py
# [[torchcell.adapters.lamoureux2023_growth_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/lamoureux2023_growth_adapter.py
# Test file: tests/torchcell/adapters/test_lamoureux2023_growth_adapter.py
"""BioCypher adapter exposing Lamoureux 2023's released growth rates as graph nodes.

One record per released PRECISE-1K sample whose ``Growth Rate (1/hr)`` cell is positive,
89 in all. 34 genotypes carry one ``BacterialDeletionPerturbation``, served as
``bacterial perturbation``, and the other 55 are wild type; the carbon source, the
supplements and the salts are environment perturbations and are first-class nodes. The
phenotype is ``environment response phenotype``, carrying the absolute rate in 1/hr with
the released project as its screen id and the released ``rep_id`` as its replicate id
(#826).
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.ecoli.lamoureux2023_growth import GrowthRateLamoureux2023Dataset


class GrowthRateLamoureux2023Adapter(CellAdapter):
    """Cell adapter that serves GrowthRateLamoureux2023Dataset to BioCypher."""

    def __init__(
        self,
        dataset: GrowthRateLamoureux2023Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "growth_rate_lamoureux2023_adapter.yaml"
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
