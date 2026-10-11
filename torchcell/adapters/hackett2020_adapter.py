# torchcell/adapters/hackett2020_adapter.py
# [[torchcell.adapters.hackett2020_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/hackett2020_adapter.py
# Test file: tests/torchcell/adapters/test_hackett2020_adapter.py
"""BioCypher adapter exposing Hackett 2020 IDEA induction time courses as graph nodes.

Each record is one array of a chemostat time course after one regulator was induced:
a genotype whose inducible promoter insertion is a first-class ``perturbation`` node,
a chemostat ``environment`` whose beta-estradiol is an ``environment perturbation``
node from the first post-induction array on, and an ``induction timecourse expression
phenotype``. The typed strain background travels on the ``genome`` node.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.scerevisiae.hackett2020 import (
    InductionTimecourseHackett2020Dataset,
)


class InductionTimecourseHackett2020Adapter(CellAdapter):
    """Cell adapter that serves the Hackett 2020 IDEA time courses to BioCypher."""

    def __init__(
        self,
        dataset: InductionTimecourseHackett2020Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "induction_timecourse_hackett2020_adapter.yaml"
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
