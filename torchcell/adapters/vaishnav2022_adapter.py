# torchcell/adapters/vaishnav2022_adapter.py
# [[torchcell.adapters.vaishnav2022_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/vaishnav2022_adapter.py
# Test file: tests/torchcell/adapters/test_vaishnav2022_adapter.py
"""BioCypher adapter exposing Vaishnav 2022 sort-seq library sequences as graph nodes.

Each record is one library sequence's measured expression: a genotype whose
variable-promoter reporter construct is a first-class ``perturbation`` node carrying the
released sequence, an SC-Ura ``environment`` with no environmental edit, and a ``pooled
sort seq expression phenotype``. No environment-perturbation methods are enabled: the
assay varies the genome's content, never the medium.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.scerevisiae.vaishnav2022 import (
    PooledPromoterExpressionVaishnav2022Dataset,
)


class PooledPromoterExpressionVaishnav2022Adapter(CellAdapter):
    """Cell adapter that serves the Vaishnav 2022 sort-seq records to BioCypher."""

    def __init__(
        self,
        dataset: PooledPromoterExpressionVaishnav2022Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "pooled_promoter_expression_vaishnav2022_adapter.yaml"
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
