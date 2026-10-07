# torchcell/adapters/goodall2018_adapter.py
# [[torchcell.adapters.goodall2018_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/goodall2018_adapter.py
# Test file: tests/torchcell/adapters/test_goodall2018_adapter.py
"""BioCypher adapter exposing the Goodall 2018 BW25113 TraDIS essentiality calls
as knowledge-graph nodes and edges.

One record per gene and condition: a single ``TransposonInsertionPerturbation`` served as
a ``bacterial perturbation`` node, and the call as a ``GeneEssentialityPhenotype``. The
plated condition records a typed temperature gap, so its temperature methods emit
nothing for those records.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.ecoli.goodall2018 import GeneEssentialityGoodall2018Dataset


class GeneEssentialityGoodall2018Adapter(CellAdapter):
    """Cell adapter that serves GeneEssentialityGoodall2018Dataset to BioCypher."""

    def __init__(
        self,
        dataset: GeneEssentialityGoodall2018Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "gene_essentiality_goodall2018_adapter.yaml"
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
