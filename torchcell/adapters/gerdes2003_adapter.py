# torchcell/adapters/gerdes2003_adapter.py
# [[torchcell.adapters.gerdes2003_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/gerdes2003_adapter.py
# Test file: tests/torchcell/adapters/test_gerdes2003_adapter.py
"""BioCypher adapter exposing the Gerdes 2003 genetic-footprinting essentiality calls
of MG1655 genes as knowledge-graph nodes and edges.

One record per gene: a single ``TransposonInsertionPerturbation`` served as a
``perturbation`` node, and the call as a ``GeneEssentialityPhenotype``. The
reference's environment is the same fermentor medium without the kanamycin.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.ecoli.gerdes2003 import GeneEssentialityGerdes2003Dataset


class GeneEssentialityGerdes2003Adapter(CellAdapter):
    """Cell adapter that serves GeneEssentialityGerdes2003Dataset to BioCypher."""

    def __init__(
        self,
        dataset: GeneEssentialityGerdes2003Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "gene_essentiality_gerdes2003_adapter.yaml"
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
