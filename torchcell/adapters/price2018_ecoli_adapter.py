# torchcell/adapters/price2018_ecoli_adapter.py
# [[torchcell.adapters.price2018_ecoli_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/price2018_ecoli_adapter.py
# Test file: tests/torchcell/adapters/test_price2018_ecoli_adapter.py
"""BioCypher adapter exposing the Price 2018 RB-TnSeq compendium (*E. coli*
BW25113, KEIO_ML9) as knowledge-graph nodes and edges.

One record per (gene, sample): a single ``TransposonInsertionPerturbation`` served as a
``bacterial perturbation`` node, and the gene fitness as an
``EnvironmentResponsePhenotype``. A carbon or nitrogen source is an
``EnvironmentPhysicalPerturbation`` and a stress compound a ``SmallMoleculePerturbation``,
both served as environment perturbations.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.ecoli.price2018 import RbTnseqPrice2018EcoliDataset


class RbTnseqPrice2018EcoliAdapter(CellAdapter):
    """Cell adapter that serves RbTnseqPrice2018EcoliDataset to BioCypher."""

    def __init__(
        self,
        dataset: RbTnseqPrice2018EcoliDataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "rbtnseq_price2018_ecoli_adapter.yaml"
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
