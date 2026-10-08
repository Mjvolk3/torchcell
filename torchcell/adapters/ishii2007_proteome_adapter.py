# torchcell/adapters/ishii2007_proteome_adapter.py
# [[torchcell.adapters.ishii2007_proteome_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/ishii2007_proteome_adapter.py
# Test file: tests/torchcell/adapters/test_bacterial_adapters.py
"""BioCypher adapter exposing the Ishii 2007 LC-MS/MS proteome as knowledge-graph
nodes and edges.

The same 24 cultures the metabolome and flux adapters serve, with each culture's
absolute abundances in mg-protein/g-dry-cell-weight as a
``ProteinAbundancePhenotype`` keyed by BW25113 locus tag. The standard error is
derived from the release's coefficient of variance over duplicate measurement.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.ecoli.ishii2007 import ProteomeIshii2007Dataset


class ProteomeIshii2007Adapter(CellAdapter):
    """Cell adapter that serves ProteomeIshii2007Dataset to BioCypher."""

    def __init__(
        self,
        dataset: ProteomeIshii2007Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(current_dir, "conf", "proteome_ishii2007_adapter.yaml")
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
