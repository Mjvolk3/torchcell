# torchcell/adapters/puddu2019_adapter.py
# [[torchcell.adapters.puddu2019_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/puddu2019_adapter.py
# Test file: tests/torchcell/adapters/test_puddu2019_adapter.py
"""BioCypher adapter exposing Puddu 2019's realized genome architecture as
knowledge-graph nodes and edges.

Each record is one sequenced colony of the homozygous diploid yeast knockout collection:
its designed kanMX deletion (with the barcodes the sequencing detected) is a
``perturbation`` on a ``genotype``, the 1.8 ml liquid YPAD culture is an ``environment``
with its media and temperature as first-class nodes, and the measured architecture is a
``genome architecture phenotype`` carrying the per-feature values as a JSON string beside
the assay that names their keys.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.scerevisiae.puddu2019 import GenomeArchitecturePuddu2019Dataset


class GenomeArchitecturePuddu2019Adapter(CellAdapter):
    """Cell adapter that serves the Puddu 2019 genome-architecture dataset."""

    def __init__(
        self,
        dataset: GenomeArchitecturePuddu2019Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "genome_architecture_puddu2019_adapter.yaml"
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
