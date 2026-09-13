# torchcell/adapters/hoepfner2014_adapter.py
# [[torchcell.adapters.hoepfner2014_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/hoepfner2014_adapter.py
# Test file: tests/torchcell/adapters/test_hoepfner2014_adapter.py
"""BioCypher adapter exposing the Hoepfner 2014 HIP-HOP chemogenomic atlas as
knowledge-graph nodes and edges.

Each record is one (deletion strain, compound condition) pair. The genotype is a
gene-keyed ``genotype`` + ``perturbation`` pair (a heterozygous engineered-CNV leaf for
HIP, a KanMX deletion leaf for HOP), the condition is an ``environment`` whose media,
temperature and dosed small molecule are first-class nodes, and the readout is an
``environment response phenotype`` carrying the screen id.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.scerevisiae.hoepfner2014 import EnvChemgenHoepfner2014Dataset


class EnvChemgenHoepfner2014Adapter(CellAdapter):
    """Cell adapter that serves the Hoepfner 2014 HIP-HOP atlas to BioCypher."""

    def __init__(
        self,
        dataset: EnvChemgenHoepfner2014Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "env_chemgen_hoepfner2014_adapter.yaml"
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
