# torchcell/adapters/schastnaya2021_adapter.py
# [[torchcell.adapters.schastnaya2021_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/schastnaya2021_adapter.py
# Test file: tests/torchcell/adapters/test_schastnaya2021_adapter.py
"""BioCypher adapter exposing the Schastnaya 2021 deletion metabolome as
knowledge-graph nodes and edges.

One record per (KEIO deletion strain, carbon source): a single
``BacterialDeletionPerturbation`` served as a ``bacterial perturbation`` node, and the
strain's FIA-TOF-MS ion log2 fold changes against the wild type as a
``MetabolitePhenotype``. The carbon source is part of the medium, so no record carries
an environment perturbation and that pair is not enabled.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.ecoli.schastnaya2021 import MetabolomeSchastnaya2021Dataset


class MetabolomeSchastnaya2021Adapter(CellAdapter):
    """Cell adapter that serves MetabolomeSchastnaya2021Dataset to BioCypher."""

    def __init__(
        self,
        dataset: MetabolomeSchastnaya2021Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "metabolome_schastnaya2021_adapter.yaml"
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
