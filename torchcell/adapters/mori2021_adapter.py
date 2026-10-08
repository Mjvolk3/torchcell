# torchcell/adapters/mori2021_adapter.py
# [[torchcell.adapters.mori2021_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/mori2021_adapter.py
# Test file: tests/torchcell/adapters/test_mori2021_adapter.py
"""BioCypher adapter exposing the Mori 2021 absolute E. coli proteome as
knowledge-graph nodes and edges.

One record per loaded *E. coli* MG1655 (EQ353) calibration sample. Every record is the
wild type with an empty genotype, so no ``bacterial perturbation`` node is served: the
paper's engineered NCM3722 derivatives (titratable ``ptsG`` and GOGAT promoters) are all
in samples this loader drops on their medium. The one environment the seven records share
carries an ``EnvironmentPhysicalPerturbation`` for the released 0.2% glucose, so the
environment-perturbation pair is enabled.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.ecoli.mori2021 import ProteomeMori2021Dataset


class ProteomeMori2021Adapter(CellAdapter):
    """Cell adapter that serves ProteomeMori2021Dataset to BioCypher."""

    def __init__(
        self,
        dataset: ProteomeMori2021Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(current_dir, "conf", "proteome_mori2021_adapter.yaml")
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
