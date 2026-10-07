# torchcell/adapters/gupta2024_adapter.py
# [[torchcell.adapters.gupta2024_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/gupta2024_adapter.py
# Test file: tests/torchcell/adapters/test_gupta2024_adapter.py
"""BioCypher adapter exposing the Gupta 2024 protein-turnover panel as knowledge-graph
nodes and edges.

One record per released growth condition of *E. coli* NCM3722. Eight of the thirteen are
the unperturbed strain and five carry protease or SsrA-pathway deletions, so
``bacterial perturbation`` is served; every chemostat condition carries a controlled-pH
``EnvironmentPhysicalPerturbation``, so the environment-perturbation pair is served too.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.ecoli.gupta2024 import ProteinTurnoverGupta2024Dataset


class ProteinTurnoverGupta2024Adapter(CellAdapter):
    """Cell adapter that serves ProteinTurnoverGupta2024Dataset to BioCypher."""

    def __init__(
        self,
        dataset: ProteinTurnoverGupta2024Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "protein_turnover_gupta2024_adapter.yaml"
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
