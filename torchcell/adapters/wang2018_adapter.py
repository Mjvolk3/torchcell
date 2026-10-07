# torchcell/adapters/wang2018_adapter.py
# [[torchcell.adapters.wang2018_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/wang2018_adapter.py
# Test file: tests/torchcell/adapters/test_wang2018_adapter.py
"""BioCypher adapter exposing the Wang 2018 pooled CRISPRi guide fitness as
knowledge-graph nodes and edges.

One record per (sgRNA, screen) row: every knockdown leaf served as a
``bacterial perturbation`` node with its ``CrisprConstruct`` as a ``crispr construct``
node, the released ``sgRNA fitness`` as an ``EnvironmentResponsePhenotype``, and the
screen's medium, 37 C temperature and added compounds as the environment.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.ecoli.wang2018 import CrispriGuideFitnessWang2018Dataset


class CrispriGuideFitnessWang2018Adapter(CellAdapter):
    """Cell adapter that serves CrispriGuideFitnessWang2018Dataset to BioCypher."""

    def __init__(
        self,
        dataset: CrispriGuideFitnessWang2018Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "crispri_guide_fitness_wang2018_adapter.yaml"
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
