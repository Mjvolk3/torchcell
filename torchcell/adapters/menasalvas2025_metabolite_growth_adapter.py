# torchcell/adapters/menasalvas2025_metabolite_growth_adapter.py
# [[torchcell.adapters.menasalvas2025_metabolite_growth_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/menasalvas2025_metabolite_growth_adapter.py
# Test file: tests/torchcell/adapters/test_menasalvas2025_metabolite_growth_adapter.py
"""BioCypher adapter exposing the Menasalvas 2025 deposited growth-phase metabolite concentrations as knowledge-graph nodes and edges.

One record per released strain of data S1-1's growth phase: the pathway, biosensor
and deletion leaves served as ``bacterial perturbation`` nodes, the TEAM-2595
record's 75 called variants of data S1-5 served as ``bacterial sequence variant
perturbation`` nodes, and the 48 Absolute micromolar concentrations as a
``MetabolitePhenotype``.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.pputida.menasalvas2025 import (
    MetaboliteGrowthPhaseMenasalvas2025Dataset,
)


class MetaboliteGrowthPhaseMenasalvas2025Adapter(CellAdapter):
    """Cell adapter that serves MetaboliteGrowthPhaseMenasalvas2025Dataset to BioCypher."""

    def __init__(
        self,
        dataset: MetaboliteGrowthPhaseMenasalvas2025Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "metabolite_growth_menasalvas2025_adapter.yaml"
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
