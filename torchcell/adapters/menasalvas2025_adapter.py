# torchcell/adapters/menasalvas2025_adapter.py
# [[torchcell.adapters.menasalvas2025_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/menasalvas2025_adapter.py
# Test file: tests/torchcell/adapters/test_menasalvas2025_adapter.py
"""BioCypher adapter exposing the Menasalvas 2025 biosensor-coupled CRISPRi
selection as knowledge-graph nodes and edges.

One record per enriched knockdown target: the pathway, chassis deletion and CRISPRi
leaves served as ``bacterial perturbation`` nodes, each CRISPRi leaf's
``CrisprConstruct`` as a ``crispr construct`` node, the enrichment as an
``EnvironmentResponsePhenotype``, and the selection compounds as environment
perturbations.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.pputida.menasalvas2025 import (
    IsoprenolSelectionMenasalvas2025Dataset,
)


class IsoprenolSelectionMenasalvas2025Adapter(CellAdapter):
    """Cell adapter that serves IsoprenolSelectionMenasalvas2025Dataset to BioCypher."""

    def __init__(
        self,
        dataset: IsoprenolSelectionMenasalvas2025Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "isoprenol_selection_menasalvas2025_adapter.yaml"
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
