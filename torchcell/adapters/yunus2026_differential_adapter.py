# torchcell/adapters/yunus2026_differential_adapter.py
# [[torchcell.adapters.yunus2026_differential_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/yunus2026_differential_adapter.py
# Test file: tests/torchcell/adapters/test_yunus2026_differential_adapter.py
"""BioCypher adapter exposing the Yunus 2026 PP_4188 differential proteome as
knowledge-graph nodes and edges.

One record: one ``BacterialCrisprInterferencePerturbation`` served as a ``bacterial
perturbation`` node, its ``CrisprConstruct`` as a ``crispr construct`` node, and the
strain's released per-protein fold change against the control as a
``ProteinAbundancePhenotype``.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.pputida.yunus2026 import (
    CrispriDifferentialProteomeYunus2026Dataset,
)


class CrispriDifferentialProteomeYunus2026Adapter(CellAdapter):
    """Cell adapter serving CrispriDifferentialProteomeYunus2026Dataset to BioCypher."""

    def __init__(
        self,
        dataset: CrispriDifferentialProteomeYunus2026Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "crispri_differential_proteome_yunus2026_adapter.yaml"
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
