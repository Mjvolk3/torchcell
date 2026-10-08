# torchcell/adapters/fang2025_adapter.py
# [[torchcell.adapters.fang2025_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/fang2025_adapter.py
# Test file: tests/torchcell/adapters/test_fang2025_adapter.py
"""BioCypher adapter exposing the Fang 2025 CRISPRi-FACS free-fatty-acid enrichment as
knowledge-graph nodes and edges.

One record per kept (sgRNA, round) row: every knockdown leaf served as a
``bacterial perturbation`` node with its ``CrisprConstruct`` as a ``crispr construct``
node, the released sort enrichment as an ``EnvironmentResponsePhenotype``, and the one
screening culture's modified-M9 medium, 30 C temperature and 1 mM IPTG as the
environment. Round-2 records carry two knockdown leaves, the library guide's and the
pcnBi host's constant pcnB repression.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.ecoli.fang2025 import CrispriGuideFfaEnrichmentFang2025Dataset


class CrispriGuideFfaEnrichmentFang2025Adapter(CellAdapter):
    """Cell adapter that serves CrispriGuideFfaEnrichmentFang2025Dataset to BioCypher."""

    def __init__(
        self,
        dataset: CrispriGuideFfaEnrichmentFang2025Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "crispri_guide_ffa_enrichment_fang2025_adapter.yaml"
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
