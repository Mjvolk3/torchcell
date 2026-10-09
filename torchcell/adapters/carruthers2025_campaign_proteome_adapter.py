# torchcell/adapters/carruthers2025_campaign_proteome_adapter.py
# [[torchcell.adapters.carruthers2025_campaign_proteome_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/carruthers2025_campaign_proteome_adapter.py
# Test file: tests/torchcell/adapters/test_carruthers2025_campaign_proteome_adapter.py
"""BioCypher adapter exposing the Carruthers 2025 campaign proteome as
knowledge-graph nodes and edges.

One record per CRISPRi ``(construct, DBTL cycle)`` strain of the campaign, carrying the
same genotype identity as the titer family's 465 records: pathway and CRISPRi leaves
served as ``bacterial perturbation`` nodes, each CRISPRi leaf's ``CrisprConstruct`` as a
``crispr construct`` node, and the percent-of-proteome abundances as a
``ProteinAbundancePhenotype``.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.pputida.carruthers2025 import (
    CampaignProteomeCarruthers2025Dataset,
)


class CampaignProteomeCarruthers2025Adapter(CellAdapter):
    """Cell adapter that serves CampaignProteomeCarruthers2025Dataset to BioCypher."""

    def __init__(
        self,
        dataset: CampaignProteomeCarruthers2025Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "campaign_proteome_carruthers2025_adapter.yaml"
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
