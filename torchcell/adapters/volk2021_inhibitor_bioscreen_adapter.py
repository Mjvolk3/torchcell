# torchcell/adapters/volk2021_inhibitor_bioscreen_adapter.py
# [[torchcell.adapters.volk2021_inhibitor_bioscreen_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/volk2021_inhibitor_bioscreen_adapter.py
# Test file: tests/torchcell/adapters/test_volk2021_inhibitor_bioscreen_adapter.py
"""BioCypher adapter for the PRIVATE 2021 Bioscreen C inhibitor runs on strain bAID.

Registered in ``PRIVATE_DATASET_ADAPTER_MAP`` only, so a build reaches it only when run
with ``--include-private``. Each record is one well: the bAID host with no edit (an
empty genotype, so no ``perturbation`` nodes), YPD carrying zero to six inhibitors as
first-class ``environment perturbation`` nodes, and a relative-growth-rate or no-growth
``environment response phenotype``. The bAID background, its integrated CRISPR-AID
cassette included, rides on the ``genome`` node's serialized ``StrainReferenceGenome``.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.private_torchcell.volk2021_inhibitor_bioscreen import (
    InhibitorBioscreenVolk2021Dataset,
)


class InhibitorBioscreenVolk2021Adapter(CellAdapter):
    """Cell adapter that serves the private 2021 Bioscreen C inhibitor runs."""

    def __init__(
        self,
        dataset: InhibitorBioscreenVolk2021Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "inhibitor_bioscreen_volk2021_adapter.yaml"
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
