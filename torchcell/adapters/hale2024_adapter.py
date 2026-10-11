# torchcell/adapters/hale2024_adapter.py
# [[torchcell.adapters.hale2024_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/hale2024_adapter.py
# Test file: tests/torchcell/adapters/test_hale2024_adapter.py
"""BioCypher adapter for Hale 2024: CRISPRi guides on BYx3S segregant backgrounds.

Each record is one double-barcode lineage in one fitness assay. The genotype is a
``perturbed segregant genotype`` node (background cross and segregant plus the guide's
target gene); its background mosaic is ALSO a ``segregant genotype`` node joined by
``segregant background of``; the guide is a ``perturbation`` with its ``crispr construct``;
the assay is an ``environment`` (SC-Ura, 21 C, with or without anhydrotetracycline); the
readout is an ``environment response phenotype``.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.scerevisiae.hale2024 import CrispriSegregantHale2024Dataset


class Hale2024Adapter(CellAdapter):
    """Cell adapter that serves the Hale 2024 CRISPRi-on-segregant dataset to BioCypher."""

    def __init__(
        self,
        dataset: CrispriSegregantHale2024Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "crispri_segregant_hale2024_adapter.yaml"
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
