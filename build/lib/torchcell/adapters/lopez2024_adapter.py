# torchcell/adapters/lopez2024_adapter.py
# [[torchcell.adapters.lopez2024_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/lopez2024_adapter.py
# Test file: tests/torchcell/adapters/test_lopez2024_adapter.py
"""BioCypher adapters exposing the Montano Lopez 2024 isobutanol biosensor datasets.

The genome-wide YKO screen and the validated strains report an isobutanol/BCAA
precursor readout, modeled as ``MetabolitePhenotype``. NOTE: the screen data is
unpublished (dissertation supplement) and its ``Publication`` uses a
methodology-anchor identifier -- whether these enter a shared graph is a per-build
decision, made at dataset selection, not by the adapter's existence.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.scerevisiae.lopez2024 import (
    IsobutanolScreenLopez2024Dataset,
    IsobutanolValidatedLopez2024Dataset,
)


class IsobutanolScreenLopez2024Adapter(CellAdapter):
    """Cell adapter serving the Lopez 2024 genome-wide isobutanol biosensor SCREEN to BioCypher."""

    def __init__(
        self,
        dataset: IsobutanolScreenLopez2024Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "isobutanol_screen_lopez2024_adapter.yaml"
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


class IsobutanolValidatedLopez2024Adapter(CellAdapter):
    """Cell adapter serving the Lopez 2024 validated isobutanol strains to BioCypher."""

    def __init__(
        self,
        dataset: IsobutanolValidatedLopez2024Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "isobutanol_validated_lopez2024_adapter.yaml"
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
