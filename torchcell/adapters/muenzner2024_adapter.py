# torchcell/adapters/muenzner2024_adapter.py
# [[torchcell.adapters.muenzner2024_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/muenzner2024_adapter.py
# Test file: tests/torchcell/adapters/test_muenzner2024_adapter.py
"""BioCypher adapters exposing the Muenzner 2024 natural-isolate proteome and turnover."""

import os.path as osp

import yaml
from omegaconf import DictConfig, OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.scerevisiae.muenzner2024 import (
    ProteinTurnoverMuenzner2024Dataset,
    ProteomeMuenzner2024Dataset,
)


def _load_conf(name: str) -> DictConfig:
    """Read ``conf/<name>`` beside this module; a missing conf raises."""
    config_path = osp.join(osp.dirname(osp.abspath(__file__)), "conf", name)
    if not osp.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as file:
        config = OmegaConf.create(yaml.safe_load(file))
    if not isinstance(config, DictConfig):
        raise TypeError(f"{config_path} is not a mapping")
    return config


class ProteomeMuenzner2024Adapter(CellAdapter):
    """Serves the 794 natural-isolate proteomes (``ProteinAbundancePhenotype``).

    Compact genotype, as for Caudal 2024: each isolate carries thousands of natural
    perturbations, held on the genotype node and off-graph in the Peter tier, so the
    per-perturbation node and edge methods are not enabled.
    """

    def __init__(
        self,
        dataset: ProteomeMuenzner2024Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        config = _load_conf("proteome_muenzner2024_adapter.yaml")
        super().__init__(
            config, dataset, process_workers, io_workers, chunk_size, loader_batch_size
        )
        self.dataset = dataset
        self.process_workers = process_workers
        self.io_workers = io_workers
        self.chunk_size = chunk_size
        self.loader_batch_size = loader_batch_size


class ProteinTurnoverMuenzner2024Adapter(CellAdapter):
    """Serves the 55 natural-isolate turnover records (``ProteinTurnoverPhenotype``)."""

    def __init__(
        self,
        dataset: ProteinTurnoverMuenzner2024Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        config = _load_conf("protein_turnover_muenzner2024_adapter.yaml")
        super().__init__(
            config, dataset, process_workers, io_workers, chunk_size, loader_batch_size
        )
        self.dataset = dataset
        self.process_workers = process_workers
        self.io_workers = io_workers
        self.chunk_size = chunk_size
        self.loader_batch_size = loader_batch_size
