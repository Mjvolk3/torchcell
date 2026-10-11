# torchcell/adapters/dechiara2022_adapter.py
# [[torchcell.adapters.dechiara2022_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/dechiara2022_adapter.py
# Test file: tests/torchcell/adapters/test_dechiara2022_adapter.py
"""BioCypher adapters exposing the three De Chiara 2022 stores as graph nodes and edges.

Each 1,011-panel isolate carries ~5,000 perturbations off S288C (sequence variants,
accessory presences, core absences), so like Caudal 2024 the per-perturbation node and
the perturbation-to-genotype edge are left out: the full perturbation set is on the
genotype node and its sequences live off-graph in Peter 2018's pinned tarball. The two
growth stores serve their carbon and nitrogen sources as environment-perturbation nodes;
the life-cycle store's protocols carry none.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.scerevisiae.dechiara2022 import (
    CellYieldDeChiara2022Dataset,
    DoublingTimeDeChiara2022Dataset,
    LifeCycleDeChiara2022Dataset,
)


class _DeChiara2022Adapter(CellAdapter):
    """A De Chiara 2022 store served through the conf its subclass names."""

    conf_name: str

    def __init__(
        self,
        dataset: (
            LifeCycleDeChiara2022Dataset
            | DoublingTimeDeChiara2022Dataset
            | CellYieldDeChiara2022Dataset
        ),
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(current_dir, "conf", self.conf_name)
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


class LifeCycleDeChiara2022Adapter(_DeChiara2022Adapter):
    """Serves LifeCycleDeChiara2022Dataset (sporulation and viability time courses)."""

    conf_name = "life_cycle_dechiara2022_adapter.yaml"


class DoublingTimeDeChiara2022Adapter(_DeChiara2022Adapter):
    """Serves DoublingTimeDeChiara2022Dataset (doubling time in 43 environments)."""

    conf_name = "doubling_time_dechiara2022_adapter.yaml"


class CellYieldDeChiara2022Adapter(_DeChiara2022Adapter):
    """Serves CellYieldDeChiara2022Dataset (cell yield in 43 environments)."""

    conf_name = "cell_yield_dechiara2022_adapter.yaml"
