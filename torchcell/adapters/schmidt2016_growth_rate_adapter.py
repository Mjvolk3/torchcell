# torchcell/adapters/schmidt2016_growth_rate_adapter.py
# [[torchcell.adapters.schmidt2016_growth_rate_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/schmidt2016_growth_rate_adapter.py
# Test file: tests/torchcell/adapters/test_schmidt2016_growth_rate_adapter.py
"""BioCypher adapter exposing Schmidt 2016's rim-deletion growth rates as
knowledge-graph nodes and edges.

One record per (KEIO deletion strain, medium): a single
``BacterialDeletionPerturbation`` served as a ``bacterial perturbation`` node, and the
strain's growth rate relative to the wild type of the SAME medium as a
``FitnessPhenotype``. The two media differ by their carbon source, which rides on
``environment.perturbations``, so the environment-perturbation pair is enabled.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.ecoli.schmidt2016_growth_rate import (
    GrowthRateSchmidt2016Dataset,
)


class GrowthRateSchmidt2016Adapter(CellAdapter):
    """Cell adapter that serves GrowthRateSchmidt2016Dataset to BioCypher."""

    def __init__(
        self,
        dataset: GrowthRateSchmidt2016Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "growth_rate_schmidt2016_adapter.yaml"
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
