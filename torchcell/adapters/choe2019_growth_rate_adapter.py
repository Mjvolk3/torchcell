# torchcell/adapters/choe2019_growth_rate_adapter.py
# [[torchcell.adapters.choe2019_growth_rate_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/choe2019_growth_rate_adapter.py
# Test file: tests/torchcell/adapters/test_choe2019_growth_rate_adapter.py
"""BioCypher adapter exposing Choe 2019's two designed MS56 deletions as
knowledge-graph nodes and edges.

One record per designed deletion strain: a ``BacterialDeletionPerturbation`` per deleted
gene, served as ``bacterial perturbation`` (21 for the large deletion, 1 for ``rpoS``),
and the strain's growth rate as a ratio to its isogenic parent MS56. The panel is one M9
glucose medium with no added compound, so the environment-perturbation pair is off; the
environment declares a ``ProvenanceGap`` on temperature, for which the temperature
methods emit nothing.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.ecoli.choe2019_growth_rate import GrowthRateChoe2019Dataset


class GrowthRateChoe2019Adapter(CellAdapter):
    """Cell adapter that serves GrowthRateChoe2019Dataset to BioCypher."""

    def __init__(
        self,
        dataset: GrowthRateChoe2019Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(current_dir, "conf", "growth_rate_choe2019_adapter.yaml")
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
