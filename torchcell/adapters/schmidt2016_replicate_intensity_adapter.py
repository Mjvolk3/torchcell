# torchcell/adapters/schmidt2016_replicate_intensity_adapter.py
# [[torchcell.adapters.schmidt2016_replicate_intensity_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/schmidt2016_replicate_intensity_adapter.py
# Test file: tests/torchcell/adapters/test_schmidt2016_replicate_intensity_adapter.py
"""BioCypher adapter exposing the Schmidt 2016 per-culture normalized LC-MS intensities
(Table S8) as knowledge-graph nodes and edges.

One record per BW25113 culture: 22 growth conditions x 3 cultures, each record that
culture's released SafeQuant-normalized intensity per protein, labeled by its Table S25
file name. Every record is the wild type with an empty genotype, so no ``perturbation``
node is served; the varying axis is the environment, and every record but the three LB
cultures carries an environment perturbation, so the environment-perturbation pair is
enabled.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.ecoli.schmidt2016_replicate_intensity import (
    ProteomeReplicateIntensitySchmidt2016Dataset,
)


class ProteomeReplicateIntensitySchmidt2016Adapter(CellAdapter):
    """Cell adapter that serves ProteomeReplicateIntensitySchmidt2016Dataset."""

    def __init__(
        self,
        dataset: ProteomeReplicateIntensitySchmidt2016Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "proteome_replicate_intensity_schmidt2016_adapter.yaml"
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
