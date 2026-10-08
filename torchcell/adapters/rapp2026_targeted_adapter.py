# torchcell/adapters/rapp2026_targeted_adapter.py
# [[torchcell.adapters.rapp2026_targeted_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/rapp2026_targeted_adapter.py
# Test file: tests/torchcell/adapters/test_rapp2026_targeted_adapter.py
"""BioCypher adapter exposing Rapp 2026's targeted LC-MS/MS screen as knowledge-graph
nodes and edges.

One record per knockdown strain the targeted screen covered: a single
``BacterialCrisprInterferencePerturbation`` served as a ``bacterial perturbation`` node
with its ``CrisprConstruct``, and that strain's EIC peak-height fold changes as a
``MetabolitePhenotype`` on its own ``measurement_type``, which is what keeps this
platform's values from being compared with the FI-MS ones. The environment is one medium
plus the anhydrotetracycline that induces the knockdown, so the
environment-perturbation pair is served as well.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.ecoli.rapp2026_platforms import (
    TargetedMetabolomeRapp2026Dataset,
)


class TargetedMetabolomeRapp2026Adapter(CellAdapter):
    """Cell adapter that serves TargetedMetabolomeRapp2026Dataset to BioCypher."""

    def __init__(
        self,
        dataset: TargetedMetabolomeRapp2026Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "targeted_metabolome_rapp2026_adapter.yaml"
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
