# torchcell/adapters/vanacloig2022_adapter.py
# [[torchcell.adapters.vanacloig2022_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/vanacloig2022_adapter.py
# Test file: tests/torchcell/adapters/test_vanacloig2022_adapter.py
"""BioCypher adapter exposing the Vanacloig-Pedros 2022 anaerobic chemical-genomic screen
as knowledge-graph nodes and edges.

Each record is one (barcoded library deletion in the 3DeltaAlpha background, inhibitor)
pair. The genotype is gene-keyed, so it serves ``genotype``/``perturbation`` nodes; the
environment carries two perturbations that are first-class nodes rather than values buried
in a serialized environment: the inhibitor at its dose and SynBase's pH 5.0.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.scerevisiae.vanacloig2022 import EnvChemgenVanacloig2022Dataset


class EnvChemgenVanacloig2022Adapter(CellAdapter):
    """Cell adapter that serves the Vanacloig-Pedros 2022 chemical-genomic screen."""

    def __init__(
        self,
        dataset: EnvChemgenVanacloig2022Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "env_chemgen_vanacloig2022_adapter.yaml"
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
