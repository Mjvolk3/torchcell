# torchcell/adapters/lim2022_adapter.py
# [[torchcell.adapters.lim2022_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/lim2022_adapter.py
# Test file: tests/torchcell/adapters/test_lim2022_adapter.py
"""BioCypher adapter exposing putidaPRECISE321 (Lim 2022) as knowledge-graph
nodes and edges.

One record per KT2440 sample: the wild type or a single ``BacterialDeletionPerturbation``
served as a ``bacterial perturbation`` node, and the absolute transcriptome as an
``RNASeqExpressionPhenotype``. The compendium does not carry temperature, so every
record gaps it and the temperature methods emit nothing.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.pputida.lim2022 import PutidaPrecise321Lim2022Dataset


class PutidaPrecise321Lim2022Adapter(CellAdapter):
    """Cell adapter that serves PutidaPrecise321Lim2022Dataset to BioCypher."""

    def __init__(
        self,
        dataset: PutidaPrecise321Lim2022Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "putida_precise321_lim2022_adapter.yaml"
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
