# torchcell/adapters/jackson2020_adapter.py
# [[torchcell.adapters.jackson2020_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/jackson2020_adapter.py
# Test file: tests/torchcell/adapters/test_jackson2020_adapter.py
"""BioCypher adapter exposing Jackson 2020 pseudobulk single-cell UMI counts as
knowledge-graph nodes and edges.

Each record is one (barcoded deletion strain, culture) pair: the genotype is one
barcoded homozygous deletion with a first-class perturbation node, the culture an
``environment`` with its medium and its rapamycin perturbation where there is one (the
flask temperature is a typed gap, so the enabled temperature methods emit nothing), and
the readout a ``pseudobulk umi count phenotype``. The FY4/FY5 background rides on the ``genome`` node.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.scerevisiae.jackson2020 import (
    TfDeletionPseudobulkJackson2020Dataset,
)


class TfDeletionPseudobulkJackson2020Adapter(CellAdapter):
    """Cell adapter that serves the Jackson 2020 pseudobulk dataset to BioCypher."""

    def __init__(
        self,
        dataset: TfDeletionPseudobulkJackson2020Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "tf_deletion_pseudobulk_jackson2020_adapter.yaml"
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
