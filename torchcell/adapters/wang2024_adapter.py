# torchcell/adapters/wang2024_adapter.py
# [[torchcell.adapters.wang2024_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/wang2024_adapter.py
# Test file: tests/torchcell/adapters/test_wang2024_adapter.py
"""BioCypher adapter exposing the Wang 2024 rifampicin Tn-seq screen as
knowledge-graph nodes and edges.

One record per (gene, condition): a single gene-level Tn5
``TransposonInsertionPerturbation`` served as a ``bacterial perturbation`` node, and the
TRANSIT log2 fold change as an ``EnvironmentResponsePhenotype``. Every record's
environment carries exactly one ``SmallMoleculePerturbation``, rifampicin at its
absolute dose, so the environment-perturbation pair is enabled and needs no
per-condition branch. No phage and no CRISPRi construct appears in any record, so those
pairs stay off, and the one medium the six conditions share means the media node is a
single entity the whole dataset joins on.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.ecoli.wang2024 import EnvChemgenWang2024Dataset


class EnvChemgenWang2024Adapter(CellAdapter):
    """Cell adapter that serves EnvChemgenWang2024Dataset to BioCypher."""

    def __init__(
        self,
        dataset: EnvChemgenWang2024Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "ecoli_env_chemgen_wang2024_adapter.yaml"
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
