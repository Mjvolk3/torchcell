# torchcell/adapters/borchert2024_adapter.py
# [[torchcell.adapters.borchert2024_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/borchert2024_adapter.py
# Test file: tests/torchcell/adapters/test_borchert2024_adapter.py
"""BioCypher adapter exposing the Borchert 2024 KT2440 RB-TnSeq compendium as
knowledge-graph nodes and edges.

One record per (gene, kept sample): a single ``TransposonInsertionPerturbation`` served
as a ``bacterial perturbation`` node, and the gene fitness as an
``EnvironmentResponsePhenotype``. The sample's carbon or nitrogen source is an
``EnvironmentPhysicalPerturbation`` and a second added species (a stressor, the DMSO
vehicle, the amino-acid mix or a nutrient dropout) is a further environment
perturbation.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.pputida.borchert2024 import RbTnseqBorchert2024Dataset


class RbTnseqBorchert2024Adapter(CellAdapter):
    """Cell adapter that serves RbTnseqBorchert2024Dataset to BioCypher."""

    def __init__(
        self,
        dataset: RbTnseqBorchert2024Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(current_dir, "conf", "rbtnseq_borchert2024_adapter.yaml")
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
