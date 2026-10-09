# torchcell/adapters/campos2018_morphology_adapter.py
# [[torchcell.adapters.campos2018_morphology_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/campos2018_morphology_adapter.py
# Test file: tests/torchcell/adapters/test_campos2018_morphology_adapter.py
"""BioCypher adapter exposing the Campos 2018 Keio cell-morphology screen.

One record per deletion strain: a single ``BacterialDeletionPerturbation`` served as a
``bacterial perturbation`` node, and the strain's 26-feature profile as a
``bacterial morphology phenotype``. The screen is one medium at one temperature, so no
environment perturbation is served.

A separate module from ``campos2018_adapter`` because it serves a separate dataset of
the same release: the growth-rate adapter emits ``fitness phenotype`` nodes and this one
emits ``bacterial morphology phenotype`` nodes, and an adapter's enable-list is what
decides which node methods run.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.ecoli.campos2018 import MorphologyCampos2018Dataset


class MorphologyCampos2018Adapter(CellAdapter):
    """Cell adapter that serves MorphologyCampos2018Dataset to BioCypher."""

    def __init__(
        self,
        dataset: MorphologyCampos2018Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "ecoli_morphology_campos2018_adapter.yaml"
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
