# torchcell/adapters/ishii2007_metabolome_adapter.py
# [[torchcell.adapters.ishii2007_metabolome_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/ishii2007_metabolome_adapter.py
# Test file: tests/torchcell/adapters/test_bacterial_adapters.py
"""BioCypher adapter exposing the Ishii 2007 CE-TOFMS metabolome as
knowledge-graph nodes and edges.

One record per BW25113 Keio disruptant grown in a glucose-limited chemostat at
0.2 h-1: a single ``BacterialDeletionPerturbation`` served as a
``bacterial perturbation`` node, and the culture's intracellular concentrations in
mM as a ``MetabolitePhenotype``. The environment is one medium with no
perturbation, and the reference is the wild-type control of the record's own series.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.ecoli.ishii2007 import MetabolomeIshii2007Dataset


class MetabolomeIshii2007Adapter(CellAdapter):
    """Cell adapter that serves MetabolomeIshii2007Dataset to BioCypher."""

    def __init__(
        self,
        dataset: MetabolomeIshii2007Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(current_dir, "conf", "metabolome_ishii2007_adapter.yaml")
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
