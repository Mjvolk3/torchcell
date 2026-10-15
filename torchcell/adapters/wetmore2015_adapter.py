# torchcell/adapters/wetmore2015_adapter.py
# [[torchcell.adapters.wetmore2015_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/wetmore2015_adapter.py
# Test file: tests/torchcell/adapters/test_wetmore2015_adapter.py
"""BioCypher adapter exposing Wetmore 2015's own RB-TnSeq release (*E. coli* BW25113,
KEIO_ML9) as knowledge-graph nodes and edges.

One record per (gene, successful sample): a single ``TransposonInsertionPerturbation``
served as a ``perturbation`` node, and the gene fitness as an
``EnvironmentResponsePhenotype`` carrying the released moderated t. The four sucrose and
D-mannitol samples the authors withdrew in 2021 are refused by the loader, so no node
here is one of theirs. A carbon or nitrogen source is an
``EnvironmentPhysicalPerturbation``, served as an environment perturbation.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.ecoli.wetmore2015 import RbTnseqWetmore2015Dataset


class RbTnseqWetmore2015Adapter(CellAdapter):
    """Cell adapter that serves RbTnseqWetmore2015Dataset to BioCypher."""

    def __init__(
        self,
        dataset: RbTnseqWetmore2015Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "rbtnseq_wetmore2015_ecoli_adapter.yaml"
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
