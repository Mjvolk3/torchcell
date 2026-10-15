# torchcell/adapters/teteneva2024_adapter.py
# [[torchcell.adapters.teteneva2024_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/teteneva2024_adapter.py
# Test file: tests/torchcell/adapters/test_teteneva2024_adapter.py
"""BioCypher adapter exposing Teteneva 2024's lake-water RB-TnSeq (*E. coli* K-12 W3110
RpoS+, Supplementary Table S4) as knowledge-graph nodes and edges.

One record per (resolved gene, lake water sample, filtration, sampling day): a single
``TransposonInsertionPerturbation`` on a W3110 RefSeq ``Y75_RS`` locus, served as a
``perturbation`` node, and the gene fitness as an
``EnvironmentResponsePhenotype``. Each condition is its own lake water ``Media`` with no
environment perturbation, so the environment-perturbation pair is off.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.ecoli.teteneva2024 import RbTnseqTeteneva2024Dataset


class RbTnseqTeteneva2024Adapter(CellAdapter):
    """Cell adapter that serves RbTnseqTeteneva2024Dataset to BioCypher."""

    def __init__(
        self,
        dataset: RbTnseqTeteneva2024Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "rbtnseq_teteneva2024_ecoli_adapter.yaml"
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
