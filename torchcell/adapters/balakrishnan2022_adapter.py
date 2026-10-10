# torchcell/adapters/balakrishnan2022_adapter.py
# [[torchcell.adapters.balakrishnan2022_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/balakrishnan2022_adapter.py
# Test file: tests/torchcell/adapters/test_balakrishnan2022_adapter.py
"""BioCypher adapter exposing Balakrishnan 2022's mRNA number fractions as
knowledge-graph nodes and edges.

One record per released sample column of Table S3: the column's per-gene fractions as an
``mrna number fraction phenotype`` node, the NCM3722 derivatives' promoter replacements
(Pu-ptsG, Plac-gltB) as ``bacterial perturbation`` nodes, and the environment's 0.2%
glucose carbon source plus the 3MBA, IPTG or chloramphenicol dose as environment
perturbations.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.ecoli.balakrishnan2022 import (
    MrnaFractionBalakrishnan2022Dataset,
)


class MrnaFractionBalakrishnan2022Adapter(CellAdapter):
    """Cell adapter that serves MrnaFractionBalakrishnan2022Dataset to BioCypher."""

    def __init__(
        self,
        dataset: MrnaFractionBalakrishnan2022Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "mrna_fraction_balakrishnan2022_adapter.yaml"
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
