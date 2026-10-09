# torchcell/adapters/wang2015_growth_adapter.py
# [[torchcell.adapters.wang2015_growth_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/wang2015_growth_adapter.py
# Test file: tests/torchcell/adapters/test_wang2015_growth_adapter.py
"""BioCypher adapter exposing Wang 2015's plain-medium growth ratios as graph nodes.

One record per Keio transporter deletion of Supplementary Table S3's ``No isoprenol``
column, 46 in all. Each genotype carries one ``BacterialDeletionPerturbation``, served
as ``bacterial perturbation``; the medium's pH is an environment perturbation and is a
first-class node, and the isoprenol edit of the sibling ``EnvChemgenWang2015Adapter`` is
absent, because this is the plain-medium arm. The phenotype is ``fitness phenotype``,
carrying the 12 h endpoint OD600 over BW25113's with the delta-method standard error of
that ratio (#826).
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.ecoli.wang2015_growth import GrowthWang2015Dataset


class GrowthWang2015Adapter(CellAdapter):
    """Cell adapter that serves GrowthWang2015Dataset to BioCypher."""

    def __init__(
        self,
        dataset: GrowthWang2015Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(current_dir, "conf", "growth_wang2015_adapter.yaml")
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
