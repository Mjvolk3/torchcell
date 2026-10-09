# torchcell/adapters/schmidt2016_s23_growth_rate_adapter.py
# [[torchcell.adapters.schmidt2016_s23_growth_rate_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/schmidt2016_s23_growth_rate_adapter.py
# Test file: tests/torchcell/adapters/test_schmidt2016_s23_growth_rate_adapter.py
"""BioCypher adapter exposing Schmidt 2016's Table S23 growth rates as graph nodes.

One record per kept (growth condition, BW25113) row of Supplementary Table 23, 15 in
all. Every released row is wild type, so the genotype carries no perturbation and no
``bacterial perturbation`` node is served; the carbon sources, the NaCl level and the pH
are environment perturbations and are first-class nodes. The phenotype is ``environment
response phenotype``, carrying the absolute growth rate in h^-1 with the released Stdev
and its conservatively resolved replicate count (#826).
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.ecoli.schmidt2016_s23_growth_rate import (
    GrowthRateS23Schmidt2016Dataset,
)


class GrowthRateS23Schmidt2016Adapter(CellAdapter):
    """Cell adapter that serves GrowthRateS23Schmidt2016Dataset to BioCypher."""

    def __init__(
        self,
        dataset: GrowthRateS23Schmidt2016Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(
            current_dir, "conf", "growth_rate_s23_schmidt2016_adapter.yaml"
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
