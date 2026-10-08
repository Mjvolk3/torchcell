# torchcell/adapters/rapp2026_growth_adapter.py
# [[torchcell.adapters.rapp2026_growth_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/rapp2026_growth_adapter.py
# Test file: tests/torchcell/adapters/test_rapp2026_growth_adapter.py
"""BioCypher adapter exposing Rapp 2026's CRISPRi growth AUC as knowledge-graph
nodes and edges.

One record per knockdown strain of the arrayed library covering every iML1515 gene: a
single ``BacterialCrisprInterferencePerturbation`` served as a
``bacterial perturbation`` node with its ``CrisprConstruct``, and the strain's growth
as a ``FitnessPhenotype`` (the trapezoid AUC of its released OD600 curve over the
control strains' mean). The environment is one medium plus the anhydrotetracycline that
induces the knockdown, so the environment-perturbation pair is served as well.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.ecoli.rapp2026_platforms import GrowthAucRapp2026Dataset


class GrowthAucRapp2026Adapter(CellAdapter):
    """Cell adapter that serves GrowthAucRapp2026Dataset to BioCypher."""

    def __init__(
        self,
        dataset: GrowthAucRapp2026Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(current_dir, "conf", "growth_auc_rapp2026_adapter.yaml")
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
