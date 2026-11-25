# torchcell/adapters/choe2019_metabolite_adapter.py
# [[torchcell.adapters.choe2019_metabolite_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/choe2019_metabolite_adapter.py
# Test file: tests/torchcell/adapters/test_choe2019_metabolite_adapter.py
"""BioCypher adapter exposing Choe 2019's per-culture metabolite levels as
knowledge-graph nodes and edges.

One record per released Fig. 1g pyruvate cell and Fig. 3d NAD(P)H ratio or ATP cell
(38), MG1655 on the bare assembly and eMS57 on its named background, every genotype
empty, so the perturbation pair is off. No compound is added to the medium, so the
environment-perturbation pair is off too. The phenotype is ``metabolite phenotype``.
"""

import os.path as osp

import yaml
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.ecoli.choe2019_growth_rate import MetaboliteChoe2019Dataset


class MetaboliteChoe2019Adapter(CellAdapter):
    """Cell adapter that serves MetaboliteChoe2019Dataset to BioCypher."""

    def __init__(
        self,
        dataset: MetaboliteChoe2019Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(current_dir, "conf", "metabolite_choe2019_adapter.yaml")
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
