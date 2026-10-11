# torchcell/adapters/messner2021_adapter.py
# [[torchcell.adapters.messner2021_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/messner2021_adapter.py
# Test file: tests/torchcell/adapters/test_messner2021_adapter.py
"""BioCypher adapter serving the Messner 2021 drug-response proteomes to the graph."""

import os.path as osp

import yaml
from omegaconf import DictConfig, OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.scerevisiae.messner2021 import ProteomeMessner2021Dataset


class ProteomeMessner2021Adapter(CellAdapter):
    """Serves the 70 drug-run proteomes (``ProteinAbundancePhenotype``).

    The genotype is the reference strain on every record, so no perturbation node is
    enabled; the compound and the DMSO control are environment perturbation nodes.
    """

    def __init__(
        self,
        dataset: ProteomeMessner2021Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter conf enable-list and initialize the base CellAdapter."""
        config_path = osp.join(
            osp.dirname(osp.abspath(__file__)),
            "conf",
            "proteome_messner2021_adapter.yaml",
        )
        if not osp.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path) as file:
            config = OmegaConf.create(yaml.safe_load(file))
        if not isinstance(config, DictConfig):
            raise TypeError(f"{config_path} is not a mapping")
        super().__init__(
            config, dataset, process_workers, io_workers, chunk_size, loader_batch_size
        )
        self.dataset = dataset
        self.process_workers = process_workers
        self.io_workers = io_workers
        self.chunk_size = chunk_size
        self.loader_batch_size = loader_batch_size
