"""BioCypher adapter for the O'Duibhir 2014 single-mutant fitness dataset."""

# torchcell/adapters/oduibhir2014_adapter.py
# [[torchcell.adapters.oduibhir2014_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/oduibhir2014_adapter.py
# Test file: tests/torchcell/adapters/test_oduibhir2014_adapter.py

import logging
import os.path as osp

import yaml
from biocypher._logger import get_logger
from omegaconf import OmegaConf

from torchcell.adapters.cell_adapter import CellAdapter
from torchcell.datasets.scerevisiae.oduibhir2014 import SmfODuibhir2014Dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
logger = get_logger("biocypher")
logger.setLevel(logging.ERROR)


class SmfODuibhir2014Adapter(CellAdapter):
    """Adapter for the O'Duibhir 2014 single-mutant fitness (SMF) dataset."""

    def __init__(
        self,
        dataset: SmfODuibhir2014Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        """Load the adapter config YAML and initialize the CellAdapter.

        Args:
            dataset: The O'Duibhir 2014 dataset to adapt.
            process_workers: Number of worker processes for processing.
            io_workers: Number of worker processes for I/O.
            chunk_size: Number of records per processing chunk.
            loader_batch_size: Batch size used when loading records.
        """
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(current_dir, "conf", "smf_oduibhir2014_adapter.yaml")

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
