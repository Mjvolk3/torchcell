# torchcell/adapters/ohya2005_adapter.py
# [[torchcell.adapters.ohya2005_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/ohya2005_adapter.py
# Test file: tests/torchcell/adapters/test_ohya2005_adapter.py

from tqdm import tqdm
import hashlib
import json
from biocypher import BioCypher
from biocypher._create import BioCypherEdge, BioCypherNode
from biocypher._logger import get_logger
import logging
from typing import Set
from torchcell.datasets.scerevisiae.Ohya2005 import (
    ScmdOhya2005Dataset,
)
from torchcell.adapters.cell_adapter import CellAdapter
import yaml
from omegaconf import OmegaConf, DictConfig
import os.path as osp


# logging
# Get the biocypher logger
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
logger = get_logger("biocypher")
logger.setLevel(logging.ERROR)


class ScmdOhya2005Adapter(CellAdapter):
    def __init__(
        self,
        dataset: ScmdOhya2005Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        current_dir = osp.dirname(osp.abspath(__file__))

        config_path = osp.join(current_dir, "conf", "scmd_ohya2005_adapter.yaml")

        if not osp.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as file:
            yaml_config = yaml.safe_load(file)

        config = OmegaConf.create(yaml_config)

        super().__init__(
            config, dataset, process_workers, io_workers, chunk_size, loader_batch_size
        )
        print(
            f"ScmdOhya2005Adapter initialized with config: {self.config}"
        )  # Debug print
        self.dataset = dataset
        self.process_workers = process_workers
        self.io_workers = io_workers
        self.chunk_size = chunk_size
        self.loader_batch_size = loader_batch_size


def main():
    import os.path as osp
    from dotenv import load_dotenv
    from datetime import datetime
    import os
    import multiprocessing as mp
    import math
    import wandb

    ##
    load_dotenv()
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    DATA_ROOT = os.getenv("DATA_ROOT")
    BIOCYPHER_CONFIG_PATH = os.getenv("BIOCYPHER_CONFIG_PATH")
    SCHEMA_CONFIG_PATH = os.getenv("SCHEMA_CONFIG_PATH")

    # SMF
    bc = BioCypher(
        output_directory=osp.join(DATA_ROOT, "database/biocypher-out", time),
        biocypher_config_path=BIOCYPHER_CONFIG_PATH,
        schema_config_path=SCHEMA_CONFIG_PATH,
    )
    dataset = ScmdOhya2005Dataset(
        osp.join(DATA_ROOT, "data/torchcell/scmd_ohya2005")
    )
    num_workers = mp.cpu_count()
    io_workers = math.ceil(0.2 * num_workers)
    process_workers = num_workers - io_workers
    adapter = ScmdOhya2005Adapter(
        dataset=dataset,
        process_workers=process_workers,
        io_workers=io_workers,
        chunk_size=int(1e4),
        loader_batch_size=int(1e4),
    )
    bc.write_nodes(adapter.get_nodes())
    bc.write_edges(adapter.get_edges())
    bc.write_import_call()
    bc.write_schema_info(as_node=True)
    wandb.finish()


if __name__ == "__main__":
    main()