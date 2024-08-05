# torchcell/adapters/costanzo2016_adapter.py
# [[torchcell.adapters.costanzo2016_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/costanzo2016_adapter.py
# Test file: tests/torchcell/adapters/test_costanzo2016_adapter.py

from tqdm import tqdm
import hashlib
import json
from biocypher import BioCypher
from biocypher._create import BioCypherEdge, BioCypherNode
from biocypher._logger import get_logger
import logging
from typing import Set
from torchcell.datasets.scerevisiae.costanzo2016 import (
    SmfCostanzo2016Dataset,
    DmfCostanzo2016Dataset,
    DmiCostanzo2016Dataset,
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


class SmfCostanzo2016Adapter(CellAdapter):
    def __init__(
        self,
        dataset: SmfCostanzo2016Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        current_dir = osp.dirname(osp.abspath(__file__))

        config_path = osp.join(current_dir, "conf", "smf_costanzo_2016_adapter.yaml")

        if not osp.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as file:
            yaml_config = yaml.safe_load(file)

        config = OmegaConf.create(yaml_config)

        super().__init__(
            config, dataset, process_workers, io_workers, chunk_size, loader_batch_size
        )
        print(
            f"SmfCostanzo2016Adapter initialized with config: {self.config}"
        )  # Debug print
        self.dataset = dataset
        self.process_workers = process_workers
        self.io_workers = io_workers
        self.chunk_size = chunk_size
        self.loader_batch_size = loader_batch_size


class DmfCostanzo2016Adapter(CellAdapter):
    def __init__(
        self,
        dataset: DmfCostanzo2016Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        current_dir = osp.dirname(osp.abspath(__file__))

        config_path = osp.join(current_dir, "conf", "dmf_costanzo_2016_adapter.yaml")

        if not osp.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as file:
            yaml_config = yaml.safe_load(file)

        config = OmegaConf.create(yaml_config)

        super().__init__(
            config, dataset, process_workers, io_workers, chunk_size, loader_batch_size
        )
        print(
            f"DmfCostanzo2016Adapter initialized with config: {self.config}"
        )  # Debug print
        self.dataset = dataset
        self.process_workers = process_workers
        self.io_workers = io_workers
        self.chunk_size = chunk_size
        self.loader_batch_size = loader_batch_size


class DmiCostanzo2016Adapter(CellAdapter):
    def __init__(
        self,
        dataset: DmiCostanzo2016Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        current_dir = osp.dirname(osp.abspath(__file__))

        config_path = osp.join(current_dir, "conf", "dmi_costanzo_2016_adapter.yaml")

        if not osp.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as file:
            yaml_config = yaml.safe_load(file)

        config = OmegaConf.create(yaml_config)

        super().__init__(
            config, dataset, process_workers, io_workers, chunk_size, loader_batch_size
        )
        print(
            f"DmiCostanzo2016Adapter initialized with config: {self.config}"
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
    dataset = SmfCostanzo2016Dataset(
        osp.join(DATA_ROOT, "data/torchcell/smf_costanzo2016")
    )
    num_workers = mp.cpu_count()
    io_workers = math.ceil(0.2 * num_workers)
    process_workers = num_workers - io_workers
    adapter = SmfCostanzo2016Adapter(
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
    # BUG printing this gives hangs entire process.
    # bc.summary()
    wandb.finish()

    # # # DMF
    # bc = BioCypher(
    #     output_directory=osp.join(DATA_ROOT, "database/biocypher-out", time),
    #     biocypher_config_path=BIOCYPHER_CONFIG_PATH,
    #     schema_config_path=SCHEMA_CONFIG_PATH,
    # )
    # # dataset = DmfCostanzo2016Dataset(
    # #     root=osp.join(DATA_ROOT, "data/torchcell/dmf_costanzo2016")
    # # )
    # dataset = DmfCostanzo2016Dataset(
    #     root=osp.join(DATA_ROOT, "data/torchcell/dmf_costanzo2016_1e4"),
    #     subset_n=int(1e4),
    # )
    # adapter = DmfCostanzo2016Adapter(
    #     dataset=dataset,
    #     process_workers=10,
    #     io_workers=10,
    #     chunk_size=100,
    #     loader_batch_size=10,
    # )
    # bc.write_nodes(adapter.get_nodes())
    # bc.write_edges(adapter.get_edges())
    # bc.write_import_call()
    # bc.write_schema_info(as_node=True)
    # bc.summary()

    # ## Dmi
    # # bc = BioCypher(
    # #     output_directory=osp.join(DATA_ROOT, "database/biocypher-out", time),
    # #     biocypher_config_path=BIOCYPHER_CONFIG_PATH,
    # #     schema_config_path=SCHEMA_CONFIG_PATH,
    # # )
    # # dataset = DmiCostanzo2016Dataset(
    # #     root=osp.join(DATA_ROOT, "data/torchcell/dmi_costanzo2016_1e6"),
    # #     subset_n=int(1e6),
    # # )
    # # adapter = DmiCostanzo2016Adapter(
    # #     dataset=dataset,
    # #     process_workers=10,
    # #     io_workers=10,
    # #     chunk_size=100,
    # #     loader_batch_size=10,
    # # )
    # # bc.write_nodes(adapter.get_nodes())
    # # bc.write_edges(adapter.get_edges())
    # # bc.write_import_call()
    # # bc.write_schema_info(as_node=True)
    # # bc.summary()


if __name__ == "__main__":
    main()
