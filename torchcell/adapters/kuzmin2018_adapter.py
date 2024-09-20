# torchcell/adapters/kuzmin2018_adapter.py
# [[torchcell.adapters.kuzmin2018_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/kuzmin2018_adapter.py
# Test file: tests/torchcell/adapters/test_kuzmin2018_adapter.py

from tqdm import tqdm
import hashlib
import json
from biocypher import BioCypher
from biocypher._create import BioCypherEdge, BioCypherNode
from biocypher._logger import get_logger
import logging
from typing import Set
from torchcell.datasets.scerevisiae.kuzmin2018 import (
    SmfKuzmin2018Dataset,
    DmfKuzmin2018Dataset,
    TmfKuzmin2018Dataset,
    DmiKuzmin2018Dataset,
    TmiKuzmin2018Dataset,
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


class SmfKuzmin2018Adapter(CellAdapter):
    def __init__(
        self,
        dataset: SmfKuzmin2018Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        current_dir = osp.dirname(osp.abspath(__file__))

        config_path = osp.join(current_dir, "conf", "smf_kuzmin2018_adapter.yaml")

        if not osp.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as file:
            yaml_config = yaml.safe_load(file)

        config = OmegaConf.create(yaml_config)

        super().__init__(
            config, dataset, process_workers, io_workers, chunk_size, loader_batch_size
        )
        print(
            f"SmfKuzmin2018Adapter initialized with config: {self.config}"
        )  # Debug print
        self.dataset = dataset
        self.process_workers = process_workers
        self.io_workers = io_workers
        self.chunk_size = chunk_size
        self.loader_batch_size = loader_batch_size


class DmfKuzmin2018Adapter(CellAdapter):
    def __init__(
        self,
        dataset: DmfKuzmin2018Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        current_dir = osp.dirname(osp.abspath(__file__))

        config_path = osp.join(current_dir, "conf", "dmf_kuzmin2018_adapter.yaml")

        if not osp.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as file:
            yaml_config = yaml.safe_load(file)

        config = OmegaConf.create(yaml_config)

        super().__init__(
            config, dataset, process_workers, io_workers, chunk_size, loader_batch_size
        )
        print(
            f"DmfKuzmin2018Adapter initialized with config: {self.config}"
        )  # Debug print
        self.dataset = dataset
        self.process_workers = process_workers
        self.io_workers = io_workers
        self.chunk_size = chunk_size
        self.loader_batch_size = loader_batch_size


class TmfKuzmin2018Adapter(CellAdapter):
    def __init__(
        self,
        dataset: TmfKuzmin2018Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        current_dir = osp.dirname(osp.abspath(__file__))

        config_path = osp.join(current_dir, "conf", "tmf_kuzmin2018_adapter.yaml")

        if not osp.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as file:
            yaml_config = yaml.safe_load(file)

        config = OmegaConf.create(yaml_config)

        super().__init__(
            config, dataset, process_workers, io_workers, chunk_size, loader_batch_size
        )
        print(
            f"TmfKuzmin2018Adapter initialized with config: {self.config}"
        )  # Debug print
        self.dataset = dataset
        self.process_workers = process_workers
        self.io_workers = io_workers
        self.chunk_size = chunk_size
        self.loader_batch_size = loader_batch_size


class DmiKuzmin2018Adapter(CellAdapter):
    def __init__(
        self,
        dataset: DmiKuzmin2018Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(current_dir, "conf", "dmi_kuzmin2018_adapter.yaml")

        if not osp.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as file:
            yaml_config = yaml.safe_load(file)

        config = OmegaConf.create(yaml_config)

        super().__init__(
            config, dataset, process_workers, io_workers, chunk_size, loader_batch_size
        )
        print(f"DmiKuzmin2018Adapter initialized with config: {self.config}")
        self.dataset = dataset
        self.process_workers = process_workers
        self.io_workers = io_workers
        self.chunk_size = chunk_size
        self.loader_batch_size = loader_batch_size


class TmiKuzmin2018Adapter(CellAdapter):
    def __init__(
        self,
        dataset: TmiKuzmin2018Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(current_dir, "conf", "tmi_kuzmin2018_adapter.yaml")

        if not osp.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as file:
            yaml_config = yaml.safe_load(file)

        config = OmegaConf.create(yaml_config)

        super().__init__(
            config, dataset, process_workers, io_workers, chunk_size, loader_batch_size
        )
        print(f"TmiKuzmin2018Adapter initialized with config: {self.config}")
        self.dataset = dataset
        self.process_workers = process_workers
        self.io_workers = io_workers
        self.chunk_size = chunk_size
        self.loader_batch_size = loader_batch_size


if __name__ == "__main__":
    from dotenv import load_dotenv
    from datetime import datetime
    import os
    import os.path as osp
    import multiprocessing as mp
    import math

    load_dotenv()
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    DATA_ROOT = os.getenv("DATA_ROOT")
    BIOCYPHER_CONFIG_PATH = os.getenv("BIOCYPHER_CONFIG_PATH")
    SCHEMA_CONFIG_PATH = os.getenv("SCHEMA_CONFIG_PATH")
    num_workers = mp.cpu_count()
    io_workers = math.ceil(0.2 * num_workers)
    process_workers = num_workers - io_workers

    ## Smf
    bc = BioCypher(
        output_directory=osp.join(DATA_ROOT, "database/biocypher-out", time),
        biocypher_config_path=BIOCYPHER_CONFIG_PATH,
        schema_config_path=SCHEMA_CONFIG_PATH,
    )
    dataset = SmfKuzmin2018Dataset(
        root=osp.join(DATA_ROOT, "data/torchcell/smf_kuzmin2018")
    )
    adapter = SmfKuzmin2018Adapter(
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
    # bc.summary()

    # ## Dmf
    # bc = BioCypher(
    #     output_directory=osp.join(DATA_ROOT, "database/biocypher-out", time),
    #     biocypher_config_path=BIOCYPHER_CONFIG_PATH,
    #     schema_config_path=SCHEMA_CONFIG_PATH,
    # )
    # dataset = DmfKuzmin2018Dataset(osp.join(DATA_ROOT, "data/torchcell/dmf_kuzmin2018"))
    # adapter = DmfKuzmin2018Adapter(
    #     dataset=dataset,
    #     process_workers=process_workers,
    #     io_workers=io_workers,
    #     chunk_size=int(1e4),
    #     loader_batch_size=int(1e4),
    # )
    # bc.write_nodes(adapter.get_nodes())
    # bc.write_edges(adapter.get_edges())
    # bc.write_import_call()
    # bc.write_schema_info(as_node=True)
    # bc.summary()

    # ## Tmf
    # bc = BioCypher(
    #     output_directory=osp.join(DATA_ROOT, "database/biocypher-out", time),
    #     biocypher_config_path=BIOCYPHER_CONFIG_PATH,
    #     schema_config_path=SCHEMA_CONFIG_PATH,
    # )
    # dataset = TmfKuzmin2018Dataset(osp.join(DATA_ROOT, "data/torchcell/tmf_kuzmin2018"))
    # adapter = TmfKuzmin2018Adapter(
    #     dataset=dataset,
    #     process_workers=process_workers,
    #     io_workers=io_workers,
    #     chunk_size=int(1e4),
    #     loader_batch_size=int(1e4),
    # )
    # bc.write_nodes(adapter.get_nodes())
    # bc.write_edges(adapter.get_edges())
    # bc.write_import_call()
    # bc.write_schema_info(as_node=True)
    # bc.summary()

    # ## Dmi
    # bc = BioCypher(
    #     output_directory=osp.join(DATA_ROOT, "database/biocypher-out", time),
    #     biocypher_config_path=BIOCYPHER_CONFIG_PATH,
    #     schema_config_path=SCHEMA_CONFIG_PATH,
    # )
    # dataset = DmiKuzmin2018Dataset(osp.join(DATA_ROOT, "data/torchcell/dmi_kuzmin2018"))
    # adapter = DmiKuzmin2018Adapter(
    #     dataset=dataset,
    #     process_workers=process_workers,
    #     io_workers=io_workers,
    #     chunk_size=int(1e4),
    #     loader_batch_size=int(1e4),
    # )
    # bc.write_nodes(adapter.get_nodes())
    # bc.write_edges(adapter.get_edges())
    # bc.write_import_call()
    # bc.write_schema_info(as_node=True)
    # bc.summary()

    # ## Dmf
    # bc = BioCypher(
    #     output_directory=osp.join(DATA_ROOT, "database/biocypher-out", time),
    #     biocypher_config_path=BIOCYPHER_CONFIG_PATH,
    #     schema_config_path=SCHEMA_CONFIG_PATH,
    # )
    # dataset = DmfKuzmin2018Dataset(osp.join(DATA_ROOT, "data/torchcell/dmf_kuzmin2018"))
    # adapter = DmfKuzmin2018Adapter(
    #     dataset=dataset,
    #     process_workers=process_workers,
    #     io_workers=io_workers,
    #     chunk_size=int(1e4),
    #     loader_batch_size=int(1e4),
    # )
    # bc.write_nodes(adapter.get_nodes())
    # bc.write_edges(adapter.get_edges())
    # bc.write_import_call()
    # bc.write_schema_info(as_node=True)
    # bc.summary()

    # ## Tmf
    # bc = BioCypher(
    #     output_directory=osp.join(DATA_ROOT, "database/biocypher-out", time),
    #     biocypher_config_path=BIOCYPHER_CONFIG_PATH,
    #     schema_config_path=SCHEMA_CONFIG_PATH,
    # )
    # dataset = TmfKuzmin2018Dataset(osp.join(DATA_ROOT, "data/torchcell/tmf_kuzmin2018"))
    # adapter = TmfKuzmin2018Adapter(
    #     dataset=dataset,
    #     process_workers=process_workers,
    #     io_workers=io_workers,
    #     chunk_size=int(1e4),
    #     loader_batch_size=int(1e4),
    # )
    # bc.write_nodes(adapter.get_nodes())
    # bc.write_edges(adapter.get_edges())
    # bc.write_import_call()
    # bc.write_schema_info(as_node=True)
    # bc.summary()
