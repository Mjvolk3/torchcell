# torchcell/adapters/kuzmin2020_adapter
# [[torchcell.adapters.kuzmin2020_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/kuzmin2020_adapter


from tqdm import tqdm
import hashlib
import json
from biocypher import BioCypher
from biocypher._create import BioCypherEdge, BioCypherNode
from biocypher._logger import get_logger
import logging
from typing import Set
from torchcell.datasets.scerevisiae.kuzmin2020 import (
    SmfKuzmin2020Dataset,
    DmfKuzmin2020Dataset,
    TmfKuzmin2020Dataset,
    DmiKuzmin2020Dataset,
    TmiKuzmin2020Dataset,
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


class SmfKuzmin2020Adapter(CellAdapter):
    def __init__(
        self,
        dataset: SmfKuzmin2020Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        current_dir = osp.dirname(osp.abspath(__file__))

        config_path = osp.join(current_dir, "conf", "smf_kuzmin2020_adapter.yaml")

        if not osp.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as file:
            yaml_config = yaml.safe_load(file)

        config = OmegaConf.create(yaml_config)

        super().__init__(
            config, dataset, process_workers, io_workers, chunk_size, loader_batch_size
        )
        print(
            f"SmfKuzmin2020Adapter initialized with config: {self.config}"
        )  # Debug print
        self.dataset = dataset
        self.process_workers = process_workers
        self.io_workers = io_workers
        self.chunk_size = chunk_size
        self.loader_batch_size = loader_batch_size


class DmfKuzmin2020Adapter(CellAdapter):
    def __init__(
        self,
        dataset: DmfKuzmin2020Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        current_dir = osp.dirname(osp.abspath(__file__))

        config_path = osp.join(current_dir, "conf", "dmf_kuzmin2020_adapter.yaml")

        if not osp.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as file:
            yaml_config = yaml.safe_load(file)

        config = OmegaConf.create(yaml_config)

        super().__init__(
            config, dataset, process_workers, io_workers, chunk_size, loader_batch_size
        )
        print(
            f"DmfKuzmin2020Adapter initialized with config: {self.config}"
        )  # Debug print
        self.dataset = dataset
        self.process_workers = process_workers
        self.io_workers = io_workers
        self.chunk_size = chunk_size
        self.loader_batch_size = loader_batch_size


class TmfKuzmin2020Adapter(CellAdapter):
    def __init__(
        self,
        dataset: TmfKuzmin2020Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        current_dir = osp.dirname(osp.abspath(__file__))

        config_path = osp.join(current_dir, "conf", "tmf_kuzmin2020_adapter.yaml")

        if not osp.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as file:
            yaml_config = yaml.safe_load(file)

        config = OmegaConf.create(yaml_config)

        super().__init__(
            config, dataset, process_workers, io_workers, chunk_size, loader_batch_size
        )
        print(
            f"TmfKuzmin2020Adapter initialized with config: {self.config}"
        )  # Debug print
        self.dataset = dataset
        self.process_workers = process_workers
        self.io_workers = io_workers
        self.chunk_size = chunk_size
        self.loader_batch_size = loader_batch_size


class DmiKuzmin2020Adapter(CellAdapter):
    def __init__(
        self,
        dataset: DmiKuzmin2020Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(current_dir, "conf", "dmi_kuzmin2020_adapter.yaml")

        if not osp.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as file:
            yaml_config = yaml.safe_load(file)

        config = OmegaConf.create(yaml_config)

        super().__init__(
            config, dataset, process_workers, io_workers, chunk_size, loader_batch_size
        )
        print(f"DmiKuzmin2020Adapter initialized with config: {self.config}")
        self.dataset = dataset
        self.process_workers = process_workers
        self.io_workers = io_workers
        self.chunk_size = chunk_size
        self.loader_batch_size = loader_batch_size


class TmiKuzmin2020Adapter(CellAdapter):
    def __init__(
        self,
        dataset: TmiKuzmin2020Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        current_dir = osp.dirname(osp.abspath(__file__))
        config_path = osp.join(current_dir, "conf", "tmi_kuzmin2020_adapter.yaml")

        if not osp.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as file:
            yaml_config = yaml.safe_load(file)

        config = OmegaConf.create(yaml_config)

        super().__init__(
            config, dataset, process_workers, io_workers, chunk_size, loader_batch_size
        )
        print(f"TmiKuzmin2020Adapter initialized with config: {self.config}")
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
    dataset = SmfKuzmin2020Dataset(
        root=osp.join(DATA_ROOT, "data/torchcell/smf_kuzmin2020")
    )
    adapter = SmfKuzmin2020Adapter(
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
    # dataset = DmfKuzmin2020Dataset(osp.join(DATA_ROOT, "data/torchcell/dmf_kuzmin2020"))
    # adapter = DmfKuzmin2020Adapter(
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
    # dataset = TmfKuzmin2020Dataset(osp.join(DATA_ROOT, "data/torchcell/tmf_kuzmin2020"))
    # adapter = TmfKuzmin2020Adapter(
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
    # dataset = DmiKuzmin2020Dataset(osp.join(DATA_ROOT, "data/torchcell/dmi_kuzmin2020"))
    # adapter = DmiKuzmin2020Adapter(
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
    # dataset = DmfKuzmin2020Dataset(osp.join(DATA_ROOT, "data/torchcell/dmf_kuzmin2020"))
    # adapter = DmfKuzmin2020Adapter(
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
    # dataset = TmfKuzmin2020Dataset(osp.join(DATA_ROOT, "data/torchcell/tmf_kuzmin2020"))
    # adapter = TmfKuzmin2020Adapter(
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
