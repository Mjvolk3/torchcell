# torchcell/adapters/synth_leth_db
# [[torchcell.adapters.synth_leth_db]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/synth_leth_db
# Test file: tests/torchcell/adapters/test_synth_leth_db.py

from tqdm import tqdm
import hashlib
import json
from biocypher import BioCypher
from biocypher._create import BioCypherEdge, BioCypherNode
from biocypher._logger import get_logger
import logging
from typing import Set
from torchcell.datasets.scerevisiae.synth_leth_db import (
    SynthLethalityYeastSynthLethDbDataset,
    SynthRescueYeastSynthLethDbDataset,
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


class SynthLethalityYeastSynthLethDbAdapter(CellAdapter):
    def __init__(
        self,
        dataset: SynthLethalityYeastSynthLethDbDataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        current_dir = osp.dirname(osp.abspath(__file__))

        config_path = osp.join(
            current_dir, "conf", "synth_lethality_yeast_synth_leth_db_adapter.yaml"
        )

        if not osp.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as file:
            yaml_config = yaml.safe_load(file)

        config = OmegaConf.create(yaml_config)

        super().__init__(
            config, dataset, process_workers, io_workers, chunk_size, loader_batch_size
        )
        print(
            f"SynthLethalityYeastSynthLethDbAdapter initialized with config: {self.config}"
        )  # Debug print
        self.dataset = dataset
        self.process_workers = process_workers
        self.io_workers = io_workers
        self.chunk_size = chunk_size
        self.loader_batch_size = loader_batch_size


class SynthRescueYeastSynthLethDbAdapter(CellAdapter):
    def __init__(
        self,
        dataset: SynthLethalityYeastSynthLethDbDataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        current_dir = osp.dirname(osp.abspath(__file__))

        config_path = osp.join(
            current_dir, "conf", "synth_rescue_yeast_synth_leth_db_adapter.yaml"
        )

        if not osp.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as file:
            yaml_config = yaml.safe_load(file)

        config = OmegaConf.create(yaml_config)

        super().__init__(
            config, dataset, process_workers, io_workers, chunk_size, loader_batch_size
        )
        print(
            f"SynthRescueYeastSynthLethDbAdapter initialized with config: {self.config}"
        )  # Debug print
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
    from torchcell.graph import SCerevisiaeGraph
    from torchcell.sequence.genome.scerevisiae.S288C import SCerevisiaeGenome

    load_dotenv()
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    DATA_ROOT = os.getenv("DATA_ROOT")
    BIOCYPHER_CONFIG_PATH = os.getenv("BIOCYPHER_CONFIG_PATH")
    SCHEMA_CONFIG_PATH = os.getenv("SCHEMA_CONFIG_PATH")
    num_workers = mp.cpu_count()
    io_workers = math.ceil(0.2 * num_workers)
    process_workers = num_workers - io_workers

    ## Synth Lethality
    bc = BioCypher(
        output_directory=osp.join(DATA_ROOT, "database/biocypher-out", time),
        biocypher_config_path=BIOCYPHER_CONFIG_PATH,
        schema_config_path=SCHEMA_CONFIG_PATH,
    )
    genome = SCerevisiaeGenome(
        data_root=osp.join(DATA_ROOT, "data/sgd/genome"), overwrite=True
    )
    dataset = SynthLethalityYeastSynthLethDbDataset(genome=genome)
    del genome

    adapter = SynthLethalityYeastSynthLethDbAdapter(
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

    ## Synth Rescue
    bc = BioCypher(
        output_directory=osp.join(DATA_ROOT, "database/biocypher-out", time),
        biocypher_config_path=BIOCYPHER_CONFIG_PATH,
        schema_config_path=SCHEMA_CONFIG_PATH,
    )
    genome = SCerevisiaeGenome(
        data_root=osp.join(DATA_ROOT, "data/sgd/genome"), overwrite=True
    )

    dataset = SynthRescueYeastSynthLethDbDataset(genome=genome)
    del genome

    adapter = SynthRescueYeastSynthLethDbAdapter(
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
