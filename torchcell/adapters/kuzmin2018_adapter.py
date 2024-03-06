# torchcell/adapters/kuzmin2018_adapter.py
# [[torchcell.adapters.kuzmin2018_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/kuzmin2018_adapter.py
# Test file: tests/torchcell/adapters/test_kuzmin2018_adapter.py

from tqdm import tqdm
import hashlib
import json
from biocypher._create import BioCypherEdge, BioCypherNode
from biocypher._logger import logger
from typing import Generator, Set
from torchcell.datasets.scerevisiae import (
    SmfKuzmin2018Dataset,
    DmfKuzmin2018Dataset,
    TmfKuzmin2018Dataset,
)
from torchcell.datamodels.schema import Genotype
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
from torchcell.adapters.cell_adapter import CellAdapter

# logger.debug(f"Loading module {__name__}.")
# logger.setLevel("INFO")


class SmfKuzmin2018Adapter(CellAdapter):
    def __init__(
        self,
        dataset: SmfKuzmin2018Dataset,
        compute_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        super().__init__(
            dataset, compute_workers, io_workers, chunk_size, loader_batch_size
        )
        self.dataset = dataset
        self.compute_workers = compute_workers
        self.io_workers = io_workers
        self.chunk_size = chunk_size
        self.loader_batch_size = loader_batch_size


class DmfKuzmin2018Adapter(CellAdapter):
    def __init__(
        self,
        dataset: DmfKuzmin2018Dataset,
        compute_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        super().__init__(
            dataset, compute_workers, io_workers, chunk_size, loader_batch_size
        )
        self.dataset = dataset
        self.compute_workers = compute_workers
        self.io_workers = io_workers
        self.chunk_size = chunk_size
        self.loader_batch_size = loader_batch_size


class TmfKuzmin2018Adapter(CellAdapter):
    def __init__(
        self,
        dataset: TmfKuzmin2018Dataset,
        compute_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
    ):
        super().__init__(
            dataset, compute_workers, io_workers, chunk_size, loader_batch_size
        )
        self.dataset = dataset
        self.compute_workers = compute_workers
        self.io_workers = io_workers
        self.chunk_size = chunk_size
        self.loader_batch_size = loader_batch_size


if __name__ == "__main__":
    from biocypher import BioCypher
    from dotenv import load_dotenv
    from datetime import datetime
    import os
    import os.path as osp

    load_dotenv()
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    DATA_ROOT = os.getenv("DATA_ROOT")
    BIOCYPHER_CONFIG_PATH = os.getenv("BIOCYPHER_CONFIG_PATH")
    SCHEMA_CONFIG_PATH = os.getenv("SCHEMA_CONFIG_PATH")
    compute_workers = 8
    io_workers = 2

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
        dataset=dataset, compute_workers=compute_workers, io_workers=io_workers
    )
    bc.write_nodes(adapter.get_nodes())
    bc.write_edges(adapter.get_edges())
    bc.write_import_call()
    bc.write_schema_info(as_node=True)
    bc.summary()

    # ## Dmf
    bc = BioCypher(
        output_directory=osp.join(DATA_ROOT, "database/biocypher-out", time),
        biocypher_config_path=BIOCYPHER_CONFIG_PATH,
        schema_config_path=SCHEMA_CONFIG_PATH,
    )
    dataset = DmfKuzmin2018Dataset(osp.join(DATA_ROOT, "data/torchcell/dmf_kuzmin2018"))
    adapter = DmfKuzmin2018Adapter(
        dataset=dataset, compute_workers=compute_workers, io_workers=io_workers
    )
    bc.write_nodes(adapter.get_nodes())
    bc.write_edges(adapter.get_edges())
    bc.write_import_call()
    bc.write_schema_info(as_node=True)
    bc.summary()

    # ## Tmf
    bc = BioCypher(
        output_directory=osp.join(DATA_ROOT, "database/biocypher-out", time),
        biocypher_config_path=BIOCYPHER_CONFIG_PATH,
        schema_config_path=SCHEMA_CONFIG_PATH,
    )
    dataset = TmfKuzmin2018Dataset(osp.join(DATA_ROOT, "data/torchcell/tmf_kuzmin2018"))
    adapter = TmfKuzmin2018Adapter(
        dataset=dataset, compute_workers=compute_workers, io_workers=io_workers
    )
    bc.show_ontology_structure()
    bc.write_nodes(adapter.get_nodes())
    bc.write_edges(adapter.get_edges())
    bc.write_import_call()
    bc.write_schema_info(as_node=True)
    bc.summary()
    print()
