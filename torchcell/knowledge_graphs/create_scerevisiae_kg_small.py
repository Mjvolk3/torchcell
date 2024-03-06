# torchcell/knowledge_graphs/create_scerevisiae_kg_small
# [[torchcell.knowledge_graphs.create_scerevisiae_kg_small]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/knowledge_graphs/create_scerevisiae_kg_small.py
# Test file: tests/torchcell/knowledge_graphs/test_create_scerevisiae_kg_small.py

from biocypher import BioCypher
from torchcell.adapters import (
    SmfCostanzo2016Adapter,
    DmfCostanzo2016Adapter,
    SmfKuzmin2018Adapter,
    DmfKuzmin2018Adapter,
    TmfKuzmin2018Adapter,
)
from torchcell.datasets.scerevisiae import (
    SmfCostanzo2016Dataset,
    DmfCostanzo2016Dataset,
    SmfKuzmin2018Dataset,
    DmfKuzmin2018Dataset,
    TmfKuzmin2018Dataset,
)
import logging
from dotenv import load_dotenv
import os
import os.path as osp
from datetime import datetime
import multiprocessing as mp
import sys
import math

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# WARNING do not print in this file! This file is used to generate a path to a bash script and printing to stdout will break the bash script path


def get_num_workers():
    """Get the number of CPUs allocated by SLURM."""
    # Try to get number of CPUs allocated by SLURM
    cpus_per_task = os.getenv("SLURM_CPUS_PER_TASK")
    if cpus_per_task is not None:
        return int(cpus_per_task)
    # Fallback: Use multiprocessing to get the total number of CPUs
    return mp.cpu_count()


def main() -> str:
    # Configure logging
    logging.basicConfig(level=logging.INFO, filename="biocypher_warnings.log")
    logging.captureWarnings(True)
    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")
    BIOCYPHER_CONFIG_PATH = os.getenv("BIOCYPHER_CONFIG_PATH")
    SCHEMA_CONFIG_PATH = os.getenv("SCHEMA_CONFIG_PATH")
    BIOCYPHER_OUT_PATH = os.getenv("BIOCYPHER_OUT_PATH")

    # Use this function to get the number of workers
    num_workers = get_num_workers()
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log.info(f"Number of workers: {num_workers}")
    bc = BioCypher(
        output_directory=osp.join(DATA_ROOT, BIOCYPHER_OUT_PATH, time),
        biocypher_config_path=BIOCYPHER_CONFIG_PATH,
        schema_config_path=SCHEMA_CONFIG_PATH,
    )
    # Partition workers
    num_workers = get_num_workers()
    io_workers = math.ceil(0.2 * num_workers)
    compute_workers = num_workers - io_workers
    chunk_size = int(1e5)
    loader_batch_size = int(1e3)

    # Ordered adapters from smallest to largest
    adapters = [
        SmfCostanzo2016Adapter(
            dataset=SmfCostanzo2016Dataset(
                root=osp.join(DATA_ROOT, "data/torchcell/smf_costanzo2016")
            ),
            compute_workers=compute_workers,
            io_workers=io_workers,
            chunk_size=chunk_size,
            loader_batch_size=loader_batch_size,
        ),
        DmfCostanzo2016Adapter(
            dataset=DmfCostanzo2016Dataset(
                root=osp.join(DATA_ROOT, "data/torchcell/dmf_costanzo2016_1e5"),
                subset_n=int(1e5),
            ),
            compute_workers=compute_workers,
            io_workers=io_workers,
            chunk_size=chunk_size,
            loader_batch_size=loader_batch_size,
        ),
        SmfKuzmin2018Adapter(
            dataset=SmfKuzmin2018Dataset(
                root=osp.join(DATA_ROOT, "data/torchcell/smf_kuzmin2018")
            ),
            compute_workers=compute_workers,
            io_workers=io_workers,
            chunk_size=chunk_size,
            loader_batch_size=loader_batch_size
        ),
        DmfKuzmin2018Adapter(
            dataset=DmfKuzmin2018Dataset(
                root=osp.join(DATA_ROOT, "data/torchcell/dmf_kuzmin2018")
            ),
            compute_workers=compute_workers,
            io_workers=io_workers,
            chunk_size=chunk_size,
            loader_batch_size=loader_batch_size
        ),
        TmfKuzmin2018Adapter(
            dataset=TmfKuzmin2018Dataset(
                root=osp.join(DATA_ROOT, "data/torchcell/tmf_kuzmin2018")
            ),
            compute_workers=compute_workers,
            io_workers=io_workers,
            chunk_size=chunk_size,
            loader_batch_size=loader_batch_size
        ),
    ]

    for adapter in adapters:
        bc.write_nodes(adapter.get_nodes())
        bc.write_edges(adapter.get_edges())

    log.info("Finished iterating nodes and edges")
    # Write admin import statement and schema information (for biochatter)
    bc.write_import_call()
    bc.write_schema_info(as_node=True)

    # bc.summary()
    # Returns bash script path

    relative_bash_script_path = osp.join(
        "biocypher-out", time, "neo4j-admin-import-call.sh"
    )
    return relative_bash_script_path


if __name__ == "__main__":
    print(main())
