# torchcell/knowledge_graphs/dmi_kuzmin_2018_kg.py
# [[torchcell.knowledge_graphs.dmi_kuzmin_2018_kg]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/knowledge_graphs/dmi_kuzmin_2018_kg
# Test file: tests/torchcell/knowledge_graphs/test_dmi_kuzmin_2018_kg.py

from biocypher import BioCypher
import torchcell
import logging
from dotenv import load_dotenv
import os
import os.path as osp
from datetime import datetime
import multiprocessing as mp
import math
import wandb
from omegaconf import OmegaConf
import hydra
import time
import certifi
from torchcell.adapters import DmiKuzmin2018Adapter
from torchcell.datasets.scerevisiae.kuzmin2018 import DmiKuzmin2018Dataset

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
os.environ["SSL_CERT_FILE"] = certifi.where()


def get_num_workers():
    cpus_per_task = os.getenv("SLURM_CPUS_PER_TASK")
    return int(cpus_per_task) if cpus_per_task else 10


@hydra.main(version_base=None, config_path="conf", config_name="kg_small")
def main(cfg):
    # Initialize wandb
    wandb.init(
        project="dmi_kuzmin_2018_kg",
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
    )

    # Load environment variables
    load_dotenv(wandb.config.env_path)
    DATA_ROOT = os.getenv("DATA_ROOT")
    BIOCYPHER_CONFIG_PATH = os.getenv("BIOCYPHER_CONFIG_PATH")
    SCHEMA_CONFIG_PATH = os.getenv("SCHEMA_CONFIG_PATH")
    BIOCYPHER_OUT_PATH = os.getenv("BIOCYPHER_OUT_PATH")

    # Set up BioCypher
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_directory = osp.join(DATA_ROOT, BIOCYPHER_OUT_PATH, time_str)
    bc = BioCypher(
        output_directory=output_directory,
        biocypher_config_path=BIOCYPHER_CONFIG_PATH,
        schema_config_path=SCHEMA_CONFIG_PATH,
    )

    # Set up workers
    num_workers = get_num_workers()
    io_workers = math.ceil(
        wandb.config.adapters["io_to_total_worker_ratio"] * num_workers
    )
    process_workers = num_workers - io_workers
    chunk_size = int(wandb.config.adapters["chunk_size"])
    loader_batch_size = int(wandb.config.adapters["loader_batch_size"])

    # Initialize dataset
    dataset = DmiKuzmin2018Dataset(
        root=osp.join(DATA_ROOT, "data/torchcell/dmi_kuzmin2018"),
        io_workers=num_workers,
    )

    # Initialize adapter
    adapter = DmiKuzmin2018Adapter(
        dataset=dataset,
        process_workers=process_workers,
        io_workers=io_workers,
        chunk_size=chunk_size,
        loader_batch_size=loader_batch_size,
    )

    # Write nodes
    log.info("Writing nodes")
    start_time = time.time()
    bc.write_nodes(adapter.get_nodes())
    end_time = time.time()
    wandb.log({"write_nodes_time(s)": end_time - start_time})

    # Write edges
    log.info("Writing edges")
    start_time = time.time()
    bc.write_edges(adapter.get_edges())
    end_time = time.time()
    wandb.log({"write_edges_time(s)": end_time - start_time})

    # Finalize BioCypher
    bc.write_import_call()
    bc.write_schema_info(as_node=True)

    # Log the output file path
    relative_bash_script_path = osp.join(
        "biocypher-out", time_str, "neo4j-admin-import-call.sh"
    )
    with open("biocypher_file_name.txt", "w") as f:
        f.write(relative_bash_script_path)

    wandb.finish()


if __name__ == "__main__":
    main()

    with open("biocypher_file_name.txt", "r") as file:
        file_name = file.read().strip()
    print(file_name)
