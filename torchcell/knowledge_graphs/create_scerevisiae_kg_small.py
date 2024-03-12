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
import math
import wandb
from omegaconf import DictConfig, OmegaConf
import json
import hashlib
import uuid
import hydra
import time
import sys
from io import StringIO

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# WARNING do not print in this file! This file is used to generate a path to a bash script and printing to stdout will break the bash script path

logging.basicConfig(level=logging.INFO, filename="biocypher_warnings.log")
logging.captureWarnings(True)


# Logic for capturing the file name
# Create a separate logger for the file name
file_name_logger = logging.getLogger("file_name_logger")
file_name_logger.setLevel(logging.INFO)

# Create a file handler for the file name logger
file_handler = logging.FileHandler("file_name.log")
file_handler.setLevel(logging.INFO)

# Create a formatter and add it to the file handler
formatter = logging.Formatter("%(message)s")
file_handler.setFormatter(formatter)

# Add the file handler to the file name logger
file_name_logger.addHandler(file_handler)


def capture_output(func):
    def wrapper(*args, **kwargs):
        # Redirect stdout to a StringIO object
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        # Call the function
        result = func(*args, **kwargs)

        # Restore stdout
        sys.stdout = old_stdout

        return result

    return wrapper

def get_num_workers():
    """Get the number of CPUs allocated by SLURM."""
    # Try to get number of CPUs allocated by SLURM
    cpus_per_task = os.getenv("SLURM_CPUS_PER_TASK")
    if cpus_per_task is not None:
        return int(cpus_per_task)
    # Fallback: Use multiprocessing to get the total number of CPUs
    return mp.cpu_count()


# @capture_output
@hydra.main(version_base=None, config_path="conf", config_name="kg_small")
def main(cfg) -> str:
    load_dotenv()
    WANDB_API_KEY = os.getenv("WANDB_API_KEY")
    log.info(f"wandb_api_key:  {WANDB_API_KEY}")
    DATA_ROOT = os.getenv("DATA_ROOT")
    BIOCYPHER_CONFIG_PATH = os.getenv("BIOCYPHER_CONFIG_PATH")
    SCHEMA_CONFIG_PATH = os.getenv("SCHEMA_CONFIG_PATH")
    BIOCYPHER_OUT_PATH = os.getenv("BIOCYPHER_OUT_PATH")

    wandb_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    slurm_job_id = os.environ.get("SLURM_JOB_ID", uuid.uuid4())
    sorted_cfg = json.dumps(wandb_cfg, sort_keys=True)
    hashed_cfg = hashlib.sha256(sorted_cfg.encode("utf-8")).hexdigest()
    group = f"{slurm_job_id}_{hashed_cfg}"
    wandb.init(
        mode=wandb_cfg["wandb"]["mode"],
        project=wandb_cfg["wandb"]["project"],
        config=wandb_cfg,
        tags=wandb_cfg["wandb"]["tags"],
        group=group,
        save_code=True,
    )

    # Use this function to get the number of workers
    num_workers = get_num_workers()
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log.info(f"Number of workers: {num_workers}")
    bc = BioCypher(
        output_directory=osp.join(DATA_ROOT, BIOCYPHER_OUT_PATH, time_str),
        biocypher_config_path=BIOCYPHER_CONFIG_PATH,
        schema_config_path=SCHEMA_CONFIG_PATH,
    )
    # Partition workers
    num_workers = get_num_workers()
    io_workers = math.ceil(
        wandb.config.adapters["io_process_worker_ratio"] * num_workers
    )
    compute_workers = num_workers - io_workers
    chunk_size = int(wandb.config.adapters["chunk_size"])
    loader_batch_size = int(wandb.config.adapters["loader_batch_size"])

    wandb.log(
        {
            "num_workers": num_workers,
            "io_workers": io_workers,
            "compute_workers": compute_workers,
        }
    )

    # Define dataset configurations
    dataset_configs = [
        # {
        #     "class": SmfCostanzo2016Dataset,
        #     "path": osp.join(DATA_ROOT, "data/torchcell/smf_costanzo2016"),
        #     "kwargs": {},
        # },
        {
            "class": SmfKuzmin2018Dataset,
            "path": osp.join(DATA_ROOT, "data/torchcell/smf_kuzmin2018"),
            "kwargs": {},
        },
        # {
        #     "class": DmfKuzmin2018Dataset,
        #     "path": osp.join(DATA_ROOT, "data/torchcell/dmf_kuzmin2018"),
        #     "kwargs": {},
        # },
        # {
        #     "class": TmfKuzmin2018Dataset,
        #     "path": osp.join(DATA_ROOT, "data/torchcell/tmf_kuzmin2018"),
        #     "kwargs": {},
        # },
        # {
        #     "class": DmfCostanzo2016Dataset,
        #     "path": osp.join(DATA_ROOT, "data/torchcell/dmf_costanzo2016_1e6"),
        #     "kwargs": {
        #         "subset_n": int(1e6),
        #         "num_workers": num_workers,
        #         "batch_size": int(1e3),
        #     },
        # },
    ]

    # Instantiate datasets
    datasets = []
    for config in dataset_configs:
        dataset_class = config["class"]
        dataset_path = config["path"]
        dataset_kwargs = config["kwargs"]
        dataset_name = dataset_class.__name__

        log.info(f"Instantiating dataset: {dataset_name}")
        start_time = time.time()
        dataset = dataset_class(root=dataset_path, **dataset_kwargs)
        end_time = time.time()
        instantiation_time = end_time - start_time
        wandb.log({f"{dataset_name}_time(s)": instantiation_time})
        datasets.append(dataset)

    # Define dataset-adapter mapping
    dataset_adapter_map = {
        DmfCostanzo2016Dataset: DmfCostanzo2016Adapter,
        SmfCostanzo2016Dataset: SmfCostanzo2016Adapter,
        SmfKuzmin2018Dataset: SmfKuzmin2018Adapter,
        DmfKuzmin2018Dataset: DmfKuzmin2018Adapter,
        TmfKuzmin2018Dataset: TmfKuzmin2018Adapter,
    }

    # Instantiate adapters based on the dataset-adapter mapping
    adapters = [
        dataset_adapter_map[type(dataset)](
            dataset=dataset,
            compute_workers=compute_workers,
            io_workers=io_workers,
            chunk_size=chunk_size,
            loader_batch_size=loader_batch_size,
        )
        for dataset in datasets
    ]

    for i, adapter in enumerate(adapters):
        adapter_name = type(adapter).__name__
        log.info(f"Writing nodes for adapter: {adapter_name}")
        start_time = time.time()
        bc.write_nodes(adapter.get_nodes())
        end_time = time.time()
        write_nodes_time = end_time - start_time
        wandb.log({f"{adapter_name}_write_nodes_time(s)": write_nodes_time})

        log.info(f"Writing edges for adapter: {adapter_name}")
        start_time = time.time()
        bc.write_edges(adapter.get_edges())
        end_time = time.time()
        write_edges_time = end_time - start_time
        wandb.log({f"{adapter_name}_write_edges_time": write_edges_time})

    log.info("Finished iterating nodes and edges")
    # Write admin import statement and schema information (for biochatter)
    bc.write_import_call()
    bc.write_schema_info(as_node=True)

    relative_bash_script_path = osp.join(
        "biocypher-out", time_str, "neo4j-admin-import-call.sh"
    )
    file_name_logger.info(relative_bash_script_path)
    # return relative_bash_script_path


if __name__ == "__main__":
    main()
    
    # Read the logged file name from the file
    with open("file_name.log", "r") as file:
        file_name = file.read().strip()
    
    print(file_name)
    