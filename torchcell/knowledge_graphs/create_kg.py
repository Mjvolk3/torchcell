from biocypher import BioCypher
import logging
from dotenv import load_dotenv
import os
import os.path as osp
import multiprocessing as mp
from datetime import datetime
import math
import wandb
from omegaconf import OmegaConf
import json
import hashlib
import uuid
import hydra
import time
import certifi
from torchcell.datasets import dataset_registry
from torchcell.knowledge_graphs import dataset_adapter_map
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.graph import SCerevisiaeGraph

import inspect

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, filename="biocypher_warnings.log")
logging.captureWarnings(True)

os.environ["SSL_CERT_FILE"] = certifi.where()

load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
BIOCYPHER_CONFIG_PATH = os.getenv("BIOCYPHER_CONFIG_PATH")
SCHEMA_CONFIG_PATH = os.getenv("SCHEMA_CONFIG_PATH")
BIOCYPHER_OUT_PATH = os.getenv("BIOCYPHER_OUT_PATH")


def get_num_workers() -> int:
    """Get the number of CPUs allocated by SLURM."""
    cpus_per_task = os.getenv("SLURM_CPUS_PER_TASK")
    if cpus_per_task is not None:
        return int(cpus_per_task)
    return mp.cpu_count()


@hydra.main(version_base=None, config_path="conf", config_name="gene_essentiality_sgd")
def main(cfg) -> str:
    # wandb configuration
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    slurm_job_id = os.environ.get("SLURM_JOB_ID", uuid.uuid4())
    sorted_cfg = json.dumps(wandb_cfg, sort_keys=True)
    hashed_cfg = hashlib.sha256(sorted_cfg.encode("utf-8")).hexdigest()
    group = f"{slurm_job_id}_{hashed_cfg}"
    wandb.init(
        mode=wandb_cfg["wandb"]["mode"],
        project=wandb_cfg["wandb"]["project"],
        config=wandb_cfg,
        group=group,
        save_code=True,
    )
    wandb.log({"slurm_job_id": str(slurm_job_id)})
    # Use this function to get the number of workers
    num_workers = get_num_workers()
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log.info(f"Number of workers: {num_workers}")
    bc = BioCypher(
        output_directory=osp.join(DATA_ROOT, BIOCYPHER_OUT_PATH, time_str),
        biocypher_config_path=BIOCYPHER_CONFIG_PATH,
        schema_config_path=SCHEMA_CONFIG_PATH,
    )
    wandb.log({"biocypher-out": bc._output_directory.split("/")[-1]})
    # Partition workers
    io_workers = math.ceil(
        wandb.config.adapters["io_to_total_worker_ratio"] * num_workers
    )
    process_workers = num_workers - io_workers
    chunk_size = int(wandb.config.adapters["chunk_size"])
    loader_batch_size = int(wandb.config.adapters["loader_batch_size"])

    wandb.log(
        {
            "num_workers": num_workers,
            "io_workers": io_workers,
            "process_workers": process_workers,
        }
    )

    # Define dataset configurations
    dataset_configs = []
    for dataset in wandb.config.datasets:
        # We need workers according to system but other kwargs
        # come from yaml, like subsetting, etc.
        if wandb.config.datasets[dataset]["kwargs"] is not None:
            kwargs = {
                **wandb.config.datasets[dataset]["kwargs"],
                **{"io_workers": num_workers},
            }
        else:
            kwargs = {"io_workers": num_workers}

        dataset_class = dataset_registry[dataset]

        # Start special cases
        # Handle special cases...
        # Concerned about this as it makes things much less general.
        if "scerevisiae_graph" in inspect.signature(dataset_class.__init__).parameters:
            genome = SCerevisiaeGenome(
                data_root=osp.join(DATA_ROOT, "data/sgd/genome"), overwrite=True
            )
            graph = SCerevisiaeGraph(
                data_root=osp.join(DATA_ROOT, "data/sgd/genome"), genome=genome
            )
            kwargs["scerevisiae_graph"] = graph
        if "genome" in inspect.signature(dataset_class.__init__).parameters:
            genome = SCerevisiaeGenome(
                data_root=osp.join(DATA_ROOT, "data/sgd/genome"), overwrite=True
            )
            kwargs["genome"] = genome

        dataset_config = {
            "class": dataset_registry[dataset],
            "path": osp.join(DATA_ROOT, wandb.config.datasets[dataset]["path"]),
            "kwargs": kwargs,
        }
        dataset_configs.append(dataset_config)  

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

    # Instantiate adapters based on the dataset-adapter mapping
    adapters = [
        dataset_adapter_map[type(dataset)](
            dataset=dataset,
            process_workers=process_workers,
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

    with open("biocypher_file_name.txt", "w") as f:
        f.write(relative_bash_script_path)
    wandb.finish()


if __name__ == "__main__":
    main()

    # Read the logged file name from the file
    with open("biocypher_file_name.txt", "r") as file:
        file_name = file.read().strip()

    print(file_name)
