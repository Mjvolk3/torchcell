# torchcell/knowledge_graphs/create_scerevisiae_kg_small
# [[torchcell.knowledge_graphs.create_scerevisiae_kg_small]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/knowledge_graphs/create_scerevisiae_kg_small
# Test file: tests/torchcell/knowledge_graphs/test_create_scerevisiae_kg_small.py
"""Build a small S. cerevisiae BioCypher knowledge graph from Costanzo/Kuzmin data."""

import hashlib
import inspect
import json
import logging
import math
import os
import os.path as osp
import time
import uuid
from datetime import datetime
from typing import Any, cast

import certifi
import hydra
import wandb
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

import torchcell
from biocypher import BioCypher  # type: ignore[attr-defined]  # untyped re-export
from torchcell.graph import SCerevisiaeGraph
from torchcell.knowledge_graphs.dataset_adapter_map import dataset_adapter_map
from torchcell.knowledge_graphs.subset import subset_dataset
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, filename="biocypher_warnings.log")
logging.captureWarnings(True)

os.environ["SSL_CERT_FILE"] = certifi.where()


def get_num_workers() -> int:
    """Get the number of CPUs allocated by SLURM."""
    # Try to get number of CPUs allocated by SLURM
    cpus_per_task = os.getenv("SLURM_CPUS_PER_TASK")
    if cpus_per_task is not None:
        return int(cpus_per_task)
    # Fallback: Use multiprocessing to get the total number of CPUs
    # return mp.cpu_count()
    return 10


def _count_while_writing(write_fn: Any, generator: Any) -> int:
    """Stream a node/edge generator into BioCypher while counting items for wandb.

    BioCypher's write_nodes/write_edges consume a generator without returning a
    count, so we wrap the generator to tally items as they pass through -- giving
    per-adapter + running-total node/edge counts to monitor build scale/progress.
    """
    count = 0

    def counting() -> Any:
        nonlocal count
        for item in generator:
            count += 1
            yield item

    write_fn(counting())
    return count


@hydra.main(version_base=None, config_path="conf", config_name="kg_small")
def main(cfg: DictConfig) -> None:
    """Run the BioCypher build for the small S. cerevisiae knowledge graph."""
    print("printing path info")
    print(os.getcwd())
    load_dotenv("/.env")
    # These env vars are required entry-point preconditions for the build script;
    # cast documents the non-None contract without altering runtime behavior.
    DATA_ROOT = cast(str, os.getenv("DATA_ROOT"))
    BIOCYPHER_CONFIG_PATH = cast(str, os.getenv("BIOCYPHER_CONFIG_PATH"))
    SCHEMA_CONFIG_PATH = cast(str, os.getenv("SCHEMA_CONFIG_PATH"))
    BIOCYPHER_OUT_PATH = cast(str, os.getenv("BIOCYPHER_OUT_PATH"))
    print("---------")
    print(DATA_ROOT)
    print(BIOCYPHER_CONFIG_PATH)
    print(SCHEMA_CONFIG_PATH)
    print(BIOCYPHER_OUT_PATH)
    print("---------")

    # wandb configuration
    wandb_cfg = cast(
        "dict[str, Any]",
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
    )
    slurm_job_id = os.environ.get("SLURM_JOB_ID", uuid.uuid4())
    sorted_cfg = json.dumps(wandb_cfg, sort_keys=True)
    hashed_cfg = hashlib.sha256(sorted_cfg.encode("utf-8")).hexdigest()
    group = f"{slurm_job_id}_{hashed_cfg}"
    wandb.init(
        mode=wandb_cfg["wandb"]["mode"],
        project=wandb_cfg["wandb"]["project"],
        config=wandb_cfg,
        group=group,
        # save_code=True,
    )
    # save_code = True only works for git repositories, so we log the kg dir.
    cast(Any, wandb.run).log_code(
        "/".join(osp.join(torchcell.__path__[0], __file__).split("/")[:-1])
    )
    wandb.log({"slurm_job_id": str(slurm_job_id)})
    # Use this function to get the number of workers
    num_workers = get_num_workers()
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log.info(f"Number of workers: {num_workers}")
    print("=========")
    print(f"DATA_ROOT: {DATA_ROOT}")
    # print types
    print(f"DATA_ROOT type: {type(DATA_ROOT)}")
    print(f"BIOCYPHER_CONFIG_PATH: {BIOCYPHER_CONFIG_PATH}")
    # print types
    print(f"BIOCYPHER_CONFIG_PATH type: {type(BIOCYPHER_CONFIG_PATH)}")
    print(f"time_str: {time_str}")
    # print types
    print(f"time_str type: {type(time_str)}")
    print("=========")
    output_directory = osp.join(DATA_ROOT, BIOCYPHER_OUT_PATH, time_str)
    print(output_directory)
    print("=========")
    bc = BioCypher(
        output_directory=output_directory,
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

    # Build a shared genome + graph once; datasets that declare them get them injected.
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"), overwrite=False
    )
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )

    # Subsetting: cap every dataset to a small random subset for a fast TEST build.
    # `subset.size: null` => full dataset -- the real KG build runs this same script
    # with subsetting flipped off.
    subset_size_cfg = wandb.config.subset["size"]
    subset_size = None if subset_size_cfg is None else int(subset_size_cfg)
    subset_seed = int(wandb.config.subset["seed"])
    # Per-dataset caps OVERRIDE the global size for named datasets (keyed by class
    # name), so the full build can cap only the giant Costanzo double-mutant sets
    # (20.7M records each) while every other dataset builds in full.
    _per = cfg.subset.get("per_dataset", {})
    _per_map = (
        cast("dict[str, Any]", OmegaConf.to_container(_per, resolve=True))
        if _per
        else {}
    )
    subset_per_dataset = {
        name: (None if v is None else int(v)) for name, v in _per_map.items()
    }

    # Build EVERY dataset in the registry (subset), each paired with its adapter.
    # A dataset's root is its loader's default; genome/graph are injected when the
    # loader declares them. Datasets whose dev-tree LMDB is absent are skipped LOUDLY
    # so a test build reports coverage instead of aborting on the first missing one.
    adapters = []
    skipped: list[str] = []
    for dataset_class, adapter_class in dataset_adapter_map.items():
        params = inspect.signature(
            dataset_class.__init__  # type: ignore[misc]  # inspecting a class's __init__
        ).parameters
        root = osp.join(DATA_ROOT, params["root"].default)
        if not osp.isdir(osp.join(root, "processed", "lmdb")):
            log.warning("SKIP %s: no LMDB at %s", dataset_class.__name__, root)
            skipped.append(dataset_class.__name__)
            continue
        kwargs: dict[str, Any] = {"io_workers": num_workers}
        if "genome" in params:
            kwargs["genome"] = genome
        if "scerevisiae_graph" in params:
            kwargs["scerevisiae_graph"] = graph
        log.info("Instantiating dataset: %s", dataset_class.__name__)
        start_time = time.time()
        ds_size = subset_per_dataset.get(dataset_class.__name__, subset_size)
        dataset = subset_dataset(
            dataset_class(root=root, **kwargs), ds_size, subset_seed
        )
        wandb.log({f"{dataset_class.__name__}_time(s)": time.time() - start_time})
        wandb.log({f"{dataset_class.__name__}_len": len(dataset)})
        adapters.append(
            adapter_class(
                dataset=dataset,
                process_workers=process_workers,
                io_workers=io_workers,
                chunk_size=chunk_size,
                loader_batch_size=loader_batch_size,
            )
        )
    log.info(
        "Built %d adapters; skipped %d with no LMDB: %s",
        len(adapters),
        len(skipped),
        skipped,
    )
    wandb.log({"n_adapters": len(adapters), "skipped_datasets": skipped})

    total_nodes = 0
    total_edges = 0
    for adapter in adapters:
        adapter_name = type(adapter).__name__
        log.info(f"Writing nodes for adapter: {adapter_name}")
        start_time = time.time()
        n_nodes = _count_while_writing(bc.write_nodes, adapter.get_nodes())
        total_nodes += n_nodes
        wandb.log(
            {
                f"{adapter_name}_write_nodes_time(s)": time.time() - start_time,
                f"{adapter_name}_n_nodes": n_nodes,
                "total_nodes": total_nodes,
            }
        )

        log.info(f"Writing edges for adapter: {adapter_name}")
        start_time = time.time()
        n_edges = _count_while_writing(bc.write_edges, adapter.get_edges())
        total_edges += n_edges
        wandb.log(
            {
                f"{adapter_name}_write_edges_time": time.time() - start_time,
                f"{adapter_name}_n_edges": n_edges,
                "total_edges": total_edges,
            }
        )

    log.info(
        "Finished iterating nodes and edges: %d nodes, %d edges across %d adapters",
        total_nodes,
        total_edges,
        len(adapters),
    )
    wandb.log({"total_nodes": total_nodes, "total_edges": total_edges})
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
    with open("biocypher_file_name.txt") as file:
        file_name = file.read().strip()

    print(file_name)
