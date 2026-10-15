# torchcell/knowledge_graphs/generate_adapter.py
# [[torchcell.knowledge_graphs.generate_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/knowledge_graphs/generate_adapter.py
# Test file: tests/torchcell/knowledge_graphs/test_generate_adapter.py
"""Generate the CSVs of ONE adapter: one array task of the per-adapter generation.

``python -m torchcell.knowledge_graphs.generate_adapter --config-name kg_uncapped
--adapter-index 65 --out-dir /var/lib/neo4j/biocypher-out/<ts>/adapters/065_X
[--include-private] [--organisms a,b] [hydra overrides]``

Index 65 is position 65 of the build order (``generation_array.adapter_order``: the
public map, then the private map with ``--include-private``, narrowed to
``--organisms``). The task instantiates that one dataset (full, never subset) and its
adapter with the config's adapter knobs, writes nodes then edges through the fast CSV
sink into ``--out-dir`` with its own telemetry under ``<out-dir>/telemetry``, writes
the per-directory import call, and LAST writes the ``DONE`` marker. A directory
without the marker is incomplete by definition, whatever files it holds; the merge
refuses it and ``generation_array missing`` prints the re-run.

The three flags are taken off ``sys.argv`` before hydra parses it, as
``create_scerevisiae_kg_small`` does for its two. ``--out-dir`` is an absolute path
inside the container; the directory must not already carry a marker.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import math
import os
import os.path as osp
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import certifi
import hydra
import wandb
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

import torchcell
from biocypher import BioCypher  # type: ignore[attr-defined]  # untyped re-export
from torchcell.build_telemetry import BuildPhase, ResourceSampler
from torchcell.data.experiment_dataset import ALL_ORGANISMS, Organism, format_organisms
from torchcell.datasets.bacteria_common import BacterialGenomeInjector
from torchcell.fast_csv import FastCsvSink, build_row_specs
from torchcell.graph import SCerevisiaeGraph
from torchcell.knowledge_graphs.dataset_adapter_map import (
    take_include_private_flag,
    take_organisms_flag,
)
from torchcell.knowledge_graphs.generation_array import (
    DONE_FILENAME,
    AdapterSlot,
    DoneMarker,
    adapter_order,
    slot_by_index,
)
from torchcell.knowledge_graphs.head_ontology import verify_head_ontology
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

log = logging.getLogger(__name__)
os.environ["SSL_CERT_FILE"] = certifi.where()

ADAPTER_INDEX_FLAG = "--adapter-index"
OUT_DIR_FLAG = "--out-dir"

#: Set in ``__main__`` before hydra parses argv; the module-level defaults are what a
#: direct ``main`` call (a test) sees.
INCLUDE_PRIVATE = False
ORGANISMS: frozenset[Organism] = ALL_ORGANISMS
ADAPTER_INDEX: int | None = None
OUT_DIR: str | None = None


def take_value_flag(argv: list[str], flag: str) -> str | None:
    """Remove ``<flag> <value>`` (or ``<flag>=<value>``) from argv; return the value.

    Absent -> None. Given twice or without a value -> ``ValueError``.
    """
    values: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == flag:
            if index + 1 == len(argv):
                raise ValueError(f"{flag} needs a value")
            values.append(argv[index + 1])
            del argv[index : index + 2]
        elif token.startswith(flag + "="):
            values.append(token[len(flag) + 1 :])
            del argv[index]
        else:
            index += 1
    if len(values) > 1:
        raise ValueError(f"{flag} given {len(values)} times")
    return values[0] if values else None


def get_num_workers() -> int:
    """The CPUs the task was given (``SLURM_CPUS_PER_TASK``, set by the array script)."""
    cpus = os.getenv("SLURM_CPUS_PER_TASK")
    if cpus is None:
        raise RuntimeError(
            "SLURM_CPUS_PER_TASK is not set; the array script passes the tier's cpus"
        )
    return int(cpus)


def resolve_slot(index: int) -> AdapterSlot:
    """The slot of ``index`` in the order of the current membership flags."""
    return slot_by_index(adapter_order(INCLUDE_PRIVATE, ORGANISMS), index)


def instantiate(slot: AdapterSlot, data_root: str, io_workers: int) -> Any:
    """Build the slot's dataset at its loader's default root with the injected genomes.

    The same injection rules as the serial build: ``genome`` and ``scerevisiae_graph``
    by parameter name for yeast, ``ecoli_genome`` / ``pputida_genome`` through the
    bacterial injector. A missing LMDB is an error here, never a skip: the task was
    asked for exactly this dataset.
    """
    from torchcell.knowledge_graphs.dataset_adapter_map import build_adapter_map

    adapter_map = build_adapter_map(
        include_private=INCLUDE_PRIVATE, organisms=ORGANISMS
    )
    dataset_class = next(c for c in adapter_map if c.__name__ == slot.dataset_class)
    adapter_class = adapter_map[dataset_class]
    params = inspect.signature(
        dataset_class.__init__  # type: ignore[misc]  # inspecting a class's __init__
    ).parameters
    root = osp.join(data_root, params["root"].default)
    if not osp.isdir(osp.join(root, "processed", "lmdb")):
        raise FileNotFoundError(f"{slot.dataset_class}: no LMDB at {root}")
    kwargs: dict[str, Any] = {"io_workers": io_workers}
    kwargs.update(BacterialGenomeInjector(data_root).genome_kwargs(dataset_class))
    if "genome" in params or "scerevisiae_graph" in params:
        genome = SCerevisiaeGenome(
            genome_root=osp.join(data_root, "data/sgd/genome"), overwrite=False
        )
        if "genome" in params:
            kwargs["genome"] = genome
        if "scerevisiae_graph" in params:
            kwargs["scerevisiae_graph"] = SCerevisiaeGraph(
                sgd_root=osp.join(data_root, "data/sgd/genome"),
                string_root=osp.join(data_root, "data/string"),
                tflink_root=osp.join(data_root, "data/tflink"),
                genome=genome,
            )
    return dataset_class(root=root, **kwargs), adapter_class


@hydra.main(version_base=None, config_path="conf", config_name="kg_uncapped")
def main(cfg: DictConfig) -> None:
    """Generate one adapter's CSVs into OUT_DIR and write its DONE marker last."""
    if ADAPTER_INDEX is None or OUT_DIR is None:
        raise ValueError(f"{ADAPTER_INDEX_FLAG} and {OUT_DIR_FLAG} are required")
    out_dir = Path(OUT_DIR)
    if (out_dir / DONE_FILENAME).exists():
        raise FileExistsError(
            f"{out_dir} already carries {DONE_FILENAME}; a complete directory is "
            "never regenerated in place (move it aside first)"
        )
    load_dotenv("/.env")
    data_root = cast(str, os.getenv("DATA_ROOT"))
    biocypher_config_path = cast(str, os.getenv("BIOCYPHER_CONFIG_PATH"))
    schema_config_path = cast(str, os.getenv("SCHEMA_CONFIG_PATH"))
    wandb_cfg = cast(
        "dict[str, Any]",
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
    )
    if wandb_cfg.get("datasets") is not None:
        raise ValueError(
            "a per-adapter generation takes the config with datasets: null"
        )
    slot = resolve_slot(ADAPTER_INDEX)
    job_id = os.environ.get("SLURM_ARRAY_JOB_ID", os.environ.get("SLURM_JOB_ID", ""))
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "")
    hashed_cfg = hashlib.sha256(
        json.dumps(wandb_cfg, sort_keys=True).encode("utf-8")
    ).hexdigest()
    tags = [str(t) for t in wandb_cfg["wandb"].get("tags") or []]
    tags += [t for t in os.environ.get("TCDB_TAGS", "").split(",") if t]
    tags.append(f"adapter:{slot.adapter_class}")
    wandb.init(
        mode=wandb_cfg["wandb"]["mode"],
        project=wandb_cfg["wandb"]["project"],
        config=wandb_cfg,
        group=f"{job_id or uuid.uuid4()}_{hashed_cfg}",
        name=f"{job_id}_{task_id}_{slot.adapter_class}" if job_id else None,
        tags=tags,
        job_type=os.environ.get("TCDB_JOB_TYPE", "generate_adapter"),
    )
    cast(Any, wandb.run).log_code(
        "/".join(osp.join(torchcell.__path__[0], __file__).split("/")[:-1])
    )
    num_workers = get_num_workers()
    io_workers = math.ceil(
        wandb_cfg["adapters"]["io_to_total_worker_ratio"] * num_workers
    )
    process_workers = num_workers - io_workers
    adapters_cfg = wandb_cfg["adapters"]
    wandb.log(
        {
            "slurm_job_id": job_id,
            "adapter_index": slot.index,
            "current_adapter": slot.adapter_class,
            "num_workers": num_workers,
            "io_workers": io_workers,
            "process_workers": process_workers,
        }
    )
    membership = (
        f"Generating adapter {slot.index} {slot.adapter_class} "
        f"({slot.dataset_class}; include_private={INCLUDE_PRIVATE}, "
        f"organisms={format_organisms(ORGANISMS)}) into {out_dir}"
    )
    print(membership)
    log.info(membership)
    verify_head_ontology(biocypher_config_path)
    bc = BioCypher(
        output_directory=str(out_dir),
        biocypher_config_path=biocypher_config_path,
        schema_config_path=schema_config_path,
    )
    sampler = ResourceSampler(out_dir / "telemetry", interval=5.0)
    sampler.start()
    build_t0 = time.time()
    BuildPhase.set("", "instantiate datasets", "setup")
    row_specs = build_row_specs(bc)
    sink = FastCsvSink(bc, row_specs)

    start_time = time.time()
    dataset, adapter_class = instantiate(slot, data_root, num_workers)
    wandb.log(
        {
            f"{slot.dataset_class}_time(s)": time.time() - start_time,
            f"{slot.dataset_class}_len": len(dataset),
        }
    )
    adapter = adapter_class(
        dataset=dataset,
        process_workers=process_workers,
        io_workers=io_workers,
        chunk_size=int(adapters_cfg["chunk_size"]),
        loader_batch_size=int(adapters_cfg["loader_batch_size"]),
    )
    adapter.inprocess_max_records = int(adapters_cfg.get("inprocess_max_records", 0))
    adapter.single_pass = bool(adapters_cfg.get("single_pass", False))
    adapter.row_specs = row_specs
    adapter.chunks_per_worker = int(adapters_cfg.get("chunks_per_worker", 2))
    adapter.single_pass_chunk_budget_bytes = int(
        float(adapters_cfg.get("single_pass_chunk_budget_mb", 48)) * 2**20
    )
    adapter.inprocess_max_bytes = int(
        float(adapters_cfg.get("inprocess_max_mb", 0)) * 2**20
    )
    adapter.completion_order = bool(adapters_cfg.get("completion_order", False))
    adapter.pool_memory_fraction = float(adapters_cfg.get("pool_memory_fraction", 0))

    log.info(f"Writing nodes for adapter: {slot.adapter_class}")
    start_time = time.time()
    n_nodes = sink.write_nodes(adapter.get_nodes())
    wandb.log(
        {
            f"{slot.adapter_class}_write_nodes_time(s)": time.time() - start_time,
            f"{slot.adapter_class}_n_nodes": n_nodes,
        }
    )
    log.info(f"Writing edges for adapter: {slot.adapter_class}")
    start_time = time.time()
    n_edges = sink.write_edges(adapter.get_edges())
    wandb.log(
        {
            f"{slot.adapter_class}_write_edges_time": time.time() - start_time,
            f"{slot.adapter_class}_n_edges": n_edges,
        }
    )
    log.info(
        "Finished iterating nodes and edges: %d nodes, %d edges across 1 adapters",
        n_nodes,
        n_edges,
    )
    sink.finish()
    generation_wall_s = time.time() - build_t0
    BuildPhase.set("", "finish", "finish")
    timings = sampler.stop()
    peak = max((t.mem_peak_gb for t in timings), default=0.0)
    wandb.log(
        {
            "total_nodes": n_nodes,
            "total_edges": n_edges,
            "generation_wall_s": generation_wall_s,
            "generation_cpu_core_s": sum(t.cpu_core_seconds for t in timings),
            "generation_mem_peak_gb": peak,
        }
    )
    log.info("Generation wall %.0f s, peak memory %.1f GiB", generation_wall_s, peak)
    bc.write_import_call()
    bc.write_schema_info(as_node=True)
    marker = DoneMarker(
        index=slot.index,
        adapter_class=slot.adapter_class,
        dataset_class=slot.dataset_class,
        dataset_len=len(dataset),
        n_nodes=n_nodes,
        n_edges=n_edges,
        generation_wall_s=generation_wall_s,
        peak_memory_gb=peak,
        torchcell_version=torchcell.__version__,
        build_commit=os.environ.get("BUILD_COMMIT", ""),
        written_at=datetime.now(UTC).isoformat(),
    )
    # The marker is the last write: everything the merge reads exists before it.
    (out_dir / DONE_FILENAME).write_text(marker.model_dump_json(indent=1) + "\n")
    print(f"DONE={out_dir / DONE_FILENAME}")
    wandb.finish()


if __name__ == "__main__":
    INCLUDE_PRIVATE = take_include_private_flag(sys.argv)
    ORGANISMS = take_organisms_flag(sys.argv)
    _index = take_value_flag(sys.argv, ADAPTER_INDEX_FLAG)
    OUT_DIR = take_value_flag(sys.argv, OUT_DIR_FLAG)
    if _index is None or OUT_DIR is None:
        raise SystemExit(f"{ADAPTER_INDEX_FLAG} and {OUT_DIR_FLAG} are required")
    ADAPTER_INDEX = int(_index)
    main()
