# experiments/tcdb-002-build-speed/scripts/fork_cost_probe.py
# [[experiments.tcdb-002-build-speed.scripts.fork_cost_probe]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/tcdb-002-build-speed/scripts/fork_cost_probe
"""Cost of forking a chunk-worker pool from a large parent: fork vs forkserver.

    nice -n 10 taskset -c 120-127 python \
        experiments/tcdb-002-build-speed/scripts/fork_cost_probe.py [--dataset NAME]

Mirrors one pool group of ``CellAdapter.get_data_by_type``: gc.collect() + gc.freeze()
in the parent, a ProcessPoolExecutor of ``WORKERS`` workers, and one real single-pass
node chunk (``adapter._all_chunked``) per worker, submitted exactly as the build does.

The parent carries a ballast of 64-char hex strings in a set (the shape of the CSV
sink's sha256 dedup id sets), grown in place to 0, 4, 8 and 16 GB of RSS. At 16 GB a
second, "holey" ballast is made by discarding every other element of the set, which
leaves the RSS in place but scatters free slots through every pymalloc pool, the way a
long-running parent that has freed chunk results looks. For each ballast and start
method it records:

- ``freeze_s``: gc.collect() + gc.freeze() in the parent;
- ``fork_wall_s``: pool creation until all workers have run a trivial task (a barrier
  of ``WORKERS`` parties guarantees every worker process exists and ran one);
- ``worker_private_dirty_*``: Private_Dirty of each worker from
  /proc/<pid>/smaps_rollup, read by the parent while the workers are alive, after the
  trivial task and again after each worker ran one chunk and reported back;
- ``worker_page_tables_gb_mean``: VmPTE of each worker at the same moment; page
  tables are not in Private_Dirty but are charged to the container's memory;
- ``chunk_s``: wall from chunk submission to the last chunk result;
- ``teardown_s``: executor shutdown.

The forkserver is started at the top, before any ballast, with the adapter modules
preloaded (``set_forkserver_preload``); its first pool is recorded as
``forkserver_cold`` (server exec + preload imports + worker forks).
"""

import argparse
import gc
import inspect
import itertools
import multiprocessing
import os
import os.path as osp
import resource
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from multiprocessing.context import BaseContext
from multiprocessing.synchronize import Barrier
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel

os.environ["WANDB_MODE"] = "disabled"

from torchcell.adapters.cell_adapter import SINGLE_PASS_NODES  # noqa: E402
from torchcell.knowledge_graphs.dataset_adapter_map import (  # noqa: E402
    dataset_adapter_map,
)

WORKERS = 8
BALLAST_GB = [0.0, 4.0, 8.0, 16.0]
CHUNK_RECORDS = 1500
BARRIER_TIMEOUT_S = 900.0
RESULTS = osp.join(
    osp.dirname(osp.dirname(osp.abspath(__file__))), "results", "fork_cost_probe.csv"
)
PRELOAD = [
    "__main__",
    "torchcell.adapters.cell_adapter",
    "torchcell.knowledge_graphs.dataset_adapter_map",
]

SERVER_FREEZE_MODULE = "fork_cost_probe_server_freeze"
"""Preloaded last when --server-freeze 1: gc.collect() + gc.freeze() in the server.

A module of its own because forkserver children do not reuse the server's __main__:
each re-runs this script as __mp_main__ (spawn.prepare), so a freeze guarded in this
file would also run a full collection in every worker. In a smoke run that cost
0.19 GB of Private_Dirty and about 0.5 s per pool. A preloaded module is imported by
the server alone and inherited, not re-executed, by the workers.
"""
_BARRIER: Barrier | None = None


class PoolMeasure(BaseModel):
    """What one pool group costs: freeze, fork, per-worker memory, chunk, teardown."""

    fork_wall_s: float
    freeze_s: float
    worker_private_dirty_trivial_gb_mean: float
    worker_private_dirty_gb_mean: float
    worker_private_dirty_gb_max: float
    worker_page_tables_gb_mean: float
    chunk_s: float
    chunk_task_s_mean: float
    chunk_rows: int
    teardown_s: float


class ProbeRow(PoolMeasure):
    """One (ballast, start method) measurement."""

    ballast_kind: str
    ballast_gb: float
    start_method: str
    workers: int
    parent_rss_gb: float


def rss_gb() -> float:
    """Resident set size of this process in GB."""
    with open("/proc/self/statm") as fh:
        return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e9


def private_dirty_gb(pid: int) -> float:
    """Private_Dirty of a process from its smaps_rollup, in GB."""
    with open(f"/proc/{pid}/smaps_rollup") as fh:
        line = next(ln for ln in fh if ln.startswith("Private_Dirty:"))
    return int(line.split()[1]) * 1024 / 1e9


def page_tables_gb(pid: int) -> float:
    """VmPTE (kernel page-table memory, charged to the cgroup) of a process, in GB."""
    with open(f"/proc/{pid}/status") as fh:
        line = next(ln for ln in fh if ln.startswith("VmPTE:"))
    return int(line.split()[1]) * 1024 / 1e9


def _init_worker(barrier: Barrier, loader_fork: bool) -> None:
    """Pool initializer: keep the barrier; optionally make the loader fork.

    A forkserver child inherits the forkserver start method as its default, so the
    chunk's CpuExperimentLoaderMultiprocessing (plain ``multiprocessing.Process``)
    would start its children through the forkserver too. ``loader_fork`` restores
    fork for them, forking from the small worker instead.
    """
    global _BARRIER
    _BARRIER = barrier
    if loader_fork:
        multiprocessing.set_start_method("fork", force=True)


def _sync() -> None:
    """Block until every worker has reached this point (one task per worker)."""
    assert _BARRIER is not None
    _BARRIER.wait(timeout=BARRIER_TIMEOUT_S)


def trivial_task() -> int:
    """Rendezvous with the other workers and report this worker's pid."""
    _sync()
    return os.getpid()


def chunk_task(
    fn: Callable[[Any, str], list[Any]], chunk: Any, method_name: str
) -> tuple[int, int, float]:
    """Run one real chunk (one per worker) and report pid, rows and seconds."""
    _sync()
    start = time.perf_counter()
    rows = len(fn(chunk, method_name))
    return os.getpid(), rows, time.perf_counter() - start


def grow_ballast(ballast: set[str], counter: Iterator[int], gb: float) -> None:
    """Add sha256 hex strings until the RSS is ``gb`` above the pre-ballast base."""
    batch = 2_000_000
    while rss_gb() - BASE_RSS_GB < gb:
        ballast.update(
            sha256(next(counter).to_bytes(8, "little")).hexdigest()
            for _ in range(batch)
        )


def measure(
    ctx: BaseContext,
    label: str,
    fn: Callable[[Any, str], list[Any]] | None,
    chunk: Any,
    loader_fork: bool,
) -> PoolMeasure:
    """Freeze, fork a pool, run trivial then (optionally) chunk tasks, tear down."""
    t = time.perf_counter()
    gc.collect()
    gc.freeze()
    freeze_s = time.perf_counter() - t

    barrier = ctx.Barrier(WORKERS)
    t = time.perf_counter()
    executor = ProcessPoolExecutor(
        max_workers=WORKERS,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(barrier, loader_fork),
    )
    pids = [f.result() for f in [executor.submit(trivial_task) for _ in range(WORKERS)]]
    fork_wall_s = time.perf_counter() - t
    assert len(set(pids)) == WORKERS, pids
    trivial_pd = [private_dirty_gb(p) for p in pids]
    pte = [page_tables_gb(p) for p in pids]

    chunk_pd = [float("nan")]
    chunk_s = task_s = float("nan")
    rows = 0
    if fn is not None:
        t = time.perf_counter()
        futures = [
            executor.submit(chunk_task, fn, chunk, SINGLE_PASS_NODES)
            for _ in range(WORKERS)
        ]
        results = [f.result() for f in futures]
        chunk_s = time.perf_counter() - t
        chunk_pids = [r[0] for r in results]
        assert sorted(chunk_pids) == sorted(pids), (chunk_pids, pids)
        chunk_pd = [private_dirty_gb(p) for p in chunk_pids]
        pte = [page_tables_gb(p) for p in chunk_pids]
        task_s = sum(r[2] for r in results) / WORKERS
        rows = results[0][1]

    t = time.perf_counter()
    executor.shutdown(wait=True)
    teardown_s = time.perf_counter() - t
    out = PoolMeasure(
        fork_wall_s=fork_wall_s,
        freeze_s=freeze_s,
        worker_private_dirty_trivial_gb_mean=sum(trivial_pd) / WORKERS,
        worker_private_dirty_gb_mean=sum(chunk_pd) / len(chunk_pd),
        worker_private_dirty_gb_max=max(chunk_pd),
        worker_page_tables_gb_mean=sum(pte) / WORKERS,
        chunk_s=chunk_s,
        chunk_task_s_mean=task_s,
        chunk_rows=rows,
        teardown_s=teardown_s,
    )
    print(
        f"[{label}] fork {fork_wall_s:.3f}s freeze {freeze_s:.3f}s "
        f"pd_trivial {out.worker_private_dirty_trivial_gb_mean:.3f} "
        f"pd_chunk mean {out.worker_private_dirty_gb_mean:.3f} "
        f"max {out.worker_private_dirty_gb_max:.3f} GB "
        f"pte {out.worker_page_tables_gb_mean:.4f} GB chunk {chunk_s:.1f}s "
        f"teardown {teardown_s:.3f}s rss {rss_gb():.2f} GB",
        flush=True,
    )
    return out


def build_chunk(name: str) -> tuple[Callable[[Any, str], list[Any]], Any]:
    """Build the adapter and one CHUNK_RECORDS chunk view, as get_data_by_type does."""
    dataset_class, adapter_class = next(
        (d, a) for d, a in dataset_adapter_map.items() if d.__name__ == name
    )
    root = osp.join(
        os.environ["DATA_ROOT"],
        inspect.signature(dataset_class).parameters["root"].default,
    )
    dataset = dataset_class(root=root)
    adapter = adapter_class(
        dataset=dataset,
        process_workers=WORKERS,
        io_workers=1,
        chunk_size=CHUNK_RECORDS,
        loader_batch_size=500,
    )
    adapter.single_pass = True
    configured = [i["method_name"] for i in adapter.config.cell_adapter.node_methods]
    adapter._single_pass_methods = [
        (n, m)
        for n, m in adapter.node_methods
        if n in configured and not m.__name__.startswith("_get_")
    ]
    chunk = dataset[0 : min(CHUNK_RECORDS, len(dataset))]
    dataset.close_lmdb()
    print(
        f"{name}: {len(dataset)} records, chunk {len(chunk)}, "
        f"{len(adapter._single_pass_methods)} folded methods",
        flush=True,
    )
    return adapter._all_chunked, chunk


BASE_RSS_GB = 0.0


def main() -> None:
    """Run the fork/forkserver sweep over ballast sizes and write the CSV."""
    global BASE_RSS_GB
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="SmfKuzmin2018Dataset")
    parser.add_argument("--ballast-gb", type=float, nargs="+", default=BALLAST_GB)
    parser.add_argument("--out", default=RESULTS)
    parser.add_argument("--server-freeze", type=int, choices=(0, 1), default=1)
    args = parser.parse_args()
    load_dotenv(osp.join(osp.dirname(osp.abspath(__file__)), "../../../.env"))

    fork_ctx = multiprocessing.get_context("fork")
    fs_ctx = multiprocessing.get_context("forkserver")
    fs_ctx.set_forkserver_preload(
        PRELOAD + [SERVER_FREEZE_MODULE] if args.server_freeze else PRELOAD
    )

    rows: list[ProbeRow] = []
    methods: list[tuple[str, BaseContext, bool]] = [
        ("fork", fork_ctx, False),
        ("forkserver", fs_ctx, False),
        ("forkserver_loaderfork", fs_ctx, True),
    ]

    def record(kind: str, gb: float, method: str, m: PoolMeasure) -> None:
        rows.append(
            ProbeRow(
                ballast_kind=kind,
                ballast_gb=round(gb, 2),
                start_method=method,
                workers=WORKERS,
                parent_rss_gb=round(rss_gb(), 2),
                **m.model_dump(),
            )
        )

    # Cold forkserver first, before the ballast and before the adapter exists.
    record(
        "none",
        0.0,
        "forkserver_cold",
        measure(fs_ctx, "forkserver_cold", None, None, False),
    )

    fn, chunk = build_chunk(args.dataset)
    BASE_RSS_GB = rss_gb()
    print(f"base RSS {BASE_RSS_GB:.2f} GB", flush=True)
    ballast: set[str] = set()
    counter = itertools.count()
    for gb in args.ballast_gb:
        t = time.perf_counter()
        grow_ballast(ballast, counter, gb)
        actual = rss_gb() - BASE_RSS_GB
        print(
            f"ballast {actual:.2f} GB ({len(ballast):,} strings) "
            f"built in {time.perf_counter() - t:.0f}s",
            flush=True,
        )
        for method, ctx, loader_fork in methods:
            record(
                "dense",
                actual,
                method,
                measure(ctx, f"dense {gb} {method}", fn, chunk, loader_fork),
            )

    victims = list(itertools.islice(ballast, 0, None, 2))
    ballast.difference_update(victims)
    del victims
    gc.collect()
    actual = rss_gb() - BASE_RSS_GB
    print(f"holey ballast {actual:.2f} GB ({len(ballast):,} strings)", flush=True)
    for method, ctx, loader_fork in methods:
        record(
            "holey",
            actual,
            method,
            measure(ctx, f"holey {method}", fn, chunk, loader_fork),
        )

    lead = [
        "ballast_kind",
        "ballast_gb",
        "start_method",
        "workers",
        "fork_wall_s",
        "freeze_s",
        "worker_private_dirty_gb_mean",
        "worker_private_dirty_gb_max",
        "chunk_s",
    ]
    frame = pd.DataFrame([r.model_dump() for r in rows])
    frame = frame[lead + [c for c in frame.columns if c not in lead]]
    frame.to_csv(args.out, index=False, float_format="%.4f")
    print(frame.to_string(index=False))
    print(
        f"parent peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB"
    )
    print(args.out)


if __name__ == "__main__":
    main()
