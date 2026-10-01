# experiments/tcdb-002-build-speed/scripts/pool_worker_retention.py
# [[experiments.tcdb-002-build-speed.scripts.pool_worker_retention]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/tcdb-002-build-speed/scripts/pool_worker_retention
"""Memory a REAL ProcessPoolExecutor chunk worker retains per chunk, and what frees it.

    PYTHONPATH=. WANDB_MODE=disabled nice -n 10 python \
        experiments/tcdb-002-build-speed/scripts/pool_worker_retention.py \
        --variant none --bc-out <scratch dir>

    # variant env: glibc reads its tunables at process start, so set them on launch
    MALLOC_ARENA_MAX=2 MALLOC_TRIM_THRESHOLD_=65536 python ... --variant env

Unlike ``worker_heap_ratchet.py`` (one hand-fed forked child), this runs the build's
own path: the adapter is built as ``create_scerevisiae_kg_small.py`` builds it (dev
tree dataset, row specs from a BioCypher instance, single pass, byte budget,
completion order) and the parent consumes ``adapter.get_nodes()``, so
``get_data_by_type`` forks the pool, windows the submissions and recycles the pool
every ``process_workers x chunks_per_worker`` chunks exactly as in the build.

The only change is that ``get_data_by_type`` is handed a ``TaggedTask`` instead of the
bound ``_all_chunked``; it pickles the same adapter and calls the same decorated
method, and around it records, INSIDE the worker at the start of every task (so the
previous task's result and call item are already released): smaps_rollup (Rss,
Private_Dirty, Anonymous), glibc ``mallinfo2`` (arena, hblkhd, uordblks, fordblks),
pymalloc's ``sys._debugmallocstats`` (arenas, bytes in allocated / available blocks)
and the GC freeze count; then applies the variant's remedy and records the same again.
The remedy therefore acts on the state a finished chunk leaves behind, which is what
"at the end of each task" means for a pool worker. After the first ``--real-chunks``
tasks (over all workers) the rest of the group is measurement-only tasks, so each
worker's state after its last chunk is read the same way.

A 0.5 s sampler thread in the parent reads smaps_rollup of the parent, every pool
worker and every loader child (keyed by pid); a chunk's ``loader_children_gb`` is the
peak summed Private_Dirty of that worker's loader children within the chunk.

Variants (one per run): ``none`` production; ``collect`` gc.collect(); ``unfreeze``
gc.unfreeze() + gc.collect(); ``trim`` malloc_trim(0); ``env`` MALLOC_ARENA_MAX=2 and
MALLOC_TRIM_THRESHOLD_=65536 set at launch; ``explicit`` runs the data_chunker body
itself with ``del data_loader`` and ``data_chunk.close_lmdb()`` at the end;
``trace`` is ``none`` with tracemalloc on, printing the allocation sites whose live
bytes grew between the end of chunk 1 and the end of each later chunk;
``unfreeze_trim`` is ``unfreeze`` followed by ``malloc_trim(0)``; ``slim`` clears the
dataset's cached experiment reference index in the parent before the single pass, so
no task ships it (a candidate fix, not a remedy).

Writes ``results/pool_worker_retention.csv`` (rows of the run's variant replaced) and
the raw sampler series to ``--series-dir``.
"""

import argparse
import csv
import ctypes
import gc
import inspect
import multiprocessing as mp
import os
import os.path as osp
import pickle
import signal
import sys
import tempfile
import threading
import time
import tracemalloc
from collections.abc import Callable, Generator, Iterator, Sequence
from multiprocessing.sharedctypes import Synchronized
from typing import Any, cast

os.environ["WANDB_MODE"] = "disabled"
import wandb  # noqa: E402

wandb.init(mode="disabled")
from dotenv import load_dotenv  # noqa: E402

from biocypher import BioCypher  # type: ignore[attr-defined]  # noqa: E402
from torchcell.adapters.cell_adapter import (  # noqa: E402
    SINGLE_PASS_EDGES,
    SINGLE_PASS_NODES,
    CellAdapter,
)
from torchcell.fast_csv import RenderedChunk, build_row_specs  # noqa: E402
from torchcell.knowledge_graphs.dataset_adapter_map import (  # noqa: E402
    dataset_adapter_map,
)
from torchcell.knowledge_graphs.subset import subset_dataset  # noqa: E402
from torchcell.loader.cpu_experiment_loader import (  # noqa: E402
    CpuExperimentLoaderMultiprocessing,
)

HERE = osp.dirname(osp.abspath(__file__))
load_dotenv(osp.join(HERE, "..", "..", "..", ".env"))

GB = 1e9
KB = 1024
VARIANTS = (
    "none",
    "collect",
    "unfreeze",
    "trim",
    "env",
    "explicit",
    "trace",
    "unfreeze_trim",
    "slim",
)
MALLOC_ENV = {"MALLOC_ARENA_MAX": "2", "MALLOC_TRIM_THRESHOLD_": "65536"}
FIELDS = [
    "variant",
    "worker",
    "pid",
    "chunk_index",
    "records",
    "rss_gb",
    "private_dirty_gb",
    "anon_gb",
    "loader_children_gb",
    "seconds",
    "rss_after_remedy_gb",
    "anon_after_remedy_gb",
    "private_dirty_after_remedy_gb",
    "remedy_seconds",
    "glibc_arena_gb",
    "glibc_hblkhd_gb",
    "glibc_inuse_gb",
    "glibc_free_gb",
    "glibc_free_after_remedy_gb",
    "pymalloc_arenas_gb",
    "pymalloc_allocated_gb",
    "pymalloc_available_gb",
    "pymalloc_arenas_after_remedy_gb",
    "tracemalloc_gb",
    "freeze_count",
    "lmdb_open",
    "parent_anon_gb",
    "source",
]

# Set in the parent before the pool forks; every worker inherits them.
VARIANT = "none"
REAL_CHUNKS = 12
REAL_COUNTER: "Synchronized[int] | None" = None
SCRATCH = tempfile.gettempdir()
_STATE: dict[str, Any] = {}


class Mallinfo2(ctypes.Structure):
    """glibc ``struct mallinfo2`` (all size_t)."""

    _fields_ = [
        (name, ctypes.c_size_t)
        for name in (
            "arena",
            "ordblks",
            "smblks",
            "hblks",
            "hblkhd",
            "usmblks",
            "fsmblks",
            "uordblks",
            "fordblks",
            "keepcost",
        )
    ]


LIBC = ctypes.CDLL("libc.so.6")
LIBC.mallinfo2.restype = Mallinfo2
LIBC.malloc_trim.argtypes = [ctypes.c_size_t]


def smaps(pid: int | str) -> dict[str, float]:
    """Rss, Private_Dirty and Anonymous of a process, in GB."""
    out: dict[str, float] = {}
    with open(f"/proc/{pid}/smaps_rollup") as fh:
        for line in fh:
            parts = line.split()
            key = parts[0].rstrip(":")
            if key in ("Rss", "Private_Dirty", "Anonymous"):
                out[key] = int(parts[1]) * KB / GB
    return out


def smaps_or_none(pid: int) -> dict[str, float] | None:
    """Return smaps of a process the sampler found, or None if it exited before the read.

    Loader children exit at every chunk end and pool workers at the group end, between
    the sampler listing them and reading them. This is the one expected race, so only
    its two errors are caught.
    """
    try:
        return smaps(pid)
    except (FileNotFoundError, ProcessLookupError):
        return None


def children(pid: int) -> list[int]:
    """Direct children of every thread of ``pid`` (empty if it has exited)."""
    try:
        tids = os.listdir(f"/proc/{pid}/task")
    except (FileNotFoundError, ProcessLookupError):
        return []
    out: list[int] = []
    for tid in tids:
        try:
            with open(f"/proc/{pid}/task/{tid}/children") as fh:
                out.extend(int(p) for p in fh.read().split())
        except (FileNotFoundError, ProcessLookupError):
            continue
    return out


def pymalloc_stats() -> dict[str, float]:
    """Arena / allocated / available bytes from ``sys._debugmallocstats`` (GB).

    The stats go to the C-level stderr, so fd 2 is pointed at a file for the call.
    """
    path = osp.join(SCRATCH, f"mallocstats_{os.getpid()}.txt")
    saved = os.dup(2)
    with open(path, "w") as fh:
        sys.stderr.flush()
        os.dup2(fh.fileno(), 2)
        sys._debugmallocstats()
        os.dup2(saved, 2)
    os.close(saved)
    values: dict[str, int] = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") and "=" in line:
                label, _, number = line.rpartition("=")
                values[label.strip()] = int(number.strip().replace(",", ""))
    arena_bytes = values["# arenas allocated current"] * 2**20
    return {
        "pymalloc_arenas_gb": arena_bytes / GB,
        "pymalloc_allocated_gb": values["# bytes in allocated blocks"] / GB,
        "pymalloc_available_gb": values["# bytes in available blocks"] / GB,
    }


def snapshot() -> dict[str, float]:
    """This worker's memory by every instrument, in GB."""
    mem = smaps("self")
    info = LIBC.mallinfo2()
    out = {
        "rss_gb": mem["Rss"],
        "private_dirty_gb": mem["Private_Dirty"],
        "anon_gb": mem["Anonymous"],
        "glibc_arena_gb": info.arena / GB,
        "glibc_hblkhd_gb": info.hblkhd / GB,
        "glibc_inuse_gb": info.uordblks / GB,
        "glibc_free_gb": info.fordblks / GB,
        "freeze_count": float(gc.get_freeze_count()),
    }
    if tracemalloc.is_tracing():
        # tracemalloc wraps the allocator, and _debugmallocstats then prints no
        # pymalloc section, so the trace variant reports traced bytes instead.
        out["tracemalloc_gb"] = tracemalloc.get_traced_memory()[0] / GB
        out |= dict.fromkeys(
            ("pymalloc_arenas_gb", "pymalloc_allocated_gb", "pymalloc_available_gb"),
            float("nan"),
        )
        return out
    return out | pymalloc_stats()


def remedy(variant: str) -> None:
    """Apply the variant's in-worker remedy to what the previous chunk left."""
    if variant == "collect":
        gc.collect()
    elif variant in ("unfreeze", "unfreeze_trim"):
        gc.unfreeze()
        gc.collect()
    if variant in ("trim", "unfreeze_trim"):
        LIBC.malloc_trim(0)


def run_explicit(adapter: CellAdapter, data_chunk: Any, method_name: str) -> list[Any]:
    """``data_chunker``'s pool branch, then delete the loader and close the chunk's LMDB."""
    batch = int(
        adapter.loader_batch_size * adapter.get_memory_reduction_factor(method_name)
    )
    data_loader = CpuExperimentLoaderMultiprocessing(
        data_chunk, batch_size=batch, num_workers=adapter.io_workers
    )
    body = CellAdapter._all_chunked.__wrapped__  # type: ignore[attr-defined]
    datas: list[Any] = []
    for records in data_loader:
        for data in records:
            out = body(adapter, data_chunk.transform_item(data), method_name)
            datas.extend(out)
    data_loader.close()
    del data_loader
    data_chunk.close_lmdb()
    return adapter._pack_chunk(datas)


def report_growth(first: tracemalloc.Snapshot, chunk: int) -> None:
    """Print the allocation sites whose live bytes grew since the end of chunk 1."""
    stats = tracemalloc.take_snapshot().compare_to(first, "traceback")
    total = sum(s.size_diff for s in stats)
    print(
        f"[{os.getpid()}] tracemalloc: live bytes {total / 1e6:+.1f} MB from end of "
        f"chunk 1 to end of chunk {chunk}; top sites:",
        flush=True,
    )
    for stat in stats[:6]:
        print(
            f"  {stat.size_diff / 1e6:+8.1f} MB {stat.count_diff:+9d} blocks",
            flush=True,
        )
        for line in stat.traceback.format(limit=6, most_recent_first=True):
            print(f"      {line}", flush=True)


class TaggedTask:
    """Pool task: measure, remedy, measure, then run the real chunk method.

    Pickles exactly the adapter the bound ``_all_chunked`` would, and returns the
    chunk's ``RenderedChunk`` list plus one tag dict.
    """

    def __init__(self, adapter: CellAdapter) -> None:
        """Hold the adapter; the variant and counters are inherited module globals."""
        self.adapter = adapter

    def __call__(self, data_chunk: Any, method_name: str) -> list[Any]:
        """Run one task in the pool worker."""
        if _STATE.get("pid") != os.getpid():
            _STATE.clear()
            _STATE.update(pid=os.getpid(), task=0, real=0, first=None)
            if VARIANT == "trace":
                tracemalloc.start(8)
        _STATE["task"] += 1
        started = time.time()
        pre = snapshot()
        t0 = time.perf_counter()
        if _STATE["task"] > 1:
            remedy(VARIANT)
        remedy_seconds = time.perf_counter() - t0
        post = snapshot()
        if VARIANT == "trace" and _STATE["real"] >= 1:
            if _STATE["first"] is None:
                _STATE["first"] = tracemalloc.take_snapshot()
            else:
                report_growth(_STATE["first"], _STATE["real"])
        assert REAL_COUNTER is not None
        with REAL_COUNTER.get_lock():
            REAL_COUNTER.value += 1
            real = REAL_COUNTER.value <= REAL_CHUNKS
        t1 = time.perf_counter()
        if real:
            if VARIANT == "explicit":
                result = run_explicit(self.adapter, data_chunk, method_name)
            else:
                result = CellAdapter._all_chunked(self.adapter, data_chunk, method_name)
            _STATE["real"] += 1
        else:
            time.sleep(3.0)
            result = []
        tag = {
            "pid": os.getpid(),
            "task": _STATE["task"],
            "chunks_before": _STATE["real"] - int(real),
            "real": real,
            "records": len(data_chunk) if real else 0,
            "started": started,
            "ended": time.time(),
            "seconds": time.perf_counter() - t1,
            "remedy_seconds": remedy_seconds,
            "lmdb_open": int(getattr(data_chunk, "env", None) is not None),
            "pre": pre,
            "post": post,
        }
        return [*result, tag]


class Sampler:
    """Parent thread: smaps of parent, pool workers and loader children every 0.5 s."""

    def __init__(self, limit_gb: float, interval: float = 0.5) -> None:
        """Start sampling; SIGKILL the pool if the tree's anon memory passes the limit."""
        self.limit_gb = limit_gb
        self.interval = interval
        self.rows: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        me = os.getpid()
        t0 = time.time()
        while not self._stop.is_set():
            now = time.time()
            total = 0.0
            procs = [(me, "parent", 0)]
            for worker in children(me):
                procs.append((worker, "worker", me))
                procs.extend((c, "loader", worker) for c in children(worker))
            for pid, role, ppid in procs:
                mem = smaps_or_none(pid)
                if mem is None:
                    continue
                # A forked child's Anonymous counts the pages it still shares with
                # its parent, so the tree's unique memory is the parent's Anonymous
                # plus every child's Private_Dirty.
                total += mem["Anonymous"] if role == "parent" else mem["Private_Dirty"]
                self.rows.append(
                    {
                        "t": now,
                        "elapsed": now - t0,
                        "role": role,
                        "pid": pid,
                        "ppid": ppid,
                        "rss_gb": mem["Rss"],
                        "private_dirty_gb": mem["Private_Dirty"],
                        "anon_gb": mem["Anonymous"],
                    }
                )
            if total > self.limit_gb:
                print(
                    f"GUARD: tree memory {total:.1f} GB > {self.limit_gb}; killing pool",
                    flush=True,
                )
                for pid, role, _ in procs:
                    if role != "parent":
                        os.kill(pid, signal.SIGKILL)
            self._stop.wait(self.interval)

    def stop(self) -> None:
        """Stop and join the thread."""
        self._stop.set()
        self._thread.join()

    def loader_peak(self, worker: int, start: float, end: float) -> float:
        """Peak summed Private_Dirty of ``worker``'s loader children in [start, end]."""
        by_t: dict[float, float] = {}
        for r in self.rows:
            if r["role"] == "loader" and r["ppid"] == worker and start <= r["t"] <= end:
                by_t[r["t"]] = by_t.get(r["t"], 0.0) + r["private_dirty_gb"]
        return max(by_t.values(), default=0.0)

    def parent_anon(self, when: float) -> float:
        """Parent Anonymous at the sample nearest ``when``."""
        rows = [r for r in self.rows if r["role"] == "parent"]
        return float(min(rows, key=lambda r: abs(r["t"] - when))["anon_gb"])

    def last_worker_sample(self, pid: int, after: float) -> dict[str, Any] | None:
        """The last sample of a worker taken after ``after`` (its idle state)."""
        rows = [r for r in self.rows if r["pid"] == pid and r["t"] > after]
        return rows[-1] if rows else None


def build_adapter(args: argparse.Namespace) -> CellAdapter:
    """The dev-tree adapter configured as the ladder build configures it."""
    dataset_class, adapter_class = next(
        (d, a) for d, a in dataset_adapter_map.items() if d.__name__ == args.dataset
    )
    root = osp.join(
        os.environ["DATA_ROOT"],
        inspect.signature(
            dataset_class.__init__  # type: ignore[misc]  # inspecting a class's __init__
        )
        .parameters["root"]
        .default,
    )
    dataset = subset_dataset(dataset_class(root=root), args.subset or None, 42, None)
    adapter: CellAdapter = adapter_class(
        dataset=dataset,
        process_workers=args.process_workers,
        io_workers=args.io_workers,
        chunk_size=args.chunk_size,
        loader_batch_size=args.loader_batch_size,
    )
    bc = BioCypher(
        output_directory=args.bc_out,
        biocypher_config_path=os.environ["BIOCYPHER_CONFIG_PATH"],
        schema_config_path=os.environ["SCHEMA_CONFIG_PATH"],
    )
    adapter.single_pass = True
    adapter.row_specs = build_row_specs(bc)
    adapter.single_pass_chunk_budget_bytes = int(args.budget_mb * 2**20)
    adapter.chunks_per_worker = args.chunks_per_worker
    adapter.completion_order = True
    return adapter


def install_tagged_task() -> None:
    """Route the single-pass traversal through ``TaggedTask``; nothing else changes."""
    original: Callable[..., Iterator[Any]] = CellAdapter.get_data_by_type

    def tagged(
        self: CellAdapter,
        chunk_processing_func: Callable[..., Any],
        method_name: str,
        is_edge: bool = False,
    ) -> Iterator[Any]:
        if method_name in (SINGLE_PASS_NODES, SINGLE_PASS_EDGES):
            chunk_processing_func = TaggedTask(self)
            # What every task ships: the reference methods ran in THIS process just
            # before the single pass, and the dataset caches their index.
            index = self.dataset._experiment_reference_index
            print(
                f"call item: TaggedTask pickles to "
                f"{len(pickle.dumps(chunk_processing_func)) / 1e6:.1f} MB; cached "
                f"experiment reference index: "
                f"{'none' if index is None else sum(len(e.member_indices) for e in index)}"
                f" member indices",
                flush=True,
            )
            if VARIANT == "slim":
                self.dataset._experiment_reference_index = None
                print(
                    f"slim: TaggedTask pickles to "
                    f"{len(pickle.dumps(chunk_processing_func)) / 1e6:.1f} MB",
                    flush=True,
                )
        return original(self, chunk_processing_func, method_name, is_edge)

    setattr(CellAdapter, "get_data_by_type", tagged)  # noqa: B010


def to_rows(
    variant: str, tags: list[dict[str, Any]], sampler: Sampler
) -> list[dict[str, Any]]:
    """One row per worker per chunks-completed count: state after that many chunks."""
    rows: list[dict[str, Any]] = []
    pids = list(
        dict.fromkeys(t["pid"] for t in sorted(tags, key=lambda t: t["started"]))
    )
    for ordinal, pid in enumerate(pids):
        mine = sorted((t for t in tags if t["pid"] == pid), key=lambda t: t["task"])
        chunks = [t for t in mine if t["real"]]
        by_before: dict[int, dict[str, Any]] = {}
        for t in mine:
            by_before.setdefault(t["chunks_before"], t)
        for k in range(len(chunks) + 1):
            done = chunks[k - 1] if k else None
            row: dict[str, Any] = {
                "variant": variant,
                "worker": ordinal,
                "pid": pid,
                "chunk_index": k,
                "records": done["records"] if done else 0,
                "seconds": done["seconds"] if done else 0.0,
                "loader_children_gb": (
                    sampler.loader_peak(pid, done["started"], done["ended"])
                    if done
                    else 0.0
                ),
                "lmdb_open": done["lmdb_open"] if done else "",
            }
            state = by_before.get(k)
            if state is not None:
                pre, post = state["pre"], state["post"]
                row |= {f: pre[f] for f in pre}
                row |= {
                    "rss_after_remedy_gb": post["rss_gb"],
                    "anon_after_remedy_gb": post["anon_gb"],
                    "private_dirty_after_remedy_gb": post["private_dirty_gb"],
                    "glibc_free_after_remedy_gb": post["glibc_free_gb"],
                    "pymalloc_arenas_after_remedy_gb": post["pymalloc_arenas_gb"],
                    "remedy_seconds": state["remedy_seconds"] if k else 0.0,
                    "parent_anon_gb": sampler.parent_anon(state["started"]),
                    "source": "tag",
                }
            else:
                assert done is not None
                last = sampler.last_worker_sample(pid, done["ended"])
                assert last is not None, (
                    f"no sample of worker {pid} after its last chunk"
                )
                row |= {
                    "rss_gb": last["rss_gb"],
                    "private_dirty_gb": last["private_dirty_gb"],
                    "anon_gb": last["anon_gb"],
                    "parent_anon_gb": sampler.parent_anon(last["t"]),
                    "source": "sampler",
                }
            rows.append(row)
    return rows


def slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Least-squares slope of ys on xs."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum(
        (x - mx) ** 2 for x in xs
    )


def summarize(rows: list[dict[str, Any]]) -> None:
    """Per variant and worker: GB per chunk (chunks 1..N) and the remedy's cost."""
    for variant in dict.fromkeys(r["variant"] for r in rows):
        print(f"\n{variant}")
        for worker in sorted(
            {int(r["worker"]) for r in rows if r["variant"] == variant}
        ):
            sel = [
                r
                for r in rows
                if r["variant"] == variant and int(r["worker"]) == worker
            ]
            sel.sort(key=lambda r: int(r["chunk_index"]))
            fit = [r for r in sel if int(r["chunk_index"]) >= 1]
            idx = [float(r["chunk_index"]) for r in fit]
            tagged = [r for r in fit if r["source"] == "tag"]
            anon = " ".join(f"{float(r['anon_gb']):.2f}" for r in sel)
            line = (
                f"  worker {worker}: anon by chunk {anon}\n"
                f"    GB/chunk anon {slope(idx, [float(r['anon_gb']) for r in fit]):+.3f}"
                f" rss {slope(idx, [float(r['rss_gb']) for r in fit]):+.3f}"
                f" private {slope(idx, [float(r['private_dirty_gb']) for r in fit]):+.3f}"
                f"  s/chunk {sum(float(r['seconds']) for r in fit) / len(fit):.1f}"
                f"  loaders max {max(float(r['loader_children_gb']) for r in fit):.2f}"
            )
            if len(tagged) >= 2:
                ti = [float(r["chunk_index"]) for r in tagged]
                line += (
                    f"\n    after remedy GB/chunk anon "
                    f"{slope(ti, [float(r['anon_after_remedy_gb']) for r in tagged]):+.3f}"
                    f"  remedy s {sum(float(r['remedy_seconds']) for r in tagged) / len(tagged):.2f}"
                    f"\n    GB/chunk glibc arena {slope(ti, [float(r['glibc_arena_gb']) for r in tagged]):+.3f}"
                    f" hblkhd {slope(ti, [float(r['glibc_hblkhd_gb']) for r in tagged]):+.3f}"
                    f" in-use {slope(ti, [float(r['glibc_inuse_gb']) for r in tagged]):+.3f}"
                    f" free {slope(ti, [float(r['glibc_free_gb']) for r in tagged]):+.3f}"
                    f" | pymalloc arenas {slope(ti, [float(r['pymalloc_arenas_gb']) for r in tagged]):+.3f}"
                    f" allocated {slope(ti, [float(r['pymalloc_allocated_gb']) for r in tagged]):+.3f}"
                    f" | frozen objs {float(tagged[-1]['freeze_count']):.0f}"
                )
            print(line)


def main() -> None:
    """Run one variant through the real pool, write its rows, print the summary."""
    global VARIANT, REAL_CHUNKS, REAL_COUNTER, SCRATCH
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, default="none")
    parser.add_argument("--dataset", default="DmfCostanzo2016Dataset")
    parser.add_argument("--subset", type=int, default=0)
    parser.add_argument("--process-workers", type=int, default=2)
    parser.add_argument("--chunks-per-worker", type=int, default=8)
    parser.add_argument("--real-chunks", type=int, default=12)
    parser.add_argument("--io-workers", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--loader-batch-size", type=int, default=1_000)
    parser.add_argument("--budget-mb", type=float, default=128)
    parser.add_argument("--limit-gb", type=float, default=22.0)
    parser.add_argument(
        "--out", default=osp.join(HERE, "..", "results", "pool_worker_retention.csv")
    )
    parser.add_argument("--bc-out", required=True)
    parser.add_argument("--series-dir", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        with open(args.out) as fh:
            summarize(list(csv.DictReader(fh)))
        return
    env_set = all(os.environ.get(k) == v for k, v in MALLOC_ENV.items())
    assert env_set == (args.variant == "env"), (
        f"variant {args.variant} needs {MALLOC_ENV} "
        f"{'set' if args.variant == 'env' else 'unset'} at launch"
    )
    label = args.label or args.variant
    VARIANT = args.variant
    REAL_CHUNKS = args.real_chunks
    REAL_COUNTER = mp.get_context("fork").Value("i", 0)
    SCRATCH = args.series_dir
    os.makedirs(SCRATCH, exist_ok=True)
    install_tagged_task()
    adapter = build_adapter(args)
    group = args.process_workers * args.chunks_per_worker
    print(
        f"{args.dataset}: {len(adapter.dataset)} records, "
        f"{adapter._estimate_record_bytes()} resolved bytes per record, group {group}",
        flush=True,
    )
    sampler = Sampler(args.limit_gb)
    tags: list[dict[str, Any]] = []
    rendered = 0
    # get_nodes is a generator function; close() tears the pool down after the group.
    nodes = cast(Generator[Any], adapter.get_nodes())
    for item in nodes:
        if isinstance(item, dict):
            tags.append(item)
            print(
                f"task pid={item['pid']} #{item['task']} real={item['real']} "
                f"before={item['chunks_before']} anon {item['pre']['anon_gb']:.2f}"
                f"->{item['post']['anon_gb']:.2f} GB "
                f"remedy {item['remedy_seconds']:.2f}s chunk {item['seconds']:.1f}s",
                flush=True,
            )
            if len(tags) == group:
                break
        elif isinstance(item, RenderedChunk):
            rendered += 1
    nodes.close()
    time.sleep(1.0)
    sampler.stop()
    assert sum(t["real"] for t in tags) == args.real_chunks, "not every real chunk ran"
    with open(osp.join(args.series_dir, f"series_{label}.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(sampler.rows[0]))
        writer.writeheader()
        writer.writerows(sampler.rows)
    rows = to_rows(label, tags, sampler)
    kept: list[dict[str, Any]] = []
    if osp.exists(args.out):
        with open(args.out) as fh:
            kept = [r for r in csv.DictReader(fh) if r["variant"] != label]
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({f: r.get(f, "") for f in FIELDS} for r in kept + rows)
    print(f"{rendered} rendered chunks consumed")
    summarize(rows)


if __name__ == "__main__":
    main()
