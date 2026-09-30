# experiments/tcdb-002-build-speed/scripts/worker_heap_ratchet.py
# [[experiments.tcdb-002-build-speed.scripts.worker_heap_ratchet]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/tcdb-002-build-speed/scripts/worker_heap_ratchet
"""Per-chunk heap ratchet of ONE single-pass pool worker, measured.

    PYTHONPATH=. WANDB_MODE=disabled python \
        experiments/tcdb-002-build-speed/scripts/worker_heap_ratchet.py \
        --k 16 --variants freeze,freeze_collect,half

The parent builds the adapter the way ``create_scerevisiae_kg_small.py`` does (dev-tree
dataset, optional random subset as on the ladder, row specs from a BioCypher instance,
single pass, the byte budget), builds the first K chunk views with the adapter's own
chunk-length rule, closes the LMDB, runs ``gc.collect(); gc.freeze()`` and forks ONE
child per variant. The parent then hands the child K node chunks and K edge chunks one
at a time over a pipe, pickled exactly as ``ProcessPoolExecutor`` pickles a call item
(the bound ``_all_chunked`` method, hence the adapter and its dataset, plus the chunk
view). The child runs the real ``data_chunker`` path, loader children included, pickles
the ``RenderedChunk`` back and drops it, as a pool worker does. After each chunk the
child records RSS and ``/proc/self/smaps_rollup`` (Rss, Pss, Private_Dirty, Anonymous),
the peak RssAnon of itself and of its loader children during the chunk (0.2 s sampler
thread), the GC's permanent-generation count, pymalloc's allocated blocks, and the
wall time.

Chunk length: by default the byte-budget length, ``budget // record_bytes`` with the
adapter's ``_estimate_record_bytes`` (about 12,000 Costanzo dmf records at 128 MiB).
``get_data_by_type`` uses the smaller of that and ``chunk_size x memory_reduction_factor``
(6,250 node and 5,555 edge records for Costanzo dmf at chunk_size 1e5), so a variant
prefixed ``adapter_`` runs at the length the build actually used.

Variants: ``freeze`` is production; ``freeze_collect`` adds ``gc.collect()`` after each
chunk; ``half`` is ``freeze`` at half the chunk length; ``unfreeze_collect`` runs
``gc.unfreeze(); gc.collect()`` after each chunk; ``trim`` runs glibc
``malloc_trim(0)`` after each chunk; ``tracemalloc`` is ``freeze`` with tracemalloc on,
printing the allocation sites that grew between node chunk 1 and node chunk K. The last
three are mechanism tests.
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
import sys
import threading
import time
import tracemalloc
from collections.abc import Callable, Sequence
from multiprocessing.connection import Connection
from typing import Any

os.environ["WANDB_MODE"] = "disabled"
import wandb  # noqa: E402

wandb.init(mode="disabled")
from dotenv import load_dotenv  # noqa: E402

from biocypher import BioCypher  # type: ignore[attr-defined]  # noqa: E402
from torchcell.adapters.cell_adapter import (  # noqa: E402
    SINGLE_PASS_EDGES,
    SINGLE_PASS_MIN_CHUNK,
    SINGLE_PASS_NODES,
    CellAdapter,
)
from torchcell.fast_csv import RenderedChunk, build_row_specs  # noqa: E402
from torchcell.knowledge_graphs.dataset_adapter_map import (  # noqa: E402
    dataset_adapter_map,
)
from torchcell.knowledge_graphs.subset import subset_dataset  # noqa: E402

load_dotenv()

GB = 1e9
KB = 1024
VARIANTS = (
    "freeze",
    "freeze_collect",
    "half",
    "unfreeze_collect",
    "trim",
    "tracemalloc",
)
FIELDS = [
    "variant",
    "chunk_index",
    "kind",
    "records",
    "rss_gb",
    "private_dirty_gb",
    "anonymous_gb",
    "pss_gb",
    "peak_worker_anon_gb",
    "peak_loader_private_gb",
    "freeze_count",
    "allocated_blocks",
    "call_item_mb",
    "payload_mb",
    "rows",
    "seconds",
]


def smaps_rollup() -> dict[str, float]:
    """Rss, Pss, Private_Dirty and Anonymous of this process, in GB."""
    out: dict[str, float] = {}
    with open("/proc/self/smaps_rollup") as fh:
        for line in fh:
            parts = line.split()
            key = parts[0].rstrip(":")
            if key in ("Rss", "Pss", "Private_Dirty", "Anonymous"):
                out[key] = int(parts[1]) * KB / GB
    return out


def status_anon_gb(pid: int) -> float:
    """RssAnon of a process in GB."""
    with open(f"/proc/{pid}/status") as fh:
        text = fh.read()
    line = next(x for x in text.splitlines() if x.startswith("RssAnon:"))
    return int(line.split()[1]) * KB / GB


def private_dirty_gb(pid: int) -> float:
    """Private_Dirty of a process in GB, 0.0 if it exited while being read.

    Loader children exit at the end of every chunk, between the sampler listing them
    and reading their smaps; a vanished pid has nothing to read, so it counts as zero.
    This is the one expected race, so only its two errors are caught.
    """
    try:
        with open(f"/proc/{pid}/smaps_rollup") as fh:
            text = fh.read()
    except (FileNotFoundError, ProcessLookupError):
        return 0.0
    lines = [x for x in text.splitlines() if x.startswith("Private_Dirty:")]
    return int(lines[0].split()[1]) * KB / GB if lines else 0.0


def child_pids() -> list[int]:
    """Direct children of this process's main thread (the loader processes)."""
    with open(f"/proc/self/task/{os.getpid()}/children") as fh:
        return [int(p) for p in fh.read().split()]


class PeakSampler:
    """Background thread: peak worker RssAnon and peak loader-children Private_Dirty."""

    def __init__(self, interval: float = 0.25) -> None:
        """Start sampling every ``interval`` seconds."""
        self.interval = interval
        self.worker_peak = 0.0
        self.loader_peak = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        me = os.getpid()
        while not self._stop.is_set():
            self.worker_peak = max(self.worker_peak, status_anon_gb(me))
            loaders = sum(private_dirty_gb(p) for p in child_pids())
            self.loader_peak = max(self.loader_peak, loaders)
            self._stop.wait(self.interval)

    def reset(self) -> tuple[float, float]:
        """Return (worker_peak, loader_peak) since the last reset and zero them."""
        assert self._thread.is_alive(), "peak sampler thread died"
        peaks = (self.worker_peak, self.loader_peak)
        self.worker_peak = 0.0
        self.loader_peak = 0.0
        return peaks


def worker(conn: Connection, variant: str) -> None:
    """The emulated pool worker: run call items until a None arrives."""
    libc = ctypes.CDLL("libc.so.6")
    if variant == "tracemalloc":
        tracemalloc.start(12)
    first: tracemalloc.Snapshot | None = None
    node_chunks = 0
    sampler = PeakSampler()
    base = smaps_rollup()
    conn.send(
        {
            "rss_gb": base["Rss"],
            "private_dirty_gb": base["Private_Dirty"],
            "anonymous_gb": base["Anonymous"],
            "pss_gb": base["Pss"],
            "freeze_count": gc.get_freeze_count(),
            "allocated_blocks": sys.getallocatedblocks(),
        }
    )
    sampler.reset()
    while True:
        blob = conn.recv_bytes()
        t0 = time.perf_counter()
        call_item = pickle.loads(blob)
        del blob
        if call_item is None:
            break
        fn, chunk, method_name = call_item
        records = len(chunk)
        is_node = method_name == SINGLE_PASS_NODES
        result = fn(chunk, method_name)
        payload = pickle.dumps(result)
        rows = sum(len(b[1]) for rc in result for b in rc.nodes.values()) + sum(
            len(b[2]) for rc in result for b in rc.edges.values()
        )
        del result
        conn.send_bytes(payload)
        payload_mb = len(payload) / 1e6
        del payload
        del fn, chunk, call_item
        if variant == "freeze_collect":
            gc.collect()
        elif variant == "unfreeze_collect":
            gc.unfreeze()
            gc.collect()
        elif variant == "trim":
            libc.malloc_trim(0)
        seconds = time.perf_counter() - t0
        if variant == "tracemalloc" and is_node:
            node_chunks += 1
            snap = tracemalloc.take_snapshot()
            if first is None:
                first = snap
            else:
                report_growth(first, snap, node_chunks)
        mem = smaps_rollup()
        worker_peak, loader_peak = sampler.reset()
        conn.send(
            {
                "records": records,
                "rss_gb": mem["Rss"],
                "private_dirty_gb": mem["Private_Dirty"],
                "anonymous_gb": mem["Anonymous"],
                "pss_gb": mem["Pss"],
                "peak_worker_anon_gb": worker_peak,
                "peak_loader_private_gb": loader_peak,
                "freeze_count": gc.get_freeze_count(),
                "allocated_blocks": sys.getallocatedblocks(),
                "payload_mb": payload_mb,
                "rows": rows,
                "seconds": seconds,
            }
        )
    conn.close()


def report_growth(
    first: tracemalloc.Snapshot, last: tracemalloc.Snapshot, chunks: int
) -> None:
    """Print the allocation sites whose live bytes grew since node chunk 1."""
    stats = last.compare_to(first, "traceback")
    total = sum(s.size_diff for s in stats)
    print(
        f"tracemalloc: live bytes grew {total / 1e6:+.1f} MB between node chunk 1 "
        f"and node chunk {chunks}; top sites:",
        flush=True,
    )
    for stat in stats[:8]:
        print(
            f"  {stat.size_diff / 1e6:+8.1f} MB {stat.count_diff:+9d} blocks",
            flush=True,
        )
        for line in stat.traceback.format(limit=12, most_recent_first=True):
            print(f"      {line}", flush=True)


def folded_methods(
    adapter: CellAdapter, kind: str
) -> list[tuple[str, Callable[..., Any]]]:
    """The chunked methods ``_yield_methods`` folds into one single pass of ``kind``."""
    methods = adapter.node_methods if kind == "node" else adapter.edge_methods
    config = (
        adapter.config.cell_adapter.node_methods
        if kind == "node"
        else adapter.config.cell_adapter.edge_methods
    )
    enabled = {m["method_name"] for m in config}
    return [
        (name, method)
        for name, method in methods
        if name in enabled and not method.__name__.startswith("_get_")
    ]


def chunk_lengths(adapter: CellAdapter, kind: str) -> dict[str, int]:
    """Chunk length of ``kind`` by the byte budget alone and by the adapter's rule."""
    pass_name = SINGLE_PASS_NODES if kind == "node" else SINGLE_PASS_EDGES
    adapter._single_pass_methods = folded_methods(adapter, kind)
    factor = adapter.get_memory_reduction_factor(pass_name, kind == "edge")
    length = int(adapter.chunk_size * factor)
    budget = max(
        SINGLE_PASS_MIN_CHUNK,
        adapter.single_pass_chunk_budget_bytes // adapter._estimate_record_bytes(),
    )
    return {"budget": budget, "adapter": min(length, budget)}


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
        process_workers=1,
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
    return adapter


def run_variant(
    adapter: CellAdapter, variant: str, k: int, lengths: dict[str, dict[str, int]]
) -> list[dict[str, Any]]:
    """Fork one worker, feed it K node then K edge chunks, return one row per chunk."""
    rule = "adapter" if variant.startswith("adapter_") else "budget"
    base = variant.removeprefix("adapter_")
    scale = 2 if base == "half" else 1
    views = {
        kind: [
            adapter.dataset[
                i * lengths[kind][rule] // scale : (i + 1)
                * lengths[kind][rule]
                // scale
            ]
            for i in range(k)
        ]
        for kind in ("node", "edge")
    }
    adapter.dataset.close_lmdb()
    gc.collect()
    gc.freeze()
    parent_conn, child_conn = mp.get_context("fork").Pipe()
    proc = mp.get_context("fork").Process(target=worker, args=(child_conn, base))
    proc.start()
    child_conn.close()
    rows: list[dict[str, Any]] = [
        {"variant": variant, "chunk_index": 0, "kind": "baseline", "records": 0}
        | parent_conn.recv()
    ]
    for kind in ("node", "edge"):
        pass_name = SINGLE_PASS_NODES if kind == "node" else SINGLE_PASS_EDGES
        adapter._single_pass_methods = folded_methods(adapter, kind)
        for i, view in enumerate(views[kind]):
            blob = pickle.dumps((adapter._all_chunked, view, pass_name))
            parent_conn.send_bytes(blob)
            payload = parent_conn.recv_bytes()
            assert isinstance(pickle.loads(payload)[0], RenderedChunk)
            del payload
            stats = parent_conn.recv()
            row = {
                "variant": variant,
                "chunk_index": i + 1,
                "kind": kind,
                "call_item_mb": len(blob) / 1e6,
            } | stats
            rows.append(row)
            print(
                f"{variant} {kind} {i + 1:>2} rec={row['records']} "
                f"rss={row['rss_gb']:.2f} priv={row['private_dirty_gb']:.2f} "
                f"anon={row['anonymous_gb']:.2f} peak={row['peak_worker_anon_gb']:.2f} "
                f"loaders={row['peak_loader_private_gb']:.2f} "
                f"frozen={row['freeze_count']} blocks={row['allocated_blocks']} "
                f"item={row['call_item_mb']:.1f}MB out={row['payload_mb']:.1f}MB "
                f"{row['seconds']:.1f}s",
                flush=True,
            )
    parent_conn.send_bytes(pickle.dumps(None))
    proc.join()
    gc.unfreeze()
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
    """Print RSS at chunks 1, 2, 4, 8, 16 and the fitted GB-per-chunk slopes."""
    for variant in dict.fromkeys(r["variant"] for r in rows):
        base = next(
            r for r in rows if r["variant"] == variant and r["kind"] == "baseline"
        )
        print(f"\n{variant}: baseline rss {float(base['rss_gb']):.2f} GB")
        for kind in ("node", "edge"):
            sel = [r for r in rows if r["variant"] == variant and r["kind"] == kind]
            idx = [int(r["chunk_index"]) for r in sel]
            marks = "  ".join(
                f"c{c}={float(r['rss_gb']):.2f}"
                for c in (1, 2, 4, 8, 16)
                for r in sel
                if int(r["chunk_index"]) == c
            )
            half = [r for r in sel if int(r["chunk_index"]) > len(sel) // 2]
            if len(half) < 2:
                half = sel
            print(
                f"  {kind}: records {sel[0]['records']}  rss {marks}\n"
                f"    slope GB/chunk: rss {slope(idx, [float(r['rss_gb']) for r in sel]):+.3f}"
                f"  private {slope(idx, [float(r['private_dirty_gb']) for r in sel]):+.3f}"
                f"  anon {slope(idx, [float(r['anonymous_gb']) for r in sel]):+.3f}"
                f"  (second half rss "
                f"{slope([int(r['chunk_index']) for r in half], [float(r['rss_gb']) for r in half]):+.3f})"
                f"  peak anon max {max(float(r['peak_worker_anon_gb']) for r in sel):.2f}"
                f"  loaders private max {max(float(r['peak_loader_private_gb']) for r in sel):.2f}"
                f"  s/chunk {sum(float(r['seconds']) for r in sel) / len(sel):.1f}"
            )


def main() -> None:
    """Parse arguments, run each variant in its own forked worker, write the CSV."""
    here = osp.dirname(osp.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="DmfCostanzo2016Dataset")
    parser.add_argument("--subset", type=int, default=2_000_000)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--variants", default="freeze,freeze_collect,half")
    parser.add_argument("--io-workers", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--loader-batch-size", type=int, default=1_000)
    parser.add_argument("--budget-mb", type=float, default=128)
    parser.add_argument(
        "--out", default=osp.join(here, "..", "results", "worker_heap_ratchet.csv")
    )
    parser.add_argument("--bc-out", required=True)
    parser.add_argument("--summarize", nargs="*", default=None)
    args = parser.parse_args()
    if args.summarize is not None:
        rows: list[dict[str, Any]] = []
        for path in args.summarize:
            with open(path) as fh:
                rows.extend(csv.DictReader(fh))
        summarize(rows)
        return
    variants = args.variants.split(",")
    unknown = sorted({v.removeprefix("adapter_") for v in variants} - set(VARIANTS))
    if unknown:
        raise ValueError(f"unknown variants {unknown}; choose from {VARIANTS}")
    adapter = build_adapter(args)
    lengths = {kind: chunk_lengths(adapter, kind) for kind in ("node", "edge")}
    print(
        f"{args.dataset}: {len(adapter.dataset)} records, "
        f"{adapter._estimate_record_bytes()} resolved bytes per record, "
        f"chunk length node {lengths['node']} edge {lengths['edge']}",
        flush=True,
    )
    rows = []
    for variant in variants:
        rows.extend(run_variant(adapter, variant, args.k, lengths))
        with open(args.out, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows({f: r.get(f, "") for f in FIELDS} for r in rows)
    summarize(rows)


if __name__ == "__main__":
    main()
