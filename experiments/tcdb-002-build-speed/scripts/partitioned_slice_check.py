# experiments/tcdb-002-build-speed/scripts/partitioned_slice_check.py
# [[experiments.tcdb-002-build-speed.scripts.partitioned_slice_check]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/tcdb-002-build-speed/scripts/partitioned_slice_check
"""Stage 3 on 033 records: the partitioned raw stage against one session, no bolt.

The probe's slice of the 033 pooled build (``query_build_cost_probe.read_slice``:
20,000 evenly spaced processed groups, flattened to their 21,184 records) is served by
a stub that answers queries as the server does: the records split into one block per
dataset (four, in the order the 033 query unions them), each block sorted by ``e.id``,
the served id (sha256 of the inlined record, recomputed here from the stored
experiment, which is the inlined record byte for byte). Each run builds
``Neo4jQueryRaw`` with the 033 build's observer (``RawStageGrouping`` keyed by the 033
``GenotypeEnvironmentAggregator``, with summaries) in its own process, so peak memory is
its own:

    --run N --out DIR   one build with fetch_workers=N into DIR; writes DIR/run.json
                        (wall, parent CPU, worker CPU, parent peak RSS and the RSS
                        before the build) and DIR/grouping.json
    (no --run)          the driver: runs N = 0, 1, 2, 3 in subprocesses, asserts every
                        LMDB key and value, experiment_reference_index.json,
                        gene_set.json and the observer state identical to N = 0, and
                        writes results/partitioned_slice_check.csv

    nice -n 10 ~/miniconda3/envs/torchcell/bin/python \
        experiments/tcdb-002-build-speed/scripts/partitioned_slice_check.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import os.path as osp
import re
import resource
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from typing import Any

import lmdb

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from query_build_cost_probe import (  # noqa: E402
    RESULTS_DIR,
    SAMPLE_GROUPS,
    SCRATCH,
    load_aggregator_class,
    read_slice,
)

from torchcell.data.neo4j_cell import RawStageGrouping  # noqa: E402
from torchcell.data.neo4j_query_raw import Neo4jQueryRaw  # noqa: E402

DATASETS = [
    "EnvChemgenVanacloig2022Dataset",
    "HetHillenmeyer2008Dataset",
    "EnvChemgenHoepfner2014Dataset",
    "EnvChemgenWildenhain2015Dataset",
]
BLOCK = """
MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
WHERE dataset.id = '{dataset}'{partition}
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
WITH DISTINCT e, ref
 ORDER BY e.id
RETURN e.serialized_data AS e_serialized, ref.serialized_data AS ref_serialized
"""
QUERY = "\nUNION ALL\n".join(BLOCK.replace("{dataset}", d) for d in DATASETS)
WORKER_COUNTS = [0, 1, 2, 3]


def blocks() -> list[list[dict[str, str]]]:
    """The slice as four dataset blocks, each sorted by the served e.id."""
    groups, _ = read_slice(SAMPLE_GROUPS)
    by_dataset: dict[str, list[dict[str, str]]] = {d: [] for d in DATASETS}
    for group in groups:
        for record in group:
            e_serialized = json.dumps(record["experiment"])
            e_id = hashlib.sha256(e_serialized.encode()).hexdigest()
            assert re.fullmatch("[0-9a-f]{64}", e_id)
            by_dataset[record["experiment"]["dataset_name"]].append(
                {
                    "e_id": e_id,
                    "e_serialized": e_serialized,
                    "ref_serialized": json.dumps(record["experiment_reference"]),
                }
            )
    return [sorted(by_dataset[d], key=lambda r: r["e_id"]) for d in DATASETS]


class SliceServer(Neo4jQueryRaw):
    """Answers whole and partition queries from ``BLOCKS`` the way the server does."""

    BLOCKS: list[list[dict[str, str]]] = []

    def fetch_query(self, query: str) -> Iterator[Any]:
        """Blocks in order, each by e.id; a partition is its block's prefix range."""
        if re.search(r"\bUNION\s+ALL\b", query):
            for block in type(self).BLOCKS:
                yield from block
            return
        dataset = re.search(r"dataset\.id = '(\w+)'", query)
        assert dataset is not None
        block = type(self).BLOCKS[DATASETS.index(dataset.group(1))]
        prefix = re.search(r"e\.id STARTS WITH '([0-9a-f]+)'", query)
        if prefix is not None:
            yield from (r for r in block if r["e_id"].startswith(prefix.group(1)))
        else:
            guard = re.search(r"NOT e\.id =~ '(.*)'", query)
            assert guard is not None
            yield from (r for r in block if not re.fullmatch(guard.group(1), r["e_id"]))


def rss_mb() -> float:
    """Current resident set size of this process in MB."""
    with open("/proc/self/statm") as fh:
        return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e6


STAGE_S: dict[str, float] = {}


def timed_method(name: str) -> None:
    """Accumulate the wall time of ``SliceServer.<name>`` in ``STAGE_S``."""
    method = getattr(Neo4jQueryRaw, name)

    def wrapped(self: Any, *args: Any) -> Any:
        t = time.perf_counter()
        result = method(self, *args)
        STAGE_S[name] = STAGE_S.get(name, 0.0) + time.perf_counter() - t
        return result

    setattr(SliceServer, name, wrapped)


def run(fetch_workers: int, out: str) -> None:
    """One build in this process; write timings, memory and the observer state.

    Peak memory is sampled from /proc every 5 ms while the build runs, so it is the
    build's own peak above the RSS it started from, not the setup's (the slice
    itself is parsed before the build starts).
    """
    SliceServer.BLOCKS = blocks()
    n = sum(len(b) for b in SliceServer.BLOCKS)
    aggregator = load_aggregator_class()(root=osp.join(out, "unused"))
    grouping = RawStageGrouping(key_fn=aggregator.aggregate_key_raw, summarize=True)
    for name in (
        "_process_single_session",
        "_process_partitioned",
        "_reference_index_from_groups",
    ):
        timed_method(name)
    rss_before = rss_mb()
    peak = [rss_before]
    done = threading.Event()

    def sample() -> None:
        while not done.wait(0.005):
            peak[0] = max(peak[0], rss_mb())

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    cpu = resource.getrusage(resource.RUSAGE_SELF)
    t = time.perf_counter()
    raw = SliceServer(
        uri="bolt://none",
        username="",
        password="",
        root_dir=out,
        query=QUERY,
        record_observers=[grouping],
        fetch_workers=fetch_workers,
    )
    wall = time.perf_counter() - t
    done.set()
    sampler.join()
    raw.close_lmdb()
    me = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    with open(osp.join(out, "grouping.json"), "w") as fh:
        json.dump({"groups": grouping.groups, "summaries": grouping.summaries}, fh)
    with open(osp.join(out, "run.json"), "w") as fh:
        json.dump(
            {
                "fetch_workers": fetch_workers,
                "records": n,
                "wall_s": wall,
                "parent_cpu_s": me.ru_utime + me.ru_stime - cpu.ru_utime - cpu.ru_stime,
                "worker_cpu_s": children.ru_utime + children.ru_stime,
                "stream_s": STAGE_S.get("_process_partitioned", 0.0)
                + STAGE_S.get("_process_single_session", 0.0),
                "reference_index_s": STAGE_S["_reference_index_from_groups"],
                "parent_rss_before_mb": rss_before,
                "parent_peak_rss_during_build_mb": peak[0],
                "worker_peak_rss_mb": children.ru_maxrss / 1e3,
            },
            fh,
        )


def outputs(out: str) -> tuple[list[tuple[bytes, bytes]], bytes, bytes, bytes]:
    """LMDB items, index file, gene set file, observer state."""
    env = lmdb.open(osp.join(out, "raw", "lmdb"), readonly=True, lock=False)
    with env.begin() as txn:
        items = [(bytes(k), bytes(v)) for k, v in txn.cursor()]
    env.close()
    files = [
        open(osp.join(out, "raw", name), "rb").read()
        for name in ("experiment_reference_index.json", "gene_set.json")
    ]
    grouping = open(osp.join(out, "grouping.json"), "rb").read()
    return items, files[0], files[1], grouping


def drive() -> None:
    """Run every worker count in its own process, compare, and write the CSV."""
    rows = []
    reference = None
    for workers in WORKER_COUNTS:
        out = tempfile.mkdtemp(prefix=f"psc_{workers}_", dir=SCRATCH)
        subprocess.run(
            [sys.executable, __file__, "--run", str(workers), "--out", out], check=True
        )
        with open(osp.join(out, "run.json")) as fh:
            stats = json.load(fh)
        result = outputs(out)
        if reference is None:
            reference = result
        same = result == reference
        assert same, f"fetch_workers={workers} differs from one session"
        n = stats["records"]
        row = {
            "fetch_workers": workers,
            "records": n,
            "identical_to_one_session": same,
            "wall_ms_per_record": round(stats["wall_s"] / n * 1e3, 4),
            "records_per_s": round(n / stats["wall_s"], 1),
            "stream_ms_per_record": round(stats["stream_s"] / n * 1e3, 4),
            "stream_records_per_s": round(n / stats["stream_s"], 1),
            "reference_index_s": round(stats["reference_index_s"], 2),
            "parent_cpu_ms_per_record": round(stats["parent_cpu_s"] / n * 1e3, 4),
            "worker_cpu_ms_per_record": round(stats["worker_cpu_s"] / n * 1e3, 4),
            "parent_rss_before_mb": round(stats["parent_rss_before_mb"]),
            "parent_peak_rss_during_build_mb": round(
                stats["parent_peak_rss_during_build_mb"]
            ),
            "worker_peak_rss_mb": round(stats["worker_peak_rss_mb"]),
        }
        rows.append(row)
        print(row, flush=True)
    path = osp.join(RESULTS_DIR, "partitioned_slice_check.csv")
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(path)


def main() -> None:
    """Dispatch to one run or the driver."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.run is None:
        drive()
    else:
        run(args.run, args.out)


if __name__ == "__main__":
    main()
