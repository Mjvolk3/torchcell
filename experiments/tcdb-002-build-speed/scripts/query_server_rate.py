# experiments/tcdb-002-build-speed/scripts/query_server_rate.py
# [[experiments.tcdb-002-build-speed.scripts.query_server_rate]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/tcdb-002-build-speed/scripts/query_server_rate
"""Measure the Neo4j server and driver side of a query build on one dataset block.

The client-side probe (``query_build_cost_probe.py``) left 1.36 h of the 033 build's
raw stage unexplained by subtraction: time the server spends matching and streaming
records plus what the driver spends decoding them. This script runs ONE block of the
033 query (the Vanacloig 2022 dataset, 143,218 served records) against the live graph
three ways and reports records per second for each:

1. ``stream``: iterate the result and only touch the two strings (no parsing, no
   writing): the server + driver ceiling for one session.
2. ``process``: ``Neo4jQueryRaw.process`` on the same block into a scratch root, the
   full single-pass raw stage (stage 1 of the query work).
3. ``partitions``: the same block split by the first hex character of ``e.id`` into
   16 ranges, streamed by N concurrent sessions (threads), for N in 1, 2, 4, 8: whether
   the server scales with concurrent sessions, which decides whether a parallel fetch
   is worth building.

Runs under slurm only (it touches the served store):

    sbatch experiments/tcdb-002-build-speed/scripts/gh_query_server_rate.slurm

Writes ``results/query_server_rate.csv`` (mode, sessions, records, seconds,
records_per_s, bytes).

``--fetch-workers 1,4,8`` instead measures the partitioned raw stage (stage 3): it
builds ``Neo4jQueryRaw`` on the same block once with ``fetch_workers=0`` (one session,
the baseline) and then once per listed N with ``fetch_workers=N`` (the block carries
``{partition}``, so it is split into ``e.id``-prefix partitions plus the guard: 16 at
``--prefix-length 1``, 256 at 2), asserts every LMDB key and value in cursor order,
``experiment_reference_index.json`` and ``gene_set.json`` identical to the baseline
(a streamed sha256, so a multi-million-record block never sits in memory), and writes
``results/query_server_rate_fetch_workers.csv`` (dataset, fetch_workers,
prefix_length, records, seconds, records_per_s, identical_to_one_session,
parent_peak_gb, worker_peak_gb; the peaks are the process-lifetime maxima of this
process and of its largest fetch worker, read after each arm, so they only grow):

    sbatch experiments/tcdb-002-build-speed/scripts/gh_query_fetch_workers.slurm

``--dataset EnvChemgenHoepfner2014Dataset --fetch-workers 8 --prefix-length 2
--cleanup`` runs the same on the 3.1M-record Hoepfner block, the setting a
multi-million-record block needs (``prefix_length`` 1 would hold 16 partitions of
about 195k rendered records in flight at 8 workers), removing each arm's build
directory once it is hashed, so the disk holds one 40 GB raw store at a time:

    sbatch experiments/tcdb-002-build-speed/scripts/gh_query_fetch_workers_hoepfner.slurm
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import os.path as osp
import resource
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from neo4j import GraphDatabase

RESULTS_DIR = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "results")

DEFAULT_DATASET = "EnvChemgenVanacloig2022Dataset"

BLOCK_TEMPLATE = """
MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
WHERE dataset.id = '{dataset}'{partition}
MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
WHERE ALL(p IN [(g)<-[:PerturbationMemberOf]-(pert) | pert]
WHERE p.systematic_gene_name IN $gene_set)
 AND SIZE([(g)<-[:PerturbationMemberOf]-(pert) | pert]) > 0
WITH DISTINCT e, ref
 ORDER BY e.id
RETURN e.serialized_data AS e_serialized, ref.serialized_data AS ref_serialized
"""

BLOCK = BLOCK_TEMPLATE.replace("{dataset}", DEFAULT_DATASET)


def block_query(dataset: str) -> str:
    """The marked block for one served dataset (``{partition}`` kept for the workers)."""
    return BLOCK_TEMPLATE.replace("{dataset}", dataset)


def stream(
    driver: Any, database: str, query: str, gene_set: list[str]
) -> tuple[int, int]:
    """Consume a query's records, touching only the two strings."""
    n = 0
    nbytes = 0
    with driver.session(database=database, fetch_size=1000) as session:
        for record in session.run(query, gene_set=gene_set):
            n += 1
            nbytes += len(record["e_serialized"]) + len(record["ref_serialized"])
    return n, nbytes


def raw_digest(raw: Any) -> tuple[str, int]:
    """sha256 over a raw stage's LMDB items in cursor order (key length, key, value
    length, value) and its reference-index and gene-set files; and the record count.

    Streams the store, so the digest of a multi-million-record block costs no memory;
    two raw stages with equal digests hold the same keys and values in the same order
    and the same two files.
    """
    import lmdb

    digest = hashlib.sha256()
    n = 0
    env = lmdb.open(raw.lmdb_dir, readonly=True, lock=False, readahead=True)
    with env.begin(buffers=True) as txn:
        for key, value in txn.cursor():
            digest.update(len(key).to_bytes(8, "little"))
            digest.update(key)
            digest.update(len(value).to_bytes(8, "little"))
            digest.update(value)
            n += 1
    env.close()
    for name in ("experiment_reference_index.json", "gene_set.json"):
        with open(osp.join(raw.raw_dir, name), "rb") as fh:
            data = fh.read()
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return digest.hexdigest(), n


def peak_rss_gb(who: int) -> float:
    """Process-lifetime peak resident set in GB (``ru_maxrss`` is in KB on Linux)."""
    return resource.getrusage(who).ru_maxrss / 1e6


def fetch_workers_mode(
    dataset: str,
    counts: list[int],
    prefix_length: int,
    gene_set: list[str],
    scratch_root: str,
    out: str,
    cleanup: bool,
) -> None:
    """Time Neo4jQueryRaw on the marked block per fetch_workers; check identity."""
    from torchcell.data.neo4j_query_raw import Neo4jQueryRaw

    uri = os.environ["NEO4J_URI"]
    user = os.environ["NEO4J_USER"]
    password = os.environ["NEO4J_PASSWORD"]
    query = block_query(dataset)
    rows: list[dict[str, Any]] = []
    baseline = None
    for workers in [0, *counts]:
        root = tempfile.mkdtemp(prefix=f"fetch_workers_{workers}_", dir=scratch_root)
        t = time.perf_counter()
        raw = Neo4jQueryRaw(
            uri=uri,
            username=user,
            password=password,
            root_dir=root,
            query=query,  # carries the {partition} marker
            cypher_kwargs={"gene_set": gene_set},
            fetch_workers=workers,
            partition_prefix_length=prefix_length,
        )
        seconds = time.perf_counter() - t
        raw.close_lmdb()
        digest, n = raw_digest(raw)
        if baseline is None:
            baseline = digest
        identical = digest == baseline
        rows.append(
            {
                "dataset": dataset,
                "fetch_workers": workers,
                "prefix_length": prefix_length if workers else 0,
                "records": n,
                "seconds": round(seconds, 1),
                "records_per_s": round(n / seconds, 1),
                "identical_to_one_session": identical,
                "parent_peak_gb": round(peak_rss_gb(resource.RUSAGE_SELF), 2),
                "worker_peak_gb": round(peak_rss_gb(resource.RUSAGE_CHILDREN), 2),
            }
        )
        print(
            f"{dataset} fetch_workers {workers} prefix {rows[-1]['prefix_length']}: "
            f"{n:,} records in {seconds:.1f} s, {n / seconds:,.0f}/s, identical to "
            f"one session: {identical}, peak parent {rows[-1]['parent_peak_gb']} GB, "
            f"largest worker {rows[-1]['worker_peak_gb']} GB",
            flush=True,
        )
        if cleanup:
            shutil.rmtree(root)
        if not identical:
            raise SystemExit(f"fetch_workers={workers} differs from one session")
    os.makedirs(osp.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gene-set", required=True, help="JSON list of genes")
    parser.add_argument("--scratch-root", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--fetch-workers",
        default=None,
        help="comma-separated worker counts: measure the partitioned raw stage only",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="served Dataset id of the block (--fetch-workers mode)",
    )
    parser.add_argument(
        "--prefix-length",
        type=int,
        default=1,
        help="Neo4jQueryRaw.partition_prefix_length for the worker arms",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="remove each arm's build directory once its digest is taken",
    )
    args = parser.parse_args()
    if args.fetch_workers is not None:
        with open(args.gene_set) as fh:
            genes = json.load(fh)
        fetch_workers_mode(
            args.dataset,
            [int(n) for n in args.fetch_workers.split(",")],
            args.prefix_length,
            genes,
            args.scratch_root,
            args.out or osp.join(RESULTS_DIR, "query_server_rate_fetch_workers.csv"),
            args.cleanup,
        )
        return
    args.out = args.out or osp.join(RESULTS_DIR, "query_server_rate.csv")

    from torchcell.data.neo4j_query_raw import Neo4jQueryRaw
    from torchcell.database.connection import neo4j_connection_settings
    from torchcell.knowledge_graphs.releases import resolve_database

    with open(args.gene_set) as fh:
        gene_set = json.load(fh)
    uri = os.environ["NEO4J_URI"]
    user = os.environ["NEO4J_USER"]
    password = os.environ["NEO4J_PASSWORD"]
    version = neo4j_connection_settings().version
    database = resolve_database(version, uri, user, password)
    print(f"database {database}; gene set {len(gene_set)} genes", flush=True)
    driver = GraphDatabase.driver(uri, auth=(user, password))
    rows: list[dict[str, Any]] = []

    def record(mode: str, sessions: int, n: int, seconds: float, nbytes: int) -> None:
        rows.append(
            {
                "mode": mode,
                "sessions": sessions,
                "records": n,
                "seconds": round(seconds, 1),
                "records_per_s": round(n / seconds, 1),
                "bytes": nbytes,
            }
        )
        print(
            f"{mode:<10} sessions {sessions}: {n:,} records in {seconds:.1f} s, "
            f"{n / seconds:,.0f}/s, {nbytes / 1e6:.0f} MB",
            flush=True,
        )

    t = time.perf_counter()
    n, nbytes = stream(driver, database, BLOCK.format(partition=""), gene_set)
    record("stream", 1, n, time.perf_counter() - t, nbytes)

    root = osp.join(args.scratch_root, "process")
    if osp.isdir(root):
        shutil.rmtree(root)
    t = time.perf_counter()
    raw = Neo4jQueryRaw(
        uri=uri,
        username=user,
        password=password,
        root_dir=root,
        query=BLOCK.format(partition=""),
        cypher_kwargs={"gene_set": gene_set},
    )
    seconds = time.perf_counter() - t
    record("process", 1, len(raw), seconds, nbytes)
    raw.close_lmdb()

    hex_digits = "0123456789abcdef"
    for sessions in (1, 2, 4, 8):
        queries = [
            BLOCK.format(partition=f" AND e.id STARTS WITH '{h}'") for h in hex_digits
        ]
        t = time.perf_counter()
        with ThreadPoolExecutor(max_workers=sessions) as pool:
            parts = list(
                pool.map(lambda q: stream(driver, database, q, gene_set), queries)
            )
        seconds = time.perf_counter() - t
        record(
            "partitions",
            sessions,
            sum(p[0] for p in parts),
            seconds,
            sum(p[1] for p in parts),
        )

    driver.close()
    os.makedirs(osp.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
