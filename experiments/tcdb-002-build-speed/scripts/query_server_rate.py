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
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import os.path as osp
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from neo4j import GraphDatabase

RESULTS_DIR = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "results")

BLOCK = """
MATCH (dataset:Dataset)<-[:ExperimentMemberOf]-(e:Experiment)
WHERE dataset.id = 'EnvChemgenVanacloig2022Dataset'{partition}
MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
MATCH (e)<-[:ExperimentReferenceOf]-(ref:ExperimentReference)
WHERE ALL(p IN [(g)<-[:PerturbationMemberOf]-(pert) | pert]
WHERE p.systematic_gene_name IN $gene_set)
 AND SIZE([(g)<-[:PerturbationMemberOf]-(pert) | pert]) > 0
WITH DISTINCT e, ref
 ORDER BY e.id
RETURN e.serialized_data AS e_serialized, ref.serialized_data AS ref_serialized
"""


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gene-set", required=True, help="JSON list of genes")
    parser.add_argument("--scratch-root", required=True)
    parser.add_argument("--out", default=osp.join(RESULTS_DIR, "query_server_rate.csv"))
    args = parser.parse_args()

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
