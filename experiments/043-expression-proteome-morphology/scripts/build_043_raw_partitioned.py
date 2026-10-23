# experiments/043-expression-proteome-morphology/scripts/build_043_raw_partitioned.py
# [[experiments.043-expression-proteome-morphology.scripts.build_043_raw_partitioned]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/build_043_raw_partitioned
"""The partitioned CANDIDATE of the store v2 raw stage, built beside the reference.

Owner, 2026-10-23: "if we speed it up we must be confident that we are getting
equivalent query." The reference is ``build_043_store.py`` under
``STORE_043_VERSION=v2`` (job 6187: one session, ``raw_043_v2``). This script runs the
SAME raw stage (``Neo4jCellDataset.load_raw``, the call ``Neo4jCellDataset.process``
makes) on ``queries/expression_proteome_morphology_043_partitioned.cql`` (the v2 query
with ``{partition}`` once per block; removing the markers gives the v2 query byte for
byte, ``generate_043_cql.py`` asserts it) with ``fetch_workers`` forked fetch workers and
``partition_prefix_length`` hex characters per ``e.id`` prefix, into
``raw_043_v2_partitioned/raw``. It runs no aggregation and no processed copy: the
equivalence test (``compare_raw_stores_043.py``) reads only ``raw/lmdb``, the reference
index and the gene set.

No record observer is attached (the reference attaches ``RecordAggregator``'s grouping,
which reads records and writes nothing into the raw LMDB). The raw dataset index
(``raw/raw_dataset_name_index.json``) is written after the stage, as the reference does.

Writes ``results/build_043_raw_partitioned.json``: wall time, record count, rate, the
settings, and the raw entries per dataset.

``--finalize`` (after ``compare_raw_stores_043.py`` says identical) writes the one file the
043 readers need beyond the raw stage, ``processed/gene_set.json`` (the 6,607-gene index,
``sorted(genome.gene_set)`` through ``Neo4jCellDataset._write_json_with_lock``, the writer
and genome setup of ``build_043_store.py``), and stops with an error unless it is byte
identical to ``raw_043/processed/gene_set.json`` and, once job 6187 has written it, to
``raw_043_v2/processed/gene_set.json``. ``--out-dir`` writes it elsewhere (a dry run).

    STORE_043_VERSION=v2 python experiments/043-expression-proteome-morphology/scripts/build_043_raw_partitioned.py --fetch-workers 8 --prefix-length 2
    python experiments/043-expression-proteome-morphology/scripts/build_043_raw_partitioned.py --finalize
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import os.path as osp
import socket
import sys
import time

import lmdb
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from data_shapes_043 import raw_dataset_index  # noqa: E402
from generate_043_cql import PARTITIONED_QUERY_NAME  # noqa: E402

from torchcell.data import Neo4jCellDataset  # noqa: E402
from torchcell.database.connection import neo4j_connection_settings  # noqa: E402
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome  # noqa: E402
from torchcell.utils.paths import experiment_results_dir  # noqa: E402

EXPERIMENT = "043-expression-proteome-morphology"
CANDIDATE_TAG = "raw_043_v2_partitioned"


def _genome(data_root: str) -> SCerevisiaeGenome:
    genome = SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
        overwrite=False,
    )
    genome.drop_empty_go()
    return genome


def finalize(data_root: str, out_dir: str | None) -> dict[str, object]:
    """Write ``processed/gene_set.json`` and check it byte for byte (module docstring)."""
    exp = osp.join(data_root, "data/torchcell/experiments", EXPERIMENT)
    out_dir = out_dir or osp.join(exp, CANDIDATE_TAG, "processed")
    os.makedirs(out_dir, exist_ok=True)
    path = osp.join(out_dir, "gene_set.json")
    Neo4jCellDataset._write_json_with_lock(path, sorted(_genome(data_root).gene_set))
    with open(path, "rb") as f:
        written = f.read()
    checks: dict[str, object] = {
        "path": path,
        "sha256": hashlib.sha256(written).hexdigest(),
    }
    for tag in ("raw_043", "raw_043_v2"):
        other = osp.join(exp, tag, "processed", "gene_set.json")
        if not osp.exists(other):
            checks[tag] = "not written yet"
            continue
        with open(other, "rb") as f:
            same = f.read() == written
        checks[tag] = "byte-identical" if same else "DIFFERENT"
        if not same:
            raise RuntimeError(f"{path} differs from {other}")
    print(json.dumps(checks, indent=1))
    return checks


def main() -> None:
    """Run the partitioned raw stage and write the build record (or ``--finalize``)."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fetch-workers", type=int)
    ap.add_argument("--prefix-length", type=int)
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--out-dir", default=None, help="--finalize: write here instead")
    args = ap.parse_args()
    data_root = os.environ["DATA_ROOT"]
    if args.finalize:
        finalize(data_root, args.out_dir)
        return
    if args.fetch_workers is None or args.prefix_length is None:
        ap.error("--fetch-workers and --prefix-length are required for the build")
    root = osp.join(data_root, "data/torchcell/experiments", EXPERIMENT, CANDIDATE_TAG)
    if osp.exists(osp.join(root, "raw", "lmdb")):
        raise FileExistsError(f"{root}/raw/lmdb exists; the candidate is built once")
    here = osp.dirname(osp.abspath(__file__))
    query_path = osp.abspath(osp.join(here, "..", "queries", PARTITIONED_QUERY_NAME))
    with open(query_path) as f:
        query = f.read()
    settings = neo4j_connection_settings()
    genome = _genome(data_root)
    t0 = time.time()
    raw = Neo4jCellDataset.load_raw(
        settings.uri,
        settings.username,
        settings.password,
        root,
        query,
        genome.gene_set,
        record_observers=[],
        fetch_workers=args.fetch_workers,
        partition_prefix_length=args.prefix_length,
    )
    raw.close_lmdb()
    seconds = time.time() - t0
    env = lmdb.open(osp.join(root, "raw", "lmdb"), readonly=True, lock=False)
    n = int(env.stat()["entries"])
    env.close()
    index = raw_dataset_index(root)
    record = {
        "generated_by": f"experiments/{EXPERIMENT}/scripts/build_043_raw_partitioned.py",
        "root": root,
        "query_file": query_path,
        "neo4j_uri": settings.uri,
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "fetch_workers": args.fetch_workers,
        "partition_prefix_length": args.prefix_length,
        "raw_stage_seconds": seconds,
        "n_records": n,
        "records_per_second": n / seconds,
        "n_records_by_dataset": {k: len(v) for k, v in sorted(index.items())},
    }
    out = osp.join(
        experiment_results_dir(EXPERIMENT, __file__), "build_043_raw_partitioned.json"
    )
    with open(out, "w") as f:
        json.dump(record, f, indent=1)
    print(json.dumps(record, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
