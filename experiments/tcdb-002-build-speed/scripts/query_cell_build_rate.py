# experiments/tcdb-002-build-speed/scripts/query_cell_build_rate.py
# [[experiments.tcdb-002-build-speed.scripts.query_cell_build_rate]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/tcdb-002-build-speed/scripts/query_cell_build_rate
"""Time a whole ``Neo4jCellDataset`` build of one 033 block from the served graph.

The query-build stages were each measured on their own: the single-pass raw stage and
its partitioned form on the served graph (``query_server_rate.py``), the folded
downstream passes on a 21k-record slice held in the page cache
(``cell_single_pass_slice_check.py``). This runs the three together as the 033 build
would: ``Neo4jCellDataset`` on one dataset block of the 033 query, the 033 build's
``GenotypeEnvironmentAggregator`` (imported read-only from the 033-build worktree), no
converter, no deduplicator, ``fetch_workers`` forked fetch workers at the given
``partition_prefix_length``, into a scratch root. On the Hoepfner 2014 block the three
stage stores are 40 GB each, past a 32 GB job's page cache, so the folded passes' I/O
shows as it would at full scale.

Reports wall seconds of each step (the raw stage, the aggregation join pass, the
processed copy, the folded index writer, the parent-side grouping ``accept`` calls)
and the whole build, the process peaks of this process and its largest worker, the
record and group counts, and whether the group count equals ``--expected-groups`` (the
033 build's ``dataset_name_index`` count for that dataset: every 033 group belongs to
one dataset, so a one-block build must reproduce it exactly). Writes
``results/query_cell_build_rate_<dataset>.csv``.

Runs under slurm only (it touches the served store):

    sbatch experiments/tcdb-002-build-speed/scripts/gh_query_cell_build_hoepfner.slurm
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import os.path as osp
import resource
import shutil
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from typing import Any

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from query_build_cost_probe import load_aggregator_class  # noqa: E402
from query_server_rate import RESULTS_DIR, block_query, peak_rss_gb  # noqa: E402

from torchcell.data import neo4j_cell as nc  # noqa: E402
from torchcell.data.graph_processor import SubgraphRepresentation  # noqa: E402
from torchcell.data.neo4j_query_raw import Neo4jQueryRaw  # noqa: E402
from torchcell.sequence import GeneSet  # noqa: E402

TIMES: dict[str, float] = defaultdict(float)


def timer(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``fn`` to add its wall time to ``TIMES[name]``."""

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        t = time.perf_counter()
        out = fn(*args, **kwargs)
        TIMES[name] += time.perf_counter() - t
        return out

    return wrapped


def install_timers(aggregator_cls: type[Any]) -> None:
    """Time the build's steps on the classes themselves."""
    Neo4jQueryRaw.process = timer("raw_stage", Neo4jQueryRaw.process)  # type: ignore[method-assign]
    aggregator_cls.process = timer("aggregator_process", aggregator_cls.process)
    for name in ("_copy_lmdb", "_write_folded_indices"):
        setattr(
            nc.Neo4jCellDataset, name, timer(name, getattr(nc.Neo4jCellDataset, name))
        )
    nc.RawStageGrouping.accept = timer(  # type: ignore[method-assign]
        "grouping_accept", nc.RawStageGrouping.accept
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="served Dataset id")
    parser.add_argument("--gene-set", required=True, help="JSON list of genes")
    parser.add_argument("--root", required=True, help="build root (created)")
    parser.add_argument("--fetch-workers", type=int, default=8)
    parser.add_argument("--prefix-length", type=int, default=2)
    parser.add_argument(
        "--expected-groups",
        type=int,
        default=None,
        help="the 033 build's dataset_name_index count for this dataset",
    )
    parser.add_argument(
        "--cleanup", action="store_true", help="remove the build root at the end"
    )
    args = parser.parse_args()

    with open(args.gene_set) as fh:
        genes = GeneSet(json.load(fh))
    aggregator_cls = load_aggregator_class()
    install_timers(aggregator_cls)
    os.makedirs(args.root, exist_ok=False)

    t = time.perf_counter()
    dataset = nc.Neo4jCellDataset(
        root=args.root,
        query=block_query(args.dataset),  # carries the {partition} marker
        gene_set=genes,
        graphs=None,
        incidence_graphs=None,
        node_embeddings=None,
        converter=None,
        deduplicator=None,
        aggregator=aggregator_cls,
        graph_processor=SubgraphRepresentation(),
        uri=os.environ["NEO4J_URI"],
        username=os.environ["NEO4J_USER"],
        password=os.environ["NEO4J_PASSWORD"],
        fetch_workers=args.fetch_workers,
        partition_prefix_length=args.prefix_length,
    )
    TIMES["dataset_build_total"] = time.perf_counter() - t
    groups = len(dataset)
    with open(osp.join(dataset.raw_dir, "experiment_reference_index.json")) as fh:
        records = sum(len(entry["member_indices"]) for entry in json.load(fh))
    by_dataset = {k: len(v) for k, v in dataset.dataset_name_index.items()}
    label_rows = len(dataset.label_df)
    dataset.close_lmdb()
    print(
        f"{args.dataset}: {records:,} records -> {groups:,} groups "
        f"(expected {args.expected_groups}), dataset_name_index {by_dataset}, "
        f"label_df rows {label_rows:,}",
        flush=True,
    )
    matches = args.expected_groups is None or groups == args.expected_groups

    rows = [
        {
            "dataset": args.dataset,
            "fetch_workers": args.fetch_workers,
            "prefix_length": args.prefix_length,
            "step": step,
            "seconds": round(seconds, 1),
            "ms_per_record": round(seconds / records * 1e3, 4),
            "records": records,
            "groups": groups,
            "groups_match_033": matches,
            "parent_peak_gb": round(peak_rss_gb(resource.RUSAGE_SELF), 2),
            "worker_peak_gb": round(peak_rss_gb(resource.RUSAGE_CHILDREN), 2),
        }
        for step, seconds in sorted(TIMES.items())
    ]
    for row in rows:
        print(
            f"  {row['step']:24s} {row['seconds']:9.1f} s  {row['ms_per_record']:.4f} ms/record"
        )
    print(
        f"peak parent {rows[0]['parent_peak_gb']} GB, largest worker {rows[0]['worker_peak_gb']} GB"
    )
    out = osp.join(RESULTS_DIR, f"query_cell_build_rate_{args.dataset}.csv")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")
    if args.cleanup:
        shutil.rmtree(args.root)
    if not matches:
        raise SystemExit(
            f"{groups} groups, the 033 build has {args.expected_groups} for this dataset"
        )


if __name__ == "__main__":
    main()
