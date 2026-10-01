# experiments/tcdb-002-build-speed/scripts/cell_single_pass_slice_check.py
# [[experiments.tcdb-002-build-speed.scripts.cell_single_pass_slice_check]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/tcdb-002-build-speed/scripts/cell_single_pass_slice_check
"""Stage 2 on 033 records: grouping during the raw stage, folded processed-store passes.

Builds ``Neo4jCellDataset`` twice over the probe's slice of the 033 pooled build
(``query_build_cost_probe.read_slice``: 20,000 evenly spaced processed groups, flattened
to their records), with the 033 build's aggregator (``GenotypeEnvironmentAggregator``,
imported read-only from the 033-build worktree), no converter, no deduplicator, through
a ``Neo4jQueryRaw`` subclass serving the slice from memory (no bolt):

* OLD: the raw stage's grouping withheld (``raw_stage_ran`` False), so aggregation runs
  its own pass 1 and the processed store's passes (``compute_phenotype_info``,
  ``label_df``, three indices) each walk the store, as before stage 2;
* NEW: this worktree's path, the grouping computed by the raw stage's observer and
  the processed-store files written from its summaries.

Asserts every key and value of the raw, aggregation and processed LMDBs and every index
and label file byte-identical, in both Cypher layouts, and reports ms per record of each
timed step (minimum of ``REPEATS`` builds, each into a fresh scratch directory; the
slice's stores sit in the page cache, so these are CPU costs). Writes
``results/cell_single_pass_slice_check.csv``.

    nice -n 10 ~/miniconda3/envs/torchcell/bin/python \
        experiments/tcdb-002-build-speed/scripts/cell_single_pass_slice_check.py
"""

from __future__ import annotations

import csv
import gc
import json
import os.path as osp
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from typing import Any

import lmdb

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from query_build_cost_probe import (  # noqa: E402
    FULL_RECORDS,
    RESULTS_DIR,
    SAMPLE_GROUPS,
    SCRATCH,
    load_aggregator_class,
    read_slice,
)

from torchcell.data import neo4j_cell as nc  # noqa: E402
from torchcell.data.aggregate import Aggregator  # noqa: E402
from torchcell.data.neo4j_query_raw import Neo4jQueryRaw, RecordObserver  # noqa: E402
from torchcell.datamodels.interned_constant import (  # noqa: E402
    split_experiment_dump,
    verified_constant,
)
from torchcell.sequence import GeneSet  # noqa: E402

REPEATS = 3
TIMES: dict[str, float] = defaultdict(float)


class SliceQueryRaw(Neo4jQueryRaw):
    """``Neo4jQueryRaw`` whose Cypher result and constants come from class state."""

    RECORDS: list[dict[str, str]] = []
    STORE: dict[str, str] = {}

    def fetch_data(self) -> Iterator[Any]:
        """Yield the slice records."""
        yield from type(self).RECORDS

    def fetch_constants(self, refs: list[str]) -> dict[str, Any]:
        """Serve and verify interned constants from memory."""
        return {r: verified_constant(r, type(self).STORE[r]) for r in refs}


def timer(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``fn`` to add its wall time to ``TIMES[name]``."""

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        t = time.perf_counter()
        out = fn(*args, **kwargs)
        TIMES[name] += time.perf_counter() - t
        return out

    return wrapped


def install_timers(aggregator_cls: type[Aggregator]) -> None:
    """Time each stage of interest on the classes themselves."""
    nc.Neo4jCellDataset.compute_phenotype_info = timer(  # type: ignore[method-assign]
        "compute_phenotype_info", nc.Neo4jCellDataset.compute_phenotype_info
    )
    label_df = nc.Neo4jCellDataset.label_df
    assert isinstance(label_df, property) and label_df.fget is not None
    nc.Neo4jCellDataset.label_df = property(  # type: ignore[method-assign,assignment]
        timer("label_df", label_df.fget)
    )
    for name in (
        "compute_phenotype_label_index",
        "compute_dataset_name_index",
        "compute_perturbation_count_index",
        "_write_folded_indices",
        "_copy_lmdb",
    ):
        setattr(
            nc.Neo4jCellDataset, name, timer(name, getattr(nc.Neo4jCellDataset, name))
        )
    aggregator_cls.process = timer("aggregator_process", aggregator_cls.process)  # type: ignore[method-assign]
    nc.RawStageGrouping.__call__ = timer(  # type: ignore[method-assign]
        "raw_stage_observer", nc.RawStageGrouping.__call__
    )
    SliceQueryRaw.process = timer("raw_stage", SliceQueryRaw.process)  # type: ignore[method-assign]


def build(
    root: str, withhold: bool, aggregator_cls: type[Aggregator], genes: GeneSet
) -> nc.Neo4jCellDataset:
    """Build the dataset with the slice; ``withhold`` gives the pre-stage-2 path."""

    def load_raw(
        uri: str,
        username: str,
        password: str,
        root_dir: str,
        query: str,
        gene_set: GeneSet,
        record_observers: list[RecordObserver],
        fetch_workers: int,
        partition_prefix_length: int,
    ) -> Neo4jQueryRaw:
        raw = SliceQueryRaw(
            uri=uri,
            username=username,
            password=password,
            root_dir=root_dir,
            query=query,
            record_observers=list(record_observers),
            fetch_workers=fetch_workers,
            partition_prefix_length=partition_prefix_length,
        )
        if withhold:
            raw.raw_stage_ran = False
        return raw

    nc.Neo4jCellDataset.load_raw = staticmethod(load_raw)  # type: ignore[method-assign,assignment]
    return nc.Neo4jCellDataset(
        root=root,
        query="slice",
        gene_set=genes,
        aggregator=aggregator_cls,
        uri="bolt://none",
        username="",
        password="",
    )


def lmdb_items(path: str) -> list[tuple[bytes, bytes]]:
    """Every key and value in cursor order."""
    env = lmdb.open(path, readonly=True, lock=False)
    with env.begin() as txn:
        items = [(bytes(k), bytes(v)) for k, v in txn.cursor()]
    env.close()
    return items


FILES = [
    ("raw", "experiment_reference_index.json"),
    ("raw", "gene_set.json"),
    ("processed", "experiment_types.json"),
    ("processed", "label_df.parquet"),
    ("processed", "phenotype_label_index.json"),
    ("processed", "dataset_name_index.json"),
    ("processed", "perturbation_count_index.json"),
]


def main() -> None:
    """Old vs new on the slice, both layouts; assert identical outputs, write timings."""
    aggregator_cls = load_aggregator_class()
    install_timers(aggregator_cls)
    groups, _ = read_slice(SAMPLE_GROUPS)
    records = [r for g in groups for r in g]
    n = len(records)
    genes = GeneSet(
        {
            p["systematic_gene_name"]
            for r in records
            for p in r["experiment"]["genotype"]["perturbations"]
        }
    )
    inline = [
        {
            "e_serialized": json.dumps(r["experiment"]),
            "ref_serialized": json.dumps(r["experiment_reference"]),
        }
        for r in records
    ]
    store: dict[str, str] = {}
    pointer = []
    for r, rec in zip(records, inline):
        pointered, constants = split_experiment_dump(r["experiment"])
        store.update({cid: payload for cid, _, payload in constants})
        pointer.append(
            {
                "e_serialized": json.dumps(pointered),
                "ref_serialized": rec["ref_serialized"],
            }
        )
    print(f"slice: {n:,} records, {len(genes):,} genes")

    rows: list[list[str]] = []
    for layout, recs, st in (("inline", inline, {}), ("pointer", pointer, store)):
        SliceQueryRaw.RECORDS = recs
        SliceQueryRaw.STORE = st
        runs: dict[str, list[dict[str, float]]] = {"old": [], "new": []}
        outputs: dict[str, Any] = {}
        for _ in range(REPEATS):
            for path in ("old", "new"):
                gc.collect()
                TIMES.clear()
                root = tempfile.mkdtemp(prefix=f"cspc_{path}_{layout}_", dir=SCRATCH)
                t = time.perf_counter()
                ds = build(root, path == "old", aggregator_cls, genes)
                TIMES["dataset_build_total"] = time.perf_counter() - t
                runs[path].append(dict(TIMES))
                outputs[path] = (
                    {
                        stage: lmdb_items(osp.join(root, stage, "lmdb"))
                        for stage in ("raw", "aggregation")
                    }
                    | {"processed": lmdb_items(osp.join(ds.processed_dir, "lmdb"))},
                    {
                        name: open(
                            osp.join(
                                ds.raw_dir if d == "raw" else ds.processed_dir, name
                            ),
                            "rb",
                        ).read()
                        for d, name in FILES
                    },
                )
        old_lmdbs, old_files = outputs["old"]
        new_lmdbs, new_files = outputs["new"]
        for stage, items in old_lmdbs.items():
            assert new_lmdbs[stage] == items, f"{layout}: {stage} LMDB differs"
        for name, data in old_files.items():
            assert new_files[name] == data, f"{layout}: {name} differs"
        n_groups = len(old_lmdbs["aggregation"])
        print(
            f"{layout}: raw ({len(old_lmdbs['raw'])}), aggregation ({n_groups}) and "
            f"processed LMDBs identical key for key; "
            + ", ".join(name for _, name in FILES)
            + " byte-identical"
        )
        steps = sorted({k for r in runs["old"] + runs["new"] for k in r})
        for path in ("old", "new"):
            for step in steps:
                values = [r[step] for r in runs[path] if step in r]
                if not values:
                    continue
                ms = min(values) / n * 1e3
                rows.append(
                    [
                        layout,
                        path,
                        step,
                        f"{ms:.4f}",
                        f"{ms * FULL_RECORDS / 3.6e6:.3f}",
                        str(n),
                        str(n_groups),
                    ]
                )
                print(f"  {layout} {path:3s} {step:34s} {ms:8.4f} ms/record")
    path = osp.join(RESULTS_DIR, "cell_single_pass_slice_check.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "layout",
                "path",
                "step",
                "ms_per_record_min_of_3",
                "cpu_hours_at_6.39M",
                "records",
                "groups",
            ]
        )
        w.writerows(rows)
    print(path)


if __name__ == "__main__":
    main()
