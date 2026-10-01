# experiments/tcdb-002-build-speed/scripts/single_pass_slice_check.py
# [[experiments.tcdb-002-build-speed.scripts.single_pass_slice_check]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/tcdb-002-build-speed/scripts/single_pass_slice_check
"""The single-pass ``Neo4jQueryRaw.process`` against the previous raw stage, on 033 records.

Takes the probe's slice of the 033 pooled build (``query_build_cost_probe.read_slice``:
20,000 evenly spaced processed groups, flattened to their records), stands the records
in for a Cypher result in both layouts (environment inline, as the served store holds
it, and the interned ``$ref`` pointer layout), and runs

* OLD: the previous per-record writer (both halves validated per record, written with
  ``json.dumps(..., default=model_dump)``), then the two streaming passes the old
  ``process`` ran after it (reference index, gene set), and
* NEW: ``Neo4jQueryRaw.process`` from this worktree, through a subclass whose
  ``fetch_data``/``fetch_constants`` serve the slice from memory (no bolt).

It asserts that every LMDB value, ``experiment_reference_index.json`` and
``gene_set.json`` are byte-identical, and reports ms per record of each (minimum of
``REPEATS`` runs, each into a fresh scratch directory). Writes
``results/single_pass_slice_check.csv``.

    nice -n 10 ~/miniconda3/envs/torchcell/bin/python \
        experiments/tcdb-002-build-speed/scripts/single_pass_slice_check.py
"""

from __future__ import annotations

import csv
import gc
import json
import os
import os.path as osp
import sys
import tempfile
import time
from collections.abc import Iterator
from typing import Any

import lmdb

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from query_build_cost_probe import (  # noqa: E402
    RESULTS_DIR,
    SAMPLE_GROUPS,
    SCRATCH,
    read_slice,
)

from torchcell.data import neo4j_query_raw as nqr  # noqa: E402
from torchcell.data.data import (  # noqa: E402
    ExperimentReferenceIndex,
    compute_sha256_hash,
)
from torchcell.datamodels.interned_constant import (  # noqa: E402
    collect_pointers,
    resolve_pointers,
    split_experiment_dump,
    verified_constant,
)
from torchcell.datamodels.schema import (  # noqa: E402
    EXPERIMENT_REFERENCE_TYPE_MAP,
    EXPERIMENT_TYPE_MAP,
)
from torchcell.sequence import GeneSet  # noqa: E402

REPEATS = 3


class SliceQueryRaw(nqr.Neo4jQueryRaw):
    """``Neo4jQueryRaw`` serving in-memory records and constants instead of bolt."""

    records: list[dict[str, str]]
    store: dict[str, str]
    write_s: float
    last_stream: Any

    def __attrs_post_init__(self) -> None:
        """Set up paths without querying; ``process`` opens its own staging store."""
        self.raw_dir = osp.join(self.root_dir, "raw")
        self.lmdb_dir = osp.join(self.raw_dir, "lmdb")
        os.makedirs(self.raw_dir, exist_ok=True)
        self.write_s = 0.0

    def _write_batch(
        self, batch: list[tuple[int, dict[str, Any], str]], constants: dict[str, Any]
    ) -> None:
        """Time the batch writes and keep the stream state ``process`` resets."""
        t = time.perf_counter()
        super()._write_batch(batch, constants)
        self.write_s += time.perf_counter() - t
        self.last_stream = self._stream

    def fetch_data(self) -> Iterator[Any]:
        """Yield the slice records."""
        yield from self.records

    def fetch_constants(self, refs: list[str]) -> dict[str, Any]:
        """Serve and verify interned constants from memory."""
        return {r: verified_constant(r, self.store[r]) for r in refs}


def make_query(
    records: list[dict[str, str]], store: dict[str, str], tag: str
) -> SliceQueryRaw:
    """A query over a fresh scratch directory."""
    query = SliceQueryRaw(
        uri="bolt://none",
        username="",
        password="",
        root_dir=tempfile.mkdtemp(prefix=f"spc_{tag}_", dir=SCRATCH),
        query="",
    )
    query.records = records
    query.store = store
    return query


def old_build(query: SliceQueryRaw) -> tuple[float, float]:
    """The previous raw stage; returns (write seconds, two-pass seconds)."""
    t = time.perf_counter()
    constants: dict[str, Any] = {}
    batch: list[tuple[int, dict[str, Any], dict[str, Any]]] = []

    def flush() -> None:
        refs: set[str] = set()
        for _, e, r in batch:
            collect_pointers(e, refs)
            collect_pointers(r, refs)
        unseen = sorted(refs - constants.keys())
        if unseen:
            constants.update(query.fetch_constants(unseen))
        with query.env.begin(write=True) as txn:
            for i, e, r in batch:
                e = resolve_pointers(e, constants)
                r = resolve_pointers(r, constants)
                experiment = EXPERIMENT_TYPE_MAP[e["experiment_type"]](
                    dataset_name=e["dataset_name"],
                    genotype=e["genotype"],
                    environment=e["environment"],
                    phenotype=e["phenotype"],
                )
                reference = EXPERIMENT_REFERENCE_TYPE_MAP[
                    r["experiment_reference_type"]
                ](**r)
                data_json = json.dumps(
                    {"experiment": experiment, "experiment_reference": reference},
                    default=lambda o: o.model_dump(),
                )
                txn.put(f"data_{i}".encode(), data_json.encode())

    for i, record in enumerate(query.records):
        batch.append(
            (
                i,
                json.loads(record["e_serialized"]),
                json.loads(record["ref_serialized"]),
            )
        )
        if len(batch) >= nqr.PROCESS_BATCH:
            flush()
            batch = []
    if batch:
        flush()
    query.close_lmdb()
    write_s = time.perf_counter() - t

    t = time.perf_counter()
    env = lmdb.open(query.lmdb_dir, readonly=True, lock=False, readahead=False)
    hash_to_indices: dict[str, list[int]] = {}
    with env.begin() as txn:
        for key, value in txn.cursor():
            idx = int(key.decode().split("_")[1])
            ref = json.loads(value.decode("utf-8"))["experiment_reference"]
            h = compute_sha256_hash(json.dumps(ref, sort_keys=True))
            hash_to_indices.setdefault(h, []).append(idx)
    env.close()
    index = [
        ExperimentReferenceIndex(
            reference=query[indices[0]]["experiment_reference"].model_dump(),
            member_indices=sorted(indices),
        )
        for indices in hash_to_indices.values()
    ]
    with open(osp.join(query.raw_dir, "experiment_reference_index.json"), "w") as f:
        json.dump([eri.model_dump() for eri in index], f)
    query.close_lmdb()
    genes = GeneSet()
    env = lmdb.open(query.lmdb_dir, readonly=True, lock=False, readahead=False)
    with env.begin() as txn:
        for _key, value in txn.cursor():
            for g in nqr.Neo4jQueryRaw.extract_systematic_gene_names(
                json.loads(value.decode("utf-8"))["experiment"]["genotype"]
            ):
                genes.add(g)
    env.close()
    with open(osp.join(query.raw_dir, "gene_set.json"), "w") as f:
        json.dump(list(sorted(genes)), f, indent=0)
    return write_s, time.perf_counter() - t


def outputs(query: SliceQueryRaw) -> tuple[list[tuple[bytes, bytes]], bytes, bytes]:
    """LMDB items in cursor order and the two index files' bytes."""
    env = lmdb.open(query.lmdb_dir, readonly=True, lock=False)
    with env.begin() as txn:
        items = [(bytes(k), bytes(v)) for k, v in txn.cursor()]
    env.close()
    with open(osp.join(query.raw_dir, "experiment_reference_index.json"), "rb") as f:
        index = f.read()
    with open(osp.join(query.raw_dir, "gene_set.json"), "rb") as f:
        genes = f.read()
    return items, index, genes


def main() -> None:
    """Compare old and new raw stages on the slice in both layouts; write the CSV."""
    groups, _ = read_slice(SAMPLE_GROUPS)
    records = [r for g in groups for r in g]
    n = len(records)
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
    print(f"slice: {n:,} records; {len(store):,} interned environment constants")

    rows: list[list[str]] = []
    for layout, recs, st in (("inline", inline, {}), ("pointer", pointer, store)):
        old_write, old_passes, new_total, new_write, warm_write = [], [], [], [], []
        old_out = new_out = None
        for _ in range(REPEATS):
            gc.collect()
            q_old = make_query(recs, st, f"old_{layout}")
            w, p = old_build(q_old)
            old_write.append(w)
            old_passes.append(p)
            old_out = outputs(q_old)
            gc.collect()
            q_new = make_query(recs, st, f"new_{layout}")
            t = time.perf_counter()
            q_new.process()
            new_total.append(time.perf_counter() - t)
            new_write.append(q_new.write_s)
            q_new.close_lmdb()
            new_out = outputs(q_new)
            # steady state: the same records again with every constant cached, as
            # nearly all records are at full scale (11,573 constants in 6.39M records)
            gc.collect()
            q_warm = make_query(recs, st, f"warm_{layout}")
            q_warm._stream = nqr._StreamState(
                environments=q_new.last_stream.environments,
                references=q_new.last_stream.references,
            )
            constants = {k: json.loads(v) for k, v in st.items()}
            items = [
                (i, json.loads(r["e_serialized"]), r["ref_serialized"])
                for i, r in enumerate(recs)
            ]
            t = time.perf_counter()
            for b in range(0, n, nqr.PROCESS_BATCH):
                q_warm._write_batch(items[b : b + nqr.PROCESS_BATCH], constants)
            warm_write.append(time.perf_counter() - t)
            q_warm.close_lmdb()
        assert old_out is not None and new_out is not None
        same_values = sum(a == b for a, b in zip(old_out[0], new_out[0]))
        assert len(old_out[0]) == len(new_out[0]) == n
        assert same_values == n, f"{layout}: {n - same_values} LMDB values differ"
        assert old_out[1] == new_out[1], f"{layout}: reference index file differs"
        assert old_out[2] == new_out[2], f"{layout}: gene set file differs"
        n_refs = len(json.loads(old_out[1]))
        n_genes = len(json.loads(old_out[2]))
        ms = {
            "old_write": min(old_write) / n * 1e3,
            "old_write_plus_two_passes": min(
                w + p for w, p in zip(old_write, old_passes)
            )
            / n
            * 1e3,
            "new_process": min(new_total) / n * 1e3,
            "new_write_batches": min(new_write) / n * 1e3,
            "new_index_and_gene_set_files": min(
                a - b for a, b in zip(new_total, new_write)
            )
            / n
            * 1e3,
            "new_write_batches_warm_cache": min(warm_write) / n * 1e3,
        }
        print(
            f"{layout}: {same_values}/{n} LMDB values byte-identical; "
            f"experiment_reference_index.json identical ({n_refs} references, "
            f"{len(old_out[1]):,} bytes); gene_set.json identical ({n_genes} genes); "
            + ", ".join(f"{k} {v:.4f} ms/record" for k, v in ms.items())
        )
        for k, v in ms.items():
            rows.append(
                [
                    layout,
                    k,
                    f"{v:.4f}",
                    str(n),
                    str(same_values),
                    str(n_refs),
                    str(n_genes),
                ]
            )
    path = osp.join(RESULTS_DIR, "single_pass_slice_check.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "layout",
                "path",
                "ms_per_record_min_of_3",
                "records",
                "values_byte_identical",
                "references",
                "genes",
            ]
        )
        w.writerows(rows)
    print(path)


if __name__ == "__main__":
    main()
