# experiments/tcdb-002-build-speed/scripts/query_build_cost_probe.py
# [[experiments.tcdb-002-build-speed.scripts.query_build_cost_probe]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/tcdb-002-build-speed/scripts/query_build_cost_probe
"""Per-record cost of every client-side step of the Neo4j query build, and a single pass.

The query build (``Neo4jQueryRaw.process`` then ``Neo4jCellDataset.process``) turns a
Cypher result into the training LMDB. This probe replays its client-side steps on a slice
of the 033 pooled chemogenomic build (slurm job 2929, 6,394,540 records in 6,042,771
aggregated groups, 7 h 48 min) WITHOUT touching Neo4j:

* The slice. The raw LMDB of that build was removed by ``overwrite_intermediates``, so
  the records come from the final processed LMDB, whose values are JSON lists of the same
  ``{"experiment", "experiment_reference"}`` records the raw stage wrote. ``SAMPLE_GROUPS``
  evenly spaced group keys are read and every member kept, which is an equal-probability
  sample of records. The Cypher strings ``e_serialized``/``ref_serialized`` are stood in
  for by ``json.dumps`` of the stored halves (the 033 graph's inline layout), and by the
  pointer layout of ``interned_constant.split_experiment_dump`` (the next graph's).
* Each step is timed alone over the whole slice, then in the combination the pipeline
  runs. The statistic is the MINIMUM over ``REPEATS`` runs: the node is shared (load
  average 70 to 86 on 128 cores during development runs) and interference only adds
  time, so the minimum is the least-contaminated estimate; it excludes no work the step
  does (garbage collection runs on every repeat alike). LMDB writes go to a scratch LMDB in 1,000-record
  transactions, as ``Neo4jQueryRaw._write_batch`` does.
* Downstream passes run off scratch LMDBs written from the slice, with the real
  ``GenotypeEnvironmentAggregator`` of the 033 build (read-only, from its worktree).
* A single-pass prototype writes the raw LMDB and computes the reference index, gene
  set, aggregation keys, group order, phenotype/dataset/perturbation-count indices and
  the label table in the same loop, validating each distinct environment and reference
  once. Its outputs are checked against the pipeline-style outputs on the slice: raw
  LMDB values byte-identical, every index equal.
* Cold reads of the real processed LMDB in cursor order and in key order, with and
  without readahead, so the gap between these CPU costs and job 2929's stage times can
  be attributed to I/O.
* The whole processed LMDB is streamed (3 workers) to count distinct environment and
  reference JSON strings, so the single pass is projected with the full-scale number of
  constants rather than the slice's.

Writes ``results/query_build_cost_probe.csv`` (step, ms_per_record,
projected_hours_at_6.39M, observed_hours_job2929, note).

    nice -n 10 ~/miniconda3/envs/torchcell/bin/python \
        experiments/tcdb-002-build-speed/scripts/query_build_cost_probe.py
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import os
import os.path as osp
import statistics
import tempfile
import time
import warnings
from collections.abc import Callable, Iterable
from multiprocessing import get_context
from typing import Any

import lmdb
import numpy as np
from dotenv import load_dotenv

WORKTREE = (
    "/home/michaelvolk/Documents/projects/torchcell.worktrees/feat/tcdb-002-build-speed"
)
load_dotenv(osp.join(WORKTREE, ".env"))

from torchcell.data.data import compute_sha256_hash  # noqa: E402
from torchcell.datamodels.identity import (  # noqa: E402
    _canonical,
    environment_identity,
    identity_sha256,
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

PROCESSED_LMDB = (
    "/scratch/projects/torchcell-scratch/data/torchcell/experiments/"
    "033-env-chemgen-pooled/001-pooled-build/processed/lmdb"
)
AGGREGATOR_FILE = (
    "/home/michaelvolk/Documents/projects/torchcell.worktrees/033-build/"
    "torchcell/data/genotype_environment_aggregate.py"
)
SCRATCH = (
    "/scratch/tmp/claude-1000/-home-michaelvolk-Documents-projects-torchcell/"
    "93b444d6-f496-44d4-ad33-c18ef14a65ff/scratchpad"
)
RESULTS_DIR = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "results")
SAMPLE_GROUPS = 20_000
FULL_RECORDS = 6_394_540
FULL_GROUPS = 6_042_771
REPEATS = 5
PROCESS_BATCH = 1000
FULL_WORKERS = 3

# Wall time of each stage of job 2929 (hours): tqdm totals from
# experiments/033-env-chemgen-pooled/slurm/output/2929_033-query-build.out in the 033-build
# worktree, and, for the stages without a progress bar, the gap between the mtimes of the
# files each stage writes (processed/lmdb/data.mdb 03:49:00, experiment_types.json
# 04:06:48, label_df.parquet 04:48:02, phenotype_label_index.json 05:06:01,
# dataset_name_index.json 05:24:06, perturbation_count_index.json 05:42:26, build
# finished 06:00:00).
OBSERVED_H: dict[str, float] = {
    "raw_stage_033": (2 * 3600 + 28 * 60 + 50) / 3600,
    "reference_index_pass": (26 * 60 + 6) / 3600,
    "gene_set_pass": (36 * 60 + 8) / 3600,
    "aggregation_pass1_grouping": (1 * 3600 + 38 * 60 + 47) / 3600,
    "aggregation_pass2_write": (21 * 60 + 20) / 3600,
    "processed_copy": (4 * 60 + 55) / 3600,
    "compute_phenotype_info_pass": (17 * 60 + 45) / 3600,
    "label_df_pass": (41 * 60 + 14) / 3600,
    "phenotype_label_index_pass": (17 * 60 + 58) / 3600,
    "dataset_name_index_pass": (18 * 60 + 5) / 3600,
    "perturbation_count_index_pass": (18 * 60 + 20) / 3600,
    "measurements_per_entry_pass": (17 * 60 + 34) / 3600,
}


def load_aggregator_class() -> type[Any]:
    """Import ``GenotypeEnvironmentAggregator`` from the 033-build worktree, read-only."""
    spec = importlib.util.spec_from_file_location(
        "genotype_environment_aggregate_033", AGGREGATOR_FILE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GenotypeEnvironmentAggregator  # type: ignore[no-any-return]


def read_slice(count: int) -> tuple[list[list[dict[str, Any]]], list[bytes]]:
    """Read ``count`` evenly spaced groups; return parsed groups and raw values."""
    keys = np.linspace(0, FULL_GROUPS - 1, count).round().astype(int)
    env = lmdb.open(PROCESSED_LMDB, readonly=True, lock=False, readahead=False)
    groups: list[list[dict[str, Any]]] = []
    values: list[bytes] = []
    with env.begin() as txn:
        for k in keys:
            value = bytes(txn.get(str(int(k)).encode()))
            values.append(value)
            groups.append(json.loads(value))
    env.close()
    return groups, values


def timed(fn: Callable[[], Any], n: int, repeats: int = REPEATS) -> float:
    """Minimum over ``repeats`` of ``fn()`` wall time, in ms per item for ``n`` items."""
    samples = []
    for _ in range(repeats):
        gc.collect()
        t = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t) / n * 1e3)
    return min(samples)


_LATEST: dict[str, str] = {}


def fresh_env(slot: str) -> Any:
    """Open a new, empty scratch LMDB for ``slot`` (a fresh directory every call).

    py-lmdb 1.7.5 segfaults in ``open_db`` when a non-empty environment is closed and
    reopened for writing in the same process, so a slot is never reused.
    """
    path = tempfile.mkdtemp(prefix=osp.basename(slot) + "_", dir=osp.dirname(slot))
    _LATEST[slot] = path
    return lmdb.open(path, map_size=int(2e10), readahead=False, meminit=False)


def latest(slot: str) -> str:
    """Directory of the most recent LMDB written for ``slot``."""
    return _LATEST[slot]


def put_batched(env: Any, items: Iterable[tuple[bytes, bytes]]) -> None:
    """Write key/value pairs in ``PROCESS_BATCH``-sized transactions."""
    batch: list[tuple[bytes, bytes]] = []
    for kv in items:
        batch.append(kv)
        if len(batch) >= PROCESS_BATCH:
            with env.begin(write=True) as txn:
                for k, v in batch:
                    txn.put(k, v)
            batch = []
    if batch:
        with env.begin(write=True) as txn:
            for k, v in batch:
                txn.put(k, v)


def dump_record(experiment: Any, reference: Any) -> str:
    """The raw-stage serialization: ``json.dumps`` with the ``model_dump`` default."""
    return json.dumps(
        {"experiment": experiment, "experiment_reference": reference},
        default=lambda o: o.model_dump(),
    )


def validate_pair(e: dict[str, Any], r: dict[str, Any]) -> tuple[Any, Any]:
    """Build the experiment and reference exactly as ``_write_batch`` does."""
    experiment = EXPERIMENT_TYPE_MAP[e["experiment_type"]](
        dataset_name=e["dataset_name"],
        genotype=e["genotype"],
        environment=e["environment"],
        phenotype=e["phenotype"],
    )
    reference = EXPERIMENT_REFERENCE_TYPE_MAP[r["experiment_reference_type"]](**r)
    return experiment, reference


def splice_experiment(experiment: Any, env_fragment: str) -> str:
    """``json.dumps(experiment.model_dump())`` with the environment's cached JSON spliced in."""
    dump = experiment.model_dump(exclude={"environment"})
    parts = []
    for name in type(experiment).model_fields:
        if name == "environment":
            parts.append(f'"environment": {env_fragment}')
        else:
            parts.append(f"{json.dumps(name)}: {json.dumps(dump[name])}")
    return "{" + ", ".join(parts) + "}"


class SinglePassResult:
    """Everything the single pass computes beside the raw LMDB."""

    def __init__(self) -> None:
        """Empty indices."""
        self.ref_hash_to_indices: dict[str, list[int]] = {}
        self.gene_set: set[str] = set()
        self.agg_key_to_group: dict[str, int] = {}
        self.group_members: list[list[int]] = []
        # per record: (label_name, dataset_name, perturbation count, scalar label | None)
        self.per_record: list[tuple[str, str, int, float | None]] = []
        self.experiment_types: set[str] = set()
        self.raw_label_index: dict[str, list[int]] = {}
        # finalized in pipeline group order (see finalize)
        self.groups: list[list[int]] = []
        self.phenotype_label_index: dict[str, list[int]] = {}
        self.dataset_name_index: dict[str, list[int]] = {}
        self.perturbation_count_index: dict[int, list[int]] = {}
        self.label_rows: dict[int, dict[str, float]] = {}

    def finalize(self) -> None:
        """Number groups and order members exactly as the pipeline does.

        ``Aggregator.process`` pass 1 walks the raw LMDB cursor, whose order is
        lexicographic in the key (``data_0, data_1, data_10, ...``), so a group's index
        is its rank by the lexicographically smallest member key, members are in key
        order, and the label table's last-non-NaN-wins rule runs in that order.
        """
        lex = [sorted(m, key=lambda i: f"data_{i}") for m in self.group_members]
        lex.sort(key=lambda m: f"data_{m[0]}")
        self.groups = lex
        pli: dict[str, set[int]] = {}
        dni: dict[str, set[int]] = {}
        pci: dict[int, set[int]] = {}
        for g, members in enumerate(lex):
            row: dict[str, float] = {}
            for i in members:
                label_name, dataset_name, n_perts, value = self.per_record[i]
                pli.setdefault(label_name, set()).add(g)
                dni.setdefault(dataset_name, set()).add(g)
                pci.setdefault(n_perts, set()).add(g)
                if value is not None:
                    row[label_name] = value
            self.label_rows[g] = row
        self.phenotype_label_index = {k: sorted(v) for k, v in pli.items()}
        self.dataset_name_index = {k: sorted(v) for k, v in dni.items()}
        self.perturbation_count_index = {k: sorted(v) for k, v in pci.items()}


class ConstantCache:
    """Validated environments and references, each built once per distinct content."""

    def __init__(self) -> None:
        """Empty caches."""
        self.env_by_key: dict[str, tuple[Any, str, dict[str, Any]]] = {}
        self.ref_by_str: dict[str, tuple[Any, str, str, str]] = {}

    def environment(
        self, key: str, env_dict: dict[str, Any], env_cls: type[Any]
    ) -> tuple[Any, str, dict[str, Any]]:
        """Return (model, JSON fragment, identity projection) for an environment."""
        hit = self.env_by_key.get(key)
        if hit is None:
            model = env_cls(**env_dict)
            hit = (model, json.dumps(model.model_dump()), environment_identity(model))
            self.env_by_key[key] = hit
        return hit

    def reference(self, ref_str: str) -> tuple[Any, str, str, str]:
        """Return (model, JSON fragment, index hash, ploidy) for a reference string."""
        hit = self.ref_by_str.get(ref_str)
        if hit is None:
            r = json.loads(ref_str)
            model = EXPERIMENT_REFERENCE_TYPE_MAP[r["experiment_reference_type"]](**r)
            dump = model.model_dump()
            hit = (
                model,
                json.dumps(dump),
                compute_sha256_hash(json.dumps(dump, sort_keys=True)),
                dump["genome_reference"]["ploidy"],
            )
            self.ref_by_str[ref_str] = hit
        return hit


def single_pass(
    strs: list[tuple[str, str]],
    layout: str,
    constants: dict[str, str],
    cache: ConstantCache,
    env: Any,
) -> SinglePassResult:
    """One loop: write the raw LMDB and compute every index the build needs.

    ``layout`` is ``inline`` (environment inside ``e_serialized``, keyed by its JSON) or
    ``interned`` (a ``$ref`` pointer, keyed by its id; the payload verified once).
    """
    out = SinglePassResult()
    batch: list[tuple[bytes, bytes]] = []
    for i, (e_str, ref_str) in enumerate(strs):
        e = json.loads(e_str)
        exp_cls = EXPERIMENT_TYPE_MAP[e["experiment_type"]]
        env_cls = exp_cls.model_fields["environment"].annotation
        if layout == "interned":
            env_id = e["environment"]["$ref"]
            if env_id in cache.env_by_key:
                env_model, env_frag, env_ident = cache.env_by_key[env_id]
            else:
                env_model, env_frag, env_ident = cache.environment(
                    env_id, verified_constant(env_id, constants[env_id]), env_cls
                )
        else:
            env_key = json.dumps(e["environment"])
            env_model, env_frag, env_ident = cache.environment(
                env_key, e["environment"], env_cls
            )
        ref_model, ref_frag, ref_hash, ploidy = cache.reference(ref_str)
        experiment = exp_cls(
            dataset_name=e["dataset_name"],
            genotype=e["genotype"],
            environment=env_model,
            phenotype=e["phenotype"],
        )
        value = (
            '{"experiment": '
            + splice_experiment(experiment, env_frag)
            + ', "experiment_reference": '
            + ref_frag
            + "}"
        )
        batch.append((f"data_{i}".encode(), value.encode()))
        if len(batch) >= PROCESS_BATCH:
            put_batched(env, batch)
            batch = []
        # indices, all off dicts already in hand
        out.ref_hash_to_indices.setdefault(ref_hash, []).append(i)
        perts = e["genotype"]["perturbations"]
        for p in perts:
            out.gene_set.add(p["systematic_gene_name"])
        agg_key = identity_sha256(
            {
                "ploidy": ploidy,
                "perturbations": sorted(
                    (
                        {
                            "gene": p["systematic_gene_name"],
                            "perturbation_type": p["perturbation_type"],
                            "copy_number": p.get("copy_number"),
                            "reference_copy_number": p.get("reference_copy_number"),
                        }
                        for p in perts
                    ),
                    key=_canonical,
                ),
                "environment": env_ident,
            }
        )
        g = out.agg_key_to_group.get(agg_key)
        if g is None:
            g = len(out.group_members)
            out.agg_key_to_group[agg_key] = g
            out.group_members.append([])
        out.group_members[g].append(i)
        label_name = e["phenotype"]["label_name"]
        out.raw_label_index.setdefault(label_name, []).append(i)
        out.experiment_types.add(e["experiment_type"])
        label_value = getattr(experiment.phenotype, label_name)
        scalar = (
            label_value
            if not isinstance(label_value, (dict, list, tuple))
            and label_value is not None
            and not np.isnan(label_value)
            else None
        )
        out.per_record.append((label_name, e["dataset_name"], len(perts), scalar))
    if batch:
        put_batched(env, batch)
    out.finalize()
    return out


IO_RECORDS = 50_000
"""Values read per I/O region; each region is disjoint from the others and the slice."""


IO_LMDB = "/db/experiments/025-solid-growth-001-full-build/processed/lmdb"
"""Cold store for the I/O rows. The 033 processed LMDB cannot be read cold any more:
the first full run of this probe streamed all 106 GB of it into the page cache (fincore
then showed it 100% resident), so the I/O rows read the 025 processed LMDB, which has
the same layout (one JSON-list value per integer key) and was 0% resident when checked
(``fincore``). Its throughput is converted to 033-sized values."""


def io_read_cost(readahead: bool, order: str, count: int) -> tuple[float, int]:
    """Cold read of ``count`` values off ``IO_LMDB``: ms/value, bytes.

    ``order`` ``cursor`` walks the cursor from a key prefix, the order every pipeline
    pass reads in (lexicographic: ``5, 50, 500, ..., 5000000, 5000001, ...``);
    ``key`` reads a run of consecutive integer keys with ``get``, the order the values
    were inserted in. Values are copied out so every page is touched; nothing is parsed.
    """
    regions = {
        (False, "cursor"): b"5",
        (True, "cursor"): b"7",
        (False, "key"): 4_100_000,
        (True, "key"): 2_300_000,
    }
    start = regions[(readahead, order)]
    env = lmdb.open(IO_LMDB, readonly=True, lock=False, readahead=readahead)
    total = 0
    t = time.perf_counter()
    with env.begin() as txn:
        if order == "cursor":
            assert isinstance(start, bytes)
            cur = txn.cursor()
            cur.set_range(start)
            for i, (_k, v) in enumerate(cur):
                total += len(bytes(v))
                if i + 1 >= count:
                    break
        else:
            assert isinstance(start, int)
            for k in range(start, start + count):
                total += len(bytes(txn.get(str(k).encode())))
    wall = time.perf_counter() - t
    env.close()
    return wall / count * 1e3, total


def full_distinct_worker(bounds: tuple[int, int]) -> tuple[set[bytes], set[bytes], int]:
    """Distinct environment and reference digests over group keys ``[lo, hi)``."""
    lo, hi = bounds
    envs: set[bytes] = set()
    refs: set[bytes] = set()
    n = 0
    env = lmdb.open(PROCESSED_LMDB, readonly=True, lock=False, readahead=False)
    with env.begin() as txn:
        for k in range(lo, hi):
            for record in json.loads(txn.get(str(k).encode())):
                envs.add(
                    hashlib.sha256(
                        json.dumps(record["experiment"]["environment"]).encode()
                    ).digest()
                )
                refs.add(
                    hashlib.sha256(
                        json.dumps(record["experiment_reference"]).encode()
                    ).digest()
                )
                n += 1
    env.close()
    return envs, refs, n


def full_distinct(limit: int) -> tuple[int, int]:
    """Count distinct environment and reference JSON over the first ``limit`` groups."""
    edges = np.linspace(0, limit, FULL_WORKERS * 8 + 1).round().astype(int)
    bounds = [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]
    t = time.perf_counter()
    envs: set[bytes] = set()
    refs: set[bytes] = set()
    n = 0
    # spawn, not fork: the parent holds several GB of slice objects that forked
    # children would copy page by page as their garbage collector touches them
    with get_context("spawn").Pool(FULL_WORKERS) as pool:
        for e, r, c in pool.imap_unordered(full_distinct_worker, bounds):
            envs |= e
            refs |= r
            n += c
    wall = time.perf_counter() - t
    print(
        f"full store ({limit:,} groups): {n:,} records, {len(envs):,} distinct "
        f"environments, {len(refs):,} distinct references ({wall / 60:.1f} min, "
        f"{FULL_WORKERS} workers)"
    )
    return len(envs), len(refs)


def main() -> None:
    """Measure every step on the slice, verify the single pass, and write the CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--io-records",
        type=int,
        default=IO_RECORDS,
        help="values read per cold I/O region (smoke runs use few, to keep them cold)",
    )
    parser.add_argument(
        "--full-limit",
        type=int,
        default=FULL_GROUPS,
        help="groups of the store scanned for distinct constants (smoke runs use fewer)",
    )
    parser.add_argument(
        "--groups",
        type=int,
        default=SAMPLE_GROUPS,
        help="evenly spaced groups to sample (smoke runs use fewer)",
    )
    args = parser.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    run_dir = tempfile.mkdtemp(prefix="qbcp_", dir=SCRATCH)
    aggregator_cls = load_aggregator_class()

    t = time.perf_counter()
    groups, group_values = read_slice(args.groups)
    read_s = time.perf_counter() - t
    records = [rec for g in groups for rec in g]
    n = len(records)
    n_groups = len(groups)
    print(
        f"slice: {n_groups:,} groups, {n:,} records, read in {read_s:.1f} s "
        f"(cold random reads off /db)"
    )

    # Cypher stand-ins: inline layout (033 graph) and pointer layout (next graph).
    inline = [
        (json.dumps(r["experiment"]), json.dumps(r["experiment_reference"]))
        for r in records
    ]
    constants_payload: dict[str, str] = {}
    interned: list[tuple[str, str]] = []
    for r, (_, ref_str) in zip(records, inline):
        pointered, consts = split_experiment_dump(r["experiment"])
        for cid, _kind, payload in consts:
            constants_payload[cid] = payload
        interned.append((json.dumps(pointered), ref_str))
    constants_parsed = {k: json.loads(v) for k, v in constants_payload.items()}

    # Distinct constants in the slice.
    def distinct(get: Callable[[dict[str, Any]], Any]) -> int:
        return len({json.dumps(get(r)) for r in records})

    distinct_counts = {
        "environment": distinct(lambda r: r["experiment"]["environment"]),
        "experiment_reference": distinct(lambda r: r["experiment_reference"]),
        "genome_reference": distinct(
            lambda r: r["experiment_reference"]["genome_reference"]
        ),
        "environment_reference": distinct(
            lambda r: r["experiment_reference"]["environment_reference"]
        ),
        "phenotype_reference": distinct(
            lambda r: r["experiment_reference"]["phenotype_reference"]
        ),
        "genotype": distinct(lambda r: r["experiment"]["genotype"]),
        "phenotype": distinct(lambda r: r["experiment"]["phenotype"]),
        "interned_constant_ids": len(constants_payload),
    }
    bytes_e = statistics.mean(len(s[0]) for s in inline)
    bytes_e_ptr = statistics.mean(len(s[0]) for s in interned)
    bytes_ref = statistics.mean(len(s[1]) for s in inline)
    bytes_env = statistics.mean(
        len(json.dumps(r["experiment"]["environment"])) for r in records
    )
    print("distinct in slice:", distinct_counts)
    print(
        f"mean bytes: e_serialized {bytes_e:.0f} (pointer layout {bytes_e_ptr:.0f}), "
        f"environment {bytes_env:.0f}, ref_serialized {bytes_ref:.0f}"
    )

    rows: list[
        tuple[str, float, str, str]
    ] = []  # (step, ms/record, observed key, note)

    def add(step: str, ms: float, observed: str = "", note: str = "") -> None:
        rows.append((step, ms, observed, note))
        print(f"{step:55s} {ms:8.4f} ms/record  {note}")

    # ---------------- raw stage, steps alone ----------------
    parsed_inline = [(json.loads(e), json.loads(r)) for e, r in inline]
    add(
        "raw.json_loads_e_and_ref_inline",
        timed(lambda: [(json.loads(e), json.loads(r)) for e, r in inline], n),
    )
    add(
        "raw.json_loads_e_and_ref_pointer_layout",
        timed(lambda: [(json.loads(e), json.loads(r)) for e, r in interned], n),
    )
    parsed_ptr = [(json.loads(e), json.loads(r)) for e, r in interned]
    add(
        "raw.resolve_pointers_pointer_layout",
        timed(
            lambda: [
                (
                    resolve_pointers(e, constants_parsed),
                    resolve_pointers(r, constants_parsed),
                )
                for e, r in parsed_ptr
            ],
            n,
        ),
        note="collect_pointers excluded; walks both dicts",
    )
    add(
        "raw.validate_experiment",
        timed(
            lambda: [
                EXPERIMENT_TYPE_MAP[e["experiment_type"]](
                    dataset_name=e["dataset_name"],
                    genotype=e["genotype"],
                    environment=e["environment"],
                    phenotype=e["phenotype"],
                )
                for e, _ in parsed_inline
            ],
            n,
        ),
    )
    add(
        "raw.validate_reference",
        timed(
            lambda: [
                EXPERIMENT_REFERENCE_TYPE_MAP[r["experiment_reference_type"]](**r)
                for _, r in parsed_inline
            ],
            n,
        ),
    )
    exp_cls0 = EXPERIMENT_TYPE_MAP[parsed_inline[0][0]["experiment_type"]]
    env_cls0 = exp_cls0.model_fields["environment"].annotation
    assert env_cls0 is not None
    add(
        "raw.validate_environment_alone",
        timed(lambda: [env_cls0(**e["environment"]) for e, _ in parsed_inline], n),
        note="the part of validate_experiment a constant cache removes",
    )
    add(
        "raw.model_construct_experiment_and_reference",
        timed(
            lambda: [
                (
                    EXPERIMENT_TYPE_MAP[e["experiment_type"]].model_construct(
                        dataset_name=e["dataset_name"],
                        genotype=e["genotype"],
                        environment=e["environment"],
                        phenotype=e["phenotype"],
                    ),
                    EXPERIMENT_REFERENCE_TYPE_MAP[
                        r["experiment_reference_type"]
                    ].model_construct(**r),
                )
                for e, r in parsed_inline
            ],
            n,
        ),
        note="no validation; nested fields stay dicts",
    )
    validated = [validate_pair(e, r) for e, r in parsed_inline]
    add(
        "raw.json_dumps_model_dump",
        timed(lambda: [dump_record(x, y) for x, y in validated], n),
    )
    add(
        "raw.model_dump_json_rust",
        timed(
            lambda: [(x.model_dump_json(), y.model_dump_json()) for x, y in validated],
            n,
        ),
        note="pydantic-core serializer; NOT byte-identical to json.dumps",
    )
    written = [dump_record(x, y).encode() for x, y in validated]
    keys = [f"data_{i}".encode() for i in range(n)]
    roundtrip_same = sum(
        w
        == json.dumps(
            {
                "experiment": r["experiment"],
                "experiment_reference": r["experiment_reference"],
            }
        ).encode()
        for w, r in zip(written, records)
    )
    print(f"re-validating a stored record reproduces its bytes on {roundtrip_same}/{n}")
    print(
        "Phenotype has a `.label` attribute (Neo4jQueryRaw.compute_phenotype_label_index "
        f"reads it): {hasattr(validated[0][0].phenotype, 'label')}"
    )
    raw_path = osp.join(run_dir, "raw_lmdb")

    def lmdb_put() -> None:
        env = fresh_env(raw_path)
        put_batched(env, zip(keys, written))
        env.close()

    add("raw.lmdb_put_batched_1000", timed(lmdb_put, n))

    # model_construct output equality (would it be a drop-in?). Dumping a constructed
    # model whose nested fields are dicts makes pydantic warn per field; the warnings are
    # the expected symptom and are silenced here.
    n_check = min(n, 2000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        constructed_same = sum(
            dump_record(
                EXPERIMENT_TYPE_MAP[e["experiment_type"]].model_construct(
                    dataset_name=e["dataset_name"],
                    genotype=e["genotype"],
                    environment=e["environment"],
                    phenotype=e["phenotype"],
                ),
                EXPERIMENT_REFERENCE_TYPE_MAP[
                    r["experiment_reference_type"]
                ].model_construct(**r),
            ).encode()
            == w
            for (e, r), w in zip(parsed_inline[:n_check], written[:n_check])
        )
    print(
        f"model_construct dump byte-identical on {constructed_same}/{n_check} records"
    )

    # ---------------- raw stage, combined as _write_batch runs it ----------------
    def raw_stage(strs: list[tuple[str, str]], consts: dict[str, Any]) -> None:
        env = fresh_env(raw_path)
        batch: list[tuple[int, dict[str, Any], dict[str, Any]]] = []

        def flush() -> None:
            refs: set[str] = set()
            for _, ed, rd in batch:
                collect_pointers(ed, refs)
                collect_pointers(rd, refs)
            assert refs <= consts.keys()
            with env.begin(write=True) as txn:
                for i, ed, rd in batch:
                    ed = resolve_pointers(ed, consts)
                    rd = resolve_pointers(rd, consts)
                    x, y = validate_pair(ed, rd)
                    txn.put(f"data_{i}".encode(), dump_record(x, y).encode())

        for i, (e_str, ref_str) in enumerate(strs):
            batch.append((i, json.loads(e_str), json.loads(ref_str)))
            if len(batch) >= PROCESS_BATCH:
                flush()
                batch = []
        if batch:
            flush()
        env.close()

    add(
        "raw.combined_write_batch_inline_layout",
        timed(lambda: raw_stage(inline, {}), n),
        "raw_stage_033",
        "what job 2929 ran client-side; observed also includes Neo4j server + driver",
    )
    add(
        "raw.combined_write_batch_pointer_layout",
        timed(lambda: raw_stage(interned, constants_parsed), n),
        note="current code on the interned graph; constants pre-fetched",
    )
    raw_stage(inline, {})  # leave the scratch raw LMDB holding the slice

    # ---------------- downstream passes over the raw LMDB ----------------
    def cursor_pass(path: str, fn: Callable[[bytes, bytes], None]) -> None:
        env = lmdb.open(path, readonly=True, lock=False, readahead=False)
        with env.begin() as txn:
            for k, v in txn.cursor():
                fn(k, v)
        env.close()

    def ref_index_pass() -> dict[str, list[int]]:
        h: dict[str, list[int]] = {}

        def fn(k: bytes, v: bytes) -> None:
            idx = int(k.decode().split("_")[1])
            ref = json.loads(v.decode("utf-8"))["experiment_reference"]
            h.setdefault(
                compute_sha256_hash(json.dumps(ref, sort_keys=True)), []
            ).append(idx)

        cursor_pass(latest(raw_path), fn)
        return h

    add("pass.reference_index", timed(ref_index_pass, n), "reference_index_pass")

    def gene_set_pass() -> set[str]:
        s: set[str] = set()

        def fn(_k: bytes, v: bytes) -> None:
            for p in json.loads(v.decode("utf-8"))["experiment"]["genotype"][
                "perturbations"
            ]:
                s.add(p["systematic_gene_name"])

        cursor_pass(latest(raw_path), fn)
        return s

    add("pass.gene_set", timed(gene_set_pass, n), "gene_set_pass")

    def raw_label_index_getitem() -> dict[str, list[int]]:
        # Neo4jQueryRaw.compute_phenotype_label_index: self[i] reopens the env and
        # validates both halves per record.
        out: dict[str, list[int]] = {}
        for i in range(n):
            env = lmdb.open(
                latest(raw_path),
                map_size=int(2e10),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            with env.begin() as txn:
                d = json.loads(txn.get(f"data_{i}".encode()).decode())
                x = EXPERIMENT_TYPE_MAP[d["experiment"]["experiment_type"]](
                    **d["experiment"]
                )
                EXPERIMENT_REFERENCE_TYPE_MAP[
                    d["experiment_reference"]["experiment_reference_type"]
                ](**d["experiment_reference"])
            env.close()
            # the library reads ``.phenotype.label``, which the schema does not have;
            # ``label_name`` is timed instead (same cost: one attribute read)
            out.setdefault(x.phenotype.label_name, []).append(i)
        return out

    add(
        "pass.raw_phenotype_label_index_self_getitem",
        timed(raw_label_index_getitem, n, repeats=1),
        note="Neo4jQueryRaw.phenotype_label_index; NOT called in job 2929",
    )

    aggregator = aggregator_cls(root=osp.join(run_dir, "agg_unused"))
    add(
        "pass.aggregation_key_raw_alone",
        timed(lambda: [aggregator.aggregate_key_raw(r) for r in records], n),
        note="GenotypeEnvironmentAggregator.aggregate_key_raw on parsed dicts",
    )
    env_models = [env_cls0(**r["experiment"]["environment"]) for r in records]
    add(
        "pass.aggregation_environment_identity_hash_alone",
        timed(
            lambda: [
                identity_sha256({"e": environment_identity(m)}) for m in env_models
            ],
            n,
        ),
        note="identity projection + sha256 on validated envs (no validation)",
    )

    def agg_pass1() -> dict[str, list[bytes]]:
        kg: dict[str, list[bytes]] = {}

        def fn(k: bytes, v: bytes) -> None:
            kg.setdefault(
                aggregator.aggregate_key_raw(json.loads(v.decode("utf-8"))), []
            ).append(bytes(k))

        cursor_pass(latest(raw_path), fn)
        return kg

    add(
        "pass.aggregation_pass1_grouping",
        timed(agg_pass1, n),
        "aggregation_pass1_grouping",
    )
    key_groups = agg_pass1()
    agg_path = osp.join(run_dir, "agg_lmdb")

    def agg_pass2() -> None:
        env_in = lmdb.open(latest(raw_path), readonly=True, lock=False, readahead=False)
        env_out = fresh_env(agg_path)
        with env_in.begin() as tin, env_out.begin(write=True) as tout:
            for idx, ks in enumerate(key_groups.values()):
                tout.put(
                    f"{idx}".encode(), b"[" + b",".join(tin.get(k) for k in ks) + b"]"
                )
        env_in.close()
        env_out.close()

    add(
        "pass.aggregation_pass2_write",
        timed(agg_pass2, n),
        "aggregation_pass2_write",
        "one write txn over all groups, as Aggregator.process",
    )
    agg_pass2()
    proc_path = osp.join(run_dir, "processed_lmdb")

    def copy_pass() -> None:
        env_src = lmdb.open(latest(agg_path), readonly=True, lock=False)
        env_dst = fresh_env(proc_path)
        with env_src.begin() as ts, env_dst.begin(write=True) as td:
            for k, v in ts.cursor():
                td.put(k, v)
        env_src.close()
        env_dst.close()

    add("pass.processed_copy", timed(copy_pass, n), "processed_copy")
    copy_pass()

    # The real Aggregator.process end to end, for a cross-check of pass1 + pass2.
    def real_aggregator() -> None:
        a = aggregator_cls(root=tempfile.mkdtemp(prefix="agg_", dir=run_dir))
        a.process(latest(raw_path), a.lmdb_dir)

    add(
        "pass.aggregator_process_real_pass1_plus_pass2",
        timed(real_aggregator, n, repeats=1),
        note="GenotypeEnvironmentAggregator.process on the slice (cross-check)",
    )

    def processed_loads_pass(fn: Callable[[int, list[dict[str, Any]]], None]) -> None:
        cursor_pass(
            latest(proc_path), lambda k, v: fn(int(k.decode()), json.loads(v.decode()))
        )

    add(
        "pass.compute_phenotype_info",
        timed(
            lambda: processed_loads_pass(
                lambda _i, dl: [d["experiment"]["experiment_type"] for d in dl]
            ),
            n,
        ),
        "compute_phenotype_info_pass",
    )

    def label_df_pass() -> dict[int, dict[str, float]]:
        out: dict[int, dict[str, float]] = {}

        def fn(i: int, dl: list[dict[str, Any]]) -> None:
            row: dict[str, float] = {}
            for d in dl:
                x = EXPERIMENT_TYPE_MAP[d["experiment"]["experiment_type"]](
                    **d["experiment"]
                )
                name = x.phenotype.label_name
                value = getattr(x.phenotype, name)
                if isinstance(value, (dict, list, tuple)):
                    continue
                if value is not None and not np.isnan(value):
                    row[name] = value
            out[i] = row

        processed_loads_pass(fn)
        return out

    add("pass.label_df", timed(label_df_pass, n), "label_df_pass")

    def index_pass(get: Callable[[dict[str, Any]], Any]) -> dict[Any, list[int]]:
        out: dict[Any, set[int]] = {}

        def fn(i: int, dl: list[dict[str, Any]]) -> None:
            for d in dl:
                out.setdefault(get(d), set()).add(i)

        processed_loads_pass(fn)
        return {k: sorted(v) for k, v in out.items()}

    add(
        "pass.phenotype_label_index",
        timed(
            lambda: index_pass(lambda d: d["experiment"]["phenotype"]["label_name"]), n
        ),
        "phenotype_label_index_pass",
    )
    add(
        "pass.dataset_name_index",
        timed(lambda: index_pass(lambda d: d["experiment"]["dataset_name"]), n),
        "dataset_name_index_pass",
    )
    add(
        "pass.perturbation_count_index",
        timed(
            lambda: index_pass(
                lambda d: len(d["experiment"]["genotype"]["perturbations"])
            ),
            n,
        ),
        "perturbation_count_index_pass",
    )
    add(
        "pass.measurements_per_entry",
        timed(lambda: processed_loads_pass(lambda _i, dl: len(dl)), n),
        "measurements_per_entry_pass",
        "033 query.py report pass, not the library",
    )

    # ---------------- single pass ----------------
    # Cost of building one cached constant (validate + dump + identity for an
    # environment; parse + validate + dump + hash for a reference), timed directly on
    # the slice's distinct constants rather than as a cold-minus-warm difference.
    distinct_env = list(
        {
            json.dumps(r["experiment"]["environment"]): r["experiment"]["environment"]
            for r in records
        }.items()
    )
    distinct_ref = list({s for _, s in inline})
    n_constants = len(distinct_env) + len(distinct_ref)

    def build_constants() -> None:
        cache = ConstantCache()
        for key, env_dict in distinct_env:
            cache.environment(key, env_dict, env_cls0)
        for ref_str in distinct_ref:
            cache.reference(ref_str)

    per_constant_ms = timed(build_constants, n_constants)
    print(
        f"constant build: {per_constant_ms:.4f} ms per constant ({n_constants} constants)"
    )
    sp_path = osp.join(run_dir, "single_pass_raw_lmdb")
    results: dict[str, SinglePassResult] = {}
    for layout, strs in (("inline", inline), ("interned", interned)):

        def cold(layout: str = layout, strs: list[tuple[str, str]] = strs) -> None:
            env = fresh_env(sp_path)
            results[layout] = single_pass(
                strs, layout, constants_payload, ConstantCache(), env
            )
            env.close()

        add(
            f"single_pass.{layout}_layout_cold_cache",
            timed(cold, n),
            note="validates every slice-distinct constant once",
        )
        warm_cache = ConstantCache()
        env = fresh_env(sp_path)
        single_pass(strs, layout, constants_payload, warm_cache, env)
        env.close()

        def warm(
            layout: str = layout,
            strs: list[tuple[str, str]] = strs,
            cache: ConstantCache = warm_cache,
        ) -> None:
            env = fresh_env(sp_path)
            single_pass(strs, layout, constants_payload, cache, env)
            env.close()

        add(
            f"single_pass.{layout}_layout_warm_cache",
            timed(warm, n),
            note="steady state: every constant already validated",
        )
        cold(layout, strs)  # leave the LMDB holding this layout's output

        # ---- verification against the pipeline-style outputs ----
        res = results[layout]
        env_sp = lmdb.open(latest(sp_path), readonly=True, lock=False)
        with env_sp.begin() as txn:
            same_bytes = all(txn.get(k) == w for k, w in zip(keys, written))
        env_sp.close()
        assert same_bytes, f"{layout}: single-pass raw LMDB differs from the pipeline's"
        ref_idx = ref_index_pass()
        assert {h: sorted(v) for h, v in res.ref_hash_to_indices.items()} == {
            h: sorted(v) for h, v in ref_idx.items()
        }, f"{layout}: reference index differs"
        assert res.gene_set == gene_set_pass(), f"{layout}: gene set differs"
        pipeline_groups = [
            [int(k.decode().split("_")[1]) for k in ks] for ks in key_groups.values()
        ]
        assert res.groups == pipeline_groups, (
            f"{layout}: aggregation group order or membership differs"
        )
        assert res.phenotype_label_index == index_pass(
            lambda d: d["experiment"]["phenotype"]["label_name"]
        ), f"{layout}: phenotype label index differs"
        assert res.dataset_name_index == index_pass(
            lambda d: d["experiment"]["dataset_name"]
        ), f"{layout}: dataset name index differs"
        assert res.perturbation_count_index == index_pass(
            lambda d: len(d["experiment"]["genotype"]["perturbations"])
        ), f"{layout}: perturbation count index differs"
        assert res.label_rows == label_df_pass(), f"{layout}: label rows differ"
        print(
            f"single pass ({layout}) verified on {n:,} records: raw values "
            f"byte-identical; reference index ({len(ref_idx)} refs), gene set "
            f"({len(res.gene_set)} genes), {len(res.groups):,} groups in pipeline "
            "order, phenotype/dataset/perturbation-count indices and label rows equal"
        )
    multi = sum(len(g) > 1 for g in key_groups.values())
    lex_vs_numeric = sum(
        1
        for a, b in zip(
            key_groups.values(),
            sorted(key_groups.values(), key=lambda ks: int(ks[0].decode()[5:])),
        )
        if a is not b
    )
    print(
        f"slice groups with >1 member: {multi}; groups whose index differs between "
        f"cursor (lexicographic) order and record order: {lex_vs_numeric}"
    )

    # ---------------- cold I/O ----------------
    group_bytes_033 = statistics.mean(len(v) for v in group_values)
    for readahead in (False, True):
        for order in ("cursor", "key"):
            ms_value, nbytes = io_read_cost(readahead, order, args.io_records)
            mb_per_s = nbytes / 1e6 / (ms_value * args.io_records / 1e3)
            add(
                f"io.cold_read_{order}_order_readahead_{'on' if readahead else 'off'}",
                group_bytes_033 / 1e6 / mb_per_s * 1e3 * FULL_GROUPS / FULL_RECORDS,
                note=(
                    f"{mb_per_s:.0f} MB/s cold copy-out of {args.io_records} values "
                    f"({nbytes / args.io_records / 1e3:.1f} KB each) off the 025 store, "
                    f"at the 033 group size {group_bytes_033 / 1e3:.1f} KB; pipeline "
                    "passes open with readahead off and read in cursor order"
                ),
            )

    # ---------------- full-store distinct constants ----------------
    full_env, full_ref = full_distinct(args.full_limit)

    add(
        "single_pass.constant_builds_full_store",
        per_constant_ms * (full_env + full_ref) / FULL_RECORDS,
        note=(
            f"{per_constant_ms:.4f} ms per constant x {full_env + full_ref} full-store "
            "distinct environments + references, spread per record"
        ),
    )

    # ---------------- projections ----------------
    ms = {step: v for step, v, _, _ in rows}
    to_h = FULL_RECORDS / 3.6e6
    pipeline_steps = [
        "raw.combined_write_batch_inline_layout",
        "pass.reference_index",
        "pass.gene_set",
        "pass.aggregation_pass1_grouping",
        "pass.aggregation_pass2_write",
        "pass.processed_copy",
        "pass.compute_phenotype_info",
        "pass.label_df",
        "pass.phenotype_label_index",
        "pass.dataset_name_index",
        "pass.perturbation_count_index",
        "pass.measurements_per_entry",
    ]
    pipeline_cpu_ms = sum(ms[s] for s in pipeline_steps)
    add(
        "projection.current_pipeline_client_cpu",
        pipeline_cpu_ms,
        note="sum of the 12 measured steps job 2929 ran, in-memory/page-cached I/O",
    )
    observed_total = sum(OBSERVED_H.values())
    server_driver_h = (
        OBSERVED_H["raw_stage_033"]
        - ms["raw.combined_write_batch_inline_layout"] * to_h
    )
    print(
        f"job 2929 observed {observed_total:.2f} h; raw stage residual not in the "
        f"client steps (Neo4j server + driver + record hydration + disk): "
        f"{server_driver_h:.2f} h"
    )
    for layout in ("inline", "interned"):
        warm = ms[f"single_pass.{layout}_layout_warm_cache"]
        sp_cpu_h = (
            warm * FULL_RECORDS + per_constant_ms * (full_env + full_ref)
        ) / 3.6e6
        sp_total_h = server_driver_h + sp_cpu_h + OBSERVED_H["aggregation_pass2_write"]
        add(
            f"projection.single_pass_{layout}_client_cpu",
            sp_cpu_h / to_h,
            note=(
                f"warm {warm:.4f} ms x 6.39M + {per_constant_ms:.3f} ms per constant x "
                f"{full_env + full_ref} full-store distinct env+ref"
            ),
        )
        add(
            f"projection.single_pass_{layout}_wall",
            sp_total_h / to_h,
            note=(
                "raw-stage server/driver residual of job 2929 (assumed unchanged) + "
                "single-pass client CPU + job 2929's observed aggregation pass 2 "
                "(the one remaining re-read); no copy, no post passes"
            ),
        )

    # ---------------- write CSV ----------------
    per_group_scale = n / n_groups
    path = osp.join(RESULTS_DIR, "query_build_cost_probe.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "step",
                "ms_per_record",
                "projected_hours_at_6.39M",
                "observed_hours_job2929",
                "note",
            ]
        )
        for step, ms, obs, note in rows:
            w.writerow(
                [
                    step,
                    f"{ms:.4f}",
                    f"{ms * FULL_RECORDS / 3.6e6:.3f}",
                    f"{OBSERVED_H[obs]:.3f}" if obs else "",
                    note,
                ]
            )
        for name, count in distinct_counts.items():
            w.writerow(
                [f"distinct_in_slice.{name}", "", "", "", f"{count} of {n} records"]
            )
        w.writerow(
            [
                "slice.records",
                "",
                "",
                "",
                f"{n} records in {n_groups} groups ({per_group_scale:.4f} per group)",
            ]
        )
        w.writerow(
            [
                "slice.mean_bytes",
                "",
                "",
                "",
                f"e {bytes_e:.0f}, e pointer layout {bytes_e_ptr:.0f}, environment {bytes_env:.0f}, ref {bytes_ref:.0f}",
            ]
        )
        w.writerow(
            [
                "observed.job2929_total",
                "",
                "",
                f"{observed_total:.3f}",
                "sum of the stage times in OBSERVED_H",
            ]
        )
        w.writerow(
            [
                "distinct_full_store.environment",
                "",
                "",
                "",
                f"{full_env} over the first {args.full_limit} groups",
            ]
        )
        w.writerow(
            [
                "distinct_full_store.experiment_reference",
                "",
                "",
                "",
                f"{full_ref} over the first {args.full_limit} groups",
            ]
        )
    print(path)
    print(f"scratch LMDBs under {run_dir}")


if __name__ == "__main__":
    main()
