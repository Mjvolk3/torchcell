# experiments/030-solid-growth-multi/scripts/subset_definitions_030.py
# [[experiments.030-solid-growth-multi.scripts.subset_definitions_030]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/030-solid-growth-multi/scripts/subset_definitions_030
"""Materialize the S3 training pool of the 030 no-merge build as an index artifact.

S3 = every single record + every closure double (a double whose gene pair lies inside
some triple's gene set) + every triple. The 025 campaign defined the same pool on the 025
build (1,121,645 records); the 030 build keeps every source entry per genotype instead
of merging, and carries +17 records in the pool, so 025's indices are off by
construction and the pool is regenerated here from 030's own caches.

Inputs (read-only):

- ``processed/perturbation_count_index.json`` and ``processed/dataset_name_index.json``
  of the 030 build (``$DATA_ROOT/data/torchcell/experiments/030-solid-growth-multi/
  001-multi-build``, a symlink to ``/db``).
- ``closure/entries.parquet`` written by ``closure_recompute_030.py``: one row per stored
  entry of every single, every closure double and every triple. Its distinct ``idx`` IS
  the S3 record set; the closure script established completeness (it scanned all 13.1 M
  doubles for pair membership), and this script re-verifies membership: every single and
  every triple of the count index is present, every double present is a count-index
  double whose pair lies inside a triple gene set, and the three counts equal the
  acceptance numbers of the plan (5,694 / 739,236 / 376,732).

Writes (``experiments/030-solid-growth-multi/results/``):

- ``subset_S3_indices.json.gz``: sorted JSON list of 030 record indices.
- ``subset_definitions_030_summary.json``: ``SubsetS3Summary`` (counts per order, the
  025 comparison, the exact-duplicate ``(genes, label, value, dataset)`` entry-tuple
  count of ``#410``, and the 15-name dataset vocabulary with per-name record counts
  inside S3).

    PYTHONPATH=$PWD python experiments/030-solid-growth-multi/scripts/subset_definitions_030.py
"""

from __future__ import annotations

import gzip
import json
import os
import os.path as osp
from itertools import combinations

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field

EXPECTED = {"1": 5_694, "2": 739_236, "3": 376_732}
S3_COUNT_025 = 1_121_645
LABEL_OF_EXP_TYPE = {"fitness": "fitness", "gene interaction": "gene_interaction"}


class SubsetS3Summary(BaseModel):
    """What the S3 pool of the 030 build contains, and how it compares with 025."""

    build: str
    entries_parquet: str
    n_singles: int
    n_closure_doubles: int
    n_triples: int
    n_s3: int
    expected: dict[str, int]
    n_s3_025: int
    s3_minus_025: int
    n_count_index_doubles: int
    n_closure_pairs: int
    n_entry_rows: int
    n_entry_rows_per_order: dict[str, int]
    n_duplicate_entry_rows: int = Field(
        description=(
            "entry rows minus distinct (genes, label, value, dataset) tuples in the S3 "
            "pool: the exact duplicates of #410, counted, not removed"
        )
    )
    n_duplicated_tuples: int = Field(
        description="distinct (genes, label, value, dataset) tuples with multiplicity > 1"
    )
    rows_in_duplicated_tuples_by_dataset: dict[str, int] = Field(
        description=(
            "every entry row that belongs to a tuple with multiplicity > 1 (keep=False), "
            "so this sums to more than n_duplicate_entry_rows, which counts only the "
            "rows beyond the first of each tuple"
        )
    )
    dataset_vocabulary: list[str] = Field(
        description="sorted keys of dataset_name_index.json; the token vocabulary"
    )
    n_records_in_s3_by_dataset: dict[str, int]
    n_entries_in_s3_by_dataset: dict[str, int]
    note: str = (
        "closure completeness (no double outside this pool has a pair inside a triple) "
        "was established by closure_recompute_030.py's scan of every count-index double; "
        "this script re-verifies membership of every listed record."
    )


def roots() -> tuple[str, str, str]:
    """(build root, entries.parquet, results dir) from the environment."""
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    experiment_root = os.environ["EXPERIMENT_ROOT"]
    exp = osp.join(data_root, "data/torchcell/experiments/030-solid-growth-multi")
    return (
        osp.join(exp, "001-multi-build"),
        osp.join(exp, "closure", "entries.parquet"),
        osp.join(experiment_root, "030-solid-growth-multi", "results"),
    )


def closure_pairs(triple_gene_sets: list[str]) -> set[frozenset[str]]:
    """Every gene pair inside some triple gene set (``'A|B|C'`` strings)."""
    pairs: set[frozenset[str]] = set()
    for gs in triple_gene_sets:
        genes = gs.split("|")
        assert len(genes) == 3, gs
        pairs.update(frozenset(p) for p in combinations(genes, 2))
    return pairs


def dump_gz(path: str, obj: object) -> None:
    """Plain gzip of ``json.dumps``."""
    with gzip.open(path, "wt") as f:
        json.dump(obj, f)


def main() -> None:
    """Derive S3 from the closure cache, verify it against the count index, write it."""
    build, entries_path, results = roots()
    assert osp.exists(entries_path), (
        f"missing {entries_path}; run closure_recompute_030.py"
    )
    os.makedirs(results, exist_ok=True)

    entries = pd.read_parquet(
        entries_path, columns=["idx", "order", "genes", "dataset", "exp_type", "value"]
    )
    records = entries[["idx", "order", "genes"]].drop_duplicates()
    per_idx = records.groupby("idx").agg(
        order=("order", "nunique"), genes=("genes", "nunique")
    )
    assert (per_idx["order"] == 1).all() and (per_idx["genes"] == 1).all(), (
        "a record carries two orders or two gene sets"
    )
    by_order = {
        str(k): set(v) for k, v in records.groupby("order")["idx"].agg(set).items()
    }
    assert set(by_order) == {"1", "2", "3"}, sorted(by_order)

    with open(osp.join(build, "processed", "perturbation_count_index.json")) as f:
        count_index: dict[str, list[int]] = json.load(f)
    ci = {k: set(v) for k, v in count_index.items()}
    assert by_order["1"] == ci["1"], "S3 singles differ from the count index"
    assert by_order["3"] == ci["3"], "S3 triples differ from the count index"
    assert by_order["2"] <= ci["2"], "an S3 double is not a count-index double"

    pairs = closure_pairs(records.loc[records["order"] == 3, "genes"].tolist())
    doubles = records.loc[records["order"] == 2, "genes"]
    outside = [gs for gs in doubles if frozenset(gs.split("|")) not in pairs]
    assert not outside, f"{len(outside)} S3 doubles have a pair outside every triple"

    counts = {k: len(v) for k, v in by_order.items()}
    print(f"counts per order: {counts}; expected {EXPECTED}", flush=True)
    for k, n in EXPECTED.items():
        assert counts[k] == n, (
            f"order {k}: {counts[k]} records in the closure cache, plan expects {n}. "
            "Do not adjust the expectation; report the number."
        )
    s3 = sorted(by_order["1"] | by_order["2"] | by_order["3"])
    assert len(s3) == sum(EXPECTED.values())
    dump_gz(osp.join(results, "subset_S3_indices.json.gz"), s3)

    tuples = pd.DataFrame(
        {
            "genes": entries["genes"],
            "label": entries["exp_type"].map(LABEL_OF_EXP_TYPE),
            "value": entries["value"],
            "dataset": entries["dataset"],
        }
    )
    assert tuples["label"].notna().all(), "an exp_type has no label"
    n_distinct = len(tuples.drop_duplicates())
    multiplicity = tuples.value_counts(dropna=False)
    dup_rows = tuples[tuples.duplicated(keep=False)]

    with open(osp.join(build, "processed", "dataset_name_index.json")) as f:
        name_index: dict[str, list[int]] = json.load(f)
    vocabulary = sorted(name_index)
    assert len(vocabulary) == 15, vocabulary
    s3_set = set(s3)
    n_records_by_dataset = {
        name: len(s3_set.intersection(name_index[name])) for name in vocabulary
    }
    from_entries = entries.groupby("dataset")["idx"].nunique().to_dict()
    for name in vocabulary:
        assert n_records_by_dataset[name] == int(from_entries.get(name, 0)), (
            f"{name}: {n_records_by_dataset[name]} S3 records by the name index, "
            f"{from_entries.get(name, 0)} by the entry cache"
        )

    summary = SubsetS3Summary(
        build=build,
        entries_parquet=entries_path,
        n_singles=counts["1"],
        n_closure_doubles=counts["2"],
        n_triples=counts["3"],
        n_s3=len(s3),
        expected=EXPECTED,
        n_s3_025=S3_COUNT_025,
        s3_minus_025=len(s3) - S3_COUNT_025,
        n_count_index_doubles=len(ci["2"]),
        n_closure_pairs=len(pairs),
        n_entry_rows=len(entries),
        n_entry_rows_per_order={
            str(k): int(v)
            for k, v in entries["order"].value_counts().sort_index().items()
        },
        n_duplicate_entry_rows=len(tuples) - n_distinct,
        n_duplicated_tuples=int((multiplicity > 1).sum()),
        rows_in_duplicated_tuples_by_dataset={
            str(k): int(v) for k, v in dup_rows["dataset"].value_counts().items()
        },
        dataset_vocabulary=vocabulary,
        n_records_in_s3_by_dataset=n_records_by_dataset,
        n_entries_in_s3_by_dataset={
            str(name): int(v) for name, v in entries["dataset"].value_counts().items()
        },
    )
    with open(osp.join(results, "subset_definitions_030_summary.json"), "w") as f:
        f.write(summary.model_dump_json(indent=2))
    print(summary.model_dump_json(indent=2))
    print("finished: S3 written and verified")


if __name__ == "__main__":
    main()
