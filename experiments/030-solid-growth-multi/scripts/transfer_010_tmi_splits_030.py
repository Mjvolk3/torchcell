# experiments/030-solid-growth-multi/scripts/transfer_010_tmi_splits_030.py
# [[experiments.030-solid-growth-multi.scripts.transfer_010_tmi_splits_030]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/030-solid-growth-multi/scripts/transfer_010_tmi_splits_030
"""Carry 010's seed-42 random tmi train/val/test assignment onto the 030 build.

Split indices do not survive across builds: 010's ``index_seed_42.json`` stores RECORD
INDICES into the 010 LMDB, and the 030 build indexes 13.5 M records in its own order.
The stable join key is the genotype identity, the SORTED GENE-NAME SET of a record's
perturbations, which is one-per-record on both sides (010 aggregates by it; 030 keeps
one record per genotype and every source entry under it). ``perturbation_type`` stays
out of the key for the reason recorded in 025's transfer: the type vocabulary was
renamed between the two builds and a type-bearing key matched 682 of 376,732.

Phase 1 reads 010's split cache + LMDB and maps identity -> split (376,732 records).
Phase 2 reads the 030 triples from ``closure/entries.parquet`` (idx -> gene set) and
joins. The join must be one-to-one: every 010 identity resolves to exactly one 030
triple and every 030 triple is claimed exactly once; anything else is an error, not a
drop, because the pinned populations must equal 025's (val 37,673, test 37,673, train
301,386) for the trigenic score to stay comparable.

Writes (``experiments/030-solid-growth-multi/results/``):

- ``pinned_splits_from_010_seed_42.json.gz``: ``{"report": {...}, "pinned": {"train":
  [...], "val": [...], "test": [...]}}``, the 025 shape, so
  ``load_index_artifact(name, "pinned")`` keeps working.
- ``transfer_010_tmi_splits_030_summary.json``: ``TransferReport``.

    PYTHONPATH=$PWD python experiments/030-solid-growth-multi/scripts/transfer_010_tmi_splits_030.py
"""

from __future__ import annotations

import gzip
import json
import os
import os.path as osp
from collections.abc import Iterator
from typing import Any

import lmdb
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel
from tqdm import tqdm

SPLITS = ("train", "val", "test")
EXPECTED_PINNED = {"train": 301_386, "val": 37_673, "test": 37_673}
IdentityKey = tuple[str, ...]


class TransferReport(BaseModel):
    """Counts of the identity join between 010's split and the 030 triples."""

    source_split_cache: str
    source_lmdb: str
    target_entries_parquet: str
    identities_010: int
    triples_030: int
    matched: int
    unmatched_010: int
    unclaimed_030: int
    multi_matched_010: int
    pinned_counts: dict[str, int]
    expected_pinned_counts: dict[str, int]
    all_pinned_order_3: bool
    join_key: str = "tuple(sorted(set(systematic_gene_name of the perturbations)))"


def record_identity(record: list[dict[str, Any]]) -> IdentityKey:
    """Sorted, de-duplicated gene names of the record's genotype perturbations.

    A post-aggregation record is a list of ``{experiment, experiment_reference}`` entries
    sharing one genotype by construction; the first entry's genotype carries the gene
    set.
    """
    perturbations = record[0]["experiment"]["genotype"]["perturbations"]
    return tuple(sorted({p["systematic_gene_name"] for p in perturbations}))


def identity_of_gene_string(genes: str) -> IdentityKey:
    """The same key from the ``'A|B|C'`` gene string of ``entries.parquet``."""
    return tuple(sorted(set(genes.split("|"))))


def iter_lmdb_records(lmdb_dir: str) -> Iterator[tuple[int, list[dict[str, Any]]]]:
    """Yield (index, record) over a processed Neo4jCellDataset LMDB, read-only."""
    env = lmdb.open(lmdb_dir, readonly=True, lock=False, readahead=False)
    with env.begin() as txn:
        n = txn.stat()["entries"]
        for i in tqdm(range(n)):
            raw = txn.get(str(i).encode())
            assert raw is not None, f"010 record {i} missing"
            yield i, json.loads(raw)
    env.close()


def join_by_identity(
    identity_to_split: dict[IdentityKey, str], target: dict[int, IdentityKey]
) -> tuple[dict[str, list[int]], dict[str, int]]:
    """Assign each target index the split of its identity.

    Returns the pinned lists (sorted) and a diagnostic dict: identities of the source
    that no target carries (``unmatched``), identities carried by more than one target
    (``multi_matched``), and targets no source identity claims (``unclaimed``). The
    caller decides whether any of those is acceptable; here none is.
    """
    by_identity: dict[IdentityKey, list[int]] = {}
    for idx, key in target.items():
        by_identity.setdefault(key, []).append(idx)
    pinned: dict[str, list[int]] = {s: [] for s in SPLITS}
    unmatched = 0
    multi = 0
    for key, split in identity_to_split.items():
        hits = by_identity.get(key)
        if hits is None:
            unmatched += 1
            continue
        if len(hits) != 1:
            multi += 1
            continue
        pinned[split].append(hits[0])
    claimed = sum(len(v) for v in pinned.values())
    diag = {
        "unmatched": unmatched,
        "multi_matched": multi,
        "unclaimed": len(target) - claimed,
    }
    return {s: sorted(v) for s, v in pinned.items()}, diag


def main() -> None:
    """Map 010's per-split identities onto 030 triple indices and write the pin file."""
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    experiment_root = os.environ["EXPERIMENT_ROOT"]
    root_010 = osp.join(
        data_root, "data/torchcell/experiments/010-kuzmin-tmi/001-small-build"
    )
    exp_030 = osp.join(data_root, "data/torchcell/experiments/030-solid-growth-multi")
    entries_path = osp.join(exp_030, "closure", "entries.parquet")
    results = osp.join(experiment_root, "030-solid-growth-multi", "results")
    os.makedirs(results, exist_ok=True)

    split_cache = osp.join(root_010, "data_module_cache/index_seed_42.json")
    with open(split_cache) as f:
        index_010: dict[str, list[int]] = json.load(f)
    split_of_010_index = {i: split for split in SPLITS for i in index_010[split]}

    identity_to_split: dict[IdentityKey, str] = {}
    lmdb_010 = osp.join(root_010, "processed/lmdb")
    print("phase 1: reading 010 records", flush=True)
    for i, record in iter_lmdb_records(lmdb_010):
        identity = record_identity(record)
        split = split_of_010_index[i]
        prior = identity_to_split.get(identity)
        assert prior is None or prior == split, f"identity {identity} in two splits"
        identity_to_split[identity] = split
    assert len(identity_to_split) == len(split_of_010_index), (
        "two 010 records share one identity"
    )
    print(f"  {len(identity_to_split)} identities", flush=True)

    print("phase 2: 030 triples from the closure cache", flush=True)
    entries = pd.read_parquet(entries_path, columns=["idx", "order", "genes"])
    triples = entries.loc[entries["order"] == 3, ["idx", "genes"]].drop_duplicates()
    assert triples["idx"].is_unique, "a 030 triple carries two gene strings"
    target = {
        int(i): identity_of_gene_string(g)
        for i, g in zip(triples["idx"], triples["genes"], strict=True)
    }
    pinned, diag = join_by_identity(identity_to_split, target)

    with open(
        osp.join(
            exp_030, "001-multi-build", "processed", "perturbation_count_index.json"
        )
    ) as f:
        order_3 = set(json.load(f)["3"])
    all_order_3 = all(i in order_3 for v in pinned.values() for i in v)

    report = TransferReport(
        source_split_cache=split_cache,
        source_lmdb=lmdb_010,
        target_entries_parquet=entries_path,
        identities_010=len(identity_to_split),
        triples_030=len(target),
        matched=sum(len(v) for v in pinned.values()),
        unmatched_010=diag["unmatched"],
        unclaimed_030=diag["unclaimed"],
        multi_matched_010=diag["multi_matched"],
        pinned_counts={s: len(v) for s, v in pinned.items()},
        expected_pinned_counts=EXPECTED_PINNED,
        all_pinned_order_3=all_order_3,
    )
    print(report.model_dump_json(indent=2))

    with gzip.open(
        osp.join(results, "pinned_splits_from_010_seed_42.json.gz"), "wt"
    ) as f:
        json.dump({"report": report.model_dump(mode="json"), "pinned": pinned}, f)
    with open(osp.join(results, "transfer_010_tmi_splits_030_summary.json"), "w") as f:
        f.write(report.model_dump_json(indent=2))

    assert diag["unmatched"] == 0, f"{diag['unmatched']} 010 identities absent from 030"
    assert diag["multi_matched"] == 0, f"{diag['multi_matched']} identities match twice"
    assert diag["unclaimed"] == 0, (
        f"{diag['unclaimed']} 030 triples claimed by no split"
    )
    assert report.pinned_counts == EXPECTED_PINNED, report.pinned_counts
    assert all_order_3, "a pinned index is not order 3 in 030"
    print("finished: every 010 triple transferred one-to-one")


if __name__ == "__main__":
    main()
