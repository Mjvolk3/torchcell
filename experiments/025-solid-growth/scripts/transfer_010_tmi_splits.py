# experiments/025-solid-growth/scripts/transfer_010_tmi_splits.py
# [[experiments.025-solid-growth.scripts.transfer_010_tmi_splits]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/025-solid-growth/scripts/transfer_010_tmi_splits
"""Carry 010's tmi train/val/test assignment into the 025 dataset, by identity.

Split indices do not survive across datasets: 010's cached ``index_seed_42.json``
stores RECORD INDICES, and adding smf/dmf/tmf/dmi/essentiality/synthleth data to the
pool changes every index after deduplication and aggregation. The stable join key is
the genotype identity: the SORTED GENE-NAME SET of a record's perturbations -- the
same key ``GenotypeAggregator`` groups by in BOTH datasets, so it is one-per-record
by construction on each side. It deliberately excludes ``perturbation_type``: the
perturbation-ontology refactor renamed the type vocabulary between the build 010 ran
against and the served build (``deletion`` -> ``sga_kanmx_deletion`` /
``mean_deletion`` / ...), and a type-bearing key matched only 682 of 376,732 records
on the first transfer attempt while the gene-set population matched exactly.

Phase 1 reads 010's split cache + processed LMDB and maps identity -> split.
Phase 2 reads 025's processed LMDB (run AFTER query.py builds it) and emits
``pinned_splits_from_010_seed_42.json``: 025 indices per split, for CellDataModule's
``pinned_split_indices``. Every 010 identity must resolve in 025 -- the tmi blocks of
025's query select the same record population -- so an unmatched identity is reported
as an error, not silently dropped.

    python experiments/025-solid-growth/scripts/transfer_010_tmi_splits.py
"""

import json
import os
import os.path as osp

import lmdb
from dotenv import load_dotenv
from tqdm import tqdm

SPLITS = ("train", "val", "test")

IdentityKey = tuple[str, ...]


def record_identity(record: list[dict]) -> IdentityKey:
    """Sorted, de-duplicated gene names of the record's genotype perturbations.

    A post-aggregation record is a list of {experiment, experiment_reference} entries
    sharing one aggregation key by construction; the first entry's genotype carries
    the gene set. Gene names only -- see the module docstring for why
    ``perturbation_type`` must stay out of the key.
    """
    perturbations = record[0]["experiment"]["genotype"]["perturbations"]
    return tuple(sorted({p["systematic_gene_name"] for p in perturbations}))


def iter_lmdb_records(lmdb_dir: str):
    """Yield (index, record) over a processed Neo4jCellDataset LMDB."""
    env = lmdb.open(lmdb_dir, readonly=True, lock=False, readahead=False)
    with env.begin() as txn:
        n = txn.stat()["entries"]
        for i in tqdm(range(n)):
            yield i, json.loads(txn.get(str(i).encode()))
    env.close()


def main() -> None:
    """Map 010's per-split identities onto 025 indices and write the pin file."""
    load_dotenv()
    data_root = os.getenv("DATA_ROOT")
    assert data_root is not None, "DATA_ROOT must be set in .env"

    root_010 = osp.join(
        data_root, "data/torchcell/experiments/010-kuzmin-tmi/001-small-build"
    )
    root_025 = osp.join(
        data_root, "data/torchcell/experiments/025-solid-growth/001-full-build"
    )

    with open(osp.join(root_010, "data_module_cache/index_seed_42.json")) as f:
        index_010: dict[str, list[int]] = json.load(f)

    # Phase 1: 010 index -> identity -> split.
    split_of_010_index: dict[int, str] = {
        i: split for split in SPLITS for i in index_010[split]
    }
    identity_to_split: dict[IdentityKey, str] = {}
    print("phase 1: reading 010 records")
    for i, record in iter_lmdb_records(osp.join(root_010, "processed/lmdb")):
        identity = record_identity(record)
        split = split_of_010_index[i]
        prior = identity_to_split.get(identity)
        # Identity collisions across splits would make the transfer ill-defined.
        assert prior is None or prior == split, f"identity {identity} in two splits"
        identity_to_split[identity] = split
    print(
        f"  {len(identity_to_split)} identities across {len(split_of_010_index)} records"
    )

    # Phase 2: 025 index -> identity -> pinned split.
    pinned: dict[str, list[int]] = {s: [] for s in SPLITS}
    matched: set[IdentityKey] = set()
    print("phase 2: reading 025 records")
    for i, record in iter_lmdb_records(osp.join(root_025, "processed/lmdb")):
        identity = record_identity(record)
        split = identity_to_split.get(identity)
        if split is not None:
            pinned[split].append(i)
            matched.add(identity)

    unmatched = set(identity_to_split) - matched
    report = {
        "identities_010": len(identity_to_split),
        "matched": len(matched),
        "unmatched": len(unmatched),
        "pinned_counts": {s: len(v) for s, v in pinned.items()},
    }
    print(json.dumps(report, indent=2))

    out_dir = "experiments/025-solid-growth/results"
    os.makedirs(out_dir, exist_ok=True)
    with open(osp.join(out_dir, "pinned_splits_from_010_seed_42.json"), "w") as f:
        json.dump({"report": report, "pinned": pinned}, f)

    if unmatched:
        sample = [list(map(list, k)) for k in list(unmatched)[:5]]
        raise SystemExit(
            f"ERROR: {len(unmatched)} 010 identities not found in 025 "
            f"(sample: {sample}). The tmi selection differs -- do not train on this."
        )
    print("finished: every 010 record transferred")


if __name__ == "__main__":
    main()
