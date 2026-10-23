# experiments/043-expression-proteome-morphology/scripts/resolve_segregant_alleles_043.py
# [[experiments.043-expression-proteome-morphology.scripts.resolve_segregant_alleles_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/resolve_segregant_alleles_043.py
"""Every Boocock 2025 and N'Guessan 2025 segregant resolved to parent alleles, per gene.

For every segregant record of the raw store (Boocock pooled and de novo cells alike) and
every gene of the 6,607 index, the definite rule of ``torchcell.data.representation_043``
(whole-gene containment in one confident run of a parent at posterior >= 0.9) assigns
the gene to parent 1, parent 2 or nobody. An assigned gene carries its parent's allele:
the S288C row for a BY parent (``peter_strain_id`` None), the parent's own row of the
isolate index (``peter_strain_id`` = AAA RM11-1a, AAS YJM145, ACD YPS163, ABL CBS2888,
ADI YJM981) otherwise. An unassigned gene carries its reason (no block, a block below the
posterior, straddling runs) and no allele.

Writes:

* ``$DATA_ROOT/data/torchcell/experiments/043-expression-proteome-morphology/
  representation_043/segregant_alleles.parquet``: one row per record, keyed by
  ``raw_index`` (the store key ``data_<raw_index>``), with the cross, the parents, the
  per-record counts, and the per-gene assignment (``assignment``: 6,607 bytes, 0/1/2)
  and reason (``reason``: 6,607 bytes, ``SegregantReason``). The per-gene, per-encoder
  sha256 of a record is the parent's key in ``contexts.npz`` (or S288C's), resolved by
  ``representation_043.load_tables``; it is not copied 82,467 x 6,607 x 4 times.
* ``results/resolve_segregant_alleles_043.json``: the summary per dataset and cross
  (assigned, BY, non-reference, unassigned gene counts per record; the literal
  single-block rule's assigned count beside the confident-run rule's; the distinct
  parent alleles the records use).

    python experiments/043-expression-proteome-morphology/scripts/resolve_segregant_alleles_043.py --workers 32
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import os.path as osp
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lmdb
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import representation_043 as rep  # noqa: E402
from dataset_043 import RAW_INDEX_NAME, Dataset043Config  # noqa: E402
from store_version_043 import active as active_store  # noqa: E402

from torchcell.data.representation_043 import (  # noqa: E402
    ContextTables,
    GapKind,
    RepresentationConfig,
    RepresentationResolver,
    SegregantReason,
    assign_segregant_genes,
)
from torchcell.datasets.scerevisiae.messner2023 import _sgd_gff_path  # noqa: E402

SEGREGANT_DATASETS = ("Boocock2025Dataset", "Nguessan2025Dataset")
COUNT_FIELDS = (
    "n_assigned",
    "n_assigned_by",
    "n_assigned_non_reference",
    "n_non_reference_allele_differs_from_s288c",
    "n_non_reference_without_allele",
    "n_unassigned_no_block",
    "n_unassigned_low_posterior",
    "n_unassigned_straddle",
    "n_assigned_single_block_rule",
    "n_blocks",
)


class SegregantRecordCounts(BaseModel):
    """Per-record counts of the definite rule (one parquet row without the bytes)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: str
    raw_index: int
    cross: str
    segregant_id: str
    parent_1: str
    parent_1_peter: str | None
    parent_2: str
    parent_2_peter: str | None
    n_assigned: int
    n_assigned_by: int
    n_assigned_non_reference: int
    n_non_reference_allele_differs_from_s288c: int
    n_non_reference_without_allele: int
    n_unassigned_no_block: int
    n_unassigned_low_posterior: int
    n_unassigned_straddle: int
    n_assigned_single_block_rule: int
    n_blocks: int


_G: dict[str, Any] = {}


def _init(data_root: str) -> None:
    load_dotenv()
    cfg = Dataset043Config(data_root=data_root)
    tables = rep.load_tables(data_root)
    _G["tables"] = tables[0]
    _G["resolver"] = rep.load_resolver(
        data_root,
        RepresentationConfig(),
        cfg.segregant_parent_vocab,
        tables[2],
        _sgd_gff_path(data_root),
        tables=tables,
    )
    _G["lmdb_path"] = osp.join(cfg.store_root, "raw", "lmdb")
    _G["env"] = None


def _work(chunk: list[tuple[str, int]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Resolve a chunk; return rows and per-parent assigned-gene masks."""
    if _G["env"] is None:
        _G["env"] = lmdb.open(
            _G["lmdb_path"], readonly=True, lock=False, readahead=False, max_readers=512
        )
    resolver: RepresentationResolver = _G["resolver"]
    tables: ContextTables = _G["tables"]
    s288c = tables.key_ids[0]
    pos = resolver.pos
    rows: list[dict[str, Any]] = []
    used: dict[str, np.ndarray] = {}
    with _G["env"].begin() as txn:
        for name, raw_index in chunk:
            record = json.loads(txn.get(f"data_{raw_index}".encode()))
            ex = record["experiment"]
            if ex["dataset_name"] != name:
                raise ValueError(
                    f"data_{raw_index} is {ex['dataset_name']}, not {name}"
                )
            g = ex["genotype"]
            r = rep.resolve_genotype(resolver, g, lambda _: None)
            blocks = [rep.Block.model_validate(b) for b in g["blocks"]]
            parent, reason = assign_segregant_genes(
                pos.chromosome, pos.start, pos.end, blocks, 0.9, "confident_run"
            )
            parent_sb, _ = assign_segregant_genes(
                pos.chromosome, pos.start, pos.end, blocks, 0.9, "single_block"
            )
            p1, p2 = g["parent_1"], g["parent_2"]
            peters = (p1["peter_strain_id"], p2["peter_strain_id"])
            by_mask = np.zeros(len(parent), dtype=bool)
            for k, peter in enumerate(peters, start=1):
                sel = parent == k
                if peter is None:
                    by_mask |= sel
                else:
                    used.setdefault(peter, np.zeros(len(parent), dtype=bool))
                    used[peter] |= sel
            nonref = (parent > 0) & ~by_mask
            has_key = (r.key_ids >= 0).all(axis=1)
            differs = nonref & has_key & (r.key_ids != s288c).any(axis=1)
            no_allele = nonref & (r.key_gap[:, 2] == GapKind.NO_ISOLATE_ALLELE)
            counts = SegregantRecordCounts(
                dataset=name,
                raw_index=raw_index,
                cross=str(g["cross"]),
                segregant_id=str(g["segregant_id"]),
                parent_1=p1["name"],
                parent_1_peter=peters[0],
                parent_2=p2["name"],
                parent_2_peter=peters[1],
                n_assigned=int((parent > 0).sum()),
                n_assigned_by=int(by_mask.sum()),
                n_assigned_non_reference=int(nonref.sum()),
                n_non_reference_allele_differs_from_s288c=int(differs.sum()),
                n_non_reference_without_allele=int(no_allele.sum()),
                n_unassigned_no_block=int((reason == SegregantReason.NO_BLOCK).sum()),
                n_unassigned_low_posterior=int(
                    (reason == SegregantReason.LOW_POSTERIOR).sum()
                ),
                n_unassigned_straddle=int((reason == SegregantReason.STRADDLE).sum()),
                n_assigned_single_block_rule=int((parent_sb > 0).sum()),
                n_blocks=len(blocks),
            )
            if counts.n_assigned_non_reference != int(r.operator_mask.sum()):
                raise ValueError(f"data_{raw_index}: marks disagree with assignment")
            rows.append(
                counts.model_dump()
                | {"assignment": parent.tobytes(), "reason": reason.tobytes()}
            )
    return rows, used


def summarize(frame: pd.DataFrame) -> dict[str, Any]:
    """Mean, median, min, max of every count."""
    return {
        "records": len(frame),
        **{
            f: {
                "mean": round(float(frame[f].mean()), 1),
                "median": float(frame[f].median()),
                "min": int(frame[f].min()),
                "max": int(frame[f].max()),
            }
            for f in COUNT_FIELDS
        },
    }


def main() -> None:
    """Resolve every segregant record; write the parquet and the summary."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk", type=int, default=400)
    parser.add_argument("--limit", type=int, default=None, help="per dataset (smoke)")
    args = parser.parse_args()
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    t0 = time.time()
    cfg = Dataset043Config(data_root=data_root)
    index = json.loads(
        Path(osp.join(cfg.store_root, "raw", RAW_INDEX_NAME)).read_text()
    )["index"]
    items = [
        (name, i)
        for name in SEGREGANT_DATASETS
        for i in (index[name][: args.limit] if args.limit else index[name])
    ]
    chunks = [items[k : k + args.chunk] for k in range(0, len(items), args.chunk)]
    print(f"{len(items)} segregant records in {len(chunks)} chunks", flush=True)
    _init(data_root)  # loaded before the fork, shared copy-on-write
    rows: list[dict[str, Any]] = []
    used: dict[str, np.ndarray] = {}
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers) as pool:
        for k, (r, u) in enumerate(pool.imap_unordered(_work, chunks), start=1):
            rows += r
            for peter, mask in u.items():
                used[peter] = used.get(peter, np.zeros_like(mask)) | mask
            if k % 20 == 0 or k == len(chunks):
                print(f"{k}/{len(chunks)} chunks, {time.time() - t0:.0f} s", flush=True)
    frame = pd.DataFrame(rows).sort_values("raw_index").reset_index(drop=True)
    out = rep.rep_root(data_root)
    if args.limit is None:
        frame.to_parquet(out / "segregant_alleles.parquet", index=False)

    tables: ContextTables = _G["tables"]
    distinct: dict[str, Any] = {}
    for peter, mask in sorted(used.items()):
        c = tables.index(peter)
        keys = tables.key_ids[c][mask]
        with_allele = (keys >= 0).all(axis=1)
        distinct[peter] = {
            "genes_ever_assigned": int(mask.sum()),
            "genes_with_all_four_keys": int(with_allele.sum()),
            "distinct_protein_keys": int(np.unique(keys[keys[:, 2] >= 0, 2]).size),
            "distinct_calm_keys": int(np.unique(keys[keys[:, 3] >= 0, 3]).size),
            "genes_whose_allele_differs_from_s288c": int(
                (with_allele & (keys != tables.key_ids[0][mask]).any(axis=1)).sum()
            ),
        }
    counts = frame.drop(columns=["assignment", "reason"])
    summary: dict[str, Any] = {
        "generated_by": "experiments/043-expression-proteome-morphology/scripts/"
        "resolve_segregant_alleles_043.py",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "wall_seconds": round(time.time() - t0, 1),
        "store_version": active_store().version,
        "rule": "whole-gene containment in one confident run (consecutive blocks of "
        "one parent, every block posterior >= 0.9); single_block = the literal "
        "one-block form",
        "coordinates": "gene spans: SGD R64-4-1 GFF; blocks: the loaders' marker "
        "positions on the S288C reference",
        "n_genes": int(_G["resolver"].n_genes),
        "limit": args.limit,
        "per_dataset": {
            n: summarize(f) for n, f in counts.groupby("dataset", sort=True)
        },
        "per_cross": {
            f"{d}:{c}": {
                "parents": [
                    f"{f['parent_1'].iloc[0]} ({f['parent_1_peter'].iloc[0]})",
                    f"{f['parent_2'].iloc[0]} ({f['parent_2_peter'].iloc[0]})",
                ],
                **summarize(f),
            }
            for (d, c), f in counts.groupby(["dataset", "cross"], sort=True)
        },
        "distinct_parent_alleles_used": distinct,
    }
    res = Path(osp.dirname(osp.dirname(osp.abspath(__file__)))) / "results"
    name = active_store().result_name(
        "resolve_segregant_alleles_043"
        if args.limit is None
        else "resolve_segregant_alleles_043_smoke"
    )
    (res / name).write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["per_dataset"], indent=1))


if __name__ == "__main__":
    main()
