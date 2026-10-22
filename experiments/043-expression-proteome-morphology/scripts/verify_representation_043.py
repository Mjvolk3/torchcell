# experiments/043-expression-proteome-morphology/scripts/verify_representation_043.py
# [[experiments.043-expression-proteome-morphology.scripts.verify_representation_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/verify_representation_043.py
"""Every record of the 043 raw store resolved to its input representation, counted.

One pass over all 173,617 records (the 15 datasets of the raw store, the training set,
the Ohya hold-out and the Sameith singles), each through ``Raw043Dataset.__getitem__``
with ``representation=RepresentationConfig(route="A")``, i.e. the exact path training
reads. Isolate records are resolved a second time under route B. Keys only: no vector is
read, so the pass does not wait on the GPU embed jobs.

Per dataset and genotype class it reports how many records resolve completely (no gene
slot and no extra token carries a gap), how many carry each typed gap and how many genes
per record carry it, the distributions of marks (change type not reference), of the
operator mask and of extra tokens per record, position gaps, and for isolates the
route-B edit-list size and how many genes have different inputs under A and B. Segregant
records are cross-checked against ``segregant_alleles.parquet``.

Writes ``results/verify_representation_043.json``, ``results/verify_representation_043.md``
(the markdown table that goes into the experiment note) and the per-record table
``$DATA_ROOT/.../representation_043/verify_records.parquet``.

    python experiments/043-expression-proteome-morphology/scripts/verify_representation_043.py --workers 32
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

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import representation_043 as rep  # noqa: E402
from dataset_043 import (  # noqa: E402
    DUPLICATE_DATASETS,
    HOLDOUT_DATASETS,
    TRAINING_DATASETS,
    Dataset043Config,
    Raw043Dataset,
)

from torchcell.data.representation_043 import (  # noqa: E402
    GapKind,
    PositionGap,
    RecordRepresentation,
    RepresentationConfig,
    RepresentationResolver,
    SegregantReason,
)
from torchcell.models.genotype_tokens_043 import (  # noqa: E402
    CHANGE_TYPE_INDEX,
    ChangeType,
)

_G: dict[str, Any] = {}
CHANGE_NAMES = {v: k.value for k, v in CHANGE_TYPE_INDEX.items()}
GENE_GAPS = [g for g in GapKind if g not in (GapKind.NONE,)]


def _stats(r: RecordRepresentation) -> dict[str, Any]:
    """The per-record counts of one representation."""
    gap_any = r.key_gap > 0
    out: dict[str, Any] = {
        "genotype_class": r.genotype_class,
        "context_id": r.context_id,
        "n_marks": int(len(r.marked)),
        "n_operator": int(r.operator_mask.sum()),
        "n_extra": len(r.extra_tokens),
        "n_extra_accessory": sum(t.kind == "accessory_orf" for t in r.extra_tokens),
        "n_extra_reporter": sum(t.kind == "reporter" for t in r.extra_tokens),
        "n_genes_with_gap": int(gap_any.any(axis=1).sum()),
        "n_marked_outside_index": r.n_marked_outside_index,
    }
    for c, name in CHANGE_NAMES.items():
        if name != "reference":
            out[f"marks_{name}"] = int((r.change_type == c).sum())
    for g in GENE_GAPS:
        out[f"genes_gap_{g.name.lower()}"] = int((r.key_gap == g).any(axis=1).sum())
    extra_gaps = [k for t in r.extra_tokens for k in t.gaps if k]
    out["extra_gap_accessory_without_row"] = sum(
        all(k == GapKind.ACCESSORY_WITHOUT_ROW for k in t.gaps)
        for t in r.extra_tokens
        if t.kind == "accessory_orf"
    )
    out["extra_gap_missing_reporter_window"] = sum(
        t.gaps[0] == GapKind.MISSING_REPORTER_WINDOW
        for t in r.extra_tokens
        if t.kind == "reporter"
    )
    for p in PositionGap:
        if p != PositionGap.NONE:
            out[f"pos_gap_{p.name.lower()}"] = int((r.position_gap == p).sum())
    out["chrom_copy_not_one"] = int((r.chrom_copy != 1).sum())
    out["complete"] = bool(not gap_any.any() and not extra_gaps)
    if r.segregant_reason is not None:
        for s in SegregantReason:
            out[f"seg_{s.name.lower()}"] = int((r.segregant_reason == s).sum())
    return out


def _init(data_root: str) -> None:
    load_dotenv()
    cfg = Dataset043Config(
        data_root=data_root,
        datasets=TRAINING_DATASETS + HOLDOUT_DATASETS + DUPLICATE_DATASETS,
        representation=RepresentationConfig(route="A"),
    )
    ds = Raw043Dataset(cfg)
    assert ds.resolver is not None
    _G["ds"] = ds
    _G["resolver_b"] = RepresentationResolver(
        RepresentationConfig(route="B"),
        ds.resolver.pos,
        ds.resolver.tables,
        ds.resolver.reporter,
        cfg.segregant_parent_vocab,
    )


def _work(chunk: list[int]) -> list[dict[str, Any]]:
    """Resolve a chunk of dataset positions."""
    ds: Raw043Dataset = _G["ds"]
    rb: RepresentationResolver = _G["resolver_b"]
    s288c = ds.resolver.tables.key_ids[0].astype(np.int64)  # type: ignore[union-attr]
    rows = []
    for i in chunk:
        sample = ds[i]
        r: RecordRepresentation = sample["representation"]
        row = {"dataset": sample["dataset"], "raw_index": sample["raw_index"]}
        row |= _stats(r)
        if r.genotype_class == "isolate":
            genotype = ds.record(sample["raw_index"])["experiment"]["genotype"]
            b = rep.resolve_genotype(rb, genotype, ds._gene_idx)
            a_keys = np.where(r.key_ids >= 0, r.key_ids, -1 - r.key_gap)
            b_keys = np.where(b.key_ids >= 0, b.key_ids, -1 - b.key_gap)
            row["route_b_edit_list"] = int(
                (
                    (b.change_type == CHANGE_TYPE_INDEX[ChangeType.variant])
                    | (b.change_type == CHANGE_TYPE_INDEX[ChangeType.loss_of_function])
                ).sum()
            )
            row["route_b_operator"] = int(b.operator_mask.sum())
            row["genes_differ_route_a_b"] = int((a_keys != b_keys).any(axis=1).sum())
            row["genes_route_a_keys_differ_s288c"] = int(
                ((r.key_ids >= 0) & (r.key_ids != s288c)).any(axis=1).sum()
            )
            row["route_b_genes_with_gap"] = int((b.key_gap > 0).any(axis=1).sum())
        rows.append(row)
    return rows


def _dist(s: pd.Series) -> dict[str, float]:
    return {
        "min": float(s.min()),
        "median": float(s.median()),
        "mean": round(float(s.mean()), 2),
        "max": float(s.max()),
    }


def summarize(frame: pd.DataFrame) -> dict[str, Any]:
    """The per (dataset, class) block of the json."""
    out: dict[str, Any] = {
        "records": len(frame),
        "records_complete": int(frame["complete"].sum()),
        "records_with_any_gap": int((~frame["complete"]).sum()),
        "marks_per_record": _dist(frame["n_marks"]),
        "operator_mask_per_record": _dist(frame["n_operator"]),
        "extra_tokens_per_record": _dist(frame["n_extra"]),
        "genes_with_gap_per_record": _dist(frame["n_genes_with_gap"]),
        "marks_by_change_type_mean": {
            c[len("marks_") :]: round(float(frame[c].mean()), 2)
            for c in frame.columns
            if c.startswith("marks_") and frame[c].sum() > 0
        },
        "gene_gaps": {
            c[len("genes_gap_") :]: {
                "records_with": int((frame[c] > 0).sum()),
                "genes_per_record": _dist(frame[c]),
            }
            for c in frame.columns
            if c.startswith("genes_gap_") and frame[c].sum() > 0
        },
        "extra_token_gaps": {
            c[len("extra_gap_") :]: {
                "records_with": int((frame[c] > 0).sum()),
                "tokens_total": int(frame[c].sum()),
            }
            for c in frame.columns
            if c.startswith("extra_gap_")
        },
        "position_gaps": {
            c[len("pos_gap_") :]: {
                "records_with": int((frame[c] > 0).sum()),
                "genes_per_record_mean": round(float(frame[c].mean()), 2),
            }
            for c in frame.columns
            if c.startswith("pos_gap_") and frame[c].sum() > 0
        },
        "records_with_chromosome_copy_not_one": int(
            (frame["chrom_copy_not_one"] > 0).sum()
        ),
        "marks_outside_index_total": int(frame["n_marked_outside_index"].sum()),
        "contexts": int(frame["context_id"].nunique()),
    }
    if "route_b_edit_list" in frame and frame["route_b_edit_list"].notna().any():
        f = frame[frame["route_b_edit_list"].notna()]
        out["route_a_vs_b"] = {
            "route_b_edit_list_per_record": _dist(f["route_b_edit_list"]),
            "route_b_operator_per_record": _dist(f["route_b_operator"]),
            "genes_differ_route_a_b_per_record": _dist(f["genes_differ_route_a_b"]),
            "genes_route_a_keys_differ_s288c_per_record": _dist(
                f["genes_route_a_keys_differ_s288c"]
            ),
            "route_b_genes_with_gap_per_record": _dist(f["route_b_genes_with_gap"]),
        }
    if "seg_assigned" in frame and frame["seg_assigned"].notna().any():
        out["segregant_rule"] = {
            c[len("seg_") :]: _dist(frame[c])
            for c in frame.columns
            if c.startswith("seg_")
        }
    return out


def markdown(summary: dict[str, Any]) -> str:
    """The table that goes into the experiment note."""
    head = (
        "| Dataset | Class | Records | Complete | No allele | Locus gap | Short flank "
        "| Splice | Absent (A) | Seg. unassigned | Acc. w/o row | Reporter w/o window "
        "| Marks | Operator | Extra tokens |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|\n"
    )
    lines = []
    for key, b in summary["per_dataset_class"].items():
        d, c = key.split(" | ")
        g = b["gene_gaps"]

        def med(k: str, g: dict[str, Any] = g) -> str:
            return f"{g[k]['genes_per_record']['median']:.0f}" if k in g else "0"

        def rng(x: dict[str, float]) -> str:
            return f"{x['median']:.0f} ({x['min']:.0f}-{x['max']:.0f})"

        lines.append(
            f"| {d.replace('Dataset', '')} | {c} | {b['records']:,} | "
            f"{b['records_complete']:,} | {med('no_isolate_allele')} | "
            f"{med('locus_gap')} | {med('flank_short')} | {med('splice_unverified')} | "
            f"{med('absent_in_context')} | {med('segregant_unassigned')} | "
            f"{b['extra_token_gaps']['accessory_without_row']['tokens_total']} | "
            f"{b['extra_token_gaps']['missing_reporter_window']['tokens_total']} | "
            f"{rng(b['marks_per_record'])} | {rng(b['operator_mask_per_record'])} | "
            f"{rng(b['extra_tokens_per_record'])} |"
        )
    return head + "\n".join(lines) + "\n"


def main() -> None:
    """One pass over the store."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk", type=int, default=300)
    parser.add_argument("--limit", type=int, default=None, help="per dataset (smoke)")
    args = parser.parse_args()
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    t0 = time.time()
    _init(data_root)  # before the fork, shared copy-on-write
    ds: Raw043Dataset = _G["ds"]
    positions = list(range(len(ds)))
    if args.limit is not None:
        seen: dict[str, int] = {}
        keep = []
        for i, (name, _) in enumerate(ds.items):
            seen[name] = seen.get(name, 0) + 1
            if seen[name] <= args.limit:
                keep.append(i)
        positions = keep
    chunks = [
        positions[k : k + args.chunk] for k in range(0, len(positions), args.chunk)
    ]
    print(f"{len(positions)} records in {len(chunks)} chunks", flush=True)
    rows: list[dict[str, Any]] = []
    with mp.get_context("fork").Pool(args.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(_work, chunks), start=1):
            rows += r
            if k % 25 == 0 or k == len(chunks):
                print(f"{k}/{len(chunks)} chunks, {time.time() - t0:.0f} s", flush=True)
    frame = pd.DataFrame(rows).sort_values("raw_index").reset_index(drop=True)
    if len(frame) != len(positions) or frame["raw_index"].duplicated().any():
        raise RuntimeError("records lost or duplicated in the pass")

    # cross-check the segregant records against the allele-resolution parquet
    seg_check: dict[str, Any] = {}
    seg_path = rep.rep_root(data_root) / "segregant_alleles.parquet"
    if seg_path.exists():
        seg = pd.read_parquet(
            seg_path,
            columns=[
                "raw_index",
                "n_assigned",
                "n_assigned_non_reference",
                "n_unassigned_no_block",
                "n_unassigned_low_posterior",
                "n_unassigned_straddle",
            ],
        )
        f = frame[frame["genotype_class"] == "segregant"].merge(
            seg, on="raw_index", how="left", validate="1:1"
        )
        seg_check = {
            "records": len(f),
            "missing_from_parquet": int(f["n_assigned"].isna().sum()),
            "assigned_mismatch": int((f["seg_assigned"] != f["n_assigned"]).sum()),
            "marks_mismatch": int(
                (f["n_marks"] != f["n_assigned_non_reference"]).sum()
            ),
            "unassigned_mismatch": int(
                (
                    (f["seg_no_block"] != f["n_unassigned_no_block"])
                    | (f["seg_low_posterior"] != f["n_unassigned_low_posterior"])
                    | (f["seg_straddle"] != f["n_unassigned_straddle"])
                ).sum()
            ),
        }

    summary: dict[str, Any] = {
        "generated_by": "experiments/043-expression-proteome-morphology/scripts/"
        "verify_representation_043.py",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "wall_seconds": round(time.time() - t0, 1),
        "records": len(frame),
        "limit": args.limit,
        "config": "RepresentationConfig(route='A'), isolates also under route='B'; "
        "segregant rule confident_run at posterior >= 0.9",
        "complete_definition": "no gene slot of the four encoders and no extra token "
        "carries a typed gap (position gaps are reported separately)",
        "records_complete": int(frame["complete"].sum()),
        "per_dataset_class": {
            f"{d} | {c}": summarize(f)
            for (d, c), f in frame.groupby(["dataset", "genotype_class"], sort=True)
        },
        "segregant_cross_check": seg_check,
    }
    res = Path(osp.dirname(osp.dirname(osp.abspath(__file__)))) / "results"
    stem = "verify_representation_043" + ("" if args.limit is None else "_smoke")
    (res / f"{stem}.json").write_text(json.dumps(summary, indent=1))
    (res / f"{stem}.md").write_text(markdown(summary))
    if args.limit is None:
        frame.to_parquet(
            rep.rep_root(data_root) / "verify_records.parquet", index=False
        )
    print(markdown(summary))
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "per_dataset_class"}, indent=1
        )
    )


if __name__ == "__main__":
    main()
