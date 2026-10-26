# experiments/043-expression-proteome-morphology/scripts/route_compare_043.py
# [[experiments.043-expression-proteome-morphology.scripts.route_compare_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/route_compare_043
"""Compare two prepared stores that differ only in the isolate route (A vs B).

Route A: each natural isolate is its own reference context (every gene token carries the
isolate's allele) plus marks for its accessory ORFs and engineered deletions. Route B:
every record is the S288C context plus an edit list; an isolate's ``variant`` and
``loss_of_function`` genes carry the isolate's keys, its ``absent`` genes keep the S288C
keys, and every non-reference gene is a mark (``representation_043.RepresentationResolver``).

Reads ``train_043_prep/<prep>_routeA`` and ``<prep>_routeB`` and checks, per dataset:

* the records (dataset, raw index, role, unit, environment, ontology profile, target row)
  are identical, and the gene-level target matrix ``Y`` is byte-identical;
* for every dataset with no isolate record, every mark array is identical record by record
  and every record reads context 0 (S288C) in both stores;
* route B has one context (S288C);

and reports the marks-per-record distribution (min, median, mean, max) and the
``ChangeType`` histogram of the marks, per dataset, for both routes. Output:
``results/route_compare_043.json`` (the store version's suffix appended).

    python experiments/043-expression-proteome-morphology/scripts/route_compare_043.py --prep prep_v4
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
from typing import Any

import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from store_version_043 import store_paths  # noqa: E402
from train_043 import PreparedStore043  # noqa: E402

from torchcell.models.genotype_tokens_043 import ChangeType  # noqa: E402

EXPERIMENT = "043-expression-proteome-morphology"
REC_FIELDS = (
    "rec_raw_index",
    "rec_role",
    "rec_env",
    "rec_yrow",
    "rec_genes2",
    "rec_unit",
    "rec_onto",
)
MARK_FIELDS = (
    "mark_gene",
    "mark_on_graph",
    "mark_change",
    "mark_parent",
    "mark_keys",
    "mark_gaps",
    "mark_replicon",
    "mark_relmid",
    "mark_chrom_copy",
    "mark_gene_copy",
    "mark_pgap",
)
CHANGE_NAMES = [c.value for c in ChangeType]


def mark_summary(store: PreparedStore043, sel: np.ndarray) -> dict[str, Any]:
    """Marks per record and the change-type histogram over records ``sel``."""
    n = (store.mark_ptr[sel + 1] - store.mark_ptr[sel]).astype(np.int64)
    idx = np.concatenate(
        [np.arange(store.mark_ptr[r], store.mark_ptr[r + 1]) for r in sel]
    ).astype(np.int64)
    ch = store.mark_change[idx]
    on = store.mark_on_graph[idx]
    hist = {CHANGE_NAMES[c]: int((ch == c).sum()) for c in np.unique(ch).astype(int)}
    per_record_mean = {
        CHANGE_NAMES[c]: round(float((ch == c).sum()) / len(sel), 1)
        for c in np.unique(ch).astype(int)
    }
    return {
        "records": len(sel),
        "marks_min": int(n.min()),
        "marks_median": float(np.median(n)),
        "marks_mean": round(float(n.mean()), 1),
        "marks_max": int(n.max()),
        "marks_total": int(n.sum()),
        "on_graph_total": int(on.sum()),
        "change_hist": hist,
        "change_per_record_mean": per_record_mean,
        "contexts": sorted(np.unique(store.rec_context[sel]).astype(int).tolist())[:5],
        "n_contexts": int(len(np.unique(store.rec_context[sel]))),
    }


def marks_equal(a: PreparedStore043, b: PreparedStore043, sel: np.ndarray) -> bool:
    """Every mark array identical, record by record, over ``sel``."""
    na = a.mark_ptr[sel + 1] - a.mark_ptr[sel]
    nb = b.mark_ptr[sel + 1] - b.mark_ptr[sel]
    if not np.array_equal(na, nb):
        return False
    ia = np.concatenate([np.arange(a.mark_ptr[r], a.mark_ptr[r + 1]) for r in sel])
    ib = np.concatenate([np.arange(b.mark_ptr[r], b.mark_ptr[r + 1]) for r in sel])
    ia, ib = ia.astype(np.int64), ib.astype(np.int64)
    return all(
        np.array_equal(
            getattr(a, f)[ia], getattr(b, f)[ib], equal_nan=f.endswith("copy")
        )
        if getattr(a, f).dtype.kind == "f"
        else np.array_equal(getattr(a, f)[ia], getattr(b, f)[ib])
        for f in MARK_FIELDS
    )


def main() -> None:
    """Load both stores, check, summarize, write the JSON."""
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", default="prep_v4")
    ap.add_argument("--store-version", default="v2")
    args = ap.parse_args()
    gen = store_paths(args.store_version)
    root = osp.join(
        os.environ["DATA_ROOT"],
        "data/torchcell/experiments",
        EXPERIMENT,
        "train_043_prep",
    )
    a = PreparedStore043.load(osp.join(root, f"{args.prep}_routeA"))
    b = PreparedStore043.load(osp.join(root, f"{args.prep}_routeB"))
    if a.meta.route != "A" or b.meta.route != "B":
        raise ValueError(f"routes {a.meta.route}, {b.meta.route}")
    if a.meta.datasets != b.meta.datasets:
        raise ValueError("dataset order differs")
    if not np.array_equal(a.rec_dataset, b.rec_dataset):
        raise ValueError("record datasets differ")
    for f in REC_FIELDS:
        if not np.array_equal(getattr(a, f), getattr(b, f)):
            raise ValueError(f"{f} differs between the routes")
    if not np.array_equal(a.Y, b.Y, equal_nan=True):
        raise ValueError("Y differs between the routes")
    if b.ctx_keys.shape[0] != 1:
        raise ValueError(f"route B holds {b.ctx_keys.shape[0]} contexts, expected 1")
    out: dict[str, Any] = {
        "prep": args.prep,
        "store_version": gen.version,
        "routeA_meta": {
            "slurm_job_id": a.meta.slurm_job_id,
            "commit": a.meta.torchcell_commit,
        },
        "routeB_meta": {
            "slurm_job_id": b.meta.slurm_job_id,
            "commit": b.meta.torchcell_commit,
        },
        "contexts": {"A": int(a.ctx_keys.shape[0]), "B": int(b.ctx_keys.shape[0])},
        "records_identical": True,
        "Y_identical": True,
        "datasets": {},
    }
    for d, short in enumerate(a.meta.datasets):
        sel = np.nonzero(a.rec_dataset == d)[0]
        if len(sel) == 0:
            continue
        isolate_like = bool((a.rec_context[sel] != 0).any())
        row = {
            "has_own_context_in_A": isolate_like,
            "A": mark_summary(a, sel),
            "B": mark_summary(b, sel),
            "marks_identical": marks_equal(a, b, sel),
        }
        if not isolate_like and not row["marks_identical"]:
            raise ValueError(f"{short}: no isolate record, but the marks differ")
        if (b.rec_context[sel] != 0).any():
            raise ValueError(f"{short}: a route-B record is not on S288C")
        out["datasets"][short] = row
        print(
            f"{short:11s} n={len(sel):6d} A marks med {row['A']['marks_median']:7.1f} "
            f"max {row['A']['marks_max']:5d} | B marks med {row['B']['marks_median']:7.1f} "
            f"max {row['B']['marks_max']:5d} | identical {row['marks_identical']} | "
            f"B hist {row['B']['change_hist']}"
        )
    path = osp.join(
        osp.dirname(osp.dirname(osp.abspath(__file__))),
        "results",
        gen.result_name("route_compare_043"),
    )
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(path)


if __name__ == "__main__":
    main()
