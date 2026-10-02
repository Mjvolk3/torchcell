# experiments/036-dataset-fixes-before-kg-build/scripts/sameith2015_sm_replicates.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.sameith2015_sm_replicates]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/sameith2015_sm_replicates
"""Measure the Sameith 2015 fixes for #478 (PubMed ID) and #479 (SM double-array fold).

Builds ``SmMicroarraySameith2015Dataset`` and ``DmMicroarraySameith2015Dataset`` into a
scratch root (``--build``; raw files are symlinked from the dev tree's ``raw/``, nothing in
the dev tree is written), then reads the scratch stores and the dev-tree stores
``$DATA_ROOT/data/torchcell/{sm,dm}_microarray_sameith2015`` read-only and reports, per
store:

- the record count and the stored PubMed IDs;
- for the SM stores, the arrays per record (the largest per-gene ``n_replicates`` of the
  record, which is the number of arrays averaged) and how many records hold more than the
  two arrays GEO has per single mutant;
- the own-gene sign check: the deleted gene's own stored ``expression_log2_ratio``,
  fraction negative and median over the records whose gene has a probe;
- for the SM ``preprocess/data.csv``, the number of rows and of rows with ``+`` in the
  title.

Writes ``results/sameith2015_sm_replicates.json`` (summary) and
``results/sameith2015_sm_replicates_records.csv`` (one row per SM record, dev vs scratch).

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/sameith2015_sm_replicates.py \
        --scratch-root <dir> [--build]
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import pickle
from typing import Any

import lmdb
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from torchcell.data.experiment_dataset import resolve_interned

RESULTS = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "results")
SM_SLUG = "sm_microarray_sameith2015"
DM_SLUG = "dm_microarray_sameith2015"
RAW_FILES = (
    "GSE42536_family.soft.gz",
    "GSE42536.pkl",
    "12915_2015_222_MOESM1_ESM.xlsx",
)


def read_store(root: str) -> list[dict[str, Any]]:
    """All records of a built store, interned sub-objects spliced back in, key order."""
    interned: dict[str, Any] = {}
    interned_dir = osp.join(root, "processed", "interned")
    if osp.isdir(interned_dir):
        ienv = lmdb.open(interned_dir, readonly=True, lock=False)
        with ienv.begin() as txn:
            interned = {k.decode(): pickle.loads(v) for k, v in txn.cursor()}
        ienv.close()
    env = lmdb.open(osp.join(root, "processed", "lmdb"), readonly=True, lock=False)
    with env.begin() as txn:
        rows = [(int(k.decode()), pickle.loads(v)) for k, v in txn.cursor()]
    env.close()
    return [resolve_interned(record, interned) for _, record in sorted(rows)]


def build(scratch_root: str, data_root: str) -> None:
    """Build both Sameith stores under ``scratch_root`` from the dev tree's raw files."""
    from torchcell.datasets.scerevisiae.sameith2015 import (
        DmMicroarraySameith2015Dataset,
        SmMicroarraySameith2015Dataset,
    )
    from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome

    genome = SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
        overwrite=False,
    )
    for slug, cls in (
        (SM_SLUG, SmMicroarraySameith2015Dataset),
        (DM_SLUG, DmMicroarraySameith2015Dataset),
    ):
        root = osp.join(scratch_root, slug)
        raw = osp.join(root, "raw")
        os.makedirs(raw, exist_ok=True)
        for name in RAW_FILES:
            target = osp.join(raw, name)
            if not osp.lexists(target):
                os.symlink(
                    osp.join(data_root, "data/torchcell", slug, "raw", name), target
                )
        dataset = cls(root=root, genome=genome)
        print(f"built {slug}: {len(dataset)} records at {root}")


def own_gene_sign(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Fraction negative and median of each record's own deleted genes' log2 ratios."""
    values = []
    for record in records:
        log2 = record["experiment"]["phenotype"]["expression_log2_ratio"]
        for perturbation in record["experiment"]["genotype"]["perturbations"]:
            gene = perturbation["systematic_gene_name"]
            if gene in log2 and np.isfinite(log2[gene]):
                values.append(float(log2[gene]))
    array = np.array(values)
    return {
        "n_gene_values": int(array.size),
        "fraction_negative": float(np.mean(array < 0)),
        "median": float(np.median(array)),
    }


def sm_rows(records: list[dict[str, Any]]) -> dict[str, int]:
    """Deleted gene -> arrays averaged into its record (largest per-gene n)."""
    out = {}
    for record in records:
        (perturbation,) = record["experiment"]["genotype"]["perturbations"]
        n = record["experiment"]["phenotype"]["n_replicates"]
        out[perturbation["systematic_gene_name"]] = int(max(n.values()))
    return out


def summarize(root: str, single: bool) -> dict[str, Any]:
    """Summary numbers for one store."""
    records = read_store(root)
    summary: dict[str, Any] = {
        "root": root,
        "n_records": len(records),
        "pubmed_ids": sorted({r["publication"]["pubmed_id"] for r in records}),
        "own_gene_sign": own_gene_sign(records),
    }
    if single:
        arrays = sm_rows(records)
        counts = pd.Series(list(arrays.values())).value_counts().sort_index()
        summary["arrays_per_record"] = {str(k): int(v) for k, v in counts.items()}
        summary["records_with_more_than_2_arrays"] = int(
            sum(v > 2 for v in arrays.values())
        )
        summary["total_arrays"] = int(sum(arrays.values()))
        table = pd.read_csv(osp.join(root, "preprocess", "data.csv"))
        summary["preprocess_rows"] = len(table)
        summary["preprocess_rows_with_plus"] = int(
            table["title"].str.contains("+", regex=False).sum()
        )
    return summary


def main() -> None:
    """Optionally build the scratch stores, then compare them with the dev tree."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scratch-root", required=True)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    if args.build:
        build(args.scratch_root, data_root)

    dev = {
        slug: osp.join(data_root, "data/torchcell", slug) for slug in (SM_SLUG, DM_SLUG)
    }
    new = {slug: osp.join(args.scratch_root, slug) for slug in (SM_SLUG, DM_SLUG)}
    result: dict[str, Any] = {
        "script": "experiments/036-dataset-fixes-before-kg-build/scripts/sameith2015_sm_replicates.py",
        "dev": {slug: summarize(dev[slug], slug == SM_SLUG) for slug in dev},
        "scratch": {slug: summarize(new[slug], slug == SM_SLUG) for slug in new},
    }

    dev_rows = sm_rows(read_store(dev[SM_SLUG]))
    new_rows = sm_rows(read_store(new[SM_SLUG]))
    genes = sorted(set(dev_rows) | set(new_rows))
    frame = pd.DataFrame(
        {
            "systematic_gene_name": genes,
            "arrays_dev": [dev_rows.get(g) for g in genes],
            "arrays_scratch": [new_rows.get(g) for g in genes],
        }
    )
    frame["changed"] = frame["arrays_dev"] != frame["arrays_scratch"]
    result["sm_records_changed_arrays"] = int(frame["changed"].sum())
    result["sm_genes_only_in_dev"] = sorted(set(dev_rows) - set(new_rows))
    result["sm_genes_only_in_scratch"] = sorted(set(new_rows) - set(dev_rows))

    os.makedirs(RESULTS, exist_ok=True)
    frame.to_csv(
        osp.join(RESULTS, "sameith2015_sm_replicates_records.csv"), index=False
    )
    with open(osp.join(RESULTS, "sameith2015_sm_replicates.json"), "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
