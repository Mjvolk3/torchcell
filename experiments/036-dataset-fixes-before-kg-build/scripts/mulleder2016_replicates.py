# experiments/036-dataset-fixes-before-kg-build/scripts/mulleder2016_replicates.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.mulleder2016_replicates]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/mulleder2016_replicates
"""Measure what issues #488 and #489 change in ``AminoAcidMulleder2016Dataset`` records.

Builds the dataset with the current loader into a SCRATCH root (``--scratch-root``; the
raw workbook is symlinked from the dev tree and ``process`` re-verifies its sha256),
then reads that store and the dev store
(``$DATA_ROOT/data/torchcell/amino_acid_mulleder2016``) record by record, resolving
interned ``$ref`` pointers, and reports:

- per-record ``n_replicates``: the distribution of the strain's ``data_raw`` row count
  and of the per-amino-acid counts, how many records changed, and the ORF x amino-acid
  cells whose count is below the strain's row count (blank raw cells);
- the reference ``n_replicates`` before and after;
- whether anything other than the two ``n_replicates`` fields differs;
- a diagnostic for #489: scikit-learn's ``MinCovDet`` (random_state 0) refit on the
  4,678 x 19 released mM matrix, compared with the released
  ``robust_summary_statistics`` mean (the paper used its own MCD implementation, so
  agreement supports, and does not prove, that the reference was fit on these strains).

Writes ``experiments/036-dataset-fixes-before-kg-build/results/mulleder2016_replicates.json``
and ``mulleder2016_replicates_multirow.csv`` (one row per strain with two or more raw rows).

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/mulleder2016_replicates.py \
        --scratch-root <scratch dir>/amino_acid_mulleder2016
"""

import argparse
import json
import os
import os.path as osp
import pickle
from collections import Counter
from typing import Any

import lmdb
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.covariance import MinCovDet

from torchcell.data.experiment_dataset import resolve_interned
from torchcell.datasets.scerevisiae.mulleder2016 import (
    _CONC_SHEET,
    _RAW_SHEET,
    _SUMMARY_SHEET,
    AMINO_ACIDS,
    DATA_FILENAME,
    AminoAcidMulleder2016Dataset,
)

RESULTS = "experiments/036-dataset-fixes-before-kg-build/results"


def read_store(root: str) -> list[dict[str, Any]]:
    """Every record of ``<root>/processed/lmdb`` in index order, interned refs resolved."""
    interned: dict[str, Any] = {}
    interned_dir = osp.join(root, "processed", "interned")
    if osp.isdir(interned_dir):
        ienv = lmdb.open(interned_dir, readonly=True, lock=False)
        with ienv.begin() as txn:
            interned = {k.decode(): pickle.loads(v) for k, v in txn.cursor()}
        ienv.close()
    env = lmdb.open(osp.join(root, "processed", "lmdb"), readonly=True, lock=False)
    with env.begin() as txn:
        n = txn.stat()["entries"]
        records = [
            resolve_interned(pickle.loads(txn.get(str(i).encode())), interned)
            for i in range(n)
        ]
    env.close()
    return records


def _without_n(record: dict[str, Any]) -> dict[str, Any]:
    """The record with both ``n_replicates`` fields blanked."""
    experiment = record["experiment"]
    reference = record["reference"]
    return {
        "experiment": {
            **experiment,
            "phenotype": {**experiment["phenotype"], "n_replicates": None},
        },
        "reference": {
            **reference,
            "phenotype_reference": {
                **reference["phenotype_reference"],
                "n_replicates": None,
            },
        },
        "publication": record["publication"],
    }


def main() -> None:
    """Build the scratch store, compare it with the dev store, write the results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch-root", required=True)
    args = parser.parse_args()
    load_dotenv()
    dev_root = osp.join(
        os.environ["DATA_ROOT"], "data/torchcell/amino_acid_mulleder2016"
    )
    raw_dir = osp.join(args.scratch_root, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    if not osp.exists(osp.join(raw_dir, DATA_FILENAME)):
        os.symlink(
            osp.join(dev_root, "raw", DATA_FILENAME), osp.join(raw_dir, DATA_FILENAME)
        )
    AminoAcidMulleder2016Dataset(root=args.scratch_root)

    dev = read_store(dev_root)
    new = read_store(args.scratch_root)
    assert len(dev) == len(new)

    row_counts: Counter[int] = Counter()
    cell_counts: Counter[int] = Counter()
    changed = 0
    other_fields_differ = 0
    below_row_count: list[dict[str, Any]] = []
    multirow: list[dict[str, Any]] = []
    workbook = osp.join(dev_root, "raw", DATA_FILENAME)
    raw = pd.read_excel(workbook, sheet_name=_RAW_SHEET)
    raw_rows = raw.groupby("ORF").size()
    for d, n in zip(dev, new, strict=True):
        orf = n["experiment"]["genotype"]["perturbations"][0]["systematic_gene_name"]
        assert (
            orf
            == d["experiment"]["genotype"]["perturbations"][0]["systematic_gene_name"]
        )
        n_rep = n["experiment"]["phenotype"]["n_replicates"]
        rows = int(raw_rows[orf])
        row_counts[rows] += 1
        cell_counts.update(n_rep.values())
        if n_rep != d["experiment"]["phenotype"]["n_replicates"]:
            changed += 1
        if _without_n(d) != _without_n(n):
            other_fields_differ += 1
        for aa, count in n_rep.items():
            if count < rows:
                below_row_count.append(
                    {"orf": orf, "amino_acid": aa, "n": count, "rows": rows}
                )
        if rows > 1:
            batches = raw[raw["ORF"] == orf]["batch"].astype(str).tolist()
            multirow.append(
                {
                    "orf": orf,
                    "n_raw_rows": rows,
                    "batches": ";".join(batches),
                    "min_n_replicates": min(n_rep.values()),
                }
            )

    conc = pd.read_excel(workbook, sheet_name=_CONC_SHEET)
    summary = pd.read_excel(workbook, sheet_name=_SUMMARY_SHEET).set_index("amino acid")
    released_mean = summary.loc[AMINO_ACIDS, "mean (mM)"].to_numpy()
    matrix = conc[AMINO_ACIDS].to_numpy()
    mcd = MinCovDet(random_state=0).fit(matrix)
    mcd_rel = np.abs(mcd.location_ / released_mean - 1)
    plain_rel = np.abs(matrix.mean(axis=0) / released_mean - 1)

    dev_ref_n = sorted(
        {
            v
            for r in dev
            for v in r["reference"]["phenotype_reference"]["n_replicates"].values()
        }
    )
    new_ref_n = sorted(
        {
            v
            for r in new
            for v in r["reference"]["phenotype_reference"]["n_replicates"].values()
        }
    )
    result = {
        "script": "experiments/036-dataset-fixes-before-kg-build/scripts/mulleder2016_replicates.py",
        "dev_root": dev_root,
        "scratch_root": args.scratch_root,
        "n_records_dev": len(dev),
        "n_records_scratch": len(new),
        "released_strains_by_data_raw_row_count": {
            str(k): v for k, v in sorted(row_counts.items())
        },
        "orf_x_amino_acid_cells_by_n_replicates": {
            str(k): v for k, v in sorted(cell_counts.items())
        },
        "n_records_experiment_n_replicates_changed": changed,
        "orf_x_amino_acid_cells_below_row_count": below_row_count,
        "n_strains_with_a_cell_below_row_count": len(
            {c["orf"] for c in below_row_count}
        ),
        "dev_reference_n_replicates_values": dev_ref_n,
        "scratch_reference_n_replicates_values": new_ref_n,
        "n_records_reference_n_replicates_changed": sum(
            r1["reference"]["phenotype_reference"]["n_replicates"]
            != r2["reference"]["phenotype_reference"]["n_replicates"]
            for r1, r2 in zip(dev, new, strict=True)
        ),
        "n_records_with_any_other_field_different": other_fields_differ,
        "n_released_strains_in_concentration_sheet": len(conc),
        "data_raw_qc_rows_without_orf": int(raw["ORF"].isna().sum()),
        "mcd_diagnostic": {
            "estimator": "sklearn.covariance.MinCovDet(random_state=0) on the released mM matrix",
            "n_strains": int(matrix.shape[0]),
            "max_abs_rel_diff_location_vs_released_mean": float(mcd_rel.max()),
            "median_abs_rel_diff_location_vs_released_mean": float(np.median(mcd_rel)),
            "max_abs_rel_diff_plain_mean_vs_released_mean": float(plain_rel.max()),
            "median_abs_rel_diff_plain_mean_vs_released_mean": float(
                np.median(plain_rel)
            ),
        },
    }
    os.makedirs(RESULTS, exist_ok=True)
    with open(osp.join(RESULTS, "mulleder2016_replicates.json"), "w") as handle:
        json.dump(result, handle, indent=2)
    pd.DataFrame(multirow).to_csv(
        osp.join(RESULTS, "mulleder2016_replicates_multirow.csv"), index=False
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
