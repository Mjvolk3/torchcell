# experiments/036-dataset-fixes-before-kg-build/scripts/mulleder2016_per_replicate.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.mulleder2016_per_replicate]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/mulleder2016_per_replicate
"""Prove the per-culture records of ``AminoAcidMulleder2016Dataset`` against the release.

Tracking issue #1178. Reads three things, never writes a store:

- the converted store at ``--scratch-root`` (built by the branch's loader);
- the previous store at ``$DATA_ROOT/data/torchcell/amino_acid_mulleder2016`` (read only);
- the pinned workbook's ``data_raw`` sheet.

Reports the record count per measurement type; for the mM records, every key path that
differs from the previous store record by record (expected: only the ``replicate_id``
key, which PR #1181 added to the schema with default None) and the max absolute
difference of every stored mM value; for the raw uM records, the count per ORF and per
replicate position (the k-th culture of an ORF in sheet order), that each record's levels
equal the ``data_raw`` cells exactly with blanks left out, that ``replicate_id`` is the
row's ``identifier``, and the reference the complemented controls give.

Writes ``experiments/036-dataset-fixes-before-kg-build/results/mulleder2016_per_replicate.json``.

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/mulleder2016_per_replicate.py \
        --scratch-root <scratch dir>/amino_acid_mulleder2016
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import pickle
from collections import Counter
from typing import Any

import lmdb
import pandas as pd
from dotenv import load_dotenv

from torchcell.data.experiment_dataset import resolve_interned
from torchcell.datasets.scerevisiae.mulleder2016 import (
    _RAW_SHEET,
    AMINO_ACIDS,
    COMPLEMENTED_CONTROL_ORFS,
    DATA_FILENAME,
    MEASUREMENT_TYPE,
    RAW_MEASUREMENT_TYPE,
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


def differing_paths(old: Any, new: Any, path: str = "") -> list[str]:
    """Key paths where ``new`` differs from ``old``; a key on one side only is named."""
    if isinstance(old, dict) and isinstance(new, dict):
        out: list[str] = []
        for key in sorted(set(old) | set(new)):
            if key not in old:
                out.append(f"{path}/{key} (new key)")
            elif key not in new:
                out.append(f"{path}/{key} (removed key)")
            else:
                out += differing_paths(old[key], new[key], f"{path}/{key}")
        return out
    if isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        return [
            p
            for i, (a, b) in enumerate(zip(old, new, strict=True))
            for p in differing_paths(a, b, f"{path}[{i}]")
        ]
    return [] if old == new else [path]


def _orf(record: dict[str, Any]) -> str:
    (perturbation,) = record["experiment"]["genotype"]["perturbations"]
    return str(perturbation["systematic_gene_name"])


def main() -> None:
    """Compare the converted store with the release and the previous store."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch-root", required=True)
    args = parser.parse_args()
    load_dotenv()
    dev_root = osp.join(
        os.environ["DATA_ROOT"], "data/torchcell/amino_acid_mulleder2016"
    )
    old = read_store(dev_root)
    new = read_store(args.scratch_root)
    by_type = Counter(r["experiment"]["phenotype"]["measurement_type"] for r in new)
    mm = [
        r
        for r in new
        if r["experiment"]["phenotype"]["measurement_type"] == MEASUREMENT_TYPE
    ]
    cultures = [
        r
        for r in new
        if r["experiment"]["phenotype"]["measurement_type"] == RAW_MEASUREMENT_TYPE
    ]

    # mM records: the previous store's records, in the same order.
    assert len(mm) == len(old)
    path_counts: Counter[str] = Counter()
    mm_max_abs = 0.0
    for before, after in zip(old, mm, strict=True):
        path_counts.update(differing_paths(before, after))
        b = before["experiment"]["phenotype"]["metabolite_level"]
        a = after["experiment"]["phenotype"]["metabolite_level"]
        assert set(a) == set(b)
        mm_max_abs = max(mm_max_abs, max(abs(a[k] - b[k]) for k in a))

    # raw uM records against the data_raw cells.
    workbook = osp.join(dev_root, "raw", DATA_FILENAME)
    raw = pd.read_excel(workbook, sheet_name=_RAW_SHEET, dtype={"batch": str})
    strain_rows = raw[raw["ORF"].notna()].reset_index(drop=True)
    assert len(strain_rows) == len(cultures)
    raw_max_abs = 0.0
    n_blank_cells = 0
    position: Counter[int] = Counter()
    seen: Counter[str] = Counter()
    for (_, row), record in zip(strain_rows.iterrows(), cultures, strict=True):
        phenotype = record["experiment"]["phenotype"]
        assert phenotype["replicate_id"] == row["identifier"]
        assert _orf(record) == row["ORF"]
        expected = {aa: float(row[aa]) for aa in AMINO_ACIDS if pd.notna(row[aa])}
        n_blank_cells += len(AMINO_ACIDS) - len(expected)
        assert set(phenotype["metabolite_level"]) == set(expected)
        assert phenotype["n_replicates"] == dict.fromkeys(expected, 1)
        assert phenotype["metabolite_level_se"] is None
        raw_max_abs = max(
            raw_max_abs,
            max(abs(phenotype["metabolite_level"][k] - v) for k, v in expected.items()),
        )
        seen[row["ORF"]] += 1
        position[seen[row["ORF"]]] += 1
        reference = record["reference"]["phenotype_reference"]
        assert reference["replicate_id"] is None
        assert set(reference["metabolite_level"]) == set(expected)

    released = {_orf(r) for r in mm}
    per_orf = Counter(seen[orf] for orf in seen if orf in released)
    absent_orfs = sorted(orf for orf in seen if orf not in released)
    controls = strain_rows[strain_rows["ORF"].isin(COMPLEMENTED_CONTROL_ORFS)]
    full_reference = next(
        r["reference"]["phenotype_reference"]
        for r in cultures
        if len(r["reference"]["phenotype_reference"]["metabolite_level"]) == 19
    )
    result = {
        "script": "experiments/036-dataset-fixes-before-kg-build/scripts/mulleder2016_per_replicate.py",
        "previous_root": dev_root,
        "scratch_root": args.scratch_root,
        "n_records_previous": len(old),
        "n_records_scratch": len(new),
        "n_records_by_measurement_type": dict(sorted(by_type.items())),
        "mm_records": {
            "key_paths_differing_from_previous_store": dict(
                sorted(path_counts.items())
            ),
            "max_abs_diff_metabolite_level_mM": mm_max_abs,
        },
        "raw_records": {
            "n_data_raw_rows": len(raw),
            "n_qc_rows_not_records": int(raw["ORF"].isna().sum()),
            "n_records": len(cultures),
            "n_records_of_released_orfs": sum(
                seen[orf] for orf in seen if orf in released
            ),
            "n_records_of_orfs_absent_from_mm_sheet": sum(
                seen[orf] for orf in absent_orfs
            ),
            "n_orfs_absent_from_mm_sheet": len(absent_orfs),
            "released_orfs_by_culture_count": {
                str(k): v for k, v in sorted(per_orf.items())
            },
            "records_by_replicate_position": {
                str(k): v for k, v in sorted(position.items())
            },
            "max_abs_diff_vs_data_raw_uM": raw_max_abs,
            "n_blank_cells_left_out": n_blank_cells,
            "replicate_id_unique": len(
                {r["experiment"]["phenotype"]["replicate_id"] for r in cultures}
            )
            == len(cultures),
            "control_rows": controls[["identifier", "ORF", "batch"]].to_dict("records"),
            "control_reference_n_replicates": sorted(
                set(full_reference["n_replicates"].values())
            ),
        },
    }
    os.makedirs(RESULTS, exist_ok=True)
    with open(osp.join(RESULTS, "mulleder2016_per_replicate.json"), "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
