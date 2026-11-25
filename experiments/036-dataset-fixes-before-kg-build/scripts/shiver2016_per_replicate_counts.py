# experiments/036-dataset-fixes-before-kg-build/scripts/shiver2016_per_replicate_counts.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.shiver2016_per_replicate_counts]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/shiver2016_per_replicate_counts
"""Prove the Shiver 2016 doubled-column records against the release and the old store.

Shiver 2016's S1 Dataset heads two gene columns with each of 12 labels; ``yfiO*`` is a
point allele (still dropped), and the other 11 labels' 22 columns were dropped by a rule
until 2026-11-24 and are now stored, told apart by ``replicate_id`` =
``S1 Dataset column <n>``. This script measures, from the pinned release bytes:

- each doubled label's two release fields and their non-blank cells over the 57 study
  conditions;
- what the two columns are, against Nichols 2011 Table S2 (raw mirror), which releases
  the same twelve labels as two rows each: every Shiver column's batch-0 profile (Shiver's
  re-analysis of Nichols's screens) correlated with both Table S2 rows of its label, over
  the conditions the Nichols release checks match by profile, plus the two columns'
  correlation with each other over the 57 study conditions;

and, given two built stores in scratch roots (never the shared dev tree):

- records before and after, records per ``replicate_id``, and the total;
- that every before-record has an identical after-record (same experiment and reference
  dicts) under the key (released label, screen_id), and that no after-record without a
  ``replicate_id`` is new;
- the mean-reproduction check: after-records grouped by (released label, screen_id) and
  averaged, against the before-store's value; the max absolute difference is stated, and
  a group the before-store never held (a doubled label) is counted, not compared.

Writes ``experiments/036-dataset-fixes-before-kg-build/results/shiver2016_per_replicate_counts.json``.

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/shiver2016_per_replicate_counts.py \
        --before-root <scratch>/before/ecoli_env_chemgen_shiver2016 \
        --after-root <scratch>/after/ecoli_env_chemgen_shiver2016
"""

import argparse
import csv
import importlib
import json
import os
import os.path as osp
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from numpy.typing import NDArray

import torchcell.datasets.ecoli.nichols2011 as n
import torchcell.datasets.ecoli.shiver2016 as s
from torchcell.verification.runners import stream_records

RESULTS = "experiments/036-dataset-fixes-before-kg-build/results"
#: The Nichols release checks hold the condition match of Shiver's batch 0 to Table S2.
sys.path.insert(0, "experiments/database/scripts")
release_checks: Any = importlib.import_module("nichols2011_release_checks")


def _value(cell: str) -> float:
    """A release cell as a float, NaN when blank."""
    return float(cell) if cell != "" else float("nan")


def _pearson(a: NDArray[np.float64], b: NDArray[np.float64]) -> tuple[float, int]:
    """Pearson r over the cells both vectors carry, and that count."""
    ok = ~np.isnan(a) & ~np.isnan(b)
    return round(float(np.corrcoef(a[ok], b[ok])[0, 1]), 4), int(ok.sum())


def _nichols_key(label: str) -> str:
    """The Shiver spelling a Table S2 doubled label corresponds to, case-folded."""
    if label.startswith("ECK2593"):
        return "yfio*"
    symbol = label.split("-")[1].rstrip("'").casefold()
    return "bame" if symbol == "smpa" else symbol


def release_measurements(data_root: str) -> dict[str, Any]:
    """The doubled columns of the release, and their match to Nichols 2011 Table S2."""
    path = Path(data_root) / s.RAW_DIR_REL / s.DATA_REL
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    header = rows[0][1:]
    study = [r for r in rows[1:] if s._batch_of(r[0]) in s.STUDY_BATCHES]
    batch0 = [r for r in rows[1:] if s._batch_of(r[0]) == 0]
    counts = Counter(header)
    doubled = {
        label: [j for j, x in enumerate(header) if x == label]
        for label, k in counts.items()
        if k > 1
    }

    table_s2 = n.read_score_matrix(Path(data_root) / n.RAW_DIR_REL / n.MATRIX_REL)
    overlap = release_checks.shiver_overlap(table_s2, data_root)
    condition = {
        m["shiver_condition"]: table_s2.conditions.index(m["nichols_column"])
        for m in overlap["matches"]
    }
    scores = np.array(
        [[np.nan if v is None else v for v in row] for row in table_s2.scores]
    )
    s2_counts = Counter(table_s2.strains)
    s2_rows = {
        _nichols_key(label): [i for i, x in enumerate(table_s2.strains) if x == label]
        for label, k in s2_counts.items()
        if k > 1
    }

    labels: dict[str, Any] = {}
    own: list[float] = []
    other: list[float] = []
    within: list[float] = []
    for label, columns in doubled.items():
        nichols_rows = s2_rows[label.casefold()]
        entries = []
        for position, j in enumerate(columns):
            profile = np.array([_value(r[j + 1]) for r in batch0])
            r_rows = []
            for i in nichols_rows:
                table = np.array([scores[i, condition[r[0]]] for r in batch0])
                r_rows.append(_pearson(profile, table))
            own.append(r_rows[position][0])
            other.append(r_rows[1 - position][0])
            entries.append(
                {
                    "release_field": j + s.FIRST_GENE_FIELD,
                    "replicate_id": s.column_replicate_id(j),
                    "nonblank_study_cells": sum(1 for r in study if r[j + 1] != ""),
                    "batch0_r_with_table_s2_rows": {
                        f"Table S2 row {i + n.FIRST_DATA_ROW}": {"r": r, "cells": c}
                        for i, (r, c) in zip(nichols_rows, r_rows, strict=True)
                    },
                }
            )
        a = np.array([_value(r[columns[0] + 1]) for r in study])
        b = np.array([_value(r[columns[1] + 1]) for r in study])
        r_pair, n_pair = _pearson(a, b)
        within.append(r_pair)
        labels[label] = {
            "columns": entries,
            "study_r_between_the_two_columns": {"r": r_pair, "cells": n_pair},
        }
    stored = [label for label in doubled if label != "yfiO*"]
    return {
        "release_sha256": s.DATA_SHA256,
        "table_s2_sha256": n.MATRIX_SHA256,
        "doubled_labels": len(doubled),
        "doubled_labels_stored": len(stored),
        "doubled_columns_stored": 2 * len(stored),
        "nonblank_cells_stored": sum(
            e["nonblank_study_cells"] for lab in stored for e in labels[lab]["columns"]
        ),
        "batch0_conditions_matched_to_table_s2": len(condition),
        "own_row_r": {"min": min(own), "max": max(own)},
        "other_row_r": {"min": min(other), "max": max(other)},
        "study_r_between_columns": {"min": min(within), "max": max(within)},
        "every_column_matches_its_own_row_better": all(
            o > x for o, x in zip(own, other, strict=True)
        ),
        "labels": labels,
    }


def _key(record: dict[str, Any]) -> tuple[str, str]:
    """(released label, screen_id): the strain column and condition of a record."""
    experiment = record["experiment"]
    (perturbation,) = experiment["genotype"]["perturbations"]
    return (
        perturbation["identifier_mapping"]["source_identifier"],
        experiment["phenotype"]["screen_id"],
    )


def store_check(before_root: str, after_root: str) -> dict[str, Any]:
    """Record counts, identity of the old records, and the mean-reproduction check."""
    before: dict[tuple[str, str], dict[str, Any]] = {}
    for record in stream_records(before_root):
        assert record["experiment"]["phenotype"]["replicate_id"] is None
        key = _key(record)
        assert key not in before, f"two before-records on {key}"
        before[key] = record
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    per_replicate: Counter[str] = Counter()
    unchanged = 0
    n_after = 0
    for record in stream_records(after_root):
        n_after += 1
        key = _key(record)
        phenotype = record["experiment"]["phenotype"]
        groups[key].append(phenotype["environment_response"])
        replicate = phenotype["replicate_id"]
        if replicate is None:
            old = before[key]
            assert old["experiment"] == record["experiment"], key
            assert old["reference"] == record["reference"], key
            unchanged += 1
        else:
            assert key not in before, f"{key} was stored before without replicate_id"
            per_replicate[replicate] += 1
    worst = 0.0
    compared = 0
    for key, values in groups.items():
        if key in before:
            compared += 1
            old_value = before[key]["experiment"]["phenotype"]["environment_response"]
            worst = max(worst, abs(statistics.fmean(values) - old_value))
    assert compared == len(before)
    return {
        "records_before": len(before),
        "records_after": n_after,
        "after_records_identical_to_a_before_record": unchanged,
        "after_records_with_replicate_id": sum(per_replicate.values()),
        "records_per_replicate_id": dict(sorted(per_replicate.items())),
        "groups_after": len(groups),
        "groups_compared_with_before": compared,
        "groups_new_doubled_labels": len(groups) - compared,
        "group_mean_max_abs_diff": worst,
    }


def main() -> None:
    """Measure, print a summary and write the results JSON."""
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-root", required=True)
    parser.add_argument("--after-root", required=True)
    args = parser.parse_args()
    data_root = os.environ["DATA_ROOT"]
    out = {
        "release": release_measurements(data_root),
        "stores": store_check(args.before_root, args.after_root),
    }
    os.makedirs(RESULTS, exist_ok=True)
    path = osp.join(RESULTS, "shiver2016_per_replicate_counts.json")
    with open(path, "w") as handle:
        json.dump(out, handle, indent=2)
    release, stores = out["release"], out["stores"]
    print(
        f"doubled labels {release['doubled_labels']} (stored "
        f"{release['doubled_labels_stored']}, {release['nonblank_cells_stored']} cells); "
        f"own-row r {release['own_row_r']}, other-row r {release['other_row_r']}, "
        f"between columns {release['study_r_between_columns']}"
    )
    print(
        f"records {stores['records_before']} -> {stores['records_after']}; "
        f"with replicate_id {stores['after_records_with_replicate_id']}; "
        f"unchanged {stores['after_records_identical_to_a_before_record']}; "
        f"group-mean max abs diff {stores['group_mean_max_abs_diff']}"
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
