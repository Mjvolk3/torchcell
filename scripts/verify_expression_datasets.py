# scripts/verify_expression_datasets.py
# [[scripts.verify_expression_datasets]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/verify_expression_datasets
"""Run the WS5 L0-L4 record-level gate over the microarray expression datasets.

Loads the built LMDB records for Sameith2015 (single + double mutant) and
Kemmeren2014 from ``$DATA_ROOT``, runs
:func:`torchcell.verification.expression.verify_expression_dataset` on each, adds an
L4 cross-source check (all datasets share the same platform gene universe), writes a
``verification_report.json`` sibling of each ``experiment_reference_index.json``, and
exits non-zero if any level fails.

Usage:
    ~/miniconda3/envs/torchcell/bin/python scripts/verify_expression_datasets.py
"""

from __future__ import annotations

import os
import os.path as osp
import pickle
import sys
from typing import Any

import lmdb
from dotenv import load_dotenv

from torchcell.verification.expression import (
    measured_gene_universe,
    verify_expression_dataset,
)
from torchcell.verification.levels import l4_cross_source
from torchcell.verification.report import Provenance, VerificationReport

# dataset key -> (root under $DATA_ROOT, expected record count, provenance)
DATASETS: dict[str, dict[str, Any]] = {
    "dm_microarray_sameith2015": {
        "root": "data/torchcell/dm_microarray_sameith2015",
        "expected_count": 72,
        "provenance": Provenance(
            source_uri="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE42536",
            citation_key="sameithHighresolutionGeneExpression2015",
            method="GEOparse GSE42536 (GPL11232), dye-swap-corrected log2(mutant/refpool)",
            page="Cell Reports, Expression Profiling; SI 'Double mutants - info'",
        ),
    },
    "sm_microarray_sameith2015": {
        "root": "data/torchcell/sm_microarray_sameith2015",
        "expected_count": 82,
        "provenance": Provenance(
            source_uri="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE42536",
            citation_key="sameithHighresolutionGeneExpression2015",
            method="GEOparse GSE42536 (GPL11232), dye-swap-corrected log2(mutant/refpool)",
            page="Cell Reports, Expression Profiling; SI 'Single mutants - info'",
        ),
    },
    "microarray_kemmeren2014": {
        "root": "data/torchcell/microarray_kemmeren2014",
        "expected_count": 1450,
        "provenance": Provenance(
            source_uri="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE42527",
            citation_key="kemmerenLargeScaleGeneticPerturbations2014",
            method="GEOparse GSE42527/GSE42526 (GPL11232), log2(mutant/wt-reference)",
            page="Cell 2014, Expression Profiling",
        ),
    },
}


def load_records(abs_root: str) -> list[dict[str, Any]]:
    """Read every LMDB entry under ``<abs_root>/processed/lmdb``."""
    env = lmdb.open(
        osp.join(abs_root, "processed", "lmdb"), readonly=True, lock=False
    )
    records: list[dict[str, Any]] = []
    with env.begin() as txn:
        cursor = txn.cursor()
        for _, value in cursor:
            records.append(pickle.loads(value))
    env.close()
    return records


def main() -> int:
    """Verify all expression datasets, write reports, return a shell exit code."""
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]

    reports: dict[str, VerificationReport] = {}
    universes: dict[str, set[str]] = {}

    for name, spec in DATASETS.items():
        abs_root = osp.join(data_root, spec["root"])
        records = load_records(abs_root)
        report = verify_expression_dataset(
            records,
            dataset_name=name,
            provenance=spec["provenance"],
            expected_count=spec["expected_count"],
        )
        reports[name] = report
        universes[name] = measured_gene_universe(records)

    # L4 cross-source: every dataset measures the same platform gene universe. Pair
    # each dataset against the first as a reference; encode presence as 1.0/0.0 so a
    # dropped gene surfaces as a disagreement.
    ref_name = next(iter(universes))
    ref_universe = universes[ref_name]
    for name, universe in universes.items():
        if name == ref_name:
            continue
        all_genes = sorted(ref_universe | universe)
        shared = [
            (g, 1.0 if g in ref_universe else 0.0, 1.0 if g in universe else 0.0)
            for g in all_genes
        ]
        result = l4_cross_source(shared, tol=0.0)
        result = result.model_copy(
            update={"name": f"gene_universe_vs_{ref_name}"}
        )
        reports[name].add(result)

    # Persist each report as a sibling of experiment_reference_index.json.
    all_passed = True
    for name, report in reports.items():
        abs_root = osp.join(data_root, DATASETS[name]["root"])
        out = osp.join(abs_root, "preprocess", "verification_report.json")
        with open(out, "w") as handle:
            handle.write(report.model_dump_json(indent=2))
        print(report.summary())
        print(f"  -> wrote {out}\n")
        all_passed = all_passed and report.passed

    print("=" * 60)
    print("ALL DATASETS PASS" if all_passed else "SOME DATASETS FAILED")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
