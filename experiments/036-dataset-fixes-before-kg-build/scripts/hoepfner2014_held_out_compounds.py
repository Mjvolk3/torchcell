# experiments/036-dataset-fixes-before-kg-build/scripts/hoepfner2014_held_out_compounds.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.hoepfner2014_held_out_compounds]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/hoepfner2014_held_out_compounds
"""Write the Hoepfner 2014 held-out compound list from the build's ledger (2026.10.15).

The loader holds out every CMB id with no released name, no released SMILES, no curated
compound-identity row and no ``RELEASED_IDENTITIES`` entry under
``UNIDENTIFIED_COMPOUND_RULE`` (database constitution principle 2). The dev build's
``<DATA_ROOT>/data/torchcell/env_chemgen_hoepfner2014/dropped_records.json`` lists each
held-out id with every column; this script flattens that list, one row per held-out
column, so the request for a public CMB-to-structure mapping can be sent as a table.

Columns: ``cmb_id``, ``assay`` (HIP or HOP), ``study``, ``concentration_um``, ``header``
(the deposited column header, verbatim) and ``n_scored_cells`` (records the column would
carry). The script raises when the ledger's per-arm accounting does not balance, so a
stale or hand-edited ledger cannot produce the table.

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/hoepfner2014_held_out_compounds.py
"""

from __future__ import annotations

import csv
import json
import os
import os.path as osp
from pathlib import Path

from dotenv import load_dotenv

from torchcell.datasets.scerevisiae.hoepfner2014 import (
    UNIDENTIFIED_COMPOUND_RULE,
    DroppedRecordReport,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT_CSV = RESULTS / "hoepfner2014_held_out_compounds.csv"
FIELDS = ["cmb_id", "assay", "study", "concentration_um", "header", "n_scored_cells"]


def ledger_path() -> str:
    """The dev-tree Hoepfner ledger under ``DATA_ROOT``."""
    return osp.join(
        os.environ["DATA_ROOT"],
        "data/torchcell/env_chemgen_hoepfner2014/dropped_records.json",
    )


def held_out_rows(report: DroppedRecordReport) -> list[dict[str, object]]:
    """One row per held-out column, ordered by CMB id then the ledger's column order."""
    if report.unidentified_compound_rule != UNIDENTIFIED_COMPOUND_RULE:
        raise ValueError(
            f"ledger rule {report.unidentified_compound_rule!r} is not "
            f"{UNIDENTIFIED_COMPOUND_RULE!r}"
        )
    for assay, row in report.accounting.items():
        held = sum(
            col.n_scored_cells
            for compound in report.unidentified_compounds_held_out
            for col in compound.columns
            if col.assay == assay
        )
        if held != row.n_unidentified_compound:
            raise ValueError(
                f"{assay}: held-out columns carry {held} cells, the accounting says "
                f"{row.n_unidentified_compound}"
            )
        total = (
            row.n_stored
            + row.n_not_orf_shaped
            + row.n_excluded_condition
            + row.n_unidentified_compound
        )
        if total != row.n_source_cells:
            raise ValueError(f"{assay}: {row.n_source_cells} cells != {total}")
    return [
        {
            "cmb_id": compound.cmb_id,
            "assay": col.assay,
            "study": col.study,
            "concentration_um": col.concentration_um,
            "header": col.header,
            "n_scored_cells": col.n_scored_cells,
        }
        for compound in report.unidentified_compounds_held_out
        for col in compound.columns
    ]


def main() -> None:
    """Read the ledger, write the CSV, print the per-arm totals."""
    load_dotenv()
    report = DroppedRecordReport.model_validate(
        json.loads(Path(ledger_path()).read_text())
    )
    rows = held_out_rows(report)
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"{len(report.unidentified_compounds_held_out)} CMB ids, {len(rows)} columns, "
        f"{sum(int(str(r['n_scored_cells'])) for r in rows)} scored cells -> {OUT_CSV}"
    )
    for assay, row in report.accounting.items():
        print(f"{assay}: stored {row.n_stored}, held out {row.n_unidentified_compound}")


if __name__ == "__main__":
    main()
