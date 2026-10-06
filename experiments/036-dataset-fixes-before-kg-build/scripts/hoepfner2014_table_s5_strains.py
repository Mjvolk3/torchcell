# experiments/036-dataset-fixes-before-kg-build/scripts/hoepfner2014_table_s5_strains.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.hoepfner2014_table_s5_strains]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/hoepfner2014_table_s5_strains
"""Regenerate the Hoepfner 2014 HIP strain table from the mirrored Table S5 (#506).

The Hoepfner loader used to read its background-mutation flags from
``experiments/017-hoepfner-background-mutations/results/table_s5_affected_strains.csv``,
which 017 built by joining Table S5 onto an older HIP strain list, so every flagged strain
missing from that list fell out (YCR061W CL2, YJL017W CL4, YIL015C-A CL4) and a strain
sequenced as ``MUT`` with no cluster label (YCR035C) was never flagged. This script reads
the ``Data`` sheet of the mirrored ``si/Table_S5.xls`` (sha256-verified) directly and
writes EVERY HIP collection entry, not only the clustered ones, because the loader now
carries each strain's construction ``Lab`` / ``Batch`` / ``Plate`` / ``Row_Column`` on its
``StrainConstruction``.

Rules (verbatim tokens, nothing normalized beyond whitespace):

- Rows whose ``ORF`` cell is the placeholder ``identifier`` (control wells) are skipped
  and counted.
- ``flagged`` = a ``Cluster`` label is present (``CLn`` positional or ``CLn+``
  correlated, per the ``Legend`` sheet) OR ``Validation Result`` is ``MUT`` ("the
  sequence revealed the expected mutation").
- ``mutation`` maps the base cluster to the ``Table`` sheet's "Mutation identified".

Writes ``experiments/036-dataset-fixes-before-kg-build/results/``:

- ``hoepfner2014_table_s5_strains.csv``: one row per Table S5 entry (the loader input,
  sha256-pinned in ``hoepfner2014.py``);
- ``hoepfner2014_table_s5_flag_diff.json``: the flag set against the 017 CSV, the
  duplicate-ORF entries, and the entries whose ORF token is not a systematic name.

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/hoepfner2014_table_s5_strains.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import os.path as osp
import re

import pandas as pd
from dotenv import load_dotenv

CITATION_KEY = "hoepfnerHighresolutionChemicalDissection2014"
TABLE_S5_SHA256 = "b123dc3e87fc10d3b4256f449fcd2eb38c91d1779000278af5a1a788356624a2"
OLD_CSV = "experiments/017-hoepfner-background-mutations/results/table_s5_affected_strains.csv"
RESULTS = "experiments/036-dataset-fixes-before-kg-build/results"
OUT_CSV = osp.join(RESULTS, "hoepfner2014_table_s5_strains.csv")
OUT_JSON = osp.join(RESULTS, "hoepfner2014_table_s5_flag_diff.json")
SYSTEMATIC = re.compile(r"^Y[A-P][LR]\d{3}[WC](-[A-Z])?$")
COLUMNS = [
    "orf",
    "table_s5_id",
    "plate",
    "row_column",
    "cluster",
    "base_cluster",
    "is_positional",
    "mutation",
    "validation_result",
    "lab",
    "batch",
    "chromosome_arm",
    "flagged",
]


def _cell(value: object) -> str:
    """A Table S5 cell as a verbatim string; an empty cell is ''."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _base_cluster(label: str) -> tuple[str, bool]:
    """(base cluster CLn, positional?) of a label such as 'CL4', 'CL3+', 'CL4, CL3+'."""
    tokens = [token.strip() for token in label.split(",")]
    positional = [token for token in tokens if re.match(r"^CL[1-4]$", token)]
    if positional:
        return positional[0], True
    return tokens[0].rstrip("+"), False


def main() -> None:
    """Write the full HIP strain table and the flag-set diff against the 017 CSV."""
    load_dotenv()
    path = osp.join(
        os.environ["DATA_ROOT"], "torchcell-library", CITATION_KEY, "si", "Table_S5.xls"
    )
    with open(path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    if digest != TABLE_S5_SHA256:
        raise RuntimeError(f"{path} sha256 {digest} != pinned {TABLE_S5_SHA256}")

    summary = pd.read_excel(path, sheet_name="Table", header=None)
    mutations: dict[str, str] = {}
    for _, row in summary.iterrows():
        name = _cell(row[0])
        match = re.match(r"^Cluster (\d)$", name)
        if match:
            mutations[f"CL{match.group(1)}"] = _cell(row[6])

    data = pd.read_excel(path, sheet_name="Data", header=0)
    rows: list[dict[str, str]] = []
    n_placeholder = 0
    for _, row in data.iterrows():
        orf = _cell(row["ORF"])
        if orf == "identifier":
            n_placeholder += 1
            continue
        cluster = _cell(row["Cluster"])
        base, positional = _base_cluster(cluster) if cluster else ("", False)
        validation = _cell(row["Validation Result"])
        rows.append(
            {
                "orf": orf,
                "table_s5_id": _cell(row["ID"]),
                "plate": _cell(row["Plate"]),
                "row_column": _cell(row["Row_Column"]),
                "cluster": cluster,
                "base_cluster": base,
                "is_positional": str(positional),
                "mutation": mutations[base] if base else "",
                "validation_result": validation,
                "lab": _cell(row["Lab"]),
                "batch": _cell(row["Batch"]),
                "chromosome_arm": _cell(row["Chromosome Arm"]),
                "flagged": str(bool(cluster) or validation == "MUT"),
            }
        )
    os.makedirs(RESULTS, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    flagged = {row["orf"] for row in rows if row["flagged"] == "True"}
    with open(OLD_CSV) as handle:
        old = {row["orf"] for row in csv.DictReader(handle)}
    by_orf: dict[str, list[dict[str, str]]] = {}
    for entry in rows:
        by_orf.setdefault(entry["orf"], []).append(entry)
    duplicates = {orf: entries for orf, entries in by_orf.items() if len(entries) > 1}
    report = {
        "table_s5_sha256": TABLE_S5_SHA256,
        "n_entries": len(rows),
        "n_placeholder_rows_skipped": n_placeholder,
        "n_distinct_orf_tokens": len(by_orf),
        "n_orf_tokens_not_systematic": sorted(
            orf for orf in by_orf if not SYSTEMATIC.match(orf)
        ),
        "n_entries_with_cluster": sum(1 for row in rows if row["cluster"]),
        "n_entries_positional": sum(
            1 for row in rows if row["is_positional"] == "True"
        ),
        "n_entries_mut_without_cluster": sorted(
            row["orf"]
            for row in rows
            if row["validation_result"] == "MUT" and not row["cluster"]
        ),
        "n_flagged_orfs": len(flagged),
        "n_flagged_orfs_017_csv": len(old),
        "flagged_not_in_017_csv": sorted(flagged - old),
        "in_017_csv_not_flagged": sorted(old - flagged),
        "n_duplicate_orfs": len(duplicates),
        "duplicate_orfs_with_differing_lab_or_batch": sorted(
            orf
            for orf, entries in duplicates.items()
            if len({(e["lab"], e["batch"]) for e in entries}) > 1
        ),
        "duplicate_orfs_with_differing_cluster": sorted(
            orf
            for orf, entries in duplicates.items()
            if len({e["cluster"] for e in entries}) > 1
        ),
        "n_entries_without_batch": sum(1 for row in rows if not row["batch"]),
        "mutations": mutations,
    }
    with open(OUT_JSON, "w") as handle:
        json.dump(report, handle, indent=2)
    for key, value in report.items():
        print(f"{key}: {value}")
    print(f"wrote {OUT_CSV} and {OUT_JSON}")


if __name__ == "__main__":
    main()
