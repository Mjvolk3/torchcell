#!/usr/bin/env python
# experiments/database/scripts/nichols2011_release_checks.py
# [[experiments.database.scripts.nichols2011_release_checks]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/nichols2011_release_checks.py
r"""Measurements on the Nichols 2011 release that the loader, its note and its PR cite.

Reads, read-only, the owner's deposit of the Cell supplemental information
(``$DATA_ROOT/torchcell-raw/nicholsPhenotypicLandscapeBacterial2011/si/``, #1046) and
Shiver 2016's S1 Dataset (``torchcell-raw/shiverChemicalGenomicScreenNeglected2016``):

1. **G3 inventory.** Every deposited file against its manifest pin, its real format
   (BIFF, OOXML under an ``.xls`` name, PDF) and, for each table, its sheets, title
   cell, header row and cell census (numeric, text, empty). The gate's own inventory
   counts worksheet rows only in files named ``.xlsx``, so it counts 0 here; these are
   the counts.
2. **Tables S3 to S7**, the five tables the loader does not read: what each is.
3. **Table S1**: 115 rows, the ``Screens`` column and its ``Total Screens`` cell.
4. **Table S2**: shape, numeric and empty cells, the value range, the per-column median
   and IQR (the "re-scaled ... to a standard normal distribution (IQR=1.35)" check),
   the label suffixes and shapes, and the twelve labels that head two rows (with the
   agreement of their two profiles).
5. **G4 against Shiver 2016**: Shiver's S1 Dataset re-analyzes Nichols's screens as its
   batch 0, which ``shiver2016.py`` deliberately does not load. Each Shiver batch-0
   condition is matched to the Table S2 column it correlates with best over the strains
   both releases name by the same symbol (runner-up kept), and the overlap is measured
   by (strain, condition) and by value: how many Shiver batch-0 cells land on a
   (strain, column) this loader stores, and how close the two numbers are. No genome
   annotation is read anywhere in this script.

Run from the repo root (a worktree: ``PYTHONPATH=<worktree>``)::

    python experiments/database/scripts/nichols2011_release_checks.py

Writes ``experiments/database/results/nichols2011_release_checks.json``.
"""

from __future__ import annotations

import csv
import json
import os
import re
import statistics
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import openpyxl
import xlrd
from dotenv import load_dotenv

from torchcell.datamodels.schema import ReleasedKind
from torchcell.datasets.ecoli import nichols2011 as n
from torchcell.datasets.ecoli import shiver2016 as s
from torchcell.literature.manifest import sha256_file

RESULTS = Path("experiments/database/results")
SI_FILES = tuple(f"si/mmc{i}.xls" for i in range(1, 8)) + ("si/mmc8.pdf",)
_LABEL_SHAPES = (
    ("ECK number and symbol", re.compile(r"^ECK\d{4}-[A-Z0-9]+$")),
    ("ECK number and symbol, primed", re.compile(r"^ECK\d{4}-[A-Z0-9]+'$")),
    ("ECK number only", re.compile(r"^ECK\d{4}$")),
    ("SPA-tagged essential gene", re.compile(r" - SPA$")),
    ("several ECK numbers", re.compile(r"^ECK\d{4}[/-]\d")),
)


def file_format(path: Path) -> str:
    """BIFF (OLE2), OOXML (zip) or PDF, by magic bytes rather than by name."""
    head = path.read_bytes()[:8]
    if head.startswith(b"\xd0\xcf\x11\xe0"):
        return "BIFF (OLE2 .xls)"
    if head.startswith(b"PK"):
        return "OOXML (zip)"
    if head.startswith(b"%PDF"):
        return "PDF"
    return f"unknown {head!r}"


def census_xls(path: Path) -> list[dict[str, Any]]:
    """Every sheet of a BIFF workbook: title cell, header row, cell kinds."""
    book = xlrd.open_workbook(str(path))
    out = []
    for sheet in book.sheets():
        kinds: Counter[str] = Counter()
        for r in range(sheet.nrows):
            for c in range(sheet.ncols):
                kinds[
                    {xlrd.XL_CELL_TEXT: "text", xlrd.XL_CELL_NUMBER: "number"}.get(
                        sheet.cell_type(r, c), "empty"
                    )
                ] += 1
        out.append(
            {
                "sheet": sheet.name,
                "rows": sheet.nrows,
                "columns": sheet.ncols,
                "title_cell": sheet.cell_value(0, 0),
                "header_row": [str(v) for v in sheet.row_values(1)],
                "cells": dict(sorted(kinds.items())),
                "data_rows": sum(
                    1
                    for r in range(2, sheet.nrows)
                    if any(str(v).strip() for v in sheet.row_values(r))
                ),
                "rows_after_title": sum(
                    1
                    for r in range(1, sheet.nrows)
                    if any(str(v).strip() for v in sheet.row_values(r))
                ),
            }
        )
    return out


def census_xlsx(path: Path) -> list[dict[str, Any]]:
    """Every sheet of an OOXML workbook, opened from a handle (its name says .xls)."""
    with path.open("rb") as handle:
        book = openpyxl.load_workbook(handle, read_only=True, data_only=True)
        out = []
        for ws in book.worksheets:
            kinds: Counter[str] = Counter()
            rows = 0
            columns = 0
            first: tuple[Any, ...] = ()
            second: tuple[Any, ...] = ()
            for i, row in enumerate(ws.iter_rows(values_only=True)):
                rows += 1
                columns = max(columns, len(row))
                if i == 0:
                    first = row
                if i == 1:
                    second = row
                for v in row:
                    kinds[
                        "empty"
                        if v is None
                        else "number"
                        if isinstance(v, int | float)
                        else "text"
                    ] += 1
            out.append(
                {
                    "sheet": ws.title,
                    "rows": rows,
                    "columns": columns,
                    "title_cell": first[0],
                    "header_first_cells": [str(v) for v in second[:4]],
                    "cells": dict(sorted(kinds.items())),
                }
            )
        book.close()
    return out


def inventory(root: Path) -> dict[str, Any]:
    """G3: every deposited file, its pin, its format and its census."""
    manifest = n.load_manifest(str(root.parents[1]))
    pins = {r.path: r.sha256 for r in manifest.files}
    files = []
    for rel in SI_FILES:
        path = root / rel
        record: dict[str, Any] = {
            "path": rel,
            "bytes": path.stat().st_size,
            "sha256_manifest": pins[rel],
            "sha256_disk": sha256_file(path),
            "format": file_format(path),
        }
        if record["format"].startswith("BIFF"):
            record["sheets"] = census_xls(path)
        elif record["format"].startswith("OOXML"):
            record["sheets"] = census_xlsx(path)
        files.append(record)
    return {
        "files": files,
        "all_intact": all(f["sha256_manifest"] == f["sha256_disk"] for f in files),
    }


def screen_list(root: Path) -> dict[str, Any]:
    """Table S1: its rows, the Screens column and the Total Screens cell."""
    rows = n.read_screen_list(root / n.SCREEN_LIST_REL)
    book = xlrd.open_workbook(str(root / n.SCREEN_LIST_REL))
    sheet = book.sheet_by_index(0)
    total = [
        (sheet.cell_value(r, 4), sheet.cell_value(r, 5))
        for r in range(sheet.nrows)
        if sheet.cell_value(r, 4) == "Total Screens"
    ]
    return {
        "condition_rows": len(rows),
        "screens_sum": sum(screens for _, _, screens in rows),
        "total_screens_cell": total,
        "note": "the per-row Screens cells sum to the Total Screens cell; summing the "
        "whole column (the total row included) gives twice that",
    }


def score_matrix(root: Path) -> tuple[dict[str, Any], n.ScoreMatrix]:
    """Table S2 measured: shape, cells, per-column standardization, label shapes."""
    matrix = n.read_score_matrix(root / n.MATRIX_REL)
    values = np.array(
        [[np.nan if v is None else v for v in row] for row in matrix.scores]
    )
    q1, med, q3 = np.nanpercentile(values, [25, 50, 75], axis=0)
    iqr = q3 - q1
    counts = Counter(matrix.strains)
    duplicates = []
    for label, k in sorted(counts.items()):
        if k < 2:
            continue
        i, j = (idx for idx, lab in enumerate(matrix.strains) if lab == label)
        a, b = values[i], values[j]
        both = ~np.isnan(a) & ~np.isnan(b)
        duplicates.append(
            {
                "label": label,
                "rows": [i, j],
                "co_present": int(both.sum()),
                "identical": int((a[both] == b[both]).sum()),
                "pearson_r": round(float(np.corrcoef(a[both], b[both])[0, 1]), 4),
            }
        )
    shapes: Counter[str] = Counter()
    for label in matrix.strains:
        for name, pattern in _LABEL_SHAPES:
            if pattern.search(label):
                shapes[name] += 1
                break
        else:
            shapes["other: alleles, linked controls, DAS, truncations"] += 1
    suffixes = Counter(label.rpartition(" - ")[2] for label in matrix.conditions)
    return (
        {
            "strain_rows": len(matrix.strains),
            "distinct_strain_labels": len(counts),
            "condition_columns": len(matrix.conditions),
            "distinct_condition_labels": len(set(matrix.conditions)),
            "numeric_cells": int((~np.isnan(values)).sum()),
            "empty_cells": int(np.isnan(values).sum()),
            "min": round(float(np.nanmin(values)), 6),
            "max": round(float(np.nanmax(values)), 6),
            "per_column_median": {
                "min": round(float(med.min()), 4),
                "max": round(float(med.max()), 4),
            },
            "per_column_iqr": {
                "min": round(float(iqr.min()), 4),
                "median": round(float(np.median(iqr)), 4),
                "max": round(float(iqr.max()), 4),
            },
            "condition_label_suffixes": dict(suffixes),
            "strain_label_shapes": dict(sorted(shapes.items())),
            "labels_heading_two_rows": duplicates,
        },
        matrix,
    )


def derived_tables(root: Path) -> dict[str, Any]:
    """Tables S3 to S7: their title, header and data-row count, and what each is."""
    out: dict[str, Any] = {}
    for rel, what in n.DERIVED_SI_TABLES.items():
        (sheet,) = census_xls(root / rel)
        out[rel] = {"is": what, **sheet}
    return out


def shiver_overlap(matrix: n.ScoreMatrix, data_root: str) -> dict[str, Any]:
    """Shiver 2016's batch-0 block against the records this loader stores.

    Release against release, no annotation read. Columns: each Shiver batch-0 condition
    is matched to the Table S2 column with the highest Pearson r over the strains both
    carry (best and runner-up kept), so no unit or spelling of either release has to be
    reconciled by hand. Strains: a Shiver bare-symbol column and a Table S2 single-gene
    deletion row are one strain when their released symbols agree ignoring case
    (``hchA`` and ``ECK1963-HCHA``), each symbol unique on both sides. A Shiver cell
    overlaps when its strain and its condition both land on a stored (row, column).
    """
    path = Path(data_root) / "torchcell-raw" / s.CITATION_KEY / s.DATA_REL
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    header = rows[0][1:]
    batch0 = [r for r in rows[1:] if r[0].endswith("{0}")]
    shiver_counts = Counter(label.casefold() for label in header)
    shiver_col = {
        label.casefold(): j
        for j, label in enumerate(header)
        if shiver_counts[label.casefold()] == 1 and "-" not in label
    }
    strains = n.strain_rows(matrix.strains)
    deletions = [
        r
        for r in strains
        if r.label.kind is ReleasedKind.deletion
        and len(r.label.ecks) == 1
        and r.label.symbol is not None
        and r.replicate_id is None
    ]
    nichols_counts = Counter(str(r.label.symbol).casefold() for r in deletions)
    nichols_row = {
        str(r.label.symbol).casefold(): r.index
        for r in deletions
        if nichols_counts[str(r.label.symbol).casefold()] == 1
    }
    shared = sorted(
        (shiver_col[k], nichols_row[k]) for k in shiver_col if k in nichols_row
    )
    nv = np.array([[np.nan if v is None else v for v in row] for row in matrix.scores])
    sv = np.array([[float(c) if c != "" else np.nan for c in r[1:]] for r in batch0])
    s_idx = np.array([j for j, _ in shared])
    n_idx = np.array([i for _, i in shared])
    matches: list[dict[str, Any]] = []
    pairs_compared = []
    for k, row in enumerate(batch0):
        a = sv[k, s_idx]
        scores = []
        for c in range(len(matrix.conditions)):
            b = nv[n_idx, c]
            ok = ~np.isnan(a) & ~np.isnan(b)
            if ok.sum() < 100:
                continue
            scores.append((float(np.corrcoef(a[ok], b[ok])[0, 1]), c, int(ok.sum())))
        scores.sort(reverse=True)
        best, runner = scores[0], scores[1]
        b = nv[n_idx, best[1]]
        ok = ~np.isnan(a) & ~np.isnan(b)
        diff = np.abs(a[ok] - b[ok])
        pairs_compared.append((a[ok], b[ok]))
        matches.append(
            {
                "shiver_condition": row[0],
                "nichols_column": matrix.conditions[best[1]],
                "pearson_r": round(best[0], 4),
                "runner_up": matrix.conditions[runner[1]],
                "runner_up_r": round(runner[0], 4),
                "cells_compared": best[2],
                "identical": int((diff == 0).sum()),
                "within_0.01": int((diff <= 0.01).sum()),
                "median_abs_diff": round(float(np.median(diff)), 4),
                "max_abs_diff": round(float(diff.max()), 4),
            }
        )
    a_all = np.concatenate([p[0] for p in pairs_compared])
    b_all = np.concatenate([p[1] for p in pairs_compared])
    d_all = np.abs(a_all - b_all)
    columns_hit = Counter(m["nichols_column"] for m in matches)
    best_r = [float(m["pearson_r"]) for m in matches]
    return {
        "shiver_batch0_conditions": len(batch0),
        "shiver_batch0_nonblank_cells": int((~np.isnan(sv)).sum()),
        "shiver_gene_columns": len(header),
        "shiver_bare_symbol_columns": len(shiver_col),
        "strains_in_both_by_released_symbol": len(shared),
        "nichols_columns_matched": len(columns_hit),
        "nichols_columns_matched_twice": sorted(
            c for c, k in columns_hit.items() if k > 1
        ),
        "overlapping_cells": int(len(d_all)),
        "identical_values": int((d_all == 0).sum()),
        "within_0.01": int((d_all <= 0.01).sum()),
        "pearson_r_pooled": round(float(np.corrcoef(a_all, b_all)[0, 1]), 4),
        "median_abs_diff": round(float(np.median(d_all)), 4),
        "best_r_per_condition": {
            "min": round(min(best_r), 4),
            "median": round(statistics.median(best_r), 4),
            "max": round(max(best_r), 4),
        },
        "matches": matches,
    }


def main() -> None:
    """Measure, print a summary and write the results JSON."""
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    root = n.raw_mirror_dir(data_root)
    s2, matrix = score_matrix(root)
    out: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "citation_key": n.CITATION_KEY,
        "g3_inventory": inventory(root),
        "table_s1": screen_list(root),
        "table_s2": s2,
        "tables_s3_to_s7": derived_tables(root),
        "g4_shiver_batch0": shiver_overlap(matrix, data_root),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    target = RESULTS / "nichols2011_release_checks.json"
    target.write_text(json.dumps(out, indent=2, default=str) + "\n")
    g4 = out["g4_shiver_batch0"]
    print(
        f"G3 intact: {out['g3_inventory']['all_intact']}; Table S1 rows "
        f"{out['table_s1']['condition_rows']}, screens {out['table_s1']['screens_sum']}"
    )
    print(
        f"Table S2 {s2['strain_rows']} x {s2['condition_columns']}, numeric "
        f"{s2['numeric_cells']}, empty {s2['empty_cells']}, IQR {s2['per_column_iqr']}"
    )
    print(
        f"Shiver batch 0: {g4['shiver_batch0_conditions']} conditions, "
        f"{g4['shiver_batch0_nonblank_cells']} cells; overlapping {g4['overlapping_cells']}"
        f" cells, identical {g4['identical_values']}, within 0.01 {g4['within_0.01']}, "
        f"pooled r {g4['pearson_r_pooled']}, median |d| {g4['median_abs_diff']}"
    )
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
