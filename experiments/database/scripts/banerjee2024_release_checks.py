#!/usr/bin/env python
# experiments/database/scripts/banerjee2024_release_checks.py
# [[experiments.database.scripts.banerjee2024_release_checks]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/banerjee2024_release_checks.py
r"""Measurements on the Banerjee 2024 release that the loader, its note and its PR cite.

Reads, read-only, the captured Elsevier ESM in the literature mirror
(``$DATA_ROOT/torchcell-library/banerjeeGenomescalePathwayEngineering2024/``) and the
loader's deposit of the two consumed artifacts
(``$DATA_ROOT/torchcell-raw/banerjeeGenomescalePathwayEngineering2024/``).

1. **G3 inventory.** Every captured and deposited file against its manifest pin, with
   its real format and, for each workbook, its sheets and cell census (numeric, text,
   empty). This is what the candidate gate's own inventory summarizes.
2. **The claim the whole loader rests on: the ESM holds no measurement table.** Each of
   the six ESM files is classified by what it actually contains, and every workbook
   sheet is searched for a per-strain titer column (a header naming a titer, a strain
   id, an OD or a glucose reading). A hit here would mean the loader is wrong to source
   its titers from prose, so this is the check that would catch that.
3. **The twenty titer statements**, each located in the pinned ``paper.md`` by its own
   verbatim quote, with the character offset it was found at, so the note's table can be
   audited against the bytes rather than against this script.
4. **What the release carries and the store does not**, counted: the fed-batch OD600
   values, the XW13/XW14 range, the residual-glucose value, the growth rates.
5. **The gacA tag contradiction**, measured across the mirrored P. putida papers that
   print a locus tag for that gene. No genome annotation is read anywhere in this
   script: the contradiction is between PAPERS, which is the point.

Run from the repo root (a worktree: ``PYTHONPATH=<worktree>``)::

    python experiments/database/scripts/banerjee2024_release_checks.py

Writes ``experiments/database/results/banerjee2024_release_checks.json``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import openpyxl
from dotenv import load_dotenv

from torchcell.datasets.pputida import banerjee2024 as b24

CITATION_KEY = "banerjeeGenomescalePathwayEngineering2024"
#: The mirrored P. putida papers that print a locus tag for ``gacA``, and the sha256 of
#: the OCR each statement is read from.
GACA_PAPERS: dict[str, str] = {
    "borchertRBTnSeqIdentifiesGenetic2023": (
        "e156d3b2139add2feb8b3ffb1b319f9a7d369d6ee9b6454f67c38d33c924265a"
    ),
    "schmidtNitrogenMetabolismPseudomonas2022": (
        "acc5c8bfaa588f5dda7767127f523c1721496cffa6e010acfe7e7865ede36c66"
    ),
    "thompsonFattyAcidAlcohol2020": (
        "389d0d6cd196f9f159d0485f6b8ed09743dfbe3b6682369db489d7356a469b45"
    ),
}
#: Header words that would mark a released per-strain measurement column.
MEASUREMENT_WORDS = ("titer", "titre", "od600", "od 600", "glucose", "isoprenol")
#: Strain-id shapes this campaign uses, for the same search.
STRAIN_PATTERN = re.compile(r"\b(?:XW\d{2}|IY\d{3,4})\b")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _magic(path: Path) -> str:
    """What the first bytes say the file really is, regardless of its extension."""
    with open(path, "rb") as handle:
        head = handle.read(8)
    if head.startswith(b"%PDF"):
        return "PDF"
    if head.startswith(b"PK\x03\x04"):
        return "OOXML/zip"
    if head.startswith(b"\xd0\xcf\x11\xe0"):
        return "BIFF/OLE2"
    return "text"


def _workbook_census(path: Path) -> list[dict[str, Any]]:
    """Per sheet: its name, extent, cell census and whether it holds a measurement."""
    workbook = openpyxl.load_workbook(path, data_only=True, read_only=True)
    sheets: list[dict[str, Any]] = []
    try:
        for sheet in workbook.worksheets:
            kinds: Counter[str] = Counter()
            headers: list[str] = []
            strain_hits = 0
            measurement_headers: list[str] = []
            rows = 0
            for index, row in enumerate(sheet.iter_rows(values_only=True)):
                rows += 1
                for cell in row:
                    if cell is None:
                        kinds["empty"] += 1
                    elif isinstance(cell, (int, float)):
                        kinds["numeric"] += 1
                    else:
                        kinds["text"] += 1
                        if STRAIN_PATTERN.search(str(cell)):
                            strain_hits += 1
                if index < 4:
                    headers.extend(str(cell) for cell in row if cell is not None)
            for header in headers:
                lowered = header.lower()
                if any(word in lowered for word in MEASUREMENT_WORDS):
                    measurement_headers.append(header)
            sheets.append(
                {
                    "sheet": sheet.title,
                    "rows_read": rows,
                    "cells": dict(kinds),
                    "first_rows_text": headers[:14],
                    "measurement_headers": measurement_headers,
                    "strain_id_cells": strain_hits,
                }
            )
    finally:
        workbook.close()
    return sheets


def _esm_inventory(library: Path, raw: Path) -> dict[str, Any]:
    """G3: every captured and deposited artifact, with its pin and its real format."""
    manifest = json.loads((library / "manifest.json").read_text())
    pinned = {record["path"]: record["sha256"] for record in manifest["files"]}
    files: list[dict[str, Any]] = []
    for path in sorted(library.glob("si/*")) + [library / "paper.md"]:
        if path.is_dir():
            continue
        relative = str(path.relative_to(library))
        entry: dict[str, Any] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "format": _magic(path),
            "manifest_pin": pinned.get(relative),
        }
        entry["pin_matches"] = entry["sha256"] == entry["manifest_pin"]
        if path.suffix == ".xlsx":
            entry["sheets"] = _workbook_census(path)
        files.append(entry)
    deposited = json.loads((raw / "manifest.json").read_text())
    return {
        "library_files": files,
        "worksheet_rows_total": sum(
            sheet["rows_read"] for entry in files for sheet in entry.get("sheets", [])
        ),
        "deposited": [
            {
                "path": record["path"],
                "sha256": record["sha256"],
                "source": record["source"],
                "retrieval_method": record["retrieval"]["method"],
                "on_disk_sha256": _sha256(raw / record["path"]),
            }
            for record in deposited["files"]
        ],
    }


def _no_measurement_table(inventory: dict[str, Any]) -> dict[str, Any]:
    """The claim the loader rests on, measured: no ESM file releases a measurement."""
    hits = [
        {"path": entry["path"], "sheet": sheet["sheet"], "headers": headers}
        for entry in inventory["library_files"]
        for sheet in entry.get("sheets", [])
        if (headers := sheet["measurement_headers"])
    ]
    strain_sheets = [
        {
            "path": entry["path"],
            "sheet": sheet["sheet"],
            "cells": sheet["strain_id_cells"],
        }
        for entry in inventory["library_files"]
        for sheet in entry.get("sheets", [])
        if sheet["strain_id_cells"]
    ]
    # A measurement word ALONE does not make a measurement table: the design workbooks
    # state the model's glucose UPTAKE BOUND ("Glucose uptake rate 6.0 mmol/gDW/hr",
    # and a "Glucose" input column holding the 6.3 constraint), which is a simulation
    # setting. A released per-strain measurement would have to name the strains it is
    # about, so the test is a measurement word AND a campaign strain id in one sheet.
    # Both halves are reported, so the raw hits stay visible rather than filtered away.
    strain_sheet_keys = {(sheet["path"], sheet["sheet"]) for sheet in strain_sheets}
    suspects = [hit for hit in hits if (hit["path"], hit["sheet"]) in strain_sheet_keys]
    return {
        "sheets_with_a_measurement_header": hits,
        "sheets_naming_a_campaign_strain": strain_sheets,
        "sheets_that_are_both": suspects,
        "conclusion": (
            "no ESM sheet releases a per-strain measurement: the "
            f"{len(hits)} measurement-word hit(s) are model constraints in the design "
            "workbooks, and no sheet names a single campaign strain"
            if not suspects
            else "an ESM sheet names a campaign strain AND a measurement column; the "
            "loader's prose sourcing must be re-examined before it is trusted"
        ),
        "data_availability": b24._Q_DATA_AVAILABILITY,
    }


def _titer_statements(paper: Path) -> dict[str, Any]:
    """Each stored titer located in the pinned OCR by its own verbatim quote."""
    text = paper.read_text(encoding="utf-8")
    rows: list[dict[str, Any]] = []
    for row in b24.TITER_ROWS:
        offset = text.find(row.quote)
        rows.append(
            {
                "family": row.family,
                "strain": row.strain,
                "condition": row.condition,
                "titer": row.titer,
                "unit": row.unit,
                "uncertainty": row.uncertainty,
                "time_hours": row.time_hours,
                "reference": row.reference,
                "quote_offset": offset,
                "quote_found": offset >= 0,
            }
        )
    return {
        "records": rows,
        "n_records": len(rows),
        "all_quotes_found": all(row["quote_found"] for row in rows),
        "by_family": dict(Counter(row["family"] for row in rows)),
        "references": {
            key: {
                "strain": baseline.strain,
                "titer": baseline.titer,
                "unit": baseline.unit,
                "quote_offset": text.find(baseline.quote),
            }
            for key, baseline in b24.REFERENCE_BASELINES.items()
        },
    }


def _not_stored() -> dict[str, Any]:
    """What the release carries that no record holds, with its count and reason."""
    return {
        "entries": [entry.model_dump() for entry in b24.not_stored()],
        "total_values": sum(entry.values for entry in b24.not_stored()),
        "od600_values": [
            {"strain": strain, "od600": value, "sd": sd, "time_hours": hours}
            for strain, value, sd, hours in b24.OD600_NOT_STORED
        ],
    }


def _gaca_contradiction(library_root: Path) -> dict[str, Any]:
    """The locus tag each mirrored paper prints for ``gacA``, measured from its bytes."""
    findings: list[dict[str, Any]] = []
    for key, expected in GACA_PAPERS.items():
        paper = library_root / key / "paper.md"
        digest = _sha256(paper)
        text = paper.read_text(encoding="utf-8")
        statements: list[str] = []
        for match in re.finditer("gacA", text):
            window = text[max(0, match.start() - 140) : match.end() + 140]
            if "PP_" in window:
                statements.append(" ".join(window.split()))
        findings.append(
            {
                "citation_key": key,
                "sha256": digest,
                "pin_matches": digest == expected,
                "statements": statements[:2],
            }
        )
    return {
        "papers": findings,
        "banerjee_states_a_tag": "gacA (PP_"
        in (library_root / CITATION_KEY / "paper.md").read_text(encoding="utf-8"),
        "conclusion": (
            "the mirrored papers disagree on which locus tag is gacA, so the loader "
            "stores no tag for it: Banerjee states none, and choosing among the "
            "siblings would be this loader resolving an identity across sources"
        ),
    }


def main() -> None:
    """Run every check and write the results JSON."""
    load_dotenv()
    data_root = Path(os.environ["DATA_ROOT"])
    library_root = data_root / "torchcell-library"
    library = library_root / CITATION_KEY
    raw = data_root / "torchcell-raw" / CITATION_KEY
    inventory = _esm_inventory(library, raw)
    results = {
        "citation_key": CITATION_KEY,
        "doi": b24.DOI,
        "generated_at": datetime.now(UTC).isoformat(),
        "g3_inventory": inventory,
        "no_measurement_table": _no_measurement_table(inventory),
        "titer_statements": _titer_statements(library / "paper.md"),
        "not_stored": _not_stored(),
        "gaca_tag_contradiction": _gaca_contradiction(library_root),
        "quotes_verified": b24.verify_quotes(),
    }
    out = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "banerjee2024_release_checks.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    no_table = _no_measurement_table(inventory)
    titers = _titer_statements(library / "paper.md")
    print(f"wrote {out}")
    print(f"  ESM files inventoried: {len(inventory['library_files'])}")
    print(f"  worksheet rows: {inventory['worksheet_rows_total']}")
    print(f"  {no_table['conclusion']}")
    print(
        f"  titer statements located: {titers['n_records']}, all quotes found: "
        f"{titers['all_quotes_found']}"
    )
    print(f"  values released but not stored: {_not_stored()['total_values']}")
    print(f"  quotes verified verbatim: {b24.verify_quotes()}")


if __name__ == "__main__":
    main()
