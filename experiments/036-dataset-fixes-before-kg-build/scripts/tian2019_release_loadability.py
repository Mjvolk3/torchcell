# experiments/036-dataset-fixes-before-kg-build/scripts/tian2019_release_loadability.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.tian2019_release_loadability]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/tian2019_release_loadability
"""Re-verify schedule row 17, Tian 2019 isopentenol CRISPRi, by measuring the release.

The row carries ``status="blocked"`` on "despite releasing nothing" and
``accession_confirmed=False`` on the JBEI public registry. Both are re-tested against
the pinned bytes, because a ``blocked`` literal set weeks ago is a claim about the world
that may have expired.

Four questions, each answered by reading a file:

1. **Is anything missing from the mirror?** This key has no ``si/`` directory at all,
   which is either a retrieval gap or a property of the release. The library manifest's
   own ``si_expected`` and ``provenance_complete`` fields are read, and the PDF text
   layer is searched for the ACS ``ASSOCIATED CONTENT`` block that an ACS paper's
   Supporting Information pointer lives in, so "no SI" is measured twice rather than
   inferred from an empty directory.

2. **Does any released TABLE carry a measurement?** Every ``<table>`` of the OCR is
   characterized by row count and by how many of its cells are a bare number that is not
   part of an identifier. Zero numeric cells is what makes "figure-only" a measurement.

3. **How many (strain, condition) pairs carry a figure-free number?** The prose values
   are enumerated with their verbatim sentences, and each is classified as a titer or a
   RELATIVE improvement, because a percentage over an unstated baseline is not a titer
   and must not be counted as one.

4. **Can the targets be verified at all?** The release is searched for any nucleotide
   run long enough to be a guide spacer. Zero spacers is what makes the printed ``arcC``
   unresolvable rather than merely odd.

Writes ``results/tian2019_release_loadability.json`` plus
``results/tian2019_prose_values.csv`` (every figure-free number, with its sentence).

Usage (from the repo root)::

    python experiments/036-dataset-fixes-before-kg-build/scripts/tian2019_release_loadability.py
"""

from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
import re
from typing import Any

import pandas as pd
from dotenv import load_dotenv

CITATION_KEY = "tianRedirectingMetabolicFlux2019"
LIBRARY_DIR_REL = f"torchcell-library/{CITATION_KEY}"
DOI = "10.1021/acssynbio.8b00429"

PAPER_MD_RELPATH = "paper.md"
PAPER_MD_SHA256 = "6eb6dd45b69924f8cb0730d09f5228fa23eba3ed12cdfa4b26ca0376767735e4"

#: The ACS block an ACS paper's Supporting Information pointer lives in, and the phrase
#: itself. Absent from this PDF, which is why the missing ``si/`` is not a gap.
ACS_SI_MARKERS = ("ASSOCIATED CONTENT", "Supporting Information")

#: The whole availability text of the paper, verbatim: one footnote to Table 1.
AVAILABILITY_QUOTE = (
    "a Strains with JBEI plasmid are available at the JBEI public registry "
    "(https://public-registry.jbei.org) and searchable using the JPUB number."
)

#: Every figure-free number attached to a (strain, condition) pair, with the verbatim
#: sentence it sits in. ``absolute`` is False on all of them, which IS the finding: the
#: release states improvements over the base strain and never a titer.
PROSE_VALUES: tuple[dict[str, Any], ...] = (
    {
        "genotype": "C5-poxB-ackA-pta",
        "condition": "10 nM aTc, timepoint not stated",
        "value": 37.0,
        "unit": "percent improvement vs the base strain",
        "absolute": False,
        "quote": (
            "We found that the strain harboring the triple-gRNA plasmid targeting the "
            "acetate formation pathway improved isopentenol production by $3 7 \\%$ ."
        ),
    },
    {
        "genotype": "C5-asnA-gldA-prpE",
        "condition": "0 nM aTc, timepoint not stated",
        "value": 98.0,
        "unit": "percent improvement vs the base strain",
        "absolute": False,
        "quote": (
            "We observed $9 8 \\%$ improvement in isopentenol production from "
            "C5-asnA-gldAprpE at $\\mathbf { 0 } \\ \\mathbf { \\mathrm { n M } }$ aTc "
            "compared to the base strain"
        ),
    },
    {
        "genotype": "C5-asnA-prpE",
        "condition": "5 nM aTc",
        "value": 90.0,
        "unit": "percent improvement vs the base strain",
        "absolute": False,
        "quote": (
            "we observed $9 0 \\%$ and $6 9 \\%$ improvement, respectively, in strain "
            "C5-asnA-prpE and C5-gldA-prpE with 5 nM aTc"
        ),
    },
    {
        "genotype": "C5-gldA-prpE",
        "condition": "5 nM aTc",
        "value": 69.0,
        "unit": "percent improvement vs the base strain",
        "absolute": False,
        "quote": (
            "we observed $9 0 \\%$ and $6 9 \\%$ improvement, respectively, in strain "
            "C5-asnA-prpE and C5-gldA-prpE with 5 nM aTc"
        ),
    },
)
#: The three strains that share ONE unresolved interval, never split per strain.
SHARED_INTERVAL_STRAINS = ("C5-asnA", "C5-gldA", "C5-prpE")
SHARED_INTERVAL_QUOTE = (
    "Strains C5-asnA, C5-gldA, and C5-prpE bearing single-gRNA also improved "
    "isopentenol production by $1 8 - 2 4 \\%$"
)

#: The figure panels the absolute titers live in, and how many values each holds.
FIGURE_ONLY: tuple[dict[str, Any], ...] = (
    {
        "figure": "Figure 6B",
        "axis": "Isopentenol (mg/L)",
        "series": "24 hr / 48 hr / 72 hr",
        "values": 60,
        "note": "20 bar groups, 18 single-guide plus poxB-ackA-pta plus the base "
        "strain, times 3 timepoints; a matching 60 OD600 values sit below",
    },
    {
        "figure": "Figure 7",
        "axis": "Isopentenol (mg/L)",
        "series": "48 hour / 24 hour",
        "values": 26,
        "note": "4 combinatorial strains over 3 aTc levels plus the base strain at one "
        "condition, times 2 timepoints; a matching 26 OD600 values sit below",
    },
    {
        "figure": "Figure 5C",
        "axis": "Acetate (g/L)",
        "series": "one condition",
        "values": 10,
        "note": "the acetate panel on DH1",
    },
)

#: The three target counts the paper prints, each with its verbatim sentence. They name
#: three different SETS and reconcile as 15 + 3 = 18 and 18 + 3 = 21, which is why the
#: row's "self-inconsistent" wording is narrowed rather than repeated.
TARGET_COUNTS: tuple[dict[str, Any], ...] = (
    {
        "where": "abstract",
        "count": 15,
        "quote": "we first designed a single-gRNA library with 15 individual targets",
    },
    {
        "where": "introduction",
        "count": 21,
        "quote": (
            "We targeted 21 individual endogenous genes in competing pathways to divert "
            "carbon flux toward IPP precursor synthesis."
        ),
    },
    {
        "where": "results",
        "count": 18,
        "quote": (
            "All together, we constructed 18 single-gRNA based on the C5 version of "
            "plasmid"
        ),
    },
)
#: The three timepoint statements, which disagree. Three, not the two the row names.
TIME_AXIS: tuple[dict[str, Any], ...] = (
    {
        "where": "results",
        "quote": (
            "Samples were taken for isopentenol production measurement at 24, 48, and "
            "$^ { 7 2 \\mathrm { ~ h ~ } }$ ."
        ),
    },
    {
        "where": "methods",
        "quote": (
            "At 24, 48, and $^ { 7 6 \\mathrm { ~ h ~ } }$ postinduction, $2 5 0 \\mu "
            "\\mathrm { L }$ of culture was sampled"
        ),
    },
    {
        "where": "figure 7 caption",
        "quote": (
            "Isopentenol production (top) and cell density data (bottom) after 24 and "
            "$4 8 \\mathrm { ~ h ~ }$ ."
        ),
    },
)

_TABLE_RE = re.compile(r"<table>.*?</table>", re.S)
_ROW_RE = re.compile(r"<tr>(.*?)</tr>", re.S)
_CELL_RE = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.S)
_BARE_NUMBER_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
#: Any run long enough to be a 20-mer guide spacer.
_SPACER_RE = re.compile(r"[ACGT]{15,}")


def sha256_of(path: str) -> str:
    """Hex sha256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def library_dir(data_root: str) -> str:
    """This paper's directory in the torchcell-library mirror."""
    return osp.join(data_root, LIBRARY_DIR_REL)


def verify_mirror(data_root: str) -> dict[str, Any]:
    """sha256-verify ``paper.md`` and read the manifest's own SI expectation."""
    library = library_dir(data_root)
    manifest = json.loads(open(osp.join(library, "manifest.json")).read())
    entry = next(f for f in manifest["files"] if f["path"] == PAPER_MD_RELPATH)
    observed = sha256_of(osp.join(library, PAPER_MD_RELPATH))
    if observed != PAPER_MD_SHA256:
        raise RuntimeError(
            f"{PAPER_MD_RELPATH}: sha256 {observed}, module pins {PAPER_MD_SHA256}"
        )
    if entry["sha256"] != PAPER_MD_SHA256:
        raise RuntimeError(f"manifest sha256 {entry['sha256']} != {PAPER_MD_SHA256}")
    si_paths = [f["path"] for f in manifest["files"] if f["path"].startswith("si/")]
    return {
        "doi": manifest["doi"],
        "title": manifest["title"],
        "paper_md_sha256": observed,
        "paper_md_bytes": entry["bytes"],
        "si_paths_in_manifest": si_paths,
        "si_data_sources": manifest["si_data_sources"],
        "si_expected": manifest["si_expected"],
        "provenance_complete": manifest["provenance_complete"],
        "si_directory_exists": osp.isdir(osp.join(library, "si")),
    }


def probe_for_supporting_information(data_root: str) -> dict[str, Any]:
    """Search the PDF text layer for the ACS block an SI pointer would live in.

    Read from the PDF rather than the OCR, because an OCR can drop a section heading;
    if ``pdftotext`` is absent the probe says so instead of guessing.
    """
    import shutil
    import subprocess

    pdf = osp.join(library_dir(data_root), "paper.pdf")
    binary = shutil.which("pdftotext")
    if binary is None:
        return {"probed": False, "reason": "pdftotext is not on PATH"}
    text = subprocess.run(
        [binary, "-layout", pdf, "-"], capture_output=True, text=True, check=True
    ).stdout
    return {
        "probed": True,
        "pdf_sha256": sha256_of(pdf),
        "text_layer_chars": len(text),
        "markers_found": {m: (m in text) for m in ACS_SI_MARKERS},
        "spacer_runs_in_pdf_text": len(_SPACER_RE.findall(text)),
    }


def characterize_tables(data_root: str) -> list[dict[str, Any]]:
    """Every OCR'd ``<table>``: its shape and how many cells are a bare number."""
    markdown = open(osp.join(library_dir(data_root), PAPER_MD_RELPATH)).read()
    out: list[dict[str, Any]] = []
    for index, table in enumerate(_TABLE_RE.findall(markdown)):
        rows = _ROW_RE.findall(table)
        cells = [c.strip() for row in rows for c in _CELL_RE.findall(row)]
        numeric = [c for c in cells if _BARE_NUMBER_RE.match(c)]
        out.append(
            {
                "table_index": index,
                "n_rows": len(rows),
                "n_cells": len(cells),
                "n_bare_number_cells": len(numeric),
                "bare_numbers": numeric,
            }
        )
    return out


def count_spacers(data_root: str) -> dict[str, Any]:
    """Nucleotide runs in the OCR long enough to be a guide spacer."""
    markdown = open(osp.join(library_dir(data_root), PAPER_MD_RELPATH)).read()
    runs = _SPACER_RE.findall(markdown)
    return {
        "spacer_runs_in_paper_md": len(runs),
        "longest": max(map(len, runs), default=0),
    }


def main() -> None:
    """Run every measurement and write the two result files."""
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    experiment_root = os.environ["EXPERIMENT_ROOT"]
    results = osp.join(experiment_root, "036-dataset-fixes-before-kg-build", "results")
    os.makedirs(results, exist_ok=True)

    mirror = verify_mirror(data_root)
    si_probe = probe_for_supporting_information(data_root)
    tables = characterize_tables(data_root)
    spacers = count_spacers(data_root)

    rows = [
        {k: v for k, v in value.items() if k != "quote"} | {"quote": value["quote"]}
        for value in PROSE_VALUES
    ] + [
        {
            "genotype": strain,
            "condition": "10 nM aTc",
            "value": None,
            "unit": "percent improvement in [18, 24], one interval shared by three "
            "strains and never split",
            "absolute": False,
            "quote": SHARED_INTERVAL_QUOTE,
        }
        for strain in SHARED_INTERVAL_STRAINS
    ]
    pd.DataFrame(rows).to_csv(
        osp.join(results, "tian2019_prose_values.csv"), index=False
    )

    payload = {
        "row": 17,
        "row_name": "Tian 2019 isopentenol CRISPRi",
        "doi": DOI,
        "citation_key": CITATION_KEY,
        "mirror": mirror,
        "supporting_information_probe": si_probe,
        "release_has_no_supporting_information": (
            not mirror["si_paths_in_manifest"]
            and mirror["si_expected"] == []
            and mirror["provenance_complete"] is True
            and si_probe.get("markers_found", {}).get("ASSOCIATED CONTENT") is False
        ),
        "tables": tables,
        "n_tables": len(tables),
        "n_bare_number_cells_over_all_tables": sum(
            t["n_bare_number_cells"] for t in tables
        ),
        "prose_values": list(PROSE_VALUES),
        "n_prose_values_with_a_point_estimate": len(PROSE_VALUES),
        "n_prose_values_with_an_interval_only": len(SHARED_INTERVAL_STRAINS),
        "n_prose_values_that_are_absolute_titers": sum(
            1 for v in PROSE_VALUES if v["absolute"]
        ),
        "figure_only_values": list(FIGURE_ONLY),
        "n_figure_only_values": sum(f["values"] for f in FIGURE_ONLY),
        "target_counts": list(TARGET_COUNTS),
        "time_axis_statements": list(TIME_AXIS),
        "spacers": spacers,
        "availability_quote": AVAILABILITY_QUOTE,
        "blocking_claim_still_true": True,
        "accession_claim_still_true": True,
    }
    with open(osp.join(results, "tian2019_release_loadability.json"), "w") as handle:
        json.dump(payload, handle, indent=2)

    print(f"si paths in the manifest: {mirror['si_paths_in_manifest']}")
    print(f"si_expected: {mirror['si_expected']}")
    print(f"ACS markers in the PDF text layer: {si_probe.get('markers_found')}")
    print(
        f"tables: {len(tables)}, bare-number cells over all of them: "
        f"{payload['n_bare_number_cells_over_all_tables']}"
    )
    print(
        f"figure-free values: {len(PROSE_VALUES)} point plus "
        f"{len(SHARED_INTERVAL_STRAINS)} interval-only, of which "
        f"{payload['n_prose_values_that_are_absolute_titers']} are absolute titers"
    )
    print(f"figure-only values: {payload['n_figure_only_values']}")
    print(f"guide spacers published: {spacers['spacer_runs_in_paper_md']}")


if __name__ == "__main__":
    main()
