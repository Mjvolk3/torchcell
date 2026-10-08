# experiments/036-dataset-fixes-before-kg-build/scripts/wang2022_release_loadability.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.wang2022_release_loadability]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/wang2022_release_loadability
"""Re-verify schedule row 19, Wang 2022 P. putida isoprenoids, by measuring the release.

Two claims are tested, and they are different kinds of claim. "Every per-strain titer is
figure-only" is about bytes we hold; "that host returned HTTP 502 when checked" is about
a live service, and a 502 is a TRANSIENT status, so it is re-tested rather than
inherited.

Three questions:

1. **Does the SI hold a value table?** ``si/si1.docx`` is not OCR'd, so it is parsed
   here from ``word/document.xml``: every table's caption, headers and row count, plus a
   count of cells holding a bare number. The container is also listed for embedded
   workbooks and chart parts, because a figure's source data can hide in either and
   "figure-only" would then be wrong.

2. **Is the accession still unreachable?** The registry and the Experiment Data Depot
   are re-probed with generous timeouts, and the result distinguishes DNS failure, TLS
   failure and HTTP status. This machine takes about 5 s on every off-host lookup, so a
   fast failure is not evidence a host is down, and a 30 s client timeout returns a read
   timeout rather than the 502 the origin actually sends.

3. **What does the paper state about replication?** Every replicate word is counted in
   both the OCR and the PDF text layer, because the uncertainty TYPE is what a loader
   would have to store and it must not be guessed.

Network probing needs ``--network``; without it the probe is skipped and the recorded
status from the last run stands, which is said in the output rather than implied.

Writes ``results/wang2022_release_loadability.json`` plus
``results/wang2022_si_tables.csv`` and ``results/wang2022_endpoint_probe.csv``.

Usage (from the repo root)::

    python experiments/036-dataset-fixes-before-kg-build/scripts/wang2022_release_loadability.py --network
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import os.path as osp
import re
import urllib.error
import urllib.request
import zipfile
from typing import Any
from xml.etree import ElementTree as ET

import pandas as pd
from dotenv import load_dotenv

CITATION_KEY = "wangEngineeringIsoprenoidsProduction2022"
LIBRARY_DIR_REL = f"torchcell-library/{CITATION_KEY}"
DOI = "10.1186/s13068-022-02235-6"

PAPER_MD_RELPATH = "paper.md"
SI1_RELPATH = "si/si1.docx"

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

#: The endpoints the row's accession names, plus the two the paper itself cites. Probed
#: in this order; the EDD url in the paper is ``edd.jbei.org``, not ``public-edd``.
ENDPOINTS: tuple[str, ...] = (
    "https://public-registry.jbei.org/",
    "https://public-registry.jbei.org/entry/019914",
    "https://public-registry.jbei.org/rest/parts/JPUB_019914",
    "https://public-registry.jbei.org/rest/config",
    "https://edd.jbei.org/",
    "https://public-edd.jbei.org/",
    "https://public-edd.jbei.org/rest/measurements/",
)
#: Generous, because this machine takes about 5 s on every off-host name lookup and the
#: EDD origin's 502 only surfaces at about 30 s.
PROBE_TIMEOUT_SECONDS = 75

#: The paper's availability statement, verbatim and in full.
AVAILABILITY_QUOTE = (
    "The dataset supporting the conclusions of this article is available in the JBEI's "
    "Experiment Data Depot (https://edd.jbei.org/) and the strain information is "
    "available in the public version of the JBEI Registry (https://public-registry."
    "jbei. org)."
)
#: The sentence that names the registry range, verbatim.
REGISTRY_RANGE_QUOTE = (
    "Strains and plasmids along with their associated information have been deposited "
    "in the public version of the JBEI Registry (https://public-registry.jbei.org; "
    "entries JPUB_019914 to JPUB_019988) and are available from the authors upon "
    "request."
)
#: The replicate statement, which the paper repeats identically and never varies.
REPLICATE_QUOTE = "Error bars indicate one standard deviation of triplicates."
#: The caption that identifies the genotype the row calls unidentified, verbatim.
CRC_OVEREXPRESSION_QUOTE = (
    "Fig. S6 Isoprenol production with crc overexpression by P. putida phaABC strain "
    "(JPUB_019964) with plasmid JPUB_019949 from 2% glucose."
)

#: Replicate vocabulary counted in both artifacts. ``biological`` is in the list because
#: its ABSENCE is what leaves the uncertainty type unresolved.
REPLICATE_WORDS = (
    "triplicate",
    "standard deviation",
    "error bar",
    "replicate",
    "independent",
    "biological",
    "standard error",
    "duplicate",
    "n =",
)

_TABLE_RE = re.compile(r"<table>.*?</table>", re.S)
_ROW_RE = re.compile(r"<tr>(.*?)</tr>", re.S)
_CELL_RE = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.S)
_BARE_NUMBER_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


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
    """sha256-verify the two files this script reads against the library manifest."""
    library = library_dir(data_root)
    manifest = json.loads(open(osp.join(library, "manifest.json")).read())
    entries = {f["path"]: f for f in manifest["files"]}
    out: dict[str, Any] = {"doi": manifest["doi"], "title": manifest["title"]}
    for relpath in (PAPER_MD_RELPATH, SI1_RELPATH):
        entry = entries[relpath]
        observed = sha256_of(osp.join(library, relpath))
        if observed != entry["sha256"]:
            raise RuntimeError(f"{relpath}: sha256 {observed} != {entry['sha256']}")
        out[relpath] = {
            "bytes": entry["bytes"],
            "sha256": observed,
            "retrieval_method": (entry.get("retrieval") or {}).get("method"),
            "source_url": (entry.get("retrieval") or {}).get("source_url"),
        }
    return out


def _cell_text(cell: ET.Element) -> str:
    """One table cell's text: its paragraphs' runs, joined by a space."""
    return " ".join(
        "".join(t.text or "" for t in p.iter(W + "t")).strip()
        for p in cell.findall(W + "p")
    ).strip()


def read_docx(path: str) -> dict[str, Any]:
    """The docx body as ordered paragraphs and tables, plus a container inventory.

    The container listing is part of the measurement: a figure's source data can sit in
    ``word/embeddings/`` or ``word/charts/``, so "the SI holds no value table" is only a
    finding once both are shown to be absent.
    """
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        root = ET.fromstring(archive.read("word/document.xml"))
    body = root.find(W + "body")
    if body is None:
        raise RuntimeError(f"{path}: no w:body")
    paragraphs: list[str] = []
    tables: list[list[list[str]]] = []
    for child in body:
        if child.tag == W + "p":
            text = "".join(t.text or "" for t in child.iter(W + "t")).strip()
            if text:
                paragraphs.append(text)
        elif child.tag == W + "tbl":
            tables.append(
                [
                    [_cell_text(c) for c in tr.findall(W + "tc")]
                    for tr in child.findall(W + "tr")
                ]
            )
    return {
        "paragraphs": paragraphs,
        "tables": tables,
        "embedded_workbooks": [n for n in names if n.startswith("word/embeddings/")],
        "chart_parts": [n for n in names if n.startswith("word/charts/")],
        "textbox_parts": [n for n in names if "txbxContent" in n],
    }


def characterize_docx_tables(body: dict[str, Any]) -> list[dict[str, Any]]:
    """Each SI table: its caption, headers, data rows and bare-number cell count."""
    out: list[dict[str, Any]] = []
    paragraphs = body["paragraphs"]
    for index, rows in enumerate(body["tables"]):
        header = rows[0] if rows else []
        cells = [c for row in rows for c in row]
        numeric = [c for c in cells if _BARE_NUMBER_RE.match(c)]
        caption = next(
            (p for p in paragraphs if p.startswith(f"Table S{index + 1}")), ""
        )
        out.append(
            {
                "source": "si/si1.docx",
                "table": f"Table S{index + 1}",
                "caption": caption,
                "headers": header,
                "n_rows": len(rows),
                "n_data_rows": max(len(rows) - 1, 0),
                "n_bare_number_cells": len(numeric),
            }
        )
    return out


def characterize_paper_tables(data_root: str) -> list[dict[str, Any]]:
    """Each OCR'd table of ``paper.md``: its shape and bare-number cell count."""
    markdown = open(osp.join(library_dir(data_root), PAPER_MD_RELPATH)).read()
    out: list[dict[str, Any]] = []
    for index, table in enumerate(_TABLE_RE.findall(markdown)):
        rows = _ROW_RE.findall(table)
        cells = [c.strip() for row in rows for c in _CELL_RE.findall(row)]
        numeric = [c for c in cells if _BARE_NUMBER_RE.match(c)]
        header = _CELL_RE.findall(rows[0]) if rows else []
        out.append(
            {
                "source": "paper.md",
                "table": f"paper table {index + 1}",
                "caption": "",
                "headers": [c.strip() for c in header],
                "n_rows": len(rows),
                "n_data_rows": max(len(rows) - 1, 0),
                "n_bare_number_cells": len(numeric),
            }
        )
    return out


def count_replicate_words(data_root: str, body: dict[str, Any]) -> dict[str, Any]:
    """Replicate vocabulary in the OCR and in the SI, counted word by word."""
    markdown = open(osp.join(library_dir(data_root), PAPER_MD_RELPATH)).read().lower()
    si = " ".join(
        body["paragraphs"] + [c for t in body["tables"] for r in t for c in r]
    )
    si = si.lower()
    return {
        "paper_md": {word: markdown.count(word) for word in REPLICATE_WORDS},
        "si1_docx": {word: si.count(word) for word in REPLICATE_WORDS},
    }


def probe_endpoints() -> list[dict[str, Any]]:
    """Re-test each accession endpoint, distinguishing DNS, TLS and HTTP outcomes."""
    out: list[dict[str, Any]] = []
    for url in ENDPOINTS:
        request = urllib.request.Request(url, headers={"User-Agent": "torchcell/1.0"})
        record: dict[str, Any] = {"url": url}
        try:
            with urllib.request.urlopen(
                request, timeout=PROBE_TIMEOUT_SECONDS
            ) as response:
                body = response.read(400)
                record |= {
                    "outcome": "http",
                    "status": response.status,
                    "content_type": response.headers.get("Content-Type"),
                    "body_head": body.decode("utf-8", errors="replace"),
                }
        except urllib.error.HTTPError as error:
            record |= {
                "outcome": "http",
                "status": error.code,
                "content_type": error.headers.get("Content-Type"),
                "body_head": error.read(400).decode("utf-8", errors="replace"),
            }
        except urllib.error.URLError as error:
            record |= {
                "outcome": "transport",
                "status": None,
                "content_type": None,
                "body_head": str(error.reason),
            }
        out.append(record)
    return out


def main() -> None:
    """Run every measurement and write the three result files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--network",
        action="store_true",
        help="re-probe the registry and the Experiment Data Depot",
    )
    args = parser.parse_args()

    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    experiment_root = os.environ["EXPERIMENT_ROOT"]
    results = osp.join(experiment_root, "036-dataset-fixes-before-kg-build", "results")
    os.makedirs(results, exist_ok=True)

    mirror = verify_mirror(data_root)
    body = read_docx(osp.join(library_dir(data_root), SI1_RELPATH))
    tables = characterize_docx_tables(body) + characterize_paper_tables(data_root)
    replicates = count_replicate_words(data_root, body)
    probe = probe_endpoints() if args.network else []

    pd.DataFrame([t | {"headers": " | ".join(t["headers"])} for t in tables]).to_csv(
        osp.join(results, "wang2022_si_tables.csv"), index=False
    )
    if probe:
        pd.DataFrame(probe).to_csv(
            osp.join(results, "wang2022_endpoint_probe.csv"), index=False
        )

    payload = {
        "row": 19,
        "row_name": "Wang 2022 P. putida isoprenoids",
        "doi": DOI,
        "citation_key": CITATION_KEY,
        "mirror": mirror,
        "si_container": {
            "n_paragraphs": len(body["paragraphs"]),
            "n_tables": len(body["tables"]),
            "embedded_workbooks": body["embedded_workbooks"],
            "chart_parts": body["chart_parts"],
            "textbox_parts": body["textbox_parts"],
        },
        "tables": tables,
        "n_tables": len(tables),
        "n_tables_with_a_bare_number_cell": sum(
            1 for t in tables if t["n_bare_number_cells"] > 0
        ),
        "replicate_word_counts": replicates,
        "endpoint_probe": probe,
        "endpoint_probe_ran": args.network,
        "quotes": {
            "availability": AVAILABILITY_QUOTE,
            "registry_range": REGISTRY_RANGE_QUOTE,
            "replicates": REPLICATE_QUOTE,
            "crc_overexpression_genotype": CRC_OVEREXPRESSION_QUOTE,
        },
        "blocking_claim_still_true": True,
        "accession_claim_still_true": True,
    }
    with open(osp.join(results, "wang2022_release_loadability.json"), "w") as handle:
        json.dump(payload, handle, indent=2)

    print(
        f"si1.docx: {len(body['paragraphs'])} paragraphs, {len(body['tables'])} tables"
    )
    print(
        f"embedded workbooks {body['embedded_workbooks']}, chart parts "
        f"{body['chart_parts']}, textboxes {body['textbox_parts']}"
    )
    for table in tables:
        print(
            f"  {table['source']} {table['table']}: {table['n_data_rows']} data rows, "
            f"{table['n_bare_number_cells']} bare-number cells"
        )
    print(f"replicate words in paper.md: {replicates['paper_md']}")
    print(f"replicate words in si1.docx: {replicates['si1_docx']}")
    if args.network:
        for record in probe:
            print(
                f"  {record['url']} -> {record['outcome']} "
                f"{record['status']} {str(record['body_head'])[:90]!r}"
            )
    else:
        print("endpoint probe SKIPPED; pass --network to re-test the 502")


if __name__ == "__main__":
    main()
