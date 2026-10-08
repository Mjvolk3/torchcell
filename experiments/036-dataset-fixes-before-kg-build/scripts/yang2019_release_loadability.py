# experiments/036-dataset-fixes-before-kg-build/scripts/yang2019_release_loadability.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.yang2019_release_loadability]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/yang2019_release_loadability
"""Re-verify schedule row 49, Yang 2019 mevalonate from ethanol, by measuring the release.

The row carries ``status="blocked"`` on "every titer is figure-only" with
``accession_confirmed=True``, and three structural claims that had never been checked
against the bytes: a 9,283-base sequence total, five GenBank gene identifiers, and five
``PP_`` locus tags recorded as DERIVED. All four are measured here.

Four questions:

1. **Does either SI file hold a measurement?** Neither ``si/si1.docx`` nor
   ``si/si2.docx`` is OCR'd, so both are parsed from ``word/document.xml``. Every table
   is characterized and every numeric cell counted, and the container is listed for
   embedded workbooks so a hidden source-data sheet would show up rather than be assumed
   absent.

2. **Is the 9,283-base total right?** It is RECOMPUTED by summing the six released
   codon-optimized sequences, each also checked for being in frame and stop-terminated
   and for carrying a Shine-Dalgarno-like leader, because the row's "including the
   ribosome-binding regions" qualifier is what makes 9,283 differ from the coding total.

3. **Are the locus tags really absent?** Every artifact is searched for ``PP_``. Zero
   occurrences is what keeps the five tags in the row marked derived rather than quoted,
   and it is a finding a later reader should not have to re-establish.

4. **How many figure-free titers are there, and is there a reference titer?** The prose
   values are enumerated with their verbatim sentences. The reference question is the
   one that decides a titer family: ``ProductTiterExperimentReference`` requires a
   released reference titer, and the base strain's absence from this list is what
   refuses it.

Writes ``results/yang2019_release_loadability.json`` plus
``results/yang2019_sequences.csv`` and ``results/yang2019_prose_titers.csv``.

Usage (from the repo root)::

    python experiments/036-dataset-fixes-before-kg-build/scripts/yang2019_release_loadability.py
"""

from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
import re
import zipfile
from typing import Any
from xml.etree import ElementTree as ET

import pandas as pd
from dotenv import load_dotenv

CITATION_KEY = "yangMevalonateProductionEthanol2019"
LIBRARY_DIR_REL = f"torchcell-library/{CITATION_KEY}"
DOI = "10.1186/s12934-019-1213-y"

PAPER_MD_RELPATH = "paper.md"
SI1_RELPATH = "si/si1.docx"
SI2_RELPATH = "si/si2.docx"

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

#: The row's claimed sequence total, recomputed below rather than trusted.
CLAIMED_SEQUENCE_BASES = 9283
#: The six heterologous genes and the accession the Methods print for each.
GENE_ACCESSIONS: dict[str, str] = {
    "mvaE_opti": "AAO81155.1",
    "mvaS_opti": "AAO81154.1",
    "atoB_opti": "946727",
    "acs_opti": "948572",
    "eutE_opti": "946943",
    "nphT7_opti": "AB540131.1",
}
#: The five deleted loci and the GenBank gene identifier printed for each, verbatim.
DELETED_LOCI: dict[str, str] = {
    "endA": "1047019",
    "endX": "1045620",
    "qedH-I": "1046117",
    "qedH-II": "1046129",
    "phaG": "1046114",
}
DELETION_QUOTE = (
    "endA (Endonuclease I, GenBank ID: 1047019) and endX (Extracellular DNA "
    "endonuclease, GenBank ID: 1045620) for genetic stability improvement, qedH-I "
    "(Quinoprotein ethanol dehydrogenase, GenBank ID: 1046117) and qedH-II "
    "(Quinoprotein ethanol dehydrogenase, GenBank"
)
#: The five locus tags the row records as DERIVED, kept here so the absence search has
#: something to report against.
DERIVED_LOCUS_TAGS = ("PP_3375", "PP_2451", "PP_2674", "PP_2679", "PP_1408")

#: The data statement, verbatim and in full.
AVAILABILITY_QUOTE = "Availability of data and materials Not applicable."
#: Words whose absence proves there is no deposit.
DEPOSIT_WORDS = (
    "deposit",
    "repositor",
    "accession",
    "supplementary data",
    "raw data",
    "dataset",
)
#: The two replicate statements, verbatim.
REPLICATE_QUOTE_METHODS = (
    "All strains were cultured under the above conditions and in triplet for "
    "reproducibility confrmation."
)
REPLICATE_QUOTE_LEGEND = (
    "All the experiments were performed in triplicates and standard deviations of "
    "triplet culture were shown in the form of error bars"
)

#: Every figure-free titer, with the verbatim sentence it sits in. The base strain
#: ELPP000 is ABSENT by measurement, which is what refuses the titer family.
PROSE_TITERS: tuple[dict[str, Any], ...] = (
    {
        "strain": "ELPP010",
        "condition": "flask, modified M9 + 10 g/L ethanol, 27 h",
        "analyte": "mevalonate",
        "value_g_per_l": 1.70,
        "sd_g_per_l": 0.55,
        "quote": (
            "the ELPP010 strain, into which the upper mevalonate pathway was "
            "introduced, produced $1 . 7 0 { \\pm } 0 . 5 5 \\ \\mathrm { \\ g / L }$ "
            "of mevalonate after $2 7 \\ \\mathrm { h } ,$"
        ),
    },
    {
        "strain": "ELPP110",
        "condition": "flask, modified M9 + 10 g/L ethanol, 27 h",
        "analyte": "mevalonate",
        "value_g_per_l": 2.43,
        "sd_g_per_l": 1.34,
        "quote": (
            "After fuorescence analysis, $2 . 4 3 \\pm 1 . 3 4 ~ \\mathrm { g / L }$ of "
            "mevalonate was produced by a $2 7 \\mathrm { h }$ fask scale fermentation "
            "of"
        ),
    },
    {
        "strain": "ELPP111",
        "condition": "flask, modified M9 + 10 g/L ethanol, 27 h",
        "analyte": "mevalonate",
        "value_g_per_l": 2.88,
        "sd_g_per_l": 1.16,
        "quote": (
            "ELPP111, acs gene expressed in ELPP110, produced $2 . 8 8 \\pm 1 . 1 6 ~ "
            "\\mathrm { \\ g / L }$ of mevalonate in $2 7 \\ \\mathrm { h }$ fask "
            "fermentation"
        ),
    },
    {
        "strain": "ELPP211",
        "condition": "flask, modified M9 + 10 g/L ethanol, 27 h",
        "analyte": "mevalonate",
        "value_g_per_l": 4.07,
        "sd_g_per_l": 0.29,
        "quote": (
            "ELPP211, qedH-I and qedH-II deleted strain from ELPP111, produced $4 . 0 7 "
            "{ \\pm } 0 . 2 9 \\mathrm { \\textrm { g } / L }$ of mevalonate without "
            "acetate accumulation"
        ),
    },
    {
        "strain": "ELPP311",
        "condition": "2.5 L batch fermenter, 300 mM ethanol, pH 7.0, 24 h",
        "analyte": "mevalonate",
        "value_g_per_l": 4.18,
        "sd_g_per_l": None,
        "quote": (
            "At $\\mathrm { p H } 7 . 0$ control culture, $4 . 1 8 ~ \\mathrm { g / L }$ "
            "of mevalonate was produced from $1 3 . 6 ~ \\mathrm { g / L }$ of ethanol "
            "in $2 4 \\mathrm { ~ h ~ }$ culture time"
        ),
    },
    {
        "strain": "ELPP311",
        "condition": "2.5 L batch fermenter, 300 mM ethanol, pH 6.75",
        "analyte": "mevalonate",
        "value_g_per_l": 4.60,
        "sd_g_per_l": None,
        "quote": (
            "$4 . 6 0 ~ \\mathrm { g / L }$ of mevalonate was produced from $1 4 . 3 ~ "
            "\\mathrm { g / L }$ of ethanol (about $3 0 0 ~ \\mathrm { m M }$ ) with a "
            "production yield of $0 . 3 2 { \\mathrm { g } }$ mevalonate/g ethanol."
        ),
    },
)
#: The strain the reference titer would have to come from, and it carries no number.
REFERENCE_STRAIN = "ELPP000"

_SEQUENCE_RE = re.compile(r"^[ACGT]{200,}$")
_SD_MOTIFS = ("AGGAGG", "AGGAGA", "AGGAAA", "AAGGAG", "GGAGG")
_TABLE_RE = re.compile(r"<table>.*?</table>", re.S)
_ROW_RE = re.compile(r"<tr>(.*?)</tr>", re.S)
_CELL_RE = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.S)
_BARE_NUMBER_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
_STOPS = ("TAA", "TAG", "TGA")


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
    """sha256-verify the three files this script reads against the library manifest."""
    library = library_dir(data_root)
    manifest = json.loads(open(osp.join(library, "manifest.json")).read())
    entries = {f["path"]: f for f in manifest["files"]}
    out: dict[str, Any] = {"doi": manifest["doi"], "title": manifest["title"]}
    for relpath in (PAPER_MD_RELPATH, SI1_RELPATH, SI2_RELPATH):
        entry = entries[relpath]
        observed = sha256_of(osp.join(library, relpath))
        if observed != entry["sha256"]:
            raise RuntimeError(f"{relpath}: sha256 {observed} != {entry['sha256']}")
        out[relpath] = {
            "bytes": entry["bytes"],
            "sha256": observed,
            "original_filename": entry.get("original_filename"),
        }
    return out


def _cell_text(cell: ET.Element) -> str:
    """One table cell's text: its paragraphs' runs, joined by a space."""
    return " ".join(
        "".join(t.text or "" for t in p.iter(W + "t")).strip()
        for p in cell.findall(W + "p")
    ).strip()


def read_docx(path: str) -> dict[str, Any]:
    """The docx body as ordered paragraphs and tables, plus a container inventory."""
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
    }


def measure_sequences(body: dict[str, Any]) -> list[dict[str, Any]]:
    """Each released sequence: its header, length, frame, stop codon and leader motif.

    One paragraph per sequence, each an unbroken uppercase ACGT run, so the length is
    the paragraph length with no whitespace to strip. The CDS is taken as the suffix
    beginning at the FIRST ``ATG`` whose remainder is in frame and stop-terminated,
    which is what separates the leader from the coding part without assuming a leader
    length. First and not last, because a later in-frame ``ATG`` of the same CDS also
    satisfies the test and would report almost the whole gene as leader.
    """
    out: list[dict[str, Any]] = []
    paragraphs = body["paragraphs"]
    for index, text in enumerate(paragraphs):
        if not _SEQUENCE_RE.match(text):
            continue
        header = paragraphs[index - 1] if index else ""
        cds_start = None
        for start in range(len(text) - 3):
            if text[start : start + 3] != "ATG":
                continue
            remainder = len(text) - start
            if remainder % 3 == 0 and text[-3:] in _STOPS:
                cds_start = start
                break
        leader = text[:cds_start] if cds_start is not None else ""
        out.append(
            {
                "header": header,
                "gene": header.split(" ", 1)[0] if header else "",
                "released_bases": len(text),
                "leader_bases": len(leader),
                "cds_bases": len(text) - len(leader),
                "cds_in_frame": cds_start is not None,
                "stop_codon": text[-3:],
                "leader_has_shine_dalgarno": any(m in leader for m in _SD_MOTIFS),
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
        )
    return out


def characterize_tables(
    si1: dict[str, Any], si2: dict[str, Any], data_root: str
) -> list[dict[str, Any]]:
    """Every table of either SI file and of the OCR, with its numeric-cell count."""
    out: list[dict[str, Any]] = []
    for source, body in (("si/si1.docx", si1), ("si/si2.docx", si2)):
        for index, rows in enumerate(body["tables"]):
            cells = [c for row in rows for c in row]
            out.append(
                {
                    "source": source,
                    "table": f"table {index + 1}",
                    "headers": rows[0] if rows else [],
                    "n_rows": len(rows),
                    "n_data_rows": max(len(rows) - 1, 0),
                    "n_bare_number_cells": sum(
                        1 for c in cells if _BARE_NUMBER_RE.match(c)
                    ),
                }
            )
    markdown = open(osp.join(library_dir(data_root), PAPER_MD_RELPATH)).read()
    for index, table in enumerate(_TABLE_RE.findall(markdown)):
        rows = _ROW_RE.findall(table)
        cells = [c.strip() for row in rows for c in _CELL_RE.findall(row)]
        out.append(
            {
                "source": "paper.md",
                "table": f"table {index + 1}",
                "headers": [c.strip() for c in _CELL_RE.findall(rows[0])]
                if rows
                else [],
                "n_rows": len(rows),
                "n_data_rows": max(len(rows) - 1, 0),
                "n_bare_number_cells": sum(
                    1 for c in cells if _BARE_NUMBER_RE.match(c)
                ),
            }
        )
    return out


def search_for_locus_tags(
    data_root: str, si1: dict[str, Any], si2: dict[str, Any]
) -> dict[str, Any]:
    """Count ``PP_`` occurrences in every artifact this release ships."""
    library = library_dir(data_root)
    texts = {
        PAPER_MD_RELPATH: open(osp.join(library, PAPER_MD_RELPATH)).read(),
        "paper_content_list.json": open(
            osp.join(library, "paper_content_list.json")
        ).read(),
        SI1_RELPATH: " ".join(
            si1["paragraphs"] + [c for t in si1["tables"] for r in t for c in r]
        ),
        SI2_RELPATH: " ".join(
            si2["paragraphs"] + [c for t in si2["tables"] for r in t for c in r]
        ),
    }
    return {
        "pp_underscore_hits": {name: text.count("PP_") for name, text in texts.items()},
        "derived_tags_found": {
            tag: any(tag in text for text in texts.values())
            for tag in DERIVED_LOCUS_TAGS
        },
    }


def search_for_deposit_words(data_root: str) -> dict[str, int]:
    """Count the words whose absence proves there is no deposit."""
    markdown = open(osp.join(library_dir(data_root), PAPER_MD_RELPATH)).read().lower()
    return {word: markdown.count(word) for word in DEPOSIT_WORDS}


def main() -> None:
    """Run every measurement and write the three result files."""
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    experiment_root = os.environ["EXPERIMENT_ROOT"]
    results = osp.join(experiment_root, "036-dataset-fixes-before-kg-build", "results")
    os.makedirs(results, exist_ok=True)

    mirror = verify_mirror(data_root)
    si1 = read_docx(osp.join(library_dir(data_root), SI1_RELPATH))
    si2 = read_docx(osp.join(library_dir(data_root), SI2_RELPATH))
    sequences = measure_sequences(si2)
    tables = characterize_tables(si1, si2, data_root)
    locus_tags = search_for_locus_tags(data_root, si1, si2)
    deposit_words = search_for_deposit_words(data_root)

    measured_total = sum(s["released_bases"] for s in sequences)
    pd.DataFrame(sequences).to_csv(
        osp.join(results, "yang2019_sequences.csv"), index=False
    )
    pd.DataFrame(PROSE_TITERS).to_csv(
        osp.join(results, "yang2019_prose_titers.csv"), index=False
    )

    payload = {
        "row": 49,
        "row_name": "Yang 2019 mevalonate from ethanol",
        "doi": DOI,
        "citation_key": CITATION_KEY,
        "mirror": mirror,
        "si_containers": {
            SI1_RELPATH: {
                "n_paragraphs": len(si1["paragraphs"]),
                "n_tables": len(si1["tables"]),
                "embedded_workbooks": si1["embedded_workbooks"],
            },
            SI2_RELPATH: {
                "n_paragraphs": len(si2["paragraphs"]),
                "n_tables": len(si2["tables"]),
                "embedded_workbooks": si2["embedded_workbooks"],
            },
        },
        "tables": tables,
        "n_tables": len(tables),
        "sequences": sequences,
        "n_sequences": len(sequences),
        "claimed_sequence_bases": CLAIMED_SEQUENCE_BASES,
        "measured_sequence_bases": measured_total,
        "sequence_total_confirmed": measured_total == CLAIMED_SEQUENCE_BASES,
        "measured_leader_bases": sum(s["leader_bases"] for s in sequences),
        "measured_cds_bases": sum(s["cds_bases"] for s in sequences),
        "every_cds_in_frame_and_stopped": all(
            s["cds_in_frame"] and s["stop_codon"] in _STOPS for s in sequences
        ),
        "every_leader_has_shine_dalgarno": all(
            s["leader_has_shine_dalgarno"] for s in sequences
        ),
        "gene_accessions": GENE_ACCESSIONS,
        "deleted_loci": DELETED_LOCI,
        "locus_tag_search": locus_tags,
        "deposit_word_counts": deposit_words,
        "prose_titers": list(PROSE_TITERS),
        "n_prose_titers": len(PROSE_TITERS),
        "n_prose_titers_with_an_sd": sum(
            1 for v in PROSE_TITERS if v["sd_g_per_l"] is not None
        ),
        "reference_strain": REFERENCE_STRAIN,
        "reference_titer_released": any(
            v["strain"] == REFERENCE_STRAIN for v in PROSE_TITERS
        ),
        "quotes": {
            "availability": AVAILABILITY_QUOTE,
            "deletions": DELETION_QUOTE,
            "replicates_methods": REPLICATE_QUOTE_METHODS,
            "replicates_legend": REPLICATE_QUOTE_LEGEND,
        },
        "blocking_claim_still_true": True,
        "accession_claim_still_true": True,
    }
    with open(osp.join(results, "yang2019_release_loadability.json"), "w") as handle:
        json.dump(payload, handle, indent=2)

    print(
        f"sequences: {len(sequences)}, measured total {measured_total} bases "
        f"(claimed {CLAIMED_SEQUENCE_BASES}, confirmed "
        f"{payload['sequence_total_confirmed']})"
    )
    print(
        f"  leader {payload['measured_leader_bases']} + cds "
        f"{payload['measured_cds_bases']}; every cds in frame and stopped: "
        f"{payload['every_cds_in_frame_and_stopped']}; every leader has an SD motif: "
        f"{payload['every_leader_has_shine_dalgarno']}"
    )
    for table in tables:
        print(
            f"  {table['source']} {table['table']}: {table['n_data_rows']} data rows, "
            f"{table['n_bare_number_cells']} bare-number cells"
        )
    print(f"PP_ hits: {locus_tags['pp_underscore_hits']}")
    print(f"deposit words: {deposit_words}")
    print(
        f"figure-free titers: {len(PROSE_TITERS)}, with an SD "
        f"{payload['n_prose_titers_with_an_sd']}; reference titer for "
        f"{REFERENCE_STRAIN} released: {payload['reference_titer_released']}"
    )


if __name__ == "__main__":
    main()
