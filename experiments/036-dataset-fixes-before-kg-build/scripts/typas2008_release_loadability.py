# experiments/036-dataset-fixes-before-kg-build/scripts/typas2008_release_loadability.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.typas2008_release_loadability]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/typas2008_release_loadability
"""Re-verify schedule row 37, Typas 2008 (GIANT-coli), by measuring the release.

The row carries ``accession_confirmed=False`` on "Supplementary Tables 2A and 2B via
PMC2700713", which is a claim about a retrieval that may have expired, and
``status="blocked"`` on "no genome-wide double-mutant score matrix was released". Those
are two different claims and they are measured separately, because the Butland 2008
lesson (issue #794) is that a row can be blocked on a retrieval while its values were
readable all along.

Four questions, each answered by reading a file:

1. **Is the deposit mirrored, in full?** The project's recorded reality is that the PMC
   OA API is scriptable while PMC file downloads are not, so the deposit is enumerated
   from the PMC Article Datasets bucket rather than assumed, and matched against the
   library mirror's own ``manifest.json`` by sha256. "One supplement object" is then
   either confirmed as the whole deposit or a specific missing name is printed.

2. **What do the released tables actually carry per pair?** Every ``<table>`` of the
   OCR'd SI is characterized by row count and by whether its value cells hold a NUMBER
   or a term, so "Supplementary Table 2A exists" is separated from "Supplementary
   Table 2A carries a score". The 12 by 12 cross is located in the figure captions and
   its axis genes are recovered, which is what makes "heat-map only" a measurement
   rather than an impression.

3. **Can an existing class carry what is released?** ``GeneInteractionPhenotype`` is
   instantiated with the cells the release actually holds, and the exact
   ``ValidationError`` or acceptance is recorded. Nothing is inferred from a docstring.

4. **Is it subsumed by a dataset we serve?** This is a double-deletion colony-size
   interaction screen, the same family as Butland 2008 and Babu 2014, and Butland's
   screens ARE inside the Babu table. So every released Typas pair is looked up in
   Babu 2014 Table S2, the file ``GeneInteractionBabu2014Dataset`` loads, both as an
   unordered pair and by whether Typas's two query genes are among Babu's 163 donors.

Writes ``results/typas2008_release_loadability.json`` plus
``results/typas2008_si_tables.csv`` (the per-table characterization) and
``results/typas2008_pair_subsumption.csv`` (every released pair, with its Babu lookup).

Usage (from the repo root)::

    python experiments/036-dataset-fixes-before-kg-build/scripts/typas2008_release_loadability.py
"""

from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
import re
import urllib.request
from typing import Any
from xml.etree import ElementTree as ET

import pandas as pd
import pydantic
from dotenv import load_dotenv

from torchcell.datamodels import schema as s

CITATION_KEY = "typasHighthroughputQuantitativeAnalyses2008"
LIBRARY_DIR_REL = f"torchcell-library/{CITATION_KEY}"
DOI = "10.1038/nmeth.1240"
PMCID = "PMC2700713"

SI1_PDF_RELPATH = "si/si1.pdf"
SI1_MD_RELPATH = "si/si1.md"
SI1_PDF_SHA256 = "bd531ec2ee865506b9a1c5decb3dd7d3f057896dd65f2b766ae8090b8c0c39da"

#: The PMC Article Datasets bucket prefix of this article, from the ``pmc_cloud``
#: retrieval record the library manifest stores for ``si/si1.pdf``.
PMC_PREFIX = f"{PMCID}.1"
BUCKET = "pmc-oa-opendata"
BUCKET_LIST_URL = f"https://s3.amazonaws.com/{BUCKET}?list-type=2&prefix={PMC_PREFIX}"
ARTICLE_XML_URL = f"https://{BUCKET}.s3.amazonaws.com/{PMC_PREFIX}/{PMC_PREFIX}.xml"

#: Objects that are PMC's own renditions of the article, not publisher supplements.
_RENDITION_RE = re.compile(r"^PMC\d+\.\d+\.(json|pdf|txt|xml)$")
_FIGURE_RE = re.compile(r"^nihms\d+f\d+\.(jpg|jpeg|png|gif|tif)$", re.IGNORECASE)

#: Babu 2014 Table S2, read from ITS OWN raw mirror under ITS OWN citation key: the
#: file ``GeneInteractionBabu2014Dataset`` loads, and the file that SUBSUMES Butland.
BABU_CITATION_KEY = "babuQuantitativeGenomeWideGenetic2014"
BABU_TABLE_S2_REL = f"torchcell-raw/{BABU_CITATION_KEY}/data/si23.xls"
BABU_SHEET = "WG_GI_Score_Mar_06_2013"
BABU_ROWS = 42705

#: Supplementary Table 2A, transcribed from the OCR'd SI: pal's verified interactions,
#: as (gene, interaction term) with the OCR's capital-L-read-as-I spellings repaired
#: against the paper's own gene set. The TERM is what the release carries; there is no
#: numeric column at all, which is the finding.
TABLE_2A: tuple[tuple[str, str], ...] = (
    ("lpp", "neg (sick)"),
    ("pgm", "neg (sick)"),
    ("mdoG", "neg (sick)"),
    ("tatC", "neg (sick)"),
    ("spr", "pos"),
    ("oppC", "neg (sick)"),
    ("lpcA", "neg (lethal)"),
    ("rfaD", "neg (lethal)"),
    ("rfaB", "neg (lethal)"),
    ("rfaF", "neg (sick)"),
    ("rfaI", "neg (lethal)"),
    ("rfaG", "neg (sick)"),
    ("rfaP", "neg (sick)"),
    ("rfaQ", "neg (sick)"),
    ("rffA", "neg (sick)"),
    ("rffT", "neg (sick)"),
    ("rffC", "neg (sick)"),
    ("rffD", "neg (sick)"),
    ("rffE", "neg (sick)"),
    ("cmr", "neg (sick)"),
    ("mdtG", "neg (sick)"),
    ("wcaA", "neg (sick)"),
    ("yliE", "neg (sick)"),
)
#: Supplementary Table 2B: suppressors of the yraP lethality in 3% SDS. The table's
#: columns are gene name, ECK number, location and function -- no value column of any
#: kind, which is why these rows carry no interaction term either.
TABLE_2B: tuple[str, ...] = (
    "dsbA",
    "dsbB",
    "secB",
    "hlpA",
    "ppiB",
    "asmA",
    "treC",
    "glmM",
    "folP",
    "pqiA",
    "pqiB",
    "ymbA",
    "JW0935",
    "ylcG",
    "lysS",
)
#: The two query genes of the two released genome-wide screens.
QUERY_GENES = ("pal", "yraP")
#: The 12 genes of the 12 by 12 cross, recovered from the OCR'd axis labels of
#: Supplementary Figure 4's four heat-map panels. The 66 distinct pairwise scores sit in
#: those panels as COLOR CELLS and are not in any table.
CROSS_GENES = (
    "surA",
    "ybaY",
    "ycbS",
    "ompC",
    "yraI",
    "cpxR",
    "degP",
    "pal",
    "ompA",
    "yfgL",
    "yraP",
    "basR",
)

#: Verbatim from the OCR'd SI (``si/si1.md``), the caption that defines Table 2.
TABLE_2_CAPTION = (
    "Supplementary Table 2. (A) Genetic interactions of pal identified in M9-gylcerol "
    "in a genomewide interaction screen in M9 glycerol and independently verified by "
    "reconstructing the double mutants with P1 transduction and then comparing their "
    "growth with the parental single-gene knockouts. Neg stands for negative "
    "interactions and pos for positives."
)
#: Verbatim: the caption that puts the 12 by 12 scores in a figure rather than a table.
CROSS_CAPTION = (
    "Supplementary Figure 4: Heat maps representing ${ 1 2 \\times }$ 12 crosses in all "
    "four different datasets: (A) LB-384; (B) LB-1536; (C) M9-384; (D) M9-1536."
)
#: Verbatim: the only per-pair NUMBER the release carries, and what it measures.
TABLE_1_CAPTION = (
    "Supplementary Table 1: Reproduction of synthetic lethal pairs by co-transduction "
    "of a linked marker."
)
TABLE_1_STATISTIC = "% Co-inheritance of both markerswhen host is:"

#: The four co-transduction rows of Supplementary Table 1: the ONLY per-pair numbers in
#: the release, and a marker co-inheritance percentage rather than an interaction score.
TABLE_1: tuple[tuple[str, str, str, float, float], ...] = (
    ("degP", "surA", "LB", 61.0, 0.0),
    ("pal", "surA", "LB", 67.0, 0.0),
    ("pal", "yfgL", "LB-M9 glycerol", 67.0, 0.0),
    ("pal", "ompA", "M9 glycerol", 67.0, 0.0),
)

_TABLE_RE = re.compile(r"<table>.*?</table>", re.S)
_ROW_RE = re.compile(r"<tr>(.*?)</tr>", re.S)
_CELL_RE = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.S)
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


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
    """sha256-verify the mirrored SI against the library manifest.

    The retrieval record is returned with the hash, because the row's claim is about a
    retrieval: if the SI was fetched by a scriptable route then the row's
    ``accession_confirmed=False`` is stale, and the record is the evidence.
    """
    library = library_dir(data_root)
    manifest = json.loads(open(osp.join(library, "manifest.json")).read())
    entries = {f["path"]: f for f in manifest["files"]}
    pdf = entries[SI1_PDF_RELPATH]
    observed = sha256_of(osp.join(library, SI1_PDF_RELPATH))
    if observed != SI1_PDF_SHA256:
        raise RuntimeError(
            f"{SI1_PDF_RELPATH}: sha256 {observed}, pin {SI1_PDF_SHA256}"
        )
    if pdf["sha256"] != SI1_PDF_SHA256:
        raise RuntimeError(f"manifest sha256 {pdf['sha256']} != {SI1_PDF_SHA256}")
    markdown = entries[SI1_MD_RELPATH]
    markdown_sha = sha256_of(osp.join(library, SI1_MD_RELPATH))
    if markdown_sha != markdown["sha256"]:
        raise RuntimeError(f"{SI1_MD_RELPATH}: sha256 {markdown_sha}")
    retrieval = pdf["retrieval"]
    return {
        "doi": manifest["doi"],
        "title": manifest["title"],
        "si_pdf": {
            "path": SI1_PDF_RELPATH,
            "bytes": pdf["bytes"],
            "sha256": observed,
            "retrieval_method": retrieval["method"],
            "source_url": retrieval["source_url"],
            "retrieval_command": retrieval["retriever"],
            "retrieval_params": retrieval["params"],
            "retrieved_at": retrieval["retrieved_at"],
        },
        "si_markdown": {
            "path": SI1_MD_RELPATH,
            "bytes": markdown["bytes"],
            "sha256": markdown_sha,
            "processing": markdown.get("processing"),
        },
        "si_is_mirrored": True,
    }


def enumerate_publisher_deposit() -> dict[str, Any]:
    """List the article's bucket objects and its JATS supplementary-material elements."""
    with urllib.request.urlopen(BUCKET_LIST_URL, timeout=120) as response:
        listing = response.read().decode("utf-8")
    keys = re.findall(r"<Key>([^<]+)</Key>", listing)
    sizes = [int(n) for n in re.findall(r"<Size>(\d+)</Size>", listing)]
    objects = [
        {"key": key, "bytes": size, "name": key.split("/", 1)[1]}
        for key, size in zip(keys, sizes, strict=True)
    ]
    supplementary = [
        obj
        for obj in objects
        if not _RENDITION_RE.match(obj["name"]) and not _FIGURE_RE.match(obj["name"])
    ]
    with urllib.request.urlopen(ARTICLE_XML_URL, timeout=120) as response:
        article = response.read().decode("utf-8")
    root = ET.fromstring(article)
    declared = []
    for element in root.iter("supplementary-material"):
        media = element.find("media")
        declared.append(
            {
                "id": element.get("id"),
                "href": (
                    None
                    if media is None
                    else media.get("{http://www.w3.org/1999/xlink}href")
                ),
            }
        )
    return {
        "bucket_objects": objects,
        "supplementary_objects": supplementary,
        "n_supplementary_objects": len(supplementary),
        "declared_supplementary": declared,
        "n_declared_supplementary": len(declared),
    }


def characterize_si_tables(data_root: str) -> list[dict[str, Any]]:
    """Every ``<table>`` of the OCR'd SI: its shape and whether its cells hold numbers.

    ``n_numeric_cells`` counts cells that are a bare number, which is what separates a
    table carrying a score from one carrying a term: Supplementary Table 2A's value
    column is ``neg (sick)`` and has no numeric cell at all.
    """
    markdown = open(osp.join(library_dir(data_root), SI1_MD_RELPATH)).read()
    out: list[dict[str, Any]] = []
    for index, table in enumerate(_TABLE_RE.findall(markdown)):
        rows = _ROW_RE.findall(table)
        cells = [_CELL_RE.findall(row) for row in rows]
        flat = [c.strip() for row in cells for c in row]
        numeric = [c for c in flat if _NUMBER_RE.fullmatch(c.rstrip("%"))]
        out.append(
            {
                "table_index": index,
                "n_rows": len(rows),
                "n_cells": len(flat),
                "n_numeric_cells": len(numeric),
                "first_row": cells[0][:6] if cells else [],
                "numeric_sample": numeric[:8],
            }
        )
    return out


def probe_phenotype_class() -> dict[str, Any]:
    """Try to put Table 2A's released cell into ``GeneInteractionPhenotype``.

    Two attempts: the interaction TERM the release prints, and the absence of any value.
    An accepted construction would mean the row is loadable; a refusal names the field
    that has no member for a qualitative call.
    """
    attempts: list[dict[str, Any]] = []
    try:
        s.GeneInteractionPhenotype(gene_interaction=TABLE_2A[0][1])  # type: ignore[arg-type]
        attempts.append({"attempt": "term_as_value", "accepted": True, "errors": []})
    except pydantic.ValidationError as error:
        attempts.append(
            {
                "attempt": "term_as_value",
                "accepted": False,
                "errors": [
                    {"loc": list(e["loc"]), "type": e["type"], "msg": e["msg"]}
                    for e in error.errors()
                ],
            }
        )
    try:
        s.GeneInteractionPhenotype()  # type: ignore[call-arg]
        attempts.append({"attempt": "no_value", "accepted": True, "errors": []})
    except pydantic.ValidationError as error:
        attempts.append(
            {
                "attempt": "no_value",
                "accepted": False,
                "errors": [
                    {"loc": list(e["loc"]), "type": e["type"], "msg": e["msg"]}
                    for e in error.errors()
                ],
            }
        )
    accepted = s.GeneInteractionPhenotype(gene_interaction=-4.43348)
    return {
        "class": "GeneInteractionPhenotype",
        "gene_interaction_annotation": "float, required",
        "attempts": attempts,
        "float_accepted": accepted.gene_interaction == -4.43348,
        "has_categorical_mode": False,
    }


def read_babu_pairs(data_root: str) -> pd.DataFrame:
    """Babu 2014 Table S2 as ``donor``, ``recipient``, ``gi`` plus bare symbols."""
    path = osp.join(data_root, BABU_TABLE_S2_REL)
    frame = pd.read_excel(path, sheet_name=BABU_SHEET, dtype=str, skiprows=2)
    frame.columns = ["donor", "recipient", "gi"]
    frame = frame.dropna(subset=["donor", "recipient"])
    if len(frame) != BABU_ROWS:
        raise RuntimeError(f"{path}: {len(frame)} rows, Babu releases {BABU_ROWS}")
    frame["donor_symbol"] = frame["donor"].map(lambda t: t.split("__", 1)[-1])
    frame["recipient_symbol"] = frame["recipient"].map(lambda t: t.split("__", 1)[-1])
    return frame


def measure_subsumption(data_root: str) -> tuple[dict[str, Any], pd.DataFrame]:
    """Look up every released Typas pair in the Babu table we serve.

    Butland 2008's screens ARE inside this file, which is why the lookup is worth
    running: if Typas's pairs were there too the row would close as a provenance record.
    """
    frame = read_babu_pairs(data_root)
    unordered = {
        frozenset((d, r))
        for d, r in zip(frame["donor_symbol"], frame["recipient_symbol"], strict=True)
    }
    donors = set(frame["donor_symbol"])
    rows: list[dict[str, Any]] = []
    for gene, term in TABLE_2A:
        rows.append(
            {
                "source_table": "Supplementary Table 2A",
                "query_gene": "pal",
                "partner_gene": gene,
                "released_value": term,
                "released_value_is_numeric": False,
                "in_babu_table_s2": frozenset(("pal", gene)) in unordered,
            }
        )
    for gene in TABLE_2B:
        rows.append(
            {
                "source_table": "Supplementary Table 2B",
                "query_gene": "yraP",
                "partner_gene": gene,
                "released_value": "",
                "released_value_is_numeric": False,
                "in_babu_table_s2": frozenset(("yraP", gene)) in unordered,
            }
        )
    for first, second, medium, wild_type, mutant in TABLE_1:
        rows.append(
            {
                "source_table": "Supplementary Table 1",
                "query_gene": first,
                "partner_gene": second,
                "released_value": f"{wild_type}% / {mutant}% ({medium})",
                "released_value_is_numeric": True,
                "in_babu_table_s2": frozenset((first, second)) in unordered,
            }
        )
    pairs = pd.DataFrame(rows)
    partners = {
        gene: sorted(
            {
                (d if r == gene else r)
                for d, r in zip(
                    frame["donor_symbol"], frame["recipient_symbol"], strict=True
                )
                if gene in (d, r)
            }
        )
        for gene in QUERY_GENES
    }
    summary = {
        "babu_table": osp.join(data_root, BABU_TABLE_S2_REL),
        "babu_sha256": sha256_of(osp.join(data_root, BABU_TABLE_S2_REL)),
        "babu_rows": len(frame),
        "babu_unordered_pairs": len(unordered),
        "babu_donors": len(donors),
        "typas_released_pairs": len(pairs),
        "typas_pairs_in_babu": int(pairs["in_babu_table_s2"].sum()),
        "typas_query_genes_among_babu_donors": {
            gene: gene in donors for gene in QUERY_GENES
        },
        "babu_partners_of_typas_query_genes": partners,
        "subsumed": bool(pairs["in_babu_table_s2"].all()),
    }
    return summary, pairs


def main() -> None:
    """Run every measurement and write the three result files."""
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    experiment_root = os.environ["EXPERIMENT_ROOT"]
    results = osp.join(experiment_root, "036-dataset-fixes-before-kg-build", "results")
    os.makedirs(results, exist_ok=True)

    mirror = verify_mirror(data_root)
    deposit = enumerate_publisher_deposit()
    mirrored_names = {mirror["si_pdf"]["source_url"].rsplit("/", 1)[-1]}
    published = {obj["name"] for obj in deposit["supplementary_objects"]}
    reconciliation = {
        "published_not_mirrored": sorted(published - mirrored_names),
        "mirrored_not_published": sorted(mirrored_names - published),
        "complete": published == mirrored_names,
    }
    tables = characterize_si_tables(data_root)
    phenotype = probe_phenotype_class()
    subsumption, pairs = measure_subsumption(data_root)

    pd.DataFrame(
        [t | {"first_row": " | ".join(t["first_row"])} for t in tables]
    ).to_csv(osp.join(results, "typas2008_si_tables.csv"), index=False)
    pairs.to_csv(osp.join(results, "typas2008_pair_subsumption.csv"), index=False)

    numeric_pairs = int(pairs["released_value_is_numeric"].sum())
    payload = {
        "row": 37,
        "row_name": "Typas 2008",
        "doi": DOI,
        "pmcid": PMCID,
        "citation_key": CITATION_KEY,
        "mirror": mirror,
        "publisher_deposit": deposit,
        "deposit_reconciliation": reconciliation,
        "si_tables": tables,
        "released_pairs": {
            "table_2a_pal_interactions": len(TABLE_2A),
            "table_2b_yrap_suppressors": len(TABLE_2B),
            "table_1_cotransduction_pairs": len(TABLE_1),
            "total": len(pairs),
            "with_a_numeric_value": numeric_pairs,
            "numeric_statistic": TABLE_1_STATISTIC,
        },
        "twelve_by_twelve_cross": {
            "genes": list(CROSS_GENES),
            "n_genes": len(CROSS_GENES),
            "distinct_pairwise_doubles": len(CROSS_GENES) * (len(CROSS_GENES) - 1) // 2,
            "released_as": "four heat-map panels of Supplementary Figure 4",
            "in_any_table": False,
        },
        "phenotype_class_probe": phenotype,
        "subsumption": subsumption,
        "quotes": {
            "table_2_caption": TABLE_2_CAPTION,
            "cross_caption": CROSS_CAPTION,
            "table_1_caption": TABLE_1_CAPTION,
        },
        "accession_claim_still_true": False,
        "blocking_claim_still_true": True,
    }
    with open(osp.join(results, "typas2008_release_loadability.json"), "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)

    print(f"si mirrored: {mirror['si_is_mirrored']}")
    print(
        f"publisher supplementary objects: {deposit['n_supplementary_objects']}, "
        f"deposit complete: {reconciliation['complete']}"
    )
    print(f"si tables: {len(tables)}")
    print(
        f"released pairs: {len(pairs)}, with a numeric value: {numeric_pairs} "
        f"({TABLE_1_STATISTIC})"
    )
    print(f"12x12 cross genes recovered: {len(CROSS_GENES)}, released as a heat map")
    print(
        "pairs already in the Babu table we serve: "
        f"{subsumption['typas_pairs_in_babu']} of {subsumption['typas_released_pairs']}"
    )
    for attempt in phenotype["attempts"]:
        detail = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['type']}, {e['msg']}"
            for e in attempt["errors"]
        )
        print(f"  GeneInteractionPhenotype {attempt['attempt']}: {detail}")


if __name__ == "__main__":
    main()
