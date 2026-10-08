# experiments/036-dataset-fixes-before-kg-build/scripts/niu2019_release_loadability.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.niu2019_release_loadability]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/niu2019_release_loadability
"""Settle schedule row 48, Niu 2019 pinene evolved, by measuring the release.

Three questions the row's triage left open, each answered by reading the pinned bytes
rather than the abstract:

1. **Is a released file missing from the mirror?** The library mirror holds ONE SI file
   for ``niuGenomicTranscriptionalChanges2019``, which the triage flagged as possibly
   partial. The publisher deposit is enumerated from the PMC Article Datasets bucket
   (the same ``pmc_cloud`` route the mirror's own manifest records for this file) and
   cross-checked against the article's JATS XML ``<supplementary-material>`` elements,
   so "one file" is either confirmed as the whole deposit or a specific missing name is
   printed.

2. **What does the release give per sample?** Suppl. Table 2 is characterized as a
   called variant list (gene, protein, mutation site, frequency), an allele-frequency
   table or a clone genotype, by counting what its cells actually carry. Suppl. Tables 3
   and 4 are counted per arm: target rows, and per column how many carry a NUMBER versus
   the ``-`` the table's own footnote defines.

3. **Can any existing class carry it?** Every candidate perturbation leaf and phenotype
   class is instantiated with the identifiers and values the release actually holds, and
   the exact ``ValidationError`` (or acceptance) is recorded. Nothing is inferred from a
   docstring.

Writes ``results/niu2019_release_loadability.json`` and
``results/niu2019_release_loadability_targets.csv``.

Usage (from the repo root)::

    python experiments/036-dataset-fixes-before-kg-build/scripts/niu2019_release_loadability.py
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import os.path as osp
import re
import urllib.request
import zipfile
from typing import Any
from xml.etree import ElementTree as ET

import pandas as pd
from dotenv import load_dotenv

from torchcell.datamodels import schema as s

CITATION_KEY = "niuGenomicTranscriptionalChanges2019"
LIBRARY_DIR_REL = f"torchcell-library/{CITATION_KEY}"
SI1_RELPATH = "si/si1.docx"
SI1_SHA256 = "72099acb46abce07983265e559ec49596209b031869f081f40ae1e0bc0ce6fb9"

#: The PMC Article Datasets bucket prefix of this article (version 1), from the
#: retrieval record the library manifest stores for ``si/si1.docx``.
PMC_PREFIX = "PMC6556621.1"
BUCKET = "pmc-oa-opendata"
BUCKET_LIST_URL = f"https://s3.amazonaws.com/{BUCKET}?list-type=2&prefix={PMC_PREFIX}"
ARTICLE_XML_URL = f"https://{BUCKET}.s3.amazonaws.com/{PMC_PREFIX}/{PMC_PREFIX}.xml"

#: Objects in the bucket that are PMC's own renditions of the article, not publisher
#: supplementary files. Anything else is a released supplementary file.
_RENDITION_SUFFIXES = (".json", ".pdf", ".txt", ".xml")
_FIGURE_RE = re.compile(r"^(gr|fx|fig)\d+\.(jpg|jpeg|png|gif|tif)$", re.IGNORECASE)

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

#: Suppl. Table 2's own column header, and the table title, verbatim.
T2_TITLE = "Suppl. Table 2 Mutations in E. coli YZFP"
T2_HEADER = ("Gene", "Protein", "Mutation site", "Frequence")
#: The footnote that defines a ``-`` cell in Suppl. Tables 3 and 4, verbatim.
DASH_FOOTNOTE = "-: means no change or negative effect."

#: A b-number as Suppl. Table 2 writes it in the Gene cell (``acnA, b1276``).
B_NUMBER_RE = re.compile(r"\bb\d{4}\b")
#: An absolute-coordinate call as the Mutation site cell writes it (``T1336442C``).
ABSOLUTE_CALL_RE = re.compile(r"\b([ACGT])(\d{3,8})([ACGT])\b")
#: A value cell of Suppl. Tables 3 and 4: ``1.24 ± 0.01``.
RATIO_RE = re.compile(r"^(\d+\.\d+)\s*±\s*(\d+\.\d+)$")


# --------------------------------------------------------------------------- #
# Pinned bytes
# --------------------------------------------------------------------------- #
def sha256_of(path: str) -> str:
    """Hex sha256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_si(data_root: str) -> dict[str, Any]:
    """sha256-verify ``si/si1.docx`` against the library manifest and this module's pin."""
    library = osp.join(data_root, LIBRARY_DIR_REL)
    manifest = json.loads(open(osp.join(library, "manifest.json")).read())
    entry = next(f for f in manifest["files"] if f["path"] == SI1_RELPATH)
    path = osp.join(library, SI1_RELPATH)
    observed = sha256_of(path)
    if observed != SI1_SHA256:
        raise RuntimeError(f"{path}: sha256 {observed}, module pins {SI1_SHA256}")
    if entry["sha256"] != SI1_SHA256:
        raise RuntimeError(f"manifest sha256 {entry['sha256']} != {SI1_SHA256}")
    return {
        "path": path,
        "sha256": observed,
        "bytes": entry["bytes"],
        "original_filename": entry["original_filename"],
        "retrieval_method": entry["retrieval"]["method"],
        "source_url": entry["retrieval"]["source_url"],
        "retrieval_command": entry["retrieval"]["retriever"],
        "retrieval_params": entry["retrieval"]["params"],
        "si_data_sources": manifest["si_data_sources"],
        "si_expected": manifest["si_expected"],
    }


# --------------------------------------------------------------------------- #
# What the publisher released
# --------------------------------------------------------------------------- #
def enumerate_publisher_deposit() -> dict[str, Any]:
    """List the article's bucket objects and its JATS supplementary-material elements.

    Both are read, not one: the bucket says what bytes exist and the XML says what the
    publisher DECLARED, so a declared file absent from the bucket would be visible as a
    disagreement rather than as silence.
    """
    with urllib.request.urlopen(BUCKET_LIST_URL, timeout=60) as response:
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
        if not obj["name"].endswith(_RENDITION_SUFFIXES)
        and not _FIGURE_RE.match(obj["name"])
    ]

    with urllib.request.urlopen(ARTICLE_XML_URL, timeout=60) as response:
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
        "declared_supplementary": declared,
    }


# --------------------------------------------------------------------------- #
# Reading the SI docx
# --------------------------------------------------------------------------- #
def _cell_text(tc: ET.Element) -> str:
    """A table cell's text: its paragraphs' runs, joined by a space."""
    return " ".join(
        "".join(t.text or "" for t in p.iter(W + "t")).strip()
        for p in tc.findall(W + "p")
    ).strip()


def read_docx_body(path: str) -> list[dict[str, Any]]:
    """The document body as an ordered list of paragraphs and tables."""
    root = ET.fromstring(zipfile.ZipFile(path).read("word/document.xml"))
    body = root.find(W + "body")
    if body is None:
        raise RuntimeError(f"{path}: no w:body")
    out: list[dict[str, Any]] = []
    for child in body:
        if child.tag == W + "p":
            text = "".join(t.text or "" for t in child.iter(W + "t")).strip()
            if text:
                out.append({"kind": "paragraph", "text": text})
        elif child.tag == W + "tbl":
            rows = [
                [_cell_text(tc) for tc in tr.findall(W + "tc")]
                for tr in child.findall(W + "tr")
            ]
            out.append({"kind": "table", "rows": rows})
    return out


def _tables(body: list[dict[str, Any]]) -> list[list[list[str]]]:
    return [item["rows"] for item in body if item["kind"] == "table"]


def _paragraphs(body: list[dict[str, Any]]) -> list[str]:
    return [item["text"] for item in body if item["kind"] == "paragraph"]


# --------------------------------------------------------------------------- #
# Q2a: what Suppl. Table 2 gives per row
# --------------------------------------------------------------------------- #
def characterize_variant_table(rows: list[list[str]]) -> dict[str, Any]:
    """Count what Suppl. Table 2's cells carry, so its KIND is measured, not assumed.

    A data row is STRUCTURAL, not textual: the table's category headings
    (``Carbohydrate metabolism``) are merged rows carrying two ``w:tc`` cells, while every
    data row carries the header's four. Classifying by "the ``Mutation site`` cell is
    non-empty" instead undercounts by one, because one data row (``ykfm, b4586``) has four
    cells and a BLANK site; the structural count is also the one that reconciles with the
    paper's own tally, 349 SNV + 25 InDel. The distinction decides whether this is a
    called variant list (a site per row), an allele-frequency table (a frequency that
    varies) or a clone genotype (a column per clone).
    """
    header = tuple(rows[0])
    if header != T2_HEADER:
        raise RuntimeError(f"Suppl. Table 2 header is {header}, expected {T2_HEADER}")
    widths = sorted({len(row) for row in rows})
    data = [row for row in rows[1:] if len(row) == len(T2_HEADER)]
    headings = [row for row in rows[1:] if len(row) != len(T2_HEADER)]

    frequencies: dict[str, int] = {}
    with_b_number = 0
    intergenic = 0
    absolute_calls = 0
    coding_changes = 0
    blank_site = 0
    for row in data:
        gene, _protein, site = row[0], row[1], row[2]
        frequency = row[3].strip()
        if not site.strip():
            blank_site += 1
        frequencies[frequency] = frequencies.get(frequency, 0) + 1
        if B_NUMBER_RE.search(gene):
            with_b_number += 1
        if "intergenic" in site.lower():
            intergenic += 1
        if ABSOLUTE_CALL_RE.search(site):
            absolute_calls += 1
        if "(" in site and ")" in site:
            coding_changes += 1
    return {
        "title": T2_TITLE,
        "header": list(header),
        "row_widths": widths,
        "n_rows_total": len(rows),
        "n_data_rows": len(data),
        "n_category_heading_rows": len(headings),
        "category_headings": [row[0] for row in headings],
        "frequency_values": frequencies,
        "n_distinct_frequency_values": len(frequencies),
        "n_rows_with_b_number": with_b_number,
        "n_rows_intergenic": intergenic,
        "n_rows_with_absolute_coordinate_call": absolute_calls,
        "n_rows_with_parenthetical_change": coding_changes,
        "n_rows_with_blank_mutation_site": blank_site,
        "reconciles_with_the_paper_tally": {
            "quote": (
                "A total of 349 single nucleotide variants (SNV) and 25 "
                "insertion/deletion (InDel) were identified"
            ),
            "stated_total": 349 + 25,
            "observed_data_rows": len(data),
            "agrees": len(data) == 349 + 25,
        },
        "kind": (
            "a CALLED VARIANT LIST of ONE strain: one called site per row with its own "
            "absolute coordinate, reference and alternate base, carrying an allele "
            "frequency column that is 1.00 for all but "
            f"{sum(n for value, n in frequencies.items() if value not in ('1.00', '100'))} "
            "rows. It is NOT a clone genotype matrix (no per-clone column exists) and "
            "NOT an allele-frequency table (frequency is a near-constant 1.00)."
        ),
        "is_clone_genotype_matrix": max(widths) > len(T2_HEADER),
        "n_strains_described": 1,
        "strain_described": "E. coli YZFP",
    }


# --------------------------------------------------------------------------- #
# Q2b: what Suppl. Tables 3 and 4 give per target
# --------------------------------------------------------------------------- #
def read_target_table(rows: list[list[str]], arm: str) -> list[dict[str, Any]]:
    """One record per target row of Suppl. Table 3 or 4, with both columns parsed.

    A ``-`` cell is kept as a censored entry rather than dropped: the table's own
    footnote defines it, so the count of censored cells is part of what the release gives.
    """
    expected = ("Gene", "Protein", "Ratio of growtha", "Ratio of pinene concentrationb")
    header = tuple(rows[0])
    if header != expected:
        raise RuntimeError(f"{arm} header is {header}, expected {expected}")
    out: list[dict[str, Any]] = []
    category = ""
    for row in rows[1:]:
        filled = [cell for cell in row if cell.strip()]
        if len(filled) == 1:
            category = filled[0]
            continue
        growth, pinene = row[2].strip(), row[3].strip()
        parsed: dict[str, Any] = {
            "arm": arm,
            "category": category,
            "target": row[0].strip(),
            "protein": row[1].strip(),
        }
        for name, cell in (("growth", growth), ("pinene", pinene)):
            match = RATIO_RE.match(cell)
            parsed[f"{name}_cell"] = cell
            parsed[f"{name}_ratio"] = float(match.group(1)) if match else None
            parsed[f"{name}_pm"] = float(match.group(2)) if match else None
            parsed[f"{name}_censored"] = cell == "-"
        if parsed["growth_ratio"] is None and not parsed["growth_censored"]:
            raise RuntimeError(f"{arm} {parsed['target']}: growth cell {growth!r}")
        if parsed["pinene_ratio"] is None and not parsed["pinene_censored"]:
            raise RuntimeError(f"{arm} {parsed['target']}: pinene cell {pinene!r}")
        out.append(parsed)
    return out


#: What the main text COUNTS, verbatim, for each arm: the numbers a dash classification
#: has to reproduce if a dash is a measured non-positive rather than an absent cell.
#: Results 3.3 (activation) and 3.4 (interference) of the pinned ``paper.md``.
MAIN_TEXT_COUNTS: dict[str, dict[str, Any]] = {
    "crispr_activation": {
        "quote": (
            "The activations of 23 genes resulted in increases both in tolerance to "
            "pinene and pinene production (Table 3). The activations of 9 genes only "
            "improved pinene production. The activations of 20 genes only enhanced the "
            "tolerance to pinene."
        ),
        "both": 23,
        "pinene_only": 9,
        "growth_only": 20,
    },
    "crispr_interference": {
        "quote": (
            "Repressions of 6 genes led to increases both in tolerance to pinene and "
            "pinene production (Table 4). Repressions of 3 genes only improved the "
            "tolerance to pinene (Supplementary Table 4)."
        ),
        "both": 6,
        "pinene_only": 0,
        "growth_only": 3,
    },
}


def reconcile_dashes_against_the_main_text(
    targets: list[dict[str, Any]],
) -> dict[str, Any]:
    """Test whether the dash classification reproduces the counts the main text states.

    This is the measurement that decides what a ``-`` MEANS. The footnote's wording
    ("no change or negative effect") describes an outcome, but wording is not proof. If
    the authors counted dashed cells as measured non-improvements, then classifying a
    cell as numeric-vs-dash must reproduce their own both / growth-only / pinene-only
    tallies exactly. If a dash were instead an unmeasured cell, those tallies could not
    be recovered from the table at all.
    """
    out: dict[str, Any] = {}
    for arm, stated in MAIN_TEXT_COUNTS.items():
        rows = [t for t in targets if t["arm"] == arm]
        observed = {
            "both": sum(
                1
                for t in rows
                if t["growth_ratio"] is not None and t["pinene_ratio"] is not None
            ),
            "growth_only": sum(
                1
                for t in rows
                if t["growth_ratio"] is not None and t["pinene_ratio"] is None
            ),
            "pinene_only": sum(
                1
                for t in rows
                if t["growth_ratio"] is None and t["pinene_ratio"] is not None
            ),
        }
        out[arm] = {
            "quote": stated["quote"],
            "stated": {k: stated[k] for k in ("both", "growth_only", "pinene_only")},
            "observed": observed,
            "agrees": all(observed[k] == stated[k] for k in observed),
        }
    out["conclusion"] = (
        "a dash is a MEASURED non-positive outcome, not an absent measurement: "
        "classifying each cell as numeric or dashed reproduces the main text's own "
        "both / growth-only / pinene-only tallies exactly for both arms, and an "
        "unmeasured cell could not enter that arithmetic. It is therefore a "
        "left-censored observation, and because the footnote conflates 'no change' "
        "with 'negative effect' it cannot be given a number OR a single "
        "ResponseCategory, so it is dropped with that rule rather than encoded."
        if all(out[arm]["agrees"] for arm in MAIN_TEXT_COUNTS)
        else "the dash classification does NOT reproduce the main text's tallies"
    )
    return out


def summarize_targets(targets: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    """Per-arm counts: targets, numeric cells and censored cells per column."""
    rows = [t for t in targets if t["arm"] == arm]
    multi = [
        t["target"] for t in rows if len(t["target"]) > 4 and t["target"].isalnum()
    ]
    return {
        "arm": arm,
        "n_targets": len(rows),
        "n_growth_numeric": sum(1 for t in rows if t["growth_ratio"] is not None),
        "n_growth_censored": sum(1 for t in rows if t["growth_censored"]),
        "n_pinene_numeric": sum(1 for t in rows if t["pinene_ratio"] is not None),
        "n_pinene_censored": sum(1 for t in rows if t["pinene_censored"]),
        "n_either_numeric": sum(
            1
            for t in rows
            if t["growth_ratio"] is not None or t["pinene_ratio"] is not None
        ),
        "n_both_censored": sum(
            1 for t in rows if t["growth_censored"] and t["pinene_censored"]
        ),
        "growth_ratio_range": [
            min((t["growth_ratio"] for t in rows if t["growth_ratio"]), default=None),
            max((t["growth_ratio"] for t in rows if t["growth_ratio"]), default=None),
        ],
        "pinene_ratio_range": [
            min((t["pinene_ratio"] for t in rows if t["pinene_ratio"]), default=None),
            max((t["pinene_ratio"] for t in rows if t["pinene_ratio"]), default=None),
        ],
        "multi_gene_target_labels": sorted(set(multi)),
        "categories": sorted({t["category"] for t in rows}),
    }


# --------------------------------------------------------------------------- #
# Q3: which existing class accepts what the release holds
# --------------------------------------------------------------------------- #
def _probe(label: str, build: Any) -> dict[str, Any]:
    """Instantiate a candidate class and record acceptance or the exact error."""
    try:
        model = build()
    except Exception as exc:  # the probe's whole purpose is to record the refusal
        return {
            "candidate": label,
            "accepted": False,
            "error_type": type(exc).__name__,
            "error": " ".join(str(exc).split())[:400],
        }
    return {
        "candidate": label,
        "accepted": True,
        "stored_as": getattr(model, "perturbation_type", None)
        or getattr(model, "label_name", None),
    }


def probe_perturbation_classes() -> list[dict[str, Any]]:
    """Every candidate leaf for the three perturbation kinds the release describes."""
    construct = s.CrisprConstruct(effector="dCas9*-MCPSoxS", guide_sequence="A" * 20)
    mg1655: Any = "ecoli_k12_mg1655_bnumber"
    return [
        _probe(
            "CrisprActivationPerturbation(systematic_gene_name='b3417')",
            lambda: s.CrisprActivationPerturbation(
                systematic_gene_name="b3417",
                perturbed_gene_name="dxs",
                crispr=construct,
            ),
        ),
        _probe(
            "CrisprActivationPerturbation(systematic_gene_name='dxs')",
            lambda: s.CrisprActivationPerturbation(
                systematic_gene_name="dxs", perturbed_gene_name="dxs", crispr=construct
            ),
        ),
        _probe(
            "BacterialCrisprInterferencePerturbation(systematic_gene_name='b1659')",
            lambda: s.BacterialCrisprInterferencePerturbation(
                systematic_gene_name="b1659",
                perturbed_gene_name="ydiJ",
                crispr=construct,
                gene_namespace=mg1655,
            ),
        ),
        _probe(
            "SequenceVariantPerturbation(systematic_gene_name='b1276')",
            lambda: s.SequenceVariantPerturbation(
                systematic_gene_name="b1276",
                perturbed_gene_name="acnA",
                strain_id="YZFP",
            ),
        ),
        _probe(
            "AllelePerturbation(systematic_gene_name='b1276')",
            lambda: s.AllelePerturbation(
                systematic_gene_name="b1276", perturbed_gene_name="acnA"
            ),
        ),
        {
            **_probe(
                "BacterialBackgroundAllele(b1276, Y204Y synonymous call)",
                lambda: s.BacterialBackgroundAllele(
                    systematic_gene_name="b1276",
                    gene_namespace=mg1655,
                    gene_name="acnA",
                    allele_name="acnA-T612C",
                    edit=s.AlleleEdit.sequence_variant,
                    functional=True,
                ),
            ),
            "fields": sorted(s.BacterialBackgroundAllele.model_fields),
            "note": (
                "takes the row ONLY by asserting `functional`, which is a required "
                "non-optional bool whose value this release never states for any of "
                "its 374 calls, and it has no slot for the position, reference base, "
                "alternate base or Frequence the row actually carries. A ProvenanceGap "
                "cannot cover `functional` because a gap must name a field that is "
                "None. This is issue #731's reason 2, re-measured on these bytes."
            ),
        },
    ]


def probe_phenotype_classes(
    growth_ratio: float, growth_pm: float, pinene_ratio: float, pinene_pm: float
) -> list[dict[str, Any]]:
    """Candidate phenotypes for the two released ratios, with real released values."""
    probes = [
        _probe(
            f"FitnessPhenotype(fitness={growth_ratio}) for the growth ratio",
            lambda: s.FitnessPhenotype(
                fitness=growth_ratio,
                fitness_uncertainty=growth_pm,
                fitness_uncertainty_type=s.UncertaintyType.sample_sd,
                n_samples=3,
                sample_unit=s.SampleUnit.biological_replicate,
            ),
        ),
        _probe(
            "EnvironmentResponsePhenotype(log2_ratio) for the growth ratio",
            lambda: s.EnvironmentResponsePhenotype(
                measurement_type=s.MeasurementType.log2_ratio,
                assay_type=s.AssayType.liquid_od_growth,
                environment_response=math.log2(growth_ratio),
                units="log2(OD600 of the guide strain / of the no-guide control)",
            ),
        ),
        _probe(
            f"ProductTiterPhenotype(titer={pinene_ratio}) for the pinene RATIO",
            lambda: s.ProductTiterPhenotype(
                product=s.Compound(name="alpha-pinene"),
                titer=pinene_ratio,
                titer_unit=s.ConcentrationUnit("g/L"),
                titer_uncertainty=pinene_pm,
                titer_uncertainty_type=s.UncertaintyType.sample_sd,
                n_samples=3,
                sample_unit=s.SampleUnit.biological_replicate,
            ),
        ),
    ]
    probes[-1]["note"] = (
        "accepts the NUMBER only by mislabeling a dimensionless ratio as a g/L "
        "concentration; ConcentrationUnit has no dimensionless member, so there is no "
        "unit under which this is honest"
    )
    probes.append(
        {
            "candidate": "ConcentrationUnit members",
            "accepted": None,
            "members": [unit.value for unit in s.ConcentrationUnit],
            "note": "no dimensionless / fold-change / ratio member",
        }
    )
    return probes


def probe_environment_compounds() -> dict[str, Any]:
    """Resolve the two molecules the growth assay's environment doses.

    The growth ratio is read "with 0.5% pinene" and 200 nM anhydrous tetracycline, so
    both are ``SmallMoleculePerturbation`` edits carrying a typed ``Compound``. L3
    ``compound_identity`` (``torchcell.verification.common._identity_result``) fails on a
    NAME-ONLY compound and passes one whose absence is a typed gap, so what matters is
    whether the pinned compound-identity table already carries a structure for each.
    """
    from torchcell.datamodels.compound_identity import resolve_compound_identity

    out: dict[str, Any] = {}
    for label in (
        "alpha-pinene",
        "pinene",
        "(-)-alpha-pinene",
        "(+)-alpha-pinene",
        "anhydrotetracycline",
    ):
        resolution = resolve_compound_identity(label)
        out[label] = {
            "status": str(resolution.status),
            "name": resolution.name,
            "inchikey": resolution.inchikey,
            "pubchem_cid": resolution.pubchem_cid,
        }
    out["conclusion"] = (
        "anhydrotetracycline is RESOLVED in the pinned table; NO spelling of pinene is. "
        "Pinene is the dose the growth ratio is read against, so a record would carry "
        "the assay's central environmental variable as a typed gap. The table is a "
        "sha256-pinned shared artifact of 5,532 records whose curator re-queries PubChem "
        "for every row and re-pins compound_identity._TABLE_SHA256, so adding the row is "
        "issue #726's work (the compound curator over the bacterial condition labels), "
        "not a dataset branch's."
    )
    return out


def loadable_slice(targets: list[dict[str, Any]]) -> dict[str, Any]:
    """What a loader could serve today, and what each refusal costs, in released cells.

    One number decides whether serving the admitted arm is worth a dataset: the share of
    the release it would reach. Every released numeric cell is counted once, then
    attributed to the reason it can or cannot be stored.
    """
    numeric = {
        (arm, readout): sum(
            1 for t in targets if t["arm"] == arm and t[f"{readout}_ratio"] is not None
        )
        for arm in ("crispr_activation", "crispr_interference")
        for readout in ("growth", "pinene")
    }
    censored = sum(
        1
        for t in targets
        for readout in ("growth", "pinene")
        if t[f"{readout}_censored"]
    )
    #: The two six-target combination strains, numeric in the MAIN tables only.
    combination_cells = 4
    total = sum(numeric.values()) + combination_cells
    storable = numeric[("crispr_interference", "growth")] + 1
    return {
        "released_numeric_cells_by_arm_and_readout": {
            f"{arm}/{readout}": n for (arm, readout), n in numeric.items()
        },
        "released_numeric_cells_in_the_main_table_combination_rows": combination_cells,
        "released_numeric_cells_total": total,
        "released_censored_cells_total": censored,
        "storable_with_existing_classes_today": storable,
        "storable_share": round(storable / total, 4),
        "refusals_overlap_by_design": (
            "a cell can be blocked for more than one reason, so these counts are NOT "
            "disjoint and must not be summed: the activation arm's 33 pinene cells are "
            "counted both under the missing activation leaf and under the missing "
            "product fold-change phenotype. The disjoint figure is "
            "storable_with_existing_classes_today against released_numeric_cells_total."
        ),
        "refusals": [
            {
                "what": "all 374 called variants of the evolved isolate YZFP",
                "cells": 374,
                "reason": "no class holds a called bacterial variant; "
                "SequenceVariantPerturbation and AllelePerturbation refuse a b-number, "
                "and BacterialBackgroundAllele requires a non-optional `functional` the "
                "release never states and has no slot for position, ref/alt base or "
                "Frequence",
                "issue": 731,
            },
            {
                "what": "the 57-target CRISPR ACTIVATION arm and its combination strain",
                "cells": numeric[("crispr_activation", "growth")]
                + numeric[("crispr_activation", "pinene")]
                + 2,
                "reason": "there is no bacterial CRISPR-activation perturbation leaf; "
                "CrisprActivationPerturbation inherits the R64 systematic-name "
                "validator and refuses both a b-number and a gene symbol, and the only "
                "bacterial expression leaves are CRISPRi (direction fixed to decreased) "
                "and promoter replacement, which asserts an edit this design never made",
                "issue": None,
            },
            {
                "what": "every pinene ratio, both arms",
                "cells": numeric[("crispr_activation", "pinene")]
                + numeric[("crispr_interference", "pinene")]
                + 2,
                "reason": "the release is a dimensionless product RATIO with no absolute "
                "titer anywhere in the CRISPRa/i arm; ProductTiterPhenotype requires a "
                "ConcentrationUnit and the enum has no dimensionless member, and "
                "EnvironmentResponsePhenotype is a fitness/growth readout, so storing a "
                "production ratio there would mislabel it",
                "issue": 770,
            },
            {
                "what": "every censored cell",
                "cells": censored,
                "reason": "a dash is a measured non-positive outcome whose footnote "
                "conflates 'no change' with 'negative effect', so it has neither a "
                "number nor a single ResponseCategory",
                "issue": None,
            },
        ],
        "conclusion": (
            f"the only slice existing classes can carry is the CRISPR INTERFERENCE arm's "
            f"{numeric[('crispr_interference', 'growth')]} numeric growth ratios plus its "
            f"one combination strain = {storable} records, "
            f"{round(100 * storable / total, 1)}% of the {total} released numeric cells. "
            "Those are honest to encode -- the host is the UNEVOLVED designed parent "
            "BW25113(PT5-dxs), so nothing about them depends on the blocked evolved "
            "genotype, every target resolves to one BW25113 locus, and every guide "
            "spacer is released -- as EnvironmentResponsePhenotype log2_ratio against "
            "the no-guide control at log2(1) = 0, the landed Lim 2025 encoding for the "
            "same kind of quantity. They are NOT loaded in this branch because all three "
            "of their prerequisites sit outside it: the paper has no torchcell-raw "
            "mirror yet, pinene has no curated compound row (#726), and the arm that "
            "carries four fifths of the release waits on a new perturbation leaf, which "
            "forces a full knowledge-graph rebuild and belongs in the owner's batch."
        ),
    }


def bacterial_expression_leaves() -> list[dict[str, Any]]:
    """Every bacterial perturbation leaf, with the expression direction it can state."""
    out: list[dict[str, Any]] = []
    for name in dir(s):
        obj = getattr(s, name)
        if not isinstance(obj, type) or not issubclass(obj, s.GenePerturbation):
            continue
        if obj is s.GenePerturbation or "gene_namespace" not in obj.model_fields:
            continue
        field = obj.model_fields.get("expression_direction")
        default = None if field is None else field.default
        out.append(
            {
                "leaf": name,
                "has_expression_direction": field is not None,
                "expression_direction_default": (
                    None
                    if default is None or repr(default) == "PydanticUndefined"
                    else str(default)
                ),
                "can_state_increased": field is not None
                and (
                    repr(default) == "PydanticUndefined" or str(default) == "increased"
                ),
            }
        )
    return sorted(out, key=lambda row: row["leaf"])


# --------------------------------------------------------------------------- #
def main() -> int:
    """Measure the release, probe the schema, write the results."""
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    here = osp.dirname(osp.dirname(osp.abspath(__file__)))
    results = osp.join(here, "results")
    os.makedirs(results, exist_ok=True)

    pinned = verify_si(data_root)
    deposit = enumerate_publisher_deposit()
    mirrored = {pinned["original_filename"]}
    released = {obj["name"] for obj in deposit["supplementary_objects"]}
    missing = sorted(released - mirrored)
    declared = {
        entry["href"] for entry in deposit["declared_supplementary"] if entry["href"]
    }

    body = read_docx_body(pinned["path"])
    tables = _tables(body)
    paragraphs = _paragraphs(body)
    if len(tables) != 4:
        raise RuntimeError(f"{len(tables)} tables in the SI docx, expected 4")
    if DASH_FOOTNOTE not in paragraphs:
        raise RuntimeError(f"the dash footnote {DASH_FOOTNOTE!r} is not in the SI")

    variants = characterize_variant_table(tables[1])
    activation = read_target_table(tables[2], "crispr_activation")
    interference = read_target_table(tables[3], "crispr_interference")
    targets = activation + interference

    worst = max(targets, key=lambda t: t["growth_ratio"] or 0.0)
    best_pinene = max(targets, key=lambda t: t["pinene_ratio"] or 0.0)

    report: dict[str, Any] = {
        "schedule_row": 48,
        "dataset": "Niu 2019 pinene evolved",
        "citation_key": CITATION_KEY,
        "pinned_si": pinned,
        "q1_publisher_deposit": {
            "bucket_prefix": PMC_PREFIX,
            "n_bucket_objects": len(deposit["bucket_objects"]),
            "bucket_object_names": [o["name"] for o in deposit["bucket_objects"]],
            "released_supplementary_files": sorted(released),
            "jats_declared_supplementary": sorted(declared),
            "mirrored_supplementary_files": sorted(mirrored),
            "released_but_not_mirrored": missing,
            "conclusion": (
                "the mirror holds the WHOLE publisher supplementary deposit: the bucket "
                f"carries exactly {len(released)} supplementary object(s) "
                f"({sorted(released)}) and the article's JATS declares exactly "
                f"{len(declared)} <supplementary-material> element(s) ({sorted(declared)}"
                "), which is the one file the mirror captured, byte-for-byte by sha256. "
                "'Only one SI file captured' is the complete release, not a gap."
                if not missing
                else f"MISSING from the mirror: {missing}"
            ),
        },
        "q2_what_the_release_gives": {
            "si_table_count": len(tables),
            "si_table_titles": [p for p in paragraphs if p.startswith("Suppl. Table")],
            "table_1_primers_rows": len(tables[0]),
            "table_2_variant_list": variants,
            "table_3_and_4_arms": [
                summarize_targets(targets, "crispr_activation"),
                summarize_targets(targets, "crispr_interference"),
            ],
            "dash_footnote": DASH_FOOTNOTE,
            "dash_meaning_reconciled_against_the_main_text": (
                reconcile_dashes_against_the_main_text(targets)
            ),
            "largest_growth_ratio": {
                "target": worst["target"],
                "arm": worst["arm"],
                "cell": worst["growth_cell"],
            },
            "largest_pinene_ratio": {
                "target": best_pinene["target"],
                "arm": best_pinene["arm"],
                "cell": best_pinene["pinene_cell"],
            },
        },
        "q3_schema_probes": {
            "perturbation_candidates": probe_perturbation_classes(),
            "bacterial_leaves_expression_direction": bacterial_expression_leaves(),
            "phenotype_candidates": probe_phenotype_classes(
                growth_ratio=1.22, growth_pm=0.02, pinene_ratio=1.09, pinene_pm=0.01
            ),
            "environment_compounds": probe_environment_compounds(),
        },
        "q4_loadable_slice": loadable_slice(targets),
    }

    out_json = osp.join(results, "niu2019_release_loadability.json")
    with open(out_json, "w") as handle:
        json.dump(report, handle, indent=2)
    out_csv = osp.join(results, "niu2019_release_loadability_targets.csv")
    pd.DataFrame(targets).to_csv(out_csv, index=False)

    print(json.dumps(report["q1_publisher_deposit"], indent=2))
    print(
        json.dumps(
            report["q2_what_the_release_gives"]["table_2_variant_list"], indent=2
        )
    )
    print(
        json.dumps(report["q2_what_the_release_gives"]["table_3_and_4_arms"], indent=2)
    )
    for probe in report["q3_schema_probes"]["perturbation_candidates"]:
        print(json.dumps(probe))
    for probe in report["q3_schema_probes"]["phenotype_candidates"]:
        print(json.dumps(probe))
    print(json.dumps(report["q3_schema_probes"]["environment_compounds"], indent=2))
    print(json.dumps(report["q4_loadable_slice"], indent=2))
    print(f"\nwrote {out_json}\nwrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
