# experiments/036-dataset-fixes-before-kg-build/scripts/bacteria_subsumed_rows.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.bacteria_subsumed_rows]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/bacteria_subsumed_rows
"""Measure three bacterial schedule rows against the stores that already serve them.

Three rows of the fifty-row bacterial schedule
(``experiments/database/scripts/build_bacteria_candidate_datasets_table.py``) were
triaged as needing no loader of their own because a dataset we already serve holds
their data. That triage lived in agent reports and issue comments. This script is the
regenerable record: every count below is recomputed from sha256-pinned release files
and from the served LMDB stores, and nothing is cited from a report.

ROW 33, Butland 2008 (``butlandESGAColiSynthetic2008``), marked ``status="blocked"``
pending a paywalled retrieval. The retrieval blocker is gone: the paper and five SI
files are mirrored, so every count the row calls unverified is measured here. But the
triage's premise is WRONG, and that is this script's main finding. Supplementary Table
4 releases the complete unfiltered 39 x 8,073 interaction matrix, so the served Babu
2014 records carry a small fraction of the release rather than all of it.

ROW 24 Schmidt 2022 nitrogen and ROW 28 Thompson 2020 fatty acid and alcohol are
measured in the sections that follow (added per row, one commit each).

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/bacteria_subsumed_rows.py
    python experiments/036-dataset-fixes-before-kg-build/scripts/bacteria_subsumed_rows.py --write-records

``--write-records`` deposits the measured record as
``$DATA_ROOT/torchcell-raw/<citation_key>/subsumption_record.json`` and extends that
key's raw manifest with the release files the evidence reads. It never overwrites a
manifest entry another run wrote: an entry whose sha256 differs raises.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, model_validator

from torchcell.literature.manifest import (
    ROLE_RAW_DATA,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
    sha256_file,
)
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import SourcedValue, audit_sourced_value

RESULTS: Final = "experiments/036-dataset-fixes-before-kg-build/results"
SCRIPT: Final = (
    "experiments/036-dataset-fixes-before-kg-build/scripts/bacteria_subsumed_rows.py"
)
LIBRARY_REL: Final = "torchcell-library"
RAW_REL: Final = "torchcell-raw"
RECORD_NAME: Final = "subsumption_record.json"
_OCR: Final = "MinerU OCR of the publisher PDF (torchcell-library mirror)"
_SHEET: Final = "spreadsheet cell of the publisher's released workbook"


# --------------------------------------------------------------------------- #
# Typed pieces every row's record is built from
# --------------------------------------------------------------------------- #
class ReleaseFile(BaseModel):
    """One released file the evidence reads, with the retrieval that produced it.

    ``mirror`` says which tree the bytes live in: ``torchcell-library`` for the
    publisher's paper and SI (the literature mirror captures those), ``torchcell-raw``
    for a dataset release file a loader would consume. Both are sha256-pinned, and the
    sha256 is the canonical anchor, not the URL.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    relpath: str
    mirror: Literal["torchcell-library", "torchcell-raw"]
    citation_key: str
    role: str
    bytes: int
    sha256: str
    source_url: str
    retrieval_method: RetrievalMethod
    retrieval_command: str
    retrieved_at: str
    purpose: str


class ReleaseProbe(BaseModel):
    """The measured HTTP status of a release endpoint that holds no mirrored file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str
    status_code: int
    probed_at: str
    note: str


class ServedCoverage(BaseModel):
    """What a served store holds of one paper's release, measured record by record.

    ``released_instances_basis`` names what the denominator IS, because the fraction
    means nothing without it: a paper that released a full matrix has an independent
    release count to divide by, and a paper whose only release is a web browser does
    not, so its fraction is 1.0 by construction and must say so.

    ``served_fraction`` is stored rather than derived on read, so a record read years
    from now states the fraction it was written with instead of recomputing it against
    whatever the two counts have become. ``model_validator`` fills it from the counts,
    so the three can never disagree.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    served_dataset: str
    served_store: str
    served_records: int
    records_from_this_paper: int
    released_instances: int
    released_instances_basis: str
    attribution_field: str
    attribution_value: str
    served_fraction: float = -1.0

    @model_validator(mode="after")
    def _fill_served_fraction(self) -> ServedCoverage:
        """Set the fraction from the two counts it is the ratio of."""
        ratio = self.records_from_this_paper / self.released_instances
        if self.served_fraction != ratio:
            object.__setattr__(self, "served_fraction", ratio)
        return self


class ButlandRelease(BaseModel):
    """Butland 2008's own release, counted from the four mirrored SI workbooks."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    array_rows: int
    array_non_essential_strains: int
    array_isolate_1: int
    array_isolate_2: int
    array_spa_tag_essential: int
    array_distinct_recipient_genes: int
    query_strains: int
    matrix_sheets: tuple[str, ...]
    matrix_query_columns: int
    matrix_recipient_rows: int
    matrix_cells: int
    matrix_populated_cells: int
    matrix_distinct_gene_pairs: int
    high_confidence_rows: int
    high_confidence_ordered_pairs: int
    high_confidence_unordered_pairs: int
    high_confidence_non_essential_pairs: int
    high_confidence_non_essential_aggravating: int
    high_confidence_non_essential_alleviating: int
    high_confidence_spa_tag_pairs: int


class ButlandVersusBabu(BaseModel):
    """How much of Butland's release the Babu 2014 store carries, measured both ways."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    babu_donors_attributed_to_butland: int
    babu_tested_pairs_over_those_donors: int
    babu_released_rows_over_those_donors: int
    babu_released_aggravating: int
    babu_released_alleviating: int
    served_butland_records: int
    served_butland_unordered_pairs: int
    served_butland_donors: int
    butland_query_genes_with_no_served_record: tuple[str, ...]
    hc_pairs_in_babu_release: int
    hc_pairs_absent_from_babu_release: int
    hc_non_essential_pairs_absent: int
    hc_spa_tag_pairs_absent: int
    hc_pairs_in_served_store: int
    hc_pairs_served_under_butland_tag: int
    hc_rows_matched_same_orientation: int
    hc_rows_with_identical_score: int
    hc_rows_with_rescored_value: int
    hc_rows_max_abs_score_difference: float
    hc_rows_matched_only_reversed: int


class SubsumptionRecord(BaseModel):
    """Why a schedule row needs no loader, or how much of it still does.

    The provenance-record outcome the Wetmore 2015 row established: no dataset class,
    a sha256-pinned mirror of the release files the evidence reads, the sourced values
    the row's counts come from, and a measurement naming the served dataset that holds
    the data. ``decision`` is the only field a reader needs to act on; everything else
    is the evidence for it.
    """

    model_config = ConfigDict(extra="forbid")

    citation_key: str
    doi: str
    title: str
    row_name: str
    row_rank: int
    decision: Literal[
        "subsumed_no_loader", "partly_subsumed_loader_warranted", "not_subsumed"
    ]
    served_coverage: ServedCoverage
    release_files: tuple[ReleaseFile, ...]
    release_probes: tuple[ReleaseProbe, ...] = ()
    sourced_values: dict[str, SourcedValue]
    conclusion: str
    loadable_slice: str
    measured_at: str
    script: str = SCRIPT


class ButlandRecord(SubsumptionRecord):
    """Butland 2008's record: its release, and what Babu 2014 serves of it."""

    release: ButlandRelease
    versus_superset: ButlandVersusBabu


# --------------------------------------------------------------------------- #
# Mirror paths
# --------------------------------------------------------------------------- #
def data_root() -> Path:
    """``DATA_ROOT`` from the environment, which both mirrors hang off."""
    return Path(os.environ["DATA_ROOT"])


def mirror_path(release: ReleaseFile) -> Path:
    """The absolute path of one release file in its mirror."""
    tree = LIBRARY_REL if release.mirror == "torchcell-library" else RAW_REL
    return data_root() / tree / release.citation_key / release.relpath


def verified(release: ReleaseFile) -> Path:
    """The release file's path after checking its bytes against the pinned sha256."""
    path = mirror_path(release)
    got = sha256_file(path)
    if got != release.sha256:
        raise RuntimeError(f"{path} sha256 {got}, pinned {release.sha256}")
    return path


def audit_quotes(record: SubsumptionRecord) -> dict[str, str]:
    """Re-open every text-sourced quote of a record and confirm it is still verbatim.

    Audits the OCR markdown only. A quote taken from a spreadsheet cell is marked
    ``method`` ``spreadsheet cell ...``; its bytes are pinned by the same sha256 the
    measurement reads, but ``audit_sourced_value`` reads the artifact as text and an
    ``.xls`` is not text. A failed audit raises: a quote that no longer reads verbatim
    means the record is wrong, not that the check should be skipped.
    """
    root = data_root() / LIBRARY_REL
    audited: dict[str, str] = {}
    for name, value in record.sourced_values.items():
        if not value.provenance.source_uri.endswith(".md"):
            audited[name] = (
                f"checked against the pinned {value.provenance.method} by the "
                "row's own measurement, not by the text audit"
            )
            continue
        result = audit_sourced_value(value, root)
        if not result.passed:
            raise RuntimeError(f"{name}: {result.message}")
        audited[name] = result.message
    return audited


# --------------------------------------------------------------------------- #
# Butland 2008: provenance anchors
# --------------------------------------------------------------------------- #
BUTLAND_KEY: Final = "butlandESGAColiSynthetic2008"
BUTLAND_DOI: Final = "10.1038/nmeth.1239"
BUTLAND_TITLE: Final = "eSGA: E. coli synthetic genetic array analysis"
BABU_KEY: Final = "babuQuantitativeGenomeWideGenetic2014"
BABU_DOI: Final = "10.1371/journal.pgen.1004120"
_ESM: Final = (
    "https://static-content.springer.com/esm/art%3A10.1038%2Fnmeth.1239/MediaObjects"
)
_PMC: Final = "https://pmc-oa-opendata.s3.amazonaws.com/PMC3930520.1"
BUTLAND_RETRIEVED_AT: Final = "2026-10-07T09:58:24.878215+00:00"
BABU_RETRIEVED_AT: Final = "2026-10-07T09:17:40.083033+00:00"


def _springer(
    relpath: str, *, moesm: int, role: str, size: int, sha256: str, purpose: str
) -> ReleaseFile:
    """A Butland SI file as the Springer ESM retriever fetched it."""
    url = f"{_ESM}/41592_2008_BFnmeth1239_MOESM{moesm}_ESM.{relpath.rsplit('.', 1)[1]}"
    return ReleaseFile(
        relpath=relpath,
        mirror="torchcell-library",
        citation_key=BUTLAND_KEY,
        role=role,
        bytes=size,
        sha256=sha256,
        source_url=url,
        retrieval_method=RetrievalMethod.springer_esm,
        retrieval_command=(f"torchcell.literature.retrieve.springer_esm(url={url!r})"),
        retrieved_at=BUTLAND_RETRIEVED_AT,
        purpose=purpose,
    )


BUTLAND_PAPER_MD: Final = ReleaseFile(
    relpath="paper.md",
    mirror="torchcell-library",
    citation_key=BUTLAND_KEY,
    role="paper_ocr",
    bytes=36906,
    sha256="a3da20a90b56b85e1cd7e23e784c318e78cec115b5bc3e94afcd89a5f94f8eef",
    source_url="https://doi.org/10.1038/nmeth.1239",
    retrieval_method=RetrievalMethod.zotero_attachment,
    retrieval_command=(
        "torchcell.literature.ocr.ocr_pdf(paper.pdf) over the Zotero attachment "
        "captured by torchcell.literature.capture"
    ),
    retrieved_at=BUTLAND_RETRIEVED_AT,
    purpose="the paper's own statement of the array, the 39 screens and the 1,288 "
    "high-confidence interactions",
)
BUTLAND_SI1_MD: Final = ReleaseFile(
    relpath="si/si1.md",
    mirror="torchcell-library",
    citation_key=BUTLAND_KEY,
    role="si_ocr",
    bytes=64216,
    sha256="a5e1648bea298f26ab23cb80de96c1c68f7aa7d5d6420aa56608991bc65aa278",
    source_url=f"{_ESM}/41592_2008_BFnmeth1239_MOESM305_ESM.pdf",
    retrieval_command=(
        "torchcell.literature.ocr.ocr_pdf(si/si1.pdf) over the springer_esm capture"
    ),
    retrieval_method=RetrievalMethod.springer_esm,
    retrieved_at=BUTLAND_RETRIEVED_AT,
    purpose="Supplementary Methods: the batch scoring of the 39 screens",
)
BUTLAND_TABLE_S1: Final = _springer(
    "si/si2.xls",
    moesm=306,
    role="si_data",
    size=717824,
    sha256="7e088873d486307183fbc0eff98b2a29b452c3e3abe8a636979966fd7b49deea",
    purpose="Supplementary Table 1: the recipient array, one row per strain with its "
    "b-number and its non-essential / SPA-tag essential label",
)
BUTLAND_TABLE_S2: Final = _springer(
    "si/si3.xls",
    moesm=307,
    role="si_data",
    size=32768,
    sha256="67886f46c2b6561d40f9c7fc24b56b6e6cfdf199d1c13cfa4acfa59fd75d9ef1",
    purpose="Supplementary Table 2: the query deletion strains of the 39 screens",
)
BUTLAND_TABLE_S3: Final = _springer(
    "si/si4.xls",
    moesm=308,
    role="si_data",
    size=290816,
    sha256="a2ecd1071d665af77d4f7ae32c10b335b1f5ed8616de0ceb95f0f53c9ad24024",
    purpose="Supplementary Table 3: the high-confidence interactions with their "
    "S-score, |Z score| and Log2(Q/R)",
)
BUTLAND_TABLE_S4: Final = _springer(
    "si/si5.xls",
    moesm=309,
    role="si_data",
    size=36617216,
    sha256="74a6ea3a0373fa6e1776b0becb4212f29b9876647b96327db5222dcd68a8822b",
    purpose="Supplementary Table 4: the unfiltered 39-screen matrix in four layers, "
    "raw colony sizes, normalized median colony sizes, |Z scores| and S scores",
)
BABU_TABLE_S1: Final = ReleaseFile(
    relpath="data/si22.xls",
    mirror="torchcell-raw",
    citation_key=BABU_KEY,
    role=ROLE_RAW_DATA,
    bytes=63488,
    sha256="7506c2edc9975d0e331d152e47fa17d4ef7203999b5a8f6a6cd136ef76f7dda7",
    source_url=f"{_PMC}/pgen.1004120.s022.xls",
    retrieval_method=RetrievalMethod.pmc_cloud,
    retrieval_command=(
        f"torchcell.literature.retrieve.pmc_cloud(url='{_PMC}/pgen.1004120.s022.xls')"
    ),
    retrieved_at=BABU_RETRIEVED_AT,
    purpose="Babu 2014 Table S1: which screen set each donor came from, and how many "
    "pairs that donor was tested against",
)
BABU_TABLE_S2: Final = ReleaseFile(
    relpath="data/si23.xls",
    mirror="torchcell-raw",
    citation_key=BABU_KEY,
    role=ROLE_RAW_DATA,
    bytes=3015168,
    sha256="0789563ada0db3e349bb7e5396311f9d705ea165090733803c009acfa7c95b96",
    source_url=f"{_PMC}/pgen.1004120.s023.xls",
    retrieval_method=RetrievalMethod.pmc_cloud,
    retrieval_command=(
        f"torchcell.literature.retrieve.pmc_cloud(url='{_PMC}/pgen.1004120.s023.xls')"
    ),
    retrieved_at=BABU_RETRIEVED_AT,
    purpose="Babu 2014 Table S2: the released high-confidence pairs the served "
    "GeneInteractionBabu2014Dataset is built from",
)
BUTLAND_RELEASE_FILES: Final = (
    BUTLAND_PAPER_MD,
    BUTLAND_SI1_MD,
    BUTLAND_TABLE_S1,
    BUTLAND_TABLE_S2,
    BUTLAND_TABLE_S3,
    BUTLAND_TABLE_S4,
    BABU_TABLE_S1,
    BABU_TABLE_S2,
)
#: The files added to the Butland raw mirror by ``--write-records``. ``si/si2.xls`` is
#: already there under ``data/si2.xls``; the other three workbooks are the rest of the
#: evidence, so the record's inputs are pinned in one tree.
BUTLAND_RAW_DEPOSIT: Final = (
    ("data/si2.xls", BUTLAND_TABLE_S1),
    ("data/si3.xls", BUTLAND_TABLE_S2),
    ("data/si4.xls", BUTLAND_TABLE_S3),
    ("data/si5.xls", BUTLAND_TABLE_S4),
)


def _quote(
    value: Any,
    quote: str,
    *,
    source: str,
    citation_key: str,
    sha256: str,
    page: str,
    method: str,
) -> SourcedValue:
    """Bind a value to a verbatim quote of one pinned artifact."""
    return SourcedValue(
        value=value,
        quote=quote,
        provenance=Provenance(
            source_uri=source,
            citation_key=citation_key,
            sha256=sha256,
            method=method,
            page=page,
        ),
    )


BUTLAND_SOURCED: Final[dict[str, SourcedValue]] = {
    "recipient_array": _quote(
        {"non_essential_genes": 3968, "isolate_2": 3956, "strains": 7924},
        "The recipient mutant strains (Supplementary Table 1 online) arrayed in "
        "twenty-five 384-well plates included the Keio single gene deletion mutant "
        "strain collection8 covering 3,968 nonessential single gene replacements "
        "marked with a kanamycin-resistance cassette (kan). Of these, 3,956 were "
        "represented by two independent single gene deletion isolates, for a total of "
        "7,924 single gene deletion mutants8.",
        source="paper.md",
        citation_key=BUTLAND_KEY,
        sha256=BUTLAND_PAPER_MD.sha256,
        page="Results, 'Global profiling with 39 genome-wide screens'",
        method=_OCR,
    ),
    "spa_tag_essential_recipients": _quote(
        149,
        "Number of Keio deletion strains = 7924\nSPA-tag essential genes = 149",
        source="si/si5.xls",
        citation_key=BUTLAND_KEY,
        sha256=BUTLAND_TABLE_S4.sha256,
        page="Supplementary Table 4, every sheet, header cell A3",
        method=_SHEET,
    ),
    "screens": _quote(
        39,
        "Thirty-nine $E$ . coli genome-wide screens are processed in batch scoring "
        "mode using colony scorer program4.",
        source="si/si1.md",
        citation_key=BUTLAND_KEY,
        sha256=BUTLAND_SI1_MD.sha256,
        page="Supplementary Methods, colony scoring",
        method=_OCR,
    ),
    "high_confidence_set": _quote(
        {"pairs": 1288, "non_essential": 799, "aggravating": 730, "alleviating": 69},
        "we reclustered the profiles after accounting for linkage (30-kbp window) and "
        "used a stringent |Z score| cut-off of $\\geq 4$ $P < 0 . 0 0 0 1$ to create a "
        "high-confidence dataset of 1,288 genetic interactions (Fig. 4b and "
        "Supplementary Table 3 online). Among these, we detected 799 genetic "
        "interactions for nonessential genes, with 730 categorized as aggravating "
        "interactions and only 69 as alleviating interactions.",
        source="paper.md",
        citation_key=BUTLAND_KEY,
        sha256=BUTLAND_PAPER_MD.sha256,
        page="Results, clustering and the high-confidence dataset",
        method=_OCR,
    ),
    "unfiltered_matrix": _quote(
        {"sheets": 4, "screens": 39},
        "Supplementary Table 4. Raw colony sizes (see Sheet 1), normalized median "
        "colony sizes (see Sheet 2), |Z scores| (see Sheet 3) and interaction (S) "
        "scores (see Sheet 4) of each mutant gene pair from 39 genome-wide screens "
        "without any filtering parameters.",
        source="si/si5.xls",
        citation_key=BUTLAND_KEY,
        sha256=BUTLAND_TABLE_S4.sha256,
        page="Supplementary Table 4, every sheet, title cells A1 and A2 joined by one space",
        method=_SHEET,
    ),
    "superset_combines_these_screens": _quote(
        39,
        "combined with our previously published GI datasets from 39 genome-wide "
        "screens to generate a comprehensive GI network.",
        source="si/si7.md",
        citation_key=BABU_KEY,
        sha256="b060ce3f5277f708675e7f26239d5ee2d357ffb2306a04da326417caad135db9",
        page="Babu 2014 Protocol S2 (pgen.1004120.s007.pdf)",
        method=_OCR,
    ),
}


# --------------------------------------------------------------------------- #
# Butland 2008: the release, counted from the mirrored workbooks
# --------------------------------------------------------------------------- #
_ARRAY_LABELS: Final = ("Non-essential", "SPA-tag essential")
_S3_COLUMNS: Final = (
    "query",
    "query_b",
    "recipient",
    "recipient_b",
    "version",
    "essentiality",
    "string",
    "s_score",
    "abs_z",
    "log2_qr",
)


def _array_rows(path: Path, sheet: str) -> pd.DataFrame:
    """The recipient-array rows of a Butland workbook whose layout starts at row 7.

    Both Supplementary Table 1 and Supplementary Table 4 carry the same four label
    columns and the same banner rows. A row is data when its first cell is one of the
    two essentiality labels, which also excludes the footnote block at the bottom.
    """
    frame = pd.read_excel(path, sheet_name=sheet, header=None)
    body = frame.iloc[6:, :]
    return body[body.iloc[:, 0].astype(str).isin(_ARRAY_LABELS)]


def high_confidence_table() -> pd.DataFrame:
    """Supplementary Table 3's data rows, typed, with the banner rows dropped."""
    frame = pd.read_excel(verified(BUTLAND_TABLE_S3), sheet_name="st3", header=2)
    frame.columns = list(_S3_COLUMNS)
    b_number = r"^b\d{4}$"
    keep = frame["query_b"].astype(str).str.match(b_number) & frame[
        "recipient_b"
    ].astype(str).str.match(b_number)
    data = frame[keep].copy()
    for column in ("s_score", "abs_z", "log2_qr"):
        data[column] = data[column].astype(float)
    return data


def _check_cell_quote(frame: pd.DataFrame, rows: tuple[int, ...], name: str) -> None:
    """Confirm a spreadsheet-sourced quote still equals the cells it was taken from.

    The text audit cannot read an ``.xls`` as text, so the two sourced values that
    quote a workbook banner cell are checked here instead, against the same pinned
    bytes the counts are measured from. A banner sentence that the workbook splits
    over two cells is joined by one space, which is what ``page`` records.
    """
    cells = " ".join(str(frame.iloc[row, 0]).strip() for row in rows)
    quote = BUTLAND_SOURCED[name].quote.strip()
    if cells != quote:
        raise RuntimeError(f"{name}: cells read {cells!r}, quote says {quote!r}")


def butland_release() -> tuple[ButlandRelease, pd.DataFrame]:
    """Count Butland 2008's release from its four mirrored workbooks."""
    array = _array_rows(verified(BUTLAND_TABLE_S1), "st1")
    labels = array.iloc[:, 0].value_counts()
    isolates = array.iloc[:, 3].value_counts()
    queries = pd.read_excel(
        verified(BUTLAND_TABLE_S2), sheet_name="donor_st2", header=None
    )
    query_rows = queries.iloc[2:, :]
    query_b = query_rows[query_rows.iloc[:, 1].astype(str).str.match(r"^b\d{4}$")]

    matrix_path = verified(BUTLAND_TABLE_S4)
    sheets = tuple(str(name) for name in pd.ExcelFile(matrix_path).sheet_names)
    populated = 0
    columns = 0
    rows = 0
    for sheet in sheets:
        frame = pd.read_excel(matrix_path, sheet_name=sheet, header=None)
        _check_cell_quote(frame, (0, 1), "unfiltered_matrix")
        _check_cell_quote(frame, (2,), "spa_tag_essential_recipients")
        body = frame.iloc[6:, :]
        data = body[body.iloc[:, 0].astype(str).isin(_ARRAY_LABELS)]
        values = data.iloc[:, 4:]
        rows = len(data)
        columns = values.shape[1]
        populated += int(values.notna().sum().sum())

    table_s3 = high_confidence_table()
    ordered = table_s3.drop_duplicates(["query_b", "recipient_b"])
    non_essential = ordered[ordered["essentiality"] == "non-essential"]
    release = ButlandRelease(
        array_rows=len(array),
        array_non_essential_strains=int(labels["Non-essential"]),
        array_isolate_1=int(isolates["Isolate 1"]),
        array_isolate_2=int(isolates["Isolate 2"]),
        array_spa_tag_essential=int(labels["SPA-tag essential"]),
        array_distinct_recipient_genes=int(array.iloc[:, 2].nunique()),
        query_strains=int(query_b.iloc[:, 1].nunique()),
        matrix_sheets=sheets,
        matrix_query_columns=columns,
        matrix_recipient_rows=rows,
        matrix_cells=rows * columns,
        matrix_populated_cells=populated // len(sheets),
        matrix_distinct_gene_pairs=columns * int(array.iloc[:, 2].nunique()),
        high_confidence_rows=len(table_s3),
        high_confidence_ordered_pairs=len(ordered),
        high_confidence_unordered_pairs=len(
            {frozenset(pair) for pair in zip(table_s3.query_b, table_s3.recipient_b)}
        ),
        high_confidence_non_essential_pairs=len(non_essential),
        high_confidence_non_essential_aggravating=int(
            (non_essential["s_score"] < 0).sum()
        ),
        high_confidence_non_essential_alleviating=int(
            (non_essential["s_score"] > 0).sum()
        ),
        high_confidence_spa_tag_pairs=len(
            ordered[ordered["essentiality"] == "SPA-tag essential"]
        ),
    )
    return release, table_s3


# --------------------------------------------------------------------------- #
# Butland 2008: what the served Babu 2014 store holds of it
# --------------------------------------------------------------------------- #
_BABU_S1_COLUMNS: Final = (
    "b_number",
    "gene",
    "jw_number",
    "ppi_degree",
    "rank",
    "gis_tested",
    "significant_gis",
    "significant_aggravating",
    "significant_alleviating",
    "functional_category",
    "screen_set",
    "chaperone_family",
    "essentiality",
    "description",
)
BUTLAND_SCREEN_TAG: Final = "Butland et al."
BABU_SLUG: Final = "data/torchcell/gene_interaction_babu2014"


def babu_donor_catalog() -> pd.DataFrame:
    """Babu 2014 Table S1's donor rows, typed, with the banner rows dropped."""
    frame = pd.read_excel(verified(BABU_TABLE_S1), header=2)
    frame.columns = list(_BABU_S1_COLUMNS)
    return frame.dropna(subset=["b_number"])


def babu_released_pairs() -> pd.DataFrame:
    """Babu 2014 Table S2's released pairs with the donor and recipient b-numbers."""
    frame = pd.read_excel(
        verified(BABU_TABLE_S2), sheet_name="WG_GI_Score_Mar_06_2013", header=2
    )
    frame.columns = ["donor", "recipient", "gi_score"]
    frame = frame.dropna(subset=["donor", "recipient"])
    frame["donor_b"] = frame["donor"].str.split("__").str[0]
    frame["recipient_b"] = frame["recipient"].str.split("__").str[0]
    frame["gi_score"] = frame["gi_score"].astype(float)
    return frame


class ServedBabu(BaseModel):
    """The served Babu 2014 store, read once: records per screen set, and its pairs.

    ``pairs`` is keyed by the unordered gene pair and ``records`` is the raw record
    count, because the two differ: the store holds 97 reciprocal pairs, measured in
    both donor-recipient orientations, so a pair-keyed map undercounts records.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    records: int
    records_per_screen: dict[str, int]
    donors_per_screen: dict[str, tuple[str, ...]]
    pairs: dict[frozenset[str], tuple[str, float]]


def served_babu() -> ServedBabu:
    """Read the served Babu 2014 store once, then close the LMDB handle.

    The handle is closed before returning, because a held handle makes the next reader
    of this store fail with "already open in this process", and that only surfaces in
    the full-suite run.
    """
    from torchcell.datasets.ecoli.babu2014 import GeneInteractionBabu2014Dataset

    dataset = GeneInteractionBabu2014Dataset(
        root=osp.join(os.environ["DATA_ROOT"], BABU_SLUG)
    )
    pairs: dict[frozenset[str], tuple[str, float]] = {}
    per_screen: Counter[str] = Counter()
    donors: dict[str, set[str]] = {}
    for index in range(len(dataset)):
        experiment = dataset[index]["experiment"]
        leaves = experiment["genotype"]["perturbations"]
        donor = next(
            leaf["systematic_gene_name"]
            for leaf in leaves
            if leaf["collection"].startswith("Hfr")
        )
        screen = experiment["phenotype"]["screen_id"]
        per_screen[screen] += 1
        donors.setdefault(screen, set()).add(donor)
        pairs[frozenset(leaf["systematic_gene_name"] for leaf in leaves)] = (
            screen,
            float(experiment["phenotype"]["gene_interaction"]),
        )
    total = len(dataset)
    dataset.close_lmdb()
    return ServedBabu(
        records=total,
        records_per_screen=dict(per_screen),
        donors_per_screen={
            screen: tuple(sorted(names)) for screen, names in donors.items()
        },
        pairs=pairs,
    )


def butland_versus_babu(
    release: ButlandRelease, table_s3: pd.DataFrame
) -> tuple[ButlandVersusBabu, ServedCoverage, pd.DataFrame]:
    """Measure Butland's release against Babu's release and against the served store."""
    catalog = babu_donor_catalog()
    attributed = catalog[catalog["screen_set"] == BUTLAND_SCREEN_TAG]
    babu_pairs = babu_released_pairs()
    babu_unordered = {
        frozenset(pair) for pair in zip(babu_pairs.donor_b, babu_pairs.recipient_b)
    }
    store = served_babu()
    served = store.pairs
    served_butland = {
        pair for pair, (screen, _) in served.items() if screen == BUTLAND_SCREEN_TAG
    }
    served_butland_donors = set(store.donors_per_screen[BUTLAND_SCREEN_TAG])

    hc_unordered = {
        frozenset(pair): essentiality
        for pair, essentiality in zip(
            zip(table_s3.query_b, table_s3.recipient_b), table_s3.essentiality
        )
    }
    absent = {pair for pair in hc_unordered if pair not in babu_unordered}
    same_orientation = table_s3.merge(
        babu_pairs,
        left_on=["query_b", "recipient_b"],
        right_on=["donor_b", "recipient_b"],
    )
    identical = same_orientation["s_score"] == same_orientation["gi_score"]
    reversed_only = table_s3.merge(
        babu_pairs,
        left_on=["query_b", "recipient_b"],
        right_on=["recipient_b", "donor_b"],
        suffixes=("", "_babu"),
    )

    evidence = ButlandVersusBabu(
        babu_donors_attributed_to_butland=len(attributed),
        babu_tested_pairs_over_those_donors=int(attributed["gis_tested"].sum()),
        babu_released_rows_over_those_donors=int(
            babu_pairs["donor_b"].isin(set(attributed["b_number"])).sum()
        ),
        babu_released_aggravating=int(attributed["significant_aggravating"].sum()),
        babu_released_alleviating=int(attributed["significant_alleviating"].sum()),
        served_butland_records=store.records_per_screen[BUTLAND_SCREEN_TAG],
        served_butland_unordered_pairs=len(served_butland),
        served_butland_donors=len(served_butland_donors),
        butland_query_genes_with_no_served_record=tuple(
            sorted(set(table_s3.query_b) - served_butland_donors)
        ),
        hc_pairs_in_babu_release=len(set(hc_unordered) & babu_unordered),
        hc_pairs_absent_from_babu_release=len(absent),
        hc_non_essential_pairs_absent=sum(
            1 for pair in absent if hc_unordered[pair] == "non-essential"
        ),
        hc_spa_tag_pairs_absent=sum(
            1 for pair in absent if hc_unordered[pair] == "SPA-tag essential"
        ),
        hc_pairs_in_served_store=len(set(hc_unordered) & set(served)),
        hc_pairs_served_under_butland_tag=len(set(hc_unordered) & served_butland),
        hc_rows_matched_same_orientation=len(same_orientation),
        hc_rows_with_identical_score=int(identical.sum()),
        hc_rows_with_rescored_value=int((~identical).sum()),
        hc_rows_max_abs_score_difference=float(
            (same_orientation["s_score"] - same_orientation["gi_score"]).abs().max()
        ),
        hc_rows_matched_only_reversed=len(reversed_only),
    )
    coverage = ServedCoverage(
        served_dataset="GeneInteractionBabu2014Dataset",
        served_store=f"$DATA_ROOT/{BABU_SLUG}",
        served_records=store.records,
        records_from_this_paper=evidence.served_butland_records,
        released_instances=release.matrix_cells,
        released_instances_basis="the S scores of Supplementary Table 4, one per "
        "(query strain, recipient strain), which the workbook releases unfiltered",
        attribution_field="GeneInteractionPhenotype.screen_id",
        attribution_value=BUTLAND_SCREEN_TAG,
    )

    per_pair = table_s3.copy()
    babu_by_pair = {
        frozenset(pair): score
        for pair, score in zip(
            zip(babu_pairs.donor_b, babu_pairs.recipient_b), babu_pairs.gi_score
        )
    }
    keys = [frozenset(pair) for pair in zip(per_pair.query_b, per_pair.recipient_b)]
    per_pair["in_babu_release"] = [key in babu_by_pair for key in keys]
    per_pair["babu_gi_score"] = [babu_by_pair.get(key) for key in keys]
    per_pair["in_served_store"] = [key in served for key in keys]
    per_pair["served_screen_id"] = [
        served[key][0] if key in served else None for key in keys
    ]
    per_pair["served_gi_score"] = [
        served[key][1] if key in served else None for key in keys
    ]
    return evidence, coverage, per_pair


def butland_record() -> tuple[ButlandRecord, pd.DataFrame]:
    """Butland 2008's measured provenance record, and the per-pair table behind it."""
    release, table_s3 = butland_release()
    evidence, coverage, per_pair = butland_versus_babu(release, table_s3)
    loadable = (
        f"{evidence.hc_non_essential_pairs_absent} of the "
        f"{release.high_confidence_non_essential_pairs} non-essential high-confidence "
        "pairs are absent from Babu's released Table S2 altogether, and the unfiltered "
        f"Supplementary Table 4 matrix holds {release.matrix_cells} S scores against "
        f"{coverage.records_from_this_paper} served records. Both slices load against "
        "the existing GeneInteractionPhenotype with no schema change, because Babu's "
        "own GI score IS this S score for the pairs the two releases share "
        f"({evidence.hc_rows_with_identical_score} of "
        f"{evidence.hc_rows_matched_same_orientation} matched rows are identical). The "
        f"{release.high_confidence_spa_tag_pairs} SPA-tag essential pairs stay blocked "
        "on the hypomorph perturbation leaf the Babu loader already filed."
    )
    conclusion = (
        f"NOT SUBSUMED, and the retrieval blocker is resolved. The served "
        f"{coverage.served_dataset} carries {coverage.records_from_this_paper} of this "
        f"paper's measurements under {coverage.attribution_field} == "
        f"{coverage.attribution_value!r}, which is "
        f"{coverage.served_fraction * 100:.2f} percent of the "
        f"{release.matrix_cells} interaction scores Supplementary Table 4 releases "
        f"without any filtering, over {release.matrix_distinct_gene_pairs} distinct "
        "gene pairs once the two Keio isolates of a gene are collapsed. Babu's release "
        "is the high-confidence tail of a re-analysis, not a superset of this one: it "
        "re-publishes "
        f"{evidence.babu_released_rows_over_those_donors} rows over these "
        f"{evidence.babu_donors_attributed_to_butland} donors, and leaves "
        f"{evidence.hc_pairs_absent_from_babu_release} of this paper's own "
        f"{release.high_confidence_unordered_pairs} high-confidence gene pairs out."
    )
    return (
        ButlandRecord(
            citation_key=BUTLAND_KEY,
            doi=BUTLAND_DOI,
            title=BUTLAND_TITLE,
            row_name="Butland 2008",
            row_rank=33,
            decision="partly_subsumed_loader_warranted",
            served_coverage=coverage,
            release_files=BUTLAND_RELEASE_FILES,
            sourced_values=BUTLAND_SOURCED,
            release=release,
            versus_superset=evidence,
            conclusion=conclusion,
            loadable_slice=loadable,
            measured_at=datetime.now(UTC).isoformat(),
        ),
        per_pair,
    )


# --------------------------------------------------------------------------- #
# Depositing a record into the raw mirror
# --------------------------------------------------------------------------- #
def deposit_release_files(
    citation_key: str, deposit: tuple[tuple[str, ReleaseFile], ...]
) -> list[ArtifactRecord]:
    """Copy the evidence files into the raw mirror and return their manifest records.

    Every source file is checked against its pin before anything is written, and a
    mirror file that already exists with a different sha256 raises rather than being
    replaced.
    """
    root = data_root() / RAW_REL / citation_key
    records: list[ArtifactRecord] = []
    for relpath, release in deposit:
        source = verified(release)
        target = root / relpath
        if target.exists() and sha256_file(target) != release.sha256:
            raise RuntimeError(f"{target} exists with another sha256; refusing")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        records.append(
            ArtifactRecord(
                path=relpath,
                role=ROLE_RAW_DATA,
                bytes=target.stat().st_size,
                sha256=release.sha256,
                source=release.source_url,
                retrieval=RetrievalRecord(
                    method=release.retrieval_method,
                    source_url=release.source_url,
                    retriever=release.retrieval_command.split("(")[0],
                    params={"url": release.source_url},
                    sha256=release.sha256,
                    retrieved_at=release.retrieved_at,
                ),
            )
        )
    return records


def extend_manifest(
    citation_key: str,
    records: list[ArtifactRecord],
    *,
    doi: str,
    title: str,
    si_expected: list[str],
) -> Path:
    """Add the deposited files to the key's raw manifest, keeping what is already there.

    An entry the manifest already holds is kept byte-identical; one that disagrees on
    sha256 raises. The manifest is never rewritten from scratch, so a record another
    run deposited survives.
    """
    root = data_root() / RAW_REL / citation_key
    path = root / "manifest.json"
    manifest = (
        Manifest.model_validate_json(path.read_text())
        if path.exists()
        else Manifest(citation_key=citation_key, doi=doi, title=title)
    )
    held = {record.path: record for record in manifest.files}
    for record in records:
        existing = held.get(record.path)
        if existing is None:
            manifest.files.append(record)
            continue
        if existing.sha256 != record.sha256:
            raise RuntimeError(
                f"{path} holds {record.path} at {existing.sha256}, measured "
                f"{record.sha256}; refusing"
            )
    manifest.files.sort(key=lambda record: record.path)
    for url in (record.source for record in records):
        if url is not None and url not in manifest.si_data_sources:
            manifest.si_data_sources.append(url)
    manifest.si_expected = si_expected
    manifest.created_at = manifest.created_at or datetime.now(UTC).isoformat()
    path.write_text(manifest.model_dump_json(indent=2))
    return path


def write_record(record: SubsumptionRecord) -> Path:
    """Write one row's provenance record into its raw mirror."""
    root = data_root() / RAW_REL / record.citation_key
    root.mkdir(parents=True, exist_ok=True)
    path = root / RECORD_NAME
    path.write_text(record.model_dump_json(indent=2))
    return path


BUTLAND_SI_EXPECTED: Final = [
    "Butland 2008 Supplementary Tables 1-4 (data/si2.xls .. data/si5.xls) are the "
    "evidence this key's subsumption_record.json reads. Supplementary Table 4 is the "
    "unfiltered 39 x 8,073 matrix in four layers, so this release is NOT subsumed by "
    "Babu 2014's Table S2 and a loader is warranted; Supplementary Table 1 (si1.pdf, "
    "Supplementary Methods and Figures) stays in the literature mirror because no "
    "measurement reads its bytes, only its OCR"
]


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    """Measure every row, write the results JSON, and optionally deposit the records."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--write-records",
        action="store_true",
        help="deposit each row's record and release files into $DATA_ROOT/torchcell-raw",
    )
    args = parser.parse_args(argv)
    load_dotenv()

    butland, butland_pairs = butland_record()
    results: dict[str, Any] = {
        "script": SCRIPT,
        "measured_at": butland.measured_at,
        "rows": {butland.row_name: butland.model_dump(mode="json")},
        "quote_audit": {butland.row_name: audit_quotes(butland)},
    }
    if args.write_records:
        deposited = deposit_release_files(BUTLAND_KEY, BUTLAND_RAW_DEPOSIT)
        manifest = extend_manifest(
            BUTLAND_KEY,
            deposited,
            doi=BUTLAND_DOI,
            title=BUTLAND_TITLE,
            si_expected=BUTLAND_SI_EXPECTED,
        )
        record = write_record(butland)
        results["written"] = [str(manifest), str(record)]

    os.makedirs(RESULTS, exist_ok=True)
    butland_pairs.to_csv(
        osp.join(RESULTS, "butland2008_high_confidence_pairs.csv"), index=False
    )
    path = osp.join(RESULTS, "bacteria_subsumed_rows.json")
    with open(path, "w") as handle:
        json.dump(results, handle, indent=2)
    print(json.dumps(results, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
