# torchcell/datasets/ecoli/cai2023
# [[torchcell.datasets.ecoli.cai2023]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/cai2023
# Test file: tests/torchcell/datasets/ecoli/test_cai2023.py
"""MCF2Chem 2023: a provenance record of an aggregation this ontology cannot admit.

Cai et al. 2023 (Biotechnol. Biofuels Bioprod. 16:170, doi:10.1186/s13068-023-02419-8,
citation key ``caiMCF2ChemManuallyCurated2023``) is row 46 of the fifty bacterial rows,
``status="aggregation"``: a manually curated knowledge base of 8,888 microbial cell
factory production records (titer, yield, productivity, content) over 1,231 compounds
and 590 species, of which bacteria are 5,276 records.

DECISION: NOT LOADED. This module registers no dataset. It is the record of why, and
every clause of the why is a measurement on sha256-pinned bytes, not a preference.

**1. There is no per-record artifact to pin.** The one distribution channel the paper
names is the web server (:data:`AVAILABILITY`), and the release's three supplementary
files hold no production records: Additional file 1 is the list of 268 review DOIs,
Additional file 2 is nine figures plus the coverage Table S2, Additional file 3 is the
two recommendation scoring equations. Measured by :func:`release_inventory`: three files,
one table each, 269 x 2, 5 x 4 and 7 x 2 cells, and not one header naming a titer, a
yield, a productivity, a strain or a compound. PMC OA serves exactly those three objects
(``MOESM4`` and ``MOESM5`` answer 404). On 2026-10-08 the accession answered a static
"Service Relocation in Progress" page and its API answered 502
(:data:`ACCESSION_PROBE_2026_10_08`). The schedule's own status vocabulary settles what
that means: "a row whose per-record values are not released is ``blocked``". There is
nothing to hash, so no retrieval command could ever be re-run and verified.

**2. The provenance chain does not reach a measurement, and that is by design.** The
values were not read from the papers that made them. They were read from review tables
(:data:`EXTRACTION_FROM_REVIEWS`, :data:`WHY_REVIEWS`), and the per-record citation was
then recovered from the review's own reference column
(:data:`PER_RECORD_REFERENCE_RULE`): 4,765 original articles spanning 1946 to 2022, 92
of the records being patents. So a stored number is a review author's transcription of a
primary paper, re-transcribed by the curators, and honoring it would mean mirroring 268
reviews plus 4,765 primary articles to verify one quote per record. The two landed
aggregation loaders set the opposite convention and it is the deciding one: both
aggregate RAW primary data that the aggregating paper RECOMPUTED itself (Lim 2022
recomputes log2(TPM + 1) from read counts, Borchert 2024 computes gene fitness from
barcode counts), and Borchert 2024 attributes a sample to a prior study only when that
study's MIRRORED Methods name its condition. An aggregation is admissible here when it
re-measures; MCF2Chem re-types.

**3. Four required schema fields have no value in the release** even granted the
records, each with the release's own statement of the limit
(:data:`SCHEMA_BLOCKERS`): ``ProductTiterExperimentReference.phenotype_reference`` needs
a released parent-strain titer that a review's headline row does not carry,
``ProductTiterPhenotype.product`` is a typed ``Compound`` while about 32% of MCF2Chem's
compounds resolve to nothing in PubChem, ``titer`` is one float while a portion of the
rows are ranges, and ``titer_unit`` is a typed enum while un-convertible units are kept
as written. The genotype is free text ("possible strain modifcation methods, strain
genotypes"), which no bacterial perturbation leaf accepts.

**What it would have duplicated, measured.** Of the 51 served bacterial dataset classes,
exactly four carry ``ProductTiterExperiment``, and MCF2Chem's extraction window closes
on 2022-07-31, so only one of the four (Foo 2014) names a paper its reviews could have
summarized; Carruthers 2025, De Siqueira 2025 and Kang 2026 all postdate the window.
None of the 268 review DOIs is the DOI of any dataset we serve or of any row in the
bacteria candidate table. The per-record overlap is NOT measured and cannot be while the
records are unreleased, so it is stated as unmeasured rather than as a null.

Script and committed results:
``experiments/036-dataset-fixes-before-kg-build/scripts/mcf2chem2023_release_inventory.py``.
Finding: [[torchcell.datasets.ecoli.cai2023]].
"""

from __future__ import annotations

import os
import os.path as osp
import zipfile
from pathlib import Path
from typing import Final
from xml.etree import ElementTree

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from torchcell.verification.report import Provenance
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
)

# --------------------------------------------------------------------------- #
# Provenance anchors
# --------------------------------------------------------------------------- #
CITATION_KEY: Final = "caiMCF2ChemManuallyCurated2023"
PAPER_DOI: Final = "10.1186/s13068-023-02419-8"
PAPER_TITLE: Final = (
    "MCF2Chem: A manually curated knowledge base of biosynthetic compound production"
)
LIBRARY_DIR_REL: Final = "torchcell-library"

PAPER_MD: Final = "paper.md"
PAPER_MD_SHA256: Final = (
    "3c3557e6fc1d031dbe62b4ad756b47c2b2d1298799d1c85f5726c50fcf0b8c4b"
)

#: The one distribution channel the paper names for the 8,888 records.
ACCESSION_URL: Final = "https://mcf.lifesynther.com"

#: MCF2Chem's extraction window: the publication dates of the reviews it read.
REVIEW_WINDOW_START: Final = "2017-08-01"
REVIEW_WINDOW_END: Final = "2022-07-31"

#: A header cell naming any of these would make a table a production-record table.
PRODUCTION_HEADER_TERMS: Final = (
    "titer",
    "yield",
    "productivity",
    "content",
    "strain",
    "compound",
    "species",
)

PAPER: Final = Provenance(
    source_uri=PAPER_MD, citation_key=CITATION_KEY, sha256=PAPER_MD_SHA256
)


class SiArtifact(BaseModel):
    """One pinned supplementary file of the release, with what it was found to hold."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relpath: str = Field(description="path inside the library mirror for this key")
    sha256: str
    n_bytes: int
    role: str = Field(description="the paper's own name for the file, verbatim")
    holds: str = Field(description="what its tables and paragraphs were measured to be")
    source_url: str = Field(description="the PMC OA object the mirror retrieved")


SI_FILES: Final[dict[str, SiArtifact]] = {
    "si1": SiArtifact(
        relpath="si/si1.docx",
        sha256="88cb9cc3a1a2eead147e11d570050f388d38996be9fdc028f036bac8cc3fd826",
        n_bytes=40017,
        role="Additional file 1: Table S1",
        holds="one table, 268 review rows of (Review_title, Review_doi)",
        source_url=(
            "https://pmc-oa-opendata.s3.amazonaws.com/PMC10625697.1/"
            "13068_2023_2419_MOESM1_ESM.docx"
        ),
    ),
    "si2": SiArtifact(
        relpath="si/si2.docx",
        sha256="e698595306f5c8c47a0256e2a4672c9ef0230c337d17b0fe952cf0443bf8dfa8",
        n_bytes=20723784,
        role="Additional file 2: Figs. S1-S9 and Table S2",
        holds=(
            "nine figure captions and one table, the Metabolic Engineering coverage "
            "analysis for 2016-2018 (66%, 65%, 60%; 63% average)"
        ),
        source_url=(
            "https://pmc-oa-opendata.s3.amazonaws.com/PMC10625697.1/"
            "13068_2023_2419_MOESM2_ESM.docx"
        ),
    ),
    "si3": SiArtifact(
        relpath="si/si3.docx",
        sha256="2c79046955b54ca54aa69d8c9256faf91c0002025444e6b538dab1c1849cbc0c",
        n_bytes=20343,
        role="Additional file 3",
        holds=(
            "the two recommendation scoring equations and one table of six "
            "(Route, Score) rows"
        ),
        source_url=(
            "https://pmc-oa-opendata.s3.amazonaws.com/PMC10625697.1/"
            "13068_2023_2419_MOESM3_ESM.docx"
        ),
    ),
}

# --------------------------------------------------------------------------- #
# What the release says about itself
# --------------------------------------------------------------------------- #
RECORD_COUNTS: Final = SourcedValue(
    value={"records": 8888, "reviews": 268, "original_articles": 4765, "patents": 92},
    provenance=PAPER.model_copy(update={"page": "Results, Database overview"}),
    quote=(
        "In total, 8888 items of production records were extracted from 268 review "
        "articles, involving information from 4765 original microbial metabolic "
        "engineering articles"
    ),
    note="The patent count is the same sentence's parenthesis, below.",
)

PATENT_RECORDS: Final = SourcedValue(
    value=92,
    provenance=PAPER.model_copy(update={"page": "Results, Database overview"}),
    quote="(92 records were those of patents",
    note=(
        "92 of the 8,888 records have no article at all behind them, so for those the "
        "chain ends at a patent."
    ),
)

BACTERIAL_RECORDS: Final = SourcedValue(
    value={
        "records": 5276,
        "products": 835,
        "species": 356,
        "articles": 2978,
        "reviews": 195,
    },
    provenance=PAPER.model_copy(update={"page": "Table 1"}),
    quote="<td>Bacteria</td><td>5276</td><td>835</td><td>356</td><td>2978</td><td>195</td>",
    note=(
        "Table 1's Bacteria row, quoted as the OCR's HTML cells. The table has no "
        "per-species row, which is why the row's recorded E. coli subset size is "
        "undetermined."
    ),
)

ORIGINAL_ARTICLE_SPAN: Final = SourcedValue(
    value=(1946, 2022),
    provenance=PAPER.model_copy(update={"page": "Results, Database overview"}),
    quote="Te 4765 articles concerned spanned the period from 1946 to 2022",
    note="The primary literature behind the records, not the reviews' own window.",
)

EXTRACTION_FROM_REVIEWS: Final = SourcedValue(
    value=(REVIEW_WINDOW_START, REVIEW_WINDOW_END),
    provenance=PAPER.model_copy(
        update={"page": "Methods, Data collection and processing"}
    ),
    quote=(
        "Te raw data of MCF2Chem were extracted from reviews of microbial biosynthesis "
        "over the last 5 years (from August 1, 2017, to July 31, 2022)."
    ),
    note=(
        "The window the duplication test uses: a primary paper published after "
        "2022-07-31 cannot be summarized by a review in this corpus."
    ),
)

PER_RECORD_REFERENCE_RULE: Final = SourcedValue(
    value="review_reference_column",
    provenance=PAPER.model_copy(
        update={"page": "Methods, Data collection and processing"}
    ),
    quote=(
        "Based on the reference columns in review tables, direct references to each "
        "record were obtained and supplemented programmatically or manually."
    ),
    note=(
        "This is the answer to whether the row carries per-record provenance to the "
        "originating paper: it carries a per-record CITATION, taken from the review's "
        "reference column, while the VALUE comes from the review's table cell. The "
        "citation is not evidence for the number beside it."
    ),
)

WHY_REVIEWS: Final = SourcedValue(
    value="reviews_were_read_instead_of_primary_literature",
    provenance=PAPER.model_copy(update={"page": "Discussion"}),
    quote=(
        "manually extracting information directly from original literature is both "
        "time-consuming and labor-intensive"
    ),
    note="The authors state the substitution plainly; it is not an inference.",
)

CURATION_LAG: Final = SourcedValue(
    value="reviews_omit_and_lag",
    provenance=PAPER.model_copy(update={"page": "Discussion"}),
    quote="owing to their lagging nature, omission of the latest data is inevitable",
)

AVAILABILITY: Final = SourcedValue(
    value=ACCESSION_URL,
    provenance=PAPER.model_copy(update={"page": "Availability of data and materials"}),
    quote="All data are available at https://mcf.lifesynther.com.",
    note=(
        "The complete statement. No repository, no deposit, no DOI'd data file, and "
        "the three supplementary files hold no records."
    ),
)

RECORD_DETAIL_PAGE: Final = SourcedValue(
    value="web_page_per_record",
    provenance=PAPER.model_copy(
        update={"page": "Results, Recommendation system and user interface"}
    ),
    quote="Each production record is also available on the Production Record Details page.",
    note="The records are a web view, which is what makes the accession the only channel.",
)

#: The statistic the release releases about its own coverage of one primary journal,
#: measured off ``si/si2.docx`` rather than quoted: a docx is a zip, so its text is
#: asserted at read time by :func:`table_s2_coverage` instead of audited after the fact.
TABLE_S2_COVERAGE: Final[dict[str, str]] = {
    "2016": "66%",
    "2017": "65%",
    "2018": "60%",
    "average": "63%",
}

PER_RECORD_VALUES_GAP: Final = ProvenanceGap(
    field="production_records",
    reason=ProvenanceGapReason.not_carried_by_curation,
    looked_in=PAPER.model_copy(
        update={"page": "Availability of data and materials; Additional files 1-3"}
    ),
    note=(
        "The 8,888 records are released only through the web server, which on "
        "2026-10-08 served a relocation notice and a 502 from its API. The three "
        "supplementary files carry the review list, the figures and the scoring "
        "equations. Terminal, not recoverable work: there is no artifact to fetch."
    ),
)


class SchemaBlocker(BaseModel):
    """One required schema field the release has no value for, and the quote."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    field: str = Field(description="fully qualified schema field that must be filled")
    requirement: str = Field(description="what the schema requires of it")
    evidence: SourcedValue = Field(
        description="the release's own statement of the limit"
    )


SCHEMA_BLOCKERS: Final[tuple[SchemaBlocker, ...]] = (
    SchemaBlocker(
        field="ProductTiterExperimentReference.phenotype_reference",
        requirement=(
            "a required ProductTiterPhenotype, so every record needs a released "
            "parent-strain titer read in the same CultureEnvironment"
        ),
        evidence=SourcedValue(
            value="no_reference_titer_column",
            provenance=PAPER.model_copy(
                update={"page": "Methods, Data collection and processing"}
            ),
            quote=(
                "Production data of compounds were divided into four columns: titer, "
                "yield, productivity, and content."
            ),
            note=(
                "Four product columns and no control column: a review table reports the "
                "engineered strain's headline number. Three sibling loaders already "
                "refuse the titer family for want of a released reference titer."
            ),
        ),
    ),
    SchemaBlocker(
        field="ProductTiterPhenotype.product",
        requirement="a typed Compound, the join key onto the shared compound layer",
        evidence=SourcedValue(
            value=0.32,
            provenance=PAPER.model_copy(update={"page": "Discussion"}),
            quote=(
                "approximately $3 2 \\%$ of the compounds in MCF2Chem cannot be "
                "retrieved from PubChem"
            ),
            note="About a third of the compound space has no resolvable identity.",
        ),
    ),
    SchemaBlocker(
        field="ProductTiterPhenotype.titer",
        requirement="one required float, validated finite and non-negative",
        evidence=SourcedValue(
            value="ranges_not_point_values",
            provenance=PAPER.model_copy(
                update={"page": "Methods, Data collection and processing"}
            ),
            quote="titer range data were divided into maximum and minimum titers",
            note="A range is two numbers; the field is one, and neither end is the value.",
        ),
    ),
    SchemaBlocker(
        field="ProductTiterPhenotype.titer_unit",
        requirement="a typed UO-aligned ConcentrationUnit",
        evidence=SourcedValue(
            value="unconverted_units_retained",
            provenance=PAPER.model_copy(
                update={"page": "Methods, Data collection and processing"}
            ),
            quote="original units were retained for those that could not be converted",
            note=(
                'The Discussion states the cause: "the production units used were '
                'diverse, and some units were difcult to unify".'
            ),
        ),
    ),
    SchemaBlocker(
        field="ProductTiterExperiment.genotype",
        requirement=(
            "a Genotype of typed bacterial perturbations, each an edit to the genomic "
            "content of the cell"
        ),
        evidence=SourcedValue(
            value="free_text_genotype",
            provenance=PAPER.model_copy(
                update={"page": "Methods, Data collection and processing"}
            ),
            quote=(
                "All other parts included possible strain modifcation methods, strain "
                "genotypes, and other information."
            ),
            note=(
                '"possible" is the release\'s own hedge on the modification method; no '
                "perturbation leaf accepts a review-table genotype string."
            ),
        ),
    ),
)


class ProbedPath(BaseModel):
    """One path of the accession and what it answered."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    status: int | None = Field(description="HTTP status, or None on transport failure")
    note: str


#: The accession as it answered on 2026-10-08, recorded so the finding does not depend
#: on re-probing a host that is mid-relocation. Reproduce with the script's ``probe``.
ACCESSION_PROBE_2026_10_08: Final[tuple[ProbedPath, ...]] = (
    ProbedPath(
        path="/",
        status=200,
        note=(
            'a 3,821-byte static page titled "Service Relocation in Progress": '
            '"The website will be temporarily unavailable during the move."'
        ),
    ),
    ProbedPath(
        path="/openapi.json",
        status=502,
        note="the FastAPI backend the paper describes is not answering",
    ),
    ProbedPath(path="/download", status=404, note="no bulk export"),
    ProbedPath(path="/api/download", status=404, note="no bulk export"),
    ProbedPath(path="/data", status=404, note="no data route"),
    ProbedPath(path="/api/data", status=404, note="no data route"),
    ProbedPath(path="/browse", status=404, note="the Browsing page of Fig. 5F is gone"),
    ProbedPath(path="/api/browse", status=404, note="no browse API"),
    ProbedPath(path="/docs", status=404, note="no API documentation"),
    ProbedPath(path="/redoc", status=404, note="no API documentation"),
)

#: Paths ``probe`` re-reads, in the recorded order.
PROBE_PATHS: Final[tuple[str, ...]] = tuple(
    row.path if row.path != "/" else "" for row in ACCESSION_PROBE_2026_10_08
)


# --------------------------------------------------------------------------- #
# Measurement: what the release contains
# --------------------------------------------------------------------------- #
_W: Final = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def library_dir(data_root: str | None = None) -> Path:
    """This key's directory in the OCR library mirror."""
    if data_root is None:
        load_dotenv()
        data_root = os.environ["DATA_ROOT"]
    return Path(osp.join(data_root, LIBRARY_DIR_REL, CITATION_KEY))


def _paragraph_text(paragraph: ElementTree.Element) -> str:
    """Concatenated run text of one WordprocessingML paragraph."""
    return "".join(node.text or "" for node in paragraph.iter(_W + "t"))


def docx_body(path: str | Path) -> ElementTree.Element:
    """The ``w:body`` element of a .docx, read with the standard library only."""
    with zipfile.ZipFile(path) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
    body = root.find(_W + "body")
    if body is None:
        raise ValueError(f"{path} has no WordprocessingML body")
    return body


def docx_text(path: str | Path) -> str:
    """Every non-empty paragraph of a .docx as newline-joined text."""
    lines = [
        text
        for paragraph in docx_body(path).iter(_W + "p")
        if (text := _paragraph_text(paragraph).strip())
    ]
    return "\n".join(lines)


def docx_tables(path: str | Path) -> list[list[list[str]]]:
    """Every table of a .docx as rows of cell strings, in document order."""
    tables: list[list[list[str]]] = []
    for table in docx_body(path).iter(_W + "tbl"):
        tables.append(
            [
                [
                    " ".join(_paragraph_text(p) for p in cell.findall(_W + "p")).strip()
                    for cell in row.findall(_W + "tc")
                ]
                for row in table.findall(_W + "tr")
            ]
        )
    return tables


class ReleaseInventory(BaseModel):
    """Every table of every supplementary file, and whether any holds a record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    citation_key: str
    accession_url: str
    si_files: list[SiArtifact]
    n_tables: int = Field(description="tables across all supplementary files")
    table_shapes: dict[str, list[tuple[int, int]]] = Field(
        description="(rows, columns) per table, by supplementary file"
    )
    n_production_tables: int = Field(
        description="tables whose header names a titer, strain or compound"
    )
    per_record_artifact_released: bool = Field(
        description="whether any released file carries the production records"
    )


def inventory_from_tables(
    si_tables: dict[str, list[list[list[str]]]],
) -> ReleaseInventory:
    """Build the inventory from already-extracted tables (the pure half).

    A table counts as a production table when a header cell names any
    :data:`PRODUCTION_HEADER_TERMS` term, which is deliberately generous: the point is
    that even a generous rule finds none.
    """
    if set(si_tables) != set(SI_FILES):
        raise ValueError(
            f"expected tables for {sorted(SI_FILES)}, got {sorted(si_tables)}"
        )
    shapes: dict[str, list[tuple[int, int]]] = {}
    n_tables = 0
    n_production = 0
    for name, tables in si_tables.items():
        shapes[name] = [(len(table), len(table[0])) for table in tables]
        n_tables += len(tables)
        for table in tables:
            header = [cell.strip().lower() for cell in table[0]]
            if any(term in cell for term in PRODUCTION_HEADER_TERMS for cell in header):
                n_production += 1
    return ReleaseInventory(
        citation_key=CITATION_KEY,
        accession_url=ACCESSION_URL,
        si_files=[SI_FILES[name] for name in sorted(SI_FILES)],
        n_tables=n_tables,
        table_shapes=shapes,
        n_production_tables=n_production,
        per_record_artifact_released=n_production > 0,
    )


def release_inventory(library: str | Path | None = None) -> ReleaseInventory:
    """Read the three pinned supplementary files and inventory them."""
    base = Path(library) if library is not None else library_dir()
    return inventory_from_tables(
        {name: docx_tables(base / record.relpath) for name, record in SI_FILES.items()}
    )


def table_s2_coverage(library: str | Path | None = None) -> dict[str, str]:
    """Table S2's coverage rates, asserted against :data:`TABLE_S2_COVERAGE`.

    A docx cannot be quote-audited against its bytes, so the statement is checked when
    it is read and a drift raises here rather than being reported later.
    """
    base = Path(library) if library is not None else library_dir()
    tables = docx_tables(base / SI_FILES["si2"].relpath)
    if len(tables) != 1:
        raise ValueError(f"Additional file 2 holds {len(tables)} tables, expected 1")
    rows = {row[0]: row[1:] for row in tables[0]}
    years = rows["Year"]
    rates = rows["MCF2Chem coverage rate"]
    measured = dict(zip(years, rates, strict=True))
    measured["average"] = rows["MCF2Chem average coverage rate"][0]
    if measured != TABLE_S2_COVERAGE:
        raise ValueError(f"Table S2 coverage moved: {measured} != {TABLE_S2_COVERAGE}")
    return measured
