# torchcell/datasets/ecoli/typas2008
# [[torchcell.datasets.ecoli.typas2008]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/typas2008
# Test file: tests/torchcell/datasets/ecoli/test_typas2008.py
r"""Typas 2008: a provenance record of a release that prints no interaction score.

Typas, Nehme, Nichols et al. 2008 (Nature Methods 5:781-787, doi:10.1038/nmeth.1240,
citation key ``typasHighthroughputQuantitativeAnalyses2008``) is row 37 of the fifty
bacterial rows: the eSGA companion paper that introduced the quantitative colony-size
analysis of *E. coli* genetic interactions.

DECISION: NOT LOADED. This module registers no dataset. It is the record of why, and
every clause is a measurement on sha256-pinned bytes rather than a preference. The
script is
``experiments/036-dataset-fixes-before-kg-build/scripts/typas2008_release_loadability.py``
and its committed results sit beside it; the finding is
[[torchcell.datasets.ecoli.typas2008]].

**1. The deposit is mirrored in full, so the old "blocked on a paywalled retrieval"
half of row 37 was STALE.** That is the Butland 2008 lesson repeated. Measured:
enumerating the PMC Article Datasets bucket for ``PMC2700713`` returns NINE objects of
which exactly ONE is supplementary, and it is the file this mirror holds
(:data:`SI_PDF`); the article's own declared supplementary list names exactly that one
href. Published-not-mirrored and mirrored-not-published are both empty, so
``accession_confirmed`` is True and the retrieval is the scriptable ``pmc_cloud`` route.

**2. The released tables carry NO SCORE, and that is why the row stays blocked.** The
one supplementary PDF holds ten tables (:data:`SI_TABLE_CENSUS`). Supplementary Table 2A
lists 23 ``pal`` partners as the TERMS "neg (sick)", "neg (lethal)" and "pos";
Supplementary Table 2B lists 15 ``yraP`` suppressors over gene name, ECK number,
location and function with no value column at all. Of the 42 released pairs exactly 4
carry a number, and that number's statistic is "% Co-inheritance of both markerswhen
host is:", a marker co-transduction check of Supplementary Table 1 rather than a
colony-size interaction score.

**3. ``GeneInteractionPhenotype`` has no categorical mode, probed rather than assumed.**
``gene_interaction`` is annotated ``float`` and required: constructing the leaf with the
released term refuses with ``float_parsing`` ("Input should be a valid number, unable to
parse string as a number") and omitting it refuses with ``missing`` ("Field required").
Both attempts are recorded in :data:`SCHEMA_PROBE`. A float IS accepted, so the blocker
is the release's content and not the leaf.

**4. The one quantitative block is a figure.** The 12 by 12 cross is released as four
heat-map panels of Supplementary Figure 4 (LB-384, LB-1536, M9-384, M9-1536). Its 12
axis genes ARE recoverable from the panel labels (:data:`CROSS_GENES`) and its 66
distinct pairwise doubles are colour cells; no table in the PDF carries them. Recovering
a number from a colour would be inventing a measurement.

**5. NOT SUBSUMED, which is the opposite of what Butland turned out to be.** Measured
against Babu 2014's pinned Table S2 (42,705 rows, 42,592 unordered pairs, 163 donors):
0 of the 38 screen pairs are in it, NEITHER ``pal`` NOR ``yraP`` is among Babu's donors,
and only 2 of the 42 released pairs overlap at all, the two verification pairs
``degP``/``surA`` and ``pal``/``ompA``. Babu does carry other partners of both query
genes as a recipient (:data:`BABU_PARTNERS_OF_QUERY_GENES`), which is why the overlap is
reported as a pair-level measurement rather than as a gene-level one. So closing this row
loses records that no other store holds, and that cost is stated rather than hidden.

**What would reopen it.** A released per-pair colony-size score: the numbers behind
Supplementary Figure 4's panels, or a deposit of the genome-wide M9-glycerol screen the
Table 2A caption describes. Neither is in the PMC deposit, which is complete, so the
reopening is an author release and not a retrieval this project can re-run.
"""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from torchcell.literature.manifest import RetrievalMethod
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import SourcedValue

# --------------------------------------------------------------------------- #
# Provenance anchors
# --------------------------------------------------------------------------- #
CITATION_KEY: Final = "typasHighthroughputQuantitativeAnalyses2008"
PAPER_DOI: Final = "10.1038/nmeth.1240"
PAPER_TITLE: Final = (
    "High-throughput, quantitative analyses of genetic interactions in E. coli"
)
PMC_ID: Final = "PMC2700713"
SCHEDULE_ROW: Final = 37
ISSUE: Final = 826

LIBRARY_DIR_REL: Final = f"torchcell-library/{CITATION_KEY}"


class MirroredArtifact(BaseModel):
    """One pinned artifact of this key's literature mirror, with its retrieval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relpath: str = Field(description="path inside the library mirror for this key")
    sha256: str
    n_bytes: int
    retrieval_method: RetrievalMethod | None = Field(
        default=None,
        description="how the file was retrieved; None for a file this project DERIVED "
        "from another pinned file rather than retrieved, whose recipe is "
        "``processing_command``",
    )
    source_url: str
    retrieval_command: str | None = None
    processing_command: str | None = None
    retrieved_at: str


#: The one supplementary object the publisher deposited, mirrored on 2026-10-07.
SI_PDF: Final = MirroredArtifact(
    relpath="si/si1.pdf",
    sha256="bd531ec2ee865506b9a1c5decb3dd7d3f057896dd65f2b766ae8090b8c0c39da",
    n_bytes=1_197_068,
    retrieval_method=RetrievalMethod.pmc_cloud,
    source_url=(
        "https://pmc-oa-opendata.s3.amazonaws.com/"
        "PMC2700713.1/NIHMS95293-supplement-Supp_Info.pdf"
    ),
    retrieval_command="torchcell.literature.retrieve.pmc_cloud_object",
    retrieved_at="2026-10-07T11:42:44.102078+00:00",
)
#: The OCR of that PDF, which is what the table census below was measured on.
SI_MARKDOWN: Final = MirroredArtifact(
    relpath="si/si1.md",
    sha256="6e41b6083e483f90cf95f5dbf5fccaebfcdcb503b801a271e7b9bd44f944e85c",
    n_bytes=38_193,
    source_url=SI_PDF.source_url,
    processing_command=(
        "torchcell.literature.ocr.ocr_pdf (mineru 2.7.6, backend pipeline, lang en, "
        "method auto, 200 dpi)"
    ),
    retrieved_at=SI_PDF.retrieved_at,
)

SI: Final = Provenance(
    source_uri=SI_MARKDOWN.relpath,
    citation_key=CITATION_KEY,
    sha256=SI_MARKDOWN.sha256,
    method="MinerU OCR of the one deposited supplementary PDF",
)


def _si(
    value: object, quote: str, *, page: str, note: str | None = None
) -> SourcedValue:
    """Bind a value to a verbatim quote of the pinned supplementary OCR."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=SI.source_uri,
            citation_key=SI.citation_key,
            sha256=SI.sha256,
            method=SI.method,
            page=page,
        ),
    )


# --------------------------------------------------------------------------- #
# 1. The deposit is complete
# --------------------------------------------------------------------------- #
class DepositReconciliation(BaseModel):
    """The publisher's deposit against this mirror, both directions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pmcid: str
    n_bucket_objects: int = Field(description="objects the PMC datasets bucket lists")
    n_supplementary_objects: int = Field(description="of those, supplementary files")
    n_declared_supplementary: int = Field(
        description="hrefs the article's own supplementary list names"
    )
    published_not_mirrored: tuple[str, ...]
    mirrored_not_published: tuple[str, ...]
    complete: bool
    probed_at: str


DEPOSIT: Final = DepositReconciliation(
    pmcid=PMC_ID,
    n_bucket_objects=9,
    n_supplementary_objects=1,
    n_declared_supplementary=1,
    published_not_mirrored=(),
    mirrored_not_published=(),
    complete=True,
    probed_at="2026-10-07",
)


# --------------------------------------------------------------------------- #
# 2. What the released tables hold
# --------------------------------------------------------------------------- #
class SiTable(BaseModel):
    """One table of the supplementary PDF, as the OCR renders it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    table_index: int = Field(description="position in the OCR, 0-based")
    n_rows: int
    n_cells: int
    n_numeric_cells: int
    holds: str = Field(description="what the table was measured to carry")


#: Every table of the one deposited PDF. Only index 4, Supplementary Table 1's
#: co-transduction check, carries a number at all, and its statistic is a marker
#: co-inheritance percentage rather than an interaction score.
SI_TABLE_CENSUS: Final[tuple[SiTable, ...]] = (
    *(
        SiTable(
            table_index=index,
            n_rows=3,
            n_cells=13,
            n_numeric_cells=0,
            holds="a Supplementary Figure 4 panel's axis gene labels, no values",
        )
        for index in (0, 1, 2, 3)
    ),
    SiTable(
        table_index=4,
        n_rows=6,
        n_cells=26,
        n_numeric_cells=8,
        holds="Supplementary Table 1: 4 synthetic-lethal pairs with a marker "
        "co-inheritance percentage, not an interaction score",
    ),
    SiTable(
        table_index=5,
        n_rows=25,
        n_cells=97,
        n_numeric_cells=0,
        holds="Supplementary Table 2A: 23 pal partners as the terms neg (sick), "
        "neg (lethal) and pos",
    ),
    SiTable(
        table_index=6,
        n_rows=16,
        n_cells=64,
        n_numeric_cells=0,
        holds="Supplementary Table 2B: 15 yraP suppressors over gene name, ECK nr, "
        "Location and Function, with no value column",
    ),
    SiTable(
        table_index=7,
        n_rows=38,
        n_cells=185,
        n_numeric_cells=0,
        holds="a prose supplementary table carrying no value column",
    ),
    SiTable(
        table_index=8,
        n_rows=24,
        n_cells=72,
        n_numeric_cells=0,
        holds="the strain table (pseudoHfr and cat/kan marked derivatives)",
    ),
    SiTable(
        table_index=9,
        n_rows=12,
        n_cells=24,
        n_numeric_cells=0,
        holds="the primer table",
    ),
)


class ReleasedPairs(BaseModel):
    """Every gene pair the release names, and how many carry a number."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    table_2a_pal_interactions: int
    table_2b_yrap_suppressors: int
    table_1_cotransduction_pairs: int
    total: int
    with_a_numeric_value: int
    numeric_statistic: str = Field(description="the released header, verbatim")


RELEASED_PAIRS: Final = ReleasedPairs(
    table_2a_pal_interactions=23,
    table_2b_yrap_suppressors=15,
    table_1_cotransduction_pairs=4,
    total=42,
    with_a_numeric_value=4,
    numeric_statistic="% Co-inheritance of both markerswhen host is:",
)

TABLE_2_CAPTION: Final = _si(
    "terms, not scores",
    "Supplementary Table 2. (A) Genetic interactions of pal identified in M9-gylcerol "
    "in a genomewide interaction screen in M9 glycerol and independently verified by "
    "reconstructing the double mutants with P1 transduction and then comparing their "
    "growth with the parental single-gene knockouts. Neg stands for negative "
    "interactions and pos for positives.",
    page="Supplementary Table 2 caption",
    note="the caption's own vocabulary is the release's value surface for these 38 "
    "pairs: 'Neg stands for negative interactions and pos for positives', which is a "
    "term and not a number. The genome-wide M9-glycerol screen it describes is not "
    "deposited",
)
TABLE_1_CAPTION: Final = _si(
    "a marker co-transduction check",
    "Supplementary Table 1: Reproduction of synthetic lethal pairs by co-transduction "
    "of a linked marker.",
    page="Supplementary Table 1 caption",
    note="the only numeric table of the deposit; its statistic is the co-inheritance "
    "percentage of a linked marker, which measures linkage rather than interaction",
)


# --------------------------------------------------------------------------- #
# 3. The schema probe
# --------------------------------------------------------------------------- #
class SchemaAttempt(BaseModel):
    """One attempt to construct the interaction leaf from what the release prints."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    attempt: str
    accepted: bool
    error_type: str
    error_message: str


class SchemaProbe(BaseModel):
    """Why ``GeneInteractionPhenotype`` cannot hold a released term."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    leaf: str
    field_annotation: str
    float_accepted: bool
    has_categorical_mode: bool
    attempts: tuple[SchemaAttempt, ...]


SCHEMA_PROBE: Final = SchemaProbe(
    leaf="GeneInteractionPhenotype",
    field_annotation="gene_interaction: float, required",
    float_accepted=True,
    has_categorical_mode=False,
    attempts=(
        SchemaAttempt(
            attempt="term_as_value",
            accepted=False,
            error_type="float_parsing",
            error_message=(
                "Input should be a valid number, unable to parse string as a number"
            ),
        ),
        SchemaAttempt(
            attempt="no_value",
            accepted=False,
            error_type="missing",
            error_message="Field required",
        ),
    ),
)


# --------------------------------------------------------------------------- #
# 4. The one quantitative block is a figure
# --------------------------------------------------------------------------- #
#: The 12 axis genes of the cross, recovered from the Supplementary Figure 4 panels.
CROSS_GENES: Final[tuple[str, ...]] = (
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
#: Distinct unordered pairs of the 12 axis genes: 12 * 11 / 2.
CROSS_DISTINCT_DOUBLES: Final = 66

CROSS_CAPTION: Final = _si(
    "four heat-map panels",
    "Supplementary Figure 4: Heat maps representing ${ 1 2 \\times }$ 12 crosses in all "
    "four different datasets: (A) LB-384; (B) LB-1536; (C) M9-384; (D) M9-1536.",
    page="Supplementary Figure 4 caption",
    note="the 12 axis genes are recoverable from the panel labels and the 66 distinct "
    "doubles are colour cells; no table of the deposit carries their values, so a "
    "stored score would be read off a colour",
)


# --------------------------------------------------------------------------- #
# 5. Not subsumed by Babu 2014
# --------------------------------------------------------------------------- #
class Subsumption(BaseModel):
    """Whether a served store already holds these pairs. Measured, not argued."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    against: str = Field(description="the served dataset the overlap was taken against")
    source_uri: str
    sha256: str
    their_rows: int
    their_unordered_pairs: int
    their_donors: int
    our_released_pairs: int
    our_pairs_in_theirs: int
    query_genes_among_their_donors: dict[str, bool]
    subsumed: bool


BABU_SUBSUMPTION: Final = Subsumption(
    against="GeneInteractionBabu2014Dataset",
    source_uri=(
        "$DATA_ROOT/torchcell-raw/babuQuantitativeGenomeWideGenetic2014/data/si23.xls"
    ),
    sha256="0789563ada0db3e349bb7e5396311f9d705ea165090733803c009acfa7c95b96",
    their_rows=42_705,
    their_unordered_pairs=42_592,
    their_donors=163,
    our_released_pairs=42,
    our_pairs_in_theirs=2,
    query_genes_among_their_donors={"pal": False, "yraP": False},
    subsumed=False,
)
#: The two released pairs Babu 2014 does carry: the verification pairs, not the screen.
OVERLAPPING_PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("degP", "surA"),
    ("pal", "ompA"),
)
#: Partners of the two query genes that Babu DOES hold, with either as the recipient.
#: Reported so the overlap claim is a pair-level measurement, not a gene-level one.
BABU_PARTNERS_OF_QUERY_GENES: Final[dict[str, tuple[str, ...]]] = {
    "pal": (
        "ddlA",
        "efp",
        "malQ",
        "ompA",
        "oppA",
        "thiM",
        "tig",
        "ycbX",
        "ygcI",
        "yghD",
    ),
    "yraP": ("csdA", "recC"),
}


# --------------------------------------------------------------------------- #
# The settled row
# --------------------------------------------------------------------------- #
class SettledRow(BaseModel):
    """The schedule row this module settles, and the terms it settles it on."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    row: int
    citation_key: str
    doi: str
    status: str
    accession_confirmed: bool
    loaded_records: int
    reopens_on: str


SETTLED: Final = SettledRow(
    row=SCHEDULE_ROW,
    citation_key=CITATION_KEY,
    doi=PAPER_DOI,
    status="blocked",
    accession_confirmed=True,
    loaded_records=0,
    reopens_on=(
        "a released per-pair colony-size score: the numbers behind Supplementary "
        "Figure 4's four panels, or a deposit of the genome-wide M9-glycerol screen "
        "the Supplementary Table 2A caption describes. The PMC deposit is complete, so "
        "this is an author release and not a retrieval this project can re-run"
    ),
)

#: Every sourced quote this record rests on, by name.
SOURCED_VALUES: Final[dict[str, SourcedValue]] = {
    "table_2_caption": TABLE_2_CAPTION,
    "table_1_caption": TABLE_1_CAPTION,
    "cross_caption": CROSS_CAPTION,
}
