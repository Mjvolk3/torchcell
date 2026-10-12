# torchcell/datasets/ecoli/gerdes2003
# [[torchcell.datasets.ecoli.gerdes2003]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/gerdes2003
# Test file: tests/torchcell/datasets/ecoli/test_gerdes2003.py
"""Gerdes 2003 genetic-footprinting essentiality calls of E. coli K-12 MG1655 genes.

Gerdes, Scholle, Campbell, ... Barabasi, Oltvai and Osterman 2003 (J Bacteriol
185:5673-5684, doi:10.1128/JB.185.19.5673-5684.2003, citation key
``gerdesExperimentalDeterminationSystem2003``) mutagenized MG1655 with the EZ::TN<KAN-2>
transposome, grew the pool for 23 population doublings in a fermentor, mapped the
surviving insertions by nested PCR, and asserted each ORF essential (E), non-essential
(N), not covered (X) or ambiguous (?) from the number and positions of its insertions.

DATA. Table S1 (``table_s1.xls``) of the authors' Supplementary Online Material at
``https://www.genome.wisc.edu/Gerdes2003/``, with that page's column comments, the two
procedure PDFs (each with a ``pdftotext -layout`` rendering) and PubMed's XML record of
the article, deposited in ``$DATA_ROOT/torchcell-raw/gerdesExperimentalDeterminationSystem2003/``
by ``experiments/036-dataset-fixes-before-kg-build/scripts/gerdes2003_release_inventory.py``.
The article body is walled for scripts and the paper is in neither Zotero library, so
every quote is anchored to the supplement or the PubMed abstract. That the Wisconsin
copy is the published Table S1 is measured: Baba 2006's Supplementary Table 3 re-serves
"Supplementary Table S1 (Gerdes et al., 2003)" per ORF, and its call equals this table's
on 4,113 of the 4,133 b-numbers both carry (the other 20: 15 Baba marks absent, 5 are
insertion-sequence b-numbers this table lists twice).

THE CALL, STORED AS RELEASED. E is ``is_essential=True``; N is False. Six rows carry
``N*`` (ftsK, rpsA, kdsA, accD, usg, nusA); the page defines E, N, X and ? but not the
asterisk, and says "accD is called nonessential ... regardless of convincing arguments
to the contrary", with accD's call written ``N*``. So ``N*`` is an N call whose asterisk
flags contrary expectation, stored False. X ("the gene was not covered") and ?
("ambiguous") state no boolean: those rows store ``is_essential=None`` beside a typed
``ProvenanceGap`` on the field. Every row keeps its token verbatim in ``released_call``.
What ``is_essential=True`` means here is that no disruptive insertion survived 23
doublings of competitive growth in a pool, which can include slow growers (Baba 2006
says so of genetic footprinting).

COUNTS. Table S1 calls 619 E, 3,114 N, 6 N*, 331 X and 221 ? (4,291 rows); the abstract
states 620 essential and 3,126 dispensable, and the procedure overview 3,746 unambiguous
assertions. The table is 7 short of 3,746 (619 + 3,120); the per-gene table is what is
stored, and the difference is recorded, not resolved (the walled article body is where
it would be explained).

STRAIN AND IDENTIFIERS, STORED AS RELEASED (owner rule of 2026-10-12: the data is the
data, and the genome never gates a record). MG1655 (``SOURCED_VALUES["background_strain"]``).
Every one of the 4,291 Table S1 rows is a record, and no annotation is read in the
build. A row whose ``Blattner ID`` is a well-formed b-number that no other row carries
(4,208 rows) is stored on ``TransposonInsertionPerturbation`` under that b-number
verbatim, with ``identifier_mapping=None``: the tag is the release's, whether or not
the pinned annotation carries it, and the verifier's L4 row reports how many of them
the annotation does, as a finding. The other 83 rows are stored on
``BacterialReleasedTransposonPerturbation`` under their ERGO id, Table S1's own unique
row identifier: 64 rows with no Blattner ID, one whose Blattner ID is ``b0018.1``, and
the 18 rows of 9 b-numbers the table lists twice (insertion-sequence copies at
different chromosome positions, each with its own call), which one b-number cannot key
twice. The Blattner ID of those 19 rows stays a released column of
``preprocess/calls.csv``.

EVIDENCE (#924). Every phenotype carries a ``GeneEssentialityEvidence`` of type
``genetic_footprinting`` (class ``insertion_depletion``): the essential rule of the
assertion criteria, the procedure's "essential under these growth conditions" as the
call's definition, the page's caveat that every call is "based exclusively on our
experimental data", the assertion-error definition, and per record the released call
letter, the recovered insertions (amino-acid coordinates) and the assertion error.
Table S1 writes the assertion error as a value (65 E rows), a bound such as ``< 0.05``
(545 E rows, ``assertion_error_is_upper_bound``), ``n.d.`` (8 E rows) or an empty cell
(proC); the last two are typed gaps. One N row (yaiI) carries ``< 0.05``; it shares its
start, 404868, with proC, the E row whose cell is empty, and is stored as released.

COUNTS. Table S1 calls 619 E, 3,114 N, 6 N*, 331 X and 221 ? (4,291 rows); the abstract
states 620 essential and 3,126 dispensable, and the procedure overview 3,746 unambiguous
assertions. The table is 7 short of 3,746 (619 + 3,120); the per-gene table is what is
stored, and the difference is recorded, not resolved (the walled article body is where
it would be explained).

STRAIN AND IDENTIFIERS. MG1655 (``SOURCED_VALUES["background_strain"]``). The
``Blattner ID`` column goes through ``reconcile_locus_tags`` on the pinned MG1655
assembly; a b-number the annotation now carries as a synonym of another locus is stored
under that locus with a ``locus_tag_synonym`` mapping. Dropped: 64 rows with no Blattner
ID, the 18 rows of 9 b-numbers this table lists twice (insertion-sequence copies with
different calls), and b-numbers on no single MG1655 locus (retired, ambiguous, or a
fragment of a locus another row holds directly).

GENOTYPE. One gene-level ``TransposonInsertionPerturbation`` (transposon
"EZ::TN<KAN-2>"). The table lists each gene's insertion positions in amino-acid
coordinates, but a record stands for every insertion in that gene, so
``insertion_position`` and ``insertion_strand`` are typed absences
(``PERTURBATION_FIELD_GAPS``); the positions are kept in ``preprocess/calls.csv``.

ENVIRONMENT. The fermentor: an "LB-based rich" medium (tryptone 10 g/L, yeast extract
5 g/L, 50 mM NaCl, Neidhardt's salts and micronutrients, nine vitamins) with kanamycin
at 10 ug/ml, 37 C, dissolved oxygen 30-50% of saturation, 23 population doublings in 12
hours. The pH was held at 6.95 by titration; ``Environment`` has no pH field and a pH
edit would be a perturbation every record and the reference share, so it is recorded
here and in the medium's provenance, not as a perturbation. Vitamins are stated in
mg/L, which is ug/mL exactly.

REFERENCE. The unperturbed MG1655 parent, viable, in the same medium WITHOUT the
kanamycin (the paper's "LB-based rich media ... without kanamycin"): the parent carries
no kanamycin marker.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import os.path as osp
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Literal

import pandas as pd
import xlrd
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    Organism,
    check_manifest_pin,
    link_verified,
    post_process,
    verify_raw_files,
)
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.schema import (
    AssemblyReferenceGenome,
    BacterialGeneEssentialityExperiment,
    BacterialGeneEssentialityExperimentReference,
    BacterialReleasedTransposonPerturbation,
    ComponentDefinition,
    Compound,
    Concentration,
    ConcentrationUnit,
    Environment,
    EssentialityEvidenceClass,
    EssentialityEvidenceType,
    EssentialityStatement,
    EssentialityStatementRole,
    Experiment,
    ExperimentReference,
    FootprintingCall,
    GeneEssentialityEvidence,
    GeneEssentialityPhenotype,
    Genotype,
    Media,
    MediaComponent,
    MediaComponentRole,
    Publication,
    Temperature,
    TransposonInsertionPerturbation,
)
from torchcell.datasets.bacteria_common import (
    STRAIN_GENE_NAMESPACES,
    assembly_reference,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.literature.manifest import Manifest
from torchcell.sequence.genome.ecoli.k12 import EcoliK12StrainName
from torchcell.verification.levels import l0_structural, l1_count
from torchcell.verification.report import (
    Level,
    LevelResult,
    Provenance,
    VerificationReport,
)
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
    audit_sourced_value,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Provenance anchors
# --------------------------------------------------------------------------- #
CITATION_KEY = "gerdesExperimentalDeterminationSystem2003"
PAPER_DOI = "10.1128/JB.185.19.5673-5684.2003"
PUBMED_ID = "13129938"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"
DATASET_ROOT_REL = "data/torchcell/gene_essentiality_gerdes2003"
RAW_RETRIEVED_AT = "2026-10-10"

TABLE_S1 = "table_s1.xls"
TABLE_S1_REL = f"data/{TABLE_S1}"
TABLE_S1_SHA256 = "2c4c24570ef1fa40303ed1c1fe24f0c02d7fa5975bd4560d48b29c9e3790b56e"
TABLE_PAGE_REL = "si/supplementary_table.html"
TABLE_PAGE_SHA256 = "f29982272aab87de1c402e2fd1b2a0c5d6cd54cbfbcd259816a2245681100510"
GENETIC_TEXT_REL = "si/genetic.pdftotext.txt"
GENETIC_TEXT_SHA256 = "932b56c3d8f8593a64477edfb9f610e3dd91b97e06b95c84862a4c2f953f9668"
ASSERTION_TEXT_REL = "si/assertion_of_essentiality.pdftotext.txt"
ASSERTION_TEXT_SHA256 = (
    "f19345e2a9c768d5ffb8e9cc3c0d0437487dbc036755170e7576401262f89f95"
)
PUBMED_XML_REL = f"paper/pubmed_{PUBMED_ID}.xml"
PUBMED_XML_SHA256 = "294205fd9da3bab725f0c8f398a0c8ddaecc7e79d569d544f3e23249c84b5d5b"

#: ``{raw file name: pinned sha256}``, the build-time check of the consumed table.
DATA_SHA256: dict[str, str] = {TABLE_S1: TABLE_S1_SHA256}

_ANCHORS: dict[str, tuple[str, str]] = {
    "page": (TABLE_PAGE_REL, TABLE_PAGE_SHA256),
    "genetic": (GENETIC_TEXT_REL, GENETIC_TEXT_SHA256),
    "assertion": (ASSERTION_TEXT_REL, ASSERTION_TEXT_SHA256),
    "pubmed": (PUBMED_XML_REL, PUBMED_XML_SHA256),
}
_ANCHOR_METHODS: dict[str, str] = {
    "page": "the Table S1 page of the Supplementary Online Material (raw mirror)",
    "genetic": "pdftotext -layout of the genetic footprinting procedure PDF (raw mirror)",
    "assertion": "pdftotext -layout of the essentiality-assertion criteria PDF (raw "
    "mirror)",
    "pubmed": "PubMed's XML record of the article, abstract (raw mirror)",
}
Anchor = Literal["page", "genetic", "assertion", "pubmed"]


def _sv(
    anchor: Anchor, value: Any, quote: str, *, note: str | None = None
) -> SourcedValue:
    """A value bound to a verbatim quote in one sha256-pinned raw-mirror text."""
    relpath, sha256 = _ANCHORS[anchor]
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=relpath,
            citation_key=CITATION_KEY,
            sha256=sha256,
            method=_ANCHOR_METHODS[anchor],
        ),
    )


# --------------------------------------------------------------------------- #
# Sourced values (verbatim quotes, pinned sha256)
# --------------------------------------------------------------------------- #
_CONDITIONS_QUOTE = (
    "Throughout the fermentation temperature was held at 37o C, dissolved oxygen at "
    "30-50% of"
)
_OUTGROWTH_QUOTE = (
    "culture for 23 population doublings (12 hrs) to a cell density of 1.4x109."
)
_MEDIUM_QUOTES: dict[str, str] = {
    "base_1": "tryptone 10 g/L, yeast extract 5 g/L, NaCl 50 mM, NH4Cl 9.5 mM, MgCl2 "
    "0.528 mM, K2SO4 0.276",
    "base_2": "mM, FeSO4 0.01 mM, CaCl2 5x10-4 mM, and K2HPO4 1.32 mM; supplemented with "
    "micronutrients:",
    "trace_1": "(NH4)6(MoO7)24 3x10-6 mM, H3BO3 4x10-4 mM, CoCl2 3x10-5 mM, CuSO4 10-5 "
    "mM, MnCl2 8x10-5",
    "trace_2": "mM, and ZnSO4 10-5 mM (Neidhard et al., 1974). The following vitamins "
    "were added (mg/L): biotin",
    "vitamins_1": "0.12, riboflavin 0.8, pantothenic acid 10.8, niacinamide 12.0, "
    "pyridoxine 2.8, thiamine 4.0, lipoic",
    "vitamins_2": "acid 2.0, folic acid 0.08, and p-aminobenzoic acid 1.37. Kanamycin was "
    "added to 10 µg/ml.",
}

SOURCED_VALUES: dict[str, SourcedValue] = {
    "background_strain": _sv(
        "genetic",
        "MG1655",
        "Strain. Escherichia coli strain MG1655 (F- λ- ilvG rfb50 rph1) (Jensen, 1993) "
        "was used.",
    ),
    "transposon": _sv(
        "genetic",
        "EZ::TN<KAN-2>",
        "artificial transposon EZ::TN<KAN-2> (Epicentre Technologies, Madison, WI)",
    ),
    "temperature_c": _sv("genetic", 37.0, _CONDITIONS_QUOTE),
    "aerobicity": _sv(
        "page",
        "aerobic",
        "under conditions of aerobic logarithmic cell growth in rich medium.",
        note="and the fermentor held dissolved oxygen at 30-50% of saturation",
    ),
    "ph": _sv(
        "genetic",
        6.95,
        "saturation, and the pH at 6.95 (via titration with 5% H3PO4).",
        note="no Environment field holds a controlled pH; recorded, not stored",
    ),
    "generations": _sv("genetic", 23.0, _OUTGROWTH_QUOTE),
    "hours": _sv("genetic", 12.0, _OUTGROWTH_QUOTE),
    "lb_based": _sv(
        "genetic",
        "LB-based rich medium; the recovery medium is it without kanamycin",
        "Cultures were immediately diluted with an LB-based rich\nmedia (see below) "
        "without kanamycin",
        note="base_medium 'LB'; the reference's medium",
    ),
    **{f"medium_{k}": _sv("genetic", k, q) for k, q in _MEDIUM_QUOTES.items()},
    "call_essential": _sv(
        "page",
        "E",
        "= essential.  Includes genes with no detectable insertions within their coding",
    ),
    "call_non_essential": _sv(
        "page",
        "N",
        "= non-essential.  Genes with one or more insertions located within the central "
        "5 to",
    ),
    "call_not_covered": _sv("page", "X", "= Indicates the gene was not covered."),
    "call_ambiguous": _sv("page", "?", "= ambiguous."),
    "n_star_is_an_n_call": _sv(
        "page",
        "N*",
        "<i>accD</i> is called nonessential and <i>pdxJ</i> is called essential "
        "regardless of convincing arguments to the",
        note="accD's Table S1 call is 'N*'; the asterisk is not defined on the page",
    ),
    "scoring_rule": _sv(
        "assertion",
        "over 240 bp and free of inserts in a covered region, or a few inserts only in "
        "the 3'-most 20% or 5'-most 5%",
        "All ORFs longer then 240 base pairs in length (>80 AA) free\nfrom inserts were "
        "asserted as essential, if located in regions for which reliable PCR data "
        "were\nobtained. In addition, genes with only a few insertions within the "
        "3’-most 20% or 5’-most 5% of the\ngene were asserted as essential.",
        note="the rule every E call follows; carried on every record's evidence (#924)",
    ),
    "call_definition": _sv(
        "assertion",
        "essential under these growth conditions",
        "Failure to recover inserts, or the presence of only a limited number of "
        "inserts at the very ends of a\ncoding sequence, suggests that the gene is "
        "essential under these growth conditions.",
        note="what is_essential=True means here (#924)",
    ),
    "calls_uncorrected": _sv(
        "page",
        "calls from the footprinting data alone",
        "For consistency all essentiality calls in Table S1 are based\nexclusively on "
        "our experimental data, without any corrections by context or otherwise.",
        note="the page's caveat on its own calls; carried on every record (#924)",
    ),
    "assertion_error": _sv(
        "page",
        "p_o(L) = exp(-rL)",
        "A numerical measure of confidence in an assertion was calculated for each "
        'essential ORF (column "assertion error").\nAssertion error shows the '
        "probability of missing an ORF by chance if insert locations were completely "
        "random.",
        note="the 'assertion error' column; the formula follows on the page "
        "(p_o(L) = exp(-rL), r the local insertion density, L the ORF length in bp)",
    ),
    "insertion_coordinates": _sv(
        "page",
        "amino-acid coordinates from the translational start",
        "within each ORF and their locations (in amino acid coordinates) relative to "
        "the translational start.",
    ),
    "essential_rule": _sv(
        "assertion",
        "ORFs over 240 bp free of inserts, or inserts only at the ends",
        "ORFs asserted as essential (E). All ORFs longer then 240 base pairs in length "
        "(>80 AA) free",
    ),
    "abstract_counts": _sv(
        "pubmed",
        {"essential": 620, "dispensable": 3126},
        "We identified 620 genes as essential and 3,126 genes as dispensable for growth "
        "under these conditions.",
        note="Table S1 calls 619 E and 3,120 N or N*; recorded, not resolved",
    ),
    "overview_unambiguous": _sv(
        "genetic",
        3746,
        "Unambiguous (non)essentiality assertions made for 3,746 (87%) E. coli ORFs",
    ),
}

BACKGROUND_STRAIN: Literal["MG1655"] = "MG1655"
if SOURCED_VALUES["background_strain"].value != BACKGROUND_STRAIN:
    raise RuntimeError("the sourced background strain is not the loader's strain")
TRANSPOSON: str = SOURCED_VALUES["transposon"].value
TEMPERATURE_C: float = SOURCED_VALUES["temperature_c"].value
DURATION_GENERATIONS: float = SOURCED_VALUES["generations"].value
DURATION_HOURS: float = SOURCED_VALUES["hours"].value
KANAMYCIN_UG_PER_ML = 10.0
AEROBICITY: str = SOURCED_VALUES["aerobicity"].value
ABSTRACT_COUNTS: dict[str, int] = SOURCED_VALUES["abstract_counts"].value
OVERVIEW_UNAMBIGUOUS: int = SOURCED_VALUES["overview_unambiguous"].value
#: The statements the medium cites, bound at import so a rebuilt environment never
#: depends on the audited dict.
MEDIUM_STATEMENTS: dict[str, SourcedValue] = {
    key: SOURCED_VALUES[f"medium_{key}"] for key in _MEDIUM_QUOTES
}
MEDIUM_PROVENANCE: tuple[SourcedValue, ...] = (
    SOURCED_VALUES["lb_based"],
    SOURCED_VALUES["temperature_c"],
    SOURCED_VALUES["ph"],
)
#: Table S1's call histogram, measured on the pinned workbook (process() requires it).
TABLE_CALLS: dict[str, int] = {"?": 221, "E": 619, "N": 3114, "N*": 6, "X": 331}
#: Below this fraction of distinct Blattner ids resolving to one MG1655 locus the build
#: stops (checklist item 4); measured 4,167 of 4,209 (0.9900).
#: A well-formed MG1655 b-number as Table S1 prints it; ``b0018.1`` is not one.
B_NUMBER_PATTERN = re.compile(r"b\d{4}")
#: What ``systematic_gene_name`` is on each of the two record classes.
IdentifierKind = Literal["MG1655 b-number", "ERGO id"]
B_NUMBER_KIND: IdentifierKind = "MG1655 b-number"
ERGO_KIND: IdentifierKind = "ERGO id"
#: Every Table S1 row is a record: 4,208 under a b-number no other row carries, 83
#: under their ERGO id (64 with no Blattner ID, 1 with ``b0018.1``, 18 on 9 b-numbers
#: the table lists twice).
EXPECTED_RECORDS = 4291
EXPECTED_B_NUMBER_RECORDS = 4208
EXPECTED_ERGO_RECORDS = 83

PUBLICATION = Publication(
    pubmed_id=PUBMED_ID,
    pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{PUBMED_ID}/",
    doi=PAPER_DOI,
    doi_url=f"https://doi.org/{PAPER_DOI}",
)

_TABLE_LOOKED_IN = Provenance(
    source_uri=f"{RAW_DIR_REL}/{TABLE_S1_REL}",
    citation_key=CITATION_KEY,
    sha256=TABLE_S1_SHA256,
    method="every column of Table S1",
)
#: Typed absences of the transposon leaf's coordinate fields; the leaf has no
#: ``provenance_gaps`` slot, so they live here and in
#: ``preprocess/perturbation_field_gaps.json``, and every record's value is None.
PERTURBATION_FIELD_GAPS: tuple[ProvenanceGap, ...] = tuple(
    ProvenanceGap(
        field=field,
        reason=ProvenanceGapReason.not_reported_by_primary,
        looked_in=_TABLE_LOOKED_IN,
        note="a record is one GENE, standing for every insertion in it after outgrowth; "
        "Table S1 lists the insertions in amino-acid coordinates of the gene, with no "
        "strand and no genome position, and an essential gene has none",
    )
    for field in ("insertion_position", "insertion_strand")
)


#: The statements every record's evidence quotes (#924).
SCORING_RULE_STATEMENT = SOURCED_VALUES["scoring_rule"]
CALL_DEFINITION_STATEMENT = SOURCED_VALUES["call_definition"]
CALLS_UNCORRECTED_STATEMENT = SOURCED_VALUES["calls_uncorrected"]
ASSERTION_ERROR_STATEMENT = SOURCED_VALUES["assertion_error"]
INSERTION_COORDINATES_STATEMENT = SOURCED_VALUES["insertion_coordinates"]
#: The source's statements every record's evidence carries, by role (#924).
EVIDENCE_STATEMENTS: tuple[EssentialityStatement, ...] = (
    EssentialityStatement(
        role=EssentialityStatementRole.scoring_rule, provenance=SCORING_RULE_STATEMENT
    ),
    EssentialityStatement(
        role=EssentialityStatementRole.call_definition,
        provenance=CALL_DEFINITION_STATEMENT,
    ),
    EssentialityStatement(
        role=EssentialityStatementRole.caveat, provenance=CALLS_UNCORRECTED_STATEMENT
    ),
    EssentialityStatement(
        role=EssentialityStatementRole.statistic_definition,
        provenance=ASSERTION_ERROR_STATEMENT,
    ),
    EssentialityStatement(
        role=EssentialityStatementRole.statistic_definition,
        provenance=INSERTION_COORDINATES_STATEMENT,
    ),
)


# --------------------------------------------------------------------------- #
# Raw mirror (the loader reads the mirror, never a live URL)
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """``DATA_ROOT`` from the environment (the mirrors and the build tree live under it)."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/gerdesExperimentalDeterminationSystem2003``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def load_manifest(data_root: str | None = None) -> Manifest:
    """Read the raw mirror's ``manifest.json``."""
    path = raw_mirror_dir(data_root) / "manifest.json"
    return Manifest.model_validate_json(path.read_text())


def manifest_sha256(manifest: Manifest, relpath: str) -> str:
    """The recorded sha256 of one mirror file."""
    for record in manifest.files:
        if record.path == relpath:
            return record.sha256
    raise KeyError(f"{relpath} is not in the raw-mirror manifest")


# --------------------------------------------------------------------------- #
# Parsing Table S1
# --------------------------------------------------------------------------- #
class TableFormatError(ValueError):
    """Table S1 is not the sheet, header or cell shapes the loader reads."""


class CallCountError(ValueError):
    """Table S1's call histogram is not the pinned one."""


TABLE_S1_SHEET = "EC_from_SP_Mark"
TABLE_S1_HEADER: tuple[str, ...] = (
    "start",
    "",
    "gene",
    "E/N",
    "length",
    "transposon insertions",
    "assertion error",
    "SP_ID",
    "fucntional role",
    "Blattner ID",
    "ERGO",
    "scale: 0 - 1",
)
Call = Literal["E", "N", "N*", "X", "?"]
#: The two calls that state no boolean, each a typed absence on ``is_essential``.
CALL_GAPS: dict[str, ProvenanceGap] = {
    "X": ProvenanceGap(
        field="is_essential",
        reason=ProvenanceGapReason.not_reported_by_primary,
        note="Table S1 calls the ORF X, which the authors' column comments define as "
        "'the gene was not covered': no insertion data, so no essentiality call",
    ),
    "?": ProvenanceGap(
        field="is_essential",
        reason=ProvenanceGapReason.not_reported_by_primary,
        note="Table S1 calls the ORF ?, which the authors' column comments define as "
        "'ambiguous': the insertion pattern supports no call either way",
    ),
}
CALLS: tuple[Call, ...] = ("E", "N", "N*", "X", "?")


class GerdesRow(BaseModel):
    """One Table S1 row: an ORF, its call, and the insertions behind it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    row: int
    ergo_id: str
    gene: str
    call: Call
    length_aa: int
    insertions_aa: tuple[int, ...]
    assertion_error: str
    blattner_id: str

    @property
    def is_essential(self) -> bool | None:
        """E is essential; N and N* are not; X and ? state no boolean (None)."""
        if self.call in CALL_GAPS:
            return None
        return self.call == "E"


def _insertions(value: Any, where: str) -> tuple[int, ...]:
    """The insertion-position cell: '-' (none), one number, or a comma list."""
    if value == "-":
        return ()
    if isinstance(value, float):
        if not value.is_integer():
            raise TableFormatError(f"{where}: insertion {value!r}")
        return (int(value),)
    if not isinstance(value, str):
        raise TableFormatError(f"{where}: insertions {value!r}")
    parts = [p.strip() for p in value.split(",")]
    if not all(p.isdigit() for p in parts):
        raise TableFormatError(f"{where}: insertions {value!r}")
    return tuple(int(p) for p in parts)


def read_table_s1(path: str | Path) -> list[GerdesRow]:
    """Every data row of Table S1, refusing an unexpected sheet, header or cell."""
    book = xlrd.open_workbook(str(path))
    if book.sheet_names() != [TABLE_S1_SHEET]:
        raise TableFormatError(f"Table S1 sheets {book.sheet_names()}")
    sheet = book.sheet_by_index(0)
    header = tuple(sheet.cell_value(2, c) for c in range(sheet.ncols))
    if header != TABLE_S1_HEADER:
        raise TableFormatError(f"Table S1 header {header}")
    rows: list[GerdesRow] = []
    for r in range(3, sheet.nrows):
        where = f"Table S1 sheet row {r + 1}"
        cells = [sheet.cell_value(r, c) for c in range(sheet.ncols)]
        call = cells[3]
        if call not in CALLS:
            raise TableFormatError(f"{where}: call {call!r}")
        length = cells[4]
        if not isinstance(length, float) or not length.is_integer():
            raise TableFormatError(f"{where}: length {length!r}")
        blattner = cells[9]
        if not isinstance(blattner, str) or blattner != blattner.strip():
            raise TableFormatError(f"{where}: Blattner ID {blattner!r}")
        rows.append(
            GerdesRow(
                row=r - 2,
                ergo_id=str(cells[1]),
                gene=str(cells[2]).strip(),
                call=call,
                length_aa=int(length),
                insertions_aa=_insertions(cells[5], where),
                assertion_error=str(cells[6]),
                blattner_id=blattner,
            )
        )
    return rows


def call_histogram(rows: Sequence[GerdesRow]) -> dict[str, int]:
    """``{call: rows}`` over the five calls, zeros included, in key order."""
    counts = Counter(r.call for r in rows)
    return {call: counts[call] for call in sorted(CALLS)}


# --------------------------------------------------------------------------- #
# Identifiers as released, and the retention ledger
# --------------------------------------------------------------------------- #
class RowIdentity(BaseModel):
    """One Table S1 row and the released identifier its record is stored under."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    gerdes: GerdesRow
    kind: IdentifierKind
    identifier: str


class DropRule(BaseModel):
    """One retention rule, the rows it removed, and why (none apply; the shape is kept
    so every bacterial ``dropped_records.json`` reads alike).
    """

    rule: str
    description: str
    n_records: int
    items: list[str] = []


class DropLog(BaseModel):
    """The retention ledger of one build: every Table S1 row is a record."""

    dataset: str
    source_rows: int
    released_calls: dict[str, int]
    kept_records: int
    kept_calls: dict[str, int]
    b_number_records: int
    ergo_records: int
    dropped_records: int
    rules: list[DropRule]
    notes: list[str]


class DuplicateRecordError(ValueError):
    """Two rows would be stored under one released identifier."""


def identify_rows(rows: Sequence[GerdesRow]) -> list[RowIdentity]:
    """Which released identifier each row is stored under.

    A well-formed b-number that no other row carries keys the row on the placed
    transposon leaf; every other row (no Blattner ID, a malformed one, or a b-number
    the table lists twice) keys on its ERGO id, the table's own unique row identifier.
    Nothing is looked up in an annotation.
    """
    counts = Counter(r.blattner_id for r in rows if r.blattner_id)
    out: list[RowIdentity] = []
    for row in rows:
        b = row.blattner_id
        if b and B_NUMBER_PATTERN.fullmatch(b) and counts[b] == 1:
            out.append(RowIdentity(gerdes=row, kind=B_NUMBER_KIND, identifier=b))
        else:
            out.append(RowIdentity(gerdes=row, kind=ERGO_KIND, identifier=row.ergo_id))
    held = Counter(i.identifier for i in out)
    twice = sorted(k for k, n in held.items() if n > 1)
    if twice:
        raise DuplicateRecordError(f"two rows on {twice[:5]}")
    return out


def build_drop_log(dataset_name: str, identities: Sequence[RowIdentity]) -> DropLog:
    """The ledger of one build: every row a record, split by identifier kind."""
    rows = [i.gerdes for i in identities]
    n_b = sum(1 for i in identities if i.kind == B_NUMBER_KIND)
    n_ergo = len(identities) - n_b
    ergo_items = [
        f"row {i.gerdes.row} {i.gerdes.ergo_id} {i.gerdes.gene} ({i.gerdes.call}): "
        f"Blattner ID {i.gerdes.blattner_id!r}"
        for i in identities
        if i.kind == ERGO_KIND
    ]
    return DropLog(
        dataset=dataset_name,
        source_rows=len(rows),
        released_calls=call_histogram(rows),
        kept_records=len(rows),
        kept_calls=call_histogram(rows),
        b_number_records=n_b,
        ergo_records=n_ergo,
        dropped_records=0,
        rules=[],
        notes=[
            "every Table S1 row is a record; no annotation is read and no row is "
            "dropped for a Blattner ID the MG1655 GenBank file does not carry",
            f"{n_b} rows store their released b-number verbatim on "
            "TransposonInsertionPerturbation (identifier_mapping None)",
            f"{n_ergo} rows store their ERGO id on "
            "BacterialReleasedTransposonPerturbation: no Blattner ID, a malformed one, "
            "or a b-number the table lists twice (the Blattner ID stays a column of "
            "calls.csv)",
            *ergo_items,
            "X and ? rows store is_essential None beside a typed ProvenanceGap, with "
            "the letter on the evidence; E, N and N* rows store the boolean and the "
            "letter",
        ],
    )


# --------------------------------------------------------------------------- #
# The fermentor medium, environments and records (pure, no files)
# --------------------------------------------------------------------------- #
_MM = ConcentrationUnit.millimolar
_GL = ConcentrationUnit.g_per_l
_UGML = ConcentrationUnit.ug_per_ml
_HYDRATION_NOTE = "the source writes the anhydrous formula with no hydration state"


class ComponentSpec(BaseModel):
    """One stated ingredient: the label, its role, the stated amount and its quote."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    role: MediaComponentRole
    value: float
    unit: ConcentrationUnit
    quote: str
    note: str | None = None


_S = MediaComponentRole
COMPONENTS: tuple[ComponentSpec, ...] = (
    ComponentSpec(label="sodium chloride", role=_S.bulk_salt, value=50.0, unit=_MM, quote="base_1"),
    ComponentSpec(label="ammonium chloride", role=_S.nitrogen_source, value=9.5, unit=_MM, quote="base_1"),
    ComponentSpec(label="magnesium chloride", role=_S.bulk_salt, value=0.528, unit=_MM, quote="base_1", note=_HYDRATION_NOTE),
    ComponentSpec(label="potassium sulfate", role=_S.bulk_salt, value=0.276, unit=_MM, quote="base_1"),
    ComponentSpec(label="iron(II) sulfate", role=_S.trace_element, value=0.01, unit=_MM, quote="base_2", note=_HYDRATION_NOTE),
    ComponentSpec(label="calcium chloride", role=_S.bulk_salt, value=5e-4, unit=_MM, quote="base_2", note="the rendering writes '5x10-4' for 5 x 10^-4"),
    ComponentSpec(label="dipotassium hydrogen phosphate", role=_S.bulk_salt, value=1.32, unit=_MM, quote="base_2"),
    ComponentSpec(label="ammonium molybdate", role=_S.trace_element, value=3e-6, unit=_MM, quote="trace_1", note="the source writes '(NH4)6(MoO7)24', the Neidhardt 1974 ammonium heptamolybdate it cites, with no hydration state"),
    ComponentSpec(label="boric acid", role=_S.trace_element, value=4e-4, unit=_MM, quote="trace_1"),
    ComponentSpec(label="cobalt chloride", role=_S.trace_element, value=3e-5, unit=_MM, quote="trace_1", note=_HYDRATION_NOTE),
    ComponentSpec(label="copper sulfate", role=_S.trace_element, value=1e-5, unit=_MM, quote="trace_1", note="'CuSO4 10-5 mM' is 10^-5 mM; " + _HYDRATION_NOTE),
    ComponentSpec(label="manganese chloride", role=_S.trace_element, value=8e-5, unit=_MM, quote="trace_1", note=_HYDRATION_NOTE),
    ComponentSpec(label="zinc sulfate", role=_S.trace_element, value=1e-5, unit=_MM, quote="trace_2", note="'ZnSO4 10-5 mM' is 10^-5 mM; " + _HYDRATION_NOTE),
    ComponentSpec(label="biotin", role=_S.vitamin, value=0.12, unit=_UGML, quote="trace_2", note="stated in mg/L, which is ug/mL"),
    ComponentSpec(label="riboflavin", role=_S.vitamin, value=0.8, unit=_UGML, quote="vitamins_1", note="stated in mg/L, which is ug/mL"),
    ComponentSpec(label="pantothenic acid", role=_S.vitamin, value=10.8, unit=_UGML, quote="vitamins_1", note="stated in mg/L, which is ug/mL"),
    ComponentSpec(label="niacinamide", role=_S.vitamin, value=12.0, unit=_UGML, quote="vitamins_1", note="stated in mg/L, which is ug/mL"),
    ComponentSpec(label="pyridoxine", role=_S.vitamin, value=2.8, unit=_UGML, quote="vitamins_1", note="stated in mg/L, which is ug/mL"),
    ComponentSpec(label="thiamine", role=_S.vitamin, value=4.0, unit=_UGML, quote="vitamins_1", note="stated in mg/L, which is ug/mL"),
    ComponentSpec(label="lipoic acid", role=_S.vitamin, value=2.0, unit=_UGML, quote="vitamins_2", note="stated in mg/L, which is ug/mL"),
    ComponentSpec(label="folic acid", role=_S.vitamin, value=0.08, unit=_UGML, quote="vitamins_2", note="stated in mg/L, which is ug/mL"),
    ComponentSpec(label="p-aminobenzoic acid", role=_S.vitamin, value=1.37, unit=_UGML, quote="vitamins_2", note="stated in mg/L, which is ug/mL"),
)  # fmt: skip


def _complex(name: str, grams_per_liter: float) -> MediaComponent:
    """Tryptone or yeast extract: a digest with no single structure."""
    return MediaComponent(
        compound=Compound(name=name),
        role=MediaComponentRole.complex_ingredient,
        concentration=Concentration(value=grams_per_liter, unit=_GL),
        definition=ComponentDefinition.intrinsically_undefined,
        provenance=[MEDIUM_STATEMENTS["base_1"]],
    )


def fermentor_medium(*, kanamycin: bool) -> Media:
    """The fermentor's LB-based rich medium, with or without its 10 ug/ml kanamycin."""
    components = [
        _complex("tryptone", 10.0),
        _complex("yeast extract", 5.0),
        *(
            MediaComponent(
                compound=resolved_compound(spec.label),
                role=spec.role,
                concentration=Concentration(value=spec.value, unit=spec.unit),
                provenance=[MEDIUM_STATEMENTS[spec.quote]],
                note=spec.note,
                defers_to=[],
            )
            for spec in COMPONENTS
        ),
    ]
    provenance = list(MEDIUM_PROVENANCE)
    name = (
        "LB-based rich fermentor medium with Neidhardt salts, micronutrients and "
        "vitamins (Gerdes 2003; pH held at 6.95)"
    )
    if not kanamycin:
        return Media(
            name=f"{name}, without kanamycin",
            state="liquid",
            is_synthetic=False,
            base_medium="LB",
            components=components,
            provenance=provenance,
        )
    components.append(
        MediaComponent(
            compound=resolved_compound("kanamycin"),
            role=MediaComponentRole.selection_agent,
            concentration=Concentration(value=KANAMYCIN_UG_PER_ML, unit=_UGML),
            provenance=[MEDIUM_STATEMENTS["vitamins_2"]],
            note="selects for the transposon's kanamycin marker",
        )
    )
    return Media(
        name=f"{name}, with 10 ug/ml kanamycin",
        state="liquid",
        is_synthetic=False,
        base_medium="LB",
        components=components,
        provenance=provenance,
    )


def environment(*, kanamycin: bool = True) -> Environment:
    """The fermentor outgrowth: 37 C, aerobic, 23 doublings in 12 hours."""
    return Environment(
        media=fermentor_medium(kanamycin=kanamycin),
        temperature=Temperature(value=TEMPERATURE_C),
        aerobicity=AEROBICITY,
        duration_hours=DURATION_HOURS,
        duration_generations=DURATION_GENERATIONS,
    )


# --- begin #924: the footprinting evidence on every record ------------------- #
#: Table S1's assertion-error cell when no value is given: ``n.d.``, or empty.
ASSERTION_ERROR_NOT_DETERMINED = "n.d."
#: A released value (``0.306647``) or bound (``< 0.05``).
_ASSERTION_ERROR_PATTERN = re.compile(r"(?P<bound>< )?(?P<number>\d+(\.\d+)?)")


class AssertionErrorCell(BaseModel):
    """One Table S1 assertion-error cell, read as released."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: float | None
    is_upper_bound: bool | None
    released: str


def read_assertion_error(text: str, where: str) -> AssertionErrorCell:
    """``"< 0.05"`` is a bound, ``"0.306647"`` a value, ``"n.d."`` or ``""`` none.

    Anything else raises: the cell's forms are the four the pinned table holds.
    """
    if text in ("", ASSERTION_ERROR_NOT_DETERMINED):
        return AssertionErrorCell(value=None, is_upper_bound=None, released=text)
    match = _ASSERTION_ERROR_PATTERN.fullmatch(text)
    if match is None:
        raise TableFormatError(f"{where}: assertion error {text!r}")
    bound = match.group("bound") is not None
    value = float(match.group("number"))
    if not 0 <= value <= 1:
        raise TableFormatError(f"{where}: assertion error {text!r} is no probability")
    return AssertionErrorCell(value=value, is_upper_bound=bound, released=text)


def assertion_error_gap(row: GerdesRow) -> ProvenanceGap:
    """The typed absence of an E row's assertion error (``n.d.`` or an empty cell)."""
    released = row.assertion_error
    note = (
        "Table S1 reads 'n.d.' for this E call's assertion error"
        if released == ASSERTION_ERROR_NOT_DETERMINED
        else "this E call's assertion-error cell is empty in Table S1"
    )
    return ProvenanceGap(
        field="assertion_error",
        reason=ProvenanceGapReason.not_reported_by_primary,
        looked_in=_TABLE_LOOKED_IN,
        note=note,
    )


def evidence(row: GerdesRow | None = None) -> GeneEssentialityEvidence:
    """The genetic-footprinting evidence (#924), with ``row``'s call when given.

    Per record: the released letter, the recovered insertions and the assertion error
    as released (value, bound, or a typed gap on an E call). The reference (``row``
    None) carries the type, rule, definition, caveat and definitions only.
    """
    stats: dict[str, Any] = {}
    gaps: list[ProvenanceGap] = []
    if row is not None:
        cell = read_assertion_error(row.assertion_error, f"row {row.row}")
        stats = {
            "footprinting_call": FootprintingCall(row.call),
            "footprint_insertions_aa": list(row.insertions_aa),
            "assertion_error": cell.value,
            "assertion_error_is_upper_bound": cell.is_upper_bound,
        }
        if cell.value is None and row.call == "E":
            gaps.append(assertion_error_gap(row))
    return GeneEssentialityEvidence(
        evidence_class=EssentialityEvidenceClass.insertion_depletion,
        evidence_type=EssentialityEvidenceType.genetic_footprinting,
        statements=list(EVIDENCE_STATEMENTS),
        provenance_gaps=gaps,
        **stats,
    )


# --- end #924 ---------------------------------------------------------------- #


def released_gene_name(identity: RowIdentity) -> str:
    """The gene name Table S1 prints, or the row's identifier where it prints none."""
    return identity.gerdes.gene or identity.identifier


def insertion_genotype(identity: RowIdentity) -> Genotype:
    """The gene-level transposon disruption of one row, under its released identifier."""
    namespace = STRAIN_GENE_NAMESPACES[BACKGROUND_STRAIN]
    if identity.kind == B_NUMBER_KIND:
        return Genotype(
            perturbations=[
                TransposonInsertionPerturbation(
                    systematic_gene_name=identity.identifier,
                    perturbed_gene_name=released_gene_name(identity),
                    gene_namespace=namespace,
                    identifier_mapping=None,
                    transposon=TRANSPOSON,
                )
            ]
        )
    return Genotype(
        perturbations=[
            BacterialReleasedTransposonPerturbation(
                systematic_gene_name=identity.identifier,
                perturbed_gene_name=released_gene_name(identity),
                gene_namespace=namespace,
                strain_label=identity.identifier,
                identifier_kind=ERGO_KIND,
                transposon=TRANSPOSON,
            )
        ]
    )


def essentiality_phenotype(row: GerdesRow) -> GeneEssentialityPhenotype:
    """The call as released: E True, N and N* False, X and ? a typed absence; every
    one with the genetic-footprinting evidence (#924), the letter on it.
    """
    gap = CALL_GAPS.get(row.call)
    return GeneEssentialityPhenotype(
        is_essential=row.is_essential,
        evidence=evidence(row),
        provenance_gaps=[] if gap is None else [gap],
    )


def build_experiment(
    dataset_name: str, identity: RowIdentity, env: Environment
) -> BacterialGeneEssentialityExperiment:
    """The record of one Table S1 row."""
    return BacterialGeneEssentialityExperiment(
        dataset_name=dataset_name,
        genotype=insertion_genotype(identity),
        environment=env,
        phenotype=essentiality_phenotype(identity.gerdes),
    )


def build_reference(
    dataset_name: str, genome_reference: AssemblyReferenceGenome
) -> BacterialGeneEssentialityExperimentReference:
    """The unperturbed MG1655 parent, viable, in the medium without the kanamycin."""
    return BacterialGeneEssentialityExperimentReference(
        dataset_name=dataset_name,
        genome_reference=genome_reference,
        environment_reference=environment(kanamycin=False),
        phenotype_reference=GeneEssentialityPhenotype(
            is_essential=False, evidence=evidence()
        ),
    )


# --------------------------------------------------------------------------- #
# The dataset
# --------------------------------------------------------------------------- #
@register_dataset
class GeneEssentialityGerdes2003Dataset(ExperimentDataset):
    """Genetic-footprinting essentiality calls of E. coli MG1655 genes (Table S1)."""

    organism = Organism.ecoli

    REFERENCE_STRAIN: ClassVar[EcoliK12StrainName] = "MG1655"

    def __init__(
        self,
        root: str = "data/torchcell/gene_essentiality_gerdes2003",
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; no genome is read, the released identifiers are stored."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return BacterialGeneEssentialityExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return BacterialGeneEssentialityExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """Table S1, linked from the raw mirror."""
        return [TABLE_S1]

    def download(self) -> None:
        """Link Table S1 into ``raw/`` after checking the manifest and sha256."""
        data_root = _data_root()
        manifest = load_manifest(data_root)
        check_manifest_pin(
            TABLE_S1_REL, manifest_sha256(manifest, TABLE_S1_REL), TABLE_S1_SHA256
        )
        os.makedirs(self.raw_dir, exist_ok=True)
        link_verified(
            raw_mirror_dir(data_root) / TABLE_S1_REL,
            osp.join(self.raw_dir, TABLE_S1),
            TABLE_S1_SHA256,
        )
        log.info("Gerdes 2003 Table S1 linked into %s (sha256 verified)", self.raw_dir)

    @post_process
    def process(self) -> None:
        """Parse Table S1, check its call histogram, resolve, write the LMDB."""
        verify_raw_files(self.raw_dir, DATA_SHA256)
        rows = read_table_s1(osp.join(self.raw_dir, TABLE_S1))
        calls = call_histogram(rows)
        if calls != TABLE_CALLS:
            raise CallCountError(f"Table S1 calls {calls}, pinned {TABLE_CALLS}")
        identities = identify_rows(rows)
        drop_log = build_drop_log(self.name, identities)
        if (
            drop_log.kept_records,
            drop_log.b_number_records,
            drop_log.ergo_records,
        ) != (EXPECTED_RECORDS, EXPECTED_B_NUMBER_RECORDS, EXPECTED_ERGO_RECORDS):
            raise CallCountError(
                f"{drop_log.kept_records} rows ({drop_log.b_number_records} b-number, "
                f"{drop_log.ergo_records} ERGO), the module declares "
                f"{EXPECTED_RECORDS} ({EXPECTED_B_NUMBER_RECORDS}, "
                f"{EXPECTED_ERGO_RECORDS})"
            )
        log.info(
            "Gerdes 2003: %d Table S1 rows -> %d records %s (%d under a b-number, %d "
            "under an ERGO id); nothing dropped",
            drop_log.source_rows,
            drop_log.kept_records,
            drop_log.kept_calls,
            drop_log.b_number_records,
            drop_log.ergo_records,
        )
        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        self._write_ledgers(drop_log, identities)

        env = environment()
        reference = build_reference(
            self.name, assembly_reference(self.REFERENCE_STRAIN)
        )
        env_out, interned_env = self._open_write_lmdb(
            osp.join(self.processed_dir, "lmdb")
        )
        with env_out.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for idx, identity in enumerate(tqdm(identities, desc="gerdes2003")):
                experiment = build_experiment(self.name, identity, env)
                txn.put(
                    f"{idx}".encode(),
                    self._intern_record(experiment, reference, PUBLICATION, itxn),
                )
        env_out.close()
        interned_env.close()
        log.info("Wrote %d Gerdes 2003 records to LMDB", len(identities))

    def _write_ledgers(
        self, drop_log: DropLog, identities: Sequence[RowIdentity]
    ) -> None:
        """The drop log, the transposon leaf's typed absences, and the per-row table
        with every released column, each row's LMDB index, its identifier kind and the
        stored call.
        """
        out = Path(self.preprocess_dir)
        (out / "dropped_records.json").write_text(drop_log.model_dump_json(indent=2))
        (out / "perturbation_field_gaps.json").write_text(
            json.dumps(
                [gap.model_dump(mode="json") for gap in PERTURBATION_FIELD_GAPS],
                indent=2,
            )
        )
        rows = [
            {
                "row": i.gerdes.row,
                "ergo_id": i.gerdes.ergo_id,
                "gene": i.gerdes.gene,
                "call": i.gerdes.call,
                "length_aa": i.gerdes.length_aa,
                "insertions_aa": " ".join(map(str, i.gerdes.insertions_aa)),
                "assertion_error": i.gerdes.assertion_error,
                "blattner_id": i.gerdes.blattner_id,
                "identifier_kind": i.kind,
                "systematic_gene_name": i.identifier,
                "is_essential": i.gerdes.is_essential,
                "record": index,
            }
            for index, i in enumerate(identities)
        ]
        pd.DataFrame(rows).astype({"record": "Int64"}).to_csv(
            out / "calls.csv", index=False
        )

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled by ``build_experiment``."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Verification (L0-L4) of a built LMDB
# --------------------------------------------------------------------------- #
VERIFIER_PROVENANCE = Provenance(
    source_uri=f"{RAW_DIR_REL}/{TABLE_S1_REL} (Supplementary Online Material Table S1)",
    citation_key=CITATION_KEY,
    sha256=TABLE_S1_SHA256,
    method="genetic-footprinting call per MG1655 ORF after 23 doublings: E = essential, "
    "N or N* = not; X and ? dropped; reference = the parent, viable",
    page="J Bacteriol 185:5673-5684, Table S1",
    retrieved=RAW_RETRIEVED_AT,
)

Record = Mapping[str, Any]


def _validate_record(record: Record) -> None:
    """L0: the experiment and its reference validate as the assembly-pinned pair."""
    BacterialGeneEssentialityExperiment.model_validate(record["experiment"])
    BacterialGeneEssentialityExperimentReference.model_validate(record["reference"])


def _perturbation(record: Record) -> Mapping[str, Any]:
    perturbations: list[Mapping[str, Any]] = record["experiment"]["genotype"][
        "perturbations"
    ]
    if len(perturbations) != 1:
        raise ValueError(f"{len(perturbations)} perturbations on one record")
    return perturbations[0]


def _l1_one_record_per_identifier(records: Sequence[Record]) -> LevelResult:
    ids = Counter(_perturbation(r)["systematic_gene_name"] for r in records)
    repeated = sorted(f"{g} x{n}" for g, n in ids.items() if n > 1)
    return LevelResult(
        level=Level.L1,
        name="one_record_per_released_identifier",
        passed=not repeated,
        message=f"{len(ids)} released identifiers, one record each"
        if not repeated
        else f"{len(repeated)} identifiers repeat",
        details={"repeated": repeated[:20]},
    )


def _l2_calls_match_table_s1(
    records: Sequence[Record], rows: Sequence[GerdesRow]
) -> LevelResult:
    """L2, independent of the build path: each record's released identifier finds one
    Table S1 row (a b-number row for the placed leaf, an ERGO-id row for the released
    leaf), and the stored call, token and gap are that row's.
    """
    by_b: dict[str, list[GerdesRow]] = {}
    by_ergo: dict[str, GerdesRow] = {}
    for row in rows:
        by_b.setdefault(row.blattner_id, []).append(row)
        by_ergo[row.ergo_id] = row
    problems: list[str] = []
    for record in records:
        p = _perturbation(record)
        identifier = p["systematic_gene_name"]
        if p["perturbation_type"] == "transposon_insertion":
            found = by_b.get(identifier, [])
            if len(found) != 1:
                problems.append(f"{identifier}: {len(found)} Table S1 rows")
                continue
            (row,) = found
        else:
            maybe = by_ergo.get(identifier)
            if maybe is None:
                problems.append(f"{identifier}: no Table S1 row")
                continue
            row = maybe
        phenotype = record["experiment"]["phenotype"]
        stored = phenotype["is_essential"]
        gapped = {gap["field"] for gap in phenotype["provenance_gaps"]}
        if stored != row.is_essential:
            problems.append(f"{identifier}: stored {stored}, call {row.call}")
        if (row.call in CALL_GAPS) != ("is_essential" in gapped):
            problems.append(f"{identifier}: gap mismatch for call {row.call}")
        ev = phenotype["evidence"]
        cell = read_assertion_error(row.assertion_error, identifier)
        held = (
            ev["footprinting_call"],
            tuple(ev["footprint_insertions_aa"]),
            ev["assertion_error"],
            ev["assertion_error_is_upper_bound"],
        )
        if held != (row.call, row.insertions_aa, cell.value, cell.is_upper_bound):
            problems.append(f"{identifier}: evidence {held}")
    return LevelResult(
        level=Level.L2,
        name="calls_match_table_s1",
        passed=not problems,
        message=f"{len(records)} records equal their Table S1 row's call and "
        "identifier, and carry its letter, insertions and assertion error as evidence"
        if not problems
        else f"{len(problems)} records disagree with Table S1",
        details={"n_problems": len(problems), "problems": problems[:20]},
    )


def _l3_call_histogram(rows: Sequence[GerdesRow]) -> LevelResult:
    calls = call_histogram(rows)
    paper = ABSTRACT_COUNTS
    return LevelResult(
        level=Level.L3,
        name="table_s1_call_histogram_is_pinned",
        passed=calls == TABLE_CALLS,
        message=f"Table S1 {calls}; the abstract states {paper} and the overview "
        f"{OVERVIEW_UNAMBIGUOUS} unambiguous (the table holds "
        f"{calls['E'] + calls['N'] + calls['N*']})",
        details={"table": calls, "abstract": paper},
    )


def _l3_reference_is_the_parent_off_selection(records: Sequence[Record]) -> LevelResult:
    expected = environment(kanamycin=False).model_dump()
    wrong = sum(
        1
        for r in records
        if r["reference"]["environment_reference"] != expected
        or r["reference"]["phenotype_reference"]["is_essential"] is not False
    )
    return LevelResult(
        level=Level.L3,
        name="reference_is_the_viable_parent_without_kanamycin",
        passed=not wrong,
        message=f"{len(records) - wrong} of {len(records)} references are the viable "
        "parent in the medium without the selection agent",
        details={"n_wrong": wrong},
    )


#: The per-record evidence fields; the rest is shared with the reference.
_RECORD_EVIDENCE_FIELDS = frozenset(
    {
        "footprinting_call",
        "footprint_insertions_aa",
        "assertion_error",
        "assertion_error_is_upper_bound",
        "provenance_gaps",
    }
)


def _l3_evidence_is_genetic_footprinting(records: Sequence[Record]) -> LevelResult:
    """L3 (#924): every record and reference carries the genetic-footprinting
    evidence with the sourced rule, definition, caveat and definitions; the reference
    has no per-record values.
    """
    reference = evidence().model_dump()
    shared = {k: v for k, v in reference.items() if k not in _RECORD_EVIDENCE_FIELDS}
    wrong = 0
    gapped = 0
    for record in records:
        ev = record["experiment"]["phenotype"]["evidence"]
        gapped += bool(ev["provenance_gaps"])
        if {k: v for k, v in ev.items() if k not in _RECORD_EVIDENCE_FIELDS} != shared:
            wrong += 1
        elif record["reference"]["phenotype_reference"]["evidence"] != reference:
            wrong += 1
    return LevelResult(
        level=Level.L3,
        name="evidence_is_genetic_footprinting",
        passed=not wrong,
        message=f"{len(records) - wrong} of {len(records)} records and references carry "
        "genetic_footprinting evidence (insertion_depletion) with the sourced rule, "
        f"definition and caveat; {gapped} E calls gap their assertion error",
        details={"n_wrong": wrong, "n_assertion_error_gaps": gapped},
    )


def _l4_containment(records: Sequence[Record], universe: set[str]) -> LevelResult:
    """L4 FINDING, never a gate: how many released b-numbers are MG1655 GenBank gene
    rows of the pinned assembly, and how many ERGO-keyed rows there are (which no
    universe can contain). A b-number the annotation lacks is still a released call.
    """
    placed = sorted(
        {
            _perturbation(r)["systematic_gene_name"]
            for r in records
            if _perturbation(r)["perturbation_type"] == "transposon_insertion"
        }
    )
    n_ergo = len(records) - sum(
        1
        for r in records
        if _perturbation(r)["perturbation_type"] == "transposon_insertion"
    )
    missing = sorted(set(placed) - universe)
    return LevelResult(
        level=Level.L4,
        name="gene_containment_mg1655_locus_tags",
        passed=True,
        message=f"{len(placed) - len(missing)} of {len(placed)} released b-numbers are "
        f"MG1655 GenBank gene rows; {len(missing)} are not, and {n_ergo} records are "
        "keyed by ERGO id (a finding, not a gate)",
        details={
            "n_b_numbers": len(placed),
            "n_universe": len(universe),
            "n_ergo_records": n_ergo,
            "missing_examples": missing[:50],
        },
    )


def verify_records(
    records: Sequence[Record],
    *,
    rows: Sequence[GerdesRow],
    universe: set[str],
    expected_count: int,
    dataset_name: str = "gene_essentiality_gerdes2003",
) -> VerificationReport:
    """The L0-L3 gate over built records plus the L4 finding, given the re-read Table
    S1 and the MG1655 gene universe (read for the finding only).
    """
    from torchcell.verification.common import shared_rule_results

    report = VerificationReport(
        dataset_name=dataset_name, provenance=VERIFIER_PROVENANCE
    )
    report.add(l0_structural(records, _validate_record))
    report.add(l1_count(len(records), expected_count))
    report.add(_l1_one_record_per_identifier(records))
    report.add(_l2_calls_match_table_s1(records, rows))
    report.add(_l3_call_histogram(rows))
    report.add(_l3_reference_is_the_parent_off_selection(records))
    report.add(_l3_evidence_is_genetic_footprinting(records))
    for result in shared_rule_results(records):
        report.add(result)
    report.add(_l4_containment(records, universe))
    return report


def run_verification(data_root: str | None = None) -> VerificationReport:
    """Verify the built dev-tree LMDB (L0-L4) plus the provenance audit of every
    ``SOURCED_VALUES`` entry, and write ``preprocess/verification_report.json``.
    """
    from torchcell.verification.runners import (
        _gene_set_for_reference,
        _write_report,
        load_records,
    )

    base = data_root or _data_root()
    abs_root = osp.join(base, DATASET_ROOT_REL)
    records = load_records(abs_root)
    drops = DropLog.model_validate_json(
        Path(abs_root, "preprocess", "dropped_records.json").read_text()
    )
    rows = read_table_s1(osp.join(abs_root, "raw", TABLE_S1))
    references = {
        json.dumps(r["reference"]["genome_reference"], sort_keys=True) for r in records
    }
    universe: set[str] = set()
    for ref in references:
        universe |= _gene_set_for_reference(json.loads(ref), base)
    report = verify_records(
        records, rows=rows, universe=universe, expected_count=drops.kept_records
    )
    mirror = Path(base) / "torchcell-raw"
    for value in SOURCED_VALUES.values():
        report.add(audit_sourced_value(value, mirror))
    _write_report(report, osp.join(abs_root, "preprocess"))
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI: ``build`` the dev LMDB or ``verify`` it."""
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(
        prog="python -m torchcell.datasets.ecoli.gerdes2003"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help="build (or load) the dev-tree LMDB")
    sub.add_parser("verify", help="run L0-L4 on the built dev-tree LMDB")
    args = parser.parse_args(argv)

    load_dotenv()
    data_root = _data_root()
    if args.command == "build":
        dataset = GeneEssentialityGerdes2003Dataset(
            root=osp.join(data_root, DATASET_ROOT_REL)
        )
        print(f"len = {len(dataset)}")
        return 0
    report = run_verification(data_root)
    print(report.summary())
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
