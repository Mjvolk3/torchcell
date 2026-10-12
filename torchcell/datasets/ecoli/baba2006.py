# torchcell/datasets/ecoli/baba2006
# [[torchcell.datasets.ecoli.baba2006]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/baba2006
# Test file: tests/torchcell/datasets/ecoli/test_baba2006.py
"""Baba 2006 Keio construction outcomes: which E. coli BW25113 ORFs could be deleted.

Baba, Ara, Hasegawa, Takai, Okumura, Baba, Datsenko, Tomita, Wanner and Mori 2006 (Mol
Syst Biol 2:2006.0008, doi:10.1038/msb4100050, citation key
``babaConstructionEscherichiaColi2006``) targeted 4,288 ORFs of E. coli K-12 BW25113
for in-frame replacement by a kanamycin cassette and kept the mutants with the expected
structure: the Keio collection. 3,985 ORFs yielded a mutant; the 303 that did not are
the paper's essential gene candidates.

DATA. Supplementary Tables 2, 3 and 6 of the PMC Article Datasets bucket
(``PMC1681482.1/msb4100050-s4.xls``, ``-s5.xls``, ``-s8.xls``), deposited with PMC's
plain text of the article in ``$DATA_ROOT/torchcell-raw/babaConstructionEscherichiaColi2006/``
by ``experiments/036-dataset-fixes-before-kg-build/scripts/baba2006_release_inventory.py``.
The literature mirror's ``si/si1.pdf`` for this key is a second copy of the article,
not the supplement. Table 3 lists every ORF with the PCR verification of its KmR
colonies (``5. verified`` of ``6. colonies tested``, ``7. percent``); Table 2 lists the
4,288 targeted ORFs; Table 6 the 303 candidates.

THE CALL. The paper's rule, applied to Table 3: "Mutants were scored as correct if two
or more colonies had the expected structure" (``SOURCED_VALUES["scoring_rule"]``). Two or
more verified colonies is an obtained deletion (``is_essential=False``); fewer is none
obtained (``is_essential=True``). Measured on the release: the rule splits the 4,288
targeted ORFs into exactly 3,985 and 303, and the 303 are exactly Table 6's rows (302
with no verified colony, plus secM with one of eight, the exception the paper explains
by a suppressor). process() refuses a release where any of that stops holding. What
``is_essential=True`` means here is a CONSTRUCTION outcome on selection plates, a
"candidate" in the paper's own word, not a growth measurement.

ROWS WITH NO CALL. Table 3's 90 ``N.A.`` rows are ORFs the paper did not target
(insertion-sequence genes, the ``ldr`` toxins, the araBAD/lacZ/rhaBAD genes BW25113
already lacks, re-annotated ORFs); no construction was attempted, so they have no call
and no record (``not_targeted``).

STRAIN AND IDENTIFIERS. BW25113 (``SOURCED_VALUES["background_strain"]``). Each ORF's
identity is its Keio JW id, which the pinned BW25113 GenBank file carries as a
``gene_synonym``; it goes through ``reconcile_locus_tags`` and the record stores the
derived ``jw_synonym`` mapping. A JW id that is not on exactly one BW25113 locus is
dropped (``jw_id_not_on_one_bw25113_locus``, measured 172: 125 the annotation does not
carry, 47 sharing a locus with another JW id). No b-number crosswalk is used.

GENOTYPE. One ``BacterialDeletionPerturbation`` per record, cassette "kanamycin
cassette". ``collection`` is "Keio collection" only for an obtained deletion: a deletion
that was never obtained is in no collection, and a record claiming one would name a
strain that does not exist.

ENVIRONMENT. The selection plates the call was made on: LB agar with kanamycin at 30
ug/ml, incubated aerobically at 37 C. The LB is the paper's own (1% tryptone, 0.5% yeast
extract, 0.5% NaCl), which are the ``LB_LENNOX`` amounts; the agar amount and the
incubation time are not stated (agar ``concentration=None``, a typed duration gap).

REFERENCE. The unperturbed BW25113 parent, viable, on the same LB agar WITHOUT the
kanamycin: the parent carries no kanamycin marker, so it is not viable on the selection
plates, and the paper grows it "with or without antibiotics". The reference therefore
states the parent's viability in the medium the selection agent was added to.

NOT STORED. The OD600 growth of the obtained mutants (Table 3, LB 22 h and MOPS 24 h /
48 h) is a separate measurement with no wild-type value in the release; it is not part
of this dataset. Table 3's re-served Gerdes 2003 / PEC / Tn5 columns are other studies'
calls.

EVIDENCE (#924). Every phenotype carries a ``GeneEssentialityEvidence`` of type
``construction_outcome`` (class ``mutant_recovery``): the scoring rule, the paper's
"candidates for essential genes" as the call's definition, its own caveat on population
methods, the definitions of the colony counts, and the record's Table 3 PCR counts
(verified, tested, percent). The reference carries the same evidence without counts.
``preprocess/calls.csv`` keeps the per-row table, untargeted and dropped rows included.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import os.path as osp
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
from torchcell.datamodels.media import LB_LENNOX
from torchcell.datamodels.schema import (
    BACTERIAL_ASSEMBLY_SETS,
    AssemblyReferenceGenome,
    BacterialDeletionPerturbation,
    BacterialGeneEssentialityExperiment,
    BacterialGeneEssentialityExperimentReference,
    Concentration,
    ConcentrationUnit,
    DerivedIdentifierMapping,
    Environment,
    EssentialityEvidenceClass,
    EssentialityEvidenceType,
    EssentialityStatement,
    EssentialityStatementRole,
    Experiment,
    ExperimentReference,
    GeneEssentialityEvidence,
    GeneEssentialityPhenotype,
    Genotype,
    Media,
    MediaComponent,
    MediaComponentRole,
    Publication,
    Temperature,
)
from torchcell.datasets.bacteria_common import (
    LOCUS_TAG_PATTERNS,
    STRAIN_GENE_NAMESPACES,
    LocusTagReconciliation,
    assembly_reference,
    bacterial_genome,
    reconcile_locus_tags,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.literature.manifest import Manifest
from torchcell.sequence.genome.base import GeneNameResolution
from torchcell.sequence.genome.ecoli.k12 import EcoliK12Genome, EcoliK12StrainName
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
CITATION_KEY = "babaConstructionEscherichiaColi2006"
PAPER_DOI = "10.1038/msb4100050"
PMCID = "PMC1681482"
#: Read from the PMC id converter (``pmc.ncbi.nlm.nih.gov/tools/idconv``) on 2026-10-10.
PUBMED_ID = "16738554"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"
DATASET_ROOT_REL = "data/torchcell/gene_essentiality_baba2006"
RAW_RETRIEVED_AT = "2026-10-10"

TABLE_S2 = "msb4100050-s4.xls"
TABLE_S3 = "msb4100050-s5.xls"
TABLE_S6 = "msb4100050-s8.xls"
PAPER_TEXT_REL = "paper/PMC1681482.1.txt"
PAPER_TEXT_SHA256 = "6e39bc2f12aecceac568f91dfcd068e109433132df618dc68624d317737bf3cd"

#: ``{raw file name: pinned sha256}``, the build-time check of every consumed table.
DATA_SHA256: dict[str, str] = {
    TABLE_S2: "897599a5401367356aa2823ed939524561894a9120ef86bad923200b56770bac",
    TABLE_S3: "a1a9c30b6cd703ff835f2809b420afe38b1c139865c9af87566a410669a30d2e",
    TABLE_S6: "ded72d1f890d47b0544fa52edcb491b187a010fa6b938437fc04418e08e31638",
}


def mirror_relpath(name: str) -> str:
    """Path of one consumed table inside the raw mirror (``data/<name>``)."""
    return f"data/{name}"


def _paper(value: Any, quote: str, *, note: str | None = None) -> SourcedValue:
    """A value bound to a verbatim quote in PMC's sha256-pinned plain text."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=PAPER_TEXT_REL,
            citation_key=CITATION_KEY,
            sha256=PAPER_TEXT_SHA256,
            method="PMC Article Datasets plain text of the article (raw mirror)",
        ),
    )


# --------------------------------------------------------------------------- #
# Sourced values (verbatim quotes, pinned sha256)
# --------------------------------------------------------------------------- #
_COUNTS_QUOTE = (
    "Of 4288 genes targeted, deletions were obtained for 3985 ORFs (Supplementary "
    "Table 3). Based on finding mutants with the predicted structure, these 3985 genes "
    "are (probably) nonessential, while the 303 genes (including 37 genes of unknown "
    "function), for which no mutants were found, are candidates for essential genes"
)
_SELECTION_QUOTE = (
    "Our standard protocol usually yielded 10–1000 KmR colonies when cells were "
    "incubated aerobically at 37°C on Luria broth (LB) agar containing 30 μg/ml "
    "kanamycin."
)

SOURCED_VALUES: dict[str, SourcedValue] = {
    "background_strain": _paper(
        "BW25113",
        "The Keio collection is comprised of 3985 deletions in duplicate (7970 total) "
        "of E. coli K-12 strain BW25113",
    ),
    "collection": _paper(
        "Keio collection",
        "These mutants—the ‘Keio collection'—provide a new resource",
        note="the collection name, set only on obtained deletions",
    ),
    "cassette": _paper(
        "kanamycin cassette",
        "Open-reading frame coding regions were replaced with a kanamycin cassette "
        "flanked by FLP recognition target sites",
    ),
    "counts": _paper(
        {"targeted": 4288, "obtained": 3985, "not_obtained": 303},
        _COUNTS_QUOTE,
        note="process() requires Table 3 under the scoring rule to give exactly these",
    ),
    "colonies_tested": _paper(
        (4, 8),
        "From every gene deletion experiment, four or eight KmR colonies were chosen and "
        "checked for ones with the correct structure by PCR",
        note="measured: 382 ORFs tested 4 colonies, 3,906 tested 8",
    ),
    "scoring_rule": _paper(
        2,
        "Mutants were scored as correct if two or more colonies had the expected "
        "structure based on PCR tests for both junction fragments.",
        note="the minimum verified colonies for an obtained deletion; applied to Table "
        "3's '5. verified' column it reproduces 3,985 / 303 and Table 6 exactly",
    ),
    "percent_column": _paper(
        "percent of the tested colonies verified by PCR",
        "bPercent of the four or eight KmR colonies shown to have the correct structure "
        "by PCR as described in text is given.",
        note="Table II footnote b; measured: '7. percent' equals 100 x verified / "
        "tested on all 4,288 targeted rows",
    ),
    "call_definition": _paper(
        "candidates for essential genes",
        "Based on finding mutants with the predicted structure, these 3985 genes are "
        "(probably) nonessential, while the 303 genes (including 37 genes of unknown "
        "function), for which no mutants were found, are candidates for essential genes",
        note="what is_essential=True means here: a candidate, from a construction that "
        "yielded no mutant with the expected structure (#924)",
    ),
    "population_methods_caveat": _paper(
        "a recovery method does not call a slow grower essential",
        "Because genetic footprinting measures cell populations, a mutation causing slow "
        "growth can lead to under-representation of the mutant and hence false "
        "classification of many genes as essential. In contrast, we sought deletion "
        "mutants as survivors without regard to growth rate.",
        note="the paper's own statement of how its evidence differs from a population "
        "method's; carried on every record's evidence (#924)",
    ),
    "secm_exception": _paper(
        "secM",
        "The exceptional case, secM, has a translational arrest sequence within its "
        "C-terminus that is required for expression of the downstream secA",
        note="secM has one verified colony of eight: under the scoring rule it is not "
        "obtained, and it is in Table 6",
    ),
    "candidates_table": _paper(
        "Supplementary Table 6",
        "Essential gene candidates are given in Supplementary Table 6.",
    ),
    "selection_plates": _paper(
        "LB agar with 30 ug/ml kanamycin, aerobic, 37 C",
        _SELECTION_QUOTE,
        note="the environment of the call; no incubation time or agar amount is stated",
    ),
    "selection_temperature_c": _paper(37.0, _SELECTION_QUOTE),
    "kanamycin_ug_per_ml": _paper(30.0, _SELECTION_QUOTE),
    "lb_formulation": _paper(
        "1% tryptone, 0.5% yeast extract, 0.5% NaCl",
        "Cells were routinely grown in LB medium containing 1% Bacto Tryptone (Difco), "
        "0.5% yeast extract (Difco), and 0.5% NaCl with or without antibiotics",
        note="the LB_LENNOX amounts (10 / 5 / 5 g/L); 'with or without antibiotics' is "
        "the parent's medium for the reference",
    ),
    "untargeted": _paper(
        "IS genes, ldr toxins, genes BW25113 lacks, re-annotated ORFs",
        "ORFs not targeted include 79 IS genes, four genes for small toxic polypeptides "
        "(ldrA, ldrB, ldrC, and ldrD), and seven genes already disrupted in BW25113",
        note="Table 3's 90 'N.A.' rows; no construction, so no call",
    ),
}

BACKGROUND_STRAIN: Literal["BW25113"] = "BW25113"
if SOURCED_VALUES["background_strain"].value != BACKGROUND_STRAIN:
    raise RuntimeError("the sourced background strain is not the loader's strain")
KEIO_COLLECTION: str = SOURCED_VALUES["collection"].value
KEIO_CASSETTE: str = SOURCED_VALUES["cassette"].value
MIN_VERIFIED_COLONIES: int = SOURCED_VALUES["scoring_rule"].value
EXPECTED_COUNTS: dict[str, int] = SOURCED_VALUES["counts"].value
SELECTION_TEMPERATURE_C: float = SOURCED_VALUES["selection_temperature_c"].value
KANAMYCIN_UG_PER_ML: float = SOURCED_VALUES["kanamycin_ug_per_ml"].value
#: The statements the media cite, bound at import so a rebuilt environment never
#: depends on the audited dict.
SELECTION_PLATES_STATEMENT = SOURCED_VALUES["selection_plates"]
LB_FORMULATION_STATEMENT = SOURCED_VALUES["lb_formulation"]
KANAMYCIN_STATEMENT = SOURCED_VALUES["kanamycin_ug_per_ml"]
#: The statements every record's evidence quotes (#924), bound at import likewise.
SCORING_RULE_STATEMENT = SOURCED_VALUES["scoring_rule"]
CALL_DEFINITION_STATEMENT = SOURCED_VALUES["call_definition"]
POPULATION_METHODS_CAVEAT = SOURCED_VALUES["population_methods_caveat"]
COLONIES_TESTED_STATEMENT = SOURCED_VALUES["colonies_tested"]
PERCENT_COLUMN_STATEMENT = SOURCED_VALUES["percent_column"]
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
        role=EssentialityStatementRole.caveat, provenance=POPULATION_METHODS_CAVEAT
    ),
    EssentialityStatement(
        role=EssentialityStatementRole.statistic_definition,
        provenance=COLONIES_TESTED_STATEMENT,
    ),
    EssentialityStatement(
        role=EssentialityStatementRole.statistic_definition,
        provenance=PERCENT_COLUMN_STATEMENT,
    ),
)
#: Below this fraction of distinct JW ids resolving to one BW25113 locus the build stops
#: (checklist item 4). The annotation carries 4,334 JW synonyms; measured 4,163 of 4,288
#: resolve (0.9708), 47 of them onto a locus another JW id also reaches.
MIN_RESOLVED_FRACTION = 0.95

_PAPER_LOOKED_IN = Provenance(
    source_uri=PAPER_TEXT_REL,
    citation_key=CITATION_KEY,
    sha256=PAPER_TEXT_SHA256,
    method="full Results and Materials and methods read",
)
SELECTION_DURATION_GAP = ProvenanceGap(
    field="duration_hours",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=_PAPER_LOOKED_IN,
    note="the transformants were spread 'to select KmR transformants at 37°C'; no "
    "incubation time is stated for the selection plates",
)

PUBLICATION = Publication(
    pubmed_id=PUBMED_ID,
    pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{PUBMED_ID}/",
    doi=PAPER_DOI,
    doi_url=f"https://doi.org/{PAPER_DOI}",
)


# --------------------------------------------------------------------------- #
# Raw mirror (the loader reads the mirror, never a live URL)
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """``DATA_ROOT`` from the environment (the mirrors and the build tree live under it)."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/babaConstructionEscherichiaColi2006``."""
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
# Parsing the released tables
# --------------------------------------------------------------------------- #
class TableFormatError(ValueError):
    """A released table whose title, header or a cell is not what the loader reads."""


class ScoringRuleError(ValueError):
    """The scoring rule no longer reproduces the paper's counts or Table 6."""


class DuplicateRecordError(ValueError):
    """Two kept rows on the same BW25113 locus."""


TABLE_S3_TITLE = (
    "Supplementary Table 3. Information on Keio collection deletion mutants"
)
TABLE_S2_TITLE = (
    "Supplementary Table 2. Extensions for creation of Keio collection deletion mutants"
)
TABLE_S6_TITLE = "Supplementary Table 6. E. coli K-12 essential gene candidates"
#: Table 3 header cells the parser reads: ``(row, column) -> text``.
TABLE_S3_HEADER: dict[tuple[int, int], str] = {
    (1, 0): "1. ECK number",
    (1, 1): "2. gene",
    (1, 2): "3. W3110 information",
    (1, 8): "4. MG1655 information",
    (1, 14): "5. verified",
    (1, 15): "6. colonies tested",
    (1, 16): "7. percent",
    (3, 2): "JW id",
    (3, 8): "b number",
}
#: Table 3's marker for an ORF that was not targeted.
NOT_TARGETED = "N.A."
ALLOWED_TESTED = frozenset({4, 8})


class KeioRow(BaseModel):
    """One Table 3 row: the ORF, and its PCR verification when it was targeted."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    row: int
    eck: str
    gene: str
    jw_id: str
    b_number: str
    verified: int | None
    tested: int | None
    percent: float | None

    @property
    def targeted(self) -> bool:
        """A construction was attempted (Table 3 gives its colony counts)."""
        return self.tested is not None

    @property
    def obtained(self) -> bool:
        """The scoring rule: at least ``MIN_VERIFIED_COLONIES`` verified colonies."""
        if self.verified is None:
            raise ValueError(f"{self.jw_id} was not targeted; it has no outcome")
        return self.verified >= MIN_VERIFIED_COLONIES


def _sheet(path: str | Path) -> xlrd.sheet.Sheet:
    """The one sheet of a released workbook."""
    book = xlrd.open_workbook(str(path))
    if book.nsheets != 1:
        raise TableFormatError(f"{path}: {book.nsheets} sheets")
    return book.sheet_by_index(0)


def _count(value: Any, where: str) -> int:
    """A whole-number cell (xlrd reads every number as a float)."""
    if not isinstance(value, float) or not value.is_integer() or value < 0:
        raise TableFormatError(f"{where}: {value!r} is not a non-negative whole number")
    return int(value)


def read_table_s3(path: str | Path) -> list[KeioRow]:
    """Every data row of Table 3, refusing an unexpected title, header or count cell.

    ``row`` is 1-based over the data rows (sheet row minus the four header rows). A
    targeted row's verified / tested / percent are whole, consistent numbers; an
    untargeted row carries ``N.A.`` in all three.
    """
    sheet = _sheet(path)
    if sheet.cell_value(0, 0) != TABLE_S3_TITLE:
        raise TableFormatError(f"Table 3 title {sheet.cell_value(0, 0)!r}")
    for (r, c), text in TABLE_S3_HEADER.items():
        if sheet.cell_value(r, c) != text:
            raise TableFormatError(
                f"Table 3 cell ({r}, {c}) {sheet.cell_value(r, c)!r}"
            )
    rows: list[KeioRow] = []
    for r in range(4, sheet.nrows):
        where = f"Table 3 sheet row {r + 1}"
        cells = [sheet.cell_value(r, c) for c in range(17)]
        jw_id = cells[2]
        if not isinstance(jw_id, str) or not jw_id or jw_id != jw_id.strip():
            raise TableFormatError(f"{where}: JW id {jw_id!r}")
        counts = (cells[14], cells[15], cells[16])
        if all(c == NOT_TARGETED for c in counts):
            verified = tested = None
            percent = None
        else:
            verified = _count(cells[14], where)
            tested = _count(cells[15], where)
            percent = float(cells[16])
            if tested not in ALLOWED_TESTED or verified > tested:
                raise TableFormatError(f"{where}: {verified} of {tested} verified")
            if abs(percent - 100.0 * verified / tested) > 1e-9:
                raise TableFormatError(f"{where}: percent {percent} != verified/tested")
        rows.append(
            KeioRow(
                row=r - 3,
                eck=str(cells[0]),
                gene=str(cells[1]),
                jw_id=jw_id,
                b_number=str(cells[8]),
                verified=verified,
                tested=tested,
                percent=percent,
            )
        )
    repeated = sorted(k for k, n in Counter(r.jw_id for r in rows).items() if n > 1)
    if repeated:
        raise TableFormatError(f"Table 3 repeats JW ids {repeated[:5]}")
    return rows


def read_jw_ids(path: str | Path, *, title: str, header_row: int) -> list[str]:
    """The JW id column (column 3) of Table 2 or Table 6, after its header rows."""
    sheet = _sheet(path)
    if sheet.cell_value(0, 0) != title:
        raise TableFormatError(f"{path}: title {sheet.cell_value(0, 0)!r}")
    if str(sheet.cell_value(header_row, 2)).lower() != "jw id":
        raise TableFormatError(f"{path}: JW header {sheet.cell_value(header_row, 2)!r}")
    return [str(sheet.cell_value(r, 2)) for r in range(header_row + 1, sheet.nrows)]


class ScoringCheck(BaseModel):
    """The scoring rule against the paper's counts, Table 2 and Table 6."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    targeted: int
    obtained: int
    not_obtained: int
    not_targeted: int
    targeted_equals_table_s2: bool
    not_obtained_equals_table_s6: bool
    not_obtained_with_one_verified: list[str]

    @property
    def counts(self) -> dict[str, int]:
        """The three counts the paper states, in its keys."""
        return {
            "targeted": self.targeted,
            "obtained": self.obtained,
            "not_obtained": self.not_obtained,
        }


def scoring_check(
    rows: Sequence[KeioRow], table_s2: Sequence[str], table_s6: Sequence[str]
) -> ScoringCheck:
    """Apply the scoring rule and compare it with the counts and the two tables."""
    targeted = [r for r in rows if r.targeted]
    not_obtained = [r for r in targeted if not r.obtained]
    return ScoringCheck(
        targeted=len(targeted),
        obtained=len(targeted) - len(not_obtained),
        not_obtained=len(not_obtained),
        not_targeted=len(rows) - len(targeted),
        targeted_equals_table_s2={r.jw_id for r in targeted} == set(table_s2),
        not_obtained_equals_table_s6={r.jw_id for r in not_obtained} == set(table_s6),
        not_obtained_with_one_verified=[
            f"{r.gene} {r.jw_id}" for r in not_obtained if r.verified == 1
        ],
    )


def require_scoring(check: ScoringCheck) -> None:
    """Refuse a release whose scoring no longer reproduces the paper and Table 6."""
    if check.counts != EXPECTED_COUNTS:
        raise ScoringRuleError(
            f"scoring rule gives {check.counts}, the paper states {EXPECTED_COUNTS}"
        )
    if not (check.targeted_equals_table_s2 and check.not_obtained_equals_table_s6):
        raise ScoringRuleError(
            "the targeted ORFs are not Table 2's or the unobtained are not Table 6's"
        )


# --------------------------------------------------------------------------- #
# Identifiers and the retention ledger
# --------------------------------------------------------------------------- #
DropRuleName = Literal["not_targeted", "jw_id_not_on_one_bw25113_locus"]
DROP_RULE_DESCRIPTIONS: dict[DropRuleName, str] = {
    "not_targeted": "Table 3 gives 'N.A.' for the verification columns: the paper did "
    "not target the ORF, so there is no construction outcome to store",
    "jw_id_not_on_one_bw25113_locus": "the row's Keio JW id is not a gene_synonym of "
    "exactly one BW25113 GenBank locus (retired, or shared with another JW id), so no "
    "locus tag of the pinned namespace names the deletion; each item gives the JW id, "
    "gene, b-number and whether a mutant was obtained",
}


class RowOutcome(BaseModel):
    """What happened to one Table 3 row: its locus and record, or the rule that dropped it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    keio: KeioRow
    locus_tag: str | None
    symbol: str | None
    drop_rule: DropRuleName | None
    reason: str | None


class DropRule(BaseModel):
    """One retention rule, the rows it removed, and why."""

    rule: DropRuleName
    description: str
    n_records: int
    items: list[str] = []


class DropLog(BaseModel):
    """Every retention rule applied to a build, in the order they were applied."""

    dataset: str
    source_rows: int
    kept_records: int
    kept_essential: int
    kept_non_essential: int
    dropped_records: int
    rules: list[DropRule]


class IdentifierLedger(BaseModel):
    """The JW reconciliation report, plus the stop threshold."""

    reconciliation: LocusTagReconciliation
    min_resolved_fraction: float


def canonical_symbol(genome: EcoliK12Genome, tag: str) -> str:
    """The annotation's own symbol of ``tag`` when it resolves back to ``tag``, else the
    tag, so the stored pair always round-trips through the resolver.
    """
    symbol = genome.genbank.loci[tag].symbol
    if symbol is None:
        return tag
    return symbol if genome.resolve_gene_name(symbol).systematic_name == tag else tag


def resolve_rows(
    genome: EcoliK12Genome, rows: Sequence[KeioRow], *, label: str
) -> tuple[list[RowOutcome], LocusTagReconciliation]:
    """Reconcile every targeted row's JW id on ``genome`` and decide each row's outcome.

    Untargeted rows are ``not_targeted``. Stops (``LocusTagResolutionError``) below
    ``MIN_RESOLVED_FRACTION``. A JW id the reconciler keeps as given is off the BW25113
    namespace and its row is dropped with the reason.
    """
    targeted = [r for r in rows if r.targeted]
    stored, report = reconcile_locus_tags(
        genome, pd.Series([r.jw_id for r in targeted], dtype=object), label=label
    )
    report.require_resolved(MIN_RESOLVED_FRACTION)
    pattern = LOCUS_TAG_PATTERNS[report.gene_namespace]
    tag_of = dict(zip((r.jw_id for r in targeted), stored.tolist(), strict=True))
    outcomes: list[RowOutcome] = []
    for row in rows:
        if not row.targeted:
            outcomes.append(
                RowOutcome(
                    keio=row,
                    locus_tag=None,
                    symbol=None,
                    drop_rule="not_targeted",
                    reason=f"{row.gene} {row.jw_id}",
                )
            )
            continue
        tag = str(tag_of[row.jw_id])
        if pattern.match(tag) is None:
            why = (
                "shares its locus with another JW id"
                if row.jw_id in report.kept_on_collision
                else "not carried by the annotation"
            )
            outcomes.append(
                RowOutcome(
                    keio=row,
                    locus_tag=None,
                    symbol=None,
                    drop_rule="jw_id_not_on_one_bw25113_locus",
                    reason=f"{row.jw_id} ({row.gene}, {row.b_number}): {why}; "
                    f"obtained {row.obtained}",
                )
            )
            continue
        outcomes.append(
            RowOutcome(
                keio=row,
                locus_tag=tag,
                symbol=canonical_symbol(genome, tag),
                drop_rule=None,
                reason=None,
            )
        )
    held = Counter(o.locus_tag for o in outcomes if o.drop_rule is None)
    repeated = sorted(str(t) for t, n in held.items() if n > 1)
    if repeated:
        raise DuplicateRecordError(f"two kept rows on {repeated[:5]}")
    return outcomes, report


def build_drop_log(dataset_name: str, outcomes: Sequence[RowOutcome]) -> DropLog:
    """The retention ledger of one build, refusing rules that miss a dropped row."""
    rules = [
        DropRule(
            rule=name,
            description=description,
            n_records=sum(1 for o in outcomes if o.drop_rule == name),
            items=[str(o.reason) for o in outcomes if o.drop_rule == name],
        )
        for name, description in DROP_RULE_DESCRIPTIONS.items()
    ]
    kept = [o for o in outcomes if o.drop_rule is None]
    log_ = DropLog(
        dataset=dataset_name,
        source_rows=len(outcomes),
        kept_records=len(kept),
        kept_essential=sum(1 for o in kept if not o.keio.obtained),
        kept_non_essential=sum(1 for o in kept if o.keio.obtained),
        dropped_records=len(outcomes) - len(kept),
        rules=rules,
    )
    if sum(r.n_records for r in rules) != log_.dropped_records:
        raise RuntimeError("drop rules do not account for every dropped row")
    return log_


# --------------------------------------------------------------------------- #
# Environments and records (pure, no files)
# --------------------------------------------------------------------------- #
def _agar() -> MediaComponent:
    """The plates' agar; its amount is not stated."""
    return MediaComponent(
        compound=resolved_compound("agar"),
        role=MediaComponentRole.gelling_agent,
        concentration=None,
        provenance=[SELECTION_PLATES_STATEMENT],
        note="the plates are LB agar; no agar amount is stated",
    )


def plate_medium(*, kanamycin: bool) -> Media:
    """Baba's LB agar, with or without the 30 ug/ml kanamycin of the selection plates.

    The paper's LB is 1% tryptone, 0.5% yeast extract, 0.5% NaCl, the ``LB_LENNOX``
    amounts, so those components are reused; the medium derives from the ``LB`` key so
    it joins there. ``LB_AGAR`` is not reused: its 2% agar is another paper's.
    """
    components = [*LB_LENNOX.components, _agar()]
    provenance = [LB_FORMULATION_STATEMENT, SELECTION_PLATES_STATEMENT]
    if not kanamycin:
        return Media(
            name="LB agar, Lennox amounts (Baba 2006; agar amount unstated)",
            state="solid",
            is_synthetic=False,
            base_medium="LB",
            components=components,
            provenance=provenance,
        )
    components.append(
        MediaComponent(
            compound=resolved_compound("kanamycin"),
            role=MediaComponentRole.selection_agent,
            concentration=Concentration(
                value=KANAMYCIN_UG_PER_ML, unit=ConcentrationUnit.ug_per_ml
            ),
            provenance=[KANAMYCIN_STATEMENT],
            note="selects for the replacement's kanamycin cassette",
        )
    )
    return Media(
        name="LB agar, Lennox amounts, with 30 ug/ml kanamycin (Baba 2006 Keio "
        "deletion selection plates; agar amount unstated)",
        state="solid",
        is_synthetic=False,
        base_medium="LB",
        components=components,
        provenance=provenance,
    )


def environment(*, kanamycin: bool = True) -> Environment:
    """The plates at 37 C, incubated aerobically; the incubation time is a typed gap."""
    return Environment(
        media=plate_medium(kanamycin=kanamycin),
        temperature=Temperature(value=SELECTION_TEMPERATURE_C),
        aerobicity="aerobic",
        provenance_gaps=[SELECTION_DURATION_GAP],
    )


def deletion_genotype(outcome: RowOutcome) -> Genotype:
    """The designed in-frame deletion; ``collection`` only when a mutant was obtained."""
    if outcome.locus_tag is None or outcome.symbol is None:
        raise ValueError(f"{outcome.keio.jw_id} was dropped; it has no genotype")
    obtained = outcome.keio.obtained
    return Genotype(
        perturbations=[
            BacterialDeletionPerturbation(
                systematic_gene_name=outcome.locus_tag,
                perturbed_gene_name=outcome.symbol,
                gene_namespace=STRAIN_GENE_NAMESPACES[BACKGROUND_STRAIN],
                identifier_mapping=DerivedIdentifierMapping(
                    source_identifier=outcome.keio.jw_id, route="jw_synonym"
                ),
                collection=KEIO_COLLECTION if obtained else None,
                cassette=KEIO_CASSETTE,
            )
        ]
    )


def evidence(row: KeioRow | None = None) -> GeneEssentialityEvidence:
    """The construction-outcome evidence (#924), with ``row``'s PCR counts when given.

    The reference (``row`` None) carries the type, rule, definition and caveat only: no
    construction was attempted on the parent.
    """
    counts: dict[str, Any] = {}
    if row is not None:
        if not row.targeted:
            raise ValueError(f"{row.jw_id} was not targeted; it has no PCR counts")
        counts = {
            "colonies_verified": row.verified,
            "colonies_tested": row.tested,
            "percent_verified": row.percent,
        }
    return GeneEssentialityEvidence(
        evidence_class=EssentialityEvidenceClass.mutant_recovery,
        evidence_type=EssentialityEvidenceType.construction_outcome,
        statements=list(EVIDENCE_STATEMENTS),
        **counts,
    )


def build_experiment(
    dataset_name: str, outcome: RowOutcome, env: Environment
) -> BacterialGeneEssentialityExperiment:
    """One targeted ORF: ``is_essential`` is True when no mutant was obtained."""
    if outcome.drop_rule is not None:
        raise ValueError(f"{outcome.keio.jw_id} was dropped; it has no record")
    return BacterialGeneEssentialityExperiment(
        dataset_name=dataset_name,
        genotype=deletion_genotype(outcome),
        environment=env,
        phenotype=GeneEssentialityPhenotype(
            is_essential=not outcome.keio.obtained, evidence=evidence(outcome.keio)
        ),
    )


def build_reference(
    dataset_name: str, genome_reference: AssemblyReferenceGenome
) -> BacterialGeneEssentialityExperimentReference:
    """The unperturbed parent, viable, on the same LB agar without the kanamycin."""
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
class GeneEssentialityBaba2006Dataset(ExperimentDataset):
    """Keio construction outcomes of 4,288 targeted E. coli BW25113 ORFs."""

    organism = Organism.ecoli

    REFERENCE_STRAIN: ClassVar[EcoliK12StrainName] = "BW25113"

    def __init__(
        self,
        root: str = "data/torchcell/gene_essentiality_baba2006",
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        ecoli_genome: EcoliK12Genome | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; ``ecoli_genome`` is injected by the build entry points."""
        self.ecoli_genome = ecoli_genome
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
        """The three consumed tables, linked from the raw mirror."""
        return list(DATA_SHA256)

    def download(self) -> None:
        """Link each mirror table into ``raw/`` after checking the manifest and sha256."""
        data_root = _data_root()
        manifest = load_manifest(data_root)
        os.makedirs(self.raw_dir, exist_ok=True)
        for name, sha256 in DATA_SHA256.items():
            relpath = mirror_relpath(name)
            check_manifest_pin(relpath, manifest_sha256(manifest, relpath), sha256)
            link_verified(
                raw_mirror_dir(data_root) / relpath,
                osp.join(self.raw_dir, name),
                sha256,
            )
        log.info("Baba 2006 tables linked into %s (sha256 verified)", self.raw_dir)

    def _raw(self, name: str) -> str:
        return osp.join(self.raw_dir, name)

    def _genome(self) -> EcoliK12Genome:
        """The injected genome, or the reference strain's default cache; another
        assembly set is refused.
        """
        if self.ecoli_genome is None:
            self.ecoli_genome = bacterial_genome("ecoli", self.REFERENCE_STRAIN)
        expected = BACTERIAL_ASSEMBLY_SETS[self.REFERENCE_STRAIN]
        if self.ecoli_genome.ASSEMBLY_SET != expected:
            raise ValueError(
                f"{type(self).__name__} needs the {expected} genome, got "
                f"{self.ecoli_genome.ASSEMBLY_SET}"
            )
        return self.ecoli_genome

    @post_process
    def process(self) -> None:
        """Score Table 3, check it against Tables 2 and 6, resolve, write the LMDB."""
        verify_raw_files(self.raw_dir, DATA_SHA256)
        rows = read_table_s3(self._raw(TABLE_S3))
        check = scoring_check(
            rows,
            read_jw_ids(self._raw(TABLE_S2), title=TABLE_S2_TITLE, header_row=2),
            read_jw_ids(self._raw(TABLE_S6), title=TABLE_S6_TITLE, header_row=3),
        )
        require_scoring(check)
        outcomes, report = resolve_rows(
            self._genome(), rows, label=f"{self.name} Table 3 JW ids"
        )
        drop_log = build_drop_log(self.name, outcomes)
        kept = [o for o in outcomes if o.drop_rule is None]
        log.info(
            "Baba 2006: %d Table 3 rows -> %d records (%d essential); dropped %s",
            drop_log.source_rows,
            drop_log.kept_records,
            drop_log.kept_essential,
            {r.rule: r.n_records for r in drop_log.rules},
        )
        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        self._write_ledgers(drop_log, report, check, outcomes)

        env = environment()
        reference = build_reference(
            self.name, assembly_reference(self.REFERENCE_STRAIN)
        )
        env_out, interned_env = self._open_write_lmdb(
            osp.join(self.processed_dir, "lmdb")
        )
        with env_out.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for idx, outcome in enumerate(tqdm(kept, desc="baba2006")):
                experiment = build_experiment(self.name, outcome, env)
                txn.put(
                    f"{idx}".encode(),
                    self._intern_record(experiment, reference, PUBLICATION, itxn),
                )
        env_out.close()
        interned_env.close()
        log.info("Wrote %d Baba 2006 records to LMDB", len(kept))

    def _write_ledgers(
        self,
        drop_log: DropLog,
        report: LocusTagReconciliation,
        check: ScoringCheck,
        outcomes: Sequence[RowOutcome],
    ) -> None:
        """The drop log, identifier ledger, scoring check, and the per-row table (with
        each kept row's LMDB index; its PCR counts are also on the record's evidence).
        """
        out = Path(self.preprocess_dir)
        (out / "dropped_records.json").write_text(drop_log.model_dump_json(indent=2))
        (out / "identifier_reconciliation.json").write_text(
            IdentifierLedger(
                reconciliation=report, min_resolved_fraction=MIN_RESOLVED_FRACTION
            ).model_dump_json(indent=2)
        )
        (out / "scoring_check.json").write_text(check.model_dump_json(indent=2))
        rows: list[dict[str, Any]] = []
        n_kept = 0
        for o in outcomes:
            record: int | None = None
            if o.drop_rule is None:
                record = n_kept
                n_kept += 1
            rows.append(
                {
                    "row": o.keio.row,
                    "eck": o.keio.eck,
                    "gene": o.keio.gene,
                    "jw_id": o.keio.jw_id,
                    "b_number": o.keio.b_number,
                    "verified": o.keio.verified,
                    "tested": o.keio.tested,
                    "percent": o.keio.percent,
                    "obtained": o.keio.obtained if o.keio.targeted else None,
                    "locus_tag": o.locus_tag,
                    "symbol": o.symbol,
                    "drop_rule": o.drop_rule,
                    "record": record,
                }
            )
        pd.DataFrame(rows).astype(
            {"record": "Int64", "verified": "Int64", "tested": "Int64"}
        ).to_csv(out / "calls.csv", index=False)

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
    source_uri=f"https://doi.org/{PAPER_DOI} (Supplementary Tables 2, 3 and 6)",
    citation_key=CITATION_KEY,
    method="Keio construction outcome per targeted BW25113 ORF: two or more PCR-verified "
    "KmR colonies = obtained (not essential); reference = the parent, viable",
    page="Mol Syst Biol 2:2006.0008, Supplementary Table 3",
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


def _l1_one_record_per_locus(records: Sequence[Record]) -> LevelResult:
    loci = Counter(_perturbation(r)["systematic_gene_name"] for r in records)
    repeated = sorted(f"{g} x{n}" for g, n in loci.items() if n > 1)
    return LevelResult(
        level=Level.L1,
        name="one_record_per_bw25113_locus",
        passed=not repeated,
        message=f"{len(loci)} loci, one single-deletion record each"
        if not repeated
        else f"{len(repeated)} loci repeat",
        details={"repeated": repeated[:20]},
    )


def _l2_calls_match_table_s3(
    records: Sequence[Record],
    rows: Sequence[KeioRow],
    resolve: Callable[[str], GeneNameResolution],
) -> LevelResult:
    """L2, independent of the build path: each record's JW id finds its Table 3 row,
    the stored call is that row's scoring outcome, ``collection`` is set exactly on an
    obtained deletion, and the JW id resolves to the stored locus.
    """
    by_jw = {r.jw_id: r for r in rows}
    problems: list[str] = []
    for record in records:
        p = _perturbation(record)
        jw_id = p["identifier_mapping"]["source_identifier"]
        row = by_jw.get(jw_id)
        if row is None or not row.targeted:
            problems.append(f"{jw_id}: no targeted Table 3 row")
            continue
        stored = record["experiment"]["phenotype"]["is_essential"]
        if stored != (not row.obtained):
            problems.append(f"{jw_id}: stored {stored}, verified {row.verified}")
        ev = record["experiment"]["phenotype"]["evidence"]
        counts = (
            ev["colonies_verified"],
            ev["colonies_tested"],
            ev["percent_verified"],
        )
        if counts != (row.verified, row.tested, row.percent):
            problems.append(f"{jw_id}: evidence counts {counts}")
        if (p["collection"] == KEIO_COLLECTION) != row.obtained:
            problems.append(f"{jw_id}: collection {p['collection']!r}")
        if resolve(jw_id).systematic_name != p["systematic_gene_name"]:
            problems.append(f"{jw_id}: does not resolve to {p['systematic_gene_name']}")
    return LevelResult(
        level=Level.L2,
        name="calls_match_table_s3",
        passed=not problems,
        message=f"{len(records)} records equal their Table 3 row's scoring outcome and "
        "carry its PCR counts as evidence"
        if not problems
        else f"{len(problems)} records disagree with Table 3",
        details={"n_problems": len(problems), "problems": problems[:20]},
    )


def _l3_scoring_rule(check: ScoringCheck) -> LevelResult:
    passed = (
        check.counts == EXPECTED_COUNTS
        and check.targeted_equals_table_s2
        and check.not_obtained_equals_table_s6
    )
    return LevelResult(
        level=Level.L3,
        name="scoring_rule_reproduces_the_paper_and_table_6",
        passed=passed,
        message=f"two or more verified colonies: {check.counts}; the paper states "
        f"{EXPECTED_COUNTS}; Table 6 equal {check.not_obtained_equals_table_s6}",
        details=check.model_dump(),
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
        "parent on LB agar without the selection agent",
        details={"n_wrong": wrong},
    )


#: The evidence fields a record shares with the reference: everything but the counts.
_SHARED_EVIDENCE_EXCLUDES = frozenset(
    {"colonies_verified", "colonies_tested", "percent_verified"}
)


def _l3_evidence_is_construction_outcome(records: Sequence[Record]) -> LevelResult:
    """L3 (#924): every record and reference carries the construction-outcome evidence,
    with the scoring rule, definition and caveat the loader sources; the reference has
    no counts.
    """
    reference = evidence().model_dump()
    shared = {k: v for k, v in reference.items() if k not in _SHARED_EVIDENCE_EXCLUDES}
    wrong = 0
    for record in records:
        ev = record["experiment"]["phenotype"]["evidence"]
        if {
            k: v for k, v in ev.items() if k not in _SHARED_EVIDENCE_EXCLUDES
        } != shared:
            wrong += 1
        elif record["reference"]["phenotype_reference"]["evidence"] != reference:
            wrong += 1
    return LevelResult(
        level=Level.L3,
        name="evidence_is_construction_outcome",
        passed=not wrong,
        message=f"{len(records) - wrong} of {len(records)} records and references carry "
        "construction_outcome evidence (mutant_recovery) with the sourced rule, "
        "definition and caveat",
        details={"n_wrong": wrong},
    )


def _l4_containment(records: Sequence[Record], universe: set[str]) -> LevelResult:
    deleted = {_perturbation(r)["systematic_gene_name"] for r in records}
    missing = sorted(deleted - universe)
    return LevelResult(
        level=Level.L4,
        name="gene_containment_bw25113_locus_tags",
        passed=bool(deleted) and not missing,
        message=f"{len(deleted) - len(missing)} of {len(deleted)} deleted loci are "
        "BW25113 GenBank gene rows",
        details={
            "n_deleted": len(deleted),
            "n_universe": len(universe),
            "missing_examples": missing[:20],
        },
    )


def verify_records(
    records: Sequence[Record],
    *,
    rows: Sequence[KeioRow],
    check: ScoringCheck,
    resolve: Callable[[str], GeneNameResolution],
    universe: set[str],
    expected_count: int,
    dataset_name: str = "gene_essentiality_baba2006",
) -> VerificationReport:
    """The L0-L4 gate over built records, given the re-read Table 3, its scoring check,
    the BW25113 resolver and the BW25113 gene universe.
    """
    from torchcell.verification.common import shared_rule_results

    report = VerificationReport(
        dataset_name=dataset_name, provenance=VERIFIER_PROVENANCE
    )
    report.add(l0_structural(records, _validate_record))
    report.add(l1_count(len(records), expected_count))
    report.add(_l1_one_record_per_locus(records))
    report.add(_l2_calls_match_table_s3(records, rows, resolve))
    report.add(_l3_scoring_rule(check))
    report.add(_l3_reference_is_the_parent_off_selection(records))
    report.add(_l3_evidence_is_construction_outcome(records))
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
    raw = osp.join(abs_root, "raw")
    rows = read_table_s3(osp.join(raw, TABLE_S3))
    check = scoring_check(
        rows,
        read_jw_ids(osp.join(raw, TABLE_S2), title=TABLE_S2_TITLE, header_row=2),
        read_jw_ids(osp.join(raw, TABLE_S6), title=TABLE_S6_TITLE, header_row=3),
    )
    genome = bacterial_genome("ecoli", "BW25113", base)
    references = {
        json.dumps(r["reference"]["genome_reference"], sort_keys=True) for r in records
    }
    universe: set[str] = set()
    for ref in references:
        universe |= _gene_set_for_reference(json.loads(ref), base)
    report = verify_records(
        records,
        rows=rows,
        check=check,
        resolve=genome.resolve_gene_name,
        universe=universe,
        expected_count=drops.kept_records,
    )
    mirror = Path(base) / "torchcell-raw"
    for value in SOURCED_VALUES.values():
        report.add(audit_sourced_value(value, mirror))
    _write_report(report, osp.join(abs_root, "preprocess"))
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI: ``build`` the dev LMDB or ``verify`` it."""
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(prog="python -m torchcell.datasets.ecoli.baba2006")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help="build (or load) the dev-tree LMDB")
    sub.add_parser("verify", help="run L0-L4 on the built dev-tree LMDB")
    args = parser.parse_args(argv)

    load_dotenv()
    data_root = _data_root()
    if args.command == "build":
        dataset = GeneEssentialityBaba2006Dataset(
            root=osp.join(data_root, DATASET_ROOT_REL)
        )
        print(f"len = {len(dataset)}")
        return 0
    report = run_verification(data_root)
    print(report.summary())
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
