# torchcell/datasets/scerevisiae/yoshikawa2009
# [[torchcell.datasets.scerevisiae.yoshikawa2009]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/yoshikawa2009
# Test file: tests/torchcell/datasets/scerevisiae/test_yoshikawa2009.py
r"""Yoshikawa 2009 deletion-collection growth rates under ethanol and NaCl stress.

Yoshikawa et al. 2009 (FEMS Yeast Res 9:32-44, doi:10.1111/j.1567-1364.2008.00456.x,
PMID 19054128) grew 4,729 MATalpha single-gene deletion strains one per well in static
96-well YPD cultures at 30 C, added 25 uL of sterile water (nonstress), of 40% (v/v)
ethanol (final 8%) or of 5 M NaCl (final 1 M) at 4 h, and fitted each well's specific
growth rate to ln(calibrated OD) over three time points. Every rate is normalized by the
plate's four standard-strain wells (BY4742 his3 deletion) and the median standard-strain
rate of its condition. Each strain was cultured twice per condition. Supporting
Information Table S1 (``FYR_456_sm_tableS1.xls``) releases both cultures' normalized
rates and the strain's call per condition.

THE SOURCE. The publisher's ``.xls`` (Wiley supporting information, deposited by hand) is
what the loader reads. YeastPhenome's ``.xlsx`` (yp-data v1.0, re-saved by a curator in
2013) is deposited beside it as a cross-check: the two have the same sheet, the same
4,731 x 14 dimensions and the same value in every one of the 56,748 data cells
(``experiments/database/scripts/yoshikawa2009_release_checks.py``).

RECORD = one (deletion strain x condition x culture) ``StrainEnvironmentResponseExperiment``:

- PHENOTYPE: ``EnvironmentResponsePhenotype``, ``measurement_type=growth_rate`` (an
  ABSOLUTE rate in 1/h), ``assay_type=liquid_od_growth``, the released rate of ONE culture
  on ``environment_response`` with ``replicate_id`` "1st experiment" or "2nd experiment"
  (the release's own column names), ``n_samples=1`` and ``sample_unit=biological_replicate``
  (the Caglar 2017 per-replicate precedent: nothing is averaged into a mean the release
  does not carry). ``category`` is the release's call for the strain in that condition,
  ``0: sensitive`` / ``1: no-change`` / ``2: tolerant`` mapped to ``sensitive`` /
  ``no_change`` / ``resistant``; the call is made from BOTH cultures ("both independent
  cultures were lower than the threshold value"), so it is identical on the strain's two
  records of one condition.
- GENOTYPE: one ``BarcodedKanMxDeletionPerturbation`` per strain, AS RELEASED: the
  release's ORF cell is ``systematic_gene_name`` (the two lower-case suffixes the release
  writes, ``YAL043C-a`` and ``YAR043C-a``, upper-cased, the only change; no two released
  ORFs coincide upper-cased) and its name cell is ``perturbed_gene_name``. No genome is
  consulted: a released ORF the current annotation renamed, merged or retired is stored
  under the name the release gives it (#1045, owner rule 2026-10-12: "the data is the
  data"). Collection "Open Biosystems", cassette kanMX4 (the paper names the Winzeler 1999
  collection, whose cassette Giaever 2014 states), barcodes gapped pending Giaever 2002
  (the strains were grown one per well, so no barcode was read).
- ENVIRONMENT: a ``CultureEnvironment`` in the shared liquid YPD restated with the
  paper's recipe, 30 C, static 96-well culture (127 uL after the addition), inoculated
  with 2 uL of a 24 h YPD pre-culture, read to 9 h (nonstress) or 11 h (stress). The
  stress is one ``SmallMoleculePerturbation``: ethanol at 8% (v/v) or sodium chloride at
  1 M; the stock's diluent is not named, a typed gap. Nonstress is the base condition
  with no perturbation (the 25 uL of water is a dilution, stated in ``units``).
- REFERENCE: per condition, the standard strain (BY4742 his3 deletion) in that same
  condition, its Table 1 average rate with Table 1's SD (``sample_sd``) over its 720
  wells, on a BY4742 ``StrainReferenceGenome``. The paper never says which strains derive
  from BY4739 instead ("derived from S. cerevisiae BY4742 (...) or BY4739 (...)"), so the
  background's ``provenance`` is a typed gap that says so.

THE 5% ETHANOL COLUMNS. Table S1 releases a "5% ethanol" condition the paper never
describes: the Methods give nonstress, 8% ethanol and 1 M NaCl, and Table 1 gives no
standard-strain rate for it. The columns are released data, so they are stored (9,458
records, #1045): ethanol at 5 (basis-free ``percent``, the release's own header), the
read time a typed gap on ``duration_hours`` ("the paper describes 8% ethanol and 1 M NaCl;
the 5% ethanol columns are released in Table S1 without a protocol or a Table 1 reference
rate"), and, because a numeric reference value is required and none is released for this
condition, the NONSTRESS standard strain (Table 1's nonstress row) as the reference; its
``units`` say so. No record is refused.

DATA SOURCE: ``$DATA_ROOT/torchcell-raw/yoshikawaComprehensivePhenotypicAnalysis2009/si/``
with a ``manifest.json`` (``write_raw_manifest``), ``DEPOSIT.md`` and ``SHA256SUMS.txt``.
The build reads only the publisher ``.xls`` and verifies its sha256.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import os.path as osp
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

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
from torchcell.datamodels.media import YPD_LIQUID, restated
from torchcell.datamodels.schema import (
    AssayType,
    BackgroundAllele,
    BarcodedKanMxDeletionPerturbation,
    Concentration,
    ConcentrationUnit,
    CultureEnvironment,
    CultureFormat,
    EndpointRule,
    EnvironmentPerturbationType,
    EnvironmentResponsePhenotype,
    EukaryoticStrainBackground,
    Experiment,
    ExperimentReference,
    Genotype,
    MatingType,
    MeasurementType,
    PreCulture,
    PreCultureSource,
    Publication,
    ResponseCategory,
    SampleUnit,
    SmallMoleculePerturbation,
    StrainEnvironmentResponseExperiment,
    StrainEnvironmentResponseExperimentReference,
    StrainReferenceGenome,
    Temperature,
    UncertaintyType,
)
from torchcell.datamodels.strain_background import (
    BRACHMANN_1998,
    GIAEVER_2002,
    KANMX4_CASSETTE,
    STANDARD_BY_GENOTYPES,
    pending_source_review,
    standard_allele,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.literature.manifest import (
    ROLE_RAW_DATA,
    ROLE_SI_DATA,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.sequence.genome.registry import (
    SGD_S288C_R64,
    SGD_S288C_R64_GENBANK_ACCESSION,
)
from torchcell.verification.report import Level, LevelResult, Provenance
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Provenance anchors
# --------------------------------------------------------------------------- #
CITATION_KEY = "yoshikawaComprehensivePhenotypicAnalysis2009"
DOI = "10.1111/j.1567-1364.2008.00456.x"
#: PubMed id: NCBI esearch on ``10.1111/j.1567-1364.2008.00456.x[doi]`` returns the single
#: id 19054128 (2026-10-12), the folder yp-data v1.0 curates this paper under.
PMID = "19054128"
TITLE = (
    "Comprehensive phenotypic analysis for identification of genes affecting growth "
    "under ethanol stress in Saccharomyces cerevisiae"
)
ZOTERO_ITEM = "DCF6RHD6"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"
DATASET_ROOT_REL = "data/torchcell/env_chemgen_yoshikawa2009"
REGISTRY_NAME = "env_chemgen_yoshikawa2009"

PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "6552adf5b44fabc293ff1980069b67c9a6598a0d60c98e90c8ce424c5120f7fa"
GIAEVER_2014_KEY = "giaeverYeastDeletionCollection2014"
GIAEVER_2014_SHA256 = "4a177a8658be57938eb0e45e736762281793ac7799d358e3a3349af61acf1719"

#: The landing page the owner followed to the supporting information (2026-10-12).
PUBLISHER_URL = f"https://doi.org/{DOI}"
#: yp-data v1.0 is commit 83e2917b (the Zenodo 7714347 freeze); a commit, not the tag.
YP_DATA_COMMIT = "83e2917bf86955ec6ba66dc70ff2ed0fe24ecbe8"
YP_DATA_URL = (
    f"https://raw.githubusercontent.com/yeastphenome/yp-data/{YP_DATA_COMMIT}/"
    f"Datasets/{PMID}/raw_data/FYR_456_sm_tableS1.xlsx"
)
RETRIEVED_AT = "2026-10-12"
MANUAL_RECIPE = (
    f"open {PUBLISHER_URL} in a browser signed in through an institutional "
    "subscription, follow 'Supporting Information' to Table S1, download "
    "FYR_456_sm_tableS1.xls unchanged (the publisher's file name), and verify its "
    "sha256; scripted requests to the publisher are refused (HTTP 401/403), so this "
    "retrieval is manual-once and every rebuild reads the deposited bytes"
)


class RawFile(BaseModel):
    """One deposited supporting-information file: where it lives, what it must hash to."""

    model_config = ConfigDict(frozen=True)

    filename: str
    sha256: str
    bytes: int

    @property
    def raw_rel(self) -> str:
        """Path inside the raw mirror."""
        return f"si/{self.filename}"


#: The publisher's Table S1, the file the loader reads.
TABLE_S1 = RawFile(
    filename="FYR_456_sm_tableS1.xls",
    sha256="1861ba647e382f28a39041b2b590276ad30e67ef334b0d8be968415f56d9f759",
    bytes=1399296,
)
#: YeastPhenome's copy (yp-data v1.0), the cross-check: cell-identical, never read here.
TABLE_S1_YEASTPHENOME = RawFile(
    filename="FYR_456_sm_tableS1.xlsx",
    sha256="f8567d2cbfe974a87cdb8629086884bc5e3c280513b99ab944065fb53d873871",
    bytes=553420,
)


def _paper(value: Any, quote: str, *, note: str | None = None) -> SourcedValue:
    """Bind a value to a verbatim quote in the sha256-pinned paper OCR mirror."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=PAPER_MD,
            citation_key=CITATION_KEY,
            sha256=PAPER_MD_SHA256,
            method="MinerU OCR of the publisher PDF (torchcell-library mirror)",
            page="Materials and methods / Results",
        ),
    )


# --------------------------------------------------------------------------- #
# Sourced constants
# --------------------------------------------------------------------------- #
STRAINS_SENTENCE = (
    "We evaluated the growth of single gene deletion strains derived from S. cerevisiae "
    "BY4742 (MATa his3D1 leu2D0 lys2D0 ura3D0) or BY4739 (MATa leu2D0 lys2D0 ura3D0) "
    "purchased from Open Biosystems."
)
#: The OCR (and the PDF text layer) write the alpha and Delta glyphs as "a" and "D"; the
#: rendered page 2 of the mirrored paper.pdf reads "MATα his3Δ1 leu2Δ0 lys2Δ0 ura3Δ0".
OCR_GLYPHS = (
    "the OCR writes MATα as 'MATa' and Δ as 'D'; checked against the rendered page 2 "
    "of the mirrored paper.pdf, which reads 'BY4742 (MATα his3Δ1 leu2Δ0 lys2Δ0 "
    "ura3Δ0) or BY4739 (MATα leu2Δ0 lys2Δ0 ura3Δ0)'"
)
BY4742_GENOTYPE = _paper(
    "BY4742",
    STRAINS_SENTENCE,
    note="sources BY4742's four alleles and MATalpha; " + OCR_GLYPHS,
)
BACKGROUND_EITHER = _paper(
    "BY4742 or BY4739",
    STRAINS_SENTENCE,
    note="each strain derives from one of the two; Table S1 has no background column, "
    "so which strain is which is not released. BY4739 carries HIS3",
)
COLLECTION = _paper(
    "Open Biosystems",
    STRAINS_SENTENCE,
    note="the physical source of the strains, verbatim",
)
WINZELER_COLLECTION = _paper(
    "the Winzeler 1999 deletion collection",
    "with a single gene deletion (Winzeler et al., 1999)",
    note="the collection is the YKO, whose cassette is KANMX4_CASSETTE (Giaever 2014)",
)
GENETICIN_SELECTION = _paper(
    "geneticin-resistant deletions",
    "containing $1 5 0 \\mu \\mathrm { g } \\mathrm { m L } ^ { - 1 }$ of geneticin",
    note="the stocks were grown on G418, the kanMX marker's selection",
)
N_STRAINS = _paper(
    4729, "4729 single gene deletion strains were eventually analyzed in this study."
)
YPD_RECIPE = _paper(
    "YPD",
    "All strains were cultured in YPD medium $1 \\%$ Bacto yeast extract, $2 \\%$ Bacto "
    "peptone, and $2 \\%$ glucose).",
    note="the shared YPD_LIQUID composition (1% yeast extract, 2% peptone, 2% glucose)",
)
PRECULTURE_SENTENCE = (
    "We prepared precultures by shaking deletion strains at 1050 strokes "
    "$\\operatorname* { m i n } ^ { - 1 }$ on a microplate shaker, TITRAMAX1000 "
    "(Heidolph Instruments, Germany), in $1 0 0 \\mu \\mathrm { L }$ of YPD medium in "
    "96-well microtiter plates (Corning Inc.) for $2 4 \\mathrm { h }$ at "
    "$3 0 ^ { \\circ } \\mathrm { C }$ ."
)
PRECULTURE = _paper(PreCultureSource.timed_culture, PRECULTURE_SENTENCE)
PRECULTURE_HOURS = _paper(24.0, PRECULTURE_SENTENCE)
CULTURE_SENTENCE = (
    "Thereafter, $2 \\mu \\mathrm { L }$ of preculture was inoculated in "
    "$1 0 0 \\mu \\mathrm { L }$ of warm YPD medium, and cells were cultured at "
    "$3 0 ^ { \\circ } \\mathrm { C }$ without shaking."
)
TEMPERATURE_C = _paper(30.0, CULTURE_SENTENCE)
STATIC_CULTURE = _paper(
    0.0,
    CULTURE_SENTENCE,
    note="static; the plates were agitated for 1 min at 1050 strokes/min only before "
    "each OD reading (AGITATION_BEFORE_READ)",
)
AEROBIC = _paper(
    "aerobic",
    CULTURE_SENTENCE,
    note="a static open microtiter well in air; the paper describes no oxygen limitation",
)
AGITATION_BEFORE_READ = _paper(
    "1 min at 1050 strokes/min before each read",
    "Before measuring the OD, the plates were agitated for $1 \\mathrm { m i n }$ at "
    "1050 strokes $\\operatorname* { m i n } ^ { - 1 }$ on a shaker.",
)
STRESS_SENTENCE = (
    "To investigate the growth behavior of deletion strains under stress, "
    "$2 5 \\mu \\mathrm { L }$ of $4 0 \\%$ (v/v) ethanol (final concentration, "
    "$8 \\%$ ) or $2 5 \\mu \\mathrm { L }$ of $5 \\mathrm { M }$ NaCl (final "
    "concentration, $1 \\mathrm { M N a C l } )$ ) was added to the culture medium "
    "$4 \\mathrm { { h } }$ later, at the time corresponding to the mid-log phase."
)
ETHANOL_PERCENT_V_V = _paper(
    8.0,
    STRESS_SENTENCE,
    note="the final concentration of a 40% (v/v) stock, so the basis is v/v",
)
NACL_MOLAR = _paper(1.0, STRESS_SENTENCE)
ADDITION_AT_HOURS = _paper(
    4.0, STRESS_SENTENCE, note="the stressor (or water) is added 4 h after inoculation"
)
WATER_SENTENCE = (
    "Sterilized water $( 2 5 \\mu \\mathrm { L } )$ was added to the culture medium to "
    "provide the nonstress condition."
)
NONSTRESS_WATER = _paper(25.0, WATER_SENTENCE)
WORKING_VOLUME_UL = _paper(
    127.0,
    CULTURE_SENTENCE,
    note="100 uL of YPD + 2 uL of pre-culture + the 25 uL addition (STRESS_SENTENCE, "
    "WATER_SENTENCE); the sum is arithmetic on three stated volumes",
)
VESSEL = _paper(
    "96-well microtiter plate",
    "A 96-well microtiter plate comprises eight rows and 12 columns.",
)
TIME_POINTS_SENTENCE = (
    "Growth of the deletion strains was monitored at 0, 2, 4, 5.5, 7, and "
    "$^ { 9 \\mathrm { h } }$ under the nonstress condition and at 0, 2, 4, 7, 9, and "
    "$1 1 \\mathrm { h }$ under the stress condition"
)
NONSTRESS_HOURS = _paper(
    9.0, TIME_POINTS_SENTENCE, note="inoculation to the last OD reading, nonstress"
)
STRESS_HOURS = _paper(
    11.0, TIME_POINTS_SENTENCE, note="inoculation to the last OD reading, stress"
)
RATE_WINDOWS = _paper(
    "5.5, 7, 9 h (nonstress); 7, 9, 11 h (stress)",
    "The specific growth rate under the nonstress condition was calculated by linear "
    "regression using the natural logarithm of calibrated OD values at 5.5, 7, and "
    "$^ { 9 \\mathrm { h } }$ , and that under the stress conditions was calculated "
    "using the natural logarithm of calibrated OD values at 7, 9, and "
    "$1 1 \\mathrm { h }$ .",
)
NORMALIZATION = _paper(
    "plate standard-strain normalization",
    "specific growth rates were normalized with those of the standard strain cultured "
    "in each plate using the formula",
    note="mu' = mu x (median standard-strain rate under the condition / mean of the "
    "plate's four standard-strain wells)",
)
TABLE_S1_IS_NORMALIZED = _paper(
    "Table S1",
    "The normalized specific growth rates of all deletion strains under each condition "
    "are shown in Supporting Information, Table S1.",
)
TWO_CULTURES = _paper(
    2,
    "All strains were separately cultured twice under nonstress and stress conditions.",
    note="the two released rates per condition are two cultures of the strain, one "
    "record each (n_samples=1, biological_replicate)",
)
STANDARD_STRAIN = _paper(
    "BY4742 his3 deletion",
    "We used the HIS3 deletion strain of BY4742 (BY4742Dhis3) as a standard because "
    "the HIS3 deletion did not alter the phenotype from the wildtype strain BY4742.",
    note="the reference strain of every condition; " + OCR_GLYPHS,
)
STANDARD_WELLS = _paper(
    720,
    "We simultaneously obtained 720 sets of information regarding the specific growth "
    "rates of the standard strain from the cultivation of all the deletion strains in "
    "90 microtiter plates twice under each condition.",
    note="n_samples of the reference: 4 wells x 90 plates x 2 runs per condition",
)
TABLE1_IS_SAMPLE_SD = _paper(
    UncertaintyType.sample_sd,
    "Table 1 shows the average and SD of the specific growth rates of 720 growth data "
    "for the standard strain under each condition.",
)
CALL_RULE = _paper(
    "both cultures below the threshold",
    "We defined stress-sensitive deletion strains as those whose growth rates in both "
    "independent cultures were lower than the threshold value.",
)
THRESHOLD_RULE = _paper(
    "mean +/- 3.3 SD of the standard strain",
    "outside the range of $3 . 3 \\ \\mathrm { S D }$ of the mean could be regarded as "
    "stress tolerant or sensitive",
)
TABLE_S1_LEGEND = _paper(
    "Table S1", "Table S1. Specific growth rates of single gene deletion strains."
)


class Table1Row(BaseModel):
    """One row of the paper's Table 1: the standard strain's rate and the thresholds."""

    model_config = ConfigDict(frozen=True)

    average: float
    sd: float
    sensitive_below: float
    tolerant_above: float
    source: SourcedValue


def _table1(label: str, avg: str, sd: str, low: str, high: str) -> Table1Row:
    """A Table 1 row from its verbatim OCR cells (the SD is printed as 'm × 10-3')."""
    mantissa, exponent = sd.split(" × 10")
    row = (
        f"<tr><td>{label}</td><td>{avg}</td><td>{sd}</td><td>{low}</td><td>{high}"
        "</td></tr>"
    )
    return Table1Row(
        average=float(avg),
        sd=float(mantissa) * 10.0 ** int(exponent),
        sensitive_below=float(low),
        tolerant_above=float(high),
        source=_paper(
            (float(avg), sd),
            row,
            note="Table 1: average specific growth rate (h-1), SD (h-1), threshold for "
            "sensitive strains, threshold for tolerant strains",
        ),
    )


TABLE1_NONSTRESS = _table1("Nonstress", "0.452", "4.4 × 10-3", "0.437", "0.466")
TABLE1_ETHANOL = _table1("8% Ethanol", "0.178", "5.6 × 10-3", "0.160", "0.196")
#: "NaCI" with a capital I is the OCR's, kept verbatim.
TABLE1_NACL = _table1("1 M NaCI", "0.203", "7.6 × 10-3", "0.178", "0.229")
TABLE1_COUNTS = _paper(
    {
        "nonstress": (591, 4095, 43),
        "8% ethanol": (864, 3862, 3),
        "1M NaCl": (637, 4080, 12),
    },
    "<tr><td>Nonstress</td><td>591 (0)</td><td>4095 (4095)</td><td>43 (43)</td></tr>"
    "<tr><td>8% Ethanol</td><td>864 (446)</td><td>3862 (3690)</td><td>3 (2)</td></tr>"
    "<tr><td>1 M NaCl</td><td>637 (329)</td><td>4080 (3799)</td><td>12 (10)</td></tr>",
    note="Table 2: sensitive / no change / tolerant strain counts per condition; the "
    "build checks Table S1's calls reproduce them",
)

SOURCED_VALUES: tuple[SourcedValue, ...] = (
    BY4742_GENOTYPE,
    BACKGROUND_EITHER,
    COLLECTION,
    WINZELER_COLLECTION,
    GENETICIN_SELECTION,
    N_STRAINS,
    YPD_RECIPE,
    PRECULTURE,
    PRECULTURE_HOURS,
    TEMPERATURE_C,
    STATIC_CULTURE,
    AEROBIC,
    AGITATION_BEFORE_READ,
    ETHANOL_PERCENT_V_V,
    NACL_MOLAR,
    ADDITION_AT_HOURS,
    NONSTRESS_WATER,
    WORKING_VOLUME_UL,
    VESSEL,
    NONSTRESS_HOURS,
    STRESS_HOURS,
    RATE_WINDOWS,
    NORMALIZATION,
    TABLE_S1_IS_NORMALIZED,
    TWO_CULTURES,
    STANDARD_STRAIN,
    STANDARD_WELLS,
    TABLE1_IS_SAMPLE_SD,
    CALL_RULE,
    THRESHOLD_RULE,
    TABLE_S1_LEGEND,
    TABLE1_NONSTRESS.source,
    TABLE1_ETHANOL.source,
    TABLE1_NACL.source,
    TABLE1_COUNTS,
)
"""Every module-level ``SourcedValue``; the test audits each quote against its mirror."""


# --------------------------------------------------------------------------- #
# The release (legacy BIFF .xls, one sheet 'data')
# --------------------------------------------------------------------------- #
SHEET_NAME = "data"
#: Row 0 names the condition over each three-column group; row 1 names the columns.
CONDITION_ROW = 0
COLUMN_ROW = 1
FIRST_DATA_ROW = 2
N_COLUMNS = 14
RELEASE_CONDITIONS: tuple[str, ...] = (
    "non-stress",
    "5% ethanol",
    "8% ethanol",
    "1M NaCl",
)
RATE_1 = "specific growth rate at 1st experiment (1/h) "
RATE_2 = "specific growth rate at 2nd experiment (1/h) "
CALL_COLUMN = "growth behavior (0: sensitive, 1: no-change, 2: tolerant)"
GROUP_COLUMNS: tuple[str, str, str] = (RATE_1, RATE_2, CALL_COLUMN)
#: The release's own names for its two cultures (its column headers, trimmed).
REPLICATE_IDS: tuple[str, str] = ("1st experiment", "2nd experiment")
#: The release's call code -> the shared axis and the release's own word for it.
CALLS: dict[int, tuple[ResponseCategory, str]] = {
    0: (ResponseCategory.sensitive, "0: sensitive"),
    1: (ResponseCategory.no_change, "1: no-change"),
    2: (ResponseCategory.resistant, "2: tolerant"),
}


class ConditionCells(BaseModel):
    """One strain's two released rates and its call in one condition."""

    model_config = ConfigDict(frozen=True)

    rates: tuple[float, float]
    call: int


class ReleaseRow(BaseModel):
    """One Table S1 strain row, verbatim: the ORF cell, the name cell, four conditions."""

    model_config = ConfigDict(frozen=True)

    orf: str
    name: str
    conditions: dict[str, ConditionCells]


def _expected_header() -> tuple[list[str], list[str]]:
    """Rows 0 and 1 of the release as text cells ('' where the sheet is blank)."""
    conditions: list[str] = ["", ""]
    for label in RELEASE_CONDITIONS:
        conditions.extend([label, "", ""])
    columns = ["Name", "Name", *(GROUP_COLUMNS * len(RELEASE_CONDITIONS))]
    return conditions, columns


def read_release(path: str | Path) -> list[ReleaseRow]:
    """Table S1's 4,729 strain rows; anything the release does not write refuses.

    The two header rows must be exactly the release's (the condition labels over each
    group, the column names); every data row carries an ORF text cell, a name text cell,
    eight rate numbers and four calls in {0, 1, 2}; no ORF appears twice.
    """
    book = xlrd.open_workbook(str(path))
    if book.sheet_names() != [SHEET_NAME]:
        raise RuntimeError(f"Table S1 sheets are {book.sheet_names()}, not ['data']")
    sheet = book.sheet_by_index(0)
    if sheet.ncols != N_COLUMNS:
        raise RuntimeError(f"Table S1 has {sheet.ncols} columns, not {N_COLUMNS}")
    want_conditions, want_columns = _expected_header()
    got_conditions = [str(sheet.cell_value(CONDITION_ROW, c)) for c in range(N_COLUMNS)]
    got_columns = [str(sheet.cell_value(COLUMN_ROW, c)) for c in range(N_COLUMNS)]
    if got_conditions != want_conditions or got_columns != want_columns:
        raise RuntimeError(
            f"Table S1 header differs from the release: {got_conditions} / {got_columns}"
        )
    rows: list[ReleaseRow] = []
    for r in range(FIRST_DATA_ROW, sheet.nrows):
        cells = [sheet.cell(r, c) for c in range(N_COLUMNS)]
        if cells[0].ctype != xlrd.XL_CELL_TEXT or cells[1].ctype != xlrd.XL_CELL_TEXT:
            raise RuntimeError(f"Table S1 row {r + 1}: the two name cells are not text")
        conditions: dict[str, ConditionCells] = {}
        for k, label in enumerate(RELEASE_CONDITIONS):
            first = 2 + 3 * k
            group = cells[first : first + 3]
            if any(cell.ctype != xlrd.XL_CELL_NUMBER for cell in group):
                raise RuntimeError(
                    f"Table S1 row {r + 1} / {label}: a cell is not a number"
                )
            rate_1, rate_2, call = (float(cell.value) for cell in group)
            if not (math.isfinite(rate_1) and math.isfinite(rate_2)):
                raise RuntimeError(f"Table S1 row {r + 1} / {label}: non-finite rate")
            if call not in (0.0, 1.0, 2.0):
                raise RuntimeError(f"Table S1 row {r + 1} / {label}: call {call!r}")
            conditions[label] = ConditionCells(rates=(rate_1, rate_2), call=int(call))
        rows.append(
            ReleaseRow(
                orf=str(cells[0].value), name=str(cells[1].value), conditions=conditions
            )
        )
    orfs = [row.orf for row in rows]
    if len(set(orfs)) != len(orfs):
        raise RuntimeError("Table S1 lists an ORF twice")
    return rows


def call_counts(rows: list[ReleaseRow], label: str) -> tuple[int, int, int]:
    """(sensitive, no change, tolerant) strain counts of one condition's calls."""
    calls = [row.conditions[label].call for row in rows]
    return (calls.count(0), calls.count(1), calls.count(2))


def check_release(rows: list[ReleaseRow]) -> None:
    """The release against the paper: 4,729 strains, and Table 2's call counts.

    Each condition's (sensitive, no change, tolerant) counts must equal the paper's
    Table 2 where Table 2 has a row (not the 5% ethanol columns), so a Table S1 whose
    calls were edited or misread refuses.
    """
    if len(rows) != N_STRAINS.value:
        raise RuntimeError(f"Table S1 has {len(rows)} strains, not {N_STRAINS.value}")
    for spec in CONDITIONS:
        if spec.table2_label is None:
            continue
        got = call_counts(rows, spec.label)
        want = TABLE1_COUNTS.value[spec.table2_label]
        if got != want:
            raise RuntimeError(
                f"Table S1 {spec.label} calls {got} differ from Table 2's {want}"
            )


# --------------------------------------------------------------------------- #
# Conditions
# --------------------------------------------------------------------------- #
ConditionKey = Literal["nonstress", "ethanol_5", "ethanol_8", "nacl_1m"]

#: Why the 5% ethanol columns carry gaps and the nonstress reference (#1045).
FIVE_PERCENT_ETHANOL_NOTE = (
    "the paper describes 8% ethanol and 1 M NaCl; the 5% ethanol columns are released "
    "in Table S1 without a protocol or a Table 1 reference rate"
)


class ConditionSpec(BaseModel):
    """A released condition: its stress, its timing, and the Table 1 row it is read by.

    ``table1`` is the standard-strain row of the reference: the condition's own, or, for
    the 5% ethanol columns Table 1 has no row for, the nonstress row (``reference_key``).
    ``table2_label`` is None where Table 2 gives no call counts. ``duration_hours`` is
    None, with ``duration_gap``, where the paper states no read time.
    """

    model_config = ConfigDict(frozen=True)

    label: str
    key: ConditionKey
    table1: Table1Row
    reference_key: ConditionKey
    table2_label: str | None
    duration_hours: float | None
    duration_gap: str | None = None
    compound_label: str | None
    dose: Concentration | None


CONDITIONS: tuple[ConditionSpec, ...] = (
    ConditionSpec(
        label="non-stress",
        key="nonstress",
        table1=TABLE1_NONSTRESS,
        reference_key="nonstress",
        table2_label="nonstress",
        duration_hours=NONSTRESS_HOURS.value,
        compound_label=None,
        dose=None,
    ),
    ConditionSpec(
        label="5% ethanol",
        key="ethanol_5",
        table1=TABLE1_NONSTRESS,
        reference_key="nonstress",
        table2_label=None,
        duration_hours=None,
        duration_gap=FIVE_PERCENT_ETHANOL_NOTE,
        compound_label="ethanol",
        dose=Concentration(value=5.0, unit=ConcentrationUnit.percent),
    ),
    ConditionSpec(
        label="8% ethanol",
        key="ethanol_8",
        table1=TABLE1_ETHANOL,
        reference_key="ethanol_8",
        table2_label="8% ethanol",
        duration_hours=STRESS_HOURS.value,
        compound_label="ethanol",
        dose=Concentration(
            value=ETHANOL_PERCENT_V_V.value, unit=ConcentrationUnit.percent_v_v
        ),
    ),
    ConditionSpec(
        label="1M NaCl",
        key="nacl_1m",
        table1=TABLE1_NACL,
        reference_key="nacl_1m",
        table2_label="1M NaCl",
        duration_hours=STRESS_HOURS.value,
        compound_label="sodium chloride",
        dose=Concentration(value=NACL_MOLAR.value, unit=ConcentrationUnit.molar),
    ),
)
#: Each condition by its key (the 5% ethanol reference reads the nonstress one).
CONDITION_BY_KEY: dict[ConditionKey, ConditionSpec] = {
    spec.key: spec for spec in CONDITIONS
}
MEASUREMENT_UNITS = (
    "normalized specific growth rate (1/h) of ONE culture, as released in Yoshikawa "
    "2009 Table S1 ('specific growth rate at 1st|2nd experiment (1/h)'): the slope of "
    "ln(calibrated OD) by linear regression over 5.5, 7 and 9 h (nonstress) or 7, 9 and "
    "11 h (stress), times the median standard-strain rate of the condition over the mean "
    "of the plate's four standard-strain (BY4742 his3 deletion) wells. Static 96-well "
    "culture, 100 uL YPD + 2 uL of a 24 h pre-culture at 30 C; 25 uL of sterile water "
    "(nonstress), 40% (v/v) ethanol (final 8%) or 5 M NaCl (final 1 M) added at 4 h "
    "(mid-log). category = the release's call for the strain in the condition from BOTH "
    "cultures ('growth behavior (0: sensitive, 1: no-change, 2: tolerant)'; sensitive = "
    "both cultures below the standard strain's mean - 3.3 SD, tolerant = both above "
    "mean + 3.3 SD, Table 1), identical on the strain's two records of the condition"
)
REFERENCE_UNITS = (
    MEASUREMENT_UNITS
    + "; reference = the standard strain (BY4742 his3 deletion, 4 wells per plate) in "
    "the same condition: Table 1's average specific growth rate over its 720 wells (90 "
    "plates x 2 runs) with Table 1's SD; the reference has no genotype slot, so its "
    "genome states the BY4742 background and the HIS3 deletion is stated here"
)
FIVE_PERCENT_REFERENCE_UNITS = (
    REFERENCE_UNITS
    + "; for the 5% ethanol records the reference is the standard strain in the "
    "NONSTRESS condition (Table 1's nonstress row): " + FIVE_PERCENT_ETHANOL_NOTE
)


# --------------------------------------------------------------------------- #
# Media, environment, background
# --------------------------------------------------------------------------- #
YOSHIKAWA_YPD = restated(YPD_LIQUID, YPD_RECIPE)
BACKGROUND_STRAIN = "BY4742"


def _background_allele(allele: BackgroundAllele) -> BackgroundAllele:
    """The allele with a pending-review gap on its (unpublished) deleted interval."""
    return BackgroundAllele.model_validate(
        {
            **allele.model_dump(),
            "provenance_gaps": [
                pending_source_review(
                    "deleted_span",
                    BRACHMANN_1998,
                    "the designation is quoted from the paper's strain sentence; the "
                    "deleted interval is the literature-standard reading of Brachmann "
                    "1998, not mirrored",
                ).model_dump()
            ],
        }
    )


def yoshikawa_background() -> EukaryoticStrainBackground:
    """BY4742 with its quoted alleles; that a given strain is BY4742 is a typed gap."""
    genotype = STANDARD_BY_GENOTYPES[BACKGROUND_STRAIN]
    alleles = [
        _background_allele(
            standard_allele(allele_name, zygosity, provenance=[BY4742_GENOTYPE])
        )
        for allele_name, zygosity in genotype.alleles.items()
    ]
    return EukaryoticStrainBackground(
        reference_strain="S288C",
        assembly_set=SGD_S288C_R64,
        name=BACKGROUND_STRAIN,
        parents=list(genotype.parents),
        construction=None,
        mating_type=MatingType.alpha,
        ploidy="haploid",
        alleles=alleles,
        provenance=None,
        provenance_gaps=[
            ProvenanceGap(
                field="provenance",
                reason=ProvenanceGapReason.not_reported_by_primary,
                looked_in=Provenance(
                    source_uri=PAPER_MD,
                    citation_key=CITATION_KEY,
                    sha256=PAPER_MD_SHA256,
                    page="Medium and yeast strains: 'derived from S. cerevisiae BY4742 "
                    "(...) or BY4739 (...)'",
                ),
                note="the paper says each deletion strain derives from BY4742 or from "
                "BY4739 and Table S1 does not say which; BY4742 is the background of "
                "the standard strain and of most of the collection, and BY4739 carries "
                "HIS3, so his3Δ1 may not hold for every strain",
            )
        ],
    )


BY4742_GENOME = StrainReferenceGenome(
    species="Saccharomyces cerevisiae",
    strain=BACKGROUND_STRAIN,
    ploidy="haploid",
    background=yoshikawa_background(),
    assembly_set=SGD_S288C_R64,
    assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
)


def _culture_format() -> CultureFormat:
    """The static 96-well culture after the 25 uL addition."""
    return CultureFormat(
        vessel=VESSEL.value,
        working_volume_ul=WORKING_VOLUME_UL.value,
        shaking_rpm=STATIC_CULTURE.value,
        endpoint=EndpointRule.fixed_duration,
        provenance=[VESSEL, WORKING_VOLUME_UL, STATIC_CULTURE, NONSTRESS_WATER],
        provenance_gaps=[
            ProvenanceGap(
                field="inoculum_od600",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note="'2 uL of preculture' after 24 h; no OD or cell count is stated",
            )
        ],
    )


def _pre_culture() -> PreCulture:
    """24 h in 100 uL YPD at 30 C, shaken (TITRAMAX1000, 1050 strokes/min)."""
    return PreCulture(
        source=PRECULTURE.value,
        medium=YOSHIKAWA_YPD,
        duration_hours=PRECULTURE_HOURS.value,
        provenance=[PRECULTURE],
    )


def _stress(spec: ConditionSpec) -> list[EnvironmentPerturbationType]:
    """The condition's one added compound, or nothing for the nonstress control."""
    if spec.compound_label is None or spec.dose is None:
        return []
    return [
        SmallMoleculePerturbation(
            compound=resolved_compound(spec.compound_label),
            concentration=spec.dose,
            solvent=None,
            provenance_gaps=[
                ProvenanceGap(
                    field="solvent",
                    reason=ProvenanceGapReason.not_reported_by_primary,
                    note="added as '25 uL of 40% (v/v) ethanol' or '25 uL of 5 M NaCl'; "
                    "the stock's diluent is not named",
                )
            ],
        )
    ]


def environment(spec: ConditionSpec) -> CultureEnvironment:
    """The condition's culture: liquid YPD, 30 C, static, read to its last OD.

    A condition the paper gives no read time for (the 5% ethanol columns) carries a
    typed gap on ``duration_hours``.
    """
    return CultureEnvironment(
        media=YOSHIKAWA_YPD,
        temperature=Temperature(value=TEMPERATURE_C.value),
        perturbations=_stress(spec),
        aerobicity=AEROBIC.value,
        duration_hours=spec.duration_hours,
        culture_format=_culture_format(),
        pre_culture=_pre_culture(),
        provenance_gaps=[]
        if spec.duration_gap is None
        else [
            ProvenanceGap(
                field="duration_hours",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note=spec.duration_gap,
            )
        ],
    )


def reference(
    spec: ConditionSpec, dataset_name: str
) -> StrainEnvironmentResponseExperimentReference:
    """The standard strain in the reference condition: Table 1's rate and SD over 720
    wells. The condition's own, except for the 5% ethanol columns, which Table 1 has no
    row for: their reference is the standard strain in the nonstress condition.
    """
    own = spec.reference_key == spec.key
    reference_spec = spec if own else CONDITION_BY_KEY[spec.reference_key]
    return StrainEnvironmentResponseExperimentReference(
        dataset_name=dataset_name,
        genome_reference=BY4742_GENOME,
        environment_reference=environment(reference_spec),
        phenotype_reference=EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.growth_rate,
            assay_type=AssayType.liquid_od_growth,
            environment_response=spec.table1.average,
            environment_response_uncertainty=spec.table1.sd,
            environment_response_uncertainty_type=TABLE1_IS_SAMPLE_SD.value,
            n_samples=STANDARD_WELLS.value,
            sample_unit=SampleUnit.biological_replicate,
            units=REFERENCE_UNITS if own else FIVE_PERCENT_REFERENCE_UNITS,
        ),
    )


def phenotype(rate: float, replicate: int, call: int) -> EnvironmentResponsePhenotype:
    """One culture's released rate, with the strain's call in the condition."""
    category, label = CALLS[call]
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.growth_rate,
        assay_type=AssayType.liquid_od_growth,
        environment_response=rate,
        category=category,
        category_label=label,
        n_samples=1,
        sample_unit=SampleUnit.biological_replicate,
        replicate_id=REPLICATE_IDS[replicate],
        units=MEASUREMENT_UNITS,
        provenance_gaps=[
            ProvenanceGap(
                field=field,
                reason=ProvenanceGapReason.not_reported_by_primary,
                note="one culture's three-point regression slope; no error of the "
                "slope is released",
            )
            for field in ("environment_response_uncertainty", "environment_response_se")
        ],
    )


def stored_orf(released: str) -> str:
    """The release's ORF cell as stored: upper-cased (``YAL043C-a`` -> ``YAL043C-A``)."""
    return released.upper()


def genotype(row: ReleaseRow) -> Genotype:
    """One kanMX4 deletion of the Open Biosystems copy of the YKO MATalpha collection,
    named exactly as the release names the strain (no genome is consulted).
    """
    return Genotype(
        perturbations=[
            BarcodedKanMxDeletionPerturbation(
                systematic_gene_name=stored_orf(row.orf),
                perturbed_gene_name=row.name,
                collection=COLLECTION.value,
                cassette=KANMX4_CASSETTE.value,
                provenance_gaps=[
                    pending_source_review(
                        "barcode",
                        GIAEVER_2002,
                        "the strains were grown one per well and no barcode was read; "
                        "the YKO barcode table gives the UPTAG by ORF",
                    ),
                    pending_source_review(
                        "downtag_barcode",
                        GIAEVER_2002,
                        "the YKO barcode table gives the DNTAG by ORF",
                    ),
                ],
            )
        ]
    )


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """``DATA_ROOT`` from the environment (the mirrors and the build tree live under it)."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/yoshikawaComprehensivePhenotypicAnalysis2009``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def _sha256(path: str | Path) -> str:
    """Streaming sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publisher_retrieval() -> RetrievalRecord:
    """The owner's manual browser download of the publisher's Table S1."""
    return RetrievalRecord(
        method=RetrievalMethod.manual_browser,
        source_url=PUBLISHER_URL,
        retriever="manual",
        params={
            "retrieval_command": MANUAL_RECIPE,
            "retrieved_by": "the owner (mjvolk3), browser download 2026-10-12 03:41 "
            "CDT, copied into the raw mirror unchanged",
            "deposit_record": "si/DEPOSIT.md",
            "checksums": "si/SHA256SUMS.txt",
        },
        sha256=TABLE_S1.sha256,
        retrieved_at=RETRIEVED_AT,
    )


def yeastphenome_retrieval() -> RetrievalRecord:
    """yp-data v1.0's copy, scriptable at the commit the v1.0 tag names."""
    return RetrievalRecord(
        method=RetrievalMethod.direct_url,
        source_url=YP_DATA_URL,
        retriever="torchcell.literature.retrieve.direct_url",
        params={"url": YP_DATA_URL},
        sha256=TABLE_S1_YEASTPHENOME.sha256,
        retrieved_at=RETRIEVED_AT,
    )


#: What the manifest says about the files this loader does not read.
NOT_CONSUMED: tuple[str, ...] = (
    "si/FYR_456_sm_tableS1.xlsx -- YeastPhenome's copy of Table S1 (yp-data v1.0, "
    "re-saved by a curator in 2013): a cross-check, cell-identical to the publisher "
    ".xls over all 56,748 data cells; the loader reads the .xls",
)


def write_raw_manifest(data_root: str | None = None) -> Path:
    """Write the raw mirror's ``manifest.json`` for the two deposited Table S1 files.

    Both files must already be in ``si/`` with their pinned sha256 and size; a missing or
    different file refuses before anything is written. Idempotent: the manifest is
    rewritten from the pins, never from what is on disk.
    """
    root = raw_mirror_dir(data_root)
    for raw in (TABLE_S1, TABLE_S1_YEASTPHENOME):
        path = root / raw.raw_rel
        if not path.exists():
            raise RuntimeError(f"raw mirror is missing {path}")
        got = _sha256(path)
        if got != raw.sha256:
            raise RuntimeError(f"{path} sha256 {got} is not the pinned {raw.sha256}")
        if path.stat().st_size != raw.bytes:
            raise RuntimeError(f"{path} is not the pinned {raw.bytes} bytes")
    records = [
        ArtifactRecord(
            path=TABLE_S1.raw_rel,
            role=ROLE_RAW_DATA,
            bytes=TABLE_S1.bytes,
            sha256=TABLE_S1.sha256,
            source=PUBLISHER_URL,
            original_filename=TABLE_S1.filename,
            retrieval=publisher_retrieval(),
        ),
        ArtifactRecord(
            path=TABLE_S1_YEASTPHENOME.raw_rel,
            role=ROLE_SI_DATA,
            bytes=TABLE_S1_YEASTPHENOME.bytes,
            sha256=TABLE_S1_YEASTPHENOME.sha256,
            source=YP_DATA_URL,
            original_filename=TABLE_S1_YEASTPHENOME.filename,
            retrieval=yeastphenome_retrieval(),
        ),
    ]
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=DOI,
        title=TITLE,
        zotero_item_key=ZOTERO_ITEM,
        files=records,
        si_data_sources=[PUBLISHER_URL, YP_DATA_URL],
        si_expected=list(NOT_CONSUMED),
        provenance_complete=True,
        created_at=datetime.now(UTC).isoformat(),
    )
    out = root / "manifest.json"
    out.write_text(manifest.model_dump_json(indent=2))
    return out


def load_manifest(data_root: str | None = None) -> Manifest:
    """Read the raw mirror's ``manifest.json``."""
    path = raw_mirror_dir(data_root) / "manifest.json"
    if not path.exists():
        raise RuntimeError(
            f"raw mirror manifest missing: {path}; run write_raw_manifest() first"
        )
    return Manifest.model_validate_json(path.read_text())


def manifest_sha256(manifest: Manifest, relpath: str) -> str:
    """The recorded sha256 of one mirror file."""
    for record in manifest.files:
        if record.path == relpath:
            return record.sha256
    raise KeyError(f"{relpath} is not in the raw-mirror manifest")


# --------------------------------------------------------------------------- #
# Ledger
# --------------------------------------------------------------------------- #
class DropRule(BaseModel):
    """One retention rule, the records it removed, and what it removed them for."""

    rule: str
    scope: Literal["condition", "strain"]
    description: str
    n_records: int
    items: list[str] = []


class DropLog(BaseModel):
    """Every retention rule applied to a build, in order (none since #1045)."""

    dataset: str
    source_records: int
    kept_records: int
    dropped_records: int
    unperturbed_records: int
    rules: list[DropRule]


@register_dataset
class EnvChemgenYoshikawa2009Dataset(ExperimentDataset):
    """Yoshikawa 2009 growth rates: env x geno -> one culture's specific growth rate."""

    organism = Organism.scerevisiae

    def __init__(
        self,
        root: str = "data/torchcell/env_chemgen_yoshikawa2009",
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset (no genome: every strain is stored as released)."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return StrainEnvironmentResponseExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return StrainEnvironmentResponseExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The publisher's Table S1 under its own file name."""
        return [TABLE_S1.filename]

    def download(self) -> None:
        """Link the publisher's Table S1 into ``raw/`` after verifying it against its pin.

        The pin is canonical; the manifest is the retrieval record and must carry the
        same digest, else ``ManifestPinMismatchError`` names both.
        """
        data_root = _data_root()
        manifest = load_manifest(data_root)
        check_manifest_pin(
            TABLE_S1.raw_rel,
            manifest_sha256(manifest, TABLE_S1.raw_rel),
            TABLE_S1.sha256,
        )
        src = raw_mirror_dir(data_root) / TABLE_S1.raw_rel
        if not src.exists():
            raise RuntimeError(f"required raw artifact missing from mirror: {src}")
        os.makedirs(self.raw_dir, exist_ok=True)
        link_verified(src, osp.join(self.raw_dir, TABLE_S1.filename), TABLE_S1.sha256)
        log.info(
            "Yoshikawa 2009 Table S1 linked into %s (sha256 verified)", self.raw_dir
        )

    # ---- the file boundary (tests stub these) ------------------------------------ #
    def _read_release(self) -> list[ReleaseRow]:
        return read_release(osp.join(self.raw_dir, TABLE_S1.filename))

    # ---- build ------------------------------------------------------------------- #
    @post_process
    def process(self) -> None:
        """One record per (strain x condition x culture), every released cell; LMDB + log."""
        verify_raw_files(self.raw_dir, {TABLE_S1.filename: TABLE_S1.sha256})
        rows = self._read_release()
        check_release(rows)
        stored = [stored_orf(row.orf) for row in rows]
        if len(set(stored)) != len(stored):
            raise RuntimeError("two released ORFs coincide once upper-cased")
        publication = Publication(
            pubmed_id=PMID,
            pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{PMID}/",
            doi=DOI,
            doi_url=f"https://doi.org/{DOI}",
        )
        references = {spec.key: reference(spec, self.name) for spec in CONDITIONS}
        environments = {spec.key: environment(spec) for spec in CONDITIONS}

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        idx = 0
        n_unperturbed = 0
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for row in tqdm(rows, desc="yoshikawa2009"):
                strain = genotype(row)
                for spec in CONDITIONS:
                    cells = row.conditions[spec.label]
                    for replicate, rate in enumerate(cells.rates):
                        experiment = StrainEnvironmentResponseExperiment(
                            dataset_name=self.name,
                            genotype=strain,
                            environment=environments[spec.key],
                            phenotype=phenotype(rate, replicate, cells.call),
                        )
                        txn.put(
                            f"{idx}".encode(),
                            self._intern_record(
                                experiment, references[spec.key], publication, itxn
                            ),
                        )
                        idx += 1
                        n_unperturbed += spec.compound_label is None
        env.close()
        interned_env.close()

        drop_log = self._drop_log(rows, idx, n_unperturbed)
        with open(osp.join(self.preprocess_dir, "dropped_records.json"), "w") as handle:
            handle.write(drop_log.model_dump_json(indent=2))
        log.info(
            "Yoshikawa2009: wrote %d records (%d strains x %d conditions x %d cultures)",
            idx,
            len(rows),
            len(CONDITIONS),
            len(REPLICATE_IDS),
        )

    def _drop_log(self, rows: list[ReleaseRow], kept: int, unperturbed: int) -> DropLog:
        """The accounting: every released cell is a record, so no rule drops any."""
        source_records = len(rows) * len(RELEASE_CONDITIONS) * len(REPLICATE_IDS)
        log_ = DropLog(
            dataset=self.name,
            source_records=source_records,
            kept_records=kept,
            dropped_records=source_records - kept,
            unperturbed_records=unperturbed,
            rules=[],
        )
        if log_.dropped_records:
            raise RuntimeError(
                f"{log_.dropped_records} released cells have no record and no rule"
            )
        return log_

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def _condition_of_edit() -> dict[tuple[str, float] | None, str]:
    """Each condition by its one added compound and dose, as the stored environment
    names them (ethanol is added at two doses); None is the nonstress condition.
    """
    out: dict[tuple[str, float] | None, str] = {}
    for spec in CONDITIONS:
        if spec.compound_label is None or spec.dose is None:
            out[None] = spec.label
            continue
        if spec.dose.value is None:
            raise RuntimeError(f"{spec.label}: a stress dose has no value")
        name = resolved_compound(spec.compound_label).name
        out[(name, spec.dose.value)] = spec.label
    return out


CONDITION_OF_EDIT = _condition_of_edit()


def condition_label(environment_: Mapping[str, Any]) -> str:
    """The released condition a stored environment is, read off its one compound edit."""
    edits = environment_["perturbations"]
    if len(edits) > 1:
        raise RuntimeError(f"an environment carries {len(edits)} edits, not 0 or 1")
    if not edits:
        return CONDITION_OF_EDIT[None]
    edit = edits[0]
    return CONDITION_OF_EDIT[
        (edit["compound"]["name"], float(edit["concentration"]["value"]))
    ]


def release_fidelity(
    records: list[dict[str, Any]], rows: list[ReleaseRow]
) -> LevelResult:
    """L2: every stored rate, call and name is its released Table S1 cell, and every
    released cell has its record.

    A record is matched to its release row by the stored ORF (the release's ORF cell
    upper-cased), and its ``perturbed_gene_name`` must be that row's name cell.
    """
    by_orf = {stored_orf(row.orf): row for row in rows}
    expected = len(rows) * len(CONDITIONS) * len(REPLICATE_IDS)
    seen: set[tuple[str, str, str]] = set()
    mismatches: list[str] = []
    for record in records:
        experiment = record["experiment"]
        (perturbation,) = experiment["genotype"]["perturbations"]
        orf = perturbation["systematic_gene_name"]
        row = by_orf.get(orf)
        if row is None or perturbation["perturbed_gene_name"] != row.name:
            mismatches.append(f"{orf}: not a released strain as named")
            continue
        label = condition_label(experiment["environment"])
        phenotype_ = experiment["phenotype"]
        replicate = str(phenotype_["replicate_id"])
        key = (orf, label, replicate)
        seen.add(key)
        cells = row.conditions[label]
        category, category_label = CALLS[cells.call]
        stored = (
            float(phenotype_["environment_response"]),
            str(phenotype_["category"]),
            phenotype_["category_label"],
        )
        want = (
            cells.rates[REPLICATE_IDS.index(replicate)],
            category.value,
            category_label,
        )
        if stored != want:
            mismatches.append(f"{key}: stored {stored}, released {want}")
    passed = not mismatches and len(seen) == len(records) == expected
    return LevelResult(
        level=Level.L2,
        name="release_fidelity",
        passed=passed,
        message=(
            f"all {len(records)} stored rates and calls equal their released Table S1 "
            f"cells, one record per released (strain, condition, culture)"
            if passed
            else f"{len(mismatches)} records differ from the release; {len(seen)} "
            f"distinct cells over {len(records)} records, {expected} expected: "
            f"{mismatches[:5]}"
        ),
        details={
            "n_records": len(records),
            "n_distinct_cells": len(seen),
            "n_expected": expected,
            "n_mismatched": len(mismatches),
        },
    )


def verify_build(dataset_root: str, data_root: str | None = None) -> Any:
    """The family's L0-L4 gate for this store plus the release-fidelity row.

    Runs ``verify_environment_response_entry`` (the registry entry declares the absolute
    readout and the nonstress records), then re-reads the publisher Table S1 from the
    raw mirror and compares every stored cell with it. What ``build_dataset_lmdb
    --verify`` calls.
    """
    from torchcell.verification.runners import (
        load_records,
        verify_environment_response_entry,
    )

    root = data_root or _data_root()
    report = verify_environment_response_entry(REGISTRY_NAME, root)
    rows = read_release(raw_mirror_dir(root) / TABLE_S1.raw_rel)
    report.add(release_fidelity(load_records(dataset_root), rows))
    return report


def main() -> None:
    """Build/load the dataset for interactive debugging."""
    from dotenv import load_dotenv

    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    root = osp.join(data_root, DATASET_ROOT_REL)
    dataset = EnvChemgenYoshikawa2009Dataset(root=root)
    print(f"len = {len(dataset)}")
    print(dataset[0])


if __name__ == "__main__":
    main()
