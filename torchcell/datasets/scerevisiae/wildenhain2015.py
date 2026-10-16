# torchcell/datasets/scerevisiae/wildenhain2015
# [[torchcell.datasets.scerevisiae.wildenhain2015]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/wildenhain2015
# Test file: tests/torchcell/datasets/scerevisiae/test_wildenhain2015.py
"""Wildenhain 2015/2016 chemical-genetic matrix (CGM): env x geno -> z-score response.

Wildenhain et al. 2015 (Cell Systems, doi:10.1016/j.cels.2015.12.003) screened 4,915
compounds from four libraries against 195 sentinel deletion strains at a single 20 uM
screening concentration, reading out growth inhibition as a normalized-OD600 Z-score. The
RELEASED data (PubChem AID 1159580) is the EXTENDED CGM of the 2016 Scientific Data
descriptor (Wildenhain et al. 2016, doi:10.1038/sdata.2016.95, mirrored as
``wildenhainSystematicChemicalgeneticChemicalchemical2016``): "the number of sentinels has
been increased from 195 to 242 yeast deletion strains" and a fifth library (Bioactive 2,
892 compounds) was added, giving "5,518 unique compounds, 242 sentinel strains and
duplicate measurements for 492,126 pairwise chemical-gene interaction tests"; its Data
citation 1 is "NCBI PubChem BioAssay 1159580". So 242 is not an unexplained excess over
195: the 47 extra strains are the 2016 extension. The 2016 paper is the methods source for
the protocol values below (temperature, medium, culture format, DMSO fraction, the BY4741
genotype, the N(1, IQR) z rule). This loader builds ONLY the CGM; the 128x128 cryptagen
chemical-chemical layer is out of scope.

GENOTYPES (#504). Both papers call every sentinel a Euroscarf deletion strain isogenic to
BY4741, whose genotype the 2016 paper states ("MATa his3Δ1 leu2Δ0 met15Δ0 ura3Δ0"); that
background rides on the reference's ``StrainReferenceGenome``. Each record's genotype is:

- a ``BarcodedKanMxDeletionPerturbation`` (cassette kanMX4, sourced to Giaever 2014; the
  release publishes no barcode, a typed gap) for every one of the 242 released ORFs,
  keyed by the released ``orf`` with the released ``sym`` of the row as
  ``perturbed_gene_name``, both verbatim. 33 of the 242 ORFs are SGD-essential
  (``ESSENTIAL_GENE_ORFS``, kept as the documented contradiction): a haploid kanMX null
  of an essential gene is not viable, so those strains must carry some other allele, but
  both papers call every sentinel a Euroscarf deletion strain and the release names no
  allele. The owner rule of 2026-10-12 stores what the authors reported and files the
  contradiction in the dataset note, so they are stored as the deletions the release
  calls them, never re-typed by an annotation at load time;
- the EMPTY genotype (the BY4741 background alone) for the released ``wild type`` screen.

The released rows whose ``orf`` is not a systematic name are strain screens, not
controls: ``NULL/wild type`` is the BY4741 screen; ``NA/NNK1`` is mapped to YKL171W
(a synonym route that the released-identifier leaf of #1067's yeast analogue replaces
once it exists; until then the mapping keeps the 678 rows served); ``NA/TSCII``,
``NULL/YGL11`` and ``NULL/wtn01`` name no strain a mirrored source identifies and are
held in the ledger under ``strain_label_unresolved`` for the same leaf. No genome is
read at build time.

DATA SOURCE. PubChem BioAssay AID 1159580 (the chemgrid.org/cgm portal is an interactive
PHP site with no bulk export). Two artifacts are mirrored and sha256-pinned: the per-AID
datapoint export ``1159580.csv.gz``, read out of the byte-stable NCBI FTP range archive with
the container's OWN sha256 asserted first (``zip_member``), and the AID's PUG-REST
description JSON (column definitions, the protocol as submitted). Both are read from the raw
mirror at build time; the URLs are retrieval metadata, not live dependencies.

WHAT THE NUMBER IS. The stored response is the released per-screen ``z_score``, one
record per released screen (the paper averaged them for its matrix, "Z scores were
calculated and averaged for the replicate screens"; the release carries the per-screen
values, so those are what is stored and the mean is a modeling-time choice). The z is computed by
"fitting a normal distribution with N(1,IQR)" to the median-normalized OD of ONE strain's
screen, so it is IQR-scaled and its 0 is that SAME strain's plate-median growth, not
BY4741 under the compound. The reference therefore states the vehicle (no-compound)
environment with z = 0; ``ExperimentReference`` has no genotype slot, so "the same
deletion strain" lives in the reference's ``units`` text (the schema gap is recorded in the
dendron note).

SCREEN COUNTING. A (strain, compound) cell recurs across the libraries, and the export
ALSO re-emits the same datapoint under two gene-symbol spellings (``MDH1`` / ``mdh1``).
Within a cell, two rows sharing a z_score share every other data column too, and no two
genuinely distinct datapoints of a cell share a z_score, so the PARSED z value is the
datapoint key (``-4.0`` and ``-4.00`` are one datapoint). A re-exported row is a true
duplicate of the release and collapses onto the first row carrying it (its
``PUBCHEM_RESULT_TAG`` is listed on the datapoint); every DISTINCT datapoint of a cell is
its own record, with ``replicate_id`` = the ``PUBCHEM_RESULT_TAG`` of the row it came
from, ``n_samples = 1`` with ``sample_unit=screen`` (the two OD reads inside a screen are
the technical duplicate the z already averages), the row's own released call as the
category, and typed gaps on the per-screen uncertainty (the release carries none) and on
``screen_id`` (no library or screen column). Earlier builds averaged the 2 to 5 screens
of a cell into one record with a sample SD; under the owner rule of 2026-10-12 every
released screen is stored as released.

RECORDS DROPPED (rule + count in ``preprocess/dropped_records.json``, counted in
released rows and applied in this order): ``strain_label_unresolved`` (the TSCII, YGL11
and wtn01 rows, held for the released-identifier leaf), then
``compound_has_no_released_name_or_structure_not_representable``
(``UNIDENTIFIED_COMPOUND_RULE``), then ``re_exported_duplicate_row`` (the release's own
byte-identical re-emissions). The ledger proves ``source_records = kept_records +`` the
three rules and the build refuses on a mismatch.

UNIDENTIFIED PERTURBING AGENTS (database constitution principle 2, owner ruling
2026-10-15: "if you cannot tell me what goes into the flask so to speak, then we have no
hope of modeling!"). The release's compound columns are ``PUBCHEM_SID``, ``PUBCHEM_CID``
and ``PUBCHEM_EXT_DATASOURCE_SMILES``; it has no name column
(``RELEASED_COMPOUND_IDENTIFIER``), and the authors put structures on the unmirrored
chemgrid.org portal instead (``STRUCTURES_ON_CHEMGRID``). The criterion is mechanical: a
cell's compound is identified iff its row carries a CID or a SMILES, the pinned
compound-identity table has a curated row for its label, or ``RELEASED_IDENTITIES`` (the
admission table, EMPTY today) has an entry for it. Five SIDs fail all four (SID
268734404, 268735390, 268737906, 277425705, 277425722): 410 released rows with no CID,
SMILES, URL or comment (only ``cryptagen = 0``), of which 4 sit on an unresolved strain
label and the other 406 rows (378 distinct screens over 368 cells) are HELD OUT under
``UNIDENTIFIED_COMPOUND_RULE``, never stored under the bare ``SID n`` label with an
``inchikey`` gap. Each is listed under ``unidentified_compounds_held_out``, which is the
request list for a public SID-to-structure release; only such a release fills
``RELEASED_IDENTITIES`` (never an identity inferred from profile similarity), and a filled
entry rebuilds the store. Every one of the 5,173 released CIDs carries a SMILES and
resolves to an InChIKey and a canonical PubChem name through the pinned table, so no CID
compound is gapped. Measured on the pinned export (sha256 ``c461c679...``) by
``_collapse_matrix`` plus the rules above: 492,126 source rows = 448,655 stored records +
4,678 strain-label rows + 406 unidentified-compound rows + 38,387 re-exported rows.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import logging
import math
import os
import os.path as osp
import re
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    Organism,
    check_manifest_pin,
    link_verified,
    post_process,
    verify_raw_files,
    verify_sha256,
)
from torchcell.datamodels.compound_identity import (
    resolve_compound_identity,
    resolved_compound,
)
from torchcell.datamodels.media import SC
from torchcell.datamodels.schema import (
    AssayType,
    BackgroundAllele,
    BarcodedKanMxDeletionPerturbation,
    Compound,
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
    MeasurementType,
    Media,
    MediaComponent,
    PreCulture,
    PreCultureSource,
    Publication,
    ResponseCategory,
    SampleUnit,
    SmallMoleculePerturbation,
    Solvent,
    StrainEnvironmentResponseExperiment,
    StrainEnvironmentResponseExperimentReference,
    StrainReferenceGenome,
    Temperature,
)
from torchcell.datamodels.strain_background import (
    BRACHMANN_1998,
    GIAEVER_2002,
    KANMX4_CASSETTE,
    pending_source_review,
    standard_background,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.datasets.scerevisiae.hoepfner2014 import ReleasedIdentity
from torchcell.literature.manifest import (
    ROLE_RAW_DATA,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.sequence.genome.registry import (
    SGD_S288C_R64,
    SGD_S288C_R64_GENBANK_ACCESSION,
)
from torchcell.verification.report import Provenance
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
CITATION_KEY = "wildenhainPredictionSynergismChemicalGenetic2015"
PAPER_DOI = "10.1016/j.cels.2015.12.003"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"

FTP_ZIP_URL = (
    "https://ftp.ncbi.nlm.nih.gov/pubchem/Bioassay/CSV/Data/1159001_1160000.zip"
)
ZIP_MEMBER = "1159001_1160000/1159580.csv.gz"
#: sha256 of the FTP range archive itself, asserted BEFORE the member is read, so a
#: re-packed upstream archive fails loudly instead of yielding different member bytes.
CONTAINER_SHA256 = "d1fd5dc2bf7c526ad9845e0a14ae9981256fb820aaf4228b48a3ba0724ee59b0"
DATA_FILENAME = "1159580.csv.gz"
DATA_SHA256 = "c461c679b63ac56045cef0f03ed9bcbb8e7f9c12146f1fc7cc8ac0c113188d64"
DATA_REL = f"data/{DATA_FILENAME}"

AID_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/1159580/description/JSON"
AID_FILENAME = "aid_1159580_description.json"
AID_SHA256 = "23c5f8c56af94786cfe8e22c93fdde0b719ca2165975305944557ab39087b0e4"
AID_REL = f"data/{AID_FILENAME}"
RETRIEVED_AT = "2026-09-13"

PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "f46409eb8f23412c9c1015d0f8f5bb581bfddfe2796d319d407585e23c757ac2"

#: The 2016 Scientific Data descriptor of the EXTENDED CGM, which is what AID 1159580
#: releases (its Data citation 1). Mirrored in the library; the protocol values below are
#: quoted from its OCR.
SCIDATA_CITATION_KEY = "wildenhainSystematicChemicalgeneticChemicalchemical2016"
SCIDATA_DOI = "10.1038/sdata.2016.95"
SCIDATA_MD_SHA256 = "89ff4d9bf1d31719ab15c18ab7aca0b7caf10f55c7c239c1b021908c95439e33"


def _aid(value: Any, quote: str, *, note: str | None = None) -> SourcedValue:
    """Bind a value to a verbatim quote in the sha256-pinned AID 1159580 description."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=AID_REL,
            citation_key=CITATION_KEY,
            sha256=AID_SHA256,
            method="PubChem PUG-REST assay description JSON (torchcell-raw mirror)",
            page="PC_AssayContainer[0].assay.descr.protocol",
            retrieved=RETRIEVED_AT,
        ),
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
            page="RESULTS 'Generation of a Chemical-Genetic Matrix' / EXPERIMENTAL PROCEDURES",
        ),
    )


def _scidata(
    value: Any, quote: str, *, line: str, note: str | None = None
) -> SourcedValue:
    """Bind a value to a verbatim quote in the sha256-pinned 2016 Sci Data OCR mirror."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=PAPER_MD,
            citation_key=SCIDATA_CITATION_KEY,
            sha256=SCIDATA_MD_SHA256,
            method="MinerU OCR of the publisher PDF (torchcell-library mirror)",
            page=f"paper.md line {line}",
        ),
    )


# --------------------------------------------------------------------------- #
# What the release is (2016 Sci Data extended CGM)
# --------------------------------------------------------------------------- #
EXTENDED_RELEASE = _scidata(
    492126,
    "This extended CGM dataset contains data for 5,518 unique compounds, 242 sentinel "
    "strains and duplicate measurements for 492,126 pairwise chemical-gene interaction "
    "tests",
    line="23",
    note="the released 1159580.csv.gz has exactly 492,126 data rows, 5,518 SIDs and 242 "
    "systematic ORFs (experiments/036-dataset-fixes-before-kg-build/scripts/"
    "wildenhain2015_inputs.py); the paper's Data citation 1 is 'NCBI PubChem BioAssay "
    "1159580 (2015).' (line 176)",
)
SENTINEL_EXTENSION = _scidata(
    242,
    "the number of sentinels has been increased from 195 to 242 yeast deletion strains",
    line="23",
    note="the 2015 Cell Systems paper's '195 non-essential deletion strains' is the "
    "original CGM; the 47 added strains and the Bioactive 2 library are the 2016 "
    "extension, so serving 242 ORFs is the release, not an excess",
)

# --------------------------------------------------------------------------- #
# Sourced environment + phenotype constants
# --------------------------------------------------------------------------- #
MEDIUM = _scidata(
    "SC",
    "All screens were conducted in synthetic complete (SC) medium with $2 \\%$ glucose.",
    line="54",
    note="the CGM screens' own medium sentence. The 2015 paper's medium sentence has "
    "'All fungal species' as its subject (the pathogen isolates), and the AID protocol "
    "says 'All strains were grown and screened in synthetic complete (SC) medium with 2% "
    "glucose.'; neither paper gives the lab's SC amino-acid recipe, so the components of "
    "WILDENHAIN_SC are the shared SC composition, unsourced for this study",
)
TEMPERATURE_C = _scidata(
    30.0,
    "All plates were incubated at $3 0 ~ ^ { \\circ } \\mathrm { C }$ without shaking for "
    "approximately $1 8 \\mathrm { { h } }$ or until solvent-treated control cultures "
    "were saturated.",
    line="54",
    note="the 2015 Cell Systems paper states no incubation temperature for the CGM; the "
    "AID protocol says the same 30 C",
)
DURATION_HOURS = _scidata(
    18.0,
    "All plates were incubated at $3 0 ~ ^ { \\circ } \\mathrm { C }$ without shaking for "
    "approximately $1 8 \\mathrm { { h } }$ or until solvent-treated control cultures "
    "were saturated.",
    line="54",
    note="approximate: the read was taken when the solvent controls saturated "
    "(ENDPOINT), so 18 h is the typical, not a fixed, duration",
)
ENDPOINT = _scidata(
    EndpointRule.until_control_saturation,
    "or until solvent-treated control cultures were saturated.",
    line="54",
)
STATIC_INCUBATION_RPM = _scidata(
    0.0,
    "All plates were incubated at $3 0 ~ ^ { \\circ } \\mathrm { C }$ without shaking",
    line="54",
    note="static during growth; the cultures were resuspended by shaking only before "
    "the OD600 read",
)
CULTURE_VESSEL = _scidata(
    "96-well plate",
    "in a screening volume of $1 0 0 \\mu \\mathrm { l }$ in 96 well plates.",
    line="54",
)
WORKING_VOLUME_UL = _scidata(
    100.0,
    "in a screening volume of $1 0 0 \\mu \\mathrm { l }$ in 96 well plates.",
    line="54",
    note="the seeded volume BEFORE the 2 uL compound addition (102 uL after it)",
)
INOCULUM_CELLS = _scidata(
    50000.0,
    "were seeded at 50,000 cells per well from fresh overnight cultures",
    line="54",
)
PRE_CULTURE_SOURCE = _scidata(
    PreCultureSource.overnight_culture,
    "were seeded at 50,000 cells per well from fresh overnight cultures",
    line="54",
    note="the overnight culture's medium, duration and phase are not stated",
)
SCREEN_CONCENTRATION_UM = _paper(
    20.0,
    "We carried out over 600 growth-based screens in duplicate at a compound "
    "concentration of $2 0 \\mu \\mathsf { M }$ .",
    note="the 2016 paper states it as a preparation: '$2 \\mu \\mathrm { l }$ of $1 "
    "\\mathrm { m M }$ compound stock was added to each well for a final compound "
    "concentration of $2 0 \\mu \\mathrm { M }$ .' (paper.md line 54). 20 uM is the "
    "NOMINAL dose (2 uL x 1 mM / 100 uL); in the 102 uL after addition it is 19.6 uM",
)
SOLVENT = _scidata(
    "DMSO",
    "compound library stocks of $1 0 \\mathrm { m M }$ were diluted to $1 \\mathrm { m M "
    "}$ working stocks in DMSO in 96 well plates.",
    line="46",
)
DMSO_WORKING_STOCK_UL = _scidata(
    2.0,
    "$2 \\mu \\mathrm { l }$ of $1 \\mathrm { m M }$ compound stock was added to each "
    "well for a final compound concentration of $2 0 \\mu \\mathrm { M }$ .",
    line="54",
)
#: Final DMSO fraction, % v/v, derived from the two quotes above and WORKING_VOLUME_UL:
#: 2 uL of working stock into a 100 uL culture is 2 / (100 + 2) = 0.019608 of the well
#: volume, i.e. 1.96 % v/v (rounded to the two decimals the inputs support).
DMSO_PERCENT_V_V = round(
    100.0
    * DMSO_WORKING_STOCK_UL.value
    / (WORKING_VOLUME_UL.value + DMSO_WORKING_STOCK_UL.value),
    2,
)
SOLVENT_PERCENT = _scidata(
    DMSO_PERCENT_V_V,
    "$2 \\mu \\mathrm { l }$ of $1 \\mathrm { m M }$ compound stock was added to each "
    "well for a final compound concentration of $2 0 \\mu \\mathrm { M }$ .",
    line="54",
    note="derived: 100 x 2 uL / (100 uL + 2 uL) = 1.96 % v/v, with the 100 uL from "
    "'in a screening volume of $1 0 0 \\mu \\mathrm { l }$' (line 54) and the working "
    "stock 'diluted to $1 \\mathrm { m M }$ working stocks in DMSO' (line 46). The "
    "derivation treats the 1 mM working stock as neat DMSO. Hypothesis (untested): the "
    "10 mM library stocks are themselves in DMSO; the paper does not state their "
    "solvent, and if they were aqueous the fraction would be 0.9 x 1.96 = 1.76 % v/v",
)
ASSAY = _aid(
    AssayType.liquid_od_growth,
    "Cultures were resuspended by shaking on the robotic platform prior to reading OD600 "
    "values on either Tecan M1000 or Tecan Sunrise plate readers.",
)
SCREEN_REPLICATION = _aid(
    "technical duplicate",
    "Screens were conducted in technical duplicate",
    note="the duplicate is the pair of OD READS inside one screen, which the z-score "
    "already averages ('normalized average reads per screen'); the independent unit of "
    "replication is therefore the SCREEN, which is what n_samples counts",
)
Z_SCORE_DEFINITION = _scidata(
    MeasurementType.z_score,
    "Z-scores for growth inhibition were calculated based on the median and the "
    "interquartile range (IQR) by fitting a normal distribution with $\\mathrm { \\Delta N "
    "( 1 , I Q R ) }$ to the experimental data.",
    line="66",
    note="the fit is to the median-normalized OD of ONE strain's screen ('Median-"
    "normalization was applied to all plates and experiments.', line 65), so z = 0 is "
    "that strain's plate-median growth and the scale is the IQR (about 1.35 SD of a "
    "normal), not a unit variance; the paper adds 'This approach slightly underestimates "
    "the significance of the Z-scores.' The AID column caption 'Z-Score calculated "
    "based on kernel density distribution' does not describe this rule; the released z "
    "is linear in the released normalized OD with zero crossing at 1.0 "
    "(experiments/036-dataset-fixes-before-kg-build/scripts/wildenhain2015_inputs.py)",
)
NORMALIZATION_BY_LIBRARY = _scidata(
    "LOWESS + plate median, or DMSO controls, by library",
    "For the seven Bioactive 1 screens with higher hit rates, and for all Bioactive 2 "
    "screens, the data was not LOWESS corrected but was instead normalized to DMSO "
    "controls.",
    line="64",
    note="every other screen was LOWESS-corrected ('LOWESS regression was used to "
    "correct spatial effects on growth across all plates for all screens performed with "
    "the LOPAC, Maybridge Hitskit 1000 and Spectrum Collection libraries, and for all "
    "but seven screens with the Bioactive 1 library.', line 63). The release has no "
    "library, screen or plate column, so the normalization of a given cell's screens is "
    "not recoverable; each record carries a typed gap on screen_id saying so",
)
Z_AVERAGED = _paper(
    "mean over the replicate screens",
    "Z scores were calculated and averaged for the replicate screens.",
)
BY4741_GENOTYPE = _scidata(
    "BY4741",
    "were obtained from the Euroscarf deletion collection and are isogenic to BY4741, "
    "which has the genotype MATa his3Δ1 leu2Δ0 met15Δ0 ura3Δ0.",
    line="50",
    note="sources the strain name, MATa, haploidy and the four allele designations. "
    "What each designation physically is (his3Δ1 an internal deletion, the Δ0 alleles "
    "full ORF deletions) is the literature-standard reading from Brachmann 1998, not "
    "mirrored, so each allele also carries a pending-review gap on deleted_span. The AID "
    "protocol states the same string with the Δ glyphs lost ('BY 4741 MATa his31 leu20 "
    "met150 ura30')",
)
COLLECTION = _scidata(
    "Euroscarf deletion collection",
    "were obtained from the Euroscarf deletion collection and are isogenic to BY4741, "
    "which has the genotype MATa his3Δ1 leu2Δ0 met15Δ0 ura3Δ0.",
    line="50",
    note="the 2015 paper says the same: 'S. cerevisiae deletion strains were obtained "
    "from the Euroscarf deletion set and are isogenic to BY4741 (Table S3).' The 33 "
    "SGD-essential released ORFs (ESSENTIAL_GENE_ORFS) contradict 'deletion strain' for "
    "those strains; they are stored as the release reports them and the contradiction is "
    "filed in the dataset note",
)
WILD_TYPE_SCREEN = _scidata(
    "BY4741 wild type",
    "we repeatedly screened the compound collections against the pdr1Δpdr3Δ strain and "
    "a wild-type S. cerevisiae strain (BY4741)",
    line="118",
    note="the released orf=NULL / sym='wild type' rows are this strain's screens, served "
    "as the empty genotype on the BY4741 background",
)
NON_REPLICATE_COLUMN = _aid(
    "non replicate",
    "test for non-replicates between first and second replicate",
    note="a screen whose two OD reads disagreed; counting SCREENS rather than reads is "
    "what keeps this flag from corrupting n_samples. The release also states 'Data "
    "points with high variation between replicates (> 3 MAD) were removed as "
    "inconsistent outliers.', so a retained flagged screen passed that filter",
)
BIOACTIVITY_COLUMN = _aid(
    "bioactivity",
    "sensitive if compound decreases fitness or resistant if compound increases fitness "
    "compared to negative control",
    note="together with the released PUBCHEM_ACTIVITY_OUTCOME this is the source of the "
    "ResponseCategory mapping below",
)

MEASUREMENT_UNITS = (
    "released PubChem AID 1159580 z_score of growth inhibition: a N(1, IQR) fit to the "
    "median-normalized OD600 (technical-duplicate read average) of the strain's OWN "
    "screen, so 0 is that strain's plate-median growth and the scale is the screen's IQR "
    "(about 1.35 SD of a normal; not unit variance); 20 uM compound in 1.96% v/v DMSO, "
    "SC + 2% glucose, 30 C, static 100 uL 96-well culture read when the solvent controls "
    "saturated (~18 h); negative = growth inhibition; averaged over the contributing "
    "screens of the cell"
)

#: The reference's units: the baseline is the SAME strain's screen center in the
#: vehicle / plate-median environment (#504 finding 5). ``ExperimentReference`` has no
#: genotype slot, so the "same strain" half is stated here and the reference genome
#: carries the shared BY4741 background only.
REFERENCE_UNITS = (
    MEASUREMENT_UNITS
    + "; reference z = 0 is the SAME strain's own screen center (plate-median growth, "
    "normalized OD 1) with no active compound, NOT BY4741 under the compound and NOT a "
    "measured wild-type value; the reference record has no genotype slot, so its genome "
    "states the shared BY4741 background only"
)


# --------------------------------------------------------------------------- #
# Medium: a Wildenhain-local SC sourced to the CGM's own sentence
# --------------------------------------------------------------------------- #
def _wildenhain_sc() -> Media:
    """The shared SC composition, re-sourced to the 2016 CGM medium sentence.

    The shared ``SC`` object's glucose carries the 2015 paper's fungal-pathogen sentence
    and the medium-level provenance is Mormino 2022's recipe; ``SC`` is imported by five
    loaders, so it is left untouched (#504). This copy has the SAME composition (so it
    joins every SC record by ``media_identity`` and passes ``media_membership`` as
    derived from ``SC``) with the Wildenhain provenance replaced.
    """
    components = []
    for component in SC.components:
        if component.compound.name == "D-glucose":
            component = MediaComponent.model_validate(
                {
                    **component.model_dump(),
                    "provenance": [
                        _scidata(
                            "2% glucose",
                            MEDIUM.quote,
                            line="54",
                            note="2% w/v = the shared SC's 20 g/L D-glucose",
                        ).model_dump()
                    ],
                }
            )
        components.append(component)
    return Media.model_validate(
        {
            **SC.model_dump(),
            "components": [c.model_dump() for c in components],
            "provenance": [MEDIUM.model_dump()],
        }
    )


WILDENHAIN_SC = _wildenhain_sc()


# --------------------------------------------------------------------------- #
# Strain background (BY4741), the 33 essential-gene strains, non-ORF strain labels
# --------------------------------------------------------------------------- #
def _background_allele(allele: BackgroundAllele) -> BackgroundAllele:
    """The allele with a pending-review gap on its (unpublished) construction."""
    return BackgroundAllele.model_validate(
        {
            **allele.model_dump(),
            "provenance_gaps": [
                pending_source_review(
                    "deleted_span",
                    BRACHMANN_1998,
                    "the designation is quoted (Sci Data line 50); the edit kind and "
                    "the deleted interval are the literature-standard reading of "
                    "Brachmann 1998, not mirrored",
                ).model_dump()
            ],
        }
    )


def _by4741_background() -> EukaryoticStrainBackground:
    """BY4741 as the 2016 paper states it, each allele's construction pending review."""
    background = standard_background("BY4741", provenance=[BY4741_GENOTYPE])
    return EukaryoticStrainBackground.model_validate(
        {
            **background.model_dump(),
            "alleles": [_background_allele(a).model_dump() for a in background.alleles],
        }
    )


BY4741_GENOME = StrainReferenceGenome(
    species="Saccharomyces cerevisiae",
    strain=BY4741_GENOTYPE.value,
    ploidy="haploid",
    background=_by4741_background(),
    assembly_set=SGD_S288C_R64,
    assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
)

#: The 33 released ORFs that are essential in torchcell's SGD essentiality set (#504
#: finding 1): a haploid kanMX null of an essential gene is not viable, so these strains
#: must carry some other allele the release does not name. They are STORED as the
#: deletions the release calls them (owner rule 2026-10-12: a reported value that
#: contradicts an annotation is stored as reported, the contradiction filed in the note).
#: Pinned here as documentation only; nothing at build time reads it. The test recomputes
#: it from ESSENTIAL_GENE_SET_SOURCE when that file is present. ILV3 (YJR016C) and SLN1
#: (YIL147C) are the two whose essentiality is condition/background dependent.
ESSENTIAL_GENE_ORFS: frozenset[str] = frozenset(
    {
        "YAR019C", "YBL105C", "YBR135W", "YBR136W", "YBR160W", "YDL017W", "YDL028C",
        "YDL108W", "YDL132W", "YDR052C", "YDR054C", "YER133W", "YFL009W", "YFL029C",
        "YFR003C", "YFR028C", "YIL147C", "YJR016C", "YKL193C", "YKL203C", "YMR001C",
        "YMR277W", "YNL006W", "YNL161W", "YNL207W", "YNL222W", "YOL078W", "YOR119C",
        "YOR329C", "YPL153C", "YPL204W", "YPL209C", "YPR025C",
    }
)  # fmt: skip
ESSENTIAL_GENE_SET_SOURCE = Provenance(
    source_uri="data/torchcell/gene_essentiality_sgd/preprocess/gene_set.json",
    sha256="f3c2ec54b69f61909eaee4a380ab6d10e84265a9ee31b053b27345e7de6c5dd8",
    method="intersection of the 1,140 SGD-essential genes with the 242 systematic ORFs "
    "of 1159580.csv.gz, measured 2026-10-02 (path under $DATA_ROOT)",
)


class NonOrfStrainLabel(BaseModel):
    """How one released (orf, sym) pair whose ``orf`` is not a systematic name is read.

    ``mapped_to_orf``: a real gene's strain released without its ORF (served on that
    ORF by a pinned map, no genome read; the released-identifier leaf replaces the map).
    ``wild_type``: the BY4741 parental screen (served as the empty genotype).
    ``unresolved``: no strain the mirrored sources name; held in the ledger under
    ``strain_label_unresolved`` for the same leaf.
    """

    orf: str
    sym: str
    disposition: Literal["mapped_to_orf", "wild_type", "unresolved"]
    systematic_gene_name: str | None = None
    reason: str


NON_ORF_STRAIN_LABELS: dict[tuple[str, str], NonOrfStrainLabel] = {
    ("NA", "NNK1"): NonOrfStrainLabel(
        orf="NA",
        sym="NNK1",
        disposition="mapped_to_orf",
        systematic_gene_name="YKL171W",
        reason="NNK1 is the standard name of YKL171W in R64; its NA-labelled SIDs are "
        "disjoint from the 11 SIDs released under orf=YKL171W, so these are the rest of "
        "the same strain's screen",
    ),
    ("NULL", "wild type"): NonOrfStrainLabel(
        orf="NULL",
        sym="wild type",
        disposition="wild_type",
        reason="the BY4741 wild-type screen (WILD_TYPE_SCREEN)",
    ),
    ("NA", "TSCII"): NonOrfStrainLabel(
        orf="NA",
        sym="TSCII",
        disposition="unresolved",
        reason="no R64 gene is named TSCII. Hypothesis (untested): TSC11 (AVO3, "
        "YER093C, essential) with '11' typed as 'II'; needs Sci Data Table 1",
    ),
    ("NULL", "YGL11"): NonOrfStrainLabel(
        orf="NULL",
        sym="YGL11",
        disposition="unresolved",
        reason="not a systematic name (no strand letter, short number). Hypothesis "
        "(untested): a truncated ORF; needs Sci Data Table 1",
    ),
    ("NULL", "wtn01"): NonOrfStrainLabel(
        orf="NULL",
        sym="wtn01",
        disposition="unresolved",
        reason="Hypothesis (untested): a second wild-type screen (its z correlates with "
        "the 'wild type' screen, #504 audit); not named in a mirrored source, so not "
        "asserted to be BY4741",
    ),
}

#: The cell key of the wild-type screen (not an ORF; never a gene).
WILD_TYPE_KEY = "BY4741 wild type"

#: Released activity outcome + bioactivity -> the shared ResponseCategory axis. The two
#: columns' own definitions are the source (``BIOACTIVITY_COLUMN``): an Inactive datapoint
#: is one the screen could not distinguish from the negative control, an Active one is a
#: called hit whose DIRECTION the bioactivity column gives, and Inconclusive is PubChem's
#: own "no call" verdict -- which is ``not_determined``, never silently an Inactive.
OUTCOME_CATEGORY: dict[tuple[str, str], ResponseCategory] = {
    ("Inactive", ""): ResponseCategory.no_change,
    ("Active", "sensitive"): ResponseCategory.sensitive,
    ("Active", "resistant"): ResponseCategory.resistant,
    ("Inconclusive", "sensitive"): ResponseCategory.not_determined,
    ("Inconclusive", "resistant"): ResponseCategory.not_determined,
    ("Inconclusive", ""): ResponseCategory.not_determined,
}

# --------------------------------------------------------------------------- #
# Unidentified perturbing agents (database constitution principle 2, 2026-10-15)
# --------------------------------------------------------------------------- #
UNIDENTIFIED_COMPOUND_RULE = (
    "compound_has_no_released_name_or_structure_not_representable"
)

#: The release's compound columns, as its own column caption defines its identifier.
RELEASED_COMPOUND_IDENTIFIER = _aid(
    "PubChem SID / CID / SMILES; no name column",
    "Pubchem_ext_datasource_regid: compound identifier used in on chemgrid.org/cgm",
    note="the per-compound columns of 1159580.csv.gz are PUBCHEM_SID, PUBCHEM_CID and "
    "PUBCHEM_EXT_DATASOURCE_SMILES; the release has no compound-name column. A row with "
    "no CID and no SMILES therefore names its compound only by its PubChem substance "
    "registry number (the SID)",
)
#: Where the authors put structures instead: a web portal, not the release.
STRUCTURES_ON_CHEMGRID = _scidata(
    "structures on chemgrid.org, not in the release",
    "Library compositions and chemical structures are available at www.chemgrid.org.",
    line="46",
    note="chemgrid.org is an interactive portal with no bulk export and is not "
    "mirrored, so it is no source of identity for a compound the PubChem release leaves "
    "without a CID or SMILES",
)

UNIDENTIFIED_COMPOUND_POLICY = (
    "a released compound identity with no PUBCHEM_CID, no PUBCHEM_EXT_DATASOURCE_SMILES "
    "(the release has no name column), no curated compound-identity row for its label "
    "and no RELEASED_IDENTITIES entry is a PubChem substance registry number only: what "
    "went into the well cannot be stated, so the experiment cannot be represented "
    "(database constitution principle 2, owner ruling 2026-10-15). Its rows are HELD "
    "OUT, never stored under the bare 'SID n' label with an inchikey gap; each is "
    "listed with its released rows, cells and distinct screens, and the label is "
    "admitted the day a public release gives its name or structure. A CID is a "
    "standardized PubChem structure record, so every CID row is identified and stored"
)

#: The ADMISSION TABLE, keyed by the released label (``"SID <n>"``): EMPTY today. A
#: label is stored iff it is identified (a CID, a released SMILES or a curated
#: compound-identity row) or has an entry here; the entry's name, SMILES and InChIKey
#: build its ``Compound``. Only a PUBLIC label-to-structure release fills it, mirrored and
#: sha256-pinned like any other artifact; an identity inferred from chemical-genetic
#: profile similarity is a hypothesis and never goes here. A filled entry changes the
#: stored records, so it rebuilds the store. The ledger's
#: ``unidentified_compounds_held_out`` is the request list for that release.
RELEASED_IDENTITIES: dict[str, ReleasedIdentity] = {}

#: Systematic ORF pattern; any other ``orf`` must be listed in NON_ORF_STRAIN_LABELS.
_SYSTEMATIC_RE = re.compile(r"^Y[A-P][LR]\d{3}[WC](-[A-Z])?$")


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """``DATA_ROOT`` from the environment (the mirror + build tree live under it)."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/wildenhainPredictionSynergismChemicalGenetic2015``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def _sha256(path: str | Path) -> str:
    """Streaming sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_relpaths() -> dict[str, str]:
    """Short name -> mirror-relative path for every file the loader reads."""
    return {DATA_FILENAME: DATA_REL, AID_FILENAME: AID_REL}


def deposit_raw_mirror(
    *,
    csv_path: str | Path,
    aid_path: str | Path,
    retrieved_at: str = RETRIEVED_AT,
    data_root: str | None = None,
) -> Path:
    """Write the raw mirror from already-retrieved files and its ``manifest.json``.

    Idempotent by sha256 (an existing file with the recorded hash is left alone, a
    differing one raises). The datapoint export's retrieval pins the FTP container FIRST
    and then reads one member out of it, so a re-packed upstream archive is detected
    rather than silently followed.
    """
    root = raw_mirror_dir(data_root)
    sources: dict[str, tuple[Path, str, RetrievalRecord]] = {
        DATA_REL: (
            Path(csv_path),
            DATA_SHA256,
            RetrievalRecord(
                method=RetrievalMethod.direct_url,
                source_url=FTP_ZIP_URL,
                retriever="torchcell.literature.retrieve.zip_member",
                params={
                    "url": FTP_ZIP_URL,
                    "member": ZIP_MEMBER,
                    "container_sha256": CONTAINER_SHA256,
                },
                sha256=DATA_SHA256,
                retrieved_at=retrieved_at,
            ),
        ),
        AID_REL: (
            Path(aid_path),
            AID_SHA256,
            RetrievalRecord(
                method=RetrievalMethod.pubchem_api,
                source_url=AID_URL,
                retriever="torchcell.literature.retrieve.direct_url",
                params={"url": AID_URL},
                sha256=AID_SHA256,
                retrieved_at=retrieved_at,
            ),
        ),
    }
    files: list[ArtifactRecord] = []
    # Both sources are verified before anything is written, so a refusal leaves no
    # mirror directory and no partial deposit behind.
    for src, expected, _ in sources.values():
        verify_sha256(src, expected)
    for relpath, (src, expected, retrieval) in sources.items():
        dest = root / relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if _sha256(dest) != expected:
                raise RuntimeError(f"{dest} exists with a different sha256; refusing")
        else:
            shutil.copy2(src, dest)
        files.append(
            ArtifactRecord(
                path=relpath,
                role=ROLE_RAW_DATA,
                bytes=dest.stat().st_size,
                sha256=expected,
                source=retrieval.source_url,
                retrieval=retrieval,
            )
        )
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=PAPER_DOI,
        title="Prediction of Synergism from Chemical-Genetic Interactions by Machine Learning",
        files=files,
        si_data_sources=[
            "https://pubchem.ncbi.nlm.nih.gov/bioassay/1159580",
            FTP_ZIP_URL,
            AID_URL,
        ],
        si_expected=[
            "Cell Systems 2015 Tables S1/S2 (the four compound libraries) and Table S3 "
            "(the 195 original sentinel strains), and Sci Data 2016 Table 1 (all 242 "
            "sentinels of the extended CGM this AID releases; 'the number of sentinels "
            "has been increased from 195 to 242') -- not mirrored (cell.com and "
            "nature.com supplements are not scriptable), so the per-strain allele, "
            "collection and accession, which would resolve the 33 essential-gene "
            "strains and the TSCII / YGL11 / wtn01 labels, are typed gaps"
        ],
        provenance_complete=True,
        created_at=datetime.now(UTC).isoformat(),
    )
    (root / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    return root


def load_manifest(data_root: str | None = None) -> Manifest:
    """Read the raw mirror's ``manifest.json``; refuse, naming the deposit step, if absent."""
    path = raw_mirror_dir(data_root) / "manifest.json"
    if not path.exists():
        raise RuntimeError(
            f"raw-mirror manifest missing: {path}. Deposit the mirror with "
            "deposit_raw_mirror() first."
        )
    return Manifest.model_validate_json(path.read_text())


def manifest_sha256(manifest: Manifest, relpath: str) -> str:
    """The recorded sha256 of one mirror file."""
    for record in manifest.files:
        if record.path == relpath:
            return record.sha256
    raise KeyError(f"{relpath} is not in the raw-mirror manifest")


# --------------------------------------------------------------------------- #
# Retention bookkeeping + the collapsed matrix cell
# --------------------------------------------------------------------------- #
class DropRule(BaseModel):
    """One retention rule, the records it removed, and the items it removed them for."""

    rule: str
    scope: Literal["compound", "library_row", "cell", "strain"]
    description: str
    n_records: int
    items: list[str] = []


class UnidentifiedCompound(BaseModel):
    """One released label held out under ``UNIDENTIFIED_COMPOUND_RULE``."""

    identity: str
    pubchem_sid: str
    n_rows: int
    n_cells: int
    n_screens: int


class DropLog(BaseModel):
    """Every retention rule applied to a build, in the order they were applied.

    ``source_records`` counts released datapoint rows (a row with a z_score);
    ``kept_records`` the records written, one per distinct released screen.
    ``source_records = kept_records + `` every rule's ``n_records`` (counted in
    released rows), checked at build time. ``compounds_without_a_structure_identifier``
    lists STORED compound identities that carry a typed ``inchikey`` gap and no CID or
    ChEBI id (an admitted ``RELEASED_IDENTITIES`` label whose key is not released);
    ``unidentified_compounds_held_out`` lists the labels held out under
    ``UNIDENTIFIED_COMPOUND_RULE``.
    """

    dataset: str
    source_records: int
    kept_records: int
    dropped_records: int
    rules: list[DropRule]
    compounds_without_a_structure_identifier: list[str] = []
    unidentified_compound_rule: str
    unidentified_compound_policy: str
    unidentified_compound_sources: list[SourcedValue]
    unidentified_compounds_held_out: list[UnidentifiedCompound]


class ScreenDatapoint(BaseModel):
    """One distinct released screen of a (strain, compound) cell: the row as released."""

    #: ``PUBCHEM_RESULT_TAG`` of the first released row carrying this datapoint; the
    #: record's ``replicate_id``.
    tag: str
    z: float
    #: the row's ``sym`` cell, verbatim (the record's ``perturbed_gene_name``)
    sym: str
    non_replicate: str
    outcome: str
    bioactivity: str
    #: tags of the release's own byte-identical re-emissions of this datapoint
    re_exported_tags: list[str] = []

    def category(self) -> tuple[ResponseCategory, str]:
        """The row's released call on the shared axis + its verbatim source words."""
        label = (
            self.outcome
            if not self.bioactivity
            else f"{self.outcome} / {self.bioactivity}"
        )
        return OUTCOME_CATEGORY[(self.outcome, self.bioactivity)], label


class MatrixCell(BaseModel):
    """One (strain, compound) cell of the CGM: its distinct released screens."""

    #: the strain key: a systematic ORF (``NA/NNK1`` rows land on YKL171W),
    #: ``WILD_TYPE_KEY`` for the BY4741 screen, or ``"<orf>/<sym>"`` for an unresolved
    #: strain label (held in the ledger, never served).
    orf: str
    identity: str
    pubchem_cid: int | None
    smiles: str | None
    #: parsed z_score -> the released screen, one entry per DISTINCT datapoint (the z
    #: value is the datapoint key, so ``-4.0`` and ``-4.00`` are one)
    screens: dict[float, ScreenDatapoint] = {}

    @property
    def n_screens(self) -> int:
        """Number of distinct released screens (one record each)."""
        return len(self.screens)

    @property
    def n_re_exported_rows(self) -> int:
        """Released rows that re-emit a datapoint this cell already carries."""
        return sum(len(screen.re_exported_tags) for screen in self.screens.values())

    @property
    def all_screens_non_replicating(self) -> bool:
        """Whether EVERY contributing screen's two OD reads failed to replicate."""
        return all(screen.non_replicate == "1" for screen in self.screens.values())


def _parse_z(z_raw: str, orf: str, identity: str) -> float:
    """The released z_score as a finite float; refuse, naming the cell, otherwise.

    A non-finite z would break the datapoint key (``nan != nan``, so two ``nan`` rows
    become two screens) and the mean / SD. Measured on the pinned export: 0 of 484,830
    strain datapoints are unparseable or non-finite.
    """
    try:
        z = float(z_raw)
    except ValueError:
        raise RuntimeError(
            f"{orf}/{identity}: z_score {z_raw!r} is not a number"
        ) from None
    if not math.isfinite(z):
        raise RuntimeError(f"{orf}/{identity}: z_score {z_raw!r} is not finite")
    return z


def _compound_is_identified(cell: MatrixCell) -> bool:
    """Whether a cell's released compound is identified (``UNIDENTIFIED_COMPOUND_RULE``).

    Identified: the row carries a ``PUBCHEM_CID`` or a released SMILES, the curated
    compound-identity table carries a row for its label, or ``RELEASED_IDENTITIES`` has
    an entry for it. The release has no name column, so a label with none of these is a
    PubChem substance registry number (``SID n``) and nothing else.
    """
    if cell.pubchem_cid is not None or cell.smiles is not None:
        return True
    if cell.identity in RELEASED_IDENTITIES:
        return True
    return resolve_compound_identity(name=cell.identity).curated


@register_dataset
class EnvChemgenWildenhain2015Dataset(ExperimentDataset):
    """Wildenhain 2015 chemical-genetic matrix: env x geno -> growth-inhibition z-score."""

    organism = Organism.scerevisiae

    def __init__(
        self,
        root: str = "data/torchcell/env_chemgen_wildenhain2015",
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset (the genome is loaded lazily inside ``process``)."""
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
        """The datapoint export and the AID description the protocol is sourced from."""
        return [DATA_FILENAME, AID_FILENAME]

    def download(self) -> None:
        """Link the mirror files into ``raw/`` after verifying each against its pin.

        ``DATA_SHA256`` and ``AID_SHA256`` are the pins; the manifest is the retrieval
        record and must carry the same digests, else ``ManifestPinMismatchError`` names
        both. The mirror is canonical. The FTP container is 151 MB and its sha256 is recorded
        with the member's, so ``deposit_raw_mirror``'s retrieval re-runs the download and
        detects a re-packed archive; a build never depends on that URL being alive.
        """
        data_root = _data_root()
        manifest = load_manifest(data_root)
        os.makedirs(self.raw_dir, exist_ok=True)
        pins = {DATA_FILENAME: DATA_SHA256, AID_FILENAME: AID_SHA256}
        for name, relpath in raw_relpaths().items():
            src = raw_mirror_dir(data_root) / relpath
            if not src.exists():
                raise RuntimeError(f"required raw artifact missing from mirror: {src}")
            check_manifest_pin(relpath, manifest_sha256(manifest, relpath), pins[name])
            link_verified(src, osp.join(self.raw_dir, name), pins[name])
        log.info(
            "Wildenhain 2015 raw files linked into %s (sha256 verified)", self.raw_dir
        )

    # ---- source reading -------------------------------------------------------- #
    def _strain_key(self, orf: str, sym: str) -> str:
        """The cell's strain key for one released row (see ``MatrixCell.orf``).

        A systematic ``orf`` is its own key. Any other ``orf`` must be a listed
        ``NON_ORF_STRAIN_LABELS`` pair; an unlisted one refuses, naming it, rather than
        being skipped as a "control row" (they are strain screens, #504 finding 3).
        """
        if _SYSTEMATIC_RE.match(orf):
            return orf
        label = NON_ORF_STRAIN_LABELS.get((orf, sym))
        if label is None:
            raise RuntimeError(
                f"released row with orf={orf!r} sym={sym!r} is neither a systematic "
                "ORF nor a listed NON_ORF_STRAIN_LABELS pair; a new strain label needs "
                "an explicit disposition, not a silent skip"
            )
        if label.disposition == "mapped_to_orf":
            assert label.systematic_gene_name is not None
            return label.systematic_gene_name
        if label.disposition == "wild_type":
            return WILD_TYPE_KEY
        return f"{orf}/{sym}"

    def _collapse_matrix(
        self,
    ) -> tuple[dict[tuple[str, str], MatrixCell], int, dict[tuple[str, str], int]]:
        """Read the datapoint CSV; group the released screens per (strain key, compound).

        Identity is the PubChem CID when present, else the SID. A re-emission of the
        SAME datapoint (equal parsed z) is listed on the first row's screen; distinct
        screens accumulate as their own datapoints. Returns the cells, the
        datapoint-row count, and the row count of every non-ORF label.
        """
        path = osp.join(self.raw_dir, DATA_FILENAME)
        cells: dict[tuple[str, str], MatrixCell] = {}
        n_rows = 0
        label_rows: dict[tuple[str, str], int] = {}
        with gzip.open(path, "rt", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            idx = {name: i for i, name in enumerate(header)}
            next(reader)  # RESULT_TYPE definition row
            for row in reader:
                if not row:
                    continue
                z_raw = row[idx["z_score"]].strip()
                if not z_raw:
                    continue
                orf = row[idx["orf"]].strip()
                sym = row[idx["sym"]].strip()
                strain = self._strain_key(orf, sym)
                if strain != orf:
                    label_rows[(orf, sym)] = label_rows.get((orf, sym), 0) + 1
                n_rows += 1
                cid = row[idx["PUBCHEM_CID"]].strip()
                sid = row[idx["PUBCHEM_SID"]].strip()
                identity = f"CID {cid}" if cid else f"SID {sid}"
                key = (strain, identity)
                cell = cells.get(key)
                if cell is None:
                    cell = MatrixCell(
                        orf=strain,
                        identity=identity,
                        pubchem_cid=int(cid) if cid else None,
                        smiles=row[idx["PUBCHEM_EXT_DATASOURCE_SMILES"]].strip()
                        or None,
                    )
                    cells[key] = cell
                z = _parse_z(z_raw, strain, identity)
                tag = row[idx["PUBCHEM_RESULT_TAG"]].strip()
                if not tag:
                    raise RuntimeError(
                        f"{strain}/{identity}: released row has no PUBCHEM_RESULT_TAG; "
                        "the tag is the record's replicate_id"
                    )
                screen = cell.screens.get(z)
                if screen is None:
                    cell.screens[z] = ScreenDatapoint(
                        tag=tag,
                        z=z,
                        sym=sym,
                        non_replicate=row[idx["non replicate"]].strip(),
                        outcome=row[idx["PUBCHEM_ACTIVITY_OUTCOME"]].strip(),
                        bioactivity=row[idx["bioactivity"]].strip(),
                    )
                else:
                    screen.re_exported_tags.append(tag)
        log.info(
            "Wildenhain2015: %d datapoints (%d on a non-ORF strain label) -> %d "
            "(strain, compound) cells",
            n_rows,
            sum(label_rows.values()),
            len(cells),
        )
        return cells, n_rows, label_rows

    # ---- environment / phenotype builders -------------------------------------- #
    def _compound(self, cell: MatrixCell) -> Compound:
        """The compound's canonical identity: the table's PubChem name + InChIKey.

        A ``RELEASED_IDENTITIES`` entry for the label supplies the name and SMILES the
        release lacks, and its released InChIKey fills a key the resolver could not.
        """
        released = RELEASED_IDENTITIES.get(cell.identity)
        if released is None:
            return resolved_compound(
                cell.identity, pubchem_cid=cell.pubchem_cid, smiles=cell.smiles
            )
        compound = resolved_compound(
            released.name or cell.identity,
            pubchem_cid=cell.pubchem_cid,
            smiles=cell.smiles or released.smiles,
            derive_from_smiles=True,
        )
        if released.inchikey is not None and compound.inchikey is None:
            compound = compound.model_copy(
                update={
                    "inchikey": released.inchikey,
                    "provenance_gaps": [
                        gap
                        for gap in compound.provenance_gaps
                        if gap.field != "inchikey"
                    ],
                }
            )
        return compound

    @staticmethod
    def _culture_environment(
        perturbations: list[EnvironmentPerturbationType],
    ) -> CultureEnvironment:
        """Static 100 uL SC (2% glucose) 96-well culture at 30 C, read at saturation."""
        return CultureEnvironment(
            media=WILDENHAIN_SC,
            temperature=Temperature(value=TEMPERATURE_C.value),
            perturbations=perturbations,
            aerobicity="aerobic",
            duration_hours=DURATION_HOURS.value,
            culture_format=CultureFormat(
                vessel=CULTURE_VESSEL.value,
                working_volume_ul=WORKING_VOLUME_UL.value,
                shaking_rpm=STATIC_INCUBATION_RPM.value,
                inoculum_cells=INOCULUM_CELLS.value,
                endpoint=ENDPOINT.value,
                provenance=[
                    CULTURE_VESSEL,
                    WORKING_VOLUME_UL,
                    STATIC_INCUBATION_RPM,
                    INOCULUM_CELLS,
                    ENDPOINT,
                ],
            ),
            pre_culture=PreCulture(
                source=PRE_CULTURE_SOURCE.value,
                provenance=[PRE_CULTURE_SOURCE],
                provenance_gaps=[
                    ProvenanceGap(
                        field="medium",
                        reason=ProvenanceGapReason.not_reported_by_primary,
                        note="'fresh overnight cultures' (Sci Data line 54); the "
                        "overnight medium is not stated",
                    )
                ],
            ),
            provenance_gaps=[
                ProvenanceGap(
                    field="duration_generations",
                    reason=ProvenanceGapReason.not_reported_by_primary,
                    note="an ~18 h liquid OD growth to control saturation doses exposure "
                    "in hours, not doublings, and neither paper nor the AID protocol "
                    "reports a doubling count",
                ),
                ProvenanceGap(
                    field="auxotroph_supplements",
                    reason=ProvenanceGapReason.not_reported_by_primary,
                    note="no supplement is named beside the medium: SC is complete, so "
                    "the His/Leu/Met/Ura the BY4741 background needs are part of the "
                    "medium's (shared SC) composition, which the lab does not state",
                ),
            ],
        )

    def _environment(self, compound: Compound) -> CultureEnvironment:
        """The culture environment carrying the compound at 20 uM in 1.96% v/v DMSO."""
        return self._culture_environment(
            [
                SmallMoleculePerturbation(
                    compound=compound,
                    concentration=Concentration(
                        value=SCREEN_CONCENTRATION_UM.value,
                        unit=ConcentrationUnit.micromolar,
                    ),
                    solvent=Solvent(
                        name=SOLVENT.value,
                        percent=SOLVENT_PERCENT.value,
                        compound=resolved_compound("dimethyl sulfoxide"),
                    ),
                )
            ]
        )

    def _reference(self) -> StrainEnvironmentResponseExperimentReference:
        """The one reference: z = 0, the strain's own screen center, no compound.

        The z is a N(1, IQR) fit WITHIN one strain's screen (``Z_SCORE_DEFINITION``),
        so its 0 is that strain's plate-median growth with (mostly) inactive
        compounds, not BY4741 under the compound (#504 finding 5). The environment is
        therefore the culture with NO compound; the genome is the BY4741 background
        every strain shares. "Same strain" cannot be a genotype here (the reference
        record has no genotype slot), so ``REFERENCE_UNITS`` states it.
        """
        return StrainEnvironmentResponseExperimentReference(
            dataset_name=self.name,
            genome_reference=BY4741_GENOME,
            environment_reference=self._culture_environment([]),
            phenotype_reference=EnvironmentResponsePhenotype(
                measurement_type=Z_SCORE_DEFINITION.value,
                assay_type=ASSAY.value,
                environment_response=0.0,
                units=REFERENCE_UNITS,
            ),
        )

    def _phenotype(self, screen: ScreenDatapoint) -> EnvironmentResponsePhenotype:
        """One released screen's z-score, as released, with its typed absences.

        ``replicate_id`` is the row's ``PUBCHEM_RESULT_TAG``. The release carries no
        per-screen error (typed gaps on both uncertainty fields) and no library / screen
        column (typed gap on ``screen_id``; the normalization differs by library,
        ``NORMALIZATION_BY_LIBRARY``).
        """
        category, label = screen.category()
        return EnvironmentResponsePhenotype(
            measurement_type=Z_SCORE_DEFINITION.value,
            assay_type=ASSAY.value,
            environment_response=screen.z,
            category=category,
            category_label=label,
            n_samples=1,
            sample_unit=SampleUnit.screen,
            units=MEASUREMENT_UNITS,
            replicate_id=screen.tag,
            provenance_gaps=[
                *(
                    ProvenanceGap(
                        field=field,
                        reason=ProvenanceGapReason.not_reported_by_primary,
                        note="one released screen per record; the release carries no "
                        "per-screen error",
                    )
                    for field in (
                        "environment_response_uncertainty",
                        "environment_response_se",
                    )
                ),
                ProvenanceGap(
                    field="screen_id",
                    reason=ProvenanceGapReason.not_reported_by_primary,
                    note="the release has no library, screen or plate column; screens "
                    "were LOWESS- or DMSO-control-normalized by library (Sci Data lines "
                    "63-64), so a cell's normalization is not recoverable",
                ),
            ],
        )

    def _genotype(self, strain: str, released_name: str) -> Genotype:
        """The screened strain's genotype on the BY4741 background.

        The wild-type screen is the empty genotype; every other strain is the kanMX4
        Euroscarf haploid deletion the release calls it, keyed by the released ORF with
        the row's released ``sym`` as its gene name and no released barcode.
        """
        if strain == WILD_TYPE_KEY:
            return Genotype(perturbations=[])
        return Genotype(
            perturbations=[
                BarcodedKanMxDeletionPerturbation(
                    systematic_gene_name=strain,
                    perturbed_gene_name=released_name,
                    collection=COLLECTION.value,
                    cassette=KANMX4_CASSETTE.value,
                    provenance_gaps=[
                        pending_source_review(
                            "barcode",
                            GIAEVER_2002,
                            "the release publishes no UPTAG/DNTAG; the YKO barcode "
                            "table gives them by ORF",
                        )
                    ],
                )
            ]
        )

    # ---- build ------------------------------------------------------------------ #
    @post_process
    def process(self) -> None:
        """Collapse the datapoint export into the CGM matrix; write LMDB."""
        verify_raw_files(
            self.raw_dir, {DATA_FILENAME: DATA_SHA256, AID_FILENAME: AID_SHA256}
        )
        cells, n_rows, label_rows = self._collapse_matrix()

        unresolved_pairs = {
            f"{orf}/{sym}": (orf, sym)
            for (orf, sym), label in NON_ORF_STRAIN_LABELS.items()
            if label.disposition == "unresolved"
        }
        unresolved = set(unresolved_pairs)
        # Rule 1: unresolved strain labels are held, whatever their compound, until the
        # released-identifier leaf exists. Counted in released rows.
        held = {key for key, cell in cells.items() if cell.orf in unresolved}
        held_by_label = {
            label: sum(1 for key in held if cells[key].orf == label)
            for label in sorted(unresolved)
        }
        n_held_rows = sum(
            label_rows[unresolved_pairs[label]]
            for label, n in held_by_label.items()
            if n
        )

        # Rule 2: a compound with no released name or structure is held out
        # (UNIDENTIFIED_COMPOUND_RULE). Counted in released rows of the cells left.
        unidentified_keys = {
            key
            for key, cell in cells.items()
            if key not in held and not _compound_is_identified(cell)
        }
        held_out: dict[str, UnidentifiedCompound] = {}
        for key in sorted(unidentified_keys):
            cell = cells[key]
            entry = held_out.setdefault(
                cell.identity,
                UnidentifiedCompound(
                    identity=cell.identity,
                    pubchem_sid=cell.identity.removeprefix("SID "),
                    n_rows=0,
                    n_cells=0,
                    n_screens=0,
                ),
            )
            entry.n_rows += cell.n_screens + cell.n_re_exported_rows
            entry.n_cells += 1
            entry.n_screens += cell.n_screens
        n_unidentified_rows = sum(entry.n_rows for entry in held_out.values())
        held |= unidentified_keys

        # Rule 3: the release's own re-emissions of a datapoint collapse onto its first
        # row (true duplicates; the other tags stay on the datapoint).
        n_re_exported = sum(
            cell.n_re_exported_rows for key, cell in cells.items() if key not in held
        )

        # Every DISTINCT stored compound identity is resolved once; one with no
        # InChIKey, CID or ChEBI id (only an admitted RELEASED_IDENTITIES label can be
        # one) is STORED with the typed inchikey gap resolved_compound attaches.
        compounds: dict[str, Compound] = {}
        for key, cell in cells.items():
            if key not in held and cell.identity not in compounds:
                compounds[cell.identity] = self._compound(cell)
        unidentified = sorted(
            identity
            for identity, compound in compounds.items()
            if compound.inchikey is None
            and compound.pubchem_cid is None
            and compound.chebi_id is None
        )
        names = [compound.name for compound in compounds.values()]
        if len(set(names)) != len(names):
            raise RuntimeError(
                "two distinct compound identities resolve to the same canonical name, "
                "which would merge two conditions into one record key"
            )

        # The served release is the 2016 Sci Data extended CGM, but a record holds ONE
        # Publication; the 2015 paper stays the record's citation and the 2016 paper is
        # cited by every record's reference background, culture format and z rule
        # (SourcedValues carrying its citation key + sha256).
        publication = Publication(doi=PAPER_DOI, doi_url=f"https://doi.org/{PAPER_DOI}")
        reference = self._reference()
        environments: dict[str, CultureEnvironment] = {}

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        idx = 0
        n_all_non_replicating = 0
        n_cells = 0
        by_kind = {"deletion": 0, "wild_type": 0}
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for key in tqdm(sorted(cells), desc="Wildenhain2015 CGM"):
                cell = cells[key]
                if key in held:
                    continue
                n_cells += 1
                if cell.identity not in environments:
                    environments[cell.identity] = self._environment(
                        compounds[cell.identity]
                    )
                if cell.all_screens_non_replicating:
                    n_all_non_replicating += 1
                kind = "wild_type" if cell.orf == WILD_TYPE_KEY else "deletion"
                for z in sorted(cell.screens):
                    screen = cell.screens[z]
                    by_kind[kind] += 1
                    experiment = StrainEnvironmentResponseExperiment(
                        dataset_name=self.name,
                        genotype=self._genotype(cell.orf, screen.sym),
                        environment=environments[cell.identity],
                        phenotype=self._phenotype(screen),
                    )
                    txn.put(
                        f"{idx}".encode(),
                        self._intern_record(experiment, reference, publication, itxn),
                    )
                    idx += 1
        env.close()
        interned_env.close()

        drop_log = DropLog(
            dataset=self.name,
            source_records=n_rows,
            kept_records=idx,
            dropped_records=n_rows - idx,
            rules=[
                DropRule(
                    rule="strain_label_unresolved",
                    scope="strain",
                    description=(
                        "the released orf is not a systematic name and the sym names "
                        "no strain a mirrored source identifies (NON_ORF_STRAIN_LABELS); "
                        "held for the released-identifier leaf. Counted in released "
                        "rows. Items: label (cells, released rows)"
                    ),
                    n_records=n_held_rows,
                    items=[
                        f"{label} ({n} cells, "
                        f"{label_rows[unresolved_pairs[label]]} rows)"
                        for label, n in held_by_label.items()
                        if n
                    ],
                ),
                DropRule(
                    rule=UNIDENTIFIED_COMPOUND_RULE,
                    scope="compound",
                    description=(
                        "the released compound has no PUBCHEM_CID, no SMILES (the "
                        "release has no name column), no curated identity row and no "
                        "RELEASED_IDENTITIES entry: a registry number only, so the "
                        "experiment cannot be represented. Counted in released rows. "
                        "Items: label (cells, released rows)"
                    ),
                    n_records=n_unidentified_rows,
                    items=[
                        f"{entry.identity} ({entry.n_cells} cells, {entry.n_rows} rows)"
                        for entry in held_out.values()
                    ],
                ),
                DropRule(
                    rule="re_exported_duplicate_row",
                    scope="library_row",
                    description=(
                        "the release re-emits a datapoint it already carries for the "
                        "cell (same parsed z_score, every other data column equal, "
                        "typically under a second gene-symbol spelling); the row is a "
                        "true duplicate and its PUBCHEM_RESULT_TAG is listed on the "
                        "datapoint it repeats"
                    ),
                    n_records=n_re_exported,
                    items=[],
                ),
            ],
            compounds_without_a_structure_identifier=unidentified,
            unidentified_compound_rule=UNIDENTIFIED_COMPOUND_RULE,
            unidentified_compound_policy=UNIDENTIFIED_COMPOUND_POLICY,
            unidentified_compound_sources=[
                RELEASED_COMPOUND_IDENTIFIER,
                STRUCTURES_ON_CHEMGRID,
            ],
            unidentified_compounds_held_out=list(held_out.values()),
        )
        with open(osp.join(self.preprocess_dir, "dropped_records.json"), "w") as handle:
            handle.write(drop_log.model_dump_json(indent=2))
        rule_total = n_held_rows + n_unidentified_rows + n_re_exported
        if drop_log.dropped_records != rule_total:
            raise RuntimeError(
                f"drop accounting mismatch: rule total {rule_total}, "
                f"{drop_log.dropped_records} records missing from the build"
            )
        log.info(
            "Wrote %d Wildenhain2015 records over %d cells (%d kanMX deletion, %d wild "
            "type; %d rows held for an unresolved strain label; %d rows of %d "
            "unidentified compounds held out; %d re-exported duplicate rows collapsed; "
            "%d compounds stored with an inchikey gap; %d cells whose every screen is "
            "non-replicate flagged)",
            idx,
            n_cells,
            by_kind["deletion"],
            by_kind["wild_type"],
            n_held_rows,
            n_unidentified_rows,
            len(held_out),
            n_re_exported,
            len(unidentified),
            n_all_non_replicating,
        )

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError


def main() -> None:
    """Build/load the dataset for interactive debugging."""
    from dotenv import load_dotenv

    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    root = osp.join(data_root, "data/torchcell/env_chemgen_wildenhain2015")
    dataset = EnvChemgenWildenhain2015Dataset(root=root)
    print(f"len = {len(dataset)}")
    print(dataset[0])


if __name__ == "__main__":
    main()
