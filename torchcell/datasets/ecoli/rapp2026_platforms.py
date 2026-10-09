# torchcell/datasets/ecoli/rapp2026_platforms
# [[torchcell.datasets.ecoli.rapp2026_platforms]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/rapp2026_platforms
# Test file: tests/torchcell/datasets/ecoli/test_rapp2026_platforms.py
"""Three further released quantity families of Rapp 2026's E. coli CRISPRi library.

``ecoli/rapp2026.py`` stores the FI-MS fold-change matrix of Table S4. The release
carries three more per-strain families, each on its own platform or scale, and each is
its own dataset here so that no two platforms ever share one record set:

1. ``GrowthAucRapp2026Dataset`` -- the paper's own growth statistic, the trapezoid area
   under each strain's released OD600 curve (Table S2), stored as
   ``BacterialFitnessExperiment`` / ``FitnessPhenotype`` with the 16 control strains as
   the denominator. **1,515 records**, one per Table S2 library gene, ``phnE`` among them
   on a ``locus_tag_synonym`` mapping. 18 of those genes have no metabolome sample at
   all, so this family EXTENDS the strain coverage of the key.
2. ``TargetedMetabolomeRapp2026Dataset`` -- the targeted LC-MS/MS screen of Table S6,
   the EIC peak-height fold change against the library median, a SECOND PLATFORM.
   **407 records, 1,246 values.** MEASURED not to duplicate the stored FI-MS values:
   joining Table S6 1:1 onto Table S5 on (gene, abbreviation, mode) gives Pearson
   r = 0.6722 on the linear fold change (0.6701 on log2) with a median absolute log2
   difference of 1.1649, reproduced at build time into
   ``preprocess/platform_agreement.json``.
3. ``MetaboliteIntensityRapp2026Dataset`` -- Table S5's ABSOLUTE FI-MS intensity of each
   accumulating annotated feature (``Mean_Int`` / ``R1_Int`` / ``R2_Int``), a different
   scale from the stored fold change. **407 records, 1,375 values**, each with a real
   per-replicate standard error.

DATA. Every file is in the citation key's raw mirror under one manifest, pinned by
``rapp2026.RAW_FILES`` (seven workbooks; Table S2 ``si3.xlsx`` and Table S6 ``si7.xlsx``
are pinned there and read here). Growth reads Table S1 + Table S2; targeted LC-MS/MS
reads Table S1 + Table S3 + Table S6 + Table S9; intensities read Table S1 + Table S3 +
Table S5 + Table S9.

STRAIN. Identical to the metabolome loader: records pin the MG1655 GenBank assembly the
released b-numbers mean, with ``YYdCas9`` as the ``BacterialStrainBackground``, and one
``BacterialCrisprInterferencePerturbation`` per record carrying the Table S1 spacer.
Every family carries the same single retention rule as the metabolome loader
(``b_number_remapped_by_the_annotation``), and in every family it now removes NOTHING.
``phnE``'s ``b4104`` is not a locus tag of the pinned annotation and is a
``/gene_synonym`` of exactly ONE locus, the pseudogene ``b4583`` (``phnE1``), so the
record is KEPT with ``identifier_mapping=DerivedIdentifierMapping(
source_identifier="b4104", route="locus_tag_synonym")`` on its perturbation
(``rapp2026.locus_tag_synonym_mapping`` re-checks every condition of that route). The
rule still fires for a released b-number this annotation relates to another locus by a
route no ``DerivedIdentifierRoute`` member names. ``argR``, the metabolome loader's other
drop, is absent from Table S1 and Table S2 and from both accumulation tables, so it never
arises here.

REFERENCES, which is where the three families differ.

* Growth: the control strain's own curves. ``fitness`` is the strain's mean AUC over the
  grand mean AUC of the 48 released control curves (16 ``ctrlN`` wells x 3 replicate
  cultures), so the reference is ``fitness = 1.0`` with ``n_samples = 48``.
* Targeted LC-MS/MS: the released statistic's own denominator. A fold change against
  "the median peak height of the whole library" (Figure 2B) is 1.0 for a strain at that
  median, so the per-record reference is 1.0 on every key the record measures.
  ``n_replicates`` there is the MEASURED number of strains the released table carries
  for that m/z feature, a LOWER BOUND on the population the median was taken over: the
  release names it only as "all other strains" and never enumerates it. This is the one
  value in this module that is a documented representative rather than a sourced number,
  and it is flagged in the PR and the dendron note.
* Intensities: the per-batch median intensity, BACK-SOLVED exactly from two released
  columns. ``Mean_Int / Mean_FC`` is the intensity the fold change is a ratio to, and it
  is MEASURED constant across every strain of a batch (``R1_Int / R1_FC`` has a maximum
  relative spread of 3.26e-16 over the 246 multi-strain (feature, batch) groups, and
  equals ``R2_Int / R2_FC`` on all 1,385 rows). So the reference level is the batch
  median itself, on the same absolute scale as the stored value, with ``n_replicates``
  the number of samples in that strain's batch.

NOT LOADED HERE, deliberately. Table S6's ``Intensity PrecMz`` is an instrument-scale
precursor intensity with no normalization, and MEASURED not even reducible to the stored
fold change: ``Intensity PrecMz / fold-change`` varies within a feature key by a median
relative spread of 0.185 (maximum 3.34), so it is not that statistic's numerator and
carries no scale a second record could be compared on. Table S7 is gated on a release
disagreement the audit records: joining Table S5 to Table S7's annotated rows on (gene,
polarity, mass) gives 689 pairs, 0 of which share a ``Mean_FC``, so Table S7 is a THIRD
set of numbers for strain-feature pairs already stored, and Table S7's own ``Mean_Int``
column holds the fold change rather than an intensity on all 9,462 rows.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import os.path as osp
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Final

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    check_manifest_pin,
    link_verified,
    post_process,
    verify_raw_files,
)
from torchcell.datamodels.schema import (
    BacterialCrisprInterferencePerturbation,
    BacterialFitnessExperiment,
    BacterialFitnessExperimentReference,
    BacterialMetaboliteExperiment,
    BacterialMetaboliteExperimentReference,
    CrisprConstruct,
    DerivedIdentifierMapping,
    Experiment,
    ExperimentReference,
    FitnessPhenotype,
    Genotype,
    MetabolitePhenotype,
    SampleUnit,
    UncertaintyType,
)
from torchcell.datasets.bacteria_common import (
    LOCUS_TAG_PATTERNS,
    STRAIN_GENE_NAMESPACES,
    assembly_reference,
    bacterial_genome,
    reconcile_locus_tags,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.datasets.ecoli.rapp2026 import (
    CAS_EFFECTOR,
    CITATION_KEY,
    CONTROL_TOKEN_PATTERN,
    DATA_SHA256,
    GUIDES_PER_GENE,
    LEGEND_SHEET,
    LIBRARY_GENES,
    MG1655_ASSEMBLY_SET,
    MIN_RESOLVED_FRACTION,
    PAPER_MD,
    PAPER_MD_SHA256,
    PROTON_MASS,
    PROTON_MASS_TOLERANCE,
    PUBLICATION,
    RAW_FILES,
    REFERENCE_STRAIN_NAME,
    TABLE_S1,
    TABLE_S2,
    TABLE_S2_SHEET,
    TABLE_S3,
    TABLE_S5,
    TABLE_S5_SHEET,
    TABLE_S6,
    TABLE_S6_SHEET,
    TABLE_S9,
    DropLog,
    DropRule,
    Guide,
    IdentifierLedger,
    Metabolite,
    SampleRow,
    SymbolDisagreement,
    canonical_symbol,
    environment,
    host_background,
    load_manifest,
    locus_tag_synonym_mapping,
    manifest_sha256,
    metabolite_identity_gaps,
    raw_mirror_dir,
    read_guides,
    read_metabolites,
    read_sample_rows,
)
from torchcell.sequence.genome.ecoli.k12 import EcoliK12Genome, EcoliK12StrainName
from torchcell.verification.report import (
    Level,
    LevelResult,
    Provenance,
    VerificationReport,
)
from torchcell.verification.sourced import SourcedValue, audit_sourced_value

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

FloatMatrix = npt.NDArray[np.float64]

#: Figure captions S1-S12, the artifact the growth statistic's definition lives in.
SI1_MD = "si/si1.md"
SI1_MD_SHA256 = "809c383ac09cab361a09206522bb1d0ca928853f5984985bc41be3f2cff58b01"


def _paper(value: Any, quote: str, *, note: str | None = None) -> SourcedValue:
    """A value bound to a verbatim quote in the sha256-pinned Rapp ``paper.md``."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=PAPER_MD,
            citation_key=CITATION_KEY,
            sha256=PAPER_MD_SHA256,
            method="MinerU OCR of the publisher PDF (torchcell-library mirror)",
        ),
    )


def _si1(value: Any, quote: str, *, note: str | None = None) -> SourcedValue:
    """A value bound to a verbatim quote in the pinned supplemental-figure OCR."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=SI1_MD,
            citation_key=CITATION_KEY,
            sha256=SI1_MD_SHA256,
            method="MinerU OCR of the supplemental information PDF (mmc1.pdf)",
            page="Figure S1 caption",
        ),
    )


# --------------------------------------------------------------------------- #
# Sourced values (verbatim quotes of the pinned OCR, LaTeX markup included)
# --------------------------------------------------------------------------- #
_Q_GROWTH_ASSAY = (
    "M9 pre-cultures were diluted 150-fold in M9 in 96-well flat-bottom plates, which "
    "were then closed using breathe-easy foil. ${ \\tt O D } _ { 6 0 0 }$ was measured "
    "every $1 0 \\mathrm { \\ m i n }$ for $^ { 6 \\mathrm { ~ h ~ } }$ at $3 7 ^ { "
    "\\circ } \\mathrm { C }$ , with shaking at 800 rpm, in a plate reader "
    "(LogPhase600, BioTek). Subsequently, cultures were diluted 1:50 in M9 with the "
    "inducer aTc. Plates were closed using breathe-easy foil and ${ \\tt O D } _ { 6 0 "
    "0 }$ was measured every 10 min for $^ { 2 4 \\mathrm { ~ h ~ } }$ at ${ } ^ { 3 7 "
    "} \\ ^ { \\circ } \\mathrm { C }$ and 800 rpm. Data was analyzed with custom "
    "MATLAB scripts to obtain the area under the curve using the trapz.m function."
)
_Q_GROWTH_CURVES = (
    "Figure S1. Growth of CRISPRi strains. Growth curves of all strains in the library "
    "were determined in plate readers. dCas9 expression was induced at $\\mathfrak { t "
    "} = 0$ h by addition of aTc. Growth curves show means from $\\mathsf { n } { = } "
    "3$ cultures cultivated in minimal glucose medium in a plate reader. The cultures "
    "were back diluted into fresh medium at $\\mathfrak { t } = 6 \\mathrm { ~ h ~ }$ "
    ". The area under the curve (AUC) was used to classify CRISPRi strains with and "
    "without growth defect (AUC cutoff of 18)."
)
_Q_GROWTH_SPLIT = (
    "Of the 1,515 CRISPRi strains, 489 CRISPRi strains showed growth defects in "
    "minimal glucose medium, whereas 1,026 CRISPRi strains showed no tangible growth "
    "defect (Figure S1; Table S2)."
)
_Q_TARGETED_PAIRS = (
    "We selected 1,256 strain-metabolite pairs with the strongest accumulation for "
    "targeted LC-MS/MS analysis at three collision energies (10, 20, and 40 eV) "
    "(Figure 2B)."
)
_Q_TARGETED_STATISTIC = (
    "(B) A maximum of ten accumulating metabolites per strain from the FI-MS screen "
    "were used for targeted LC-MS/MS analysis to confirm accumulation. Fold changes of "
    "the peak height in the extracted ion chromatogram (EIC), compared with the median "
    "peak height of the whole library, were used to quantify the accumulation of the "
    "respective metabolite in the CRISPRi strain."
)
_Q_TARGETED_SELECTION = (
    "A maximum of 10 metabolites were measured per strain (those with highest mean "
    "fold change)."
)
_Q_TARGETED_QC = (
    "Therefore, we selected LC-MS/MS data with a precursor intensity $> 5 , 0 0 0$ , a "
    "${ \\mathsf { l o g } } _ { 2 }$ -fold change $> 1$ , high precursor purity in "
    "MS1 spectra, and at least one fragment in the ${ \\mathsf { M } } { \\mathsf { S "
    "} } ^ { 2 }$ spectra (Table S6). Of the 1,256 measurements, 569 fulfilled these "
    "criteria"
)
_Q_TARGETED_CONFIRMATION = (
    "These data confirmed metabolite increase in $8 5 \\%$ of the tested pairs ( ${ "
    "\\mathsf { l o g } } _ { 2 }$ -fold change $> 1$ of the maximum intensity in "
    "extracted ion chromatograms) (Figure 2C; Table S6)."
)

SOURCED_VALUES: dict[str, SourcedValue] = {
    "growth_assay": _paper(
        "trapezoid area under the OD600 curve (MATLAB trapz.m)",
        _Q_GROWTH_ASSAY,
        note="METHOD DETAILS, 'Growth analysis of arrayed library'. The AUC is the "
        "paper's own growth statistic and the only one it reports per strain; the two "
        "measurement stages (10 min spacing for 6 h, then 10 min spacing for 24 h "
        "after the 1:50 back dilution) are the released 0-30 h axis",
    ),
    "growth_replicates": _si1(
        3,
        _Q_GROWTH_CURVES,
        note="three cultures per strain, which Table S2 releases as 'Replicate Nr.' "
        "1, 2 and 3; n_samples counts those cultures and sample_unit is "
        "biological_replicate",
    ),
    "auc_cutoff": _si1(
        18.0,
        _Q_GROWTH_CURVES,
        note="the paper's growth-defect threshold on the AUC. Not a stored field: the "
        "records carry the AUC RATIO, and the cutoff is reproduced at build time into "
        "preprocess/growth_defect_check.json",
    ),
    "back_dilution_hours": _si1(
        6.0,
        _Q_GROWTH_CURVES,
        note="why the released curve spans 30 h while the Methods name a 24 h "
        "measurement: the axis is the 6 h before the back dilution plus the 24 h "
        "after it",
    ),
    "growth_defect_split": _paper(
        {"growth_defect": 489, "no_growth_defect": 1026},
        _Q_GROWTH_SPLIT,
        note="the paper's own classification of its 1,515 strains at the AUC cutoff. "
        "MEASURED from the released curves: 490 strains below 18 and 1,025 at or "
        "above, both when the AUC is averaged over the three replicates and when it is "
        "taken of the mean curve, so the statistic reproduces to within one strain. "
        "The build pins the measured 490 and records the one-strain difference",
    ),
    "targeted_pairs": _paper(
        1256,
        _Q_TARGETED_PAIRS,
        note="process() requires Table S6 to hold exactly this many rows, and Table "
        "S5's own 'LC-MS/MS' flag marks exactly this many of its 1,385 rows",
    ),
    "targeted_statistic": _paper(
        "EIC peak-height fold change against the median peak height of the library",
        _Q_TARGETED_STATISTIC,
        note="Figure 2B caption. The denominator is a library-wide median, so a strain "
        "at that median has fold change 1.0, which is the per-record reference level. "
        "The Table S6 legend writes the same denominator as 'median peak intensity of "
        "all other strains' and the release never enumerates that population",
    ),
    "targeted_max_per_strain": _paper(
        10,
        _Q_TARGETED_SELECTION,
        note="METHOD DETAILS, 'Targeted LC-MS/MS measurements'; MEASURED on Table S6: "
        "1 to 10 metabolites per strain, never more",
    ),
    "targeted_qc_passed": _paper(
        569,
        _Q_TARGETED_QC,
        note="the QC flag is stored on no record (it is a quality call, not a "
        "measurement); process() requires Table S6's 'QC passed' column to carry "
        "exactly this many ones, and the count is written to the build ledger",
    ),
    "targeted_confirmation_share": _paper(
        0.85,
        _Q_TARGETED_CONFIRMATION,
        note="MEASURED on Table S6: log2 of the released fold change exceeds 1 on "
        "1,062 of the 1,256 rows, a share of 0.8455, which is the paper's 85%",
    ),
}

#: Three replicate cultures per strain (``SOURCED_VALUES['growth_replicates']``).
GROWTH_REPLICATES: Final[int] = SOURCED_VALUES["growth_replicates"].value
#: The paper's growth-defect threshold on the AUC.
AUC_DEFECT_CUTOFF: Final[float] = SOURCED_VALUES["auc_cutoff"].value
#: The paper's own growth-defect split of its 1,515 strains.
PAPER_GROWTH_DEFECT: Final[int] = SOURCED_VALUES["growth_defect_split"].value[
    "growth_defect"
]
PAPER_NO_GROWTH_DEFECT: Final[int] = SOURCED_VALUES["growth_defect_split"].value[
    "no_growth_defect"
]
#: Strains MEASURED below the cutoff on the released curves; the paper reports 489.
MEASURED_GROWTH_DEFECT: Final[int] = 490
#: Table S6 rows (``SOURCED_VALUES['targeted_pairs']``).
TARGETED_PAIRS: Final[int] = SOURCED_VALUES["targeted_pairs"].value
#: Rows of Table S6 whose ``QC passed`` flag is 1.
TARGETED_QC_PASSED: Final[int] = SOURCED_VALUES["targeted_qc_passed"].value
#: Metabolites measured per strain by targeted LC-MS/MS, at most.
TARGETED_MAX_PER_STRAIN: Final[int] = SOURCED_VALUES["targeted_max_per_strain"].value

#: Table S2 rows: 1,515 genes x 3 replicates plus 16 control wells x 3.
TABLE_S2_ROWS: Final[int] = 4593
#: Table S2 control wells (``ctrl1`` .. ``ctrl16``; every one is present, unlike the
#: metabolome tables, where ``ctrl2`` is absent).
GROWTH_CONTROL_TOKENS: Final[int] = 16
#: Table S2's time columns: 0 to 30 h at 10 min spacing.
GROWTH_TIME_POINTS: Final[int] = 181
GROWTH_LAST_HOUR: Final[float] = 30.0
GROWTH_INTERVAL_HOURS: Final[float] = 1.0 / 6.0
#: Tolerance of the time-axis check (the released header rounds to six decimals).
GROWTH_INTERVAL_TOLERANCE: Final[float] = 1e-5
#: Table S5 rows: one per accumulating strain-metabolite pair.
INTENSITY_ROWS: Final[int] = 1385
#: Strain tokens both accumulation tables carry, and the four control tokens among them.
ACCUMULATION_TOKENS: Final[int] = 411
ACCUMULATION_CONTROL_TOKENS: Final[int] = 4
#: Relative tolerance of the ``Mean_Int == mean(R1_Int, R2_Int)`` check (measured 0.0)
#: and of the batch-median constancy check (measured 3.26e-16).
INTENSITY_TOLERANCE: Final[float] = 1e-9

#: What each stored number is. One per family, never shared.
GROWTH_STATISTIC = "auc_trapz_od600_0_to_30h_mean_of_3_cultures_over_control_mean"
TARGETED_MEASUREMENT_TYPE = "lc_msms_eic_peak_height_fold_change_vs_library_median"
INTENSITY_MEASUREMENT_TYPE = (
    "fi_ms_annotated_feature_absolute_intensity_mean_of_2_plates"
)

#: Measured agreement between the two platforms, reproduced at build time.
EXPECTED_PLATFORM_AGREEMENT = {
    "n_pairs": 1256,
    "pearson_r_linear": 0.6722,
    "pearson_r_log2": 0.6701,
    "median_abs_log2_difference": 1.1649,
}
#: The agreement statistics are pinned to four decimals, which is how the audit that
#: commissioned these datasets reported them.
AGREEMENT_TOLERANCE: Final[float] = 5e-5

#: A Table S6 ``DataFile``: ``<metabolite>_<gene>_P<plate><well><injection>_<polarity>``.
DATAFILE_PATTERN = re.compile(
    r"^(?P<metabolite>.+)_(?P<gene>[^_]+)_P(?P<plate>\d+)(?P<well>[A-H]\d+)"
    r"(?P<injection>msAV\d+)_(?P<polarity>neg|pos)\.mzML$"
)
#: Table S5 / Table S6 write a missing KEGG id as ``NaN`` where Table S9 writes ``XXX``:
#: as the literal string inside a ``-``-joined isobaric set, and as an empty cell when
#: the whole group has none (which pandas reads as a float nan).
MISSING_KEGG_RELEASED = "NaN"
MISSING_KEGG_IDENTITY = "XXX"
#: ``Mode`` is the adduct; a protonated adduct is the positive polarity.
POSITIVE_POLARITY = "pos"
NEGATIVE_POLARITY = "neg"


# --------------------------------------------------------------------------- #
# Column legends: the released column descriptions, checked against the bytes
# --------------------------------------------------------------------------- #
class SourcedColumn(BaseModel):
    """One released column bound to its own legend text in the pinned workbook.

    ``SourcedValue`` cannot carry this: ``audit_sourced_value`` reads the artifact as
    text, and these quotes live in a sheet of a zipped workbook. So the BUILD re-reads
    the ``Legend`` sheet out of the sha256-pinned bytes and refuses unless the legend
    still says this verbatim, which is a stronger check than a text audit: the quote is
    re-read from the same file the values come from.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    file: str
    sheet: str
    column: str
    quote: str
    sha256: str
    note: str | None = None


def _legend(
    file: str, column: str, quote: str, *, note: str | None = None
) -> SourcedColumn:
    """One ``Legend``-sheet row of a pinned workbook."""
    return SourcedColumn(
        file=file,
        sheet=LEGEND_SHEET,
        column=column,
        quote=quote,
        sha256=DATA_SHA256[file],
        note=note,
    )


#: Table S5's intensity columns, which the intensity family stores.
INTENSITY_COLUMNS: tuple[SourcedColumn, ...] = (
    _legend(
        TABLE_S5,
        "Mean_Int",
        "Mean intensity of annotated m/z-feature of both replicates.",
        note="the stored level; MEASURED to equal mean(R1_Int, R2_Int) on all 1,385 "
        "rows with a maximum absolute difference of 0.0",
    ),
    _legend(
        TABLE_S5,
        "R1_Int+R2_Int",
        "Intensity of annotated m/z-feature in replicate 1 (R1) and replicate 2 (R2).",
        note="the two released per-plate intensities, so n_replicates is 2 and "
        "metabolite_level_se is |R1_Int - R2_Int| / 2, the standard error of the "
        "stored mean over those two plates",
    ),
    _legend(
        TABLE_S5,
        "R1_FC + R2_FC",
        "Fold changes of the intensity of the annotated m/z feature relative to the "
        "median on a per batch basis in replicate 1 (R1) and replicate 2 (R2).",
        note="the denominator of these ratios is the reference level: R1_Int / R1_FC "
        "is the batch median intensity, MEASURED constant across the strains of a "
        "batch and equal to R2_Int / R2_FC on every row",
    ),
    _legend(
        TABLE_S5,
        "LC-MS/MS",
        "1 - Accumulation is analyzed via targeted LC-MS/MS (a maximum number of 10 "
        "accumulating metabolites per strain is analyzed via targeted LC-MS/MS; "
        "metabolites with highest mean fold-change are used).",
        note="not stored; the flag is 1 on exactly the 1,256 rows Table S6 carries, "
        "which is how the two tables are checked against each other at build time",
    ),
)
#: Table S6's targeted columns, which the LC-MS/MS family stores or gates on.
TARGETED_COLUMNS: tuple[SourcedColumn, ...] = (
    _legend(
        TABLE_S6,
        "fold-change",
        "Intensity of highest peak in EIC of the strain compared to median peak "
        "intensity of all other strains.",
        note="the stored level. One value per strain-metabolite pair, with no released "
        "uncertainty and no replicate column, so n_replicates is 1",
    ),
    _legend(
        TABLE_S6,
        "QC passed",
        "Measurement met the following criteria: Precursor intensity >5000, log2-fold "
        "change >1, high precursor purity in MS1 (mass deviation <0.003, no side peaks "
        "within isolation width of quadrupole) and at least one MS2 fragment. LC-MS/MS "
        "where quality control is passed are uploaded on GNPS and saved in .MGF "
        "datafile (Supplementary datafile 2) as putative reference spectra.",
        note="a quality call, not a measurement, so it is stored on no record; the "
        "build requires the released count (569) and ledgers it",
    ),
    _legend(
        TABLE_S6,
        "Intensity PrecMz",
        "Intensity of precursor ion at peak maxiumum within the EIC.",
        note="NOT loaded: an instrument-scale intensity with no normalization, and "
        "MEASURED not to be the numerator of the released fold change (Intensity "
        "PrecMz / fold-change varies within a key by a median relative spread of "
        "0.185, maximum 3.34), so it reconstructs nothing",
    ),
    _legend(
        TABLE_S6,
        "Datafile Name",
        "Name of LC-MS/MS datafile",
        note="one file per released row; its well MEASURED to be the strain's own "
        "Table S3 well on all 1,256 rows, while its plate number is the LC-MS/MS run "
        "plate rather than the library plate",
    ),
)


def read_column_legends(path: str | Path, sheet: str = LEGEND_SHEET) -> dict[str, str]:
    """The ``Legend`` sheet of a released workbook as ``{column: description}``."""
    frame = pd.read_excel(path, sheet_name=sheet, header=None)
    if frame.shape[1] != 2:
        raise ValueError(f"{sheet} of {path} has {frame.shape[1]} columns, expected 2")
    return {
        str(column).strip(): str(description).strip()
        for column, description in zip(frame[0], frame[1], strict=True)
    }


def check_column_legends(
    path: str | Path, columns: Sequence[SourcedColumn]
) -> list[SourcedColumn]:
    """Require every quoted legend to be verbatim in the workbook's ``Legend`` sheet."""
    legends = read_column_legends(path, columns[0].sheet)
    for column in columns:
        released = legends.get(column.column)
        if released is None:
            raise ValueError(
                f"{column.file} {column.sheet!r} has no row for column "
                f"{column.column!r}; it carries {sorted(legends)}"
            )
        if released != column.quote:
            raise ValueError(
                f"{column.file} {column.sheet!r} row {column.column!r} reads "
                f"{released!r}, not the pinned {column.quote!r}"
            )
    return list(columns)


# --------------------------------------------------------------------------- #
# Strain resolution (one rule, the metabolome loader's second one)
# --------------------------------------------------------------------------- #
#: The retention rule every family here applies, worded as the metabolome loader does.
REMAP_RULE = "b_number_remapped_by_the_annotation"
REMAP_RULE_DESCRIPTION = (
    "the pinned MG1655 annotation does not carry the released b-number as a locus tag "
    "of its own, and relates it to the locus the record would store by a route no "
    "DerivedIdentifierRoute member names, so no DerivedIdentifierMapping can record "
    "the remap and the record is dropped rather than remapped silently. A released "
    "b-number the annotation lists as a /gene_synonym of exactly one locus of the same "
    "namespace is kept instead, with a locus_tag_synonym mapping on its perturbation"
)


class ResolvedGene(BaseModel):
    """One kept library gene: its MG1655 locus tag, canonical symbol and sgRNA.

    ``identifier_mapping`` is ``None`` for a gene whose Table S1 b-number IS the stored
    locus tag, and a ``locus_tag_synonym`` mapping for one the annotation relates to its
    stored tag through a ``/gene_synonym`` (``rapp2026.locus_tag_synonym_mapping``).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    gene: str
    locus_tag: str
    symbol: str
    guide: Guide
    identifier_mapping: DerivedIdentifierMapping | None = None


def resolve_genes(
    genome: EcoliK12Genome,
    genes: Sequence[str],
    guides: Mapping[str, Guide],
    *,
    label: str,
) -> tuple[list[ResolvedGene], DropRule, IdentifierLedger]:
    """Settle each released gene symbol: its Table S1 b-number, locus tag and guide.

    The b-number is Table S1's, not the consuming table's: Table S2 leaves ``b-Nr.``
    blank on four of its 1,515 genes (``cydX``, ``gatC``, ``kdpF``, ``phnE``) while
    Table S1 releases one for every gene, and the two agree on the other 1,511.

    Stops (``LocusTagResolutionError``) below :data:`MIN_RESOLVED_FRACTION`.
    """
    missing = sorted(gene for gene in genes if gene not in guides)
    if missing:
        raise ValueError(f"{label}: {len(missing)} genes have no Table S1 sgRNA")
    b_numbers = pd.Series([guides[gene].b_number for gene in genes], dtype=object)
    stored, report = reconcile_locus_tags(genome, b_numbers, label=label)
    report.require_resolved(MIN_RESOLVED_FRACTION)
    pattern = LOCUS_TAG_PATTERNS[report.gene_namespace]

    kept: list[ResolvedGene] = []
    remapped: list[str] = []
    synonyms: list[str] = []
    disagreements: list[SymbolDisagreement] = []
    for gene, tag in zip(genes, stored.tolist(), strict=True):
        guide = guides[gene]
        mapping: DerivedIdentifierMapping | None = None
        if tag != guide.b_number or pattern.match(tag) is None:
            mapping = locus_tag_synonym_mapping(genome, guide.b_number, tag, pattern)
            resolution = genome.resolve_gene_name(guide.b_number)
            if mapping is None:
                remapped.append(
                    f"{gene} ({guide.b_number}): the annotation carries it as a "
                    f"{resolution.note}, so the record would store {tag}"
                )
                continue
            synonyms.append(
                f"{gene} ({guide.b_number}): the annotation carries it as a "
                f"{resolution.note}, so the record stores {tag} with route "
                f"{mapping.route}"
            )
        symbol_resolution = genome.resolve_gene_name(gene)
        if symbol_resolution.systematic_name != tag:
            disagreements.append(
                SymbolDisagreement(
                    gene=gene,
                    b_number=guide.b_number,
                    locus_tag=tag,
                    symbol_resolution=f"{symbol_resolution.status.value} "
                    f"{symbol_resolution.systematic_name}",
                )
            )
        kept.append(
            ResolvedGene(
                gene=gene,
                locus_tag=tag,
                symbol=canonical_symbol(genome, tag),
                guide=guide,
                identifier_mapping=mapping,
            )
        )
    rule = DropRule(
        rule=REMAP_RULE,
        description=REMAP_RULE_DESCRIPTION,
        n_records=len(remapped),
        items=remapped,
    )
    ledger = IdentifierLedger(
        reconciliation=report,
        min_resolved_fraction=MIN_RESOLVED_FRACTION,
        symbol_disagreements=disagreements,
        locus_tag_synonyms=synonyms,
    )
    return kept, rule, ledger


def crispri_genotype(resolved: ResolvedGene) -> Genotype:
    """The one guide-directed knockdown, named by its MG1655 b-number."""
    return Genotype(
        perturbations=[
            BacterialCrisprInterferencePerturbation(
                systematic_gene_name=resolved.locus_tag,
                perturbed_gene_name=resolved.symbol,
                gene_namespace=STRAIN_GENE_NAMESPACES[REFERENCE_STRAIN_NAME],
                identifier_mapping=resolved.identifier_mapping,
                crispr=CrisprConstruct(
                    effector=CAS_EFFECTOR,
                    guide_sequence=resolved.guide.spacer,
                    n_guides=GUIDES_PER_GENE,
                    library_pool=None,
                    effector_plasmid_ref=None,
                ),
            )
        ]
    )


def _drop_log(
    dataset: str,
    *,
    strain_tokens: int,
    reference_tokens: Sequence[str],
    kept: int,
    rule: DropRule,
) -> DropLog:
    """The retention ledger of one build, with its single rule accounted for."""
    source_records = strain_tokens - len(reference_tokens)
    log_record = DropLog(
        dataset=dataset,
        strain_tokens=strain_tokens,
        reference_tokens=sorted(reference_tokens),
        source_records=source_records,
        kept_records=kept,
        dropped_records=source_records - kept,
        rules=[rule],
    )
    if sum(r.n_records for r in log_record.rules) != log_record.dropped_records:
        raise RuntimeError("drop rules do not account for every dropped strain")
    return log_record


# --------------------------------------------------------------------------- #
# Family 1: the growth curves of Table S2, as the paper's AUC
# --------------------------------------------------------------------------- #
#: Table S2's label columns, left of the 181 time columns (header on the second row).
TABLE_S2_LABEL_COLUMNS = (
    "Replicate Nr.",
    "RXN Nr.",
    "Gene",
    "Guide Nr.",
    "b-Nr.",
    "Plate ID",
)


class GrowthCurveRow(BaseModel):
    """One Table S2 row: one culture of one strain, over the released time axis."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    row: int
    gene: str | None
    guide_id: str
    plate_well: str
    replicate: int

    @property
    def is_control(self) -> bool:
        """True for a control well (``Guide Nr.`` ``ctrlN``, ``Gene`` left empty)."""
        return CONTROL_TOKEN_PATTERN.match(self.guide_id) is not None

    @property
    def token(self) -> str:
        """What this row's strain is called: the gene, or the control well's id."""
        return self.guide_id if self.is_control else str(self.gene)


class GrowthTable(BaseModel):
    """Table S2, parsed: its time axis, its culture rows and the OD600 matrix."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    hours: tuple[float, ...]
    rows: tuple[GrowthCurveRow, ...]
    matrix: FloatMatrix


def read_growth_table(path: str | Path) -> GrowthTable:
    """Table S2's cultures and OD600 curves, refusing any other released shape.

    Checks the label columns, the 181-point 0-30 h axis at 10 min spacing, that every
    cell is finite, and that each strain carries replicates 1 to 3.
    """
    frame = pd.read_excel(path, sheet_name=TABLE_S2_SHEET, header=1)
    labels = list(TABLE_S2_LABEL_COLUMNS)
    if list(frame.columns[: len(labels)]) != labels:
        raise ValueError(
            f"Table S2 starts with {list(frame.columns[: len(labels)])}, expected "
            f"{labels}"
        )
    hours = tuple(float(column) for column in frame.columns[len(labels) :])
    if len(hours) != GROWTH_TIME_POINTS:
        raise ValueError(
            f"Table S2 holds {len(hours)} time columns, expected {GROWTH_TIME_POINTS}"
        )
    if hours[0] != 0.0 or hours[-1] != GROWTH_LAST_HOUR:
        raise ValueError(
            f"Table S2's axis runs {hours[0]} to {hours[-1]} h, expected 0 to "
            f"{GROWTH_LAST_HOUR}"
        )
    spacing = np.diff(np.asarray(hours, dtype=np.float64))
    if np.abs(spacing - GROWTH_INTERVAL_HOURS).max() > GROWTH_INTERVAL_TOLERANCE:
        raise ValueError(
            f"Table S2's time spacing runs {spacing.min():.6f} to {spacing.max():.6f} "
            f"h, not the {GROWTH_INTERVAL_HOURS:.6f} h the Methods state"
        )
    rows = tuple(
        GrowthCurveRow(
            row=index,
            gene=None if pd.isna(record["Gene"]) else str(record["Gene"]).strip(),
            guide_id=str(record["Guide Nr."]).strip(),
            plate_well=str(record["Plate ID"]).strip(),
            replicate=int(record["Replicate Nr."]),
        )
        for index, record in enumerate(
            frame.iloc[:, : len(labels)].to_dict("records"), start=1
        )
    )
    if len(rows) != TABLE_S2_ROWS:
        raise ValueError(f"Table S2 holds {len(rows)} rows, expected {TABLE_S2_ROWS}")
    matrix = frame.iloc[:, len(labels) :].to_numpy(dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError(
            f"Table S2 holds {int((~np.isfinite(matrix)).sum())} non-finite OD600 cells"
        )
    replicates = sorted(range(1, GROWTH_REPLICATES + 1))
    by_token: dict[str, list[int]] = {}
    for row in rows:
        by_token.setdefault(row.token, []).append(row.replicate)
    wrong = {
        token: reps for token, reps in by_token.items() if sorted(reps) != replicates
    }
    if wrong:
        token, reps = next(iter(sorted(wrong.items())))
        raise ValueError(
            f"Table S2 gives {token!r} replicates {sorted(reps)}, expected {replicates}"
        )
    return GrowthTable(hours=hours, rows=rows, matrix=matrix)


class StrainGrowth(BaseModel):
    """One strain's released growth, as the paper's AUC over its three cultures."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    token: str
    is_control: bool
    plate_well: str
    guide_id: str
    areas: tuple[float, ...]

    @property
    def mean_area(self) -> float:
        """The strain's AUC: the mean over its replicate cultures."""
        return float(np.mean(self.areas))


def strain_growth(table: GrowthTable) -> list[StrainGrowth]:
    """The trapezoid AUC of every released culture, grouped by strain.

    ``numpy.trapezoid`` over the released hour axis is the paper's ``trapz.m``
    (``SOURCED_VALUES['growth_assay']``).
    """
    areas = np.asarray(
        np.trapezoid(table.matrix, np.asarray(table.hours, dtype=np.float64), axis=1),
        dtype=np.float64,
    )
    grouped: dict[str, list[tuple[int, GrowthCurveRow]]] = {}
    for index, row in enumerate(table.rows):
        grouped.setdefault(row.token, []).append((index, row))
    strains: list[StrainGrowth] = []
    for token, entries in grouped.items():
        entries.sort(key=lambda item: item[1].replicate)
        first = entries[0][1]
        strains.append(
            StrainGrowth(
                token=token,
                is_control=first.is_control,
                plate_well=first.plate_well,
                guide_id=first.guide_id,
                areas=tuple(float(areas[index]) for index, _ in entries),
            )
        )
    strains.sort(key=lambda strain: strain.token)
    return strains


class GrowthDefectCheck(BaseModel):
    """The paper's growth-defect split, recomputed from the released curves."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cutoff: float
    n_strains: int
    measured_growth_defect: int
    measured_no_growth_defect: int
    paper_growth_defect: int
    paper_no_growth_defect: int
    strain_difference: int
    measured_from_mean_curve: int
    auc_minimum: float
    auc_maximum: float


def check_growth_defects(
    table: GrowthTable, strains: Sequence[StrainGrowth]
) -> GrowthDefectCheck:
    """Reproduce the paper's AUC classification, refusing a different count.

    Two routes to the per-strain AUC are computed, because the caption's "means from
    n=3 cultures" allows either: the mean of the three cultures' areas, and the area of
    the mean curve. They agree on the split, which is what makes the stored statistic
    the paper's rather than one reading of it.
    """
    genes = [strain for strain in strains if not strain.is_control]
    if len(genes) != LIBRARY_GENES:
        raise ValueError(
            f"Table S2 holds {len(genes)} library genes, the paper states "
            f"{LIBRARY_GENES}"
        )
    areas = np.asarray([strain.mean_area for strain in genes], dtype=np.float64)
    hours = np.asarray(table.hours, dtype=np.float64)
    by_token: dict[str, list[int]] = {}
    for index, row in enumerate(table.rows):
        if not row.is_control:
            by_token.setdefault(row.token, []).append(index)
    mean_curves = np.asarray(
        [table.matrix[by_token[strain.token], :].mean(axis=0) for strain in genes]
    )
    from_mean_curve = int(
        (np.trapezoid(mean_curves, hours, axis=1) < AUC_DEFECT_CUTOFF).sum()
    )
    measured = int((areas < AUC_DEFECT_CUTOFF).sum())
    result = GrowthDefectCheck(
        cutoff=AUC_DEFECT_CUTOFF,
        n_strains=len(genes),
        measured_growth_defect=measured,
        measured_no_growth_defect=len(genes) - measured,
        paper_growth_defect=PAPER_GROWTH_DEFECT,
        paper_no_growth_defect=PAPER_NO_GROWTH_DEFECT,
        strain_difference=measured - PAPER_GROWTH_DEFECT,
        measured_from_mean_curve=from_mean_curve,
        auc_minimum=float(areas.min()),
        auc_maximum=float(areas.max()),
    )
    if measured != MEASURED_GROWTH_DEFECT:
        raise ValueError(
            f"{measured} strains fall below the AUC cutoff {AUC_DEFECT_CUTOFF}, not "
            f"the {MEASURED_GROWTH_DEFECT} measured on the pinned table"
        )
    if from_mean_curve != measured:
        raise ValueError(
            f"the AUC of the mean curve classifies {from_mean_curve} strains as "
            f"defective, the mean of the three areas {measured}"
        )
    return result


def fitness_phenotype(ratios: Sequence[float], n_samples: int) -> FitnessPhenotype:
    """One strain's AUC ratio against the control mean, with its replicate spread.

    The three released cultures give three ratios, so the reported uncertainty is their
    SAMPLE standard deviation and ``fitness_se`` is derived from it as ``sd / sqrt(n)``.
    """
    array = np.asarray(ratios, dtype=np.float64)
    if array.size != n_samples:
        raise ValueError(f"{array.size} ratios for n_samples={n_samples}")
    return FitnessPhenotype(
        fitness=float(array.mean()),
        fitness_uncertainty=float(array.std(ddof=1)),
        fitness_uncertainty_type=UncertaintyType.sample_sd,
        n_samples=n_samples,
        sample_unit=SampleUnit.biological_replicate,
    )


# --------------------------------------------------------------------------- #
# Families 2 and 3: the accumulation tables (Table S6 and Table S5)
# --------------------------------------------------------------------------- #
class Accumulation(BaseModel):
    """One released strain-metabolite pair, on whichever platform released it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    row: int
    gene: str
    key: str
    abbreviation: str
    adduct: str
    polarity: str
    mz: float
    monoisotopic_mass: float
    kegg: str
    value: float
    replicates: tuple[float, ...]
    reference_level: float
    reference_n: int
    qc_passed: bool | None = None
    datafile: str | None = None

    @property
    def is_control(self) -> bool:
        """True for a control well's token (``ctrlN``)."""
        return CONTROL_TOKEN_PATTERN.match(self.gene) is not None


def _accumulation_key(abbreviation: str, adduct: str) -> str:
    """The stored metabolite key: Table S4's ``Abbr`` form, ``frdp[M-H]-``."""
    return f"{abbreviation}{adduct}"


def _released_kegg(kegg: str) -> str:
    """One accumulation table's KEGG string on Table S9's spelling.

    Both tables write a missing id as ``NaN`` where Table S9 writes ``XXX``, so each
    ``-``-joined member is normalized before the two strings are compared.
    """
    return "-".join(
        MISSING_KEGG_IDENTITY
        if token.lower() == MISSING_KEGG_RELEASED.lower()
        else token
        for token in kegg.split("-")
    )


def _check_identity(
    row: int,
    key: str,
    abbreviation: str,
    adduct: str,
    polarity: str,
    mz: float,
    monoisotopic_mass: float,
    kegg: str,
    metabolites: Mapping[str, Metabolite],
) -> Metabolite:
    """Join one released pair to Table S9's identity layer, refusing a mismatch.

    Four checks: the abbreviation is a Table S9 group, the released monoisotopic mass is
    that group's exactly, a singly charged m/z is that mass plus or minus a proton, and
    the released KEGG string is Table S9's (the accumulation tables write a missing id
    as ``NaN`` where Table S9 writes ``XXX``). The polarity must be the adduct's.
    """
    metabolite = metabolites.get(abbreviation)
    if metabolite is None:
        raise ValueError(
            f"row {row}: {key!r} names abbreviation {abbreviation!r}, which Table S9 "
            "does not carry"
        )
    if monoisotopic_mass != metabolite.monoisotopic_mass:
        raise ValueError(
            f"row {row}: {key!r} releases monoisotopic mass {monoisotopic_mass}, "
            f"Table S9 has {metabolite.monoisotopic_mass}"
        )
    expected_polarity = POSITIVE_POLARITY if "+" in adduct else NEGATIVE_POLARITY
    if polarity != expected_polarity:
        raise ValueError(
            f"row {row}: {key!r} is polarity {polarity!r}, but adduct {adduct!r} is "
            f"{expected_polarity!r}"
        )
    if adduct in ("[M+H]+", "[M-H]-"):
        sign = 1.0 if adduct == "[M+H]+" else -1.0
        offset = mz - metabolite.monoisotopic_mass - sign * PROTON_MASS
        if abs(offset) > PROTON_MASS_TOLERANCE:
            raise ValueError(
                f"row {row}: {key!r} m/z minus the monoisotopic mass is "
                f"{mz - metabolite.monoisotopic_mass:.6f}, not {sign * PROTON_MASS}"
            )
    released = _released_kegg(kegg)
    if released != metabolite.kegg:
        raise ValueError(
            f"row {row}: {key!r} releases KEGG {kegg!r}, Table S9 has "
            f"{metabolite.kegg!r}"
        )
    return metabolite


def read_targeted(
    path: str | Path, metabolites: Mapping[str, Metabolite], wells: Mapping[str, str]
) -> list[Accumulation]:
    """Table S6's targeted LC-MS/MS fold changes, one per strain-metabolite pair.

    ``wells`` maps a strain token to its Table S3 well, which every released
    ``DataFile`` name carries; the run plate in that name is the LC-MS/MS plate and not
    the library plate, so only the well is checked.
    """
    frame = pd.read_excel(path, sheet_name=TABLE_S6_SHEET, header=0)
    if len(frame) != TARGETED_PAIRS:
        raise ValueError(
            f"Table S6 holds {len(frame)} rows, the paper states {TARGETED_PAIRS}"
        )
    counts = (
        frame["Abbreviation"].astype(str).str.strip()
        + frame["Mode"].astype(str).str.strip()
    ).value_counts()
    out: list[Accumulation] = []
    seen: set[tuple[str, str]] = set()
    for row, record in enumerate(frame.to_dict("records"), start=1):
        gene = str(record["Gene"]).strip()
        abbreviation = str(record["Abbreviation"]).strip()
        adduct = str(record["Mode"]).strip()
        key = _accumulation_key(abbreviation, adduct)
        if (gene, key) in seen:
            raise ValueError(f"Table S6 lists ({gene!r}, {key!r}) more than once")
        seen.add((gene, key))
        metabolite = _check_identity(
            row,
            key,
            abbreviation,
            adduct,
            str(record["Polarity"]).strip(),
            float(record["PrecMz"]),
            float(record["Monoisotopic Mass"]),
            str(record["Kegg"]).strip(),
            metabolites,
        )
        datafile = str(record["DataFile"]).strip()
        match = DATAFILE_PATTERN.match(datafile)
        if match is None:
            raise ValueError(f"Table S6 row {row}: {datafile!r} is not an mzML name")
        if match["gene"] != gene or match["metabolite"] != abbreviation:
            raise ValueError(
                f"Table S6 row {row}: {datafile!r} names "
                f"({match['metabolite']!r}, {match['gene']!r}), not ({abbreviation!r}, "
                f"{gene!r})"
            )
        if match["well"] != wells[gene]:
            raise ValueError(
                f"Table S6 row {row}: {datafile!r} is well {match['well']!r}, but "
                f"Table S3 puts {gene!r} in {wells[gene]!r}"
            )
        value = float(record["fold-change"])
        out.append(
            Accumulation(
                row=row,
                gene=gene,
                key=key,
                abbreviation=abbreviation,
                adduct=adduct,
                polarity=str(record["Polarity"]).strip(),
                mz=float(record["PrecMz"]),
                monoisotopic_mass=metabolite.monoisotopic_mass,
                kegg=metabolite.kegg,
                value=value,
                replicates=(value,),
                reference_level=1.0,
                reference_n=int(counts[key]),
                qc_passed=bool(int(record["QC passed"])),
                datafile=datafile,
            )
        )
    qc = sum(1 for item in out if item.qc_passed)
    if qc != TARGETED_QC_PASSED:
        raise ValueError(
            f"Table S6 marks {qc} rows QC passed, the paper states {TARGETED_QC_PASSED}"
        )
    per_strain = pd.Series([item.gene for item in out]).value_counts()
    if int(per_strain.max()) > TARGETED_MAX_PER_STRAIN:
        raise ValueError(
            f"Table S6 gives one strain {int(per_strain.max())} metabolites, above the "
            f"stated maximum of {TARGETED_MAX_PER_STRAIN}"
        )
    return out


def batch_sizes(samples: Mapping[str, SampleRow]) -> dict[int, int]:
    """``{batch: samples in it}`` from Table S3, the population each median is over."""
    sizes: dict[int, int] = {}
    for row in samples.values():
        sizes[row.sample.batch] = sizes.get(row.sample.batch, 0) + 1
    return dict(sorted(sizes.items()))


def batch_of_gene(samples: Mapping[str, SampleRow]) -> dict[str, int]:
    """``{strain token: its FI-MS batch}``, which both its plates share on all 1,513."""
    batches: dict[str, int] = {}
    for row in samples.values():
        seen = batches.setdefault(row.sample.gene, row.sample.batch)
        if seen != row.sample.batch:
            raise ValueError(
                f"Table S3 puts {row.sample.gene!r} in batches {seen} and "
                f"{row.sample.batch}, so it has no single batch median"
            )
    return batches


def read_intensities(
    path: str | Path,
    metabolites: Mapping[str, Metabolite],
    reference_n: Mapping[str, int],
) -> list[Accumulation]:
    """Table S5's absolute FI-MS intensities, with the batch median they are ratios to.

    Three checks make the stored numbers the release's own: ``Mean_Int`` is the mean of
    the two released plate intensities, ``Mean_Int / Mean_FC`` equals both
    ``R1_Int / R1_FC`` and ``R2_Int / R2_FC`` (so the denominator is one number per
    feature per batch), and that denominator is constant across the strains of a batch.
    """
    frame = pd.read_excel(path, sheet_name=TABLE_S5_SHEET, header=0)
    if len(frame) != INTENSITY_ROWS:
        raise ValueError(f"Table S5 holds {len(frame)} rows, expected {INTENSITY_ROWS}")
    out: list[Accumulation] = []
    seen: set[tuple[str, str]] = set()
    for row, record in enumerate(frame.to_dict("records"), start=1):
        gene = str(record["Gene"]).strip()
        abbreviation = str(record["Metabolite Abbreviation"]).strip()
        adduct = str(record["Mode"]).strip()
        key = _accumulation_key(abbreviation, adduct)
        if (gene, key) in seen:
            raise ValueError(f"Table S5 lists ({gene!r}, {key!r}) more than once")
        seen.add((gene, key))
        metabolite = _check_identity(
            row,
            key,
            abbreviation,
            adduct,
            str(record["Polarity"]).strip(),
            float(record["Mass"]),
            float(record["MonoMass"]),
            str(record["Kegg ID"]).strip(),
            metabolites,
        )
        mean_intensity = float(record["Mean_Int"])
        plates = (float(record["R1_Int"]), float(record["R2_Int"]))
        folds = (float(record["R1_FC"]), float(record["R2_FC"]))
        if abs(mean_intensity - float(np.mean(plates))) > (
            INTENSITY_TOLERANCE * mean_intensity
        ):
            raise ValueError(
                f"Table S5 row {row}: Mean_Int {mean_intensity} is not the mean of "
                f"{plates}"
            )
        medians = (plates[0] / folds[0], plates[1] / folds[1])
        median = mean_intensity / float(record["Mean_FC"])
        worst = max(abs(value - median) for value in medians)
        if worst > INTENSITY_TOLERANCE * median:
            raise ValueError(
                f"Table S5 row {row}: the per-plate batch medians {medians} differ "
                f"from Mean_Int / Mean_FC {median} by up to {worst:.3g}"
            )
        out.append(
            Accumulation(
                row=row,
                gene=gene,
                key=key,
                abbreviation=abbreviation,
                adduct=adduct,
                polarity=str(record["Polarity"]).strip(),
                mz=float(record["Mass"]),
                monoisotopic_mass=metabolite.monoisotopic_mass,
                kegg=metabolite.kegg,
                value=mean_intensity,
                replicates=plates,
                reference_level=median,
                reference_n=reference_n[gene],
            )
        )
    flagged = int(frame["LC-MS/MS"].fillna(0).astype(int).sum())
    if flagged != TARGETED_PAIRS:
        raise ValueError(
            f"Table S5 flags {flagged} rows for targeted LC-MS/MS, Table S6 holds "
            f"{TARGETED_PAIRS}"
        )
    return out


class BatchMedianCheck(BaseModel):
    """The back-solved per-batch median intensity, per feature per batch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_groups: int
    n_multi_strain_groups: int
    max_relative_spread: float
    tolerance: float
    batch_sizes: dict[int, int]


def check_batch_medians(
    items: Sequence[Accumulation], batch_of: Mapping[str, int], sizes: Mapping[int, int]
) -> BatchMedianCheck:
    """Require the back-solved median to be one number per (feature, batch).

    This is what makes the reference level a MEASUREMENT rather than a reading of the
    Methods: the median the released fold change divides by cannot depend on the strain,
    so every strain of a batch must back-solve the same number for a feature.
    """
    groups: dict[tuple[str, int], list[float]] = {}
    for item in items:
        groups.setdefault((item.key, batch_of[item.gene]), []).append(
            item.reference_level
        )
    worst = 0.0
    multi = 0
    for values in groups.values():
        if len(values) < 2:
            continue
        multi += 1
        mean = float(np.mean(values))
        worst = max(worst, (max(values) - min(values)) / mean)
    result = BatchMedianCheck(
        n_groups=len(groups),
        n_multi_strain_groups=multi,
        max_relative_spread=worst,
        tolerance=INTENSITY_TOLERANCE,
        batch_sizes=dict(sorted(sizes.items())),
    )
    if worst > INTENSITY_TOLERANCE:
        raise ValueError(
            f"the back-solved batch median varies within a (feature, batch) group by "
            f"up to {worst:.3g}, above {INTENSITY_TOLERANCE}, so it is not the "
            "denominator of the released fold change"
        )
    return result


class PlatformAgreement(BaseModel):
    """How far the targeted LC-MS/MS values are from the stored FI-MS ones."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_pairs: int
    pearson_r_linear: float
    pearson_r_log2: float
    median_abs_log2_difference: float
    expected: dict[str, float]
    tolerance: float


def pearson_r(x: FloatMatrix, y: FloatMatrix) -> float:
    """Pearson correlation of two equal-length vectors, written out."""
    dx = x - x.mean()
    dy = y - y.mean()
    return float((dx * dy).sum() / np.sqrt((dx**2).sum() * (dy**2).sum()))


def check_platform_agreement(
    targeted: Sequence[Accumulation], intensities: Sequence[Accumulation]
) -> PlatformAgreement:
    """Measure the two platforms against each other, refusing a different answer.

    The Table S6 rows join 1:1 onto Table S5 on (gene, abbreviation, adduct), and the
    comparison is the stored FI-MS fold change (``Mean_FC``, which is exactly what the
    metabolome loader stores) against the targeted EIC fold change. The numbers are
    pinned: if a re-run ever reproduced the stored values, this family would be a
    duplicate rather than a second platform, and the build must say so.
    """
    fold_change = {
        (item.gene, item.key): item.value / item.reference_level for item in intensities
    }
    missing = [
        (item.gene, item.key)
        for item in targeted
        if (item.gene, item.key) not in fold_change
    ]
    if missing:
        raise ValueError(
            f"{len(missing)} Table S6 pairs are not in Table S5, first {missing[0]}"
        )
    fi_ms = np.asarray(
        [fold_change[(item.gene, item.key)] for item in targeted], dtype=np.float64
    )
    lc_ms = np.asarray([item.value for item in targeted], dtype=np.float64)
    result = PlatformAgreement(
        n_pairs=len(targeted),
        pearson_r_linear=pearson_r(fi_ms, lc_ms),
        pearson_r_log2=pearson_r(np.log2(fi_ms), np.log2(lc_ms)),
        median_abs_log2_difference=float(
            np.median(np.abs(np.log2(lc_ms) - np.log2(fi_ms)))
        ),
        expected=dict(EXPECTED_PLATFORM_AGREEMENT),
        tolerance=AGREEMENT_TOLERANCE,
    )
    measured = {
        "n_pairs": float(result.n_pairs),
        "pearson_r_linear": result.pearson_r_linear,
        "pearson_r_log2": result.pearson_r_log2,
        "median_abs_log2_difference": result.median_abs_log2_difference,
    }
    for name, expected in EXPECTED_PLATFORM_AGREEMENT.items():
        if abs(measured[name] - expected) > AGREEMENT_TOLERANCE:
            raise ValueError(
                f"the two platforms' {name} is {measured[name]:.6f}, not the pinned "
                f"{expected}"
            )
    return result


def metabolite_phenotype(
    items: Sequence[Accumulation], measurement_type: str, target_ids: Mapping[str, str]
) -> MetabolitePhenotype:
    """One strain's released values on one platform, keyed by feature.

    ``metabolite_level_se`` is the standard error of the stored value over the released
    replicates, which is ``|r1 - r2| / 2`` for the two FI-MS plates and is left off
    entirely for the single targeted injection.
    """
    level = {item.key: item.value for item in items}
    n_replicates = {item.key: len(item.replicates) for item in items}
    standard_error = {
        item.key: float(
            np.std(np.asarray(item.replicates, dtype=np.float64), ddof=1)
            / np.sqrt(len(item.replicates))
        )
        for item in items
        if len(item.replicates) > 1
    }
    stored_ids = {
        item.key: target_ids[item.key] for item in items if item.key in target_ids
    }
    return MetabolitePhenotype(
        metabolite_level=level,
        metabolite_level_se=standard_error or None,
        n_replicates=n_replicates,
        measurement_type=measurement_type,
        target_metabolite_ids=stored_ids or None,
        provenance_gaps=metabolite_identity_gaps(list(level), stored_ids),
    )


def reference_phenotype(
    items: Sequence[Accumulation], measurement_type: str, target_ids: Mapping[str, str]
) -> MetabolitePhenotype:
    """The denominator of one strain's released values, on the same keys and scale."""
    stored_ids = {
        item.key: target_ids[item.key] for item in items if item.key in target_ids
    }
    return MetabolitePhenotype(
        metabolite_level={item.key: item.reference_level for item in items},
        metabolite_level_se=None,
        n_replicates={item.key: item.reference_n for item in items},
        measurement_type=measurement_type,
        target_metabolite_ids=stored_ids or None,
        provenance_gaps=metabolite_identity_gaps(
            [item.key for item in items], stored_ids
        ),
    )


def target_metabolite_ids(
    items: Sequence[Accumulation], metabolites: Mapping[str, Metabolite]
) -> dict[str, str]:
    """Feature key -> its single BiGG id, where the group names exactly one metabolite.

    A merged isobaric group (FI-MS cannot separate equal masses, and the targeted method
    inherits its annotation) is ABSENT from the map rather than assigned one of its
    candidates, exactly as the metabolome loader leaves it.
    """
    out: dict[str, str] = {}
    for item in items:
        metabolite = metabolites[item.abbreviation]
        if metabolite.n_isobaric == 1:
            out[item.key] = metabolite.bigg_ids[0]
    return out


class AccumulationLedger(BaseModel):
    """What one accumulation family stored, and what it left out."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    measurement_type: str
    n_rows: int
    n_tokens: int
    control_tokens: tuple[str, ...]
    n_records: int
    n_values: int
    n_keys: int
    n_single_identity_keys: int
    values_per_record_min: int
    values_per_record_max: int
    columns: tuple[SourcedColumn, ...]


def accumulation_ledger(
    measurement_type: str,
    items: Sequence[Accumulation],
    stored: Mapping[str, Sequence[Accumulation]],
    target_ids: Mapping[str, str],
    columns: Sequence[SourcedColumn],
) -> AccumulationLedger:
    """Count one family's released rows, kept records and identity coverage."""
    sizes = [len(group) for group in stored.values()]
    keys = {item.key for group in stored.values() for item in group}
    return AccumulationLedger(
        measurement_type=measurement_type,
        n_rows=len(items),
        n_tokens=len({item.gene for item in items}),
        control_tokens=tuple(sorted({item.gene for item in items if item.is_control})),
        n_records=len(stored),
        n_values=sum(sizes),
        n_keys=len(keys),
        n_single_identity_keys=len(keys & set(target_ids)),
        values_per_record_min=min(sizes),
        values_per_record_max=max(sizes),
        columns=tuple(columns),
    )


# --------------------------------------------------------------------------- #
# The datasets
# --------------------------------------------------------------------------- #
class _Rapp2026Dataset(ExperimentDataset):
    """Shared plumbing: the injected genome and the raw-mirror link step."""

    REFERENCE_STRAIN: ClassVar[EcoliK12StrainName] = "MG1655"
    #: The mirrored workbooks this dataset reads, a subset of ``rapp2026.RAW_FILES``.
    CONSUMED_FILES: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        root: str,
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
    def raw_file_names(self) -> list[str]:
        """The workbooks this family reads, linked from the shared raw mirror."""
        return list(self.CONSUMED_FILES)

    def download(self) -> None:
        """Link this family's mirror files into ``raw/`` after checking their sha256."""
        data_root = _data_root()
        manifest = load_manifest(data_root)
        os.makedirs(self.raw_dir, exist_ok=True)
        for raw in RAW_FILES:
            if raw.name not in self.CONSUMED_FILES:
                continue
            check_manifest_pin(
                raw.mirror_relpath,
                manifest_sha256(manifest, raw.mirror_relpath),
                raw.sha256,
            )
            src = raw_mirror_dir(data_root) / raw.mirror_relpath
            if not src.exists():
                raise RuntimeError(f"required raw artifact missing from mirror: {src}")
            link_verified(src, osp.join(self.raw_dir, raw.name), raw.sha256)
        log.info("%s raw files linked into %s", type(self).__name__, self.raw_dir)

    def _raw(self, name: str) -> str:
        return osp.join(self.raw_dir, name)

    def _consumed_sha256(self) -> dict[str, str]:
        """``{name: sha256}`` of the files this family reads."""
        return {name: DATA_SHA256[name] for name in self.CONSUMED_FILES}

    def _genome(self) -> EcoliK12Genome:
        """The injected genome, or the reference strain's default cache; another
        assembly set is refused.
        """
        if self.ecoli_genome is None:
            self.ecoli_genome = bacterial_genome("ecoli", self.REFERENCE_STRAIN)
        if self.ecoli_genome.ASSEMBLY_SET != MG1655_ASSEMBLY_SET:
            raise ValueError(
                f"{type(self).__name__} needs the {MG1655_ASSEMBLY_SET} genome, got "
                f"{self.ecoli_genome.ASSEMBLY_SET}"
            )
        return self.ecoli_genome

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Preprocessing is handled inside process() for these datasets."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled by this module's builders."""
        raise NotImplementedError


@register_dataset
class GrowthAucRapp2026Dataset(_Rapp2026Dataset):
    """Growth of Rapp 2026's CRISPRi library, as the paper's AUC over the controls."""

    CONSUMED_FILES: ClassVar[tuple[str, ...]] = (TABLE_S1, TABLE_S2)

    def __init__(
        self,
        root: str = "data/torchcell/growth_auc_rapp2026",
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        ecoli_genome: EcoliK12Genome | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with this family's default dev-tree root."""
        super().__init__(
            root, io_workers, transform, pre_transform, ecoli_genome, **kwargs
        )

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return BacterialFitnessExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return BacterialFitnessExperimentReference

    @post_process
    def process(self) -> None:
        """Parse Table S1 + Table S2 into one fitness record per strain."""
        verify_raw_files(self.raw_dir, self._consumed_sha256())
        guides = read_guides(self._raw(TABLE_S1))
        table = read_growth_table(self._raw(TABLE_S2))
        strains = strain_growth(table)
        defects = check_growth_defects(table, strains)

        controls = [strain for strain in strains if strain.is_control]
        if len(controls) != GROWTH_CONTROL_TOKENS:
            raise ValueError(
                f"Table S2 holds {len(controls)} control wells, expected "
                f"{GROWTH_CONTROL_TOKENS}"
            )
        control_areas = [area for strain in controls for area in strain.areas]
        baseline = float(np.mean(control_areas))

        genome = self._genome()
        genes = [strain for strain in strains if not strain.is_control]
        resolved, rule, identifiers = resolve_genes(
            genome,
            [strain.token for strain in genes],
            guides,
            label=f"{self.name} Table S1 b-numbers",
        )
        growth_of = {strain.token: strain for strain in genes}
        drops = _drop_log(
            self.name,
            strain_tokens=len(strains),
            reference_tokens=[strain.token for strain in controls],
            kept=len(resolved),
            rule=rule,
        )
        log.info(
            "Rapp 2026 growth: %d strain tokens (%d control) -> %d records; drops %s; "
            "control AUC %.4f over %d cultures; %d of %d strains below the AUC cutoff",
            len(strains),
            len(controls),
            len(resolved),
            {rule.rule: rule.n_records},
            baseline,
            len(control_areas),
            defects.measured_growth_defect,
            defects.n_strains,
        )

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        out = Path(self.preprocess_dir)
        (out / "dropped_records.json").write_text(drops.model_dump_json(indent=2))
        (out / "identifier_reconciliation.json").write_text(
            identifiers.model_dump_json(indent=2)
        )
        (out / "growth_defect_check.json").write_text(defects.model_dump_json(indent=2))
        (out / "control_baseline.json").write_text(
            json.dumps(
                {
                    "statistic": GROWTH_STATISTIC,
                    "n_control_wells": len(controls),
                    "n_control_cultures": len(control_areas),
                    "mean_auc": baseline,
                    "sd_auc": float(np.std(control_areas, ddof=1)),
                    "per_well_mean_auc": {
                        strain.token: strain.mean_area for strain in controls
                    },
                },
                indent=2,
            )
        )
        pd.DataFrame(
            [
                {
                    "record": index,
                    "gene": item.gene,
                    "b_number": item.guide.b_number,
                    "locus_tag": item.locus_tag,
                    "symbol": item.symbol,
                    "identifier_route": (
                        ""
                        if item.identifier_mapping is None
                        else item.identifier_mapping.route
                    ),
                    "sgrna_id": item.guide.sgrna_id,
                    "spacer": item.guide.spacer,
                    "plate_well": growth_of[item.gene].plate_well,
                    "auc_replicates": "; ".join(
                        f"{area:.6f}" for area in growth_of[item.gene].areas
                    ),
                    "mean_auc": growth_of[item.gene].mean_area,
                    "fitness": growth_of[item.gene].mean_area / baseline,
                    "growth_defect": growth_of[item.gene].mean_area < AUC_DEFECT_CUTOFF,
                }
                for index, item in enumerate(resolved)
            ]
        ).to_csv(out / "strains.csv", index=False)

        env = environment()
        reference = BacterialFitnessExperimentReference(
            dataset_name=self.name,
            genome_reference=assembly_reference(
                self.REFERENCE_STRAIN, background=host_background()
            ),
            environment_reference=env.model_copy(),
            phenotype_reference=fitness_phenotype(
                [area / baseline for area in control_areas], len(control_areas)
            ),
        )
        lmdb_env, interned_env = self._open_write_lmdb(
            osp.join(self.processed_dir, "lmdb")
        )
        with lmdb_env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for index, item in enumerate(tqdm(resolved, desc="rapp2026-growth")):
                experiment = BacterialFitnessExperiment(
                    dataset_name=self.name,
                    genotype=crispri_genotype(item),
                    environment=env,
                    phenotype=fitness_phenotype(
                        [area / baseline for area in growth_of[item.gene].areas],
                        GROWTH_REPLICATES,
                    ),
                )
                txn.put(
                    f"{index}".encode(),
                    self._intern_record(experiment, reference, PUBLICATION, itxn),
                )
        lmdb_env.close()
        interned_env.close()
        log.info("Wrote %d Rapp 2026 growth experiments to LMDB", len(resolved))


class _AccumulationDataset(_Rapp2026Dataset):
    """Shared build for the two accumulation families (Table S6 and Table S5)."""

    #: What the stored number is; one per family.
    MEASUREMENT_TYPE: ClassVar[str] = ""
    #: The released columns this family stores or gates on.
    COLUMNS: ClassVar[tuple[SourcedColumn, ...]] = ()

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return BacterialMetaboliteExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return BacterialMetaboliteExperimentReference

    def _read(
        self, metabolites: Mapping[str, Metabolite], samples: Mapping[str, SampleRow]
    ) -> tuple[list[Accumulation], dict[str, Any]]:
        """This family's released rows, plus whatever it checked while reading them."""
        raise NotImplementedError

    def _build_records(self) -> None:
        """Parse one accumulation table into one record per CRISPRi strain."""
        verify_raw_files(self.raw_dir, self._consumed_sha256())
        guides = read_guides(self._raw(TABLE_S1))
        metabolites = read_metabolites(self._raw(TABLE_S9))
        samples = read_sample_rows(self._raw(TABLE_S3))
        for file in sorted({column.file for column in self.COLUMNS}):
            check_column_legends(
                self._raw(file),
                [column for column in self.COLUMNS if column.file == file],
            )
        items, checks = self._read(metabolites, samples)
        tokens = {item.gene for item in items}
        if len(tokens) != ACCUMULATION_TOKENS:
            raise ValueError(
                f"{self.name}: the table holds {len(tokens)} strain tokens, expected "
                f"{ACCUMULATION_TOKENS}"
            )
        controls = sorted(
            token for token in tokens if CONTROL_TOKEN_PATTERN.match(token)
        )
        if len(controls) != ACCUMULATION_CONTROL_TOKENS:
            raise ValueError(
                f"{self.name}: {len(controls)} control tokens, expected "
                f"{ACCUMULATION_CONTROL_TOKENS}"
            )

        genome = self._genome()
        resolved, rule, identifiers = resolve_genes(
            genome,
            sorted(token for token in tokens if token not in controls),
            guides,
            label=f"{self.name} Table S1 b-numbers",
        )
        by_gene: dict[str, list[Accumulation]] = {}
        for item in items:
            by_gene.setdefault(item.gene, []).append(item)
        stored = {item.gene: by_gene[item.gene] for item in resolved}
        target_ids = target_metabolite_ids(items, metabolites)
        ledger = accumulation_ledger(
            self.MEASUREMENT_TYPE, items, stored, target_ids, self.COLUMNS
        )
        drops = _drop_log(
            self.name,
            strain_tokens=len(tokens),
            reference_tokens=controls,
            kept=len(resolved),
            rule=rule,
        )
        log.info(
            "%s: %d released rows over %d tokens (%d control) -> %d records, %d "
            "values; drops %s; %d of %d keys carry one BiGG id",
            self.name,
            ledger.n_rows,
            ledger.n_tokens,
            len(controls),
            ledger.n_records,
            ledger.n_values,
            {rule.rule: rule.n_records},
            ledger.n_single_identity_keys,
            ledger.n_keys,
        )

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        out = Path(self.preprocess_dir)
        (out / "dropped_records.json").write_text(drops.model_dump_json(indent=2))
        (out / "identifier_reconciliation.json").write_text(
            identifiers.model_dump_json(indent=2)
        )
        (out / "accumulation_ledger.json").write_text(ledger.model_dump_json(indent=2))
        for name, check in checks.items():
            (out / f"{name}.json").write_text(
                check.model_dump_json(indent=2)
                if isinstance(check, BaseModel)
                else json.dumps(check, indent=2)
            )
        pd.DataFrame(
            [
                {
                    "record": index,
                    "gene": item.gene,
                    "b_number": item.guide.b_number,
                    "locus_tag": item.locus_tag,
                    "symbol": item.symbol,
                    "identifier_route": (
                        ""
                        if item.identifier_mapping is None
                        else item.identifier_mapping.route
                    ),
                    "sgrna_id": item.guide.sgrna_id,
                    "spacer": item.guide.spacer,
                    "n_values": len(stored[item.gene]),
                    "keys": "; ".join(value.key for value in stored[item.gene]),
                }
                for index, item in enumerate(resolved)
            ]
        ).to_csv(out / "strains.csv", index=False)

        env = environment()
        genome_reference = assembly_reference(
            self.REFERENCE_STRAIN, background=host_background()
        )
        lmdb_env, interned_env = self._open_write_lmdb(
            osp.join(self.processed_dir, "lmdb")
        )
        with lmdb_env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for index, strain in enumerate(tqdm(resolved, desc=self.name)):
                values = stored[strain.gene]
                experiment = BacterialMetaboliteExperiment(
                    dataset_name=self.name,
                    genotype=crispri_genotype(strain),
                    environment=env,
                    phenotype=metabolite_phenotype(
                        values, self.MEASUREMENT_TYPE, target_ids
                    ),
                )
                reference = BacterialMetaboliteExperimentReference(
                    dataset_name=self.name,
                    genome_reference=genome_reference,
                    environment_reference=env.model_copy(),
                    phenotype_reference=reference_phenotype(
                        values, self.MEASUREMENT_TYPE, target_ids
                    ),
                )
                txn.put(
                    f"{index}".encode(),
                    self._intern_record(experiment, reference, PUBLICATION, itxn),
                )
        lmdb_env.close()
        interned_env.close()
        log.info("Wrote %d %s experiments to LMDB", len(resolved), self.name)


@register_dataset
class TargetedMetabolomeRapp2026Dataset(_AccumulationDataset):
    """Targeted LC-MS/MS of Rapp 2026's accumulating strain-metabolite pairs."""

    CONSUMED_FILES: ClassVar[tuple[str, ...]] = (
        TABLE_S1,
        TABLE_S3,
        TABLE_S5,
        TABLE_S6,
        TABLE_S9,
    )
    MEASUREMENT_TYPE: ClassVar[str] = TARGETED_MEASUREMENT_TYPE
    COLUMNS: ClassVar[tuple[SourcedColumn, ...]] = TARGETED_COLUMNS

    def __init__(
        self,
        root: str = "data/torchcell/targeted_metabolome_rapp2026",
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        ecoli_genome: EcoliK12Genome | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with this family's default dev-tree root."""
        super().__init__(
            root, io_workers, transform, pre_transform, ecoli_genome, **kwargs
        )

    def _read(
        self, metabolites: Mapping[str, Metabolite], samples: Mapping[str, SampleRow]
    ) -> tuple[list[Accumulation], dict[str, Any]]:
        """Table S6's fold changes, measured against the stored FI-MS values.

        Table S5 is read here too, and only for that comparison: a second platform has
        to be shown not to be a copy of the first.
        """
        wells = {row.sample.gene: row.well for row in samples.values()}
        sizes = batch_sizes(samples)
        reference_n = {
            gene: sizes[batch] for gene, batch in batch_of_gene(samples).items()
        }
        items = read_targeted(self._raw(TABLE_S6), metabolites, wells)
        intensities = read_intensities(self._raw(TABLE_S5), metabolites, reference_n)
        agreement = check_platform_agreement(items, intensities)
        return items, {"platform_agreement": agreement}

    @post_process
    def process(self) -> None:
        """Parse Table S6 into one targeted-LC-MS/MS record per strain."""
        self._build_records()


@register_dataset
class MetaboliteIntensityRapp2026Dataset(_AccumulationDataset):
    """Absolute FI-MS intensity of Rapp 2026's accumulating annotated features."""

    CONSUMED_FILES: ClassVar[tuple[str, ...]] = (TABLE_S1, TABLE_S3, TABLE_S5, TABLE_S9)
    MEASUREMENT_TYPE: ClassVar[str] = INTENSITY_MEASUREMENT_TYPE
    COLUMNS: ClassVar[tuple[SourcedColumn, ...]] = INTENSITY_COLUMNS

    def __init__(
        self,
        root: str = "data/torchcell/metabolite_intensity_rapp2026",
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        ecoli_genome: EcoliK12Genome | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with this family's default dev-tree root."""
        super().__init__(
            root, io_workers, transform, pre_transform, ecoli_genome, **kwargs
        )

    def _read(
        self, metabolites: Mapping[str, Metabolite], samples: Mapping[str, SampleRow]
    ) -> tuple[list[Accumulation], dict[str, Any]]:
        """Table S5's intensities, with the batch median they are ratios to."""
        sizes = batch_sizes(samples)
        batch_of = batch_of_gene(samples)
        reference_n = {gene: sizes[batch] for gene, batch in batch_of.items()}
        items = read_intensities(self._raw(TABLE_S5), metabolites, reference_n)
        medians = check_batch_medians(items, batch_of, sizes)
        return items, {"batch_median_check": medians}

    @post_process
    def process(self) -> None:
        """Parse Table S5 into one absolute-intensity record per strain."""
        self._build_records()


# --------------------------------------------------------------------------- #
# Verification (L0-L4) of the built LMDBs
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """``DATA_ROOT`` from the environment (the mirrors and build tree live under it)."""
    return os.environ["DATA_ROOT"]


GROWTH_ROOT_REL = "data/torchcell/growth_auc_rapp2026"
TARGETED_ROOT_REL = "data/torchcell/targeted_metabolome_rapp2026"
INTENSITY_ROOT_REL = "data/torchcell/metabolite_intensity_rapp2026"

GROWTH_PROVENANCE = Provenance(
    source_uri=f"torchcell-raw/{CITATION_KEY}/data/{TABLE_S2}",
    citation_key=CITATION_KEY,
    sha256=DATA_SHA256[TABLE_S2],
    method="trapezoid area under the released 181-point OD600 curve (0-30 h), the mean "
    "of the strain's three cultures over the grand mean of the 48 control cultures; "
    "the paper's own statistic (MATLAB trapz.m) and its AUC < 18 growth-defect split "
    "reproduced at build time",
    page="Cell Syst 2026 Table S2 (mmc3.xlsx, sheet Table_S2)",
)
TARGETED_PROVENANCE = Provenance(
    source_uri=f"torchcell-raw/{CITATION_KEY}/data/{TABLE_S6}",
    citation_key=CITATION_KEY,
    sha256=DATA_SHA256[TABLE_S6],
    method="targeted LC-MS/MS EIC peak-height fold change against the library median "
    "(one injection per strain-metabolite pair, no released uncertainty); reference = "
    "1.0, that median on the released scale. MEASURED distinct from the stored FI-MS "
    "fold changes: Pearson r 0.6722, median absolute log2 difference 1.1649",
    page="Cell Syst 2026 Table S6 (mmc7.xlsx, sheet Table_S6)",
)
INTENSITY_PROVENANCE = Provenance(
    source_uri=f"torchcell-raw/{CITATION_KEY}/data/{TABLE_S5}",
    citation_key=CITATION_KEY,
    sha256=DATA_SHA256[TABLE_S5],
    method="absolute FI-MS intensity of each accumulating annotated m/z feature, the "
    "mean of the strain's two plates (Mean_Int) with |R1_Int - R2_Int| / 2 as its "
    "standard error; reference = the per-batch median intensity that intensity's fold "
    "change is a ratio to, back-solved as Mean_Int / Mean_FC",
    page="Cell Syst 2026 Table S5 (mmc6.xlsx, sheet TableS5)",
)


def _expected_records(abs_root: str) -> int:
    """The kept-record count this build's own retention ledger reports."""
    return DropLog.model_validate_json(
        Path(abs_root, "preprocess", "dropped_records.json").read_text()
    ).kept_records


def _audit(report: VerificationReport, data_root: str) -> VerificationReport:
    """Add the provenance audit of every sourced value this module pins."""
    library = Path(data_root) / "torchcell-library"
    for value in SOURCED_VALUES.values():
        report.add(audit_sourced_value(value, library))
    return report


def verify_growth(data_root: str | None = None) -> VerificationReport:
    """Run the fitness L0-L4 gate on the built growth LMDB and write its report."""
    from torchcell.verification.fitness import verify_fitness_dataset
    from torchcell.verification.runners import _write_report, load_records

    base = data_root or _data_root()
    abs_root = osp.join(base, GROWTH_ROOT_REL)
    genome = bacterial_genome("ecoli", "MG1655", base)
    records = load_records(abs_root)
    report = verify_fitness_dataset(
        records,
        dataset_name="growth_auc_rapp2026",
        provenance=GROWTH_PROVENANCE,
        expected_count=_expected_records(abs_root),
        sgd_genes=set(genome.genbank.loci),
        gene_universe_label=f"MG1655 ({genome.ASSEMBLY_SET})",
        resolve_gene_name=genome.resolve_gene_name,
    )
    _write_report(_audit(report, base), osp.join(abs_root, "preprocess"))
    return report


def _verify_accumulation(
    root_rel: str, dataset_name: str, provenance: Provenance, data_root: str | None
) -> VerificationReport:
    """Run the metabolite L0-L4 gate on one built accumulation LMDB."""
    from torchcell.verification.metabolite import (
        metabolite_gene_set,
        verify_metabolite_dataset,
    )
    from torchcell.verification.runners import (
        _gene_set_for_reference,
        _write_report,
        load_records,
    )

    base = data_root or _data_root()
    abs_root = osp.join(base, root_rel)
    records = load_records(abs_root)
    report = verify_metabolite_dataset(
        records,
        dataset_name=dataset_name,
        provenance=provenance,
        expected_count=_expected_records(abs_root),
        reference_centered=False,
    )
    universe: set[str] = set()
    for reference in {
        json.dumps(record["reference"]["genome_reference"], sort_keys=True)
        for record in records
    }:
        universe |= _gene_set_for_reference(json.loads(reference), base)
    knocked_down = metabolite_gene_set(records)
    missing = sorted(knocked_down - universe)
    report.add(
        LevelResult(
            level=Level.L4,
            name="gene_containment_mg1655_b_numbers",
            passed=not missing,
            message=f"{len(knocked_down) - len(missing)} of {len(knocked_down)} "
            "knocked-down loci are MG1655 GenBank gene rows",
            details={
                "n_knocked_down": len(knocked_down),
                "n_universe": len(universe),
                "missing_examples": missing[:20],
            },
        )
    )
    _write_report(_audit(report, base), osp.join(abs_root, "preprocess"))
    return report


def verify_targeted(data_root: str | None = None) -> VerificationReport:
    """Run L0-L4 on the built targeted-LC-MS/MS LMDB and write its report."""
    return _verify_accumulation(
        TARGETED_ROOT_REL,
        "targeted_metabolome_rapp2026",
        TARGETED_PROVENANCE,
        data_root,
    )


def verify_intensity(data_root: str | None = None) -> VerificationReport:
    """Run L0-L4 on the built FI-MS intensity LMDB and write its report."""
    return _verify_accumulation(
        INTENSITY_ROOT_REL,
        "metabolite_intensity_rapp2026",
        INTENSITY_PROVENANCE,
        data_root,
    )


#: ``family -> (dataset class, dev-tree root, verifier)``, the CLI's whole surface.
FAMILIES: dict[
    str, tuple[type[_Rapp2026Dataset], str, Callable[[str], VerificationReport]]
] = {
    "growth": (GrowthAucRapp2026Dataset, GROWTH_ROOT_REL, verify_growth),
    "targeted": (TargetedMetabolomeRapp2026Dataset, TARGETED_ROOT_REL, verify_targeted),
    "intensity": (
        MetaboliteIntensityRapp2026Dataset,
        INTENSITY_ROOT_REL,
        verify_intensity,
    ),
}


def main(argv: list[str] | None = None) -> int:
    """CLI: ``build`` or ``verify`` one family's dev-tree LMDB (the mirror is shared
    with ``torchcell.datasets.ecoli.rapp2026 deposit``).
    """
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(
        prog="python -m torchcell.datasets.ecoli.rapp2026_platforms"
    )
    parser.add_argument("command", choices=("build", "verify"))
    parser.add_argument("family", choices=sorted(FAMILIES))
    args = parser.parse_args(argv)

    load_dotenv()
    data_root = _data_root()
    dataset_cls, root_rel, verifier = FAMILIES[args.family]
    if args.command == "build":
        dataset = dataset_cls(root=osp.join(data_root, root_rel))
        print(f"len = {len(dataset)}")
        return 0
    report = verifier(data_root)
    print(report.summary())
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
