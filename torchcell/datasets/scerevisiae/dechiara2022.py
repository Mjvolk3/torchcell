# torchcell/datasets/scerevisiae/dechiara2022.py
# [[torchcell.datasets.scerevisiae.dechiara2022]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/dechiara2022.py
# Test file: tests/torchcell/datasets/scerevisiae/test_dechiara2022.py
"""De Chiara 2022: the life cycle of the 1,011-isolate panel, three stores.

De Chiara et al. 2022 (Nat Ecol Evol 6:448-460, doi:10.1038/s41559-022-01671-9)
phenotyped the 1,011 sequenced S. cerevisiae isolates of Peter 2018 for the three parts
of the yeast life cycle the paper names: "asexual growth in many environments, capacity
to enter and pass through the sexual life cycle and form gametes and the chronological
life span (CLS)". Every per-isolate value is in Supplementary Table 3 of the
Supplementary Tables workbook (MOESM4), which this module reads from the raw mirror.

Three stores, one per phenotype family:

- ``DoublingTimeDeChiara2022Dataset`` (``EnvironmentResponseExperiment``,
  ``MeasurementType.growth_rate``): one record per (isolate, growth environment), the
  colony population doubling time in hours on 43 carbon and nitrogen environments.
- ``CellYieldDeChiara2022Dataset`` (``EnvironmentResponseExperiment``,
  ``MeasurementType.cell_yield``): the same 43 environments, the colony cell yield.
- ``LifeCycleDeChiara2022Dataset`` (``LifeCycleExperiment``): one record per (isolate,
  assay protocol), a culture's time course: sporulation efficiency and dyad fraction in
  potassium acetate (24 h, 72 h), sporulation efficiency in PBS (8 d), and
  stationary-phase viability in PBS (8 d, 20 d), SDC and calorie-restricted SDC (7, 21
  and 35 d).

GROWTH: Supplementary Table 3 carries each growth trait twice, as the control-normalized
log2 value the paper analysed and as an absolute value (hours, cells). The two blocks
describe different isolates on 519 of 1,011 rows: the normalized block is shifted by one
name over the alphabetical run BDM..CRL, and only the absolute block keeps the SUC2 and
MAL-locus sugar associations in those rows (measured by
``experiments/036-dataset-fixes-before-kg-build/scripts/dechiara2022_release_inventory.py``).
The absolute block is loaded and the normalized block is refused; where the two agree the
normalized value is the absolute one less a per-environment constant, so nothing the
normalized block carries is lost.

GENOTYPE: each isolate is a perturbation set off S288C built by
``torchcell.sequence.natural_isolate`` (``NaturalIsolateGenotypes``), the module every
natural-isolate loader on the Peter panel shares, so an isolate here is the same
``Genotype`` as in the Caudal 2024, Peter 2018 and Muenzner 2024 stores
(``SequenceVariantPerturbation`` per gene whose allele differs from the R64 reference
slice, ``NaturalGenePresencePerturbation`` per non-reference pangenome ORF the isolate
carries, ``NaturalGeneAbsencePerturbation`` per reference ORF it lacks, the HO
``DeletionPerturbation`` Peter's Table S1 marks, ``NaturalLossOfFunctionPerturbation``
per call of Peter's two loss-of-function matrices, and ``Genotype.natural_isolate`` with
the Table S1 ploidy, zygosity and aneuploidy the paper's life-cycle results turn on). An
isolate's genotype is the same in every one of its records, so it is written ONCE to the
store's ``interned`` env and each record points to it: a 5,000-perturbation genotype is
~0.6 MB, and inlining it in each of ~36,000 growth records would make a 22 GB store whose
readers re-validate the same genotype 43 times.

PHENOTYPE REFERENCE: no S288C control was measured, so each record's reference phenotype
is the POPULATION MEAN over the built isolates for that environment or protocol (the
Caudal 2024 convention), not a measured control.
"""

from __future__ import annotations

import logging
import math
import os
import os.path as osp
import pickle
import re
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar, NamedTuple, cast

import lmdb
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    Organism,
    copy_verified,
    post_process,
    verify_raw_files,
    verify_sha256,
)
from torchcell.data.data import compute_sha256_hash
from torchcell.data.experiment_dataset import canonical_json
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import (
    PBS,
    SPORULATION_KAC,
    YPD_AGAR,
    YPD_LIQUID,
    restated,
)
from torchcell.datamodels.schema import (
    AssayType,
    ComponentDefinition,
    Compound,
    Concentration,
    ConcentrationUnit,
    CultureEnvironment,
    CultureFormat,
    DeletionPerturbation,
    Environment,
    EnvironmentPhysicalPerturbation,
    EnvironmentResponseExperiment,
    EnvironmentResponseExperimentReference,
    EnvironmentResponsePhenotype,
    Experiment,
    ExperimentReference,
    Genotype,
    LifeCycleAssay,
    LifeCycleExperiment,
    LifeCycleExperimentReference,
    LifeCycleMeasurementType,
    LifeCyclePhenotype,
    LifeCycleTimeUnit,
    MeasurementType,
    Media,
    MediaComponent,
    MediaComponentRole,
    NaturalGeneAbsencePerturbation,
    NaturalGenePresencePerturbation,
    NaturalIsolateDescriptor,
    NaturalLossOfFunctionPerturbation,
    PhysicalFactor,
    PreCulture,
    PreCultureSource,
    Publication,
    ReferenceGenome,
    SampleUnit,
    SequenceVariantPerturbation,
    Temperature,
    life_cycle_key,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.literature.manifest import (
    ROLE_SI_DATA,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
    sha256_file,
)
from torchcell.sequence.genome.registry import (
    PETER2018_1011,
    SGD_S288C_R64,
    SGD_S288C_R64_GENBANK_ACCESSION,
    load_genome_manifest,
)
from torchcell.sequence.natural_isolate import (
    GENOTYPE_FILES,
    NaturalIsolateGenotypes,
    genotype_isolate,
    link_genotype_files,
    panel_isolates,
)
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
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Paper, mirror and raw-file pins
# --------------------------------------------------------------------------- #
CITATION_KEY = "dechiaraDomesticationReprogrammedBudding2022"
DOI = "10.1038/s41559-022-01671-9"
TITLE = "Domestication reprogrammed the budding yeast life cycle"
ZOTERO_ITEM_KEY = "N6R6CPDT"
LIBRARY_ID = "6582362"
RAW_MIRROR_REL = f"torchcell-raw/{CITATION_KEY}"

#: The Supplementary Tables workbook (MOESM4): the only file the records are built from.
TABLES_FILENAME = "41559_2022_1671_MOESM4_ESM.xlsx"
TABLES_REL = f"si/{TABLES_FILENAME}"
TABLES_SHA256 = "b2eb2e28cdd2c3037ce14bb1eb8f58ee84609157dd592b3904595b24daf8c99e"
TABLES_URL = (
    "https://static-content.springer.com/esm/art%3A10.1038%2Fs41559-022-01671-9/"
    f"MediaObjects/{TABLES_FILENAME}"
)
#: The same bytes in the literature mirror, where the SI capture wrote them.
TABLES_LIBRARY_REL = "si/si4.xlsx"
TABLES_RETRIEVED_AT = "2026-10-11T10:39:08.479117+00:00"

#: Text artifacts the sourced values quote, all in the literature mirror.
PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "ca68d65691eaff4d6c1b6bf474a4ecef57a5bc41467b51ac993d816cfa033ea1"
REPORTING_TEXT = "si/si2.pdftotext.txt"
REPORTING_TEXT_SHA256 = (
    "ce323f8d2b0c2e59f4fef4551794599b350eb372ef247652aea814fef5075757"
)
PEER_REVIEW_MD = "si/si3.md"
PEER_REVIEW_MD_SHA256 = (
    "6ffb9b0b807b3b04958eee76041148db1281521355b7899aefeb0f50d8ed33a4"
)

ST3_SHEET = "Supplementary Table 3"
ST4_SHEET = "Supplementary Table 4"
ST10_SHEET = "Supplementary Table 10"
#: Row 3 (0-based 2) of each table sheet holds its column header.
HEADER_ROW = 2
ISOLATE_COLUMN = "Standard name"

SPECIES = "Saccharomyces cerevisiae"
GROWTH_TEMPERATURE_C = 30.0
SPORULATION_TEMPERATURE_C = 23.0

#: The workbook plus the six Peter-tier files an isolate genotype is built from.
RAW_FILES = [TABLES_FILENAME, *GENOTYPE_FILES]
#: What the isolate genotypes of this build carry, counted (``natural_isolate``).
GENOTYPE_LEDGER = "natural_isolate_genotype_ledger.json"
#: The classes an isolate genotype holds, built in ``torchcell.sequence.natural_isolate``.
#: Named here because the schema-impact closure of this loader is read from this file's
#: own imports: a change to any of them must re-stale these stores.
GENOTYPE_CLASSES = (
    SequenceVariantPerturbation,
    NaturalGenePresencePerturbation,
    NaturalGeneAbsencePerturbation,
    DeletionPerturbation,
    NaturalLossOfFunctionPerturbation,
    NaturalIsolateDescriptor,
)


# --------------------------------------------------------------------------- #
# Sourced values: each quote is a verbatim substring of its sha256-pinned mirror file
# --------------------------------------------------------------------------- #
def _paper(
    value: Any, quote: str, *, page: str, note: str | None = None
) -> SourcedValue:
    """Bind a value to the mirrored MinerU OCR of the paper (``paper.md``)."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=PAPER_MD,
            citation_key=CITATION_KEY,
            sha256=PAPER_MD_SHA256,
            method="MinerU OCR of the publisher PDF (literature mirror)",
            page=page,
        ),
    )


def _reporting(value: Any, quote: str, *, note: str | None = None) -> SourcedValue:
    """Bind a value to the ``pdftotext -layout`` rendering of the Reporting Summary."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=REPORTING_TEXT,
            citation_key=CITATION_KEY,
            sha256=REPORTING_TEXT_SHA256,
            method="pdftotext -layout of the Reporting Summary (MOESM2), literature "
            "mirror; derived by experiments/036-dataset-fixes-before-kg-build/scripts/"
            "dechiara2022_reporting_summary_text.py",
            page="Life sciences study design",
        ),
    )


def _peer_review(value: Any, quote: str, *, note: str | None = None) -> SourcedValue:
    """Bind a value to the MinerU OCR of the Peer Review File (MOESM3)."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=PEER_REVIEW_MD,
            citation_key=CITATION_KEY,
            sha256=PEER_REVIEW_MD_SHA256,
            method="MinerU OCR of the Peer Review File (literature mirror)",
            page="authors' response to referee 1",
        ),
    )


_METHODS_GROWTH = "Methods, 'Growth phase of the asexual life cycle'"
_METHODS_SPORULATION = (
    "Methods, 'Sexual reproduction measured as sporulation efficiency and spore "
    "viability'"
)
_METHODS_WATER = "Methods, 'Sporulation in plain water'"
_METHODS_CLS = "Methods, 'Chronological life span assay'"
_METHODS_VIABILITY = "Methods, 'Viability in all environments'"

GROWTH_BACKGROUND_MEDIUM = _paper(
    "SC on 2% agar: 0.14% YNB, 0.5% ammonium sulfate, 0.077% CSM, 2% (w/v) glucose, "
    "pH 5.8 with 1% (w/v) succinic acid and 0.6% (w/v) NaOH",
    "plates containing synthetic complete medium ( $0 . 1 4 \\%$ YNB, $0 . 5 \\%$ "
    "ammonium sulfate, $0 . 0 7 7 \\%$ complete supplement mixture (CSM, Formedium), "
    "$2 \\%$ (w/v) glucose and pH buffered to 5.8 with $1 \\%$ (w/v) succinic acid and "
    "$0 . 6 \\%$ (w/v) NaOH)",
    page=_METHODS_GROWTH,
    note="the preculture medium, which the experimental plates repeat ('experimental "
    "plates containing the medium of interest (background medium as above)')",
)
GROWTH_EXPERIMENTAL_PLATES = _paper(
    "experimental plates are the background medium with the variable of interest",
    "Precultures were subsampled with short pins and subsamples transferred as 1,536 "
    "arrays to experimental plates containing the medium of interest (background "
    "medium as above).",
    page=_METHODS_GROWTH,
)
CARBON_REPLACEMENT = _paper(
    "the 2% glucose is replaced by the carbon source at the stated concentration",
    "For alternative carbon sources or concentrations, $2 \\%$ glucose was replaced by "
    "the indicated carbon source at the indicated concentration.",
    page=_METHODS_GROWTH,
    note="the glucose is stated in % (w/v), so the replacing concentration is read on "
    "the same basis; the concentration itself is the column name of Supplementary "
    "Table 3 and the 'Concentration' column of Supplementary Table 4",
)
NITROGEN_REPLACEMENT = _paper(
    "ammonium sulfate and CSM are replaced by the nitrogen source at 30 mg N/L",
    "For alternative nitrogen sources, ammonium sulfate and CSM were replaced by the "
    "alternative nitrogen source at growth yield limiting concentrations $( 3 0 "
    "\\mathrm { m g N l ^ { - 1 } } ,$ ).",
    page=_METHODS_GROWTH,
)
NH4_IS_AMMONIUM_SULFATE = _paper(
    "the NH4 environment's nitrogen source is ammonium sulfate at 30 mg N/L",
    "For alternative nitrogen sources, CSM was excluded and ammonium sulfate limited "
    "to $3 0 \\mathrm { m g N l ^ { - 1 } }$ to prevent nitrogen storage and subsequent "
    "growth on stored nitrogen.",
    page=_METHODS_GROWTH,
    note="the column 'NH4_...' names the ion; the background medium's only ammonium "
    "salt is ammonium sulfate, so the NH4 environment is that salt at the limiting "
    "30 mg N/L every nitrogen environment uses",
)
GROWTH_TEMPERATURE = _paper(
    GROWTH_TEMPERATURE_C,
    "Cultivation using the Scan-o-matic platform82 was performed in thermostatic "
    "cabinets at $3 0 ^ { \\circ } \\mathrm { C }$ and constant, high humidity.",
    page=_METHODS_GROWTH,
)
GROWTH_READOUT = _paper(
    "population doubling time (d) and total cell yield (y) from smoothed growth curves",
    "Population size growth curves were established, smoothed and transformed into "
    "population doubling times (d) and total cell yields (y), as described82.",
    page=_METHODS_GROWTH,
)
GROWTH_REPLICATES = _reporting(
    4,
    "Automatic growth curve were conducted in 4 replicates, median values were taken "
    "for each isolates.",
    note="n_samples = 4 colony growth curves per (isolate, environment); the released "
    "value is their median, and no dispersion is released",
)
ABSOLUTE_VALUES = _peer_review(
    "Supplementary Table 3 carries absolute doubling times (h) and yields (cells)",
    "We now also show and report in table S3 the absolute cell doubling time and cell "
    "yield, in hours and numbers of cells",
)
YPD_RECIPE = _paper(
    "YPD: 2% dextrose, 1% yeast extract, 2% peptone, 2% agar",
    "Isolates were precultivated in YPD $2 \\%$ dextrose, $1 \\%$ yeast extract, $2 "
    "\\%$ peptone, $2 \\%$ agar)",
    page=_METHODS_SPORULATION,
    note="the paper's one statement of its YPD recipe; the 'YPD' growth environment "
    "is read as this medium on the Scan-o-matic agar plates",
)
PRESPORULATION = _paper(
    "48 h at 30 C in YPA (2% potassium acetate, 1% yeast extract, 2% peptone), "
    "inoculated 1:50 from YPD",
    "before being diluted 1:50 into $1 0 \\mathrm { m l }$ of presporulation media "
    "(YPA; $2 \\%$ potassium acetate, $1 \\%$ yeast extract, $2 \\%$ peptone) and grown "
    "for $4 8 \\mathrm { h }$ at $3 0 ^ { \\circ } \\mathrm { C }$ (shaking at $2 5 0 "
    "\\mathrm { r . p . m . }$ ).",
    page=_METHODS_SPORULATION,
)
KAC_CULTURE = _paper(
    "250 ml flask, 5:1 volume/medium (50 ml), OD600 0.6",
    "Presporulated cells were transferred to sporulation media $( 2 \\% \\mathrm { K A "
    "c }$ ) into $2 5 0 \\mathrm { m l }$ flasks at a 5:1 volume/medium ratio and a final "
    "optical density $\\mathrm { ( O D } _ { 6 0 0 } )$ ) of 0.6.",
    page=_METHODS_SPORULATION,
    note="a 250 ml flask at a 5:1 volume/medium ratio holds 50 ml of medium",
)
KAC_TEMPERATURE = _paper(
    SPORULATION_TEMPERATURE_C,
    "Flasks were kept at $2 3 ^ { \\circ } \\mathrm { C }$ and shaken at $2 5 0 "
    "\\mathrm { r . p . m }$ .",
    page=_METHODS_SPORULATION,
)
KAC_COUNTING = _paper(
    ">100 cells per sample counted by optical microscopy at 24 h and 72 h",
    "To estimate sporulation efficiency, ${ \\scriptstyle > } 1 0 0$ cells per sample "
    "were counted at 24 and $7 2 \\mathrm { h }$ post-transfer to sporulation medium "
    "using an optical microscope (Zeiss Axio Lab.A1).",
    page=_METHODS_SPORULATION,
)
SPORULATION_EFFICIENCY = _paper(
    "sporulation efficiency = asci / total cells",
    "Sporulation efficiency was estimated as the number of asci divided by the total "
    "cell count.",
    page=_METHODS_SPORULATION,
)
DYAD_FRACTION = _paper(
    "dyads are released as a percentage of the asci",
    "the percentage of produced asci that are dyads",
    page="Extended Data Fig. 4a caption",
    note="measured on the release: every isolate with a dyad value has asci > 0, and "
    "every isolate with asci > 0 has a dyad value (BFC: 8 dyads of 51% asci gives the "
    "released 15.69)",
)
WATER_PRECULTURE = _paper(
    "24 h at 30 C in 200 ul YPD, 1:100 from an overnight YPD culture, then PBS",
    "Isogenic cultures were grown overnight in liquid YPD in 96-well plates, diluted "
    "1:100 in $2 0 0 \\mu \\mathrm { l }$ of fresh YPD in a 96-well plate and incubated "
    "for $2 4 \\mathrm { h }$ at $3 0 ^ { \\circ } \\mathrm { C }$ A total of $2 0 \\mu "
    "\\mathrm { l }$ of each culture was transferred",
    page=_METHODS_WATER,
)
WATER_RESUSPENSION = _paper(
    "200 ul of PBS per well, day 0 at resuspension",
    "Cells were washed twice with PBS to remove the remaining YPD and finally "
    "resuspended in $2 0 0 \\mu \\mathrm { l }$ of PBS (day 0).",
    page=_METHODS_WATER,
)
WATER_FACS = _paper(
    "10,000 cells per sample, spores told from cells by forward and side scatter",
    "For each sample and time point, we counted 10,000 cells in a FACS-Calibur flow "
    "cytometer (Becton Dickinson) using the forward (FSC; cell size) and side (SSC; "
    "cell granularity) scatter to distinguish two cell populations",
    page=_METHODS_WATER,
)
WATER_TIME = _paper(
    8.0,
    "FACS-based analysis of the sporulation of domesticated and wild yeast after "
    "8\u2009d in water.",
    page="Fig. 3d caption",
    note="the released column 'Sporulation in water' is the data of Fig. 3d and 3e; "
    "the OCR keeps the source's thin space (U+2009) between 8 and d",
)
WATER_CLS_TEMPERATURE = _paper(
    GROWTH_TEMPERATURE_C,
    "Plates were sealed with adhesive aluminium foil and kept at $3 0 ^ { \\circ } "
    "\\mathrm { C }$ .",
    page=_METHODS_CLS,
)
CLS_PRECULTURE = _paper(
    "overnight SDC culture diluted 1:100 into 200 ul SDC or CR",
    "Isogenic cultures were independently grown overnight in liquid SDC $2 \\%$ "
    "dextrose, $0 . 6 7 5 \\%$ yeast nitrogen base (Formedium), $0 . 0 8 8 \\%$ "
    "complete amino acid supplement (Formedium), $\\mathrm { p H }$ buffered to 6.0 "
    "with $2 . 5 \\mathrm { M } \\mathrm { N a O H }$ ), diluted 1:100 in $2 0 0 \\mu "
    "\\mathrm { l }$ of either fresh SDC or calorie-restricted SDC $0 . 5 \\%$ dextrose "
    "instead of $2 \\%$ ) in 96-well plates.",
    page=_METHODS_CLS,
)
CLS_TEMPERATURE = _paper(
    GROWTH_TEMPERATURE_C,
    "Plates were incubated at $3 0 ^ { \\circ } \\mathrm { C }$ and the cells were kept "
    "in saturated media for the whole experiment.",
    page=_METHODS_CLS,
)
CLS_TIME_ORIGIN = _paper(
    "days are counted from saturation, 72 h after inoculation",
    "Aging was considered to start at saturation of the cultures, $7 2 \\mathrm { h }$ "
    "post-inoculation84.",
    page=_METHODS_CLS,
)
CLS_READOUT = _paper(
    "viable cells (%) by PI and YO-PRO-1 exclusion on a flow cytometer",
    "CLS was measured as viable cells $( \\% )$ by flow cytometry on the basis of the "
    "uptake of the fluorescent molecules propidium iodide (PI, Sigma-Aldrich) and "
    "YO-PRO-1 iodide (YP, Invitrogen) as previously described63.",
    page=_METHODS_VIABILITY,
)
CLS_REPLICATES = _paper(
    1,
    "Experiments were run in a single replicate and data were normalized to four "
    "internal controls distributed in each plate. Ten-thousand cells per sample were "
    "analysed.",
    page=_METHODS_VIABILITY,
    note="n_samples = 1 culture per isolate and environment, 10,000 cells per sample; "
    "the plate normalization is why a viability can exceed 100",
)
SPORULATION_REPLICATES_UNSTATED = _reporting(
    "the collection-wide sporulation runs state no replicate count",
    "Other phenotypes were replicated (at",
    note="the sentence continues on the next line, 'least four replicas) to verify "
    "reproducibility before being applied to the whole strain collection.': replicas "
    "were run to validate a protocol before it was applied to the collection, which "
    "does not say how many cultures stand behind a released value",
)
HAPLOID_EXCLUSION = _reporting(
    "haploid isolates are left out of the sporulation statistics, not the release",
    "Haploid isolates were not considered for sporulation statistics given they cannot "
    "sporulate.",
    note="measured: 0 haploid isolates carry a KAc sporulation value; 45 haploid "
    "isolates carry a 'Sporulation in water' value, which is loaded as released",
)
NORMALIZED_BLOCK_REFUSED = _paper(
    "the normalized growth block is log2 relative to the plate controls",
    "To account for spatial variation within and between plates, fixed genotypes (the "
    "diploid hybrid DBVPG6756xYPS128) were included in every fourth of the 1,536 "
    "colony positions on each plate and their $\\log _ { 2 } d$ and y measures were used "
    "to normalize the corresponding $d$ and $y$ measures of adjacent colonies.",
    page=_METHODS_GROWTH,
    note="refused: its row names disagree with the absolute block on 519 of 1,011 "
    "rows, and genotype says the absolute block is the correctly named one",
)

#: Every sourced value of the module, so a verifier can audit each against the mirror.
SOURCED_VALUES: tuple[SourcedValue, ...] = (
    GROWTH_BACKGROUND_MEDIUM,
    GROWTH_EXPERIMENTAL_PLATES,
    CARBON_REPLACEMENT,
    NITROGEN_REPLACEMENT,
    NH4_IS_AMMONIUM_SULFATE,
    GROWTH_TEMPERATURE,
    GROWTH_READOUT,
    GROWTH_REPLICATES,
    ABSOLUTE_VALUES,
    YPD_RECIPE,
    PRESPORULATION,
    KAC_CULTURE,
    KAC_TEMPERATURE,
    KAC_COUNTING,
    SPORULATION_EFFICIENCY,
    DYAD_FRACTION,
    WATER_PRECULTURE,
    WATER_RESUSPENSION,
    WATER_FACS,
    WATER_TIME,
    WATER_CLS_TEMPERATURE,
    CLS_PRECULTURE,
    CLS_TEMPERATURE,
    CLS_TIME_ORIGIN,
    CLS_READOUT,
    CLS_REPLICATES,
    SPORULATION_REPLICATES_UNSTATED,
    HAPLOID_EXCLUSION,
    NORMALIZED_BLOCK_REFUSED,
)


# --------------------------------------------------------------------------- #
# Media
# --------------------------------------------------------------------------- #
_PCT = ConcentrationUnit.percent_w_v


#: Recorded on every ingredient whose percentage the source writes with no basis.
_PERCENT_BASIS_NOTE = (
    "the source writes '%' with no w/v or v/v basis; recorded as w/v, the basis it "
    "states for the glucose, succinic acid and NaOH of the same recipe"
)


def _component(
    name: str,
    role: MediaComponentRole,
    percent: float,
    sourced: SourcedValue,
    *,
    mixture: bool = False,
    basis_stated: bool = False,
) -> MediaComponent:
    """One stated ingredient: a resolved compound, or a deferred-composition mixture."""
    note = None if basis_stated else _PERCENT_BASIS_NOTE
    if mixture:
        return MediaComponent(
            compound=Compound(name=name),
            role=role,
            concentration=Concentration(value=percent, unit=_PCT),
            definition=ComponentDefinition.composition_deferred,
            provenance=[sourced],
            note=note,
        )
    return MediaComponent(
        compound=resolved_compound(name),
        role=role,
        concentration=Concentration(value=percent, unit=_PCT),
        provenance=[sourced],
        note=note,
    )


def _growth_buffer_and_agar() -> list[MediaComponent]:
    """The succinate buffer and the 2% agar every growth plate shares."""
    return [
        _component(
            "succinic acid",
            MediaComponentRole.buffer,
            1.0,
            GROWTH_BACKGROUND_MEDIUM,
            basis_stated=True,
        ),
        _component(
            "sodium hydroxide",
            MediaComponentRole.buffer,
            0.6,
            GROWTH_BACKGROUND_MEDIUM,
            basis_stated=True,
        ),
        _component(
            "agar", MediaComponentRole.gelling_agent, 2.0, GROWTH_BACKGROUND_MEDIUM
        ),
    ]


#: The carbon-source plates: the background medium without its glucose. The carbon
#: source is each environment's ``EnvironmentPhysicalPerturbation(carbon_source)``.
GROWTH_CARBON_BASE = Media(
    name="De Chiara 2022 synthetic complete agar, carbon source varied",
    state="solid",
    is_synthetic=True,
    base_medium="SC",
    components=[
        _component(
            "yeast nitrogen base",
            MediaComponentRole.other,
            0.14,
            GROWTH_BACKGROUND_MEDIUM,
            mixture=True,
        ),
        _component(
            "ammonium sulfate",
            MediaComponentRole.nitrogen_source,
            0.5,
            GROWTH_BACKGROUND_MEDIUM,
        ),
        _component(
            "complete supplement mixture (CSM, Formedium)",
            MediaComponentRole.amino_acid,
            0.077,
            GROWTH_BACKGROUND_MEDIUM,
            mixture=True,
        ),
        *_growth_buffer_and_agar(),
    ],
    provenance=[
        GROWTH_BACKGROUND_MEDIUM,
        GROWTH_EXPERIMENTAL_PLATES,
        CARBON_REPLACEMENT,
    ],
)

#: The nitrogen-source plates: the background medium without ammonium sulfate or CSM.
#: The nitrogen source is each environment's ``EnvironmentPhysicalPerturbation``.
GROWTH_NITROGEN_BASE = Media(
    name="De Chiara 2022 YNB glucose agar, nitrogen source varied",
    state="solid",
    is_synthetic=True,
    base_medium="YNB",
    components=[
        _component(
            "yeast nitrogen base",
            MediaComponentRole.other,
            0.14,
            GROWTH_BACKGROUND_MEDIUM,
            mixture=True,
        ),
        _component(
            "D-glucose",
            MediaComponentRole.carbon_source,
            2.0,
            GROWTH_BACKGROUND_MEDIUM,
            basis_stated=True,
        ),
        *_growth_buffer_and_agar(),
    ],
    provenance=[
        GROWTH_BACKGROUND_MEDIUM,
        GROWTH_EXPERIMENTAL_PLATES,
        NITROGEN_REPLACEMENT,
    ],
)

#: The 'YPD' growth environment: the library's solid YPD with De Chiara's own recipe.
GROWTH_YPD = restated(YPD_AGAR, YPD_RECIPE)

#: The KAc protocol's pre-culture medium: YP plus 2% potassium acetate.
YPA_LIQUID = Media(
    name="YPA (yeast extract / peptone / potassium acetate), liquid",
    state="liquid",
    is_synthetic=False,
    base_medium="YP",
    components=[
        _component(
            "yeast extract",
            MediaComponentRole.complex_ingredient,
            1.0,
            PRESPORULATION,
            mixture=True,
        ),
        _component(
            "peptone",
            MediaComponentRole.complex_ingredient,
            2.0,
            PRESPORULATION,
            mixture=True,
        ),
        _component(
            "potassium acetate", MediaComponentRole.carbon_source, 2.0, PRESPORULATION
        ),
    ],
    provenance=[PRESPORULATION],
)


def _sdc(name: str, dextrose_percent: float) -> Media:
    """SDC (2% dextrose) or calorie-restricted SDC (0.5%), liquid, pH 6.0 with NaOH."""
    return Media(
        name=name,
        state="liquid",
        is_synthetic=True,
        base_medium="SC",
        components=[
            _component(
                "D-glucose",
                MediaComponentRole.carbon_source,
                dextrose_percent,
                CLS_PRECULTURE,
            ),
            _component(
                "yeast nitrogen base (Formedium)",
                MediaComponentRole.other,
                0.675,
                CLS_PRECULTURE,
                mixture=True,
            ),
            _component(
                "complete amino acid supplement (Formedium)",
                MediaComponentRole.amino_acid,
                0.088,
                CLS_PRECULTURE,
                mixture=True,
            ),
            MediaComponent(
                compound=resolved_compound("sodium hydroxide"),
                role=MediaComponentRole.buffer,
                concentration=None,
                provenance=[CLS_PRECULTURE],
                note="added to pH 6.0 from a 2.5 M stock; the amount is not stated",
            ),
        ],
        provenance=[CLS_PRECULTURE],
    )


SDC = _sdc("SDC (synthetic defined complete, 2% dextrose, pH 6.0)", 2.0)
SDC_CALORIE_RESTRICTED = _sdc(
    "calorie-restricted SDC (0.5% dextrose instead of 2%, pH 6.0)", 0.5
)

DECHIARA_SPORULATION_KAC = restated(SPORULATION_KAC, KAC_CULTURE)
DECHIARA_PBS = restated(PBS, WATER_RESUSPENSION)
DECHIARA_YPD_LIQUID = restated(YPD_LIQUID, WATER_PRECULTURE)


# --------------------------------------------------------------------------- #
# Growth environments: Supplementary Table 3's 43 column stems
# --------------------------------------------------------------------------- #
class NutrientKind(StrEnum):
    """Which nutrient a growth environment varies (Supplementary Table 4 'Element')."""

    carbon = "Carbon"
    nitrogen = "Nitrogen"
    rich = "YPD"


class GrowthEnvironment(BaseModel):
    """One Supplementary Table 3 growth environment, parsed from its column stem.

    ``stem`` is the column name minus its trait suffix (``Glucose_0.5``, ``Malic_Acid``,
    ``NH4``, ``YPD``); ``source`` and ``concentration_percent`` are the stem's two parts,
    and ``kind`` is Supplementary Table 4's 'Element' for the stem (YPD is not in that
    table: it is the rich-medium environment).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    stem: str
    source: str
    concentration_percent: float | None
    kind: NutrientKind


#: The compound label a stem's source is resolved under. The paper's own word, lowered;
#: no stereochemistry is added that the source does not state, so an unresolved label
#: carries the resolver's typed gap rather than an assumed enantiomer.
SOURCE_COMPOUND_LABEL: dict[str, str] = {
    "Adenine": "adenine",
    "Alanine": "alanine",
    "Arabinose": "arabinose",
    "Citrulline": "citrulline",
    "Fructose": "fructose",
    "Glucose": "glucose",
    "Glycerol": "glycerol",
    "Glycine": "glycine",
    "Isoleucine": "isoleucine",
    "Malic_Acid": "malic acid",
    "Maltose": "maltose",
    "Mannitol": "mannitol",
    "Methionine": "methionine",
    "NH4": "ammonium sulfate",
    "Phenylalanine": "phenylalanine",
    "Raffinose": "raffinose",
    "Serine": "serine",
    "Sucrose": "sucrose",
    "Threonine": "threonine",
    "Tryptophan": "tryptophan",
    "Valine": "valine",
    "Xylose": "xylose",
}

_STEM_RE = re.compile(r"^(?P<source>[A-Za-z0-9_]+?)(?:_(?P<conc>\d+(?:\.\d+)?))?$")


def parse_growth_stem(
    stem: str, kinds: Mapping[str, NutrientKind]
) -> GrowthEnvironment:
    """Split a column stem into source and concentration; ``kinds`` keys the source.

    ``kinds`` maps a Supplementary Table 4 'Source' to its 'Element'; ``YPD`` is the rich
    medium. A stem whose source is in neither raises.
    """
    if stem == "YPD":
        return GrowthEnvironment(
            stem=stem, source="YPD", concentration_percent=None, kind=NutrientKind.rich
        )
    match = _STEM_RE.match(stem)
    if match is None or match.group("source") not in kinds:
        raise ValueError(
            f"growth column stem {stem!r} names no Supplementary Table 4 source"
        )
    conc = match.group("conc")
    return GrowthEnvironment(
        stem=stem,
        source=match.group("source"),
        concentration_percent=None if conc is None else float(conc),
        kind=kinds[match.group("source")],
    )


def _nitrogen_dose() -> Concentration:
    """30 mg of nitrogen per litre, the dose every nitrogen environment uses."""
    return Concentration(value=30.0, unit=ConcentrationUnit.mg_nitrogen_per_l)


def growth_environment(env: GrowthEnvironment) -> Environment:
    """The ``Environment`` of one growth column: base plate plus the varied nutrient."""
    temperature = Temperature(value=GROWTH_TEMPERATURE_C)
    if env.kind is NutrientKind.rich:
        return Environment(media=GROWTH_YPD, temperature=temperature)
    agent = resolved_compound(SOURCE_COMPOUND_LABEL[env.source])
    if env.kind is NutrientKind.carbon:
        if env.concentration_percent is None:
            perturbation = EnvironmentPhysicalPerturbation(
                factor=PhysicalFactor.carbon_source,
                magnitude=None,
                agent=agent,
                provenance_gaps=[
                    ProvenanceGap(
                        field="magnitude",
                        reason=ProvenanceGapReason.not_reported_by_primary,
                        note=f"Supplementary Table 3 names the {env.source} column "
                        "without a concentration and Supplementary Table 4 leaves its "
                        "'Concentration' cell empty; no other mirrored text states it",
                    )
                ],
            )
        else:
            perturbation = EnvironmentPhysicalPerturbation(
                factor=PhysicalFactor.carbon_source,
                magnitude=Concentration(value=env.concentration_percent, unit=_PCT),
                agent=agent,
            )
        return Environment(
            media=GROWTH_CARBON_BASE,
            temperature=temperature,
            perturbations=[perturbation],
        )
    return Environment(
        media=GROWTH_NITROGEN_BASE,
        temperature=temperature,
        perturbations=[
            EnvironmentPhysicalPerturbation(
                factor=PhysicalFactor.nitrogen_source,
                magnitude=_nitrogen_dose(),
                agent=agent,
            )
        ],
    )


# --------------------------------------------------------------------------- #
# Life-cycle protocols: one per (assay, environment) the release measured
# --------------------------------------------------------------------------- #
class LifeCycleProtocol(StrEnum):
    """The five culture protocols behind Supplementary Table 3's 13 life-cycle columns."""

    kac_sporulation = "kac_sporulation"
    water_sporulation = "water_sporulation"
    water_viability = "water_viability"
    sdc_viability = "sdc_viability"
    calorie_restricted_viability = "calorie_restricted_viability"


class LifeCycleColumn(BaseModel):
    """One released life-cycle column and the time point it is."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    column: str
    protocol: LifeCycleProtocol
    measurement_type: LifeCycleMeasurementType
    time: float
    unit: LifeCycleTimeUnit

    @property
    def key(self) -> str:
        """The ``LifeCyclePhenotype.life_cycle`` key this column fills."""
        return life_cycle_key(self.measurement_type, self.time, self.unit)


_SPO = LifeCycleMeasurementType.sporulation_efficiency
_DYAD = LifeCycleMeasurementType.dyad_fraction
_VIAB = LifeCycleMeasurementType.stationary_phase_viability
_H = LifeCycleTimeUnit.hour
_D = LifeCycleTimeUnit.day
_P = LifeCycleProtocol

LIFE_CYCLE_COLUMNS: tuple[LifeCycleColumn, ...] = (
    LifeCycleColumn(
        column="Asci at 24h",
        protocol=_P.kac_sporulation,
        measurement_type=_SPO,
        time=24,
        unit=_H,
    ),
    LifeCycleColumn(
        column="Dyads at 24h",
        protocol=_P.kac_sporulation,
        measurement_type=_DYAD,
        time=24,
        unit=_H,
    ),
    LifeCycleColumn(
        column="Asci at 72h",
        protocol=_P.kac_sporulation,
        measurement_type=_SPO,
        time=72,
        unit=_H,
    ),
    LifeCycleColumn(
        column="Dyads at 72h",
        protocol=_P.kac_sporulation,
        measurement_type=_DYAD,
        time=72,
        unit=_H,
    ),
    LifeCycleColumn(
        column="Sporulation in water",
        protocol=_P.water_sporulation,
        measurement_type=_SPO,
        time=8,
        unit=_D,
    ),
    LifeCycleColumn(
        column="CLS in water (day 8)",
        protocol=_P.water_viability,
        measurement_type=_VIAB,
        time=8,
        unit=_D,
    ),
    LifeCycleColumn(
        column="CLS in water (day 20)",
        protocol=_P.water_viability,
        measurement_type=_VIAB,
        time=20,
        unit=_D,
    ),
    LifeCycleColumn(
        column="CLS in SDC (day 7)",
        protocol=_P.sdc_viability,
        measurement_type=_VIAB,
        time=7,
        unit=_D,
    ),
    LifeCycleColumn(
        column="CLS in SDC (day 21)",
        protocol=_P.sdc_viability,
        measurement_type=_VIAB,
        time=21,
        unit=_D,
    ),
    LifeCycleColumn(
        column="CLS in SDC (day 35)",
        protocol=_P.sdc_viability,
        measurement_type=_VIAB,
        time=35,
        unit=_D,
    ),
    LifeCycleColumn(
        column="CLS in crSDC (day 7)",
        protocol=_P.calorie_restricted_viability,
        measurement_type=_VIAB,
        time=7,
        unit=_D,
    ),
    LifeCycleColumn(
        column="CLS in crSDC (day 21)",
        protocol=_P.calorie_restricted_viability,
        measurement_type=_VIAB,
        time=21,
        unit=_D,
    ),
    LifeCycleColumn(
        column="CLS in crSDC (day 35)",
        protocol=_P.calorie_restricted_viability,
        measurement_type=_VIAB,
        time=35,
        unit=_D,
    ),
)

#: Supplementary Table 10's spelling of each life-cycle column, for the count check.
ST10_LIFE_CYCLE_NAME: dict[str, str] = {
    "Sporulation in water": "Sporulation_in_water",
    "Asci at 24h": "Asci_at_24h",
    "Dyads at 24h": "Dyads_at_24h",
    "Asci at 72h": "Asci_at_72h",
    "Dyads at 72h": "Dyads_at_72h",
    "CLS in water (day 8)": "CLS_in_water_.day_8.",
    "CLS in water (day 20)": "CLS_in_water_.day_20.",
    "CLS in SDC (day 7)": "cls_d7_sdc",
    "CLS in SDC (day 21)": "cls_d21_sdc",
    "CLS in SDC (day 35)": "cls_d35_sdc",
    "CLS in crSDC (day 7)": "cls_d7_cr",
    "CLS in crSDC (day 21)": "cls_d21_cr",
    "CLS in crSDC (day 35)": "cls_d35_cr",
}


def _not_stated(field: str, sourced: SourcedValue) -> ProvenanceGap:
    """A field the primary does not state, anchored to the passage that was read."""
    return ProvenanceGap(
        field=field,
        reason=ProvenanceGapReason.not_reported_by_primary,
        looked_in=sourced.provenance,
        note=sourced.note,
    )


def protocol_environment(protocol: LifeCycleProtocol) -> CultureEnvironment:
    """The culture environment one life-cycle protocol ran in."""
    plate = CultureFormat(
        vessel="96-well plate", working_volume_ul=200.0, provenance=[WATER_RESUSPENSION]
    )
    water_preculture = PreCulture(
        source=PreCultureSource.timed_culture,
        medium=DECHIARA_YPD_LIQUID,
        duration_hours=24.0,
        source_label="incubated for 24 h at 30 C",
        provenance=[WATER_PRECULTURE],
    )
    if protocol is LifeCycleProtocol.kac_sporulation:
        return CultureEnvironment(
            media=DECHIARA_SPORULATION_KAC,
            temperature=Temperature(value=SPORULATION_TEMPERATURE_C),
            culture_format=CultureFormat(
                vessel="250 ml flask",
                working_volume_ul=50_000.0,
                shaking_rpm=250.0,
                inoculum_od600=0.6,
                provenance=[KAC_CULTURE, KAC_TEMPERATURE],
            ),
            pre_culture=PreCulture(
                source=PreCultureSource.timed_culture,
                medium=YPA_LIQUID,
                duration_hours=48.0,
                source_label="presporulation media (YPA), 48 h at 30 C",
                provenance=[PRESPORULATION],
            ),
        )
    if protocol is LifeCycleProtocol.water_sporulation:
        return CultureEnvironment(
            media=DECHIARA_PBS,
            temperature=None,
            culture_format=plate,
            pre_culture=water_preculture,
            provenance_gaps=[
                ProvenanceGap(
                    field="temperature",
                    reason=ProvenanceGapReason.not_reported_by_primary,
                    looked_in=WATER_PRECULTURE.provenance,
                    note="'Sporulation in plain water' states 30 C for the YPD step "
                    "and no temperature for the PBS incubation; the CLS-in-water "
                    "paragraph states 30 C for its own PBS plates but does not say "
                    "they are the same plates",
                )
            ],
        )
    if protocol is LifeCycleProtocol.water_viability:
        return CultureEnvironment(
            media=DECHIARA_PBS,
            temperature=Temperature(value=GROWTH_TEMPERATURE_C),
            culture_format=plate,
            pre_culture=water_preculture,
        )
    medium = (
        SDC if protocol is LifeCycleProtocol.sdc_viability else SDC_CALORIE_RESTRICTED
    )
    return CultureEnvironment(
        media=medium,
        temperature=Temperature(value=GROWTH_TEMPERATURE_C),
        culture_format=CultureFormat(
            vessel="96-well plate", working_volume_ul=200.0, provenance=[CLS_PRECULTURE]
        ),
        pre_culture=PreCulture(
            source=PreCultureSource.overnight_culture,
            medium=SDC,
            provenance=[CLS_PRECULTURE],
        ),
    )


def protocol_phenotype(
    protocol: LifeCycleProtocol, life_cycle: Mapping[str, float]
) -> LifeCyclePhenotype:
    """A life-cycle phenotype with the protocol's assay and sampling design."""
    if protocol is LifeCycleProtocol.kac_sporulation:
        return LifeCyclePhenotype(
            life_cycle=dict(life_cycle),
            assay=LifeCycleAssay.microscopy_ascus_count,
            cells_per_sample=100,
            cells_per_sample_is_minimum=True,
            provenance_gaps=[
                _not_stated("n_samples", SPORULATION_REPLICATES_UNSTATED),
                _not_stated("sample_unit", SPORULATION_REPLICATES_UNSTATED),
            ],
        )
    if protocol is LifeCycleProtocol.water_sporulation:
        return LifeCyclePhenotype(
            life_cycle=dict(life_cycle),
            assay=LifeCycleAssay.flow_cytometry_scatter,
            cells_per_sample=10_000,
            provenance_gaps=[
                _not_stated("n_samples", SPORULATION_REPLICATES_UNSTATED),
                _not_stated("sample_unit", SPORULATION_REPLICATES_UNSTATED),
            ],
        )
    return LifeCyclePhenotype(
        life_cycle=dict(life_cycle),
        assay=LifeCycleAssay.flow_cytometry_viability_stain,
        n_samples=1,
        sample_unit=SampleUnit.biological_replicate,
        cells_per_sample=10_000,
    )


# --------------------------------------------------------------------------- #
# Release reader
# --------------------------------------------------------------------------- #
class GrowthTrait(StrEnum):
    """The two growth traits, by the suffix of their ABSOLUTE column."""

    doubling_time = "doublingtime_hours"
    cell_yield = "yield_cellscount"


#: The suffix of the refused, control-normalized column of each trait.
NORMALIZED_SUFFIX: dict[GrowthTrait, str] = {
    GrowthTrait.doubling_time: "doublingtime",
    GrowthTrait.cell_yield: "yield",
}


class Release(BaseModel):
    """The parsed Supplementary Tables workbook: what the loaders read, nothing more."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    st3: pd.DataFrame
    environments: tuple[GrowthEnvironment, ...]
    st10_counts: dict[str, int]


def read_release(path: str) -> Release:
    """Parse Supplementary Tables 3, 4 and 10 and check their mutual invariants.

    Raises when the 43 growth stems are not exactly Supplementary Table 4's 42
    (source, concentration) rows plus YPD, when a column's non-missing count differs
    from Supplementary Table 10, or when the dyad columns break the measured rule that
    a dyad value is present exactly where the asci value is above 0.
    """
    st3 = pd.read_excel(path, sheet_name=ST3_SHEET, header=HEADER_ROW, na_values=["NA"])
    if not st3[ISOLATE_COLUMN].is_unique:
        raise ValueError("Supplementary Table 3 repeats an isolate name")
    st4 = pd.read_excel(path, sheet_name=ST4_SHEET, header=HEADER_ROW)
    kinds = {
        str(s): NutrientKind(str(e)) for s, e in zip(st4["Source"], st4["Element"])
    }
    st4_pairs = {
        (str(s), None if pd.isna(c) else float(c))
        for s, c in zip(st4["Source"], st4["Concentration"])
    }
    stems = [
        c[: -len(f"_{GrowthTrait.doubling_time.value}")]
        for c in st3.columns
        if str(c).endswith(f"_{GrowthTrait.doubling_time.value}")
    ]
    environments = tuple(parse_growth_stem(stem, kinds) for stem in stems)
    parsed_pairs = {
        (e.source, e.concentration_percent)
        for e in environments
        if e.kind is not NutrientKind.rich
    }
    if parsed_pairs != st4_pairs:
        raise ValueError(
            "growth stems disagree with Supplementary Table 4: "
            f"{sorted(map(str, parsed_pairs ^ st4_pairs))}"
        )
    st10 = pd.read_excel(path, sheet_name=ST10_SHEET, header=HEADER_ROW).dropna(
        how="all"
    )
    st10_counts = {
        str(p): int(n) for p, n in zip(st10["Phenotype"], st10["Total n. of isolates"])
    }
    for column in LIFE_CYCLE_COLUMNS:
        observed = int(st3[column.column].notna().sum())
        if observed != st10_counts[ST10_LIFE_CYCLE_NAME[column.column]]:
            raise ValueError(
                f"{column.column}: {observed} values, Table 10 says otherwise"
            )
    for env in environments:
        for trait in GrowthTrait:
            name = f"{env.stem}_{trait.value}"
            if int(st3[name].notna().sum()) != st10_counts[name]:
                raise ValueError(f"{name}: count differs from Supplementary Table 10")
    for hours in ("24h", "72h"):
        asci = st3[f"Asci at {hours}"]
        dyads = st3[f"Dyads at {hours}"]
        if not (dyads.notna() == (asci > 0)).all():
            raise ValueError(f"Dyads at {hours} is not present exactly where asci > 0")
    return Release(st3=st3, environments=environments, st10_counts=st10_counts)


def growth_values(
    release: Release, trait: GrowthTrait, isolates: Sequence[str]
) -> dict[tuple[str, str], float]:
    """``(isolate, stem) -> released absolute value`` for every non-missing cell."""
    table = release.st3.set_index(ISOLATE_COLUMN)
    out: dict[tuple[str, str], float] = {}
    for isolate in isolates:
        row = table.loc[isolate]
        for env in release.environments:
            value = row[f"{env.stem}_{trait.value}"]
            if not pd.isna(value):
                out[(isolate, env.stem)] = float(value)
    return out


def life_cycle_values(
    release: Release, isolates: Sequence[str]
) -> dict[tuple[str, LifeCycleProtocol], dict[str, float]]:
    """``(isolate, protocol) -> {life_cycle key: released percent}``, non-empty only."""
    table = release.st3.set_index(ISOLATE_COLUMN)
    out: dict[tuple[str, LifeCycleProtocol], dict[str, float]] = {}
    for isolate in isolates:
        row = table.loc[isolate]
        for column in LIFE_CYCLE_COLUMNS:
            value = row[column.column]
            if not pd.isna(value):
                out.setdefault((isolate, column.protocol), {})[column.key] = float(
                    value
                )
    return out


# --------------------------------------------------------------------------- #
# Genotype: interned once per isolate (built by ``torchcell.sequence.natural_isolate``)
# --------------------------------------------------------------------------- #
class InternedGenotype(NamedTuple):
    """One isolate's genotype with the pointer and payload its records share."""

    genotype: Genotype
    pointer: dict[str, str]
    payload: bytes


def intern_genotype(genotype: Genotype, isolate: str) -> InternedGenotype:
    """The ``{"$ref", "name"}`` pointer and pickled dump of one isolate's genotype.

    The ref is the sha256 of the genotype's canonical JSON, the content hash
    ``ExperimentDataset._maybe_intern`` gives every interned constant.
    """
    digest = compute_sha256_hash(canonical_json(genotype))
    return InternedGenotype(
        genotype=genotype,
        pointer={"$ref": digest, "name": f"genotype {isolate}"},
        payload=pickle.dumps(genotype.model_dump()),
    )


# --------------------------------------------------------------------------- #
# Ledger of what the build kept and refused
# --------------------------------------------------------------------------- #
class RefusedBlock(BaseModel):
    """A released block of values the loaders do not serve, with the measured reason."""

    block: str
    n_cells: int
    reason: str


class ReleaseLedger(BaseModel):
    """Every isolate and every released cell of one store, kept or refused."""

    dataset: str
    n_isolates_in_release: int
    n_isolates_without_peter_genome: int
    isolates_without_peter_genome: list[str]
    n_isolates_built: int
    n_released_cells: int = Field(description="isolates x columns of the store's block")
    n_missing_cells: int = Field(description="cells the release leaves 'NA'")
    n_refused_cells: int = Field(
        description="non-missing cells of isolates without a Peter genome"
    )
    n_kept_cells: int = Field(description="cells stored in a record")
    n_records: int
    refused_blocks: list[RefusedBlock] = Field(default_factory=list)


def _publication() -> Publication:
    return Publication(doi=DOI, doi_url=f"https://doi.org/{DOI}")


def _genome_reference() -> ReferenceGenome:
    return ReferenceGenome(
        species=SPECIES,
        strain="S288C",
        assembly_set=SGD_S288C_R64,
        assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
    )


# --------------------------------------------------------------------------- #
# Raw mirror deposit
# --------------------------------------------------------------------------- #
def raw_mirror_dir(data_root: str | None = None) -> str:
    """``$DATA_ROOT/torchcell-raw/dechiaraDomesticationReprogrammedBudding2022``."""
    return osp.join(data_root or os.environ["DATA_ROOT"], RAW_MIRROR_REL)


def deposit_raw_mirror(*, data_root: str | None = None) -> str:
    """Copy the consumed workbook from the literature mirror into the raw mirror.

    The SI capture wrote MOESM4 to ``torchcell-library/<key>/si/si4.xlsx`` with a
    ``springer_esm`` retrieval record; the raw mirror keeps the same bytes under the
    publisher's filename with that retrieval restated. Idempotent by sha256; a different
    file at the destination is refused.
    """
    root = Path(data_root or os.environ["DATA_ROOT"])
    source = root / "torchcell-library" / CITATION_KEY / TABLES_LIBRARY_REL
    verify_sha256(source, TABLES_SHA256)
    dest_dir = Path(raw_mirror_dir(str(root)))
    dest = dest_dir / TABLES_REL
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if sha256_file(dest) != TABLES_SHA256:
            raise RuntimeError(f"{dest} exists with a different sha256; refusing")
    else:
        copy_verified(source, dest, TABLES_SHA256)
    record = ArtifactRecord(
        path=TABLES_REL,
        role=ROLE_SI_DATA,
        bytes=dest.stat().st_size,
        sha256=TABLES_SHA256,
        source=TABLES_URL,
        original_filename=TABLES_FILENAME,
        retrieval=RetrievalRecord(
            method=RetrievalMethod.springer_esm,
            source_url=TABLES_URL,
            retriever="torchcell.literature.retrieve.springer_esm",
            params={"url": TABLES_URL},
            sha256=TABLES_SHA256,
            retrieved_at=TABLES_RETRIEVED_AT,
        ),
    )
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=DOI,
        title=TITLE,
        library_id=LIBRARY_ID,
        zotero_item_key=ZOTERO_ITEM_KEY,
        files=[record],
        si_data_sources=[TABLES_URL],
        si_expected=["Supplementary Tables 1-13 (MOESM4)"],
        provenance_complete=True,
        created_at=datetime.now(UTC).isoformat(),
    )
    (dest_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    return str(dest_dir)


# --------------------------------------------------------------------------- #
# Dataset classes
# --------------------------------------------------------------------------- #
class _DeChiara2022Dataset(ExperimentDataset):
    """Shared download, genotype and interned-write machinery of the three stores."""

    def __init__(
        self,
        root: str,
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; records are built by ``process`` on first use."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def raw_file_names(self) -> list[str]:
        """The workbook and the six Peter 2018 genome-tier files."""
        return RAW_FILES

    @staticmethod
    def _data_root() -> str:
        from dotenv import load_dotenv

        load_dotenv()
        return os.environ["DATA_ROOT"]

    def download(self) -> None:
        """Copy the workbook from the raw mirror; link the Peter files from the tier."""
        data_root = self._data_root()
        os.makedirs(self.raw_dir, exist_ok=True)
        workbook = osp.join(self.raw_dir, TABLES_FILENAME)
        if not osp.exists(workbook):
            copy_verified(
                osp.join(raw_mirror_dir(data_root), TABLES_REL), workbook, TABLES_SHA256
            )
        link_genotype_files(self.raw_dir, data_root)

    def _verify_raw(self, data_root: str) -> None:
        peter = load_genome_manifest(PETER2018_1011, data_root)
        verify_raw_files(
            self.raw_dir,
            {
                TABLES_FILENAME: TABLES_SHA256,
                **{name: peter.record(name).sha256 for name in GENOTYPE_FILES},
            },
        )

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Preprocessing is handled inside ``process``."""
        return df

    def create_experiment(self) -> None:
        """Records are built per isolate inside ``process``."""
        raise NotImplementedError("De Chiara 2022 records are built in process()")

    def _prepare(self) -> tuple[Release, NaturalIsolateGenotypes, list[str], list[str]]:
        """Verify raw files, read the release, split isolates by Peter-genome status."""
        data_root = self._data_root()
        self._verify_raw(data_root)
        os.makedirs(self.preprocess_dir, exist_ok=True)
        release = read_release(osp.join(self.raw_dir, TABLES_FILENAME))
        panel = panel_isolates(self.raw_dir)
        names = release.st3[ISOLATE_COLUMN].astype(str).tolist()
        built = [n for n in names if n in panel]
        without = [n for n in names if n not in panel]
        genotypes = NaturalIsolateGenotypes(
            self.raw_dir, self.preprocess_dir, built, data_root
        )
        genotypes.write_ledger(osp.join(self.preprocess_dir, GENOTYPE_LEDGER))
        return release, genotypes, built, without

    def _write(
        self,
        records: Iterator[tuple[Experiment, ExperimentReference, InternedGenotype]],
        batch: int = 500,
    ) -> int:
        """Write records, interning environment, reference, publication and genotype.

        Commits every ``batch`` records, the interned txn first (the context managers
        exit last-in first-out) so a crash never leaves a record pointing at a missing
        constant.
        """
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        publication = _publication()
        n = 0
        pending: list[tuple[Experiment, ExperimentReference, InternedGenotype]] = []

        def flush() -> None:
            nonlocal n
            with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
                for experiment, reference, genotype in pending:
                    txn.put(
                        f"{n}".encode(),
                        self._serialize(
                            experiment, reference, publication, genotype, itxn
                        ),
                    )
                    n += 1
            pending.clear()

        for item in records:
            pending.append(item)
            if len(pending) >= batch:
                flush()
        if pending:
            flush()
        env.close()
        interned_env.close()
        return n

    def _serialize(
        self,
        experiment: Experiment,
        reference: ExperimentReference,
        publication: Publication,
        genotype: InternedGenotype,
        itxn: Any,
    ) -> bytes:
        """The record dict with the genotype as an interned pointer, constants interned.

        Same layout and same content hashes as ``ExperimentDataset._intern_record``; the
        genotype is the one addition, since a natural isolate's genotype is a constant
        of all its records.
        """
        dump = experiment.model_dump(exclude={"genotype"})
        exp_dict = {
            name: (dict(genotype.pointer) if name == "genotype" else dump[name])
            for name in type(experiment).model_fields
        }
        key = genotype.pointer["$ref"].encode()
        if itxn.get(key) is None:
            itxn.put(key, genotype.payload)
        rec: dict[str, Any] = {
            "experiment": exp_dict,
            "reference": reference.model_dump(),
            "publication": publication.model_dump(),
        }
        self._maybe_intern(
            rec["experiment"],
            "environment",
            experiment.environment,
            experiment.environment.media.name,
            itxn,
        )
        self._maybe_intern(rec, "reference", reference, reference.dataset_name, itxn)
        self._maybe_intern(rec, "publication", publication, DOI, itxn)
        return pickle.dumps(rec)

    def _write_ledger(self, ledger: ReleaseLedger) -> None:
        with open(osp.join(self.preprocess_dir, "release_ledger.json"), "w") as handle:
            handle.write(ledger.model_dump_json(indent=2))


#: The refused normalized block, as the ledger states it.
_NORMALIZED_REFUSAL = (
    "control-normalized log2 values: the row names disagree with the absolute block on "
    "519 of 1,011 rows (shifted by one name over BDM..CRL), and Peter 2018's SUC2 and "
    "MAL-locus presence keeps the sugar associations only in the absolute block; where "
    "the blocks agree the normalized value is the absolute one less a per-environment "
    "constant (experiments/036-dataset-fixes-before-kg-build/results/"
    "dechiara2022_release_inventory.json)"
)


class _GrowthDeChiara2022Dataset(_DeChiara2022Dataset):
    """One record per (isolate, growth environment) of one absolute growth trait."""

    trait: ClassVar[GrowthTrait]
    measurement_type: ClassVar[MeasurementType]
    units: ClassVar[str]
    reference_units: ClassVar[str]

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return EnvironmentResponseExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return EnvironmentResponseExperimentReference

    def _phenotype(self, value: float) -> EnvironmentResponsePhenotype:
        return EnvironmentResponsePhenotype(
            measurement_type=self.measurement_type,
            assay_type=AssayType.colony_size_array,
            environment_response=value,
            n_samples=GROWTH_REPLICATES.value,
            sample_unit=SampleUnit.colony,
            units=self.units,
        )

    def _reference_phenotype(self, mean: float) -> EnvironmentResponsePhenotype:
        return EnvironmentResponsePhenotype(
            measurement_type=self.measurement_type,
            assay_type=AssayType.colony_size_array,
            environment_response=mean,
            units=self.reference_units,
        )

    @post_process
    def process(self) -> None:
        """Build one record per released (isolate, environment) cell of the trait."""
        release, genotypes, built, without = self._prepare()
        values = growth_values(release, self.trait, built)
        by_stem: dict[str, list[float]] = {}
        for (_, stem), value in values.items():
            by_stem.setdefault(stem, []).append(value)
        environments = {e.stem: growth_environment(e) for e in release.environments}
        references = {
            stem: EnvironmentResponseExperimentReference(
                dataset_name=self.name,
                genome_reference=_genome_reference(),
                environment_reference=environments[stem].model_copy(),
                phenotype_reference=self._reference_phenotype(float(np.mean(cells))),
            )
            for stem, cells in by_stem.items()
        }

        def records() -> Iterator[
            tuple[Experiment, ExperimentReference, InternedGenotype]
        ]:
            for isolate in tqdm(built, desc=f"{self.name} isolates"):
                stems = [
                    e.stem for e in release.environments if (isolate, e.stem) in values
                ]
                if not stems:
                    continue
                interned = intern_genotype(genotypes.genotype(isolate), isolate)
                for stem in stems:
                    experiment = EnvironmentResponseExperiment(
                        dataset_name=self.name,
                        genotype=interned.genotype,
                        environment=environments[stem],
                        phenotype=self._phenotype(values[(isolate, stem)]),
                    )
                    yield experiment, references[stem], interned

        n = self._write(records())
        n_cells = len(release.st3) * len(release.environments)
        n_present = sum(
            int(release.st3[f"{e.stem}_{self.trait.value}"].notna().sum())
            for e in release.environments
        )
        normalized = sum(
            int(release.st3[f"{e.stem}_{NORMALIZED_SUFFIX[self.trait]}"].notna().sum())
            for e in release.environments
        )
        self._write_ledger(
            ReleaseLedger(
                dataset=self.name,
                n_isolates_in_release=len(release.st3),
                n_isolates_without_peter_genome=len(without),
                isolates_without_peter_genome=without,
                n_isolates_built=len({i for i, _ in values}),
                n_released_cells=n_cells,
                n_missing_cells=n_cells - n_present,
                n_refused_cells=n_present - len(values),
                n_kept_cells=len(values),
                n_records=n,
                refused_blocks=[
                    RefusedBlock(
                        block=f"Supplementary Table 3 '*_{NORMALIZED_SUFFIX[self.trait]}'",
                        n_cells=normalized,
                        reason=_NORMALIZED_REFUSAL,
                    )
                ],
            )
        )
        log.info("%s: wrote %d records over %d isolates", self.name, n, len(built))


@register_dataset
class DoublingTimeDeChiara2022Dataset(_GrowthDeChiara2022Dataset):
    """Colony population doubling time (h) of 1,011 isolates in 43 environments."""

    organism = Organism.scerevisiae

    trait = GrowthTrait.doubling_time
    measurement_type = MeasurementType.growth_rate
    units = (
        "h; colony population doubling time from Scan-o-matic growth curves on a "
        "1,536 array, the median of 4 replicate curves, released as an absolute "
        "value (Supplementary Table 3 'Doubling time - absolute values')"
    )
    reference_units = (
        "h; population mean of the built isolates' doubling times in this environment "
        "(a baseline over the panel, not a measured control)"
    )

    def __init__(
        self,
        root: str = "data/torchcell/doubling_time_dechiara2022",
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the doubling-time store."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)


@register_dataset
class CellYieldDeChiara2022Dataset(_GrowthDeChiara2022Dataset):
    """Colony cell yield (cells) of 1,011 isolates in 43 environments."""

    organism = Organism.scerevisiae

    trait = GrowthTrait.cell_yield
    measurement_type = MeasurementType.cell_yield
    units = (
        "cells; total cell yield of a colony population from Scan-o-matic calibrated "
        "cell counts on a 1,536 array, the median of 4 replicate curves, released as "
        "an absolute value (Supplementary Table 3 'Yield - absolute values')"
    )
    reference_units = (
        "cells; population mean of the built isolates' cell yields in this environment "
        "(a baseline over the panel, not a measured control)"
    )

    def __init__(
        self,
        root: str = "data/torchcell/cell_yield_dechiara2022",
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the cell-yield store."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)


@register_dataset
class LifeCycleDeChiara2022Dataset(_DeChiara2022Dataset):
    """Sporulation and stationary-phase survival time courses of 1,011 isolates."""

    organism = Organism.scerevisiae

    def __init__(
        self,
        root: str = "data/torchcell/life_cycle_dechiara2022",
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the life-cycle store."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return LifeCycleExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return LifeCycleExperimentReference

    @post_process
    def process(self) -> None:
        """Build one record per (isolate, protocol) with any released time point."""
        release, genotypes, built, without = self._prepare()
        values = life_cycle_values(release, built)
        environments = {p: protocol_environment(p) for p in LifeCycleProtocol}
        sums: dict[LifeCycleProtocol, dict[str, list[float]]] = {}
        for (_, protocol), keyed in values.items():
            for key, value in keyed.items():
                sums.setdefault(protocol, {}).setdefault(key, []).append(value)
        references = {
            protocol: LifeCycleExperimentReference(
                dataset_name=self.name,
                genome_reference=_genome_reference(),
                environment_reference=environments[protocol].model_copy(),
                phenotype_reference=protocol_phenotype(
                    protocol, {k: float(np.mean(v)) for k, v in keyed.items()}
                ),
            )
            for protocol, keyed in sums.items()
        }

        def records() -> Iterator[
            tuple[Experiment, ExperimentReference, InternedGenotype]
        ]:
            for isolate in tqdm(built, desc=f"{self.name} isolates"):
                protocols = [p for p in LifeCycleProtocol if (isolate, p) in values]
                if not protocols:
                    continue
                interned = intern_genotype(genotypes.genotype(isolate), isolate)
                for protocol in protocols:
                    experiment = LifeCycleExperiment(
                        dataset_name=self.name,
                        genotype=interned.genotype,
                        environment=environments[protocol],
                        phenotype=protocol_phenotype(
                            protocol, values[(isolate, protocol)]
                        ),
                    )
                    yield experiment, references[protocol], interned

        n = self._write(records())
        n_cells = len(release.st3) * len(LIFE_CYCLE_COLUMNS)
        n_values = sum(len(v) for v in values.values())
        n_present = sum(
            int(release.st3[c.column].notna().sum()) for c in LIFE_CYCLE_COLUMNS
        )
        self._write_ledger(
            ReleaseLedger(
                dataset=self.name,
                n_isolates_in_release=len(release.st3),
                n_isolates_without_peter_genome=len(without),
                isolates_without_peter_genome=without,
                n_isolates_built=len({i for i, _ in values}),
                n_released_cells=n_cells,
                n_missing_cells=n_cells - n_present,
                n_refused_cells=n_present - n_values,
                n_kept_cells=n_values,
                n_records=n,
            )
        )
        log.info("%s: wrote %d records over %d isolates", self.name, n, len(built))


# --------------------------------------------------------------------------- #
# Verification (L0-L4), one entry point for the three stores
# --------------------------------------------------------------------------- #
class StoreFamily(StrEnum):
    """Which of the three De Chiara stores a dataset root holds."""

    doubling_time = "doubling_time_dechiara2022"
    cell_yield = "cell_yield_dechiara2022"
    life_cycle = "life_cycle_dechiara2022"


_GROWTH_TRAIT: dict[StoreFamily, GrowthTrait] = {
    StoreFamily.doubling_time: GrowthTrait.doubling_time,
    StoreFamily.cell_yield: GrowthTrait.cell_yield,
}
_GROWTH_TYPE: dict[StoreFamily, MeasurementType] = {
    StoreFamily.doubling_time: MeasurementType.growth_rate,
    StoreFamily.cell_yield: MeasurementType.cell_yield,
}

#: Released viabilities above 100 (plate-normalized), measured on the pinned workbook:
#: 34 at SDC day 7, 2 at day 21, 1 at day 35. Observed must equal declared.
EXPECTED_VIABILITY_ABOVE_100 = 37
#: The share of distinct sequence-variant genes that must be current S288C genes. Peter
#: 2018's gene files predate R64-4-1, so a handful of retired names is expected.
MIN_VARIANT_GENE_CONTAINMENT = 0.9


def _stream_resolved(dataset_root: str) -> Iterator[dict[str, Any]]:
    """Every record with its interned constants spliced in, sharing one object per ref.

    Sharing is the point: a genotype referenced by 43 records is one dict here, so a
    verifier validates it once.
    """
    processed = osp.join(dataset_root, "processed")
    interned: dict[str, Any] = {}
    ienv = lmdb.open(osp.join(processed, "interned"), readonly=True, lock=False)
    with ienv.begin() as itxn:
        for key, value in itxn.cursor():
            interned[key.decode()] = pickle.loads(value)
    ienv.close()

    def splice(obj: Any) -> Any:
        if isinstance(obj, dict):
            ref = obj.get("$ref")
            if ref is not None:
                return interned[ref]
            return {k: splice(v) for k, v in obj.items()}
        return obj

    env = lmdb.open(osp.join(processed, "lmdb"), readonly=True, lock=False)
    try:
        with env.begin() as txn:
            for _, value in txn.cursor():
                record = pickle.loads(value)
                experiment = record["experiment"]
                yield {
                    "experiment": {k: splice(v) for k, v in experiment.items()},
                    "reference": splice(record["reference"]),
                    "publication": splice(record["publication"]),
                }
    finally:
        env.close()


def _result(
    level: Level, name: str, passed: bool, message: str, **details: Any
) -> LevelResult:
    return LevelResult(
        level=level, name=name, passed=passed, message=message, details=details
    )


def _growth_signatures(release: Release) -> dict[str, str]:
    """Environment JSON -> growth stem, from the same builder the loader uses."""
    return {
        growth_environment(env).model_dump_json(): env.stem
        for env in release.environments
    }


def _protocol_signatures() -> dict[str, LifeCycleProtocol]:
    """Environment JSON -> protocol, from the same builder the loader uses."""
    return {
        protocol_environment(protocol).model_dump_json(): protocol
        for protocol in LifeCycleProtocol
    }


def verify_build(
    dataset_root: str,
    *,
    data_root: str | None = None,
    expected_viability_above_100: int = EXPECTED_VIABILITY_ABOVE_100,
) -> VerificationReport:
    """L0-L4 for one De Chiara store; the store is named by its root's basename.

    - L0: every record validates as its experiment, reference and publication class;
      a genotype shared by many records is validated once.
    - L1: the record count is the released non-missing count; one record per (isolate,
      condition); every record's genotype names one isolate of Peter's panel.
    - L2: every stored value equals its released cell; growth values are positive;
      viabilities above 100 equal the declared count.
    - L3: one measurement type (growth) or one assay per protocol (life cycle); each
      reference is the population mean of its condition; every sourced quote is
      verbatim in its pinned mirror file; the shared media, compound and gap rules on
      the records with their genotype set aside.
    - L4: the sequence-variant genes are current S288C genes (containment floor).
    - The four isolate-genotype rules every natural-isolate store shares
      (``natural_isolate_genotype.IsolateGenotypeAudit``), once per distinct genotype.
    """
    from pydantic import TypeAdapter

    from torchcell.datamodels.schema import (
        EXPERIMENT_REFERENCE_TYPE_MAP,
        EXPERIMENT_TYPE_MAP,
    )
    from torchcell.verification.common import SharedRecordRules
    from torchcell.verification.natural_isolate_genotype import IsolateGenotypeAudit
    from torchcell.verification.runners import _sgd_gene_set
    from torchcell.verification.sourced import audit_sourced_value

    family = StoreFamily(osp.basename(osp.normpath(dataset_root)))
    data_root = data_root or _DeChiara2022Dataset._data_root()
    raw_dir = osp.join(dataset_root, "raw")
    workbook = osp.join(raw_dir, TABLES_FILENAME)
    verify_sha256(workbook, TABLES_SHA256)
    release = read_release(workbook)
    panel = panel_isolates(raw_dir)
    built = [n for n in release.st3[ISOLATE_COLUMN].astype(str) if n in panel]
    report = VerificationReport(
        dataset_name=family.value,
        provenance=Provenance(
            source_uri=f"$DATA_ROOT/{RAW_MIRROR_REL}/{TABLES_REL}",
            citation_key=CITATION_KEY,
            sha256=TABLES_SHA256,
            method="Supplementary Table 3 of the MOESM4 workbook (raw mirror); genotype "
            "from Peter 2018's gene-keyed stores in the genomes tier",
            page="Nat Ecol Evol 6:448-460, Supplementary Tables",
        ),
    )

    expected: dict[tuple[str, str], Any]
    signatures: dict[str, Any]
    if family is StoreFamily.life_cycle:
        expected = {
            (isolate, protocol.value): keyed
            for (isolate, protocol), keyed in life_cycle_values(release, built).items()
        }
        signatures = {k: v.value for k, v in _protocol_signatures().items()}
    else:
        expected = growth_values(release, _GROWTH_TRAIT[family], built)
        signatures = _growth_signatures(release)
    means: dict[str, Any] = {}
    for (_, condition), value in expected.items():
        if isinstance(value, dict):
            bucket = means.setdefault(condition, {})
            for key, v in value.items():
                bucket.setdefault(key, []).append(v)
        else:
            means.setdefault(condition, []).append(value)

    audit = IsolateGenotypeAudit(
        NaturalIsolateGenotypes(
            raw_dir,
            osp.join(dataset_root, "preprocess"),
            sorted({isolate for isolate, _ in expected}),
            data_root,
        ),
        osp.join(data_root, "torchcell-library"),
    )
    validated: dict[int, Any] = {}
    strains: dict[int, str | None] = {}
    variant_genes: set[str] = set()
    validate_publication = TypeAdapter(Publication).validate_python
    shared = SharedRecordRules()
    n = 0
    l0_failures: list[dict[str, Any]] = []
    seen: Counter[tuple[str, str]] = Counter()
    off_panel: list[str] = []
    mismatched: list[dict[str, Any]] = []
    bad_reference: list[str] = []
    kinds: dict[str, set[str]] = {}
    nonpositive = 0
    above_100 = 0
    for record in _stream_resolved(dataset_root):
        n += 1
        experiment = record["experiment"]
        genotype = experiment["genotype"]
        gid = id(genotype)
        if gid not in validated:
            validated[gid] = Genotype.model_validate(genotype)
            audit.add(genotype)
            strains[gid] = genotype_isolate(genotype)
            for p in genotype["perturbations"]:
                if p["perturbation_type"] == "sequence_variant":
                    variant_genes.add(p["systematic_gene_name"])
        try:
            experiment_class = cast(
                "type[BaseModel]", EXPERIMENT_TYPE_MAP[experiment["experiment_type"]]
            )
            experiment_class.model_validate({**experiment, "genotype": validated[gid]})
            reference = record["reference"]
            rid = id(reference)
            if rid not in validated:
                reference_class = cast(
                    "type[BaseModel]",
                    EXPERIMENT_REFERENCE_TYPE_MAP[
                        reference["experiment_reference_type"]
                    ],
                )
                validated[rid] = reference_class.model_validate(reference)
            validate_publication(record["publication"])
        except (ValueError, TypeError) as err:
            l0_failures.append({"index": n - 1, "error": str(err)[:300]})
        strain = strains[gid]
        if strain is None or strain not in panel:
            off_panel.append(str(strain))
            continue
        matched = signatures.get(
            Environment.model_validate(experiment["environment"]).model_dump_json()
            if family is not StoreFamily.life_cycle
            else CultureEnvironment.model_validate(
                experiment["environment"]
            ).model_dump_json()
        )
        key = (strain, str(matched))
        seen[key] += 1
        phenotype = experiment["phenotype"]
        reference_phenotype = record["reference"]["phenotype_reference"]
        if family is StoreFamily.life_cycle:
            stored = dict(phenotype["life_cycle"])
            kinds.setdefault(str(matched), set()).add(str(phenotype["assay"]))
            above_100 += sum(1 for v in stored.values() if v > 100.0)
            target = {
                k: float(np.mean(v)) for k, v in means.get(str(matched), {}).items()
            }
            if dict(reference_phenotype["life_cycle"]) != target:
                bad_reference.append(str(matched))
        else:
            stored = phenotype["environment_response"]
            kinds.setdefault("all", set()).add(str(phenotype["measurement_type"]))
            nonpositive += int(not (math.isfinite(stored) and stored > 0.0))
            target_mean = float(np.mean(means.get(str(matched), [math.nan])))
            if reference_phenotype["environment_response"] != target_mean:
                bad_reference.append(str(matched))
        if expected.get(key) != stored:
            mismatched.append({"isolate": strain, "condition": str(matched)})
        shared.add(
            {
                "experiment": {**experiment, "genotype": {"perturbations": []}},
                "reference": record["reference"],
            }
        )

    report.add(
        _result(
            Level.L0,
            "structural",
            not l0_failures,
            f"{n} records validated ({len(validated)} distinct genotypes and references)"
            if not l0_failures
            else f"{len(l0_failures)}/{n} records failed schema validation",
            failures=l0_failures[:20],
        )
    )
    report.add(
        _result(
            Level.L1,
            "count",
            n == len(expected),
            f"observed {n}, expected {len(expected)} released non-missing conditions",
            observed=n,
            expected=len(expected),
        )
    )
    duplicates = {f"{k[0]}|{k[1]}": c for k, c in seen.items() if c > 1}
    report.add(
        _result(
            Level.L1,
            "isolate_condition_uniqueness",
            not duplicates,
            f"{len(seen)} unique (isolate, condition) records",
            duplicates=dict(list(duplicates.items())[:20]),
        )
    )
    report.add(
        _result(
            Level.L1,
            "isolates_in_peter_panel",
            not off_panel,
            f"every record's genotype names one isolate of Peter's {len(panel)}-isolate "
            f"panel ({len({k[0] for k in seen})} isolates)"
            if not off_panel
            else f"{len(off_panel)} records name no single panel isolate",
            examples=off_panel[:20],
        )
    )
    missing = sorted(f"{k[0]}|{k[1]}" for k in set(expected) - set(seen))
    report.add(
        _result(
            Level.L2,
            "released_value_fidelity",
            not mismatched and not missing,
            f"every stored value equals its released Supplementary Table 3 cell "
            f"({len(seen)} conditions)"
            if not mismatched and not missing
            else f"{len(mismatched)} stored values differ from the release; "
            f"{len(missing)} released conditions are not stored",
            mismatched=mismatched[:20],
            missing=missing[:20],
        )
    )
    if family is StoreFamily.life_cycle:
        report.add(
            _result(
                Level.L2,
                "viability_above_100",
                above_100 == expected_viability_above_100,
                f"{above_100} plate-normalized viabilities above 100, "
                f"{expected_viability_above_100} declared",
                observed=above_100,
                declared=expected_viability_above_100,
            )
        )
        mixed = {c: sorted(v) for c, v in kinds.items() if len(v) != 1}
        report.add(
            _result(
                Level.L3,
                "one_assay_per_protocol",
                not mixed,
                f"{len(kinds)} protocols, one assay each",
                assays={c: sorted(v) for c, v in kinds.items()},
            )
        )
    else:
        report.add(
            _result(
                Level.L2,
                "positive_values",
                nonpositive == 0,
                f"{n - nonpositive} of {n} values finite and positive",
                n_nonpositive=nonpositive,
            )
        )
        types = sorted(kinds.get("all", set()))
        report.add(
            _result(
                Level.L3,
                "measurement_type_consistent",
                types == [_GROWTH_TYPE[family].value],
                f"measurement types {types}",
                measurement_types=types,
            )
        )
    report.add(
        _result(
            Level.L3,
            "reference_is_population_mean",
            not bad_reference,
            "every reference phenotype is the mean of its condition's released values"
            if not bad_reference
            else f"{len(bad_reference)} records carry a reference that is not the "
            "population mean of their condition",
            examples=sorted(set(bad_reference))[:20],
        )
    )
    library_root = osp.join(data_root, "torchcell-library")
    audits = [audit_sourced_value(sv, library_root) for sv in SOURCED_VALUES]
    failed = [a.details["source"] for a in audits if not a.passed]
    report.add(
        _result(
            Level.L3,
            "sourced_quotes_are_verbatim",
            not failed,
            f"{len(audits)} quotes re-read against their sha256-pinned mirror files; "
            f"{len(failed)} failed",
            failed=failed,
        )
    )
    for result in shared.results():
        report.add(result)
    for result in audit.results():
        report.add(result)
    genes = _sgd_gene_set(data_root)
    inside = len(variant_genes & genes)
    share = inside / len(variant_genes) if variant_genes else 0.0
    report.add(
        _result(
            Level.L4,
            "sequence_variant_genes_are_s288c_genes",
            share >= MIN_VARIANT_GENE_CONTAINMENT,
            f"{share:.3f} of {len(variant_genes)} sequence-variant genes are SGD R64-4-1 "
            f"genes (>= {MIN_VARIANT_GENE_CONTAINMENT})",
            n_genes=len(variant_genes),
            n_inside=inside,
            outside_examples=sorted(variant_genes - genes)[:20],
        )
    )
    return report


def main() -> None:
    """Build (or load) the three stores and print their record counts."""
    from dotenv import load_dotenv

    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    stores: dict[type[_DeChiara2022Dataset], StoreFamily] = {
        LifeCycleDeChiara2022Dataset: StoreFamily.life_cycle,
        DoublingTimeDeChiara2022Dataset: StoreFamily.doubling_time,
        CellYieldDeChiara2022Dataset: StoreFamily.cell_yield,
    }
    for cls, family in stores.items():
        dataset = cls(root=osp.join(data_root, "data/torchcell", family.value))
        print(f"{cls.__name__}: {len(dataset)} records")


if __name__ == "__main__":
    main()
