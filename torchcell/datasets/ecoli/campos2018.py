# torchcell/datasets/ecoli/campos2018
# [[torchcell.datasets.ecoli.campos2018]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/campos2018
# Test file: tests/torchcell/datasets/ecoli/test_campos2018.py
r"""Campos 2018: the imaged Keio collection, and the one of its 26 features we can store.

Campos et al. 2018 (Mol Syst Biol 14:e7573, doi:10.15252/msb.20177573; PMC6018989)
imaged the Keio single-gene deletion collection in one medium and quantified a
26-feature phenotype per strain: "we imaged 4,227 strains of the Keio collection"
grown "in 96-well plates in M9 medium supplemented with $0 . 1 \%$ casamino acids and
$0 . 2 \%$ glucose at $3 0 ^ { \circ } \mathrm { C }$ ."

WHAT THE 26 VALUES ARE. They are 26 FEATURES of ONE condition, not 26 conditions. The
paper counts them in three groups that sum to 26, and Appendix Table S1 ("Features
considered in this study and their associated symbols") names every symbol:

- 19 MORPHOLOGICAL -- "In total, each strain was characterized by 19 morphological
  features": the mean and CV of length, width, area, volume, surface area, perimeter,
  surface-area-to-volume ratio, circularity and aspect ratio (18), plus the CV of the
  division ratio. The mean division ratio is deliberately absent, because pole identity
  was unknown and "measurements of mean division ratio were meaningless and not included
  in our analysis".
- 2 GROWTH -- "we recorded the growth curves of all the strains (Fig 1A) and estimated
  two population-growth features. We fitted the Gompertz function to estimate the
  maximal growth rate ... and used the last hour of growth to calculate the saturating
  density $\mathrm { ( O D } _ { \operatorname* { m a x } } \mathrm { ) }$ of each
  culture".
- 5 CELL CYCLE -- "each strain was associated with five cell cycle features (Dataset
  EV2), in addition to the 19 morphological features and two growth features mentioned
  above": rho_CD, CDN_C0, the relative timings of cell constriction and of nucleoid
  separation, and %2N.

19 + 2 + 5 = 26, and the main text's own cross-sum confirms the split: "the close-to-zero
correlations between growth rate and any of the 24 morphological and cell cycle features
considered in our screen" (19 + 5). Dataset EV2 releases 30 numeric score columns, which
is the 26 plus the mean and CV of nucleoid area (Appendix Table S1 lists those under
morphological, giving 21 there and 28 symbols in all) plus %non-div and %1N, the raw
proportions the two relative timings are computed from.

RECORD = one (deletion strain) ``BacterialFitnessExperiment``. One record per strain, not
26, because 25 of the 26 features have no phenotype class (below).

- GENOTYPE: one ``BacterialDeletionPerturbation`` against BW25113
  (``ecoli_k12_bw25113_locus_tag``), the Keio background -- "240 replicates of the
  parental strain (BW25113, here referred to as WT) were also grown and imaged under the
  same conditions as the mutants." Dataset EV2 labels a row by its deleted gene's NAME
  ("Name of the deleted gene"), so each label is resolved to a BW25113 GenBank locus tag
  through ``bacteria_common.reconcile_locus_tags`` and the record carries
  ``identifier_mapping=DerivedIdentifierMapping(route="gene_symbol")``. ``cassette`` IS
  sourced here: "Genes in the Keio collection were deleted by an in-frame replacement of
  a kanamycin-resistance cassette". The released Keio plate and well ride on
  ``construction``.
- ENVIRONMENT: :data:`CAMPOS2018_M9_CASAMINO_GLUCOSE` at 30 C, aerobic, with NO
  environment perturbation -- the screen is one medium, which is exactly why the 26
  features cannot be spread across an environment axis.
- PHENOTYPE: ``FitnessPhenotype``, the DERIVED ratio of the strain's corrected maximal
  growth rate to the parent's, with the reference at 1.0.

WHAT IS SERVED, AND WHY ONLY ONE OF 26. Of the 26 features exactly one has a faithful
home in the schema, and ``schema.py`` is not changed here:

1. ``alpha_max`` (max growth rate, min^-1) IS a growth rate, and ``FitnessPhenotype``
   is documented as ``ko_growth_rate/wt_growth_rate``. It is served.
2. The 19 morphological and 5 cell cycle features have NO phenotype class. The only
   multi-feature morphology class is ``CalMorphPhenotype``, whose shape fits well (a
   dict of named means plus a dict of named CVs, which is exactly Campos's mean/CV
   split) but whose two ``field_validator``s reject any key outside ``CALMORPH_LABELS``
   (281 Ohya 2005 CalMorph base parameters) and ``CALMORPH_STATISTICS`` (220 CalMorph
   CV parameters). CalMorph is a yeast image-analysis program; Campos measured with
   MicrobeTracker and Oufti, and ``<L>``, ``CV_L``, ``rho_CD`` and the rest are not
   CalMorph parameters. Storing them under those keys would assert a measurement that
   was not made, and ``EnvironmentResponsePhenotype`` cannot hold them either: it
   carries ONE score per (strain, environment) and has no field naming WHICH feature a
   number is, so 24 features in one medium collide on the L1 key. There is also no
   ``BacterialCalMorphExperiment``. Serving them needs either E. coli symbols added to a
   yeast program's vocabulary or a new host-neutral multi-feature morphology phenotype,
   both edits to ``schema.py``. That is the finding, recorded rather than forced.
3. ``ODmax`` (saturating optical density) is a carrying capacity, not a rate: it is not a
   ``FitnessPhenotype`` ratio, and ``MeasurementType`` has no member for it
   (``growth_rate`` is "absolute or normalized growth rate / doubling time",
   ``colony_size`` is an absolute colony size). A second ``FitnessPhenotype`` record per
   strain would also collide on the L1 key, since the environment is the same. It is not
   served, and no ``MeasurementType`` member is added.

THE FITNESS IS DERIVED, AND THE DERIVATION IS VERIFIED. The loader reads the
``alpha_max (min-1)`` column of Dataset EV2's "Normalized data" sheet, the authors'
plate-, position-, time- and OD-corrected value, and divides it by the MEDIAN of the same
column over the 240 wild-type replicate rows (``WT0511`` x48, ``WT0813`` x96, ``WT0815``
x96), which is 0.0097366516 min^-1. The WT median is the right denominator because the
correction is anchored on it: "For each plate, we set the median values of each feature,
$F _ { ; }$ , to the median feature value of the parental strain." Cross-check, measured
over the 3,664 kept rows: the released ``alpha_max`` SCORE is an exact affine function of
the derived fitness, ``score = 23.777071 * (fitness - 1)``, with a maximum absolute
residual of 1.4e-13 and a Pearson r of 1.000000000000. That is the algebra the paper
states -- ``s = 1.35 * (F_i - median(F_i^WT)) / iqr(F_i^WT)`` -- so the ratio is the
released number in different units, not a reinterpretation. No kept fitness is
non-positive (measured range 0.121789 to 1.351417, median 1.004160), so
``FitnessPhenotype``'s clamp never fires.

IDENTIFIERS, measured on the release. 4,158 distinct non-wild-type labels reach the
reconciler and 3,720 resolve to one BW25113 locus (3,586 RENAMED, 134
NON_GENE_FEATURE), all through the gene-symbol layer (3,718) or an ECK ``gene_synonym``
(2); 438 are retired with no collision and no ambiguity, so the resolved fraction is
0.8947 and :data:`MIN_RESOLVED_FRACTION` is set below it. The 438 split into two
measured causes:

- 412 Keio JW strain ids. Campos labels a row by its JW id exactly when the strain has
  no current gene name. The BW25113 GenBank annotation DOES carry JW ids as
  ``gene_synonym`` (4,334 distinct ones, which is how Fuhrer 2017 resolves its strains),
  and NONE of these 412 is among them -- so these are the Keio strains whose deleted
  locus the pinned annotation no longer carries, not a resolver miss. MG1655 carries no
  JW synonym at all (measured: 0), so no second assembly rescues them, and the Keio
  strain-to-gene table is Baba 2006's web table, which the mirror does not hold. No
  ``JW####`` -> ``b####`` arithmetic is invented.
- 26 gene names the 2014 BW25113 annotation predates (``adeP``, ``adeQ``, ``elyC``,
  ``ettA``, ``ghoS``, ``ghoT``, ``ghxP``, ``ghxQ``, ``htgA``, ``iceT``, ``lgoD``,
  ``lgoR``, ``lgoT``, ``opgE``, ``rhoL``, ``sslE``, ``tiaE``, ``waaO``, ``yahH``,
  ``ybbV``, ``yiaI``, ``ykiB``, ``ymfH``, ``yzcX``, ``yzfA``, ``zapE``). Measured: the
  MG1655 annotation resolves 22 of the 26, so ``bacteria_common.eck_crosswalk`` is the
  identified route to recover them; it is a second, two-step derived route for 22 of
  4,227 rows (0.5%) and is left to a follow-up rather than added here.

RECORDS DROPPED (rule + counts + items in ``preprocess/dropped_records.json``):

1. ``row_is_a_table_footer_summary`` -- 4 rows. The "Normalized data" sheet ends with a
   blank row and three unlabelled summary rows (``Mean``, ``Stdev``, ``CV`` in the Well
   column). They carry no gene.
2. ``row_is_a_wild_type_replicate`` -- 240 rows, the ``WT0511`` / ``WT0813`` / ``WT0815``
   labels. An unperturbed parent is the experiment REFERENCE, not a record; their median
   is the fitness denominator and their count is the reference ``n_samples``.
3. ``label_carries_a_different_culture_or_check_annotation`` -- 5 rows (``hfq*``,
   ``pgm*``, ``rapZ*``, ``rodZ*``, ``fabH``+degree). The Dataset EV2 legend defines the
   two markers: ``*`` is "Re-imaged from 2mL liquid cultures" and the degree sign is
   "Strains independently checked with a different phenotype". A 2 mL tube is not the
   96-well screen environment, and an independent check with a different phenotype is a
   QC statement about the strain, so neither row is a measurement of this screen. None of
   the five bare names appears on another row (measured), so nothing is lost twice.
4. ``label_is_not_in_the_bw25113_annotation`` -- 438 rows, the retired labels above.
   ``BacterialDeletionPerturbation`` stores a locus tag of the declared namespace and
   nothing else, so a label with no locus cannot be written.
5. ``label_is_a_fragment_of_a_merged_bw25113_locus`` -- 0 rows on this release, because
   the reconciler reports no collision here. The rule exists so a future annotation that
   merges two of these labels onto one locus is dropped rather than silently merged.
6. ``label_is_ambiguous_in_bw25113`` -- 0 rows on this release (no label matches more
   than one locus). Same reason: a rule, not a measured count.
7. ``label_heads_more_than_one_row`` -- 120 rows, 56 labels. Each label heads 2 to 4
   rows at distinct Keio plate/well positions, so they are distinct strains the release
   names identically (``ygaQ`` heads four rows, and Shiver 2016 shows why: the current
   annotation merges the ``ygaQ_1``..``ygaQ_4`` fragments into one locus). Keeping them
   would put several records on one locus tag; keeping one would be an arbitrary choice
   between real measurements, so the whole group goes.

Arithmetic: 4,471 released rows - 4 - 240 - 5 - 438 - 0 - 0 - 120 = 3,664 records, each
one a distinct BW25113 locus tag.

SOURCED VALUES are module-level ``SourcedValue``s anchored to the sha256 of
``camposGenomewidePhenotypicAnalysis2018/paper.md`` in the literature mirror, or to the
sha256 of Dataset EV2 for the column-legend values, or typed ``ProvenanceGap``s. The
genuine absences:

- Uncertainty. Dataset EV2 releases one corrected ``alpha_max`` per strain and no
  dispersion, so ``fitness_uncertainty`` and ``fitness_se`` carry
  ``not_reported_by_primary`` gaps. There is one growth curve per strain -- "Cultures
  were diluted 1:300 in $1 5 0 ~ \mu \mathrm { l }$ of fresh M9 medium ... and grown in
  96-well plates at $3 0 ^ { \circ } \mathrm { C }$ with continuous shaking in a BioTek
  plate reader." -- one well per row (measured: 4,467 rows, 4,467 distinct (plate, well)
  pairs), so ``n_samples`` is 1 ``biological_replicate`` for a record and 240 for the
  reference. The 360 +/- 165 imaged cells per strain are the replicate design of the
  MORPHOLOGICAL features, not of the growth curve, so they are not this phenotype's
  ``n_samples``.
- ``Environment.duration_hours``. The endpoint is an optical-density rule, not a clock:
  every strain was sampled at an OD600 of 0.2 +/- 0.1. The field carries a
  ``not_reported_by_primary`` gap.

DATA SOURCE: Dataset EV2 (``MSB-14-e7573-s004.xlsx``) from the PMC Article Datasets
bucket (``pmc_cloud``, key ``PMC6018989.1/MSB-14-e7573-s004.xlsx``), deposited in
``$DATA_ROOT/torchcell-raw/camposGenomewidePhenotypicAnalysis2018/`` with a
``manifest.json``. Dataset EV1 (the pre-normalization raw table) and the Appendix (whose
Table S1 names every feature) are already mirrored under the literature key and are NOT
duplicated into the raw mirror; ``si_expected`` names both with their sha256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import os.path as osp
import re
import shutil
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    check_manifest_pin,
    link_verified,
    post_process,
    verify_raw_files,
    write_verified,
)
from torchcell.datamodels.bacterial_morphology_features import (
    CAMPOS2018_MORPHOLOGY_ASSAY as MORPHOLOGY_ASSAY,
)
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import M9
from torchcell.datamodels.schema import (
    BacterialDeletionPerturbation,
    BacterialFitnessExperiment,
    BacterialFitnessExperimentReference,
    BacterialMorphologyExperiment,
    BacterialMorphologyExperimentReference,
    BacterialMorphologyPhenotype,
    ComponentDefinition,
    Compound,
    Concentration,
    ConcentrationUnit,
    DerivedIdentifierMapping,
    Environment,
    Experiment,
    ExperimentReference,
    FitnessPhenotype,
    Genotype,
    Media,
    MediaComponent,
    MediaComponentRole,
    Publication,
    SampleUnit,
    StrainConstruction,
    Temperature,
)
from torchcell.datasets.bacteria_common import (
    STRAIN_GENE_NAMESPACES,
    LocusTagReconciliation,
    assembly_reference,
    bacterial_genome,
    reconcile_locus_tags,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.literature.manifest import (
    ROLE_RAW_DATA,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.literature.provenance import run_retriever
from torchcell.literature.retrieve import pmc_cloud_url
from torchcell.sequence.genome.base import GeneNameStatus
from torchcell.sequence.genome.ecoli.k12 import (
    EcoliK12BW25113Genome,
    EcoliK12Genome,
    EcoliK12StrainName,
)
from torchcell.verification.report import Provenance, VerificationReport
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
)

log = logging.getLogger(__name__)

DOI = "10.15252/msb.20177573"
PMCID = "PMC6018989"
TITLE = (
    "Genomewide phenotypic analysis of growth, cell morphogenesis, and cell cycle "
    "events in Escherichia coli"
)

CITATION_KEY = "camposGenomewidePhenotypicAnalysis2018"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"
DATASET_ROOT_REL = "data/torchcell/ecoli_growth_rate_campos2018"

#: Dataset EV2: the normalized-data and scores sheets. The loader's only raw input.
DATA_FILENAME = "MSB-14-e7573-s004.xlsx"
DATA_REL = f"data/{DATA_FILENAME}"
DATA_SHA256 = "10188365b9ebcf40309c4bdcb415b13472a460c6d0966ed860ad9e59df208274"
DATA_RETRIEVED_AT = "2026-10-07"
#: The PMC Article Datasets bucket key of Dataset EV2 (article version 1).
PMC_CLOUD_KEY = f"{PMCID}.1/{DATA_FILENAME}"
DATA_URL = pmc_cloud_url(PMC_CLOUD_KEY)

PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "1bf2f74bbd528f1cd88e46bf96e829b06a7c70dca9a8ad35502fa45e9c594e83"

#: Dataset EV1, the pre-normalization raw table. Its ``nb Cells`` column is the number
#: of segmented cells each strain's morphology means and CVs are computed over, which is
#: the morphology phenotype's ``n_samples``, so the morphology dataset CONSUMES it and it
#: is deposited in the raw mirror beside Dataset EV2 (the fitness dataset does not read
#: it). The literature mirror holds the same bytes as ``si/si3.xlsx``.
RAW_CELL_COUNTS_FILENAME = "MSB-14-e7573-s003.xlsx"
CELL_COUNTS_REL = f"data/{RAW_CELL_COUNTS_FILENAME}"
DATASET_EV1_REL = "si/si3.xlsx"
DATASET_EV1_SHA256 = "82f0777cf85c3277be95e04c04e893fdf46c412e2640d61b35e70af75b312837"
#: The PMC Article Datasets bucket key of Dataset EV1 (article version 1), the retrieval
#: the literature mirror recorded for the same bytes.
CELL_COUNTS_PMC_CLOUD_KEY = f"{PMCID}.1/{RAW_CELL_COUNTS_FILENAME}"
CELL_COUNTS_URL = pmc_cloud_url(CELL_COUNTS_PMC_CLOUD_KEY)
CELL_COUNTS_RETRIEVED_AT = "2026-10-09"
#: The Appendix, whose Table S1 names every feature and its symbol.
APPENDIX_REL = "si/si1.docx"
APPENDIX_SHA256 = "72cc3510fa63cbf625bb1cd17acebf2a4ed764be0b11acae312b7afbaec40c75"

NORMALIZED_SHEET = "Normalized data"
SCORES_SHEET = "Scores"
#: Dataset EV1's data sheet and the one column the morphology dataset reads from it.
RAW_SHEET = "Raw data"
CELL_COUNT_COLUMN = "nb Cells"
LABEL_COLUMN = "Gene deletion"
PLATE_COLUMN = "Plate nb"
WELL_COLUMN = "Well nb"
ALPHA_COLUMN = "alpha_max (min-1)"

#: Rows of the "Normalized data" sheet, footer rows included.
RELEASED_ROWS = 4471
#: Strains the paper states it imaged; the released rows minus footer and wild type.
IMAGED_STRAINS = 4227
#: Wild-type replicate rows, and their labels.
WILD_TYPE_ROWS = 240
WILD_TYPE_LABEL = re.compile(r"^WT\d+$")
#: The two Dataset EV2 row annotations: ``*`` and the degree sign.
ANNOTATION_MARKER = re.compile("[*°]")

KEIO_COLLECTION = "Keio collection"
KEIO_CASSETTE = "kanamycin-resistance cassette"
BW25113_NAMESPACE = STRAIN_GENE_NAMESPACES["BW25113"]

#: The 19 morphological features of the paper's count, by Dataset EV2 column name.
MORPHOLOGICAL_FEATURES: tuple[str, ...] = (
    "<L>",
    "CV_L",
    "<W>",
    "CV_W",
    "<A>",
    "CV_A",
    "<V>",
    "CV_V",
    "<SA>",
    "CV_SA",
    "<P>",
    "CV_P",
    "<SA/V>",
    "CV_SA/V",
    "<C>",
    "CV_C",
    "<Ar>",
    "CV_Ar",
    "CV_DR",
)
#: The two population-growth features.
GROWTH_FEATURES: tuple[str, ...] = ("alpha_max", "ODmax")
#: The five cell cycle features.
CELL_CYCLE_FEATURES: tuple[str, ...] = (
    "rho_CD",
    "CDN_C0",
    "Rel.timing div",
    "Rel.timing nuc",
    "%2N",
)
#: The paper's 26-feature phenotype: 19 + 2 + 5.
PAPER_FEATURES: tuple[str, ...] = (
    *MORPHOLOGICAL_FEATURES,
    *GROWTH_FEATURES,
    *CELL_CYCLE_FEATURES,
)
#: The one feature the FITNESS dataset serves.
SERVED_FEATURE = "alpha_max"
#: The 26 morphology symbols the MORPHOLOGY dataset serves, as Appendix Table S1 names
#: them: the paper's 19 headline morphological features plus the mean and variability of
#: nucleoid area that Table S1 also files as morphological, plus the 5 cell cycle
#: features. ``MORPHOLOGY_ASSAY`` is the authority; this tuple is its order.
MORPHOLOGY_FEATURES: tuple[str, ...] = tuple(
    feature.symbol for feature in MORPHOLOGY_ASSAY.features
)
#: The one feature of the release with no phenotype class left (issue #774 closed the
#: other 25 by adding ``BacterialMorphologyPhenotype``).
UNSERVED_FEATURES: dict[str, str] = {
    "ODmax": "a saturating optical density is a carrying capacity, not a growth-rate "
    "ratio, so it is not a FitnessPhenotype, and MeasurementType has no member for it; "
    "it is a plate-reader population measurement and not morphology, so it is not a "
    "BacterialMorphologyPhenotype feature either"
}

UNITS = (
    "fitness: the strain's corrected maximal growth rate (Gompertz alpha_max, min^-1) "
    "divided by the median of the 240 wild-type replicates"
)


# --------------------------------------------------------------------------- #
# Sourced values
# --------------------------------------------------------------------------- #
def _paper(
    value: Any, quote: str, *, page: str, note: str | None = None
) -> SourcedValue:
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
            page=page,
        ),
    )


def _legend(
    value: Any, quote: str, *, page: str, note: str | None = None
) -> SourcedValue:
    """Bind a value to a verbatim cell of Dataset EV2's own column legend."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=DATA_REL,
            citation_key=CITATION_KEY,
            sha256=DATA_SHA256,
            method="Dataset EV2 legend sheet, read with pandas.read_excel",
            page=page,
        ),
    )


def _appendix(
    value: Any, quote: str, *, page: str, note: str | None = None
) -> SourcedValue:
    """Bind a value to a verbatim cell or caption of the sha256-pinned Appendix."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=APPENDIX_REL,
            citation_key=CITATION_KEY,
            sha256=APPENDIX_SHA256,
            method="Appendix (.docx) read as the text runs of word/document.xml with "
            "the stdlib XML parser; a subscript run is written _{...}, a Symbol-font "
            "glyph is named, and an OMML equation contributes no text run, so a "
            "formula shows as a gap in the quote",
            page=page,
        ),
    )


_RESULTS = (
    "Results, 'High-throughput imaging and growth measurements of the Keio collection'"
)
_MORPH = "Results, 'Quantification of cell morphological features across the genome'"
_CYCLE = "Results, 'Quantification of growth and cell cycle features across the genome'"
_DEPENDENCIES = (
    "Results, 'Dependencies between cellular dimensions and cell cycle progression'"
)
_PATHWAYS = (
    "Results, 'Genes, functions, and pathways associated with cell size and shape'"
)
_GROWTH_CONDITIONS = "Materials and Methods, 'Screening setup and microscopy'"
_DATA_PROCESSING = "Materials and Methods, 'Data processing'"
_LEGEND_SCORES = "Dataset EV2, sheet 'Legend scores'"

LIBRARY_STRAIN = _paper(
    "BW25113",
    "To provide a reference, 240 replicates of the parental strain (BW25113, here "
    "referred to as WT) were also grown and imaged under the same conditions as the "
    "mutants.",
    page=_RESULTS,
    note="the Keio background, hence the pinned assembly and the gene namespace; the "
    "240 replicates are the reference's n_samples and the fitness denominator",
)
SCREEN_SIZE = _paper(
    IMAGED_STRAINS,
    "we imaged 4,227 strains of the Keio collection",
    page=_RESULTS,
    note="the released rows minus the 4 footer rows and the 240 wild-type replicates",
)
NONESSENTIAL_COVERAGE = _paper(
    "98% of the non-essential genome",
    "This set of single-gene deletion strains represents $9 8 \\%$ of the non-essential "
    "genome",
    page=_RESULTS,
)
MEDIUM = _paper(
    "M9 + 0.1% casamino acids + 0.2% glucose, 30 C",
    "The strains were grown in 96-well plates in M9 medium supplemented with "
    "$0 . 1 \\%$ casamino acids and $0 . 2 \\%$ glucose at "
    "$3 0 ^ { \\circ } \\mathrm { C }$ .",
    page=_RESULTS,
    note="the one base medium of the screen; there is no environment perturbation",
)
CULTURE = _paper(
    1,
    "Cultures were diluted 1:300 in $1 5 0 ~ \\mu \\mathrm { l }$ of fresh M9 medium "
    "supplemented with $0 . 1 \\%$ casamino acids and $0 . 2 \\%$ glucose and grown in "
    "96-well plates at $3 0 ^ { \\circ } \\mathrm { C }$ with continuous shaking in a "
    "BioTek plate reader.",
    page=_GROWTH_CONDITIONS,
    note="one well per row, so n_samples is 1 biological_replicate; continuous shaking "
    "is the aerobicity",
)
GROWTH_FEATURE_DEFINITION = _paper(
    GROWTH_FEATURES,
    "In parallel, using a microplate reader, we recorded the growth curves of all the "
    "strains (Fig 1A) and estimated two population-growth features. We fitted the "
    "Gompertz function to estimate the maximal growth rate "
    "$\\left( \\mathsf { a } _ { \\mathrm { m a x } } \\right)$ and used the last hour of",
    page=_RESULTS,
    note="alpha_max is a growth RATE, which is what FitnessPhenotype's ratio is of",
)
ODMAX_DEFINITION = _paper(
    "ODmax",
    "growth to calculate the saturating density $\\mathrm { ( O D } _ { "
    "\\operatorname* { m a x } } \\mathrm { ) }$ of each culture",
    page=_RESULTS,
    note="a carrying capacity, not a rate: no phenotype class and no MeasurementType "
    "member fits it, so it is not served",
)
MORPHOLOGICAL_COUNT = _paper(
    len(MORPHOLOGICAL_FEATURES),
    "In total, each strain was characterized by 19 morphological features",
    page=_MORPH,
)
CELL_CYCLE_COUNT = _paper(
    len(CELL_CYCLE_FEATURES),
    "As a result, each strain was associated with five cell cycle features (Dataset "
    "EV2), in addition to the 19 morphological features and two growth features "
    "mentioned above",
    page=_CYCLE,
    note="19 + 2 + 5 = 26, which is what one strain's phenotype is",
)
FEATURE_CROSS_SUM = _paper(
    len(MORPHOLOGICAL_FEATURES) + len(CELL_CYCLE_FEATURES),
    "the close-to-zero correlations between growth rate and any of the 24 "
    "morphological and cell cycle features considered in our screen",
    page=_DEPENDENCIES,
    note="the paper's own cross-sum of the 19 morphological and 5 cell cycle features",
)
NORMALIZATION_ANCHOR = _paper(
    "median of the parental strain",
    "For each plate, we set the median values of each feature, $F _ { ; }$ , to the "
    "median feature value of the parental strain.",
    page=_DATA_PROCESSING,
    note="why the wild-type median is the fitness denominator: the correction is "
    "anchored on it, so the ratio is plate-comparable",
)
SCORE_TRANSFORM = _paper(
    "robust z-score",
    "The $F$ values were transformed into normalized scores by a transformation akin "
    "to a $z$ -score transformation but more robust to outliers.",
    page=_DATA_PROCESSING,
    note="s = 1.35 * (F_i - median(F_i^WT)) / iqr(F_i^WT), which makes the released "
    "score an exact affine function of the derived fitness ratio",
)
CASSETTE = _paper(
    KEIO_CASSETTE,
    "Genes in the Keio collection were deleted by an in-frame replacement of a "
    "kanamycin-resistance cassette that has a constitutive promoter and no "
    "transcriptional terminator to ensure expression of downstream genes in operons "
    "(Baba et al, 2006).",
    page=_PATHWAYS,
    note="the arrayed library's cassette, stored on every perturbation",
)
CELLS_PER_STRAIN = _paper(
    360,
    "On average, about 360 $\\left( \\pm 1 6 5 \\right)$ cells were imaged for each "
    "strain.",
    page=_RESULTS,
    note="the replicate design of the MORPHOLOGICAL features, not of the growth curve, "
    "so it is not this phenotype's n_samples",
)
LABELS_ARE_GENE_NAMES = _legend(
    "gene names",
    "Name of the deleted gene",
    page=_LEGEND_SCORES,
    note="the identifier space of the label column, hence route='gene_symbol'",
)
PLATE_AND_WELL = _legend(
    (PLATE_COLUMN, WELL_COLUMN),
    "Number of the Keio plate",
    page=_LEGEND_SCORES,
    note="the released collection position, stored on StrainConstruction",
)
MARKER_REIMAGED = _legend(
    "*",
    "Re-imaged from 2mL liquid cultures",
    page=_LEGEND_SCORES,
    note="a 2 mL tube is not the 96-well screen environment, so these rows are dropped",
)
MARKER_CHECKED = _legend(
    "°",
    "Strains independently checked with a different phenotype",
    page=_LEGEND_SCORES,
    note="a QC statement about the strain, so these rows are dropped",
)

# --------------------------------------------------------------------------- #
# Morphology: what each feature IS, and where its name and unit come from
# --------------------------------------------------------------------------- #
_TABLE_S1 = "Appendix Table S1, 'Features considered in this study and their associated symbols'"
_LEGEND_NORMALIZED = "Dataset EV2, sheet 'Legend normalized data'"

FEATURE_TABLE_IS_THE_AUTHORITY = _paper(
    "Appendix Table S1",
    "The name and abbreviation for all the features can be found in Appendix Table S1.",
    page=_MORPH,
    note="why the morphology vocabulary is Table S1 and not the main text: the main "
    "text points at Table S1 for every name and symbol",
)
MORPHOLOGICAL_TABLE_COUNT = _appendix(
    21,
    "Morphological features",
    page=_TABLE_S1,
    note="Table S1's morphological block holds 21 symbols, the 19 of the main text's "
    "headline count plus <NA> (Mean nucleoid area) and CV_NA (Nucleoid area "
    "variability); 21 + 5 cell cycle = the 26 morphology symbols served",
)
CV_DEFINITION = _paper(
    "standard deviation divided by the mean",
    "We also measured the variability of these features by calculating their "
    "coefficient of variation (CV, the standard deviation divided by the mean).",
    page=_MORPH,
    note="the statistic of every CV_ symbol, which is why they are the ones filed under "
    "morphology_coefficient_of_variation",
)
DIMENSIONS_MEASURED = _paper(
    ("length", "width", "perimeter", "area", "aspect ratio", "circularity"),
    "From phase-contrast images, we measured cellular dimensions, such as length, "
    "width, perimeter, cross-sectional area, aspect ratio (width/length), and "
    "circularity $4 \\pi$ area/(perimeter)2 ).",
    page=_MORPH,
    note="the per-cell quantities the <X> means are means OF; surface area, volume and "
    "the surface-to-volume ratio are derived from the same two series",
)
DIVISION_RATIO_HAS_NO_MEAN = _paper(
    "CV_DR",
    "Therefore, measurements of mean division ratio were meaningless and not included "
    "in our analysis. However, the CV of the division ratio was included since a high "
    "CV indicated either an asymmetric division or an imprecise division site "
    "selection.",
    page=_MORPH,
    note="why the vocabulary holds 10 means against 11 CVs: there is deliberately no "
    "<DR> to pair with CV_DR",
)
CELL_CYCLE_DEFINITIONS = _paper(
    CELL_CYCLE_FEATURES,
    "From the images, we also calculated the degree of constriction for each cell and "
    "determined the fraction of constricting cells in the population for each strain "
    "(see Materials and Methods). From the latter, we inferred the timing of initiation "
    "of cell constriction relative to the cell cycle (Powell, 1956; Collins & Richmond, "
    "1962; Wold et al, 1994). In addition, the analysis of DAPI-stained nucleoids with "
    "the objectDetection module of Oufti (Paintdakhi et al, 2016) provided additional "
    "parameters, such as the number of nucleoids per cell and the fraction of cells "
    "with one versus two nucleoids. From the fraction of cells with two nucleoids, we "
    "estimated the relative timing of nucleoid separation (Powell, 1956; Collins & "
    "Richmond, 1962; Wold et al, 1994). We also measured the degree of nucleoid "
    "constriction in each cell for each strain and compared it to the degree of cell "
    "constriction to obtain the Pearson correlation between these two parameters, as "
    "well as the average degree of nucleoid separation at the onset of cell constriction "
    "(Appendix Fig S1H).",
    page=_CYCLE,
    note="the statistic of each of the five: rho_CD is a Pearson correlation, CDN_C0 a "
    "fitted degree at the onset of constriction, the two relative timings are inferred "
    "from population fractions, and %2N is a fraction of cells",
)
CELL_CYCLE_STATISTICS = _appendix(
    ("constriction degree <0.15", "Pearson correlation coefficient", "intercept"),
    "The relative timing of cell constriction and nucleoid separation were estimated as "
    "the proportions of cells without any significant constriction (constriction degree "
    "<0.15) or with a single nucleoid, respectively. For all cells with a significant "
    "constriction degree, we calculated the Pearson correlation coefficient between the "
    "constriction degrees of the cell and of its nucleoid ([SYM char=F072 font=Symbol] "
    "CD). The nucleoid constriction degree at the initiation of cell constriction "
    "(CDN_{C0}) was determined as the intercept of a line with a slope determined by "
    "the correlation coefficient that best fitted the single-cell data used to "
    "calculate [SYM char=F072 font=Symbol] CD (see Appendix Fig S1E).",
    page=f"{_TABLE_S1}, caption",
    note="the subset of cells each cell cycle feature is computed over, and the exact "
    "statistic of rho_CD and CDN_C0; the Symbol-font glyph F072 is a Greek rho, which "
    "the released column writes rho_CD",
)
SHAPE_FACTOR_DEFINITIONS = _appendix(
    ("<Ar>", "<C>"),
    "The aspect ratio was defined as the ratio of cell width over cell length at the "
    "single-cell level. The circularity, C, was defined as , at the single-cell level, "
    "where P stands for perimeter and A for area.",
    page=f"{_TABLE_S1}, caption",
    note="the gap after 'defined as' is an OMML equation, which contributes no text "
    "run; it reads C = 4*pi*A/P^2. Both are dimensionless per-cell shape factors, which "
    "is why the release states no unit for either, and width over length puts a rod "
    "below 1",
)
FEATURE_UNITS = _legend(
    {
        feature.symbol: feature.unit
        for feature in MORPHOLOGY_ASSAY.features
        if feature.unit is not None
    },
    "Mean cell length (µm)",
    page=_LEGEND_NORMALIZED,
    note="the release is the ONLY source of a feature unit: Appendix Table S1 has no "
    "unit column and the paper states no feature unit. The legend sheet names each "
    "feature with its unit and the 'Normalized data' header repeats it, which is why "
    "the loader builds each column name as 'symbol (unit)' from the vocabulary and "
    "stops if the header has moved",
)
CELLS_PER_STRAIN_RETAINED = _paper(
    291,
    "retaining about 1,300,000 identified cells ( $2 9 1 \\pm 1 1 6$ cells/strain)",
    page=_RESULTS,
    note="the cells each morphology mean and CV is computed over AFTER curation, and the "
    "mean of Dataset EV1's nb Cells column over its 4,467 rows is 291.26 with SD 116.71, "
    "so that column is this number per strain and is the morphology n_samples",
)
NON_DETERMINED_FIELDS = _legend(
    None,
    "NaN (Not a Number) values are attributed to non-determined fields.",
    page=_LEGEND_NORMALIZED,
    note="why a record carries the features determined for ITS strain rather than the "
    "whole vocabulary: 278 of the 4,227 imaged strains have no nucleoid channel, so "
    "their 7 nucleoid-derived features do not exist",
)

#: Every sourced value anchored to the Appendix.
APPENDIX_SOURCED_VALUES: tuple[SourcedValue, ...] = (
    MORPHOLOGICAL_TABLE_COUNT,
    CELL_CYCLE_STATISTICS,
    SHAPE_FACTOR_DEFINITIONS,
)

#: Every module-level sourced value, for the mirror audit.
SOURCED_VALUES: tuple[SourcedValue, ...] = (
    LIBRARY_STRAIN,
    SCREEN_SIZE,
    NONESSENTIAL_COVERAGE,
    MEDIUM,
    CULTURE,
    GROWTH_FEATURE_DEFINITION,
    ODMAX_DEFINITION,
    MORPHOLOGICAL_COUNT,
    CELL_CYCLE_COUNT,
    FEATURE_CROSS_SUM,
    NORMALIZATION_ANCHOR,
    SCORE_TRANSFORM,
    CASSETTE,
    CELLS_PER_STRAIN,
    FEATURE_TABLE_IS_THE_AUTHORITY,
    CV_DEFINITION,
    DIMENSIONS_MEASURED,
    DIVISION_RATIO_HAS_NO_MEAN,
    CELL_CYCLE_DEFINITIONS,
    CELLS_PER_STRAIN_RETAINED,
)
#: The sourced values whose anchor is Dataset EV2's legend, not the paper OCR.
LEGEND_SOURCED_VALUES: tuple[SourcedValue, ...] = (
    LABELS_ARE_GENE_NAMES,
    PLATE_AND_WELL,
    MARKER_REIMAGED,
    MARKER_CHECKED,
    FEATURE_UNITS,
    NON_DETERMINED_FIELDS,
)


# --------------------------------------------------------------------------- #
# Media and environment
# --------------------------------------------------------------------------- #
#: The one screen medium. ``M9`` is the shared salts-only recipe; casamino acids is an
#: acid hydrolysate of casein with no structure to resolve, so it is a bare ``Compound``
#: marked ``intrinsically_undefined`` (the Wang 2018 convention), which is also why
#: ``is_synthetic`` is False.
CAMPOS2018_M9_CASAMINO_GLUCOSE = Media(
    name="M9 with 0.1% (w/v) casamino acids and 0.2% (w/v) glucose (Campos 2018)",
    state="liquid",
    is_synthetic=False,
    base_medium="M9",
    components=[
        *M9.components,
        MediaComponent(
            compound=Compound(name="casamino acids"),
            role=MediaComponentRole.complex_ingredient,
            concentration=Concentration(value=0.1, unit=ConcentrationUnit.percent_w_v),
            definition=ComponentDefinition.intrinsically_undefined,
            provenance=[MEDIUM, CULTURE],
            note="an acid hydrolysate of casein, so there is no structure to resolve",
        ),
        MediaComponent(
            compound=resolved_compound("D-glucose"),
            role=MediaComponentRole.carbon_source,
            concentration=Concentration(value=0.2, unit=ConcentrationUnit.percent_w_v),
            provenance=[MEDIUM, CULTURE],
            note="the one carbon source; it is fixed across the screen, so it is a "
            "component rather than a varied physical factor",
        ),
    ],
    provenance=[MEDIUM],
)

TEMPERATURE_C = 30.0
#: The endpoint is an optical-density rule, not a clock, so there is no duration.
DURATION_GAP = ProvenanceGap(
    field="duration_hours",
    reason=ProvenanceGapReason.not_reported_by_primary,
    note="every strain was sampled at an OD600 of 0.2 +/- 0.1, so the endpoint is a "
    "density rule; the paper states no growth duration for the imaged cultures",
)


def environment() -> Environment:
    """The one screen environment: the base medium at 30 C, aerobic, unperturbed."""
    return Environment(
        media=CAMPOS2018_M9_CASAMINO_GLUCOSE,
        temperature=Temperature(value=TEMPERATURE_C),
        perturbations=[],
        aerobicity="aerobic",
        duration_hours=None,
        provenance_gaps=[DURATION_GAP],
    )


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """``DATA_ROOT`` from the environment (the mirror and build tree live under it)."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/camposGenomewidePhenotypicAnalysis2018``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def library_dir(data_root: str | None = None) -> Path:
    """The literature mirror of this citation key."""
    return Path(data_root or _data_root()) / "torchcell-library" / CITATION_KEY


def _sha256(path: str | Path) -> str:
    """Streaming sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_ev2_retrieval(retrieved_at: str = DATA_RETRIEVED_AT) -> RetrievalRecord:
    """The recorded retrieval of Dataset EV2: one PMC Article Datasets object.

    ``run_retriever`` on this record re-fetches the bytes; the pinned sha256 is the
    anchor a rebuild verifies against.
    """
    return RetrievalRecord(
        method=RetrievalMethod.pmc_cloud,
        source_url=DATA_URL,
        retriever="torchcell.literature.retrieve.pmc_cloud_object",
        params={"key": PMC_CLOUD_KEY},
        sha256=DATA_SHA256,
        retrieved_at=retrieved_at,
    )


def retrieve_raw_files(dest_dir: str | Path) -> dict[str, Path]:
    """Run the recorded retrieval and write the verified bytes into ``dest_dir``.

    The recorded ``RetrievalRecord`` is what runs, so this IS the re-runnable retrieval;
    a byte mismatch raises before anything is written.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / DATA_FILENAME
    write_verified(run_retriever(dataset_ev2_retrieval()), path, DATA_SHA256, DATA_URL)
    return {DATA_FILENAME: path}


def deposit_raw_mirror(
    *,
    data_path: str | Path,
    retrieved_at: str = DATA_RETRIEVED_AT,
    data_root: str | None = None,
) -> Path:
    """Write the raw mirror from Dataset EV2 plus its ``manifest.json``.

    Idempotent by sha256: an existing mirror file with the recorded hash is left alone
    and a differing one raises rather than being overwritten. The bytes are verified
    against ``DATA_SHA256`` before anything is written.
    """
    got = _sha256(data_path)
    if got != DATA_SHA256:
        raise RuntimeError(
            f"{data_path} sha256 mismatch: got {got}, expected {DATA_SHA256}"
        )
    root = raw_mirror_dir(data_root)
    dest = root / DATA_REL
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if _sha256(dest) != DATA_SHA256:
            raise RuntimeError(f"{dest} exists with a different sha256; refusing")
    else:
        shutil.copy2(data_path, dest)
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=DOI,
        title=TITLE,
        files=[
            ArtifactRecord(
                path=DATA_REL,
                role=ROLE_RAW_DATA,
                bytes=dest.stat().st_size,
                sha256=DATA_SHA256,
                source=DATA_URL,
                retrieval=dataset_ev2_retrieval(retrieved_at),
            )
        ],
        si_data_sources=[DATA_URL],
        si_expected=[
            f"Dataset EV2 ({DATA_FILENAME}): the corrected per-strain table. The "
            f"loader consumes the '{NORMALIZED_SHEET}' sheet's '{ALPHA_COLUMN}' "
            "column; the 'Scores' sheet is the released robust z-score of the same "
            "value and is the derivation's cross-check",
            f"Dataset EV1 (MSB-14-e7573-s003.xlsx): the pre-normalization raw table "
            f"(cell counts, sampling OD, elapsed time). Already mirrored under the "
            f"literature key as {DATASET_EV1_REL} (sha256 {DATASET_EV1_SHA256}), and "
            "not consumed by this loader, so it is NOT duplicated here",
            f"Appendix (MSB-14-e7573-s001.docx): Table S1 names every feature and its "
            f"symbol, which is how the 26-feature count was established. Already "
            f"mirrored under the literature key as {APPENDIX_REL} (sha256 "
            f"{APPENDIX_SHA256}), so it is NOT duplicated here",
            "Computer Code EV1/EV2 (MSB-14-e7573-s005.zip, -s006.zip): the authors' "
            "analysis scripts, not data. NOT mirrored",
        ],
        provenance_complete=True,
        created_at=datetime.now(UTC).isoformat(),
    )
    (root / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    return root


def load_manifest(data_root: str | None = None) -> Manifest:
    """Read the raw mirror's ``manifest.json``."""
    path = raw_mirror_dir(data_root) / "manifest.json"
    return Manifest.model_validate_json(path.read_text())


def manifest_sha256(manifest: Manifest, relpath: str) -> str:
    """The sha256 the manifest records for ``relpath``."""
    for record in manifest.files:
        if record.path == relpath:
            return record.sha256
    raise KeyError(f"{relpath} is not in the raw manifest of {CITATION_KEY}")


# --------------------------------------------------------------------------- #
# Reader
# --------------------------------------------------------------------------- #
class StrainRow(BaseModel):
    """One row of Dataset EV2's normalized table, as the loader reads it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    plate: str
    well: str
    alpha_max: float


class NormalizedTable(BaseModel):
    """Dataset EV2's normalized sheet: the wild-type rows and the mutant rows."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    released_rows: int
    footer_rows: int
    wild_type: tuple[StrainRow, ...]
    mutants: tuple[StrainRow, ...]

    @property
    def wild_type_median_alpha_max(self) -> float:
        """The fitness denominator: the median wild-type corrected growth rate."""
        values = sorted(row.alpha_max for row in self.wild_type)
        if not values:
            raise ValueError("no wild-type replicate rows: no fitness denominator")
        middle = len(values) // 2
        if len(values) % 2:
            return values[middle]
        return (values[middle - 1] + values[middle]) / 2


def read_normalized_table(path: str | Path) -> NormalizedTable:
    """Read Dataset EV2's normalized sheet, splitting footer, wild-type and mutant rows.

    The four pinned columns must be present and the sheet must carry the released row
    count, so a release whose shape moved stops the build instead of being reinterpreted.
    Rows with no gene label are the sheet's footer summary (``Mean``, ``Stdev``, ``CV``);
    rows labelled ``WT<digits>`` are the parental replicates.
    """
    frame = pd.read_excel(path, sheet_name=NORMALIZED_SHEET)
    missing = [
        column
        for column in (LABEL_COLUMN, PLATE_COLUMN, WELL_COLUMN, ALPHA_COLUMN)
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"{path}: {NORMALIZED_SHEET} is missing columns {missing}")
    released = len(frame)
    if released != RELEASED_ROWS:
        raise ValueError(
            f"{path}: {NORMALIZED_SHEET} has {released} rows, expected {RELEASED_ROWS}"
        )
    labelled = frame[frame[LABEL_COLUMN].notna()].reset_index(drop=True)
    footer = released - len(labelled)
    labels = [str(value).strip() for value in labelled[LABEL_COLUMN].tolist()]
    plates = [str(value).strip() for value in labelled[PLATE_COLUMN].tolist()]
    wells = [str(value).strip() for value in labelled[WELL_COLUMN].tolist()]
    alphas = labelled[ALPHA_COLUMN].astype(float).tolist()
    rows = [
        StrainRow(label=label, plate=plate, well=well, alpha_max=alpha)
        for label, plate, well, alpha in zip(labels, plates, wells, alphas, strict=True)
    ]
    positions = Counter((row.plate, row.well) for row in rows)
    repeated = sorted(key for key, count in positions.items() if count > 1)
    if repeated:
        raise ValueError(
            f"{path}: {len(repeated)} (plate, well) positions appear twice: "
            f"{repeated[:5]}"
        )
    wild_type = tuple(row for row in rows if WILD_TYPE_LABEL.match(row.label))
    if len(wild_type) != WILD_TYPE_ROWS:
        raise ValueError(
            f"{path}: {len(wild_type)} wild-type rows, expected {WILD_TYPE_ROWS}"
        )
    mutants = tuple(row for row in rows if not WILD_TYPE_LABEL.match(row.label))
    return NormalizedTable(
        released_rows=released, footer_rows=footer, wild_type=wild_type, mutants=mutants
    )


def released_alpha_max_scores(path: str | Path) -> dict[tuple[str, str], float]:
    """Dataset EV2's released ``alpha_max`` score, keyed by (plate, well).

    The cross-check the module docstring states: the score is an exact affine function
    of the derived fitness ratio.
    """
    frame = pd.read_excel(path, sheet_name=SCORES_SHEET)
    plates = [str(value).strip() for value in frame[PLATE_COLUMN].tolist()]
    wells = [str(value).strip() for value in frame[WELL_COLUMN].tolist()]
    scores = frame[SERVED_FEATURE].astype(float).tolist()
    return {
        (plate, well): score
        for plate, well, score in zip(plates, wells, scores, strict=True)
    }


# --------------------------------------------------------------------------- #
# Row resolution
# --------------------------------------------------------------------------- #
DROP_FOOTER = "row_is_a_table_footer_summary"
DROP_WILD_TYPE = "row_is_a_wild_type_replicate"
DROP_ANNOTATED = "label_carries_a_different_culture_or_check_annotation"
DROP_NOT_IN_ANNOTATION = "label_is_not_in_the_bw25113_annotation"
DROP_MERGED_LOCUS = "label_is_a_fragment_of_a_merged_bw25113_locus"
DROP_AMBIGUOUS = "label_is_ambiguous_in_bw25113"
DROP_DUPLICATE_LABEL = "label_heads_more_than_one_row"

ROW_RULES: tuple[tuple[str, str], ...] = (
    (
        DROP_FOOTER,
        "the normalized sheet's trailing blank and Mean/Stdev/CV summary rows carry no "
        "gene label",
    ),
    (
        DROP_WILD_TYPE,
        "an unperturbed BW25113 parent is the experiment reference, not a record; the "
        "240 replicates supply the fitness denominator and the reference n_samples",
    ),
    (
        DROP_ANNOTATED,
        "the Dataset EV2 legend marks the row '*' (re-imaged from 2 mL liquid cultures, "
        "which is not the 96-well screen environment) or with a degree sign (the strain "
        "was independently checked with a different phenotype)",
    ),
    (
        DROP_NOT_IN_ANNOTATION,
        "the label is no locus tag, symbol or synonym of the pinned BW25113 assembly "
        "(Keio JW strain ids and gene names the annotation predates), and a bacterial "
        "deletion leaf stores a locus tag of its declared namespace and nothing else",
    ),
    (
        DROP_MERGED_LOCUS,
        "two or more labels resolve to ONE current locus, because the collection "
        "deleted fragments of a region the pinned annotation merges; no member IS the "
        "locus tag, so the whole group goes rather than merging distinct strains",
    ),
    (
        DROP_AMBIGUOUS,
        "the label matches more than one locus of the pinned assembly, so no single "
        "locus tag can be written for it",
    ),
    (
        DROP_DUPLICATE_LABEL,
        "the label heads more than one row, at distinct Keio plate/well positions, so "
        "the rows are distinct strains the release names identically; mapping them onto "
        "one locus would merge strains and keeping one would be arbitrary",
    ),
)

#: Fraction of the release's distinct mutant labels that must resolve to one BW25113
#: locus. Below it the build stops and reports instead of dropping strains. Measured on
#: the release: 3,720 of 4,158 (0.8947), the shortfall being 412 Keio JW strain ids the
#: pinned annotation does not carry and 26 gene names it predates.
MIN_RESOLVED_FRACTION = 0.89


class StrainRecord(BaseModel):
    """One kept row: its source label, its BW25113 locus tag and its growth rate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    row: StrainRow
    locus_tag: str


class RowResolution(BaseModel):
    """What the row rules did to Dataset EV2's normalized sheet."""

    model_config = ConfigDict(extra="forbid")

    kept: list[StrainRecord]
    dropped_rows: dict[str, int]
    dropped_labels: dict[str, list[str]]
    reconciliation: LocusTagReconciliation


def resolve_rows(
    table: NormalizedTable, genome: EcoliK12BW25113Genome, *, label: str
) -> RowResolution:
    """Map each mutant row to a BW25113 locus tag, applying the five row rules.

    The retain-all reconciliation decides which labels resolve; this function decides
    which rows can be WRITTEN. Every dropped row is attributed to exactly one rule, in
    :data:`ROW_RULES` order.
    """
    rows = list(table.mutants)
    annotated = [row for row in rows if ANNOTATION_MARKER.search(row.label)]
    rest = [row for row in rows if not ANNOTATION_MARKER.search(row.label)]
    stored, reconciliation = reconcile_locus_tags(
        genome, pd.Series([row.label for row in rest]), label=label
    )
    reconciliation.require_resolved(MIN_RESOLVED_FRACTION)
    by_rule: dict[str, set[str]] = {
        DROP_NOT_IN_ANNOTATION: set(reconciliation.retired_kept),
        DROP_MERGED_LOCUS: set(reconciliation.kept_on_collision),
        DROP_AMBIGUOUS: set(reconciliation.ambiguous_kept),
    }
    unwritable = set().union(*by_rule.values())
    tags = [str(value) for value in stored.tolist()]
    resolved = [
        StrainRecord(row=row, locus_tag=tag)
        for row, tag in zip(rest, tags, strict=True)
        if row.label not in unwritable
    ]
    counts = Counter(record.row.label for record in resolved)
    repeated = {name for name, count in counts.items() if count > 1}
    kept = [record for record in resolved if record.row.label not in repeated]
    claimed = Counter(record.locus_tag for record in kept)
    collisions = sorted(tag for tag, count in claimed.items() if count > 1)
    if collisions:
        raise RuntimeError(
            f"{label}: {len(collisions)} locus tags are claimed by more than one kept "
            f"row after the drop rules: {collisions[:10]}"
        )
    outside = sorted(
        {
            record.locus_tag
            for record in kept
            if genome.resolve_gene_name(record.locus_tag).status
            not in (GeneNameStatus.CURRENT, GeneNameStatus.NON_GENE_FEATURE)
        }
    )
    if outside:
        raise RuntimeError(
            f"{label}: {len(outside)} kept tags are not loci of the pinned assembly: "
            f"{outside[:10]}"
        )
    return RowResolution(
        kept=kept,
        dropped_rows={
            DROP_FOOTER: table.footer_rows,
            DROP_WILD_TYPE: len(table.wild_type),
            DROP_ANNOTATED: len(annotated),
            **{
                rule: sum(1 for row in rest if row.label in names)
                for rule, names in by_rule.items()
            },
            DROP_DUPLICATE_LABEL: len(resolved) - len(kept),
        },
        dropped_labels={
            DROP_FOOTER: [],
            DROP_WILD_TYPE: sorted({row.label for row in table.wild_type}),
            DROP_ANNOTATED: sorted({row.label for row in annotated}),
            **{
                rule: sorted({row.label for row in rest if row.label in names})
                for rule, names in by_rule.items()
            },
            DROP_DUPLICATE_LABEL: sorted(repeated),
        },
        reconciliation=reconciliation,
    )


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
_GAP_UNCERTAINTY = ProvenanceGap(
    field="fitness_uncertainty",
    reason=ProvenanceGapReason.not_reported_by_primary,
    note="Dataset EV2 releases one corrected alpha_max per strain and no dispersion; "
    "there is one growth curve per well, so no within-strain spread exists to report",
)
_GAP_SE = ProvenanceGap(
    field="fitness_se",
    reason=ProvenanceGapReason.not_reported_by_primary,
    note="no uncertainty is reported, so no standard error can be derived",
)
PHENOTYPE_GAPS: tuple[ProvenanceGap, ...] = (_GAP_UNCERTAINTY, _GAP_SE)


def phenotype(alpha_max: float, wild_type_median: float) -> FitnessPhenotype:
    """One strain's corrected maximal growth rate, as a ratio to the parent's."""
    return FitnessPhenotype(
        fitness=alpha_max / wild_type_median,
        n_samples=1,
        sample_unit=SampleUnit.biological_replicate,
        provenance_gaps=list(PHENOTYPE_GAPS),
    )


def reference_phenotype(n_replicates: int = WILD_TYPE_ROWS) -> FitnessPhenotype:
    """The parent in the same medium: the ratio of its own median to itself, 1.0."""
    return FitnessPhenotype(
        fitness=1.0,
        n_samples=n_replicates,
        sample_unit=SampleUnit.biological_replicate,
        provenance_gaps=list(PHENOTYPE_GAPS),
    )


def genotype(record: StrainRecord) -> Genotype:
    """One precise gene deletion of the Keio collection, written against BW25113."""
    return Genotype(
        perturbations=[
            BacterialDeletionPerturbation(
                systematic_gene_name=record.locus_tag,
                perturbed_gene_name=record.row.label,
                gene_namespace=BW25113_NAMESPACE,
                identifier_mapping=DerivedIdentifierMapping(
                    source_identifier=record.row.label, route="gene_symbol"
                ),
                collection=KEIO_COLLECTION,
                cassette=KEIO_CASSETTE,
                construction=StrainConstruction(
                    plate=record.row.plate, well=record.row.well
                ),
            )
        ]
    )


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
#: Records of the full build: 4,471 released rows - 4 - 240 - 5 - 438 - 120.
EXPECTED_RECORDS = 3664


@register_dataset
class GrowthRateCampos2018Dataset(ExperimentDataset):
    """Campos 2018 Keio maximal-growth-rate fitness in M9 casamino-acid glucose."""

    #: The strain whose genome the build entry points inject as ``ecoli_genome``: the
    #: Keio collection's background, and the assembly every record pins.
    REFERENCE_STRAIN: ClassVar[EcoliK12StrainName] = "BW25113"

    def __init__(
        self,
        root: str = DATASET_ROOT_REL,
        io_workers: int = 0,
        ecoli_genome: EcoliK12Genome | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset; the BW25113 genome is injected or opened in process."""
        self.ecoli_genome = ecoli_genome
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return BacterialFitnessExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return BacterialFitnessExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The mirrored Dataset EV2."""
        return [DATA_FILENAME]

    def download(self) -> None:
        """Link the mirror file into ``raw/`` after verifying it against ``DATA_SHA256``.

        The mirror plus the pin is canonical; the PMC bucket URL is retrieval metadata
        ``deposit_raw_mirror`` records, never a live build dependency.
        """
        data_root = _data_root()
        manifest = load_manifest(data_root)
        check_manifest_pin(DATA_REL, manifest_sha256(manifest, DATA_REL), DATA_SHA256)
        src = raw_mirror_dir(data_root) / DATA_REL
        if not src.exists():
            raise RuntimeError(f"required raw artifact missing from mirror: {src}")
        os.makedirs(self.raw_dir, exist_ok=True)
        link_verified(src, osp.join(self.raw_dir, DATA_FILENAME), DATA_SHA256)
        log.info(
            "Campos 2018 Dataset EV2 linked into %s (sha256 verified)", self.raw_dir
        )

    def _genome(self) -> EcoliK12BW25113Genome:
        """The BW25113 genome, injected by a build or opened from its cache root."""
        if self.ecoli_genome is None:  # a direct run; the build entry points inject it
            self.ecoli_genome = bacterial_genome("ecoli", self.REFERENCE_STRAIN)
        genome = self.ecoli_genome
        if not isinstance(genome, EcoliK12BW25113Genome):
            raise TypeError(
                f"{self.name} needs the BW25113 genome, got {type(genome).__name__}"
            )
        return genome

    @post_process
    def process(self) -> None:
        """Parse Dataset EV2 into one fitness record per kept strain; write the LMDB."""
        verify_raw_files(self.raw_dir, {DATA_FILENAME: DATA_SHA256})
        path = osp.join(self.raw_dir, DATA_FILENAME)
        table = read_normalized_table(path)
        genome = self._genome()
        resolution = resolve_rows(table, genome, label=self.name)
        denominator = table.wild_type_median_alpha_max
        env = environment()
        reference = BacterialFitnessExperimentReference(
            dataset_name=self.name,
            genome_reference=assembly_reference(self.REFERENCE_STRAIN),
            environment_reference=env.model_copy(),
            phenotype_reference=reference_phenotype(len(table.wild_type)),
        )
        publication = Publication(doi=DOI, doi_url=f"https://doi.org/{DOI}")

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        lmdb_env, interned_env = self._open_write_lmdb(
            osp.join(self.processed_dir, "lmdb")
        )
        idx = 0
        with lmdb_env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for record in tqdm(resolution.kept, desc="campos2018"):
                experiment = BacterialFitnessExperiment(
                    dataset_name=self.name,
                    genotype=genotype(record),
                    environment=env,
                    phenotype=phenotype(record.row.alpha_max, denominator),
                )
                txn.put(
                    f"{idx}".encode(),
                    self._intern_record(experiment, reference, publication, itxn),
                )
                idx += 1
        lmdb_env.close()
        interned_env.close()

        self._write_reports(table, resolution, denominator, kept_records=idx)
        log.info(
            "Campos2018: wrote %d records from %d released rows; dropped rows %s; "
            "wild-type median alpha_max %.10f min^-1",
            idx,
            table.released_rows,
            resolution.dropped_rows,
            denominator,
        )

    def _write_reports(
        self,
        table: NormalizedTable,
        resolution: RowResolution,
        denominator: float,
        *,
        kept_records: int,
    ) -> None:
        """Write the drop log and the identifier report, and check the arithmetic."""
        dropped = sum(resolution.dropped_rows.values())
        if table.released_rows - dropped != kept_records:
            raise RuntimeError(
                f"drop accounting mismatch: {table.released_rows} released rows - "
                f"{dropped} dropped != {kept_records} records"
            )
        out = Path(self.preprocess_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "dropped_records.json").write_text(
            json.dumps(
                {
                    "dataset": self.name,
                    "released_rows": table.released_rows,
                    "kept_records": kept_records,
                    "dropped_records": dropped,
                    "rules": [
                        {
                            "rule": rule,
                            "scope": "row",
                            "description": description,
                            "n_rows": resolution.dropped_rows[rule],
                            "labels": resolution.dropped_labels[rule],
                        }
                        for rule, description in ROW_RULES
                    ],
                },
                indent=2,
            )
        )
        (out / "identifier_reconciliation.json").write_text(
            json.dumps(
                {
                    "dataset": self.name,
                    "released_rows": table.released_rows,
                    "imaged_strains": len(table.mutants),
                    "kept_records": kept_records,
                    "min_resolved_fraction": MIN_RESOLVED_FRACTION,
                    "reconciliation": resolution.reconciliation.model_dump(mode="json"),
                    "identifier_route": "gene_symbol",
                },
                indent=2,
            )
        )
        (out / "served_features.json").write_text(
            json.dumps(
                {
                    "dataset": self.name,
                    "paper_features": list(PAPER_FEATURES),
                    "morphological_features": list(MORPHOLOGICAL_FEATURES),
                    "growth_features": list(GROWTH_FEATURES),
                    "cell_cycle_features": list(CELL_CYCLE_FEATURES),
                    "served": [SERVED_FEATURE],
                    "unserved": UNSERVED_FEATURES,
                    "wild_type_median_alpha_max": denominator,
                    "wild_type_replicates": len(table.wild_type),
                },
                indent=2,
            )
        )

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError(
            "GrowthRateCampos2018Dataset builds records in process()"
        )


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def _bw25113(data_root: str | None = None) -> EcoliK12BW25113Genome:
    """The BW25113 genome from its default cache root."""
    genome = bacterial_genome("ecoli", "BW25113", data_root)
    if not isinstance(genome, EcoliK12BW25113Genome):
        raise TypeError(f"expected the BW25113 genome, got {type(genome).__name__}")
    return genome


def verify_build(
    dataset_root: str,
    *,
    genome: EcoliK12BW25113Genome | None = None,
    data_root: str | None = None,
    expected_count: int = EXPECTED_RECORDS,
) -> VerificationReport:
    """Run the fitness L0-L4 verifier on a built tree and write its report.

    Every record is checked against the BW25113 genome its references pin: the resolver
    of the canonical-name rule, and as the L4 universe every GenBank locus of the
    assembly. One resolver cannot serve two strains, which is why the host-aware gene set
    is read from the record's own pinned assembly rather than from the yeast default. The
    report is written to ``preprocess/verification_report.json``.
    """
    from torchcell.verification.fitness import verify_fitness_dataset
    from torchcell.verification.runners import stream_records

    if genome is None:
        genome = _bw25113(data_root)
    records: Sequence[Any] = list(stream_records(dataset_root))
    report = verify_fitness_dataset(
        records,
        dataset_name=osp.basename(osp.normpath(dataset_root)),
        provenance=Provenance(
            source_uri=f"$DATA_ROOT/{RAW_DIR_REL}/{DATA_REL}",
            citation_key=CITATION_KEY,
            sha256=DATA_SHA256,
            method="Dataset EV2 'Normalized data' sheet, column 'alpha_max (min-1)'; "
            "one BacterialFitnessExperiment per kept Keio deletion row, fitness = the "
            "strain's corrected maximal growth rate over the median of the 240 "
            "wild-type replicates",
            page=f"Dataset EV2 ({DATA_FILENAME}), sheet '{NORMALIZED_SHEET}'",
            retrieved=DATA_RETRIEVED_AT,
        ),
        expected_count=expected_count,
        sgd_genes=set(genome.genbank.loci),
        gene_universe_label=f"BW25113 ({genome.ASSEMBLY_SET})",
        resolve_gene_name=genome.resolve_gene_name,
    )
    preprocess = osp.join(dataset_root, "preprocess")
    os.makedirs(preprocess, exist_ok=True)
    with open(osp.join(preprocess, "verification_report.json"), "w") as handle:
        handle.write(report.model_dump_json(indent=2))
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI: ``deposit`` the raw mirror, ``build`` the dev LMDB, or ``verify`` it."""
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(
        prog="python -m torchcell.datasets.ecoli.campos2018"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    deposit = sub.add_parser("deposit", help="deposit the raw mirror")
    deposit.add_argument(
        "--retrieve-into",
        default=None,
        help="re-run the recorded PMC retrieval into this directory and deposit those "
        "bytes; without it the literature mirror's captured Dataset EV2 is used",
    )
    sub.add_parser("build", help="build (or load) the dev-tree LMDB")
    sub.add_parser("verify", help="run L0-L4 on the built dev-tree LMDB")
    args = parser.parse_args(argv)

    load_dotenv()
    data_root = _data_root()
    if args.command == "deposit":
        if args.retrieve_into is not None:
            source: str | Path = retrieve_raw_files(args.retrieve_into)[DATA_FILENAME]
        else:
            source = library_dir(data_root) / "si" / "si4.xlsx"
        print(deposit_raw_mirror(data_path=source, data_root=data_root))
        return 0
    root = osp.join(data_root, DATASET_ROOT_REL)
    if args.command == "build":
        dataset = GrowthRateCampos2018Dataset(root=root)
        print(f"len = {len(dataset)}")
        return 0
    report = verify_build(root, data_root=data_root)
    print(report.summary())
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
