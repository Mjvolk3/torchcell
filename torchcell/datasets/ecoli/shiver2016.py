# torchcell/datasets/ecoli/shiver2016
# [[torchcell.datasets.ecoli.shiver2016]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/shiver2016
# Test file: tests/torchcell/datasets/ecoli/test_shiver2016.py
"""Shiver 2016: the neglected-antibiotic chemical-genomic screen of E. coli K-12.

Shiver et al. 2016 (PLoS Genet 12(6):e1006124, doi:10.1371/journal.pgen.1006124;
PMC4927156) pinned an arrayed K-12 mutant library onto agar plates carrying one stress,
imaged the plates once the colonies reached a defined size, quantified colony opacity
with Iris, and assigned each mutant a fitness-score per condition. "We tested the
sensitivities of 3975 mutants of E. coli K-12 to 57 stresses, split between new and
previously screened conditions." S1 Dataset is the integrated matrix: 3,975 gene columns
by 292 condition rows, of which 235 rows are Nichols et al. 2011 re-analyzed (batch 0)
and 57 are this study's own screen (batches 1 and 4). 3,975 x 57 = 226,575 released
cells, and those 57 columns of work are what this loader serves.

RECORD = one (deletion strain x screen condition)
``BacterialEnvironmentResponseExperiment``:

- GENOTYPE: one ``BacterialDeletionPerturbation`` against BW25113
  (``ecoli_k12_bw25113_locus_tag``), because "The KEIO deletion library is derived from
  BW25113". The release labels its columns with GENE NAMES ("Gene names are used to
  label the mutation"), so each label is resolved to a BW25113 GenBank locus tag through
  ``bacteria_common.reconcile_locus_tags`` and the record carries
  ``identifier_mapping=DerivedIdentifierMapping(route="gene_symbol")``. No ECK crosswalk
  is needed or used: the release names symbols, not another strain's locus tags.
- ENVIRONMENT: the screen plate. ``SHIVER2016_LB_LENNOX_AGAR`` by default ("Chemical
  sensitivity screens used LB Lennox agar plates ... unless otherwise specified"), or
  ``SHIVER2016_M9_MINIMAL_AGAR`` for the nine conditions the release prefixes "M9min".
  Each dosed chemical is a ``SmallMoleculePerturbation`` at a sourced ``Concentration``;
  the M9 plates' carbon source is an ``EnvironmentPhysicalPerturbation``
  (``factor=carbon_source``); UV is one (``factor=radiation``); temperature lives on
  ``Environment.temperature`` (M2), which is how 10 C, 25 C and the 4 C survival
  condition are encoded.
- PHENOTYPE: ``EnvironmentResponsePhenotype``, ``measurement_type=z_score``,
  ``assay_type=colony_size_array``, ``screen_id`` = the verbatim released condition
  label. The reference carries 0.0: a fitness-score of 0 is no change in colony size.

WHY ``EnvironmentResponsePhenotype`` AND NOT ``FitnessPhenotype``. The released number
is SIGNED and routinely negative (measured over the release's full 57-condition grid,
218,103 non-blank cells of 226,575: min -30.935, max 10.399, mean -0.0907): "These fitness-scores represent the statistical significance of a change in
colony size for a particular condition, with negative and positive fitness-scores
representing sensitivity and resistance, respectively." ``FitnessPhenotype`` is a
strictly positive ko/wt ratio that CLAMPS non-positive values and whose verifier wants a
1.0 reference, so it would destroy more than half of this dataset's information.

WHY ``measurement_type=z_score``. The fitness-score is the Collins/Nichols S-score: the
Methods defer the pipeline ("The chemical genomics screen was conducted using the same
methodology as reported previously [8] with few modifications", and the "original
analysis pipeline [18]"), where [8] is Nichols et al. 2011 (Cell 144:143) and [18] is
Collins et al. 2006 (Genome Biol 7:R63), whose S-score is a modified t-statistic. That
is a STANDARDIZED colony-size deviation, which is what ``MeasurementType.z_score``
("standardized fitness/growth deviation") names. It is deliberately NOT ``log2_ratio``
(the number is not a log of a ratio), NOT ``differential_fitness`` (not a plain
subtraction of two normalized fitnesses), NOT ``colony_size`` (that member is defined as
an absolute, unnormalized size) and NOT ``sensitivity_score`` (a one-sided fitness-defect
score). ``MeasurementType`` has no ``s_score`` member and ``schema.py`` is not changed
here; the PR records that an ``s_score`` member would be more precise than ``z_score``.

UNITS CONVERTED, NEVER INVENTED. The released doses use seven units. Four are typed as
released (``mM``, ``uM``, ``% (w/v)``, ``% (v/v)``). The three mass-per-volume units are
stored in the single typed ``ug/mL`` by exact decimal conversion, recorded in
:data:`UNIT_CONVERSIONS`: ``ug/mL`` x1, ``ng/mL`` x1e-3, ``mg/mL`` x1e3. No new
``ConcentrationUnit`` member is added, and ``mg/L`` (deliberately deferred) is not
needed: it is numerically identical to the ``ug/mL`` the enum already carries.

SOURCED VALUES are module-level ``SourcedValue``s anchored to the sha256 of
``shiverChemicalGenomicScreenNeglected2016/paper.md`` in the literature mirror, or typed
``ProvenanceGap``s. The two that are genuinely absent:

- ``n_samples`` / ``sample_unit``: the number of colonies behind one fitness-score is
  nowhere in this paper or its SI. The Methods say only "Unreliable measurements were
  removed from the dataset at multiple points in the analysis and each condition had a
  different number of measurements (fitness-scores) that passed analysis", and the
  replicate design is deferred to Nichols 2011 ("the same methodology as reported
  previously [8]"), which is NOT in the mirror. Both fields carry a
  ``deferred_pending_source_review`` gap whose ``resolve_with`` names Nichols 2011, so
  the deferral chain (this Methods -> ref [8] -> ref [18]) is the record, not a guess.
- Uncertainty: S1 Dataset releases one number per cell and no dispersion, so
  ``environment_response_uncertainty`` and ``environment_response_se`` carry
  ``not_reported_by_primary`` gaps. The score is itself a significance statistic, so the
  paper's own dispersion statement is the FDR cutoff band, not a per-cell error:
  "95% of the cutoff values for negative (sensitization) fitness-scores fell in the range
  (-2.0,-1.2) while 95% of the cutoff values for positive (resistance) fitness-scores
  fell in the range (+1.2,+2.1)".

The UV dose is reported as an EXPOSURE TIME with no irradiance ("UV [12 sec]"), so no
fluence exists to store; the radiation perturbation's ``magnitude`` is ``None`` with a
``not_reported_by_primary`` gap, and the 12 s exposure survives verbatim in
``screen_id``. ``ConcentrationUnit`` carries no time or fluence unit, which the PR
records as a finding rather than a schema edit.

Growth duration is not a property of the screen: the endpoint is a colony-size rule
("incubated at 37 C until the majority of colonies reached a defined size (~8 hours)"),
and the cold conditions necessarily ran longer. ``Environment.duration_hours`` is
therefore ``None`` with a ``not_reported_by_primary`` gap for 56 conditions. The 4 C
survival condition is the exception: its exposure IS released ("transfer of the colony
array to 4 C for 5 weeks"), so it stores 840.0 h.

``cassette`` is left ``None`` on every perturbation. Shiver names a kanamycin cassette
only for the MG1655 operon deletions built AFTER the screen ("Operon deletions were
generated using lambda-red recombineering to replace the operon with a kanamycin
resistance cassette amplified from pKD4"), never for the arrayed library, and a
``GenePerturbation`` is not a gap carrier, so the absence is recorded here and in the
dendron note instead of on the record.

RECORDS DROPPED (rule + counts + items in ``preprocess/dropped_records.json``). Each
column rule removes one column across all 57 conditions:

1. ``label_is_not_a_gene_deletion`` -- 133 labels, 134 columns. The arrayed library is
   not only Keio: the labels include Nichols 2011's essential-gene arrays, marked by the
   release's own suffixes (``-SPA`` affinity tags, ``-DAS``/``-DAS+4`` degrons, ``-kan``
   insertions, ``{...}`` point mutants, and ``yfiO*``). The S1 Dataset legend is explicit
   that these are the exceptions: "Unless otherwise specified, the mutations are precise
   gene deletions." A SPA-tagged hypomorph is not a deletion, and the schema has no
   bacterial tagged-allele or degron leaf, so storing it as a
   ``BacterialDeletionPerturbation`` would assert an absence that did not happen.
2. ``label_is_not_in_the_bw25113_annotation`` -- 49 labels. The label is no locus tag,
   symbol or synonym of GCA_000750555.1: multi-gene deletions (``ecnAB``, ``rdlABC``,
   ``rdlABCD``, ``sibABCDE``), old-annotation ORF fragments (``ygaQ_1`` to ``ygaQ_4``,
   ``ypjM_1`` to ``ypjM_3``, ``phnE_1``, ``phnE_2``, ``ybeM_1``, ``ybeM_2``,
   ``yghX_1``, ``yghX_2``, ``gapC_2``, ``arpB_1``, ``lomR_2``, ``wbbL_1``, ``efeU_1``,
   ``ycgH_1``, ``yeeL_1``, ``yhiS_1``), retired symbols and small RNAs (``istR-1``,
   ``istR-2``, ``rybD``, ``ryeF``, ``ryfD/clpB``, ``tisA``, ``tp2``, ``tpkE70``,
   ``atl``, ``comR``, ``htgA``, ``yahH``, ``ybhU``, ``ygdT``, ``ykiB``, ``ymfH``,
   ``yzcX``, ``yzfA``), an ECK accession (``ECK0503``), and the ``-A``/``-B``/``-C``
   suffixed ``murE`` and ``waaA`` alleles.
3. ``label_is_a_fragment_of_a_merged_bw25113_locus`` -- 48 labels. Two or more labels
   resolve to ONE current locus, because the collection deleted fragments of a
   pseudogene the current annotation merges (``glvB``/``glvC``/``glvG``, ``ygeK`` to
   ``ygeQ``, the ``yai`` cluster). No member IS the locus tag, so the whole group goes:
   mapping them onto one locus would merge distinct strains, which is the retain-all
   policy's collision case.
4. ``label_is_ambiguous_in_bw25113`` -- 2 labels (``rffT``, ``spr``), each matching more
   than one locus.
5. ``label_has_two_columns_in_the_release`` -- 11 labels, 22 columns. Twelve labels head
   two columns each; ``yfiO*`` is already dropped by rule 1. The two columns always
   disagree where both are present (measured: 236 to 256 co-present cells per pair, zero
   identical), so they are two independent measurements, and the release carries nothing
   -- no plate, well, or allele id -- that tells the two strains apart. Keeping both
   would put two records on one (strain, condition) key; keeping one would be an
   arbitrary choice between two real measurements, so both go.
6. ``cell_is_blank`` -- 8,007 cells. The strain has no fitness-score in that condition;
   the Methods state the filtering ("Unreliable measurements were removed ... each
   condition had a different number of measurements ... that passed analysis").

Arithmetic: 3,975 columns - (134 + 49 + 48 + 2 + 22) = 3,720 kept columns, each one
distinct BW25113 locus tag. 3,720 x 57 = 212,040 cells, minus 8,007 blanks = 204,033
records. Against the full release grid: 226,575 - 255 x 57 - 8,007 = 204,033.

PARSE CROSS-CHECK (recorded, not run in the build). S1 Table ("Cold-sensitive genes from
the screen", ``si/si6.docx``, sha256
6521f4ef4242e3e66781713a4fea2ef85342dca3077d26927d24cc72a2fab3e2 in the literature
mirror) prints a "10 C fitness-score" column to one decimal. Ten of its genes against
the ``10C [-] {4}`` row of S1 Dataset: typA -10.456/-10.5, ihfB -8.732/-8.7, ihfA
-8.679/-8.7, dinJ -8.345/-8.3, rbfA -6.903/-6.9, deaD -6.701/-6.7, hfq -5.741/-5.7, crr
-4.830/-4.8, ycbK -4.061/-4.1, nfuA -4.041/-4.0. Ten of ten agree at the released
precision, which fixes the matrix orientation, the condition label and the sign.

DATA SOURCE: S1 Dataset, ``pgen.1006124.s005.txt``, from the PMC Article Datasets bucket
(``pmc_cloud``, key ``PMC4927156.1/pgen.1006124.s005.txt``), deposited in
``$DATA_ROOT/torchcell-raw/shiverChemicalGenomicScreenNeglected2016/`` with a
``manifest.json``. Re-fetched 2026-10-07 and sha256-identical to the copy the literature
mirror already held, so the retrieval is scriptable and reproducible. The Dryad deposit
(doi:10.5061/dryad.f3kc0, mirrored on Zenodo) holds the plate images and Iris output
behind the scoring pipeline; it is NOT needed, because S1 Dataset already releases the
scored matrix, and it is not mirrored.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import os.path as osp
import re
import shutil
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    check_manifest_pin,
    link_verified,
    post_process,
    verify_raw_files,
)
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import LB_LENNOX, M9
from torchcell.datamodels.schema import (
    AssayType,
    BacterialDeletionPerturbation,
    BacterialEnvironmentResponseExperiment,
    BacterialEnvironmentResponseExperimentReference,
    Concentration,
    ConcentrationUnit,
    DerivedIdentifierMapping,
    DoseBasis,
    Environment,
    EnvironmentPerturbationType,
    EnvironmentPhysicalPerturbation,
    EnvironmentResponsePhenotype,
    Experiment,
    ExperimentReference,
    Genotype,
    MeasurementType,
    Media,
    MediaComponent,
    MediaComponentRole,
    PhysicalFactor,
    Publication,
    SmallMoleculePerturbation,
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

DOI = "10.1371/journal.pgen.1006124"
PMCID = "PMC4927156"
TITLE = (
    "A Chemical-Genomic Screen of Neglected Antibiotics Reveals Illicit Transport of "
    "Kasugamycin and Blasticidin S"
)

CITATION_KEY = "shiverChemicalGenomicScreenNeglected2016"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"

DATA_FILENAME = "pgen.1006124.s005.txt"
DATA_REL = f"data/{DATA_FILENAME}"
DATA_SHA256 = "9edcf6f6f34b957661b23f2a5e7f6f14e25c387ab825e0b4c79a5f995f8de690"
DATA_RETRIEVED_AT = "2026-10-07"
#: The PMC Article Datasets bucket key of S1 Dataset (article version 1).
PMC_CLOUD_KEY = f"{PMCID}.1/{DATA_FILENAME}"
DATA_URL = pmc_cloud_url(PMC_CLOUD_KEY)

PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "4a1bec97d10f6ef1c02c99329f7adce0d79623819fcf421c098058a7c82ced72"

#: S1 Table in the literature mirror, the parse cross-check of the module docstring.
S1_TABLE_REL = "si/si6.docx"
S1_TABLE_SHA256 = "6521f4ef4242e3e66781713a4fea2ef85342dca3077d26927d24cc72a2fab3e2"

#: The Dryad deposit of the plate images and Iris output behind the released scores.
DRYAD_DOI = "10.5061/dryad.f3kc0"

#: The S1 Dataset column heading of the condition labels.
CONDITION_COLUMN = "Condition"
#: Columns of the release: one condition label plus 3,975 gene labels.
RELEASED_COLUMNS = 3976
#: Condition rows of the release: 235 Nichols 2011 (batch 0) plus this study's 57.
RELEASED_CONDITION_ROWS = 292

#: This study's own batches. "The batch groups conditions that were measured in the same
#: experiment and normalized as a group. Conditions from Nichols et al. [8] were
#: assigned batch '0'."
STUDY_BATCHES: tuple[int, ...] = (1, 4)

#: Fraction of the release's distinct gene labels that must resolve to one BW25113 locus
#: (the bacterial checklist's item 4). Below it the build stops and reports instead of
#: dropping strains. Measured on the release: 3,731 of 3,963 (0.954).
MIN_RESOLVED_FRACTION = 0.95

KEIO_COLLECTION = "KEIO deletion library"

UNITS = (
    "fitness-score: the Collins/Nichols S-score of the condition, a standardized "
    "(modified-t) deviation of Iris colony opacity; negative = sensitivity, positive = "
    "resistance, 0 = no change in colony size"
)
UNITS_REFERENCE = "the unaffected strain of this condition (fitness-score 0)"


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


_MEDIA = "Methods, 'Media, growth conditions, strains, plasmids, and oligos'"
_COLLECTION = "Methods, 'Data collection and processing'"
_ANALYSIS = "Methods, 'Clustering, significant phenotypes and correlations, and network analysis'"
_SCREEN_RESULTS = (
    "Results, 'The chemical-genomic screen substantially expands known connections in "
    "E. coli'"
)
_S1_DATASET = "Supporting Information, 'S1 Dataset'"

LIBRARY_STRAIN = _paper(
    "BW25113",
    "The KEIO deletion library is derived from BW25113 (F- λ- Δ(araD-araB)657 "
    "ΔlacZ4787(::rrnB-3) rph-1 Δ(rhaD-rhaB)568 hsdR514) [60]",
    page=_MEDIA,
    note="the screened library's background, and so the assembly every record pins; the "
    "MG1655 strains of the same paragraph are the follow-up experiments, not the screen",
)
SCREEN_SIZE = _paper(
    (3975, 57),
    "We tested the sensitivities of 3975 mutants of E. coli K-12 to 57 stresses, split "
    "between new and previously screened conditions.",
    page=_SCREEN_RESULTS,
    note="3,975 equals the gene columns of S1 Dataset and 57 the condition rows of "
    "batches 1 and 4; their product, 226,575, is the released cell count",
)
FITNESS_SCORE_MEANING = _paper(
    "signed significance of a colony-size change",
    "These fitness-scores represent the statistical significance of a change in colony "
    "size for a particular condition, with negative and positive fitness-scores "
    "representing sensitivity and resistance, respectively.",
    page=_SCREEN_RESULTS,
    note="why the readout is an EnvironmentResponsePhenotype and not a FitnessPhenotype: "
    "the number is signed, and 0 is the unaffected baseline",
)
SCORE_PIPELINE = _paper(
    "Collins/Nichols S-score pipeline",
    "The chemical genomics screen was conducted using the same methodology as reported "
    "previously [8] with few modifications.",
    page=_COLLECTION,
    note="[8] is Nichols et al. 2011 (Cell 144:143), which builds on [18] Collins et "
    "al. 2006 (Genome Biol 7:R63); the S-score is a modified t-statistic, which is why "
    "measurement_type is z_score and not log2_ratio. Neither cited paper is in the "
    "literature mirror, so the deferral chain is the citation",
)
OPACITY_READOUT = _paper(
    "Iris colony opacity",
    "Images were analyzed using Iris to measure the total intensity of pix els within "
    "the colony to calculate an opacity metric.",
    page=_COLLECTION,
    note="the quantified metric is total pixel intensity (opacity), not colony area; "
    "AssayType.colony_size_array is the typed pinned-array design it belongs to",
)
PIPELINE_STEPS = _paper(
    "filter, quartic surface normalization, power transform, variance normalization",
    "Steps added to the original analysis pipeline [18] include simultaneous input and "
    "analysis of colony size, opacity, and circularity from Iris (read_data.m), manual "
    "removal of data based on plate position (to eliminate false positives from minor "
    "pinning problems) (filter_data.m), higher-order surface normalization",
    page=_COLLECTION,
)
REPLICATES_NOT_REPORTED = _paper(
    "per-condition measurement counts, not per-record replicates",
    "Unreliable measurements were removed from the dataset at multiple points in the "
    "analysis and each condition had a different number of measurements "
    "(fitness-scores) that passed analysis.",
    page=_ANALYSIS,
    note="the only statement about measurement counts in the paper or its SI; it counts "
    "fitness-scores per condition, not colonies per fitness-score, so n_samples stays a "
    "typed gap deferred to Nichols 2011",
)
FDR_CUTOFF_BAND = _paper(
    ((-2.0, -1.2), (1.2, 2.1)),
    "Using this method, $9 5 \\%$ of the cutoff values for negative (sensitization) "
    "fitness-scores fell in the range (-2.0,-1.2) while $9 5 \\%$ of the cutoff values "
    "for positive (resistance) fitness-scores fell in the range $( + 1 . 2 , + 2 . 1 )$",
    page=_ANALYSIS,
    note="the per-condition FDR 5% hit thresholds; a property of the score's scale, not "
    "a per-record uncertainty, and the per-condition values are not released",
)
LB_RECIPE = _paper(
    (
        "1% (w/v) tryptone",
        "0.5% (w/v) yeast extract",
        "90 mM sodium chloride",
        "2% (w/v) bacto agar",
    ),
    "Chemical sensitivity screens used LB Lennox agar plates $1 \\%$ (w/v) tryptone, "
    "$0 . 5 \\%$ (w/v) yeast extract, $9 0 \\mathrm { m M }$ sodium chloride, $2 \\%$ "
    "(w/v) bacto agar) unless otherwise specified.",
    page=_MEDIA,
    note="the default screen plate; 1% (w/v) tryptone is 10 g/L and 0.5% (w/v) yeast "
    "extract 5 g/L, the Lennox amounts the shared LB_LENNOX carries, while the salt is "
    "stated as 90 mM rather than Lennox's 5 g/L (85.6 mM), so the salt is restated here",
)
M9_RECIPE = _paper(
    ("M9 salts", "0.2% (w/v) glucose", "2% (w/v) bacto agar"),
    "M9 minimal plates used in the screen contained M9 salts, $0 . 2 \\%$ (w/v) "
    "glucose, and $2 \\%$ $\\scriptstyle \\left( \\mathbf { w } / \\mathbf { v } "
    "\\right)$ bacto agar.",
    page=_MEDIA,
    note="the plate behind every 'M9min' condition label, and the source of the 0.2% "
    "(w/v) glucose carbon source the M9min conditions carry unless their own label "
    "names another carbon source",
)
TEMPERATURE_C = _paper(
    37.0,
    "Ordered libraries grown during the chemical-genomics screens were incubated at "
    "$3 7 ^ { \\circ } \\mathrm { C }$ until the majority of colonies reached a defined "
    "size ( ${ \\sim } 8$ hours) then a photograph was taken of the plate.",
    page=_MEDIA,
    note="the default incubation temperature; the 10 C, 25 C and 4 C conditions state "
    "their own in the released condition label",
)
DURATION_IS_A_SIZE_RULE = _paper(
    "endpoint is a colony-size rule, not a fixed time",
    "until the majority of colonies reached a defined size ( ${ \\sim } 8$ hours)",
    page=_MEDIA,
    note="why duration_hours is a typed gap on 56 of the 57 conditions: the incubation "
    "ends at a size, the paper gives only an approximate 8 hours for the 37 C plates, "
    "and the cold conditions necessarily ran longer",
)
SURVIVAL_4C = _paper(
    (4.0, 840.0),
    "One condition, $4 ^ { \\circ } \\mathrm { C }$ survival, involved growth of "
    "colonies on LB plates at $3 7 ^ { \\circ } \\mathrm { C }$ for 6 hours, and "
    "transfer of the colony array to $4 ^ { \\circ } \\mathrm { C }$ for 5 weeks.",
    page=_COLLECTION,
    note="the one condition whose exposure is released: 5 weeks at 4 C is 840 h, and "
    "the released label agrees ('4C survival [5 wk] {4}')",
)
AEROBICITY = _paper(
    "aerobic",
    "Ordered libraries were arrayed on rectangular agar plates, grown in the presence "
    "of antibiotic or other stress until the colonies reached a defined average size, "
    "and then imaged.",
    page=_COLLECTION,
    note="agar plates incubated in air; no chamber or gas control is described, and the "
    "release's only anaerobic condition sits in batch 0 (Nichols 2011), outside the 57",
)
BATCH_MEANING = _paper(
    STUDY_BATCHES,
    "The batch groups conditions that were measured in the same experiment and "
    "normalized as a group. Conditions from Nichols et al. [8] were assigned batch "
    "$^ { \\alpha } 0 ^ { \\gamma }$ .",
    page=_S1_DATASET,
    note="batch 0 is Nichols 2011 re-analyzed and is excluded; batches 1 and 4 are this "
    "study's 57 conditions. Independent normalization per batch is why the batch is part "
    "of the screen identity, and the verbatim label is stored as screen_id",
)
LABELS_ARE_GENE_NAMES = _paper(
    "gene names",
    "Gene names are used to label the mutation. Unless otherwise specified, the "
    "mutations are precise gene deletions.",
    page=_S1_DATASET,
    note="the identifier space of the columns (hence route='gene_symbol'), and the rule "
    "that drops the suffixed non-deletion alleles: they ARE otherwise specified",
)
MATRIX_ORIENTATION = _paper(
    "genes in columns, conditions in rows",
    "A table of fitness scores for genes (columns) by conditions (rows) suitable for "
    "clustering [62] and other downstream analyses.",
    page=_S1_DATASET,
)
DATA_AVAILABILITY = _paper(
    DRYAD_DOI,
    "Raw data associated with colony size and GSEA analysis are available at Dryad "
    "(http://dx.doi. org/10.5061/dryad.f3kc0).",
    page="front matter, 'Data Availability Statement'",
    note="the plate images and Iris output behind the scores; not needed, because S1 "
    "Dataset releases the scored matrix, and not mirrored",
)
FOLLOWUP_CASSETTE = _paper(
    "kanamycin resistance cassette amplified from pKD4",
    "Operon deletions were generated using $\\lambda$ -red recombineering to replace "
    "the operon with a kanamycin resistance cassette amplified from pKD4.",
    page=_MEDIA,
    note="the cassette of the MG1655 operon deletions built AFTER the screen; the paper "
    "never states the arrayed library's cassette, so the perturbations leave it None",
)

#: Every module-level sourced value, for the mirror audit.
SOURCED_VALUES: tuple[SourcedValue, ...] = (
    LIBRARY_STRAIN,
    SCREEN_SIZE,
    FITNESS_SCORE_MEANING,
    SCORE_PIPELINE,
    OPACITY_READOUT,
    PIPELINE_STEPS,
    REPLICATES_NOT_REPORTED,
    FDR_CUTOFF_BAND,
    LB_RECIPE,
    M9_RECIPE,
    TEMPERATURE_C,
    DURATION_IS_A_SIZE_RULE,
    SURVIVAL_4C,
    AEROBICITY,
    BATCH_MEANING,
    LABELS_ARE_GENE_NAMES,
    MATRIX_ORIENTATION,
    DATA_AVAILABILITY,
    FOLLOWUP_CASSETTE,
)


# --------------------------------------------------------------------------- #
# Media
# --------------------------------------------------------------------------- #
def _agar_component() -> MediaComponent:
    """2% (w/v) bacto agar, the gelling agent of both screen plates."""
    return MediaComponent(
        compound=resolved_compound("agar"),
        role=MediaComponentRole.gelling_agent,
        concentration=Concentration(value=2.0, unit=ConcentrationUnit.percent_w_v),
        provenance=[LB_RECIPE, M9_RECIPE],
        note="the release calls it bacto agar; the shared compound row is agar",
    )


#: The default screen plate. ``LB_LENNOX`` is the shared recipe (tryptone 10 g/L, yeast
#: extract 5 g/L, the amounts Shiver states as 1% and 0.5% (w/v)); Shiver's salt is
#: stated as 90 mM, so the library's 5 g/L sodium-chloride component is replaced by the
#: stated molarity rather than silently reused.
SHIVER2016_LB_LENNOX_AGAR = Media(
    name="LB Lennox agar, 90 mM NaCl, 2% (w/v) bacto agar (Shiver 2016)",
    state="solid",
    is_synthetic=False,
    base_medium="LB_LENNOX",
    components=[
        *(
            component
            for component in LB_LENNOX.components
            if component.compound.name != "sodium chloride"
        ),
        MediaComponent(
            compound=resolved_compound("sodium chloride"),
            role=MediaComponentRole.bulk_salt,
            concentration=Concentration(value=90.0, unit=ConcentrationUnit.millimolar),
            provenance=[LB_RECIPE],
            note="Shiver states 90 mM, not the 5 g/L (85.6 mM) of the shared Lennox row",
        ),
        _agar_component(),
    ],
    provenance=[LB_RECIPE],
)

#: The plate behind every "M9min" condition. ``M9`` is the shared salts-only recipe; the
#: carbon source is the varied factor, so it is an ``EnvironmentPhysicalPerturbation``
#: rather than a component (the Tong 2020 convention).
SHIVER2016_M9_MINIMAL_AGAR = Media(
    name="M9 minimal agar, carbon source varied, 2% (w/v) bacto agar (Shiver 2016)",
    state="solid",
    is_synthetic=True,
    base_medium="M9",
    components=[*M9.components, _agar_component()],
    provenance=[M9_RECIPE],
)

MediaKey = Literal["lb_lennox_agar", "m9_minimal_agar"]
SCREEN_MEDIA: dict[MediaKey, Media] = {
    "lb_lennox_agar": SHIVER2016_LB_LENNOX_AGAR,
    "m9_minimal_agar": SHIVER2016_M9_MINIMAL_AGAR,
}


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """``DATA_ROOT`` from the environment (the mirror and build tree live under it)."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/shiverChemicalGenomicScreenNeglected2016``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def _sha256(path: str | Path) -> str:
    """Streaming sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def s1_dataset_retrieval(retrieved_at: str = DATA_RETRIEVED_AT) -> RetrievalRecord:
    """The recorded retrieval of S1 Dataset: one PMC Article Datasets object.

    ``run_retriever`` on this record re-fetches the bytes; the pinned sha256 is the
    anchor a rebuild verifies against. Re-fetched 2026-10-07 and byte-identical to the
    copy the literature mirror holds under the same key.
    """
    return RetrievalRecord(
        method=RetrievalMethod.pmc_cloud,
        source_url=DATA_URL,
        retriever="torchcell.literature.retrieve.pmc_cloud_object",
        params={"key": PMC_CLOUD_KEY},
        sha256=DATA_SHA256,
        retrieved_at=retrieved_at,
    )


def deposit_raw_mirror(
    *,
    data_path: str | Path,
    retrieved_at: str = DATA_RETRIEVED_AT,
    data_root: str | None = None,
) -> Path:
    """Write the raw mirror from the retrieved S1 Dataset plus its ``manifest.json``.

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
                retrieval=s1_dataset_retrieval(retrieved_at),
            )
        ],
        si_data_sources=[DATA_URL],
        si_expected=[
            "S1 Dataset (pgen.1006124.s005.txt): the integrated fitness-score matrix, "
            "3,975 gene columns by 292 condition rows; the 57 rows of batches 1 and 4 "
            "are this study's screen and are what the loader consumes",
            "S1 Table (pgen.1006124.s006.docx): 'Cold-sensitive genes from the screen', "
            "whose 10 C fitness-score column is the loader's parse cross-check. Already "
            f"mirrored under the literature key as {S1_TABLE_REL} "
            f"(sha256 {S1_TABLE_SHA256}), so it is NOT duplicated here",
            f"Dryad {DRYAD_DOI} (plate images, Iris output, GSEA inputs): the raw "
            "material behind the released scores. NOT mirrored, because S1 Dataset "
            "releases the scored matrix the loader needs",
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
    """The recorded sha256 of one mirror file."""
    for record in manifest.files:
        if record.path == relpath:
            return record.sha256
    raise KeyError(f"{relpath} is not in the raw-mirror manifest")


# --------------------------------------------------------------------------- #
# The 57 conditions, each typed from its released label plus the Methods
# --------------------------------------------------------------------------- #
#: Every dose unit the release uses, and the ``ConcentrationUnit`` plus multiplier it is
#: stored as. The three mass-per-volume units collapse onto the one typed ``ug/mL`` by
#: exact decimal conversion, so no ``ConcentrationUnit`` member is added and the
#: deliberately deferred ``mg/L`` is not needed (1 ug/mL IS 1 mg/L).
UNIT_CONVERSIONS: dict[str, tuple[ConcentrationUnit, float]] = {
    "ug/mL": (ConcentrationUnit.ug_per_ml, 1.0),
    "ng/mL": (ConcentrationUnit.ug_per_ml, 1e-3),
    "mg/mL": (ConcentrationUnit.ug_per_ml, 1e3),
    "mM": (ConcentrationUnit.millimolar, 1.0),
    "uM": (ConcentrationUnit.micromolar, 1.0),
    "% (w/v)": (ConcentrationUnit.percent_w_v, 1.0),
    "% (v/v)": (ConcentrationUnit.percent_v_v, 1.0),
}

SourceUnit = Literal["ug/mL", "ng/mL", "mg/mL", "mM", "uM", "% (w/v)", "% (v/v)"]


class Dose(BaseModel):
    """One released dose: the source's own agent label, number and unit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    value: float
    unit: SourceUnit

    @property
    def concentration(self) -> Concentration:
        """The dose as a typed ``Concentration``, converted per :data:`UNIT_CONVERSIONS`."""
        unit, multiplier = UNIT_CONVERSIONS[self.unit]
        return Concentration(
            value=self.value * multiplier, unit=unit, basis=DoseBasis.fixed
        )


#: The 0.2% (w/v) glucose of the screen's M9 minimal plate recipe, the carbon source of
#: every "M9min" condition whose own label does not name another one.
M9_GLUCOSE_DOSE = Dose(label="glucose", value=0.2, unit="% (w/v)")


class ConditionSpec(BaseModel):
    """One of the 57 screen conditions, typed from its label and the Methods.

    ``label`` is the released condition label verbatim (name, dose in square brackets,
    batch in curly brackets); it is also the ``screen_id`` every record of the condition
    carries, which is what keeps two independently normalized screens of one compound at
    one dose distinct (``gliotoxin-A`` and ``gliotoxin-B`` in batch 1; ``EDTA [1 mM]``
    in batches 1 and 4).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    batch: int
    base: MediaKey
    temperature_c: float = 37.0
    duration_hours: float | None = None
    carbon_source: Dose | None = None
    small_molecules: tuple[Dose, ...] = ()
    irradiated: bool = False

    @model_validator(mode="after")
    def _check(self) -> ConditionSpec:
        """The label states this batch, and only an M9 plate names a carbon source."""
        if not self.label.endswith(f"{{{self.batch}}}"):
            raise ValueError(f"{self.label!r} does not end with batch {self.batch}")
        if self.batch not in STUDY_BATCHES:
            raise ValueError(f"{self.label!r}: batch {self.batch} is not this study's")
        if (self.base == "m9_minimal_agar") != (self.carbon_source is not None):
            raise ValueError(
                f"{self.label!r}: an M9 minimal plate names its carbon source and an LB "
                "plate does not"
            )
        return self


#: The 57 conditions of batches 1 and 4, in the release's own row order. Each row is
#: transcribed from the released label; the medium, the default temperature and the 4 C
#: exposure come from the Methods quotes above. The "M9min" prefix is what selects the M9
#: minimal plate, and a "M9min <carbon source> [<amount>]" label is read as that carbon
#: source at that amount, the reading the release's own
#: "M9min glucose [0.2% (w/v)]" rows confirm against the stated plate recipe.
CONDITIONS: tuple[ConditionSpec, ...] = (
    ConditionSpec(
        label="A22 [5 ug/mL] {1}",
        batch=1,
        base="lb_lennox_agar",
        small_molecules=(Dose(label="A22", value=5.0, unit="ug/mL"),),
    ),
    ConditionSpec(
        label="D,L-serine hydroxamate [600 ug/mL] {1}",
        batch=1,
        base="lb_lennox_agar",
        small_molecules=(
            Dose(label="D,L-serine hydroxamate", value=600.0, unit="ug/mL"),
        ),
    ),
    ConditionSpec(
        label="DMSO [9.5% (v/v)] {1}",
        batch=1,
        base="lb_lennox_agar",
        small_molecules=(Dose(label="DMSO", value=9.5, unit="% (v/v)"),),
    ),
    ConditionSpec(
        label="EDTA [1 mM] {1}",
        batch=1,
        base="lb_lennox_agar",
        small_molecules=(Dose(label="EDTA", value=1.0, unit="mM"),),
    ),
    ConditionSpec(
        label="M9min 5-fluorouridine [250 ng/mL] {1}",
        batch=1,
        base="m9_minimal_agar",
        carbon_source=M9_GLUCOSE_DOSE,
        small_molecules=(Dose(label="5-fluorouridine", value=250.0, unit="ng/mL"),),
    ),
    ConditionSpec(
        label="M9min 5-methylanthranilic acid [20 ug/mL] {1}",
        batch=1,
        base="m9_minimal_agar",
        carbon_source=M9_GLUCOSE_DOSE,
        small_molecules=(
            Dose(label="5-methylanthranilic acid", value=20.0, unit="ug/mL"),
        ),
    ),
    ConditionSpec(
        label="M9min 5-methyltryptophan [20 ug/mL] {1}",
        batch=1,
        base="m9_minimal_agar",
        carbon_source=M9_GLUCOSE_DOSE,
        small_molecules=(Dose(label="5-methyltryptophan", value=20.0, unit="ug/mL"),),
    ),
    ConditionSpec(
        label="M9min 7-azatryptophan [14 ug/mL] {1}",
        batch=1,
        base="m9_minimal_agar",
        carbon_source=M9_GLUCOSE_DOSE,
        small_molecules=(Dose(label="7-azatryptophan", value=14.0, unit="ug/mL"),),
    ),
    ConditionSpec(
        label="M9min casamino acids [0.4% (w/v)] {1}",
        batch=1,
        base="m9_minimal_agar",
        carbon_source=M9_GLUCOSE_DOSE,
        small_molecules=(Dose(label="casamino acids", value=0.4, unit="% (w/v)"),),
    ),
    ConditionSpec(
        label="M9min glucose-A [0.2% (w/v)] {1}",
        batch=1,
        base="m9_minimal_agar",
        carbon_source=M9_GLUCOSE_DOSE,
    ),
    ConditionSpec(
        label="M9min glucose-B [0.2% (w/v)] {1}",
        batch=1,
        base="m9_minimal_agar",
        carbon_source=M9_GLUCOSE_DOSE,
    ),
    ConditionSpec(
        label="SDS+EDTA [0.5% (w/v); 500 uM] {1}",
        batch=1,
        base="lb_lennox_agar",
        small_molecules=(
            Dose(label="SDS", value=0.5, unit="% (w/v)"),
            Dose(label="EDTA", value=500.0, unit="uM"),
        ),
    ),
    ConditionSpec(
        label="SDS [1% (w/v)] {1}",
        batch=1,
        base="lb_lennox_agar",
        small_molecules=(Dose(label="SDS", value=1.0, unit="% (w/v)"),),
    ),
    ConditionSpec(
        label="acriflavine [10 ug/mL] {1}",
        batch=1,
        base="lb_lennox_agar",
        small_molecules=(Dose(label="acriflavine", value=10.0, unit="ug/mL"),),
    ),
    ConditionSpec(
        label="amoxicillin [1.5 ug/mL] {1}",
        batch=1,
        base="lb_lennox_agar",
        small_molecules=(Dose(label="amoxicillin", value=1.5, unit="ug/mL"),),
    ),
    ConditionSpec(
        label="ampicillin [4 ug/mL] {1}",
        batch=1,
        base="lb_lennox_agar",
        small_molecules=(Dose(label="ampicillin", value=4.0, unit="ug/mL"),),
    ),
    ConditionSpec(
        label="azelaic acid [1 mg/mL] {1}",
        batch=1,
        base="lb_lennox_agar",
        small_molecules=(Dose(label="azelaic acid", value=1.0, unit="mg/mL"),),
    ),
    ConditionSpec(
        label="bicyclomycin [20 ug/mL] {1}",
        batch=1,
        base="lb_lennox_agar",
        small_molecules=(Dose(label="bicyclomycin", value=20.0, unit="ug/mL"),),
    ),
    ConditionSpec(
        label="bile salts [2% (w/v)] {1}",
        batch=1,
        base="lb_lennox_agar",
        small_molecules=(Dose(label="bile salts", value=2.0, unit="% (w/v)"),),
    ),
    ConditionSpec(
        label="blasticidin S [33 ug/mL] {1}",
        batch=1,
        base="lb_lennox_agar",
        small_molecules=(Dose(label="blasticidin S", value=33.0, unit="ug/mL"),),
    ),
    ConditionSpec(
        label="gliotoxin-A [10 ug/mL] {1}",
        batch=1,
        base="lb_lennox_agar",
        small_molecules=(Dose(label="gliotoxin", value=10.0, unit="ug/mL"),),
    ),
    ConditionSpec(
        label="gliotoxin-B [10 ug/mL] {1}",
        batch=1,
        base="lb_lennox_agar",
        small_molecules=(Dose(label="gliotoxin", value=10.0, unit="ug/mL"),),
    ),
    ConditionSpec(
        label="holomycin [2.5 ug/mL] {1}",
        batch=1,
        base="lb_lennox_agar",
        small_molecules=(Dose(label="holomycin", value=2.5, unit="ug/mL"),),
    ),
    ConditionSpec(
        label="isopropanol [5% (v/v)] {1}",
        batch=1,
        base="lb_lennox_agar",
        small_molecules=(Dose(label="isopropanol", value=5.0, unit="% (v/v)"),),
    ),
    ConditionSpec(
        label="kasugamycin [20 ug/mL] {1}",
        batch=1,
        base="lb_lennox_agar",
        small_molecules=(Dose(label="kasugamycin", value=20.0, unit="ug/mL"),),
    ),
    ConditionSpec(
        label="n-butanol [1% (v/v)] {1}",
        batch=1,
        base="lb_lennox_agar",
        small_molecules=(Dose(label="n-butanol", value=1.0, unit="% (v/v)"),),
    ),
    ConditionSpec(
        label="rifampicin [4 ug/mL] {1}",
        batch=1,
        base="lb_lennox_agar",
        small_molecules=(Dose(label="rifampicin", value=4.0, unit="ug/mL"),),
    ),
    ConditionSpec(
        label="t-butanol [5% (v/v)] {1}",
        batch=1,
        base="lb_lennox_agar",
        small_molecules=(Dose(label="t-butanol", value=5.0, unit="% (v/v)"),),
    ),
    ConditionSpec(
        label="thiolutin [7 ug/mL] {1}",
        batch=1,
        base="lb_lennox_agar",
        small_molecules=(Dose(label="thiolutin", value=7.0, unit="ug/mL"),),
    ),
    ConditionSpec(
        label="urea [750 mM] {1}",
        batch=1,
        base="lb_lennox_agar",
        small_molecules=(Dose(label="urea", value=750.0, unit="mM"),),
    ),
    ConditionSpec(
        label="10C [-] {4}", batch=4, base="lb_lennox_agar", temperature_c=10.0
    ),
    ConditionSpec(
        label="25C [-] {4}", batch=4, base="lb_lennox_agar", temperature_c=25.0
    ),
    ConditionSpec(
        label="EDTA [1 mM] {4}",
        batch=4,
        base="lb_lennox_agar",
        small_molecules=(Dose(label="EDTA", value=1.0, unit="mM"),),
    ),
    ConditionSpec(
        label="M9min 7-azatryptophan [5 ug/mL] {4}",
        batch=4,
        base="m9_minimal_agar",
        carbon_source=M9_GLUCOSE_DOSE,
        small_molecules=(Dose(label="7-azatryptophan", value=5.0, unit="ug/mL"),),
    ),
    ConditionSpec(
        label="M9min acetate [0.6% (w/v)] {4}",
        batch=4,
        base="m9_minimal_agar",
        carbon_source=Dose(label="acetate", value=0.6, unit="% (w/v)"),
    ),
    ConditionSpec(
        label="M9min glucose+UV [0.2% (w/v); 12 sec] {4}",
        batch=4,
        base="m9_minimal_agar",
        carbon_source=M9_GLUCOSE_DOSE,
        irradiated=True,
    ),
    ConditionSpec(
        label="M9min glucose [0.2% (w/v)] {4}",
        batch=4,
        base="m9_minimal_agar",
        carbon_source=M9_GLUCOSE_DOSE,
    ),
    ConditionSpec(
        label="SDS [1% (w/v)] {4}",
        batch=4,
        base="lb_lennox_agar",
        small_molecules=(Dose(label="SDS", value=1.0, unit="% (w/v)"),),
    ),
    ConditionSpec(
        label="UV+10C [12 sec] {4}",
        batch=4,
        base="lb_lennox_agar",
        temperature_c=10.0,
        irradiated=True,
    ),
    ConditionSpec(
        label="UV [12 sec] {4}", batch=4, base="lb_lennox_agar", irradiated=True
    ),
    ConditionSpec(
        label="ampicillin [4 ug/mL] {4}",
        batch=4,
        base="lb_lennox_agar",
        small_molecules=(Dose(label="ampicillin", value=4.0, unit="ug/mL"),),
    ),
    ConditionSpec(
        label="chlorhexidine dihydrochloride [5 ug/mL] {4}",
        batch=4,
        base="lb_lennox_agar",
        small_molecules=(
            Dose(label="chlorhexidine dihydrochloride", value=5.0, unit="ug/mL"),
        ),
    ),
    ConditionSpec(
        label="cinoxacin [3 ug/mL] {4}",
        batch=4,
        base="lb_lennox_agar",
        small_molecules=(Dose(label="cinoxacin", value=3.0, unit="ug/mL"),),
    ),
    ConditionSpec(
        label="ciprofloxacin [1 ng/mL] {4}",
        batch=4,
        base="lb_lennox_agar",
        small_molecules=(Dose(label="ciprofloxacin", value=1.0, unit="ng/mL"),),
    ),
    ConditionSpec(
        label="clindamycin [64 ug/mL] {4}",
        batch=4,
        base="lb_lennox_agar",
        small_molecules=(Dose(label="clindamycin", value=64.0, unit="ug/mL"),),
    ),
    ConditionSpec(
        label="copper(II) [2 mM] {4}",
        batch=4,
        base="lb_lennox_agar",
        small_molecules=(Dose(label="copper(II)", value=2.0, unit="mM"),),
    ),
    ConditionSpec(
        label="deoxycholate [1% (w/v)] {4}",
        batch=4,
        base="lb_lennox_agar",
        small_molecules=(Dose(label="deoxycholate", value=1.0, unit="% (w/v)"),),
    ),
    ConditionSpec(
        label="guanidine hydrochloride [30 mM] {4}",
        batch=4,
        base="lb_lennox_agar",
        small_molecules=(Dose(label="guanidine hydrochloride", value=30.0, unit="mM"),),
    ),
    ConditionSpec(
        label="isopentanol [0.5% (v/v)] {4}",
        batch=4,
        base="lb_lennox_agar",
        small_molecules=(Dose(label="isopentanol", value=0.5, unit="% (v/v)"),),
    ),
    ConditionSpec(
        label="phenol [0.1% (v/v)] {4}",
        batch=4,
        base="lb_lennox_agar",
        small_molecules=(Dose(label="phenol", value=0.1, unit="% (v/v)"),),
    ),
    ConditionSpec(
        label="pseudomonic acid A [36 ug/mL] {4}",
        batch=4,
        base="lb_lennox_agar",
        small_molecules=(Dose(label="pseudomonic acid A", value=36.0, unit="ug/mL"),),
    ),
    ConditionSpec(
        label="pyocyanin [10 ug/mL] {4}",
        batch=4,
        base="lb_lennox_agar",
        small_molecules=(Dose(label="pyocyanin", value=10.0, unit="ug/mL"),),
    ),
    ConditionSpec(
        label="silver(II) [1 uM] {4}",
        batch=4,
        base="lb_lennox_agar",
        small_molecules=(Dose(label="silver(II)", value=1.0, unit="uM"),),
    ),
    ConditionSpec(
        label="sodium fluoride [100 mM] {4}",
        batch=4,
        base="lb_lennox_agar",
        small_molecules=(Dose(label="sodium fluoride", value=100.0, unit="mM"),),
    ),
    ConditionSpec(
        label="tetracycline [500 ng/mL] {4}",
        batch=4,
        base="lb_lennox_agar",
        small_molecules=(Dose(label="tetracycline", value=500.0, unit="ng/mL"),),
    ),
    ConditionSpec(
        label="urea [320 mM] {4}",
        batch=4,
        base="lb_lennox_agar",
        small_molecules=(Dose(label="urea", value=320.0, unit="mM"),),
    ),
    ConditionSpec(
        label="4C survival [5 wk] {4}",
        batch=4,
        base="lb_lennox_agar",
        temperature_c=4.0,
        duration_hours=840.0,
    ),
)

CONDITION_LABELS: tuple[str, ...] = tuple(spec.label for spec in CONDITIONS)


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
_DURATION_GAP = ProvenanceGap(
    field="duration_hours",
    reason=ProvenanceGapReason.not_reported_by_primary,
    note="the screen's endpoint is a colony-size rule, not a fixed time ('until the "
    "majority of colonies reached a defined size (~8 hours)'), the per-condition "
    "incubation is not released, and the cold conditions necessarily ran longer",
)
_RADIATION_GAP = ProvenanceGap(
    field="magnitude",
    reason=ProvenanceGapReason.not_reported_by_primary,
    note="the released label gives the UV dose as an exposure TIME ('12 sec') and the "
    "paper reports no irradiance, so no fluence exists to store; the exposure survives "
    "verbatim in the phenotype's screen_id",
)


def small_molecule(dose: Dose) -> SmallMoleculePerturbation:
    """One dosed chemical, its identity resolved through the shared compound layer.

    A label with no curated row comes back carrying a typed gap on ``inchikey``; no
    structure is ever guessed from a near-miss name.
    """
    return SmallMoleculePerturbation(
        compound=resolved_compound(dose.label), concentration=dose.concentration
    )


def carbon_source(dose: Dose) -> EnvironmentPhysicalPerturbation:
    """The M9 plate's carbon source as the varied physical factor."""
    return EnvironmentPhysicalPerturbation(
        factor=PhysicalFactor.carbon_source,
        agent=resolved_compound(dose.label),
        magnitude=dose.concentration,
    )


def radiation() -> EnvironmentPhysicalPerturbation:
    """The UV exposure: a radiation factor with no storable dose."""
    return EnvironmentPhysicalPerturbation(
        factor=PhysicalFactor.radiation, provenance_gaps=[_RADIATION_GAP]
    )


def environment(spec: ConditionSpec) -> Environment:
    """The screen plate of one condition."""
    perturbations: list[EnvironmentPerturbationType] = []
    if spec.carbon_source is not None:
        perturbations.append(carbon_source(spec.carbon_source))
    perturbations.extend(small_molecule(dose) for dose in spec.small_molecules)
    if spec.irradiated:
        perturbations.append(radiation())
    return Environment(
        media=SCREEN_MEDIA[spec.base],
        temperature=Temperature(value=spec.temperature_c),
        perturbations=perturbations,
        aerobicity=str(AEROBICITY.value),
        duration_hours=spec.duration_hours,
        provenance_gaps=[] if spec.duration_hours is not None else [_DURATION_GAP],
    )


# --------------------------------------------------------------------------- #
# Phenotype
# --------------------------------------------------------------------------- #
_NICHOLS2011 = Provenance(
    source_uri="https://doi.org/10.1016/j.cell.2010.11.052",
    citation_key="nicholsPhenotypicLandscapeBacterial2011",
    method="the screen's own deferral: 'the same methodology as reported previously "
    "[8]'; [8]'s Extended Experimental Procedures hold the colony replicate design "
    "behind one fitness-score, and defer the score itself to Collins 2006 "
    "(doi:10.1186/gb-2006-7-7-r63)",
    page="Cell 144(1):143-156, Extended Experimental Procedures",
)
_GAP_N_SAMPLES = ProvenanceGap(
    field="n_samples",
    reason=ProvenanceGapReason.deferred_pending_source_review,
    resolve_with=_NICHOLS2011,
    note="the number of colonies behind one fitness-score is in neither this paper nor "
    "its SI; its only measurement-count statement counts fitness-scores per condition, "
    "not colonies per fitness-score",
)
_GAP_SAMPLE_UNIT = ProvenanceGap(
    field="sample_unit",
    reason=ProvenanceGapReason.deferred_pending_source_review,
    resolve_with=_NICHOLS2011,
    note="whether the replicates behind a fitness-score are colonies on one plate or "
    "independent plates follows from the same deferred methodology",
)
_GAP_UNCERTAINTY = ProvenanceGap(
    field="environment_response_uncertainty",
    reason=ProvenanceGapReason.not_reported_by_primary,
    note="S1 Dataset releases one fitness-score per cell and no dispersion; the score "
    "is itself a significance statistic, and the paper's dispersion statement is the "
    "per-condition FDR 5% cutoff band, not a per-cell error",
)
_GAP_SE = ProvenanceGap(
    field="environment_response_se",
    reason=ProvenanceGapReason.not_reported_by_primary,
    note="no uncertainty is released, so no standard error can be derived",
)
PHENOTYPE_GAPS: tuple[ProvenanceGap, ...] = (
    _GAP_N_SAMPLES,
    _GAP_SAMPLE_UNIT,
    _GAP_UNCERTAINTY,
    _GAP_SE,
)


def phenotype(value: float, spec: ConditionSpec) -> EnvironmentResponsePhenotype:
    """One released fitness-score of one strain in one condition."""
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.z_score,
        assay_type=AssayType.colony_size_array,
        environment_response=value,
        screen_id=spec.label,
        units=UNITS,
        provenance_gaps=list(PHENOTYPE_GAPS),
    )


def reference_phenotype(spec: ConditionSpec) -> EnvironmentResponsePhenotype:
    """The unaffected strain of the same condition: a fitness-score of 0."""
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.z_score,
        assay_type=AssayType.colony_size_array,
        environment_response=0.0,
        screen_id=spec.label,
        units=UNITS_REFERENCE,
    )


# --------------------------------------------------------------------------- #
# Genotype
# --------------------------------------------------------------------------- #
BW25113_NAMESPACE = STRAIN_GENE_NAMESPACES["BW25113"]


def genotype(source_label: str, locus_tag: str) -> Genotype:
    """One precise gene deletion of the arrayed library, written against BW25113."""
    return Genotype(
        perturbations=[
            BacterialDeletionPerturbation(
                systematic_gene_name=locus_tag,
                perturbed_gene_name=source_label,
                gene_namespace=BW25113_NAMESPACE,
                identifier_mapping=DerivedIdentifierMapping(
                    source_identifier=source_label, route="gene_symbol"
                ),
                collection=KEIO_COLLECTION,
            )
        ]
    )


# --------------------------------------------------------------------------- #
# Reading S1 Dataset
# --------------------------------------------------------------------------- #
class FitnessMatrix(BaseModel):
    """S1 Dataset as read: the gene labels, and this study's condition rows.

    The release's own shape is :data:`RELEASED_COLUMNS` by
    :data:`RELEASED_CONDITION_ROWS`; those are pinned by the data test rather than by
    the reader, because the sha256 pin ``download`` verifies already fixes the bytes and
    a synthetic fixture is narrower on purpose.
    """

    model_config = ConfigDict(extra="forbid")

    gene_labels: tuple[str, ...]
    #: Released condition label -> its row of raw cell strings, one per gene label.
    rows: dict[str, tuple[str, ...]]
    n_condition_rows: int


def read_fitness_matrix(path: str | Path) -> FitnessMatrix:
    """Read S1 Dataset and keep the condition rows of this study's batches.

    Every data row carries exactly as many fields as the header; a row that does not is
    refused rather than padded. The 57 labels of :data:`CONDITIONS` must all be present
    and nothing else of batch 1 or 4 may be, so a release whose labels moved stops the
    build instead of silently dropping a condition.
    """
    wanted = set(CONDITION_LABELS)
    rows: dict[str, tuple[str, ...]] = {}
    with open(path, newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        if header[0] != CONDITION_COLUMN:
            raise ValueError(
                f"{path}: expected the first column to be {CONDITION_COLUMN!r}, "
                f"got {header[0]!r}"
            )
        gene_labels = tuple(header[1:])
        n_rows = 0
        for row in reader:
            n_rows += 1
            if len(row) != len(header):
                raise ValueError(
                    f"{path}: condition row {row[0]!r} has {len(row)} fields, "
                    f"expected {len(header)}"
                )
            label = row[0]
            batch = _batch_of(label)
            if batch not in STUDY_BATCHES:
                continue
            if label not in wanted:
                raise ValueError(
                    f"{path}: condition {label!r} is in batch {batch} but is not one of "
                    "the 57 this loader types"
                )
            if label in rows:
                raise ValueError(f"{path}: condition {label!r} appears twice")
            rows[label] = tuple(row[1:])
    missing = sorted(wanted - set(rows))
    if missing:
        raise ValueError(f"{path}: typed conditions absent from the release: {missing}")
    return FitnessMatrix(gene_labels=gene_labels, rows=rows, n_condition_rows=n_rows)


_BATCH_RE = re.compile(r"\{(\d+)\}$")


def _batch_of(label: str) -> int:
    """The batch a released condition label ends with."""
    match = _BATCH_RE.search(label)
    if match is None:
        raise ValueError(f"condition {label!r} names no batch in curly brackets")
    return int(match.group(1))


# --------------------------------------------------------------------------- #
# Strain resolution and retention
# --------------------------------------------------------------------------- #
#: Release suffixes that mark a column as an allele other than a precise gene deletion:
#: an SPA affinity tag, a DAS degron, a kanamycin insertion, a point/indel mutant in
#: curly brackets, or a starred allele. The S1 Dataset legend is what makes these the
#: exceptions: "Unless otherwise specified, the mutations are precise gene deletions."
NON_DELETION_ALLELE = re.compile(r"(-SPA|-DAS(\+4)?|-kan|\{.*\}|\*)$", re.IGNORECASE)

DROP_NOT_A_DELETION = "label_is_not_a_gene_deletion"
DROP_NOT_IN_ANNOTATION = "label_is_not_in_the_bw25113_annotation"
DROP_MERGED_LOCUS = "label_is_a_fragment_of_a_merged_bw25113_locus"
DROP_AMBIGUOUS = "label_is_ambiguous_in_bw25113"
DROP_DUPLICATE_COLUMN = "label_has_two_columns_in_the_release"
DROP_BLANK_CELL = "cell_is_blank"

#: Each column rule, in the order a column is tested against them, with the description
#: the drop log carries.
COLUMN_RULES: tuple[tuple[str, str], ...] = (
    (
        DROP_NOT_A_DELETION,
        "the label names an allele other than a precise gene deletion (an -SPA affinity "
        "tag, a -DAS/-DAS+4 degron, a -kan insertion, a {...} point or indel mutant, or "
        "a starred allele) from the essential-gene arrays the screened library carries "
        "beside the KEIO deletions. BacterialDeletionPerturbation would assert an "
        "absence that did not happen, and the schema has no bacterial tagged-allele or "
        "degron leaf",
    ),
    (
        DROP_NOT_IN_ANNOTATION,
        "the label is no locus tag, symbol or synonym of GCA_000750555.1: a multi-gene "
        "deletion, an old-annotation ORF fragment, a retired symbol or small RNA, an "
        "ECK accession, or a -A/-B/-C suffixed allele",
    ),
    (
        DROP_MERGED_LOCUS,
        "two or more labels resolve to ONE current BW25113 locus, because the "
        "collection deleted fragments of a pseudogene the current annotation merges; no "
        "member IS the locus tag, so the whole group is dropped rather than merged into "
        "one strain",
    ),
    (DROP_AMBIGUOUS, "the label matches more than one BW25113 locus"),
    (
        DROP_DUPLICATE_COLUMN,
        "the label heads two columns of the release, whose values disagree wherever "
        "both are present, so they are two independent measurements of two strains the "
        "release gives nothing to tell apart; keeping both would put two records on one "
        "(strain, condition) key and keeping one would be an arbitrary choice",
    ),
)


class StrainColumn(BaseModel):
    """One kept gene column: its index, its released label and its BW25113 locus tag."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    index: int
    source_label: str
    locus_tag: str


class ColumnResolution(BaseModel):
    """What the identifier step did to the release's 3,975 gene columns."""

    model_config = ConfigDict(extra="forbid")

    kept: list[StrainColumn]
    #: Drop rule -> the released labels it removed, sorted.
    dropped_labels: dict[str, list[str]]
    #: Drop rule -> the number of columns it removed.
    dropped_columns: dict[str, int]
    reconciliation: LocusTagReconciliation


def resolve_columns(
    gene_labels: tuple[str, ...], genome: EcoliK12BW25113Genome, *, label: str
) -> ColumnResolution:
    """Map each gene column to a BW25113 locus tag, applying the five column rules.

    The retain-all reconciliation decides which labels resolve; this function decides
    which of the rest can be WRITTEN, because ``BacterialDeletionPerturbation`` stores a
    locus tag of the declared namespace and nothing else. Every dropped column is
    attributed to exactly one rule, in :data:`COLUMN_RULES` order.
    """
    stored, reconciliation = reconcile_locus_tags(
        genome, pd.Series(gene_labels), label=label
    )
    reconciliation.require_resolved(MIN_RESOLVED_FRACTION)
    retired = set(reconciliation.retired_kept)
    by_rule: dict[str, set[str]] = {
        DROP_NOT_A_DELETION: {
            name for name in retired if NON_DELETION_ALLELE.search(name)
        },
        DROP_NOT_IN_ANNOTATION: {
            name for name in retired if not NON_DELETION_ALLELE.search(name)
        },
        DROP_MERGED_LOCUS: set(reconciliation.kept_on_collision),
        DROP_AMBIGUOUS: set(reconciliation.ambiguous_kept),
    }
    counts = Counter(gene_labels)
    claimed: set[str] = set().union(*by_rule.values())
    by_rule[DROP_DUPLICATE_COLUMN] = {
        name for name, count in counts.items() if count > 1 and name not in claimed
    }
    dropped = claimed | by_rule[DROP_DUPLICATE_COLUMN]
    kept = [
        StrainColumn(index=index, source_label=name, locus_tag=str(stored.iloc[index]))
        for index, name in enumerate(gene_labels)
        if name not in dropped
    ]
    tags = Counter(column.locus_tag for column in kept)
    collisions = sorted(tag for tag, count in tags.items() if count > 1)
    if collisions:
        raise RuntimeError(
            f"{label}: {len(collisions)} locus tags are claimed by more than one kept "
            f"column after the drop rules: {collisions[:10]}"
        )
    outside = sorted(
        {
            column.locus_tag
            for column in kept
            if genome.resolve_gene_name(column.locus_tag).status
            not in (GeneNameStatus.CURRENT, GeneNameStatus.NON_GENE_FEATURE)
        }
    )
    if outside:
        raise RuntimeError(
            f"{label}: {len(outside)} kept tags are not loci of the pinned assembly: "
            f"{outside[:10]}"
        )
    return ColumnResolution(
        kept=kept,
        dropped_labels={rule: sorted(by_rule[rule]) for rule, _ in COLUMN_RULES},
        dropped_columns={
            rule: sum(counts[name] for name in by_rule[rule])
            for rule, _ in COLUMN_RULES
        },
        reconciliation=reconciliation,
    )


# --------------------------------------------------------------------------- #
# Drop log
# --------------------------------------------------------------------------- #
class DropRule(BaseModel):
    """One retention rule, the records it removed, and the items it removed them for."""

    rule: str
    scope: Literal["column", "cell"]
    description: str
    n_columns: int
    n_records: int
    items: list[str] = []


class DropLog(BaseModel):
    """Every retention rule applied to a build, in the order they were applied."""

    dataset: str
    source_records: int
    kept_records: int
    dropped_records: int
    rules: list[DropRule]


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
@register_dataset
class EnvChemgenShiver2016Dataset(ExperimentDataset):
    """Shiver 2016 KEIO deletion fitness-scores across 57 chemical-genomic conditions."""

    #: The strain whose genome the build entry points inject as ``ecoli_genome``: the
    #: KEIO deletion library's background, and the assembly every record pins.
    REFERENCE_STRAIN: ClassVar[EcoliK12StrainName] = "BW25113"

    def __init__(
        self,
        root: str = "data/torchcell/ecoli_env_chemgen_shiver2016",
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
        return BacterialEnvironmentResponseExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return BacterialEnvironmentResponseExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The mirrored S1 Dataset."""
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
            "Shiver 2016 S1 Dataset linked into %s (sha256 verified)", self.raw_dir
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
        """Parse S1 Dataset into per-(strain, condition) records; write the LMDB."""
        verify_raw_files(self.raw_dir, {DATA_FILENAME: DATA_SHA256})
        matrix = read_fitness_matrix(osp.join(self.raw_dir, DATA_FILENAME))
        genome = self._genome()
        columns = resolve_columns(matrix.gene_labels, genome, label=self.name)
        genome_reference = assembly_reference(self.REFERENCE_STRAIN)
        environments = {spec.label: environment(spec) for spec in CONDITIONS}
        references = {
            spec.label: BacterialEnvironmentResponseExperimentReference(
                dataset_name=self.name,
                genome_reference=genome_reference,
                environment_reference=environments[spec.label],
                phenotype_reference=reference_phenotype(spec),
            )
            for spec in CONDITIONS
        }
        publication = Publication(doi=DOI, doi_url=f"https://doi.org/{DOI}")

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        idx = 0
        n_blank_cells = 0
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for spec in tqdm(CONDITIONS, desc="shiver2016"):
                row = matrix.rows[spec.label]
                for column in columns.kept:
                    cell = row[column.index]
                    if cell == "":
                        n_blank_cells += 1
                        continue
                    experiment = BacterialEnvironmentResponseExperiment(
                        dataset_name=self.name,
                        genotype=genotype(column.source_label, column.locus_tag),
                        environment=environments[spec.label],
                        phenotype=phenotype(float(cell), spec),
                    )
                    txn.put(
                        f"{idx}".encode(),
                        self._intern_record(
                            experiment, references[spec.label], publication, itxn
                        ),
                    )
                    idx += 1
        env.close()
        interned_env.close()

        self._write_reports(matrix, columns, kept_records=idx, blanks=n_blank_cells)
        log.info(
            "Shiver2016: wrote %d records from %d kept columns x %d conditions; "
            "dropped columns %s, blank cells %d",
            idx,
            len(columns.kept),
            len(CONDITIONS),
            columns.dropped_columns,
            n_blank_cells,
        )

    def _write_reports(
        self,
        matrix: FitnessMatrix,
        columns: ColumnResolution,
        *,
        kept_records: int,
        blanks: int,
    ) -> None:
        """Write the drop log and the identifier report, and check the arithmetic."""
        n_conditions = len(CONDITIONS)
        source_records = len(matrix.gene_labels) * n_conditions
        rules = [
            DropRule(
                rule=rule,
                scope="column",
                description=description,
                n_columns=columns.dropped_columns[rule],
                n_records=columns.dropped_columns[rule] * n_conditions,
                items=columns.dropped_labels[rule],
            )
            for rule, description in COLUMN_RULES
        ]
        rules.append(
            DropRule(
                rule=DROP_BLANK_CELL,
                scope="cell",
                description="the strain has no fitness-score in this condition; the "
                "Methods state the filtering ('Unreliable measurements were removed "
                "from the dataset at multiple points in the analysis and each condition "
                "had a different number of measurements (fitness-scores) that passed "
                "analysis')",
                n_columns=0,
                n_records=blanks,
                items=[],
            )
        )
        drop_log = DropLog(
            dataset=self.name,
            source_records=source_records,
            kept_records=kept_records,
            dropped_records=source_records - kept_records,
            rules=rules,
        )
        accounted = sum(rule.n_records for rule in rules)
        if accounted != drop_log.dropped_records:
            raise RuntimeError(
                f"drop accounting mismatch: rules total {accounted}, "
                f"{drop_log.dropped_records} records missing from the build"
            )
        with open(osp.join(self.preprocess_dir, "dropped_records.json"), "w") as handle:
            handle.write(drop_log.model_dump_json(indent=2))
        identifiers = {
            "dataset": self.name,
            "released_gene_columns": len(matrix.gene_labels),
            "released_condition_rows": matrix.n_condition_rows,
            "study_conditions": n_conditions,
            "kept_columns": len(columns.kept),
            "distinct_locus_tags": len({c.locus_tag for c in columns.kept}),
            "min_resolved_fraction": MIN_RESOLVED_FRACTION,
            "reconciliation": columns.reconciliation.model_dump(mode="json"),
            "dropped_columns": columns.dropped_columns,
            "dropped_labels": columns.dropped_labels,
            "identifier_route": "gene_symbol",
        }
        with open(
            osp.join(self.preprocess_dir, "identifier_reconciliation.json"), "w"
        ) as handle:
            json.dump(identifiers, handle, indent=2)

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError(
            "EnvChemgenShiver2016Dataset builds records in process()"
        )


# --------------------------------------------------------------------------- #
# Verification (L0-L4) of a built tree
# --------------------------------------------------------------------------- #
#: Records of the full build: 3,720 kept columns x 57 conditions, minus 8,007 blanks.
EXPECTED_RECORDS = 204033


def verify_build(
    dataset_root: str,
    *,
    genome: EcoliK12BW25113Genome | None = None,
    data_root: str | None = None,
    expected_count: int = EXPECTED_RECORDS,
) -> VerificationReport:
    """Run the environment-response L0-L4 verifier on a built tree and write its report.

    The LMDB is streamed once. Every record is checked against the BW25113 genome its
    references pin: the resolver of the canonical-name rule, and as the L4 universe every
    GenBank locus of the assembly (pseudogenes included, the same set as the verification
    runners' ``_ecoli_k12_gene_set``). One resolver cannot serve two strains, which is
    why the host-aware gene set is read from the record's own pinned assembly rather than
    from the yeast runner's S288C default. The report is written to
    ``preprocess/verification_report.json``.
    """
    from torchcell.verification.environment_response import (
        verify_environment_response_dataset_streaming,
    )
    from torchcell.verification.runners import stream_records

    if genome is None:
        genome = _bw25113(data_root)
    report = verify_environment_response_dataset_streaming(
        stream_records(dataset_root),
        dataset_name=osp.basename(osp.normpath(dataset_root)),
        provenance=Provenance(
            source_uri=f"$DATA_ROOT/{RAW_DIR_REL}/{DATA_REL}",
            citation_key=CITATION_KEY,
            sha256=DATA_SHA256,
            method="S1 Dataset integrated fitness-score matrix, the 57 condition rows "
            "of batches 1 and 4; one BacterialEnvironmentResponseExperiment per "
            "(deletion strain, condition), z_score fitness-score",
            page="S1 Dataset (pgen.1006124.s005.txt), rows '... {1}' and '... {4}'",
            retrieved=DATA_RETRIEVED_AT,
        ),
        expected_count=expected_count,
        sgd_genes=set(genome.genbank.loci),
        resolve_gene_name=genome.resolve_gene_name,
    )
    preprocess = osp.join(dataset_root, "preprocess")
    os.makedirs(preprocess, exist_ok=True)
    with open(osp.join(preprocess, "verification_report.json"), "w") as handle:
        handle.write(report.model_dump_json(indent=2))
    return report


def _bw25113(data_root: str | None = None) -> EcoliK12BW25113Genome:
    """The BW25113 genome from its default cache root."""
    genome = bacterial_genome("ecoli", "BW25113", data_root)
    if not isinstance(genome, EcoliK12BW25113Genome):
        raise TypeError(f"expected the BW25113 genome, got {type(genome).__name__}")
    return genome


def main() -> None:
    """Build the dataset under ``DATA_ROOT`` and verify it, for interactive debugging."""
    from dotenv import load_dotenv

    load_dotenv()
    data_root = _data_root()
    root = osp.join(data_root, "data/torchcell/ecoli_env_chemgen_shiver2016")
    dataset = EnvChemgenShiver2016Dataset(root=root)
    print(f"len = {len(dataset)}")
    print(dataset[0])
    print(
        json.dumps(
            json.loads(Path(root, "preprocess", "dropped_records.json").read_text())[
                "rules"
            ],
            indent=2,
        )[:2000]
    )
    print(verify_build(root, data_root=data_root).summary())


if __name__ == "__main__":
    main()
