# torchcell/datasets/ecoli/nichols2011
# [[torchcell.datasets.ecoli.nichols2011]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/nichols2011
# Test file: tests/torchcell/datasets/ecoli/test_nichols2011.py
"""Nichols 2011: the phenotypic landscape of E. coli K-12 in 324 screens.

Nichols et al. 2011 (Cell 144:143-156, doi:10.1016/j.cell.2010.11.052, citation key
``nicholsPhenotypicLandscapeBacterial2011``, issue #691) pinned the Keio single-gene
deletion collection, SPA- and DAS-tagged essential genes, essential-gene alleles with
their linked controls, and an sRNA/small protein deletion library at 1,536 colonies per
plate onto agar carrying one stress each, imaged the plates, and scored every
(strain, screen) colony size with the Collins 2006 S-score procedure. Table S2 ("Complete
Data Set", ``si/mmc2.xls``, an OOXML workbook under an ``.xls`` name) releases the score
matrix: 3,979 strain rows x 324 screen columns, 1,286,962 scores and 2,234 empty cells.

THE DATA IS THE DATA (owner, 2026-10-12). Every released score is a record and nothing
is looked up: 1,286,962 records, one per non-empty Table S2 cell. The 2,234 empty cells
are absent, as the release has them (:data:`EMPTY_CELL_RULE`, ``cell_accounting.json``).

STRAINS, AS RELEASED. Table S2 labels rows by ECK accession and upper-case symbol
(``ECK1963-HCHA``), with a suffix for the essential-gene arrays (``- SPA``, ``- DAS``,
``- Linked``, ``- Allele``, ``- Truncation``). Each row's genotype is one
``ReleasedIdentifierPerturbation`` per ECK accession of its label (the leaf this
dataset added): ``systematic_gene_name`` is the accession verbatim, ``perturbed_gene_name``
the released symbol, ``strain_label`` the whole row label, ``released_kind`` the class
its suffix states (:data:`SUFFIX_KINDS`), ``cassette="kanamycin"`` ("All mutant strains
are marked with or linked to kanamycin"), and ``collection`` the SPA or DAS collection
name where the suffix states one. No genome, synonym table or crosswalk is read: an
accession no annotation carries (the 41 ECK-only labels, the ECK5000-series sRNAs), a
designed allele released as a protein change only (``ECK0097-B-LPXC(G210S)-KAN``) and a
multi-gene deletion (``ECK1205/1207/1209-RDLABC``, one perturbation per accession) are
stored exactly as released. A multi-gene label names only the set, so each of its genes
carries the name the release gives that accession on its own row (``ECK4143-ECNB``), or
the accession where it has none (:func:`released_gene_names`). Twelve labels head two
rows each (the two truncations of yfiO share one label); both rows are stored, told
apart by ``replicate_id`` = ``Table S2 row <n>`` (24 rows).

Every record pins BW25113 ("During the entire study, BW25113 is referred to as
wild-type"). OPEN: "sRNA and small protein deletions were constructed in MG1655 (Hobbs
et al., 2010) and used in that background", and Table S2 does not say which 100 rows
those are, so their records pin BW25113 like every other row; recorded in the dataset
note and the PR as an owner question.

PHENOTYPE. ``EnvironmentResponsePhenotype``, ``measurement_type=z_score``,
``assay_type=colony_size_array``, ``screen_id`` = the verbatim Table S2 column label,
reference 0.0. The number is the S-score procedure's colony-size score, "analogous to
''unaveraged'' S-scores (Collins et al., 2006)" because a condition-gene interaction is
built one way only, and the Phenotype Analysis states the per-screen step the release
carries: "re-scaled the data to a standard normal distribution (IQR=1.35)". Measured on
Table S2 (``experiments/database/scripts/nichols2011_release_checks.py``): all 324
columns have a median within 0.022 of 0 and an IQR of 1.309 to 1.379 (median 1.348). A
per-screen standard-normal rescaling IS what ``MeasurementType.z_score`` ("standardized
fitness/growth deviation") names, and that member's docstring already names Nichols
2011, so no phenotype schema change is made. Sign: "80% of the phenotypes were negative
(gene deletion more sensitive) and 20% positive (gene deletion more resistant)".

REPLICATES. "Two independently derived clones of each mutant strain were analyzed (for
sRNA mutants, a single isolate was arrayed twice), and screens were performed at least
twice, enabling scores to be based on four to six independent measurements." No
per-record count is released and no companion statistic exists to back-solve one, so
``n_samples=4`` (the conservative lower end of the range) with ``sample_unit=colony``.
No dispersion is released: ``environment_response_uncertainty`` and
``environment_response_se`` are ``not_reported_by_primary`` gaps.

CONDITIONS, AS TABLE S1 NAMES THEM. Table S1 (``si/mmc1.xls``) lists 114 screens (the
paper's "114 unique stresses") whose ``Screens`` cells sum to its ``Total Screens``
cell, 324. Each row is transcribed verbatim in :data:`SCREENS` and
:func:`check_screen_list` refuses a build whose bytes read otherwise; every number in a
row's ``Concentrations Screened`` cell is one of its transcribed column doses or an
annotation (the 100 mM buffer, the 6 of "G6P"). Each Table S2 label
(``<condition>-<dose> - <suffix>``) matches exactly one Table S1 column
(:func:`match_conditions`, a bijection) and the record stores Table S1's dose. Each
chemical is a ``Compound`` under the name Table S1 writes (``Ampicillin``,
``Clarythromycin``, ``CoCl2``, ``Calcofluor (F3543 Fluorescent Brightener 28)``); the
curated compound table only attaches an InChIKey where it already holds that name
(:func:`released_compound`), and a name it lacks keeps a typed ``inchikey`` gap.

- 296 small-molecule columns (two of them the MOPS iron plates): one
  ``SmallMoleculePerturbation`` per compound (two for the cefsulodin + mecillinam,
  trimethoprim + sulfamethizole, SDS + EDTA and fosfomycin + glucose 6P combinations).
  ``ng/ml`` and ``mg/ml`` are stored in ``ug/mL`` by exact decimal conversion; a bare
  ``%`` is ``ConcentrationUnit.percent`` (no basis is stated in the release).
- 7 carbon-source columns (``factor=carbon_source``): Table S1 prints a bare fraction
  (``0.006``), read as a percent because the release's own MMS label (``MMS-0.05%`` for
  Table S1's ``0.0005``) does so. Six rows say "(M9)" and sit on M9 agar;
  ``N-acetyl Glucosamine`` does not, so it sits on the default LB agar ("mixed into LB
  agar (except where indicated)") with its Target cell's "alternative carbon source".
- 3 MOPS columns: high and low iron (``FeSO4`` at 1 mM and 2 uM, "normal 100uM") and
  ``NH4Cl`` as the nitrogen source at 9.5 mM (Table S1 writes "9.5mm").
- 8 buffered-pH columns (``factor=ph``): the basic series names TAPS at 100 mM, stored
  as a second perturbation; the acidic series names "MES-HOMOPIPES" and not which buffer
  set which pH, so its agent is a typed gap.
- 7 temperature columns (16, 18, 20, 40, 42, 43.5, 45 C) on ``Environment.temperature``;
  every other plate is 37 C. 4 UV columns as ``PhysicalExposurePerturbation`` dosed by
  Table S1's exposure time (irradiance and fluence typed gaps). 1 anaerobic column.
- ``duration_hours`` is a typed gap on every environment: 14 to 16 h for "the majority
  of conditions", and "Several stresses, e.g. low temperature, required longer
  incubation times" without naming them.

MEDIA. The paper names no LB formulation and no amounts, so :data:`NICHOLS2011_LB_AGAR`
derives from the ``LB`` library key with every component's concentration ``None``; the
M9 and MOPS plates derive from ``M9`` and ``MOPS_MINIMAL`` the same way.

SHIVER 2016 (G4). ``shiver2016.py`` serves only Shiver's own 57 conditions; its S1
Dataset re-analyzes 235 of these screens as batch 0, which it does not load. Measured
(``nichols2011_release_checks.py``): the 235 batch-0 conditions match 235 distinct Table
S2 columns by profile correlation, and 831,939 of their 888,431 cells land on a
(strain, column) stored here, with 0 identical values, 10,052 within 0.01, a pooled
Pearson r of 0.800 and a median absolute difference of 0.458: a re-scoring, not a copy,
and none of it served twice.

TABLES S3 TO S7 (not read; :data:`DERIVED_SI_TABLES`). Each is an analysis of Table S2:
multi-stress responsive strains (94 labels), conditionally essential genes (116 rich
media, 102 auxotrophs), rich-media lethal events (831 strain-condition pairs), orphan to
annotated gene correlations (832 pairs) and drug-GO scores (719 rows).

SOURCED VALUES are quoted from the MinerU OCR of ``si/mmc8.pdf`` (the article and its
Supplemental Information, byte-identical to Elsevier's CDN object), written beside the
PDF in the raw mirror (:func:`record_article_ocr` pins it), because the literature
mirror has no key for this paper: the Zotero item ``REEL3RAD`` carries no PDF (#691).

DATA SOURCE: the owner's browser download of the Cell supplemental information,
``$DATA_ROOT/torchcell-raw/nicholsPhenotypicLandscapeBacterial2011/si/mmc1.xls`` to
``mmc8.pdf`` (``manual_browser``, PR #1046), every file byte-identical to
``ars.els-cdn.com/content/image/1-s2.0-S0092867410013747-mmc<N>``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import os.path as osp
import re
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any, ClassVar, Final, Literal

import openpyxl
import xlrd
from pydantic import BaseModel, ConfigDict, model_validator
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
from torchcell.datamodels.media import LB, M9
from torchcell.datamodels.schema import (
    AssayType,
    BacterialEnvironmentResponseExperiment,
    BacterialEnvironmentResponseExperimentReference,
    Compound,
    Concentration,
    ConcentrationUnit,
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
    PhysicalExposurePerturbation,
    PhysicalFactor,
    Publication,
    ReleasedIdentifierPerturbation,
    ReleasedKind,
    SampleUnit,
    SmallMoleculePerturbation,
    Temperature,
)
from torchcell.datasets.bacteria_common import (
    STRAIN_GENE_NAMESPACES,
    assembly_reference,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.literature.manifest import (
    ROLE_OCR_IMAGE,
    ROLE_OCR_LAYOUT,
    ROLE_OCR_PROVENANCE,
    ROLE_SI_OCR,
    ArtifactRecord,
    Manifest,
    ProcessingRecord,
)
from torchcell.literature.manual_deposit import extend_manifest
from torchcell.sequence.genome.ecoli.k12 import EcoliK12StrainName
from torchcell.verification.report import Provenance, VerificationReport
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
)

log = logging.getLogger(__name__)

DOI: Final = "10.1016/j.cell.2010.11.052"
#: Elsevier's publication item identifier; the CDN names every supplementary object by it.
PII: Final = "S0092867410013747"
TITLE: Final = "Phenotypic Landscape of a Bacterial Cell"
CITATION_KEY: Final = "nicholsPhenotypicLandscapeBacterial2011"
RAW_DIR_REL: Final = f"torchcell-raw/{CITATION_KEY}"

#: Table S1, the screen list (BIFF).
SCREEN_LIST_REL: Final = "si/mmc1.xls"
SCREEN_LIST_SHA256: Final = (
    "349ae759627153dbb6921aee6cc6fc4ef2d6a3e3304da5c68afeb16f5eb02e30"
)
#: Table S2, the complete data set: an OOXML workbook released under an ``.xls`` name,
#: so it is opened from a binary handle (openpyxl refuses the ``.xls`` suffix by path).
MATRIX_REL: Final = "si/mmc2.xls"
MATRIX_SHA256: Final = (
    "54be9cbc384bdbc7b1aa5a1da872d0e10ca513526f1f446ed068d27675f10fa8"
)
#: The article (pages 1 to 14) followed by its Supplemental Information (pages 15 to 23).
ARTICLE_PDF_REL: Final = "si/mmc8.pdf"
ARTICLE_PDF_SHA256: Final = (
    "e0c4e40678d83bc47252d8e47cad64bdd8178ca4713cad2496e0a72b77909e2c"
)
#: Its MinerU OCR, written beside it by ``torchcell.literature.ocr.ocr_pdf`` (the
#: function ``lit_capture_si --ocr-recorded`` and ``reocr_si`` call), CPU mode, under
#: slurm (``experiments/036-dataset-fixes-before-kg-build/scripts/
#: gh_ocr_nichols2011_mmc8.slurm``). Every article quote is a substring of this file.
ARTICLE_MD_REL: Final = "si/mmc8.md"
ARTICLE_MD_SHA256: Final = (
    "0fbf32478f0491b25b27a031b47f639e1d67cbba4b5311f42940673e5ab33e92"
)
ARTICLE_OCR_PROVENANCE: Final = "si/mmc8_ocr_provenance.json"
ARTICLE_OCR_LAYOUT: Final = ("si/mmc8_content_list.json", "si/mmc8_middle.json")
ARTICLE_OCR_IMAGES: Final = "si/images/mmc8"

#: The four released tables the loader does NOT read, measured on the deposit (the
#: G3 inventory, ``experiments/database/scripts/nichols2011_release_checks.py``). Each
#: is an analysis derived from Table S2, not a measurement of its own.
DERIVED_SI_TABLES: Final[dict[str, str]] = {
    "si/mmc3.xls": "Table S3 - Multi-Stress Responsive Genes: 94 strain labels, one "
    "column, no values (strains with phenotypes in many stresses)",
    "si/mmc4.xls": "Table S4 - Conditionally-Essential Genes: two label lists, 116 "
    "'Rich-Media CE' and 102 'Auxotrophs', no values",
    "si/mmc5.xls": "Table S5. Rich-media CE Lethal Events: 831 (Strain, Lethal "
    "Condition) pairs, the conditionally-essential calls (probability score 1 and raw "
    "colony size below 60 pixels), no values",
    "si/mmc6.xls": "Table S6 - Highly Correlated Orphan-Annotated Gene Pairs (r >= "
    "0.5): 832 (orphan gene, annotated gene, correlation coefficient, function) rows, "
    "correlations of Table S2 profiles",
    "si/mmc7.xls": "Table S7 - Drug-GO Interactions at p<= 10-3: 719 (GO biological "
    "process, drug, drug-GO score) rows on a sheet named 'TableS5-Drug-GO "
    "interactions.tx', a gene-set statistic over Table S2",
}

#: The Table S2 title cell, header cell and shape, measured on the pinned bytes.
MATRIX_SHEET: Final = "TableS2-FinalData"
MATRIX_TITLE: Final = "Table S2 - Complete Data Set"
MATRIX_HEADER: Final = "Gene"
RELEASED_STRAIN_ROWS: Final = 3979
RELEASED_CONDITIONS: Final = 324
SCREEN_LIST_TITLE: Final = "Table S1 - Screen List"
SCREEN_LIST_HEADER: Final = (
    "Antibiotic/Condition",
    "Supplier",
    "Item No",
    "Target",
    "Concentrations Screened",
    "Screens",
)

KEIO_COLLECTION: Final = "Keio Collection"
SPA_COLLECTION: Final = "SPA-tagged derivatives of essential genes"
DAS_COLLECTION: Final = "DAS-tagged essential genes"
#: "All mutant strains are marked with or linked to kanamycin."
KANAMYCIN: Final = "kanamycin"

UNITS: Final = (
    "Table S2 fitness score: the colony-size score of the Collins 2006 S-score "
    "procedure, 'analogous to unaveraged S-scores', re-scaled per screen to a standard "
    "normal distribution (IQR 1.35); negative = the deletion is more sensitive, "
    "positive = more resistant, 0 = the screen's typical strain"
)
UNITS_REFERENCE: Final = "the screen's typical strain (re-scaled score 0)"


# --------------------------------------------------------------------------- #
# Sourced values (verbatim quotes, pinned sha256)
# --------------------------------------------------------------------------- #
def _article(
    value: Any, quote: str, *, page: str, note: str | None = None
) -> SourcedValue:
    """A value bound to the MinerU OCR of the deposited article PDF (``si/mmc8.md``).

    ``source_uri`` is relative to ``$DATA_ROOT/torchcell-raw/<citation_key>/``;
    :func:`sourced_value_root` resolves it there. The literature mirror has no key for
    this paper (the Zotero item ``REEL3RAD`` carries no PDF, #691), so the article is
    quoted from the owner's deposit of Elsevier's ``mmc8.pdf``, which holds it whole.
    """
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=ARTICLE_MD_REL,
            citation_key=CITATION_KEY,
            sha256=ARTICLE_MD_SHA256,
            method=f"MinerU OCR (CPU) of {ARTICLE_PDF_REL} (sha256 "
            f"{ARTICLE_PDF_SHA256}), the owner's manual deposit (torchcell-raw mirror)",
            page=page,
        ),
    )


_STRAINS = "Extended Experimental Procedures, 'Bacterial Strains'"
_SCREEN = "Extended Experimental Procedures, 'Screen Conditions and Data Processing'"
_PHENOTYPE = "Extended Experimental Procedures, 'Phenotype Analysis'"
_RESULTS = "Results, 'Phenomic Profiling of E. coli K12 Yields a Robust, High-Quality Data Set'"
_RESPONSIVE = (
    "Results, 'Phenomic Profiling Defines Responsive and Conditionally Essential Genes'"
)

COLLECTION = _article(
    (4334, 148, 9, 5, 2, 100),
    "The screened collection includes 4334 deletions of non-essential genes (Keio "
    "Collection; Baba et al., 2006), 148 SPA-tagged derivatives of essential genes "
    "(Butland et al., 2005), 9 point-mutant alleles of essential genes",
    page=_STRAINS,
    note="the collections the array was built from; the SPA and DAS rows carry the "
    "collection names this paragraph gives",
)
FINAL_STRAINS = _article(
    {
        "Keio": 3737,
        "SPA": 117,
        "DAS": 5,
        "alleles": 9,
        "linked controls": 9,
        "truncations": 2,
        "sRNAs": 100,
    },
    "leaving a total of 3979 strains in the final dataset (3737 Keio, 117 SPA, 5 DAS, 9 "
    "alleles, 9 linked controls, 2 truncations, and 100 sRNAs).",
    page=_STRAINS,
    note="3,979 is the Table S2 row count; the release marks the SPA, DAS, allele, "
    "linked and truncation rows by a label suffix and does NOT mark which deletion row "
    "is a Keio strain and which one of the 100 sRNA/small protein strains",
)
KANAMYCIN_MARKED = _article(
    KANAMYCIN,
    "All mutant strains are marked with or linked to kanamycin.",
    page=_STRAINS,
    note="the cassette every stored perturbation names",
)
BUILT_IN_BW25113 = _article(
    "BW25113",
    "All linked alleles and controls, DAS-tags, and truncations were constructed in "
    "BW25113 (Keio Collection strain background) using standard recombineering "
    "methods.",
    page=_STRAINS,
)
SPA_TRANSDUCED = _article(
    "BW25113",
    "SPA-tagged essential genes, originally constructed in W3110 (Butland et al., "
    "2005), were transduced into BW25113 using P1 phage.",
    page=_STRAINS,
)
SRNA_BACKGROUND = _article(
    "MG1655",
    "sRNA and small protein deletions were constructed in MG1655 (Hobbs et al., 2010) "
    "and used in that background.",
    page=_STRAINS,
    note="the release does not say which 100 deletion rows these are, so every record "
    "pins BW25113 (WILD_TYPE) and the 100 are an open owner question in the dataset "
    "note",
)
WILD_TYPE = _article(
    "BW25113",
    "During the entire study, BW25113 is referred to as wild-type.",
    page=_STRAINS,
    note="the background every record's AssemblyReferenceGenome pins and the "
    "gene_namespace every perturbation names",
)
HYPOMORPHS = _article(
    ("C", "hypomorphic"),
    "essential gene hypomorphs (C-terminally tandem-affinity tagged [Butland et al., "
    "2005] or specific alleles)",
    page=_RESULTS,
    note="what the '- SPA' and '- Allele' rows are; the released-strain leaf stores "
    "the label and its class, so the C terminus and the hypomorph call stay in this "
    "quote rather than on the record",
)
LINKED_CONTROLS = _article(
    "a kanamycin cassette linked to the wild-type allele",
    "in which the antibiotic-resistance cassette was linked to the wild-type allele as "
    "a control)",
    page=_STRAINS,
    note="what the nine '- Linked' rows are (ReleasedKind.linked_marker)",
)
REPLICATE_DESIGN = _article(
    (4, 6),
    "Two independently derived clones of each mutant strain were analyzed (for sRNA "
    "mutants, a single isolate was arrayed twice), and screens were performed at least "
    "twice, enabling scores to be based on four to six independent measurements.",
    page=_RESULTS,
    note="a range with no per-record count released and no companion statistic to "
    "back-solve it from, so n_samples takes the conservative lower end, 4 (CLAUDE.md, "
    "'Resolving a range with no per-record value'), and one sample is a colony",
)
SCREEN_MEDIUM = _article(
    "LB agar",
    "For each screen, the chemical/stress was mixed into LB agar (except where "
    "indicated) at the concentrations listed in Table S1.",
    page=_SCREEN,
    note="no LB formulation (Miller or Lennox) and no amounts are stated, so the LB "
    "components carry no concentration",
)
ARRAY_PLATES = _article(
    "1536-format agar plates",
    "The collection was arrayed in 1536-format onto six LB-agar plates using a Singer "
    "Rotor robot.",
    page=_SCREEN,
)
INCUBATION = _article(
    37.0,
    "For the majority of conditions, plates were removed from the incubator following "
    "14-16 hrs growth at $3 7 ^ { \\circ } \\mathrm { C }$ .",
    page=_SCREEN,
    note="37 C is the temperature of every plate but the cold and heat shocks, whose "
    "Table S1 rows state their own",
)
LONGER_INCUBATION = _article(
    "per-condition incubation not released",
    "Several stresses, e.g. low temperature, required longer incubation times.",
    page=_SCREEN,
    note="why duration_hours is a typed gap: the majority ran 14 to 16 h, a range, "
    "and the conditions that ran longer are not listed",
)
SCORING = _article(
    "Collins 2006 S-score procedure",
    "All image processing, colony size measurements, and score generation was carried "
    "out as described previously for genetic interactions (Collins et al., 2006; Typas "
    "et al., 2008).",
    page=_SCREEN,
)
UNAVERAGED_S_SCORES = _article(
    "unaveraged S-score",
    "so the scores generated here are analogous to ‘‘unaveraged’’ "
    "S-scores (Collins et al., 2006).",
    page=_SCREEN,
    note="a condition-gene interaction is constructed one way only, so there is no "
    "reciprocal score to average",
)
RESCALED = _article(
    "standard normal per screen",
    "For each screen, we estimated the mean and IQR of Condition-Gene scores and "
    "re-scaled the data to a standard normal distribution $( \\mathsf { I Q R } = 1 . "
    "3 5 )$",
    page=_PHENOTYPE,
    note="measured on Table S2 (nichols2011_release_checks.py): every one of the 324 "
    "columns has a median within 0.022 of 0 and an IQR of 1.309 to 1.379 (median "
    "1.348), so the released numbers ARE the re-scaled scores, which is what "
    "measurement_type z_score names",
)
SIGN = _article(
    "negative = more sensitive, positive = more resistant",
    "of the phenotypes were negative (gene deletion more sensitive) and $20 \\%$ "
    "positive (gene deletion more resistant)",
    page=_RESPONSIVE,
)
QUALITY_CONTROL = _article(
    "screens with high IQR or poorly correlated replicates were removed",
    "Screens with high IQR, or with poor correlation within a series of technical "
    "replicates were eliminated from the dataset to minimize noise.",
    page=_SCREEN,
    note="the per-screen filter behind the released 324; Table S2's 2,234 empty cells "
    "are per-cell absences the paper does not explain, and are left absent",
)
SRNA_NORMALIZED_SEPARATELY = _article(
    "sRNA/small protein colonies normalized on their own",
    "raw colony sizes of all sRNA/small protein mutants (which were arrayed on the same "
    "plate of the collection) were normalized separately to eliminate the relative "
    "comparisons with the BW25113-based mutants arrayed on the plate.",
    page=_SCREEN,
)
TABLE_S2 = _article(
    (3979, 324),
    "The entire matrix (3979 mutants $\\times 3 2 4$ conditions)",
    page=_RESULTS,
    note="the Table S2 shape: 3,979 strain rows by 324 screen columns",
)

#: Every module-level article quote, for the mirror audit.
SOURCED_VALUES: tuple[SourcedValue, ...] = (
    COLLECTION,
    FINAL_STRAINS,
    KANAMYCIN_MARKED,
    BUILT_IN_BW25113,
    SPA_TRANSDUCED,
    SRNA_BACKGROUND,
    WILD_TYPE,
    HYPOMORPHS,
    LINKED_CONTROLS,
    REPLICATE_DESIGN,
    SCREEN_MEDIUM,
    ARRAY_PLATES,
    INCUBATION,
    LONGER_INCUBATION,
    SCORING,
    UNAVERAGED_S_SCORES,
    RESCALED,
    SIGN,
    QUALITY_CONTROL,
    SRNA_NORMALIZED_SEPARATELY,
    TABLE_S2,
)

#: Conservative lower end of "four to six independent measurements" (REPLICATE_DESIGN).
N_SAMPLES: Final = 4


# --------------------------------------------------------------------------- #
# Media: three screen plates, each derived from a library key, no amount invented
# --------------------------------------------------------------------------- #
def _unstated(component: MediaComponent, statement: SourcedValue) -> MediaComponent:
    """A library component of the base medium, kept by identity with no amount."""
    return MediaComponent(
        compound=component.compound,
        role=component.role,
        concentration=None,
        definition=component.definition,
        provenance=[statement],
        note="the paper names the medium but neither its formulation nor this "
        "component's amount, so none is asserted",
    )


def _agar() -> MediaComponent:
    """The plates' gelling agent; the screen pins colonies onto agar, amount unstated."""
    return MediaComponent(
        compound=resolved_compound("agar"),
        role=MediaComponentRole.gelling_agent,
        concentration=None,
        provenance=[ARRAY_PLATES],
        note="agar plates; the amount is not stated",
    )


#: The default screen plate: "mixed into LB agar (except where indicated)".
NICHOLS2011_LB_AGAR: Final = Media(
    name="LB agar, formulation and amounts not stated (Nichols 2011)",
    state="solid",
    is_synthetic=False,
    base_medium="LB",
    components=[*(_unstated(c, SCREEN_MEDIUM) for c in LB.components), _agar()],
    provenance=[SCREEN_MEDIUM, ARRAY_PLATES],
)


def _screen_list_statement(row: str, cell: str) -> SourcedValue:
    """A Table S1 row's ``Concentrations Screened`` cell, verbatim, as its statement."""
    return SourcedValue(
        value=cell,
        quote=cell,
        provenance=Provenance(
            source_uri=SCREEN_LIST_REL,
            citation_key=CITATION_KEY,
            sha256=SCREEN_LIST_SHA256,
            method="Table S1 of the deposited supplement (torchcell-raw mirror), read "
            "with xlrd; the quote is the cell's text",
            page=f"Table S1, row {row!r}, column 'Concentrations Screened'",
        ),
    )


#: The six carbon-source rows read "<carbon source> (M9)" in Table S1.
_M9_ROW = _screen_list_statement("Glucose (M9)", "0.002")
NICHOLS2011_M9_AGAR: Final = Media(
    name="M9 agar, salt amounts not stated, carbon source set per condition "
    "(Nichols 2011)",
    state="solid",
    is_synthetic=True,
    base_medium="M9",
    components=[*(_unstated(c, _M9_ROW) for c in M9.components), _agar()],
    provenance=[_M9_ROW, ARRAY_PLATES],
)

#: The three rows Table S1 marks "(MOPS)": high and low iron, and ammonium chloride.
_MOPS_ROW = _screen_list_statement("NH4Cl (MOPS)", "9.5mm")
NICHOLS2011_MOPS_AGAR: Final = Media(
    name="MOPS agar, composition and carbon source not stated (Nichols 2011)",
    state="solid",
    is_synthetic=True,
    base_medium="MOPS_MINIMAL",
    components=[_agar()],
    provenance=[_MOPS_ROW, ARRAY_PLATES],
)

MediumKey = Literal["lb_agar", "m9_agar", "mops_agar"]
SCREEN_MEDIA: Final[dict[MediumKey, Media]] = {
    "lb_agar": NICHOLS2011_LB_AGAR,
    "m9_agar": NICHOLS2011_M9_AGAR,
    "mops_agar": NICHOLS2011_MOPS_AGAR,
}


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """``DATA_ROOT`` from the environment (the mirror and build tree live under it)."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/nicholsPhenotypicLandscapeBacterial2011``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def sourced_value_root(value: SourcedValue, data_root: str | None = None) -> Path:
    """The mirror root a ``SourcedValue``'s ``source_uri`` is relative to.

    Every quote of this loader sits in the raw mirror under this key: the article OCR
    and Table S1.
    """
    del value
    return Path(data_root or _data_root()) / "torchcell-raw"


def _sha256(path: str | Path) -> str:
    """Streaming sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(data_root: str | None = None) -> Manifest:
    """Read the raw mirror's ``manifest.json`` (written by the #1046 deposit record)."""
    path = raw_mirror_dir(data_root) / "manifest.json"
    return Manifest.model_validate_json(path.read_text())


def manifest_sha256(manifest: Manifest, relpath: str) -> str:
    """The recorded sha256 of one mirror file."""
    for record in manifest.files:
        if record.path == relpath:
            return record.sha256
    raise KeyError(f"{relpath} is not in the raw-mirror manifest")


def article_ocr_records(root: Path) -> list[ArtifactRecord]:
    """The article's MinerU outputs, each hashed in place, the markdown with its record.

    The markdown must hash to ``ARTICLE_MD_SHA256`` (the quotes' anchor) and its
    ``ocr_pdf`` processing record must name the deposited PDF's sha256 as its input.
    """
    processing = ProcessingRecord.model_validate_json(
        (root / ARTICLE_OCR_PROVENANCE).read_text()
    )
    if processing.input_sha256 != [ARTICLE_PDF_SHA256]:
        raise RuntimeError(
            f"{ARTICLE_OCR_PROVENANCE} is not the OCR of {ARTICLE_PDF_REL}"
        )
    md = root / ARTICLE_MD_REL
    if _sha256(md) != ARTICLE_MD_SHA256:
        raise RuntimeError(f"{md} does not hash to its pin {ARTICLE_MD_SHA256}")
    derived = f"mineru OCR of {ARTICLE_PDF_REL}"
    records = [
        ArtifactRecord(
            path=ARTICLE_MD_REL,
            role=ROLE_SI_OCR,
            bytes=md.stat().st_size,
            sha256=ARTICLE_MD_SHA256,
            source=derived,
            processing=processing,
        )
    ]
    others = [(ARTICLE_OCR_PROVENANCE, ROLE_OCR_PROVENANCE)]
    others += [(rel, ROLE_OCR_LAYOUT) for rel in ARTICLE_OCR_LAYOUT]
    others += [
        (str(path.relative_to(root)), ROLE_OCR_IMAGE)
        for path in sorted((root / ARTICLE_OCR_IMAGES).iterdir())
    ]
    for rel, role in others:
        path = root / rel
        records.append(
            ArtifactRecord(
                path=rel,
                role=role,
                bytes=path.stat().st_size,
                sha256=_sha256(path),
                source=derived,
            )
        )
    return records


def record_article_ocr(data_root: str | None = None) -> Path:
    """Extend the raw mirror's ``manifest.json`` with the article OCR and its outputs.

    The deposit itself (``si/mmc1.xls`` to ``si/mmc8.pdf``, ``manual_browser``) was
    recorded by #1046; this adds only the derived files, each pinned. The OCR is
    produced by :func:`torchcell.literature.ocr.ocr_pdf`, never here.
    """
    root = raw_mirror_dir(data_root)
    return extend_manifest(
        root, article_ocr_records(root), citation_key=CITATION_KEY, doi=DOI, title=TITLE
    )


# --------------------------------------------------------------------------- #
# The 114 screens of Table S1 and the 324 conditions of Table S2
# --------------------------------------------------------------------------- #
#: Every dose unit Table S1 uses, and the ``ConcentrationUnit`` plus exact decimal
#: multiplier it is stored as. ``fraction`` is a bare number with no unit (the seven
#: carbon sources and MMS print ``0.006`` and the like); the release's own MMS column
#: label (``MMS-0.05%`` for Table S1's ``0.0005``) is what reads it as a fraction, so it
#: is stored as a percent, and as ``percent`` rather than ``percent_w_v``: no basis is
#: stated anywhere in the release.
SourceUnit = Literal["ug/ml", "ng/ml", "mg/ml", "uM", "mM", "percent", "fraction"]
UNIT_CONVERSIONS: Final[dict[SourceUnit, tuple[ConcentrationUnit, Decimal]]] = {
    "ug/ml": (ConcentrationUnit.ug_per_ml, Decimal(1)),
    "ng/ml": (ConcentrationUnit.ug_per_ml, Decimal("0.001")),
    "mg/ml": (ConcentrationUnit.ug_per_ml, Decimal(1000)),
    "uM": (ConcentrationUnit.micromolar, Decimal(1)),
    "mM": (ConcentrationUnit.millimolar, Decimal(1)),
    "percent": (ConcentrationUnit.percent, Decimal(1)),
    "fraction": (ConcentrationUnit.percent, Decimal(100)),
}


class Dose(BaseModel):
    """One released dose: the compound's resolver label, the amount and its unit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    compound: str
    amount: str
    unit: SourceUnit

    @property
    def concentration(self) -> Concentration:
        """The dose as a typed ``Concentration``, converted per :data:`UNIT_CONVERSIONS`."""
        unit, multiplier = UNIT_CONVERSIONS[self.unit]
        return Concentration(
            value=float(Decimal(self.amount) * multiplier),
            unit=unit,
            basis=DoseBasis.fixed,
        )


ScreenKind = Literal[
    "small_molecule",
    "carbon_source",
    "nitrogen_source",
    "ph",
    "temperature",
    "uv",
    "anaerobic",
]


class Screen(BaseModel):
    """One Table S1 row, transcribed, and the Table S2 label stem its columns carry.

    ``row``, ``cell`` and ``screens`` are the row's ``Antibiotic/Condition``,
    ``Concentrations Screened`` and ``Screens`` cells verbatim (a number cell as its
    text), and :func:`check_screen_list` refuses a build whose bytes read otherwise.
    ``amounts`` holds one tuple per Table S2 column of the screen, one amount per
    compound in ``compounds`` order (a pH, a temperature or an exposure time for the
    physical screens); every number in ``cell`` is one of them or one of
    ``annotation_numbers`` (the ``100`` of "(100mM buffer)", the ``6`` of "G6P").
    ``compounds`` are the chemicals the row names, as Table S1 writes them: the row's
    own text with only the screen's medium marker ("(M9)", "(MOPS)") or condition words
    ("Cobalt stress-", "Iron excess-") taken off, whitespace collapsed, and a
    combination split at its "+" ("Cefsulodin + Mecillinam"). No name is rewritten; the
    curated compound table only attaches an identity where it already holds the name
    (:func:`released_compound`).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    row: str
    cell: str
    screens: int
    stem: str
    kind: ScreenKind
    compounds: tuple[str, ...] = ()
    units: tuple[SourceUnit, ...] = ()
    amounts: tuple[tuple[str, ...], ...]
    annotation_numbers: tuple[str, ...] = ()
    medium: MediumKey = "lb_agar"

    @model_validator(mode="after")
    def _check(self) -> Screen:
        """One amount per compound per column, one column per screen, numbers agree."""
        if len(self.amounts) != self.screens:
            raise ValueError(
                f"{self.row!r}: {len(self.amounts)} columns transcribed, Table S1 "
                f"says {self.screens}"
            )
        if len(self.units) != len(self.compounds):
            raise ValueError(f"{self.row!r}: one unit per compound")
        width = {
            "small_molecule": len(self.compounds),
            "carbon_source": 1,
            "nitrogen_source": 1,
            "ph": 1,
            "temperature": 1,
            "uv": 1,
            "anaerobic": 0,
        }[self.kind]
        if any(len(column) != width for column in self.amounts):
            raise ValueError(f"{self.row!r}: every column carries {width} amount(s)")
        cell_numbers = set(_NUMBER.findall(self.cell))
        transcribed = {a for column in self.amounts for a in column}
        if cell_numbers != transcribed | set(self.annotation_numbers):
            raise ValueError(
                f"{self.row!r}: the cell {self.cell!r} reads {sorted(cell_numbers)}, "
                f"the transcription {sorted(transcribed | set(self.annotation_numbers))}"
            )
        return self


_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _single(
    row: str,
    cell: str,
    stem: str,
    compound: str,
    unit: SourceUnit,
    amounts: tuple[str, ...],
    *,
    medium: MediumKey = "lb_agar",
) -> Screen:
    """A one-compound screen at each of ``amounts``."""
    return Screen(
        row=row,
        cell=cell,
        screens=len(amounts),
        stem=stem,
        kind="small_molecule",
        compounds=(compound,),
        units=(unit,),
        amounts=tuple((a,) for a in amounts),
        medium=medium,
    )


def _carbon(row: str, cell: str, stem: str, compound: str) -> Screen:
    """An M9 carbon-source screen: one plate at Table S1's bare fraction."""
    return Screen(
        row=row,
        cell=cell,
        screens=1,
        stem=stem,
        kind="carbon_source",
        compounds=(compound,),
        units=("fraction",),
        amounts=((cell,),),
        medium="m9_agar",
    )


_UG: Final = "ug/ml"
SCREENS: Final[tuple[Screen, ...]] = (
    _single("A22", "0.5,2,5,15 µg/ml", "A22", "A22", _UG, ("0.5", "2", "5", "15")),
    _carbon("Acetate (M9)", "0.006", "ACETATE", "Acetate"),
    Screen(
        row="acidic pH (MES-HOMOPIPES)",
        cell="4,4.5,5,6 (100mM buffer) ",
        screens=4,
        stem="PH",
        kind="ph",
        amounts=(("4",), ("4.5",), ("5",), ("6",)),
        annotation_numbers=("100",),
    ),
    _single(
        "Acriflavine", "2,10 µg/ml", "ACRIFLAVINE", "Acriflavine", _UG, ("2", "10")
    ),
    _single(
        "Actinomycin D ",
        "2.5,5,10,15 µg/ml",
        "ACTINOMYCIND",
        "Actinomycin D",
        _UG,
        ("2.5", "5", "10", "15"),
    ),
    _single(
        "Amikacin",
        "0.05,0.1,0.2 µg/ml",
        "AMIKACIN",
        "Amikacin",
        _UG,
        ("0.05", "0.1", "0.2"),
    ),
    _single(
        "Amoxicillin ",
        "0.25,0.5,1,1.5 µg/ml",
        "AMOXICILLIN",
        "Amoxicillin",
        _UG,
        ("0.25", "0.5", "1", "1.5"),
    ),
    _single(
        "Ampicillin",
        "1,2,4,8 µg/ml",
        "AMPICILLIN",
        "Ampicillin",
        _UG,
        ("1", "2", "4", "8"),
    ),
    Screen(
        row="Anaerobic",
        cell="-",
        screens=1,
        stem="ANAEROBIC",
        kind="anaerobic",
        amounts=((),),
    ),
    _single(
        "Azidothymidine",
        "0.5,1,2.5 ng/ml",
        "AZIDOTHYMIDINE",
        "Azidothymidine",
        "ng/ml",
        ("0.5", "1", "2.5"),
    ),
    _single(
        "Azithromycin",
        "0.02,0.1,1 µg/ml",
        "AZITHROMYCIN",
        "Azithromycin",
        _UG,
        ("0.02", "0.1", "1"),
    ),
    _single(
        "Aztreonam", "0.02,0.04 µg/ml", "AZTREONAM", "Aztreonam", _UG, ("0.02", "0.04")
    ),
    _single(
        "Bacitracin",
        "100,200,300 µg/ml",
        "BACITRACIN",
        "Bacitracin",
        _UG,
        ("100", "200", "300"),
    ),
    Screen(
        row="basic pH (TAPS)",
        cell="8,9,9.5,10 (100mM buffer) ",
        screens=4,
        stem="PH",
        kind="ph",
        compounds=("TAPS",),
        units=("mM",),
        amounts=(("8",), ("9",), ("9.5",), ("10",)),
        annotation_numbers=("100",),
    ),
    _single(
        "Benzalkonium",
        "1,10,25 µg/ml",
        "BENZALKONIUM",
        "Benzalkonium",
        _UG,
        ("1", "10", "25"),
    ),
    _single(
        "Bicyclomycin", "1,10 µg/ml", "BICYCLOMYCIN", "Bicyclomycin", _UG, ("1", "10")
    ),
    _single(
        "Bile salts",
        "0.1,0.5,1,2 %",
        "BILE",
        "Bile salts",
        "percent",
        ("0.1", "0.5", "1", "2"),
    ),
    _single(
        "Bleomycin",
        "0.1,0.5,1,2 µg/ml",
        "BLEOMYCIN",
        "Bleomycin",
        _UG,
        ("0.1", "0.5", "1", "2"),
    ),
    _single(
        "Calcofluor (F3543    \nFluorescent Brightener 28) ",
        "50 µg/ml",
        "CALCOFLUOR",
        "Calcofluor (F3543 Fluorescent Brightener 28)",
        _UG,
        ("50",),
    ),
    _single(
        "Carbenicillin",
        "0.5,1,1.5 µg/ml",
        "CARBENICILLIN",
        "Carbenicillin",
        _UG,
        ("0.5", "1", "1.5"),
    ),
    _single(
        "CCCP (Carbonyl cyanide 3-chlorophenylhydrazone)",
        "0.1,0.5,2 µg/ml",
        "CCCP",
        "CCCP (Carbonyl cyanide 3-chlorophenylhydrazone)",
        _UG,
        ("0.1", "0.5", "2"),
    ),
    _single(
        "Cecropin B", "0.1,0.3 µg/ml", "CECROPINB", "Cecropin B", _UG, ("0.1", "0.3")
    ),
    _single("Cefaclor", "1,2,3 µg/ml", "CEFACLOR", "Cefaclor", _UG, ("1", "2", "3")),
    _single(
        "Cefoxitin",
        "0.25,0.5,0.75,1 µg/ml",
        "CEFOXITIN",
        "Cefoxitin",
        _UG,
        ("0.25", "0.5", "0.75", "1"),
    ),
    _single(
        "Cefsulodin",
        "6,12,18,24 µg/ml",
        "CEFSULODIN",
        "Cefsulodin",
        _UG,
        ("6", "12", "18", "24"),
    ),
    Screen(
        row="Cefsulodin + Mecillinam",
        cell="6 + 0.03 µg/ml",
        screens=1,
        stem="CEFSULODIN,MECILLINAM",
        kind="small_molecule",
        compounds=("Cefsulodin", "Mecillinam"),
        units=(_UG, _UG),
        amounts=(("6", "0.03"),),
    ),
    _single(
        "Ceftazidime",
        "0.025,0.05,0.075 µg/ml",
        "CEFTAZIDIME",
        "Ceftazidime",
        _UG,
        ("0.025", "0.05", "0.075"),
    ),
    _single(
        "Cerulenin",
        "1,2,4,6  µg/ml ",
        "CERULENIN",
        "Cerulenin",
        _UG,
        ("1", "2", "4", "6"),
    ),
    _single(
        "CHIR-090",
        "0.02,0.025,0.04,0.05,0.075 µg/ml",
        "CHIR090",
        "CHIR-090",
        _UG,
        ("0.02", "0.025", "0.04", "0.05", "0.075"),
    ),
    _single(
        "Chloramphenicol",
        "0.5,1,1.5,2 µg/ml",
        "CHLORAMPHENICOL",
        "Chloramphenicol",
        _UG,
        ("0.5", "1", "1.5", "2"),
    ),
    _single(
        "Chlorpromazine",
        "3,6,12,24 µM",
        "CHLOROPROMAZINE",
        "Chlorpromazine",
        "uM",
        ("3", "6", "12", "24"),
    ),
    _single(
        "Cholate",
        "0.1,0.5,1,2 %",
        "CHOLATE",
        "Cholate",
        "percent",
        ("0.1", "0.5", "1", "2"),
    ),
    _single(
        "Ciprofloxacin",
        "0.004,0.006,0.008 µg/ml",
        "CIPROFLOXACIN",
        "Ciprofloxacin",
        _UG,
        ("0.004", "0.006", "0.008"),
    ),
    _single(
        "Cisplatin",
        "20,50,100 µg/ml",
        "CISPLATIN",
        "Cisplatin",
        _UG,
        ("20", "50", "100"),
    ),
    _single(
        "Clarythromycin",
        "0.1,1,5,10 µg/ml",
        "CLARYTHROMYCIN",
        "Clarythromycin",
        _UG,
        ("0.1", "1", "5", "10"),
    ),
    _single(
        "Cobalt stress-CoCl2",
        "0.1,0.5 mM (LB)",
        "HIGHCOBALT",
        "CoCl2",
        "mM",
        ("0.1", "0.5"),
    ),
    Screen(
        row="Cold shock ",
        cell="16,18,20 °C",
        screens=3,
        stem="TEMPERATURE",
        kind="temperature",
        amounts=(("16",), ("18",), ("20",)),
    ),
    _single(
        "Copper stress-CuCl2",
        "1,2,4 mM (LB)",
        "HIGHCOPPER",
        "CuCl2",
        "mM",
        ("1", "2", "4"),
    ),
    _single("Cycloserine D", "16 µg/ml", "CYCLOSERINED", "Cycloserine D", _UG, ("16",)),
    _single(
        "Deoxycholate",
        "0.1,0.5,2 %",
        "DEOXYCHOLATE",
        "Deoxycholate",
        "percent",
        ("0.1", "0.5", "2"),
    ),
    _single(
        "Dibucaine",
        "0.4,0.8,1.2 mM",
        "DIBUCAINE",
        "Dibucaine",
        "mM",
        ("0.4", "0.8", "1.2"),
    ),
    _single(
        "Doxorubicin", "1,10 µg/ml", "DOXORUBICIN", "Doxorubicin", _UG, ("1", "10")
    ),
    _single(
        "Doxycycline ",
        "0.25,0.5,0.75,1 µg/ml",
        "DOXYCYCLINE",
        "Doxycycline",
        _UG,
        ("0.25", "0.5", "0.75", "1"),
    ),
    _single("EDTA", "0.1, 0.5, 1 mM", "EDTA", "EDTA", "mM", ("0.1", "0.5", "1")),
    _single(
        "EGTA", "0.1, 0.5, 1, 2 mM", "EGTA", "EGTA", "mM", ("0.1", "0.5", "1", "2")
    ),
    _single(
        "Epigallocatechin gallate (EGCG)",
        "5,20,50 µM",
        "EGCG",
        "Epigallocatechin gallate (EGCG)",
        "uM",
        ("5", "20", "50"),
    ),
    _single(
        "Epinephrine",
        "50,250,1000 µM ",
        "EPINEPHRINE",
        "Epinephrine",
        "uM",
        ("50", "250", "1000"),
    ),
    _single(
        "Erythromycin",
        "0.1,1,5,10 µg/ml",
        "ERYTHROMYCIN",
        "Erythromycin",
        _UG,
        ("0.1", "1", "5", "10"),
    ),
    _single(
        "Ethidium Bromide",
        "2,10,50 µg/ml",
        "ETHIDIUMBROMIDE",
        "Ethidium Bromide",
        _UG,
        ("2", "10", "50"),
    ),
    _single("EtOH", "2,4,6 %", "ETHANOL", "EtOH", "percent", ("2", "4", "6")),
    _single("Fosfomycin", "1 µg/ml", "FOSFOMYCIN", "Fosfomycin", _UG, ("1",)),
    Screen(
        row="Fosfomycin +Glucose 6P",
        cell="0.05,0.2 µg/ml (in 50 mg/ml G6P) ",
        screens=2,
        stem="GLUFOSFOMYCIN",
        kind="small_molecule",
        compounds=("Fosfomycin", "Glucose 6P"),
        units=(_UG, "mg/ml"),
        amounts=(("0.05", "50"), ("0.2", "50")),
        annotation_numbers=("6",),
    ),
    _single(
        "Fusidic acid",
        "1,5,20,50 µg/ml",
        "FUSIDICACID",
        "Fusidic acid",
        _UG,
        ("1", "5", "20", "50"),
    ),
    _single(
        "Gentamicin", "0.05,0.1 µg/ml", "GENTAMICIN", "Gentamicin", _UG, ("0.05", "0.1")
    ),
    _carbon("Glucosamine (M9)", "0.002", "GLUCOSAMINE", "Glucosamine"),
    _carbon("Glucose (M9)", "0.002", "GLUCOSE", "Glucose"),
    _carbon("Glycerol (M9)", "0.004", "GLYCEROL", "Glycerol"),
    Screen(
        row="Heat shock ",
        cell="40,42,43.5,45 °C",
        screens=4,
        stem="TEMPERATURE",
        kind="temperature",
        amounts=(("40",), ("42",), ("43.5",), ("45",)),
    ),
    _single(
        "Hydrogen peroxide",
        "0.1,0.5,1,2 mM",
        "PEROXIDE",
        "Hydrogen peroxide",
        "mM",
        ("0.1", "0.5", "1", "2"),
    ),
    _single(
        "Hydroxyurea", "1,5,10 mM", "HYDROXYUREA", "Hydroxyurea", "mM", ("1", "5", "10")
    ),
    _single("Indolicidin", "0.1 µg/ml", "INDOLICIDIN", "Indolicidin", _UG, ("0.1",)),
    Screen(
        row="Iron excess-FeSO4",
        cell="1mM (MOPS; normal 100µM)",
        screens=1,
        stem="HIGHFE",
        kind="small_molecule",
        compounds=("FeSO4",),
        units=("mM",),
        amounts=(("1",),),
        annotation_numbers=("100",),
        medium="mops_agar",
    ),
    Screen(
        row="Iron starvation-FeSO4",
        cell="2µM (MOPS; normal 100µM)",
        screens=1,
        stem="LOWFE",
        kind="small_molecule",
        compounds=("FeSO4",),
        units=("uM",),
        amounts=(("2",),),
        annotation_numbers=("100",),
        medium="mops_agar",
    ),
    _single(
        "Isoniazid", "0.2,1,1.5 mM", "ISONIAZID", "Isoniazid", "mM", ("0.2", "1", "1.5")
    ),
    _single(
        "Levofloxacin", "0.002 µg/ml", "LEVOFLOXACIN", "Levofloxacin", _UG, ("0.002",)
    ),
    _carbon("Maltose  (M9)", "0.001", "MALTOSE", "Maltose"),
    _single(
        "Mecillinam",
        "0.03,0.06,0.09,0.12 µg/ml",
        "MECILLINAM",
        "Mecillinam",
        _UG,
        ("0.03", "0.06", "0.09", "0.12"),
    ),
    _single(
        "Methotrexate", "1,25 µg/ml", "METHOTREXATE", "Methotrexate", _UG, ("1", "25")
    ),
    _single(
        "Minocycline",
        "0.2,0.5,1 µg/ml",
        "MINOCYCLINE",
        "Minocycline",
        _UG,
        ("0.2", "0.5", "1"),
    ),
    _single("Mitomycin C", "0.1 µg/ml", "MITOMYCINC", "Mitomycin C", _UG, ("0.1",)),
    _single("MMS", "0.0005", "MMS", "MMS", "fraction", ("0.0005",)),
    Screen(
        row="N-acetyl Glucosamine",
        cell="0.0015",
        screens=1,
        stem="N-ACETYLGLUCOSAMINE",
        kind="carbon_source",
        compounds=("N-acetyl Glucosamine",),
        units=("fraction",),
        amounts=(("0.0015",),),
    ),
    _single(
        "NaCl", "150,300,450,600 mM", "NACL", "NaCl", "mM", ("150", "300", "450", "600")
    ),
    _single(
        "Nalidixic acid",
        "0.5,1,1.5,2 µg/ml",
        "NALIDIXICACID",
        "Nalidixic acid",
        _UG,
        ("0.5", "1", "1.5", "2"),
    ),
    Screen(
        row="NH4Cl (MOPS)",
        cell="9.5mm",
        screens=1,
        stem="NH4CL",
        kind="nitrogen_source",
        compounds=("NH4Cl",),
        units=("mM",),
        amounts=(("9.5",),),
        medium="mops_agar",
    ),
    _single(
        "Nickel stress-NiCl2",
        "0.1,1 mM (LB)",
        "HIGHNICKEL",
        "NiCl2",
        "mM",
        ("0.1", "1"),
    ),
    _single(
        "Nigericin", "0.1,1,5 µM", "NIGERICIN", "Nigericin", "uM", ("0.1", "1", "5")
    ),
    _single(
        "Nitrofurnatoin",
        "0.1,0.5,1,1.5,2 µg/ml",
        "NITROFURANTOIN",
        "Nitrofurnatoin",
        _UG,
        ("0.1", "0.5", "1", "1.5", "2"),
    ),
    _single(
        "Norepinephrine",
        "100,1000 µM",
        "NOREPINEPHRINE",
        "Norepinephrine",
        "uM",
        ("100", "1000"),
    ),
    _single(
        "Norfloxacin",
        "0.01,0.02,0.04 µg/ml",
        "NORFLOXACIN",
        "Norfloxacin",
        _UG,
        ("0.01", "0.02", "0.04"),
    ),
    _single(
        "Novobiocin",
        "4,6,8,10,12,30 µg/ml ",
        "NOVOBIOCIN",
        "Novobiocin",
        _UG,
        ("4", "6", "8", "10", "12", "30"),
    ),
    _single(
        "Oxacillin", "0.5,5,40 µg/ml", "OXACILLIN", "Oxacillin", _UG, ("0.5", "5", "40")
    ),
    _single(
        "Paraquat dichloride",
        "0.2,1,5,10,18 µM",
        "PARAQUAT",
        "Paraquat dichloride",
        "uM",
        ("0.2", "1", "5", "10", "18"),
    ),
    _single(
        "Phenazine methosulfate (PMS)",
        "0.02,0.05,0.1 mM",
        "PMS",
        "Phenazine methosulfate (PMS)",
        "mM",
        ("0.02", "0.05", "0.1"),
    ),
    _single(
        "Phleomycin",
        "0.2,0.5,1 µg/ml",
        "PHLEOMYCIN",
        "Phleomycin",
        _UG,
        ("0.2", "0.5", "1"),
    ),
    _single(
        "Polymyxin B",
        "1,2,4,6 µg/ml",
        "POLYMYXINB",
        "Polymyxin B",
        _UG,
        ("1", "2", "4", "6"),
    ),
    _single(
        "Procaine", "1,5,10,30 mM", "PROCAINE", "Procaine", "mM", ("1", "5", "10", "30")
    ),
    _single(
        "Propidium iodide",
        "1,20,50 µg/ml",
        "PROPIDIUMIODIDE",
        "Propidium iodide",
        _UG,
        ("1", "20", "50"),
    ),
    _single(
        "Puromycin", "1,5,25 µg/ml", "PUROMYCIN", "Puromycin", _UG, ("1", "5", "25")
    ),
    _single(
        "Pyocyanin", "0.2,1,10 µg/ml", "PYOCYANIN", "Pyocyanin", _UG, ("0.2", "1", "10")
    ),
    _single("Radicicol", "1,5,10 µM", "RADICICOL", "Radicicol", "uM", ("1", "5", "10")),
    _single("Rifampicin", "1,2 µg/ml ", "RIFAMPICIN", "Rifampicin", _UG, ("1", "2")),
    _single(
        "SDS", "0.5,1,2,3,4 %", "SDS", "SDS", "percent", ("0.5", "1", "2", "3", "4")
    ),
    Screen(
        row="SDS+EDTA",
        cell="0.5%/0.1mM,0.5%/0.5mM,1%/0.5mM",
        screens=3,
        stem="SDS/EDTA",
        kind="small_molecule",
        compounds=("SDS", "EDTA"),
        units=("percent", "mM"),
        amounts=(("0.5", "0.1"), ("0.5", "0.5"), ("1", "0.5")),
    ),
    _single(
        "Spectinomycin", "4,6 µg/ml ", "SPECTINOMYCIN", "Spectinomycin", _UG, ("4", "6")
    ),
    _single(
        "Spiramycin", "1,5,20 µg/ml", "SPIRAMYCIN", "Spiramycin", _UG, ("1", "5", "20")
    ),
    _single(
        "Streptomycin", "0.05 µg/ml", "STREPTOMYCIN", "Streptomycin", _UG, ("0.05",)
    ),
    _single(
        "Streptonigrin",
        "0.1,0.2,0.5 µg/ml",
        "STREPTONIGRIN",
        "Streptonigrin",
        _UG,
        ("0.1", "0.2", "0.5"),
    ),
    _carbon("Succinate (M9)", "0.003", "SUCCINATE", "Succinate"),
    _single(
        "Sulfamethizole",
        "100,200,300 µg/ml",
        "SULFAMETHIZOLE",
        "Sulfamethizole",
        _UG,
        ("100", "200", "300"),
    ),
    _single(
        "Sulfamonomethoxine",
        "50,100 µg/ml",
        "SULFAMONOMETHOXINE",
        "Sulfamonomethoxine",
        _UG,
        ("50", "100"),
    ),
    _single(
        "Taurocholate",
        "0.1,0.5,1 %",
        "TAUROCHOLATE",
        "Taurocholate",
        "percent",
        ("0.1", "0.5", "1"),
    ),
    _single(
        "Tetracycline",
        "0.25,0.5,0.75,1 µg/ml",
        "TETRACYCLINE",
        "Tetracycline",
        _UG,
        ("0.25", "0.5", "0.75", "1"),
    ),
    _single(
        "Theophylline",
        "10,100 µg/ml",
        "THEOPHYLLINE",
        "Theophylline",
        _UG,
        ("10", "100"),
    ),
    _single(
        "Thiolactomycin",
        "1,5,50 µM",
        "THIOLACTOMYCIN",
        "Thiolactomycin",
        "uM",
        ("1", "5", "50"),
    ),
    _single(
        "Tobramycin",
        "0.05,0.1,0.2,0.4 µg/ml",
        "TOBRAMYCIN",
        "Tobramycin",
        _UG,
        ("0.05", "0.1", "0.2", "0.4"),
    ),
    _single(
        "Triclosan/Irgasan",
        "0.05 µg/ml",
        "TRICLOSAN",
        "Triclosan/Irgasan",
        _UG,
        ("0.05",),
    ),
    _single(
        "Trimethoprim",
        "0.1,0.2,0.3,0.4 µg/ml",
        "TRIMETHOPRIM",
        "Trimethoprim",
        _UG,
        ("0.1", "0.2", "0.3", "0.4"),
    ),
    Screen(
        row="Trimethoprim + Sulfamethizole",
        cell="0.1 + 50 µg/ml",
        screens=1,
        stem="TRIMETHOPRIM,SULFAMETHIZOLE",
        kind="small_molecule",
        compounds=("Trimethoprim", "Sulfamethizole"),
        units=(_UG, _UG),
        amounts=(("0.1", "50"),),
    ),
    _single(
        "Triton X-100",
        "0.01,0.03,0.2 %",
        "TRITONX",
        "Triton X-100",
        "percent",
        ("0.01", "0.03", "0.2"),
    ),
    _single(
        "Tunicamycin",
        "1,3,7.5 µg/ml",
        "TUNICAMYCIN",
        "Tunicamycin",
        _UG,
        ("1", "3", "7.5"),
    ),
    Screen(
        row="UV",
        cell="6,12,18,24 sec",
        screens=4,
        stem="UV",
        kind="uv",
        amounts=(("6",), ("12",), ("18",), ("24",)),
    ),
    _single(
        "Vancomycin",
        "10,20,50 µg/ml",
        "VANCOMYCIN",
        "Vancomycin",
        _UG,
        ("10", "20", "50"),
    ),
    _single(
        "Verapamil", "0.1,0.5,1 mM", "VERAPAMIL", "Verapamil", "mM", ("0.1", "0.5", "1")
    ),
)


class Condition(BaseModel):
    """One Table S2 column: its verbatim label and the Table S1 column it is.

    ``amounts`` is the matched Table S1 column (one amount per compound, or the pH,
    temperature or exposure time), which is what every record stores; the label's own
    numbers are only the key that matched it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    screen: Screen
    amounts: tuple[str, ...]

    @property
    def doses(self) -> tuple[Dose, ...]:
        """The dosed compounds of a small-molecule or nutrient screen, in label order."""
        if self.screen.kind in ("ph", "temperature", "uv", "anaerobic"):
            return ()
        return tuple(
            Dose(compound=c, amount=a, unit=u)
            for c, a, u in zip(
                self.screen.compounds, self.amounts, self.screen.units, strict=True
            )
        )

    @property
    def temperature_c(self) -> float:
        """The plate's temperature: the shock's own value, else 37 C (INCUBATION)."""
        if self.screen.kind == "temperature":
            return float(self.amounts[0])
        return float(INCUBATION.value)


_TEMPERATURE_LABEL = re.compile(r"^(\d+(?:\.\d+)?)C$")
_PH_LABEL = re.compile(r"^PH(\d+(?:\.\d+)?)$")
_UV_LABEL = re.compile(r"^UV-(\d+)SEC$")
_SDS_EDTA_LABEL = re.compile(r"^SDS(\d+(?:\.\d+)?)%/EDTA(\d+(?:\.\d+)?)$")
_PART = re.compile(
    r"^(?P<stem>[A-Z0-9][A-Z0-9-]*?)(?:-(?P<amount>\d+(?:\.\d+)?)(?P<pct>%)?)?$"
)
#: The two label suffixes the release writes after " - " (309 and 15 columns).
LABEL_SUFFIXES: Final = ("UNSPECIFIED", "")


class ParsedLabel(BaseModel):
    """A Table S2 condition label split into its screen stem and its numbers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stem: str
    amounts: tuple[str, ...]
    percent: bool = False


def parse_label(label: str) -> ParsedLabel:
    """Split ``<body> - <suffix>`` into the Table S1 stem and the label's amounts."""
    body, separator, suffix = label.rpartition(" - ")
    if not separator or suffix not in LABEL_SUFFIXES:
        raise ValueError(f"{label!r} is not '<condition> - <suffix>'")
    for pattern, stem in (
        (_TEMPERATURE_LABEL, "TEMPERATURE"),
        (_PH_LABEL, "PH"),
        (_UV_LABEL, "UV"),
    ):
        match = pattern.fullmatch(body)
        if match is not None:
            return ParsedLabel(stem=stem, amounts=(match[1],))
    match = _SDS_EDTA_LABEL.fullmatch(body)
    if match is not None:
        return ParsedLabel(stem="SDS/EDTA", amounts=(match[1], match[2]), percent=True)
    parts = [_PART.fullmatch(part) for part in body.split(",")]
    if any(part is None for part in parts):
        raise ValueError(f"{label!r}: unparsed condition body {body!r}")
    matched = [part for part in parts if part is not None]
    amounts = tuple(p["amount"] for p in matched if p["amount"] is not None)
    if amounts and len(amounts) != len(matched):
        raise ValueError(f"{label!r}: a combination names an amount for every part")
    return ParsedLabel(
        stem=",".join(p["stem"] for p in matched),
        amounts=amounts,
        percent=any(p["pct"] is not None for p in matched),
    )


def _same_amount(label_amount: str, percent: bool, amount: str, unit: str) -> bool:
    """A label number names a Table S1 amount: equal, or the percent of a fraction."""
    released = Decimal(amount)
    if percent and unit == "fraction":
        released *= 100
    return Decimal(label_amount) == released


def match_condition(label: str) -> Condition:
    """The one Table S1 column a Table S2 label names; anything else is refused."""
    parsed = parse_label(label)
    candidates: list[Condition] = []
    for screen in SCREENS:
        if screen.stem != parsed.stem:
            continue
        for column in screen.amounts:
            if not parsed.amounts:
                if screen.screens == 1:
                    candidates.append(
                        Condition(label=label, screen=screen, amounts=column)
                    )
                continue
            units = screen.units or ("",) * len(column)
            if len(parsed.amounts) <= len(column) and all(
                _same_amount(a, parsed.percent, b, u)
                for a, b, u in zip(parsed.amounts, column, units, strict=False)
            ):
                candidates.append(Condition(label=label, screen=screen, amounts=column))
    if len(candidates) != 1:
        raise ValueError(
            f"{label!r} matches {len(candidates)} Table S1 columns, not exactly one"
        )
    return candidates[0]


def match_conditions(labels: Sequence[str]) -> tuple[Condition, ...]:
    """Every Table S2 column matched, and every Table S1 column matched exactly once."""
    conditions = tuple(match_condition(label) for label in labels)
    used = Counter((c.screen.row, c.amounts) for c in conditions)
    expected = {(s.row, column) for s in SCREENS for column in s.amounts}
    if set(used) != expected or any(n != 1 for n in used.values()):
        missing = sorted(map(str, expected - set(used)))
        repeated = sorted(str(k) for k, n in used.items() if n != 1)
        raise ValueError(
            f"Table S2 columns do not map one to one onto Table S1: missing {missing}, "
            f"repeated {repeated}"
        )
    return conditions


def read_screen_list(path: str | Path) -> list[tuple[str, str, int]]:
    """Table S1's (condition, concentrations screened, screens) rows, as text.

    A number cell is read as its text (``0.006``) so the transcription in
    :data:`SCREENS` can be compared with it character for character.
    """
    book = xlrd.open_workbook(str(path))
    sheet = book.sheet_by_index(0)
    if sheet.cell_value(0, 0) != SCREEN_LIST_TITLE:
        raise ValueError(f"{path}: title cell is {sheet.cell_value(0, 0)!r}")
    header = tuple(str(sheet.cell_value(1, c)) for c in range(sheet.ncols))
    if header != SCREEN_LIST_HEADER:
        raise ValueError(f"{path}: header is {header}")
    rows: list[tuple[str, str, int]] = []
    for r in range(2, sheet.nrows):
        name = sheet.cell_value(r, 0)
        if name == "":
            continue
        cell = sheet.cell(r, 4)
        text = (
            repr(cell.value) if cell.ctype == xlrd.XL_CELL_NUMBER else str(cell.value)
        )
        rows.append((str(name), text, int(sheet.cell_value(r, 5))))
    return rows


def check_screen_list(path: str | Path) -> None:
    """Refuse a build whose Table S1 reads differently from :data:`SCREENS`."""
    released = read_screen_list(path)
    transcribed = [(s.row, s.cell, s.screens) for s in SCREENS]
    if released != transcribed:
        diff = [(a, b) for a, b in zip(released, transcribed, strict=False) if a != b]
        raise ValueError(
            f"Table S1 ({len(released)} rows) disagrees with the transcription "
            f"({len(transcribed)} rows): {diff[:5]}"
        )
    if sum(s.screens for s in SCREENS) != RELEASED_CONDITIONS:
        raise ValueError("Table S1's Screens column does not sum to 324")


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
_DURATION_GAP = ProvenanceGap(
    field="duration_hours",
    reason=ProvenanceGapReason.not_reported_by_primary,
    note="'For the majority of conditions, plates were removed from the incubator "
    "following 14-16 hrs growth at 37 C' and 'Several stresses, e.g. low temperature, "
    "required longer incubation times': a range for most plates, unlisted exceptions, "
    "and no per-condition time in Table S1",
)
_IRRADIANCE_GAP = ProvenanceGap(
    field="irradiance_w_per_m2",
    reason=ProvenanceGapReason.not_reported_by_primary,
    note="Table S1 gives the UV dose as an exposure time ('6,12,18,24 sec') and the "
    "article and its Extended Experimental Procedures name no lamp, wavelength or "
    "irradiance",
)
_FLUENCE_GAP = ProvenanceGap(
    field="fluence_j_per_m2",
    reason=ProvenanceGapReason.not_reported_by_primary,
    note="a fluence is an irradiance times a time and the irradiance is absent; it is "
    "not back-computed",
)
_ACIDIC_BUFFER_GAP = ProvenanceGap(
    field="agent",
    reason=ProvenanceGapReason.not_reported_by_primary,
    note="Table S1 names two buffers for the acidic series, 'acidic pH "
    "(MES-HOMOPIPES)', at '4,4.5,5,6 (100mM buffer)', and does not say which buffer "
    "set which pH or whether they were mixed",
)
#: The basic series' one named buffer, at the 100 mM its Table S1 cell states.
BASIC_BUFFER_MM: Final = "100"


def released_compound(name: str) -> Compound:
    """The compound under the name Table S1 gives it; the table only attaches identity.

    The curated compound table is asked for ``name``. Where it already holds that name
    (or one of its recorded synonyms), the row's InChIKey and cross-references are
    attached; the stored ``name`` stays the released one, so the record says what the
    source says. A name the table lacks keeps the typed ``inchikey`` gap
    ``resolved_compound`` attaches, and is stored all the same.
    """
    return resolved_compound(name).model_copy(update={"name": name})


def small_molecule(dose: Dose) -> SmallMoleculePerturbation:
    """One dosed chemical under its Table S1 name, at its Table S1 dose."""
    return SmallMoleculePerturbation(
        compound=released_compound(dose.compound), concentration=dose.concentration
    )


def nutrient(dose: Dose, factor: PhysicalFactor) -> EnvironmentPhysicalPerturbation:
    """A carbon or nitrogen source as the varied physical factor (the Tong 2020 shape)."""
    return EnvironmentPhysicalPerturbation(
        factor=factor,
        agent=released_compound(dose.compound),
        magnitude=dose.concentration,
    )


def ph_perturbations(condition: Condition) -> list[EnvironmentPerturbationType]:
    """The buffered pH of an acidic or basic plate (and TAPS at 100 mM when named)."""
    ph = EnvironmentPhysicalPerturbation(
        factor=PhysicalFactor.ph,
        magnitude=Concentration(
            value=float(condition.amounts[0]),
            unit=ConcentrationUnit.ph,
            basis=DoseBasis.fixed,
        ),
        agent=(
            released_compound(condition.screen.compounds[0])
            if condition.screen.compounds
            else None
        ),
        provenance_gaps=[] if condition.screen.compounds else [_ACIDIC_BUFFER_GAP],
    )
    if not condition.screen.compounds:
        return [ph]
    buffer = Dose(
        compound=condition.screen.compounds[0],
        amount=BASIC_BUFFER_MM,
        unit=condition.screen.units[0],
    )
    return [ph, small_molecule(buffer)]


def radiation(condition: Condition) -> PhysicalExposurePerturbation:
    """The UV exposure, dosed by Table S1's exposure time; irradiance and fluence gapped."""
    return PhysicalExposurePerturbation(
        factor=PhysicalFactor.radiation,
        exposure_duration_seconds=float(condition.amounts[0]),
        provenance_gaps=[_IRRADIANCE_GAP, _FLUENCE_GAP],
    )


def environment(condition: Condition) -> Environment:
    """The screen plate of one Table S2 column."""
    kind = condition.screen.kind
    perturbations: list[EnvironmentPerturbationType] = []
    if kind == "small_molecule":
        perturbations.extend(small_molecule(d) for d in condition.doses)
    elif kind == "carbon_source":
        perturbations.extend(
            nutrient(d, PhysicalFactor.carbon_source) for d in condition.doses
        )
    elif kind == "nitrogen_source":
        perturbations.extend(
            nutrient(d, PhysicalFactor.nitrogen_source) for d in condition.doses
        )
    elif kind == "ph":
        perturbations.extend(ph_perturbations(condition))
    elif kind == "uv":
        perturbations.append(radiation(condition))
    return Environment(
        media=SCREEN_MEDIA[condition.screen.medium],
        temperature=Temperature(value=condition.temperature_c),
        perturbations=perturbations,
        aerobicity="anaerobic" if kind == "anaerobic" else "aerobic",
        duration_hours=None,
        provenance_gaps=[_DURATION_GAP],
    )


# --------------------------------------------------------------------------- #
# Phenotype
# --------------------------------------------------------------------------- #
_GAP_UNCERTAINTY = ProvenanceGap(
    field="environment_response_uncertainty",
    reason=ProvenanceGapReason.not_reported_by_primary,
    note="Table S2 releases one re-scaled score per cell and no dispersion; the score "
    "is itself a standardized statistic",
)
_GAP_SE = ProvenanceGap(
    field="environment_response_se",
    reason=ProvenanceGapReason.not_reported_by_primary,
    note="no uncertainty is released, so no standard error can be derived",
)
PHENOTYPE_GAPS: Final = (_GAP_UNCERTAINTY, _GAP_SE)


def phenotype(
    value: float, condition: Condition, replicate_id: str | None = None
) -> EnvironmentResponsePhenotype:
    """One released Table S2 score of one strain row in one screen."""
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.z_score,
        assay_type=AssayType.colony_size_array,
        environment_response=value,
        n_samples=N_SAMPLES,
        sample_unit=SampleUnit.colony,
        screen_id=condition.label,
        replicate_id=replicate_id,
        units=UNITS,
        provenance_gaps=list(PHENOTYPE_GAPS),
    )


def reference_phenotype(condition: Condition) -> EnvironmentResponsePhenotype:
    """The screen's typical strain: the re-scaled score 0."""
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.z_score,
        assay_type=AssayType.colony_size_array,
        environment_response=0.0,
        screen_id=condition.label,
        units=UNITS_REFERENCE,
    )


# --------------------------------------------------------------------------- #
# Strains: the 3,979 Table S2 row labels, stored as released
# --------------------------------------------------------------------------- #
#: The label suffixes the release writes for the non-deletion arrays, and the class
#: each states; a label with none of them is a deletion.
SUFFIX_KINDS: Final[dict[str, ReleasedKind]] = {
    " - SPA": ReleasedKind.affinity_tag,
    " - DAS": ReleasedKind.degron,
    " - Linked": ReleasedKind.linked_marker,
    " - Allele": ReleasedKind.sequence_allele,
    " - Truncation": ReleasedKind.truncation,
}
#: What the label's leading identifier is, as the release writes it.
IDENTIFIER_KIND: Final = "ECK accession"
#: The collection each class's strains come from, as the Extended Experimental
#: Procedures name it (COLLECTION); None where the release does not say which of its
#: collections a row belongs to (a deletion row is a Keio or an sRNA/small protein
#: strain, and Table S2 does not mark which).
KIND_COLLECTIONS: Final[dict[ReleasedKind, str]] = {
    ReleasedKind.affinity_tag: SPA_COLLECTION,
    ReleasedKind.degron: DAS_COLLECTION,
}
_ECK_LABEL = re.compile(r"^ECK(?P<numbers>\d{4}(?:[/-]\d{4})*)(?:-(?P<rest>.+))?$")
_ALLELE_REST = re.compile(
    r"^(?:(?P<clone>[A-D])(?:-|$))?(?P<symbol>[A-Z][A-Z0-9]*?)?"
    r"(?P<detail>\(.*\)|\*|\d+)?(?P<degron>-DAS\+4|-DAS)?(?P<marker>-SPA|-KAN)?$"
)


class StrainLabel(BaseModel):
    """What one Table S2 row label says, split on its own grammar and nothing else.

    ``ecks`` are the label's ECK accessions in released order (several for a
    multi-gene deletion, ``ECK1205/1207/1209-RDLABC``, whose symbol names the set and so
    no single gene); ``symbol`` is the label's gene symbol as released, its trailing
    prime kept (``ECK2350-YFDM'``), or ``None`` when the label carries none.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    index: int
    label: str
    kind: ReleasedKind
    ecks: tuple[str, ...]
    symbol: str | None = None


def parse_strain_label(index: int, label: str) -> StrainLabel:
    """Split a row label into its class (by suffix), its ECK ids and its symbol."""
    kind = ReleasedKind.deletion
    body = label
    for suffix, suffix_kind in SUFFIX_KINDS.items():
        if label.endswith(suffix):
            kind, body = suffix_kind, label[: -len(suffix)]
            break
    primed = body.endswith("'")
    match = _ECK_LABEL.fullmatch(body.rstrip("'"))
    if match is None:
        raise ValueError(f"row {index}: {label!r} is not an ECK label")
    ecks = tuple(f"ECK{n}" for n in re.split(r"[/-]", match["numbers"]))
    rest = match["rest"]
    if len(ecks) > 1:
        if kind is not ReleasedKind.deletion:
            raise ValueError(f"row {index}: {label!r} is a multi-gene allele")
        return StrainLabel(index=index, label=label, kind=kind, ecks=ecks)
    if kind is ReleasedKind.deletion:
        symbol = None if rest is None else rest + ("'" if primed else "")
        return StrainLabel(
            index=index, label=label, kind=kind, ecks=ecks, symbol=symbol
        )
    parts = _ALLELE_REST.fullmatch(rest or "")
    if parts is None:
        raise ValueError(f"row {index}: {label!r}: unparsed allele label {rest!r}")
    return StrainLabel(
        index=index, label=label, kind=kind, ecks=ecks, symbol=parts["symbol"]
    )


def released_gene_names(labels: Sequence[StrainLabel]) -> dict[str, str]:
    """Each ECK accession's name as Table S2 releases it.

    A single-gene label names its gene (its symbol, else the accession itself). A
    multi-gene label names only the set (``RDLABC``), so each of its genes takes the
    name the release gives that accession on its own row (``ECK4143-ECNB``), else the
    accession. An accession released under two names is refused rather than chosen
    between (measured: none is).
    """
    names: dict[str, set[str]] = {}
    for row in labels:
        if len(row.ecks) == 1:
            names.setdefault(row.ecks[0], set()).add(row.symbol or row.ecks[0])
    conflicts = {eck: sorted(n) for eck, n in names.items() if len(n) > 1}
    if conflicts:
        raise ValueError(f"accessions released under two names: {conflicts}")
    out = {eck: next(iter(n)) for eck, n in names.items()}
    for row in labels:
        for eck in row.ecks:
            out.setdefault(eck, eck)
    return out


class StrainRow(BaseModel):
    """One Table S2 row: its label, the released name of each gene, its replicate id.

    ``replicate_id`` is set only on the 24 rows whose label heads two rows (the
    spreadsheet row, ``Table S2 row <n>``): the release gives the two rows one strain
    identifier and two different score profiles, and both are stored, told apart by it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    index: int
    label: StrainLabel
    gene_names: tuple[str, ...]
    replicate_id: str | None = None


#: Table S2's first data row is spreadsheet row 3 (title row 1, header row 2).
FIRST_DATA_ROW: Final = 3


def strain_rows(labels: Sequence[str]) -> tuple[StrainRow, ...]:
    """Every Table S2 row, in released order; nothing is looked up and nothing dropped."""
    parsed = [parse_strain_label(i, label) for i, label in enumerate(labels)]
    names = released_gene_names(parsed)
    counts = Counter(labels)
    return tuple(
        StrainRow(
            index=row.index,
            label=row,
            gene_names=tuple(names[eck] for eck in row.ecks),
            replicate_id=(
                f"Table S2 row {row.index + FIRST_DATA_ROW}"
                if counts[row.label] > 1
                else None
            ),
        )
        for row in parsed
    )


# --------------------------------------------------------------------------- #
# Genotype
# --------------------------------------------------------------------------- #
BW25113_NAMESPACE: Final = STRAIN_GENE_NAMESPACES["BW25113"]


def genotype(row: StrainRow) -> Genotype:
    """The row's strain: one released-identifier perturbation per ECK accession.

    ``gene_namespace`` names the study's wild type (WILD_TYPE); the accessions are not
    tags of it, and no annotation is read.
    """
    return Genotype(
        perturbations=[
            ReleasedIdentifierPerturbation(
                systematic_gene_name=eck,
                perturbed_gene_name=name,
                gene_namespace=BW25113_NAMESPACE,
                strain_label=row.label.label,
                identifier_kind=IDENTIFIER_KIND,
                released_kind=row.label.kind,
                cassette=KANAMYCIN,
                collection=KIND_COLLECTIONS.get(row.label.kind),
            )
            for eck, name in zip(row.label.ecks, row.gene_names, strict=True)
        ]
    )


# --------------------------------------------------------------------------- #
# Reading Table S2
# --------------------------------------------------------------------------- #
class ScoreMatrix(BaseModel):
    """Table S2 as read: the condition labels, the row labels and the score cells."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    conditions: tuple[str, ...]
    strains: tuple[str, ...]
    #: one tuple per strain row, one cell per condition; ``None`` is an empty cell
    scores: tuple[tuple[float | None, ...], ...]


def read_score_matrix(path: str | Path) -> ScoreMatrix:
    """Read Table S2 from its OOXML bytes; its title, header and shape are checked."""
    with open(path, "rb") as handle:
        book = openpyxl.load_workbook(handle, read_only=True, data_only=True)
        if book.sheetnames != [MATRIX_SHEET]:
            raise ValueError(f"{path}: sheets {book.sheetnames}")
        rows: Iterator[tuple[Any, ...]] = book[MATRIX_SHEET].iter_rows(values_only=True)
        title = next(rows)
        if title[0] != MATRIX_TITLE:
            raise ValueError(f"{path}: title cell {title[0]!r}")
        header = next(rows)
        if header[0] != MATRIX_HEADER:
            raise ValueError(f"{path}: header cell {header[0]!r}")
        conditions = tuple(str(cell) for cell in header[1:])
        strains: list[str] = []
        scores: list[tuple[float | None, ...]] = []
        for row in rows:
            cells = row[1 : 1 + len(conditions)]
            if any(
                cell is not None and not isinstance(cell, float | int) for cell in cells
            ):
                raise ValueError(f"{path}: a non-numeric score on row {row[0]!r}")
            strains.append(str(row[0]))
            scores.append(
                tuple(None if cell is None else float(cell) for cell in cells)
            )
        book.close()
    if (len(strains), len(conditions)) != (RELEASED_STRAIN_ROWS, RELEASED_CONDITIONS):
        raise ValueError(
            f"{path}: {len(strains)} x {len(conditions)}, expected "
            f"{RELEASED_STRAIN_ROWS} x {RELEASED_CONDITIONS}"
        )
    return ScoreMatrix(
        conditions=conditions, strains=tuple(strains), scores=tuple(scores)
    )


# --------------------------------------------------------------------------- #
# Cell accounting
# --------------------------------------------------------------------------- #
#: How the release's empty cells are kept: as the release has them, absent.
EMPTY_CELL_RULE: Final = (
    "absent: an empty Table S2 cell is no record, exactly as the release has it; no "
    "typed gap is written for it because there is no measurement to carry one (the "
    "paper does not say why a cell is empty)"
)


class CellAccounting(BaseModel):
    """Every Table S2 cell accounted for: a record, or an empty cell left absent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: str
    released_cells: int
    records: int
    empty_cells: int
    empty_cell_rule: str


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
@register_dataset
class EnvChemgenNichols2011Dataset(ExperimentDataset):
    """Nichols 2011 Table S2: 3,979 E. coli strains' re-scaled scores in 324 screens."""

    organism = Organism.ecoli

    #: The study's wild type, whose assembly every record's reference pins (WILD_TYPE).
    #: No genome is injected or read: strains are stored under their released labels.
    REFERENCE_STRAIN: ClassVar[EcoliK12StrainName] = "BW25113"

    def __init__(
        self,
        root: str = "data/torchcell/ecoli_env_chemgen_nichols2011",
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset."""
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
        """Table S1 and Table S2 of the deposited supplement."""
        return [osp.basename(SCREEN_LIST_REL), osp.basename(MATRIX_REL)]

    def download(self) -> None:
        """Link the two mirror files into ``raw/`` after verifying their pins.

        The mirror plus the pins is canonical; Elsevier's CDN is retrieval metadata the
        #1046 deposit record names, never a live build dependency.
        """
        data_root = _data_root()
        manifest = load_manifest(data_root)
        os.makedirs(self.raw_dir, exist_ok=True)
        for relpath, pin in (
            (SCREEN_LIST_REL, SCREEN_LIST_SHA256),
            (MATRIX_REL, MATRIX_SHA256),
        ):
            check_manifest_pin(relpath, manifest_sha256(manifest, relpath), pin)
            src = raw_mirror_dir(data_root) / relpath
            if not src.exists():
                raise RuntimeError(f"required raw artifact missing from mirror: {src}")
            link_verified(src, osp.join(self.raw_dir, osp.basename(relpath)), pin)
        log.info("Nichols 2011 Tables S1 and S2 linked into %s", self.raw_dir)

    @post_process
    def process(self) -> None:
        """Every non-empty Table S2 cell as one record; write the LMDB."""
        screen_list = osp.join(self.raw_dir, osp.basename(SCREEN_LIST_REL))
        matrix_path = osp.join(self.raw_dir, osp.basename(MATRIX_REL))
        verify_raw_files(
            self.raw_dir,
            {
                osp.basename(SCREEN_LIST_REL): SCREEN_LIST_SHA256,
                osp.basename(MATRIX_REL): MATRIX_SHA256,
            },
        )
        check_screen_list(screen_list)
        matrix = read_score_matrix(matrix_path)
        conditions = match_conditions(matrix.conditions)
        rows = strain_rows(matrix.strains)
        genotypes = [genotype(row) for row in rows]
        genome_reference = assembly_reference(self.REFERENCE_STRAIN)
        environments = [environment(c) for c in conditions]
        references = [
            BacterialEnvironmentResponseExperimentReference(
                dataset_name=self.name,
                genome_reference=genome_reference,
                environment_reference=environments[j],
                phenotype_reference=reference_phenotype(c),
            )
            for j, c in enumerate(conditions)
        ]
        publication = Publication(doi=DOI, doi_url=f"https://doi.org/{DOI}")

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        idx = 0
        empty = 0
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for j, condition in enumerate(tqdm(conditions, desc="nichols2011")):
                for row in rows:
                    value = matrix.scores[row.index][j]
                    if value is None:
                        empty += 1
                        continue
                    experiment = BacterialEnvironmentResponseExperiment(
                        dataset_name=self.name,
                        genotype=genotypes[row.index],
                        environment=environments[j],
                        phenotype=phenotype(value, condition, row.replicate_id),
                    )
                    txn.put(
                        f"{idx}".encode(),
                        self._intern_record(
                            experiment, references[j], publication, itxn
                        ),
                    )
                    idx += 1
        env.close()
        interned_env.close()
        self._write_reports(matrix, rows, records=idx, empty=empty)
        log.info(
            "Nichols2011: wrote %d records over %d rows x %d screens; %d empty cells "
            "hold no record",
            idx,
            len(rows),
            len(conditions),
            empty,
        )

    def _write_reports(
        self,
        matrix: ScoreMatrix,
        rows: Sequence[StrainRow],
        *,
        records: int,
        empty: int,
    ) -> None:
        """Write the cell accounting and the per-row label report; check the arithmetic."""
        cells = RELEASED_STRAIN_ROWS * RELEASED_CONDITIONS
        if records + empty != cells:
            raise RuntimeError(
                f"cell accounting mismatch: {records} records + {empty} empty cells "
                f"!= {cells}"
            )
        accounting = CellAccounting(
            dataset=self.name,
            released_cells=cells,
            records=records,
            empty_cells=empty,
            empty_cell_rule=EMPTY_CELL_RULE,
        )
        with open(osp.join(self.preprocess_dir, "cell_accounting.json"), "w") as handle:
            handle.write(accounting.model_dump_json(indent=2))
        report = {
            "dataset": self.name,
            "released_strain_rows": len(rows),
            "distinct_released_labels": len({r.label.label for r in rows}),
            "rows_by_kind": dict(
                sorted(Counter(str(r.label.kind) for r in rows).items())
            ),
            "distinct_eck_accessions": len({e for r in rows for e in r.label.ecks}),
            "rows_with_replicate_id": sum(r.replicate_id is not None for r in rows),
            "identifier_kind": IDENTIFIER_KIND,
            "genome_lookups": 0,
            "rows": [
                {
                    "index": r.index,
                    "label": r.label.label,
                    "kind": str(r.label.kind),
                    "ecks": list(r.label.ecks),
                    "gene_names": list(r.gene_names),
                    "replicate_id": r.replicate_id,
                }
                for r in rows
            ],
        }
        with open(osp.join(self.preprocess_dir, "strain_labels.json"), "w") as handle:
            json.dump(report, handle, indent=2)

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError(
            "EnvChemgenNichols2011Dataset builds records in process()"
        )


# --------------------------------------------------------------------------- #
# Verification (L0-L4) of a built tree
# --------------------------------------------------------------------------- #
#: Records of the full build: every non-empty Table S2 cell (3,979 x 324 = 1,289,196
#: cells, 2,234 of them empty).
EXPECTED_RECORDS: Final = 1286962


def released_accessions(data_root: str | None = None) -> set[str]:
    """Every ECK accession Table S2's row labels carry, read off the raw mirror.

    The L4 universe of this dataset: a stored identifier must be one the release
    carries. No annotation is consulted, because none decides what is stored.
    """
    matrix = read_score_matrix(raw_mirror_dir(data_root) / MATRIX_REL)
    return {
        eck
        for i, label in enumerate(matrix.strains)
        for eck in parse_strain_label(i, label).ecks
    }


def verify_build(
    dataset_root: str,
    *,
    data_root: str | None = None,
    expected_count: int = EXPECTED_RECORDS,
) -> VerificationReport:
    """Run the environment-response L0-L4 verifier on a built tree and write its report.

    The LMDB is streamed once. L4 containment is against the release's own accessions
    (:func:`released_accessions`); the canonical-name rule runs without a resolver, so
    it checks that each accession carries one released name. The report is written to
    ``preprocess/verification_report.json``.
    """
    from torchcell.verification.environment_response import (
        verify_environment_response_dataset_streaming,
    )
    from torchcell.verification.runners import stream_records

    report = verify_environment_response_dataset_streaming(
        stream_records(dataset_root),
        dataset_name=osp.basename(osp.normpath(dataset_root)),
        provenance=Provenance(
            source_uri=f"$DATA_ROOT/{RAW_DIR_REL}/{MATRIX_REL}",
            citation_key=CITATION_KEY,
            sha256=MATRIX_SHA256,
            method="Table S2 complete data set, one "
            "BacterialEnvironmentResponseExperiment per non-empty (strain row, screen) "
            "cell; z_score, the per-screen re-scaled S-score; strains under their "
            "released ECK labels",
            page="Table S2 - Complete Data Set (mmc2), sheet 'TableS2-FinalData'",
            retrieved="2026-10-12",
        ),
        expected_count=expected_count,
        sgd_genes=released_accessions(data_root),
        gene_universe_label="Table S2 ECK accessions",
    )
    preprocess = osp.join(dataset_root, "preprocess")
    os.makedirs(preprocess, exist_ok=True)
    with open(osp.join(preprocess, "verification_report.json"), "w") as handle:
        handle.write(report.model_dump_json(indent=2))
    return report


def main() -> None:
    """Build the dataset under ``DATA_ROOT`` and verify it, for interactive debugging."""
    from dotenv import load_dotenv

    load_dotenv()
    data_root = _data_root()
    root = osp.join(data_root, "data/torchcell/ecoli_env_chemgen_nichols2011")
    dataset = EnvChemgenNichols2011Dataset(root=root)
    print(f"len = {len(dataset)}")
    print(dataset[0])
    print(verify_build(root, data_root=data_root).summary())


if __name__ == "__main__":
    main()
