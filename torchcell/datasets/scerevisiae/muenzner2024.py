# torchcell/datasets/scerevisiae/muenzner2024
# [[torchcell.datasets.scerevisiae.muenzner2024]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/muenzner2024
# Test file: tests/torchcell/datasets/scerevisiae/test_muenzner2024.py
"""Muenzner 2024 natural-isolate proteomes and protein turnover (Nature 630:149).

Muenzner et al. 2024 (doi:10.1038/s41586-024-07442-9, PMID 38778096) measured SWATH-MS
(DIA-NN) proteomes of 796 natural *S. cerevisiae* isolates of the 1,011-genomes panel
(Peter 2018), grown in synthetic minimal (SM) medium, and dynamic-SILAC protein turnover
for 55 of them. Two dataset classes, one per released measurement:

- ``ProteomeMuenzner2024Dataset``: one ``ProteinAbundanceExperiment`` per isolate, from
  Supplementary Table 3 ("Proteomes of 796 yeast isolates (non-imputed)"). The value is
  the released linear DIA-NN MaxLFQ quantity after precursor-level batch-median
  correction; a protein the release leaves blank is key-absent, never imputed (the paper
  imputed by KNN for its own analyses, the release ships the matrix before that step).
- ``ProteinTurnoverMuenzner2024Dataset``: one ``ProteinTurnoverExperiment`` per
  isolate, from Supplementary Table 17: ``kdp`` (turnover rate, 1/h) as
  ``degradation_rate`` and ``HL`` (h) as ``half_life``. The paper computed both "without
  doubling time or dilution rate correction", so the rate is total replacement
  (degradation plus dilution by growth), named in ``measurement_type``.
- ``ProteomeTimecourseMuenzner2024Dataset`` (2026.11.24, #1176): one
  ``ProteinAbundanceExperiment`` per (isolate, biological replicate, time point) of the
  24 h batch time course of Supplementary Table 13 ("Proteomes of two natural isolates
  (AMC, BBV) across different growth stages"): 2 isolates x 3 replicate cultures x 5
  time points = 30 records. The peer review file states "The experiment was performed
  in biological triplicates", so each released ``replicate`` is one culture, sampled at
  five times; ``replicate_id`` is the release's replicate number and
  ``Environment.duration_hours`` the released ``time (h)`` since the dilution into the
  30 ml culture. A different instrument and search (Q Exactive Plus DIA, DIA-NN 1.8,
  library-free against UniProt), so its own ``measurement_type``, and keyed by the
  release's ``Genes`` column verbatim (mostly standard names), never re-mapped. The
  reference is the paper's own: per time point, the median over the replicates of the
  euploid isolate (AMC) for each protein measured in at least two of its three.

GENOTYPE. Each isolate is a perturbation set off S288C built by
``torchcell.sequence.natural_isolate`` (``NaturalIsolateGenotypes``), the module every
natural-isolate loader on the Peter panel shares, so an isolate here is the same
``Genotype`` as in the Caudal 2024, Peter 2018 and De Chiara 2022 stores: sequence
variants from Peter's reference-gene tarball, natural presence of non-reference pangenome
ORFs and natural absence of reference ORFs from Peter's matrices, the HO deletion Peter's
Table S1 marks (#988), the natural loss-of-function calls of Peter's two matrices (#987),
and ``Genotype.natural_isolate``, which carries the ploidy, zygosity and whole-chromosome
aneuploidy of Peter's Table S1 (#1001). The release spells the 93 ``SACE_`` codes of the
Peter panel with a dot (``SACE.YAB``); ``peter_code`` maps them back, and every one of the
67 such proteome isolates carries the same alternative strain name in SI Table 1 as in
Peter's Table S1 (checked in the dendron note). One isolate with no Peter genome is still
refused and counted in the ledger: ``XTRA.DGU`` (KS11, SI Table 1 ploidy 3). Stored the
way FY4 is, it would need its own reference genome, and ``ReferenceGenome.ploidy`` admits
only haploid and diploid, so a triploid has no honest carrier without a schema change (an
owner decision, #1045). The ledger also keeps each isolate's SI Table 1 karyotype with the
paper's SI Table 5 / 6 / 7 flags.

FY4 (2026.10.12, #1045 row 18). The release's ``Reference`` column (SI Table 1: FY4,
ploidy 1) is the S288C-derived laboratory strain, resolved the way the other S288C-derived
backgrounds are (BY4741/BY4742 in the deletion loaders): its record's reference names the
strain (``ReferenceGenome(strain="FY4")``) on the S288C R64 coordinate system, and its
genotype carries no perturbation off S288C and no ``natural_isolate`` descriptor (FY4 is
not an isolate of the Peter panel). The release names the column ``Reference`` and its
Methods infer every strain "on the basis of the reference genome of the S288c strain" and
take "the proteome for the reference strain S288c" from UniProt. That FY4 descends from
S288C (Winston 1995) is not mirrored; the ledger says so.

REFERENCE. The paper's own: "the median mRNA or protein expression value of the respective
gene across all euploid strains" (Methods). The proteome reference is the per-protein
median over the 518 euploid isolates of the paper's integrated set (SI Table 7), on the
released non-imputed values. The turnover reference is the per-protein median ``kdp`` over
the euploid isolates of the turnover set, the "euploid chromosomes of euploid isolates"
baseline of the turnover Methods (the paper quantile-normalizes first; the reference here
is on the released rates). A protein measured in no euploid isolate has no reference key.

Source: the Supplementary Tables workbook (MOESM3), deposited in the raw mirror
``$DATA_ROOT/torchcell-raw/muenznerNaturalProteomeDiversity2024/`` with a per-file
provenance record; Peter's three files come from the genomes tier.
"""

from __future__ import annotations

import argparse
import logging
import os
import os.path as osp
import pickle
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import lmdb
import numpy as np
import openpyxl
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    Organism,
    check_manifest_pin,
    copy_verified,
    link_verified,
    post_process,
    verify_raw_files,
)
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import SM
from torchcell.datamodels.schema import (
    ChromosomeCopyChange,
    Concentration,
    ConcentrationUnit,
    DeletionPerturbation,
    Environment,
    Experiment,
    ExperimentReference,
    Genotype,
    Media,
    MediaComponent,
    MediaComponentRole,
    NaturalGeneAbsencePerturbation,
    NaturalGenePresencePerturbation,
    NaturalIsolateDescriptor,
    NaturalLossOfFunctionPerturbation,
    ProteinAbundanceExperiment,
    ProteinAbundanceExperimentReference,
    ProteinAbundancePhenotype,
    ProteinTurnoverExperiment,
    ProteinTurnoverExperimentReference,
    ProteinTurnoverPhenotype,
    Publication,
    ReferenceGenome,
    SequenceVariantPerturbation,
    Temperature,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.literature.manifest import (
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.sequence.genome.registry import (
    PETER2018_1011,
    SGD_S288C_R64,
    SGD_S288C_R64_GENBANK_ACCESSION,
    load_genome_manifest,
)
from torchcell.sequence.natural_isolate import (
    GENOTYPE_FILES,
    PRESENCE_NAME,
    NaturalIsolateGenotypes,
    genotype_isolate,
    link_genotype_files,
    panel_isolates,
    parse_aneuploidy,
)
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

CITATION_KEY = "muenznerNaturalProteomeDiversity2024"
DOI = "10.1038/s41586-024-07442-9"
PUBMED_ID = "38778096"
TITLE = "Natural proteome diversity links aneuploidy tolerance to protein turnover"

# The library mirror's OCR of the paper and of the SI guide + reporting summary.
PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "1534d1e3d1218b3a5e34c3b57c22cf038919cfa318e6049bd67401587aa86cef"
REPORTING_SUMMARY_MD = "si/si2.md"
REPORTING_SUMMARY_MD_SHA256 = (
    "4b2c31e1595f606bde9052f549129952ba03933d521b0042ee48b24c9a357e06"
)
#: The peer review file (MOESM4): the only text that states the time course's replicate
#: kind and count.
PEER_REVIEW_MD = "si/si4.md"
PEER_REVIEW_MD_SHA256 = (
    "78ef93680a232b07a2dafdcf13931e36b7c1d03b8468bc8bead514cf9ad967db"
)

# The one raw file both loaders consume: the Supplementary Tables workbook (MOESM3),
# captured from the PMC Article Datasets bucket (PMC11153158) and deposited in the raw
# mirror. Canonical = these bytes + sha256, not the URL.
PMC_PREFIX = "PMC11153158.1"
SI_TABLES_FILENAME = "41586_2024_7442_MOESM3_ESM.xlsx"
SI_TABLES_REL = f"si/{SI_TABLES_FILENAME}"
SI_TABLES_SHA256 = "1e78de77ffeb5ffb4f64f3999765db62ec9f6b6729a71f75feb1b955f5d6e005"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"
#: When the library capture fetched the workbook (its manifest's ``retrieved_at``).
RAW_RETRIEVED_AT = "2026-10-11T10:43:58.262073+00:00"
PMC_RETRIEVER = "torchcell.literature.retrieve.pmc_cloud_object"
ROLE_RAW_DATA = "raw_data"
MASSIVE_ACCESSION = "MSV000090435"
PRIDE_DISOMES = "PXD044526"
PRIDE_TURNOVER = "PXD048219"

#: The SI workbook plus the six Peter-tier files an isolate genotype is built from.
RAW_FILES = [SI_TABLES_FILENAME, *GENOTYPE_FILES]
#: What the isolate genotypes of this build carry, counted (``natural_isolate``).
GENOTYPE_LEDGER = "natural_isolate_genotype_ledger.json"
#: The classes an isolate genotype holds, built in ``torchcell.sequence.natural_isolate``.
#: Named here because the schema-impact closure of this loader is read from this file's
#: own imports: a change to any of them must re-stale this store.
GENOTYPE_CLASSES = (
    SequenceVariantPerturbation,
    NaturalGenePresencePerturbation,
    NaturalGeneAbsencePerturbation,
    DeletionPerturbation,
    NaturalLossOfFunctionPerturbation,
    NaturalIsolateDescriptor,
)

PROTEOME_MEASUREMENT_TYPE = "swath_ms_maxlfq_precursor_batch_corrected_quantity"
TURNOVER_MEASUREMENT_TYPE = "dynamic_silac_lys8_kdp_per_hour_no_dilution_correction"
#: SI Table 13: Q Exactive Plus DIA, DIA-NN 1.8 library-free against UniProt, keyed by
#: the release's ``Genes`` column. Not the SWATH quantity of SI Table 3 (another
#: instrument, search and normalization), so never pooled with it.
TIMECOURSE_MEASUREMENT_TYPE = "dia_ms_dia_nn_1_8_library_free_gene_quantity"

#: The release's spelling of the S288C-derived control column (SI Table 1: FY4).
FY4_RELEASE_CODE = "Reference"
#: SI Table 1's alternative strain name of that column, checked at build.
FY4_STRAIN = "FY4"
#: Release codes of S288C-derived laboratory strains -> their SI Table 1 name.
S288C_DERIVED: dict[str, str] = {FY4_RELEASE_CODE: FY4_STRAIN}
EUPLOID = "euploid"


def pmc_cloud_url(filename: str) -> str:
    """The PMC Article Datasets bucket URL of one supplementary file."""
    return f"https://pmc-oa-opendata.s3.amazonaws.com/{PMC_PREFIX}/{filename}"


def pmc_cloud_key(filename: str) -> str:
    """The bucket key the ``pmc_cloud_object`` retriever is called with."""
    return f"{PMC_PREFIX}/{filename}"


# --------------------------------------------------------------------------- #
# Sourced statements (verbatim substrings of the mirrored OCR, audited by the tests)
# --------------------------------------------------------------------------- #
def _paper_sv(value: object, quote: str, note: str | None = None) -> SourcedValue:
    """A SourcedValue pinned to the mirrored ``paper.md``."""
    return SourcedValue(
        value=value,
        provenance=Provenance(
            source_uri=PAPER_MD, citation_key=CITATION_KEY, sha256=PAPER_MD_SHA256
        ),
        quote=quote,
        note=note,
    )


SM_RECIPE_QUOTE = (
    "the collection was grown on agar plates containing synthetic minimal (SM) medium "
    "( $\\cdot 6 . 8 \\mathsf { g l } ^ { - 1 }$ yeast nitrogen base, $2 \\%$ glucose, "
    "without amino acids). Subsequently, colonies were inoculated in SM liquid medium "
    "$( 2 0 0 \\mu \\mathrm { l } )$ and incubated at $3 0 ^ { \\circ } \\mathrm { C }$ "
    "overnight."
)
SM_GROWTH_QUOTE = (
    "Plates were sealed using an oxygen-permeable membrane and grown at "
    "$3 0 ^ { \\circ } \\mathrm { C }$ to exponential phase (Supplementary Table\u00a02), "
    "shaking at 1,000\u2009rpm for $^ { 8 \\mathrm { h } }$ ."
)
YNB_PRODUCT_QUOTE = "yeast nitrogen base without amino acids (Y0626)"
SILAC_MEDIUM_QUOTE = (
    "were grown on synthetic medium containing $6 . 7 \\mathrm { g } 1 ^ { - 1 }$ yeast "
    "nitrogen base, $2 \\%$ glucose and $8 0 \\mathrm { m g l ^ { - 1 } L }$ -lysine"
)
SILAC_HEAVY_QUOTE = (
    "For SILAC labelling, l-lysine was swapped for $8 0 \\mathrm { m g l ^ { - 1 } }$ "
    "heavy $[ ^ { 1 3 } { \\bf C } _ { 6 } / ^ { 1 5 } { \\bf N } _ { 2 } ]$ lysine "
    "$\\mathbf { \\boldsymbol { S } M } +$ Lys-8)."
)
SILAC_GROWTH_QUOTE = (
    "The heavy-labelled cultures were grown at $3 0 ^ { \\circ } \\mathrm { C } , 3 0 0$ "
    "rpm and at three time points (90\u2009min, 135\u2009min and $1 8 0 \\mathrm { m i n }$ )"
)
NO_DILUTION_CORRECTION_QUOTE = (
    "we therefore calculated turnover rates (kdp) and the corresponding half-lives "
    "without doubling time or\u00a0dilution rate correction (Supplementary Table\u00a016)."
)
HALF_LIFE_QUOTE = (
    "Half-lives were calculated from the resulting kdp as log(2)/kdp "
    "(Supplementary Table\u00a017)."
)
EUPLOID_REFERENCE_QUOTE = (
    "Relative abundances for transcriptomic and proteomic data were calculated "
    "gene-wise as the $\\mathbf { l o g } _ { 2 }$ ratio between a gene’s mRNA or "
    "protein abundance in a given strain and the median mRNA or protein expression "
    "value of the respective gene across all euploid strains (‘all-euploid strain’ "
    "method)."
)
TURNOVER_REFERENCE_QUOTE = (
    "We calculated the median quantile-normalized turnover rate for each protein when "
    "expressed from aneuploid chromosomes, from euploid chromosomes of aneuploid "
    "isolates or from euploid chromosomes of euploid isolates."
)
IMPUTATION_QUOTE = (
    "Missing values (less than $4 \\%$ of all values) were imputed using $k$ -nearest "
    "neighbours (KNN) imputation65."
)
BATCH_CORRECTION_QUOTE = (
    "Subsequently, batch correction was performed at the precursor level by bringing "
    "median precursor quantities of each batch to the same value."
)
S288C_GENOME_QUOTE = (
    "Protein-wise fasta files were created by inferring single-nucleotide polymorphisms "
    "for each strain on the basis of the reference genome of the S288c strain."
)
REFERENCE_STRAIN_QUOTE = (
    "The proteome for the reference strain S288c was obtained from UniProt"
)
REPLICATION_QUOTE = (
    "The proteomic dataset for the small strain library (disomic lab-engineered "
    "strains) was acquired in triplicates, while the large dataset (> 1000 isolates) "
    "was acquired once."
)
TIMECOURSE_MEDIUM_QUOTE = (
    "Yeast isolates were cultivated on SM medium (as above) in batch culture."
)
TIMECOURSE_CULTURE_QUOTE = (
    "of 0.1 in 30 ml medium, and incubated for $2 4 \\mathrm { h }$ at "
    "$3 0 ^ { \\circ } \\mathrm { C }$ , 750 rpm."
)
TIMECOURSE_SAMPLING_QUOTE = (
    "at five time points to cover early exponential, mid-exponential, late exponential "
    "and stationary phases of growth."
)
TIMECOURSE_REPLICATES_QUOTE = "The experiment was performed in biological triplicates."
TIMECOURSE_STRAINS_QUOTE = (
    "we performed an experiment to compare attenuation during the the growth of a "
    "diploid aneuploid strain, BBV, and a diploid euploid strain, AMC"
)
TIMECOURSE_REFERENCE_QUOTE = (
    "-transformed protein expression levels between the aneuploid and the euploid "
    "isolate were calculated per time point for each protein present in at least two "
    "of the three biological replicates of the euploid strain in the given time point"
)
TIMECOURSE_PROCESSING_QUOTE = (
    "Raw data were processed using DIA-NN v.1.8 (ref. 42) with the scan window size "
    "set to 7"
)
TIMECOURSE_LIBRARY_QUOTE = (
    "A spectral library-free approach and yeast UniProt (UP000002311, reviewed, "
    "canonical, downloaded 18 November 2021)63 were used for annotation."
)


def _peer_review_sv(value: object, quote: str, note: str | None = None) -> SourcedValue:
    """A SourcedValue pinned to the mirrored peer review file (``si/si4.md``)."""
    return SourcedValue(
        value=value,
        provenance=Provenance(
            source_uri=PEER_REVIEW_MD,
            citation_key=CITATION_KEY,
            sha256=PEER_REVIEW_MD_SHA256,
        ),
        quote=quote,
        note=note,
    )


SOURCED_VALUES: dict[str, SourcedValue] = {
    "proteome_medium": _paper_sv(
        "SM: 6.8 g/L yeast nitrogen base without amino acids, 2% glucose",
        SM_RECIPE_QUOTE,
        note="the proteome culture medium; the same Methods print 6.7 g/L for the "
        "SILAC and ubiquitinome SM, recorded as read and not corrected",
    ),
    "proteome_growth": _paper_sv(
        "30 C, 8 h in deep-well plates to exponential phase", SM_GROWTH_QUOTE
    ),
    "ynb_product": _paper_sv("Sigma Y0626", YNB_PRODUCT_QUOTE),
    "silac_medium": _paper_sv(
        "SM + 80 mg/L L-lysine (6.7 g/L YNB, 2% glucose)", SILAC_MEDIUM_QUOTE
    ),
    "silac_heavy": _paper_sv("80 mg/L [13C6/15N2] L-lysine", SILAC_HEAVY_QUOTE),
    "silac_growth": _paper_sv("30 C, sampled at 90, 135, 180 min", SILAC_GROWTH_QUOTE),
    "no_dilution_correction": _paper_sv(
        "kdp is not corrected for dilution by growth", NO_DILUTION_CORRECTION_QUOTE
    ),
    "half_life": _paper_sv("HL = log(2)/kdp", HALF_LIFE_QUOTE),
    "euploid_reference": _paper_sv(
        "per-protein median over euploid strains", EUPLOID_REFERENCE_QUOTE
    ),
    "turnover_reference": _paper_sv(
        "per-protein median over euploid isolates", TURNOVER_REFERENCE_QUOTE
    ),
    "s288c_genome": _paper_sv(
        "every strain's genome is written against the S288c reference",
        S288C_GENOME_QUOTE,
    ),
    "reference_strain": _paper_sv(
        "the release's 'Reference' column (SI Table 1: FY4) is the S288C-derived "
        "laboratory reference",
        REFERENCE_STRAIN_QUOTE,
        note="SI Table 1 names the column 'Reference' with alternative_strain_name "
        "'FY4' (checked at build); FY4's descent from S288C (Winston 1995) is not "
        "mirrored",
    ),
    "imputation": _paper_sv(
        "the paper imputed (<4%) by KNN; SI Table 3 is the non-imputed matrix",
        IMPUTATION_QUOTE,
    ),
    "batch_correction": _paper_sv(
        "precursor-level batch-median correction", BATCH_CORRECTION_QUOTE
    ),
    "replication": SourcedValue(
        value=1,
        provenance=Provenance(
            source_uri=REPORTING_SUMMARY_MD,
            citation_key=CITATION_KEY,
            sha256=REPORTING_SUMMARY_MD_SHA256,
        ),
        quote=REPLICATION_QUOTE,
        note="one acquisition per natural isolate, so n_replicates = 1 per protein",
    ),
    "timecourse_medium": _paper_sv(
        "SM as for the proteome cultures, batch culture", TIMECOURSE_MEDIUM_QUOTE
    ),
    "timecourse_culture": _paper_sv(
        "30 ml culture from OD600 0.1, 24 h at 30 C, 750 rpm; time (h) counts from "
        "this dilution",
        TIMECOURSE_CULTURE_QUOTE,
    ),
    "timecourse_sampling": _paper_sv(
        "five sampling times per culture, early exponential to stationary",
        TIMECOURSE_SAMPLING_QUOTE,
        note="the release names them t1..t5 with their own 'time (h)'; which phase "
        "each one is, is not stated per time point, so no growth phase is stored",
    ),
    "timecourse_replicates": _peer_review_sv(
        "3 biological replicate cultures per isolate",
        TIMECOURSE_REPLICATES_QUOTE,
        note="the authors' reply to Reviewer 2 point 8, which introduced the AMC/BBV "
        "time course; SI Table 13's 'replicate' column numbers the three cultures, and "
        "their OD600 differs at every time point",
    ),
    "timecourse_strains": _peer_review_sv(
        "BBV is the aneuploid isolate, AMC the euploid one", TIMECOURSE_STRAINS_QUOTE
    ),
    "timecourse_reference": _paper_sv(
        "per time point, the median over the euploid isolate's replicates of each "
        "protein measured in at least 2 of its 3",
        TIMECOURSE_REFERENCE_QUOTE,
    ),
    "timecourse_processing": _paper_sv("DIA-NN 1.8", TIMECOURSE_PROCESSING_QUOTE),
    "timecourse_library": _paper_sv(
        "library-free search annotated with UniProt UP000002311",
        TIMECOURSE_LIBRARY_QUOTE,
    ),
}


# --------------------------------------------------------------------------- #
# Media
# --------------------------------------------------------------------------- #
def _sm_component(index: int) -> MediaComponent:
    """One component of the library ``SM`` (YNB line, ammonium sulfate, glucose)."""
    return SM.components[index]


MUENZNER_SM: Media = Media(
    name="SM (synthetic minimal: 6.8 g/L YNB without amino acids + 2% glucose), liquid",
    state="liquid",
    is_synthetic=True,
    base_medium="SM",
    components=[
        _sm_component(0).model_copy(
            update={
                "concentration": Concentration(
                    value=6.8, unit=ConcentrationUnit.g_per_l
                ),
                "provenance": [
                    SOURCED_VALUES["proteome_medium"],
                    SOURCED_VALUES["ynb_product"],
                ],
                "note": "Sigma Y0626, the product of the library SM line; this paper "
                "prints 6.8 g/L for the proteome cultures and 6.7 g/L for its SILAC "
                "and ubiquitinome SM",
            }
        ),
        _sm_component(1),
        _sm_component(2).model_copy(
            update={"provenance": [SOURCED_VALUES["proteome_medium"]]}
        ),
    ],
    provenance=[SOURCED_VALUES["proteome_medium"], SOURCED_VALUES["proteome_growth"]],
)
"""The proteome culture medium: the library SM product line at the 6.8 g/L Muenzner
prints, plus the same ammonium sulfate identity and 2% glucose."""

HEAVY_LYSINE = "L-lysine-13C6,15N2"

MUENZNER_SM_LYS8: Media = Media(
    name="SM + 80 mg/L [13C6/15N2] L-lysine (SILAC Lys-8), liquid",
    state="liquid",
    is_synthetic=True,
    base_medium="SM",
    components=[
        *SM.components,
        MediaComponent(
            compound=resolved_compound(HEAVY_LYSINE),
            role=MediaComponentRole.other,
            concentration=Concentration(value=80.0, unit=ConcentrationUnit.ug_per_ml),
            provenance=[SOURCED_VALUES["silac_medium"], SOURCED_VALUES["silac_heavy"]],
            note="the SILAC label: the pre-culture carried 80 mg/L unlabelled "
            "L-lysine (Lys-0) and the cultures were switched to the heavy isotopologue "
            "before the three sampling time points; the isotopologue has no row in the "
            "compound identity table, so its InChIKey is a typed gap",
        ),
    ],
    provenance=[
        SOURCED_VALUES["silac_medium"],
        SOURCED_VALUES["silac_heavy"],
        SOURCED_VALUES["silac_growth"],
    ],
)
"""The dynamic-SILAC medium after the switch: library SM (6.7 g/L, the amount this
section prints) plus 80 mg/L heavy lysine."""


# --------------------------------------------------------------------------- #
# The Supplementary Tables workbook
# --------------------------------------------------------------------------- #
class SheetLayoutError(ValueError):
    """A worksheet does not carry the caption or header the loader was written for."""


class SheetSpec(BaseModel):
    """What one consumed worksheet must look like, read off the deposited workbook.

    ``caption`` holds the exact text of the leading caption rows (column A, top down);
    ``header`` the exact header cells (trailing blanks trimmed). A wide sheet whose
    header continues with data columns gives only ``first_header``.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    caption: tuple[str, ...]
    header_row: int
    header: tuple[str | None, ...] | None = None
    first_header: str | None = None


T1 = SheetSpec(
    name="SI Table 1",
    caption=(
        "SI Table 1:  Strain annotations of natural isolates.",
        'Aneuploid strains are classified as: "aneu;+/-X*Y", where +/- indicates '
        "chromosome gain or loss, X the number of aberrant copies and Y the affected "
        "chromosome.",
    ),
    header_row=4,
    header=(
        None,
        "Standardized_name",
        "alternative_strain_name",
        "Ploidy",
        "Aneuploidies",
        "Clades",
        "Clade_class",
        "Single_strain_class",
        "Simple_Ecology",
        "Isolation",
        "Geographical.origins",
    ),
)
T3 = SheetSpec(
    name="SI Table 3",
    caption=("SI Table 3: Proteomes of 796 yeast isolates (non-imputed).",),
    header_row=3,
    first_header="Protein",
)
T5 = SheetSpec(
    name="SI Table 5",
    caption=(
        "SI Table 5: List of strains showing a discrepancy between aneuploidy "
        "annotation as shown in SI Table 1 and the median gene copy number per "
        "chromosome as published in 10.1038/s41586-018-0030-5.",
    ),
    header_row=3,
    header=("strain",),
)
T6 = SheetSpec(
    name="SI Table 6",
    caption=(
        "SI Table 6: List of strains showing a discrepancy between their median "
        "normalized chromosomal log2 mRNA ratios and the predicted median based on "
        "the aneuploidy annotation.",
        "Strains containing at least one chromosome with a mismatch between aneuploidy "
        "annotation and the relative mRNA expression (as described in Materials and "
        "Methods). Strains that also are listed in SI Table 3 are excluded from this "
        "table.",
    ),
    header_row=4,
    header=("strain",),
)
T7 = SheetSpec(
    name="SI Table 7",
    caption=(
        "SI Table 7: Strains included in the integrated dataset after exclusion of "
        "strains with mismatch between either median chromosomal gene copy number and "
        "aneuploidy annotation, or between median log2 RNA expression levels (log2 "
        "strain/euploid) and aneuploidy annotation.",
    ),
    header_row=4,
    header=("Standardized_name", "Ploidy", "Aneuploidies", "complex_aneuploidy"),
)
T13 = SheetSpec(
    name="SI Table 13",
    # The release's caption ends in a space; it is matched as released.
    caption=(
        "SI Table 13: Proteomes of two natural isolates (AMC, BBV) across different "
        "growth stages. ",
    ),
    header_row=4,
    header=(
        "strain",
        "replicate",
        "timepoint",
        "time (h)",
        "OD600",
        "Genes",
        "abundance",
    ),
)
T16 = SheetSpec(
    name="SI Table 16",
    caption=(
        "SI Table 16: Doubling times of natural isolates as measured during turnover "
        "experiments.",
        "Doubling times were calculated by linear regression of log2 OD600 ratios "
        "(OD600(t1)/OD600(t0)) vs. time. P value and adjusted R squared show "
        "goodness-of-fit of the regressions.",
    ),
    header_row=3,
    header=("strain", "doubling time (min)", "p.value", "adj.r.squared"),
)
T17 = SheetSpec(
    name="SI Table 17",
    caption=(
        "SI Table 17: Protein turnover data for 55 natural euploid and aneuploid "
        "isolates.",
        "kdp denotes the turnover rate in (1/h), HL the protein half-life (in h), and "
        "Intensity is an abundance estimate generated from summing intensities over "
        "heavy and light SILAC channels.",
    ),
    header_row=4,
    header=(
        "strain",
        "Ploidy",
        "Aneuploidies",
        "Leading razor protein",
        "systematic_name",
        "gene_name",
        "kdp_value",
        "Intensity",
        "HL",
    ),
)


def _trim(row: tuple[Any, ...]) -> tuple[Any, ...]:
    """Drop trailing ``None`` cells."""
    end = len(row)
    while end and row[end - 1] is None:
        end -= 1
    return row[:end]


def sheet_frame(rows: list[tuple[Any, ...]], spec: SheetSpec) -> pd.DataFrame:
    """Check one worksheet's caption and header and return its data rows as a frame.

    ``rows`` are the sheet's cell values top down (``openpyxl`` ``values_only``). The
    caption rows must equal ``spec.caption`` exactly and the header must equal
    ``spec.header`` (or start with ``spec.first_header``); a data cell right of the
    header raises, and so does an empty row followed by more data. Trailing empty rows
    are dropped. A ``None`` header cell (SI Table 1's row-number column) is named
    ``row_number``.
    """
    for i, text in enumerate(spec.caption):
        cell = rows[i][0] if i < len(rows) and rows[i] else None
        if cell != text:
            raise SheetLayoutError(
                f"{spec.name} row {i + 1} column A is {cell!r}, expected {text!r}"
            )
    header = _trim(rows[spec.header_row])
    if spec.header is not None and header != spec.header:
        raise SheetLayoutError(
            f"{spec.name} header is {header!r}, expected {spec.header!r}"
        )
    if spec.first_header is not None and (not header or header[0] != spec.first_header):
        raise SheetLayoutError(
            f"{spec.name} header starts {header[:1]!r}, expected {spec.first_header!r}"
        )
    width = len(header)
    data = [_trim(r) for r in rows[spec.header_row + 1 :]]
    while data and not data[-1]:
        data.pop()
    for k, row in enumerate(data):
        if not row:
            raise SheetLayoutError(
                f"{spec.name}: empty row {spec.header_row + 2 + k} inside the data"
            )
        if len(row) > width:
            raise SheetLayoutError(
                f"{spec.name}: row {spec.header_row + 2 + k} has cells right of the "
                f"header: {row[width:]!r}"
            )
    columns = ["row_number" if h is None else str(h) for h in header]
    return pd.DataFrame(
        [list(r) + [None] * (width - len(r)) for r in data], columns=columns
    )


def read_sheets(path: str, specs: list[SheetSpec]) -> dict[str, pd.DataFrame]:
    """Read and check the named worksheets of the deposited workbook."""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        out = {}
        for spec in specs:
            rows = [tuple(r) for r in wb[spec.name].iter_rows(values_only=True)]
            out[spec.name] = sheet_frame(rows, spec)
    finally:
        wb.close()
    return out


# --------------------------------------------------------------------------- #
# Isolates: matching to the Peter panel, karyotype annotations, refusals
# --------------------------------------------------------------------------- #
def peter_code(release_code: str) -> str:
    """The Peter 2018 spelling of a release isolate code (``SACE.YAB`` -> ``SACE_YAB``).

    The release writes the ``SACE_`` and ``XTRA_`` prefixes with a dot; three-letter
    codes are unchanged.
    """
    return release_code.replace(".", "_")


class IsolateKaryotype(BaseModel):
    """The released karyotype of one built isolate: what the genotype cannot carry."""

    isolate: str
    release_code: str
    ploidy: str = Field(description="SI Table 1 'Ploidy', as released")
    aneuploidies: str = Field(description="SI Table 1 'Aneuploidies', as released")
    copy_changes: list[ChromosomeCopyChange]
    integrated: bool = Field(description="listed in SI Table 7 (the integrated set)")
    excluded_by: str | None = Field(
        default=None,
        description="'SI Table 5' or 'SI Table 6' when the paper excluded the isolate "
        "for a karyotype-annotation mismatch",
    )


class RefusedIsolate(BaseModel):
    """A release isolate no record is built for, with the measured reason."""

    release_code: str
    alternative_strain_name: str | None
    reason: str


class S288cDerivedIsolate(BaseModel):
    """A release column that is an S288C-derived laboratory strain, built on S288C."""

    release_code: str
    strain: str = Field(description="SI Table 1 alternative_strain_name, as released")
    ploidy: str = Field(description="SI Table 1 'Ploidy', as released")
    aneuploidies: str = Field(description="SI Table 1 'Aneuploidies', as released")
    basis: str = Field(description="How the strain is resolved, and what is unsourced")


def s288c_derived_isolate(code: str, annotations: pd.DataFrame) -> S288cDerivedIsolate:
    """The SI Table 1 row of an S288C-derived column, which must name its strain."""
    row = annotations.set_index("Standardized_name").loc[code]
    strain = str(row["alternative_strain_name"])
    if strain != S288C_DERIVED[code]:
        raise ValueError(
            f"SI Table 1 names {code!r} {strain!r}, not {S288C_DERIVED[code]!r}"
        )
    if str(row["Ploidy"]) != "1":
        raise ValueError(f"SI Table 1 gives {strain} ploidy {row['Ploidy']}, not 1")
    return S288cDerivedIsolate(
        release_code=code,
        strain=strain,
        ploidy=str(row["Ploidy"]),
        aneuploidies=str(row["Aneuploidies"]),
        basis=f"the release's {code!r} column, {strain}: an S288C-derived laboratory "
        "strain stored with no perturbation off S288C and its own reference genome "
        f"ReferenceGenome(strain={strain!r}, haploid); its descent from S288C "
        "(Winston 1995) is not mirrored",
    )


def match_isolates(
    release_codes: list[str], peter_isolates: set[str], annotations: pd.DataFrame
) -> tuple[dict[str, str], list[S288cDerivedIsolate], list[RefusedIsolate]]:
    """``{release code: Peter code}``, the S288C-derived columns, and the refusals.

    ``annotations`` is SI Table 1; every release code must be one of its rows. An
    S288C-derived column (``S288C_DERIVED``) is built on S288C, not refused (#1045).
    """
    names = annotations.set_index("Standardized_name")["alternative_strain_name"]
    missing = sorted(set(release_codes) - set(names.index))
    if missing:
        raise ValueError(f"release isolates absent from SI Table 1: {missing}")
    matched: dict[str, str] = {}
    s288c: list[S288cDerivedIsolate] = []
    refused: list[RefusedIsolate] = []
    for code in release_codes:
        if code in S288C_DERIVED:
            s288c.append(s288c_derived_isolate(code, annotations))
            continue
        peter = peter_code(code)
        if peter in peter_isolates:
            matched[code] = peter
            continue
        alt = names[code]
        refused.append(
            RefusedIsolate(
                release_code=code,
                alternative_strain_name=None if pd.isna(alt) else str(alt),
                reason=(
                    f"no Peter 2018 genome: {peter!r} is not an isolate of "
                    f"{PRESENCE_NAME}, so no genotype can be written"
                ),
            )
        )
    if len(set(matched.values())) != len(matched):
        raise ValueError("two release codes map to one Peter isolate")
    return matched, s288c, refused


def s288c_derived_reference(
    reference: ExperimentReference, isolate: S288cDerivedIsolate
) -> ExperimentReference:
    """The shared reference with the strain's own genome: the S288C-derived background
    on the S288C R64 coordinate system, as the deletion loaders state BY4741.
    """
    return reference.model_copy(
        update={
            "genome_reference": ReferenceGenome(
                species="Saccharomyces cerevisiae",
                strain=isolate.strain,
                ploidy="haploid",
                assembly_set=SGD_S288C_R64,
                assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
            )
        }
    )


def karyotypes(
    matched: dict[str, str],
    annotations: pd.DataFrame,
    integrated: set[str],
    table5: set[str],
    table6: set[str],
) -> list[IsolateKaryotype]:
    """The SI Table 1 karyotype of each matched isolate, with the paper's flags."""
    rows = annotations.set_index("Standardized_name")
    out = []
    for code, peter in sorted(matched.items(), key=lambda kv: kv[1]):
        aneu = str(rows.loc[code, "Aneuploidies"])
        excluded = "SI Table 5" if code in table5 else None
        if code in table6:
            excluded = "SI Table 6"
        out.append(
            IsolateKaryotype(
                isolate=peter,
                release_code=code,
                ploidy=str(rows.loc[code, "Ploidy"]),
                aneuploidies=aneu,
                copy_changes=parse_aneuploidy(aneu),
                integrated=code in integrated,
                excluded_by=excluded,
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Ledgers (written to preprocess/)
# --------------------------------------------------------------------------- #
class ProteomeLedger(BaseModel):
    """Every isolate column and cell of SI Table 3, accounted for."""

    sheet: str = T3.name
    n_release_isolates: int
    n_release_proteins: int
    refused: list[RefusedIsolate]
    s288c_derived: list[S288cDerivedIsolate] = Field(
        description="release columns built on S288C with no perturbation (FY4)"
    )
    n_built: int
    n_cells_built: int
    n_measured_built: int
    n_missing_built: int = Field(
        description="blank cells of built isolates: key-absent, never imputed"
    )
    missing_by_isolate: dict[str, int]
    reference_isolates: list[str] = Field(
        description="the euploid isolates of SI Table 7 the reference medians run over"
    )
    karyotypes: list[IsolateKaryotype]


class TurnoverLedger(BaseModel):
    """Every row of SI Table 17, and the turnover-specific context it does not carry."""

    sheet: str = T17.name
    n_release_rows: int
    n_release_isolates: int
    refused: list[RefusedIsolate]
    n_built_isolates: int
    n_built_rows: int
    reference_isolates: list[str]
    n_proteins: int
    n_proteins_with_reference: int
    proteins_without_reference: list[str]
    n_rows_without_reference: int
    half_life_identity_max_relative_error: float = Field(
        description="max |HL - ln2/kdp| / HL over the released rows"
    )
    doubling_time_minutes: dict[str, float] = Field(
        description="SI Table 16, per built isolate: the culture's doubling time during "
        "the turnover experiment, which kdp is NOT corrected for; no record field "
        "carries it"
    )
    karyotypes: list[IsolateKaryotype]


class TimecourseSample(BaseModel):
    """One released (isolate, replicate culture, time point) sample of SI Table 13."""

    release_code: str
    isolate: str = Field(description="the Peter 2018 code the genotype is built for")
    replicate: int = Field(description="SI Table 13 'replicate', as released")
    timepoint: str = Field(description="SI Table 13 'timepoint', as released")
    time_hours: float = Field(description="SI Table 13 'time (h)', as released")
    od600: float = Field(
        description="SI Table 13 'OD600' at sampling, as released; no field of "
        "ProteinAbundancePhenotype carries a harvest density, so it lives here"
    )
    n_measured: int = Field(description="genes with a numeric abundance")
    n_missing: int = Field(description="genes the release writes 'NA' for")


class TimecourseLedger(BaseModel):
    """Every row and sample of SI Table 13, accounted for."""

    sheet: str = T13.name
    n_release_rows: int
    n_genes: int
    n_measured_cells: int
    n_missing_cells: int
    refused: list[RefusedIsolate]
    samples: list[TimecourseSample]
    reference_isolate: str = Field(
        description="the euploid isolate (SI Table 1) whose replicates the per-time-point "
        "reference medians run over"
    )
    reference_min_replicates: int
    reference_genes_by_timepoint: dict[str, int]
    karyotypes: list[IsolateKaryotype]


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def _env_data_root() -> str:
    """``DATA_ROOT`` from the environment or the repo ``.env``."""
    from dotenv import load_dotenv

    load_dotenv()
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/muenznerNaturalProteomeDiversity2024``."""
    return Path(data_root or _env_data_root()) / RAW_DIR_REL


def deposit_raw_mirror(source: str | Path, data_root: str | None = None) -> Path:
    """Copy the workbook into the raw mirror and write its ``manifest.json``.

    ``source`` is the library capture ``si/si3.xlsx`` (fetched from the PMC bucket by
    ``torchcell.literature.retrieve.pmc_cloud_object``). The source is verified against
    the pin before anything is written; an existing mirror file whose digest differs
    raises rather than being overwritten.
    """
    root = raw_mirror_dir(data_root)
    dest = root / SI_TABLES_REL
    dest.parent.mkdir(parents=True, exist_ok=True)
    copy_verified(source, dest, SI_TABLES_SHA256)
    url = pmc_cloud_url(SI_TABLES_FILENAME)
    record = ArtifactRecord(
        path=SI_TABLES_REL,
        role=ROLE_RAW_DATA,
        bytes=dest.stat().st_size,
        sha256=SI_TABLES_SHA256,
        source=url,
        retrieval=RetrievalRecord(
            method=RetrievalMethod.pmc_cloud,
            source_url=url,
            retriever=PMC_RETRIEVER,
            params={"key": pmc_cloud_key(SI_TABLES_FILENAME)},
            sha256=SI_TABLES_SHA256,
            retrieved_at=RAW_RETRIEVED_AT,
        ),
    )
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=DOI,
        title=TITLE,
        files=[record],
        si_data_sources=[
            url,
            f"https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?accession={MASSIVE_ACCESSION}",
            f"https://www.ebi.ac.uk/pride/archive/projects/{PRIDE_DISOMES}",
            f"https://www.ebi.ac.uk/pride/archive/projects/{PRIDE_TURNOVER}",
        ],
        si_expected=[
            "Supplementary Tables 1-20 (MOESM3) -- deposited; the proteome loader reads "
            "SI Tables 1, 3, 5, 6 and 7, the turnover loader SI Tables 1, 16 and 17",
            "Supplementary Information guide + Supplementary Fig. 1 (MOESM1), the "
            "reporting summary (MOESM2) and the peer review file (MOESM4) are quoted "
            "from the torchcell-library OCR mirror, not the raw mirror",
            f"MassIVE {MASSIVE_ACCESSION} (natural collection), PRIDE {PRIDE_DISOMES} "
            f"(disomes) and PRIDE {PRIDE_TURNOVER} (turnover) -- raw and processed "
            "mass-spectrometry files. NOT deposited: no loader consumes spectra or "
            "search output; the released matrices are SI Tables 3, 4 and 17",
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
# Shared dataset machinery
# --------------------------------------------------------------------------- #
GENOME_REFERENCE = ReferenceGenome(
    species="Saccharomyces cerevisiae",
    strain="S288C",
    assembly_set=SGD_S288C_R64,
    assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
)
PUBLICATION = Publication(
    pubmed_id=PUBMED_ID,
    pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{PUBMED_ID}/",
    doi=DOI,
    doi_url=f"https://doi.org/{DOI}",
)


class _MuenznerIsolateDataset(ExperimentDataset):
    """Raw files, download and genotype assembly shared by the two Muenzner loaders."""

    def _data_root(self) -> str:
        """Return ``$DATA_ROOT`` (mirror, genomes tier and build trees live under it)."""
        return _env_data_root()

    @property
    def raw_file_names(self) -> list[str]:
        """The SI workbook and Peter's six genotype files."""
        return RAW_FILES

    def download(self) -> None:
        """Link the workbook (raw mirror) and Peter's files (genomes tier) into ``raw/``.

        The workbook's manifest record must equal the module pin; each Peter file is
        verified against the genomes tier's manifest. ``process`` re-verifies all seven.
        """
        data_root = self._data_root()
        os.makedirs(self.raw_dir, exist_ok=True)
        manifest = load_manifest(data_root)
        check_manifest_pin(
            SI_TABLES_REL, manifest_sha256(manifest, SI_TABLES_REL), SI_TABLES_SHA256
        )
        link_verified(
            raw_mirror_dir(data_root) / SI_TABLES_REL,
            osp.join(self.raw_dir, SI_TABLES_FILENAME),
            SI_TABLES_SHA256,
        )
        link_genotype_files(self.raw_dir, data_root)
        log.info("Muenzner 2024 raw files linked into %s", self.raw_dir)

    def _verify_raw(self, data_root: str) -> None:
        """Verify all seven raw files before any is read."""
        peter = load_genome_manifest(PETER2018_1011, data_root)
        verify_raw_files(
            self.raw_dir,
            {
                SI_TABLES_FILENAME: SI_TABLES_SHA256,
                **{name: peter.record(name).sha256 for name in GENOTYPE_FILES},
            },
        )

    def _genotypes(self, data_root: str, strains: list[str]) -> NaturalIsolateGenotypes:
        """The isolate genotypes of ``strains``; their ledger goes to ``preprocess/``."""
        genotypes = NaturalIsolateGenotypes(
            self.raw_dir, self.preprocess_dir, strains, data_root
        )
        genotypes.write_ledger(osp.join(self.preprocess_dir, GENOTYPE_LEDGER))
        return genotypes

    def _write_records(
        self,
        records: Iterator[tuple[Experiment, ExperimentReference]],
        n: int,
        desc: str,
    ) -> int:
        """Write ``(experiment, reference)`` pairs to ``processed/lmdb``."""
        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(2e11))
        idx = 0
        with env.begin(write=True) as txn:
            for experiment, reference in tqdm(records, total=n, desc=desc):
                txn.put(
                    f"{idx}".encode(),
                    pickle.dumps(
                        {
                            "experiment": experiment.model_dump(),
                            "reference": reference.model_dump(),
                            "publication": PUBLICATION.model_dump(),
                        }
                    ),
                )
                idx += 1
        env.close()
        return idx

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Preprocessing happens inside ``process()``."""
        return df

    def create_experiment(self) -> None:
        """Records are built inside ``process()``."""
        raise NotImplementedError("records are built inside process()")


def _flags(tables: dict[str, pd.DataFrame]) -> tuple[set[str], set[str], set[str]]:
    """Release codes of SI Table 7 (integrated), 5 and 6 (karyotype exclusions)."""
    integrated = set(tables[T7.name]["Standardized_name"].astype(str))
    table5 = set(tables[T5.name]["strain"].astype(str))
    table6 = set(tables[T6.name]["strain"].astype(str))
    return integrated, table5, table6


# --------------------------------------------------------------------------- #
# Proteome
# --------------------------------------------------------------------------- #
#: How SI Table 3 writes a missing quantity: the R text ``NA`` (37,667 cells), never an
#: empty cell or a zero.
RELEASE_MISSING = "NA"


def proteome_matrix(table3: pd.DataFrame) -> pd.DataFrame:
    """SI Table 3 as a float frame, proteins (systematic ORF) x release isolate codes.

    The release writes a missing quantity as the text ``NA``; that becomes NaN (a
    key-absent protein on the record). Any other non-numeric cell, an empty cell, a
    non-positive value or a repeated protein raises: the released quantity is a positive
    linear MaxLFQ value or ``NA``.
    """
    values = table3.set_index("Protein")
    if values.index.has_duplicates:
        raise ValueError("SI Table 3 repeats a protein")
    if bool(values.isna().to_numpy().any()):
        raise ValueError("SI Table 3 holds an empty cell (missing is written 'NA')")
    numeric = values.mask(values == RELEASE_MISSING).apply(
        pd.to_numeric, errors="coerce"
    )
    bad = int(((values != RELEASE_MISSING) & numeric.isna()).to_numpy().sum())
    if bad:
        raise ValueError(f"SI Table 3 holds {bad} non-numeric cells other than 'NA'")
    if bool((numeric <= 0).to_numpy().any()):
        raise ValueError("SI Table 3 holds a non-positive quantity")
    numeric.index = numeric.index.astype(str)
    out: pd.DataFrame = numeric.astype(float)
    return out


def median_reference(
    values: pd.DataFrame, isolates: list[str]
) -> tuple[dict[str, float], dict[str, int]]:
    """Per-row median over the measured cells of ``isolates`` and how many there were.

    A row no listed isolate measured has no key.
    """
    sub = values[isolates]
    counts = sub.notna().sum(axis=1)
    medians = sub.median(axis=1, skipna=True)
    keep = counts > 0
    return (
        {str(k): float(v) for k, v in medians[keep].items()},
        {str(k): int(v) for k, v in counts[keep].items()},
    )


@register_dataset
class ProteomeMuenzner2024Dataset(_MuenznerIsolateDataset):
    """796 SWATH-MS proteomes (SI Table 3): 794 isolates with a Peter genome and FY4."""

    organism = Organism.scerevisiae

    def __init__(
        self,
        root: str = "data/torchcell/proteome_muenzner2024",
        io_workers: int = 0,
        transform: Any | None = None,
        pre_transform: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset (one record per natural isolate)."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return ProteinAbundanceExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return ProteinAbundanceExperimentReference

    @post_process
    def process(self) -> None:
        """Build one proteome record per matched isolate and write the ledger."""
        data_root = self._data_root()
        self._verify_raw(data_root)
        os.makedirs(self.preprocess_dir, exist_ok=True)
        tables = read_sheets(
            osp.join(self.raw_dir, SI_TABLES_FILENAME), [T1, T3, T5, T6, T7]
        )
        values = proteome_matrix(tables[T3.name])
        release_codes = [str(c) for c in values.columns]
        matched, s288c, refused = match_isolates(
            release_codes, panel_isolates(self.raw_dir), tables[T1.name]
        )
        integrated, table5, table6 = _flags(tables)
        t7 = tables[T7.name]
        reference_codes = sorted(
            t7.loc[t7["Aneuploidies"] == EUPLOID, "Standardized_name"].astype(str)
        )
        outside = sorted(set(reference_codes) - set(matched))
        if outside:
            raise ValueError(f"reference isolates without a built record: {outside}")
        ref_abundance, ref_n = median_reference(values, reference_codes)
        if set(ref_abundance) != set(values.index):
            raise ValueError("a protein has no euploid measurement in SI Table 7")
        built_name = {
            **matched,
            **{isolate.release_code: isolate.strain for isolate in s288c},
        }
        built = values[list(built_name)]
        missing = built.isna().sum(axis=0)
        ledger = ProteomeLedger(
            n_release_isolates=len(release_codes),
            n_release_proteins=len(values.index),
            refused=refused,
            s288c_derived=s288c,
            n_built=len(built_name),
            n_cells_built=int(built.size),
            n_measured_built=int(built.notna().to_numpy().sum()),
            n_missing_built=int(missing.sum()),
            missing_by_isolate={
                built_name[str(c)]: int(n) for c, n in missing.items() if n
            },
            reference_isolates=[matched[c] for c in reference_codes],
            karyotypes=karyotypes(matched, tables[T1.name], integrated, table5, table6),
        )
        with open(osp.join(self.preprocess_dir, "muenzner2024_ledger.json"), "w") as f:
            f.write(ledger.model_dump_json(indent=2))
        log.info(
            "SI Table 3: %d isolates, %d refused (%s), %d built (%d S288C-derived); "
            "%d missing cells",
            len(release_codes),
            len(refused),
            [r.release_code for r in refused],
            ledger.n_built,
            len(s288c),
            ledger.n_missing_built,
        )

        environment = Environment(media=MUENZNER_SM, temperature=Temperature(value=30))
        reference = ProteinAbundanceExperimentReference(
            dataset_name=self.name,
            genome_reference=GENOME_REFERENCE,
            environment_reference=environment.model_copy(),
            phenotype_reference=ProteinAbundancePhenotype(
                protein_abundance=ref_abundance,
                protein_abundance_se=None,
                n_replicates=ref_n,
                measurement_type=PROTEOME_MEASUREMENT_TYPE,
            ),
        )
        by_peter = {p: c for c, p in matched.items()}
        strains = sorted(by_peter)
        genotypes = self._genotypes(data_root, strains)
        pd.DataFrame(
            {"strain_id": [*strains, *(isolate.strain for isolate in s288c)]}
        ).to_csv(osp.join(self.preprocess_dir, "data.csv"), index=False)

        def phenotype_of(code: str) -> ProteinAbundancePhenotype:
            column = values[code].dropna()
            abundance = {str(k): float(v) for k, v in column.items()}
            return ProteinAbundancePhenotype(
                protein_abundance=abundance,
                protein_abundance_se=None,
                n_replicates={k: 1 for k in abundance},
                measurement_type=PROTEOME_MEASUREMENT_TYPE,
            )

        def records() -> Iterator[tuple[Experiment, ExperimentReference]]:
            for strain in strains:
                experiment = ProteinAbundanceExperiment(
                    dataset_name=self.name,
                    genotype=genotypes.genotype(strain),
                    environment=environment,
                    phenotype=phenotype_of(by_peter[strain]),
                )
                yield experiment, reference
            # An S288C-derived column (FY4): no perturbation off S288C, its own genome.
            for isolate in s288c:
                experiment = ProteinAbundanceExperiment(
                    dataset_name=self.name,
                    genotype=Genotype(perturbations=[]),
                    environment=environment,
                    phenotype=phenotype_of(isolate.release_code),
                )
                yield experiment, s288c_derived_reference(reference, isolate)

        n = self._write_records(
            records(), len(strains) + len(s288c), "proteome isolates"
        )
        log.info("Wrote %d proteome records", n)


# --------------------------------------------------------------------------- #
# Turnover
# --------------------------------------------------------------------------- #
def turnover_frame(table17: pd.DataFrame) -> pd.DataFrame:
    """SI Table 17 with numeric ``kdp_value`` and ``HL``.

    A repeated (strain, protein) pair, or a blank or non-numeric rate or half-life,
    raises.
    """
    frame = table17.copy()
    frame["strain"] = frame["strain"].astype(str)
    frame["systematic_name"] = frame["systematic_name"].astype(str)
    for column in ("kdp_value", "HL"):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if bool(numeric.isna().any()):
            raise ValueError(
                f"SI Table 17 '{column}' holds a blank or non-numeric cell"
            )
        frame[column] = numeric.astype(float)
    if bool(frame.duplicated(["strain", "systematic_name"]).any()):
        raise ValueError("SI Table 17 repeats a (strain, systematic_name) pair")
    return frame


def half_life_identity_error(frame: pd.DataFrame) -> float:
    """Maximum of ``|HL - ln 2 / kdp| / HL`` over the rows (Methods: ``log(2)/kdp``)."""
    rel = (np.log(2) / frame["kdp_value"] - frame["HL"]).abs() / frame["HL"]
    return float(rel.max())


def check_annotation_agreement(frame: pd.DataFrame, annotations: pd.DataFrame) -> None:
    """SI Table 17's per-row Ploidy / Aneuploidies equal SI Table 1's for that isolate."""
    rows = annotations.set_index("Standardized_name")
    for code, sub in frame.groupby("strain"):
        for column in ("Ploidy", "Aneuploidies"):
            released = {str(v) for v in sub[column]}
            expected = str(rows.loc[str(code), column])
            if released != {expected}:
                raise ValueError(
                    f"SI Table 17 {column} for {code} is {sorted(released)}, "
                    f"SI Table 1 says {expected!r}"
                )


@register_dataset
class ProteinTurnoverMuenzner2024Dataset(_MuenznerIsolateDataset):
    """Dynamic-SILAC turnover rates for 55 natural isolates (SI Table 17)."""

    organism = Organism.scerevisiae

    def __init__(
        self,
        root: str = "data/torchcell/protein_turnover_muenzner2024",
        io_workers: int = 0,
        transform: Any | None = None,
        pre_transform: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset (one record per natural isolate)."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return ProteinTurnoverExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return ProteinTurnoverExperimentReference

    @post_process
    def process(self) -> None:
        """Build one turnover record per matched isolate and write the ledger."""
        data_root = self._data_root()
        self._verify_raw(data_root)
        os.makedirs(self.preprocess_dir, exist_ok=True)
        tables = read_sheets(
            osp.join(self.raw_dir, SI_TABLES_FILENAME), [T1, T5, T6, T7, T16, T17]
        )
        frame = turnover_frame(tables[T17.name])
        annotations = tables[T1.name]
        check_annotation_agreement(frame, annotations)
        release_codes = sorted(frame["strain"].unique())
        matched, s288c, refused = match_isolates(
            release_codes, panel_isolates(self.raw_dir), annotations
        )
        if s288c:
            raise RuntimeError(
                f"SI Table 17 lists the S288C-derived column(s) "
                f"{[i.release_code for i in s288c]}; the released Table 17 has none, "
                "and this loader has no record path for one"
            )
        built = frame[frame["strain"].isin(set(matched))]
        aneu = annotations.set_index("Standardized_name")["Aneuploidies"]
        reference_codes = sorted(c for c in matched if str(aneu[c]) == EUPLOID)
        rates = built.pivot(
            index="systematic_name", columns="strain", values="kdp_value"
        )
        ref_rate, ref_n = median_reference(rates, reference_codes)
        without = sorted(set(rates.index) - set(ref_rate))
        doubling = tables[T16.name].set_index("strain")["doubling time (min)"]
        integrated, table5, table6 = _flags(tables)
        ledger = TurnoverLedger(
            n_release_rows=len(frame),
            n_release_isolates=len(release_codes),
            refused=refused,
            n_built_isolates=len(matched),
            n_built_rows=len(built),
            reference_isolates=[matched[c] for c in reference_codes],
            n_proteins=len(rates.index),
            n_proteins_with_reference=len(ref_rate),
            proteins_without_reference=without,
            n_rows_without_reference=int(built["systematic_name"].isin(without).sum()),
            half_life_identity_max_relative_error=half_life_identity_error(frame),
            doubling_time_minutes={
                matched[c]: float(doubling[c]) for c in sorted(matched)
            },
            karyotypes=karyotypes(matched, annotations, integrated, table5, table6),
        )
        with open(osp.join(self.preprocess_dir, "muenzner2024_ledger.json"), "w") as f:
            f.write(ledger.model_dump_json(indent=2))
        log.info(
            "SI Table 17: %d rows, %d isolates (%d refused), reference over %d euploid "
            "isolates; %d proteins (%d rows) have no euploid reference",
            len(frame),
            len(release_codes),
            len(refused),
            len(reference_codes),
            len(without),
            ledger.n_rows_without_reference,
        )

        environment = Environment(
            media=MUENZNER_SM_LYS8, temperature=Temperature(value=30)
        )
        reference = ProteinTurnoverExperimentReference(
            dataset_name=self.name,
            genome_reference=GENOME_REFERENCE,
            environment_reference=environment.model_copy(),
            phenotype_reference=ProteinTurnoverPhenotype(
                degradation_rate=ref_rate,
                n_replicates=ref_n,
                measurement_type=TURNOVER_MEASUREMENT_TYPE,
            ),
        )
        by_peter = {p: c for c, p in matched.items()}
        strains = sorted(by_peter)
        genotypes = self._genotypes(data_root, strains)
        pd.DataFrame({"strain_id": strains}).to_csv(
            osp.join(self.preprocess_dir, "data.csv"), index=False
        )
        grouped = {str(code): sub for code, sub in built.groupby("strain")}

        def records() -> Iterator[tuple[Experiment, ExperimentReference]]:
            for strain in strains:
                sub = grouped[by_peter[strain]]
                keys = sub["systematic_name"].tolist()
                experiment = ProteinTurnoverExperiment(
                    dataset_name=self.name,
                    genotype=genotypes.genotype(strain),
                    environment=environment,
                    phenotype=ProteinTurnoverPhenotype(
                        degradation_rate=dict(
                            zip(keys, sub["kdp_value"].tolist(), strict=True)
                        ),
                        half_life=dict(zip(keys, sub["HL"].tolist(), strict=True)),
                        n_replicates={k: 1 for k in keys},
                        measurement_type=TURNOVER_MEASUREMENT_TYPE,
                    ),
                )
                yield experiment, reference

        n = self._write_records(records(), len(strains), "turnover isolates")
        log.info("Wrote %d turnover records", n)


# --------------------------------------------------------------------------- #
# Time course (SI Table 13)
# --------------------------------------------------------------------------- #
#: The paper's own reference rule: a protein enters a time point's reference when the
#: euploid isolate measured it in at least two of its three replicates.
TIMECOURSE_REFERENCE_MIN_REPLICATES = 2
SAMPLE_KEYS = ["strain", "replicate", "timepoint"]

SINGLE_CULTURE_SE_NOTE = (
    "one replicate culture at one time point is one measurement, so a standard error "
    "over replicates does not apply to the record; the other two cultures of the "
    "triplicate are their own records"
)


def timecourse_frame(table13: pd.DataFrame) -> pd.DataFrame:
    """SI Table 13 typed: ``abundance`` float (the release's ``NA`` -> NaN).

    Every abundance cell is a positive number or the text ``NA``; an empty or other
    cell raises. A repeated (strain, replicate, timepoint, gene) row raises, and so does
    a sample whose rows disagree on ``time (h)`` or ``OD600``, or a time point whose
    replicates of one isolate were not sampled at one ``time (h)`` (the time is part of
    the environment the replicates share).
    """
    frame = table13.copy()
    for column in ("strain", "timepoint", "Genes"):
        if bool(frame[column].isna().any()):
            raise ValueError(f"SI Table 13 '{column}' holds an empty cell")
        frame[column] = frame[column].astype(str)
    frame["replicate"] = frame["replicate"].astype(int)
    for column in ("time (h)", "OD600"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    raw = frame["abundance"]
    if bool(raw.isna().any()):
        raise ValueError("SI Table 13 holds an empty abundance cell (missing is 'NA')")
    numeric = pd.to_numeric(raw.where(raw != RELEASE_MISSING), errors="coerce")
    bad = int(((raw != RELEASE_MISSING) & numeric.isna()).sum())
    if bad:
        raise ValueError(f"SI Table 13 holds {bad} non-numeric cells other than 'NA'")
    if bool((numeric <= 0).any()):
        raise ValueError("SI Table 13 holds a non-positive abundance")
    frame["abundance"] = numeric.astype(float)
    if bool(frame.duplicated([*SAMPLE_KEYS, "Genes"]).any()):
        raise ValueError(
            "SI Table 13 repeats a (strain, replicate, timepoint, gene) row"
        )
    per_sample = frame.groupby(SAMPLE_KEYS)[["time (h)", "OD600"]].nunique()
    if bool((per_sample > 1).to_numpy().any()):
        raise ValueError("an SI Table 13 sample carries two times or two OD600 values")
    per_time = frame.groupby(["strain", "timepoint"])["time (h)"].nunique()
    if bool((per_time > 1).any()):
        raise ValueError(
            "the replicates of one SI Table 13 isolate and time point were sampled at "
            "different times"
        )
    return frame


def timecourse_samples(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per released sample: its time, OD600 and measured / missing counts."""
    grouped = frame.groupby(SAMPLE_KEYS, sort=False)
    samples = grouped.agg(
        time_hours=("time (h)", "first"),
        od600=("OD600", "first"),
        n_measured=("abundance", "count"),
        n_rows=("abundance", "size"),
    ).reset_index()
    samples["n_missing"] = samples["n_rows"] - samples["n_measured"]
    out: pd.DataFrame = samples.sort_values(["strain", "replicate", "time_hours"])
    return out.reset_index(drop=True)


def timecourse_reference(
    frame: pd.DataFrame, strain: str, min_replicates: int
) -> dict[str, tuple[dict[str, float], dict[str, int]]]:
    """Per time point, ``strain``'s per-gene median over its replicates and their count.

    A gene enters a time point's reference only when at least ``min_replicates`` of the
    strain's replicates measured it (the paper's rule); the median runs over those.
    """
    sub = frame[frame["strain"] == strain]
    out: dict[str, tuple[dict[str, float], dict[str, int]]] = {}
    for timepoint, block in sub.groupby("timepoint"):
        wide = block.pivot(index="Genes", columns="replicate", values="abundance")
        counts = wide.notna().sum(axis=1)
        keep = counts >= min_replicates
        medians = wide[keep].median(axis=1, skipna=True)
        out[str(timepoint)] = (
            {str(k): float(v) for k, v in medians.items()},
            {str(k): int(v) for k, v in counts[keep].items()},
        )
    return out


@register_dataset
class ProteomeTimecourseMuenzner2024Dataset(_MuenznerIsolateDataset):
    """SI Table 13: two isolates x 3 replicate cultures x 5 time points over 24 h."""

    organism = Organism.scerevisiae

    def __init__(
        self,
        root: str = "data/torchcell/proteome_timecourse_muenzner2024",
        io_workers: int = 0,
        transform: Any | None = None,
        pre_transform: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset (one record per replicate culture and time point)."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return ProteinAbundanceExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return ProteinAbundanceExperimentReference

    @post_process
    def process(self) -> None:
        """Build one record per released sample and write the ledger."""
        data_root = self._data_root()
        self._verify_raw(data_root)
        os.makedirs(self.preprocess_dir, exist_ok=True)
        tables = read_sheets(
            osp.join(self.raw_dir, SI_TABLES_FILENAME), [T1, T5, T6, T7, T13]
        )
        frame = timecourse_frame(tables[T13.name])
        annotations = tables[T1.name]
        release_codes = sorted(frame["strain"].unique())
        matched, s288c, refused = match_isolates(
            release_codes, panel_isolates(self.raw_dir), annotations
        )
        if s288c or refused:
            raise RuntimeError(
                "SI Table 13 names an isolate with no Peter 2018 genome or an "
                f"S288C-derived column ({[r.release_code for r in refused]}, "
                f"{[i.release_code for i in s288c]}); the released table has neither, "
                "and this loader has no record path for one"
            )
        aneu = annotations.set_index("Standardized_name")["Aneuploidies"]
        euploid = [c for c in release_codes if str(aneu[c]) == EUPLOID]
        if len(euploid) != 1:
            raise ValueError(
                f"SI Table 13 must hold exactly one euploid isolate, got {euploid}"
            )
        reference_code = euploid[0]
        references = timecourse_reference(
            frame, reference_code, TIMECOURSE_REFERENCE_MIN_REPLICATES
        )
        samples = timecourse_samples(frame)
        timepoints = set(samples["timepoint"])
        if set(references) != timepoints:
            raise ValueError(
                f"the euploid isolate {reference_code} lacks time points "
                f"{sorted(timepoints - set(references))}"
            )
        ref_time = (
            samples[samples["strain"] == reference_code]
            .groupby("timepoint")["time_hours"]
            .first()
        )
        integrated, table5, table6 = _flags(tables)
        ledger = TimecourseLedger(
            n_release_rows=len(frame),
            n_genes=int(frame["Genes"].nunique()),
            n_measured_cells=int(frame["abundance"].notna().sum()),
            n_missing_cells=int(frame["abundance"].isna().sum()),
            refused=refused,
            samples=[
                TimecourseSample(
                    release_code=str(row["strain"]),
                    isolate=matched[str(row["strain"])],
                    replicate=int(row["replicate"]),
                    timepoint=str(row["timepoint"]),
                    time_hours=float(row["time_hours"]),
                    od600=float(row["od600"]),
                    n_measured=int(row["n_measured"]),
                    n_missing=int(row["n_missing"]),
                )
                for row in samples.to_dict("records")
            ],
            reference_isolate=matched[reference_code],
            reference_min_replicates=TIMECOURSE_REFERENCE_MIN_REPLICATES,
            reference_genes_by_timepoint={
                t: len(references[t][0]) for t in sorted(references)
            },
            karyotypes=karyotypes(matched, annotations, integrated, table5, table6),
        )
        with open(osp.join(self.preprocess_dir, "muenzner2024_ledger.json"), "w") as f:
            f.write(ledger.model_dump_json(indent=2))
        log.info(
            "SI Table 13: %d rows, %d samples, %d measured cells; reference isolate %s",
            ledger.n_release_rows,
            len(ledger.samples),
            ledger.n_measured_cells,
            reference_code,
        )

        def environment_at(hours: float) -> Environment:
            return Environment(
                media=MUENZNER_SM,
                temperature=Temperature(value=30),
                duration_hours=hours,
            )

        reference_by_timepoint: dict[str, ProteinAbundanceExperimentReference] = {}
        for timepoint, (abundance, counts) in references.items():
            reference_by_timepoint[timepoint] = ProteinAbundanceExperimentReference(
                dataset_name=self.name,
                genome_reference=GENOME_REFERENCE,
                environment_reference=environment_at(float(ref_time[timepoint])),
                phenotype_reference=ProteinAbundancePhenotype(
                    protein_abundance=abundance,
                    protein_abundance_se=None,
                    n_replicates=counts,
                    measurement_type=TIMECOURSE_MEASUREMENT_TYPE,
                    replicate_id=None,
                ),
            )
        strains = sorted(matched.values())
        genotypes = self._genotypes(data_root, strains)
        samples["isolate"] = samples["strain"].map(matched)
        samples = samples.sort_values(["isolate", "replicate", "time_hours"])
        samples[["isolate", "replicate", "timepoint", "time_hours"]].to_csv(
            osp.join(self.preprocess_dir, "data.csv"), index=False
        )
        grouped: dict[tuple[str, int, str], pd.DataFrame] = {}
        for key, block in frame.groupby(SAMPLE_KEYS):
            strain, replicate, timepoint = cast(tuple[Any, Any, Any], key)
            grouped[(str(strain), int(replicate), str(timepoint))] = block

        def records() -> Iterator[tuple[Experiment, ExperimentReference]]:
            for row in samples.to_dict("records"):
                block = grouped[
                    (str(row["strain"]), int(row["replicate"]), str(row["timepoint"]))
                ]
                measured = block.dropna(subset=["abundance"])
                abundance = dict(
                    zip(
                        measured["Genes"].tolist(),
                        measured["abundance"].astype(float).tolist(),
                        strict=True,
                    )
                )
                experiment = ProteinAbundanceExperiment(
                    dataset_name=self.name,
                    genotype=genotypes.genotype(str(row["isolate"])),
                    environment=environment_at(float(row["time_hours"])),
                    phenotype=ProteinAbundancePhenotype(
                        protein_abundance=abundance,
                        protein_abundance_se=None,
                        n_replicates=dict.fromkeys(abundance, 1),
                        measurement_type=TIMECOURSE_MEASUREMENT_TYPE,
                        replicate_id=str(row["replicate"]),
                        provenance_gaps=[
                            ProvenanceGap(
                                field="protein_abundance_se",
                                reason=ProvenanceGapReason.not_applicable_single_measurement,
                                note=SINGLE_CULTURE_SE_NOTE,
                            )
                        ],
                    ),
                )
                yield experiment, reference_by_timepoint[str(row["timepoint"])]

        n = self._write_records(records(), len(samples), "time-course samples")
        log.info("Wrote %d time-course records", n)


# --------------------------------------------------------------------------- #
# Verification (``build_dataset_lmdb --verify`` calls ``verify_build``)
# --------------------------------------------------------------------------- #
#: Records each built store must hold (measured from the release; see the ledger):
#: 794 isolates with a Peter genome plus the S288C-derived FY4 column (#1045).
EXPECTED_PROTEOME_RECORDS = 795
EXPECTED_TURNOVER_RECORDS = 55
#: SI Table 13: 2 isolates x 3 replicate cultures x 5 time points (measured).
EXPECTED_TIMECOURSE_RECORDS = 30
PROTEOME_STORE = "proteome_muenzner2024"
TURNOVER_STORE = "protein_turnover_muenzner2024"
TIMECOURSE_STORE = "proteome_timecourse_muenzner2024"
#: Isolates of the built Caudal 2024 store that also carry a proteome (measured).
EXPECTED_CAUDAL_OVERLAP = 759
CAUDAL_ROOT = "data/torchcell/caudal_pantranscriptome2024"


def verify_build(dataset_root: str, *, data_root: str | None = None) -> Any:
    """Run the L0-L4 gate for whichever Muenzner store ``dataset_root`` is."""
    from torchcell.verification.natural_isolate_genotype import IsolateGenotypeAudit
    from torchcell.verification.protein import (
        natural_isolate_of,
        verify_natural_isolate_proteome,
        verify_natural_isolate_timecourse_proteome,
        verify_natural_isolate_turnover,
    )
    from torchcell.verification.runners import load_records

    name = osp.basename(osp.normpath(dataset_root))
    methods = {
        PROTEOME_STORE: (
            T3.name,
            "SI Table 3 non-imputed MaxLFQ quantities per natural isolate; genotype "
            "vs S288C from Peter 2018 by torchcell.sequence.natural_isolate",
        ),
        TURNOVER_STORE: (
            T17.name,
            "SI Table 17 dynamic-SILAC kdp (1/h) and HL (h) per natural isolate",
        ),
        TIMECOURSE_STORE: (
            T13.name,
            "SI Table 13 DIA-NN 1.8 gene quantities per (isolate, replicate culture, "
            "time point), keyed by the released gene names; reference = the euploid "
            "isolate's per-time-point replicate median",
        ),
    }
    if name not in methods:
        raise ValueError(f"{dataset_root} is not a Muenzner 2024 store")
    data_root = data_root or _env_data_root()
    records = load_records(dataset_root)
    page, method = methods[name]
    provenance = Provenance(
        source_uri=f"$DATA_ROOT/{RAW_DIR_REL}/{SI_TABLES_REL}",
        citation_key=CITATION_KEY,
        sha256=SI_TABLES_SHA256,
        method=method,
        page=page,
        retrieved=RAW_RETRIEVED_AT,
    )
    # The isolates of the Peter panel the store's genotypes name; FY4, an S288C-derived
    # background with no descriptor, is not one of them (#1045).
    isolates = sorted(
        {
            isolate
            for r in records
            if (isolate := genotype_isolate(r["experiment"]["genotype"])) is not None
        }
    )
    audit = IsolateGenotypeAudit(
        NaturalIsolateGenotypes(
            osp.join(dataset_root, "raw"),
            osp.join(dataset_root, "preprocess"),
            isolates,
            data_root,
        ),
        osp.join(data_root, "torchcell-library"),
    )
    if name == TIMECOURSE_STORE:
        ledger_t13 = TimecourseLedger.model_validate_json(
            Path(dataset_root, "preprocess", "muenzner2024_ledger.json").read_text()
        )
        return verify_natural_isolate_timecourse_proteome(
            records,
            dataset_name=name,
            provenance=provenance,
            expected_count=EXPECTED_TIMECOURSE_RECORDS,
            reference_isolate=ledger_t13.reference_isolate,
            min_replicates=ledger_t13.reference_min_replicates,
            sgd_names=sgd_gene_and_standard_names(data_root),
            genotype_audit=audit,
        )
    if name == PROTEOME_STORE:
        caudal_order = caudal_strain_order(data_root)
        built = {
            strain for r in records if (strain := natural_isolate_of(r)) is not None
        }
        return verify_natural_isolate_proteome(
            records,
            dataset_name=name,
            provenance=provenance,
            expected_count=EXPECTED_PROTEOME_RECORDS,
            sgd_genes=_sgd_genes(data_root),
            caudal_isolates=set(caudal_order),
            expected_caudal_overlap=EXPECTED_CAUDAL_OVERLAP,
            caudal_genotypes=caudal_genotype_sample(
                data_root, caudal_order, built & set(caudal_order)
            ),
            genotype_audit=audit,
        )
    ledger = TurnoverLedger.model_validate_json(
        Path(dataset_root, "preprocess", "muenzner2024_ledger.json").read_text()
    )
    return verify_natural_isolate_turnover(
        records,
        dataset_name=name,
        provenance=provenance,
        expected_count=EXPECTED_TURNOVER_RECORDS,
        sgd_genes=_sgd_genes(data_root),
        expected_rows_without_reference=ledger.n_rows_without_reference,
        genotype_audit=audit,
    )


def caudal_strain_order(data_root: str) -> list[str]:
    """The built Caudal 2024 store's isolates in LMDB order (its ``preprocess/data.csv``).

    The Caudal loader writes record ``i`` for the ``i``-th of its sorted strains and the
    same list to ``data.csv``; ``caudal_genotype_sample`` re-checks that per record.
    """
    path = osp.join(data_root, CAUDAL_ROOT, "preprocess", "data.csv")
    return pd.read_csv(path)["strain_id"].astype(str).tolist()


def caudal_genotype_sample(
    data_root: str, order: list[str], overlap: set[str]
) -> dict[str, dict[str, Any]]:
    """Genotype dumps of a fixed sample of shared isolates, read from Caudal's LMDB.

    The sample is the first, middle and last shared isolate by code plus the first
    ``SACE_`` one (the spelling the release changes), so the dot-to-underscore mapping is
    always among the compared genotypes.
    """
    ordered = sorted(overlap)
    sace = [s for s in ordered if s.startswith("SACE_")]
    picks = sorted({ordered[0], ordered[len(ordered) // 2], ordered[-1], *sace[:1]})
    env = lmdb.open(
        osp.join(data_root, CAUDAL_ROOT, "processed", "lmdb"),
        readonly=True,
        lock=False,
        readahead=False,
    )
    out: dict[str, dict[str, Any]] = {}
    with env.begin() as txn:
        for strain in picks:
            raw = txn.get(str(order.index(strain)).encode())
            if raw is None:
                raise KeyError(f"Caudal store has no record for {strain}")
            genotype = pickle.loads(raw)["experiment"]["genotype"]
            if genotype_isolate(genotype) != strain:
                raise ValueError(
                    f"Caudal record {order.index(strain)} is "
                    f"{genotype_isolate(genotype)}, not {strain}"
                )
            out[strain] = genotype
    env.close()
    return out


def _sgd_genes(data_root: str) -> set[str]:
    """The S288C systematic-name universe the yeast RNA-seq L4 uses."""
    from torchcell.verification.runners import _sgd_gene_set

    return _sgd_gene_set(data_root)


def sgd_gene_and_standard_names(data_root: str) -> set[str]:
    """Every S288C systematic AND standard name in the SGD ORF + RNA FASTA headers.

    SI Table 13 keys its quantities by the release's ``Genes`` column, mostly standard
    names (``AAC1``) and the systematic name where a gene has none (``YBL029C-A``); a
    header reads ``>YAL069W YAL069W SGDID:...``, systematic then standard (or the
    systematic again).
    """
    from torchcell.sequence.genome.registry import resolve
    from torchcell.verification.runners import SGD_GENE_FASTAS

    names: set[str] = set()
    for fasta in SGD_GENE_FASTAS:
        with open(resolve(SGD_S288C_R64, fasta, data_root=data_root)) as handle:
            for line in handle:
                if line.startswith(">"):
                    names.update(line[1:].split()[:2])
    return names


def main() -> None:
    """Deposit the raw mirror (``--deposit-from``) or build/load both stores."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--deposit-from",
        help="library capture of the SI workbook (si/si3.xlsx) to deposit",
    )
    args = parser.parse_args()
    data_root = _env_data_root()
    if args.deposit_from:
        print(deposit_raw_mirror(args.deposit_from, data_root))
        return
    proteome = ProteomeMuenzner2024Dataset(
        root=osp.join(data_root, "data/torchcell/proteome_muenzner2024")
    )
    turnover = ProteinTurnoverMuenzner2024Dataset(
        root=osp.join(data_root, "data/torchcell/protein_turnover_muenzner2024")
    )
    timecourse = ProteomeTimecourseMuenzner2024Dataset(
        root=osp.join(data_root, "data/torchcell/proteome_timecourse_muenzner2024")
    )
    print(
        f"proteome {len(proteome)} records, turnover {len(turnover)} records, "
        f"time course {len(timecourse)} records"
    )


if __name__ == "__main__":
    main()
