# torchcell/datasets/ecoli/wetmore2015
# [[torchcell.datasets.ecoli.wetmore2015]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/wetmore2015
# Test file: tests/torchcell/datasets/ecoli/test_wetmore2015.py
"""Wetmore 2015 RB-TnSeq, E. coli BW25113 (KEIO_ML9): the paper's own release, loaded.

Wetmore et al. 2015 (mBio 6:e00306-15, doi:10.1128/mBio.00306-15, citation key
``wetmoreRapidQuantificationMutant2015``) is the RB-TnSeq method paper, row 2 of the
fifty bacterial rows. Its E. coli arm is the BW25113 Tn5 library KEIO_ML9, grown in 101
condition samples of which 92 passed the paper's quality rules (Data Set S1), released by
the authors at statistics version 1.0.0.

DECISION (owner policy of 2026-10-10): LOADED FROM ITS OWN RELEASE as
:class:`RbTnseqWetmore2015Dataset`. A served compendium is a processing reference and a
sanity check, not the source of record: an original study with its own per-record
release is loaded from that release, attributed to itself, and cross-checked against the
compendium's copy. This supersedes the 2026-10-07 decision, "SUBSUMED, NOT LOADED"
(checklist item 7 of [[plan.bacteria-ontology-genome]]), which held that the Price 2018
compendium (rank 21, ``priceMutantPhenotypesThousands2018``) carries this paper's samples
re-analyzed at statistics version 1.0.3, so loading both would store each shared sample
twice. Both facts still hold and are still measured here, by the same functions; what
changed is the policy that reads them. The duplication is now explicit and joinable
(both datasets key a sample as ``Keio:<name>``), and the compendium is the L4 check of
this release rather than its replacement.

THE RECORDS. One record per (released gene, kept sample) of the release's all-sample gene
fitness table (``fit_logratios.tab``; for the successful samples it is byte-identical to
``fit_logratios_good.tab``, which the build checks), in
``EnvironmentResponseExperiment`` as an ``EnvironmentResponsePhenotype``:

* the value is the normalized gene fitness, a signed log2 ratio (``GENE_FITNESS_IS_LOG2``);
* the uncertainty is the per-gene standard error AS THE PAPER DEFINES IT in its moderated
  t, ``t = f / sqrt(sigma^2 + max(V_e, V_n))`` with sigma 0.1 (``T_STATISTIC``,
  ``T_STATISTIC_SIGMA``, ``PER_GENE_STANDARD_ERROR``): the larger of the release's
  strain-consistency SE (``fit_standard_error_obs.tab``, the R image's ``se``) and its
  read-count SE (``fit_standard_error_naive.tab``, ``sdNaive``). The build checks the
  released t against it for every stored value;
* the released moderated t itself is stored in ``environment_response_t_statistic``;
* genotype: one ``TransposonInsertionPerturbation`` per gene (``transposon`` = Table 1's
  "Tn5 transpososome", ``library_pool`` = Table S1's KEIO_ML9), on BW25113 as Table S1
  states it ("wild-type strain"), so the reference is ``assembly_reference("BW25113")``
  with no background beyond the assembly. Since #1057 (the owner rule of 2026-10-12:
  the stored key is the released identifier) every gene is stored under the release's
  ``sysName``, an MG1655 b-number, in the MG1655 namespace, named by ``fit_genes.tab``'s
  own ``name`` (else the b-number); no annotation is read, and the ECK crosswalk to
  BW25113 tags (:func:`map_genes`) is a derived artifact of the ``report`` CLI;
* environment: Data Set S1's own columns. ``Media`` through the media library
  (``M9_NOCARBON_WETMORE2015``, ``M9_NONITROGEN_WETMORE2015``, ``LB_LENNOX``, each
  quoting Data Set S1's Media tab, and, since #1045 row 16, Price 2018 Table S18's
  ``MOPS_RICH_DEFINED_NOCARBON_PRICE2018`` for the 4 samples on ``MOPS Rich Defined
  media_noCarbon``: Data Set S1's own row for that label agrees on every macro row and
  prints its six micronutrients in mM at 40,000 to 3,000,000, so the same lab's later
  table, which re-releases these same 4 samples, is the recipe and Data Set S1's row is
  quoted beside it), ``Condition_1`` / ``Concentration_1`` / ``Units_1``
  as a carbon- or nitrogen-source ``EnvironmentPhysicalPerturbation``, ``Temperature
  (°C)``, and ``Total Generations`` as ``duration_generations``;
* reference: the typical gene of the same sample, response 0 by the normalization.

Not stored, with its typed reason in ``preprocess/dropped_records.json``: only the 4
sucrose and D-mannitol samples the authors themselves withdrew in 2021
(:data:`AUTHOR_WITHDRAWAL`, the rule the compendium loader refuses the same four samples
under; an author withdrawal is the one drop, and the refusal carries the authors'
instruction and its cause verbatim with their sha256). The 9 condition samples Data Set
S1 scores ``u = FALSE`` (the paper's own quality rules) were dropped until #1057; they
are stored from ``fit_logratios.tab`` with their released SE and t, and their quality
verdict as released (``u``, every metric, the rules each fails) is in
``preprocess/samples.json`` and ``dropped_records.json`` under ``quality_flagged``,
keyed by the records' own ``screen_id``: ``EnvironmentResponsePhenotype`` has no field
for a sample's QC verdict. Two of them are ``stress`` rows that name no compound
(``short`` ``LB``), so their environment is the released LB with no added
perturbation. The genes with no one-to-one ECK pair were dropped too until #1057. The 4 samples on ``MOPS Rich Defined
media_noCarbon`` were refused under ``medium_not_in_media_library`` until #1045 row 16
gave the label a library entry; they are stored now.

WHAT THIS MODULE STILL PROVIDES TO OTHERS. The RB-TnSeq statistics as ``SourcedValue``
constants (every RB-TnSeq row defers to this paper), the record of which Price 2018
samples this paper first reported (:func:`subsumption_record`, read by the Price 2018
loader for its attribution and its withdrawn samples), the typed author withdrawal both
RB-TnSeq loaders refuse on (:data:`AUTHOR_WITHDRAWAL`), the release-to-release value
comparison (:func:`release_comparison`), the identifier finding
(:func:`identifier_reconciliation`), and the raw mirror (:func:`deposit_raw_mirror`).
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import os
import os.path as osp
import shutil
import statistics
import warnings
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Final, Literal

import numpy as np
import numpy.typing as npt
import openpyxl
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    Organism,
    check_manifest_pin,
    link_verified,
    post_process,
    verify_raw_files,
    verify_sha256,
    write_verified,
)
from torchcell.datamodels.compound_identity import (
    resolve_compound_identity,
    resolved_compound,
)
from torchcell.datamodels.media import (
    LB_LENNOX,
    M9_NOCARBON_WETMORE2015,
    M9_NONITROGEN_WETMORE2015,
    MOPS_RICH_DEFINED_NOCARBON_PRICE2018,
)
from torchcell.datamodels.schema import (
    AssayType,
    BacterialGeneNamespace,
    Concentration,
    ConcentrationUnit,
    Environment,
    EnvironmentPerturbationType,
    EnvironmentPhysicalPerturbation,
    EnvironmentResponseExperiment,
    EnvironmentResponseExperimentReference,
    EnvironmentResponsePhenotype,
    Experiment,
    ExperimentReference,
    Genotype,
    MeasurementType,
    Media,
    PhysicalFactor,
    Publication,
    ReferenceGenome,
    Temperature,
    TransposonInsertionPerturbation,
    UncertaintyType,
)
from torchcell.datasets.bacteria_common import (
    LOCUS_TAG_PATTERNS,
    STRAIN_GENE_NAMESPACES,
    EckCrosswalk,
    LocusTagResolutionError,
    assembly_reference,
    bacterial_genome,
    eck_crosswalk,
    reconcile_locus_tags,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.literature.manifest import (
    ROLE_RAW_DATA,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
    sha256_file,
)
from torchcell.literature.retrieve import direct_url
from torchcell.sequence.genome.base import GeneNameResolution, GeneNameStatus
from torchcell.sequence.genome.ecoli.k12 import (
    EcoliK12BW25113Genome,
    EcoliK12Genome,
    EcoliK12MG1655Genome,
    EcoliK12StrainName,
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

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Provenance anchors
# --------------------------------------------------------------------------- #
CITATION_KEY = "wetmoreRapidQuantificationMutant2015"
PAPER_DOI = "10.1128/mBio.00306-15"
PAPER_TITLE = (
    "Rapid Quantification of Mutant Fitness in Diverse Bacteria by Sequencing "
    "Randomly Bar-Coded Transposons"
)
LIBRARY_DIR_REL = "torchcell-library"
RAW_ROOT_REL = "torchcell-raw"
RAW_DIR_REL = f"{RAW_ROOT_REL}/{CITATION_KEY}"

PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "ca3e7ef27a22a2a28e52ccbe5fdbb60890fea93d03b18b46e1d2b77b033b3cdb"
#: Data Set S1: per-sample metadata (``Expts_Keio``) and quality (``Expt_Quality_Keio``).
DATA_SET_S1 = "si/si1.xlsx"
DATA_SET_S1_SHA256 = "428a06cae37867c8d21541f64d80da082e67743e1b1def9238a14e7726bc7150"
DATA_SET_S1_EXPERIMENTS_SHEET = "Expts_Keio"
DATA_SET_S1_QUALITY_SHEET = "Expt_Quality_Keio"
#: Table S1 (strains), OCR of ``si/si7.pdf``.
TABLE_S1_MD = "si/si7.md"
TABLE_S1_MD_SHA256 = "f10134f0d2ecde8f5c7043c514cbd043240fb33e6866f1c4efad0fff9a18a279"

#: The strain the E. coli library was built in (``STRAIN``); the compendium loader pins
#: ``assembly_reference(REFERENCE_STRAIN)``.
REFERENCE_STRAIN: Final = "BW25113"

SUPERSET_CITATION_KEY = "priceMutantPhenotypesThousands2018"
SUPERSET_PAPER_MD_SHA256 = (
    "f3443cdcb2f722b5e6aa6d999f67d68a6f845eb9eb45ad0bac1cc4c24ea53e2d"
)
SUPERSET_TABLE_S5 = "si/si3.xlsx"
SUPERSET_TABLE_S5_SHA256 = (
    "e5dbf3d5c97cfc12f49d7fd83f84bc95c16cbe963309a561fff20f442788b879"
)
SUPERSET_TABLE_S5_SHEET = "TableS5_Experiments"
#: The compendium's organism id for E. coli BW25113 (Table S5 ``orgId``).
SUPERSET_ORG_ID = "Keio"

# --------------------------------------------------------------------------- #
# Raw mirror: the authors' release files the evidence reads
# --------------------------------------------------------------------------- #
#: When the files the 2026-10-07 provenance record reads were retrieved.
RAW_RETRIEVED_AT = "2026-10-07"
#: When the files the loader adds (the t and standard-error tables, the all-sample
#: fitness table) were retrieved, by the same recorded ``direct_url`` route.
LOADER_RETRIEVED_AT = "2026-10-10"
RELEASE_PAGE = "data/rbarseq/html/Keio/index.html"
RELEASE_FITNESS = "data/rbarseq/html/Keio/fit_logratios_good.tab"
RELEASE_GENES = "data/rbarseq/html/Keio/fit_genes.tab"
RELEASE_FITNESS_ALL = "data/rbarseq/html/Keio/fit_logratios.tab"
RELEASE_T = "data/rbarseq/html/Keio/fit_t.tab"
RELEASE_SE_OBS = "data/rbarseq/html/Keio/fit_standard_error_obs.tab"
RELEASE_SE_NAIVE = "data/rbarseq/html/Keio/fit_standard_error_naive.tab"
SUPERSET_TOP_PAGE = "data/bigfit/index.html"
SUPERSET_PAGE = "data/bigfit/html/Keio/index.html"
SUPERSET_QUALITY = "data/bigfit/html/Keio/fit_quality.tab"
SUPERSET_FITNESS = "data/bigfit/html/Keio/fit_logratios_good.tab"


class RawFile(BaseModel):
    """One release file of the raw mirror: where it came from and why it is kept."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    relpath: str
    url: str
    sha256: str
    purpose: str
    retrieved_at: str


_RBARSEQ = "https://genomics.lbl.gov/supplemental/rbarseq"
_BIGFIT = "https://genomics.lbl.gov/supplemental/bigfit"

RAW_FILES: dict[str, RawFile] = {
    raw.relpath: raw
    for raw in (
        RawFile(
            relpath=RELEASE_PAGE,
            url=f"{_RBARSEQ}/html/Keio/",
            sha256="9085883893edd0d5216327fd003ea68fd09f98c92d4d0b0300e3fd629f3bc875",
            purpose="this paper's Keio release page: statistics version, the per-gene "
            "table definitions and the quality rules",
            retrieved_at=RAW_RETRIEVED_AT,
        ),
        RawFile(
            relpath=RELEASE_FITNESS,
            url=f"{_RBARSEQ}/html/Keio/fit_logratios_good.tab",
            sha256="f65a093b7fba660f12a894200073e3d731cd0b8a3f313205a5760584f4e55935",
            purpose="this paper's gene fitness for its successful samples (1.0.0)",
            retrieved_at=RAW_RETRIEVED_AT,
        ),
        RawFile(
            relpath=RELEASE_GENES,
            url=f"{_RBARSEQ}/html/Keio/fit_genes.tab",
            sha256="3eae36f697a32008673a03f6bcbca0d1b0581aa980175264c55e698a93a9ef73",
            purpose="the gene table both releases key on: b-number, symbol, scaffold",
            retrieved_at=RAW_RETRIEVED_AT,
        ),
        RawFile(
            relpath=SUPERSET_TOP_PAGE,
            url=f"{_BIGFIT}/",
            sha256="c4819e452ab54dfd3748242a3d4cf0f04021071aefadb2f07f2806afa32e3555",
            purpose="the compendium page carrying the 2021 sucrose / D-mannitol note",
            retrieved_at=RAW_RETRIEVED_AT,
        ),
        RawFile(
            relpath=SUPERSET_PAGE,
            url=f"{_BIGFIT}/html/Keio/",
            sha256="f05323ffce7701df45ddac9aebc2bab3838b111899b584838d90af0deee23803",
            purpose="the compendium's Keio release page: statistics version 1.0.3",
            retrieved_at=RAW_RETRIEVED_AT,
        ),
        RawFile(
            relpath=SUPERSET_QUALITY,
            url=f"{_BIGFIT}/html/Keio/fit_quality.tab",
            sha256="7d4b15733d8e258628a8f71831867c2d72294b82a0afca7ad53a6a9a603336e5",
            purpose="the compendium's per-sample quality verdicts",
            retrieved_at=RAW_RETRIEVED_AT,
        ),
        RawFile(
            relpath=SUPERSET_FITNESS,
            url=f"{_BIGFIT}/html/Keio/fit_logratios_good.tab",
            sha256="b37f038702ef0b792bdc02af9f42adf3a5838bc08513249c89983f6a7647e8fd",
            purpose="the compendium's gene fitness for its successful samples (1.0.3)",
            retrieved_at=RAW_RETRIEVED_AT,
        ),
        RawFile(
            relpath=RELEASE_T,
            url=f"{_RBARSEQ}/html/Keio/fit_t.tab",
            sha256="fcebd1de3403898bc76bc499952bde61875d75f419477abf5d18cd1148b08dd0",
            purpose="the released moderated t of every sample (1.0.0): stored on each "
            "record, and checked against fit / sqrt(0.1^2 + SE^2) at build time",
            retrieved_at=LOADER_RETRIEVED_AT,
        ),
        RawFile(
            relpath=RELEASE_SE_OBS,
            url=f"{_RBARSEQ}/html/Keio/fit_standard_error_obs.tab",
            sha256="06cc6122f7a3551d948084e0b600d4f185936720f026ba0dba152b6f57828d3f",
            purpose="the strain-consistency standard error (the R image's se, sqrt of "
            "the paper's V_e), one of the two the stored SE is the larger of",
            retrieved_at=LOADER_RETRIEVED_AT,
        ),
        RawFile(
            relpath=RELEASE_SE_NAIVE,
            url=f"{_RBARSEQ}/html/Keio/fit_standard_error_naive.tab",
            sha256="6b2c70e805c337f1ec1353e3e2d30e7261e5b29300266ef9c094a37e4ea9489b",
            purpose="the read-count standard error (the R image's sdNaive, sqrt of the "
            "paper's V_n), the other of the two",
            retrieved_at=LOADER_RETRIEVED_AT,
        ),
        RawFile(
            relpath=RELEASE_FITNESS_ALL,
            url=f"{_RBARSEQ}/html/Keio/fit_logratios.tab",
            sha256="adf240fcb677193cc9a14ea59ea0e86bfdf8ddb9345e7cf55c99f3df3cc15102",
            purpose="gene fitness of ALL samples (1.0.0), the failed ones included: the "
            "release-level comparison reads set2IT045, which failed this paper's cor12 "
            "rule and which the compendium carries",
            retrieved_at=LOADER_RETRIEVED_AT,
        ),
    )
}


# --------------------------------------------------------------------------- #
# Sourced values
# --------------------------------------------------------------------------- #
_OCR_METHOD = "MinerU OCR of the publisher PDF (torchcell-library mirror)"
_RELEASE_METHOD = "the authors' data-release page, retrieved verbatim (torchcell-raw)"
_BARSEQ_ANALYSIS = (
    "Materials and Methods, 'BarSeq data analysis and calculation of gene fitness'"
)


def _paper(
    value: Any, quote: str, *, page: str, note: str | None = None
) -> SourcedValue:
    """Bind a value to a verbatim quote of the pinned Wetmore 2015 OCR."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=PAPER_MD,
            citation_key=CITATION_KEY,
            sha256=PAPER_MD_SHA256,
            method=_OCR_METHOD,
            page=page,
        ),
    )


def _table_s1(value: Any, quote: str, *, note: str | None = None) -> SourcedValue:
    """Bind a value to a verbatim quote of the pinned Table S1 (strains) OCR."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=TABLE_S1_MD,
            citation_key=CITATION_KEY,
            sha256=TABLE_S1_MD_SHA256,
            method=_OCR_METHOD,
            page="Table S1, 'Strains used in this study'",
        ),
    )


def _superset_paper(
    value: Any, quote: str, *, page: str, note: str | None = None
) -> SourcedValue:
    """Bind a value to a verbatim quote of the pinned Price 2018 OCR."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=PAPER_MD,
            citation_key=SUPERSET_CITATION_KEY,
            sha256=SUPERSET_PAPER_MD_SHA256,
            method=_OCR_METHOD,
            page=page,
        ),
    )


def _release(
    value: Any, quote: str, *, relpath: str, page: str, note: str | None = None
) -> SourcedValue:
    """Bind a value to a verbatim quote of a release page in this key's raw mirror.

    The source path resolves under ``$DATA_ROOT/torchcell-raw`` (not the literature
    mirror), so audit these with that root.
    """
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=relpath,
            citation_key=CITATION_KEY,
            sha256=RAW_FILES[relpath].sha256,
            method=_RELEASE_METHOD,
            page=page,
            retrieved=RAW_FILES[relpath].retrieved_at,
        ),
    )


# The strain and the library ------------------------------------------------- #
STRAIN = _paper(
    REFERENCE_STRAIN,
    "the model bacterium Escherichia coli BW25113 (a K-12 strain; parent strain of the "
    "Keio deletion collection [20])",
    page="Results, 'Generation of complex mutant populations' (paper.md line 27)",
    note="BW25113, not MG1655: the compendium loader pins assembly_reference('BW25113')",
)
STRAIN_SOURCE = _paper(
    "Coli Genetic Stock Center",
    "Escherichia coli strain BW25113 was purchased from the Coli Genetic Stock Center.",
    page="Materials and Methods, 'Strains and standard growth conditions'",
)
STRAIN_TABLE_S1 = _table_s1(
    REFERENCE_STRAIN,
    "<td rowspan=1 colspan=1>Escherichia coli strain BW25113</td><td rowspan=1 "
    "colspan=1>wild-type strain</td>",
)
MUTANT_LIBRARY = _table_s1(
    "KEIO_ML9",
    "<td rowspan=1 colspan=1>KEIO_ML9</td><td rowspan=1 colspan=1>Escherichia coli "
    "strainBW25113 transposon mutantlibrary</td>",
    note="the OCR runs words together ('strainBW25113'); Data Set S1 spells the "
    "library 'Keio_ML9'",
)
TRANSPOSON = _paper(
    "Tn5 transpososome",
    "<td>Transposon</td><td>Tn5 transpososome</td>",
    page="Table 1 (first data column is Escherichia coli BW25113)",
)
N_BARCODED_STRAINS = _paper(
    152018,
    "<td>No. of strains with unique bar codesa</td><td>152,018</td>",
    page="Table 1 and footnote a ('Only strains with insertions in the genome')",
)
GENES_WITH_FITNESS = _paper(
    3471,
    "<td>No. with fitness estimates (% of total)</td><td>3,471 (84)</td>",
    page="Table 1, 'Protein-coding genes'",
    note="protein-coding genes only; the release's fitness table also carries 175 "
    "loci of other types (RNA genes, pseudogenes), 3,646 rows in all",
)

# Fitness and its statistics (the deferral target of the other RB-TnSeq rows) ---- #
STRAIN_FITNESS = _paper(
    "normalized log2(treatment count / time-zero count) per strain",
    "Roughly, strain fitness is the normalized $\\log _ { 2 }$ ratio of counts between "
    "the treatment sample (i.e., after growth in a certain medium) and the reference "
    "“time-zero” sample.",
    page=_BARSEQ_ANALYSIS,
)
GENE_FITNESS = _paper(
    "weighted average of the gene's strain fitness values",
    "Gene fitness is the weighted average of the strain fitness, and a t score is "
    "computed based on the consistency of the strain fitness values for each gene.",
    page=_BARSEQ_ANALYSIS,
)
GENE_FITNESS_WEIGHTS = _paper(
    "inverse strain variance, capped at the weight of a 20-read strain",
    "we impose a ceiling on the weight $w _ { i } ,$ with the maximum weight being "
    "that of a strain with 20 reads in each sample:",
    page="Materials and Methods, '(i) Gene fitness'",
)
STRAIN_INCLUSION = _paper(
    {
        "min_time_zero_reads_per_strain": 3,
        "min_time_zero_reads_per_gene": 30,
        "central_fraction_of_gene": (0.1, 0.9),
    },
    "In more detail, we first select a subset of strains and genes that have adequate "
    "coverage in the time-zero samples (3 reads per strain and 30 reads per gene, "
    "considering only the adequate strains). Only strains that lie within the central "
    "10 to $9 0 \\%$ of a gene are considered.",
    page="Materials and Methods, '(i) Gene fitness'",
)
GENE_FITNESS_IS_LOG2 = _release(
    MeasurementType.log2_ratio,
    "<P>Gene fitness is a log<SUB>2</SUB> ratio.",
    relpath=RELEASE_PAGE,
    page="Documentation, 'Gene Fitness'",
    note="signed and centered on zero. FitnessPhenotype.validate_fitness clamps a "
    "non-positive fitness to 0, so a loader storing this number there would erase "
    "every defect; the signed home is EnvironmentResponsePhenotype with "
    "measurement_type=log2_ratio, as the Hillenmeyer HIP/HOP loader does",
)
NORMALIZATION = _paper(
    "the typical gene is set to fitness zero",
    "The fitness data are normalized so that the typical gene has a fitness of zero "
    "(see Materials and Methods).",
    page="Results, 'Mutant fitness profiling by sequencing pools of random DNA bar codes'",
)
GENERATIONS = _paper(
    (4, 6),
    "More specifically, the fitness of a strain is the $\\log _ { 2 }$ change in "
    "abundance during growth (typically 4 to 6 generations); the fitness of a gene is "
    "roughly the average of the fitness of the strains that have insertions within "
    "that gene.",
    page="Results, 'Mutant fitness profiling by sequencing pools of random DNA bar codes'",
    note="the typical range only; Data Set S1 'Total Generations' gives the per-sample "
    "value where it was determined",
)
TIME_ZERO_POOLING = _paper(
    "summed across replicate time-zero samples",
    "We sum the per-strain counts across replicate time-zero samples.",
    page=_BARSEQ_ANALYSIS,
)
TIME_ZERO_REPLICATES = _paper(
    "independent gDNA extraction and PCR per time-zero replicate",
    "Also, we usually have multiple replicates of any given time zero, with "
    "independent extraction of genomic DNA and independent PCR with a different index.",
    page=_BARSEQ_ANALYSIS,
)
T_STATISTIC = _paper(
    "moderated t",
    "(iv) $\\pmb { t }$ -like test statistic. To estimate the reliability of the "
    "fitness measurement for each gene $f ,$ we use a moderated $t$ statistic:",
    page="Materials and Methods, '(iv) t-like test statistic'",
    note="t = f / sqrt(sigma^2 + max(V_e, V_n)); a significance statistic, not an "
    "uncertainty, so it is no UncertaintyType",
)
T_STATISTIC_SIGMA = _paper(
    0.1,
    "where $\\sigma$ is a small constant (we use 0.1) that represents uncertainty in "
    "the normalization for small fitness values, $V _ { e }$ represents the estimated "
    "variance, and $V _ { n }$ represents the naive variance.",
    page="Materials and Methods, '(iv) t-like test statistic'",
)
T_STATISTIC_N = _paper(
    "the number of strains of the gene",
    "where $n$ is the number of strains and $V _ { g }$ is a prior estimate of the "
    "variance in gene fitness.",
    page="Materials and Methods, '(iv) t-like test statistic'",
)
SIGNIFICANT_ABS_T = _paper(
    4.0,
    "Genes with $\\left| t \\right|$ of ${ > } 4$ have highly significant phenotypes "
    "that are largely reproducible in biological replicate experiments (Fig. 2A).",
    page="Results, 'Mutant fitness profiling by sequencing pools of random DNA bar codes'",
)
UNCERTAINTY_TYPE = _release(
    UncertaintyType.standard_error,
    "<LI><B>se</B> -- an estimate of how noisy this gene's measurement is (se is "
    "short for standard error)",
    relpath=RELEASE_PAGE,
    page="Documentation, 'The R image', per-gene information",
    note="already an SE, so derive_se uses it as-is and needs no n_samples",
)
STANDARD_ERROR_TABLE = _release(
    "fit_standard_error_obs.tab",
    '<LI>Or <A HREF="fit_standard_error_obs.tab">estimated standard error</A> (based '
    "on variation across strains)",
    relpath=RELEASE_PAGE,
    page="Tables, 'Genes'",
    note="the released column carrying UNCERTAINTY_TYPE; fit_standard_error_naive.tab "
    "is the best-case Poisson value and is not the reported uncertainty",
)
N_STRAINS_PER_GENE_FIELD = _release(
    "n",
    "<LI><B>n</B> -- number of usable strains for each gene",
    relpath=RELEASE_PAGE,
    page="Documentation, 'The R image', per-gene information",
    note="the strains averaged into one gene fitness value; released only inside the "
    "R image (fit.image), not in any tab-delimited table (N_SAMPLES_GAP)",
)
MEDIAN_STRAINS_PER_GENE = _paper(
    16,
    "<td>Median no. of strains per genec</td><td>16</td>",
    page="Table 1 and footnote c",
    note="the median over genes with a fitness estimate of the strains used for it; "
    "the per-gene count varies, so it is a description, never an n_samples value",
)
REPLICATE_DEFINITION = _release(
    "samples sharing a 'short' description are replicates",
    "<LI><B>short</B> -- shortened description. Samples with the same value are "
    "replicates.",
    relpath=RELEASE_PAGE,
    page="Documentation, 'The quality scores in the table of experiments'",
    note="each biological replicate is its own sample with its own gene fitness "
    "column, so a stored value is ONE sample and replication lies across records",
)
REPLICATE_DESIGN = _paper(
    2,
    "Among the 387 successful experiments, we studied 163 different bacterium-condition "
    "combinations, including 130 different bacterium-carbon source combinations, all "
    "but 5 with at least two biological replicates.",
    page="Results, 'Scalability of BarSeq with random bar codes'",
    note="the minimum replicate count for all but 5 of the 163 combinations across the "
    "five bacteria",
)
SUCCESS_RULES = _release(
    {
        "g_med_min": 50.0,
        "mad12_max": 0.5,
        "cor12_min": 0.1,
        "abs_gccor_max": 0.2,
        "abs_adjcor_max": 0.25,
    },
    "<LI>gMed &ge;50\n<LI>mad12 &le; 0.5\n<LI>cor12 &ge; 0.1\n<LI> |gccor| &le; 0.2 and "
    "|adjcor| &le; 0.25",
    relpath=RELEASE_PAGE,
    page="Documentation, 'As of April 9, 2014, the requirements for a successful "
    "experiment are'",
    note="plus 'not a Time0'; the paper's Methods (v) states the same five thresholds",
)
SUCCESSFUL_ASSAYS = _paper(
    387,
    "Overall, 387 of the 501 BarSeq assays that we performed met each of these "
    "metrics and were deemed successful.",
    page="Materials and Methods, '(v) Assessment of experiment quality'",
)
RELEASE_VERSION = _release(
    {"condition_samples": 101, "successful": 92, "statistics_version": "1.0.0"},
    "<P><small>101 condition samples (92 successful), Sat Sep 27 07:53:29 2014, "
    "statistics version 1.0.0</small>",
    relpath=RELEASE_PAGE,
    page="page header",
)
CODE_RELEASE = _paper(
    "1.0.0",
    "All analyses were performed with release 1.0.0.",
    page="Materials and Methods, 'Code availability'",
)
DATA_RELEASE = _paper(
    "http://genomics.lbl.gov/supplemental/rbarseq/",
    "Raw sequence data and processed fitness values are available from "
    "http://genomics.lbl.gov/supplemental/rbarseq/ along with scripts for reproducing "
    "all of our results.",
    page="Materials and Methods, 'Code availability'",
)

# The superset ----------------------------------------------------------------- #
SUPERSET_INCLUDES_THIS_PAPER = _superset_paper(
    385,
    "Our analysis includes 385 successful experiments from Wetmore et al.9 and 36 "
    "successful experiments from Melnyk et al.12.",
    page="Methods (paper.md line 177)",
    note="385 of this paper's 387 successes across five bacteria; for E. coli the "
    "count is measured sample by sample in subsumption_record",
)
SUPERSET_RELEASE = _release(
    {"condition_samples": 207, "successful": 162, "statistics_version": "1.0.3"},
    "<P><small>207 condition samples (162 successful), Fri Feb 19 10:53:17 2016, "
    "statistics version 1.0.3</small>",
    relpath=SUPERSET_PAGE,
    page="page header",
)
STOCK_SOLUTION_CORRECTION = _release(
    ("Sucrose", "D-Mannitol"),
    "Please disregard any of the data from this publication regarding sucrose or "
    "D-mannitol.",
    relpath=SUPERSET_TOP_PAGE,
    page="'Note added September 10, 2021'",
    note="the note names 'our original fitness assays for E. coli' on mannitol, where "
    "mtlA and mtlD were not important and manX and manY were. This paper's D-mannitol "
    "samples set1IT043/set1IT044 show exactly that in its own release, and its sucrose "
    "samples set1IT007/set1IT008 grew although BW25113 does not use sucrose",
)


class AuthorWithdrawal(BaseModel):
    """A source's own authors telling readers to disregard part of what they published.

    The typed refusal reason both RB-TnSeq loaders refuse on. Author-withdrawn data is
    bad data: the authors state the measured response is not a response to the named
    compound, so no record of it is stored, by either this paper's loader or the
    compendium's. ``rule`` is the drop-rule name the refusal is counted under in each
    loader's ``preprocess/dropped_records.json``; ``conditions`` are the Data Set S1 /
    Table S5 ``Condition_1`` labels the note covers; ``instruction`` and ``cause`` are
    the authors' own words, each bound to the sha256-pinned mirror file they were read
    from, so the refusal is audited the way every sourced value is.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    rule: str
    conditions: frozenset[str]
    instruction: SourcedValue = Field(
        description="the authors' instruction to disregard the data, verbatim"
    )
    cause: SourcedValue = Field(
        description="what the authors found was wrong, verbatim: why the data is bad"
    )


#: The 2021 stock-solution withdrawal, the one refusal reason that is the source's own.
AUTHOR_WITHDRAWAL = AuthorWithdrawal(
    rule="authors_withdrew_sucrose_and_mannitol_2021",
    conditions=frozenset(STOCK_SOLUTION_CORRECTION.value),
    instruction=STOCK_SOLUTION_CORRECTION,
    cause=_release(
        ("Sucrose", "D-Mannitol"),
        "After publishing this data, Morgan Price and Adam Deutschbauer discovered "
        "that our stock solutions for sucrose and D-mannitol were problematic.",
        relpath=SUPERSET_TOP_PAGE,
        page="'Note added September 10, 2021'",
        note="the stock solution was wrong, so the measured response is not a response "
        "to the named compound: the values are refused, not stored and flagged",
    ),
)

N_SAMPLES_GAP = ProvenanceGap(
    field="n_samples",
    reason=ProvenanceGapReason.deferred_pending_source_review,
    looked_in=Provenance(
        source_uri=RAW_FILES[RELEASE_FITNESS].url,
        citation_key=CITATION_KEY,
        sha256=RAW_FILES[RELEASE_FITNESS].sha256,
        method=_RELEASE_METHOD,
        page="the tab-delimited per-gene tables carry no strain count column",
    ),
    resolve_with=Provenance(
        source_uri=f"{_RBARSEQ}/html/Keio/fit.image",
        citation_key=CITATION_KEY,
        method="R image of the release, field fit$n",
        page="per-gene information, 'n -- number of usable strains for each gene'",
    ),
    note="a gene fitness value averages its usable strains within ONE sample. The "
    "count is per gene and per sample; Table 1 gives only its median "
    "(MEDIAN_STRAINS_PER_GENE). The released SE needs no n, so the gap does not block "
    "an SE",
)
SAMPLE_UNIT_GAP = ProvenanceGap(
    field="sample_unit",
    reason=ProvenanceGapReason.deferred_pending_source_review,
    looked_in=N_SAMPLES_GAP.looked_in,
    resolve_with=N_SAMPLES_GAP.resolve_with,
    note="travels with n_samples: one unit would be a usable insertion strain, for "
    "which SampleUnit has no member",
)


# The loader's statistics (owner policy 2026-10-10) ----------------------------- #
_XLSX_METHOD = (
    "row rendering of Data Set S1: a row's non-empty cells joined by ' | ', consecutive "
    "rows by ' / ' (xlsx_text)"
)


def _data_set_s1(
    value: Any, quote: str, *, page: str, note: str | None = None
) -> SourcedValue:
    """Bind a value to a row rendering of the pinned Data Set S1 workbook."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=DATA_SET_S1,
            citation_key=CITATION_KEY,
            sha256=DATA_SET_S1_SHA256,
            method=_XLSX_METHOD,
            page=page,
        ),
    )


PER_GENE_STANDARD_ERROR = _paper(
    "sqrt(max(V_e, V_n)): the larger of the release's se and sdNaive",
    "t = { \\frac { f } { \\sqrt { \\mathbf { { \\sigma } } \\mathbf { { \\sigma } } ^ "
    "{ 2 } + \\operatorname* { m a x } ( V _ { e } , V _ { n } ) } } }",
    page="Materials and Methods, '(iv) t-like test statistic' (paper.md line 177)",
    note="the paper's own uncertainty of a gene fitness value is the denominator of its "
    "moderated t less sigma: sqrt(max(V_e, V_n)). V_e is the estimated variance "
    "(ESTIMATED_VARIANCE, the release's se squared) and V_n the naive one "
    "(NAIVE_VARIANCE, sdNaive squared), so the stored SE is max(se, sdNaive) and the "
    "build checks t = fit / sqrt(0.1^2 + SE^2) on every stored value",
)
NAIVE_VARIANCE = _paper(
    "V_n, the best-case Poisson variance",
    "First, to prevent underestimates of the variance in some cases, we use an alternate "
    "\u201cnaive\u201d variance estimate, $V _ { n } ,$ based on the best-case Poisson noise:",
    page="Materials and Methods, '(iv) t-like test statistic'",
)
ESTIMATED_VARIANCE = _paper(
    "V_e, the strain-consistency variance",
    "Second, we look at the observed consistency of the measurements for the gene and "
    "take into account data from other genes, especially if there are few strains for "
    "this gene:",
    page="Materials and Methods, '(iv) t-like test statistic'",
)
SD_NAIVE_FIELD = _release(
    "sdNaive",
    "<LI><B>sdNaive</B> -- the best-case of how noisy this gene's measurement would be, "
    "based on the total counts",
    relpath=RELEASE_PAGE,
    page="Documentation, 'The R image', per-gene information",
    note="sqrt(V_n): the read-count SE, fit_standard_error_naive.tab",
)
SE_NAIVE_TABLE = _release(
    "fit_standard_error_naive.tab",
    '<LI>Or <A HREF="fit_standard_error_naive.tab">naive standard error</A> (based on '
    "total counts)",
    relpath=RELEASE_PAGE,
    page="Tables, 'Genes'",
)
T_FIELD = _release(
    "t",
    "<LI><B>t</B> -- a t-like test statistic that gives an estimate of how significant "
    "this gene's measurement is",
    relpath=RELEASE_PAGE,
    page="Documentation, 'The R image', per-gene information",
)
T_TABLE = _release(
    "fit_t.tab",
    '<LI><A HREF="fit_t.tab">t-like test statistic</A> (based on consistency of '
    "measurements for the gene)",
    relpath=RELEASE_PAGE,
    page="Tables, 'Genes'",
    note="stored per record in environment_response_t_statistic",
)
GOOD_FITNESS_TABLE = _release(
    "fit_logratios_good.tab",
    '<LI><A HREF="fit_logratios_good.tab">Gene fitness</A> (normalized log2 ratios for '
    "successful experiments )",
    relpath=RELEASE_PAGE,
    page="Tables, 'Genes'",
    note="the stored values: one column per successful sample",
)
ALL_FITNESS_TABLE = _release(
    "fit_logratios.tab",
    '<LI>or for <A HREF="fit_logratios.tab">all</A> experiments</LI>',
    relpath=RELEASE_PAGE,
    page="Tables, 'Genes'",
    note="the failed samples' values; not stored, read by the release comparison",
)
SUCCESS_FIELD = _release(
    "u",
    "<LI><B>u</B> -- Whether the experiment was successful and is used in global "
    "analyses.",
    relpath=RELEASE_PAGE,
    page="Documentation, 'The quality scores in the table of experiments'",
    note="Data Set S1 Expt_Quality_Keio carries the same column; the paper's five rules "
    "(SUCCESS_RULES) reproduce it for all 101 condition samples",
)
SATURATED_GROWTH = _paper(
    "grown to saturation, no fixed duration",
    "In general, all of the mutant library assays reached saturated growth",
    page="Materials and Methods, 'Competitive mutant fitness assays'",
    note="no assay duration in hours is stated, so duration_hours is a typed gap",
)
TOTAL_GENERATIONS_COLUMN = _data_set_s1(
    "duration_generations",
    "Total Generations | Total number of population doublings of experiment (if "
    "determined).",
    page="sheet ReadMe, the Expts tab's column definitions",
    note="stored as Environment.duration_generations; blank for set2IT069, which then "
    "carries a typed gap",
)
CONDITION_COLUMN = _data_set_s1(
    "Condition_1",
    "Condition_1 | If applicable, the compound added to the experiment (beyond what is "
    "described in the base media).",
    page="sheet ReadMe, the Expts tab's column definitions",
    note="the carbon or nitrogen source of a defined-medium sample, stored as the agent "
    "of EnvironmentPhysicalPerturbation with Concentration_1 / Units_1 as its dose",
)
VALUES_IN_THE_RELEASE = _data_set_s1(
    "http://genomics.lbl.gov/supplemental/rbarseq/",
    "Fitness_Data and t_values data; available  at "
    "http://genomics.lbl.gov/supplemental/rbarseq/",
    page="sheet ReadMe",
    note="Data Set S1 defers the per-gene values and t to the release, which is why the "
    "values come from the deposited release tables and the sample metadata from S1",
)

# The compendium, as the sanity check ------------------------------------------- #
SUPERSET_REANALYSIS = _superset_paper(
    CITATION_KEY,
    "Fitness data was analysed as previously described9 .",
    page="Methods, 'Computation of fitness values'",
    note="ref 9 is this paper: the compendium re-ran this paper's method on the same "
    "reads, so its values for a shared sample are a second analysis of one sample",
)
SUPERSET_STATISTICS_VERSIONS = _superset_paper(
    ("1.0.3", "1.1.0", "1.1.1"),
    "we used statistics versions 1.0.3, 1.1.0, or 1.1.1 of the code",
    page="Methods, 'Data and code availability'",
    note="Keio is 1.0.3 (SUPERSET_RELEASE) against this release's 1.0.0 (CODE_RELEASE)",
)
SUPERSET_REPLICATE_AGREEMENT = _superset_paper(
    0.87,
    "The remaining experiments showed good agreement between exact replicates, with a "
    "median correlation of 0.87 for gene fitness values from defined media experiments. "
    "Ninety-five per cent of the replicate defined media experiments had $r \\geq 0 . 6 "
    "4$ .",
    page="Methods, 'Computation of fitness values'",
    note="the cross-check's tolerance (COMPENDIUM_MIN_PEARSON_R): two analyses of the "
    "SAME sample's reads must agree at least as well as the compendium's median pair of "
    "INDEPENDENT exact replicates; 0.64 is the 95% floor for such a pair",
)


# --------------------------------------------------------------------------- #
# Paths and pins
# --------------------------------------------------------------------------- #
def _data_root(data_root: str | None) -> Path:
    """``data_root`` when given, else ``DATA_ROOT`` (the repo-root ``.env``)."""
    if data_root is not None:
        return Path(data_root)
    load_dotenv()
    return Path(os.environ["DATA_ROOT"])


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/wetmoreRapidQuantificationMutant2015``."""
    return _data_root(data_root) / RAW_DIR_REL


def library_dir(citation_key: str, data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-library/<citation_key>``."""
    return _data_root(data_root) / LIBRARY_DIR_REL / citation_key


def load_manifest(data_root: str | None = None) -> Manifest:
    """The raw mirror's ``manifest.json``."""
    path = raw_mirror_dir(data_root) / "manifest.json"
    return Manifest.model_validate_json(path.read_text())


def manifest_sha256(manifest: Manifest, relpath: str) -> str:
    """The sha256 a manifest records for one file."""
    for record in manifest.files:
        if record.path == relpath:
            return record.sha256
    raise KeyError(f"{relpath} is not in the {manifest.citation_key} manifest")


def raw_path(relpath: str, data_root: str | None = None) -> Path:
    """A raw-mirror file after checking the manifest pin and the bytes against it."""
    pin = RAW_FILES[relpath].sha256
    check_manifest_pin(relpath, manifest_sha256(load_manifest(data_root), relpath), pin)
    path = raw_mirror_dir(data_root) / relpath
    verify_sha256(path, pin)
    return path


def library_path(
    citation_key: str, relpath: str, pin: str, data_root: str | None = None
) -> Path:
    """A literature-mirror file after checking its manifest pin and its bytes."""
    root = library_dir(citation_key, data_root)
    manifest = Manifest.model_validate_json((root / "manifest.json").read_text())
    check_manifest_pin(relpath, manifest_sha256(manifest, relpath), pin)
    path = root / relpath
    verify_sha256(path, pin)
    return path


# --------------------------------------------------------------------------- #
# Retrieval and deposit
# --------------------------------------------------------------------------- #
def retrieve_raw_files(dest_dir: str | Path) -> Path:
    """Run the recorded retrieval for every raw file into ``dest_dir`` (sha256-checked).

    A changed upstream page raises ``RawSha256MismatchError`` and writes nothing for that
    file, so drift is detected rather than deposited.
    """
    dest = Path(dest_dir)
    for raw in RAW_FILES.values():
        target = dest / raw.relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        write_verified(direct_url(raw.url), target, raw.sha256, raw.url)
    return dest


def _raw_manifest(root: Path, created_at: str) -> Manifest:
    """The manifest the raw mirror carries, built from the deposited bytes."""
    return Manifest(
        citation_key=CITATION_KEY,
        doi=PAPER_DOI,
        title=PAPER_TITLE,
        files=[
            ArtifactRecord(
                path=raw.relpath,
                role=ROLE_RAW_DATA,
                bytes=(root / raw.relpath).stat().st_size,
                sha256=raw.sha256,
                source=raw.url,
                retrieval=RetrievalRecord(
                    method=RetrievalMethod.direct_url,
                    source_url=raw.url,
                    retriever="torchcell.literature.retrieve.direct_url",
                    params={"url": raw.url},
                    sha256=raw.sha256,
                    retrieved_at=raw.retrieved_at,
                ),
            )
            for raw in RAW_FILES.values()
        ],
        si_data_sources=[f"{_RBARSEQ}/", f"{_BIGFIT}/"],
        si_expected=[
            "the per-strain tables (strain_fit.tab, all.poolcount) and the R image "
            "fit.image of both releases are NOT mirrored: the loader and the provenance "
            "record read gene-level tables only (n, the usable strains per gene, is "
            "released only in fit.image and is a typed gap on every record)"
        ],
        provenance_complete=True,
        created_at=created_at,
    )


def deposit_raw_mirror(*, source_dir: str | Path, data_root: str | None = None) -> Path:
    """Copy the retrieved release files into the raw mirror and write its manifest.

    Every source file is checked against its pin, and every file already in the mirror
    against the same pin, BEFORE anything is written, so a refusal leaves the mirror as
    it was. Idempotent by sha256: a matching mirror file is left alone, a differing one
    raises. An existing manifest is kept when it records exactly these files, EXTENDED
    when every record it holds is identical to this module's record of that file (the
    loader's files were added to the 2026-10-07 provenance record's this way; the
    first deposit's ``created_at`` is kept), and refused when it records anything else.
    """
    source = Path(source_dir)
    root = raw_mirror_dir(data_root)
    for raw in RAW_FILES.values():
        got = sha256_file(source / raw.relpath)
        if got != raw.sha256:
            raise RuntimeError(
                f"{source / raw.relpath} sha256 mismatch: got {got}, expected "
                f"{raw.sha256}"
            )
    for raw in RAW_FILES.values():
        dest = root / raw.relpath
        if dest.exists() and sha256_file(dest) != raw.sha256:
            raise RuntimeError(f"{dest} exists with a different sha256; refusing")
    manifest_path = root / "manifest.json"
    existing = (
        Manifest.model_validate_json(manifest_path.read_text())
        if manifest_path.exists()
        else None
    )
    for raw in RAW_FILES.values():
        dest = root / raw.relpath
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / raw.relpath, dest)
    if existing is None:
        manifest = _raw_manifest(root, datetime.now(UTC).isoformat())
        manifest_path.write_text(manifest.model_dump_json(indent=2))
        return root
    if existing.created_at is None:
        raise RuntimeError(f"{manifest_path} records no created_at; refusing")
    manifest = _raw_manifest(root, existing.created_at)
    if existing.files == manifest.files:
        return root
    recorded = {record.path: record for record in manifest.files}
    if any(recorded.get(record.path) != record for record in existing.files):
        raise RuntimeError(
            f"{manifest_path} records other files or retrievals; refusing"
        )
    manifest_path.write_text(manifest.model_dump_json(indent=2))
    return root


# --------------------------------------------------------------------------- #
# The provenance record
# --------------------------------------------------------------------------- #
class QualityMetrics(BaseModel):
    """The quality metrics one release scored a sample with (Data Set S1 columns)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    successful: bool
    g_med: float
    mad12: float
    cor12: float
    adjcor: float
    gccor: float

    def failed_rules(self) -> tuple[str, ...]:
        """The SUCCESS_RULES this sample fails, by name."""
        rules = SUCCESS_RULES.value
        checks = {
            "g_med_min": self.g_med >= rules["g_med_min"],
            "mad12_max": self.mad12 <= rules["mad12_max"],
            "cor12_min": self.cor12 >= rules["cor12_min"],
            "abs_gccor_max": abs(self.gccor) <= rules["abs_gccor_max"],
            "abs_adjcor_max": abs(self.adjcor) <= rules["abs_adjcor_max"],
        }
        return tuple(name for name, ok in checks.items() if not ok)


class SourceExperiment(BaseModel):
    """One E. coli condition sample of Data Set S1 and whether the compendium has it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    short: str
    group: str
    media: str
    condition: str | None
    concentration: float | None
    units: str | None
    total_generations: float | None
    end_od: float | None
    source_quality: QualityMetrics
    in_superset: bool
    disregarded_by_authors: bool


class VerdictFlip(BaseModel):
    """A sample one release scored successful and the other did not."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    short: str
    source: QualityMetrics
    superset: QualityMetrics


class Wetmore2015Subsumption(BaseModel):
    """Which of this paper's E. coli samples the Price 2018 compendium carries.

    The record the compendium loader reads to say "first reported in Wetmore 2015" and
    to flag the samples the authors later withdrew; never a second load of the values.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    citation_key: str
    strain: str
    mutant_library: str
    superset_citation_key: str
    superset_org_id: str
    superset_experiment_names: tuple[str, ...]
    experiments: tuple[SourceExperiment, ...]
    verdict_flips: tuple[VerdictFlip, ...]

    @property
    def successful(self) -> tuple[str, ...]:
        """Samples this paper scored successful."""
        return tuple(e.name for e in self.experiments if e.source_quality.successful)

    @property
    def carried(self) -> tuple[str, ...]:
        """Samples of this paper the compendium carries (whatever this paper scored)."""
        return tuple(e.name for e in self.experiments if e.in_superset)

    @property
    def covered(self) -> tuple[str, ...]:
        """Successful samples the compendium carries."""
        return tuple(
            e.name
            for e in self.experiments
            if e.source_quality.successful and e.in_superset
        )

    @property
    def not_covered(self) -> tuple[str, ...]:
        """Successful samples the compendium does not carry."""
        return tuple(
            e.name
            for e in self.experiments
            if e.source_quality.successful and not e.in_superset
        )

    @property
    def disregarded(self) -> tuple[str, ...]:
        """Carried samples the 2021 stock-solution note withdraws."""
        return tuple(
            e.name
            for e in self.experiments
            if e.in_superset and e.disregarded_by_authors
        )

    @property
    def superset_only(self) -> tuple[str, ...]:
        """Compendium samples that are not this paper's."""
        own = {e.name for e in self.experiments}
        return tuple(n for n in self.superset_experiment_names if n not in own)


def _optional_str(value: Any) -> str | None:
    """A spreadsheet cell as text, or None when empty."""
    return None if pd.isna(value) else str(value)


def _optional_float(value: Any) -> float | None:
    """A spreadsheet cell as a float, or None when empty."""
    return None if pd.isna(value) else float(value)


def _row(table: pd.DataFrame, name: str) -> pd.Series:
    """The one row of a name-indexed table (``_unique_index`` guarantees one)."""
    row = table.loc[name]
    if not isinstance(row, pd.Series):
        raise ValueError(f"{name} is not exactly one row")
    return row


def _quality(table: pd.DataFrame, name: str) -> QualityMetrics:
    """One sample's row of a name-indexed quality table (Data Set S1 columns)."""
    row = _row(table, name)
    return QualityMetrics(
        successful=bool(row["u"]),
        g_med=float(row["gMed"]),
        mad12=float(row["mad12"]),
        cor12=float(row["cor12"]),
        adjcor=float(row["adjcor"]),
        gccor=float(row["gccor"]),
    )


def _unique_index(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """``frame`` indexed by ``name``, refusing a repeated sample name."""
    repeated = sorted(frame.loc[frame["name"].duplicated(), "name"])
    if repeated:
        raise ValueError(f"{label} repeats sample names: {repeated}")
    return frame.set_index("name")


#: The Table S5 columns that must agree with Data Set S1 for a shared sample name to
#: count as the same sample.
SHARED_METADATA = ("short", "Media", "Condition_1", "Concentration_1")


def _metadata(row: pd.Series) -> tuple[str, str, str | None, float | None]:
    """A sample's ``SHARED_METADATA`` values, empty cells as None."""
    return (
        str(row["short"]),
        str(row["Media"]),
        _optional_str(row["Condition_1"]),
        _optional_float(row["Concentration_1"]),
    )


def build_subsumption(
    experiments: pd.DataFrame,
    quality: pd.DataFrame,
    superset_experiments: pd.DataFrame,
    superset_quality: pd.DataFrame,
) -> Wetmore2015Subsumption:
    """Build the record from Data Set S1's two Keio sheets and the compendium's tables.

    ``experiments`` and ``quality`` are the ``Expts_Keio`` and ``Expt_Quality_Keio``
    sheets; time-zero rows are references, not condition samples, and are left out.
    ``superset_experiments`` are the compendium's successful samples (the Table S5
    ``Keio`` rows) and ``superset_quality`` its ``fit_quality.tab``. Refused: a
    condition sample with no quality row, a repeated name, a shared name whose
    ``SHARED_METADATA`` differ between the two papers (then it is not the same sample),
    and a verdict flip the compendium scored no row for.
    """
    condition_rows = experiments.loc[experiments["Group"] != "Time0"]
    samples = _unique_index(condition_rows, "Expts_Keio")
    scored = _unique_index(quality, "Expt_Quality_Keio")
    superset = _unique_index(superset_experiments, SUPERSET_TABLE_S5_SHEET)
    superset_scored = _unique_index(superset_quality, "fit_quality.tab")
    unscored = sorted(set(samples.index) - set(scored.index))
    if unscored:
        raise ValueError(f"condition samples with no quality row: {unscored}")
    carried = {str(name) for name in superset.index}
    differing = sorted(
        str(name)
        for name, row in samples.iterrows()
        if str(name) in carried
        and _metadata(row) != _metadata(_row(superset, str(name)))
    )
    if differing:
        raise ValueError(f"shared names whose {SHARED_METADATA} differ: {differing}")

    records: list[SourceExperiment] = []
    flips: list[VerdictFlip] = []
    for index, row in samples.iterrows():
        name = str(index)
        source_quality = _quality(scored, name)
        in_superset = name in carried
        condition = _optional_str(row["Condition_1"])
        records.append(
            SourceExperiment(
                name=name,
                short=str(row["short"]),
                group=str(row["Group"]),
                media=str(row["Media"]),
                condition=condition,
                concentration=_optional_float(row["Concentration_1"]),
                units=_optional_str(row["Units_1"]),
                total_generations=_optional_float(row["Total Generations"]),
                end_od=_optional_float(row["EndOD"]),
                source_quality=source_quality,
                in_superset=in_superset,
                disregarded_by_authors=condition in AUTHOR_WITHDRAWAL.conditions,
            )
        )
        if source_quality.successful != in_superset:
            if name not in superset_scored.index:
                raise ValueError(f"{name}: the compendium release scored no row")
            flips.append(
                VerdictFlip(
                    name=name,
                    short=str(row["short"]),
                    source=source_quality,
                    superset=_quality(superset_scored, name),
                )
            )
    return Wetmore2015Subsumption(
        citation_key=CITATION_KEY,
        strain=STRAIN.value,
        mutant_library=MUTANT_LIBRARY.value,
        superset_citation_key=SUPERSET_CITATION_KEY,
        superset_org_id=SUPERSET_ORG_ID,
        superset_experiment_names=tuple(sorted(carried)),
        experiments=tuple(records),
        verdict_flips=tuple(flips),
    )


def read_superset_experiments(table_s5: str | Path) -> pd.DataFrame:
    """The ``Keio`` rows of Price 2018 Supplementary Table 5 (below its preamble)."""
    sheet = pd.read_excel(table_s5, sheet_name=SUPERSET_TABLE_S5_SHEET, header=None)
    header_rows = sheet.index[sheet[0] == "orgId"]
    if len(header_rows) != 1:
        raise ValueError(f"{SUPERSET_TABLE_S5_SHEET}: expected one 'orgId' header row")
    start = int(header_rows[0])
    table = sheet.iloc[start + 1 :].copy()
    table.columns = sheet.iloc[start].tolist()
    keio = table.loc[table["orgId"] == SUPERSET_ORG_ID].reset_index(drop=True)
    keio["name"] = keio["name"].astype(str)
    return keio


def subsumption_record(data_root: str | None = None) -> Wetmore2015Subsumption:
    """The record built from the pinned Data Set S1, Table S5 and compendium quality."""
    s1 = library_path(CITATION_KEY, DATA_SET_S1, DATA_SET_S1_SHA256, data_root)
    s5 = library_path(
        SUPERSET_CITATION_KEY, SUPERSET_TABLE_S5, SUPERSET_TABLE_S5_SHA256, data_root
    )
    return build_subsumption(
        experiments=pd.read_excel(s1, sheet_name=DATA_SET_S1_EXPERIMENTS_SHEET),
        quality=pd.read_excel(s1, sheet_name=DATA_SET_S1_QUALITY_SHEET),
        superset_experiments=read_superset_experiments(s5),
        superset_quality=pd.read_csv(raw_path(SUPERSET_QUALITY, data_root), sep="\t"),
    )


# --------------------------------------------------------------------------- #
# Value-level evidence: the two releases of the shared samples
# --------------------------------------------------------------------------- #
class ExperimentAgreement(BaseModel):
    """How one shared sample's gene fitness agrees between the two releases."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    n_genes: int
    pearson_r: float
    max_abs_difference: float


class ReleaseComparison(BaseModel):
    """This paper's release (1.0.0) against the compendium's (1.0.3), gene by gene."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_source_genes: int
    n_superset_genes: int
    n_source_genes_in_superset: int
    agreements: tuple[ExperimentAgreement, ...]

    @property
    def min_pearson_r(self) -> float:
        """The weakest per-sample agreement."""
        return min(a.pearson_r for a in self.agreements)

    @property
    def median_pearson_r(self) -> float:
        """The median per-sample agreement."""
        return statistics.median(a.pearson_r for a in self.agreements)


def _fitness_columns(table: pd.DataFrame, label: str) -> dict[str, str]:
    """Sample name (the header's first token) to its column, refusing a repeat."""
    columns: dict[str, str] = {}
    for column in table.columns:
        name = str(column).split(" ")[0]
        if not name.startswith("set"):
            continue
        if name in columns:
            raise ValueError(f"{label}: sample {name} has two columns")
        columns[name] = str(column)
    return columns


def compare_fitness_tables(
    source: pd.DataFrame, superset: pd.DataFrame
) -> ReleaseComparison:
    """Per-sample Pearson r and largest gene difference over the genes both carry."""
    source_columns = _fitness_columns(source, "source")
    superset_columns = _fitness_columns(superset, "superset")
    shared_genes = sorted(set(source["locusId"]) & set(superset["locusId"]))
    left = source.set_index("locusId").loc[shared_genes]
    right = superset.set_index("locusId").loc[shared_genes]
    agreements = []
    for name in sorted(set(source_columns) & set(superset_columns)):
        x = left[source_columns[name]].to_numpy(dtype=float)
        y = right[superset_columns[name]].to_numpy(dtype=float)
        agreements.append(
            ExperimentAgreement(
                name=name,
                n_genes=len(shared_genes),
                pearson_r=float(np.corrcoef(x, y)[0, 1]),
                max_abs_difference=float(np.max(np.abs(x - y))),
            )
        )
    return ReleaseComparison(
        n_source_genes=int(source["locusId"].nunique()),
        n_superset_genes=int(superset["locusId"].nunique()),
        n_source_genes_in_superset=len(shared_genes),
        agreements=tuple(agreements),
    )


def release_comparison(data_root: str | None = None) -> ReleaseComparison:
    """:func:`compare_fitness_tables` on the two deposited ``fit_logratios_good.tab``."""
    return compare_fitness_tables(
        pd.read_csv(raw_path(RELEASE_FITNESS, data_root), sep="\t"),
        pd.read_csv(raw_path(SUPERSET_FITNESS, data_root), sep="\t"),
    )


# --------------------------------------------------------------------------- #
# Identifier finding for the compendium loader
# --------------------------------------------------------------------------- #
class Reconciliation(BaseModel):
    """One ``reconcile_locus_tags`` run, reduced to its histograms."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    unique_names: int
    status_histogram: dict[str, int]
    layer_histogram: dict[str, int]
    resolved_fraction: float


class IdentifierFinding(BaseModel):
    """How the release's gene table maps onto the two K-12 assemblies."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_genes: int
    scaffold_ids: tuple[int, ...]
    b_number_sysnames: int
    mg1655_by_sysname: Reconciliation
    bw25113_by_sysname: Reconciliation
    bw25113_by_symbol: Reconciliation
    eck_one_to_one: int
    eck_numeric_disagreements: int


def _reconcile(genome: EcoliK12Genome, names: pd.Series, label: str) -> Reconciliation:
    """Run the shared reconciler and keep the histograms."""
    _, report = reconcile_locus_tags(genome, names, label=label)
    return Reconciliation(
        unique_names=report.unique_names,
        status_histogram={s.value: n for s, n in report.status_histogram.items()},
        layer_histogram=dict(report.layer_histogram),
        resolved_fraction=report.resolved_fraction,
    )


def identifier_reconciliation(
    genes: pd.DataFrame,
    locus_ids: Iterable[int],
    mg1655: EcoliK12Genome,
    bw25113: EcoliK12Genome,
    crosswalk: EckCrosswalk,
) -> IdentifierFinding:
    """Map the release's genes (``fit_genes.tab`` rows of ``locus_ids``) both ways.

    ``sysName`` is reconciled against MG1655 and against BW25113, the ``name`` symbol
    against BW25113, and the b-numbers are counted through the one-to-one ECK pairs.
    """
    wanted = set(locus_ids)
    rows = genes.loc[genes["locusId"].isin(wanted)]
    missing = sorted(wanted - set(rows["locusId"]))
    if missing:
        raise ValueError(f"locusIds absent from the gene table: {missing[:10]}")
    sysnames = rows["sysName"].astype(str)
    by_bnumber = {pair.mg1655: pair for pair in crosswalk.pairs}
    paired = [by_bnumber[name] for name in sysnames if name in by_bnumber]
    return IdentifierFinding(
        n_genes=len(rows),
        scaffold_ids=tuple(sorted(int(s) for s in rows["scaffoldId"].unique())),
        b_number_sysnames=int(sysnames.str.fullmatch(r"b\d{4}").sum()),
        mg1655_by_sysname=_reconcile(mg1655, sysnames, "sysName on MG1655"),
        bw25113_by_sysname=_reconcile(bw25113, sysnames, "sysName on BW25113"),
        bw25113_by_symbol=_reconcile(
            bw25113, rows["name"].astype(str), "symbol on BW25113"
        ),
        eck_one_to_one=len(paired),
        eck_numeric_disagreements=sum(not pair.numerics_agree for pair in paired),
    )


def release_identifier_finding(data_root: str | None = None) -> IdentifierFinding:
    """:func:`identifier_reconciliation` on the compendium's genes with fitness."""
    mg1655 = bacterial_genome("ecoli", "MG1655", data_root)
    bw25113 = bacterial_genome("ecoli", "BW25113", data_root)
    if not isinstance(mg1655, EcoliK12MG1655Genome):
        raise TypeError(f"MG1655 resolved to {type(mg1655).__name__}")
    if not isinstance(bw25113, EcoliK12BW25113Genome):
        raise TypeError(f"BW25113 resolved to {type(bw25113).__name__}")
    superset = pd.read_csv(raw_path(SUPERSET_FITNESS, data_root), sep="\t")
    return identifier_reconciliation(
        genes=pd.read_csv(raw_path(RELEASE_GENES, data_root), sep="\t"),
        locus_ids=superset["locusId"],
        mg1655=mg1655,
        bw25113=bw25113,
        crosswalk=eck_crosswalk(mg1655, bw25113),
    )


# --------------------------------------------------------------------------- #
# The loader: this paper's own release (owner policy 2026-10-10)
# --------------------------------------------------------------------------- #
DATASET_ROOT_REL = "data/torchcell/rbtnseq_wetmore2015_ecoli"
#: The organism id the release and the compendium both use; ``screen_id`` is
#: ``Keio:<sample>``, the same key the Price 2018 records carry, so the two analyses of
#: one sample join on it.
ORG_ID: Final = "Keio"
#: Data Set S1's ``Mutant Library`` value for every E. coli sample; Table S1 spells the
#: library ``KEIO_ML9`` (``MUTANT_LIBRARY``), which is what the records store.
DATA_SET_S1_LIBRARY = "Keio_ML9"


@functools.cache
def xlsx_text(path: str) -> str:
    """Every sheet's non-empty rows, cells joined by ' | ', rows by ' / '.

    The rendering ``_data_set_s1`` quotes are cut from (the same one the Price 2018
    loader and ``media.py`` quote their workbooks by); one line per sheet.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        workbook = openpyxl.load_workbook(path, read_only=True)
    sheets = []
    for sheet in workbook.worksheets:
        rows = [
            " | ".join(str(cell) for cell in row if cell is not None)
            for row in sheet.iter_rows(values_only=True)
            if any(cell is not None for cell in row)
        ]
        sheets.append(" / ".join(rows))
    return "\n".join(sheets)


#: ``raw/`` file name -> (mirror, path in it, sha256). The names are distinct.
LOADER_FILE_NAMES: dict[str, tuple[Literal["raw", "library"], str, str]] = {
    "fit_logratios_good.tab": (
        "raw",
        RELEASE_FITNESS,
        RAW_FILES[RELEASE_FITNESS].sha256,
    ),
    "fit_logratios.tab": (
        "raw",
        RELEASE_FITNESS_ALL,
        RAW_FILES[RELEASE_FITNESS_ALL].sha256,
    ),
    "fit_genes.tab": ("raw", RELEASE_GENES, RAW_FILES[RELEASE_GENES].sha256),
    "fit_t.tab": ("raw", RELEASE_T, RAW_FILES[RELEASE_T].sha256),
    "fit_standard_error_obs.tab": (
        "raw",
        RELEASE_SE_OBS,
        RAW_FILES[RELEASE_SE_OBS].sha256,
    ),
    "fit_standard_error_naive.tab": (
        "raw",
        RELEASE_SE_NAIVE,
        RAW_FILES[RELEASE_SE_NAIVE].sha256,
    ),
    "si1.xlsx": ("library", DATA_SET_S1, DATA_SET_S1_SHA256),
}


def loader_source_path(name: str, data_root: str | None = None) -> Path:
    """The mirror file behind ``raw/<name>``, after checking its manifest pin and bytes."""
    root = str(_data_root(data_root))
    mirror, relpath, pin = LOADER_FILE_NAMES[name]
    if mirror == "library":
        return library_path(CITATION_KEY, relpath, pin, root)
    return raw_path(relpath, root)


# The release tables ------------------------------------------------------------ #
#: Largest |t - fit / sqrt(sigma^2 + SE^2)| accepted; 1.99e-13 is measured on all 106
#: samples of the 1.0.0 release.
T_IDENTITY_TOLERANCE = 1e-9


class ReleaseTables(BaseModel):
    """The release's Keio gene tables, aligned gene by gene and sample by sample.

    ``fitness``, ``standard_error`` and ``t`` are genes x samples. ``standard_error``
    is the paper's (``PER_GENE_STANDARD_ERROR``): the larger of the strain-consistency
    and read-count estimates.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    locus_ids: tuple[int, ...]
    b_numbers: tuple[str, ...]
    samples: tuple[str, ...]
    fitness: npt.NDArray[np.float64]
    standard_error: npt.NDArray[np.float64]
    t: npt.NDArray[np.float64]
    n_values: int
    n_naive_larger: int = Field(
        description="values whose read-count SE exceeds the strain-consistency SE"
    )
    t_max_abs_deviation: float = Field(
        description="max |t - fit / sqrt(sigma^2 + SE^2)| over every value"
    )


def align_release(
    fitness: pd.DataFrame,
    t: pd.DataFrame,
    se_obs: pd.DataFrame,
    se_naive: pd.DataFrame,
    samples: Sequence[str],
    sigma: float,
) -> ReleaseTables:
    """Align the four gene tables on ``samples`` and check them against each other.

    ``fitness`` is the all-sample table, as the t and standard-error tables are (#1057:
    the samples the paper's quality rules fail are stored too). Refused: tables whose
    gene rows (locusId, sysName) differ; a sample of ``samples`` missing from a table or
    whose header text differs between tables; a missing value; a non-positive standard
    error; and a released t that is not ``fit / sqrt(sigma^2 + max(se_obs,
    se_naive)^2)``.
    """
    keys = ["locusId", "sysName"]
    for label, table in (("t", t), ("se_obs", se_obs), ("se_naive", se_naive)):
        if not table[keys].equals(fitness[keys]):
            raise ValueError(f"{label}: gene rows differ from the fitness table's")
    fit_columns = _fitness_columns(fitness, "fitness")
    absent = sorted(set(samples) - set(fit_columns))
    if absent:
        raise ValueError(f"fitness: kept samples absent {absent}")
    arrays: dict[str, npt.NDArray[np.float64]] = {}
    for label, table in (("t", t), ("se_obs", se_obs), ("se_naive", se_naive)):
        columns = _fitness_columns(table, label)
        missing = sorted(set(samples) - set(columns))
        if missing:
            raise ValueError(f"{label}: samples absent {missing}")
        differing = sorted(s for s in samples if columns[s] != fit_columns[s])
        if differing:
            raise ValueError(f"{label}: header text differs for {differing}")
        arrays[label] = table[[columns[s] for s in samples]].to_numpy(dtype=np.float64)
    values = fitness[[fit_columns[s] for s in samples]].to_numpy(dtype=np.float64)
    for label, array in {"fitness": values, **arrays}.items():
        if np.isnan(array).any():
            raise ValueError(f"{label}: {int(np.isnan(array).sum())} missing values")
    if (arrays["se_obs"] <= 0).any() or (arrays["se_naive"] <= 0).any():
        raise ValueError("a standard-error estimate is not positive")
    standard_error = np.maximum(arrays["se_obs"], arrays["se_naive"])
    deviation = float(
        np.max(np.abs(arrays["t"] - values / np.sqrt(sigma**2 + standard_error**2)))
    )
    if deviation > T_IDENTITY_TOLERANCE:
        raise ValueError(
            f"the released t is not fit / sqrt({sigma}^2 + SE^2): max deviation "
            f"{deviation:.3g}"
        )
    return ReleaseTables(
        locus_ids=tuple(int(v) for v in fitness["locusId"]),
        b_numbers=tuple(str(v) for v in fitness["sysName"]),
        samples=tuple(samples),
        fitness=values,
        standard_error=standard_error,
        t=arrays["t"],
        n_values=int(values.size),
        n_naive_larger=int((arrays["se_naive"] > arrays["se_obs"]).sum()),
        t_max_abs_deviation=deviation,
    )


def read_release(raw_dir: str, samples: Sequence[str]) -> ReleaseTables:
    """:func:`align_release` on the all-sample tables linked into ``raw_dir``.

    The successful-sample table must hold the same values as the all-sample one for
    every sample it carries, so the two release files are proved to be one release.
    """

    def read(name: str) -> pd.DataFrame:
        return pd.read_csv(osp.join(raw_dir, name), sep="\t")

    every = read("fit_logratios.tab")
    good = read("fit_logratios_good.tab")
    good_columns = _fitness_columns(good, "fitness good")
    every_columns = _fitness_columns(every, "fitness all")
    differing = sorted(
        s
        for s, header in good_columns.items()
        if not np.array_equal(
            good[header].to_numpy(), every[every_columns[s]].to_numpy()
        )
    )
    if differing:
        raise ValueError(
            f"fit_logratios_good.tab differs from the all table on {differing}"
        )
    return align_release(
        fitness=every,
        t=read("fit_t.tab"),
        se_obs=read("fit_standard_error_obs.tab"),
        se_naive=read("fit_standard_error_naive.tab"),
        samples=samples,
        sigma=float(T_STATISTIC_SIGMA.value),
    )


# Samples: retention, the refusals, the environment ------------------------------- #
#: The authors' own withdrawal, the same rule name the compendium loader refuses under.
DROP_WITHDRAWN = AUTHOR_WITHDRAWAL.rule

DROP_RULES: dict[str, str] = {
    DROP_WITHDRAWN: "the source's own authors told readers to disregard it: the "
    f"compendium page's 2021 note ({AUTHOR_WITHDRAWAL.instruction.quote!r}) after they "
    f"found that {AUTHOR_WITHDRAWAL.cause.quote!r} The measured response is therefore "
    "not a response to the named compound, so no record of these samples is stored "
    "(the compendium loader refuses the same four samples under the same rule); their "
    "values stay in the deposited fit_logratios_good.tab"
}
#: The source statement the one rule left rests on: a verbatim quote bound to its sha256.
DROP_RULE_SOURCES: dict[str, SourcedValue] = {
    DROP_WITHDRAWN: AUTHOR_WITHDRAWAL.instruction
}

#: Data Set S1 ``Media`` -> the library object. Each of the first three quotes Data Set
#: S1's Media tab; the MOPS rich entry is Price 2018 Table S18's recipe for the same label
#: (Data Set S1's own row prints its micronutrients at unusable amounts) and quotes Data
#: Set S1's row beside it (#1045 row 16).
MEDIA_BY_LABEL: dict[str, Media] = {
    "M9 minimal media_noCarbon": M9_NOCARBON_WETMORE2015,
    "M9 minimal media_noNitrogen": M9_NONITROGEN_WETMORE2015,
    "LB": LB_LENNOX,
    "MOPS Rich Defined media_noCarbon": MOPS_RICH_DEFINED_NOCARBON_PRICE2018,
}

GroupKind = Literal["carbon_source", "nitrogen_source", "plain"]
#: Data Set S1 ``Group`` -> how its Condition_1 enters the environment. The release's two
#: E. coli ``stress`` rows name no compound (``short`` ``LB``), so they are the released
#: LB with no added perturbation (#1057); a stress row naming a compound would be refused
#: rather than guessed at, since ``plain`` requires an empty Condition_1.
GROUP_KINDS: dict[str, GroupKind] = {
    "carbon source": "carbon_source",
    "nitrogen source": "nitrogen_source",
    "LB": "plain",
    "stress": "plain",
}
#: The media a group may name; a pairing outside this is refused.
GROUP_MEDIA: dict[GroupKind, frozenset[str]] = {
    "carbon_source": frozenset(
        {"M9 minimal media_noCarbon", "MOPS Rich Defined media_noCarbon"}
    ),
    "nitrogen_source": frozenset({"M9 minimal media_noNitrogen"}),
    "plain": frozenset({"LB"}),
}
#: Data Set S1 ``Units_1`` -> the typed unit. ``mg/mL`` IS g/L (1 mg/mL = 1 g/L), so the
#: value is kept as printed.
UNITS_BY_LABEL: dict[str, ConcentrationUnit] = {
    "mM": ConcentrationUnit.millimolar,
    "mg/mL": ConcentrationUnit.g_per_l,
}
TEMPERATURE_COLUMN = "Temperature (°C)"


class SampleSpec(BaseModel):
    """One Data Set S1 condition sample: what it was, how it scored, and its fate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    short: str
    group: str
    media_label: str
    condition: str | None
    concentration: float | None
    units: str | None
    temperature_c: float
    total_generations: float | None
    quality: QualityMetrics
    drop_rule: str | None

    @property
    def screen_id(self) -> str:
        """``Keio:<name>``: sample names are unique only within an organism."""
        return f"{ORG_ID}:{self.name}"


def _require(row: pd.Series, column: str, expected: str) -> None:
    """Refuse a Data Set S1 row whose ``column`` is not ``expected``."""
    if str(row[column]) != expected:
        raise ValueError(
            f"{row['name']}: {column} is {row[column]!r}, not {expected!r}"
        )


def classify_samples(
    experiments: pd.DataFrame, quality: pd.DataFrame
) -> tuple[SampleSpec, ...]:
    """Every condition sample of Data Set S1's Keio sheets as a :class:`SampleSpec`.

    ``experiments`` and ``quality`` are the ``Expts_Keio`` and ``Expt_Quality_Keio``
    sheets; time-zero rows are the denominator of every log ratio, not samples. Each
    sample must be the Keio_ML9 library, liquid, aerobic, and carry no second
    condition, else it is refused; its quality verdict ``u`` must equal the paper's
    five rules applied to its own metrics, else it is refused. The one drop is the
    authors' 2021 withdrawal (#1057: a sample the quality rules fail is stored, its
    verdict kept on its spec). A kept sample must name a known group whose medium fits
    it.
    """
    condition_rows = experiments.loc[experiments["Group"] != "Time0"]
    samples = _unique_index(condition_rows, "Expts_Keio")
    scored = _unique_index(quality, "Expt_Quality_Keio")
    unscored = sorted(set(samples.index) - set(scored.index))
    if unscored:
        raise ValueError(f"condition samples with no quality row: {unscored}")
    specs: list[SampleSpec] = []
    for index, row in samples.iterrows():
        name = str(index)
        row = row.copy()
        row["name"] = name
        _require(row, "Mutant Library", DATA_SET_S1_LIBRARY)
        _require(row, "Liquid_v_solid", "Liquid")
        _require(row, "Aerobic_v_Aerobic", "Aerobic")
        if not pd.isna(row["Condition_2"]):
            raise ValueError(f"{name}: a second condition {row['Condition_2']!r}")
        metrics = _quality(scored, name)
        if metrics.successful == bool(metrics.failed_rules()):
            raise ValueError(
                f"{name}: u = {metrics.successful} but the paper's rules fail "
                f"{metrics.failed_rules()}"
            )
        condition = _optional_str(row["Condition_1"])
        media_label = str(row["Media"])
        group = str(row["Group"])
        drop: str | None = None
        if condition in AUTHOR_WITHDRAWAL.conditions:
            drop = DROP_WITHDRAWN
        else:
            if media_label not in MEDIA_BY_LABEL:
                raise ValueError(f"{name}: unknown medium {media_label!r}")
            if group not in GROUP_KINDS:
                raise ValueError(f"{name}: no kept sample of group {group!r} expected")
            if media_label not in GROUP_MEDIA[GROUP_KINDS[group]]:
                raise ValueError(f"{name}: group {group!r} on medium {media_label!r}")
        specs.append(
            SampleSpec(
                name=name,
                short=str(row["short"]),
                group=group,
                media_label=media_label,
                condition=condition,
                concentration=_optional_float(row["Concentration_1"]),
                units=_optional_str(row["Units_1"]),
                temperature_c=float(row[TEMPERATURE_COLUMN]),
                total_generations=_optional_float(row["Total Generations"]),
                quality=metrics,
                drop_rule=drop,
            )
        )
    return tuple(specs)


def _dose(spec: SampleSpec) -> Concentration:
    """The sample's Condition_1 dose with its typed unit."""
    if spec.concentration is None or spec.units is None:
        raise ValueError(f"{spec.name}: {spec.condition!r} has no dose")
    if spec.units not in UNITS_BY_LABEL:
        raise ValueError(f"{spec.name}: unknown unit {spec.units!r}")
    return Concentration(value=spec.concentration, unit=UNITS_BY_LABEL[spec.units])


DURATION_HOURS_GAP = ProvenanceGap(
    field="duration_hours",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=Provenance(
        source_uri=PAPER_MD,
        citation_key=CITATION_KEY,
        sha256=PAPER_MD_SHA256,
        method=_OCR_METHOD,
        page="Materials and Methods, 'Competitive mutant fitness assays'; Data Set S1 "
        "Expts_Keio has no time column",
    ),
    note="the assays ran to saturation (SATURATED_GROWTH), so no duration in hours is "
    "stated; the population doublings are Data Set S1's Total Generations",
)
GENERATIONS_GAP = ProvenanceGap(
    field="duration_generations",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=Provenance(
        source_uri=DATA_SET_S1,
        citation_key=CITATION_KEY,
        sha256=DATA_SET_S1_SHA256,
        method=_XLSX_METHOD,
        page="sheet Expts_Keio, column 'Total Generations'",
    ),
    note="the column is 'Total number of population doublings of experiment (if "
    "determined)' and is blank for this sample (a 48-well microplate LB sample with no "
    "EndOD)",
)


def build_environment(spec: SampleSpec) -> Environment:
    """The environment of a kept sample from its own Data Set S1 columns.

    A carbon or nitrogen source is the varied factor of a medium that leaves it out
    (``CARBON_FREE_MEDIA``), so it is an ``EnvironmentPhysicalPerturbation`` with the
    compound as ``agent``; the plain LB sample carries no perturbation.
    ``duration_generations`` is the sample's Total Generations, a typed gap where the
    column is blank; ``duration_hours`` is always a typed gap.
    """
    if spec.drop_rule is not None:
        raise ValueError(f"{spec.name} is dropped ({spec.drop_rule})")
    kind = GROUP_KINDS[spec.group]
    perturbations: list[EnvironmentPerturbationType]
    if kind == "plain":
        if spec.condition is not None:
            raise ValueError(f"{spec.name}: a plain sample names {spec.condition!r}")
        perturbations = []
    elif spec.condition is None:
        raise ValueError(f"{spec.name}: a {spec.group} sample names no condition")
    else:
        perturbations = [
            EnvironmentPhysicalPerturbation(
                factor=(
                    PhysicalFactor.carbon_source
                    if kind == "carbon_source"
                    else PhysicalFactor.nitrogen_source
                ),
                magnitude=_dose(spec),
                agent=resolved_compound(spec.condition),
            )
        ]
    gaps = [DURATION_HOURS_GAP]
    if spec.total_generations is None:
        gaps.append(GENERATIONS_GAP)
    return Environment(
        media=MEDIA_BY_LABEL[spec.media_label],
        temperature=Temperature(value=spec.temperature_c),
        perturbations=perturbations,
        aerobicity="aerobic",
        duration_generations=spec.total_generations,
        provenance_gaps=gaps,
    )


# Genes: MG1655 b-numbers placed on BW25113 through one-to-one ECK pairs ---------- #
#: The ECK route must place at least this fraction of the release's genes, else the
#: build stops rather than dropping more (checklist item 4).
MIN_ECK_ROUTE_FRACTION = 0.99

UnmappedReason = Literal[
    "b_number_not_in_mg1655_annotation",
    "mg1655_locus_has_no_eck_synonym",
    "eck_absent_from_bw25113",
    "eck_not_one_to_one",
]


#: The namespace a gene the ECK route cannot place is stored in: the release's own.
RELEASED_NAMESPACE = STRAIN_GENE_NAMESPACES["MG1655"]
RELEASED_PATTERN = LOCUS_TAG_PATTERNS[RELEASED_NAMESPACE]


class GeneMapping(BaseModel):
    """One release gene: placed on its BW25113 locus through a one-to-one ECK pair, or,
    when no pair places it, stored as the released MG1655 b-number (#1057).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    b_number: str
    eck: str
    locus_tag: str
    perturbed_gene_name: str
    numerics_agree: bool
    gene_namespace: BacterialGeneNamespace = STRAIN_GENE_NAMESPACES[REFERENCE_STRAIN]


class UnmappedGene(BaseModel):
    """A release gene the ECK route cannot place, and why."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    b_number: str
    reason: UnmappedReason
    eck: tuple[str, ...]


class IdentifierReport(BaseModel):
    """The ECK route over the release's genes, with the reconciler's histograms."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_source_genes: int
    n_mapped: int
    mapped_fraction: float
    min_fraction: float
    numeric_disagreements: tuple[tuple[str, str, str], ...] = Field(
        description="(b-number, BW25113 tag, ECK) of mapped pairs whose numbers differ"
    )
    unmapped: tuple[UnmappedGene, ...]
    unmapped_by_reason: dict[str, int]
    reconcile_status_histogram: dict[str, int]
    reconcile_layer_histogram: dict[str, int]
    n_symbol_names: int = Field(
        description="perturbed_gene_name is the BW25113 gene symbol"
    )
    n_tag_names: int = Field(
        description="perturbed_gene_name is the locus tag (symbol absent or not unique)"
    )
    released_as: tuple[str, ...] = Field(
        default=(),
        description="the genes the ECK route cannot place, stored as the released "
        "MG1655 b-number (#1057)",
    )


def released_as(gene: UnmappedGene, name: str) -> GeneMapping:
    """A gene no one-to-one ECK pair places, stored as the released b-number (#1057)."""
    if RELEASED_PATTERN.match(gene.b_number) is None:
        raise ValueError(
            f"{gene.b_number!r} is not an MG1655 b-number in form, so no leaf can hold "
            "it as released"
        )
    return GeneMapping(
        b_number=gene.b_number,
        eck=",".join(gene.eck),
        locus_tag=gene.b_number,
        perturbed_gene_name=name,
        numerics_agree=True,
        gene_namespace=RELEASED_NAMESPACE,
    )


def unmapped_reason(
    b_number: str, crosswalk: EckCrosswalk, mg1655_ecks: Mapping[str, tuple[str, ...]]
) -> UnmappedGene:
    """Why a b-number has no one-to-one ECK pair (it must not have one)."""
    if any(pair.mg1655 == b_number for pair in crosswalk.pairs):
        raise ValueError(f"{b_number} has a one-to-one pair")
    if b_number not in mg1655_ecks:
        return UnmappedGene(
            b_number=b_number, reason="b_number_not_in_mg1655_annotation", eck=()
        )
    ecks = mg1655_ecks[b_number]
    if not ecks:
        return UnmappedGene(
            b_number=b_number, reason="mg1655_locus_has_no_eck_synonym", eck=()
        )
    if any(eck in set(crosswalk.not_one_to_one) for eck in ecks):
        return UnmappedGene(b_number=b_number, reason="eck_not_one_to_one", eck=ecks)
    if all(eck in set(crosswalk.mg1655_only) for eck in ecks):
        return UnmappedGene(
            b_number=b_number, reason="eck_absent_from_bw25113", eck=ecks
        )
    raise ValueError(f"{b_number}: ECK {ecks} is paired, absent and not shared")


def gene_symbol(
    symbol: str | None, tag: str, resolve: Callable[[str], GeneNameResolution]
) -> str:
    """The locus's gene symbol when it resolves back to that locus, else the tag."""
    if symbol:
        resolution = resolve(symbol)
        if resolution.systematic_name == tag and resolution.status in (
            GeneNameStatus.CURRENT,
            GeneNameStatus.RENAMED,
        ):
            return symbol
    return tag


def map_genes(
    b_numbers: Sequence[str],
    mg1655: EcoliK12MG1655Genome,
    bw25113: EcoliK12BW25113Genome,
    *,
    label: str,
) -> tuple[dict[str, GeneMapping], IdentifierReport]:
    """Place the release's b-numbers on BW25113 locus tags and report how.

    The route is the one-to-one ECK synonym join of the two deposited annotations
    (``eck_crosswalk``). The placed genes' ECK ids are then run through
    ``reconcile_locus_tags`` on BW25113, which must return the crosswalk's tag for every
    one. Below ``MIN_ECK_ROUTE_FRACTION`` the build stops.
    """
    crosswalk = eck_crosswalk(mg1655, bw25113)
    by_b = {pair.mg1655: pair for pair in crosswalk.pairs}
    placed = [b for b in b_numbers if b in by_b]
    fraction = len(placed) / len(b_numbers)
    if fraction < MIN_ECK_ROUTE_FRACTION:
        raise LocusTagResolutionError(
            f"{label}: the ECK route places {len(placed)} of {len(b_numbers)} genes "
            f"({fraction:.4f}), below {MIN_ECK_ROUTE_FRACTION}"
        )
    stored, reconciliation = reconcile_locus_tags(
        bw25113, pd.Series([by_b[b].eck for b in placed]), label=f"{label} ECK ids"
    )
    differing = [
        b for b, tag in zip(placed, stored, strict=True) if tag != by_b[b].bw25113
    ]
    if differing:
        raise ValueError(f"{label}: reconciler and crosswalk disagree on {differing}")
    mg1655_ecks = {
        tag: tuple(s for s in locus.synonyms if s.startswith("ECK"))
        for tag, locus in mg1655.genbank.loci.items()
    }
    loci = bw25113.genbank.loci
    mapping = {
        b: GeneMapping(
            b_number=b,
            eck=by_b[b].eck,
            locus_tag=by_b[b].bw25113,
            perturbed_gene_name=gene_symbol(
                loci[by_b[b].bw25113].symbol, by_b[b].bw25113, bw25113.resolve_gene_name
            ),
            numerics_agree=by_b[b].numerics_agree,
        )
        for b in placed
    }
    unplaced = tuple(
        unmapped_reason(b, crosswalk, mg1655_ecks) for b in b_numbers if b not in by_b
    )
    n_symbol = sum(1 for m in mapping.values() if m.perturbed_gene_name != m.locus_tag)
    mg1655_loci = mg1655.genbank.loci
    for gene in unplaced:
        mapping[gene.b_number] = released_as(
            gene,
            gene_symbol(
                mg1655_loci[gene.b_number].symbol,
                gene.b_number,
                mg1655.resolve_gene_name,
            )
            if gene.b_number in mg1655_loci
            else gene.b_number,
        )
    report = IdentifierReport(
        n_source_genes=len(b_numbers),
        n_mapped=len(placed),
        mapped_fraction=fraction,
        min_fraction=MIN_ECK_ROUTE_FRACTION,
        numeric_disagreements=tuple(
            (m.b_number, m.locus_tag, m.eck)
            for m in mapping.values()
            if not m.numerics_agree
        ),
        unmapped=unplaced,
        unmapped_by_reason=dict(Counter(u.reason for u in unplaced)),
        reconcile_status_histogram={
            status.value: n for status, n in reconciliation.status_histogram.items()
        },
        reconcile_layer_histogram=dict(reconciliation.layer_histogram),
        n_symbol_names=n_symbol,
        n_tag_names=len(placed) - n_symbol,
        released_as=tuple(gene.b_number for gene in unplaced),
    )
    return mapping, report


# Records ------------------------------------------------------------------------ #
UNITS = (
    "RB-TnSeq gene fitness, Wetmore 2015 release (statistics version 1.0.0): normalized "
    "log2 ratio of the gene's insertion-strain barcode abundance after growth in the "
    "condition over the time-zero sample, weighted over the gene's usable strains, "
    "typical gene = 0; SE = sqrt(max(V_e, V_n)) = max(se, sdNaive), the paper's per-gene "
    "standard error (Methods (iv)); environment_response_t_statistic = the released "
    "moderated t = fit / sqrt(0.1^2 + SE^2)"
)
UNITS_REFERENCE = "the typical gene of this sample (gene fitness is normalized to 0)"
DESCRIPTION = (
    "Tn5 transposon insertion disrupting a gene; gene-level call over the gene's usable "
    "insertion strains (no per-strain barcode or mapped site). Locus tag AS RELEASED: "
    "the release's sysName, an MG1655 b-number, in the MG1655 namespace, named by "
    "fit_genes.tab; the library strain is BW25113"
)
PUBLICATION = Publication(doi=PAPER_DOI, doi_url=f"https://doi.org/{PAPER_DOI}")


class StoredGene(BaseModel):
    """One release gene exactly as the release gives it (#1057).

    ``b_number`` is the ``sysName``; ``name`` is ``fit_genes.tab``'s own ``name``,
    else the b-number. No annotation is read.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    b_number: str
    name: str


def released_genes(
    b_numbers: Sequence[str], gene_table: pd.DataFrame
) -> dict[str, StoredGene]:
    """Each released sysName, named by the release's gene table, in release order.

    A sysName that is not an MG1655 b-number in form, or that the gene table names
    twice or not at all, is refused: the release changed shape.
    """
    names: dict[str, str] = {}
    for sys_name, name in zip(gene_table["sysName"], gene_table["name"], strict=True):
        if pd.isna(sys_name):
            continue
        key = str(sys_name)
        if key in names:
            raise ValueError(f"fit_genes.tab names {key} twice")
        names[key] = "" if pd.isna(name) else str(name).strip()
    out: dict[str, StoredGene] = {}
    for b in b_numbers:
        if RELEASED_PATTERN.match(b) is None:
            raise ValueError(f"{b!r} is not an MG1655 b-number in form")
        if b not in names:
            raise ValueError(f"{b} is in the fitness table but not in fit_genes.tab")
        out[b] = StoredGene(b_number=b, name=names[b] or b)
    return out


def build_genotype(gene: StoredGene) -> Genotype:
    """A Tn5 insertion in one gene of KEIO_ML9, under the released b-number."""
    return Genotype(
        perturbations=[
            TransposonInsertionPerturbation(
                systematic_gene_name=gene.b_number,
                perturbed_gene_name=gene.name,
                gene_namespace=RELEASED_NAMESPACE,
                description=DESCRIPTION,
                transposon=str(TRANSPOSON.value),
                library_pool=str(MUTANT_LIBRARY.value),
            )
        ]
    )


def build_phenotype(
    fitness: float, standard_error: float, t_statistic: float, *, screen_id: str
) -> EnvironmentResponsePhenotype:
    """One gene's fitness in one sample, with the paper's SE and the released t."""
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.log2_ratio,
        assay_type=AssayType.pooled_competitive_growth_barcode,
        environment_response=fitness,
        environment_response_uncertainty=standard_error,
        environment_response_uncertainty_type=UncertaintyType.standard_error,
        environment_response_t_statistic=t_statistic,
        screen_id=screen_id,
        units=UNITS,
        provenance_gaps=[N_SAMPLES_GAP, SAMPLE_UNIT_GAP],
    )


def build_reference(
    dataset_name: str,
    environment: Environment,
    genome_reference: ReferenceGenome,
    screen_id: str,
) -> EnvironmentResponseExperimentReference:
    """The typical gene of the same sample: fitness 0 by the normalization."""
    return EnvironmentResponseExperimentReference(
        dataset_name=dataset_name,
        genome_reference=genome_reference,
        environment_reference=environment,
        phenotype_reference=EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.log2_ratio,
            assay_type=AssayType.pooled_competitive_growth_barcode,
            environment_response=0.0,
            screen_id=screen_id,
            units=UNITS_REFERENCE,
        ),
    )


class SampleRecords(BaseModel):
    """A kept sample with everything its records share."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    spec: SampleSpec
    column: int
    environment: Environment
    reference: EnvironmentResponseExperimentReference


def sample_records(
    dataset_name: str,
    specs: Sequence[SampleSpec],
    release: ReleaseTables,
    genome_reference: ReferenceGenome,
) -> list[SampleRecords]:
    """The kept samples, in Data Set S1 order, each with its release column."""
    columns = {name: j for j, name in enumerate(release.samples)}
    out: list[SampleRecords] = []
    for spec in specs:
        if spec.drop_rule is not None:
            continue
        environment = build_environment(spec)
        out.append(
            SampleRecords(
                spec=spec,
                column=columns[spec.name],
                environment=environment,
                reference=build_reference(
                    dataset_name, environment, genome_reference, spec.screen_id
                ),
            )
        )
    return out


def iter_records(
    dataset_name: str,
    samples: Sequence[SampleRecords],
    release: ReleaseTables,
    genes: Mapping[str, StoredGene],
) -> Iterator[
    tuple[
        EnvironmentResponseExperiment,
        EnvironmentResponseExperimentReference,
        Publication,
    ]
]:
    """Every record, sample by sample then gene by gene in release order."""
    rows = [(i, build_genotype(genes[b])) for i, b in enumerate(release.b_numbers)]
    for sample in samples:
        j = sample.column
        for i, genotype in rows:
            yield (
                EnvironmentResponseExperiment(
                    dataset_name=dataset_name,
                    genotype=genotype,
                    environment=sample.environment,
                    phenotype=build_phenotype(
                        float(release.fitness[i, j]),
                        float(release.standard_error[i, j]),
                        float(release.t[i, j]),
                        screen_id=sample.spec.screen_id,
                    ),
                ),
                sample.reference,
                PUBLICATION,
            )


#: What the build writes: 97 samples (84 measured on the deposited release 2026-10-11,
#: the 4 MOPS rich samples #1045 row 16 keeps, and the 9 the quality rules fail, kept
#: since #1057) x all 3,646 released genes, each under its released b-number.
EXPECTED_SAMPLES = 97
EXPECTED_GENES = 3646
EXPECTED_RECORDS = EXPECTED_SAMPLES * EXPECTED_GENES
#: The withdrawn samples, refused: no record of them is stored.
EXPECTED_WITHDRAWN: tuple[str, ...] = (
    "set1IT007",
    "set1IT008",
    "set1IT043",
    "set1IT044",
)


# Inventory ---------------------------------------------------------------------- #
class DropRuleCount(BaseModel):
    """One retention rule, what it removed, and the released values involved.

    ``n_records`` counts released gene-fitness values (``fit_logratios.tab`` covers
    every sample), so the rules close over ``Inventory.source_records``: stored + every
    rule's values = condition samples x release genes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    rule: str
    scope: Literal["sample", "gene"]
    description: str
    source: SourcedValue | None = Field(
        default=None,
        description="the source's own statement the rule rests on, verbatim and bound "
        "to the sha256-pinned file it was read from (DROP_RULE_SOURCES); the author "
        "withdrawal is the only rule left (#1057)",
    )
    items: tuple[str, ...]
    n_records: int


class FailedSample(BaseModel):
    """A stored sample the paper's own rules fail (``u = FALSE``), as released."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    short: str
    failed_rules: tuple[str, ...]
    g_med: float
    cor12: float


class Inventory(BaseModel):
    """The sample and record accounting the dendron note states."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    condition_samples: int
    successful: int
    quality_flagged: tuple[FailedSample, ...] = Field(
        description="the samples Data Set S1 scores u = FALSE, each STORED since #1057 "
        "with this verdict as released (no record field holds a sample's QC verdict)"
    )
    quality_rules_reproduce_u: bool
    stored_samples: int
    stored_samples_by_group: dict[str, int]
    stored_samples_by_medium: dict[str, int]
    withdrawn_refused: tuple[str, ...] = Field(
        description="the samples the authors' 2021 note withdraws: refused, not stored"
    )
    source_genes: int
    stored_genes: int
    source_records: int = Field(
        description="condition samples x release genes: every released gene-fitness "
        "value (fit_logratios.tab), the failed samples' included"
    )
    good_table_records: int = Field(
        description="successful samples x release genes: the values of the release's "
        "gene-fitness table (fit_logratios_good.tab)"
    )
    stored_records: int
    rules: tuple[DropRuleCount, ...]
    unidentified_compounds: dict[str, int] = Field(
        description="Condition_1 labels of stored samples the identity table does not "
        "resolve -> stored samples naming them (records carry the typed inchikey gap)"
    )


def inventory(specs: Sequence[SampleSpec], n_genes: int) -> Inventory:
    """Count samples, genes and records kept, flagged and dropped, by rule."""
    successful = [s for s in specs if s.quality.successful]
    kept = [s for s in specs if s.drop_rule is None]
    genes = n_genes
    rules = [
        DropRuleCount(
            rule=rule,
            scope="sample",
            description=DROP_RULES[rule],
            source=DROP_RULE_SOURCES.get(rule),
            items=tuple(s.name for s in specs if s.drop_rule == rule),
            n_records=sum(1 for s in specs if s.drop_rule == rule) * n_genes,
        )
        for rule in (DROP_WITHDRAWN,)
    ]
    withdrawn = tuple(s.name for s in specs if s.drop_rule == DROP_WITHDRAWN)
    unidentified = Counter(
        s.condition
        for s in kept
        if s.condition is not None
        and not resolve_compound_identity(name=s.condition).identified
    )
    return Inventory(
        condition_samples=len(specs),
        successful=len(successful),
        quality_flagged=tuple(
            FailedSample(
                name=s.name,
                short=s.short,
                failed_rules=s.quality.failed_rules(),
                g_med=s.quality.g_med,
                cor12=s.quality.cor12,
            )
            for s in specs
            if not s.quality.successful
        ),
        quality_rules_reproduce_u=all(
            s.quality.successful == (not s.quality.failed_rules()) for s in specs
        ),
        stored_samples=len(kept),
        stored_samples_by_group=dict(Counter(s.group for s in kept)),
        stored_samples_by_medium=dict(Counter(s.media_label for s in kept)),
        withdrawn_refused=withdrawn,
        source_genes=n_genes,
        stored_genes=genes,
        source_records=len(specs) * n_genes,
        good_table_records=len(successful) * n_genes,
        stored_records=len(kept) * genes,
        rules=tuple(rules),
        unidentified_compounds=dict(sorted(unidentified.items())),
    )


# Dataset ------------------------------------------------------------------------ #
@register_dataset
class RbTnseqWetmore2015Dataset(ExperimentDataset):
    """Wetmore 2015, E. coli BW25113 KEIO_ML9: gene fitness per successful sample."""

    organism = Organism.ecoli

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
        """Initialize; ``ecoli_genome`` is the BW25113 genome the build entry points inject."""
        self.ecoli_genome = ecoli_genome
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return EnvironmentResponseExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return EnvironmentResponseExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The release tables and Data Set S1, linked into ``raw/`` by ``download``."""
        return list(LOADER_FILE_NAMES)

    def download(self) -> None:
        """Link each pinned mirror file into ``raw/`` after verifying its sha256."""
        os.makedirs(self.raw_dir, exist_ok=True)
        for name, (_, _, pin) in LOADER_FILE_NAMES.items():
            link_verified(loader_source_path(name), osp.join(self.raw_dir, name), pin)
        log.info("Wetmore 2015 Keio raw files linked into %s", self.raw_dir)

    def _bw25113(self) -> EcoliK12BW25113Genome:
        """The injected genome, or the BW25113 genome reopened read-only (direct runs)."""
        genome = (
            bacterial_genome("ecoli", REFERENCE_STRAIN)
            if self.ecoli_genome is None
            else self.ecoli_genome
        )
        if not isinstance(genome, EcoliK12BW25113Genome):
            raise TypeError(f"expected the BW25113 genome, got {type(genome).__name__}")
        self.ecoli_genome = genome
        return genome

    @post_process
    def process(self) -> None:
        """Build one record per (released gene, kept sample) and the preprocess reports."""
        verify_raw_files(
            self.raw_dir, {name: pin for name, (_, _, pin) in LOADER_FILE_NAMES.items()}
        )
        s1 = osp.join(self.raw_dir, "si1.xlsx")
        specs = classify_samples(
            pd.read_excel(s1, sheet_name=DATA_SET_S1_EXPERIMENTS_SHEET),
            pd.read_excel(s1, sheet_name=DATA_SET_S1_QUALITY_SHEET),
        )
        release = read_release(
            self.raw_dir, [s.name for s in specs if s.drop_rule is None]
        )
        genes = released_genes(
            release.b_numbers,
            pd.read_csv(osp.join(self.raw_dir, "fit_genes.tab"), sep="\t"),
        )
        samples = sample_records(
            self.name, specs, release, assembly_reference(REFERENCE_STRAIN)
        )
        counts = inventory(specs, len(genes))

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        idx = 0
        batch_size = 100_000
        txn = env.begin(write=True)
        itxn = interned_env.begin(write=True)
        for experiment, reference, publication in tqdm(
            iter_records(self.name, samples, release, genes),
            total=counts.stored_records,
            desc="Wetmore 2015 Keio",
        ):
            txn.put(
                f"{idx}".encode(),
                self._intern_record(experiment, reference, publication, itxn),
            )
            idx += 1
            if idx % batch_size == 0:
                itxn.commit()
                txn.commit()
                txn = env.begin(write=True)
                itxn = interned_env.begin(write=True)
        itxn.commit()
        txn.commit()
        env.close()
        interned_env.close()
        if idx != counts.stored_records:
            raise RuntimeError(
                f"wrote {idx} records, inventory says {counts.stored_records}"
            )
        self._write_reports(specs, release, counts)
        log.info("Wrote %d Wetmore 2015 Keio records", idx)

    def _write_reports(
        self, specs: Sequence[SampleSpec], release: ReleaseTables, counts: Inventory
    ) -> None:
        """``dropped_records.json``, ``samples.json`` (each sample's quality verdict as
        released) and ``standard_error.json`` in ``preprocess/``.
        """
        payloads: dict[str, Any] = {
            "dropped_records.json": counts.model_dump(mode="json"),
            "samples.json": [s.model_dump(mode="json") for s in specs],
            "standard_error.json": {
                "rule": PER_GENE_STANDARD_ERROR.quote,
                "n_values": release.n_values,
                "n_naive_larger": release.n_naive_larger,
                "t_max_abs_deviation": release.t_max_abs_deviation,
                "sigma": T_STATISTIC_SIGMA.value,
            },
        }
        for name, payload in payloads.items():
            with open(osp.join(self.preprocess_dir, name), "w") as handle:
                json.dump(payload, handle, indent=2)

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError(
            "Wetmore 2015 builds its records in process(); see iter_records"
        )


# --------------------------------------------------------------------------- #
# Verification (L0 to L4), with the compendium as the L4 sanity check
# --------------------------------------------------------------------------- #
VERIFY_PROVENANCE = Provenance(
    source_uri=RAW_FILES[RELEASE_FITNESS].url,
    citation_key=CITATION_KEY,
    sha256=RAW_FILES[RELEASE_FITNESS].sha256,
    method=(
        "this paper's own release (statistics version 1.0.0): gene fitness of Data Set "
        "S1's successful Keio samples; SE = max(fit_standard_error_obs, "
        "fit_standard_error_naive); t from fit_t.tab; MG1655 b-numbers placed on "
        "BW25113 through one-to-one ECK pairs"
    ),
    page="Data Set S1 (si/si1.xlsx), sheets Expts_Keio and Expt_Quality_Keio",
    retrieved=RAW_RETRIEVED_AT,
)
#: The served compendium's dev store the L4 cross-check reads, read-only.
SUPERSET_DATASET_ROOT_REL = "data/torchcell/rbtnseq_price2018_ecoli"
#: The cross-check's bar (SUPERSET_REPLICATE_AGREEMENT): a shared sample's two analyses
#: must correlate at least as well as the compendium's median pair of exact replicates.
COMPENDIUM_MIN_PEARSON_R = float(SUPERSET_REPLICATE_AGREEMENT.value)
#: How many of the largest single-gene differences the cross-check lists as findings.
N_LARGEST_DIFFERENCES = 10


def stored_tags_are_loci(
    tags: Iterable[str], resolve: Callable[[str], GeneNameResolution]
) -> LevelResult:
    """SUPPLEMENTARY L1: every stored tag resolves to itself as a locus of BW25113."""
    statuses: Counter[str] = Counter()
    elsewhere: list[str] = []
    ordered = sorted(set(tags))
    for tag in ordered:
        resolution = resolve(tag)
        statuses[str(resolution.status.value)] += 1
        if resolution.systematic_name != tag or resolution.status not in (
            GeneNameStatus.CURRENT,
            GeneNameStatus.NON_GENE_FEATURE,
        ):
            elsewhere.append(tag)
    return LevelResult(
        level=Level.L1,
        name="stored_tags_are_loci_of_the_pinned_assembly",
        passed=not elsewhere,
        message=(
            f"SUPPLEMENTARY: {len(ordered)} stored tags, statuses {dict(statuses)}; "
            f"{len(elsewhere)} do not resolve to themselves"
        ),
        details={"statuses": dict(statuses), "not_a_locus": elsewhere[:20]},
    )


class StoredValue(BaseModel):
    """The cells of one stored record the supplementary rows read."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample: str
    tag: str
    gene_namespace: str
    value: float
    standard_error: float | None
    t_statistic: float | None


def stored_values(records: Iterable[Mapping[str, Any]]) -> list[StoredValue]:
    """One :class:`StoredValue` per record of an RB-TnSeq store (``Keio:<sample>``)."""
    out: list[StoredValue] = []
    for record in records:
        phenotype = record["experiment"]["phenotype"]
        perturbations = record["experiment"]["genotype"]["perturbations"]
        if len(perturbations) != 1:
            raise ValueError(f"expected one perturbation, got {len(perturbations)}")
        out.append(
            StoredValue(
                sample=str(phenotype["screen_id"]).removeprefix(f"{ORG_ID}:"),
                tag=str(perturbations[0]["systematic_gene_name"]),
                gene_namespace=str(perturbations[0]["gene_namespace"]),
                value=float(phenotype["environment_response"]),
                standard_error=phenotype["environment_response_se"],
                t_statistic=phenotype.get("environment_response_t_statistic"),
            )
        )
    return out


def withdrawal_census(values: Sequence[StoredValue]) -> LevelResult:
    """SUPPLEMENTARY L1: not one record of a withdrawn sample is in the store.

    The authors told readers to disregard these four samples, so the store must hold no
    record of them at all: a stored record would put data its own authors withdrew in
    front of a consumer who never read the note. The row also checks that the stored
    samples are the rest of the kept set, so a refusal cannot quietly take more.
    """
    stored = sorted({v.sample for v in values})
    withdrawn_present = sorted(set(stored) & set(EXPECTED_WITHDRAWN))
    n_records = sum(1 for v in values if v.sample in set(EXPECTED_WITHDRAWN))
    passed = not withdrawn_present and len(stored) == EXPECTED_SAMPLES
    return LevelResult(
        level=Level.L1,
        name="withdrawn_samples_refused_not_stored",
        passed=passed,
        message=(
            f"SUPPLEMENTARY: {n_records} records of the {len(EXPECTED_WITHDRAWN)} "
            f"samples the authors withdrew ({list(EXPECTED_WITHDRAWN)}); "
            f"{len(withdrawn_present)} of them are in the store; {len(stored)} samples "
            f"stored, expected {EXPECTED_SAMPLES}"
        ),
        details={
            "withdrawn": list(EXPECTED_WITHDRAWN),
            "withdrawn_present": withdrawn_present,
            "n_stored_samples": len(stored),
            "rule": DROP_WITHDRAWN,
            "quote": AUTHOR_WITHDRAWAL.instruction.quote,
            "sha256": AUTHOR_WITHDRAWAL.instruction.provenance.sha256,
        },
    )


def t_identity(values: Sequence[StoredValue], sigma: float) -> LevelResult:
    """SUPPLEMENTARY L2: every stored t is fit / sqrt(sigma^2 + SE^2) for its own SE.

    Pins that the stored SE is the paper's (the denominator of its moderated t) and
    that the t and the value were read from the same cell.
    """
    deviation = 0.0
    missing = 0
    for v in values:
        if v.t_statistic is None or v.standard_error is None:
            missing += 1
            continue
        expected = v.value / float(np.sqrt(sigma**2 + v.standard_error**2))
        deviation = max(deviation, abs(v.t_statistic - expected))
    passed = missing == 0 and deviation <= T_IDENTITY_TOLERANCE
    return LevelResult(
        level=Level.L2,
        name="t_statistic_matches_the_stored_se",
        passed=passed,
        message=(
            f"SUPPLEMENTARY: max |t - fit / sqrt({sigma}^2 + SE^2)| = {deviation:.3g} "
            f"over {len(values) - missing} records (tolerance {T_IDENTITY_TOLERANCE}); "
            f"{missing} records lack a t or an SE"
        ),
        details={"max_abs_deviation": deviation, "missing": missing},
    )


class CompendiumAgreement(BaseModel):
    """One shared sample: this release's stored values against the compendium's."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample: str
    n_genes: int = Field(description="genes both stores hold for the sample, compared")
    n_only_here: int = Field(
        description="genes this store holds that the compendium's sample lacks; the "
        "compendium's gene set contains every one of this release's genes, so 0"
    )
    n_only_there: int = Field(
        description="genes only the compendium holds (its larger gene set; not compared)"
    )
    pearson_r: float
    max_abs_difference: float
    median_abs_difference: float


class GeneDifference(BaseModel):
    """One (sample, gene) whose two analyses differ: a finding, not an error."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample: str
    tag: str
    wetmore_1_0_0: float
    price_1_0_3: float

    @property
    def difference(self) -> float:
        """This release's value minus the compendium's."""
        return self.wetmore_1_0_0 - self.price_1_0_3


class CompendiumCrossCheck(BaseModel):
    """The L4 sanity check of this release against the served compendium's store."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    shared_by_name: tuple[str, ...] = Field(
        description="this paper's samples the compendium's Table S5 carries"
    )
    compared: tuple[str, ...] = Field(description="shared samples in both stores")
    absent_here: dict[str, str] = Field(
        description="shared samples this store does not hold -> this loader's rule"
    )
    absent_there: dict[str, str] = Field(
        description="shared samples the compendium store does not hold -> its own "
        "drop rule (its preprocess/samples.json)"
    )
    min_pearson_r: float
    agreements: tuple[CompendiumAgreement, ...]
    largest_differences: tuple[GeneDifference, ...]
    n_values_compared: int

    @property
    def passed(self) -> bool:
        """Every compared sample holds no gene the compendium lacks and agrees at least
        as well as the replicate bar.
        """
        return bool(self.agreements) and all(
            a.pearson_r >= self.min_pearson_r and a.n_only_here == 0
            for a in self.agreements
        )


def compendium_cross_check(
    own: Sequence[StoredValue],
    served: Sequence[StoredValue],
    shared_by_name: Sequence[str],
    absent_here: Mapping[str, str],
    served_drop_rules: Mapping[str, str | None],
    *,
    min_pearson_r: float = COMPENDIUM_MIN_PEARSON_R,
) -> CompendiumCrossCheck:
    """Per shared sample, Pearson r and the differences of the two stores' values.

    ``absent_here`` names the shared samples this store does not hold, with this
    loader's rule; ``served_drop_rules`` is the compendium store's own per-sample rule
    (None = kept), which names why a shared sample is absent there. A shared sample
    absent from a store without a rule naming it is refused. Each compared sample is
    scored on the genes both stores hold (the compendium's gene set is larger: 3,768
    against this release's 3,628 placed genes, each store also holding its as-released
    genes since #1057); a gene held here and not there is
    counted and fails the row, and two stores sharing no gene are refused.
    """
    mine: dict[str, dict[str, float]] = {}
    for v in own:
        mine.setdefault(v.sample, {})[v.tag] = v.value
    theirs: dict[str, dict[str, float]] = {}
    for v in served:
        theirs.setdefault(v.sample, {})[v.tag] = v.value
    compared = sorted(s for s in shared_by_name if s in mine and s in theirs)
    unexplained_here = sorted(
        s for s in shared_by_name if s not in mine and s not in absent_here
    )
    absent_there = {
        s: str(served_drop_rules[s])
        for s in shared_by_name
        if s not in theirs and served_drop_rules.get(s) is not None
    }
    unexplained_there = sorted(
        s for s in shared_by_name if s not in theirs and s not in absent_there
    )
    if unexplained_here or unexplained_there:
        raise ValueError(
            f"shared samples absent without a rule: here {unexplained_here}, in the "
            f"compendium store {unexplained_there}"
        )
    agreements: list[CompendiumAgreement] = []
    differences: list[GeneDifference] = []
    n_values = 0
    for sample in compared:
        tags = sorted(set(mine[sample]) & set(theirs[sample]))
        if not tags:
            raise ValueError(f"{sample}: the two stores share no gene")
        x = np.array([mine[sample][g] for g in tags])
        y = np.array([theirs[sample][g] for g in tags])
        gaps = np.abs(x - y)
        n_values += len(tags)
        agreements.append(
            CompendiumAgreement(
                sample=sample,
                n_genes=len(tags),
                n_only_here=len(set(mine[sample]) - set(theirs[sample])),
                n_only_there=len(set(theirs[sample]) - set(mine[sample])),
                pearson_r=float(np.corrcoef(x, y)[0, 1]),
                max_abs_difference=float(gaps.max()),
                median_abs_difference=float(np.median(gaps)),
            )
        )
        for k in np.argsort(-gaps)[:N_LARGEST_DIFFERENCES]:
            differences.append(
                GeneDifference(
                    sample=sample,
                    tag=tags[int(k)],
                    wetmore_1_0_0=float(x[int(k)]),
                    price_1_0_3=float(y[int(k)]),
                )
            )
    differences.sort(key=lambda d: -abs(d.difference))
    return CompendiumCrossCheck(
        shared_by_name=tuple(sorted(shared_by_name)),
        compared=tuple(compared),
        absent_here={
            s: absent_here[s] for s in sorted(shared_by_name) if s not in mine
        },
        absent_there=dict(sorted(absent_there.items())),
        min_pearson_r=min_pearson_r,
        agreements=tuple(agreements),
        largest_differences=tuple(differences[:N_LARGEST_DIFFERENCES]),
        n_values_compared=n_values,
    )


def cross_check_result(check: CompendiumCrossCheck, store: str) -> LevelResult:
    """The cross-check as an L4 row: pass on the replicate bar, differences as findings."""
    rs = [a.pearson_r for a in check.agreements]
    worst = max(
        check.largest_differences, key=lambda d: abs(d.difference), default=None
    )
    finding = (
        f"; largest single-gene difference {worst.difference:+.3f} log2 "
        f"({worst.sample} {worst.tag}), a finding, not an error"
        if worst is not None
        else ""
    )
    summary = (
        f"min {min(rs):.4f} / median {statistics.median(rs):.4f}" if rs else "none"
    )
    return LevelResult(
        level=Level.L4,
        name="compendium_cross_check_price2018",
        passed=check.passed,
        message=(
            f"SUPPLEMENTARY: {len(check.compared)} of {len(check.shared_by_name)} shared "
            f"samples compared against {store} ({check.n_values_compared} values); "
            f"per-sample Pearson r {summary} against the bar "
            f"{check.min_pearson_r} (the compendium's median exact-replicate r){finding}; "
            f"absent here {check.absent_here}, absent there {check.absent_there}"
        ),
        details=check.model_dump(mode="json"),
    )


def read_served_drop_rules(superset_root: str) -> dict[str, str | None]:
    """The compendium store's own per-sample fate: ``preprocess/samples.json``."""
    with open(osp.join(superset_root, "preprocess", "samples.json")) as handle:
        rows = json.load(handle)
    return {str(row["name"]): row["drop_rule"] for row in rows}


def verify_build(
    dataset_root: str,
    *,
    data_root: str | None = None,
    expected_count: int = EXPECTED_RECORDS,
) -> VerificationReport:
    """Run the environment-response L0-L4 verifier on a built tree and write its report.

    The gene universe and resolver are MG1655's, the namespace every released b-number
    is stored in (#1057); its genome-reference rows report and never gate. Three
    SUPPLEMENTARY rows are added: no record of a withdrawn sample is stored (L1), every
    stored t matches its stored SE (L2), and the cross-check against the Price 2018
    compendium's dev store (L4, :func:`compendium_cross_check`), which reads that store
    read-only and refuses when it is absent; both stores key a gene by its released
    b-number.
    """
    from torchcell.verification.environment_response import (
        verify_environment_response_dataset_streaming,
    )
    from torchcell.verification.runners import stream_records

    root = str(_data_root(data_root))
    genome = bacterial_genome("ecoli", "MG1655", root)
    report = verify_environment_response_dataset_streaming(
        stream_records(dataset_root),
        dataset_name=RbTnseqWetmore2015Dataset.__name__,
        provenance=VERIFY_PROVENANCE,
        expected_count=expected_count,
        sgd_genes=set(genome.genbank.loci),
        min_containment=1.0,
        resolve_gene_name=genome.resolve_gene_name,
    )
    own = stored_values(stream_records(dataset_root))
    report.add(withdrawal_census(own))
    report.add(t_identity(own, float(T_STATISTIC_SIGMA.value)))
    superset_root = osp.join(root, SUPERSET_DATASET_ROOT_REL)
    if not osp.isdir(osp.join(superset_root, "processed", "lmdb")):
        raise FileNotFoundError(
            f"{superset_root}: the compendium store the L4 cross-check reads is absent"
        )
    with open(osp.join(dataset_root, "preprocess", "samples.json")) as handle:
        own_specs = [SampleSpec.model_validate(row) for row in json.load(handle)]
    record = subsumption_record(root)
    check = compendium_cross_check(
        own,
        stored_values(stream_records(superset_root)),
        record.carried,
        {s.name: str(s.drop_rule) for s in own_specs if s.drop_rule is not None},
        read_served_drop_rules(superset_root),
    )
    report.add(cross_check_result(check, SUPERSET_DATASET_ROOT_REL))
    preprocess = osp.join(dataset_root, "preprocess")
    os.makedirs(preprocess, exist_ok=True)
    with open(osp.join(preprocess, "verification_report.json"), "w") as handle:
        handle.write(report.model_dump_json(indent=2))
    with open(osp.join(preprocess, "compendium_cross_check.json"), "w") as handle:
        handle.write(check.model_dump_json(indent=2))
    return report


# --------------------------------------------------------------------------- #
# Command line: the recorded retrieval and the report the dendron note quotes
# --------------------------------------------------------------------------- #
def loader_inventory(
    data_root: str | None = None,
) -> tuple[Inventory, IdentifierReport]:
    """The loader's sample, gene and record accounting off the pinned files, and the
    ECK crosswalk as the derived artifact it is (#1057: no record reads it).
    """
    root = str(_data_root(data_root))
    s1 = loader_source_path("si1.xlsx", root)
    specs = classify_samples(
        pd.read_excel(s1, sheet_name=DATA_SET_S1_EXPERIMENTS_SHEET),
        pd.read_excel(s1, sheet_name=DATA_SET_S1_QUALITY_SHEET),
    )
    fitness = pd.read_csv(raw_path(RELEASE_FITNESS, root), sep="\t")
    mg1655 = bacterial_genome("ecoli", "MG1655", root)
    bw25113 = bacterial_genome("ecoli", REFERENCE_STRAIN, root)
    if not isinstance(mg1655, EcoliK12MG1655Genome) or not isinstance(
        bw25113, EcoliK12BW25113Genome
    ):
        raise TypeError("the K-12 genomes resolved to the wrong classes")
    b_numbers = [str(b) for b in fitness["sysName"]]
    _, identifiers = map_genes(b_numbers, mg1655, bw25113, label="inventory")
    return inventory(specs, len(b_numbers)), identifiers


def all_sample_comparison(data_root: str | None = None) -> ReleaseComparison:
    """:func:`compare_fitness_tables` with EVERY sample of this release on the left.

    The all-sample table (``fit_logratios.tab``) holds the samples this paper failed,
    so the 92 samples the compendium carries are all compared, ``set2IT045`` included.
    """
    return compare_fitness_tables(
        pd.read_csv(raw_path(RELEASE_FITNESS_ALL, data_root), sep="\t"),
        pd.read_csv(raw_path(SUPERSET_FITNESS, data_root), sep="\t"),
    )


def _comparison_summary(comparison: ReleaseComparison) -> dict[str, Any]:
    """The counts and the per-sample agreement range of one release comparison."""
    worst = max(comparison.agreements, key=lambda a: a.max_abs_difference)
    return {
        "n_source_genes": comparison.n_source_genes,
        "n_superset_genes": comparison.n_superset_genes,
        "n_source_genes_in_superset": comparison.n_source_genes_in_superset,
        "n_shared_samples": len(comparison.agreements),
        "min_pearson_r": comparison.min_pearson_r,
        "median_pearson_r": comparison.median_pearson_r,
        "max_abs_difference": worst.max_abs_difference,
        "max_abs_difference_sample": worst.name,
        "samples_below_replicate_bar": [
            a.name
            for a in comparison.agreements
            if a.pearson_r < COMPENDIUM_MIN_PEARSON_R
        ],
    }


def report(data_root: str | None = None) -> dict[str, Any]:
    """Every number the dendron note states, recomputed from the pinned files."""
    record = subsumption_record(data_root)
    counts, _ = loader_inventory(data_root)
    return {
        "condition_samples": len(record.experiments),
        "successful": len(record.successful),
        "carried_by_superset": len(record.carried),
        "covered": len(record.covered),
        "not_covered": list(record.not_covered),
        "disregarded": list(record.disregarded),
        "superset_samples": len(record.superset_experiment_names),
        "superset_only": len(record.superset_only),
        "verdict_flips": [
            {
                "name": flip.name,
                "short": flip.short,
                "source_failed": list(flip.source.failed_rules()),
                "superset_failed": list(flip.superset.failed_rules()),
                "source_g_med": flip.source.g_med,
                "superset_g_med": flip.superset.g_med,
                "source_cor12": flip.source.cor12,
                "superset_cor12": flip.superset.cor12,
            }
            for flip in record.verdict_flips
        ],
        "comparison": _comparison_summary(release_comparison(data_root)),
        "comparison_all_samples": _comparison_summary(all_sample_comparison(data_root)),
        "loader": counts.model_dump(mode="json"),
        "identifiers": release_identifier_finding(data_root).model_dump(),
    }


def main(argv: list[str] | None = None) -> None:
    """``retrieve`` / ``deposit`` the release files, print the ``report``, or ``verify``
    the built dev store (L0 to L4, with the compendium cross-check).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    retrieve = commands.add_parser("retrieve", help="fetch the release files")
    retrieve.add_argument("--dest", required=True)
    deposit = commands.add_parser("deposit", help="deposit fetched files")
    deposit.add_argument("--source-dir", required=True)
    commands.add_parser("report", help="print the evidence as JSON")
    commands.add_parser("verify", help="verify the built dev store (L0 to L4)")
    args = parser.parse_args(argv)
    if args.command == "retrieve":
        print(retrieve_raw_files(args.dest))
    elif args.command == "deposit":
        print(deposit_raw_mirror(source_dir=args.source_dir))
    elif args.command == "report":
        print(json.dumps(report(), indent=2))
    else:
        root = str(_data_root(None))
        print(verify_build(osp.join(root, DATASET_ROOT_REL), data_root=root).summary())


if __name__ == "__main__":
    main()
