# torchcell/datasets/ecoli/wetmore2015
# [[torchcell.datasets.ecoli.wetmore2015]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/wetmore2015
# Test file: tests/torchcell/datasets/ecoli/test_wetmore2015.py
"""Wetmore 2015 RB-TnSeq: a provenance record subsumed by the Price 2018 E. coli compendium.

Wetmore et al. 2015 (mBio 6:e00306-15, doi:10.1128/mBio.00306-15, citation key
``wetmoreRapidQuantificationMutant2015``) is the RB-TnSeq method paper, rank 2 of the
fifty bacterial rows. Its E. coli arm is the BW25113 transposon library KEIO_ML9, grown
in 101 condition samples of which 92 passed the paper's quality rules (Data Set S1).

DECISION: SUBSUMED, NOT LOADED (checklist item 7 of [[plan.bacteria-ontology-genome]]).
This module registers no dataset. The Price 2018 compendium (rank 21,
``priceMutantPhenotypesThousands2018``) is the superset that gets loaded, and this module
is the provenance record naming which of its E. coli (``orgId`` ``Keio``) samples this
paper first reported. :func:`subsumption_record` and :func:`release_comparison` measure
the evidence from sha256-pinned files; the counts live in the dendron note and are
asserted by the data-gated tests:

* Price 2018 Supplementary Table 5 lists the compendium's 162 successful ``Keio``
  experiments. 92 of this paper's 101 condition samples are among them under the same
  sample name, condition, concentration and medium: 91 of its 92 successes, plus
  ``set2IT045`` (LB), which failed this paper's cor12 rule and passes the compendium's
  re-analysis. The one success the compendium drops, ``set1IT029`` (fumarate), sat
  exactly at the gMed >= 50 rule and reads 48 in the re-analysis. Its replicate
  ``set1IT030`` failed both releases, so the compendium carries no E. coli fumarate
  sample: one condition (3,646 gene values at 1.0.0) is lost by not loading this
  release, and is accepted, since the authors' own later analysis rejects it.
* The compendium's values are a re-analysis of the same samples (statistics version 1.0.3
  against this paper's 1.0.0) over a gene set containing every one of this paper's
  genes. Loading both releases would store each shared sample twice.
* The authors' 2021 note says to disregard the E. coli sucrose and D-mannitol data (bad
  stock solutions). ``set1IT007``, ``set1IT008`` (sucrose) and ``set1IT043``,
  ``set1IT044`` (D-mannitol) are carried by the compendium and are flagged
  ``disregarded_by_authors`` for its loader.

What this module provides, because every RB-TnSeq row defers to this paper:

* the RB-TnSeq statistics as ``SourcedValue`` constants, each a verbatim quote of the
  pinned paper OCR or of the paper's own data-release page: strain and gene fitness, the
  moderated t statistic, the per-gene standard error, what one sample and one replicate
  are, and the quality rules;
* the identifier finding the compendium loader needs (:func:`identifier_reconciliation`):
  the release's gene table names MG1655 b-numbers at MicrobesOnline scaffold 7023
  coordinates for a BW25113 library;
* the raw mirror of the release files the evidence reads (:func:`deposit_raw_mirror`).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from torchcell.data import check_manifest_pin, verify_sha256, write_verified
from torchcell.datamodels.schema import MeasurementType, UncertaintyType
from torchcell.datasets.bacteria_common import (
    EckCrosswalk,
    bacterial_genome,
    eck_crosswalk,
    reconcile_locus_tags,
)
from torchcell.literature.manifest import (
    ROLE_RAW_DATA,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
    sha256_file,
)
from torchcell.literature.retrieve import direct_url
from torchcell.sequence.genome.ecoli.k12 import (
    EcoliK12BW25113Genome,
    EcoliK12Genome,
    EcoliK12MG1655Genome,
)
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
)

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
RAW_RETRIEVED_AT = "2026-10-07"
RELEASE_PAGE = "data/rbarseq/html/Keio/index.html"
RELEASE_FITNESS = "data/rbarseq/html/Keio/fit_logratios_good.tab"
RELEASE_GENES = "data/rbarseq/html/Keio/fit_genes.tab"
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
        ),
        RawFile(
            relpath=RELEASE_FITNESS,
            url=f"{_RBARSEQ}/html/Keio/fit_logratios_good.tab",
            sha256="f65a093b7fba660f12a894200073e3d731cd0b8a3f313205a5760584f4e55935",
            purpose="this paper's gene fitness for its successful samples (1.0.0)",
        ),
        RawFile(
            relpath=RELEASE_GENES,
            url=f"{_RBARSEQ}/html/Keio/fit_genes.tab",
            sha256="3eae36f697a32008673a03f6bcbca0d1b0581aa980175264c55e698a93a9ef73",
            purpose="the gene table both releases key on: b-number, symbol, scaffold",
        ),
        RawFile(
            relpath=SUPERSET_TOP_PAGE,
            url=f"{_BIGFIT}/",
            sha256="c4819e452ab54dfd3748242a3d4cf0f04021071aefadb2f07f2806afa32e3555",
            purpose="the compendium page carrying the 2021 sucrose / D-mannitol note",
        ),
        RawFile(
            relpath=SUPERSET_PAGE,
            url=f"{_BIGFIT}/html/Keio/",
            sha256="f05323ffce7701df45ddac9aebc2bab3838b111899b584838d90af0deee23803",
            purpose="the compendium's Keio release page: statistics version 1.0.3",
        ),
        RawFile(
            relpath=SUPERSET_QUALITY,
            url=f"{_BIGFIT}/html/Keio/fit_quality.tab",
            sha256="7d4b15733d8e258628a8f71831867c2d72294b82a0afca7ad53a6a9a603336e5",
            purpose="the compendium's per-sample quality verdicts",
        ),
        RawFile(
            relpath=SUPERSET_FITNESS,
            url=f"{_BIGFIT}/html/Keio/fit_logratios_good.tab",
            sha256="b37f038702ef0b792bdc02af9f42adf3a5838bc08513249c89983f6a7647e8fd",
            purpose="the compendium's gene fitness for its successful samples (1.0.3)",
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
            retrieved=RAW_RETRIEVED_AT,
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
#: Data Set S1 ``Condition_1`` values the 2021 note covers.
DISREGARDED_CONDITIONS: frozenset[str] = frozenset(STOCK_SOLUTION_CORRECTION.value)

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


# --------------------------------------------------------------------------- #
# Paths and pins
# --------------------------------------------------------------------------- #
def _data_root(data_root: str | None) -> Path:
    """``data_root`` when given, else ``$DATA_ROOT``."""
    return Path(data_root if data_root is not None else os.environ["DATA_ROOT"])


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


def _raw_manifest(root: Path, retrieved_at: str, created_at: str) -> Manifest:
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
                    retrieved_at=retrieved_at,
                ),
            )
            for raw in RAW_FILES.values()
        ],
        si_data_sources=[f"{_RBARSEQ}/", f"{_BIGFIT}/"],
        si_expected=[
            "the per-strain tables (strain_fit.tab, all.poolcount) and the R image "
            "fit.image of both releases are NOT mirrored: the provenance record reads "
            "gene-level tables only, and the compendium loader mirrors what it loads"
        ],
        provenance_complete=True,
        created_at=created_at,
    )


def deposit_raw_mirror(
    *,
    source_dir: str | Path,
    retrieved_at: str = RAW_RETRIEVED_AT,
    data_root: str | None = None,
) -> Path:
    """Copy the retrieved release files into the raw mirror and write its manifest.

    Every source file is checked against its pin, and every file already in the mirror
    against the same pin, BEFORE anything is written, so a refusal leaves the mirror as
    it was. Idempotent by sha256: a matching mirror file is left alone, a differing one
    raises. An existing manifest is kept when it records the same files and refused when
    it records others; it is never overwritten.
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
    manifest = _raw_manifest(root, retrieved_at, datetime.now(UTC).isoformat())
    if existing is not None:
        if existing.files != manifest.files:
            raise RuntimeError(
                f"{manifest_path} records other files or retrievals; refusing"
            )
        return root
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
                disregarded_by_authors=condition in DISREGARDED_CONDITIONS,
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
# Command line: the recorded retrieval and the report the dendron note quotes
# --------------------------------------------------------------------------- #
def report(data_root: str | None = None) -> dict[str, Any]:
    """Every number the dendron note states, recomputed from the pinned files."""
    record = subsumption_record(data_root)
    comparison = release_comparison(data_root)
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
        "comparison": {
            "n_source_genes": comparison.n_source_genes,
            "n_superset_genes": comparison.n_superset_genes,
            "n_source_genes_in_superset": comparison.n_source_genes_in_superset,
            "n_shared_samples": len(comparison.agreements),
            "min_pearson_r": comparison.min_pearson_r,
            "median_pearson_r": comparison.median_pearson_r,
            "max_abs_difference": max(
                a.max_abs_difference for a in comparison.agreements
            ),
        },
        "identifiers": release_identifier_finding(data_root).model_dump(),
    }


def main(argv: list[str] | None = None) -> None:
    """``retrieve`` the release files, ``deposit`` them, or print the ``report``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    retrieve = commands.add_parser("retrieve", help="fetch the release files")
    retrieve.add_argument("--dest", required=True)
    deposit = commands.add_parser("deposit", help="deposit fetched files")
    deposit.add_argument("--source-dir", required=True)
    commands.add_parser("report", help="print the evidence as JSON")
    args = parser.parse_args(argv)
    if args.command == "retrieve":
        print(retrieve_raw_files(args.dest))
    elif args.command == "deposit":
        print(deposit_raw_mirror(source_dir=args.source_dir))
    else:
        print(json.dumps(report(), indent=2))


if __name__ == "__main__":
    main()
