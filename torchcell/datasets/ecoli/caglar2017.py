# torchcell/datasets/ecoli/caglar2017
# [[torchcell.datasets.ecoli.caglar2017]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/caglar2017
# Test file: tests/torchcell/datasets/ecoli/test_caglar2017.py
"""Caglar 2017 E. coli molecular phenotype: the RNA-seq and proteome loaders.

Caglar et al. 2017 (Scientific Reports, doi:10.1038/srep45303) measured mRNA (RNA-seq),
protein (LC-MS/MS) and 13C central-carbon flux ratios of one wild-type strain across 34
conditions: four carbon sources, Mg2+ and Na+ series, exponential and stationary phase,
and two starvation time courses. The processed data are the paper's Supplementary
Tables S1 (sample sheet), S2 (normalized mRNA), S3 (normalized protein) and S4 (flux
ratios).

STRAIN. ``REL606``, an *E. coli* **B** strain (``STRAIN``, ``LINEAGE``), and the paper's
identifiers are REL606 identifiers: Table S2 is keyed by ``ECB_`` locus tags of GenBank
CP000819.1 and Table S3 by retired RefSeq ``YP_`` protein accessions of NC_012967.1
(``IDENTIFIER_FORMS``). ``require_pinnable_strain`` is the gate each loader calls first;
it returns now that ``REL606`` is in ``BacterialReferenceStrain`` (assembly set
``ecoli_B_REL606_ASM1798v1``) and raises ``UnpinnedStrainError`` carrying ``STRAIN_GAP``
against a vocabulary without it. ``REL606_TIER_ADDITION`` records the tier addition
that opened it.

RAW MIRROR. ``deposit_raw_mirror`` writes ``$DATA_ROOT/torchcell-raw/
caglarColiMolecularPhenotype2017/`` from bytes produced by the recorded retrievers
(``raw_file_specs``): Tables S1 to S4 from the PMC Article Datasets bucket, and the NCBI
protein records of every ``YP_`` accession in Table S3, the only scriptable bridge from the
protein table to ``ECB_`` locus tags. ``identifier_coverage`` re-measures that bridge.

THE LOG BASE, BACK-SOLVED. The paper says the tables were "normalized, and
log-transformed" and never names the base (``LOG_TRANSFORMED``). ``back_solve_counts``
settles it from the released numbers: every cell of Table S2 (4,196 x 152) and Table S3
(4,196 x 105) inverts, through ``q = (2**y - P)**2 / 2**y`` with ``P = 2**min(table)``, to
an integer count times one per-sample factor, and that factor equals DESeq2's
median-of-ratios size factor computed on the reconstructed counts plus the +1
pseudocount the Methods state (``SIZE_FACTOR_PSEUDOCOUNT``). That inverse is the one of
DESeq2's parametric variance-stabilizing transform, which is ``log2`` of the normalized
count for large counts, so the base is 2; base e and base 10 leave half the cells 0.5 off
an integer. The build refuses a table whose reconstruction misses an integer by more than
``COUNT_INTEGER_TOLERANCE`` or a size factor by more than ``SIZE_FACTOR_TOLERANCE``, and
writes the evidence (``VstBackSolve``) and a ``StatDerivation`` of the base.

RNA-SEQ (``RnaseqCaglar2017Dataset``, ``RNASeqExpressionPhenotype``). One record per
Table S2 sample, i.e. per biological-replicate library (152). ``expression_count`` is the
reconstructed HTSeq read count of each protein-coding gene (``READ_COUNTING``,
``CODING_READS_ONLY``); ``expression_tpm`` is the TPM of those counts over the 4,196
genes, with each gene's length the span of its GenBank gene feature in the REL606
genome. The TPM is computed here: the paper reports DESeq2-normalized values, not TPM.

PROTEOME (``ProteomeCaglar2017Dataset``, ``ProteinAbundancePhenotype``). One record per
Table S3 sample (105). ``protein_abundance`` is the DESeq2 size-factor-normalized
spectral count, the reconstructed integer count divided by the sample's size factor (the
"Normalized protein counts" the SI names), keyed by the ``ECB_`` tag the deposited NCBI
record of each ``YP_`` accession names. An unobserved protein is a 0 the source wrote
(``UNOBSERVED_PROTEINS``). ``n_replicates`` is 1: a record is one biological culture.

GENOTYPE AND ENVIRONMENT. Every record is wild-type REL606 (reference-only, an empty
``Genotype``); the conditions are environment edits on Davis Minimal medium, read from
the Table S1 row of the sample: ``DM500`` for glucose, ``DAVIS_MINIMAL`` plus a
``carbon_source`` factor at 0.5 g/L for glycerol, lactate and gluconate; a magnesium
sulfate level other than the base as a ``SmallMoleculePerturbation`` at the stated final
concentration; NaCl added to reach a Na+ level above the ~5 mM base; 37 C, shaken
flasks; ``duration_hours`` is the sample's ``growthTime_hr`` (the time it was collected).
The growth phase (exponential, stationary, late stationary) has no slot on
``Environment``; it selects the record's reference and is kept in the ledgers.

REFERENCE. "The reference conditions always had glucose as carbon source and base Na+ and
Mg2+ concentrations", one per phase (``REFERENCE_CONDITIONS``). Each record's reference
is that condition in the record's phase (the late-stationary one applies the same rule to
the third phase), with the phenotype averaged over the condition's samples.

PER-RECORD ATTRIBUTION (#771). 27 of the 152 mRNA samples and 27 of the 105 protein
samples are not this paper's measurements: they are Houser 2015's glucose time course,
released again here. Caglar says so three times, and each sentence is quoted from the
pinned OCR: the Results ("Results from one of these conditions, long-term glucose
starvation, have been presented previously10", ``HOUSER2015_DEFERRAL``), reference 10
itself (``HOUSER2015_CITATION``), and the Data availability sentence, which splits the
deposits along the same line ("accession GSE67402 for the glucose time-course previously
published10, accession GSE94117 for all other experiments", ``HOUSER2015_DEPOSITS``).
Every record therefore stores the ``Publication`` of the study that FIRST reported its
sample (``SOURCE_STUDIES``, ``attribute_sample``), which is the Borchert 2024 pattern
already set on this program; the split is Table S1's own ``experiment`` column, and the
ledger goes to ``preprocess/source_study_attribution.json``.

HOUSER 2015 IS NOT MIRRORED (``HOUSER2015_IS_MIRRORED``), has no raw mirror and no
loader, so nothing in the graph stores those measurements twice and the paper itself is
unread here: the attribution is sourced entirely from Caglar's own citation of it. That
citation gives journal, volume and article id but no DOI, so ``HOUSER2015_DOI`` records
how the DOI string was fixed and the check that it names the right record.

PROTEIN FOLD CHANGE (``ProteinFoldChangeCaglar2017Dataset``,
``ProteinFoldChangePhenotype``). The protein arm of Supplementary Table S8
(``srep45303-s9.csv``, retrieved from the same PMC Article Datasets bucket as Tables S1
to S4), one record per differential-proteomics CONTRAST rather than per sample: a
different estimand from the proteome loader's absolute levels. Table S8 holds 48 groups
of 4,196 rows, 24 of them protein-level, which are 6 contrasts x 2 growth phases x 2
CONTROL PARAMETERIZATIONS (``TABLE_S8_CONTROL_PARAMETERS``). The loader keeps the
batch-only groups, which is the paper's primary design (``PRIMARY_DESIGN``), and refuses
the doubling-time ones (``DOUBLING_TIME_DESIGN``): those differ in no released
experimental column, so they are a second estimate of one genotype x environment cell.
Of the 12 remaining, the two ``highMg`` and the two ``highNa`` groups are refused as
well, because Table S1 puts several doses in each of those levels (MgSO4 at 8 and
200 mM; Na+ at 100, 200 and 300 mM) and a ``SmallMoleculePerturbation`` takes one
``Concentration``. The 8 records store ``log2FoldChange`` with ``lfcSE``, ``pvalue`` and
``padj`` on ``fold_change_scale=log2``, keyed by the ``ECB_`` tags the GenPept bridge
gives; ``p_value_adjustment_method`` is back-solved from the released p-values
(``adjustment_back_solve``), not stated. ``n_replicates`` is the contrast's TEST-group
protein-sample count from Table S1, the conservative lower end
(``replicate_derivations``). The reference phenotype is the log2 scale's neutral value for
every stored key, which is what a fold change's denominator carries by definition. The
gene-level arm's 100,704 rows are out of scope: no gene-level expression fold-change
phenotype exists.

The flux arm (Table S4) is not loaded: it holds flux RATIOS, which neither
``MetabolitePhenotype`` (pool sizes) nor ``FluxPhenotype`` (signed net flux) can store.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import math
import os
import os.path as osp
import pickle
import re
import shutil
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar, Literal, get_args

import numpy as np
import pandas as pd
from Bio import SeqIO
from pydantic import BaseModel, ConfigDict, Field

from torchcell.data import (
    ExperimentDataset,
    check_manifest_pin,
    link_verified,
    post_process,
    verify_raw_files,
)
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import DAVIS_MINIMAL, DM500
from torchcell.datamodels.schema import (
    BACTERIAL_LOCUS_TAG_PATTERNS,
    BacterialProteinAbundanceExperiment,
    BacterialProteinAbundanceExperimentReference,
    BacterialProteinFoldChangeExperiment,
    BacterialProteinFoldChangeExperimentReference,
    BacterialReferenceStrain,
    BacterialRNASeqExpressionExperiment,
    BacterialRNASeqExpressionExperimentReference,
    Concentration,
    ConcentrationUnit,
    Environment,
    EnvironmentPerturbationType,
    EnvironmentPhysicalPerturbation,
    Experiment,
    ExperimentReference,
    FoldChangeScale,
    Genotype,
    PhysicalFactor,
    ProteinAbundancePhenotype,
    ProteinFoldChangePhenotype,
    Publication,
    RNASeqExpressionPhenotype,
    SmallMoleculePerturbation,
    Temperature,
)
from torchcell.datasets.bacteria_common import (
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
    sha256_file,
)
from torchcell.literature.provenance import run_retriever
from torchcell.literature.retrieve import pmc_cloud_url
from torchcell.sequence import GeneSet
from torchcell.sequence.genome.ecoli.rel606 import EcoliBREL606Genome
from torchcell.verification.levels import l0_structural, l1_count, l2_value_fidelity
from torchcell.verification.report import (
    DerivationMethod,
    Level,
    LevelResult,
    Provenance,
    StatDerivation,
    VerificationReport,
)
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
    audit_sourced_value,
)

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Provenance anchors
# --------------------------------------------------------------------------- #
CITATION_KEY = "caglarColiMolecularPhenotype2017"
PAPER_DOI = "10.1038/srep45303"
PAPER_TITLE = "The E. coli molecular phenotype under different growth conditions"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"

PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "0878d5e7d49bcea4570aa2db225318a8469f4755645f1ddf73563effe7b3109b"
SI1_MD = "si/si1.md"
SI1_MD_SHA256 = "1b7b8ed0f8b21c1909568f8a05d6a92d99bffaf851aa474a3a8673ebc2da4bdc"

#: The article version in the PMC Article Datasets bucket that serves the SI tables.
PMC_ARTICLE = "PMC5394689.1"
#: The date the raw-mirror bytes were produced by the recorded retrievers.
RAW_RETRIEVED_AT = "2026-10-07"
#: Per-table retrieval date, where it is not ``RAW_RETRIEVED_AT``. Tables S5 and S8 were
#: fetched for the doubling-time and fold-change loaders, which the first deposit did not
#: need (``deposit_si_table``), so their records carry their OWN date.
TABLE_RETRIEVED_AT: dict[str, str] = {"S5": "2026-10-09", "S8": "2026-10-09"}

_OCR_METHOD = "MinerU OCR of the publisher PDF (torchcell-library mirror)"


def _paper(
    value: Any, quote: str, *, page: str, note: str | None = None
) -> SourcedValue:
    """Bind a value to a verbatim quote in the sha256-pinned paper OCR."""
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


def _si(value: Any, quote: str, *, note: str | None = None) -> SourcedValue:
    """Bind a value to a verbatim quote in the OCR of the Supplementary Information PDF."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=SI1_MD,
            citation_key=CITATION_KEY,
            sha256=SI1_MD_SHA256,
            method=_OCR_METHOD,
            page="List of Supplementary Tables",
        ),
    )


# --------------------------------------------------------------------------- #
# The strain, the reference the authors mapped to, and the identifier forms
# --------------------------------------------------------------------------- #
STRAIN = _paper(
    "REL606",
    "E. coli B REL606 was inoculated from a freezer stock",
    page="Methods, Cell Growth",
    note="every sample comes from this one strain: 'We grew multiple cultures of E. "
    "coli REL606, from the same stock' (Results) and 'used the exact same $E$ . coli "
    "genotype throughout' (Discussion)",
)
LINEAGE = _paper(
    "E. coli B",
    "the REL606 Escherichia coli B genome",
    page="Methods, RNA-seq",
    note="a B strain, not K-12: no deposited assembly set (MG1655, BW25113) is its genome",
)
SAME_STOCK = _paper(
    True,
    "We grew multiple cultures of E. coli REL606, from the same stock, under a variety "
    "of different growth conditions.",
    page="Results, Experimental design and data collection",
)
SAME_GENOTYPE = _paper(
    True, "used the exact same $E$ . coli genotype throughout", page="Discussion"
)
RNASEQ_REFERENCE = _paper(
    "NC_012967.1",
    "we implemented a custom analysis pipeline using the REL606 Escherichia coli B "
    "genome (GenBank:NC_012967.1) as the reference sequence",
    page="Methods, RNA-seq",
    note="NC_012967.1 is the RefSeq copy of GenBank CP000819.1, the one replicon of "
    "assembly GCA_000017985.1 / GCF_000017985.1 (ASM1798v1)",
)
PROTEOME_REFERENCE = _paper(
    "REL606 protein sequence database",
    "Spectra were searched against an $E _ { \\ast }$ . coli strain REL606 protein "
    "sequence database",
    page="Methods, Proteomics",
)
IDENTIFIER_FORMS = _si(
    {"mrna": "ECB_", "protein": "YP_"},
    "Gene id (ECB number for mRNA and YP number for proteins), and corresponding gene "
    "name",
    note="stated for Table S8; Tables S2 and S3 carry the same two forms in their first "
    "column (identifier_coverage measures 4,196 ECB_ and 4,196 YP_ ids)",
)
MATCHED_GENES = _paper(
    4196,
    "This resulted in 4196 matching mRNA and protein counts for each sample.",
    page="Methods, Normalization and quality control of RNA and protein counts",
)

# --------------------------------------------------------------------------- #
# Replicate structure (what a loader's n_samples and SampleUnit come from)
# --------------------------------------------------------------------------- #
BIOLOGICAL_REPLICATES = _paper(
    3,
    "For each experimental condition, bacteria were grown in three biological "
    "replicates.",
    page="Figure 1 legend",
    note="Table S2 and Table S3 are per SAMPLE (one column per biological replicate "
    "culture), not per-condition means",
)
REPLICATE_DAYS = _paper(
    "separate day",
    "Each of the three biological replicates was performed on a separate day.",
    page="Methods, Cell Growth",
)
TECHNICAL_REPLICATE_COLUMNS = _si(
    ("RNA_Data_Freq", "Protein_Data_Freq"),
    "number of RNA samples (technical replicates), number of protein samples "
    "(technical replicates)",
    note="the Table S1 columns that count technical replicates per sample; measured "
    "on the mirror: RNA 1 for all 152 RNA samples, protein 1 for 93 and 2 for 12 of "
    "the 105 protein samples",
)
FLUX_REPLICATES = _paper(
    3,
    "For each condition, flux samples were analyzed in triplicate (except one, which "
    "was analyzed in duplicate only), and 13 different flux ratios were measured for "
    "each sample.",
    page="Results, Metabolic flux ratios under salt stress",
    note="the lower end of the stated structure is 2; which condition had 2 is not "
    "named in the mirror (a gap for the flux record's n_samples)",
)
FLUX_AVERAGED = _paper(
    "mean over replicates",
    "The flux ratios were then averaged across replicates",
    page="Results, Metabolic flux ratios under salt stress",
)
FLUX_TECHNICAL_INJECTIONS = _paper(
    3, "three technical replicates of each vial", page="Methods, Flux analysis"
)
DOUBLING_TIME_REPLICATES = _paper(
    3,
    "Means and confidence intervals were calculated from three replicate growth "
    "curves for all conditions except for gluconate and lactate, which had "
    "measurements for only two replicates.",
    page="Methods, Cell Growth",
)

# --------------------------------------------------------------------------- #
# The data home and how the values were processed
# --------------------------------------------------------------------------- #
PROCESSED_TABLES = _paper(
    ("S2", "S3", "S4"),
    "final processed data are available as Supplementary\u00a0Tables\u00a0S2,\u00a0S3\u00a0"
    "and\u00a0S4.",
    page="Results, Experimental design and data collection",
    note="the OCR separates these words with no-break spaces (U+00A0), kept verbatim",
)
NORMALIZATION = _paper(
    "DESeq2 size factors, then log-transformed",
    "we normalized read counts using size-factors calculated via $\\mathrm { D E S e q "
    "} 2 ^ { 1 7 }$",
    page="Methods, Normalization and quality control of RNA and protein counts",
    note="'All resulting data sets were checked for quality, normalized, and "
    "log-transformed.' (Results). The base of the log is not stated in the mirror and "
    "the Methods defer to ref 10, which is not mirrored; back_solve_counts settles it "
    "from the released numbers as 2 (the DESeq2 parametric variance-stabilizing "
    "transform)",
)
TABLE_S2 = _si(
    (4196, 152),
    "Supplementary Table S2: Normalized mRNA counts. Includes data for 4196 distinct "
    "proteins each for 152 samples.",
)
TABLE_S3 = _si(
    (4196, 105),
    "Supplementary Table S3: Normalized protein counts. Includes data for 4196 "
    "distinct proteins each for 105 samples.",
)
TABLE_S4 = _si(
    13,
    "Supplementary Table S4: Mean flux ratios for 13 branches each, measured for "
    "varying Mg $^ { 2 + }$ and Na $^ +$ concentrations in exponential and stationary "
    "phase.",
)
TABLE_S8 = _si(
    "Combined results from tests for differential expression",
    "Supplementary Table S8: Combined results from tests for differential expression "
    "for all genes and all distinct tests considered.",
    note="measured on the mirror: 201,408 rows x 24 columns, 100,704 mRNA rows keyed by "
    "ECB_ locus tag and 100,704 protein rows keyed by YP_ accession, as 48 groups of "
    "exactly 4,196 rows, one per fullFileName",
)
TABLE_S8_COLUMNS = _si(
    ("baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"),
    "Results from DeSeq2 calculation, including base mean value, log2FoldChange, "
    "ifcSE, stat, pvalue, padj.",
    note="the SI writes 'ifcSE'; the column in the released file is 'lfcSE'",
)
TABLE_S8_CONTROL_PARAMETERS = _si(
    ("batch only", "batch plus doubling time"),
    "Control parameters of the test (batch only or batch plus doubling time)",
    note="the investigatedEffect column: the test's STATISTICAL control, not an "
    "experimental factor, so the two parameterizations of one contrast are two "
    "estimates of one genotype x environment cell and not two records",
)
PRIMARY_DESIGN = _paper(
    "~batch_number + variable_of_interest",
    "In general, our design formula was \\~batch_number $^ +$ variable_of_interest, "
    "where variable_of_interest was either a categorical variable representing the "
    "carbons source or growth phase (exponential or stationary) or a quantitative "
    "variable representing",
    page="Methods, Identifying differentially expressed genes",
    note="the paper's primary design; the loader keeps the batch-only groups of "
    "Table S8 and refuses the doubling-time ones (REFUSED_CONTROL_MODEL)",
)
DOUBLING_TIME_DESIGN = _paper(
    "~batch_number + doubling_time + variable_of_interest",
    "We repeated our DeSeq2 analyses but included in our design formula a term "
    "representing the doubling time (see Methods).",
    page="Results, Differentially expressed genes and growth rate",
    note="a REPEAT of the same analyses under a second control model; the released "
    "columns of the two sets of groups differ only in investigatedEffect",
)
FOLD_CHANGE_BASIS = _paper(
    "glucose, 5 mM Na+, 0.8 mM Mg2+, same growth phase",
    "For each growth phase, we defined the base level reference condition to be growth "
    "in glucose with $5 \\mathrm { m M }$ $\\mathrm { N a ^ { + } }$ and "
    "$0 . 8 \\bar { \\mathrm { m M } } \\mathrm { M g } ^ { 2 + }$ .",
    page="Results, Identification of differentially expressed genes",
    note="what reference_basis names: the DENOMINATOR of every Table S8 fold change",
)
FDR_CORRECTED = _paper(
    "FDR-corrected P value",
    "at a false-discovery-rate (FDR) corrected $P$ value $< 0 . 0 5$",
    page="Results, Identification of differentially expressed genes",
    note="the correction is named only as FDR; which FDR procedure produced padj is "
    "back-solved from the released p-values (adjustment_back_solve) as "
    "Benjamini-Hochberg, DESeq2's default",
)
BATCH_IN_DESIGN = _paper(
    "batch number is a predictor",
    "We corrected for possible batch effects by including batch number as a predictor "
    "variable in the design formula of DESeq2.",
    page="Methods, Identifying differentially expressed genes",
)
GEO_ACCESSION = _paper(
    "GSE94117",
    "accession GSE94117 for all other experiments",
    page="Methods, Statistical analysis and data availability",
)
PRIDE_ACCESSION = _paper(
    "PXD005721",
    "accession PXD005721 for all other experiments",
    page="Methods, Statistical analysis and data availability",
)

# --------------------------------------------------------------------------- #
# Environment (for the loader the gate is waiting on)
# --------------------------------------------------------------------------- #
BASE_MEDIUM = _paper(
    "DM500",
    "Davis Minimal medium supplemented with $2 \\mu \\mathrm { g } / 1$ thiamine $( "
    "\\mathrm { D M } ) ^ { 3 6 }$ and limiting glucose at $5 0 0 \\mathrm { m g / l }$ "
    "(DM500)",
    page="Methods, Cell Growth",
    note="MEDIA_LIBRARY key DM500 (base DAVIS_MINIMAL); the DM salts defer to ref 36, "
    "Lenski 1991",
)
CARBON_SOURCE_SWAP = _paper(
    0.5,
    "the Davis Minimal (DM) medium used was supplemented with $0 . 5 { \\mathrm { g } } "
    "/ { \\mathrm { L } }$ of the specified compound (glycerol, lactate, or gluconate) "
    "instead of glucose.",
    page="Methods, Cell Growth",
    note="g/L of the replacing carbon source",
)
MAGNESIUM_SERIES = _paper(
    0.83,
    "$\\mathbf { M } \\mathbf { g } ^ { 2 + }$ concentrations were varied by changing "
    "the amount of $\\mathrm { M g S O _ { 4 } }$ added to DM media from the concentration "
    "of $0 . 8 3 \\mathrm { m M }$ that is normally present.",
    page="Methods, Cell Growth",
    note="mM MgSO4 in DM; Table S1's Mg_mM column states 0.8 for the base level",
)
SODIUM_SERIES = _paper(
    5,
    "The base recipe for DM already contains ${ \\sim } 5 \\mathrm { m M N a ^ { + } }$ "
    "due to the inclusion of sodium citrate",
    page="Methods, Cell Growth",
    note="mM Na+ at base; NaCl is added to reach each higher level",
)
NACL_ADDITION = _paper(
    95,
    "so $9 5 \\mathrm { m M N a C l }$ was added for the $1 0 0 \\mathrm { m M N a ^ { + } }$ "
    "condition, for example",
    page="Methods, Cell Growth",
    note="mM NaCl added = the Na+ level less the ~5 mM base; the loader applies the "
    "stated arithmetic to the 200 and 300 mM levels (195 and 295 mM added)",
)
CULTURE_CONDITIONS = _paper(
    37.0,
    "This culture was incubated at $3 7 ^ { \\circ } \\mathrm { C }$ with 120 r.p.m. "
    "orbital shaking",
    page="Methods, Cell Growth",
    note="degrees Celsius; 50 ml cultures in 250 ml flasks, orbitally shaken, so the "
    "oxygen regime is recorded as aerobic",
)
EXPONENTIAL_SAMPLING = _paper(
    "20-60% of maximal OD600",
    "Exponential-phase samples were taken during growth when the $\\mathrm { O D } _ { 6 "
    "0 0 }$ reached $2 0 { - } 6 0 \\%$ of the maximum achieved after saturating growth.",
    page="Methods, Cell Growth",
    note="the phase is set by an optical density, so a phase's samples are collected "
    "at different times",
)
STATIONARY_SAMPLING = _paper(
    "20-24 h after the exponential sample",
    "Stationary phase samples were collected 20–24 hours after the corresponding "
    "exponential sample.",
    page="Methods, Cell Growth",
)
SAMPLING_TIMES = _paper(
    "Table S1 growthTime_hr",
    "The exact sampling times for each condition are provided in "
    "Supplementary Table S1.",
    page="Methods, Cell Growth",
    note="the OCR separates the last three words with no-break spaces, kept verbatim",
)
SAMPLE_TIME_COLUMN = _si(
    "growthTime_hr",
    "the growth time at which the sample was collected",
    note="Table S1's description of its growth-time column; the loader stores it as "
    "Environment.duration_hours",
)
REFERENCE_CONDITIONS = _paper(
    ("exponential", "stationary"),
    "We used two reference conditions in our comparisons, one for exponential phase and "
    "one for stationary phase. The reference conditions always had glucose as carbon "
    "source and base $\\mathrm { N a ^ { + } }$ and $\\mathbf { M } \\mathbf { g } ^ { "
    "2 + }$ concentrations.",
    page="Methods, Identifying differentially expressed genes",
    note="the loader applies the same rule to the late-stationary samples, for which "
    "the paper names no reference",
)
GLUCOSE_TIME_COURSE_PRIOR = _paper(
    "ref 10",
    "Results from one of these conditions, long-term glucose starvation, have been "
    "presented previously10.",
    page="Results, Experimental design and data collection",
    note="the glucose time-course samples (GEO GSE67402 / PRIDE PXD002140) are columns "
    "of Tables S2 and S3, processed with the rest",
)

# --------------------------------------------------------------------------- #
# How Tables S2 and S3 were made (what the back-solve inverts)
# --------------------------------------------------------------------------- #
_NORMALIZATION_PAGE = (
    "Methods, Normalization and quality control of RNA and protein counts"
)

LOG_TRANSFORMED = _paper(
    "log-transformed; base not stated",
    "All resulting data sets were checked for quality, normalized, and log-transformed.",
    page="Results, Experimental design and data collection",
    note="the base is back-solved, not stated: every cell of Tables S2 and S3 inverts "
    "exactly as the DESeq2 parametric variance-stabilizing transform, base 2 "
    "(VstBackSolve)",
)
SIZE_FACTOR_PSEUDOCOUNT = _paper(
    1,
    "Because we had many mRNAs and proteins with counts of zero at some condition, we "
    "added pseudo-counts of $+ 1$ to all counts before calculating size factors.",
    page=_NORMALIZATION_PAGE,
)
SIZE_FACTORS_ON_RAW = _paper(
    "size factors divide the raw counts",
    "We then used those size factors to normalize the original raw counts (i.e., "
    "without pseudo-counts).",
    page=_NORMALIZATION_PAGE,
)
READ_COUNTING = _paper(
    "HTSeq read counts per gene",
    "The raw number of reads mapping to each gene were counted using HTSeq",
    page="Methods, RNA-seq",
)
CODING_READS_ONLY = _paper(
    "reads overlapping protein-coding genes",
    "For RNA, we only analyzed the counts of reads that overlapped annotated protein "
    "coding genes, i.e., reads mapping to mRNAs.",
    page=_NORMALIZATION_PAGE,
    note="so a record's counts are the reads of the 4,196 coding genes, not every "
    "mapped read",
)
PROTEIN_COUNTS_FRACTIONAL = _paper(
    "spectral counts shared among proteins",
    "Protein counts can be fractional, because some peptide spectra cannot be uniquely "
    "mapped to a single protein, so they are equally divided amongst these proteins.",
    page=_NORMALIZATION_PAGE,
)
PROTEIN_COUNTS_ROUNDED = _paper(
    "rounded to integers",
    "We rounded all protein counts to the nearest integer for subsequent analysis.",
    page=_NORMALIZATION_PAGE,
)
UNOBSERVED_PROTEINS = _paper(
    0, "We set the counts of all unobserved proteins to zero.", page=_NORMALIZATION_PAGE
)
QC_FLAGGED_SAMPLES = _paper(
    ("MURI_091", "MURI_130"),
    "Out of 152 mRNA samples we found only two samples (samples MURI_091 and MURI_130, "
    "Supplementary Table S1) that seemed to deviate from their biological "
    "replicas.",
    page=_NORMALIZATION_PAGE,
)
ALL_SAMPLES_KEPT = _paper(
    True,
    "Because of this broad consistency among all samples for the same growth "
    "conditions, we keep all samples for subsequent analysis.",
    page=_NORMALIZATION_PAGE,
    note="MURI_091 and MURI_130 are records like every other sample",
)
GEO_PROCESSED_COUNTS = _paper(
    "GSE94117",
    "Raw Illumina read data and processed files of read counts per gene and normalized "
    "expression levels per gene have been deposited in the NCBI GEO database",
    page="Methods, Statistical analysis and data availability",
    note="not mirrored: Table S2 is the processed matrix the paper names, and its "
    "counts are reconstructed exactly (back_solve_counts)",
)


# --------------------------------------------------------------------------- #
# The strain gate
# --------------------------------------------------------------------------- #
class TierMember(BaseModel):
    """One file the REL606 assembly set would hold, as fetched and checked on 2026-10-07."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    url: str
    role: str = Field(description="genomes-tier role: annotation | sequence | index")
    bytes: int
    md5: str = Field(description="NCBI's md5checksums.txt value, matched on retrieval")
    sha256: str


class AssemblyTierAddition(BaseModel):
    """The genomes-tier and schema addition that would make REL606 pinnable.

    Every number here was measured on the bytes retrieved on ``retrieved_at``; the
    sha256 values are retrieval evidence, not a deposited manifest (a deposit re-fetches
    and ``deposit_assembly_set`` re-hashes).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    assembly_set: str = Field(
        description="proposed id, after ecoli_K12_MG1655_ASM584v2"
    )
    organism: str = Field(description="'# Organism name' of the assembly report")
    strain: str
    assembly_name: str
    genbank_accession: str
    refseq_accession: str
    genbank_replicon: str
    refseq_replicon: str
    replicon_length_bp: int
    gene_namespace: str = Field(description="proposed BacterialGeneNamespace member")
    locus_tag_pattern: str = Field(
        description="matches all GenBank gene features and no deposited namespace"
    )
    genbank_gene_features: int
    genbank_url: str
    refseq_url: str
    md5_checksum_files: dict[str, str] = Field(
        description="URL of each directory's md5checksums.txt -> sha256 of the copy used"
    )
    members: tuple[TierMember, ...]
    edits_needed: tuple[str, ...]
    retrieved_at: str


_NCBI = "https://ftp.ncbi.nlm.nih.gov/genomes/all"
_GCA_DIR = f"{_NCBI}/GCA/000/017/985/GCA_000017985.1_ASM1798v1"
_GCF_DIR = f"{_NCBI}/GCF/000/017/985/GCF_000017985.1_ASM1798v1"


def _member(
    directory: str, name: str, role: str, size: int, md5: str, sha: str
) -> TierMember:
    return TierMember(
        path=name, url=f"{directory}/{name}", role=role, bytes=size, md5=md5, sha256=sha
    )


REL606_TIER_ADDITION = AssemblyTierAddition(
    assembly_set="ecoli_B_REL606_ASM1798v1",
    organism="Escherichia coli B str. REL606 (E. coli)",
    strain="REL606",
    assembly_name="ASM1798v1",
    genbank_accession="GCA_000017985.1",
    refseq_accession="GCF_000017985.1",
    genbank_replicon="CP000819.1",
    refseq_replicon="NC_012967.1",
    replicon_length_bp=4629812,
    gene_namespace="ecoli_b_rel606_locus_tag",
    locus_tag_pattern=r"^ECB_[rt]?\d{5}$",
    genbank_gene_features=4383,
    genbank_url=f"{_GCA_DIR}/",
    refseq_url=f"{_GCF_DIR}/",
    md5_checksum_files={
        f"{_GCA_DIR}/md5checksums.txt": (
            "c7a6602a83d3e5573c46f8a0c8af1ff81716f686b6888cd162926afe485b578c"
        ),
        f"{_GCF_DIR}/md5checksums.txt": (
            "cff986bda5f799e5308ed7341c7af3fb74486fb163b3b486e604146b8e64eaf2"
        ),
    },
    members=(
        _member(
            _GCA_DIR,
            "GCA_000017985.1_ASM1798v1_genomic.gbff.gz",
            "annotation",
            3239604,
            "9c93575508d0eb2559106185330e483b",
            "aacf2559815f959c9417984ce1632228fd94caeac4b62b7910f714e310542e6b",
        ),
        _member(
            _GCA_DIR,
            "GCA_000017985.1_ASM1798v1_genomic.fna.gz",
            "sequence",
            1375449,
            "84ac547b7fa22c6291aa519f2d1ce444",
            "070a03fc2e2813853d5327608ee3ebcb4b0b2fe7faa239169921b3362b24adfa",
        ),
        _member(
            _GCA_DIR,
            "GCA_000017985.1_ASM1798v1_genomic.gff.gz",
            "annotation",
            270806,
            "673a2039153095efd66128386d2d9b80",
            "b928f83a99ea3ec7e64137f36490c37aa4585689de1abb884ff9cbaa4e1199d5",
        ),
        _member(
            _GCA_DIR,
            "GCA_000017985.1_ASM1798v1_protein.faa.gz",
            "sequence",
            889482,
            "732ed5bf0131047db14f5c019a8286db",
            "40f1748bf2e86f5a43d7a8bb1515a0b3812f66f27f4d7fb9dc62a0c348962663",
        ),
        _member(
            _GCA_DIR,
            "GCA_000017985.1_ASM1798v1_feature_table.txt.gz",
            "index",
            173353,
            "52ce70f6a9e672ea4f044474c24fc8e2",
            "5cba47c018f4a5180eb1af9f06e4b9103837f894a08f05fb5f6f70e8795379ec",
        ),
        _member(
            _GCA_DIR,
            "GCA_000017985.1_ASM1798v1_assembly_report.txt",
            "index",
            1172,
            "e89ce0ff1c31066a8639ec94e40e66a2",
            "51968f440a6497669ad8ccf703c437d5a8055990d2c7e27194b9cc1ffeeda369",
        ),
        _member(
            _GCF_DIR,
            "GCF_000017985.1_ASM1798v1_genomic.gbff.gz",
            "annotation",
            3428153,
            "15b4bbd969d274c99919bcd5d57d4d87",
            "b90a8ab7a8f1e9e736952b6e17017a9cd6bc6567cb11b1ecdc2e7c895695a26b",
        ),
        _member(
            _GCF_DIR,
            "GCF_000017985.1_ASM1798v1_genomic.gff.gz",
            "annotation",
            433859,
            "9a52f090f2e985a44e5e6752018b0b23",
            "27c302a37ac517de79999cc8438c744367e5c12ad4ac60b34f55bfd753214f25",
        ),
        _member(
            _GCF_DIR,
            "GCF_000017985.1_ASM1798v1_gene_ontology.gaf.gz",
            "annotation",
            158466,
            "42ac1448a06841ec7d7e86aab3f916e0",
            "4cbd6f5767d0f8651346891af174eaf9fd3353c25b6916fdee1f3d6c399374b1",
        ),
    ),
    edits_needed=(
        "torchcell/sequence/genome/registry.py: ECOLI_B_REL606 = "
        "'ecoli_B_REL606_ASM1798v1'",
        "scripts/provision_bacterial_genomes.py: a REL606 set of the nine members "
        "above (the GCF directory lists a _gene_ontology.gaf.gz, so it is a member), "
        "fetched by direct_url, md5-checked, deposited by deposit_assembly_set",
        "torchcell/datamodels/schema.py: 'REL606' in BacterialReferenceStrain; "
        "BACTERIAL_ASSEMBLY_SETS['REL606']; the set id in BacterialAssemblySet; "
        "ASSEMBLY_SET_ACCESSIONS[set] = ('GCA_000017985.1', 'GCF_000017985.1'); "
        "'ecoli_b_rel606_locus_tag' in BacterialGeneNamespace with pattern "
        "^ECB_[rt]?\\d{5}$ in BACTERIAL_LOCUS_TAG_PATTERNS (disjoint from the three "
        "existing patterns and from yeast systematic names)",
        "torchcell/sequence/genome/ecoli/: an E. coli B REL606 genome class beside "
        "EcoliK12Genome (the K12 classes and the ECK crosswalk do not apply to a B "
        "strain)",
        "torchcell/datasets/bacteria_common.py: REL606 in HOST_STRAINS['ecoli'], "
        "STRAIN_GENE_NAMESPACES and BACTERIAL_GENOME_CLASSES",
        "torchcell/verification/runners.py: a REL606 gene universe beside "
        "_ecoli_k12_gene_set (4,383 GenBank gene features)",
    ),
    retrieved_at="2026-10-07",
)

STRAIN_GAP = ProvenanceGap(
    field="genome_reference",
    reason=ProvenanceGapReason.deferred_pending_source_review,
    looked_in=Provenance(
        source_uri=PAPER_MD,
        citation_key=CITATION_KEY,
        sha256=PAPER_MD_SHA256,
        method=_OCR_METHOD,
        page="Methods, Cell Growth and RNA-seq",
    ),
    resolve_with=Provenance(
        source_uri=REL606_TIER_ADDITION.genbank_url,
        sha256=REL606_TIER_ADDITION.members[0].sha256,
        method="deposit NCBI assembly GCA_000017985.1 / GCF_000017985.1 (ASM1798v1) "
        "into the genomes tier as ecoli_B_REL606_ASM1798v1",
        page="GCA_000017985.1_ASM1798v1_genomic.gbff.gz",
        retrieved=REL606_TIER_ADDITION.retrieved_at,
    ),
    note="The paper's strain is E. coli B REL606 and its identifiers are REL606 ECB_ "
    "locus tags and NC_012967.1 YP_ proteins; the genomes tier holds only K-12 MG1655, "
    "K-12 BW25113 and KT2440, so no assembly pin is honest. The paper reports the "
    "genome; the gap is the tier's, recoverable by the deposit named in resolve_with.",
)


class StrainPinFinding(BaseModel):
    """Whether this paper's strain can be pinned to a deposited assembly set."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    strain: str
    lineage: str
    tier_strains: tuple[str, ...]
    pinnable: bool
    gap: ProvenanceGap | None
    tier_addition: AssemblyTierAddition | None


class UnpinnedStrainError(RuntimeError):
    """The paper's strain has no assembly set in the genomes tier."""

    def __init__(self, finding: StrainPinFinding) -> None:
        """Carry the finding (its ``gap`` and ``tier_addition``) on the error."""
        self.finding = finding
        super().__init__(
            f"{CITATION_KEY}: strain {finding.strain} ({finding.lineage}) is not one of "
            f"the tier's strains {list(finding.tier_strains)}; "
            f"{finding.gap.note if finding.gap is not None else ''}"
        )


def strain_pin_finding() -> StrainPinFinding:
    """The paper's strain against the schema's ``BacterialReferenceStrain`` vocabulary."""
    tier_strains: tuple[str, ...] = get_args(BacterialReferenceStrain)
    pinnable = STRAIN.value in tier_strains
    return StrainPinFinding(
        strain=STRAIN.value,
        lineage=LINEAGE.value,
        tier_strains=tier_strains,
        pinnable=pinnable,
        gap=None if pinnable else STRAIN_GAP,
        tier_addition=None if pinnable else REL606_TIER_ADDITION,
    )


def require_pinnable_strain() -> StrainPinFinding:
    """The gate a Caglar 2017 loader calls before building any record."""
    finding = strain_pin_finding()
    if not finding.pinnable:
        raise UnpinnedStrainError(finding)
    return finding


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
class RawFileSpec(BaseModel):
    """One raw-mirror file: where it lives, its pin, and the retriever that made it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relpath: str
    sha256: str
    description: str
    retrieval: RetrievalRecord


#: Supplementary table -> (PMC object, sha256, what it is). The SI lists the tables as
#: S1 to S14 after the SI PDF, so PMC object ``-s<N+1>`` is Table S<N>; the content
#: agrees (the S2 and S3 shapes equal the SI's 4196 x 152 and 4196 x 105).
SI_TABLES: dict[str, tuple[str, str, str]] = {
    "S1": (
        "srep45303-s2.csv",
        "1486290bf6a340ae64ee20c915435c0a00ff5eede489de1f56b62733b66f8940",
        "Supplementary Table S1 (tableS1_meta_data.csv): one row per sample with "
        "carbon source, Mg2+ and Na+ levels, growth phase, batch, technical-replicate "
        "counts and doubling time",
    ),
    "S2": (
        "srep45303-s3.csv",
        "df3e28237ea8e03c68ec93c577be1b1032ccc2d1f60372f45a72a6107de94d95",
        "Supplementary Table S2 (tableS2_mRNA_normalized_raw_data.csv): normalized, "
        "log-transformed mRNA counts, 4196 ECB_ genes x 152 samples",
    ),
    "S3": (
        "srep45303-s4.csv",
        "a391afb5784edebf2e44c52d622b868eaf7096258658f7473640090fb1429b11",
        "Supplementary Table S3 (tableS3_protein_normalized_raw_data.csv): normalized, "
        "log-transformed protein counts, 4196 YP_ proteins x 105 samples",
    ),
    "S4": (
        "srep45303-s5.csv",
        "eb9e6746dc7813327e463ad1ee5fc5f6eed17495bf59ef3d2ff9175c38ce5073",
        "Supplementary Table S4 (tableS4_fluxData.csv): mean and SDE of 13 branch flux "
        "ratios per salt, concentration and phase",
    ),
    # Added 2026-10-09 for the doubling-time loader (#776) and the fold-change loader
    # (#770), both of which are REVISIONS of this mirror: the files the paper's SI
    # lists, fetched by the recorded retriever. The four tables above are untouched, so
    # `deposit_raw_mirror` extends the manifest additively.
    "S5": (
        "srep45303-s6.csv",
        "76411accacbdc28310622cc15289b65ad44937bdc915051bbcfc8c4da1b04c60",
        "Supplementary Table S5: doubling time in exponential phase, one row per "
        "biological replicate growth curve (55 rows over 19 conditions), each with its "
        "own 95% confidence interval and the r^2 of the linear fit to OD600",
    ),
    "S8": (
        "srep45303-s9.csv",
        "738e1ee3f17e62a76a610741bc1b60ea21ee6eb80bdaa06846587dae791b44f5",
        "Supplementary Table S8 (tableS8_combinedOutputDF_DeSeq.csv): the DESeq2 "
        "differential-expression results of every test considered, 201,408 rows x 24 "
        "columns -- 100,704 mRNA rows keyed by ECB_ locus tag and 100,704 protein rows "
        "keyed by YP_ accession, as 48 groups of 4,196 (6 contrasts x 2 growth phases "
        "x 2 control parameterizations x 2 data types)",
    ),
}

#: NCBI E-utilities batch size for the YP_ protein records (GET, one URL per batch).
#: Measured 2026-10-07: a 200-record batch took about 85 to 95 s against the 120 s
#: timeout of the direct_url retriever; the 42 batches of 100 took 38 to 85 s each.
YP_BATCH_SIZE = 100
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

#: sha256 of each GenPept batch of Table S3's YP_ accessions, in table order, as
#: retrieved on RAW_RETRIEVED_AT. A second retrieval of batches 0 and 41 was
#: byte-identical, so the URL is a reproducible retrieval, not a moving one.
YP_BATCH_SHA256: tuple[str, ...] = (
    "2cf634697c26238d0421d87458d93d1f683bfa2ee3f0aeb1d49de49e5e464d53",
    "a8aee1572d8096584b2d4647065ddef8d21b7e5bfe40495eaa83e26b2da96780",
    "f36c65eb33d9da5ef5bc673835ff37bd6f6f1f4f3a709514b1f3c0de72d669ac",
    "2edd91e904d97d58db4fb68b32526e21355d880d1cb4bbc2dfd24a01e0eaea9d",
    "03c311f5e4cfaa06fd13877f72ef1f793cdf84163b1962ee01d16b56a5cf775e",
    "e714cfcc2446c8c0fe7272b6349bbb1b8f2d1a87673c136bf71eba3672476c1d",
    "f8798c0725438cf41661fe160ba4f609e166fc2ad191f462ab511a3022174d08",
    "0938f934992518750d84a99aa6ca8f4aff3b011916d62d9108da9102d035997f",
    "7a4b54840a35d5e1e7a800daa839cf1a9a3ed02e80abea8e8c2f0eff5dc9677a",
    "f66c0601555b7f8b19d289ee8ab1a9381c51f6f18705de81365a8d49b37e92d5",
    "8bb28eaaebb6cfa47fbb47fdf528c70ea396e6d6a6be0d890f6ca92bc2266982",
    "a264ac610977f0ad381dea2df3dabfcd39cc1623bdd488f55f252004728d3469",
    "f3ab3016232436a0f95f92d7fbfdf701275fc8f72c19f77797126f27a7fe3d6d",
    "2f44565ad628f1c32bcc8b00ee6e71ad6cc55c2ad119ad480073d0e38b8e640b",
    "ae360be5e5f27e4601bae44539e09ddd62c363bd195511a872428b8413c7f2f9",
    "540e05de73de93eefb8155bcc5e94dbb098fbaa2163ac1b6dbf330a99c2a4a05",
    "934b9ed33d6171484469751b9b565041ad76dfe1fd3a85ba9f6e0dede3938184",
    "89b7a382b9f0775fddff175b7431670f13786cae156bd117f2338564d9edce74",
    "641faca533aeb55b6c8200aacaad0bc5d0addf95c11ebfd6c0b974bbf22ba979",
    "863854f5fea4f3f85f0957edb9d367f4dbbd507f17899207fcda56a447558ca3",
    "e8880562a2e1f37d6fb1baf3b4e78d1cc7558afb7652e2ca410a36e14bed4b6c",
    "31b60b1cc3e6a28306de274f3be9d62e991029ce7677fd545d69c3fbe07d552b",
    "7c8295750ed805a32922779007ae74190fe0bcd282448108f59be2bad2e3c985",
    "3b07b8b94e6866e84d2caadc3050ded36ba2f38339f4f34acd63a2a96d83b583",
    "acf357b3f0b39dc8b95846d3431a207f2d2077d32377e3281c2fc23e10b15dd3",
    "18356152fbfb36f763e344a274f52b575365be569898c3158e4d519889c51a0f",
    "77a9e511f2349b56f540dd5dc7c69c17780cdd18074d98711294fe0e7b9f474b",
    "bc38b788f3a35dc0985f5b2fe54219079fdb92c29784a3761f1d537d700f3b51",
    "6989074acd7a54d0d042baf87c7b3afe41cc84df94ba4e4f947b74d5c8aed2de",
    "ea2ab9083ddcd85309f310a88800a771af1e9ad97603e780dd89774cdb272a31",
    "f3f2529499a0968d529ef47986896ecc6eeb8e212eaacc7cc1a88c9e8ae6545b",
    "e7eef3e8baf471f5b22b2c730885bb85fcf0e2fe969aad6a8fb262a2c68d0503",
    "2bd3ba9416284f601af94d32bc2c1d25c693f9ecfa9d583c62b0ec9d9144a025",
    "3f12fec04dc067965f8cb5739e9c8a30dd811899d807b8b22f79fbf442a1a2d9",
    "5539f56f18f93cd545cbe6a4a230f0f7ed17836feaf44e0102cc7511c9a1f33d",
    "0f9bfb72c65d2989526cb69c02b336830cffa92469fe1490f616a9f8f781bd58",
    "f1cecb2f33fabe75f61a879e4b42af99c40803df099cbdfcb16e7a9da391965d",
    "1e1c85455aa8bb5ba1d450d73e32d33071f98a2741d754c113c5818b882d0707",
    "4cb163db9981307e9c6c7c0201d98ba46028253a37c8667aa8720484f1e0e2e6",
    "eb033a796b24f0e0ec9094cba8f08bd4a09b69519ae9949a5c25f2f02f6a48ad",
    "e6965153cd89ed5e2a866f80ce3dd7f74aae248611f3a53c1bc02beeef02dc94",
    "96281a84a76b69243e0cfb5c75572f7f76fb731bb3fe1f9ab22c74ff9f6aa53b",
)


def si_table_relpath(table: str) -> str:
    """Mirror path of one supplementary table."""
    return f"data/{SI_TABLES[table][0]}"


def yp_batch_relpath(index: int) -> str:
    """Mirror path of one GenPept batch."""
    return f"ncbi_protein/yp_batch_{index:02d}.gp"


def yp_batch_url(accessions: Iterable[str]) -> str:
    """The efetch GET URL returning the GenPept records of ``accessions``."""
    return f"{EFETCH}?db=protein&rettype=gp&retmode=text&id={','.join(accessions)}"


def _si_table_spec(table: str) -> RawFileSpec:
    obj, sha, description = SI_TABLES[table]
    key = f"{PMC_ARTICLE}/{obj}"
    return RawFileSpec(
        relpath=si_table_relpath(table),
        sha256=sha,
        description=description,
        retrieval=RetrievalRecord(
            method=RetrievalMethod.pmc_cloud,
            source_url=pmc_cloud_url(key),
            retriever="torchcell.literature.retrieve.pmc_cloud_object",
            params={"key": key},
            sha256=sha,
            retrieved_at=TABLE_RETRIEVED_AT.get(table, RAW_RETRIEVED_AT),
        ),
    )


def si_table_specs() -> list[RawFileSpec]:
    """The supplementary tables the loaders read, in table order (S1 to S5 and S8)."""
    return [_si_table_spec(table) for table in SI_TABLES]


def yp_batch_specs(protein_ids: list[str]) -> list[RawFileSpec]:
    """One GenPept batch per ``YP_BATCH_SIZE`` accessions of Table S3, in table order.

    The URLs are a function of Table S3's first column, which is pinned, so the specs
    are reproducible from the mirror; the count must equal the pinned batch count.
    """
    batches = [
        protein_ids[start : start + YP_BATCH_SIZE]
        for start in range(0, len(protein_ids), YP_BATCH_SIZE)
    ]
    if len(batches) != len(YP_BATCH_SHA256):
        raise ValueError(
            f"{len(protein_ids)} protein ids make {len(batches)} batches; "
            f"{len(YP_BATCH_SHA256)} are pinned"
        )
    specs = []
    for index, (batch, sha) in enumerate(zip(batches, YP_BATCH_SHA256, strict=True)):
        url = yp_batch_url(batch)
        specs.append(
            RawFileSpec(
                relpath=yp_batch_relpath(index),
                sha256=sha,
                description=(
                    f"NCBI GenPept records of Table S3 YP_ accessions "
                    f"{batch[0]}..{batch[-1]} ({len(batch)}); each record names its "
                    "REL606 /locus_tag"
                ),
                retrieval=RetrievalRecord(
                    method=RetrievalMethod.direct_url,
                    source_url=url,
                    retriever="torchcell.literature.retrieve.direct_url",
                    params={"url": url},
                    sha256=sha,
                    retrieved_at=RAW_RETRIEVED_AT,
                ),
            )
        )
    return specs


def read_table_ids(path: str | Path) -> list[str]:
    """First column of Table S2 or S3 (the gene or protein identifier), in file order."""
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        return [row[0] for row in reader]


def raw_file_specs(protein_ids: list[str]) -> list[RawFileSpec]:
    """Every raw-mirror file, in the order the manifest lists them."""
    return si_table_specs() + yp_batch_specs(protein_ids)


def _data_root() -> str:
    """``DATA_ROOT`` from the environment (the mirror lives under it)."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/caglarColiMolecularPhenotype2017``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def retrieve_raw_files(staging_dir: str | Path) -> Path:
    """Run every recorded retriever into ``staging_dir`` and check each pin.

    The tables come first because the protein batches are built from Table S3's ids.
    A file already staged with its pinned sha256 is not fetched again; bytes that do not
    match a pin raise (upstream drift is reported, never followed).
    """
    staging = Path(staging_dir)
    for spec in si_table_specs():
        _retrieve_one(spec, staging)
    protein_ids = read_table_ids(staging / si_table_relpath("S3"))
    for spec in yp_batch_specs(protein_ids):
        _retrieve_one(spec, staging)
    return staging


def _retrieve_one(spec: RawFileSpec, staging: Path) -> None:
    dest = staging / spec.relpath
    if dest.exists() and sha256_file(dest) == spec.sha256:
        return
    body = run_retriever(spec.retrieval)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(body)
    got = sha256_file(dest)
    if got != spec.sha256:
        raise RuntimeError(
            f"{spec.relpath}: retrieved sha256 {got} differs from the pin {spec.sha256}"
        )


def raw_manifest(specs: list[RawFileSpec], sizes: dict[str, int]) -> Manifest:
    """The raw mirror's ``manifest.json`` content for ``specs`` (``created_at`` unset)."""
    return Manifest(
        citation_key=CITATION_KEY,
        doi=PAPER_DOI,
        title=PAPER_TITLE,
        files=[
            ArtifactRecord(
                path=spec.relpath,
                role=ROLE_RAW_DATA,
                bytes=sizes[spec.relpath],
                sha256=spec.sha256,
                source=spec.retrieval.source_url,
                retrieval=spec.retrieval,
            )
            for spec in specs
        ],
        si_data_sources=[
            f"https://pmc-oa-opendata.s3.amazonaws.com/{PMC_ARTICLE}/",
            EFETCH,
        ],
        si_expected=[
            "GEO GSE94117 (raw reads, per-gene read counts) -- not mirrored; Table S2 is "
            "the processed matrix the paper names",
            "PRIDE PXD005721 (raw spectra) -- not mirrored; Table S3 is the processed "
            "matrix the paper names",
            "Texas Data Repository doi:10.18738/T8/UG3TUR (raw GC-MS) -- not mirrored; "
            "Table S4 is the processed flux-ratio table",
        ],
        provenance_complete=True,
    )


def deposit_raw_mirror(*, source_dir: str | Path, data_root: str | None = None) -> Path:
    """Write the raw mirror from retrieved files (``retrieve_raw_files``) + its manifest.

    Every staged file, every existing mirror file and an existing manifest are checked
    before anything is written. Idempotent by sha256: a mirror file already holding its
    pinned bytes is left alone, and one holding other bytes raises rather than being
    overwritten. An existing ``manifest.json`` whose file records equal the new ones is
    left untouched (its ``created_at`` is the first deposit's); one that differs raises.

    This is the FROM-SCRATCH deposit: it writes a manifest of exactly the files it
    knows, so it refuses a mirror another of this citation key's loaders extended. A
    REVISION that adds one table to an existing mirror goes through
    ``deposit_si_table``.
    """
    source = Path(source_dir)
    root = raw_mirror_dir(data_root)
    specs = raw_file_specs(read_table_ids(source / si_table_relpath("S3")))
    for spec in specs:
        staged = source / spec.relpath
        got = sha256_file(staged)
        if got != spec.sha256:
            raise RuntimeError(f"{staged}: sha256 {got}, pinned {spec.sha256}")
        dest = root / spec.relpath
        if dest.exists() and sha256_file(dest) != spec.sha256:
            raise RuntimeError(f"{dest} exists with a different sha256; refusing")
    manifest = raw_manifest(
        specs, {spec.relpath: (source / spec.relpath).stat().st_size for spec in specs}
    )
    path = root / "manifest.json"
    existing = Manifest.model_validate_json(path.read_text()) if path.exists() else None
    if existing is not None and existing.model_dump(
        exclude={"created_at"}
    ) != manifest.model_dump(exclude={"created_at"}):
        raise RuntimeError(f"{path} records other files; refusing to overwrite")
    for spec in specs:
        dest = root / spec.relpath
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / spec.relpath, dest)
    if existing is None:
        manifest.created_at = datetime.now(UTC).isoformat()
        path.write_text(manifest.model_dump_json(indent=2))
    return root


def deposit_si_table(
    table: str, *, source_dir: str | Path, data_root: str | None = None
) -> Path:
    """Add ONE supplementary table to an already-deposited raw mirror, additively.

    The revision path of the provenance principle: a loader that needs a table the first
    deposit did not fetch retrieves it with its own recorded retriever, verifies its
    pin, and APPENDS its record to ``manifest.json`` with every earlier record left
    byte-identical and the first deposit's ``created_at`` kept. It is not
    ``deposit_raw_mirror``, which writes a manifest of exactly the files it knows: this
    citation key's mirror is written by several loaders, so a whole-mirror equality
    check would read another loader's deposit as drift. A record already present with
    other content, and a mirror file already holding other bytes, both raise.
    """
    source = Path(source_dir)
    root = raw_mirror_dir(data_root)
    spec = _si_table_spec(table)
    staged = source / spec.relpath
    got = sha256_file(staged)
    if got != spec.sha256:
        raise RuntimeError(f"{staged}: sha256 {got}, pinned {spec.sha256}")
    dest = root / spec.relpath
    if dest.exists() and sha256_file(dest) != spec.sha256:
        raise RuntimeError(f"{dest} exists with a different sha256; refusing")
    path = root / "manifest.json"
    if not path.exists():
        raise RuntimeError(f"{path} does not exist; run the full deposit first")
    manifest = Manifest.model_validate_json(path.read_text())
    record = ArtifactRecord(
        path=spec.relpath,
        role=ROLE_RAW_DATA,
        bytes=staged.stat().st_size,
        sha256=spec.sha256,
        source=spec.retrieval.source_url,
        retrieval=spec.retrieval,
    )
    prior = {existing.path: existing for existing in manifest.files}
    if spec.relpath in prior:
        if prior[spec.relpath].model_dump() != record.model_dump():
            raise RuntimeError(
                f"{path} records {spec.relpath} differently; refusing to overwrite"
            )
    else:
        tables = [
            index
            for index, existing in enumerate(manifest.files)
            if existing.path.startswith("data/")
        ]
        at = tables[-1] + 1 if tables else len(manifest.files)
        manifest.files.insert(at, record)
        path.write_text(manifest.model_dump_json(indent=2))
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(staged, dest)
    return dest


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
# Identifier coverage (the measurement behind the strain finding)
# --------------------------------------------------------------------------- #
class GenBankIdentifiers(BaseModel):
    """The identifiers one GenBank flat file carries."""

    model_config = ConfigDict(extra="forbid")

    locus_tags: set[str]
    old_locus_tags: set[str]
    protein_ids: set[str]


def genbank_identifiers(gbff_gz: str | Path) -> GenBankIdentifiers:
    """Locus tags (gene features), ``old_locus_tag`` values and CDS ``protein_id``s."""
    locus_tags: set[str] = set()
    old_locus_tags: set[str] = set()
    protein_ids: set[str] = set()
    with gzip.open(gbff_gz, "rt") as handle:
        for record in SeqIO.parse(handle, "genbank"):  # type: ignore[no-untyped-call]  # Bio.SeqIO is untyped
            for feature in record.features:
                if feature.type == "gene":
                    locus_tags.update(feature.qualifiers["locus_tag"])
                    old_locus_tags.update(feature.qualifiers.get("old_locus_tag", []))
                elif feature.type == "CDS":
                    protein_ids.update(feature.qualifiers.get("protein_id", []))
    return GenBankIdentifiers(
        locus_tags=locus_tags, old_locus_tags=old_locus_tags, protein_ids=protein_ids
    )


def genpept_locus_tags(path: str | Path) -> dict[str, str]:
    """``YP_`` accession.version -> the ``/locus_tag`` of its one CDS, from a GenPept file.

    A record with no CDS, or with a CDS naming no locus tag or several, is refused.
    """
    out: dict[str, str] = {}
    with open(path) as handle:
        for record in SeqIO.parse(handle, "genbank"):  # type: ignore[no-untyped-call]  # Bio.SeqIO is untyped
            tags = [
                feature.qualifiers.get("locus_tag", [])
                for feature in record.features
                if feature.type == "CDS"
            ]
            if len(tags) != 1 or len(tags[0]) != 1:
                raise ValueError(f"{record.id}: CDS locus tags {tags}, expected one")
            if record.id in out:
                raise ValueError(f"{record.id} appears twice")
            out[record.id] = tags[0][0]
    return out


class IdentifierCoverage(BaseModel):
    """How the paper's identifiers land on REL606 and on the deposited namespaces."""

    model_config = ConfigDict(extra="forbid")

    mrna_ids: int
    mrna_ecb: int
    mrna_in_genbank_locus_tags: int
    mrna_in_refseq_old_locus_tags: int
    protein_ids: int
    protein_yp: int
    protein_in_genbank_protein_ids: int
    protein_in_refseq_protein_ids: int
    protein_resolved_by_ncbi_record: int
    protein_locus_tag_in_genbank: int
    protein_row_aligned_with_mrna: int
    deposited_namespace_matches: dict[str, int] = Field(
        description="paper identifiers matching each deposited namespace's pattern"
    )
    rel606_pattern_matches: int


def identifier_coverage(
    raw_root: str | Path, genbank_gbff: str | Path, refseq_gbff: str | Path
) -> IdentifierCoverage:
    """Measure the paper's identifiers against the REL606 annotation and the tier.

    ``raw_root`` is the raw mirror (or a staging directory with the same layout);
    the two flat files are the GCA and GCF ``_genomic.gbff.gz`` of ASM1798v1.
    """
    root = Path(raw_root)
    mrna = read_table_ids(root / si_table_relpath("S2"))
    protein = read_table_ids(root / si_table_relpath("S3"))
    genbank = genbank_identifiers(genbank_gbff)
    refseq = genbank_identifiers(refseq_gbff)
    yp_to_tag: dict[str, str] = {}
    for spec in yp_batch_specs(protein):
        for accession, tag in genpept_locus_tags(root / spec.relpath).items():
            if accession in yp_to_tag:
                raise ValueError(f"{accession} appears in two batches")
            yp_to_tag[accession] = tag
    rel606 = re.compile(REL606_TIER_ADDITION.locus_tag_pattern)
    every_id = mrna + protein
    return IdentifierCoverage(
        mrna_ids=len(mrna),
        mrna_ecb=sum(name.startswith("ECB_") for name in mrna),
        mrna_in_genbank_locus_tags=sum(name in genbank.locus_tags for name in mrna),
        mrna_in_refseq_old_locus_tags=sum(
            name in refseq.old_locus_tags for name in mrna
        ),
        protein_ids=len(protein),
        protein_yp=sum(name.startswith("YP_") for name in protein),
        protein_in_genbank_protein_ids=sum(
            name in genbank.protein_ids for name in protein
        ),
        protein_in_refseq_protein_ids=sum(
            name in refseq.protein_ids for name in protein
        ),
        protein_resolved_by_ncbi_record=sum(name in yp_to_tag for name in protein),
        protein_locus_tag_in_genbank=sum(
            yp_to_tag[name] in genbank.locus_tags
            for name in protein
            if name in yp_to_tag
        ),
        protein_row_aligned_with_mrna=sum(
            yp_to_tag.get(p) == m for m, p in zip(mrna, protein, strict=True)
        ),
        deposited_namespace_matches={
            namespace: sum(bool(re.match(pattern, name)) for name in every_id)
            for namespace, pattern in BACTERIAL_LOCUS_TAG_PATTERNS.items()
        },
        rel606_pattern_matches=sum(bool(rel606.match(name)) for name in mrna),
    )


class AnnotationSummary(BaseModel):
    """The REL606 annotation facts ``REL606_TIER_ADDITION`` states, re-measured."""

    model_config = ConfigDict(extra="forbid")

    genbank_gene_features: int
    genbank_tag_prefixes: dict[str, int] = Field(
        description="locus-tag prefix (the tag less its trailing digits) -> genes"
    )
    genbank_pseudogenes: int
    genbank_pattern_matches: int
    refseq_gene_features: int
    refseq_tag_prefixes: dict[str, int]
    refseq_genes_with_old_locus_tag: int
    gaf_rows: int
    gaf_objects: int
    gaf_evidence: dict[str, int]


def _gene_qualifiers(gbff_gz: str | Path) -> list[dict[str, list[str]]]:
    with gzip.open(gbff_gz, "rt") as handle:
        return [
            dict(feature.qualifiers)
            for record in SeqIO.parse(handle, "genbank")  # type: ignore[no-untyped-call]  # Bio.SeqIO is untyped
            for feature in record.features
            if feature.type == "gene"
        ]


def _prefixes(tags: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tag in tags:
        prefix = re.sub(r"\d+$", "", tag)
        counts[prefix] = counts.get(prefix, 0) + 1
    return dict(sorted(counts.items()))


def annotation_summary(
    genbank_gbff: str | Path, refseq_gbff: str | Path, refseq_gaf: str | Path
) -> AnnotationSummary:
    """Gene features, tag forms and GO rows of the ASM1798v1 GenBank and RefSeq files."""
    genbank = _gene_qualifiers(genbank_gbff)
    refseq = _gene_qualifiers(refseq_gbff)
    genbank_tags = [q["locus_tag"][0] for q in genbank]
    pattern = re.compile(REL606_TIER_ADDITION.locus_tag_pattern)
    objects: set[str] = set()
    evidence: dict[str, int] = {}
    rows = 0
    with gzip.open(refseq_gaf, "rt") as handle:
        for line in handle:
            if line.startswith("!"):
                continue
            columns = line.rstrip("\n").split("\t")
            rows += 1
            objects.add(columns[1])
            evidence[columns[6]] = evidence.get(columns[6], 0) + 1
    return AnnotationSummary(
        genbank_gene_features=len(genbank),
        genbank_tag_prefixes=_prefixes(genbank_tags),
        genbank_pseudogenes=sum("pseudo" in q for q in genbank),
        genbank_pattern_matches=sum(bool(pattern.match(tag)) for tag in genbank_tags),
        refseq_gene_features=len(refseq),
        refseq_tag_prefixes=_prefixes(q["locus_tag"][0] for q in refseq),
        refseq_genes_with_old_locus_tag=sum("old_locus_tag" in q for q in refseq),
        gaf_rows=rows,
        gaf_objects=len(objects),
        gaf_evidence=dict(sorted(evidence.items())),
    )


# --------------------------------------------------------------------------- #
# The log base, back-solved: Tables S2 and S3 are DESeq2 VST values of counts
# --------------------------------------------------------------------------- #
FloatArray = np.ndarray[Any, np.dtype[np.float64]]

#: The base of the logarithm in Tables S2 and S3. Back-solved (``back_solve_counts``),
#: not stated: the paper says only "log-transformed" (``LOG_TRANSFORMED``).
VST_LOG_BASE = 2.0
#: A reconstructed count further than this from an integer refuses the build. Measured
#: on the mirror: 8.3e-9 (Table S2) and 1.1e-10 (Table S3) at worst.
COUNT_INTEGER_TOLERANCE = 1e-6
#: A per-sample factor further than this (relative) from DESeq2's median-of-ratios size
#: factor of the reconstructed counts refuses the build.
SIZE_FACTOR_TOLERANCE = 1e-6


def vst_forward(normalized: FloatArray, floor: float) -> FloatArray:
    """DESeq2's parametric VST of size-factor-normalized counts, written by its floor.

    DESeq2 computes ``log2((1 + e + 2 a q + 2 sqrt(a q (1 + e + a q))) / (4 a))`` for
    dispersion ``a + e / mean``. Tying ``e`` to the transform of a zero count,
    ``floor = log2((1 + e) / (4 a))``, leaves ``log2((q + 2P + sqrt(q**2 + 4 P q)) / 2)``
    with ``P = 2**floor``: one parameter, the table's minimum. For large ``q`` it is
    ``log2(q)``.
    """
    level = VST_LOG_BASE**floor
    root = np.sqrt(normalized**2 + 4.0 * level * normalized)
    return np.asarray(
        np.log2((normalized + 2.0 * level + root) / 2.0), dtype=np.float64
    )


def vst_inverse(values: FloatArray, floor: float) -> FloatArray:
    """The normalized count of each VST value: ``(2**y - P)**2 / 2**y``, ``P = 2**floor``."""
    level = VST_LOG_BASE**floor
    scaled = np.exp2(values)
    return np.asarray((scaled - level) ** 2 / scaled, dtype=np.float64)


def deseq2_size_factors(counts: FloatArray) -> FloatArray:
    """DESeq2 median-of-ratios size factors of a genes x samples count matrix.

    The Methods add the +1 pseudocount before the size factors are calculated
    (``SIZE_FACTOR_PSEUDOCOUNT``), so every gene has a finite geometric mean and enters
    the median, which is what DESeq2's ``estimateSizeFactorsForMatrix`` does then.
    """
    logs = np.log(counts + float(SIZE_FACTOR_PSEUDOCOUNT.value))
    return np.asarray(
        np.exp(np.median(logs - logs.mean(axis=1, keepdims=True), axis=0)),
        dtype=np.float64,
    )


class VstBackSolve(BaseModel):
    """The evidence that one table's values are DESeq2 VST values of integer counts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    table: str
    n_features: int
    n_samples: int
    log_base: float
    floor_value: float = Field(description="The table's minimum, the VST of a zero.")
    floor_level: float = Field(description="``2**floor_value``, the P of the inverse.")
    zero_cells: int = Field(description="Cells at the floor, i.e. zero counts.")
    samples_with_a_zero: int
    max_count_integer_deviation: float = Field(
        description="Largest distance of a reconstructed count from an integer."
    )
    max_size_factor_relative_deviation: float = Field(
        description="Largest |s / s_DESeq2 - 1| over samples, s_DESeq2 computed on the "
        "reconstructed counts plus the +1 pseudocount."
    )
    size_factor_min: float
    size_factor_max: float
    library_size_min: int
    library_size_median: float
    library_size_max: int


class CountReconstruction(BaseModel):
    """Integer counts and per-sample size factors recovered from one released table."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    evidence: VstBackSolve
    counts: pd.DataFrame
    size_factors: pd.Series


def back_solve_counts(table: pd.DataFrame, *, label: str) -> CountReconstruction:
    """Invert a genes x samples VST table to its integer counts and size factors.

    The floor (the table minimum) is the transform of a zero count. Each sample's
    smallest nonzero normalized count is taken as a count of 1, so its size factor is
    the reciprocal; every other count is then the normalized count times that factor.
    Two checks make this a measurement and not an assumption: every reconstructed
    count must be an integer (``COUNT_INTEGER_TOLERANCE``), and the factors must equal
    DESeq2's median-of-ratios size factors of the reconstructed counts with the stated
    +1 pseudocount (``SIZE_FACTOR_TOLERANCE``). A sample whose smallest count is not 1
    fails the second check.
    """
    values = table.to_numpy(dtype=np.float64)
    floor = float(values.min())
    normalized = vst_inverse(values, floor)
    at_floor = values == floor
    normalized[at_floor] = 0.0
    size_factors = np.empty(values.shape[1], dtype=np.float64)
    for j in range(values.shape[1]):
        positive = normalized[:, j][normalized[:, j] > 0.0]
        if positive.size == 0:
            raise RuntimeError(
                f"{label}: sample {table.columns[j]} has no nonzero count"
            )
        size_factors[j] = 1.0 / float(positive.min())
    raw = normalized * size_factors
    counts = np.rint(raw)
    deviation = float(np.abs(raw - counts).max())
    if deviation > COUNT_INTEGER_TOLERANCE:
        raise RuntimeError(
            f"{label}: a reconstructed count is {deviation} from an integer (tolerance "
            f"{COUNT_INTEGER_TOLERANCE}); the table is not a base-{VST_LOG_BASE} VST of "
            "counts"
        )
    relative = float(np.abs(size_factors / deseq2_size_factors(counts) - 1.0).max())
    if relative > SIZE_FACTOR_TOLERANCE:
        raise RuntimeError(
            f"{label}: the reconstructed size factors differ from DESeq2's (+1 "
            f"pseudocount) by up to {relative} (tolerance {SIZE_FACTOR_TOLERANCE})"
        )
    totals = counts.sum(axis=0)
    evidence = VstBackSolve(
        table=label,
        n_features=int(values.shape[0]),
        n_samples=int(values.shape[1]),
        log_base=VST_LOG_BASE,
        floor_value=floor,
        floor_level=VST_LOG_BASE**floor,
        zero_cells=int(at_floor.sum()),
        samples_with_a_zero=int(at_floor.any(axis=0).sum()),
        max_count_integer_deviation=deviation,
        max_size_factor_relative_deviation=relative,
        size_factor_min=float(size_factors.min()),
        size_factor_max=float(size_factors.max()),
        library_size_min=int(totals.min()),
        library_size_median=float(np.median(totals)),
        library_size_max=int(totals.max()),
    )
    return CountReconstruction(
        evidence=evidence,
        counts=pd.DataFrame(
            counts.astype(np.int64), index=table.index, columns=table.columns
        ),
        size_factors=pd.Series(size_factors, index=table.columns),
    )


def log_base_derivation(evidence: Sequence[VstBackSolve]) -> StatDerivation:
    """The log base as a ``StatDerivation``: back-solved from the released tables."""
    return StatDerivation(
        field="log_base",
        method=DerivationMethod.back_solve,
        value=VST_LOG_BASE,
        statistic="every released cell inverts through the base-2 DESeq2 VST to an "
        "integer count times a per-sample factor equal to DESeq2's +1-pseudocount "
        "median-of-ratios size factor",
        diagnostics={
            **{
                f"{e.table}_max_count_integer_deviation": e.max_count_integer_deviation
                for e in evidence
            },
            **{
                f"{e.table}_max_size_factor_relative_deviation": (
                    e.max_size_factor_relative_deviation
                )
                for e in evidence
            },
        },
        provenance=LOG_TRANSFORMED.provenance,
        rationale="the paper says 'log-transformed' with no base and defers to ref 10, "
        "which is not mirrored; base e and base 10 versions of the same inverse leave "
        "cells 0.5 from an integer, base 2 leaves none further than 1e-8",
    )


# --------------------------------------------------------------------------- #
# Table S1: the sample sheet
# --------------------------------------------------------------------------- #
COL_SAMPLE = "dataSet"
COL_EXPERIMENT = "experiment"
COL_TIME = "growthTime_hr"
COL_BATCH = "batchNumber"
COL_CARBON = "carbonSource"
COL_MG = "Mg_mM"
COL_MG_LEVEL = "Mg_mM_Levels"
COL_NA = "Na_mM"
COL_NA_LEVEL = "Na_mM_Levels"
COL_PHASE = "growthPhase"
COL_CONDITION = "uniqueCondition"
COL_RNA = "RNA_Data_Freq"
COL_PROTEIN = "Protein_Data_Freq"

#: Table S1's base Mg2+ level (the Methods state 0.83 mM; the sheet writes 0.8).
BASE_MG_SHEET_MM = 0.8
#: Table S1's base Na+ level, the ~5 mM the sodium citrate brings (``SODIUM_SERIES``).
BASE_NA_MM = 5.0
#: The carbon source of DM500 and of the reference conditions.
BASE_CARBON = "glucose"

CarbonSource = Literal["glucose", "glycerol", "lactate", "gluconate"]
MgLevel = Literal["lowMg", "baseMg", "highMg"]
NaLevel = Literal["baseNa", "highNa"]


class GrowthPhase(StrEnum):
    """Table S1's ``growthPhase`` cell of a sample with data."""

    exponential = "exponential"
    stationary = "stationary"
    late_stationary = "late_stationary"


class SampleRow(BaseModel):
    """One Table S1 row that carries RNA or protein data."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample: str
    experiment: str
    condition: str = Field(description="Table S1 ``uniqueCondition``.")
    growth_time_hr: float
    batch: int
    carbon_source: CarbonSource
    mg_mm: float
    mg_level: MgLevel
    na_mm: float
    na_level: NaLevel
    growth_phase: GrowthPhase
    rna_technical_replicates: int
    protein_technical_replicates: int

    @property
    def reference_condition(self) -> bool:
        """Glucose with base Mg2+ and base Na+, the paper's reference rule."""
        return (
            self.carbon_source == BASE_CARBON
            and self.mg_level == "baseMg"
            and self.na_level == "baseNa"
        )


def read_sample_sheet(path: str | Path) -> list[SampleRow]:
    """Every Table S1 row with RNA or protein data, in sheet order.

    Rows with neither (the pilot cultures and the repeated glucose time course) are not
    samples of either table. A cell of no known form raises (pydantic), and a level
    label that disagrees with its number (``baseMg`` off 0.8 mM, ``baseNa`` off 5 mM)
    is refused.
    """
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    rows: list[SampleRow] = []
    for cells in frame.to_dict(orient="records"):
        rna = int(cells[COL_RNA])
        protein = int(cells[COL_PROTEIN])
        if rna == 0 and protein == 0:
            continue
        row = SampleRow.model_validate(
            {
                "sample": cells[COL_SAMPLE],
                "experiment": cells[COL_EXPERIMENT],
                "condition": cells[COL_CONDITION],
                "growth_time_hr": float(cells[COL_TIME]),
                "batch": int(cells[COL_BATCH]),
                "carbon_source": cells[COL_CARBON],
                "mg_mm": float(cells[COL_MG]),
                "mg_level": cells[COL_MG_LEVEL],
                "na_mm": float(cells[COL_NA]),
                "na_level": cells[COL_NA_LEVEL],
                "growth_phase": cells[COL_PHASE],
                "rna_technical_replicates": rna,
                "protein_technical_replicates": protein,
            }
        )
        if (row.mg_level == "baseMg") != (row.mg_mm == BASE_MG_SHEET_MM):
            raise ValueError(f"{row.sample}: {row.mg_level} at {row.mg_mm} mM Mg2+")
        if (row.na_level == "baseNa") != (row.na_mm == BASE_NA_MM):
            raise ValueError(f"{row.sample}: {row.na_level} at {row.na_mm} mM Na+")
        rows.append(row)
    if len({row.sample for row in rows}) != len(rows):
        raise ValueError("Table S1 repeats a sample id")
    return rows


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
#: The carbon source's dose when it replaces glucose (``CARBON_SOURCE_SWAP``).
SWAPPED_CARBON_G_PER_L = 0.5


def carbon_source_perturbation(carbon: str) -> EnvironmentPhysicalPerturbation:
    """Glycerol, lactate or gluconate at 0.5 g/L in place of DM500's glucose."""
    return EnvironmentPhysicalPerturbation(
        factor=PhysicalFactor.carbon_source,
        magnitude=Concentration(
            value=SWAPPED_CARBON_G_PER_L, unit=ConcentrationUnit.g_per_l
        ),
        agent=resolved_compound(carbon),
    )


def magnesium_perturbation(mg_mm: float) -> SmallMoleculePerturbation:
    """Magnesium sulfate at Table S1's level, in place of Davis Minimal's 0.83 mM."""
    return SmallMoleculePerturbation(
        compound=resolved_compound("magnesium sulfate"),
        concentration=Concentration(value=mg_mm, unit=ConcentrationUnit.millimolar),
        description=f"magnesium sulfate set to {mg_mm:g} mM final (Table S1 Mg_mM) in "
        "place of the 0.83 mM Davis Minimal medium normally holds",
    )


def sodium_perturbation(na_mm: float) -> SmallMoleculePerturbation:
    """NaCl added to raise Na+ from the ~5 mM base to Table S1's level."""
    added = na_mm - BASE_NA_MM
    if added <= 0:
        raise ValueError(f"{na_mm} mM Na+ is not above the {BASE_NA_MM} mM base")
    return SmallMoleculePerturbation(
        compound=resolved_compound("sodium chloride"),
        concentration=Concentration(value=added, unit=ConcentrationUnit.millimolar),
        description=f"sodium chloride added to reach {na_mm:g} mM Na+ (Table S1 Na_mM) "
        f"over the ~{BASE_NA_MM:g} mM the medium's sodium citrate brings",
    )


def build_environment(row: SampleRow) -> Environment:
    """The environment of one sample: medium, edits, 37 C, aerobic, collection time."""
    perturbations: list[EnvironmentPerturbationType] = []
    if row.carbon_source == BASE_CARBON:
        media = DM500
    else:
        media = DAVIS_MINIMAL
        perturbations.append(carbon_source_perturbation(row.carbon_source))
    if row.mg_level != "baseMg":
        perturbations.append(magnesium_perturbation(row.mg_mm))
    if row.na_level != "baseNa":
        perturbations.append(sodium_perturbation(row.na_mm))
    return Environment(
        media=media,
        temperature=Temperature(value=float(CULTURE_CONDITIONS.value)),
        perturbations=perturbations,
        aerobicity="aerobic",
        duration_hours=row.growth_time_hr,
    )


def reference_duration_gap(phase: GrowthPhase, times: Iterable[float]) -> ProvenanceGap:
    """Why a reference environment has no ``duration_hours``."""
    stated = ", ".join(f"{t:g}" for t in sorted(set(times)))
    return ProvenanceGap(
        field="duration_hours",
        reason=ProvenanceGapReason.not_reported_by_primary,
        looked_in=EXPONENTIAL_SAMPLING.provenance,
        note=f"the {phase.value} reference condition pools samples Table S1 dates at "
        f"{stated} h; the paper sets the phase by optical density, so no single "
        "duration describes the reference",
    )


def reference_environment(phase: GrowthPhase, times: Iterable[float]) -> Environment:
    """DM500 with base Mg2+ and Na+ (the reference rule) in ``phase``."""
    return Environment(
        media=DM500,
        temperature=Temperature(value=float(CULTURE_CONDITIONS.value)),
        perturbations=[],
        aerobicity="aerobic",
        provenance_gaps=[reference_duration_gap(phase, times)],
    )


def reference_rows(rows: Sequence[SampleRow]) -> dict[GrowthPhase, list[SampleRow]]:
    """Each phase's reference samples: one Table S1 condition, glucose, base Mg and Na."""
    out: dict[GrowthPhase, list[SampleRow]] = {}
    for phase in GrowthPhase:
        if not any(r.growth_phase is phase for r in rows):
            continue
        members = [r for r in rows if r.reference_condition and r.growth_phase is phase]
        conditions = {r.condition for r in members}
        if len(conditions) != 1:
            raise RuntimeError(
                f"the {phase.value} reference rule selects conditions "
                f"{sorted(conditions)}"
            )
        out[phase] = members
    return out


# --------------------------------------------------------------------------- #
# Phenotypes
# --------------------------------------------------------------------------- #
RNA_MEASUREMENT_TYPE = "rnaseq_tpm"
PROTEIN_MEASUREMENT_TYPE = "lcmsms_spectral_count_deseq2_size_factor_normalized"
TPM_TOTAL = 1e6

#: The stored counts are the reads of the coding genes, not the library's mapped reads.
MAPPED_READS_GAP = ProvenanceGap(
    field="n_mapped_reads",
    reason=ProvenanceGapReason.deferred_pending_source_review,
    looked_in=Provenance(
        source_uri=f"{RAW_DIR_REL}/{si_table_relpath('S2')}",
        citation_key=CITATION_KEY,
        sha256=SI_TABLES["S2"][1],
        page="Table S2 (coding-gene counts only; no read-depth row)",
    ),
    resolve_with=Provenance(
        source_uri="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE94117",
        citation_key=CITATION_KEY,
        page="GEO series files: read counts per gene",
    ),
    note="the reconstructed counts sum the reads of the 4,196 protein-coding genes "
    "('For RNA, we only analyzed the counts of reads that overlapped annotated protein "
    "coding genes'); the library's total mapped reads are in the GEO deposit, which is "
    "not mirrored",
)


def gene_lengths(genome: EcoliBREL606Genome, tags: Sequence[str]) -> FloatArray:
    """Each locus's GenBank gene-feature span (bp); a joined location is refused."""
    lengths = []
    for tag in tags:
        locus = genome.genbank.loci[tag]
        if locus.segments != 1:
            raise ValueError(f"{tag} has a {locus.segments}-part location")
        lengths.append(float(locus.end - locus.start + 1))
    return np.asarray(lengths, dtype=np.float64)


def tpm(counts: FloatArray, lengths: FloatArray) -> FloatArray:
    """Transcripts per million: counts per base, scaled to sum to one million."""
    rate = counts / lengths
    return np.asarray(rate / rate.sum() * TPM_TOTAL, dtype=np.float64)


def rnaseq_phenotype(
    genes: Sequence[str], counts: FloatArray, lengths: FloatArray
) -> RNASeqExpressionPhenotype:
    """One library's TPM and integer read counts over the stored genes."""
    return RNASeqExpressionPhenotype(
        expression_tpm=dict(
            zip(genes, (float(v) for v in tpm(counts, lengths)), strict=True)
        ),
        expression_count={g: int(c) for g, c in zip(genes, counts, strict=True)},
        measurement_type=RNA_MEASUREMENT_TYPE,
        provenance_gaps=[MAPPED_READS_GAP],
    )


def rnaseq_reference_phenotype(
    genes: Sequence[str], counts: FloatArray, lengths: FloatArray
) -> RNASeqExpressionPhenotype:
    """The mean TPM and mean count (rounded half to even) of a genes x samples block."""
    profiles = np.stack([tpm(counts[:, j], lengths) for j in range(counts.shape[1])])
    return RNASeqExpressionPhenotype(
        expression_tpm=dict(
            zip(genes, (float(v) for v in profiles.mean(axis=0)), strict=True)
        ),
        expression_count={
            g: int(c) for g, c in zip(genes, np.rint(counts.mean(axis=1)), strict=True)
        },
        measurement_type=RNA_MEASUREMENT_TYPE,
        provenance_gaps=[MAPPED_READS_GAP],
    )


def protein_phenotype(
    proteins: Sequence[str], counts: FloatArray, size_factor: float
) -> ProteinAbundancePhenotype:
    """One sample's size-factor-normalized spectral counts; one culture, n = 1."""
    return ProteinAbundancePhenotype(
        protein_abundance={
            p: float(c) / size_factor for p, c in zip(proteins, counts, strict=True)
        },
        n_replicates=dict.fromkeys(proteins, 1),
        measurement_type=PROTEIN_MEASUREMENT_TYPE,
    )


def protein_reference_phenotype(
    proteins: Sequence[str], normalized: FloatArray
) -> ProteinAbundancePhenotype:
    """Mean, standard error and sample count of a proteins x samples block."""
    n = normalized.shape[1]
    mean = normalized.mean(axis=1)
    se = (
        normalized.std(axis=1, ddof=1) / math.sqrt(n)
        if n > 1
        else np.full(len(proteins), math.nan)
    )
    return ProteinAbundancePhenotype(
        protein_abundance={p: float(v) for p, v in zip(proteins, mean, strict=True)},
        protein_abundance_se={p: float(v) for p, v in zip(proteins, se, strict=True)},
        n_replicates=dict.fromkeys(proteins, n),
        measurement_type=PROTEIN_MEASUREMENT_TYPE,
    )


# --------------------------------------------------------------------------- #
# Source studies (#771): which paper FIRST reported each sample
#
# 27 of the 152 mRNA samples and 27 of the 105 protein samples are Houser 2015's
# glucose time course, released again here. Caglar says so in three places, all quoted
# below from the sha256-pinned OCR, and splits its deposits to match. Houser 2015 is NOT
# mirrored, so there is no second copy in the graph and this is an attribution question
# rather than a de-duplication one: before this, those 54 records asserted Caglar 2017
# as the source of measurements Houser 2015 published. Each record now carries the
# ``Publication`` of the study that first reported it, which is the Borchert 2024
# pattern (``torchcell/datasets/pputida/borchert2024.py``, ``SOURCE_STUDIES`` +
# ``attribute_sample``) already set on this program. No schema class changes.
# --------------------------------------------------------------------------- #
#: The Table S1 ``experiment`` value whose samples Houser 2015 first reported.
HOUSER2015_EXPERIMENT = "glucose_time_course"
#: Houser 2015's DOI. NOT stated by Caglar, which cites it by journal, volume and
#: article id only (``HOUSER2015_CITATION``). PLOS mints DOIs as
#: ``10.1371/journal.<journal code>.<article number>``, so the citation's "PLOS Comput
#: Biol 11, e1004400" fixes this string; resolving it on 2026.10.09 by DOI content
#: negotiation (``curl -LH 'Accept: application/vnd.citationstyles.csl+json'
#: https://doi.org/10.1371/journal.pcbi.1004400``) returned title, container-title,
#: volume and page equal to the citation's, which is the check that it is the right
#: record rather than a guess.
HOUSER2015_DOI = "10.1371/journal.pcbi.1004400"
#: From the same resolution, confirmed by an esearch of PubMed on that DOI.
HOUSER2015_PUBMED_ID = "26275208"
#: Houser 2015 has NO mirror and no loader: no `torchcell-library` key, no
#: `torchcell-raw` key, no dataset class. The attribution is therefore sourced entirely
#: from Caglar's own citation of it, and the paper itself is unread here.
HOUSER2015_IS_MIRRORED = False

HOUSER2015_DEFERRAL = _paper(
    HOUSER2015_EXPERIMENT,
    "Results from one of these conditions, long-term glucose starvation, have been "
    "presented previously10.",
    page="Results, Experimental design and data collection",
    note="reference 10 is Houser 2015 (HOUSER2015_CITATION). 'long-term glucose "
    "starvation' is Table S1's experiment == 'glucose_time_course', the 27 samples "
    "that appear as columns of BOTH Table S2 (mRNA) and Table S3 (protein); those 54 "
    "records are attributed to Houser 2015 rather than to this paper (#771)",
)
HOUSER2015_CITATION = _paper(
    HOUSER2015_DOI,
    "Houser, J. R. et al. Controlled Measurement and Comparative Analysis of Cellular "
    "Components in E. coli Reveals Broad Regulatory Changes in Response to Glucose "
    "Starvation. PLOS Comput Biol 11, e1004400 (2015).",
    page="References, reference 10",
    note="the full identity of the originating study, as this paper gives it. It "
    "carries no DOI, so HOUSER2015_DOI records how that string was fixed and checked",
)
HOUSER2015_DEPOSITS = _paper(
    ("GSE67402", "PXD002140"),
    "accession GSE67402 for the glucose time-course previously published10, accession "
    "GSE94117 for all other experiments",
    page="Data availability",
    note="the authors split the deposits along the same line: GSE67402 and PXD002140 "
    "are the glucose time course's earlier accessions, GSE94117 and PXD005721 this "
    "paper's own. The proteomics half reads 'accession PXD002140 for the glucose "
    "time-course previously published10, accession PXD005721 for all other "
    "experiments'",
)

SourceStudyKey = Literal["caglar2017", "houser2015"]


class SourceStudy(BaseModel):
    """A paper a sample of this release is attributed to."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: SourceStudyKey
    citation: str = Field(description="how the release itself names the study")
    doi: str
    pubmed_id: str | None = None
    title: str | None = Field(
        default=None, description="verbatim title, for a study with no mirror key"
    )
    citation_key: str | None = Field(
        default=None, description="the mirrored library key, None when unmirrored"
    )
    is_mirrored: bool

    @property
    def publication(self) -> Publication:
        """The ``Publication`` every record attributed to this study stores."""
        return Publication(
            doi=self.doi,
            doi_url=f"https://doi.org/{self.doi}",
            pubmed_id=self.pubmed_id,
            pubmed_url=(
                None
                if self.pubmed_id is None
                else f"https://pubmed.ncbi.nlm.nih.gov/{self.pubmed_id}/"
            ),
        )


SOURCE_STUDIES: dict[SourceStudyKey, SourceStudy] = {
    "caglar2017": SourceStudy(
        key="caglar2017",
        citation="Caglar MU et al. 2017, Sci Rep 7:45303 (this release)",
        doi=PAPER_DOI,
        citation_key=CITATION_KEY,
        is_mirrored=True,
    ),
    "houser2015": SourceStudy(
        key="houser2015",
        citation=HOUSER2015_CITATION.quote,
        doi=HOUSER2015_DOI,
        pubmed_id=HOUSER2015_PUBMED_ID,
        title="Controlled Measurement and Comparative Analysis of Cellular Components "
        "in E. coli Reveals Broad Regulatory Changes in Response to Glucose Starvation",
        citation_key=None,
        is_mirrored=HOUSER2015_IS_MIRRORED,
    ),
}


class SampleAttribution(BaseModel):
    """Which paper first reported one sample, and on what evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample: str
    experiment: str
    study: SourceStudyKey
    doi: str
    evidence: tuple[str, ...] = Field(
        description="module-level SourcedValue names whose quotes carry this attribution"
    )


#: The ``SourcedValue`` names that carry the Houser 2015 attribution.
HOUSER2015_EVIDENCE: tuple[str, ...] = (
    "HOUSER2015_DEFERRAL",
    "HOUSER2015_CITATION",
    "HOUSER2015_DEPOSITS",
)


def attribute_sample(row: SampleRow) -> SampleAttribution:
    """The study that first reported ``row``'s sample.

    The split is Table S1's own ``experiment`` column: ``glucose_time_course`` is the
    condition Caglar says was "presented previously", everything else is this paper's.
    """
    study: SourceStudyKey = (
        "houser2015" if row.experiment == HOUSER2015_EXPERIMENT else "caglar2017"
    )
    return SampleAttribution(
        sample=row.sample,
        experiment=row.experiment,
        study=study,
        doi=SOURCE_STUDIES[study].doi,
        evidence=HOUSER2015_EVIDENCE if study == "houser2015" else (),
    )


def source_publications(rows: Sequence[SampleRow]) -> dict[str, Publication]:
    """One ``Publication`` per sample id, keyed by the study that first reported it."""
    return {
        row.sample: SOURCE_STUDIES[attribute_sample(row).study].publication
        for row in rows
    }


# --------------------------------------------------------------------------- #
# Build bookkeeping
# --------------------------------------------------------------------------- #
#: Every released id must resolve to a REL606 locus. Measured on the mirror: 4,196 of
#: 4,196 for both tables, so anything less means the table or the annotation moved.
MIN_RESOLVED_FRACTION = 1.0


class RecordSample(BaseModel):
    """What one LMDB record was built from: its Table S1 row and its back-solve."""

    model_config = ConfigDict(extra="forbid")

    index: int
    sample: str
    experiment: str
    condition: str
    growth_phase: GrowthPhase
    growth_time_hr: float
    batch: int
    technical_replicates: int = Field(
        description="Table S1's technical-replicate count for this assay."
    )
    size_factor: float
    library_size: int = Field(description="Sum of the reconstructed counts.")


class ReplicateGroup(BaseModel):
    """The records of one Table S1 ``uniqueCondition``: its biological replicates.

    A time course puts several collection times under one condition, and a salt series
    collects its three replicates at different times (sampling is set by optical
    density), so the group lists each record's time and batch.
    """

    model_config = ConfigDict(extra="forbid")

    condition: str
    growth_phase: GrowthPhase
    carbon_source: str
    mg_mm: float
    na_mm: float
    samples: list[str]
    record_indices: list[int]
    growth_times_hr: list[float]
    batches: list[int]
    experiments: list[str]


class BuildAccounting(BaseModel):
    """The retention arithmetic of one build. No rule drops a sample: every Table S1
    row with this assay's data is a record, and a cell the loader cannot read raises.
    """

    model_config = ConfigDict(extra="forbid")

    dataset: str
    table: str
    sheet_rows_with_data: int = Field(
        description="Table S1 rows with RNA or protein data."
    )
    candidate_records: int = Field(description="Samples (columns) of the table.")
    kept_records: int
    dropped_records: int
    drops_by_reason: dict[str, int]
    kept_by_growth_phase: dict[str, int]
    kept_by_carbon_source: dict[str, int]
    kept_by_experiment: dict[str, int]
    reference_samples: dict[str, list[str]]
    notes: list[str]

    def check(self) -> None:
        """Kept plus dropped is every candidate, and the reasons add up."""
        if self.kept_records + self.dropped_records != self.candidate_records:
            raise RuntimeError(f"{self.dataset}: retention does not add up")
        if sum(self.drops_by_reason.values()) != self.dropped_records:
            raise RuntimeError(f"{self.dataset}: drop reasons do not add up")


def _counts_of(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _accounting(
    dataset: str,
    table: str,
    sheet_rows: int,
    rows: Sequence[SampleRow],
    references: Mapping[GrowthPhase, Sequence[SampleRow]],
    notes: list[str],
) -> BuildAccounting:
    accounting = BuildAccounting(
        dataset=dataset,
        table=table,
        sheet_rows_with_data=sheet_rows,
        candidate_records=len(rows),
        kept_records=len(rows),
        dropped_records=0,
        drops_by_reason={},
        kept_by_growth_phase=_counts_of(r.growth_phase.value for r in rows),
        kept_by_carbon_source=_counts_of(r.carbon_source for r in rows),
        kept_by_experiment=_counts_of(r.experiment for r in rows),
        reference_samples={
            phase.value: [r.sample for r in members]
            for phase, members in references.items()
        },
        notes=notes,
    )
    accounting.check()
    return accounting


def replicate_groups(rows: Sequence[SampleRow]) -> list[ReplicateGroup]:
    """The records grouped by Table S1 condition, in condition order."""
    groups: dict[str, ReplicateGroup] = {}
    for index, row in enumerate(rows):
        group = groups.setdefault(
            row.condition,
            ReplicateGroup(
                condition=row.condition,
                growth_phase=row.growth_phase,
                carbon_source=row.carbon_source,
                mg_mm=row.mg_mm,
                na_mm=row.na_mm,
                samples=[],
                record_indices=[],
                growth_times_hr=[],
                batches=[],
                experiments=[],
            ),
        )
        defining = (row.growth_phase, row.carbon_source, row.mg_mm, row.na_mm)
        if defining != (
            group.growth_phase,
            group.carbon_source,
            group.mg_mm,
            group.na_mm,
        ):
            raise RuntimeError(
                f"{row.sample} differs from its condition {row.condition}"
            )
        group.samples.append(row.sample)
        group.record_indices.append(index)
        group.growth_times_hr.append(row.growth_time_hr)
        group.batches.append(row.batch)
        group.experiments.append(row.experiment)
    return [groups[key] for key in sorted(groups)]


def _attribution_ledger(rows: Sequence[SampleRow]) -> dict[str, Any]:
    """The per-sample source-study ledger this build wrote (#771).

    Named rather than inlined so the attribution is auditable from the store without
    re-reading Table S1: it carries each study's identity, whether we mirror it, the
    per-study sample counts, and one row per sample naming its study and the quotes the
    attribution rests on.
    """
    attributions = [attribute_sample(row) for row in rows]
    return {
        "issue": "771",
        "rule": (
            "a sample whose Table S1 experiment column is "
            f"{HOUSER2015_EXPERIMENT!r} was first reported by Houser 2015; every other "
            "sample is this release's own"
        ),
        "studies": {
            key: study.model_dump(mode="json") for key, study in SOURCE_STUDIES.items()
        },
        "samples_by_study": dict(
            sorted(Counter(a.study for a in attributions).items())
        ),
        "samples": [a.model_dump(mode="json") for a in attributions],
    }


def _write_json(directory: str, name: str, payload: Any) -> None:
    with open(osp.join(directory, name), "w") as handle:
        json.dump(payload, handle, indent=2)


def _write_model(directory: str, name: str, model: BaseModel) -> None:
    with open(osp.join(directory, name), "w") as handle:
        handle.write(model.model_dump_json(indent=2))


# --------------------------------------------------------------------------- #
# Shared loader steps
# --------------------------------------------------------------------------- #
def _table_pin(table: str) -> tuple[str, str]:
    """``(mirror relpath, sha256)`` of one supplementary table."""
    return si_table_relpath(table), SI_TABLES[table][1]


def _batch_pins() -> list[tuple[str, str]]:
    """``(mirror relpath, sha256)`` of every pinned GenPept batch, in table order."""
    return [(yp_batch_relpath(i), sha) for i, sha in enumerate(YP_BATCH_SHA256)]


def _link_mirror_files(raw_dir: str, pins: Iterable[tuple[str, str]]) -> None:
    """Link each pinned mirror file into ``raw/`` under its base name.

    The manifest must record each pin (``ManifestPinMismatchError`` otherwise) and the
    file must hash to it; the PMC and NCBI URLs are retrieval metadata, never read here.
    """
    data_root = _data_root()
    manifest = load_manifest(data_root)
    os.makedirs(raw_dir, exist_ok=True)
    for relpath, expected in pins:
        check_manifest_pin(relpath, manifest_sha256(manifest, relpath), expected)
        src = raw_mirror_dir(data_root) / relpath
        if not src.exists():
            raise RuntimeError(f"required raw artifact missing from mirror: {src}")
        link_verified(src, osp.join(raw_dir, Path(relpath).name), expected)


def _raw_pins(pins: Iterable[tuple[str, str]]) -> dict[str, str]:
    """``{file name in raw/: sha256}`` for ``verify_raw_files``."""
    return {Path(relpath).name: sha for relpath, sha in pins}


def read_released_table(path: str | Path, rows: Sequence[SampleRow]) -> pd.DataFrame:
    """Table S2 or S3, whose columns must be ``rows``' samples in sheet order."""
    table = pd.read_csv(path, index_col=0)
    if list(table.columns) != [row.sample for row in rows]:
        raise RuntimeError(
            f"{Path(path).name}: columns are not the Table S1 samples with this assay's "
            "data, in sheet order"
        )
    if not table.index.is_unique:
        raise RuntimeError(f"{Path(path).name} repeats an identifier")
    return table


def protein_crosswalk(
    batch_paths: Iterable[str | Path], accessions: Sequence[str]
) -> dict[str, str]:
    """``YP_`` accession -> the ``ECB_`` locus tag its deposited NCBI record names.

    The batches must cover exactly ``accessions`` (Table S3's first column), each once.
    """
    crosswalk: dict[str, str] = {}
    for path in batch_paths:
        for accession, tag in genpept_locus_tags(path).items():
            if accession in crosswalk:
                raise ValueError(f"{accession} appears in two batches")
            crosswalk[accession] = tag
    if set(crosswalk) != set(accessions):
        missing = sorted(set(accessions) - set(crosswalk))
        extra = sorted(set(crosswalk) - set(accessions))
        raise RuntimeError(
            f"the NCBI batches miss {missing[:10]} and add {extra[:10]} against Table S3"
        )
    return crosswalk


def reconcile_rel606(
    genome: EcoliBREL606Genome, names: Sequence[str], *, label: str
) -> tuple[list[str], LocusTagReconciliation]:
    """Each name as a REL606 locus tag, at ``MIN_RESOLVED_FRACTION`` or the build stops."""
    stored, report = reconcile_locus_tags(genome, pd.Series(list(names)), label=label)
    report.require_resolved(MIN_RESOLVED_FRACTION)
    if report.outside_namespace:
        raise RuntimeError(f"{label}: not REL606 locus tags {report.outside_namespace}")
    tags = [str(tag) for tag in stored]
    if len(set(tags)) != len(tags):
        raise RuntimeError(f"{label}: two identifiers name one locus")
    return tags, report


def environment_cache() -> Callable[[SampleRow], Environment]:
    """``build_environment`` memoized on the cells that define an environment."""
    cache: dict[tuple[str, str, float, str, float, float], Environment] = {}

    def environment_of(row: SampleRow) -> Environment:
        key = (
            row.carbon_source,
            row.mg_level,
            row.mg_mm,
            row.na_level,
            row.na_mm,
            row.growth_time_hr,
        )
        if key not in cache:
            cache[key] = build_environment(row)
        return cache[key]

    return environment_of


def measured_gene_set(dataset: ExperimentDataset, key: str) -> GeneSet:
    """The loci a built dataset's phenotypes are keyed by (``phenotype[key]``).

    Every Caglar record is wild type, so no genotype names a gene and the base class's
    genotype scan would return an empty set, which it refuses. The dataset's genes are
    then the REL606 loci it measures.
    """
    if dataset.env is None:
        dataset._init_db()
    genes = GeneSet()
    with dataset.env.begin() as txn:
        for _, value in txn.cursor():
            genes.update(pickle.loads(value)["experiment"]["phenotype"][key])
    dataset.close_lmdb()
    return genes


def _record_samples(
    rows: Sequence[SampleRow],
    reconstruction: CountReconstruction,
    technical: Callable[[SampleRow], int],
) -> list[dict[str, Any]]:
    totals = reconstruction.counts.sum(axis=0)
    return [
        RecordSample(
            index=index,
            sample=row.sample,
            experiment=row.experiment,
            condition=row.condition,
            growth_phase=row.growth_phase,
            growth_time_hr=row.growth_time_hr,
            batch=row.batch,
            technical_replicates=technical(row),
            size_factor=float(reconstruction.size_factors[row.sample]),
            library_size=int(totals[row.sample]),
        ).model_dump(mode="json")
        for index, row in enumerate(rows)
    ]


# --------------------------------------------------------------------------- #
# Family 1: RNA-seq (Table S2)
# --------------------------------------------------------------------------- #
@register_dataset
class RnaseqCaglar2017Dataset(ExperimentDataset):
    """Caglar 2017 RNA-seq: one record per REL606 library (Table S2 sample)."""

    REFERENCE_STRAIN: ClassVar[Literal["REL606"]] = "REL606"

    def __init__(
        self,
        root: str = "data/torchcell/rnaseq_caglar2017",
        io_workers: int = 0,
        ecoli_genome: EcoliBREL606Genome | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; the REL606 genome is injected by the build or opened in process."""
        self.ecoli_genome = ecoli_genome
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @staticmethod
    def pins() -> list[tuple[str, str]]:
        """The mirror files this family reads: Tables S1 and S2."""
        return [_table_pin("S1"), _table_pin("S2")]

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return BacterialRNASeqExpressionExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return BacterialRNASeqExpressionExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """Tables S1 and S2, required before processing."""
        return list(_raw_pins(self.pins()))

    def download(self) -> None:
        """Link Tables S1 and S2 from the raw mirror after verifying their pins."""
        _link_mirror_files(self.raw_dir, self.pins())

    def _genome(self) -> EcoliBREL606Genome:
        """The injected REL606 genome, or the default cache opened read-only."""
        if self.ecoli_genome is None:
            self.ecoli_genome = bacterial_genome("ecoli", self.REFERENCE_STRAIN)
        return self.ecoli_genome

    def compute_gene_set(self) -> GeneSet:
        """The 4,196 REL606 loci the expression profiles are keyed by."""
        return measured_gene_set(self, "expression_tpm")

    @post_process
    def process(self) -> None:
        """Back-solve Table S2 to counts, then write one record per library."""
        require_pinnable_strain()
        verify_raw_files(self.raw_dir, _raw_pins(self.pins()))
        genome = self._genome()
        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        sheet = read_sample_sheet(osp.join(self.raw_dir, SI_TABLES["S1"][0]))
        rows = [row for row in sheet if row.rna_technical_replicates > 0]
        table = read_released_table(osp.join(self.raw_dir, SI_TABLES["S2"][0]), rows)
        reconstruction = back_solve_counts(table, label="table_s2")
        genes, report = reconcile_rel606(
            genome, [str(g) for g in table.index], label=f"{self.name} Table S2 genes"
        )
        lengths = gene_lengths(genome, genes)
        counts = reconstruction.counts.to_numpy(dtype=np.float64)

        pin = assembly_reference(self.REFERENCE_STRAIN)
        reference_members = reference_rows(rows)
        column = {row.sample: j for j, row in enumerate(rows)}
        references = {
            phase: BacterialRNASeqExpressionExperimentReference(
                dataset_name=self.name,
                genome_reference=pin,
                environment_reference=reference_environment(
                    phase, (r.growth_time_hr for r in members)
                ),
                phenotype_reference=rnaseq_reference_phenotype(
                    genes, counts[:, [column[r.sample] for r in members]], lengths
                ),
            )
            for phase, members in reference_members.items()
        }
        environment_of = environment_cache()
        genotype = Genotype(perturbations=[])
        # #771: the record cites the study that FIRST reported its sample, which for
        # the 27 glucose-time-course samples is Houser 2015, not this paper.
        pub_of = source_publications(rows)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for index, row in enumerate(rows):
                experiment = BacterialRNASeqExpressionExperiment(
                    dataset_name=self.name,
                    genotype=genotype,
                    environment=environment_of(row),
                    phenotype=rnaseq_phenotype(genes, counts[:, index], lengths),
                )
                txn.put(
                    f"{index}".encode(),
                    self._intern_record(
                        experiment,
                        references[row.growth_phase],
                        pub_of[row.sample],
                        itxn,
                    ),
                )
        env.close()
        interned_env.close()

        out = self.preprocess_dir
        _write_model(out, "vst_back_solve.json", reconstruction.evidence)
        _write_model(
            out,
            "log_base_derivation.json",
            log_base_derivation([reconstruction.evidence]),
        )
        _write_model(out, "locus_tag_reconciliation.json", report)
        _write_json(
            out,
            "gene_lengths.json",
            {g: int(n) for g, n in zip(genes, lengths, strict=True)},
        )
        _write_json(
            out,
            "record_samples.json",
            _record_samples(rows, reconstruction, lambda r: r.rna_technical_replicates),
        )
        _write_json(
            out,
            "replicate_groups.json",
            [g.model_dump(mode="json") for g in replicate_groups(rows)],
        )
        _write_json(out, "source_study_attribution.json", _attribution_ledger(rows))
        _write_model(
            out,
            "build_accounting.json",
            _accounting(
                self.name,
                "Table S2 (srep45303-s3.csv)",
                len(sheet),
                rows,
                reference_members,
                notes=[
                    "every Table S1 row with RNA data is a record (the authors kept all "
                    "152, including the two QC-flagged MURI_091 and MURI_130)",
                    "expression_count is the integer count back-solved from Table S2; "
                    "expression_tpm is computed here from it and the GenBank gene span",
                    "the reference of each record is its phase's glucose, base Mg2+ "
                    "and base Na+ condition, averaged over that condition's samples",
                ],
            ),
        )
        log.info(
            "Caglar 2017 RNA-seq: %d records over %d genes; back-solve max integer "
            "deviation %.3g",
            len(rows),
            len(genes),
            reconstruction.evidence.max_count_integer_deviation,
        )

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Family 2: proteome (Table S3)
# --------------------------------------------------------------------------- #
@register_dataset
class ProteomeCaglar2017Dataset(ExperimentDataset):
    """Caglar 2017 proteome: one record per REL606 LC-MS/MS sample (Table S3)."""

    REFERENCE_STRAIN: ClassVar[Literal["REL606"]] = "REL606"

    def __init__(
        self,
        root: str = "data/torchcell/proteome_caglar2017",
        io_workers: int = 0,
        ecoli_genome: EcoliBREL606Genome | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; the REL606 genome is injected by the build or opened in process."""
        self.ecoli_genome = ecoli_genome
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @staticmethod
    def pins() -> list[tuple[str, str]]:
        """Tables S1 and S3 and every NCBI batch that keys Table S3 to ``ECB_`` tags."""
        return [_table_pin("S1"), _table_pin("S3"), *_batch_pins()]

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return BacterialProteinAbundanceExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return BacterialProteinAbundanceExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """Tables S1 and S3 and the GenPept batches, required before processing."""
        return list(_raw_pins(self.pins()))

    def download(self) -> None:
        """Link the tables and batches from the raw mirror after verifying their pins."""
        _link_mirror_files(self.raw_dir, self.pins())

    def _genome(self) -> EcoliBREL606Genome:
        """The injected REL606 genome, or the default cache opened read-only."""
        if self.ecoli_genome is None:
            self.ecoli_genome = bacterial_genome("ecoli", self.REFERENCE_STRAIN)
        return self.ecoli_genome

    def compute_gene_set(self) -> GeneSet:
        """The 4,196 REL606 loci the abundance profiles are keyed by."""
        return measured_gene_set(self, "protein_abundance")

    @post_process
    def process(self) -> None:
        """Key Table S3 to REL606 loci, back-solve it, write one record per sample."""
        require_pinnable_strain()
        verify_raw_files(self.raw_dir, _raw_pins(self.pins()))
        genome = self._genome()
        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        sheet = read_sample_sheet(osp.join(self.raw_dir, SI_TABLES["S1"][0]))
        rows = [row for row in sheet if row.protein_technical_replicates > 0]
        table = read_released_table(osp.join(self.raw_dir, SI_TABLES["S3"][0]), rows)
        accessions = [str(a) for a in table.index]
        crosswalk = protein_crosswalk(
            (
                osp.join(self.raw_dir, Path(relpath).name)
                for relpath, _ in _batch_pins()
            ),
            accessions,
        )
        reconstruction = back_solve_counts(table, label="table_s3")
        proteins, report = reconcile_rel606(
            genome,
            [crosswalk[a] for a in accessions],
            label=f"{self.name} Table S3 proteins (YP_ -> ECB_ by NCBI record)",
        )
        counts = reconstruction.counts.to_numpy(dtype=np.float64)
        size_factors = reconstruction.size_factors.to_numpy(dtype=np.float64)
        normalized = counts / size_factors

        pin = assembly_reference(self.REFERENCE_STRAIN)
        reference_members = reference_rows(rows)
        column = {row.sample: j for j, row in enumerate(rows)}
        references = {
            phase: BacterialProteinAbundanceExperimentReference(
                dataset_name=self.name,
                genome_reference=pin,
                environment_reference=reference_environment(
                    phase, (r.growth_time_hr for r in members)
                ),
                phenotype_reference=protein_reference_phenotype(
                    proteins, normalized[:, [column[r.sample] for r in members]]
                ),
            )
            for phase, members in reference_members.items()
        }
        environment_of = environment_cache()
        genotype = Genotype(perturbations=[])
        # #771: see the RNA-seq family; the same 27 samples are Houser 2015's here.
        pub_of = source_publications(rows)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for index, row in enumerate(rows):
                experiment = BacterialProteinAbundanceExperiment(
                    dataset_name=self.name,
                    genotype=genotype,
                    environment=environment_of(row),
                    phenotype=protein_phenotype(
                        proteins, counts[:, index], float(size_factors[index])
                    ),
                )
                txn.put(
                    f"{index}".encode(),
                    self._intern_record(
                        experiment,
                        references[row.growth_phase],
                        pub_of[row.sample],
                        itxn,
                    ),
                )
        env.close()
        interned_env.close()

        out = self.preprocess_dir
        _write_model(out, "vst_back_solve.json", reconstruction.evidence)
        _write_model(
            out,
            "log_base_derivation.json",
            log_base_derivation([reconstruction.evidence]),
        )
        _write_model(out, "locus_tag_reconciliation.json", report)
        _write_json(
            out, "protein_crosswalk.json", {a: crosswalk[a] for a in accessions}
        )
        _write_json(
            out,
            "record_samples.json",
            _record_samples(
                rows, reconstruction, lambda r: r.protein_technical_replicates
            ),
        )
        _write_json(
            out,
            "replicate_groups.json",
            [g.model_dump(mode="json") for g in replicate_groups(rows)],
        )
        _write_json(out, "source_study_attribution.json", _attribution_ledger(rows))
        _write_model(
            out,
            "build_accounting.json",
            _accounting(
                self.name,
                "Table S3 (srep45303-s4.csv)",
                len(sheet),
                rows,
                reference_members,
                notes=[
                    "every Table S1 row with protein data is a record",
                    "protein_abundance is the back-solved integer spectral count over "
                    "the sample's DESeq2 size factor; an unobserved protein is the 0 "
                    "the authors wrote",
                    "n_replicates is 1 per record (one biological culture); Table S1's "
                    "technical-replicate count is in record_samples.json",
                    "the reference of each record is its phase's glucose, base Mg2+ "
                    "and base Na+ condition: mean, standard error and sample count",
                ],
            ),
        )
        log.info(
            "Caglar 2017 proteome: %d records over %d proteins; back-solve max "
            "integer deviation %.3g",
            len(rows),
            len(proteins),
            reconstruction.evidence.max_count_integer_deviation,
        )

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Family 3: the protein fold changes of Table S8
# --------------------------------------------------------------------------- #
#: Table S8's ``dataType`` cell of a protein row (the other value is ``mrna``).
S8_PROTEIN = "protein"
#: Table S8's ``dataType`` cell of a gene-level row.
S8_MRNA = "mrna"
#: The Table S8 columns the loader reads.
S8_COLUMNS: tuple[str, ...] = (
    "id",
    "baseMean",
    "log2FoldChange",
    "lfcSE",
    "pvalue",
    "padj",
    "dataType",
    "growthPhase.x",
    "test_for",
    "contrast",
    "base",
    "fullFileName",
    "investigatedEffect",
    "testVSbase",
)
#: Table S8's ``growthPhase.x`` cell -> the growth phase Table S1 names.
S8_PHASES: dict[str, GrowthPhase] = {
    "Exp": GrowthPhase.exponential,
    "Sta": GrowthPhase.stationary,
}
#: The substring that marks the doubling-time control model in ``investigatedEffect``.
DOUBLING_TIME_MARKER = "doublingTimeMinutes"
#: What a stored number is: DESeq2's Wald log2 fold change under the primary design.
FOLD_CHANGE_MEASUREMENT_TYPE = (
    "deseq2_wald_log2_fold_change_design_batch_plus_condition"
)
#: The correction behind ``padj``, back-solved from the released p-values
#: (``adjustment_back_solve``); the paper names only "FDR corrected" (``FDR_CORRECTED``).
P_VALUE_ADJUSTMENT_METHOD = "benjamini_hochberg"
#: A released ``padj`` further than this from the Benjamini-Hochberg value of the
#: group's own p-values refuses the build. Measured on the mirror over all 48 groups:
#: 1.04e-13 at worst.
ADJUSTMENT_TOLERANCE = 1e-9
#: The denominator of every Table S8 fold change (``FOLD_CHANGE_BASIS``).
FOLD_CHANGE_REFERENCE_BASIS = (
    "the base level reference condition of the same growth phase: glucose with 5 mM "
    "Na+ and 0.8 mM Mg2+ on Davis Minimal medium (DM500)"
)


class ContrastFactor(BaseModel):
    """How one ``test_for`` value of Table S8 selects its samples out of Table S1."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    test_for: str
    level_field: str = Field(description="the SampleRow field the level names")
    fixed: tuple[tuple[str, str], ...] = Field(
        description="SampleRow fields held at the base level in both groups"
    )
    dose_field: str | None = Field(
        description="the SampleRow field whose value the environment stores; None when "
        "the level itself is the edit (the carbon source, dosed by CARBON_SOURCE_SWAP)"
    )


#: One entry per ``test_for`` value Table S8 carries, measured on the mirror:
#: ``carbonSource``, ``Mg_mM_Levels`` and ``Na_mM_Levels``.
S8_FACTORS: dict[str, ContrastFactor] = {
    "carbonSource": ContrastFactor(
        test_for="carbonSource",
        level_field="carbon_source",
        fixed=(("mg_level", "baseMg"), ("na_level", "baseNa")),
        dose_field=None,
    ),
    "Mg_mM_Levels": ContrastFactor(
        test_for="Mg_mM_Levels",
        level_field="mg_level",
        fixed=(("carbon_source", BASE_CARBON), ("na_level", "baseNa")),
        dose_field="mg_mm",
    ),
    "Na_mM_Levels": ContrastFactor(
        test_for="Na_mM_Levels",
        level_field="na_level",
        fixed=(("carbon_source", BASE_CARBON), ("mg_level", "baseMg")),
        dose_field="na_mm",
    ),
}


def benjamini_hochberg(p_values: FloatArray) -> FloatArray:
    """Benjamini-Hochberg adjusted p-values of ``p_values`` (step-up, capped at 1)."""
    n = p_values.size
    order = np.argsort(p_values, kind="stable")
    scaled = p_values[order] * n / np.arange(1, n + 1, dtype=np.float64)
    monotone = np.minimum.accumulate(scaled[::-1])[::-1]
    out = np.empty(n, dtype=np.float64)
    out[order] = np.minimum(monotone, 1.0)
    return out


class AdjustmentBackSolve(BaseModel):
    """The evidence that Table S8's ``padj`` is the Benjamini-Hochberg value of its
    own ``pvalue`` column, per group.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: str
    groups: int
    rows_with_both: int = Field(
        description="Rows carrying a pvalue and a padj (DESeq2's independent filtering "
        "writes NA padj for the rest)."
    )
    max_abs_deviation: float
    worst_group: str


def adjustment_back_solve(frame: pd.DataFrame) -> AdjustmentBackSolve:
    """Back-solve the multiple-testing correction from the released columns.

    The paper names the correction only as "FDR corrected" (``FDR_CORRECTED``) and
    defers the procedure to DESeq2, which is not mirrored, so the method is MEASURED
    here: within each ``fullFileName`` group, the Benjamini-Hochberg adjustment of the
    rows that carry both a ``pvalue`` and a ``padj`` must reproduce ``padj``. A group
    that misses by more than ``ADJUSTMENT_TOLERANCE`` refuses the build.
    """
    worst = -1.0
    worst_group = ""
    rows = 0
    groups = 0
    for name, group in frame.groupby("fullFileName", sort=True):
        tested = group[group["pvalue"].notna() & group["padj"].notna()]
        groups += 1
        rows += len(tested)
        if tested.empty:
            raise RuntimeError(f"{name}: no row carries both a pvalue and a padj")
        deviation = float(
            np.abs(
                benjamini_hochberg(tested["pvalue"].to_numpy(dtype=np.float64))
                - tested["padj"].to_numpy(dtype=np.float64)
            ).max()
        )
        if deviation > worst:
            worst = deviation
            worst_group = str(name)
    if worst > ADJUSTMENT_TOLERANCE:
        raise RuntimeError(
            f"{worst_group}: padj differs from the Benjamini-Hochberg value of its own "
            f"p-values by {worst} (tolerance {ADJUSTMENT_TOLERANCE}); the released "
            "correction is not Benjamini-Hochberg"
        )
    return AdjustmentBackSolve(
        method=P_VALUE_ADJUSTMENT_METHOD,
        groups=groups,
        rows_with_both=rows,
        max_abs_deviation=worst,
        worst_group=worst_group,
    )


class ContrastGroup(BaseModel):
    """One Table S8 group: its released descriptors and its 4,196 rows' file name."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    file_name: str = Field(description="Table S8 ``fullFileName``, the group's key")
    test_vs_base: str
    growth_phase: GrowthPhase
    test_for: str
    contrast: str = Field(description="the test level")
    base: str = Field(description="the reference level")
    investigated_effect: str
    rows: int

    @property
    def primary_control_model(self) -> bool:
        """True for the paper's primary design, batch only (``PRIMARY_DESIGN``)."""
        return DOUBLING_TIME_MARKER not in self.investigated_effect


def read_table_s8_protein(path: str | Path) -> pd.DataFrame:
    """Table S8's protein rows, with the columns the loader reads.

    A ``dataType`` cell of neither known form, or a ``growthPhase.x`` cell that is not
    a phase Table S8 carries, raises rather than being skipped.
    """
    frame = pd.read_csv(path, usecols=list(S8_COLUMNS))
    types = set(frame["dataType"])
    if types != {S8_PROTEIN, S8_MRNA}:
        raise RuntimeError(f"{Path(path).name}: dataType values {sorted(types)}")
    protein = frame[frame["dataType"] == S8_PROTEIN].copy()
    phases = set(protein["growthPhase.x"])
    if not phases <= set(S8_PHASES):
        raise RuntimeError(
            f"{Path(path).name}: protein rows carry growth phases {sorted(phases)}"
        )
    factors = set(protein["test_for"])
    if not factors <= set(S8_FACTORS):
        raise RuntimeError(
            f"{Path(path).name}: protein rows test for {sorted(factors)}"
        )
    return protein


def contrast_groups(protein: pd.DataFrame) -> list[ContrastGroup]:
    """Every protein-level Table S8 group, in file-name order.

    A group whose descriptor columns are not constant over its rows raises: the group
    key is the file name, and the descriptors are what the record is built from.
    """
    groups: list[ContrastGroup] = []
    for name, rows in protein.groupby("fullFileName", sort=True):
        cells = {
            column: sorted(set(rows[column]))
            for column in (
                "testVSbase",
                "growthPhase.x",
                "test_for",
                "contrast",
                "base",
                "investigatedEffect",
            )
        }
        varying = {k: v for k, v in cells.items() if len(v) != 1}
        if varying:
            raise RuntimeError(f"{name}: descriptor columns vary {varying}")
        groups.append(
            ContrastGroup(
                file_name=str(name),
                test_vs_base=cells["testVSbase"][0],
                growth_phase=S8_PHASES[cells["growthPhase.x"][0]],
                test_for=cells["test_for"][0],
                contrast=cells["contrast"][0],
                base=cells["base"][0],
                investigated_effect=cells["investigatedEffect"][0],
                rows=len(rows),
            )
        )
    return groups


def table_s8_protein_order(
    protein: pd.DataFrame, groups: Sequence[ContrastGroup]
) -> list[str]:
    """The ``YP_`` accessions every protein group lists, in their shared file order.

    Measured on the mirror: all 24 protein groups carry the same 4,196 accessions in
    the same order, which is Table S3's order, so one crosswalk and one reconciliation
    key every record. A group in another order raises rather than being re-sorted.
    """
    orders = {
        group.file_name: [
            str(value)
            for value in protein.loc[protein["fullFileName"] == group.file_name, "id"]
        ]
        for group in groups
    }
    order = orders[groups[0].file_name]
    differing = sorted(name for name, ids in orders.items() if ids != order)
    if differing:
        raise RuntimeError(f"Table S8 groups {differing} list the ids in another order")
    if len(set(order)) != len(order):
        raise RuntimeError("a Table S8 protein group repeats an accession")
    return order


class ContrastSamples(BaseModel):
    """The Table S1 protein samples of one contrast's two groups, and its dose."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    test_samples: tuple[str, ...]
    base_samples: tuple[str, ...]
    test_times_hr: tuple[float, ...]
    base_times_hr: tuple[float, ...]
    doses: tuple[float, ...] = Field(
        description="the distinct values of the factor's dose field over the test "
        "samples; empty when the level itself is the edit"
    )


def contrast_samples(
    rows: Sequence[SampleRow], group: ContrastGroup
) -> ContrastSamples:
    """The protein samples Table S1 puts in a contrast's test and base groups.

    Selected by the group's own released cells: the factor's level field at
    ``contrast`` or ``base``, every other factor at its base level
    (``S8_FACTORS``), in the group's growth phase. An empty group raises.
    """
    factor = S8_FACTORS[group.test_for]

    def members(level: str) -> list[SampleRow]:
        return [
            row
            for row in rows
            if row.growth_phase is group.growth_phase
            and getattr(row, factor.level_field) == level
            and all(getattr(row, field) == value for field, value in factor.fixed)
        ]

    test = members(group.contrast)
    base = members(group.base)
    if not test or not base:
        raise RuntimeError(
            f"{group.file_name}: Table S1 puts {len(test)} samples in the "
            f"{group.contrast} group and {len(base)} in the {group.base} group"
        )
    doses = (
        ()
        if factor.dose_field is None
        else tuple(sorted({float(getattr(row, factor.dose_field)) for row in test}))
    )
    return ContrastSamples(
        test_samples=tuple(row.sample for row in test),
        base_samples=tuple(row.sample for row in base),
        test_times_hr=tuple(sorted({row.growth_time_hr for row in test})),
        base_times_hr=tuple(sorted({row.growth_time_hr for row in base})),
        doses=doses,
    )


class RefusedGroup(BaseModel):
    """A Table S8 group the loader does not turn into a record, and the measurement
    that refused it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    file_name: str
    test_vs_base: str
    growth_phase: str
    investigated_effect: str
    rows: int
    reason: str
    measurement: str


def refuse_control_model(group: ContrastGroup) -> RefusedGroup:
    """The doubling-time parameterization of a contrast the primary design also covers."""
    return RefusedGroup(
        file_name=group.file_name,
        test_vs_base=group.test_vs_base,
        growth_phase=group.growth_phase.value,
        investigated_effect=group.investigated_effect,
        rows=group.rows,
        reason="secondary control model",
        measurement="investigatedEffect adds doublingTimeMinutes to the primary "
        f"design; its test_for, contrast, base and growth phase equal those of "
        f"{group.investigated_effect.replace('PLUS' + DOUBLING_TIME_MARKER, '')}, so "
        "the two groups are two estimates of one genotype x environment cell under "
        "two statistical control parameterizations",
    )


def refuse_pooled_dose(group: ContrastGroup, samples: ContrastSamples) -> RefusedGroup:
    """A level whose test samples hold several doses, so no single dose is honest."""
    factor = S8_FACTORS[group.test_for]
    return RefusedGroup(
        file_name=group.file_name,
        test_vs_base=group.test_vs_base,
        growth_phase=group.growth_phase.value,
        investigated_effect=group.investigated_effect,
        rows=group.rows,
        reason="level pools several doses",
        measurement=f"Table S1 puts {len(samples.test_samples)} protein samples in the "
        f"{group.contrast} group at {factor.dose_field} "
        f"{[f'{d:g}' for d in samples.doses]}; SmallMoleculePerturbation requires one "
        "Concentration and the released cell is a level, not a dose",
    )


#: Provenance of the Table S8 group a replicate count is measured against.
S8_PROVENANCE = Provenance(
    source_uri=f"{RAW_DIR_REL}/{si_table_relpath('S8')}",
    citation_key=CITATION_KEY,
    sha256=SI_TABLES["S8"][1],
    method="Table S1's protein samples selected by the group's own released cells "
    "(contrast_samples)",
    page="Table S8 fullFileName (the set id 'set00_StcYtcNasAgrNgrMgh')",
)


def replicate_derivations(built: Sequence[Mapping[str, Any]]) -> list[StatDerivation]:
    """``n_replicates`` as one ``StatDerivation`` per record: the conservative lower end.

    The samples of a DESeq2 fit are not released. Table S8 names them only by the set
    id inside ``fullFileName``, whose definition is in the authors' code, and ``lfcSE``
    does not invert to a sample count, so a back-solve is precluded. The stored value is
    therefore the number of protein samples Table S1 puts in the contrast's TEST group,
    the low end of the range whose high end is both groups: the base group's samples
    enter the same two-group difference and DESeq2 pools dispersion over the whole fit,
    so the true n is larger and this end never overstates the precision (CLAUDE.md
    "Resolving a range with no per-record value", rule 2).
    """
    return [
        StatDerivation(
            field=f"n_replicates[{record['file_name']}]",
            method=DerivationMethod.conservative_low,
            value=float(record["n_test_samples"]),
            range_low=float(record["n_test_samples"]),
            range_high=float(
                int(record["n_test_samples"]) + int(record["n_base_samples"])
            ),
            statistic="protein samples Table S1 puts in the contrast's test group, in "
            "the record's growth phase",
            diagnostics={
                "n_test_samples": float(record["n_test_samples"]),
                "n_base_samples": float(record["n_base_samples"]),
            },
            provenance=S8_PROVENANCE,
            rationale="the fit's sample set is encoded only in the unreleased set id "
            "of fullFileName and lfcSE does not invert to n, so a back-solve is "
            "precluded; the test group alone is the low end of the two groups the "
            "contrast differences",
        )
        for record in built
    ]


def contrast_duration_gap(
    group: ContrastGroup, times: Sequence[float], *, level: str
) -> ProvenanceGap:
    """Why a contrast's environment has no ``duration_hours``."""
    stated = ", ".join(f"{t:g}" for t in times)
    return ProvenanceGap(
        field="duration_hours",
        reason=ProvenanceGapReason.not_reported_by_primary,
        looked_in=EXPONENTIAL_SAMPLING.provenance,
        note=f"the {group.growth_phase.value} {level} group of "
        f"{group.test_vs_base} pools samples Table S1 dates at {stated} h; the paper "
        "sets the phase by optical density, so no single duration describes the group",
    )


def contrast_environment(group: ContrastGroup, samples: ContrastSamples) -> Environment:
    """The test condition of one contrast: medium, the level's edit, 37 C, aerobic.

    The edits are the existing per-sample builders (``carbon_source_perturbation``,
    ``magnesium_perturbation``, ``sodium_perturbation``) at the one dose the test
    group holds; a group that pools doses never reaches here (``refuse_pooled_dose``).
    """
    factor = S8_FACTORS[group.test_for]
    perturbations: list[EnvironmentPerturbationType] = []
    if factor.dose_field is None:
        media = DM500 if group.contrast == BASE_CARBON else DAVIS_MINIMAL
        if group.contrast != BASE_CARBON:
            perturbations.append(carbon_source_perturbation(group.contrast))
    else:
        media = DM500
        if len(samples.doses) != 1:
            raise RuntimeError(
                f"{group.file_name}: {factor.dose_field} {list(samples.doses)}"
            )
        dose = samples.doses[0]
        perturbations.append(
            magnesium_perturbation(dose)
            if factor.dose_field == "mg_mm"
            else sodium_perturbation(dose)
        )
    return Environment(
        media=media,
        temperature=Temperature(value=float(CULTURE_CONDITIONS.value)),
        perturbations=perturbations,
        aerobicity="aerobic",
        provenance_gaps=[
            contrast_duration_gap(group, samples.test_times_hr, level=group.contrast)
        ],
    )


def protein_fold_change_phenotype(
    proteins: Sequence[str], values: pd.DataFrame, n_test_samples: int
) -> ProteinFoldChangePhenotype:
    """One contrast's log2 fold changes, keyed by the REL606 loci that carry a value.

    A protein the group left blank is simply not a key: DESeq2 writes no fold change
    where the fit's base mean is 0, and sets ``pvalue`` and ``padj`` to NA for a row it
    flagged as an outlier or filtered out, so the three maps are nested subsets of the
    4,196 rows rather than padded with a neutral 0.
    """

    def column(name: str) -> dict[str, float]:
        return {
            protein: float(value)
            for protein, value in zip(proteins, values[name], strict=True)
            if not math.isnan(float(value))
        }

    fold = column("log2FoldChange")
    if not fold:
        raise RuntimeError("the group carries no fold change")
    return ProteinFoldChangePhenotype(
        protein_fold_change=fold,
        fold_change_scale=FoldChangeScale.log2,
        reference_basis=FOLD_CHANGE_REFERENCE_BASIS,
        protein_fold_change_se={p: v for p, v in column("lfcSE").items() if p in fold},
        protein_fold_change_p_value={
            p: v for p, v in column("pvalue").items() if p in fold
        },
        protein_fold_change_p_value_adjusted={
            p: v for p, v in column("padj").items() if p in fold
        },
        p_value_adjustment_method=P_VALUE_ADJUSTMENT_METHOD,
        n_replicates=dict.fromkeys(fold, n_test_samples),
        measurement_type=FOLD_CHANGE_MEASUREMENT_TYPE,
    )


def fold_change_reference_phenotype(
    phenotype: ProteinFoldChangePhenotype, n_base_samples: int
) -> ProteinFoldChangePhenotype:
    """The denominator's own phenotype: the neutral value of the record's scale.

    Not a measurement (``ProteinFoldChangePhenotype.neutral_reference``): a log2 fold
    change's reference is 0 by definition, so experiment minus reference reproduces the
    released number exactly and nothing is imputed.
    """
    neutral = phenotype.neutral_reference()
    return ProteinFoldChangePhenotype(
        protein_fold_change=neutral,
        fold_change_scale=phenotype.fold_change_scale,
        reference_basis=phenotype.reference_basis,
        n_replicates=dict.fromkeys(neutral, n_base_samples),
        measurement_type=phenotype.measurement_type,
    )


class FoldChangeRecord(BaseModel):
    """What one fold-change record was built from."""

    model_config = ConfigDict(extra="forbid")

    index: int
    file_name: str
    test_vs_base: str
    growth_phase: GrowthPhase
    test_for: str
    contrast: str
    base: str
    investigated_effect: str
    n_test_samples: int
    n_base_samples: int
    test_samples: list[str]
    base_samples: list[str]
    test_times_hr: list[float]
    doses: list[float]
    n_fold_change: int
    n_se: int
    n_p_value: int
    n_p_value_adjusted: int


class FoldChangeAccounting(BaseModel):
    """The retention arithmetic of the fold-change build, group by group."""

    model_config = ConfigDict(extra="forbid")

    dataset: str
    table: str
    table_rows: int
    mrna_rows: int
    protein_rows: int
    candidate_groups: int
    kept_records: int
    refused_groups: int
    refused_rows: int
    refusals_by_reason: dict[str, int]
    refused: list[RefusedGroup]
    kept_by_growth_phase: dict[str, int]
    kept_by_test_for: dict[str, int]
    notes: list[str]

    def check(self) -> None:
        """Kept plus refused is every candidate group, and the reasons add up."""
        if self.kept_records + self.refused_groups != self.candidate_groups:
            raise RuntimeError(f"{self.dataset}: retention does not add up")
        if sum(self.refusals_by_reason.values()) != self.refused_groups:
            raise RuntimeError(f"{self.dataset}: refusal reasons do not add up")
        if sum(r.rows for r in self.refused) != self.refused_rows:
            raise RuntimeError(f"{self.dataset}: refused rows do not add up")


@register_dataset
class ProteinFoldChangeCaglar2017Dataset(ExperimentDataset):
    """Caglar 2017 differential proteomics: one record per Table S8 protein contrast.

    The protein arm of Table S8 under the paper's primary design (``PRIMARY_DESIGN``),
    one record per (growth phase, test level) whose test condition Table S1 states at a
    single dose. The gene-level arm, the doubling-time control model and the two
    pooled-dose salt levels are refused in ``build_accounting.json``.
    """

    REFERENCE_STRAIN: ClassVar[Literal["REL606"]] = "REL606"

    def __init__(
        self,
        root: str = "data/torchcell/protein_fold_change_caglar2017",
        io_workers: int = 0,
        ecoli_genome: EcoliBREL606Genome | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; the REL606 genome is injected by the build or opened in process."""
        self.ecoli_genome = ecoli_genome
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @staticmethod
    def pins() -> list[tuple[str, str]]:
        """Tables S1 and S8 and every NCBI batch that keys ``YP_`` to ``ECB_`` tags."""
        return [_table_pin("S1"), _table_pin("S8"), *_batch_pins()]

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return BacterialProteinFoldChangeExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return BacterialProteinFoldChangeExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """Tables S1 and S8 and the GenPept batches, required before processing."""
        return list(_raw_pins(self.pins()))

    def download(self) -> None:
        """Link the tables and batches from the raw mirror after verifying their pins."""
        _link_mirror_files(self.raw_dir, self.pins())

    def _genome(self) -> EcoliBREL606Genome:
        """The injected REL606 genome, or the default cache opened read-only."""
        if self.ecoli_genome is None:
            self.ecoli_genome = bacterial_genome("ecoli", self.REFERENCE_STRAIN)
        return self.ecoli_genome

    def compute_gene_set(self) -> GeneSet:
        """The REL606 loci the fold-change profiles are keyed by."""
        return measured_gene_set(self, "protein_fold_change")

    @post_process
    def process(self) -> None:
        """Keep each admissible Table S8 protein contrast as one record."""
        require_pinnable_strain()
        verify_raw_files(self.raw_dir, _raw_pins(self.pins()))
        genome = self._genome()
        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        sheet = read_sample_sheet(osp.join(self.raw_dir, SI_TABLES["S1"][0]))
        rows = [row for row in sheet if row.protein_technical_replicates > 0]
        s8_path = osp.join(self.raw_dir, SI_TABLES["S8"][0])
        data_types = pd.read_csv(s8_path, usecols=["dataType"])["dataType"]
        mrna_rows = int((data_types == S8_MRNA).sum())
        protein = read_table_s8_protein(s8_path)
        adjustment = adjustment_back_solve(protein)
        groups = contrast_groups(protein)

        order = table_s8_protein_order(protein, groups)
        crosswalk = protein_crosswalk(
            (
                osp.join(self.raw_dir, Path(relpath).name)
                for relpath, _ in _batch_pins()
            ),
            order,
        )
        proteins, report = reconcile_rel606(
            genome,
            [crosswalk[a] for a in order],
            label=f"{self.name} Table S8 proteins (YP_ -> ECB_ by NCBI record)",
        )

        pin = assembly_reference(self.REFERENCE_STRAIN)
        reference_members = reference_rows(rows)
        genotype = Genotype(perturbations=[])
        pub = publication()
        refused: list[RefusedGroup] = []
        kept: list[tuple[ContrastGroup, ContrastSamples]] = []
        for group in groups:
            if not group.primary_control_model:
                refused.append(refuse_control_model(group))
                continue
            samples = contrast_samples(rows, group)
            if (
                S8_FACTORS[group.test_for].dose_field is not None
                and len(samples.doses) != 1
            ):
                refused.append(refuse_pooled_dose(group, samples))
                continue
            kept.append((group, samples))

        built: list[dict[str, Any]] = []
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for index, (group, samples) in enumerate(kept):
                values = (
                    protein[protein["fullFileName"] == group.file_name]
                    .set_index("id")
                    .reindex(order)
                )
                phenotype = protein_fold_change_phenotype(
                    proteins, values, len(samples.test_samples)
                )
                members = reference_members[group.growth_phase]
                if {row.sample for row in members} != set(samples.base_samples):
                    raise RuntimeError(
                        f"{group.file_name}: its base group is not the phase's "
                        "reference condition"
                    )
                reference = BacterialProteinFoldChangeExperimentReference(
                    dataset_name=self.name,
                    genome_reference=pin,
                    environment_reference=reference_environment(
                        group.growth_phase, (r.growth_time_hr for r in members)
                    ),
                    phenotype_reference=fold_change_reference_phenotype(
                        phenotype, len(samples.base_samples)
                    ),
                )
                experiment = BacterialProteinFoldChangeExperiment(
                    dataset_name=self.name,
                    genotype=genotype,
                    environment=contrast_environment(group, samples),
                    phenotype=phenotype,
                )
                txn.put(
                    f"{index}".encode(),
                    self._intern_record(experiment, reference, pub, itxn),
                )
                built.append(
                    FoldChangeRecord(
                        index=index,
                        file_name=group.file_name,
                        test_vs_base=group.test_vs_base,
                        growth_phase=group.growth_phase,
                        test_for=group.test_for,
                        contrast=group.contrast,
                        base=group.base,
                        investigated_effect=group.investigated_effect,
                        n_test_samples=len(samples.test_samples),
                        n_base_samples=len(samples.base_samples),
                        test_samples=list(samples.test_samples),
                        base_samples=list(samples.base_samples),
                        test_times_hr=list(samples.test_times_hr),
                        doses=list(samples.doses),
                        n_fold_change=len(phenotype.protein_fold_change),
                        n_se=len(phenotype.protein_fold_change_se or {}),
                        n_p_value=len(phenotype.protein_fold_change_p_value or {}),
                        n_p_value_adjusted=len(
                            phenotype.protein_fold_change_p_value_adjusted or {}
                        ),
                    ).model_dump(mode="json")
                )
        env.close()
        interned_env.close()

        out = self.preprocess_dir
        _write_model(out, "adjustment_back_solve.json", adjustment)
        _write_json(
            out,
            "n_replicates_derivation.json",
            [d.model_dump(mode="json") for d in replicate_derivations(built)],
        )
        _write_model(out, "locus_tag_reconciliation.json", report)
        _write_json(out, "protein_crosswalk.json", {a: crosswalk[a] for a in order})
        _write_json(out, "fold_change_records.json", built)
        accounting = FoldChangeAccounting(
            dataset=self.name,
            table=f"Table S8 ({SI_TABLES['S8'][0]})",
            table_rows=int(data_types.size),
            mrna_rows=mrna_rows,
            protein_rows=len(protein),
            candidate_groups=len(groups),
            kept_records=len(kept),
            refused_groups=len(refused),
            refused_rows=sum(r.rows for r in refused),
            refusals_by_reason=_counts_of(r.reason for r in refused),
            refused=refused,
            kept_by_growth_phase=_counts_of(g.growth_phase.value for g, _ in kept),
            kept_by_test_for=_counts_of(g.test_for for g, _ in kept),
            notes=[
                "the gene-level arm of Table S8 is out of scope: no gene-level "
                f"expression fold-change phenotype exists, and its {mrna_rows} rows "
                "would need one",
                "a record is one protein-level contrast under the paper's primary "
                "design, batch only; the doubling-time parameterization of the same "
                "contrast is a second estimate of one genotype x environment cell",
                "n_replicates is the contrast's TEST-group protein-sample count from "
                "Table S1, the conservative lower end "
                "(n_replicates_derivation.json)",
                "the reference phenotype is the neutral value of the log2 scale for "
                "every stored key, which is what a fold change's denominator carries "
                "by definition",
            ],
        )
        accounting.check()
        _write_model(out, "build_accounting.json", accounting)
        log.info(
            "Caglar 2017 protein fold changes: %d records from %d protein groups "
            "(%d refused, %d rows); padj reproduced to %.3g",
            len(kept),
            len(groups),
            len(refused),
            accounting.refused_rows,
            adjustment.max_abs_deviation,
        )

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Verification (L0-L4) of the built dev LMDBs
# --------------------------------------------------------------------------- #
Family = Literal["rnaseq", "proteome", "protein_fold_change"]
DATASET_SLUGS: dict[Family, str] = {
    "rnaseq": "rnaseq_caglar2017",
    "proteome": "proteome_caglar2017",
    "protein_fold_change": "protein_fold_change_caglar2017",
}
REL606_PIN = ("ecoli_B_REL606_ASM1798v1", "GCA_000017985.1")


def sourced_values() -> dict[str, SourcedValue]:
    """Every module-level ``SourcedValue``, by name (what the audit re-reads)."""
    return {
        name: value
        for name, value in globals().items()
        if isinstance(value, SourcedValue)
    }


#: The supplementary table each family's verifier re-reads.
FAMILY_TABLES: dict[Family, str] = {
    "rnaseq": "S2",
    "proteome": "S3",
    "protein_fold_change": "S8",
}


def _verifier_provenance(family: Family) -> Provenance:
    table = FAMILY_TABLES[family]
    obj, sha, _ = SI_TABLES[table]
    return Provenance(
        source_uri=f"{RAW_DIR_REL}/data/{obj}",
        citation_key=CITATION_KEY,
        sha256=sha,
        method=(
            "Released DESeq2 log2 fold changes, standard errors and p-values read "
            "verbatim, with the multiple-testing correction back-solved "
            "(adjustment_back_solve)"
            if family == "protein_fold_change"
            else "Table values inverted through the base-2 DESeq2 VST to integer "
            "counts (back_solve_counts)"
        ),
        page=f"Supplementary Table {table}",
    )


def _l1_distinct_profiles(
    records: Sequence[Mapping[str, Any]], key: str
) -> LevelResult:
    """L1 per sample: no two records carry the same measured profile."""
    profiles = Counter(
        json.dumps(r["experiment"]["phenotype"][key], sort_keys=True) for r in records
    )
    repeated = sum(n for n in profiles.values() if n > 1)
    return LevelResult(
        level=Level.L1,
        name="sample_uniqueness",
        passed=repeated == 0,
        message=f"{len(profiles)} distinct {key} profiles over {len(records)} records",
        details={"n_records": len(records), "n_in_repeated_profiles": repeated},
    )


def _l3_assembly_pin(records: Sequence[Mapping[str, Any]]) -> LevelResult:
    pins = sorted(
        {
            (
                str(r["reference"]["genome_reference"]["assembly_set"]),
                str(r["reference"]["genome_reference"]["assembly_accession"]),
            )
            for r in records
        }
    )
    return LevelResult(
        level=Level.L3,
        name="assembly_pin",
        passed=pins == [REL606_PIN],
        message=f"assembly pins {pins}",
        details={"pins": pins},
    )


def _l3_back_solve(evidence: VstBackSolve) -> LevelResult:
    passed = (
        evidence.max_count_integer_deviation <= COUNT_INTEGER_TOLERANCE
        and evidence.max_size_factor_relative_deviation <= SIZE_FACTOR_TOLERANCE
        and evidence.log_base == VST_LOG_BASE
    )
    return LevelResult(
        level=Level.L3,
        name="log_base_back_solve",
        passed=passed,
        message=f"{evidence.table}: base {evidence.log_base}; counts within "
        f"{evidence.max_count_integer_deviation:.3g} of integers, size factors within "
        f"{evidence.max_size_factor_relative_deviation:.3g} of DESeq2's",
        details=evidence.model_dump(),
    )


def _l4_containment(measured: set[str], universe: Collection[str]) -> LevelResult:
    outside = sorted(measured - set(universe))
    return LevelResult(
        level=Level.L4,
        name="gene_containment_rel606",
        passed=not outside,
        message=f"{len(measured) - len(outside)} of {len(measured)} measured loci are "
        "REL606 GenBank gene rows",
        details={"outside": outside[:20], "n_universe": len(universe)},
    )


def verify_rnaseq_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_count: int,
    gene_universe: Collection[str],
    back_solve: VstBackSolve,
) -> VerificationReport:
    """The L0-L4 gate of the RNA-seq family, one record per library.

    The family verifier keys L1 on one record per (strain, environment), which a
    replicate-level dataset breaks by design: the replicates of a condition share both.
    L1 here is the count plus distinct count profiles.
    """
    from pydantic import TypeAdapter

    from torchcell.datamodels.schema import ExperimentType

    report = VerificationReport(
        dataset_name=DATASET_SLUGS["rnaseq"], provenance=_verifier_provenance("rnaseq")
    )
    validate: Callable[[Any], object] = TypeAdapter(ExperimentType).validate_python
    report.add(l0_structural((r["experiment"] for r in records), validate))
    report.add(l1_count(len(records), expected_count))
    report.add(_l1_distinct_profiles(records, "expression_count"))
    phenotypes = [r["experiment"]["phenotype"] for r in records]
    fidelity = l2_value_fidelity(
        (float(v) for p in phenotypes for v in p["expression_tpm"].values()),
        allow_nan=False,
        minimum=0.0,
    )
    report.add(fidelity.model_copy(update={"name": "tpm_value_fidelity"}))
    bad_counts = sum(
        1
        for p in phenotypes
        for v in p["expression_count"].values()
        if not isinstance(v, int) or isinstance(v, bool) or v < 0
    )
    report.add(
        LevelResult(
            level=Level.L2,
            name="count_value_fidelity",
            passed=bad_counts == 0,
            message=f"{bad_counts} counts are not non-negative integers",
            details={"n_bad": bad_counts},
        )
    )
    totals = [sum(p["expression_tpm"].values()) for p in phenotypes]
    off_scale = [t for t in totals if not math.isclose(t, TPM_TOTAL, rel_tol=1e-9)]
    report.add(
        LevelResult(
            level=Level.L2,
            name="tpm_scale",
            passed=not off_scale,
            message=f"{len(totals) - len(off_scale)}/{len(totals)} records sum to one "
            "million TPM",
            details={"off_scale": off_scale[:10]},
        )
    )
    types = sorted({p["measurement_type"] for p in phenotypes})
    report.add(
        LevelResult(
            level=Level.L3,
            name="measurement_type_consistent",
            passed=types == [RNA_MEASUREMENT_TYPE],
            message=f"measurement types {types}",
            details={"measurement_types": types},
        )
    )
    references = {
        json.dumps(
            r["reference"]["phenotype_reference"]["expression_tpm"], sort_keys=True
        )
        for r in records
    }
    reference_bad = 0
    for text in references:
        values = [float(v) for v in json.loads(text).values()]
        finite = all(math.isfinite(v) and v >= 0 for v in values)
        if not finite or not math.isclose(sum(values), TPM_TOTAL, rel_tol=1e-9):
            reference_bad += 1
    report.add(
        LevelResult(
            level=Level.L3,
            name="reference_tpm",
            passed=reference_bad == 0,
            message=f"{len(references) - reference_bad}/{len(references)} references "
            "are finite, non-negative and sum to one million TPM",
            details={"n_references": len(references), "n_bad": reference_bad},
        )
    )
    report.add(_l3_assembly_pin(records))
    report.add(_l3_back_solve(back_solve))
    measured = {g for p in phenotypes for g in p["expression_tpm"]}
    report.add(_l4_containment(measured, gene_universe))
    return report


def verify_proteome_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_count: int,
    gene_universe: Collection[str],
    back_solve: VstBackSolve,
) -> VerificationReport:
    """The protein family verifier (L0-L3) plus distinct profiles, the abundance floor,
    the assembly pin, the back-solve and the REL606 containment (L4).
    """
    from torchcell.verification.protein import verify_protein_dataset

    report = verify_protein_dataset(
        [dict(r) for r in records],
        dataset_name=DATASET_SLUGS["proteome"],
        provenance=_verifier_provenance("proteome"),
        expected_count=expected_count,
    )
    report.add(_l1_distinct_profiles(records, "protein_abundance"))
    floor = l2_value_fidelity(
        (
            float(v)
            for r in records
            for v in r["experiment"]["phenotype"]["protein_abundance"].values()
        ),
        minimum=0.0,
    )
    report.add(floor.model_copy(update={"name": "abundance_nonnegative"}))
    report.add(_l3_assembly_pin(records))
    report.add(_l3_back_solve(back_solve))
    measured = {
        p for r in records for p in r["experiment"]["phenotype"]["protein_abundance"]
    }
    report.add(_l4_containment(measured, gene_universe))
    return report


def _l3_adjustment_back_solve(evidence: AdjustmentBackSolve) -> LevelResult:
    """L3: the released ``padj`` is its group's own Benjamini-Hochberg adjustment."""
    passed = evidence.max_abs_deviation <= ADJUSTMENT_TOLERANCE
    return LevelResult(
        level=Level.L3,
        name="p_value_adjustment_back_solve",
        passed=passed,
        message=f"{evidence.method} reproduces padj over {evidence.groups} groups and "
        f"{evidence.rows_with_both} rows to {evidence.max_abs_deviation:.3g}",
        details=evidence.model_dump(),
    )


def _l1_distinct_contrast_environments(
    records: Sequence[Mapping[str, Any]],
) -> LevelResult:
    """L1 for an environment-contrast panel: one record per (environment, reference).

    The shared gate's contrast key is the genotype and the denominator, which a
    wild-type panel shares across every record by design: here the contrast IS the
    environment, and the growth phase rides on the environment's and the reference
    environment's duration gaps, so this is the pair that must be unique.
    """
    keys = Counter(
        (
            json.dumps(r["experiment"]["environment"], sort_keys=True),
            json.dumps(r["reference"]["environment_reference"], sort_keys=True),
        )
        for r in records
    )
    repeated = sum(n for n in keys.values() if n > 1)
    return LevelResult(
        level=Level.L1,
        name="contrast_environment_uniqueness",
        passed=repeated == 0,
        message=f"{len(keys)} distinct (environment, reference environment) pairs over "
        f"{len(records)} records",
        details={"n_records": len(records), "n_in_repeated_pairs": repeated},
    )


def verify_protein_fold_change_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_count: int,
    gene_universe: Collection[str],
    adjustment: AdjustmentBackSolve,
) -> VerificationReport:
    """The fold-change family gate (L0-L3) plus this panel's own L1, the assembly pin,
    the Benjamini-Hochberg back-solve and the REL606 containment (L4).
    """
    from torchcell.verification.protein_fold_change import (
        verify_protein_fold_change_dataset,
    )

    report = verify_protein_fold_change_dataset(
        [dict(r) for r in records],
        dataset_name=DATASET_SLUGS["protein_fold_change"],
        provenance=_verifier_provenance("protein_fold_change"),
        expected_count=expected_count,
    )
    report.add(_l1_distinct_contrast_environments(records))
    report.add(_l1_distinct_profiles(records, "protein_fold_change"))
    se = l2_value_fidelity(
        (
            float(v)
            for r in records
            for v in (
                r["experiment"]["phenotype"]["protein_fold_change_se"] or {}
            ).values()
        ),
        allow_nan=False,
        minimum=0.0,
    )
    report.add(se.model_copy(update={"name": "fold_change_se_nonnegative"}))
    adjusted = [
        float(v)
        for r in records
        for v in (
            r["experiment"]["phenotype"]["protein_fold_change_p_value_adjusted"] or {}
        ).values()
    ]
    off_range = [v for v in adjusted if not 0.0 < v <= 1.0]
    report.add(
        LevelResult(
            level=Level.L2,
            name="adjusted_p_values_are_probabilities",
            passed=not off_range,
            message=f"{len(adjusted) - len(off_range)} of {len(adjusted)} adjusted "
            "p-values lie in (0, 1]",
            details={"n_values": len(adjusted), "n_bad": len(off_range)},
        )
    )
    nested = sum(
        1
        for r in records
        for name in (
            "protein_fold_change_se",
            "protein_fold_change_p_value",
            "protein_fold_change_p_value_adjusted",
        )
        if not set(r["experiment"]["phenotype"][name] or {})
        <= set(r["experiment"]["phenotype"]["protein_fold_change"])
    )
    report.add(
        LevelResult(
            level=Level.L2,
            name="statistic_keys_are_nested",
            passed=nested == 0,
            message=f"{nested} statistic maps key a protein the record carries no fold "
            "change for",
            details={"n_bad": nested},
        )
    )
    methods = sorted(
        {r["experiment"]["phenotype"]["p_value_adjustment_method"] for r in records}
    )
    report.add(
        LevelResult(
            level=Level.L3,
            name="p_value_adjustment_method_consistent",
            passed=methods == [P_VALUE_ADJUSTMENT_METHOD],
            message=f"adjustment methods {methods}",
            details={"methods": methods},
        )
    )
    report.add(_l3_assembly_pin(records))
    report.add(_l3_adjustment_back_solve(adjustment))
    measured = {
        p for r in records for p in r["experiment"]["phenotype"]["protein_fold_change"]
    }
    report.add(_l4_containment(measured, gene_universe))
    return report


def run_verification(
    family: Family, data_root: str | None = None
) -> VerificationReport:
    """Verify one built dev-tree LMDB (L0-L4) plus the audit of every ``SourcedValue``,
    and write ``preprocess/verification_report.json``.
    """
    from torchcell.verification.runners import (
        _gene_set_for_reference,
        _write_report,
        load_records,
    )

    base = data_root or _data_root()
    abs_root = osp.join(base, "data/torchcell", DATASET_SLUGS[family])
    preprocess = osp.join(abs_root, "preprocess")
    records = load_records(abs_root)
    references = {
        json.dumps(r["reference"]["genome_reference"], sort_keys=True) for r in records
    }
    universe: set[str] = set()
    for reference in references:
        universe |= _gene_set_for_reference(json.loads(reference), base)
    if family == "protein_fold_change":
        fold_change_accounting = FoldChangeAccounting.model_validate_json(
            Path(preprocess, "build_accounting.json").read_text()
        )
        report = verify_protein_fold_change_records(
            records,
            expected_count=fold_change_accounting.kept_records,
            gene_universe=universe,
            adjustment=AdjustmentBackSolve.model_validate_json(
                Path(preprocess, "adjustment_back_solve.json").read_text()
            ),
        )
    else:
        accounting = BuildAccounting.model_validate_json(
            Path(preprocess, "build_accounting.json").read_text()
        )
        back_solve = VstBackSolve.model_validate_json(
            Path(preprocess, "vst_back_solve.json").read_text()
        )
        verify = (
            verify_rnaseq_records if family == "rnaseq" else verify_proteome_records
        )
        report = verify(
            records,
            expected_count=accounting.kept_records,
            gene_universe=universe,
            back_solve=back_solve,
        )
    library = Path(base) / "torchcell-library"
    for value in sourced_values().values():
        report.add(audit_sourced_value(value, library))
    _write_report(report, preprocess)
    return report


DATASET_CLASSES: dict[Family, type[ExperimentDataset]] = {
    "rnaseq": RnaseqCaglar2017Dataset,
    "proteome": ProteomeCaglar2017Dataset,
    "protein_fold_change": ProteinFoldChangeCaglar2017Dataset,
}


def main(argv: list[str] | None = None) -> int:
    """Deposit the raw mirror, measure identifier coverage, build or verify a family."""
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(
        prog="python -m torchcell.datasets.ecoli.caglar2017"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    deposit = sub.add_parser("deposit", help="retrieve into --staging, then deposit")
    deposit.add_argument("--staging", required=True)
    measure = sub.add_parser("measure", help="identifier coverage on the raw mirror")
    measure.add_argument("--genbank-gbff", required=True)
    measure.add_argument("--refseq-gbff", required=True)
    measure.add_argument("--refseq-gaf", required=True)
    for name, text in (
        ("build", "build (or load) a family's dev-tree LMDB"),
        ("verify", "run L0-L4 on a family's built dev-tree LMDB"),
    ):
        command = sub.add_parser(name, help=text)
        command.add_argument("--family", choices=sorted(DATASET_SLUGS), required=True)
    args = parser.parse_args(argv)
    if args.command == "deposit":
        print(deposit_raw_mirror(source_dir=retrieve_raw_files(args.staging)))
        return 0
    if args.command == "measure":
        coverage = identifier_coverage(
            raw_mirror_dir(), args.genbank_gbff, args.refseq_gbff
        )
        print(coverage.model_dump_json(indent=2))
        summary = annotation_summary(
            args.genbank_gbff, args.refseq_gbff, args.refseq_gaf
        )
        print(summary.model_dump_json(indent=2))
        print(
            strain_pin_finding().model_dump_json(
                indent=2, include={"strain", "pinnable"}
            )
        )
        return 0
    family: Family = args.family
    if args.command == "build":
        root = osp.join(_data_root(), "data/torchcell", DATASET_SLUGS[family])
        dataset = DATASET_CLASSES[family](root=root)
        print(f"{type(dataset).__name__}: len = {len(dataset)}")
        return 0
    report = run_verification(family)
    print(report.summary())
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
