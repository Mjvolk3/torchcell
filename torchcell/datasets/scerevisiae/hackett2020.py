# torchcell/datasets/scerevisiae/hackett2020
# [[torchcell.datasets.scerevisiae.hackett2020]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/hackett2020
# Test file: tests/torchcell/datasets/scerevisiae/test_hackett2020.py
"""Hackett 2020 IDEA: whole-transcriptome time courses after inducing one regulator.

Hackett et al. 2020 (Mol Syst Biol 16:e9174, doi:10.15252/msb.20199174; citation key
``hackettLearningCausalNetworks2020``) placed each of ~200 yeast regulators under a
beta-estradiol-inducible synthetic promoter, grew each strain to steady state in a
chemostat, added 1 uM beta-estradiol and hybridized RNA from the culture against a
common reference RNA on two-colour Agilent 8x15k arrays at (usually) eight time points.

RELEASES. Two, both the authors':

- ``idea_tall_expression_data.zip`` (the Calico IDEA site the paper's Data availability
  names, ``storage.googleapis.com/calico-website-pin-public-bucket/datasets``): one row
  per (time course, time, gene), 10,454,275 rows = 1,693 arrays x 6,175 genes over 217
  time courses, with the two channel medians and five progressively processed log2
  layers plus the noise-model standard deviation. Its column dictionary is the site's
  own page, deposited beside it.
- Dataset EV2 (``MSB-16-e9174-s006.xlsx``): the ten validation time courses, 80 arrays
  x 6,175 genes, with the channel medians, the unprocessed and the cleaned log2 ratio.
  The site table does not carry them; EV2 carries no noise model and no thresholded or
  shrunken layer, so those fields are typed gaps on these records.

Dataset EV1 (``MSB-16-e9174-s005.xlsx``) is the shrunken layer of the site table as a
gene x array matrix; the verifier compares the stored ``log2_shrunken`` with it.

RECORD. One ``InductionTimecourseExpressionExperiment`` per (time course, time point):
one array. Primary layer ``log2_cleaned`` (relative to the same culture at t = 0),
statistic ``log2_noise_sd``; every other released layer is kept. The reference is the
same culture at t = 0 on its typed background (``StrainReferenceGenome``) in its
chemostat (``CultureEnvironment``), so every log2 layer of the reference is 0.

GENOTYPE. Supplementary Table EV1 (``MSB-16-e9174-s002.xls``) states each strain's
genotype. The designed edit is an ``InduciblePromoterInsertionPerturbation`` on the
induced gene (``KanMX-Z3EVpr-ACA1`` and its kin); the rest of the string is the
background: the synthetic activator cassette (GEV, ZEV, 4ZEV, Z3EV) and the alleles that
make the promoter answer it only (``gal4d::LEU2``, ``(Pgal10-1+Gal1)::loxP``). Five
backgrounds, named after the control strain whose Table EV1 genotype is exactly the
shared part where one exists (DBY12021, SMY69, SMY254). The two control time-course
families (GEV, Z3EV) carry no gene perturbation.

ENVIRONMENT. ``CultureEnvironment`` on the IDEA chemostat media (``media.py``: phosphate-,
nitrogen- or methionine-limited by the release's ``restriction`` token), 30 C, aerobic,
the 500 ml vessel at 300 ml stirred at 400 rpm. ``duration_hours`` is the sampling time
after beta-estradiol addition (minutes / 60), the convention the served Brunk 2016 time
course uses for hours post-induction. The t = 0 array was taken BEFORE the addition (GEO
GSE142864 treatment protocol), so its environment carries no estradiol; every later
array carries ``SmallMoleculePerturbation(estradiol, 1 uM)``.

GENE KEYS (2026.10.12, the data is the data). Every expression layer is keyed by the
release's own ``GeneName`` token verbatim (6,175 keys: ``AAC1``, ``YIL082W-A``, ...);
no genome is consulted at build time and no name is rewritten to an R64 systematic name
(until 2026.10.12 the loader re-keyed 5,082 of them through the annotation). The one
release-level correction kept is ``OCT1``, which Excel turned into a date in both SI
workbooks (``EXCEL_DATE_GENE``). Reconciling a release name to the current annotation is
a downstream step at adapter or modeling time.

The induced gene of each strain is named by the release only by a common-name token
(Table EV1's genotype string, the tall table's ``TF`` column); no released file carries
its ORF, and ``InduciblePromoterInsertionPerturbation.systematic_gene_name`` must be an
ORF. ``RELEASE_TF_TOKENS_WITHOUT_AN_ORF`` is the one remaining crosswalk: the token -> ORF
pairs the KG build served (derived from that build's ledger, not from a genome call at
build time), pending group 1's released-identifier leaf, after which the token is stored
verbatim and the ORF becomes a typed gap.

REFUSED. Three time courses whose strain (SMY10, SMY19, SMY22) has no Table EV1 row: 21
arrays. Their genotypes are stated only in McIsaac 2011/2012, which are not mirrored, and
the record's reference genome needs the strain's background, which has no typed gap, so
the refusal stays until a carrier exists.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import os.path as osp
import pickle
import re
import shutil
import zipfile
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import lmdb
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator

from torchcell.data import (
    ExperimentDataset,
    Organism,
    check_manifest_pin,
    compute_sha256_hash,
    link_verified,
    post_process,
    verify_raw_files,
)
from torchcell.data.experiment_dataset import canonical_json
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import (
    CHEMOSTAT_MET_LIMITED_HACKETT2020,
    CHEMOSTAT_N_LIMITED_HACKETT2020,
    CHEMOSTAT_P_LIMITED_HACKETT2020,
)
from torchcell.datamodels.schema import (
    SGD_S288C_NAMESPACE,
    AlleleEdit,
    BackgroundAllele,
    Concentration,
    ConcentrationUnit,
    CultureEnvironment,
    CultureFormat,
    EndpointRule,
    EukaryoticStrainBackground,
    Experiment,
    ExperimentReference,
    Genotype,
    InduciblePromoterInsertionPerturbation,
    InductionTimecourseExpressionExperiment,
    InductionTimecourseExpressionExperimentReference,
    InductionTimecourseExpressionPhenotype,
    IntegratedCassette,
    Media,
    PreCulture,
    PreCultureSource,
    Publication,
    SmallMoleculePerturbation,
    StrainReferenceGenome,
    Temperature,
    Zygosity,
)
from torchcell.datamodels.strain_background import BRACHMANN_1998, standard_allele
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.literature.manifest import (
    ROLE_RAW_DATA,
    ROLE_SI_DATA,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
    sha256_file,
)
from torchcell.sequence import GeneSet
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
CITATION_KEY = "hackettLearningCausalNetworks2020"
PAPER_DOI = "10.15252/msb.20199174"
PAPER_PMID = "32181581"
PAPER_TITLE = (
    "Learning causal networks using inducible transcription factors and "
    "transcriptome-wide time series"
)
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"

#: The IDEA site's public bucket, as its page writes it, and the release objects in it.
IDEA_BUCKET = "https://storage.googleapis.com/calico-website-pin-public-bucket/datasets"
IDEA_SITE = "https://idea.research.calicolabs.com"
TALL_ZIP_NAME = "idea_tall_expression_data.zip"
TALL_MEMBER = "idea_tall_expression_data.tsv"
TALL_ZIP_SHA256 = "2fec89f98c602049f2964ca80821d540b50e579fe799d8694fd742731830accc"
#: The bucket's own integrity record for the object (base64 md5 and generation), read
#: from the GCS JSON listing at retrieval and checked against the retrieved bytes.
TALL_ZIP_GCS_MD5_BASE64 = "XOJNBALqApEP249WbeOMug=="
TALL_ZIP_GCS_GENERATION = "1577397324560244"
#: The site page that defines every column of the tall table (its data dictionary).
SITE_JS_NAME = "idea_website_main.915bfc49.chunk.js"
SITE_JS_URL = f"{IDEA_SITE}/static/js/main.915bfc49.chunk.js"
SITE_JS_SHA256 = "41c8070fd8286fea284ca435ba554b8e8d51ccd732f02d05a9b8910b8a8fd8aa"
#: GEO GSE142864 series matrices (sample metadata and protocols, both platforms).
GEO_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE142nnn/GSE142864/matrix/"
GEO_GPL16244_NAME = "GSE142864-GPL16244_series_matrix.txt.gz"
GEO_GPL20785_NAME = "GSE142864-GPL20785_series_matrix.txt.gz"
GEO_GPL16244_SHA256 = "871d1fddba1a32d2152d1dd6117d57b1b00876221a075f72ac54d8a50489cb08"
GEO_GPL20785_SHA256 = "eed815375636378282ff082f5ecc93309a1f9499393d66f920bdeaacb7ff4f57"
#: The two SI tables the loader reads, as the PMC Article Datasets bucket names them.
STRAINS_NAME = "MSB-16-e9174-s002.xls"
VALIDATION_NAME = "MSB-16-e9174-s006.xlsx"
STRAINS_SHA256 = "22b665e6c40cc36d0f6fdcf9ee51bc18548b72e450745aebba6de8abdfe9e2dc"
VALIDATION_SHA256 = "2dd041ba81ed89b000f4198112317f3600aa5035f36892a9b63fb9c6f95751dd"
#: Library-mirror paths of the same two SI files (the verifier reads EV1 there too).
STRAINS_LIBRARY = "si/si2.xls"
VALIDATION_LIBRARY = "si/si6.xlsx"
SHRUNKEN_EV1_LIBRARY = "si/si5.xlsx"
SHRUNKEN_EV1_SHA256 = "3e24a7dd82dda3c0d2e71312619584225dc6af1d622dacfa9451d0f082db3ec6"
PMC_BUCKET = "https://pmc-oa-opendata.s3.amazonaws.com/PMC7076914.1/"

TALL_REL = f"data/{TALL_ZIP_NAME}"
SITE_JS_REL = f"data/{SITE_JS_NAME}"
GEO_GPL16244_REL = f"data/{GEO_GPL16244_NAME}"
GEO_GPL20785_REL = f"data/{GEO_GPL20785_NAME}"
STRAINS_REL = f"si/{STRAINS_NAME}"
VALIDATION_REL = f"si/{VALIDATION_NAME}"

#: Raw files linked into the dataset's ``raw/`` and their build-time pins.
RAW_FILES: dict[str, str] = {
    TALL_ZIP_NAME: TALL_REL,
    STRAINS_NAME: STRAINS_REL,
    VALIDATION_NAME: VALIDATION_REL,
}
RAW_SHA256: dict[str, str] = {
    TALL_ZIP_NAME: TALL_ZIP_SHA256,
    STRAINS_NAME: STRAINS_SHA256,
    VALIDATION_NAME: VALIDATION_SHA256,
}

PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "78ec9f04356e813f0a2042f8c84435a37364523cb633437c0e07d7eadf4828a8"
APPENDIX_MD = "si/si1.md"
APPENDIX_MD_SHA256 = "46dadc5e3f44ed81a40cfe9ccb0c121180c8c17e5e844b3c15eaa4c3c0ed7745"

# --------------------------------------------------------------------------- #
# Release shape (measured on the deposited bytes, see the dendron note)
# --------------------------------------------------------------------------- #
N_GENES = 6175
N_TALL_TIMECOURSES = 217
N_TALL_ARRAYS = 1693
N_VALIDATION_TIMECOURSES = 10
N_VALIDATION_ARRAYS = 80
TALL_COLUMNS = [
    "TF",
    "strain",
    "date",
    "restriction",
    "mechanism",
    "time",
    "GeneName",
    "green_median",
    "red_median",
    "log2_ratio",
    "log2_cleaned_ratio",
    "log2_noise_model",
    "log2_cleaned_ratio_zth2d",
    "log2_selected_timecourses",
    "log2_shrunken_timecourses",
]
VALIDATION_COLUMNS = [
    "TF",
    "restriction",
    "mechanism",
    "time",
    "GeneName",
    "green_median",
    "red_median",
    "log2_ratio",
    "log2_cleaned_ratio",
]
VALIDATION_SHEET = "Dataset EV2"
STRAINS_SHEET = "TableS1.txt"
#: The one gene Excel turned into a date in both SI workbooks (EV1, EV2):
#: 2019-10-01, the OCT1 cell. The tall TSV carries the name as text.
EXCEL_DATE_GENE = "OCT1"
EXCEL_DATE_VALUE = datetime(2019, 10, 1)
#: Time courses the release carries whose strain Table EV1 does not list.
UNLISTED_STRAINS = ("SMY10", "SMY19", "SMY22")
REFUSAL_UNLISTED_STRAIN = "strain_not_in_table_ev1"

INDUCER_UM = 1.0
TEMPERATURE_C = 30.0
VESSEL = "500 ml chemostat vessel (Infors Sixfors or Multifors)"
WORKING_VOLUME_UL = 300_000.0
STIR_RPM = 400.0


def _sv(
    value: object,
    quote: str,
    *,
    uri: str = PAPER_MD,
    sha: str = PAPER_MD_SHA256,
    page: str | None = None,
    note: str | None = None,
) -> SourcedValue:
    """A SourcedValue pinned to a mirrored Hackett 2020 artifact (quote + sha256)."""
    return SourcedValue(
        value=value,
        provenance=Provenance(
            source_uri=uri, citation_key=CITATION_KEY, sha256=sha, page=page
        ),
        quote=quote,
        note=note,
    )


def _appendix_sv(value: object, quote: str, note: str | None = None) -> SourcedValue:
    """A SourcedValue on the Appendix OCR (``si/si1.md``)."""
    return _sv(value, quote, uri=APPENDIX_MD, sha=APPENDIX_MD_SHA256, note=note)


def _geo_sv(value: object, quote: str, note: str | None = None) -> SourcedValue:
    """A SourcedValue on the deposited GEO GSE142864 GPL20785 series matrix (gzip)."""
    return _sv(
        value,
        quote,
        uri=f"{RAW_DIR_REL}/{GEO_GPL20785_REL}",
        sha=GEO_GPL20785_SHA256,
        note=note,
    )


def _site_sv(value: object, quote: str, note: str | None = None) -> SourcedValue:
    """A SourcedValue on the deposited IDEA site page (the tall table's dictionary)."""
    return _sv(
        value, quote, uri=f"{RAW_DIR_REL}/{SITE_JS_REL}", sha=SITE_JS_SHA256, note=note
    )


# --- paper.md (Methods unless noted) --------------------------------------- #
Q_TIMES = (
    "The majority of experiments were sampled at $t = 0$ , 5, 10, 15, 20, 30, 45, and "
    "$9 0 ~ \\mathrm { { m i n } }$ following $\\beta$ -estradiol addition in "
    "phosphate-limited chemostats."
)
Q_ESTRADIOL = (
    "Cells were grown to steady state, as determined by culture density, prior to the "
    "addition of $1 ~ \\mu \\mathrm { M }$ $\\beta$ -estradiol to the culture and "
    "subsequent sampling."
)
Q_AERATION = (
    "Cultures were aerated with $6 1 /$ min of humidified air at $3 0 ^ { \\circ } "
    "\\mathrm { C }$ , maintained at $3 0 0 ~ \\mathrm { m l }$ , and stirred with a "
    "magnetic impeller at 400 RPM."
)
Q_VESSEL = (
    "For all experiments, cells were grown under continuous culturing conditions in "
    "$5 0 0 \\mathrm { - m l }$ vessels as previously described with minor adjustments "
    "(Saldanha et al, 2004)."
)
Q_GROWTH_RATE = "Growth rates were maintained from 0.15 to $0 . 1 7 / \\mathrm { h }$ ."
Q_PRECULTURE = (
    "Batch growth in the chemostat vessels was initiated from a 1:60 dilution of a "
    "saturated overnight culture prior to turning on the chemostat pumps."
)
Q_SYSTEMS = (
    "Chemostat experiments were performed with either the Infors Sixfors or the "
    "Multifors systems."
)
Q_INSERTION = (
    "Primers were designed using custom software in R such that the cassette was "
    "introduced directly between a target gene’s first methionine residue and its "
    "native promoter to prevent the removal of any genomic DNA."
)
Q_PARENT = (
    "Parent strains were engineered to constitutively express an artificial "
    "transcription factor that is inducible with estradiol."
)
Q_REFERENCE_RNA = (
    "Reference RNA was extracted from DBY12001, a laboratory wild-type strain, grown to "
    "steady state in a phosphatelimited chemostat at $\\mathrm { D } \\sim 0 . 1 8 / "
    "\\mathrm { h }$ ."
)
Q_LABELS = (
    "RNA from references and samples was labeled with Cy3-CTP and Cy5-CTP, "
    "respectively."
)
Q_ARRAY = (
    "Labeled RNA was hybridized to Agilent $8 \\times 1 5 \\mathrm { k }$ microarrays, "
    "which were then washed, scanned, and processed using the Agilent Feature "
    "Extraction software with default settings and loess dye bias correction."
)
Q_SPOTS = "Each microarray has some number of spots (usually 2) for each gene."
Q_CLEANED = (
    "We refer to the dataset where outliers are removed and the GEV/ZEV signal is "
    "removed as the “cleaned” dataset."
)
Q_EV1 = "The full transcriptional dataset is available as Dataset EV1."
Q_RETAINED = (
    "The signals from these 100,036 timecourses were retained $( \\sim 8 \\%$ of "
    "timecourses), while all other timecourses were set as invariant (a $\\log _ { 2 }$ "
    "fold change of zero)."
)
Q_COUNT = (
    "In total, IDEA contains $> 1 { , } 6 5 0$ microarrays comprising 217 distinct "
    "induction experiments."
)
Q_FOLD53 = (
    "Induction of a target TF is detectable in $< 5$ min and reaches saturation within "
    "$\\sim 1 0 ~ \\mathrm { { m i n } }$ following $\\beta$ -estradiol addition at a "
    "median level 53-fold higher than at $t = 0 ~ \\mathrm { m i n }$ (Appendix Fig S1B)"
)
Q_VALIDATION = (
    "To validate putative regulators, a separate $\\beta$ -estradiol induction "
    "experiment was generated for each of ten candidate regulators (Dataset EV2)."
)
Q_GEO = (
    "Microarray data, Gene Expression: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?"
    "acc $=$ GSE142864; http://idea.research.calicolabs.com."
)
Q_REPEATS = (
    "Fifteen induction experiments were repeated at least once using the same induced "
    "gene to either capture late changes in some experiments, confirm similar expression "
    "response to two different induction systems [referred to as ZEV (McIsaac et al, "
    "2013, 2014) and GEV (McIsaac et al, 2011)], or to verify reproducibility (Appendix "
    "Fig S1A)."
)
# --- Appendix (si/si1.md) ---------------------------------------------------- #
Q_NOISE = (
    "Under the assumption that any gene is not affected in most experiments, the width "
    "of the central quintile (20%) of “cleaned” log-ratio measurements is an accurate "
    "estimate of the total noise level for that gene."
)
Q_FULL_CLEANED = (
    "Collectively, the full cleaned dataset is 1,693 microarrays in 217 experiments, "
    "each with 6,175 genes."
)
Q_118134 = (
    "Of the 1.34 million time series, 118,134 (8.8%) have at least one point that "
    "passes the full noise model."
)
Q_VALIDATION_METHOD = (
    "Each predicted regulator was separately induced and genome-wide expression was "
    "tracked at eight time points using the same experimental and computational "
    "methodology used to construct the “cleaned” dataset."
)
Q_SHRUNKEN = "This is the “shrunken” dataset."
# --- GEO GSE142864 series matrix (sample protocols) -------------------------- #
Q_GEO_T0 = (
    "The t = 0 was taken (between 3 and 5 mL of culture) and vacuum filtered. "
    "Estradiol was then added the continuous culture and sampled at various timepoints."
)
Q_GEO_BACKGROUND = "background: s288c"
Q_GEO_METHIONINE = (
    "Methionine-limited cultures (McIsaac et al, 2012) were maintained at 7.5 mg/L "
    "methionine."
)
# --- IDEA site page (the tall table's column dictionary) --------------------- #
Q_SITE_TALL = (
    "For each of 10.4M gene expression observations, raw and progressively processed "
    "forms of the data are provided."
)
Q_SITE_GREEN = "median of green (reference) channel fluorescence"
Q_SITE_RED = "median of red (experimental) channel fluorescence"
Q_SITE_LOG2 = '"log2_ratio")," - log2(red / green) subtracting value at time zero"'
Q_SITE_CLEANED = "Non-specific stress response and prominent outliers removed"
Q_SITE_NOISE = "estimated noise standard deviation"
Q_SITE_ZTH2D = (
    "cleaned timecourses hard-thresholded based on multiple observations (or last "
    "observation) passing the noise model"
)
Q_SITE_SELECTED = (
    "cleaned timecourses hard-thresholded based on single observations passing noise "
    "model and impulse evaluation of biological feasibility"
)
Q_SITE_SHRUNKEN = (
    "selected timecourses with observation-level shrinkage based on local FDR (false "
    "discovery rate). "
)
Q_SITE_MOST_USERS = "Most users of the data will want to use this column."

#: Every quote this module (and its media) cites, for the verifier's provenance audit.
SOURCED_VALUES: dict[str, SourcedValue] = {
    "sampling_times": _sv("0, 5, 10, 15, 20, 30, 45, 90 min (most courses)", Q_TIMES),
    "inducer": _sv("beta-estradiol 1 uM at steady state", Q_ESTRADIOL),
    "temperature": _sv(TEMPERATURE_C, Q_AERATION),
    "culture_format": _sv("500 ml vessel, 300 ml, 400 rpm", Q_AERATION),
    "vessel": _sv("500 ml chemostat vessel", Q_VESSEL),
    "dilution_rate_range": _sv("0.15 to 0.17 per h", Q_GROWTH_RATE),
    "pre_culture": _sv("saturated overnight culture, 1:60", Q_PRECULTURE),
    "chemostat_systems": _sv("Infors Sixfors or Multifors", Q_SYSTEMS),
    "insertion": _sv("cassette between ATG and native promoter", Q_INSERTION),
    "parent_activator": _sv("background expresses the synthetic activator", Q_PARENT),
    "reference_rna": _sv("DBY12001 reference RNA", Q_REFERENCE_RNA),
    "channel_labels": _sv("reference Cy3, sample Cy5", Q_LABELS),
    "platform": _sv("Agilent 8x15k two-colour", Q_ARRAY),
    "spots": _sv("usually 2 spots per gene, aggregated", Q_SPOTS),
    "cleaned": _sv("cleaned = outliers and GEV/ZEV signal removed", Q_CLEANED),
    "dataset_ev1": _sv("Dataset EV1 = the released transcriptional dataset", Q_EV1),
    "retained_timecourses": _sv(100036, Q_RETAINED),
    "n_experiments": _sv(217, Q_COUNT),
    "induction_fold": _sv(53, Q_FOLD53),
    "validation": _sv(10, Q_VALIDATION),
    "geo_accession": _sv("GSE142864", Q_GEO),
    "repeats": _sv("repeated time courses", Q_REPEATS),
    "noise_model": _appendix_sv("noise SD from the central quintile", Q_NOISE),
    "full_cleaned": _appendix_sv(1693, Q_FULL_CLEANED),
    "thresholded_timecourses": _appendix_sv(118134, Q_118134),
    "validation_method": _appendix_sv(
        "validation built as the cleaned dataset", Q_VALIDATION_METHOD
    ),
    "shrunken": _appendix_sv("shrunken dataset", Q_SHRUNKEN),
    "t0_before_inducer": _geo_sv("t = 0 sampled before estradiol", Q_GEO_T0),
    "background_s288c": _geo_sv("S288C", Q_GEO_BACKGROUND),
    "methionine_limitation": _geo_sv("7.5 mg/L methionine", Q_GEO_METHIONINE),
    "site_tall": _site_sv("tall table, raw and processed layers", Q_SITE_TALL),
    "site_green": _site_sv("green_median = reference channel", Q_SITE_GREEN),
    "site_red": _site_sv("red_median = sample channel", Q_SITE_RED),
    "site_log2": _site_sv("log2_ratio = log2(red/green) minus t0", Q_SITE_LOG2),
    "site_cleaned": _site_sv("log2_cleaned_ratio", Q_SITE_CLEANED),
    "site_noise": _site_sv("log2_noise_model = noise SD", Q_SITE_NOISE),
    "site_zth2d": _site_sv("log2_cleaned_ratio_zth2d", Q_SITE_ZTH2D),
    "site_selected": _site_sv("log2_selected_timecourses", Q_SITE_SELECTED),
    "site_shrunken": _site_sv("log2_shrunken_timecourses", Q_SITE_SHRUNKEN),
    "site_most_users": _site_sv("shrunken is the authors' default", Q_SITE_MOST_USERS),
}

# --------------------------------------------------------------------------- #
# Media by the release's restriction token
# --------------------------------------------------------------------------- #
#: ``restriction`` token -> medium. P and N are the Methods' two limitations; M is the
#: methionine limitation of the met6-delta strains, stated by the GEO growth protocol.
MEDIA_BY_RESTRICTION: dict[str, Media] = {
    "P": CHEMOSTAT_P_LIMITED_HACKETT2020,
    "N": CHEMOSTAT_N_LIMITED_HACKETT2020,
    "M": CHEMOSTAT_MET_LIMITED_HACKETT2020,
}


# --------------------------------------------------------------------------- #
# Strain table (Supplementary Table EV1)
# --------------------------------------------------------------------------- #
class StrainRow(BaseModel):
    """One Table EV1 row, every cell verbatim (``sheet_row`` is 1-based)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    strain: str
    genotype: str
    strain_type: str
    source: str
    sheet_row: int


class InsertionSpec(BaseModel):
    """The inducible promoter insertion a Table EV1 genotype string names."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    gene_token: str
    promoter_name: str
    marker: str
    cassette: str
    responsive_to: str


BackgroundKey = Literal["DBY12021", "SMY69", "SMY254", "SMY257_4ZEV", "Z3EV_HAP1"]


class StrainSpec(BaseModel):
    """A Table EV1 row parsed into its background and its designed edit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    row: StrainRow
    background: BackgroundKey
    insertion: InsertionSpec | None


_D = "[∆dΔ]"
#: Every Table EV1 genotype string matches exactly one of these (fullmatch), or the
#: parse refuses it. ``gene`` is the induced gene's token as the table writes it.
_GENOTYPE_TEMPLATES: list[tuple[re.Pattern[str], BackgroundKey, str, str, str]] = [
    (
        re.compile(
            r"Same as DBY12021 with (?P<cassette>KanMX-Pgal1-(?P<gene>[A-Z0-9]+))"
        ),
        "DBY12021",
        "Pgal1",
        "KanMX",
        "GEV",
    ),
    (
        re.compile(
            r"\(Pgal10-1\+Gal1\)::loxP gal4d::LEU2 leu2d0::pACT1-GEV-NatMX "
            r"met6d::HphMX4 (?P<cassette>KanMX-Pgal1-?(?P<gene>[A-Z0-9]+))"
        ),
        "SMY69",
        "Pgal1",
        "KanMX",
        "GEV",
    ),
    (
        re.compile(
            r"(?P<cassette>KanMX-Z3EVpr-(?P<gene>[A-Za-z0-9]+)), "
            r"leu2d0::ACT1pr-ZEV-NatMX,? ybr032wd::LEU2pr\(-314 --> -1\)-LEU2"
        ),
        "SMY254",
        "Z3EVpr",
        "KanMX",
        "ZEV",
    ),
    (
        re.compile(
            r"leu2d0::ACT1pr-4ZEV-NatMX ybr032wd::LEU2pr\(-314 --> -1\)-LEU2 "
            r"(?P<cassette>KanMX-4ZEVpr-(?P<gene>[A-Z0-9]+))"
        ),
        "SMY257_4ZEV",
        "4ZEVpr",
        "KanMX",
        "4ZEV",
    ),
    (
        re.compile(
            rf"HAP1\+::pACT1-Z3EV-NatMX::eno2term ura3?{_D}0 can1{_D}(?:::)?Ste2pr-Sphis5 "
            rf"his3{_D}1 lyp1?{_D}0 (?P<cassette>URA3::Z3EVpr-(?P<gene>[A-Za-z0-9]+))"
        ),
        "Z3EV_HAP1",
        "Z3EVpr",
        "URA3",
        "Z3EV",
    ),
]
#: The three control rows: a background with no insertion.
CONTROL_GENOTYPES: dict[str, BackgroundKey] = {
    "(Pgal10-1+Gal1)::loxP leu2d0::ACT1pr-GEV-NatMX gal4d::LEU2": "DBY12021",
    "(Pgal10-1+Gal1)::loxP gal4d::LEU2 leu2d0::pACT1-GEV-NatMX met6d::HphMX4": "SMY69",
    "leu2d0::ACT1pr-ZEV-NatMX ybr032wd::LEU2pr(-314 --> -1)-LEU2": "SMY254",
}


class StrainTableError(ValueError):
    """A Table EV1 row is not one of the genotype shapes this loader states."""


def read_strain_table(path: str | Path) -> dict[str, StrainRow]:
    """Every Table EV1 row by strain, verbatim; refuses a duplicated strain."""
    sheet = pd.read_excel(path, sheet_name=STRAINS_SHEET, header=None, dtype=str)
    if sheet.iloc[0, 0] != "Table EV1: Strains":
        raise StrainTableError(f"unexpected title cell {sheet.iloc[0, 0]!r}")
    header = sheet.iloc[1].tolist()
    if header != ["Strain", "Genotype", "Type", "Experiment/Microarray Source"]:
        raise StrainTableError(f"unexpected header {header!r}")
    rows: dict[str, StrainRow] = {}
    for index in range(2, len(sheet)):
        strain, genotype, strain_type, source = sheet.iloc[index].tolist()
        if strain in rows:
            raise StrainTableError(f"strain {strain!r} appears twice")
        rows[strain] = StrainRow(
            strain=strain,
            genotype=genotype,
            strain_type=strain_type,
            source=source,
            sheet_row=index + 1,
        )
    return rows


def parse_strain(row: StrainRow) -> StrainSpec:
    """A row's background and insertion, by the one template its genotype matches."""
    if row.genotype in CONTROL_GENOTYPES:
        if row.strain_type != "Control":
            raise StrainTableError(
                f"{row.strain}: control genotype on a non-control row"
            )
        return StrainSpec(
            row=row, background=CONTROL_GENOTYPES[row.genotype], insertion=None
        )
    matches = [
        (pattern.fullmatch(row.genotype), key, promoter, marker, activator)
        for pattern, key, promoter, marker, activator in _GENOTYPE_TEMPLATES
    ]
    hits = [m for m in matches if m[0] is not None]
    if len(hits) != 1:
        raise StrainTableError(
            f"{row.strain}: genotype {row.genotype!r} matches {len(hits)} templates"
        )
    match, key, promoter, marker, activator = hits[0]
    assert match is not None
    return StrainSpec(
        row=row,
        background=key,
        insertion=InsertionSpec(
            gene_token=match.group("gene"),
            promoter_name=promoter,
            marker=marker,
            cassette=match.group("cassette"),
            responsive_to=activator,
        ),
    )


def table_strain_of(release_strain: str) -> str:
    """The Table EV1 strain a release strain token names.

    ``yRSM<n>`` and ``SMY<n>`` are the same number with the same induced gene (checked
    by the caller against the row's insertion); a trailing ``n`` marks the nitrogen-
    limited time course of a strain (``SMY108n``), a trailing ``a``/``b``/``c`` one of
    several time courses of a strain on one date (``SMY2266a``); ``YukoSMY2047`` is
    SMY2047. Anything else is refused.
    """
    match = re.fullmatch(r"(?:yRSM|YukoSMY|SMY)(\d+)[abcn]?", release_strain)
    if match is None:
        raise StrainTableError(f"unrecognized release strain token {release_strain!r}")
    return f"SMY{match.group(1)}"


def _row_quote(row: StrainRow) -> str:
    """The row rendering a Table EV1 quote uses: the four cells, tab-joined."""
    return "\t".join([row.strain, row.genotype, row.strain_type, row.source])


def strain_row_sv(
    row: StrainRow, value: object, note: str | None = None
) -> SourcedValue:
    """A SourcedValue on one Table EV1 row (the verifier re-reads the cells)."""
    return _sv(
        value,
        _row_quote(row),
        uri=STRAINS_LIBRARY,
        sha=STRAINS_SHA256,
        page=f"sheet '{STRAINS_SHEET}' row {row.sheet_row}",
        note=note,
    )


# --------------------------------------------------------------------------- #
# Backgrounds
# --------------------------------------------------------------------------- #
#: What was read before concluding that no construct sequence is available for the
#: synthetic activator cassettes. Table EV1 gives each one as a name string and the
#: Methods name the papers the strains come from; the primary deposits no map, so the
#: gap is ``not_released_by_primary`` and closing it takes those papers' deposits.
_CASSETTE_PLACEMENT_NOTE = (
    "Table EV1 gives the cassette as a name string (e.g. "
    "'HAP1+::pACT1-Z3EV-NatMX::eno2term') and states no contig, accession or "
    "coordinate; the Methods cite the McIsaac strain papers for the constructions, and "
    "no construct map is deposited with this paper"
)


def _cassette_placement_gap() -> ProvenanceGap:
    """The typed absence of a construct sequence for an activator cassette."""
    return ProvenanceGap(
        field="placement",
        reason=ProvenanceGapReason.not_released_by_primary,
        looked_in=Provenance(
            source_uri=STRAINS_LIBRARY,
            citation_key=CITATION_KEY,
            sha256=STRAINS_SHA256,
            page=f"sheet {STRAINS_SHEET!r}",
        ),
        note=_CASSETTE_PLACEMENT_NOTE,
    )


MCISAAC_2011 = Provenance(
    source_uri="not mirrored; McIsaac et al. 2011, Mol Biol Cell (Table EV1 source "
    "'McIsaac et al (2011, MBoC)'), the DBY12021 construction",
    method="deferral target for the GEV background's mating type",
)
MCISAAC_2012 = Provenance(
    source_uri="not mirrored; McIsaac et al. 2012, Mol Biol Cell (Table EV1 source "
    "'McIsaac et al (2012, MBoC)'), the met6-delta GEV strains",
    method="deferral target for the met6-delta GEV background's mating type",
)
MCISAAC_2013 = Provenance(
    source_uri="not mirrored; McIsaac et al. 2013, Nucleic Acids Res (Table EV1 "
    "source 'McIsaac et al (2013, NAR)'), the ZEV and 4ZEV strains",
    method="deferral target for the ZEV backgrounds' mating type",
)
_HAPLOID_NOTE = (
    "ploidy is not stated anywhere in the mirror; every Table EV1 genotype is written "
    "with one allele per locus and no '/', the haploid convention of yeast genetic "
    "nomenclature, so the background is recorded haploid (owner decision taken by "
    "recommendation)"
)
_BY_ALLELE_NOTE = (
    "Table EV1 writes the allele ('leu2d0', 'ura3∆0', 'his3∆1'); which edit it is (the "
    "Delta0 designer deletion, the his3-Delta1 internal deletion) is the Brachmann 1998 "
    "reading, not stated in a mirrored source"
)


def _mating_gap(resolve_with: Provenance | None, who: str) -> ProvenanceGap:
    if resolve_with is None:
        return ProvenanceGap(
            field="mating_type",
            reason=ProvenanceGapReason.not_reported_by_primary,
            note=f"{who}: Table EV1 and the Methods state no mating type",
        )
    return ProvenanceGap(
        field="mating_type",
        reason=ProvenanceGapReason.deferred_pending_source_review,
        resolve_with=resolve_with,
        note=f"{who}: Table EV1 states no mating type; its source paper is not mirrored",
    )


def _replacement(
    row: StrainRow,
    *,
    systematic: str,
    gene: str,
    allele_name: str,
    cassette: str,
    note: str | None = None,
) -> BackgroundAllele:
    """A ``locus::cassette`` allele: the '::' says the cassette replaced the ORF."""
    return BackgroundAllele(
        systematic_gene_name=systematic,
        gene_namespace=SGD_S288C_NAMESPACE,
        gene_name=gene,
        allele_name=allele_name,
        edit=AlleleEdit.cassette_replacement,
        functional=False,
        zygosity=Zygosity.haploid,
        cassette=cassette,
        provenance=[strain_row_sv(row, allele_name, note=note)],
    )


def _activator_cassette(
    row: StrainRow, *, name: str, locus: str, systematic: str, elements: list[str]
) -> IntegratedCassette:
    """The synthetic activator cassette a background carries at a named locus."""
    return IntegratedCassette(
        name=name,
        locus=locus,
        locus_systematic_gene_name=systematic,
        elements=elements,
        marker="NatMX",
        zygosity=Zygosity.haploid,
        placement=None,
        provenance=[strain_row_sv(row, name), SOURCED_VALUES["parent_activator"]],
        provenance_gaps=[_cassette_placement_gap()],
    )


def _leu2_delta0() -> BackgroundAllele:
    return standard_allele(
        "leu2Δ0", Zygosity.haploid, resolve_with=BRACHMANN_1998, note=_BY_ALLELE_NOTE
    )


_GAL1_NOTE = (
    "the designation names the GAL10 promoter and GAL1 ('Pgal10-1+Gal1') replaced by a "
    "loxP site; the exact span is not stated, so no deleted_span is recorded and the "
    "allele is placed at GAL1"
)


def build_background(key: BackgroundKey, row: StrainRow) -> EukaryoticStrainBackground:
    """One of the five backgrounds, sourced to the Table EV1 row that states it.

    ``row`` is the control row when the background has one (DBY12021, SMY69, SMY254)
    and otherwise the first row the background is read from (SMY257; SMY2047 for the
    HAP1+ Z3EV background).
    """
    if key in ("DBY12021", "SMY69"):
        gev = (
            "leu2d0::ACT1pr-GEV-NatMX"
            if key == "DBY12021"
            else "leu2d0::pACT1-GEV-NatMX"
        )
        alleles = [
            _replacement(
                row,
                systematic="YBR020W",
                gene="GAL1",
                allele_name="(Pgal10-1+Gal1)::loxP",
                cassette="loxP",
                note=_GAL1_NOTE,
            ),
            _replacement(
                row,
                systematic="YPL248C",
                gene="GAL4",
                allele_name="gal4d::LEU2",
                cassette="LEU2",
            ),
            _leu2_delta0(),
        ]
        if key == "SMY69":
            alleles.append(
                _replacement(
                    row,
                    systematic="YER091C",
                    gene="MET6",
                    allele_name="met6d::HphMX4",
                    cassette="HphMX4",
                )
            )
        integrations = [
            _activator_cassette(
                row,
                name=gev,
                locus="leu2",
                systematic="YCL018W",
                elements=gev.split("::")[1].split("-"),
            )
        ]
        resolve = MCISAAC_2011 if key == "DBY12021" else MCISAAC_2012
        name: str = key
    elif key in ("SMY254", "SMY257_4ZEV"):
        zev = (
            "leu2d0::ACT1pr-ZEV-NatMX"
            if key == "SMY254"
            else "leu2d0::ACT1pr-4ZEV-NatMX"
        )
        alleles = [
            _replacement(
                row,
                systematic="YBR032W",
                gene="YBR032W",
                allele_name="ybr032wd::LEU2pr(-314 --> -1)-LEU2",
                cassette="LEU2pr(-314 --> -1)-LEU2",
            ),
            _leu2_delta0(),
        ]
        integrations = [
            _activator_cassette(
                row,
                name=zev,
                locus="leu2",
                systematic="YCL018W",
                elements=zev.split("::")[1].split("-"),
            )
        ]
        resolve = MCISAAC_2013
        name = "SMY254" if key == "SMY254" else "SMY257 background (ACT1pr-4ZEV)"
    else:
        alleles = [
            standard_allele(
                allele,
                Zygosity.haploid,
                resolve_with=BRACHMANN_1998,
                note=_BY_ALLELE_NOTE,
            )
            for allele in ("ura3Δ0", "his3Δ1", "can1Δ::STE2pr-Sp_his5")
        ]
        alleles.append(
            BackgroundAllele(
                systematic_gene_name="YNL268W",
                gene_namespace=SGD_S288C_NAMESPACE,
                gene_name="LYP1",
                allele_name="lyp1Δ0",
                edit=AlleleEdit.full_deletion,
                functional=False,
                zygosity=Zygosity.haploid,
                provenance=None,
                provenance_gaps=[
                    ProvenanceGap(
                        field="provenance",
                        reason=ProvenanceGapReason.deferred_pending_source_review,
                        resolve_with=BRACHMANN_1998,
                        note="Table EV1 writes 'lyp∆0', 'lyp1∆0' or 'lypd0'; reading "
                        "it as the marker-free Delta0 full deletion of LYP1 is the "
                        "designer-deletion convention, not stated in a mirrored source",
                    )
                ],
            )
        )
        integrations = [
            IntegratedCassette(
                name="HAP1+::pACT1-Z3EV-NatMX::eno2term",
                locus="HAP1",
                locus_systematic_gene_name="YLR256W",
                elements=["pACT1", "Z3EV", "NatMX", "eno2term"],
                marker="NatMX",
                zygosity=Zygosity.haploid,
                placement=None,
                provenance_gaps=[_cassette_placement_gap()],
                provenance=[
                    strain_row_sv(
                        row,
                        "HAP1+::pACT1-Z3EV-NatMX::eno2term",
                        note="the '+' is kept verbatim in the name; whether it marks a "
                        "repaired HAP1 allele or an undisrupted HAP1 at the "
                        "integration site is not stated, so no HAP1 allele is recorded",
                    ),
                    SOURCED_VALUES["parent_activator"],
                ],
            )
        ]
        resolve = None
        name = "HAP1+ Z3EV background (SMY2047, SMY2605-SMY2614)"
    return EukaryoticStrainBackground(
        reference_strain="S288C",
        assembly_set=SGD_S288C_R64,
        name=name,
        parents=None,
        construction=None,
        mating_type=None,
        ploidy="haploid",
        alleles=alleles,
        integrations=integrations,
        provenance=[
            strain_row_sv(row, row.genotype, note=_HAPLOID_NOTE),
            SOURCED_VALUES["background_s288c"],
        ],
        provenance_gaps=[_mating_gap(resolve, name)],
    )


#: The row each background is sourced to: its control row, else its first IDEA row.
BACKGROUND_SOURCE_ROW: dict[BackgroundKey, str] = {
    "DBY12021": "DBY12021",
    "SMY69": "SMY69",
    "SMY254": "SMY254",
    "SMY257_4ZEV": "SMY257",
    "Z3EV_HAP1": "SMY2047",
}


def genome_reference(background: EukaryoticStrainBackground) -> StrainReferenceGenome:
    """The record's reference genome: S288C coordinates, the typed background."""
    return StrainReferenceGenome(
        species="Saccharomyces cerevisiae",
        strain=background.name,
        ploidy="haploid",
        background=background,
        assembly_set=SGD_S288C_R64,
        assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
    )


def insertion_perturbation(
    spec: StrainSpec, systematic: str, standard: str
) -> InduciblePromoterInsertionPerturbation:
    """The designed edit of one strain on its induced gene."""
    assert spec.insertion is not None
    return InduciblePromoterInsertionPerturbation(
        systematic_gene_name=systematic,
        perturbed_gene_name=standard,
        promoter_name=spec.insertion.promoter_name,
        cassette=spec.insertion.cassette,
        marker=spec.insertion.marker,
        responsive_to=spec.insertion.responsive_to,
        native_promoter_retained=True,
        strain_id=spec.row.strain,
    )


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
def culture_format() -> CultureFormat:
    """The chemostat: 500 ml vessel at 300 ml, stirred at 400 rpm, sampled by time."""
    return CultureFormat(
        vessel=VESSEL,
        working_volume_ul=WORKING_VOLUME_UL,
        shaking_rpm=STIR_RPM,
        endpoint=EndpointRule.fixed_duration,
        provenance=[
            SOURCED_VALUES["culture_format"],
            SOURCED_VALUES["vessel"],
            SOURCED_VALUES["chemostat_systems"],
            SOURCED_VALUES["sampling_times"],
        ],
        provenance_gaps=[
            ProvenanceGap(
                field="inoculum_cells",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note="the inoculum is stated as a 1:60 dilution of a saturated "
                "overnight culture, not as a cell count or an OD600",
            ),
            ProvenanceGap(
                field="inoculum_od600",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note="no OD600 is stated for the inoculum",
            ),
        ],
    )


def pre_culture() -> PreCulture:
    """The saturated overnight culture the batch phase started from."""
    return PreCulture(
        source=PreCultureSource.overnight_culture,
        medium=None,
        provenance=[SOURCED_VALUES["pre_culture"]],
        provenance_gaps=[
            ProvenanceGap(
                field="medium",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note="the overnight culture's medium is not stated",
            )
        ],
    )


def estradiol() -> SmallMoleculePerturbation:
    """1 uM beta-estradiol, added to the steady-state chemostat at t = 0."""
    return SmallMoleculePerturbation(
        compound=resolved_compound("estradiol"),
        concentration=Concentration(
            value=INDUCER_UM, unit=ConcentrationUnit.micromolar
        ),
        provenance_gaps=[
            ProvenanceGap(
                field="solvent",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note="the Methods and the GEO protocols state the final 1 uM and no "
                "stock solvent",
            )
        ],
    )


def environment(restriction: str, minutes: float) -> CultureEnvironment:
    """The chemostat of one restriction at ``minutes`` after beta-estradiol addition.

    t = 0 was sampled before the addition, so its environment has no inducer.
    """
    return CultureEnvironment(
        media=MEDIA_BY_RESTRICTION[restriction],
        temperature=Temperature(value=TEMPERATURE_C),
        perturbations=[] if minutes == 0 else [estradiol()],
        aerobicity="aerobic",
        duration_hours=minutes / 60.0,
        dilution_rate_per_hour=None,
        culture_format=culture_format(),
        pre_culture=pre_culture(),
        auxotroph_supplements=None,
        provenance_gaps=[
            ProvenanceGap(
                field="dilution_rate_per_hour",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note="the Methods give a range for all cultures, 'from 0.15 to 0.17/h'; "
                "the per-culture dilution rate is not released, and a range has no "
                "conservative end for a rate, so none is recorded",
            ),
            ProvenanceGap(
                field="auxotroph_supplements",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note="no supplement is stated beyond the limiting nutrient; the "
                "methionine of the met6-delta courses is the limiting nutrient of "
                "their medium",
            ),
        ],
    )


# --------------------------------------------------------------------------- #
# Releases
# --------------------------------------------------------------------------- #
class ReleaseShapeError(ValueError):
    """A release does not have the shape the paper and this loader state."""


def read_tall(path: str | Path) -> pd.DataFrame:
    """The IDEA tall table, exact decimals, every row; refuses a changed shape.

    ``round_trip`` keeps each stored float equal to the TSV's decimal string, so the
    verifier's re-read compares bit for bit.
    """
    with zipfile.ZipFile(path) as archive, archive.open(TALL_MEMBER) as handle:
        frame = pd.read_csv(
            handle,
            sep="\t",
            dtype=dict.fromkeys(TALL_COLUMNS[:5] + ["GeneName"], "category"),
            float_precision="round_trip",
            keep_default_na=False,
        )
    if list(frame.columns) != TALL_COLUMNS:
        raise ReleaseShapeError(f"tall columns {list(frame.columns)}")
    numeric = frame[TALL_COLUMNS[7:]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ReleaseShapeError("a non-finite value in the tall table")
    if len(frame) != N_TALL_ARRAYS * N_GENES:
        raise ReleaseShapeError(
            f"{len(frame)} rows, expected {N_TALL_ARRAYS * N_GENES}"
        )
    return frame


def read_validation(path: str | Path, genes: Iterable[str]) -> pd.DataFrame:
    """Dataset EV2, with the one date-mangled gene cell restored.

    Excel turned ``OCT1`` into 2019-10-01. The cell is restored to the name only when
    the sheet's gene set is the tall table's minus OCT1 plus exactly that one date,
    so a different corruption is refused rather than guessed at.
    """
    frame = pd.read_excel(path, sheet_name=VALIDATION_SHEET)
    if list(frame.columns) != VALIDATION_COLUMNS:
        raise ReleaseShapeError(f"EV2 columns {list(frame.columns)}")
    not_text = frame["GeneName"].map(lambda v: not isinstance(v, str))
    dates = set(frame.loc[not_text, "GeneName"])
    expected = set(genes)
    text_genes = set(frame.loc[~not_text, "GeneName"])
    if dates != {EXCEL_DATE_VALUE} or text_genes != expected - {EXCEL_DATE_GENE}:
        raise ReleaseShapeError(
            f"EV2 gene cells are not the tall genes with OCT1 dated: {sorted(map(str, dates))}"
        )
    frame.loc[not_text, "GeneName"] = EXCEL_DATE_GENE
    numeric = frame[VALIDATION_COLUMNS[5:]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ReleaseShapeError("a non-finite value in Dataset EV2")
    if len(frame) != N_VALIDATION_ARRAYS * N_GENES:
        raise ReleaseShapeError(f"EV2 has {len(frame)} rows")
    return frame


def timecourse_id(
    tf: str, strain: str, date: str, restriction: str, mechanism: str
) -> str:
    """The release's key of an IDEA time course (EV1's header without the time)."""
    return f"{tf}_{strain}_{date}_{restriction}_{mechanism}"


def validation_timecourse_id(tf: str, restriction: str, mechanism: str) -> str:
    """EV2 names a validation time course by its gene, restriction and system."""
    return f"{tf}_{restriction}_{mechanism}"


# --------------------------------------------------------------------------- #
# Gene keys
# --------------------------------------------------------------------------- #
#: The induced gene's released token -> the ORF the KG build served it under. The
#: release names the induced gene only by a common-name token (Table EV1's genotype
#: string, the tall table's ``TF``; ``ACE1`` in Table EV1 is ``CUP2`` in the release,
#: ``YGR066c`` is Table EV1's spelling of the release's ``YGR066C``) and the insertion
#: leaf requires an ORF. This table was derived from the 2026.10.11 build's
#: ``gene_keys.json`` once; the loader makes no genome call. It is the one remaining
#: crosswalk, pending group 1's released-identifier leaf (a GenePerturbation child that
#: carries the released symbol with ``systematic_gene_name`` as a typed gap), after which
#: the token is stored verbatim and this table goes.
RELEASE_TF_TOKENS_WITHOUT_AN_ORF: dict[str, str] = {
    "ACA1": "YER045C",
    "ACE1": "YGL166W",
    "ACE2": "YLR131C",
    "ADA2": "YDR448W",
    "ADR1": "YDR216W",
    "AFT1": "YGL071W",
    "AFT2": "YPL202C",
    "ANB1": "YJR047C",
    "ARG80": "YMR042W",
    "ARG81": "YML099C",
    "ARN2": "YHL047C",
    "ARO80": "YDR421W",
    "ARR1": "YPR199C",
    "ASH1": "YKL185W",
    "AZF1": "YOR113W",
    "BAS1": "YKR099W",
    "BDF2": "YDL070W",
    "BMH1": "YER177W",
    "CAD1": "YDR423C",
    "CAM1": "YPL048W",
    "CAT8": "YMR280C",
    "CBF1": "YJR060W",
    "CDC73": "YLR418C",
    "CHA4": "YLR098C",
    "CIN5": "YOR028C",
    "CRF1": "YDR223W",
    "CRZ1": "YNL027W",
    "CSE2": "YNR010W",
    "CST6": "YIL036W",
    "CTI6": "YPL181W",
    "CTR9": "YOL145C",
    "CUP2": "YGL166W",
    "CUP9": "YPL177C",
    "DAL80": "YKR034W",
    "DAL81": "YIR023W",
    "DAL82": "YNL314W",
    "DIG1": "YPL049C",
    "DIG2": "YDR480W",
    "DST1": "YGL043W",
    "ECM22": "YLR228C",
    "ERT1": "YBR239C",
    "FAP1": "YNL023C",
    "FHL1": "YPR104C",
    "FKH1": "YIL131C",
    "FKH2": "YNL068C",
    "FLO8": "YER109C",
    "FMP48": "YGR052W",
    "FZF1": "YGL254W",
    "GAL11": "YOL051W",
    "GAL4": "YPL248C",
    "GAL80": "YML051W",
    "GAT2": "YMR136W",
    "GAT3": "YLR013W",
    "GAT4": "YIR013C",
    "GCN4": "YEL009C",
    "GCN5": "YGR252W",
    "GCR1": "YPL075W",
    "GCR2": "YNL199C",
    "GIS1": "YDR096W",
    "GLN3": "YER040W",
    "GRX3": "YDR098C",
    "GRX4": "YER174C",
    "GSM1": "YJL103C",
    "GTS1": "YGL181W",
    "GZF3": "YJL110C",
    "HAA1": "YPR008W",
    "HAC1": "YFL031W",
    "HAP1": "YLR256W",
    "HAP2": "YGL237C",
    "HAP3": "YBL021C",
    "HAP4": "YKL109W",
    "HAP5": "YOR358W",
    "HCM1": "YCR065W",
    "HFI1": "YPL254W",
    "HIR1": "YBL008W",
    "HIR2": "YOR038C",
    "HIR3": "YJR140C",
    "HMRA1": "YCR097W",
    "HMS1": "YOR032C",
    "HMX1": "YLR205C",
    "HOT1": "YMR172W",
    "IME1": "YJR094C",
    "INO4": "YOL108C",
    "KAR4": "YCL055W",
    "KSS1": "YGR040W",
    "LEO1": "YOR123C",
    "LEU3": "YLR451W",
    "LOT6": "YLR011W",
    "LYS14": "YDR034C",
    "MAC1": "YMR021C",
    "MBP1": "YDL056W",
    "MED2": "YDL005C",
    "MET28": "YIR017C",
    "MET31": "YPL038W",
    "MET32": "YDR253C",
    "MET4": "YNL103W",
    "MGA1": "YGR249W",
    "MIG1": "YGL035C",
    "MIG2": "YGL209W",
    "MIG3": "YER028C",
    "MOT3": "YMR070W",
    "MSN2": "YMR037C",
    "MSN4": "YKL062W",
    "MSS11": "YMR164C",
    "MTF1": "YMR228W",
    "NDT80": "YHR124W",
    "NRG1": "YDR043C",
    "NRM1": "YNR009W",
    "OAF1": "YAL051W",
    "OAF3": "YKR064W",
    "OPI1": "YHL020C",
    "PAF1": "YBR279W",
    "PDR1": "YGL013C",
    "PDR3": "YBL005W",
    "PDR8": "YLR266C",
    "PGD1": "YGL025C",
    "PHD1": "YKL043W",
    "PHM6": "YDR281C",
    "PHO11": "YAR071W",
    "PHO2": "YDL106C",
    "PHO4": "YFR034C",
    "PHO5": "YBR093C",
    "PIP2": "YOR363C",
    "PPR1": "YLR014C",
    "RDR1": "YOR380W",
    "RDS2": "YPL133C",
    "RFX1": "YLR176C",
    "RGM1": "YMR182C",
    "RGR1": "YLR071C",
    "RGT1": "YKL038W",
    "RIM101": "YHL027W",
    "RLM1": "YPL089C",
    "RME1": "YGR044C",
    "ROX1": "YPR065W",
    "ROX3": "YBL093C",
    "RPD3": "YNL330C",
    "RPH1": "YER169W",
    "RPN4": "YDL020C",
    "RRN10": "YBL025W",
    "RSF2": "YJR127C",
    "RTF1": "YGL244W",
    "RTG1": "YOL067C",
    "RTG2": "YGL252C",
    "RTG3": "YBL103C",
    "SFL1": "YOR140W",
    "SFP1": "YLR403W",
    "SGF29": "YCL010C",
    "SIN3": "YOL004W",
    "SIN4": "YNL236W",
    "SIP3": "YNL257C",
    "SIP4": "YJL089W",
    "SIR2": "YDL042C",
    "SKN7": "YHR206W",
    "SKO1": "YNL167C",
    "SMP1": "YBR182C",
    "SNF2": "YOR290C",
    "SNF5": "YBR289W",
    "SOK2": "YMR016C",
    "SPT20": "YOL148C",
    "SPT3": "YDR392W",
    "SPT4": "YGR063C",
    "SPT8": "YLR055C",
    "SRB2": "YHR041C",
    "SRB5": "YGR104C",
    "SRB8": "YCR081W",
    "SSN2": "YDR443C",
    "STB1": "YNL309W",
    "STB5": "YHR178W",
    "STE12": "YHR084W",
    "STP1": "YDR463W",
    "STP2": "YHR006W",
    "STP4": "YDL048C",
    "SUB1": "YMR039C",
    "SUM1": "YDR310C",
    "SUT1": "YGL162W",
    "SUT2": "YPR009W",
    "SWI1": "YPL016W",
    "SWI5": "YDR146C",
    "SWI6": "YLR182W",
    "TAF2": "YCR042C",
    "TEA1": "YOR337W",
    "TEC1": "YBR083W",
    "TFB5": "YDR079C-A",
    "THI3": "YDL080C",
    "TOG1": "YER184C",
    "TYE7": "YOR344C",
    "UAF30": "YOR295W",
    "UGA3": "YDL170W",
    "UME1": "YPL139C",
    "UME6": "YDR207C",
    "UPC2": "YDR213W",
    "URE2": "YNL229C",
    "USV1": "YPL230W",
    "VHR1": "YIL056W",
    "WAR1": "YML076C",
    "WHI5": "YOR083W",
    "WTM1": "YOR230W",
    "WTM2": "YOR229W",
    "XBP1": "YIL101C",
    "YAP1": "YML007W",
    "YAP3": "YHL009C",
    "YAP5": "YIR018W",
    "YAP6": "YDR259C",
    "YGL117W": "YGL117W",
    "YGR066C": "YGR066C",
    "YGR066c": "YGR066C",
    "YHP1": "YDR451C",
    "YLL054C": "YLL054C",
    "YOX1": "YML027W",
    "YRM1": "YOR172W",
    "YRR1": "YOR162C",
    "ZAP1": "YJL056C",
    "ZNF1": "YFL052W",
}


def induced_orf(token: str) -> str:
    """The ORF the pinned table pairs with a released induced-gene token."""
    if token not in RELEASE_TF_TOKENS_WITHOUT_AN_ORF:
        raise StrainTableError(
            f"{token!r}: an induced-gene token the pinned table does not carry"
        )
    return RELEASE_TF_TOKENS_WITHOUT_AN_ORF[token]


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
TALL_LAYERS: dict[str, str] = {
    "log2_cleaned": "log2_cleaned_ratio",
    "log2_noise_sd": "log2_noise_model",
    "sample_channel_intensity": "red_median",
    "reference_channel_intensity": "green_median",
    "log2_ratio_unprocessed": "log2_ratio",
    "log2_thresholded": "log2_cleaned_ratio_zth2d",
    "log2_selected": "log2_selected_timecourses",
    "log2_shrunken": "log2_shrunken_timecourses",
}
VALIDATION_LAYERS: dict[str, str] = {
    "log2_cleaned": "log2_cleaned_ratio",
    "sample_channel_intensity": "red_median",
    "reference_channel_intensity": "green_median",
    "log2_ratio_unprocessed": "log2_ratio",
}
_EV2_NOT_RELEASED = {
    "log2_noise_sd": "Dataset EV2 releases no noise-model column for the validation "
    "time courses",
    "log2_thresholded": "Dataset EV2 releases the unprocessed and cleaned ratios only",
    "log2_selected": "Dataset EV2 releases the unprocessed and cleaned ratios only",
    "log2_shrunken": "Dataset EV2 releases the unprocessed and cleaned ratios only",
}


def phenotype(
    layers: Mapping[str, dict[str, float]],
    *,
    timecourse: str,
    strain_label: str,
    experiment_date: str | None,
    release_table: str,
) -> InductionTimecourseExpressionPhenotype:
    """One array's phenotype; a layer the release does not carry is a typed gap."""
    gaps = [
        ProvenanceGap(
            field=field, reason=ProvenanceGapReason.not_reported_by_primary, note=note
        )
        for field, note in _EV2_NOT_RELEASED.items()
        if field not in layers
    ]
    if experiment_date is None:
        gaps.append(
            ProvenanceGap(
                field="experiment_date",
                reason=ProvenanceGapReason.not_reported_by_primary,
                note="Dataset EV2 has no date column",
            )
        )
    return InductionTimecourseExpressionPhenotype(
        log2_cleaned=layers["log2_cleaned"],
        log2_noise_sd=layers.get("log2_noise_sd"),
        sample_channel_intensity=layers["sample_channel_intensity"],
        reference_channel_intensity=layers["reference_channel_intensity"],
        log2_ratio_unprocessed=layers["log2_ratio_unprocessed"],
        log2_thresholded=layers.get("log2_thresholded"),
        log2_selected=layers.get("log2_selected"),
        log2_shrunken=layers.get("log2_shrunken"),
        timecourse_id=timecourse,
        strain_label=strain_label,
        experiment_date=experiment_date,
        release_table=release_table,
        provenance_gaps=gaps,
    )


class Timecourse(BaseModel):
    """One released time course resolved to its strain, ready to emit records."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    timecourse_id: str
    group_key: list[str]
    release_table: str
    tf: str
    strain_label: str
    table_strain: str
    experiment_date: str | None
    restriction: str
    mechanism: str
    times: list[float]


class RefusedTimecourse(BaseModel):
    """A released time course the build does not serve, with the reason."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    timecourse_id: str
    strain_label: str
    n_arrays: int
    reason: str


class ReleaseLedger(BaseModel):
    """Every released array accounted for (``preprocess/release_ledger.json``)."""

    model_config = ConfigDict(extra="forbid")

    tall_sha256: str
    validation_sha256: str
    strains_sha256: str
    n_genes: int
    n_tall_timecourses: int
    n_tall_arrays: int
    n_validation_timecourses: int
    n_validation_arrays: int
    n_records: int
    n_served_timecourses: int
    refused: list[RefusedTimecourse]
    n_refused_arrays: int
    excel_date_gene_restored: str
    n_gene_keys_as_released: int
    n_tf_tokens_pinned: int
    arrays_per_timecourse: dict[str, int]
    strains_per_background: dict[str, int]
    yrsm_identified: dict[str, str]

    @model_validator(mode="after")
    def _accounted(self) -> ReleaseLedger:
        served = self.n_records + self.n_refused_arrays
        if served != self.n_tall_arrays + self.n_validation_arrays:
            raise ValueError("served + refused arrays must equal released arrays")
        if sum(r.n_arrays for r in self.refused) != self.n_refused_arrays:
            raise ValueError("refused arrays do not sum")
        return self


def publication() -> Publication:
    """The Mol Syst Biol paper."""
    return Publication(
        pubmed_id=PAPER_PMID,
        pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{PAPER_PMID}/",
        doi=PAPER_DOI,
        doi_url=f"https://doi.org/{PAPER_DOI}",
    )


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    from dotenv import load_dotenv

    load_dotenv()
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/hackettLearningCausalNetworks2020``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def library_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-library/hackettLearningCausalNetworks2020``."""
    return Path(data_root or _data_root()) / "torchcell-library" / CITATION_KEY


def gcs_md5_base64(path: str | Path) -> str:
    """Base64 md5 of a file, the form a GCS object listing publishes."""
    digest = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode()


def _direct(url: str, sha: str, retrieved_at: str, **params: str) -> RetrievalRecord:
    return RetrievalRecord(
        method=RetrievalMethod.direct_url,
        source_url=url,
        retriever="torchcell.literature.retrieve.direct_url",
        params={"url": url, **params},
        sha256=sha,
        retrieved_at=retrieved_at,
    )


def deposit_raw_mirror(
    *, download_dir: str | Path, retrieved_at: str, data_root: str | None = None
) -> Path:
    """Write the raw mirror from already-retrieved files and its ``manifest.json``.

    Run once by hand after ``curl -sSL -O`` of the four direct-URL objects into
    ``download_dir`` (the tall zip from the IDEA bucket, the site page that defines its
    columns, the two GEO series matrices). The zip's base64 md5 must equal the one the
    bucket's object listing publishes. The two SI tables are copied from the library
    mirror, each checked against the library manifest's sha256 and recorded with the
    library's ``pmc_cloud`` retrieval. Idempotent by sha256; a differing file raises.
    """
    root = raw_mirror_dir(data_root)
    download = Path(download_dir)
    tall = download / TALL_ZIP_NAME
    if gcs_md5_base64(tall) != TALL_ZIP_GCS_MD5_BASE64:
        raise ValueError(f"{tall} does not have the bucket's published md5")
    direct = {
        TALL_REL: (
            tall,
            f"{IDEA_BUCKET}/{TALL_ZIP_NAME}",
            {
                "gcs_md5_base64": TALL_ZIP_GCS_MD5_BASE64,
                "gcs_generation": TALL_ZIP_GCS_GENERATION,
            },
        ),
        SITE_JS_REL: (download / "idea_main.js", SITE_JS_URL, {}),
        GEO_GPL16244_REL: (
            download / GEO_GPL16244_NAME,
            GEO_URL + GEO_GPL16244_NAME,
            {},
        ),
        GEO_GPL20785_REL: (
            download / GEO_GPL20785_NAME,
            GEO_URL + GEO_GPL20785_NAME,
            {},
        ),
    }
    library = Manifest.model_validate_json(
        (library_dir(data_root) / "manifest.json").read_text()
    )
    by_path = {record.path: record for record in library.files}
    files: list[ArtifactRecord] = []
    for relpath, (src, url, params) in direct.items():
        files.append(
            _deposit(root, relpath, src, ROLE_RAW_DATA, url, retrieved_at, params)
        )
    for relpath, library_rel in (
        (STRAINS_REL, STRAINS_LIBRARY),
        (VALIDATION_REL, VALIDATION_LIBRARY),
    ):
        record = by_path[library_rel]
        src = library_dir(data_root) / library_rel
        if sha256_file(src) != record.sha256:
            raise RuntimeError(f"{src} differs from the library manifest's sha256")
        dest = root / relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if sha256_file(dest) != record.sha256:
                raise RuntimeError(f"{dest} exists with a different sha256; refusing")
        else:
            shutil.copy2(src, dest)
        files.append(
            ArtifactRecord(
                path=relpath,
                role=ROLE_SI_DATA,
                bytes=dest.stat().st_size,
                sha256=record.sha256,
                source=record.source,
                retrieval=record.retrieval,
            )
        )
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=PAPER_DOI,
        title=PAPER_TITLE,
        files=files,
        si_data_sources=[IDEA_BUCKET, GEO_URL, PMC_BUCKET, IDEA_SITE],
        si_expected=[TALL_ZIP_NAME, STRAINS_NAME, VALIDATION_NAME],
        provenance_complete=True,
        created_at=datetime.now(UTC).isoformat(),
    )
    (root / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    return root


def _deposit(
    root: Path,
    relpath: str,
    src: Path,
    role: str,
    url: str,
    retrieved_at: str,
    params: dict[str, str],
) -> ArtifactRecord:
    sha = sha256_file(src)
    dest = root / relpath
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if sha256_file(dest) != sha:
            raise RuntimeError(f"{dest} exists with a different sha256; refusing")
    else:
        shutil.copy2(src, dest)
    return ArtifactRecord(
        path=relpath,
        role=role,
        bytes=dest.stat().st_size,
        sha256=sha,
        source=url,
        retrieval=_direct(url, sha, retrieved_at, **params),
    )


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
# Dataset
# --------------------------------------------------------------------------- #
@register_dataset
class InductionTimecourseHackett2020Dataset(ExperimentDataset):
    """Hackett 2020 IDEA: one record per (time course, time point), 1,752 records."""

    organism = Organism.scerevisiae

    def __init__(
        self,
        root: str = "data/torchcell/induction_timecourse_hackett2020",
        io_workers: int = 0,
        transform: Any | None = None,
        pre_transform: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Build or load the store under ``root``."""
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return InductionTimecourseExpressionExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return InductionTimecourseExpressionExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The tall table, Table EV1 and Dataset EV2."""
        return list(RAW_FILES)

    def download(self) -> None:
        """Link the three raw files from the raw mirror, each sha256-verified.

        Each pin is the module constant; the raw-mirror manifest must record the same
        digest (``ManifestPinMismatchError`` otherwise).
        """
        data_root = _data_root()
        manifest = load_manifest(data_root)
        os.makedirs(self.raw_dir, exist_ok=True)
        for name, relpath in RAW_FILES.items():
            check_manifest_pin(
                relpath, manifest_sha256(manifest, relpath), RAW_SHA256[name]
            )
            link_verified(
                raw_mirror_dir(data_root) / relpath,
                osp.join(self.raw_dir, name),
                RAW_SHA256[name],
            )
        log.info(
            "Hackett 2020 raw files linked into %s (sha256 verified)", self.raw_dir
        )

    # ---- build ---------------------------------------------------------------
    @post_process
    def process(self) -> None:
        """Build every served array, time course by time course, time ascending."""
        verify_raw_files(self.raw_dir, RAW_SHA256)
        os.makedirs(self.preprocess_dir, exist_ok=True)
        strains = read_strain_table(osp.join(self.raw_dir, STRAINS_NAME))
        specs = {name: parse_strain(row) for name, row in strains.items()}
        tall = read_tall(osp.join(self.raw_dir, TALL_ZIP_NAME))
        genes = sorted(tall["GeneName"].unique())
        if len(genes) != N_GENES:
            raise ReleaseShapeError(f"{len(genes)} genes, expected {N_GENES}")
        validation = read_validation(osp.join(self.raw_dir, VALIDATION_NAME), genes)
        standard_of = self._standard_names(tall, validation)

        backgrounds = {
            key: build_background(key, strains[row])
            for key, row in BACKGROUND_SOURCE_ROW.items()
        }
        references_of_bg = {k: genome_reference(b) for k, b in backgrounds.items()}
        pub = publication()

        tall_courses, refused = self._tall_timecourses(tall, specs)
        valid_courses = self._validation_timecourses(validation, specs)
        groups = {
            TALL_ZIP_NAME: tall.groupby(
                list(TALL_COLUMNS[:5]), sort=False, observed=True
            ),
            VALIDATION_NAME: validation.groupby(
                ["TF", "restriction", "mechanism"], sort=False
            ),
        }

        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        idx = 0
        n_by_course: dict[str, int] = {}
        gene_names: set[str] = set()
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            env_pointers: dict[tuple[str, float], dict[str, str]] = {}
            pub_pointer = self._intern(pub, PAPER_DOI, itxn)
            for course in [*tall_courses, *valid_courses]:
                spec = specs[course.table_strain]
                genotype = self._genotype(spec, standard_of)
                gene_names.update(genotype.systematic_gene_names)
                arrays = self._arrays(
                    groups[course.release_table].get_group(tuple(course.group_key)),
                    course,
                )
                reference = InductionTimecourseExpressionExperimentReference(
                    dataset_name=self.name,
                    genome_reference=references_of_bg[spec.background],
                    environment_reference=environment(course.restriction, 0.0),
                    phenotype_reference=arrays[0.0],
                )
                ref_pointer = self._intern(reference, course.timecourse_id, itxn)
                for minutes in course.times:
                    env_key = (course.restriction, minutes)
                    environment_ = environment(*env_key)
                    if env_key not in env_pointers:
                        env_pointers[env_key] = self._intern(
                            environment_, environment_.media.stated_name, itxn
                        )
                    experiment = InductionTimecourseExpressionExperiment(
                        dataset_name=self.name,
                        genotype=genotype,
                        environment=environment_,
                        phenotype=arrays[minutes],
                    )
                    exp = experiment.model_dump(exclude={"environment"})
                    exp["environment"] = dict(env_pointers[env_key])
                    record = {
                        "experiment": exp,
                        "reference": dict(ref_pointer),
                        "publication": dict(pub_pointer),
                    }
                    txn.put(f"{idx}".encode(), pickle.dumps(record))
                    idx += 1
                n_by_course[course.timecourse_id] = len(course.times)
        env.close()
        interned_env.close()
        self._gene_names = gene_names
        self._write_ledger(
            idx, tall_courses, valid_courses, refused, genes, specs, n_by_course
        )
        pd.DataFrame(
            [
                {
                    "timecourse_id": c.timecourse_id,
                    "release_table": c.release_table,
                    "table_strain": c.table_strain,
                    "n_arrays": len(c.times),
                }
                for c in [*tall_courses, *valid_courses]
            ]
        ).to_csv(osp.join(self.preprocess_dir, "data.csv"), index=False)
        log.info("Wrote %d Hackett 2020 records", idx)

    @staticmethod
    def _intern(obj: BaseModel, hint: str, itxn: Any) -> dict[str, str]:
        """Intern a constant sub-object once; return the pointer records carry.

        The digest and bytes are what ``_maybe_intern`` writes (sha256 of the canonical
        JSON, the python-mode dump pickled), computed once per object.
        """
        digest = compute_sha256_hash(canonical_json(obj))
        if itxn.get(digest.encode()) is None:
            itxn.put(digest.encode(), pickle.dumps(obj.model_dump()))
        return {"$ref": digest, "name": str(hint)}

    @staticmethod
    def _standard_names(tall: pd.DataFrame, validation: pd.DataFrame) -> dict[str, str]:
        """Each induced gene's pinned ORF -> the release's TF token for it.

        The table and the release must name the same gene (``ACE1`` in Table EV1 and
        ``CUP2`` in the release are pinned to YGL166W); the release's token is the
        stored standard name, since it is what the arrays are keyed under.
        """
        out: dict[str, str] = {}
        for tf in [*tall["TF"].unique(), *validation["TF"].unique()]:
            if str(tf) in RELEASE_TF_TOKENS_WITHOUT_AN_ORF:
                out[induced_orf(str(tf))] = str(tf)
        return out

    @staticmethod
    def _genotype(spec: StrainSpec, standard_of: Mapping[str, str]) -> Genotype:
        """The strain's designed edit (none for a control strain).

        The induced gene is the table token's pinned ORF; its stored standard name is
        the release's TF token for that gene (``CUP2`` where Table EV1 writes ``ACE1``),
        since the release names the arrays' genes that way.
        """
        if spec.insertion is None:
            return Genotype(perturbations=[])
        systematic = induced_orf(spec.insertion.gene_token)
        standard = standard_of[systematic]
        return Genotype(
            perturbations=[insertion_perturbation(spec, systematic, standard)]
        )

    @staticmethod
    def _tall_timecourses(
        tall: pd.DataFrame, specs: Mapping[str, StrainSpec]
    ) -> tuple[list[Timecourse], list[RefusedTimecourse]]:
        """The tall table's time courses in release order, refusals split out."""
        key = list(TALL_COLUMNS[:5])
        courses: list[Timecourse] = []
        refused: list[RefusedTimecourse] = []
        grouped = tall.groupby(key, sort=False, observed=True)["time"].unique()
        if len(grouped) != N_TALL_TIMECOURSES:
            raise ReleaseShapeError(f"{len(grouped)} tall time courses")
        for raw_key, times in grouped.items():
            group_key = cast(tuple[Any, ...], raw_key)
            tf, strain, date, restriction, mechanism = (str(v) for v in group_key)
            tc = timecourse_id(tf, strain, date, restriction, mechanism)
            table_strain = table_strain_of(strain)
            ordered = sorted(float(t) for t in times)
            if table_strain in UNLISTED_STRAINS:
                if table_strain in specs:
                    raise StrainTableError(f"{table_strain} is listed after all")
                refused.append(
                    RefusedTimecourse(
                        timecourse_id=tc,
                        strain_label=strain,
                        n_arrays=len(ordered),
                        reason=REFUSAL_UNLISTED_STRAIN,
                    )
                )
                continue
            if ordered[0] != 0.0:
                raise ReleaseShapeError(f"{tc} has no t = 0 array")
            _check_tf_matches(specs[table_strain], tf, mechanism)
            courses.append(
                Timecourse(
                    timecourse_id=tc,
                    group_key=[str(v) for v in group_key],
                    release_table=TALL_ZIP_NAME,
                    tf=tf,
                    strain_label=strain,
                    table_strain=table_strain,
                    experiment_date=date,
                    restriction=restriction,
                    mechanism=mechanism,
                    times=ordered,
                )
            )
        return courses, refused

    @staticmethod
    def _validation_timecourses(
        validation: pd.DataFrame, specs: Mapping[str, StrainSpec]
    ) -> list[Timecourse]:
        """EV2's ten courses, each matched to its one Table EV1 'Prediction' row."""
        by_gene = {
            spec.insertion.gene_token.upper(): name
            for name, spec in specs.items()
            if spec.row.strain_type == "Prediction" and spec.insertion is not None
        }
        courses: list[Timecourse] = []
        grouped = validation.groupby(["TF", "restriction", "mechanism"], sort=False)[
            "time"
        ].unique()
        if len(grouped) != N_VALIDATION_TIMECOURSES:
            raise ReleaseShapeError(f"{len(grouped)} validation time courses")
        for raw_key, times in grouped.items():
            group_key = cast(tuple[Any, ...], raw_key)
            tf, restriction, mechanism = (str(v) for v in group_key)
            strain = by_gene[tf.upper()]
            ordered = sorted(float(t) for t in times)
            if ordered[0] != 0.0:
                raise ReleaseShapeError(f"validation {tf} has no t = 0 array")
            courses.append(
                Timecourse(
                    timecourse_id=validation_timecourse_id(tf, restriction, mechanism),
                    group_key=[str(v) for v in group_key],
                    release_table=VALIDATION_NAME,
                    tf=tf,
                    strain_label=strain,
                    table_strain=strain,
                    experiment_date=None,
                    restriction=restriction,
                    mechanism=mechanism,
                    times=ordered,
                )
            )
        return courses

    @staticmethod
    def _arrays(
        sub: pd.DataFrame, course: Timecourse
    ) -> dict[float, InductionTimecourseExpressionPhenotype]:
        """Every array of one time course (its release rows), keyed by minutes.

        Each layer is keyed by the release's ``GeneName`` token verbatim.
        """
        layers_of = (
            TALL_LAYERS if course.release_table == TALL_ZIP_NAME else VALIDATION_LAYERS
        )
        out: dict[float, InductionTimecourseExpressionPhenotype] = {}
        for minutes, block in sub.groupby("time", sort=True):
            if len(block) != N_GENES or block["GeneName"].nunique() != N_GENES:
                raise ReleaseShapeError(
                    f"{course.timecourse_id} t={minutes}: {len(block)} gene rows"
                )
            keys = [str(g) for g in block["GeneName"]]
            layers = {
                field: dict(
                    zip(keys, block[column].astype(float).tolist(), strict=True)
                )
                for field, column in layers_of.items()
            }
            out[float(cast(float, minutes))] = phenotype(
                layers,
                timecourse=course.timecourse_id,
                strain_label=course.strain_label,
                experiment_date=course.experiment_date,
                release_table=course.release_table,
            )
        return out

    def _write_ledger(
        self,
        n_records: int,
        tall_courses: list[Timecourse],
        valid_courses: list[Timecourse],
        refused: list[RefusedTimecourse],
        genes: list[str],
        specs: Mapping[str, StrainSpec],
        n_by_course: Mapping[str, int],
    ) -> None:
        used = {c.table_strain for c in [*tall_courses, *valid_courses]}
        per_bg: dict[str, int] = {}
        for strain in used:
            bg = specs[strain].background
            per_bg[bg] = per_bg.get(bg, 0) + 1
        ledger = ReleaseLedger(
            tall_sha256=TALL_ZIP_SHA256,
            validation_sha256=VALIDATION_SHA256,
            strains_sha256=STRAINS_SHA256,
            n_genes=N_GENES,
            n_tall_timecourses=N_TALL_TIMECOURSES,
            n_tall_arrays=N_TALL_ARRAYS,
            n_validation_timecourses=N_VALIDATION_TIMECOURSES,
            n_validation_arrays=N_VALIDATION_ARRAYS,
            n_records=n_records,
            n_served_timecourses=len(tall_courses) + len(valid_courses),
            refused=refused,
            n_refused_arrays=sum(r.n_arrays for r in refused),
            excel_date_gene_restored=EXCEL_DATE_GENE,
            n_gene_keys_as_released=len(genes),
            n_tf_tokens_pinned=len(RELEASE_TF_TOKENS_WITHOUT_AN_ORF),
            arrays_per_timecourse=dict(n_by_course),
            strains_per_background=dict(sorted(per_bg.items())),
            yrsm_identified={
                c.strain_label: c.table_strain
                for c in tall_courses
                if c.strain_label.startswith("yRSM")
            },
        )
        with open(osp.join(self.preprocess_dir, "release_ledger.json"), "w") as handle:
            handle.write(ledger.model_dump_json(indent=2))
        with open(osp.join(self.preprocess_dir, "gene_keys.json"), "w") as handle:
            json.dump(
                {
                    "rule": "every expression layer is keyed by the release's "
                    "GeneName token verbatim; the induced gene's ORF comes from "
                    "RELEASE_TF_TOKENS_WITHOUT_AN_ORF, pending group 1's "
                    "released-identifier leaf",
                    "n_gene_keys_as_released": len(genes),
                    "tf_tokens_pinned": dict(
                        sorted(RELEASE_TF_TOKENS_WITHOUT_AN_ORF.items())
                    ),
                },
                handle,
                indent=1,
            )

    # ---- gene set --------------------------------------------------------------
    def compute_gene_set(self) -> GeneSet:
        """The induced genes' systematic names (one read of every record otherwise)."""
        names: set[str] | None = getattr(self, "_gene_names", None)
        if names is None:
            names = set()
            env = lmdb.open(
                osp.join(self.processed_dir, "lmdb"), readonly=True, lock=False
            )
            with env.begin() as txn:
                for _, value in txn.cursor():
                    genotype = pickle.loads(value)["experiment"]["genotype"]
                    names.update(self.extract_systematic_gene_names(genotype))
            env.close()
        return GeneSet(names)

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Preprocessing happens inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Records are assembled inside process()."""
        raise NotImplementedError(
            "InductionTimecourseHackett2020Dataset builds its records in process()"
        )


def _check_tf_matches(spec: StrainSpec, tf: str, mechanism: str) -> None:
    """The release's induced gene and system agree with the strain's Table EV1 row.

    A control row has no insertion and the release names its activator (``GEV``,
    ``Z3EV``) as the TF. Otherwise both tokens must be pinned to ONE ORF (``CUP2`` in
    the release and ``ACE1`` in Table EV1 are YGL166W), and the release's system must
    be the promoter's family (Pgal1 answers GEV; Z3EVpr and 4ZEVpr answer ZEV-family
    activators, which the release writes ``ZEV``).
    """
    if spec.insertion is None:
        if tf not in {"GEV", "Z3EV"}:
            raise StrainTableError(f"{spec.row.strain}: control row induced as {tf}")
        return
    family = "GEV" if spec.insertion.responsive_to == "GEV" else "ZEV"
    if mechanism != family:
        raise StrainTableError(
            f"{spec.row.strain}: release system {mechanism}, table promoter "
            f"{spec.insertion.promoter_name}"
        )
    if induced_orf(tf) != induced_orf(spec.insertion.gene_token):
        raise StrainTableError(
            f"{spec.row.strain}: release induces {tf}, table names "
            f"{spec.insertion.gene_token}"
        )


def verify_build(dataset_root: str, data_root: str | None = None) -> Any:
    """L0-L4 report for a built store (``build_dataset_lmdb --verify`` calls this)."""
    from torchcell.verification import runners

    return runners.verify_induction_timecourse_dataset(
        "induction_timecourse_hackett2020",
        data_root or _data_root(),
        abs_root=dataset_root,
    )


def main() -> None:
    """Build/load the dataset and print one record's shape."""
    data_root = _data_root()
    dataset = InductionTimecourseHackett2020Dataset(
        root=osp.join(data_root, "data/torchcell/induction_timecourse_hackett2020")
    )
    print(f"len = {len(dataset)}")
    record = dataset[1]
    print(record["experiment"].genotype, record["experiment"].phenotype)


if __name__ == "__main__":
    main()
