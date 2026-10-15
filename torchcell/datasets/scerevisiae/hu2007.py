# torchcell/datasets/scerevisiae/hu2007
# [[torchcell.datasets.scerevisiae.hu2007]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/scerevisiae/hu2007
# Test file: tests/torchcell/datasets/scerevisiae/test_hu2007.py
"""Hu 2007 transcription-factor deletion transcriptomes (error-model X scores).

Hu, Killion and Iyer 2007 (Nat Genet 39:683-687, doi:10.1038/ng2012; PMID 17417638)
profiled S. cerevisiae strains carrying single transcription-factor (TF) perturbations
against a wild-type reference on two-color spotted cDNA arrays (GEO platform GPL3588,
"UT Austin Iyer yeast 6.4k ORF-only cDNA platform") and deposited the result as GEO
series GSE4654: one sample per strain, 269 samples, 6,429 array elements each.

WHAT THE RELEASE IS. Every GSE4654 sample carries two columns per array element,
``#VALUE = X score`` and ``#P = p value``, and the series summary says "Data values are
X scores and corresponding p-values derived using an error model". The X score is the
Hughes 2000 error model's "normalized, confidence-weighted measure of each ratio",
averaged over the replicate arrays ("weighted duplicate averaging"); the p-value is its
two-sided normal tail (measured: ``max |p - 2 * (1 - Phi(|X|))| = 5.0e-7`` over all
1,707,575 released scores, the release's six-decimal rounding). No log ratio and no
channel intensity is released: the per-array data went to the lab's Longhorn Array
Database and ``www.iyerlab.org/tfnetwork``, which now redirects to a faculty page and
was never archived (Wayback CDX for ``iyerlab.org``, 2026-10-11). The score is
therefore stored as released in ``MicroarrayExpressionScorePhenotype`` (#1022), never
as a log ratio in ``MicroarrayExpressionPhenotype``.

RECORD = one GSE4654 sample whose strain is a deletion grown to mid-log in YPD:

- GENOTYPE: one ``KanMxDeletionPerturbation`` in BY4741 (the Open Biosystems
  collection, ``MATa his3D1 leu2D0 met15D0 ura3D0``), named as the release names it:
  the systematic name of Supplementary Table 1 and its gene name, or the systematic
  name again where the table gives no gene name (the release names that strain by its
  ORF alone).
- ENVIRONMENT: ``YPD_LIQUID``; the growth temperature is stated nowhere in the paper or
  the Supplementary Methods (the 39 C figure is the heat-shock arm's shift), so it is a
  typed ``ProvenanceGap``. Aerobicity is the schema default; the paper states none.
- PHENOTYPE: ``expression_score`` = the released X score per element, keyed by the
  element's released GPL3588 ID, ``expression_score_p_value`` = the released p-value,
  ``n_replicates = 2`` (the CONSERVATIVE lower end of "Each experiment was
  independently carried out at least twice", which the "weighted duplicate averaging"
  sentence corroborates), and ``harvest_optical_density`` = the strain's "Average OD of
  replicates" from Supplementary Table 1.
- REFERENCE: the wild-type culture "cultured together with each batch of mutants",
  labeled Cy3 against the deletion's Cy5 on every sample (no dye swap is released). Its
  score is the neutral 0 for every element the record scores, with no p-value.

ARRAY ELEMENTS. GPL3588's ID column holds 2006-era ORF names ("May not match current
SGD annotations"). Every element is stored under that released ID, verbatim, with no
resolution against R64 and no status beside it (owner rule, 2026-10-12: the data is
the data; no loader drops a released row on genome or annotation grounds). So the
intergenic (``i``- and ``Int``-prefixed), Ty and LTR elements, the ORFs R64 retired and
the dubious ORFs a neighbour absorbed are all keys, each under its own released name;
the 6,429 IDs are unique as released. A released null score is not a key: over the 263
kept samples 9 values are null in both columns and 18,645 carry ``p = 1`` beside a
null score, which leaves no score to key the p-value to (3,180 and 18,646 over all 269
samples). ``preprocess/array_elements.json`` counts them.

RECORDS REFUSED (``preprocess/dropped_records.json``):

1. ``teto7_promoter_strain_reference_and_treatment_unresolved`` (6 samples: REB1, HSF1,
   ABF1, MCM1, RAP1, GCR1). These are TetO7 promoter-replacement strains, not
   deletions. Supplementary Table 1 lists them under "mig-log phase, tetracycline
   treatment", but the Methods grew them "in the presence or absence of 10 ug/ml
   doxycycline", so which culture is the sample and which the reference is not stated,
   and MCM1's own element scores X = +5.03, the opposite of a shut-off. Doxycycline also
   has no row in the compound table. Refused pending an owner decision (#1024).

NOT RELEASED: the 23 heat-shock profiles Supplementary Table 1 lists (22 deletions plus
TetO7-HSF1) are absent from GSE4654, whose 269 samples are all "mid-log growth"; the
Supplementary Table 1 rows are kept in ``preprocess/strain_table.csv``.

DATA SOURCE: ``GSE4654_family.soft.gz`` from the GEO FTP tree, and Supplementary
Table 1 (``si/si2.pdf``, the Springer ESM file) with its pypdf text layer
``si/si2.txt``, all three deposited in the raw mirror
``$DATA_ROOT/torchcell-raw/huGeneticReconstructionFunctional2007/`` with a
``manifest.json``. The loader reads the mirror and verifies every sha256; the URLs are
retrieval metadata.
"""

from __future__ import annotations

import csv
import gzip
import json
import logging
import os
import os.path as osp
import re
import shutil
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    Organism,
    check_manifest_pin,
    link_verified,
    post_process,
    verify_raw_files,
)
from torchcell.data.experiment_dataset import file_sha256, verify_sha256
from torchcell.datamodels.media import YPD_LIQUID
from torchcell.datamodels.schema import (
    Environment,
    Experiment,
    ExperimentReference,
    Genotype,
    KanMxDeletionPerturbation,
    MicroarrayExpressionScoreExperiment,
    MicroarrayExpressionScoreExperimentReference,
    MicroarrayExpressionScorePhenotype,
    Publication,
    ReferenceGenome,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.literature.manifest import (
    ROLE_RAW_DATA,
    ROLE_SI_PDF,
    ROLE_SI_TEXT,
    ArtifactRecord,
    Manifest,
    ProcessingRecord,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.sequence.genome.registry import (
    SGD_S288C_R64,
    SGD_S288C_R64_GENBANK_ACCESSION,
)
from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome
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
    audit_sourced_value,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

CITATION_KEY = "huGeneticReconstructionFunctional2007"
DOI = "10.1038/ng2012"
PMID = "17417638"
TITLE = "Genetic reconstruction of a functional transcriptional regulatory network"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"
RETRIEVED_AT = "2026-10-11"
DATASET_ROOT = "data/torchcell/microarray_score_hu2007"

GEO_SERIES = "GSE4654"
GEO_PLATFORM = "GPL3588"
PLATFORM_TITLE = "UT Austin Iyer yeast 6.4k ORF-only cDNA platform"
#: Array elements per GSE4654 sample (``!Platform_data_row_count``).
N_ARRAY_ELEMENTS = 6429
#: GSE4654 samples (``!Series_sample_id`` lines).
N_GEO_SAMPLES = 269

#: The library mirror (MinerU OCR) the verbatim quotes are cut from.
PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "74ada81539e7d599f36972a45703dd2b843552328c0b7e389b31728d639555ab"
SI_METHODS_MD = "si/si8.md"
SI_METHODS_MD_SHA256 = (
    "1e8111da0f5452354b5162ae62eecebeb36094fd3ffa72bb74a04f2bc0982f59"
)
SI_TABLE1_MD = "si/si2.md"
SI_TABLE1_MD_SHA256 = "f0c59f2b5473b6a1ffcdcd67a5228984902b8296c6c353dcc25620b0e5f848c3"

#: pypdf version that rendered ``si/si2.txt`` (every page's extract_text joined by "\n").
SI_TEXT_PYPDF_VERSION = "6.19.0"
SI_TEXT_PAGE_SEPARATOR = "\n"


class RawFile(BaseModel):
    """One file of the raw mirror: its pinned bytes, role and origin."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relpath: str
    role: str
    sha256: str
    bytes: int
    source_url: str | None
    method: RetrievalMethod | None
    retriever: str | None
    description: str


SOFT = RawFile(
    relpath="data/GSE4654_family.soft.gz",
    role=ROLE_RAW_DATA,
    sha256="0e8bff679733a659b29c0828812ea47ccafd5230d202e0a71813a5d9c05082b4",
    bytes=28122230,
    source_url=(
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE4nnn/GSE4654/soft/"
        "GSE4654_family.soft.gz"
    ),
    method=RetrievalMethod.direct_url,
    retriever="torchcell.literature.retrieve.direct_url",
    description="The GSE4654 family SOFT file: the series header, the GPL3588 platform "
    "table and all 269 sample tables (X score and p-value per array element). The "
    "only expression data the paper released",
)
SI_TABLE1_PDF = RawFile(
    relpath="si/si2.pdf",
    role=ROLE_SI_PDF,
    sha256="3b9d38d2399979fb8dcd33a041a275342eb7cc7e9c5b12044dd1c290059369ce",
    bytes=119987,
    source_url=(
        "https://static-content.springer.com/esm/art%3A10.1038%2Fng2012/"
        "MediaObjects/41588_2007_BFng2012_MOESM20_ESM.pdf"
    ),
    method=RetrievalMethod.springer_esm,
    retriever="torchcell.literature.retrieve.springer_esm",
    description="Supplementary Table 1, 'Saccharomyces cerevsiae strain information': "
    "the strain, growth condition and average harvest OD of every profile. The same "
    "bytes the literature mirror holds as si/si2.pdf",
)
SI_TABLE1_TEXT = RawFile(
    relpath="si/si2.txt",
    role=ROLE_SI_TEXT,
    sha256="22f64c0635f85220a878e99e1013b526e2e6def4b430aa364878b09a9f861a8d",
    bytes=10570,
    source_url=None,
    method=None,
    retriever=None,
    description="The text layer of si/si2.pdf rendered by pypdf (every page's "
    "extract_text joined by a newline). Supplementary Table 1 is read from it; the PDF "
    "stays the canonical artifact",
)
RAW_FILES: tuple[RawFile, ...] = (SOFT, SI_TABLE1_PDF, SI_TABLE1_TEXT)
SI_EXPECTED = [
    "GEO GSE4654 family SOFT (data/GSE4654_family.soft.gz): every released expression "
    "value, X score and p-value per array element per strain",
    "Supplementary Table 1 (si/si2.pdf, si/si2.txt): strain, growth condition and "
    "average harvest OD per profile",
    "Per-array log ratios and intensities: deposited only to the Longhorn Array Database "
    "and www.iyerlab.org/tfnetwork (now a redirect, never archived); not retrievable, no "
    "value is read from them",
    "Supplementary Tables 2-6 and Figure 1 (si1, si3-si7): derived analyses (ChIP "
    "overlap, motifs, GO enrichment, RNA-binding proteins, HSF1 overexpression); no "
    "record is built from them",
]


# --------------------------------------------------------------------------- #
# Verbatim quotes
# --------------------------------------------------------------------------- #
def _paper(
    value: Any,
    quote: str,
    *,
    page: str,
    note: str | None = None,
    uri: str = PAPER_MD,
    sha: str = PAPER_MD_SHA256,
) -> SourcedValue:
    """Bind a value to a verbatim quote in the sha256-pinned library mirror."""
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=uri,
            citation_key=CITATION_KEY,
            sha256=sha,
            method="MinerU OCR of the publisher PDF (torchcell-library mirror)",
            page=page,
        ),
    )


_Q_STRAINS = (
    "Yeast knockout strains were obtained from Open Biosystems and were derived from a "
    "BY4741 (MATa his3D1 leu2D0 met15D0 ura3D0) parent26."
)
_Q_GROWTH = (
    "For each of the 263 nonessential transcription factors, knockout strains were "
    "grown in YPD medium to mid-log phase and then harvested."
)
_Q_TETO7 = (
    "For six essential transcription factors, $T e t O _ { 7 }$ -promoter regulated "
    "mutant strains12 were grown in YPD medium in the presence or absence of $1 0 ~ "
    "\\mathrm { \\mu g / m l }$ doxycycline (Sigma) for $1 4 { - } 1 6 \\mathrm { ~ h ~ }$ "
    "and then harvested."
)
_Q_HEAT_SHOCK = (
    "For 22 of these transcription factors implicated in the heat shock response, "
    "deletion strains were grown in YPD medium to mid-log phase and then were subjected "
    "to a $1 5 \\mathrm { - m i n }$ heat shock at $3 9 ~ ^ { \\circ } \\mathrm { C }$ "
    "before harvest."
)
_Q_REPLICATES = "Each experiment was independently carried out at least twice."
_Q_WT_CONTROL = (
    "A wild-type strain was cultured together with each batch of mutants as a growth "
    "control."
)
_Q_COMMON_REFERENCE = (
    "Because we used a common reference RNA design for all microarray hybridizations, "
    "we calculated the true relative expression level of each gene in the mutant "
    "relative to the wild-type, normalizing for growth and reference batch effects29."
)
_Q_DYES = (
    "cDNA samples from wild-type controls were generally labeled with Cy3, and samples "
    "from knockout strains were generally labeled with $\\mathrm { C y } 5$ ."
)
_Q_X_SCORE = (
    "To identify significantly regulated target genes, we adapted an error model5, "
    "which generates a normalized, confidenceweighted measure of each ratio called the "
    "$X$ score (Supplementary Methods)."
)
_Q_DUPLICATE_AVERAGING = (
    "We used normalized microarray data to generate $X$ scores in MATLAB and performed "
    "weighted duplicate averaging with custom Java software."
)
_Q_P_VALUE = (
    "A $P$ value for every gene in the resulting data set was calculated in MATLAB "
    "based on the averaged $X$ scores."
)
_Q_ACCESSION = (
    "Expression data from this study have been deposited in the Gene Expression "
    "Omnibus (GEO) database under the series accession number GSE4654."
)
_Q_GROWTH_TABLE = (
    "Supplementary Table 1 contains additional information on the growth rate of all "
    "the strains used."
)
_Q_BIOLOGICAL = (
    "The repeatability of each gene's measured expression change across multiple "
    "independent biologically repeated experiments is the strongest input to the "
    "assignment of a confidence value."
)
_Q_OD_HEADER = "Average OD ofreplicates"

PARENT_STRAIN = _paper("BY4741", _Q_STRAINS, page="Methods, 'Strains'")
MEDIUM = _paper(
    "YPD_LIQUID",
    _Q_GROWTH,
    page="Methods, 'Growth conditions'",
    note="a liquid culture harvested at mid-log for RNA, so the shared liquid YPD",
)
GROWTH_PHASE = _paper(
    "mid-log phase",
    _Q_GROWTH,
    page="Methods, 'Growth conditions'",
    note="no Environment field carries a harvest phase; Supplementary Table 1's "
    "per-strain average OD at harvest is stored as harvest_optical_density",
)
N_REPLICATES = _paper(
    2,
    _Q_REPLICATES,
    page="Methods, 'Growth conditions'",
    note="the source gives a lower bound and no per-strain count, so the CLAUDE.md "
    "range rule takes the CONSERVATIVE lower end, 2. The 'weighted duplicate "
    "averaging' sentence (Methods, 'Data analysis') and the Supplementary Methods' "
    "'multiple independent biologically repeated experiments' say the replicates are "
    "independent cultures, and the release resolves no per-gene array count",
)
BIOLOGICAL_REPLICATES = _paper(
    "independent biological replicate cultures",
    _Q_BIOLOGICAL,
    page="Supplementary Methods, 'Significant Target Determination'",
    uri=SI_METHODS_MD,
    sha=SI_METHODS_MD_SHA256,
)
DUPLICATE_AVERAGING = _paper(
    "weighted duplicate averaging",
    _Q_DUPLICATE_AVERAGING,
    page="Methods, 'Data analysis'",
)
MEASUREMENT_TYPE = _paper(
    "error_model_x_score_weighted_duplicate_average",
    _Q_X_SCORE,
    page="Methods, 'Data analysis'",
    note="the Hughes 2000 error model's X score, averaged over the replicate arrays; "
    "GSE4654 labels the column '#VALUE = X score'",
)
P_VALUE = _paper(
    "two-sided normal tail of the averaged X score",
    _Q_P_VALUE,
    page="Methods, 'Data analysis'",
    note="measured on the release: max |p - 2 * (1 - Phi(|X|))| = 5.0e-7 over every "
    "released score, the six-decimal rounding of the P column",
)
REFERENCE_CULTURE = _paper(
    "wild-type culture grown with each batch",
    _Q_WT_CONTROL,
    page="Methods, 'Growth conditions'",
)
COMMON_REFERENCE = _paper(
    "common reference RNA design", _Q_COMMON_REFERENCE, page="Methods, 'Data analysis'"
)
DYE_ORIENTATION = _paper(
    {"wild_type": "Cy3", "deletion": "Cy5"},
    _Q_DYES,
    page="Methods, 'Expression profiling'",
    note="'generally', but every GSE4654 sample records ch1 = Cy5 = the deletion and "
    "ch2 = Cy3 = 'WT mid-log growth', which the loader asserts per sample",
)
GEO_ACCESSION = _paper(GEO_SERIES, _Q_ACCESSION, page="Methods, 'Accession codes'")
HARVEST_OD = _paper(
    "Average OD of replicates",
    _Q_OD_HEADER,
    page="Supplementary Table 1, column header",
    uri=SI_TABLE1_MD,
    sha=SI_TABLE1_MD_SHA256,
    note="the wavelength is not stated, so the field is an optical density, not an "
    "OD600; the paper calls the table 'additional information on the growth rate'",
)
TETO7_TREATMENT = _paper(
    "presence or absence of 10 ug/ml doxycycline",
    _Q_TETO7,
    page="Methods, 'Growth conditions'",
    note="which of the two cultures is the released sample, and whether the reference "
    "is the wild type or the untreated strain, is not stated",
)
HEAT_SHOCK = _paper(
    "15-min heat shock at 39 C",
    _Q_HEAT_SHOCK,
    page="Methods, 'Growth conditions'",
    note="the 23 heat-shock profiles are not in GSE4654",
)
SOURCED_VALUES: tuple[SourcedValue, ...] = (
    PARENT_STRAIN,
    MEDIUM,
    GROWTH_PHASE,
    N_REPLICATES,
    BIOLOGICAL_REPLICATES,
    DUPLICATE_AVERAGING,
    MEASUREMENT_TYPE,
    P_VALUE,
    REFERENCE_CULTURE,
    COMMON_REFERENCE,
    DYE_ORIENTATION,
    GEO_ACCESSION,
    HARVEST_OD,
    TETO7_TREATMENT,
    HEAT_SHOCK,
    _paper(_Q_GROWTH_TABLE, _Q_GROWTH_TABLE, page="Methods, 'Growth conditions'"),
)

TEMPERATURE_GAP = ProvenanceGap(
    field="temperature",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=Provenance(
        source_uri=PAPER_MD,
        citation_key=CITATION_KEY,
        sha256=PAPER_MD_SHA256,
        page="Methods, 'Growth conditions'; Supplementary Methods",
    ),
    note="the Methods state YPD to mid-log and, for the heat-shock arm only, a 15-min "
    "shift to 39 C; no growth temperature is given, and the Methods defer only RNA "
    "extraction, array manufacture and hybridization (refs 7 and 27), not growth",
)

REFERENCE_BASIS = (
    "the wild-type BY4741 culture grown alongside each batch of mutants, labeled Cy3 "
    "against the deletion's Cy5 (common reference RNA design)"
)
ARRAY_PLATFORM = f"GEO {GEO_PLATFORM}: {PLATFORM_TITLE} (spotted cDNA, two-color)"

#: Supplementary Table 1 condition strings, verbatim (with the table's "mig-log" typo).
CONDITION_MID_LOG = "mig-log phase"
CONDITION_TETO7 = "mig-log phase, tetracycline treatment"
CONDITION_HEAT_SHOCK = "mig-log phase, heat-shock"
CONDITION_TETO7_HEAT_SHOCK = "mig-log phase, tetracycline treatment, heat-shock"
CONDITIONS = frozenset(
    {
        CONDITION_MID_LOG,
        CONDITION_TETO7,
        CONDITION_HEAT_SHOCK,
        CONDITION_TETO7_HEAT_SHOCK,
    }
)

_SYSTEMATIC_RE = re.compile(r"^Y[A-P][LR]\d{3}[WC](?:-[A-Z])?$")
_TABLE1_ROW = re.compile(
    r"^(Y[A-P][LR]\d{3}[WC](?:-[A-Z])?) (\S*) ?"
    r"(mig-log phase(?:, [a-z -]+?)*) (\d+\.\d+)\s*$"
)


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """``DATA_ROOT`` from the environment (the mirror and the build tree live under it)."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/huGeneticReconstructionFunctional2007``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def _artifact_record(raw: RawFile, dest: Path, retrieved_at: str) -> ArtifactRecord:
    """The manifest record of one deposited file: retrieved, or derived from one."""
    if raw.source_url is None:
        return ArtifactRecord(
            path=raw.relpath,
            role=raw.role,
            bytes=dest.stat().st_size,
            sha256=raw.sha256,
            source=f"derived from {SI_TABLE1_PDF.relpath}",
            processing=ProcessingRecord(
                processor="pypdf.PdfReader(...).pages[*].extract_text",
                tool="pypdf",
                version=SI_TEXT_PYPDF_VERSION,
                params={"page_separator": SI_TEXT_PAGE_SEPARATOR},
                input_sha256=[SI_TABLE1_PDF.sha256],
            ),
        )
    if raw.method is None or raw.retriever is None:
        raise RuntimeError(f"{raw.relpath} is retrieved but names no retriever")
    return ArtifactRecord(
        path=raw.relpath,
        role=raw.role,
        bytes=dest.stat().st_size,
        sha256=raw.sha256,
        source=raw.source_url,
        original_filename=raw.source_url.rsplit("/", 1)[-1],
        retrieval=RetrievalRecord(
            method=raw.method,
            source_url=raw.source_url,
            retriever=raw.retriever,
            params={"url": raw.source_url},
            sha256=raw.sha256,
            retrieved_at=retrieved_at,
        ),
    )


def deposit_raw_mirror(
    sources: dict[str, str | Path],
    *,
    retrieved_at: str = RETRIEVED_AT,
    data_root: str | None = None,
) -> Path:
    """Write the raw mirror and its ``manifest.json`` from local copies of each file.

    ``sources`` maps every ``RAW_FILES`` relpath to a local file. Idempotent by sha256:
    a mirror file whose digest matches is left alone and a differing one raises. Every
    source is verified BEFORE anything is written, so a refusal leaves no partial
    deposit.
    """
    for raw in RAW_FILES:
        if raw.relpath not in sources:
            raise RuntimeError(f"no source given for {raw.relpath}")
        verify_sha256(sources[raw.relpath], raw.sha256)
    root = raw_mirror_dir(data_root)
    records: list[ArtifactRecord] = []
    for raw in RAW_FILES:
        dest = root / raw.relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if file_sha256(dest) != raw.sha256:
                raise RuntimeError(f"{dest} exists with a different sha256; refusing")
        else:
            shutil.copy2(sources[raw.relpath], dest)
        records.append(_artifact_record(raw, dest, retrieved_at))
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=DOI,
        title=TITLE,
        files=records,
        si_data_sources=[
            f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={GEO_SERIES}",
            str(SOFT.source_url),
            str(SI_TABLE1_PDF.source_url),
            "http://www.iyerlab.org/tfnetwork",
        ],
        si_expected=SI_EXPECTED,
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
# Readers
# --------------------------------------------------------------------------- #
class SoftSample(BaseModel):
    """One GSE4654 sample: its strain token and its released (score, p) per element.

    ``values`` holds every element with a released score; ``null_both`` counts the
    elements released null in both columns and ``null_score_p_one`` those released
    with a null score and ``p = 1``.
    """

    model_config = ConfigDict(extra="forbid")

    gsm: str
    title: str
    orf: str
    values: dict[str, tuple[float, float]]
    null_both: int
    null_score_p_one: int


class SoftRelease(BaseModel):
    """The parts of the GSE4654 family SOFT file a build reads."""

    model_config = ConfigDict(extra="forbid")

    platform_ids: list[str]
    samples: list[SoftSample]


#: Sample header lines every GSE4654 sample must carry, verbatim.
_SAMPLE_HEADER = {
    "!Sample_platform_id": GEO_PLATFORM,
    "!Sample_label_ch1": "Cy5",
    "!Sample_label_ch2": "Cy3",
    "!Sample_source_name_ch2": "WT mid-log growth",
    "!Sample_characteristics_ch1": "mid-log growth",
    "#VALUE": "X score",
    "#P": "p value",
}


def _check_sample_header(gsm: str, header: dict[str, str]) -> None:
    """Refuse a sample whose columns, channels or condition differ from the series."""
    for key, expected in _SAMPLE_HEADER.items():
        if header.get(key) != expected:
            raise RuntimeError(
                f"{gsm}: {key} is {header.get(key)!r}, expected {expected!r}"
            )
    if header.get("!Sample_data_row_count") != str(N_ARRAY_ELEMENTS):
        raise RuntimeError(f"{gsm}: row count is not {N_ARRAY_ELEMENTS}")
    title = header["!Sample_title"]
    if header.get("!Sample_source_name_ch1") != f"{title} mid-log growth":
        raise RuntimeError(f"{gsm}: ch1 source does not name the sample's own strain")


def _sample_blocks(
    lines: Iterator[str],
) -> Iterator[tuple[str, dict[str, str], list[str]]]:
    """Yield ``(gsm, header, table rows)`` per ``^SAMPLE`` block."""
    gsm: str | None = None
    header: dict[str, str] = {}
    rows: list[str] = []
    in_table = False
    for line in lines:
        line = line.rstrip("\n")
        if line.startswith("^SAMPLE = "):
            gsm, header, rows, in_table = line.split(" = ", 1)[1], {}, [], False
        elif gsm is None:
            continue
        elif line == "!sample_table_begin":
            in_table = True
        elif line == "!sample_table_end":
            yield gsm, header, rows
            gsm = None
        elif in_table:
            rows.append(line)
        elif " = " in line:
            key, value = line.split(" = ", 1)
            header.setdefault(key, value)


def read_soft(path: str | Path) -> SoftRelease:
    """Parse the GSE4654 family SOFT file into the platform ids and the 269 samples.

    Every sample is checked against the series design (:data:`_SAMPLE_HEADER`), every
    table against its ``ID_REF VALUE P`` header and the platform's element list, and a
    null score must sit beside a null p-value or ``p = 1``; anything else raises.
    """
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        text = handle.read()
    platform = text.split("!platform_table_begin\n", 1)[1].split(
        "!platform_table_end", 1
    )[0]
    platform_rows = platform.splitlines()
    if platform_rows[0].split("\t")[0] != "ID":
        raise RuntimeError("GPL3588 table does not start with its ID header")
    platform_ids = [row.split("\t", 1)[0] for row in platform_rows[1:]]
    if len(platform_ids) != N_ARRAY_ELEMENTS or len(set(platform_ids)) != len(
        platform_ids
    ):
        raise RuntimeError(f"GPL3588 holds {len(platform_ids)} ids, not unique 6429")
    samples: list[SoftSample] = []
    for gsm, header, rows in _sample_blocks(iter(text.splitlines())):
        _check_sample_header(gsm, header)
        if rows[0] != "ID_REF\tVALUE\tP":
            raise RuntimeError(f"{gsm}: table header is {rows[0]!r}")
        title = header["!Sample_title"]
        orf = title.removesuffix(" deletion").upper()
        if title != f"{title.split(' ')[0]} deletion" or not _SYSTEMATIC_RE.match(orf):
            raise RuntimeError(f"{gsm}: title {title!r} is not '<orf> deletion'")
        values: dict[str, tuple[float, float]] = {}
        null_both = null_p_one = 0
        ids: list[str] = []
        for row in rows[1:]:
            element, score, p_value = row.split("\t")
            ids.append(element)
            if score == "null":
                if p_value == "null":
                    null_both += 1
                elif p_value == "1":
                    null_p_one += 1
                else:
                    raise RuntimeError(
                        f"{gsm} {element}: null score beside p={p_value}"
                    )
                continue
            values[element] = (float(score), float(p_value))
        if sorted(ids) != sorted(platform_ids):
            raise RuntimeError(f"{gsm}: elements differ from the GPL3588 table")
        samples.append(
            SoftSample(
                gsm=gsm,
                title=title,
                orf=orf,
                values=values,
                null_both=null_both,
                null_score_p_one=null_p_one,
            )
        )
    if len(samples) != N_GEO_SAMPLES:
        raise RuntimeError(f"GSE4654 holds {len(samples)} samples, not {N_GEO_SAMPLES}")
    return SoftRelease(platform_ids=platform_ids, samples=samples)


class StrainCondition(BaseModel):
    """One Supplementary Table 1 row: a strain, a growth condition and its harvest OD."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    orf: str
    gene_name: str | None
    condition: str
    average_od: float


def read_table1(path: str | Path) -> list[StrainCondition]:
    """Parse Supplementary Table 1 from its pypdf text layer (one row per line)."""
    rows: list[StrainCondition] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not re.match(r"^Y[A-P][LR]\d{3}", line):
            continue
        match = _TABLE1_ROW.match(line)
        if match is None:
            raise RuntimeError(f"unparsed Supplementary Table 1 row: {line!r}")
        orf, gene, condition, od = match.groups()
        if condition not in CONDITIONS:
            raise RuntimeError(f"unknown Supplementary Table 1 condition {condition!r}")
        rows.append(
            StrainCondition(
                orf=orf,
                gene_name=gene or None,
                condition=condition,
                average_od=float(od),
            )
        )
    keys = [(row.orf, row.condition) for row in rows]
    if len(set(keys)) != len(keys):
        raise RuntimeError("Supplementary Table 1 repeats a (strain, condition) row")
    return rows


# --------------------------------------------------------------------------- #
# Retention bookkeeping
# --------------------------------------------------------------------------- #
class DropRule(BaseModel):
    """One retention rule, the records it removed, and the strains it removed."""

    rule: str
    description: str
    n_records: int
    items: list[str] = []


class DropLog(BaseModel):
    """Every retention rule applied to a build, in the order they were applied."""

    dataset: str
    source_records: int
    kept_records: int
    dropped_records: int
    rules: list[DropRule]
    not_released: list[str]


def phenotype_of(sample: SoftSample, od: float) -> MicroarrayExpressionScorePhenotype:
    """One strain's released scores and p-values, keyed by the released element ID."""
    scores = {e: x for e, (x, _) in sample.values.items()}
    p_values = {e: p for e, (_, p) in sample.values.items()}
    return MicroarrayExpressionScorePhenotype(
        expression_score=scores,
        expression_score_p_value=p_values,
        measurement_type=MEASUREMENT_TYPE.value,
        reference_basis=REFERENCE_BASIS,
        array_platform=ARRAY_PLATFORM,
        n_replicates=N_REPLICATES.value,
        harvest_optical_density=od,
    )


def reference_phenotype(keys: Sequence[str]) -> MicroarrayExpressionScorePhenotype:
    """The wild-type reference: the neutral score 0 for every scored element."""
    return MicroarrayExpressionScorePhenotype(
        expression_score=dict.fromkeys(keys, 0.0),
        expression_score_p_value=None,
        measurement_type=MEASUREMENT_TYPE.value,
        reference_basis=REFERENCE_BASIS,
        array_platform=ARRAY_PLATFORM,
        n_replicates=N_REPLICATES.value,
        harvest_optical_density=None,
    )


def environment() -> Environment:
    """Liquid YPD to mid-log; the growth temperature is a typed gap."""
    return Environment(
        media=YPD_LIQUID, temperature=None, provenance_gaps=[TEMPERATURE_GAP]
    )


@register_dataset
class MicroarrayScoreHu2007Dataset(ExperimentDataset):
    """Hu 2007 TF-deletion transcriptomes as released: error-model X scores + p."""

    organism = Organism.scerevisiae

    def __init__(
        self,
        root: str = DATASET_ROOT,
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset; it keys every element by its released ID, so it
        reads no genome.
        """
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return MicroarrayExpressionScoreExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return MicroarrayExpressionScoreExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The GSE4654 family SOFT file and the Supplementary Table 1 text layer."""
        return [osp.basename(SOFT.relpath), osp.basename(SI_TABLE1_TEXT.relpath)]

    def download(self) -> None:
        """Link the two mirror files into ``raw/`` after verifying their sha256 pins."""
        data_root = _data_root()
        manifest = load_manifest(data_root)
        os.makedirs(self.raw_dir, exist_ok=True)
        for raw in (SOFT, SI_TABLE1_TEXT):
            check_manifest_pin(
                raw.relpath, manifest_sha256(manifest, raw.relpath), raw.sha256
            )
            src = raw_mirror_dir(data_root) / raw.relpath
            if not src.exists():
                raise RuntimeError(f"required raw artifact missing from mirror: {src}")
            link_verified(
                src, osp.join(self.raw_dir, osp.basename(raw.relpath)), raw.sha256
            )

    @post_process
    def process(self) -> None:
        """Build one record per released deletion profile; write the LMDB and ledgers."""
        soft_name = osp.basename(SOFT.relpath)
        table_name = osp.basename(SI_TABLE1_TEXT.relpath)
        verify_raw_files(
            self.raw_dir, {soft_name: SOFT.sha256, table_name: SI_TABLE1_TEXT.sha256}
        )
        release = read_soft(osp.join(self.raw_dir, soft_name))
        table = read_table1(osp.join(self.raw_dir, table_name))
        conditions: dict[str, dict[str, StrainCondition]] = {}
        for row in table:
            conditions.setdefault(row.orf, {})[row.condition] = row
        genome_reference = ReferenceGenome(
            species="Saccharomyces cerevisiae",
            strain=PARENT_STRAIN.value,
            assembly_set=SGD_S288C_R64,
            assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
        )
        publication = Publication(
            pubmed_id=PMID,
            pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{PMID}/",
            doi=DOI,
            doi_url=f"https://doi.org/{DOI}",
        )
        env_ = environment()

        teto7: list[str] = []
        kept: list[SoftSample] = []
        for sample in release.samples:
            rows = conditions.get(sample.orf, {})
            if CONDITION_MID_LOG in rows:
                kept.append(sample)
            elif CONDITION_TETO7 in rows:
                teto7.append(sample.orf)
            else:
                raise RuntimeError(
                    f"{sample.gsm} ({sample.orf}) has no mid-log or TetO7 row in "
                    "Supplementary Table 1"
                )

        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        idx = 0
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for sample in tqdm(kept, desc="hu2007"):
                row = conditions[sample.orf][CONDITION_MID_LOG]
                phenotype = phenotype_of(sample, row.average_od)
                experiment = MicroarrayExpressionScoreExperiment(
                    dataset_name=self.name,
                    genotype=Genotype(
                        perturbations=[
                            KanMxDeletionPerturbation(
                                systematic_gene_name=sample.orf,
                                # the release's own name for the strain
                                perturbed_gene_name=row.gene_name or sample.orf,
                            )
                        ]
                    ),
                    environment=env_,
                    phenotype=phenotype,
                )
                reference = MicroarrayExpressionScoreExperimentReference(
                    dataset_name=self.name,
                    genome_reference=genome_reference,
                    environment_reference=env_.model_copy(),
                    phenotype_reference=reference_phenotype(
                        list(phenotype.expression_score)
                    ),
                )
                txn.put(
                    f"{idx}".encode(),
                    self._intern_record(experiment, reference, publication, itxn),
                )
                idx += 1
        env.close()
        interned_env.close()
        self._write_ledgers(release, table, kept, teto7, idx)

    def _write_ledgers(
        self,
        release: SoftRelease,
        table: list[StrainCondition],
        kept: list[SoftSample],
        teto7: list[str],
        n_written: int,
    ) -> None:
        """The drop log, the element census and the strain table, beside the LMDB."""
        released = {sample.orf for sample in release.samples}
        not_released = sorted(
            f"{row.orf} ({row.condition})"
            for row in table
            if row.condition in (CONDITION_HEAT_SHOCK, CONDITION_TETO7_HEAT_SHOCK)
        )
        rules = [
            DropRule(
                rule="teto7_promoter_strain_reference_and_treatment_unresolved",
                description=(
                    "a TetO7 promoter-replacement strain (Supplementary Table 1: "
                    f"{CONDITION_TETO7!r}), not a deletion. The Methods grew these "
                    "strains 'in the presence or absence of 10 ug/ml doxycycline', so "
                    "which culture is the released sample and whether the reference is "
                    "the wild type or the untreated strain is not stated; MCM1's own "
                    "element scores X = +5.03, the opposite of a shut-off, and "
                    "doxycycline has no compound-table row. Refused pending an owner "
                    "decision (#1024)"
                ),
                n_records=len(teto7),
                items=sorted(teto7),
            )
        ]
        drop_log = DropLog(
            dataset=self.name,
            source_records=len(release.samples),
            kept_records=n_written,
            dropped_records=len(release.samples) - n_written,
            rules=rules,
            not_released=not_released,
        )
        if sum(rule.n_records for rule in rules) != drop_log.dropped_records:
            raise RuntimeError("drop accounting does not cover every unwritten sample")
        with open(osp.join(self.preprocess_dir, "dropped_records.json"), "w") as handle:
            handle.write(drop_log.model_dump_json(indent=2))
        element_log = {
            "n_elements": len(release.platform_ids),
            "n_keyed": len({e for s in kept for e in s.values}),
            "n_stored_values": sum(len(s.values) for s in kept),
            "released_null_both": sum(s.null_both for s in kept),
            "released_null_score_p_one": sum(s.null_score_p_one for s in kept),
        }
        with open(osp.join(self.preprocess_dir, "array_elements.json"), "w") as handle:
            json.dump(element_log, handle, indent=2)
        with open(
            osp.join(self.preprocess_dir, "strain_table.csv"), "w", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["orf", "gene_name", "condition", "average_od", "in_gse4654"]
            )
            for row in table:
                in_release = row.orf in released and row.condition in (
                    CONDITION_MID_LOG,
                    CONDITION_TETO7,
                )
                writer.writerow(
                    [
                        row.orf,
                        row.gene_name or "",
                        row.condition,
                        row.average_od,
                        in_release,
                    ]
                )
        log.info(
            "Hu2007: wrote %d records (%d TetO7 refused); %d of %d elements keyed",
            n_written,
            len(teto7),
            element_log["n_keyed"],
            len(release.platform_ids),
        )

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
#: Expected records: the 263 Supplementary Table 1 "mig-log phase" strains GSE4654
#: carries, the 269 samples less the 6 TetO7 strains.
EXPECTED_RECORDS = 263
#: The release's p column has six decimals, so a recomputed tail differs by <= 5e-7.
TWO_SIDED_NORMAL_TOL = 5e-7 + 1e-9
#: One-sided Mann-Whitney level for the matched-vs-mismatched Kemmeren comparison.
KEMMEREN_ALPHA = 1e-3
KEMMEREN_ROOT = "data/torchcell/microarray_kemmeren2014"
VERIFICATION_PROVENANCE = Provenance(
    source_uri=f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={GEO_SERIES}",
    citation_key=CITATION_KEY,
    sha256=SOFT.sha256,
    method=f"GSE4654 family SOFT ({GEO_PLATFORM}): released error-model X score and "
    "p-value per element, keyed by the element's released GPL3588 ID",
    page="Nat Genet 39:683, Methods; Supplementary Table 1",
    retrieved=RETRIEVED_AT,
)


def _l2_released_fidelity(
    records: Sequence[dict[str, Any]], release: SoftRelease
) -> LevelResult:
    """L2: every stored (score, p) is the released pair, and no keyed one is missing."""
    by_orf = {sample.orf: sample for sample in release.samples}
    mismatched: list[dict[str, Any]] = []
    n_values = 0
    for record in records:
        orf = record["experiment"]["genotype"]["perturbations"][0][
            "systematic_gene_name"
        ]
        phenotype = record["experiment"]["phenotype"]
        expected = dict(by_orf[orf].values)
        stored = {
            gene: (score, phenotype["expression_score_p_value"][gene])
            for gene, score in phenotype["expression_score"].items()
        }
        n_values += len(stored)
        if stored != expected:
            mismatched.append(
                {
                    "strain": orf,
                    "n_stored": len(stored),
                    "n_released": len(expected),
                    "n_differ": sum(
                        1
                        for g in set(stored) | set(expected)
                        if stored.get(g) != expected.get(g)
                    ),
                }
            )
    return LevelResult(
        level=Level.L2,
        name="released_value_fidelity",
        passed=not mismatched,
        message=(
            f"{n_values} stored (score, p) pairs equal the GSE4654 release exactly"
            if not mismatched
            else f"{len(mismatched)} records differ from the GSE4654 release"
        ),
        details={"n_values": n_values, "mismatched": mismatched[:20]},
    )


def _l3_provenance_audit(library_root: str) -> LevelResult:
    """L3: every sourced quote is still verbatim in its sha256-pinned mirror file."""
    results = [audit_sourced_value(sv, library_root) for sv in SOURCED_VALUES]
    failed = [r.details for r in results if not r.passed]
    return LevelResult(
        level=Level.L3,
        name="sourced_value_audit",
        passed=not failed,
        message=(
            f"{len(results)} sourced values backed by verbatim quotes"
            if not failed
            else f"{len(failed)}/{len(results)} sourced values failed the audit"
        ),
        details={"n_sourced": len(results), "failed": failed},
    )


def _l4_sgd_containment(
    records: Sequence[dict[str, Any]], genome: SCerevisiaeGenome
) -> LevelResult:
    """L4 finding: which deleted ORFs are R64 loci by their own id.

    A locus is a gene or a non-gene locus such as FLO8's blocked reading frame; the
    non-gene ones and any ORF R64 does not carry are named in the details. A FINDING,
    never a gate (owner rule, 2026-10-12: the genome is a separate layer and does not
    fail a store), so the row always passes and the message carries the count.
    """
    deleted = {
        record["experiment"]["genotype"]["perturbations"][0]["systematic_gene_name"]
        for record in records
    }
    genes = set(genome.gene_set)
    loci = genes | set(genome.feature_index["locus_type"])
    missing = sorted(deleted - loci)
    non_gene = sorted((deleted & loci) - genes)
    return LevelResult(
        level=Level.L4,
        name="deleted_gene_containment_sgd",
        passed=True,
        message=f"finding: {len(deleted) - len(missing)} of {len(deleted)} deleted ORFs "
        f"are R64 loci ({len(non_gene)} of them non-gene loci, {len(missing)} not in "
        "R64)",
        details={"n_deleted": len(deleted), "missing": missing, "non_gene": non_gene},
    )


def verify_build(
    dataset_root: str,
    data_root: str | None = None,
    *,
    genome: SCerevisiaeGenome | None = None,
) -> VerificationReport:
    """L0-L4 for a built store, re-reading the released bytes from the raw mirror.

    L0-L3 are the family gate (:mod:`torchcell.verification.expression_score`) plus
    the exact released-value comparison and the sourced-quote audit; L4 is the SGD
    containment finding (reported, never failing) and the matched-strain agreement
    with the Kemmeren 2014 dev store.
    """
    from torchcell.verification.expression_score import (
        l4_matched_profile_agreement,
        verify_expression_score_dataset,
    )
    from torchcell.verification.runners import load_records

    data_root = data_root or _data_root()
    if genome is None:
        genome = SCerevisiaeGenome(
            genome_root=osp.join(data_root, "data/sgd/genome"),
            go_root=osp.join(data_root, "data/go"),
            overwrite=False,
        )
    records = load_records(dataset_root)
    report = verify_expression_score_dataset(
        records,
        dataset_name="microarray_score_hu2007",
        provenance=VERIFICATION_PROVENANCE,
        expected_count=EXPECTED_RECORDS,
        two_sided_normal_tol=TWO_SIDED_NORMAL_TOL,
    )
    mirror = raw_mirror_dir(data_root)
    verify_sha256(mirror / SOFT.relpath, SOFT.sha256)
    release = read_soft(mirror / SOFT.relpath)
    report.add(_l2_released_fidelity(records, release))
    report.add(_l3_provenance_audit(osp.join(data_root, "torchcell-library")))
    report.add(_l4_sgd_containment(records, genome))
    report.add(
        l4_matched_profile_agreement(
            records,
            load_records(osp.join(data_root, KEMMEREN_ROOT)),
            other_name="microarray_kemmeren2014",
            other_field="expression_log2_ratio",
            alpha=KEMMEREN_ALPHA,
        )
    )
    return report


def main() -> None:
    """Build the dataset and print its first record and drop log."""
    from dotenv import load_dotenv

    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    root = osp.join(data_root, DATASET_ROOT)
    dataset = MicroarrayScoreHu2007Dataset(root=root)
    print(f"len = {len(dataset)}")
    print(dataset[0])
    print(Path(osp.join(root, "preprocess/dropped_records.json")).read_text())


if __name__ == "__main__":
    main()
