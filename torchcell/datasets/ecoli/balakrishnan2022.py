# torchcell/datasets/ecoli/balakrishnan2022
# [[torchcell.datasets.ecoli.balakrishnan2022]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/balakrishnan2022
# Test file: tests/torchcell/datasets/ecoli/test_balakrishnan2022.py
"""Balakrishnan 2022, the steady-state E. coli transcriptome as mRNA number fractions.

Balakrishnan, Mori, Segota, Zhang, Aebersold, Ludwig and Hwa 2022 (Science 378,
eabk2066, doi:10.1126/science.abk2066, PMID 36480614, PMC9804519) measured the
transcriptome of E. coli NCM3722 and three NCM3722 derivatives across three
growth-limitation series. :class:`MrnaFractionBalakrishnan2022Dataset` serves one
``MrnaNumberFractionExperiment`` per released, identifiable sample column of Table S3
sheet ``2 - RNAseq-ss (fractions)``.

THE STORED QUANTITY. The paper defines it: "RNA-sequencing was used to determine the
mRNA number fractions psi_m,i = [mR_i]/[mR] for the corresponding mRNAs, with [mR] =
sum_i [mR_i] being the total mRNA concentration". It is dimensionless; every released
column sums to 1 within ``NORMALIZATION_ATOL`` (``check_normalization`` asserts it on
the bytes). No read count is released anywhere (GEO GSE205717 holds the raw FASTQ and
this table only), which is why this is the first consumer of
``MrnaNumberFractionPhenotype`` (issue #854) and not of ``RNASeqExpressionPhenotype``.

TWENTY-EIGHT RECORDS FROM TWENTY-NINE DESCRIBED SAMPLES. The description sheet lists 29
samples (C-limitation 10, A-limitation 8, R-limitation 11). The fractions sheet has 29
sample columns, but ``a4_1`` heads two of them (columns 16 and 17) and the two are
identical in every one of the 4,342 rows, while the described sample ``a3_1`` heads
none. So:

- ``DROP_DUPLICATE_HEADER``: the second ``a4_1`` column is the same bytes under the same
  header; it is not a second measurement, so it is dropped and the first is stored as
  ``a4_1`` (the release's own label for both).
- ``a3_1`` is described and NOT released: no column carries it. It is recorded in the
  ledger as unreleased, never reconstructed.

29 columns - 1 duplicate = 28 records. (Issue #854 states 27; that is 29 - 2, which
would also drop the ``a4_1`` data the release labels. The release labels it ``a4_1``
twice and the data cannot be shown to belong to another sample: its log10 Pearson r
against ``a4`` is 0.9888 and against ``a3`` 0.9882, measured on the pinned bytes, so the
header stands.)

THE GENE KEYS. 4,342 released rows = 4,176 stored keys + 148 refused b-numbers + 18
rows whose released locus is ``0`` (the IS-element rows ``insAB-*``, ``insCD-*``,
``insEF-*``, ``insJK``). The 148 are released b-numbers that name no gene of the pinned
MG1655 assembly's gene set, measured through the genome's own resolver in three groups:
74 are locus tags of a non-gene feature (pseudogene loci), 69 are released
split-fragment rows (``arpB_1``, ``gapC_2``, ...) whose b-number is a synonym of such a
feature, and 5 are retired b-numbers (``b3036``, ``b3776``, ``b4223``, ``b4274``,
``b4590``). Every stored record carries all 4,176 keys; a released ``0.0`` is stored as
``0.0``, because the column sums to 1 over its zero rows too (it is a library with no
reads on that gene, not a missing value).

STRAINS AND GENOTYPES. Every record is written against the MG1655 GenBank assembly
with an ``NCM3722`` background (no NCM3722 assembly is in the genomes tier; the Gupta
2024 precedent). The released strain column names NCM3722, NQ1243, NQ1390 and NQ393,
and its description cells name the edits: "Titratable glucose uptake (Pu-ptsG)" and
"Titratable ammonia assimilation (Plac-GOGAT)". Table S1 (the strain table) is behind
the proof-of-work page (#853), so the construction is quoted from the same lab's
mirrored Mori 2021 Appendix, which describes these same strains:

- NQ1243 and NQ1390: ``PromoterReplacementPerturbation`` of ``ptsG`` (b1101) by the
  inducible ``Pu`` promoter, expression decreased. The xylR activator each carries
  (driven by Ptet in NQ1243 and by a lacIq promoter in NQ1390) is NOT typed: no mirrored
  source states its location or integration site (:data:`UNTYPED_EDITS`). The two
  strains therefore share one typed genotype and differ only in that untyped driver.
- NQ393: ``PromoterReplacementPerturbation`` of ``gltB`` (b3212, the first gene of the
  GOGAT operon) by ``Plac``, expression decreased ("low GOGAT expression"). Its
  "GDH-null background" is NOT typed: the allele (deletion or point lesion) of ``gdhA``
  is not stated in any mirrored source.

THE ENVIRONMENT. The release names the base medium ("M9", "MOPS"), the carbon source
("0.2% glucose"), the nitrogen source ("11.34 mM (NH4)2SO4", "10 mM NH4Cl") and the
supplement (3MBA, IPTG, chloramphenicol, in uM). The base-medium recipes and the culture
temperature are in the walled SI Methods, so each medium is a loader-local object whose
base composition is a deferral (``composition_deferred``) beside the stated nitrogen
salt, and the temperature is a ``ProvenanceGap``. Three nitrogen cells read
"(NH4)2SO5", "SO6" and "SO7" on the consecutive rows ``a2_1``, ``a3_1``, ``a4_1``
(:data:`NITROGEN_FILL_SERIES`); no such salts exist, the digit increments by one down
consecutive rows from the "(NH4)2SO4" above them, and every other M9 row reads
"(NH4)2SO4", so the two loaded rows are read as "(NH4)2SO4" and the verbatim cells stay
in the ledger. "3MBA" is stored under that name: no mirrored source expands it.

THE REFERENCE. "For E. coli K-12 strain NCM3722 growing exponentially in glucose minimal
medium (reference condition, growth rate 0.91/h)". Each record's reference is its base
medium's wild-type condition: ``c5`` and ``c0_1`` (NCM3722 in M9) for the C- and
A-limitation series, ``r0`` and ``r0_1`` (NCM3722 in MOPS) for the R-limitation series,
stored as the per-gene mean of the two libraries with ``n_libraries=2``.

DATA. Two files are consumed from this paper and deposited under
``$DATA_ROOT/torchcell-raw/balakrishnanPrinciplesGeneRegulation2022/data/``: the GEO
``GSE205717_Processed_data_Table_S3.xlsx`` (direct URL) and the author-manuscript text
``PMC9804519.1.txt`` (PMC Article Datasets bucket). The Mori 2021 Appendix is read from
its own raw mirror and checked against its own pin.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import os.path as osp
import pickle
import shutil
import statistics
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import openpyxl
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm

from torchcell.data import (
    ExperimentDataset,
    check_manifest_pin,
    link_verified,
    post_process,
    verify_raw_files,
    write_verified,
)
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.schema import (
    BacterialAssemblySet,
    BacterialGeneNamespace,
    BacterialStrainBackground,
    ComponentDefinition,
    Compound,
    Concentration,
    ConcentrationUnit,
    DerivedIdentifierMapping,
    Environment,
    EnvironmentPerturbationType,
    EnvironmentPhysicalPerturbation,
    Experiment,
    ExperimentReference,
    Genotype,
    Media,
    MediaComponent,
    MediaComponentRole,
    MrnaNumberFractionExperiment,
    MrnaNumberFractionExperimentReference,
    MrnaNumberFractionPhenotype,
    PhysicalFactor,
    PromoterReplacementPerturbation,
    Publication,
    SmallMoleculePerturbation,
)
from torchcell.datasets.bacteria_common import (
    BACTERIAL_ASSEMBLY_SETS,
    assembly_reference,
    bacterial_genome,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.datasets.ecoli import mori2021
from torchcell.literature.manifest import (
    ROLE_RAW_DATA,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.literature.provenance import run_retriever
from torchcell.sequence import GeneSet
from torchcell.sequence.genome.ecoli.k12 import EcoliK12Genome, EcoliK12StrainName
from torchcell.verification.report import Provenance, VerificationReport
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
)

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# The pinned artifacts
# --------------------------------------------------------------------------- #
CITATION_KEY = "balakrishnanPrinciplesGeneRegulation2022"
PAPER_DOI = "10.1126/science.abk2066"
PAPER_TITLE = (
    "Principles of gene regulation quantitatively connect DNA to RNA and proteins in "
    "bacteria"
)
PUBMED_ID = "36480614"
PMC_ID = "PMC9804519"
GEO_SERIES = "GSE205717"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"

TABLE_S3 = "GSE205717_Processed_data_Table_S3.xlsx"
PAPER_TXT = "PMC9804519.1.txt"
TABLE_S3_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE205nnn/GSE205717/suppl/" + TABLE_S3
)
PAPER_TXT_KEY = f"{PMC_ID}.1/{PAPER_TXT}"

#: ``{raw file name: pinned sha256}`` of every consumed file of this paper.
DATA_SHA256: dict[str, str] = {
    TABLE_S3: "9d7df03bbbcdf1f92e93027014e746da5bcf42bf4d6b4be9061dd1803f71cc20",
    PAPER_TXT: "8f7c13f6c2c491abf95bace26bcdbf7927cef3ae4c276e43c4a291c108f824e9",
}
MIRROR_RELPATH: dict[str, str] = {name: f"data/{name}" for name in DATA_SHA256}
RETRIEVED_AT = "2026-10-10T00:00:00+00:00"
RETRIEVALS: dict[str, RetrievalRecord] = {
    TABLE_S3: RetrievalRecord(
        method=RetrievalMethod.direct_url,
        source_url=TABLE_S3_URL,
        retriever="torchcell.literature.retrieve.direct_url",
        params={"url": TABLE_S3_URL},
        sha256=DATA_SHA256[TABLE_S3],
        retrieved_at=RETRIEVED_AT,
    ),
    PAPER_TXT: RetrievalRecord(
        method=RetrievalMethod.pmc_cloud,
        source_url=f"https://pmc-oa-opendata.s3.amazonaws.com/{PAPER_TXT_KEY}",
        retriever="torchcell.literature.retrieve.pmc_cloud_object",
        params={"key": PAPER_TXT_KEY},
        sha256=DATA_SHA256[PAPER_TXT],
        retrieved_at=RETRIEVED_AT,
    ),
}
#: The sister paper's Appendix, read from ITS raw mirror under ITS pin.
MORI_APPENDIX = "mori2021_si1.docx"
MORI_APPENDIX_SHA256 = mori2021.DATA_SHA256[mori2021.APPENDIX]
MORI_APPENDIX_RELPATH = mori2021.MIRROR_RELPATH[mori2021.APPENDIX]
RAW_SHA256: dict[str, str] = {**DATA_SHA256, MORI_APPENDIX: MORI_APPENDIX_SHA256}

#: Released locations this loader does NOT mirror, with why.
NOT_MIRRORED = (
    "Supplementary Information PDF and Tables S4-S7: behind the PMC proof-of-work page "
    "(issue #853); no value is read from them",
    "Table S3 sheets '3 - RNAseq-deg (description)' and '4 - RNAseq-deg (fractions)': "
    "rifampicin decay time courses whose derived rates are Table S6 (#853); fitting "
    "them here would store a derivation, not the release",
    "GEO GSE205717 raw FASTQ: no loader re-quantifies reads",
)

# --------------------------------------------------------------------------- #
# Verbatim quotes, each a substring of the pinned bytes (``check_quotes``)
# --------------------------------------------------------------------------- #
_Q_PSI = (
    "RNA-sequencing was used to determine the mRNA number fractions ψm,i ≡ [mRi]/[mR] "
    "for the corresponding mRNAs, with [mR] ≡ ∑i[mRi] being the total mRNA "
    "concentration; see SI Methods."
)
_Q_REFERENCE = (
    "For E. coli K-12 strain NCM3722 growing exponentially in glucose minimal medium "
    "(reference condition, growth rate 0.91/h)"
)
_Q_FOLD_REFERENCE = (
    "The fold-change was calculated between the “reference condition” (WT cells grown "
    "in glucose minimal medium) and one with ~3x slower growth, for each of the three "
    "types of growth limitation imposed."
)
_Q_METHODS_IN_SM = (
    "Experimental methods for cellular growth, RNA sequencing, and quantification of "
    "mRNA abundance, synthesis fluxes and degradation rates, as well as numerical and "
    "statistical methods, are reported in the Supplementary Material."
)
_Q_GEO = (
    "Raw RNA sequencing reads are uploaded to Gene Expression Omnibus (GEO accession "
    "number GSE205717)."
)
_Q_STRAIN_TABLE = (
    "See Tables S1 and S2 for list of strains and conditions in this study, and Table "
    "S3–4 for transcriptomics and proteomics data."
)
_Q_MORI_DERIVATIVES = (
    "A few derivatives of NCM3722 were also used: NQ1243 and NQ1390, which allow to "
    "titrate the glucose intake flux (Basan et al, 2015a); NQ393, which expresses GOGAT "
    "from a titratable promoter in GDH-null background (Hui et al., 2015);"
)
_Q_MORI_PU_PTSG = (
    "This is done by replacing the ptsG promoter with a titratable Pu promoter from "
    "Pseudomonas putida; the activity of the Pu promoter is regulated by xylR upon "
    "induction by 3MBA."
)
_Q_MORI_XYLR = (
    "Compared to NQ1243, strain NQ1390 differs in the promoter driving the xylR gene: "
    "expression of xylR is driven by a lacIq promoter in strain NQ1390, as opposed to a "
    "Ptet promoter in NQ1243."
)
_Q_MORI_LOWER = (
    "This allows to titrate PtsG (and hence the growth rate) to lower levels in NQ1390 "
    "compared to NQ1243."
)
_Q_MORI_NQ393 = (
    "the pre-culture of NQ393 (an NCM3722 derivative with low GOGAT expression in "
    "GDH-null background"
)
#: Description-sheet cells, verbatim.
_Q_PU_PTSG = "Titratable glucose uptake (Pu-ptsG)"
_Q_PLAC_GOGAT = "Titratable ammonia assimilation (Plac-GOGAT)"

PAPER_QUOTES: dict[str, str] = {
    "psi": _Q_PSI,
    "reference": _Q_REFERENCE,
    "fold_reference": _Q_FOLD_REFERENCE,
    "methods_in_supplement": _Q_METHODS_IN_SM,
    "geo": _Q_GEO,
    "strain_table": _Q_STRAIN_TABLE,
}
MORI_QUOTES: dict[str, str] = {
    "derivatives": _Q_MORI_DERIVATIVES,
    "pu_ptsg": _Q_MORI_PU_PTSG,
    "xylr": _Q_MORI_XYLR,
    "lower": _Q_MORI_LOWER,
    "nq393": _Q_MORI_NQ393,
}
WORKBOOK_QUOTES: dict[str, str] = {"pu_ptsg": _Q_PU_PTSG, "plac_gogat": _Q_PLAC_GOGAT}

PAPER = Provenance(
    source_uri=MIRROR_RELPATH[PAPER_TXT],
    citation_key=CITATION_KEY,
    sha256=DATA_SHA256[PAPER_TXT],
)
TABLE_S3_SOURCE = Provenance(
    source_uri=MIRROR_RELPATH[TABLE_S3],
    citation_key=CITATION_KEY,
    sha256=DATA_SHA256[TABLE_S3],
)
MORI_APPENDIX_SOURCE = Provenance(
    source_uri=MORI_APPENDIX_RELPATH,
    citation_key=mori2021.CITATION_KEY,
    sha256=MORI_APPENDIX_SHA256,
)


def _paper(value: object, quote: str, *, note: str | None = None) -> SourcedValue:
    """A ``SourcedValue`` quoting the pinned paper text."""
    return SourcedValue(value=value, provenance=PAPER, quote=quote, note=note)


def _table(value: object, quote: str, *, note: str | None = None) -> SourcedValue:
    """A ``SourcedValue`` quoting a cell or row rendering of the pinned Table S3."""
    return SourcedValue(value=value, provenance=TABLE_S3_SOURCE, quote=quote, note=note)


def _mori(value: object, quote: str, *, note: str | None = None) -> SourcedValue:
    """A ``SourcedValue`` quoting the pinned Mori 2021 Appendix."""
    return SourcedValue(
        value=value, provenance=MORI_APPENDIX_SOURCE, quote=quote, note=note
    )


REFERENCE_STRAIN: EcoliK12StrainName = "MG1655"
ASSEMBLY_SET: BacterialAssemblySet = "ecoli_K12_MG1655_ASM584v2"
GENE_NAMESPACE: BacterialGeneNamespace = "ecoli_k12_mg1655_bnumber"
MEASURED_STRAIN = "NCM3722"
MEASUREMENT_TYPE = "rnaseq_mrna_number_fraction"

SOURCED_VALUES: dict[str, SourcedValue] = {
    "stored_quantity": _paper("mRNA number fraction", _Q_PSI),
    "measured_strain": _paper(MEASURED_STRAIN, _Q_REFERENCE),
    "reference_condition": _paper(
        "NCM3722 in glucose minimal medium",
        _Q_REFERENCE,
        note="the M9 wild-type pair c5 / c0_1 for the C- and A-limitation series and the "
        "MOPS wild-type pair r0 / r0_1 for the R-limitation series",
    ),
    "fold_change_reference": _paper(
        "WT cells grown in glucose minimal medium", _Q_FOLD_REFERENCE
    ),
    "methods_in_supplement": _paper(
        None,
        _Q_METHODS_IN_SM,
        note="why the media recipes, the temperature and the genotypes are not quoted "
        "from this paper: they are in the Supplementary Material behind #853",
    ),
    "geo": _paper(GEO_SERIES, _Q_GEO),
    "strain_table": _paper("Tables S1 and S2", _Q_STRAIN_TABLE),
    "pu_ptsg": _table("Pu-ptsG", _Q_PU_PTSG),
    "plac_gogat": _table("Plac-GOGAT", _Q_PLAC_GOGAT),
    "derivatives": _mori(["NQ1243", "NQ1390", "NQ393"], _Q_MORI_DERIVATIVES),
    "pu_replaces_ptsg_promoter": _mori("Pu", _Q_MORI_PU_PTSG),
    "xylr_driver": _mori(
        {"NQ1243": "Ptet", "NQ1390": "lacIq promoter"},
        _Q_MORI_XYLR,
        note="not typed: no mirrored source states where the xylR cassette sits",
    ),
    "ptsg_decreased": _mori(
        "decreased",
        _Q_MORI_LOWER,
        note="every released NQ1243 / NQ1390 sample grows slower (0.30 to 0.78 /h) than "
        "the wild-type reference (0.91, 0.95 /h), measured on Table S3",
    ),
    "gogat_decreased": _mori("decreased", _Q_MORI_NQ393),
}

# --------------------------------------------------------------------------- #
# Workbook geometry and the declared samples
# --------------------------------------------------------------------------- #
SHEET_DESCRIPTION = "1 - RNAseq-ss (description)"
SHEET_FRACTIONS = "2 - RNAseq-ss (fractions)"
IDENTITY_COLUMNS = ("gene", "locus", "gene length (nt)")
N_IDENTITY_COLUMNS = len(IDENTITY_COLUMNS)
DESCRIPTION_HEADER = (
    "Sample ID",
    "Group",
    "Growth rate (1/h)",
    "Strain",
    "Growth medium",
    "Carbon source",
    "Nitrogen source",
    "Supplement",
    "Description",
    "File name",
)
#: Measured: every released column sums to 1 within this absolute tolerance.
NORMALIZATION_ATOL = 1e-7
EXPECTED_SOURCE_ROWS = 4342
EXPECTED_GENE_KEYS = 4176
EXPECTED_ZERO_LOCUS_ROWS = 18
EXPECTED_REFUSED_B_NUMBERS = 148
#: The released nitrogen cells that are a spreadsheet fill-series, verbatim.
NITROGEN_FILL_SERIES: dict[str, str] = {
    "a2_1": "11.34 mM (NH4)2SO5",
    "a3_1": "11.34 mM (NH4)2SO6",
    "a4_1": "11.34 mM (NH4)2SO7",
}
M9_NITROGEN = "11.34 mM (NH4)2SO4"
MOPS_NITROGEN = "10 mM NH4Cl"
CARBON_SOURCE = "0.2% glucose"


class DropReason(BaseModel):
    """Why one described sample or released column is not a record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rule: str
    description: str


DROP_DUPLICATE_HEADER = DropReason(
    rule="duplicate_column_under_the_same_header",
    description="the release heads two columns 'a4_1' and they are identical in every "
    "row; the second is the same bytes, not a second library",
)
DROP_UNRELEASED = DropReason(
    rule="described_sample_has_no_released_column",
    description="the description sheet lists 'a3_1' and no fractions column carries it",
)


class SampleSpec(BaseModel):
    """One described sample: its released metadata and what became of it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample: str
    series: str = Field(description="C-limitation, A-limitation or R-limitation")
    strain: str
    medium: str
    supplement: str | None
    growth_rate_per_h: float
    drop: DropReason | None = None


def _s(
    sample: str,
    series: str,
    rate: float,
    strain: str,
    medium: str,
    supplement: str | None = None,
    drop: DropReason | None = None,
) -> SampleSpec:
    return SampleSpec(
        sample=sample,
        series=series,
        strain=strain,
        medium=medium,
        supplement=supplement,
        growth_rate_per_h=rate,
        drop=drop,
    )


_C, _A, _R = "C-limitation", "A-limitation", "R-limitation"
#: The description sheet, in its own row order. ``check_sample_metadata`` re-reads it.
SAMPLES: tuple[SampleSpec, ...] = (
    _s("c5", _C, 0.91, "NCM3722", "M9"),
    _s("c1", _C, 0.75, "NQ1243", "M9", "300 µM 3MBA"),
    _s("c2", _C, 0.56, "NQ1243", "M9"),
    _s("c3", _C, 0.51, "NQ1390", "M9", "400 µM 3MBA"),
    _s("c4", _C, 0.3, "NQ1390", "M9", "40 µM 3MBA"),
    _s("c0_1", _C, 0.95, "NCM3722", "M9"),
    _s("c1_1", _C, 0.78, "NQ1243", "M9", "300 µM 3MBA"),
    _s("c2_1", _C, 0.59, "NQ1243", "M9"),
    _s("c3_1", _C, 0.46, "NQ1390", "M9", "400 µM 3MBA"),
    _s("c4_1", _C, 0.39, "NQ1390", "M9", "40 µM 3MBA"),
    _s("a1", _A, 0.69, "NQ393", "M9", "60 µM IPTG"),
    _s("a2", _A, 0.51, "NQ393", "M9", "40 µM IPTG"),
    _s("a3", _A, 0.33, "NQ393", "M9", "28 µM IPTG"),
    _s("a4", _A, 0.24, "NQ393", "M9", "20 µM IPTG"),
    _s("a1_1", _A, 0.76, "NQ393", "M9", "60 µM IPTG"),
    _s("a2_1", _A, 0.55, "NQ393", "M9", "40 µM IPTG"),
    _s("a3_1", _A, 0.45, "NQ393", "M9", "30 µM IPTG", DROP_UNRELEASED),
    _s("a4_1", _A, 0.34, "NQ393", "M9", "20 µM IPTG"),
    _s("r0", _R, 0.89, "NCM3722", "MOPS"),
    _s("r1", _R, 0.71, "NCM3722", "MOPS", "2 µM chloramphenicol"),
    _s("r2", _R, 0.53, "NCM3722", "MOPS", "4 µM chloramphenicol"),
    _s("r3", _R, 0.45, "NCM3722", "MOPS", "6 µM chloramphenicol"),
    _s("r4", _R, 0.4, "NCM3722", "MOPS", "8 µM chloramphenicol"),
    _s("r5", _R, 0.38, "NCM3722", "MOPS", "9 µM chloramphenicol"),
    _s("r0_1", _R, 0.91, "NCM3722", "MOPS"),
    _s("r1_1", _R, 0.61, "NCM3722", "MOPS", "2 µM chloramphenicol"),
    _s("r2_1", _R, 0.44, "NCM3722", "MOPS", "4 µM chloramphenicol"),
    _s("r3_1", _R, 0.35, "NCM3722", "MOPS", "6 µM chloramphenicol"),
    _s("r4_1", _R, 0.28, "NCM3722", "MOPS", "8 µM chloramphenicol"),
)
LOADED = tuple(spec for spec in SAMPLES if spec.drop is None)
#: The released fractions headers, in sheet order (``a4_1`` twice, ``a3_1`` absent).
RELEASED_HEADERS: tuple[str, ...] = (
    "c4", "c4_1", "c3_1", "c3", "c2", "c2_1", "c1", "c1_1", "c5", "c0_1",
    "a4", "a3", "a4_1", "a4_1", "a2", "a2_1", "a1", "a1_1",
    "r4_1", "r3_1", "r5", "r4", "r2_1", "r3", "r2", "r1_1", "r1", "r0", "r0_1",
)  # fmt: skip
#: 29 released columns - 1 duplicate = 28.
EXPECTED_RECORDS = 28
#: The wild-type reference libraries of each base medium.
REFERENCE_SAMPLES: dict[str, tuple[str, str]] = {
    "M9": ("c5", "c0_1"),
    "MOPS": ("r0", "r0_1"),
}

# --------------------------------------------------------------------------- #
# Genotype
# --------------------------------------------------------------------------- #
_LOOKED_IN_PAPER = Provenance(
    source_uri=MIRROR_RELPATH[PAPER_TXT],
    citation_key=CITATION_KEY,
    sha256=DATA_SHA256[PAPER_TXT],
    method="PMC Article Datasets author-manuscript text",
    page="Materials and Methods (deferred to the Supplementary Material)",
)
NCM3722_GENOTYPE_GAP = ProvenanceGap(
    field="genotype_statement",
    reason=ProvenanceGapReason.deferred_pending_source_review,
    looked_in=_LOOKED_IN_PAPER,
    note="the main text names 'E. coli K-12 strain NCM3722' and defers the strain "
    "table to Table S1, which is behind the PMC proof-of-work page (#853); NCM3722's "
    "lesions relative to the pinned MG1655 assembly are unknown rather than empty",
)
NCM3722_BACKGROUND = BacterialStrainBackground(
    name=MEASURED_STRAIN,
    reference_strain=REFERENCE_STRAIN,
    assembly_set=ASSEMBLY_SET,
    provenance=[SOURCED_VALUES["measured_strain"]],
    provenance_gaps=[NCM3722_GENOTYPE_GAP],
)
PTSG = ("ptsG", "b1101")
GLTB = ("gltB", "b3212")
#: Released edits that no record types, and why.
UNTYPED_EDITS: dict[str, str] = {
    "NQ1243 xylR (Ptet)": "the activator the Pu promoter needs; no mirrored source "
    "states its location, so a gene-addition leaf would assert a localization",
    "NQ1390 xylR (lacIq promoter)": "as for NQ1243; the two strains share one typed "
    "genotype because of this",
    "NQ393 GDH-null": "the gdhA allele (deletion or point lesion) is not stated in any "
    "mirrored source, so a deletion leaf would assert a mechanism",
}


def _promoter(gene: tuple[str, str], promoter: str, native: str | None) -> Any:
    symbol, tag = gene
    return PromoterReplacementPerturbation(
        systematic_gene_name=tag,
        perturbed_gene_name=symbol,
        gene_namespace=GENE_NAMESPACE,
        identifier_mapping=DerivedIdentifierMapping(
            source_identifier=symbol, route="gene_symbol"
        ),
        promoter_name=promoter,
        native_promoter=native,
        expression_direction="decreased",
        is_inducible=True,
    )


def strain_genotype(strain: str) -> Genotype:
    """One released strain's typed edits relative to the NCM3722 background."""
    if strain == MEASURED_STRAIN:
        return Genotype(perturbations=[])
    if strain in ("NQ1243", "NQ1390"):
        return Genotype(perturbations=[_promoter(PTSG, "Pu", "ptsG promoter")])
    if strain == "NQ393":
        return Genotype(perturbations=[_promoter(GLTB, "Plac", None)])
    raise ValueError(f"{strain!r} is not a strain this release names")


def check_gene_symbols(genome: EcoliK12Genome) -> None:
    """The two typed promoter targets resolve to the declared b-numbers."""
    for symbol, tag in (PTSG, GLTB):
        resolved = genome.resolve_gene_name(symbol).systematic_name
        if resolved != tag:
            raise RuntimeError(f"{symbol} resolves to {resolved}, declared {tag}")


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
TEMPERATURE_GAP = ProvenanceGap(
    field="temperature",
    reason=ProvenanceGapReason.deferred_pending_source_review,
    looked_in=_LOOKED_IN_PAPER,
    note="the culture temperature is in the Supplementary Material Methods, behind #853",
)
_DEFERS_TO_SI = (
    "Balakrishnan 2022 Supplementary Material, Materials and Methods (behind the PMC "
    "proof-of-work page, issue #853)"
)


def _medium(label: str, base: str, salt: str, amount_mm: float, quote: str) -> Media:
    """A base medium the release names, its base salts deferred, its nitrogen stated."""
    return Media(
        name=f"{label} minimal medium with {quote} as the nitrogen source, base salts "
        "deferred to the SI (Balakrishnan 2022)",
        state="liquid",
        is_synthetic=True,
        base_medium=base,
        components=[
            MediaComponent(
                compound=Compound(
                    name=f"{label} minimal medium base (amounts in the Balakrishnan "
                    "2022 SI Methods)"
                ),
                role=MediaComponentRole.other,
                definition=ComponentDefinition.composition_deferred,
                provenance=[_table(label, label)],
                note="the release names the base medium and no amount of any base "
                f"salt; base_medium names the {base} family it belongs to and asserts "
                "none of that object's amounts",
                defers_to=[_DEFERS_TO_SI],
            ),
            MediaComponent(
                compound=resolved_compound(salt),
                role=MediaComponentRole.nitrogen_source,
                concentration=Concentration(
                    value=amount_mm, unit=ConcentrationUnit.millimolar
                ),
                provenance=[_table(quote, quote)],
            ),
        ],
        provenance=[_table(label, label), _paper(None, _Q_METHODS_IN_SM)],
    )


M9_BALAKRISHNAN2022 = _medium("M9", "M9", "ammonium sulfate", 11.34, M9_NITROGEN)
MOPS_BALAKRISHNAN2022 = _medium(
    "MOPS", "MOPS_MINIMAL", "ammonium chloride", 10.0, MOPS_NITROGEN
)
MEDIA: dict[str, Media] = {"M9": M9_BALAKRISHNAN2022, "MOPS": MOPS_BALAKRISHNAN2022}
SUPPLEMENT_COMPOUNDS = {
    "3MBA": "3MBA",
    "IPTG": "IPTG",
    "chloramphenicol": "chloramphenicol",
}


def _supplement(cell: str) -> SmallMoleculePerturbation:
    """``'<n> µM <compound>'`` as a small-molecule edit at that dose."""
    amount, unit, compound = cell.split(" ", 2)
    if unit != "µM" or compound not in SUPPLEMENT_COMPOUNDS:
        raise ValueError(f"supplement cell {cell!r} is not '<n> µM <known compound>'")
    return SmallMoleculePerturbation(
        compound=resolved_compound(SUPPLEMENT_COMPOUNDS[compound]),
        concentration=Concentration(
            value=float(amount), unit=ConcentrationUnit.micromolar
        ),
    )


def build_environment(medium: str, supplement: str | None) -> Environment:
    """The released medium, 0.2% glucose, and the supplement when there is one."""
    perturbations: list[EnvironmentPerturbationType] = [
        EnvironmentPhysicalPerturbation(
            factor=PhysicalFactor.carbon_source,
            magnitude=Concentration(value=0.2, unit=ConcentrationUnit.percent_w_v),
            agent=resolved_compound("glucose"),
        )
    ]
    if supplement is not None:
        perturbations.append(_supplement(supplement))
    return Environment(
        media=MEDIA[medium],
        temperature=None,
        perturbations=perturbations,
        provenance_gaps=[TEMPERATURE_GAP],
    )


# --------------------------------------------------------------------------- #
# Reading and checking the pinned workbook
# --------------------------------------------------------------------------- #
def _rows(path: str, sheet: str) -> list[tuple[Any, ...]]:
    """Every non-blank row of one sheet, values only."""
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        rows = list(book[sheet].iter_rows(values_only=True))
    finally:
        book.close()
    return [row for row in rows if any(cell is not None for cell in row)]


def _render(row: Sequence[Any]) -> str:
    """One row as its non-empty cells joined by ``" | "``, the quote rendering."""
    return " | ".join(
        str(cell).strip() for cell in row if cell is not None and str(cell).strip()
    )


def check_quotes(paths: Mapping[str, str]) -> dict[str, int]:
    """Every quote is a substring of the pinned bytes it names."""
    text = Path(paths[PAPER_TXT]).read_text(encoding="utf-8")
    for name, quote in PAPER_QUOTES.items():
        if quote not in text:
            raise RuntimeError(f"paper quote {name!r} is not in the pinned text")
    appendix = mori2021.appendix_text(paths[MORI_APPENDIX])
    for name, quote in MORI_QUOTES.items():
        if quote not in appendix:
            raise RuntimeError(f"Mori 2021 Appendix quote {name!r} is not pinned")
    rendered = "\n".join(_render(r) for r in _rows(paths[TABLE_S3], SHEET_DESCRIPTION))
    for name, quote in WORKBOOK_QUOTES.items():
        if quote not in rendered:
            raise RuntimeError(f"Table S3 quote {name!r} is not a row rendering")
    return {
        "n_quotes_checked": len(PAPER_QUOTES) + len(MORI_QUOTES) + len(WORKBOOK_QUOTES)
    }


def check_sample_metadata(path: str) -> dict[str, Any]:
    """The declared samples are the description sheet, cell for cell.

    The ``Group`` cell is blank below the first row of a series and inherits it, which
    is how the sheet reads. The fill-series nitrogen cells must be exactly the declared
    ones and every other nitrogen cell must be its medium's stated salt.
    """
    rows = _rows(path, SHEET_DESCRIPTION)
    header = tuple(str(c).strip() for c in rows[0][: len(DESCRIPTION_HEADER)])
    if header != DESCRIPTION_HEADER:
        raise RuntimeError(f"{SHEET_DESCRIPTION} header is {header}")
    series = ""
    seen: list[str] = []
    fill: dict[str, str] = {}
    for row, spec in zip(rows[1:], SAMPLES, strict=True):
        cells = dict(
            zip(DESCRIPTION_HEADER, row[: len(DESCRIPTION_HEADER)], strict=True)
        )
        if cells["Group"] is not None:
            series = str(cells["Group"])
        released = (
            str(cells["Sample ID"]),
            series,
            str(cells["Strain"]),
            str(cells["Growth medium"]),
            None if cells["Supplement"] is None else str(cells["Supplement"]),
            float(cells["Growth rate (1/h)"]),
        )
        declared = (
            spec.sample,
            spec.series,
            spec.strain,
            spec.medium,
            spec.supplement,
            spec.growth_rate_per_h,
        )
        if released != declared:
            raise RuntimeError(f"release says {released}, module declares {declared}")
        if str(cells["Carbon source"]) != CARBON_SOURCE:
            raise RuntimeError(f"{spec.sample}: carbon source {cells['Carbon source']}")
        nitrogen = str(cells["Nitrogen source"])
        expected = M9_NITROGEN if spec.medium == "M9" else MOPS_NITROGEN
        if nitrogen != expected:
            fill[spec.sample] = nitrogen
        seen.append(spec.sample)
    if fill != NITROGEN_FILL_SERIES:
        raise RuntimeError(
            f"off-recipe nitrogen cells {fill}, declared {NITROGEN_FILL_SERIES}"
        )
    return {"n_described": len(seen), "nitrogen_fill_series": fill}


class FractionRow(BaseModel):
    """One released fractions row: identifiers and the per-column values."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    row_number: int
    gene: str
    locus: str
    values: dict[str, float]


def read_fractions(path: str) -> tuple[list[FractionRow], dict[str, Any]]:
    """Read the fractions sheet, checking headers, the duplicate, and normalization."""
    rows = _rows(path, SHEET_FRACTIONS)
    header = [str(c).strip() for c in rows[0]]
    if tuple(header[:N_IDENTITY_COLUMNS]) != IDENTITY_COLUMNS:
        raise RuntimeError(f"identity block is {header[:N_IDENTITY_COLUMNS]}")
    samples = tuple(header[N_IDENTITY_COLUMNS:])
    if samples != RELEASED_HEADERS:
        raise RuntimeError(f"sample headers are {samples}, declared {RELEASED_HEADERS}")
    body = rows[1:]
    dup = [i for i, name in enumerate(header) if name == "a4_1"]
    identical = sum(1 for r in body if r[dup[0]] == r[dup[1]])
    if identical != len(body):
        raise RuntimeError(
            f"the two a4_1 columns differ in {len(body) - identical} rows"
        )
    worst = 0.0
    for j in range(N_IDENTITY_COLUMNS, len(header)):
        total = sum(float(r[j]) for r in body)
        worst = max(worst, abs(total - 1.0))
    if worst > NORMALIZATION_ATOL:
        raise RuntimeError(f"a released column sums {worst} away from 1")
    first = {name: header.index(name) for name in set(samples)}
    out = [
        FractionRow(
            row_number=offset + 2,
            gene=str(r[0]).strip(),
            locus=str(r[1]).strip(),
            values={name: float(r[i]) for name, i in first.items()},
        )
        for offset, r in enumerate(body)
    ]
    return out, {
        "n_columns": len(samples),
        "duplicate_columns_1_based": [i + 1 for i in dup],
        "duplicate_identical_rows": identical,
        "worst_abs_column_sum_deviation": worst,
    }


class RefusedRow(BaseModel):
    """One released row that is not a stored gene key, with why."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    gene: str
    locus: str
    rule: str


REFUSE_ZERO_LOCUS = "release_writes_locus_0"
REFUSE_NON_GENE_LOCUS = "b_number_is_a_non_gene_feature_locus"
REFUSE_FRAGMENT_SYNONYM = "b_number_names_no_locus_only_a_feature_synonym"
REFUSE_RETIRED = "b_number_is_retired_in_the_pinned_assembly"


def select_genes(
    rows: Sequence[FractionRow], genome: EcoliK12Genome
) -> tuple[list[FractionRow], list[RefusedRow]]:
    """Keep the rows whose released b-number is a gene of the pinned gene set."""
    genes = set(genome.gene_set)
    loci = set(genome.genbank.loci)
    kept: list[FractionRow] = []
    refused: list[RefusedRow] = []
    for row in rows:
        if row.locus == "0":
            rule = REFUSE_ZERO_LOCUS
        elif row.locus in genes:
            kept.append(row)
            continue
        elif row.locus in loci:
            rule = REFUSE_NON_GENE_LOCUS
        elif genome.resolve_gene_name(row.locus).status.value == "retired":
            rule = REFUSE_RETIRED
        else:
            rule = REFUSE_FRAGMENT_SYNONYM
        refused.append(RefusedRow(gene=row.gene, locus=row.locus, rule=rule))
    if len({row.locus for row in kept}) != len(kept):
        raise RuntimeError("two kept rows share one b-number")
    return kept, refused


# --------------------------------------------------------------------------- #
# Phenotypes and the publication
# --------------------------------------------------------------------------- #
def build_phenotype(
    rows: Sequence[FractionRow], sample: str
) -> MrnaNumberFractionPhenotype:
    """One released column, verbatim, over the kept genes."""
    return MrnaNumberFractionPhenotype(
        mrna_number_fraction={row.locus: row.values[sample] for row in rows},
        n_libraries=1,
        measurement_type=MEASUREMENT_TYPE,
    )


def build_reference_phenotype(
    rows: Sequence[FractionRow], samples: Sequence[str]
) -> MrnaNumberFractionPhenotype:
    """The per-gene mean of a medium's wild-type reference libraries."""
    return MrnaNumberFractionPhenotype(
        mrna_number_fraction={
            row.locus: statistics.fmean(row.values[s] for s in samples) for row in rows
        },
        n_libraries=len(samples),
        measurement_type=MEASUREMENT_TYPE,
    )


def publication() -> Publication:
    """The paper, by its PubMed id and DOI."""
    return Publication(
        pubmed_id=PUBMED_ID,
        pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{PUBMED_ID}/",
        doi=PAPER_DOI,
        doi_url=f"https://doi.org/{PAPER_DOI}",
    )


# --------------------------------------------------------------------------- #
# The raw mirror
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/balakrishnanPrinciplesGeneRegulation2022``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def retrieve_raw_files(dest_dir: str | Path) -> dict[str, Path]:
    """Re-run the recorded retrievals and write the verified bytes into ``dest_dir``."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for name, record in RETRIEVALS.items():
        path = dest / name
        write_verified(
            run_retriever(record), path, DATA_SHA256[name], str(record.source_url)
        )
        out[name] = path
    return out


def deposit_raw_mirror(*, source_dir: str | Path, data_root: str | None = None) -> Path:
    """Write the raw mirror plus its manifest; idempotent by sha256, never overwrites."""
    root = raw_mirror_dir(data_root)
    records: list[ArtifactRecord] = []
    for name, relpath in MIRROR_RELPATH.items():
        src = Path(source_dir) / name
        if _sha256(src) != DATA_SHA256[name]:
            raise RuntimeError(f"{src} does not hash to its pin")
        dest = root / relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if _sha256(dest) != DATA_SHA256[name]:
                raise RuntimeError(f"{dest} exists with a different sha256; refusing")
        else:
            shutil.copy2(src, dest)
        records.append(
            ArtifactRecord(
                path=relpath,
                role=ROLE_RAW_DATA,
                bytes=dest.stat().st_size,
                sha256=DATA_SHA256[name],
                source=str(RETRIEVALS[name].source_url),
                retrieval=RETRIEVALS[name],
            )
        )
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=PAPER_DOI,
        title=PAPER_TITLE,
        files=records,
        si_data_sources=[
            *(str(r.source_url) for r in RETRIEVALS.values()),
            f"https://pmc.ncbi.nlm.nih.gov/articles/{PMC_ID}/",
            f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={GEO_SERIES}",
        ],
        si_expected=list(NOT_MIRRORED),
        provenance_complete=False,
        created_at=datetime.now(UTC).isoformat(),
    )
    (root / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    return root


def load_manifest(data_root: str | None = None) -> Manifest:
    """Read the raw mirror's ``manifest.json``."""
    path = raw_mirror_dir(data_root) / "manifest.json"
    return Manifest.model_validate_json(path.read_text())


# --------------------------------------------------------------------------- #
# The dataset
# --------------------------------------------------------------------------- #
@register_dataset
class MrnaFractionBalakrishnan2022Dataset(ExperimentDataset):
    """NCM3722-family mRNA number fractions, one record per released sample column."""

    REFERENCE_STRAIN: ClassVar[EcoliK12StrainName] = REFERENCE_STRAIN

    def __init__(
        self,
        root: str = "data/torchcell/mrna_fraction_balakrishnan2022",
        io_workers: int = 0,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        ecoli_genome: EcoliK12Genome | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; ``ecoli_genome`` is injected by the build entry points."""
        self.ecoli_genome = ecoli_genome
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

    @property
    def experiment_class(self) -> type[Experiment]:
        """Experiment schema class produced by this dataset."""
        return MrnaNumberFractionExperiment

    @property
    def reference_class(self) -> type[ExperimentReference]:
        """Experiment-reference schema class produced by this dataset."""
        return MrnaNumberFractionExperimentReference

    @property
    def raw_file_names(self) -> list[str]:
        """The two files of this paper plus the sister Appendix, linked from mirrors."""
        return [*DATA_SHA256, MORI_APPENDIX]

    def download(self) -> None:
        """Link the mirrored files into ``raw/`` after checking manifests and sha256."""
        data_root = _data_root()
        os.makedirs(self.raw_dir, exist_ok=True)
        manifest = load_manifest(data_root)
        pins = {record.path: record.sha256 for record in manifest.files}
        for name, relpath in MIRROR_RELPATH.items():
            check_manifest_pin(relpath, pins[relpath], DATA_SHA256[name])
            link_verified(
                raw_mirror_dir(data_root) / relpath,
                osp.join(self.raw_dir, name),
                DATA_SHA256[name],
            )
        mori_manifest = mori2021.load_manifest(data_root)
        check_manifest_pin(
            MORI_APPENDIX_RELPATH,
            mori2021.manifest_sha256(mori_manifest, MORI_APPENDIX_RELPATH),
            MORI_APPENDIX_SHA256,
        )
        link_verified(
            mori2021.raw_mirror_dir(data_root) / MORI_APPENDIX_RELPATH,
            osp.join(self.raw_dir, MORI_APPENDIX),
            MORI_APPENDIX_SHA256,
        )

    def compute_gene_set(self) -> GeneSet:
        """The MG1655 loci the stored fractions are keyed by."""
        if self.env is None:
            self._init_db()
        genes = GeneSet()
        with self.env.begin() as txn:
            for _, value in txn.cursor():
                record = pickle.loads(value)
                genes.update(record["experiment"]["phenotype"]["mrna_number_fraction"])
        self.close_lmdb()
        return genes

    def _genome(self) -> EcoliK12Genome:
        if self.ecoli_genome is None:
            self.ecoli_genome = bacterial_genome("ecoli", self.REFERENCE_STRAIN)
        expected = BACTERIAL_ASSEMBLY_SETS[self.REFERENCE_STRAIN]
        if self.ecoli_genome.ASSEMBLY_SET != expected:
            raise ValueError(
                f"{type(self).__name__} needs the {expected} genome, got "
                f"{self.ecoli_genome.ASSEMBLY_SET}"
            )
        return self.ecoli_genome

    @post_process
    def process(self) -> None:
        """Build one record per loaded sample column and write the LMDB."""
        verify_raw_files(self.raw_dir, RAW_SHA256)
        paths = {name: osp.join(self.raw_dir, name) for name in RAW_SHA256}
        quotes = check_quotes(paths)
        metadata = check_sample_metadata(paths[TABLE_S3])
        rows, sheet_check = read_fractions(paths[TABLE_S3])
        if len(rows) != EXPECTED_SOURCE_ROWS:
            raise RuntimeError(f"{len(rows)} rows, declared {EXPECTED_SOURCE_ROWS}")
        genome = self._genome()
        check_gene_symbols(genome)
        kept, refused = select_genes(rows, genome)
        n_zero_locus = sum(1 for r in refused if r.rule == REFUSE_ZERO_LOCUS)
        if (len(kept), n_zero_locus, len(refused) - n_zero_locus) != (
            EXPECTED_GENE_KEYS,
            EXPECTED_ZERO_LOCUS_ROWS,
            EXPECTED_REFUSED_B_NUMBERS,
        ):
            raise RuntimeError(
                f"{len(kept)} kept, {n_zero_locus} locus-0, "
                f"{len(refused) - n_zero_locus} refused b-numbers"
            )
        if len(LOADED) != EXPECTED_RECORDS:
            raise RuntimeError(f"{len(LOADED)} records, declared {EXPECTED_RECORDS}")

        reference_genome = assembly_reference(
            self.REFERENCE_STRAIN, background=NCM3722_BACKGROUND
        )
        references = {
            medium: MrnaNumberFractionExperimentReference(
                dataset_name=self.name,
                genome_reference=reference_genome,
                environment_reference=build_environment(medium, None),
                phenotype_reference=build_reference_phenotype(kept, samples),
            )
            for medium, samples in REFERENCE_SAMPLES.items()
        }
        pub = publication()
        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        sample_rows: list[dict[str, Any]] = []
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for idx, spec in enumerate(tqdm(LOADED, desc="balakrishnan2022")):
                phenotype = build_phenotype(kept, spec.sample)
                experiment = MrnaNumberFractionExperiment(
                    dataset_name=self.name,
                    genotype=strain_genotype(spec.strain),
                    environment=build_environment(spec.medium, spec.supplement),
                    phenotype=phenotype,
                )
                txn.put(
                    f"{idx}".encode(),
                    self._intern_record(experiment, references[spec.medium], pub, itxn),
                )
                values = phenotype.mrna_number_fraction.values()
                sample_rows.append(
                    {
                        **spec.model_dump(exclude={"drop"}),
                        "nitrogen_cell_verbatim": NITROGEN_FILL_SERIES.get(
                            spec.sample,
                            M9_NITROGEN if spec.medium == "M9" else MOPS_NITROGEN,
                        ),
                        "n_gene_keys": len(phenotype.mrna_number_fraction),
                        "n_zero_values": sum(1 for v in values if v == 0.0),
                        "stored_fraction_sum": sum(values),
                    }
                )
        env.close()
        interned_env.close()
        self._write_ledgers(
            rows, kept, refused, sample_rows, quotes, metadata, sheet_check
        )

    def _write_ledgers(
        self,
        rows: Sequence[FractionRow],
        kept: Sequence[FractionRow],
        refused: Sequence[RefusedRow],
        sample_rows: Sequence[Mapping[str, Any]],
        quotes: Mapping[str, Any],
        metadata: Mapping[str, Any],
        sheet_check: Mapping[str, Any],
    ) -> None:
        """The drop log, the sourcing table, the sample and refused-row tables."""
        out = Path(self.preprocess_dir)
        by_rule: dict[str, list[str]] = {}
        for row in refused:
            by_rule.setdefault(row.rule, []).append(f"{row.gene} ({row.locus})")
        ledger = {
            "dataset": self.name,
            "described_samples": len(SAMPLES),
            "released_columns": len(RELEASED_HEADERS),
            "records": len(sample_rows),
            "sample_drops": {
                DROP_DUPLICATE_HEADER.rule: {
                    "description": DROP_DUPLICATE_HEADER.description,
                    "items": ["a4_1 (column 16)"],
                },
                DROP_UNRELEASED.rule: {
                    "description": DROP_UNRELEASED.description,
                    "items": [s.sample for s in SAMPLES if s.drop == DROP_UNRELEASED],
                },
            },
            "released_rows": len(rows),
            "stored_gene_keys": len(kept),
            "refused_rows": {
                rule: {"n": len(v), "items": v} for rule, v in by_rule.items()
            },
            "untyped_edits": UNTYPED_EDITS,
            "nitrogen_fill_series": NITROGEN_FILL_SERIES,
            "notes": [
                f"{len(RELEASED_HEADERS)} released columns - 1 duplicate = "
                f"{len(sample_rows)} records; a3_1 is described and unreleased",
                f"{len(rows)} released rows = {len(kept)} stored keys + "
                + " + ".join(f"{len(v)} ({k})" for k, v in by_rule.items()),
                "Environment has no growth-rate slot, so the released growth rate is in "
                "samples.csv and not on a record",
            ],
        }
        if len(kept) + len(refused) != len(rows):
            raise RuntimeError("the row ledger does not add up")
        (out / "dropped_records.json").write_text(json.dumps(ledger, indent=2))
        (out / "released_statistics_check.json").write_text(
            json.dumps(
                {
                    "quotes": dict(quotes),
                    "sample_metadata": dict(metadata),
                    "fractions_sheet": dict(sheet_check),
                    "normalization_atol": NORMALIZATION_ATOL,
                },
                indent=2,
            )
        )
        (out / "sourced_values.json").write_text(
            json.dumps(
                {k: v.model_dump(mode="json") for k, v in SOURCED_VALUES.items()},
                indent=2,
            )
        )
        pd.DataFrame(list(sample_rows)).to_csv(out / "samples.csv", index=False)
        pd.DataFrame([r.model_dump() for r in refused]).to_csv(
            out / "refused_rows.csv", index=False
        )

    def preprocess_raw(self, df: Any, preprocess: dict[str, Any] | None = None) -> Any:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# L0-L4 verification
# --------------------------------------------------------------------------- #
def verify_build(
    dataset_root: str,
    data_root: str | None = None,
    *,
    genome: EcoliK12Genome | None = None,
    expected_count: int = EXPECTED_RECORDS,
) -> VerificationReport:
    """Run the shared RNA-seq gate (number-fraction branch) plus the L4 containment."""
    from torchcell.verification.rnaseq import rnaseq_gene_set, verify_rnaseq_dataset
    from torchcell.verification.runners import (
        RNASEQ_DATASETS,
        _l4_assembly_gene_containment,
        load_records,
    )

    name = osp.basename(osp.normpath(dataset_root))
    spec = RNASEQ_DATASETS[name]
    records = load_records(dataset_root)
    if genome is None:
        genome = bacterial_genome("ecoli", REFERENCE_STRAIN, data_root)
    report = verify_rnaseq_dataset(
        records,
        dataset_name=name,
        provenance=spec["provenance"],
        expected_count=expected_count,
        replicate_aware=True,
    )
    report.add(
        _l4_assembly_gene_containment(
            set(genome.gene_set),
            (ASSEMBLY_SET,),
            rnaseq_gene_set(records),
            min_containment=spec["min_containment"],
        )
    )
    out = osp.join(dataset_root, "preprocess", "verification_report.json")
    os.makedirs(osp.dirname(out), exist_ok=True)
    with open(out, "w") as handle:
        handle.write(report.model_dump_json(indent=2))
    return report


def main() -> None:
    """Deposit the raw mirror, or build and verify the dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deposit", action="store_true")
    parser.add_argument(
        "--root", default="data/torchcell/mrna_fraction_balakrishnan2022"
    )
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv()
    data_root = _data_root()
    if args.deposit:
        staging = raw_mirror_dir(data_root) / "_staging"
        retrieve_raw_files(staging)
        root = deposit_raw_mirror(source_dir=staging, data_root=data_root)
        shutil.rmtree(staging)
        print(f"deposited {len(DATA_SHA256)} files into {root}")
        return
    build_root = osp.join(data_root, args.root)
    dataset = MrnaFractionBalakrishnan2022Dataset(root=build_root)
    print(f"len = {len(dataset)}")
    print(verify_build(build_root, data_root).summary())


if __name__ == "__main__":
    main()
