# torchcell/datasets/pputida/lim2022
# [[torchcell.datasets.pputida.lim2022]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/pputida/lim2022
# Test file: tests/torchcell/datasets/pputida/test_lim2022.py
"""Lim 2022 putidaPRECISE321: a P. putida KT2440 RNA-seq compendium, one record per sample.

Lim et al. 2022 (Metab. Eng., doi:10.1016/j.ymben.2022.04.004) assembled 321 RNA-seq
profiles of KT2440 and its derivatives, from 118 conditions across 21 projects, aligned to
the KT2440 chromosome AE015451.2 and quantified as log2(TPM + 1). It is an AGGREGATION:
225 of the 321 samples were reprocessed from other studies (SRA, plus nine samples
obtained from Bentley 2020), and 96 were generated for this paper (the sample sheet's
``Note`` column, "in-house data"). Every record names its source study: the record's
``Publication`` is the source paper when the sample sheet gives one (a DOI, else a PMID),
and Lim 2022 otherwise; ``preprocess/sample_ledger.json`` carries the full per-sample
source record (SRX, BioProject, GEO series, center, project, condition, replicate).

DATA. Both consumed files sit in the raw mirror
``$DATA_ROOT/torchcell-raw/limMachinelearningPseudomonasPutida2022/``:

- ``si/si2.xlsx``: the paper's Supplementary Data (Elsevier ``mmc2.xlsx``), sheets
  ``1-Sample_list`` (550 rows, 321 flagged ``putidaPRECISE321``), ``2-Gene table`` and
  ``5-X`` (the log2(TPM + 1) matrix, 5,564 genes x 503 samples).
- ``data/counts.csv``: the raw featureCounts matrix of the repository the paper cites,
  SBRG/modulome_ppu at commit ``f63a0df`` (5,564 genes x 521 columns).

PHENOTYPE (``RNASeqExpressionPhenotype``, ``measurement_type="rnaseq_tpm"``). ``expression_tpm``
is ``2**X - 1`` from the published X matrix; the transform is measured, not assumed: for
all 321 samples the back-transformed values sum to 1,000,000 over the 5,564 genes.
``expression_count`` is the paired column of ``counts.csv``. The pairing is measured too
(:func:`pair_count_columns`): exactly one count column reproduces each sample's X column
to below 1e-6 in log2 units. 291 samples pair with the column of their own name; 30 pair
with an in-house ``SBRG_*`` column, and for 16 of those (the ionic-liquid ALE samples) the
same-named SRX column exists but does NOT reproduce the published values.

GENOTYPE. A wild-type sample has an empty genotype on the KT2440 reference. A sample whose
condition names a gene deletion carries a ``ReleasedIdentifierPerturbation`` of
``released_kind=deletion`` per deleted gene, keyed by the symbol the condition label
releases (``Crc:Del_crcZ_crcY`` -> ``crcZ``, ``crcY``), the condition label as the strain
label (#1045 part 2): the loader resolves no symbol, so the 11 samples whose symbols the
pinned annotation does not resolve (``crc``, ``crcZ``, ``crcY``, ``finR``, ``turA``,
``turB``) are stored too, and a symbol-to-``PP_`` crosswalk is a modeling-time join.

``ALE:IL_PP5350`` (2 samples) is stored on a ``BacterialStrainBackground`` named as the
sample sheet's ``biosample_strain`` writes it (``KT2440 PP_5350'``), whose
``genotype_statement`` is its ``biosample_genotype`` cell verbatim ("9bp deletion in its
PP_5350") (#1045 part 2). ``PP_5350`` is a released locus tag, so the released-strain leaf
refuses it by design (a tag of the pinned assembly belongs on a placed leaf), and the
placed leaves cannot hold the edit: a whole-gene ``DeletionStrainPerturbation`` would
misstate a 9 bp in-frame deletion, and ``LocusVariantCallPerturbation`` needs a
``BacterialVariantCall`` with replicon coordinates the release does not give. The
background states the edit as released and types nothing it does not say.

Every other derivative is stored too (#1045 part 3; 128 samples dropped before), each
on a ``BacterialStrainBackground`` named as the release names it, its genotype statement
verbatim where the release gives one and a typed gap (``not_carried_by_curation``, the
compendium names the strain and its source study holds the genotype) where it does not:

- ``Muconate:<code>_<carbon>`` (71): named by the code (``CJ522``, ``GB032``, ``GB045``,
  ``GB062``); genotype a gap (Bentley 2020, not mirrored).
- ``ALE:IL_T8`` / ``ALE:IL_T9`` (4): named by the ``biosample_strain`` cell (``KT2440
  TA8_F83_I1``), the ``biosample_genotype`` cell ``evolved`` the statement.
- ``ALE:Galactose*`` / ``ALE:Xylose*`` (14): named by the label token (``XyloseA1``), the
  paper's "engineered P. putida KT2440 strains with heterologous gene expression" the
  statement.
- ``Mobile_gene*`` (24): ``UWC1`` as the cell writes it, a KT2440 derivative because the
  compendium keeps only KT2440 samples; genotype a gap. Both projects' baselines are UWC1
  conditions, so their references carry the UWC1 background.
- ``HNS_protein:WT_pCAR1*`` (12): ``WT_pCAR1`` from the label, genotype a gap, and the
  ``Del_turA`` / ``Del_turB`` / ``Del_pmr`` deletions as released-strain deletions.
- ``FleQ:OE_wspR`` (3): the ``biosample_strain`` cell, the ``biosample_genotype`` cell
  ``wild type with multicopy wspR`` the statement.

Plasmid and integration content is not captured as sequence; it is in the statement or
the gap. ``DropReason`` keeps every former reason, each "no longer applied".

ENVIRONMENT. The compendium carries no medium or temperature for any sample (the sample
sheet has no such column). Each condition's environment is therefore a
composition-deferred placeholder medium naming the condition and its source study (the
SynLethDB precedent), with a typed temperature gap. The one exception is the aromatic
project, whose medium the Supplementary Methods state: Lim 2022's NREL-type M9 without
its carbon source, plus the stated 2.5 g/L carbon source as an
``EnvironmentPhysicalPerturbation(factor=carbon_source)``.

REFERENCE. Each project's reference is its own baseline condition (the sample sheet's
``reference_condition``, the condition the paper centers each project on):
``expression_tpm = 2**mean(X) - 1`` over that condition's replicates, counts their rounded
mean, on the KT2440 assembly pin.
"""

from __future__ import annotations

import json
import logging
import math
import os
import os.path as osp
import pickle
import shutil
import zipfile
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar, Literal
from xml.etree import ElementTree

import lmdb
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from torchcell.data import (
    ExperimentDataset,
    Organism,
    check_manifest_pin,
    link_verified,
    post_process,
    verify_raw_files,
)
from torchcell.data.experiment_dataset import file_sha256
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.schema import (
    AssemblyReferenceGenome,
    BacterialAssemblySet,
    BacterialRNASeqExpressionExperiment,
    BacterialRNASeqExpressionExperimentReference,
    BacterialStrainBackground,
    ComponentDefinition,
    Compound,
    Concentration,
    ConcentrationUnit,
    Environment,
    EnvironmentPhysicalPerturbation,
    Experiment,
    ExperimentReference,
    Genotype,
    Media,
    MediaComponent,
    MediaComponentRole,
    PhysicalFactor,
    Publication,
    ReleasedIdentifierPerturbation,
    ReleasedKind,
    RNASeqExpressionPhenotype,
)
from torchcell.datasets.bacteria_common import (
    BACTERIAL_ASSEMBLY_SETS,
    LOCUS_TAG_PATTERNS,
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
from torchcell.sequence.genome.pputida.kt2440 import PPutidaKT2440Genome
from torchcell.verification.levels import l0_structural, l1_count, l2_value_fidelity
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
CITATION_KEY = "limMachinelearningPseudomonasPutida2022"
PAPER_DOI = "10.1016/j.ymben.2022.04.004"
RAW_DIR_REL = f"torchcell-raw/{CITATION_KEY}"

#: The library-mirror files the quotes are cut from (``manifest.json`` of the key).
PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "d5f49a99d0cf59427a7414eacd5db44ad39831c1fe89783911bf8f358520282a"
SI_DOCX = "si/si1.docx"
SI_DOCX_SHA256 = "64e1df3103b221051fb52e9d62afb39582367e20996bedc46f4d6f23e73f4750"

#: The Supplementary Data workbook (Elsevier ``mmc2.xlsx``), as the library captured it.
SI_XLSX_REL = "si/si2.xlsx"
SI_XLSX_NAME = "si2.xlsx"
SI_XLSX_SHA256 = "10e81a18fdfd08b9e27581f877dd0ae2a169946508b1ed5dabbe444210420970"
SI_XLSX_PII = "S1096717622000635"
SI_XLSX_FILENAME = "mmc2.xlsx"
SI_XLSX_URL = "https://ars.els-cdn.com/content/image/1-s2.0-S1096717622000635-mmc2.xlsx"
SI_XLSX_RETRIEVED_AT = "2026-10-07T11:44:14.766070+00:00"

#: The raw featureCounts matrix of the repository the paper's Data availability names,
#: pinned to the repository's last commit (2022-04-28), which no later commit follows.
GITHUB_REPO = "SBRG/modulome_ppu"
GITHUB_COMMIT = "f63a0dfab124ea9022123ce6227d132f39de0108"
COUNTS_PATH_IN_REPO = "data/raw_data/counts.csv"
COUNTS_URL = (
    f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_COMMIT}/"
    f"{COUNTS_PATH_IN_REPO}"
)
COUNTS_NAME = "counts.csv"
COUNTS_REL = f"data/{COUNTS_NAME}"
COUNTS_SHA256 = "59a8a1ac6b45c44a490d15a8c24a58c1083b623ed215aec758cad687cd8dbfc0"
COUNTS_RETRIEVED_AT = "2026-10-07"

SAMPLE_SHEET = "1-Sample_list"
GENE_SHEET = "2-Gene table"
X_SHEET = "5-X"

MEASUREMENT_TYPE = "rnaseq_tpm"
#: The compendium's own size statements (paper.md line 38 and line 44).
N_COMPENDIUM_SAMPLES = 321
N_CONDITIONS = 118
N_PROJECTS = 21
#: Every X gene id must resolve to a current KT2440 locus tag (measured: 5,564 of 5,564).
MIN_RESOLVED_FRACTION = 1.0
#: Largest log2(TPM + 1) deviation a count column may show and still be a sample's pair.
PAIRING_TOLERANCE = 1e-6
#: The back-transformed X columns must sum to one million (the TPM scale).
TPM_TOTAL = 1_000_000.0
GENE_NAMESPACE: Literal["pputida_kt2440_locus_tag"] = "pputida_kt2440_locus_tag"
#: The KT2440 namespace's compiled locus-tag pattern (``PP_`` tags, RNA tags included).
KT2440_LOCUS_TAG = LOCUS_TAG_PATTERNS[GENE_NAMESPACE]


def _paper(
    value: Any, quote: str, *, page: str, note: str | None = None
) -> SourcedValue:
    """Bind a value to a verbatim quote of the mirrored ``paper.md``."""
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


def _si_docx(
    value: Any,
    quote: str,
    *,
    note: str | None = None,
    page: str = "Supplementary Method 1. Transcriptome sequencing (RNA-seq)",
) -> SourcedValue:
    """Bind a value to a verbatim quote of the Supplementary Information docx.

    The quote is a substring of one paragraph as :func:`docx_paragraphs` renders it
    (the concatenated ``w:t`` runs of one ``w:p``).
    """
    return SourcedValue(
        value=value,
        quote=quote,
        note=note,
        provenance=Provenance(
            source_uri=SI_DOCX,
            citation_key=CITATION_KEY,
            sha256=SI_DOCX_SHA256,
            method="paragraph text of word/document.xml (docx_paragraphs)",
            page=page,
        ),
    )


# --------------------------------------------------------------------------- #
# Sourced statements (verbatim quotes, pinned sha256)
# --------------------------------------------------------------------------- #
COMPENDIUM_SIZE = _paper(
    {"samples": 321, "previously_published": 305, "newly_generated": 16},
    "consisting of 321 high-quality gene expression profiles, of which 305 were "
    "previously published and 16 were newly generated in this study (Supplementary "
    "Data)",
    page="Introduction (paper.md line 38)",
    note="the Introduction's split; the Methods count 97 newly generated samples among "
    "the 541 collected, and the sample sheet flags 96 of the 321 'in-house data' "
    "(80 NREL muconate, 12 aromatic, 4 ALE). 16 equals the 'SBRG' center count of the "
    "in-house samples; the 80 NREL samples share their BioProject PRJNA796354",
)
COMPENDIUM_STRUCTURE = _paper(
    {"samples": 321, "conditions": 118, "projects": 21},
    "It consists of 321 samples from 118 unique experimental conditions across 21 "
    "projects",
    page="Results 2.1 (paper.md line 44)",
)
COLLECTION = _paper(
    {"collected": 541, "sra": 435, "bentley2020": 9, "newly_generated": 97},
    "a total of 541 P. putida RNA-Seq samples were collected; 435 samples were "
    "obtained from the NCBI Sequence Read Archive (SRA, "
    "https://www.ncbi.nlm.nih.gov/sra, published before Aug 31, 2020); 9 samples were "
    "obtained from a previous study (Bentley et al., 2020); 97 samples were newly "
    "generated in this study (Supplementary Data)",
    page="Methods 4.1 (paper.md line 146)",
)
REFERENCE_GENOME = _paper(
    "AE015451.2",
    "Sequencing reads were aligned to the reference genome (AE015451.2) using Bowtie "
    "(Langmead et al., 2009)",
    page="Methods 4.1 (paper.md line 146)",
    note="AE015451.2 is the single chromosome of GCA_000007565.2 (ASM756v2), the "
    "pputida_KT2440_ASM756v2 set's deposited assembly report",
)
KT2440_ONLY = _paper(
    "KT2440",
    "Additionally, non-P. putida KT2440 samples were excluded for consistency of gene "
    "expression profiles.",
    page="Methods 4.1 (paper.md line 148)",
)
EXPRESSION_UNIT = _paper(
    "log2(TPM + 1)",
    "Read counts were generated by using RSEQC (Wang et al., 2012) and featureCounts "
    "(Liao et al., 2014) and converted into $\\log _ { 2 }$ transcripts per million "
    "(TPM).",
    page="Methods 4.1 (paper.md line 146)",
    note="the pseudocount is not stated; it is back-solved: 2**X - 1 sums to exactly "
    "1,000,000 over the 5,564 genes in all 321 samples, so X = log2(TPM + 1)",
)
REPLICATE_FILTER = _paper(
    "biological replicates, R^2 >= 0.95",
    "samples with low correlation within biological replicates $( \\mathbb { R } ^ { "
    "2 } < 0 . 9 5 )$ were discarded",
    page="Methods 4.1 (paper.md line 148)",
    note="each record is one biological-replicate library; the replicates of a "
    "condition share its project:condition identifier",
)
CONDITION_IDENTIFIERS = _paper(
    "project:condition",
    "For 321 samples that passed the aforementioned QC metrics, unique project and "
    "condition identifiers were given (project:condition, Supplementary Data).",
    page="Methods 4.1 (paper.md line 148)",
)
PROJECT_BASELINE = _paper(
    "per-project reference condition",
    "We centered each project to a baseline condition to remove batch effects "
    "(Supplementary Fig. 2)",
    page="Results 2.1 (paper.md line 44)",
)
COUNTS_REPOSITORY = _paper(
    "https://github.com/SBRG/modulome_ppu",
    "Source codes for iModulon analysis and figures are available at "
    "https://github.com/SBRG/modulome_ppu.",
    page="Data availability (paper.md line 176)",
)
ENGINEERED_SUGAR_STRAINS = _paper(
    "engineered xylose and galactose strains",
    "gene expression profiles with xylose and galactose utilization were obtained from "
    "engineered P. putida KT2440 strains with heterologous gene expression (Lim et al., "
    "2021)",
    page="Results 2.4 (paper.md line 98)",
)
IN_HOUSE_MEDIA = _si_docx(
    "LB or the modified minimal M9 medium",
    "Briefly, cells were cultured in either LB medium (10 g/L tryptone, 5 g/L yeast "
    "extract, 10 g/L NaCl) or the modified minimal M9 medium. The minimal medium "
    "contains 4 g/L glucose, 2 g/L (NH4)2SO4, 6.8 g/L Na2HPO4, 3 g/L KH2PO4, 0.5 g/L "
    "NaCl, 2 mM MgSO4, 0.1 mM CaCl2, 500 μL/L 2000× trace element solution).",
)
TRACE_ELEMENTS = _si_docx(
    {
        "ZnSO4·7H2O": 4.5,
        "MnCl2·4H2O": 0.7,
        "CoCl2·6H2O": 0.3,
        "CuSO4·2H2O": 0.2,
        "Na2MoO4·2H2O": 0.4,
        "CaCl2·2H2O": 4.5,
        "FeSO4·7H2O": 3.0,
        "H3BO3": 1.0,
        "KI": 0.1,
        "disodium ethylenediaminetetraacetate": 15.0,
    },
    "The composition of the trace element solution is 4.5 g/L ZnSO4·7H2O, 0.7 g/L "
    "MnCl2·4H2O, 0.3 g/L CoCl2·6H2O, 0.2 g/L CuSO4·2H2O, 0.4 g/L Na2MoO4·2H2O, 4.5 g/L "
    "CaCl2·2H2O, 3.0 g/L FeSO4·7H2O, 1.0 g/L H3BO3, 0.1 g/L KI, 15 g/L disodium "
    "ethylenediaminetetraacetate.",
    note="the 2000x stock in g/L; 500 uL/L dilutes it 2000-fold. Kept as one stated "
    "sub-mix rather than ten components: five of the ten hydrates have no row in the "
    "compound identity table",
)
AROMATIC_MEDIUM = _si_docx(
    "M9 + 2.5 g/L coumarate, ferulate, coumarate and ferulate, or glucose",
    "The 12 samples for the “aromatic” project were prepared with several "
    "modifications. Cells were grown in the M9 medium with 2.5 g/L of either "
    "coumarate, ferulate, a mixture of coumarate and ferulate, or glucose.",
    note="'the M9 medium' is read as the modified minimal M9 the previous paragraph "
    "defines, with its 4 g/L glucose replaced by the stated carbon source; "
    "'coumarate' is read as p-coumarate (the compound table's p-coumaric acid) and "
    "'ferulate' as ferulic acid's anion",
)
MUCONATE_DEFERRAL = _si_docx(
    "Bentley 2020",
    "The 81 samples for the “muconate” project were prepared as described previously "
    "(Bentley et al., 2020).",
    note="Bentley 2020 (doi:10.1016/j.ymben.2020.01.001) is not mirrored",
)
IN_HOUSE_TEMPERATURE = _si_docx(
    30.0,
    "incubated at 30 ℃ with continuous stirring at 1,100 rpm",
    note="stated for the M9 culture protocol; which medium the four in-house ALE "
    "samples used is not stated, and the aromatic paragraph lists its modifications "
    "without a temperature, so no record asserts 30 C",
)

#: The genotype the paper states for the seven engineered, evolved sugar strains (#1045
#: part 3): stored verbatim as each one's background genotype statement.
SUGAR_STRAIN_GENOTYPE = _paper(
    "engineered P. putida KT2440 strains with heterologous gene expression",
    "engineered P. putida KT2440 strains with heterologous gene expression",
    page="Results 2.4 (paper.md line 98)",
    note="the ALE:Galactose* and ALE:Xylose* samples; the heterologous genes are in "
    "Lim 2021 (doi:10.1021/acssuschemeng.1c03765), which is not mirrored",
)
#: How an evolved isolate's name reads (#1045 part 3: ALE:IL_T8, ALE:IL_T9).
EVOLVED_NAMING = _si_docx(
    "ALE (A), flask (F), isolate (I)",
    "Strains were previously named according to ALEdb conventions(Phaneuf et al., "
    "2019), with each separate evolution experiment receiving a different ALE (A) "
    "number, which is iteratively assigned new flask (F) numbers as the experiment "
    "runs; one isolate (I) was taken and sequenced from each flask of interest.",
    page="Supplementary Note 3. The naming of evolved strains",
    note="the biosample_strain cells 'KT2440 TA8_F83_I1' and 'KT2440 TA9_F60_I1' name "
    "an isolate of an evolution experiment; its mutations are in Lim 2020 "
    "(doi:10.1039/D0GC01663B), which is not mirrored",
)

#: The sourced statements a reader of the loader can audit in one place.
SOURCED_VALUES: tuple[SourcedValue, ...] = (
    COMPENDIUM_SIZE,
    COMPENDIUM_STRUCTURE,
    COLLECTION,
    REFERENCE_GENOME,
    KT2440_ONLY,
    EXPRESSION_UNIT,
    REPLICATE_FILTER,
    CONDITION_IDENTIFIERS,
    PROJECT_BASELINE,
    COUNTS_REPOSITORY,
    ENGINEERED_SUGAR_STRAINS,
    IN_HOUSE_MEDIA,
    TRACE_ELEMENTS,
    AROMATIC_MEDIUM,
    MUCONATE_DEFERRAL,
    IN_HOUSE_TEMPERATURE,
    SUGAR_STRAIN_GENOTYPE,
    EVOLVED_NAMING,
)

_PAPER_PROVENANCE = Provenance(
    source_uri=PAPER_MD, citation_key=CITATION_KEY, sha256=PAPER_MD_SHA256
)
_SAMPLE_SHEET_PROVENANCE = Provenance(
    source_uri=SI_XLSX_REL,
    citation_key=CITATION_KEY,
    sha256=SI_XLSX_SHA256,
    page=f"sheet {SAMPLE_SHEET!r}",
)

N_MAPPED_READS_GAP = ProvenanceGap(
    field="n_mapped_reads",
    reason=ProvenanceGapReason.deferred_pending_source_review,
    looked_in=_SAMPLE_SHEET_PROVENANCE,
    note="the sample sheet and the X matrix carry no mapped-read total; the "
    "repository's data/raw_data/multiqc_stats.tsv holds the alignment statistics and "
    "is not mirrored. The column sums of counts.csv are reads assigned to genes, not "
    "mapped reads, so they are not stored here",
)


def temperature_gap(in_house: bool) -> ProvenanceGap:
    """The typed absence behind every record's growth temperature."""
    if in_house:
        return ProvenanceGap(
            field="temperature",
            reason=ProvenanceGapReason.not_reported_by_primary,
            looked_in=IN_HOUSE_TEMPERATURE.provenance,
            note=IN_HOUSE_TEMPERATURE.note,
        )
    return ProvenanceGap(
        field="temperature",
        reason=ProvenanceGapReason.not_carried_by_curation,
        looked_in=_SAMPLE_SHEET_PROVENANCE,
        note="the compendium's sample sheet has no temperature column; the source "
        "study states it",
    )


# --------------------------------------------------------------------------- #
# Raw mirror (the loader reads the mirror, never the live URLs)
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """``DATA_ROOT`` from the environment (the mirrors and the build tree live under it)."""
    return os.environ["DATA_ROOT"]


def raw_mirror_dir(data_root: str | None = None) -> Path:
    """``$DATA_ROOT/torchcell-raw/limMachinelearningPseudomonasPutida2022``."""
    return Path(data_root or _data_root()) / RAW_DIR_REL


def _deposit(src: str | Path, dest: Path, expected: str) -> None:
    """Copy ``src`` to ``dest`` once; an existing ``dest`` must already hash to ``expected``."""
    got = file_sha256(src)
    if got != expected:
        raise RuntimeError(f"{src} sha256 mismatch: got {got}, expected {expected}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if file_sha256(dest) != expected:
            raise RuntimeError(f"{dest} exists with a different sha256; refusing")
        return
    shutil.copy2(src, dest)


def raw_manifest(si_xlsx_bytes: int, counts_bytes: int) -> Manifest:
    """The raw mirror's ``manifest.json``: both consumed files and how each was fetched."""
    return Manifest(
        citation_key=CITATION_KEY,
        doi=PAPER_DOI,
        title=(
            "Machine-learning from Pseudomonas putida KT2440 transcriptomes reveals its "
            "transcriptional regulatory network"
        ),
        files=[
            ArtifactRecord(
                path=SI_XLSX_REL,
                role=ROLE_RAW_DATA,
                bytes=si_xlsx_bytes,
                sha256=SI_XLSX_SHA256,
                source=SI_XLSX_URL,
                original_filename=SI_XLSX_FILENAME,
                retrieval=RetrievalRecord(
                    method=RetrievalMethod.direct_url,
                    source_url=SI_XLSX_URL,
                    retriever="torchcell.literature.retrieve.elsevier_mmc",
                    params={"pii": SI_XLSX_PII, "filename": SI_XLSX_FILENAME},
                    sha256=SI_XLSX_SHA256,
                    retrieved_at=SI_XLSX_RETRIEVED_AT,
                ),
            ),
            ArtifactRecord(
                path=COUNTS_REL,
                role=ROLE_RAW_DATA,
                bytes=counts_bytes,
                sha256=COUNTS_SHA256,
                source=COUNTS_URL,
                retrieval=RetrievalRecord(
                    method=RetrievalMethod.direct_url,
                    source_url=COUNTS_URL,
                    retriever="torchcell.literature.retrieve.direct_url",
                    params={"url": COUNTS_URL},
                    sha256=COUNTS_SHA256,
                    retrieved_at=COUNTS_RETRIEVED_AT,
                ),
            ),
        ],
        si_data_sources=[
            SI_XLSX_URL,
            f"https://github.com/{GITHUB_REPO}/tree/{GITHUB_COMMIT}",
            COUNTS_URL,
        ],
        si_expected=[
            "Supplementary Data (mmc2.xlsx): sample list, gene table, TRN, iModulon "
            "table, X, M and A matrices",
            "per-sample source studies: 21 projects, DOIs / PMIDs / BioProjects in the "
            "sample list (none of the source papers is mirrored)",
        ],
        provenance_complete=True,
        created_at=datetime.now(UTC).isoformat(),
    )


def deposit_raw_mirror(
    *, si_xlsx_path: str | Path, counts_path: str | Path, data_root: str | None = None
) -> Path:
    """Write the raw mirror from the two already-retrieved files and its ``manifest.json``.

    Idempotent by sha256: a mirror file already holding the pinned bytes is left alone,
    and a differing one raises rather than being overwritten. ``si_xlsx_path`` is the
    library mirror's ``si/si2.xlsx`` (retrieved by ``elsevier_mmc``); ``counts_path`` is
    the commit-pinned GitHub file (``direct_url``). Both retrievals re-run as recorded.
    """
    root = raw_mirror_dir(data_root)
    _deposit(si_xlsx_path, root / SI_XLSX_REL, SI_XLSX_SHA256)
    _deposit(counts_path, root / COUNTS_REL, COUNTS_SHA256)
    manifest = raw_manifest(
        (root / SI_XLSX_REL).stat().st_size, (root / COUNTS_REL).stat().st_size
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
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def docx_paragraphs(path: str | Path) -> list[str]:
    """Every paragraph of a docx as the concatenation of its ``w:t`` text runs.

    The rendering the Supplementary Information quotes are cut from: formatting (sub-
    and superscripts) is dropped, the characters are kept as written.
    """
    with zipfile.ZipFile(path) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
    return [
        "".join(run.text or "" for run in paragraph.iter(f"{_W}t"))
        for paragraph in root.iter(f"{_W}p")
    ]


class SupplementaryData(BaseModel):
    """The three sheets of the Supplementary Data workbook the loader consumes."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    samples: pd.DataFrame = Field(description="'1-Sample_list', one row per sample")
    genes: pd.DataFrame = Field(description="'2-Gene table', indexed by locus_tag")
    log_tpm: pd.DataFrame = Field(description="'5-X', genes x samples, log2(TPM + 1)")


def read_supplementary_data(path: str | Path) -> SupplementaryData:
    """Parse the sample list, the gene table and the X matrix from the workbook.

    ``5-X`` opens with one blank row, so its header is the second row.
    """
    with pd.ExcelFile(path, engine="openpyxl") as workbook:
        samples = workbook.parse(SAMPLE_SHEET, index_col=0)
        genes = workbook.parse(GENE_SHEET).set_index("locus_tag")
        log_tpm = workbook.parse(X_SHEET, header=1).set_index("Geneid")
    return SupplementaryData(samples=samples, genes=genes, log_tpm=log_tpm)


def compendium_samples(samples: pd.DataFrame) -> pd.DataFrame:
    """The rows the sample sheet flags ``putidaPRECISE321``, in sheet order.

    Checked against the paper's own counts: 321 samples, 118 conditions, 21 projects.
    """
    flagged = samples[samples["putidaPRECISE321"] == 1].copy()
    observed = (
        len(flagged),
        flagged["full_name"].nunique(),
        flagged["project"].nunique(),
    )
    if observed != (N_COMPENDIUM_SAMPLES, N_CONDITIONS, N_PROJECTS):
        raise ValueError(
            f"sample sheet flags {observed} (samples, conditions, projects); the paper "
            f"states {(N_COMPENDIUM_SAMPLES, N_CONDITIONS, N_PROJECTS)}"
        )
    if flagged["Sample_name"].duplicated().any():
        raise ValueError("sample sheet repeats a compendium Sample_name")
    return flagged


def tpm_from_log(log_tpm: pd.DataFrame) -> pd.DataFrame:
    """``2**X - 1`` per cell, after checking each column sums to one million.

    The check is what establishes the transform (``EXPRESSION_UNIT``): a different
    pseudocount, or a log of TPM without one, would not sum to the TPM scale.
    """
    tpm = pd.DataFrame(
        np.exp2(log_tpm.to_numpy(dtype=float)) - 1.0,
        index=log_tpm.index,
        columns=log_tpm.columns,
    )
    totals = tpm.sum(axis=0)
    off = totals[~np.isclose(totals, TPM_TOTAL, rtol=1e-6, atol=0.0)]
    if not off.empty:
        raise ValueError(
            f"{len(off)} columns of 2**X - 1 do not sum to {TPM_TOTAL:.0f}: "
            f"{off.head().to_dict()}"
        )
    return tpm


# --------------------------------------------------------------------------- #
# Aggregation: which study each sample comes from
# --------------------------------------------------------------------------- #
_DOI_PREFIX = "10."


def _cell(value: Any) -> str | None:
    """A sample-sheet cell as a stripped string, ``None`` when empty."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    return text or None


def _pmid(value: Any) -> str | None:
    """The first PMID of a sample-sheet ``PMID`` cell (``a;b`` lists several)."""
    text = _cell(value)
    if text is None:
        return None
    first = text.split(";")[0].strip()
    return str(int(float(first)))


class SourceStudy(BaseModel):
    """One sample's origin as the sample sheet records it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    project: str
    doi: str | None = Field(description="a DOI from the 'DOI' column, when it is one")
    doi_column: str | None = Field(description="the 'DOI' cell verbatim (may be a URL)")
    pmid: str | None
    bioproject: str | None
    geo_series: str | None
    center: str | None
    generated_in_this_study: bool = Field(
        description="the 'Note' cell starts 'in-house data' (case-insensitive)"
    )

    @property
    def study_key(self) -> str:
        """The study a sample is grouped under: in-house project, DOI, PMID, or project.

        A reprocessed sample whose row names no publication is grouped by its compendium
        project, because those submitters deposit one BioProject per sample.
        """
        if self.generated_in_this_study:
            return f"Lim 2022 in-house ({self.project})"
        if self.doi is not None:
            return f"doi:{self.doi}"
        if self.pmid is not None:
            return f"pmid:{self.pmid}"
        return f"no publication in the sheet ({self.project})"

    @property
    def label(self) -> str:
        """The most specific identifier: DOI, then PMID, then BioProject."""
        if self.doi is not None:
            return f"doi:{self.doi}"
        if self.pmid is not None:
            return f"pmid:{self.pmid}"
        if self.bioproject is not None:
            return f"bioproject:{self.bioproject}"
        return f"project:{self.project}"


def source_study(row: pd.Series[Any]) -> SourceStudy:
    """The ``SourceStudy`` of one sample-sheet row."""
    doi_cell = _cell(row["DOI"])
    note = _cell(row["Note"])
    return SourceStudy(
        project=str(row["project"]),
        doi=doi_cell
        if doi_cell is not None and doi_cell.startswith(_DOI_PREFIX)
        else None,
        doi_column=doi_cell,
        pmid=_pmid(row["PMID"]),
        bioproject=_cell(row["BioProject"]),
        geo_series=_cell(row["GEO Series"]),
        center=_cell(row["CenterName"]),
        generated_in_this_study=note is not None
        and note.lower().startswith("in-house data"),
    )


LIM2022_PUBLICATION = Publication(doi=PAPER_DOI, doi_url=f"https://doi.org/{PAPER_DOI}")


def record_publication(source: SourceStudy) -> Publication:
    """The publication a record carries: its source paper, else Lim 2022.

    A sample generated for Lim 2022 is published by it. A reprocessed sample is
    published by the paper its sheet row names: the DOI when the ``DOI`` column holds
    one, else the first PMID. A sample whose row names neither (a BioProject URL, a
    thesis, or nothing) is first published, as a processed profile, by Lim 2022.
    """
    if source.generated_in_this_study:
        return LIM2022_PUBLICATION
    if source.doi is not None:
        return Publication(doi=source.doi, doi_url=f"https://doi.org/{source.doi}")
    if source.pmid is not None:
        return Publication(
            pubmed_id=source.pmid,
            pubmed_url=f"https://pubmed.ncbi.nlm.nih.gov/{source.pmid}/",
        )
    return LIM2022_PUBLICATION


class SourceCount(BaseModel):
    """How many compendium samples one source contributes."""

    model_config = ConfigDict(extra="forbid")

    source: str
    projects: list[str]
    n_samples: int
    n_conditions: int
    generated_in_this_study: bool
    bioprojects: list[str]
    doi_cells: list[str] = Field(description="distinct 'DOI' cells, verbatim")


class AggregationSummary(BaseModel):
    """The compendium broken down by origin (the aggregation finding)."""

    model_config = ConfigDict(extra="forbid")

    n_samples: int
    n_generated_in_this_study: int
    n_reprocessed: int
    by_source: list[SourceCount]


def aggregation_summary(compendium: pd.DataFrame) -> AggregationSummary:
    """Count compendium samples by source study (``SourceStudy.study_key``)."""
    groups: dict[tuple[str, bool], list[tuple[pd.Series[Any], SourceStudy]]] = {}
    for _, row in compendium.iterrows():
        source = source_study(row)
        key = (source.study_key, source.generated_in_this_study)
        groups.setdefault(key, []).append((row, source))
    by_source = [
        SourceCount(
            source=label,
            projects=sorted({s.project for _, s in members}),
            n_samples=len(members),
            n_conditions=len({str(r["full_name"]) for r, _ in members}),
            generated_in_this_study=in_house,
            bioprojects=sorted({s.bioproject for _, s in members if s.bioproject}),
            doi_cells=sorted({s.doi_column for _, s in members if s.doi_column}),
        )
        for (label, in_house), members in sorted(
            groups.items(), key=lambda item: (-len(item[1]), item[0][0])
        )
    ]
    n_in_house = sum(s.n_samples for s in by_source if s.generated_in_this_study)
    return AggregationSummary(
        n_samples=len(compendium),
        n_generated_in_this_study=n_in_house,
        n_reprocessed=len(compendium) - n_in_house,
        by_source=by_source,
    )


# --------------------------------------------------------------------------- #
# Count pairing: which counts.csv column each published X column came from
# --------------------------------------------------------------------------- #
class CountPairingError(ValueError):
    """A sample has no, or more than one, count column reproducing its X column."""


class CountPairing(BaseModel):
    """The measured pairing of X columns to ``counts.csv`` columns."""

    model_config = ConfigDict(extra="forbid")

    pairs: dict[str, str] = Field(description="X sample name -> count column")
    n_same_name: int
    renamed: dict[str, str] = Field(description="pairs whose column has another name")
    same_name_rejected: dict[str, float] = Field(
        description="samples whose same-named count column exists but deviates; the "
        "value is that column's largest log2(TPM + 1) deviation"
    )
    max_deviation: float = Field(description="largest deviation over the chosen pairs")
    tolerance: float
    excluded_genes: list[str] = Field(
        description="genes left out of the comparison (length disagreement)"
    )
    n_genes_compared: int


def length_disagreements(
    genome_lengths: pd.Series, release_lengths: pd.Series
) -> list[str]:
    """Genes whose pinned-annotation length differs from the release's gene table.

    The release table gives ``end - start + 1``; the genome gives the feature's sequence
    length. They differ only where the table records part of a feature (measured: one
    gene, PP_1495 prfB, whose programmed frameshift makes the table's 72 bp the first
    segment of a 1,095 bp CDS).
    """
    joined = pd.concat(
        [genome_lengths.rename("genome"), release_lengths.rename("release")], axis=1
    )
    if joined.isna().any().any():
        missing = joined[joined.isna().any(axis=1)].index.tolist()
        raise ValueError(f"lengths missing for {missing[:10]}")
    return sorted(joined.index[joined["genome"] != joined["release"]].tolist())


def gene_length(genome: PPutidaKT2440Genome, tag: str) -> float:
    """The sequence length of one gene of the pinned annotation; an absent tag raises."""
    gene = genome[tag]
    if gene is None:
        raise KeyError(f"{tag} is not a gene of {genome.ASSEMBLY_SET}")
    return float(len(gene.seq))


def pair_count_columns(
    log_tpm: pd.DataFrame,
    counts: pd.DataFrame,
    lengths: pd.Series,
    *,
    exclude: Collection[str] = (),
    tolerance: float = PAIRING_TOLERANCE,
) -> CountPairing:
    """Find, for every X column, the one count column that reproduces it.

    For a candidate column ``c`` the prediction is ``log2(k * c / L + 1)`` with ``L`` the
    gene length and ``k`` the median of ``TPM / (c / L)`` over genes where both are
    positive, which leaves the per-sample TPM scale free. A pair is accepted when the
    largest deviation from X over the compared genes is below ``tolerance``. Every X
    column must have exactly one such column; anything else raises
    :class:`CountPairingError`. ``exclude`` drops genes whose length is in doubt.
    """
    if set(counts.index) != set(log_tpm.index):
        raise ValueError("counts and X do not cover the same genes")
    genes = [g for g in log_tpm.index if g not in set(exclude)]
    length = lengths.loc[genes].to_numpy(dtype=float)
    rate = counts.loc[genes].to_numpy(dtype=float) / length[:, None]
    columns = list(counts.columns)
    pairs: dict[str, str] = {}
    deviations: dict[str, float] = {}
    rejected: dict[str, float] = {}
    for sample in log_tpm.columns:
        x = log_tpm.loc[genes, sample].to_numpy(dtype=float)
        tpm = np.exp2(x) - 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(
                (rate > 0) & (tpm[:, None] > 0), tpm[:, None] / rate, np.nan
            )
        usable = ~np.all(np.isnan(ratio), axis=0)
        scale = np.full(len(columns), np.nan)
        scale[usable] = np.nanmedian(ratio[:, usable], axis=0)
        predicted = np.log2(rate * scale[None, :] + 1.0)
        deviation = np.abs(predicted - x[:, None]).max(axis=0)
        hits = [columns[i] for i in np.flatnonzero(deviation < tolerance)]
        if len(hits) != 1:
            raise CountPairingError(
                f"{sample}: {len(hits)} count columns reproduce it within {tolerance} "
                f"({hits[:5]})"
            )
        pairs[str(sample)] = hits[0]
        deviations[str(sample)] = float(deviation[columns.index(hits[0])])
        if hits[0] != sample and sample in columns:
            rejected[str(sample)] = float(deviation[columns.index(sample)])
    renamed = {s: c for s, c in pairs.items() if s != c}
    return CountPairing(
        pairs=pairs,
        n_same_name=len(pairs) - len(renamed),
        renamed=renamed,
        same_name_rejected=rejected,
        max_deviation=max(deviations.values()),
        tolerance=tolerance,
        excluded_genes=sorted(exclude),
        n_genes_compared=len(genes),
    )


# --------------------------------------------------------------------------- #
# Strain calls: what each compendium condition's strain is, from the metadata
# --------------------------------------------------------------------------- #
class StrainKind(StrEnum):
    """How a condition's strain is written against KT2440."""

    reference = "reference"
    deletion = "deletion"
    #: a strain stored on a background named as released, its genotype statement
    #: verbatim or a typed gap (#1045 part 2: ``ALE:IL_PP5350``; part 3: every other
    #: derivative)
    released_background = "released_background"
    dropped = "dropped"


class DropReason(StrEnum):
    """Why a compendium sample was not stored.

    Since #1045 part 3 none applies: every sample is stored, a derivative on a
    background named as released.
    """

    engineered_strain_code = "engineered_strain_code"
    evolved_isolate = "evolved_isolate"
    engineered_evolved_sugar_strain = "engineered_evolved_sugar_strain"
    sequence_variant_allele = "sequence_variant_allele"
    non_reference_background = "non_reference_background"
    plasmid_content = "plasmid_content"
    deleted_gene_unresolved = "deleted_gene_unresolved"


DROP_REASON_TEXT: dict[DropReason, str] = {
    DropReason.engineered_strain_code: "no longer applied (#1045 part 3): a "
    "muconate-producing strain the compendium names only by its code (CJ522, GB032, "
    "GB045, GB062) is stored on a background of that name, its genotype a typed gap "
    "(Bentley 2020, not mirrored)",
    DropReason.evolved_isolate: "no longer applied (#1045 part 3): an evolved isolate "
    "is stored on a background named by its biosample_strain cell, its "
    "biosample_genotype cell ('evolved') the genotype statement",
    DropReason.engineered_evolved_sugar_strain: "no longer applied (#1045 part 3): an "
    "engineered, evolved xylose or galactose strain is stored on a background named "
    "by its condition label, the paper's 'engineered P. putida KT2440 strains with "
    "heterologous gene expression' the genotype statement",
    DropReason.sequence_variant_allele: "no longer applied (#1045 part 2): the 9 bp "
    "in-frame deletion inside PP_5350 is stored on a background named as released, its "
    "biosample_genotype cell the genotype statement",
    DropReason.non_reference_background: "no longer applied (#1045 part 3): a UWC1 "
    "sample is stored on a background named 'UWC1' as its biosample_strain cell "
    "writes it, its genotype a typed gap",
    DropReason.plasmid_content: "no longer applied (#1045 part 3): a pCAR1 sample is "
    "stored on a background named 'WT_pCAR1' from its condition label (its deletions "
    "keyed by the released symbol), and the multicopy wspR sample on a background "
    "whose biosample_genotype cell is the genotype statement; plasmid sequence is not "
    "captured",
    DropReason.deleted_gene_unresolved: "no longer applied (#1045 part 2): every "
    "deletion is keyed by the symbol the condition label releases, resolved or not",
}


class StrainCall(BaseModel):
    """One condition's strain, as its metadata states it.

    A ``released_background`` call names the background as the release does: from the
    ``biosample_strain`` cell, or (when that cell is blank or names only KT2440) from
    the strain token of the condition label. Its genotype statement is the
    ``biosample_genotype`` cell verbatim, else a paper statement (``genotype_quote``),
    else a typed gap whose note is ``genotype_gap_note``. ``biosample_cells`` are the
    two cells verbatim; the build refuses a sample whose cells differ, so the call is
    the release's, not this module's.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: StrainKind
    deleted_symbols: tuple[str, ...] = ()
    drop_reason: DropReason | None = None
    background_name: str | None = None
    name_from: Literal["biosample_strain", "condition_label"] | None = None
    biosample_cells: tuple[str | None, str | None] | None = None
    genotype_statement: str | None = None
    genotype_quote: SourcedValue | None = None
    genotype_gap_note: str | None = None
    #: further statements the background rests on (the naming convention, the
    #: compendium's KT2440-only rule)
    supporting: tuple[SourcedValue, ...] = ()

    @model_validator(mode="after")
    def _released_background_is_named_and_stated_or_gapped(self) -> StrainCall:
        """A background call names its source cells; its genotype is stated or gapped."""
        if self.kind is not StrainKind.released_background:
            return self
        if (
            self.background_name is None
            or self.name_from is None
            or self.biosample_cells is None
        ):
            raise ValueError("a released background names its strain and its cells")
        strain_cell, genotype_cell = self.biosample_cells
        if self.name_from == "biosample_strain" and strain_cell != self.background_name:
            raise ValueError(
                f"{self.background_name!r} is not the biosample_strain cell "
                f"{strain_cell!r}"
            )
        if (self.genotype_statement is None) == (self.genotype_gap_note is None):
            raise ValueError("a genotype statement or a gap note, exactly one")
        if self.genotype_quote is not None:
            if genotype_cell is not None:
                raise ValueError("a paper genotype only where the cell is blank")
            if self.genotype_statement != self.genotype_quote.quote:
                raise ValueError("the genotype statement is the quote verbatim")
        elif self.genotype_statement is not None and (
            genotype_cell != self.genotype_statement
        ):
            raise ValueError(
                f"{self.genotype_statement!r} is not the biosample_genotype cell "
                f"{genotype_cell!r}"
            )
        if self.genotype_gap_note is not None and genotype_cell is not None:
            raise ValueError("a genotype gap only where the cell is blank")
        return self


_REF = StrainCall(kind=StrainKind.reference)


def _drop(reason: DropReason) -> StrainCall:
    return StrainCall(kind=StrainKind.dropped, drop_reason=reason)


def _del(*symbols: str) -> StrainCall:
    return StrainCall(kind=StrainKind.deletion, deleted_symbols=symbols)


#: Why a strain the compendium names only by a code has no genotype statement.
MUCONATE_GENOTYPE_GAP = (
    "the sample sheet names this muconate strain only by its code (biosample_strain "
    "and biosample_genotype blank) and the Supplementary Information defers the "
    "project to Bentley 2020 ('The 81 samples for the “muconate” project were prepared "
    "as described previously (Bentley et al., 2020).'), which is not mirrored"
)
#: Why the pCAR1 background has no genotype statement.
PCAR1_GENOTYPE_GAP = (
    "the condition label releases the strain as 'WT_pCAR1' (KT2440 carrying the "
    "catabolic plasmid pCAR1) and the biosample_genotype cell is blank; the plasmid's "
    "content is in its source study (Nakamura 2020, doi:10.3389/fmicb.2020.01099), "
    "which is not mirrored, and plasmid sequence is not captured"
)
#: Why the UWC1 background has no genotype statement.
UWC1_GENOTYPE_GAP = (
    "the biosample_strain cell is 'UWC1' and the biosample_genotype cell is blank; "
    "the compendium keeps only KT2440 samples, so UWC1 is stored as a KT2440 "
    "derivative whose content beyond KT2440 the release does not state"
)


def _cell_background(
    name: str,
    genotype: str | None,
    *,
    gap_note: str | None = None,
    supporting: tuple[SourcedValue, ...] = (),
) -> StrainCall:
    """A background named by its ``biosample_strain`` cell, stated by its genotype cell."""
    return StrainCall(
        kind=StrainKind.released_background,
        background_name=name,
        name_from="biosample_strain",
        biosample_cells=(name, genotype),
        genotype_statement=genotype,
        genotype_gap_note=gap_note,
        supporting=supporting,
    )


def _labelled_background(
    name: str,
    cells: tuple[str | None, str | None],
    *,
    gap_note: str | None = None,
    genotype_quote: SourcedValue | None = None,
    deleted: tuple[str, ...] = (),
) -> StrainCall:
    """A background named by the strain token of its condition label."""
    return StrainCall(
        kind=StrainKind.released_background,
        deleted_symbols=deleted,
        background_name=name,
        name_from="condition_label",
        biosample_cells=cells,
        genotype_statement=None if genotype_quote is None else genotype_quote.quote,
        genotype_quote=genotype_quote,
        genotype_gap_note=gap_note,
    )


_MUCONATE_CARBON = (
    "fructose",
    "fructose_glucose",
    "fructose_glucose_gluconate",
    "glucose_gluconate",
    "gluconate",
    "glucose",
)

#: Every compendium condition whose strain is not the KT2440 reference, with the call its
#: label and the sample sheet's biosample fields support. Read from the labels the sheet
#: gives ('Del_<gene>', 'OE_<gene>', 'pCAR1', a strain code), the 'biosample_strain' /
#: 'biosample_genotype' columns and the paper's own statements.
NON_REFERENCE_CONDITIONS: dict[str, StrainCall] = {
    **{
        f"Muconate:{code}_{carbon}": _labelled_background(
            code, (None, None), gap_note=MUCONATE_GENOTYPE_GAP
        )
        for code in ("CJ522", "GB032", "GB045", "GB062")
        for carbon in _MUCONATE_CARBON
    },
    "ALE:IL_T8": _cell_background(
        "KT2440 TA8_F83_I1", "evolved", supporting=(EVOLVED_NAMING,)
    ),
    "ALE:IL_T9": _cell_background(
        "KT2440 TA9_F60_I1", "evolved", supporting=(EVOLVED_NAMING,)
    ),
    **{
        f"ALE:{name}": _labelled_background(
            name, (None, None), genotype_quote=SUGAR_STRAIN_GENOTYPE
        )
        for name in (
            "GalactoseA5",
            "GalactoseA6",
            "GalactoseA8",
            "XyloseA1",
            "XyloseA2",
            "XyloseA3",
            "XyloseA4",
        )
    },
    "ALE:IL_PP5350": _cell_background("KT2440 PP_5350'", "9bp deletion in its PP_5350"),
    **{
        name: _cell_background(
            "UWC1", None, gap_note=UWC1_GENOTYPE_GAP, supporting=(KT2440_ONLY,)
        )
        for name in (
            "Mobile_gene:2737_exp",
            "Mobile_gene:2737_reg",
            "Mobile_gene:3227_exp",
            "Mobile_gene:3227_reg",
            "Mobile_gene:mfsR_exp",
            "Mobile_gene2:Condition1",
            "Mobile_gene2:Condition2",
        )
    },
    **{
        f"HNS_protein:{name}": _labelled_background(
            "WT_pCAR1", ("KT2440", None), gap_note=PCAR1_GENOTYPE_GAP, deleted=deleted
        )
        for name, deleted in (
            ("WT_pCAR1_ex", ()),
            ("WT_pCAR1_Del_turA_ex", ("turA",)),
            ("WT_pCAR1_Del_turA_st", ("turA",)),
            ("WT_pCAR1_Del_turB_ex", ("turB",)),
            ("WT_pCAR1_Del_turB_st", ("turB",)),
            ("WT_pCAR1_Del_pmr_ex", ("pmr",)),
        )
    },
    "FleQ:OE_wspR": _cell_background(
        "Pseudomonas putida KT2440", "wild type with multicopy wspR"
    ),
    "RelA:Del_relA": _del("relA"),
    "RelA:Del_relA_lignin": _del("relA"),
    "FleQ:Del_fleQ": _del("fleQ"),
    "FinR:Del_finR": _del("finR"),
    "HNS_protein:Del_turB_ex": _del("turB"),
    "HNS_protein:WT_Del_turA_ex": _del("turA"),
    "Crc:Del_crc": _del("crc"),
    "Crc:Del_crcZ_crcY": _del("crcZ", "crcY"),
}

#: Every compendium condition whose label and biosample fields name no strain edit, so
#: its strain is the KT2440 reference (the compendium admits only KT2440, KT2440_ONLY).
#: 'ALE:IL_Mixed' and 'ALE:IL_Control2' are the sheet's in-house 'Wildtype_...' samples.
REFERENCE_CONDITIONS: frozenset[str] = frozenset(
    {
        *(f"Muconate:KT2440_{carbon}" for carbon in _MUCONATE_CARBON),
        "ALE:IL_Control",
        "ALE:IL_TEOH_OAc",
        "ALE:IL_TEA_HS",
        "ALE:IL_Glc+Ace",
        "ALE:IL_Mixed",
        "ALE:IL_Control2",
        "Fuel:Butanol",
        "Fuel:Control",
        "Fuel:Isopentanol",
        "Aromatic:Coumarate",
        "Aromatic:Coumarate+ferulate",
        "Aromatic:Ferulate",
        "Aromatic:Glucose",
        "RelA:Control",
        "Multistress:Control",
        "Multistress:NaCl_7min",
        "Multistress:NaCl_60min",
        "Multistress:Imipenem_7min",
        "Multistress:Imipenem_60min",
        "Multistress:H2O2_7min",
        "Multistress:H2O2_60min",
        "Phase:WT_Tr",
        "Phase:WT_Ex",
        "Phase:WT_St",
        "Glycolaldehyde:Glycolaldehyde",
        "Glycolaldehyde:Control",
        "FinR:Control",
        "Myristic_acid:Control",
        "Myristic_acid:Myristic_acid",
        "Carbon:Citrate",
        "Carbon:Ferulate",
        "Carbon:Glucose",
        "Carbon:Serine",
        "Bioreactor1:24h",
        "Bioreactor2:S_0min",
        "Bioreactor2:S_15min",
        "Bioreactor2:P1_15min",
        "Bioreactor2:P3_15min",
        "Bioreactor2:P5_15min",
        "Bioreactor2:S_1h",
        "Bioreactor2:S_3h",
        "Bioreactor2:S_9h",
        "Bioreactor2:S_25h",
        "Bioreactor2:P1_25h",
        "Bioreactor2:P3_25h",
        "Bioreactor2:P5_25h",
        "FleQ:Control",
        "Carbon2:Glucose",
        "Carbon2:Coumarate",
        "HNS_protein:Control_ex",
        "Zn:Zn_ion",
        "Zn:Control",
        "ZnO:Control",
        "ZnO:Nano",
        "ZnO:Zn_ion",
        "Crc:Control",
    }
)


class UnclassifiedConditionError(KeyError):
    """A compendium condition has no strain call."""


def released_deletion(symbol: str, strain_label: str) -> ReleasedIdentifierPerturbation:
    """One deleted gene, keyed by the symbol the condition label releases (#1045 part 2)."""
    return ReleasedIdentifierPerturbation(
        systematic_gene_name=symbol,
        perturbed_gene_name=symbol,
        gene_namespace=GENE_NAMESPACE,
        strain_label=strain_label,
        identifier_kind="gene symbol",
        released_kind=ReleasedKind.deletion,
    )


def released_background(call: StrainCall, full_name: str) -> BacterialStrainBackground:
    """The strain of a ``released_background`` call, named and stated as released.

    The name is quoted from the cell or the condition label it comes from; the genotype
    statement is the cell or the paper verbatim, or ``None`` beside a typed gap.
    """
    if call.kind is not StrainKind.released_background or call.background_name is None:
        raise ValueError(f"{full_name}: a {call.kind.value} call names no background")
    if call.name_from == "condition_label" and call.background_name not in full_name:
        raise ValueError(f"{call.background_name!r} is not in the label {full_name!r}")
    provenance = [
        SourcedValue(
            value=call.background_name,
            quote=call.background_name
            if call.name_from == "biosample_strain"
            else full_name,
            provenance=_SAMPLE_SHEET_PROVENANCE,
            note="the biosample_strain cell"
            if call.name_from == "biosample_strain"
            else "the strain token of the full_name cell (the condition label)",
        )
    ]
    if call.genotype_quote is not None:
        provenance.append(call.genotype_quote)
    elif call.genotype_statement is not None:
        provenance.append(
            SourcedValue(
                value=call.genotype_statement,
                quote=call.genotype_statement,
                provenance=_SAMPLE_SHEET_PROVENANCE,
                note="the biosample_genotype cell",
            )
        )
    provenance.extend(call.supporting)
    return BacterialStrainBackground(
        name=call.background_name,
        reference_strain="KT2440",
        assembly_set=TypeAdapter(BacterialAssemblySet).validate_python(
            BACTERIAL_ASSEMBLY_SETS["KT2440"]
        ),
        genotype_statement=call.genotype_statement,
        provenance=provenance,
        provenance_gaps=[]
        if call.genotype_gap_note is None
        else [
            ProvenanceGap(
                field="genotype_statement",
                reason=ProvenanceGapReason.not_carried_by_curation,
                looked_in=_SAMPLE_SHEET_PROVENANCE,
                note=call.genotype_gap_note,
            )
        ],
    )


def strain_call(full_name: str) -> StrainCall:
    """The strain call of one ``project:condition``; an unlisted one raises."""
    if full_name in NON_REFERENCE_CONDITIONS:
        return NON_REFERENCE_CONDITIONS[full_name]
    if full_name in REFERENCE_CONDITIONS:
        return _REF
    raise UnclassifiedConditionError(
        f"{full_name!r} has no strain call; add it to NON_REFERENCE_CONDITIONS or "
        "REFERENCE_CONDITIONS from its metadata"
    )


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
#: Conditions grown in Lim 2022's stated M9: the aromatic project. Each value is the
#: carbon source(s) the SI names; a mixture has no stated split.
AROMATIC_CARBON: dict[str, tuple[str, ...]] = {
    "Aromatic:Coumarate": ("p-coumaric acid",),
    "Aromatic:Ferulate": ("ferulic acid",),
    "Aromatic:Coumarate+ferulate": ("p-coumaric acid", "ferulic acid"),
    "Aromatic:Glucose": ("D-glucose",),
}
AROMATIC_CARBON_G_PER_L = 2.5


def _salt(
    name: str,
    role: MediaComponentRole,
    value: float,
    unit: ConcentrationUnit,
    read: str,
) -> MediaComponent:
    return MediaComponent(
        compound=resolved_compound(name),
        role=role,
        concentration=Concentration(value=value, unit=unit),
        provenance=[
            SourcedValue(
                value=read,
                quote=IN_HOUSE_MEDIA.quote,
                provenance=IN_HOUSE_MEDIA.provenance,
            )
        ],
    )


_GL = ConcentrationUnit.g_per_l
_MM = ConcentrationUnit.millimolar

LIM2022_M9_NO_CARBON = Media(
    name="M9 (NREL type: ammonium sulfate + 2000x trace elements), no carbon source "
    "(Lim 2022 aromatic project)",
    state="liquid",
    is_synthetic=True,
    base_medium="M9",
    components=[
        _salt(
            "ammonium sulfate", MediaComponentRole.nitrogen_source, 2.0, _GL, "2 g/L"
        ),
        _salt(
            "disodium hydrogen phosphate",
            MediaComponentRole.bulk_salt,
            6.8,
            _GL,
            "6.8 g/L",
        ),
        _salt(
            "potassium dihydrogen phosphate",
            MediaComponentRole.bulk_salt,
            3.0,
            _GL,
            "3 g/L",
        ),
        _salt("sodium chloride", MediaComponentRole.bulk_salt, 0.5, _GL, "0.5 g/L"),
        _salt("magnesium sulfate", MediaComponentRole.bulk_salt, 2.0, _MM, "2 mM"),
        _salt("calcium chloride", MediaComponentRole.bulk_salt, 0.1, _MM, "0.1 mM"),
        MediaComponent(
            compound=Compound(name="2000x trace element solution (Lim 2022 SI)"),
            role=MediaComponentRole.trace_element,
            definition=ComponentDefinition.composition_deferred,
            concentration=Concentration(value=0.05, unit=ConcentrationUnit.percent_v_v),
            provenance=[
                SourcedValue(
                    value="500 uL/L: 0.05% v/v",
                    quote=IN_HOUSE_MEDIA.quote,
                    provenance=IN_HOUSE_MEDIA.provenance,
                ),
                TRACE_ELEMENTS,
            ],
            note="the composition is stated (TRACE_ELEMENTS) and not expanded into "
            "components",
        ),
    ],
    dropouts=[resolved_compound("ammonium chloride")],
    provenance=[
        IN_HOUSE_MEDIA,
        SourcedValue(
            value="the carbon source is the variable",
            quote=AROMATIC_MEDIUM.quote,
            provenance=AROMATIC_MEDIUM.provenance,
            note="the base's 4 g/L glucose is left out; each condition carries its "
            "2.5 g/L carbon source as EnvironmentPhysicalPerturbation(factor="
            "carbon_source). Ammonium sulfate replaces the M9 base's ammonium "
            "chloride, which is the dropout. The salts are the amounts M9_NREL_LIM2025 "
            "states, sourced here to this paper's own sentence",
        ),
    ],
)
"""Lim 2022's in-house M9 without its carbon source, the base of the aromatic project."""


def _carbon_source(
    compound_name: str, *, mixture: bool
) -> EnvironmentPhysicalPerturbation:
    """One stated carbon source of an aromatic condition."""
    if not mixture:
        return EnvironmentPhysicalPerturbation(
            factor=PhysicalFactor.carbon_source,
            magnitude=Concentration(value=AROMATIC_CARBON_G_PER_L, unit=_GL),
            agent=resolved_compound(compound_name),
        )
    return EnvironmentPhysicalPerturbation(
        factor=PhysicalFactor.carbon_source,
        magnitude=None,
        agent=resolved_compound(compound_name),
        provenance_gaps=[
            ProvenanceGap(
                field="magnitude",
                reason=ProvenanceGapReason.not_reported_by_primary,
                looked_in=AROMATIC_MEDIUM.provenance,
                note="'2.5 g/L of ... a mixture of coumarate and ferulate': the split "
                "between the two is not stated",
            )
        ],
    )


def aromatic_environment(full_name: str) -> Environment:
    """The aromatic project's stated environment: Lim 2022 M9 + its carbon source(s)."""
    carbons = AROMATIC_CARBON[full_name]
    return Environment(
        media=LIM2022_M9_NO_CARBON,
        temperature=None,
        perturbations=[
            _carbon_source(name, mixture=len(carbons) > 1) for name in carbons
        ],
        provenance_gaps=[temperature_gap(in_house=True)],
    )


def deferred_environment(full_name: str, sources: Sequence[SourceStudy]) -> Environment:
    """A condition whose environment the compendium does not carry.

    The medium is a composition-deferred placeholder naming the condition, so two
    conditions never share an environment the source studies may distinguish, and the
    note names where the recipe is stated. Same stand-in as SynLethDB's: ``Media`` has
    no ``provenance_gaps`` and ``Environment.media`` is required. ``sources`` are the
    condition's samples; they must agree on whether they were generated in-house.
    """
    in_house = {source.generated_in_this_study for source in sources}
    if len(in_house) != 1:
        raise ValueError(f"{full_name}: mixes in-house and reprocessed samples")
    source = sources[0]
    if source.generated_in_this_study and source.project == "Muconate":
        where = (
            "Bentley 2020 (doi:10.1016/j.ymben.2020.01.001; SI: 'prepared as "
            "described previously'), not mirrored"
        )
        statement = MUCONATE_DEFERRAL
    elif source.generated_in_this_study:
        where = (
            "Lim 2022 SI Method 1, which names LB and the modified M9 without saying "
            "which medium these samples used"
        )
        statement = IN_HOUSE_MEDIA
    else:
        labels = ", ".join(sorted({s.label for s in sources}))
        where = f"the source study ({labels}), not mirrored"
        statement = SourcedValue(
            value="no medium or temperature carried by the compendium",
            quote=CONDITION_IDENTIFIERS.quote,
            provenance=CONDITION_IDENTIFIERS.provenance,
            note="the sample sheet identifies a sample by project:condition and has no "
            "medium, temperature or dose column",
        )
    label = (
        f"growth environment of putidaPRECISE321 condition {full_name!r} (medium, "
        "additions and timing not carried by the compendium)"
    )
    return Environment(
        media=Media(
            name=label,
            state="liquid",
            is_synthetic=False,
            components=[
                MediaComponent(
                    compound=Compound(name=label),
                    role=MediaComponentRole.other,
                    definition=ComponentDefinition.composition_deferred,
                    note=f"stated by {where}. state='liquid' and is_synthetic=False "
                    "are NOT sourced: the schema requires both",
                )
            ],
            provenance=[statement],
        ),
        temperature=None,
        provenance_gaps=[temperature_gap(in_house=source.generated_in_this_study)],
    )


def condition_environment(
    full_name: str, sources: Sequence[SourceStudy]
) -> Environment:
    """The environment of one compendium condition (``sources``: its samples')."""
    if full_name in AROMATIC_CARBON:
        return aromatic_environment(full_name)
    return deferred_environment(full_name, sources)


# --------------------------------------------------------------------------- #
# Phenotype
# --------------------------------------------------------------------------- #
def expression_phenotype(
    tpm: Mapping[str, float], counts: Mapping[str, int]
) -> RNASeqExpressionPhenotype:
    """One sample's (or one reference's) absolute expression."""
    return RNASeqExpressionPhenotype(
        expression_tpm=dict(tpm),
        expression_count=dict(counts),
        measurement_type=MEASUREMENT_TYPE,
        n_mapped_reads=None,
        provenance_gaps=[N_MAPPED_READS_GAP],
    )


def reference_phenotype(
    log_tpm: pd.DataFrame, counts: pd.DataFrame
) -> RNASeqExpressionPhenotype:
    """A baseline condition's expression from its replicate columns.

    TPM is ``2**mean(X) - 1``, the level the paper's centering subtracts in log space;
    counts are the rounded arithmetic mean of the paired count columns.
    """
    mean_log = log_tpm.mean(axis=1)
    tpm = np.exp2(mean_log) - 1.0
    mean_count = counts.mean(axis=1).round().astype(int)
    return expression_phenotype(
        {str(g): float(v) for g, v in tpm.items()},
        {str(g): int(v) for g, v in mean_count.items()},
    )


# --------------------------------------------------------------------------- #
# Ledgers
# --------------------------------------------------------------------------- #
class SampleRecord(BaseModel):
    """The per-sample provenance ledger row (one per compendium sample)."""

    model_config = ConfigDict(extra="forbid")

    index: int | None = Field(description="LMDB index; None for a dropped sample")
    sample_name: str
    srx: str | None
    project: str
    condition: str
    full_name: str
    rep_name: int | None
    reference_condition: str
    source: SourceStudy
    publication: Publication
    count_column: str
    strain: StrainKind
    deleted_genes: list[str]
    drop_reason: DropReason | None


class DropRule(BaseModel):
    """One drop reason, the samples it removed, and the conditions they belong to."""

    model_config = ConfigDict(extra="forbid")

    reason: DropReason
    explanation: str
    n_samples: int
    conditions: list[str]


class DropLog(BaseModel):
    """Every compendium sample is either stored or dropped under one reason."""

    model_config = ConfigDict(extra="forbid")

    n_compendium: int
    n_stored: int
    rules: list[DropRule]


class BuildReport(BaseModel):
    """What the build measured, written to ``preprocess/build_report.json``."""

    model_config = ConfigDict(extra="forbid")

    aggregation: AggregationSummary
    stored_aggregation: AggregationSummary
    expression_genes: LocusTagReconciliation
    deleted_symbols: list[str] = Field(
        description="every deleted gene symbol the stored condition labels release, "
        "stored as released (#1045 part 2)"
    )
    count_pairing: CountPairing
    drops: DropLog
    replicates_per_condition: dict[str, int] = Field(
        description="stored conditions by replicate count, e.g. {'3': 40}"
    )


def drop_log(rows: Sequence[SampleRecord]) -> DropLog:
    """Tally the dropped samples by reason."""
    by_reason: dict[DropReason, list[SampleRecord]] = {}
    for row in rows:
        if row.drop_reason is not None:
            by_reason.setdefault(row.drop_reason, []).append(row)
    return DropLog(
        n_compendium=len(rows),
        n_stored=sum(1 for row in rows if row.drop_reason is None),
        rules=[
            DropRule(
                reason=reason,
                explanation=DROP_REASON_TEXT[reason],
                n_samples=len(members),
                conditions=sorted({m.full_name for m in members}),
            )
            for reason, members in sorted(by_reason.items(), key=lambda kv: kv[0].value)
        ],
    )


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
@register_dataset
class PutidaPrecise321Lim2022Dataset(ExperimentDataset):
    """putidaPRECISE321: per-sample absolute KT2440 transcriptomes (TPM + counts)."""

    organism = Organism.pputida

    REFERENCE_STRAIN: ClassVar[Literal["KT2440"]] = "KT2440"

    def __init__(
        self,
        root: str = "data/torchcell/putida_precise321_lim2022",
        io_workers: int = 0,
        pputida_genome: PPutidaKT2440Genome | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dataset; the build entry points inject ``pputida_genome``."""
        self.pputida_genome = pputida_genome
        super().__init__(root, io_workers, transform, pre_transform, **kwargs)

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
        """The Supplementary Data workbook and the GitHub count matrix."""
        return [SI_XLSX_NAME, COUNTS_NAME]

    def download(self) -> None:
        """Link the two mirror files into ``raw/`` after verifying each against its pin.

        The manifest is the retrieval record and must carry the same digests, else
        ``ManifestPinMismatchError`` names both.
        """
        data_root = _data_root()
        manifest = load_manifest(data_root)
        os.makedirs(self.raw_dir, exist_ok=True)
        for relpath, name, expected in (
            (SI_XLSX_REL, SI_XLSX_NAME, SI_XLSX_SHA256),
            (COUNTS_REL, COUNTS_NAME, COUNTS_SHA256),
        ):
            check_manifest_pin(relpath, manifest_sha256(manifest, relpath), expected)
            src = raw_mirror_dir(data_root) / relpath
            if not src.exists():
                raise RuntimeError(f"required raw artifact missing from mirror: {src}")
            link_verified(src, osp.join(self.raw_dir, name), expected)
        log.info("Lim 2022 raw files linked into %s (sha256 verified)", self.raw_dir)

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Preprocessing is handled inside ``process`` for this dataset."""
        return df

    @post_process
    def process(self) -> None:
        """Build one record per stored compendium sample and write the LMDB + ledgers."""
        verify_raw_files(
            self.raw_dir, {SI_XLSX_NAME: SI_XLSX_SHA256, COUNTS_NAME: COUNTS_SHA256}
        )
        os.makedirs(self.preprocess_dir, exist_ok=True)
        genome = self.pputida_genome
        if genome is None:  # a direct run; the build entry points inject it
            genome = bacterial_genome("pputida", self.REFERENCE_STRAIN)
            self.pputida_genome = genome

        data = read_supplementary_data(osp.join(self.raw_dir, SI_XLSX_NAME))
        compendium = compendium_samples(data.samples)
        sample_names = compendium["Sample_name"].astype(str).tolist()
        log_tpm = data.log_tpm[sample_names]
        tpm = tpm_from_log(log_tpm)

        # Genes: every X id must be a current KT2440 locus tag.
        _stored_genes, gene_report = reconcile_locus_tags(
            genome, pd.Series(log_tpm.index.astype(str)), label=f"{self.name} X genes"
        )
        gene_report.require_resolved(MIN_RESOLVED_FRACTION)
        if gene_report.remapped or gene_report.outside_namespace:
            raise ValueError(
                f"X gene ids are not all locus tags as given: {gene_report}"
            )

        # Counts: pair each X column with the counts.csv column that reproduces it.
        counts = pd.read_csv(osp.join(self.raw_dir, COUNTS_NAME), index_col=0)
        genome_lengths = pd.Series(
            {tag: gene_length(genome, str(tag)) for tag in log_tpm.index}, dtype=float
        )
        release_lengths = (data.genes["end"] - data.genes["start"] + 1).astype(float)
        pairing = pair_count_columns(
            log_tpm,
            counts,
            genome_lengths,
            exclude=length_disagreements(genome_lengths, release_lengths),
        )

        # Strains: the call of every condition; deletions keep their released symbols.
        calls = {fn: strain_call(fn) for fn in compendium["full_name"].unique()}
        symbols = sorted({s for c in calls.values() for s in c.deleted_symbols})

        reference = assembly_reference(self.REFERENCE_STRAIN)
        rows, records = self._assemble(
            compendium, log_tpm, tpm, counts, pairing, calls, reference
        )
        self._write_lmdb(records)

        stored = compendium.loc[
            np.array([row.drop_reason is None for row in rows], dtype=bool)
        ]
        replicates = Counter(Counter(str(fn) for fn in stored["full_name"]).values())
        build_report = BuildReport(
            aggregation=aggregation_summary(compendium),
            stored_aggregation=aggregation_summary(stored),
            expression_genes=gene_report,
            deleted_symbols=symbols,
            count_pairing=pairing,
            drops=drop_log(rows),
            replicates_per_condition={str(k): v for k, v in sorted(replicates.items())},
        )
        self._write_json("build_report.json", build_report.model_dump(mode="json"))
        self._write_json(
            "sample_ledger.json", [row.model_dump(mode="json") for row in rows]
        )
        self._write_json(
            "dropped_records.json", build_report.drops.model_dump(mode="json")
        )
        log.info(
            "Wrote %d of %d compendium samples; drops %s",
            len(records),
            len(rows),
            {r.reason.value: r.n_samples for r in build_report.drops.rules},
        )

    def _assemble(
        self,
        compendium: pd.DataFrame,
        log_tpm: pd.DataFrame,
        tpm: pd.DataFrame,
        counts: pd.DataFrame,
        pairing: CountPairing,
        calls: Mapping[str, StrainCall],
        reference: AssemblyReferenceGenome,
    ) -> tuple[list[SampleRecord], list[dict[str, Any]]]:
        """The ledger rows of every compendium sample and the records of the stored ones."""
        sources = {
            str(row["Sample_name"]): source_study(row)
            for _, row in compendium.iterrows()
        }
        by_condition: dict[str, list[SourceStudy]] = {}
        for _, row in compendium.iterrows():
            by_condition.setdefault(str(row["full_name"]), []).append(
                sources[str(row["Sample_name"])]
            )
        environments = {
            full_name: condition_environment(full_name, members)
            for full_name, members in by_condition.items()
        }
        references = self._project_references(
            compendium, log_tpm, counts, pairing, calls, environments, reference
        )
        genes = [str(g) for g in log_tpm.index]
        rows: list[SampleRecord] = []
        records: list[dict[str, Any]] = []
        for _, row in compendium.iterrows():
            sample = str(row["Sample_name"])
            full_name = str(row["full_name"])
            call = calls[full_name]
            source = sources[sample]
            drop_reason = call.drop_reason
            publication = record_publication(source)
            index: int | None = None
            if drop_reason is None:
                index = len(records)
                perturbations = [
                    released_deletion(symbol, full_name)
                    for symbol in call.deleted_symbols
                ]
                record_reference = references[str(row["project"])]
                if call.kind is StrainKind.released_background:
                    cells = (
                        _cell(row["biosample_strain"]),
                        _cell(row["biosample_genotype"]),
                    )
                    if cells != call.biosample_cells:
                        raise ValueError(
                            f"{sample}: biosample cells {cells} are not the call's "
                            f"{call.biosample_cells}"
                        )
                    record_reference = record_reference.model_copy(
                        update={
                            "genome_reference": assembly_reference(
                                self.REFERENCE_STRAIN,
                                background=released_background(call, full_name),
                            )
                        }
                    )
                column = pairing.pairs[sample]
                experiment = self.create_experiment(
                    perturbations=perturbations,
                    environment=environments[full_name],
                    tpm=dict(zip(genes, tpm[sample].tolist(), strict=True)),
                    counts=dict(zip(genes, counts[column].tolist(), strict=True)),
                )
                records.append(
                    {
                        "experiment": experiment.model_dump(),
                        "reference": record_reference.model_dump(),
                        "publication": publication.model_dump(),
                    }
                )
            rows.append(
                SampleRecord(
                    index=index,
                    sample_name=sample,
                    srx=_cell(row["SRX"]),
                    project=str(row["project"]),
                    condition=str(row["condition"]),
                    full_name=full_name,
                    rep_name=None
                    if _cell(row["rep_name"]) is None
                    else int(row["rep_name"]),
                    reference_condition=str(row["reference_condition"]),
                    source=source,
                    publication=publication,
                    count_column=pairing.pairs[sample],
                    strain=call.kind if drop_reason is None else StrainKind.dropped,
                    deleted_genes=list(call.deleted_symbols),
                    drop_reason=drop_reason,
                )
            )
        return rows, records

    def create_experiment(  # type: ignore[override]
        self,
        *,
        perturbations: Sequence[ReleasedIdentifierPerturbation],
        environment: Environment,
        tpm: Mapping[str, float],
        counts: Mapping[str, int],
    ) -> BacterialRNASeqExpressionExperiment:
        """One sample's experiment: its genotype on KT2440, its condition, its profile."""
        return BacterialRNASeqExpressionExperiment(
            dataset_name=self.name,
            genotype=Genotype(perturbations=list(perturbations)),
            environment=environment,
            phenotype=expression_phenotype(tpm, counts),
        )

    def _project_references(
        self,
        compendium: pd.DataFrame,
        log_tpm: pd.DataFrame,
        counts: pd.DataFrame,
        pairing: CountPairing,
        calls: Mapping[str, StrainCall],
        environments: Mapping[str, Environment],
        reference: AssemblyReferenceGenome,
    ) -> dict[str, BacterialRNASeqExpressionExperimentReference]:
        """Each project's baseline-condition reference (projects with a stored sample).

        The baseline is a reference-strain condition, or a condition on a released
        background with no further edit (the UWC1 projects, #1045 part 3), whose
        reference then carries that background; a baseline with a deletion raises
        rather than borrowing another baseline.
        """
        stored_projects = {
            str(row["project"])
            for _, row in compendium.iterrows()
            if calls[str(row["full_name"])].kind is not StrainKind.dropped
        }
        references: dict[str, BacterialRNASeqExpressionExperimentReference] = {}
        for project in sorted(stored_projects):
            members = compendium[compendium["project"] == project]
            baselines = members["reference_condition"].unique()
            if len(baselines) != 1:
                raise ValueError(f"{project}: reference conditions {list(baselines)}")
            baseline = f"{project}:{baselines[0]}"
            call = calls[baseline]
            if call.kind is StrainKind.reference:
                genome_reference = reference
            elif (
                call.kind is StrainKind.released_background and not call.deleted_symbols
            ):
                genome_reference = assembly_reference(
                    self.REFERENCE_STRAIN,
                    background=released_background(call, baseline),
                )
            else:
                raise ValueError(f"{project}: baseline {baseline} carries an edit")
            replicates = members[members["full_name"] == baseline]
            names = replicates["Sample_name"].astype(str).tolist()
            columns = [pairing.pairs[n] for n in names]
            references[project] = BacterialRNASeqExpressionExperimentReference(
                dataset_name=self.name,
                genome_reference=genome_reference,
                environment_reference=environments[baseline],
                phenotype_reference=reference_phenotype(
                    log_tpm[names], counts[columns]
                ),
            )
        return references

    def _write_lmdb(self, records: Sequence[dict[str, Any]]) -> None:
        """Write the records in order under keys ``0..n-1``."""
        env = lmdb.open(osp.join(self.processed_dir, "lmdb"), map_size=int(1e11))
        with env.begin(write=True) as txn:
            for index, record in enumerate(records):
                txn.put(f"{index}".encode(), pickle.dumps(record))
        env.close()

    def _write_json(self, name: str, payload: Any) -> None:
        """Write one preprocess ledger file."""
        with open(osp.join(self.preprocess_dir, name), "w") as handle:
            json.dump(payload, handle, indent=2)


# --------------------------------------------------------------------------- #
# Verification (L0-L4) for replicate-level records
# --------------------------------------------------------------------------- #
def verify_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_count: int,
    gene_universe: Collection[str],
) -> VerificationReport:
    """The L0-L4 gate for this dataset's records.

    The family verifier (``verify_rnaseq_dataset``) keys L1 on one record per (strain,
    environment), which a replicate-level compendium breaks by design: replicates share
    both, and a wild-type record carries no perturbation to hold a strain id. L1 here
    is per sample instead: the count matches, and no two records carry the same profile.
    """
    from pydantic import TypeAdapter

    from torchcell.datamodels.schema import ExperimentType

    report = VerificationReport(
        dataset_name="putida_precise321_lim2022",
        provenance=Provenance(
            source_uri=f"https://doi.org/{PAPER_DOI}",
            citation_key=CITATION_KEY,
            method="Supplementary Data X matrix (log2(TPM + 1)) back-transformed to "
            "TPM; counts from SBRG/modulome_ppu@f63a0df counts.csv, paired by "
            "reproduction",
        ),
    )
    validate: Callable[[Any], object] = TypeAdapter(ExperimentType).validate_python
    report.add(l0_structural((r["experiment"] for r in records), validate))
    report.add(l1_count(len(records), expected_count))

    profiles = Counter(
        json.dumps(r["experiment"]["phenotype"]["expression_count"], sort_keys=True)
        for r in records
    )
    n_repeated = sum(n for n in profiles.values() if n > 1)
    report.add(
        LevelResult(
            level=Level.L1,
            name="sample_uniqueness",
            passed=n_repeated == 0,
            message=f"{len(profiles)} distinct count profiles over {len(records)} records",
            details={"n_records": len(records), "n_in_repeated_profiles": n_repeated},
        )
    )

    tpm_values = [
        float(v)
        for r in records
        for v in r["experiment"]["phenotype"]["expression_tpm"].values()
    ]
    fidelity = l2_value_fidelity(tpm_values, allow_nan=False, minimum=0.0)
    report.add(
        LevelResult(
            level=Level.L2,
            name="tpm_value_fidelity",
            passed=fidelity.passed,
            message=fidelity.message,
            details=fidelity.details,
        )
    )
    bad_counts = sum(
        1
        for r in records
        for v in r["experiment"]["phenotype"]["expression_count"].values()
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
    totals = [
        sum(r["experiment"]["phenotype"]["expression_tpm"].values()) for r in records
    ]
    off_scale = [t for t in totals if not math.isclose(t, TPM_TOTAL, rel_tol=1e-6)]
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

    types = {r["experiment"]["phenotype"]["measurement_type"] for r in records}
    report.add(
        LevelResult(
            level=Level.L3,
            name="measurement_type_consistent",
            passed=types == {MEASUREMENT_TYPE},
            message=f"measurement types {sorted(types)}",
            details={"measurement_types": sorted(types)},
        )
    )
    reference_bad = sum(
        1
        for r in records
        for v in r["reference"]["phenotype_reference"]["expression_tpm"].values()
        if not math.isfinite(float(v)) or float(v) < 0
    )
    report.add(
        LevelResult(
            level=Level.L3,
            name="reference_finite",
            passed=reference_bad == 0,
            message=f"{reference_bad} reference TPMs non-finite or negative",
            details={"n_bad": reference_bad},
        )
    )
    pins = {
        (
            r["reference"]["genome_reference"]["assembly_set"],
            r["reference"]["genome_reference"]["assembly_accession"],
        )
        for r in records
    }
    report.add(
        LevelResult(
            level=Level.L3,
            name="assembly_pin",
            passed=pins == {("pputida_KT2440_ASM756v2", "GCA_000007565.2")},
            message=f"assembly pins {sorted(pins)}",
            details={"pins": sorted(pins)},
        )
    )

    universe = set(gene_universe)
    measured = {
        g for r in records for g in r["experiment"]["phenotype"]["expression_tpm"]
    }
    perturbed = {
        p["systematic_gene_name"]
        for r in records
        for p in r["experiment"]["genotype"]["perturbations"]
    }
    outside = sorted((measured | perturbed) - universe)
    report.add(
        LevelResult(
            level=Level.L4,
            name="gene_containment_kt2440",
            passed=not outside,
            message=f"{len(measured)} measured and {len(perturbed)} perturbed genes; "
            f"{len(outside)} outside the KT2440 locus universe",
            details={"outside": outside[:20], "n_universe": len(universe)},
        )
    )
    return report


def main() -> None:
    """Build or load the dataset and print its size and one record's shape."""
    from dotenv import load_dotenv

    load_dotenv()
    root = osp.join(os.environ["DATA_ROOT"], "data/torchcell/putida_precise321_lim2022")
    dataset = PutidaPrecise321Lim2022Dataset(root=root)
    print(f"len = {len(dataset)}")
    record = dataset[0]
    experiment = record["experiment"]
    print("record[0] genes:", len(experiment["phenotype"]["expression_tpm"]))
    print("record[0] genotype:", experiment["genotype"]["perturbations"])
    print("record[0] genome_reference:", record["reference"]["genome_reference"])
    print("record[0] publication:", record["publication"])


if __name__ == "__main__":
    main()
