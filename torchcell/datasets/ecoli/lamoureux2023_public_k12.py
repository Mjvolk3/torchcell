# torchcell/datasets/ecoli/lamoureux2023_public_k12
# [[torchcell.datasets.ecoli.lamoureux2023_public_k12]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/ecoli/lamoureux2023_public_k12
# Test file: tests/torchcell/datasets/ecoli/test_lamoureux2023_public_k12.py
r"""Public K-12 (Lamoureux 2023): the 1,675 reprocessed public E. coli K-12 RNA-seq samples.

Beside PRECISE-1K (its own lab's 1,035 single-protocol libraries, served by
``RnaseqLamoureux2023Dataset``), Lamoureux et al. 2023 curated every publicly available
*E. coli* K-12 RNA-seq run in the SRA and reprocessed it through the same pipeline:
"Finally, the 0.95 minimum replicate correlation threshold was applied, yielding the
final set of 1675 high-quality publicly-available samples". Those 1,675 samples are OTHER
labs' experiments, so they are a separate dataset from PRECISE-1K, not more of it, and
each record names the SRA experiment accession it came from.

DEDUPLICATION, measured on the release (``preprocess/accession_ledger.json``) and asserted
at build time:

- **The experiment accession is the record identity.** The 1,675 rows carry 1,675 distinct
  SRA/ENA/DDBJ experiment accessions (1,523 ``SRX``, 112 ``ERX``, 40 ``DRX``) and 1,675
  distinct run accessions. The build refuses a repeated accession.
- **``BioSample`` is NOT the dedup key, and the release proves it.** The 1,675 rows map to
  only 1,568 distinct ``BioSample`` accessions: 38 BioSamples carry 2 to 12 rows apiece
  (145 rows in all). Those co-registered rows are not duplicates: 7 of the 38 span several
  conditions or strains under one BioSample (``SAMN12285586`` covers
  ``hyperpersistence:wt``, ``:pth_mut`` and ``:metG_mut``), so a submitter registered one
  BioSample for several libraries. Collapsing on BioSample would merge distinct genotypes.
- **No library is counted twice, and none is a PRECISE-1K library.** Measured on the count
  matrix: 0 of the 1,675 count columns is byte-identical to another, and 0 is identical to
  any of the 1,055 ``data/precise1k/counts.csv`` columns the sibling dataset serves. The
  Public K-12 count file carries no ``p1k_`` column at all. The build asserts both.
- **Against our own store: one BioProject overlaps, no record does.** ``PRJNA645443``
  appears in this table (12 ``phage_resist`` BW25113 RNA-seq samples) and in
  ``PhageRbTnseqMutalik2020Dataset``'s manifest, where it is recorded as holding that
  paper's RAW READS, which the mirror does not keep; that dataset serves RB-TnSeq fitness,
  a different assay. None of the 38 PMIDs and none of the 30 GEO series in this table
  belongs to any other *E. coli* loader (measured against every ``torchcell/datasets/ecoli``
  and ``torchcell/datasets/pputida`` module). ``RnaseqCaglar2017Dataset`` is REL606, an
  *E. coli* B strain the curation discarded: "RNA-seq samples were discarded if the strain
  was not from a K-12 strain".

THE VALUE IS COMPUTED HERE, AND WHY. The paper's Data Availability says "The
$\\log _ { 2 } [ \\mathrm { T P M } ]$ , raw read count, QC data files and sample metadata
for the high-quality public samples may be found in the data directory of this project's
GitHub repository." Measured on the pinned archive and, independently, on the live
repository tree: the raw read counts (``data/k12_modulome/counts.csv``), the QC table
(``multiqc_stats.tsv``) and the metadata are there, and **the log2[TPM] matrix is not**.
There is no ``k12_modulome/log_tpm*.csv`` member, and the three packaged ``IcaData``
objects (``k12_modulome.json.gz``, ``k12_only_p1k_ctrl.json.gz``,
``k12_only_proj_ref.json.gz``) all carry ``X: null`` and ``log_tpm: null``. So
``expression_tpm`` is computed here from the released counts and the release's own gene
coordinates, the ``RnaseqCaglar2017Dataset`` convention for a source that releases counts
and not TPM. It is NOT the paper's TPM, and the release does not let anyone reproduce that:
on PRECISE-1K, where both a count matrix and a log2[TPM] matrix ARE released, this same
derivation misses the released values (max absolute difference 9.48 log2 units; with a
per-sample scale fitted, 88.8% of cells agree to 1e-4 and the implied normalization total
is 1.1% above the sum over the released genes). That measurement is why the derivation is
named in ``measurement_type`` instead of being passed off as the released value.

PHENOTYPE (``RNASeqExpressionPhenotype``, ``measurement_type="rnaseq_tpm_from_released_counts"``).
``expression_count`` is the gene's cell of the release's count matrix, over its own 4,355
genes (the pre-QC-filter gene set, so the 98 short / low-FPKM genes the expression
component dropped are present here). ``expression_tpm`` is ``(count / length) / sum(count /
length) * 1e6``, so a record's TPM sums to exactly one million. Each gene's length is the
span of its row in the release's ``data/annotation/gene_info.csv``, the annotation
featureCounts counted against; 4,313 of the 4,352 spans equal the pinned ASM584v2 GenBank
span of the same locus and 39 differ (``preprocess/gene_length_divergence.json``), which is
why the release's own coordinates are used rather than the genome's. ``n_mapped_reads`` is
the library's featureCounts ``Assigned`` total from ``multiqc_stats.tsv``, which is a read,
not a derivation: it equals the sum of the stored counts for 1,675 of 1,675 samples, and
its minimum over them (533,469) clears the paper's own floor, "At least $5 0 0 ~ 0 0 0$
reads mapped to coding sequences (CDS) from the reference genome (NC 000913.3)".

RECORD GRAIN. One record per sequenced library, as in PRECISE-1K and putidaPRECISE321: the
replicates of a condition share a genotype and an environment, so the RNA-seq verifier's
``strain_uniqueness`` rule cannot apply and the registry entry passes
``replicate_aware=True`` to get ``replicate_groups`` instead.

GENOTYPE SETTLING (first matching rule, counted in ``preprocess/dropped_records.json``).
Only MG1655 is kept: the table names 13 strain cells over 15 K-12 substrains, and writing
another strain's transcriptome against MG1655 is the cross-strain inference plan D9
refuses. Within MG1655, the ``Strain Description`` edit tokens are classified by an
explicit table and an unknown token raises: a plasmid-borne construct, an amino-acid
substitution, a phage infection, an evolved or mutator-evolved isolate, an allele stated
only in prose ("with inactive relA"), a deletion that is not one gene (``del_7prrn``), a
deletion inside a gene (``crp_ar1_ar2``), an undefined background label (``Z1``,
``lacIq``, ``SR``, ``1-2-``, ``hupB+``) and a deletion whose symbol the pinned annotation
resolves to no b-number (``ihf``, which is the IhfA/IhfB heterodimer rather than a gene,
and the ``rrnC`` / ``rrnD`` / ``rrnE`` rRNA operons) are each dropped with their reason.

ENVIRONMENT. Written with the sibling loader's own encoding, so one condition means the
same thing in both datasets: a loader-local ``Media`` derived from the ``M9`` or ``LB``
library key, the carbon and nitrogen sources and the pH as
``EnvironmentPhysicalPerturbation``, each supplement compound as a
``SmallMoleculePerturbation``. The oxygen regime comes from this table's own
``aerobicity`` column (the ``Electron Acceptor`` column PRECISE-1K uses is blank on every
public row), and ``duration_hours`` stays a typed gap: the ``time`` column has no unit in
its header, the paper never mentions it, and its 13 distinct values on the kept rows mix
``12``, ``12:00:00``, ``0.25`` and ``0:00:30``, so neither a unit nor one format can be
sourced. Dropped, because the record could not say what the sample saw: a base medium with
no ``MEDIA_LIBRARY`` key, a non-batch culture, an unstated oxygen regime, an
aerobic-anaerobic transition and a dissolved-oxygen setpoint (neither of which
``Environment.aerobicity`` can express), an unstated temperature or pH, and a minimal
medium naming no carbon or nitrogen source.

REFERENCE. One reference for every record, the condition the release centers this dataset
on: "After centering the Public K-12 dataset to the PRECISE-1K control condition". That is
the two ``control:wt_glc`` samples of PRECISE-1K (wild-type MG1655 in M9 with glucose),
whose counts come from the already-mirrored ``data/precise1k/counts.csv`` and whose
environment is built from the already-mirrored ``data/precise1k/metadata_qc.csv`` by the
sibling loader's own settler, so the reference environment is the same object the sibling
writes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import os.path as osp
import re
import shutil
import zipfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from torchcell.data import (
    ExperimentDataset,
    check_manifest_pin,
    link_verified,
    post_process,
    verify_raw_files,
)
from torchcell.datamodels.schema import (
    BacterialDeletionPerturbation,
    BacterialRNASeqExpressionExperiment,
    BacterialRNASeqExpressionExperimentReference,
    ConcentrationUnit,
    Environment,
    Experiment,
    ExperimentReference,
    Genotype,
    Publication,
    RNASeqExpressionPhenotype,
)
from torchcell.datasets.bacteria_common import (
    LocusTagReconciliation,
    assembly_reference,
    bacterial_genome,
    reconcile_locus_tags,
)
from torchcell.datasets.dataset_registry import register_dataset
from torchcell.datasets.ecoli.lamoureux2023 import (
    ARCHIVE_PREFIX,
    ARCHIVE_SHA256,
    ARCHIVE_URL,
    CITATION_KEY,
    COL_CARBON,
    COL_CULTURE,
    COL_DESCRIPTION,
    COL_FULL_NAME,
    COL_MEDIA,
    COL_NITROGEN,
    COL_PH,
    COL_PROJECT,
    COL_STRAIN,
    COL_SUPPLEMENT,
    COL_TEMPERATURE,
    CONTROL_PROJECT,
    GENE_NAMESPACE,
    GITHUB_RELEASE_URL,
    PAPER_DOI,
    PAPER_TITLE,
    RAW_DIR_REL,
    REFERENCE_STRAIN_NAME,
    ZENODO_DOI,
    ZENODO_RECORD_URL,
    Amount,
    ConditionSpec,
    DropLog,
    DropRule,
    RawFile,
    build_environment,
    parse_amount,
    raw_mirror_dir,
    read_metadata,
    retrieval_record,
    settle_environment,
)
from torchcell.datasets.ecoli.lamoureux2023 import COUNTS as P1K_COUNTS
from torchcell.datasets.ecoli.lamoureux2023 import LIBRARY_BASES as LIBRARY_BASES
from torchcell.datasets.ecoli.lamoureux2023 import METADATA as P1K_METADATA
from torchcell.datasets.ecoli.lamoureux2023 import _paper as _paper
from torchcell.literature.manifest import ROLE_RAW_DATA, ArtifactRecord, Manifest
from torchcell.sequence.genome.ecoli.k12 import EcoliK12Genome, EcoliK12StrainName
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
)

log = logging.getLogger(__name__)

#: The released value is a TPM this loader computed from the released counts, not the
#: paper's unreleased log2[TPM]; the tag says so wherever the records are read.
MEASUREMENT_TYPE = "rnaseq_tpm_from_released_counts"

TPM_TOTAL = 1e6
#: A record's TPM must sum to one million to this many TPM (float summation slack only).
TPM_TOTAL_TOLERANCE = 1e-6

#: The expression genes must resolve to MG1655 loci at this fraction or the build stops.
#: Measured on the release: 4,352 of 4,355 (0.99931); the three that do not (``b3036``,
#: ``b4223``, ``b4590``) are b-numbers the ASM584v2 annotation no longer carries and are
#: kept as given, exactly as the sibling dataset keeps them.
EXPRESSION_MIN_RESOLVED = 0.99

#: Measured: of the 4,352 released genes whose b-number resolves to an ASM584v2 locus, 39
#: have a ``gene_info.csv`` span that differs from the GenBank span of that locus. The
#: build writes them out and refuses a different count, so a changed annotation surfaces.
GENE_LENGTH_DIVERGENCE_N = 39


# --------------------------------------------------------------------------- #
# Raw members (the four this dataset adds to the shared raw mirror)
# --------------------------------------------------------------------------- #
PUBLIC_METADATA = RawFile(
    name="k12_modulome_metadata_qc.csv",
    relpath="data/k12_modulome/metadata_qc.csv",
    sha256="3904fb6db7c30396899dd78b045e0b18358ae0f83dbb33e6984024532e84d53e",
    purpose="Public K-12 sample metadata, 2,710 rows (1,035 PRECISE-1K + 1,675 public) "
    "x 54 columns",
)
PUBLIC_COUNTS = RawFile(
    name="k12_modulome_counts.csv",
    relpath="data/k12_modulome/counts.csv",
    sha256="e60ac8370095e180badc89179a1cfd448c748c2ebf9dde0a31d2dc381318ec20",
    purpose="raw featureCounts read counts of the curated public samples, 4,355 genes x "
    "3,125 SRA experiment accessions (the pre-QC superset of the 1,675)",
)
PUBLIC_MULTIQC = RawFile(
    name="k12_modulome_multiqc_stats.tsv",
    relpath="data/k12_modulome/multiqc_stats.tsv",
    sha256="6cd04f09591e5e73471c0fd033ca9a2f5e9d8c027004dbd041fa7001825dad65",
    purpose="per-sample MultiQC summary of the same 3,125 samples; read for the "
    "featureCounts 'Assigned' total (n_mapped_reads)",
)
GENE_INFO = RawFile(
    name="annotation_gene_info.csv",
    relpath="data/annotation/gene_info.csv",
    sha256="355de1157a4837180cb0c3d625a7a4f26903ba0d50f2bf3ca769949e90dbcb90",
    purpose="the release's own gene annotation, 4,355 rows; read for each gene's start "
    "and end, the span featureCounts counted against",
)
PUBLIC_RAW_FILES: tuple[RawFile, ...] = (
    PUBLIC_METADATA,
    PUBLIC_COUNTS,
    PUBLIC_MULTIQC,
    GENE_INFO,
)
#: The two already-mirrored PRECISE-1K members this dataset reads for its reference.
REFERENCE_RAW_FILES: tuple[RawFile, ...] = (P1K_COUNTS, P1K_METADATA)
#: Every member the mirror must hold for this dataset, each pinned by its own sha256.
CONSUMED_RAW_FILES: tuple[RawFile, ...] = PUBLIC_RAW_FILES + REFERENCE_RAW_FILES


def public_provenance(raw: RawFile, column: str) -> Provenance:
    """A column of one mirrored release file."""
    return Provenance(
        source_uri=raw.relpath,
        citation_key=CITATION_KEY,
        sha256=raw.sha256,
        method=f"cell of the release's {raw.name} (raw mirror {RAW_DIR_REL})",
        page=f"column {column!r}",
    )


def _cell(value: Any, raw: RawFile, column: str) -> SourcedValue:
    """Bind a value to a verbatim cell of one mirrored release column."""
    return SourcedValue(
        value=value, quote=str(value), provenance=public_provenance(raw, column)
    )


# --------------------------------------------------------------------------- #
# Sourced constants (verbatim quotes, pinned sha256)
# --------------------------------------------------------------------------- #
CURATION = _paper(
    1675,
    "Finally, the 0.95 minimum replicate correlation threshold was applied, yielding "
    "the final set of 1675 high-quality publicly-available samples",
    page="Methods, 'Compiling the public K-12 dataset'",
)
CURATION_STRAIN_FILTER = _paper(
    "K-12 only",
    "RNA-seq samples were discarded if the strain was not from a K-12 strain, if the "
    "strain was missing, or if the type of experiment was not actually RNA-seq.",
    page="Methods, 'Compiling the public K-12 dataset'",
)
REPLICATE_RULE = _paper(
    2,
    "Only conditions with at least two biological replicates were kept at this step.",
    page="Methods, 'Compiling the public K-12 dataset'",
)
COMBINED = _paper(
    2710,
    "Next, these 1675 samples were combined with the 1035 samples of PRECISE-1K to "
    "yield the ‘Public K-12’ dataset, comprising 2710 curated, high quality expression "
    "profiles",
    page="Methods, 'Compiling the public K-12 dataset'",
)
SUBSTRAINS = _paper(
    15,
    "These profiles come from 134 different projects, including $1 5 ~ \\mathrm { K } "
    "\\ – \\bar { 1 } 2$ substrains and 9 distinct temperatures and pHs.",
    page="Results, 'Incorporating 1675 high-quality publicly-available transcriptomes'",
)
RELEASED_FILES = _paper(
    "log2[TPM], raw read count, QC data, metadata",
    "The $\\log _ { 2 } [ \\mathrm { T P M } ]$ , raw read count, QC data files and "
    "sample metadata for the high-quality public samples may be found in the data "
    "directory of this project’s GitHub repository.",
    page="Methods, 'Compiling the public K-12 dataset'",
    note="measured 2026-10-07 on the pinned archive and on the live repository tree: "
    "the counts, the MultiQC table and the metadata are released; no "
    "k12_modulome/log_tpm*.csv member exists and k12_modulome.json.gz, "
    "k12_only_p1k_ctrl.json.gz and k12_only_proj_ref.json.gz all carry X: null and "
    "log_tpm: null, so the log2[TPM] matrix is not released",
)
CENTERING = _paper(
    "PRECISE-1K control condition",
    "After centering the Public K-12 dataset to the PRECISE-1K control condition",
    page="Methods, 'Compiling the public K-12 dataset'",
)
READ_DEPTH_FLOOR = _paper(
    500000,
    "At least $5 0 0 ~ 0 0 0$ reads mapped to coding sequences (CDS) from the reference "
    "genome (NC 000913.3)",
    page="Methods, 'RNA-seq processing and quality control', QC criteria",
    note="measured over the 1,675 kept samples: the featureCounts 'Assigned' total "
    "ranges 533,469 to 60,616,247 and equals the sum of the stored counts for all of "
    "them",
)
COUNTING = _paper(
    "featureCounts -p -B -C -P -fracOverlap 0.5",
    "The read direction was inferred using RSEQC (24) before generating read counts "
    "using featureCounts (25) with the following non-default options: -p -B -C -P "
    "-fracOverlap 0.5.",
    page="Methods, 'RNA-seq processing and quality control'",
)

#: The ``time`` column has no unit in its header, the paper never names the column, and
#: its values mix a bare number with an H:MM:SS clock, so neither a unit nor one format
#: can be sourced. Rank 19 of the E. coli SI audit: the unit is itself the gap.
DURATION_GAP = ProvenanceGap(
    field="duration_hours",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=public_provenance(PUBLIC_METADATA, "time"),
    note="the 'time' column carries no unit in its header and is never mentioned in the "
    "paper or the supplementary file; measured on the kept rows, its 13 distinct values "
    "mix bare numbers ('12', '0.25', '0.08') with an H:MM:SS clock ('12:00:00', "
    "'0:00:30'), so a duration in hours would be a guess at both the unit and the format",
)
#: The pH column names no titrant, exactly as in PRECISE-1K.
PH_AGENT_GAP = ProvenanceGap(
    field="agent",
    reason=ProvenanceGapReason.not_reported_by_primary,
    looked_in=public_provenance(PUBLIC_METADATA, "pH"),
    note="the release states the pH value only; the acid or base that set it is named "
    "only where it is itself a listed supplement",
)


# --------------------------------------------------------------------------- #
# Metadata columns this dataset reads beyond the shared PRECISE-1K ones
# --------------------------------------------------------------------------- #
COL_SAMPLE_LABEL = "sample_id"
COL_REP = "rep_id"
COL_RUN = "Run"
COL_BIOPROJECT = "BioProject"
COL_BIOSAMPLE = "BioSample"
COL_SCIENTIFIC_NAME = "ScientificName"
COL_GEO_SERIES = "GEO"
COL_GEO_SAMPLE = "GEO Sample"
COL_PMID = "PMID"
COL_AEROBICITY = "aerobicity"
COL_TIME = "time"
COL_REFERENCE_CONDITION = "reference_condition"

#: The release's own sample id is a PRECISE-1K library exactly when it has this prefix;
#: every other row is one of the 1,675 reprocessed public samples.
P1K_PREFIX = "p1k_"
#: An SRA / ENA / DDBJ experiment accession, which is this dataset's record identity.
_ACCESSION = re.compile(r"^(?:SRX|ERX|DRX)\d+$")

#: Columns of ``multiqc_stats.tsv`` this loader reads.
COL_MULTIQC_SAMPLE = "Sample"
COL_MULTIQC_ASSIGNED = "Assigned"

#: Columns of ``gene_info.csv`` this loader reads.
COL_GENE_START = "start"
COL_GENE_END = "end"


# --------------------------------------------------------------------------- #
# Genotype settling
# --------------------------------------------------------------------------- #
class PublicGenotypeRule(StrEnum):
    """How one public sample's genotype was settled; the first matching rule wins."""

    wild_type = "wild_type"
    deletion = "deletion"
    strain_not_mg1655 = "strain_not_mg1655"
    plasmid_borne_construct = "plasmid_borne_construct"
    point_mutation_allele = "point_mutation_allele"
    phage_infected = "phage_infected"
    evolved_isolate = "evolved_isolate"
    allele_described_in_prose = "allele_described_in_prose"
    deletion_not_one_gene = "deletion_not_one_gene"
    partial_gene_edit = "partial_gene_edit"
    background_label_undefined = "background_label_undefined"
    deletion_symbol_unresolved = "deletion_symbol_unresolved"


KEPT_GENOTYPE_RULES = frozenset(
    {PublicGenotypeRule.wild_type, PublicGenotypeRule.deletion}
)

PUBLIC_GENOTYPE_RULE_REASONS: dict[PublicGenotypeRule, str] = {
    PublicGenotypeRule.strain_not_mg1655: (
        "the 'Strain' cell is not MG1655: the curation kept 15 K-12 substrains and the "
        "expression is keyed by MG1655 b-numbers, so writing another strain's "
        "transcriptome against the MG1655 assembly set would be the cross-strain "
        "inference plan D9 refuses (BW25113 has its own set; W3110, AR3110, MC4100, "
        "JM109, PFM2, LE392, CV104 and the lysogens have none in the genomes tier)"
    ),
    PublicGenotypeRule.plasmid_borne_construct: (
        "a plasmid-borne construct (pACYC184, pBAD24, pBR322, pBRplac, pCA24N, pEF21, "
        "pJV300, pKK3535, pNT3, pNW129, pORF1, pSEVA, pTP-011, pUUH239.2, pZA12, "
        "pdCas9 or a p_ Methanothermus histone vector): the construct, its source "
        "organism and its coding sequence are not in the release"
    ),
    PublicGenotypeRule.point_mutation_allele: (
        "an amino-acid substitution or a classical point allele (cpxA F218Y, crp K100Q/R, "
        "cyaA N600Y, fusA A608E, hupA E34K, rpoD L261Q, topA S180L, dnaA46): the "
        "bacterial schema has no allele leaf for a substitution"
    ),
    PublicGenotypeRule.phage_infected: (
        "a phage-infected or phage-complemented culture (T7, T4, Svi3-3): an infecting "
        "phage is not a genomic edit of the host and the release gives no titer or "
        "timing per sample"
    ),
    PublicGenotypeRule.evolved_isolate: (
        "an evolved or mutator-evolved isolate ('CHL evolved', 'AMX evolved', "
        "'del_mutS Mutated'): its acquired mutations are not in the release"
    ),
    PublicGenotypeRule.allele_described_in_prose: (
        "an allele stated only in prose ('with inactive relA', 'with truncated relA'): "
        "the release names neither the lesion nor its coordinates"
    ),
    PublicGenotypeRule.deletion_not_one_gene: (
        "a deletion that is not one gene ('del_7prrn', the seven rRNA operons): the "
        "release gives no member list, and a whole-gene deletion would misstate it"
    ),
    PublicGenotypeRule.partial_gene_edit: (
        "a deletion of a region inside a gene (crp activating regions AR1/AR2): the "
        "release gives no coordinates"
    ),
    PublicGenotypeRule.background_label_undefined: (
        "a background label the release and the paper do not define ('Z1', 'lacIq', "
        "'SR', '1-2-', 'hupB+')"
    ),
    PublicGenotypeRule.deletion_symbol_unresolved: (
        "a deletion whose gene symbol the pinned ASM584v2 annotation resolves to no "
        "b-number: 'ihf' names the IhfA/IhfB heterodimer rather than a gene, and "
        "'rrnC', 'rrnD' and 'rrnE' name rRNA operons, not loci"
    ),
}

WILD_TYPE_DESCRIPTION = "MG1655"
_DELETION_TOKEN = re.compile(r"^del_?(?P<gene>[a-z]{3}[A-Z0-9]?)$")

#: Deletion symbols the pinned annotation resolves to no b-number (measured through
#: ``reconcile_locus_tags``: 37 of the 41 released symbols resolve at the gene-symbol
#: layer and these four do not). A row naming one of them is dropped, never guessed.
UNRESOLVED_DELETION_SYMBOLS = frozenset({"ihf", "rrnC", "rrnD", "rrnE"})

#: Every non-deletion edit token of the MG1655 rows, each bound to the rule it triggers.
#: Measured: 128 distinct tokens over the 1,006 MG1655 ``Strain Description`` cells, 59
#: of them ``del_<gene>``. An unlisted token raises rather than being guessed at.
PUBLIC_EDIT_TOKENS: dict[str, PublicGenotypeRule] = {
    token: PublicGenotypeRule.plasmid_borne_construct
    for token in (
        "pACYC184_dcm",
        "pACYC184_ecoP15Imod",
        "pACYC184_ecoP1Imod",
        "pACYC184_empty",
        "pBAD24",
        "pBAD24_symE",
        "pBR322",
        "pBR322_csrA",
        "pBR322_csrB",
        "pBR322_csrD",
        "pBRplac",
        "pBRplac_MicA",
        "pCA24N,-gfp/DA4201",
        "pEF21_Hfq",
        "pJV300",
        "pJV300_cyaR",
        "pKK3535-BBB",
        "pKK3535-HBB",
        "pNT3_aroP",
        "pNT3_bamC",
        "pNT3_cusA",
        "pNT3_empty",
        "pNT3_hemX",
        "pNT3_topA",
        "pNT3_yicR",
        "pNT3_yqjD",
        "pNW129",
        "pNW129_motB",
        "pORF1",
        "pSEVA_crp",
        "pTP-011",
        "pUUH239.2",
        "pUUH239.2_no_ctx-m-15",
        "pUUH239.2_no_res",
        "pUUH239.2_no_tetA",
        "pUUH239.2_no_tetAR",
        "pUUH239.2_no_tetR",
        "pZA12_gcvB",
        "p_HMfA",
        "p_HMfB",
        "p_empty",
        "p_nbHMfA",
        "p_nbHMfB",
        "pdCas9-bacteria",
    )
} | {
    "cpxAF218Y": PublicGenotypeRule.point_mutation_allele,
    "crpK100Q": PublicGenotypeRule.point_mutation_allele,
    "crpK100R": PublicGenotypeRule.point_mutation_allele,
    "cyaAN600Y": PublicGenotypeRule.point_mutation_allele,
    "dnaA46": PublicGenotypeRule.point_mutation_allele,
    "fusAA608E": PublicGenotypeRule.point_mutation_allele,
    "hupAE34K": PublicGenotypeRule.point_mutation_allele,
    "rpoDL261Q": PublicGenotypeRule.point_mutation_allele,
    "topAS180L": PublicGenotypeRule.point_mutation_allele,
    "infect_T7": PublicGenotypeRule.phage_infected,
    "T4_infect": PublicGenotypeRule.phage_infected,
    "Svi3-3": PublicGenotypeRule.phage_infected,
    "comp.": PublicGenotypeRule.phage_infected,
    "evolved": PublicGenotypeRule.evolved_isolate,
    "AMX": PublicGenotypeRule.evolved_isolate,
    "CHL": PublicGenotypeRule.evolved_isolate,
    "Mutated": PublicGenotypeRule.evolved_isolate,
    "with": PublicGenotypeRule.allele_described_in_prose,
    "inactive": PublicGenotypeRule.allele_described_in_prose,
    "truncated": PublicGenotypeRule.allele_described_in_prose,
    "relA": PublicGenotypeRule.allele_described_in_prose,
    "del_7prrn": PublicGenotypeRule.deletion_not_one_gene,
    "crp_ar1_ar2": PublicGenotypeRule.partial_gene_edit,
    "Z1": PublicGenotypeRule.background_label_undefined,
    "lacIq": PublicGenotypeRule.background_label_undefined,
    "SR": PublicGenotypeRule.background_label_undefined,
    "1-2-": PublicGenotypeRule.background_label_undefined,
    "hupB+": PublicGenotypeRule.background_label_undefined,
}

#: The order the non-deletion rules are applied in, so a row carrying two kinds of edit
#: is reported under the one that makes it unwritable first.
PUBLIC_RULE_ORDER: tuple[PublicGenotypeRule, ...] = (
    PublicGenotypeRule.evolved_isolate,
    PublicGenotypeRule.phage_infected,
    PublicGenotypeRule.plasmid_borne_construct,
    PublicGenotypeRule.point_mutation_allele,
    PublicGenotypeRule.allele_described_in_prose,
    PublicGenotypeRule.deletion_not_one_gene,
    PublicGenotypeRule.partial_gene_edit,
    PublicGenotypeRule.background_label_undefined,
)


class PublicGenotypeVerdict(BaseModel):
    """The settled genotype of one public sample, or the rule that dropped it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rule: PublicGenotypeRule
    deleted_symbols: tuple[str, ...] = ()

    @property
    def kept(self) -> bool:
        """True for wild type or whole-gene deletions."""
        return self.rule in KEPT_GENOTYPE_RULES


def settle_public_genotype(row: Mapping[str, str]) -> PublicGenotypeVerdict:
    """Settle one public metadata row's genotype against MG1655 (first matching rule).

    ``row`` holds the verbatim cells (blank = ""). A ``Strain Description`` that does not
    start with its ``Strain`` cell, or an edit token of no listed form, raises rather
    than being guessed at.
    """
    if row[COL_STRAIN] != REFERENCE_STRAIN_NAME:
        return PublicGenotypeVerdict(rule=PublicGenotypeRule.strain_not_mg1655)
    description = row[COL_DESCRIPTION]
    if description == WILD_TYPE_DESCRIPTION:
        return PublicGenotypeVerdict(rule=PublicGenotypeRule.wild_type)
    if not description.startswith(f"{WILD_TYPE_DESCRIPTION} "):
        raise ValueError(f"unrecognized Strain Description {description!r}")
    tokens = description[len(WILD_TYPE_DESCRIPTION) + 1 :].split()
    unknown = [
        t
        for t in tokens
        if t not in PUBLIC_EDIT_TOKENS and _DELETION_TOKEN.match(t) is None
    ]
    if unknown:
        raise ValueError(
            f"Strain Description {description!r}: edit tokens {unknown} are of no "
            "listed form"
        )
    rules = {PUBLIC_EDIT_TOKENS[t] for t in tokens if t in PUBLIC_EDIT_TOKENS}
    for rule in PUBLIC_RULE_ORDER:
        if rule in rules:
            return PublicGenotypeVerdict(rule=rule)
    genes = [
        match.group("gene")
        for match in (_DELETION_TOKEN.match(t) for t in tokens)
        if match is not None
    ]
    if len(set(genes)) != len(genes):
        raise ValueError(f"Strain Description {description!r} repeats a deletion")
    if set(genes) & UNRESOLVED_DELETION_SYMBOLS:
        return PublicGenotypeVerdict(
            rule=PublicGenotypeRule.deletion_symbol_unresolved,
            deleted_symbols=tuple(genes),
        )
    return PublicGenotypeVerdict(
        rule=PublicGenotypeRule.deletion, deleted_symbols=tuple(genes)
    )


# --------------------------------------------------------------------------- #
# Environment settling
# --------------------------------------------------------------------------- #
class PublicEnvironmentRule(StrEnum):
    """Why a genotype-kept public sample's environment could not be written."""

    medium_not_in_library = "medium_not_in_library"
    culture_not_batch = "culture_not_batch"
    oxygen_regime_not_stated = "oxygen_regime_not_stated"
    oxygen_regime_transition = "oxygen_regime_transition"
    oxygen_setpoint_not_expressible = "oxygen_setpoint_not_expressible"
    temperature_not_stated = "temperature_not_stated"
    ph_not_stated = "ph_not_stated"
    minimal_medium_source_not_stated = "minimal_medium_source_not_stated"


PUBLIC_ENVIRONMENT_RULE_REASONS: dict[PublicEnvironmentRule, str] = {
    PublicEnvironmentRule.medium_not_in_library: (
        "the 'Base Media' label has no MEDIA_LIBRARY key (MOPS, NM3, L-broth, Gutnick, "
        "tryptone broth, minimal medium A, M4 minimal, Mueller-Hinton, LB-Miller, "
        "'LB,MOPS', 'M9 RDM', 'MOPS w/o KH2PO4') or is blank, and no mirrored source "
        "states its recipe, so the medium would be free text that joins nothing"
    ),
    PublicEnvironmentRule.culture_not_batch: (
        "the 'Culture Type' cell is not batch (bioreactor, blank, or a growth phase "
        "written into the culture-type column): Environment has no culture-type slot, "
        "so the record would merge with batch records of the same medium"
    ),
    PublicEnvironmentRule.oxygen_regime_not_stated: (
        "the 'aerobicity' cell is blank, and Environment.aerobicity cannot be a typed "
        "gap, so the oxygen regime would be a guess"
    ),
    PublicEnvironmentRule.oxygen_regime_transition: (
        "the 'aerobicity' cell is 'transition' (the anaerobic-aerobic transition "
        "project): one Environment states one regime, so a regime that changes during "
        "the culture cannot be written"
    ),
    PublicEnvironmentRule.oxygen_setpoint_not_expressible: (
        "the 'aerobicity' cell is 'aerobic(30% DO)': a dissolved-oxygen setpoint has no "
        "slot anywhere on Environment, and dropping the setpoint would merge the record "
        "with an unregulated aerobic culture"
    ),
    PublicEnvironmentRule.temperature_not_stated: (
        "the 'Temperature (C)' cell is blank; temperature is part of every sibling "
        "record's environment identity, so a gapped one would not be comparable"
    ),
    PublicEnvironmentRule.ph_not_stated: (
        "the 'pH' cell is blank; pH is an EnvironmentPhysicalPerturbation on every "
        "PRECISE-1K record, so a record without one would not be comparable"
    ),
    PublicEnvironmentRule.minimal_medium_source_not_stated: (
        "an M9 row whose carbon or nitrogen source cell is blank: a minimal medium with "
        "no stated source does not say what the cells grew on"
    ),
}

BATCH = "batch"
#: ``aerobicity`` cell -> the oxygen regime, or the rule that drops the row. 'O2' is the
#: electron acceptor token PRECISE-1K's own ``Electron Acceptor`` column uses for an
#: aerobic culture, reused in this table's ``aerobicity`` column.
AEROBICITY_CELLS: dict[str, Literal["aerobic", "anaerobic"] | PublicEnvironmentRule] = {
    "aerobic": "aerobic",
    "O2": "aerobic",
    "anaerobic": "anaerobic",
    "": PublicEnvironmentRule.oxygen_regime_not_stated,
    "transition": PublicEnvironmentRule.oxygen_regime_transition,
    "aerobic(30% DO)": PublicEnvironmentRule.oxygen_setpoint_not_expressible,
}

#: Unit tokens the public cells use that PRECISE-1K's own cells never do, restated in the
#: enum's units by exact decimal arithmetic: g/L is its own unit, and 100 ng/mL is
#: 0.1 ug/mL.
EXTRA_UNIT_TOKENS: dict[str, tuple[ConcentrationUnit, float]] = {
    "g/L": (ConcentrationUnit.g_per_l, 1.0),
    "ng/mL": (ConcentrationUnit.ug_per_ml, 0.001),
}

#: The separators the public ``Supplement`` cells use, in the order they are applied. A
#: bare comma is NOT one: "2,2'-Dipyridyl(200uM)" names one compound.
_SUPPLEMENT_SEPARATORS = (";", ", ")


def parse_public_supplements(cell: str) -> tuple[Amount, ...]:
    """Every compound of a public ``Supplement`` cell.

    Measured on the release: the public cells separate compounds with ``;`` or with
    ``, ``, never with the ``+`` PRECISE-1K uses, and one compound name carries a comma
    with no space after it.
    """
    if not cell:
        return ()
    parts = [cell]
    for separator in _SUPPLEMENT_SEPARATORS:
        parts = [piece for part in parts for piece in part.split(separator)]
    return tuple(
        parse_amount(part, default_unit=None, extra_units=EXTRA_UNIT_TOKENS)
        for part in parts
    )


class PublicEnvironmentVerdict(BaseModel):
    """A parsed public condition, or the rule that dropped the sample."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rule: PublicEnvironmentRule | None = None
    spec: ConditionSpec | None = None


def settle_public_environment(row: Mapping[str, str]) -> PublicEnvironmentVerdict:
    """Parse one public metadata row's environment, or name the rule that drops it."""
    if row[COL_MEDIA] not in LIBRARY_BASES:
        return PublicEnvironmentVerdict(
            rule=PublicEnvironmentRule.medium_not_in_library
        )
    if row[COL_CULTURE].lower() != BATCH:
        return PublicEnvironmentVerdict(rule=PublicEnvironmentRule.culture_not_batch)
    aerobicity = AEROBICITY_CELLS[row[COL_AEROBICITY]]
    if isinstance(aerobicity, PublicEnvironmentRule):
        return PublicEnvironmentVerdict(rule=aerobicity)
    if not row[COL_TEMPERATURE]:
        return PublicEnvironmentVerdict(
            rule=PublicEnvironmentRule.temperature_not_stated
        )
    if not row[COL_PH]:
        return PublicEnvironmentVerdict(rule=PublicEnvironmentRule.ph_not_stated)
    carbon = row[COL_CARBON]
    nitrogen = row[COL_NITROGEN]
    if row[COL_MEDIA] == "M9" and not (carbon and nitrogen):
        return PublicEnvironmentVerdict(
            rule=PublicEnvironmentRule.minimal_medium_source_not_stated
        )
    return PublicEnvironmentVerdict(
        spec=ConditionSpec(
            base_media=row[COL_MEDIA],
            temperature_c=float(row[COL_TEMPERATURE]),
            ph=float(row[COL_PH]),
            carbon=parse_amount(
                carbon,
                default_unit=ConcentrationUnit.g_per_l,
                extra_units=EXTRA_UNIT_TOKENS,
            )
            if carbon
            else None,
            nitrogen=parse_amount(
                nitrogen,
                default_unit=ConcentrationUnit.g_per_l,
                extra_units=EXTRA_UNIT_TOKENS,
            )
            if nitrogen
            else None,
            aerobicity=aerobicity,
            electron_acceptor=None,
            trace=None,
            supplements=parse_public_supplements(row[COL_SUPPLEMENT]),
            antibiotic=None,
        )
    )


def build_public_environment(spec: ConditionSpec) -> Environment:
    """The sibling loader's environment encoding with this dataset's typed gaps."""
    environment = build_environment(spec)
    return Environment(
        media=environment.media,
        temperature=environment.temperature,
        perturbations=environment.perturbations,
        aerobicity=environment.aerobicity,
        provenance_gaps=[DURATION_GAP],
    )


# --------------------------------------------------------------------------- #
# Expression values
# --------------------------------------------------------------------------- #
class GeneLengthDivergence(BaseModel):
    """One gene whose release span differs from its pinned-assembly span."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    released_name: str
    locus_tag: str
    release_span_bp: int
    assembly_span_bp: int


def gene_lengths(
    annotation: pd.DataFrame, released_names: Sequence[str]
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Each released gene's span in the release's own annotation, in base pairs."""
    rows = annotation.loc[list(released_names)]
    spans = rows[COL_GENE_END].to_numpy(dtype=np.int64) - rows[COL_GENE_START].to_numpy(
        dtype=np.int64
    )
    return np.asarray(spans + 1, dtype=np.float64)


def gene_length_divergence(
    released_names: Sequence[str],
    stored_tags: Sequence[str],
    lengths: np.ndarray[Any, np.dtype[np.float64]],
    genome: EcoliK12Genome,
) -> list[GeneLengthDivergence]:
    """Released spans that differ from the pinned assembly's span of the same locus."""
    divergence: list[GeneLengthDivergence] = []
    for name, tag, length in zip(released_names, stored_tags, lengths, strict=True):
        if tag not in genome.genbank.loci:
            continue
        locus = genome.genbank.loci[tag]
        assembly = locus.end - locus.start + 1
        if int(length) != assembly:
            divergence.append(
                GeneLengthDivergence(
                    released_name=name,
                    locus_tag=tag,
                    release_span_bp=int(length),
                    assembly_span_bp=assembly,
                )
            )
    return divergence


def tpm(
    counts: np.ndarray[Any, np.dtype[np.float64]],
    lengths: np.ndarray[Any, np.dtype[np.float64]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Transcripts per million: counts per base, scaled to sum to one million."""
    rate = counts / lengths
    return np.asarray(rate / rate.sum() * TPM_TOTAL, dtype=np.float64)


def rnaseq_phenotype(
    genes: Sequence[str],
    counts: np.ndarray[Any, np.dtype[np.int64]],
    lengths: np.ndarray[Any, np.dtype[np.float64]],
    n_mapped_reads: int,
) -> RNASeqExpressionPhenotype:
    """One library's computed TPM, its released counts and its featureCounts total."""
    values = tpm(counts.astype(np.float64), lengths)
    if abs(float(values.sum()) - TPM_TOTAL) > TPM_TOTAL_TOLERANCE:
        raise RuntimeError(f"TPM sums to {values.sum()}, not {TPM_TOTAL}")
    return RNASeqExpressionPhenotype(
        expression_tpm=dict(zip(genes, (float(v) for v in values), strict=True)),
        expression_count={g: int(c) for g, c in zip(genes, counts, strict=True)},
        measurement_type=MEASUREMENT_TYPE,
        n_mapped_reads=n_mapped_reads,
    )


def reference_phenotype(
    genes: Sequence[str],
    counts: np.ndarray[Any, np.dtype[np.int64]],
    lengths: np.ndarray[Any, np.dtype[np.float64]],
    gap: ProvenanceGap,
) -> RNASeqExpressionPhenotype:
    """The control condition's mean TPM and mean count (rounded half to even)."""
    profiles = np.stack(
        [tpm(counts[:, j].astype(np.float64), lengths) for j in range(counts.shape[1])]
    )
    return RNASeqExpressionPhenotype(
        expression_tpm=dict(
            zip(genes, (float(v) for v in profiles.mean(axis=0)), strict=True)
        ),
        expression_count={
            g: int(c) for g, c in zip(genes, np.rint(counts.mean(axis=1)), strict=True)
        },
        measurement_type=MEASUREMENT_TYPE,
        provenance_gaps=[gap],
    )


# --------------------------------------------------------------------------- #
# Raw mirror (the loader reads the mirror, never the live Zenodo URL)
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    """``DATA_ROOT`` from the environment (the mirror + build tree live under it)."""
    return os.environ["DATA_ROOT"]


def _sha256(path: str | Path) -> str:
    """Streaming sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deposit_public_raw_mirror(
    *,
    archive_path: str | Path,
    retrieved_at: str = "2026-10-07",
    data_root: str | None = None,
) -> Path:
    """Add the four Public K-12 members of the release archive to the shared raw mirror.

    The archive must carry ``ARCHIVE_SHA256`` and each member its pinned sha256. The
    manifest is rewritten ADDITIVELY: every record already in it that this deposit does
    not write is kept, so the sibling's PRECISE-1K pins survive and one ``manifest.json``
    pins every member either loader consumes. Idempotent by sha256: an existing mirror
    file with the recorded hash is left alone and a differing one raises rather than being
    overwritten.
    """
    got = _sha256(archive_path)
    if got != ARCHIVE_SHA256:
        raise RuntimeError(
            f"{archive_path} sha256 mismatch: got {got}, expected {ARCHIVE_SHA256}"
        )
    root = raw_mirror_dir(data_root)
    with zipfile.ZipFile(archive_path) as archive:
        for raw in PUBLIC_RAW_FILES:
            data = archive.read(raw.member)
            member_sha = hashlib.sha256(data).hexdigest()
            if member_sha != raw.sha256:
                raise RuntimeError(
                    f"{raw.member} sha256 mismatch: got {member_sha}, expected "
                    f"{raw.sha256}"
                )
            dest = root / raw.relpath
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                if _sha256(dest) != raw.sha256:
                    raise RuntimeError(
                        f"{dest} exists with a different sha256; refusing"
                    )
            else:
                tmp = dest.with_suffix(dest.suffix + ".partial")
                tmp.write_bytes(data)
                shutil.move(str(tmp), str(dest))
    written = {raw.relpath: raw for raw in PUBLIC_RAW_FILES}
    manifest_path = root / "manifest.json"
    kept = [
        record
        for record in (
            Manifest.model_validate_json(manifest_path.read_text()).files
            if manifest_path.exists()
            else []
        )
        if record.path not in written
    ]
    files = kept + [
        ArtifactRecord(
            path=raw.relpath,
            role=ROLE_RAW_DATA,
            bytes=(root / raw.relpath).stat().st_size,
            sha256=raw.sha256,
            source=f"{ARCHIVE_URL}#{raw.member}",
            retrieval=retrieval_record(raw, retrieved_at),
        )
        for raw in PUBLIC_RAW_FILES
    ]
    manifest = Manifest(
        citation_key=CITATION_KEY,
        doi=PAPER_DOI,
        title=PAPER_TITLE,
        files=files,
        si_data_sources=[
            f"https://doi.org/{ZENODO_DOI}",
            ZENODO_RECORD_URL,
            GITHUB_RELEASE_URL,
            ARCHIVE_URL,
        ],
        si_expected=[
            "Zenodo 10.5281/zenodo.8284223 (SBRG/precise1k v1.0): the four consumed "
            "members of data/precise1k/, the three consumed members of "
            "data/k12_modulome/ and data/annotation/gene_info.csv are mirrored; the "
            "rest of the archive (iModulon matrices, notebooks, the SRA curation "
            "intermediates) is not consumed and not mirrored",
            "data/k12_modulome/log_tpm*.csv does not exist: measured on this archive "
            "and on the live repository tree, the Public K-12 log2[TPM] matrix the "
            "paper's Data Availability names is not released, and the packaged IcaData "
            "objects carry X: null and log_tpm: null",
        ],
        provenance_complete=True,
        created_at=datetime.now(UTC).isoformat(),
    )
    manifest_path.write_text(manifest.model_dump_json(indent=2))
    return root


def load_manifest(data_root: str | None = None) -> Manifest:
    """Read the shared raw mirror's ``manifest.json``."""
    path = raw_mirror_dir(data_root) / "manifest.json"
    return Manifest.model_validate_json(path.read_text())


def manifest_sha256(manifest: Manifest, relpath: str) -> str:
    """The recorded sha256 of one mirror file."""
    for record in manifest.files:
        if record.path == relpath:
            return record.sha256
    raise KeyError(f"{relpath} is not in the raw-mirror manifest")


# --------------------------------------------------------------------------- #
# Build bookkeeping
# --------------------------------------------------------------------------- #
class AccessionRecord(BaseModel):
    """One kept library's identity and origin, as the release records it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_accession: str
    run_accession: str
    biosample: str
    bioproject: str
    scientific_name: str
    sample_label: str
    full_name: str
    replicate: str
    geo_series: str | None
    geo_sample: str | None
    pmid: str | None
    record_index: int


class AccessionLedger(BaseModel):
    """The deduplication evidence of one build (checklist item 7)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_rows: int
    public_rows: int
    p1k_rows: int
    distinct_experiment_accessions: int
    distinct_run_accessions: int
    distinct_biosamples: int
    biosamples_with_several_rows: int
    rows_sharing_a_biosample: int
    biosamples_spanning_several_conditions: int
    identical_count_profiles_within_public: int
    identical_count_profiles_against_precise1k: int
    distinct_bioprojects: int
    distinct_pmids: int
    distinct_geo_series: int
    records: list[AccessionRecord]


class ReplicateGroup(BaseModel):
    """The kept records of one ``project:condition``: its biological replicates."""

    model_config = ConfigDict(extra="forbid")

    full_name: str
    condition_cells: dict[str, str]
    experiment_accessions: list[str]
    record_indices: list[int]


#: The columns that define a public sample's strain and condition; every replicate of one
#: ``full_name`` must carry the same cells (checked at build).
CONDITION_COLUMNS: tuple[str, ...] = (
    COL_DESCRIPTION,
    COL_STRAIN,
    COL_CULTURE,
    COL_MEDIA,
    COL_TEMPERATURE,
    COL_PH,
    COL_CARBON,
    COL_NITROGEN,
    COL_SUPPLEMENT,
    COL_AEROBICITY,
)


def genotype_class(verdict: PublicGenotypeVerdict) -> str:
    """``wild_type`` or ``deletion_<n>`` for a kept genotype."""
    if verdict.rule is PublicGenotypeRule.wild_type:
        return "wild_type"
    return f"deletion_{len(verdict.deleted_symbols)}"


def public_rows(metadata: pd.DataFrame) -> list[str]:
    """The release ids of the reprocessed public samples, sorted.

    The Public K-12 table is the union of the two arms; a ``p1k_`` id is a PRECISE-1K
    library that ``RnaseqLamoureux2023Dataset`` already serves, so it is not a record
    here. Every remaining id must be an SRA, ENA or DDBJ experiment accession.
    """
    ids = sorted(str(i) for i in metadata.index if not str(i).startswith(P1K_PREFIX))
    bad = [i for i in ids if _ACCESSION.match(i) is None]
    if bad:
        raise RuntimeError(f"public rows with no experiment accession: {bad[:5]}")
    return ids


def _optional(cell: str) -> str | None:
    """A metadata cell, or ``None`` when the release leaves it blank."""
    return cell or None


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
@register_dataset
class RnaseqPublicK12Lamoureux2023Dataset(ExperimentDataset):
    """Public K-12: one record per reprocessed public MG1655 RNA-seq library."""

    REFERENCE_STRAIN: ClassVar[EcoliK12StrainName] = REFERENCE_STRAIN_NAME

    def __init__(
        self,
        root: str = "data/torchcell/rnaseq_public_k12_lamoureux2023",
        io_workers: int = 0,
        ecoli_genome: EcoliK12Genome | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize; the MG1655 genome is injected by the build or opened in process."""
        self.ecoli_genome = ecoli_genome
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
        """The six release members required before processing."""
        return [raw.name for raw in CONSUMED_RAW_FILES]

    def download(self) -> None:
        """Link each mirror file into ``raw/`` after verifying it against its pin.

        The mirror + the pins are canonical; the Zenodo URL is retrieval metadata that
        ``deposit_public_raw_mirror`` re-runs, never a live build dependency. The
        manifest must carry each pin, else ``ManifestPinMismatchError`` names both
        digests.
        """
        data_root = _data_root()
        manifest = load_manifest(data_root)
        os.makedirs(self.raw_dir, exist_ok=True)
        for raw in CONSUMED_RAW_FILES:
            check_manifest_pin(
                raw.relpath, manifest_sha256(manifest, raw.relpath), raw.sha256
            )
            src = raw_mirror_dir(data_root) / raw.relpath
            if not src.exists():
                raise RuntimeError(f"required raw artifact missing from mirror: {src}")
            link_verified(src, osp.join(self.raw_dir, raw.name), raw.sha256)
        log.info(
            "Public K-12 raw members linked into %s (sha256 verified)", self.raw_dir
        )

    def _genome(self) -> EcoliK12Genome:
        """The injected MG1655 genome, or the default cache opened read-only."""
        if self.ecoli_genome is None:
            self.ecoli_genome = bacterial_genome("ecoli", self.REFERENCE_STRAIN)
        return self.ecoli_genome

    def _write_json(self, name: str, payload: Any) -> None:
        with open(osp.join(self.preprocess_dir, name), "w") as handle:
            json.dump(payload, handle, indent=2, default=str)

    def _deletion_tags(
        self, genome: EcoliK12Genome, symbols: list[str]
    ) -> tuple[dict[str, tuple[str, str]], LocusTagReconciliation]:
        """Each deleted gene symbol -> (b-number, the genome's own symbol for it)."""
        names = pd.Series(sorted(set(symbols)))
        stored, report = reconcile_locus_tags(
            genome, names, label=f"{self.name} deleted genes"
        )
        report.require_resolved(1.0)
        if report.outside_namespace:
            raise RuntimeError(
                f"deleted-gene symbols not stored as b-numbers: {report.outside_namespace}"
            )
        mapping: dict[str, tuple[str, str]] = {}
        for symbol, tag in zip(names, stored, strict=True):
            canonical = genome.genbank.loci[tag].symbol or symbol
            if genome.resolve_gene_name(canonical).systematic_name != tag:
                raise RuntimeError(
                    f"{canonical!r}, the genome's symbol of {tag}, does not resolve back"
                )
            mapping[symbol] = (tag, canonical)
        return mapping, report

    def _genotype(
        self, verdict: PublicGenotypeVerdict, tags: Mapping[str, tuple[str, str]]
    ) -> Genotype:
        return Genotype(
            perturbations=[
                BacterialDeletionPerturbation(
                    systematic_gene_name=tags[symbol][0],
                    perturbed_gene_name=tags[symbol][1],
                    gene_namespace=GENE_NAMESPACE,
                )
                for symbol in verdict.deleted_symbols
            ]
        )

    @post_process
    def process(self) -> None:
        """Settle every public sample, prove no library repeats, then write the LMDB."""
        verify_raw_files(
            self.raw_dir, {raw.name: raw.sha256 for raw in CONSUMED_RAW_FILES}
        )
        genome = self._genome()
        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        metadata = read_metadata(osp.join(self.raw_dir, PUBLIC_METADATA.name))
        counts = pd.read_csv(osp.join(self.raw_dir, PUBLIC_COUNTS.name), index_col=0)
        annotation = pd.read_csv(osp.join(self.raw_dir, GENE_INFO.name), index_col=0)
        multiqc = pd.read_csv(
            osp.join(self.raw_dir, PUBLIC_MULTIQC.name),
            sep="\t",
            index_col=COL_MULTIQC_SAMPLE,
        )
        samples = public_rows(metadata)
        if not set(samples) <= set(counts.columns):
            raise RuntimeError("the count matrix lacks public metadata samples")
        if not set(samples) <= set(multiqc.index):
            raise RuntimeError("the MultiQC table lacks public metadata samples")
        if set(counts.index) != set(annotation.index):
            raise RuntimeError(
                "the count matrix and the annotation name different genes"
            )

        source_genes = [str(g) for g in counts.index]
        stored_genes, gene_report = reconcile_locus_tags(
            genome, pd.Series(source_genes), label=f"{self.name} expression genes"
        )
        gene_report.require_resolved(EXPRESSION_MIN_RESOLVED)
        genes = [str(g) for g in stored_genes]
        if len(set(genes)) != len(genes):
            raise RuntimeError("reconciled gene keys are not unique")
        lengths = gene_lengths(annotation, source_genes)
        divergence = gene_length_divergence(source_genes, genes, lengths, genome)
        if len(divergence) != GENE_LENGTH_DIVERGENCE_N:
            raise RuntimeError(
                f"{len(divergence)} released gene spans differ from the pinned "
                f"assembly, not {GENE_LENGTH_DIVERGENCE_N}"
            )
        self._write_json(
            "gene_length_divergence.json", [d.model_dump() for d in divergence]
        )

        rows = {
            sample: {str(k): str(v) for k, v in metadata.loc[sample].items()}
            for sample in samples
        }
        genotype_verdicts: dict[str, PublicGenotypeVerdict] = {}
        environment_verdicts: dict[str, PublicEnvironmentVerdict] = {}
        dropped: dict[str, list[str]] = {}
        stage_of: dict[str, Literal["genotype", "environment"]] = {}
        kept: list[str] = []
        for sample in samples:
            row = rows[sample]
            g_verdict = settle_public_genotype(row)
            genotype_verdicts[sample] = g_verdict
            if not g_verdict.kept:
                dropped.setdefault(g_verdict.rule.value, []).append(sample)
                stage_of[g_verdict.rule.value] = "genotype"
                continue
            e_verdict = settle_public_environment(row)
            environment_verdicts[sample] = e_verdict
            if e_verdict.rule is not None:
                dropped.setdefault(e_verdict.rule.value, []).append(sample)
                stage_of[e_verdict.rule.value] = "environment"
                continue
            kept.append(sample)

        ledger = self._accession_ledger(metadata, rows, samples, kept, counts)
        deleted = [
            symbol for s in kept for symbol in genotype_verdicts[s].deleted_symbols
        ]
        tags, deletion_report = self._deletion_tags(genome, deleted)
        self._write_json(
            "locus_tag_reconciliation.json",
            {
                "expression_genes": gene_report.model_dump(mode="json"),
                "deleted_genes": deletion_report.model_dump(mode="json"),
                "deleted_symbol_to_locus": {
                    s: list(v) for s, v in sorted(tags.items())
                },
            },
        )

        count_values = counts.loc[source_genes, kept].to_numpy(dtype=np.int64)
        assigned = multiqc.loc[kept, COL_MULTIQC_ASSIGNED].to_numpy(dtype=np.int64)
        totals = count_values.sum(axis=0)
        if not np.array_equal(totals, assigned):
            raise RuntimeError(
                "the stored counts do not sum to the MultiQC Assigned total for "
                f"{int((totals != assigned).sum())} of {len(kept)} samples"
            )
        if int(assigned.min()) < int(READ_DEPTH_FLOOR.value):
            raise RuntimeError(
                f"a kept sample has {int(assigned.min())} assigned reads, below the "
                f"paper's QC floor of {READ_DEPTH_FLOOR.value}"
            )

        reference = self._reference(genes, source_genes, lengths, genome)
        publication = Publication(doi=PAPER_DOI, doi_url=f"https://doi.org/{PAPER_DOI}")
        environments: dict[ConditionSpec, Environment] = {}

        def environment_of(sample: str) -> Environment:
            spec = environment_verdicts[sample].spec
            if spec is None:
                raise RuntimeError(f"kept sample {sample} has no parsed condition")
            if spec not in environments:
                environments[spec] = build_public_environment(spec)
            return environments[spec]

        env, interned_env = self._open_write_lmdb(osp.join(self.processed_dir, "lmdb"))
        groups: dict[str, ReplicateGroup] = {}
        with env.begin(write=True) as txn, interned_env.begin(write=True) as itxn:
            for idx, sample in enumerate(kept):
                experiment = BacterialRNASeqExpressionExperiment(
                    dataset_name=self.name,
                    genotype=self._genotype(genotype_verdicts[sample], tags),
                    environment=environment_of(sample),
                    phenotype=rnaseq_phenotype(
                        genes, count_values[:, idx], lengths, int(assigned[idx])
                    ),
                )
                txn.put(
                    f"{idx}".encode(),
                    self._intern_record(experiment, reference, publication, itxn),
                )
                full_name = rows[sample][COL_FULL_NAME]
                cells = {c: rows[sample][c] for c in CONDITION_COLUMNS}
                group = groups.setdefault(
                    full_name,
                    ReplicateGroup(
                        full_name=full_name,
                        condition_cells=cells,
                        experiment_accessions=[],
                        record_indices=[],
                    ),
                )
                if group.condition_cells != cells:
                    raise RuntimeError(
                        f"replicates of {full_name} differ in their condition cells"
                    )
                group.experiment_accessions.append(sample)
                group.record_indices.append(idx)
        env.close()
        interned_env.close()

        rules = [
            DropRule(
                rule=rule,
                stage=stage_of[rule],
                description=(
                    PUBLIC_GENOTYPE_RULE_REASONS[PublicGenotypeRule(rule)]
                    if stage_of[rule] == "genotype"
                    else PUBLIC_ENVIRONMENT_RULE_REASONS[PublicEnvironmentRule(rule)]
                ),
                n_records=len(items),
                samples=items,
            )
            for rule, items in dropped.items()
        ]
        drop_log = DropLog(
            dataset=self.name,
            source_records=len(samples),
            kept_records=len(kept),
            dropped_records=len(samples) - len(kept),
            rules=sorted(rules, key=lambda r: (r.stage != "genotype", -r.n_records)),
            kept_by_genotype_class=dict(
                sorted(
                    Counter(genotype_class(genotype_verdicts[s]) for s in kept).items()
                )
            ),
            kept_by_base_media=dict(
                sorted(Counter(rows[s][COL_MEDIA] for s in kept).items())
            ),
        )
        if sum(r.n_records for r in drop_log.rules) != drop_log.dropped_records:
            raise RuntimeError("drop accounting mismatch")
        self._write_json("dropped_records.json", drop_log.model_dump())
        self._write_json("accession_ledger.json", ledger.model_dump())
        self._write_json(
            "replicate_groups.json",
            [
                g.model_dump()
                for g in sorted(groups.values(), key=lambda g: g.full_name)
            ],
        )
        self._write_json("record_samples.json", kept)
        log.info(
            "Wrote %d Public K-12 records (%d dropped) over %d conditions",
            len(kept),
            drop_log.dropped_records,
            len(groups),
        )

    def _accession_ledger(
        self,
        metadata: pd.DataFrame,
        rows: Mapping[str, Mapping[str, str]],
        samples: Sequence[str],
        kept: Sequence[str],
        counts: pd.DataFrame,
    ) -> AccessionLedger:
        """Prove the deduplication policy on this build, or refuse the build.

        The record identity is the experiment accession. ``BioSample`` is measured and
        reported but is NOT the key: the release registers one BioSample for several
        libraries, sometimes of different strains. The value-level rule is profile
        identity, within the public arm and against the PRECISE-1K counts the sibling
        dataset serves.
        """
        runs = [rows[s][COL_RUN] for s in samples]
        if len(set(samples)) != len(samples):
            raise RuntimeError("public experiment accessions repeat")
        if len(set(runs)) != len(runs):
            raise RuntimeError("public run accessions repeat")
        public_columns = counts[list(samples)].to_numpy(dtype=np.int64)
        digests = [
            hashlib.sha256(public_columns[:, i].tobytes()).hexdigest()
            for i in range(public_columns.shape[1])
        ]
        repeated = sum(c for c in Counter(digests).values() if c > 1)
        if repeated:
            raise RuntimeError(f"{repeated} public count columns repeat another")
        p1k_counts = pd.read_csv(osp.join(self.raw_dir, P1K_COUNTS.name), index_col=0)
        if list(p1k_counts.index) != list(counts.index):
            raise RuntimeError(
                "the PRECISE-1K and Public K-12 count matrices name different genes"
            )
        p1k_columns = p1k_counts.to_numpy(dtype=np.int64)
        p1k_digests = {
            hashlib.sha256(p1k_columns[:, i].tobytes()).hexdigest()
            for i in range(p1k_columns.shape[1])
        }
        shared = len(set(digests) & p1k_digests)
        if shared:
            raise RuntimeError(
                f"{shared} public count columns are identical to a PRECISE-1K column"
            )
        by_biosample: dict[str, list[str]] = {}
        for sample in samples:
            by_biosample.setdefault(rows[sample][COL_BIOSAMPLE], []).append(sample)
        several = {b: m for b, m in by_biosample.items() if len(m) > 1}
        spanning = sum(
            1
            for members in several.values()
            if len({rows[s][COL_FULL_NAME] for s in members}) > 1
        )
        index = {sample: i for i, sample in enumerate(kept)}
        return AccessionLedger(
            source_rows=int(metadata.shape[0]),
            public_rows=len(samples),
            p1k_rows=int(metadata.shape[0]) - len(samples),
            distinct_experiment_accessions=len(set(samples)),
            distinct_run_accessions=len(set(runs)),
            distinct_biosamples=len(by_biosample),
            biosamples_with_several_rows=len(several),
            rows_sharing_a_biosample=sum(len(m) for m in several.values()),
            biosamples_spanning_several_conditions=spanning,
            identical_count_profiles_within_public=repeated,
            identical_count_profiles_against_precise1k=shared,
            distinct_bioprojects=len({rows[s][COL_BIOPROJECT] for s in samples}),
            distinct_pmids=len(
                {rows[s][COL_PMID] for s in samples if rows[s][COL_PMID]}
            ),
            distinct_geo_series=len(
                {rows[s][COL_GEO_SERIES] for s in samples if rows[s][COL_GEO_SERIES]}
            ),
            records=[
                AccessionRecord(
                    experiment_accession=sample,
                    run_accession=rows[sample][COL_RUN],
                    biosample=rows[sample][COL_BIOSAMPLE],
                    bioproject=rows[sample][COL_BIOPROJECT],
                    scientific_name=rows[sample][COL_SCIENTIFIC_NAME],
                    sample_label=rows[sample][COL_SAMPLE_LABEL],
                    full_name=rows[sample][COL_FULL_NAME],
                    replicate=rows[sample][COL_REP],
                    geo_series=_optional(rows[sample][COL_GEO_SERIES]),
                    geo_sample=_optional(rows[sample][COL_GEO_SAMPLE]),
                    pmid=_optional(rows[sample][COL_PMID]),
                    record_index=index[sample],
                )
                for sample in kept
            ],
        )

    def _reference(
        self,
        genes: Sequence[str],
        source_genes: Sequence[str],
        lengths: np.ndarray[Any, np.dtype[np.float64]],
        genome: EcoliK12Genome,
    ) -> BacterialRNASeqExpressionExperimentReference:
        """The PRECISE-1K control condition, the condition the release centers on."""
        p1k_metadata = read_metadata(osp.join(self.raw_dir, P1K_METADATA.name))
        controls = sorted(
            str(sample)
            for sample in p1k_metadata.index
            if str(p1k_metadata.loc[sample, COL_PROJECT]) == CONTROL_PROJECT
        )
        if not controls:
            raise RuntimeError("the PRECISE-1K metadata names no control samples")
        specs = set()
        for control in controls:
            row = {str(k): str(v) for k, v in p1k_metadata.loc[control].items()}
            verdict = settle_environment(row)
            if verdict.spec is None:
                raise RuntimeError(
                    f"control sample {control} has no parsed condition: {verdict.rule}"
                )
            specs.add(verdict.spec)
        if len(specs) != 1:
            raise RuntimeError("the control samples do not share one environment")
        p1k_counts = pd.read_csv(osp.join(self.raw_dir, P1K_COUNTS.name), index_col=0)
        control_counts = p1k_counts.loc[list(source_genes), controls].to_numpy(
            dtype=np.int64
        )
        gap = ProvenanceGap(
            field="n_mapped_reads",
            reason=ProvenanceGapReason.deferred_pending_source_review,
            looked_in=Provenance(
                source_uri=P1K_COUNTS.relpath,
                citation_key=CITATION_KEY,
                sha256=P1K_COUNTS.sha256,
                page="PRECISE-1K count matrix (no read-depth row)",
            ),
            resolve_with=Provenance(
                source_uri=(
                    f"{ZENODO_RECORD_URL} {ARCHIVE_PREFIX}"
                    "data/precise1k/multiqc_stats.tsv"
                ),
                citation_key=CITATION_KEY,
                sha256=ARCHIVE_SHA256,
                page="the PRECISE-1K MultiQC summary, one row per sample",
            ),
            note="the public arm's own MultiQC table is consumed and fills "
            "n_mapped_reads on every record; the PRECISE-1K control samples behind this "
            "reference are covered by the PRECISE-1K MultiQC table, which no loader "
            "consumes",
        )
        return BacterialRNASeqExpressionExperimentReference(
            dataset_name=self.name,
            genome_reference=assembly_reference(self.REFERENCE_STRAIN),
            environment_reference=build_public_environment(specs.pop()),
            phenotype_reference=reference_phenotype(
                genes, control_counts, lengths, gap
            ),
        )

    def preprocess_raw(
        self, df: pd.DataFrame, preprocess: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Preprocessing is handled inside process() for this dataset."""
        return df

    def create_experiment(self) -> None:
        """Experiment construction is handled inline in process() for this dataset."""
        raise NotImplementedError


def main() -> None:
    """Build/load the dataset for interactive debugging."""
    from dotenv import load_dotenv

    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    root = osp.join(data_root, "data/torchcell/rnaseq_public_k12_lamoureux2023")
    dataset = RnaseqPublicK12Lamoureux2023Dataset(root=root)
    print(f"len = {len(dataset)}")
    print(dataset[0]["experiment"]["genotype"])
    drop_log = json.loads(
        Path(osp.join(root, "preprocess/dropped_records.json")).read_text()
    )
    print(json.dumps({r["rule"]: r["n_records"] for r in drop_log["rules"]}, indent=2))


if __name__ == "__main__":
    main()
