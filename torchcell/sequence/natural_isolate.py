# torchcell/sequence/natural_isolate
# [[torchcell.sequence.natural_isolate]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/sequence/natural_isolate
# Test file: tests/torchcell/sequence/test_natural_isolate.py
"""One natural-isolate genotype per isolate of the Peter 2018 1,011-genome panel.

Every loader whose records are natural isolates of that panel (Caudal 2024, Peter 2018,
Muenzner 2024, De Chiara 2022) builds each isolate's genotype here and nowhere else, so
an isolate is the same ``Genotype`` (byte for byte, one content id, one graph node) in
every store, and a change to what an isolate genotype carries reaches every store at once
(issue #988). Before this module the construction lived inside the Caudal 2024 loader and
the other three imported its private helpers and static methods.

A genotype is a perturbation set off S288C R64 plus one descriptor:

- ``SequenceVariantPerturbation`` per reference gene whose inferred allele
  (``allReferenceGenesWithSNPsAndIndelsInferred.tar.gz``) differs from the S288C slice
  its FASTA header names.
- ``NaturalGenePresencePerturbation`` per non-reference pangenome ORF the isolate
  carries and ``NaturalGeneAbsencePerturbation`` per reference ORF it lacks
  (``genesMatrix_PresenceAbsence.tab.gz``, copy number from
  ``genesMatrix_CopyNumber.tab.gz``).
- ``DeletionPerturbation`` of HO (YDL227C) where Supplementary Table 1 reads 'yes' in
  'HO deletion' (#988). That deletion is engineered, so where the presence matrix marks
  HO absent the natural absence it used to carry is replaced, not duplicated.

Every content edit and call names its gene by the systematic name in
``perturbed_gene_name`` (as the natural absences always have), so a gene has one spelling
across a store: two isolates lack HO naturally, and an ``HO`` deletion beside their
``YDL227C`` absences would split it. The release's own gene name is ``release_gene_name``;
only ``SequenceVariantPerturbation`` keeps the tarball header's symbol, an allele
designation the shared ``canonical_gene_names`` rule exempts.
- ``NaturalLossOfFunctionPerturbation`` per released call of the two loss-of-function
  matrices (#987): ``1011LossOfFunction.xls.gz`` (nonsense, SIFT-deleterious or
  frameshift, folded into one bit) and ``genesMatrix_Frameshift.tab.gz`` (homozygous
  frameshift). A call is kept whatever the other releases say about the same allele;
  ``allele_status`` and ``gene_present_in_isolate`` name the disagreement instead.
- ``Genotype.natural_isolate``: ploidy, genome zygosity, whole-chromosome aneuploidy and
  the manipulation Table S1 marks, each the table's value or a typed gap.

Every input is a sha256-pinned file of the genomes-tier set ``peter2018_1011_assemblies``
(``GENOTYPE_FILES``) or of ``sgd_S288C_R64-4-1_20230830`` (the R64 chromosomes and GFF).
Supplementary Table 1 is the paper's Supplementary Tables workbook, registered into the
Peter tier by ``register_supplementary_tables`` with its PMC retrieval record.

Resolution of the 533 frameshift rows the release gives without a pangenome code: the
row's ``Gene`` is a live R64-4-1 systematic name, or a name ``gene_alias_resolution``
resolves to exactly one live ORF (one ambiguous name, SLM6, by a pinned standard-name
rule), and the ORF must overlap the row's R64-1-1 span. A row that fails is refused and
counted in the ledger, never mapped by coordinates alone: ``AAD16`` (YFL057C in R64-1-1,
now an alias of the YFL056C pseudogene) and ``DDI3`` (its row is YNL335W, whose R64-4-1
standard name is DDI2, while DDI3 now names YFL061W).
"""

from __future__ import annotations

import gzip
import io
import json
import logging
import os.path as osp
import re
import tarfile
import urllib.parse
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm

from torchcell.data import copy_verified, link_verified
from torchcell.datamodels.schema import (
    ArtifactRef,
    ChromosomeCopyChange,
    DeletionPerturbation,
    GenePerturbationType,
    GenomeZygosity,
    GenomicSpan,
    Genotype,
    IsolateManipulation,
    LossOfFunctionAlleleStatus,
    NaturalGeneAbsencePerturbation,
    NaturalGenePresencePerturbation,
    NaturalIsolateDescriptor,
    NaturalLossOfFunctionCause,
    NaturalLossOfFunctionPerturbation,
    SequenceVariantPerturbation,
)
from torchcell.literature.manifest import (
    ArtifactRecord,
    RetrievalMethod,
    RetrievalRecord,
    sha256_file,
)
from torchcell.sequence.genome.registry import (
    PETER2018_1011,
    ROLE_ANNOTATION,
    SGD_S288C_R64,
    assembly_set_dir,
    extend_assembly_set,
    load_genome_manifest,
    resolve,
)
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
)

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Files
# --------------------------------------------------------------------------- #
CITATION_KEY = "peterGenomeEvolution10112018"
SEQUENCE_SOURCE = CITATION_KEY

REFGENE_TAR_NAME = "allReferenceGenesWithSNPsAndIndelsInferred.tar.gz"
PRESENCE_NAME = "genesMatrix_PresenceAbsence.tab.gz"
COPYNUMBER_NAME = "genesMatrix_CopyNumber.tab.gz"
LOF_NAME = "1011LossOfFunction.xls.gz"
FRAMESHIFT_NAME = "genesMatrix_Frameshift.tab.gz"
#: The paper's Supplementary Tables workbook (Tables S1 to S21), publisher file name.
SUPPLEMENTARY_TABLES_NAME = "41586_2018_30_MOESM3_ESM.xls"
SGD_FSA_NAME = "S288C_reference_sequence_R64-4-1_20230830.fsa"
SGD_GFF_NAME = "saccharomyces_cerevisiae_R64-4-1_20230830.gff"

#: Every class an isolate genotype holds. Each natural-isolate loader names the same
#: tuple from its own imports (the schema-impact closure reads a loader's imports only).
ISOLATE_GENOTYPE_CLASSES: tuple[type[Any], ...] = (
    SequenceVariantPerturbation,
    NaturalGenePresencePerturbation,
    NaturalGeneAbsencePerturbation,
    DeletionPerturbation,
    NaturalLossOfFunctionPerturbation,
    NaturalIsolateDescriptor,
)

#: The Peter-tier files an isolate genotype is read from, in the order loaders list them.
GENOTYPE_FILES: tuple[str, ...] = (
    REFGENE_TAR_NAME,
    PRESENCE_NAME,
    COPYNUMBER_NAME,
    LOF_NAME,
    FRAMESHIFT_NAME,
    SUPPLEMENTARY_TABLES_NAME,
)

#: Where the library mirror holds the workbook, and its PMC retrieval record's source.
LIBRARY_TABLES_REL = "si/si3.xls"
TABLES_SHA256 = "b11de13b50bcf91bb2a40cdbd0f2f35372bdef6fc9a018d7538cf5e5eea7f273"
TABLES_URL = (
    "https://pmc-oa-opendata.s3.amazonaws.com/PMC6784862.1/" + SUPPLEMENTARY_TABLES_NAME
)
TABLES_RETRIEVED_AT = "2026-10-11T10:36:36.578778+00:00"

TABLE_S1_SHEET = "Table S1"
#: 0-based row of the column header in Table S1 (sheet row 4).
TABLE_S1_HEADER_ROW = 3
TABLE_S1_COLUMNS = [
    "Standardized name",
    "HO deletion",
    "Ploidy",
    "Aneuploidies",
    "Zygosity",
]

PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "21c49934d546f162b43025c85254bcd1cedcb80ab9a691935301b5962a6a12da"

#: HO, the gene Table S1's 'HO deletion' column names (SGD: HO = YDL227C).
HO_SYSTEMATIC_NAME = "YDL227C"
HO_DELETION_DESCRIPTION = (
    "HO deletion marked 'yes' in Peter 2018 Supplementary Table 1 ('HO deletion'); the "
    "table and the paper do not name the marker that replaced HO"
)
#: The reference the two loss-of-function matrices were called against (Methods).
RELEASE_ASSEMBLY = "R64-1-1"
EUPLOID = "euploid"

_ROMAN = "I II III IV V VI VII VIII IX X XI XII XIII XIV XV XVI".split()
_ROMAN_TO_INT = {r: i + 1 for i, r in enumerate(_ROMAN)}
S288C_RE = re.compile(r"^(Y[A-P][LR]\d{3}[WC](-[A-Z])?|Q\d{4}|YNC[A-Q]\d{4}[WC])$")
_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")
_ANEU_RE = re.compile(r"^aneu;(?:[+-]\d+\*\d+;)+$")
_CHANGE_RE = re.compile(r"([+-])(\d+)\*(\d+);")

LOF_META = ["CODE", "NAME", "Gene", "SGDorder", "CH", "START", "STOP"]
FRAMESHIFT_META = ["CODE", "Gene", "SGDorder", "CH", "START", "STOP"]


# --------------------------------------------------------------------------- #
# Sourced statements the genotype rests on (re-read by the tests and verifiers)
# --------------------------------------------------------------------------- #
def _paper_sv(value: object, quote: str, line: int, note: str) -> SourcedValue:
    return SourcedValue(
        value=value,
        provenance=Provenance(
            source_uri=PAPER_MD,
            citation_key=CITATION_KEY,
            sha256=PAPER_MD_SHA256,
            page=f"paper.md line {line}",
        ),
        quote=quote,
        note=note,
    )


MANIPULATED_QUOTE = (
    "However, among the 918 selected isolates, 124 were non-natural haploid with the "
    "$H O$ gene deleted and the 93 external isolates were genetically manipulated and "
    "made homozygous before sequencing."
)
PLOIDY_STATE_QUOTE = (
    "As 217 strains were genetically manipulated and no longer in their natural ploidy "
    "states, we assigned a relative ploidy state to the other 794 isolates (Methods)"
)
LOF_DEFINITION_QUOTE = (
    "Non-synonymous SNPs predicted as deleterious by SIFT $( \\mathrm { v } . 5 . 2 . 2 "
    ") ^ { 1 7 }$ as well as nonsense mutations were considered as deleterious for "
    "protein function."
)
FRAMESHIFT_QUOTE = (
    "-genesMatrix_Frameshift.tab.gz: for each isolate, the presence or absence "
    "(indicated by 1 or 0, respectively) of homozygous frameshift is reported in each "
    "gene, based on the number of bases affected by indels."
)
REFERENCE_VERSION_QUOTE = "the reads were mapped to the S. cerevisiae S288C reference genome (version R64-1-1)"
#: Table S1, sheet row 2, column A: the 'Aneuploidies' notation.
ANEUPLOIDY_LEGEND = (
    'Aneuploid strains are described as follow in the Aneuploidies column: "aneu;+/-X*Y" '
    "with +/- the gain or loss of chromosome, X the number of supernumerary copies and Y "
    "the concerned chromosome"
)

SOURCED_VALUES: dict[str, SourcedValue] = {
    "ho_deletion": _paper_sv(
        IsolateManipulation.ho_deletion.value,
        MANIPULATED_QUOTE,
        174,
        "Table S1 marks the 124 HO deletions in 'HO deletion'; it marks no column for "
        "the 93 external isolates' manipulation",
    ),
    "ploidy_state": _paper_sv(
        "manipulated isolates are not in their natural ploidy",
        PLOIDY_STATE_QUOTE,
        36,
        "124 HO deletions plus 93 external isolates",
    ),
    "nonsense_sift_or_frameshift": _paper_sv(
        NaturalLossOfFunctionCause.nonsense_sift_or_frameshift.value,
        LOF_DEFINITION_QUOTE,
        202,
        "the class 1011LossOfFunction.xls.gz holds; the file also holds every "
        "homozygous frameshift call of genesMatrix_Frameshift.tab.gz on its genes "
        "(measured, see the ledger)",
    ),
    "homozygous_frameshift": _paper_sv(
        NaturalLossOfFunctionCause.homozygous_frameshift.value,
        FRAMESHIFT_QUOTE,
        290,
        "Data availability",
    ),
    "release_assembly": _paper_sv(
        RELEASE_ASSEMBLY,
        REFERENCE_VERSION_QUOTE,
        180,
        "the coordinates of both loss-of-function matrices are read on this reference",
    ),
}


# --------------------------------------------------------------------------- #
# FASTA, coordinates and pangenome ids
# --------------------------------------------------------------------------- #
class UnextractableMemberError(RuntimeError):
    """A regular file in the reference-gene tarball could not be opened for reading."""


def parse_fasta(text: str) -> list[tuple[str, str]]:
    """Parse FASTA text into ``[(header_without_gt, sequence), ...]``."""
    records: list[tuple[str, str]] = []
    header: str | None = None
    seq: list[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(seq)))
            header = line[1:]
            seq = []
        else:
            seq.append(line.strip())
    if header is not None:
        records.append((header, "".join(seq)))
    return records


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA string (A/C/G/T/N, case-preserving)."""
    return seq.translate(_COMPLEMENT)[::-1]


def sgd_chromosomes(fsa_path: str | Path) -> dict[str, str]:
    """Map each SGD chromosome id (roman I..XVI, ``MT``) to its uppercase sequence."""
    chrom: dict[str, str] = {}
    key: str | None = None
    buf: list[str] = []
    with open(fsa_path) as handle:
        for line in handle:
            if line.startswith(">"):
                if key is not None:
                    chrom[key] = "".join(buf).upper()
                match = re.search(r"\[chromosome=([IVX]+)\]", line)
                if match:
                    key = match.group(1)
                elif "[location=mitochondrion]" in line:
                    key = "MT"
                else:
                    key = None
                buf = []
            else:
                buf.append(line.strip())
    if key is not None:
        chrom[key] = "".join(buf).upper()
    return chrom


def reference_slice(header: str, chrom: Mapping[str, str]) -> str:
    """Reconstruct the S288C reference allele from a Peter gene-file header coordinate."""
    match = re.search(r"chromosome(\d+):(\d+)-(\d+)\s+([+-])", header)
    if match is None:
        raise ValueError(f"unparseable coordinate header: {header!r}")
    chrom_n = int(match.group(1))
    start = int(match.group(2))
    end = int(match.group(3))
    strand = match.group(4)
    key = "MT" if chrom_n == 17 else _ROMAN[chrom_n - 1]
    seq = chrom[key][start - 1 : end]
    if strand == "-":
        seq = reverse_complement(seq)
    return seq.upper()


def isolate_and_symbol(token: str, sys_name: str) -> tuple[str, str]:
    """Split a Peter gene-FASTA header token into ``(isolate_code, gene_symbol)``.

    A token is ``<isolate>_<systematic_name>_<symbol>`` and the isolate code appears
    VERBATIM, in either of two forms: ``AEE_YAL001C_TFC3`` and
    ``SACE_YAU_YAL001C_TFC3``. ``SACE_YAU`` is that isolate's own code, NOT a species
    prefix on ``YAU``: 93 of the 1011 codes indexing ``genesMatrix_PresenceAbsence``
    carry it, and stripping it (the fix proposed in issue 73) leaves those isolates
    unmatchable, so the prefix is KEPT.

    Raises on a token that does not carry ``_<sys_name>_`` or that names an empty
    isolate; no header is ever skipped silently.
    """
    split_key = f"_{sys_name}_"
    if split_key not in token:
        raise ValueError(f"header token {token!r} carries no {split_key!r} gene key")
    iso, symbol = token.split(split_key, 1)
    if not iso:
        raise ValueError(f"header token {token!r} names an empty isolate")
    return iso, symbol


def assert_all_isolates_seen(matched: set[str], seen: set[str]) -> None:
    """Fail loudly when a built isolate never appeared in a Peter gene-FASTA header.

    An isolate absent from every header would be assembled with zero sequence variants
    and so reconstruct as a perfect S288C match: the strain and its phenotype present,
    its genotype silently empty.
    """
    missing = sorted(matched - seen)
    if missing:
        raise ValueError(
            f"{len(missing)} of {len(matched)} matched isolates never appeared in a "
            f"Peter gene-FASTA header, so their genotype would be silently empty: "
            f"{missing}"
        )


def demangle_orf(col: str) -> str:
    """Reverse R ``make.names`` on a pangenome-ORF matrix column -> the raw ORF id.

    ``X1834.YAL063C`` -> ``1834-YAL063C``; ``X1.EC1118_1F14_0012g`` ->
    ``1-EC1118_1F14_0012g`` (leading ``X`` dropped; the first ``.`` was the ``-`` after
    the ORF number).
    """
    stem = col[1:] if col.startswith("X") else col
    return stem.replace(".", "-", 1)


def orf_to_s288c(orf_id: str) -> str | None:
    """Return the S288C systematic name a pangenome ORF id encodes, else None.

    ``1834-YAL063C`` -> ``YAL063C``; ``1-EC1118_1F14_0012g`` -> None. A
    ``_NumOfGenes_N`` suffix marks a cluster of N paralogous copies of a reference gene
    (793 of them), mapped to its S288C name: the cluster is present iff any copy is.
    """
    suffix = re.sub(r"^\d+-", "", orf_id).replace(".", "-")
    suffix = re.sub(r"_NumOfGenes_\d+$", "", suffix)
    return suffix if S288C_RE.match(suffix) else None


def orf_code(column: str) -> int:
    """The pangenome ORF number of a presence-matrix column (``X1834.YAL063C`` -> 1834)."""
    return int(demangle_orf(column).split("-", 1)[0])


def parse_aneuploidy(text: str) -> list[ChromosomeCopyChange]:
    """Parse a Table S1 'Aneuploidies' cell; ``euploid`` is the empty list.

    The notation is ``ANEUPLOIDY_LEGEND``. Any other text raises.
    """
    if text == EUPLOID:
        return []
    if not _ANEU_RE.match(text):
        raise ValueError(f"unparseable aneuploidy annotation {text!r}")
    return [
        ChromosomeCopyChange(
            chromosome=int(chrom), change=int(copies) * (1 if sign == "+" else -1)
        )
        for sign, copies, chrom in _CHANGE_RE.findall(text)
    ]


# --------------------------------------------------------------------------- #
# Artifact refs
# --------------------------------------------------------------------------- #
def tier_file_ref(name: str, data_root: str | None = None) -> ArtifactRef:
    """File-level ``ArtifactRef`` of one Peter-tier file (sha256 + bytes from its manifest)."""
    record = load_genome_manifest(PETER2018_1011, data_root).record(name)
    return ArtifactRef(
        tier="genomes",
        key=PETER2018_1011,
        path=record.path,
        sha256=record.sha256,
        bytes=record.bytes,
    )


def refgene_tarball_ref(data_root: str | None = None) -> ArtifactRef:
    """The file-level ``ArtifactRef`` of the reference-gene tarball."""
    return tier_file_ref(REFGENE_TAR_NAME, data_root)


def refgene_member_ref(tarball: ArtifactRef, sys_name: str, token: str) -> ArtifactRef:
    """``tarball`` narrowed to one allele: member ``<sys_name>.fasta#<token>``."""
    return tarball.model_copy(update={"member": f"{sys_name}.fasta#{token}"})


# --------------------------------------------------------------------------- #
# Raw files: every loader links and verifies the same six
# --------------------------------------------------------------------------- #
def genotype_file_pins(data_root: str | None = None) -> dict[str, str]:
    """``{file name: sha256}`` of ``GENOTYPE_FILES`` from the Peter tier manifest."""
    manifest = load_genome_manifest(PETER2018_1011, data_root)
    return {name: manifest.record(name).sha256 for name in GENOTYPE_FILES}


def link_genotype_files(raw_dir: str, data_root: str | None = None) -> None:
    """Symlink ``GENOTYPE_FILES`` from the Peter tier into ``raw_dir``, each verified."""
    for name, sha in genotype_file_pins(data_root).items():
        link_verified(
            resolve(PETER2018_1011, name, verify=False, data_root=data_root),
            osp.join(raw_dir, name),
            sha,
        )


def register_supplementary_tables(data_root: str | None = None) -> str:
    """Copy the Supplementary Tables workbook into ``peter2018_1011_assemblies``.

    The bytes are the library mirror's ``si/si3.xls`` (fetched from the PMC Open Access
    bucket by ``torchcell.literature.retrieve.pmc_cloud_object``), verified against
    ``TABLES_SHA256`` before anything is written; the tier record restates that
    retrieval. Idempotent: a workbook the manifest already lists is left alone.
    """
    manifest = load_genome_manifest(PETER2018_1011, data_root)
    if SUPPLEMENTARY_TABLES_NAME in {rec.path for rec in manifest.files}:
        return osp.join(assembly_set_dir(PETER2018_1011, data_root), "manifest.json")
    tier = Path(assembly_set_dir(PETER2018_1011, data_root))
    # $DATA_ROOT/torchcell-genomes/<set> -> $DATA_ROOT/torchcell-library/<key>/si/si3.xls
    source = tier.parents[1] / "torchcell-library" / CITATION_KEY / LIBRARY_TABLES_REL
    dest = tier / SUPPLEMENTARY_TABLES_NAME
    copy_verified(source, dest, TABLES_SHA256)
    record = ArtifactRecord(
        path=SUPPLEMENTARY_TABLES_NAME,
        role=ROLE_ANNOTATION,
        bytes=dest.stat().st_size,
        sha256=TABLES_SHA256,
        source=TABLES_URL,
        retrieval=RetrievalRecord(
            method=RetrievalMethod.pmc_cloud,
            source_url=TABLES_URL,
            retriever="torchcell.literature.retrieve.pmc_cloud_object",
            params={"key": "PMC6784862.1/" + SUPPLEMENTARY_TABLES_NAME},
            sha256=TABLES_SHA256,
            retrieved_at=TABLES_RETRIEVED_AT,
        ),
    )
    return extend_assembly_set(PETER2018_1011, [record], data_root)


# --------------------------------------------------------------------------- #
# Sequence variants (the reference-gene tarball diff)
# --------------------------------------------------------------------------- #
class VariantCacheIndex(BaseModel):
    """Sidecar of ``sequence_variants.parquet``: which isolates and genes it covers."""

    model_config = ConfigDict(extra="forbid")

    tarball_sha256: str
    isolates: list[str]
    reference_genes: list[str] = Field(
        description="every gene file of the tarball (the genes an allele is inferred for)"
    )


class ReferenceGeneDiff(BaseModel):
    """The diff of every gene file against S288C over a set of isolates."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    variants: pd.DataFrame = Field(
        description="strain_id, systematic_gene_name, symbol, header_token per variant"
    )
    reference_genes: frozenset[str]


def diff_reference_genes(
    tar_path: str | Path, sgd_fsa_path: str | Path, matched: set[str]
) -> ReferenceGeneDiff:
    """Every (isolate, gene) whose allele differs from the S288C reference slice.

    A gene file's first header gives the R64 coordinate, ``reference_slice``
    reconstructs the reference allele, and each header naming one of ``matched`` whose
    sequence differs is a variant. Raises if any isolate of ``matched`` is in no header.
    """
    chrom = sgd_chromosomes(sgd_fsa_path)
    rows: dict[str, list[str]] = {
        "strain_id": [],
        "systematic_gene_name": [],
        "symbol": [],
        "header_token": [],
    }
    genes: set[str] = set()
    seen: set[str] = set()
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tqdm(tf, desc="diffing reference genes"):
            if not member.isfile():
                continue
            sys_name = member.name.replace(".fasta", "")
            genes.add(sys_name)
            extracted = tf.extractfile(member)
            if extracted is None:
                raise UnextractableMemberError(
                    f"{tar_path}: member {member.name!r} is a regular file but tarfile "
                    "returned no file object for it; refusing to drop its variants"
                )
            records = parse_fasta(extracted.read().decode("latin-1"))
            if not records:
                continue
            ref = reference_slice(records[0][0], chrom)
            for header, seq in records:
                token = header.split()[0].split("\t")[0]
                iso, symbol = isolate_and_symbol(token, sys_name)
                if iso not in matched:
                    continue
                seen.add(iso)
                if seq.upper() != ref:
                    rows["strain_id"].append(iso)
                    rows["systematic_gene_name"].append(sys_name)
                    rows["symbol"].append(symbol)
                    rows["header_token"].append(token)
    assert_all_isolates_seen(matched, seen)
    return ReferenceGeneDiff(
        variants=pd.DataFrame(rows), reference_genes=frozenset(genes)
    )


def load_or_diff_variants(
    tar_path: str | Path,
    sgd_fsa_path: str | Path,
    isolates: set[str],
    cache_dir: str | Path,
) -> ReferenceGeneDiff:
    """``diff_reference_genes`` over ``isolates``, cached in ``cache_dir``.

    The cache is ``sequence_variants.parquet`` (the variant rows) and
    ``sequence_variants.index.json`` (the tarball sha256, the isolates diffed and the
    tarball's genes). It is reused when the sidecar names the tarball's sha256 and a
    superset of ``isolates`` (an isolate's diff does not depend on the others, so a
    verifier re-deriving the isolates a store holds reads the build's cache); otherwise
    the tarball is re-diffed and both files rewritten.
    """
    parquet = Path(cache_dir) / "sequence_variants.parquet"
    sidecar = Path(cache_dir) / "sequence_variants.index.json"
    tar_sha = sha256_file(Path(tar_path))
    if parquet.exists() and sidecar.exists():
        index = VariantCacheIndex.model_validate_json(sidecar.read_text())
        if isolates <= set(index.isolates) and index.tarball_sha256 == tar_sha:
            log.info("Loading cached sequence variants from %s", parquet)
            variants = pd.read_parquet(parquet)
            return ReferenceGeneDiff(
                variants=variants[variants["strain_id"].isin(isolates)].reset_index(
                    drop=True
                ),
                reference_genes=frozenset(index.reference_genes),
            )
    diff = diff_reference_genes(tar_path, sgd_fsa_path, isolates)
    diff.variants.to_parquet(parquet, index=False)
    sidecar.write_text(
        VariantCacheIndex(
            tarball_sha256=tar_sha,
            isolates=sorted(isolates),
            reference_genes=sorted(diff.reference_genes),
        ).model_dump_json(indent=2)
    )
    return diff


# --------------------------------------------------------------------------- #
# Pangenome matrices
# --------------------------------------------------------------------------- #
class PangenomeMatrices(BaseModel):
    """Peter's presence and copy-number matrices, decoded once for all isolates."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    orf_ids: list[str]
    s288c_names: list[str | None]
    s288c_mask: np.ndarray[Any, Any]
    presence: np.ndarray[Any, Any]
    copy_number: np.ndarray[Any, Any]
    row_of: dict[str, int]
    column_of_code: dict[int, int]


def read_pangenome(
    presence_path: str | Path, copynumber_path: str | Path
) -> PangenomeMatrices:
    """Decode the presence matrix and the copy-number matrix reindexed onto its ORFs."""
    presence = pd.read_csv(presence_path, sep="\t", index_col=0)
    copynumber = pd.read_csv(copynumber_path, sep="\t", index_col=0).reindex(
        columns=presence.columns
    )
    orf_ids = [demangle_orf(c) for c in presence.columns]
    s288c_names = [orf_to_s288c(o) for o in orf_ids]
    codes = [orf_code(c) for c in presence.columns]
    if len(set(codes)) != len(codes):
        raise ValueError("two presence-matrix columns carry one pangenome ORF number")
    return PangenomeMatrices(
        orf_ids=orf_ids,
        s288c_names=s288c_names,
        s288c_mask=np.array([n is not None for n in s288c_names], dtype=bool),
        presence=presence.to_numpy(dtype=float),
        copy_number=copynumber.to_numpy(dtype=float),
        row_of={str(s): i for i, s in enumerate(presence.index.astype(str))},
        column_of_code={code: j for j, code in enumerate(codes)},
    )


def panel_isolates(raw_dir: str | Path) -> set[str]:
    """The isolate codes indexing the presence matrix in ``raw_dir`` (the 1,011 genomes)."""
    frame = pd.read_csv(Path(raw_dir) / PRESENCE_NAME, sep="\t", usecols=[0])
    return set(frame.iloc[:, 0].astype(str))


def content_perturbations(
    strain: str, pangenome: PangenomeMatrices
) -> tuple[list[NaturalGenePresencePerturbation], list[NaturalGeneAbsencePerturbation]]:
    """AXIS-1 presence/absence edits for one isolate, RELATIVE TO S288C.

    Presence/absence vs S288C is a question of reference MEMBERSHIP (does the pangenome
    column map to an S288C systematic name, paralog clusters included), not population
    frequency: a non-reference ORF present is a natural presence (with observed copy
    number), a reference ORF absent is a natural absence. A reference ORF present and an
    accessory ORF absent are both no-ops vs S288C.
    """
    row = pangenome.row_of[strain]
    presence_row = pangenome.presence[row]
    copynumber_row = pangenome.copy_number[row]
    presence: list[NaturalGenePresencePerturbation] = []
    for j in np.nonzero((~pangenome.s288c_mask) & (presence_row == 1))[0]:
        cn = copynumber_row[j]
        copy_number = float(cn) if np.isfinite(cn) and cn > 0 else 1.0
        orf_id = pangenome.orf_ids[j]
        presence.append(
            NaturalGenePresencePerturbation(
                systematic_gene_name=orf_id,
                perturbed_gene_name=orf_id,
                copy_number=copy_number,
                strain_id=strain,
                pangenome_orf_id=orf_id,
                origin=None,
                sequence_source=SEQUENCE_SOURCE,
            )
        )
    absence: list[NaturalGeneAbsencePerturbation] = []
    for j in np.nonzero(pangenome.s288c_mask & (presence_row == 0))[0]:
        name = pangenome.s288c_names[j]
        if name is None:
            raise ValueError(
                f"reference column {pangenome.orf_ids[j]} has no S288C name"
            )
        absence.append(
            NaturalGeneAbsencePerturbation(
                systematic_gene_name=name,
                perturbed_gene_name=name,
                strain_id=strain,
                pangenome_orf_id=pangenome.orf_ids[j],
                sequence_source=SEQUENCE_SOURCE,
            )
        )
    return presence, absence


def sequence_perturbations(
    strain: str, variants: Iterable[tuple[str, str, str]], tarball: ArtifactRef
) -> list[SequenceVariantPerturbation]:
    """One ``SequenceVariantPerturbation`` per variant, pointing at its tarball member."""
    return [
        SequenceVariantPerturbation(
            systematic_gene_name=sys_name,
            perturbed_gene_name=symbol or sys_name,
            strain_id=strain,
            sequence_source=SEQUENCE_SOURCE,
            sequence_ref=refgene_member_ref(tarball, sys_name, token),
        )
        for sys_name, symbol, token in variants
    ]


# --------------------------------------------------------------------------- #
# Supplementary Table 1: the isolate descriptor
# --------------------------------------------------------------------------- #
class IsolateAnnotationRow(BaseModel):
    """One isolate's Table S1 cells, verbatim (``None`` = an empty cell)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    isolate: str
    sheet_row: int = Field(description="1-based row of the Table S1 sheet")
    ho_deletion: str
    ploidy: str | None
    aneuploidies: str | None
    zygosity: str


def read_table_s1(path: str | Path) -> dict[str, IsolateAnnotationRow]:
    """``{isolate: row}`` for every isolate row of Supplementary Table 1.

    The sheet holds 1,011 isolate rows under a four-row header and a reference list
    below them; a row is an isolate row exactly when 'Standardized name' is filled.
    """
    frame = pd.read_excel(
        path, sheet_name=TABLE_S1_SHEET, header=TABLE_S1_HEADER_ROW, dtype=object
    )
    missing = sorted(set(TABLE_S1_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Table S1 lacks the columns {missing}")
    out: dict[str, IsolateAnnotationRow] = {}
    for idx, row in frame.iterrows():
        name = row["Standardized name"]
        if pd.isna(name):
            continue
        isolate = str(name).strip()
        if isolate in out:
            raise ValueError(f"Table S1 lists {isolate!r} twice")

        def cell(column: str) -> str | None:
            value = row[column]
            return None if pd.isna(value) else str(value).strip()

        ho = cell("HO deletion")
        zyg = cell("Zygosity")
        if ho is None or zyg is None:
            raise ValueError(
                f"Table S1 row of {isolate}: empty HO deletion or Zygosity"
            )
        out[isolate] = IsolateAnnotationRow(
            isolate=isolate,
            sheet_row=int(str(idx)) + TABLE_S1_HEADER_ROW + 2,
            ho_deletion=ho,
            ploidy=cell("Ploidy"),
            aneuploidies=cell("Aneuploidies"),
            zygosity=zyg,
        )
    return out


def _table_s1_provenance(ref: ArtifactRef, row: int) -> Provenance:
    return Provenance(
        source_uri=f"{ref.tier}/{ref.key}/{ref.path}",
        citation_key=CITATION_KEY,
        sha256=ref.sha256,
        page=f"sheet '{TABLE_S1_SHEET}' row {row}",
    )


def isolate_descriptor(
    row: IsolateAnnotationRow, workbook: ArtifactRef
) -> NaturalIsolateDescriptor:
    """The typed descriptor of one Table S1 row; a cell outside the table's vocabulary raises."""
    gaps: list[ProvenanceGap] = []
    where = _table_s1_provenance(workbook, row.sheet_row)
    if row.ho_deletion == "yes":
        manipulation: IsolateManipulation | None = IsolateManipulation.ho_deletion
    elif row.ho_deletion == "no":
        manipulation = None
        gaps.append(
            ProvenanceGap(
                field="genetic_manipulation",
                reason=ProvenanceGapReason.not_reported_by_primary,
                looked_in=where,
                note="Table S1 reads 'no' in 'HO deletion'. The paper adds that 'the 93 "
                "external isolates were genetically manipulated and made homozygous "
                "before sequencing' (paper.md line 174) and no Table S1 column marks "
                "which isolates those are, so whether this one was manipulated is not "
                "stated",
            )
        )
    else:
        raise ValueError(f"{row.isolate}: 'HO deletion' reads {row.ho_deletion!r}")
    ploidy: int | None
    if row.ploidy is not None and re.fullmatch(r"[1-9]\d*", row.ploidy):
        ploidy = int(row.ploidy)
    elif row.ploidy == "Xn":
        ploidy = None
        gaps.append(
            ProvenanceGap(
                field="ploidy",
                reason=ProvenanceGapReason.not_reported_by_primary,
                looked_in=where,
                note="Table S1 writes the ploidy as 'Xn' (not determined)",
            )
        )
    else:
        raise ValueError(f"{row.isolate}: 'Ploidy' reads {row.ploidy!r}")
    changes: list[ChromosomeCopyChange] | None
    if row.aneuploidies is None:
        changes = None
        gaps.append(
            ProvenanceGap(
                field="chromosome_copy_changes",
                reason=ProvenanceGapReason.not_reported_by_primary,
                looked_in=where,
                note="Table S1 leaves 'Aneuploidies' empty for this isolate",
            )
        )
    else:
        changes = parse_aneuploidy(row.aneuploidies)
    return NaturalIsolateDescriptor(
        isolate=row.isolate,
        annotation_ref=workbook.model_copy(update={"member": TABLE_S1_SHEET}),
        genetic_manipulation=manipulation,
        ploidy=ploidy,
        zygosity=GenomeZygosity(row.zygosity),
        chromosome_copy_changes=changes,
        provenance_gaps=gaps,
    )


# --------------------------------------------------------------------------- #
# R64-4-1 gene names, for the frameshift rows released without a pangenome code
# --------------------------------------------------------------------------- #
_GENE_FEATURE_TYPES = frozenset(
    {
        "gene",
        "pseudogene",
        "transposable_element_gene",
        "blocked_reading_frame",
        "ncRNA_gene",
        "tRNA_gene",
        "snoRNA_gene",
        "snRNA_gene",
        "rRNA_gene",
        "telomerase_RNA_gene",
    }
)


class R64GeneIndex(BaseModel):
    """The three lookups ``gene_alias_resolution`` reads, plus coordinates, from the GFF.

    Satisfies ``GenomeNameIndex``: ``gene_set`` (live locus ids), ``alias_to_systematic``
    (upper-case GFF ``Alias`` -> ids) and ``feature_index['standard_to_ids']``
    (upper-case GFF ``gene=`` standard name -> ids).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    gene_set: frozenset[str]
    alias_to_systematic: dict[str, list[str]]
    feature_index: dict[str, dict[str, list[str]]]
    spans: dict[str, tuple[str, int, int]]


def read_r64_gene_index(gff_path: str | Path) -> R64GeneIndex:
    """Gene-level features of the R64-4-1 GFF (ids, standard names, aliases, spans)."""
    alias: dict[str, set[str]] = {}
    standard: dict[str, set[str]] = {}
    spans: dict[str, tuple[str, int, int]] = {}
    with open(gff_path) as handle:
        for line in handle:
            if line.startswith("##FASTA"):
                break
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] not in _GENE_FEATURE_TYPES:
                continue
            attrs = dict(kv.split("=", 1) for kv in fields[8].split(";") if "=" in kv)
            gid = attrs["ID"]
            spans[gid] = (fields[0], int(fields[3]), int(fields[4]))
            if "gene" in attrs:
                standard.setdefault(
                    urllib.parse.unquote(attrs["gene"]).upper(), set()
                ).add(gid)
            for name in urllib.parse.unquote(attrs.get("Alias", "")).split(","):
                if name:
                    alias.setdefault(name.upper(), set()).add(gid)
    return R64GeneIndex(
        gene_set=frozenset(spans),
        alias_to_systematic={k: sorted(v) for k, v in alias.items()},
        feature_index={"standard_to_ids": {k: sorted(v) for k, v in standard.items()}},
        spans=spans,
    )


#: The GFF row of YBR266C, which names SLM6 its standard name (``gene=SLM6``).
SLM6_GFF_ROW = (
    "chrII\tSGD\tgene\t739936\t740388\t.\t-\t.\tID=YBR266C;Name=YBR266C;gene=SLM6;"
    "Alias=SLM6"
)


def codeless_frameshift_pins() -> list[Any]:
    """Pins for the ambiguous names of the frameshift rows released without a code.

    ``SLM6`` is the standard name of YBR266C and an alias of YPR164W (MMS1); the R64-4-1
    GFF gives the standard name to YBR266C alone and the release's span
    (chrII:739936-740388) is YBR266C's, so the ``SGD_STANDARD_NAME`` rule holds and is
    re-checked on every build. ``DDI3`` is NOT pinned: its row sits at
    chrXIV:11452-12129, YNL335W, whose R64-4-1 standard name is now DDI2, while the
    standard name DDI3 belongs to YFL061W on chrVI. No rule ``gene_alias_resolution``
    admits picks the ORF by the release's coordinates, so the row is refused.
    """
    from torchcell.datamodels.strain_background import R64_GFF
    from torchcell.datasets.gene_alias_resolution import (
        AliasResolutionRule,
        PinnedAliasResolution,
    )

    return [
        PinnedAliasResolution(
            alias="SLM6",
            systematic_name="YBR266C",
            rule=AliasResolutionRule.SGD_STANDARD_NAME,
            provenance=R64_GFF,
            quote=SLM6_GFF_ROW,
        )
    ]


# --------------------------------------------------------------------------- #
# The two loss-of-function matrices
# --------------------------------------------------------------------------- #
class ReleaseRow(BaseModel):
    """One row of a loss-of-function matrix, keyed as the release keys it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cause: NaturalLossOfFunctionCause
    systematic_gene_name: str
    release_code: int | None
    release_gene_name: str
    span: GenomicSpan


class RefusedRow(BaseModel):
    """A released row no perturbation is written for, with the measured reason."""

    model_config = ConfigDict(extra="forbid")

    cause: NaturalLossOfFunctionCause
    release_gene_name: str
    span: GenomicSpan
    reason: str
    n_calls_in_built_isolates: int


class CodelessResolution(BaseModel):
    """How the frameshift rows without a pangenome code were named."""

    model_config = ConfigDict(extra="forbid")

    n_rows: int
    by_systematic_name: int
    by_alias: dict[str, str] = Field(description="release name -> resolved ORF")
    spans_matching_r64_4_1: int
    spans_differing_from_r64_4_1: dict[str, str]
    refused: list[str]


def _span(ch: Any, start: Any, stop: Any) -> GenomicSpan:
    n = int(ch)
    if not 1 <= n <= 16:
        raise ValueError(f"loss-of-function row on chromosome {ch!r}")
    return GenomicSpan(
        chromosome=f"chr{_ROMAN[n - 1]}",
        start=int(start),
        end=int(stop),
        assembly=RELEASE_ASSEMBLY,
    )


def _matrix_isolates(
    frame: pd.DataFrame, meta: list[str], panel: set[str]
) -> list[str]:
    isolates = [str(c) for c in frame.columns if c not in meta]
    if set(isolates) != panel or len(isolates) != len(panel):
        raise ValueError("a loss-of-function matrix does not index the 1,011 isolates")
    values = frame[isolates].to_numpy()
    if not np.isin(values, (0, 1)).all():
        raise ValueError("a loss-of-function matrix holds a value other than 0 or 1")
    return isolates


def read_lof_matrix(path: str | Path) -> pd.DataFrame:
    """``1011LossOfFunction.xls.gz``: gzip of tab-separated text with CR line ends."""
    text = gzip.open(path).read().decode("latin-1").replace("\r", "\n")
    return pd.read_csv(io.StringIO(text), sep="\t", low_memory=False)


def read_frameshift_matrix(path: str | Path) -> pd.DataFrame:
    """``genesMatrix_Frameshift.tab.gz``; ``CODE`` is empty on 533 rows."""
    return pd.read_csv(path, sep="\t", low_memory=False)


def lof_rows(
    lof: pd.DataFrame, pangenome: PangenomeMatrices
) -> tuple[list[ReleaseRow], np.ndarray[Any, Any], list[str]]:
    """The loss-of-function matrix's rows, named by their pangenome code.

    Every row carries a code, and ``NAME`` must equal the S288C name the code's
    presence-matrix column encodes (it does for all 6,031): the genotype names the gene
    as its natural absence would.
    """
    isolates = _matrix_isolates(lof, LOF_META, set(pangenome.row_of))
    if lof["CODE"].isna().any() or lof["CODE"].duplicated().any():
        raise ValueError("1011LossOfFunction: a row without a code, or a code twice")
    rows: list[ReleaseRow] = []
    for _, r in lof.iterrows():
        code = int(r["CODE"])
        name = pangenome.s288c_names[pangenome.column_of_code[code]]
        if name is None or name != str(r["NAME"]):
            raise ValueError(
                f"1011LossOfFunction: CODE {code} names {name}, NAME {r['NAME']}"
            )
        rows.append(
            ReleaseRow(
                cause=NaturalLossOfFunctionCause.nonsense_sift_or_frameshift,
                systematic_gene_name=name,
                release_code=code,
                release_gene_name=str(r["NAME"]),
                span=_span(r["CH"], r["START"], r["STOP"]),
            )
        )
    return rows, lof[isolates].to_numpy(dtype=np.int8), isolates


def frameshift_rows(
    frame: pd.DataFrame, pangenome: PangenomeMatrices, genes: R64GeneIndex
) -> tuple[
    list[ReleaseRow | RefusedRow], np.ndarray[Any, Any], list[str], CodelessResolution
]:
    """The frameshift matrix's rows: coded rows by their code, the rest by name.

    A coded row is named by the S288C name its code's presence-matrix column encodes. A
    row without a code is named by its ``Gene``: a live R64-4-1 locus id, else the one
    live ORF ``gene_alias_resolution`` resolves it to (``codeless_frameshift_pins`` for
    an ambiguous name). A name that resolves to none, or to an ORF that does not overlap
    the release's R64-1-1 span, is refused (a ``RefusedRow``); whether an overlapping
    span is identical is recorded, never used to pick the ORF.
    """
    # Imported here, not at module top: ``torchcell.datasets`` imports every loader on
    # first import, and the natural-isolate loaders import this module.
    from torchcell.datasets.gene_alias_resolution import (
        GeneNameRefused,
        pin_table,
        resolve_gene_name_strict,
    )

    pins = pin_table(codeless_frameshift_pins())

    isolates = _matrix_isolates(frame, FRAMESHIFT_META, set(pangenome.row_of))
    coded = frame["CODE"].dropna()
    if coded.duplicated().any():
        raise ValueError("genesMatrix_Frameshift: a code appears twice")
    rows: list[ReleaseRow | RefusedRow] = []
    by_systematic = 0
    by_alias: dict[str, str] = {}
    matching = 0
    differing: dict[str, str] = {}
    refused: list[str] = []
    n_codeless = 0
    for _, r in frame.iterrows():
        span = _span(r["CH"], r["START"], r["STOP"])
        gene = str(r["Gene"])
        if not pd.isna(r["CODE"]):
            code = int(r["CODE"])
            name = pangenome.s288c_names[pangenome.column_of_code[code]]
            if name is None:
                raise ValueError(
                    f"genesMatrix_Frameshift: CODE {code} is not a reference ORF"
                )
            rows.append(
                ReleaseRow(
                    cause=NaturalLossOfFunctionCause.homozygous_frameshift,
                    systematic_gene_name=name,
                    release_code=code,
                    release_gene_name=gene,
                    span=span,
                )
            )
            continue
        n_codeless += 1
        try:
            name = resolve_gene_name_strict(genes, gene, pins)
        except GeneNameRefused as refusal:
            refused.append(gene)
            rows.append(
                RefusedRow(
                    cause=NaturalLossOfFunctionCause.homozygous_frameshift,
                    release_gene_name=gene,
                    span=span,
                    reason=f"{refusal.refusal.reason.value}: {refusal.refusal.detail}",
                    n_calls_in_built_isolates=0,
                )
            )
            continue
        chrom, start, end = genes.spans[name]
        if chrom != span.chromosome or end < span.start or start > span.end:
            refused.append(gene)
            rows.append(
                RefusedRow(
                    cause=NaturalLossOfFunctionCause.homozygous_frameshift,
                    release_gene_name=gene,
                    span=span,
                    reason=f"resolves to {name} at {chrom}:{start}-{end}, which does "
                    "not overlap the release's span",
                    n_calls_in_built_isolates=0,
                )
            )
            continue
        if name == gene.upper():
            by_systematic += 1
        else:
            by_alias[gene] = name
        if (chrom, start, end) == (span.chromosome, span.start, span.end):
            matching += 1
        else:
            differing[gene] = (
                f"{name} {chrom}:{start}-{end} (R64-4-1) vs {span.chromosome}:"
                f"{span.start}-{span.end} ({RELEASE_ASSEMBLY})"
            )
        rows.append(
            ReleaseRow(
                cause=NaturalLossOfFunctionCause.homozygous_frameshift,
                systematic_gene_name=name,
                release_code=None,
                release_gene_name=gene,
                span=span,
            )
        )
    resolution = CodelessResolution(
        n_rows=n_codeless,
        by_systematic_name=by_systematic,
        by_alias=by_alias,
        spans_matching_r64_4_1=matching,
        spans_differing_from_r64_4_1=differing,
        refused=refused,
    )
    return rows, frame[isolates].to_numpy(dtype=np.int8), isolates, resolution


# --------------------------------------------------------------------------- #
# Ledger
# --------------------------------------------------------------------------- #
class LossOfFunctionTally(BaseModel):
    """Calls of one cause over the built isolates, split by what the other releases say."""

    model_config = ConfigDict(extra="forbid")

    n_calls: int
    n_genes: int
    by_allele_status: dict[str, int]
    n_on_gene_absent_in_isolate: int = Field(
        description="calls whose pangenome ORF the presence matrix marks absent"
    )
    n_without_pangenome_code: int


class NaturalIsolateGenotypeLedger(BaseModel):
    """What the genotypes of one build carry, counted (``preprocess/`` JSON)."""

    model_config = ConfigDict(extra="forbid")

    n_isolates: int
    n_sequence_variants: int
    n_natural_presences: int
    n_natural_absences: int
    ho_deleted: list[str]
    ho_deleted_absent_in_matrix: list[str] = Field(
        description="HO-deleted isolates the presence matrix marks HO absent: the "
        "natural absence they carried before #988 is replaced by the deletion"
    )
    ho_deleted_with_ho_sequence_variant: list[str]
    ploidy: dict[str, int] = Field(description="Table S1 'Ploidy' -> isolates")
    zygosity: dict[str, int]
    n_aneuploid: int
    n_euploid: int
    n_aneuploidy_not_annotated: int
    n_manipulation_not_stated: int
    loss_of_function: dict[str, LossOfFunctionTally]
    lof_frameshift_containment: dict[str, int] = Field(
        description="over the built isolates and the genes both matrices code: "
        "frameshift calls, and how many of them the loss-of-function matrix also calls"
    )
    codeless_frameshift_rows: CodelessResolution
    refused_rows: list[RefusedRow]


# --------------------------------------------------------------------------- #
# The builder every natural-isolate loader uses
# --------------------------------------------------------------------------- #
class NaturalIsolateGenotypes:
    """Every input of an isolate genotype, read once; ``genotype(isolate)`` builds one.

    ``raw_dir`` holds ``GENOTYPE_FILES`` (linked by ``link_genotype_files`` and verified
    by the loader); ``cache_dir`` holds the tarball diff over ``isolates``. The S288C
    chromosomes and GFF are resolved from the SGD tier (sha256-verified on resolve).
    """

    def __init__(
        self,
        raw_dir: str | Path,
        cache_dir: str | Path,
        isolates: Iterable[str],
        data_root: str | None = None,
    ) -> None:
        """Read the matrices and Table S1; diff (or load) the alleles of ``isolates``."""
        raw = Path(raw_dir)
        self.raw_dir = raw
        self.isolates: frozenset[str] = frozenset(isolates)
        self.pangenome = read_pangenome(raw / PRESENCE_NAME, raw / COPYNUMBER_NAME)
        outside = sorted(self.isolates - set(self.pangenome.row_of))
        if outside:
            raise ValueError(f"{len(outside)} isolates have no Peter genome: {outside}")
        self.tarball = refgene_tarball_ref(data_root)
        self.diff = load_or_diff_variants(
            raw / REFGENE_TAR_NAME,
            resolve(SGD_S288C_R64, SGD_FSA_NAME, data_root=data_root),
            set(self.isolates),
            cache_dir,
        )
        self._variants: dict[str, list[tuple[str, str, str]]] = {
            iso: [] for iso in self.isolates
        }
        self._token: dict[tuple[str, str], str] = {}
        for strain, gene, symbol, token in self.diff.variants[
            ["strain_id", "systematic_gene_name", "symbol", "header_token"]
        ].itertuples(index=False):
            self._variants[str(strain)].append((str(gene), str(symbol), str(token)))
            self._token[(str(strain), str(gene))] = str(token)
        workbook = tier_file_ref(SUPPLEMENTARY_TABLES_NAME, data_root)
        table = read_table_s1(raw / SUPPLEMENTARY_TABLES_NAME)
        if set(table) != set(self.pangenome.row_of):
            raise ValueError("Table S1 does not list exactly the 1,011 genome isolates")
        self.descriptors = {
            iso: isolate_descriptor(table[iso], workbook) for iso in self.isolates
        }
        genes = read_r64_gene_index(
            resolve(SGD_S288C_R64, SGD_GFF_NAME, data_root=data_root)
        )
        self._lof_ref = tier_file_ref(LOF_NAME, data_root)
        self._fs_ref = tier_file_ref(FRAMESHIFT_NAME, data_root)
        lof, lof_matrix, lof_iso = lof_rows(
            read_lof_matrix(raw / LOF_NAME), self.pangenome
        )
        fs, fs_matrix, fs_iso, self.codeless = frameshift_rows(
            read_frameshift_matrix(raw / FRAMESHIFT_NAME), self.pangenome, genes
        )
        self._calls: dict[str, list[ReleaseRow]] = {iso: [] for iso in self.isolates}
        self.refused: list[RefusedRow] = []
        for rows, matrix, columns in (
            (lof, lof_matrix, lof_iso),
            (fs, fs_matrix, fs_iso),
        ):
            built = [j for j, iso in enumerate(columns) if iso in self.isolates]
            for i, row in enumerate(rows):
                hits = [columns[j] for j in built if matrix[i, j] == 1]
                if isinstance(row, RefusedRow):
                    self.refused.append(
                        row.model_copy(update={"n_calls_in_built_isolates": len(hits)})
                    )
                    continue
                for iso in hits:
                    self._calls[iso].append(row)
        self._containment = self._lof_frameshift_containment(
            lof, lof_matrix, lof_iso, fs, fs_matrix, fs_iso
        )

    def _lof_frameshift_containment(
        self,
        lof: list[ReleaseRow],
        lof_matrix: np.ndarray[Any, Any],
        lof_iso: list[str],
        fs: list[ReleaseRow | RefusedRow],
        fs_matrix: np.ndarray[Any, Any],
        fs_iso: list[str],
    ) -> dict[str, int]:
        lof_index = {row.release_code: i for i, row in enumerate(lof)}
        fs_col = {iso: j for j, iso in enumerate(fs_iso)}
        cols = [j for j, iso in enumerate(lof_iso) if iso in self.isolates]
        fs_cols = [fs_col[lof_iso[j]] for j in cols]
        n_frameshift = 0
        n_also_lof = 0
        for i, row in enumerate(fs):
            if isinstance(row, RefusedRow) or row.release_code not in lof_index:
                continue
            f = fs_matrix[i, fs_cols]
            ell = lof_matrix[lof_index[row.release_code], cols]
            n_frameshift += int(f.sum())
            n_also_lof += int(((f == 1) & (ell == 1)).sum())
        return {"frameshift_calls": n_frameshift, "also_called_by_lof": n_also_lof}

    # ---- one isolate -----------------------------------------------------------
    def allele_status(self, isolate: str, gene: str) -> LossOfFunctionAlleleStatus:
        """What the reference-gene tarball says about ``isolate``'s allele of ``gene``."""
        if gene not in self.diff.reference_genes:
            return LossOfFunctionAlleleStatus.no_inferred_allele
        if (isolate, gene) in self._token:
            return LossOfFunctionAlleleStatus.sequence_variant
        return LossOfFunctionAlleleStatus.allele_equals_reference

    def _loss_of_function(
        self, isolate: str
    ) -> list[NaturalLossOfFunctionPerturbation]:
        out: list[NaturalLossOfFunctionPerturbation] = []
        row_of = self.pangenome.row_of[isolate]
        for call in sorted(
            self._calls[isolate],
            key=lambda c: (
                c.systematic_gene_name,
                c.cause.value,
                -1 if c.release_code is None else c.release_code,
                c.release_gene_name,
            ),
        ):
            status = self.allele_status(isolate, call.systematic_gene_name)
            token = self._token.get((isolate, call.systematic_gene_name))
            present: bool | None = None
            if call.release_code is not None:
                column = self.pangenome.column_of_code[call.release_code]
                present = bool(self.pangenome.presence[row_of, column] == 1)
            out.append(
                NaturalLossOfFunctionPerturbation(
                    systematic_gene_name=call.systematic_gene_name,
                    perturbed_gene_name=call.systematic_gene_name,
                    strain_id=isolate,
                    cause=call.cause,
                    allele_status=status,
                    sequence_source=None if token is None else SEQUENCE_SOURCE,
                    sequence_ref=None
                    if token is None
                    else refgene_member_ref(
                        self.tarball, call.systematic_gene_name, token
                    ),
                    call_ref=self._lof_ref
                    if call.cause
                    is NaturalLossOfFunctionCause.nonsense_sift_or_frameshift
                    else self._fs_ref,
                    release_code=call.release_code,
                    release_gene_name=call.release_gene_name,
                    release_span=call.span,
                    gene_present_in_isolate=present,
                )
            )
        return out

    def base_perturbations(self, isolate: str) -> list[GenePerturbationType]:
        """The genotype as the loaders built it before #987 and #988.

        Sequence variants, natural presences and natural absences, nothing else: what
        ``experiments/036-dataset-fixes-before-kg-build/scripts/natural_isolate_genotype_identity.py``
        compares with the stores built before this module, perturbation for perturbation.
        """
        presence, absence = content_perturbations(isolate, self.pangenome)
        sequence = sequence_perturbations(
            isolate, self._variants[isolate], self.tarball
        )
        return [*sequence, *presence, *absence]

    def genotype(self, isolate: str) -> Genotype:
        """The isolate's full genotype: base perturbations, HO deletion, calls, descriptor."""
        descriptor = self.descriptors[isolate]
        perts = self.base_perturbations(isolate)
        if descriptor.genetic_manipulation is IsolateManipulation.ho_deletion:
            perts = [
                p
                for p in perts
                if not (
                    isinstance(p, NaturalGeneAbsencePerturbation)
                    and p.systematic_gene_name == HO_SYSTEMATIC_NAME
                )
            ]
            perts.append(
                DeletionPerturbation(
                    systematic_gene_name=HO_SYSTEMATIC_NAME,
                    perturbed_gene_name=HO_SYSTEMATIC_NAME,
                    description=HO_DELETION_DESCRIPTION,
                )
            )
        perts.extend(self._loss_of_function(isolate))
        return Genotype(perturbations=perts, natural_isolate=descriptor)

    # ---- ledger -----------------------------------------------------------------
    def ledger(self) -> NaturalIsolateGenotypeLedger:
        """Counts over every built isolate (one pass over the calls and matrices)."""
        ho = sorted(
            iso
            for iso, d in self.descriptors.items()
            if d.genetic_manipulation is IsolateManipulation.ho_deletion
        )
        ho_cols = [
            j
            for j, name in enumerate(self.pangenome.s288c_names)
            if name == HO_SYSTEMATIC_NAME
        ]
        tallies: dict[str, LossOfFunctionTally] = {}
        for cause in NaturalLossOfFunctionCause:
            statuses: Counter[str] = Counter()
            genes: set[str] = set()
            absent = 0
            codeless = 0
            n = 0
            for iso in self.isolates:
                row_of = self.pangenome.row_of[iso]
                for call in self._calls[iso]:
                    if call.cause is not cause:
                        continue
                    n += 1
                    genes.add(call.systematic_gene_name)
                    statuses[
                        self.allele_status(iso, call.systematic_gene_name).value
                    ] += 1
                    if call.release_code is None:
                        codeless += 1
                        continue
                    column = self.pangenome.column_of_code[call.release_code]
                    absent += int(self.pangenome.presence[row_of, column] == 0)
            tallies[cause.value] = LossOfFunctionTally(
                n_calls=n,
                n_genes=len(genes),
                by_allele_status=dict(sorted(statuses.items())),
                n_on_gene_absent_in_isolate=absent,
                n_without_pangenome_code=codeless,
            )
        descriptors = list(self.descriptors.values())
        n_absences = 0
        n_presences = 0
        for iso in self.isolates:
            row = self.pangenome.presence[self.pangenome.row_of[iso]]
            n_presences += int(((row == 1) & ~self.pangenome.s288c_mask).sum())
            n_absences += int(((row == 0) & self.pangenome.s288c_mask).sum())
        return NaturalIsolateGenotypeLedger(
            n_isolates=len(self.isolates),
            n_sequence_variants=len(self.diff.variants),
            n_natural_presences=n_presences,
            n_natural_absences=n_absences,
            ho_deleted=ho,
            ho_deleted_absent_in_matrix=[
                iso
                for iso in ho
                if any(
                    self.pangenome.presence[self.pangenome.row_of[iso], j] == 0
                    for j in ho_cols
                )
            ],
            ho_deleted_with_ho_sequence_variant=[
                iso for iso in ho if (iso, HO_SYSTEMATIC_NAME) in self._token
            ],
            ploidy=dict(
                sorted(
                    Counter(
                        "Xn" if d.ploidy is None else str(d.ploidy) for d in descriptors
                    ).items()
                )
            ),
            zygosity=dict(
                sorted(Counter(d.zygosity.value for d in descriptors).items())
            ),
            n_aneuploid=sum(1 for d in descriptors if d.chromosome_copy_changes),
            n_euploid=sum(1 for d in descriptors if d.chromosome_copy_changes == []),
            n_aneuploidy_not_annotated=sum(
                1 for d in descriptors if d.chromosome_copy_changes is None
            ),
            n_manipulation_not_stated=sum(
                1 for d in descriptors if d.genetic_manipulation is None
            ),
            loss_of_function=tallies,
            lof_frameshift_containment=self._containment,
            codeless_frameshift_rows=self.codeless,
            refused_rows=self.refused,
        )

    def write_ledger(self, path: str | Path) -> NaturalIsolateGenotypeLedger:
        """Write ``ledger()`` as JSON to ``path`` and return it."""
        ledger = self.ledger()
        Path(path).write_text(ledger.model_dump_json(indent=2))
        return ledger


def isolate_genotype(isolate: str, genotypes: NaturalIsolateGenotypes) -> Genotype:
    """One isolate's genotype (the function name #988 asks every loader to share)."""
    return genotypes.genotype(isolate)


def genotype_isolate(genotype: Mapping[str, Any]) -> str | None:
    """The isolate a stored genotype dump describes, or None when it names none.

    Read from ``natural_isolate``; ``Genotype`` refuses a strain-carrying perturbation
    that names another isolate, so the descriptor speaks for the whole genotype.
    """
    descriptor = genotype.get("natural_isolate")
    if not isinstance(descriptor, Mapping):
        return None
    return str(descriptor["isolate"])


def ledger_json(ledger: NaturalIsolateGenotypeLedger) -> str:
    """Stable JSON of a ledger (sorted keys), for comparing two builds' ledgers."""
    return json.dumps(ledger.model_dump(mode="json"), sort_keys=True, indent=2)
