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

A genotype is the release's own description of the isolate, stored as released: a
perturbation set off S288C R64 plus one descriptor.

- ``SequenceVariantPerturbation`` per reference gene whose inferred allele
  (``allReferenceGenesWithSNPsAndIndelsInferred.tar.gz``) differs from the S288C slice
  its FASTA header names.
- ``NaturalGenePresencePerturbation`` per non-reference pangenome ORF the isolate
  carries and ``NaturalGeneAbsencePerturbation`` per reference ORF it lacks
  (``genesMatrix_PresenceAbsence.tab.gz``, copy number from
  ``genesMatrix_CopyNumber.tab.gz``).
- ``DeletionPerturbation`` of HO (YDL227C, the gene the presence matrix names
  ``X2672.YDL227C``) where Supplementary Table 1 reads 'yes' in 'HO deletion' (#988).
  Every other perturbation stands as its own file states it, including a natural
  absence of the same gene in an HO-deleted isolate: two released statements, both kept.
- ``NaturalLossOfFunctionPerturbation`` per released call of the two loss-of-function
  matrices (#987), ``1011LossOfFunction.xls.gz`` ("nonsense mutation + SIFT
  predictions", one bit) and ``genesMatrix_Frameshift.tab.gz`` (homozygous
  frameshift), keyed as each matrix keys its rows: by ``NAME`` (and ``CODE``) in the
  first, by the code's pangenome ORF name or, on the 533 rows the release gives no
  code, by ``Gene`` verbatim in the second. No call is dropped, re-keyed or annotated
  by a lookup in any other file or in the S288C annotation.
- ``Genotype.natural_isolate``: ploidy, genome zygosity, whole-chromosome aneuploidy and
  the manipulation Table S1 marks, each the table's value or a typed gap.

Every content edit and call names its gene in ``perturbed_gene_name`` by the same key as
``systematic_gene_name`` (as the natural absences always have); only
``SequenceVariantPerturbation`` keeps the tarball header's symbol, an allele designation
the shared ``canonical_gene_names`` rule exempts.

Every input is a sha256-pinned file of the genomes-tier set ``peter2018_1011_assemblies``
(``GENOTYPE_FILES``) or of ``sgd_S288C_R64-4-1_20230830`` (the R64 chromosomes the
inferred alleles are diffed against). Supplementary Table 1 is the paper's Supplementary
Tables workbook, registered into the Peter tier by ``register_supplementary_tables`` with
its PMC retrieval record.
"""

from __future__ import annotations

import gzip
import io
import json
import logging
import os.path as osp
import re
import tarfile
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
    NaturalGeneAbsencePerturbation,
    NaturalGenePresencePerturbation,
    NaturalIsolateDescriptor,
    NaturalLossOfFunctionCause,
    NaturalLossOfFunctionPerturbation,
    SequenceVariantPerturbation,
)
from torchcell.literature.manifest import (
    ROLE_RAW_DATA,
    ArtifactRecord,
    Manifest,
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

#: The release host's own description of its files, deposited in the raw mirror.
RELEASE_HOST = "http://1002genomes.u-strasbg.fr/files/"
README_NAME = "README.txt"
README_REL = f"data/{README_NAME}"
README_SHA256 = "f48e32771b44d38e88b9c8833f6f47e1a30eb6634ffccf55be42b117dd24f025"
README_RETRIEVED_AT = "2026-10-12T14:45:03Z"

PAPER_MD = "paper.md"
PAPER_MD_SHA256 = "21c49934d546f162b43025c85254bcd1cedcb80ab9a691935301b5962a6a12da"

#: HO, the gene Table S1's 'HO deletion' column names (SGD: HO = YDL227C).
HO_SYSTEMATIC_NAME = "YDL227C"
HO_DELETION_DESCRIPTION = (
    "HO deletion marked 'yes' in Peter 2018 Supplementary Table 1 ('HO deletion'); the "
    "table and the paper do not name the marker that replaced HO"
)
#: The reference the two loss-of-function matrices were called against (Methods), kept
#: verbatim in the ``release_assembly`` sourced value. The spans pin the deposited set
#: ``SGD_S288C_R64``: R64-1-1 and R64-4-1 are annotation releases of one sequence,
#: NCBI assembly R64 (GCA_000146045.2), measured byte-identical across all 16 nuclear
#: chromosomes on 2026-10-15 (``registry.SGD_S288C_R64_GENBANK_ACCESSION``).
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
#: ``README.txt`` line 14, the release's description of ``1011LossOfFunction.xls.gz``.
LOF_README_QUOTE = (
    "1011LossOfFunction.xls.gz: loss of function predicted for all genes in the 1011 "
    "dataset (nonsense mutation + SIFT predictions)"
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
    "nonsense_or_sift_methods": _paper_sv(
        NaturalLossOfFunctionCause.nonsense_or_sift.value,
        LOF_DEFINITION_QUOTE,
        202,
        "the Methods' definition of the two predictions the release's README names for "
        "1011LossOfFunction.xls.gz (RELEASE_SOURCED_VALUES)",
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


#: Statements the release host makes about its files, audited against the raw mirror
#: (``$DATA_ROOT/torchcell-raw``), not the library.
RELEASE_SOURCED_VALUES: dict[str, SourcedValue] = {
    "nonsense_or_sift": SourcedValue(
        value=NaturalLossOfFunctionCause.nonsense_or_sift.value,
        provenance=Provenance(
            source_uri=README_REL,
            citation_key=CITATION_KEY,
            sha256=README_SHA256,
            page="README.txt line 14",
        ),
        quote=LOF_README_QUOTE,
        note="the release folds the two predictions into one bit per gene and isolate",
    )
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


def register_release_readme(
    source_path: str | Path, data_root: str | None = None
) -> Path:
    """Copy the release host's ``README.txt`` into the Peter 2018 raw mirror.

    The bytes were fetched with ``curl -sSL -O http://1002genomes.u-strasbg.fr/files/README.txt``
    on ``README_RETRIEVED_AT`` and are verified against ``README_SHA256`` before
    anything is written; the file is appended to the mirror's ``manifest.json`` with a
    ``direct_url`` record. It is the release's own description of its files, the source
    of ``NaturalLossOfFunctionCause.nonsense_or_sift``. Idempotent: a README the
    manifest already lists is left alone.
    """
    tier = Path(assembly_set_dir(PETER2018_1011, data_root))
    # $DATA_ROOT/torchcell-genomes/<set> -> $DATA_ROOT/torchcell-raw/<key>
    mirror = tier.parents[1] / "torchcell-raw" / CITATION_KEY
    manifest_path = mirror / "manifest.json"
    manifest = Manifest.model_validate_json(manifest_path.read_text())
    if README_REL in {rec.path for rec in manifest.files}:
        return manifest_path
    copy_verified(source_path, mirror / README_REL, README_SHA256)
    url = RELEASE_HOST + README_NAME
    manifest.files.append(
        ArtifactRecord(
            path=README_REL,
            role=ROLE_RAW_DATA,
            bytes=(mirror / README_REL).stat().st_size,
            sha256=README_SHA256,
            source=url,
            retrieval=RetrievalRecord(
                method=RetrievalMethod.direct_url,
                source_url=url,
                retriever="torchcell.literature.retrieve.direct_url",
                params={"url": url},
                sha256=README_SHA256,
                retrieved_at=README_RETRIEVED_AT,
            ),
        )
    )
    manifest_path.write_text(manifest.model_dump_json(indent=2))
    return manifest_path


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
                gene_namespace="scerevisiae_peter2018_pangenome_orf",
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
                gene_namespace="scerevisiae_s288c_sgd_systematic",
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
# The two loss-of-function matrices
# --------------------------------------------------------------------------- #
class ReleaseRow(BaseModel):
    """One row of a loss-of-function matrix, keyed as the release keys it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cause: NaturalLossOfFunctionCause
    systematic_gene_name: str = Field(
        description="the release's gene key for the row (see "
        "NaturalLossOfFunctionPerturbation)"
    )
    release_code: int | None
    release_name: str | None
    release_gene_name: str
    span: GenomicSpan


def _span(ch: Any, start: Any, stop: Any) -> GenomicSpan:
    n = int(ch)
    if not 1 <= n <= 16:
        raise ValueError(f"loss-of-function row on chromosome {ch!r}")
    return GenomicSpan(
        chromosome=f"chr{_ROMAN[n - 1]}",
        start=int(start),
        end=int(stop),
        assembly=SGD_S288C_R64,
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
    lof: pd.DataFrame, panel: set[str]
) -> tuple[list[ReleaseRow], np.ndarray[Any, Any], list[str]]:
    """The loss-of-function matrix's rows, keyed by their ``NAME`` (and ``CODE``)."""
    isolates = _matrix_isolates(lof, LOF_META, panel)
    if lof["CODE"].isna().any() or lof["NAME"].isna().any():
        raise ValueError("1011LossOfFunction: a row without a CODE or a NAME")
    rows = [
        ReleaseRow(
            cause=NaturalLossOfFunctionCause.nonsense_or_sift,
            systematic_gene_name=str(r["NAME"]),
            release_code=int(r["CODE"]),
            release_name=str(r["NAME"]),
            release_gene_name=str(r["Gene"]),
            span=_span(r["CH"], r["START"], r["STOP"]),
        )
        for _, r in lof.iterrows()
    ]
    return rows, lof[isolates].to_numpy(dtype=np.int8), isolates


def frameshift_rows(
    frame: pd.DataFrame, pangenome: PangenomeMatrices
) -> tuple[list[ReleaseRow], np.ndarray[Any, Any], list[str]]:
    """The frameshift matrix's rows: by the code's ORF name, else by ``Gene``.

    The matrix has no ``NAME`` column. A coded row is keyed by the name the release's
    pangenome gives its code (the presence-matrix column ``X<code>.<ORF>``, the same
    name the natural absences carry); a row released without a code is keyed by its
    ``Gene``, verbatim.
    """
    isolates = _matrix_isolates(frame, FRAMESHIFT_META, set(pangenome.row_of))
    rows: list[ReleaseRow] = []
    for _, r in frame.iterrows():
        span = _span(r["CH"], r["START"], r["STOP"])
        gene = str(r["Gene"])
        code: int | None = None if pd.isna(r["CODE"]) else int(r["CODE"])
        if code is None:
            key = gene
        else:
            name = pangenome.s288c_names[pangenome.column_of_code[code]]
            if name is None:
                raise ValueError(
                    f"genesMatrix_Frameshift: CODE {code} is not a reference ORF"
                )
            key = name
        rows.append(
            ReleaseRow(
                cause=NaturalLossOfFunctionCause.homozygous_frameshift,
                systematic_gene_name=key,
                release_code=code,
                release_name=None,
                release_gene_name=gene,
                span=span,
            )
        )
    return rows, frame[isolates].to_numpy(dtype=np.int8), isolates


# --------------------------------------------------------------------------- #
# Ledger
# --------------------------------------------------------------------------- #
class LossOfFunctionTally(BaseModel):
    """Calls of one cause over the built isolates, as stored."""

    model_config = ConfigDict(extra="forbid")

    n_calls: int
    n_genes: int = Field(description="distinct release gene keys called at least once")
    n_without_pangenome_code: int


class NaturalIsolateGenotypeLedger(BaseModel):
    """What the genotypes of one build carry, counted (``preprocess/`` JSON)."""

    model_config = ConfigDict(extra="forbid")

    n_isolates: int
    n_sequence_variants: int
    n_natural_presences: int
    n_natural_absences: int
    ho_deleted: list[str]
    ploidy: dict[str, int] = Field(description="Table S1 'Ploidy' -> isolates")
    zygosity: dict[str, int]
    n_aneuploid: int
    n_euploid: int
    n_aneuploidy_not_annotated: int
    n_manipulation_not_stated: int
    loss_of_function: dict[str, LossOfFunctionTally]


# --------------------------------------------------------------------------- #
# The builder every natural-isolate loader uses
# --------------------------------------------------------------------------- #
class NaturalIsolateGenotypes:
    """Every input of an isolate genotype, read once; ``genotype(isolate)`` builds one.

    ``raw_dir`` holds ``GENOTYPE_FILES`` (linked by ``link_genotype_files`` and verified
    by the loader); ``cache_dir`` holds the tarball diff over ``isolates``. The S288C
    chromosomes are resolved from the SGD tier (sha256-verified on resolve).
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
        self._lof_ref = tier_file_ref(LOF_NAME, data_root)
        self._fs_ref = tier_file_ref(FRAMESHIFT_NAME, data_root)
        lof, lof_matrix, lof_iso = lof_rows(
            read_lof_matrix(raw / LOF_NAME), set(self.pangenome.row_of)
        )
        fs, fs_matrix, fs_iso = frameshift_rows(
            read_frameshift_matrix(raw / FRAMESHIFT_NAME), self.pangenome
        )
        self._calls: dict[str, list[ReleaseRow]] = {iso: [] for iso in self.isolates}
        for rows, matrix, columns in (
            (lof, lof_matrix, lof_iso),
            (fs, fs_matrix, fs_iso),
        ):
            built = [j for j, iso in enumerate(columns) if iso in self.isolates]
            for i, row in enumerate(rows):
                for j in built:
                    if matrix[i, j] == 1:
                        self._calls[columns[j]].append(row)

    # ---- one isolate -----------------------------------------------------------
    def _loss_of_function(
        self, isolate: str
    ) -> list[NaturalLossOfFunctionPerturbation]:
        return [
            NaturalLossOfFunctionPerturbation(
                systematic_gene_name=call.systematic_gene_name,
                gene_namespace="scerevisiae_s288c_sgd_systematic",
                perturbed_gene_name=call.systematic_gene_name,
                strain_id=isolate,
                cause=call.cause,
                call_ref=self._lof_ref
                if call.cause is NaturalLossOfFunctionCause.nonsense_or_sift
                else self._fs_ref,
                release_code=call.release_code,
                release_name=call.release_name,
                release_gene_name=call.release_gene_name,
                release_span=call.span,
            )
            for call in sorted(
                self._calls[isolate],
                key=lambda c: (
                    c.systematic_gene_name,
                    c.cause.value,
                    -1 if c.release_code is None else c.release_code,
                    c.release_gene_name,
                ),
            )
        ]

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
        tallies: dict[str, LossOfFunctionTally] = {}
        for cause in NaturalLossOfFunctionCause:
            calls = [
                call
                for iso in self.isolates
                for call in self._calls[iso]
                if call.cause is cause
            ]
            tallies[cause.value] = LossOfFunctionTally(
                n_calls=len(calls),
                n_genes=len({call.systematic_gene_name for call in calls}),
                n_without_pangenome_code=sum(
                    1 for call in calls if call.release_code is None
                ),
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
            ho_deleted=sorted(
                iso
                for iso, d in self.descriptors.items()
                if d.genetic_manipulation is IsolateManipulation.ho_deletion
            ),
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
