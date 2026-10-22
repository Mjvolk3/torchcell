# experiments/043-expression-proteome-morphology/scripts/compute_isolate_composite.py
# [[experiments.043-expression-proteome-morphology.scripts.compute_isolate_composite]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/compute_isolate_composite
"""Composite allele embeddings for every Peter 2018 isolate (1,011) and their positions.

The composite is the one the trigenic case study trained on,
``node_embeddings: [fudt_upstream, calm, prot_T5_all, fudt_downstream]``
(``experiments/030-solid-growth-multi/conf/cgt_030_s3_r_tok_embfit_003.yaml``). This
script computes it for every allele any isolate carries, keyed by the sha256 of the exact
string each encoder reads, so an identical allele is one row in each table.

Stages (each idempotent; a rerun skips work whose output exists and whose inputs match):

``extract``   CPU, one job. One pass over the gene-keyed tarball: per (isolate, reference
              gene) the released gene span, its spliced CDS (intron genes spliced with the
              R64 exon map, through a global alignment when the allele carries an indel),
              the translated protein, the CaLM input, their sha256s, the R64 position from
              the gene-file header, presence and copy number from the pangenome matrices;
              plus one row per accessory pangenome ORF the isolate carries. Writes
              ``index/extract.parquet``, the distinct ``sequences/{cds,protein,calm}``
              tables and one BLAST query FASTA per isolate.
``unpack``    CPU, one job. Streams ``1011Assemblies.tar.gz`` once and writes each member
              FASTA with its sha256 (``work/assemblies/members.json``).
``align``     CPU, slurm array over isolate chunks. BLAST+ megablast of each isolate's
              query set against its own assembly; the locus, identity, coverage and the
              available 5' and 3' flank; the 1,003 bp upstream and 300 bp downstream
              windows sliced with the reference geometry; a typed gap where the locus is
              not found, is partial, or the flank is short (never a padded window). Writes
              a derived GFF3 plus a processing record per isolate under the genomes tier
              ``derived/<isolate>/``.
``merge``     CPU, one job. Joins extract and align into ``index/isolate_gene.parquet``,
              builds the distinct window tables, adds the S288C reference context and
              records the dedup counts in ``manifest.json``.
``seed``      CPU, one job. Copies the existing S288C rows of the four composite tables
              into ``alleles/<encoder>.h5`` under the sha256 of the string each row
              actually embedded, so an isolate allele identical to S288C reuses the exact
              row the composite trained on.
``embed``     GPU. Per encoder, embeds only the distinct inputs not yet in its table
              (fp16 storage), after reproducing a sample of the S288C rows through the
              batched path and stopping if they disagree.
``verify``    CPU. Every (isolate, feature) resolves to a row in each table or carries a
              typed gap; per-isolate counts; a json report.

Window geometry (``FungalUpDownTransformerDataset.MODEL_TO_WINDOW`` and
``AnnotatedGene.window_five_prime / window_three_prime``): the upstream window is the
1,000 bp before the first CDS base plus the start codon (1,003 bp, ending on the start
codon); the downstream window is the stop codon plus the 297 bp after it (300 bp). An
isolate window is taken only when the whole flank exists on the contig.

Conventions that match the S288C tables (measured, see the script note):
ProtT5 reads the translated protein WITH its trailing ``*`` (all 6,607 stored S288C
proteins end in ``*``); CaLM reads the spliced CDS truncated to 3,072 nt, here also
trimmed to whole codons; FUDT reads the two windows above.

    python experiments/043-expression-proteome-morphology/scripts/compute_isolate_composite.py --stage extract
    python .../compute_isolate_composite.py --stage align --chunk 0 --n-chunks 64
"""

from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
import os
import os.path as osp
import re
import shutil
import socket
import subprocess
import tarfile
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from Bio.Align import PairwiseAligner
from Bio.Seq import Seq
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from torchcell.datamodels.schema import ArtifactRef
from torchcell.sequence.genome.registry import PETER2018_1011, resolve
from torchcell.sequence.natural_isolate import (
    COPYNUMBER_NAME,
    PRESENCE_NAME,
    REFGENE_TAR_NAME,
    isolate_and_symbol,
    parse_fasta,
    read_pangenome,
    reverse_complement,
    tier_file_ref,
)
from torchcell.utils.paths import experiment_results_dir

EXPERIMENT = "043-expression-proteome-morphology"
OUT_REL = "data/scerevisiae/isolate_embeddings/peter2018_1011"
DERIVED_REL = "torchcell-genomes/peter2018_1011_assemblies/derived"
ASSEMBLY_TAR_NAME = "1011Assemblies.tar.gz"
PANGENOME_NAME = "allORFs_pangenome.fasta.gz"
MANIFEST_VERSION = 1
# ``chromosome15:619840-624375\t+ 5215``; the ten YOR153W headers add a per-isolate
# ``/1-<len>`` after the strand, so the gene coordinate is groups 1-4 only
HEADER_RE = re.compile(r"chromosome(\d+):(\d+)-(\d+)\s+([+-])(?:/\d+-\d+)?\s+(\d+)")
# ``AAA_6.re.fa``; CRL's member is ``CRL_.re.fa`` (underscore, no digit)
ASSEMBLY_MEMBER_RE = re.compile(r"([A-Za-z0-9]+?)(?:_\d*)?\.re\.fa$")
BLAST_FIELDS = (
    "qseqid sseqid pident length qlen qstart qend sstart send sstrand bitscore slen"
)

EncoderName = Literal[
    "prot_t5_xl_uniref50", "calm", "fudt_species_upstream", "fudt_species_downstream"
]
InputKind = Literal["protein", "calm", "window_up", "window_down"]


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
class Strict(BaseModel):
    """Frozen, closed pydantic base for every record this script writes."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class EncoderSpec(Strict):
    """One composite member: what it reads, which S288C table it reproduces."""

    name: EncoderName
    composite_key: str = Field(description="The name in cell_dataset.node_embeddings.")
    input_kind: InputKind
    hf_model: str
    s288c_table: str = Field(description="Relative to DATA_ROOT.")
    s288c_key: str = Field(description="Key of dna_windows / embeddings in the table.")
    dim: int
    compute_dtype: Literal["float16", "float32"] = Field(
        description="Precision the S288C table was computed in, reused here."
    )


ENCODERS: dict[str, EncoderSpec] = {
    "fudt_species_upstream": EncoderSpec(
        name="fudt_species_upstream",
        composite_key="fudt_upstream",
        input_kind="window_up",
        hf_model="gagneurlab/SpeciesLM@upstream_species_lm",
        s288c_table="data/scerevisiae/fudt_embedding/processed/species_upstream.pt",
        s288c_key="species_upstream",
        dim=768,
        compute_dtype="float32",
    ),
    "calm": EncoderSpec(
        name="calm",
        composite_key="calm",
        input_kind="calm",
        hf_model="oxpig/CaLM",
        s288c_table="data/scerevisiae/calm_embedding/processed/calm.pt",
        s288c_key="calm",
        dim=768,
        compute_dtype="float32",
    ),
    "prot_t5_xl_uniref50": EncoderSpec(
        name="prot_t5_xl_uniref50",
        composite_key="prot_T5_all",
        input_kind="protein",
        hf_model="Rostlab/prot_t5_xl_uniref50",
        s288c_table=(
            "data/scerevisiae/protT5_embedding/processed/prot_t5_xl_uniref50_all.pt"
        ),
        s288c_key="prot_t5_xl_uniref50_all",
        dim=1024,
        compute_dtype="float16",
    ),
    "fudt_species_downstream": EncoderSpec(
        name="fudt_species_downstream",
        composite_key="fudt_downstream",
        input_kind="window_down",
        hf_model="gagneurlab/SpeciesLM@downstream_species_lm",
        s288c_table="data/scerevisiae/fudt_embedding/processed/species_downstream.pt",
        s288c_key="species_downstream",
        dim=768,
        compute_dtype="float32",
    ),
}
# column of the index holding each input kind's sha256, and its distinct-sequence table
INPUT_COLUMN: dict[str, str] = {
    "protein": "protein_sha256",
    "calm": "calm_sha256",
    "window_up": "up_sha256",
    "window_down": "down_sha256",
}
SEQUENCE_TABLE: dict[str, str] = {
    "protein": "sequences/protein.parquet",
    "calm": "sequences/calm.parquet",
    "window_up": "sequences/window_up_1003.parquet",
    "window_down": "sequences/window_down_300.parquet",
}


class WindowGeometry(Strict):
    """The reference window geometry (FungalUpDownTransformerDataset.MODEL_TO_WINDOW)."""

    window_up_bp: int = 1003
    window_down_bp: int = 300
    upstream_flank_bp: int = 1000
    downstream_flank_bp: int = 297
    codon_bp: int = 3
    calm_max_nt: int = 3072


class IsolateCompositeConfig(Strict):
    """Everything a stage reads; the manifest records it verbatim."""

    isolates: tuple[str, ...] = Field(description="Peter codes, SACE_ prefix kept.")
    refgene_tarball: ArtifactRef
    assembly_tarball: ArtifactRef
    pangenome_fasta: ArtifactRef
    presence_matrix: ArtifactRef
    copy_number_matrix: ArtifactRef
    encoders: tuple[EncoderName, ...] = (
        "fudt_species_upstream",
        "calm",
        "prot_t5_xl_uniref50",
        "fudt_species_downstream",
    )
    geometry: WindowGeometry = WindowGeometry()
    min_identity: float = Field(
        default=0.95, description="Below it the locus is a typed gap (fraction)."
    )
    max_end_extension_bp: int = Field(
        default=10,
        description=(
            "The best HSP may stop this many bp short of either query end; the CDS end "
            "on the contig is then extrapolated colinearly. More is a partial locus."
        ),
    )
    tie_fraction: float = Field(
        default=0.999, description="HSPs within this bitscore fraction are ties."
    )
    blastn: Path
    makeblastdb: Path
    blast_threads: int = 4
    blast_evalue: float = 1e-10
    blast_max_target_seqs: int = 5
    out_root: Path
    derived_root: Path
    genome_root: Path
    go_root: Path
    device: str = "cuda"
    max_tokens_protein: int = 12000
    max_batch_protein: int = 16
    max_tokens_calm: int = 48000
    max_batch_calm: int = 64
    max_batch_window: int = 64
    flush_every_batches: int = 200
    n_reference_check: int = 48
    min_reference_cosine: float = 0.999


class LocusGap(StrEnum):
    """Why an (isolate, feature) has no locus on the isolate's own assembly."""

    NO_ASSEMBLY = "no_assembly"
    NO_HIT = "no_hit"
    BELOW_IDENTITY = "below_identity"
    PARTIAL_ALIGNMENT = "partial_alignment"
    CDS_OFF_CONTIG = "cds_off_contig"


class WindowGap(StrEnum):
    """Why a window is absent (the locus gap, or a flank shorter than the geometry)."""

    LOCUS = "locus_gap"
    FLANK_SHORT = "flank_short"


class RepresentationGap(StrEnum):
    """An input that exists but that its encoder cannot read."""

    AMBIGUOUS_SEQUENCE = "ambiguous_sequence"


#: What each encoder's vocabulary tokenizes WITHOUT an unknown token (measured from the
#: tokenizers 2026-10-22): CaLM holds the 64 ACGU codons and its ``encode`` raises on
#: any other codon; FUDT holds the 4,096 ACGT 6-mers and maps any other 6-mer to
#: ``[UNK]`` silently; ProtT5 holds the 20 amino acids plus B, O, U, X, Z (the wrapper
#: maps U, Z, O, B to X) and maps anything else to ``<unk>``. The protein alphabet is the
#: 20 amino acids plus ``*``: every S288C ProtT5 input ends in ``*`` (read as ``<unk>``),
#: the convention the trained composite carries, so ``*`` is kept; X, B, Z and J (an
#: ambiguous codon's translation) are not. An input with any other character is a
#: typed gap ``ambiguous_sequence``: never resolved to a base, never embedded.
ENCODER_ALPHABET: dict[str, frozenset[str]] = {
    "window_up": frozenset("ACGT"),
    "window_down": frozenset("ACGT"),
    "calm": frozenset("ACGT"),
    "protein": frozenset("ACDEFGHIKLMNPQRSTVWY*"),
}
#: Per input kind, the index column holding its gap (protein and CaLM no longer share).
GAP_COLUMN: dict[str, str] = {
    "protein": "protein_gap",
    "calm": "calm_gap",
    "window_up": "up_gap",
    "window_down": "down_gap",
}
#: Per input kind, the index column holding its offending codes.
CODES_COLUMN: dict[str, str] = {
    "protein": "protein_untokenizable_codes",
    "calm": "calm_untokenizable_codes",
    "window_up": "up_untokenizable_codes",
    "window_down": "down_untokenizable_codes",
}


def untokenizable_codes(kind: str, seq: str) -> str:
    """The characters of ``seq`` outside its encoder's alphabet, sorted ('' = none)."""
    return "".join(sorted(set(seq) - ENCODER_ALPHABET[kind]))


class CdsGap(StrEnum):
    """Why an intron gene's allele could not be spliced."""

    SPLICE_UNVERIFIED = "splice_unverified"
    EXON_BOUNDARY_IN_GAP = "exon_boundary_in_alignment_gap"


class StageRecord(Strict):
    """One stage run."""

    stage: str
    started_at: str
    finished_at: str
    wall_seconds: float
    host: str
    slurm_job_id: str | None
    torchcell_commit: str
    counts: dict[str, int | float | str] = Field(default_factory=dict)


class IsolateAlignmentRecord(Strict):
    """Processing record of one isolate's derived GFF3 (beside it in the genomes tier)."""

    isolate: str
    assembly_member: str | None
    assembly_member_sha256: str | None
    assembly_tarball: ArtifactRef
    query_fasta_sha256: str
    refgene_tarball: ArtifactRef
    pangenome_fasta: ArtifactRef
    tool: str = "blastn"
    tool_version: str
    makeblastdb_version: str
    parameters: dict[str, Any]
    acceptance: dict[str, Any]
    geometry: WindowGeometry
    n_queries: int
    n_loci: int
    gaps: dict[str, int]
    created_at: str
    torchcell_commit: str


class IsolateEmbeddingManifest(Strict):
    """``manifest.json`` of the isolate composite store."""

    version: int = MANIFEST_VERSION
    assembly_set: str = PETER2018_1011
    config: IsolateCompositeConfig
    encoders: tuple[EncoderSpec, ...]
    geometry: WindowGeometry
    conventions: dict[str, str]
    tools: dict[str, str]
    dedup_counts: dict[str, int] = Field(default_factory=dict)
    representation_gaps: dict[str, Any] = Field(
        default_factory=dict,
        description="per encoder: untokenizable inputs (ambiguous_sequence), by code",
    )
    stages: dict[str, StageRecord] = Field(default_factory=dict)
    created_at: str
    updated_at: str


CONVENTIONS = {
    "protein": (
        "Biopython standard-table translation of the spliced CDS truncated to whole "
        "codons, trailing '*' kept (the S288C table keeps it on all 6,607 genes), "
        "internal '*' kept, ambiguity codes translated as Biopython does (X, B, Z, J)."
    ),
    "calm": (
        "Spliced CDS truncated to 3,072 nt and then to whole codons. The S288C table's "
        "stored string is the unspliced gene span for its 288 intron or noncontiguous "
        "genes; its vector is the spliced-CDS embedding for 266 of them and the "
        "unspliced-span embedding for 22, so the seed keys each row by the string it "
        "embedded (verified by re-embedding) and the 22 are recomputed spliced."
    ),
    "window_up": (
        "1,000 bp before the first CDS base plus the start codon, gene orientation; "
        "taken only when the whole flank lies on the contig."
    ),
    "window_down": (
        "Stop codon plus the 297 bp after it, gene orientation; taken only when the "
        "whole flank lies on the contig."
    ),
    "position": (
        "Reference position from the gene-file header (R64-1-1 coordinates, chromosome "
        "17 = mitochondrion as Peter numbers it); accessory ORFs carry none."
    ),
    "isolate_alleles": (
        "Reference genes: the tarball allele (the released gene span, SNPs and indels "
        "inferred, IUPAC codes kept). Accessory ORFs: the pangenome representative "
        "sequence; only its locus and flanks are isolate-specific."
    ),
}


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def sha256_text(text: str) -> str:
    """sha256 hex of a UTF-8 string."""
    return hashlib.sha256(text.encode()).hexdigest()


def sha256_path(path: Path) -> str:
    """sha256 hex of a file, streamed."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def now() -> str:
    """UTC timestamp, ISO 8601."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def git_commit() -> str:
    """Short HEAD of the checkout the script runs from."""
    out = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=osp.dirname(osp.abspath(__file__)),
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def tool_version(path: Path) -> str:
    """First line of ``<tool> -version``."""
    out = subprocess.run(
        [str(path), "-version"], capture_output=True, text=True, check=True
    )
    return out.stdout.splitlines()[0].strip()


def translate(cds: str) -> str:
    """Standard-table translation of whole codons, every ``*`` kept."""
    usable = cds[: len(cds) - len(cds) % 3]
    return str(Seq(usable).translate())  # type: ignore[no-untyped-call]


def calm_input(cds: str, max_nt: int) -> str:
    """CDS truncated to ``max_nt`` and then to whole codons."""
    head = cds[:max_nt]
    return head[: len(head) - len(head) % 3]


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write atomically (temp file then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def write_json(obj: Any, path: Path) -> None:
    """Write json atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


def read_fasta_file(path: Path) -> dict[str, str]:
    """``{first header token: sequence}`` of a plain FASTA."""
    seqs: dict[str, str] = {}
    cid: str | None = None
    buf: list[str] = []
    with open(path, encoding="latin-1") as fh:
        for line in fh:
            if line.startswith(">"):
                if cid is not None:
                    seqs[cid] = "".join(buf)
                cid = line[1:].split()[0]
                buf = []
            else:
                buf.append(line.strip())
    if cid is not None:
        seqs[cid] = "".join(buf)
    return seqs


def assembly_key(isolate: str) -> str:
    """Isolate code -> assembly member key (a ``SACE_`` prefix is dropped for lookup)."""
    return isolate.split("_")[-1] if isolate.startswith("SACE_") else isolate


# --------------------------------------------------------------------------- #
# Configuration and manifest
# --------------------------------------------------------------------------- #
def build_config(
    data_root: str, out_root: Path | None = None
) -> IsolateCompositeConfig:
    """The config of this run: all 1,011 isolates, tier refs from the Peter manifest.

    ``out_root`` overrides the store (a smoke run); the derived GFF3s then go under it
    too, never into the genomes tier.
    """
    distinct = (
        Path(experiment_results_dir(EXPERIMENT, __file__)) / "distinct_alleles.json"
    )
    isolates = tuple(json.loads(distinct.read_text())["isolates"])
    if len(isolates) != 1011:
        raise ValueError(f"expected 1011 isolates in {distinct}, found {len(isolates)}")
    derived_root = Path(data_root) / DERIVED_REL
    if out_root is None:
        out_root = Path(data_root) / OUT_REL
    else:
        derived_root = out_root / "derived"
    return IsolateCompositeConfig(
        isolates=isolates,
        refgene_tarball=tier_file_ref(REFGENE_TAR_NAME, data_root),
        assembly_tarball=tier_file_ref(ASSEMBLY_TAR_NAME, data_root),
        pangenome_fasta=tier_file_ref(PANGENOME_NAME, data_root),
        presence_matrix=tier_file_ref(PRESENCE_NAME, data_root),
        copy_number_matrix=tier_file_ref(COPYNUMBER_NAME, data_root),
        blastn=Path(
            os.environ.get("TC_BLASTN", "~/miniconda3/envs/biomni_e1/bin/blastn")
        ).expanduser(),
        makeblastdb=Path(
            os.environ.get(
                "TC_MAKEBLASTDB", "~/miniconda3/envs/biomni_e1/bin/makeblastdb"
            )
        ).expanduser(),
        out_root=out_root,
        derived_root=derived_root,
        genome_root=Path(data_root) / "data/sgd/genome",
        go_root=Path(data_root) / "data/go",
    )


def tier_path(name: str, verify: bool, data_root: str) -> Path:
    """Absolute path of a Peter-tier file, sha256-verified when asked."""
    return Path(resolve(PETER2018_1011, name, verify=verify, data_root=data_root))


def load_manifest(cfg: IsolateCompositeConfig) -> IsolateEmbeddingManifest:
    """The store manifest, created on first use."""
    path = cfg.out_root / "manifest.json"
    if path.exists():
        manifest = IsolateEmbeddingManifest.model_validate_json(path.read_text())
        if manifest.config != cfg:
            raise ValueError(
                f"{path} was written under a different config; a changed config is a "
                "new store, not a rerun"
            )
        return manifest
    return IsolateEmbeddingManifest(
        config=cfg,
        encoders=tuple(ENCODERS[e] for e in cfg.encoders),
        geometry=cfg.geometry,
        conventions=CONVENTIONS,
        tools={
            "blastn": tool_version(cfg.blastn),
            "makeblastdb": tool_version(cfg.makeblastdb),
        },
        created_at=now(),
        updated_at=now(),
    )


def record_stage(
    cfg: IsolateCompositeConfig,
    stage: str,
    t0: float,
    started: str,
    counts: dict[str, int | float | str],
    dedup: dict[str, int] | None = None,
    representation_gaps: dict[str, Any] | None = None,
) -> StageRecord:
    """Add a stage record (and optional dedup counts) to the manifest and save it."""
    rec = StageRecord(
        stage=stage,
        started_at=started,
        finished_at=now(),
        wall_seconds=round(time.time() - t0, 1),
        host=socket.gethostname(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        torchcell_commit=git_commit(),
        counts=counts,
    )
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    with open(cfg.out_root / ".manifest.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = load_manifest(cfg)
        manifest = manifest.model_copy(
            update={
                "stages": {**manifest.stages, stage: rec},
                "dedup_counts": {**manifest.dedup_counts, **(dedup or {})},
                "representation_gaps": {
                    **manifest.representation_gaps,
                    **(representation_gaps or {}),
                },
                "updated_at": now(),
            }
        )
        write_json(manifest.model_dump(mode="json"), cfg.out_root / "manifest.json")
    print(json.dumps(rec.model_dump(mode="json"), indent=1))
    return rec


# --------------------------------------------------------------------------- #
# Stage 1: extract
# --------------------------------------------------------------------------- #
class ReferenceGeneModel(Strict):
    """The R64 model of one tarball gene: header span, exon map, agreement flags."""

    gene: str
    chromosome: int
    start: int
    end: int
    strand: str
    header_order: int
    in_sgd_gene_set: bool
    sgd_model_agrees: bool = Field(
        description="SGD R64-4-1 gene span equals the header span (the S288C windows "
        "were anchored on the SGD span)."
    )
    rel_exons: tuple[tuple[int, int], ...] | None = Field(
        description="Half-open exon intervals relative to the oriented header span, "
        "None for a contiguous gene."
    )
    splice_verified: bool
    ref_span: str


def reference_models(
    cfg: IsolateCompositeConfig, headers: dict[str, str]
) -> dict[str, ReferenceGeneModel]:
    """R64 models of every tarball gene from its header and the SGD genome."""
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

    genome = SCerevisiaeGenome(
        genome_root=str(cfg.genome_root), go_root=str(cfg.go_root), overwrite=False
    )
    gene_set = set(genome.gene_set)
    nc_of = {c: genome.chr_to_nc[c] for c in genome.chr_to_nc}
    models: dict[str, ReferenceGeneModel] = {}
    for gene, header in headers.items():
        m = HEADER_RE.search(header)
        if m is None:
            raise ValueError(f"{gene}: unparseable header {header!r}")
        chrom, start, end, strand, order = (
            int(m[1]),
            int(m[2]),
            int(m[3]),
            m[4],
            int(m[5]),
        )
        sgd_chrom = 0 if chrom == 17 else chrom
        raw = str(genome.fasta_dna[nc_of[sgd_chrom]].seq[start - 1 : end]).upper()
        ref_span = raw if strand == "+" else reverse_complement(raw)
        in_set = gene in gene_set
        agrees = False
        rel_exons: tuple[tuple[int, int], ...] | None = None
        verified = True
        if in_set:
            g = genome[gene]
            agrees = (g.chromosome, g.start, g.end, g.strand) == (
                sgd_chrom,
                start,
                end,
                strand,
            )
            cds_feats = sorted(
                (f.start, f.end)
                for f in genome.db.children(g.feature_id, featuretype="CDS")
            )
            sgd_cds = str(g.cds.seq).upper()
            contiguous = len(cds_feats) == 1 and cds_feats[0] == (start, end)
            if not contiguous and len(cds_feats) > 0:
                if strand == "+":
                    rel = [(a - start, b - start + 1) for a, b in cds_feats]
                else:
                    rel = [(end - b, end - a + 1) for a, b in reversed(cds_feats)]
                inside = all(0 <= s < e <= len(ref_span) for s, e in rel)
                spliced = "".join(ref_span[s:e] for s, e in rel) if inside else ""
                if spliced == sgd_cds:
                    rel_exons = tuple(rel)
                elif ref_span == sgd_cds:
                    rel_exons = None
                else:
                    verified = False
            elif not contiguous:
                verified = ref_span == sgd_cds
        models[gene] = ReferenceGeneModel(
            gene=gene,
            chromosome=chrom,
            start=start,
            end=end,
            strand=strand,
            header_order=order,
            in_sgd_gene_set=in_set,
            sgd_model_agrees=agrees,
            rel_exons=rel_exons,
            splice_verified=verified,
            ref_span=ref_span,
        )
    return models


_ALIGNER: PairwiseAligner | None = None


def aligner() -> PairwiseAligner:
    """Global aligner for mapping exon boundaries onto an indel-carrying allele."""
    global _ALIGNER
    if _ALIGNER is None:
        a = PairwiseAligner()
        a.mode = "global"
        a.match_score = 2
        a.mismatch_score = -3
        a.open_gap_score = -6
        a.extend_gap_score = -1
        _ALIGNER = a
    return _ALIGNER


def splice_allele(
    model: ReferenceGeneModel, allele: str
) -> tuple[str | None, str, CdsGap | None]:
    """Spliced CDS of one allele: ``(cds, method, gap)``.

    Contiguous genes return the allele itself. An intron gene whose allele has the
    reference span length is cut at the reference exon offsets; otherwise the allele is
    globally aligned to the reference span and each exon boundary is carried through
    the aligned blocks. A boundary that falls in an alignment gap is a typed gap.
    """
    if not model.splice_verified:
        return None, "unverified", CdsGap.SPLICE_UNVERIFIED
    if model.rel_exons is None:
        return allele, "contiguous", None
    if len(allele) == len(model.ref_span):
        return "".join(allele[s:e] for s, e in model.rel_exons), "direct", None
    aln = aligner().align(model.ref_span, allele)[0]
    ref_blocks, alt_blocks = aln.aligned

    def carry(p: int) -> int | None:
        for (rs, re_), (as_, _) in zip(ref_blocks, alt_blocks, strict=True):
            if rs <= p < re_:
                return int(as_ + (p - rs))
        return None

    pieces: list[str] = []
    for s, e in model.rel_exons:
        a, b = carry(s), carry(e - 1)
        if a is None or b is None:
            return None, "aligned", CdsGap.EXON_BOUNDARY_IN_GAP
        pieces.append(allele[a : b + 1])
    return "".join(pieces), "aligned", None


class SequenceStore:
    """Distinct sequences of one kind keyed by sha256 (kept in memory, written once)."""

    def __init__(self) -> None:
        """Start empty."""
        self.seqs: dict[str, str] = {}

    def add(self, seq: str) -> str:
        """Hash, remember, return the sha256."""
        h = sha256_text(seq)
        self.seqs.setdefault(h, seq)
        return h

    def write(self, path: Path) -> int:
        """Write ``(sha256, length, sequence)`` sorted by length; return the row count."""
        items = sorted(self.seqs.items(), key=lambda kv: (len(kv[1]), kv[0]))
        frame = pd.DataFrame(
            {
                "sha256": [k for k, _ in items],
                "length": np.array([len(v) for _, v in items], dtype=np.int32),
                "sequence": [v for _, v in items],
            }
        )
        write_parquet(frame, path)
        return len(frame)


def stage_extract(
    cfg: IsolateCompositeConfig, data_root: str, limit: int | None = None
) -> None:
    """Per (isolate, feature): allele, CDS, protein, CaLM input, hashes, position."""
    t0, started = time.time(), now()
    out = cfg.out_root
    if (out / "index/extract.parquet").exists():
        print("extract: index/extract.parquet exists, nothing to do")
        return
    tar_path = tier_path(REFGENE_TAR_NAME, True, data_root)
    pang_path = tier_path(PANGENOME_NAME, True, data_root)
    presence_path = tier_path(PRESENCE_NAME, True, data_root)
    copy_path = tier_path(COPYNUMBER_NAME, True, data_root)
    wanted = set(cfg.isolates)
    pang = read_pangenome(presence_path, copy_path)
    missing_rows = sorted(wanted - set(pang.row_of))
    if missing_rows:
        raise ValueError(f"isolates missing from the presence matrix: {missing_rows}")
    ref_cols: dict[str, list[int]] = defaultdict(list)
    for j, name in enumerate(pang.s288c_names):
        if name is not None:
            ref_cols[name].append(j)

    # pass 1 over the tarball: headers only (first record of each gene file)
    headers: dict[str, str] = {}
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf:
            if member.isfile() and member.name.endswith(".fasta"):
                handle = tf.extractfile(member)
                if handle is None:
                    raise RuntimeError(member.name)
                headers[Path(member.name).stem] = handle.readline().decode()
                if limit is not None and len(headers) >= limit:
                    break
    models = reference_models(cfg, headers)
    print(f"extract: {len(models)} gene models, {time.time() - t0:.0f} s")

    queries_dir = out / "work/queries"
    queries_dir.mkdir(parents=True, exist_ok=True)
    handles = {iso: open(queries_dir / f"{iso}.fa.tmp", "w") for iso in sorted(wanted)}
    cds_store, prot_store, calm_store, span_store = (
        SequenceStore(),
        SequenceStore(),
        SequenceStore(),
        SequenceStore(),
    )
    cols: dict[str, list[Any]] = defaultdict(list)
    tally: Counter[str] = Counter()

    def add_row(**kw: Any) -> None:
        for k, v in kw.items():
            cols[k].append(v)

    with tarfile.open(tar_path, "r:gz") as tf:
        for n_files, member in enumerate(tf, start=1):
            if not (member.isfile() and member.name.endswith(".fasta")):
                continue
            gene = Path(member.name).stem
            if gene not in models:
                break  # only under --limit-gene-files
            model = models[gene]
            handle = tf.extractfile(member)
            if handle is None:
                raise RuntimeError(member.name)
            memo: dict[str, tuple[Any, ...]] = {}
            jcols = ref_cols.get(gene, [])
            gene_match = HEADER_RE.search(headers[gene])
            if gene_match is None:
                raise ValueError(f"{gene}: unparseable header")
            gene_coord = gene_match.groups()[:4]
            for header, raw in parse_fasta(handle.read().decode()):
                token = header.split()[0]
                iso, symbol = isolate_and_symbol(token, gene)
                if iso not in wanted:
                    continue
                coord = HEADER_RE.search(header)
                if coord is None or coord.groups()[:4] != gene_coord:
                    raise ValueError(f"{gene}/{iso}: header coordinate differs")
                allele = raw.upper()
                if allele not in memo:
                    cds, method, cds_gap = splice_allele(model, allele)
                    tally[f"splice_{method}"] += 1
                    if cds is None:
                        memo[allele] = (
                            span_store.add(allele),
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            method,
                            cds_gap.value if cds_gap else None,
                            False,
                        )
                    else:
                        prot = translate(cds)
                        memo[allele] = (
                            span_store.add(allele),
                            cds_store.add(cds),
                            len(cds),
                            prot_store.add(prot),
                            len(prot),
                            calm_store.add(calm_input(cds, cfg.geometry.calm_max_nt)),
                            len(cds) % 3,
                            method,
                            None,
                            "*" in prot[:-1],
                        )
                (
                    span_sha,
                    cds_sha,
                    cds_len,
                    prot_sha,
                    prot_len,
                    calm_sha,
                    mod3,
                    method,
                    cds_gap_v,
                    internal_stop,
                ) = memo[allele]
                row = pang.row_of[iso]
                if jcols:
                    pres = [pang.presence[row, j] for j in jcols]
                    state = "present" if any(p == 1 for p in pres) else "absent"
                    cn: float | None = (
                        float(pang.copy_number[row, jcols[0]])
                        if len(jcols) == 1
                        else None
                    )
                else:
                    state, cn = "not_in_matrix", None
                add_row(
                    isolate=iso,
                    feature_id=gene,
                    feature_kind="reference_gene",
                    symbol=symbol,
                    state=state,
                    copy_number=cn,
                    n_matrix_columns=len(jcols),
                    ref_chromosome=model.chromosome,
                    ref_start=model.start,
                    ref_end=model.end,
                    ref_strand=model.strand,
                    ref_header_order=model.header_order,
                    in_sgd_gene_set=model.in_sgd_gene_set,
                    sgd_model_agrees=model.sgd_model_agrees,
                    is_reference_allele=allele == model.ref_span,
                    span_sha256=span_sha,
                    span_len=len(allele),
                    n_ambiguous_bases=len(allele)
                    - sum(allele.count(c) for c in "ACGT"),
                    splice_method=method,
                    cds_gap=cds_gap_v,
                    cds_sha256=cds_sha,
                    cds_len=cds_len,
                    cds_len_mod3=mod3,
                    protein_sha256=prot_sha,
                    protein_len=prot_len,
                    protein_internal_stop=internal_stop,
                    calm_sha256=calm_sha,
                )
                handles[iso].write(f">{gene}\n{allele}\n")
            if n_files % 500 == 0:
                print(
                    f"extract: {n_files} gene files, {len(cols['isolate'])} rows, "
                    f"{time.time() - t0:.0f} s",
                    flush=True,
                )

    # accessory pangenome ORFs each isolate carries (presence == 1, no S288C name)
    # the FASTA keeps the release's hyphens where the matrix column (R make.names) has
    # dots, so records are joined by the unique leading ORF number, and the row keeps
    # the matrix-derived id the genotypes use (pangenome_orf_id)
    with gzip.open(pang_path, "rt") as fh:
        pang_records = [(h.split()[0], s.upper()) for h, s in parse_fasta(fh.read())]
    pang_by_code = {int(h.split("-", 1)[0]): (h, s) for h, s in pang_records}
    if len(pang_by_code) != len(pang_records):
        raise ValueError("two pangenome FASTA records carry one ORF number")
    code_of = {code: j for code, j in pang.column_of_code.items()}
    col_code = {j: code for code, j in code_of.items()}
    acc_cols = [j for j, n in enumerate(pang.s288c_names) if n is None]
    no_seq = [pang.orf_ids[j] for j in acc_cols if col_code[j] not in pang_by_code]
    if no_seq:
        raise ValueError(
            f"{len(no_seq)} accessory ORFs have no pangenome record: {no_seq[:5]}"
        )
    acc_memo: dict[str, tuple[Any, ...]] = {}
    for iso in sorted(wanted):
        row = pang.row_of[iso]
        for j in acc_cols:
            if pang.presence[row, j] != 1:
                continue
            orf = pang.orf_ids[j]
            fasta_id, seq = pang_by_code[col_code[j]]
            if re.sub(r"[^A-Za-z0-9_]", ".", fasta_id) != re.sub(
                r"[^A-Za-z0-9_]", ".", orf
            ):
                raise ValueError(f"{orf}: FASTA record {fasta_id} names another ORF")
            if orf not in acc_memo:
                prot = translate(seq)
                acc_memo[orf] = (
                    span_store.add(seq),
                    cds_store.add(seq),
                    prot_store.add(prot),
                    len(prot),
                    calm_store.add(calm_input(seq, cfg.geometry.calm_max_nt)),
                    "*" in prot[:-1],
                )
            span_sha, cds_sha, prot_sha, prot_len, calm_sha, internal_stop = acc_memo[
                orf
            ]
            cnv = pang.copy_number[row, j]
            add_row(
                isolate=iso,
                feature_id=orf,
                feature_kind="accessory_orf",
                symbol=None,
                state="present",
                copy_number=float(cnv) if np.isfinite(cnv) else None,
                n_matrix_columns=1,
                ref_chromosome=None,
                ref_start=None,
                ref_end=None,
                ref_strand=None,
                ref_header_order=None,
                in_sgd_gene_set=False,
                sgd_model_agrees=False,
                is_reference_allele=False,
                span_sha256=span_sha,
                span_len=len(seq),
                n_ambiguous_bases=len(seq) - sum(seq.count(c) for c in "ACGT"),
                splice_method="contiguous",
                cds_gap=None,
                cds_sha256=cds_sha,
                cds_len=len(seq),
                cds_len_mod3=len(seq) % 3,
                protein_sha256=prot_sha,
                protein_len=prot_len,
                protein_internal_stop=internal_stop,
                calm_sha256=calm_sha,
            )
            handles[iso].write(f">{orf}\n{seq}\n")
    for iso, fh in handles.items():
        fh.close()
        os.replace(queries_dir / f"{iso}.fa.tmp", queries_dir / f"{iso}.fa")

    frame = pd.DataFrame(cols)
    for c in ("copy_number",):
        frame[c] = frame[c].astype("float64")
    for c in (
        "ref_chromosome",
        "ref_start",
        "ref_end",
        "ref_header_order",
        "cds_len",
        "cds_len_mod3",
        "protein_len",
    ):
        frame[c] = frame[c].astype("Int64")
    frame = frame.sort_values(["isolate", "feature_kind", "feature_id"], kind="stable")
    dedup = {
        "distinct_span": span_store.write(out / "sequences/span.parquet"),
        "distinct_cds": cds_store.write(out / "sequences/cds.parquet"),
        "distinct_protein": prot_store.write(out / SEQUENCE_TABLE["protein"]),
        "distinct_calm": calm_store.write(out / SEQUENCE_TABLE["calm"]),
    }
    models_frame = pd.DataFrame(
        [
            m.model_dump(exclude={"ref_span"}) | {"ref_span_len": len(m.ref_span)}
            for m in models.values()
        ]
    )
    models_frame["rel_exons"] = models_frame["rel_exons"].map(
        lambda v: None if v is None else json.dumps(v)
    )
    write_parquet(models_frame, out / "index/reference_gene_models.parquet")
    write_parquet(frame, out / "index/extract.parquet")
    kinds = frame["feature_kind"].value_counts().to_dict()
    record_stage(
        cfg,
        "extract",
        t0,
        started,
        counts={
            "rows": len(frame),
            "isolates": int(frame["isolate"].nunique()),
            "reference_gene_rows": int(kinds.get("reference_gene", 0)),
            "accessory_orf_rows": int(kinds.get("accessory_orf", 0)),
            "distinct_accessory_orfs": len(acc_memo),
            "gene_models": len(models),
            "intron_gene_models": sum(m.rel_exons is not None for m in models.values()),
            "splice_unverified_models": sum(
                not m.splice_verified for m in models.values()
            ),
            "sgd_model_disagrees": sum(
                m.in_sgd_gene_set and not m.sgd_model_agrees for m in models.values()
            ),
            "not_in_sgd_gene_set": sum(not m.in_sgd_gene_set for m in models.values()),
            **{k: int(v) for k, v in tally.items()},
            "cds_gap_rows": int(frame["cds_gap"].notna().sum()),
            "rows_cds_len_not_mod3": int((frame["cds_len_mod3"].fillna(0) != 0).sum()),
            "rows_internal_stop": int(frame["protein_internal_stop"].sum()),
        },
        dedup=dedup,
    )


# --------------------------------------------------------------------------- #
# Stage 2a: unpack the assemblies
# --------------------------------------------------------------------------- #
def stage_unpack(cfg: IsolateCompositeConfig, data_root: str) -> None:
    """Stream the assembly tarball once; write each member FASTA with its sha256."""
    t0, started = time.time(), now()
    adir = cfg.out_root / "work/assemblies"
    members_json = adir / "members.json"
    if members_json.exists():
        print("unpack: members.json exists, nothing to do")
        return
    tar_path = tier_path(ASSEMBLY_TAR_NAME, True, data_root)
    adir.mkdir(parents=True, exist_ok=True)
    members: dict[str, dict[str, Any]] = {}
    other: list[str] = []
    with tarfile.open(tar_path, "r:gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            match = ASSEMBLY_MEMBER_RE.match(osp.basename(m.name))
            if match is None:
                other.append(m.name)
                continue
            key = match.group(1)
            if key in members:
                raise ValueError(f"two assembly members for {key}: {m.name}")
            src = tf.extractfile(m)
            if src is None:
                raise RuntimeError(m.name)
            h = hashlib.sha256()
            dest = adir / f"{key}.fa"
            with open(dest.with_suffix(".fa.tmp"), "wb") as fh:
                for block in iter(lambda: src.read(1 << 20), b""):
                    h.update(block)
                    fh.write(block)
            os.replace(dest.with_suffix(".fa.tmp"), dest)
            members[key] = {"member": m.name, "bytes": m.size, "sha256": h.hexdigest()}
    keys = {assembly_key(i) for i in cfg.isolates}
    without = sorted(i for i in cfg.isolates if assembly_key(i) not in members)
    write_json(
        {
            "tarball": cfg.assembly_tarball.model_dump(mode="json"),
            "members": members,
            "non_assembly_members": other,
            "isolates_without_member": without,
            "members_not_isolates": sorted(set(members) - keys),
        },
        members_json,
    )
    record_stage(
        cfg,
        "unpack",
        t0,
        started,
        counts={
            "assembly_members": len(members),
            "non_assembly_members": len(other),
            "isolates_without_member": len(without),
            "isolates_without_member_ids": ",".join(without),
        },
    )


# --------------------------------------------------------------------------- #
# Stage 2b: align and slice
# --------------------------------------------------------------------------- #
class Hsp(Strict):
    """One BLAST HSP of a query on the isolate assembly (``outfmt 6`` fields)."""

    contig: str
    pident: float
    length: int
    qlen: int
    qstart: int
    qend: int
    sstart: int
    send: int
    strand: str
    bitscore: float
    slen: int


def implied_cds(h: Hsp) -> tuple[int, int]:
    """0-based half-open CDS span on the contig, the HSP extended colinearly to both
    query ends (the HSP may stop short of the ends by a few mismatching bases).
    """
    left, right = h.qstart - 1, h.qlen - h.qend
    if h.strand == "plus":
        return h.sstart - 1 - left, h.send + right
    # minus: query start sits at the high contig coordinate sstart
    return h.send - 1 - right, h.sstart + left


def slice_windows(
    contig: str, lo: int, hi: int, strand: str, geo: WindowGeometry
) -> tuple[str | None, str | None, int, int]:
    """``(up, down, flank5_available, flank3_available)``; a window is None when its
    flank is shorter than the geometry (never padded, never truncated).
    """
    n, up_f, dn_f, c = (
        len(contig),
        geo.upstream_flank_bp,
        geo.downstream_flank_bp,
        geo.codon_bp,
    )
    if strand == "plus":
        f5, f3 = lo, n - hi
        up = contig[lo - up_f : lo + c] if f5 >= up_f else None
        down = contig[hi - c : hi + dn_f] if f3 >= dn_f else None
    else:
        f5, f3 = n - hi, lo
        up = reverse_complement(contig[hi - c : hi + up_f]) if f5 >= up_f else None
        down = reverse_complement(contig[lo - dn_f : lo + c]) if f3 >= dn_f else None
    return up, down, f5, f3


def overlaps(a: Hsp, b: Hsp) -> bool:
    """Same contig and overlapping subject spans."""
    if a.contig != b.contig:
        return False
    a0, a1 = sorted((a.sstart, a.send))
    b0, b1 = sorted((b.sstart, b.send))
    return a0 <= b1 and b0 <= a1


def run_blast(
    cfg: IsolateCompositeConfig, query: Path, assembly: Path, work: Path
) -> dict[str, list[Hsp]]:
    """Build the BLAST db and run megablast; HSPs per query."""
    work.mkdir(parents=True, exist_ok=True)
    db = work / "db"
    subprocess.run(
        [
            str(cfg.makeblastdb),
            "-in",
            str(assembly),
            "-dbtype",
            "nucl",
            "-out",
            str(db),
        ],
        check=True,
        capture_output=True,
    )
    proc = subprocess.run(
        [
            str(cfg.blastn),
            "-task",
            "megablast",
            "-query",
            str(query),
            "-db",
            str(db),
            "-outfmt",
            f"6 {BLAST_FIELDS}",
            "-evalue",
            str(cfg.blast_evalue),
            "-max_target_seqs",
            str(cfg.blast_max_target_seqs),
            "-num_threads",
            str(cfg.blast_threads),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    hits: dict[str, list[Hsp]] = defaultdict(list)
    for line in proc.stdout.splitlines():
        p = line.split("\t")
        hits[p[0]].append(
            Hsp(
                contig=p[1],
                pident=float(p[2]),
                length=int(p[3]),
                qlen=int(p[4]),
                qstart=int(p[5]),
                qend=int(p[6]),
                sstart=int(p[7]),
                send=int(p[8]),
                strand=p[9],
                bitscore=float(p[10]),
                slen=int(p[11]),
            )
        )
    return hits


def blast_parameters(cfg: IsolateCompositeConfig) -> dict[str, Any]:
    """The BLAST parameters recorded in every processing record."""
    return {
        "task": "megablast",
        "evalue": cfg.blast_evalue,
        "max_target_seqs": cfg.blast_max_target_seqs,
        "num_threads": cfg.blast_threads,
        "outfmt": f"6 {BLAST_FIELDS}",
        "makeblastdb": "-dbtype nucl",
    }


def locate(
    cfg: IsolateCompositeConfig, qseq: str, hsps: list[Hsp], contigs: dict[str, str]
) -> dict[str, Any]:
    """Pick the locus of one query and slice its windows; every outcome is a row."""
    geo = cfg.geometry
    row: dict[str, Any] = {"n_hsps": len(hsps), "n_tied": 0, "second_bitscore": None}
    if not hsps:
        return row | {"locus_gap": LocusGap.NO_HIT.value}
    best_bs = max(h.bitscore for h in hsps)
    tied = [h for h in hsps if h.bitscore >= cfg.tie_fraction * best_bs]

    def score(h: Hsp) -> tuple[int, int, str, int]:
        lo, hi = implied_cds(h)
        if lo < 0 or hi > h.slen:
            return (0, 0, h.contig, h.sstart)
        _, _, f5, f3 = slice_windows(contigs[h.contig], lo, hi, h.strand, geo)
        complete = int(f5 >= geo.upstream_flank_bp) + int(f3 >= geo.downstream_flank_bp)
        return (
            -complete,
            -(min(f5, geo.upstream_flank_bp) + min(f3, geo.downstream_flank_bp)),
            h.contig,
            h.sstart,
        )

    best = min(tied, key=score)
    distinct_loci: list[Hsp] = []
    for h in sorted(hsps, key=lambda x: -x.bitscore):
        if not any(overlaps(h, d) for d in distinct_loci):
            distinct_loci.append(h)
    others = [h.bitscore for h in distinct_loci if not overlaps(h, best)]
    lo, hi = implied_cds(best)
    row |= {
        "n_tied": sum(
            1 for h in distinct_loci if h.bitscore >= cfg.tie_fraction * best_bs
        ),
        "second_bitscore": max(others) if others else None,
        "contig": best.contig,
        "contig_len": best.slen,
        "hsp_start": min(best.sstart, best.send),
        "hsp_end": max(best.sstart, best.send),
        "qstart": best.qstart,
        "qend": best.qend,
        "pident": best.pident,
        "coverage": best.length / best.qlen,
        "bitscore": best.bitscore,
        "locus_strand": "+" if best.strand == "plus" else "-",
    }
    if best.pident < 100 * cfg.min_identity:
        return row | {"locus_gap": LocusGap.BELOW_IDENTITY.value}
    if (best.qstart - 1) > cfg.max_end_extension_bp or (
        best.qlen - best.qend
    ) > cfg.max_end_extension_bp:
        return row | {"locus_gap": LocusGap.PARTIAL_ALIGNMENT.value}
    if lo < 0 or hi > best.slen:
        return row | {"locus_gap": LocusGap.CDS_OFF_CONTIG.value}
    contig = contigs[best.contig]
    up, down, f5, f3 = slice_windows(contig, lo, hi, best.strand, geo)
    on_contig = contig[lo:hi]
    on_contig = on_contig if best.strand == "plus" else reverse_complement(on_contig)
    row |= {
        "locus_gap": None,
        "locus_start": lo + 1,
        "locus_end": hi,
        "end_extension_bp": (best.qstart - 1) + (best.qlen - best.qend),
        "flank5_available": f5,
        "flank3_available": f3,
        "start_codon_agrees": on_contig[:3] == qseq[:3],
        "stop_codon_agrees": on_contig[-3:] == qseq[-3:],
        "locus_len_minus_query": len(on_contig) - len(qseq),
        "up_gap": None if up is not None else WindowGap.FLANK_SHORT.value,
        "down_gap": None if down is not None else WindowGap.FLANK_SHORT.value,
        "_up": up,
        "_down": down,
    }
    return row


def write_gff3(path: Path, isolate: str, rows: list[dict[str, Any]]) -> None:
    """Derived GFF3 of one isolate: one feature per located query."""
    lines = ["##gff-version 3"]
    regions = sorted(
        {(r["contig"], r["contig_len"]) for r in rows if r["locus_gap"] is None}
    )
    lines += [f"##sequence-region {c} 1 {n}" for c, n in regions]
    for r in rows:
        if r["locus_gap"] is not None:
            continue
        ftype = "gene" if r["feature_kind"] == "reference_gene" else "ORF"
        attrs = {
            "ID": f"{isolate}:{r['feature_id']}",
            "Name": r["feature_id"],
            "identity": f"{r['pident']:.3f}",
            "coverage": f"{r['coverage']:.4f}",
            "qstart": r["qstart"],
            "qend": r["qend"],
            "flank5": r["flank5_available"],
            "flank3": r["flank3_available"],
            "n_tied": r["n_tied"],
            "span_sha256": r["span_sha256"],
            "up_sha256": r["up_sha256"] or "",
            "down_sha256": r["down_sha256"] or "",
        }
        attr = ";".join(f"{k}={v}" for k, v in attrs.items())
        lines.append(
            "\t".join(
                [
                    r["contig"],
                    "torchcell_blastn",
                    ftype,
                    str(r["locus_start"]),
                    str(r["locus_end"]),
                    f"{r['pident']:.3f}",
                    r["locus_strand"],
                    ".",
                    attr,
                ]
            )
        )
    tmp = path.with_suffix(".gff3.tmp")
    tmp.write_text("\n".join(lines) + "\n")
    os.replace(tmp, path)


def align_isolate(
    cfg: IsolateCompositeConfig,
    iso: str,
    members: dict[str, Any],
    tools: dict[str, str],
) -> dict[str, Any]:
    """Locus, windows, GFF3 and processing record of one isolate."""
    t0 = time.time()
    out = cfg.out_root
    query = out / "work/queries" / f"{iso}.fa"
    part = out / "work/align" / f"{iso}.parquet"
    qsha = sha256_path(query)
    record_path = cfg.derived_root / iso / "processing.json"
    if part.exists() and record_path.exists():
        rec = json.loads(record_path.read_text())
        if rec["query_fasta_sha256"] == qsha:
            return {"isolate": iso, "skipped": True}
    qseqs = read_fasta_file(query)
    key = assembly_key(iso)
    member = members["members"].get(key)
    # features of this isolate (feature_kind and span sha for the GFF attributes)
    meta = pq.read_table(
        out / "index/extract.parquet",
        columns=["isolate", "feature_id", "feature_kind", "span_sha256"],
        filters=[("isolate", "=", iso)],
    ).to_pandas()
    meta_of = {r.feature_id: r for r in meta.itertuples(index=False)}
    if set(meta_of) != set(qseqs):
        raise ValueError(f"{iso}: query FASTA and extract index disagree")
    rows: list[dict[str, Any]] = []
    if member is None:
        for fid in qseqs:
            rows.append(
                {"locus_gap": LocusGap.NO_ASSEMBLY.value, "n_hsps": 0, "n_tied": 0}
            )
        assembly_sha = None
    else:
        assembly = out / "work/assemblies" / f"{key}.fa"
        assembly_sha = sha256_path(assembly)
        if assembly_sha != member["sha256"]:
            raise ValueError(f"{iso}: assembly bytes differ from members.json")
        contigs = read_fasta_file(assembly)
        work = out / "work/blastdb" / iso
        hits = run_blast(cfg, query, assembly, work)
        shutil.rmtree(work)
        for fid, qseq in qseqs.items():
            rows.append(locate(cfg, qseq, hits.get(fid, []), contigs))
    windows: dict[str, tuple[str, str]] = {}
    for fid, r in zip(qseqs, rows, strict=True):
        m = meta_of[fid]
        r["isolate"] = iso
        r["feature_id"] = fid
        r["feature_kind"] = m.feature_kind
        r["span_sha256"] = m.span_sha256
        up, down = r.pop("_up", None), r.pop("_down", None)
        if r["locus_gap"] is not None:
            r["up_gap"] = r["down_gap"] = WindowGap.LOCUS.value
        r["up_sha256"] = sha256_text(up) if up is not None else None
        r["down_sha256"] = sha256_text(down) if down is not None else None
        if up is not None:
            windows[r["up_sha256"]] = ("window_up", up)
        if down is not None:
            windows[r["down_sha256"]] = ("window_down", down)
    frame = pd.DataFrame(rows)
    write_parquet(frame, part)
    write_parquet(
        pd.DataFrame(
            {
                "sha256": list(windows),
                "kind": [v[0] for v in windows.values()],
                "sequence": [v[1] for v in windows.values()],
            }
        ),
        out / "work/align" / f"{iso}.windows.parquet",
    )
    ddir = cfg.derived_root / iso
    ddir.mkdir(parents=True, exist_ok=True)
    write_gff3(ddir / f"{iso}.gff3", iso, rows)
    gaps = Counter(r["locus_gap"] or "located" for r in rows)
    gaps.update(f"up_{r['up_gap']}" for r in rows if r.get("up_gap"))
    gaps.update(f"down_{r['down_gap']}" for r in rows if r.get("down_gap"))
    rec = IsolateAlignmentRecord(
        isolate=iso,
        assembly_member=member["member"] if member else None,
        assembly_member_sha256=assembly_sha,
        assembly_tarball=cfg.assembly_tarball,
        query_fasta_sha256=qsha,
        refgene_tarball=cfg.refgene_tarball,
        pangenome_fasta=cfg.pangenome_fasta,
        tool_version=tools["blastn"],
        makeblastdb_version=tools["makeblastdb"],
        parameters=blast_parameters(cfg),
        acceptance={
            "min_identity": cfg.min_identity,
            "max_end_extension_bp": cfg.max_end_extension_bp,
            "tie_fraction": cfg.tie_fraction,
            "tie_break": "complete windows, then available flank, then contig name",
        },
        geometry=cfg.geometry,
        n_queries=len(rows),
        n_loci=gaps["located"],
        gaps=dict(gaps),
        created_at=now(),
        torchcell_commit=git_commit(),
    )
    write_json(rec.model_dump(mode="json"), record_path)
    return {"isolate": iso, "seconds": round(time.time() - t0, 1), **dict(gaps)}


def stage_align(cfg: IsolateCompositeConfig, chunk: int, n_chunks: int) -> None:
    """One array task: the isolates ``cfg.isolates[chunk::n_chunks]``."""
    t0 = time.time()
    members = json.loads((cfg.out_root / "work/assemblies/members.json").read_text())
    manifest = load_manifest(cfg)
    isolates = cfg.isolates[chunk::n_chunks]
    log = []
    for iso in isolates:
        res = align_isolate(cfg, iso, members, manifest.tools)
        log.append(res)
        print(json.dumps(res), flush=True)
    write_json(
        {
            "chunk": chunk,
            "n_chunks": n_chunks,
            "seconds": round(time.time() - t0, 1),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "isolates": log,
        },
        cfg.out_root / "work/align_logs" / f"chunk_{chunk:04d}_of_{n_chunks}.json",
    )


# --------------------------------------------------------------------------- #
# Stage 2c: merge
# --------------------------------------------------------------------------- #
def s288c_reference_context(cfg: IsolateCompositeConfig) -> pd.DataFrame:
    """S288C rows under THIS script's conventions (the reference join of the index).

    Protein: the SGD protein (what the S288C ProtT5 table embedded). CaLM: the SGD CDS
    under the CaLM convention above. Windows: the genome's own window methods with the
    S288C table's arguments (undersize allowed, as that table did).
    """
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

    genome = SCerevisiaeGenome(
        genome_root=str(cfg.genome_root), go_root=str(cfg.go_root), overwrite=False
    )
    geo = cfg.geometry
    rows = []
    for gid in genome.gene_set:
        g = genome[gid]
        cds = str(g.cds.seq).upper()
        prot = str(g.protein.seq)
        up = g.window_five_prime(geo.window_up_bp, True, allow_undersize=True).seq
        down = g.window_three_prime(geo.window_down_bp, True, allow_undersize=True).seq
        rows.append(
            {
                "feature_id": gid,
                "protein": prot,
                "calm": calm_input(cds, geo.calm_max_nt),
                "window_up": up,
                "window_down": down,
                "translation_matches_sgd_protein": translate(cds) == prot,
            }
        )
    return pd.DataFrame(rows)


def stage_merge(cfg: IsolateCompositeConfig) -> None:
    """Join extract and align; distinct window tables; S288C context; dedup counts."""
    t0, started = time.time(), now()
    out = cfg.out_root
    extract = pd.read_parquet(out / "index/extract.parquet")
    parts = sorted((out / "work/align").glob("*.windows.parquet"))
    align_files = [out / "work/align" / f"{iso}.parquet" for iso in cfg.isolates]
    missing = [p.name for p in align_files if not p.exists()]
    if missing:
        raise ValueError(f"{len(missing)} isolates have no align part: {missing[:10]}")
    align = pd.concat([pd.read_parquet(p) for p in align_files], ignore_index=True)
    align = align.drop(columns=["feature_kind", "span_sha256"])
    index = extract.merge(
        align, on=["isolate", "feature_id"], how="left", validate="1:1"
    )
    if (index["locus_gap"].isna() != index["locus_start"].notna()).any():
        raise ValueError("a row has neither a locus nor a locus gap, or both")
    up_store, down_store = SequenceStore(), SequenceStore()
    for p in parts:
        w = pd.read_parquet(p)
        for sha, kind, seq in zip(w["sha256"], w["kind"], w["sequence"], strict=True):
            (up_store if kind == "window_up" else down_store).seqs.setdefault(sha, seq)

    # S288C reference context: add its four inputs to the distinct tables
    ref = s288c_reference_context(cfg)
    stores = {"protein": SequenceStore(), "calm": SequenceStore()}
    for kind in ("protein", "calm"):
        prev = pd.read_parquet(out / SEQUENCE_TABLE[kind])
        stores[kind].seqs = dict(zip(prev["sha256"], prev["sequence"], strict=True))
    ref_rows = {"feature_id": ref["feature_id"].tolist()}
    for kind, store in (
        ("protein", stores["protein"]),
        ("calm", stores["calm"]),
        ("window_up", up_store),
        ("window_down", down_store),
    ):
        ref_rows[INPUT_COLUMN[kind]] = [store.add(s) for s in ref[kind]]
    ref_frame = pd.DataFrame(ref_rows)
    ref_frame["translation_matches_sgd_protein"] = ref[
        "translation_matches_sgd_protein"
    ]
    write_parquet(ref_frame, out / "index/s288c_reference.parquet")

    dedup = {
        "distinct_protein_with_s288c": stores["protein"].write(
            out / SEQUENCE_TABLE["protein"]
        ),
        "distinct_calm_with_s288c": stores["calm"].write(out / SEQUENCE_TABLE["calm"]),
        "distinct_window_up_with_s288c": up_store.write(
            out / SEQUENCE_TABLE["window_up"]
        ),
        "distinct_window_down_with_s288c": down_store.write(
            out / SEQUENCE_TABLE["window_down"]
        ),
        "distinct_window_up_isolates": int(index["up_sha256"].nunique()),
        "distinct_window_down_isolates": int(index["down_sha256"].nunique()),
        "distinct_protein_isolates": int(index["protein_sha256"].nunique()),
        "distinct_calm_isolates": int(index["calm_sha256"].nunique()),
        "distinct_cds_isolates": int(index["cds_sha256"].nunique()),
    }
    write_parquet(index, out / "index/isolate_gene.parquet")
    record_stage(
        cfg,
        "merge",
        t0,
        started,
        counts={
            "rows": len(index),
            "located": int(index["locus_gap"].isna().sum()),
            **{
                f"locus_gap_{k}": int(v)
                for k, v in index["locus_gap"].value_counts().items()
            },
            "up_window": int(index["up_sha256"].notna().sum()),
            "down_window": int(index["down_sha256"].notna().sum()),
            "s288c_genes": len(ref_frame),
            "s288c_translation_mismatch": int(
                (~ref_frame["translation_matches_sgd_protein"]).sum()
            ),
        },
        dedup=dedup,
    )


# --------------------------------------------------------------------------- #
# Stage 2d: alphabet (typed gaps for untokenizable inputs)
# --------------------------------------------------------------------------- #
KIND_ENCODER: dict[str, str] = {
    spec.input_kind: name for name, spec in ENCODERS.items()
}


def stage_alphabet(cfg: IsolateCompositeConfig) -> None:
    """Typed gap ``ambiguous_sequence`` for every input its encoder cannot tokenize.

    Per input kind: the distinct inputs carrying a character outside the encoder's
    alphabet (``ENCODER_ALPHABET``), by code, written to
    ``sequences/untokenizable_<kind>.parquet``. In the index each affected row gets its
    own per-encoder gap (``protein_gap`` and ``calm_gap`` now separate from
    ``cds_gap``), its offending codes, and its hash moved to ``<col>_untokenizable``
    (the input column is emptied, so a row still has a key XOR a gap). Rows an encoder
    already embedded from such an input (the FUDT tokenizer maps them to ``[UNK]``
    without failing) are relabelled ``untokenizable`` in the h5 ``source`` column and
    are referenced by nothing. Counts go to the manifest and
    ``results/compute_isolate_composite_alphabet.json``.
    """
    import h5py

    t0, started = time.time(), now()
    out = cfg.out_root
    index_path = out / "index/isolate_gene.parquet"
    index = pd.read_parquet(index_path)
    if "calm_gap" in index.columns:
        print("alphabet: index already carries the per-encoder gaps, nothing to do")
        return
    ref = pd.read_parquet(out / "index/s288c_reference.parquet")
    index["protein_gap"] = index["cds_gap"]
    index["calm_gap"] = index["cds_gap"]
    report: dict[str, Any] = {}
    for kind in ("window_up", "window_down", "calm", "protein"):
        name = KIND_ENCODER[kind]
        table = pq.read_table(
            out / SEQUENCE_TABLE[kind], columns=["sha256", "sequence"]
        )
        bad: dict[str, str] = {}
        for k, sq in zip(
            table["sha256"].to_pylist(), table["sequence"].to_pylist(), strict=True
        ):
            c = untokenizable_codes(kind, sq)
            if c:
                bad[k] = c
        n_distinct = table.num_rows
        del table
        write_parquet(
            pd.DataFrame({"sha256": list(bad), "codes": list(bad.values())}),
            out / f"sequences/untokenizable_{kind}.parquet",
        )
        col = INPUT_COLUMN[kind]
        has_input = index[col].notna()
        codes = index[col].map(bad)
        mask = codes.notna()
        if (mask & index[GAP_COLUMN[kind]].notna()).any():
            raise ValueError(f"{kind}: a row carries both an input and a gap")
        index[CODES_COLUMN[kind]] = codes.where(mask, None)
        index[f"{col}_untokenizable"] = index[col].where(mask, None)
        index.loc[mask, GAP_COLUMN[kind]] = RepresentationGap.AMBIGUOUS_SEQUENCE.value
        index.loc[mask, col] = None
        rows_by_code = Counter(ch for c in codes[mask] for ch in c)
        distinct_by_code = Counter(ch for c in bad.values() for ch in c)
        s288c_bad = sorted(ref.loc[ref[col].isin(set(bad)), "feature_id"])
        relabelled = 0
        h5 = h5_path(cfg, name)
        if h5.exists():
            with h5py.File(h5, "a") as f:
                keys = np.array([k.decode() for k in f["sha256"][:]])
                hit = np.nonzero(np.isin(keys, list(bad)))[0]
                if hit.size:
                    src = f["source"][:]
                    src[hit] = b"untokenizable"
                    f["source"][:] = src
                relabelled = int(hit.size)
        report[name] = {
            "input_kind": kind,
            "alphabet": "".join(sorted(ENCODER_ALPHABET[kind])),
            "distinct_inputs": n_distinct,
            "distinct_untokenizable": len(bad),
            "distinct_untokenizable_by_code": dict(sorted(distinct_by_code.items())),
            "rows_with_input_before": int(has_input.sum()),
            "rows_untokenizable": int(mask.sum()),
            "rows_untokenizable_by_code": dict(sorted(rows_by_code.items())),
            "rows_untokenizable_by_feature_kind": index.loc[mask, "feature_kind"]
            .value_counts()
            .to_dict(),
            "isolates_touched": int(index.loc[mask, "isolate"].nunique()),
            "s288c_reference_inputs_untokenizable": s288c_bad,
            "h5_rows_already_embedded_relabelled": relabelled,
        }
        print(json.dumps({name: report[name]}, indent=1), flush=True)
    write_parquet(index, index_path)
    res = {
        "generated_by": f"experiments/{EXPERIMENT}/scripts/compute_isolate_composite.py",
        "stage": "alphabet",
        "created_at": now(),
        "rule": (
            "an input with a character outside its encoder's alphabet is a typed gap "
            "ambiguous_sequence: never resolved to a base, never embedded"
        ),
        "encoders": report,
    }
    write_json(
        res,
        Path(experiment_results_dir(EXPERIMENT, __file__))
        / "compute_isolate_composite_alphabet.json",
    )
    record_stage(
        cfg,
        "alphabet",
        t0,
        started,
        counts={
            f"{n}_rows_untokenizable": r["rows_untokenizable"]
            for n, r in report.items()
        },
        representation_gaps=report,
    )


# --------------------------------------------------------------------------- #
# Stage 3a: seed the tables with the S288C rows the composite trained on
# --------------------------------------------------------------------------- #
def h5_path(cfg: IsolateCompositeConfig, encoder: str) -> Path:
    """``alleles/<encoder>.h5``."""
    return cfg.out_root / "alleles" / f"{encoder}.h5"


def h5_keys(path: Path) -> set[str]:
    """sha256 keys already in a table."""
    import h5py

    if not path.exists():
        return set()
    with h5py.File(path, "r") as f:
        return {k.decode() for k in f["sha256"][:]}


def h5_append(
    path: Path, keys: list[str], emb: np.ndarray, dim: int, source: str
) -> None:
    """Append rows (fp16) to a table, creating it on first write."""
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "a") as f:
        if "sha256" not in f:
            f.create_dataset(
                "sha256", shape=(0,), maxshape=(None,), dtype="S64", chunks=(4096,)
            )
            f.create_dataset(
                "embedding",
                shape=(0, dim),
                maxshape=(None, dim),
                dtype="float16",
                chunks=(1024, dim),
            )
            f.create_dataset(
                "source", shape=(0,), maxshape=(None,), dtype="S16", chunks=(4096,)
            )
            f.attrs["dim"] = dim
        n = f["sha256"].shape[0]
        m = len(keys)
        f["sha256"].resize((n + m,))
        f["embedding"].resize((n + m, dim))
        f["source"].resize((n + m,))
        f["sha256"][n:] = np.array([k.encode() for k in keys], dtype="S64")
        f["embedding"][n:] = emb.astype(np.float16)
        f["source"][n:] = np.array([source.encode()] * m, dtype="S16")


def load_s288c_table(
    data_root: str, spec: EncoderSpec
) -> tuple[list[str], list[str], Any]:
    """``(gene ids, stored input strings, embeddings [n, dim] float32)``."""
    import torch

    data, _ = torch.load(
        Path(data_root) / spec.s288c_table, map_location="cpu", weights_only=False
    )
    raw = data.dna_windows[spec.s288c_key]
    seqs = [s if isinstance(s, str) else s.seq for s in raw]
    emb = data.embeddings[spec.s288c_key].float()
    return list(data.id), seqs, emb


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def convention_strings(cfg: IsolateCompositeConfig, kind: str) -> dict[str, str]:
    """``{gene: the S288C input under this script's convention}`` for one input kind."""
    ctx = s288c_reference_context(cfg)
    return dict(zip(ctx["feature_id"], ctx[kind], strict=True))


def stage_seed(cfg: IsolateCompositeConfig, data_root: str) -> None:
    """Copy each S288C table into its allele table, keyed by the string it EMBEDDED.

    A row is keyed by the sha256 of its stored input string when that string equals
    this script's convention string for the gene. Where the two differ (CaLM's intron
    genes: the stored string is the unspliced span, the embedding is not always of it)
    both strings are re-embedded on the CPU and the row is keyed by the one that
    reproduces the stored vector (cosine >= min_reference_cosine); a row neither
    reproduces is not seeded and is computed fresh. Every decision is written to
    ``index/s288c_seed_<encoder>.parquet``.
    """
    t0, started = time.time(), now()
    counts: dict[str, int | float | str] = {}
    for name in cfg.encoders:
        spec = ENCODERS[name]
        path = h5_path(cfg, name)
        have = h5_keys(path)
        ids, seqs, emb = load_s288c_table(data_root, spec)
        conv = convention_strings(cfg, spec.input_kind)
        decisions = []
        embedder: Embedder | None = None
        keys, rows, seen = [], [], set(have)
        for i, (gene, stored) in enumerate(zip(ids, seqs, strict=True)):
            c = conv[gene]
            if stored == c:
                key, source, cos_c, cos_s = (
                    sha256_text(stored),
                    "stored_is_convention",
                    None,
                    None,
                )
            else:
                if embedder is None:
                    embedder = Embedder(spec, "cpu")
                ref = emb[i].numpy()
                cos_c = cosine(embedder([c])[0], ref)
                cos_s = cosine(embedder([stored])[0], ref)
                if cos_c >= cfg.min_reference_cosine:
                    key, source = sha256_text(c), "embedded_convention_string"
                elif cos_s >= cfg.min_reference_cosine:
                    key, source = sha256_text(stored), "embedded_stored_string"
                else:
                    key, source = None, "unresolved_not_seeded"
            decisions.append(
                {
                    "gene": gene,
                    "sha256": key,
                    "source": source,
                    "cosine_convention": cos_c,
                    "cosine_stored": cos_s,
                }
            )
            if key is not None and key not in seen:
                seen.add(key)
                keys.append(key)
                rows.append(i)
        if keys:
            h5_append(path, keys, emb[rows].numpy(), spec.dim, "s288c_table")
        dec = pd.DataFrame(decisions)
        write_parquet(dec, cfg.out_root / "index" / f"s288c_seed_{name}.parquet")
        counts[f"{name}_seeded"] = len(keys)
        counts[f"{name}_s288c_rows"] = len(ids)
        for src, n in dec["source"].value_counts().items():
            counts[f"{name}_{src}"] = int(n)
        del embedder
    record_stage(cfg, "seed", t0, started, counts=counts)


# --------------------------------------------------------------------------- #
# Stage 3b: embed (GPU)
# --------------------------------------------------------------------------- #
class Embedder:
    """Batched forward passes that equal the S288C tables' one-at-a-time passes.

    Batches never pad: every batch holds inputs of one token length, so a mean over
    tokens is the same mean the per-sequence call takes.
    """

    def __init__(self, spec: EncoderSpec, device: str) -> None:
        """Load the encoder onto ``device`` in the S288C table's precision."""
        import torch

        self.spec, self.device, self.torch = spec, device, torch
        if spec.name.startswith("fudt"):
            from torchcell.models.fungal_up_down_transformer import (
                FungalUpDownTransformer,
            )

            hf = (
                "upstream_species_lm"
                if spec.name.endswith("upstream")
                else "downstream_species_lm"
            )
            self.fudt = FungalUpDownTransformer(hf)
            self.fudt.model.to(device).float().eval()
        elif spec.name == "calm":
            from calm import CaLM

            self.calm = CaLM()
            self.calm.model.to(device).float().eval()
        else:
            from torchcell.models.protT5 import ProtT5

            self.pt5 = ProtT5("prot_t5_xl_uniref50")
            self.pt5.model.to(device)
            # the S288C table ran fp16 on a card; a CPU check runs fp32
            if device.startswith("cuda"):
                self.pt5.model.half().eval()
            else:
                self.pt5.model.float().eval()

    def __call__(self, seqs: list[str]) -> np.ndarray:
        """Mean embeddings ``[len(seqs), dim]`` float32 (inputs of one token length).

        An input outside the encoder's alphabet never reaches the model: it raises here
        (``stage_embed`` filters and counts them first).
        """
        bad = [s for s in seqs if untokenizable_codes(self.spec.input_kind, s)]
        if bad:
            raise ValueError(
                f"{self.spec.name}: {len(bad)} untokenizable inputs reached the model"
            )
        torch = self.torch
        with torch.no_grad():
            if self.spec.name.startswith("fudt"):
                tok = self.fudt.tokenizer(
                    [
                        "candida_glabrata "
                        + " ".join(s[i : i + 6] for i in range(len(s) - 5))
                        for s in seqs
                    ],
                    return_tensors="pt",
                )
                n_tok = tok["input_ids"].shape[1]
                if self.spec.name.endswith("upstream") and n_tok < 1001:
                    raise ValueError(
                        "upstream window shorter than 1003 bp reached the batch path"
                    )
                tok = {k: v.to(self.device) for k, v in tok.items()}
                hs = self.fudt.model(**tok, output_hidden_states=True).hidden_states
                emb = torch.stack(hs[8:]).mean(0).mean(1)
            elif self.spec.name == "calm":
                from calm.sequence import CodonSequence

                _, _, tokens = self.calm.bc([("", CodonSequence(s).seq) for s in seqs])
                if (tokens == self.calm.alphabet.padding_idx).any():
                    raise ValueError("CaLM batch is padded")
                rep = self.calm.model(tokens.to(self.device), repr_layers=[12])
                emb = rep["representations"][12].mean(1)
            else:
                prepared = [self.pt5._prepare_sequence(s) for s in seqs]
                ids = self.pt5.tokenizer(
                    prepared,
                    add_special_tokens=True,
                    padding="longest",
                    return_tensors="pt",
                )
                if not bool(ids["attention_mask"].all()):
                    raise ValueError("ProtT5 batch is padded")
                out = self.pt5.model(
                    input_ids=ids["input_ids"].to(self.device),
                    attention_mask=ids["attention_mask"].to(self.device),
                )
                emb = out.last_hidden_state.mean(1)
        return emb.float().cpu().numpy()

    def token_length(self, s: str) -> int:
        """Token count the batch path will see for ``s``."""
        if self.spec.name.startswith("fudt"):
            return len(s) - 5 + 3
        if self.spec.name == "calm":
            return (len(s) + 2) // 3 + 2
        # one token per residue ('*' and 'J' are one <unk> each) plus </s>; measured on
        # the ProtT5 tokenizer, and every batch still asserts it carries no padding
        return len(s) + 1


def batches(
    cfg: IsolateCompositeConfig, emb: Embedder, items: list[tuple[str, str]]
) -> list[list[tuple[str, str]]]:
    """Group by exact token length, then cut by the token and count budgets."""
    kind = emb.spec.input_kind
    groups: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for sha, seq in items:
        groups[emb.token_length(seq)].append((sha, seq))
    if kind == "protein":
        max_tok, max_n = cfg.max_tokens_protein, cfg.max_batch_protein
    elif kind == "calm":
        max_tok, max_n = cfg.max_tokens_calm, cfg.max_batch_calm
    else:
        max_tok, max_n = 10**9, cfg.max_batch_window
    out: list[list[tuple[str, str]]] = []
    for n_tok in sorted(groups):
        size = max(1, min(max_n, max_tok // max(n_tok, 1)))
        g = groups[n_tok]
        out += [g[i : i + size] for i in range(0, len(g), size)]
    return out


def reference_check(
    cfg: IsolateCompositeConfig, data_root: str, emb: Embedder
) -> dict[str, float]:
    """Embed S288C inputs through the batch path and compare with the stored rows."""
    spec = emb.spec
    ids, seqs, stored = load_s288c_table(data_root, spec)
    conv = convention_strings(cfg, spec.input_kind)
    full = [
        i
        for i, s in enumerate(seqs)
        if s == conv[ids[i]]
        and not untokenizable_codes(spec.input_kind, s)
        and not (spec.input_kind == "window_up" and len(s) != cfg.geometry.window_up_bp)
        and not (
            spec.input_kind == "window_down" and len(s) != cfg.geometry.window_down_bp
        )
        and not (spec.input_kind == "calm" and len(s) % 3)
    ]
    rng = np.random.default_rng(0)
    pick = sorted(
        rng.choice(full, size=min(cfg.n_reference_check, len(full)), replace=False)
    )
    items = [(str(i), seqs[i]) for i in pick]
    got: dict[str, np.ndarray] = {}
    for b in batches(cfg, emb, items):
        vecs = emb([s for _, s in b])
        for (k, _), v in zip(b, vecs, strict=True):
            got[k] = v
    ref = stored.numpy()
    cos = []
    maxabs = []
    for k, v in got.items():
        r = ref[int(k)]
        cos.append(float(np.dot(v, r) / (np.linalg.norm(v) * np.linalg.norm(r))))
        maxabs.append(float(np.abs(v - r).max()))
    res = {
        "n": len(cos),
        "min_cosine": min(cos),
        "mean_cosine": float(np.mean(cos)),
        "max_abs_diff": max(maxabs),
    }
    print(f"reference check {spec.name}: {res}", flush=True)
    if res["min_cosine"] < cfg.min_reference_cosine:
        raise RuntimeError(
            f"{spec.name}: batch path does not reproduce the S288C table "
            f"(min cosine {res['min_cosine']:.5f} < {cfg.min_reference_cosine})"
        )
    return res


def stage_embed(
    cfg: IsolateCompositeConfig, data_root: str, encoders: list[str]
) -> None:
    """Per encoder: the distinct inputs not yet in its table."""
    import torch

    for name in encoders:
        t0, started = time.time(), now()
        spec = ENCODERS[name]
        path = h5_path(cfg, name)
        have = h5_keys(path)
        table = pd.read_parquet(cfg.out_root / SEQUENCE_TABLE[spec.input_kind])
        pending = [
            (k, s)
            for k, s in zip(table["sha256"], table["sequence"], strict=True)
            if k not in have
        ]
        # an input the encoder cannot tokenize is a typed gap: skipped and counted
        # here, before any batch is formed (the alphabet stage records it in the index)
        skipped: Counter[str] = Counter()
        todo = []
        for k, s in pending:
            codes = untokenizable_codes(spec.input_kind, s)
            if codes:
                skipped.update(codes)
                skipped["__inputs__"] += 1
            else:
                todo.append((k, s))
        print(
            f"embed {name}: {len(table)} distinct, {len(pending)} without a row, "
            f"{skipped['__inputs__']} untokenizable skipped {dict(skipped)}, "
            f"{len(todo)} to compute",
            flush=True,
        )
        emb = Embedder(spec, cfg.device)
        check = reference_check(cfg, data_root, emb)
        plan = batches(cfg, emb, todo)
        buf_k: list[str] = []
        buf_v: list[np.ndarray] = []
        n_done = 0
        for bi, b in enumerate(plan, start=1):
            vecs = emb([s for _, s in b])
            buf_k += [k for k, _ in b]
            buf_v.append(vecs)
            if bi % cfg.flush_every_batches == 0 or bi == len(plan):
                h5_append(path, buf_k, np.concatenate(buf_v), spec.dim, "computed")
                n_done += len(buf_k)
                buf_k, buf_v = [], []
                el = time.time() - t0
                print(
                    f"embed {name}: {n_done}/{len(todo)} in {el:.0f} s "
                    f"({n_done / max(el, 1e-9):.1f} per s)",
                    flush=True,
                )
        del emb
        torch.cuda.empty_cache()
        record_stage(
            cfg,
            f"embed_{name}",
            t0,
            started,
            counts={
                "distinct": len(table),
                "computed": len(todo),
                "untokenizable_skipped": skipped["__inputs__"],
                "batches": len(plan),
                **{f"reference_{k}": v for k, v in check.items()},
            },
        )


# --------------------------------------------------------------------------- #
# Stage 4: verify
# --------------------------------------------------------------------------- #
def stage_verify(cfg: IsolateCompositeConfig) -> None:
    """Every (isolate, feature) resolves to a row in each table or carries a gap."""
    t0, started = time.time(), now()
    out = cfg.out_root
    index = pd.read_parquet(out / "index/isolate_gene.parquet")
    ref = pd.read_parquet(out / "index/s288c_reference.parquet")
    report: dict[str, Any] = {
        "generated_by": f"experiments/{EXPERIMENT}/scripts/compute_isolate_composite.py",
        "store": str(out),
        "created_at": now(),
        "rows": len(index),
        "isolates": int(index["isolate"].nunique()),
    }
    failures: list[str] = []
    if set(index["isolate"]) != set(cfg.isolates):
        failures.append("isolate set differs from the config")
    n_ref = index[index["feature_kind"] == "reference_gene"].groupby("isolate").size()
    if n_ref.nunique() != 1:
        failures.append(
            f"reference-gene rows per isolate vary: {n_ref.describe().to_dict()}"
        )
    report["reference_genes_per_isolate"] = int(n_ref.iloc[0])
    if "calm_gap" not in index.columns:
        raise ValueError("run --stage alphabet before verify (per-encoder gap columns)")
    gap_col = GAP_COLUMN
    tables: dict[str, Any] = {}
    for name in cfg.encoders:
        spec = ENCODERS[name]
        col = INPUT_COLUMN[spec.input_kind]
        seq_keys = set(
            pd.read_parquet(out / SEQUENCE_TABLE[spec.input_kind], columns=["sha256"])[
                "sha256"
            ]
        )
        emb_keys = h5_keys(h5_path(cfg, name))
        has = index[col].notna()
        gapped = index[gap_col[spec.input_kind]].notna()
        neither = int((~has & ~gapped).sum())
        both = int((has & gapped).sum())
        seq_ok = index.loc[has, col].isin(seq_keys)
        emb_ok = index.loc[has, col].isin(emb_keys)
        ref_emb_ok = ref[col].isin(emb_keys)
        tables[name] = {
            "rows_with_input": int(has.sum()),
            "rows_with_typed_gap": int(gapped.sum()),
            "rows_with_neither": neither,
            "rows_with_both": both,
            "inputs_in_sequence_table": int(seq_ok.sum()),
            "inputs_with_embedding_row": int(emb_ok.sum()),
            "distinct_inputs": int(index.loc[has, col].nunique()),
            "distinct_inputs_with_embedding_row": int(
                index.loc[has, col][emb_ok].nunique()
            ),
            "embedding_table_rows": len(emb_keys),
            "s288c_genes_with_embedding_row": int(ref_emb_ok.sum()),
            "gap_types": index.loc[gapped, gap_col[spec.input_kind]]
            .value_counts()
            .to_dict(),
        }
        if neither or both:
            failures.append(
                f"{name}: {neither} rows with neither input nor gap, {both} with both"
            )
        if int(seq_ok.sum()) != int(has.sum()):
            failures.append(f"{name}: inputs missing from the sequence table")
    report["tables"] = tables
    report["embedding_complete"] = all(
        t["inputs_with_embedding_row"] == t["rows_with_input"] for t in tables.values()
    )

    # geometry check: an isolate carrying the exact S288C allele, located with full
    # flanks, should usually reproduce the S288C window (the S288C table's own string)
    refgenes = index[
        (index["feature_kind"] == "reference_gene") & index["is_reference_allele"]
    ]
    refmap = ref.set_index("feature_id")
    geom: dict[str, Any] = {}
    for kind in ("window_up", "window_down"):
        col = INPUT_COLUMN[kind]
        sub = refgenes[
            refgenes[col].notna() & refgenes["feature_id"].isin(refmap.index)
        ]
        same = sub[col].to_numpy() == refmap.loc[sub["feature_id"], col].to_numpy()
        agree = sub["sgd_model_agrees"].to_numpy()
        geom[kind] = {
            "rows_reference_allele_with_window": len(sub),
            "identical_to_s288c_window": int(same.sum()),
            "fraction_identical": round(float(same.mean()), 4) if len(sub) else None,
            "fraction_identical_sgd_model_agrees": round(float(same[agree].mean()), 4)
            if agree.any()
            else None,
        }
    report["geometry_check_reference_alleles"] = geom

    loc = index["locus_gap"].fillna("located")
    per_iso = (
        index.assign(loc=loc)
        .groupby("isolate")
        .agg(
            reference_genes=(
                "feature_kind",
                lambda s: int((s == "reference_gene").sum()),
            ),
            accessory_orfs=(
                "feature_kind",
                lambda s: int((s == "accessory_orf").sum()),
            ),
            located=("loc", lambda s: int((s == "located").sum())),
            up_windows=("up_sha256", lambda s: int(s.notna().sum())),
            down_windows=("down_sha256", lambda s: int(s.notna().sum())),
            present=("state", lambda s: int((s == "present").sum())),
        )
    )
    report["locus_outcomes"] = loc.value_counts().to_dict()
    report["locus_outcomes_reference_present"] = (
        loc[(index["feature_kind"] == "reference_gene") & (index["state"] == "present")]
        .value_counts()
        .to_dict()
    )
    report["per_isolate_summary"] = {
        c: {k: round(float(v), 2) for k, v in per_iso[c].describe().items()}
        for c in per_iso.columns
    }
    report["isolates_lowest_located"] = (
        per_iso.sort_values("located").head(10)["located"].to_dict()
    )
    report["failures"] = failures
    report["passed"] = not failures
    write_parquet(per_iso.reset_index(), out / "index/per_isolate_counts.parquet")
    res_path = (
        Path(experiment_results_dir(EXPERIMENT, __file__))
        / "compute_isolate_composite_verify.json"
    )
    write_json(report, res_path)
    write_json(report, out / "verify_report.json")
    record_stage(
        cfg,
        "verify",
        t0,
        started,
        counts={
            "passed": str(report["passed"]),
            "embedding_complete": str(report["embedding_complete"]),
            "failures": len(failures),
        },
    )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "per_isolate_summary"},
            indent=1,
            default=str,
        )
    )


# --------------------------------------------------------------------------- #
def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--stage",
        required=True,
        choices=[
            "extract",
            "unpack",
            "align",
            "merge",
            "seed",
            "embed",
            "verify",
            "config",
            "check",
            "alphabet",
        ],
    )
    parser.add_argument("--chunk", type=int, default=None)
    parser.add_argument("--n-chunks", type=int, default=None)
    parser.add_argument(
        "--isolate", default=None, help="align one isolate (timing run)"
    )
    parser.add_argument("--encoders", default=None, help="comma list for --stage embed")
    parser.add_argument(
        "--out-root", default=None, help="smoke-run store (not the real one)"
    )
    parser.add_argument(
        "--limit-gene-files", type=int, default=None, help="smoke runs only"
    )
    parser.add_argument("--device", default=None, help="--stage check: device")
    parser.add_argument("--n-check", type=int, default=None, help="--stage check: n")
    args = parser.parse_args()
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    cfg = build_config(data_root, Path(args.out_root) if args.out_root else None)
    if args.stage == "config":
        print(cfg.model_dump_json(indent=1))
    elif args.stage == "extract":
        stage_extract(cfg, data_root, args.limit_gene_files)
    elif args.stage == "unpack":
        stage_unpack(cfg, data_root)
    elif args.stage == "align":
        if args.isolate is not None:
            members = json.loads(
                (cfg.out_root / "work/assemblies/members.json").read_text()
            )
            print(
                json.dumps(
                    align_isolate(cfg, args.isolate, members, load_manifest(cfg).tools)
                )
            )
        else:
            if args.chunk is None:
                args.chunk = int(os.environ["SLURM_ARRAY_TASK_ID"])
            if args.n_chunks is None:
                args.n_chunks = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
            stage_align(cfg, args.chunk, args.n_chunks)
    elif args.stage == "merge":
        stage_merge(cfg)
    elif args.stage == "seed":
        stage_seed(cfg, data_root)
    elif args.stage == "embed":
        encoders = args.encoders.split(",") if args.encoders else list(cfg.encoders)
        stage_embed(cfg, data_root, encoders)
    elif args.stage == "verify":
        stage_verify(cfg)
    elif args.stage == "alphabet":
        stage_alphabet(cfg)
    elif args.stage == "check":
        # the batch path against the S288C tables, on any device, writing nothing
        encoders = args.encoders.split(",") if args.encoders else list(cfg.encoders)
        ccfg = cfg.model_copy(
            update={"n_reference_check": args.n_check or cfg.n_reference_check}
        )
        res = {}
        for name in encoders:
            t0 = time.time()
            emb = Embedder(ENCODERS[name], args.device or cfg.device)
            res[name] = reference_check(ccfg, data_root, emb) | {
                "seconds": round(time.time() - t0, 1)
            }
            del emb
        print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
