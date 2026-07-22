# experiments/embeddings/compute_isolate_embeddings.py
# [[experiments.embeddings.compute_isolate_embeddings]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/embeddings/compute_isolate_embeddings
"""Per-isolate sequence embeddings for the Caudal 2024 natural isolates (WS9 / Fig 4).

Each of the 943 Caudal natural isolates has its OWN copy of every S288C reference gene,
with SNPs + indels already inferred, in the Peter et al. 2018 gene-keyed store
(``allReferenceGenesWithSNPsAndIndelsInferred.tar.gz``). This runner reads an isolate's
per-gene coding sequences DIRECTLY from that store, translates them, and produces:

  * ESM2 mean-pooled protein embeddings (one vector per gene), reusing ``torchcell.models``
    ``Esm2`` -- the SAME model/pooling as ``torchcell.datasets.esm2.Esm2Dataset``. FULLY
    IMPLEMENTED: the isolate CDS is directly available, translate -> ESM2.

  * FUDT (FungalUpDownTransformer) species-aware 5'/3' window embeddings (WS9b). The
    gene-keyed store carries ONLY the CDS body (no promoter/terminator flank), so the flank
    is recovered from the isolate's OWN de-novo assembly (``1011Assemblies.tar.gz``): BLASTN
    each isolate CDS back to its assembly to locate the gene (contig + coords + strand), then
    slice the 5' (1000 bp upstream + start codon) and 3' (stop codon + 297 bp downstream)
    windows from the contig and embed them with the SAME FungalUpDownTransformer /
    window geometry as ``FungalUpDownTransformerDataset``. Requires BLAST+ (``--with-fudt``).

STRAIN-ID CONVENTION (load-bearing, ties to issue #73): the canonical isolate id is the one
used by the ``CaudalPanTranscriptome2024Dataset`` build -- the Caudal ``Strain`` code, which
for 78 isolates RETAINS a ``SACE_`` prefix (e.g. ``SACE_YAV``). The Peter presence matrix AND
the FASTA headers use that SAME ``SACE_`` form, so header->strain matching drops NOTHING for
all 943 built isolates PROVIDED the prefix is kept verbatim. This runner therefore matches by
the full canonical id and NEVER strips ``SACE_`` -- stripping it is exactly the #73 defect
(headers keep the prefix; a bare-code key would silently miss those 78 isolates' sequences).

Storage (embedding-dataset convention, one dir per isolate):
  ``$DATA_ROOT/data/scerevisiae/caudal2024_isolate_embeddings/<isolate>/esm2_<model>.pt``
each a ``(data, slices)`` collated tuple (gene id -> embedding), loadable exactly like an
``Esm2Dataset`` processed file.
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import re
import shutil
import subprocess
import tarfile
import tempfile
from typing import TYPE_CHECKING

import torch
from Bio.Seq import Seq
from dotenv import load_dotenv
from pydantic import BaseModel
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

if TYPE_CHECKING:
    from torchcell.models.esm2 import Esm2
    from torchcell.models.fungal_up_down_transformer import FungalUpDownTransformer

# Peter 2018 gene-keyed per-isolate CDS store (mirror-relative, sha256-pinned by the loader).
REFGENE_TAR_REL = (
    "torchcell-library/peterGenomeEvolution10112018/data/"
    "allReferenceGenesWithSNPsAndIndelsInferred.tar.gz"
)
# Peter 2018 per-isolate de-novo assembly store (contig-level FASTA, no per-strain GFF).
ASSEMBLY_TAR_REL = (
    "torchcell-library/peterGenomeEvolution10112018/data/1011Assemblies.tar.gz"
)
DEFAULT_ESM2_MODEL = "esm2_t33_650M_UR50D"
ISOLATE_EMBED_REL = "data/scerevisiae/caudal2024_isolate_embeddings"

# FUDT (SpeciesLM) window geometry -- MUST match torchcell.sequence window_five_prime /
# window_three_prime and FungalUpDownTransformerDataset.MODEL_TO_WINDOW exactly:
#   5' upstream window = 1000 bp upstream of the CDS start + the start codon (1003 bp total)
#   3' downstream window = the stop codon + 297 bp downstream of the CDS end (300 bp total)
FUDT_FIVE_PRIME_UPSTREAM = 1000
FUDT_THREE_PRIME_DOWNSTREAM = 297
# SpeciesLM downstream model requires > 11 bp; below this a per-isolate 3' window is
# un-embeddable and we fall back (flagged) to the S288C reference window.
FUDT_MIN_DOWNSTREAM_BP = 12
FUDT_MODEL_NAMES = ("species_upstream", "species_downstream")


def _data_root() -> str:
    """Return ``$DATA_ROOT`` (parent of the library mirror)."""
    load_dotenv()
    return os.environ["DATA_ROOT"]


def _parse_fasta(text: str) -> list[tuple[str, str]]:
    """Parse FASTA text into ``[(header_without_gt, sequence), ...]``."""
    records: list[tuple[str, str]] = []
    header: str | None = None
    buf: list[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(buf)))
            header = line[1:]
            buf = []
        else:
            buf.append(line.strip())
    if header is not None:
        records.append((header, "".join(buf)))
    return records


def extract_isolate_orfs(
    tar_path: str, isolate_id: str, limit: int | None = None
) -> dict[str, str]:
    """Return ``{systematic_gene_name: cds_dna}`` for one isolate from the gene-keyed store.

    The header token is ``<ISOLATE>_<SYS>_<SYMBOL>``; the isolate id is the prefix before
    ``_<SYS>_``. Matching is EXACT against ``isolate_id`` (``SACE_`` prefix kept -- see #73).
    ``limit`` caps the number of gene files scanned (for a light dry/smoke path).
    """
    orfs: dict[str, str] = {}
    n_files = 0
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            sys_name = member.name.replace(".fasta", "")
            extracted = tf.extractfile(member)
            if extracted is None:
                continue
            records = _parse_fasta(extracted.read().decode("latin-1"))
            split_key = f"_{sys_name}_"
            for header, seq in records:
                token = header.split()[0].split("\t")[0]
                if split_key not in token:
                    continue
                iso = token.split(split_key, 1)[0]
                if iso == isolate_id:
                    orfs[sys_name] = seq.upper()
                    break
            n_files += 1
            if limit is not None and n_files >= limit:
                break
    return orfs


def extract_multi_isolate_orfs(
    tar_path: str, isolate_ids: list[str], limit: int | None = None
) -> dict[str, dict[str, str]]:
    """Return ``{isolate_id: {gene: cds}}`` for several isolates in ONE tar scan.

    The gene-keyed store is a single large tarball; scanning it once and pulling every
    requested isolate's record per gene avoids re-decompressing it per isolate (each scan
    is ~1 min). ``limit`` caps the number of gene files scanned (light PoC path).
    """
    wanted = set(isolate_ids)
    out: dict[str, dict[str, str]] = {iso: {} for iso in isolate_ids}
    n_files = 0
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            sys_name = member.name.replace(".fasta", "")
            extracted = tf.extractfile(member)
            if extracted is None:
                continue
            records = _parse_fasta(extracted.read().decode("latin-1"))
            split_key = f"_{sys_name}_"
            for header, seq in records:
                token = header.split()[0].split("\t")[0]
                if split_key not in token:
                    continue
                iso = token.split(split_key, 1)[0]
                if iso in wanted:
                    out[iso][sys_name] = seq.upper()
            n_files += 1
            if limit is not None and n_files >= limit:
                break
    return out


def translate_orfs(orfs: dict[str, str]) -> dict[str, str]:
    """Translate each isolate CDS to a protein string (stop codons stripped from the end).

    Isolate CDS lengths vary by indels; sequences not divisible by 3 are truncated to the
    last whole codon before translation (Biopython's default). Internal stop codons (rare,
    from indel frameshifts) are kept as ``*`` so ESM2 sees the honest translated product.
    """
    proteins: dict[str, str] = {}
    for sys_name, dna in orfs.items():
        usable = dna[: len(dna) - (len(dna) % 3)]
        protein = str(Seq(usable).translate())  # type: ignore[no-untyped-call]  # Bio.Seq.translate is untyped
        proteins[sys_name] = protein.rstrip("*")
    return proteins


def _load_esm2(model_name: str) -> Esm2:
    """Instantiate the ESM2 model (heavy; imported lazily so ``--dry-run`` stays light)."""
    from torchcell.models.esm2 import Esm2

    return Esm2(model_name=model_name)


def compute_isolate_esm2(
    isolate_id: str,
    proteins: dict[str, str],
    esm2_model_name: str,
    output_root: str,
) -> str:
    """Embed each isolate protein with ESM2 (mean-pooled) and save the collated dataset.

    Output mirrors ``Esm2Dataset``: a ``(data, slices)`` tuple of PyG ``Data`` objects each
    carrying ``id=<gene>`` and ``embeddings={<key>: tensor}``.
    """
    model = _load_esm2(esm2_model_name)
    key = f"esm2_{esm2_model_name}"
    data_list: list[Data] = []
    for gene_id, protein in tqdm(proteins.items(), desc=f"esm2 {isolate_id}"):
        emb = model.embed([protein], mean_embedding=True).cpu().squeeze()
        data = Data(id=gene_id)
        data.embeddings = {key: emb}
        data_list.append(data)
    out_dir = osp.join(output_root, isolate_id)
    os.makedirs(out_dir, exist_ok=True)
    out_path = osp.join(out_dir, f"esm2_{esm2_model_name}.pt")
    torch.save(InMemoryDataset.collate(data_list), out_path)
    return out_path


# ---------------------------------------------------------------------------------------
# Per-isolate FUDT (WS9b): recover 5'/3' flanking windows by aligning each isolate's own
# CDS back to its own de-novo assembly, then slicing the flank from the assembly contig.
# ---------------------------------------------------------------------------------------


class BlastHit(BaseModel):
    """Best CDS->assembly BLASTN hit locating a gene on one isolate contig (1-based)."""

    gene: str
    contig: str
    start: int  # 1-based inclusive, low coordinate on the contig plus strand
    end: int  # 1-based inclusive, high coordinate on the contig plus strand
    strand: str  # "plus" | "minus" -- gene orientation relative to the contig
    pident: float  # percent identity of the aligned segment
    aln_len: int  # aligned length (bp)
    qlen: int  # query CDS length (bp)
    bitscore: float
    second_bitscore: float | None = None  # runner-up hit bitscore (unique-hit margin)

    @property
    def coverage(self) -> float:
        """Fraction of the CDS covered by the aligned segment."""
        return self.aln_len / self.qlen if self.qlen else 0.0


class FudtWindowSlice(BaseModel):
    """5'/3' FUDT windows sliced from an isolate assembly around a located gene."""

    gene: str
    five_prime: str  # 1000 bp upstream + start codon (<=1003 bp; may be truncated)
    three_prime: str  # stop codon + 297 bp downstream (<=300 bp; may be truncated)
    five_prime_full: bool  # got the full 1000 bp upstream flank
    three_prime_full: bool  # got the full 297 bp downstream flank
    source: str  # "assembly" | "reference_fallback"


def _revcomp(seq: str) -> str:
    """Reverse complement (uppercase ACGTN; other symbols pass through)."""
    return str(Seq(seq).reverse_complement())  # type: ignore[no-untyped-call]


def resolve_blast_tools(
    blastn: str | None = None, makeblastdb: str | None = None
) -> tuple[str, str]:
    """Resolve ``blastn`` / ``makeblastdb`` executables, or raise with a clear message.

    Resolution order per tool: explicit argument -> ``$TC_BLASTN`` / ``$TC_MAKEBLASTDB``
    env var -> ``PATH`` (``shutil.which``). BLAST+ is not on the base ``torchcell`` PATH in
    this env; set the env vars (or pass paths) to a BLAST+ 2.1x install.
    """
    resolved: list[str] = []
    for name, explicit, env in (
        ("blastn", blastn, "TC_BLASTN"),
        ("makeblastdb", makeblastdb, "TC_MAKEBLASTDB"),
    ):
        path = explicit or os.environ.get(env) or shutil.which(name)
        if not path or not osp.exists(path):
            raise RuntimeError(
                f"{name} not found. Per-isolate FUDT needs BLAST+; pass an explicit path, "
                f"set ${env}, or put {name} on PATH (e.g. conda install -c bioconda blast)."
            )
        resolved.append(path)
    return resolved[0], resolved[1]


def build_assembly_member_index(tar_path: str) -> dict[str, str]:
    """Map each isolate 3-letter key -> its assembly member name in the assembly tarball.

    Assembly members are ``GENOMES_ASSEMBLED/<KEY>[_<N>].re.fa``; the CDS-store isolate key
    is the leading 3-letter code (e.g. ``AAA`` -> ``GENOMES_ASSEMBLED/AAA_6.re.fa``). Built
    once (a full tar listing) and cached next to the tar as ``<tar>.member_index.tsv``.
    """
    cache = tar_path + ".member_index.tsv"
    if osp.exists(cache):
        index: dict[str, str] = {}
        with open(cache) as fh:
            for line in fh:
                key, member = line.rstrip("\n").split("\t")
                index[key] = member
        return index
    index = {}
    pat = re.compile(r"([A-Za-z0-9]+?)(?:_\d+)?\.re\.fa$")
    with tarfile.open(tar_path, "r:gz") as tf:
        for ti in tf:
            base = osp.basename(ti.name)
            m = pat.match(base)
            if m:
                index[m.group(1)] = ti.name
    with open(cache, "w") as fh:
        for key, member in sorted(index.items()):
            fh.write(f"{key}\t{member}\n")
    return index


def _assembly_key(isolate_id: str) -> str:
    """Isolate id -> 3-letter assembly-lookup key (drops a ``SACE_`` prefix for lookup only)."""
    return isolate_id.split("_")[-1] if isolate_id.startswith("SACE_") else isolate_id


def extract_isolate_assemblies(
    tar_path: str, isolate_ids: list[str], cache_dir: str
) -> dict[str, str]:
    """Extract several isolates' assembly FASTAs in ONE forward pass; return ``{id: path}``.

    The assembly tarball is a ~4 GB gzip stream. Members are written by STREAMING (forward
    only, ``copyfileobj``): calling ``extractfile(name)`` for a member after the member list
    has been fully loaded triggers a backward seek in the gzip stream that silently yields 0
    bytes on the 2nd+ call, so we iterate members once and copy each wanted member as we reach
    it. Any ``SACE_`` prefix is stripped for the assembly-file lookup only (#73 keeps the full
    id for CDS matching). Already-extracted non-empty files are reused.
    """
    index = build_assembly_member_index(tar_path)
    key_to_id: dict[str, str] = {}
    want_members: dict[str, str] = {}
    os.makedirs(cache_dir, exist_ok=True)
    result: dict[str, str] = {}
    for iso in isolate_ids:
        key = _assembly_key(iso)
        if key not in index:
            raise RuntimeError(f"no assembly member for key {key!r} (id {iso!r})")
        member = index[key]
        out_path = osp.join(cache_dir, osp.basename(member))
        result[iso] = out_path
        if osp.exists(out_path) and osp.getsize(out_path) > 0:
            continue
        want_members[member] = out_path
        key_to_id[member] = iso
    if want_members:
        with tarfile.open(tar_path, "r:gz") as tf:
            for m in tf:
                if m.name in want_members:
                    src = tf.extractfile(m)
                    if src is None:
                        raise RuntimeError(f"could not read assembly member {m.name!r}")
                    with open(want_members[m.name], "wb") as fh:
                        shutil.copyfileobj(src, fh)
                    del want_members[m.name]
                    if not want_members:
                        break
    return result


def extract_isolate_assembly(tar_path: str, isolate_id: str, cache_dir: str) -> str:
    """Extract one isolate's assembly FASTA from the assembly tar into ``cache_dir``.

    Thin single-isolate wrapper over ``extract_isolate_assemblies`` (streaming, forward-only).
    """
    return extract_isolate_assemblies(tar_path, [isolate_id], cache_dir)[isolate_id]


def _read_fasta_seqs(path: str) -> dict[str, str]:
    """Read a FASTA into ``{contig_id: sequence}`` (id = first whitespace token)."""
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


def build_blast_db(fasta_path: str, db_dir: str, makeblastdb: str) -> str:
    """Build a nucleotide BLAST db from an assembly FASTA; return the db path prefix."""
    os.makedirs(db_dir, exist_ok=True)
    db_path = osp.join(db_dir, osp.basename(fasta_path))
    if osp.exists(db_path + ".nsq") or osp.exists(db_path + ".nin"):
        return db_path
    subprocess.run(
        [makeblastdb, "-in", fasta_path, "-dbtype", "nucl", "-out", db_path],
        check=True,
        capture_output=True,
    )
    return db_path


def locate_cds_in_assembly(
    orfs: dict[str, str], db_path: str, blastn: str
) -> dict[str, BlastHit]:
    """BLASTN every isolate CDS against the isolate's own assembly; return best hit per gene.

    Writes all CDS to one temp query FASTA and runs a single BLASTN, keeping the top hit by
    bitscore per gene and recording the runner-up bitscore (unique-hit margin). Genes with no
    hit are absent from the returned dict.
    """
    with tempfile.TemporaryDirectory() as tmp:
        query = osp.join(tmp, "query.fa")
        with open(query, "w") as fh:
            for gene, seq in orfs.items():
                fh.write(f">{gene}\n{seq}\n")
        proc = subprocess.run(
            [
                blastn,
                "-query", query,
                "-db", db_path,
                "-outfmt", "6 qseqid sseqid pident length qlen sstart send sstrand bitscore",
                "-max_target_seqs", "5",
                "-evalue", "1e-10",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    best: dict[str, BlastHit] = {}
    second: dict[str, float] = {}
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) != 9:
            continue
        gene, contig, pident, length, qlen, sstart, send, sstrand, bitscore = parts
        bs = float(bitscore)
        lo, hi = sorted((int(sstart), int(send)))
        hit = BlastHit(
            gene=gene,
            contig=contig,
            start=lo,
            end=hi,
            strand=sstrand,
            pident=float(pident),
            aln_len=int(length),
            qlen=int(qlen),
            bitscore=bs,
        )
        if gene not in best or bs > best[gene].bitscore:
            if gene in best:
                second[gene] = max(second.get(gene, 0.0), best[gene].bitscore)
            best[gene] = hit
        else:
            second[gene] = max(second.get(gene, 0.0), bs)
    for gene, hit in best.items():
        hit.second_bitscore = second.get(gene)
    return best


def slice_fudt_windows(
    contig_seq: str,
    hit: BlastHit,
    five_up: int = FUDT_FIVE_PRIME_UPSTREAM,
    three_down: int = FUDT_THREE_PRIME_DOWNSTREAM,
) -> FudtWindowSlice:
    """Slice the FUDT 5'/3' windows from a contig around a located gene, strand-aware.

    5' window = ``five_up`` bp upstream of the CDS start + the start codon; 3' window = the
    stop codon + ``three_down`` bp downstream of the CDS end. Windows are returned in the
    gene's 5'->3' orientation (minus-strand hits are reverse-complemented). Near a contig
    end the flank is truncated and the corresponding ``*_full`` flag is ``False``.
    """
    n = len(contig_seq)
    lo, hi = hit.start, hit.end  # 1-based inclusive
    cds_start0 = lo - 1  # 0-based first CDS base
    cds_end0 = hi  # 0-based exclusive end of CDS
    if hit.strand == "plus":
        f_s, f_e = cds_start0 - five_up, cds_start0 + 3
        five = contig_seq[max(0, f_s):f_e]
        five_full = f_s >= 0
        t_s, t_e = cds_end0 - 3, cds_end0 + three_down
        three = contig_seq[t_s:min(n, t_e)]
        three_full = t_e <= n
    else:  # minus strand: gene reads high->low on the contig
        f_s, f_e = cds_end0 - 3, cds_end0 + five_up
        five = _revcomp(contig_seq[f_s:min(n, f_e)])
        five_full = f_e <= n
        t_s, t_e = cds_start0 - three_down, cds_start0 + 3
        three = _revcomp(contig_seq[max(0, t_s):t_e])
        three_full = t_s >= 0
    return FudtWindowSlice(
        gene=hit.gene,
        five_prime=five,
        three_prime=three,
        five_prime_full=five_full,
        three_prime_full=three_full,
        source="assembly",
    )


def _load_fudt(model_name: str) -> FungalUpDownTransformer:
    """Instantiate the SpeciesLM up/down-stream transformer (heavy; imported lazily)."""
    from torchcell.models.fungal_up_down_transformer import FungalUpDownTransformer

    hf_name = (
        "upstream_species_lm"
        if model_name == "species_upstream"
        else "downstream_species_lm"
    )
    return FungalUpDownTransformer(model_name=hf_name)


def compute_isolate_fudt(
    isolate_id: str,
    orfs: dict[str, str],
    output_root: str,
    assembly_tar: str,
    assembly_cache_dir: str,
    blast_work_dir: str,
    blastn: str | None = None,
    makeblastdb: str | None = None,
    allow_reference_fallback: bool = False,
    reference_genome: object | None = None,
) -> dict[str, str]:
    """Per-isolate FUDT 5'/3' window embeddings via CDS->assembly alignment + flank slicing.

    Pipeline: extract the isolate's own de-novo assembly, BLASTN every isolate CDS against
    it to locate each gene (contig + coords + strand), slice the 5'/3' FUDT windows from the
    assembly, and embed them with ``FungalUpDownTransformer`` (mean-pooled) -- the SAME model
    and window geometry as ``FungalUpDownTransformerDataset``. Genes whose 3' window is too
    short to embed (or that fail to map) fall back to the S288C reference window ONLY when
    ``allow_reference_fallback`` is set (flagged ``source="reference_fallback"``); otherwise
    they are skipped and counted. Returns ``{fudt_model_name: output_path}``.
    """
    blastn_bin, makeblastdb_bin = resolve_blast_tools(blastn, makeblastdb)
    asm_path = extract_isolate_assembly(assembly_tar, isolate_id, assembly_cache_dir)
    db_path = build_blast_db(asm_path, blast_work_dir, makeblastdb_bin)
    contigs = _read_fasta_seqs(asm_path)
    hits = locate_cds_in_assembly(orfs, db_path, blastn_bin)

    slices: dict[str, FudtWindowSlice] = {}
    n_unmapped = 0
    for gene in orfs:
        hit = hits.get(gene)
        if hit is None:
            n_unmapped += 1
            continue
        slices[gene] = slice_fudt_windows(contigs[hit.contig], hit)

    out_paths: dict[str, str] = {}
    for model_name in FUDT_MODEL_NAMES:
        transformer = _load_fudt(model_name)
        window_attr = "five_prime" if model_name == "species_upstream" else "three_prime"
        data_list: list[Data] = []
        for gene, sl in tqdm(slices.items(), desc=f"fudt {model_name} {isolate_id}"):
            seq = getattr(sl, window_attr)
            source = sl.source
            if model_name == "species_downstream" and len(seq) < FUDT_MIN_DOWNSTREAM_BP:
                if not (allow_reference_fallback and reference_genome is not None):
                    continue
                seq = _reference_window(reference_genome, gene, model_name)
                source = "reference_fallback"
                if seq is None:
                    continue
            emb = transformer.embed([seq], mean_embedding=True).cpu().squeeze()
            data = Data(id=gene)
            data.embeddings = {model_name: emb}
            data.fudt_source = source
            data_list.append(data)
        out_dir = osp.join(output_root, isolate_id)
        os.makedirs(out_dir, exist_ok=True)
        out_path = osp.join(out_dir, f"fudt_{model_name}.pt")
        torch.save(InMemoryDataset.collate(data_list), out_path)
        out_paths[model_name] = out_path
    print(
        f"  fudt {isolate_id}: mapped {len(slices)}/{len(orfs)} genes "
        f"({n_unmapped} unmapped)"
    )
    return out_paths


def _reference_window(genome: object, gene: str, model_name: str) -> str | None:
    """Return the S288C reference FUDT window sequence for ``gene`` (fallback path)."""
    if gene not in genome.gene_set:  # type: ignore[attr-defined]
        return None
    seq_obj = genome[gene]  # type: ignore[index]
    if model_name == "species_upstream":
        dna = seq_obj.window_five_prime(1003, True, allow_undersize=True)
    else:
        dna = seq_obj.window_three_prime(300, True, allow_undersize=True)
    return str(dna.seq)


def run(
    isolate_id: str,
    esm2_model_name: str,
    output_root: str,
    n_genes: int | None,
    dry_run: bool,
    with_fudt: bool,
    skip_esm2: bool = False,
) -> None:
    """Extract an isolate's ORFs, translate, and compute embeddings (ESM2 + FUDT).

    ESM2 embeds the translated isolate protein; FUDT (when ``with_fudt``) recovers per-isolate
    5'/3' windows by aligning each CDS back to the isolate's own assembly (see
    ``compute_isolate_fudt``). ``skip_esm2`` computes FUDT only (avoids the heavy ESM2 load
    when validating the FUDT path).
    """
    tar_path = osp.join(_data_root(), REFGENE_TAR_REL)
    if not osp.exists(tar_path):
        raise RuntimeError(f"Peter gene-keyed store missing: {tar_path}")
    orfs = extract_isolate_orfs(tar_path, isolate_id, limit=n_genes)
    if not orfs:
        raise RuntimeError(
            f"no ORFs matched isolate {isolate_id!r} (check the id -- SACE_ prefix is kept)"
        )
    proteins = translate_orfs(orfs)
    print(
        f"isolate {isolate_id}: extracted {len(orfs)} ORFs; "
        f"translated {len(proteins)} proteins "
        f"(mean protein len {sum(len(p) for p in proteins.values()) / len(proteins):.0f})"
    )
    if dry_run:
        sample = next(iter(proteins.items()))
        print(f"  dry-run: sample {sample[0]} -> {len(sample[1])} aa: {sample[1][:30]}...")
        print("  dry-run: skipping model load / embedding.")
        return
    if not skip_esm2:
        esm2_path = compute_isolate_esm2(isolate_id, proteins, esm2_model_name, output_root)
        print(f"  wrote ESM2 embeddings -> {esm2_path}")
    if with_fudt:
        data_root = _data_root()
        out = compute_isolate_fudt(
            isolate_id=isolate_id,
            orfs=orfs,
            output_root=output_root,
            assembly_tar=osp.join(data_root, ASSEMBLY_TAR_REL),
            assembly_cache_dir=osp.join(output_root, "_assembly_cache"),
            blast_work_dir=osp.join(output_root, "_blast_db"),
        )
        for model_name, path in out.items():
            print(f"  wrote FUDT {model_name} embeddings -> {path}")


def main() -> None:
    """CLI entry point for per-isolate embedding computation."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-isolate ESM2 + FUDT (CDS->assembly aligned) sequence embeddings "
            "for a Caudal 2024 natural isolate."
        )
    )
    parser.add_argument(
        "--isolate",
        required=True,
        help="Canonical Caudal/build strain id (keep any SACE_ prefix, e.g. SACE_YAV).",
    )
    parser.add_argument(
        "--esm2-model",
        default=DEFAULT_ESM2_MODEL,
        help=f"ESM2 checkpoint name (default: {DEFAULT_ESM2_MODEL}).",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help=f"Output root (default: $DATA_ROOT/{ISOLATE_EMBED_REL}).",
    )
    parser.add_argument(
        "--n-genes",
        type=int,
        default=None,
        help="Cap gene files scanned (light smoke/dry path); default: all ~6015.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract + translate only; skip model load / embedding (no heavy compute).",
    )
    parser.add_argument(
        "--with-fudt",
        action="store_true",
        help="Compute per-isolate FUDT windows via CDS->assembly alignment (needs BLAST+).",
    )
    parser.add_argument(
        "--skip-esm2",
        action="store_true",
        help="Skip ESM2 (compute FUDT only); useful when validating the FUDT path.",
    )
    args = parser.parse_args()
    output_root = args.output_root or osp.join(_data_root(), ISOLATE_EMBED_REL)
    run(
        isolate_id=args.isolate,
        esm2_model_name=args.esm2_model,
        output_root=output_root,
        n_genes=args.n_genes,
        dry_run=args.dry_run,
        with_fudt=args.with_fudt,
        skip_esm2=args.skip_esm2,
    )


if __name__ == "__main__":
    main()
