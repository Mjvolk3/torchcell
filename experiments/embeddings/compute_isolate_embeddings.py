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

  * FUDT (FungalUpDownTransformer) species-aware 5'/3' window embeddings -- STUBBED. See
    ``compute_isolate_fudt`` for the precise gap: the gene-keyed store carries ONLY the CDS
    body (no upstream promoter / downstream terminator flank), so per-isolate FUDT windows
    are NOT reconstructible from the mirrored per-isolate store today.

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
import tarfile
from typing import TYPE_CHECKING

import torch
from Bio.Seq import Seq
from dotenv import load_dotenv
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

if TYPE_CHECKING:
    from torchcell.models.esm2 import Esm2

# Peter 2018 gene-keyed per-isolate CDS store (mirror-relative, sha256-pinned by the loader).
REFGENE_TAR_REL = (
    "torchcell-library/peterGenomeEvolution10112018/data/"
    "allReferenceGenesWithSNPsAndIndelsInferred.tar.gz"
)
DEFAULT_ESM2_MODEL = "esm2_t33_650M_UR50D"
ISOLATE_EMBED_REL = "data/scerevisiae/caudal2024_isolate_embeddings"


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


def compute_isolate_fudt(
    isolate_id: str,
    orfs: dict[str, str],
    fudt_model_name: str,
    output_root: str,
) -> str:
    """STUB: per-isolate FUDT 5'/3' window embeddings.

    GAP (STEP-1 finding): the FungalUpDownTransformer 5' window
    (``window_five_prime``, 1003 bp = ~1000 bp upstream promoter + CDS codon) and 3' window
    (``window_three_prime``, 300 bp downstream terminator) require genomic FLANKING context
    around each gene. The Peter gene-keyed store
    (``allReferenceGenesWithSNPsAndIndelsInferred.tar.gz``) contains ONLY the CDS body
    (verified: header coord span == sequence length; pure ACGT; ATG..stop), so no isolate
    upstream/downstream sequence is available from it.

    EXACT INPUTS NEEDED TO UNBLOCK (any one):
      1. Per-isolate flanking windows: slice +/-1003 bp (5') / +300 bp (3') from each
         isolate's own genome assembly (``1011Assemblies.tar.gz``, contig-level de novo
         FASTA, N50 ~136 kb) using per-strain gene coordinates. The assemblies carry NO
         per-strain GFF/annotation, so this needs a gene-locus mapping step (align each
         S288C gene to the isolate contigs, or lift over coordinates) before slicing.
      2. Interim reference-reuse: for isolates whose only divergence at a gene is a CODING
         SNP/indel, the promoter/terminator equals the S288C window -> reuse the existing
         S288C FUDT embedding (``torchcell.datasets.fungal_up_down_transformer``). This adds
         no per-isolate signal and is a model-wiring fallback, not a recompute here.

    Until (1) is provided, this function raises to avoid emitting a silently-wrong
    (reference-substituted) per-isolate FUDT artifact.
    """
    raise NotImplementedError(
        "Per-isolate FUDT embeddings are blocked: the Peter gene-keyed store holds only CDS "
        "bodies, not the 5' promoter / 3' terminator flank the FungalUpDownTransformer "
        "windows require. Provide per-isolate flanking windows from 1011Assemblies.tar.gz "
        "(needs a per-strain gene-locus mapping) -- see compute_isolate_fudt docstring."
    )


def run(
    isolate_id: str,
    esm2_model_name: str,
    output_root: str,
    n_genes: int | None,
    dry_run: bool,
    with_fudt: bool,
) -> None:
    """Extract an isolate's ORFs, translate, and compute embeddings (ESM2; FUDT stubbed)."""
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
    esm2_path = compute_isolate_esm2(isolate_id, proteins, esm2_model_name, output_root)
    print(f"  wrote ESM2 embeddings -> {esm2_path}")
    if with_fudt:
        compute_isolate_fudt(isolate_id, orfs, "species_upstream", output_root)


def main() -> None:
    """CLI entry point for per-isolate embedding computation."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-isolate ESM2 (implemented) + FUDT (stubbed) sequence embeddings "
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
        help="Attempt FUDT embeddings (currently raises -- see compute_isolate_fudt).",
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
    )


if __name__ == "__main__":
    main()
