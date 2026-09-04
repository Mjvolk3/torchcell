# experiments/026-metabolism-flux/scripts/embed_prot_t5.py
# [[experiments.026-metabolism-flux.scripts.embed_prot_t5]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/embed_prot_t5.py

r"""Mean-pooled ProtT5-XL-UniRef50 embeddings for GEM enzymes, sharded over the GPUs.

SEQUENCES COME FROM THE GENOME OBJECT, NOT FROM THE GEM's BUNDLED TABLE
------------------------------------------------------------------------
yeast-GEM ships a ``swissprot.tsv`` with a sequence column, and it is tempting because it
sits beside the model. It is the wrong source. Everything in torchcell derives from the
S288C reference FASTA plus the GAF, so a protein sequence has to come from
``SCerevisiaeGenome`` like every other sequence does, through ``genome[orf].protein.seq``,
which is what the ESM2 embedding dataset already uses.

The difference is provenance rather than content. A sequence taken from a third-party
table is a claim by that table's compiler; the same sequence taken from the genome is on
the same derivation chain as the perturbation ontology that edits it, so a strain's
genotype and the protein a predictor sees cannot silently disagree. Gene names from the
GEM are reconciled to current R64 identifiers through the shared
``genome.resolve_gene_name``, so a GEM gene carrying a retired or common name still lands
on the right ORF instead of being dropped.

SHARDING
--------
ProtT5-XL is a 3B-parameter encoder and there are four idle GPUs, so the distinct
sequences are split by index across them and each shard writes its own ``npz``, merged by
``--merge``. Within a shard, sequences are sorted by length and batched, which is where
most of the speedup comes from: padding a batch of similar-length proteins wastes almost
nothing, and the per-sequence masked mean makes the result identical to running them one
at a time.

Truncation and the ``UZOB -> X`` substitution match UniKP's preprocessing exactly, because
they change the features the released regressors see rather than only the plumbing.
"""

import argparse
import glob
import hashlib
import os
import os.path as osp
import re
from typing import cast

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv(
    osp.join(
        "/home/michaelvolk/Documents/projects/torchcell.worktrees",
        "feat/kinetics-equilibrator-datasets",
        ".env",
    )
)
DATA_ROOT = cast(str, os.getenv("DATA_ROOT"))
GEM_DIR = osp.join(DATA_ROOT, "data", "torchcell", "yeast-GEM", "yeast-GEM-9.0.2")
PROT_T5 = "Rostlab/prot_t5_xl_uniref50"


def sequence_key(sequence: str) -> str:
    """Content hash of a sequence, the key the pinned UniKP runner looks up.

    Keying on content rather than on gene makes the cache reusable by any predictor that
    wants a ProtT5 representation, and correct when two ORFs translate to one sequence.
    """
    return hashlib.sha256(sequence.encode()).hexdigest()[:24]


def gem_gene_sequences() -> dict[str, str]:
    """Systematic ORF name -> protein sequence, for every gene in yeast-GEM.

    The GEM supplies the gene LIST; the genome supplies the SEQUENCE.
    """
    import cobra

    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

    model = cobra.io.read_sbml_model(osp.join(GEM_DIR, "model", "yeast-GEM.xml"))
    # go_root is a SEPARATE argument defaulting to a relative "data/go". Left unset it
    # resolves against the working directory, misses, and the constructor tries to fetch
    # go.obo from current.geneontology.org, which answers 403.
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
    )

    sequences: dict[str, str] = {}
    for gene in model.genes:
        resolution = genome.resolve_gene_name(gene.id)
        orf = getattr(resolution, "resolved_name", None) or gene.id
        # Index directly. ``orf in genome`` looks safer and is not: the genome defines no
        # __contains__, so Python falls back to integer indexing and calls genome[0],
        # which raises FeatureNotFoundError before the real lookup ever happens.
        sequences[orf] = str(genome[orf].protein.seq).rstrip("*")

    print(f"{len(model.genes)} GEM genes -> {len(sequences)} sequences from the genome")
    return sequences


def embed(sequences: list[str], device: str, batch_size: int) -> np.ndarray:
    """Mean-pooled ProtT5 embeddings for a list of sequences, batched by length."""
    from transformers import T5EncoderModel, T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained(PROT_T5, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(PROT_T5).to(device).eval()

    prepared = []
    for sequence in sequences:
        trimmed = sequence[:500] + sequence[-500:] if len(sequence) > 1000 else sequence
        prepared.append(re.sub(r"[UZOB]", "X", " ".join(trimmed)))

    # Sort by length so a batch pads to nearly its own longest member, then restore the
    # caller's order at the end.
    order = sorted(range(len(prepared)), key=lambda i: len(prepared[i]))
    vectors = np.zeros((len(prepared), 1024), dtype=np.float32)

    for start in range(0, len(order), batch_size):
        block = order[start : start + batch_size]
        encoded = tokenizer.batch_encode_plus(
            [prepared[i] for i in block], add_special_tokens=True, padding=True
        )
        input_ids = torch.tensor(encoded["input_ids"]).to(device)
        attention_mask = torch.tensor(encoded["attention_mask"]).to(device)
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = output.last_hidden_state.cpu().numpy()
        for row, index in enumerate(block):
            # Drop the final attended token, the end-of-sequence marker, exactly as the
            # unbatched path does. Padding beyond that is never read.
            length = int(attention_mask[row].sum()) - 1
            vectors[index] = hidden[row][:length].mean(axis=0)
        if start % (batch_size * 10) == 0:
            print(f"  {start}/{len(order)}", flush=True)
    return vectors


def main() -> None:
    """Embed one shard of the GEM's proteins, or merge the shards into one cache."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out", required=True)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--dump-sequences", action="store_true")
    parser.add_argument(
        "--sequences",
        default="/scratch/projects/torchcell-scratch/data/torchcell/kinetics/_features/"
        "gem_protein_sequences.json",
        help="Genome-derived ORF to protein sequence map the shards read.",
    )
    args = parser.parse_args()

    # The genome is read exactly ONCE, by --dump-sequences, and the GPU shards read the
    # dump. Four processes opening the gffutils SQLite database concurrently raced and
    # made it briefly unreadable, and the shards need the genome for nothing but a
    # sequence lookup, so the dependency is removed rather than serialized.
    if args.dump_sequences:
        import json

        sequences = gem_gene_sequences()
        os.makedirs(osp.dirname(args.sequences), exist_ok=True)
        with open(args.sequences, "w") as handle:
            json.dump(sequences, handle)
        print(f"wrote {args.sequences}: {len(sequences)} sequences")
        return

    if args.merge:
        keys: list[str] = []
        vectors: list[np.ndarray] = []
        shards = sorted(glob.glob(args.out.replace(".npz", ".shard*.npz")))
        for path in shards:
            loaded = np.load(path, allow_pickle=True)
            keys.extend(loaded["keys"].tolist())
            vectors.append(loaded["vectors"])
        stacked = np.concatenate(vectors)
        # A duplicate key across shards would mean the same sequence was embedded twice,
        # which the index-based split cannot produce; assert rather than dedupe silently.
        if len(set(keys)) != len(keys):
            raise ValueError(f"{len(keys) - len(set(keys))} duplicate keys across shards")
        np.savez_compressed(args.out, keys=np.array(keys), vectors=stacked)
        print(f"merged {len(shards)} shards -> {args.out}: {stacked.shape}")
        return

    import json

    with open(args.sequences) as handle:
        sequences = json.load(handle)
    distinct = sorted(set(sequences.values()))
    mine = [s for i, s in enumerate(distinct) if i % args.num_shards == args.shard]
    print(
        f"shard {args.shard}/{args.num_shards} on {args.device}: "
        f"{len(mine)} of {len(distinct)} distinct sequences",
        flush=True,
    )

    vectors = embed(mine, args.device, args.batch_size)
    target = args.out.replace(".npz", f".shard{args.shard}.npz")
    os.makedirs(osp.dirname(target), exist_ok=True)
    np.savez_compressed(
        target, keys=np.array([sequence_key(s) for s in mine]), vectors=vectors
    )
    print(f"wrote {target}: {vectors.shape}")


if __name__ == "__main__":
    main()
