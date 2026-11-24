# experiments/050-ecoli-benchmark/scripts/embed_genes.py
# [[experiments.050-ecoli-benchmark.scripts.embed_genes]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/050-ecoli-benchmark/scripts/embed_genes
"""Mean-pooled sequence-model embeddings of every protein-coding gene of the two genomes.

One model per run (``--model esm2 | prot_t5 | calm``). ESM-2 and ProtT5 read the protein
sequence, CaLM reads the coding sequence as codons. Identical sequences are embedded
once. A sequence longer than the model's window is truncated from the 3' / C-terminal
end and the number truncated is printed.

Output, one file per genome::

    $DATA_ROOT/data/torchcell/experiments/050-ecoli-benchmark/embeddings/<model>/<assembly_set>.npz

with ``genes`` (locus tags, sorted) and ``x`` (float32, genes by dimension).
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import time

import numpy as np
import torch
from dotenv import load_dotenv

from torchcell.sequence.genome.standard import load_genome

ASSEMBLY_SETS = ("ecoli_K12_BW25113_ASM75055v1", "ecoli_K12_MG1655_ASM584v2")
ESM2 = "facebook/esm2_t33_650M_UR50D"
PROT_T5 = "Rostlab/prot_t5_xl_uniref50"
MAX_RESIDUES = 1022  # ESM-2 position limit; the same cap is applied to ProtT5 and CaLM


def sequences(kind: str) -> dict[str, dict[str, str]]:
    """Assembly set -> locus tag -> protein or coding sequence, protein-coding genes only."""
    out: dict[str, dict[str, str]] = {}
    for assembly_set in ASSEMBLY_SETS:
        genome = load_genome(assembly_set)
        table: dict[str, str] = {}
        for locus in sorted(genome.gene_set):
            gene = genome[locus]
            protein = gene.protein
            if protein is None or len(protein.seq) == 0:
                continue
            if kind == "protein":
                table[locus] = str(protein.seq).rstrip("*")
            else:
                cds = gene.cds
                table[locus] = str(getattr(cds, "seq", cds)).upper()
        out[assembly_set] = table
    return out


@torch.no_grad()
def embed_esm2(unique: list[str], device: torch.device) -> dict[str, np.ndarray]:
    """Mean over residue positions of the last layer of ESM-2 650M."""
    from transformers import AutoTokenizer, EsmModel

    tokenizer = AutoTokenizer.from_pretrained(ESM2)
    model = EsmModel.from_pretrained(ESM2).eval().to(device)
    out: dict[str, np.ndarray] = {}
    order = sorted(unique, key=len)
    for i in range(0, len(order), 8):
        batch = order[i : i + 8]
        tokens = tokenizer(
            [s[:MAX_RESIDUES] for s in batch], return_tensors="pt", padding=True
        ).to(device)
        hidden = model(**tokens).last_hidden_state
        for j, s in enumerate(batch):
            n = min(len(s), MAX_RESIDUES)
            out[s] = hidden[j, 1 : n + 1].mean(0).float().cpu().numpy()
        if i % 400 == 0:
            print(f"esm2 {i}/{len(order)}", flush=True)
    return out


@torch.no_grad()
def embed_prot_t5(unique: list[str], device: torch.device) -> dict[str, np.ndarray]:
    """Mean over residue positions of the ProtT5-XL-UniRef50 encoder."""
    import re

    from transformers import T5EncoderModel, T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained(PROT_T5, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(PROT_T5).eval().to(device)
    out: dict[str, np.ndarray] = {}
    order = sorted(unique, key=len)
    for i in range(0, len(order), 8):
        batch = order[i : i + 8]
        spaced = [" ".join(re.sub(r"[UZOB]", "X", s[:MAX_RESIDUES])) for s in batch]
        tokens = tokenizer(spaced, return_tensors="pt", padding=True).to(device)
        hidden = model(**tokens).last_hidden_state
        for j, s in enumerate(batch):
            n = min(len(s), MAX_RESIDUES)
            out[s] = hidden[j, :n].mean(0).float().cpu().numpy()
        if i % 400 == 0:
            print(f"prot_t5 {i}/{len(order)}", flush=True)
    return out


@torch.no_grad()
def embed_calm(unique: list[str], device: torch.device) -> dict[str, np.ndarray]:
    """Mean over codon positions of the last layer of CaLM (CPU; one sequence a call)."""
    from calm import CaLM

    model = CaLM()
    out: dict[str, np.ndarray] = {}
    for i, s in enumerate(unique):
        codons = len(s) // 3
        cds = s[: 3 * min(codons, MAX_RESIDUES)]
        out[s] = model.embed_sequence(cds, average=True)[0].float().cpu().numpy()
        if i % 400 == 0:
            print(f"calm {i}/{len(unique)}", flush=True)
    return out


def main() -> None:
    """Embed both genomes with one model and write one npz per genome."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, choices=("esm2", "prot_t5", "calm"))
    args = parser.parse_args()
    load_dotenv(".env")
    out_dir = osp.join(
        os.environ["DATA_ROOT"],
        "data/torchcell/experiments/050-ecoli-benchmark/embeddings",
        args.model,
    )
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    by_set = sequences("cds" if args.model == "calm" else "protein")
    unique = sorted({s for table in by_set.values() for s in table.values()})
    unit = 3 if args.model == "calm" else 1
    truncated = sum(len(s) // unit > MAX_RESIDUES for s in unique)
    print(
        f"{args.model}: {len(unique)} unique sequences, {truncated} truncated to "
        f"{MAX_RESIDUES}, device {device}, threads {torch.get_num_threads()}",
        flush=True,
    )
    started = time.perf_counter()
    embed = {"esm2": embed_esm2, "prot_t5": embed_prot_t5, "calm": embed_calm}
    vectors = embed[args.model](unique, device)
    print(f"embedded in {time.perf_counter() - started:.0f} s", flush=True)
    for assembly_set, table in by_set.items():
        genes = sorted(table)
        x = np.stack([vectors[table[g]] for g in genes]).astype(np.float32)
        path = osp.join(out_dir, f"{assembly_set}.npz")
        np.savez_compressed(path, genes=np.array(genes), x=x)
        print(f"wrote {path} {x.shape}")


if __name__ == "__main__":
    main()
