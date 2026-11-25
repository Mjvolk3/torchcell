# experiments/050-ecoli-benchmark/scripts/embed_genes.py
# [[experiments.050-ecoli-benchmark.scripts.embed_genes]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/050-ecoli-benchmark/scripts/embed_genes
"""Mean-pooled sequence-model embeddings of every protein-coding gene of the two genomes.

One model per run (``--model``). ESM-2 and ProtT5 read the protein sequence. CaLM and
GenSLM read the coding sequence as codons, the Nucleotide Transformers read it as
6-mers, and ``codon_frequency`` counts its codons. Identical sequences are embedded
once. A sequence longer than the model's window is truncated from the 3' / C-terminal
end (1,022 residues or codons for ESM-2, ProtT5 and CaLM; the model's own token limit
for the Nucleotide Transformers; 2,046 codons for GenSLM).

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


CODONS = [a + b + c for a in "ACGT" for b in "ACGT" for c in "ACGT"]


def embed_codon_frequency(
    unique: list[str], device: torch.device
) -> dict[str, np.ndarray]:
    """Relative frequency of each of the 64 codons in the coding sequence."""
    index = {codon: i for i, codon in enumerate(CODONS)}
    out: dict[str, np.ndarray] = {}
    for s in unique:
        counts = np.zeros(64)
        for k in range(0, len(s) - len(s) % 3, 3):
            codon = s[k : k + 3]
            if codon in index:
                counts[index[codon]] += 1
        out[s] = (counts / max(counts.sum(), 1)).astype(np.float32)
    return out


@torch.no_grad()
def embed_nucleotide_transformer(
    unique: list[str], device: torch.device, name: str
) -> dict[str, np.ndarray]:
    """Mean over non-pad tokens of the last layer of a Nucleotide Transformer.

    The coding sequence is tokenized as 6-mers and truncated to the model's token
    limit (1,000 tokens for v1, 2,048 for v2).
    """
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(name, trust_remote_code=True)
    model = model.eval().to(device)
    limit = int(tokenizer.model_max_length)
    out: dict[str, np.ndarray] = {}
    order = sorted(unique, key=len)
    for i in range(0, len(order), 8):
        batch = order[i : i + 8]
        tokens = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=limit
        )["input_ids"].to(device)
        mask = tokens != tokenizer.pad_token_id
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            hidden = model(
                tokens,
                attention_mask=mask,
                encoder_attention_mask=mask,
                output_hidden_states=True,
            )["hidden_states"][-1].float()
        weights = mask.unsqueeze(-1)
        pooled = (hidden * weights).sum(1) / weights.sum(1)
        for j, s in enumerate(batch):
            out[s] = pooled[j].cpu().numpy()
        if i % 400 == 0:
            print(f"{name} {i}/{len(order)}", flush=True)
    return out


@torch.no_grad()
def embed_genslm(
    unique: list[str], device: torch.device, name: str
) -> dict[str, np.ndarray]:
    """Mean over codon positions of the last layer of a GenSLM foundation model."""
    from torchcell.models.genslm import GENSLM_SEQ_LENGTH, GenSLM

    model = GenSLM(name)
    out: dict[str, np.ndarray] = {}
    order = sorted(unique, key=len)
    for i in range(0, len(order), 8):
        batch = order[i : i + 8]
        trimmed = [s[: 3 * min(len(s) // 3, GENSLM_SEQ_LENGTH - 2)] for s in batch]
        pooled = model.embed(trimmed, mean_embedding=True).float().cpu().numpy()
        for j, s in enumerate(batch):
            out[s] = pooled[j]
        if i % 400 == 0:
            print(f"{name} {i}/{len(order)}", flush=True)
    return out


NT_V1 = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
NT_V2 = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
EMBEDDERS = {
    "esm2": embed_esm2,
    "prot_t5": embed_prot_t5,
    "calm": embed_calm,
    "codon_frequency": embed_codon_frequency,
    "nt_v1_2500m": lambda u, d: embed_nucleotide_transformer(u, d, NT_V1),
    "nt_v2_500m": lambda u, d: embed_nucleotide_transformer(u, d, NT_V2),
    "genslm_25m": lambda u, d: embed_genslm(u, d, "genslm_25M_patric"),
    "genslm_250m": lambda u, d: embed_genslm(u, d, "genslm_250M_patric"),
}
PROTEIN_MODELS = ("esm2", "prot_t5")


def main() -> None:
    """Embed both genomes with one model and write one npz per genome."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, choices=tuple(EMBEDDERS))
    args = parser.parse_args()
    load_dotenv(".env")
    out_dir = osp.join(
        os.environ["DATA_ROOT"],
        "data/torchcell/experiments/050-ecoli-benchmark/embeddings",
        args.model,
    )
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    by_set = sequences("protein" if args.model in PROTEIN_MODELS else "cds")
    unique = sorted({s for table in by_set.values() for s in table.values()})
    print(
        f"{args.model}: {len(unique)} unique sequences, device {device}, "
        f"threads {torch.get_num_threads()}",
        flush=True,
    )
    started = time.perf_counter()
    vectors = EMBEDDERS[args.model](unique, device)
    print(f"embedded in {time.perf_counter() - started:.0f} s", flush=True)
    for assembly_set, table in by_set.items():
        genes = sorted(table)
        x = np.stack([vectors[table[g]] for g in genes]).astype(np.float32)
        path = osp.join(out_dir, f"{assembly_set}.npz")
        np.savez_compressed(path, genes=np.array(genes), x=x)
        print(f"wrote {path} {x.shape}")


if __name__ == "__main__":
    main()
