# experiments/019-simb-multimodal/scripts/knn_embedding_probe.py
# [[experiments.019-simb-multimodal.scripts.knn_embedding_probe]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/knn_embedding_probe
"""Parameter-free kNN readout in embedding space -- does embedding proximity imply
phenotype similarity?

WHY. The _005 sweep found every content embedding (ESM2, CaLM, NT, FUDT, prot_T5) tied
with `random_100` on validation. Two very different explanations fit that:

  (i)  the embeddings carry no deletion-phenotype-transferable signal, or
  (ii) the signal is there and the transformer is not extracting it (in which case the
       graph-regularization channel may be doing all the work in every arm, since a FIXED
       random vector plus graph attention still yields a computable neighborhood
       fingerprint for an unseen gene).

kNN separates them. It has ZERO learned parameters, so it cannot memorize, and it has no
access to the graph -- its only input is embedding geometry. If kNN beats the transformer,
the signal is in the embedding and the model is failing to use it. If kNN also ties with
`random_100`, the embeddings genuinely lack transferable structure at this data scale.

WHY kNN AND NOT RF/SVR. The traditional-ML baselines in this repo (experiments/002, and
[[paper.nature-biotech.si.traditional-ml-baselines]]) regress a SCALAR (fitness) and had
~1e5 samples. Here the target is a VECTOR -- 278 CalMorph features or 6,169 reporter genes
-- from ~4,000 training genes. RF/SVR would need one independent model per output
dimension with no parameter sharing. kNN is the natural vector-output analogue: it copies
and averages whole target vectors.

METHOD. Training genes g in T carry embedding e_g in R^d and target Y_g in R^F. For a
held-out gene g*:

    s_j = cos(e_g*, e_gj)                          (cosine similarity, train genes only)
    take the top-k by s_j, weights w_j = relu(s_j) / sum_j relu(s_j)
    Yhat_g* = sum_j w_j Y_gj                       (the whole F-vector at once)

Scored with the SAME metric the sweep ranks on:

    pearson_per_feature = (1/F) sum_f  r( Yhat[:,f], Y[:,f] )   across held-out strains

Uses the identical split the running jobs use (`index_seed_0.json`, 4074/534/517), so the
numbers are directly comparable to the sweep leaderboards. Also reports the TRAIN-MEAN
predictor as an anchor: it is the mean-collapse solution, and its per-feature Pearson must
be ~0 by construction.

EMBEDDINGS. `fudt` is evaluated as the CONCATENATION of upstream+downstream (it is the
S. cerevisiae-specific promoter/terminator representation and only makes sense as a pair).
Note fudt_downstream is missing the 28 mitochondrial Q0* genes, so strains perturbing those
are dropped for that embedding only; the count is reported.

Run from repo root:
    python experiments/019-simb-multimodal/scripts/knn_embedding_probe.py
"""

from __future__ import annotations

import json
import os
import os.path as osp
from typing import Any

import lmdb
import numpy as np
import torch
from dotenv import load_dotenv

# MUST precede the torchcell imports: torchcell.graph.sgd reads DATA_ROOT at IMPORT time
# and raises if it is unset, so loading the .env inside main() is too late.
load_dotenv("/home/michaelvolk/Documents/projects/torchcell/.env")

from torchcell.datamodels.calmorph_labels import CALMORPH_LABELS  # noqa: E402
from torchcell.datasets.node_embedding_builder import NodeEmbeddingBuilder  # noqa: E402
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome  # noqa: E402

# Same three dropped features as the training config (multitask.drop_features.global).
DROPPED = ["A113_A", "D203", "D205"]
MORPH_FEATS = [f for f in CALMORPH_LABELS if f not in DROPPED]

# Concatenated groups are expressed as tuples; a bare string is a single embedding.
#
# Chosen to span FOUR distinct representational axes, because "the embeddings don't work"
# is only meaningful if the axes are separated:
#   PROTEIN         prot_T5_all, esm2 -- what the deleted protein IS.
#   REGULATORY DNA  fudt (promoter+terminator) and its Nucleotide Transformer analogue
#                   (5' 1003 + 3' 300) -- how the gene is CONTROLLED. A different axis
#                   from protein identity, so it gets a like-for-like NT comparator
#                   rather than being represented by fudt alone.
#   LOCUS DNA       nt_window_5979 -- the gene plus its genomic neighbourhood.
#   CODON/ORF       calm -- a codon language model over the coding sequence.
#   CONTROL         random_100 -- unique but meaningless; must score ~0.
#
# Absent: nt_window_three_prime_5979, which would give the 5979-scale 3' partner to
# nt_window_five_prime_5979. Its .pt is built and present on disk, but loading it raises
# `Invalid model_name 'window_three_prime_5979'` -- the name reaches BaseEmbeddingDataset's
# check having lost its `nt_` prefix. Tracked separately; the 1003/300 pair covers the
# regulatory axis meanwhile.
EMBEDDINGS: dict[str, tuple[str, ...]] = {
    "prot_T5_all": ("prot_T5_all",),
    "esm2_t33_650M_UR50D_all": ("esm2_t33_650M_UR50D_all",),
    "fudt_up_down": ("fudt_upstream", "fudt_downstream"),
    "nt_5prime_3prime": ("nt_window_five_prime_1003", "nt_window_three_prime_300"),
    "nt_window_5979": ("nt_window_5979",),
    "calm": ("calm",),
    # Explicit codon USAGE (64-d frequency vector), distinct from calm's learned codon
    # language model. Separates "is the signal raw codon bias / expression level proxy"
    # from "is it what the language model abstracted on top of codon bias".
    "codon_frequency": ("codon_frequency",),
    "random_100": ("random_100",),
}
K_GRID = [1, 3, 5, 10, 25, 50]

DATASET_TAG = "019-simb-multimodal/fig3_core"
SEED = 0


def _load_split(base: str) -> dict[str, list[int]]:
    with open(osp.join(base, "data_module_cache", f"index_seed_{SEED}.json")) as f:
        return json.load(f)


def _load_targets(base: str, ids: list[int]) -> dict[str, Any]:
    """gene -> {morphology: [278], expression: [6169]} for the given LMDB row ids."""
    env = lmdb.open(osp.join(base, "processed", "lmdb"), readonly=True, lock=False, subdir=True)
    genes: list[str] = []
    morph: dict[str, np.ndarray] = {}
    expr: dict[str, np.ndarray] = {}
    expr_keys: list[str] | None = None
    with env.begin() as txn:
        for i in ids:
            recs = json.loads(txn.get(f"{i}".encode()).decode())
            if isinstance(recs, dict):
                recs = [recs]
            perts = recs[0]["experiment"]["genotype"]["perturbations"]
            if len(perts) != 1:
                continue  # single-deletion strains only; the probe is gene-indexed
            gene = perts[0]["systematic_gene_name"]
            genes.append(gene)
            for r in recs:
                ph = r["experiment"]["phenotype"]
                if ph["label_name"] == "calmorph":
                    d = ph["calmorph"]
                    morph[gene] = np.array([d[f] for f in MORPH_FEATS], dtype=np.float32)
                elif ph["label_name"] == "expression_log2_ratio":
                    d = ph["expression_log2_ratio"]
                    if expr_keys is None:
                        expr_keys = sorted(d)
                    expr[gene] = np.array([d[k] for k in expr_keys], dtype=np.float32)
    env.close()
    return {"genes": genes, "morphology": morph, "expression": expr}


def _embedding_matrix(names: tuple[str, ...], data_root: str, genome: Any) -> dict[str, np.ndarray]:
    """gene -> concatenated embedding vector, over the given embedding names."""
    per_gene: dict[str, list[np.ndarray]] = {}
    counts: dict[str, int] = {}
    for name in names:
        built = NodeEmbeddingBuilder.build(
            embedding_names=[name], data_root=data_root, genome=genome, graph=None
        )
        ds = built[name]
        for item in ds:
            # BaseEmbeddingDataset loads with map_location=cuda when a GPU is visible,
            # so move to host before numpy.
            vec = torch.cat([t.flatten() for t in item.embeddings.values()]).cpu().numpy()
            per_gene.setdefault(item.id, []).append(vec)
            counts[item.id] = counts.get(item.id, 0) + 1
    # keep only genes present in EVERY constituent embedding
    return {
        g: np.concatenate(v).astype(np.float32)
        for g, v in per_gene.items()
        if counts[g] == len(names)
    }


def _pearson_per_feature(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean over features of the across-strain Pearson. pred/true: [n_strains, n_features]."""
    p = pred - pred.mean(axis=0, keepdims=True)
    t = true - true.mean(axis=0, keepdims=True)
    denom = np.sqrt((p**2).sum(axis=0) * (t**2).sum(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (p * t).sum(axis=0) / denom
    return float(np.nanmean(r))


def _knn_predict(
    e_val: np.ndarray, e_train: np.ndarray, y_train: np.ndarray, k: int
) -> np.ndarray:
    en_v = e_val / np.linalg.norm(e_val, axis=1, keepdims=True)
    en_t = e_train / np.linalg.norm(e_train, axis=1, keepdims=True)
    sim = en_v @ en_t.T  # [n_val, n_train] cosine
    idx = np.argpartition(-sim, kth=min(k, sim.shape[1] - 1), axis=1)[:, :k]
    rows = np.arange(sim.shape[0])[:, None]
    w = np.maximum(sim[rows, idx], 0.0)
    w = w / np.clip(w.sum(axis=1, keepdims=True), 1e-12, None)
    return np.einsum("nk,nkf->nf", w, y_train[idx])


def main() -> None:
    data_root = os.environ["DATA_ROOT"]
    base = osp.join(data_root, "data/torchcell/experiments", DATASET_TAG)

    genome = SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
        overwrite=False,
    )
    genome.drop_empty_go()

    split = _load_split(base)
    train = _load_targets(base, split["train"])
    val = _load_targets(base, split["val"])
    print(f"split seed {SEED}: train strains={len(split['train'])} val strains={len(split['val'])}")
    for mod in ("morphology", "expression"):
        print(f"  {mod:<12} train genes={len(train[mod]):>5}  val genes={len(val[mod]):>4}")

    results: dict[str, Any] = {"seed": SEED, "k_grid": K_GRID, "arms": {}}

    for label, names in EMBEDDINGS.items():
        emb = _embedding_matrix(names, data_root, genome)
        dim = len(next(iter(emb.values())))
        print(f"\n### {label}  ({'+'.join(names)}, dim={dim}, genes={len(emb)})")
        arm: dict[str, Any] = {"dim": dim, "n_genes": len(emb), "modalities": {}}

        for mod in ("morphology", "expression"):
            tr_genes = [g for g in train[mod] if g in emb]
            va_genes = [g for g in val[mod] if g in emb]
            dropped = len(val[mod]) - len(va_genes)
            e_tr = np.stack([emb[g] for g in tr_genes])
            e_va = np.stack([emb[g] for g in va_genes])
            y_tr = np.stack([train[mod][g] for g in tr_genes])
            y_va = np.stack([val[mod][g] for g in va_genes])

            per_k = {}
            for k in K_GRID:
                pred = _knn_predict(e_va, e_tr, y_tr, k)
                per_k[k] = _pearson_per_feature(pred, y_va)
            # Anchor: predict the train mean for every strain -> per-feature r is ~0
            # (zero variance across strains), i.e. the mean-collapse solution.
            mean_pred = np.repeat(y_tr.mean(axis=0, keepdims=True), len(va_genes), axis=0)
            anchor = _pearson_per_feature(mean_pred, y_va)

            best_k = max(per_k, key=lambda kk: per_k[kk])
            print(
                f"  {mod:<12} train={len(tr_genes):>5} val={len(va_genes):>4}"
                f" dropped={dropped:>3} | "
                + "  ".join(f"k={k}:{per_k[k]:+.4f}" for k in K_GRID)
                + f"  || best k={best_k} r={per_k[best_k]:+.4f}  train-mean anchor={anchor:+.4f}"
            )
            arm["modalities"][mod] = {
                "n_train": len(tr_genes),
                "n_val": len(va_genes),
                "n_val_dropped_missing_embedding": dropped,
                "pearson_per_feature_by_k": {str(k): v for k, v in per_k.items()},
                "best_k": best_k,
                "best_pearson_per_feature": per_k[best_k],
                "train_mean_anchor": anchor,
            }
        results["arms"][label] = arm

    out = osp.join(
        os.environ["EXPERIMENT_ROOT"], "019-simb-multimodal/results/knn_embedding_probe.json"
    )
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
