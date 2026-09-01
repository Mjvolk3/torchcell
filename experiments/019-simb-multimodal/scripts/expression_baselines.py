# experiments/019-simb-multimodal/scripts/expression_baselines.py
# [[experiments.019-simb-multimodal.scripts.expression_baselines]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/expression_baselines
"""The standard baselines the perturbation-response literature scores against.

WHY THIS RUNS BEFORE THE GPU ROUND. Ahlmann-Eltze, Huber and Anders 2025
(doi:10.1038/s41592-025-02772-6) benchmarked seven deep models for gene-perturbation
effect prediction and none outperformed deliberately simple baselines; their winning
linear model is a rank-limited bilinear term plus a per-gene intercept, and it beat GEARS
while using GEARS's own perturbation embeddings. State (doi:10.1101/2025.06.26.661135)
reports the same about itself on genetic perturbations, at 600M parameters. So a
deep-learning number on this task is not interpretable without these, and B2 below carries
a STOP CONDITION on the launch round: if it reaches the incumbent's 0.196 +/- 0.022, the
finding is that a bilinear model suffices and the head comparison is the wrong question.

THE FOUR BASELINES.
  B0  per-gene training mean.  The definitional floor. Constant across strains, so
      per_feature_pearson is 0 BY CONSTRUCTION (every column has zero variance and is
      dropped as undefined) and nmse is 1.0 up to train-to-val mean drift, since the
      per-gene training mean is the quantity nmse divides by. Reported as the anchor that
      proves the data path is wired correctly, not as a competitor.
  B1  no change: predict log2 ratio 0 for every gene.  "Nothing happened." Also constant,
      so also 0 on per_feature_pearson; its nmse says how far the panel sits from the
      wild-type reference.
  B2  the linear model, Y ~ G W P^T + b, fit by ridge on the FIT strains and swept over
      rank.  G is a rank-K gene basis from the fit residuals (SVD); P is the deleted gene's
      EXTERNAL EMBEDDING. THE ONE THAT CAN BEAT US.
  B3  embedding-neighbor average: predict a held-out strain's profile as the mean of the
      k nearest deleted genes in embedding space, among fit strains. Parameter-free.

WHY B2 AND B3 BOTH NEED AN EXTERNAL EMBEDDING, WHICH IS A MEASURED FACT ABOUT THIS PANEL,
NOT A DESIGN PREFERENCE. The loader returns 1,482 single-deletion strains carrying 1,482
DISTINCT deleted genes -- each gene is deleted exactly once. So no deleted gene appears in
both the fit and the val split, and **every validation strain is an unseen perturbation**.
A perturbation representation derived from the deleted gene's own observed response (the
obvious in-data choice) is therefore undefined for every val strain, and was measured to
cover 0 of 155. The only representation that transfers is one built from the gene itself,
which is exactly the regime Ahlmann-Eltze et al. benchmark with external gene embeddings
and the reason their linear model uses G and P from outside the response matrix.

SPLIT, AND THE CAVEAT THAT GOES WITH IT. This uses the ORACLE-FAMILY split -- one seed-0
permutation with val = perm[:155] -- byte-identical to masked_conditioning_oracle.py,
residual_covariance_diagnostic.py and lowrank_output_ceiling.py, via the same
`_load_expression` traversal. That is what makes these numbers commensurable with the
0.727 rank-32 ceiling, the 0.4838 imputation oracle and the 0.117 neighbor probe already
quoted in the retrospective. It is NOT the CellDataModule partition the trained arms use,
so a baseline-to-arm comparison here is approximate in exactly the way every other
oracle-to-model comparison in that document is. Stated rather than hidden.

Metric is `per_feature_pearson` copied from the same family: correlate each gene's column
across strains, average over genes, dropping columns whose prediction or target is
constant. `nmse` is MSE divided by the per-gene variance of the target about the FIT-set
per-gene mean, so B0 lands at 1.0 by construction.

Run from the repo root:
    python experiments/019-simb-multimodal/scripts/expression_baselines.py
"""

from __future__ import annotations

import json
import os
import os.path as osp

import lmdb
import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()

from torchcell.datasets.node_embedding_builder import NodeEmbeddingBuilder  # noqa: E402
from torchcell.graph import SCerevisiaeGraph  # noqa: E402
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome  # noqa: E402
from torchcell.utils.paths import experiment_results_dir  # noqa: E402

# The perturbation representations B2/B3 can use. Deliberately a SUBSET of
# knn_embedding_probe.py's list, spanning its four representational axes at one entry
# each, because this is a baseline gate rather than an embedding study: prot_T5 is its
# measured best (0.1174), calm its measured 019 default (0.0692), and the other two cover
# the regulatory-DNA and graph-derived axes.
EMBEDDINGS: dict[str, tuple[str, ...]] = {
    "prot_T5_all": ("prot_T5_all",),
    "calm": ("calm",),
    "species_lm_5p_3p": ("fudt_upstream", "fudt_downstream"),
    "normalized_chrom_pathways": ("normalized_chrom_pathways",),
}


def _embedding_matrix(
    names: tuple[str, ...], data_root: str, genome: object, graph: object
) -> dict[str, np.ndarray]:
    """Gene -> concatenated embedding vector. Same construction as knn_embedding_probe.py."""
    per_gene: dict[str, list[np.ndarray]] = {}
    counts: dict[str, int] = {}
    for name in names:
        built = NodeEmbeddingBuilder.build(
            embedding_names=[name], data_root=data_root, genome=genome, graph=graph
        )
        for item in built[name]:
            vec = torch.cat(
                [t.flatten() for t in item.embeddings.values()]
            ).cpu().numpy()
            per_gene.setdefault(item.id, []).append(vec)
            counts[item.id] = counts.get(item.id, 0) + 1
    return {
        g: np.concatenate(v).astype(np.float32)
        for g, v in per_gene.items()
        if counts[g] == len(names)
    }

DATASET_TAG = "019-simb-multimodal/fig3_core"
SEED = 0
# Identical to lowrank_output_ceiling.py / masked_conditioning_oracle.py, so val is the
# SAME 155 strains from the SAME permutation of the SAME seed.
N_VAL = 155
N_TUNE = 155
# B2 rank grid. K is the gene-basis rank, L the perturbation-representation rank; the
# published linear model sweeps both, and 33 is on the grid because the residual
# covariance has measured effective rank 32.78.
RANK_GRID = [1, 2, 4, 8, 16, 32, 33, 64, 128, 256]
RIDGE_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
KNN_GRID = [1, 3, 5, 10, 25]


def _load_expression(base: str) -> tuple[list[str], np.ndarray]:
    """Return (perturbed gene per strain, Y [S, F]) for single-deletion strains.

    Byte-identical traversal to lowrank_output_ceiling.py and
    residual_covariance_diagnostic.py. These baselines are only comparable to those
    scripts' ceilings if the matrix is the same one.
    """
    env = lmdb.open(
        osp.join(base, "processed", "lmdb"), readonly=True, lock=False, subdir=True
    )
    genes: list[str] = []
    rows: list[np.ndarray] = []
    keys: list[str] | None = None
    with env.begin() as txn:
        for _, value in txn.cursor():
            recs = json.loads(value.decode())
            if isinstance(recs, dict):
                recs = [recs]
            perts = recs[0]["experiment"]["genotype"]["perturbations"]
            if len(perts) != 1:
                continue
            for r in recs:
                ph = r["experiment"]["phenotype"]
                if ph["label_name"] != "expression_log2_ratio":
                    continue
                d = ph["expression_log2_ratio"]
                if keys is None:
                    keys = sorted(d)
                genes.append(perts[0]["systematic_gene_name"])
                rows.append(np.array([d[k] for k in keys], dtype=np.float32))
    env.close()
    return genes, np.stack(rows)


def per_feature_pearson(pred: np.ndarray, true: np.ndarray) -> float:
    """Correlate each gene's column across strains, average over genes.

    Copied from lowrank_output_ceiling.py, which copied it from
    masked_conditioning_oracle.py. Columns whose prediction or target is (near-)constant
    have an undefined correlation and are DROPPED rather than counted as zero -- which is
    also what the training metric does, and it is why a constant baseline returns 0.0
    (nothing survives) rather than a misleadingly small positive number.
    """
    p = pred - pred.mean(axis=0, keepdims=True)
    t = true - true.mean(axis=0, keepdims=True)
    num = (p * t).sum(axis=0)
    den = np.linalg.norm(p, axis=0) * np.linalg.norm(t, axis=0)
    ok = den > 1e-8
    if not ok.any():
        return 0.0
    return float((num[ok] / den[ok]).mean())


def nmse(pred: np.ndarray, true: np.ndarray, mu_fit: np.ndarray) -> float:
    """MSE divided by the target's variance about the FIT per-gene mean.

    This is the training metric's definition, which is what makes B0 come out at 1.0 and
    therefore what validates the data path: a B0 that is not ~1.0 means the split or the
    matrix is wrong, and every other number here would be wrong with it.
    """
    num = float(((pred - true) ** 2).mean())
    den = float(((true - mu_fit) ** 2).mean())
    return num / den


def _bilinear(
    R_fit: np.ndarray,
    R_val: np.ndarray,
    P_fit: np.ndarray,
    P_val: np.ndarray,
    k_gene: int,
    ridge: float,
) -> np.ndarray:
    """Fit Y ~ G W P^T on fit residuals, predict val residuals.

    G is the top-``k_gene`` right singular vectors of the fit residuals (a gene basis), so
    the fit target in basis coordinates is ``C_fit = R_fit G`` [n_fit, k_gene]. W is then
    the ridge solution of ``C_fit ~ P_fit W``, and the val prediction is
    ``R_hat = (P_val W) G^T``. This is the same object as the published
    ``argmin_W ||Y - (G W P^T + b)||^2`` with b already removed by centering.
    """
    g_basis = np.linalg.svd(R_fit, full_matrices=False)[2][:k_gene].T  # [genes, k]
    c_fit = R_fit @ g_basis  # [n_fit, k]
    ptp = P_fit.T @ P_fit
    w = np.linalg.solve(ptp + ridge * np.eye(ptp.shape[0]), P_fit.T @ c_fit)
    return (P_val @ w) @ g_basis.T


def main() -> None:
    data_root = os.environ["DATA_ROOT"]
    base = osp.join(data_root, "data/torchcell/experiments", DATASET_TAG)
    rng = np.random.default_rng(SEED)

    genes, y_mat = _load_expression(base)
    n_strain, n_gene = y_mat.shape
    print(f"strains={n_strain}  genes={n_gene}  (perturbed genes: {len(set(genes))})")

    perm = rng.permutation(n_strain)
    val_idx, fit_idx = perm[:N_VAL], perm[N_VAL:]
    print(f"split: val={len(val_idx)}  fit={len(fit_idx)}  (oracle-family split, seed {SEED})")

    mu = y_mat[fit_idx].mean(axis=0, keepdims=True)
    y_val = y_mat[val_idx].astype(np.float64)
    r_fit = (y_mat[fit_idx] - mu).astype(np.float64)
    r_val = y_val - mu

    out: dict[str, object] = {
        "generated_by": "experiments/019-simb-multimodal/scripts/expression_baselines.py",
        "dataset_tag": DATASET_TAG,
        "split": {"kind": "oracle-family seed-0 permutation", "n_val": int(N_VAL),
                  "n_fit": int(len(fit_idx)), "n_strain": int(n_strain),
                  "n_gene": int(n_gene)},
    }

    # ---- B0: per-gene training mean -------------------------------------------------
    pred_b0 = np.repeat(mu.astype(np.float64), len(val_idx), axis=0)
    b0 = {"pearson_per_feature": per_feature_pearson(pred_b0, y_val),
          "nmse": nmse(pred_b0, y_val, mu)}
    out["B0_per_gene_mean"] = b0
    print(f"B0 per-gene mean      pf={b0['pearson_per_feature']:+.4f}  nmse={b0['nmse']:.4f}")

    # ---- B1: no change ---------------------------------------------------------------
    pred_b1 = np.zeros_like(y_val)
    b1 = {"pearson_per_feature": per_feature_pearson(pred_b1, y_val),
          "nmse": nmse(pred_b1, y_val, mu)}
    out["B1_no_change"] = b1
    print(f"B1 no change          pf={b1['pearson_per_feature']:+.4f}  nmse={b1['nmse']:.4f}")

    # SELF-CHECK. B0 must be exactly 0 on the metric and ~1 on nmse; if it is not, the
    # matrix or the split is wrong and nothing below this line means anything. Fail loudly
    # rather than emit a plausible number.
    if abs(b0["pearson_per_feature"]) > 1e-9:
        raise ValueError(f"B0 pearson_per_feature is {b0['pearson_per_feature']}, must be 0")
    if not 0.9 < b0["nmse"] < 1.15:
        raise ValueError(f"B0 nmse is {b0['nmse']}, expected ~1.0 by construction")

    # ---- perturbation representation: the deleted gene's external embedding -----------
    gene_arr = np.array(genes)
    fit_genes, val_genes = gene_arr[fit_idx], gene_arr[val_idx]
    n_repeat = len(fit_genes) - len(set(fit_genes))
    print(f"deleted genes: {len(set(gene_arr))} distinct over {n_strain} strains "
          f"({n_repeat} repeated deletions in fit) -- every val strain is an UNSEEN "
          f"perturbation, so B2/B3 require an external embedding")
    out["panel"] = {"n_distinct_deleted_genes": int(len(set(gene_arr))),
                    "n_repeated_deletions_in_fit": int(n_repeat),
                    "all_val_unseen_perturbations": True}

    genome = SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
        overwrite=False,
    )
    genome.drop_empty_go()
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(data_root, "data/sgd/genome"),
        string_root=osp.join(data_root, "data/string"),
        tflink_root=osp.join(data_root, "data/tflink"),
        genome=genome,
    )

    b2_by_emb: dict[str, object] = {}
    b3_by_emb: dict[str, object] = {}
    for emb_name in EMBEDDINGS:
        emb = _embedding_matrix(EMBEDDINGS[emb_name], data_root, genome, graph)
        have = np.array([g in emb for g in gene_arr])
        f_ok = have[fit_idx]
        v_ok = have[val_idx]
        if f_ok.sum() < 50 or v_ok.sum() < 20:
            print(f"  {emb_name}: too few genes covered "
                  f"(fit {int(f_ok.sum())}, val {int(v_ok.sum())}); skipped")
            continue
        dim = len(next(iter(emb.values())))
        p_fit_full = np.stack([emb[g] if g in emb else np.zeros(dim, np.float32)
                               for g in fit_genes]).astype(np.float64)
        p_val_full = np.stack([emb[g] if g in emb else np.zeros(dim, np.float32)
                               for g in val_genes]).astype(np.float64)
        # Standardize on FIT rows only; an embedding column's scale is arbitrary and ridge
        # is not scale-free, so this is part of the estimator rather than preprocessing.
        pm = p_fit_full[f_ok].mean(axis=0, keepdims=True)
        ps = p_fit_full[f_ok].std(axis=0, keepdims=True) + 1e-8
        p_fit_full = (p_fit_full - pm) / ps
        p_val_full = (p_val_full - pm) / ps

        # ---- B2 -----------------------------------------------------------------------
        best_b2 = None
        for k_rank in RANK_GRID:
            if k_rank > min(r_fit[f_ok].shape) - 1:
                continue
            for ridge in RIDGE_GRID:
                r_hat = _bilinear(r_fit[f_ok], r_val, p_fit_full[f_ok],
                                  p_val_full, k_rank, ridge)
                pf = per_feature_pearson(r_hat[v_ok], r_val[v_ok])
                if best_b2 is None or pf > best_b2["pearson_per_feature"]:
                    best_b2 = {"k_gene": k_rank, "ridge": ridge, "emb_dim": int(dim),
                               "pearson_per_feature": pf,
                               "nmse": nmse(r_hat[v_ok] + mu, y_val[v_ok], mu)}
        b2_by_emb[emb_name] = {"best": best_b2, "n_val_scored": int(v_ok.sum())}
        # ---- B3 -----------------------------------------------------------------------
        # Neighbors are FIT genes only, ranked by cosine similarity of the deleted gene's
        # embedding; the prediction is the mean fit RESIDUAL of those neighbors.
        e_fit = p_fit_full[f_ok]
        e_fit_n = e_fit / (np.linalg.norm(e_fit, axis=1, keepdims=True) + 1e-12)
        e_val_n = p_val_full / (np.linalg.norm(p_val_full, axis=1, keepdims=True) + 1e-12)
        sim = e_val_n @ e_fit_n.T
        r_fit_ok = r_fit[f_ok]
        best_b3 = None
        for k in KNN_GRID:
            nn = np.argsort(-sim, axis=1)[:, :k]
            preds = r_fit_ok[nn].mean(axis=1)
            pf = per_feature_pearson(preds[v_ok], r_val[v_ok])
            if best_b3 is None or pf > best_b3["pearson_per_feature"]:
                best_b3 = {"k": k, "pearson_per_feature": pf,
                           "nmse": nmse(preds[v_ok] + mu, y_val[v_ok], mu)}
        b3_by_emb[emb_name] = {"best": best_b3, "n_val_scored": int(v_ok.sum())}
        print(f"  {emb_name:<34} B2 pf={best_b2['pearson_per_feature']:+.4f}  "
              f"B3 pf={best_b3['pearson_per_feature']:+.4f}  "
              f"(dim {dim}, val {int(v_ok.sum())})")

    def _top(d: dict) -> tuple[str, float]:
        best = max(d, key=lambda k: d[k]["best"]["pearson_per_feature"])
        return best, d[best]["best"]["pearson_per_feature"]

    out["B2_bilinear"] = {"by_embedding": b2_by_emb}
    out["B3_neighbor_average"] = {"by_embedding": b3_by_emb}
    if b2_by_emb:
        n2, v2 = _top(b2_by_emb)
        n3, v3 = _top(b3_by_emb)
        out["B2_bilinear"]["best_embedding"] = {"name": n2, "pearson_per_feature": v2}
        out["B3_neighbor_average"]["best_embedding"] = {"name": n3,
                                                        "pearson_per_feature": v3}
        print(f"\nB2 bilinear   BEST  pf={v2:+.4f}  ({n2})")
        print(f"B3 neighbor   BEST  pf={v3:+.4f}  ({n3})")

    dst = osp.join(
        experiment_results_dir("019-simb-multimodal", __file__), "expression_baselines.json"
    )
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {dst}")
    print("\nREFERENCE: the incumbent quantile config scores 0.1965 +/- 0.0222 "
          "(n=8 identical-config runs at 9,900 epochs), best run 0.2382.")


if __name__ == "__main__":
    main()
