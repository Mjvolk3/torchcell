# experiments/019-simb-multimodal/scripts/cross_study_conditioning_oracle.py
# [[experiments.019-simb-multimodal.scripts.cross_study_conditioning_oracle]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/cross_study_conditioning_oracle
"""Is the masked-conditioning oracle predicting BIOLOGY or the MEASUREMENT? Cross-study test.

THE QUESTION. `masked_conditioning_oracle.py` measured, WITHIN Kemmeren 2014, that
conditioning on m observed genes predicts the held-out genes at
`pearson_per_feature` = 0.0 / 0.4084 / 0.6756 / 0.7932 for m = 0 / 10 / 100 / 1000
(results/masked_conditioning_oracle.json, n_val = 155 Kemmeren strains). The m = 1000 value
EXCEEDS the replicate-based expression ceiling of 0.775
(`expression_ceiling_replicate.py`, mean_g sqrt(r_g) over the 82 cross-study replicate
pairs). A number above the ceiling for predicting a *re-measurement* is the signature of
predicting something the re-measurement does not share.

THE CONFOUND. Both the observed genes and the held-out genes in that run are read off the
SAME two-colour microarray for the same strain: same hybridization, same dye pair, same
print batch, same per-array normalization. Anything that perturbs one spot on an array
perturbs its neighbours in a reproducible, gene-indexed way. That produces exactly the
signature `residual_covariance_diagnostic.py` found -- a residual gene-gene correlation
pattern that REPLICATES on held-out strains (split-half r = 0.8687 vs a permutation null of
8.45e-05, effective rank 32.78) -- because a technical mode is just as reproducible across
strains as a regulatory module is. Reproducibility separates structure from sampling noise;
it does NOT separate co-regulation from co-measurement. Within one study the two are
inseparable in principle, because every strain carries both at once.

THE SEPARATION. Kemmeren 2014 (GSE42527/42526) and Sameith 2015 (GSE42536) are independent
experiments, and all 82 Sameith single deletions are also in Kemmeren -- the same genotype,
measured twice on different arrays at different times. Array-level technical state is
therefore INDEPENDENT between the two measurements of a strain, while the biology is
shared. So:

    fit Sigma on Kemmeren strains ONLY, excluding the 82 shared ones;
    then on those 82 strains, observe genes M in ONE study and predict genes U
    as measured in the OTHER study.

Any conditioning power that survives is carried by structure the two studies share; any
power that is lost was tied to the individual array.

ARMS (all four use the SAME random observed set and the SAME tuned lam per draw, so the
only thing that differs between them is which study supplied the input and which supplied
the target -- that is what makes the drop attributable):

    within_kem        observe Kemmeren -> predict Kemmeren   (matched control)
    within_sam        observe Sameith  -> predict Sameith    (matched control, other study)
    cross_kem_to_sam  observe Kemmeren -> predict Sameith
    cross_sam_to_kem  observe Sameith  -> predict Kemmeren

`within_kem` vs `cross_kem_to_sam` is THE comparison: identical input values, identical
gene sets, identical strains, identical n -- only the study that measured the target
changes.

WHAT EACH OUTCOME MEANS.
  * cross ~= within  -> the conditioning structure is study-independent, i.e. biological
    (or at least reproduced across labs/arrays/years). The within-study oracle is then a
    fair bound and a masked objective is buying real cross-gene structure.
  * cross << within   -> a large part of the within-study number was same-array technical
    structure. The masked objective would then be partly training the model to reproduce
    Kemmeren's measurement process, and the within-study oracle overstates the prize by
    the size of the gap.
  * cross ~= 0        -> essentially ALL of the within-study conditioning was measurement
    structure.

THE CROSS ARM HAS ITS OWN CEILING, and it is not 1.0. Predicting a NOISY re-measurement
caps at sqrt(reliability); on these same 82 strains the per-feature test-retest correlation
is r = 0.611 and the ceiling is mean_g sqrt(r_g) = 0.775
(`expression_ceiling_replicate.json`). So the cross number must be read against the
`xstudy_agreement` reference computed here on the identical held-out gene set at identical
n: the per-feature Pearson between the two studies' own measurements of those genes. That
reference is what a perfect biological predictor would score, and a cross score near it
means the conditioning has recovered everything recoverable.

METHOD (deliberately identical to `masked_conditioning_oracle.py` so the numbers are
directly comparable -- `conditional_mean` and `per_feature_pearson` are copied verbatim).
  * Residual R = Y - mu, with mu the per-gene mean over the TRAIN Kemmeren strains only.
    The 82 evaluation strains never enter mu or Sigma.
  * E[R_U | R_M] = Sigma_UM (Sigma_MM + lam I)^{-1} R_M, Sigma from TRAIN strains only.
  * lam selected on a TUNE subset of the fitting strains (within-study, since that is the
    only leak-free option), then applied UNCHANGED to all four arms. An oracle-lam score
    (best lam chosen on the evaluation strains themselves) is also recorded, clearly
    labelled, so a cross-arm collapse cannot be blamed on the shared lam.
  * m in {10, 100, 1000}; 5 independent draws of the observed set; mean and sd across draws.
  * Metric: per_feature_pearson on the held-out genes only.

NOTE ON STUDY OFFSETS. Subtracting Kemmeren's per-gene mu from the Sameith values leaves a
per-gene constant offset if the studies normalize differently. That is harmless: a constant
added to a gene's column shifts the prediction by a constant across strains, and
per-feature Pearson is invariant to per-column shifts. Per-gene SCALE differences are not
neutralized and are a residual limitation.

SAMPLE SIZE WARNING. There are only 82 evaluation strains (vs 155 in the within-study run),
so every per-gene correlation has ~80 degrees of freedom and the averaged metric is noisy.
A permutation null (strain correspondence broken between input and target) is computed per
arm per draw to fix the chance level empirically. Do not read differences smaller than the
across-draw sd plus that null spread.

Run from repo root:
    python experiments/019-simb-multimodal/scripts/cross_study_conditioning_oracle.py

Writes ``results/cross_study_conditioning_oracle.json``.
"""

from __future__ import annotations

import json
import os
import os.path as osp
import pickle
from typing import Any

import lmdb
import numpy as np
from dotenv import load_dotenv

_WT_ENV = osp.abspath(osp.join(osp.dirname(__file__), "..", "..", "..", ".env"))
load_dotenv(
    _WT_ENV
    if osp.exists(_WT_ENV)
    else osp.expanduser("~/Documents/projects/torchcell/.env")
)

from torchcell.utils.paths import experiment_results_dir  # noqa: E402

# Same raw study LMDBs `expression_ceiling_replicate.py` pairs. NOT the fig3_core build:
# that one is already merged/deduplicated across studies, and this analysis needs the two
# measurements of a strain kept apart.
KEMMEREN_LMDB = "data/torchcell/microarray_kemmeren2014/processed/lmdb"
SAMEITH_SM_LMDB = "data/torchcell/sm_microarray_sameith2015/processed/lmdb"

SEED = 0
# m = 0 is omitted: predicting the per-gene mean is constant across strains, so the metric
# is identically 0 by construction in every arm (recorded in the within-study run).
M_GRID = [10, 100, 1000]
N_DRAWS = 5
RIDGE_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
# Matches the within-study run's tune size, so lam is selected from a comparably sized
# inner split. The evaluation set is not a free parameter here -- it is the 82 shared
# deletions, whatever that number turns out to be.
N_TUNE = 155

ARMS = ("within_kem", "within_sam", "cross_kem_to_sam", "cross_sam_to_kem")

# Reference values measured elsewhere, quoted so the JSON is self-contained.
WITHIN_STUDY_ORACLE = {"0": 0.0, "10": 0.4084, "100": 0.6756, "1000": 0.7932}
REPLICATE_CEILING = 0.775


def _load_lmdb(path: str) -> list[dict[str, Any]]:
    """Traversal copied from expression_ceiling_replicate.py (pickled records here, not JSON)."""
    env = lmdb.open(path, readonly=True, lock=False, subdir=True)
    records: list[dict[str, Any]] = []
    with env.begin() as txn:
        for _, value in txn.cursor():
            records.append(pickle.loads(value))
    env.close()
    return records


def _deletion(record: dict[str, Any]) -> str:
    """Genotype key used to match a Sameith strain to its Kemmeren counterpart.

    Verbatim from expression_ceiling_replicate.py, which is the authoritative pairing --
    the ceiling this script is arguing against was computed with exactly these pairs.
    """
    perts = record["experiment"]["genotype"]["perturbations"]
    names = sorted(p["systematic_gene_name"] for p in perts)
    return "|".join(names)


def _matrix(records: list[dict[str, Any]], genes: list[str], field: str) -> np.ndarray:
    """[n_strains, n_genes] from the per-gene dicts, NaN where a gene is absent."""
    out = np.full((len(records), len(genes)), np.nan, dtype=np.float64)
    for i, rec in enumerate(records):
        d = rec["experiment"]["phenotype"][field]
        for j, g in enumerate(genes):
            v = d.get(g)
            if v is not None:
                out[i, j] = v
    return out


def per_feature_pearson(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean over FEATURES of the across-STRAIN correlation. Matches the project metric.

    COPIED VERBATIM from masked_conditioning_oracle.py -- the whole point is that the cross
    number and the 0.7932 within number are the same statistic computed the same way.
    """
    p = pred - pred.mean(axis=0, keepdims=True)
    t = true - true.mean(axis=0, keepdims=True)
    num = (p * t).sum(axis=0)
    den = np.sqrt((p**2).sum(axis=0)) * np.sqrt((t**2).sum(axis=0))
    r = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)
    return float(r.mean())


def conditional_mean(
    R_fit: np.ndarray, R_eval: np.ndarray, obs: np.ndarray, unobs: np.ndarray, lam: float
) -> np.ndarray:
    """E[R_U | R_M] = Sigma_UM (Sigma_MM + lam I)^{-1} R_M, Sigma from R_fit only.

    COPIED VERBATIM from masked_conditioning_oracle.py. Note the asymmetry this script
    exploits: `R_fit` (which sets Sigma) and `R_eval` (which supplies R_M) are separate
    arguments, so the conditioner can be fitted on Kemmeren and driven by Sameith values
    without touching the estimator.
    """
    n = R_fit.shape[0]
    A = R_fit[:, obs].astype(np.float64)
    B = R_fit[:, unobs].astype(np.float64)
    s_mm = (A.T @ A) / (n - 1)
    s_um = (B.T @ A) / (n - 1)
    s_mm.flat[:: s_mm.shape[0] + 1] += lam
    # solve(S_mm, R_M^T) rather than an explicit inverse: same answer, better conditioned.
    w = np.linalg.solve(s_mm, R_eval[:, obs].astype(np.float64).T)
    return (s_um @ w).T


def main() -> None:
    data_root = os.environ["DATA_ROOT"]
    rng = np.random.default_rng(SEED)

    kem = _load_lmdb(osp.join(data_root, KEMMEREN_LMDB))
    sam = _load_lmdb(osp.join(data_root, SAMEITH_SM_LMDB))
    print(f"Kemmeren strains: {len(kem)}   Sameith single-mutant strains: {len(sam)}")

    kem_genes = set(kem[0]["experiment"]["phenotype"]["expression_log2_ratio"])
    sam_genes = set(sam[0]["experiment"]["phenotype"]["expression_log2_ratio"])
    genes = sorted(kem_genes & sam_genes)
    print(
        f"reporter genes: kemmeren={len(kem_genes)} sameith={len(sam_genes)} "
        f"shared={len(genes)}"
    )

    # Pairing: verbatim from expression_ceiling_replicate.py.
    kem_by_del = {_deletion(r): i for i, r in enumerate(kem)}
    pairs = [
        (kem_by_del[d], i) for i, r in enumerate(sam) if (d := _deletion(r)) in kem_by_del
    ]
    kem_eval_rows = [k for k, _ in pairs]
    sam_eval_rows = [s for _, s in pairs]
    print(f"shared single deletions (Kemmeren AND Sameith): {len(pairs)}")

    Y_kem_all = _matrix(kem, genes, "expression_log2_ratio")
    Y_sam_all = _matrix(sam, genes, "expression_log2_ratio")

    # Drop any reporter gene that is not fully observed in either matrix. The covariance
    # solve has no NaN semantics, and imputing here would fabricate exactly the kind of
    # cross-gene structure the script is trying to audit.
    keep = np.isfinite(Y_kem_all).all(axis=0) & np.isfinite(
        Y_sam_all[sam_eval_rows]
    ).all(axis=0)
    n_dropped = int((~keep).sum())
    genes = [g for g, k in zip(genes, keep) if k]
    Y_kem_all = Y_kem_all[:, keep]
    Y_sam_all = Y_sam_all[:, keep]
    n_gene = len(genes)
    print(f"complete-case reporter genes: {n_gene}  (dropped {n_dropped} with any NaN)")
    assert n_gene > max(M_GRID) + 100, "not enough complete-case genes for the m grid"

    # Fitting pool = every Kemmeren strain that is NOT one of the shared 82. Excluding them
    # is what makes the evaluation honest: a shared strain in Sigma would let the fit see
    # the very biology it is later asked to predict.
    eval_set = set(kem_eval_rows)
    fit_pool = np.array([i for i in range(len(kem)) if i not in eval_set])
    perm = rng.permutation(len(fit_pool))
    tune_idx = fit_pool[perm[:N_TUNE]]
    train_idx = fit_pool[perm[N_TUNE:]]
    n_eval = len(pairs)
    print(
        f"split: train={len(train_idx)} tune={len(tune_idx)} "
        f"eval={n_eval} (shared strains, held out of both mu and Sigma)"
    )

    # Gene means from TRAIN ONLY, and the SAME mu is applied to the Sameith values. Any
    # per-gene offset between studies survives this, which is harmless for a per-column
    # Pearson (see the module docstring).
    mu = Y_kem_all[train_idx].mean(axis=0, keepdims=True)
    R_tr = Y_kem_all[train_idx] - mu
    R_tu = Y_kem_all[tune_idx] - mu
    R_kem = Y_kem_all[kem_eval_rows] - mu
    R_sam = Y_sam_all[sam_eval_rows] - mu

    out: dict[str, Any] = {
        "n_strains_kemmeren": len(kem),
        "n_strains_sameith_sm": len(sam),
        "n_shared_deletions_eval": n_eval,
        "n_genes_complete_case": n_gene,
        "n_genes_dropped_nan": n_dropped,
        "n_train": int(len(train_idx)),
        "n_tune": int(len(tune_idx)),
        "seed": SEED,
        "n_draws": N_DRAWS,
        "ridge_grid": RIDGE_GRID,
        "reference_within_study_oracle_155_val_strains": WITHIN_STUDY_ORACLE,
        "reference_replicate_ceiling_mean_sqrt_r": REPLICATE_CEILING,
        "results": [],
    }

    for m in M_GRID:
        # Accumulators, one list per arm plus the two references.
        scores: dict[str, list[float]] = {a: [] for a in ARMS}
        oracle: dict[str, list[float]] = {a: [] for a in ARMS}
        nulls: dict[str, list[float]] = {a: [] for a in ARMS}
        agree: list[float] = []
        chosen_lams: list[float] = []

        for _draw in range(N_DRAWS):
            obs = rng.choice(n_gene, size=m, replace=False)
            mask = np.ones(n_gene, dtype=bool)
            mask[obs] = False
            unobs = np.flatnonzero(mask)

            # lam on the TUNE strains (within-Kemmeren, the only leak-free option), then
            # frozen for all four arms so the within-vs-cross gap cannot be a tuning gap.
            best_lam, best_score = RIDGE_GRID[0], -np.inf
            for lam in RIDGE_GRID:
                pred_tu = conditional_mean(R_tr, R_tu, obs, unobs, lam)
                sc = per_feature_pearson(pred_tu, R_tu[:, unobs])
                if sc > best_score:
                    best_lam, best_score = lam, sc
            chosen_lams.append(best_lam)

            # Input study -> target study for each arm. Sigma always comes from R_tr.
            source = {
                "within_kem": (R_kem, R_kem),
                "within_sam": (R_sam, R_sam),
                "cross_kem_to_sam": (R_kem, R_sam),
                "cross_sam_to_kem": (R_sam, R_kem),
            }
            # Strain-order permutation of the INPUT only: destroys the strain
            # correspondence while preserving every marginal, giving the empirical chance
            # level at n = 82 rather than assuming it is 0.
            shuf = rng.permutation(n_eval)

            for arm, (R_in, R_out) in source.items():
                pred = conditional_mean(R_tr, R_in, obs, unobs, best_lam)
                scores[arm].append(per_feature_pearson(pred, R_out[:, unobs]))

                pred_null = conditional_mean(R_tr, R_in[shuf], obs, unobs, best_lam)
                nulls[arm].append(per_feature_pearson(pred_null, R_out[:, unobs]))

                # Best lam chosen ON THE EVALUATION STRAINS -- optimistically biased, kept
                # only to show a cross-arm collapse is not an artifact of the frozen lam.
                oracle[arm].append(
                    max(
                        per_feature_pearson(
                            conditional_mean(R_tr, R_in, obs, unobs, lam),
                            R_out[:, unobs],
                        )
                        for lam in RIDGE_GRID
                    )
                )

            # What a PERFECT biological predictor of the Sameith measurement could score on
            # this exact held-out gene set: the other study's own measurement of it. This is
            # the cross arm's ceiling at matched n and matched genes.
            agree.append(per_feature_pearson(R_kem[:, unobs], R_sam[:, unobs]))

        def _stat(v: list[float]) -> dict[str, Any]:
            a = np.array(v)
            return {
                "mean": float(a.mean()),
                "sd": float(a.std(ddof=1)),
                "per_draw": [float(x) for x in a],
            }

        row: dict[str, Any] = {
            "m": int(m),
            "n_eval_strains": n_eval,
            "n_held_out_genes": int(n_gene - m),
            "lam": [float(x) for x in chosen_lams],
            "arms": {a: _stat(scores[a]) for a in ARMS},
            "arms_oracle_lam_optimistic": {a: _stat(oracle[a]) for a in ARMS},
            "arms_permuted_strain_null": {a: _stat(nulls[a]) for a in ARMS},
            "xstudy_agreement_ceiling": _stat(agree),
        }
        w = row["arms"]["within_kem"]["mean"]
        c = row["arms"]["cross_kem_to_sam"]["mean"]
        row["retention_cross_over_within"] = float(c / w) if w != 0 else None
        out["results"].append(row)

        print(f"\n=== m = {m}   (n_eval = {n_eval} strains, {n_gene - m} held-out genes) ===")
        print(f"{'arm':<20}{'mean':>9}{'sd':>8}{'null':>9}{'oracle-lam':>12}")
        for a in ARMS:
            print(
                f"{a:<20}{np.mean(scores[a]):>9.4f}"
                f"{np.std(scores[a], ddof=1):>8.4f}"
                f"{np.mean(nulls[a]):>9.4f}{np.mean(oracle[a]):>12.4f}"
            )
        print(
            f"{'xstudy agreement':<20}{np.mean(agree):>9.4f}{np.std(agree, ddof=1):>8.4f}"
            "   <- ceiling for the cross arms"
        )
        print(
            f"  within-study run at this m (155 Kemmeren val strains): "
            f"{WITHIN_STUDY_ORACLE[str(m)]:.4f}"
        )
        print(f"  cross/within retention: {row['retention_cross_over_within']:.3f}")
        print(f"  lam (tuned, per draw): {sorted(set(chosen_lams))}")

    path = osp.join(
        experiment_results_dir("019-simb-multimodal", __file__),
        "cross_study_conditioning_oracle.json",
    )
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}")

    print(
        "\nREAD THIS AS: `within_kem` and `cross_kem_to_sam` share the input values, the\n"
        "gene sets, the strains and lam -- only the study that measured the TARGET differs.\n"
        "Their ratio is the fraction of the within-study conditioning that survives being\n"
        "measured on a different array. Compare the cross arms against `xstudy_agreement`,\n"
        "not against 1.0: that is the most any predictor of a noisy re-measurement can get."
    )


if __name__ == "__main__":
    main()
