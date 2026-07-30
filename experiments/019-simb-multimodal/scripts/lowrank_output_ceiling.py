# experiments/019-simb-multimodal/scripts/lowrank_output_ceiling.py
# [[experiments.019-simb-multimodal.scripts.lowrank_output_ceiling]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/lowrank_output_ceiling
"""CEILING of a rank-r OUTPUT parameterization, so we know which ranks deserve a GPU slot.

THE QUESTION. The 019 decoder emits 6,169 INDEPENDENT per-gene marginals -- one scalar (or
one distributional parameter vector) per reporter gene, each free to move on its own. But
the response matrix does not look 6,169-dimensional. `residual_covariance_diagnostic.py`
measured, on the same matrix this script loads, a participation-ratio effective rank of
32.78, with the rank-32 subspace carrying 59.1% of residual variance and rank-128 carrying
76.0% (split-half r = 0.869 against a permutation null of 8.45e-05, so the structure is
reproducible, not sampling noise). If the achievable signal really lives in ~33 dimensions,
then predicting 6,169 free marginals is the wrong parameterization and the head should
predict COEFFICIENTS in a low-rank gene basis instead.

`multitask.response_basis_rank` (-> `ResponseBasisHead`) already implements exactly that
mechanism, but it has been run once, at a single rank. Before spending GPU on a rank sweep,
this script measures -- with no training and no GPU -- how much of the metric a rank-r
bottleneck can possibly retain.

METHOD.
  1. Load Y [strains x reporter genes] with the same LMDB traversal as
     `residual_covariance_diagnostic.py::_load_expression`, so all three scripts are talking
     about one matrix.
  2. Split strains with seed 0 and the same permutation as
     `masked_conditioning_oracle.py`: val = perm[:155]. Scores are then directly comparable
     to that script's numbers on the same 155 held-out strains.
  3. Per-gene mean mu from the FIT strains only; center both splits with it. Centering with
     a mean that saw val leaks, and this leak is the kind that manufactures correlation.
  4. SVD the fit residuals; V_r = top-r right singular vectors [genes x r]. This is the
     variance-optimal rank-r gene basis obtainable from fit data.
  5. Project the VAL residuals into that basis and reconstruct, R_hat = R_val V_r V_r^T, and
     score with `per_feature_pearson` (correlate across strains per gene, average over
     genes) -- copied verbatim from `masked_conditioning_oracle.py`.

WHY STEP 5 IS AN ORACLE, AND THEREFORE A BOUND. The coefficients C = R_val V_r come from
the VAL residuals themselves. No model has access to them; a model must PREDICT the r
coefficients from the perturbation alone. So this is the score a rank-r head would get if
its coefficient predictor were PERFECT -- every error attributable to the coefficient
predictor has been set to zero, and only the error forced by the r-dimensional bottleneck
remains. Any real rank-r head scores at or below this.

Two honest qualifications on the word "bound":
  * The basis is fit-optimal in the FROBENIUS sense, and per_feature_pearson is not the
    Frobenius norm -- it weights a low-variance gene exactly as much as a high-variance one,
    while the SVD spends its components on high-variance genes. So the train-basis curve is
    an oracle over COEFFICIENTS but not provably the metric-maximizing basis. To close that
    gap the script also reports `ceiling_val_basis`: the same construction with V_r taken
    from the VAL residuals themselves, which is the best any rank-r subspace can do on val
    in reconstruction terms. It is defined only for r <= n_val and is a fully cheating upper
    envelope; the gap between the two curves is how much of the ceiling is basis-choice
    rather than rank.
  * `ResponseBasisHead` ADDS its rank-r term to a full-rank local readout
    (y_i = y_i^local + sum_j b_ij a_j), so this curve bounds a head whose ENTIRE output is
    rank-r. It bounds the low-rank TERM in isolation, not the existing additive arm.

WHAT EACH OUTCOME IMPLIES.
  * If the rank-33 ceiling is already at or above the achievable target, a rank-33 output
    head loses nothing measurable and the 6,169 free marginals are pure parameter waste --
    run the rank sweep at small r.
  * If the curve is still climbing steeply past 128, the low-rank story is weaker than the
    effective-rank number suggests: participation ratio is a variance-concentration
    statistic, and the metric averages CORRELATIONS, in which the many low-variance genes
    the tail components carry count equally. Then the useful ranks are large and the
    parameter saving is small.

THE TARGET TO COMPARE AGAINST is NOT the full-rank value (which is 1.0 by construction --
reconstructing val from val is exact). It is the measured replicate noise ceiling for this
phenotype, `expression_ceiling_replicate.json::primary_ceiling_mean_sqrt_r.ceiling`, since a
model scored against noisy val labels cannot exceed it. Both that number and the masked
conditioning oracle's are copied into the output JSON from their own result files rather
than transcribed, so the comparison never drifts.

Usage
-----
    python experiments/019-simb-multimodal/scripts/lowrank_output_ceiling.py

Writes ``results/lowrank_output_ceiling.json``.
"""

from __future__ import annotations

import json
import os
import os.path as osp
from typing import Any

import lmdb
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from torchcell.utils.paths import experiment_results_dir  # noqa: E402

DATASET_TAG = "019-simb-multimodal/fig3_core"
SEED = 0
# Split sizes copied from masked_conditioning_oracle.py so the held-out 155 strains are the
# SAME 155 strains, drawn from the SAME permutation of the SAME seed.
N_VAL = 155
N_TUNE = 155
RANK_GRID = [1, 2, 4, 8, 16, 32, 33, 64, 128, 256, 512]


def _load_expression(base: str) -> tuple[list[str], np.ndarray]:
    """Return (perturbed gene per strain, Y [S, F]) for single-deletion strains.

    Byte-identical traversal to residual_covariance_diagnostic.py and
    masked_conditioning_oracle.py. This script's entire argument is a comparison against
    those two scripts' numbers, which is only legitimate if the matrix is the same one.
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
    """Mean over FEATURES of the across-STRAIN correlation. Matches the project metric.

    Verbatim from masked_conditioning_oracle.py. A feature with zero predicted variance
    contributes 0, not NaN: a constant prediction has no correlation with anything, and
    dropping such features would silently score the model only on the genes it moved. That
    convention matters here -- at r = 1 most genes are nearly constant in the reconstruction.
    """
    p = pred - pred.mean(axis=0, keepdims=True)
    t = true - true.mean(axis=0, keepdims=True)
    num = (p * t).sum(axis=0)
    den = np.sqrt((p**2).sum(axis=0)) * np.sqrt((t**2).sum(axis=0))
    r = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)
    return float(r.mean())


def _rank_curve(
    Vt: np.ndarray, R_val: np.ndarray, grid: list[int]
) -> list[dict[str, float]]:
    """Ceiling + captured val variance for each rank in ``grid``, for a given basis.

    ``Vt`` is [k, F] with orthonormal rows (the right singular vectors). The reconstruction
    R_val V_r V_r^T is the orthogonal projection of each val residual vector onto the span
    of the first r rows, i.e. the least-squares-optimal rank-r reconstruction OF VAL ITSELF.
    """
    total = float((R_val**2).sum())
    out: list[dict[str, float]] = []
    for r in grid:
        if r > Vt.shape[0]:
            continue
        V_r = Vt[:r].T  # [F, r]
        coeff = R_val @ V_r  # [n_val, r]  -- val's OWN coefficients; this is the oracle
        R_hat = coeff @ V_r.T
        out.append(
            {
                "rank": int(r),
                "ceiling_pearson_per_feature": per_feature_pearson(R_hat, R_val),
                "frac_val_variance_captured": float((R_hat**2).sum() / total),
            }
        )
    return out


def main() -> None:
    data_root = os.environ["DATA_ROOT"]
    base = osp.join(data_root, "data/torchcell/experiments", DATASET_TAG)
    rng = np.random.default_rng(SEED)

    genes, Y = _load_expression(base)
    n_strain, n_gene = Y.shape
    print(f"strains={n_strain}  genes={n_gene}  (perturbed genes: {len(set(genes))})")

    # Same rng call sequence as masked_conditioning_oracle.py -- one permutation, val first.
    perm = rng.permutation(n_strain)
    val_idx = perm[:N_VAL]
    fit_idx = perm[
        N_VAL:
    ]  # train + tune: nothing here is tuned, so all of it is fit data
    train_only_idx = perm[N_VAL + N_TUNE :]  # that script's TRAIN set, for sensitivity
    print(
        f"split: val={len(val_idx)}  fit={len(fit_idx)} "
        f"(train_only={len(train_only_idx)} used as a sensitivity check)"
    )

    # Gene means from FIT strains only. Everything downstream is residual after this mean,
    # which is also what the model's per-gene head is effectively predicting on top of.
    mu = Y[fit_idx].mean(axis=0, keepdims=True)
    R_fit = (Y[fit_idx] - mu).astype(np.float64)
    R_val = (Y[val_idx] - mu).astype(np.float64)

    # Right singular vectors of the fit residuals = the variance-optimal rank-r gene basis
    # available from fit data. rank(R_fit) <= n_fit - 1 (the fit rows sum to zero after
    # centering), so ranks beyond that are undefined and are dropped from the curve.
    sv, Vt = np.linalg.svd(R_fit, full_matrices=False)[1:]
    ev = sv**2
    cum = np.cumsum(ev) / ev.sum()
    max_rank = int(np.sum(sv > sv[0] * 1e-10))
    print(f"fit residual: numerical rank {max_rank} of {min(R_fit.shape)}")

    curve = _rank_curve(Vt, R_val, RANK_GRID)

    # Full rank: the entire fit basis. Note this is NOT 1.0 in general -- the fit basis
    # spans at most n_fit - 1 of the 6,169 gene dimensions, so val components orthogonal to
    # every fit direction are unreachable. The identity ceiling below is the 1.0 reference.
    full = _rank_curve(Vt, R_val, [max_rank])[0]
    identity = per_feature_pearson(R_val, R_val)

    # Basis-choice envelope: V_r from VAL's own residuals. Fully cheating -- it is the best
    # any r-dimensional subspace could do on these strains -- so the gap to the fit-basis
    # curve separates "the rank is too small" from "the fit basis is the wrong one".
    Vt_val = np.linalg.svd(R_val, full_matrices=False)[2]
    curve_val_basis = _rank_curve(Vt_val, R_val, RANK_GRID)

    # Sensitivity: refit mu and the basis on masked_conditioning_oracle.py's TRAIN set only
    # (1172 strains, its 155 tune strains excluded). Nothing in this script is tuned, so the
    # tune strains are legitimately fit data -- this check says whether that choice moved
    # anything, rather than asserting it did not.
    mu_tr = Y[train_only_idx].mean(axis=0, keepdims=True)
    Vt_tr = np.linalg.svd(
        (Y[train_only_idx] - mu_tr).astype(np.float64), full_matrices=False
    )[2]
    curve_train_only = _rank_curve(
        Vt_tr, (Y[val_idx] - mu_tr).astype(np.float64), RANK_GRID
    )

    by_rank_val = {row["rank"]: row for row in curve_val_basis}
    by_rank_tr = {row["rank"]: row for row in curve_train_only}
    print(
        f"\n{'rank':>6}{'ceiling pf':>13}{'frac val var':>14}{'frac of full':>14}"
        f"{'val-basis':>12}{'train-only':>12}"
    )
    for row in curve:
        r = row["rank"]
        vb = by_rank_val.get(r)
        tr = by_rank_tr.get(r)
        print(
            f"{r:>6}{row['ceiling_pearson_per_feature']:>13.4f}"
            f"{row['frac_val_variance_captured']:>14.4f}"
            f"{row['ceiling_pearson_per_feature'] / full['ceiling_pearson_per_feature']:>14.4f}"
            f"{(vb['ceiling_pearson_per_feature'] if vb else float('nan')):>12.4f}"
            f"{(tr['ceiling_pearson_per_feature'] if tr else float('nan')):>12.4f}"
        )
    print(
        f"{'full':>6}{full['ceiling_pearson_per_feature']:>13.4f}"
        f"{full['frac_val_variance_captured']:>14.4f}{1.0:>14.4f}"
        f"   (r={max_rank}, the whole fit basis)"
    )
    print(f"{'ident':>6}{identity:>13.4f}{1.0:>14.4f}   (R_hat = R_val exactly)")
    print(
        "\nsingular spectrum (fit): "
        + "  ".join(f"s{i + 1}={sv[i]:.1f}" for i in [0, 1, 3, 7, 15, 31, 63, 127])
    )
    print(
        "cumulative fit variance: "
        + "  ".join(
            f"r={r}:{100 * cum[r - 1]:.1f}%" for r in RANK_GRID if r <= len(cum)
        )
    )

    # Reference numbers are READ from the result files that measured them, never
    # transcribed, so this JSON cannot drift away from its own citations.
    results_dir = experiment_results_dir("019-simb-multimodal", __file__)
    with open(osp.join(results_dir, "expression_ceiling_replicate.json")) as f:
        replicate = json.load(f)
    with open(osp.join(results_dir, "residual_covariance_diagnostic.json")) as f:
        rcd = json.load(f)
    with open(osp.join(results_dir, "masked_conditioning_oracle.json")) as f:
        mask = json.load(f)

    noise_ceiling = float(replicate["primary_ceiling_mean_sqrt_r"]["ceiling"])
    observed = float(replicate["observed"])
    m1000 = [r for r in mask["results"] if r["m"] == 1000][0]["val_mean"]

    out: dict[str, Any] = {
        "n_strains": int(n_strain),
        "n_genes": int(n_gene),
        "n_val": int(len(val_idx)),
        "n_fit": int(len(fit_idx)),
        "seed": SEED,
        "rank_grid": RANK_GRID,
        "fit_basis_numerical_rank": max_rank,
        "ceiling_train_basis": curve,
        "ceiling_val_basis": curve_val_basis,
        "ceiling_train_only_basis": curve_train_only,
        "full_rank": {
            "rank": max_rank,
            "ceiling_pearson_per_feature": full["ceiling_pearson_per_feature"],
            "frac_val_variance_captured": full["frac_val_variance_captured"],
        },
        "identity_ceiling_pearson_per_feature": identity,
        "singular_values_top": [float(x) for x in sv[: max(RANK_GRID) + 1]],
        "cumulative_variance_fit": {
            str(r): float(cum[r - 1]) for r in RANK_GRID if r <= len(cum)
        },
        "reference": {
            "replicate_noise_ceiling": noise_ceiling,
            "observed_model_pearson_per_feature": observed,
            "masked_oracle_m1000_val_mean": m1000,
            "effective_rank_residual_covariance": rcd["effective_rank"],
        },
    }

    # Knee, stated by a rule rather than by eye: the smallest swept rank reaching 90% / 95%
    # of the full-fit-basis ceiling, and the smallest reaching the replicate noise ceiling
    # (the only threshold at which a rank-r head provably costs nothing measurable).
    def _first_at(threshold: float) -> int | None:
        hits = [
            row["rank"]
            for row in curve
            if row["ceiling_pearson_per_feature"] >= threshold
        ]
        return int(hits[0]) if hits else None

    out["knee"] = {
        "rank_at_90pct_of_full": _first_at(0.90 * full["ceiling_pearson_per_feature"]),
        "rank_at_95pct_of_full": _first_at(0.95 * full["ceiling_pearson_per_feature"]),
        "rank_at_replicate_noise_ceiling": _first_at(noise_ceiling),
        "rank_at_masked_oracle_m1000": _first_at(m1000),
    }
    print(
        f"\nknee: 90% of full at r={out['knee']['rank_at_90pct_of_full']}, "
        f"95% at r={out['knee']['rank_at_95pct_of_full']}, "
        f"replicate ceiling ({noise_ceiling:.4f}) at "
        f"r={out['knee']['rank_at_replicate_noise_ceiling']}"
    )

    path = osp.join(results_dir, "lowrank_output_ceiling.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}")

    print(
        "\nREAD THIS AS: the best per-feature Pearson a rank-r output head could reach if it\n"
        "predicted its r coefficients PERFECTLY. Compare each row against the replicate\n"
        f"noise ceiling {noise_ceiling:.4f} (not against 1.0): a rank whose ceiling clears\n"
        "that costs nothing measurable, because the labels themselves are that noisy. A\n"
        f"curve still climbing past r=128 means the effective rank {rcd['effective_rank']:.1f}\n"
        "overstates how compressible the metric is, since per-feature Pearson weights the\n"
        "low-variance genes in the spectral tail equally."
    )


if __name__ == "__main__":
    main()
