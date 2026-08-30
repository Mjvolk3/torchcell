# experiments/019-simb-multimodal/scripts/conditioning_gene_budget.py
# [[experiments.019-simb-multimodal.scripts.conditioning_gene_budget]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/conditioning_gene_budget
"""HOW FEW genes must be measured for the conditioning oracle to work, and does CHOOSING
them beat drawing them at random?

THE QUESTION THIS ANSWERS. `masked_conditioning_oracle.py` and
`cross_study_conditioning_oracle.py` evaluated the conditional-mean oracle at
m = 10 / 100 / 1000 observed genes, all drawn UNIFORMLY AT RANDOM. The headline was taken
at m = 1000, and measuring 1000 transcripts is not a cheap capability, so the reported
number is a statement about an expensive instrument. Two things were never measured:

  1. the SHAPE of the cheap end. Between m = 10 and m = 100 there is a single interpolated
     segment, and m < 10 was never evaluated at all, so "how few genes are enough" has no
     answer in the existing artifacts.
  2. whether a CHOSEN observed set beats a random one at the same m. The oracle's leverage
     comes from a residual covariance whose variance is concentrated in ~33 effective
     components (`residual_covariance_diagnostic.json`: effective rank 32.78, rank-32
     subspace = 59.1% of residual variance). A random draw of m genes spends most of its
     budget on genes that are nearly redundant with each other; a set chosen to SPAN that
     subspace should not.

ESTIMATOR (unchanged from the two prior oracles, so the numbers are comparable).
Residual R = Y - mu with mu the per-gene mean over TRAIN strains only. Given observed set
M and unobserved set U,

    E[R_U | R_M] = Sigma_UM (Sigma_MM + lam I)^{-1} R_M ,

Sigma estimated on TRAIN strains only, lam selected on a TUNE split of the training
strains and then applied unchanged to every evaluation set and every arm. Metric is
`per_feature_pearson`: correlate across strains within each held-out gene, then average
over held-out genes.

SELECTION RULES (all computed from TRAIN residuals only -- no evaluation strain and no
Sameith value ever enters the choice of which genes to observe).

  random      m genes drawn uniformly without replacement, 5 independent draws. This is
              the rule the published oracle used, and the control the other rules must beat.
  variance    the m genes with the largest across-strain variance of the train residual.
              The cheapest possible non-random rule, and the one a practitioner would try
              first: measure the genes that move.
  qr_leverage column-pivoted QR on the top-r right singular vectors of the train residual
              (r = max m), i.e. classical column-subset selection / sensor placement. It
              picks genes that SPAN the dominant residual subspace rather than genes that
              are individually loud, so it should avoid the redundancy a variance rule
              walks into. The pivot order is nested, so the m = 5 set is a subset of the
              m = 50 set, which is what makes it usable as a fixed assay panel.

NOT RUN, and named rather than approximated: selection by EASE OF MEASUREMENT (which
transcripts or metabolites a given instrument reads cheaply). No measurement-cost
annotation exists in the repo, so any such ranking would be invented. Also not run: greedy
forward selection scored on the tune split, which is O(F) solves per added gene and does
not fit the budget of this pass.

THREE EVALUATION SETS, reported side by side because they answer different questions.

  val_kem_155       155 held-out Kemmeren strains. Within-study: observed and held-out
                    genes come from the SAME array for the same strain, so this number
                    includes any conditioning power carried by shared measurement state.
                    It is the number that exceeds the replicate ceiling of 0.775.
  within_kem_82     the 82 strains that Kemmeren and Sameith both measured, scored
                    within Kemmeren. Matched-n control for the cross arms.
  cross_kem_to_sam  observe those 82 strains in Kemmeren, predict them as measured in
                    Sameith. Array-level technical state is independent between the two
                    measurements while the biology is shared, so this is the honest form of
                    the result. `cross_sam_to_kem` is the reverse direction.

The cross arms' ceiling is NOT 1.0: it is `xstudy_agreement`, the per-feature Pearson
between the two studies' own measurements of the same held-out genes on the same 82
strains, computed here per draw on the identical gene set.

NULL. For every arm and every observed set, a permuted-strain null: the strain
correspondence between input and target is destroyed while every marginal is preserved.
At n = 82 the chance level is not exactly 0 and is measured rather than assumed.

Run from repo root:
    python experiments/019-simb-multimodal/scripts/conditioning_gene_budget.py

Writes ``results/conditioning_gene_budget.json``.
"""

from __future__ import annotations

import json
import os
import os.path as osp
import pickle
from typing import Any

import lmdb
import numpy as np
import scipy.linalg
from dotenv import load_dotenv

_WT_ENV = osp.abspath(osp.join(osp.dirname(__file__), "..", "..", "..", ".env"))
load_dotenv(
    _WT_ENV
    if osp.exists(_WT_ENV)
    else osp.expanduser("~/Documents/projects/torchcell/.env")
)

from torchcell.utils.paths import experiment_results_dir  # noqa: E402

# Raw per-study LMDBs, the same two `expression_ceiling_replicate.py` and
# `cross_study_conditioning_oracle.py` pair. NOT the fig3_core build, which is already
# merged across studies and would collapse the two measurements of a shared strain.
KEMMEREN_LMDB = "data/torchcell/microarray_kemmeren2014/processed/lmdb"
SAMEITH_SM_LMDB = "data/torchcell/sm_microarray_sameith2015/processed/lmdb"

SEED = 0
M_GRID = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
N_DRAWS_RANDOM = 5
RIDGE_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
N_TUNE = 155
N_VAL_KEM = 155
RULES = ("random", "variance", "qr_leverage")
ARMS = ("val_kem_155", "within_kem_82", "cross_kem_to_sam", "cross_sam_to_kem")

# Reference values measured elsewhere, quoted so the JSON is self-contained.
REPLICATE_CEILING = 0.775  # expression_ceiling_replicate.json, mean_g sqrt(r_g)
WITHIN_STUDY_ORACLE_PUBLISHED = {"10": 0.4084, "100": 0.6756, "1000": 0.7932}
CROSS_STUDY_ORACLE_PUBLISHED = {"10": 0.2335, "100": 0.3815, "1000": 0.4838}


def _load_lmdb(path: str) -> list[dict[str, Any]]:
    env = lmdb.open(path, readonly=True, lock=False, subdir=True)
    records: list[dict[str, Any]] = []
    with env.begin() as txn:
        for _, value in txn.cursor():
            records.append(pickle.loads(value))
    env.close()
    return records


def _deletion(record: dict[str, Any]) -> str:
    perts = record["experiment"]["genotype"]["perturbations"]
    return "|".join(sorted(p["systematic_gene_name"] for p in perts))


def _matrix(records: list[dict[str, Any]], genes: list[str], field: str) -> np.ndarray:
    out = np.full((len(records), len(genes)), np.nan, dtype=np.float64)
    for i, rec in enumerate(records):
        d = rec["experiment"]["phenotype"][field]
        for j, g in enumerate(genes):
            v = d.get(g)
            if v is not None:
                out[i, j] = v
    return out


def per_feature_pearson(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean over FEATURES of the across-STRAIN correlation.

    Copied verbatim from masked_conditioning_oracle.py so this script's numbers are the
    same statistic computed the same way.
    """
    p = pred - pred.mean(axis=0, keepdims=True)
    t = true - true.mean(axis=0, keepdims=True)
    num = (p * t).sum(axis=0)
    den = np.sqrt((p**2).sum(axis=0)) * np.sqrt((t**2).sum(axis=0))
    r = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)
    return float(r.mean())


class Conditioner:
    """Sigma blocks for one observed set, cached across the ridge grid.

    The two prior scripts recomputed Sigma_MM and Sigma_UM inside the lam loop. Same
    estimator, but the blocks do not depend on lam, so they are formed once here. That is
    what makes a 10-point m grid times three selection rules affordable.
    """

    def __init__(self, R_fit: np.ndarray, obs: np.ndarray, unobs: np.ndarray) -> None:
        """Form Sigma_MM and Sigma_UM from the fitting residuals for one observed set."""
        n = R_fit.shape[0]
        A = R_fit[:, obs]
        B = R_fit[:, unobs]
        self.s_mm = (A.T @ A) / (n - 1)
        self.s_um = (B.T @ A) / (n - 1)
        self.obs = obs
        self.unobs = unobs

    def predict(self, R_eval: np.ndarray, lam: float) -> np.ndarray:
        """E[R_U | R_M] on the evaluation residuals at ridge ``lam``."""
        s = self.s_mm.copy()
        s.flat[:: s.shape[0] + 1] += lam
        w = np.linalg.solve(s, R_eval[:, self.obs].T)
        return (self.s_um @ w).T


def _select(rule: str, m: int, R_tr: np.ndarray, pivots: np.ndarray, rng) -> np.ndarray:
    """Observed gene indices for one rule at one m. TRAIN residuals only."""
    if rule == "random":
        return rng.choice(R_tr.shape[1], size=m, replace=False)
    if rule == "variance":
        return np.argsort(-R_tr.var(axis=0))[:m]
    if rule == "qr_leverage":
        return pivots[:m]
    raise ValueError(rule)


def main() -> None:
    data_root = os.environ["DATA_ROOT"]
    rng = np.random.default_rng(SEED)

    kem = _load_lmdb(osp.join(data_root, KEMMEREN_LMDB))
    sam = _load_lmdb(osp.join(data_root, SAMEITH_SM_LMDB))
    kem_genes = set(kem[0]["experiment"]["phenotype"]["expression_log2_ratio"])
    sam_genes = set(sam[0]["experiment"]["phenotype"]["expression_log2_ratio"])
    genes = sorted(kem_genes & sam_genes)
    print(f"Kemmeren strains {len(kem)}  Sameith SM strains {len(sam)}")

    kem_by_del = {_deletion(r): i for i, r in enumerate(kem)}
    pairs = [
        (kem_by_del[d], i)
        for i, r in enumerate(sam)
        if (d := _deletion(r)) in kem_by_del
    ]
    kem_eval_rows = [k for k, _ in pairs]
    sam_eval_rows = [s for _, s in pairs]
    n_eval = len(pairs)

    Y_kem = _matrix(kem, genes, "expression_log2_ratio")
    Y_sam = _matrix(sam, genes, "expression_log2_ratio")
    keep = np.isfinite(Y_kem).all(axis=0) & np.isfinite(Y_sam[sam_eval_rows]).all(
        axis=0
    )
    genes = [g for g, k in zip(genes, keep) if k]
    Y_kem = Y_kem[:, keep]
    Y_sam = Y_sam[:, keep]
    n_gene = len(genes)
    print(f"complete-case reporter genes {n_gene}   shared deletions {n_eval}")
    assert n_gene > max(M_GRID) + 100

    # Fit pool = Kemmeren strains that are NOT one of the shared 82; from it, a tune split
    # for lam and a within-study validation split, leaving the rest to estimate Sigma.
    eval_set = set(kem_eval_rows)
    fit_pool = np.array([i for i in range(len(kem)) if i not in eval_set])
    perm = rng.permutation(len(fit_pool))
    tune_idx = fit_pool[perm[:N_TUNE]]
    val_idx = fit_pool[perm[N_TUNE : N_TUNE + N_VAL_KEM]]
    train_idx = fit_pool[perm[N_TUNE + N_VAL_KEM :]]
    print(
        f"split: train={len(train_idx)} tune={len(tune_idx)} "
        f"val_kem={len(val_idx)} shared_eval={n_eval}"
    )

    mu = Y_kem[train_idx].mean(axis=0, keepdims=True)
    R_tr = Y_kem[train_idx] - mu
    R_tu = Y_kem[tune_idx] - mu
    R_va = Y_kem[val_idx] - mu
    R_kem82 = Y_kem[kem_eval_rows] - mu
    R_sam82 = Y_sam[sam_eval_rows] - mu

    # QR pivot order, computed ONCE on the train residual's dominant right singular
    # subspace. r = max(M_GRID) pivots, nested by construction.
    r = max(M_GRID)
    print(
        f"computing QR-leverage pivot order over the top {r} right singular vectors..."
    )
    _, _, vt = np.linalg.svd(R_tr, full_matrices=False)
    _, _, piv = scipy.linalg.qr(vt[:r], pivoting=True, mode="economic")
    pivots = np.asarray(piv[:r])

    # Input study -> target study for each arm; Sigma always comes from R_tr.
    arm_pairs = {
        "val_kem_155": (R_va, R_va),
        "within_kem_82": (R_kem82, R_kem82),
        "cross_kem_to_sam": (R_kem82, R_sam82),
        "cross_sam_to_kem": (R_sam82, R_kem82),
    }

    out: dict[str, Any] = {
        "n_strains_kemmeren": len(kem),
        "n_strains_sameith_sm": len(sam),
        "n_genes_complete_case": n_gene,
        "n_train": int(len(train_idx)),
        "n_tune": int(len(tune_idx)),
        "n_val_kem": int(len(val_idx)),
        "n_shared_deletions_eval": n_eval,
        "seed": SEED,
        "m_grid": M_GRID,
        "n_draws_random": N_DRAWS_RANDOM,
        "ridge_grid": RIDGE_GRID,
        "rules": list(RULES),
        "arms": list(ARMS),
        "reference_replicate_ceiling_mean_sqrt_r": REPLICATE_CEILING,
        "reference_within_study_oracle_published": WITHIN_STUDY_ORACLE_PUBLISHED,
        "reference_cross_study_oracle_published": CROSS_STUDY_ORACLE_PUBLISHED,
        "results": [],
    }

    draw_rng = np.random.default_rng(SEED + 1)
    for rule in RULES:
        n_draws = N_DRAWS_RANDOM if rule == "random" else 1
        for m in M_GRID:
            scores: dict[str, list[float]] = {a: [] for a in ARMS}
            nulls: dict[str, list[float]] = {a: [] for a in ARMS}
            agree: list[float] = []
            lams: list[float] = []

            for _draw in range(n_draws):
                obs = _select(rule, m, R_tr, pivots, draw_rng)
                mask = np.ones(n_gene, dtype=bool)
                mask[obs] = False
                unobs = np.flatnonzero(mask)
                cond = Conditioner(R_tr, obs, unobs)

                best_lam, best = RIDGE_GRID[0], -np.inf
                for lam in RIDGE_GRID:
                    sc = per_feature_pearson(cond.predict(R_tu, lam), R_tu[:, unobs])
                    if sc > best:
                        best_lam, best = lam, sc
                lams.append(best_lam)

                shuf = draw_rng.permutation(n_eval)
                shuf_val = draw_rng.permutation(len(val_idx))
                for arm, (R_in, R_out) in arm_pairs.items():
                    scores[arm].append(
                        per_feature_pearson(
                            cond.predict(R_in, best_lam), R_out[:, unobs]
                        )
                    )
                    p = shuf_val if arm == "val_kem_155" else shuf
                    nulls[arm].append(
                        per_feature_pearson(
                            cond.predict(R_in[p], best_lam), R_out[:, unobs]
                        )
                    )
                agree.append(per_feature_pearson(R_kem82[:, unobs], R_sam82[:, unobs]))

            def _stat(v: list[float]) -> dict[str, Any]:
                a = np.array(v)
                return {
                    "mean": float(a.mean()),
                    "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
                    "per_draw": [float(x) for x in a],
                }

            out["results"].append(
                {
                    "rule": rule,
                    "m": int(m),
                    "n_draws": n_draws,
                    "n_held_out_genes": int(n_gene - m),
                    "lam": [float(x) for x in lams],
                    "arms": {a: _stat(scores[a]) for a in ARMS},
                    "arms_permuted_strain_null": {a: _stat(nulls[a]) for a in ARMS},
                    "xstudy_agreement_ceiling": _stat(agree),
                }
            )
            print(
                f"{rule:<12} m={m:>5}  "
                + "  ".join(f"{a}={np.mean(scores[a]):+.4f}" for a in ARMS)
                + f"  lam={sorted(set(lams))}"
            )

    # Chosen vs random at matched m, on the arm the honest claim is made on.
    by = {(r_["rule"], r_["m"]): r_ for r_ in out["results"]}
    advantage: dict[str, Any] = {}
    for arm in ARMS:
        advantage[arm] = {
            rule: {
                str(m): float(
                    by[(rule, m)]["arms"][arm]["mean"]
                    - by[("random", m)]["arms"][arm]["mean"]
                )
                for m in M_GRID
            }
            for rule in RULES
            if rule != "random"
        }
    out["chosen_minus_random"] = advantage

    # How few genes are enough: the smallest m on the grid reaching a fraction of the
    # m = 1000 random value, per rule, per arm.
    budgets: dict[str, Any] = {}
    for arm in ARMS:
        ref = by[("random", 1000)]["arms"][arm]["mean"]
        budgets[arm] = {"reference_random_m1000": float(ref)}
        for frac in (0.5, 0.75, 0.9):
            budgets[arm][f"m_to_reach_{int(frac * 100)}pct"] = {
                rule: next(
                    (
                        m
                        for m in M_GRID
                        if by[(rule, m)]["arms"][arm]["mean"] >= frac * ref
                    ),
                    None,
                )
                for rule in RULES
            }
    out["gene_budget"] = budgets

    path = osp.join(
        experiment_results_dir("019-simb-multimodal", __file__),
        "conditioning_gene_budget.json",
    )
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}")

    print("\n=== genes needed to reach a fraction of the m=1000 random score ===")
    for arm in ARMS:
        b = budgets[arm]
        print(f"{arm:<18} ref={b['reference_random_m1000']:.4f}", end="  ")
        for frac in (50, 75, 90):
            print(f"{frac}%: {b[f'm_to_reach_{frac}pct']}", end="  ")
        print()


if __name__ == "__main__":
    main()
