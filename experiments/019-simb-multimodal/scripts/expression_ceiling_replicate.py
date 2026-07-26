# experiments/019-simb-multimodal/scripts/expression_ceiling_replicate.py
# [[experiments.019-simb-multimodal.scripts.expression_ceiling_replicate]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/expression_ceiling_replicate
"""Replicate-based per-gene Pearson ceiling for Kemmeren-2014 expression.

REPLACES the p-value-inversion estimate in `expression_noise_ceiling.py`, which is not
usable: it returns mean ceiling 0.092 while the observed model score is 0.109 (118% of a
"ceiling"), and its per-gene distribution is degenerate (median 0.0000, IQR [0, 0] --
84% of genes clipped to zero reliability) even though 41.6% of that same file's p-values
are <= 0.05. Its noise term inverts a limma *moderated-t* p through a *normal* quantile,
sigma_g = |M| / Phi^-1(1 - p/2), and was never validated (its own docstring defers the
cross-check to a TODO).

This script instead estimates the noise term from ACTUAL REPLICATES, which is the same
construction `morphology_noise_ceiling.py` uses (empirical noise variance from repeated
measurement of one genotype + total variance across all mutants), so the expression and
morphology ceilings become methodologically comparable rather than one being a
distributional inversion and the other a variance decomposition.

Estimand. Our ranked metric `val/expression/pearson_per_feature` is, per reporter gene,
the Pearson across strains between prediction and the *measured* (noisy) log2 ratio. Write
one measurement as x = s + e with signal variance sigma_s^2 and noise variance sigma_e^2.
A PERFECT predictor of s scores

    ceiling_g = corr(s, s + e) = sqrt(sigma_s^2 / (sigma_s^2 + sigma_e^2)) = sqrt(rel_g),

so the ceiling is the square root of the reliability. (Note a test-retest correlation
between two independent measurements estimates rel_g itself, NOT the ceiling -- the
ceiling is its square root and is therefore the LARGER number. Both are reported below.)

Two candidate noise estimates are computed, and one of them is then EMPIRICALLY REJECTED:

  ROUTE A -- REPORTED SE.  Kemmeren's own LMDB carries, per strain x reporter gene,
  `expression_log2_ratio_se` over `n_replicates` (= 2, the dye-swap pair). Taking the
  reported target to be that replicate MEAN gives noise variance se^2.

  ROUTE B -- CROSS-STUDY REPLICATE.  All 82 single deletions in Sameith 2015 (GSE42536)
  are also in Kemmeren (GSE42527/42526) -- independent re-measurement of the same genotype
  in a different study. For the shared reporter genes, var_strains(x - y) = sigma_ex^2 +
  sigma_ey^2, so sigma_e^2 ~= var(x - y) / 2 under comparable per-study noise.

A validation step decides between them: if the reported se were a complete noise model,
observed cross-study disagreement would equal se_kem^2 + se_sam^2. Measured, the ratio is
**0.10** (IQR [0.08, 0.15]) -- the reported se explains TEN TIMES more disagreement than
two independent studies actually show, i.e. it overstates noise ~3x in SD. So ROUTE A is
rejected, and with it the superseded p-value estimate, which recovers the same inflated
sigma (limma's p is computed from that SE). Their degenerate median-0 ceilings and the
"118% of ceiling" violation are one artifact, not two independent findings.

ROUTE B is therefore the primary estimate. Both routes take sigma_total^2 from the FULL
1,484-strain Kemmeren panel and use the paired data only for sigma_e^2. The concern that
the 82 shared deletions (TF/kinase mutants) would be an unrepresentatively high-effect
subset does NOT hold empirically: their median across-strain variance is 0.70x the full
panel's, i.e. slightly LOWER, so if anything Route B is mildly conservative.

Result: ceiling ~0.86, cross-checked assumption-free by sqrt(per-feature test-retest r) =
sqrt(0.611) = 0.78. Observed 0.109 is ~13% of ceiling -- expression is NOT near-saturated.

Run from repo root:
    python experiments/019-simb-multimodal/scripts/expression_ceiling_replicate.py
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

# Best observed val/expression/pearson_per_feature to date, for the realized-fraction
# report. expr_002 controlled 1,161 split; expr_005 (in flight) currently peaks at 0.0795.
OBS = 0.109

KEMMEREN_LMDB = "data/torchcell/microarray_kemmeren2014/processed/lmdb"
SAMEITH_SM_LMDB = "data/torchcell/sm_microarray_sameith2015/processed/lmdb"


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
    names = sorted(p["systematic_gene_name"] for p in perts)
    return "|".join(names)


def _matrix(
    records: list[dict[str, Any]], genes: list[str], field: str
) -> np.ndarray:
    """[n_strains, n_genes] from the per-gene dicts, NaN where a gene is absent."""
    out = np.full((len(records), len(genes)), np.nan, dtype=np.float64)
    for i, rec in enumerate(records):
        d = rec["experiment"]["phenotype"][field]
        for j, g in enumerate(genes):
            v = d.get(g)
            if v is not None:
                out[i, j] = v
    return out


def _summarize(name: str, ceiling: np.ndarray, rel: np.ndarray) -> dict[str, Any]:
    ok = np.isfinite(ceiling)
    c = ceiling[ok]
    stats = {
        "n_genes": int(ok.sum()),
        "mean_ceiling": float(c.mean()),
        "median_ceiling": float(np.median(c)),
        "iqr": [float(np.quantile(c, 0.25)), float(np.quantile(c, 0.75))],
        "mean_reliability": float(np.nanmean(rel[ok])),
        "frac_above": {
            f"{t:.2f}": float((c > t).mean()) for t in (0.05, 0.1, 0.2, 0.3, 0.5)
        },
        "observed": OBS,
        "frac_of_ceiling_realized": float(OBS / c.mean()),
    }
    print(f"\n== {name} (n={stats['n_genes']} genes) ==")
    print(f"  mean ceiling  : {stats['mean_ceiling']:.4f}")
    print(
        f"  median ceiling: {stats['median_ceiling']:.4f}"
        f"   IQR [{stats['iqr'][0]:.3f}, {stats['iqr'][1]:.3f}]"
    )
    print(f"  mean reliability: {stats['mean_reliability']:.4f}")
    for t, f in stats["frac_above"].items():
        print(f"    genes with ceiling > {t}: {100 * f:5.1f}%")
    print(
        f"  observed {OBS:.3f} -> {100 * stats['frac_of_ceiling_realized']:.1f}% of ceiling"
    )
    return stats


def main() -> None:
    load_dotenv("/home/michaelvolk/Documents/projects/torchcell/.env")
    data_root = os.environ["DATA_ROOT"]

    kem = _load_lmdb(osp.join(data_root, KEMMEREN_LMDB))
    sam = _load_lmdb(osp.join(data_root, SAMEITH_SM_LMDB))
    print(f"Kemmeren strains: {len(kem)}   Sameith single-mutant strains: {len(sam)}")

    kem_genes = set(kem[0]["experiment"]["phenotype"]["expression_log2_ratio"])
    sam_genes = set(sam[0]["experiment"]["phenotype"]["expression_log2_ratio"])
    genes = sorted(kem_genes & sam_genes)
    print(f"reporter genes: kemmeren={len(kem_genes)} sameith={len(sam_genes)} shared={len(genes)}")

    M_kem = _matrix(kem, genes, "expression_log2_ratio")
    SE_kem = _matrix(kem, genes, "expression_log2_ratio_se")
    n_rep = _matrix(kem, genes, "n_replicates")
    print(f"kemmeren n_replicates: unique={np.unique(n_rep[np.isfinite(n_rep)])[:5]}")

    # sigma_total^2 across the FULL panel -- shared by both routes.
    total_var = np.nanvar(M_kem, axis=0, ddof=1)

    results: dict[str, Any] = {
        "n_strains_kemmeren": len(kem),
        "n_strains_sameith_sm": len(sam),
        "n_reporter_genes": len(genes),
        "observed": OBS,
    }

    # ---- ROUTE A: within-study technical noise (se of the reported replicate mean) ----
    noise_var_a = np.nanmean(SE_kem**2, axis=0)
    rel_a = np.clip(1.0 - noise_var_a / total_var, 0.0, 1.0)
    ceil_a = np.sqrt(rel_a)
    results["route_a_within_study"] = _summarize(
        "ROUTE A -- reported-SE noise (INVALIDATED, see validation)", ceil_a, rel_a
    )

    # ---- ROUTE B: cross-study Kemmeren vs Sameith on the shared deletions ----
    kem_by_del = {_deletion(r): i for i, r in enumerate(kem)}
    pairs = [(kem_by_del[d], i) for i, r in enumerate(sam) if (d := _deletion(r)) in kem_by_del]
    print(f"\nshared single deletions (Kemmeren AND Sameith): {len(pairs)}")

    X = M_kem[[k for k, _ in pairs], :]
    M_sam = _matrix(sam, genes, "expression_log2_ratio")
    Y = M_sam[[s for _, s in pairs], :]

    # Per-STRAIN correlation across genes (the "profiles agree" number, per-instance).
    per_instance = np.array(
        [
            np.corrcoef(X[i][m], Y[i][m])[0, 1]
            for i in range(X.shape[0])
            if (m := np.isfinite(X[i]) & np.isfinite(Y[i])).sum() > 2
        ]
    )
    # Per-GENE correlation across strains (test-retest reliability, the metric's axis).
    per_feature = np.array(
        [
            np.corrcoef(X[:, j][m], Y[:, j][m])[0, 1]
            for j in range(X.shape[1])
            if (m := np.isfinite(X[:, j]) & np.isfinite(Y[:, j])).sum() > 2
        ]
    )
    print(
        f"  per-INSTANCE  (across genes, per strain): mean r={np.nanmean(per_instance):.4f} "
        f"median={np.nanmedian(per_instance):.4f}"
    )
    print(
        f"  per-FEATURE   (across strains, per gene): mean r={np.nanmean(per_feature):.4f} "
        f"median={np.nanmedian(per_feature):.4f}"
    )
    results["cross_study_test_retest"] = {
        "n_shared_deletions": len(pairs),
        "per_instance_mean_r": float(np.nanmean(per_instance)),
        "per_instance_median_r": float(np.nanmedian(per_instance)),
        "per_feature_mean_r": float(np.nanmean(per_feature)),
        "per_feature_median_r": float(np.nanmedian(per_feature)),
    }

    # ---- VALIDATION: do the reported SEs explain the observed cross-study disagreement? ----
    # If Kemmeren's own se is a complete noise model, then for the shared strains
    #     var(X - Y) should equal  se_kem^2 + se_sam^2.
    # A ratio >> 1 means the reported (dye-swap technical) se UNDERSTATES real measurement
    # noise, which is exactly the failure mode that would make ROUTE A's ceiling too low.
    SE_sam = _matrix(sam, genes, "expression_log2_ratio_se")
    se2_pair = (
        SE_kem[[k for k, _ in pairs], :] ** 2 + SE_sam[[s for _, s in pairs], :] ** 2
    )
    observed_disagree = np.nanvar(X - Y, axis=0, ddof=1)
    predicted_disagree = np.nanmean(se2_pair, axis=0)
    ratio = observed_disagree / predicted_disagree
    ok_r = np.isfinite(ratio) & (predicted_disagree > 0)
    print("\n== VALIDATION: observed cross-study disagreement / SE-predicted ==")
    print(
        f"  median ratio: {np.nanmedian(ratio[ok_r]):.2f}   "
        f"IQR [{np.nanquantile(ratio[ok_r], 0.25):.2f}, {np.nanquantile(ratio[ok_r], 0.75):.2f}]"
    )
    print("  (1.0 = reported se fully explains real noise; >1 = se understates it)")
    results["se_validation"] = {
        "median_ratio_observed_over_predicted": float(np.nanmedian(ratio[ok_r])),
        "iqr": [
            float(np.nanquantile(ratio[ok_r], 0.25)),
            float(np.nanquantile(ratio[ok_r], 0.75)),
        ],
    }

    # Effect-size context: the 82 shared deletions are TF/kinase mutants. Quantify how
    # atypical their across-strain spread is, since reliability scales with signal size.
    var82 = np.nanvar(X, axis=0, ddof=1)
    print(
        f"\n  across-strain variance, 82 shared vs full 1,484 panel: "
        f"median ratio {np.nanmedian(var82 / total_var):.2f}x"
    )
    results["shared_subset_variance_ratio"] = float(np.nanmedian(var82 / total_var))

    noise_var_b = np.nanvar(X - Y, axis=0, ddof=1) / 2.0
    rel_b = np.clip(1.0 - noise_var_b / total_var, 0.0, 1.0)
    ceil_b = np.sqrt(rel_b)
    results["route_b_cross_study"] = _summarize(
        "ROUTE B -- cross-study replicate noise (PRIMARY)", ceil_b, rel_b
    )

    # ---- Verdict ----------------------------------------------------------------
    # Route A is INVALIDATED by the SE-validation check above: the reported se explains
    # ~10x more disagreement than two independent studies actually show, i.e. it overstates
    # noise by ~3x in SD. That inflated noise term is also what the superseded p-value
    # inversion recovers (limma's p is computed from the same SE), so 0.092 and Route A's
    # 0.061 are the SAME artifact, not two independent estimates.
    #
    # Route B is the primary estimate: its noise term is measured directly from independent
    # re-measurement, and the shared deletions turn out to be representative in spread
    # (median across-strain variance 0.70x the full panel), so it is not a high-effect
    # subset as feared. Cross-checked by the assumption-free per-feature test-retest r:
    # ceiling = sqrt(r) = sqrt(0.611) = 0.78, consistent with Route B's 0.86.
    primary = results["route_b_cross_study"]["mean_ceiling"]
    xcheck = float(np.sqrt(np.nanmean(per_feature)))
    print("\n" + "=" * 68)
    print(f"EXPRESSION per-feature ceiling (replicate-based): {primary:.3f}")
    print(f"  cross-check sqrt(test-retest r) = sqrt({np.nanmean(per_feature):.3f}) = {xcheck:.3f}")
    print(f"  observed {OBS:.3f} -> {100 * OBS / primary:.0f}% of ceiling")
    print("  SUPERSEDES 0.092 (p-value inversion) and Route A 0.061 -- both use an")
    print("  SE that overstates noise ~10x in variance, and both are VIOLATED by 0.109.")
    print("=" * 68)
    results["verdict"] = {
        "primary_ceiling": primary,
        "cross_check_sqrt_test_retest": xcheck,
        "superseded_estimates": {"p_value_inversion": 0.092, "route_a_reported_se": hi_a}
        if (hi_a := results["route_a_within_study"]["mean_ceiling"])
        else {},
    }

    out = osp.join(
        os.environ["EXPERIMENT_ROOT"],
        "019-simb-multimodal/results/expression_ceiling_replicate.json",
    )
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
