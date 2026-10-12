# experiments/027-betaxanthin-metabolic/scripts/cbm_scoring.py
# [[experiments.027-betaxanthin-metabolic.cpu-baselines-2026-09-14]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/027-betaxanthin-metabolic/scripts/cbm_scoring
"""The one scoring rule every CPU baseline in this experiment is read through.

A bare Spearman at n ~ 630 hides its own width: the null standard error is
``1 / sqrt(n - 1) = 0.0399``, so a correlation of +0.04 is one null width from zero and is
not evidence of anything. Every readout therefore arrives with a permutation p, a bootstrap
95 % interval, the top-k precision against a stated base rate, and the fraction of genotypes
sitting at the modal value, because a constraint-based readout is heavily tied and a rank
statistic over a tied vector is not what it looks like.

``TOP_FRAC`` is 0.15625 = 100/640, the share of Merzbacher's released betaxanthin test genes
labelled high producers, so top-k precision is read against the same base rate the 020
head-to-head uses rather than an invented one.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import spearmanr

#: High-producer share, 100 of Merzbacher's 640 scored genes.
TOP_FRAC = 100.0 / 640.0
#: Top-k cut-offs. 50 resolves best at n = 3 seeds (review section 5a); 10 is quantized in
#: steps of 0.1 and is reported for continuity with the 020 figures, not as a decider.
TOP_K = (10, 25, 50)
N_PERM = 2000
N_BOOT = 2000


def frac_at_mode(x: np.ndarray) -> float:
    """Share of finite entries sitting at the single most common value, rounded to 1e-9."""
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return float("nan")
    _, inverse = np.unique(np.round(finite, 9), return_inverse=True)
    return float(np.bincount(inverse).max() / finite.size)


def topk_precision(pred: np.ndarray, true: np.ndarray, k: int, thr: float) -> float:
    """Share of the k highest-ranked genes that measured in the top ``TOP_FRAC``."""
    if pred.size < k:
        return float("nan")
    order = np.argsort(-pred)[:k]
    return float((true[order] >= thr).mean())


def assess(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    n_perm: int = N_PERM,
    n_boot: int = N_BOOT,
) -> dict[str, Any]:
    """Spearman + permutation p + bootstrap CI + top-k precision + tie fraction.

    ``x`` is the readout, ``y`` the measured value; entries where either is not finite are
    dropped and the surviving count is reported as ``n``.
    """
    finite = np.isfinite(x) & np.isfinite(y)
    xs, ys = np.asarray(x, float)[finite], np.asarray(y, float)[finite]
    out: dict[str, Any] = {
        "n": int(xs.size),
        "n_distinct_values": int(np.unique(np.round(xs, 9)).size),
        "frac_at_mode": frac_at_mode(xs),
        "min": float(np.min(xs)) if xs.size else float("nan"),
        "max": float(np.max(xs)) if xs.size else float("nan"),
    }
    if xs.size < 10 or np.unique(np.round(xs, 9)).size < 2 or ys.std() < 1e-12:
        out.update(
            spearman=None,
            permutation_p=None,
            bootstrap_ci95=None,
            topk_precision=None,
            base_rate=None,
            note="degenerate readout: constant, or fewer than 10 scorable genes",
        )
        return out
    rho = float(spearmanr(xs, ys).statistic)
    thr = float(np.quantile(ys, 1.0 - TOP_FRAC))
    out.update(
        spearman=rho,
        topk_precision={str(k): topk_precision(xs, ys, k, thr) for k in TOP_K},
        base_rate=float((ys >= thr).mean()),
        high_threshold=thr,
    )
    # `n_perm=0` / `n_boot=0` is how a hyperparameter SWEEP asks for the point estimate
    # only; the selected cell is then re-scored with both. A p or an interval is never
    # reported as absent when it was merely skipped: the field is None and says so.
    if n_perm:
        null = np.array(
            [spearmanr(xs, rng.permutation(ys)).statistic for _ in range(n_perm)]
        )
        out["permutation_p"] = float((np.abs(null) >= abs(rho)).mean())
        out["permutation_null_sd"] = float(null.std())
    else:
        out["permutation_p"] = None
    if n_boot:
        boot = []
        for _ in range(n_boot):
            idx = rng.integers(0, xs.size, xs.size)
            if np.unique(np.round(xs[idx], 9)).size > 1 and ys[idx].std() > 1e-12:
                boot.append(spearmanr(xs[idx], ys[idx]).statistic)
        out["bootstrap_ci95"] = (
            [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
            if boot
            else None
        )
        out["n_bootstrap_used"] = len(boot)
    else:
        out["bootstrap_ci95"] = None
    return out


def assess_per_feature(
    pred: np.ndarray, true: np.ndarray, columns: list[str], rng: np.random.Generator
) -> dict[str, Any]:
    """Per-column assessment for a width-F target, plus the mean over columns.

    The columns are not measured on the same strains, so ``n_per_feature`` sits beside the
    mean: an average over per-column Spearmans whose sample sizes differ by an order of
    magnitude is not an average over one set of genotypes.
    """
    per: dict[str, Any] = {}
    rhos: list[float] = []
    for f, name in enumerate(columns):
        cell = assess(pred[:, f], true[:, f], rng)
        per[name] = cell
        rhos.append(0.0 if cell["spearman"] is None else float(cell["spearman"]))
    return {
        "mean_spearman": float(np.mean(rhos)),
        "n_features": len(columns),
        "n_per_feature": [per[c]["n"] for c in columns],
        "per_feature": per,
    }
