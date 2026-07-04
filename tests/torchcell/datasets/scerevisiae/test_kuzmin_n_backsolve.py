"""Tripwire: Kuzmin combined-mutant n_samples must stay consistent with the data.

`N_SAMPLES_COMBINED_MUTANT` (kuzmin2018/2020) is a DERIVED value, not one stated
per-record in the SI. It was fixed by back-solving against the reported interaction
p-values (the value whose single-term normal model reproduces the reported p-value
median), corroborated by the conservative lower-end of the Baryshnikova 4-8 range.
See ``[[torchcell.datasets.scerevisiae.costanzo2016.noise-computation]]``.

This test re-runs that back-solve so the derivation is reproducible AND tripwired:
if the raw data or the recorded constant drifts such that a different n best matches
the reported p-values, this fails and a human re-decides. The back-solve is bespoke
to Kuzmin (it relies on the reported P-value column + a single-term normal model),
so it lives here, in a test -- NOT in the dataset build's critical path, which uses
the value directly. Skips when the raw TSV is absent (it is not shipped to CI).
"""

import os

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

# Candidate raw locations (any Kuzmin-2018 dataset root holds the same Data S1 TSV).
_RAW_CANDIDATES = [
    "data/torchcell/dmf_kuzmin2018/raw/aao1729_data_s1.tsv",
    "data/torchcell/smf_kuzmin2018/raw/aao1729_data_s1.tsv",
    "data/torchcell/tmf_kuzmin2018/raw/aao1729_data_s1.tsv",
]
_CANDIDATE_N = (2, 4, 6, 8)


def _raw_path() -> str | None:
    """First existing raw Data S1 TSV, or None (data not shipped to CI)."""
    for p in _RAW_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def test_combined_mutant_n_matches_reported_pvalue() -> None:
    path = _raw_path()
    if path is None:
        pytest.skip("Kuzmin raw Data S1 TSV not present (not shipped to CI)")

    # Imported here, after the skip gate: importing the dataset module pulls in
    # DATA_ROOT-dependent code, which is unset in CI (where the raw data is also
    # absent, so we skip above before reaching this).
    from torchcell.datasets.scerevisiae.kuzmin2018 import N_SAMPLES_COMBINED_MUTANT

    df = pd.read_csv(path, sep="\t")
    d = df[df["Combined mutant type"] == "digenic"]
    eps = d["Raw genetic interaction score (epsilon)"].to_numpy(dtype=float)
    sd = d["Combined mutant fitness standard deviation"].to_numpy(dtype=float)
    pv = d["P-value"].to_numpy(dtype=float)
    mask = np.isfinite(eps) & np.isfinite(sd) & np.isfinite(pv) & (sd > 0) & (pv > 0)
    eps, sd, pv = eps[mask], sd[mask], pv[mask]
    assert mask.sum() > 10_000, "expected the full digenic table; got too few rows"

    # Effective n = the one whose single-term normal model p-median matches the
    # reported p-median: p(n) = 2 * Phi(-|eps| / (sd / sqrt(n))).
    reported_median = float(np.median(pv))
    gap = {
        n: abs(
            float(np.median(2 * norm.cdf(-np.abs(eps) / (sd / np.sqrt(n)))))
            - reported_median
        )
        for n in _CANDIDATE_N
    }
    best_n = min(gap, key=lambda k: gap[k])
    assert best_n == N_SAMPLES_COMBINED_MUTANT, (
        f"back-solve best n={best_n} != recorded {N_SAMPLES_COMBINED_MUTANT}; "
        f"reported p median={reported_median:.3f}, per-n |median gap|={gap}"
    )

    # The stored SD column must drive the p-value ranking (scale-invariant sanity).
    p_pred = 2 * norm.cdf(-np.abs(eps) / (sd / np.sqrt(N_SAMPLES_COMBINED_MUTANT)))
    spearman = pd.Series(p_pred).corr(pd.Series(pv), method="spearman")
    assert spearman > 0.9, f"col-12 SD should rank p-values; spearman={spearman:.3f}"
