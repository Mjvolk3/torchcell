# experiments/019-simb-multimodal/scripts/expression_noise_ceiling.py
# [[experiments.019-simb-multimodal.scripts.expression_noise_ceiling]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/expression_noise_ceiling
"""Estimate the ACHIEVABLE per-gene Pearson ceiling for Kemmeren-2014 expression.

Parallel to morphology_noise_ceiling.py but for the deleteome. Each mutant gives one
microarray profile of log2(mutant/wt) M-values over ~6,123 reporters. We need per-gene
NOISE variance (wild-type variability) to split observed variance into signal + noise.

The deleteome '*_ex_wt_var_controls.txt' EXCLUDES the WT-vs-WT control columns, but every
(gene, mutant) cell carries a p_value computed from that gene's WT variability. For a
gene tested against its WT noise SD sigma_g, a two-sided test gives
    p = 2 * (1 - Phi(|M| / sigma_g))   ->   sigma_g = |M| / Phi^{-1}(1 - p/2).
Because z := Phi^{-1}(1-p/2) = |M|/sigma_g holds for EVERY mutant (hit or null), the ratio
|M|/z estimates sigma_g regardless of effect size; we take the median over mutants per gene
for robustness. (The deleteome uses limma moderated-t; with ~28 WT replicates t≈z, so this
is a close, slightly-shrunk approximation — cross-check vs the GEO WT channels is a TODO.)

Then, as in the morphology analysis:
    reliability_g = 1 - sigma_g^2 / total_var_g   (signal_var / total_var across mutants)
    ceiling_g     = sqrt(reliability_g)           (max across-strain Pearson vs noisy target)
Our metric val/per_gene/pearson_per_gene = mean_g ceiling_g. Compare to observed ~0.11.

Run from repo root:  python experiments/019-simb-multimodal/scripts/expression_noise_ceiling.py
"""

from __future__ import annotations

import os
import os.path as osp

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import norm

OBS = 0.109  # best expr_002 val/per_gene/pearson_per_gene (controlled _002, 1,161 split)


def main() -> None:
    load_dotenv("/home/michaelvolk/Documents/projects/torchcell/.env")
    f = osp.join(
        os.environ["DATA_ROOT"], "data/deleteome/deleteome_all_mutants_ex_wt_var_controls.txt"
    )
    raw = pd.read_csv(f, sep="\t", header=None, dtype=str, low_memory=False)
    dtype_row = raw.iloc[1].tolist()
    body = raw.iloc[2:].reset_index(drop=True)
    m_cols = [i for i, d in enumerate(dtype_row) if d == "M"]
    p_cols = [i for i, d in enumerate(dtype_row) if d == "p_value"]
    M = body[m_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
    P = body[p_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
    print(f"deleteome: genes={M.shape[0]} mutants={M.shape[1]} (M cols={len(m_cols)} p cols={len(p_cols)})")

    total_var = np.nanvar(M, axis=1, ddof=1)                 # signal + noise, per gene
    z = norm.isf(P / 2.0)                                    # two-sided normal quantile
    with np.errstate(divide="ignore", invalid="ignore"):
        sig = np.abs(M) / z
    sig[(P <= 0) | (P >= 1) | ~np.isfinite(sig)] = np.nan
    sigma_g = np.nanmedian(sig, axis=1)                      # per-gene WT noise SD
    noise_var = sigma_g**2
    rel = np.clip(1.0 - noise_var / total_var, 0.0, 1.0)
    ceiling = np.sqrt(rel)
    ok = np.isfinite(ceiling)

    c = ceiling[ok]
    print(f"\n== Kemmeren expression per-gene ceiling (n={ok.sum()} genes) ==")
    print(f"  mean ceiling (= max achievable mean per-gene Pearson): {c.mean():.4f}")
    print(f"  median ceiling: {np.median(c):.4f}   |  IQR [{np.quantile(c,.25):.3f}, {np.quantile(c,.75):.3f}]")
    print(f"  reliability mean: {np.nanmean(rel[ok]):.4f}")
    for thr in (0.05, 0.10, 0.20, 0.30, 0.50):
        print(f"    genes with ceiling > {thr:.2f}: {(c > thr).sum():5d}  ({100*(c>thr).mean():.1f}%)")

    print("\n" + "=" * 64)
    print(f"OBSERVED expr per-gene (expr_002, 1,161 control): {OBS:.3f}")
    print(f"CEILING  expr per-gene (deleteome noise)        : {c.mean():.3f}")
    print(f"fraction of ceiling realized: {OBS/c.mean():.1%}")
    print(f"headroom (ceiling - observed): {c.mean()-OBS:.3f}")
    print("=" * 64)


if __name__ == "__main__":
    main()
