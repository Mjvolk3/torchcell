# experiments/W019-echo-crispr-array/scripts/run4_wt_reference_diagnostic.py
# [[experiments.W019-echo-crispr-array.scripts.run4_wt_reference_diagnostic]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/W019-echo-crispr-array/scripts/run4_wt_reference_diagnostic
"""DIAGNOSTIC ONLY -- re-score run 4 against the plate median instead of the WT wells.

**This is not a proposed correction and its output must not be used as a result.** Run 4's
single mutants score above the on-plate wild type (8 of 12), which inflates the
multiplicative expectation f_a*f_b past 1.0 and drives every digenic interaction negative.
Why the wild-type reference moved is unknown. This script asks ONE narrow question:

    is the shift specific to the WT WELLS, or is it a property of the whole round?

by swapping the denominator. If normalizing to the plate median puts the singles back on
run 3's scale, the anomaly is localized to the WT wells and is worth chasing at the bench. If
it does not, the WT wells are not the story and something broader changed between rounds.

Swapping the denominator does NOT make the numbers correct -- it makes them differently
normalized. The plate median is itself a mutant-panel statistic here (25 of 26 strains are
mutants, 13 of them doubles that genuinely grow less), so it is a biased stand-in for a wild
type and will compress the very fitness range the assay exists to measure. It answers a
diagnostic question and nothing else.

Run from repo root (reads the scored colony tables; no GPU, no re-segmentation):
    ~/miniconda3/envs/torchcell/bin/python \
        experiments/W019-echo-crispr-array/scripts/run4_wt_reference_diagnostic.py
"""

from __future__ import annotations

import os
import os.path as osp

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import pearsonr, spearmanr

load_dotenv()
EXP_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
RESULTS_DIR = osp.join(EXP_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

PLATES = ["P1", "P2", "P3"]
WT_NAME = "WT"
BLANK_NAME = "Blank_media"
SEED = 1234
N_BOOT = 4000


def rescore(denominator: str) -> pd.DataFrame:
    """Per-plate per-strain fitness under a chosen reference.

    ``denominator`` is either ``"wt"`` (the shipped scoring: strain median / WT median) or
    ``"plate_median"`` (strain median / median over all non-blank strains on the plate).
    Both read the same per-strain median_norm values, so the ONLY thing that changes between
    them is the denominator -- any difference in the output is attributable to that and
    nothing else.
    """
    rows = []
    for p in PLATES:
        d = pd.read_csv(osp.join(RESULTS_DIR, f"run4_strain_scores_{p}.csv"))
        d = d[d["strain"] != BLANK_NAME]
        if denominator == "wt":
            ref = float(d.loc[d["strain"] == WT_NAME, "median_norm"].iloc[0])
        else:
            ref = float(d.loc[d["strain"] != WT_NAME, "median_norm"].median())
        for _, r in d.iterrows():
            rows.append(
                dict(plate=p, strain=r["strain"], fitness=r["median_norm"] / ref)
            )
    return pd.DataFrame(rows)


def compare_to_run3(fit: pd.DataFrame, label: str) -> dict:
    """Singles vs run 3's bootstrap fitness for the strains common to both rounds."""
    r3 = pd.read_csv(osp.join(RESULTS_DIR, "run3_vs_reference.csv"))[
        ["orf", "boot_fitness"]
    ]
    mean = fit.groupby("strain")["fitness"].mean().rename("run4").reset_index()
    m = r3.merge(mean, left_on="orf", right_on="strain", how="inner")
    pr, pp = pearsonr(m["boot_fitness"], m["run4"])
    ratio = (m["run4"] / m["boot_fitness"]).median()
    return dict(
        reference=label,
        n=len(m),
        pearson_r=round(float(pr), 3),
        p=round(float(pp), 4),
        median_ratio_run4_over_run3=round(float(ratio), 3),
        mean_abs_diff_after_rescale=round(
            float((m["run4"] / ratio - m["boot_fitness"]).abs().mean()), 3
        ),
    )


def singles_vs_costanzo(fit: pd.DataFrame, label: str) -> dict:
    ref = pd.read_csv(osp.join(RESULTS_DIR, "reference_smf_12panel.csv"))[
        ["orf", "costanzo_smf"]
    ].dropna()
    mean = fit.groupby("strain")["fitness"].mean().rename("run4").reset_index()
    m = ref.merge(mean, left_on="orf", right_on="strain", how="inner")
    pr, pp = pearsonr(m["costanzo_smf"], m["run4"])
    sr, _ = spearmanr(m["costanzo_smf"], m["run4"])
    return dict(reference=label, n=len(m), pearson_r=round(float(pr), 3),
                p=round(float(pp), 4), spearman_rho=round(float(sr), 3))


def interactions(fit: pd.DataFrame, label: str) -> pd.DataFrame:
    """Eps per double under this reference, bootstrapped across plates."""
    rng = np.random.default_rng(SEED)
    wide = fit.pivot_table(index="strain", columns="plate", values="fitness")
    doubles = [s for s in wide.index if "+" in s]
    out = []
    for name in doubles:
        a, b = name.split("+")
        if a not in wide.index or b not in wide.index:
            continue
        e, exp = [], []
        for p in PLATES:
            f_ab, f_a, f_b = wide.at[name, p], wide.at[a, p], wide.at[b, p]
            if np.isnan(f_ab) or np.isnan(f_a) or np.isnan(f_b):
                continue
            e.append(f_ab - f_a * f_b)
            exp.append(f_a * f_b)
        if not e:
            continue
        e = np.array(e)
        draws = rng.choice(e, size=(N_BOOT, e.size), replace=True).mean(axis=1)
        out.append(dict(reference=label, double=name, expected=float(np.mean(exp)),
                        eps=float(e.mean()), eps_se=float(draws.std(ddof=1))))
    return pd.DataFrame(out)


def main() -> None:
    print(__doc__.split("Run from repo root")[0].strip())
    print("=" * 78)

    singles_summary, inter_all = [], []
    for denom, label in [("wt", "WT wells (as shipped)"), ("plate_median", "plate median")]:
        fit = rescore(denom)
        singles = fit[~fit["strain"].str.contains(r"\+") & (fit["strain"] != WT_NAME)]
        above = int((singles.groupby("strain")["fitness"].mean() > 1).sum())
        n_sing = singles["strain"].nunique()

        vs3 = compare_to_run3(fit, label)
        vsc = singles_vs_costanzo(fit, label)
        inter = interactions(fit, label)
        inter_all.append(inter)

        print(f"\n--- reference: {label} ---")
        print(f"  singles above reference        : {above}/{n_sing}")
        print(f"  vs run 3 (n={vs3['n']})            : r={vs3['pearson_r']}  "
              f"median run4/run3 ratio={vs3['median_ratio_run4_over_run3']}")
        print(f"  vs Costanzo SMF (n={vsc['n']})     : r={vsc['pearson_r']} "
              f"(p={vsc['p']})  rho={vsc['spearman_rho']}")
        print(f"  expectation f_a*f_b > 1        : "
              f"{int((inter['expected'] > 1).sum())}/{len(inter)}  "
              f"(max {inter['expected'].max():.3f})")
        print(f"  eps negative                   : "
              f"{int((inter['eps'] < 0).sum())}/{len(inter)}")
        r, _ = pearsonr(inter["expected"], inter["eps"])
        print(f"  corr(eps, f_a*f_b)             : {r:+.3f}   "
              f"(near 0 = eps not tracking its own expectation)")
        singles_summary.append({**vs3, "singles_above_reference": f"{above}/{n_sing}",
                                "costanzo_r": vsc["pearson_r"]})

    out_s = osp.join(RESULTS_DIR, "run4_wt_diagnostic_singles.csv")
    out_i = osp.join(RESULTS_DIR, "run4_wt_diagnostic_interactions.csv")
    pd.DataFrame(singles_summary).to_csv(out_s, index=False)
    pd.concat(inter_all, ignore_index=True).to_csv(out_i, index=False)
    print(f"\nwrote {out_s}")
    print(f"wrote {out_i}")
    print(
        "\nREMINDER: this is a diagnostic, not a correction. The plate median is a "
        "mutant-panel\nstatistic (25 of 26 strains are mutants, 13 of them doubles), so it "
        "is a biased stand-in\nfor a wild type and compresses the fitness range the assay "
        "exists to measure."
    )


if __name__ == "__main__":
    main()
