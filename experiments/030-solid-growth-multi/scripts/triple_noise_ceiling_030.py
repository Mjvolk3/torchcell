# experiments/030-solid-growth-multi/scripts/triple_noise_ceiling_030.py
# [[experiments.030-solid-growth-multi.scripts.triple_noise_ceiling_030]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/030-solid-growth-multi/scripts/triple_noise_ceiling_030
"""How well does a trigenic interaction score reproduce itself on THIS build?

The one published replicate figure for the adjusted trigenic score is Dango's 0.59 Pearson
between the two replicate screens of Kuzmin 2018, on that screen's diagnostic array only.
Kuzmin 2020 reports no all-triples number. This script estimates the reproducibility on
the 030 build's own triples, two ways, and reports Spearman beside Pearson because the
model readout ranks triples.

A. Re-measured triples. The no-merge build keeps every stored entry, and 12,914 of the
   376,732 S3 triples carry two or more interaction entries: the same gene triple reached
   by a different query pair, by a different array allele, in both Kuzmin screens, or by
   the same strains in two of Kuzmin 2020's tables. Each pair of entries is one empirical
   re-measurement. Pairs are classified by what differs between the two entries, so the
   reader can pick the class that matches the question: same strains in both entries is
   as close to a technical replicate as the released tables allow; a different array
   allele is a different genotype that shares the gene triple. One random pair per triple,
   so a triple measured 17 times counts once. Bootstrap over triples for the interval.

B. Noise propagated from the released p-values. Every Kuzmin interaction row carries a
   one-sided normal p-value of the score against its propagated variance (the stored
   maximum is 0.5 on both screens), so the standard error of each score is |tau| / z(p)
   with z the upper-tail quantile. Two synthetic replicates tau + N(0, SE) per row give
   the replicate-replicate correlation the screen's own error model implies, over ALL
   triples of each screen, and the correlation of one noisy draw with the released value
   approximates what a predictor of the denoised score could reach against the released
   one. Rows whose p is too close to 1 for the z to be stable take the screen's median SE.

Both are estimates of reproducibility, not of a mathematical bound: a model scored against
the released score is capped by the square root of the released score's reliability, and
the released score already combines two replicate screens.

    PYTHONPATH=$PWD python experiments/030-solid-growth-multi/scripts/triple_noise_ceiling_030.py

Reads ``closure/entries.parquet`` under ``$DATA_ROOT``; writes
``results/triple_noise_ceiling_030.json``.
"""

from __future__ import annotations

import json
import os
import os.path as osp
import re
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from scipy import stats

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from arm_030 import results_dir  # noqa: E402

TM_TOKEN = re.compile(r"(tm\d+)")
ARRAY_ALLELE = re.compile(r"_(dma|tsa|sn|tsq)\d*")
SEED = 0
N_BOOT = 1000
N_SIM = 20


class PairClass(BaseModel):
    """Correlation between two measurements of the same gene triple, one class."""

    n_triples: int
    pearson: float
    spearman: float
    spearman_ci95: tuple[float, float]
    mean_abs_diff: float
    median_abs_diff: float


class NoiseModel(BaseModel):
    """What the released p-values imply for one screen's triples."""

    n_rows: int
    n_rows_se_from_p: int = Field(description="rows whose z(p) was stable enough to use")
    median_se: float
    sd_of_scores: float
    reliability: float = Field(description="var(score) minus median SE^2, over var(score)")
    replicate_replicate_pearson: float
    replicate_replicate_spearman: float
    noisy_vs_released_pearson: float
    noisy_vs_released_spearman: float


class TripleNoiseCeiling(BaseModel):
    """The report."""

    entries_parquet: str
    n_triples: int
    n_triples_multi_measured: int
    pair_classes: dict[str, PairClass]
    noise_model: dict[str, NoiseModel]
    published: dict[str, str]


def query_token(strain_ids: str | None) -> str | None:
    m = TM_TOKEN.search(strain_ids or "")
    return m.group(1) if m else None


def array_allele(strain_ids: str | None) -> str | None:
    m = ARRAY_ALLELE.search(strain_ids or "")
    return m.group(1) if m else None


def corr_block(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> PairClass:
    sp = float(stats.spearmanr(a, b).statistic)
    boots = []
    n = a.size
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        boots.append(stats.spearmanr(a[i], b[i]).statistic)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    d = np.abs(a - b)
    return PairClass(
        n_triples=int(n),
        pearson=float(np.corrcoef(a, b)[0, 1]),
        spearman=sp,
        spearman_ci95=(float(lo), float(hi)),
        mean_abs_diff=float(d.mean()),
        median_abs_diff=float(np.median(d)),
    )


def main() -> None:
    load_dotenv()
    rng = np.random.default_rng(SEED)
    path = osp.join(
        os.environ["DATA_ROOT"],
        "data/torchcell/experiments/030-solid-growth-multi/closure/entries.parquet",
    )
    df = pd.read_parquet(
        path, columns=["idx", "order", "dataset", "exp_type", "value", "p", "strain_ids"]
    )
    tri = df[
        (df["order"] == 3)
        & (df["exp_type"] == "gene interaction")
        & df["dataset"].str.startswith("Tmi")
    ].copy()
    tri = tri[np.isfinite(tri["value"])]
    n_triples = int(tri["idx"].nunique())
    tri["query"] = tri["strain_ids"].map(query_token)
    tri["allele"] = tri["strain_ids"].map(array_allele)

    # --- A. one random pair of distinct entries per multi-measured triple
    counts = tri.groupby("idx").size()
    multi = tri[tri["idx"].isin(counts[counts >= 2].index)]
    rows = []
    for idx, g in multi.groupby("idx", sort=False):
        g = g.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
        a, b = g.iloc[0], g.iloc[1]
        if a["value"] == b["value"] and a["strain_ids"] == b["strain_ids"]:
            continue  # an exact duplicate row, not a re-measurement
        rows.append(
            {
                "idx": idx,
                "va": a["value"],
                "vb": b["value"],
                "same_dataset": a["dataset"] == b["dataset"],
                "dataset_a": a["dataset"],
                "dataset_b": b["dataset"],
                "same_query": (a["query"] == b["query"]) and a["query"] is not None,
                "same_allele": a["allele"] == b["allele"],
                "same_strains": a["strain_ids"] == b["strain_ids"],
            }
        )
    pairs = pd.DataFrame(rows)
    print(f"triples {n_triples}; multi-measured {len(counts[counts >= 2])}; pairs {len(pairs)}")
    classes: dict[str, pd.DataFrame] = {
        "all_pairs": pairs,
        "same_dataset": pairs[pairs["same_dataset"]],
        "cross_dataset_2018_vs_2020": pairs[~pairs["same_dataset"]],
        "same_dataset_same_strains": pairs[pairs["same_dataset"] & pairs["same_strains"]],
        "same_dataset_same_query_different_array_allele": pairs[
            pairs["same_dataset"] & pairs["same_query"] & ~pairs["same_allele"]
        ],
        "same_dataset_different_query": pairs[pairs["same_dataset"] & ~pairs["same_query"]],
        "kuzmin2020_only": pairs[
            (pairs["dataset_a"] == "TmiKuzmin2020Dataset")
            & (pairs["dataset_b"] == "TmiKuzmin2020Dataset")
        ],
        "kuzmin2018_only": pairs[
            (pairs["dataset_a"] == "TmiKuzmin2018Dataset")
            & (pairs["dataset_b"] == "TmiKuzmin2018Dataset")
        ],
    }
    pair_classes: dict[str, PairClass] = {}
    for name, sub in classes.items():
        if len(sub) < 10:
            print(f"{name}: n={len(sub)} (too few, skipped)")
            continue
        pc = corr_block(sub["va"].to_numpy(float), sub["vb"].to_numpy(float), rng)
        pair_classes[name] = pc
        print(
            f"{name:48s} n={pc.n_triples:6d} pearson {pc.pearson:.3f} spearman {pc.spearman:.3f} "
            f"[{pc.spearman_ci95[0]:.3f}, {pc.spearman_ci95[1]:.3f}] median|d| {pc.median_abs_diff:.4f}"
        )

    # --- B. noise propagated from the released p-values, per screen
    noise: dict[str, NoiseModel] = {}
    for ds, g in tri.groupby("dataset"):
        v = g["value"].to_numpy(float)
        p = g["p"].to_numpy(float)
        # The stored Kuzmin p is ONE-SIDED (its maximum on both screens is 0.5), so the
        # z is the upper-tail quantile of p itself, not of p / 2. Rows with p near 0.5
        # have z near 0 and an unbounded SE; they take the screen median instead.
        z = stats.norm.isf(np.clip(p, 1e-300, 0.5))
        ok = np.isfinite(p) & (p < 0.4) & (np.abs(v) > 0)
        se = np.full(v.size, np.nan)
        se[ok] = np.abs(v[ok]) / z[ok]
        med = float(np.nanmedian(se))
        se[~ok] = med
        rr_p, rr_s, nr_p, nr_s = [], [], [], []
        for _ in range(N_SIM):
            r1 = v + rng.normal(0.0, se)
            r2 = v + rng.normal(0.0, se)
            rr_p.append(np.corrcoef(r1, r2)[0, 1])
            rr_s.append(stats.spearmanr(r1, r2).statistic)
            nr_p.append(np.corrcoef(r1, v)[0, 1])
            nr_s.append(stats.spearmanr(r1, v).statistic)
        var_v = float(np.var(v))
        nm = NoiseModel(
            n_rows=int(v.size),
            n_rows_se_from_p=int(ok.sum()),
            median_se=med,
            sd_of_scores=float(np.sqrt(var_v)),
            reliability=float(max(var_v - med**2, 0.0) / var_v),
            replicate_replicate_pearson=float(np.mean(rr_p)),
            replicate_replicate_spearman=float(np.mean(rr_s)),
            noisy_vs_released_pearson=float(np.mean(nr_p)),
            noisy_vs_released_spearman=float(np.mean(nr_s)),
        )
        noise[ds] = nm
        print(
            f"{ds}: n={nm.n_rows} se_from_p={nm.n_rows_se_from_p} median SE {nm.median_se:.4f} "
            f"sd(tau) {nm.sd_of_scores:.4f} reliability {nm.reliability:.3f} | rep-rep pearson "
            f"{nm.replicate_replicate_pearson:.3f} spearman {nm.replicate_replicate_spearman:.3f} | "
            f"noisy-vs-released pearson {nm.noisy_vs_released_pearson:.3f} spearman {nm.noisy_vs_released_spearman:.3f}"
        )

    out = TripleNoiseCeiling(
        entries_parquet=path,
        n_triples=n_triples,
        n_triples_multi_measured=int((counts >= 2).sum()),
        pair_classes=pair_classes,
        noise_model=noise,
        published={
            "dango_2018_all_triples_pearson": "0.59 (zhangDANGOPredictingHigherorder2020, two replicate screens, all 91,050 triples of the Kuzmin 2018 diagnostic array)",
            "kuzmin2018_si_adjusted_tau_significant_only": "0.74 to 0.81 (si/si1.md line 171, p < 0.05 scores only)",
            "kuzmin2018_si_raw_triple_score_significant_only": "0.90 to 0.91 (same)",
            "kuzmin2020": "no all-triples replicate correlation published; Fig. S2 shows significant scores only",
        },
    )
    dest = osp.join(results_dir(), "triple_noise_ceiling_030.json")
    with open(dest, "w") as f:
        f.write(out.model_dump_json(indent=2))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
