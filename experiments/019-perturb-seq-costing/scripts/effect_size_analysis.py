# experiments/019-perturb-seq-costing/scripts/effect_size_analysis.py
# [[experiments.019-perturb-seq-costing.scripts.effect_size_analysis]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-perturb-seq-costing/scripts/effect_size_analysis
"""How loud is a knockdown, and does volatility grow with more perturbations?

Supplies the empirical parameter that Sec. 4.4 of the perturb-seq review is
missing. Everything in the cells-per-perturbation calculation currently assumes a
nominal two-fold change; this measures the real distribution instead, from
expression compendia already in the library mirror.

Runs anywhere the built LMDBs are present under ``$DATA_ROOT/data/torchcell/``,
and fails fast if they are absent -- no fallbacks, no synthetic data, per the
repo's no-fallback rule.

READS THE LMDB DIRECTLY, and does not import ``torchcell``. Two reasons, and the
first is not merely convenience. (1) The package uses PEP 695 generic syntax and
so needs Python 3.12+, while the only environment on this machine carrying torch
and PyTorch Geometric is 3.11 -- importing the dataset classes fails with a
SyntaxError before any data is touched. (2) Nothing here needs the class
machinery: each LMDB value is a pickled dict of plain builtins, and the two
fields this analysis wants (the perturbation list and the per-gene
``expression_log2_ratio``) are read straight out of it. Depending only on the
stored records rather than on the reader also means this script keeps working if
the dataset classes are refactored.

The record contract it relies on, which is what to check if this ever breaks:

    record["experiment"]["genotype"]["perturbations"] -> [{systematic_gene_name}]
    record["experiment"]["phenotype"]["expression_log2_ratio"] -> {gene: log2 FC}

    python experiments/019-perturb-seq-costing/scripts/effect_size_analysis.py

Three questions, in the order Sec. 4.4 asks them:

1. **What threshold defines a responder?** Two-fold is almost certainly too
   strict for a regulatory response. A ladder of thresholds is swept, and each is
   also expressed relative to the dataset's own technical noise, because a fold
   change is only meaningful against the measurement error that produced it.
2. **What is the distribution of |log2 FC| over responding genes?** That, not a
   nominal two-fold, is what belongs in the power calculation
   (``cost_model.umis_needed_for_fold_change``).
3. **Does a second deletion increase the response?** The defensible comparison is
   Sameith singles vs Sameith doubles -- same lab, same platform, same noise
   floor. Comparing Kemmeren singles against Sameith doubles would confound the
   perturbation count with the study, and Kemmeren is used only for the much
   larger singles sample.

The output that matters is the slope of responders vs perturbation count. It is
the only principled basis for extrapolating to the 8--10 perturbation regime the
multiplexing argument depends on, and a linear fit is the conservative choice --
saturation would make high-plex screens *easier* to read, not harder.
"""

from __future__ import annotations

import json
import os
import os.path as osp

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

import lmdb
import pickle

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
RESULTS_DIR = osp.join(
    os.environ["EXPERIMENT_ROOT"], "019-perturb-seq-costing", "results"
)

# Fold-change thresholds to sweep. 1.25x is nearer the convention for a
# regulatory response than the 2x the cost model currently assumes; 2x is kept so
# the current assumption can be read off the same table.
FOLD_THRESHOLDS = [1.25, 1.5, 2.0, 4.0]


def log2_thresh(fold: float) -> float:
    return float(np.log2(fold))


class LmdbRecords:
    """Minimal reader over a built torchcell dataset LMDB.

    Keys are the record index as ASCII bytes ("0", "1", ...); values are pickled
    dicts. Opened read-only with locking off so it cannot disturb a concurrent
    build on another machine.
    """

    def __init__(self, root: str) -> None:
        self.path = osp.join(root, "processed", "lmdb")
        self.env = lmdb.open(self.path, readonly=True, lock=False, subdir=True)
        with self.env.begin() as tx:
            self.n = tx.stat()["entries"]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        with self.env.begin() as tx:
            raw = tx.get(str(i).encode())
        if raw is None:
            raise KeyError(f"no record {i} in {self.path}")
        return pickle.loads(raw)


def extract(dataset, name: str) -> list[dict]:
    """One record per strain: its perturbed genes and its response vector.

    Kept deliberately close to experiments/012-sameith-kemmeren's extraction so
    the two analyses are comparable; the difference is that this one keeps the
    per-strain technical SD alongside the effect, which is what lets a responder
    threshold be calibrated against noise rather than asserted.
    """
    out = []
    for i in tqdm(range(len(dataset)), desc=name):
        d = dataset[i]
        perts = d["experiment"]["genotype"]["perturbations"]
        genes = [p["systematic_gene_name"] for p in perts]
        pheno = d["experiment"]["phenotype"]
        ratios = np.array(list(pheno["expression_log2_ratio"].values()), dtype=float)
        ratios = ratios[~np.isnan(ratios)]
        if ratios.size == 0:
            continue
        rec = {
            "dataset": name,
            "n_perturbations": len(genes),
            "genes": "|".join(genes),
            "n_measured": int(ratios.size),
        }
        # Technical SD, when the dataset carries it. Used for the noise-calibrated
        # responder definition; absent for some records, which is why the
        # z-based columns are computed separately and may be NaN.
        sd_key = "expression_log2_ratio_std"
        if sd_key in pheno and pheno[sd_key] is not None:
            sds = np.array(list(pheno[sd_key].values()), dtype=float)
            sds = sds[~np.isnan(sds)]
            rec["median_technical_sd"] = float(np.median(sds)) if sds.size else np.nan
        else:
            rec["median_technical_sd"] = np.nan

        a = np.abs(ratios)
        for f in FOLD_THRESHOLDS:
            rec[f"n_resp_{f}x"] = int((a > log2_thresh(f)).sum())
            rec[f"frac_resp_{f}x"] = float((a > log2_thresh(f)).mean())
        # Distribution of the response among genes that respond at the 1.25x
        # level -- the population that actually enters the power calculation.
        resp = a[a > log2_thresh(1.25)]
        rec["median_abs_log2fc_responders"] = float(np.median(resp)) if resp.size else np.nan
        rec["p90_abs_log2fc_responders"] = float(np.percentile(resp, 90)) if resp.size else np.nan
        rec["max_abs_log2fc"] = float(a.max())
        out.append(rec)
        PER_GENE.setdefault(name, []).append(ratios)
        # Boolean responder set at 1.25x, kept only for the compendium the
        # union model uses; 1,484 x 6,169 bools is ~9 MB, the others are not
        # large enough samples to draw from.
        if len(genes) == 1:
            RESP_SETS.setdefault(name, []).append(a > log2_thresh(1.25))
    return out


# Populated by extract(); consumed by distributions(). Module-level because the
# per-strain table and the distributions are two views of one pass over the data,
# and reading 1.5 GB of LMDB twice to keep them separate is not worth it.
PER_GENE: dict[str, list] = {}
RESP_SETS: dict[str, list] = {}


# Fine ladder for the responders-vs-threshold curve. FOLD_THRESHOLDS above is the
# coarse set the table reports; this is for the plot, where the shape between the
# reported points is the whole message.
FOLD_LADDER = [1.1, 1.15, 1.2, 1.25, 1.35, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 6.0]

# Histogram of |log2 FC| pooled over strains. Emitting binned counts rather than
# the raw ~9 million per-gene values keeps the artifact small and committable
# while preserving the shape the plot needs.
HIST_EDGES = np.linspace(0.0, 3.0, 61)


def distributions(per_gene: dict[str, list], out_dir: str) -> None:
    """Write the pooled |log2 FC| histogram and the threshold ladder."""
    rows = []
    for name, vals in per_gene.items():
        a = np.concatenate(vals)
        counts, _ = np.histogram(a, bins=HIST_EDGES)
        for lo, hi, c in zip(HIST_EDGES[:-1], HIST_EDGES[1:], counts):
            rows.append({"dataset": name, "lo": lo, "hi": hi, "count": int(c),
                         "frac": float(c) / a.size})
    pd.DataFrame(rows).to_csv(
        osp.join(out_dir, "effect_size_histogram.csv"), index=False)

    rows = []
    for name, vals in per_gene.items():
        n_strains = len(vals)
        for f in FOLD_LADDER:
            t = log2_thresh(f)
            per_strain = np.array([float((np.abs(v) > t).sum()) for v in vals])
            # Median |log2 FC| among the genes that clear THIS threshold, pooled
            # over strains. Recorded per rung because the headline "the median
            # responding gene moves 1.34x" is not a property of the biology: the
            # distribution falls off monotonically, so the median of its upper
            # tail sits just above wherever the cut is put, and it moves with the
            # cut. Sec. 4.4 has to be able to show that rather than assert one
            # number, and the power calculation has to be able to be redone at a
            # different definition of responding.
            pooled = np.concatenate([np.abs(v) for v in vals])
            resp = pooled[pooled > t]
            rows.append({
                "dataset": name, "fold": f,
                "median_responders": float(np.median(per_strain)),
                "q25_responders": float(np.percentile(per_strain, 25)),
                "q75_responders": float(np.percentile(per_strain, 75)),
                "median_abs_log2fc_responders": (
                    float(np.median(resp)) if resp.size else np.nan),
                "median_fold_responders": (
                    float(2 ** np.median(resp)) if resp.size else np.nan),
                "n_strains": n_strains,
            })
    pd.DataFrame(rows).to_csv(
        osp.join(out_dir, "effect_size_threshold_ladder.csv"), index=False)


# --- extrapolating to more perturbations --------------------------------------
# Sameith gives only two levels (1 and 2 deletions), and ANY two points are fitted
# exactly by a line, a power law or a saturating curve -- so a fit to them cannot
# choose between models, and extrapolating one to k=5 would be a decision
# disguised as a measurement.
#
# There is a better estimator available. Kemmeren has 1,484 SINGLE deletions with
# per-gene responses, so the responder SET of each is known. Draw k of them at
# random and take the union: that is exactly "what would respond if k
# perturbations were combined and their effects did not interact", measured with
# the real overlap structure rather than an independence assumption. Overlap is
# not incidental here -- stress and ribosomal genes respond to almost anything, so
# an independent-sets calculation would materially overcount the union.
#
# This is a NULL model, and naming it that way is the point:
#   * it is a lower bound if perturbations synergise (Sameith's pairs, chosen to
#     interact, sit above it),
#   * it is an upper bound if they buffer one another,
#   * and the gap between it and Sameith's observed doubles is a direct estimate
#     of how much epistasis moves the answer at k=2.
UNION_DRAWS = 2000
UNION_K = list(range(1, 11))


def union_extrapolation(resp_sets: np.ndarray, rng: np.random.Generator,
                        n_draws: int = UNION_DRAWS) -> pd.DataFrame:
    """Expected responders when k independent single-deletion sets are combined.

    ``resp_sets`` is a boolean strains x genes matrix of "did this gene respond".
    Sampling is without replacement within a draw: combining a strain with itself
    is not a two-perturbation experiment.
    """
    n_strains = resp_sets.shape[0]
    rows = []
    for k in UNION_K:
        sizes = np.empty(n_draws, dtype=float)
        for j in range(n_draws):
            idx = rng.choice(n_strains, size=k, replace=False)
            sizes[j] = float(resp_sets[idx].any(axis=0).sum())
        rows.append({
            "k": k,
            "median": float(np.median(sizes)),
            "q25": float(np.percentile(sizes, 25)),
            "q75": float(np.percentile(sizes, 75)),
        })
    return pd.DataFrame(rows)


def fit_saturating(df: pd.DataFrame) -> dict:
    """Fit R(k) = G_eff (1 - (1 - p)^k) to the null curve.

    Two parameters with a mechanistic reading: ``G_eff`` is the size of the pool
    of genes that respond to anything at all, and ``p`` is the chance a given
    perturbation moves a given gene in that pool. Fitted on the null curve rather
    than on the two observed levels, because the null curve has ten points and a
    known generating process.

    Reported alongside a plain linear extrapolation so the two can be compared
    where they diverge, which is the honest way to present an extrapolation from
    a k<=2 observation.
    """
    k = df["k"].to_numpy(dtype=float)
    y = df["median"].to_numpy(dtype=float)
    best = None
    # Grid over BOTH parameters. The objective is smooth and two-dimensional, so
    # a grid is clearer than an optimiser whose convergence we would have to
    # argue about, and it makes a boundary solution visible rather than silent.
    for g_eff in np.linspace(y.max() * 1.02, 6169.0, 300):
        for pp in np.linspace(0.005, 0.35, 300):
            pred = g_eff * (1.0 - (1.0 - pp) ** k)
            sse = float(((pred - y) ** 2).sum())
            if best is None or sse < best[0]:
                best = (sse, g_eff, pp)
    _, g_eff, p = best
    return {"G_eff": float(g_eff), "p": float(p),
            "predicted": {int(kk): float(g_eff * (1 - (1 - p) ** kk))
                          for kk in UNION_K}}


def require(path: str, what: str) -> str:
    if not osp.isdir(path):
        raise SystemExit(
            f"MISSING DATASET: {what}\n  expected at: {path}\n\n"
            "Build it, or rsync processed/ and preprocess/ from the machine "
            "that did.\nIt does not fall back to synthetic or cached data by design."
        )
    return path


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    kem_root = require(
        osp.join(DATA_ROOT, "data/torchcell/microarray_kemmeren2014"), "Kemmeren 2014"
    )
    sm_root = require(
        osp.join(DATA_ROOT, "data/torchcell/sm_microarray_sameith2015"),
        "Sameith 2015 singles",
    )
    dm_root = require(
        osp.join(DATA_ROOT, "data/torchcell/dm_microarray_sameith2015"),
        "Sameith 2015 doubles",
    )

    recs = []
    recs += extract(LmdbRecords(kem_root),
                    "kemmeren2014_single")
    recs += extract(LmdbRecords(sm_root),
                    "sameith2015_single")
    recs += extract(LmdbRecords(dm_root),
                    "sameith2015_double")

    df = pd.DataFrame(recs)
    df.to_csv(osp.join(RESULTS_DIR, "effect_size_per_strain.csv"), index=False)
    distributions(PER_GENE, RESULTS_DIR)

    # Union null model, and its saturating fit.
    rng = np.random.default_rng(0)  # fixed seed: the curve must be reproducible
    unions = {}
    for name, rows in RESP_SETS.items():
        unions[name] = union_extrapolation(np.vstack(rows), rng)
        unions[name]["dataset"] = name
    pd.concat(unions.values()).to_csv(
        osp.join(RESULTS_DIR, "union_extrapolation.csv"), index=False)

    # Extrapolation curve from KEMMEREN: 1,484 singles, an unbiased draw from the
    # deletion collection, so it carries the best estimate of overlap structure.
    union = unions["kemmeren2014_single"]
    fit = fit_saturating(union)

    # Epistasis check WITHIN Sameith: its own singles combined, against its own
    # observed doubles. Using the Kemmeren null here would confound the deviation
    # with the difference between the two studies' baselines (269 vs 220).
    sm = unions["sameith2015_single"]
    obs2 = float(df[df.dataset == "sameith2015_double"]["n_resp_1.25x"].median())
    fit["sameith_null_k2"] = float(sm[sm.k == 2]["median"].iloc[0])
    fit["sameith_observed_k2"] = obs2
    fit["sameith_epistasis_ratio_k2"] = obs2 / fit["sameith_null_k2"]
    fit["kemmeren_null_k2"] = float(union[union.k == 2]["median"].iloc[0])
    with open(osp.join(RESULTS_DIR, "union_extrapolation_fit.json"), "w") as fh:
        json.dump(fit, fh, indent=2)
    print("\n=== union null model (Kemmeren singles combined) ===")
    print(union.to_string(index=False))
    print(f"\nsaturating fit: G_eff={fit['G_eff']:.0f}, p={fit['p']:.4f}")
    lin = union[union.k == 1]["median"].iloc[0]
    print("k   union   linear kR(1)   union/linear")
    for _, r in union.iterrows():
        print(f"{int(r.k):<3} {r['median']:7.0f} {lin*r.k:12.0f} "
              f"{r['median']/(lin*r.k):13.2f}")
    print(f"\nWITHIN Sameith at k=2: null {fit['sameith_null_k2']:.0f} vs "
          f"observed {obs2:.0f} -> ratio {fit['sameith_epistasis_ratio_k2']:.2f}")

    # --- Q1/Q2: responder counts and effect distribution, by dataset ----------
    agg = (
        df.groupby(["dataset", "n_perturbations"])
        .agg(
            n_strains=("n_measured", "size"),
            median_n_measured=("n_measured", "median"),
            median_technical_sd=("median_technical_sd", "median"),
            **{
                f"median_n_resp_{f}x": (f"n_resp_{f}x", "median")
                for f in FOLD_THRESHOLDS
            },
            median_abs_log2fc_responders=("median_abs_log2fc_responders", "median"),
            p90_abs_log2fc_responders=("p90_abs_log2fc_responders", "median"),
        )
        .reset_index()
    )
    agg.to_csv(osp.join(RESULTS_DIR, "effect_size_summary.csv"), index=False)

    # --- Q3: does a second perturbation increase the response? ---------------
    # WITHIN Sameith only. Comparing across datasets would confound perturbation
    # count with study, platform, and noise floor.
    sam = df[df.dataset.str.startswith("sameith2015")]
    slopes = {}
    for f in FOLD_THRESHOLDS:
        col = f"n_resp_{f}x"
        g = sam.groupby("n_perturbations")[col]
        means = g.mean()
        if len(means) < 2:
            continue
        x = means.index.to_numpy(dtype=float)
        y = means.to_numpy(dtype=float)
        # Two points at n=1,2 -> the "fit" is a difference. Reported as a slope
        # so it extrapolates, and flagged as such: with only two levels this is
        # an assumption of linearity, not evidence for it.
        slope = float(np.polyfit(x, y, 1)[0])
        slopes[f"{f}x"] = {
            "mean_responders_by_nperts": {int(k): float(v) for k, v in means.items()},
            "slope_responders_per_perturbation": slope,
            "n_levels_observed": int(len(means)),
            "linearity_tested": bool(len(means) > 2),
        }

    # Extrapolate to the multiplex regime the review argues about.
    extrap = {}
    for f, s in slopes.items():
        base = s["mean_responders_by_nperts"].get(1)
        if base is None:
            continue
        extrap[f] = {
            str(k): base + s["slope_responders_per_perturbation"] * (k - 1)
            for k in (1, 2, 3, 5, 8, 10)
        }

    out = {
        "thresholds_fold": FOLD_THRESHOLDS,
        "within_sameith_slopes": slopes,
        "linear_extrapolation_responders": extrap,
        "caveat": (
            "Only n_perturbations in {1,2} is observed, so the slope is a "
            "difference extrapolated linearly, not a fitted trend. Treat the "
            "8-10 perturbation numbers as an upper bound on transcriptome "
            "volatility: saturation would make high-plex screens easier to read, "
            "not harder."
        ),
    }
    with open(osp.join(RESULTS_DIR, "effect_size_multiplex.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\n=== responders and effect size, by dataset and perturbation count ===")
    print(agg.to_string(index=False))
    print("\n=== within-Sameith slope (responders per added perturbation) ===")
    print(json.dumps(out["within_sameith_slopes"], indent=2))
    print(f"\nwrote 3 files to {RESULTS_DIR}")
    print(
        "\nNEXT: feed median_abs_log2fc_responders into "
        "cost_model.umis_needed_for_fold_change() to replace the nominal 2x, "
        "then regenerate tables (render_tex_tables.py) and rebuild the document."
    )


if __name__ == "__main__":
    main()
