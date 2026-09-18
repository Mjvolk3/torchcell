# experiments/025-solid-growth/scripts/s3_closure_recompute.py
# [[experiments.025-solid-growth.scripts.s3_closure_recompute]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/025-solid-growth/scripts/s3_closure_recompute

"""Recompute the interaction scores AND their p-values from the S3 closure pool.

S3 is the 025 build restricted to every single-gene record, every double whose gene pair
lies inside some measured triple, and every triple (1,121,645 records). The interaction
scores are defined from fitness alone,

    eps_ab  = f_ab  - f_a f_b
    tau_abc = f_abc - f_a f_b f_c - eps_ab f_c - eps_ac f_b - eps_bc f_a,

so the pool carries, for every triple, every term of its own defining equation. This
script asks two questions of the STORED values and answers both against the SOURCE
tables the build was assembled from:

strength    Does the fitness in the build reproduce the interaction in the build?
            (recomputed eps / tau against stored dmi / tmi, per stratum)
confidence  Does anything in the build reproduce the p-value in the build?
            (a z-test propagated from the stored uncertainties against the stored p;
            and, for records the build merged from several screens, the merged p
            against the p-values the sources reported)

The within-screen control recomputes eps from the raw Costanzo 2016 and Kuzmin 2018 /
2020 tables, where the same identity holds row by row, so any loss seen in the build is
attributable to the join (cross-screen averaging of singles, essentiality and synthetic
lethality entering as fitness 0, duplicate screens averaged, p-values replaced by a
t-test over the duplicates) and not to the formula.

Stages (each caches its output under $DATA_ROOT/data/torchcell/experiments/025-solid-growth/s3_closure/):

    python experiments/025-solid-growth/scripts/s3_closure_recompute.py scan     # S3 pool -> s3_records.parquet
    python experiments/025-solid-growth/scripts/s3_closure_recompute.py raw      # source tables -> raw_*.parquet
    python experiments/025-solid-growth/scripts/s3_closure_recompute.py analyze  # results/s3_closure_*.json, tables, figures

Outputs (experiments/025-solid-growth/results/):
- s3_closure_recompute_summary.json
- s3_closure_composition.csv          (records per order and source combination)
Figures (ASSET_IMAGES_DIR/025-solid-growth/s3_closure_*.svg|png) and LaTeX tables
(notes-tex/025-s3-closure/tables/*.tex) are written by `analyze`.
"""

import gzip
import json
import os
import os.path as osp
import sys
from collections import Counter
from itertools import combinations
from multiprocessing import Pool

import lmdb
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy import stats

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]

BUILD = osp.join(
    DATA_ROOT, "data/torchcell/experiments/025-solid-growth/001-full-build/processed"
)
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "025-solid-growth/results")
CACHE_DIR = osp.join(DATA_ROOT, "data/torchcell/experiments/025-solid-growth/s3_closure")
IMG_DIR = osp.join(ASSET_IMAGES_DIR, "025-solid-growth")
TABLE_DIR = osp.join(osp.dirname(EXPERIMENT_ROOT), "notes-tex/025-s3-closure/tables")
RAW_COSTANZO = osp.join(DATA_ROOT, "data/torchcell/dmf_costanzo2016/raw")
RAW_KUZMIN_2018 = osp.join(DATA_ROOT, "data/torchcell/dmf_kuzmin2018/raw/aao1729_data_s1.tsv")
RAW_KUZMIN_2020 = [
    osp.join(DATA_ROOT, "data/torchcell/dmf_kuzmin2020/raw/aaz5667-Table-S1.xlsx"),
    osp.join(DATA_ROOT, "data/torchcell/dmf_kuzmin2020/raw/aaz5667-Table-S3.xlsx"),
]

# Colony replicates behind a combined-mutant fitness SD (Costanzo 2016 SI: "4 replicate
# colonies per double mutant"; the Kuzmin loaders record n_samples = 4 for the same
# reason). Used only where the record carries no n_samples of its own.
N_COLONIES_DEFAULT = 4
os.makedirs(CACHE_DIR, exist_ok=True)


# --------------------------------------------------------------------------- scan
_ENV = None


def _init_worker() -> None:
    global _ENV
    _ENV = lmdb.open(osp.join(BUILD, "lmdb"), readonly=True, lock=False, max_readers=128)


def _parse(idx: int, raw: bytes) -> dict:
    entries = json.loads(raw.decode())
    genes = sorted(
        {
            p["systematic_gene_name"]
            for p in entries[0]["experiment"]["genotype"]["perturbations"]
        }
    )
    fit_vals, fit_sd, fit_ds, fit_dup, fit_n = [], [], [], 0, None
    gi_val, gi_p, gi_ds, gi_dup = None, None, None, 0
    for item in entries:
        e = item["experiment"]
        ph = e["phenotype"]
        ds = e["dataset_name"]
        if e["experiment_type"] == "fitness":
            fit_vals.append(ph["fitness"])
            fit_ds.append(ds)
            if ph.get("fitness_std") is not None:
                fit_sd.append(ph["fitness_std"])
            if ph.get("n_samples") is not None:
                fit_n = ph["n_samples"]
            for p in e["genotype"]["perturbations"]:
                if p.get("num_duplicates"):
                    fit_dup = max(fit_dup, int(p["num_duplicates"]))
        elif e["experiment_type"] == "gene interaction":
            # one interaction entry per genotype after deduplication
            gi_val = ph["gene_interaction"]
            gi_p = ph.get("gene_interaction_p_value")
            gi_ds = ds
            gi_dup = ds.count("+") + 1
    all_fit_ds = "+".join(fit_ds)
    return {
        "idx": idx,
        "order": len(genes),
        "genes": "|".join(genes),
        "fit_n_entries": len(fit_vals),
        "fit_mean": float(np.mean(fit_vals)) if fit_vals else np.nan,
        "fit_min": float(np.min(fit_vals)) if fit_vals else np.nan,
        "fit_max": float(np.max(fit_vals)) if fit_vals else np.nan,
        "fit_sd": float(np.sqrt(np.mean(np.square(fit_sd)))) if fit_sd else np.nan,
        "fit_n_samples": fit_n,
        "fit_dup": fit_dup,
        "fit_ds": all_fit_ds,
        "has_ess": "GeneEssentialitySgd" in all_fit_ds,
        "has_sl": "SynthLeth" in all_fit_ds,
        "fit_kuzmin_only": bool(fit_ds) and all(("Kuzmin" in d) for d in fit_ds),
        "gi": np.nan if gi_val is None else float(gi_val),
        "gi_p": np.nan if gi_p is None else float(gi_p),
        "gi_ds": gi_ds or "",
        "gi_dup": gi_dup,
    }


def _scan_chunk(idxs: list[int]) -> list[dict]:
    assert _ENV is not None
    out = []
    with _ENV.begin() as txn:
        for i in idxs:
            out.append(_parse(i, txn.get(str(i).encode())))
    return out


def scan() -> None:
    with gzip.open(osp.join(RESULTS_DIR, "subset_S3_indices.json.gz"), "rt") as f:
        idxs = json.load(f)
    print(f"S3 records: {len(idxs):,}", flush=True)
    chunks = [idxs[i : i + 2000] for i in range(0, len(idxs), 2000)]
    rows: list[dict] = []
    with Pool(48, initializer=_init_worker) as pool:
        for n, part in enumerate(pool.imap_unordered(_scan_chunk, chunks)):
            rows.extend(part)
            if n % 50 == 0:
                print(f"  chunks done {n}/{len(chunks)}", flush=True)
    df = pd.DataFrame(rows).sort_values("idx").reset_index(drop=True)
    df.to_parquet(osp.join(CACHE_DIR, "s3_records.parquet"), index=False)
    print(df["order"].value_counts().sort_index().to_string(), flush=True)


# --------------------------------------------------------------------------- raw
COST_COLS = {
    "Query Strain ID": "q",
    "Array Strain ID": "a",
    "Arraytype/Temp": "arr",
    "Genetic interaction score (ε)": "eps",
    "P-value": "p",
    "Query single mutant fitness (SMF)": "f_q",
    "Array SMF": "f_a",
    "Double mutant fitness": "f_qa",
    "Double mutant fitness standard deviation": "sd_qa",
}
KUZ_COLS = {
    "Query strain ID": "q",
    "Array strain ID": "a",
    "Combined mutant type": "type",
    "Raw genetic interaction score (epsilon)": "eps_raw",
    "Adjusted genetic interaction score (epsilon or tau)": "score",
    "P-value": "p",
    "Query single/double mutant fitness": "f_q",
    "Array single mutant fitness": "f_a",
    "Combined mutant fitness": "f_qa",
    "Combined mutant fitness standard deviation": "sd_qa",
    # Kuzmin 2020 names the same two columns differently
    "Double/triple mutant fitness": "f_qa",
    "Double/triple mutant fitness standard deviation": "sd_qa",
}


def _gene(strain: pd.Series) -> pd.Series:
    return strain.str.split("_", n=1).str[0]


def _closure_pairs() -> set[frozenset]:
    df = pd.read_parquet(osp.join(CACHE_DIR, "s3_records.parquet"), columns=["order", "genes"])
    return {frozenset(g.split("|")) for g in df.loc[df["order"] == 2, "genes"]}


def raw() -> None:
    pairs = _closure_pairs()
    print(f"closure pairs: {len(pairs):,}", flush=True)
    parts = []
    for name in ["SGA_NxN.txt", "SGA_ExN_NxE.txt", "SGA_ExE.txt", "SGA_DAmP.txt"]:
        d = pd.read_csv(
            osp.join(RAW_COSTANZO, name), sep="\t", usecols=list(COST_COLS), low_memory=False
        ).rename(columns=COST_COLS)
        d["gene_q"], d["gene_a"] = _gene(d["q"]), _gene(d["a"])
        key = [frozenset((x, y)) for x, y in zip(d["gene_q"], d["gene_a"])]
        keep = np.fromiter((k in pairs for k in key), dtype=bool, count=len(key))
        d = d[keep].copy()
        d["file"] = name
        d["source"] = "costanzo2016"
        parts.append(d)
        print(f"  {name}: kept {keep.sum():,} of {len(keep):,}", flush=True)
    kz = [pd.read_csv(RAW_KUZMIN_2018, sep="\t").rename(columns=KUZ_COLS).assign(source="kuzmin2018")]
    for path in RAW_KUZMIN_2020:
        kz.append(
            pd.read_excel(path, skiprows=1).rename(columns=KUZ_COLS).assign(source="kuzmin2020")
        )
    k = pd.concat(kz, ignore_index=True)
    k["file"] = k["source"]
    k.to_parquet(osp.join(CACHE_DIR, "raw_kuzmin_all.parquet"), index=False)
    print(f"kuzmin rows: {len(k):,} ({k['type'].value_counts().to_dict()})", flush=True)
    kd = k[k["type"] == "digenic"].copy()
    # a Kuzmin digenic query is "GENE+YDL227C_tmNNNN" (the ho-deletion partner); strip it
    kd["gene_q"] = kd["q"].str.split("_", n=1).str[0].str.replace("+YDL227C", "", regex=False)
    kd["gene_q"] = kd["gene_q"].str.replace("YDL227C+", "", regex=False)
    kd["gene_a"] = _gene(kd["a"])
    key = [frozenset((x, y)) for x, y in zip(kd["gene_q"], kd["gene_a"])]
    keep = np.fromiter((kk in pairs for kk in key), dtype=bool, count=len(key))
    kd = kd[keep].rename(columns={"score": "eps"})[
        ["q", "a", "gene_q", "gene_a", "eps", "eps_raw", "p", "f_q", "f_a", "f_qa", "sd_qa", "file", "source"]
    ]
    print(f"  kuzmin digenic rows on closure pairs: {len(kd):,}", flush=True)
    allc = pd.concat(parts + [kd], ignore_index=True)
    allc.to_parquet(osp.join(CACHE_DIR, "raw_digenic_closure.parquet"), index=False)
    print(f"raw digenic rows on closure pairs: {len(allc):,}", flush=True)


# --------------------------------------------------------------------------- analyze
def _stats(x: np.ndarray, y: np.ndarray) -> dict:
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return {"n": int(len(x))}
    lr = stats.linregress(x, y)
    return {
        "n": int(len(x)),
        "pearson": float(stats.pearsonr(x, y)[0]),
        "spearman": float(stats.spearmanr(x, y)[0]),
        "slope": float(lr.slope),
        "intercept": float(lr.intercept),
        "rmse": float(np.sqrt(np.mean((x - y) ** 2))),
        "median_abs_residual": float(np.median(np.abs(x - y))),
    }


def _calls(score: np.ndarray, p: np.ndarray, cut: float = 0.08) -> np.ndarray:
    return (np.abs(score) > cut) & (p < 0.05)


def _call_table(stored: np.ndarray, recomputed: np.ndarray) -> dict:
    m = np.isfinite(stored) & np.isfinite(recomputed)
    s, r = stored[m], recomputed[m]
    both = s & r
    return {
        "n": int(m.sum()),
        "stored_called": int(s.sum()),
        "recomputed_called": int(r.sum()),
        "both": int(both.sum()),
        "recall_of_stored": float(both.sum() / s.sum()) if s.sum() else None,
        "precision_vs_stored": float(both.sum() / r.sum()) if r.sum() else None,
        "jaccard": float(both.sum() / (s | r).sum()) if (s | r).sum() else None,
    }


def z_p(score: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Two-sided normal p of a score whose null sd is sigma (Kuzmin 2020 SI, Eq. for
    query-query interactions; two-sided so |score| enters as in the loaders' back-solve)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return 2.0 * stats.norm.sf(np.abs(score) / sigma)


def analyze() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from s3_closure_plots import make_figures, write_tables  # noqa: E402

    df = pd.read_parquet(osp.join(CACHE_DIR, "s3_records.parquet"))
    singles = df[df["order"] == 1].set_index("genes").copy()
    # An essentiality entry enters the mean as a fitness of 0 (CompositeFitnessConverter).
    # With k merged entries of which exactly one is that 0, the measured mean is
    # k/(k-1) times the stored mean; a single with no measured entry (k = 0 duplicates)
    # is "essential only" and its stored fitness is 0 with no measurement behind it.
    singles["ess_only"] = singles["has_ess"] & (singles["fit_dup"] == 0)
    k = singles["fit_dup"].to_numpy(dtype=float)
    singles["fit_measured"] = np.where(
        singles["has_ess"] & (k >= 2), singles["fit_mean"] * k / np.maximum(k - 1, 1), singles["fit_mean"]
    )
    singles.loc[singles["ess_only"], "fit_measured"] = np.nan
    doubles = df[df["order"] == 2].copy()
    triples = df[df["order"] == 3].copy()
    summary: dict = {"n_records": int(len(df)), "n_by_order": df["order"].value_counts().sort_index().to_dict()}

    # ---- composition: what the pool joined
    comp = (
        df.assign(fit_src=df["fit_ds"].map(lambda s: "+".join(sorted(set(s.split("+"))))))
        .groupby(["order", "fit_src"])
        .size()
        .reset_index(name="n")
        .sort_values(["order", "n"], ascending=[True, False])
    )
    comp.to_csv(osp.join(RESULTS_DIR, "s3_closure_composition.csv"), index=False)
    summary["fitness_conflicts"] = {
        "n_records_two_fitness_entries": int((df["fit_n_entries"] > 1).sum()),
        "n_singles_with_essentiality_entry": int(singles["has_ess"].sum()),
        "n_singles_with_synthleth_entry": int(singles["has_sl"].sum()),
        "n_doubles_with_synthleth_entry": int(doubles["has_sl"].sum()),
        "n_doubles_merged_interaction": int((doubles["gi_dup"] > 1).sum()),
        "n_triples_merged_interaction": int((triples["gi_dup"] > 1).sum()),
    }

    # ---- strength: digenic
    ga = doubles["genes"].str.split("|").str[0]
    gb = doubles["genes"].str.split("|").str[1]
    fa, fb = singles["fit_mean"].reindex(ga).to_numpy(), singles["fit_mean"].reindex(gb).to_numpy()
    sa, sb = singles["fit_sd"].reindex(ga).to_numpy(), singles["fit_sd"].reindex(gb).to_numpy()
    ess_a, ess_b = singles["has_ess"].reindex(ga).to_numpy(), singles["has_ess"].reindex(gb).to_numpy()
    doubles["f_a"], doubles["f_b"] = fa, fb
    doubles["eps_rec"] = doubles["fit_mean"].to_numpy() - fa * fb
    doubles["ess_single"] = np.nan_to_num(ess_a.astype(float)) + np.nan_to_num(ess_b.astype(float)) > 0
    eo_a, eo_b = singles["ess_only"].reindex(ga).to_numpy(), singles["ess_only"].reindex(gb).to_numpy()
    doubles["ess_only_single"] = np.nan_to_num(eo_a.astype(float)) + np.nan_to_num(eo_b.astype(float)) > 0
    # the same identity with the essentiality 0 taken back out of the singles
    ma, mb = singles["fit_measured"].reindex(ga).to_numpy(), singles["fit_measured"].reindex(gb).to_numpy()
    doubles["eps_rec_measured"] = doubles["fit_mean"].to_numpy() - ma * mb
    doubles["merged"] = doubles["gi_dup"] > 1
    d_ok = doubles[np.isfinite(doubles["eps_rec"]) & np.isfinite(doubles["gi"])]

    def _d(mask, col="eps_rec"):
        return _stats(d_ok.loc[mask, "gi"].to_numpy(), d_ok.loc[mask, col].to_numpy())

    summary["digenic_strength"] = {
        "all": _stats(d_ok["gi"].to_numpy(), d_ok["eps_rec"].to_numpy()),
        "single_screen": _d(~d_ok["merged"]),
        "merged_screens": _d(d_ok["merged"]),
        "no_essential_single": _d(~d_ok["ess_single"]),
        "essential_single": _d(d_ok["ess_single"]),
        "essential_single_measured_mean_restored": _d(d_ok["ess_single"] & ~d_ok["ess_only_single"], "eps_rec_measured"),
        "essential_only_single": _d(d_ok["ess_only_single"]),
        "no_synthleth": _d(~d_ok["has_sl"]),
    }

    # ---- confidence: digenic. Null sd of eps propagated from the stored uncertainties:
    # var(f_ab) = sd_ab^2 / n colonies ; var(f_a f_b) = f_b^2 sd_a^2 + f_a^2 sd_b^2.
    n_ab = doubles["fit_n_samples"].fillna(N_COLONIES_DEFAULT).to_numpy(dtype=float)
    sd_ab = doubles["fit_sd"].to_numpy()
    var_obs = sd_ab**2 / n_ab
    var_exp = np.nan_to_num(fb**2 * sa**2) + np.nan_to_num(fa**2 * sb**2)
    # Three z-tests, all two-sided normal on the STORED score:
    #   obs        null sd = sd_ab / sqrt(n)               (the double's own colony sd)
    #   prop       null sd adds the singles' stored sd through f_a f_b
    #   obs_rec    as obs, but on the RECOMPUTED eps (strength and confidence together)
    doubles["p_rec_obs"] = z_p(doubles["gi"].to_numpy(), np.sqrt(var_obs))
    doubles["p_rec_prop"] = z_p(doubles["gi"].to_numpy(), np.sqrt(var_obs + var_exp))
    doubles["p_rec_obs_on_rec"] = z_p(doubles["eps_rec"].to_numpy(), np.sqrt(var_obs))
    sgl, mrg = ~doubles["merged"], doubles["merged"]

    def _rho(mask, col):
        m = mask & np.isfinite(doubles["gi_p"]) & np.isfinite(doubles[col])
        return float(stats.spearmanr(doubles.loc[m, "gi_p"], doubles.loc[m, col])[0])

    def _calls_vs(mask, score_col, p_col):
        return _call_table(
            _calls(doubles.loc[mask, "gi"].to_numpy(), doubles.loc[mask, "gi_p"].to_numpy()),
            _calls(doubles.loc[mask, score_col].to_numpy(), doubles.loc[mask, p_col].to_numpy()),
        )

    summary["digenic_confidence"] = {
        "n_single_screen": int(sgl.sum()),
        "n_merged": int(mrg.sum()),
        "spearman_single_screen_stored_vs_obs": _rho(sgl, "p_rec_obs"),
        "spearman_single_screen_stored_vs_propagated": _rho(sgl, "p_rec_prop"),
        "spearman_merged_stored_vs_obs": _rho(mrg, "p_rec_obs"),
        "spearman_merged_stored_vs_propagated": _rho(mrg, "p_rec_prop"),
        "calls_single_screen_stored_vs_obs": _calls_vs(sgl, "gi", "p_rec_obs"),
        "calls_single_screen_stored_vs_obs_on_recomputed_eps": _calls_vs(sgl, "eps_rec", "p_rec_obs_on_rec"),
        "calls_single_screen_stored_vs_propagated": _calls_vs(sgl, "gi", "p_rec_prop"),
        "calls_merged_stored_vs_obs": _calls_vs(mrg, "gi", "p_rec_obs"),
        "frac_p_below_0.05_single_screen": float((doubles.loc[sgl, "gi_p"] < 0.05).mean()),
        "frac_p_below_0.05_merged": float((doubles.loc[mrg, "gi_p"] < 0.05).mean()),
        "frac_obs_p_below_0.05_single_screen": float((doubles.loc[sgl, "p_rec_obs"] < 0.05).mean()),
    }

    # ---- the merged records against the p-values their sources reported
    rawd = pd.read_parquet(osp.join(CACHE_DIR, "raw_digenic_closure.parquet"))
    rawd["pair"] = [
        "|".join(sorted((x, y))) for x, y in zip(rawd["gene_q"], rawd["gene_a"])
    ]
    # Kuzmin SI: "all NaN fitness estimates were assigned a value of 1.0" during scoring,
    # so the within-screen identity is checked with that substitution and the fraction
    # of rows it touches is reported (hazard H5).
    rawd["eps_rec_raw"] = rawd["f_qa"] - rawd["f_q"].fillna(1.0) * rawd["f_a"].fillna(1.0)
    summary["within_screen_control"] = {
        src: _stats(g["eps"].to_numpy(), g["eps_rec_raw"].to_numpy())
        | {
            "frac_query_fitness_nan": float(g["f_q"].isna().mean()),
            "frac_array_fitness_nan": float(g["f_a"].isna().mean()),
        }
        for src, g in rawd.groupby("source")
    }
    summary["within_screen_control"]["kuzmin_eps_raw_column_vs_identity"] = {
        src: _stats(g["eps_raw"].to_numpy(), g["eps_rec_raw"].to_numpy())
        for src, g in rawd[rawd["source"].str.startswith("kuzmin")].groupby("source")
    }
    src_agg = rawd.groupby("pair").agg(
        n_source_rows=("p", "size"),
        p_source_min=("p", "min"),
        p_source_median=("p", "median"),
        p_source_max=("p", "max"),
        eps_source_mean=("eps", "mean"),
        eps_source_min=("eps", "min"),
        eps_source_max=("eps", "max"),
        n_sources=("source", "nunique"),
    )
    dm = doubles.merge(src_agg, left_on="genes", right_index=True, how="left")
    merged = dm[dm["merged"] & np.isfinite(dm["p_source_min"])]
    summary["merged_p_against_sources"] = {
        "n_merged_with_source_rows": int(len(merged)),
        "n_merged_all": int(dm["merged"].sum()),
        "spearman_stored_vs_source_median_p": float(stats.spearmanr(merged["gi_p"], merged["p_source_median"])[0]),
        "frac_stored_p_below_0.05": float((merged["gi_p"] < 0.05).mean()),
        "frac_source_median_p_below_0.05": float((merged["p_source_median"] < 0.05).mean()),
        "frac_all_source_rows_below_0.05": float((merged["p_source_max"] < 0.05).mean()),
        "frac_all_sources_significant_but_stored_not": float(
            ((merged["p_source_max"] < 0.05) & (merged["gi_p"] >= 0.05)).mean()
        ),
        "n_all_sources_significant_but_stored_not": int(
            ((merged["p_source_max"] < 0.05) & (merged["gi_p"] >= 0.05)).sum()
        ),
        "calls_stored_vs_source_median": _call_table(
            _calls(merged["gi"].to_numpy(), merged["gi_p"].to_numpy()),
            _calls(merged["eps_source_mean"].to_numpy(), merged["p_source_median"].to_numpy()),
        ),
        "single_screen_stored_equals_source": {
            "n": int((~dm["merged"] & np.isfinite(dm["p_source_min"])).sum()),
            "max_abs_p_diff": float(
                np.nanmax(np.abs(dm.loc[~dm["merged"], "gi_p"] - dm.loc[~dm["merged"], "p_source_median"]))
            ),
        },
    }

    # ---- strength: trigenic
    tg = triples["genes"].str.split("|", expand=True)
    tg.columns = ["a", "b", "c"]
    for k in "abc":
        triples[f"f_{k}"] = singles["fit_mean"].reindex(tg[k]).to_numpy()
        triples[f"sd_{k}"] = singles["fit_sd"].reindex(tg[k]).to_numpy()
        triples[f"ess_{k}"] = singles["has_ess"].reindex(tg[k]).to_numpy()
        triples[f"esso_{k}"] = singles["ess_only"].reindex(tg[k]).to_numpy()
        triples[f"fm_{k}"] = singles["fit_measured"].reindex(tg[k]).to_numpy()
    dbl = doubles.set_index("genes")
    for x, y in (("a", "b"), ("a", "c"), ("b", "c")):
        key = ["|".join(sorted((p, q))) for p, q in zip(tg[x], tg[y])]
        triples[f"f_{x}{y}"] = dbl["fit_mean"].reindex(key).to_numpy()
        triples[f"sd_{x}{y}"] = dbl["fit_sd"].reindex(key).to_numpy()
        triples[f"n_{x}{y}"] = dbl["fit_n_samples"].reindex(key).to_numpy()
        triples[f"eps_{x}{y}_stored"] = dbl["gi"].reindex(key).to_numpy()
    fa, fb, fc = (triples[f"f_{k}"].to_numpy() for k in "abc")
    f_ab, f_ac, f_bc = (triples[f"f_{k}"].to_numpy() for k in ("ab", "ac", "bc"))
    f_abc = triples["fit_mean"].to_numpy()
    triples["tau_rec"] = f_abc - f_ab * fc - f_ac * fb - f_bc * fa + 2.0 * fa * fb * fc
    e_ab, e_ac, e_bc = (triples[f"eps_{k}_stored"].to_numpy() for k in ("ab", "ac", "bc"))
    triples["tau_dmi"] = f_abc - e_ab * fc - e_ac * fb - e_bc * fa - fa * fb * fc
    ma, mb, mc = (triples[f"fm_{k}"].to_numpy() for k in "abc")
    triples["tau_rec_measured"] = f_abc - f_ab * mc - f_ac * mb - f_bc * ma + 2.0 * ma * mb * mc
    triples["ess_single"] = triples[["ess_a", "ess_b", "ess_c"]].fillna(False).any(axis=1)
    triples["ess_only_single"] = triples[["esso_a", "esso_b", "esso_c"]].fillna(False).any(axis=1)
    triples["merged"] = triples["gi_dup"] > 1
    t_ok = triples[np.isfinite(triples["tau_rec"]) & np.isfinite(triples["gi"])]

    def _t(mask, col="tau_rec"):
        return _stats(t_ok.loc[mask, "gi"].to_numpy(), t_ok.loc[mask, col].to_numpy())

    summary["trigenic_strength"] = {
        "n_triples_full_closure": int(np.isfinite(triples["tau_rec"]).sum()),
        "all": _stats(t_ok["gi"].to_numpy(), t_ok["tau_rec"].to_numpy()),
        "single_screen": _t(~t_ok["merged"]),
        "merged_screens": _t(t_ok["merged"]),
        "no_essential_single": _t(~t_ok["ess_single"]),
        "essential_single": _t(t_ok["ess_single"]),
        "essential_single_measured_mean_restored": _t(t_ok["ess_single"] & ~t_ok["ess_only_single"], "tau_rec_measured"),
        "essential_only_single": _t(t_ok["ess_only_single"]),
        "from_stored_dmi": _stats(triples["gi"].to_numpy(), triples["tau_dmi"].to_numpy()),
        "calls_stored_vs_recomputed_at_0.08": _call_table(
            np.abs(t_ok["gi"].to_numpy()) > 0.08, np.abs(t_ok["tau_rec"].to_numpy()) > 0.08
        ),
    }

    # ---- confidence: trigenic. First-order propagation of every term's uncertainty.
    def v(sd: np.ndarray, n: np.ndarray | None = None) -> np.ndarray:
        sd2 = np.nan_to_num(sd, nan=0.0) ** 2
        return sd2 / n if n is not None else sd2

    n_abc = triples["fit_n_samples"].fillna(N_COLONIES_DEFAULT).to_numpy(dtype=float)
    var_tau = (
        v(triples["fit_sd"].to_numpy(), n_abc)
        + v(triples["sd_ab"].to_numpy(), triples["n_ab"].fillna(N_COLONIES_DEFAULT).to_numpy(dtype=float)) * fc**2
        + v(triples["sd_ac"].to_numpy(), triples["n_ac"].fillna(N_COLONIES_DEFAULT).to_numpy(dtype=float)) * fb**2
        + v(triples["sd_bc"].to_numpy(), triples["n_bc"].fillna(N_COLONIES_DEFAULT).to_numpy(dtype=float)) * fa**2
        + v(triples["sd_a"].to_numpy()) * (f_bc - 2 * fb * fc) ** 2
        + v(triples["sd_b"].to_numpy()) * (f_ac - 2 * fa * fc) ** 2
        + v(triples["sd_c"].to_numpy()) * (f_ab - 2 * fa * fb) ** 2
    )
    triples["p_rec_prop"] = z_p(triples["tau_rec"].to_numpy(), np.sqrt(var_tau))
    triples["p_rec_obs"] = z_p(triples["gi"].to_numpy(), np.sqrt(v(triples["fit_sd"].to_numpy(), n_abc)))
    tm = np.isfinite(triples["gi_p"]) & np.isfinite(triples["p_rec_prop"]) & ~triples["merged"]
    summary["trigenic_confidence"] = {
        "n": int(tm.sum()),
        "spearman_p_stored_vs_propagated": float(stats.spearmanr(triples.loc[tm, "gi_p"], triples.loc[tm, "p_rec_prop"])[0]),
        "spearman_p_stored_vs_obs_only": float(stats.spearmanr(triples.loc[tm, "gi_p"], triples.loc[tm, "p_rec_obs"])[0]),
        "frac_stored_p_below_0.05": float((triples.loc[tm, "gi_p"] < 0.05).mean()),
        "frac_propagated_p_below_0.05": float((triples.loc[tm, "p_rec_prop"] < 0.05).mean()),
        "calls_stored_vs_propagated": _call_table(
            _calls(triples.loc[tm, "gi"].to_numpy(), triples.loc[tm, "gi_p"].to_numpy()),
            _calls(triples.loc[tm, "tau_rec"].to_numpy(), triples.loc[tm, "p_rec_prop"].to_numpy()),
        ),
        "calls_stored_vs_obs_only_on_stored_tau": _call_table(
            _calls(triples.loc[tm, "gi"].to_numpy(), triples.loc[tm, "gi_p"].to_numpy()),
            _calls(triples.loc[tm, "gi"].to_numpy(), triples.loc[tm, "p_rec_obs"].to_numpy()),
        ),
        "n_merged_triples": int(triples["merged"].sum()),
    }
    # merged triples: the stored p is a t-test over 2 values; the sources' p-values
    rawk = pd.read_parquet(osp.join(CACHE_DIR, "raw_kuzmin_all.parquet"))
    rawt = rawk[rawk["type"] == "trigenic"].copy()
    qq = rawt["q"].str.split("_", n=1).str[0].str.split("+")
    rawt["genes"] = [
        "|".join(sorted([*pair, g])) for pair, g in zip(qq, _gene(rawt["a"]))
    ]
    tsrc = rawt.groupby("genes").agg(
        n_source_rows=("p", "size"), p_source_min=("p", "min"), p_source_max=("p", "max"),
        p_source_median=("p", "median"), tau_source_mean=("score", "mean"),
    )
    tmerged = triples[triples["merged"]].merge(tsrc, left_on="genes", right_index=True, how="left")
    tmerged = tmerged[np.isfinite(tmerged["p_source_min"])]
    summary["merged_triples_against_sources"] = {
        "n": int(len(tmerged)),
        "frac_stored_p_below_0.05": float((tmerged["gi_p"] < 0.05).mean()),
        "frac_source_median_p_below_0.05": float((tmerged["p_source_median"] < 0.05).mean()),
        "n_all_sources_significant_but_stored_not": int(
            ((tmerged["p_source_max"] < 0.05) & (tmerged["gi_p"] >= 0.05)).sum()
        ),
        "n_all_sources_significant": int((tmerged["p_source_max"] < 0.05).sum()),
    }
    # single-screen triples: the stored p equals the source p (identity check of the join)
    tsingle = triples[~triples["merged"]].merge(tsrc, left_on="genes", right_index=True, how="left")
    ok = np.isfinite(tsingle["p_source_median"])
    summary["trigenic_confidence"]["single_screen_stored_equals_source"] = {
        "n": int(ok.sum()),
        "max_abs_p_diff": float(np.nanmax(np.abs(tsingle.loc[ok, "gi_p"] - tsingle.loc[ok, "p_source_median"]))),
        "max_abs_tau_diff": float(np.nanmax(np.abs(tsingle.loc[ok, "gi"] - tsingle.loc[ok, "tau_source_mean"]))),
    }

    # ---- what the essentiality and synthetic-lethality entries do to a single's fitness
    ess = singles[singles["has_ess"]]
    with_meas = ess[~ess["ess_only"]]
    summary["essentiality_in_singles"] = {
        "n": int(len(ess)),
        "n_essential_only_no_measurement": int(ess["ess_only"].sum()),
        "n_with_measured_entries": int(len(with_meas)),
        "stored_fitness_median_with_measured": float(with_meas["fit_mean"].median()),
        "measured_fitness_median_reconstructed": float(with_meas["fit_measured"].median()),
        "n_doubles_touching": int(doubles["ess_single"].sum()),
        "n_doubles_touching_essential_only": int(doubles["ess_only_single"].sum()),
        "n_triples_touching": int(triples["ess_single"].sum()),
        "n_triples_touching_essential_only": int(triples["ess_only_single"].sum()),
    }
    singles.reset_index().to_parquet(osp.join(CACHE_DIR, "s3_singles_annotated.parquet"), index=False)

    doubles.to_parquet(osp.join(CACHE_DIR, "s3_doubles_recomputed.parquet"), index=False)
    triples.to_parquet(osp.join(CACHE_DIR, "s3_triples_recomputed.parquet"), index=False)
    dm.to_parquet(osp.join(CACHE_DIR, "s3_doubles_with_sources.parquet"), index=False)
    with open(osp.join(RESULTS_DIR, "s3_closure_recompute_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))
    write_tables(summary, comp, TABLE_DIR)
    make_figures(singles, doubles, triples, dm, tmerged, rawd, summary, IMG_DIR)


if __name__ == "__main__":
    sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
    stage = sys.argv[1] if len(sys.argv) > 1 else "analyze"
    {"scan": scan, "raw": raw, "analyze": analyze}[stage]()
