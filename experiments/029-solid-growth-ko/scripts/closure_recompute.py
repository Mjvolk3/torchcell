# experiments/029-solid-growth-ko/scripts/closure_recompute.py
# [[experiments.029-solid-growth-ko.scripts.closure_recompute]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/029-solid-growth-ko/scripts/closure_recompute

"""Recompute the interaction scores of the 029 deletion-only build from its own fitness,
under explicit label policies, and set the result beside the 025 build.

The 029 build (experiments/029-solid-growth-ko/queries/001_ko_solid_growth.cql, build
001) keeps every source entry of a genotype: no merge, so a single carries its Costanzo
26 C and 30 C entries at both markers, its Kuzmin query fitness where one exists, and an
SGD essentiality entry converted to 0 where the gene is essential; a double carries both
Costanzo orientations and every Kuzmin digenic screen; a triple carries one entry per
Kuzmin screen. What the trainer reads is therefore a POLICY over those entries, and this
script scores the trigenic identity

    tau_abc = f_abc - f_ab f_c - f_ac f_b - f_bc f_a + 2 f_a f_b f_c

and the digenic identity eps_ab = f_ab - f_a f_b under four of them:

    mean_all        the mean over every entry, essentiality 0 included: the 025 join
                    replayed on the deletion-only universe
    measured_else_0 the mean over measured entries; the converted 0 only when the gene
                    has no measurement at all
    kuzmin_first    the Kuzmin screen of the same year, then the other Kuzmin screen,
                    then Costanzo 30 C, then 26 C, then the converted 0
    costanzo_first  Costanzo 30 C, then 26 C, then Kuzmin, then the converted 0

Each policy's recomputed score is compared with every STORED interaction entry (each
entry is one source screen; in this build a stored p-value is the source p-value by
construction). The pool is the closure of the 029 triples: every single, every double
whose pair lies inside some 029 triple, every triple.

Stages (cache under $DATA_ROOT/data/torchcell/experiments/029-solid-growth-ko/closure/):

    python experiments/029-solid-growth-ko/scripts/closure_recompute.py scan     # build -> entries.parquet
    python experiments/029-solid-growth-ko/scripts/closure_recompute.py analyze  # results, tables, figure

Outputs (experiments/029-solid-growth-ko/results/):
- closure_recompute_summary.json
- closure_policy_stats.csv              (one row per policy x order x stored source)
- pinned_triple_survival.json           (the 010 pinned val/test triples present in 029)
Tables notes-tex/025-s3-closure/tables/t6-query-comparison.tex, t7-survival.tex and the
figure $ASSET_IMAGES_DIR/029-solid-growth-ko/closure_query_comparison.svg|png.
"""

import gzip
import json
import os
import os.path as osp
import re
import sys
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

BUILD = osp.join(DATA_ROOT, "data/torchcell/experiments/029-solid-growth-ko/001-ko-build/processed")
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "029-solid-growth-ko/results")
CACHE_DIR = osp.join(DATA_ROOT, "data/torchcell/experiments/029-solid-growth-ko/closure")
IMG_DIR = osp.join(ASSET_IMAGES_DIR, "029-solid-growth-ko")
TABLE_DIR = osp.join(osp.dirname(EXPERIMENT_ROOT), "notes-tex/025-s3-closure/tables")
RESULTS_025 = osp.join(EXPERIMENT_ROOT, "025-solid-growth/results")
CACHE_025 = osp.join(DATA_ROOT, "data/torchcell/experiments/025-solid-growth/s3_closure")
N_COLONIES_DEFAULT = 4
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

ENTRY_COLS = ["idx", "order", "genes", "dataset", "exp_type", "temp", "marker", "value", "sd", "n_samples", "p"]
POLICIES = ["mean_all", "measured_else_0", "kuzmin_first", "costanzo_first"]
PRECEDENCE = {
    ("kuzmin_first", "K18"): ["K18", "K20", "C30", "C26", "Z"],
    ("kuzmin_first", "K20"): ["K20", "K18", "C30", "C26", "Z"],
    ("costanzo_first", "K18"): ["C30", "C26", "K18", "K20", "Z"],
    ("costanzo_first", "K20"): ["C30", "C26", "K20", "K18", "Z"],
}

# --------------------------------------------------------------------------- scan
_ENV = None
_GENE_RE = re.compile(rb'"systematic_gene_name":\s*"([^"]*)"')


def _init_worker() -> None:
    global _ENV
    _ENV = lmdb.open(osp.join(BUILD, "lmdb"), readonly=True, lock=False, max_readers=128)


def _gene_key_bytes(raw: bytes) -> str:
    head = raw[: raw.find(b'"experiment_reference"')]
    return "|".join(sorted({g.decode() for g in _GENE_RE.findall(head)}))


def _entries(idx: int, raw: bytes) -> list[tuple]:
    ent = json.loads(raw)
    rows = []
    for item in ent:
        e = item["experiment"]
        ph = e["phenotype"]
        perts = sorted(e["genotype"]["perturbations"], key=lambda p: p["systematic_gene_name"])
        genes = "|".join(p["systematic_gene_name"] for p in perts)
        marker = "|".join(p["perturbation_type"].replace("sga_", "").replace("_deletion", "") for p in perts)
        temp = float(e["environment"]["temperature"]["value"])
        if e["experiment_type"] == "fitness":
            rows.append((idx, len(perts), genes, e["dataset_name"], "fitness", temp, marker,
                         ph["fitness"], ph.get("fitness_std"), ph.get("n_samples"), None))
        elif e["experiment_type"] == "gene interaction":
            rows.append((idx, len(perts), genes, e["dataset_name"], "gene interaction", temp, marker,
                         ph["gene_interaction"], None, None, ph.get("gene_interaction_p_value")))
    return rows


def _keys_chunk(idxs: list[int]) -> list[tuple[int, str]]:
    assert _ENV is not None
    with _ENV.begin() as txn:
        return [(i, _gene_key_bytes(txn.get(str(i).encode()))) for i in idxs]


def _entries_chunk(idxs: list[int]) -> list[tuple]:
    assert _ENV is not None
    out: list[tuple] = []
    with _ENV.begin() as txn:
        for i in idxs:
            out.extend(_entries(i, txn.get(str(i).encode())))
    return out


def _run(fn, idxs: list[int], label: str, chunk: int = 2000) -> list:
    chunks = [idxs[i : i + chunk] for i in range(0, len(idxs), chunk)]
    rows: list = []
    with Pool(48, initializer=_init_worker) as pool:
        for n, part in enumerate(pool.imap_unordered(fn, chunks, chunksize=1)):
            rows.extend(part)
            if n % 200 == 0:
                print(f"  {label}: chunks {n}/{len(chunks)}", flush=True)
    return rows


def scan() -> None:
    with open(osp.join(BUILD, "perturbation_count_index.json")) as f:
        by_order = json.load(f)
    print({k: len(v) for k, v in by_order.items()}, flush=True)
    triples = _run(_entries_chunk, by_order["3"], "triples")
    tdf = pd.DataFrame(triples, columns=ENTRY_COLS)
    pairs: set[str] = set()
    for g in tdf["genes"].unique():
        a, b, c = g.split("|")
        pairs.update({f"{a}|{b}", f"{a}|{c}", f"{b}|{c}"})
    print(f"triple gene sets {tdf['genes'].nunique():,}; closure pairs {len(pairs):,}", flush=True)
    keys = _run(_keys_chunk, by_order["2"], "double keys")
    keep = [i for i, k in keys if k in pairs]
    print(f"doubles in the closure: {len(keep):,} of {len(keys):,}", flush=True)
    doubles = _run(_entries_chunk, keep, "closure doubles")
    singles = _run(_entries_chunk, by_order["1"], "singles")
    df = pd.concat(
        [pd.DataFrame(singles, columns=ENTRY_COLS), pd.DataFrame(doubles, columns=ENTRY_COLS), tdf],
        ignore_index=True,
    )
    df.to_parquet(osp.join(CACHE_DIR, "entries.parquet"), index=False)
    print(df.groupby(["order", "exp_type"]).size().to_string(), flush=True)
    print(df.groupby(["order", "dataset"]).size().to_string(), flush=True)


# --------------------------------------------------------------------------- analyze
def _cls(ds: str, temp: float) -> str:
    if "Kuzmin2018" in ds:
        return "K18"
    if "Kuzmin2020" in ds:
        return "K20"
    if "Costanzo2016" in ds:
        return "C30" if temp == 30.0 else "C26"
    if "Essentiality" in ds or "SynthLeth" in ds:
        return "Z"  # a converted 0, no measurement behind it
    raise ValueError(ds)


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


def _policy_table(fit: pd.DataFrame) -> pd.DataFrame:
    """Per gene set: one fitness value per policy (and per Kuzmin year for the ordered
    policies), plus the sd and n behind the entries used by mean_all."""
    piv = fit.pivot_table(index="genes", columns="cls", values="value", aggfunc="mean")
    for c in ["K18", "K20", "C30", "C26", "Z"]:
        if c not in piv:
            piv[c] = np.nan
    out = pd.DataFrame(index=piv.index)
    out["mean_all"] = fit.groupby("genes")["value"].mean()
    measured = fit[fit["cls"] != "Z"].groupby("genes")["value"].mean()
    out["measured_else_0"] = measured.reindex(piv.index)
    has_zero = piv["Z"].notna()
    out.loc[out["measured_else_0"].isna() & has_zero, "measured_else_0"] = 0.0
    for (pol, yr), order in PRECEDENCE.items():
        out[f"{pol}_{yr}"] = piv[order].bfill(axis=1).iloc[:, 0]
    out["sd_rms"] = fit.groupby("genes")["sd"].apply(lambda s: float(np.sqrt(np.nanmean(np.square(s.astype(float))))) if s.notna().any() else np.nan)
    out["n_samples"] = fit.groupby("genes")["n_samples"].max()
    out["has_zero"] = has_zero
    out["n_measured"] = fit[fit["cls"] != "Z"].groupby("genes").size().reindex(piv.index).fillna(0).astype(int)
    return out


def _value_for(table: pd.DataFrame, keys: pd.Series, policy: str, year: pd.Series) -> np.ndarray:
    if policy in ("mean_all", "measured_else_0"):
        return table[policy].reindex(keys).to_numpy()
    v18 = table[f"{policy}_K18"].reindex(keys).to_numpy()
    v20 = table[f"{policy}_K20"].reindex(keys).to_numpy()
    return np.where(year.to_numpy() == "K18", v18, v20)


def analyze() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from closure_plots import make_figure, write_tables  # noqa: E402

    df = pd.read_parquet(osp.join(CACHE_DIR, "entries.parquet"))
    df["cls"] = [_cls(d, t) for d, t in zip(df["dataset"], df["temp"])]
    fit = df[df["exp_type"] == "fitness"]
    gi = df[df["exp_type"] == "gene interaction"].copy()
    summary: dict = {
        "n_records_by_order": df.groupby("order")["idx"].nunique().to_dict(),
        "n_entries_by_order_type": {f"{o}/{t}": int(n) for (o, t), n in df.groupby(["order", "exp_type"]).size().items()},
        "n_entries_by_order_dataset": {f"{o}/{d}": int(n) for (o, d), n in df.groupby(["order", "dataset"]).size().items()},
        "stored_p_is_source_p": {
            "n_interaction_entries": int(len(gi)),
            "n_with_p": int(gi["p"].notna().sum()),
            "note": "no merge stage in build 001: every interaction entry is one source row and keeps its p-value",
        },
    }
    single_t = _policy_table(fit[fit["order"] == 1])
    double_t = _policy_table(fit[fit["order"] == 2])
    triple_t = _policy_table(fit[fit["order"] == 3])
    summary["singles"] = {
        "n": int(len(single_t)),
        "n_with_converted_zero": int(single_t["has_zero"].sum()),
        "n_zero_only_no_measurement": int((single_t["has_zero"] & (single_t["n_measured"] == 0)).sum()),
        "n_kuzmin_query_fitness": int(fit[(fit["order"] == 1) & fit["cls"].isin(["K18", "K20"])]["genes"].nunique()),
    }

    # ---- digenic: every stored eps entry against the policy's recomputed eps
    gd = gi[gi["order"] == 2].copy()
    gd["a"], gd["b"] = gd["genes"].str.split("|").str[0], gd["genes"].str.split("|").str[1]
    gd["year"] = np.where(gd["cls"] == "K18", "K18", "K20")  # Costanzo entries take the 2020 ordering
    rows = []
    for pol in POLICIES:
        fa = _value_for(single_t, gd["a"], pol, gd["year"])
        fb = _value_for(single_t, gd["b"], pol, gd["year"])
        fab = _value_for(double_t, gd["genes"], pol, gd["year"])
        gd[f"eps_{pol}"] = fab - fa * fb
        st = _stats(gd["value"].to_numpy(), gd[f"eps_{pol}"].to_numpy())
        rows.append({"policy": pol, "order": 2, "stored_source": "all", **st})
        for src, g in gd.groupby("cls"):
            rows.append({"policy": pol, "order": 2, "stored_source": src, **_stats(g["value"].to_numpy(), g[f"eps_{pol}"].to_numpy())})
    # ---- trigenic
    gt = gi[gi["order"] == 3].copy()
    tg = gt["genes"].str.split("|", expand=True)
    tg.columns = ["a", "b", "c"]
    gt["year"] = gt["cls"]
    for pol in POLICIES:
        f = {k: _value_for(single_t, tg[k], pol, gt["year"]) for k in "abc"}
        fd = {}
        for x, y in (("a", "b"), ("a", "c"), ("b", "c")):
            key = pd.Series(["|".join(sorted((p, q))) for p, q in zip(tg[x], tg[y])], index=gt.index)
            fd[x + y] = _value_for(double_t, key, pol, gt["year"])
        fabc = _value_for(triple_t, gt["genes"], pol, gt["year"])
        gt[f"tau_{pol}"] = (
            fabc - fd["ab"] * f["c"] - fd["ac"] * f["b"] - fd["bc"] * f["a"] + 2.0 * f["a"] * f["b"] * f["c"]
        )
        rows.append({"policy": pol, "order": 3, "stored_source": "all", **_stats(gt["value"].to_numpy(), gt[f"tau_{pol}"].to_numpy())})
        for src, g in gt.groupby("cls"):
            rows.append({"policy": pol, "order": 3, "stored_source": src, **_stats(g["value"].to_numpy(), g[f"tau_{pol}"].to_numpy())})
    pol_stats = pd.DataFrame(rows)
    pol_stats.to_csv(osp.join(RESULTS_DIR, "closure_policy_stats.csv"), index=False)
    summary["policy_stats"] = {
        f"{r.policy}/{r.order}/{r.stored_source}": {k: r._asdict()[k] for k in pol_stats.columns if k not in ("policy", "order", "stored_source") and pd.notna(r._asdict()[k])}
        for r in pol_stats.itertuples(index=False)
    }
    summary["calls_trigenic_at_0.08"] = {
        pol: {
            "n": int(np.isfinite(gt[f"tau_{pol}"]).sum()),
            "stored_called": int((np.abs(gt["value"]) > 0.08).sum()),
            "recomputed_called": int((np.abs(gt[f"tau_{pol}"]) > 0.08).sum()),
            "both": int(((np.abs(gt["value"]) > 0.08) & (np.abs(gt[f"tau_{pol}"]) > 0.08)).sum()),
        }
        for pol in POLICIES
    }

    # ---- confidence: stored p (= source p) against a z-test from the record's own sd
    def z_p(score, sigma):
        with np.errstate(divide="ignore", invalid="ignore"):
            return 2.0 * stats.norm.sf(np.abs(score) / sigma)

    n_ab = double_t["n_samples"].reindex(gd["genes"]).fillna(N_COLONIES_DEFAULT).to_numpy(dtype=float)
    gd["p_z_own_sd"] = z_p(gd["value"].to_numpy(), double_t["sd_rms"].reindex(gd["genes"]).to_numpy() / np.sqrt(n_ab))
    n_abc = triple_t["n_samples"].reindex(gt["genes"]).fillna(N_COLONIES_DEFAULT).to_numpy(dtype=float)
    gt["p_z_own_sd"] = z_p(gt["value"].to_numpy(), triple_t["sd_rms"].reindex(gt["genes"]).to_numpy() / np.sqrt(n_abc))
    md = np.isfinite(gd["p"]) & np.isfinite(gd["p_z_own_sd"])
    mt = np.isfinite(gt["p"]) & np.isfinite(gt["p_z_own_sd"])
    summary["confidence"] = {
        "doubles_n": int(md.sum()),
        "doubles_spearman_stored_vs_z_own_sd": float(stats.spearmanr(gd.loc[md, "p"], gd.loc[md, "p_z_own_sd"])[0]),
        "triples_n": int(mt.sum()),
        "triples_spearman_stored_vs_z_own_sd": float(stats.spearmanr(gt.loc[mt, "p"], gt.loc[mt, "p_z_own_sd"])[0]),
    }

    # ---- the 025 build beside it
    with open(osp.join(RESULTS_025, "s3_closure_recompute_summary.json")) as f:
        s025 = json.load(f)
    summary["build_025"] = {
        "n_by_order": s025["n_by_order"],
        "digenic_all": s025["digenic_strength"]["all"],
        "trigenic_all": s025["trigenic_strength"]["all"],
        "trigenic_no_essential_single": s025["trigenic_strength"]["no_essential_single"],
        "singles_with_converted_zero": s025["essentiality_in_singles"]["n"],
        "singles_zero_only_no_measurement": s025["essentiality_in_singles"]["n_essential_only_no_measurement"],
        "n_merged_doubles": s025["fitness_conflicts"]["n_doubles_merged_interaction"],
        "n_merged_triples": s025["fitness_conflicts"]["n_triples_merged_interaction"],
        "within_screen_control": {k: v for k, v in s025["within_screen_control"].items() if k != "kuzmin_eps_raw_column_vs_identity"},
    }

    # ---- survival of the 010 pinned triples (025 indices -> gene sets -> 029 triples)
    rec025 = pd.read_parquet(osp.join(CACHE_025, "s3_records.parquet"), columns=["idx", "order", "genes"])
    g025 = rec025.set_index("idx")["genes"]
    with gzip.open(osp.join(RESULTS_025, "pinned_splits_from_010_seed_42.json.gz"), "rt") as f:
        pinned = json.load(f)["pinned"]
    t029 = set(gt["genes"].unique())
    survival = {}
    for split, idxs in pinned.items():
        gs = g025.reindex(idxs)
        assert gs.notna().all(), f"{split}: pinned index outside the 025 S3 cache"
        alive = gs.isin(t029)
        survival[split] = {"n_025": int(len(gs)), "n_in_029": int(alive.sum()), "frac": float(alive.mean())}
    survival["triples_025_total"] = int((rec025["order"] == 3).sum())
    survival["triples_029_total"] = int(len(t029))
    survival["triples_in_both"] = int(len(t029 & set(rec025.loc[rec025["order"] == 3, "genes"])))
    summary["pinned_survival"] = survival
    with open(osp.join(RESULTS_DIR, "pinned_triple_survival.json"), "w") as f:
        json.dump(survival, f, indent=2)

    gd.to_parquet(osp.join(CACHE_DIR, "doubles_recomputed.parquet"), index=False)
    gt.to_parquet(osp.join(CACHE_DIR, "triples_recomputed.parquet"), index=False)
    single_t.reset_index().to_parquet(osp.join(CACHE_DIR, "singles_policy.parquet"), index=False)
    with open(osp.join(RESULTS_DIR, "closure_recompute_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps({k: v for k, v in summary.items() if k != "policy_stats"}, indent=2, default=float))
    print(pol_stats[pol_stats["stored_source"] == "all"].to_string())
    write_tables(summary, pol_stats, TABLE_DIR)
    make_figure(gd, gt, pol_stats, summary, IMG_DIR)


if __name__ == "__main__":
    sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
    stage = sys.argv[1] if len(sys.argv) > 1 else "analyze"
    {"scan": scan, "analyze": analyze}[stage]()
