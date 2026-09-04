# experiments/010-kuzmin-tmi/scripts/kuzmin_ladder_census.py
# [[experiments.010-kuzmin-tmi.scripts.kuzmin_ladder_census]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/kuzmin_ladder_census
#
# Does a monotone 1 -> 2 -> 3 deletion ladder exist anywhere in the Kuzmin trigenic
# corpus, and if so, where?
#
# WHY THIS AND NOT MORE INFERENCE. The panel premise is that a triple can be reached by
# stacking deletions one at a time without ever losing fitness, so a trigenic
# interaction converts into a strain that is better than every intermediate. On the one
# screen already walked (pcl6-delta pcl7-delta) 3 of 5,502 paths did that. That is one
# query out of roughly four hundred, and a single screen cannot say whether the ladder
# is rare everywhere or concentrated in queries nobody has looked at. This asks the
# whole corpus, using only measurements. It costs minutes; a new inference space costs
# GPU-days, so the census comes first.
#
# WHAT A PATH IS. Every Kuzmin trigenic record is a query DOUBLE crossed against one
# array gene, so each triple is {q1, q2, X} and is reachable by six orderings:
#     WT (f = 1) -> single a -> double {a,b} -> triple {q1,q2,X}
# A path is MONOTONE when no step lowers fitness. Two of the six orderings route
# through the query double {q1,q2}, whose fitness is one value per screen.
#
# WHERE EACH RUNG COMES FROM, per rung, never assumed:
#   triple            the trigenic table's combined-mutant fitness, in screen
#   double {q1,q2}    the same table's query fitness column, in screen
#   doubles {q,X}     DmfKuzmin2018/2020 first (same screens), DmfCostanzo2016 at 30 C
#   singles           SmfKuzmin2018/2020 first, SmfCostanzo2016 at 30 C after
#
# Deletion arrays only. Temperature-sensitive and DAmP arrays are not gene removals and
# are counted separately rather than pooled.
#
# Run from repo root:
#   ~/miniconda3/envs/torchcell/bin/python \
#     experiments/010-kuzmin-tmi/scripts/kuzmin_ladder_census.py
#
# Outputs:
#   results/kuzmin_ladder_census_wins.csv      the 44,462 triples that beat wild type
#   results/kuzmin_ladder_census_queries.csv   one row per query double
#   results/kuzmin_ladder_census.json          the counts quoted in prose
#   $DATA_ROOT/.../010-kuzmin-tmi/kuzmin_ladder_census_triples.parquet  the full frame
#   $ASSET_IMAGES_DIR/010-kuzmin-tmi/kuzmin_ladder_census.{png,svg}

import json
import os
import os.path as osp

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.ticker import MultipleLocator

from torchcell.datasets.scerevisiae import DmfCostanzo2016Dataset, SmfCostanzo2016Dataset
from torchcell.datasets.scerevisiae.kuzmin2018 import (
    DmfKuzmin2018Dataset,
    SmfKuzmin2018Dataset,
    TmfKuzmin2018Dataset,
)
from torchcell.datasets.scerevisiae.kuzmin2020 import (
    DmfKuzmin2020Dataset,
    SmfKuzmin2020Dataset,
    TmfKuzmin2020Dataset,
)
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]

RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi", "results")
IMAGES_DIR = osp.join(ASSET_IMAGES_DIR, "010-kuzmin-tmi")

TAU_CUT, P_CUT, STRONG_CUT = 0.08, 0.05, 0.16
HO = "YDL227C"  # the ho-delta half of a Kuzmin single-mutant cross


def root(name: str) -> str:
    return osp.join(DATA_ROOT, "data/torchcell", name)


def set_plot_style():
    """Inside the plot call: constructing a dataset resets rcParams and savefig.bbox."""
    plt.rcParams.update(
        {
            "font.family": "Arial", "font.size": 6, "axes.labelsize": 6,
            "axes.titlesize": 6, "xtick.labelsize": 6, "ytick.labelsize": 6,
            "legend.fontsize": 5, "legend.title_fontsize": 5, "figure.titlesize": 6,
            "svg.fonttype": "none", "axes.linewidth": 0.5,
            "savefig.bbox": None, "savefig.pad_inches": 0.0,
        }
    )


def trigenic_frame() -> pd.DataFrame:
    """Every Kuzmin trigenic record from both releases, columns normalized."""
    frames = []
    for cls, rt, fit_col, sd_col, year in (
        (TmfKuzmin2018Dataset, "tmf_kuzmin2018",
         "Combined mutant fitness", "Combined mutant fitness standard deviation", 2018),
        (TmfKuzmin2020Dataset, "tmf_kuzmin2020",
         "Double/triple mutant fitness", "Double/triple mutant fitness standard deviation", 2020),
    ):
        df = cls(root=root(rt)).df
        df = df[df["Combined mutant type"] == "trigenic"]
        frames.append(
            pd.DataFrame(
                {
                    "release": year,
                    "q1": df["Query systematic name_1"].to_numpy(),
                    "q2": df["Query systematic name_2"].to_numpy(),
                    "x": df["Array systematic name"].to_numpy(),
                    "array_allele": df["Array allele name"].to_numpy(),
                    "array_perturbation_type": df["array_perturbation_type"].to_numpy(),
                    "f_triple": df[fit_col].to_numpy(dtype=float),
                    "sd_triple": df[sd_col].to_numpy(dtype=float),
                    "f_query_double": df["Query single/double mutant fitness"].to_numpy(dtype=float),
                    "tau": df["Adjusted genetic interaction score (epsilon or tau)"].to_numpy(dtype=float),
                    "tau_p_value": df["P-value"].to_numpy(dtype=float),
                }
            )
        )
    out = pd.concat(frames, ignore_index=True)
    # Order the query pair so one screen has one key regardless of column order.
    a, b = out["q1"].astype(str), out["q2"].astype(str)
    swap = a > b
    out["q1"] = np.where(swap, b, a)
    out["q2"] = np.where(swap, a, b)
    out["query_double"] = out["q1"] + "+" + out["q2"]
    return out


def single_lookup(genes: set[str]) -> dict[str, float]:
    """gene -> fitness. Kuzmin's own screens first, Costanzo 2016 at 30 C after."""
    out: dict[str, float] = {}
    for cls, rt in ((SmfKuzmin2020Dataset, "smf_kuzmin2020"),):
        df = cls(root=root(rt)).df
        gene = np.where(df["ORF1"] == HO, df["ORF2"], df["ORF1"])
        df = df.assign(gene=gene)
        df = df[df["gene"].isin(genes) & (df["Mutant type"] == "Single mutant")]
        for g, block in df.groupby("gene"):
            f = float(block["Fitness"].mean())
            if np.isfinite(f):
                out.setdefault(str(g), f)

    k18 = SmfKuzmin2018Dataset(root=root("smf_kuzmin2018")).df
    name_col = "Query systematic name no ho"
    if name_col in k18.columns:
        sub = k18[k18[name_col].isin(genes)]
        for g, block in sub.groupby(name_col):
            f = float(block["Combined mutant fitness"].mean())
            if np.isfinite(f):
                out.setdefault(str(g), f)

    cs = SmfCostanzo2016Dataset(root=root("smf_costanzo2016")).df
    cs = cs[
        cs["Systematic gene name"].isin(genes)
        & (cs["Temperature"] == 30)
        & cs["perturbation_type"].str.contains("deletion")
    ]
    for g, block in cs.groupby("Systematic gene name"):
        f = float(block["Single mutant fitness"].mean())
        if np.isfinite(f):
            out.setdefault(str(g), f)
    return out


def _pair_keys(a: np.ndarray, b: np.ndarray, code: dict[str, int], n: int) -> np.ndarray:
    """Order-free int64 key for a gene pair, or -1 when either gene is unknown."""
    ca = np.array([code.get(g, -1) for g in a], dtype=np.int64)
    cb = np.array([code.get(g, -1) for g in b], dtype=np.int64)
    lo, hi = np.minimum(ca, cb), np.maximum(ca, cb)
    return np.where((lo < 0), -1, lo * n + hi)


def double_lookup(needed: np.ndarray, code: dict[str, int], n: int) -> dict[int, float]:
    """pair key -> double-mutant fitness, for the pairs the census actually needs.

    Only the needed pairs are kept. Costanzo 2016 alone is 14.6 million rows at 30 C,
    and materializing all of it as a dict would cost far more memory than the census.
    """
    want = np.unique(needed[needed >= 0])
    out: dict[int, float] = {}

    for cls, rt, fit_col in (
        (DmfKuzmin2020Dataset, "dmf_kuzmin2020", "Double/triple mutant fitness"),
        (DmfKuzmin2018Dataset, "dmf_kuzmin2018", "Combined mutant fitness"),
    ):
        df = cls(root=root(rt)).df
        qcol = "Query systematic name no ho"
        keys = _pair_keys(
            df[qcol].to_numpy().astype(str),
            df["Array systematic name"].to_numpy().astype(str),
            code, n,
        )
        hit = np.isin(keys, want)
        agg = pd.DataFrame(
            {"key": keys[hit], "f": df[fit_col].to_numpy(dtype=float)[hit]}
        ).groupby("key")["f"].mean()
        for k, f in agg.items():
            if np.isfinite(f):
                out.setdefault(int(k), float(f))

    cd = DmfCostanzo2016Dataset(root=root("dmf_costanzo2016")).df
    cd = cd[cd["Temperature"] == 30]
    keys = _pair_keys(
        cd["Query Systematic Name"].to_numpy().astype(str),
        cd["Array Systematic Name"].to_numpy().astype(str),
        code, n,
    )
    hit = np.isin(keys, want)
    agg = pd.DataFrame(
        {"key": keys[hit], "f": cd["Double mutant fitness"].to_numpy(dtype=float)[hit]}
    ).groupby("key")["f"].mean()
    for k, f in agg.items():
        if np.isfinite(f):
            out.setdefault(int(k), float(f))
    return out


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    tri = trigenic_frame()
    print(f"Kuzmin trigenic records: {len(tri):,} over "
          f"{tri['query_double'].nunique():,} query doubles")
    print(tri["array_perturbation_type"].value_counts().to_string())

    tri = tri[tri["array_perturbation_type"] == "KanMX_deletion"].reset_index(drop=True)
    print(f"deletion-array records: {len(tri):,} over "
          f"{tri['query_double'].nunique():,} query doubles")

    genes = set(tri["q1"]) | set(tri["q2"]) | set(tri["x"])
    print(f"distinct genes: {len(genes):,}")
    singles = single_lookup(genes)
    print(f"singles resolved: {len(singles):,} of {len(genes):,}")

    vocab = sorted(genes)
    code = {g: i for i, g in enumerate(vocab)}
    n = len(vocab)
    key_q1x = _pair_keys(tri["q1"].to_numpy().astype(str), tri["x"].to_numpy().astype(str), code, n)
    key_q2x = _pair_keys(tri["q2"].to_numpy().astype(str), tri["x"].to_numpy().astype(str), code, n)
    doubles = double_lookup(np.concatenate([key_q1x, key_q2x]), code, n)
    print(f"doubles resolved: {len(doubles):,} pairs")

    nan = np.nan
    f_q1 = np.array([singles.get(g, nan) for g in tri["q1"]])
    f_q2 = np.array([singles.get(g, nan) for g in tri["q2"]])
    f_x = np.array([singles.get(g, nan) for g in tri["x"]])
    f_q1x = np.array([doubles.get(int(k), nan) for k in key_q1x])
    f_q2x = np.array([doubles.get(int(k), nan) for k in key_q2x])
    f_qq = tri["f_query_double"].to_numpy(dtype=float)
    f_abc = tri["f_triple"].to_numpy(dtype=float)

    complete = np.isfinite(f_q1) & np.isfinite(f_q2) & np.isfinite(f_x) \
        & np.isfinite(f_q1x) & np.isfinite(f_q2x) & np.isfinite(f_qq) & np.isfinite(f_abc)
    print(f"\ntriples with a complete lattice: {int(complete.sum()):,} of {len(tri):,}")

    # The six orderings as (first single, the double it enters).
    routes = [
        ("q1", f_q1, f_qq), ("q1", f_q1, f_q1x),
        ("q2", f_q2, f_qq), ("q2", f_q2, f_q2x),
        ("x", f_x, f_q1x), ("x", f_x, f_q2x),
    ]
    mono = np.zeros(len(tri), dtype=np.int16)
    best_min_rung = np.full(len(tri), -np.inf)
    for _, f_first, f_pair in routes:
        ok = (f_first >= 1.0) & (f_pair >= f_first) & (f_abc >= f_pair)
        mono += np.where(complete & ok, 1, 0).astype(np.int16)
        rung_min = np.minimum(f_first, f_pair)
        best_min_rung = np.where(complete, np.maximum(best_min_rung, rung_min), best_min_rung)

    out = pd.DataFrame(
        {
            "release": tri["release"], "query_double": tri["query_double"],
            "q1": tri["q1"], "q2": tri["q2"], "x": tri["x"],
            "array_allele": tri["array_allele"],
            "f_q1": f_q1, "f_q2": f_q2, "f_x": f_x,
            "f_query_double": f_qq, "f_q1x": f_q1x, "f_q2x": f_q2x,
            "f_triple": f_abc, "sd_triple": tri["sd_triple"],
            "tau": tri["tau"], "tau_p_value": tri["tau_p_value"],
            "complete_lattice": complete, "n_monotone_routes": mono,
            "best_min_rung": np.where(np.isfinite(best_min_rung), best_min_rung, nan),
        }
    )
    comp = out[out["complete_lattice"]].copy()
    # The full frame is 286k rows and 40 MB, which is derived data, not a result to
    # version. It goes to DATA_ROOT; only the wins, the part anything downstream
    # selects from, are small enough to commit.
    big_dir = osp.join(DATA_ROOT, "data/torchcell/experiments/010-kuzmin-tmi")
    os.makedirs(big_dir, exist_ok=True)
    comp.to_parquet(osp.join(big_dir, "kuzmin_ladder_census_triples.parquet"), index=False)
    print(f"full per-triple frame -> {big_dir}/kuzmin_ladder_census_triples.parquet")

    has = comp["n_monotone_routes"] > 0
    called = (comp["tau"] > TAU_CUT) & (comp["tau_p_value"] < P_CUT)
    strong = (comp["tau"] > STRONG_CUT) & (comp["tau_p_value"] < P_CUT)
    above_wt = comp["f_triple"] > 1.0

    per_query = comp.groupby("query_double").agg(
        n_triples=("x", "size"),
        n_with_monotone=("n_monotone_routes", lambda s: int((s > 0).sum())),
        n_monotone_routes=("n_monotone_routes", "sum"),
        max_f_triple=("f_triple", "max"),
        n_above_wt=("f_triple", lambda s: int((s > 1.0).sum())),
        f_query_double=("f_query_double", "first"),
    ).reset_index().sort_values("n_with_monotone", ascending=False)
    per_query.to_csv(osp.join(RESULTS_DIR, "kuzmin_ladder_census_queries.csv"), index=False)

    wins = comp[above_wt].copy()
    wins["greedy_reachable"] = wins["n_monotone_routes"] > 0
    wins["valley_depth"] = 1.0 - wins["best_min_rung"]
    wins.sort_values("f_triple", ascending=False).to_csv(
        osp.join(RESULTS_DIR, "kuzmin_ladder_census_wins.csv"), index=False
    )

    # The anti-greedy stratum. A monotone route is exactly what a greedy stack-and-keep
    # search can find: it never has to accept a loss. A triple that beats wild type with
    # NO monotone route is a win sitting behind a valley, invisible to that search and
    # reachable only by predicting the endpoint. That gap is the demonstration.
    valley_win = above_wt & ~has
    print(f"\nbeats wild type: {int(above_wt.sum()):,}")
    print(f"  reachable greedily (a monotone route exists): {int((above_wt & has).sum()):,}")
    print(f"  behind a valley (no monotone route): {int(valley_win.sum()):,} "
          f"= {valley_win.sum() / max(above_wt.sum(), 1):.1%} of the wins")
    print(f"  of those, a called positive interaction: {int((valley_win & called).sum()):,}")
    print(f"  of those, strong: {int((valley_win & strong).sum()):,}")
    depth = 1.0 - comp.loc[valley_win, "best_min_rung"]
    print(f"  valley depth on the best route, median {depth.median():.4f}, "
          f"90th pct {depth.quantile(0.9):.4f}")

    print(f"\ntriples with at least one monotone route: {int(has.sum()):,} "
          f"({has.mean():.3%} of complete)")
    print(f"total monotone routes: {int(comp['n_monotone_routes'].sum()):,} of "
          f"{6 * len(comp):,}")
    print(f"query doubles hosting at least one: "
          f"{int((per_query['n_with_monotone'] > 0).sum()):,} of {len(per_query):,}")
    print(f"triples above wild type: {int(above_wt.sum()):,} ({above_wt.mean():.3%})")
    print(f"monotone AND a called positive interaction: {int((has & called).sum()):,}")
    print(f"monotone AND strong ({STRONG_CUT:+.2f}, P<{P_CUT}): {int((has & strong).sum()):,}")

    print("\n=== 15 query doubles hosting the most monotone triples ===")
    print(per_query.head(15).to_string(index=False))

    print("\n=== best monotone triples by endpoint fitness ===")
    win = comp[has].sort_values("f_triple", ascending=False)
    print(win.head(20)[["query_double", "x", "array_allele", "f_q1", "f_q2", "f_x",
                        "f_query_double", "f_triple", "tau", "tau_p_value",
                        "n_monotone_routes"]].to_string(index=False))

    print("\n=== monotone AND called positive ===")
    print(comp[has & called].sort_values("f_triple", ascending=False)
          .head(20)[["query_double", "x", "array_allele", "f_triple", "tau",
                     "tau_p_value", "n_monotone_routes"]].to_string(index=False))

    summary = {
        "n_trigenic_records": int(len(tri)),
        "n_query_doubles": int(tri["query_double"].nunique()),
        "n_complete_lattice": int(len(comp)),
        "n_triples_with_monotone_route": int(has.sum()),
        "frac_triples_with_monotone_route": float(has.mean()),
        "n_monotone_routes": int(comp["n_monotone_routes"].sum()),
        "n_routes_evaluated": int(6 * len(comp)),
        "n_query_doubles_with_monotone": int((per_query["n_with_monotone"] > 0).sum()),
        "n_query_doubles_complete": int(len(per_query)),
        "n_triples_above_wt": int(above_wt.sum()),
        "n_monotone_and_called": int((has & called).sum()),
        "n_monotone_and_strong": int((has & strong).sum()),
        "max_f_triple_monotone": float(comp.loc[has, "f_triple"].max()) if has.any() else None,
        "max_f_triple_overall": float(comp["f_triple"].max()),
        "n_wins_greedy_reachable": int((above_wt & has).sum()),
        "n_wins_behind_a_valley": int(valley_win.sum()),
        "frac_wins_behind_a_valley": float(valley_win.sum() / max(above_wt.sum(), 1)),
        "n_valley_wins_called_positive": int((valley_win & called).sum()),
        "n_valley_wins_strong": int((valley_win & strong).sum()),
        "median_valley_depth_of_wins": float(
            (1.0 - comp.loc[valley_win, "best_min_rung"]).median()
        ),
        "max_f_triple_valley_win": float(comp.loc[valley_win, "f_triple"].max()),
        # The ladders that do exist may just be riding a beneficial query double rather
        # than any three-way effect. This is the number that tells them apart.
        "frac_monotone_triples_with_query_double_above_wt": float(
            (comp.loc[has, "f_query_double"] > 1.0).mean()
        ),
        "frac_all_triples_with_query_double_above_wt": float(
            (comp["f_query_double"] > 1.0).mean()
        ),
        "median_tau_of_monotone_triples": float(comp.loc[has, "tau"].median()),
        "median_tau_of_all_triples": float(comp["tau"].median()),
    }
    with open(osp.join(RESULTS_DIR, "kuzmin_ladder_census.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n" + json.dumps(summary, indent=2))

    plot(comp, per_query, has, called, valley_win,
         osp.join(IMAGES_DIR, "kuzmin_ladder_census"))
    print(f"\nwrote figures to {IMAGES_DIR}")


def plot(comp, per_query, has, called, valley_win, out_stem):
    set_plot_style()
    fig, axes = plt.subplots(
        1, 3,
        figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(56.0)),
        gridspec_kw={"width_ratios": [1.0, 1.05, 1.1]},
    )

    ax = axes[0]
    ax.hist(comp["f_triple"], bins=120, color="0.80", edgecolor="none", zorder=3,
            label=f"all complete triples (n={len(comp):,})")
    ax.hist(comp.loc[valley_win, "f_triple"], bins=120, color=PLOT_PALETTE[0],
            edgecolor="none", zorder=4,
            label=f"beats WT, behind a valley (n={int(valley_win.sum()):,})")
    ax.hist(comp.loc[has, "f_triple"], bins=120, color=PLOT_PALETTE[1], edgecolor="none",
            zorder=5, label=f"beats WT, greedy reachable (n={int(has.sum()):,})")
    ax.axvline(1.0, color="black", linewidth=0.5, linestyle="--", zorder=6)
    ax.set_yscale("log")
    ax.set_xlabel("Triple-mutant fitness")
    ax.set_ylabel("Triples")
    ax.set_title("a  What greedy stacking can and cannot reach",
                 fontsize=6, loc="left", pad=3)
    ax.legend(loc="upper left", frameon=True, fontsize=5, handlelength=1.0,
              labelspacing=0.25, borderpad=0.3)

    # The query double's own fitness, split by whether the query hosts any ladder. If
    # the ladders are just riding a beneficial query, the two distributions separate.
    ax = axes[1]
    hosts = per_query["n_with_monotone"] > 0
    bins = np.linspace(
        float(per_query["f_query_double"].min()),
        float(per_query["f_query_double"].max()), 40,
    )
    ax.hist(per_query.loc[~hosts, "f_query_double"], bins=bins, color="0.72",
            edgecolor="black", linewidth=0.3, zorder=3,
            label=f"hosts no ladder (n={int((~hosts).sum())})")
    ax.hist(per_query.loc[hosts, "f_query_double"], bins=bins, color=PLOT_PALETTE[1],
            edgecolor="black", linewidth=0.3, alpha=0.85, zorder=4,
            label=f"hosts a ladder (n={int(hosts.sum())})")
    ax.axvline(1.0, color="black", linewidth=0.5, linestyle="--", zorder=5)
    ax.set_xlabel("Fitness of the query double itself")
    ax.set_ylabel("Query doubles")
    ax.set_title(
        "b  Ladders ride a beneficial query double\n"
        f"median {per_query.loc[hosts, 'f_query_double'].median():.3f} against "
        f"{per_query.loc[~hosts, 'f_query_double'].median():.3f}",
        fontsize=6, loc="left", pad=3,
    )
    ax.legend(loc="upper left", frameon=True, fontsize=5, handlelength=1.0,
              labelspacing=0.25, borderpad=0.3)

    ax = axes[2]
    ax.scatter(comp.loc[~has, "tau"], comp.loc[~has, "f_triple"], s=0.6,
               color="0.80", linewidths=0, zorder=3, label="no monotone route")
    ax.scatter(comp.loc[has, "tau"], comp.loc[has, "f_triple"], s=2.4,
               color=PLOT_PALETTE[1], linewidths=0, zorder=4, label="has a monotone route")
    ax.scatter(comp.loc[has & called, "tau"], comp.loc[has & called, "f_triple"], s=6.0,
               color=PLOT_PALETTE[0], edgecolor="black", linewidths=0.3, zorder=5,
               label=f"and a called positive (n={int((has & called).sum())})")
    ax.axhline(1.0, color="black", linewidth=0.5, linestyle="--", zorder=2)
    ax.axvline(TAU_CUT, color="black", linewidth=0.5, linestyle=":", zorder=2)
    ax.set_xlabel("Measured trigenic interaction $\\tau$")
    ax.set_ylabel("Triple-mutant fitness")
    ax.set_title("c  Ladder against interaction", fontsize=6, loc="left", pad=3)
    ax.legend(loc="lower left", frameon=True, fontsize=5, handlelength=1.0,
              labelspacing=0.25, borderpad=0.3, scatterpoints=1)
    ax.xaxis.set_major_locator(MultipleLocator(0.4))

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_color("black")
        ax.grid(axis="y", which="major", color="0.85", linewidth=0.3, zorder=0)
        ax.set_axisbelow(True)

    fig.suptitle(
        "Every measured deletion path in the Kuzmin trigenic corpus. A route is monotone "
        "when no deletion lowers fitness, starting from wild type.",
        fontsize=6, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(f"{out_stem}.png", dpi=300)
    savefig_true_size_svg(fig, f"{out_stem}.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
