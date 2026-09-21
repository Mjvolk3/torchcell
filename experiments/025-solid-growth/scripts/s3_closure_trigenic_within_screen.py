# experiments/025-solid-growth/scripts/s3_closure_trigenic_within_screen.py
# [[experiments.025-solid-growth.scripts.s3_closure_recompute]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/025-solid-growth/scripts/s3_closure_trigenic_within_screen

"""Recompute the TRIGENIC score inside one Kuzmin screen, matching by strain identifier.

The digenic within-screen control of ``s3_closure_recompute.py`` reproduces the reported
epsilon from the columns of the same raw row, because every term of eps = f_ab - f_a f_b
sits on that row. The trigenic identity does not fit on one row:

    tau_ijk = f_ijk - f_i f_j f_k - eps_ij f_k - eps_ik f_j - eps_jk f_i
            = f_ijk - f_ij f_k - eps_ik f_j - eps_jk f_i
            = f_ijk - f_ij f_k - eps_ik - eps_jk          (as published: f_i = f_j = 1)

(the first three terms collapse because eps_ij = f_ij - f_i f_j, with f_ij the DOUBLE
MUTANT QUERY fitness). The third line is what the released scores were actually computed
from, verified here: the single-mutant query fitness enters as 1.0 even in Kuzmin 2018,
where 99.2 percent of the control rows carry a measured value. Using those measured
values instead raises the median residual from 2.9e-05, which is the rounding of the
published five-decimal score, to 1.6e-03. The Kuzmin supplementary methods say where each term is measured
(kuzminSystematicAnalysisComplex2018/si/si1.md line 191): "Digenic interactions between
Q_i-A_k or Q_j-A_k were measured using our single mutant control queries. Query pair
interactions (Q_i-Q_j) were measured using the single and double mutant fitness standard
... and applying the multiplicative model". So on a trigenic row, f_ij, f_k and f_ijk are
the row's own query fitness, array fitness and combined fitness, while eps_ik, eps_jk and
f_i, f_j come from the two SINGLE MUTANT CONTROL QUERY rows of the same screen: query
"<gene>+YDL227C_tm<id>" (YDL227C is HO) against the same array strain.

This script does that join on the raw tables and asks whether the reported tau is
recoverable when every term is taken from the screen that produced it. It is the
trigenic counterpart of the digenic control, and it separates two explanations for the
r = 0.230 the build reaches: a wrong formula, or terms drawn from different screens.

    python experiments/025-solid-growth/scripts/s3_closure_trigenic_within_screen.py

Reads the cache written by ``s3_closure_recompute.py raw``
($DATA_ROOT/.../025-solid-growth/s3_closure/raw_kuzmin_all.parquet). Writes
experiments/025-solid-growth/results/s3_closure_trigenic_within_screen.json and the
LaTeX table notes-tex/025-s3-closure/tables/t8-trigenic-within-screen.tex.
"""

import json
import os
import os.path as osp

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy import stats

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
CACHE_DIR = osp.join(DATA_ROOT, "data/torchcell/experiments/025-solid-growth/s3_closure")
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "025-solid-growth/results")
TABLE_DIR = osp.join(osp.dirname(EXPERIMENT_ROOT), "notes-tex/025-s3-closure/tables")
HO = "YDL227C"


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
        "rmse": float(np.sqrt(np.mean((x - y) ** 2))),
        "median_abs_residual": float(np.median(np.abs(x - y))),
    }


def _query_genes(q: pd.Series) -> pd.DataFrame:
    """Split a query strain id 'GENE1+GENE2_tmNNNN' into its two ORF names."""
    stem = q.str.split("_", n=1).str[0]
    parts = stem.str.split("+", expand=True)
    return parts.rename(columns={0: "g1", 1: "g2"})


def controls(d: pd.DataFrame) -> pd.DataFrame:
    """Single-mutant control query rows: query '<gene>+YDL227C' (or the reverse).

    Returns one row per (control gene, array strain) with the control's epsilon and its
    query single-mutant fitness, averaged when a gene was screened more than once
    against the same array strain (different batches, different tm identifiers).
    """
    g = _query_genes(d["q"])
    is_ctrl = (g["g1"] == HO) | (g["g2"] == HO)
    d = d[is_ctrl].copy()
    g = g[is_ctrl]
    d["gene"] = np.where(g["g1"] == HO, g["g2"], g["g1"])
    # Kuzmin 2020 releases no query fitness on 99.6 percent of digenic rows; the source
    # scoring assigned 1.0 to every NaN fitness (si1.md), so the control uses that value.
    d["f_ctrl"] = d["f_q"].fillna(1.0)
    return (
        d.groupby(["gene", "a"])
        .agg(eps=("score", "mean"), f_single=("f_ctrl", "mean"), n_ctrl=("score", "size"))
        .reset_index()
    )


def reconstruct(k: pd.DataFrame, source: str) -> tuple[pd.DataFrame, dict]:
    t = k[(k["type"] == "trigenic") & (k["source"] == source)].copy()
    d = k[(k["type"] == "digenic") & (k["source"] == source)]
    g = _query_genes(t["q"])
    t["gi"], t["gj"] = g["g1"], g["g2"]
    ctrl = controls(d)
    n_raw = len(t)
    for side in ("i", "j"):
        m = ctrl.rename(columns={"gene": f"g{side}", "eps": f"eps_{side}k", "f_single": f"f_{side}", "n_ctrl": f"n_ctrl_{side}"})
        t = t.merge(m, on=[f"g{side}", "a"], how="left")
    t["f_ij"] = t["f_q"].fillna(1.0)
    # The PUBLISHED score sets the single-mutant query fitness to 1.0: with f_i = f_j = 1
    # the two control terms enter unscaled. Verified below against the alternative of
    # reading the control rows' own released query fitness.
    t["tau_rec"] = t["f_qa"] - t["f_ij"] * t["f_a"] - t["eps_ik"] - t["eps_jk"]
    t["tau_fi_from_control_rows"] = (
        t["f_qa"] - t["f_ij"] * t["f_a"] - t["eps_ik"] * t["f_j"] - t["eps_jk"] * t["f_i"]
    )
    # the same identity written from single-mutant fitness only, which is what a closure
    # over the database computes: it needs eps_ij, unavailable here, so f_ij is dropped
    # in favor of the product of the two control singles
    t["tau_no_query_double"] = t["f_qa"] - t["f_a"] - t["eps_ik"] - t["eps_jk"]
    both = t["eps_ik"].notna() & t["eps_jk"].notna()
    info = {
        "n_trigenic_rows": int(n_raw),
        "n_with_both_control_screens": int(both.sum()),
        "frac_with_both_control_screens": float(both.mean()),
        "frac_query_double_fitness_nan": float(k[(k["type"] == "trigenic") & (k["source"] == source)]["f_q"].isna().mean()),
        "mean_control_screens_per_gene_array": float(
            pd.concat([t["n_ctrl_i"], t["n_ctrl_j"]]).mean()
        ),
        "tau_within_screen": _stats(t["score"].to_numpy(), t["tau_rec"].to_numpy())
        | {
            "frac_exact_1e-4": float((np.abs(t["score"] - t["tau_rec"]) < 1e-4).mean()),
            "frac_exact_1e-4_where_query_double_fitness_released": float(
                (np.abs(t.loc[t["f_q"].notna(), "score"] - t.loc[t["f_q"].notna(), "tau_rec"]) < 1e-4).mean()
            ),
        },
        "tau_with_control_query_fitness_instead_of_one": _stats(
            t["score"].to_numpy(), t["tau_fi_from_control_rows"].to_numpy()
        ),
        "frac_control_rows_with_released_query_fitness": float(d["f_q"].notna().mean()),
        "tau_without_query_double_fitness": _stats(
            t["score"].to_numpy(), t["tau_no_query_double"].to_numpy()
        ),
        "raw_eps_column_vs_observed_minus_expected": _stats(
            t["eps_raw"].to_numpy(), (t["f_qa"] - t["f_ij"] * t["f_a"]).to_numpy()
        ),
    }
    return t, info


def main() -> None:
    k = pd.read_parquet(osp.join(CACHE_DIR, "raw_kuzmin_all.parquet"))
    summary = {}
    for source in ["kuzmin2018", "kuzmin2020"]:
        t, info = reconstruct(k, source)
        summary[source] = info
        t.to_parquet(osp.join(CACHE_DIR, f"trigenic_within_screen_{source}.parquet"), index=False)
        print(f"== {source}")
        print(json.dumps(info, indent=2))
    with open(osp.join(RESULTS_DIR, "s3_closure_trigenic_within_screen.json"), "w") as f:
        json.dump(summary, f, indent=2)
    write_table(summary)


def write_table(summary: dict) -> None:
    os.makedirs(TABLE_DIR, exist_ok=True)
    head = (
        "%% SOURCE: experiments/025-solid-growth/scripts/s3_closure_trigenic_within_screen.py "
        "(results/s3_closure_trigenic_within_screen.json) -- GENERATED, do not edit\n"
    )
    lines = [head, r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"screen & terms & $n$ & $r$ & $\rho$ & slope & rmse \\", r"\midrule"]
    label = {"kuzmin2018": "Kuzmin 2018", "kuzmin2020": "Kuzmin 2020"}
    rows = []
    for src, info in summary.items():
        rows.append((label[src], "every term from the same screen", info["tau_within_screen"]))
        rows.append((r"\quad", "the query double fitness dropped",
                     info["tau_without_query_double_fitness"]))
    best = max(v["pearson"] for _, _, v in rows if "pearson" in v)
    for name, terms, v in rows:
        if "pearson" not in v:
            continue
        r_txt = f"{v['pearson']:.3f}"
        if v["pearson"] == best:
            r_txt = r"\textbf{" + r_txt + "}"
        lines.append(f"{name} & {terms} & {v['n']:,} & {r_txt} & {v['spearman']:.3f} & "
                     f"{v['slope']:.2f} & {v['rmse']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(osp.join(TABLE_DIR, "t8-trigenic-within-screen.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
