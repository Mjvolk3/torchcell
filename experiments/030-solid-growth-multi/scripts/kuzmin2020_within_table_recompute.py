# experiments/030-solid-growth-multi/scripts/kuzmin2020_within_table_recompute.py
# [[experiments.030-solid-growth-multi.scripts.kuzmin2020_within_table_recompute]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/030-solid-growth-multi/scripts/kuzmin2020_within_table_recompute

"""Recompute the Kuzmin 2020 trigenic score on the raw tables, matching within one table.

Kuzmin 2020 releases its raw screens in two tables: S1, the double-mutant queries and
their single-mutant controls against the diagnostic array of about 1,200 strains, and
S3, the pilot screens of 11 double-mutant queries against the genome-wide deletion array
and the temperature-sensitive array. The array single-mutant fitness on a row is a
constant of the array strain inside one table and differs between the tables for 1,159
of the 4,553 strains, so a screen identity is part of every term. The loaders read both
tables into one dataset without recording which one a row came from.

For every trigenic row this script takes f_ijk, f_ij and f_k from the row itself and
eps_ik, eps_jk from the single-mutant control queries' rows against the same array strain,
first inside the row's own table, then pooled over both tables (what a screen-level match
does), and compares tau = f_ijk - f_ij f_k - eps_ik - eps_jk with the released score.

    python experiments/030-solid-growth-multi/scripts/kuzmin2020_within_table_recompute.py

Reads the two xlsx files of the dmf_kuzmin2020 raw mirror; writes
results/kuzmin2020_within_table_recompute.json and .csv.
"""

from __future__ import annotations

import json
import os
import os.path as osp
import re
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from dotenv import load_dotenv
from scipy import stats

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RAW = osp.join(DATA_ROOT, "data/torchcell/dmf_kuzmin2020/raw")
RESULTS = osp.join(EXPERIMENT_ROOT, "030-solid-growth-multi/results")
HO = "YDL227C"
GENE = re.compile(r"^(Y[A-P][LR]\d{3}[CW](?:-[A-Z])?)")


def _stats(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
) -> dict[str, float | int]:
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    lr = stats.linregress(x, y)
    return {
        "n": int(len(x)),
        "pearson": float(stats.pearsonr(x, y)[0]),
        "spearman": float(stats.spearmanr(x, y)[0]),
        "slope": float(lr.slope),
        "rmse": float(np.sqrt(np.mean((x - y) ** 2))),
        "median_abs_residual": float(np.median(np.abs(x - y))),
        "frac_abs_residual_below_1e-3": float(np.mean(np.abs(x - y) < 1e-3)),
    }


def _query_genes(strain_id: str) -> tuple[str, ...]:
    head = strain_id.split("_")[0]
    return tuple(g for g in head.split("+") if GENE.match(g))


def load() -> pd.DataFrame:
    frames = []
    for table in ("S1", "S3"):
        d = pd.read_excel(osp.join(RAW, f"aaz5667-Table-{table}.xlsx"), skiprows=1)
        d["table"] = table
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(
        columns={
            "Query strain ID": "q",
            "Array strain ID": "a",
            "Combined mutant type": "type",
            "Adjusted genetic interaction score (epsilon or tau)": "score",
            "Query single/double mutant fitness": "f_q",
            "Array single mutant fitness": "f_a",
            "Double/triple mutant fitness": "f_qa",
        }
    )
    df["genes"] = df["q"].map(_query_genes)
    return df


def controls(df: pd.DataFrame) -> pd.DataFrame:
    """Single-mutant control query rows: query '<gene>+HO', one per (gene, array strain, table)."""
    dig = df[df["type"] == "digenic"].copy()
    is_control = dig["q"].str.contains(HO, regex=False)
    ctrl = dig[is_control].copy()
    ctrl["gene"] = ctrl["genes"].map(lambda g: [x for x in g if x != HO][0])
    return ctrl


def _write_table(out: pd.DataFrame, path: str) -> None:
    """The per-table result for notes-tex/025-s3-closure, in the style of its other tables."""
    names = {
        "S1": "S1, diagnostic array",
        "S3": "S3, pilot screens",
        "both": "both tables",
    }
    lines = [
        "%% SOURCE: experiments/030-solid-growth-multi/scripts/kuzmin2020_within_table_recompute.py "
        "(results/kuzmin2020_within_table_recompute.csv) -- GENERATED, do not edit",
        "",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"rows & terms & $n$ & $r$ & slope & rmse & $|$residual$|<10^{-3}$ \\",
        r"\midrule",
    ]
    for table, group in out.groupby("table", sort=False):
        for k, (_, r) in enumerate(group.iterrows()):
            first = names[str(table)] if k == 0 else ""
            form = str(r["form"]).replace("f_k", "$f_k$")
            rr = f"{r['pearson']:.3f}"
            rr = rf"\textbf{{{rr}}}" if k == 0 else rr
            lines.append(
                f"{first} & {form} & {int(r['n']):,} & {rr} & {r['slope']:.3f} & "
                f"{r['rmse']:.5f} & {100 * r['frac_abs_residual_below_1e-3']:.1f}\\% \\\\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    df = load()
    ctrl = controls(df)
    print(
        f"rows {len(df):,}; trigenic {int((df['type'] == 'trigenic').sum()):,}; "
        f"control rows {len(ctrl):,} over {ctrl['gene'].nunique():,} control genes",
        flush=True,
    )
    # eps of a control gene against an array strain, within a table and pooled over both
    within = ctrl.groupby(["table", "gene", "a"])["score"].mean()
    pooled = ctrl.groupby(["gene", "a"])["score"].mean()
    n_pooled_multi = int((ctrl.groupby(["gene", "a"]).size() > 1).sum())

    tri = df[(df["type"] == "trigenic") & (df["genes"].map(len) == 2)].copy()
    tri["gi"] = tri["genes"].map(lambda g: g[0])
    tri["gj"] = tri["genes"].map(lambda g: g[1])

    def look(index: pd.Series, keys: list[tuple[Any, ...]]) -> npt.NDArray[np.float64]:
        return index.reindex(pd.MultiIndex.from_tuples(keys)).to_numpy(dtype=float)

    eps_ik_w = look(within, list(zip(tri["table"], tri["gi"], tri["a"])))
    eps_jk_w = look(within, list(zip(tri["table"], tri["gj"], tri["a"])))
    eps_ik_p = look(pooled, list(zip(tri["gi"], tri["a"])))
    eps_jk_p = look(pooled, list(zip(tri["gj"], tri["a"])))
    f_ijk = tri["f_qa"].to_numpy(dtype=float)
    f_ij = tri["f_q"].to_numpy(dtype=float)
    f_k = tri["f_a"].to_numpy(dtype=float)
    y = tri["score"].to_numpy(dtype=float)

    tau_within = f_ijk - f_ij * f_k - eps_ik_w - eps_jk_w
    tau_pooled = f_ijk - f_ij * f_k - eps_ik_p - eps_jk_p
    # the table's own array single replaced by the other table's value where it differs
    other = df.groupby(["table", "a"])["f_a"].first().unstack("table")
    swapped = np.where(
        tri["table"].to_numpy() == "S1",
        other["S3"].reindex(tri["a"]).to_numpy(dtype=float),
        other["S1"].reindex(tri["a"]).to_numpy(dtype=float),
    )
    f_k_pooled_mean = np.nanmean(np.vstack([f_k, swapped]), axis=0)
    tau_fk_mean = f_ijk - f_ij * f_k_pooled_mean - eps_ik_w - eps_jk_w

    rows = []
    for table in ("S1", "S3", "both"):
        sel = (
            np.ones(len(tri), dtype=bool)
            if table == "both"
            else (tri["table"] == table).to_numpy()
        )
        rows.append(
            {
                "table": table,
                "form": "controls within the row's table",
                **_stats(y[sel], tau_within[sel]),
            }
        )
        rows.append(
            {
                "table": table,
                "form": "controls pooled over both tables",
                **_stats(y[sel], tau_pooled[sel]),
            }
        )
        rows.append(
            {
                "table": table,
                "form": "within-table controls, f_k averaged over tables",
                **_stats(y[sel], tau_fk_mean[sel]),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(osp.join(RESULTS, "kuzmin2020_within_table_recompute.csv"), index=False)
    _write_table(out, osp.join(RESULTS, "t11-kuzmin2020-within-table.tex"))
    summary = {
        "n_trigenic_rows": int(len(tri)),
        "n_trigenic_by_table": tri["table"].value_counts().to_dict(),
        "n_control_gene_array_pairs_in_both_tables": n_pooled_multi,
        "n_array_strains": int(df["a"].nunique()),
        "n_array_strains_with_two_f_a": int(
            (df.groupby("a")["f_a"].nunique() > 1).sum()
        ),
        "n_array_strains_f_a_varies_within_a_table": int(
            (df.groupby(["table", "a"])["f_a"].nunique() > 1).sum()
        ),
        "rows": rows,
    }
    with open(osp.join(RESULTS, "kuzmin2020_within_table_recompute.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
