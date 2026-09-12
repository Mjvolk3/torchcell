# experiments/007-kuzmin-tm/scripts/fba_screen_medium_si.py
# [[experiments.007-kuzmin-tm.scripts.fba_screen_medium_si]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/007-kuzmin-tm/scripts/fba_screen_medium_si
"""Statistics, table and panel g of FigS-yeast9-fba for the screen-medium rerun.

Reads the arms written by ``fba_screen_medium.py`` (``results/fba_screen_medium/<arm>/``)
and scores each one exactly as ``fba_baseline_si.py`` scores the frozen baseline: the
predicted trigenic interaction and triple-mutant fitness of every triple against the
measured values of the same gene set from the raw Kuzmin 2018 / 2020 tables (Pearson,
Spearman), the growth bands of the single, double and triple deletions, how many predicted
interactions are nonzero, and the internal consistency checks (a double or triple with at
most one model gene must reproduce its single's fitness; a triple cannot grow faster than
its slowest double). The frozen baseline's numbers are read from
``results/fba_baseline_si/stats.json`` so the table shows the four media side by side.

Outputs:
  results/fba_screen_medium/stats.json                  every number of the note and the table
  results/fba_screen_medium/summary.csv                 one row per arm
  results/fba_screen_medium/triples_matched_<arm>.parquet
  paper/nature-biotech/sections/tab-fba-screen-medium.tex
  $ASSET_IMAGES_DIR/007-kuzmin-tm/fba_screen_medium_tau.{svg,png}      panel g
  $ASSET_IMAGES_DIR/007-kuzmin-tm/fba_screen_medium_fitness.{svg,png}  note only
  $ASSET_IMAGES_DIR/007-kuzmin-tm/fba_screen_medium_bands.{svg,png}    note only

Run from the repo root, after the arms exist:
    python experiments/007-kuzmin-tm/scripts/fba_screen_medium_si.py
"""

import json
import os.path as osp
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from fba_baseline_si import (  # noqa: E402
    GRAY,
    ORANGE,
    ORANGE_F,
    PURPLE,
    PURPLE_F,
    RED,
    REPO_ROOT,
    TAU_ZERO,
    TEX_DIR,
    WT_LIKE,
    box,
    consistency,
    gene_coverage,
    growth_bands,
    load_model,
    raw_labels,
    save,
)
from fba_baseline_si import OUT as BASELINE_OUT
from fba_screen_medium import ARMS, OUT  # noqa: E402

from torchcell.utils import PANEL_WIDTHS_MM, PLOT_PALETTE, mm_to_in  # noqa: E402

#: Arm order in the table and the bands panel: the frozen run first, then the two
#: controls, then the rerun the note is about.
ORDER = ["frozen_baseline", "yeast9_default", "sm_glucose_3.3", "screen_medium"]
LABELS = {
    "frozen_baseline": "Yeast9 default (frozen run, 2025-09-15)",
    "yeast9_default": "Yeast9 default (rerun)",
    "sm_glucose_3.3": "SM: ammonium, glucose 3.3",
    "screen_medium": "Screen medium: SD/MSG -His/Arg/Lys/Ura",
}
SHORT_TEX = {
    "frozen_baseline": "Default, frozen run",
    "yeast9_default": "Default, rerun",
    "sm_glucose_3.3": "SM, glucose 3.3",
    "screen_medium": "Screen medium",
}
SHORT = {
    "frozen_baseline": "Default\n(frozen)",
    "yeast9_default": "Default\n(rerun)",
    "sm_glucose_3.3": "SM\nglc 3.3",
    "screen_medium": "Screen\nmedium",
}


def load_arm(arm: str) -> dict:
    d = osp.join(OUT, arm)
    return {
        "meta": json.load(open(osp.join(d, "fba_metadata.json"))),
        "medium": json.load(open(osp.join(d, "medium_bounds.json"))),
        "wt": pd.read_csv(osp.join(d, "wt_growth.csv")),
        "singles": pd.read_parquet(osp.join(d, "singles_deletions.parquet")),
        "doubles": pd.read_parquet(osp.join(d, "doubles_deletions.parquet")),
        "triples": pd.read_parquet(osp.join(d, "triples_deletions.parquet")),
        "trigenic": pd.read_parquet(osp.join(d, "trigenic_interactions.parquet")),
    }


def matched_triples(trigenic: pd.DataFrame, labels: pd.DataFrame, model_genes: set) -> pd.DataFrame:
    tri = trigenic.copy()
    tri["genes"] = [",".join(sorted(t)) for t in zip(tri.gene1, tri.gene2, tri.gene3)]
    tri["n_in_model"] = [sum(g in model_genes for g in t) for t in zip(tri.gene1, tri.gene2, tri.gene3)]
    tri = tri.rename(columns={"tau": "tau_fba", "fitness_observed": "fitness_fba"})
    out = tri[["genes", "n_in_model", "tau_fba", "fitness_fba"]].merge(labels, on="genes", how="left")
    assert len(out) == len(tri), (len(out), len(tri))
    n_unmatched = int(out.p_min.isna().sum())
    assert n_unmatched == 0, f"{n_unmatched} triples have no raw record"
    return out


def arm_stats(arm: str, data: dict, labels: pd.DataFrame, model, perts: dict) -> tuple[dict, pd.DataFrame]:
    model_genes, n_in, tri, cov = gene_coverage(model, perts, data["trigenic"], None)
    m = matched_triples(data["trigenic"], labels, model_genes)
    bands = growth_bands(data["singles"], data["doubles"], data["triples"])
    consist = consistency(model_genes, data["singles"], data["doubles"], data["triples"])
    tau, meas = m.tau_fba.to_numpy(), m.tau_measured.to_numpy()
    r_tau, p_tau = stats.pearsonr(tau, meas)
    rho_tau, _ = stats.spearmanr(tau, meas)
    r_fit, _ = stats.pearsonr(m.fitness_fba, m.fitness_measured)
    covd = m[m.n_in_model > 0]
    r_fit_cov, _ = stats.pearsonr(covd.fitness_fba, covd.fitness_measured)
    nz = m[np.abs(m.tau_fba) > TAU_ZERO]
    nz_counts = pd.Series(np.round(nz.tau_fba.to_numpy(), 2)).value_counts()
    fvc = pd.Series(np.round(m.fitness_fba.to_numpy(), 3)).value_counts()
    wt = data["wt"].iloc[0]
    st = {
        "arm": arm,
        "label": LABELS[arm],
        "description": ARMS[arm],
        "run": {k: data["meta"][k] for k in ("timestamp", "torchcell_commit", "cobra_version", "optlang_version",
                                             "solver", "n_processes", "fba_runtime_seconds", "total_runtime_seconds",
                                             "solver_status_counts", "perturbations_sha256")},
        "medium": {
            "n_open_exchanges": data["medium"]["n_open_exchanges"],
            "glucose_uptake_bound": data["medium"]["open_exchanges"]["r_1714"]["uptake_bound"],
            "glutamate_uptake_bound": data["medium"]["open_exchanges"].get("r_1889", {}).get("uptake_bound", 0.0),
            "ammonium_uptake_bound": data["medium"]["open_exchanges"].get("r_1654", {}).get("uptake_bound", 0.0),
            "n_amino_acid_exchanges_open": int(sum(
                v["metabolite"].startswith("L-") or v["metabolite"] == "L-glycine"
                for v in data["medium"]["open_exchanges"].values()
            )),
            "excluded_by_role": [] if data["medium"]["media_bounds"] is None else [
                r["component_name"] for r in data["medium"]["media_bounds"]["resolutions"] if r["outcome"] == "excluded_by_role"
            ],
            "unresolved": [] if data["medium"]["media_bounds"] is None else [
                r["component_name"] for r in data["medium"]["media_bounds"]["resolutions"] if r["outcome"] == "unresolved"
            ],
        },
        "wt": {"growth": float(wt.growth), "glucose_uptake": float(wt.glucose_uptake), "oxygen_uptake": float(wt.oxygen_uptake),
               "ammonium_exchange": float(wt.ammonium_exchange), "glutamate_exchange": float(wt.glutamate_exchange)},
        "growth_bands": bands.to_dict(orient="records"),
        "tau": {
            "n": int(len(m)),
            "pearson_r": float(r_tau),
            "pearson_p": float(p_tau),
            "spearman_rho": float(rho_tau),
            "frac_abs_below_1e-6": float((np.abs(tau) < 1e-6).mean()),
            "frac_abs_below_1e-3": float((np.abs(tau) < TAU_ZERO).mean()),
            "n_abs_above_1e-3": int(len(nz)),
            "nonzero_value_counts": {str(k): int(v) for k, v in nz_counts.head(8).items()},
            "n_nonzero_with_one_model_gene": int((nz.n_in_model <= 1).sum()),
            "pred_sd": float(m.tau_fba.std(ddof=1)),
        },
        "fitness": {
            "pearson_r": float(r_fit),
            "pearson_r_triples_with_model_gene": float(r_fit_cov),
            "n_triples_with_model_gene": int(len(covd)),
            "n_distinct_round3": int(len(fvc)),
            "value_counts_round3": {str(k): int(v) for k, v in fvc.head(8).items()},
            "pred_sd": float(m.fitness_fba.std(ddof=1)),
        },
        "coverage": {"triples_by_n_in_model": {int(r.n_in_model): int(r.n_triples) for r in cov.itertuples()},
                     "tau_nonzero_by_n_in_model": {int(r.n_in_model): int(r.n_tau_nonzero) for r in cov.itertuples()}},
        "consistency": consist,
    }
    return st, m


def frozen_row(base: dict) -> dict:
    """The frozen baseline in the same shape, from fba_baseline_si's stats.json."""
    bands = base["growth_bands"]
    return {
        "arm": "frozen_baseline",
        "label": LABELS["frozen_baseline"],
        "medium": {"n_open_exchanges": base["model"]["n_medium_exchanges_open"],
                   "glucose_uptake_bound": base["model"]["glucose_uptake_bound"],
                   "glutamate_uptake_bound": 0.0, "ammonium_uptake_bound": 1000.0, "n_amino_acid_exchanges_open": 0},
        "wt": {"growth": base["model"]["wt_growth"]},
        "growth_bands": bands,
        "tau": base["correlations"]["tau"],
        "fitness": base["correlations"]["fitness"],
        "consistency": base["consistency"],
    }


def summary_table(rows: list[dict]) -> pd.DataFrame:
    out = []
    for st in rows:
        tb = {b["order"]: b for b in st["growth_bands"]}
        out.append({
            "arm": st["arm"],
            "label": st["label"],
            "open_exchanges": st["medium"]["n_open_exchanges"],
            "glucose_bound": st["medium"]["glucose_uptake_bound"],
            "amino_acid_exchanges": st["medium"]["n_amino_acid_exchanges_open"],
            "wt_growth": st["wt"]["growth"],
            "frac_wt_like_single": tb["single"]["frac_wt_like"],
            "frac_wt_like_double": tb["double"]["frac_wt_like"],
            "frac_wt_like_triple": tb["triple"]["frac_wt_like"],
            "frac_lethal_triple": tb["triple"]["frac_lethal"],
            "n_tau_nonzero": st["tau"]["n_abs_above_1e-3"],
            "pearson_r_tau": st["tau"]["pearson_r"],
            "spearman_rho_tau": st["tau"]["spearman_rho"],
            "pearson_r_fitness": st["fitness"]["pearson_r"],
            "pearson_r_fitness_covered": st["fitness"]["pearson_r_triples_with_model_gene"],
            "doubles_inconsistent": st["consistency"]["doubles_inconsistent_with_single"],
            "triples_above_min_double": st["consistency"]["triples_above_min_double"],
        })
    return pd.DataFrame(out)


def write_table(summary: pd.DataFrame, path: str):
    lines = [
        "%% SOURCE: experiments/007-kuzmin-tm/scripts/fba_screen_medium_si.py -- AUTO-GENERATED, do not hand-edit; rerun the script.",
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\caption{The Yeast9 flux-balance baseline on four media. Open, exchange reactions open"
        r" for uptake (glucose bound in mmol\,gDW$^{-1}$\,h$^{-1}$ in parentheses); $\mu_{\mathrm{WT}}$,"
        r" wild-type growth (h$^{-1}$); WT-like, the percentage of single / double / triple deletion"
        r" sets whose predicted growth is within $10^{-3}$ of the wild type's; None, triples with no"
        r" growth (fitness below 0.01); $n_{\tau\neq 0}$, triples with predicted"
        r" $\lvert\tau_{ijk}\rvert>10^{-3}$; $r(\tau)$ and $r(f)$, Pearson $r$ of predicted against"
        r" measured $\tau_{ijk}$ and triple-mutant fitness over all 332,313 triples ($r(f)$ within the"
        r" 111,921 triples that contain a model gene in parentheses). The frozen run is the value of"
        r" Fig.~\ref{fig:ggi}d and its medium is \supptab{tab:fba-medium}; the rerun repeats it with"
        r" the current cobrapy; SM is ammonium nitrogen, the YNB vitamins, and glucose at 3.3 with no"
        r" amino acid; the screen medium is the SGA triple-mutant selection medium, 17 amino acids"
        r" and adenine at 0.165 and monosodium glutamate at 0.165.}",
        r"\label{tab:fba-screen-medium}",
        r"\begin{tabular}{@{}l r r l r r r r@{}}",
        r"\toprule",
        r"\textbf{Medium} & \textbf{Open} & $\mu_{\mathrm{WT}}$ & \textbf{WT-like 1 / 2 / 3} &"
        r" \textbf{None} & $n_{\tau\neq 0}$ & $r(\tau)$ & $r(f)$\\",
        r"\midrule",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"{SHORT_TEX[r.arm]} & {int(r.open_exchanges)} ({r.glucose_bound:g}) & {r.wt_growth:.4f} &"
            f" {100 * r.frac_wt_like_single:.1f} / {100 * r.frac_wt_like_double:.1f} / {100 * r.frac_wt_like_triple:.1f}\\% &"
            f" {100 * r.frac_lethal_triple:.1f}\\% & {int(r.n_tau_nonzero):,} & {r.pearson_r_tau:.4f} &"
            f" {r.pearson_r_fitness:.3f} ({r.pearson_r_fitness_covered:.3f})\\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"wrote {path}")


# ----------------------------------------------------------------------------- panels
def panel_tau(m: pd.DataFrame, st: dict):
    """Panel g: predicted against measured tau on the screen medium, the layout of panel b."""
    w = mm_to_in(PANEL_WIDTHS_MM["third"])
    fig, ax = plt.subplots(figsize=(w, mm_to_in(50)))
    fig.subplots_adjust(left=0.2, right=0.82, bottom=0.17, top=0.95)
    cmap = LinearSegmentedColormap.from_list("amber", ["#FFFFFF", ORANGE_F, ORANGE, PLOT_PALETTE[6]])
    hb = ax.hexbin(m.tau_fba, m.tau_measured, gridsize=(36, 24), extent=(-2.1, 2.1, -1.15, 1.15),
                   norm=LogNorm(vmin=1, vmax=max(1, len(m))), cmap=cmap, mincnt=1, linewidths=0.1, edgecolors="none")
    cb = fig.colorbar(hb, ax=ax, pad=0.02, fraction=0.06)
    cb.set_label("Triples", labelpad=1)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(width=0.5, length=2)
    ax.axhline(0, color="black", lw=0.4, ls=":")
    ax.axvline(0, color="black", lw=0.4, ls=":")
    ax.set_xlim(-2.1, 2.1)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel(r"Yeast9 FBA $\tau_{ijk}$, screen medium")
    ax.set_ylabel(r"Measured $\tau_{ijk}$")
    c = st["tau"]
    # Upper right: the data stand in a column at x = 0, so the upper-left corner is not free.
    ax.text(0.97, 0.97, f"SD/MSG -His/Arg/Lys/Ura\nPearson $r$ = {c['pearson_r']:.4f}\n$n$ = {c['n']:,}\n"
            f"{100 * c['frac_abs_below_1e-3']:.2f}% at $|\\tau|<10^{{-3}}$",
            transform=ax.transAxes, va="top", ha="right", fontsize=6)
    box(ax)
    save(fig, "fba_screen_medium_tau")


def panel_fitness(m: pd.DataFrame, st: dict):
    w = mm_to_in(PANEL_WIDTHS_MM["third"])
    fig, ax = plt.subplots(figsize=(w, mm_to_in(50)))
    fig.subplots_adjust(left=0.2, right=0.82, bottom=0.17, top=0.95)
    cmap = LinearSegmentedColormap.from_list("lilac", ["#FFFFFF", PURPLE_F, PURPLE, PLOT_PALETTE[8]])
    hb = ax.hexbin(m.fitness_fba, m.fitness_measured, gridsize=(30, 24), extent=(-0.05, 1.05, -0.05, 1.55),
                   norm=LogNorm(vmin=1, vmax=max(1, len(m))), cmap=cmap, mincnt=1, linewidths=0.1, edgecolors="none")
    cb = fig.colorbar(hb, ax=ax, pad=0.02, fraction=0.06)
    cb.set_label("Triples", labelpad=1)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(width=0.5, length=2)
    ax.plot([0, 1.05], [0, 1.05], color="black", lw=0.4, ls=":")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.55)
    ax.set_xlabel(r"Yeast9 FBA $f_{ijk}$, screen medium")
    ax.set_ylabel(r"Measured $f_{ijk}$")
    ax.text(0.03, 0.97, f"Pearson $r$ = {st['fitness']['pearson_r']:.4f}\n$n$ = {st['tau']['n']:,}",
            transform=ax.transAxes, va="top", ha="left", fontsize=6)
    box(ax)
    save(fig, "fba_screen_medium_fitness")


def panel_bands(rows: list[dict]):
    """Triple-deletion growth bands per medium: wild-type-like, reduced, no growth."""
    w = mm_to_in(PANEL_WIDTHS_MM["third"])
    fig, ax = plt.subplots(figsize=(w, mm_to_in(50)))
    fig.subplots_adjust(left=0.2, right=0.97, bottom=0.24, top=0.95)
    cats = [("frac_wt_like", "Wild-type growth", GRAY), ("frac_intermediate", "Reduced", ORANGE), ("frac_lethal", "No growth", RED)]
    x = np.arange(len(rows))
    bw = 0.26
    for k, (col, label, color) in enumerate(cats):
        vals = np.array([next(b for b in st["growth_bands"] if b["order"] == "triple")[col] for st in rows])
        ax.bar(x + (k - 1) * bw, vals, bw, color=color, edgecolor="black", linewidth=0.4, label=label)
        for xi, v in zip(x + (k - 1) * bw, vals):
            ax.text(xi, v + 0.015, f"{100 * v:.1f}", ha="center", va="bottom", fontsize=5, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[st["arm"]] for st in rows])
    ax.set_ylabel("Fraction of triple deletions")
    ax.set_ylim(0, 1.5)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0, 1.01, 0.1), minor=True)
    ax.tick_params(axis="y", which="minor", length=0)
    ax.grid(axis="y", which="both", color="0.85", lw=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, handlelength=1.0, handleheight=0.8, borderaxespad=0.2)
    box(ax)
    save(fig, "fba_screen_medium_bands")


def main():
    base = json.load(open(osp.join(BASELINE_OUT, "stats.json")))
    labels, raw_meta = raw_labels()
    gem, model = load_model()
    perts = json.load(open(osp.join(REPO_ROOT, "experiments", "007-kuzmin-tm", "results",
                                    "cobra-fba-growth_backup_20250923_134447", "unique_perturbations.json")))
    rows, matched = [frozen_row(base)], {}
    for arm in ORDER[1:]:
        st, m = arm_stats(arm, load_arm(arm), labels, model, perts)
        rows.append(st)
        matched[arm] = m
        m.to_parquet(osp.join(OUT, f"triples_matched_{arm}.parquet"), index=False)
    summary = summary_table(rows)
    summary.to_csv(osp.join(OUT, "summary.csv"), index=False)
    rerun, frozen = rows[1], rows[0]
    check = {
        "rerun_vs_frozen_wt_growth_abs_diff": abs(rerun["wt"]["growth"] - frozen["wt"]["growth"]),
        "rerun_vs_frozen_pearson_r_tau_abs_diff": abs(rerun["tau"]["pearson_r"] - frozen["tau"]["pearson_r"]),
        "rerun_vs_frozen_n_tau_nonzero": [rerun["tau"]["n_abs_above_1e-3"], frozen["tau"]["n_abs_above_1e-3"]],
        "rerun_vs_frozen_triples_wt_like": [
            next(b for b in rerun["growth_bands"] if b["order"] == "triple")["n_wt_like"],
            next(b for b in frozen["growth_bands"] if b["order"] == "triple")["n_wt_like"],
        ],
    }
    stats_out = {
        "arms": {st["arm"]: st for st in rows},
        "summary": summary.to_dict(orient="records"),
        "rerun_reproduces_frozen": check,
        "raw_labels": raw_meta,
        "thresholds": {"wt_like": WT_LIKE, "tau_zero": TAU_ZERO},
        "model": {"version": gem.version, "id": model.id},
    }
    json.dump(stats_out, open(osp.join(OUT, "stats.json"), "w"), indent=2)
    print(summary.to_string())
    print(json.dumps(check, indent=2))
    write_table(summary, osp.join(TEX_DIR, "tab-fba-screen-medium.tex"))
    panel_tau(matched["screen_medium"], rows[-1])
    panel_fitness(matched["screen_medium"], rows[-1])
    panel_bands(rows)


if __name__ == "__main__":
    main()
