# experiments/031-env-chemgen-inhibitor-tolerance/scripts/replicate_noise_and_ceilings.py
# [[experiments.031-env-chemgen-inhibitor-tolerance.scripts.replicate_noise_and_ceilings]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/031-env-chemgen-inhibitor-tolerance/scripts/replicate_noise_and_ceilings
"""Replication, error, and prediction ceilings for Vanacloig 2022 and Hillenmeyer 2008.

Two views of noise, per condition, and where they can be joined, against each other:

1. **Served uncertainty.** From the flattened records (``flatten_records.py``): the
   replicate count ``n_samples``, the served SE, and the reliability index
   ``rel = 1 - mean(SE^2) / var(response)`` over genes. The **ceiling** on the Pearson
   correlation between a measurement and the noise-free truth is ``sqrt(rel)``; the
   ceiling on the correlation between two independent measurements of the same
   condition (which is what a held-out replicate or a second dataset can reach) is
   ``rel`` itself.
2. **Empirical replicates.** From the raw matrices the loaders consumed: Vanacloig's
   three batch replicates (one column per batch, log2 CPM ratio against that batch's
   own control columns, the loader's quantity), and Hillenmeyer's replicate arrays of
   one condition label (HOM z-score, HET log-ratio, one column per array). Pairwise
   Spearman between replicates over genes, and the Jaccard of the bottom-5% hit sets
   between replicates, are the model-free reliability of a condition and of its hits.

Raw condition labels are mapped to the served compound name with the same resolver
the loaders use (``resolved_compound``), so the two views join on the served name.

Writes under ``results/``: ``noise_summary.csv`` (one row per dataset),
``condition_noise_<dataset>.csv`` (one row per condition), and three figures under
``$ASSET_IMAGES_DIR/031-env-chemgen-inhibitor-tolerance/``.
"""

from __future__ import annotations

import os
import os.path as osp
import re
import sys
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from torchcell.datamodels.compound_identity import resolved_compound  # noqa: E402
from torchcell.timestamp import timestamp  # noqa: E402
from torchcell.utils import (  # noqa: E402
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    panel_label,
    savefig_true_size_svg,
)

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
RESULTS_DIR = osp.join(
    EXPERIMENT_ROOT, "031-env-chemgen-inhibitor-tolerance", "results"
)
IMAGES_DIR = osp.join(ASSET_IMAGES_DIR, "031-env-chemgen-inhibitor-tolerance")
#: ``--stable`` writes un-timestamped file names (the reviewed figures the note and the
#: notes-tex document reference); the default keeps the timestamp for iteration.
STABLE_NAMES = "--stable" in sys.argv
RAW = osp.join(DATA_ROOT, "data", "torchcell")
INK = "#000000"
ORANGE, RED, PURPLE, YELLOW, BLUE, GRAY = PLOT_PALETTE[:6]
HIT_Q = 0.05
DATASETS = ["vanacloig2022", "hillenmeyer2008_hom", "hillenmeyer2008_het"]
LABELS = {
    "vanacloig2022": "Vanacloig 2022",
    "hillenmeyer2008_hom": "Hillenmeyer HOM",
    "hillenmeyer2008_het": "Hillenmeyer HET",
}
COLORS = {
    "vanacloig2022": PURPLE,
    "hillenmeyer2008_hom": RED,
    "hillenmeyer2008_het": BLUE,
}


def _apply_rc() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
            "font.size": 6,
            "axes.titlesize": 6,
            "axes.labelsize": 6,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "svg.fonttype": "none",
            "axes.linewidth": 0.5,
            "savefig.bbox": None,
        }
    )


def _box(ax: Axes) -> None:
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_color(INK)
        ax.spines[s].set_linewidth(0.5)
    ax.tick_params(colors=INK, width=0.5, length=2)


def _save(fig: plt.Figure, name: str) -> str:
    os.makedirs(IMAGES_DIR, exist_ok=True)
    stem = osp.join(IMAGES_DIR, name if STABLE_NAMES else f"{name}_{timestamp()}")
    fig.savefig(stem + ".png", dpi=300)
    savefig_true_size_svg(fig, stem + ".svg")
    return stem + ".svg"


# ------------------------------------------------------------------ served view
def _background(df: pd.DataFrame) -> set[str]:
    sets = [set(g.split("|")) for g in df["gene"].unique()]
    return set.intersection(*sets) if len(sets) > 1 else set()


def condition_label(df: pd.DataFrame) -> pd.Series:
    """Coarse condition: the dosed compound, else the physical factor, else the raised or
    lowered temperature, else the swapped medium (Hillenmeyer's SD, SC-dropout and YP
    glycerol arms carry no compound, no physical factor and an unstated temperature).
    """
    cond = df["compound"].where(df["compound"] != "", df["physical"])
    temp = df["temperature_c"].map(lambda t: "" if pd.isna(t) else f"T={t:g}")
    cond = cond.where(cond != "", temp)
    medium = df["media_name"].str.split(" (", regex=False).str[0].str.strip()
    return cond.where(cond != "", "medium=" + medium)


def served_condition_table(name: str) -> pd.DataFrame:
    """Per served condition: replicate count, SE, spread, reliability, ceilings."""
    df = pd.read_parquet(osp.join(RESULTS_DIR, f"records_{name}.parquet"))
    bg = _background(df)
    df["query_gene"] = df["gene"].map(
        lambda g: "|".join(x for x in g.split("|") if x not in bg)
    )
    df = df[df["query_gene"].str.count(r"\|") == 0]
    df = df.assign(condition=condition_label(df))
    rows = []
    for c, g in df.groupby("condition"):
        per_gene = g.groupby("query_gene").agg(
            r=("response", "mean"),
            se2=("response_se", lambda s: np.nanmean(s**2)),
            k=("response", "size"),
        )
        se2 = per_gene["se2"] / per_gene["k"]
        var = per_gene["r"].var()
        rel = np.nan if se2.isna().all() else 1 - np.nanmean(se2) / var
        rows.append(
            {
                "dataset": name,
                "condition": c,
                "n_genes": len(per_gene),
                "records_per_gene": per_gene["k"].mean(),
                "n_samples_mean": g["n_samples"].mean(),
                "frac_with_se": g["response_se"].notna().mean(),
                "se_median": np.nanmedian(g["response_se"]),
                "response_sd": np.sqrt(var),
                "reliability": rel,
                "ceiling_r_truth": np.sqrt(max(rel, 0)) if np.isfinite(rel) else np.nan,
                "ceiling_r_replicate": max(rel, 0) if np.isfinite(rel) else np.nan,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------- empirical view
def _pairwise(M: pd.DataFrame) -> tuple[float, float, int]:
    """Mean pairwise Spearman and mean pairwise hit Jaccard across the columns of M."""
    rhos, jacs = [], []
    for i, j in combinations(M.columns, 2):
        ok = M[i].notna() & M[j].notna()
        if ok.sum() < 200:
            continue
        a, b = M.loc[ok, i], M.loc[ok, j]
        rhos.append(spearmanr(a, b)[0])
        ha, hb = a <= a.quantile(HIT_Q), b <= b.quantile(HIT_Q)
        jacs.append((ha & hb).sum() / (ha | hb).sum())
    if not rhos:
        return np.nan, np.nan, 0
    return float(np.mean(rhos)), float(np.mean(jacs)), len(rhos)


def _served_name(label: str) -> str:
    """Raw condition label -> the served compound name, through the loaders' resolver."""
    c = resolved_compound(label)
    return c.name


def empirical_vanacloig() -> pd.DataFrame:
    vp = pd.read_csv(
        osp.join(
            RAW,
            "env_chemgen_vanacloig2022/raw/GSE186866_ChemGenomics_Raw_Counts_matrix.txt.gz",
        ),
        sep="\t",
    )
    vp["orf"] = vp["gene"].str.split("_").str[0]
    counts = (
        vp.drop(columns=["gene", "std_name"]).set_index("orf").groupby(level=0).sum()
    )
    counts = counts.loc[counts.sum(axis=1) > 0]
    cpm = counts / counts.sum(axis=0) * 1e6
    cols = pd.DataFrame(
        [re.match(r"(.+?)_(CG\d{3})(?:_rep(\d+))?$", c).groups() for c in cpm.columns],
        columns=["token", "batch", "rep"],
        index=cpm.columns,
    )
    ctrl = {
        b: cpm[
            cols.index[(cols.batch == b) & cols.token.str.startswith("Control")]
        ].mean(axis=1)
        for b in cols.batch.unique()
    }
    rows = []
    for token, grp in cols[~cols.token.str.startswith("Control")].groupby("token"):
        if token in ("QUADRIS1", "QUADRIS2", "MBO", "DMSO"):
            continue
        M = pd.concat(
            [
                np.log2((cpm[c] + 1) / (ctrl[grp.loc[c, "batch"]] + 1)).rename(c)
                for c in grp.index
            ],
            axis=1,
        )
        rho, jac, n = _pairwise(M)
        rows.append(
            {
                "dataset": "vanacloig2022",
                "raw_label": token,
                "condition": _served_name(token),
                "n_replicates": M.shape[1],
                "replicate_rho": rho,
                "hit_jaccard": jac,
                "n_pairs": n,
            }
        )
    return pd.DataFrame(rows)


def empirical_hillenmeyer(arm: str) -> pd.DataFrame:
    d = osp.join(RAW, f"env_chemgen_hillenmeyer2008_{arm}/raw")
    mat = pd.read_csv(
        osp.join(
            d, "hom.z_result_nm.pub" if arm == "hom" else "het.ratio_result_nm.pub"
        ),
        sep="\t",
    )
    mat["orf"] = mat["Orf"].str.split(":").str[0]
    mat = mat.drop(columns=["Orf"]).set_index("orf").groupby(level=0).mean()
    mat.columns = [c.split(":")[0] for c in mat.columns]
    meta = pd.read_csv(osp.join(d, f"{arm}.txt"), sep="\t").set_index("filename")
    cond = meta["condition"].reindex(mat.columns)
    rows = []
    for label, grp in cond.groupby(cond.values):
        if len(grp) < 2:
            continue
        rho, jac, n = _pairwise(-mat[grp.index])  # growth orientation for the hit call
        c = resolved_compound(label)
        rows.append(
            {
                "dataset": f"hillenmeyer2008_{arm}",
                "raw_label": label,
                "condition": c.name,
                "n_replicates": len(grp),
                "replicate_rho": rho,
                "hit_jaccard": jac,
                "n_pairs": n,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- figures
def fig_reliability_distributions(cond: pd.DataFrame) -> str:
    fig, axes = plt.subplots(
        1, 3, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(50))
    )
    fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.18, wspace=0.3)
    rng = np.random.default_rng(0)
    for ax, (col, ylabel, letter) in zip(
        axes,
        [
            ("reliability", "reliability index (served SE)", "a"),
            ("replicate_rho", "replicate Spearman (raw)", "b"),
            ("hit_jaccard", "bottom-5% hit Jaccard (raw)", "c"),
        ],
        strict=True,
    ):
        for k, name in enumerate(DATASETS):
            v = cond.loc[cond.dataset == name, col].dropna()
            x = k + rng.uniform(-0.18, 0.18, len(v))
            ax.scatter(
                x, v, s=4, color=COLORS[name], edgecolors=INK, linewidths=0.2, alpha=0.8
            )
            ax.hlines(v.median(), k - 0.3, k + 0.3, color=INK, linewidth=0.8)
            ax.text(k, 1.08, f"n={len(v)}", ha="center", va="center", fontsize=5)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["Vanacloig", "HOM", "HET"])
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.2 if col == "reliability" else 0, 1.15)
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.tick_params(which="minor", length=0)
        ax.grid(True, axis="y", which="both", linewidth=0.3, alpha=0.3)
        _box(ax)
        panel_label(ax, letter)
    path = _save(fig, "noise_distributions")
    plt.close(fig)
    return path


def fig_index_vs_empirical(cond: pd.DataFrame) -> str:
    fig, axes = plt.subplots(
        1, 2, figsize=(mm_to_in(PANEL_WIDTHS_MM["wide"]), mm_to_in(55))
    )
    fig.subplots_adjust(left=0.09, right=0.99, top=0.86, bottom=0.18, wspace=0.3)
    for ax, (col, xlabel, letter) in zip(
        axes,
        [
            ("replicate_rho", "replicate Spearman (raw)", "d"),
            ("hit_jaccard", "bottom-5% hit Jaccard (raw)", "e"),
        ],
        strict=True,
    ):
        for name in DATASETS:
            s = cond[(cond.dataset == name)].dropna(subset=[col, "reliability"])
            if not len(s):
                continue
            r = spearmanr(s[col], s["reliability"])[0]
            ax.scatter(
                s[col],
                s["reliability"],
                s=6,
                color=COLORS[name],
                edgecolors=INK,
                linewidths=0.2,
                label=f"{LABELS[name]} (n={len(s)}, rho {r:+.2f})",
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("reliability index (served SE)")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.2, 1)
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.grid(True, which="major", linewidth=0.3, alpha=0.3)
        ax.legend(frameon=False, loc="lower right")
        _box(ax)
        panel_label(ax, letter)
    path = _save(fig, "reliability_index_vs_replicates")
    plt.close(fig)
    return path


def fig_vanacloig_ceilings(cond: pd.DataFrame) -> str:
    v = cond[cond.dataset == "vanacloig2022"].sort_values(
        "ceiling_r_truth", ascending=False
    )
    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(80)))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.84, bottom=0.44)
    x = np.arange(len(v))
    ax.bar(
        x,
        v["ceiling_r_truth"].fillna(0),
        color=ORANGE,
        edgecolor=INK,
        linewidth=0.4,
        label="ceiling on r vs truth, sqrt(reliability)",
    )
    ax.plot(
        x,
        v["replicate_rho"],
        "o",
        color=PURPLE,
        markersize=2.5,
        markeredgecolor=INK,
        markeredgewidth=0.3,
        label="replicate Spearman (raw, 3 batches)",
    )
    ax.plot(
        x,
        v["hit_jaccard"],
        "s",
        color=BLUE,
        markersize=2.5,
        markeredgecolor=INK,
        markeredgewidth=0.3,
        label="bottom-5% hit Jaccard between batches",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [c if len(c) <= 26 else c[:24] + ".." for c in v["condition"]], rotation=90
    )
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which="minor", length=0)
    ax.grid(True, axis="y", which="both", linewidth=0.3, alpha=0.3)
    ax.set_ylabel("ceiling / agreement")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    _box(ax)
    panel_label(ax, "f")
    path = _save(fig, "vanacloig_ceilings")
    plt.close(fig)
    return path


def main() -> None:
    _apply_rc()
    served = pd.concat([served_condition_table(n) for n in DATASETS], ignore_index=True)
    emp = pd.concat(
        [
            empirical_vanacloig(),
            empirical_hillenmeyer("hom"),
            empirical_hillenmeyer("het"),
        ],
        ignore_index=True,
    )
    # one raw label per served name in Vanacloig; Hillenmeyer can have several labels
    # (doses) per served name, so the empirical numbers are averaged over them
    emp_c = emp.groupby(["dataset", "condition"], as_index=False).agg(
        n_raw_labels=("raw_label", "size"),
        n_replicates=("n_replicates", "sum"),
        replicate_rho=("replicate_rho", "mean"),
        hit_jaccard=("hit_jaccard", "mean"),
    )
    cond = served.merge(emp_c, on=["dataset", "condition"], how="left")
    for n in DATASETS:
        cond[cond.dataset == n].drop(columns="dataset").to_csv(
            osp.join(RESULTS_DIR, f"condition_noise_{n}.csv"), index=False
        )
    emp.to_csv(osp.join(RESULTS_DIR, "raw_replicate_agreement.csv"), index=False)

    summ = []
    for n in DATASETS:
        s, e = cond[cond.dataset == n], emp[emp.dataset == n]
        df = pd.read_parquet(
            osp.join(RESULTS_DIR, f"records_{n}.parquet"),
            columns=["n_samples", "response_se", "response"],
        )
        summ.append(
            {
                "dataset": n,
                "records": len(df),
                "n_samples_1": (df.n_samples == 1).mean(),
                "n_samples_ge3": (df.n_samples >= 3).mean(),
                "frac_with_se": df.response_se.notna().mean(),
                "se_median": np.nanmedian(df.response_se),
                "response_sd": df.response.std(),
                "conditions": len(s),
                "reliability_median": s.reliability.median(),
                "reliability_q25": s.reliability.quantile(0.25),
                "ceiling_r_truth_median": s.ceiling_r_truth.median(),
                "conditions_no_signal": int((s.reliability <= 0).sum()),
                "raw_conditions_with_replicates": len(e),
                "replicate_rho_median": e.replicate_rho.median(),
                "hit_jaccard_median": e.hit_jaccard.median(),
                "index_vs_replicate_rho": spearmanr(
                    *s.dropna(subset=["replicate_rho", "reliability"])[
                        ["replicate_rho", "reliability"]
                    ].values.T
                )[0]
                if s.replicate_rho.notna().sum() > 3
                else np.nan,
                "unjoined_raw_labels": int((~e.condition.isin(s.condition)).sum()),
            }
        )
    summary = pd.DataFrame(summ)
    summary.to_csv(osp.join(RESULTS_DIR, "noise_summary.csv"), index=False)
    print(summary.T.to_string())
    paths = [
        fig_reliability_distributions(cond),
        fig_index_vs_empirical(cond),
        fig_vanacloig_ceilings(cond),
    ]
    print("\n".join(paths))


if __name__ == "__main__":
    main()
