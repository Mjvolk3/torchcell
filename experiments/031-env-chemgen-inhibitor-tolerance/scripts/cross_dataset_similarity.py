# experiments/031-env-chemgen-inhibitor-tolerance/scripts/cross_dataset_similarity.py
# [[experiments.031-env-chemgen-inhibitor-tolerance.scripts.cross_dataset_similarity]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/031-env-chemgen-inhibitor-tolerance/scripts/cross_dataset_similarity
"""How much of Vanacloig 2022's per-gene response structure does Hillenmeyer 2008 share?

Reads the flattened served records (``flatten_records.py``) and builds one gene x
condition matrix per dataset, oriented so NEGATIVE = fitness defect: Vanacloig's
log2(inhibitor/control) as stored; Hillenmeyer's fitness-defect scores negated. A
condition is the coarse label (compound, else physical factor, else temperature); the
several doses / generation counts / screens Hillenmeyer ran for one compound are
averaged, so this is the most favorable view for a compound-level comparison.

Writes to ``results/``:
- ``reliability.csv``: per condition, a served-uncertainty reliability index
  ``1 - mean(SE^2) / var(response)`` over genes (the share of across-gene variance not
  attributable to replicate noise; the ceiling for any cross-dataset correlation).
- ``cross_spearman_<partner>.csv``: Spearman between every Vanacloig condition and
  every partner condition over the shared queried genes.
- ``top_matches_<partner>.csv``: the two best partner conditions per Vanacloig condition.
- ``shared_compounds_<partner>.csv``: for compounds both datasets dose, the correlation,
  the rank of the true match among all partner conditions, and the hit overlap
  (hit = bottom 5% of genes; Fisher one-sided).
- ``structure_<partner>.csv``: the per-gene mean-response correlation and the
  gene-gene similarity agreement (Mantel-style Spearman on the two gene-profile
  correlation matrices, against a gene-permutation null).

Figures go to ``$ASSET_IMAGES_DIR/031-env-chemgen-inhibitor-tolerance/`` as true-size
SVG + PNG (repo standards: palette, Arial 6 pt, boxed axes, standard panel widths).
"""

from __future__ import annotations

import os
import os.path as osp
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402
from scipy.stats import fisher_exact, spearmanr  # noqa: E402

from torchcell.timestamp import timestamp  # noqa: E402
from torchcell.utils import (  # noqa: E402
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    panel_label,
    savefig_true_size_svg,
)

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
RESULTS_DIR = osp.join(
    EXPERIMENT_ROOT, "031-env-chemgen-inhibitor-tolerance", "results"
)
IMAGES_DIR = osp.join(ASSET_IMAGES_DIR, "031-env-chemgen-inhibitor-tolerance")
#: ``--stable`` writes un-timestamped file names (the reviewed figures the note and the
#: notes-tex document reference); the default keeps the timestamp for iteration.
STABLE_NAMES = "--stable" in sys.argv
INK = "#000000"
ORANGE, RED, PURPLE, YELLOW, BLUE, GRAY = PLOT_PALETTE[:6]
HIT_QUANTILE = 0.05
MIN_GENES = 1000
SEED = 0

PARTNERS = {"hom": "hillenmeyer2008_hom", "het": "hillenmeyer2008_het"}
# Vanacloig compound -> Hillenmeyer condition label, for compounds dosed by both (or the
# salt/acid near-pair, which the graph does NOT join and is here as the negative control).
SHARED = {
    "benomyl": "benomyl",
    "methyl methanesulfonate": "methyl methanesulfonate",
    "ferulic acid": "ferulic acid",
    "sodium acetate": "acetic acid",
}


def _apply_rc() -> None:
    """Repo type + rule standards (Arial 6 pt, editable SVG text, 0.5 pt lines)."""
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
    """Full black border on all four spines."""
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_color(INK)
        ax.spines[s].set_linewidth(0.5)
    ax.tick_params(colors=INK, width=0.5, length=2)


def _save(fig: plt.Figure, name: str) -> str:
    """Save PNG + true-size SVG with a timestamp; return the SVG path."""
    os.makedirs(IMAGES_DIR, exist_ok=True)
    stem = osp.join(IMAGES_DIR, name if STABLE_NAMES else f"{name}_{timestamp()}")
    fig.savefig(stem + ".png", dpi=300)
    savefig_true_size_svg(fig, stem + ".svg")
    return stem + ".svg"


# ----------------------------------------------------------------------------- data
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


def load_matrix(name: str, sign: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(gene x condition response matrix, gene x condition SE matrix), growth-oriented."""
    df = pd.read_parquet(osp.join(RESULTS_DIR, f"records_{name}.parquet"))
    bg = _background(df)
    df["query_gene"] = df["gene"].map(
        lambda g: "|".join(x for x in g.split("|") if x not in bg)
    )
    df = df[df["query_gene"].str.count(r"\|") == 0]  # single queried gene only
    df = df.assign(condition=condition_label(df), r=sign * df["response"])
    g = df.groupby(["query_gene", "condition"])
    M = g["r"].mean().unstack()
    n = g["r"].size().unstack()
    # SE of the condition-level mean: pooled SE of the averaged records / sqrt(records)
    se2 = (df["response_se"] ** 2).groupby(
        [df["query_gene"], df["condition"]]
    ).mean().unstack() / n
    return M, np.sqrt(se2)


def reliability(M: pd.DataFrame, SE: pd.DataFrame, name: str) -> pd.DataFrame:
    """1 - mean(SE^2)/var(response) per condition; NaN when no SE is served."""
    rows = []
    for c in M.columns:
        r, s = M[c].dropna(), SE[c].reindex(M[c].dropna().index)
        rel = np.nan if s.isna().all() else 1 - np.nanmean(s**2) / r.var()
        rows.append(
            {
                "dataset": name,
                "condition": c,
                "n_genes": len(r),
                "var_response": r.var(),
                "mean_se2": np.nanmean(s**2),
                "reliability": rel,
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------- analyses
def cross_similarity(V: pd.DataFrame, H: pd.DataFrame) -> pd.DataFrame:
    """Spearman between every V condition (rows) and every H condition (columns)."""
    shared = V.index.intersection(H.index)
    X = pd.concat(
        [V.loc[shared].add_prefix("V|"), H.loc[shared].add_prefix("H|")], axis=1
    )
    rho = X.corr(method="spearman", min_periods=MIN_GENES)
    return rho.loc[
        [c for c in rho.index if c.startswith("V|")],
        [c for c in rho.columns if c.startswith("H|")],
    ].rename(index=lambda s: s[2:], columns=lambda s: s[2:])


def shared_compound_stats(
    V: pd.DataFrame, H: pd.DataFrame, cross: pd.DataFrame
) -> pd.DataFrame:
    """Correlation, rank of the true match, and hit overlap for each shared compound."""
    shared = V.index.intersection(H.index)
    rows = []
    for v, h in SHARED.items():
        if v not in V.columns or h not in H.columns:
            continue
        a, b = V.loc[shared, v], H.loc[shared, h]
        ok = a.notna() & b.notna()
        a, b = a[ok], b[ok]
        ha, hb = a <= a.quantile(HIT_QUANTILE), b <= b.quantile(HIT_QUANTILE)
        tab = [
            [int((ha & hb).sum()), int((ha & ~hb).sum())],
            [int((~ha & hb).sum()), int((~ha & ~hb).sum())],
        ]
        odds, p = fisher_exact(tab, alternative="greater")
        row = cross.loc[v]
        rows.append(
            {
                "vanacloig": v,
                "hillenmeyer": h,
                "n_genes": int(ok.sum()),
                "spearman": row[h],
                "rank_of_true_match": int((row > row[h]).sum()) + 1,
                "n_partner_conditions": len(row),
                "best_match": row.idxmax(),
                "best_rho": row.max(),
                "hits_each": int(ha.sum()),
                "hit_overlap": tab[0][0],
                "hit_overlap_expected": ha.sum() * hb.sum() / ok.sum(),
                "odds_ratio": odds,
                "fisher_p": p,
            }
        )
    return pd.DataFrame(rows)


def structure_agreement(
    V: pd.DataFrame, H: pd.DataFrame, n_perm: int = 20
) -> dict[str, float]:
    """Per-gene mean-response correlation + Mantel-style gene-gene similarity agreement."""
    shared = V.index.intersection(H.index)
    V, H = V.loc[shared], H.loc[shared]
    gV, gH = V.mean(axis=1), H.mean(axis=1)
    ok = gV.notna() & gH.notna()
    mean_rho = spearmanr(gV[ok], gH[ok])[0]

    def gene_corr(M: pd.DataFrame) -> pd.DataFrame:
        M = M.loc[:, M.notna().mean() > 0.8]
        M = M.loc[M.notna().mean(axis=1) > 0.8]
        return M.T.corr(method="pearson", min_periods=10)

    rng = np.random.default_rng(SEED)
    keep = V.index[(V.std(axis=1) > 0) & (H.std(axis=1) > 0)]
    sub = rng.choice(keep, size=min(1500, len(keep)), replace=False)
    cV, cH = gene_corr(V.loc[sub]), gene_corr(H.loc[sub])
    common = cV.index.intersection(cH.index)
    cV, cH = cV.loc[common, common], cH.loc[common, common]
    iu = np.triu_indices(len(common), 1)
    a, b = cV.values[iu], cH.values[iu]
    ok2 = ~np.isnan(a) & ~np.isnan(b)
    obs = spearmanr(a[ok2], b[ok2])[0]
    null = []
    for _ in range(n_perm):
        p = rng.permutation(len(common))
        bp = cH.values[np.ix_(p, p)][iu]
        okp = ~np.isnan(a) & ~np.isnan(bp)
        null.append(spearmanr(a[okp], bp[okp])[0])
    return {
        "n_shared_genes": len(shared),
        "mean_response_spearman": mean_rho,
        "n_mean_genes": int(ok.sum()),
        "mantel_genes": len(common),
        "mantel_spearman": obs,
        "mantel_null_mean": float(np.mean(null)),
        "mantel_null_sd": float(np.std(null)),
        "n_perm": n_perm,
    }


# ------------------------------------------------------------------------ figures
def fig_heatmap(cross: pd.DataFrame, partner: str, top_k: int = 40) -> str:
    """Vanacloig conditions x the top-K partner conditions by max |rho|."""
    cols = cross.abs().max(axis=0).sort_values(ascending=False).index[:top_k]
    sub = cross[cols]
    order = sub.max(axis=1).sort_values(ascending=False).index
    sub = sub.loc[order]
    cmap = LinearSegmentedColormap.from_list("bwo", [BLUE, "#FFFFFF", ORANGE])
    lim = float(np.nanmax(np.abs(sub.values)))
    short = lambda s: s if len(s) <= 26 else s[:24] + ".."  # noqa: E731
    fig = plt.figure(figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(125)))
    ax = fig.add_axes([0.24, 0.36, 0.66, 0.57])
    im = ax.imshow(
        sub.values,
        cmap=cmap,
        vmin=-lim,
        vmax=lim,
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([short(c) for c in cols], rotation=90)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([short(c) for c in order])
    ax.set_xlabel(
        f"Hillenmeyer 2008 {partner.upper()} condition (top {top_k} by max |rho|)"
    )
    ax.set_ylabel("Vanacloig 2022 condition")
    _box(ax)
    cax = fig.add_axes([0.915, 0.36, 0.015, 0.57])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Spearman rho over shared genes")
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(width=0.5, length=2)
    panel_label(ax, "a")
    path = _save(fig, f"cross_similarity_heatmap_{partner}")
    plt.close(fig)
    return path


def fig_reliability(
    rel_v: pd.DataFrame, rel_h: pd.DataFrame, cross: pd.DataFrame, partner: str
) -> str:
    """Per Vanacloig condition: reliability index (bar) and best cross-dataset rho (marker)."""
    r = rel_v.set_index("condition").loc[cross.index]
    best = cross.max(axis=1)
    order = r["reliability"].sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(80)))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.90, bottom=0.50)
    x = np.arange(len(order))
    ax.bar(
        x,
        r.loc[order, "reliability"].clip(lower=0),
        color=PURPLE,
        edgecolor=INK,
        linewidth=0.4,
        label="reliability index (served SE)",
    )
    ax.plot(
        x,
        best.loc[order],
        "o",
        color=RED,
        markersize=2.5,
        markeredgecolor=INK,
        markeredgewidth=0.3,
        label=f"best Spearman vs any {partner.upper()} condition",
    )
    med = np.nanmedian(rel_h["reliability"])
    ax.axhline(
        med,
        color=BLUE,
        linewidth=0.6,
        linestyle="--",
        label=f"{partner.upper()} median reliability",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [c if len(c) <= 26 else c[:24] + ".." for c in order], rotation=90
    )
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which="minor", length=0)
    ax.grid(True, axis="y", which="both", linewidth=0.3, alpha=0.3)
    ax.set_ylabel("index / rho")
    ax.legend(frameon=False, loc="center right")
    _box(ax)
    panel_label(ax, "b")
    path = _save(fig, f"reliability_vs_cross_{partner}")
    plt.close(fig)
    return path


def fig_shared_scatter(
    V: pd.DataFrame, H: pd.DataFrame, stats: pd.DataFrame, partner: str
) -> str:
    """Scatter of shared-gene responses for each shared compound, hits colored."""
    shared = V.index.intersection(H.index)
    n = len(stats)
    fig, axes = plt.subplots(
        1, n, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(60))
    )
    fig.subplots_adjust(left=0.06, right=0.99, top=0.84, bottom=0.16, wspace=0.30)
    for ax, (_, s), letter in zip(
        np.atleast_1d(axes), stats.iterrows(), "cde", strict=False
    ):
        a, b = V.loc[shared, s["vanacloig"]], H.loc[shared, s["hillenmeyer"]]
        ok = a.notna() & b.notna()
        a, b = a[ok], b[ok]
        both = (a <= a.quantile(HIT_QUANTILE)) & (b <= b.quantile(HIT_QUANTILE))
        ax.scatter(a[~both], b[~both], s=1.5, color=GRAY, alpha=0.5, linewidths=0)
        ax.scatter(
            a[both],
            b[both],
            s=3,
            color=RED,
            edgecolors=INK,
            linewidths=0.2,
            label=f"hit in both ({int(both.sum())})",
        )
        ax.set_xlabel(f"Vanacloig {s['vanacloig']} log2 ratio")
        ax.set_ylabel(f"{partner.upper()} {s['hillenmeyer']} (growth-oriented)")
        ax.set_title(
            f"rho {s['spearman']:+.3f}, rank {s['rank_of_true_match']}/{s['n_partner_conditions']}, Fisher p {s['fisher_p']:.1e}"
        )
        ax.legend(frameon=False, loc="lower right")
        _box(ax)
        panel_label(ax, letter)
    path = _save(fig, f"shared_compound_scatter_{partner}")
    plt.close(fig)
    return path


def main() -> None:
    """Run every analysis for HOM and HET (``--partners``) as partners of Vanacloig."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--partners", nargs="+", default=list(PARTNERS), choices=list(PARTNERS)
    )
    ap.add_argument(
        "--stable", action="store_true", help="write un-timestamped figure names"
    )
    args = ap.parse_args()
    _apply_rc()
    V, SEv = load_matrix("vanacloig2022", +1.0)
    rel_v = reliability(V, SEv, "vanacloig2022")
    rel_all = [rel_v]
    paths = []
    for partner in args.partners:
        name = PARTNERS[partner]
        H, SEh = load_matrix(name, -1.0)
        rel_h = reliability(H, SEh, name)
        rel_all.append(rel_h)
        cross = cross_similarity(V, H)
        cross.to_csv(osp.join(RESULTS_DIR, f"cross_spearman_{partner}.csv"))
        top = pd.DataFrame(
            {
                "vanacloig": cross.index,
                "best": cross.idxmax(axis=1).values,
                "best_rho": cross.max(axis=1).values,
                "second": [
                    cross.loc[i].drop(cross.loc[i].idxmax()).idxmax()
                    for i in cross.index
                ],
                "second_rho": [
                    cross.loc[i].drop(cross.loc[i].idxmax()).max() for i in cross.index
                ],
            }
        ).sort_values("best_rho", ascending=False)
        top.to_csv(osp.join(RESULTS_DIR, f"top_matches_{partner}.csv"), index=False)
        stats = shared_compound_stats(V, H, cross)
        stats.to_csv(
            osp.join(RESULTS_DIR, f"shared_compounds_{partner}.csv"), index=False
        )
        struct = structure_agreement(V, H)
        vals = cross.values[~np.isnan(cross.values)]
        struct.update(
            {
                "cross_median": float(np.median(vals)),
                "cross_p95": float(np.percentile(vals, 95)),
                "cross_max": float(vals.max()),
                "n_v_conditions": cross.shape[0],
                "n_h_conditions": cross.shape[1],
            }
        )
        pd.DataFrame([struct]).to_csv(
            osp.join(RESULTS_DIR, f"structure_{partner}.csv"), index=False
        )
        paths += [
            fig_heatmap(cross, partner),
            fig_reliability(rel_v, rel_h, cross, partner),
        ]
        if len(stats):
            paths.append(fig_shared_scatter(V, H, stats, partner))
        print(f"\n=== {partner}: {struct}")
        print(stats.to_string(index=False))
        print(top.head(15).to_string(index=False))
    pd.concat(rel_all).to_csv(osp.join(RESULTS_DIR, "reliability.csv"), index=False)
    print(
        "\nreliability medians:",
        pd.concat(rel_all).groupby("dataset")["reliability"].median().to_dict(),
    )
    print("\n".join(paths))


if __name__ == "__main__":
    main()
