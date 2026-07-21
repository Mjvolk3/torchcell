# experiments/010-kuzmin-tmi/scripts/construction_validation_doubles.py
# [[experiments.010-kuzmin-tmi.scripts.construction_validation_doubles]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/construction_validation_doubles
"""Doubles chosen for BOTH triple reconstruction AND assay validation.

The pure triple-coverage set-cover
([[experiments.010-kuzmin-tmi.scripts.optimized_doubles_setcover_constructed_10]])
minimizes doubles, but its 8 doubles are all near-neutral (Costanzo DMF span 0.15,
zero significant interactions) — so an echo-plating assay has almost no dynamic
range to validate against. This adds a validation tier: doubles selected for DMF
dynamic range and real digenic-interaction signal (both signs), so the wet-lab has
variance to check the assay on while still reconstructing the triples.

Two tiers, unioned:
  - coverage   : the 8 greedy set-cover doubles (reconstruct all 31 within-10 top-k triples)
  - validation : doubles with a SIGNIFICANT Costanzo interaction (p<0.05 & |eps|>0.08),
                 plus the lowest- and highest-DMF doubles (fitness dynamic range)

Source: results/inference_3/doubles_table_panel12_k200_queried.csv (within-10),
        results/inference_3/top_k_constructible_panel12_k200.csv (triple coverage).
Output: results/construction_validation_doubles.csv
        notes/assets/images/010-kuzmin-tmi/construction_validation_doubles.{png,svg}

Run from repo root:
  ~/miniconda3/envs/torchcell/bin/python \
    experiments/010-kuzmin-tmi/scripts/construction_validation_doubles.py
"""
import os
import os.path as osp
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.ticker import MultipleLocator

from torchcell.datasets.scerevisiae.costanzo2016 import N_SAMPLES_DOUBLE_MUTANT
from torchcell.utils import PLOT_PALETTE, PANEL_WIDTHS_MM, mm_to_in, savefig_true_size_svg

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi", "results")
OUT_DIR = osp.join(ASSET_IMAGES_DIR, "010-kuzmin-tmi")
INF3 = osp.join(RESULTS_DIR, "inference_3")

TEN = {"YBR203W", "YDR057W", "YER079W", "YGL087C", "YJR060W",
       "YKL033W-A", "YLL012W", "YLR312C-B", "YPL046C", "YPL081W"}

# Costanzo significant-interaction thresholds (their SI: |eps|>=0.08, P<0.05).
EPS_THRESH, P_THRESH = 0.08, 0.05
COLOR_COV = PLOT_PALETTE[4]   # blue #6C8EBF — coverage doubles (triple reconstruction)
COLOR_VAL = PLOT_PALETTE[1]   # red  #B85450 — validation doubles (dynamic range / signal)
COLOR_OTHER = "0.82"          # light neutral — de-emphasized (unselected) background

plt.rcParams.update({"font.family": "Arial", "font.size": 6,
                     "svg.fonttype": "none", "axes.linewidth": 0.5})


def within_ten_triples() -> list[frozenset]:
    tri = pd.read_csv(osp.join(INF3, "top_k_constructible_panel12_k200.csv"))
    return [frozenset([r.gene1, r.gene2, r.gene3]) for _, r in tri.iterrows()
            if {r.gene1, r.gene2, r.gene3}.issubset(TEN)]


def setcover(triples: list[frozenset]) -> set[tuple]:
    cand = set().union(*[{tuple(sorted(p)) for p in combinations(sorted(t), 2)}
                         for t in triples])
    cov = {d: {i for i, t in enumerate(triples) if set(d).issubset(t)} for d in cand}
    unc, chosen = set(range(len(triples))), []
    while unc:
        best = max(cand, key=lambda d: len(cov[d] & unc))
        chosen.append(best)
        unc -= cov[best]
    return set(chosen)


def load_doubles() -> pd.DataFrame:
    df = pd.read_csv(osp.join(INF3, "doubles_table_panel12_k200_queried.csv"))
    df = df[df.apply(lambda r: {r.gene1, r.gene2}.issubset(TEN), axis=1)].copy()
    g = df.apply(lambda r: tuple(sorted((r.gene1, r.gene2))), axis=1)
    df["gene1"], df["gene2"] = [x[0] for x in g], [x[1] for x in g]
    df = df.dropna(subset=["DmfCostanzo2016_fitness"]).reset_index(drop=True)
    df["pair"] = list(zip(df.gene1, df.gene2))
    df["eps"] = df["DmiCostanzo2016_gene_interaction"]
    df["p"] = df["DmiCostanzo2016_gene_interaction_p_value"]
    df["significant"] = (df.p < P_THRESH) & (df.eps.abs() > EPS_THRESH)
    df["se"] = df["DmfCostanzo2016_std"] / np.sqrt(N_SAMPLES_DOUBLE_MUTANT)
    return df


def annotate_tiers(df: pd.DataFrame, triples: list[frozenset]) -> pd.DataFrame:
    """Tag EVERY measured within-10 double as coverage / validation / other."""
    coverage = setcover(triples)
    # validation = significant interactions + DMF extremes not already covered
    sig = set(df.loc[df.significant, "pair"])
    lo = {df.loc[df.DmfCostanzo2016_fitness.idxmin(), "pair"]}
    hi = {df.loc[df.DmfCostanzo2016_fitness.idxmax(), "pair"]}
    validation = (sig | lo | hi) - coverage

    def tier_of(p: tuple) -> str:
        if p in coverage:
            return "coverage"
        if p in validation:
            return "validation"
        return "other"

    df = df.copy()
    df["tier"] = df.pair.apply(tier_of)
    df["n_triples_enabled"] = df.pair.apply(
        lambda pr: sum(set(pr).issubset(t) for t in triples)
    )
    return df


def plot(sel: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_WIDTHS_MM["wide"]), mm_to_in(85)),
                           layout="constrained")
    for tier, color in [("coverage", COLOR_COV), ("validation", COLOR_VAL)]:
        s = sel[sel.tier == tier]
        ax.errorbar(s.DmfCostanzo2016_fitness, s.eps, xerr=s.DmfCostanzo2016_std,
                    fmt="o", ms=4, lw=0, elinewidth=0.6, capsize=1.5, color=color,
                    label=f"{tier} (n={len(s)})", zorder=3)
    # ring the significant-interaction doubles (tight so it does not cover error bars)
    sig = sel[sel.significant]
    ax.scatter(sig.DmfCostanzo2016_fitness, sig.eps, s=40, facecolors="none",
               edgecolors="black", linewidths=0.6, zorder=4,
               label=f"significant ε (n={len(sig)})")
    # label each validation double; place text INTO the plot (away from the near edge)
    xmid = sel.DmfCostanzo2016_fitness.median()
    for _, r in sel[sel.tier == "validation"].iterrows():
        right = r.DmfCostanzo2016_fitness > xmid
        top = r.eps > 0.05
        dx, ha = (-5, "right") if right else (5, "left")
        dy, va = (-6, "top") if top else (5, "bottom")
        ax.annotate(f"{r.gene1}+{r.gene2}", (r.DmfCostanzo2016_fitness, r.eps),
                    textcoords="offset points", xytext=(dx, dy), ha=ha, va=va,
                    fontsize=5, color="0.2")
    ax.axhline(0.0, color="0.4", ls=":", lw=0.8)
    for thr in (EPS_THRESH, -EPS_THRESH):
        ax.axhline(thr, color="0.75", ls="--", lw=0.5)
    ax.margins(x=0.08, y=0.10)  # room so labels/rings do not clip the frame
    # tenth gridlines on the DMF (0-1 metric) axis: line every 0.1, label every 0.2
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which="minor", length=0)
    ax.grid(which="both", axis="x", lw=0.3, color="0.9", zorder=0)
    ax.set_xlabel("Double-mutant fitness")
    ax.set_ylabel("Digenic interaction ε")
    ax.set_title("Doubles for construction + assay validation (Costanzo2016)", fontsize=7)
    ax.legend(frameon=True, fontsize=5.5, loc="upper left")
    for sp in ("top", "right", "left", "bottom"):
        ax.spines[sp].set_visible(True)
        ax.spines[sp].set_linewidth(0.5)
    # constrained_layout (set at subplots) fits all labels inside the fixed-size
    # figure, so neither the PNG nor the true-size SVG clips the axis titles.
    fig.savefig(osp.join(OUT_DIR, "construction_validation_doubles.png"), dpi=200)
    savefig_true_size_svg(fig, osp.join(OUT_DIR, "construction_validation_doubles.svg"))
    plt.close(fig)


TIER_COLOR = {"coverage": COLOR_COV, "validation": COLOR_VAL, "other": COLOR_OTHER}


def plot_forest(df: pd.DataFrame) -> None:
    """All 45 within-10 doubles ranked by DMF, colored by tier."""
    d = df.sort_values("DmfCostanzo2016_fitness").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_WIDTHS_MM["half_plus"]), mm_to_in(150)))
    for i, r in d.iterrows():
        c = TIER_COLOR[r.tier]
        big = r.tier != "other"
        ax.errorbar(r.DmfCostanzo2016_fitness, i, xerr=r.DmfCostanzo2016_std, fmt="o",
                    ms=3.4 if big else 2.2, lw=0, elinewidth=0.6, capsize=1.3,
                    color=c, zorder=4 if big else 2)
        if r.significant:  # ring significant interactions
            ax.scatter(r.DmfCostanzo2016_fitness, i, s=42, facecolors="none",
                       edgecolors="black", linewidths=0.6, zorder=5)
    ax.axvline(1.0, color="0.4", ls=":", lw=0.8, zorder=1)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([f"{r.gene1}+{r.gene2}" for _, r in d.iterrows()], fontsize=4.5)
    ax.set_xlabel("Double-mutant fitness (Costanzo2016)")
    ax.set_title(f"All {len(d)} measured within-10 doubles by DMF\n"
                 "color = tier; ring = significant ε")
    handles = [plt.Line2D([], [], marker="o", lw=0, markersize=4, color=TIER_COLOR[t],
                          label=f"{t} (n={int((d.tier == t).sum())})")
               for t in ("coverage", "validation", "other")]
    ax.legend(handles=handles, frameon=True, fontsize=5.5, loc="lower right",
              labelspacing=1.0, handletextpad=0.5, borderpad=0.6)
    for sp in ("top", "right", "left", "bottom"):
        ax.spines[sp].set_visible(True)
        ax.spines[sp].set_linewidth(0.5)
    fig.tight_layout()
    fig.savefig(osp.join(OUT_DIR, "construction_validation_doubles_forest.png"), dpi=200)
    savefig_true_size_svg(fig, osp.join(OUT_DIR, "construction_validation_doubles_forest.svg"))
    plt.close(fig)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    triples = within_ten_triples()
    df = annotate_tiers(load_doubles(), triples)
    sel = df[df.tier != "other"].sort_values(
        ["tier", "DmfCostanzo2016_fitness"]).reset_index(drop=True)

    cols = ["gene1", "gene2", "tier", "n_triples_enabled",
            "DmfCostanzo2016_fitness", "DmfCostanzo2016_std", "se",
            "eps", "p", "significant", "DmfCostanzo2016_strain_id"]
    out = osp.join(RESULTS_DIR, "construction_validation_doubles.csv")
    sel[cols].to_csv(out, index=False)
    plot(sel)
    plot_forest(df)

    cov, val = sel[sel.tier == "coverage"], sel[sel.tier == "validation"]
    covered = sum(any(set(pr).issubset(t) for pr in sel.pair) for t in triples)
    print(f"total doubles: {len(sel)}  (coverage {len(cov)} + validation {len(val)})")
    print(f"triples reconstructed: {covered}/{len(triples)} within-10 top-k")
    print(f"DMF range: {sel.DmfCostanzo2016_fitness.min():.3f}-{sel.DmfCostanzo2016_fitness.max():.3f}"
          f"  (span {sel.DmfCostanzo2016_fitness.max()-sel.DmfCostanzo2016_fitness.min():.3f})")
    print(f"eps range: {sel.eps.min():+.3f} to {sel.eps.max():+.3f}; significant: {int(sel.significant.sum())}")
    print(f"\nSaved: {out}")
    print(sel[["gene1", "gene2", "tier", "DmfCostanzo2016_fitness", "eps", "p", "significant",
               "n_triples_enabled"]].to_string(index=False))


if __name__ == "__main__":
    main()
