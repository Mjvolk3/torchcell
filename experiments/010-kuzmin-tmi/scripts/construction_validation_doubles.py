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
import shutil
import subprocess
import tempfile
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
# Only the first three palette colors (orange, red, purple) for data series.
COLOR_COV = PLOT_PALETTE[2]   # purple #9673A6 — coverage / Costanzo reference
COLOR_VAL = PLOT_PALETTE[1]   # red    #B85450 — validation doubles (dynamic range / signal)
COLOR_OTHER = "0.82"          # light neutral — de-emphasized (unselected) background only

# Set the label/tick sizes ABSOLUTELY (not the relative "medium", which resolves to
# the default 10 pt and then gets ~1.4x-scaled by the true-size SVG export -> huge
# axis labels). Nature Biotech in-figure text is 5-7 pt.
plt.rcParams.update({"font.family": "Arial", "font.size": 6,
                     "axes.labelsize": 6, "axes.titlesize": 7,
                     "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "legend.fontsize": 5.5,
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


LOW_DMF = 0.75  # "low-DMF" band for choosing a tight low-fitness anchor


def load_doubles(triples: list[frozenset]) -> pd.DataFrame:
    """ALL 45 within-10 pairs (C(10,2)) -- kept even when Costanzo is unmeasured,
    with Kuzmin DMF alongside so cross-dataset disagreement is visible."""
    df = pd.read_csv(osp.join(INF3, "doubles_table_panel12_k200_queried.csv"))
    df = df[df.apply(lambda r: {r.gene1, r.gene2}.issubset(TEN), axis=1)].copy()
    g = df.apply(lambda r: tuple(sorted((r.gene1, r.gene2))), axis=1)
    df["gene1"], df["gene2"] = [x[0] for x in g], [x[1] for x in g]
    df = df.reset_index(drop=True)
    df["pair"] = list(zip(df.gene1, df.gene2))
    df["eps"] = df["DmiCostanzo2016_gene_interaction"]
    df["p"] = df["DmiCostanzo2016_gene_interaction_p_value"]
    df["significant"] = (df.p < P_THRESH) & (df.eps.abs() > EPS_THRESH)
    df["se"] = df["DmfCostanzo2016_std"] / np.sqrt(N_SAMPLES_DOUBLE_MUTANT)
    df["measured"] = df["DmfCostanzo2016_fitness"].notna()
    df["n_triples_enabled"] = df.pair.apply(
        lambda pr: sum(set(pr).issubset(t) for t in triples)
    )
    return df


def annotate_tiers(df: pd.DataFrame, triples: list[frozenset]) -> pd.DataFrame:
    """Tag every within-10 double: coverage / validation / novel / other.

    validation = the 3 significant interactions + the highest-DMF double + a LOW-DMF
    anchor chosen for QUALITY not just extremeness: among low-DMF (<0.75) measured
    doubles, prefer one that also reconstructs triples, breaking ties by lowest SD
    (tightest reference). This avoids the noisy DMF-minimum (YGL087C+YJR060W, SD 0.172)
    in favour of YER079W+YJR060W (covers 3 triples, SD 0.018).
    novel = pairs unmeasured by Costanzo AND Kuzmin -- construction candidates.
    """
    coverage = setcover(triples)
    meas = df[df.measured]
    sig = set(meas.loc[meas.significant, "pair"])
    hi = {meas.loc[meas.DmfCostanzo2016_fitness.idxmax(), "pair"]}

    low = meas[meas.DmfCostanzo2016_fitness < LOW_DMF]
    covering = low[low.n_triples_enabled > 0]
    pool = covering if len(covering) else low
    low_anchor = {pool.loc[pool.DmfCostanzo2016_std.idxmin(), "pair"]}

    validation = (sig | hi | low_anchor) - coverage
    novel = set(df.loc[~df.measured, "pair"])  # unmeasured everywhere

    def tier_of(p: tuple, measured: bool) -> str:
        if p in coverage:
            return "coverage"
        if p in validation:
            return "validation"
        if p in novel:
            return "novel"
        return "other"

    df = df.copy()
    df["tier"] = [tier_of(p, m) for p, m in zip(df.pair, df.measured)]
    return df


def plot_table(df: pd.DataFrame) -> None:
    """Standalone LaTeX-booktabs image of the table (13 selected + 1 novel), all black.

    Renders a real booktabs table via pdflatex (crisp LaTeX typography), crops to the
    table bbox with ghostscript, and emits PDF + PNG + SVG. Rows sorted by Costanzo
    DMF (novel, unmeasured, last); check mark for significant interactions.
    """
    meas = df[df.tier.isin(["coverage", "validation"])].sort_values(
        "DmfCostanzo2016_fitness")
    nov = df[df.tier == "novel"]
    rows = pd.concat([meas, nov]).reset_index(drop=True)

    def dmf(v, s):
        return "---" if pd.isna(v) else rf"${v:.3f} \pm {s:.3f}$"

    body = []
    for i, r in rows.iterrows():
        body.append(" & ".join([
            str(i + 1),
            f"{r.gene1}+{r.gene2}",
            r.tier,
            dmf(r.DmfCostanzo2016_fitness, r.DmfCostanzo2016_std),
            "---" if pd.isna(r.eps) else rf"${r.eps:+.3f}$",
            "---" if pd.isna(r.p) else f"{r.p:.3f}",
            r"$\checkmark$" if r.significant else "",
            str(int(r.n_triples_enabled)),
        ]) + r" \\")

    tabular = (r"\begin{tabular}{r l l c c c c c}" + "\n\\toprule\n"
               r"\# & Double & Tier & DMF $\pm$ SD & $\varepsilon$ & $p$ & sig & triples \\"
               + "\n\\midrule\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}")
    _render_latex_table(tabular, "construction_validation_doubles_table")


# common names for the panel-10 genes (SGD R64; "" = no standard name)
COMMON = {"YBR203W": "COS111", "YDR057W": "YOS9", "YER079W": "", "YGL087C": "MMS2",
          "YJR060W": "CBF1", "YKL033W-A": "", "YLL012W": "YEH1",
          "YLR312C-B": "(SPH1 locus)", "YPL046C": "ELC1", "YPL081W": "RPS9A"}


def plot_genes(df: pd.DataFrame) -> None:
    """LaTeX-booktabs image of the unique single-KO genes to inoculate for the 14
    doubles, with common name and how many of the 14 doubles each participates in."""
    sel = df[df.tier.isin(["coverage", "validation", "novel"])]
    genes = sorted(set(sel.gene1) | set(sel.gene2))
    body = []
    for i, g in enumerate(genes, 1):
        n = int(((sel.gene1 == g) | (sel.gene2 == g)).sum())
        common = COMMON.get(g, "") or "---"
        body.append(f"{i} & {g} & {common} & {n} " + r"\\")
    tabular = (r"\begin{tabular}{r l l c}" + "\n\\toprule\n"
               r"\# & Systematic & Common & \# doubles \\"
               + "\n\\midrule\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}")
    _render_latex_table(tabular, "construction_validation_genes_to_inoculate")


def _render_latex_table(tabular: str, base: str) -> None:
    """Compile a booktabs tabular via pdflatex, crop to its bbox, emit PDF/PNG/SVG."""
    tex = (r"""\documentclass[10pt]{article}
\usepackage[margin=6pt,paperwidth=24cm,paperheight=16cm]{geometry}
\usepackage{booktabs}
\usepackage{amssymb}
\pagestyle{empty}
\begin{document}
""" + tabular + "\n\\end{document}\n")
    with tempfile.TemporaryDirectory() as d:
        with open(osp.join(d, "t.tex"), "w") as fh:
            fh.write(tex)
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "t.tex"],
                       cwd=d, check=True, capture_output=True)
        bbox = subprocess.run(["gs", "-q", "-dBATCH", "-dNOPAUSE", "-sDEVICE=bbox",
                               osp.join(d, "t.pdf")], capture_output=True, text=True)
        line = [x for x in bbox.stderr.splitlines() if "HiResBoundingBox" in x][0]
        x0, y0, x1, y1 = (float(v) for v in line.split(":")[1].split())
        pad = 4
        w, h = (x1 - x0) + 2 * pad, (y1 - y0) + 2 * pad
        # resize the page to the bbox so the crop is honored by pdftoppm/pdftocairo
        subprocess.run(["gs", "-o", osp.join(d, "crop.pdf"), "-sDEVICE=pdfwrite",
                        f"-dDEVICEWIDTHPOINTS={w:.2f}", f"-dDEVICEHEIGHTPOINTS={h:.2f}",
                        "-dFIXEDMEDIA",
                        "-c", f"<</PageOffset [{pad - x0:.2f} {pad - y0:.2f}]>> setpagedevice",
                        "-f", osp.join(d, "t.pdf")], check=True, capture_output=True)
        shutil.copy(osp.join(d, "crop.pdf"), osp.join(OUT_DIR, f"{base}.pdf"))
        subprocess.run(["pdftoppm", "-png", "-r", "300", osp.join(d, "crop.pdf"),
                        osp.join(d, "png")], check=True, capture_output=True)
        shutil.copy(osp.join(d, "png-1.png"), osp.join(OUT_DIR, f"{base}.png"))
        subprocess.run(["pdftocairo", "-svg", osp.join(d, "crop.pdf"),
                        osp.join(OUT_DIR, f"{base}.svg")], check=True, capture_output=True)


def plot_cross_dataset(df: pd.DataFrame) -> None:
    """Costanzo vs Kuzmin DMF for the within-10 doubles measured in BOTH -- a
    dumbbell per double (Costanzo o--o Kuzmin) so the disagreement is visible."""
    d = df.copy()
    d["kuzmin"] = d["DmfKuzmin2018_fitness"].fillna(d["DmfKuzmin2020_fitness"])
    d["ksrc"] = np.where(d["DmfKuzmin2018_fitness"].notna(), "K2018",
                         np.where(d["DmfKuzmin2020_fitness"].notna(), "K2020", ""))
    d = d[d.measured & d.kuzmin.notna()].copy()
    d["gap"] = (d.DmfCostanzo2016_fitness - d.kuzmin).abs()
    d = d.sort_values("gap").reset_index(drop=True)

    kcolor = {"K2018": PLOT_PALETTE[0], "K2020": PLOT_PALETTE[1]}  # orange / red
    #        (Costanzo = COLOR_COV purple; three-color scheme = first three palette colors)
    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_WIDTHS_MM["wide"]), mm_to_in(65)),
                           layout="constrained")
    for i, r in d.iterrows():
        ax.errorbar(r.DmfCostanzo2016_fitness, i, xerr=r.DmfCostanzo2016_std, fmt="o",
                    ms=4, elinewidth=0.6, capsize=1.5, color=COLOR_COV, zorder=3)
        ax.scatter(r.kuzmin, i, s=22, color=kcolor[r.ksrc], zorder=3)  # color denotes K2018/K2020
    ax.margins(x=0.04, y=0.10)
    ax.axvline(1.0, color="0.4", ls=":", lw=0.8, zorder=0)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([f"{r.gene1}+{r.gene2}" for _, r in d.iterrows()], fontsize=5.5)
    ax.set_xlabel("Double-mutant fitness")
    ax.set_title("Cross-dataset DMF: Costanzo vs Kuzmin (within-10 doubles in both)",
                 fontsize=7)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which="minor", length=0)
    ax.grid(which="both", axis="x", lw=0.3, color="0.9", zorder=0)
    ax.legend(handles=[plt.Line2D([], [], marker="o", lw=0, ms=4, color=COLOR_COV,
                                  label="Costanzo2016 (± SD)"),
                       plt.Line2D([], [], marker="o", lw=0, ms=4, color=kcolor["K2018"],
                                  label="Kuzmin2018"),
                       plt.Line2D([], [], marker="o", lw=0, ms=4, color=kcolor["K2020"],
                                  label="Kuzmin2020")],
              frameon=True, fontsize=5.5, loc="lower left")
    for sp in ("top", "right", "left", "bottom"):
        ax.spines[sp].set_visible(True)
        ax.spines[sp].set_linewidth(0.5)
    fig.savefig(osp.join(OUT_DIR, "construction_validation_cross_dataset.png"), dpi=200)
    savefig_true_size_svg(fig, osp.join(OUT_DIR, "construction_validation_cross_dataset.svg"))
    plt.close(fig)


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


TIER_COLOR = {"coverage": COLOR_COV, "validation": COLOR_VAL,
              "novel": PLOT_PALETTE[0], "other": COLOR_OTHER}


def plot_forest(df: pd.DataFrame) -> None:
    """All measured within-10 doubles ranked by DMF, colored by tier."""
    d = df[df.measured].sort_values("DmfCostanzo2016_fitness").reset_index(drop=True)
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
    df = annotate_tiers(load_doubles(triples), triples)

    # ---- full 45-row SI table: EVERY within-10 pair, tier blank if unselected,
    #      Costanzo + Kuzmin DMF side-by-side so disagreement is auditable ----
    si = df.copy()
    si["tier_si"] = si.tier.map(lambda t: "" if t == "other" else t)
    si_cols = ["gene1", "gene2", "tier_si", "n_triples_enabled", "significant",
               "DmfCostanzo2016_fitness", "DmfCostanzo2016_std", "se",
               "eps", "p",
               "DmfKuzmin2018_fitness", "DmfKuzmin2018_std",
               "DmfKuzmin2020_fitness", "DmfKuzmin2020_std",
               "DmfCostanzo2016_strain_id"]
    si = si[si_cols].rename(columns={"tier_si": "tier"}).sort_values(
        "DmfCostanzo2016_fitness", na_position="last").reset_index(drop=True)
    out = osp.join(RESULTS_DIR, "construction_validation_doubles.csv")
    si.to_csv(out, index=False)

    sel = df[df.tier.isin(["coverage", "validation"])]
    plot(sel)
    plot_table(df)
    plot_genes(df)
    plot_cross_dataset(df)
    plot_forest(df)

    covered = sum(any(set(pr).issubset(t) for pr in sel.pair) for t in triples)
    print(f"selected: {len(sel)} (coverage {int((df.tier=='coverage').sum())} + "
          f"validation {int((df.tier=='validation').sum())}); "
          f"novel/unmeasured {int((df.tier=='novel').sum())}")
    print(f"triples reconstructed: {covered}/{len(triples)} within-10 top-k")
    print(f"DMF range: {sel.DmfCostanzo2016_fitness.min():.3f}-{sel.DmfCostanzo2016_fitness.max():.3f}")
    print(f"novel pairs (unmeasured Costanzo & Kuzmin): "
          f"{[f'{a}+{b}' for a,b in df.loc[df.tier=='novel','pair']]}")
    print(f"\nSaved SI table ({len(si)} rows): {out}")
    print(df[df.tier.isin(['coverage','validation','novel'])][
        ['gene1','gene2','tier','DmfCostanzo2016_fitness','eps','p','significant',
         'n_triples_enabled']].sort_values(['tier','DmfCostanzo2016_fitness']).to_string(index=False))


if __name__ == "__main__":
    main()
