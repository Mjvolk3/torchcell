# experiments/007-kuzmin-tm/scripts/fba_baseline_si.py
# [[experiments.007-kuzmin-tm.scripts.fba_baseline_si]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/007-kuzmin-tm/scripts/fba_baseline_si
"""Every number and panel of the Yeast9 FBA Supplementary Note (note:fba), from the frozen run.

The Yeast9 flux-balance baseline for trigenic interactions (Fig. 2d, "Yeast9 FBA") was run
once, on 2025-09-15, by ``targeted_fba_growth_fast.py`` (this folder) and matched to the
Kuzmin 2018 labels by ``match_fba_to_experiments.py``; the slurm driver
``gh_cobra-fba-growth.slurm`` moved that run's outputs to
``results/cobra-fba-growth_backup_20250923_134447/`` on 2025-09-23 before a re-run whose
outputs were never committed. The backup directory is the only frozen record of the
baseline and is the sole input here.

This script (1) re-reads those files and recomputes the Pearson correlations and the
distributional facts the note states, (2) loads the same yeast-GEM 9.0.2 SBML through
``torchcell.metabolism.yeast_GEM.YeastGEM`` to record the medium the run used (the model's
default ``model.medium``, which the run never modified) and to verify that the wild-type
growth rate matches the frozen ``wt_growth.csv``, (3) writes the frozen inputs of the note
to ``results/fba_baseline_si/`` (``stats.json``, ``yeast9_default_medium.csv``,
``triples_matched.parquet``, ``growth_bands.csv``, ``coverage.csv``), (4) writes
``paper/nature-biotech/sections/tab-fba-medium.tex``, and (5) draws the data panels of
``FigS-yeast9-fba`` as true-size SVGs (plus PNG fallbacks) into
``ASSET_IMAGES_DIR/007-kuzmin-tm/``. ``fba_baseline_compose_figure.py`` (this folder) then
assembles the draw.io figure from the panels and ``stats.json``.

Run from the repo root:
    python experiments/007-kuzmin-tm/scripts/fba_baseline_si.py
"""

import json
import os
import os.path as osp

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402
from scipy import stats  # noqa: E402

from torchcell.metabolism.yeast_GEM import YeastGEM  # noqa: E402
from torchcell.utils import (  # noqa: E402
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    PLOT_PALETTE_FILL,
    mm_to_in,
    savefig_true_size_svg,
)

# Set AFTER the torchcell imports: the repo mplstyle is applied on import by some
# torchcell modules and would override these.
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6,
        "axes.titlesize": 6,
        "axes.labelsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "lines.linewidth": 0.8,
        "svg.fonttype": "none",
        "savefig.bbox": None,
    }
)

load_dotenv()
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
assert ASSET_IMAGES_DIR is not None, "ASSET_IMAGES_DIR must be set in the environment"

SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
EXP_DIR = osp.dirname(SCRIPT_DIR)
REPO_ROOT = osp.dirname(osp.dirname(EXP_DIR))
FROZEN = osp.join(EXP_DIR, "results", "cobra-fba-growth_backup_20250923_134447")
OUT = osp.join(EXP_DIR, "results", "fba_baseline_si")
IMG_DIR = osp.join(ASSET_IMAGES_DIR, "007-kuzmin-tm")
TEX_DIR = osp.join(REPO_ROOT, "paper", "nature-biotech", "sections")

ORANGE, RED, PURPLE, YELLOW, BLUE, GRAY = PLOT_PALETTE[:6]
ORANGE_F, RED_F, PURPLE_F, YELLOW_F = PLOT_PALETTE_FILL[:4]

# Thresholds. ``LETHAL`` is the pipeline's own "lethal" cut (targeted_fba_growth_fast.py
# reports fitness < 0.01 as lethal). ``WT_LIKE`` and ``TAU_ZERO`` separate LP round-off
# from a modeled effect: fitness values within 1e-3 of 1 and |tau| below 1e-3 differ from
# the wild type by less than the solver's optimum-to-optimum jitter observed in the files
# (of order 1e-7) times the number of terms in tau, and no biological band in the run
# sits there (the smallest non-wild-type fitness band is 0.994).
LETHAL = 0.01
WT_LIKE = 1e-3
TAU_ZERO = 1e-3


def box(ax):
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)
        s.set_edgecolor("black")


def save(fig, name):
    os.makedirs(IMG_DIR, exist_ok=True)
    savefig_true_size_svg(fig, osp.join(IMG_DIR, f"{name}.svg"))
    fig.savefig(osp.join(IMG_DIR, f"{name}.png"), dpi=300)
    plt.close(fig)
    print(f"saved {osp.join(IMG_DIR, name)}.{{svg,png}}")


# ----------------------------------------------------------------------------- inputs
def load_frozen():
    meta = json.load(open(osp.join(FROZEN, "fba_metadata.json")))
    wt = pd.read_csv(osp.join(FROZEN, "wt_growth.csv"))
    singles = pd.read_parquet(osp.join(FROZEN, "singles_deletions.parquet"))
    doubles = pd.read_parquet(osp.join(FROZEN, "doubles_deletions.parquet"))
    triples = pd.read_parquet(osp.join(FROZEN, "triples_deletions.parquet"))
    trigenic = pd.read_parquet(osp.join(FROZEN, "trigenic_interactions.parquet"))
    matched = pd.read_parquet(osp.join(FROZEN, "matched_fba_experimental_fixed.parquet"))
    perts = json.load(open(osp.join(FROZEN, "unique_perturbations.json")))
    return meta, wt, singles, doubles, triples, trigenic, matched, perts


def load_model():
    gem = YeastGEM()
    model = gem.model
    return gem, model


# ----------------------------------------------------------------------------- statistics
def medium_table(model) -> pd.DataFrame:
    rows = []
    for rid, bound in sorted(model.medium.items()):
        rxn = model.reactions.get_by_id(rid)
        met = list(rxn.metabolites)[0]
        rows.append(
            {
                "reaction_id": rid,
                "reaction_name": rxn.name,
                "metabolite": met.name,
                "uptake_bound_mmol_gDW_h": bound,
                "lower_bound": rxn.lower_bound,
                "upper_bound": rxn.upper_bound,
            }
        )
    return pd.DataFrame(rows)


def gene_coverage(model, perts, trigenic, matched):
    model_genes = {g.id for g in model.genes}
    n_in = {
        "singles": int(sum(g in model_genes for g in perts["singles"])),
        "doubles": int(sum(all(g in model_genes for g in d) for d in perts["doubles"])),
    }
    tri = trigenic.copy()
    tri["n_in_model"] = [
        sum(g in model_genes for g in (a, b, c))
        for a, b, c in zip(tri.gene1, tri.gene2, tri.gene3)
    ]
    tri["tau_nonzero"] = tri.tau.abs() > TAU_ZERO
    cov = (
        tri.groupby("n_in_model")
        .agg(n_triples=("tau", "size"), n_tau_nonzero=("tau_nonzero", "sum"))
        .reset_index()
    )
    cov["frac_triples"] = cov.n_triples / len(tri)
    cov["frac_tau_nonzero_within"] = cov.n_tau_nonzero / cov.n_triples
    return model_genes, n_in, tri, cov


def consistency(model_genes, singles, doubles, triples) -> dict:
    """Internal checks the run's own files allow.

    A gene outside the model changes no reaction, so a double or triple with at most one
    gene in the model must reproduce that gene's single-deletion fitness (1 when none is in
    the model), and adding a deletion can never raise the FBA optimum, so a triple's
    fitness cannot exceed the smallest fitness of its three doubles. Counts of violations
    are solver failures recorded with status ``optimal``; their cause was not determined.
    """
    sf = singles.set_index("gene").fitness

    def expected(genes):
        ins = [g for g in genes if g in model_genes]
        if len(ins) == 0:
            return 1.0
        if len(ins) == 1:
            return float(sf[ins[0]])
        return np.nan

    d_exp = np.array([expected((a, b)) for a, b in zip(doubles.gene1, doubles.gene2)])
    d_chk = ~np.isnan(d_exp)
    d_bad = d_chk & (np.abs(doubles.fitness.to_numpy() - d_exp) > TAU_ZERO)
    t_exp = np.array([expected((a, b, c)) for a, b, c in zip(triples.gene1, triples.gene2, triples.gene3)])
    t_chk = ~np.isnan(t_exp)
    t_bad = t_chk & (np.abs(triples.fitness.to_numpy() - t_exp) > TAU_ZERO)
    dk = {}
    for a, b, f in zip(doubles.gene1, doubles.gene2, doubles.fitness):
        dk[(a, b)] = f
        dk[(b, a)] = f
    mono = 0
    for a, b, c, f in zip(triples.gene1, triples.gene2, triples.gene3, triples.fitness):
        pairs = [dk[k] for k in ((a, b), (a, c), (b, c)) if k in dk]
        if pairs and f > min(pairs) + TAU_ZERO:
            mono += 1
    mono_d = int(sum(f > min(sf.get(a, 1.0), sf.get(b, 1.0)) + TAU_ZERO for a, b, f in zip(doubles.gene1, doubles.gene2, doubles.fitness)))
    bad_genes = pd.Series(
        [g for a, b, bad in zip(doubles.gene1, doubles.gene2, d_bad) if bad for g in (a, b) if g in model_genes]
    ).value_counts()
    return {
        "doubles_checkable": int(d_chk.sum()),
        "doubles_inconsistent_with_single": int(d_bad.sum()),
        "triples_checkable": int(t_chk.sum()),
        "triples_inconsistent_with_single": int(t_bad.sum()),
        "triples_above_min_double": int(mono),
        "doubles_above_min_single": mono_d,
        "inconsistent_doubles_model_gene_counts": {str(k): int(v) for k, v in bad_genes.head(5).items()},
    }


def growth_bands(singles, doubles, triples) -> pd.DataFrame:
    rows = []
    for order, df in [("single", singles), ("double", doubles), ("triple", triples)]:
        f = df.fitness.to_numpy()
        lethal = f < LETHAL
        wt_like = np.abs(f - 1.0) < WT_LIKE
        inter = ~lethal & ~wt_like
        rows.append(
            {
                "order": order,
                "n": len(f),
                "n_wt_like": int(wt_like.sum()),
                "n_intermediate": int(inter.sum()),
                "n_lethal": int(lethal.sum()),
                "frac_wt_like": float(wt_like.mean()),
                "frac_intermediate": float(inter.mean()),
                "frac_lethal": float(lethal.mean()),
                "n_status_optimal": int((df.status == "optimal").sum()),
            }
        )
    return pd.DataFrame(rows)


def correlations(matched) -> dict:
    out = {}
    for pt, key in [("gene_interaction", "tau"), ("fitness", "fitness")]:
        sub = matched[matched.phenotype_type == pt].dropna(subset=["experimental", "fba_predicted"])
        r, p = stats.pearsonr(sub.fba_predicted, sub.experimental)
        rho, _ = stats.spearmanr(sub.fba_predicted, sub.experimental)
        out[key] = {
            "n": int(len(sub)),
            "n_total_rows": int((matched.phenotype_type == pt).sum()),
            "pearson_r": float(r),
            "pearson_p": float(p),
            "spearman_rho": float(rho),
            "pred_mean": float(sub.fba_predicted.mean()),
            "pred_sd": float(sub.fba_predicted.std(ddof=1)),
            "exp_mean": float(sub.experimental.mean()),
            "exp_sd": float(sub.experimental.std(ddof=1)),
        }
    fi = matched[matched.phenotype_type == "fitness"]
    fvc = pd.Series(np.round(fi.fba_predicted.to_numpy(), 3)).value_counts()
    out["fitness"]["value_counts_round3"] = {str(k): int(v) for k, v in fvc.head(8).items()}
    out["fitness"]["n_distinct_round3"] = int(len(fvc))
    gi = matched[matched.phenotype_type == "gene_interaction"]
    tau = gi.fba_predicted.to_numpy()
    out["tau"]["frac_abs_below_1e-6"] = float((np.abs(tau) < 1e-6).mean())
    out["tau"]["frac_abs_below_1e-3"] = float((np.abs(tau) < TAU_ZERO).mean())
    out["tau"]["n_abs_above_1e-3"] = int((np.abs(tau) > TAU_ZERO).sum())
    nz = pd.Series(np.round(tau[np.abs(tau) > TAU_ZERO], 2)).value_counts()
    out["tau"]["nonzero_value_counts"] = {str(k): int(v) for k, v in nz.head(8).items()}
    sub = gi[np.abs(gi.fba_predicted) > TAU_ZERO]
    r_nz, _ = stats.pearsonr(sub.fba_predicted, sub.experimental)
    out["tau"]["pearson_r_nonzero_only"] = float(r_nz)
    out["tau"]["n_nonzero_only"] = int(len(sub))
    return out


# ----------------------------------------------------------------------------- panels
def panel_tau(matched, corr):
    """Predicted vs measured tau as a hexbin with log counts."""
    gi = matched[matched.phenotype_type == "gene_interaction"]
    w = mm_to_in(PANEL_WIDTHS_MM["third"])
    fig, ax = plt.subplots(figsize=(w, mm_to_in(50)))
    fig.subplots_adjust(left=0.2, right=0.82, bottom=0.17, top=0.95)
    cmap = LinearSegmentedColormap.from_list("amber", ["#FFFFFF", ORANGE_F, ORANGE, PLOT_PALETTE[6]])
    hb = ax.hexbin(
        gi.fba_predicted,
        gi.experimental,
        gridsize=(36, 24),
        extent=(-2.1, 2.1, -1.15, 1.15),
        norm=LogNorm(vmin=1, vmax=max(1, len(gi))),
        cmap=cmap,
        mincnt=1,
        linewidths=0.1,
        edgecolors="none",
    )
    cb = fig.colorbar(hb, ax=ax, pad=0.02, fraction=0.06)
    cb.set_label("Triples", labelpad=1)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(width=0.5, length=2)
    ax.axhline(0, color="black", lw=0.4, ls=":")
    ax.axvline(0, color="black", lw=0.4, ls=":")
    ax.set_xlim(-2.1, 2.1)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel(r"Yeast9 FBA $\tau_{ijk}$")
    ax.set_ylabel(r"Measured $\tau_{ijk}$")
    c = corr["tau"]
    ax.text(
        0.03,
        0.97,
        f"Pearson $r$ = {c['pearson_r']:.4f}\n$n$ = {c['n']:,}\n"
        f"{100 * c['frac_abs_below_1e-3']:.2f}% at $|\\tau|<10^{{-3}}$",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=6,
    )
    box(ax)
    save(fig, "fba_baseline_tau")


def panel_fitness(matched, corr):
    """Predicted vs measured triple-mutant fitness as a hexbin with log counts."""
    fi = matched[matched.phenotype_type == "fitness"].dropna(subset=["experimental", "fba_predicted"])
    w = mm_to_in(PANEL_WIDTHS_MM["third"])
    fig, ax = plt.subplots(figsize=(w, mm_to_in(50)))
    fig.subplots_adjust(left=0.2, right=0.82, bottom=0.17, top=0.95)
    cmap = LinearSegmentedColormap.from_list("lilac", ["#FFFFFF", PURPLE_F, PURPLE, PLOT_PALETTE[8]])
    hb = ax.hexbin(
        fi.fba_predicted,
        fi.experimental,
        gridsize=(30, 24),
        extent=(-0.05, 1.05, -0.05, 1.55),
        norm=LogNorm(vmin=1, vmax=max(1, len(fi))),
        cmap=cmap,
        mincnt=1,
        linewidths=0.1,
        edgecolors="none",
    )
    cb = fig.colorbar(hb, ax=ax, pad=0.02, fraction=0.06)
    cb.set_label("Triples", labelpad=1)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(width=0.5, length=2)
    ax.plot([0, 1.05], [0, 1.05], color="black", lw=0.4, ls=":")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.55)
    ax.set_xlabel(r"Yeast9 FBA $f_{ijk}$")
    ax.set_ylabel(r"Measured $f_{ijk}$")
    c = corr["fitness"]
    ax.text(0.03, 0.97, f"Pearson $r$ = {c['pearson_r']:.4f}\n$n$ = {c['n']:,}", transform=ax.transAxes, va="top", ha="left", fontsize=6)
    box(ax)
    save(fig, "fba_baseline_fitness")


def panel_growth_bands(bands: pd.DataFrame):
    """Fraction of single, double and triple deletions in each FBA growth band."""
    w = mm_to_in(PANEL_WIDTHS_MM["third"])
    fig, ax = plt.subplots(figsize=(w, mm_to_in(50)))
    fig.subplots_adjust(left=0.2, right=0.97, bottom=0.17, top=0.95)
    cats = [("frac_wt_like", "Wild-type growth", GRAY), ("frac_intermediate", "Reduced", ORANGE), ("frac_lethal", "No growth", RED)]
    x = np.arange(len(bands))
    bw = 0.26
    for k, (col, label, color) in enumerate(cats):
        vals = bands[col].to_numpy()
        ax.bar(x + (k - 1) * bw, vals, bw, color=color, edgecolor="black", linewidth=0.4, label=label)
        for xi, v in zip(x + (k - 1) * bw, vals):
            ax.text(xi, v + 0.015, f"{100 * v:.1f}", ha="center", va="bottom", fontsize=5, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{o.capitalize()}\n$n$ = {n:,}" for o, n in zip(bands.order, bands.n)])
    ax.set_ylabel("Fraction of deletions")
    ax.set_ylim(0, 1.5)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0, 1.01, 0.1), minor=True)
    ax.tick_params(axis="y", which="minor", length=0)
    ax.grid(axis="y", which="both", color="0.85", lw=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, handlelength=1.0, handleheight=0.8, borderaxespad=0.2)
    box(ax)
    save(fig, "fba_baseline_growth_bands")


def panel_coverage(cov: pd.DataFrame):
    """Triples by how many of their three genes Yeast9 carries, and how many have a nonzero tau."""
    w = mm_to_in(PANEL_WIDTHS_MM["third"])
    fig, ax = plt.subplots(figsize=(w, mm_to_in(50)))
    fig.subplots_adjust(left=0.2, right=0.97, bottom=0.17, top=0.95)
    x = cov.n_in_model.to_numpy()
    ax.bar(x, cov.frac_triples, 0.6, color=PURPLE, edgecolor="black", linewidth=0.4)
    for xi, f, n, nz in zip(x, cov.frac_triples, cov.n_triples, cov.n_tau_nonzero):
        ax.text(xi, f + 0.02, f"{n:,}\n$|\\tau|>10^{{-3}}$: {nz:,}", ha="center", va="bottom", fontsize=5)
    ax.set_xticks(x)
    ax.set_xlabel("Genes of the triple in Yeast9")
    ax.set_ylabel("Fraction of triples")
    ax.set_ylim(0, 0.9)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(axis="y", which="minor", length=0)
    ax.grid(axis="y", which="both", color="0.85", lw=0.4)
    ax.set_axisbelow(True)
    box(ax)
    save(fig, "fba_baseline_coverage")


# ----------------------------------------------------------------------------- tables
def write_medium_table(med: pd.DataFrame, model_stats: dict, path: str):
    obj_id = model_stats["objective_id"].replace("_", r"\_")
    ngam_id = model_stats["ngam_id"].replace("_", r"\_")
    lines = [
        "%% SOURCE: experiments/007-kuzmin-tm/scripts/fba_baseline_si.py -- AUTO-GENERATED, do not hand-edit; rerun the script.",
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\caption{The medium of the Yeast9 flux-balance baseline: the exchange reactions open for"
        r" uptake in yeast-GEM " + model_stats["version"] + r" as distributed (\texttt{model.medium}),"
        r" which the run used unchanged. Every other exchange reaction is closed to uptake."
        r" Uptake bounds are in mmol\,gDW$^{-1}$\,h$^{-1}$; 1000 is the model's unbounded value."
        r" Glucose is the sole carbon source at 1 mmol\,gDW$^{-1}$\,h$^{-1}$ and ammonium the sole"
        r" nitrogen source; no amino acid, vitamin, or nucleobase is supplied. The objective is the"
        r" growth pseudoreaction \texttt{" + obj_id + r"} and non-growth-associated"
        r" maintenance is fixed at " + f"{model_stats['ngam_lb']:.1f}" + r" mmol ATP\,gDW$^{-1}$\,h$^{-1}$"
        r" (\texttt{" + ngam_id + r"}). Wild-type growth under this medium is"
        r" " + f"{model_stats['wt_growth']:.4f}" + r"\,h$^{-1}$.}",
        r"\label{tab:fba-medium}",
        r"\begin{tabular}{@{}l l r@{}}",
        r"\toprule",
        r"\textbf{Reaction} & \textbf{Metabolite} & \textbf{Uptake bound}\\",
        r"\midrule",
    ]
    for _, r in med.iterrows():
        rid = r.reaction_id.replace("_", r"\_")
        lines.append(f"\\texttt{{{rid}}} & {r.metabolite} & {r.uptake_bound_mmol_gDW_h:g}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"wrote {path}")


# ----------------------------------------------------------------------------- main
def main():
    os.makedirs(OUT, exist_ok=True)
    meta, wt, singles, doubles, triples, trigenic, matched, perts = load_frozen()
    gem, model = load_model()

    # Model, objective, medium, and the wild-type check against the frozen run.
    obj = [r for r in model.reactions if r.objective_coefficient != 0]
    assert len(obj) == 1, obj
    ngam = model.reactions.get_by_id("r_4046")
    sol = model.optimize()
    wt_frozen = float(wt.growth.iloc[0])
    assert sol.status == "optimal"
    assert abs(sol.objective_value - wt_frozen) < 1e-6, (sol.objective_value, wt_frozen)
    med = medium_table(model)
    med.to_csv(osp.join(OUT, "yeast9_default_medium.csv"), index=False)
    model_stats = {
        "version": gem.version,
        "model_id": model.id,
        "sbml_path": osp.relpath(osp.join(gem.model_dir, "model", "yeast-GEM.xml"), os.getenv("DATA_ROOT")),
        "n_reactions": len(model.reactions),
        "n_metabolites": len(model.metabolites),
        "n_genes": len(model.genes),
        "n_exchange_reactions": len(model.exchanges),
        "n_medium_exchanges_open": len(med),
        "glucose_uptake_bound": float(model.medium["r_1714"]),
        "objective_id": obj[0].id,
        "objective_name": obj[0].name,
        "ngam_id": ngam.id,
        "ngam_name": ngam.name,
        "ngam_lb": float(ngam.lower_bound),
        "solver": model.solver.interface.__name__,
        "wt_growth": wt_frozen,
        "wt_growth_recomputed": float(sol.objective_value),
    }

    # Coverage of the screened genes by the model, growth bands, correlations.
    model_genes, n_in, tri, cov = gene_coverage(model, perts, trigenic, matched)
    cov.to_csv(osp.join(OUT, "coverage.csv"), index=False)
    bands = growth_bands(singles, doubles, triples)
    bands.to_csv(osp.join(OUT, "growth_bands.csv"), index=False)
    corr = correlations(matched)
    consist = consistency(model_genes, singles, doubles, triples)

    # Frozen per-triple table: measured and predicted fitness and tau, model coverage.
    gi = matched[matched.phenotype_type == "gene_interaction"][["genes", "experimental", "fba_predicted"]]
    gi = gi.rename(columns={"experimental": "tau_measured", "fba_predicted": "tau_fba"})
    fi = matched[matched.phenotype_type == "fitness"][["genes", "experimental", "fba_predicted"]]
    fi = fi.rename(columns={"experimental": "fitness_measured", "fba_predicted": "fitness_fba"})
    tri["genes"] = tri.gene1 + "," + tri.gene2 + "," + tri.gene3
    frozen = gi.merge(fi, on="genes", how="left").merge(tri[["genes", "n_in_model"]], on="genes", how="left")
    assert len(frozen) == len(gi) == meta["n_triples"], (len(frozen), len(gi), meta["n_triples"])
    frozen.to_parquet(osp.join(OUT, "triples_matched.parquet"), index=False)

    stats_out = {
        "frozen_dir": osp.relpath(FROZEN, REPO_ROOT),
        "run": {
            "timestamp": meta["timestamp"],
            "runtime_seconds": meta["runtime_seconds"],
            "n_processes": meta["n_processes"],
            "n_singles": meta["n_singles"],
            "n_doubles": meta["n_doubles"],
            "n_triples": meta["n_triples"],
            "solver_timeout_s": 60,
            "n_time_limit_status": int(sum((df.status == "time_limit").sum() for df in (singles, doubles, triples))),
        },
        "model": model_stats,
        "coverage": {
            "n_screened_genes": len(perts["singles"]),
            "n_screened_genes_in_model": n_in["singles"],
            "n_doubles_both_in_model": n_in["doubles"],
            "triples_by_n_in_model": {int(r.n_in_model): int(r.n_triples) for r in cov.itertuples()},
            "tau_nonzero_by_n_in_model": {int(r.n_in_model): int(r.n_tau_nonzero) for r in cov.itertuples()},
        },
        "growth_bands": bands.to_dict(orient="records"),
        "thresholds": {"lethal": LETHAL, "wt_like": WT_LIKE, "tau_zero": TAU_ZERO},
        "correlations": corr,
        "consistency": consist,
        "fig2d_value_in_manuscript": 0.0006,
    }
    json.dump(stats_out, open(osp.join(OUT, "stats.json"), "w"), indent=2)
    print(json.dumps(stats_out, indent=2))

    write_medium_table(med, model_stats, osp.join(TEX_DIR, "tab-fba-medium.tex"))
    panel_tau(matched, corr)
    panel_fitness(matched, corr)
    panel_growth_bands(bands)
    panel_coverage(cov)


if __name__ == "__main__":
    main()
