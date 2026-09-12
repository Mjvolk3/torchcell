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
``triples_matched.parquet``, ``growth_bands.csv``, ``coverage.csv``,
``measured_landscape.csv``, the last from the MEASURED tau and P-value of the raw Kuzmin 2018
Data S1 table by the paper's |tau| > 0.08, P < 0.05 convention), (4) writes
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
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome  # noqa: E402
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
#: Translucent backing for in-axes annotations that sit over data (panels b, g).
ANNOT_BOX = dict(boxstyle="square,pad=0.3", facecolor="white", alpha=0.8, edgecolor="none")

# Thresholds. ``LETHAL`` is the pipeline's own "lethal" cut (targeted_fba_growth_fast.py
# reports fitness < 0.01 as lethal). ``WT_LIKE`` and ``TAU_ZERO`` separate LP round-off
# from a modeled effect: fitness values within 1e-3 of 1 and |tau| below 1e-3 differ from
# the wild type by less than the solver's optimum-to-optimum jitter observed in the files
# (of order 1e-7) times the number of terms in tau, and no biological band in the run
# sits there (the smallest non-wild-type fitness band is 0.994).
LETHAL = 0.01
WT_LIKE = 1e-3
TAU_ZERO = 1e-3
# Kuzmin 2018's "intermediate score cutoff" for a significant interaction.
SIG_TAU = 0.08
SIG_P = 0.05
RAW_ZIP = osp.join(REPO_ROOT, "data", "host", "kuzmin2018", "aao1729_data_s1.zip")
#: Kuzmin 2020 Table S1 (the main screens) and Table S3 (the pilot screens); the
#: TmiKuzmin2020Dataset loader ingests both.
RAW_2020_XLSX = [
    osp.join(os.getenv("DATA_ROOT"), "data", "torchcell", "tmi_kuzmin2020", "raw", f"aaz5667-Table-{t}.xlsx")
    for t in ("S1", "S3")
]


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


def raw_labels() -> tuple[pd.DataFrame, dict]:
    """The measured labels of every triple, keyed by its sorted gene set, from the raw tables.

    The triples of the build are the trigenic records of Kuzmin 2018 and Kuzmin 2020 (the
    query ``queries/001_small_build.cql`` unions ``TmiKuzmin2018Dataset`` and
    ``TmiKuzmin2020Dataset``). The raw tables the loaders ingest are Kuzmin 2018 Data S1
    (``data/host/kuzmin2018/aao1729_data_s1.zip``) and Kuzmin 2020 Tables S1 and S3 (main and
    pilot screens, ``$DATA_ROOT/data/torchcell/tmi_kuzmin2020/raw/aaz5667-Table-S{1,3}.xlsx``),
    all in the same 12-column layout; the ``trigenic`` rows are used. Per gene set: the mean
    adjusted interaction score (tau) and the mean triple-mutant fitness over its records,
    which is what ``MeanExperimentDeduplicator`` produces for a duplicated genotype, and the
    smallest P-value. These raw-derived labels are the ones used for every correlation and
    classification, because the ``experimental`` column of the frozen
    ``matched_fba_experimental_fixed.parquet`` is misaligned with its gene sets (see
    ``correlations``). The parquet cache in ``results/fba_baseline_si/`` is rebuilt when
    absent; the raw files' sha256 are recorded either way.
    """
    import hashlib
    import zipfile

    def sha256(path):
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    cache = osp.join(OUT, "raw_trigenic_labels.parquet")
    if osp.exists(cache):
        grp = pd.read_parquet(cache)
    else:
        with zipfile.ZipFile(RAW_ZIP) as z:
            with z.open("aao1729_data_s1.tsv") as f:
                raw18 = pd.read_csv(f, sep="\t")
        raw18 = raw18.rename(columns={"Combined mutant fitness": "fitness"})
        raw20 = pd.concat([pd.read_excel(p, skiprows=1) for p in RAW_2020_XLSX], ignore_index=True)
        raw20 = raw20.rename(columns={"Double/triple mutant fitness": "fitness"})
        raw18["source"] = "kuzmin2018"
        raw20["source"] = "kuzmin2020"
        raw = pd.concat([raw18, raw20], ignore_index=True)
        raw = raw[raw["Combined mutant type"] == "trigenic"].copy()
        q = raw["Query strain ID"].str.split("+", expand=True)
        q1 = q[0]
        q2 = q[1].str.split("_", expand=True)[0]
        arr = raw["Array strain ID"].str.split("_", expand=True)[0]
        raw["genes"] = [",".join(sorted(t)) for t in zip(q1, q2, arr)]
        grp = raw.groupby("genes").agg(
            n_raw_records=("P-value", "size"),
            p_min=("P-value", "min"),
            tau_measured=("Adjusted genetic interaction score (epsilon or tau)", "mean"),
            fitness_measured=("fitness", "mean"),
            in_kuzmin2018=("source", lambda s: bool((s == "kuzmin2018").any())),
            in_kuzmin2020=("source", lambda s: bool((s == "kuzmin2020").any())),
        ).reset_index()
        grp.to_parquet(cache, index=False)
    meta = {
        "raw_kuzmin2018": osp.relpath(RAW_ZIP, REPO_ROOT),
        "raw_kuzmin2018_sha256": sha256(RAW_ZIP),
        "raw_kuzmin2020": [osp.relpath(p, os.getenv("DATA_ROOT")) for p in RAW_2020_XLSX],
        "raw_kuzmin2020_sha256": [sha256(p) for p in RAW_2020_XLSX],
        "n_raw_trigenic_gene_sets": int(len(grp)),
    }
    return grp, meta


def measured_landscape(frozen: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Classify the MEASURED tau of every triple by the Kuzmin 2018 convention, split by how
    many of the triple's genes Yeast9 carries.

    Significant: |tau| > 0.08 and P < 0.05 (Kuzmin 2018, "intermediate score cutoff"), on
    the raw-derived labels of ``raw_labels``. "Not significant" holds everything else,
    including |tau| > 0.08 at P >= 0.05.
    """
    df = frozen
    sig = (df.tau_measured.abs() > SIG_TAU) & (df.p_min < SIG_P)
    df["class"] = np.where(sig & (df.tau_measured < 0), "sig_negative", np.where(sig, "sig_positive", "not_significant"))
    df["coverage"] = pd.cut(df.n_in_model, [-1, 0, 2, 3], labels=["uncovered", "partial", "full"]).astype(str)
    rows = []
    for cov_key, label in [("full", "All 3 genes"), ("partial", "1 or 2 genes"), ("uncovered", "No gene")]:
        sub = df[df.coverage == cov_key]
        row = {"coverage": cov_key, "coverage_label": label, "n": int(len(sub))}
        for c in ("sig_negative", "not_significant", "sig_positive"):
            row[f"n_{c}"] = int((sub["class"] == c).sum())
            row[f"frac_{c}"] = float((sub["class"] == c).mean())
        row["n_abs_tau_above_0.08"] = int((sub.tau_measured.abs() > SIG_TAU).sum())
        row["n_p_below_0.05"] = int((sub.p_min < SIG_P).sum())
        rows.append(row)
    land = pd.DataFrame(rows)
    meta = {
        "n_triples_in_kuzmin2018": int(df.in_kuzmin2018.sum()),
        "n_triples_in_kuzmin2020": int(df.in_kuzmin2020.sum()),
        "n_triples_in_both": int((df.in_kuzmin2018 & df.in_kuzmin2020).sum()),
        "n_triples_with_duplicate_records": int((df.n_raw_records > 1).sum()),
        "sig_tau": SIG_TAU,
        "sig_p": SIG_P,
    }
    return land, meta, df[["genes", "class", "coverage"]]


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


def correlations(frozen: pd.DataFrame) -> dict:
    """Predicted against measured, on the raw-derived labels, plus the same statistics on
    the labels as stored in the frozen matched file, and the evidence that the stored labels
    are a permutation of the true ones.

    The stored ``experimental`` column was written by ``match_fba_to_experiments.py``, which
    paired ``dataset[i]``'s gene set with ``dataset.label_df.iloc[i]``; the two orders do not
    agree. Measured here: the sorted stored values and the sorted raw-derived values coincide
    (a permutation), while row by row they agree for about 1% of triples.
    """
    out = {}
    for key, pred, meas, stored in [
        ("tau", "tau_fba", "tau_measured", "tau_stored"),
        ("fitness", "fitness_fba", "fitness_measured", "fitness_stored"),
    ]:
        sub = frozen.dropna(subset=[pred, meas])
        r, p = stats.pearsonr(sub[pred], sub[meas])
        rho, _ = stats.spearmanr(sub[pred], sub[meas])
        st = frozen.dropna(subset=[pred, stored])
        r_st, _ = stats.pearsonr(st[pred], st[stored])
        both = frozen.dropna(subset=[meas, stored])
        a = np.sort(both[meas].to_numpy())
        b = np.sort(both[stored].to_numpy())
        out[key] = {
            "n": int(len(sub)),
            "pearson_r": float(r),
            "pearson_p": float(p),
            "spearman_rho": float(rho),
            "pred_mean": float(sub[pred].mean()),
            "pred_sd": float(sub[pred].std(ddof=1)),
            "meas_mean": float(sub[meas].mean()),
            "meas_sd": float(sub[meas].std(ddof=1)),
            "as_stored": {
                "n": int(len(st)),
                "pearson_r": float(r_st),
                "n_stored_and_raw": int(len(both)),
                "frac_rows_stored_equals_raw_1e-3": float((np.abs(both[meas] - both[stored]) < 1e-3).mean()),
                "frac_sorted_stored_equals_sorted_raw_1e-3": float((np.abs(a - b) < 1e-3).mean()),
                "pearson_r_stored_vs_raw_rowwise": float(stats.pearsonr(both[meas], both[stored])[0]),
            },
        }
    fvc = pd.Series(np.round(frozen.fitness_fba.to_numpy(), 3)).value_counts()
    out["fitness"]["value_counts_round3"] = {str(k): int(v) for k, v in fvc.head(8).items()}
    out["fitness"]["n_distinct_round3"] = int(len(fvc))
    tau = frozen.tau_fba.to_numpy()
    out["tau"]["frac_abs_below_1e-6"] = float((np.abs(tau) < 1e-6).mean())
    out["tau"]["frac_abs_below_1e-3"] = float((np.abs(tau) < TAU_ZERO).mean())
    out["tau"]["n_abs_above_1e-3"] = int((np.abs(tau) > TAU_ZERO).sum())
    nz = pd.Series(np.round(tau[np.abs(tau) > TAU_ZERO], 2)).value_counts()
    out["tau"]["nonzero_value_counts"] = {str(k): int(v) for k, v in nz.head(8).items()}
    sub = frozen[np.abs(frozen.tau_fba) > TAU_ZERO]
    r_nz, _ = stats.pearsonr(sub.tau_fba, sub.tau_measured)
    out["tau"]["pearson_r_nonzero_only"] = float(r_nz)
    out["tau"]["n_nonzero_only"] = int(len(sub))
    # Fitness correlation within the triples the model can act on at all.
    cov = frozen[frozen.n_in_model > 0]
    out["fitness"]["pearson_r_triples_with_model_gene"] = float(stats.pearsonr(cov.fitness_fba, cov.fitness_measured)[0])
    out["fitness"]["n_triples_with_model_gene"] = int(len(cov))
    return out


# ----------------------------------------------------------------------------- panels
def panel_tau(frozen, corr):
    """Predicted vs measured tau as a hexbin with log counts."""
    gi = frozen
    w = mm_to_in(PANEL_WIDTHS_MM["third"])
    fig, ax = plt.subplots(figsize=(w, mm_to_in(50)))
    fig.subplots_adjust(left=0.2, right=0.82, bottom=0.17, top=0.95)
    cmap = LinearSegmentedColormap.from_list("amber", ["#FFFFFF", ORANGE_F, ORANGE, PLOT_PALETTE[6]])
    hb = ax.hexbin(
        gi.tau_fba,
        gi.tau_measured,
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
    # Upper right: the data stand in a column at x = 0, so the upper-left corner is not free.
    ax.text(
        0.97,
        0.97,
        f"Pearson $r$ = {c['pearson_r']:.4f}\n$n$ = {c['n']:,}\n"
        f"{100 * c['frac_abs_below_1e-3']:.2f}% at $|\\tau|<10^{{-3}}$",
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=6,
        bbox=ANNOT_BOX,
    )
    box(ax)
    save(fig, "fba_baseline_tau")


def panel_fitness(frozen, corr):
    """Predicted vs measured triple-mutant fitness as a hexbin with log counts."""
    fi = frozen.dropna(subset=["fitness_measured", "fitness_fba"])
    w = mm_to_in(PANEL_WIDTHS_MM["third"])
    fig, ax = plt.subplots(figsize=(w, mm_to_in(50)))
    fig.subplots_adjust(left=0.2, right=0.82, bottom=0.17, top=0.95)
    cmap = LinearSegmentedColormap.from_list("lilac", ["#FFFFFF", PURPLE_F, PURPLE, PLOT_PALETTE[8]])
    hb = ax.hexbin(
        fi.fitness_fba,
        fi.fitness_measured,
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


def panel_evaluable(cov: pd.DataFrame, n_in: dict, n_genes: int, n_doubles: int):
    """How much of the screen Yeast9 can score: genes and doubles with every member in the
    model, and triples by how many of their three genes are in the model.
    """
    w = mm_to_in(PANEL_WIDTHS_MM["third"])
    fig, ax = plt.subplots(figsize=(w, mm_to_in(50)))
    fig.subplots_adjust(left=0.2, right=0.97, bottom=0.24, top=0.95)
    n_tri = int(cov.n_triples.sum())
    labels = ["Genes", "Doubles", "0", "1", "2", "3"]
    counts = [n_in["singles"], n_in["doubles"]] + [int(cov.loc[cov.n_in_model == k, "n_triples"].iloc[0]) for k in range(4)]
    denoms = [n_genes, n_doubles] + [n_tri] * 4
    fracs = [c / d for c, d in zip(counts, denoms)]
    colors = [GRAY, GRAY, PURPLE, PURPLE, PURPLE, PURPLE]
    x = np.array([0, 1.2, 2.7, 3.7, 4.7, 5.7])
    ax.bar(x, fracs, 0.7, color=colors, edgecolor="black", linewidth=0.4)
    for xi, f, c in zip(x, fracs, counts):
        ax.text(xi, f + 0.02, f"{c:,}", ha="center", va="bottom", fontsize=5, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    xlo, xhi = -0.6, 6.3
    ax.set_xlim(xlo, xhi)
    xa = lambda v: (v - xlo) / (xhi - xlo)  # data x -> axes fraction
    ax.text(xa(0.6), -0.17, "all members\nin Yeast9", ha="center", va="top", fontsize=6, transform=ax.transAxes)
    ax.text(xa(4.2), -0.17, "Triples, by genes\nin Yeast9", ha="center", va="top", fontsize=6, transform=ax.transAxes)
    ax.set_ylabel("Fraction of the set")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(axis="y", which="minor", length=0)
    ax.grid(axis="y", which="both", color="0.85", lw=0.4)
    ax.set_axisbelow(True)
    box(ax)
    save(fig, "fba_baseline_evaluable")


def panel_landscape(land: pd.DataFrame):
    """Measured trigenic interactions of the screen, by how much of the triple Yeast9 covers."""
    w = mm_to_in(PANEL_WIDTHS_MM["third"])
    fig, ax = plt.subplots(figsize=(w, mm_to_in(50)))
    fig.subplots_adjust(left=0.2, right=0.97, bottom=0.24, top=0.95)
    groups = list(land.coverage)
    cats = [("sig_negative", "Negative", RED), ("not_significant", "Not significant", GRAY), ("sig_positive", "Positive", ORANGE)]
    x = np.arange(len(groups))
    bw = 0.26
    base = 0.5  # log axis: bars rise from 0.5, so a count of 1 is visible
    for k, (col, label, color) in enumerate(cats):
        cnt = land[f"n_{col}"].to_numpy()
        ax.bar(x + (k - 1) * bw, np.maximum(cnt, base), bw, bottom=base, color=color, edgecolor="black", linewidth=0.4, label=label)
        for xi, c in zip(x + (k - 1) * bw, cnt):
            ax.text(xi, max(c, base) * 1.4, f"{c:,}", ha="center", va="bottom", fontsize=5, rotation=90)
    ax.set_yscale("log")
    ax.set_ylim(base, 1e9)
    ax.set_yticks([1, 10, 100, 1e3, 1e4, 1e5, 1e6])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g}\n$n$ = {n:,}" for g, n in zip(land.coverage_label, land.n)])
    ax.set_ylabel("Triples")
    ax.grid(axis="y", which="major", color="0.85", lw=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", frameon=False, ncol=3, columnspacing=0.6, handlelength=0.8, handletextpad=0.3,
              handleheight=0.8, borderaxespad=0.2, title=r"Measured $\tau_{ijk}$: $|\tau|>0.08$ and $P<0.05$", title_fontsize=6)
    box(ax)
    save(fig, "fba_baseline_landscape")


# ----------------------------------------------------------------------------- panel a glyphs
def panel_schematic_genes(n_genome: int, n_model: int, n_screened: int, n_screened_in_model: int):
    """Two proportion bars for the Yeast9 card of panel a: of the genome's protein-coding
    genes and of the screened genes, the part the model carries (filled).
    """
    fig, ax = plt.subplots(figsize=(mm_to_in(27), mm_to_in(15)))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.96)
    rows = [("Genome", n_genome, n_model), ("Screened", n_screened, n_screened_in_model)]
    for i, (label, total, inside) in enumerate(rows):
        y = 1 - i
        ax.barh(y, 1.0, height=0.42, color="white", edgecolor="black", lw=0.5, zorder=2)
        ax.barh(y, inside / total, height=0.42, color=ORANGE, edgecolor="black", lw=0.5, zorder=3)
        ax.text(0, y + 0.31, f"{label}: {total:,} genes", ha="left", va="bottom", fontsize=6)
        ax.text(inside / total + 0.02, y, f"{inside:,} in Yeast9 ({100 * inside / total:.0f}%)", ha="left", va="center", fontsize=6)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.4, 1.75)
    ax.axis("off")
    save(fig, "fba_schematic_genes")


def panel_schematic_deletions(counts: dict, covered: dict):
    """Bar chart for the deletion-sets card of panel a: the single, double and triple
    deletion sets of the screen (log count), with the sets whose every gene is in Yeast9
    as the darker part.
    """
    fig, ax = plt.subplots(figsize=(mm_to_in(27), mm_to_in(25)))
    fig.subplots_adjust(left=0.3, right=0.97, bottom=0.25, top=0.82)
    orders = ["singles", "doubles", "triples"]
    x = np.arange(3)
    tot = [counts[o] for o in orders]
    cov = [covered[o] for o in orders]
    ax.bar(x, tot, 0.62, color=YELLOW, edgecolor="black", lw=0.5, zorder=3, label="all sets")
    ax.bar(x, cov, 0.62, color=PLOT_PALETTE[9], edgecolor="black", lw=0.5, zorder=4, label="in Yeast9")
    for xi, t, cv in zip(x, tot, cov):
        ax.text(xi, t * 1.35, f"{t:,}", ha="center", va="bottom", fontsize=5)
        ax.text(xi, cv * 0.5, f"{cv:,}", ha="center", va="center", fontsize=5, color="white")
    ax.set_yscale("log")
    ax.set_ylim(100, 5e6)
    ax.set_yticks([1e2, 1e4, 1e6])
    ax.set_xticks(x)
    ax.set_xticklabels(["1", "2", "3"])
    ax.set_xlim(-0.6, 2.75)
    ax.set_xlabel("Genes deleted", labelpad=1)
    ax.set_ylabel("Deletion sets", labelpad=1)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=5, handlelength=0.8,
              handleheight=0.8, borderaxespad=0.0, columnspacing=0.8, handletextpad=0.4)
    box(ax)
    save(fig, "fba_schematic_deletions")


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
    genome = SCerevisiaeGenome(genome_root=osp.join(os.getenv("DATA_ROOT"), "data/sgd/genome"),
                               go_root=osp.join(os.getenv("DATA_ROOT"), "data/go"), overwrite=False)
    n_genome_genes = len(genome.gene_set)

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
        "n_genome_genes": n_genome_genes,
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
    consist = consistency(model_genes, singles, doubles, triples)

    # Frozen per-triple table: predicted tau and fitness (from the run), the stored labels
    # of the matched file, the raw-derived labels, and model coverage.
    gi = matched[matched.phenotype_type == "gene_interaction"][["genes", "experimental", "fba_predicted"]]
    gi = gi.rename(columns={"experimental": "tau_stored", "fba_predicted": "tau_fba"})
    fi = matched[matched.phenotype_type == "fitness"][["genes", "experimental", "fba_predicted"]]
    fi = fi.rename(columns={"experimental": "fitness_stored", "fba_predicted": "fitness_fba"})
    tri["genes"] = tri.gene1 + "," + tri.gene2 + "," + tri.gene3
    labels, raw_meta = raw_labels()
    frozen = (
        gi.merge(fi, on="genes", how="left")
        .merge(tri[["genes", "n_in_model"]], on="genes", how="left")
        .merge(labels, on="genes", how="left")
    )
    assert len(frozen) == len(gi) == meta["n_triples"], (len(frozen), len(gi), meta["n_triples"])
    n_unmatched = int(frozen.p_min.isna().sum())
    assert n_unmatched == 0, f"{n_unmatched} triples have no raw record in Kuzmin 2018 or 2020"
    corr = correlations(frozen)
    land, land_meta, land_rows = measured_landscape(frozen)
    land_meta = {**raw_meta, **land_meta}
    frozen = frozen.merge(land_rows, on="genes", how="left")
    frozen.to_parquet(osp.join(OUT, "triples_matched.parquet"), index=False)
    land.to_csv(osp.join(OUT, "measured_landscape.csv"), index=False)

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
        "measured_landscape": {"meta": land_meta, "by_coverage": land.to_dict(orient="records")},
        "fig2d_value_in_manuscript": 0.0006,
    }
    json.dump(stats_out, open(osp.join(OUT, "stats.json"), "w"), indent=2)
    print(json.dumps(stats_out, indent=2))

    write_medium_table(med, model_stats, osp.join(TEX_DIR, "tab-fba-medium.tex"))
    panel_tau(frozen, corr)
    panel_fitness(frozen, corr)
    panel_growth_bands(bands)
    panel_evaluable(cov, n_in, len(perts["singles"]), len(perts["doubles"]))
    panel_landscape(land)
    panel_schematic_genes(n_genome_genes, len(model.genes), len(perts["singles"]), n_in["singles"])
    panel_schematic_deletions(
        {"singles": len(perts["singles"]), "doubles": len(perts["doubles"]), "triples": len(perts["triples"])},
        {"singles": n_in["singles"], "doubles": n_in["doubles"], "triples": int(cov.loc[cov.n_in_model == 3, "n_triples"].iloc[0])},
    )


if __name__ == "__main__":
    main()
