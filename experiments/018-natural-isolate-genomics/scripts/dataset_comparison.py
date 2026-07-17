# experiments/018-natural-isolate-genomics/scripts/dataset_comparison.py
# [[experiments.018-natural-isolate-genomics.dataset-comparison]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/018-natural-isolate-genomics/scripts/dataset_comparison

"""Fig 4 descriptive panels: how different are engineered-KO and natural-isolate strains?

One panel, one question (see [[experiments.018-natural-isolate-genomics.expression-modeling-setup]]):

- **b** genotype divergence — # reference ORFs absent (x) vs % sequence divergence on
  shared genes (y). Engineered KOs sit at (1 gene, 0 bp); natural isolates spread far out.
  The two occupy disjoint genotype design spaces.
- **c** per-strain spread violins — IQR of genome-wide log2 FC per strain, one axis.
- **d** transcriptome spread — Kemmeren single-KO / Sameith single-KO / Sameith double-KO /
  Caudal natural-isolate as matched sorted spread bands on ONE shared y-scale.
- **d2** size-aligned spread — the same per-strain IQR profiles OVERLAID on one axis and
  right-aligned at the highest-variance strain, so tail length reads off dataset size.
- **e** how many genes move — per-strain DE-count distribution, KO (Kemmeren) vs natural
  (Caudal), on the paper-exact rule.
- **f** transcriptome design-space coverage — PCA and UMAP of the joint expression matrix,
  coloured by dataset (per-gene z-scored WITHIN dataset to blunt the microarray/RNA-seq
  platform confound; residual separation is partly platform, stated on the panel).

Palette (green-free): Kemmeren = orange, Caudal = red (the focus). In c/d/e both Sameith
arms share purple (separated by position); in the OVERLAY panels f and d2 they split into
dark purple (single) and dark red (double) via ``SPLIT_COLORS`` so they read where they
overlap. Repo figure standard throughout (Arial 6 pt, boxed, true-size SVG + PNG). Genotype
inputs (b) come from the 018 result parquets; expression (c, d, d2, e, f) from the built LMDBs.
"""

import logging
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.lines import Line2D
from scipy.stats import false_discovery_control, norm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from umap import UMAP

from torchcell.datasets.scerevisiae.caudal2024 import CaudalPanTranscriptome2024Dataset
from torchcell.datasets.scerevisiae.kemmeren2014 import MicroarrayKemmeren2014Dataset
from torchcell.datasets.scerevisiae.sameith2015 import (
    DmMicroarraySameith2015Dataset,
    SmMicroarraySameith2015Dataset,
)
from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    PLOT_PALETTE_FILL,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")
RESULTS = osp.join(EXPERIMENT_ROOT, "018-natural-isolate-genomics/results")
IMG_DIR = osp.join(ASSET_IMAGES_DIR, "018-natural-isolate-genomics")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

CAUDAL_PSEUDOCOUNT = 1.0  # matches differential_expression_comparison.py:127
LOG2_FC = float(np.log2(1.7))  # Kemmeren 2014 DE rule: |log2 FC| > log2(1.7) = 0.766
INK = "#000000"
GRID = "#4A4A4A"
PANEL_H_MM = (
    35.7  # canonical wide-strip height unit (figure-sizing-template.drawio.svg)
)

# per-dataset colours, consistent across every panel; green-free. Red is our lead
# hue (Caudal, the natural-isolate focus). Kemmeren single = orange, Caudal = red,
# and BOTH Sameith arms = purple (one lab/platform, grouped as a single colour).
# Yellow is intentionally retired from the arms so the scheme reads as three hues.
DS = {
    "kemmeren_single": {
        "label": "Kemmeren single KO",
        "line": PLOT_PALETTE[0],  # orange
        "fill": PLOT_PALETTE_FILL[0],
        "marker": "o",
    },
    "caudal": {
        "label": "Caudal natural isolate",
        "line": PLOT_PALETTE[1],  # red
        "fill": PLOT_PALETTE_FILL[1],
        "marker": "o",
    },
    "sameith_single": {
        "label": "Sameith single KO",
        "line": PLOT_PALETTE[2],  # purple
        "fill": PLOT_PALETTE_FILL[2],
        "marker": "o",
    },
    "sameith_double": {
        "label": "Sameith double KO",
        "line": PLOT_PALETTE[2],  # purple (grouped with Sameith single)
        "fill": PLOT_PALETTE_FILL[2],
        "marker": "o",
    },
}

# Panels f and d2 OVERLAY the two sparse Sameith arms (embedding scatter; right-aligned
# curves), so unlike c/d/e — which separate them by position and keep both purple — they
# need distinct colours. Use the DARK tier so the least-populous Sameith arms read on top:
# Sameith single = dark purple, Sameith double = dark red (distinct from Caudal's brighter
# red). Kemmeren orange and Caudal red are unchanged.
SPLIT_COLORS = {
    "kemmeren_single": PLOT_PALETTE[0],  # orange
    "caudal": PLOT_PALETTE[1],  # red
    "sameith_single": PLOT_PALETTE[8],  # dark purple
    "sameith_double": PLOT_PALETTE[7],  # dark red
}


def _apply_rc() -> None:
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
            "svg.fonttype": "none",
            "axes.linewidth": 0.5,
            "savefig.bbox": None,  # torchcell.mplstyle sets 'tight'; would break true-size
        }
    )


def _box(ax):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_color(INK)
        ax.spines[s].set_linewidth(0.5)
    ax.tick_params(colors=INK, width=0.5, length=2)
    ax.grid(True, alpha=0.15, linewidth=0.4, color=GRID)
    ax.set_axisbelow(True)


def _save(fig, name):
    os.makedirs(IMG_DIR, exist_ok=True)
    png, svg = osp.join(IMG_DIR, f"{name}.png"), osp.join(IMG_DIR, f"{name}.svg")
    fig.savefig(png, dpi=300, facecolor="white")
    savefig_true_size_svg(fig, svg, facecolor="white")
    plt.close(fig)
    logger.info(f"✓ {png} + {osp.basename(svg)}")


# ----------------------------------------------------------------------------- data
def _log2_from_ratio(dataset, name, sample=None):
    """Kemmeren/Sameith: {strain: {gene: log2ratio}} straight from expression_log2_ratio."""
    logger.info(f"Extracting {name}")
    out = {}
    rng = range(len(dataset) if sample is None else min(sample, len(dataset)))
    for i in tqdm(rng, desc=name):
        rec = dataset[i]
        perts = rec["experiment"]["genotype"]["perturbations"]
        strain = "+".join(sorted(p["systematic_gene_name"] for p in perts))
        prof = rec["experiment"]["phenotype"]["expression_log2_ratio"]
        out[strain] = {g: v for g, v in prof.items() if not np.isnan(v)}
    return out


def _log2_from_caudal(dataset, sample=None):
    """Caudal: {strain: {gene: log2((tpm+1)/(ref+1))}} vs the per-record population mean."""
    logger.info("Extracting Caudal")
    out = {}
    rng = range(len(dataset) if sample is None else min(sample, len(dataset)))
    for i in tqdm(rng, desc="Caudal"):
        rec = dataset[i]
        perts = rec["experiment"]["genotype"]["perturbations"]
        # a natural isolate has thousands of perturbations, ALL stamped with the same
        # isolate id -- key by strain_id, not a perturbation gene (which collides).
        strain = perts[0]["strain_id"] if perts else f"isolate_{i}"
        tpm = rec["experiment"]["phenotype"]["expression_tpm"]
        ref = rec["reference"]["phenotype_reference"]["expression_tpm"]
        out[strain] = {
            g: float(
                np.log2((tpm[g] + CAUDAL_PSEUDOCOUNT) / (ref[g] + CAUDAL_PSEUDOCOUNT))
            )
            for g in tpm
            if g in ref
        }
    return out


def _quantile_frame(log2_by_strain):
    """Per-strain p5/q1/median/q3/p95 of the genome-wide log2, sorted by IQR."""
    rows = []
    for strain, prof in log2_by_strain.items():
        v = np.fromiter(prof.values(), dtype=float)
        p5, q1, med, q3, p95 = np.percentile(v, [5, 25, 50, 75, 95])
        rows.append(
            {
                "strain": strain,
                "p5": p5,
                "q1": q1,
                "median": med,
                "q3": q3,
                "p95": p95,
                "iqr": q3 - q1,
            }
        )
    return pd.DataFrame(rows).sort_values("iqr").reset_index(drop=True)


def _sameith_dm_de_counts(dm, sample=None):
    """Per-strain noise-controlled DE count for Sameith double KOs, on the same rule as
    the other arms (|log2 FC| > log2(1.7) AND BH-adjusted p < 0.05). p comes from each
    gene's stored log2-ratio SE: z = M / SE, two-sided normal, BH-adjusted within strain.
    """
    counts = []
    rng = range(len(dm) if sample is None else min(sample, len(dm)))
    for i in tqdm(rng, desc="Sameith-DM DE"):
        ph = dm[i]["experiment"]["phenotype"]
        m_d, se_d = ph["expression_log2_ratio"], ph["expression_log2_ratio_se"]
        genes = [
            g
            for g in m_d
            if g in se_d
            and se_d[g]
            and np.isfinite(m_d[g])
            and np.isfinite(se_d[g])
            and se_d[g] > 0
        ]
        m = np.array([m_d[g] for g in genes])
        se = np.array([se_d[g] for g in genes])
        p_adj = false_discovery_control(2.0 * norm.sf(np.abs(m / se)))
        counts.append(int(np.sum((np.abs(m) > LOG2_FC) & (p_adj < 0.05))))
    return np.array(counts)


# ----------------------------------------------------------------------------- panel b
def panel_b_genotype_divergence():
    """# reference ORFs absent vs % sequence divergence; KO at (1, 0), isolates far out."""
    burden = pd.read_parquet(osp.join(RESULTS, "natural_ko_burden.parquet"))
    div = pd.read_parquet(osp.join(RESULTS, "per_strain_divergence_summary.parquet"))
    iso = burden.merge(
        div[["strain", "genome_wide_divergence"]], on="strain", how="inner"
    )
    iso["pct"] = 100.0 * iso["genome_wide_divergence"]

    _apply_rc()
    w = mm_to_in(PANEL_WIDTHS_MM["half_plus"])
    fig, ax = plt.subplots(figsize=(w, mm_to_in(66)), constrained_layout=True)
    ax.scatter(
        iso["n_absent"].clip(lower=0.5),
        iso["pct"],
        s=5,
        marker="o",
        facecolor=DS["caudal"]["line"],
        edgecolor="none",
        alpha=0.55,
        label=DS["caudal"]["label"],
    )
    # engineered KO: 1 gene fully removed, 0 bp change on the rest of the genome.
    ax.scatter(
        [1],
        [0.001],
        s=16,
        marker="D",
        facecolor=DS["kemmeren_single"]["line"],
        edgecolor=INK,
        linewidth=0.5,
        zorder=5,
        label="engineered KO (Kemmeren / Sameith)",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("reference ORFs absent (count)")
    ax.set_ylabel("sequence divergence on shared genes (%)")
    ax.set_title(
        "Genotype: engineered vs natural occupy disjoint space",
        loc="left",
        color=INK,
        fontsize=7,
    )
    ax.legend(loc="lower right", frameon=False, handletextpad=0.4, borderpad=0.3)
    _box(ax)
    _save(fig, "comparison_b_genotype_divergence")
    return {
        "n_isolates": int(len(iso)),
        "median_absent": float(iso["n_absent"].median()),
        "median_pct_div": float(iso["pct"].median()),
    }


# ----------------------------------------------------------------------------- panel c
def panel_c_spread_by_dataset(frames):
    """Per-strain transcriptome spread (IQR of log2 FC) per dataset, on one shared axis.

    The cross-dataset noise comparison panel d's ranked bands cannot show together --
    each dataset has a different strain count, hence a different x-axis. Collapsing each
    strain to its IQR puts all four on one directly comparable axis.
    """
    _apply_rc()
    order = ["kemmeren_single", "sameith_single", "sameith_double", "caudal"]
    data = [frames[k]["iqr"].to_numpy() for k in order]
    w = mm_to_in(PANEL_WIDTHS_MM["half_plus"])
    fig, ax = plt.subplots(figsize=(w, mm_to_in(66)), constrained_layout=True)
    parts = ax.violinplot(
        data,
        positions=range(len(order)),
        showmedians=True,
        showextrema=False,
        widths=0.8,
    )
    for pc, key in zip(parts["bodies"], order):
        pc.set_facecolor(DS[key]["line"])
        pc.set_edgecolor(INK)
        pc.set_linewidth(0.4)
        pc.set_alpha(0.85)
    parts["cmedians"].set_color(INK)
    parts["cmedians"].set_linewidth(0.8)
    labs = [
        "Kemmeren\nsingle KO",
        "Sameith\nsingle KO",
        "Sameith\ndouble KO",
        "Caudal\nisolate",
    ]
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(
        [f"{lab}\n(n = {len(frames[k])})" for k, lab in zip(order, labs)]
    )
    ax.set_ylabel("per-strain spread (IQR of log2 FC)")
    ax.set_title(
        "How much each strain's transcriptome moves", loc="left", color=INK, fontsize=7
    )
    _box(ax)
    _save(fig, "comparison_c_spread_by_dataset")
    return {k: float(np.median(frames[k]["iqr"])) for k in order}


# ----------------------------------------------------------------------------- panel d
def panel_d_transcriptome_bands(frames):
    """Four datasets as matched sorted spread bands on one shared y-scale."""
    _apply_rc()
    order = ["kemmeren_single", "sameith_single", "sameith_double", "caudal"]
    ymax = float(
        np.ceil(
            max(max(f["p95"].abs().max(), f["p5"].abs().max()) for f in frames.values())
            * 10
        )
        / 10
    )
    fig, axes = plt.subplots(
        len(order),
        1,
        figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(4 * PANEL_H_MM)),
        constrained_layout=True,
        sharex=False,
    )
    for ax, key in zip(axes, order):
        df = frames[key]
        x = np.arange(len(df))
        ax.fill_between(
            x, df["p5"], df["p95"], facecolor=DS[key]["fill"], linewidth=0, zorder=2
        )
        ax.fill_between(
            x, df["q1"], df["q3"], facecolor=DS[key]["line"], linewidth=0, zorder=3
        )
        ax.plot(x, df["median"], color=INK, linewidth=0.5, zorder=4)
        ax.set_ylim(-ymax, ymax)
        ax.axhline(0, color=GRID, linestyle=":", linewidth=0.5, zorder=1)
        ax.set_xlim(0, max(len(df) - 1, 1))
        ax.set_ylabel("log2 FC")
        ax.set_title(
            f"{DS[key]['label']}  (n = {len(df)})", loc="left", color=INK, fontsize=6
        )
        _box(ax)
    axes[-1].set_xlabel("deletion / isolate strains, ranked by transcriptome spread")
    _save(fig, "comparison_d_transcriptome_bands")
    return {"shared_ymax": ymax, **{k: int(len(v)) for k, v in frames.items()}}


# ---------------------------------------------------------------------------- panel d2
def panel_d2_size_aligned(frames):
    """Companion to d: all four per-strain IQR profiles OVERLAID on one axis and
    RIGHT-ALIGNED at each dataset's highest-variance strain (x = 0, right edge). Panel d
    stretches every dataset to the same panel width, hiding the size gap; here each
    dataset's curve extends left for exactly as many strains as it has, so the TAIL LENGTH
    reads off dataset size (Sameith peters out fast; Kemmeren/Caudal run the full width)
    while the curve HEIGHT is the spread profile. Uses the split Sameith colours (f-scheme)
    so the two arms are distinguishable where they overlap.
    """
    _apply_rc()
    order = ["kemmeren_single", "caudal", "sameith_single", "sameith_double"]
    w = mm_to_in(PANEL_WIDTHS_MM["full"])
    fig, ax = plt.subplots(
        figsize=(w, mm_to_in(2 * PANEL_H_MM)), constrained_layout=True
    )
    for key in order:
        df = frames[
            key
        ]  # sorted ascending by IQR -> last row is the highest-variance strain
        n = len(df)
        x = np.arange(n) - (
            n - 1
        )  # highest-variance strain at x=0 (right); tail runs left
        ax.plot(
            x,
            df["iqr"].to_numpy(),
            color=SPLIT_COLORS[key],
            linewidth=0.9,
            label=f"{DS[key]['label']}  (n = {n})",
            zorder=2,
        )
    ax.axvline(0, color=GRID, linestyle=":", linewidth=0.5, zorder=1)
    ax.set_xlabel(
        "strains ranked by spread, right-aligned at the highest-variance strain (x = 0)"
    )
    ax.set_ylabel("per-strain IQR of log2 FC")
    ax.set_title(
        "Dataset size and spread on one aligned axis", loc="left", color=INK, fontsize=7
    )
    ax.legend(
        loc="upper left", frameon=False, handletextpad=0.4, borderpad=0.3, fontsize=6
    )
    _box(ax)
    _save(fig, "comparison_d2_size_aligned")
    return {k: int(len(frames[k])) for k in order}


# ----------------------------------------------------------------------------- panel e
def panel_e_de_counts(sameith_de):
    """Per-strain DE-count distribution: single KO (Kemmeren) vs double KO (Sameith) vs
    natural isolate (Caudal), all under the same noise-controlled rule.
    """
    de = pd.read_parquet(osp.join(RESULTS, "de_counts_per_strain.parquet"))
    kem = de[de["dataset"] == "kemmeren2014_single_ko"]["n_de_paper_exact"].to_numpy()
    cau = de[de["dataset"].str.startswith("caudal")]["n_de_paper_exact"].to_numpy()

    _apply_rc()
    w = mm_to_in(PANEL_WIDTHS_MM["half_plus"])
    fig, ax = plt.subplots(figsize=(w, mm_to_in(60)), constrained_layout=True)
    hi = max(kem.max(), cau.max(), int(sameith_de.max()) if len(sameith_de) else 1) + 1
    bins = np.logspace(0, np.log10(hi), 40)
    for arr, key, lbl in [
        (kem, "kemmeren_single", "Kemmeren single KO"),
        (sameith_de, "sameith_double", "Sameith double KO"),
        (cau, "caudal", "Caudal isolate"),
    ]:
        ax.hist(
            np.clip(arr, 1, None),
            bins=bins,
            color=DS[key]["line"],
            edgecolor=INK,
            linewidth=0.2,
            alpha=0.6,
            label=f"{lbl} (median {int(np.median(arr))})",
        )
    ax.set_xscale("log")
    ax.set_xlabel("differentially expressed genes per strain")
    ax.set_ylabel("strains")
    ax.set_title(
        "Genes moved per strain, by perturbation class",
        loc="left",
        color=INK,
        fontsize=7,
    )
    ax.legend(loc="upper right", frameon=False, handletextpad=0.4, borderpad=0.3)
    _box(ax)
    _save(fig, "comparison_e_de_counts")
    return {
        "kemmeren_median": float(np.median(kem)),
        "sameith_double_median": float(np.median(sameith_de)),
        "caudal_median": float(np.median(cau)),
    }


# ----------------------------------------------------------------------------- panel f
def panel_f_expression_embedding(log2_by_dataset):
    """PCA + UMAP of the joint expression matrix, per-gene z-scored within dataset."""
    order = ["kemmeren_single", "sameith_single", "sameith_double", "caudal"]
    shared = set.intersection(
        *[set.union(*[set(p) for p in log2_by_dataset[k].values()]) for k in order]
    )
    genes = sorted(shared)
    logger.info(f"Panel f: {len(genes)} shared genes across all four datasets")

    mats, labels = [], []
    for key in order:
        rows = [
            [prof.get(g, np.nan) for g in genes]
            for prof in log2_by_dataset[key].values()
        ]
        m = np.asarray(rows, dtype=float)
        m = np.where(
            np.isnan(m), np.nanmean(m, axis=0, keepdims=True), m
        )  # gene-mean impute
        m = StandardScaler().fit_transform(
            m
        )  # per-gene z-score WITHIN dataset (confound guard)
        mats.append(m)
        labels += [key] * m.shape[0]
    X = np.vstack(mats)
    labels = np.asarray(labels)

    pca = PCA(n_components=2, random_state=0).fit(X)
    xy_pca = pca.transform(X)
    xy_umap = UMAP(
        n_components=2, random_state=42, n_neighbors=30, min_dist=0.3
    ).fit_transform(X)

    _apply_rc()
    fig, (axp, axu) = plt.subplots(
        1,
        2,
        figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(74)),
        constrained_layout=True,
    )
    for ax, xy, ttl in [
        (
            axp,
            xy_pca,
            f"PCA (PC1 {pca.explained_variance_ratio_[0] * 100:.0f}%, PC2 {pca.explained_variance_ratio_[1] * 100:.0f}%)",
        ),
        (axu, xy_umap, "UMAP"),
    ]:
        # dots only (no shapes) — colour alone distinguishes the arms. Plot the dense
        # big clouds first with SMALL dots so they don't blob; the two sparse Sameith
        # arms (dark purple / dark red) go on top a little larger so they stay pickable.
        plot_order = ["kemmeren_single", "caudal", "sameith_single", "sameith_double"]
        sizes = {
            "kemmeren_single": 6,
            "caudal": 6,
            "sameith_single": 22,
            "sameith_double": 22,
        }
        for z, key in enumerate(plot_order):
            m = labels == key
            ax.scatter(
                xy[m, 0],
                xy[m, 1],
                s=sizes[key],
                marker="o",
                facecolor=SPLIT_COLORS[key],
                edgecolor="none",
                alpha=0.6,
                label=DS[key]["label"],
                zorder=2 + z,
            )
        ax.set_title(ttl, loc="left", color=INK, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        _box(ax)
    axp.set_xlabel("PC1", fontsize=7)
    axp.set_ylabel("PC2", fontsize=7)
    axu.set_xlabel("UMAP-1", fontsize=7)
    # custom legend: uniform-size circles decoupled from the scatter sizes; four entries
    # (the two Sameith arms are now distinct dark colours).
    legend_spec = [
        ("kemmeren_single", "Kemmeren single KO"),
        ("caudal", "Caudal natural isolate"),
        ("sameith_single", "Sameith single KO"),
        ("sameith_double", "Sameith double KO"),
    ]
    handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            markerfacecolor=SPLIT_COLORS[k],
            markeredgecolor="none",
            markersize=5,
            label=lab,
        )
        for k, lab in legend_spec
    ]
    axu.legend(
        handles=handles,
        loc="best",
        frameon=False,
        handletextpad=0.3,
        borderpad=0.3,
        fontsize=7,
    )
    fig.suptitle(
        "Transcriptome coverage (z-scored within dataset; split is partly platform: "
        "Kemmeren/Sameith microarray vs Caudal RNA-seq)",
        x=0.005,
        ha="left",
        fontsize=8,
        color=INK,
    )
    _save(fig, "comparison_f_expression_embedding")
    return {
        "n_shared_genes": len(genes),
        "pc1_var": float(pca.explained_variance_ratio_[0]),
    }


# ------------------------------------------------------- further explanation (1) overlap
def panel_overlap_response(log2_by_dataset):
    """Per gene: KO-response frequency (fraction of Kemmeren single KOs where the gene is
    DE) vs natural-isolate variability (SD across Caudal isolates). Do the two modalities
    move the SAME genes, or complementary ones? Caveat: Kemmeren is microarray, Caudal is
    RNA-seq -- the per-gene dynamic ranges are not identical.
    """
    kem, cau = log2_by_dataset["kemmeren_single"], log2_by_dataset["caudal"]
    genes = sorted(
        set.union(*[set(p) for p in kem.values()])
        & set.union(*[set(p) for p in cau.values()])
    )
    kem_mat = np.array([[p.get(g, np.nan) for g in genes] for p in kem.values()])
    cau_mat = np.array([[p.get(g, np.nan) for g in genes] for p in cau.values()])
    ko_freq = 100.0 * np.nanmean(np.abs(kem_mat) > LOG2_FC, axis=0)
    iso_sd = np.nanstd(cau_mat, axis=0)
    ok = np.isfinite(ko_freq) & np.isfinite(iso_sd)
    r = float(np.corrcoef(ko_freq[ok], iso_sd[ok])[0, 1])

    _apply_rc()
    w = mm_to_in(PANEL_WIDTHS_MM["half_plus"])
    fig, ax = plt.subplots(figsize=(w, mm_to_in(66)), constrained_layout=True)
    ax.scatter(
        ko_freq[ok],
        iso_sd[ok],
        s=3,
        facecolor=PLOT_PALETTE[1],  # red -- default single hue for supporting panels
        edgecolor="none",
        alpha=0.3,
    )
    ax.set_xlabel("KO-response frequency (% of Kemmeren KOs where gene is DE)")
    ax.set_ylabel("natural-isolate variability (SD of log2 across Caudal)")
    ax.set_title(
        f"Do KOs and isolates move the same genes?  r = {r:.2f}",
        loc="left",
        color=INK,
        fontsize=7,
    )
    _box(ax)
    _save(fig, "comparison_overlap_response")
    return {"n_genes": int(ok.sum()), "r": r}


# --------------------------------------------------- further explanation (2) regulatory
def panel_regulatory_divergence():
    """Nucleotide diversity by region: regulatory (up/downstream) vs coding, across the
    1,011 natural isolates. Motivates encoding the promoter/terminator window.
    """
    reg = pd.read_parquet(osp.join(RESULTS, "regulatory_divergence_by_region.parquet"))
    order_reg = ["cds", "upstream_1000", "downstream_297", "intergenic_other"]
    labs = ["CDS", "upstream\n(1000 bp)", "downstream\n(297 bp)", "intergenic"]
    pis = reg.set_index("region").loc[order_reg, "pi_percent"].to_numpy()

    _apply_rc()
    w = mm_to_in(PANEL_WIDTHS_MM["half_plus"])
    fig, ax = plt.subplots(figsize=(w, mm_to_in(60)), constrained_layout=True)
    ax.bar(
        range(len(order_reg)),
        pis,
        color=PLOT_PALETTE[1],  # red -- default single hue for supporting panels
        edgecolor=INK,
        linewidth=0.5,
        width=0.7,
    )
    ax.set_xticks(range(len(order_reg)))
    ax.set_xticklabels(labs)
    ax.set_ylabel("nucleotide diversity π (%)")
    ax.set_title(
        "Regulatory sequence is ~2× as variable as coding",
        loc="left",
        color=INK,
        fontsize=7,
    )
    _box(ax)
    _save(fig, "comparison_regulatory_divergence")
    return dict(zip(order_reg, [float(x) for x in pis]))


# --------------------------------------------------- further explanation (3) decoupling
def panel_decoupling():
    """Genome divergence does not predict transcriptome response (one point per isolate);
    natural isolates teach WHICH genes move, not HOW MANY.
    """
    div = pd.read_parquet(osp.join(RESULTS, "per_strain_divergence_summary.parquet"))
    de = pd.read_parquet(osp.join(RESULTS, "de_counts_per_strain.parquet"))
    cau = de[de["dataset"].str.startswith("caudal")][["strain", "n_de_paper_exact"]]
    m = cau.merge(div[["strain", "genome_wide_divergence"]], on="strain", how="inner")
    m["pct"] = 100.0 * m["genome_wide_divergence"]
    r = float(np.corrcoef(m["pct"], m["n_de_paper_exact"])[0, 1])

    _apply_rc()
    w = mm_to_in(PANEL_WIDTHS_MM["half_plus"])
    fig, ax = plt.subplots(figsize=(w, mm_to_in(66)), constrained_layout=True)
    ax.scatter(
        m["pct"],
        m["n_de_paper_exact"],
        s=6,
        facecolor=PLOT_PALETTE[1],  # red -- default single hue for supporting panels
        edgecolor="none",
        alpha=0.5,
    )
    ax.set_xlabel("genome-wide divergence from S288C (%)")
    ax.set_ylabel("differentially expressed genes")
    ax.set_title(
        f"Genome divergence poorly predicts response  (r = {r:.2f})",
        loc="left",
        color=INK,
        fontsize=7,
    )
    _box(ax)
    _save(fig, "comparison_decoupling")
    return {"r": r, "n": int(len(m))}


def main():
    logger.info("=" * 80)
    sample = int(os.getenv("SAMPLE_SIZE", "0")) or None
    # Inject a genome so the Kemmeren loader can resolve alias-only KO names (CDK8,
    # MED*, ATG*, ...) to systematic ids via the shared R64 reconciler; without it the
    # loader silently drops those 34 strains (1,484 -> 1,450). Cached roots avoid the
    # go.obo re-download (upstream 403s).
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    # process_workers>0: the sequential build path silently loses the LMDB write for
    # this loader (documented); the parallel path materialises processed/lmdb reliably.
    kem = MicroarrayKemmeren2014Dataset(
        root=osp.join(DATA_ROOT, "data/torchcell/microarray_kemmeren2014"),
        io_workers=0,
        process_workers=8,
        genome=genome,
    )
    sm = SmMicroarraySameith2015Dataset(
        root=osp.join(DATA_ROOT, "data/torchcell/sm_microarray_sameith2015"),
        io_workers=0,
    )
    dm = DmMicroarraySameith2015Dataset(
        root=osp.join(DATA_ROOT, "data/torchcell/dm_microarray_sameith2015"),
        io_workers=0,
    )
    cau = CaudalPanTranscriptome2024Dataset(
        root=osp.join(DATA_ROOT, "data/torchcell/caudal_pantranscriptome2024"),
        io_workers=0,
    )

    log2 = {
        "kemmeren_single": _log2_from_ratio(kem, "Kemmeren", sample),
        "sameith_single": _log2_from_ratio(sm, "Sameith-SM", sample),
        "sameith_double": _log2_from_ratio(dm, "Sameith-DM", sample),
        "caudal": _log2_from_caudal(cau, sample),
    }
    frames = {k: _quantile_frame(v) for k, v in log2.items()}
    sameith_de = _sameith_dm_de_counts(dm, sample)

    summary = {
        "panel_b": panel_b_genotype_divergence(),
        "panel_c": panel_c_spread_by_dataset(frames),
        "panel_d": panel_d_transcriptome_bands(frames),
        "panel_d2": panel_d2_size_aligned(frames),
        "panel_e": panel_e_de_counts(sameith_de),
        "panel_f": panel_f_expression_embedding(log2),
        "overlap": panel_overlap_response(log2),
        "regulatory": panel_regulatory_divergence(),
        "decoupling": panel_decoupling(),
    }
    os.makedirs(RESULTS, exist_ok=True)
    pd.Series(summary, dtype=object).to_json(
        osp.join(RESULTS, "dataset_comparison_summary.json"), indent=2
    )
    logger.info(f"summary: {summary}")
    logger.info("✓ COMPLETE")


if __name__ == "__main__":
    main()
