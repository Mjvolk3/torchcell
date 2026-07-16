# experiments/018-natural-isolate-genomics/scripts/dataset_comparison.py
# [[experiments.018-natural-isolate-genomics.dataset-comparison]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/018-natural-isolate-genomics/scripts/dataset_comparison

"""Fig 4 descriptive panels: how different are engineered-KO and natural-isolate strains?

One panel, one question (see [[experiments.018-natural-isolate-genomics.expression-modeling-setup]]):

- **b** genotype divergence — # reference ORFs absent (x) vs % sequence divergence on
  shared genes (y). Engineered KOs sit at (1 gene, 0 bp); natural isolates spread far out.
  The two occupy disjoint genotype design spaces.
- **d** transcriptome spread — Kemmeren single-KO / Sameith single-KO / Sameith double-KO /
  Caudal natural-isolate as matched sorted spread bands on ONE shared y-scale.
- **e** how many genes move — per-strain DE-count distribution, KO (Kemmeren) vs natural
  (Caudal), on the paper-exact rule.
- **f** transcriptome design-space coverage — PCA and UMAP of the joint expression matrix,
  coloured by dataset (per-gene z-scored WITHIN dataset to blunt the microarray/RNA-seq
  platform confound; residual separation is partly platform, stated on the panel).

Palette per dataset, consistent across panels: Kemmeren = orange, Caudal = red (the focus),
Sameith single = purple, Sameith double = yellow. Repo figure standard throughout
(Arial 6 pt, boxed, true-size SVG + PNG). Genotype inputs (b) come from the 018 result
parquets; expression (d, e, f) from the built LMDBs.
"""

import logging
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CAUDAL_PSEUDOCOUNT = 1.0  # matches differential_expression_comparison.py:127
INK = "#000000"
GRID = "#4A4A4A"
PANEL_H_MM = 35.7  # canonical wide-strip height unit (figure-sizing-template.drawio.svg)

# per-dataset colours, consistent across every panel (line, fill); green-free, warm first.
DS = {
    "kemmeren_single": {"label": "Kemmeren single KO", "line": PLOT_PALETTE[0], "fill": PLOT_PALETTE_FILL[0]},
    "caudal": {"label": "Caudal natural isolate", "line": PLOT_PALETTE[1], "fill": PLOT_PALETTE_FILL[1]},
    "sameith_single": {"label": "Sameith single KO", "line": PLOT_PALETTE[2], "fill": PLOT_PALETTE_FILL[2]},
    "sameith_double": {"label": "Sameith double KO", "line": PLOT_PALETTE[3], "fill": PLOT_PALETTE_FILL[3]},
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
            g: float(np.log2((tpm[g] + CAUDAL_PSEUDOCOUNT) / (ref[g] + CAUDAL_PSEUDOCOUNT)))
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
        rows.append({"strain": strain, "p5": p5, "q1": q1, "median": med, "q3": q3, "p95": p95, "iqr": q3 - q1})
    return pd.DataFrame(rows).sort_values("iqr").reset_index(drop=True)


# ----------------------------------------------------------------------------- panel b
def panel_b_genotype_divergence():
    """# reference ORFs absent vs % sequence divergence; KO at (1, 0), isolates far out."""
    burden = pd.read_parquet(osp.join(RESULTS, "natural_ko_burden.parquet"))
    div = pd.read_parquet(osp.join(RESULTS, "per_strain_divergence_summary.parquet"))
    iso = burden.merge(div[["strain", "genome_wide_divergence"]], on="strain", how="inner")
    iso["pct"] = 100.0 * iso["genome_wide_divergence"]

    _apply_rc()
    w = mm_to_in(PANEL_WIDTHS_MM["half_plus"])
    fig, ax = plt.subplots(figsize=(w, mm_to_in(66)), constrained_layout=True)
    ax.scatter(
        iso["n_absent"].clip(lower=0.5), iso["pct"], s=5, marker="o",
        facecolor=DS["caudal"]["line"], edgecolor="none", alpha=0.55, label=DS["caudal"]["label"],
    )
    # engineered KO: 1 gene fully removed, 0 bp change on the rest of the genome.
    ax.scatter([1], [0.001], s=34, marker="D", facecolor=DS["kemmeren_single"]["line"],
               edgecolor=INK, linewidth=0.5, zorder=5, label="engineered KO (Kemmeren / Sameith)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("reference ORFs absent (count)")
    ax.set_ylabel("sequence divergence on shared genes (%)")
    ax.set_title("Genotype: engineered vs natural occupy disjoint space", loc="left", color=INK, fontsize=7)
    ax.legend(loc="lower right", frameon=False, handletextpad=0.4, borderpad=0.3)
    _box(ax)
    _save(fig, "comparison_b_genotype_divergence")
    return {"n_isolates": int(len(iso)), "median_absent": float(iso["n_absent"].median()),
            "median_pct_div": float(iso["pct"].median())}


# ----------------------------------------------------------------------------- panel d
def panel_d_transcriptome_bands(frames):
    """Four datasets as matched sorted spread bands on one shared y-scale."""
    _apply_rc()
    order = ["kemmeren_single", "sameith_single", "sameith_double", "caudal"]
    ymax = float(np.ceil(max(max(f["p95"].abs().max(), f["p5"].abs().max()) for f in frames.values()) * 10) / 10)
    fig, axes = plt.subplots(
        len(order), 1, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(4 * PANEL_H_MM)),
        constrained_layout=True, sharex=False,
    )
    for ax, key in zip(axes, order):
        df = frames[key]
        x = np.arange(len(df))
        ax.fill_between(x, df["p5"], df["p95"], facecolor=DS[key]["fill"], linewidth=0, zorder=2)
        ax.fill_between(x, df["q1"], df["q3"], facecolor=DS[key]["line"], linewidth=0, zorder=3)
        ax.plot(x, df["median"], color=INK, linewidth=0.5, zorder=4)
        ax.set_ylim(-ymax, ymax)
        ax.axhline(0, color=GRID, linestyle=":", linewidth=0.5, zorder=1)
        ax.set_xlim(0, max(len(df) - 1, 1))
        ax.set_ylabel("log2 FC")
        ax.set_title(f"{DS[key]['label']}  (n = {len(df)})", loc="left", color=INK, fontsize=6)
        _box(ax)
    axes[-1].set_xlabel("deletion / isolate strains, ranked by transcriptome spread")
    _save(fig, "comparison_d_transcriptome_bands")
    return {"shared_ymax": ymax, **{k: int(len(v)) for k, v in frames.items()}}


# ----------------------------------------------------------------------------- panel e
def panel_e_de_counts():
    """Per-strain DE-count distribution, KO (Kemmeren) vs natural (Caudal)."""
    de = pd.read_parquet(osp.join(RESULTS, "de_counts_per_strain.parquet"))
    kem = de[de["dataset"] == "kemmeren2014_single_ko"]["n_de_paper_exact"].to_numpy()
    cau = de[de["dataset"].str.startswith("caudal")]["n_de_paper_exact"].to_numpy()

    _apply_rc()
    w = mm_to_in(PANEL_WIDTHS_MM["half_plus"])
    fig, ax = plt.subplots(figsize=(w, mm_to_in(60)), constrained_layout=True)
    bins = np.logspace(0, np.log10(max(kem.max(), cau.max()) + 1), 40)
    ax.hist(np.clip(kem, 1, None), bins=bins, color=DS["kemmeren_single"]["line"],
            edgecolor=INK, linewidth=0.2, alpha=0.75, label=f"Kemmeren single KO (median {int(np.median(kem))})")
    ax.hist(np.clip(cau, 1, None), bins=bins, color=DS["caudal"]["line"],
            edgecolor=INK, linewidth=0.2, alpha=0.75, label=f"Caudal isolate (median {int(np.median(cau))})")
    ax.set_xscale("log")
    ax.set_xlabel("differentially expressed genes per strain")
    ax.set_ylabel("strains")
    ax.set_title("A single KO moves few genes; an isolate moves many", loc="left", color=INK, fontsize=7)
    ax.legend(loc="upper right", frameon=False, handletextpad=0.4, borderpad=0.3)
    _box(ax)
    _save(fig, "comparison_e_de_counts")
    return {"kemmeren_median": float(np.median(kem)), "caudal_median": float(np.median(cau))}


# ----------------------------------------------------------------------------- panel f
def panel_f_expression_embedding(log2_by_dataset):
    """PCA + UMAP of the joint expression matrix, per-gene z-scored within dataset."""
    order = ["kemmeren_single", "sameith_single", "sameith_double", "caudal"]
    shared = set.intersection(*[
        set.union(*[set(p) for p in log2_by_dataset[k].values()]) for k in order
    ])
    genes = sorted(shared)
    logger.info(f"Panel f: {len(genes)} shared genes across all four datasets")

    mats, labels = [], []
    for key in order:
        rows = [[prof.get(g, np.nan) for g in genes] for prof in log2_by_dataset[key].values()]
        m = np.asarray(rows, dtype=float)
        m = np.where(np.isnan(m), np.nanmean(m, axis=0, keepdims=True), m)  # gene-mean impute
        m = StandardScaler().fit_transform(m)  # per-gene z-score WITHIN dataset (confound guard)
        mats.append(m)
        labels += [key] * m.shape[0]
    X = np.vstack(mats)
    labels = np.asarray(labels)

    pca = PCA(n_components=2, random_state=0).fit(X)
    xy_pca = pca.transform(X)
    xy_umap = UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3).fit_transform(X)

    _apply_rc()
    fig, (axp, axu) = plt.subplots(
        1, 2, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(74)), constrained_layout=True
    )
    for ax, xy, ttl in [
        (axp, xy_pca, f"PCA (PC1 {pca.explained_variance_ratio_[0] * 100:.0f}%, PC2 {pca.explained_variance_ratio_[1] * 100:.0f}%)"),
        (axu, xy_umap, "UMAP"),
    ]:
        for key in order:
            m = labels == key
            ax.scatter(xy[m, 0], xy[m, 1], s=4, facecolor=DS[key]["line"], edgecolor="none",
                       alpha=0.55, label=DS[key]["label"])
        ax.set_title(ttl, loc="left", color=INK, fontsize=6)
        ax.set_xticks([])
        ax.set_yticks([])
        _box(ax)
    axp.set_xlabel("PC1")
    axp.set_ylabel("PC2")
    axu.set_xlabel("UMAP-1")
    axu.legend(loc="best", frameon=False, handletextpad=0.3, borderpad=0.3, markerscale=1.6)
    fig.suptitle(
        "Transcriptome coverage (per-gene z-scored within dataset; residual split is partly platform)",
        x=0.005, ha="left", fontsize=7, color=INK,
    )
    _save(fig, "comparison_f_expression_embedding")
    return {"n_shared_genes": len(genes), "pc1_var": float(pca.explained_variance_ratio_[0])}


def main():
    logger.info("=" * 80)
    sample = int(os.getenv("SAMPLE_SIZE", "0")) or None
    kem = MicroarrayKemmeren2014Dataset(root=osp.join(DATA_ROOT, "data/torchcell/microarray_kemmeren2014"), io_workers=0)
    sm = SmMicroarraySameith2015Dataset(root=osp.join(DATA_ROOT, "data/torchcell/sm_microarray_sameith2015"), io_workers=0)
    dm = DmMicroarraySameith2015Dataset(root=osp.join(DATA_ROOT, "data/torchcell/dm_microarray_sameith2015"), io_workers=0)
    cau = CaudalPanTranscriptome2024Dataset(root=osp.join(DATA_ROOT, "data/torchcell/caudal_pantranscriptome2024"), io_workers=0)

    log2 = {
        "kemmeren_single": _log2_from_ratio(kem, "Kemmeren", sample),
        "sameith_single": _log2_from_ratio(sm, "Sameith-SM", sample),
        "sameith_double": _log2_from_ratio(dm, "Sameith-DM", sample),
        "caudal": _log2_from_caudal(cau, sample),
    }
    frames = {k: _quantile_frame(v) for k, v in log2.items()}

    summary = {
        "panel_b": panel_b_genotype_divergence(),
        "panel_d": panel_d_transcriptome_bands(frames),
        "panel_e": panel_e_de_counts(),
        "panel_f": panel_f_expression_embedding(log2),
    }
    os.makedirs(RESULTS, exist_ok=True)
    pd.Series(summary, dtype=object).to_json(osp.join(RESULTS, "dataset_comparison_summary.json"), indent=2)
    logger.info(f"summary: {summary}")
    logger.info("✓ COMPLETE")


if __name__ == "__main__":
    main()
