# experiments/019-simb-multimodal/scripts/proteome_expression_eda.py
# [[experiments.019-simb-multimodal.proteome-expression-eda]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/proteome_expression_eda

"""Proteome <-> expression EDA for the SIMB-2026 Fig-3 proteome strand.

Question (plan proteome strand, steps a/b): before spending GPU on joint
proteome + expression training, does the Messner YKO proteome carry signal
ORTHOGONAL to the Kemmeren YKO expression, and how well does one LINEARLY map
to the other? Both are single-gene-deletion collections measured on the same
S288C background, so a strain is identified by its perturbed systematic gene and
the two modalities can be aligned strain-by-strain and gene-by-gene.

Data (served DB, read-only):

* ``ProteomeMessner2023Dataset`` -- ``protein_abundance`` = {systematic_gene:
  abundance}; ~4.7k single-KO strains; media **SM** (synthetic minimal), 30 C.
  The loader already maps UniProt -> systematic ORF, so protein ids are
  systematic gene names (no extra mapping needed here).
* ``MicroarrayKemmeren2014Dataset`` -- ``expression_log2_ratio`` = {systematic_
  gene: log2(deletion/WT)}; 1484 single-KO strains; media **SC** (synthetic
  complete), 30 C.

Alignment: KO strain = the single perturbed systematic gene; measured gene =
systematic gene the value is keyed on. We intersect KO strains AND measured
genes, build two strains x genes matrices, then:

1. Overlap census (# shared KO strains; # shared measured genes).
2. Per-gene mRNA<->protein correlation ACROSS strains (z-scored per gene first,
   because proteome is absolute abundance and expression is a log2-ratio):
   distribution (median, IQR, % > 0.3).
3. Per-strain correlation across shared genes (z-scored per gene first).
4. Linear map proteome->expression AND expression->proteome (ridge, held-out
   R^2, per-gene standardized) vs a predict-per-gene-mean baseline.
5. Plots (ASSET_IMAGES_DIR, timestamped png+svg): per-gene correlation
   histogram; representative-gene scatters; linear-map R^2 summary.

MEDIA CONFOUND: proteome is SM, expression is SC. The two modalities are
measured under DIFFERENT nutrient conditions, so a KO can perturb protein and
mRNA of the same gene through partly condition-specific routes. This
ATTENUATES any observed mRNA<->protein agreement and caps what a "proteome helps
expression" claim can assert -- the verdict is written with this in mind.
"""

from __future__ import annotations

import json
import os
import os.path as osp
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.ticker import MultipleLocator
from neo4j import Driver, GraphDatabase
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

from torchcell.timestamp import timestamp
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    savefig_true_size_svg,
)

NEO4J_URI = "neo4j+s://torchcell-database.ncsa.illinois.edu:7687"
NEO4J_AUTH = ("readonly", "ReadOnly")
NEO4J_DB = "torchcell"

PROTEOME = ("ProteomeMessner2023Dataset", "ProteinAbundancePhenotype", "protein_abundance")
EXPRESSION = (
    "MicroarrayKemmeren2014Dataset",
    "MicroarrayExpressionPhenotype",
    "expression_log2_ratio",
)

# Regression: keep genes with >= this fraction of shared strains measured, then
# impute the residual (proteome) gaps with the per-gene mean (0 after z-scoring).
COVERAGE_THR = 0.9
# Correlation: a shared gene needs at least this many strains measured in BOTH
# modalities to get a stable across-strain correlation.
MIN_PAIR_STRAINS = 30
CORR_STRONG = 0.3
SEED = 42
INK = "#000000"
GRID = "#4A4A4A"

load_dotenv()
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
IMG_DIR = osp.join(ASSET_IMAGES_DIR, "019-simb-multimodal")


# --------------------------------------------------------------------------- data
def pull_dataset(
    driver: Driver, dataset_id: str, phenotype_label: str, field: str
) -> dict[str, dict[str, float]]:
    """Pull {KO systematic gene -> {measured gene -> value}} for a single-KO dataset.

    ``systematic_gene_names`` on the Genotype node is a plain string for a
    single-KO strain; we keep only single-gene genotypes (no comma/space) so the
    strain is unambiguously identified by one perturbed gene. The phenotype
    vector is stored as a JSON string on the phenotype node.
    """
    query = f"""
    MATCH (d:Dataset {{id:$id}})<-[:ExperimentMemberOf]-(e:Experiment)
    MATCH (e)<-[:GenotypeMemberOf]-(g:Genotype)
    MATCH (e)<-[:PhenotypeMemberOf]-(p:{phenotype_label})
    RETURN g.systematic_gene_names AS ko, p.{field} AS vec
    """
    out: dict[str, dict[str, float]] = {}
    with driver.session(database=NEO4J_DB) as session:
        for record in session.run(query, id=dataset_id):
            ko = record["ko"]
            if not isinstance(ko, str) or "," in ko or " " in ko:
                continue
            out.setdefault(ko, json.loads(record["vec"]))
    return out


def build_matrices(
    proteome: dict[str, dict[str, float]],
    expression: dict[str, dict[str, float]],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return (P, E) strains x genes frames over shared KO strains and shared genes.

    P = proteome (Messner, SM); E = expression (Kemmeren, SC). Rows are the KO
    strains measured in BOTH; columns are the systematic genes measured in BOTH
    (proteins mapping to expression genes). Cells are NaN where unmeasured.
    """
    shared_strains = sorted(set(proteome) & set(expression))
    proteome_genes = set().union(*(set(proteome[s]) for s in shared_strains))
    expression_genes = set().union(*(set(expression[s]) for s in shared_strains))
    shared_genes = sorted(proteome_genes & expression_genes)
    p_frame = (
        pd.DataFrame({s: pd.Series(proteome[s]) for s in shared_strains})
        .T.reindex(index=shared_strains, columns=shared_genes)
    )
    e_frame = (
        pd.DataFrame({s: pd.Series(expression[s]) for s in shared_strains})
        .T.reindex(index=shared_strains, columns=shared_genes)
    )
    return p_frame, e_frame, shared_genes


def zscore_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Z-score each gene (column) across strains, ignoring NaN (ddof=0)."""
    mean = frame.mean(axis=0, skipna=True)
    std = frame.std(axis=0, skipna=True, ddof=0)
    std = std.replace(0.0, np.nan)  # constant genes -> all-NaN (excluded downstream)
    return (frame - mean) / std


# ----------------------------------------------------------------- correlations
def per_gene_correlations(
    p_z: pd.DataFrame, e_z: pd.DataFrame, min_strains: int
) -> pd.DataFrame:
    """Across-strain Pearson r between mRNA and protein for each shared gene."""
    records: list[dict[str, Any]] = []
    for gene in p_z.columns:
        pair = pd.concat([p_z[gene], e_z[gene]], axis=1, keys=["protein", "mrna"]).dropna()
        if len(pair) < min_strains:
            continue
        if pair["protein"].std() == 0 or pair["mrna"].std() == 0:
            continue
        r = float(pair["protein"].corr(pair["mrna"]))
        records.append({"gene": gene, "n_strains": len(pair), "r": r})
    return pd.DataFrame(records).set_index("gene")


def per_strain_correlations(p_z: pd.DataFrame, e_z: pd.DataFrame) -> pd.DataFrame:
    """Across-gene Pearson r between mRNA and protein for each shared strain."""
    records: list[dict[str, Any]] = []
    for strain in p_z.index:
        pair = pd.concat(
            [p_z.loc[strain], e_z.loc[strain]], axis=1, keys=["protein", "mrna"]
        ).dropna()
        if len(pair) < MIN_PAIR_STRAINS:
            continue
        if pair["protein"].std() == 0 or pair["mrna"].std() == 0:
            continue
        r = float(pair["protein"].corr(pair["mrna"]))
        records.append({"strain": strain, "n_genes": len(pair), "r": r})
    return pd.DataFrame(records).set_index("strain")


def _distribution(values: np.ndarray) -> dict[str, float]:
    """Summarize a 1-D array of correlations."""
    return {
        "n": int(values.size),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
        "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
        "min": float(values.min()),
        "max": float(values.max()),
        "frac_gt_0.3": float(np.mean(values > CORR_STRONG)),
        "frac_lt_0": float(np.mean(values < 0.0)),
    }


# --------------------------------------------------------------------- linear map
def _standardize_train_test(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Per-gene z-score using TRAIN stats; impute residual NaN with 0 (train mean)."""
    mean = train.mean(axis=0, skipna=True)
    std = train.std(axis=0, skipna=True, ddof=0).replace(0.0, np.nan)
    tr = ((train - mean) / std).fillna(0.0).to_numpy()
    te = ((test - mean) / std).fillna(0.0).to_numpy()
    return tr, te


def linear_map(
    source: pd.DataFrame, target: pd.DataFrame, name: str
) -> dict[str, Any]:
    """Fit ridge source->target on a train split; report held-out R^2 + baseline.

    Both matrices are per-gene standardized on TRAIN strains (proteome is
    absolute abundance, expression is log2-ratio), so the baseline "predict the
    per-gene mean" corresponds to predicting 0 and its held-out R^2 is ~0 by
    construction -- the ridge R^2 above it is the linear signal one modality
    carries about the other.
    """
    rng = np.random.default_rng(SEED)
    strains = source.index.to_numpy()
    order = rng.permutation(len(strains))
    n_test = int(round(0.2 * len(strains)))
    test_idx = strains[order[:n_test]]
    train_idx = strains[order[n_test:]]

    x_train, x_test = _standardize_train_test(
        source.loc[train_idx], source.loc[test_idx]
    )
    y_train, y_test = _standardize_train_test(
        target.loc[train_idx], target.loc[test_idx]
    )

    model = RidgeCV(alphas=(1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0))
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    r2_uniform = float(r2_score(y_test, y_pred, multioutput="uniform_average"))
    r2_weighted = float(r2_score(y_test, y_pred, multioutput="variance_weighted"))
    per_gene = r2_score(y_test, y_pred, multioutput="raw_values")

    baseline_pred = np.zeros_like(y_test)  # per-gene train mean == 0 after z-score
    base_uniform = float(r2_score(y_test, baseline_pred, multioutput="uniform_average"))

    return {
        "name": name,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_features": int(source.shape[1]),
        "n_targets": int(target.shape[1]),
        "alpha": float(model.alpha_),
        "r2_uniform_average": r2_uniform,
        "r2_variance_weighted": r2_weighted,
        "r2_per_gene_median": float(np.median(per_gene)),
        "r2_per_gene_frac_positive": float(np.mean(per_gene > 0)),
        "baseline_r2_uniform_average": base_uniform,
    }


# --------------------------------------------------------------------------- plots
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
            "savefig.bbox": None,
        }
    )


def _box(ax: Any) -> None:
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(0.5)
    ax.tick_params(colors=INK, width=0.5, length=2)
    ax.grid(True, alpha=0.15, linewidth=0.4, color=GRID)
    ax.set_axisbelow(True)


def _save(fig: Any, name: str, stamp: str) -> tuple[str, str]:
    os.makedirs(IMG_DIR, exist_ok=True)
    png = osp.join(IMG_DIR, f"{name}_{stamp}.png")
    svg = osp.join(IMG_DIR, f"{name}_{stamp}.svg")
    fig.savefig(png, dpi=300, facecolor="white")
    savefig_true_size_svg(fig, svg, facecolor="white")
    plt.close(fig)
    return png, svg


def plot_gene_hist(gene_corr: pd.DataFrame, stamp: str) -> tuple[str, str]:
    """Histogram of the per-gene across-strain mRNA<->protein correlations."""
    r = gene_corr["r"].to_numpy(dtype=float)
    fig, ax = plt.subplots(
        figsize=(mm_to_in(PANEL_WIDTHS_MM["half"]), mm_to_in(55.0))
    )
    ax.hist(r, bins=np.arange(-0.6, 0.9001, 0.05).tolist(), color=PLOT_PALETTE[0],
            edgecolor=INK, linewidth=0.4)
    med = float(np.median(r))
    ax.axvline(0.0, color=INK, linewidth=0.6, linestyle="--")
    ax.axvline(med, color=PLOT_PALETTE[1], linewidth=0.8)
    ax.set_xlabel("per-gene mRNA–protein correlation (across strains)")
    ax.set_ylabel("number of genes")
    ax.set_title(
        f"n={len(r)} genes | median r={med:.2f} | "
        f"{100 * np.mean(r > CORR_STRONG):.0f}% > {CORR_STRONG}"
    )
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which="minor", length=0)
    _box(ax)
    fig.tight_layout()
    return _save(fig, "proteome_expression_per_gene_corr_hist", stamp)


def plot_representative_scatters(
    p_z: pd.DataFrame, e_z: pd.DataFrame, gene_corr: pd.DataFrame, stamp: str
) -> tuple[str, str, list[str]]:
    """Scatter z-scored mRNA vs protein across strains for representative genes."""
    ranked = gene_corr.sort_values("r")
    hi = ranked.index[-1]
    lo = ranked.index[0]
    med_gene = ranked.index[len(ranked) // 2]
    q75_gene = ranked.index[int(0.75 * len(ranked))]
    genes = [hi, q75_gene, med_gene, lo]
    fig, axes = plt.subplots(
        1, 4, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(48.0))
    )
    for ax, gene in zip(axes, genes):
        pair = pd.concat(
            [p_z[gene], e_z[gene]], axis=1, keys=["protein", "mrna"]
        ).dropna()
        ax.scatter(
            pair["protein"], pair["mrna"], s=3, color=PLOT_PALETTE[0],
            edgecolor="none", alpha=0.6,
        )
        r = float(gene_corr.loc[gene, "r"])
        ax.set_title(f"{gene}\nr={r:.2f} (n={len(pair)})")
        ax.set_xlabel("protein (z)")
        ax.set_ylabel("mRNA (z)")
        ax.axhline(0.0, color=GRID, linewidth=0.3)
        ax.axvline(0.0, color=GRID, linewidth=0.3)
        _box(ax)
    fig.tight_layout()
    png, svg = _save(fig, "proteome_expression_representative_scatters", stamp)
    return png, svg, list(genes)


def plot_r2_summary(maps: list[dict[str, Any]], stamp: str) -> tuple[str, str]:
    """Bar chart of held-out R^2 for each linear map vs its baseline."""
    fig, ax = plt.subplots(
        figsize=(mm_to_in(PANEL_WIDTHS_MM["half"]), mm_to_in(55.0))
    )
    labels = [m["name"] for m in maps]
    x = np.arange(len(labels))
    width = 0.38
    ridge = [m["r2_uniform_average"] for m in maps]
    base = [m["baseline_r2_uniform_average"] for m in maps]
    ax.bar(x - width / 2, ridge, width, color=PLOT_PALETTE[0], edgecolor=INK,
           linewidth=0.4, label="ridge")
    ax.bar(x + width / 2, base, width, color=PLOT_PALETTE[5], edgecolor=INK,
           linewidth=0.4, label="baseline (per-gene mean)")
    for xi, val in zip(x - width / 2, ridge):
        ax.text(xi, val + 0.005, f"{val:.3f}", ha="center", va="bottom", fontsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.set_ylabel("held-out $R^2$ (uniform avg over genes)")
    ax.axhline(0.0, color=INK, linewidth=0.5)
    ax.legend(frameon=False, loc="upper right")
    _box(ax)
    fig.tight_layout()
    return _save(fig, "proteome_expression_linear_map_r2", stamp)


# ---------------------------------------------------------------------------- main
def main() -> None:
    here = osp.dirname(osp.abspath(__file__))
    results_dir = osp.abspath(osp.join(here, "..", "results"))
    os.makedirs(results_dir, exist_ok=True)
    _apply_rc()
    stamp = timestamp()

    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    proteome = pull_dataset(driver, *PROTEOME)
    expression = pull_dataset(driver, *EXPRESSION)
    driver.close()

    p_frame, e_frame, shared_genes = build_matrices(proteome, expression)
    shared_strains = list(p_frame.index)

    proteome_gene_union = len(set().union(*(set(v) for v in proteome.values())))
    expression_gene_union = len(set().union(*(set(v) for v in expression.values())))

    # coverage-filtered gene set for the (dense) linear map
    coverage = p_frame.notna().mean(axis=0)  # expression is essentially complete
    dense_genes = [g for g in shared_genes if coverage[g] >= COVERAGE_THR]

    overlap = {
        "proteome_unique_ko_strains": len(proteome),
        "expression_unique_ko_strains": len(expression),
        "shared_ko_strains": len(shared_strains),
        "proteome_measured_genes_union": proteome_gene_union,
        "expression_measured_genes_union": expression_gene_union,
        "shared_measured_genes": len(shared_genes),
        "linear_map_gene_coverage_threshold": COVERAGE_THR,
        "linear_map_dense_genes": len(dense_genes),
        "proteome_media": "SM",
        "expression_media": "SC",
    }
    print("=" * 72)
    print("OVERLAP")
    for key, value in overlap.items():
        print(f"  {key}: {value}")

    # z-scored (descriptive) matrices for correlations
    p_z = zscore_columns(p_frame)
    e_z = zscore_columns(e_frame)

    gene_corr = per_gene_correlations(p_z, e_z, MIN_PAIR_STRAINS)
    strain_corr = per_strain_correlations(p_z, e_z)
    gene_dist = _distribution(gene_corr["r"].to_numpy())
    strain_dist = _distribution(strain_corr["r"].to_numpy())

    print("\nPER-GENE across-strain correlation distribution")
    for key, value in gene_dist.items():
        print(f"  {key}: {value}")
    print("\nPER-STRAIN across-gene correlation distribution")
    for key, value in strain_dist.items():
        print(f"  {key}: {value}")

    # linear maps on the dense gene set
    p_dense = p_frame[dense_genes]
    e_dense = e_frame[dense_genes]
    map_p2e = linear_map(p_dense, e_dense, "proteome → expression")
    map_e2p = linear_map(e_dense, p_dense, "expression → proteome")
    print("\nLINEAR MAP (held-out)")
    for m in (map_p2e, map_e2p):
        print(
            f"  {m['name']}: R2_uniform={m['r2_uniform_average']:.3f} "
            f"R2_weighted={m['r2_variance_weighted']:.3f} "
            f"baseline={m['baseline_r2_uniform_average']:.3f} alpha={m['alpha']:.0f}"
        )

    # plots
    hist_png, hist_svg = plot_gene_hist(gene_corr, stamp)
    scat_png, scat_svg, rep_genes = plot_representative_scatters(
        p_z, e_z, gene_corr, stamp
    )
    r2_png, r2_svg = plot_r2_summary([map_p2e, map_e2p], stamp)

    report = {
        "generated": stamp,
        "sources": {
            "proteome": {"dataset": PROTEOME[0], "media": "SM"},
            "expression": {"dataset": EXPRESSION[0], "media": "SC"},
        },
        "overlap": overlap,
        "per_gene_correlation": gene_dist,
        "per_strain_correlation": strain_dist,
        "linear_map": {"proteome_to_expression": map_p2e, "expression_to_proteome": map_e2p},
        "representative_genes": rep_genes,
        "figures": {
            "per_gene_corr_hist": {"png": hist_png, "svg": hist_svg},
            "representative_scatters": {"png": scat_png, "svg": scat_svg},
            "linear_map_r2": {"png": r2_png, "svg": r2_svg},
        },
    }
    out_path = osp.join(results_dir, "proteome_expression_eda.json")
    with open(out_path, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nWrote {out_path}")
    print(f"Figures in {IMG_DIR}")


if __name__ == "__main__":
    main()
