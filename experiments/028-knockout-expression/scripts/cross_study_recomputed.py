# experiments/028-knockout-expression/scripts/cross_study_recomputed.py
# [[experiments.028-knockout-expression.scripts.cross_study_recomputed]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-knockout-expression/scripts/cross_study_recomputed
"""Does recomputing the Nadal-Ribelles fold change from the single-cell object recover
agreement with Kemmeren?

cross_study_ko_expression.py found that the STORED Nadal-Ribelles values (scanpy Wilcoxon
`logfoldchanges`, one cell in five a zero-mean sentinel) agree with the Kemmeren
microarrays at a per-strain median of 0.003 over 914 shared deletions, and that the
agreement does not track the number of cells behind a genotype's pseudobulk. That points
at the statistic rather than the sampling. nadal_pseudobulk_recompute.R recomputed three
statistics per (genotype, gene) from the raw UMI counts of the same cells:

  A  pseudobulk_log2fc   summed UMI -> CPM -> log2((cpm_g + 1) / (cpm_wt + 1)),
                         absent below 10 summed UMI     (bulk-like, the microarray analog)
  B  seurat_avg_log2fc   Seurat FindMarkers avg_log2FC   (per-cell mean, pseudocount 1)
  C  scanpy_logfc        scanpy's formula on the same means (should reproduce the stored
                         values; a check, not a candidate)

This script scores each against Kemmeren with exactly the measurements of
cross_study_ko_expression.py (per-strain, per-reporter, strong responders, scale), beside
the stored values, and reports whether C reproduces what the loader stored.

Gene names in the Seurat object are the paper's common names; they are resolved to R64
ORFs with the genome alias table, the same resolution the loader applies.

Run from the repo root, after the R script:
    python experiments/028-knockout-expression/scripts/cross_study_recomputed.py
"""

from __future__ import annotations

import json
import os
import os.path as osp
import re
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from cross_study_ko_expression import (  # noqa: E402
    LMDB,
    _is_control,
    _load_lmdb,
    _pair,
    _plain_log_x,
    _profiles,
)

from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome  # noqa: E402
from torchcell.utils import (  # noqa: E402
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    panel_label,
    savefig_true_size_svg,
)
from torchcell.utils.paths import experiment_results_dir  # noqa: E402

RECOMPUTED_REL = "data/torchcell/nadal_ribelles_perturbseq2025/recomputed"
STATS = {
    "pseudobulk_log2fc": "A: pseudobulk CPM ratio",
    "seurat_avg_log2fc": "B: Seurat avg_log2FC",
    "scanpy_logfc": "C: scanpy formula (check)",
}
_SYSTEMATIC_RE = re.compile(r"^Y[A-P][LR]\d{3}[WC](-[A-Z])?$")


def _resolver(genome: SCerevisiaeGenome):  # type: ignore[no-untyped-def]
    """Common name or alias -> R64 ORF, as the loader resolves the DEG tables."""
    df = genome.gene_attribute_table
    ids = set(df["ID"].astype(str))
    alias_map = genome.alias_to_systematic
    gene_col = dict(zip(df["gene"].astype(str), df["ID"].astype(str)))
    alias_col = dict(zip(df["Alias"].astype(str), df["ID"].astype(str)))

    def resolve(token: str) -> str | None:
        gene = str(token).upper()
        if gene in ids:
            return gene
        if _SYSTEMATIC_RE.match(gene):
            cand = alias_map.get(gene, [])
            if cand and cand[0] in ids:
                return cand[0]
        if gene in gene_col:
            return gene_col[gene]
        if gene in alias_col:
            return alias_col[gene]
        cand = alias_map.get(gene, [])
        if cand and cand[0] in ids:
            return cand[0]
        return None

    return resolve


def _recomputed_profiles(
    data_root: str, stat: str, resolve: Any, genotypes: pd.DataFrame
) -> tuple[dict[str, dict[str, float]], dict[str, int], dict[str, Any]]:
    """{deleted ORF: {reporter ORF: value}} for one statistic, plus cell counts."""
    path = osp.join(data_root, RECOMPUTED_REL, f"{stat}.tsv.gz")
    m = pd.read_csv(path, sep="\t", index_col=0)
    orfs = [resolve(g) for g in m.index]
    keep = [o is not None for o in orfs]
    n_unresolved = int(len(orfs) - sum(keep))
    m = m.loc[keep]
    m.index = [o for o in orfs if o is not None]
    m = m[~m.index.duplicated(keep="first")]
    geno = genotypes.dropna(subset=["kogene"]).set_index("label")
    profiles: dict[str, dict[str, float]] = {}
    cells: dict[str, int] = {}
    n_multi = 0
    for label, row in geno.iterrows():
        if label == "WT" or label not in m.columns:
            continue
        orf = str(row["kogene"])
        col = m[label].dropna()
        prof = {str(g): float(v) for g, v in col.items()}
        if orf in profiles:
            # Replacement strains deleting the same ORF: average, as the stored path does.
            n_multi += 1
            acc = profiles[orf]
            for g, v in prof.items():
                acc[g] = (acc[g] + v) / 2 if g in acc else v
            cells[orf] += int(row["n_cells"])
        else:
            profiles[orf] = prof
            cells[orf] = int(row["n_cells"])
    info = {
        "genes_in_object": int(len(orfs)),
        "genes_unresolved": n_unresolved,
        "genotypes_with_orf": int(len(geno)),
        "orfs": len(profiles),
        "orfs_with_replacement_strain": n_multi,
    }
    return profiles, cells, info


def _summary(p: dict[str, Any]) -> dict[str, Any]:
    return {
        "n_shared_deletions": p["n_shared_deletions"],
        "n_shared_reporters": p["n_shared_reporters"],
        "per_strain_median_r": p["per_strain"]["median_r"],
        "per_strain_iqr": p["per_strain"]["iqr"],
        "per_reporter_median_r": p["per_reporter"]["median_r"],
        "per_reporter_mean_sqrt_r_ceiling": p["per_reporter"]["mean_sqrt_r_ceiling"],
        "strong_median_r": p["strong_responders"]["median_r"],
        "strong_sign_agreement": p["strong_responders"]["median_sign_agreement"],
        "ols_slope_kemmeren_on_nadal": p["scale"]["ols_slope_a_on_b"],
        "cell_pearson": p["scale"]["cell_pearson"],
        "sd_nadal": p["scale"]["sd_b"],
    }


def main() -> None:
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    images = osp.join(os.environ["ASSET_IMAGES_DIR"], "028-knockout-expression")
    os.makedirs(images, exist_ok=True)
    results_dir = experiment_results_dir("028-knockout-expression", __file__)

    genome = SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
        overwrite=False,
    )
    resolve = _resolver(genome)
    genotypes = pd.read_csv(
        osp.join(data_root, RECOMPUTED_REL, "genotypes.tsv"), sep="\t"
    )

    kem_records = _load_lmdb(osp.join(data_root, LMDB["kemmeren"]))
    kem, _, _ = _profiles(kem_records)
    nad_records = [
        r for r in _load_lmdb(osp.join(data_root, LMDB["nadal"])) if _is_control(r)
    ]
    stored, _, _ = _profiles(nad_records)  # sentinels already dropped by _profiles
    # The stored values BEFORE the sentinel cut, for the reproduction check of C.
    stored_raw: dict[str, dict[str, float]] = {}
    for r in nad_records:
        perts = r["experiment"]["genotype"]["perturbations"]
        if len(perts) == 1:
            stored_raw.setdefault(
                perts[0]["systematic_gene_name"],
                r["experiment"]["phenotype"]["expression_log2_ratio"],
            )

    versions: dict[str, dict[str, dict[str, float]]] = {"stored": stored}
    cells: dict[str, int] = {}
    info: dict[str, Any] = {}
    for stat in STATS:
        versions[stat], cells, info[stat] = _recomputed_profiles(
            data_root, stat, resolve, genotypes
        )
        print(f"{stat}: {info[stat]}")

    # Reproduction check: does C match the stored values cell by cell?
    xs, ys = [], []
    for orf, prof in versions["scanpy_logfc"].items():
        raw = stored_raw.get(orf)
        if raw is None:
            continue
        for g, v in prof.items():
            w = raw.get(g)
            if w is not None:
                xs.append(v)
                ys.append(w)
    x = np.asarray(xs)
    y = np.asarray(ys)
    both_sentinel = (
        float(((np.abs(x) >= 20) == (np.abs(y) >= 20)).mean())
        if len(x)
        else float("nan")
    )
    fin = (np.abs(x) < 20) & (np.abs(y) < 20)
    repro = {
        "n_cells_matched": int(len(x)),
        "sentinel_status_agreement": both_sentinel,
        "pearson_non_sentinel": float(np.corrcoef(x[fin], y[fin])[0, 1])
        if fin.sum() > 2
        else float("nan"),
        "median_abs_diff_non_sentinel": float(np.median(np.abs(x[fin] - y[fin])))
        if fin.sum()
        else float("nan"),
    }
    print("reproduction of the stored values by C:", repro)

    pairs = {k: _pair(kem, v) for k, v in versions.items()}
    summary = {k: _summary(p) for k, p in pairs.items()}
    for k, s in summary.items():
        print(
            f"\n== Kemmeren vs Nadal [{k}]: shared {s['n_shared_deletions']} deletions, "
            f"{s['n_shared_reporters']} reporters\n"
            f"  per-strain median r {s['per_strain_median_r']:.3f}  IQR {s['per_strain_iqr'][0]:.3f}..{s['per_strain_iqr'][1]:.3f}\n"
            f"  per-reporter median r {s['per_reporter_median_r']:.3f}  ceiling {s['per_reporter_mean_sqrt_r_ceiling']:.3f}\n"
            f"  strong responders r {s['strong_median_r']:.3f}, sign agreement {s['strong_sign_agreement']:.2f}\n"
            f"  slope {s['ols_slope_kemmeren_on_nadal']:.3f}, cell r {s['cell_pearson']:.3f}, sd {s['sd_nadal']:.3f}"
        )

    # Agreement against cell count, per version (the sampling test).
    cells_track: dict[str, float] = {}
    for k, p in pairs.items():
        r = np.asarray(p["per_strain"]["values"], dtype=float)
        n = np.asarray(
            [cells.get(o, np.nan) for o in p["per_strain"]["deletions"]], dtype=float
        )
        m = np.isfinite(r) & np.isfinite(n)
        cells_track[k] = (
            float(spearmanr(n[m], r[m]).correlation) if m.sum() > 10 else float("nan")
        )
    print(
        "Spearman(per-strain r, cells):",
        {k: round(v, 3) for k, v in cells_track.items()},
    )

    purity = pd.read_csv(
        osp.join(data_root, RECOMPUTED_REL, "assignment_purity.tsv"), sep="\t"
    )
    ok = purity["wt_frac_detecting"] >= 0.5
    ok2 = purity["wt_frac_detecting"] >= 0.2
    purity_summary = {
        "source": "experiments/028-knockout-expression/scripts/nadal_assignment_purity.R",
        "n_genotypes_scored": int(len(purity)),
        "wt_detect_ge_0.5": {
            "n": int(ok.sum()),
            "frac_own_cells_detecting_ko_median": float(
                purity.loc[ok, "frac_cells_detecting_ko"].median()
            ),
            "impurity_ratio_median": float(
                purity.loc[ok, "impurity_upper_bound"].median()
            ),
            "purity_excess_zero_median": float(
                purity.loc[ok, "purity_excess_zero"].median()
            ),
            "purity_excess_zero_iqr": [
                float(purity.loc[ok, "purity_excess_zero"].quantile(q))
                for q in (0.25, 0.75)
            ],
        },
        "wt_detect_ge_0.2": {
            "n": int(ok2.sum()),
            "purity_excess_zero_median": float(
                purity.loc[ok2, "purity_excess_zero"].median()
            ),
            "purity_excess_zero_iqr": [
                float(purity.loc[ok2, "purity_excess_zero"].quantile(q))
                for q in (0.25, 0.75)
            ],
        },
    }
    print("assignment purity:", json.dumps(purity_summary, indent=1))

    out = {
        "generated_by": "experiments/028-knockout-expression/scripts/cross_study_recomputed.py",
        "assignment_purity": purity_summary,
        "recompute_script": "experiments/028-knockout-expression/scripts/nadal_pseudobulk_recompute.R",
        "statistics": STATS,
        "resolution": info,
        "reproduction_of_stored_by_scanpy_formula": repro,
        "kemmeren_vs_nadal": summary,
        "spearman_per_strain_r_vs_cells": cells_track,
    }
    with open(osp.join(results_dir, "cross_study_recomputed.json"), "w") as f:
        json.dump(out, f, indent=1)

    # ------------------------------------------------------------------ figure
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "axes.linewidth": 0.5,
            "svg.fonttype": "none",
            "axes.titlesize": 6,
            "legend.fontsize": 6,
        }
    )
    order = ["stored", "pseudobulk_log2fc", "seurat_avg_log2fc"]
    names = {
        "stored": "stored (scanpy)",
        "pseudobulk_log2fc": "A: pseudobulk",
        "seurat_avg_log2fc": "B: Seurat",
    }
    col = dict(zip(order, [PLOT_PALETTE[1], PLOT_PALETTE[0], PLOT_PALETTE[2]]))
    legend_kw = dict(frameon=True, edgecolor="black", fancybox=False, framealpha=1.0)
    fig, axes = plt.subplots(
        2, 3, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(105))
    )

    def filled(ax, v, key, bins, label):
        ax.hist(
            v,
            bins=bins,
            histtype="stepfilled",
            facecolor=to_rgba(col[key], 0.45),
            edgecolor="black",
            lw=0.4,
            density=True,
            label=label,
        )

    # a. value distributions of the three Nadal versions against Kemmeren's.
    ax = axes[0, 0]
    bins = np.linspace(-4, 4, 81)
    kv = np.concatenate([np.fromiter(p.values(), dtype=float) for p in kem.values()])
    ax.hist(
        kv,
        bins=bins,
        histtype="step",
        color="black",
        lw=0.8,
        density=True,
        label=f"Kemmeren (sd {kv.std():.2f})",
    )
    for k in order:
        v = np.concatenate(
            [np.fromiter(p.values(), dtype=float) for p in versions[k].values()]
        )
        filled(ax, v, k, bins, f"{names[k]} (sd {v.std():.2f})")
    ax.set_yscale("log")
    # Headroom of two decades so the framed legend sits above every histogram.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 3000)
    ax.set_xlabel("log2 value")
    ax.set_ylabel("density")
    ax.set_title("value distributions")
    ax.legend(loc="upper right", **legend_kw)

    # b. per-strain agreement with Kemmeren, three versions.
    ax = axes[0, 1]
    bins = np.linspace(-0.4, 1.0, 36)
    for k in order:
        v = np.asarray(pairs[k]["per_strain"]["values"], dtype=float)
        v = v[np.isfinite(v)]
        filled(ax, v, k, bins, f"{names[k]} (med {np.median(v):.2f})")
    ax.set_xlabel("per-strain Pearson r, Kemmeren vs Nadal")
    ax.set_ylabel("density")
    ax.set_title("per-strain agreement")
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which="minor", length=0)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.6)
    ax.legend(loc="upper right", **legend_kw)

    # c. per-reporter agreement, three versions.
    ax = axes[0, 2]
    for k in order:
        v = np.asarray(pairs[k]["per_reporter"]["values"], dtype=float)
        v = v[np.isfinite(v)]
        filled(ax, v, k, bins, f"{names[k]} (med {np.median(v):.2f})")
    ax.set_xlabel("per-reporter Pearson r, Kemmeren vs Nadal")
    ax.set_ylabel("density")
    ax.set_title("per-reporter agreement (test-retest)")
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which="minor", length=0)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.6)
    ax.legend(loc="upper right", **legend_kw)

    # d. Kemmeren against A on shared cells.
    ax = axes[1, 0]
    xs, ys = [], []
    pa = versions["pseudobulk_log2fc"]
    for orf in set(kem) & set(pa):
        pb = pa[orf]
        for g, v in kem[orf].items():
            w = pb.get(g)
            if w is not None:
                xs.append(w)
                ys.append(v)
    x = np.asarray(xs)
    y = np.asarray(ys)
    hb = ax.hexbin(x, y, gridsize=60, bins="log", cmap="Greys", linewidths=0)
    lim = 4.0
    sl = pairs["pseudobulk_log2fc"]["scale"]["ols_slope_a_on_b"]
    ax.plot([-lim, lim], [-lim, lim], color="black", lw=0.5, ls="--", label="identity")
    ax.plot(
        [-lim, lim],
        [-lim * sl, lim * sl],
        color=PLOT_PALETTE[1],
        lw=0.8,
        label=f"OLS slope {sl:.2f}",
    )
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Nadal-Ribelles A: pseudobulk log2 FC")
    ax.set_ylabel("Kemmeren log2 ratio")
    ax.set_title(
        f"shared cells (n = {len(x):,}), r = {pairs['pseudobulk_log2fc']['scale']['cell_pearson']:.2f}"
    )
    ax.legend(loc="lower right", **legend_kw)
    cbar = fig.colorbar(hb, ax=ax, fraction=0.05, pad=0.02)
    cbar.ax.set_title("count", fontsize=6, pad=3)

    # e. per-strain agreement of A against cell count.
    ax = axes[1, 1]
    p = pairs["pseudobulk_log2fc"]
    r = np.asarray(p["per_strain"]["values"], dtype=float)
    n = np.asarray(
        [cells.get(o, np.nan) for o in p["per_strain"]["deletions"]], dtype=float
    )
    m = np.isfinite(r) & np.isfinite(n)
    ax.scatter(n[m], r[m], s=3, color=col["pseudobulk_log2fc"], lw=0, alpha=0.7)
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.set_xscale("log")
    _plain_log_x(ax, [10, 30, 100, 300, 1000])
    ax.set_xlabel("Nadal-Ribelles cells per genotype")
    ax.set_ylabel("per-strain r, Kemmeren vs A")
    ax.set_title(
        f"agreement vs cell count, Spearman {cells_track['pseudobulk_log2fc']:.2f}"
    )

    # f. Assignment purity: the deleted gene in its own genotype's cells against WT.
    ax = axes[1, 2]
    p2 = purity[ok2]
    ax.scatter(
        p2["wt_frac_detecting"],
        p2["frac_cells_detecting_ko"],
        s=3,
        color=col["pseudobulk_log2fc"],
        lw=0,
        alpha=0.7,
    )
    ax.plot([0, 1], [0, 1], color="black", lw=0.5, ls="--", label="cells look like WT")
    ax.axhline(0, color="black", lw=0.5, ls=":", label="clean deletion")
    ax.set_xlim(0.2, 1.0)
    ax.set_ylim(-0.02, 1.0)
    ax.set_xlabel("WT cells detecting the deleted gene")
    ax.set_ylabel("own cells detecting the deleted gene")
    ax.set_title(
        f"assignment purity, median {purity_summary['wt_detect_ge_0.2']['purity_excess_zero_median']:.2f} (n = {int(ok2.sum())})"
    )
    ax.legend(loc="upper left", **legend_kw)

    for ax in axes.flat:
        for s in ax.spines.values():
            s.set_visible(True)
    fig.subplots_adjust(
        left=0.06, right=0.98, bottom=0.09, top=0.92, wspace=0.42, hspace=0.6
    )
    for ax, letter in zip(axes.flat, "abcdef"):
        panel_label(ax, letter)
    stem = osp.join(images, "cross_study_recomputed")
    fig.savefig(stem + ".png", dpi=300)
    savefig_true_size_svg(fig, stem + ".svg")
    print(f"\nfigure: {stem}.svg")
    print(f"results: {osp.join(results_dir, 'cross_study_recomputed.json')}")


if __name__ == "__main__":
    main()
