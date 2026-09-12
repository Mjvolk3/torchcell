# experiments/028-knockout-expression/scripts/nadal_replication_coverage.py
# [[experiments.028-knockout-expression.scripts.nadal_replication_coverage]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-knockout-expression/scripts/nadal_replication_coverage
"""Within-study replication, gene coverage, and the paper's own deleteome comparison.

Reads the tables written by nadal_batch_replication.R and
nadal_paper_deleteome_comparison.R and draws them beside the microarray reference:

  a  same-genotype cross-batch r (per-batch WT reference) against the different-genotype
     null, and the same with a pooled WT reference, where the batch effect shows up as
     a shift common to both;
  b  split-half r within a batch (WT split too) against cells per genotype and against
     the other-genotype null;
  c  gene coverage: genes a Nadal pseudobulk reports (>= 10 summed UMI with WT) against
     cells per genotype, with Kemmeren's 6,182 reporters per profile as the line;
  d  Kemmeren's changed-gene count per mutant (FC > 1.7, p < 0.05; responsive at >= 4)
     and the same for Nadal's DEG counts (their DEG.Rdata: p < 0.05, |logFC| >= 1);
  e  the paper's Supplementary Fig. 1i, reproduced from its released objects: log DEG
     count in Nadal against log changed-gene count in Kemmeren, per genotype, with the
     Spearman the paper does not print, colored by cells per genotype;
  f  the per-genotype profile Spearman the paper's script computes (`javsma`) and does
     not show, with and without the sentinel genes.

Run from the repo root after the two R scripts:
    python experiments/028-knockout-expression/scripts/nadal_replication_coverage.py
"""

from __future__ import annotations

import json
import os
import os.path as osp
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from cross_study_ko_expression import _plain_log_x  # noqa: E402
from cross_study_recomputed import RECOMPUTED_REL  # noqa: E402

from torchcell.utils import (  # noqa: E402
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    panel_label,
    savefig_true_size_svg,
)
from torchcell.utils.paths import experiment_results_dir  # noqa: E402

KEMMEREN_REPORTERS = 6182  # reporters with M and p in every deleteome profile
MIN_UMI = 10  # the pseudobulk recompute's absence threshold
OWN_UMI = 5  # UMI in the genotype's own cells for a gene to count as covered by it


def _partial_spearman(x: pd.Series, y: pd.Series, z: pd.Series) -> float:
    d = pd.DataFrame({"x": x, "y": y, "z": z}).dropna().rank()
    rx = d["x"] - np.polyval(np.polyfit(d["z"], d["x"], 1), d["z"])
    ry = d["y"] - np.polyval(np.polyfit(d["z"], d["y"], 1), d["z"])
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    images = osp.join(os.environ["ASSET_IMAGES_DIR"], "028-knockout-expression")
    os.makedirs(images, exist_ok=True)
    results_dir = experiment_results_dir("028-knockout-expression", __file__)
    rec = osp.join(data_root, RECOMPUTED_REL)

    same = pd.read_csv(osp.join(rec, "batch_replication_pairs.tsv"), sep="\t")
    null = pd.read_csv(osp.join(rec, "batch_replication_null.tsv"), sep="\t")
    sh = pd.read_csv(osp.join(rec, "split_half_pairs.tsv"), sep="\t")
    wtb = pd.read_csv(osp.join(rec, "wt_batch_effect.tsv"), sep="\t")
    depth = pd.read_csv(osp.join(rec, "cell_depth.tsv"), sep="\t")
    paper = pd.read_csv(osp.join(rec, "paper_deleteome_comparison.tsv"), sep="\t")
    resp = pd.read_csv(osp.join(rec, "kemmeren_responsive.tsv"), sep="\t")
    resp = resp[resp["mutant"].str.contains("-del")]
    genotypes = pd.read_csv(osp.join(rec, "genotypes.tsv"), sep="\t")
    umi = pd.read_csv(osp.join(rec, "pseudobulk_umi.tsv.gz"), sep="\t", index_col=0)

    # Coverage per genotype: genes with >= MIN_UMI summed UMI (genotype + WT).
    # Two coverages: the recompute's rule (genotype + WT summed UMI >= MIN_UMI, which the
    # 500 WT cells satisfy for most genes on their own) and the genotype's own depth
    # (genes with >= OWN_UMI UMI in the genotype's cells alone).
    wt = umi["WT"]
    cover = ((umi.add(wt, axis=0)) >= MIN_UMI).sum().drop("WT")
    own = (umi >= OWN_UMI).sum().drop("WT")
    cells = genotypes.set_index("label")["n_cells"].reindex(cover.index)
    cov = pd.DataFrame({"cells": cells, "genes": cover, "genes_own": own}).dropna()
    same_min = same[["cells_i", "cells_j"]].min(axis=1)

    out: dict[str, Any] = {
        "generated_by": "experiments/028-knockout-expression/scripts/nadal_replication_coverage.py",
        "cross_batch": {
            "n_pairs": int(len(same)),
            "same_genotype_median_r_batch_ref": float(same["r_batch_ref"].median()),
            "same_genotype_median_r_pool_ref": float(same["r_pool_ref"].median()),
            "null_median_r_batch_ref": float(null["r_batch_ref"].median()),
            "null_median_r_pool_ref": float(null["r_pool_ref"].median()),
            "same_frac_r_above_0.2": float((same["r_batch_ref"] > 0.2).mean()),
            "null_frac_r_above_0.2": float((null["r_batch_ref"] > 0.2).mean()),
            "same_by_min_cells": {
                f"{lo}-{hi}": {
                    "n": int(((same_min >= lo) & (same_min < hi)).sum()),
                    "median_r": float(
                        same.loc[
                            (same_min >= lo) & (same_min < hi), "r_batch_ref"
                        ].median()
                    ),
                }
                for lo, hi in ((15, 30), (30, 60), (60, 120), (120, 10**6))
            },
            "spearman_r_vs_min_cells": float(
                spearmanr(same_min, same["r_batch_ref"], nan_policy="omit").correlation
            ),
        },
        "split_half": {
            "n": int(len(sh)),
            "median_r": float(sh["r_split_half"].median()),
            "null_median_r": float(sh["r_null_other_genotype"].median()),
            "frac_r_above_0.2": float((sh["r_split_half"] > 0.2).mean()),
            "spearman_r_vs_cells": float(
                spearmanr(
                    sh["cells"], sh["r_split_half"], nan_policy="omit"
                ).correlation
            ),
            "by_cells": {
                f"{lo}-{hi}": {
                    "n": int(((sh["cells"] >= lo) & (sh["cells"] < hi)).sum()),
                    "median_r": float(
                        sh.loc[
                            (sh["cells"] >= lo) & (sh["cells"] < hi), "r_split_half"
                        ].median()
                    ),
                }
                for lo, hi in ((30, 60), (60, 120), (120, 300), (300, 10**6))
            },
        },
        "wt_batch_effect": {
            "n_batch_pairs": int(len(wtb)),
            "median_sd_log2fc": float(wtb["sd_log2fc"].median()),
            "median_frac_abs_gt1": float(wtb["frac_abs_gt1"].median()),
        },
        "depth": {
            "cells": int(len(depth)),
            "median_umi_per_cell": float(depth["umi"].median()),
            "median_genes_per_cell": float(depth["genes_detected"].median()),
        },
        "coverage": {
            "kemmeren_reporters_per_profile": KEMMEREN_REPORTERS,
            "nadal_genes_in_object": int(len(umi)),
            "nadal_median_genes_reported_per_genotype": float(cov["genes"].median()),
            "nadal_median_genes_ge5umi_own_cells": float(cov["genes_own"].median()),
            "nadal_by_cells_reported": {
                f"{lo}-{hi}": float(
                    cov.loc[
                        (cov["cells"] >= lo) & (cov["cells"] < hi), "genes"
                    ].median()
                )
                for lo, hi in ((1, 10), (10, 30), (30, 100), (100, 300), (300, 10**6))
            },
            "nadal_by_cells_own": {
                f"{lo}-{hi}": float(
                    cov.loc[
                        (cov["cells"] >= lo) & (cov["cells"] < hi), "genes_own"
                    ].median()
                )
                for lo, hi in ((1, 10), (10, 30), (30, 100), (100, 300), (300, 10**6))
            },
        },
        "kemmeren_responsive": {
            "n_profiles": int(len(resp)),
            "frac_responsive": float(resp["responsive"].mean()),
            "median_changed_genes": float(resp["n_sig_fc1p7"].median()),
        },
        "paper_figS1i": {
            "n": int(len(paper)),
            "spearman_log_counts": float(
                spearmanr(
                    np.log(paper["MAsig"] + 1), np.log(paper["JAnsig"] + 1)
                ).correlation
            ),
            "spearman_partial_cells": _partial_spearman(
                paper["MAsig"], paper["JAnsig"], paper["ncells"]
            ),
            "spearman_nadal_degs_vs_cells": float(
                spearmanr(paper["JAnsig"], paper["ncells"]).correlation
            ),
            "spearman_kemmeren_changed_vs_cells": float(
                spearmanr(paper["MAsig"], paper["ncells"]).correlation
            ),
            "javsma_median": float(paper["javsma"].median()),
            "javsma_iqr": [
                float(paper["javsma"].quantile(0.25)),
                float(paper["javsma"].quantile(0.75)),
            ],
            "javsma_no_sentinel_median": float(paper["javsma_no_sentinel"].median()),
            "javsma_frac_above_0.2": float((paper["javsma"] > 0.2).mean()),
        },
    }
    print(json.dumps(out, indent=1))
    with open(osp.join(results_dir, "nadal_replication_coverage.json"), "w") as f:
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
    legend_kw = dict(frameon=True, edgecolor="black", fancybox=False, framealpha=1.0)
    c_nad, c_kem, c_null = PLOT_PALETTE[0], PLOT_PALETTE[3], PLOT_PALETTE[5]
    fig, axes = plt.subplots(
        2, 3, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(105))
    )

    def filled(ax, v, color, bins, label, ls="-"):
        ax.hist(
            v,
            bins=bins,
            histtype="stepfilled",
            facecolor=to_rgba(color, 0.45),
            edgecolor="black",
            lw=0.4,
            ls=ls,
            density=True,
            label=label,
        )

    # a. cross-batch replication
    ax = axes[0, 0]
    bins = np.linspace(-0.4, 1.0, 36)
    filled(
        ax,
        same["r_batch_ref"].dropna(),
        c_nad,
        bins,
        f"same genotype (med {same['r_batch_ref'].median():.2f})",
    )
    filled(
        ax,
        null["r_batch_ref"].dropna(),
        c_null,
        bins,
        f"different genotypes (med {null['r_batch_ref'].median():.2f})",
    )
    ax.hist(
        same["r_pool_ref"].dropna(),
        bins=bins,
        histtype="step",
        color=c_nad,
        lw=0.9,
        ls="--",
        density=True,
        label=f"same, pooled WT ref (med {same['r_pool_ref'].median():.2f})",
    )
    ax.set_xlabel("r between two batches of a genotype, Nadal A")
    ax.set_ylabel("density")
    ax.set_title(f"cross-batch replication ({len(same):,} pairs)")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.8)
    ax.legend(loc="upper right", **legend_kw)

    # b. split-half within batch vs cells
    ax = axes[0, 1]
    ax.scatter(
        sh["cells"],
        sh["r_null_other_genotype"],
        s=3,
        color=c_null,
        lw=0,
        alpha=0.5,
        label=f"other genotype (med {sh['r_null_other_genotype'].median():.2f})",
    )
    ax.scatter(
        sh["cells"],
        sh["r_split_half"],
        s=3,
        color=c_nad,
        lw=0,
        alpha=0.7,
        label=f"split half (med {sh['r_split_half'].median():.2f})",
    )
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.set_xscale("log")
    _plain_log_x(ax, [30, 100, 300, 1000])
    ax.set_xlabel("cells of the genotype in the batch")
    ax.set_ylabel("r between halves, each vs its own WT half")
    rho = out["split_half"]["spearman_r_vs_cells"]
    ax.set_title(f"split-half within batch, Spearman vs cells {rho:.2f}")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + (hi - lo) * 0.45)
    ax.legend(loc="upper left", **legend_kw)

    # c. coverage
    ax = axes[0, 2]
    ax.scatter(
        cov["cells"],
        cov["genes"],
        s=2,
        color=c_null,
        lw=0,
        alpha=0.4,
        label=f"reported (genotype + WT >= {MIN_UMI} UMI)",
    )
    ax.scatter(
        cov["cells"],
        cov["genes_own"],
        s=2,
        color=c_nad,
        lw=0,
        alpha=0.4,
        label=f"own cells >= {OWN_UMI} UMI",
    )
    ax.axhline(
        KEMMEREN_REPORTERS,
        color=c_kem,
        lw=1.0,
        label=f"Kemmeren, every profile ({KEMMEREN_REPORTERS:,})",
    )
    ax.axhline(
        len(umi),
        color="black",
        lw=0.5,
        ls=":",
        label=f"genes in the Nadal object ({len(umi):,})",
    )
    ax.set_xscale("log")
    _plain_log_x(ax, [1, 10, 100, 1000])
    ax.set_xlabel("cells per genotype")
    ax.set_ylabel("genes in the profile")
    ax.set_title(
        f"genes per profile ({out['depth']['median_umi_per_cell']:.0f} UMI per cell)"
    )
    # Headroom above the two reference lines so the legend sits clear of them.
    ax.set_ylim(0, 13500)
    ax.legend(loc="upper left", **legend_kw)

    # d. changed-gene counts, both studies
    ax = axes[1, 0]
    bins = np.logspace(0, 3.2, 33)
    filled(
        ax,
        resp["n_sig_fc1p7"] + 1,
        c_kem,
        bins,
        f"Kemmeren (med {resp['n_sig_fc1p7'].median():.0f}; {100 * resp['responsive'].mean():.0f}% >= 4)",
    )
    filled(
        ax,
        paper["JAnsig"] + 1,
        c_nad,
        bins,
        f"Nadal, paper's DEGs (med {paper['JAnsig'].median():.0f})",
    )
    ax.set_xscale("log")
    _plain_log_x(ax, [1, 10, 100, 1000])
    ax.set_xlabel("changed genes per deletion + 1")
    ax.set_ylabel("density")
    ax.set_title("changed genes per deletion, each study's own rule")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.7)
    ax.legend(loc="upper right", **legend_kw)

    # e. the paper's Supp Fig 1i
    ax = axes[1, 1]
    sc = ax.scatter(
        paper["MAsig"] + 1,
        paper["JAnsig"] + 1,
        s=4,
        c=np.log10(paper["ncells"]),
        cmap="Greys",
        lw=0.2,
        edgecolor="black",
        alpha=0.9,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    _plain_log_x(ax, [1, 10, 100, 1000])
    ax.set_xlabel("Kemmeren changed genes + 1 (|M| > 1, p < 0.05)")
    ax.set_ylabel(
        f"Nadal DEGs + 1 (Spearman vs cells {out['paper_figS1i']['spearman_nadal_degs_vs_cells']:.2f})"
    )
    s = out["paper_figS1i"]
    ax.set_title(f"paper's Fig. S1i, Spearman {s['spearman_log_counts']:.2f}")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02)
    cbar.ax.set_title("log10 cells", fontsize=6, pad=3)

    # f. the unshown per-genotype profile correlation
    ax = axes[1, 2]
    bins = np.linspace(-0.3, 0.6, 46)
    filled(
        ax,
        paper["javsma"].dropna(),
        c_nad,
        bins,
        f"as the paper computes it (med {paper['javsma'].median():.3f})",
    )
    ax.hist(
        paper["javsma_no_sentinel"].dropna(),
        bins=bins,
        histtype="step",
        color="black",
        lw=0.8,
        density=True,
        label=f"sentinel genes removed (med {paper['javsma_no_sentinel'].median():.3f})",
    )
    ax.axvline(0, color="black", lw=0.5, ls="--")
    ax.set_xlabel("per-genotype Spearman, Nadal logFC vs Kemmeren")
    ax.set_ylabel("density")
    ax.set_title(f"the paper's unshown profile Spearman (n = {len(paper)})")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.6)
    ax.legend(loc="upper right", **legend_kw)

    for ax in axes.flat:
        for sp in ax.spines.values():
            sp.set_visible(True)
    fig.subplots_adjust(
        left=0.06, right=0.98, bottom=0.09, top=0.92, wspace=0.42, hspace=0.6
    )
    for ax, letter in zip(axes.flat, "abcdef"):
        panel_label(ax, letter)
    stem = osp.join(images, "nadal_replication_coverage")
    fig.savefig(stem + ".png", dpi=300)
    savefig_true_size_svg(fig, stem + ".svg")
    print(f"\nfigure: {stem}.svg")
    print(f"results: {osp.join(results_dir, 'nadal_replication_coverage.json')}")


if __name__ == "__main__":
    main()
