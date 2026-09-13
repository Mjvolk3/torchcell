# experiments/028-knockout-expression/scripts/gene_covariation_all.py
# [[experiments.028-knockout-expression.scripts.gene_covariation_all]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-knockout-expression/scripts/gene_covariation_all
"""Gene x gene co-variation in every panel, and how much of it the panels share.

No strain is aligned across panels here. Within one panel, two genes get one Pearson
across that panel's own strains (Messner's 4,549 deletions, Caudal's 943 natural
isolates, Kemmeren's 1,484 deletions, Sameith's single deletions, Nadal-Ribelles A's
deletions with >= 50 cells), so a panel becomes one gene x gene matrix on a common
gene list. The comparison between two panels is then the Spearman between the upper
triangles of their matrices: do the same gene pairs co-vary, whatever perturbed the
strains. That is the only comparison a deletion panel and an isolate panel admit, and
it is the same one the proteome figure draws for Messner alone.

Two gene lists, because Messner measures 1,830 proteins while the mRNA panels report
about 6,000 genes: the proteins measured in every panel (all five matrices), and the
genes measured in every mRNA panel (four matrices). Each gene x gene matrix is drawn
in one ordering, Kemmeren's average-linkage clustering on that gene list, so a block
that appears in Kemmeren can be looked for at the same place in every other panel.

Inputs: the LMDBs of proteome_messner2023, microarray_kemmeren2014,
sm_microarray_sameith2015, caudal_pantranscriptome2024 and the Nadal pseudobulk
recompute. Run from the repo root:
    python experiments/028-knockout-expression/scripts/gene_covariation_all.py
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
from matplotlib.colors import LinearSegmentedColormap, to_rgba  # noqa: E402
from scipy.cluster.hierarchy import leaves_list, linkage  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from cross_study_ko_expression import LMDB, _load_lmdb, _profiles  # noqa: E402
from cross_study_recomputed import (  # noqa: E402
    RECOMPUTED_REL,
    _recomputed_profiles,
    _resolver,
)
from cross_study_structure import _matrix, _palette_cmap, _upper  # noqa: E402
from proteome_expression_covariation import (  # noqa: E402
    LMDB_CAUDAL,
    LMDB_PROTEOME,
    MIN_STRAINS_FOR_COVAR,
    NADAL_MIN_CELLS,
    _caudal,
    _gene_covar,
    _messner,
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

ORDER = ["messner", "caudal", "kemmeren", "sameith", "nadalA"]
NAME = {
    "messner": "Messner",
    "caudal": "Caudal",
    "kemmeren": "Kemmeren",
    "sameith": "Sameith",
    "nadalA": "Nadal A",
}
STRAIN = {
    "messner": "deletions",
    "caudal": "isolates",
    "kemmeren": "deletions",
    "sameith": "deletions",
    "nadalA": "deletions",
}
MODE = {k: ("protein" if k == "messner" else "mRNA") for k in STRAIN}
COLOR = {
    "messner": PLOT_PALETTE[1],
    "caudal": PLOT_PALETTE[2],
    "kemmeren": PLOT_PALETTE[3],
    "sameith": PLOT_PALETTE[0],
    "nadalA": PLOT_PALETTE[4],
}
HEAT_LIM = 0.6  # gene-pair r color range, symmetric
STRONG_PAIR = 0.3  # |r| beyond this counts as a strong pair


def _common_covar(
    mats: dict[str, pd.DataFrame], keys: list[str]
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """One gene x gene matrix per panel on the genes measured in every panel of
    `keys` (each in >= MIN_STRAINS_FOR_COVAR of the panel's strains), all matrices
    on the same gene list.
    """
    genes = sorted(set.intersection(*(set(mats[k].columns) for k in keys)))
    cov = {k: _gene_covar(mats[k], genes) for k in keys}
    common = sorted(set.intersection(*(set(cov[k].index) for k in keys)))
    return {k: cov[k].loc[common, common] for k in keys}, common


def _pairwise(cov: dict[str, pd.DataFrame], keys: list[str]) -> pd.DataFrame:
    """Spearman between upper triangles, gene pairs finite in both."""
    tri = {k: _upper(cov[k]) for k in keys}
    out = pd.DataFrame(np.nan, index=keys, columns=keys, dtype=float)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            ok = np.isfinite(tri[a]) & np.isfinite(tri[b])
            rho = spearmanr(tri[a][ok], tri[b][ok]).correlation
            out.loc[a, b] = out.loc[b, a] = rho
    return out


def _structure(cov: dict[str, pd.DataFrame]) -> dict[str, dict[str, float]]:
    """How much co-variation a panel holds at all: spread of the gene-pair r."""
    out = {}
    for k, c in cov.items():
        u = _upper(c)
        u = u[np.isfinite(u)]
        out[k] = {
            "n_pairs": int(len(u)),
            "sd": float(u.std()),
            "median_abs": float(np.median(np.abs(u))),
            f"frac_abs_above_{STRONG_PAIR}": float((np.abs(u) > STRONG_PAIR).mean()),
        }
    return out


def _cluster_order(c: pd.DataFrame) -> list[str]:
    """Average-linkage leaves on correlation distance; missing pairs count as r = 0."""
    r = c.to_numpy(dtype=float)
    r = np.where(np.isfinite(r), r, 0.0)
    np.fill_diagonal(r, 1.0)
    d = 1.0 - (r + r.T) / 2
    z = linkage(squareform(d, checks=False), method="average")
    return [c.index[i] for i in leaves_list(z)]


def _diverging() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "pal_div", [PLOT_PALETTE[4], "#FFFFFF", PLOT_PALETTE[1]]
    )


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

    P, _ = _messner(_load_lmdb(osp.join(data_root, LMDB_PROTEOME)))
    kem, _, _ = _profiles(_load_lmdb(osp.join(data_root, LMDB["kemmeren"])))
    sam, _, _ = _profiles(_load_lmdb(osp.join(data_root, LMDB["sameith"])))
    nadA, cells, _ = _recomputed_profiles(
        data_root, "pseudobulk_log2fc", resolve, genotypes
    )
    mats = {
        "messner": P,
        "caudal": _caudal(_load_lmdb(osp.join(data_root, LMDB_CAUDAL)), resolve),
        "kemmeren": _matrix(kem, sorted(kem)),
        "sameith": _matrix(sam, sorted(sam)),
        "nadalA": _matrix(
            nadA, sorted(o for o in nadA if cells.get(o, 0) >= NADAL_MIN_CELLS)
        ),
    }
    for k, m in mats.items():
        print(f"{k}: {m.shape[0]} strains x {m.shape[1]} genes")

    sets = {"protein": ORDER, "expression": [k for k in ORDER if k != "messner"]}
    out: dict[str, Any] = {
        "generated_by": "experiments/028-knockout-expression/scripts/gene_covariation_all.py",
        "min_fraction_of_strains_per_gene": MIN_STRAINS_FOR_COVAR,
        "nadal_min_cells": NADAL_MIN_CELLS,
        "strains": {k: int(m.shape[0]) for k, m in mats.items()},
        "gene_sets": {},
    }
    cov_all: dict[str, dict[str, pd.DataFrame]] = {}
    rho_all: dict[str, pd.DataFrame] = {}
    order_all: dict[str, list[str]] = {}
    for name, keys in sets.items():
        cov, genes = _common_covar(mats, keys)
        rho = _pairwise(cov, keys)
        order = _cluster_order(cov["kemmeren"])
        cov_all[name], rho_all[name], order_all[name] = cov, rho, order
        out["gene_sets"][name] = {
            "panels": keys,
            "n_genes": len(genes),
            "pairwise_spearman": {
                a: {b: float(rho.loc[a, b]) for b in keys if b != a} for a in keys
            },
            "structure": _structure(cov),
            "ordering": "average-linkage clustering of the Kemmeren matrix",
        }
        print(f"\n{name} gene set: {len(genes)} genes")
        print(rho.round(3).to_string())
        print(json.dumps(out["gene_sets"][name]["structure"], indent=1))
    with open(osp.join(results_dir, "gene_covariation_all.json"), "w") as f:
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
            "savefig.dpi": 300,
        }
    )
    legend_kw = dict(frameon=True, edgecolor="black", fancybox=False, framealpha=1.0)
    fig = plt.figure(figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(150)))
    # Two grids: the top row needs room for the row labels of b, the matrix block
    # needs none (no tick labels), so its cells can be nearly square and larger.
    gs_top = fig.add_gridspec(
        1,
        3,
        width_ratios=[2, 2, 1.15],
        left=0.085,
        right=0.985,
        bottom=0.70,
        top=0.935,
        wspace=0.72,
    )
    gs = fig.add_gridspec(
        2, 5, left=0.03, right=0.985, bottom=0.045, top=0.585, wspace=0.22, hspace=0.40
    )
    axes_top = [fig.add_subplot(gs_top[0, i]) for i in range(3)]
    letters = iter("abcdefghijklmn")

    def rho_heat(ax, rho: pd.DataFrame, title: str) -> None:
        keys = list(rho.index)
        v = rho.to_numpy(dtype=float)
        shown = np.where(np.isfinite(v), v, np.nan)
        ax.imshow(
            shown, cmap=_palette_cmap(PLOT_PALETTE[1]), vmin=0, vmax=0.6, aspect="auto"
        )
        for i in range(len(keys)):
            for j in range(len(keys)):
                if i != j:
                    ax.text(
                        j, i, f"{v[i, j]:.2f}", ha="center", va="center", fontsize=6
                    )
        ax.set_xticks(range(len(keys)))
        ax.set_yticks(range(len(keys)))
        ax.set_xticklabels([NAME[k] for k in keys], rotation=30, ha="right")
        ax.set_yticklabels([NAME[k] for k in keys])
        ax.set_title(title)

    ng = {n: out["gene_sets"][n]["n_genes"] for n in sets}
    rho_heat(
        axes_top[0],
        rho_all["protein"],
        f"gene-pair r agreement, {ng['protein']:,} proteins in all five",
    )
    rho_heat(
        axes_top[1],
        rho_all["expression"],
        f"same, {ng['expression']:,} genes in the four mRNA panels",
    )

    # c. how much co-variation each panel holds (protein gene set)
    ax = axes_top[2]
    bins = np.linspace(-1, 1, 81)
    for k in ORDER:
        u = _upper(cov_all["protein"][k])
        u = u[np.isfinite(u)]
        ax.hist(
            u,
            bins=bins,
            histtype="stepfilled",
            facecolor=to_rgba(COLOR[k], 0.45),
            edgecolor="black",
            lw=0.4,
            density=True,
            label=NAME[k],
        )
    ax.set_xlabel("gene-pair r within a panel")
    ax.set_ylabel("density")
    ax.set_title("spread of gene-pair r")
    ax.set_xlim(-1, 1)
    hist_handles = ax.get_legend_handles_labels()

    for ax in axes_top:
        panel_label(ax, next(letters))

    # the matrix block: the first cell holds the key (legend of c, colorbar of the
    # matrices), then the five protein-list matrices and the four mRNA-list matrices
    # flow in reading order, one ordering per gene list.
    cmap = _diverging()
    last_im = None
    slots = [(r, c) for r in range(2) for c in range(5)][1:]
    for name in ("protein", "expression"):
        keys = sets[name]
        order = order_all[name]
        for k in keys:
            row, col = slots.pop(0)
            ax = fig.add_subplot(gs[row, col])
            c = cov_all[name][k].loc[order, order].to_numpy(dtype=float)
            last_im = ax.imshow(
                c,
                cmap=cmap,
                vmin=-HEAT_LIM,
                vmax=HEAT_LIM,
                interpolation="nearest",
                aspect="equal",
                rasterized=True,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(
                f"{NAME[k]}, {len(order):,} {'proteins' if name == 'protein' else 'genes'}"
            )
            ax.set_xlabel(f"{mats[k].shape[0]:,} {STRAIN[k]}, {MODE[k]}")
            panel_label(ax, next(letters))
    key = fig.add_subplot(gs[0, 0])
    key.set_axis_off()
    key.legend(*hist_handles, loc="upper left", title="c", **legend_kw)
    cax = key.inset_axes([0.08, 0.10, 0.84, 0.07])
    cb = fig.colorbar(last_im, cax=cax, orientation="horizontal")
    cb.set_label(f"d to l: gene-pair r, clipped at ±{HEAT_LIM}")
    cb.outline.set_linewidth(0.5)

    for ax in fig.axes:
        for sp in ax.spines.values():
            sp.set_visible(True)
    stem = osp.join(images, "gene_covariation_all")
    fig.savefig(stem + ".png", dpi=300)
    savefig_true_size_svg(fig, stem + ".svg", dpi=300)
    print(f"\nfigure: {stem}.svg")
    print(f"results: {osp.join(results_dir, 'gene_covariation_all.json')}")


if __name__ == "__main__":
    main()
