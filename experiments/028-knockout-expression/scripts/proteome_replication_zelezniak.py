# experiments/028-knockout-expression/scripts/proteome_replication_zelezniak.py
# [[experiments.028-knockout-expression.scripts.proteome_replication_zelezniak]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-knockout-expression/scripts/proteome_replication_zelezniak
"""Does a second knockout proteome reproduce Messner, and does it track Kemmeren mRNA?

Zelezniak et al. 2018 measured 726 proteins in 97 kinase deletions of the same
collection (SWATH, SM medium, three replicates against a 12-replicate wild type).
Messner 2023 (same lab, five years later, one measurement per strain) covers all 97, and
Kemmeren 2014 covers 94 of them. So on these strains three comparisons are possible,
strain by strain:

  Zelezniak protein vs Messner protein   the replication check a knockout proteome has
                                         not had here: same deletions, same modality,
                                         different years, platform version and reference
  Zelezniak protein vs Kemmeren mRNA     protein against mRNA on a second proteome
  Messner protein vs Kemmeren mRNA       the existing pairing on the same 94 strains,
                                         so the three numbers are on one footing

Each value is a log ratio over the study's own wild type (Zelezniak's stored log signal
minus its reference, Messner's log2 over the HIS3 reference), z-scored per gene across
the shared strains. Per deletion the Pearson is over the proteins both measure, per
protein across the shared deletions. Gene co-variation (the gene-pair matrix across the
97 strains) is also compared, with the caveat that 97 strains give a noisier matrix
than the thousands behind the other panels.

Run from the repo root:
    python experiments/028-knockout-expression/scripts/proteome_replication_zelezniak.py
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
from cross_study_ko_expression import LMDB, _load_lmdb, _profiles  # noqa: E402
from cross_study_structure import _matrix, _palette_cmap, _upper  # noqa: E402
from proteome_expression_covariation import LMDB_PROTEOME, _messner  # noqa: E402

from torchcell.utils import (  # noqa: E402
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    panel_label,
    savefig_true_size_svg,
)
from torchcell.utils.paths import experiment_results_dir  # noqa: E402

LMDB_ZELEZNIAK = "data/torchcell/proteome_zelezniak2018/processed/lmdb"
MIN_SHARED_FOR_STRAIN_R = 100  # genes a strain needs on both sides
MIN_STRAINS_FOR_GENE_R = 40  # strains a gene needs on both sides (97 available)
MIN_COVERAGE = 0.8  # fraction of strains a gene must be measured in for co-variation


def _zelezniak(records: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.Series]:
    """Deletion x protein log ratio over the study's wild-type reference."""
    ref = pd.Series(records[0]["reference"]["phenotype_reference"]["protein_abundance"])
    rows = {}
    for rec in records:
        perts = rec["experiment"]["genotype"]["perturbations"]
        if len(perts) != 1:
            continue
        orf = perts[0]["systematic_gene_name"]
        ab = pd.Series(rec["experiment"]["phenotype"]["protein_abundance"])
        rows[orf] = (ab - ref).dropna()
    return pd.DataFrame(rows).T, ref


def _aligned(A: pd.DataFrame, B: pd.DataFrame) -> dict[str, Any]:
    """Per-strain and per-gene Pearson between two strain x gene matrices on their
    shared strains and genes, each z-scored per gene across the shared strains.
    """
    strains = sorted(set(A.index) & set(B.index))
    genes = sorted(set(A.columns) & set(B.columns))
    a = A.loc[strains, genes]
    b = B.loc[strains, genes]
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    both = a.notna() & b.notna()
    per_strain = pd.Series(np.nan, index=strains)
    for s in strains:
        m = both.loc[s]
        if m.sum() >= MIN_SHARED_FOR_STRAIN_R:
            per_strain[s] = np.corrcoef(a.loc[s, m], b.loc[s, m])[0, 1]
    per_gene = pd.Series(np.nan, index=genes)
    for g in genes:
        m = both[g]
        if m.sum() >= MIN_STRAINS_FOR_GENE_R:
            per_gene[g] = np.corrcoef(a.loc[m, g], b.loc[m, g])[0, 1]
    return {
        "n_shared_strains": len(strains),
        "n_shared_genes": len(genes),
        "per_strain_median_r": float(per_strain.median()),
        "per_strain_n": int(per_strain.notna().sum()),
        "per_gene_median_r": float(per_gene.median()),
        "per_gene_n": int(per_gene.notna().sum()),
        "per_gene_frac_above_0.3": float((per_gene > 0.3).mean()),
        "_per_strain": per_strain,
        "_per_gene": per_gene,
    }


def _covar(m: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    sub = m[[g for g in genes if g in m.columns]]
    sub = sub.loc[:, sub.notna().mean() >= MIN_COVERAGE]
    return sub.corr(min_periods=int(MIN_COVERAGE * len(sub)))


def main() -> None:
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    images = osp.join(os.environ["ASSET_IMAGES_DIR"], "028-knockout-expression")
    os.makedirs(images, exist_ok=True)
    results_dir = experiment_results_dir("028-knockout-expression", __file__)

    Z, z_ref = _zelezniak(_load_lmdb(osp.join(data_root, LMDB_ZELEZNIAK)))
    P, p_ref = _messner(_load_lmdb(osp.join(data_root, LMDB_PROTEOME)))
    kem, _, _ = _profiles(_load_lmdb(osp.join(data_root, LMDB["kemmeren"])))
    K = _matrix(kem, sorted(kem))
    strains_zp = sorted(set(Z.index) & set(P.index))
    strains_zk = sorted(set(Z.index) & set(K.index))
    strains_all = sorted(set(strains_zp) & set(strains_zk))
    print(
        f"Zelezniak {Z.shape}; shared with Messner {len(strains_zp)}, with Kemmeren "
        f"{len(strains_zk)}, all three {len(strains_all)}"
    )

    pairs = {
        "zelezniak_vs_messner": _aligned(Z, P),
        "zelezniak_vs_kemmeren": _aligned(Z, K),
        # Messner against Kemmeren restricted to the same strains and Zelezniak's genes,
        # so the third number is on the same footing as the first two.
        "messner_vs_kemmeren_same_strains": _aligned(
            P.loc[strains_all, [g for g in Z.columns if g in P.columns]], K
        ),
    }
    out: dict[str, Any] = {
        "generated_by": "experiments/028-knockout-expression/scripts/proteome_replication_zelezniak.py",
        "zelezniak_shape": list(Z.shape),
        "shared_strains": {
            "zelezniak_messner": len(strains_zp),
            "zelezniak_kemmeren": len(strains_zk),
            "all_three": len(strains_all),
        },
        "strain_aligned": {
            k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
            for k, v in pairs.items()
        },
    }
    print(json.dumps(out["strain_aligned"], indent=1))

    # gene co-variation on Zelezniak's proteins across the shared strains
    genes = sorted(set(Z.columns) & set(P.columns) & set(K.columns))
    cov = {
        "zelezniak": _covar(Z.loc[strains_all], genes),
        "messner": _covar(P.loc[strains_all], genes),
        "kemmeren": _covar(K.loc[strains_all], genes),
        # Messner across all its deletions on the same genes: does the 97-strain matrix
        # resemble the full one, which says how much the strain count limits it.
        "messner_all": _covar(P, genes),
    }
    common = sorted(set.intersection(*(set(c.index) for c in cov.values())))
    cov = {k: c.loc[common, common] for k, c in cov.items()}
    tri = {k: _upper(c) for k, c in cov.items()}
    covar_pairs = [
        ("zelezniak", "messner"),
        ("zelezniak", "kemmeren"),
        ("messner", "kemmeren"),
        ("zelezniak", "messner_all"),
        ("messner", "messner_all"),
    ]
    out["gene_covariation"] = {"n_genes": len(common)}
    for a, b in covar_pairs:
        ok = np.isfinite(tri[a]) & np.isfinite(tri[b])
        out["gene_covariation"][f"{a}_vs_{b}"] = {
            "n_pairs": int(ok.sum()),
            "spearman": float(spearmanr(tri[a][ok], tri[b][ok]).correlation),
        }
    print(json.dumps(out["gene_covariation"], indent=1))

    # abundance level: the two wild-type references
    lev = pd.DataFrame({"zelezniak": z_ref, "messner": p_ref}).dropna()
    out["level"] = {
        "n_genes": int(len(lev)),
        "spearman": float(spearmanr(lev["zelezniak"], lev["messner"]).correlation),
    }
    with open(osp.join(results_dir, "proteome_replication_zelezniak.json"), "w") as f:
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
    legend_kw = dict(
        frameon=True, edgecolor="black", fancybox=False, framealpha=1.0, markerscale=2.5
    )
    col = {
        "zelezniak_vs_messner": PLOT_PALETTE[1],
        "zelezniak_vs_kemmeren": PLOT_PALETTE[3],
        "messner_vs_kemmeren_same_strains": PLOT_PALETTE[4],
    }
    name = {
        "zelezniak_vs_messner": "Zelezniak vs Messner, protein",
        "zelezniak_vs_kemmeren": "Zelezniak vs Kemmeren mRNA",
        "messner_vs_kemmeren_same_strains": "Messner vs Kemmeren mRNA",
    }
    fig, axes = plt.subplots(
        2, 3, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(105))
    )

    def filled(ax, v, color, bins, label):
        ax.hist(
            v,
            bins=bins,
            histtype="stepfilled",
            facecolor=to_rgba(color, 0.45),
            edgecolor="black",
            lw=0.4,
            density=True,
            label=label,
        )

    ax = axes[0, 0]
    bins = np.linspace(-0.6, 1.0, 33)
    for k, v in pairs.items():
        filled(ax, v["_per_strain"].dropna(), col[k], bins, name[k])
    ax.axvline(0, color="black", lw=0.5, ls="--")
    ax.set_xlabel("per-deletion r over shared genes (z-scored)")
    ax.set_ylabel("density")
    meds = ", ".join(f"{v['per_strain_median_r']:.2f}" for v in pairs.values())
    ax.set_title(f"per deletion, medians {meds}")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.7)
    ax.legend(loc="upper right", **legend_kw)

    ax = axes[0, 1]
    for k, v in pairs.items():
        filled(ax, v["_per_gene"].dropna(), col[k], bins, name[k])
    ax.axvline(0, color="black", lw=0.5, ls="--")
    ax.set_xlabel("per-gene r across shared deletions")
    ax.set_ylabel("density")
    meds = ", ".join(f"{v['per_gene_median_r']:.2f}" for v in pairs.values())
    ax.set_title(f"per gene, medians {meds}")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.7)
    ax.legend(loc="upper right", **legend_kw)

    ax = axes[0, 2]
    ax.hexbin(
        lev["zelezniak"],
        lev["messner"],
        gridsize=35,
        bins="log",
        cmap=_palette_cmap(PLOT_PALETTE[1]),
        linewidths=0,
    )
    ax.set_xlabel("Zelezniak: wild-type log signal")
    ax.set_ylabel("Messner: HIS3 reference log2")
    ax.set_title(
        f"level per protein, Spearman {out['level']['spearman']:.2f} (n = {len(lev)})"
    )

    for ax, (a, b), color in zip(
        axes[1],
        [
            ("zelezniak", "messner"),
            ("zelezniak", "kemmeren"),
            ("messner", "messner_all"),
        ],
        [PLOT_PALETTE[1], PLOT_PALETTE[3], PLOT_PALETTE[2]],
    ):
        ok = np.isfinite(tri[a]) & np.isfinite(tri[b])
        ax.hexbin(
            tri[a][ok],
            tri[b][ok],
            gridsize=40,
            bins="log",
            cmap=_palette_cmap(color),
            linewidths=0,
        )
        ax.axhline(0, color="black", lw=0.5, ls="--")
        ax.axvline(0, color="black", lw=0.5, ls="--")
        lab = {
            "zelezniak": f"Zelezniak, {len(strains_all)} deletions",
            "messner": f"Messner, {len(strains_all)} deletions",
            "kemmeren": f"Kemmeren mRNA, {len(strains_all)} deletions",
            "messner_all": f"Messner, {P.shape[0]:,} deletions",
        }
        ax.set_xlabel(f"gene-pair r, {lab[a]}")
        ax.set_ylabel(f"same pair, {lab[b]}")
        rho = out["gene_covariation"][f"{a}_vs_{b}"]["spearman"]
        ax.set_title(f"gene pairs: Spearman {rho:.2f}, {len(common)} proteins")

    for ax in axes.flat:
        for sp in ax.spines.values():
            sp.set_visible(True)
    fig.subplots_adjust(
        left=0.06, right=0.98, bottom=0.09, top=0.92, wspace=0.42, hspace=0.6
    )
    for ax, letter in zip(axes.flat, "abcdef"):
        panel_label(ax, letter)
    stem = osp.join(images, "proteome_replication_zelezniak")
    fig.savefig(stem + ".png", dpi=300)
    savefig_true_size_svg(fig, stem + ".svg")
    print(f"\nfigure: {stem}.svg")


if __name__ == "__main__":
    main()
