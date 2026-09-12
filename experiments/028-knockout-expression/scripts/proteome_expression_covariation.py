# experiments/028-knockout-expression/scripts/proteome_expression_covariation.py
# [[experiments.028-knockout-expression.scripts.proteome_expression_covariation]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-knockout-expression/scripts/proteome_expression_covariation
"""Does the Messner 2023 knockout proteome agree with expression, and with which panel?

Two levels, because the panels differ in what a strain is:

  strain-aligned   Messner and Kemmeren and Nadal-Ribelles all delete single genes of
                   the same collection, so a deletion's protein profile can be set
                   beside its mRNA profile. Per shared deletion, the Pearson between
                   the protein log2 ratio (strain over the 388-replicate HIS3 reference)
                   and the mRNA log2 ratio over the proteins measured in both; and per
                   protein, the Pearson across shared deletions. The earlier EDA on the
                   served graph (2026-07-22) found per-gene median 0.08 and per-strain
                   0.04 for Kemmeren; recomputed here from the LMDBs, and extended to
                   Nadal-Ribelles A.
  gene-aligned     Caudal 2024 profiles natural isolates, not deletions, so no strain is
                   shared with Messner. What can be compared is the gene x gene
                   co-variation: the correlation between two genes across the strains of
                   one panel. Messner (across 4,699 deletions), Caudal (across 943
                   isolates), Kemmeren (across 1,484 deletions) and Nadal A (across
                   deletions with >= 50 cells) each give one gene x gene matrix on the
                   proteins Messner measures; the Spearman between two matrices' upper
                   triangles says whether the same gene pairs co-vary. Also the level
                   comparison: mean log2 protein in the HIS3 reference against mean log2
                   TPM across Caudal isolates, per gene.

Inputs: the LMDBs of proteome_messner2023, microarray_kemmeren2014,
caudal_pantranscriptome2024, nadal_ribelles_perturbseq2025 (control) and the Nadal
pseudobulk recompute. Run from the repo root:
    python experiments/028-knockout-expression/scripts/proteome_expression_covariation.py
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
from cross_study_recomputed import (  # noqa: E402
    RECOMPUTED_REL,
    _recomputed_profiles,
    _resolver,
)
from cross_study_structure import _matrix, _upper  # noqa: E402

from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome  # noqa: E402
from torchcell.utils import (  # noqa: E402
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    panel_label,
    savefig_true_size_svg,
)
from torchcell.utils.paths import experiment_results_dir  # noqa: E402

LMDB_PROTEOME = "data/torchcell/proteome_messner2023/processed/lmdb"
LMDB_CAUDAL = "data/torchcell/caudal_pantranscriptome2024/processed/lmdb"
MIN_SHARED_FOR_STRAIN_R = 200  # proteins a strain needs on both sides
MIN_STRAINS_FOR_GENE_R = 100  # strains a protein needs on both sides
MIN_STRAINS_FOR_COVAR = 0.8  # fraction of a panel's strains a gene must be measured in
NADAL_MIN_CELLS = 50


def _messner(records: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.Series]:
    """Deletion x protein log2 ratio over the HIS3 reference, averaging duplicate
    strains of one ORF; and the reference log2 abundance.
    """
    ref = records[0]["reference"]["phenotype_reference"]["protein_abundance"]
    ref_log = pd.Series({g: np.log2(v) for g, v in ref.items() if v > 0})
    rows: dict[str, list[pd.Series]] = {}
    for rec in records:
        perts = rec["experiment"]["genotype"]["perturbations"]
        if len(perts) != 1:
            continue
        orf = perts[0]["systematic_gene_name"]
        ab = rec["experiment"]["phenotype"]["protein_abundance"]
        s = pd.Series({g: np.log2(v) for g, v in ab.items() if v > 0}) - ref_log
        rows.setdefault(orf, []).append(s.dropna())
    m = pd.DataFrame({o: pd.concat(v, axis=1).mean(axis=1) for o, v in rows.items()}).T
    return m, ref_log


def _caudal(records: list[dict[str, Any]], resolve: Any) -> pd.DataFrame:
    """Isolate x gene log2(TPM + 1), gene ids resolved to S288C ORFs where they are
    S288C names (the accessory-gene ids of the pan-genome do not resolve and are
    dropped).
    """
    rows = {}
    for i, rec in enumerate(records):
        tpm = rec["experiment"]["phenotype"]["expression_tpm"]
        rows[f"isolate_{i}"] = {g: np.log2(v + 1.0) for g, v in tpm.items()}
    m = pd.DataFrame(rows).T
    cols = {}
    for c in m.columns:
        name = c[1:] if c.startswith("X") else c
        orf = resolve(name)
        if orf is not None and orf not in cols.values():
            cols[c] = orf
    m = m[list(cols)]
    m.columns = [cols[c] for c in m.columns]
    return m


def _gene_covar(m: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    """Gene x gene Pearson across strains, genes measured in most strains."""
    sub = m[[g for g in genes if g in m.columns]]
    keep = sub.notna().mean() >= MIN_STRAINS_FOR_COVAR
    sub = sub.loc[:, keep]
    return sub.corr(min_periods=int(MIN_STRAINS_FOR_COVAR * len(sub)))


def _strain_aligned(P: pd.DataFrame, E: pd.DataFrame) -> dict[str, Any]:
    strains = sorted(set(P.index) & set(E.index))
    genes = sorted(set(P.columns) & set(E.columns))
    A = P.loc[strains, genes]
    B = E.loc[strains, genes]
    # z-score per gene across strains, as the earlier EDA did, so a gene's scale in
    # either modality does not dominate.
    A = (A - A.mean()) / A.std()
    B = (B - B.mean()) / B.std()
    both = A.notna() & B.notna()
    per_strain = pd.Series(np.nan, index=strains)
    for s in strains:
        m = both.loc[s]
        if m.sum() >= MIN_SHARED_FOR_STRAIN_R:
            per_strain[s] = np.corrcoef(A.loc[s, m], B.loc[s, m])[0, 1]
    per_gene = pd.Series(np.nan, index=genes)
    for g in genes:
        m = both[g]
        if m.sum() >= MIN_STRAINS_FOR_GENE_R:
            per_gene[g] = np.corrcoef(A.loc[m, g], B.loc[m, g])[0, 1]
    # The deleted gene's own protein: is it the most reduced protein in its own strain?
    self_rank = {}
    for s in strains:
        if s in P.columns and np.isfinite(P.loc[s, s]):
            row = P.loc[s].dropna()
            self_rank[s] = (int((row < row[s]).sum()) + 1, len(row), float(row[s]))
    sr = (
        pd.DataFrame(self_rank, index=["rank", "n", "value"]).T
        if self_rank
        else pd.DataFrame()
    )
    return {
        "n_shared_strains": len(strains),
        "n_shared_genes": len(genes),
        "per_strain_median_r": float(per_strain.median()),
        "per_strain_iqr": [
            float(per_strain.quantile(0.25)),
            float(per_strain.quantile(0.75)),
        ],
        "per_strain_n": int(per_strain.notna().sum()),
        "per_gene_median_r": float(per_gene.median()),
        "per_gene_iqr": [
            float(per_gene.quantile(0.25)),
            float(per_gene.quantile(0.75)),
        ],
        "per_gene_n": int(per_gene.notna().sum()),
        "per_gene_frac_above_0.3": float((per_gene > 0.3).mean()),
        "self_protein": {
            "n": int(len(sr)),
            "frac_rank_1": float((sr["rank"] == 1).mean()) if len(sr) else float("nan"),
            "frac_bottom_1pct": float((sr["rank"] / sr["n"] <= 0.01).mean())
            if len(sr)
            else float("nan"),
            "median_log2": float(sr["value"].median()) if len(sr) else float("nan"),
        },
        "_per_strain": per_strain,
        "_per_gene": per_gene,
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

    P, ref_log = _messner(_load_lmdb(osp.join(data_root, LMDB_PROTEOME)))
    kem, _, _ = _profiles(_load_lmdb(osp.join(data_root, LMDB["kemmeren"])))
    K = _matrix(kem, sorted(kem))
    nadA, cells, _ = _recomputed_profiles(
        data_root, "pseudobulk_log2fc", resolve, genotypes
    )
    N = _matrix(nadA, sorted(o for o in nadA if cells.get(o, 0) >= NADAL_MIN_CELLS))
    C = _caudal(_load_lmdb(osp.join(data_root, LMDB_CAUDAL)), resolve)
    print(
        f"Messner {P.shape}, Kemmeren {K.shape}, Nadal A (>= {NADAL_MIN_CELLS} cells) {N.shape}, Caudal {C.shape}"
    )

    out: dict[str, Any] = {
        "generated_by": "experiments/028-knockout-expression/scripts/proteome_expression_covariation.py",
        "shapes": {
            "messner": list(P.shape),
            "kemmeren": list(K.shape),
            "nadalA": list(N.shape),
            "caudal": list(C.shape),
        },
    }

    # ---- strain-aligned --------------------------------------------------------
    sa_k = _strain_aligned(P, K)
    sa_n = _strain_aligned(P, N)
    out["strain_aligned"] = {
        "messner_vs_kemmeren": {k: v for k, v in sa_k.items() if not k.startswith("_")},
        "messner_vs_nadalA": {k: v for k, v in sa_n.items() if not k.startswith("_")},
    }
    print("strain-aligned:", json.dumps(out["strain_aligned"], indent=1))

    # ---- gene-aligned co-variation ----------------------------------------------
    genes = sorted(set(P.columns) & set(C.columns) & set(K.columns))
    cov = {
        "messner": _gene_covar(P, genes),
        "caudal": _gene_covar(C, genes),
        "kemmeren": _gene_covar(K, genes),
        "nadalA": _gene_covar(N, genes),
    }
    for k, v in cov.items():
        print(f"gene covariation matrix {k}: {v.shape}")
    pairs = [
        ("messner", "caudal"),
        ("messner", "kemmeren"),
        ("messner", "nadalA"),
        ("kemmeren", "caudal"),
        ("kemmeren", "nadalA"),
        ("caudal", "nadalA"),
    ]
    mant: dict[str, Any] = {}
    tri: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for a, b in pairs:
        g = cov[a].index.intersection(cov[b].index)
        ua, ub = _upper(cov[a].loc[g, g]), _upper(cov[b].loc[g, g])
        ok = np.isfinite(ua) & np.isfinite(ub)
        ua, ub = ua[ok], ub[ok]
        rho = spearmanr(ua, ub).correlation
        top = ua >= np.quantile(ua, 0.99)
        mant[f"{a}_vs_{b}"] = {
            "n_genes": int(len(g)),
            "n_pairs": int(ok.sum()),
            "spearman": float(rho),
            "median_b_for_top1pct_a": float(np.median(ub[top])),
            "median_b_rest": float(np.median(ub[~top])),
        }
        tri[f"{a}_vs_{b}"] = (ua, ub)
        print(
            f"co-variation {a} vs {b}: genes {len(g)}, Spearman {rho:.3f}, "
            f"top-1% {a} pairs have median {b} r {np.median(ub[top]):.3f} vs {np.median(ub[~top]):.3f}"
        )
    out["gene_covariation"] = mant

    # ---- level: HIS3 protein abundance against Caudal mean expression -------------
    lev = pd.DataFrame(
        {"protein_log2": ref_log, "caudal_mean_log2tpm": C.mean()}
    ).dropna()
    rho_lev = spearmanr(lev["protein_log2"], lev["caudal_mean_log2tpm"]).correlation
    out["level"] = {
        "n_genes": int(len(lev)),
        "spearman_protein_vs_caudal_mean_tpm": float(rho_lev),
    }
    # Variability: protein sd across deletions vs mRNA sd across isolates and across deletions.
    var = pd.DataFrame(
        {
            "messner_sd": P.std(),
            "caudal_sd": C.std(),
            "kemmeren_sd": K.std(),
            "nadalA_sd": N.std(),
        }
    ).dropna(subset=["messner_sd"])
    out["variability"] = {
        f"messner_vs_{k}": float(
            spearmanr(var["messner_sd"], var[f"{k}_sd"], nan_policy="omit").correlation
        )
        for k in ("caudal", "kemmeren", "nadalA")
    }
    print(
        "level Spearman",
        round(rho_lev, 3),
        "| variability",
        json.dumps(out["variability"]),
    )
    with open(osp.join(results_dir, "proteome_expression_covariation.json"), "w") as f:
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
    col = {
        "kemmeren": PLOT_PALETTE[3],
        "nadalA": PLOT_PALETTE[0],
        "caudal": PLOT_PALETTE[2],
        "messner": PLOT_PALETTE[1],
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

    # a. per-strain protein vs mRNA
    ax = axes[0, 0]
    bins = np.linspace(-0.5, 0.8, 40)
    filled(
        ax,
        sa_k["_per_strain"].dropna(),
        col["kemmeren"],
        bins,
        f"Kemmeren (med {sa_k['per_strain_median_r']:.2f}, n = {sa_k['per_strain_n']})",
    )
    filled(
        ax,
        sa_n["_per_strain"].dropna(),
        col["nadalA"],
        bins,
        f"Nadal A (med {sa_n['per_strain_median_r']:.2f}, n = {sa_n['per_strain_n']})",
    )
    ax.axvline(0, color="black", lw=0.5, ls="--")
    ax.set_xlabel("per-deletion r, Messner protein vs mRNA (z-scored)")
    ax.set_ylabel("density")
    ax.set_title("strain-aligned: one deletion, protein vs mRNA")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.6)
    ax.legend(loc="upper right", **legend_kw)

    # b. per-gene protein vs mRNA across deletions
    ax = axes[0, 1]
    filled(
        ax,
        sa_k["_per_gene"].dropna(),
        col["kemmeren"],
        bins,
        f"Kemmeren (med {sa_k['per_gene_median_r']:.2f}, n = {sa_k['per_gene_n']})",
    )
    filled(
        ax,
        sa_n["_per_gene"].dropna(),
        col["nadalA"],
        bins,
        f"Nadal A (med {sa_n['per_gene_median_r']:.2f}, n = {sa_n['per_gene_n']})",
    )
    ax.axvline(0, color="black", lw=0.5, ls="--")
    ax.set_xlabel("per-protein r across shared deletions, protein vs mRNA")
    ax.set_ylabel("density")
    ax.set_title("strain-aligned: one protein across deletions")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.6)
    ax.legend(loc="upper right", **legend_kw)

    # c. level
    ax = axes[0, 2]
    ax.hexbin(
        lev["caudal_mean_log2tpm"],
        lev["protein_log2"],
        gridsize=40,
        bins="log",
        cmap="Greys",
        linewidths=0,
    )
    ax.set_xlabel("Caudal: mean log2(TPM + 1) over isolates")
    ax.set_ylabel("Messner: log2 protein, HIS3 reference")
    ax.set_title(f"level per gene, Spearman {rho_lev:.2f} (n = {len(lev):,})")

    # d, e, f. gene co-variation
    for ax, key in zip(
        axes[1], ["messner_vs_caudal", "messner_vs_kemmeren", "messner_vs_nadalA"]
    ):
        ua, ub = tri[key]
        ax.hexbin(ua, ub, gridsize=50, bins="log", cmap="Greys", linewidths=0)
        ax.axhline(0, color="black", lw=0.5, ls="--")
        ax.axvline(0, color="black", lw=0.5, ls="--")
        b = key.split("_vs_")[1]
        name = {
            "caudal": "Caudal isolates",
            "kemmeren": "Kemmeren deletions",
            "nadalA": "Nadal A deletions",
        }[b]
        ax.set_xlabel("gene-pair r across Messner deletions, protein")
        ax.set_ylabel(f"r across {name}, mRNA")
        m = mant[key]
        ax.set_title(
            f"co-variation, Spearman {m['spearman']:.2f} ({m['n_genes']:,} genes)"
        )

    for ax in axes.flat:
        for sp in ax.spines.values():
            sp.set_visible(True)
    fig.subplots_adjust(
        left=0.06, right=0.98, bottom=0.09, top=0.92, wspace=0.42, hspace=0.6
    )
    for ax, letter in zip(axes.flat, "abcdef"):
        panel_label(ax, letter)
    stem = osp.join(images, "proteome_expression_covariation")
    fig.savefig(stem + ".png", dpi=300)
    savefig_true_size_svg(fig, stem + ".svg")
    print(f"\nfigure: {stem}.svg")
    print(f"results: {osp.join(results_dir, 'proteome_expression_covariation.json')}")


if __name__ == "__main__":
    main()
