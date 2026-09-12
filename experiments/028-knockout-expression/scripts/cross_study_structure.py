# experiments/028-knockout-expression/scripts/cross_study_structure.py
# [[experiments.028-knockout-expression.scripts.cross_study_structure]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-knockout-expression/scripts/cross_study_structure
"""Is there agreement between Kemmeren 2014 and Nadal-Ribelles 2025 at a level the
per-gene correlation misses?

A per-strain Pearson r asks whether two profiles agree gene by gene on one strain. Two
data sets could fail that and still agree on coarser structure: which genes vary at all,
which strains respond at all, which strains resemble each other, and which functional
modules move. Each is tested here, with Kemmeren against Sameith as the same-platform
reference wherever Sameith covers the strains.

  gene variability   sd of a reporter across the shared deletions, Kemmeren against
                     Nadal A (pseudobulk); Spearman. Are the responsive genes the same?
  strain magnitude   Kemmeren's count of changed genes (FC > 1.7, p < 0.05, the paper's
                     rule) against Nadal's within-study reproducibility (same-genotype
                     cross-batch r) and against Nadal's profile sd. Do the strains with a
                     transcriptional phenotype in one study have one in the other?
  strain similarity  the strain x strain correlation matrix within each study on the
                     shared deletions; Spearman between the two matrices' upper
                     triangles (a Mantel-type statistic), and Nadal's similarity of the
                     pairs Kemmeren calls most similar against the rest.
  GO co-annotation   within one study, are deletions annotated to the same GO biological
                     process more similar than other pairs? AUROC of the pair
                     correlation, same-term pairs against the rest. This needs no second
                     study; it asks whether a study's profiles carry function at all.
  ribosome module    mean log2 over cytosolic ribosomal protein genes per strain, the
                     axis Kemmeren reports as the slow-growth signature; Kemmeren against
                     Nadal A, and against Nadal's own per-cell iESR score averaged per
                     genotype.
  best case          per-strain cross-study r restricted to deletions Kemmeren calls
                     responsive AND whose Nadal profile replicates across batches.

Inputs: the three LMDBs, the recomputed Nadal tables (nadal_pseudobulk_recompute.R,
nadal_batch_replication.R, nadal_paper_deleteome_comparison.R). Run from the repo root:
    python experiments/028-knockout-expression/scripts/cross_study_structure.py
"""

from __future__ import annotations

import json
import os
import os.path as osp
import sys
from itertools import combinations
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from scipy.stats import mannwhitneyu, spearmanr  # noqa: E402

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from cross_study_ko_expression import (  # noqa: E402
    LMDB,
    _is_control,
    _load_lmdb,
    _pair,
    _plain_log_x,
    _profiles,
)
from cross_study_recomputed import (  # noqa: E402
    RECOMPUTED_REL,
    _recomputed_profiles,
    _resolver,
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

MIN_STRAINS_PER_GENE = 100  # strains a reporter needs on both sides for its sd
MIN_GENES_FOR_CORR = 300  # genes two strains need in common for a pair correlation
GO_TERM_MIN, GO_TERM_MAX = 5, 100  # term size, in strains of the data set
# Module gene sets by GO term name (direct annotations, is_a-propagated): the
# ribosomal protein set from cellular-component terms naming a cytosolic ribosomal
# subunit, the biogenesis set from process terms naming ribosome biogenesis or rRNA
# processing.
RP_NAMES = (
    "cytosolic large ribosomal subunit",
    "cytosolic small ribosomal subunit",
    "cytosolic ribosome",
)
RIBI_NAMES = (
    "ribosome biogenesis",
    "rRNA processing",
    "ribosomal large subunit biogenesis",
    "ribosomal small subunit biogenesis",
)
RESPONSIVE_MIN = 4  # Kemmeren: responsive when >= 4 genes at FC > 1.7, p < 0.05
REPLICATES_MIN_R = 0.10  # Nadal: cross-batch same-genotype r above this


def _matrix(profiles: dict[str, dict[str, float]], strains: list[str]) -> pd.DataFrame:
    genes = sorted(set().union(*(profiles[s].keys() for s in strains)))
    m = pd.DataFrame(np.nan, index=strains, columns=genes, dtype=float)
    for s in strains:
        p = profiles[s]
        m.loc[s, list(p.keys())] = list(p.values())
    return m


def _strain_corr(m: pd.DataFrame) -> pd.DataFrame:
    """Strain x strain Pearson over genes both strains report."""
    return m.T.corr(min_periods=MIN_GENES_FOR_CORR)


def _upper(c: pd.DataFrame) -> np.ndarray:
    iu = np.triu_indices(len(c), k=1)
    return c.to_numpy()[iu]


def _go_sets(genome: SCerevisiaeGenome) -> dict[str, dict[str, set[str]]]:
    """Per namespace, GO term -> genes; direct annotations propagated up the is_a
    ancestors (the OBO is loaded without part_of, so a part_of parent is not filled).
    """
    dag = genome.go_dag
    out: dict[str, dict[str, set[str]]] = {
        "biological_process": {},
        "cellular_component": {},
        "molecular_function": {},
    }
    for term, genes in genome.go_genes.items():
        if term not in dag:
            continue
        node = dag[term]
        for t in {term} | node.get_all_parents():
            out[dag[t].namespace].setdefault(t, set()).update(genes)
    return out


def _go_pair_auroc(c: pd.DataFrame, go_sets: dict[str, set[str]]) -> dict[str, Any]:
    """Term size is counted genome-wide (GO_TERM_MIN..GO_TERM_MAX genes), so a term is
    specific in the same sense for every data set; a pair is 'same' when both strains
    are annotated to one such term.
    """
    strains = list(c.index)
    idx = {s: i for i, s in enumerate(strains)}
    same = np.zeros((len(strains), len(strains)), dtype=bool)
    n_terms = 0
    for genes in go_sets.values():
        if not (GO_TERM_MIN <= len(genes) <= GO_TERM_MAX):
            continue
        members = [idx[g] for g in genes if g in idx]
        if len(members) >= 2:
            n_terms += 1
            for a, b in combinations(members, 2):
                same[a, b] = same[b, a] = True
    iu = np.triu_indices(len(strains), k=1)
    r = c.to_numpy()[iu]
    s = same[iu]
    ok = np.isfinite(r)
    r, s = r[ok], s[ok]
    if s.sum() < 10 or (~s).sum() < 10:
        return {
            "auroc": float("nan"),
            "n_same": int(s.sum()),
            "n_other": int((~s).sum()),
        }
    u = mannwhitneyu(r[s], r[~s], alternative="greater")
    auroc = float(u.statistic / (s.sum() * (~s).sum()))
    return {
        "auroc": auroc,
        "p_greater": float(u.pvalue),
        "n_same": int(s.sum()),
        "n_other": int((~s).sum()),
        "n_terms": n_terms,
        "median_r_same": float(np.median(r[s])),
        "median_r_other": float(np.median(r[~s])),
    }


def _module_score(
    profiles: dict[str, dict[str, float]], genes: set[str], strains: list[str]
) -> pd.Series:
    out = {}
    for s in strains:
        v = [profiles[s][g] for g in genes if g in profiles[s]]
        out[s] = float(np.mean(v)) if len(v) >= 20 else np.nan
    return pd.Series(out)


def main() -> None:
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    images = osp.join(os.environ["ASSET_IMAGES_DIR"], "028-knockout-expression")
    os.makedirs(images, exist_ok=True)
    results_dir = experiment_results_dir("028-knockout-expression", __file__)
    rec = osp.join(data_root, RECOMPUTED_REL)

    genome = SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
        overwrite=False,
    )
    resolve = _resolver(genome)
    genotypes = pd.read_csv(osp.join(rec, "genotypes.tsv"), sep="\t")
    label_to_orf = dict(zip(genotypes["label"], genotypes["kogene"]))

    kem, _, _ = _profiles(_load_lmdb(osp.join(data_root, LMDB["kemmeren"])))
    sam, _, _ = _profiles(_load_lmdb(osp.join(data_root, LMDB["sameith"])))
    nad_stored, _, _ = _profiles(
        [r for r in _load_lmdb(osp.join(data_root, LMDB["nadal"])) if _is_control(r)]
    )
    nadA, cells, _ = _recomputed_profiles(
        data_root, "pseudobulk_log2fc", resolve, genotypes
    )
    print(
        f"profiles: kemmeren {len(kem)}, sameith {len(sam)}, nadal stored {len(nad_stored)}, nadal A {len(nadA)}"
    )

    # Kemmeren's responsive call and Nadal's within-study reproducibility per ORF.
    resp = pd.read_csv(osp.join(rec, "kemmeren_responsive.tsv"), sep="\t")
    resp = resp[resp["mutant"].str.contains("-del")].dropna(subset=["systematic"])
    kem_nsig = resp.groupby("systematic")["n_sig_fc1p7"].max()
    rep = pd.read_csv(osp.join(rec, "batch_replication_pairs.tsv"), sep="\t")
    rep["orf"] = rep["genotype"].map(label_to_orf)
    nad_rep = rep.dropna(subset=["orf"]).groupby("orf")["r_batch_ref"].median()
    meta = pd.read_csv(osp.join(rec, "genotype_meta_means.tsv"), sep="\t")
    meta["orf"] = meta["genotype"].map(label_to_orf)
    meta = meta.dropna(subset=["orf"]).groupby("orf").first()

    out: dict[str, Any] = {
        "generated_by": "experiments/028-knockout-expression/scripts/cross_study_structure.py"
    }

    # ---- shared strains and dense matrices ---------------------------------------
    shared_kn = sorted(set(kem) & set(nadA))
    shared_ks = sorted(set(kem) & set(sam))
    K_n = _matrix(kem, shared_kn)
    N_n = _matrix(nadA, shared_kn)
    S_n = _matrix(nad_stored, [s for s in shared_kn if s in nad_stored])
    K_s = _matrix(kem, shared_ks)
    S_s = _matrix(sam, shared_ks)
    print(f"shared Kemmeren-Nadal {len(shared_kn)}, Kemmeren-Sameith {len(shared_ks)}")

    # ---- 1. gene variability across strains --------------------------------------
    def gene_sd(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
        genes = a.columns.intersection(b.columns)
        both = a[genes].notna() & b[genes].notna()
        n = both.sum()
        genes = n.index[n >= min(MIN_STRAINS_PER_GENE, int(0.8 * len(a)))]
        sa = a[genes].where(both[genes]).std()
        sb = b[genes].where(both[genes]).std()
        return pd.DataFrame({"sd_a": sa, "sd_b": sb})

    gv_kn = gene_sd(K_n, N_n)
    gv_ks = gene_sd(K_s, S_s)
    # Nadal's per-gene sd is expected to track depth (fewer UMI, noisier ratio); the
    # WT CPM of the gene is the covariate, from the pseudobulk UMI table.
    umi = pd.read_csv(osp.join(rec, "pseudobulk_umi.tsv.gz"), sep="\t", index_col=0)
    umi.index = [resolve(g) or g for g in umi.index]
    umi = umi[~umi.index.duplicated()]
    wt_cpm = umi["WT"] / umi["WT"].sum() * 1e6
    gv_kn["wt_cpm"] = wt_cpm.reindex(gv_kn.index)
    rho_kn = spearmanr(gv_kn["sd_a"], gv_kn["sd_b"]).correlation
    rho_ks = spearmanr(gv_ks["sd_a"], gv_ks["sd_b"]).correlation
    # Partial Spearman controlling log WT CPM (rank residuals).
    g2 = gv_kn.dropna()
    ranks = g2.rank()
    x = np.log10(g2["wt_cpm"] + 1)
    res_a = ranks["sd_a"] - np.polyval(np.polyfit(x, ranks["sd_a"], 1), x)
    res_b = ranks["sd_b"] - np.polyval(np.polyfit(x, ranks["sd_b"], 1), x)
    rho_kn_partial = float(np.corrcoef(res_a, res_b)[0, 1])
    rho_nad_sd_vs_depth = spearmanr(g2["sd_b"], g2["wt_cpm"]).correlation
    out["gene_variability"] = {
        "kemmeren_vs_nadalA": {
            "n_genes": int(len(gv_kn)),
            "spearman": float(rho_kn),
            "spearman_partial_wt_cpm": rho_kn_partial,
            "spearman_nadal_sd_vs_wt_cpm": float(rho_nad_sd_vs_depth),
        },
        "kemmeren_vs_sameith": {"n_genes": int(len(gv_ks)), "spearman": float(rho_ks)},
    }
    print(
        "gene variability Spearman: Kem-NadA",
        round(rho_kn, 3),
        "partial",
        round(rho_kn_partial, 3),
        "| Kem-Sam",
        round(rho_ks, 3),
        "| NadA sd vs depth",
        round(rho_nad_sd_vs_depth, 3),
    )

    # ---- 2. strain magnitude -----------------------------------------------------
    sm = pd.DataFrame(
        {
            "kem_nsig": kem_nsig.reindex(shared_kn),
            "nad_rep": nad_rep.reindex(shared_kn),
            "nad_sd": N_n.std(axis=1),
            "kem_sd": K_n.std(axis=1),
            "cells": pd.Series(cells).reindex(shared_kn),
        }
    )

    def sp(a: str, b: str) -> dict[str, float]:
        d = sm[[a, b]].dropna()
        r = spearmanr(d[a], d[b])
        return {
            "n": int(len(d)),
            "spearman": float(r.correlation),
            "p": float(r.pvalue),
        }

    out["strain_magnitude"] = {
        "kem_nsig_vs_nad_rep": sp("kem_nsig", "nad_rep"),
        "kem_nsig_vs_nad_sd": sp("kem_nsig", "nad_sd"),
        "kem_sd_vs_nad_sd": sp("kem_sd", "nad_sd"),
        "nad_sd_vs_cells": sp("nad_sd", "cells"),
        "nad_rep_vs_cells": sp("nad_rep", "cells"),
    }
    print("strain magnitude:", json.dumps(out["strain_magnitude"]))

    # ---- 3. strain similarity structure ------------------------------------------
    C_kn_k = _strain_corr(K_n)
    C_kn_n = _strain_corr(N_n)
    C_ks_k = _strain_corr(K_s)
    C_ks_s = _strain_corr(S_s)

    def mantel(ca: pd.DataFrame, cb: pd.DataFrame) -> dict[str, Any]:
        a, b = _upper(ca), _upper(cb)
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
        rho = spearmanr(a, b).correlation
        top = a >= np.quantile(a, 0.99)
        u = mannwhitneyu(b[top], b[~top], alternative="greater")
        return {
            "n_pairs": int(ok.sum()),
            "spearman": float(rho),
            "top1pct_a_pairs": int(top.sum()),
            "median_b_top1pct": float(np.median(b[top])),
            "median_b_rest": float(np.median(b[~top])),
            "auroc_top1pct": float(u.statistic / (top.sum() * (~top).sum())),
            "a": a,
            "b": b,
        }

    man_kn = mantel(C_kn_k, C_kn_n)
    man_ks = mantel(C_ks_k, C_ks_s)
    out["strain_similarity"] = {
        "kemmeren_vs_nadalA": {k: v for k, v in man_kn.items() if k not in ("a", "b")},
        "kemmeren_vs_sameith": {k: v for k, v in man_ks.items() if k not in ("a", "b")},
    }
    print(
        "Mantel Spearman: Kem-NadA",
        round(man_kn["spearman"], 3),
        "top1% AUROC",
        round(man_kn["auroc_top1pct"], 3),
        "| Kem-Sam",
        round(man_ks["spearman"], 3),
        round(man_ks["auroc_top1pct"], 3),
    )

    # ---- 4. GO co-annotation within each study ------------------------------------
    go_all = _go_sets(genome)
    go_sets = go_all["biological_process"]
    go_cc = go_all["cellular_component"]
    kem_all = _matrix(kem, sorted(kem))
    nad_all_strains = sorted(o for o in nadA if cells.get(o, 0) >= 50)
    nad_all = _matrix(nadA, nad_all_strains)
    # Kemmeren's responsive strains only, as the paper's own complex/pathway comparison
    # (its Figure 2C) is over responsive mutants; a nonresponsive profile is noise.
    kem_resp = [s for s in kem_all.index if kem_nsig.get(s, 0) >= RESPONSIVE_MIN]
    C_kem_all = _strain_corr(kem_all)
    C_nad_all = _strain_corr(nad_all)
    C_S_n = _strain_corr(S_n)
    go: dict[str, Any] = {}
    for ns_name, sets in (("bp", go_sets), ("cc", go_cc)):
        go[ns_name] = {
            "kemmeren_all": _go_pair_auroc(C_kem_all, sets),
            "kemmeren_responsive": _go_pair_auroc(
                C_kem_all.loc[kem_resp, kem_resp], sets
            ),
            "kemmeren_shared": _go_pair_auroc(C_kn_k, sets),
            "sameith_shared": _go_pair_auroc(C_ks_s, sets),
            "nadalA_shared": _go_pair_auroc(C_kn_n, sets),
            "nadalA_ge50cells": _go_pair_auroc(C_nad_all, sets),
            "nadal_stored_shared": _go_pair_auroc(C_S_n, sets),
        }
    out["go_coannotation"] = go
    for ns_name in ("bp", "cc"):
        for k, v in go[ns_name].items():
            print(
                f"GO {ns_name} pair AUROC {k:22s} {v['auroc']:.3f}  same {v['n_same']}  other {v['n_other']}  terms {v.get('n_terms')}"
            )

    # ---- 5. ribosome module -------------------------------------------------------
    dag = genome.go_dag
    rp: set[str] = set()
    for t, genes in go_cc.items():
        if dag[t].name in RP_NAMES:
            rp |= genes
    ribi: set[str] = set()
    for t, genes in go_sets.items():
        if dag[t].name in RIBI_NAMES:
            ribi |= genes
    print(f"module sizes: ribosomal protein {len(rp)}, ribosome biogenesis {len(ribi)}")
    mod = pd.DataFrame(
        {
            "kem_rp": _module_score(kem, rp, shared_kn),
            "nad_rp": _module_score(nadA, rp, shared_kn),
            "kem_ribi": _module_score(kem, ribi, shared_kn),
            "nad_ribi": _module_score(nadA, ribi, shared_kn),
            "nad_iesr": meta["iESR_Gasch2017_UCell"].reindex(shared_kn),
            "nad_percribo": meta["percRibo"].reindex(shared_kn),
        }
    )
    mod_s = pd.DataFrame(
        {
            "kem_rp": _module_score(kem, rp, shared_ks),
            "sam_rp": _module_score(sam, rp, shared_ks),
        }
    )

    def sp2(df: pd.DataFrame, a: str, b: str) -> dict[str, float]:
        d = df[[a, b]].dropna()
        r = spearmanr(d[a], d[b])
        return {
            "n": int(len(d)),
            "spearman": float(r.correlation),
            "p": float(r.pvalue),
        }

    out["ribosome_module"] = {
        "n_rp_genes": len(rp),
        "n_ribi_genes": len(ribi),
        "kem_rp_vs_nad_rp": sp2(mod, "kem_rp", "nad_rp"),
        "kem_ribi_vs_nad_ribi": sp2(mod, "kem_ribi", "nad_ribi"),
        "kem_rp_vs_nad_iesr": sp2(mod, "kem_rp", "nad_iesr"),
        "kem_rp_vs_nad_percribo": sp2(mod, "kem_rp", "nad_percribo"),
        "kem_rp_vs_sam_rp": sp2(mod_s, "kem_rp", "sam_rp"),
    }
    print("ribosome module:", json.dumps(out["ribosome_module"]))

    # ---- 6. best case: responsive in Kemmeren, replicating in Nadal ----------------
    pair = _pair(kem, nadA)
    r_strain = pd.Series(
        pair["per_strain"]["values"], index=pair["per_strain"]["deletions"]
    )
    bc = pd.DataFrame(
        {
            "r": r_strain,
            "kem_nsig": kem_nsig.reindex(r_strain.index),
            "nad_rep": nad_rep.reindex(r_strain.index),
        }
    ).dropna()
    bc["kem_responsive"] = bc["kem_nsig"] >= RESPONSIVE_MIN
    bc["nad_replicates"] = bc["nad_rep"] >= REPLICATES_MIN_R
    strata: dict[str, Any] = {}
    for kr in (False, True):
        for nr in (False, True):
            d = bc[(bc["kem_responsive"] == kr) & (bc["nad_replicates"] == nr)]["r"]
            strata[f"kem_responsive={kr},nad_replicates={nr}"] = {
                "n": int(len(d)),
                "median_r": float(d.median()) if len(d) else float("nan"),
                "frac_above_0.2": float((d > 0.2).mean()) if len(d) else float("nan"),
            }
    top = bc[bc["kem_responsive"] & bc["nad_replicates"]].sort_values(
        "r", ascending=False
    )
    out["best_case"] = {
        "responsive_min": RESPONSIVE_MIN,
        "replicates_min_r": REPLICATES_MIN_R,
        "strata": strata,
        "top10": top.head(10)
        .round(3)
        .reset_index()
        .rename(columns={"index": "orf"})
        .to_dict("records"),
    }
    print("best case:", json.dumps(strata))

    with open(osp.join(results_dir, "cross_study_structure.json"), "w") as f:
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
    c_nad, c_sam, c_kem = PLOT_PALETTE[0], PLOT_PALETTE[2], PLOT_PALETTE[3]
    fig, axes = plt.subplots(
        2, 3, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(105))
    )

    # a. gene variability
    ax = axes[0, 0]
    ax.scatter(
        gv_ks["sd_a"],
        gv_ks["sd_b"],
        s=2,
        color=c_sam,
        lw=0,
        alpha=0.4,
        label=f"Sameith (Spearman {rho_ks:.2f})",
    )
    ax.scatter(
        gv_kn["sd_a"],
        gv_kn["sd_b"],
        s=2,
        color=c_nad,
        lw=0,
        alpha=0.4,
        label=f"Nadal A (Spearman {rho_kn:.2f})",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("reporter sd across shared strains, Kemmeren")
    ax.set_ylabel("reporter sd, other study")
    ax.set_title("which genes vary")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 8)
    ax.legend(loc="upper left", **legend_kw)

    # b. strain magnitude: Kemmeren changed genes vs Nadal cross-batch replication
    ax = axes[0, 1]
    d = sm[["kem_nsig", "nad_rep"]].dropna()
    ax.scatter(d["kem_nsig"] + 1, d["nad_rep"], s=3, color=c_nad, lw=0, alpha=0.6)
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.set_xscale("log")
    _plain_log_x(ax, [1, 10, 100, 1000])
    ax.set_xlabel("Kemmeren genes changed + 1 (FC > 1.7, p < 0.05)")
    ax.set_ylabel("Nadal same-genotype cross-batch r")
    rho = out["strain_magnitude"]["kem_nsig_vs_nad_rep"]["spearman"]
    ax.set_title(f"which strains respond, Spearman {rho:.2f}")

    # c. strain similarity structure
    ax = axes[0, 2]
    ax.hexbin(
        man_kn["a"], man_kn["b"], gridsize=50, bins="log", cmap="Greys", linewidths=0
    )
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.axvline(0, color="black", lw=0.5, ls="--")
    ax.set_xlabel("strain-pair r within Kemmeren")
    ax.set_ylabel("strain-pair r within Nadal A")
    ax.set_title(
        f"strain similarity, Spearman {man_kn['spearman']:.2f} (Sameith {man_ks['spearman']:.2f})"
    )

    # d. GO co-annotation AUROC
    ax = axes[1, 0]
    keys = [
        "kemmeren_all",
        "kemmeren_responsive",
        "sameith_shared",
        "nadalA_shared",
        "nadalA_ge50cells",
        "nadal_stored_shared",
    ]
    names = [
        "Kem all",
        "Kem responsive",
        "Sam",
        "Nad A shared",
        "Nad A >= 50 cells",
        "Nad stored",
    ]
    cols = [c_kem, c_kem, c_sam, c_nad, c_nad, PLOT_PALETTE[1]]
    xs = np.arange(len(keys))
    ax.bar(
        xs - 0.2,
        [go["cc"][k]["auroc"] for k in keys],
        width=0.4,
        color=cols,
        edgecolor="black",
        lw=0.5,
        label="same cellular component (complex)",
    )
    ax.bar(
        xs + 0.2,
        [go["bp"][k]["auroc"] for k in keys],
        width=0.4,
        color=cols,
        edgecolor="black",
        lw=0.5,
        hatch="////",
        label="same biological process",
    )
    ax.axhline(0.5, color="black", lw=0.5, ls="--")
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0.4, 0.85)
    ax.set_ylabel("AUROC, co-annotated pairs vs other pairs")
    ax.set_title("does a profile carry function (within study)")
    ax.legend(loc="upper right", **legend_kw)

    # e. ribosome module
    ax = axes[1, 1]
    d = mod[["kem_rp", "nad_rp"]].dropna()
    r1 = out["ribosome_module"]["kem_rp_vs_nad_rp"]["spearman"]
    ax.scatter(
        d["kem_rp"],
        d["nad_rp"],
        s=3,
        color=c_nad,
        lw=0,
        alpha=0.6,
        label=f"Nadal A (Spearman {r1:.2f})",
    )
    d2 = mod_s.dropna()
    r2 = out["ribosome_module"]["kem_rp_vs_sam_rp"]["spearman"]
    ax.scatter(
        d2["kem_rp"],
        d2["sam_rp"],
        s=3,
        color=c_sam,
        lw=0,
        alpha=0.8,
        label=f"Sameith (Spearman {r2:.2f})",
    )
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.axvline(0, color="black", lw=0.5, ls="--")
    ax.set_xlabel("Kemmeren: mean log2 over ribosomal protein genes")
    ax.set_ylabel("other study: mean log2, same genes")
    ax.set_title(f"ribosomal protein module ({len(rp)} genes)")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + (hi - lo) * 0.5)
    ax.legend(loc="upper left", **legend_kw)

    # f. best case strata
    ax = axes[1, 2]
    groups = [(False, False), (True, False), (False, True), (True, True)]
    labels = ["K-\nN-", "K+\nN-", "K-\nN+", "K+\nN+"]
    rng = np.random.default_rng(0)
    for i, (kr, nr) in enumerate(groups):
        d = bc[(bc["kem_responsive"] == kr) & (bc["nad_replicates"] == nr)][
            "r"
        ].to_numpy()
        ax.scatter(
            i + rng.uniform(-0.25, 0.25, len(d)), d, s=3, color=c_nad, lw=0, alpha=0.5
        )
        if len(d):
            ax.hlines(np.median(d), i - 0.3, i + 0.3, color="black", lw=1.0)
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels)
    ax.set_ylabel("per-strain r, Kemmeren vs Nadal A")
    ax.set_xlabel("K+: Kemmeren responsive; N+: Nadal replicates")
    ns = [strata[f"kem_responsive={kr},nad_replicates={nr}"]["n"] for kr, nr in groups]
    ax.set_title("best case: n = " + ", ".join(str(n) for n in ns))

    for ax in axes.flat:
        for s in ax.spines.values():
            s.set_visible(True)
    fig.subplots_adjust(
        left=0.06, right=0.98, bottom=0.14, top=0.92, wspace=0.42, hspace=0.62
    )
    for ax, letter in zip(axes.flat, "abcdef"):
        panel_label(ax, letter)
    stem = osp.join(images, "cross_study_structure")
    fig.savefig(stem + ".png", dpi=300)
    savefig_true_size_svg(fig, stem + ".svg")
    print(f"\nfigure: {stem}.svg")
    print(f"results: {osp.join(results_dir, 'cross_study_structure.json')}")


if __name__ == "__main__":
    main()
