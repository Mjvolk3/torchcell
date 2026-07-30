# experiments/019-simb-multimodal/scripts/pigment_noise_ceiling.py
# [[experiments.019-simb-multimodal.scripts.pigment_noise_ceiling]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/pigment_noise_ceiling
"""Reproducibility ceilings for the two pigment production targets.

Mirrors ``morphology_noise_ceiling.py`` / ``expression_noise_ceiling.py``, but the two
pigment targets need DIFFERENT ceilings because they are different kinds of measurement.
Getting this right before training is the point: a model at r = 0.30 against a ceiling of
0.35 is a success and against a ceiling of 0.90 is a failure, and only the ceiling tells
you which.

BETAXANTHIN (Cachera 2023) -- a quantitative CRI-SPA corrected fluorescence, reported as
the mean over n colonies with a per-record standard error. So the reported value is
``signal + noise`` where ``Var(noise) = SE^2`` is KNOWN per strain, and the standard
broad-sense-reliability argument applies exactly as in the morphology script:

    reliability = 1 - mean(SE^2) / Var(values across strains)
    ceiling     = sqrt(reliability) = max achievable Pearson vs the noisy target.

BETA-CAROTENE (Ozaydin 2013) -- a SUBJECTIVE ORDINAL colony-colour score on -5..+5. There
is no SE and a Pearson ceiling would be the wrong object: the honest question is whether
independent scorings agree on the ORDER of strains. So the ceiling here is RANK agreement
(Spearman), from two sources:

1. ``visual_score`` vs ``visual_score_min`` on the replicated rows. The loader stores the
   MAX over replicate colonies as ``visual_score`` and the MIN as ``visual_score_min``
   (only when n_replicates > 1), so this is a max-vs-min agreement over the same replicate
   set. It is the comparison the plan specifies, and it is CONSERVATIVE-BIASED IN BOTH
   DIRECTIONS: max and min of one set are positively coupled (inflates rho), while
   max-vs-min is also the widest possible split of that set (deflates rho). Report it, but
   do not read it as a clean split-half reliability.
2. The paper's own independent re-screen: SI sheet ``2ndRoundOfTransformations`` holds 157
   strains re-transformed and re-scored, giving PAIRED ``1st Screen`` / ``2nd Screen``
   values. That is a genuine test-retest -- but it was run on SELECTED TOP HITS, so the
   1st-screen scores are severely range-restricted (they sit almost entirely in 3..5 while
   the full screen spans -5..+5). A raw correlation over a restricted range is deflated
   toward zero for a reason that has nothing to do with scorer reliability, so we also
   report the Thorndike case-2 range-restriction correction using the full-screen SD. The
   corrected value is an ESTIMATE with real assumptions (linearity, homoscedasticity), not
   a measurement; the primary reported ceiling stays (1).

MULLEDER (2016) has ``n_replicates = 1`` and ``metabolite_level_se = None`` for every
strain, so it admits no within-dataset ceiling at all; the honest external check is the
tyrosine-vs-betaxanthin correlation, which this script also reports (it is the mechanistic
premise of the whole transfer experiment).

Sources (sha256-pinned mirrors, see each loader's manifest):
  $DATA_ROOT/data/torchcell/betaxanthin_cachera2023/processed/lmdb
  $DATA_ROOT/data/torchcell/carotenoid_ozaydin2013/processed/lmdb
  $DATA_ROOT/data/torchcell/amino_acid_mulleder2016/processed/lmdb
  $DATA_ROOT/data/torchcell/carotenoid_ozaydin2013/raw/1-s2.0-S109671761200081X-mmc1.xlsx

Run from repo root:
  PYTHONPATH=. python experiments/019-simb-multimodal/scripts/pigment_noise_ceiling.py
"""

from __future__ import annotations

import json
import os
import os.path as osp
import pickle
from typing import Any

import lmdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import pearsonr, spearmanr

from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    savefig_true_size_svg,
)

INK = "#000000"
GRID = "#4A4A4A"
PANEL_H_MM = 52.0
OZAYDIN_SI = "1-s2.0-S109671761200081X-mmc1.xlsx"
RESCREEN_SHEET = "2ndRoundOfTransformations"


def _apply_rc() -> None:
    """Arial 6 pt, editable SVG text, thin axes -- the repo figure standard."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6,
            "axes.titlesize": 6,
            "axes.labelsize": 6,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "svg.fonttype": "none",
            "axes.linewidth": 0.5,
            # torchcell.datasets flips savefig.bbox -> "tight" globally, which re-crops at
            # save time and defeats the strict width template. Pin it back.
            "savefig.bbox": None,
        }
    )


def _box(ax: Any) -> None:
    """Full black border on all four spines, light grid behind the data."""
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_color(INK)
        ax.spines[s].set_linewidth(0.5)
    ax.tick_params(colors=INK, width=0.5, length=2)
    ax.grid(True, alpha=0.15, linewidth=0.4, color=GRID)
    ax.set_axisbelow(True)


def _read_records(lmdb_dir: str) -> list[dict[str, Any]]:
    """Return every experiment record's phenotype + perturbed gene names from an LMDB."""
    env = lmdb.open(lmdb_dir, readonly=True, lock=False, readahead=False)
    out: list[dict[str, Any]] = []
    with env.begin() as txn:
        for _, raw in txn.cursor():
            rec = pickle.loads(raw)
            exp = rec["experiment"]
            out.append(
                {
                    "phenotype": exp["phenotype"],
                    "genes": [
                        p["systematic_gene_name"]
                        for p in exp["genotype"]["perturbations"]
                        if p["perturbation_type"].endswith("deletion")
                    ],
                }
            )
    env.close()
    return out


def betaxanthin_ceiling(data_root: str) -> dict[str, Any]:
    """SE-based broad-sense reliability ceiling for the betaxanthin fluorescence."""
    recs = _read_records(
        osp.join(data_root, "data/torchcell/betaxanthin_cachera2023/processed/lmdb")
    )
    level = np.array([r["phenotype"]["metabolite_level"]["betaxanthin"] for r in recs])
    se = np.array([r["phenotype"]["metabolite_level_se"]["betaxanthin"] for r in recs])
    n = np.array([r["phenotype"]["n_replicates"]["betaxanthin"] for r in recs])
    # n == 1 records carry SE = NaN (std/sqrt(n) is undefined for a single colony); they
    # are excluded from the noise estimate rather than imputed.
    ok = np.isfinite(se)
    total_var = float(np.var(level, ddof=1))
    noise_var = float(np.mean(se[ok] ** 2))
    reliability = max(0.0, min(1.0, 1.0 - noise_var / total_var))
    return {
        "n_records": int(len(recs)),
        "n_with_se": int(ok.sum()),
        "n_replicates_min": int(n.min()),
        "n_replicates_median": float(np.median(n)),
        "n_replicates_max": int(n.max()),
        "value_mean": float(level.mean()),
        "value_sd": float(np.sqrt(total_var)),
        "total_var": total_var,
        "mean_se_squared": noise_var,
        "median_se": float(np.median(se[ok])),
        "reliability": reliability,
        "ceiling_pearson": float(np.sqrt(reliability)),
        "_level": level,
        "_se": se,
        "_genes": [r["genes"] for r in recs],
    }


def beta_carotene_ceiling(data_root: str) -> dict[str, Any]:
    """Rank-agreement ceiling for the ordinal beta-carotene visual score."""
    recs = _read_records(
        osp.join(data_root, "data/torchcell/carotenoid_ozaydin2013/processed/lmdb")
    )
    score = np.array([r["phenotype"]["visual_score"] for r in recs])
    smin = np.array(
        [
            np.nan
            if r["phenotype"]["visual_score_min"] is None
            else r["phenotype"]["visual_score_min"]
            for r in recs
        ]
    )
    nrep = np.array([r["phenotype"]["n_replicates"] for r in recs])
    rep = np.isfinite(smin)
    rho, p_rho = spearmanr(score[rep], smin[rep])
    r_pear, _ = pearsonr(score[rep], smin[rep])

    # Independent re-screen: 157 strains re-transformed and re-scored (SI sheet).
    si = osp.join(data_root, "data/torchcell/carotenoid_ozaydin2013/raw", OZAYDIN_SI)
    resc = pd.read_excel(si, sheet_name=RESCREEN_SHEET, skiprows=2)
    resc = resc.dropna(subset=["1st Screen", "2nd Screen"])
    first = pd.to_numeric(resc["1st Screen"], errors="coerce")
    second = pd.to_numeric(resc["2nd Screen"], errors="coerce")
    keep = first.notna() & second.notna()
    rho2, p_rho2 = spearmanr(first[keep], second[keep])
    r2, _ = pearsonr(first[keep], second[keep])

    # Thorndike case-2 correction for direct range restriction on the 1st screen:
    #   R_corrected = r (S/s) / sqrt(1 + r^2 (S^2/s^2 - 1))
    # with S the full-screen SD and s the SD within the re-screened (top-hit) subset.
    sd_full = float(np.std(score, ddof=1))
    sd_sub = float(np.std(first[keep].to_numpy(dtype=float), ddof=1))
    ratio = sd_full / sd_sub
    r2_corrected = float(r2 * ratio / np.sqrt(1.0 + r2**2 * (ratio**2 - 1.0)))

    return {
        "n_records": int(len(recs)),
        "n_replicates_hist": {
            int(k): int(v) for k, v in zip(*np.unique(nrep, return_counts=True))
        },
        "n_replicated_rows": int(rep.sum()),
        "score_min": float(score.min()),
        "score_max": float(score.max()),
        "replicate_max_vs_min": {
            "n": int(rep.sum()),
            "spearman": float(rho),
            "spearman_p": float(p_rho),
            "pearson": float(r_pear),
        },
        "independent_rescreen": {
            "sheet": RESCREEN_SHEET,
            "n": int(keep.sum()),
            "spearman": float(rho2),
            "spearman_p": float(p_rho2),
            "pearson": float(r2),
            "first_screen_mean": float(first[keep].mean()),
            "second_screen_mean": float(second[keep].mean()),
            "sd_full_screen": sd_full,
            "sd_rescreened_subset": sd_sub,
            "range_restriction_ratio": ratio,
            "pearson_range_restriction_corrected": r2_corrected,
            "caveat": (
                "re-screen covers SELECTED TOP HITS only (1st-screen scores concentrated "
                "in 3..5 of a -5..+5 scale), so the raw value is deflated by range "
                "restriction; the corrected value assumes linearity + homoscedasticity."
            ),
        },
        # PRIMARY: the replicate max-vs-min rank agreement on the 130 replicated rows --
        # the estimate the plan specifies, and the only one computed over unrestricted
        # strains. The re-screen above is reported alongside it, not instead of it.
        "ceiling_spearman": float(rho),
        "ceiling_source": "replicate_max_vs_min",
        "_score": score,
        "_score_min": smin,
        "_rep": rep,
        "_rescreen": (first[keep].to_numpy(), second[keep].to_numpy()),
        "_genes": [r["genes"] for r in recs],
    }


def tyrosine_betaxanthin_check(data_root: str, bx: dict[str, Any]) -> dict[str, Any]:
    """External sanity check: does measured tyrosine track measured betaxanthin?

    Mulleder is n=1 with no SE, so it has no within-dataset ceiling. Its relevance to the
    transfer experiment is instead the MECHANISM: betaxanthin is synthesized from
    tyrosine, so if the two measured quantities have no relationship at all on the shared
    deletions, the tyrosine-transfer hypothesis has no observable basis to begin with.
    Reported per amino acid so tyrosine can be read against its 18 siblings rather than in
    isolation.
    """
    recs = _read_records(
        osp.join(data_root, "data/torchcell/amino_acid_mulleder2016/processed/lmdb")
    )
    aa_keys = sorted(recs[0]["phenotype"]["metabolite_level"])
    aa_by_gene: dict[str, dict[str, float]] = {}
    for r in recs:
        if len(r["genes"]) == 1:
            aa_by_gene[r["genes"][0]] = r["phenotype"]["metabolite_level"]
    bx_by_gene: dict[str, float] = {}
    for genes, val in zip(bx["_genes"], bx["_level"]):
        if len(genes) == 1:
            bx_by_gene[genes[0]] = float(val)
    shared = sorted(set(aa_by_gene) & set(bx_by_gene))
    y = np.array([bx_by_gene[g] for g in shared])
    per_aa: dict[str, dict[str, float]] = {}
    for aa in aa_keys:
        x = np.array([aa_by_gene[g][aa] for g in shared])
        r_p, p_p = pearsonr(x, y)
        rho_s, _ = spearmanr(x, y)
        per_aa[aa] = {
            "pearson": float(r_p),
            "pearson_p": float(p_p),
            "spearman": float(rho_s),
        }
    ranked = sorted(per_aa.items(), key=lambda kv: -abs(kv[1]["pearson"]))
    return {
        "n_shared_deletions": len(shared),
        "per_amino_acid": per_aa,
        "tyrosine_rank_by_abs_pearson": (1 + [k for k, _ in ranked].index("tyrosine")),
        "ranking_by_abs_pearson": [k for k, _ in ranked],
        "_shared": shared,
        "_tyr": np.array([aa_by_gene[g]["tyrosine"] for g in shared]),
        "_bx": y,
    }


def make_figure(
    bx: dict[str, Any], bc: dict[str, Any], tyr: dict[str, Any], out_dir: str
) -> tuple[str, str]:
    """Three-panel ceiling figure: betaxanthin SE, beta-carotene rank, tyrosine check."""
    _apply_rc()
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(PANEL_H_MM)),
        constrained_layout=True,
    )

    # (a) betaxanthin: per-record SE vs value; the SE cloud IS the noise floor.
    ax = axes[0]
    ok = np.isfinite(bx["_se"])
    ax.scatter(
        bx["_level"][ok],
        bx["_se"][ok],
        s=1.0,
        c=PLOT_PALETTE[0],
        linewidths=0,
        rasterized=True,
    )
    ax.set_xlabel("betaxanthin (corrected fluorescence)")
    ax.set_ylabel("per-strain SE")
    ax.set_title(
        f"a  betaxanthin  ceiling r = {bx['ceiling_pearson']:.2f}\n"
        f"reliability {bx['reliability']:.2f}  (n = {bx['n_with_se']})",
        loc="left",
    )
    _box(ax)

    # (b) beta-carotene rank agreement. Filled = the primary estimate (replicate max vs
    # min, unrestricted strains); open = the independent re-screen, drawn on the same axes
    # so its range restriction (1st-screen scores bunched at 3..5) is visible rather than
    # buried in a caption.
    ax = axes[1]
    rng = np.random.default_rng(0)
    rep = bc["_rep"]
    x_rep = bc["_score"][rep]
    y_rep = bc["_score_min"][rep]
    ax.scatter(
        x_rep + rng.normal(0, 0.08, size=x_rep.size),
        y_rep + rng.normal(0, 0.08, size=y_rep.size),
        s=3.0,
        c=PLOT_PALETTE[1],
        linewidths=0,
        label=f"replicate max vs min (n={rep.sum()})",
    )
    first, second = bc["_rescreen"]
    ax.scatter(
        first + rng.normal(0, 0.08, size=first.size),
        second + rng.normal(0, 0.08, size=second.size),
        s=4.0,
        facecolors="none",
        edgecolors=PLOT_PALETTE[5],
        linewidths=0.4,
        label=f"re-screen, top hits (n={bc['independent_rescreen']['n']})",
    )
    lim = [-5.5, 5.5]
    ax.plot(lim, lim, color=GRID, linewidth=0.5, linestyle=":")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("score / 1st screen")
    ax.set_ylabel("replicate min / 2nd screen")
    ax.set_title(
        f"b  beta-carotene  ceiling rho = {bc['ceiling_spearman']:.2f}\n"
        f"re-screen rho = {bc['independent_rescreen']['spearman']:.2f} "
        "(range-restricted)",
        loc="left",
    )
    ax.legend(loc="lower right", frameon=False, fontsize=5, handletextpad=0.3)
    _box(ax)

    # (c) the mechanistic premise: tyrosine vs betaxanthin on shared deletions.
    ax = axes[2]
    ax.scatter(
        tyr["_tyr"], tyr["_bx"], s=1.0, c=PLOT_PALETTE[2], linewidths=0, rasterized=True
    )
    r_tyr = tyr["per_amino_acid"]["tyrosine"]["pearson"]
    ax.set_xlabel("tyrosine (mM)")
    ax.set_ylabel("betaxanthin (corrected fluorescence)")
    ax.set_title(
        f"c  tyrosine vs betaxanthin  r = {r_tyr:.3f}\n"
        f"shared deletions n = {tyr['n_shared_deletions']}  "
        f"(rank {tyr['tyrosine_rank_by_abs_pearson']}/19)",
        loc="left",
    )
    _box(ax)

    os.makedirs(out_dir, exist_ok=True)
    png = osp.join(out_dir, "pigment_noise_ceiling.png")
    svg = osp.join(out_dir, "pigment_noise_ceiling.svg")
    fig.savefig(png, dpi=300, facecolor="white")
    savefig_true_size_svg(fig, svg, facecolor="white")
    plt.close(fig)
    return png, svg


def main() -> None:
    """Compute both ceilings + the tyrosine check, write JSON and the figure."""
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    asset_dir = os.environ["ASSET_IMAGES_DIR"]
    here = osp.dirname(osp.abspath(__file__))
    results_dir = osp.abspath(osp.join(here, "..", "results"))
    os.makedirs(results_dir, exist_ok=True)

    bx = betaxanthin_ceiling(data_root)
    bc = beta_carotene_ceiling(data_root)
    tyr = tyrosine_betaxanthin_check(data_root, bx)

    print("=" * 72)
    print("BETAXANTHIN (Cachera 2023) -- SE-based Pearson ceiling")
    print("=" * 72)
    print(
        f"  records {bx['n_records']} ({bx['n_with_se']} with a finite SE); "
        f"n_replicates min/median/max = {bx['n_replicates_min']}/"
        f"{bx['n_replicates_median']:.0f}/{bx['n_replicates_max']}"
    )
    print(
        f"  Var(values) = {bx['total_var']:.5f}   mean(SE^2) = {bx['mean_se_squared']:.5f}"
    )
    print(
        f"  reliability = {bx['reliability']:.4f}   CEILING r = {bx['ceiling_pearson']:.4f}"
    )

    print("\n" + "=" * 72)
    print("BETA-CAROTENE (Ozaydin 2013) -- ordinal, so the ceiling is RANK agreement")
    print("=" * 72)
    print(f"  records {bc['n_records']}; n_replicates hist {bc['n_replicates_hist']}")
    mm = bc["replicate_max_vs_min"]
    print(
        f"  visual_score(max) vs visual_score_min, n={mm['n']}: "
        f"spearman {mm['spearman']:.4f} (p={mm['spearman_p']:.2e}), pearson {mm['pearson']:.4f}"
    )
    rs = bc["independent_rescreen"]
    print(
        f"  independent re-screen, n={rs['n']}: spearman {rs['spearman']:.4f} "
        f"(p={rs['spearman_p']:.2e}), pearson {rs['pearson']:.4f}"
    )
    print(
        f"    RANGE-RESTRICTED: re-screened top hits sd={rs['sd_rescreened_subset']:.3f} "
        f"vs full-screen sd={rs['sd_full_screen']:.3f} "
        f"(ratio {rs['range_restriction_ratio']:.2f}); mean drifted "
        f"{rs['first_screen_mean']:.2f} -> {rs['second_screen_mean']:.2f}"
    )
    print(
        f"    Thorndike case-2 corrected pearson = "
        f"{rs['pearson_range_restriction_corrected']:.4f}"
    )
    print(
        f"  CEILING rho = {bc['ceiling_spearman']:.4f} (source: {bc['ceiling_source']})"
    )

    print("\n" + "=" * 72)
    print("MULLEDER (2016) -- n_replicates=1, metabolite_level_se=None -> NO ceiling")
    print("=" * 72)
    print(f"  external check on {tyr['n_shared_deletions']} shared deletions:")
    for aa in tyr["ranking_by_abs_pearson"][:5]:
        s = tyr["per_amino_acid"][aa]
        print(f"    {aa:14s} r = {s['pearson']:+.4f}  rho = {s['spearman']:+.4f}")
    t = tyr["per_amino_acid"]["tyrosine"]
    print(
        f"  TYROSINE r = {t['pearson']:+.4f} (p={t['pearson_p']:.2e}), "
        f"rho = {t['spearman']:+.4f}, rank {tyr['tyrosine_rank_by_abs_pearson']}/19 by |r|"
    )

    png, svg = make_figure(bx, bc, tyr, osp.join(asset_dir, "019-simb-multimodal"))
    print(f"\nwrote {png}\nwrote {svg}")

    report = {
        "betaxanthin": {k: v for k, v in bx.items() if not k.startswith("_")},
        "beta_carotene": {k: v for k, v in bc.items() if not k.startswith("_")},
        "mulleder_external_check": {
            k: v for k, v in tyr.items() if not k.startswith("_")
        },
        "figure": {"png": png, "svg": svg},
    }
    out = osp.join(results_dir, "pigment_noise_ceiling.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
