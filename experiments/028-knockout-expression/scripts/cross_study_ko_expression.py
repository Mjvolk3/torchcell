# experiments/028-knockout-expression/scripts/cross_study_ko_expression.py
# [[experiments.028-knockout-expression.scripts.cross_study_ko_expression]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-knockout-expression/scripts/cross_study_ko_expression
"""Do three knockout-expression studies measure the same thing?

Kemmeren 2014 (1,484 single deletions, two-color microarray, log2 ratio vs a wild-type
pool), Sameith 2015 (82 single deletions, same platform and lab, all of them also in
Kemmeren) and Nadal-Ribelles 2025 (about 3,100 single deletions, single-cell RNA-seq
pseudobulked per genotype, log2 fold change vs the wild type profiled in the same run) are
the three genome-scale single-deletion expression panels in the database. Before they are
pooled into one training set, this asks, on the deletions they share, how far two studies
agree on a strain's profile and on a reporter gene's response across strains, and what
that agreement implies for the per-reporter Pearson ceiling of a pooled panel.

Only the base-YPD (control) records of Nadal-Ribelles are used. Its 0.4 M NaCl records
are a different environment and belong to a later question.

Four measurements, all on the pairwise-shared deletions and the reporter genes measured in
both studies of a pair:

  1. COVERAGE. Deletion-gene and reporter-gene overlaps of the three studies.
  2. PER-STRAIN AGREEMENT. For each shared deletion, the Pearson across reporter genes
     between its two profiles. This is the quantity a "does the model reproduce this
     strain" reader has in mind, and the one single-cell papers report; it is dominated
     by the strains with large effects.
  3. PER-REPORTER AGREEMENT. For each reporter gene, the Pearson across shared strains
     between its two response vectors. This is the test-retest reliability rel_g of the
     campaign's metric (`val/expression/pearson_per_feature`), and sqrt(rel_g) is the
     score a perfect predictor of the shared signal would reach on the other study
     (expression_ceiling_replicate.py, Route B). With ~1,400 Kemmeren x Nadal strains it
     is estimated on 17x more strains than the 82-strain Kemmeren x Sameith pair.
  4. SCALE. The slope and spread of one study's log2 values against the other's on the
     shared (strain, gene) cells, since a pooled panel that mixes a microarray ratio
     with a pseudobulk fold change needs to know whether the two are on one scale.

Nadal-Ribelles carries replacement strains that delete the same ORF (`bc_YBR020W-1`,
`-2`); for a per-ORF comparison their profiles are averaged and the count is reported.
Its per-genotype gene set is ragged (a gene with zero counts in a comparison is absent,
not zero), so every per-strain correlation is over the genes present in both records.

THE SENTINEL VALUES. The stored Nadal-Ribelles numbers are scanpy `logfoldchanges` from
a Wilcoxon `rank_genes_groups` run (loader docstring), and that statistic is
log2((expm1(mean_group) + 1e-9) / (expm1(mean_rest) + 1e-9)) over log-normalized means,
which returns about +-23 to +-33 whenever one group's mean is zero. Measured on the
control records: 20.3% of all stored cells have |log2 FC| > 20, and 22.6% of genes carry
that value in more than half of their records. Those are not fold changes; they are
"one side had no counts". This script treats |log2 FC| >= SENTINEL as absent, reports
the rate, and scores agreement on what remains. A pooled panel would need the fold
changes recomputed from the single-cell object with a pseudocount and the paper's own
mean-count filter (Methods: independent filtering at mean normalized counts of 1.27 in
control), which is not in the raw mirror (only the DEG tables and the per-genotype
summary are).

A fifth measurement addresses the reading "the microarray effects are mostly noise, so a
whole-profile correlation is a correlation of noise with noise": STRONG RESPONDERS. Per
shared strain, over the reporters where |Kemmeren log2 ratio| > 1, the Pearson with the
Nadal value and the fraction of agreeing signs. Sign agreement at 0.5 is chance.

Outputs: `results/cross_study_ko_expression.json` (every number below) and one figure in
`ASSET_IMAGES_DIR/028-knockout-expression/`.

Run from the repo root:
    python experiments/028-knockout-expression/scripts/cross_study_ko_expression.py
"""

from __future__ import annotations

import json
import os
import os.path as osp
import pickle
from collections import defaultdict
from itertools import combinations
from typing import Any

import lmdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from matplotlib.ticker import (
    FixedLocator,
    MultipleLocator,
    NullFormatter,
    ScalarFormatter,
)  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from torchcell.utils import (  # noqa: E402
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    panel_label,
    savefig_true_size_svg,
)
from torchcell.utils.paths import experiment_results_dir  # noqa: E402

LMDB = {
    "kemmeren": "data/torchcell/microarray_kemmeren2014/processed/lmdb",
    "sameith": "data/torchcell/sm_microarray_sameith2015/processed/lmdb",
    "nadal": "data/torchcell/nadal_ribelles_perturbseq2025/processed/lmdb",
}
LABEL = {
    "kemmeren": "Kemmeren 2014",
    "sameith": "Sameith 2015",
    "nadal": "Nadal-Ribelles 2025",
}
PAIRS = [("kemmeren", "nadal"), ("sameith", "nadal"), ("kemmeren", "sameith")]
# Reporter genes with fewer shared strains than this in a pair are not scored: a
# per-reporter correlation over a handful of strains is a coin flip.
MIN_STRAINS_PER_REPORTER = 20
MIN_GENES_PER_STRAIN = 200
# |log2 FC| at or above this is scanpy's zero-mean sentinel, not a measurement (see the
# docstring). Real single-deletion effects in these panels top out near |6|.
SENTINEL = 20.0
# Strong responders: reporters whose Kemmeren log2 ratio is beyond this in a strain.
STRONG = 1.0
MIN_STRONG_PER_STRAIN = 10


def _load_lmdb(path: str) -> list[dict[str, Any]]:
    env = lmdb.open(path, readonly=True, lock=False, subdir=True)
    records: list[dict[str, Any]] = []
    with env.begin() as txn:
        for _, value in txn.cursor():
            records.append(pickle.loads(value))
    env.close()
    return records


def _is_control(record: dict[str, Any]) -> bool:
    """True when the environment carries no perturbation (base medium)."""
    return not record["experiment"]["environment"].get("perturbations")


def _profiles(
    records: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, float]], int, dict[str, Any]]:
    """Per-deletion log2 profile, averaging records that delete the same single ORF.

    Returns the profiles, how many ORFs had more than one record (replacement strains
    in Nadal-Ribelles; none expected in the microarray sets), and the sentinel
    accounting: cells at |log2 FC| >= SENTINEL are dropped as absent, and the fraction
    of cells and of genes affected is reported.
    """
    per_orf: dict[str, list[dict[str, float]]] = defaultdict(list)
    n_cells = 0
    n_sentinel = 0
    gene_total: dict[str, int] = defaultdict(int)
    gene_sentinel: dict[str, int] = defaultdict(int)
    for rec in records:
        perts = rec["experiment"]["genotype"]["perturbations"]
        if len(perts) != 1:
            continue
        raw = rec["experiment"]["phenotype"]["expression_log2_ratio"]
        kept: dict[str, float] = {}
        for g, v in raw.items():
            n_cells += 1
            gene_total[g] += 1
            if abs(v) >= SENTINEL:
                n_sentinel += 1
                gene_sentinel[g] += 1
                continue
            kept[g] = v
        per_orf[perts[0]["systematic_gene_name"]].append(kept)
    gene_rate = np.array([gene_sentinel[g] / gene_total[g] for g in gene_total])
    sentinel = {
        "threshold": SENTINEL,
        "frac_cells": (n_sentinel / n_cells) if n_cells else 0.0,
        "frac_genes_rate_above_0.5": float((gene_rate > 0.5).mean())
        if len(gene_rate)
        else 0.0,
        "frac_genes_rate_above_0.1": float((gene_rate > 0.1).mean())
        if len(gene_rate)
        else 0.0,
    }
    out: dict[str, dict[str, float]] = {}
    n_multi = 0
    for orf, profs in per_orf.items():
        if len(profs) == 1:
            out[orf] = profs[0]
            continue
        n_multi += 1
        acc: dict[str, list[float]] = defaultdict(list)
        for p in profs:
            for g, v in p.items():
                acc[g].append(v)
        out[orf] = {g: float(np.mean(v)) for g, v in acc.items()}
    return out, n_multi, sentinel


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _pair(
    a: dict[str, dict[str, float]], b: dict[str, dict[str, float]]
) -> dict[str, Any]:
    shared = sorted(set(a) & set(b))
    genes_a: set[str] = set().union(*a.values())
    genes_b: set[str] = set().union(*b.values())
    reporters = sorted(genes_a & genes_b)
    gi = {g: j for j, g in enumerate(reporters)}

    # Dense matrices over shared strains x shared reporters, NaN where absent.
    A = np.full((len(shared), len(reporters)), np.nan)
    B = np.full_like(A, np.nan)
    for i, orf in enumerate(shared):
        for g, v in a[orf].items():
            j = gi.get(g)
            if j is not None:
                A[i, j] = v
        for g, v in b[orf].items():
            j = gi.get(g)
            if j is not None:
                B[i, j] = v
    both = np.isfinite(A) & np.isfinite(B)

    per_strain = np.full(len(shared), np.nan)
    for i in range(len(shared)):
        m = both[i]
        if m.sum() >= MIN_GENES_PER_STRAIN:
            per_strain[i] = _pearson(A[i, m], B[i, m])

    per_reporter = np.full(len(reporters), np.nan)
    n_per_reporter = both.sum(axis=0)
    for j in range(len(reporters)):
        m = both[:, j]
        if m.sum() >= MIN_STRAINS_PER_REPORTER:
            per_reporter[j] = _pearson(A[m, j], B[m, j])

    # Strong responders: agreement where study a's effect is large.
    strong_r = np.full(len(shared), np.nan)
    strong_sign = np.full(len(shared), np.nan)
    strong_n = np.zeros(len(shared), dtype=int)
    for i in range(len(shared)):
        m = both[i] & (np.abs(np.nan_to_num(A[i])) > STRONG)
        strong_n[i] = int(m.sum())
        if m.sum() >= MIN_STRONG_PER_STRAIN:
            strong_r[i] = _pearson(A[i, m], B[i, m])
            strong_sign[i] = float((np.sign(A[i, m]) == np.sign(B[i, m])).mean())

    # Scale: total least squares would be symmetric, but the question a pooled panel
    # asks is "what multiplies b to put it on a's scale", so plain OLS of a on b.
    x = B[both]
    y = A[both]
    slope = float(np.polyfit(x, y, 1)[0]) if len(x) > 2 else float("nan")
    cell_r = _pearson(x, y)

    ok_s = np.isfinite(per_strain)
    ok_r = np.isfinite(per_reporter)
    rel = per_reporter[ok_r]
    ceiling = np.sqrt(np.clip(rel, 0, None))
    return {
        "n_shared_deletions": len(shared),
        "n_shared_reporters": len(reporters),
        "n_cells_both": int(both.sum()),
        "per_strain": {
            "n": int(ok_s.sum()),
            "median_r": float(np.nanmedian(per_strain)),
            "mean_r": float(np.nanmean(per_strain)),
            "iqr": [
                float(np.nanquantile(per_strain, 0.25)),
                float(np.nanquantile(per_strain, 0.75)),
            ],
            "frac_above_0.3": float((per_strain[ok_s] > 0.3).mean()),
            "values": [float(v) for v in per_strain],
            "deletions": shared,
        },
        "per_reporter": {
            "n": int(ok_r.sum()),
            "min_strains": MIN_STRAINS_PER_REPORTER,
            "median_strains_per_reporter": float(np.median(n_per_reporter[ok_r])),
            "median_r": float(np.median(rel)),
            "mean_r": float(rel.mean()),
            "iqr": [float(np.quantile(rel, 0.25)), float(np.quantile(rel, 0.75))],
            "mean_sqrt_r_ceiling": float(ceiling.mean()),
            "median_sqrt_r_ceiling": float(np.median(ceiling)),
            "frac_r_above_0.3": float((rel > 0.3).mean()),
            "values": [float(v) for v in per_reporter],
            "reporters": reporters,
        },
        "strong_responders": {
            "threshold_abs_log2_a": STRONG,
            "min_per_strain": MIN_STRONG_PER_STRAIN,
            "n_strains": int(np.isfinite(strong_r).sum()),
            "median_r": float(np.nanmedian(strong_r)),
            "iqr_r": [
                float(np.nanquantile(strong_r, 0.25)),
                float(np.nanquantile(strong_r, 0.75)),
            ],
            "median_sign_agreement": float(np.nanmedian(strong_sign)),
            "frac_strains_r_above_0.3": float(
                (strong_r[np.isfinite(strong_r)] > 0.3).mean()
            ),
            "values_r": [float(v) for v in strong_r],
            "values_sign": [float(v) for v in strong_sign],
        },
        "scale": {
            "ols_slope_a_on_b": slope,
            "cell_pearson": cell_r,
            "sd_a": float(np.nanstd(y)),
            "sd_b": float(np.nanstd(x)),
        },
    }


def _sd_of_values(profiles: dict[str, dict[str, float]]) -> float:
    vals = np.concatenate(
        [np.fromiter(p.values(), dtype=float) for p in profiles.values()]
    )
    return float(np.std(vals))


def _strain_sd(profiles: dict[str, dict[str, float]]) -> dict[str, float]:
    """Per-deletion spread of the log2 profile (sd across reporters)."""
    return {
        o: float(np.std(np.fromiter(p.values(), dtype=float)))
        for o, p in profiles.items()
        if len(p) >= MIN_GENES_PER_STRAIN
    }


def _plain_log_x(ax, ticks):
    """Plain tick labels on a log x axis (no 2 x 10^-1 minor labels)."""
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{v:g}" for v in ticks])


def _figure(
    images: str,
    profiles: dict[str, dict[str, dict[str, float]]],
    pairs: dict[str, Any],
    nadal_raw_values: np.ndarray,
    nadal_cells: dict[str, int],
) -> tuple[str, dict[str, float]]:
    """The 3 x 3 comparison figure; returns the SVG path and the panel statistics.

    Row 1 is about the values themselves (scale, the sentinel, per-strain spread), row 2
    about agreement on the shared deletions, row 3 about what agreement tracks. Kemmeren
    against Nadal-Ribelles is the pair of interest; Kemmeren against Sameith is the
    reference for what two studies of the same platform look like.
    """
    from matplotlib.colors import to_rgba

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
    col = {
        "kemmeren": PLOT_PALETTE[0],
        "sameith": PLOT_PALETTE[2],
        "nadal": PLOT_PALETTE[1],
    }
    short = {"kemmeren": "Kemmeren", "sameith": "Sameith", "nadal": "Nadal-Ribelles"}
    kn = pairs["kemmeren_vs_nadal"]

    fig, axes = plt.subplots(
        3, 3, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(160))
    )
    legend_kw = dict(frameon=True, edgecolor="black", fancybox=False, framealpha=1.0)

    def filled_hist(ax, values, key, bins, label=None, density=True):
        ax.hist(
            values,
            bins=bins,
            histtype="stepfilled",
            facecolor=to_rgba(col[key], 0.45),
            edgecolor="black",
            lw=0.4,
            density=density,
            label=label or short[key],
        )

    def box(ax):
        for s in ax.spines.values():
            s.set_visible(True)

    # a. The values, sentinels removed: the two platforms sit on different scales.
    ax = axes[0, 0]
    bins = np.linspace(-6, 6, 97)
    for key in ("kemmeren", "sameith", "nadal"):
        vals = np.concatenate(
            [np.fromiter(p.values(), dtype=float) for p in profiles[key].values()]
        )
        filled_hist(ax, vals, key, bins, label=f"{short[key]} (sd {vals.std():.2f})")
    ax.set_yscale("log")
    # Headroom of three decades so the framed legend clears the histogram tops.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 3e3)
    ax.set_xlabel("log2 value, sentinels removed")
    ax.set_ylabel("density")
    ax.set_title("value distributions")
    ax.legend(loc="upper right", **legend_kw)

    # b. Nadal-Ribelles as stored: the +-23 sentinel spikes.
    ax = axes[0, 1]
    bins = np.linspace(-35, 35, 141)
    frac = float((np.abs(nadal_raw_values) >= SENTINEL).mean())
    filled_hist(
        ax, nadal_raw_values, "nadal", bins, label=f"{short['nadal']} as stored"
    )
    ax.axvline(-SENTINEL, color="black", lw=0.5, ls="--")
    ax.axvline(SENTINEL, color="black", lw=0.5, ls="--")
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 30)
    ax.set_xlabel("log2 fold change as stored")
    ax.set_ylabel("density")
    ax.set_title(f"stored values, {100 * frac:.1f}% at |value| >= {SENTINEL:g}")
    ax.legend(loc="upper right", **legend_kw)

    # c. Per-strain spread of the profile.
    ax = axes[0, 2]
    sds = {k: _strain_sd(profiles[k]) for k in ("kemmeren", "sameith", "nadal")}
    bins = np.logspace(-1.3, 0.7, 41)
    for key in ("kemmeren", "sameith", "nadal"):
        v = np.fromiter(sds[key].values(), dtype=float)
        filled_hist(ax, v, key, bins, label=f"{short[key]} (median {np.median(v):.2f})")
    ax.set_xscale("log")
    _plain_log_x(ax, [0.1, 0.2, 0.5, 1, 2])
    ax.set_xlabel("per-strain sd of the profile (log2)")
    ax.set_ylabel("density")
    ax.set_title("per-strain effect spread")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.7)
    ax.legend(loc="upper right", **legend_kw)

    # d. Kemmeren against Nadal-Ribelles on every shared (strain, gene) cell.
    ax = axes[1, 0]
    ka, kb = profiles["kemmeren"], profiles["nadal"]
    xs, ys = [], []
    for orf in set(ka) & set(kb):
        pb = kb[orf]
        for g, v in ka[orf].items():
            w = pb.get(g)
            if w is not None:
                xs.append(w)
                ys.append(v)
    x = np.asarray(xs)
    y = np.asarray(ys)
    hb = ax.hexbin(x, y, gridsize=60, bins="log", cmap="Greys", linewidths=0)
    lim = 4.0
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    sl = kn["scale"]["ols_slope_a_on_b"]
    ax.plot([-lim, lim], [-lim, lim], color="black", lw=0.5, ls="--", label="identity")
    ax.plot(
        [-lim, lim],
        [-lim * sl, lim * sl],
        color=PLOT_PALETTE[1],
        lw=0.8,
        label=f"OLS slope {sl:.3f}",
    )
    ax.set_xlabel("Nadal-Ribelles log2 FC")
    ax.set_ylabel("Kemmeren log2 ratio")
    ax.set_title(
        f"shared cells (n = {len(x):,}), r = {kn['scale']['cell_pearson']:.2f}"
    )
    ax.legend(loc="lower right", **legend_kw)
    cbar = fig.colorbar(hb, ax=ax, fraction=0.05, pad=0.02)
    # Title rather than a side label, which would sit on the neighbouring y label.
    cbar.ax.set_title("count", fontsize=6, pad=3)

    # e. Per-strain agreement.
    ax = axes[1, 1]
    bins = np.linspace(-0.4, 1.0, 36)
    series = [
        ("kemmeren_vs_sameith", "per_strain", "values", "Kem vs Sam", "sameith", "-"),
        ("kemmeren_vs_nadal", "per_strain", "values", "Kem vs Nad", "nadal", "-"),
        (
            "kemmeren_vs_nadal",
            "strong_responders",
            "values_r",
            f"Kem vs Nad, |Kem| > {STRONG:g}",
            "nadal",
            "--",
        ),
    ]
    for key, block, field, label, ckey, ls in series:
        v = np.asarray(pairs[key][block][field], dtype=float)
        v = v[np.isfinite(v)]
        if ls == "-":
            filled_hist(ax, v, ckey, bins, label=f"{label} (med {np.median(v):.2f})")
        else:
            ax.hist(
                v,
                bins=bins,
                histtype="step",
                color=col[ckey],
                lw=0.9,
                ls=ls,
                density=True,
                label=f"{label} (med {np.median(v):.2f})",
            )
    ax.set_xlabel("per-strain Pearson r between studies")
    ax.set_ylabel("density")
    ax.set_title("per-strain agreement")
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which="minor", length=0)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.9)
    ax.legend(loc="upper right", **legend_kw)

    # f. Per-reporter agreement (test-retest reliability).
    ax = axes[1, 2]
    for key, label, ckey in (
        ("kemmeren_vs_sameith", "Kem vs Sam", "sameith"),
        ("kemmeren_vs_nadal", "Kem vs Nad", "nadal"),
    ):
        v = np.asarray(pairs[key]["per_reporter"]["values"], dtype=float)
        v = v[np.isfinite(v)]
        filled_hist(ax, v, ckey, bins, label=f"{label} (med {np.median(v):.2f})")
    ax.set_xlabel("per-reporter Pearson r between studies")
    ax.set_ylabel("density")
    ax.set_title("per-reporter agreement (test-retest)")
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which="minor", length=0)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.8)
    ax.legend(loc="upper right", **legend_kw)

    # g. Per-strain agreement against the strain's effect size in Kemmeren.
    ax = axes[2, 0]
    dels = kn["per_strain"]["deletions"]
    r_strain = np.asarray(kn["per_strain"]["values"], dtype=float)
    kem_sd = np.asarray([sds["kemmeren"].get(o, np.nan) for o in dels])
    m = np.isfinite(r_strain) & np.isfinite(kem_sd)
    ax.scatter(kem_sd[m], r_strain[m], s=3, color=col["nadal"], lw=0, alpha=0.7)
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.set_xscale("log")
    _plain_log_x(ax, [0.1, 0.2, 0.3, 0.5])
    ax.set_xlabel("Kemmeren per-strain sd (log2)")
    ax.set_ylabel("per-strain r, Kem vs Nad")
    rho = spearmanr(kem_sd[m], r_strain[m]).correlation
    stats: dict[str, Any] = {"spearman_strain_r_vs_kemmeren_strain_sd": float(rho)}
    ax.set_title(f"agreement vs effect size, Spearman {rho:.2f}")

    # h. Per-strain agreement against the number of cells behind the pseudobulk.
    ax = axes[2, 1]
    cells = np.asarray([nadal_cells.get(o, np.nan) for o in dels], dtype=float)
    m = np.isfinite(r_strain) & np.isfinite(cells) & (cells > 0)
    ax.scatter(cells[m], r_strain[m], s=3, color=col["nadal"], lw=0, alpha=0.7)
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.set_xscale("log")
    _plain_log_x(ax, [10, 30, 100, 300, 1000])
    ax.set_xlabel("Nadal-Ribelles cells per genotype")
    ax.set_ylabel("per-strain r, Kem vs Nad")
    rho = spearmanr(cells[m], r_strain[m]).correlation
    stats["spearman_strain_r_vs_nadal_cells"] = float(rho)
    stats["nadal_cells_per_genotype_median"] = float(np.median(cells[m]))
    ax.set_title(f"agreement vs cell count, Spearman {rho:.2f}")

    # i. Per-reporter agreement against the reporter's variance in Kemmeren.
    ax = axes[2, 2]
    reps = kn["per_reporter"]["reporters"]
    r_rep = np.asarray(kn["per_reporter"]["values"], dtype=float)
    shared = set(dels)
    rep_sd = np.full(len(reps), np.nan)
    for j, g in enumerate(reps):
        vals = [ka[o][g] for o in shared if g in ka[o]]
        if len(vals) >= MIN_STRAINS_PER_REPORTER:
            rep_sd[j] = np.std(vals)
    m = np.isfinite(r_rep) & np.isfinite(rep_sd)
    ax.scatter(rep_sd[m], r_rep[m], s=2, color=col["nadal"], lw=0, alpha=0.5)
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.set_xscale("log")
    _plain_log_x(ax, [0.05, 0.1, 0.2, 0.5, 1])
    ax.set_xlabel("Kemmeren per-reporter sd across shared strains")
    ax.set_ylabel("per-reporter r, Kem vs Nad")
    rho = spearmanr(rep_sd[m], r_rep[m]).correlation
    stats["spearman_reporter_r_vs_kemmeren_reporter_sd"] = float(rho)
    stats["per_strain_sd_median"] = {
        k: float(np.median(np.fromiter(v.values(), dtype=float)))
        for k, v in sds.items()
    }
    ax.set_title(f"reporter agreement vs its variance, Spearman {rho:.2f}")

    for ax in axes.flat:
        box(ax)
    fig.subplots_adjust(
        left=0.06, right=0.98, bottom=0.06, top=0.95, wspace=0.38, hspace=0.62
    )
    for ax, letter in zip(axes.flat, "abcdefghi"):
        panel_label(ax, letter)
    # Stable name: the expression document's `plots` rule converts figures/NAME.pdf from
    # NAME.svg, and a timestamped file cannot be its target.
    stem = osp.join(images, "cross_study_ko_expression")
    fig.savefig(stem + ".png", dpi=300)
    savefig_true_size_svg(fig, stem + ".svg")
    return stem + ".svg", stats


def main() -> None:
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    images = osp.join(os.environ["ASSET_IMAGES_DIR"], "028-knockout-expression")
    os.makedirs(images, exist_ok=True)
    results_dir = experiment_results_dir("028-knockout-expression", __file__)

    raw = {k: _load_lmdb(osp.join(data_root, p)) for k, p in LMDB.items()}
    n_nadal_all = len(raw["nadal"])
    raw["nadal"] = [r for r in raw["nadal"] if _is_control(r)]
    profiles: dict[str, dict[str, dict[str, float]]] = {}
    multi: dict[str, int] = {}
    sentinel: dict[str, Any] = {}
    for k, recs in raw.items():
        profiles[k], multi[k], sentinel[k] = _profiles(recs)
        print(
            f"{LABEL[k]:22s} records {len(recs):5d}  single-deletion ORFs "
            f"{len(profiles[k]):5d}  ORFs with >1 record {multi[k]}  "
            f"sentinel cells {100 * sentinel[k]['frac_cells']:.1f}%"
        )
    print(f"Nadal-Ribelles control records {len(raw['nadal'])} of {n_nadal_all} total")

    dele = {k: set(v) for k, v in profiles.items()}
    reps = {k: set().union(*v.values()) for k, v in profiles.items()}
    coverage: dict[str, Any] = {
        "records": {k: len(v) for k, v in raw.items()},
        "nadal_records_total": n_nadal_all,
        "orfs_with_multiple_records": multi,
        "deletions": {k: len(v) for k, v in dele.items()},
        "reporters": {k: len(v) for k, v in reps.items()},
        "deletions_shared": {
            f"{a}&{b}": len(dele[a] & dele[b]) for a, b in combinations(dele, 2)
        },
        "deletions_all_three": len(dele["kemmeren"] & dele["sameith"] & dele["nadal"]),
        "deletions_union": len(set().union(*dele.values())),
        "reporters_shared": {
            f"{a}&{b}": len(reps[a] & reps[b]) for a, b in combinations(reps, 2)
        },
        "reporters_all_three": len(reps["kemmeren"] & reps["sameith"] & reps["nadal"]),
        "log2_sd": {k: _sd_of_values(v) for k, v in profiles.items()},
        "sentinel": sentinel,
    }
    print(json.dumps({k: v for k, v in coverage.items()}, indent=1))

    pairs: dict[str, Any] = {}
    for a, b in PAIRS:
        key = f"{a}_vs_{b}"
        pairs[key] = _pair(profiles[a], profiles[b])
        p = pairs[key]
        print(
            f"\n== {LABEL[a]} vs {LABEL[b]}: shared deletions {p['n_shared_deletions']}, "
            f"shared reporters {p['n_shared_reporters']}"
        )
        print(
            f"  per-strain r   median {p['per_strain']['median_r']:.3f}  "
            f"IQR [{p['per_strain']['iqr'][0]:.3f}, {p['per_strain']['iqr'][1]:.3f}]  "
            f"n {p['per_strain']['n']}"
        )
        print(
            f"  per-reporter r median {p['per_reporter']['median_r']:.3f}  "
            f"IQR [{p['per_reporter']['iqr'][0]:.3f}, {p['per_reporter']['iqr'][1]:.3f}]  "
            f"n {p['per_reporter']['n']}  -> mean sqrt(r) ceiling "
            f"{p['per_reporter']['mean_sqrt_r_ceiling']:.3f}"
        )
        s = p["strong_responders"]
        print(
            f"  strong responders (|{a}| > {STRONG}): median r {s['median_r']:.3f} "
            f"IQR [{s['iqr_r'][0]:.3f}, {s['iqr_r'][1]:.3f}], median sign agreement "
            f"{s['median_sign_agreement']:.2f}, n strains {s['n_strains']}"
        )
        print(
            f"  scale: OLS slope of {a} on {b} {p['scale']['ols_slope_a_on_b']:.3f}, "
            f"cell r {p['scale']['cell_pearson']:.3f}, sd {p['scale']['sd_a']:.3f} vs "
            f"{p['scale']['sd_b']:.3f}"
        )

    out = {
        "generated_by": "experiments/028-knockout-expression/scripts/cross_study_ko_expression.py",
        "nadal_condition": "control (base YPD) only",
        "min_strains_per_reporter": MIN_STRAINS_PER_REPORTER,
        "min_genes_per_strain": MIN_GENES_PER_STRAIN,
        "coverage": coverage,
        "pairs": {
            k: {
                kk: (
                    {
                        x: y
                        for x, y in vv.items()
                        if x
                        not in (
                            "values",
                            "values_r",
                            "values_sign",
                            "deletions",
                            "reporters",
                        )
                    }
                    if isinstance(vv, dict)
                    else vv
                )
                for kk, vv in v.items()
            }
            for k, v in pairs.items()
        },
    }
    with open(osp.join(results_dir, "cross_study_ko_expression.json"), "w") as f:
        json.dump(out, f, indent=1)
    # Per-strain and per-reporter vectors for the figure and any later table.
    with open(
        osp.join(results_dir, "cross_study_ko_expression_vectors.json"), "w"
    ) as f:
        json.dump(
            {
                k: {
                    "per_strain": dict(
                        zip(v["per_strain"]["deletions"], v["per_strain"]["values"])
                    ),
                    "per_reporter": dict(
                        zip(v["per_reporter"]["reporters"], v["per_reporter"]["values"])
                    ),
                }
                for k, v in pairs.items()
            },
            f,
        )

    # ------------------------------------------------------------------ figure
    nadal_raw_values = np.concatenate(
        [
            np.fromiter(
                r["experiment"]["phenotype"]["expression_log2_ratio"].values(),
                dtype=float,
            )
            for r in raw["nadal"]
        ]
    )
    nadal_cells: dict[str, int] = defaultdict(int)
    for r in raw["nadal"]:
        perts = r["experiment"]["genotype"]["perturbations"]
        n = r["experiment"]["phenotype"].get("n_cells")
        if len(perts) == 1 and n is not None:
            nadal_cells[perts[0]["systematic_gene_name"]] += int(n)
    svg, fig_stats = _figure(
        images, profiles, pairs, nadal_raw_values, dict(nadal_cells)
    )
    out["figure_stats"] = fig_stats
    with open(osp.join(results_dir, "cross_study_ko_expression.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(fig_stats, indent=1))
    print(f"\nfigure: {svg}")
    print(f"results: {osp.join(results_dir, 'cross_study_ko_expression.json')}")


if __name__ == "__main__":
    main()
