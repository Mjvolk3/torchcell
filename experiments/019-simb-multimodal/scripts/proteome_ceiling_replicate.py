# experiments/019-simb-multimodal/scripts/proteome_ceiling_replicate.py
# [[experiments.019-simb-multimodal.scripts.proteome_ceiling_replicate]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/proteome_ceiling_replicate
"""Replicate-based per-protein Pearson ceiling for the Messner 2023 knockout proteome.

The v14 round scores `val/proteome/pearson_per_feature` at 0.08 to 0.12 and peaks by
epoch 150 on every partition. Whether that is a fifth of what the label allows or most of
it is the question this script answers, with the same estimand as
`expression_ceiling_replicate.py`: per protein, a measurement is x = s + e, and a perfect
predictor of s scores corr(s, x) = sqrt(var_s / (var_s + var_e)) = sqrt(reliability).

Messner measured every knockout once ("Strains were not measured in replicates") against
a HIS3-complemented reference measured 388 times across the same 57 plates, in the same
batch-corrected matrix (`yeast5k_noimpute_wide.csv`, the raw mirror the loader consumes).
So the noise of ONE measurement is read from the 388 reference columns, per protein, on
the log2 scale the model trains on, and the total variance from the 4,699 knockout
columns as log2(strain / reference mean):

  ROUTE W -- WITHIN-STUDY REPLICATE. var_e = var over the 388 HIS3 columns of log2 x;
  var_total = var over the knockout columns of log2 x; reliability = 1 - var_e / var_total
  clipped at 0. This is the ceiling against measurement noise in Messner's own hands.

  ROUTE Z -- CROSS-STUDY. Zelezniak 2018 re-measured 89 of the deletions (three
  replicates, same laboratory, same medium, five years earlier); the per-protein Pearson
  between the two studies across those strains estimates the reliability of a strain's
  protein effect across studies, and its square root the ceiling. Read from
  `experiments/028-knockout-expression/results/proteome_replication_zelezniak.json`
  (median per-protein r 0.086 over 714 proteins).

The two routes bound different things. Route W is what a model trained and validated
INSIDE Messner can reach (the v14 setting). Route Z is what transfers to another
measurement of the same genotype. Both are reported; the gap between them is
study-level variance that is signal within Messner and noise across studies.

A third number places the structure: the rank-k ceiling of the knockout matrix
(`lowrank_output_ceiling.py` construction, train-basis SVD projected onto held-out
strains, per-feature Pearson against the measured values), which says how much of the
per-feature score a model that only gets the first k response directions right can earn.

Run from repo root:
    python experiments/019-simb-multimodal/scripts/proteome_ceiling_replicate.py
"""

from __future__ import annotations

import json
import os
import os.path as osp
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.ticker import MultipleLocator

from torchcell.timestamp import timestamp
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    panel_label,
    savefig_true_size_svg,
)
from torchcell.utils.paths import experiment_results_dir

RAW_DIR = "data/torchcell/proteome_messner2023/raw"
MATRIX = "yeast5k_noimpute_wide.csv"
META = "yeast5k_metadata.csv"
ZELEZNIAK_JSON = (
    "experiments/028-knockout-expression/results/proteome_replication_zelezniak.json"
)
LOWRANK_EXPR_JSON = "experiments/019-simb-multimodal/results/lowrank_output_ceiling.json"
# Best v14 matched-epoch partition mean and the B3 ProtT5 kNN val mean over four splits
# (results/v14_proteome_readout.json), for the realized-fraction report.
OBS_V14 = 0.099
OBS_B3 = 0.073
RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
MIN_KO_PER_PROTEIN = 0.9  # a protein needs >= 90% of knockouts measured for the SVD


def _summarize(name: str, ceiling: np.ndarray, rel: np.ndarray) -> dict[str, Any]:
    ok = np.isfinite(ceiling)
    c = ceiling[ok]
    stats: dict[str, Any] = {
        "n_proteins": int(ok.sum()),
        "mean_ceiling": float(c.mean()),
        "median_ceiling": float(np.median(c)),
        "iqr": [float(np.percentile(c, 25)), float(np.percentile(c, 75))],
        "mean_reliability": float(np.nanmean(rel[ok])),
        "median_reliability": float(np.nanmedian(rel[ok])),
        "frac_above": {
            f"{t:.2f}": float((c > t).mean()) for t in (0.3, 0.5, 0.7, 0.9)
        },
        "v14_frac_of_ceiling": float(OBS_V14 / c.mean()),
        "b3_frac_of_ceiling": float(OBS_B3 / c.mean()),
    }
    print(f"[{name}] n={stats['n_proteins']}")
    print(f"  mean ceiling   : {stats['mean_ceiling']:.4f}")
    print(f"  median ceiling : {stats['median_ceiling']:.4f}  IQR {stats['iqr']}")
    print(f"  mean reliability: {stats['mean_reliability']:.4f}")
    for t, f in stats["frac_above"].items():
        print(f"    proteins with ceiling > {t}: {100 * f:5.1f}%")
    print(f"  v14 {OBS_V14:.3f} -> {100 * stats['v14_frac_of_ceiling']:.1f}% of ceiling")
    return stats


def _rank_ceiling(y: np.ndarray, rng: np.random.Generator) -> list[dict[str, Any]]:
    """Rank-k ceiling on a 90/10 strain split of the complete-case matrix y (z per column)."""
    n = y.shape[0]
    perm = rng.permutation(n)
    n_val = n // 10
    val, fit = y[perm[:n_val]], y[perm[n_val:]]
    fit = fit - fit.mean(0)
    val_c = val - fit.mean(0)
    _, s, vt = np.linalg.svd(fit, full_matrices=False)
    out = []
    for k in RANKS:
        if k > vt.shape[0]:
            break
        proj = val_c @ vt[:k].T @ vt[:k]
        r = _per_feature_pearson(proj, val_c)
        out.append(
            {
                "rank": k,
                "ceiling_pearson_per_feature": float(np.nanmean(r)),
                "frac_fit_variance": float((s[:k] ** 2).sum() / (s**2).sum()),
            }
        )
    return out


def _per_feature_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(0)
    b = b - b.mean(0)
    num = (a * b).sum(0)
    den = np.sqrt((a**2).sum(0) * (b**2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.asarray(num / den)


def main() -> None:
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    img_dir = osp.join(os.environ["ASSET_IMAGES_DIR"], "019-simb-multimodal")
    os.makedirs(img_dir, exist_ok=True)
    results_dir = experiment_results_dir("019-simb-multimodal", __file__)

    matrix = pd.read_csv(osp.join(data_root, RAW_DIR, MATRIX)).set_index("Protein.Group")
    meta = pd.read_csv(osp.join(data_root, RAW_DIR, META)).set_index("Filename")
    cols = [c for c in matrix.columns if c in meta.index]
    matrix = matrix[cols]
    kind = meta.loc[cols, "sampletype"]
    wt = matrix.loc[:, (kind == "HIS3").to_numpy()]
    ko = matrix.loc[:, (kind == "ko").to_numpy()]
    print(f"proteins {matrix.shape[0]}, HIS3 columns {wt.shape[1]}, ko columns {ko.shape[1]}")

    log_wt = np.log2(wt.to_numpy(dtype=float))
    log_ko = np.log2(ko.to_numpy(dtype=float))
    ref_mean = np.nanmean(log_wt, axis=1)
    ratio_ko = log_ko - ref_mean[:, None]  # what the loader stores, per strain

    var_e = np.nanvar(log_wt, axis=1, ddof=1)
    var_tot = np.nanvar(ratio_ko, axis=1, ddof=1)
    n_wt = np.isfinite(log_wt).sum(1)
    n_ko = np.isfinite(log_ko).sum(1)
    ok = (n_wt >= 30) & (n_ko >= 100)
    rel_w = np.where(ok, np.clip(1 - var_e / var_tot, 0, 1), np.nan)
    ceil_w = np.sqrt(rel_w)
    route_w = _summarize("route W, HIS3 replicate", ceil_w, rel_w)
    route_w.update(
        {
            "n_his3_columns": int(wt.shape[1]),
            "n_ko_columns": int(ko.shape[1]),
            "median_sd_log2_his3": float(np.nanmedian(np.sqrt(var_e[ok]))),
            "median_sd_log2_ko": float(np.nanmedian(np.sqrt(var_tot[ok]))),
            "median_cv_pct_his3": float(
                100 * np.nanmedian(np.sqrt(np.exp(np.log(2) ** 2 * var_e[ok]) - 1))
            ),
            "frac_ko_entries_beyond_2sd_his3": float(
                np.nanmean(np.abs(ratio_ko[ok]) > 2 * np.sqrt(var_e[ok])[:, None])
            ),
        }
    )

    # ROUTE D -- DUPLICATE STRAINS. 145 deletion ORFs occupy two or three knockout columns
    # (98.6% of them on different plates), so the same genotype was grown and measured
    # again inside Messner. Per protein, the Pearson across those ORFs between the first
    # and second column estimates the within-study test-retest reliability of a strain's
    # protein effect; its square root is the ceiling for a genotype-only predictor.
    orf = meta.loc[ko.columns, "ORF"].astype(str).to_numpy()
    plate = meta.loc[ko.columns, "Plate (batch) nr"].astype(str).to_numpy()
    first: dict[str, int] = {}
    pairs: list[tuple[int, int]] = []
    for j, o in enumerate(orf):
        if o in first:
            pairs.append((first[o], j))
        else:
            first[o] = j
    ia = np.array([p[0] for p in pairs])
    ib = np.array([p[1] for p in pairs])
    a, b = ratio_ko[:, ia], ratio_ko[:, ib]
    fin = np.isfinite(a) & np.isfinite(b)
    rel_d = np.full(a.shape[0], np.nan)
    for g in range(a.shape[0]):
        m = fin[g]
        if m.sum() >= 30:
            rel_d[g] = np.corrcoef(a[g, m], b[g, m])[0, 1]
    ceil_d = np.sqrt(np.clip(rel_d, 0, 1))
    route_d = _summarize("route D, duplicate strains", ceil_d, rel_d)
    route_d.update(
        {
            "n_duplicate_pairs": int(len(pairs)),
            "n_orfs_duplicated": int(len({orf[i] for i in ia})),
            "frac_pairs_on_different_plates": float(np.mean(plate[ia] != plate[ib])),
            "frac_proteins_r_above_0.3": float(np.nanmean(rel_d > 0.3)),
            "per_strain_median_r": float(
                np.nanmedian(
                    [
                        np.corrcoef(a[fin[:, k], k], b[fin[:, k], k])[0, 1]
                        for k in range(len(pairs))
                    ]
                )
            ),
        }
    )
    print(
        f"  pairs {route_d['n_duplicate_pairs']}, per-strain median r"
        f" {route_d['per_strain_median_r']:.3f}"
    )

    # PLATE VARIANCE. Fraction of each protein's knockout variance explained by the 57
    # plates (one-way ANOVA R^2). Plate structure is not predictable from genotype, so it
    # counts against the genotype ceiling even though it is signal to a within-study
    # replicate that lands on the same plate.
    plate_ids, plate_idx = np.unique(plate, return_inverse=True)
    r2_plate = np.full(ratio_ko.shape[0], np.nan)
    for g in np.where(ok)[0]:
        row = ratio_ko[g]
        m = np.isfinite(row)
        y_g = row[m]
        grp = plate_idx[m]
        means = np.bincount(grp, weights=y_g, minlength=len(plate_ids)) / np.maximum(
            np.bincount(grp, minlength=len(plate_ids)), 1
        )
        ss_between = ((means[grp] - y_g.mean()) ** 2).sum()
        ss_total = ((y_g - y_g.mean()) ** 2).sum()
        r2_plate[g] = ss_between / ss_total
    plate_stats = {
        "n_plates": int(len(plate_ids)),
        "median_r2_plate": float(np.nanmedian(r2_plate)),
        "mean_r2_plate": float(np.nanmean(r2_plate)),
        "q75_r2_plate": float(np.nanpercentile(r2_plate, 75)),
    }
    print(f"[plate ANOVA] median R^2 {plate_stats['median_r2_plate']:.3f}, mean {plate_stats['mean_r2_plate']:.3f}")

    zel = json.load(open(ZELEZNIAK_JSON))["strain_aligned"]["zelezniak_vs_messner"]
    rel_z = float(zel["per_gene_median_r"])
    route_z = {
        "n_proteins": int(zel["per_gene_n"]),
        "n_strains": int(zel["n_shared_strains"]),
        "median_reliability": rel_z,
        "median_ceiling": float(np.sqrt(max(rel_z, 0.0))),
        "frac_proteins_r_above_0.3": float(zel["per_gene_frac_above_0.3"]),
        "source": ZELEZNIAK_JSON,
    }
    print(
        f"[route Z, Zelezniak cross-study] median r {rel_z:.3f} over {route_z['n_proteins']}"
        f" proteins -> ceiling {route_z['median_ceiling']:.3f}"
    )

    # Rank-k ceiling on the complete-case knockout matrix (strains x proteins, z-scored).
    # Proteins measured in >= 90% of knockouts, strains carrying >= 95% of those proteins;
    # the remaining missing entries (a fraction reported below) take the per-protein mean
    # (zero after z-scoring), the same fill the baselines use for the sparse target.
    keep_p = ok & (n_ko >= MIN_KO_PER_PROTEIN * ko.shape[1])
    y = ratio_ko[keep_p].T
    keep_s = np.isfinite(y).mean(1) >= 0.95
    y = y[keep_s]
    y = (y - np.nanmean(y, 0)) / np.nanstd(y, 0, ddof=1)
    frac_imputed = float(np.isnan(y).mean())
    y = np.where(np.isfinite(y), y, 0.0)
    rng = np.random.default_rng(0)
    rank_curve = _rank_ceiling(y, rng)
    print(
        f"rank ceiling on {y.shape[0]} strains x {y.shape[1]} proteins,"
        f" {100 * frac_imputed:.2f}% entries mean-filled"
    )
    for r in rank_curve:
        print(
            f"  rank {r['rank']:4d}: ceiling {r['ceiling_pearson_per_feature']:.3f}"
            f"  fit var {100 * r['frac_fit_variance']:.1f}%"
        )
    expr_lr = json.load(open(LOWRANK_EXPR_JSON))["ceiling_train_basis"]

    out = {
        "generated_by": "experiments/019-simb-multimodal/scripts/proteome_ceiling_replicate.py",
        "estimand": "per-protein corr(prediction, measured log2 ratio); ceiling = sqrt(reliability)",
        "observed": {"v14_partition_mean": OBS_V14, "b3_prot_T5_val_mean": OBS_B3},
        "route_w_his3_replicate": route_w,
        "route_d_duplicate_strains": route_d,
        "plate_anova": plate_stats,
        "route_z_zelezniak_cross_study": route_z,
        "rank_ceiling_knockout_matrix": {
            "n_strains": int(y.shape[0]),
            "n_proteins": int(y.shape[1]),
            "split": "90/10 strains, seed 0, train-basis SVD",
            "frac_entries_mean_filled": frac_imputed,
            "curve": rank_curve,
        },
    }
    with open(osp.join(results_dir, "proteome_ceiling_replicate.json"), "w") as fh:
        json.dump(out, fh, indent=1)

    # Figure: (a) per-protein ceiling, (b) sd across knockouts against sd of the reference,
    # (c) rank-k ceiling, proteome beside expression.
    plt.rcParams.update(
        {"font.family": "Arial", "font.size": 6, "svg.fonttype": "none", "axes.linewidth": 0.5}
    )
    fig, axes = plt.subplots(
        1, 3, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(52))
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.2, wspace=0.42)
    ax = axes[0]
    bins = np.linspace(0, 1, 41)
    ax.hist(ceil_w[ok], bins=bins, color=PLOT_PALETTE[0], edgecolor="black", linewidth=0.3,
            label="HIS3 replicate (noise floor)")
    ax.hist(ceil_d[np.isfinite(ceil_d)], bins=bins, histtype="step", color=PLOT_PALETTE[1],
            linewidth=0.8, label="duplicate strains (genotype)")
    ax.axvline(OBS_V14, color=PLOT_PALETTE[2], linewidth=0.8, linestyle=":", label="v14 partition mean")
    ax.set_xlabel("per-protein ceiling, sqrt(reliability)")
    ax.set_ylabel("proteins")
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.set_ylim(0, 210)
    ax.legend(frameon=True, edgecolor="black", fancybox=False, fontsize=5, loc="upper right")
    ax = axes[1]
    ax.scatter(
        np.sqrt(var_e[ok]), np.sqrt(var_tot[ok]), s=2, color=PLOT_PALETTE[0], linewidths=0
    )
    lim = float(np.nanmax(np.sqrt(var_tot[ok]))) * 1.05
    ax.plot([0, lim], [0, lim], color="black", linewidth=0.5)
    ax.set_xlabel("sd of log2 abundance, 388 HIS3 references")
    ax.set_ylabel("sd of log2 ratio across knockouts")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax = axes[2]
    ax.plot(
        [r["rank"] for r in rank_curve],
        [r["ceiling_pearson_per_feature"] for r in rank_curve],
        marker="o", markersize=2.5, linewidth=0.8, color=PLOT_PALETTE[0], label="proteome (Messner)",
    )
    ax.plot(
        [r["rank"] for r in expr_lr],
        [r["ceiling_pearson_per_feature"] for r in expr_lr],
        marker="s", markersize=2.5, linewidth=0.8, color=PLOT_PALETTE[1], label="expression (Kemmeren)",
    )
    ax.axhline(OBS_V14, color=PLOT_PALETTE[2], linewidth=0.8, linestyle=":", label="v14 partition mean")
    ax.axhline(OBS_B3, color=PLOT_PALETTE[3], linewidth=0.8, linestyle="--", label="B3 kNN ProtT5")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("rank of the response basis")
    ax.set_ylabel("per-feature Pearson ceiling")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.legend(
        frameon=True, edgecolor="black", fancybox=False, fontsize=5,
        loc="lower right", bbox_to_anchor=(0.99, 0.15),
    )
    for a in axes:
        for s in a.spines.values():
            s.set_linewidth(0.5)
        a.tick_params(which="minor", length=0)
        a.grid(True, axis="y", which="both", linewidth=0.3, alpha=0.4)
    for a, letter in zip(axes, "abc"):
        panel_label(a, letter)
    stem = "proteome_ceiling_replicate"
    for name in (f"{stem}_{timestamp()}", stem):
        fig.savefig(osp.join(img_dir, f"{name}.png"), dpi=300)
        savefig_true_size_svg(fig, osp.join(img_dir, f"{name}.svg"))
    print("wrote", osp.join(results_dir, "proteome_ceiling_replicate.json"))
    print("wrote", osp.join(img_dir, f"{stem}.svg"))


if __name__ == "__main__":
    main()
