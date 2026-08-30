# experiments/019-simb-multimodal/scripts/plot_imputation_oracle_and_knn.py
# [[experiments.019-simb-multimodal.scripts.plot_imputation_oracle_and_knn]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/plot_imputation_oracle_and_knn
"""One figure for the two parameter-free probes: the conditioning oracle and the kNN readout.

Four panels, each readable on its own:

  a  WITHIN-STUDY gene budget. Per-feature Pearson on 155 held-out Kemmeren strains as a
     function of how many genes are observed, for a random observed set and for two rules
     that CHOOSE the set. Read against the replicate ceiling (0.775) and against the
     trained model's genotype-only 0.198.
  b  CROSS-STUDY gene budget. Same curves, but the observed values come from Kemmeren and
     the target is Sameith's independent measurement of the same 82 strains, so shared
     array state cannot contribute. Its ceiling is the two studies' own agreement on the
     same genes (~0.611), not 1.0.
  c  Is the conditioning signal reachable from genotype? The oracle re-run on residuals
     after removing a genotype-conditional kNN prediction, against the same oracle on
     mean-centered residuals. Bars that match mean that removing the genotype predictor
     removes none of the conditioning gain.
  d  kNN embedding probe. A parameter-free similarity average over training genes, scored
     on the same metric as the sweep, against the swept transformer and against the noise
     floor set by random-vector controls.

Inputs (all committed artifacts):
  results/conditioning_gene_budget.json        (conditioning_gene_budget.py)
  results/conditioning_gain_after_genotype.json (conditioning_gain_after_genotype.py)
  results/knn_embedding_probe.json             (knn_embedding_probe.py)

Format follows the repo standard (CLAUDE.md "Figure & Plotting Standards"): PLOT_PALETTE,
Arial 6 pt, all four spines, strict PANEL_WIDTHS_MM width, panel letters from
torchcell.utils.panel_label, true-size SVG plus a PNG raster fallback.

Run from repo root:
    python experiments/019-simb-multimodal/scripts/plot_imputation_oracle_and_knn.py
"""

from __future__ import annotations

import json
import os.path as osp

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from matplotlib.ticker import MultipleLocator

from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    panel_label,
    savefig_true_size_svg,
)
from torchcell.utils.paths import asset_images_dir, experiment_results_dir

load_dotenv()

STEM = "imputation_oracle_and_knn_probe"

# Trained-model reference on the same metric, quoted from the artifacts it came from.
CGT_EXPRESSION_M0 = 0.198  # val pearson_per_feature, genotype only (masked-oracle note)
CGT_KNN_SPLIT = {"morphology": 0.0333, "expression": 0.0795}  # swept CGT, probe's split

RULE_STYLE = {
    "random": (PLOT_PALETTE[5], "o", "random"),
    "variance": (PLOT_PALETTE[0], "s", "top variance"),
    "qr_leverage": (PLOT_PALETTE[2], "^", "QR leverage"),
}

# A readable subset of the 19 probe arms: one row per representational axis plus the two
# controls that make the panel interpretable on its own.
KNN_ROWS = [
    ("prot_T5", "prot_T5_all", 0),
    ("ESM2 650M", "esm2_t33_650M_UR50D_all", 0),
    ("chrom. pathways", "normalized_chrom_pathways", 1),
    ("CaLM", "calm", 2),
    ("codon freq.", "codon_frequency", 2),
    ("species LM 5'", "species_lm_five_prime", 3),
    ("NT locus 5979", "nt_window_5979", 4),
    ("one-hot gene", "one_hot_gene", 5),
    ("random d=1000", "random_1000", 5),
]
RANDOM_KEYS = ["random_1", "random_10", "random_100", "random_1000"]


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 6,
            "axes.labelsize": 6,
            "axes.titlesize": 6,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 5,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "lines.linewidth": 0.9,
            "svg.fonttype": "none",
        }
    )


def _budget_panel(
    ax, budget: dict, arm: str, title: str, ceiling, ceiling_label
) -> None:
    by = {(r["rule"], r["m"]): r for r in budget["results"]}
    ms = budget["m_grid"]
    for rule, (color, marker, label) in RULE_STYLE.items():
        mean = np.array([by[(rule, m)]["arms"][arm]["mean"] for m in ms])
        sd = np.array([by[(rule, m)]["arms"][arm]["sd"] for m in ms])
        n_draws = by[(rule, ms[0])]["n_draws"]
        suffix = f" ({n_draws} draws)" if n_draws > 1 else " (nested set)"
        ax.plot(
            ms, mean, marker=marker, ms=2.2, color=color, label=label + suffix, zorder=3
        )
        if sd.max() > 0:
            ax.fill_between(
                ms, mean - sd, mean + sd, color=color, alpha=0.20, lw=0, zorder=1
            )
    null = np.array(
        [by[("random", m)]["arms_permuted_strain_null"][arm]["mean"] for m in ms]
    )
    ax.plot(
        ms, null, color="black", lw=0.6, ls=":", zorder=2, label="permuted-strain null"
    )
    ax.axhline(
        ceiling, color=PLOT_PALETTE[1], lw=0.8, ls="--", zorder=2, label=ceiling_label
    )
    ax.set_xscale("log")
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(["1", "10", "100", "1000"])
    ax.set_xlabel("genes observed, $m$")
    ax.set_ylabel("per-feature Pearson $r$ on held-out genes")
    ax.set_title(title, pad=3)
    ax.set_ylim(-0.05, 0.85)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which="minor", length=0)
    ax.grid(which="both", lw=0.3, color="0.85", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, borderpad=0.3)


def main() -> None:
    _style()
    results_dir = experiment_results_dir("019-simb-multimodal", __file__)
    images_dir = asset_images_dir(__file__, subdir="019-simb-multimodal")

    with open(osp.join(results_dir, "conditioning_gene_budget.json")) as f:
        budget = json.load(f)
    with open(osp.join(results_dir, "conditioning_gain_after_genotype.json")) as f:
        gain = json.load(f)
    with open(osp.join(results_dir, "knn_embedding_probe.json")) as f:
        knn = json.load(f)

    fig, axes = plt.subplots(
        2, 2, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(118.0))
    )
    ax_a, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # --- a: within-study budget -------------------------------------------------------
    _budget_panel(
        ax_a,
        budget,
        "val_kem_155",
        f"within study (Kemmeren, n = {budget['n_val_kem']} strains)",
        budget["reference_replicate_ceiling_mean_sqrt_r"],
        f"replicate ceiling {budget['reference_replicate_ceiling_mean_sqrt_r']:.3f}",
    )
    ax_a.axhline(CGT_EXPRESSION_M0, color="black", lw=0.7, ls="-.", zorder=2)
    ax_a.annotate(
        f"trained CGT, genotype only ($m$ = 0): {CGT_EXPRESSION_M0:.3f}",
        xy=(1.15, CGT_EXPRESSION_M0),
        xytext=(1.15, CGT_EXPRESSION_M0 + 0.03),
        fontsize=5,
    )
    panel_label(ax_a, "a")

    # --- b: cross-study budget --------------------------------------------------------
    xstudy = np.mean(
        [
            r["xstudy_agreement_ceiling"]["mean"]
            for r in budget["results"]
            if r["rule"] == "random"
        ]
    )
    _budget_panel(
        ax_b,
        budget,
        "cross_kem_to_sam",
        "cross study (observe Kemmeren, predict Sameith, "
        f"n = {budget['n_shared_deletions_eval']} strains)",
        xstudy,
        f"cross-study agreement {xstudy:.3f}",
    )
    # Two markers on the cheap end, as a fraction of the m = 1000 random value: the whole
    # point of the panel is that the curve is most of the way there long before m = 1000.
    by_b = {(r["rule"], r["m"]): r for r in budget["results"]}
    ref_b = by_b[("random", 1000)]["arms"]["cross_kem_to_sam"]["mean"]
    for m_mark, dy in ((10, 0.10), (100, 0.10)):
        v = by_b[("random", m_mark)]["arms"]["cross_kem_to_sam"]["mean"]
        ax_b.annotate(
            f"{100 * v / ref_b:.0f}% of\nthe $m$ = 1000\nvalue",
            xy=(m_mark, v),
            xytext=(m_mark, v + dy),
            fontsize=5,
            ha="center",
            arrowprops={"arrowstyle": "->", "lw": 0.4, "color": "black"},
        )
    panel_label(ax_b, "b")

    # --- c: conditioning gain after removing a genotype predictor ---------------------
    base_key = "col_mean"
    knn_key = next(k for k in gain["baselines"] if k != base_key)
    ms_c = [r["m"] for r in gain["baselines"][base_key] if r["m"] > 0]
    x = np.arange(len(ms_c))
    w = 0.36
    for i, (key, color, label) in enumerate(
        [
            (base_key, PLOT_PALETTE[0], "residual vs per-gene mean"),
            (knn_key, PLOT_PALETTE[1], f"residual vs genotype kNN ({knn_key})"),
        ]
    ):
        rows = {r["m"]: r for r in gain["baselines"][key]}
        vals = [rows[m]["val_mean"] for m in ms_c]
        errs = [rows[m]["val_sd"] for m in ms_c]
        ax_c.bar(
            x + (i - 0.5) * w,
            vals,
            width=w,
            color=color,
            edgecolor="black",
            linewidth=0.4,
            yerr=errs,
            error_kw={"lw": 0.5, "capsize": 1.2},
            label=label,
            zorder=3,
        )
    for j, m in enumerate(ms_c):
        frac = gain["retained_fraction"][str(m)]
        top = max(
            r["val_mean"]
            for k in gain["baselines"]
            for r in gain["baselines"][k]
            if r["m"] == m
        )
        ax_c.text(
            x[j], top + 0.03, f"{100 * frac:.1f}%\nretained", ha="center", fontsize=5
        )
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([str(m) for m in ms_c])
    ax_c.set_xlabel("genes observed, $m$")
    ax_c.set_ylabel("per-feature Pearson $r$")
    ax_c.set_ylim(0, 1.0)
    ax_c.set_title(
        "oracle before and after removing a genotype predictor\n"
        f"(within study, n = {gain['n_val']} held-out strains, {gain['n_genes']} genes)",
        pad=3,
    )
    ax_c.yaxis.set_major_locator(MultipleLocator(0.2))
    ax_c.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax_c.tick_params(which="minor", length=0)
    ax_c.grid(axis="y", which="both", lw=0.3, color="0.85", zorder=0)
    ax_c.set_axisbelow(True)
    ax_c.legend(loc="upper left", frameon=True, framealpha=0.95, borderpad=0.3)
    panel_label(ax_c, "c")

    # --- d: kNN embedding probe -------------------------------------------------------
    arms = knn["arms"]
    floor = {
        mod: max(
            abs(arms[k]["modalities"][mod]["best_pearson_per_feature"])
            for k in RANDOM_KEYS
        )
        for mod in ("morphology", "expression")
    }
    y = np.arange(len(KNN_ROWS))
    hb = 0.36
    mod_color = {"morphology": PLOT_PALETTE[3], "expression": PLOT_PALETTE[2]}
    for i, mod in enumerate(("morphology", "expression")):
        vals, undef = [], []
        for _, key, _ in KNN_ROWS:
            m = arms[key]["modalities"][mod]
            deg = bool(m["degenerate_orthogonal_geometry"]) or not np.isfinite(
                m["best_pearson_per_feature"]
            )
            undef.append(deg)
            vals.append(0.0 if deg else m["best_pearson_per_feature"])
        bars = ax_d.barh(
            y + (0.5 - i) * hb,
            vals,
            height=hb,
            color=mod_color[mod],
            edgecolor="black",
            linewidth=0.4,
            label=f"{mod} (n$_{{val}}$="
            f"{arms['prot_T5_all']['modalities'][mod]['n_val']} genes)",
            zorder=3,
        )
        for row, (b, deg) in enumerate(zip(bars, undef)):
            if deg:
                # A zero-height bar would be invisible, and an absent bar reads as "not
                # measured" rather than "structurally undefined". Draw a hatched stub the
                # width of the noise floor, and say so in words next to it.
                b.set_width(max(floor.values()))
                b.set_hatch("////")
                b.set_facecolor("white")
                if i == 0:
                    ax_d.text(
                        max(floor.values()) + 0.004,
                        row,
                        "undefined",
                        va="center",
                        fontsize=5,
                    )
        ax_d.axvline(
            CGT_KNN_SPLIT[mod],
            color=mod_color[mod],
            lw=0.8,
            ls="--",
            zorder=4,
            label=f"swept CGT, {mod} ({CGT_KNN_SPLIT[mod]:.3f})",
        )
    ax_d.axvspan(
        -max(floor.values()),
        max(floor.values()),
        color=PLOT_PALETTE[5],
        alpha=0.14,
        lw=0,
        zorder=0,
        label=f"noise floor (|r| < {max(floor.values()):.3f})",
    )
    ax_d.axvline(0.0, color="black", lw=0.5, zorder=1)
    ax_d.set_yticks(y)
    ax_d.set_yticklabels([f"{lab}  ({arms[k]['dim']}d)" for lab, k, _ in KNN_ROWS])
    ax_d.invert_yaxis()
    ax_d.set_xlabel("per-feature Pearson $r$ (best $k$)")
    ax_d.set_title(
        "kNN readout in embedding space, no learned parameters\n"
        "(one-hot is undefined: all cosine similarities are 0)",
        pad=3,
    )
    ax_d.set_xlim(-0.02, 0.14)
    ax_d.xaxis.set_major_locator(MultipleLocator(0.04))
    ax_d.xaxis.set_minor_locator(MultipleLocator(0.02))
    ax_d.tick_params(which="minor", length=0)
    ax_d.grid(axis="x", which="both", lw=0.3, color="0.85", zorder=0)
    ax_d.set_axisbelow(True)
    ax_d.legend(loc="lower right", frameon=True, framealpha=0.95, borderpad=0.3)
    panel_label(ax_d, "d")

    for ax in axes.ravel():
        for s in ax.spines.values():
            s.set_visible(True)

    fig.tight_layout(pad=0.7, h_pad=1.6, w_pad=1.4)
    stem = osp.join(images_dir, STEM)
    fig.savefig(f"{stem}.png", dpi=300)
    savefig_true_size_svg(fig, f"{stem}.svg")
    print(f"wrote {stem}.png")
    print(f"wrote {stem}.svg")


if __name__ == "__main__":
    main()
