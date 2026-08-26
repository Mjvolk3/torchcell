# experiments/008-xue-ffa/scripts/linear_vs_log_scale_sign_test.py
# [[experiments.008-xue-ffa.scripts.linear_vs_log_scale_sign_test]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/008-xue-ffa/scripts/linear_vs_log_scale_sign_test
#
# Does the LOG SCALE explain why the two log-scale models find no positive trigenic
# interaction while the two linear-scale models do?
#
# 008 fits four epistasis models. On total FFA titer the multiplicative and additive
# models report positive trigenic interactions (11 and 6 clear FDR); the GLM log-link and
# log-OLS models report ZERO. The candidate explanation is that these are not two fits of
# one quantity but two DIFFERENT ESTIMANDS, and that the log transform is what removes the
# positives. This script tests that directly rather than assuming it.
#
# Both estimands are computed from the SAME normalized strain means, so nothing differs
# except the scale:
#
#   linear (multiplicative, Kuzmin tau-SGA):
#       tau_lin = f_ijk - f_ij f_k - f_ik f_j - f_jk f_i + 2 f_i f_j f_k
#
#   log (the saturated three-way interaction on log fitness, with g = log f and
#   g_null = log 1 = 0, which is what a log-link / log-OLS coefficient estimates):
#       tau_log = g_ijk - g_ij - g_ik - g_jk + g_i + g_j + g_k
#
# The script ASSERTS that tau_log reproduces the shipped GLM and log-OLS coefficients
# before drawing any conclusion from it. If that assertion fails, the two are not the same
# estimand and the whole comparison is void, so it is a hard failure rather than a warning.

import os
import os.path as osp
import re
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.ticker import MultipleLocator
from scipy import stats

import torchcell
from torchcell.timestamp import timestamp
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    savefig_true_size_svg,
)

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from ffa_total_titer_trajectories import read_abbreviations  # noqa: E402
from free_fatty_acid_interactions import (  # noqa: E402
    load_ffa_data,
    normalize_by_reference,
    parse_genotype,
)

load_dotenv()
plt.style.use(osp.join(osp.dirname(torchcell.__file__), "torchcell.mplstyle"))
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6,
        "axes.titlesize": 6,
        "axes.labelsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.5,
        "lines.linewidth": 0.7,
        "savefig.bbox": "standard",
    }
)

DATA_ROOT = os.getenv("DATA_ROOT")
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")

RAW_TITER_PATH = osp.join(
    DATA_ROOT, "data/torchcell/ffa_xue2025/raw/Supplementary Data 1_Raw titers.xlsx"
)
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "008-xue-ffa/results")
IMAGES_DIR = osp.join(ASSET_IMAGES_DIR, "008-xue-ffa")
PHENOTYPE = "Total Titer"

# Established 008 roles: amber for positive interactions, brick for negative, gray for the
# reference rules. Lilac marks the sign-flipped subset, which is the finding.
C_POS = PLOT_PALETTE[0]
C_NEG = PLOT_PALETTE[1]
C_FLIP = PLOT_PALETTE[2]
C_GRAY = PLOT_PALETTE[5]


def normalize_triple(gene_set):
    """Gene-set label to a canonical sorted key.

    The model families write different separators (`RPD3_SPT3_YAP6` from the linear-scale
    scripts, `FKH1:GCN5:MED4` from the regression scripts), so a naive join across them
    silently matches nothing.
    """
    return "-".join(sorted(re.split(r"[_:|-]", str(gene_set))))


def collect_means():
    """frozenset(genes) -> mean normalized total titer, split by mutation order."""
    abbreviations = read_abbreviations(RAW_TITER_PATH)
    averaged_df, _, replicate_dict = load_ffa_data(RAW_TITER_PATH)
    _, normalized_replicates = normalize_by_reference(averaged_df, replicate_dict)
    reference_strain = list(normalized_replicates.keys())[0]

    singles, doubles, triples = {}, {}, {}
    for genotype, ffa_data in normalized_replicates.items():
        genes = parse_genotype(
            genotype, abbreviations, reference_strain=reference_strain
        )
        if not genes:
            continue
        reps = ffa_data[PHENOTYPE]
        valid = reps[~np.isnan(reps)]
        if not len(valid):
            continue
        target = {1: singles, 2: doubles, 3: triples}.get(len(genes))
        if target is not None:
            target[frozenset(genes)] = float(np.mean(valid))
    return singles, doubles, triples


def both_estimands(singles, doubles, triples):
    """One row per triple carrying the linear and the log three-way interaction."""
    rows = []
    for tri, f_ijk in triples.items():
        i, j, k = sorted(tri)
        f_i, f_j, f_k = (singles[frozenset([g])] for g in (i, j, k))
        f_ij = doubles[frozenset([i, j])]
        f_ik = doubles[frozenset([i, k])]
        f_jk = doubles[frozenset([j, k])]

        tau_lin = f_ijk - f_ij * f_k - f_ik * f_j - f_jk * f_i + 2 * f_i * f_j * f_k
        g = np.log
        tau_log = (
            g(f_ijk)
            - g(f_ij)
            - g(f_ik)
            - g(f_jk)
            + g(f_i)
            + g(f_j)
            + g(f_k)
        )
        rows.append(
            {
                "triple": "-".join(sorted(tri)),
                "tau_linear": tau_lin,
                "tau_log": tau_log,
                "f_triple": f_ijk,
            }
        )
    return pd.DataFrame(rows)


def validate_against_shipped(df):
    """Hard-fail unless tau_log reproduces the shipped log-scale model coefficients.

    This is the load-bearing check. The claim "the log scale explains the sign difference"
    only means anything if the tau_log computed here IS what those models fit.
    """
    checks = {
        "glm_log_link": "glm_log_link/glm_log_link_trigenic_interactions.csv",
        "log_ols": "glm_models/log_ols_trigenic_interactions.csv",
    }
    report = {}
    for name, rel in checks.items():
        shipped = pd.read_csv(osp.join(RESULTS_DIR, rel))
        shipped = shipped[shipped["ffa_type"] == PHENOTYPE].copy()
        shipped["triple"] = shipped["gene_set"].map(normalize_triple)
        merged = df.merge(
            shipped[["triple", "interaction_score"]], on="triple", how="inner"
        )
        if len(merged) < len(df):
            raise ValueError(
                f"{name}: only {len(merged)} of {len(df)} triples joined; "
                "gene-set separators likely differ"
            )
        r = float(merged["tau_log"].corr(merged["interaction_score"]))
        agree = int((np.sign(merged["tau_log"]) == np.sign(merged["interaction_score"])).sum())
        if r < 0.99:
            raise ValueError(
                f"{name}: tau_log correlates r={r:.3f} with the shipped coefficient, so "
                "it is not the same estimand and the comparison is void"
            )
        report[name] = (r, agree, len(merged))
    return report


def plot_sign_test(df, out_stem):
    """Left: the two estimands against each other. Right: what happens to the positives."""
    fig, (ax_l, ax_r) = plt.subplots(
        1,
        2,
        figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(84.0)),
        gridspec_kw={"width_ratios": [1.45, 1.0], "wspace": 0.30},
    )

    flipped = (df["tau_linear"] > 0) & (df["tau_log"] <= 0)
    kept_pos = (df["tau_linear"] > 0) & (df["tau_log"] > 0)
    neg = df["tau_linear"] < 0

    for mask, color, label in (
        (neg, C_NEG, f"negative on both (n={int(neg.sum())})"),
        (kept_pos, C_POS, f"positive on both (n={int(kept_pos.sum())})"),
        (flipped, C_FLIP, f"positive $\\to$ non-positive (n={int(flipped.sum())})"),
    ):
        ax_l.scatter(
            df.loc[mask, "tau_linear"],
            df.loc[mask, "tau_log"],
            s=7,
            facecolor=color,
            edgecolor="black",
            linewidth=0.25,
            label=label,
            zorder=3,
        )

    ax_l.axhline(0, color="black", linewidth=0.5, linestyle="--", zorder=2)
    ax_l.axvline(0, color="black", linewidth=0.5, linestyle="--", zorder=2)
    r = float(df["tau_linear"].corr(df["tau_log"]))
    rho = float(df["tau_linear"].corr(df["tau_log"], method="spearman"))
    ax_l.set_xlabel("$\\tau$ linear scale (multiplicative)")
    ax_l.set_ylabel("$\\tau$ log scale (GLM log-link / log-OLS)")
    ax_l.set_title(
        f"The two estimands agree in magnitude (r = {r:.3f}, $\\rho$ = {rho:.3f})\n"
        "but the upper-left quadrant is empty: no negative ever becomes positive",
        fontsize=6,
        pad=4,
    )
    ax_l.legend(loc="upper left", frameon=True, handlelength=1.2, borderpad=0.4)

    # Right: paired slopes for the linear-positive triples only.
    pos = df[df["tau_linear"] > 0]
    for _, row in pos.iterrows():
        is_flip = row["tau_log"] <= 0
        ax_r.plot(
            [0, 1],
            [row["tau_linear"], row["tau_log"]],
            color=C_FLIP if is_flip else C_POS,
            linewidth=0.6,
            marker="o",
            markersize=2.0,
            markeredgecolor="black",
            markeredgewidth=0.2,
            zorder=3 if is_flip else 2,
        )
    ax_r.axhline(0, color="black", linewidth=0.5, linestyle="--", zorder=1)
    ax_r.set_xticks([0, 1])
    ax_r.set_xticklabels(["linear", "log"])
    ax_r.set_xlim(-0.15, 1.15)
    ax_r.set_ylabel("$\\tau$")
    n_flip = int((pos["tau_log"] <= 0).sum())
    ax_r.set_title(
        f"Every triple positive on the linear scale\n"
        f"{n_flip} of {len(pos)} lose that sign on the log scale",
        fontsize=6,
        pad=4,
    )

    for ax in (ax_l, ax_r):
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.tick_params(axis="y", which="minor", length=0)
        ax.grid(axis="y", which="major", color="0.85", linewidth=0.3, zorder=0)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_color("black")

    # tight_layout, not bbox_inches="tight": it reflows the axes INSIDE the fixed canvas,
    # so the two-line titles and the x label all fit without changing the panel width the
    # figure was authored at.
    fig.tight_layout(pad=0.6, w_pad=1.4)

    fig.savefig(f"{out_stem}.png", dpi=300)
    savefig_true_size_svg(fig, f"{out_stem}.svg")
    plt.close(fig)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    singles, doubles, triples = collect_means()
    df = both_estimands(singles, doubles, triples)
    print(f"triples = {len(df)}")

    report = validate_against_shipped(df)
    for name, (r, agree, n) in report.items():
        print(f"  tau_log vs shipped {name}: r = {r:.4f}, sign agreement {agree}/{n}")

    pos = df["tau_linear"] > 0
    flipped = pos & (df["tau_log"] <= 0)
    neg_to_pos = (df["tau_linear"] < 0) & (df["tau_log"] > 0)
    print(
        f"\nlinear positive: {int(pos.sum())} | of those non-positive on log: "
        f"{int(flipped.sum())} ({100 * flipped.sum() / pos.sum():.0f}%)\n"
        f"linear negative: {int((~pos).sum())} | of those positive on log: "
        f"{int(neg_to_pos.sum())}"
    )
    # Is the flip predictable from magnitude? Weakly, and worth stating as weak.
    u = stats.mannwhitneyu(
        df.loc[flipped, "tau_linear"],
        df.loc[pos & ~flipped, "tau_linear"],
        alternative="less",
    )
    print(
        f"flipped tau_linear median {df.loc[flipped, 'tau_linear'].median():.3f} vs "
        f"kept {df.loc[pos & ~flipped, 'tau_linear'].median():.3f} "
        f"(Mann-Whitney p = {u.pvalue:.4f}); largest flipped "
        f"{df.loc[flipped, 'tau_linear'].max():.3f} exceeds smallest kept "
        f"{df.loc[pos & ~flipped, 'tau_linear'].min():.3f}, so magnitude is a weak "
        "predictor, not a rule"
    )

    df["sign_class"] = np.where(
        df["tau_linear"] < 0,
        "negative_both",
        np.where(df["tau_log"] > 0, "positive_both", "positive_to_nonpositive"),
    )
    csv_path = osp.join(RESULTS_DIR, "linear_vs_log_scale_sign_test.csv")
    df.sort_values("tau_linear", ascending=False).to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path}")

    stamp = "" if os.getenv("NO_TIMESTAMP") else f"_{timestamp()}"
    out_stem = osp.join(IMAGES_DIR, f"linear_vs_log_scale_sign_test{stamp}")
    plot_sign_test(df, out_stem)
    print(f"wrote {out_stem}.png / .svg")


if __name__ == "__main__":
    main()
