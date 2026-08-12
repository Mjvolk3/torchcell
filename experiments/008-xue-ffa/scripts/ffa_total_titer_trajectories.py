# experiments/008-xue-ffa/scripts/ffa_total_titer_trajectories.py
# [[experiments.008-xue-ffa.scripts.ffa_total_titer_trajectories]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/008-xue-ffa/scripts/ffa_total_titer_trajectories
#
# Total-FFA-titer trajectories across the TF deletion ladder:
#   3-delta base (0 TF KO) -> 1 TF KO -> 2 TF KO -> 3 TF KO
#
# The Xue 2025 panel is a COMPLETE combinatorial design over 10 TFs layered on the
# POX1-FAA1-FAA4 (3-delta) FFA-overproduction platform: 10 singles (4-delta),
# 45 doubles (5-delta), 120 triples (6-delta), 3 replicates each.
#
# Anchor: the ladder is anchored at the '+ve Ctrl' row, which IS the 3-delta platform
# strain and is the normalization reference used by free_fatty_acid_interactions.py
# (f = strain / ref_mean), so the anchor sits at f = 1 by construction. 'wt BY4741' is
# a DIFFERENT genetic background (no FFA machinery, ~0.25 of the platform) and is drawn
# only as a dashed floor reference -- starting a TF trajectory there would spend the
# whole first step on platform engineering rather than TF biology.
#
# Trajectory construction: for each triple {i,j,k} the intermediate rungs average over
# all constituent genotypes (the 3 singles, then the 3 doubles), which is the
# order-agnostic mean over all 6 mutation orderings. A "best path" would pre-select for
# an upward trend and work against the negative-interaction signal being shown.
#
# Trigenic interaction (multiplicative, matching free_fatty_acid_interactions.py:436):
#   tau_ijk = f_ijk - f_ij*f_k - f_ik*f_j - f_jk*f_i + 2*f_i*f_j*f_k

import os
import os.path as osp
import sys
from itertools import combinations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.ticker import MultipleLocator

import torchcell
from torchcell.timestamp import timestamp
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    savefig_true_size_svg,
)

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
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
        "legend.title_fontsize": 6,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.5,
        "lines.linewidth": 0.7,
        "patch.linewidth": 0.4,
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

# 'Total Titer' is index 5 of the FFA axis used throughout this experiment.
FFA_NAMES = ["C14:0", "C16:0", "C18:0", "C16:1", "C18:1", "Total Titer"]
TOTAL_IDX = FFA_NAMES.index("Total Titer")

# Palette roles. Negative interactions are the finding, so they take brick; positive
# takes amber. Both are warm primaries, per the repo's primaries-first ordering.
C_NEG = PLOT_PALETTE[1]  # brick  #B85450
C_POS = PLOT_PALETTE[0]  # amber  #D79B00
C_GRAY = PLOT_PALETTE[5]  # gray   #666666

STEP_LABELS = ["3$\\Delta$ base\n(0 TF KO)", "1 TF KO", "2 TF KO", "3 TF KO"]


def read_abbreviations(path):
    """Letter -> TF gene name for all 10 TFs.

    NOTE: ``load_ffa_data`` reads this sheet with a default header row, which consumes
    FKH1 as the header and silently yields only 9 of the 10 TFs (strains containing
    'F' then carry the bare letter as their gene name). Reading with ``header=None``
    recovers all 10.
    """
    df = pd.read_excel(path, sheet_name="Abbreviations", header=None)
    abbrev = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
    if len(abbrev) != 10:
        raise ValueError(f"expected 10 TF abbreviations, got {len(abbrev)}: {abbrev}")
    return abbrev


def collect_total_titer(normalized_replicates, abbreviations, reference_strain):
    """Map genotype-order -> {frozenset(genes): (mean_f, sd_f)} for Total Titer."""
    singles, doubles, triples = {}, {}, {}
    for genotype, ffa_data in normalized_replicates.items():
        genes = parse_genotype(
            genotype, abbreviations, reference_strain=reference_strain
        )
        if not genes:
            continue
        reps = ffa_data["Total Titer"]
        valid = reps[~np.isnan(reps)]
        if len(valid) == 0:
            continue
        mean_f = float(np.mean(valid))
        sd_f = float(np.std(valid, ddof=1)) if len(valid) > 1 else np.nan
        key = frozenset(genes)
        target = {1: singles, 2: doubles, 3: triples}.get(len(genes))
        if target is not None:
            target[key] = (mean_f, sd_f)
    return singles, doubles, triples


def build_trajectories(singles, doubles, triples):
    """One row per triple: the 4 ladder rungs plus its multiplicative tau."""
    rows = []
    for tri_key, (f_ijk, sd_ijk) in triples.items():
        genes = sorted(tri_key)
        if len(genes) != 3:
            continue
        if not all(frozenset([g]) in singles for g in genes):
            continue
        pair_keys = [frozenset(p) for p in combinations(genes, 2)]
        if not all(pk in doubles for pk in pair_keys):
            continue

        i, j, k = genes
        f_i, f_j, f_k = (singles[frozenset([g])][0] for g in genes)
        f_ij = doubles[frozenset([i, j])][0]
        f_ik = doubles[frozenset([i, k])][0]
        f_jk = doubles[frozenset([j, k])][0]

        # Multiplicative trigenic interaction (Kuzmin form), matching
        # free_fatty_acid_interactions.py:436.
        tau = (
            f_ijk
            - f_ij * f_k
            - f_ik * f_j
            - f_jk * f_i
            + 2 * f_i * f_j * f_k
        )

        rows.append(
            {
                "triple": "-".join(genes),
                "gene_i": i,
                "gene_j": j,
                "gene_k": k,
                "f_base": 1.0,
                "f_single_mean": float(np.mean([f_i, f_j, f_k])),
                "f_double_mean": float(np.mean([f_ij, f_ik, f_jk])),
                "f_triple": f_ijk,
                "sd_triple": sd_ijk,
                "tau_multiplicative": tau,
            }
        )
    return pd.DataFrame(rows)


def plot_trajectories(traj_df, f_wt, out_stem):
    """Two panels: all trajectories by tau sign, and the median ladder with IQR."""
    width_mm = PANEL_WIDTHS_MM["full"]
    fig, (ax_l, ax_r) = plt.subplots(
        1,
        2,
        figsize=(mm_to_in(width_mm), mm_to_in(70.0)),
        gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.28},
    )

    x = np.arange(4)
    cols = ["f_base", "f_single_mean", "f_double_mean", "f_triple"]
    Y = traj_df[cols].to_numpy(dtype=float)
    neg = traj_df["tau_multiplicative"].to_numpy() < 0

    # --- Left: every triple's trajectory, colored by sign of tau -----------------
    for row, is_neg in zip(Y, neg):
        ax_l.plot(
            x,
            row,
            color=C_NEG if is_neg else C_POS,
            linewidth=0.4,
            alpha=0.55,
            solid_capstyle="round",
            zorder=2 if is_neg else 3,
        )

    ax_l.axhline(1.0, color="black", linewidth=0.5, linestyle="--", zorder=4)
    ax_l.axhline(
        f_wt, color=C_GRAY, linewidth=0.5, linestyle=":", zorder=4
    )
    ax_l.text(
        0.06,
        f_wt + 0.02,
        "wt BY4741 (different background)",
        color=C_GRAY,
        fontsize=6,
        va="bottom",
        ha="left",
    )

    n_neg, n_pos = int(neg.sum()), int((~neg).sum())
    handles = [
        plt.Line2D([], [], color=C_NEG, linewidth=1.0),
        plt.Line2D([], [], color=C_POS, linewidth=1.0),
    ]
    ax_l.legend(
        handles,
        [f"Negative $\\tau$ (n={n_neg})", f"Positive $\\tau$ (n={n_pos})"],
        loc="upper left",
        frameon=True,
        fontsize=6,
        handlelength=1.6,
        borderpad=0.4,
    )
    ax_l.set_ylabel("Total FFA titer (relative to 3$\\Delta$ base)")
    ax_l.set_title(
        f"Total FFA titer trajectories ({len(traj_df)} triples)", fontsize=6, pad=4
    )

    # --- Right: median ladder with IQR band, split by tau sign -------------------
    for mask, color, label in (
        (neg, C_NEG, "Negative $\\tau$"),
        (~neg, C_POS, "Positive $\\tau$"),
    ):
        if mask.sum() == 0:
            continue
        block = Y[mask]
        med = np.median(block, axis=0)
        q1 = np.percentile(block, 25, axis=0)
        q3 = np.percentile(block, 75, axis=0)
        ax_r.fill_between(x, q1, q3, color=color, alpha=0.18, linewidth=0, zorder=2)
        ax_r.plot(
            x,
            med,
            color=color,
            linewidth=1.2,
            marker="o",
            markersize=2.5,
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.3,
            label=label,
            zorder=3,
        )

    ax_r.axhline(1.0, color="black", linewidth=0.5, linestyle="--", zorder=4)
    ax_r.legend(loc="upper left", frameon=True, fontsize=6, handlelength=1.6)
    ax_r.set_ylabel("Median total FFA titer (rel. 3$\\Delta$ base)")
    ax_r.set_title("Median ladder (band = IQR)", fontsize=6, pad=4)

    for ax in (ax_l, ax_r):
        ax.set_xticks(x)
        ax.set_xticklabels(STEP_LABELS)
        ax.set_xlim(-0.15, 3.15)
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.tick_params(axis="y", which="minor", length=0)
        ax.grid(axis="y", which="both", color="0.85", linewidth=0.3, zorder=0)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_color("black")

    fig.savefig(f"{out_stem}.png", dpi=300, bbox_inches="tight")
    savefig_true_size_svg(fig, f"{out_stem}.svg")
    plt.close(fig)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    abbreviations = read_abbreviations(RAW_TITER_PATH)
    averaged_df, _loader_abbrev, replicate_dict = load_ffa_data(RAW_TITER_PATH)
    _normalized_df, normalized_replicates = normalize_by_reference(
        averaged_df, replicate_dict
    )
    reference_strain = list(normalized_replicates.keys())[0]

    singles, doubles, triples = collect_total_titer(
        normalized_replicates, abbreviations, reference_strain
    )
    print(f"singles={len(singles)}  doubles={len(doubles)}  triples={len(triples)}")

    # 'wt BY4741' floor reference, already normalized to the 3-delta base.
    wt_key = next(k for k in normalized_replicates if "BY4741" in str(k))
    wt_reps = normalized_replicates[wt_key]["Total Titer"]
    f_wt = float(np.nanmean(wt_reps))
    print(f"wt BY4741 relative total titer = {f_wt:.4f}")

    traj_df = build_trajectories(singles, doubles, triples)
    n_neg = int((traj_df["tau_multiplicative"] < 0).sum())
    print(
        f"trajectories={len(traj_df)}  negative tau={n_neg} "
        f"({100 * n_neg / len(traj_df):.1f}%)"
    )
    print(
        traj_df[["f_base", "f_single_mean", "f_double_mean", "f_triple"]]
        .median()
        .round(4)
        .to_string()
    )

    csv_path = osp.join(RESULTS_DIR, "ffa_total_titer_trajectories.csv")
    traj_df.sort_values("tau_multiplicative").to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")

    out_stem = osp.join(IMAGES_DIR, f"ffa_total_titer_trajectories_{timestamp()}")
    plot_trajectories(traj_df, f_wt, out_stem)
    print(f"wrote {out_stem}.png / .svg")


if __name__ == "__main__":
    main()
