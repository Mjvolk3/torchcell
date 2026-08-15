# experiments/008-xue-ffa/scripts/ffa_epistatic_path_panels.py
# [[experiments.008-xue-ffa.scripts.ffa_epistatic_path_panels]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/008-xue-ffa/scripts/ffa_epistatic_path_panels
#
# Per-triple KO trajectory panels: every ORDER in which a TF triple can be built.
#
# Companion to ffa_total_titer_trajectories.py. That script averages the rungs over all
# orderings on purpose (an order-agnostic ladder). This one does the complement: it
# RESOLVES the 6 orderings of each triple so the spread of the path intermediates is
# visible. Same figure grammar as the Kuzmin panel-12 trajectory figure
# (010-kuzmin-tmi/scripts/12_panel_inference_3_fitness_comparison.py::
# plot_all_paths_hero_triples), but every rung here is a MEASURED strain rather than a
# model-predicted fitness -- the Xue 2025 panel is a complete combinatorial design over
# 10 TFs (10 singles + 45 doubles + 120 triples, 3 GC replicates each) layered on the
# POX1-FAA1-FAA4 (3-delta) FFA platform, so all 6 x 120 paths are fully observed.
#
# Path: 3-delta base (f = 1 by construction, the '+ve Ctrl' row) -> single a -> double
# {a,b} -> triple {a,b,c}. All 6 orderings of a triple share the same endpoint, so the
# panel isolates how much the ROUTE matters relative to the destination.
#
# ACCESSIBILITY. A path is 'monotone' (greedily accessible) when every rung strictly
# improves on the previous one: 1 < f_a < f_ab < f_ijk. That is the question a strain
# engineer actually faces -- if you stack deletions one at a time and keep only the
# improvements, which triples can you even reach? A relaxed variant counts a step as
# non-decreasing when it drops by less than the propagated 1 SE of the two rungs, so the
# strict count is not an artifact of replicate noise.
#
# Interaction sign uses the same multiplicative trigenic tau as the rest of 008:
#   tau_ijk = f_ijk - f_ij*f_k - f_ik*f_j - f_jk*f_i + 2*f_i*f_j*f_k

import argparse
import os
import os.path as osp
import sys
from itertools import combinations, permutations

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

# The canonical trigenic interaction table for 008, written by
# free_fatty_acid_interactions.py. We read the significance columns from it rather than
# re-deriving them, so the panels cannot disagree with the experiment's own statistics --
# and we ASSERT its interaction_score equals the tau computed here (see load_significance).
TRIGENIC_PATH = osp.join(
    RESULTS_DIR, "multiplicative_trigenic_interactions_3_delta_normalized.csv"
)

# Rung 0 is the BASE PRODUCTION STRAIN: the '+ve Ctrl' row, i.e. the pox1/faa1/faa4
# FFA-overproduction chassis. Its three deletions are the platform, present in EVERY
# strain on the ladder; the TF knockouts counted on the x axis are ADDITIONAL on top of
# it (so a '6-delta' genotype = 3 chassis + 3 TF). Labelling rung 0 '3-delta' invited
# reading the chassis as part of the TF series, so it is named for what it is.
STEP_LABELS = ["Base strain", "1 TF KO", "2 TF KO", "3 TF KO"]
Y_LABEL = "Total FFA titer (rel. base strain)"

# Every panel plots the SAME phenotype: 'Total Titer', the sum of the five measured FFAs
# (C14:0, C16:0, C18:0, C16:1, C18:1) formed per replicate before averaging. No panel is
# a single-species titer.
PHENOTYPE = "Total Titer"

# One color per mutation ORDER. Six orderings -> the first six palette entries, which is
# exactly the warm-primaries-first block (amber, brick, lilac, wheat, steel blue, gray).
ORDER_COLORS = PLOT_PALETTE[:6]


def collect_total_titer_stats(normalized_replicates, abbreviations, reference_strain):
    """Genotype order -> {frozenset(genes): (mean_f, se_f, n_reps)} for Total Titer.

    Mirrors ffa_total_titer_trajectories.collect_total_titer but also returns the
    standard error, which the per-path panels draw as rung error bars. n == 1 strains
    have no sample SD, so their SE is 0.0 (drawn as no bar) and n is reported so the
    single-replicate rungs stay identifiable in the CSV.
    """
    singles, doubles, triples = {}, {}, {}
    for genotype, ffa_data in normalized_replicates.items():
        genes = parse_genotype(
            genotype, abbreviations, reference_strain=reference_strain
        )
        if not genes:
            continue
        reps = ffa_data[PHENOTYPE]
        valid = reps[~np.isnan(reps)]
        if len(valid) == 0:
            continue
        mean_f = float(np.mean(valid))
        se_f = (
            float(np.std(valid, ddof=1) / np.sqrt(len(valid)))
            if len(valid) > 1
            else 0.0
        )
        target = {1: singles, 2: doubles, 3: triples}.get(len(genes))
        if target is not None:
            target[frozenset(genes)] = (mean_f, se_f, len(valid))
    return singles, doubles, triples


def build_paths(singles, doubles, triples):
    """One row per (triple, ordering): the 4 rungs, their SEs, and accessibility.

    Rows are emitted for all 6 permutations of every triple whose 3 singles and 3
    doubles are present, so the frame is the full path-level view of the panel.
    """
    rows = []
    for tri_key, (f_ijk, se_ijk, n_ijk) in triples.items():
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
        tau = f_ijk - f_ij * f_k - f_ik * f_j - f_jk * f_i + 2 * f_i * f_j * f_k

        for order_idx, (a, b, c) in enumerate(permutations(genes)):
            f_a, se_a, n_a = singles[frozenset([a])]
            f_ab, se_ab, n_ab = doubles[frozenset([a, b])]

            rungs = np.array([1.0, f_a, f_ab, f_ijk])
            ses = np.array([0.0, se_a, se_ab, se_ijk])
            steps = np.diff(rungs)
            # Propagated 1 SE of each step (the base rung is exact by construction).
            step_se = np.sqrt(ses[:-1] ** 2 + ses[1:] ** 2)

            rows.append(
                {
                    "triple": "-".join(genes),
                    "order": f"{a}>{b}>{c}",
                    "order_idx": order_idx,
                    "gene_1": a,
                    "gene_2": b,
                    "gene_3": c,
                    "f_base": 1.0,
                    "f_single": f_a,
                    "f_double": f_ab,
                    "f_triple": f_ijk,
                    "se_single": se_a,
                    "se_double": se_ab,
                    "se_triple": se_ijk,
                    "n_single": n_a,
                    "n_double": n_ab,
                    "n_triple": n_ijk,
                    "tau_multiplicative": tau,
                    # Strict greedy accessibility: every rung strictly improves.
                    "monotone": bool(np.all(steps > 0)),
                    # Relaxed: no step drops by more than its propagated 1 SE.
                    "monotone_within_se": bool(np.all(steps > -step_se)),
                    "min_rung": float(rungs.min()),
                    # Depth of the valley a path must cross before the endpoint.
                    "valley_depth": float(1.0 - rungs[1:3].min()),
                }
            )
    return pd.DataFrame(rows)


def summarize_triples(path_df):
    """Collapse the path frame to one row per triple, with path-spread statistics."""
    grp = path_df.groupby("triple", sort=False)
    summary = grp.agg(
        f_triple=("f_triple", "first"),
        se_triple=("se_triple", "first"),
        tau_multiplicative=("tau_multiplicative", "first"),
        n_monotone=("monotone", "sum"),
        n_monotone_within_se=("monotone_within_se", "sum"),
        max_valley_depth=("valley_depth", "max"),
    ).reset_index()
    # Spread of the intermediate rungs across the 6 orderings: how much the ROUTE
    # matters when the destination is fixed.
    spread = grp.apply(
        lambda d: float(
            np.max(np.r_[d["f_single"], d["f_double"]])
            - np.min(np.r_[d["f_single"], d["f_double"]])
        ),
        include_groups=False,
    ).rename("intermediate_spread")
    return summary.merge(spread, on="triple")


def load_significance():
    """Trigenic significance for the plotted phenotype, keyed by sorted triple name.

    The gene_set labels in that table carry a known defect: load_ffa_data reads the
    Abbreviations sheet with a default header row, which consumes FKH1 as the header, so
    all 36 FKH1-containing triples are labelled with the bare letter 'F'. Only the LABEL
    is affected -- the grouping and therefore every statistic is computed off a consistent
    gene identity -- so remapping 'F' -> 'FKH1' here recovers the correct name. The
    assertion in attach_significance is what proves the values are sound.
    """
    df = pd.read_csv(TRIGENIC_PATH)
    df = df[df["ffa_type"] == PHENOTYPE].copy()
    df["triple"] = df["gene_set"].apply(
        lambda s: "-".join(sorted("FKH1" if g == "F" else g for g in s.split("_")))
    )
    return df[
        [
            "triple",
            "interaction_score",
            "p_value",
            "fdr_corrected_p",
            "significant_p05",
            "significant_fdr05",
        ]
    ]


def attach_significance(summary):
    """Join the canonical significance columns onto the per-triple summary.

    Hard-fails if a triple is missing or if the table's interaction_score disagrees with
    the tau computed here -- either means the results CSV is stale relative to the loaders
    and the panels would annotate figures with statistics for a different model.
    """
    merged = summary.merge(load_significance(), on="triple", how="left", validate="1:1")

    missing = merged.loc[merged["interaction_score"].isna(), "triple"].tolist()
    if missing:
        raise ValueError(
            f"{len(missing)} triples absent from {TRIGENIC_PATH}: {missing[:5]} ... "
            "regenerate it with free_fatty_acid_interactions.py"
        )
    dev = (merged["tau_multiplicative"] - merged["interaction_score"]).abs().max()
    if dev > 1e-8:
        raise ValueError(
            f"tau disagrees with the canonical interaction_score by up to {dev:.3e}; "
            f"{TRIGENIC_PATH} is stale relative to the loaders"
        )
    return merged


def significance_mark(row):
    """(marker, fillstyle, label) encoding the sign and significance of the triple's tau.

    Direction encodes the SIGN of the interaction (up = positive, down = negative); fill
    encodes how much support it has. Kept in black per the repo's solid-black convention
    for pattern marks, so the badge cannot be mistaken for one of the six ordering colors.
    """
    marker = "^" if row["interaction_score"] > 0 else "v"
    if row["significant_fdr05"]:
        return marker, "full", "FDR<0.05"
    if row["significant_p05"]:
        return marker, "bottom", "P<0.05"
    return marker, "none", "n.s."


def select_triples(summary, mode, n_panels):
    """Pick the panel triples: highest endpoint titer, or widest path spread."""
    if mode == "top":
        return summary.nlargest(n_panels, "f_triple")
    if mode == "divergent":
        return summary.nlargest(n_panels, "intermediate_spread")
    raise ValueError(f"unknown selection mode: {mode}")


def plot_path_panels(path_df, selected, mode, out_stem, ncols=3):
    """Grid of per-triple panels, one colored line per mutation ordering."""
    nrows = int(np.ceil(len(selected) / ncols))
    # Shared y across every panel: the panels are compared to each other, so an
    # independent per-panel scale would make equal rises look different and different
    # valleys look equal.
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(55.0 * nrows)),
        squeeze=False,
        sharey=True,
    )
    x = np.arange(4)

    shown = path_df[path_df["triple"].isin(selected["triple"])]
    rungs = shown[["f_base", "f_single", "f_double", "f_triple"]].to_numpy()
    errs = shown[["f_base", "se_single", "se_double", "se_triple"]].to_numpy()
    errs[:, 0] = 0.0
    lo = min(float((rungs - errs).min()), 1.0)
    hi = max(float((rungs + errs).max()), 1.0)
    span = max(hi - lo, 0.1)
    # Headroom for the 6-entry legend, which sits inside the panel.
    y_lim = (lo - 0.06 * span, hi + 0.30 * span)

    for panel_idx, (_, tri) in enumerate(selected.iterrows()):
        ax = axes[panel_idx // ncols][panel_idx % ncols]
        block = path_df[path_df["triple"] == tri["triple"]].sort_values("order_idx")

        for color_idx, (_, p) in enumerate(block.iterrows()):
            y = np.array([p["f_base"], p["f_single"], p["f_double"], p["f_triple"]])
            yerr = np.array([0.0, p["se_single"], p["se_double"], p["se_triple"]])
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                color=ORDER_COLORS[color_idx % len(ORDER_COLORS)],
                linewidth=0.8,
                marker="o",
                markersize=2.0,
                markeredgecolor="black",
                markeredgewidth=0.25,
                elinewidth=0.4,
                capsize=1.0,
                capthick=0.4,
                label=f"{p['gene_1']} → {p['gene_2']}",
                zorder=3,
            )

        ax.axhline(1.0, color="black", linewidth=0.5, linestyle="--", zorder=2)

        ax.set_ylim(*y_lim)

        marker, fillstyle, sig_label = significance_mark(tri)
        ax.set_title(
            f"{tri['triple'].replace('-', '–')}\n"
            f"$f_{{ijk}}$ = {tri['f_triple']:.3f}   "
            f"$\\tau_{{ijk}}$ = {tri['tau_multiplicative']:+.3f} ({sig_label})   "
            f"{int(tri['n_monotone'])}/6 monotone",
            fontsize=6,
            pad=3,
        )
        # Sign-and-support badge, top right inside the panel.
        ax.plot(
            [0.955],
            [0.93],
            transform=ax.transAxes,
            marker=marker,
            fillstyle=fillstyle,
            markersize=4.0,
            markeredgewidth=0.6,
            markeredgecolor="black",
            markerfacecolor="black",
            linestyle="none",
            clip_on=False,
            zorder=5,
        )
        ax.legend(
            title="Single → Double",
            loc="upper left",
            frameon=True,
            fontsize=5,
            title_fontsize=5,
            handlelength=1.4,
            labelspacing=0.25,
            borderpad=0.3,
            handletextpad=0.5,
        )
        if panel_idx % ncols == 0:
            ax.set_ylabel(Y_LABEL)

    for blank_idx in range(len(selected), nrows * ncols):
        axes[blank_idx // ncols][blank_idx % ncols].set_visible(False)

    for panel_idx in range(len(selected)):
        ax = axes[panel_idx // ncols][panel_idx % ncols]
        ax.set_xticks(x)
        ax.set_xticklabels(STEP_LABELS)
        ax.set_xlim(-0.18, 3.18)
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.tick_params(axis="y", which="minor", length=0)
        ax.grid(axis="y", which="both", color="0.85", linewidth=0.3, zorder=0)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_color("black")

    title = {
        "top": "Highest-titer triples",
        "divergent": "Most route-dependent triples",
    }[mode]
    fig.suptitle(
        f"{title}: all 6 KO orders per triple. Total FFA titer relative to the base "
        "production strain (pox1$\\Delta$ faa1$\\Delta$ faa4$\\Delta$); TF knockouts are "
        "additional to it.\nBadge = trigenic interaction: "
        "$\\blacktriangle$ positive $\\tau$, $\\blacktriangledown$ negative $\\tau$; "
        "filled = FDR<0.05, half = P<0.05, open = not significant.",
        fontsize=6,
        y=0.997,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    fig.savefig(f"{out_stem}.png", dpi=300)
    savefig_true_size_svg(fig, f"{out_stem}.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--select",
        choices=["top", "divergent", "both"],
        default="both",
        help="panel selection: highest endpoint titer, widest path spread, or both",
    )
    parser.add_argument("--n-panels", type=int, default=6)
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="write stable filenames (drop the timestamp suffix)",
    )
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    abbreviations = read_abbreviations(RAW_TITER_PATH)
    averaged_df, _loader_abbrev, replicate_dict = load_ffa_data(RAW_TITER_PATH)
    _normalized_df, normalized_replicates = normalize_by_reference(
        averaged_df, replicate_dict
    )
    reference_strain = list(normalized_replicates.keys())[0]

    singles, doubles, triples = collect_total_titer_stats(
        normalized_replicates, abbreviations, reference_strain
    )
    print(f"singles={len(singles)}  doubles={len(doubles)}  triples={len(triples)}")

    path_df = build_paths(singles, doubles, triples)
    summary = attach_significance(summarize_triples(path_df))
    print(f"paths={len(path_df)} over {len(summary)} triples")

    n_pos = int((summary["interaction_score"] > 0).sum())
    print(
        f"trigenic tau on {PHENOTYPE}: {n_pos} positive / "
        f"{len(summary) - n_pos} negative; "
        f"{int(summary['significant_p05'].sum())} at raw P<0.05, "
        f"{int(summary['significant_fdr05'].sum())} at FDR<0.05 "
        f"({int(((summary['interaction_score'] > 0) & summary['significant_p05']).sum())}"
        " positive AND raw P<0.05)"
    )

    n_reachable = int((summary["n_monotone"] > 0).sum())
    n_reachable_se = int((summary["n_monotone_within_se"] > 0).sum())
    n_singles_up = sum(1 for f, _se, _n in singles.values() if f > 1.0)
    print(
        f"singles above the 3-delta base: {n_singles_up}/{len(singles)}\n"
        f"triples with >=1 strictly monotone path: {n_reachable}/{len(summary)} "
        f"(mean {summary['n_monotone'].mean():.2f} of 6 paths)\n"
        f"triples with >=1 monotone-within-1SE path: {n_reachable_se}/{len(summary)}\n"
        f"median intermediate spread across orderings: "
        f"{summary['intermediate_spread'].median():.3f}"
    )

    paths_csv = osp.join(RESULTS_DIR, "ffa_epistatic_paths.csv")
    path_df.to_csv(paths_csv, index=False)
    summary_csv = osp.join(RESULTS_DIR, "ffa_epistatic_path_accessibility.csv")
    summary.sort_values("f_triple", ascending=False).to_csv(summary_csv, index=False)
    print(f"wrote {paths_csv}\nwrote {summary_csv}")

    modes = ["top", "divergent"] if args.select == "both" else [args.select]
    for mode in modes:
        selected = select_triples(summary, mode, args.n_panels)
        suffix = "" if args.no_timestamp else f"_{timestamp()}"
        out_stem = osp.join(IMAGES_DIR, f"ffa_epistatic_path_panels_{mode}{suffix}")
        plot_path_panels(path_df, selected, mode, out_stem)
        print(f"wrote {out_stem}.png / .svg")


if __name__ == "__main__":
    main()
