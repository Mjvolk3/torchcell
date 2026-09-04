# experiments/010-kuzmin-tmi/scripts/pcl6_deletion_paths.py
# [[experiments.010-kuzmin-tmi.scripts.pcl6_deletion_paths]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/pcl6_deletion_paths
#
# Every 1 -> 2 -> 3 deletion path through the pcl6-delta pcl7-delta screen, measured.
#
# WHY THIS SCREEN. YER059W (PCL6) is the hub behind the positive-interaction signal in
# inference_1, and every YER059W record in the 010 build comes from ONE Kuzmin 2020
# query double, YER059W + YIL050W (PCL7), which SGD records as its whole-genome
# duplication paralog. So the triples are all {PCL6, PCL7, X} for an array gene X, and
# the whole lattice is measurable rather than predicted.
#
# WHAT A PATH IS. Deletions are stacked one at a time, so a triple {i,j,k} is reachable
# by 6 orderings. Each ordering is a 4-rung ladder
#     WT (f = 1 by definition) -> single a -> double {a,b} -> triple {i,j,k}
# and the engineering question is whether any ordering climbs without ever stepping
# back. That is the "no backwards moves" criterion: every step must not lower fitness.
#
# WHERE EACH RUNG COMES FROM. All four rungs are published measurements, and the source
# is recorded per rung rather than assumed:
#   triple {PCL6,PCL7,X}   TmfKuzmin2020 "Double/triple mutant fitness" (in screen)
#   double {PCL6,PCL7}     TmfKuzmin2020 "Query single/double mutant fitness" (in screen,
#                          one value for the whole screen since it is the query strain)
#   double {PCL6,X}, {PCL7,X}
#                          DmfKuzmin2020 first (same screen family, same normalization),
#                          DmfCostanzo2016 at 30 C as fallback
#   singles                SmfKuzmin2020 first, SmfCostanzo2016 at 30 C as fallback,
#                          and the in-screen "Array single mutant fitness" is carried
#                          alongside as a cross-check, never silently substituted
#
# UNCERTAINTY. Kuzmin and Costanzo double/triple columns are a SAMPLE SD over colony
# replicates, so the standard error is SD / sqrt(n) with n from the loader's sourced
# replicate constant. Costanzo SMF is already a bootstrap SE and is never divided again.
# Both the SD and the derived SE are written to the CSV; the figures draw the SE.
#
# ARRAY PERTURBATION TYPE. 178 of the 1,098 array strains are temperature-sensitive
# alleles of essential genes, not deletions. The question asked is about removing genes,
# so deletion arrays are the primary set and the TS arrays are reported separately
# rather than pooled.
#
# Run from repo root:
#   ~/miniconda3/envs/torchcell/bin/python \
#     experiments/010-kuzmin-tmi/scripts/pcl6_deletion_paths.py
#
# Outputs:
#   results/pcl6_deletion_paths.csv           one row per (triple, ordering)
#   results/pcl6_deletion_paths_triples.csv   one row per triple
#   results/pcl6_deletion_paths_summary.json  the counts quoted in prose
#   $ASSET_IMAGES_DIR/010-kuzmin-tmi/pcl6_deletion_paths_all.{png,svg}
#   $ASSET_IMAGES_DIR/010-kuzmin-tmi/pcl6_deletion_path_panels.{png,svg}

import json
import os
import os.path as osp
from itertools import permutations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from torchcell.datasets.scerevisiae import DmfCostanzo2016Dataset, SmfCostanzo2016Dataset
from torchcell.datasets.scerevisiae.kuzmin2020 import (
    N_SAMPLES_COMBINED_MUTANT as N_KUZMIN2020,
)
from torchcell.datasets.scerevisiae.kuzmin2020 import (
    DmfKuzmin2020Dataset,
    SmfKuzmin2020Dataset,
    TmfKuzmin2020Dataset,
)
from torchcell.datasets.scerevisiae.costanzo2016 import N_SAMPLES_DOUBLE_MUTANT
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]

RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi", "results")
IMAGES_DIR = osp.join(ASSET_IMAGES_DIR, "010-kuzmin-tmi")

# The query double behind every YER059W record in the 010 build.
QUERY_1, QUERY_2 = "YER059W", "YIL050W"
COMMON = {"YER059W": "PCL6", "YIL050W": "PCL7"}

# Kuzmin 2020 scores a positive interaction symmetrically at tau > +0.08 with p < 0.05.
TAU_CUT, P_CUT = 0.08, 0.05
# Baryshnikova 2010's stringent tier is sign-asymmetric and +0.16 is its positive arm.
# This is the cut the rest of this document calls "strong".
STRONG_CUT = 0.16
TIER_CUTS = [0.08, 0.12, 0.16, 0.20]

STEP_LABELS = ["WT\n(0$\\Delta$)", "1$\\Delta$", "2$\\Delta$", "3$\\Delta$"]
Y_LABEL = "Fitness (relative to wild type)"
ORDER_COLORS = PLOT_PALETTE[:6]

def set_plot_style():
    """Apply the repo figure style.

    This is called inside each plotting function rather than once at import, because
    constructing any of the scerevisiae datasets resets matplotlib's rcParams to
    16 pt DejaVu Sans. Setting the style at module level looks correct and silently
    produces a figure with 16 pt ticks.

    That same reset also flips ``savefig.bbox`` to ``tight``, which recrops the canvas
    and defeats the fixed panel width, so the saved figure comes out a few percent wider
    than PANEL_WIDTHS_MM asked for. Both are restored here.
    """
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "savefig.bbox": None,
            "savefig.pad_inches": 0.0,
            "axes.labelsize": 6,
            "axes.titlesize": 6,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 5,
            "legend.title_fontsize": 5,
            "figure.titlesize": 6,
            "svg.fonttype": "none",
            "axes.linewidth": 0.5,
        }
    )


def root(name: str) -> str:
    return osp.join(DATA_ROOT, "data/torchcell", name)


def triples_frame() -> pd.DataFrame:
    """The pcl6-delta pcl7-delta trigenic screen, one row per array gene."""
    df = TmfKuzmin2020Dataset(root=root("tmf_kuzmin2020")).df
    q1, q2 = df["Query systematic name_1"], df["Query systematic name_2"]
    keep = ((q1 == QUERY_1) & (q2 == QUERY_2)) | ((q1 == QUERY_2) & (q2 == QUERY_1))
    sub = df[keep].copy()

    out = pd.DataFrame(
        {
            "array_gene": sub["Array systematic name"],
            "array_allele": sub["Array allele name"],
            "array_perturbation_type": sub["array_perturbation_type"],
            "f_triple": sub["Double/triple mutant fitness"],
            "sd_triple": sub["Double/triple mutant fitness standard deviation"],
            "f_query_double_in_screen": sub["Query single/double mutant fitness"],
            "f_array_single_in_screen": sub["Array single mutant fitness"],
            "tau": sub["Adjusted genetic interaction score (epsilon or tau)"],
            "tau_p_value": sub["P-value"],
        }
    )
    # One array gene appears twice under two alleles; keep the better-replicated row.
    out = out.sort_values("sd_triple").drop_duplicates("array_gene", keep="first")
    return out.reset_index(drop=True)


def single_lookup(genes: set[str]) -> dict[str, tuple[float, float, str]]:
    """gene -> (fitness, SE, source). Kuzmin 2020 first, Costanzo 2016 at 30 C after.

    Costanzo's SMF stddev column is a BOOTSTRAP SE over control screens, already an
    error on the mean, so it is used as-is. Kuzmin's is a sample SD over colony
    replicates and is divided by sqrt(n).
    """
    out: dict[str, tuple[float, float, str]] = {}

    ks = SmfKuzmin2020Dataset(root=root("smf_kuzmin2020")).df
    # A Kuzmin single is the non-HO half of a ho-delta x gene-delta cross.
    ho = "YDL227C"
    orf = np.where(ks["ORF1"] == ho, ks["ORF2"], ks["ORF1"])
    ks = ks.assign(gene=orf)
    ks = ks[ks["gene"].isin(genes) & (ks["Mutant type"] == "Single mutant")]
    for gene, block in ks.groupby("gene"):
        f = float(block["Fitness"].mean())
        sd = float(block["St.dev."].mean())
        if np.isfinite(f):
            out[str(gene)] = (f, sd / np.sqrt(N_KUZMIN2020), "SmfKuzmin2020")

    cs = SmfCostanzo2016Dataset(root=root("smf_costanzo2016")).df
    cs = cs[
        cs["Systematic gene name"].isin(genes)
        & (cs["Temperature"] == 30)
        & cs["perturbation_type"].str.contains("deletion")
    ]
    for gene, block in cs.groupby("Systematic gene name"):
        if gene in out:
            continue
        f = float(block["Single mutant fitness"].mean())
        se = float(block["Single mutant fitness stddev"].mean())
        if np.isfinite(f):
            out[str(gene)] = (f, se, "SmfCostanzo2016")
    return out


def double_lookup(partners: set[str]) -> dict[tuple[str, str], tuple[float, float, str]]:
    """(query, partner) -> (fitness, SE, source) for PCL6+X and PCL7+X.

    Kuzmin 2020's digenic controls were run in the same screen family as the trigenic
    data, so they are preferred; Costanzo 2016 at 30 C fills the rest. Both columns are
    a sample SD over colony replicates, so both are divided by sqrt(n).
    """
    out: dict[tuple[str, str], tuple[float, float, str]] = {}

    kd = DmfKuzmin2020Dataset(root=root("dmf_kuzmin2020")).df
    q = kd["Query systematic name no ho"]
    a = kd["Array systematic name"]
    for query in (QUERY_1, QUERY_2):
        block = kd[((q == query) & a.isin(partners))]
        for partner, rows in block.groupby("Array systematic name"):
            f = float(rows["Double/triple mutant fitness"].mean())
            sd = float(rows["Double/triple mutant fitness standard deviation"].mean())
            if np.isfinite(f):
                out[(query, str(partner))] = (
                    f,
                    sd / np.sqrt(N_KUZMIN2020),
                    "DmfKuzmin2020",
                )

    cd = DmfCostanzo2016Dataset(root=root("dmf_costanzo2016")).df
    cd = cd[cd["Temperature"] == 30]
    qn, an = cd["Query Systematic Name"], cd["Array Systematic Name"]
    for query in (QUERY_1, QUERY_2):
        hit = ((qn == query) & an.isin(partners)) | ((an == query) & qn.isin(partners))
        block = cd[hit].copy()
        if block.empty:
            continue
        block["partner"] = np.where(
            block["Query Systematic Name"] == query,
            block["Array Systematic Name"],
            block["Query Systematic Name"],
        )
        for partner, rows in block.groupby("partner"):
            if (query, str(partner)) in out:
                continue
            f = float(rows["Double mutant fitness"].mean())
            sd = float(rows["Double mutant fitness standard deviation"].mean())
            if np.isfinite(f):
                out[(query, str(partner))] = (
                    f,
                    sd / np.sqrt(N_SAMPLES_DOUBLE_MUTANT),
                    "DmfCostanzo2016",
                )
    return out


def build_paths(tri: pd.DataFrame, singles, doubles, f_qq: float, se_qq: float):
    """One row per (triple, ordering) with the 4 rungs and the accessibility verdicts."""
    rows = []
    for rec in tri.itertuples(index=False):
        x = rec.array_gene
        genes = (QUERY_1, QUERY_2, x)
        if not all(g in singles for g in genes):
            continue
        if (QUERY_1, x) not in doubles or (QUERY_2, x) not in doubles:
            continue

        pair_f = {
            frozenset((QUERY_1, QUERY_2)): (f_qq, se_qq, "TmfKuzmin2020 query strain"),
            frozenset((QUERY_1, x)): doubles[(QUERY_1, x)],
            frozenset((QUERY_2, x)): doubles[(QUERY_2, x)],
        }
        se_triple = float(rec.sd_triple) / np.sqrt(N_KUZMIN2020)

        for order_idx, (a, b, c) in enumerate(permutations(genes)):
            f_a, se_a, src_a = singles[a]
            f_ab, se_ab, src_ab = pair_f[frozenset((a, b))]

            rungs = np.array([1.0, f_a, f_ab, float(rec.f_triple)])
            ses = np.array([0.0, se_a, se_ab, se_triple])
            steps = np.diff(rungs)
            step_se = np.sqrt(ses[:-1] ** 2 + ses[1:] ** 2)

            rows.append(
                {
                    "array_gene": x,
                    "array_allele": rec.array_allele,
                    "array_perturbation_type": rec.array_perturbation_type,
                    "triple": f"{QUERY_1}+{QUERY_2}+{x}",
                    "order": f"{a}>{b}>{c}",
                    "order_idx": order_idx,
                    "gene_1": a,
                    "gene_2": b,
                    "gene_3": c,
                    "route": "paralog double" if {a, b} == {QUERY_1, QUERY_2} else "mixed double",
                    "f_wt": 1.0,
                    "f_single": f_a,
                    "f_double": f_ab,
                    "f_triple": float(rec.f_triple),
                    "se_single": se_a,
                    "se_double": se_ab,
                    "se_triple": se_triple,
                    "sd_triple": float(rec.sd_triple),
                    "source_single": src_a,
                    "source_double": src_ab,
                    "f_array_single_in_screen": float(rec.f_array_single_in_screen),
                    "tau": float(rec.tau),
                    "tau_p_value": float(rec.tau_p_value),
                    # Strict: every deletion must not lower fitness, starting from WT.
                    "monotone": bool(np.all(steps >= 0)),
                    # Relaxed: no step falls by more than its propagated 1 SE.
                    "monotone_within_se": bool(np.all(steps > -step_se)),
                    # Allows the first deletion to cost, nothing after it.
                    "monotone_after_first": bool(np.all(steps[1:] >= 0)),
                    "step_1": float(steps[0]),
                    "step_2": float(steps[1]),
                    "step_3": float(steps[2]),
                    "min_rung": float(rungs.min()),
                    "valley_depth": float(1.0 - rungs[1:3].min()),
                    "endpoint_gain": float(rungs[3] - 1.0),
                }
            )
    return pd.DataFrame(rows)


def summarize_triples(paths: pd.DataFrame) -> pd.DataFrame:
    agg = paths.groupby("triple").agg(
        array_gene=("array_gene", "first"),
        array_allele=("array_allele", "first"),
        array_perturbation_type=("array_perturbation_type", "first"),
        f_triple=("f_triple", "first"),
        se_triple=("se_triple", "first"),
        tau=("tau", "first"),
        tau_p_value=("tau_p_value", "first"),
        n_paths=("order", "size"),
        n_monotone=("monotone", "sum"),
        n_monotone_within_se=("monotone_within_se", "sum"),
        n_monotone_after_first=("monotone_after_first", "sum"),
        max_valley_depth=("valley_depth", "max"),
        min_valley_depth=("valley_depth", "min"),
        best_min_rung=("min_rung", "max"),
        endpoint_gain=("endpoint_gain", "first"),
    )
    agg["positive_call"] = (agg["tau"] > TAU_CUT) & (agg["tau_p_value"] < P_CUT)
    return agg.reset_index().sort_values("f_triple", ascending=False)


def tier_table(tri_summary: pd.DataFrame) -> pd.DataFrame:
    """What each interaction tier costs in fitness.

    Selecting on interaction and selecting on fitness pull against each other here, and
    this is the table that shows by how much: for each tau cut, how many triples clear
    it, the best endpoint among them, how many beat wild type, and how many have any
    route that avoids a backwards move.
    """
    rows = []
    for cut in TIER_CUTS:
        for label, mask in (
            ("called (P<0.05)", (tri_summary["tau"] > cut) & (tri_summary["tau_p_value"] < P_CUT)),
            ("magnitude only", tri_summary["tau"] > cut),
        ):
            block = tri_summary[mask]
            rows.append(
                {
                    "tau_cut": cut,
                    "criterion": label,
                    "n_triples": int(len(block)),
                    "max_f_triple": float(block["f_triple"].max()) if len(block) else np.nan,
                    "n_above_wt": int((block["f_triple"] > 1.0).sum()),
                    "n_with_monotone_route": int((block["n_monotone"] > 0).sum()),
                    "n_with_monotone_within_se_route": int(
                        (block["n_monotone_within_se"] > 0).sum()
                    ),
                }
            )
    return pd.DataFrame(rows)


def _style(ax):
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(axis="y", which="minor", length=0)
    ax.grid(axis="y", which="both", color="0.85", linewidth=0.3, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color("black")


def plot_all_paths(paths: pd.DataFrame, out_stem: str):
    """Panel a: every path. Panel b: median ladder by interaction call. Panel c: attrition."""
    set_plot_style()
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(58.0)),
        gridspec_kw={"width_ratios": [1.3, 0.85, 1.05]},
    )
    x = np.arange(4)
    cols = ["f_wt", "f_single", "f_double", "f_triple"]
    rungs = paths[cols].to_numpy()

    ax = axes[0]
    strict = paths["monotone"].to_numpy()
    loose = paths["monotone_within_se"].to_numpy() & ~strict
    for mask, color, width, alpha, z in (
        (~(strict | loose), "0.72", 0.20, 0.35, 2),
        (loose, PLOT_PALETTE[0], 0.5, 0.85, 3),
        (strict, PLOT_PALETTE[1], 0.9, 1.0, 4),
    ):
        if mask.sum() == 0:
            continue
        block = rungs[mask]
        ax.plot(x, block.T, color=color, linewidth=width, alpha=alpha, zorder=z)
    ax.axhline(1.0, color="black", linewidth=0.5, linestyle="--", zorder=5)
    ax.set_ylabel(Y_LABEL)
    ax.set_title(
        f"a  All {len(paths):,} deletion paths\n"
        f"{int(strict.sum())} never step back, "
        f"{int(loose.sum())} within 1 SE",
        fontsize=6,
        loc="left",
        pad=3,
    )
    ax.legend(
        handles=[
            Line2D([], [], color=PLOT_PALETTE[1], lw=1.0, label="no backwards move"),
            Line2D([], [], color=PLOT_PALETTE[0], lw=1.0, label="none beyond 1 SE"),
            Line2D([], [], color="0.72", lw=1.0, label="steps back"),
        ],
        loc="lower left",
        frameon=True,
        fontsize=5,
        handlelength=1.4,
        labelspacing=0.25,
        borderpad=0.3,
    )

    # Panel b: attrition under the strict rule, applied one deletion at a time.
    ax = axes[1]
    steps = paths[["step_1", "step_2", "step_3"]].to_numpy()
    alive = np.ones(len(paths), dtype=bool)
    frac = [1.0]
    for s in range(3):
        alive = alive & (steps[:, s] >= 0)
        frac.append(alive.mean())
    ax.bar(x, frac, color=PLOT_PALETTE[2], edgecolor="black", linewidth=0.4, zorder=3)
    for xi, fv in zip(x, frac):
        ax.text(xi, fv + 0.025, f"{fv:.3f}", ha="center", va="bottom", fontsize=5)
    ax.set_ylim(0, 1.14)
    ax.set_ylabel("Fraction of paths still climbing")
    ax.set_title(
        "b  Where the ladder breaks\n"
        f"the first deletion alone ends {1 - frac[1]:.1%} of paths",
        fontsize=6,
        loc="left",
        pad=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(STEP_LABELS)
    ax.set_xlim(-0.6, 3.6)
    _style(ax)

    # Panel c: does a positive trigenic interaction buy a fitter strain?
    ax = axes[2]
    tri = paths.drop_duplicates("triple")
    pos = (tri["tau"] > TAU_CUT) & (tri["tau_p_value"] < P_CUT)
    ax.axhline(1.0, color="black", linewidth=0.5, linestyle="--", zorder=2)
    ax.axvline(TAU_CUT, color="black", linewidth=0.5, linestyle=":", zorder=2)
    ax.scatter(tri.loc[~pos, "tau"], tri.loc[~pos, "f_triple"], s=1.6,
               color="0.72", linewidths=0, zorder=3, label=f"not called (n={int((~pos).sum())})")
    ax.scatter(tri.loc[pos, "tau"], tri.loc[pos, "f_triple"], s=4.0,
               color=PLOT_PALETTE[0], edgecolor="black", linewidths=0.25, zorder=4,
               label=f"positive $\\tau$ call (n={int(pos.sum())})")
    r = float(np.corrcoef(tri["tau"], tri["f_triple"])[0, 1])
    ax.set_xlabel("Trigenic interaction score $\\tau$")
    ax.set_ylabel("Triple-mutant fitness")
    ax.set_title(
        "c  Interaction against endpoint\n"
        f"r = {r:+.2f}; {int((tri['f_triple'] > 1.0).sum())} of {len(tri)} triples beat wild type",
        fontsize=6,
        loc="left",
        pad=3,
    )
    ax.legend(loc="lower right", frameon=True, fontsize=5, handlelength=1.0,
              labelspacing=0.25, borderpad=0.3, scatterpoints=1)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(axis="x", which="minor", length=0)
    _style(ax)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(STEP_LABELS)
    axes[0].set_xlim(-0.35, 3.35)
    _style(axes[0])

    fig.suptitle(
        f"Stacking deletions on the {COMMON[QUERY_1].lower()}$\\Delta$ "
        f"{COMMON[QUERY_2].lower()}$\\Delta$ trigenic screen: all six orders per triple, "
        "every rung a published measurement (Kuzmin 2020, Costanzo 2016 fallback).",
        fontsize=6,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(f"{out_stem}.png", dpi=300)
    savefig_true_size_svg(fig, f"{out_stem}.svg")
    plt.close(fig)


def plot_path_panels(
    paths: pd.DataFrame, selected: pd.DataFrame, out_stem: str, subtitle: str, ncols=3
):
    set_plot_style()
    nrows = int(np.ceil(len(selected) / ncols))
    # The header carries a two-line suptitle and the shared route legend, so it is a
    # fixed band in mm rather than a fraction of a height that changes with nrows.
    header_mm, panel_mm = 16.0, 52.0
    total_mm = panel_mm * nrows + header_mm
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(total_mm)),
        squeeze=False,
        sharey=True,
    )
    x = np.arange(4)

    shown = paths[paths["triple"].isin(selected["triple"])]
    r = shown[["f_wt", "f_single", "f_double", "f_triple"]].to_numpy()
    e = shown[["f_wt", "se_single", "se_double", "se_triple"]].to_numpy()
    e[:, 0] = 0.0
    lo = min(float((r - e).min()), 1.0)
    hi = max(float((r + e).max()), 1.0)
    span = max(hi - lo, 0.1)
    y_lim = (lo - 0.08 * span, hi + 0.08 * span)

    # permutations() is deterministic, so order_idx labels the same route in every
    # panel and one shared legend replaces six crowded per-panel ones. X is the array
    # gene named in each panel title.
    q1, q2 = COMMON[QUERY_1], COMMON[QUERY_2]
    route_labels = [
        f"{q1} → {q2}", f"{q1} → X", f"{q2} → {q1}",
        f"{q2} → X", f"X → {q1}", f"X → {q2}",
    ]

    for panel_idx, (_, tri) in enumerate(selected.iterrows()):
        ax = axes[panel_idx // ncols][panel_idx % ncols]
        block = paths[paths["triple"] == tri["triple"]].sort_values("order_idx")
        for _, p in block.iterrows():
            y = np.array([p["f_wt"], p["f_single"], p["f_double"], p["f_triple"]])
            yerr = np.array([0.0, p["se_single"], p["se_double"], p["se_triple"]])
            ax.errorbar(
                x, y, yerr=yerr,
                color=ORDER_COLORS[int(p["order_idx"]) % len(ORDER_COLORS)],
                linewidth=0.8, marker="o", markersize=2.0,
                markeredgecolor="black", markeredgewidth=0.25,
                elinewidth=0.4, capsize=1.0, capthick=0.4, zorder=3,
            )
        ax.axhline(1.0, color="black", linewidth=0.5, linestyle="--", zorder=2)
        ax.set_ylim(*y_lim)
        call = "called" if tri["positive_call"] else "not called"
        ax.set_title(
            f"X = {tri['array_gene']} ({tri['array_allele'].replace('_delta', '$\\Delta$')})\n"
            f"$f_{{ijk}}$ = {tri['f_triple']:.3f},  "
            f"$\\tau$ = {tri['tau']:+.3f} (P = {tri['tau_p_value']:.3f}, {call})\n"
            f"{int(tri['n_monotone'])} of 6 routes never step back",
            fontsize=6, pad=3,
        )
        if panel_idx % ncols == 0:
            ax.set_ylabel(Y_LABEL)

    for blank in range(len(selected), nrows * ncols):
        axes[blank // ncols][blank % ncols].set_visible(False)
    for panel_idx in range(len(selected)):
        ax = axes[panel_idx // ncols][panel_idx % ncols]
        ax.set_xticks(x)
        ax.set_xticklabels(STEP_LABELS)
        ax.set_xlim(-0.2, 3.2)
        _style(ax)

    fig.legend(
        handles=[
            Line2D([], [], color=ORDER_COLORS[i], lw=0.9, marker="o", markersize=2.4,
                   markeredgecolor="black", markeredgewidth=0.25, label=route_labels[i])
            for i in range(6)
        ],
        title="Deletion order, 1$\\Delta$ → 2$\\Delta$",
        loc="upper center", bbox_to_anchor=(0.5, 1 - 7.5 / total_mm), ncols=6,
        frameon=True, fontsize=5, title_fontsize=5, handlelength=1.4,
        columnspacing=1.2, borderpad=0.3, handletextpad=0.5,
    )
    fig.suptitle(subtitle, fontsize=6, y=1 - 1.5 / total_mm)
    fig.tight_layout(rect=(0, 0, 1, 1 - header_mm / total_mm))
    fig.savefig(f"{out_stem}.png", dpi=300)
    savefig_true_size_svg(fig, f"{out_stem}.svg")
    plt.close(fig)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    tri = triples_frame()
    print(f"{QUERY_1}+{QUERY_2} trigenic records: {len(tri)}")
    print(tri["array_perturbation_type"].value_counts().to_string())

    f_qq = float(tri["f_query_double_in_screen"].iloc[0])
    n_qq = tri["f_query_double_in_screen"].round(4).nunique()
    print(f"query double fitness in screen: {f_qq:.4f} ({n_qq} distinct value(s))")

    arrays = set(tri["array_gene"])
    singles = single_lookup(arrays | {QUERY_1, QUERY_2})
    doubles = double_lookup(arrays)
    for g in (QUERY_1, QUERY_2):
        f, se, src = singles[g]
        print(f"single {g} ({COMMON[g]}): {f:.4f} +/- {se:.4f} SE  [{src}]")

    # The two SMF sources disagree on the query singles, and the disagreement flips the
    # sign of the paralog double's digenic term. Kuzmin's own singles are used because
    # they share the screen's normalization with the double and the triple; Costanzo's
    # are reported so the choice is visible rather than buried.
    cross = SmfCostanzo2016Dataset(root=root("smf_costanzo2016")).df
    cross = cross[(cross["Temperature"] == 30) & cross["perturbation_type"].str.contains("deletion")]
    costanzo_single = {
        g: float(cross.loc[cross["Systematic gene name"] == g, "Single mutant fitness"].mean())
        for g in (QUERY_1, QUERY_2)
    }
    eps_kuzmin = f_qq - singles[QUERY_1][0] * singles[QUERY_2][0]
    eps_costanzo = f_qq - costanzo_single[QUERY_1] * costanzo_single[QUERY_2]
    print(f"paralog double epsilon: {eps_kuzmin:+.4f} on Kuzmin singles, "
          f"{eps_costanzo:+.4f} on Costanzo singles")
    # The paralog double has no replicate SD of its own in the trigenic table; use the
    # median SE of the digenic doubles as its error bar rather than drawing none.
    se_qq = float(np.median([v[1] for v in doubles.values()]))

    paths = build_paths(tri, singles, doubles, f_qq, se_qq)
    deletions = paths[paths["array_perturbation_type"] == "KanMX_deletion"].copy()
    print(f"\ncomplete paths: {len(paths):,} over {paths['triple'].nunique():,} triples")
    print(f"deletion-array paths: {len(deletions):,} over "
          f"{deletions['triple'].nunique():,} triples")
    print(deletions["source_double"].value_counts().to_string())

    tri_summary = summarize_triples(deletions)
    paths.to_csv(osp.join(RESULTS_DIR, "pcl6_deletion_paths.csv"), index=False)
    tri_summary.to_csv(
        osp.join(RESULTS_DIR, "pcl6_deletion_paths_triples.csv"), index=False
    )

    strict = int(deletions["monotone"].sum())
    loose = int(deletions["monotone_within_se"].sum())
    after = int(deletions["monotone_after_first"].sum())
    best = tri_summary.iloc[0]
    pos = tri_summary["positive_call"]

    summary = {
        "query_double": f"{QUERY_1}+{QUERY_2}",
        "query_double_fitness": f_qq,
        "single_fitness_kuzmin2020": {g: singles[g][0] for g in (QUERY_1, QUERY_2)},
        "single_fitness_costanzo2016": costanzo_single,
        "paralog_double_epsilon_on_kuzmin_singles": float(eps_kuzmin),
        "paralog_double_epsilon_on_costanzo_singles": float(eps_costanzo),
        "n_trigenic_records": int(len(tri)),
        "n_deletion_arrays": int((tri["array_perturbation_type"] == "KanMX_deletion").sum()),
        "n_ts_arrays": int((tri["array_perturbation_type"] == "temperature_sensitive").sum()),
        "n_triples_complete_lattice": int(deletions["triple"].nunique()),
        "n_paths": int(len(deletions)),
        "n_paths_monotone": strict,
        "n_paths_monotone_within_se": loose,
        "n_paths_monotone_after_first": after,
        "n_triples_any_monotone": int((tri_summary["n_monotone"] > 0).sum()),
        "n_triples_any_monotone_within_se": int(
            (tri_summary["n_monotone_within_se"] > 0).sum()
        ),
        "n_triples_positive_call": int(pos.sum()),
        "n_triples_above_wt": int((tri_summary["f_triple"] > 1.0).sum()),
        "max_triple_fitness": float(tri_summary["f_triple"].max()),
        "best_triple": str(best["triple"]),
        "median_triple_fitness": float(tri_summary["f_triple"].median()),
        "median_valley_depth": float(deletions["valley_depth"].median()),
        "tiers": tier_table(tri_summary).to_dict("records"),
    }
    with open(osp.join(RESULTS_DIR, "pcl6_deletion_paths_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n" + json.dumps(summary, indent=2))

    tiers = tier_table(tri_summary)
    tiers.to_csv(osp.join(RESULTS_DIR, "pcl6_deletion_paths_tiers.csv"), index=False)
    print("\n=== interaction tiers against the fitness goal ===")
    print(tiers.to_string(index=False))

    plot_all_paths(deletions, osp.join(IMAGES_DIR, "pcl6_deletion_paths_all"))

    # Ranked by endpoint fitness. Naming the ranking matters: these panels are the
    # fittest strains in the screen, and most of them carry a weak or uncalled tau.
    plot_path_panels(
        deletions,
        tri_summary.head(6),
        osp.join(IMAGES_DIR, "pcl6_deletion_path_panels"),
        subtitle=(
            f"Six fittest triples of the {COMMON[QUERY_1].lower()}$\\Delta$ "
            f"{COMMON[QUERY_2].lower()}$\\Delta$ screen, RANKED BY ENDPOINT FITNESS, not "
            "by interaction. All six deletion orders each, error bars 1 SE.\nTwo of the "
            "six routes pass through the paralog double, whose fitness is one value for "
            "the whole screen."
        ),
    )

    # Ranked by interaction, at the strong cut the rest of the document uses.
    strong = tri_summary[
        (tri_summary["tau"] > STRONG_CUT) & (tri_summary["tau_p_value"] < P_CUT)
    ].sort_values("tau", ascending=False)
    plot_path_panels(
        deletions,
        strong,
        osp.join(IMAGES_DIR, "pcl6_deletion_path_panels_strong"),
        subtitle=(
            f"Every triple in the screen clearing the strong positive tier, "
            f"$\\tau > +{STRONG_CUT:.2f}$ with P < {P_CUT:.2f}, ranked by $\\tau$. "
            f"{int((strong['f_triple'] > 1.0).sum())} of {len(strong)} reach wild type; "
            f"{int((strong['n_monotone_within_se'] > 0).sum())} of {len(strong)} have a "
            "route that avoids a backwards move within 1 SE.\n"
            "All six deletion orders each, error bars 1 SE."
        ),
        ncols=max(len(strong), 1),
    )
    print(f"\nwrote figures to {IMAGES_DIR}")


if __name__ == "__main__":
    main()
