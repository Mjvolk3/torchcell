# experiments/020-cachera-betaxanthin/scripts/plot_merzbacher_comparison.py
# [[experiments.020-cachera-betaxanthin.merzbacher-comparison]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/020-cachera-betaxanthin/scripts/plot_merzbacher_comparison
"""Comparison panels against Merzbacher 2025 Fig 4, on their own 639 test genes.

THREE SETS APPEAR THROUGHOUT, always in the same colors:
    truth    their released labels           108 low / 431 medium / 100 high
    Cachera  their shipped model predictions gene level, majority vote over flux samples
    ours     the CGT regression, binned      absolute thresholds AND rank-matched

WHY RANK-MATCHING, AND WHY IT IS APPLIED TO BOTH SIDES. Their models emit CLASSES; ours emits
a regression value. Turning ours into classes with their absolute cuts (0.40/0.65 on a
train-pool min-max scale) charges us for calibration as well as for ordering -- and we are
trained on a correlation objective, so our predictions are compressed and almost everything
lands in the middle band. Sorting our predictions and cutting at the class COUNTS removes the
scale and leaves only the ordering.

Doing that to our side alone would be a rigged comparison: forcing the true marginal is
information their classifier never got. Their `fig4c_*.csv` ships per-flux-sample class
PROBABILITIES (`score0/1/2`) with `knockout_name`, so their model also has a continuous
gene-level score -- E[class] = 0*p_low + 1*p_med + 2*p_high, averaged over each gene's flux
samples. Both sides are therefore rank-matched by the SAME rule, and that panel is the one
apples-to-apples comparison in this figure.

Panels (each written as .png AND true-size .svg):
    a  gene-level accuracy, their 4 models vs our grid cells         (their Fig 4b analogue)
    b  predicted class distribution: truth vs Cachera vs ours        (the "everything is
                                                                      medium" failure mode)
    c  per-class recall, grouped                                     (what accuracy hides)
    d  row-normalized confusion matrices, matched rank-binning       (apples-to-apples)

    python experiments/020-cachera-betaxanthin/scripts/plot_merzbacher_comparison.py
"""

from __future__ import annotations

import glob
import json
import os
import os.path as osp
import re
import sys
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import hypergeom, norm, pearsonr, rankdata, spearmanr
from sklearn.metrics import roc_auc_score, roc_curve

from torchcell.timestamp import timestamp
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    PLOT_PALETTE_FILL,
    mm_to_in,
    savefig_true_size_svg,
)

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from build_merzbacher_split import (  # noqa: E402
    MERZBACHER_THRESHOLDS,
    load_merzbacher_split,
    read_cachera_raw,
)

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "020-cachera-betaxanthin", "results")
SPLIT_PATH = osp.join(RESULTS_DIR, "merzbacher_nested_split.json")
BASELINE_PATH = osp.join(RESULTS_DIR, "merzbacher_baseline_analysis.json")
#: yeast-GEM defines WHAT FCL CAN SEE. Their features are flux samples from this genome-scale
#: metabolic model, so a deletion of a gene the model does not contain changes no flux and
#: yields no feature -- the method has nothing to predict from. Their own test set is
#: 639/640 (99.8 %) inside it, which is the empirical confirmation.
GEM_PATH = osp.join(DATA_ROOT, "data", "scerevisiae", "yeast-GEM.xlsx")
FIG4_DIR = osp.join(
    DATA_ROOT,
    "data",
    "merzbacher2025_fcl",
    "deletionprediction-main",
    "figures",
    "fig4",
)
OUT_DIR = osp.join(ASSET_IMAGES_DIR, "020-cachera-betaxanthin")

#: low / medium / high. Red -> sand -> blue echoes their own Fig 4 encoding while staying
#: inside the repo palette (indices 1, 3, 4), and it is ORDERED, which a categorical hue
#: assignment would not be. The PALE variants were tried and rejected -- washed out.
CLASS_COLORS = [PLOT_PALETTE[1], PLOT_PALETTE[3], PLOT_PALETTE[4]]
CLASS_NAMES = ["low", "medium", "high"]
#: TWO MODELS, TWO PRIMARY COLORS, and the binning is encoded by LIGHTNESS within a color --
#: the sanctioned two-level bar (line color = the deployed binning, its pale companion from
#: PLOT_PALETTE_FILL = rank-matched). Hue therefore carries MODEL and lightness carries
#: BINNING, so the two encodings never collide.
#: Only palette indices 0-5 (the primaries) are used anywhere in this figure.
CGT_C, CGT_FILL = PLOT_PALETTE[0], PLOT_PALETTE_FILL[0]  # orange
RF_C, RF_FILL = PLOT_PALETTE[5], PLOT_PALETTE_FILL[5]  # gray
TRUTH_C = PLOT_PALETTE[2]  # purple

#: Their best model, and the ONLY Cachera model shown. Fig 4b's other three are strictly worse
#: on their own gene-level numbers (0.56-0.64 against RF's 0.700), so carrying them would add
#: three rows that only restate "RF wins among their models".
RF_MODEL = "RandomForestClassifier_Resampled"
#: THE METHOD, NOT THE DATASET. Cachera is the betaxanthin SCREEN we score against;
#: Flux Cone Learning (FCL) is Merzbacher et al.'s METHOD, and RandomForest_Resampled is its
#: best variant. Labelling their model "Cachera RF" conflated the data with the approach --
#: the comparison is CGT vs FCL, both evaluated on the Cachera screen.
RF_LABEL = "FCL RF"
CGT_LABEL = "CGT"
#: Axis label for their gene-level score, E[class] = 0*p_low + 1*p_med + 2*p_high.
#: DELIBERATELY NOT mathtext. `$\mathbb{E}$` renders U+1D53C (double-struck capital E),
#: which Arial does not contain -- and because the repo standard sets
#: ``svg.fonttype: "none"`` the SVG keeps it as a text run in Arial rather than as a path,
#: so the glyph is missing by the time the note is rendered to PDF. A plain "E" is in every
#: font we use and survives the whole matplotlib -> SVG -> PDF chain.
FCL_AXIS_LABEL = "FCL RF   E[class]"

#: A sequential map built from the palette orange, replacing matplotlib's built-in `Oranges`
#: -- which is close to our hue but is NOT our color, and a figure that mixes the two reads as
#: two different oranges.
CGT_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "tc_orange", ["#FFFFFF", PLOT_PALETTE_FILL[0], PLOT_PALETTE[0], PLOT_PALETTE[6]]
)
_SETTING_TAG = re.compile(r"^s\d+_")

#: EVERY NUMBER THE NOTE QUOTES, written to a results file next to the figures.
#:
#: The note's nine tables used to be transcribed from stdout, which has two costs. The
#: figures are built from `$DATA_ROOT/test-predictions/`, a directory that GROWS as Delta
#: cells finish -- so a re-render silently moves the numbers and nothing on disk records
#: what they were. And several quoted values (Fig 1's Pearson column, Fig 3's hypergeometric
#: p-values) were never computed by this script at all; they came from a side calculation
#: with no artifact. Both are the STRICT RULE in CLAUDE.md: a number in a note comes from a
#: committed script reading real result files.
#:
#: Each fig_* function records what it DRAWS, at the point it draws it, so the JSON cannot
#: drift from the figure the way a recomputation in a second place would.
STATS: dict[str, Any] = {}

#: THE CELL PANELS b/c/d FOCUS ON, and it is SELECTED BY VALIDATION, never by its test score.
#: `s09_L6_maskon_lr0.0001_yj` holds the highest `val/betaxanthin/pearson_per_feature` (0.3639)
#: of the finished 020 grid cells. Picking the cell that looks best on THEIR test genes would
#: be selection on the test set -- with ~10 cells at sigma ~ 0.03 that is worth about 2 sigma
#: of free improvement, i.e. most of the gap being argued about.
#:
#: This was originally `sorted(ours)[0]`, which silently selected `s00` -- the WORST cell in
#: the grid (val 0.2158, top-50 precision 0.100). The panels rendered fine and understated our
#: side by a wide margin, which is the failure mode a default should never have.
FOCUS_SETTING = "s09_L6_maskon_lr0.0001_yj"


def style() -> None:
    """Repo figure standards: Arial 6 pt, boxed axes, editable SVG text."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 6,
            "axes.labelsize": 6,
            "axes.titlesize": 6,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.linewidth": 0.5,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "svg.fonttype": "none",
            "figure.dpi": 300,
        }
    )


def tenth_grid(ax: plt.Axes) -> None:
    """Gridline every 0.1, labelled every 0.2 -- the repo convention for 0-1 axes."""
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
    ax.grid(axis="y", which="both", lw=0.3, color="0.85", zorder=0)
    ax.tick_params(axis="y", which="minor", length=0)
    ax.set_axisbelow(True)


def save(fig: plt.Figure, name: str, ts: str) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    stem = osp.join(OUT_DIR, f"{name}_{ts}")
    fig.savefig(f"{stem}.png", dpi=300)
    savefig_true_size_svg(fig, f"{stem}.svg")
    plt.close(fig)
    print(f"  wrote {stem}.{{png,svg}}")


# ----------------------------------------------------------------------------- data loading


def rank_bins(values: np.ndarray, counts: tuple[int, int, int]) -> np.ndarray:
    """Sort and cut so exactly `counts` land in low / medium / high.

    `counts` HERE ARE THE TRUE TEST COUNTS (108/431/100), i.e. ORACLE INFORMATION, and that
    has to be said plainly because nothing in the figures announces it. Both models are handed
    the marginal distribution of the answer, which no deployed model would have. The sibling
    artifact `evaluate_merzbacher_head_to_head.py` deliberately uses the TRAIN-POOL marginal
    instead (107/504/28) and is leak-free; these two scripts answer different questions and
    should not be quoted as if they were the same number.

    WHAT THIS DOES AND DOES NOT LICENSE. The same marginal is imposed on BOTH sides, so the
    RELATIVE claim -- "CGT is level with their best RF" -- is sound. The ABSOLUTE values are
    not deployable accuracy. Measured both ways (2026.08.01):

        marginal                RF acc   CGT acc   RF hi-rec   CGT hi-rec
        test counts (oracle)    0.5556   0.5493      0.290        0.290
        train-pool (leak-free)  0.6228   0.6166      0.190        0.190

    RF leads by ~0.006 either way and high-producer recall is identical either way, so the
    comparison is robust to the choice; only the level moves. Note the pool expects just 28
    high producers where the test set holds 100 -- their test genes are enriched for extremes
    relative to the full screen, which is what suppresses high recall under the pool marginal.

    Ties are broken by position, which is arbitrary but affects only exact ties. Verified
    inert: over 200 random permutations before ranking, their accuracy moves 0.5563 +- 0.0008
    and high-producer recall not at all -- ties fall inside a class, never across a boundary.
    """
    order = np.argsort(np.argsort(values))
    n_lo, n_med, _ = counts
    return np.where(order < n_lo, 0, np.where(order < n_lo + n_med, 1, 2))


def focus_cell(ours: dict[str, Any]) -> str:
    """The val-selected cell, or the only one present. Raises rather than guessing."""
    if FOCUS_SETTING in ours:
        return FOCUS_SETTING
    if len(ours) == 1:
        return next(iter(ours))
    raise SystemExit(
        f"FOCUS_SETTING {FOCUS_SETTING!r} has no dump. Available: {sorted(ours)}.\n"
        "Set FOCUS_SETTING to the cell with the best VALIDATION pearson -- never the best "
        "test score, which would be selection on the comparison set."
    )


def scale_bins(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Their absolute 3-class rule on an externally supplied (train-pool) scale."""
    t1, t2 = MERZBACHER_THRESHOLDS
    scaled = (values - lo) / (hi - lo)
    return np.where(scaled < t1, 0, np.where(scaled < t2, 1, 2))


def load_our_dumps() -> list[dict[str, Any]]:
    """Per-gene betaxanthin predictions for every grid cell that has finished."""
    out = []
    for f in sorted(glob.glob(osp.join(DATA_ROOT, "test-predictions", "*.json"))):
        with open(f) as fh:
            payload = json.load(fh)
        if "betaxanthin" not in payload.get("predictions", {}):
            continue
        tags = [t for t in payload.get("wandb_tags", []) if _SETTING_TAG.match(t)]
        payload["_setting"] = tags[0] if tags else osp.basename(f)[:12]
        payload["_genes"] = {
            r["genes"][0]: float(r["pred"][0])
            for r in payload["predictions"]["betaxanthin"]
        }
        out.append(payload)
    return out


def load_their_gene_scores() -> dict[str, dict[str, Any]]:
    """{model: {gene: E[class]}} from the shipped per-flux-sample class probabilities.

    E[class] = 0*p_low + 1*p_med + 2*p_high, averaged over a gene's flux samples. This is the
    continuous score their ternary plots are drawn from, and it is what makes THEIR side
    rank-matchable on the same rule as ours -- without it, only our side could be re-binned
    and the comparison would be rigged in our favour.

    THEY DO RELEASE MORE THAN CLASSES, which is the thing to check before believing any of
    this: `fig4c_*.csv` carries `score0/score1/score2` -- P(low)/P(medium)/P(high), summing to
    exactly 1.0 -- for each of 124 flux samples per gene, over 640 genes. Aggregated, that is
    558 distinct gene-level values, i.e. a real ordering rather than a re-shuffle of three
    labels.

    TWO ROBUSTNESS CHECKS, both run before this was used for anything (2026.08.01):

    * TIES ARE INERT. 82 genes share a score with another, and `rank_bins` breaks ties by
      array position, which is arbitrary. Over 200 random permutations before ranking, their
      accuracy moves 0.5563 +- 0.0008 and high-producer recall not at all -- the ties fall
      INSIDE a class, never across a boundary.
    * THE AGGREGATION IS NOT LOAD-BEARING, and the one used is the most generous to them:
      E[class] gives acc 0.5556 / high-recall 0.290; `p_high - p_low` is identical; `p_high`
      alone gives 0.5446 / 0.280. Collapsing the 3-simplex to one number does lose
      information -- (0.5, 0, 0.5) and (0, 1, 0) both map to 1.0 -- but not enough to move
      the comparison.
    """
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(glob.glob(osp.join(FIG4_DIR, "fig4c_*.csv"))):
        model = osp.basename(path)[len("fig4c_") : -len(".csv")]
        df = pd.read_csv(path)
        df["expected_class"] = df["score1"] + 2.0 * df["score2"]
        g = df.groupby("knockout_name")
        out[model] = {
            "score": g["expected_class"].mean().to_dict(),
            "true": g["true_label"].first().to_dict(),
            # Their DEPLOYED hard call, by majority vote over the gene's flux samples --
            # `knockout_voting` in their code. Kept separately from the score so the
            # absolute-vs-rank distinction is visible on their side too.
            "hard": g["prediction"].agg(lambda s: s.value_counts().idxmax()).to_dict(),
        }
    return out


def main() -> None:
    style()
    ts = timestamp()
    with open(SPLIT_PATH) as fh:
        split = json.load(fh)
    with open(BASELINE_PATH) as fh:
        baseline = json.load(fh)
    _, their_val = load_merzbacher_split()
    raw, _ = read_cachera_raw()

    truth = {str(r["knockout"]): int(r["label"]) for _, r in their_val.iterrows()}
    dumps = load_our_dumps()
    if not dumps:
        raise SystemExit(f"no dumps under {osp.join(DATA_ROOT, 'test-predictions')}")
    theirs = load_their_gene_scores()

    pool = [raw[g] for g in split["split"]["train_val_pool"] if g in raw]
    lo, hi = float(np.nanmin(pool)), float(np.nanmax(pool))

    # The gene set every panel is computed on: genes with THEIR label, OUR prediction (in
    # every dump), and a raw screen value. Fixed once so all panels and all models are scored
    # on identical genes -- otherwise "their accuracy" and "our accuracy" would be means over
    # different populations and the bars would not be comparable.
    genes = set(truth) & set(raw)
    for d in dumps:
        genes &= set(d["_genes"])
    genes = sorted(genes)
    t = np.array([truth[g] for g in genes], dtype=int)
    counts = tuple(int(x) for x in np.bincount(t, minlength=3))
    print(f"scoring on {len(genes)} genes; truth low/med/high = {counts}")

    ours = {}
    for d in dumps:
        pr = np.array([d["_genes"][g] for g in genes], dtype=float)
        ours[d["_setting"]] = {
            "abs": scale_bins(pr, lo, hi),
            "rank": rank_bins(pr, counts),  # type: ignore[arg-type]
            "raw": pr,
        }

    best = focus_cell(ours)
    rf_score = np.array([theirs[RF_MODEL]["score"].get(g, np.nan) for g in genes])
    if np.isnan(rf_score).any():
        raise SystemExit(f"{RF_MODEL} has no score for some genes -- name mismatch?")
    rf = {
        # Their DEPLOYED call: majority vote over each gene's flux samples, exactly as they
        # report it. This is the number in their paper.
        "published": np.array([theirs[RF_MODEL]["hard"][g] for g in genes], dtype=int),
        "rank": rank_bins(rf_score, counts),  # type: ignore[arg-type]
    }
    cgt = ours[best]

    # PROVENANCE CHECK, not decoration: our re-derivation of their DEPLOYED gene-level call
    # must reproduce the accuracy they publish. If it does not, we are comparing against
    # something other than their published model.
    published = baseline["gene_level"][RF_MODEL]["gene_level_accuracy"]
    recomputed = acc(rf["published"], t)
    if abs(published - recomputed) > 0.005:
        raise SystemExit(
            f"re-derived {RF_MODEL} accuracy {recomputed:.4f} != published {published:.4f}"
        )
    print(f"  their published RF acc {published:.4f}; re-derived {recomputed:.4f} OK")
    STATS["provenance"] = {
        "note": "[[experiments.020-cachera-betaxanthin.merzbacher-comparison]]",
        "timestamp": ts,
        "n_cgt_cells": len(dumps),
        "cgt_cells": sorted(ours),
        "cgt_val_selected_cell": best,
        "fcl_model": RF_MODEL,
        "n_genes_scored": len(genes),
        "truth_counts_low_med_high": list(counts),
        "their_published_rf_accuracy": published,
        "our_rederivation_of_it": recomputed,
        "bin_scale": {"source": "train_val_pool min-max", "min": lo, "max": hi},
    }

    obs = np.array([raw[g] for g in genes], dtype=float)
    # TWO FIGURES BY DEFAULT. Fig 1 is the scatter because it is the only one that shows why
    # every other comparison here is degenerate: both models emit a nearly constant value, so
    # a classification metric on a 67 %-majority problem has almost nothing to measure. Fig 2
    # is the accuracy collapse, which is the EVIDENCE for that claim rather than a result in
    # its own right.
    rf_sc = np.array([theirs[RF_MODEL]["score"][g] for g in genes], dtype=float)
    fig_scatter(cgt, rf, t, obs, theirs, genes, best, ts)
    fig_accuracy(cgt, rf, t, best, ts)
    fig_topk(ours, rf_sc, t, best, ts)
    fig_score_by_class(cgt, rf_sc, t, best, ts)
    fig_roc(ours, rf_sc, t, best, ts)
    fig_cell_spread(ours, rf_sc, obs, t, best, ts)
    fig_label_provenance(t, obs, ts)
    fig_screen_coverage(raw, ts)
    fig_nonmetabolic(dumps, raw, split, best, ts)
    # The class-distribution, per-class-recall and confusion panels are diagnostics that
    # informed the reading above and are cited in the note; they are not carried in the figure
    # set. `--all` regenerates them.
    if "--all" in sys.argv:
        fig_distribution(cgt, rf, t, best, ts)
        fig_recall(cgt, rf, t, best, ts)
        fig_confusion(cgt, rf, t, best, ts)

    # EVERY QUOTED NUMBER, ON DISK. Written LAST so it holds what the figures actually drew.
    # Unstamped on purpose: it is the current state of the comparison, and a stamped copy per
    # render would leave the note pointing at whichever one someone happened to open. The
    # render timestamp is inside, and `git diff` on this file is how a re-render's movement
    # becomes visible instead of silent.
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stats_path = osp.join(RESULTS_DIR, "merzbacher_comparison_figures.json")
    with open(stats_path, "w") as fh:
        json.dump(STATS, fh, indent=2, sort_keys=True)
    print(f"  wrote {stats_path}")


# ---------------------------------------------------------------------------------- panels


def acc(pred: np.ndarray, true: np.ndarray) -> float:
    return float((pred == true).mean())


def recall(pred: np.ndarray, true: np.ndarray, c: int) -> float:
    return float((pred[true == c] == c).mean()) if (true == c).any() else float("nan")


def draw_accuracy(ax: plt.Axes, cgt: dict, rf: dict, t: np.ndarray, best: str) -> None:
    """As-published vs rank-matched accuracy, both models.

    The gap between filled and open is the whole message: their RF's headline 0.700 clears
    the 0.674 majority rate only because it calls 95 % of genes medium, and stripping that
    away drops it to 0.556 -- next to CGT's 0.549.
    """
    for i, (name, c, dep, rk, dep_lab) in enumerate(
        [
            (RF_LABEL, RF_C, rf["published"], rf["rank"], "as published"),
            (CGT_LABEL, CGT_C, cgt["abs"], cgt["rank"], "absolute bins"),
        ]
    ):
        a_dep, a_rk = acc(dep, t), acc(rk, t)
        ax.plot([i - 0.13, i + 0.13], [a_dep, a_rk], lw=0.7, color=c, zorder=3)
        ax.scatter(
            i - 0.13,
            a_dep,
            s=30,
            color=c,
            zorder=4,
            edgecolors="black",
            linewidths=0.4,
            label=dep_lab if i == 0 else None,
        )
        ax.scatter(
            i + 0.13,
            a_rk,
            s=30,
            color="white",
            zorder=4,
            edgecolors=c,
            linewidths=1.0,
            label="rank-matched" if i == 0 else None,
        )
        ax.text(
            i - 0.13,
            a_dep + 0.013,
            f"{a_dep:.3f}",
            ha="center",
            va="bottom",
            fontsize=5,
        )
        ax.text(
            i + 0.13, a_rk - 0.015, f"{a_rk:.3f}", ha="center", va="top", fontsize=5
        )
    maj = float(np.bincount(t, minlength=3).max() / len(t))
    ax.axhline(maj, ls="--", lw=0.6, color=TRUTH_C, zorder=2)
    # Anchored bottom-RIGHT: at top-left it sat on top of the RF "as published" marker,
    # which lands only 0.027 above the majority line -- the very proximity the panel is about.
    ax.text(
        1.45,
        maj - 0.010,
        f"majority class {maj:.3f}",
        ha="right",
        va="top",
        fontsize=5,
        color=TRUTH_C,
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels([RF_LABEL, CGT_LABEL], fontsize=6)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("gene-level accuracy")
    ax.set_ylim(0.40, 0.80)
    tenth_grid(ax)
    ax.legend(frameon=False, loc="lower left", fontsize=5, handletextpad=0.4)


def draw_distribution(ax: plt.Axes, cgt: dict, rf: dict, t: np.ndarray) -> None:
    """Where each model actually puts the genes -- the shared failure mode."""
    rows = [
        ("truth", t),
        (f"{RF_LABEL}\nas published", rf["published"]),
        (f"{CGT_LABEL}\nabsolute bins", cgt["abs"]),
        ("both\nrank-matched", cgt["rank"]),
    ]
    y = np.arange(len(rows))
    left = np.zeros(len(rows))
    for c in range(3):
        w = np.array([float((p == c).mean()) for _, p in rows])
        ax.barh(
            y,
            w,
            left=left,
            height=0.6,
            color=CLASS_COLORS[c],
            edgecolor="black",
            linewidth=0.4,
            label=CLASS_NAMES[c],
            zorder=3,
        )
        for yi, (wi, li) in enumerate(zip(w, left)):
            if wi > 0.05:
                ax.text(
                    li + wi / 2, yi, f"{wi:.2f}", ha="center", va="center", fontsize=5
                )
        left += w
    ax.set_yticks(y)
    ax.set_yticklabels([n for n, _ in rows], fontsize=5)
    ax.invert_yaxis()
    ax.set_xlabel("fraction of genes assigned")
    ax.set_xlim(0, 1)
    ax.legend(
        frameon=False,
        ncol=3,
        fontsize=5,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        handletextpad=0.4,
        columnspacing=1.0,
    )


def draw_recall(ax: plt.Axes, cgt: dict, rf: dict, t: np.ndarray) -> None:
    """Per-class recall, rank-matched -- where CGT actually wins."""
    counts = np.bincount(t, minlength=3)
    series = [
        (f"{RF_LABEL} as published", RF_C, rf["published"]),
        (f"{RF_LABEL} rank-matched", RF_FILL, rf["rank"]),
        (f"{CGT_LABEL} rank-matched", CGT_C, cgt["rank"]),
    ]
    w = 0.8 / len(series)
    x = np.arange(3)
    for i, (name, c, pred) in enumerate(series):
        vals = [recall(pred, t, k) for k in range(3)]
        ax.bar(
            x + i * w - 0.4 + w / 2,
            vals,
            width=w * 0.9,
            color=c,
            edgecolor="black",
            linewidth=0.4,
            label=name,
            zorder=3,
        )
        for xv, v in zip(x + i * w - 0.4 + w / 2, vals):
            ax.text(xv, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=4.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n(n={counts[i]})" for i, n in enumerate(CLASS_NAMES)])
    ax.set_ylabel("recall within true class")
    # 1.08, not 1.0: the tallest bar is 0.98 and its value label sits 0.015 above it, so a
    # 1.0 ceiling clipped the number clean off the axes.
    ax.set_ylim(0, 1.08)
    tenth_grid(ax)
    # ABOVE the axes, not inside: the medium bars reach 0.98, so any in-axes placement
    # collides with them.
    ax.legend(
        frameon=False,
        fontsize=5,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.01),
        handletextpad=0.4,
        columnspacing=1.0,
    )


def draw_confusion(axes, cgt: dict, rf: dict, t: np.ndarray) -> None:
    """Row-normalized confusion, both sides rank-matched -- apples-to-apples."""
    for ax, (name, pred) in zip(
        axes, [(RF_LABEL, rf["rank"]), (CGT_LABEL, cgt["rank"])]
    ):
        cm = np.zeros((3, 3))
        for a in range(3):
            for b in range(3):
                cm[a, b] = float(((t == a) & (pred == b)).sum())
        cmn = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(cmn, cmap=CGT_CMAP, vmin=0, vmax=1)
        for a in range(3):
            for b in range(3):
                ax.text(
                    b,
                    a,
                    f"{cmn[a, b]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="black" if cmn[a, b] < 0.65 else "white",
                )
        ax.set_xticks(range(3))
        ax.set_xticklabels(CLASS_NAMES, fontsize=5)
        ax.set_yticks(range(3))
        ax.set_yticklabels(CLASS_NAMES, fontsize=5)
        ax.set_xlabel("predicted")
        ax.set_title(f"{name} rank-matched\nacc {acc(pred, t):.3f}", fontsize=5, pad=3)
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
    axes[0].set_ylabel("true")


def write_high_disagreement(
    genes: list,
    t: np.ndarray,
    rf_hi: np.ndarray,
    cgt_hi: np.ndarray,
    rf_score: np.ndarray,
    cgt_score: np.ndarray,
    ts: str,
) -> None:
    """The per-gene backing for panel 1c's rings -- names the figure cannot hold.

    `call` is which model(s) put the gene in their rank-matched top 100. The rows that
    matter for follow-up are `only_cgt` / `only_rf` with `released_label == 2`: genes the
    FCL paper calls high that exactly one of the two methods recovers.
    """
    call = np.where(
        rf_hi & cgt_hi,
        "both",
        np.where(rf_hi, "only_rf", np.where(cgt_hi, "only_cgt", "neither")),
    )
    keep = call != "neither"
    df = pd.DataFrame(
        {
            "gene": np.asarray(genes)[keep],
            "released_label": t[keep],
            "call": call[keep],
            "fcl_rf_e_class": rf_score[keep],
            "cgt_predicted_z": cgt_score[keep],
        }
    ).sort_values(["call", "released_label", "gene"])
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = osp.join(RESULTS_DIR, f"merzbacher_high_call_disagreement_{ts}.csv")
    df.to_csv(path, index=False)
    n_tru = int(((call != "both") & keep & (t == 2)).sum())
    print(
        f"  high-call sets: both {int((call == 'both').sum())}, "
        f"only_rf {int((call == 'only_rf').sum())}, only_cgt {int((call == 'only_cgt').sum())}; "
        f"{n_tru} true-high genes recovered by exactly one model"
    )
    print(f"  wrote {path}")


def fig_scatter(
    cgt: dict,
    rf: dict,
    t: np.ndarray,
    obs: np.ndarray,
    theirs: dict,
    genes: list,
    best: str,
    ts: str,
) -> None:
    """STANDALONE: every one of the 639 test genes, model score against measured production.

    THE SPREAD IS THE POINT, and it is what every aggregate number in the other figures hides.
    Both models compress: a predictor that emits nearly the same value for every gene can
    still post a respectable accuracy by riding the majority class, and only a scatter shows
    that this is what is happening.

    Colored by TRUE class, so vertical structure = the model separating the classes and a
    single horizontal smear = it not doing so. Spearman is annotated because with this much
    overplotting the eye is a poor judge of monotone association.

    Their y-axis is E[class] (0-2) and ours is a standardized regression output; the axes are
    NOT comparable in units and are not meant to be. What compares is the SHAPE.
    """
    rf_score = np.array([theirs[RF_MODEL]["score"][g] for g in genes], dtype=float)
    # THREE PAIRINGS, so the third is not left out: each model against the MEASUREMENT, and
    # then the two models against EACH OTHER. The third panel is the one that says whether
    # matching aggregate scores mean shared information -- two models can tie on every summary
    # statistic and still be right about different genes, which panels 1 and 2 cannot show.
    panels = [
        (
            f"{RF_LABEL} vs measured",
            obs,
            rf_score,
            "measured betaxanthin, Cachera screen (z)",
            FCL_AXIS_LABEL,
        ),
        (
            f"{CGT_LABEL} vs measured",
            obs,
            cgt["raw"],
            "measured betaxanthin, Cachera screen (z)",
            "CGT   predicted (z)",
        ),
        (
            f"{CGT_LABEL} vs {RF_LABEL}",
            rf_score,
            cgt["raw"],
            FCL_AXIS_LABEL,
            "CGT   predicted (z)",
        ),
    ]
    # wspace is explicit: at the default the y-label of panels 2 and 3 crowds the previous
    # panel's tick labels, which reads as one plot bleeding into the next.
    # ONE SCALE PER VARIABLE, not per panel. `measured` is the x of panels 1 and 2, FCL's
    # E[class] is the y of panel 1 AND the x of panel 3, and CGT's prediction is the y of
    # panels 2 and 3. Letting matplotlib autoscale each axis independently drew the same
    # quantity at three different magnifications, so panel 3 could not be visually compared
    # with panels 1-2. Limits and tick spacing are now keyed to the variable.
    #
    # The E[class] range carries deliberate headroom at the top: it is panel 1's y, and the
    # legend sits in that empty strip. Data ranges are measured -2.28..2.34, E[class]
    # 0.10..1.99, CGT -0.59..1.60.
    # CGT's prediction shares the MEASURED scale, because it is the same quantity in the same
    # units -- z-scored betaxanthin, predicted vs observed. Giving the prediction its own
    # tighter axis magnified it to fill the panel and hid the very thing the figure is about:
    # on the measurement's own scale the predictions collapse into a narrow horizontal band.
    # FCL's E[class] keeps its own limits; it is a 0-2 class expectation, not a z-score, and
    # forcing it onto the same axis would assert a comparability that does not exist.
    # TICK STYLE FOLLOWS WHAT THE NUMBER IS, which the first version had backwards.
    # E[class] is anchored on the CLASS VALUES 0 / 1 / 2 (low / medium / high) -- a tick at
    # 0.5 names nothing -- so it ticks at integers and is formatted without decimals. The
    # z-scores are genuinely continuous, so they carry a decimal to say so, with minor ticks
    # at 0.5 for resolution the labels do not have to spend space on.
    #                   (limits)        major  minor  format
    axis_spec = {
        "measured": ((-2.6, 2.6), 1.0, 0.5, "%.1f"),
        "fcl": ((0.0, 2.45), 1.0, 0.5, "%.0f"),
        "cgt": ((-2.6, 2.6), 1.0, 0.5, "%.1f"),
    }
    var_of = {
        "measured betaxanthin, Cachera screen (z)": "measured",
        FCL_AXIS_LABEL: "fcl",
        "CGT   predicted (z)": "cgt",
    }
    fig, axes = plt.subplots(
        1, 3, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(58))
    )
    for ax, (name, x, y, xlab, ylab) in zip(axes, panels):
        for c in range(3):
            m = t == c
            ax.scatter(
                x[m],
                y[m],
                s=4,
                color=CLASS_COLORS[c],
                linewidths=0.0,
                label=f"{CLASS_NAMES[c]} (n={int(m.sum())})",
                zorder=3,
            )
        # NO y = x LINE. It was drawn on panel 2 only -- the one panel where both axes are the
        # same quantity in the same units (z-scored betaxanthin), so it was the honest
        # perfect-prediction reference and could not be drawn on panels 1 or 3, whose axes are
        # not commensurable. Removed anyway (author call, 2026.08.02): a reference line on one
        # panel of three reads as an inconsistency rather than as a statement about units, and
        # the compression it pointed at is already carried by the shaded band and the title.
        # PANEL 3 ONLY: OF THE 100 TRUE HIGH PRODUCERS, WHICH DOES EACH METHOD RECOVER?
        # Both sides are rank-matched into the same 100 high slots, so "called high" is a
        # like-for-like set on each axis and the two cut lines partition the panel into
        # agreement / disagreement quadrants.
        #
        # THE COUNTS ARE OVER THE BLUE (TRUE HIGH) SET, not over all 639 genes -- that is the
        # question the panel exists to answer, and the all-gene count is only the quadrant's
        # denominator. Reporting the all-gene numbers as the headline (an earlier version did)
        # answered a question nobody asked: 85 CGT-only genes is uninformative when most of
        # them are not high producers in the first place.
        #
        # The pattern is COMPLEMENTARITY: each method recovers 29 of 100, they share 9, and
        # their union is 49 -- close to additive. Two methods that tie on recall (0.290 each)
        # are finding largely DIFFERENT high producers, which is an argument for running both
        # rather than for picking one. Names go to the CSV written alongside this figure.
        if var_of[xlab] == "fcl" and var_of[ylab] == "cgt":
            rf_hi, cgt_hi = rf["rank"] == 2, cgt["rank"] == 2
            x_cut, y_cut = float(x[rf_hi].min()), float(y[cgt_hi].min())
            for setline, cut in ((ax.axvline, x_cut), (ax.axhline, y_cut)):
                setline(cut, ls=(0, (4, 2)), lw=0.5, color="0.35", zorder=2)
            tru_hi = t == 2
            recovered = (rf_hi | cgt_hi) & tru_hi
            # Ring every true high producer at least one method recovers. Blue already marks
            # the true-high set, but panel 3's cloud is dense enough that the ring is what
            # makes the recovered ones findable inside it.
            ax.scatter(
                x[recovered],
                y[recovered],
                s=16,
                facecolors="none",
                edgecolors="black",
                linewidths=0.45,
                zorder=4,
            )
            xl, yl = axis_spec["fcl"][0], axis_spec["cgt"][0]
            # Counts sit in the corner of the quadrant they describe: true-high first
            # (the answer), all-genes in parentheses (the denominator).
            for qx, qy, ha, va, mask, txt in (
                (xl[1], yl[1], "right", "top", rf_hi & cgt_hi, "both"),
                (xl[0], yl[1], "left", "top", cgt_hi & ~rf_hi, "CGT only"),
                (xl[1], yl[0], "right", "bottom", rf_hi & ~cgt_hi, "FCL RF only"),
                (xl[0], yl[0], "left", "bottom", ~rf_hi & ~cgt_hi, "neither"),
            ):
                ax.text(
                    qx - 0.04 * (1 if ha == "right" else -1) * (xl[1] - xl[0]),
                    qy - 0.04 * (1 if va == "top" else -1) * (yl[1] - yl[0]),
                    f"{txt}\n{int((mask & tru_hi).sum())} high\n({int(mask.sum())} genes)",
                    ha=ha,
                    va=va,
                    fontsize=4.5,
                    color="0.25",
                    linespacing=1.2,
                    zorder=5,
                )
            # NO in-axes legend for the rings. All four corners now carry a quadrant count,
            # so there is nowhere inside the panel a legend can sit without landing on one;
            # the ring is explained in the figure-level caption instead.
            write_high_disagreement(genes, t, rf_hi, cgt_hi, x, y, ts)
        rho = float(spearmanr(y, x).statistic)
        # THE BAND IS THE MIDDLE 80 % OF THE Y-AXIS MODEL'S OUTPUT (10th-90th percentile),
        # and it is labelled in-axes because an unlabelled grey box explains nothing. It is
        # here because the axis LIMITS hide the compression: panel 1 spans 0.1-2.0 while 80 %
        # of its points sit inside 0.22 of each other, so the cloud looks better spread than
        # it is. The band is the visual form of the number in the title.
        p10, p90 = np.percentile(y, [10, 90])
        ax.axhspan(p10, p90, color=PLOT_PALETTE_FILL[5], zorder=1)
        # The span goes in the TITLE and the band is explained ONCE in a figure caption.
        # An in-axes annotation was tried and landed on top of the point cloud, where it was
        # unreadable and hid the data it was describing.
        # "spans" was ambiguous on its own -- it is the WIDTH of the shaded band, i.e. how
        # far apart the 10th and 90th percentiles of the y-axis model's own output are, in
        # that axis's units. Spelled out here so the number is readable without the caption.
        # PEARSON IS RECORDED THOUGH IT IS NOT DRAWN. The note's Fig 1 table carries a
        # Pearson column next to the Spearman one -- the two disagreeing is the evidence for
        # "each model gets a few extreme genes right and is otherwise flat" -- so the number
        # has to come from here rather than from a side calculation.
        STATS.setdefault("fig1_scatter", {})[name] = {
            "pearson": float(pearsonr(y, x).statistic),
            "spearman": rho,
            "y_p10_p90_width": float(p90 - p10),
            "n": int(len(x)),
        }
        ax.set_title(
            f"{name}\nSpearman {rho:+.3f}    80% of y within {p90 - p10:.2f} (10-90 pct)",
            fontsize=5.5,
            pad=3,
        )
        ax.set_xlabel(xlab, fontsize=5.5)
        ax.set_ylabel(ylab, fontsize=5.5)
        for setlim, axis, lab in (
            (ax.set_xlim, ax.xaxis, xlab),
            (ax.set_ylim, ax.yaxis, ylab),
        ):
            lims, major, minor, fmt = axis_spec[var_of[lab]]
            setlim(*lims)
            axis.set_major_locator(mpl.ticker.MultipleLocator(major))
            axis.set_minor_locator(mpl.ticker.MultipleLocator(minor))
            axis.set_major_formatter(mpl.ticker.FormatStrFormatter(fmt))
        ax.tick_params(which="minor", length=0)
        ax.tick_params(labelsize=5)
        ax.grid(lw=0.3, color="0.92", zorder=0)
        ax.set_axisbelow(True)
    # FIGURE-level legend BELOW the panels, not inside panel 1. In-axes it sat in the
    # upper-left of the first panel, directly on top of two red points -- and there is no
    # in-axes corner that is reliably empty across all three panels, since each has a
    # different x variable. Outside the axes it cannot collide with data at all.
    # BACK INSIDE panel 1, sitting in the empty strip the shared E[class] upper limit
    # (2.45 against a data max of 1.99) deliberately leaves. A figure-level legend below the
    # panels read as detached from the plot; an in-axes one at the default limits covered two
    # red points. Making room for it is the fix, not moving it away.
    axes[0].legend(
        frameon=False,
        fontsize=4.5,
        loc="upper left",
        ncol=3,
        handletextpad=0.2,
        columnspacing=0.8,
        markerscale=1.6,
        title="released class label (FCL paper)",
        title_fontsize=4.5,
    )
    # tight_layout FIRST (it computes label extents), then widen the gutters and reserve the
    # caption strip -- doing it in the other order lets tight_layout overwrite both, which is
    # how the x-labels got clipped off the bottom.
    fig.tight_layout(pad=0.4, rect=(0.0, 0.10, 1.0, 1.0))
    fig.subplots_adjust(wspace=0.40)
    fig.text(
        0.5,
        0.020,
        "shaded band = middle 80% (10th-90th percentile) of the y-axis model's own output; a "
        "narrow band means the model returns nearly the same value for every gene.\n"
        "Panel 3 quadrants are cut at each model's rank-matched high threshold; counts are "
        "TRUE HIGH producers (blue) with the quadrant's all-gene total in parentheses. Rings "
        "= the 49 of 100 true highs found by at least one method (FCL 29, CGT 29, shared 9).\n"
        "Color is the FCL paper's RELEASED class label; x is our copy of the Cachera screen. "
        "They agree at Spearman 0.73 (88% of attainable) and 18.8% of genes fall in a "
        "different bin, which is why the colors overlap along x -- see Fig 7.",
        ha="center",
        fontsize=4.5,
        color="0.35",
    )
    save(fig, "merzbacher_fig1_scatter_spread", ts)


def fig_accuracy(cgt: dict, rf: dict, t: np.ndarray, best: str, ts: str) -> None:
    """STANDALONE: as-published vs rank-matched accuracy."""
    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_WIDTHS_MM["half"]), mm_to_in(60)))
    draw_accuracy(ax, cgt, rf, t, best)
    fig.tight_layout(pad=0.4)
    save(fig, "merzbacher_fig2_accuracy_artifact", ts)


def fig_distribution(cgt: dict, rf: dict, t: np.ndarray, best: str, ts: str) -> None:
    """STANDALONE: where each model puts the genes."""
    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_WIDTHS_MM["half"]), mm_to_in(52)))
    draw_distribution(ax, cgt, rf, t)
    fig.tight_layout(pad=0.4)
    save(fig, "merzbacher_diag_class_distribution", ts)


def fig_recall(cgt: dict, rf: dict, t: np.ndarray, best: str, ts: str) -> None:
    """STANDALONE: per-class recall."""
    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_WIDTHS_MM["half"]), mm_to_in(58)))
    draw_recall(ax, cgt, rf, t)
    fig.tight_layout(pad=0.4)
    save(fig, "merzbacher_diag_per_class_recall", ts)


def fig_confusion(cgt: dict, rf: dict, t: np.ndarray, best: str, ts: str) -> None:
    """STANDALONE: the two rank-matched confusion matrices.

    These stay in ONE figure because they are the same plot twice, side by side, which is how
    a confusion comparison is read -- not two different plot types stacked under a/b labels.
    """
    fig, axes = plt.subplots(
        1, 2, figsize=(mm_to_in(PANEL_WIDTHS_MM["half"]), mm_to_in(52))
    )
    draw_confusion(axes, cgt, rf, t)
    fig.tight_layout(pad=0.4)
    save(fig, "merzbacher_diag_confusion_rank_matched", ts)


def z_axis(axis, lims: tuple[float, float] | None = None) -> None:
    """Format an axis carrying a z-score: one decimal, minor ticks at 0.5.

    Centralized because "which axes are z-scores" is a property of the DATA, and having each
    figure decide for itself is how Fig 7 and Fig 9 ended up with integer ticks while Fig 1
    had decimals -- the same quantity styled two ways in one document.
    """
    axis.set_major_locator(mpl.ticker.MultipleLocator(1.0))
    axis.set_minor_locator(mpl.ticker.MultipleLocator(0.5))
    axis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.1f"))
    if lims is not None:
        axis.axes.set_xlim(*lims) if axis is axis.axes.xaxis else axis.axes.set_ylim(
            *lims
        )


def _pct(x: np.ndarray) -> np.ndarray:
    """Percentile rank 0-100. Puts two models with incomparable units on ONE axis."""
    return rankdata(x) / len(x) * 100.0


def _precision_at_k(score: np.ndarray, is_high: np.ndarray, kmax: int) -> np.ndarray:
    """Fraction of the top-k highest-scoring genes that are TRUE high producers."""
    order = np.argsort(-score)
    hits = np.cumsum(is_high[order][:kmax])
    return hits / np.arange(1, kmax + 1)


def fig_topk(ours: dict, rf_sc: np.ndarray, t: np.ndarray, best: str, ts: str) -> None:
    """Precision@k for high producers -- the decision-relevant, binning-free comparison.

    Every other comparison here has to choose a binning; this one does not. It answers the
    question a strain-design campaign actually asks: "if I take the k genes the model likes
    most, what fraction are genuinely high producers?" The random baseline is the class rate.

    ALL TEN CGT CELLS ARE DRAWN, faintly, behind the val-selected one. A single curve would
    hide that the grid spans a wide band and would invite reading the selected cell as "the"
    CGT result rather than one draw from it.
    """
    is_high = (t == 2).astype(float)
    kmax = 200
    ks = np.arange(1, kmax + 1)
    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_WIDTHS_MM["half"]), mm_to_in(58)))
    for name, d in sorted(ours.items()):
        if name == best:
            continue
        ax.plot(
            ks,
            _precision_at_k(d["raw"], is_high, kmax),
            lw=0.5,
            color=CGT_C,
            alpha=0.30,
            zorder=2,
        )
    ax.plot(
        [], [], lw=0.5, color=CGT_C, alpha=0.30, label="CGT, other grid cells (n=9)"
    )
    ax.plot(
        ks,
        _precision_at_k(rf_sc, is_high, kmax),
        lw=1.4,
        color=RF_C,
        zorder=4,
        label=f"{RF_LABEL}",
    )
    ax.plot(
        ks,
        _precision_at_k(ours[best]["raw"], is_high, kmax),
        lw=1.4,
        color=CGT_C,
        zorder=5,
        label=f"{CGT_LABEL} (val-selected)",
    )
    base = float(is_high.mean())
    ax.axhline(base, ls="--", lw=0.7, color=TRUTH_C, zorder=3)
    ax.text(
        kmax,
        base + 0.012,
        f"random {base:.3f}",
        ha="right",
        va="bottom",
        fontsize=5,
        color=TRUTH_C,
    )
    ax.set_xlabel("k = number of genes nominated")
    ax.set_ylabel("precision@k  (fraction truly high)")
    ax.set_xlim(1, kmax)
    # 1.0, not 0.85: precision@k hits 1.000 at k=1 for both models, so the old ceiling
    # sliced the top off the very region the figure is about.
    ax.set_ylim(0, 1.0)
    tenth_grid(ax)
    ax.legend(frameon=False, fontsize=5, loc="upper right", handletextpad=0.5)
    # THE p-VALUES THE NOTE QUOTES ARE COMPUTED HERE, not drawn. The curve shows enrichment;
    # whether a given k clears chance is a hypergeometric question (N genes, K true highs,
    # k drawn), one-sided because only ENRICHMENT is a claim. Recorded for every k the note
    # tabulates, plus the spread across the unselected cells at k=50.
    n_all, k_hi = len(is_high), int(is_high.sum())

    def topk_row(score: np.ndarray, k: int) -> dict[str, float]:
        hits = int(is_high[np.argsort(-score)][:k].sum())
        return {
            "hits": hits,
            "precision": hits / k,
            "enrichment": (hits / k) / (k_hi / n_all),
            "p_hypergeom": float(hypergeom.sf(hits - 1, n_all, k_hi, k)),
        }

    STATS["fig3_precision_at_k"] = {
        "n_genes": n_all,
        "n_high": k_hi,
        "base_rate": k_hi / n_all,
        "models": {
            lab: {f"k={k}": topk_row(sc, k) for k in (10, 25, 50)}
            for lab, sc in ((RF_LABEL, rf_sc), (CGT_LABEL, ours[best]["raw"]))
        },
        "other_cells_precision_at_50": {
            n: float(is_high[np.argsort(-d["raw"])][:50].mean())
            for n, d in sorted(ours.items())
            if n != best
        },
    }
    fig.tight_layout(pad=0.4)
    save(fig, "merzbacher_fig3_precision_at_k", ts)


def fig_score_by_class(
    cgt: dict, rf_sc: np.ndarray, t: np.ndarray, best: str, ts: str
) -> None:
    """Each model's own score, as a PERCENTILE RANK, split by Merzbacher's label.

    Percentile rank is what makes the two models comparable at all: their E[class] lives on
    0-2 and CGT's output is a z-score, so raw values share no axis. Under a perfect model the
    three boxes step cleanly from low to high; under a useless one all three sit on 50.

    This is the separation question with NO binning decision anywhere in it -- no thresholds,
    no marginal, no tie-breaking.
    """
    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_WIDTHS_MM["half"]), mm_to_in(56)))
    series = [(RF_LABEL, RF_C, _pct(rf_sc)), (CGT_LABEL, CGT_C, _pct(cgt["raw"]))]
    w = 0.34
    for i, (name, color, pr) in enumerate(series):
        data = [pr[t == c] for c in range(3)]
        pos = np.arange(3) + (i - 0.5) * w
        bp = ax.boxplot(
            data,
            positions=pos,
            widths=w * 0.82,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "lw": 0.8},
            whiskerprops={"lw": 0.5},
            capprops={"lw": 0.5},
            boxprops={"lw": 0.4, "edgecolor": "black"},
        )
        for b in bp["boxes"]:
            b.set_facecolor(color)
        ax.plot([], [], color=color, lw=4, label=name)
        for c, pos_c in zip(range(3), pos):
            ax.text(
                pos_c,
                103,
                f"{np.median(pr[t == c]):.0f}",
                ha="center",
                va="bottom",
                fontsize=4.5,
                color=color,
            )
    ax.axhline(50, ls="--", lw=0.6, color=TRUTH_C, zorder=1)
    ax.text(
        2.45, 51, "no information", ha="right", va="bottom", fontsize=5, color=TRUTH_C
    )
    ax.set_xticks(range(3))
    ax.set_xticklabels(
        [f"{n}\n(n={int((t == c).sum())})" for c, n in enumerate(CLASS_NAMES)]
    )
    ax.set_xlabel("released class label (FCL paper)")
    ax.set_ylabel("model's own score, percentile rank")
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", lw=0.3, color="0.9", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=5, loc="lower right", handlelength=1.2)
    STATS["fig4_score_by_class"] = {
        "median_percentile_rank": {
            name: {CLASS_NAMES[c]: float(np.median(pr[t == c])) for c in range(3)}
            for name, _, pr in series
        },
        "n_per_class": {CLASS_NAMES[c]: int((t == c).sum()) for c in range(3)},
    }
    fig.tight_layout(pad=0.4)
    save(fig, "merzbacher_fig4_score_by_class", ts)


def fig_roc(ours: dict, rf_sc: np.ndarray, t: np.ndarray, best: str, ts: str) -> None:
    """ROC for "is this gene a high producer" -- threshold-free, binning-free.

    Collapses the whole ranking into one number per model with no cut points to argue about,
    which is why it is the cleanest single-number comparison in this set. All ten CGT cells
    are drawn faintly so the selected cell is visibly one draw from a spread.
    """
    is_high = (t == 2).astype(int)
    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_WIDTHS_MM["half"]), mm_to_in(58)))
    for name, d in sorted(ours.items()):
        if name == best:
            continue
        fpr, tpr, _ = roc_curve(is_high, d["raw"])
        ax.plot(fpr, tpr, lw=0.5, color=CGT_C, alpha=0.30, zorder=2)
    ax.plot([], [], lw=0.5, color=CGT_C, alpha=0.30, label="CGT, other cells (n=9)")
    for score, color, lab in (
        (rf_sc, RF_C, RF_LABEL),
        (ours[best]["raw"], CGT_C, f"{CGT_LABEL} (val-selected)"),
    ):
        fpr, tpr, _ = roc_curve(is_high, score)
        ax.plot(
            fpr,
            tpr,
            lw=1.4,
            color=color,
            zorder=4,
            label=f"{lab}   AUC {roc_auc_score(is_high, score):.3f}",
        )
    ax.plot([0, 1], [0, 1], ls="--", lw=0.7, color=TRUTH_C, zorder=3)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate (high producers found)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    tenth_grid(ax)
    ax.legend(frameon=False, fontsize=5, loc="lower right", handletextpad=0.5)
    # HOW MANY CELLS BEAT FCL IS COMPUTED, NEVER COUNTED BY EYE. The note once said 3 in one
    # place and 4 in another, because the 4th clears FCL by 0.0013 (0.5709 vs 0.5696) and the
    # displayed values both round to 0.57. The margin is recorded next to the count so a
    # near-tie can never again be read off the rendered figure as a clean win.
    rf_auc = float(roc_auc_score(is_high, rf_sc))
    cell_auc = {
        n: float(roc_auc_score(is_high, d["raw"])) for n, d in sorted(ours.items())
    }
    above = {n: v for n, v in cell_auc.items() if v > rf_auc}
    STATS["fig5_roc"] = {
        "fcl_rf_auc": rf_auc,
        "cgt_val_selected": best,
        "cgt_val_selected_auc": cell_auc[best],
        "cell_auc": cell_auc,
        "n_cells_above_fcl": len(above),
        "cells_above_fcl": {
            n: {"auc": v, "margin": v - rf_auc} for n, v in above.items()
        },
    }
    fig.tight_layout(pad=0.4)
    save(fig, "merzbacher_fig5_roc_high_producers", ts)


def fig_cell_spread(
    ours: dict, rf_sc: np.ndarray, obs: np.ndarray, t: np.ndarray, best: str, ts: str
) -> None:
    """How much of "CGT beats their RF" is the CELL we picked?

    Ten grid cells, two binning-free metrics, their RF as a reference line. This is the
    honesty panel: with sigma ~ 0.03 between cells and a max taken over ten of them, a single
    quoted number is an order statistic. The val-selected cell is marked so it is visible
    whether we are quoting a typical cell or a lucky one -- and four cells here are the
    lr = 1e-3 runs that collapsed to a constant predictor, which is why the low tail exists.
    """
    is_high = (t == 2).astype(int)
    names = sorted(ours)
    metrics = [
        (
            "Spearman vs measured",
            [float(spearmanr(ours[n]["raw"], obs).statistic) for n in names],
            float(spearmanr(rf_sc, obs).statistic),
        ),
        (
            "ROC AUC, high producers",
            [float(roc_auc_score(is_high, ours[n]["raw"])) for n in names],
            float(roc_auc_score(is_high, rf_sc)),
        ),
    ]
    fig, axes = plt.subplots(
        1, 2, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(56))
    )
    for ax, (lab, vals, rf_val) in zip(axes, metrics):
        x = np.random.default_rng(0).normal(0, 0.045, len(vals))
        for xi, v, n in zip(x, vals, names):
            sel = n == best
            ax.scatter(
                xi,
                v,
                s=34 if sel else 16,
                color=CGT_C if sel else "white",
                edgecolors=CGT_C,
                linewidths=1.0 if sel else 0.6,
                zorder=5 if sel else 3,
            )
        ax.axhline(rf_val, ls="-", lw=1.2, color=RF_C, zorder=2)
        ax.text(
            0.16,
            rf_val,
            f" {RF_LABEL} {rf_val:.3f}",
            va="center",
            fontsize=5,
            color=RF_C,
        )
        ax.scatter(
            [], [], s=34, color=CGT_C, edgecolors=CGT_C, label="CGT val-selected"
        )
        ax.scatter(
            [], [], s=16, color="white", edgecolors=CGT_C, label="CGT other cells"
        )
        ax.set_xlim(-0.18, 0.42)
        ax.set_xticks([])
        ax.set_ylabel(lab)
        ax.grid(axis="y", lw=0.3, color="0.9", zorder=0)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=5, loc="lower right", handletextpad=0.3)
    STATS["fig6_cell_spread"] = {
        lab: {
            "per_cell": dict(zip(names, [float(v) for v in vals])),
            "min": float(min(vals)),
            "max": float(max(vals)),
            "fcl_rf": float(rf_val),
            "n_cells_above_fcl": int(sum(v > rf_val for v in vals)),
            "val_selected": float(vals[names.index(best)]),
        }
        for lab, vals, rf_val in metrics
    }
    fig.tight_layout(pad=0.4)
    save(fig, "merzbacher_fig6_cell_spread", ts)


def fig_label_provenance(t: np.ndarray, obs: np.ndarray, ts: str) -> None:
    """How well does MERZBACHER'S label track OUR copy of the same screen?

    The whole comparison is scored against their released labels, so it matters how tightly
    those labels correspond to the measurement we hold. They do not correspond tightly: their
    rule min-max scales production and cuts at 0.40/0.65, and the extremes it scales by differ
    between their (smaller) copy and ours. The classes therefore OVERLAP on our axis -- their
    "low" reaches +0.02 while their "medium" starts at -1.05.

    This is a ceiling on the whole exercise, not a footnote: a model perfectly predicting OUR
    measured value would still be scored wrong on ~19 % of genes by these labels.
    """
    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_WIDTHS_MM["half"]), mm_to_in(52)))
    rng = np.random.default_rng(0)
    for c in range(3):
        v = obs[t == c]
        ax.scatter(
            v,
            c + rng.normal(0, 0.09, len(v)),
            s=5,
            color=CLASS_COLORS[c],
            linewidths=0.0,
            zorder=3,
        )
        ax.plot(
            [np.median(v)] * 2, [c - 0.28, c + 0.28], color="black", lw=1.0, zorder=4
        )
    order = np.argsort(obs)
    n0, n1, _ = np.bincount(t, minlength=3)
    ideal = np.zeros(len(obs), dtype=int)
    ideal[order[:n0]] = 0
    ideal[order[n0 : n0 + n1]] = 1
    ideal[order[n0 + n1 :]] = 2
    disagree = float((ideal != t).mean())
    ax.set_yticks(range(3))
    ax.set_yticklabels(
        [f"{n}\n(n={int((t == c).sum())})" for c, n in enumerate(CLASS_NAMES)]
    )
    ax.set_xlabel("measured betaxanthin, Cachera screen (z)")
    ax.set_ylabel("released class label (FCL paper)")
    z_axis(ax.xaxis)
    ax.tick_params(which="minor", length=0)
    # A 3-level ordinal against a continuous variable is bounded well below 1 by its own
    # ties, so the raw Spearman is uninterpretable alone. The ceiling is the Spearman a
    # PERFECT 3-way rank-split of our values would achieve, and the ratio is what says whether
    # their labels track our copy of the screen well. They do: 88 % of attainable.
    rho_obs = float(spearmanr(t, obs).statistic)
    rho_max = float(spearmanr(ideal, obs).statistic)
    ax.set_title(
        "how much headroom do the released labels leave?\n"
        f"they track our measurement at {rho_obs:.3f} of an attainable {rho_max:.3f} "
        f"= {100 * rho_obs / rho_max:.0f}% of ceiling; "
        f"{100 * disagree:.1f}% of genes sit in a different bin than our values would give",
        fontsize=5,
        pad=3,
    )
    STATS["fig7_label_provenance"] = {
        "spearman_labels_vs_our_measurement": rho_obs,
        "spearman_ceiling_perfect_3way_split": rho_max,
        "fraction_of_ceiling": rho_obs / rho_max,
        "fraction_of_genes_in_a_different_bin": disagree,
    }
    ax.grid(axis="x", lw=0.3, color="0.9", zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.4)
    save(fig, "merzbacher_fig7_label_provenance", ts)


def load_gem_genes() -> set[str]:
    """Systematic ORF names in yeast-GEM -- the boundary of what a flux-based method can see."""
    df = pd.read_excel(GEM_PATH, sheet_name="GENES")
    return {str(v).strip().upper() for v in df["NAME"].dropna()}


def fig_screen_coverage(raw: dict, ts: str) -> None:
    """WHY a genome-scale model is needed at all -- how little of the screen FCL can reach.

    This is the motivating figure for the capability comparison, and it uses only the
    MEASURED screen: no model, no prediction, nothing to argue about. yeast-GEM contains
    1,161 genes, of which 915 appear in the Cachera screen -- 19.4 % of its 4,721 deletions.
    A flux-based method has features for those and for nothing else.

    The consequence is the second bar: of the 223 deletions the paper's own 0.65 cut calls
    HIGH producers, 163 (73 %) lie outside the metabolic model.

    STATE THE COUNTER-POINT HONESTLY, because it is real and it is small: metabolic genes ARE
    enriched among high producers -- 26.9 % of the highs against 19.4 % of the screen, a 1.4x
    enrichment, which is exactly what the biology predicts. It is simply nowhere near enough
    to matter: they are such a small share of the genome that the large majority of high
    producers is still outside the model. Restricting to metabolism buys a 1.4x concentration
    and costs three quarters of the hits.
    """
    gem = load_gem_genes()
    g = sorted(raw)
    v = np.array([raw[x] for x in g], dtype=float)
    in_gem = np.array([x in gem for x in g])
    lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
    is_hi = scale_bins(v, lo, hi) == 2
    order = np.argsort(-v)
    top100 = np.zeros(len(g), dtype=bool)
    top100[order[:100]] = True

    rows = [
        ("every deletion\nin the screen", np.ones(len(g), dtype=bool)),
        ("top 100 measured\nproducers", top100),
        ("all HIGH producers\n(their 0.65 cut)", is_hi),
    ]
    fig, ax = plt.subplots(
        figsize=(mm_to_in(PANEL_WIDTHS_MM["half_plus"]), mm_to_in(52))
    )
    y = np.arange(len(rows))
    for i, (name, mask) in enumerate(rows):
        n = int(mask.sum())
        n_gem = int((mask & in_gem).sum())
        n_out = n - n_gem
        STATS.setdefault("fig8_screen_coverage", {})[name.replace("\n", " ")] = {
            "n": n,
            "in_yeast_gem": n_gem,
            "outside_yeast_gem": n_out,
            "fraction_in_gem": n_gem / n,
        }
        ax.barh(
            i,
            n_gem / n,
            color=RF_C,
            edgecolor="black",
            linewidth=0.4,
            height=0.6,
            zorder=3,
            label="in yeast-GEM (FCL can score)" if i == 0 else None,
        )
        ax.barh(
            i,
            n_out / n,
            left=n_gem / n,
            color=CGT_C,
            edgecolor="black",
            linewidth=0.4,
            height=0.6,
            zorder=3,
            label="outside yeast-GEM (FCL cannot)" if i == 0 else None,
        )
        ax.text(
            n_gem / n / 2,
            i,
            f"{n_gem}\n{n_gem / n:.0%}",
            ha="center",
            va="center",
            fontsize=5,
            color="white",
        )
        ax.text(
            n_gem / n + n_out / n / 2,
            i,
            f"{n_out}\n{n_out / n:.0%}",
            ha="center",
            va="center",
            fontsize=5,
        )
        ax.text(1.015, i, f"n={n}", va="center", fontsize=5, color="0.35")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of deletions")
    ax.legend(
        frameon=False,
        ncol=2,
        fontsize=5,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        handletextpad=0.4,
        columnspacing=1.0,
    )
    # THE SCALE IS RECORDED, because this figure derives "high" from the FULL-SCREEN min-max
    # while Fig 9 derives it from the TRAIN-POOL min-max. Today those coincide exactly -- the
    # screen's extreme genes both sit in the pool -- so the two figures' labels agree. That is
    # a property of the current split, not a guarantee, and if a future split moves an extreme
    # gene into test the two "high" definitions would silently diverge.
    STATS["fig8_screen_coverage"]["_scale"] = {
        "rule": "full-screen min-max, thresholds " + str(list(MERZBACHER_THRESHOLDS)),
        "min": lo,
        "max": hi,
        "yeast_gem_genes": len(gem),
        "metabolic_enrichment_among_highs": float(
            (in_gem & is_hi).sum() / is_hi.sum() / (in_gem.sum() / len(g))
        ),
    }
    fig.tight_layout(pad=0.4)
    save(fig, "merzbacher_fig8_screen_coverage", ts)


def fig_nonmetabolic(dumps: list, raw: dict, split: dict, best: str, ts: str) -> None:
    """CGT on genes INSIDE vs OUTSIDE the metabolic model -- the capability difference.

    THE ARGUMENT. Merzbacher's features are flux samples drawn from yeast-GEM. Deleting a gene
    the model does not contain perturbs no reaction, so the flux distribution is unchanged and
    the method has no feature to predict from: its output for such a gene is not poor, it is
    UNDEFINED. Their own test set is 639/640 (99.8 %) inside yeast-GEM, which is what that
    constraint looks like in practice. CGT takes the whole-genome graph and is under no such
    restriction.

    So the question this figure asks is one their method cannot be asked at all: among the
    deletions OUTSIDE the metabolic model, does CGT still find high betaxanthin producers?

    LABELS ARE DERIVED HERE, and they have to be: Merzbacher never labelled these genes. Both
    panels therefore use the SAME rule on the SAME scale (their 0.40/0.65 cuts on a train-pool
    min-max), so the two sides are comparable to each other. That also means the left panel's
    class counts differ from their released labels -- see Fig 7 -- and the comparison to make
    is panel-vs-panel, not panel-vs-their-paper.

    A STRUCTURAL claim, not a measured one: we did not run their model on non-GEM genes. The
    claim is that their feature construction admits no such input, which follows from what
    flux sampling is.
    """
    gem = load_gem_genes()
    pool = np.array(
        [raw[x] for x in split["split"]["train_val_pool"] if x in raw], dtype=float
    )
    lo, hi = float(np.nanmin(pool)), float(np.nanmax(pool))
    d = [x for x in dumps if x["_setting"] == best][0]

    # THE COVERAGE RECONCILIATION THE NOTE QUOTES IN PROSE. Our test split is the pinned FCL
    # genes PLUS the ratio-driven remainder, and the two halves sit on opposite sides of the
    # metabolic model -- which is the whole reason the right panel has anything in it. Counted
    # over ALL test records here (the panels below drop the few with no raw screen value, so
    # their denominators are smaller).
    pinned = set(split["split"]["test"])
    other = [x for x in d["_genes"] if x not in pinned]
    STATS.setdefault("fig9_metabolic_vs_nonmetabolic", {})["_population"] = {
        "test_records": len(d["_genes"]),
        "pinned_fcl_genes": len(set(d["_genes"]) & pinned),
        "pinned_in_gem": sum(1 for x in d["_genes"] if x in pinned and x in gem),
        "other_test_genes": len(other),
        "other_in_gem": sum(1 for x in other if x in gem),
        "with_a_raw_screen_value": sum(1 for x in d["_genes"] if x in raw),
    }

    # SHARED LIMITS ACROSS THE TWO PANELS. They exist to be compared with each other, and
    # independent autoscaling drew the two gene sets at different zooms -- which would make a
    # difference in spread look like a difference in signal. Computed over the union first.
    all_obs = np.array([raw[x] for x in d["_genes"] if x in raw], dtype=float)
    all_pred = np.array([d["_genes"][x] for x in d["_genes"] if x in raw], dtype=float)
    pad = 0.12
    xlim = (all_obs.min() - pad, all_obs.max() + pad)
    ylim = (all_pred.min() - pad, all_pred.max() + pad)

    fig, axes = plt.subplots(
        1, 2, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(62))
    )
    panels = [
        ("in yeast-GEM\nFlux Cone Learning CAN model these", lambda x: x in gem),
        (
            "NOT in yeast-GEM\nFlux Cone Learning has no features here",
            lambda x: x not in gem,
        ),
    ]
    for ax, (title, sel) in zip(axes, panels):
        gs = sorted(x for x in d["_genes"] if x in raw and sel(x))
        obs = np.array([raw[x] for x in gs], dtype=float)
        pred = np.array([d["_genes"][x] for x in gs], dtype=float)
        cls = scale_bins(obs, lo, hi)
        is_hi = (cls == 2).astype(int)
        n, k_hi = len(gs), int(is_hi.sum())
        for c in range(3):
            m = cls == c
            ax.scatter(
                obs[m],
                pred[m],
                s=6,
                color=CLASS_COLORS[c],
                linewidths=0.0,
                label=f"{CLASS_NAMES[c]} (n={int(m.sum())})",
                zorder=3,
            )
        top = np.argsort(-pred)[:25]
        hits = int(is_hi[top].sum())
        pv = hypergeom.sf(hits - 1, n, k_hi, 25)
        ax.scatter(
            obs[top],
            pred[top],
            s=26,
            facecolors="none",
            edgecolors="black",
            linewidths=0.5,
            zorder=4,
            label="CGT top 25",
        )
        rho = float(spearmanr(pred, obs).statistic)
        STATS.setdefault("fig9_metabolic_vs_nonmetabolic", {})[
            "in_gem" if title.startswith("in yeast-GEM") else "not_in_gem"
        ] = {
            "n": n,
            "n_high_derived": k_hi,
            "base_rate": k_hi / n,
            "spearman_vs_measured": rho,
            "top25_high_found": hits,
            "enrichment": hits / (25 * k_hi / n),
            "p_hypergeom": float(pv),
        }
        ax.set_title(
            f"{title}\nn={n}, {k_hi} high ({k_hi / n:.1%})   Spearman {rho:+.3f}\n"
            f"CGT top 25: {hits} high = {hits / 25:.0%} vs {k_hi / n:.0%} expected "
            f"({hits / (25 * k_hi / n):.1f}x, p={pv:.0e})",
            fontsize=5,
            pad=4,
        )
        ax.set_xlabel("measured betaxanthin, Cachera screen (z)")
        ax.set_ylabel("CGT predicted betaxanthin (z)")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        z_axis(ax.xaxis)
        z_axis(ax.yaxis)
        ax.tick_params(which="minor", length=0)
        ax.grid(lw=0.3, color="0.92", zorder=0)
        ax.set_axisbelow(True)
        ax.legend(
            frameon=False,
            fontsize=4.5,
            loc="upper left",
            handletextpad=0.2,
            markerscale=1.4,
        )
    # IS THE RIGHT PANEL'S HIGHER CORRELATION A REAL DIFFERENCE? The two rho's are measured on
    # DIFFERENT, non-overlapping gene sets, so the comparison needs an explicit test rather
    # than an eyeball -- a Fisher r-to-z on two independent samples. It comes out marginal
    # (~2 sigma), which is why the note states the difference WITH its p-value. The claim that
    # does not depend on this test is the one that carries the figure: CGT finds high
    # producers outside the metabolic model at all, where FCL has no features.
    panel = STATS["fig9_metabolic_vs_nonmetabolic"]
    z_in, z_out = (
        np.arctanh(panel[k]["spearman_vs_measured"]) for k in ("in_gem", "not_in_gem")
    )
    se = np.sqrt(1 / (panel["in_gem"]["n"] - 3) + 1 / (panel["not_in_gem"]["n"] - 3))
    panel["_difference_test"] = {
        "method": "Fisher r-to-z, two independent samples",
        "delta_spearman": panel["not_in_gem"]["spearman_vs_measured"]
        - panel["in_gem"]["spearman_vs_measured"],
        "z": float((z_out - z_in) / se),
        "p_two_sided": float(2 * norm.sf(abs((z_out - z_in) / se))),
    }
    fig.tight_layout(pad=0.4, rect=(0.0, 0.09, 1.0, 1.0))
    fig.text(
        0.5,
        0.015,
        "Both panels are HELD-OUT test genes for CGT, under the same derived labels (their "
        "0.40/0.65 cuts on a train-pool min-max scale).\n"
        "The right panel is a question Flux Cone Learning cannot be asked: flux sampling gives "
        "no feature for a deletion outside the metabolic model.",
        ha="center",
        fontsize=4.5,
        color="0.35",
    )
    save(fig, "merzbacher_fig9_metabolic_vs_nonmetabolic", ts)


if __name__ == "__main__":
    main()
