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
from scipy.stats import spearmanr

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
RF_LABEL = "Cachera RF"
CGT_LABEL = "CGT"

#: A sequential map built from the palette orange, replacing matplotlib's built-in `Oranges`
#: -- which is close to our hue but is NOT our color, and a figure that mixes the two reads as
#: two different oranges.
CGT_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "tc_orange", ["#FFFFFF", PLOT_PALETTE_FILL[0], PLOT_PALETTE[0], PLOT_PALETTE[6]]
)
_SETTING_TAG = re.compile(r"^s\d+_")

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

    obs = np.array([raw[g] for g in genes], dtype=float)
    fig_scatter(cgt, rf, t, obs, theirs, genes, best, ts)
    fig_accuracy(cgt, rf, t, best, ts)
    fig_distribution(cgt, rf, t, best, ts)
    fig_recall(cgt, rf, t, best, ts)
    fig_confusion(cgt, rf, t, best, ts)


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
            "measured betaxanthin (screen)",
            "Cachera RF   E[class]",
        ),
        (
            f"{CGT_LABEL} vs measured",
            obs,
            cgt["raw"],
            "measured betaxanthin (screen)",
            "CGT   predicted (z)",
        ),
        (
            f"{CGT_LABEL} vs {RF_LABEL}",
            rf_score,
            cgt["raw"],
            "Cachera RF   E[class]",
            "CGT   predicted (z)",
        ),
    ]
    fig, axes = plt.subplots(
        1, 3, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(56))
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
        rho = float(spearmanr(y, x).statistic)
        # The 10th-90th percentile band of the y variable: the numeric statement of how
        # compressed that model's output is, independent of where the cloud sits on the axis.
        p10, p90 = np.percentile(y, [10, 90])
        ax.axhspan(p10, p90, color="0.88", alpha=0.5, zorder=1)
        ax.set_title(
            f"{name}\nSpearman {rho:+.3f}    80% of y within {p90 - p10:.2f}",
            fontsize=5,
            pad=3,
        )
        ax.set_xlabel(xlab, fontsize=5.5)
        ax.set_ylabel(ylab, fontsize=5.5)
        ax.tick_params(labelsize=5)
        ax.grid(lw=0.3, color="0.92", zorder=0)
        ax.set_axisbelow(True)
    axes[0].legend(
        frameon=False,
        fontsize=4.5,
        loc="upper left",
        handletextpad=0.2,
        markerscale=1.8,
        title="true class",
        title_fontsize=4.5,
    )
    fig.tight_layout(pad=0.4)
    save(fig, "merzbacher_fig5_scatter_spread", ts)


def fig_accuracy(cgt: dict, rf: dict, t: np.ndarray, best: str, ts: str) -> None:
    """STANDALONE: as-published vs rank-matched accuracy."""
    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_WIDTHS_MM["half"]), mm_to_in(60)))
    draw_accuracy(ax, cgt, rf, t, best)
    fig.tight_layout(pad=0.4)
    save(fig, "merzbacher_fig1_accuracy_artifact", ts)


def fig_distribution(cgt: dict, rf: dict, t: np.ndarray, best: str, ts: str) -> None:
    """STANDALONE: where each model puts the genes."""
    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_WIDTHS_MM["half"]), mm_to_in(52)))
    draw_distribution(ax, cgt, rf, t)
    fig.tight_layout(pad=0.4)
    save(fig, "merzbacher_fig2_class_distribution", ts)


def fig_recall(cgt: dict, rf: dict, t: np.ndarray, best: str, ts: str) -> None:
    """STANDALONE: per-class recall."""
    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_WIDTHS_MM["half"]), mm_to_in(58)))
    draw_recall(ax, cgt, rf, t)
    fig.tight_layout(pad=0.4)
    save(fig, "merzbacher_fig3_per_class_recall", ts)


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
    save(fig, "merzbacher_fig4_confusion_rank_matched", ts)


if __name__ == "__main__":
    main()
