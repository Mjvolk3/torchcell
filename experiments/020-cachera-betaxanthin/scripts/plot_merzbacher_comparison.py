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

from torchcell.timestamp import timestamp
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
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
#: assignment would not be.
CLASS_COLORS = [PLOT_PALETTE[1], PLOT_PALETTE[3], PLOT_PALETTE[4]]
CLASS_NAMES = ["low", "medium", "high"]
#: ours = orange (palette 0), Cachera = gray (palette 5), truth = purple (palette 2).
OURS_C, THEIRS_C, TRUTH_C = PLOT_PALETTE[0], PLOT_PALETTE[5], PLOT_PALETTE[2]
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

    Ties are broken by position, which is arbitrary but affects only exact ties; the
    regression outputs are continuous so this is inert in practice.
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
        p = np.array([d["_genes"][g] for g in genes], dtype=float)
        ours[d["_setting"]] = {
            "abs": scale_bins(p, lo, hi),
            "rank": rank_bins(p, counts),  # type: ignore[arg-type]
            "raw": p,
        }

    panel_a(ours, t, baseline, ts, theirs, genes, counts)
    panel_b(ours, t, theirs, genes, counts, ts)
    panel_c(ours, t, theirs, genes, counts, ts)
    panel_d(ours, t, theirs, genes, counts, ts)


# ---------------------------------------------------------------------------------- panels


def acc(pred: np.ndarray, true: np.ndarray) -> float:
    return float((pred == true).mean())


def panel_a(
    ours: dict[str, Any],
    t: np.ndarray,
    baseline: dict[str, Any],
    ts: str,
    theirs: dict[str, Any] | None = None,
    genes: list[str] | None = None,
    counts: tuple[int, ...] | None = None,
) -> None:
    """Gene-level accuracy: their four models against our grid cells.

    GENE LEVEL ON BOTH SIDES. Their Fig 4b plots per-FLUX-SAMPLE accuracy across 5 folds
    (0.63-0.70); their gene-level numbers are different (0.56-0.70) and are the only ones
    comparable to ours, since we predict one value per gene. Plotting our gene-level numbers
    against their Fig 4b points would be a different population on each axis.
    """
    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_WIDTHS_MM["half"]), mm_to_in(52)))
    # NOT named `theirs` -- that is the parameter carrying their per-gene SCORES, and
    # shadowing it here silently fed the wrong dict to the rank-matched block below.
    their_gl = baseline["gene_level"]
    names = sorted(their_gl, key=lambda k: their_gl[k]["gene_level_accuracy"])
    tv = [their_gl[k]["gene_level_accuracy"] for k in names]
    ax.scatter(
        range(len(names)),
        tv,
        s=14,
        color=THEIRS_C,
        zorder=3,
        label="Cachera (as published)",
    )
    # THEIR rank-matched accuracy, on the same rule applied to our side. Without this the
    # open markers would look like a penalty we alone pay, when in fact forcing the true
    # marginal costs BOTH models the conservative-majority advantage.
    if theirs and genes and counts:
        for i, nm in enumerate(names):
            if nm not in theirs:
                continue
            sc = np.array([theirs[nm]["score"].get(g, np.nan) for g in genes])
            ok = ~np.isnan(sc)
            if ok.sum() < 10:
                continue
            ax.scatter(
                i,
                acc(rank_bins(sc[ok], counts), t[ok]),
                s=14,
                color=THEIRS_C,
                zorder=3,
                facecolors="none",
                linewidths=0.6,
                label="Cachera (rank-matched)" if i == 0 else None,
            )
    # Ours: absolute binning (deployable) and rank-matched (ordering only).
    keys = sorted(ours)
    x0 = len(names)
    for i, k in enumerate(keys):
        ax.scatter(
            x0 + i,
            acc(ours[k]["abs"], t),
            s=14,
            color=OURS_C,
            zorder=3,
            label="ours (absolute bins)" if i == 0 else None,
        )
        ax.scatter(
            x0 + i,
            acc(ours[k]["rank"], t),
            s=14,
            color=OURS_C,
            zorder=3,
            facecolors="none",
            linewidths=0.6,
            label="ours (rank-matched)" if i == 0 else None,
        )
    maj = float(np.bincount(t, minlength=3).max() / len(t))
    ax.axhline(maj, ls="--", lw=0.6, color="0.35", zorder=2)
    ax.text(
        len(names) + len(keys) - 0.4,
        maj + 0.004,
        f"majority class {maj:.3f}",
        ha="right",
        va="bottom",
        fontsize=5,
        color="0.35",
    )
    labels = [n.replace("Classifier", "").replace("_", "\n") for n in names] + [
        k.split("_", 1)[1] if "_" in k else k for k in keys
    ]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=4.5)
    ax.set_ylabel("gene-level accuracy")
    ax.set_ylim(0.40, 0.80)
    tenth_grid(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=5, handletextpad=0.4)
    fig.tight_layout(pad=0.3)
    save(fig, "merzbacher_cmp_a_accuracy", ts)


def panel_b(
    ours: dict[str, Any],
    t: np.ndarray,
    theirs: dict[str, Any],
    genes: list[str],
    counts: tuple[int, ...],
    ts: str,
) -> None:
    """Predicted class distribution: truth vs their deployed call vs ours.

    This is the panel that shows the shared failure mode. Their best model calls 94.8 % of
    genes medium; our absolute binning calls a similar share and never calls LOW at all. A
    model that predicts the majority everywhere scores well on accuracy and has learned
    nothing about the tails, which is the whole point of the task.
    """
    rows: list[tuple[str, np.ndarray]] = [("truth", t)]
    for model in ("RandomForestClassifier_Resampled", "LogisticRegression"):
        if model in theirs:
            hard = np.array([theirs[model]["hard"].get(g, 1) for g in genes], dtype=int)
            rows.append((f"Cachera\n{model.replace('Classifier', '')}", hard))
    best = focus_cell(ours)
    rows.append((f"ours abs\n{best.split('_', 1)[-1]}", ours[best]["abs"]))
    rows.append((f"ours rank\n{best.split('_', 1)[-1]}", ours[best]["rank"]))

    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_WIDTHS_MM["half"]), mm_to_in(48)))
    y = np.arange(len(rows))
    left = np.zeros(len(rows))
    for c in range(3):
        w = np.array([float((p == c).mean()) for _, p in rows])
        ax.barh(
            y,
            w,
            left=left,
            height=0.62,
            color=CLASS_COLORS[c],
            edgecolor="black",
            linewidth=0.4,
            label=CLASS_NAMES[c],
            zorder=3,
        )
        for yi, (wi, li) in enumerate(zip(w, left)):
            if wi > 0.06:
                ax.text(
                    li + wi / 2,
                    yi,
                    f"{wi:.2f}",
                    ha="center",
                    va="center",
                    fontsize=4.5,
                    color="black",
                )
        left += w
    ax.set_yticks(y)
    ax.set_yticklabels([n for n, _ in rows], fontsize=4.5)
    ax.invert_yaxis()
    ax.set_xlabel("fraction of genes assigned to each class")
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
    fig.tight_layout(pad=0.3)
    save(fig, "merzbacher_cmp_b_class_distribution", ts)


def panel_c(
    ours: dict[str, Any],
    t: np.ndarray,
    theirs: dict[str, Any],
    genes: list[str],
    counts: tuple[int, ...],
    ts: str,
) -> None:
    """Per-class recall -- what a single accuracy number hides.

    Their own caption concedes the trade ("class rebalancing can increase accuracy for high
    producers, often at the expense of overall accuracy"), but Fig 4b plots only the overall
    number. Recall per class is where the models actually differ.
    """
    series: list[tuple[str, np.ndarray]] = []
    for model in ("RandomForestClassifier_Resampled", "LogisticRegression"):
        if model in theirs:
            series.append(
                (
                    f"Cachera {model.replace('Classifier', '')}",
                    np.array(
                        [theirs[model]["hard"].get(g, 1) for g in genes], dtype=int
                    ),
                )
            )
    best = focus_cell(ours)
    series.append((f"ours abs {best.split('_', 1)[-1]}", ours[best]["abs"]))
    series.append((f"ours rank {best.split('_', 1)[-1]}", ours[best]["rank"]))

    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_WIDTHS_MM["half"]), mm_to_in(46)))
    w = 0.8 / len(series)
    x = np.arange(3)
    for i, (name, pred) in enumerate(series):
        rec = [
            float((pred[t == c] == c).mean()) if (t == c).any() else np.nan
            for c in range(3)
        ]
        ax.bar(
            x + i * w - 0.4 + w / 2,
            rec,
            width=w * 0.9,
            color=PLOT_PALETTE[[5, 11, 0, 6][i % 4]],
            edgecolor="black",
            linewidth=0.4,
            label=name,
            zorder=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n(n={counts[i]})" for i, n in enumerate(CLASS_NAMES)])
    ax.set_ylabel("recall within true class")
    ax.set_ylim(0, 1.0)
    tenth_grid(ax)
    ax.legend(
        frameon=False,
        fontsize=4.5,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.22),
        handletextpad=0.4,
        columnspacing=0.8,
    )
    fig.tight_layout(pad=0.3)
    save(fig, "merzbacher_cmp_c_per_class_recall", ts)


def panel_d(
    ours: dict[str, Any],
    t: np.ndarray,
    theirs: dict[str, Any],
    genes: list[str],
    counts: tuple[int, ...],
    ts: str,
) -> None:
    """Row-normalized confusion matrices with BOTH sides rank-matched -- apples-to-apples.

    Their model is re-binned from its own gene-level E[class] score by the same rule applied
    to ours, so neither side gets the conservative "call everything medium" advantage and
    neither gets a marginal the other lacks. This is the only panel where the two numbers
    answer exactly the same question.
    """
    model = "RandomForestClassifier_Resampled"
    their_score = np.array([theirs[model]["score"].get(g, np.nan) for g in genes])
    ok = ~np.isnan(their_score)
    best = focus_cell(ours)
    panels = [
        (
            f"Cachera {model.replace('Classifier', '')}\nrank-matched",
            rank_bins(their_score[ok], counts),
            t[ok],
        ),  # type: ignore[arg-type]
        (f"ours {best.split('_', 1)[-1]}\nrank-matched", ours[best]["rank"], t),
    ]
    fig, axes = plt.subplots(
        1, len(panels), figsize=(mm_to_in(PANEL_WIDTHS_MM["half"]), mm_to_in(54))
    )
    for ax, (name, pred, tt) in zip(np.atleast_1d(axes), panels):
        cm = np.zeros((3, 3))
        for a in range(3):
            for b in range(3):
                cm[a, b] = float(((tt == a) & (pred == b)).sum())
        cmn = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(cmn, cmap="Oranges", vmin=0, vmax=1)
        for a in range(3):
            for b in range(3):
                ax.text(
                    b,
                    a,
                    f"{cmn[a, b]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=4.5,
                    color="black" if cmn[a, b] < 0.6 else "white",
                )
        ax.set_xticks(range(3))
        ax.set_xticklabels(CLASS_NAMES, fontsize=4.5)
        ax.set_yticks(range(3))
        ax.set_yticklabels(CLASS_NAMES, fontsize=4.5)
        ax.set_xlabel("predicted")
        ax.set_title(f"{name}\nacc {acc(pred, tt):.3f}", fontsize=4.5, pad=3)
        for s in ax.spines.values():
            s.set_linewidth(0.5)
    np.atleast_1d(axes)[0].set_ylabel("true")
    fig.tight_layout(pad=0.3)
    save(fig, "merzbacher_cmp_d_confusion_rank_matched", ts)


if __name__ == "__main__":
    main()
