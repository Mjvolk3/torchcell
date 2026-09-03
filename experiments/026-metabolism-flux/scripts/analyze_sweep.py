# experiments/026-metabolism-flux/scripts/analyze_sweep.py
# [[experiments.026-metabolism-flux.scripts.analyze_sweep]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/analyze_sweep.py

r"""Score the overnight sweep against its own calibrated null.

WHAT THIS FIXES
---------------
The banked 20-epoch comparison reports each arm by the MAXIMUM of its validation
Pearson over epochs. That is an upward-biased order statistic whose bias grows with the
number of epochs run, so it is not the arm's score and two arms run for different
numbers of epochs are not comparable at all. Worse, nothing so far establishes how large
that maximum gets when there is no signal to find, and the validation set is small
enough for that to be the dominant term: at n = 353 betaxanthin measurements a single
Pearson has null width :math:`1/\sqrt{n-3} = 0.0535`.

So every number here is reported three ways, and the scoring rule is named next to the
number rather than left implicit:

``peak``
    the maximum over epochs, the statistic the earlier runs used. Kept only so the new
    results can be compared against the old ones on equal terms.
``peak_epoch``
    where that maximum fell. A peak at epoch 3 of 30 is an overfitting signature, not a
    performance one.
``last5``
    the mean of the final five epochs, which is not an order statistic and is therefore
    the honest summary of where training actually ended up.

THE NULL, AND WHY IT IS THE POINT
----------------------------------
The ``null`` grid runs the same architectures with training targets permuted and
validation left real, so its ``peak`` values are draws from the null distribution of the
exact statistic the arms are reported with, under the real epoch-to-epoch correlation
rather than an assumed independence. An arm is only distinguishable from nothing if its
peak sits above that distribution. The empirical p-value reported per arm is the
fraction of null draws at or above the arm's mean peak, which needs no normality
assumption and no correction for the epoch maximum, because the null was drawn through
the same maximum.

Run from the worktree root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/analyze_sweep.py
"""

import glob
import json
import os
import os.path as osp
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from matplotlib.ticker import MultipleLocator
from pydantic import BaseModel, ConfigDict

from torchcell.timestamp import timestamp
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()

ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")

#: The phenotype the flux layer was built for. Betaxanthin is the production phenotype
#: whose 0.914 replicate ceiling motivates the whole layer.
HEAD = "betaxanthin"

#: Scored files, and the group each contributes to. The `flux_arms_gpu*` files are the
#: banked 20-epoch runs; they are read so the new seeds extend them rather than replace
#: them, but they are tagged so a 20-epoch peak is never silently pooled with a
#: 30-epoch peak, which would compare two different order statistics.
SWEEP_GLOBS = {
    "null": "sweep_null.json",
    "arms": "sweep_arms_*.json",
    "reg": "sweep_reg_*.json",
    "banked": "flux_arms_gpu*.json",
}


class RunScore(BaseModel):
    """One training run reduced to the three statistics, with its provenance."""

    model_config = ConfigDict(extra="forbid")

    group: str
    label: str
    arm: str
    seed: int
    epochs: int
    n_val: int
    peak: float
    peak_epoch: int
    last5: float
    source_file: str


def score_history(history: list[dict[str, Any]]) -> tuple[float, int, float, int]:
    """Reduce one run's per-epoch history to (peak, peak epoch, last-5 mean, n_val).

    Epochs whose metric could not be computed are dropped rather than treated as zero.
    A NaN epoch is an absent measurement, and scoring it as 0.0 would drag the last-five
    mean toward zero and make a broken run look like a merely bad one.
    """
    vals = [(int(r["epoch"]), float(r[f"val_{HEAD}"])) for r in history]
    finite = [(e, v) for e, v in vals if np.isfinite(v)]
    if not finite:
        return float("nan"), -1, float("nan"), 0
    peak_epoch, peak = max(finite, key=lambda ev: ev[1])
    last5 = float(np.mean([v for _, v in finite[-5:]]))
    n_val = int(history[-1].get(f"n_val_{HEAD}", 0))
    return peak, peak_epoch, last5, n_val


def load_scores() -> list[RunScore]:
    """Read every sweep results file present and reduce each run to a RunScore."""
    scores: list[RunScore] = []
    for group, pattern in SWEEP_GLOBS.items():
        for path in sorted(glob.glob(osp.join(RESULTS_DIR, pattern))):
            payload = json.load(open(path))
            for run in payload["runs"]:
                peak, peak_epoch, last5, n_val = score_history(run["history"])
                cell = run.get("cell", {})
                scores.append(
                    RunScore(
                        group=group,
                        label=cell.get("label", run["arm"]),
                        arm=run["arm"],
                        seed=int(run["seed"]),
                        epochs=len(run["history"]),
                        n_val=n_val,
                        peak=peak,
                        peak_epoch=peak_epoch,
                        last5=last5,
                        source_file=osp.basename(path),
                    )
                )
    return scores


def summarize(scores: list[RunScore]) -> dict[str, Any]:
    """Per-label means, and every arm's empirical p-value against the null draws."""
    null_peaks = np.array(
        [s.peak for s in scores if s.group == "null" and np.isfinite(s.peak)]
    )
    by_label: dict[str, list[RunScore]] = {}
    for s in scores:
        if s.group == "null":
            continue
        by_label.setdefault(f"{s.group}:{s.label}", []).append(s)

    rows: list[dict[str, Any]] = []
    for key, runs in sorted(by_label.items()):
        peaks = np.array([r.peak for r in runs if np.isfinite(r.peak)])
        last5 = np.array([r.last5 for r in runs if np.isfinite(r.last5)])
        if peaks.size == 0:
            continue
        # The empirical p-value: how often a model that provably cannot learn the
        # association still reaches this arm's mean peak. `+1` in both terms is the
        # standard finite-sample correction, so a p-value is never reported as exactly
        # zero on a null of finite size.
        p_emp = (
            float((np.sum(null_peaks >= peaks.mean()) + 1) / (null_peaks.size + 1))
            if null_peaks.size
            else float("nan")
        )
        rows.append(
            {
                "label": key,
                "n_runs": int(peaks.size),
                "peak_mean": float(peaks.mean()),
                "peak_sem": float(peaks.std(ddof=1) / np.sqrt(peaks.size))
                if peaks.size > 1
                else float("nan"),
                "peak_epoch_median": float(
                    np.median([r.peak_epoch for r in runs if r.peak_epoch >= 0])
                ),
                "last5_mean": float(last5.mean()) if last5.size else float("nan"),
                "p_vs_null": p_emp,
            }
        )

    # `n_val_<head>` was added with the sweep, so the banked 20-epoch files do not carry
    # it and report 0. Reporting the analytic null width off a count that is not there
    # would invent a number, so it stays absent instead.
    n_vals = [s.n_val for s in scores if s.n_val > 0]
    n_val = int(np.median(n_vals)) if n_vals else 0
    return {
        "head": HEAD,
        "n_val_median": n_val,
        "analytic_null_sd": float((n_val - 3) ** -0.5) if n_val > 3 else float("nan"),
        "null": {
            "n_draws": int(null_peaks.size),
            "peak_mean": float(null_peaks.mean()) if null_peaks.size else float("nan"),
            "peak_sd": float(null_peaks.std(ddof=1))
            if null_peaks.size > 1
            else float("nan"),
            "peak_p95": float(np.percentile(null_peaks, 95))
            if null_peaks.size
            else float("nan"),
        },
        "rows": rows,
    }


def plot(summary: dict[str, Any], scores: list[RunScore], out_stem: str) -> None:
    """Per-arm peak against the calibrated null band, one `half` panel."""
    rows = [r for r in summary["rows"] if r["label"].startswith("arms:")]
    if not rows:
        return
    rows = sorted(rows, key=lambda r: r["peak_mean"])
    names = [r["label"].removeprefix("arms:arms-") for r in rows]
    means = [r["peak_mean"] for r in rows]
    sems = [r["peak_sem"] for r in rows]

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "svg.fonttype": "none",
            "axes.linewidth": 0.5,
        }
    )
    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_WIDTHS_MM["half"]), mm_to_in(62.0)))

    null = summary["null"]
    if np.isfinite(null["peak_p95"]):
        ax.axhspan(
            0.0, null["peak_p95"], color=PLOT_PALETTE[5], alpha=0.15, lw=0, zorder=0
        )
        ax.axhline(
            null["peak_mean"],
            color=PLOT_PALETTE[5],
            lw=0.6,
            ls="--",
            zorder=1,
            label=f"null mean ({null['n_draws']} permuted runs)",
        )
        ax.axhline(
            null["peak_p95"],
            color=PLOT_PALETTE[5],
            lw=0.6,
            ls=":",
            zorder=1,
            label="null 95th percentile",
        )

    x = np.arange(len(names))
    ax.bar(
        x,
        means,
        yerr=sems,
        color=PLOT_PALETTE[: len(names)],
        edgecolor="black",
        linewidth=0.5,
        capsize=1.5,
        error_kw={"elinewidth": 0.5, "capthick": 0.5},
        zorder=2,
    )
    for xi, r in zip(x, rows):
        ax.annotate(
            f"n={r['n_runs']}",
            (xi, 0.002),
            ha="center",
            va="bottom",
            fontsize=5,
            color="black",
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Validation Pearson (peak over epochs)")
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.025))
    ax.tick_params(which="minor", length=0)
    ax.grid(axis="y", which="both", lw=0.3, color="0.85", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=5, loc="upper left", handlelength=1.6, borderpad=0.2)
    for spine in ax.spines.values():
        spine.set_visible(True)
    fig.tight_layout(pad=0.3)

    png = osp.join(ASSET_IMAGES_DIR, f"{out_stem}.png")
    svg = osp.join(ASSET_IMAGES_DIR, f"{out_stem}.svg")
    fig.savefig(png, dpi=300)
    savefig_true_size_svg(fig, svg)
    plt.close(fig)
    print(f"figure -> {png}\nfigure -> {svg}")


def main() -> None:
    scores = load_scores()
    if not scores:
        raise SystemExit(f"no sweep results found under {RESULTS_DIR}")
    summary = summarize(scores)

    print(f"\nhead {summary['head']}, n_val median {summary['n_val_median']}")
    print(f"analytic null sd 1/sqrt(n-3) = {summary['analytic_null_sd']:.4f}")
    n = summary["null"]
    print(
        f"empirical null over {n['n_draws']} permuted runs: "
        f"mean peak {n['peak_mean']:.4f}, sd {n['peak_sd']:.4f}, "
        f"95th pct {n['peak_p95']:.4f}\n"
    )
    print(f"{'label':44}{'n':>4}{'peak':>9}{'sem':>8}{'pk@':>6}{'last5':>9}{'p':>8}")
    for r in summary["rows"]:
        print(
            f"{r['label']:44}{r['n_runs']:>4}{r['peak_mean']:>9.4f}"
            f"{r['peak_sem']:>8.4f}{r['peak_epoch_median']:>6.0f}"
            f"{r['last5_mean']:>9.4f}{r['p_vs_null']:>8.3f}"
        )

    stamp = timestamp()
    out_json = osp.join(RESULTS_DIR, f"sweep_summary_{stamp}.json")
    with open(out_json, "w") as f:
        json.dump(
            {"summary": summary, "runs": [s.model_dump() for s in scores]}, f, indent=2
        )
    print(f"\nsummary -> {out_json}")
    plot(summary, scores, f"026-metabolism-flux/sweep_arms_vs_null_{stamp}")


if __name__ == "__main__":
    main()
