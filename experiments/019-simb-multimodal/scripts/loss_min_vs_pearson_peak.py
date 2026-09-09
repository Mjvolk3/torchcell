# experiments/019-simb-multimodal/scripts/loss_min_vs_pearson_peak.py
# [[experiments.019-simb-multimodal.scripts.loss_min_vs_pearson_peak]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/loss_min_vs_pearson_peak
"""Where does `val/loss` bottom out, and where does the Pearson peak, on the v9 runs?

THE CLAIM UNDER TEST. The Pearson curves in the v9 objective round keep rising past 10,000
epochs, which is what motivated the long-budget arms. The loss curves may say something
different: if every `val/loss` minimum sits inside the first few hundred epochs, then the
later Pearson gains are bought while the model is already past its loss minimum, and the
long budget optimizes a metric the objective is not fitting. This script reads the FULL
history (not the downsampled 500-point leaderboard curve) of every v9 run that logs the
plain expression Pearson and reports, per run, the epoch of the loss minimum, the epoch of
the Pearson peak, and how far above its minimum the loss sits at a few fixed epochs.

WHAT IS COMPARED. `val/loss` is the objective the optimizer sees on the validation split:
mse for `point`, the CRPS or pinball loss for the distributional heads, so it is NOT the
same quantity across `dist` levels and only the epoch structure is comparable, never the
value. `val/expression/pearson_per_feature` is the leaderboard's primary metric, scored as
the leaderboard scores it: `_roll_max`, a centered ROLL_WINDOW-point rolling mean, imported
from pull_round_leaderboards.py so the statistic cannot drift. `epoch_loss_min` is the raw
argmin; the loss curve is far less noisy than the Pearson curve and needs no smoothing.

WHAT THE TABLE CANNOT SAY. A loss minimum at epoch 80 followed by a Pearson peak at epoch
12,000 shows the two quantities disagree; it does not say which one the paper should
believe. That is a decision about which metric is the target, and the table only sizes the
gap between them.

Resumed runs (curve starting above RESUME_START_EPOCH) are dropped, same rule as the two
sibling scripts, because a resume carries no history below its restart and would report a
loss minimum at the restart epoch. Collapsed runs (final Pearson numerically zero) are kept
in the table and flagged, and excluded from the summary statistics.
"""

from __future__ import annotations

import importlib.util
import json
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from dotenv import load_dotenv
from matplotlib.ticker import LogLocator

load_dotenv()

from torchcell.utils import (  # noqa: E402
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    PLOT_PALETTE_FILL,
    experiment_results_dir,
    mm_to_in,
    panel_label,
    savefig_true_size_svg,
)

_SPEC = importlib.util.spec_from_file_location(
    "_plb", osp.join(osp.dirname(osp.abspath(__file__)), "pull_round_leaderboards.py")
)
_PLB = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PLB)

ENTITY = _PLB.ENTITY
ROLL_WINDOW = _PLB.ROLL_WINDOW
PROJECT = "torchcell_019_expr_v9"
METRIC = "val/expression/pearson_per_feature"
LOSS = "val/loss"

RESULTS = experiment_results_dir("019-simb-multimodal", __file__)
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
IMAGE_DIR = osp.join(ASSET_IMAGES_DIR, "019-simb-multimodal")

# Large enough that W&B returns every logged row rather than a downsample. The longest v9
# run logs 18,990 epochs. The script checks the returned row count against the run's final
# epoch and reports any run that came back short, so a silent cap cannot pass as full data.
FULL_HISTORY_SAMPLES = 50_000
RESUME_START_EPOCH = 100
COLLAPSE_EPS = 1e-6
# Six v9 runs died inside their first 40 epochs (the DataLoader fd-exhaustion crashes).
# Their "loss minimum" is just their last epoch. They stay in the CSV and are excluded from
# every summary statistic and from the figure.
MIN_EPOCHS = 1000
# The Pearson-objective arms (2026-09-06) share the project but not the question.
METRIC_ALIGNED_DISTS = ("pearson", "pearson_mse")

# Fixed epochs at which the loss's excess over its minimum is tabulated. 100 is the user's
# stated threshold; the rest bracket it.
PROBE_EPOCHS = (50, 100, 200, 500, 1000, 2000)
# "Within tolerance of the minimum": the first epoch at which the loss is inside this
# relative margin of its eventual minimum.
TOL_REL = (0.01, 0.001)


def roll_mean(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values).rolling(window, center=True, min_periods=1).mean().to_numpy()


def nearest(hist: pd.DataFrame, epoch: int, col: str) -> float:
    i = int(np.argmin(np.abs(hist.epoch.to_numpy() - epoch)))
    return float(hist[col].iloc[i])


def main() -> None:
    api = wandb.Api(timeout=120)
    runs = list(api.runs(f"{ENTITY}/{PROJECT}", per_page=100))
    print(f"{len(runs)} runs in {PROJECT}")

    rows = []
    curves: dict[str, pd.DataFrame] = {}
    skipped: dict[str, str] = {}
    for run in runs:
        if METRIC not in run.summary or LOSS not in run.summary:
            skipped[run.id] = "no plain Pearson or no val/loss in summary"
            continue
        if run.config.get("dist") in METRIC_ALIGNED_DISTS:
            # These heads train on 1 - r, so their loss minimum and Pearson peak coincide
            # by construction on train and the question this script asks is empty for them.
            skipped[run.id] = f"metric-aligned head {run.config.get('dist')}"
            continue
        h = run.history(keys=["epoch", LOSS, METRIC], samples=FULL_HISTORY_SAMPLES)
        h = h.dropna(subset=["epoch", LOSS, METRIC]).sort_values("epoch")
        h = h.drop_duplicates("epoch", keep="last").reset_index(drop=True)
        if h.empty:
            skipped[run.id] = "empty history"
            continue
        if h.epoch.min() > RESUME_START_EPOCH:
            skipped[run.id] = f"resume, starts at epoch {int(h.epoch.min())}"
            continue
        n_epochs = int(h.epoch.max())
        short_by = (n_epochs + 1) - len(h)
        loss = h[LOSS].to_numpy()
        pear = h[METRIC].to_numpy()
        pear_roll = roll_mean(pear, ROLL_WINDOW)
        i_loss = int(np.argmin(loss))
        i_pear = int(np.argmax(pear_roll))
        loss_min = float(loss[i_loss])
        row = {
            "run_id": run.id,
            "dist": run.config.get("dist"),
            "seed": run.config.get("seed"),
            "dropout": run.config.get("dropout"),
            "n_epochs": n_epochs,
            "rows_short_by": int(short_by),
            "epoch_loss_min": int(h.epoch.iloc[i_loss]),
            "loss_min": loss_min,
            "loss_final": float(loss[-1]),
            "loss_final_rel_excess": float((loss[-1] - loss_min) / abs(loss_min)),
            "epoch_pearson_peak": int(h.epoch.iloc[i_pear]),
            "pearson_roll_max": float(pear_roll[i_pear]),
            "pearson_final": float(pear[-1]),
            "pearson_roll_at_loss_min": float(pear_roll[i_loss]),
            "loss_at_pearson_peak_rel_excess": float(
                (loss[i_pear] - loss_min) / abs(loss_min)
            ),
        }
        for e in PROBE_EPOCHS:
            if n_epochs >= e:
                row[f"loss_rel_excess_at_{e}"] = (nearest(h, e, LOSS) - loss_min) / abs(
                    loss_min
                )
                row[f"pearson_roll_at_{e}"] = float(
                    pear_roll[int(np.argmin(np.abs(h.epoch.to_numpy() - e)))]
                )
            else:
                row[f"loss_rel_excess_at_{e}"] = np.nan
                row[f"pearson_roll_at_{e}"] = np.nan
        for tol in TOL_REL:
            within = np.nonzero(loss <= loss_min + tol * abs(loss_min))[0]
            row[f"first_epoch_within_{tol:g}_of_loss_min"] = int(h.epoch.iloc[within[0]])
        row["collapsed"] = abs(float(pear[-1])) < COLLAPSE_EPS
        rows.append(row)
        curves[run.id] = h
        print(
            f"  {run.id} {str(row['dist']):>13} s{row['seed']} ep={n_epochs:>6} "
            f"loss_min@{row['epoch_loss_min']:>6} pear_peak@{row['epoch_pearson_peak']:>6} "
            f"roll_max={row['pearson_roll_max']:.4f} "
            f"loss@100 +{row['loss_rel_excess_at_100'] * 100:.2f}% "
            f"loss@end +{row['loss_final_rel_excess'] * 100:.2f}%"
            + ("  COLLAPSED" if row["collapsed"] else "")
            + (f"  SHORT BY {short_by} ROWS" if short_by else "")
        )

    t = pd.DataFrame(rows).sort_values(["dist", "seed", "n_epochs"]).reset_index(drop=True)
    csv_path = osp.join(RESULTS, "loss_min_vs_pearson_peak.csv")
    t.to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path}")
    for rid, why in skipped.items():
        print(f"  skipped {rid}: {why}")

    t["short"] = t.n_epochs < MIN_EPOCHS
    live = t[~t.collapsed & ~t.short]
    print(
        f"\n{len(t)} runs kept, {int(t.collapsed.sum())} collapsed, "
        f"{int(t.short.sum())} shorter than {MIN_EPOCHS} epochs, {len(live)} live"
    )
    print("\n=== live runs: where the loss bottoms out vs where the Pearson peaks ===")
    cols = [
        "run_id", "dist", "seed", "n_epochs", "epoch_loss_min", "epoch_pearson_peak",
        "pearson_roll_max", "pearson_roll_at_loss_min", "loss_rel_excess_at_100",
        "loss_final_rel_excess", "loss_at_pearson_peak_rel_excess",
    ]
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(live[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    summary: dict[str, object] = {
        "project": PROJECT,
        "n_runs": int(len(t)),
        "n_collapsed": int(t.collapsed.sum()),
        "n_short": int(t.short.sum()),
        "min_epochs": MIN_EPOCHS,
        "n_live": int(len(live)),
        "n_skipped": len(skipped),
        "skipped": skipped,
        "n_rows_short": int((t.rows_short_by > 0).sum()),
        "roll_window": ROLL_WINDOW,
        "live": {
            "epoch_loss_min_median": float(live.epoch_loss_min.median()),
            "epoch_loss_min_max": int(live.epoch_loss_min.max()),
            "n_loss_min_within_100": int((live.epoch_loss_min <= 100).sum()),
            "epoch_pearson_peak_median": float(live.epoch_pearson_peak.median()),
            "n_pearson_peak_within_100": int((live.epoch_pearson_peak <= 100).sum()),
            "pearson_gain_after_loss_min_median": float(
                (live.pearson_roll_max - live.pearson_roll_at_loss_min).median()
            ),
            "loss_rel_excess_at_100_median": float(live.loss_rel_excess_at_100.median()),
            "loss_final_rel_excess_median": float(live.loss_final_rel_excess.median()),
            "loss_at_pearson_peak_rel_excess_median": float(
                live.loss_at_pearson_peak_rel_excess.median()
            ),
        },
        "by_dist": {},
    }
    print("\n=== by dist (live only) ===")
    for dist, sub in live.groupby("dist"):
        d = {
            "n": int(len(sub)),
            "epoch_loss_min_median": float(sub.epoch_loss_min.median()),
            "epoch_loss_min_range": [int(sub.epoch_loss_min.min()), int(sub.epoch_loss_min.max())],
            "epoch_pearson_peak_median": float(sub.epoch_pearson_peak.median()),
            "epoch_pearson_peak_range": [
                int(sub.epoch_pearson_peak.min()), int(sub.epoch_pearson_peak.max()),
            ],
            "pearson_roll_at_loss_min_mean": float(sub.pearson_roll_at_loss_min.mean()),
            "pearson_roll_max_mean": float(sub.pearson_roll_max.mean()),
            "loss_final_rel_excess_median": float(sub.loss_final_rel_excess.median()),
        }
        summary["by_dist"][str(dist)] = d
        print(
            f"  {str(dist):>13} n={d['n']:>2} loss_min@ median {d['epoch_loss_min_median']:>7.0f} "
            f"range {d['epoch_loss_min_range']}  pearson_peak@ median "
            f"{d['epoch_pearson_peak_median']:>7.0f} range {d['epoch_pearson_peak_range']}  "
            f"pearson at loss-min {d['pearson_roll_at_loss_min_mean']:.4f} -> at peak "
            f"{d['pearson_roll_max_mean']:.4f}  loss at end +{d['loss_final_rel_excess_median'] * 100:.1f}%"
        )
    json_path = osp.join(RESULTS, "loss_min_vs_pearson_peak.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {json_path}")

    plot(t, curves)


def plot(t: pd.DataFrame, curves: dict[str, pd.DataFrame]) -> None:
    os.makedirs(IMAGE_DIR, exist_ok=True)
    t = t[~t.short].reset_index(drop=True)
    dists = sorted(t.dist.astype(str).unique())
    color = {d: PLOT_PALETTE[i] for i, d in enumerate(dists)}
    # Collapsed runs take the series' pale fill color (a lighter SOLID, per the two-level
    # convention), never an alpha, which would fade the black marker edges too.
    pale = {d: PLOT_PALETTE_FILL[i] for i, d in enumerate(dists)}
    plt.rcParams.update({"legend.frameon": True, "legend.fancybox": False,
                         "legend.framealpha": 1.0, "legend.edgecolor": "black",
                         "legend.facecolor": "white", "patch.linewidth": 0.5,
                         "font.family": "Arial", "svg.fonttype": "none"})
    # 58 mm tall with an 0.17 bottom margin: at 55 mm and 0.14 the x labels were clipped
    # off the bottom of the exported panel.
    fig, axes = plt.subplots(
        1, 3, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(58.0))
    )
    ax_loss, ax_pear, ax_sc = axes
    for _, r in t.iterrows():
        h = curves[r.run_id]
        c = pale[str(r.dist)] if r.collapsed else color[str(r.dist)]
        style = dict(color=c, lw=0.5)
        ax_loss.plot(h.epoch + 1, h[LOSS] / r.loss_min, **style)
        ax_pear.plot(h.epoch + 1, roll_mean(h[METRIC].to_numpy(), ROLL_WINDOW), **style)
        ax_loss.plot(
            r.epoch_loss_min + 1, 1.0, marker="o", ms=2, mec="black", mew=0.4, mfc=c, ls=""
        )
        ax_sc.plot(
            r.epoch_loss_min + 1,
            r.epoch_pearson_peak + 1,
            marker="o", ms=3, mec="black", mew=0.4, mfc=c, ls="",
        )
    ax_loss.set_xscale("log")
    # Linear, clipped: the rise after the minimum is 5-30% and a log axis renders that band
    # as unreadable "x10^0" ticks. One run starts at 5x its minimum and leaves the top.
    ax_loss.set_ylim(0.95, 1.6)
    ax_loss.set_xlabel("epoch + 1")
    ax_loss.set_ylabel("val/loss relative to its minimum")
    ax_loss.set_title("val/loss, marker at the minimum", fontsize=6, pad=3)
    ax_pear.set_xscale("log")
    ax_pear.set_xlabel("epoch + 1")
    ax_pear.set_ylabel(f"val Pearson per feature ({ROLL_WINDOW}-epoch mean)")
    # Headroom above every curve (max ~0.24) so the framed legend sits on clear white.
    ax_pear.set_ylim(-0.04, 0.34)
    ax_pear.set_title("Pearson, the leaderboard metric", fontsize=6, pad=3)
    lim = (0.8, max(t.n_epochs.max(), 10) * 1.5)
    ax_sc.plot(lim, lim, color="black", lw=0.5, ls="--")
    ax_sc.set_xscale("log")
    ax_sc.set_yscale("log")
    ax_sc.set_xlim(lim)
    ax_sc.set_ylim(lim)
    ax_sc.set_xlabel("epoch of val/loss minimum + 1")
    ax_sc.set_ylabel("epoch of Pearson peak + 1")
    ax_sc.set_title("loss minimum vs Pearson peak", fontsize=6, pad=3)
    for ax in axes:
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=6))
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(0.5)
        ax.tick_params(labelsize=6, width=0.5, length=2)
        ax.xaxis.label.set_size(6)
        ax.yaxis.label.set_size(6)
    handles = [
        plt.Line2D([], [], color=color[d], lw=1, label=f"{d} (n={int((t.dist.astype(str) == d).sum())})")
        for d in dists
    ]
    # The pale swatch takes the head that collapses most often, so the legend shows a
    # pale color that actually appears in the panels.
    most_collapsed = t[t.collapsed].dist.astype(str).mode()
    pale_dist = most_collapsed.iloc[0] if len(most_collapsed) else dists[0]
    handles.append(plt.Line2D([], [], color=pale[pale_dist], lw=1, label="pale: collapsed run"))
    ax_pear.legend(handles=handles, fontsize=5, loc="upper left", ncol=2, borderpad=0.3,
                   columnspacing=0.8)
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.17, top=0.85, wspace=0.32)
    for ax, letter in zip(axes, "abc"):
        panel_label(ax, letter)
    # Stable name, no timestamp: notes-tex's `make plots` converts figures by name.
    stem = osp.join(IMAGE_DIR, "loss_min_vs_pearson_peak")
    fig.savefig(stem + ".png", dpi=300)
    savefig_true_size_svg(fig, stem + ".svg")
    print(f"wrote {stem}.png\nwrote {stem}.svg")


if __name__ == "__main__":
    main()
