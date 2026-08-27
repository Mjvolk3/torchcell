# experiments/019-simb-multimodal/scripts/expression_objective_diagnosis.py
# [[experiments.019-simb-multimodal.scripts.expression_objective_diagnosis]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/expression_objective_diagnosis
"""What the expression objective is actually doing, from the longest run ever trained.

THE CLAIM THIS REPLACES. An earlier retrospective said loss and metric "point in opposite
directions" on expression. That is true of the QUANTILE LOSS and false of squared error,
and collapsing the two hid the finding. Measured on the longest run (10,000 epochs,
91 hours), `val/loss` bottoms at epoch ~463 and rises for the remaining 9,500, while
`val/expression/mse` bottoms at epoch ~9,175, essentially alongside the Pearson peak at
~9,674. So squared error and rank agreement improve TOGETHER, late; it is the quantile
loss that turns around early.

THE SECOND FINDING, and it is the sharper one. `nmse` is normalized so that
`nmse = 1` is exactly "predict each gene's training mean". The model never gets below 1
after epoch ~900. It reaches Pearson 0.236 while being no better than the mean predictor in
squared error, which is the precise form of "it has to fight MSE".

WHY THAT HAPPENS, and it is arithmetic rather than a story. Write s for the ratio of
prediction SD to target SD (`pred_sd_ratio`) and r for the per-feature correlation. For
predictions that are a scaled version of a correlated signal,

    nmse = 1 + s^2 - 2 r s,

which is minimized at s* = r, giving nmse* = 1 - r^2. At the Pearson peak this run sits at
s = 0.487 against r = 0.236, so its predictions are roughly TWICE as spread out as its own
correlation justifies. The identity is checked numerically against the logged `nmse` here
rather than assumed, and the gap it predicts is reported as a POST-HOC RESCALE: multiplying
predictions by r/s changes no correlation at all and moves nmse from above 1 to 1 - r^2.

That is a calibration statement, not a capability one. It says the ordering the model
already produces is worth more than its magnitudes, and that reporting Pearson alongside a
`nmse` above 1 is reporting a model that would lose to the mean on a squared-error
scoreboard until it is rescaled.

Run from repo root (needs network; W&B login):
  PYTHONPATH=. python experiments/019-simb-multimodal/scripts/expression_objective_diagnosis.py
"""

from __future__ import annotations

import json
import os
import os.path as osp

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from dotenv import load_dotenv
from matplotlib.ticker import MultipleLocator

from torchcell.timestamp import timestamp
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    savefig_true_size_svg,
)

EXPERIMENT = "019-simb-multimodal"
ENTITY = "zhao-group"

# The two longest expression runs. v9 `hx8pxdic` is the project best and the run the
# review pointed at; v8 `b50f93ju` is the best without the masked-label objective, carried
# so the diagnosis is not read off a single run.
RUNS = [
    ("v9 M_fine (masked)", "torchcell_019_expr_v9", "hx8pxdic"),
    ("v8 V_basis64", "torchcell_019_expr_v8", "b50f93ju"),
]

KEYS = [
    "val/expression/pearson_per_feature",
    "val/expression/mse",
    "val/expression/nmse",
    "val/loss",
    "val/expression/pred_sd_ratio",
    "traineval/expression/pearson_per_feature",
]

# Smoothing window in LOGGED POINTS, not epochs: W&B downsamples history, so a 25-point
# window spans more epochs on a longer run. It is used only to locate turning points, and
# the located epoch is reported as approximate for that reason.
SMOOTH = 25
SAMPLES = 4000


def fetch(api: wandb.Api, project: str, run_id: str) -> tuple[pd.DataFrame, dict]:
    run = api.run(f"{ENTITY}/{project}/{run_id}")
    frames = {}
    for key in KEYS:
        hist = run.history(keys=["epoch", key], samples=SAMPLES)
        if hist.empty or key not in hist:
            continue
        frames[key] = hist.set_index("epoch")[key]
    meta = {
        "run_id": run_id,
        "project": project,
        "tags": list(run.tags),
        "state": run.state,
        "runtime_s": float(run.summary.get("_runtime", float("nan"))),
        "last_epoch": float(run.summary.get("epoch", float("nan"))),
    }
    meta["runtime_h"] = meta["runtime_s"] / 3600.0
    meta["runtime_days"] = meta["runtime_s"] / 86400.0
    if np.isfinite(meta["last_epoch"]) and meta["last_epoch"] > 0:
        meta["seconds_per_epoch"] = meta["runtime_s"] / meta["last_epoch"]
    return pd.DataFrame(frames).sort_index(), meta


def turning_points(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    out = {}
    for col in df.columns:
        smooth = df[col].rolling(SMOOTH, center=True, min_periods=1).mean()
        out[col] = {
            "min": float(smooth.min()),
            "epoch_at_min": float(smooth.idxmin()),
            "max": float(smooth.max()),
            "epoch_at_max": float(smooth.idxmax()),
            "last": float(df[col].dropna().iloc[-1]),
        }
    return out


def calibration(df: pd.DataFrame) -> dict[str, float]:
    """Check nmse = 1 + s^2 - 2 r s at the Pearson peak, and price the rescale."""
    # Metrics are logged on different cadences (validation every epoch, the eval-mode
    # train pass every 25), so a single history row can carry one column and not another.
    # Forward-filling before reading the peak row means each quantity is its most recent
    # logged value at that epoch rather than NaN.
    filled = df.ffill()
    corr = filled["val/expression/pearson_per_feature"].rolling(
        SMOOTH, center=True, min_periods=1
    ).mean()
    peak_epoch = float(corr.idxmax())
    # get_indexer returns a POSITION, so it is read with .iloc; passing it to .loc looks
    # up an epoch label that happens to equal the position and raises.
    pos = int(filled.index.get_indexer([peak_epoch], method="nearest")[0])
    row = filled.iloc[pos]
    r = float(row["val/expression/pearson_per_feature"])
    s = float(row["val/expression/pred_sd_ratio"])
    nmse_logged = float(row["val/expression/nmse"])
    nmse_identity = 1.0 + s**2 - 2.0 * r * s
    return {
        "peak_epoch": peak_epoch,
        "pearson_at_peak": r,
        "pred_sd_ratio_at_peak": s,
        "nmse_logged": nmse_logged,
        "nmse_from_identity": nmse_identity,
        "identity_residual": nmse_logged - nmse_identity,
        "optimal_sd_ratio": r,
        "over_dispersion_factor": s / r if r else float("nan"),
        "nmse_after_rescale": 1.0 - r**2,
        "rescale_multiplier": r / s if s else float("nan"),
    }


def make_figure(curves: dict, out_png: str, out_svg: str) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "axes.linewidth": 0.5,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(48.0)),
        constrained_layout=True,
    )
    label, df = next(iter(curves.items()))

    def smooth(col: str) -> pd.Series:
        return df[col].rolling(SMOOTH, center=True, min_periods=1).mean()

    # (a) the quantile loss turns early, Pearson does not.
    ax = axes[0]
    ax.plot(df.index, smooth("val/expression/pearson_per_feature"),
            color=PLOT_PALETTE[0], linewidth=0.9, label="val Pearson")
    ax.set_xlabel("epoch")
    ax.set_ylabel("val Pearson (per feature)")
    ax.set_ylim(0, 0.3)
    twin = ax.twinx()
    twin.plot(df.index, smooth("val/loss"), color=PLOT_PALETTE[5],
              linewidth=0.9, linestyle=(0, (3, 2)), label="val loss")
    twin.set_ylabel("val loss (quantile)")
    loss_min = float(smooth("val/loss").idxmin())
    ax.axvline(loss_min, color=PLOT_PALETTE[5], linewidth=0.5, linestyle=(0, (1, 2)))
    ax.text(loss_min, 0.29, f" loss min ep {loss_min:.0f}", fontsize=5, va="top")
    ax.set_title("a  Quantile loss turns at ~500", loc="left", fontsize=6, fontweight="bold")

    # (b) squared error does NOT turn early; nmse never returns below 1.
    ax = axes[1]
    ax.plot(df.index, smooth("val/expression/nmse"), color=PLOT_PALETTE[1],
            linewidth=0.9, label="val nmse")
    ax.axhline(1.0, color="black", linewidth=0.5, linestyle=(0, (3, 2)))
    ax.text(df.index.max() * 0.02, 1.002, "nmse = 1 is predict each gene's mean",
            fontsize=5, va="bottom")
    ax.set_xlabel("epoch")
    ax.set_ylabel("val nmse")
    mse_min = float(smooth("val/expression/mse").idxmin())
    ax.axvline(mse_min, color=PLOT_PALETTE[2], linewidth=0.5, linestyle=(0, (1, 2)))
    ax.text(mse_min, ax.get_ylim()[1], f" mse min ep {mse_min:.0f}", fontsize=5,
            va="top", ha="right")
    ax.set_title("b  Squared error turns LATE", loc="left", fontsize=6, fontweight="bold")

    # (c) the calibration gap: where the run sits against s = r.
    ax = axes[2]
    r_series = smooth("val/expression/pearson_per_feature")
    s_series = smooth("val/expression/pred_sd_ratio")
    ax.plot(r_series, s_series, color=PLOT_PALETTE[3], linewidth=0.9)
    lim = np.linspace(0, 0.3, 50)
    ax.plot(lim, lim, color="black", linewidth=0.6, linestyle=(0, (3, 2)))
    ax.text(0.16, 0.17, "s = r (nmse-optimal)", fontsize=5, rotation=32)
    ax.scatter([r_series.max()], [s_series.loc[r_series.idxmax()]], s=16,
               color=PLOT_PALETTE[1], edgecolor="black", linewidth=0.4, zorder=3)
    ax.set_xlabel("val Pearson r")
    ax.set_ylabel("prediction SD ratio s")
    ax.set_xlim(0, 0.3)
    ax.set_ylim(0, 0.6)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.set_title("c  Predictions are over-dispersed", loc="left", fontsize=6,
                 fontweight="bold")

    for axis in axes:
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
    fig.savefig(out_png, dpi=300)
    savefig_true_size_svg(fig, out_svg)
    plt.close(fig)


def main() -> None:
    load_dotenv()
    experiment_root = os.environ["EXPERIMENT_ROOT"]
    images_dir = os.environ["ASSET_IMAGES_DIR"]
    results_dir = osp.join(experiment_root, EXPERIMENT, "results")
    out_dir = osp.join(images_dir, EXPERIMENT)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    api = wandb.Api(timeout=120)
    curves: dict[str, pd.DataFrame] = {}
    payload: dict[str, object] = {"smoothing_points": SMOOTH, "runs": {}}
    for label, project, run_id in RUNS:
        df, meta = fetch(api, project, run_id)
        curves[label] = df
        df.to_csv(osp.join(results_dir, f"expression_curve_{run_id}.csv"))
        payload["runs"][label] = {
            **meta,
            "turning_points": turning_points(df),
            "calibration": calibration(df),
        }

    png = osp.join(out_dir, "expression_objective_diagnosis.png")
    svg = osp.join(out_dir, "expression_objective_diagnosis.svg")
    make_figure(curves, png, svg)
    payload["figure"] = {"png": png, "svg": svg, "written_at": timestamp()}
    with open(osp.join(results_dir, "expression_objective_diagnosis.json"), "w") as fh:
        json.dump(payload, fh, indent=2)
    print(json.dumps(payload["runs"], indent=2)[:4000])
    print(f"-> {svg}")


if __name__ == "__main__":
    main()
