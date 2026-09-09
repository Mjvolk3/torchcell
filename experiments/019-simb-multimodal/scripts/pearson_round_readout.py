# experiments/019-simb-multimodal/scripts/pearson_round_readout.py
# [[experiments.019-simb-multimodal.scripts.pearson_round_readout]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/pearson_round_readout
"""Read out the metric-aligned objective round: `pearson`, `pearson_mse`, `pearson` at
batch 64, project v9, IGB cabbi jobs 2378262 / 2378267 / 2378268.

THE QUESTION. The v9 proper-scoring heads bottom out on `val/loss` at a few hundred epochs
while validation Pearson keeps rising (loss_min_vs_pearson_peak.py). These arms train on
1 - mean per-feature Pearson over the hidden genes (torchcell/losses/distributional.py,
`pearson_loss`), so objective and metric cannot disagree on train. What happens on
validation, and does the objective stay alive at all?

WHAT IS READ, per run, from the FULL per-epoch history:
  - `val/expression/pearson_per_feature` and its `roll_max` (pull_round_leaderboards.py
    rule) over the epochs reached so far, plus the epoch it occurs at;
  - the COLLAPSE epoch: the first epoch after which the metric is numerically zero for the
    rest of the history (|r| < COLLAPSE_EPS). The Pearson objective drops constant columns
    (they carry no gradient), so a network whose every output column is constant scores
    exactly 0 on the metric while its loss reads only the few still-varying columns;
  - `val/expression/pred_sd_ratio` (predicted over true spread), because pure Pearson is
    scale-free and a scale blow-up is invisible in the loss;
  - `val/loss`, comparable only WITHIN an arm (pearson_mse adds a masked MSE at weight 1).
The eight long-budget quantile-head runs of short_budget_spread.json are the reference
band, drawn at every budget the file holds. They are the eight arms of the v9
mask-schedule round, not replicates (found 2026-09-08): the band is an arm spread plus
nondeterminism and bounds the replicate spread from above.

THE RUNS ARE IN FLIGHT when this is first run (five-day wall from 2026-09-06 19:45 CT), so
every number is partial and the JSON records the epoch reached. Re-run after the wall.

Outputs: results/pearson_round_readout.{csv,json} and a full-width three-panel figure.
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
from matplotlib.ticker import MultipleLocator

load_dotenv()

from torchcell.timestamp import timestamp  # noqa: E402
from torchcell.utils import (  # noqa: E402
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    experiment_results_dir,
    mm_to_in,
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
NMSE = "val/expression/nmse"
SD_RATIO = "val/expression/pred_sd_ratio"
STAGE_TAGS = ("stage-pearson", "stage-pearson_b64")
ARMS = ("Q_pearson", "Q_pearson_mse", "Q_pearson_b64")

RESULTS = experiment_results_dir("019-simb-multimodal", __file__)
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
IMAGE_DIR = osp.join(ASSET_IMAGES_DIR, "019-simb-multimodal")
SPREAD_JSON = osp.join(RESULTS, "short_budget_spread.json")

FULL_HISTORY_SAMPLES = 50_000
# The canary (job 2375312) was a one-batch fast_dev_run of the same arms; this excludes it.
MIN_EPOCHS = 100
COLLAPSE_EPS = 1e-6
# A smoothed validation Pearson under this is the chance band: the per-gene mean baseline
# scores 0.0000 and the healthiest early curve in this round sits above 0.06 by epoch 10.
FLOOR = 0.02


def roll_mean(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values).rolling(window, center=True, min_periods=1).mean().to_numpy()


def collapse_epoch(h: pd.DataFrame, tol: float, smoothed: bool = False) -> int | None:
    """First epoch from which |Pearson| < tol holds to the end, or None.

    `smoothed` tests the ROLL_WINDOW rolling mean instead of the raw value: a collapsed run's
    raw metric keeps flickering to 1e-3 for thousands of epochs, so the raw test dates the
    collapse thousands of epochs after the curve actually fell to the floor.
    """
    v = h[METRIC].to_numpy()
    if smoothed:
        v = roll_mean(v, ROLL_WINDOW)
    zero = np.abs(v) < tol
    if not zero[-1]:
        return None
    alive = np.nonzero(~zero)[0]
    return int(h.epoch.iloc[alive[-1] + 1]) if len(alive) else int(h.epoch.iloc[0])


def fetch(api: wandb.Api) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    runs = [r for r in api.runs(f"{ENTITY}/{PROJECT}", per_page=100)
            if any(t in r.tags for t in STAGE_TAGS)]
    print(f"{len(runs)} runs tagged {STAGE_TAGS} in {PROJECT}")
    rows, curves = [], {}
    for run in runs:
        arm = [t for t in run.tags if t in ARMS]
        assert len(arm) == 1, run.tags
        final_epoch = run.summary.get("epoch")
        if final_epoch is None or final_epoch < MIN_EPOCHS:
            print(f"  drop {run.id} {arm[0]} at epoch {final_epoch}: canary or dead launch")
            continue
        h = run.history(keys=["epoch", METRIC, LOSS, NMSE, SD_RATIO],
                        samples=FULL_HISTORY_SAMPLES)
        # W&B hands back a non-finite value as the string "NaN" in an otherwise float column
        # (seen on val/loss of the collapsed runs), which breaks every reduction below.
        for col in (METRIC, LOSS, NMSE, SD_RATIO):
            h[col] = pd.to_numeric(h[col], errors="coerce")
        h = h.dropna(subset=["epoch", METRIC]).sort_values("epoch")
        h = h.drop_duplicates("epoch", keep="last").reset_index(drop=True)
        roll = roll_mean(h[METRIC].to_numpy(), ROLL_WINDOW)
        j = int(np.argmax(roll))
        rows.append({
            "run_id": run.id, "arm": arm[0], "seed": int(run.config["seed"]),
            "dist": run.config.get("dist"), "batch_size": run.config.get("batch_size"),
            "state": run.state, "n_epochs": int(h.epoch.max()),
            "rows_short_by": int(h.epoch.max() + 1 - len(h)),
            "roll_max": float(roll[j]), "epoch_at_roll_max": int(h.epoch.iloc[j]),
            "pearson_last": float(h[METRIC].iloc[-1]),
            # The strict test is the leaderboard's. `floor_from` is when the smoothed curve
            # fell under FLOOR for good, which is the epoch a reader would call the collapse.
            "collapse_epoch": collapse_epoch(h, COLLAPSE_EPS),
            "floor_from": collapse_epoch(h, FLOOR, smoothed=True),
            "loss_last": float(h[LOSS].iloc[-1]),
            "loss_min": float(h[LOSS].min()),
            "loss_min_epoch": int(h.epoch.iloc[int(h[LOSS].idxmin())]),
            "nmse_at_roll_max": float(h[NMSE].iloc[j]),
            "nmse_last": float(h[NMSE].iloc[-1]),
            "sd_ratio_at_roll_max": float(h[SD_RATIO].iloc[j]),
            "sd_ratio_last": float(h[SD_RATIO].iloc[-1]),
        })
        curves[run.id] = h
    t = pd.DataFrame(rows).sort_values(["arm", "seed"]).reset_index(drop=True)
    return t, curves


def incumbent_curve() -> pd.DataFrame:
    with open(SPREAD_JSON) as fh:
        d = json.load(fh)
    return pd.DataFrame(d["by_budget"]).sort_values("budget_epochs")


def figure(t: pd.DataFrame, curves: dict[str, pd.DataFrame], inc: pd.DataFrame) -> str:
    plt.rcParams.update({"font.family": "Arial", "font.size": 6, "svg.fonttype": "none",
                         "axes.linewidth": 0.5})
    fig, axes = plt.subplots(1, 3, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(58)))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.17, wspace=0.38)
    color = {a: PLOT_PALETTE[i] for i, a in enumerate(ARMS)}
    style = {0: "-", 1: "--", 2: ":"}

    ax = axes[0]
    ax.fill_between(inc.budget_epochs, inc["mean"] - inc["sd"], inc["mean"] + inc["sd"],
                    color=PLOT_PALETTE[3], alpha=0.25, lw=0, zorder=0)
    ax.plot(inc.budget_epochs, inc["mean"], color=PLOT_PALETTE[3], lw=0.8, zorder=0,
            label="v9 long-budget quantile arms, n=8")
    for _, r in t.iterrows():
        h = curves[r.run_id]
        ax.plot(h.epoch, roll_mean(h[METRIC].to_numpy(), ROLL_WINDOW), color=color[r.arm],
                ls=style[r.seed], lw=0.6, label=f"{r.arm} seed {r.seed}")
    ax.set_xscale("log")
    ax.set_xlim(10, 10_000)
    ax.set_ylim(-0.02, 0.32)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.set_xlabel("epoch")
    ax.set_ylabel(f"val Pearson, {ROLL_WINDOW}-epoch rolling mean")
    ax.grid(lw=0.3, alpha=0.35)
    ax.legend(frameon=False, loc="upper left", fontsize=4.5, ncol=2, handlelength=1.6,
              columnspacing=0.8, labelspacing=0.3)
    ax.set_title("metric-aligned arms vs the incumbent band", fontsize=6, pad=3)
    ax.text(-0.2, 1.04, "a", transform=ax.transAxes, fontsize=8, fontweight="bold")

    ax = axes[1]
    for _, r in t.iterrows():
        h = curves[r.run_id]
        ax.plot(h.epoch, h[SD_RATIO], color=color[r.arm], ls=style[r.seed], lw=0.6)
    ax.axhline(1.0, color="black", lw=0.5, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(10, 10_000)
    ax.set_xlabel("epoch")
    ax.set_ylabel("predicted / true spread (pred_sd_ratio)")
    ax.grid(lw=0.3, alpha=0.35, which="both")
    ax.set_title("scale, which the Pearson loss does not see", fontsize=6, pad=3)
    ax.text(-0.2, 1.04, "b", transform=ax.transAxes, fontsize=8, fontweight="bold")

    ax = axes[2]
    for _, r in t.iterrows():
        h = curves[r.run_id]
        ax.plot(h.epoch, h[LOSS], color=color[r.arm], ls=style[r.seed], lw=0.6)
    ax.set_xscale("log")
    ax.set_xlim(10, 10_000)
    ax.set_xlabel("epoch")
    ax.set_ylabel("val/loss (pearson_mse carries an MSE term)")
    ax.grid(lw=0.3, alpha=0.35)
    ax.set_title("the objective on validation", fontsize=6, pad=3)
    ax.text(-0.2, 1.04, "c", transform=ax.transAxes, fontsize=8, fontweight="bold")

    for a in axes:
        for s in a.spines.values():
            s.set_visible(True)
    os.makedirs(IMAGE_DIR, exist_ok=True)
    # Stable name, no timestamp: notes-tex's `make plots` converts figures by name. The
    # JSON's `read_at` carries the timestamp instead, since these runs are in flight.
    stem = osp.join(IMAGE_DIR, "pearson_round_readout")
    fig.savefig(stem + ".png", dpi=300)
    savefig_true_size_svg(fig, stem + ".svg")
    plt.close(fig)
    return stem


def main() -> None:
    api = wandb.Api(timeout=120)
    t, curves = fetch(api)
    csv_path = osp.join(RESULTS, "pearson_round_readout.csv")
    t.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")
    with pd.option_context("display.width", 240):
        print(t.drop(columns=["rows_short_by", "dist"]).to_string(
            index=False, float_format=lambda x: f"{x:.4f}"))

    inc = incumbent_curve()
    print("\n=== each run against the incumbent band at the nearest tabulated budget ===")
    for _, r in t.iterrows():
        b = inc.iloc[int(np.argmin(np.abs(inc.budget_epochs - r.n_epochs)))]
        print(f"  {r.run_id} {r.arm:15s} seed{r.seed} ep {r.n_epochs:>5} roll_max "
              f"{r.roll_max:.4f}@{r.epoch_at_roll_max:<5} incumbent@{int(b.budget_epochs)} "
              f"{b['mean']:.4f}+/-{b['sd']:.4f}"
              + (f"  COLLAPSED, on the floor from epoch {int(r.floor_from)}"
                 if pd.notna(r.floor_from) else ""))
    stem = figure(t, curves, inc)
    print(f"figure: {stem}.svg")
    with open(osp.join(RESULTS, "pearson_round_readout.json"), "w") as fh:
        json.dump({
            "generated_by": "experiments/019-simb-multimodal/scripts/pearson_round_readout.py",
            "project": PROJECT, "stage_tags": STAGE_TAGS, "metric": METRIC,
            "roll_window": ROLL_WINDOW, "history_samples": FULL_HISTORY_SAMPLES,
            "collapse_eps": COLLAPSE_EPS, "read_at": timestamp(),
            "runs": t.to_dict(orient="records"), "figure": stem + ".svg",
        }, fh, indent=2, default=float)


if __name__ == "__main__":
    main()
