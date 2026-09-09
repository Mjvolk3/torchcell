# experiments/019-simb-multimodal/scripts/mech_round_readout.py
# [[experiments.019-simb-multimodal.scripts.mech_round_readout]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/mech_round_readout
"""Read out the mechanism round: pair term and per-gene readout, two seeds, project v9.

THE ROUND (igb_expr_wave5.slurm stage `mech`, arms in gh_expr_008_arm.sh). Four arms on the
v9 incumbent (quantile head, masked-label objective): `R_ref` (nothing added), `R_basis64`
(rank-64 response basis, the pair term), `R_pergene` (per-gene output weight and bias on
the readout), `R_pergene_basis64` (both). Seed 0 ran on cabbi Ada cards (job 2369697, two
runs per card), seed 1 on one A40 node (job 2371531 tasks 2 and 3, two runs per card), so
GPU type is a seed-level block and no pair straddles the two. The seed-1 tasks ended in the
cgroup out-of-memory handler on 2026-09-07: in each task the `basis64` run was the one
killed (epochs 7,020 and 6,590) and its sibling finished its 8,500 epochs.

THE SCORE. `roll_max` (imported from pull_round_leaderboards.py: max of a centered
ROLL_WINDOW-epoch rolling mean of `val/expression/pearson_per_feature`) over epochs <= B for
a set of budgets B, from the FULL per-epoch history. The matched budget is the smallest
final epoch across the eight runs, computed, not assumed. Contrasts are PAIRED within seed
against `R_ref`, which is the in-round reference the round was designed around, and every
arm is also placed against the eight long-budget v9 runs of short_budget_spread.json at
the nearest tabulated budget. Those eight are the arms of the v9 mask-schedule round, not
replicates (found 2026-09-08), so that band is an arm spread and only bounds the
replicate spread from above; the paired contrasts within seed do not depend on it.

WHAT IT CANNOT SAY. Two seeds resolve a paired gap of about 0.06 (the design note's own
figure); anything smaller is "not resolved", not "null". Runs shorter than MIN_EPOCHS are
the crashed first attempts of the same arms (the DataLoader fd-exhaustion deaths) and are
excluded by that rule alone.

Outputs: results/mech_round_readout.{csv,json} and a full-width two-panel figure.
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
NMSE = "val/expression/nmse"
SD_RATIO = "val/expression/pred_sd_ratio"
CONFIG_TAG = "cgt_expr_v9_mask"
ARMS = ("R_ref", "R_basis64", "R_pergene", "R_pergene_basis64")

RESULTS = experiment_results_dir("019-simb-multimodal", __file__)
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
IMAGE_DIR = osp.join(ASSET_IMAGES_DIR, "019-simb-multimodal")
SPREAD_JSON = osp.join(RESULTS, "short_budget_spread.json")

FULL_HISTORY_SAMPLES = 50_000
MIN_EPOCHS = 1000
# Fixed budgets at which every arm is also scored, besides the computed matched budget.
BUDGETS = (1000, 2000, 4000, 6000)


def roll_mean(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values).rolling(window, center=True, min_periods=1).mean().to_numpy()


def roll_max_within(h: pd.DataFrame, budget: int) -> tuple[float, int]:
    roll = roll_mean(h[METRIC].to_numpy(), ROLL_WINDOW)
    ep = h.epoch.to_numpy()
    if ep.min() > budget:
        return float("nan"), -1
    j = int(np.argmax(np.where(ep <= budget, roll, -np.inf)))
    return float(roll[j]), int(ep[j])


def fetch(api: wandb.Api) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    # Selection is by ARM tag plus the round's config tag, NOT by `stage-mech`: the arm
    # script's `case` had an earlier wave-4b `R_ref` branch that shadowed the wave-5 one, so
    # every R_ref run of this round carries `stage-wave4b`. The config tag is what all eight
    # share. Where an arm and seed has several runs above MIN_EPOCHS, the LONGEST is kept:
    # the seed-1 half was cancelled at epoch ~1,530 on cabbi and restarted from scratch on
    # the A40 node, and the abandoned first attempt is not a replicate.
    runs = [r for r in api.runs(f"{ENTITY}/{PROJECT}", per_page=100)
            if CONFIG_TAG in r.tags and any(t in ARMS for t in r.tags)]
    print(f"{len(runs)} runs tagged {CONFIG_TAG} with a mechanism arm in {PROJECT}")
    best: dict[tuple[str, int], wandb.apis.public.Run] = {}
    for run in runs:
        arm = [t for t in run.tags if t in ARMS]
        assert len(arm) == 1, run.tags
        final_epoch = run.summary.get("epoch")
        key = (arm[0], int(run.config["seed"]))
        if final_epoch is None or final_epoch < MIN_EPOCHS:
            print(f"  drop {run.id} {key} at epoch {final_epoch}: shorter than {MIN_EPOCHS}")
            continue
        if key in best:
            keep, drop = ((run, best[key]) if final_epoch > best[key].summary["epoch"]
                          else (best[key], run))
            print(f"  drop {drop.id} {key} at epoch {drop.summary['epoch']}: superseded by "
                  f"{keep.id} at epoch {keep.summary['epoch']}")
            best[key] = keep
        else:
            best[key] = run
    rows, curves = [], {}
    for (arm_name, _), run in sorted(best.items()):
        arm = [arm_name]
        keys = ["epoch", METRIC, NMSE, SD_RATIO]
        h = run.history(keys=keys, samples=FULL_HISTORY_SAMPLES)
        h = h.dropna(subset=["epoch", METRIC]).sort_values("epoch")
        h = h.drop_duplicates("epoch", keep="last").reset_index(drop=True)
        rows.append({
            "run_id": run.id, "arm": arm[0], "seed": int(run.config["seed"]),
            "state": run.state, "n_epochs": int(h.epoch.max()),
            "rows_short_by": int(h.epoch.max() + 1 - len(h)),
            "start_epoch": int(h.epoch.min()),
            "pearson_last": float(h[METRIC].iloc[-1]),
        })
        curves[run.id] = h
    t = pd.DataFrame(rows).sort_values(["arm", "seed"]).reset_index(drop=True)
    assert t.groupby(["arm", "seed"]).size().max() == 1, t
    return t, curves


def incumbent_band(budget: int) -> dict[str, float]:
    with open(SPREAD_JSON) as fh:
        d = json.load(fh)
    r = min(d["by_budget"], key=lambda r: abs(r["budget_epochs"] - budget))
    return {"budget_epochs": r["budget_epochs"], "mean": r["mean"], "sd": r["sd"], "n": r["n"]}


def figure(t: pd.DataFrame, curves: dict[str, pd.DataFrame], matched: int,
           inc: dict[str, float]) -> str:
    plt.rcParams.update({"font.family": "Arial", "font.size": 6, "svg.fonttype": "none",
                         "axes.linewidth": 0.5})
    fig, axes = plt.subplots(1, 2, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(60)),
                             gridspec_kw={"width_ratios": [1.5, 1.0], "wspace": 0.3})
    fig.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.16)
    color = {a: PLOT_PALETTE[i] for i, a in enumerate(ARMS)}
    style = {0: "-", 1: "--"}

    ax = axes[0]
    for _, r in t.iterrows():
        h = curves[r.run_id]
        ax.plot(h.epoch, roll_mean(h[METRIC].to_numpy(), ROLL_WINDOW), color=color[r.arm],
                ls=style[r.seed], lw=0.6, label=f"{r.arm} seed {r.seed}")
    ax.axvline(matched, color="black", lw=0.5, ls=":")
    ax.set_xscale("log")
    ax.set_xlim(10, 10_000)
    ax.set_ylim(-0.02, 0.25)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.set_xlabel("epoch")
    ax.set_ylabel(f"val Pearson, {ROLL_WINDOW}-epoch rolling mean")
    ax.grid(lw=0.3, alpha=0.35)
    ax.legend(frameon=False, loc="upper left", fontsize=5, ncol=2, handlelength=1.6)
    ax.set_title("mechanism round, both seeds; dotted = matched budget", fontsize=6, pad=3)
    ax.text(-0.1, 1.04, "a", transform=ax.transAxes, fontsize=8, fontweight="bold")

    ax = axes[1]
    x = {a: i for i, a in enumerate(ARMS)}
    marker = {0: "o", 1: "s"}
    for _, r in t.iterrows():
        ax.scatter(x[r.arm], r[f"roll_max_le_{matched}"], s=16, marker=marker[r.seed],
                   facecolor=color[r.arm], edgecolor="black", lw=0.4, zorder=3)
    for s in (0, 1):
        sub = t[t.seed == s].set_index("arm")
        ax.plot([x[a] for a in ARMS if a in sub.index],
                [sub.loc[a, f"roll_max_le_{matched}"] for a in ARMS if a in sub.index],
                color="black", lw=0.4, ls=style[s], zorder=1)
    ax.axhspan(inc["mean"] - inc["sd"], inc["mean"] + inc["sd"], color=PLOT_PALETTE[3],
               alpha=0.25, lw=0, zorder=0)
    ax.axhline(inc["mean"], color=PLOT_PALETTE[3], lw=0.8, zorder=0)
    ax.text(-0.4, inc["mean"] - inc["sd"] - 0.003,
            f"v9 long-budget arms, n={inc['n']} at {inc['budget_epochs']:,} ep",
            fontsize=5, color=PLOT_PALETTE[9], ha="left", va="top")
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels(ARMS, rotation=20, fontsize=5)
    ax.set_xlim(-0.5, len(ARMS) - 0.5)
    ax.set_ylim(0.1, 0.25)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.01))
    ax.tick_params(axis="y", which="minor", length=0)
    ax.grid(axis="y", which="both", lw=0.3, alpha=0.35)
    ax.set_ylabel(f"roll_max, epochs <= {matched:,}")
    ax.scatter([], [], s=16, marker="o", facecolor="white", edgecolor="black", label="seed 0")
    ax.scatter([], [], s=16, marker="s", facecolor="white", edgecolor="black", label="seed 1")
    ax.legend(frameon=False, loc="lower right", fontsize=5)
    ax.set_title("matched-budget score per arm and seed", fontsize=6, pad=3)
    ax.text(-0.2, 1.04, "b", transform=ax.transAxes, fontsize=8, fontweight="bold")

    for a in axes:
        for s in a.spines.values():
            s.set_visible(True)
    os.makedirs(IMAGE_DIR, exist_ok=True)
    # Stable name, no timestamp: notes-tex's `make plots` converts figures by name.
    stem = osp.join(IMAGE_DIR, "mech_round_readout")
    fig.savefig(stem + ".png", dpi=300)
    savefig_true_size_svg(fig, stem + ".svg")
    plt.close(fig)
    return stem


def main() -> None:
    api = wandb.Api(timeout=120)
    t, curves = fetch(api)
    matched = int(t.n_epochs.min())
    budgets = sorted(set(BUDGETS) | {matched})
    print(f"matched budget = {matched}; final epochs "
          f"{t.set_index(['arm', 'seed']).n_epochs.to_dict()}")
    for b in budgets:
        vals = [roll_max_within(curves[rid], b) for rid in t.run_id]
        t[f"roll_max_le_{b}"] = [v for v, _ in vals]
        t[f"epoch_at_le_{b}"] = [e for _, e in vals]
    t["roll_max_full"] = [roll_max_within(curves[rid], 10**9)[0] for rid in t.run_id]
    t["nmse_at_matched_peak"] = [
        float(curves[rid].set_index("epoch")[NMSE].get(e, np.nan))
        for rid, e in zip(t.run_id, t[f"epoch_at_le_{matched}"])
    ]
    csv_path = osp.join(RESULTS, "mech_round_readout.csv")
    t.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")
    with pd.option_context("display.width", 220):
        print(t.drop(columns=["rows_short_by"]).to_string(
            index=False, float_format=lambda x: f"{x:.4f}"))

    # Paired contrasts against R_ref within seed, at every budget both arms reached.
    contrasts: dict[str, dict[str, object]] = {}
    for b in budgets:
        col = f"roll_max_le_{b}"
        wide = t.pivot(index="seed", columns="arm", values=col)
        if "R_ref" not in wide:
            continue
        contrasts[str(b)] = {}
        for arm in ARMS[1:]:
            if arm not in wide:
                continue
            diff = (wide[arm] - wide["R_ref"]).dropna()
            contrasts[str(b)][arm] = {
                "n_pairs": int(len(diff)),
                "diffs": {int(s): float(v) for s, v in diff.items()},
                "mean_diff": float(diff.mean()) if len(diff) else float("nan"),
            }
    print("\n=== paired contrasts, arm minus R_ref within seed ===")
    for b, arms in contrasts.items():
        for arm, c in arms.items():
            print(f"  epochs <= {int(b):>5}  {arm:18s} n={c['n_pairs']}  "
                  f"mean {c['mean_diff']:+.4f}  " + "  ".join(
                      f"seed{s} {v:+.4f}" for s, v in c["diffs"].items()))

    inc = incumbent_band(matched)
    print(f"\nincumbent band at {inc['budget_epochs']} epochs: {inc['mean']:.4f} +/- "
          f"{inc['sd']:.4f} (n={inc['n']})")
    stem = figure(t, curves, matched, inc)
    print(f"figure: {stem}.svg")

    with open(osp.join(RESULTS, "mech_round_readout.json"), "w") as fh:
        json.dump({
            "generated_by": "experiments/019-simb-multimodal/scripts/mech_round_readout.py",
            "project": PROJECT, "config_tag": CONFIG_TAG, "metric": METRIC,
            "roll_window": ROLL_WINDOW, "history_samples": FULL_HISTORY_SAMPLES,
            "min_epochs": MIN_EPOCHS, "matched_budget_epochs": matched, "budgets": budgets,
            "runs": t.to_dict(orient="records"), "paired_contrasts": contrasts,
            "incumbent_band": inc, "figure": stem + ".svg",
        }, fh, indent=2, default=float)


if __name__ == "__main__":
    main()
