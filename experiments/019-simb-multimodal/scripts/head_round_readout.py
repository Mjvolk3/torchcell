# experiments/019-simb-multimodal/scripts/head_round_readout.py
# [[experiments.019-simb-multimodal.scripts.head_round_readout]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/head_round_readout
"""Read out the v12 readout (head) round: eight per-gene readouts, four seeds, project v12.

THE ROUND (igb_expr_wave5.slurm stage `head`, arms in gh_expr_008_arm.sh, config
cgt_expr_v12_head.yaml = v11 E_full embeddings [fudt_upstream, calm, prot_T5_all,
fudt_downstream] under the pinball objective, 1,400 epochs). Eight arms on that trunk:
`H_ref` (shared two-layer MLP readout), `H_linear` (linear readout), `H_pergene` (GEARS
gene-specific row w_u, b_u), `H_gears` (row + cross-gene MLP, rank 64), `H_basis64` (scGPT
MVC form, rank-64 response basis), `H_pergene_basis64`, `H_concat` (State SE decoder form,
cell vector concatenated to the gene token), `H_state` (State ST form, per-gene affine row
on the strain context, zero-gated). Seed-major packing, FOUR runs per card: IGB array
tasks 2392379_0 and _1 (cabbi, seed 0; slurm job ids 2392379 and 2392380) and 2392381_2
to _7 (gpu A40, seeds 1 to 3; job ids 2392381 to 2392386). Card type is therefore a
seed-level block, and no paired contrast straddles the two.

THE SCORE. `roll_max` (imported from pull_round_leaderboards.py: max of a centered
ROLL_WINDOW-epoch rolling mean of `val/expression/pearson_per_feature`) over epochs <= B,
from the FULL per-epoch history. The matched budget is the smallest final epoch across the
runs present, computed, not assumed. Contrasts are PAIRED within seed against `H_ref`, the
in-round reference; every arm is also placed against the eight long-budget v9 runs of
short_budget_spread.json at the nearest tabulated budget (an arm spread, not a replicate
spread, found 2026-09-08; it bounds the replicate spread from above).

WHAT IT CANNOT SAY. Four seeds resolve a paired gap of roughly 0.03 (the v11 design note's
figure for three seeds is 0.03); anything smaller is "not resolved", not "null". Runs
without an `epoch` in their summary are the GilaHyper smoke tests that were synced into the
project before launch (fast_dev_run, `failed` or `finished` at epoch None) and are excluded
by MIN_EPOCHS alone. W&B `state` is not evidence that a run ended (offline runs are stamped
`finished` by every sync); the epoch reached is the only budget statement made.

Outputs: results/head_round_readout.{csv,json} and a full-width three-panel figure.
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
PROJECT = "torchcell_019_expr_v12"
METRIC = "val/expression/pearson_per_feature"
NMSE = "val/expression/nmse"
SD_RATIO = "val/expression/pred_sd_ratio"
CONFIG_TAG = "cgt_expr_v12_head"
REF = "H_ref"
ARMS = (
    "H_ref",
    "H_linear",
    "H_pergene",
    "H_gears",
    "H_basis64",
    "H_pergene_basis64",
    "H_concat",
    "H_state",
)
SEEDS = (0, 1, 2, 3)

RESULTS = experiment_results_dir("019-simb-multimodal", __file__)
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
IMAGE_DIR = osp.join(ASSET_IMAGES_DIR, "019-simb-multimodal")
SPREAD_JSON = osp.join(RESULTS, "short_budget_spread.json")

FULL_HISTORY_SAMPLES = 50_000
MIN_EPOCHS = 200
BUDGETS = (500, 1000, 1400)


def roll_mean(values: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(values).rolling(window, center=True, min_periods=1).mean().to_numpy()
    )


def roll_max_within(h: pd.DataFrame, budget: int) -> tuple[float, int]:
    roll = roll_mean(h[METRIC].to_numpy(), ROLL_WINDOW)
    ep = h.epoch.to_numpy()
    if ep.min() > budget:
        return float("nan"), -1
    j = int(np.argmax(np.where(ep <= budget, roll, -np.inf)))
    return float(roll[j]), int(ep[j])


def fetch(api: wandb.Api) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    runs = [
        r
        for r in api.runs(f"{ENTITY}/{PROJECT}", per_page=100)
        if CONFIG_TAG in r.tags and any(t in ARMS for t in r.tags)
    ]
    print(f"{len(runs)} runs tagged {CONFIG_TAG} with a head arm in {PROJECT}")
    best: dict[tuple[str, int], wandb.apis.public.Run] = {}
    for run in runs:
        arm = [t for t in run.tags if t in ARMS]
        assert len(arm) == 1, run.tags
        final_epoch = run.summary.get("epoch")
        key = (arm[0], int(run.config["seed"]))
        if final_epoch is None or final_epoch < MIN_EPOCHS:
            print(
                f"  drop {run.id} {key} at epoch {final_epoch}: shorter than {MIN_EPOCHS}"
            )
            continue
        if key in best and best[key].summary["epoch"] >= final_epoch:
            print(
                f"  drop {run.id} {key} at epoch {final_epoch}: superseded by {best[key].id}"
            )
            continue
        if key in best:
            print(
                f"  drop {best[key].id} {key}: superseded by {run.id} at epoch {final_epoch}"
            )
        best[key] = run
    rows, curves = [], {}
    for (arm_name, _), run in sorted(best.items()):
        keys = ["epoch", METRIC, NMSE, SD_RATIO]
        h = run.history(keys=keys, samples=FULL_HISTORY_SAMPLES)
        h = h.dropna(subset=["epoch", METRIC]).sort_values("epoch")
        h = h.drop_duplicates("epoch", keep="last").reset_index(drop=True)
        rows.append(
            {
                "run_id": run.id,
                "arm": arm_name,
                "seed": int(run.config["seed"]),
                "state": run.state,
                "n_epochs": int(h.epoch.max()),
                "rows_short_by": int(h.epoch.max() + 1 - len(h)),
                "pearson_last": float(h[METRIC].iloc[-1]),
                "sd_ratio_last": float(h[SD_RATIO].dropna().iloc[-1])
                if h[SD_RATIO].notna().any()
                else float("nan"),
                "url": f"https://wandb.ai/{ENTITY}/{PROJECT}/runs/{run.id}",
            }
        )
        curves[run.id] = h
    t = pd.DataFrame(rows).sort_values(["arm", "seed"]).reset_index(drop=True)
    assert t.groupby(["arm", "seed"]).size().max() == 1, t
    return t, curves


def incumbent_band(budget: int) -> dict[str, float]:
    with open(SPREAD_JSON) as fh:
        d = json.load(fh)
    r = min(d["by_budget"], key=lambda r: abs(r["budget_epochs"] - budget))
    return {
        "budget_epochs": r["budget_epochs"],
        "mean": r["mean"],
        "sd": r["sd"],
        "n": r["n"],
    }


def figure(
    t: pd.DataFrame,
    curves: dict[str, pd.DataFrame],
    matched: int,
    inc: dict[str, float],
    contrasts: dict[str, dict[str, object]],
) -> str:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "svg.fonttype": "none",
            "axes.linewidth": 0.5,
            "legend.frameon": True,
            "legend.fancybox": False,
            "legend.framealpha": 1.0,
            "legend.edgecolor": "black",
            "legend.facecolor": "white",
            "patch.linewidth": 0.5,
        }
    )
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(62)),
        gridspec_kw={"width_ratios": [1.4, 1.0, 1.0], "wspace": 0.35},
    )
    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.2)
    color = {a: PLOT_PALETTE[i] for i, a in enumerate(ARMS)}
    style = {0: "-", 1: "--", 2: ":", 3: "-."}
    marker = {0: "o", 1: "s", 2: "^", 3: "D"}
    present = [a for a in ARMS if a in set(t.arm)]

    ax = axes[0]
    for arm in present:
        for _, r in t[t.arm == arm].iterrows():
            h = curves[r.run_id]
            ax.plot(
                h.epoch,
                roll_mean(h[METRIC].to_numpy(), ROLL_WINDOW),
                color=color[arm],
                ls=style[r.seed],
                lw=0.6,
                label=arm if r.seed == t[t.arm == arm].seed.min() else None,
            )
    ax.axvline(matched, color="black", lw=0.5, ls=":")
    ax.set_xscale("log")
    ax.set_xlim(10, 2_000)
    ax.set_ylim(-0.02, 0.33)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.set_xlabel("epoch (line style = seed)")
    ax.set_ylabel(f"val Pearson, {ROLL_WINDOW}-epoch rolling mean")
    ax.grid(lw=0.3, alpha=0.35)
    ax.legend(loc="upper left", fontsize=5, ncol=2, handlelength=1.6, borderpad=0.3)
    ax.set_title("head round; dotted = matched budget", fontsize=6, pad=3)
    panel_label(ax, "a")

    ax = axes[1]
    x = {a: i for i, a in enumerate(present)}
    col = f"roll_max_le_{matched}"
    for _, r in t.iterrows():
        ax.scatter(
            x[r.arm],
            r[col],
            s=16,
            marker=marker[r.seed],
            facecolor=color[r.arm],
            edgecolor="black",
            lw=0.4,
            zorder=3,
        )
    for s in sorted(set(t.seed)):
        sub = t[t.seed == s].set_index("arm")
        xs = [x[a] for a in present if a in sub.index]
        ax.plot(
            xs,
            [sub.loc[a, col] for a in present if a in sub.index],
            color="black",
            lw=0.4,
            ls=style[s],
            zorder=1,
        )
    ax.axhspan(
        inc["mean"] - inc["sd"],
        inc["mean"] + inc["sd"],
        color=PLOT_PALETTE_FILL[3],
        lw=0,
        zorder=0,
        label=f"v9 arms, n={inc['n']}, {inc['budget_epochs']:,} ep",
    )
    ax.axhline(inc["mean"], color=PLOT_PALETTE[3], lw=0.8, zorder=0)
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(
        [a.replace("H_", "") for a in present], rotation=30, fontsize=5, ha="right"
    )
    ax.set_xlim(-0.5, len(present) - 0.5)
    ax.set_ylim(0.04, 0.29)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.01))
    ax.tick_params(axis="y", which="minor", length=0)
    ax.grid(axis="y", which="both", lw=0.3, alpha=0.35)
    ax.set_ylabel(f"roll_max, epochs <= {matched:,}")
    for s in sorted(set(t.seed)):
        ax.scatter(
            [],
            [],
            s=16,
            marker=marker[s],
            facecolor="white",
            edgecolor="black",
            label=f"seed {s}",
        )
    ax.legend(
        loc="upper left",
        fontsize=5,
        borderpad=0.3,
        ncol=2,
        columnspacing=0.8,
        handlelength=1.2,
    )
    ax.set_title("matched-budget score per arm and seed", fontsize=6, pad=3)
    panel_label(ax, "b")

    ax = axes[2]
    arms_c = [a for a in present if a != REF]
    c = contrasts.get(str(matched), {})
    for i, arm in enumerate(arms_c):
        d = c.get(arm, {}).get("diffs", {})
        for s, v in d.items():
            ax.scatter(
                i,
                v,
                s=16,
                marker=marker[int(s)],
                facecolor=color[arm],
                edgecolor="black",
                lw=0.4,
                zorder=3,
            )
        if d:
            m = float(np.mean(list(d.values())))
            ax.hlines(m, i - 0.3, i + 0.3, color="black", lw=1.0, zorder=2)
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.set_xticks(range(len(arms_c)))
    ax.set_xticklabels(
        [a.replace("H_", "") for a in arms_c], rotation=30, fontsize=5, ha="right"
    )
    ax.set_xlim(-0.5, len(arms_c) - 0.5)
    ax.set_ylim(-0.13, 0.08)
    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.grid(axis="y", lw=0.3, alpha=0.35)
    ax.set_ylabel(f"arm minus {REF}, paired within seed")
    ax.set_title("arm minus H_ref within seed (bar = mean)", fontsize=6, pad=3)
    panel_label(ax, "c")

    for a in axes:
        for s in a.spines.values():
            s.set_visible(True)
    os.makedirs(IMAGE_DIR, exist_ok=True)
    stem = osp.join(IMAGE_DIR, "head_round_readout")
    fig.savefig(stem + ".png", dpi=300)
    savefig_true_size_svg(fig, stem + ".svg")
    plt.close(fig)
    return stem


def main() -> None:
    api = wandb.Api(timeout=120)
    t, curves = fetch(api)
    matched = int(t.n_epochs.min())
    budgets = sorted(set(BUDGETS) | {matched})
    print(
        f"matched budget = {matched}; final epochs {t.set_index(['arm', 'seed']).n_epochs.to_dict()}"
    )
    for b in budgets:
        vals = [roll_max_within(curves[rid], b) for rid in t.run_id]
        t[f"roll_max_le_{b}"] = [v for v, _ in vals]
        t[f"epoch_at_le_{b}"] = [e for _, e in vals]
    t["roll_max_full"] = [roll_max_within(curves[rid], 10**9)[0] for rid in t.run_id]
    csv_path = osp.join(RESULTS, "head_round_readout.csv")
    t.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")
    with pd.option_context("display.width", 250):
        print(
            t.drop(columns=["rows_short_by", "url"]).to_string(
                index=False, float_format=lambda x: f"{x:.4f}"
            )
        )

    contrasts: dict[str, dict[str, object]] = {}
    for b in budgets:
        col = f"roll_max_le_{b}"
        wide = t.pivot(index="seed", columns="arm", values=col)
        if REF not in wide:
            continue
        contrasts[str(b)] = {}
        for arm in ARMS[1:]:
            if arm not in wide:
                continue
            diff = (wide[arm] - wide[REF]).dropna()
            contrasts[str(b)][arm] = {
                "n_pairs": int(len(diff)),
                "diffs": {int(s): float(v) for s, v in diff.items()},
                "mean_diff": float(diff.mean()) if len(diff) else float("nan"),
                "sd_diff": float(diff.std(ddof=1)) if len(diff) > 1 else float("nan"),
            }
    print(f"\n=== paired contrasts, arm minus {REF} within seed ===")
    for b, arms in contrasts.items():
        for arm, c in arms.items():
            print(
                f"  epochs <= {int(b):>5}  {arm:18s} n={c['n_pairs']}  mean {c['mean_diff']:+.4f}  "
                + "  ".join(f"seed{s} {v:+.4f}" for s, v in c["diffs"].items())
            )

    # Arm means over seeds at the matched budget, for the leaderboard reading.
    col = f"roll_max_le_{matched}"
    arm_means = (
        t.groupby("arm")[col]
        .agg(["mean", "std", "count"])
        .reindex([a for a in ARMS if a in set(t.arm)])
    )
    print(f"\n=== arm mean roll_max at epochs <= {matched} ===")
    print(arm_means.to_string(float_format=lambda x: f"{x:.4f}"))

    inc = incumbent_band(matched)
    print(
        f"\nincumbent band at {inc['budget_epochs']} epochs: {inc['mean']:.4f} +/- {inc['sd']:.4f} (n={inc['n']})"
    )
    stem = figure(t, curves, matched, inc, contrasts)
    print(f"figure: {stem}.svg")

    with open(osp.join(RESULTS, "head_round_readout.json"), "w") as fh:
        json.dump(
            {
                "generated_by": "experiments/019-simb-multimodal/scripts/head_round_readout.py",
                "project": PROJECT,
                "config_tag": CONFIG_TAG,
                "metric": METRIC,
                "roll_window": ROLL_WINDOW,
                "history_samples": FULL_HISTORY_SAMPLES,
                "min_epochs": MIN_EPOCHS,
                "matched_budget_epochs": matched,
                "budgets": budgets,
                "runs": t.to_dict(orient="records"),
                "paired_contrasts": contrasts,
                "arm_means_at_matched": arm_means.reset_index().to_dict(
                    orient="records"
                ),
                "incumbent_band": inc,
                "figure": stem + ".svg",
            },
            fh,
            indent=2,
            default=float,
        )


if __name__ == "__main__":
    main()
