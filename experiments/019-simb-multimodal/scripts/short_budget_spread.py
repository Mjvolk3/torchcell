# experiments/019-simb-multimodal/scripts/short_budget_spread.py
# [[experiments.019-simb-multimodal.scripts.short_budget_spread]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/short_budget_spread
"""What is the run-to-run spread at a SHORT budget, and what does it cost in power?

WHY THIS EXISTS. Every power number in the launch plan rests on one measurement: the
replicate spread of 0.0222, taken from the eight identical-config runs at 9,900 epochs
(sec:launch, Evidence 1). A Delta round cannot reach 9,900 epochs, because that partition
caps a job at two days, so the design has to assume the spread holds at roughly 3,000.
That assumption is testable for free: those eight runs each logged a full validation curve,
so their scores at ANY truncated budget can be recomputed from history that already exists.
No new training.

WHAT IT COMPUTES. For each budget E, each replicate's score is the roll_max of its
validation curve restricted to epochs <= E, which is exactly the scoring rule the
leaderboard uses, applied to a prefix. `_roll_max` and `ROLL_WINDOW` are IMPORTED from
pull_round_leaderboards.py rather than reimplemented, so the statistic is the same object
and cannot drift from the leaderboard's.

WHY THE SPREAD MIGHT NOT BE FLAT IN E, in both directions. Truncation removes the late
peaks (three of the eight peaked past epoch 8,800), which compresses the top of the
distribution and should REDUCE the spread. It also catches curves mid-rise at different
points, which should INCREASE it. Which dominates is an empirical question and the whole
reason to measure rather than assume.

WHAT IT REPORTS. Mean, sd, min and max at each budget, plus the smallest gap an 80%-power,
alpha=0.05 two-sided comparison could detect for the three contrasts a 2x2x2 factorial at
64 runs actually makes: main effect (32 v 32), cell against cell (16 v 16), and a pair of
arms at 8 replicates each.

A CAVEAT THE NUMBERS CARRY. W&B history is downsampled to at most HISTORY_SAMPLES points
per run, so a truncated curve has fewer samples than the full one and the rolling window
spans more epochs per step. The effect is small at these budgets but it is real, and it is
why the reported epoch grid is coarse.

Run from the repo root:
    python experiments/019-simb-multimodal/scripts/short_budget_spread.py
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
from scipy.stats import norm

load_dotenv()

from torchcell.utils import (  # noqa: E402
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    panel_label,
    savefig_true_size_svg,
)
from torchcell.utils.paths import experiment_results_dir  # noqa: E402

# Import the leaderboard puller BY PATH: it is a script beside this one rather than an
# installed module, and `_roll_max` must be the identical function so a number here is
# comparable to a number there.
_SPEC = importlib.util.spec_from_file_location(
    "pull_round_leaderboards", osp.join(osp.dirname(osp.abspath(__file__)),
                                        "pull_round_leaderboards.py")
)
assert _SPEC is not None and _SPEC.loader is not None
_PLB = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PLB)

ENTITY = _PLB.ENTITY
ROLL_WINDOW = _PLB.ROLL_WINDOW
HISTORY_SAMPLES = _PLB.HISTORY_SAMPLES
METRIC = "val/expression/pearson_per_feature"

RESULTS = experiment_results_dir("019-simb-multimodal", __file__)
LEADERBOARD = osp.join(RESULTS, "round_leaderboards.csv")

# The budget grid. 2,800 was the cap a 2-day Delta job was thought to afford the SLOWEST arm
# at two runs per card, from an assumed 64 epochs/h; 4,000 was what the reference arms were
# thought to afford. BOTH WERE WRONG. The timing run measured 33.6 epochs/h, at which 2,000
# epochs needs 58.9 h against a 48 h wall, so the grid budget was cut to 700 and the round is
# now scored at a budget below anything this script originally covered. 350 and 500 bracket
# 700 from below so the trend near the real budget is measured rather than extrapolated off
# the bottom of the old grid.
BUDGETS = [350, 500, 700, 1000, 1500, 2000, 2800, 4000, 6000, 9900]

# The config the eight replicates share. Asserted, not assumed -- if the leaderboard ever
# admits a different run to this filter, the spread stops being a replicate spread and the
# whole power argument silently changes.
REPLICATE_CONFIG = {
    "dist": "quantile",
    "lr": 0.0003,
    "dropout": 0.1,
    "num_layers": 6.0,
    "hidden_channels": 90.0,
    "graph_prior": "mask",
    "decoder": "s1_pool",
    "seed": 0.0,
}
# 80% power, alpha=0.05 two-sided: z_{0.975} + z_{0.80}.
Z_SUM = float(norm.ppf(0.975) + norm.ppf(0.80))
# The contrasts a 2x2x2 factorial at 64 runs actually makes.
CONTRASTS = {"main_effect_32v32": (32, 32), "cell_16v16": (16, 16), "arm_8v8": (8, 8)}

plt.rcParams.update({
    "font.family": "Arial", "font.size": 6, "axes.linewidth": 0.5,
    "svg.fonttype": "none", "axes.labelsize": 6,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
})


# The longest runs in the project are no longer all independent. Two of them are RESUMES
# that restart the epoch counter where a previous run stopped (one spans 10,000-12,229, the
# other 12,230-18,276), and because a resume inherits the config it passes the config
# assertion unnoticed. Admitting them would break the measurement twice over: they are not
# independent draws, so the sd would no longer be a replicate spread, and they carry no
# history below epoch 10,000 at all, so every truncated budget would be computed from a
# prefix that does not exist. A resume is identified by where its curve STARTS, which is the
# only thing that distinguishes it, and the leaderboard stores only the last epoch. So the
# split happens after the curves are fetched rather than in the frame filter.
RESUME_START_EPOCH = 100


def replicate_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Long identical-config expression runs. Resumes are dropped later, by curve start."""
    e = df[df.strand.astype(str).str.contains("expr", case=False, na=False)]
    n = e[e.epochs >= 9000].copy()
    for col, want in REPLICATE_CONFIG.items():
        vals = set(n[col])
        if vals != {want}:
            raise ValueError(f"9,000+ epoch runs are not one config: {col}={vals}")
    return n


def detectable(sd: float, n_a: int, n_b: int) -> float:
    """Smallest difference in means detectable at 80% power, alpha=0.05 two-sided."""
    return Z_SUM * sd * float(np.sqrt(1.0 / n_a + 1.0 / n_b))


def main() -> None:
    df = pd.read_csv(LEADERBOARD)
    reps = replicate_runs(df)
    project = sorted(set(reps.project))
    if len(project) != 1:
        raise ValueError(f"replicates span several projects: {project}")
    project = project[0]
    run_ids = sorted(reps.run_id)
    print(f"{len(run_ids)} replicates in {ENTITY}/{project}")

    api = wandb.Api(timeout=60)
    curves: dict[str, pd.DataFrame] = {}
    for rid in run_ids:
        run = api.run(f"{ENTITY}/{project}/{rid}")
        h = run.history(keys=["epoch", METRIC], samples=HISTORY_SAMPLES)
        h = h.dropna(subset=["epoch", METRIC]).sort_values("epoch")
        if h.empty:
            raise ValueError(f"run {rid} returned no {METRIC} history")
        curves[rid] = h
        print(f"  {rid}: {len(h)} points, epochs {int(h.epoch.min())}-{int(h.epoch.max())}")

    resumed = {r: c for r, c in curves.items() if c.epoch.min() > RESUME_START_EPOCH}
    for rid, c in sorted(resumed.items()):
        print(f"  DROPPED {rid}: resume, curve starts at epoch {int(c.epoch.min())}")
        del curves[rid]
    if not curves:
        raise ValueError("every candidate was a resume; no independent replicates left")
    print(f"{len(curves)} independent replicates ({len(resumed)} resumes dropped)")

    rows = []
    per_budget_scores: dict[int, list[float]] = {}
    for budget in BUDGETS:
        scores = []
        for rid, h in curves.items():
            pref = h[h.epoch <= budget]
            if pref.empty:
                raise ValueError(f"run {rid} has no history at or below epoch {budget}")
            best, idx = _PLB._roll_max(pref[METRIC].to_numpy(), window=ROLL_WINDOW)
            scores.append(best)
        s = np.asarray(scores, dtype=float)
        per_budget_scores[budget] = [round(float(v), 4) for v in np.sort(s)]
        sd = float(s.std(ddof=1))
        rows.append({
            "budget_epochs": budget,
            "n": int(s.size),
            "mean": round(float(s.mean()), 4),
            "sd": round(sd, 4),
            "min": round(float(s.min()), 4),
            "max": round(float(s.max()), 4),
            **{k: round(detectable(sd, a, b), 4) for k, (a, b) in CONTRASTS.items()},
        })
        print(f"  budget {budget:>5}: mean {s.mean():.4f}  sd {sd:.4f}  "
              f"main-effect resolves {detectable(sd, 32, 32):.4f}")

    table = pd.DataFrame(rows)
    out = {
        "generated_by": "experiments/019-simb-multimodal/scripts/short_budget_spread.py",
        "entity": ENTITY, "project": project, "metric": METRIC,
        "roll_window": ROLL_WINDOW, "history_samples": HISTORY_SAMPLES,
        "run_ids": sorted(curves), "replicate_config": REPLICATE_CONFIG,
        "resumes_dropped": {r: int(c.epoch.min()) for r, c in sorted(resumed.items())},
        "power": {"alpha": 0.05, "power": 0.80, "z_sum": round(Z_SUM, 4)},
        "by_budget": rows,
        "scores_by_budget": {str(k): v for k, v in per_budget_scores.items()},
    }
    os.makedirs(RESULTS, exist_ok=True)
    with open(osp.join(RESULTS, "short_budget_spread.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    # ---- figure: spread and what it resolves, against budget -------------------------
    w = mm_to_in(PANEL_WIDTHS_MM["half"])
    fig, axes = plt.subplots(1, 2, figsize=(w * 2, mm_to_in(52)))

    ax = axes[0]
    for rid, h in curves.items():
        ax.plot(h.epoch, h[METRIC], lw=0.4, alpha=0.55, color=PLOT_PALETTE[4])
    ax.plot(table.budget_epochs, table["mean"], "o-", lw=1.0, ms=3,
            color=PLOT_PALETTE[1], label="mean roll_max at budget")
    ax.fill_between(table.budget_epochs, table["mean"] - table["sd"],
                    table["mean"] + table["sd"], color=PLOT_PALETTE[1], alpha=0.18,
                    lw=0, label="$\\pm$ 1 sd")
    ax.set_xlabel("epoch budget")
    ax.set_ylabel("val pearson per feature")
    ax.set_xscale("log")
    ax.legend(frameon=False, loc="lower right", handlelength=1.4, borderpad=0.2)
    ax.set_title("eight identical configs, truncated", fontsize=6, pad=3)
    for s in ax.spines.values():
        s.set_visible(True)
    panel_label(ax, "a")

    ax = axes[1]
    ax.plot(table.budget_epochs, table.sd, "o-", lw=1.0, ms=3, color=PLOT_PALETTE[0],
            label="replicate sd")
    ax.plot(table.budget_epochs, table.main_effect_32v32, "s--", lw=1.0, ms=3,
            color=PLOT_PALETTE[2], label="main effect, 32 v 32")
    ax.plot(table.budget_epochs, table.arm_8v8, "^--", lw=1.0, ms=3,
            color=PLOT_PALETTE[5], label="arm pair, 8 v 8")
    ax.set_xlabel("epoch budget")
    ax.set_ylabel("smallest detectable gap")
    ax.set_xscale("log")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="upper right", handlelength=1.6, borderpad=0.2)
    ax.set_title("what a 64-run factorial could resolve", fontsize=6, pad=3)
    for s in ax.spines.values():
        s.set_visible(True)
    panel_label(ax, "b")

    fig.tight_layout(pad=0.4)
    stem = osp.join(os.environ["ASSET_IMAGES_DIR"], "019-simb-multimodal",
                    "short_budget_spread")
    os.makedirs(osp.dirname(stem), exist_ok=True)
    fig.savefig(f"{stem}.png", dpi=300)
    savefig_true_size_svg(fig, f"{stem}.svg")

    print("\n" + table.to_string(index=False))
    print(f"\nwrote {osp.join(RESULTS, 'short_budget_spread.json')}")
    print(f"wrote {stem}.svg")


if __name__ == "__main__":
    main()
