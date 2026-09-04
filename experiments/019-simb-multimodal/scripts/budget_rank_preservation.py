# experiments/019-simb-multimodal/scripts/budget_rank_preservation.py
# [[experiments.019-simb-multimodal.scripts.budget_rank_preservation]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/budget_rank_preservation
"""Does a SHORT-budget screen rank configurations the way a long run does?

WHY THIS EXISTS. The Delta v10 grid scores every cell at 700 epochs, because the measured
throughput of 33.6 epochs/h against a 48 h wall admits nothing longer. `short_budget_spread`
answers what the NOISE is at that budget. It cannot answer the prior question: whether the
ordering a 700-epoch screen produces has anything to do with the ordering at convergence. A
screen that resolves differences precisely and ranks them wrongly is worse than no screen,
because its precision makes the wrong ordering look trustworthy.

WHAT IT COMPUTES. For every long expression run, the same `roll_max` the leaderboard uses,
evaluated twice: on the prefix up to SCREEN_EPOCHS, and on the prefix up to REFERENCE_EPOCHS.
Then the Spearman rank correlation between the two, and the overlap between the top-k sets.
`_roll_max` and `ROLL_WINDOW` are IMPORTED from pull_round_leaderboards.py so the statistic
is the same object as the leaderboard's and cannot drift.

WHY 3,000 IS THE REFERENCE and not 9,900. Only ten runs reach 9,900 and all ten share one
`dist`, so a comparison against them would hold the very factor the grid varies constant. The
reference is itself unconverged, so this measures rank preservation from 700 to 3,000, NOT to
convergence, and is an upper bound on how well the screen predicts a 10,000-epoch outcome.

WHAT THIS CANNOT MEASURE ON THE CURRENT CORPUS, and the distinction is the whole point. The
candidate set looks like it spans four `dist` levels and three dropout levels. It does not,
once resumes are excluded: nearly every non-quantile long run in v8/v9 is a RESUME, so the
surviving live set is 23 runs that are almost one config (quantile, seed 0, dropout 0.1 for
22 of 23). A rank correlation over near-replicates has no true config signal to recover, and
a value near zero is what it should produce whether or not a short screen can rank configs.

So the reported correlation is a WITHIN-CONFIG persistence measurement: does a run's score at
700 epochs predict THAT SAME RUN's score at 3,000. That is worth knowing on its own, because
it is exactly the assumption any early-stopping or trial-pruning rule makes. It is NOT the
cross-config screening validity the grid depends on, and `n_config_cells` in the output
records which of the two the number can support. Reading it as the latter would be the
laundering this repo's evidence discipline exists to prevent.

COLLAPSED RUNS ARE REPORTED SEPARATELY, and the reason is that including them inflates the
answer. A collapsed run sits at the bottom at both budgets, so every collapsed run adds a
concordant pair essentially for free. The headline correlation is over LIVE runs only; the
all-runs figure is reported beside it to show the size of that inflation.

RESUMES ARE EXCLUDED, by where their curve starts. A resumed run has no history below its
restart epoch, so its "prefix up to 700" does not exist.

Run from the repo root:
    PYTHONPATH=. python experiments/019-simb-multimodal/scripts/budget_rank_preservation.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import os.path as osp

import numpy as np
import pandas as pd
import wandb
from dotenv import load_dotenv
from scipy.stats import spearmanr

load_dotenv()

from torchcell.utils import experiment_results_dir  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "_plb", osp.join(osp.dirname(osp.abspath(__file__)), "pull_round_leaderboards.py")
)
_PLB = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PLB)

ENTITY = _PLB.ENTITY
ROLL_WINDOW = _PLB.ROLL_WINDOW
HISTORY_SAMPLES = _PLB.HISTORY_SAMPLES
METRIC = "val/expression/pearson_per_feature"

RESULTS = experiment_results_dir("019-simb-multimodal", __file__)
LEADERBOARD = osp.join(RESULTS, "round_leaderboards.csv")

# The budget the Delta v10 grid actually runs at, and the longest reference that keeps all
# four `dist` levels in the set. See the module docstring for why not 9,900.
SCREEN_EPOCHS = 700
REFERENCE_EPOCHS = 3000

# A curve starting above this epoch is a resume, not a fresh run. Same rule and same reason
# as short_budget_spread.py.
RESUME_START_EPOCH = 100

# A run whose final value is numerically zero is a collapsed constant predictor. This is the
# direct test on the last logged value, NOT the leaderboard's `is_collapsed`, which tests
# max|value| over the whole curve and so misses a run that blips nonzero early and then
# collapses for good. Several v9 runs do exactly that.
COLLAPSE_EPS = 1e-6

TOP_K = (3, 5, 10)


def prefix_roll_max(hist: pd.DataFrame, budget: int) -> float:
    """`roll_max` over the part of a curve at or below `budget` epochs."""
    pref = hist[hist.epoch <= budget]
    if pref.empty:
        raise ValueError(f"no history at or below epoch {budget}")
    best, _ = _PLB._roll_max(pref[METRIC].to_numpy(), window=ROLL_WINDOW)
    return best


def main() -> None:
    df = pd.read_csv(LEADERBOARD)
    e = df[df.strand.astype(str).str.contains("expr", case=False, na=False)]
    cand = e[(e.epochs >= REFERENCE_EPOCHS) & (e.primary_roll_max.notna())]
    if cand.empty:
        raise ValueError(f"no expression runs reach {REFERENCE_EPOCHS} epochs")
    print(f"{len(cand)} candidate runs reaching {REFERENCE_EPOCHS}+ epochs")

    api = wandb.Api(timeout=60)
    rows = []
    for _, r in cand.iterrows():
        run = api.run(f"{ENTITY}/{r.project}/{r.run_id}")
        h = run.history(keys=["epoch", METRIC], samples=HISTORY_SAMPLES)
        h = h.dropna(subset=["epoch", METRIC]).sort_values("epoch")
        if h.empty:
            raise ValueError(f"run {r.run_id} returned no {METRIC} history")
        if h.epoch.min() > RESUME_START_EPOCH:
            print(f"  DROPPED {r.run_id}: resume, starts at epoch {int(h.epoch.min())}")
            continue
        rows.append({
            "run_id": r.run_id,
            "project": r.project,
            "dist": r.dist,
            "dropout": r.dropout,
            "seed": r.seed,
            "epochs": float(r.epochs),
            "screen": prefix_roll_max(h, SCREEN_EPOCHS),
            "reference": prefix_roll_max(h, REFERENCE_EPOCHS),
            "final_value": float(r.primary_last),
        })
    t = pd.DataFrame(rows)
    t["collapsed"] = t.final_value.abs() < COLLAPSE_EPS
    live = t[~t.collapsed].reset_index(drop=True)
    print(f"{len(t)} runs kept, {int(t.collapsed.sum())} collapsed, {len(live)} live")

    def report(sub: pd.DataFrame, label: str) -> dict[str, object]:
        rho, p = spearmanr(sub.screen, sub.reference)
        out: dict[str, object] = {
            "n": int(len(sub)),
            "spearman": round(float(rho), 4),
            "p_value": float(p),
        }
        for k in TOP_K:
            if k > len(sub):
                continue
            a = set(sub.nlargest(k, "screen").run_id)
            b = set(sub.nlargest(k, "reference").run_id)
            out[f"top{k}_overlap"] = int(len(a & b))
        print(f"  {label:12s} n={len(sub):3d}  spearman {rho:+.4f} (p={p:.3g})  "
              + "  ".join(f"top{k} {out.get(f'top{k}_overlap')}/{k}"
                          for k in TOP_K if f"top{k}_overlap" in out))
        return out

    # HOW MANY DISTINCT CONFIG CELLS SURVIVED. One cell means the correlation below is a
    # within-config persistence measurement and carries nothing about ranking configs
    # against each other. Printed loudly because the candidate set before resume exclusion
    # looks far more varied than what is left.
    factors = ["dist", "dropout", "seed"]
    cells = live[factors].drop_duplicates()
    n_in_largest = int(live.groupby(factors, dropna=False).size().max())
    print(f"realized factor variation: {len(cells)} config cell(s) over {len(live)} live "
          f"runs; largest cell holds {n_in_largest}")
    for f in factors:
        print(f"  {f:9s} {live[f].value_counts().to_dict()}")
    if len(cells) == 1 or n_in_largest >= len(live) - 1:
        print("  -> effectively ONE config: this is WITHIN-config persistence, "
              "NOT cross-config screening validity")

    print(f"rank preservation, {SCREEN_EPOCHS} -> {REFERENCE_EPOCHS} epochs:")
    summary = {
        "live_only": report(live, "live only"),
        "all_runs": report(t, "all runs"),
        "n_config_cells": int(len(cells)),
        "n_in_largest_cell": n_in_largest,
        "measures": (
            "within_config_persistence"
            if len(cells) == 1 or n_in_largest >= len(live) - 1
            else "cross_config_screening_validity"
        ),
    }

    # Per-factor means at both budgets. A screen can preserve the OVERALL ordering while
    # inverting the one contrast the grid is built to measure, so the factor the round
    # actually varies is reported separately rather than folded into the correlation.
    by_dist = (live.groupby("dist")[["screen", "reference"]]
               .agg(["mean", "count"]).round(4))
    print(f"\nper-`dist` means, live runs only:\n{by_dist}")
    summary["by_dist"] = json.loads(
        live.groupby("dist")[["screen", "reference"]].mean().round(4).to_json(orient="index")
    )

    out = {
        "generated_by": (
            "experiments/019-simb-multimodal/scripts/budget_rank_preservation.py"
        ),
        "entity": ENTITY, "metric": METRIC, "roll_window": ROLL_WINDOW,
        "history_samples": HISTORY_SAMPLES,
        "screen_epochs": SCREEN_EPOCHS, "reference_epochs": REFERENCE_EPOCHS,
        "summary": summary,
        "runs": json.loads(t.round(4).to_json(orient="records")),
    }
    os.makedirs(RESULTS, exist_ok=True)
    path = osp.join(RESULTS, "budget_rank_preservation.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
