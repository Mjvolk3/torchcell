# experiments/027-betaxanthin-metabolic/scripts/penalty_ladder.py
# [[experiments.027-betaxanthin-metabolic.scripts.penalty_ladder]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/027-betaxanthin-metabolic/scripts/penalty_ladder
"""The active-row balance weight against score and feasibility, and against the EPOCH.

The balance-weight ladder was launched to draw a tradeoff curve: pay more for the physics,
lose prediction. It does not draw one, and the reason matters more than the curve would
have.

Every warm-started cell begins NEARLY FEASIBLE, because the warm start is the pFBA
reference, and then training destroys that feasibility monotonically, at every penalty
weight including the largest. The active-row balance maximum starts near 1.0 and reaches
2.0, its ceiling, by the selected epoch in three of the four cells. The one cell that looks
feasible at its selected epoch (weight 0.5, balance max 1.0091) was selected at EPOCH 7.

So the feasible cells are not the heavily penalized ones. They are the ones the validation
selector happened to stop early, which is the same phenomenon as the 10-epoch cells reading
0.98 population feasibility. The penalty weight is not the lever; the epoch is.

This script reports, per cell:

* score, prediction spread ratio and feasibility AT THE SELECTED EPOCH, which is what a
  results table would quote;
* feasibility at a fixed ladder of epochs (0, 5, 10, 25, 50, 100, final), which is what
  shows the decay and is not confounded with where the selector stopped;
* the epoch at which balance is best, and Spearman's rank correlation between epoch and
  balance max over the whole run, which is the decay as one number;
* the Spearman score against its own null standard error, because 0.088 on 629 genotypes is
  about 2.2 standard errors and 0.034 is under one.

Run from the worktree root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/027-betaxanthin-metabolic/scripts/penalty_ladder.py

Writes ``results/penalty_ladder.json``.
"""

from __future__ import annotations

import json
import os
import os.path as osp

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS = osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic", "results")
OUT = osp.join(RESULTS, "penalty_ladder.json")
#: label -> result file. All seed 2700, all the frozen warm-start head at K=100 boxforce,
#: all 200 epochs, differing ONLY in the flux penalty weights.
LADDER = {
    "balance_0.0": "wave10_warmstart_frozen_bw00_k100_boxforce_e200_s2700.json",
    "balance_0.1": "wave10_warmstart_frozen_bw01_k100_boxforce_e200_s2700.json",
    "balance_0.5": "wave9_warmstart_frozen_k100_boxforce_e200_s2700.json",
    "balance_2.0": "wave10_warmstart_frozen_bw2_k100_boxforce_e200_s2700.json",
    "all_penalties_off": "wave10_warmstart_frozen_nopen_k100_boxforce_e200_s2700.json",
}
#: The short-budget cells, whose whole point is that selection cannot leave the early window.
SHORT_BUDGET = {
    f"short_budget_e10_s{seed}": (
        f"wave10_warmstart_frozen_e10_k100_boxforce_e10_s{seed}.json"
    )
    for seed in (2700, 2701, 2702)
}
EPOCH_LADDER = (0, 5, 10, 25, 50, 100)
#: The balance ratio is |(Sv)_i| / (turnover_i + eps), which is bounded above by 2.
BALANCE_CEILING = 2.0


class CellLadderRow(BaseModel):
    """One cell: what a table would quote, and what the trajectory actually does."""

    model_config = ConfigDict(extra="forbid")
    label: str
    balance_weight: float
    parsimony_weight: float
    constraint_scale: float
    epochs_run: int
    select_epoch: int
    test_spearman: float
    n_test: int
    spearman_z: float = Field(
        description="Spearman over its null standard error, 1/sqrt(n-1)."
    )
    spearman_p: float
    val_pred_sd_ratio_at_select: float
    balance_max_at_select: float
    balance_p95_at_select: float
    balance_max_by_epoch: dict[str, float] = Field(
        description="Balance max at a FIXED epoch ladder, not confounded with selection."
    )
    best_balance_epoch: int
    best_balance_max: float
    epoch_vs_balance_spearman: float = Field(
        description="Rank correlation of epoch with balance max; positive means it decays."
    )
    at_ceiling_at_select: bool


class AcrossCells(BaseModel):
    """What predicts a cell's feasibility and score: the epoch it stopped at, or the weight.

    Computed ACROSS cells, over the long runs, which is the comparison the ladder was built
    to make. Both candidate explanations are scored the same way so neither is privileged.
    """

    model_config = ConfigDict(extra="forbid")
    n_cells: int
    select_epoch_vs_balance_spearman: float = Field(
        description="Selected epoch against balance max AT that epoch; the epoch explanation."
    )
    balance_weight_vs_balance_spearman: float = Field(
        description="Penalty weight against balance max at select; the penalty explanation."
    )
    select_epoch_vs_score_spearman: float
    balance_weight_vs_score_spearman: float
    verdict: str


class PenaltyLadderReport(BaseModel):
    """The ladder, and the epoch-decay finding that replaces the tradeoff it was built for."""

    model_config = ConfigDict(extra="forbid")
    rows: list[CellLadderRow]
    across_cells: AcrossCells
    finding: str
    note: str


def describe(label: str, filename: str) -> CellLadderRow | None:
    path = osp.join(RESULTS, filename)
    if not osp.isfile(path):
        return None
    with open(path) as fh:
        d = json.load(fh)
    cell = d["cell"]
    history = d["history"]
    pinned = d["test"]["pinned"]
    selected = int(d["select_epoch"])
    at_select = next((e for e in history if e["epoch"] == selected), history[-1])
    balance = np.array(
        [float(e.get("feas_balance_active_max", np.nan)) for e in history], dtype=float
    )
    epochs = np.array([int(e["epoch"]) for e in history], dtype=float)
    finite = np.isfinite(balance)
    if finite.sum() > 2:
        decay = float(stats.spearmanr(epochs[finite], balance[finite]).statistic)
        best_index = int(np.nanargmin(balance))
    else:
        decay, best_index = float("nan"), 0
    n = int(pinned["n"])
    rho = float(pinned["spearman"])
    se = 1.0 / np.sqrt(n - 1)
    z = rho / se
    ladder = {
        str(e): float(balance[e])
        for e in EPOCH_LADDER
        if e < len(balance) and np.isfinite(balance[e])
    }
    ladder["final"] = float(balance[-1])
    return CellLadderRow(
        label=label,
        balance_weight=float(cell["flux_balance_active_weight"]),
        parsimony_weight=float(cell["flux_parsimony_weight"]),
        constraint_scale=float(cell["constraint_scale"]),
        epochs_run=len(history),
        select_epoch=selected,
        test_spearman=rho,
        n_test=n,
        spearman_z=float(z),
        spearman_p=float(2 * (1 - stats.norm.cdf(abs(z)))),
        val_pred_sd_ratio_at_select=float(at_select["val_pred_sd_ratio"]),
        balance_max_at_select=float(at_select.get("feas_balance_active_max", np.nan)),
        balance_p95_at_select=float(at_select.get("feas_balance_active_p95", np.nan)),
        balance_max_by_epoch=ladder,
        best_balance_epoch=int(epochs[best_index]),
        best_balance_max=float(balance[best_index]),
        epoch_vs_balance_spearman=decay,
        at_ceiling_at_select=bool(
            at_select.get("feas_balance_active_max", 0) >= BALANCE_CEILING - 1e-3
        ),
    )


def main() -> None:
    rows = [
        r
        for r in (
            describe(label, name) for label, name in {**LADDER, **SHORT_BUDGET}.items()
        )
        if r is not None
    ]
    assert rows, "no warm-start ladder result files."

    long_runs = [r for r in rows if r.epochs_run > 50]
    # THE SIGN IS NOT AVERAGED, because one cell flips it and a mean would hide that. The
    # weight-2.0 cell reads a NEGATIVE epoch-balance correlation, and the reason is not that
    # its physics improves: it is already AT the ceiling of 2.0 by epoch 10, so it has
    # nowhere left to decay to and the correlation is measuring noise around a pinned value.
    decaying = [r for r in long_runs if r.epoch_vs_balance_spearman > 0]
    best_epochs = [r.best_balance_epoch for r in long_runs]
    # THE LADDER'S ACTUAL QUESTION, asked across cells: does the penalty weight or the
    # stopping epoch explain where a cell lands? Both are scored the same way.
    epochs = [float(r.select_epoch) for r in long_runs]
    weights = [r.balance_weight for r in long_runs]
    balances = [r.balance_max_at_select for r in long_runs]
    scores = [r.test_spearman for r in long_runs]

    def rank(a: list[float], b: list[float]) -> float:
        return float(stats.spearmanr(a, b).statistic)

    across = AcrossCells(
        n_cells=len(long_runs),
        select_epoch_vs_balance_spearman=rank(epochs, balances),
        balance_weight_vs_balance_spearman=rank(weights, balances),
        select_epoch_vs_score_spearman=rank(epochs, scores),
        balance_weight_vs_score_spearman=rank(weights, scores),
        verdict=(
            "READ THE WITHIN-CELL TRAJECTORIES, NOT THIS TABLE. These correlations are over "
            "five cells, where the 5 percent critical value for Spearman is about 0.90, so "
            "only the epoch-against-score entry reaches it and the two balance entries "
            "(+0.60 for the epoch, +0.46 for the weight) are not distinguishable from each "
            "other or from zero. What is strong is the per-cell evidence, which has 200 "
            "epochs per cell rather than five points: every cell reaches its best balance "
            "within 8 epochs and degrades after, at every weight. The across-cell direction "
            "agrees with that reading, a late-selected cell being infeasible and scoring "
            "better, but on its own it would not establish it."
        ),
    )

    report = PenaltyLadderReport(
        rows=rows,
        across_cells=across,
        finding=(
            f"The best balance of the whole run is reached at epoch {min(best_epochs)} to "
            f"{max(best_epochs)} in ALL {len(long_runs)} long cells, whatever the penalty "
            f"weight. In {len(decaying)} of them the balance maximum then rises with the "
            "epoch (rank correlation "
            + ", ".join(f"{r.epoch_vs_balance_spearman:+.2f}" for r in decaying)
            + f"); the remaining {len(long_runs) - len(decaying)} is already at the ceiling "
            f"of {BALANCE_CEILING} by epoch 10 and so cannot decay further. The one cell "
            "that looks feasible at its selected epoch is the one selected at epoch 7. So "
            "the penalty weight is not what buys feasibility; stopping early is, which is "
            "the same phenomenon as the 10-epoch cells reading 0.98 population feasibility."
        ),
        note=(
            "The balance ratio is bounded above by 2, so a cell reading 2.000 is at the "
            "ceiling and not merely large. A Spearman of 0.088 on 629 genotypes is 2.2 "
            "null standard errors (p = 0.027) and one of 0.034 is 0.84 (p = 0.40), so the "
            "score column separates the low-weight cells from the high-weight ones only "
            "weakly and at one seed."
        ),
    )
    with open(OUT, "w") as fh:
        json.dump(report.model_dump(), fh, indent=2)

    print(
        f"{'cell':22s} {'bw':>5s} {'sel':>5s} {'rho':>8s} {'p':>6s} {'sd_r':>7s} "
        f"{'bal@sel':>8s} {'bal@ep10':>9s} {'best(ep)':>12s} {'decay':>7s}"
    )
    for r in rows:
        best = f"{r.best_balance_max:.3f}({r.best_balance_epoch})"
        at10 = r.balance_max_by_epoch.get("10", float("nan"))
        print(
            f"{r.label:22s} {r.balance_weight:5.1f} {r.select_epoch:5d} "
            f"{r.test_spearman:+8.4f} {r.spearman_p:6.3f} "
            f"{r.val_pred_sd_ratio_at_select:7.4f} {r.balance_max_at_select:8.4f} "
            f"{at10:9.4f} {best:>12s} {r.epoch_vs_balance_spearman:+7.3f}"
        )
    a = report.across_cells
    print(f"\nACROSS THE {a.n_cells} LONG CELLS (Spearman):")
    print(
        f"  selected epoch  vs balance-at-select {a.select_epoch_vs_balance_spearman:+.3f}"
        f"   |  vs score {a.select_epoch_vs_score_spearman:+.3f}"
    )
    print(
        f"  balance weight  vs balance-at-select {a.balance_weight_vs_balance_spearman:+.3f}"
        f"   |  vs score {a.balance_weight_vs_score_spearman:+.3f}"
    )
    print(f"  {a.verdict}")
    print(f"\nFINDING: {report.finding}")
    print(f"\n{OUT}")


if __name__ == "__main__":
    main()
