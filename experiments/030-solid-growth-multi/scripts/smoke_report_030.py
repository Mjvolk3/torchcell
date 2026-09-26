# experiments/030-solid-growth-multi/scripts/smoke_report_030.py
# [[experiments.030-solid-growth-multi.scripts.smoke_report_030]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/030-solid-growth-multi/scripts/smoke_report_030
r"""PASS/FAIL of the dataset-token smoke test from the two runs' measurements.

Reads the ``SmokeResult`` JSON of the token run and of the control run
(``results/smoke/<config>_seed<seed>.json``, written by the training script under
``smoke.enabled``) and applies the three criteria of the plan (decision 10), with
``delta`` the normalized offset of the synthetic token:

1. The readout learned the offset: ``mean_diff`` (prediction under the synthetic token
   minus under the own token, same genotype) within 10% of ``delta``, and its sd across
   rows below ``delta / 5``.
2. The token carries the target: reading real rows under the synthetic token raises
   their MSE by about ``delta**2`` (between 0.5x and 1.5x).
3. Without the token the loss floor is ``delta**2 / 4``: the control's MSE over all rows
   exceeds the token run's by at least half of that.

Prints a table and exits 0 on PASS, 1 on FAIL.

    python experiments/030-solid-growth-multi/scripts/smoke_report_030.py \\
        --tok results/smoke/cgt_030_smoke_tok_000_seed0.json \\
        --ctl results/smoke/cgt_030_smoke_ctl_000_seed0.json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from pydantic import BaseModel


class Criterion(BaseModel):
    """One PASS/FAIL line of the report."""

    name: str
    value: float
    bound: str
    passed: bool


def criteria(tok: dict[str, Any], ctl: dict[str, Any]) -> list[Criterion]:
    """The three criteria from the two measurement files."""
    delta = float(tok["delta"])
    assert float(ctl["delta"]) == delta, "token and control runs used different deltas"
    assert not tok["control"] and ctl["control"], (
        "pass --tok the token run, --ctl the control"
    )
    mean_diff = float(tok["mean_diff"])
    sd_diff = float(tok["sd_diff"])
    mse_rise = float(tok["mse_swapped"]) - float(tok["mse_own"])
    floor_gap = float(ctl["mse_all_rows"]) - float(tok["mse_all_rows"])
    d2 = delta * delta
    return [
        Criterion(
            name="1a mean(pred_syn - pred_own) within 10% of delta",
            value=mean_diff,
            bound=f"[{0.9 * delta:.4f}, {1.1 * delta:.4f}]",
            passed=abs(mean_diff - delta) <= 0.1 * abs(delta),
        ),
        Criterion(
            name="1b sd of that difference below delta/5",
            value=sd_diff,
            bound=f"< {abs(delta) / 5:.4f}",
            passed=sd_diff < abs(delta) / 5,
        ),
        Criterion(
            name="2  MSE rise under the swapped token about delta^2",
            value=mse_rise,
            bound=f"[{0.5 * d2:.4f}, {1.5 * d2:.4f}]",
            passed=0.5 * d2 <= mse_rise <= 1.5 * d2,
        ),
        Criterion(
            name="3  control MSE(all rows) - token MSE(all rows) >= delta^2/8",
            value=floor_gap,
            bound=f">= {d2 / 8:.4f} (floor delta^2/4 = {d2 / 4:.4f})",
            passed=floor_gap >= d2 / 8,
        ),
    ]


def main() -> int:
    """Print the report; 0 on PASS, 1 on FAIL."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tok", required=True)
    parser.add_argument("--ctl", required=True)
    args = parser.parse_args()
    with open(args.tok) as f:
        tok = json.load(f)
    with open(args.ctl) as f:
        ctl = json.load(f)
    lines = criteria(tok, ctl)
    print(
        f"token run  : {tok['config_name']} seed {tok['seed']} wandb {tok['wandb_run_id']} rows {tok['n_rows']}"
    )
    print(
        f"control run: {ctl['config_name']} seed {ctl['seed']} wandb {ctl['wandb_run_id']} rows {ctl['n_rows']}"
    )
    print(
        f"delta {tok['delta']}  token {tok['token_name']} (index {tok['token_index']})"
    )
    print(
        f"token: mse_own {tok['mse_own']:.4f} mse_swapped {tok['mse_swapped']:.4f} "
        f"mse_all {tok['mse_all_rows']:.4f} resid real {tok['mean_resid_real']:+.4f} clone {tok['mean_resid_clone']:+.4f}"
    )
    print(
        f"ctl  : mse_own {ctl['mse_own']:.4f} mse_swapped {ctl['mse_swapped']:.4f} "
        f"mse_all {ctl['mse_all_rows']:.4f} resid real {ctl['mean_resid_real']:+.4f} clone {ctl['mean_resid_clone']:+.4f}"
    )
    for key in (
        "val/gene_interaction/Pearson",
        "val_ess/auroc_released_smf",
        "val_ess/auroc_matched_smf",
    ):
        print(
            f"  {key:36s} token {tok['callback_metrics'].get(key, float('nan')):.4f}  ctl {ctl['callback_metrics'].get(key, float('nan')):.4f}"
        )
    print()
    for c in lines:
        print(
            f"[{'PASS' if c.passed else 'FAIL'}] {c.name:60s} {c.value:+.4f}  {c.bound}"
        )
    ok = all(c.passed for c in lines)
    print()
    print("SMOKE PASS" if ok else "SMOKE FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
