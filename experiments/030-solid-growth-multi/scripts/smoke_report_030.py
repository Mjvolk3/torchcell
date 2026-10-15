# experiments/030-solid-growth-multi/scripts/smoke_report_030.py
# [[experiments.030-solid-growth-multi.scripts.smoke_report_030]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/030-solid-growth-multi/scripts/smoke_report_030
r"""PASS/FAIL of the dataset-token smoke test from the token run's measurements.

Reads the ``SmokeResult`` JSON of the token run (``results/smoke/<config>_seed<seed>.json``,
written by the training script under ``smoke.enabled``) and applies the criteria of the
plan (decision 10) with ``delta`` the normalized offset of each source's synthetic twin:

1. The readout learned the offset: ``mean_diff`` (prediction under the twin token minus
   under the own token, same genotype, same encoder pass) within 10% of ``delta``, and
   its sd across rows below ``delta / 5``.
2. The token carries the target: reading real rows under the twin token raises their MSE
   by about ``delta**2`` (between 0.5x and 1.5x).

The plan's third criterion, a control run whose clones keep the original token and whose
loss must sit at least ``delta**2 / 4`` above the token run's, is not applied. Jobs 2863
and 2866 ran it: the floor is 0.0225 and the two runs' validation point losses differed by
0.06 from initialization alone, so the comparison cannot resolve the floor at any budget a
smoke can afford. Within the token run the same fact is criterion 1: a readout that could
not separate the twin from the source would show ``mean_diff`` near 0.

Prints a table and exits 0 on PASS, 1 on FAIL.

    python experiments/030-solid-growth-multi/scripts/smoke_report_030.py \\
        --tok results/smoke/cgt_030_smoke_tok_000_seed0.json
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


def criteria(tok: dict[str, Any]) -> list[Criterion]:
    """The criteria from the token run's measurement file."""
    delta = float(tok["delta"])
    assert not tok["control"], "pass --tok the token run, not a control"
    mean_diff = float(tok["mean_diff"])
    sd_diff = float(tok["sd_diff"])
    mse_rise = float(tok["mse_swapped"]) - float(tok["mse_own"])
    d2 = delta * delta
    return [
        Criterion(
            name="1a mean(pred_twin - pred_own) within 10% of delta",
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
            name="2  MSE rise under the twin token about delta^2",
            value=mse_rise,
            bound=f"[{0.5 * d2:.4f}, {1.5 * d2:.4f}]",
            passed=0.5 * d2 <= mse_rise <= 1.5 * d2,
        ),
    ]


def main() -> int:
    """Print the report; 0 on PASS, 1 on FAIL."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tok", required=True)
    args = parser.parse_args()
    with open(args.tok) as f:
        tok = json.load(f)
    lines = criteria(tok)
    print(
        f"token run: {tok['config_name']} seed {tok['seed']} wandb {tok['wandb_run_id']} "
        f"rows {tok['n_rows']} over {tok['n_batches']} validation batches"
    )
    print(f"delta {tok['delta']}  twins {tok['twins']}")
    print(
        f"mse_own {tok['mse_own']:.4f} mse_twin {tok['mse_swapped']:.4f} "
        f"mse_all {tok['mse_all_rows']:.4f} resid real {tok['mean_resid_real']:+.4f} "
        f"clone {tok['mean_resid_clone']:+.4f}"
    )
    cm = tok["callback_metrics"]
    for key in (
        "val/gene_interaction/Pearson",
        "val/fitness/Pearson",
        "val_ess/auroc_released_smf",
        "val_ess/auroc_matched_smf",
    ):
        value = cm.get(key)
        print(f"  {key:36s} {'nan' if value is None else f'{value:.4f}'}")
    print()
    for c in lines:
        print(
            f"[{'PASS' if c.passed else 'FAIL'}] {c.name:56s} {c.value:+.4f}  {c.bound}"
        )
    ok = all(c.passed for c in lines)
    print()
    print("SMOKE PASS" if ok else "SMOKE FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
