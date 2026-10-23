#!/usr/bin/env python
# scripts/check_wandb_run_keys.py
# [[scripts.check_wandb_run_keys]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/check_wandb_run_keys
"""Gate: a smoke run must log every required curve before a round is submitted.

Reads one W&B run's history and fails (exit 1) when a required key never appears,
or appears on fewer than ``--min-rows`` rows (a key logged once at the end is a
summary, not a curve). The default key set is the standing rule for every torchcell
trainer: train loss AND validation loss (the same objective over the validation pass),
learning rate, gradient norm, and the epoch counter. Extra keys for a round (per-target
validation loss, per-dataset scores) are passed with ``--require``.

    python scripts/check_wandb_run_keys.py zhao-group/<project>/<run_id> \
        --require val/kemmeren/loss --min-rows 2

The output lists every required key with its row count. The run id is the 8-character
id from the run URL, not the run name.
"""

from __future__ import annotations

import argparse
import sys

import wandb

DEFAULT_KEYS: tuple[str, ...] = (
    "epoch",
    "train/loss",
    "val/loss",
    "train/lr",
    "train/grad_norm",
)


def row_counts(path: str, keys: list[str], page_size: int = 5000) -> dict[str, int]:
    """Rows of the UNSAMPLED history on which each key is present and not null."""
    run = wandb.Api().run(path)
    counts = dict.fromkeys(keys, 0)
    for row in run.scan_history(page_size=page_size):
        for key in keys:
            if row.get(key) is not None:
                counts[key] += 1
    return counts


def main() -> int:
    """Print the row count of every required key; exit 1 when one is not a curve."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_path", help="entity/project/run_id")
    parser.add_argument(
        "--require",
        action="append",
        default=[],
        help="an additional key that must be logged (repeatable)",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=2,
        help="rows a key needs to count as a curve (default 2: one per evaluation)",
    )
    args = parser.parse_args()
    keys = list(DEFAULT_KEYS) + args.require
    counts = row_counts(args.run_path, keys)
    width = max(len(k) for k in keys)
    missing: list[str] = []
    for key in keys:
        n = counts[key]
        verdict = "ok" if n >= args.min_rows else "MISSING"
        if n < args.min_rows:
            missing.append(key)
        print(f"{key:<{width}}  rows {n:>6}  {verdict}")
    if missing:
        print(
            f"\nFAIL {args.run_path}: {len(missing)} required key(s) not logged as a "
            f"curve: {', '.join(missing)}. Fix the trainer and rerun the smoke before "
            "submitting the round."
        )
        return 1
    print(
        f"\nOK {args.run_path}: every required key is logged on >= {args.min_rows} rows"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
