# experiments/043-expression-proteome-morphology/scripts/alcf/wandb_run_live_043.py
# [[experiments.043-expression-proteome-morphology.scripts.alcf.wandb_run_live_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/alcf/wandb_run_live_043.py
"""Exit 0 when a W&B run of the 043 project is live elsewhere, 1 when it is not.

Live = state ``running`` AND a heartbeat within ``--fresh-minutes`` (default 10). The
heartbeat test lets a requeued job take back its own run after a preemption kill, which
W&B reports as ``running`` until the heartbeat times out. A run that does not exist is
not live. Used by run_cell_043.sh so a cell queued on two routes (preemptable and
capacity, same resume_id) never trains twice.

    python wandb_run_live_043.py o1023_all_onto_hlgauss_equal_s0
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime

import wandb

PATH = "zhao-group/torchcell_043-expression-proteome-morphology"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--fresh-minutes", type=float, default=10.0)
    args = ap.parse_args()
    try:
        run = wandb.Api(timeout=60).run(f"{PATH}/{args.run_id}")
    except wandb.errors.CommError:
        print(f"{args.run_id}: no such run")
        return 1
    beat = run._attrs.get("heartbeatAt")
    age = None
    if beat:
        t = datetime.fromisoformat(beat.replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=UTC)
        age = (datetime.now(UTC) - t).total_seconds() / 60
    live = run.state == "running" and age is not None and age < args.fresh_minutes
    print(
        f"{args.run_id}: state {run.state}, heartbeat {age if age is None else round(age, 1)} min ago -> {'LIVE' if live else 'free'}"
    )
    return 0 if live else 1


if __name__ == "__main__":
    sys.exit(main())
