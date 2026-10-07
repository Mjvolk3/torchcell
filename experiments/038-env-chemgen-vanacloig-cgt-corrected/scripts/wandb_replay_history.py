# experiments/038-env-chemgen-vanacloig-cgt-corrected/scripts/wandb_replay_history.py
# [[experiments.038-env-chemgen-vanacloig-cgt-corrected.scripts.wandb_replay_history]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/038-env-chemgen-vanacloig-cgt-corrected/scripts/wandb_replay_history
"""Replay a trainer's ``_history.csv`` into one W&B run with per-step curves.

The factorized and hit trainers log a single summary row per config, so the training
curves live only in ``results/factorized/<sweep>/<name>_history.csv``. This logs that
file back as a run named ``<name>:history`` in the same project: one metric per fold and
seed for the validation score, the train loss, and the graph penalty, keyed by step, so a
curve that is still rising at the last checkpoint can be seen on a run page.

Usage: ``python wandb_replay_history.py results/factorized/r2_cgt/cgt_bil_prior1_history.csv``
"""

from __future__ import annotations

import argparse
import os
import os.path as osp

import pandas as pd
import wandb
from dotenv import load_dotenv

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
PROJECT = "torchcell_038-env-chemgen-vanacloig-cgt-corrected"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("history", help="path to a <name>_history.csv")
    args = parser.parse_args()
    history = pd.read_csv(args.history)
    sweep = osp.basename(osp.dirname(args.history))
    name = osp.basename(args.history).removesuffix("_history.csv")
    run = wandb.init(
        project=PROJECT,
        group=f"{sweep}/{name}",
        name=f"{name}:history",
        tags=["history", sweep],
        config={"sweep": sweep, "name": name, "source": args.history},
        dir=osp.join(
            DATA_ROOT, "wandb-experiments", "038-env-chemgen-vanacloig-cgt-corrected"
        ),
    )
    run.define_metric("step")
    run.define_metric("*", step_metric="step")
    folds = sorted(history["fold"].unique()) if "fold" in history else [0]
    for step, rows in history.groupby("step", sort=True):
        record: dict[str, float] = {"step": int(step)}
        for _, r in rows.iterrows():
            key = f"fold{int(r.get('fold', 0))}_seed{int(r['seed'])}"
            record[f"val_centered_mean/{key}"] = float(r["val_centered_mean"])
            record[f"train_loss/{key}"] = float(r["train_loss"])
            if "penalty" in r:
                record[f"penalty/{key}"] = float(r["penalty"])
        record["val_centered_mean/mean_over_folds"] = float(
            rows["val_centered_mean"].mean()
        )
        run.log(record)
    run.summary["folds"] = len(folds)
    run.summary["selected_steps"] = (
        history.groupby(["fold", "seed"])["selected_step"].first().tolist()
        if "fold" in history
        else history.groupby("seed")["selected_step"].first().tolist()
    )
    print(run.url, flush=True)
    run.finish()


if __name__ == "__main__":
    main()
