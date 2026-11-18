# experiments/048-pputida-isoprenol-cgt/scripts/summarize_arms.py
# [[experiments.048-pputida-isoprenol-cgt.scripts.summarize_arms]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/048-pputida-isoprenol-cgt/scripts/summarize_arms
"""One table over every finished 048 training run: score and wall time.

Reads the per-run JSON that ``train_isoprenol_cgt.py`` writes and produces two
CSVs under ``results/``: one row per run, and one row per arm. The per-arm row
is what a training-time estimate should be read from, since it is measured.

The run's ``seconds`` field covers the training loop and the final evaluation,
not the dataset and cell-graph load that precedes it, so the job-level overhead
is the slurm elapsed time minus the summed ``seconds`` and is reported from
``sacct`` rather than from here.

Statistics, named: Pearson r between predicted and measured titer on the split,
for the checkpoint with the lowest validation MSE. The arm value is the mean and
the sample standard deviation (ddof=1) over seeds.
"""

from __future__ import annotations

import glob
import json
import os
import os.path as osp

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

RESULTS = osp.join(
    os.environ["EXPERIMENT_ROOT"], "048-pputida-isoprenol-cgt", "results"
)


def run_table() -> pd.DataFrame:
    """One row per finished run."""
    rows = []
    for path in sorted(glob.glob(osp.join(RESULTS, "train_*_seed*.json"))):
        with open(path) as handle:
            report = json.load(handle)
        args = report["args"]
        nulls = report["lower_order_nulls"]
        rows.append(
            {
                "arm": args["tag"],
                "split": args["split"],
                "seed": args["seed"],
                "model": report["model_label"],
                "n_params": report["n_params"],
                "n_train": report["split_sizes"]["train"],
                "n_val": report["split_sizes"]["val"],
                "n_test": report["split_sizes"]["test"],
                "epochs_run": report["epochs_run"],
                "best_epoch": report["best_epoch"],
                "seconds": report["seconds"],
                "seconds_per_epoch": report["seconds"] / report["epochs_run"],
                "cgt_val_pearson": report["cgt"]["val"]["pearson"],
                "cgt_test_pearson": report["cgt"]["test"]["pearson"],
                "ridge_test_pearson": nulls["ridge"]["test_pearson"],
                "trees_val_pearson": nulls["boosted_trees"]["val_pearson"],
                "trees_test_pearson": nulls["boosted_trees"]["test_pearson"],
            }
        )
    return pd.DataFrame(rows)


def arm_table(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per arm: mean and sample SD over seeds."""
    grouped = runs.groupby("arm", sort=True)
    table = grouped.agg(
        n_seeds=("seed", "count"),
        n_train=("n_train", "first"),
        n_test=("n_test", "first"),
        epochs_run_mean=("epochs_run", "mean"),
        epochs_run_max=("epochs_run", "max"),
        seconds_mean=("seconds", "mean"),
        seconds_max=("seconds", "max"),
        seconds_total=("seconds", "sum"),
        seconds_per_epoch=("seconds_per_epoch", "mean"),
        cgt_val_pearson_mean=("cgt_val_pearson", "mean"),
        cgt_test_pearson_mean=("cgt_test_pearson", "mean"),
        cgt_test_pearson_sd=("cgt_test_pearson", "std"),
        ridge_test_pearson_mean=("ridge_test_pearson", "mean"),
        trees_test_pearson_mean=("trees_test_pearson", "mean"),
        trees_test_pearson_sd=("trees_test_pearson", "std"),
    )
    return table.reset_index()


def main() -> None:
    runs = run_table()
    arms = arm_table(runs)
    runs.to_csv(osp.join(RESULTS, "arm_runs.csv"), index=False)
    arms.to_csv(osp.join(RESULTS, "arm_summary.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(runs.round(3).to_string(index=False))
        print()
        print(arms.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
