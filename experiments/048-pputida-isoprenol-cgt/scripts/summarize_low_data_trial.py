# experiments/048-pputida-isoprenol-cgt/scripts/summarize_low_data_trial.py
# [[experiments.048-pputida-isoprenol-cgt.scripts.summarize_low_data_trial]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/048-pputida-isoprenol-cgt/scripts/summarize_low_data_trial
"""Every cell of the low-data trial in two tables, per cell and per configuration.

The trial ran four passes over one GPU and wrote one JSON per cell. This reads
all of them and is the only thing that should be quoted: the tables in the goal
note come from here, not from the driver's own ``sweep_summary.csv``, which each
pass overwrites with its own rows, and not from anything typed at a prompt.

**The offset column is the one worth explaining.** MSE decomposes exactly as

    MSE = bias^2 + (sd_o^2 + sd_p^2 - 2 r sd_o sd_p)

where ``bias`` is the mean observed minus the mean predicted, so the constant
component of a cell's error is recoverable from its reported Pearson and the two
standard deviations without saving a single prediction. That is how the flux
head's miscentering was found: the offset is 0 to 14 percent of test MSE for
every pooled cell and 12 to 86 percent for every flux cell. Negative values
under the square root are clipped at zero, which happens only from rounding in
the stored statistics.

``nmse`` is normalized by the variance of the split being scored, so 1.0 is the
score of predicting that split's own mean, and a cell above 1.0 is worse than a
constant however well it ranks.

Writes ``results/low_data_trial/trial_cells.csv`` and
``trial_configurations.csv``.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import os.path as osp
from typing import Any

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
TRIAL = osp.join(
    EXPERIMENT_ROOT, "048-pputida-isoprenol-cgt", "results", "low_data_trial"
)


def offset(metrics: dict[str, Any]) -> float:
    """The constant component of the error, in ug/mL."""
    sd_o = metrics["observed_sd"]
    sd_p = metrics["predicted_sd"]
    scatter = sd_o**2 + sd_p**2 - 2.0 * metrics["pearson"] * sd_o * sd_p
    return math.sqrt(max(metrics["mse_raw"] - scatter, 0.0))


def cell_rows() -> pd.DataFrame:
    """One row per trained cell."""
    rows: list[dict[str, Any]] = []
    for path in sorted(glob.glob(osp.join(TRIAL, "train_*.json"))):
        with open(path) as handle:
            report = json.load(handle)
        arguments = report["args"]
        test = report["cgt"]["test"]
        val = report["cgt"]["val"]
        nulls = report["lower_order_nulls"]
        rows.append(
            {
                "split": arguments["split"],
                "tag": arguments["tag"],
                "seed": arguments["seed"],
                "flux": bool(arguments["flux"]),
                "graph_reg_lambda": float(arguments["graph_reg_lambda"]),
                "dropout": float(arguments["dropout"]),
                "weight_decay": float(arguments["weight_decay"]),
                "select_on": arguments.get("select_on", "val_mse"),
                "no_decay_bias": bool(arguments.get("no_decay_bias", False)),
                "epochs": int(arguments["epochs"]),
                "patience": int(arguments["patience"]),
                "n_test": test["n"],
                "best_epoch": report["best_epoch"],
                "epochs_run": report["epochs_run"],
                "val_spearman": val["spearman"],
                "test_spearman": test["spearman"],
                "test_pearson": test["pearson"],
                "test_rmse": test["rmse"],
                "test_mae": test["mae"],
                "test_nmse": test["nmse"],
                "test_r2": test["r2"],
                "test_offset": offset(test),
                "ols_test_spearman": (
                    nulls["ols"]["test_spearman"] if "ols" in nulls else float("nan")
                ),
                "seconds": report["seconds"],
                "has_predictions": "test_predicted" in report,
            }
        )
    if not rows:
        raise ValueError(
            f"no train_*.json under {TRIAL}; the trial writes one report per "
            f"cell and there is nothing to summarize"
        )
    return pd.DataFrame(rows)


def attach_ols(cells: pd.DataFrame) -> pd.DataFrame:
    """Fill the OLS baseline for cells trained before it was standardized.

    OLS depends only on the split, not on the model, so scoring each split once
    with `ols_baseline.py` gives every cell its baseline without retraining a
    single one. The cycle split is deterministic and carries seed -1; the design
    split re-rolls per seed and is matched on the seed.
    """
    path = osp.join(TRIAL, "ols_baseline.csv")
    if not osp.exists(path):
        return cells
    baseline = pd.read_csv(path)[["split", "seed", "test_spearman"]]
    baseline = baseline.rename(columns={"test_spearman": "_ols"})
    cycle = baseline[baseline.split == "cycle"]["_ols"]
    cycle_value = float(cycle.iloc[0]) if len(cycle) else float("nan")
    design = baseline[baseline.split == "design"].set_index("seed")["_ols"].to_dict()

    def lookup(row: pd.Series) -> float:
        if not pd.isna(row["ols_test_spearman"]):
            return float(row["ols_test_spearman"])
        if row["split"] == "cycle":
            return cycle_value
        return float(design.get(int(row["seed"]), float("nan")))

    cells = cells.copy()
    cells["ols_test_spearman"] = cells.apply(lookup, axis=1)
    return cells


def configuration_rows(cells: pd.DataFrame) -> pd.DataFrame:
    """One row per split and configuration, aggregated over seeds.

    The standard deviations mean different things on the two splits and the
    column names cannot carry that, so it is said here. On the cycle split the
    partition is deterministic and every seed scores the same 59 cycle-6
    records, so the spread is initialization variance only and says nothing
    about the test set's size. On the design split the seed re-rolls the split
    as well as the initialization, so the spread does include test-set
    resampling. A cycle-split standard deviation is therefore NOT a sampling
    error and must not be read as one; ``bootstrap_headline_gap.py`` resamples
    records for that.
    """
    grouped = cells.groupby(["split", "tag"], as_index=False).agg(
        n_seeds=("seed", "count"),
        test_spearman_mean=("test_spearman", "mean"),
        test_spearman_sd=("test_spearman", "std"),
        test_pearson_mean=("test_pearson", "mean"),
        test_pearson_sd=("test_pearson", "std"),
        test_rmse_mean=("test_rmse", "mean"),
        test_rmse_sd=("test_rmse", "std"),
        test_nmse_mean=("test_nmse", "mean"),
        test_r2_mean=("test_r2", "mean"),
        test_r2_sd=("test_r2", "std"),
        test_offset_mean=("test_offset", "mean"),
        test_offset_sd=("test_offset", "std"),
        val_spearman_mean=("val_spearman", "mean"),
        ols_mean=("ols_test_spearman", "mean"),
        best_epoch_min=("best_epoch", "min"),
        best_epoch_max=("best_epoch", "max"),
    )
    grouped["gap_over_ols"] = grouped["test_spearman_mean"] - grouped["ols_mean"]
    return grouped.sort_values(["split", "val_spearman_mean"], ascending=[True, False])


#: What actually identifies a configuration. The TAG does not: the trial ran the
#: same settings under two different tags, `heavy_pooled_reg1e-05` in the cycle
#: grid and `selrule_val_spearman` in the selection-rule arm, at the same three
#: seeds. Grouping by tag reported them as two configurations scoring 0.620 and
#: 0.580, which is a 0.040 disagreement between runs that differ in nothing but
#: when they were launched. Grouping by these columns pools them, which is the
#: honest estimate and also the measurement of how reproducible a cell is.
#:
#: The epoch budget and the patience belong here too, and leaving them out was a
#: real error caught in use: a 2-epoch smoke cell shares every other setting with
#: the headline configuration, so it pooled straight into it and pulled the mean
#: from 0.600 to 0.589. A truncated run is not the same configuration however
#: identical its flags.
IDENTITY = (
    "split",
    "flux",
    "graph_reg_lambda",
    "dropout",
    "weight_decay",
    "select_on",
    "no_decay_bias",
    "epochs",
    "patience",
)


def pooled_rows(cells: pd.DataFrame) -> pd.DataFrame:
    """One row per genuinely distinct configuration, pooling duplicate tags.

    The spread here mixes initialization variance with run-to-run
    nondeterminism, and the latter is not zero even at a fixed seed: the trainer
    seeds Python, NumPy and Torch and sets cuDNN deterministic, but it does not
    set ``torch.use_deterministic_algorithms``, and the forward pass uses
    ``index_add_`` and scatter reductions whose CUDA implementations accumulate
    in nondeterministic order. Two launches of one seed therefore need not agree.
    A standard deviation from this table is the right one to quote for "how well
    does this configuration do", and it is wider than a single tag's.
    """
    grouped = cells.groupby(list(IDENTITY), as_index=False).agg(
        n_runs=("seed", "count"),
        n_seeds=("seed", "nunique"),
        tags=("tag", lambda s: ",".join(sorted(set(s)))),
        test_spearman_mean=("test_spearman", "mean"),
        test_spearman_sd=("test_spearman", "std"),
        test_pearson_mean=("test_pearson", "mean"),
        test_rmse_mean=("test_rmse", "mean"),
        test_nmse_mean=("test_nmse", "mean"),
        test_r2_mean=("test_r2", "mean"),
        test_r2_sd=("test_r2", "std"),
        test_offset_mean=("test_offset", "mean"),
        ols_mean=("ols_test_spearman", "mean"),
    )
    grouped["gap_over_ols"] = grouped["test_spearman_mean"] - grouped["ols_mean"]
    return grouped.sort_values(["split", "test_spearman_mean"], ascending=[True, False])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default=None, choices=("design", "cycle"))
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument(
        "--pooled",
        action="store_true",
        help="print the by-hyperparameter table instead of the by-tag one",
    )
    args = parser.parse_args()

    cells = attach_ols(cell_rows())
    configurations = configuration_rows(cells)
    pooled = pooled_rows(cells)
    cells.to_csv(osp.join(TRIAL, "trial_cells.csv"), index=False)
    configurations.to_csv(osp.join(TRIAL, "trial_configurations.csv"), index=False)
    pooled.to_csv(osp.join(TRIAL, "trial_pooled.csv"), index=False)

    if args.pooled:
        shown = pooled if args.split is None else pooled[pooled.split == args.split]
        columns = [
            "split",
            "tags",
            "n_runs",
            "n_seeds",
            "test_spearman_mean",
            "test_spearman_sd",
            "test_rmse_mean",
            "test_nmse_mean",
            "test_r2_mean",
            "test_offset_mean",
            "gap_over_ols",
        ]
        print(f"{len(cells)} cells, {len(pooled)} distinct configurations")
        print(shown[columns].head(args.top).to_string(index=False, float_format="%.3f"))
        print(osp.join(TRIAL, "trial_pooled.csv"))
        return

    shown = configurations
    if args.split is not None:
        shown = shown[shown.split == args.split]
    columns = [
        "split",
        "tag",
        "n_seeds",
        "val_spearman_mean",
        "test_spearman_mean",
        "test_spearman_sd",
        "test_pearson_mean",
        "test_rmse_mean",
        "test_nmse_mean",
        "test_r2_mean",
        "test_offset_mean",
        "gap_over_ols",
    ]
    print(f"{len(cells)} cells, {len(configurations)} split-configuration pairs")
    print(shown[columns].head(args.top).to_string(index=False, float_format="%.3f"))
    print()
    print(f"cells with saved predictions: {int(cells.has_predictions.sum())}")
    print(f"total GPU seconds: {cells.seconds.sum():.0f}")
    print(osp.join(TRIAL, "trial_cells.csv"))
    print(osp.join(TRIAL, "trial_configurations.csv"))


if __name__ == "__main__":
    main()
