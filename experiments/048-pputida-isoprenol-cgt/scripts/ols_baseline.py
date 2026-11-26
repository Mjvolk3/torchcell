# experiments/048-pputida-isoprenol-cgt/scripts/ols_baseline.py
# [[experiments.048-pputida-isoprenol-cgt.scripts.ols_baseline]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/048-pputida-isoprenol-cgt/scripts/ols_baseline
"""The standard baseline for every 048 model: OLS on the guide-target vector.

**Why OLS and nothing else.** A baseline with settings invites the question of
whether it was tuned as hard as the model it is compared against, and on a
465-record campaign that question has no cheap answer. Ridge has ``alpha``, a
tree ensemble has depth and an iteration count; ordinary least squares has
nothing. Fixing the baseline this way makes the comparison identical across
experiments and removes a degree of freedom from the person reporting.

**What it is fit on.** The multi-hot guide-target vector: for each strain, a
0/1 entry per gene saying whether the CRISPRi design targets it. That is exactly
what the transformer is told about the genotype, minus the graph and the
sequence, so the gap between them is the accounting of what the structure buys.

**The column restriction matters and is not a tuning knob.** The indicator spans
all 5,729 genes but only the campaign's CRISPRi targets are ever nonzero, so the
raw matrix has far more columns than training rows and OLS is underdetermined;
``lstsq`` would hand back a minimum-norm solution, which is ridge by the back
door. Columns constant across the training rows carry no information about the
fit, so dropping them changes no prediction and makes the problem well posed.

The trainer computes this inline for new runs. This script exists to score the
splits of runs made BEFORE the baseline was standardized, so the whole trial can
be read against one baseline without retraining 72 cells. It writes
``results/low_data_trial/ols_baseline.csv``, one row per split and seed.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from pputida_cgt_data import (  # noqa: E402
    build_dataset,
    campaign_table,
    cycle_splits,
    design_splits,
)
from train_isoprenol_cgt import guide_indicator, regression_metrics  # noqa: E402

load_dotenv()

EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
TRIAL = osp.join(
    EXPERIMENT_ROOT, "048-pputida-isoprenol-cgt", "results", "low_data_trial"
)


def score_split(
    dataset: Any, table: pd.DataFrame, splits: dict[str, list[int]]
) -> dict[str, Any]:
    """Fit OLS on the training rows and score validation and test."""
    targets = {
        name: np.asarray(
            [float(table.loc[table["index"] == i, "titer"].iloc[0]) for i in idx]
        )
        for name, idx in splits.items()
    }
    features = {name: guide_indicator(dataset, idx) for name, idx in splits.items()}
    informative = features["train"].std(axis=0) > 0.0
    features = {name: matrix[:, informative] for name, matrix in features.items()}

    mean = float(targets["train"].mean())
    sd = float(targets["train"].std())
    model = LinearRegression()
    model.fit(features["train"], targets["train"])

    row: dict[str, Any] = {
        "n_features": int(informative.sum()),
        "n_train": len(targets["train"]),
        "n_test": len(targets["test"]),
    }
    for name in ("val", "test"):
        prediction = model.predict(features[name])
        metrics = regression_metrics(targets[name], prediction, mean, sd)
        for key, value in metrics.items():
            row[f"{name}_{key}"] = value
        if name == "test":
            # Kept per record so the model-versus-baseline gap can be
            # bootstrapped over the records both scored, which a summary
            # statistic cannot support.
            row["_test_indices"] = [int(i) for i in splits["test"]]
            row["_test_observed"] = [float(x) for x in targets["test"]]
            row["_test_predicted"] = [float(x) for x in prediction]
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        default="42,43,44,45,46",
        help="design-split seeds to score; the cycle split is deterministic",
    )
    args = parser.parse_args()

    dataset = build_dataset()
    table = campaign_table(dataset)
    rows = []

    cycle = score_split(dataset, table, cycle_splits(table))
    cycle.update({"split": "cycle", "seed": -1})
    rows.append(cycle)

    for seed in (int(s) for s in args.seeds.split(",") if s):
        design = score_split(dataset, table, design_splits(table, seed=seed))
        design.update({"split": "design", "seed": seed})
        rows.append(design)
    dataset.close_lmdb()

    predictions = {
        f"{row['split']}_{row['seed']}": {
            "test_indices": row.pop("_test_indices"),
            "test_observed": row.pop("_test_observed"),
            "test_predicted": row.pop("_test_predicted"),
        }
        for row in rows
    }
    with open(osp.join(TRIAL, "ols_predictions.json"), "w") as handle:
        json.dump(predictions, handle)

    frame = pd.DataFrame(rows)
    columns = ["split", "seed", "n_features", "n_train", "n_test"] + [
        c for c in frame.columns if c.startswith(("val_", "test_"))
    ]
    frame = frame[columns]
    os.makedirs(TRIAL, exist_ok=True)
    path = osp.join(TRIAL, "ols_baseline.csv")
    frame.to_csv(path, index=False)

    show = [
        "split",
        "seed",
        "n_features",
        "n_train",
        "n_test",
        "test_spearman",
        "test_pearson",
        "test_r2",
        "test_rmse",
    ]
    print("OLS on the guide-target vector, the standard 048 baseline")
    print(frame[show].to_string(index=False, float_format="%.3f"))
    print(path)


if __name__ == "__main__":
    main()
