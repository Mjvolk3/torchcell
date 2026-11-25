# experiments/048-pputida-isoprenol-cgt/scripts/sweep_low_data_trial.py
# [[experiments.048-pputida-isoprenol-cgt.scripts.sweep_low_data_trial]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/048-pputida-isoprenol-cgt/scripts/sweep_low_data_trial
"""The low-data feasibility trial: how good a Spearman will 375 records support?

One question, asked with one GPU against a deadline. The campaign gives 465
records over 402 designs, which is two orders of magnitude less data than any
other arm of this program trains on, so the trial is not "tune a model" but
"find out whether a Cell Graph Transformer can be fit at this size at all, and
by how much the graph prior and the metabolic module move it". Boosted trees on
the multi-hot guide vector are the thing to beat, not a formality: on the design
split they score Spearman 0.773 on test, and a 17.6M-parameter transformer that
cannot pass that has not earned its place.

**The objective is Spearman on the held-out split**, and the checkpoint is
selected on validation Spearman to match. Pearson, MSE, RMSE, MAE, NMSE and R^2
are recorded on train, validation and test at every epoch regardless, because a
rank correlation is invariant to any increasing transform of the predictions and
so cannot by itself report that the model has the scale wrong.

Three stages, run in order, each only started if it fits before the deadline:

stage 1
    One seed over the full configuration grid, which ranks the configurations.
stage 2
    The best configurations from stage 1 replicated over further seeds, because
    a 44-record test split has a wide sampling distribution and a single seed is
    not a measurement.
stage 3
    The best configuration on the cycle split, which is the extrapolation
    question: train on cycles 0 to 4, select on cycle 5, predict cycle 6.

The deadline is real and the driver obeys it rather than being killed by it: it
launches nothing whose projected finish passes ``--deadline``, so the trial ends
with every run it did start complete and summarized.

    python experiments/048-pputida-isoprenol-cgt/scripts/sweep_low_data_trial.py \
        --deadline 2026-11-25T14:00 --wandb
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import os.path as osp
import subprocess
import sys
import time
from typing import Any

from dotenv import load_dotenv

load_dotenv()

EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
EXPERIMENT = "048-pputida-isoprenol-cgt"
RESULTS = osp.join(EXPERIMENT_ROOT, EXPERIMENT, "results", "low_data_trial")
TRAINER = osp.join(osp.dirname(osp.abspath(__file__)), "train_isoprenol_cgt.py")

PROJECT = "torchcell_048-pputida-isoprenol-low-data-trial"

#: The grid. Four lambdas because the KL term's magnitude was measured before the
#: sweep rather than guessed: at lambda 1e-3 it contributes 33 against an MSE near
#: 1, so 1e-3 is not a weak prior, it is the whole objective. The ladder brackets
#: the point where the term is a correction rather than the loss.
GRAPH_REG_LAMBDAS = (0.0, 1e-6, 1e-5, 1e-4)

#: Metabolism on and off. Without --flux the cell graph still carries the reaction
#: and metabolite stores but the forward pass never reads them (measured: deleting
#: them moves the head output by 5.2e-08), so this flag is the only thing that
#: makes the metabolic module part of a prediction.
FLUX = (False, True)

#: Weight decay and dropout together, as one "how hard do we lean on the
#: parameters" axis rather than two. With 375 training records and 17.6M
#: parameters the model can memorize the split outright, so the strong setting is
#: not a fine adjustment of the baseline, it is a different hypothesis about what
#: limits the fit.
CAPACITY_CONTROL = (
    {"name": "light", "dropout": 0.1, "weight_decay": 1e-4},
    {"name": "heavy", "dropout": 0.3, "weight_decay": 1e-2},
)

SEEDS_STAGE1 = (42,)
SEEDS_STAGE2 = (43, 44)
SEEDS_STAGE3 = (42, 43, 44, 45, 46)
N_PROMOTED = 3

#: Measured on this card: 9.4 s/epoch with the regularizer and the flux layer on,
#: 4.6 s/epoch with both off, plus about 150 s to build the dataset, the
#: embeddings and the GEM. Used only to decide whether a run fits before the
#: deadline, never to report a result.
SECONDS_PER_EPOCH_REG = 9.6
SECONDS_PER_EPOCH_PLAIN = 4.8
SECONDS_STARTUP = 170.0


def configurations() -> list[dict[str, Any]]:
    """The grid, ordered so the most informative runs happen first.

    Ordered by graph-regularization lambda ascending inside flux ascending, so
    that if the deadline truncates stage 1 what survives is a complete picture of
    the unregularized baseline plus as much of the lambda ladder as fitted,
    rather than a scatter of half-comparable cells.
    """
    out: list[dict[str, Any]] = []
    for control in CAPACITY_CONTROL:
        for flux in FLUX:
            for lam in GRAPH_REG_LAMBDAS:
                out.append(
                    {
                        "tag": (
                            f"{control['name']}"
                            f"_{'flux' if flux else 'pooled'}"
                            f"_reg{lam:g}"
                        ),
                        "flux": flux,
                        "graph_reg_lambda": lam,
                        "dropout": control["dropout"],
                        "weight_decay": control["weight_decay"],
                    }
                )
    return out


def projected_seconds(config: dict[str, Any], epochs: int) -> float:
    """How long a run of this configuration should take, at worst."""
    per_epoch = (
        SECONDS_PER_EPOCH_REG
        if config["graph_reg_lambda"] > 0.0
        else SECONDS_PER_EPOCH_PLAIN
    )
    return SECONDS_STARTUP + per_epoch * epochs


def report_path(split: str, tag: str, seed: int) -> str:
    return osp.join(RESULTS, f"train_{split}_{tag}_seed{seed}.json")


def run_one(
    config: dict[str, Any],
    seed: int,
    split: str,
    args: argparse.Namespace,
    log_handle: Any,
) -> dict[str, Any] | None:
    """Train one cell and return its report, or None if the trainer failed.

    A failed cell does not stop the sweep, and that is deliberate rather than
    permissive: the sweep's value is the comparison across cells, and losing one
    cell to an out-of-memory or a bad lambda should not cost the other fifty. The
    failure is recorded with its exit status and its last output, so a cell that
    is missing from the summary can always be told apart from one that scored
    badly.
    """
    out = report_path(split, config["tag"], seed)
    if osp.exists(out) and not args.overwrite:
        with open(out) as handle:
            return json.load(handle)

    command = [
        sys.executable,
        TRAINER,
        "--split",
        split,
        "--seed",
        str(seed),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--select-on",
        "val_spearman",
        "--dropout",
        str(config["dropout"]),
        "--weight-decay",
        str(config["weight_decay"]),
        "--graph-reg-lambda",
        str(config["graph_reg_lambda"]),
        "--tag",
        config["tag"],
        "--results-subdir",
        "low_data_trial",
        "--project",
        PROJECT,
        "--group",
        f"{split}-{config['tag']}",
    ]
    if config["flux"]:
        command.append("--flux")
    if args.wandb:
        command.append("--wandb")

    print(f"[{dt.datetime.now():%H:%M:%S}] {split} {config['tag']} seed {seed}")
    log_handle.write(f"\n===== {split} {config['tag']} seed {seed} =====\n")
    log_handle.flush()
    completed = subprocess.run(
        command, stdout=log_handle, stderr=subprocess.STDOUT, check=False
    )
    if completed.returncode != 0 or not osp.exists(out):
        print(f"  FAILED, exit {completed.returncode}")
        return None
    with open(out) as handle:
        return json.load(handle)


def cell_row(report: dict[str, Any]) -> dict[str, Any]:
    """One summary row per trained cell, objective first."""
    arguments = report["args"]
    test = report["cgt"]["test"]
    val = report["cgt"]["val"]
    nulls = report["lower_order_nulls"]
    return {
        "split": arguments["split"],
        "tag": arguments["tag"],
        "seed": arguments["seed"],
        "flux": bool(arguments["flux"]),
        "graph_reg_lambda": float(arguments["graph_reg_lambda"]),
        "dropout": float(arguments["dropout"]),
        "weight_decay": float(arguments["weight_decay"]),
        "test_spearman": test["spearman"],
        "val_spearman": val["spearman"],
        "test_pearson": test["pearson"],
        "test_mse": test["mse"],
        "test_rmse": test["rmse"],
        "test_mae": test["mae"],
        "test_nmse": test["nmse"],
        "test_r2": test["r2"],
        "trees_test_spearman": nulls["boosted_trees"]["test_spearman"],
        "ridge_test_spearman": nulls["ridge"]["test_spearman"],
        "best_epoch": report["best_epoch"],
        "epochs_run": report["epochs_run"],
        "seconds": report["seconds"],
        "graph_reg_max": (
            0.0
            if report["graph_reg"] is None
            else report["graph_reg"]["max_epoch_loss"]
        ),
    }


def write_summary(rows: list[dict[str, Any]]) -> str:
    """Write the sweep table, sorted by the objective."""
    import pandas as pd

    frame = pd.DataFrame(rows)
    # Sorted by VALIDATION Spearman, not test: a table ordered by the held-out
    # score reads as a ranking of the configurations, and ordering by the number
    # being reported is how a selection decision gets made on the test set
    # without anyone intending it.
    frame = frame.sort_values("val_spearman", ascending=False)
    path = osp.join(RESULTS, "sweep_summary.csv")
    frame.to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deadline",
        required=True,
        help="local-time ISO timestamp after which no new run is launched",
    )
    parser.add_argument(
        "--grid-split",
        default="design",
        choices=("design", "cycle"),
        help=(
            "the split the configuration grid runs on. The cycle split is the "
            "honest rank objective and the design split is not: measured on the "
            "44-record design test set, ranking by DBTL cycle number alone "
            "scores Spearman 0.807, beating every model, because titer rose "
            "across the campaign and the design split mixes cycles across train "
            "and test. Every cycle-split test record is cycle 6, so there the "
            "chronology has zero variance and can carry no rank information. "
            "See cycle_confound_audit.py"
        ),
    )
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--margin-seconds",
        type=float,
        default=900.0,
        help="headroom kept before the deadline for the summary and the writeup",
    )
    args = parser.parse_args()

    deadline = dt.datetime.fromisoformat(args.deadline).timestamp()
    os.makedirs(RESULTS, exist_ok=True)
    log_path = osp.join(RESULTS, "sweep_low_data_trial.log")
    rows: list[dict[str, Any]] = []

    def fits(config: dict[str, Any]) -> bool:
        remaining = deadline - args.margin_seconds - time.time()
        return projected_seconds(config, args.epochs) <= remaining

    with open(log_path, "a") as log_handle:
        grid = configurations()
        print(
            f"stage 1: {len(grid)} configurations on the {args.grid_split} "
            f"split, seed {SEEDS_STAGE1[0]}"
        )
        for config in grid:
            if not fits(config):
                print(f"  skipping {config['tag']}, will not finish before deadline")
                continue
            for seed in SEEDS_STAGE1:
                report = run_one(config, seed, args.grid_split, args, log_handle)
                if report is not None:
                    rows.append(cell_row(report))
        if rows:
            print(f"stage 1 summary -> {write_summary(rows)}")

        # Promote on VALIDATION Spearman. Ranking on the test score, which this
        # did until 2026-11-24, chooses which configurations get replicated using
        # the held-out set, so "the best three configurations" comes out
        # optimistically biased even though each cell's own test score stays an
        # honest held-out measurement. The design pass's stage 2 ran under the
        # old key and its three promoted tags are recorded as test-selected.
        ranked = sorted(
            {row["tag"] for row in rows},
            key=lambda tag: max(
                row["val_spearman"] for row in rows if row["tag"] == tag
            ),
            reverse=True,
        )
        # Kept in RANK order, not grid order, because stage 3 trains the single
        # best configuration on the cycle split and reading the first element of
        # a grid-ordered list would silently pick whichever promoted cell came
        # earliest in the sweep instead.
        by_tag = {c["tag"]: c for c in grid}
        promoted = [by_tag[tag] for tag in ranked[:N_PROMOTED]]
        print(f"stage 2: replicating {[c['tag'] for c in promoted]}")
        for config in promoted:
            for seed in SEEDS_STAGE2:
                if not fits(config):
                    print(f"  skipping {config['tag']} seed {seed}, out of time")
                    continue
                report = run_one(config, seed, args.grid_split, args, log_handle)
                if report is not None:
                    rows.append(cell_row(report))
        if rows:
            print(f"stage 2 summary -> {write_summary(rows)}")

        # Stage 3 takes the winner to the OTHER split, whichever the grid did
        # not use, so a pass always ends with the same configuration scored under
        # both regimes. That pairing is the only way to say how much of a
        # design-split score was the campaign's chronology.
        other = "cycle" if args.grid_split == "design" else "design"
        if promoted:
            best = promoted[0]
            print(f"stage 3: {best['tag']} on the {other} split")
            for seed in SEEDS_STAGE3:
                if not fits(best):
                    print(f"  skipping {other} seed {seed}, out of time")
                    continue
                report = run_one(best, seed, other, args, log_handle)
                if report is not None:
                    rows.append(cell_row(report))

    if rows:
        print(f"\nfinal summary -> {write_summary(rows)}")
    print(f"log -> {log_path}")


if __name__ == "__main__":
    main()
