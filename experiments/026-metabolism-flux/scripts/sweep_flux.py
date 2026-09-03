# experiments/026-metabolism-flux/scripts/sweep_flux.py
# [[experiments.026-metabolism-flux.scripts.sweep_flux]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/sweep_flux.py

r"""Overnight sweeps for the flux layer, built around one measured problem.

WHY THIS EXISTS RATHER THAN A LONGER ``train_flux.py`` RUN
-----------------------------------------------------------
The obvious use of a 12 h window is more epochs. The existing runs say that is the
wrong experiment. Reading ``results/flux_arms_gpu*.json``, over all 18 completed
(arm, seed) runs at 20 epochs:

* **validation Pearson peaks at a MEDIAN EPOCH OF 4.5** and never later than 18, and
  the mean of the last five epochs is at or below zero for 13 of the 18 runs. The
  models overfit almost immediately, so a longer budget buys a worse number.
* **the reported per-arm score is a maximum over 20 epochs**, which is an
  upward-biased order statistic, not an estimate of the arm's performance.
* **the validation set carries 353 betaxanthin measurements**, measured at seed 999 by
  the new ``n_val_betaxanthin`` counter. A Pearson correlation on n observations has
  null width :math:`1/\sqrt{n-3}`, which is 0.0535 here, and the arms are separated by
  0.004 to 0.08. Every reported difference sits inside one null width of the others.

Those three facts have one shape: the experiment is currently unable to distinguish
any arm from any other, or from nothing at all. More epochs does not change that.
Replication and a calibrated null do.

THE THREE GRIDS
---------------
``null``
    ``pooled`` and ``flux_anchored`` with ``--permute-train-targets``, which destroys
    the genotype-to-phenotype association in training while scoring validation against
    real labels. The maximum over epochs of each such run is one draw from the null
    distribution of the exact statistic every arm is reported with. Twelve seeds per
    arm gives that distribution directly, so "is 0.087 a result" stops being a
    judgment call. This is the single most decisive cell in the sweep: if the null
    routinely reaches 0.09, no arm reported so far has measured anything.

``reg``
    Learning rate crossed with weight decay crossed with hidden width, on one arm.
    Overfitting by epoch 5 is a regularization statement, and this asks whether any
    setting in the neighborhood escapes the noise floor rather than peaking early and
    decaying. It is the only grid that could raise the ceiling.

``arms``
    The five registered arms at fresh seeds. Combined with the three seeds already
    banked, this brings each arm to thirteen replicates, enough that a standard error
    near 0.009 can resolve a real 0.02 separation. It answers the ordering question the
    20-epoch run could not.

SCORING, AND WHY THE FULL HISTORY IS KEPT
------------------------------------------
Every cell stores its complete per-epoch history, including the new
``n_val_<head>`` count, so the scoring rule is chosen at analysis time and can be
stated alongside the number. Nothing here reports a bare maximum.

Run from the worktree root, one GPU per invocation::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/sweep_flux.py \
        --grid arms --seeds 101,102,103,104,105 --out sweep_arms_a.json
"""

import argparse
import json
import os
import os.path as osp
import sys
import time
from typing import Any

import numpy as np
import wandb
from pydantic import BaseModel, ConfigDict

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from train_flux import ARMS, RESULTS_DIR, build_dataset, run_arm  # noqa: E402

#: The two arms the null is calibrated on: the baseline readout and the fullest flux
#: arm. A null measured on one architecture does not transfer to another, because the
#: statistic's width depends on how much the model can chase the validation set.
NULL_ARMS = ["pooled", "flux_anchored"]

#: The regularization neighborhood. The incumbent is (1e-3, 1e-4, 32), which peaks at
#: epoch 5, so the grid moves down in learning rate and up in decay from there, and
#: brackets width in both directions.
REG_LR = [1e-3, 3e-4, 1e-4]
REG_WEIGHT_DECAY = [1e-4, 1e-2]
REG_HIDDEN = [32, 64]


class SweepCell(BaseModel):
    """One training run, fully specified. Serialized into the results file verbatim.

    Kept as a pydantic model rather than a dict so a results file records exactly what
    each cell ran, and so an unrecognized key is an error at construction instead of a
    silently ignored hyperparameter.
    """

    model_config = ConfigDict(extra="forbid")

    label: str
    arm: str
    seed: int
    epochs: int
    hidden: int = 32
    layers: int = 2
    batch_size: int = 128
    num_workers: int = 3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    permute_train_targets: bool = False


def score_run(history: list[dict[str, Any]]) -> tuple[float, int, float]:
    """Peak, peak epoch and last-five mean of betaxanthin validation Pearson.

    Duplicated in `analyze_sweep.py` on purpose in the sense that the analysis owns the
    full scoring; this exists only so the W&B summary carries the same three numbers and
    the live table can be sorted without waiting for the analysis to run. Non-finite
    epochs are dropped rather than scored as zero, which keeps a run that broke
    distinguishable from a run that merely performed badly.
    """
    vals = [
        (int(r["epoch"]), float(r["val_betaxanthin"]))
        for r in history
        if np.isfinite(r["val_betaxanthin"])
    ]
    if not vals:
        return float("nan"), -1, float("nan")
    peak_epoch, peak = max(vals, key=lambda ev: ev[1])
    return peak, peak_epoch, float(np.mean([v for _, v in vals[-5:]]))


def build_cells(grid: str, seeds: list[int], arm: str, epochs: int) -> list[SweepCell]:
    """Expand a named grid into its cells, SEED-MAJOR.

    Seed-major ordering is load-bearing under a wall clock. Cell-major finishes every
    seed of one configuration before starting the next, so an interrupted job yields
    complete data for some configurations and none for others, which supports no
    comparison. Seed-major makes any prefix of the job a balanced experiment, and each
    further pass adds a replicate to all of them at once.
    """
    cells: list[SweepCell] = []
    if grid == "null":
        for seed in seeds:
            for arm_name in NULL_ARMS:
                cells.append(
                    SweepCell(
                        label=f"null-{arm_name}",
                        arm=arm_name,
                        seed=seed,
                        epochs=epochs,
                        permute_train_targets=True,
                    )
                )
    elif grid == "reg":
        for seed in seeds:
            for lr in REG_LR:
                for wd in REG_WEIGHT_DECAY:
                    for hidden in REG_HIDDEN:
                        cells.append(
                            SweepCell(
                                label=f"reg-{arm}-lr{lr:g}-wd{wd:g}-h{hidden}",
                                arm=arm,
                                seed=seed,
                                epochs=epochs,
                                hidden=hidden,
                                lr=lr,
                                weight_decay=wd,
                            )
                        )
    elif grid == "arms":
        for seed in seeds:
            for arm_name in ARMS:
                cells.append(
                    SweepCell(
                        label=f"arms-{arm_name}", arm=arm_name, seed=seed, epochs=epochs
                    )
                )
    else:
        raise ValueError(f"unknown grid {grid!r}; expected null, reg or arms")
    return cells


def main() -> None:
    """Run every cell of one grid on one GPU, checkpointing after each."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", required=True, choices=["null", "reg", "arms"])
    parser.add_argument("--seeds", required=True)
    parser.add_argument(
        "--arm", default="flux_anchored", help="the arm the reg grid sweeps"
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--max-hours",
        type=float,
        default=11.0,
        help=(
            "Stop before starting a cell that is not expected to finish within this "
            "budget, so the job ends on a complete run rather than being killed "
            "mid-cell by the scheduler and losing it."
        ),
    )
    parser.add_argument("--wandb-project", default="torchcell_026_flux_sweep")
    parser.add_argument(
        "--wandb-mode",
        default="online",
        choices=["online", "offline", "disabled"],
        help=(
            "Online is the point: four unattended jobs whose only other output is a "
            "JSON file written once per 24-minute cell cannot be watched in time to "
            "act on. Offline still records everything for a later sync."
        ),
    )
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    cells = build_cells(args.grid, seeds, args.arm, args.epochs)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = osp.join(RESULTS_DIR, args.out)

    print(
        f"grid {args.grid}: {len(cells)} cells, budget {args.max_hours} h", flush=True
    )
    dataset = build_dataset()
    print(f"dataset: {len(dataset)} aggregated genotypes", flush=True)

    t_start = time.time()
    runs: list[dict[str, Any]] = []
    durations: list[float] = []
    for i, cell in enumerate(cells):
        elapsed_h = (time.time() - t_start) / 3600.0
        if durations:
            projected = elapsed_h + (max(durations) / 3600.0)
            if projected > args.max_hours:
                print(
                    f"stopping before cell {i + 1}/{len(cells)}: {elapsed_h:.2f} h used, "
                    f"a further {max(durations) / 3600.0:.2f} h would exceed "
                    f"{args.max_hours} h",
                    flush=True,
                )
                break
        run_args = argparse.Namespace(
            **cell.model_dump(exclude={"label", "arm", "seed", "permute_train_targets"})
        )
        print(
            f"[cell {i + 1}/{len(cells)}] {cell.label} seed {cell.seed} "
            f"({elapsed_h:.2f} h used)",
            flush=True,
        )
        # One W&B run per CELL, not per job. A job is 24 cells of different arms and
        # hyperparameters, so a single run would interleave 24 unrelated curves onto one
        # step axis and none of them would be readable. Grouping by grid keeps the four
        # concurrent jobs' cells comparable in one view.
        wandb_run = wandb.init(
            project=args.wandb_project,
            group=args.grid,
            name=f"{cell.label}-s{cell.seed}",
            tags=[args.grid, cell.arm, f"seed{cell.seed}"],
            config=cell.model_dump(),
            reinit=True,
            mode=args.wandb_mode,
        )
        result = run_arm(
            cell.arm,
            cell.seed,
            dataset,
            run_args,
            permute_train_targets=cell.permute_train_targets,
            on_epoch=lambda row: wandb.log(row, step=int(row["epoch"])),
        )
        result["cell"] = cell.model_dump()
        result["wandb_run_id"] = wandb_run.id
        result["wandb_url"] = wandb_run.url
        # The summary carries the three scoring rules the analysis reports, so the W&B
        # table is directly sortable on them and never on a bare maximum alone.
        peak, peak_epoch, last5 = score_run(result["history"])
        wandb_run.summary.update(
            {
                "peak_betaxanthin": peak,
                "peak_epoch": peak_epoch,
                "last5_betaxanthin": last5,
                "n_val_betaxanthin": result["history"][-1].get("n_val_betaxanthin", 0),
                "wall_time_s": result["wall_time_s"],
                "n_parameters": result["n_parameters"],
            }
        )
        wandb_run.finish()
        durations.append(float(result["wall_time_s"]))
        runs.append(result)
        with open(out_path, "w") as f:
            json.dump(
                {"grid": args.grid, "args": vars(args), "runs": runs},
                f,
                indent=2,
                default=str,
            )
        print(f"checkpointed {len(runs)}/{len(cells)} -> {out_path}", flush=True)

    print(
        f"wrote {out_path}: {len(runs)} of {len(cells)} cells in "
        f"{(time.time() - t_start) / 3600.0:.2f} h"
    )


if __name__ == "__main__":
    main()
