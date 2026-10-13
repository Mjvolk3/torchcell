# experiments/040-inhibitor-synergy-wetlab/scripts/r3_readout.py
# [[experiments.040-inhibitor-synergy-wetlab.scripts.r3_readout]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/040-inhibitor-synergy-wetlab/scripts/r3_readout
"""Read every scored mixture arm, on this machine and pulled from Delta, into two tables.

``results/mixture/`` holds the GilaHyper runs and ``results/mixture-delta/`` the Delta
runs' tables (rsynced from the Delta tree without the checkpoints). The same arm can
have been run on both; the ``machine`` column keeps them apart, and a reader that wants
one copy per arm takes the one with the more folds or the one named.

``results/r3_readout_arms.csv``: one row per (sweep, arm), the median over folds of the
seed-mean of the gene-level held-out score and the wet-lab readouts under the served
call, plus the per-source held-out scores an arm with auxiliary sources carries.

``results/r3_readout_selection.csv``: one row per run (sweep, config, seed), the epoch
``select: val`` chose, the validation and test scores at that step, the best and last test
scores over the history, and the epoch at which each source's held-out loss bottomed.
The gap between the test score at the selected step and the best test score over the
history is the cost of selecting on four validation compounds, and the last-step score is
what ``select: last`` would have reported; all three are read from the history file every
run writes, so no arm has to be rerun to be read either way.

Round 3 runs one seed per fold on three folds, so an arm's row is a screen reading: the
single-seed spread across host-head variants of the same model (0.25 to 0.33 on the
gene-level score) is the noise floor a difference has to clear.

    python experiments/040-inhibitor-synergy-wetlab/scripts/r3_readout.py
"""

from __future__ import annotations

import glob
import os.path as osp
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from mixture_data import read_history  # noqa: E402

EXPERIMENT = osp.dirname(osp.dirname(osp.abspath(__file__)))
RESULTS = osp.join(EXPERIMENT, "results")
MACHINES = {
    "gilahyper": osp.join(RESULTS, "mixture"),
    "delta": osp.join(RESULTS, "mixture-delta"),
}

#: (column, task, subset, metric, call) of the readouts an arm row carries
READOUTS: list[tuple[str, str, str, str, str | None]] = [
    ("vanacloig", "vanacloig", "test_compounds", "centered_spearman_median", None),
    ("vanacloig_genes", "vanacloig", "test_genes", "raw_spearman_median", None),
    (
        "vanacloig_genes_centered",
        "vanacloig",
        "test_genes",
        "gene_centered_spearman_median",
        None,
    ),
    ("growth_auroc", "ex23", "combinations", "growth_auroc", "served"),
    ("worst_scenario", "ex23", "combinations", "worst_scenario_spearman", "served"),
    ("grown_fitness", "ex23", "combinations_that_grew", "fitness_spearman", "served"),
    (
        "ff_x_aa_grid",
        "isobole_ex26",
        "interior_cells",
        "mean_observed_minus_predicted",
        "served",
    ),
    (
        "fa_x_aa_grid",
        "isobole_ex27",
        "interior_cells",
        "mean_observed_minus_predicted",
        "served",
    ),
    (
        "hmf_x_aa_grid",
        "isobole_ex28",
        "interior_cells",
        "mean_observed_minus_predicted",
        "served",
    ),
    (
        "hoepfner_compound",
        "hoepfner_hop",
        "heldout_compound",
        "centered_spearman_median",
        None,
    ),
    ("hoepfner_dose", "hoepfner_hop", "heldout_dose", "centered_spearman_median", None),
    (
        "hillenmeyer_compound",
        "hillenmeyer_het",
        "heldout_compound",
        "centered_spearman_median",
        None,
    ),
    (
        "wildenhain_compound",
        "wildenhain",
        "heldout_compound",
        "centered_spearman_median",
        None,
    ),
]


def arm_of(name: str) -> str:
    """The config name without its fold and seed suffixes (``wandb_view.arm_of``)."""
    return re.sub(r"_s\d+$", "", re.sub(r"_f\d+$", "", name))


def scores() -> pd.DataFrame:
    frames = []
    for machine, root in MACHINES.items():
        for path in sorted(glob.glob(osp.join(root, "*", "*_scores.csv"))):
            sweep = osp.basename(osp.dirname(path))
            if "smoke" in sweep:
                continue
            frames.append(pd.read_csv(path).assign(sweep=sweep, machine=machine))
    table = pd.concat(frames, ignore_index=True)
    table["arm"] = table["name"].map(arm_of)
    return table


def arm_table(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (machine, sweep, arm), group in table.groupby(["machine", "sweep", "arm"]):
        row: dict[str, object] = {
            "machine": machine,
            "sweep": sweep,
            "arm": arm,
            "folds": group["fold"].nunique(),
            "seeds": group.loc[
                group["member"].str.startswith("seed"), "member"
            ].nunique(),
        }
        for column, task, subset, metric, call in READOUTS:
            mask = (
                (group["task"] == task)
                & (group["subset"] == subset)
                & (group["metric"] == metric)
            )
            if call is not None:
                mask &= group["call"] == call
            per_fold = group[mask].groupby("fold")["value"].mean()
            row[column] = float(per_fold.median()) if len(per_fold) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def selection_table() -> pd.DataFrame:
    rows = []
    paths = [
        (machine, path)
        for machine, root in MACHINES.items()
        for path in sorted(glob.glob(osp.join(root, "*", "*_history.csv")))
    ]
    for machine, path in paths:
        sweep = osp.basename(osp.dirname(path))
        if "smoke" in sweep:
            continue
        history = read_history(path)
        name = osp.basename(path).replace("_history.csv", "")
        for seed, run in history.groupby("seed"):
            epochs = run["epoch"].to_numpy(dtype=float)
            steps_per_epoch = float(run["step"].max()) / float(epochs.max())
            chosen = int(run["selected_step"].iloc[0])
            at = run[run["step"] == chosen]
            row: dict[str, object] = {
                "machine": machine,
                "sweep": sweep,
                "name": name,
                "arm": arm_of(name),
                "seed": int(str(seed)),
                "epochs": int(round(float(run["epoch"].max()))),
                "selected_epoch": int(round(chosen / steps_per_epoch)),
                "val_at_selected": float(at["score/val_centered_spearman"].iloc[0]),
                "test_at_selected": float(at["score/test_centered_spearman"].iloc[0]),
                "test_best": float(run["score/test_centered_spearman"].max()),
                "test_best_epoch": int(
                    round(
                        epochs[
                            int(
                                np.argmax(
                                    run["score/test_centered_spearman"].to_numpy()
                                )
                            )
                        ]
                    )
                ),
                "test_last": float(run["score/test_centered_spearman"].iloc[-1]),
            }
            for column in [c for c in run.columns if c.startswith("test_loss/")]:
                source = column.replace("test_loss/", "")
                row[f"heldout_loss_min_epoch_{source}"] = int(
                    round(epochs[int(np.argmin(run[column].to_numpy()))])
                )
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    arms = arm_table(scores())
    arms.to_csv(osp.join(RESULTS, "r3_readout_arms.csv"), index=False)
    selection = selection_table()
    selection.to_csv(osp.join(RESULTS, "r3_readout_selection.csv"), index=False)
    pd.set_option("display.width", 250)
    print(arms.round(3).to_string(index=False))
    print()
    print(selection.round(3).to_string(index=False))
    print(f"\nwrote {osp.join(RESULTS, 'r3_readout_arms.csv')}")
    print(f"wrote {osp.join(RESULTS, 'r3_readout_selection.csv')}")


if __name__ == "__main__":
    main()
