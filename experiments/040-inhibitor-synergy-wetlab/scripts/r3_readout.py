# experiments/040-inhibitor-synergy-wetlab/scripts/r3_readout.py
# [[experiments.040-inhibitor-synergy-wetlab.scripts.r3_readout]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/040-inhibitor-synergy-wetlab/scripts/r3_readout
"""Read every scored mixture arm on this machine into two tables.

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

import numpy as np
import pandas as pd

EXPERIMENT = osp.dirname(osp.dirname(osp.abspath(__file__)))
RESULTS = osp.join(EXPERIMENT, "results")
MIXTURE = osp.join(RESULTS, "mixture")

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
    for path in sorted(glob.glob(osp.join(MIXTURE, "*", "*_scores.csv"))):
        sweep = osp.basename(osp.dirname(path))
        if "smoke" in sweep:
            continue
        frames.append(pd.read_csv(path).assign(sweep=sweep))
    table = pd.concat(frames, ignore_index=True)
    table["arm"] = table["name"].map(arm_of)
    return table


def arm_table(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (sweep, arm), group in table.groupby(["sweep", "arm"]):
        row: dict[str, object] = {
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
    for path in sorted(glob.glob(osp.join(MIXTURE, "*", "*_history.csv"))):
        sweep = osp.basename(osp.dirname(path))
        if "smoke" in sweep:
            continue
        history = pd.read_csv(path)
        name = osp.basename(path).replace("_history.csv", "")
        for seed, run in history.groupby("seed"):
            epochs = run["epoch"].to_numpy(dtype=float)
            steps_per_epoch = float(run["step"].max()) / float(epochs.max())
            chosen = int(run["selected_step"].iloc[0])
            at = run[run["step"] == chosen]
            row: dict[str, object] = {
                "sweep": sweep,
                "name": name,
                "arm": arm_of(name),
                "seed": int(str(seed)),
                "epochs": int(round(float(run["epoch"].max()))),
                "selected_epoch": int(round(chosen / steps_per_epoch)),
                "val_at_selected": float(at["val_centered_mean"].iloc[0]),
                "test_at_selected": float(at["test_centered_mean"].iloc[0]),
                "test_best": float(run["test_centered_mean"].max()),
                "test_best_epoch": int(
                    round(epochs[int(np.argmax(run["test_centered_mean"].to_numpy()))])
                ),
                "test_last": float(run["test_centered_mean"].iloc[-1]),
            }
            for column in [c for c in run.columns if c.startswith("heldout_loss_")]:
                source = column.replace("heldout_loss_", "")
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
