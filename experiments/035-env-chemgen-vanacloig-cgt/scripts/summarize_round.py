# experiments/035-env-chemgen-vanacloig-cgt/scripts/summarize_round.py
# [[experiments.035-env-chemgen-vanacloig-cgt.scripts.summarize_round]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/035-env-chemgen-vanacloig-cgt/scripts/summarize_round
"""Every finished fold of every arm, beside the baselines on the same compounds.

Reads ``results/<round>/<arm>/fold<k>_seed<s>_{scores,history}.csv`` as the trainer
wrote them and ``results/baselines_same_folds.csv``. A fold counts as FINISHED only when
its history holds the run's full epoch budget, read from the largest epoch count any fold
of that arm reached; a fold still training is reported as partial and left out of the
medians, because its selected epoch can still change.

The comparison with a baseline is paired: both are scored on the same held-out
compounds, those of the arm's finished folds, so an arm with three folds done is
compared with ridge on those three folds' compounds and not on all 41.

Writes ``results/round_summary.csv`` (one row per round, arm and target) and
``results/round_per_compound.csv`` (one row per compound, with every model's score).
"""

from __future__ import annotations

import glob
import os
import os.path as osp
import re

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "035-env-chemgen-vanacloig-cgt", "results")
RUN = re.compile(r"fold(?P<fold>\d+)_seed(?P<seed>\d+)_scores\.csv$")


def load_runs() -> pd.DataFrame:
    """Test-compound scores of every fold, with how far its training got."""
    frames = []
    for path in sorted(glob.glob(osp.join(RESULTS_DIR, "*", "*", "fold*_scores.csv"))):
        match = RUN.search(path)
        assert match is not None, path
        history = pd.read_csv(path.replace("_scores.csv", "_history.csv"))
        scores = pd.read_csv(path)
        scores = scores[scores["split"] == "test"]
        frames.append(
            scores.assign(
                round=osp.basename(osp.dirname(osp.dirname(path))),
                epochs_run=len(history),
                train_mse=float(history["train/mse"].iloc[-1]),
                val_mse=float(history["val/mse"].iloc[-1]),
            )
        )
    runs = pd.concat(frames, ignore_index=True)
    budget = runs.groupby(["round", "arm"])["epochs_run"].transform("max")
    return runs.assign(epoch_budget=budget, finished=runs["epochs_run"] == budget)


def main() -> None:
    runs = load_runs()
    baselines = pd.read_csv(osp.join(RESULTS_DIR, "baselines_same_folds.csv"))
    baselines = baselines[baselines["split"] == "test"]
    reference = baselines.pivot_table(
        index=["compound", "target"], columns="model", values="spearman"
    ).rename(columns={"gene_mean": "gene_mean_spearman", "ridge": "ridge_spearman"})

    per_compound = runs.merge(
        reference.reset_index(), on=["compound", "target"], how="left", validate="m:1"
    )
    per_compound.to_csv(osp.join(RESULTS_DIR, "round_per_compound.csv"), index=False)

    rows = []
    for (rnd, arm, target), g in per_compound.groupby(["round", "arm", "target"]):
        done = g[g["finished"]]
        scored = done.dropna(subset=["spearman"])
        paired = scored.dropna(subset=["ridge_spearman"])
        rows.append(
            {
                "round": rnd,
                "arm": arm,
                "target": target,
                "epoch_budget": int(g["epoch_budget"].iloc[0]),
                "folds_finished": done["fold"].nunique(),
                "folds_partial": g.loc[~g["finished"], "fold"].nunique(),
                "compounds_scored": len(scored),
                "spearman_median": scored["spearman"].median(),
                "pearson_median": scored["pearson"].median(),
                "ceiling_median": scored["ceiling"].median(),
                "fraction_of_ceiling_median": (
                    scored["spearman"] / scored["ceiling"].where(scored["ceiling"] > 0)
                ).median(),
                "selected_epoch_median": done.drop_duplicates("fold")["epoch"].median(),
                "ridge_same_compounds": paired["ridge_spearman"].median(),
                "gene_mean_same_compounds": scored["gene_mean_spearman"].median(),
                "compounds_above_ridge": int(
                    (paired["spearman"] > paired["ridge_spearman"]).sum()
                ),
                "compounds_paired": len(paired),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(osp.join(RESULTS_DIR, "round_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
