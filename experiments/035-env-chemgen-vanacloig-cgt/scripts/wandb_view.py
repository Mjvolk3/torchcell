# experiments/035-env-chemgen-vanacloig-cgt/scripts/wandb_view.py
# [[experiments.035-env-chemgen-vanacloig-cgt.scripts.wandb_view]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/035-env-chemgen-vanacloig-cgt/scripts/wandb_view
"""Group the round 8 to 11 runs by arm and build the W&B Charts view for them.

``train_factorized.py`` runs one fold per W&B run and logs its curves under fold- and
seed-specific keys (``fold0_seed0/test_centered_mean``), with the run's own name as the
group, so no two runs share a metric key or a group page. For every FINISHED run of
rounds 8 to 11 this script

1. sets the W&B group to the arm (the config name without its fold, fold-seed and
   seed-block suffixes), so ``/groups/<arm>`` shows the arm's folds and seeds together,
   and writes the top-level config keys ``arm``, ``round``, ``protocol``, ``fold``,
   ``fold_seed`` and ``view``;
2. resumes the run once and logs its curves again under keys every run shares, the mean
   over the run's seeds at each epoch, read from the run's ``_history.csv``:
   ``curve/held_out_centered_spearman`` (mean over the fold's held-out compounds of the
   centered Spearman across strains), ``curve/validation_centered_spearman`` (the 4
   validation compounds; in-sample for arms fit on the pool),
   ``curve/train_loss_standardized_mse``, ``curve/graph_prior_penalty``,
   ``curve/grad_norm``, ``curve/lr``, ``curve/gpu_peak_gb``, ``curve/minutes``, with
   ``epoch`` as the x axis, plus ``final/*`` summaries against nested ridge on the run's
   own held-out compounds;
3. overwrites a SAVED workspace view whose runset is grouped by ``arm``.

A run is re-logged once (config ``curves_relogged``); the grouping and the view are
rewritten on every call. Rerun after each job finishes.

    python experiments/035-env-chemgen-vanacloig-cgt/scripts/wandb_view.py
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import re
import sys

import numpy as np
import pandas as pd
import wandb
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws
from dotenv import load_dotenv

sys.path.insert(0, osp.dirname(__file__))
from train_factorized import CELL_TABLE, EMBEDDING_DIR, PREDICTIONS  # noqa: E402
from vanacloig_data import load_cells, make_folds  # noqa: E402

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT = osp.join(os.environ["EXPERIMENT_ROOT"], "035-env-chemgen-vanacloig-cgt")
RESULTS = osp.join(EXPERIMENT, "results")
ENTITY = "zhao-group"
PROJECT = "torchcell_035-env-chemgen-vanacloig-cgt"
VIEW_NAME = "035 rounds 8 to 12: cell graph transformer arms grouped"
VIEW_TAG = "035-r8-r11"
# Pinned after the first `save_as_new_view()`; None creates the view and prints its id.
VIEW_ID: str | None = "sy9905pud6q"
X = "epoch"
# every panel draws up to this many runs or groups; the W&B default is 10
MAX_SHOWN = 100
SWEEP = re.compile(
    r"^(r8_(small|deep|mid)_[a-z]|r9_\w+|r10_\w+|r11_\w+|r12_\w+|r13_\w+|r14_\w+)$"
)

PROTOCOL = {
    "r8": "fit on training compounds, epoch picked on 4 validation compounds, 1 seed",
    "r9": "fit on the non-test pool, fixed 50 epochs keeping the last, 3 seeds averaged",
    "r10": "fit on the non-test pool, fixed 50 epochs keeping the last, 3 seeds averaged",
    "r11": "fit on the non-test pool, fixed 50 epochs keeping the last, 3 seeds averaged",
    "r12": "fit on training compounds, epoch picked on 4 validation compounds, 1 seed",
    "r13": "fit on the non-test pool, fixed 50 epochs keeping the last, 3 seeds averaged",
    "r14": "fit on the non-test pool, fixed 50 epochs keeping the last, 3 seeds averaged",
}

LOSS_COLUMNS = {
    "curve/train_loss_full_standardized_mse": "train_loss_full",
    "curve/held_out_loss_standardized_mse": "test_loss",
}
# The 4 validation compounds are out of sample only for an arm fit on the training
# compounds. An arm fit on the pool has trained on them, and its "validation" score (about
# 0.7 centered Spearman) is a training score, so these keys are written for train-fit
# arms only and the pool-fit arms are simply absent from the validation panels.
VALIDATION_COLUMNS = {
    "curve/validation_out_of_sample_centered_spearman": "val_centered_mean",
    "curve/validation_out_of_sample_loss_standardized_mse": "val_loss",
}

SECTIONS: list[tuple[str, list[tuple[str, list[str]]]]] = [
    (
        "1 held-out compounds (centered Spearman across strains, mean over compounds)",
        [
            (
                "held-out centered Spearman, by epoch",
                ["curve/held_out_centered_spearman"],
            ),
            (
                "validation centered Spearman, by epoch (4 compounds never fitted; "
                "arms fit on training compounds only)",
                ["curve/validation_out_of_sample_centered_spearman"],
            ),
            (
                "held-out and out-of-sample validation centered Spearman",
                [
                    "curve/held_out_centered_spearman",
                    "curve/validation_out_of_sample_centered_spearman",
                ],
            ),
        ],
    ),
    (
        "2 loss (MSE on the standardized response, every strain; per-epoch curves exist "
        "for runs from round 11 on)",
        [
            (
                "validation loss, by epoch (4 compounds never fitted; arms fit on "
                "training compounds only)",
                ["curve/validation_out_of_sample_loss_standardized_mse"],
            ),
            (
                "train loss over all fitted compounds, by epoch",
                ["curve/train_loss_full_standardized_mse"],
            ),
            (
                "held-out loss, by epoch (the fold's test compounds)",
                ["curve/held_out_loss_standardized_mse"],
            ),
            (
                "train, validation and held-out loss on one axis",
                [
                    "curve/train_loss_full_standardized_mse",
                    "curve/validation_out_of_sample_loss_standardized_mse",
                    "curve/held_out_loss_standardized_mse",
                ],
            ),
            (
                "train loss of the last batch of the epoch (every run)",
                ["curve/train_loss_standardized_mse"],
            ),
            ("graph prior penalty (unweighted KL sum)", ["curve/graph_prior_penalty"]),
        ],
    ),
    (
        "3 optimization",
        [
            ("gradient norm before clipping at 10", ["curve/grad_norm"]),
            ("learning rate (head group)", ["curve/lr"]),
        ],
    ),
    (
        "4 bookkeeping",
        [
            ("peak GPU memory allocated (GB)", ["curve/gpu_peak_gb"]),
            ("wall time per seed (minutes)", ["curve/minutes"]),
        ],
    ),
]


def arm_of(name: str) -> str:
    """The config name without its fold, fold-seed and seed-block suffixes."""
    name = re.sub(r"_f\d$", "", name)
    name = re.sub(r"_fs\d$", "", name)
    return re.sub(r"_s\d{3}$", "", name)


def ridge_reference() -> pd.Series:
    """Nested ridge's centered Spearman, indexed by (fold seed, compound)."""
    ladder = pd.read_csv(osp.join(RESULTS, "ladder", "ladder_r2_scores.csv"))
    ridge = ladder[
        (ladder["model"] == "krr")
        & (ladder["kernel"] == "linear:fcfp4_count")
        & (ladder["target"] == "centered")
    ]
    return ridge.set_index(["fold_seed", "compound"])["spearman"]


def curves(history: pd.DataFrame, fit_on: str) -> pd.DataFrame:
    """Per-epoch means over the run's seeds, under the shared key names."""
    history = history.assign(epoch=history["epoch"].round().astype(int))
    mean = history.groupby("epoch").mean(numeric_only=True)
    curve = pd.DataFrame(
        {
            "curve/held_out_centered_spearman": mean["test_centered_mean"],
            "curve/train_loss_standardized_mse": mean["train_loss"],
            "curve/graph_prior_penalty": mean["penalty"],
            "curve/grad_norm": mean["grad_norm"],
            "curve/lr": mean["lr"],
            "curve/gpu_peak_gb": mean["gpu_peak_gb"],
            "curve/minutes": mean["seconds"] / 60,
        }
    )
    # the losses over every strain are logged per epoch from round 11 on; a history
    # written before that has no such columns and the run gets final values only
    columns = LOSS_COLUMNS | (VALIDATION_COLUMNS if fit_on == "train" else {})
    for key, column in columns.items():
        if column in mean:
            curve[key] = mean[column]
    return curve


def final_losses(
    cells, y: np.ndarray, sweep: str, name: str, fold: int, fold_seed: int, fit_on: str
) -> dict[str, float]:
    """Standardized MSE of the run's saved (seed-averaged) prediction, per compound set."""
    split = make_folds(len(cells.compounds), 5, 4, fold_seed)[fold]
    fitted = split.train if fit_on == "train" else sorted(split.train + split.val)
    prediction = np.load(
        osp.join(PREDICTIONS, sweep, f"{name}_fold{fold}_seed{fold_seed}.npy")
    ).astype(np.float64)
    residual = (prediction - y) / np.nanstd(y[:, fitted])

    def mse(columns: list[int]) -> float:
        return float(np.nanmean(residual[:, columns] ** 2))

    final = {
        "final/train_loss_standardized_mse": mse(fitted),
        "final/held_out_loss_standardized_mse": mse(split.test),
    }
    if fit_on == "train":
        final["final/validation_out_of_sample_loss_standardized_mse"] = mse(split.val)
    return final


def label_and_relog(api: wandb.Api) -> dict[str, dict[str, object]]:
    """Group every finished round 8 to 11 run by arm and log its shared-key curves."""
    ridge = ridge_reference()
    cells = load_cells(CELL_TABLE, osp.join(EMBEDDING_DIR, "fcfp4_count.npz"))
    y = cells.matrix(cells.response)
    groups: dict[str, dict[str, object]] = {}
    for run in api.runs(f"{ENTITY}/{PROJECT}", filters={"state": "finished"}):
        sweeps = [t for t in run.tags if SWEEP.match(t)]
        if not sweeps:
            continue
        sweep = sweeps[0]
        history_path = osp.join(RESULTS, "factorized", sweep, f"{run.name}_history.csv")
        scores_path = history_path.replace("_history.csv", "_scores.csv")
        if not osp.exists(scores_path):
            continue  # a run of a cancelled job that was later rerun under this name
        arm = arm_of(run.name)
        rnd = sweep.split("_")[0]
        scores = pd.read_csv(scores_path)
        ensemble = scores[
            (scores["member"] == "ensemble") & (scores["target"] == "centered")
        ]
        fold_seed = int(ensemble["fold_seed"].iloc[0])
        reference = ridge.loc[[(fold_seed, c) for c in ensemble["compound"]]].to_numpy()
        final = {
            "final/median_centered_spearman_held_out": float(
                ensemble["spearman"].median()
            ),
            "final/ridge_median_centered_spearman_same_compounds": float(
                pd.Series(reference).median()
            ),
            "final/mean_paired_diff_centered_spearman_vs_ridge": float(
                (ensemble["spearman"].to_numpy() - reference).mean()
            ),
            "final/held_out_compounds": len(ensemble),
        } | final_losses(
            cells,
            y,
            sweep,
            run.name,
            int(ensemble["fold"].iloc[0]),
            fold_seed,
            run.config.get("fit_on", "train"),
        )
        fit_on = run.config.get("fit_on", "train")
        first = not run.config.get("curves_relogged")
        # a run re-logged before the validation keys were split by fit gets them once
        split_validation = (
            not first and fit_on == "train" and not run.config.get("validation_split")
        )
        if first or split_validation:
            live = wandb.init(
                entity=ENTITY,
                project=PROJECT,
                id=run.id,
                resume="must",
                dir=osp.join(
                    DATA_ROOT, "wandb-experiments", "035-env-chemgen-vanacloig-cgt"
                ),
                settings=wandb.Settings(silent=True),
            )
            live.define_metric(X)
            live.define_metric("curve/*", step_metric=X)
            curve = curves(pd.read_csv(history_path), fit_on)
            if split_validation:
                curve = curve[[k for k in VALIDATION_COLUMNS if k in curve]]
            for epoch, row in curve.iterrows():
                live.log({X: int(epoch)} | {k: float(v) for k, v in row.items()})
            live.summary.update(final)
            live.config.update(
                {"curves_relogged": True, "validation_split": True},
                allow_val_change=True,
            )
            live.finish()
            run = api.run(f"{ENTITY}/{PROJECT}/{run.id}")
        for key, value in final.items():
            run.summary[key] = value
        run.summary.update()
        run.group = arm
        run.config["arm"] = arm
        run.config["round"] = rnd
        run.config["protocol"] = PROTOCOL[rnd]
        run.config["fold"] = int(ensemble["fold"].iloc[0])
        run.config["fold_seed_"] = fold_seed
        run.config["view"] = VIEW_TAG
        run.update()
        entry = groups.setdefault(
            arm,
            {"url": f"https://wandb.ai/{ENTITY}/{PROJECT}/groups/{arm}", "run_ids": []},
        )
        entry["run_ids"].append(run.id)  # type: ignore[union-attr]
        print(f"{arm:20s} {run.name:32s} {run.id}", flush=True)
    return groups


def populate_view() -> str:
    """Overwrite (or create) the saved Charts view grouped by arm."""
    sections = [
        ws.Section(
            name=name,
            is_open=True,
            layout_settings=ws.SectionLayoutSettings(columns=3, rows=1),
            panel_settings=ws.SectionPanelSettings(x_axis=X, smoothing_type="none"),
            panels=[
                wr.LinePlot(
                    x=X,
                    y=keys,
                    title=title,
                    title_x="epoch",
                    max_runs_to_show=MAX_SHOWN,
                    layout=wr.Layout(w=8, h=7),
                )
                for title, keys in panels
            ],
        )
        for name, panels in SECTIONS
    ]
    sections.insert(
        0,
        ws.Section(
            name="0 final score per arm (held-out compounds)",
            is_open=True,
            layout_settings=ws.SectionLayoutSettings(columns=2, rows=1),
            panels=[
                wr.BarPlot(
                    title="mean paired difference in centered Spearman vs nested ridge",
                    metrics=["final/mean_paired_diff_centered_spearman_vs_ridge"],
                    max_runs_to_show=MAX_SHOWN,
                    max_bars_to_show=MAX_SHOWN,
                    layout=wr.Layout(w=12, h=8),
                ),
                wr.BarPlot(
                    title="median centered Spearman on the held-out compounds",
                    metrics=["final/median_centered_spearman_held_out"],
                    max_runs_to_show=MAX_SHOWN,
                    max_bars_to_show=MAX_SHOWN,
                    layout=wr.Layout(w=12, h=8),
                ),
                wr.BarPlot(
                    title="validation loss of the saved prediction (standardized MSE; "
                    "4 compounds never fitted, train-fit arms only)",
                    metrics=["final/validation_out_of_sample_loss_standardized_mse"],
                    max_runs_to_show=MAX_SHOWN,
                    max_bars_to_show=MAX_SHOWN,
                    layout=wr.Layout(w=12, h=8),
                ),
                wr.BarPlot(
                    title="held-out loss of the saved prediction (standardized MSE)",
                    metrics=["final/held_out_loss_standardized_mse"],
                    max_runs_to_show=MAX_SHOWN,
                    max_bars_to_show=MAX_SHOWN,
                    layout=wr.Layout(w=12, h=8),
                ),
                wr.BarPlot(
                    title="train loss of the saved prediction (standardized MSE)",
                    metrics=["final/train_loss_standardized_mse"],
                    max_runs_to_show=MAX_SHOWN,
                    max_bars_to_show=MAX_SHOWN,
                    layout=wr.Layout(w=12, h=8),
                ),
            ],
        ),
    )
    settings = ws.WorkspaceSettings(
        x_axis=X,
        smoothing_type="none",
        max_runs=MAX_SHOWN,
        sort_panels_alphabetically=False,
    )
    runset_settings = ws.RunsetSettings(
        filters=[ws.Config("view") == VIEW_TAG],
        groupby=[ws.Config("arm")],
        order=[ws.Ordering(ws.Metric("Name"), ascending=True)],
    )
    if VIEW_ID is None:
        view = ws.Workspace(
            entity=ENTITY,
            project=PROJECT,
            name=VIEW_NAME,
            sections=sections,
            settings=settings,
            runset_settings=runset_settings,
        )
        view.save_as_new_view()
        print(f"NEW saved view: {view.url}\n  pin its nw= id into VIEW_ID")
        return str(view.url)
    view = ws.Workspace.from_url(f"https://wandb.ai/{ENTITY}/{PROJECT}?nw={VIEW_ID}")
    view.name = VIEW_NAME
    view.sections = sections
    view.settings = settings
    view.runset_settings = runset_settings
    view.save()
    return str(view.url)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--view-only",
        action="store_true",
        help="rewrite the saved view's panels without touching the runs",
    )
    args = parser.parse_args()
    if args.view_only:
        print(populate_view())
        return
    api = wandb.Api()
    groups = label_and_relog(api)
    url = populate_view()
    with open(osp.join(RESULTS, "wandb_views.json"), "w") as f:
        json.dump({"view": url, "groups": groups}, f, indent=2)
    print(url)
    for arm, entry in sorted(groups.items()):
        print(f"{arm}: {len(entry['run_ids'])} runs  {entry['url']}")  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
