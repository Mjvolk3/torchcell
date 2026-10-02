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

import json
import os
import os.path as osp
import re

import pandas as pd
import wandb
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws
from dotenv import load_dotenv

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT = osp.join(os.environ["EXPERIMENT_ROOT"], "035-env-chemgen-vanacloig-cgt")
RESULTS = osp.join(EXPERIMENT, "results")
ENTITY = "zhao-group"
PROJECT = "torchcell_035-env-chemgen-vanacloig-cgt"
VIEW_NAME = "035 rounds 8 to 11: cell graph transformer arms grouped"
VIEW_TAG = "035-r8-r11"
# Pinned after the first `save_as_new_view()`; None creates the view and prints its id.
VIEW_ID: str | None = "sy9905pud6q"
X = "epoch"
SWEEP = re.compile(r"^(r8_(small|deep|mid)_[a-z]|r9_\w+|r10_\w+|r11_\w+)$")

PROTOCOL = {
    "r8": "fit on training compounds, epoch picked on 4 validation compounds, 1 seed",
    "r9": "fit on the non-test pool, fixed 50 epochs keeping the last, 3 seeds averaged",
    "r10": "fit on the non-test pool, fixed 50 epochs keeping the last, 3 seeds averaged",
    "r11": "fit on the non-test pool, fixed 50 epochs keeping the last, 3 seeds averaged",
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
                "validation centered Spearman, by epoch (in-sample for pool-fit arms)",
                ["curve/validation_centered_spearman"],
            ),
            (
                "held-out and validation centered Spearman",
                [
                    "curve/held_out_centered_spearman",
                    "curve/validation_centered_spearman",
                ],
            ),
        ],
    ),
    (
        "2 training side",
        [
            (
                "train loss (MSE on the standardized response)",
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


def curves(history: pd.DataFrame) -> pd.DataFrame:
    """Per-epoch means over the run's seeds, under the shared key names."""
    history = history.assign(epoch=history["epoch"].round().astype(int))
    mean = history.groupby("epoch").mean(numeric_only=True)
    return pd.DataFrame(
        {
            "curve/held_out_centered_spearman": mean["test_centered_mean"],
            "curve/validation_centered_spearman": mean["val_centered_mean"],
            "curve/train_loss_standardized_mse": mean["train_loss"],
            "curve/graph_prior_penalty": mean["penalty"],
            "curve/grad_norm": mean["grad_norm"],
            "curve/lr": mean["lr"],
            "curve/gpu_peak_gb": mean["gpu_peak_gb"],
            "curve/minutes": mean["seconds"] / 60,
        }
    )


def label_and_relog(api: wandb.Api) -> dict[str, dict[str, object]]:
    """Group every finished round 8 to 11 run by arm and log its shared-key curves."""
    ridge = ridge_reference()
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
        }
        if not run.config.get("curves_relogged"):
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
            for epoch, row in curves(pd.read_csv(history_path)).iterrows():
                live.log({X: int(epoch)} | {k: float(v) for k, v in row.items()})
            live.summary.update(final)
            live.config.update({"curves_relogged": True}, allow_val_change=True)
            live.finish()
            run = api.run(f"{ENTITY}/{PROJECT}/{run.id}")
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
                    layout=wr.Layout(w=12, h=8),
                ),
                wr.BarPlot(
                    title="median centered Spearman on the held-out compounds",
                    metrics=["final/median_centered_spearman_held_out"],
                    layout=wr.Layout(w=12, h=8),
                ),
            ],
        ),
    )
    settings = ws.WorkspaceSettings(
        x_axis=X, smoothing_type="none", max_runs=200, sort_panels_alphabetically=False
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
