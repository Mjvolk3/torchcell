# experiments/040-inhibitor-synergy-wetlab/scripts/wandb_view.py
# [[experiments.040-inhibitor-synergy-wetlab.scripts.wandb_view]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/040-inhibitor-synergy-wetlab/scripts/wandb_view
"""Group the mixture-round runs by arm and build the W&B Charts view for them.

``train_mixture.py`` runs one fold per W&B run and logs its curves under fold- and
seed-specific keys (``fold0_seed0/test_centered_mean``), with the sweep and config name as
the group, so no two runs share a curve key or a group page. The wet-lab scores are
already logged under shared keys at the end of every run (``ex23/combinations/
growth_auroc/served`` and the like), so the bar panels need no re-logging. For every
FINISHED run of the mixture rounds, on GilaHyper or on Delta, this script

1. sets the W&B group to the arm (the config name without its fold and fold-seed
   suffixes), so ``/groups/<arm>`` shows the arm's folds together, and writes the
   top-level config keys ``arm``, ``round``, ``host_train_``, ``dose_``, ``fold``,
   ``fold_seed_`` and ``view``;
2. resumes the run once and logs its curves again under keys every run shares, the
   mean over the run's seeds at each epoch, read from the run's own W&B history (no
   local file is needed, so Delta runs are covered): ``curve/held_out_centered_spearman``
   (the fold's test compounds), ``curve/validation_centered_spearman``,
   ``curve/gene_loss`` (masked MSE on the standardized Vanacloig response),
   ``curve/host_loss`` (MSE on the host growth records), ``curve/grad_norm``,
   ``curve/lr``, ``curve/gpu_peak_gb`` and ``curve/minutes``, with ``epoch`` as the x
   axis;
3. overwrites a SAVED workspace view whose runset is grouped by ``arm``: the wet-lab
   scores first (growth AUROC, worst-scenario and grown-fitness Spearman, the isobole
   residuals, the public centered Spearman), then the curves.

A run is re-logged once (config ``curves_relogged``); the grouping and the view are
rewritten on every call. Rerun after each job finishes.

    python experiments/040-inhibitor-synergy-wetlab/scripts/wandb_view.py
    python experiments/040-inhibitor-synergy-wetlab/scripts/wandb_view.py --view-only
"""

from __future__ import annotations

import argparse
import glob
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
EXPERIMENT = osp.join(os.environ["EXPERIMENT_ROOT"], "040-inhibitor-synergy-wetlab")
RESULTS = osp.join(EXPERIMENT, "results")
ENTITY = "zhao-group"
PROJECT = "torchcell_040-inhibitor-synergy-wetlab"
VIEW_NAME = "040 mixture rounds: arms grouped, wet-lab scores first"
VIEW_TAG = "040-mixture"
# Pinned after the first `save_as_new_view()`; None creates the view and prints its id.
VIEW_ID: str | None = "n94xqhmif86"
X = "epoch"
# every panel draws up to this many runs or groups; the W&B default is 10 and the user wants every run visible
MAX_SHOWN = 1000
# r1_mix_<arm>, r2_mix_<arm>, r3_<arm>, r3w2_<arm>, r4_<arm>; smokes are skipped by name
SWEEP = re.compile(r"^r\d+(w\d+)?_\w+$")
KEY = re.compile(r"^fold(\d+)_seed(\d+)/(\w+)$")

CURVES = {
    "curve/held_out_centered_spearman": "test_centered_mean",
    "curve/validation_centered_spearman": "val_centered_mean",
    "curve/gene_loss_standardized_mse": "gene_loss",
    "curve/host_loss_mse": "host_loss",
    # round 3: a held-out loss per source, logged every evaluation
    "curve/heldout_loss_vanacloig": "heldout_loss_vanacloig",
    "curve/heldout_loss_hoepfner_hop": "heldout_loss_hoepfner_hop",
    "curve/heldout_loss_hillenmeyer_het": "heldout_loss_hillenmeyer_het",
    "curve/heldout_loss_wildenhain": "heldout_loss_wildenhain",
    "curve/train_loss_hoepfner_hop": "loss_hoepfner_hop",
    "curve/train_loss_hillenmeyer_het": "loss_hillenmeyer_het",
    "curve/grad_norm": "grad_norm",
    "curve/lr": "lr",
    "curve/gpu_peak_gb": "gpu_peak_gb",
    "curve/minutes": "seconds",
}

FINAL = {
    "growth AUROC over the 63 combinations (served call; Loewe 0.958)": (
        "ex23/combinations/growth_auroc/served"
    ),
    "worst-scenario Spearman over all 63, no growth as 0 (Loewe 0.772)": (
        "ex23/combinations/worst_scenario_spearman/served"
    ),
    "fitness Spearman over the combinations that grew (Bliss 0.573)": (
        "ex23/combinations_that_grew/fitness_spearman/served"
    ),
    "furfural x acetic acid grid, observed minus predicted (Bliss -0.08)": (
        "isobole_ex26/interior_cells/mean_observed_minus_predicted/served"
    ),
    "formic x acetic acid grid, observed minus predicted (Bliss -0.45)": (
        "isobole_ex27/interior_cells/mean_observed_minus_predicted/served"
    ),
    "5-HMF x acetic acid grid, observed minus predicted (Bliss -0.02)": (
        "isobole_ex28/interior_cells/mean_observed_minus_predicted/served"
    ),
    "Vanacloig compound-cold centered Spearman, median (ridge 0.359)": (
        "vanacloig/test_compounds/centered_spearman_median/both"
    ),
    "Vanacloig gene-cold, gene-centered Spearman over held-out strains": (
        "vanacloig/test_genes/gene_centered_spearman_median/both"
    ),
    "Hoepfner HOP held-out COMPOUND, centered Spearman (ridge 0.189)": (
        "hoepfner_hop/heldout_compound/centered_spearman_median/both"
    ),
    "Hoepfner HOP held-out DOSE of a seen compound, centered Spearman": (
        "hoepfner_hop/heldout_dose/centered_spearman_median/both"
    ),
    "Hillenmeyer HET held-out compound, centered Spearman (ridge 0.126)": (
        "hillenmeyer_het/heldout_compound/centered_spearman_median/both"
    ),
    "Wildenhain held-out compound, centered Spearman (ridge 0.394)": (
        "wildenhain/heldout_compound/centered_spearman_median/both"
    ),
    "growth AUROC, software call": "ex23/combinations/growth_auroc/software",
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
                "held-out and validation on one axis",
                [
                    "curve/held_out_centered_spearman",
                    "curve/validation_centered_spearman",
                ],
            ),
        ],
    ),
    (
        "2 held-out loss per source, by epoch (round 3; where each one bottoms is that "
        "source's training length)",
        [
            ("Vanacloig held-out loss", ["curve/heldout_loss_vanacloig"]),
            ("Hoepfner HOP held-out loss", ["curve/heldout_loss_hoepfner_hop"]),
            ("Hillenmeyer HET held-out loss", ["curve/heldout_loss_hillenmeyer_het"]),
            ("Wildenhain held-out loss", ["curve/heldout_loss_wildenhain"]),
            (
                "auxiliary TRAINING losses (the ones that were still falling at 50)",
                ["curve/train_loss_hoepfner_hop", "curve/train_loss_hillenmeyer_het"],
            ),
        ],
    ),
    (
        "3 loss",
        [
            (
                "gene loss, by epoch (masked MSE on the standardized response)",
                ["curve/gene_loss_standardized_mse"],
            ),
            (
                "host loss, by epoch (MSE on the host growth records)",
                ["curve/host_loss_mse"],
            ),
            (
                "gene and host loss on one axis",
                ["curve/gene_loss_standardized_mse", "curve/host_loss_mse"],
            ),
        ],
    ),
    (
        "4 optimization",
        [
            ("gradient norm before clipping", ["curve/grad_norm"]),
            ("learning rate", ["curve/lr"]),
        ],
    ),
    (
        "5 bookkeeping",
        [
            ("peak GPU memory allocated (GB)", ["curve/gpu_peak_gb"]),
            ("wall time per seed (minutes)", ["curve/minutes"]),
        ],
    ),
]


def machines_from_logs() -> dict[str, str]:
    """W&B run id to the machine whose slurm log announced it.

    GilaHyper logs sit in ``slurm/output/``; Delta's are rsynced to
    ``slurm/output-delta/`` (``rsync delta:.../slurm/output/ slurm/output-delta/``).
    """
    out: dict[str, str] = {}
    for machine, sub in (("gilahyper", "output"), ("delta", "output-delta")):
        for path in glob.glob(osp.join(EXPERIMENT, "slurm", sub, "*.out")):
            with open(path, errors="replace") as fh:
                for rid in re.findall(r"/runs/([a-z0-9]{8})", fh.read()):
                    out[rid] = machine
    return out


MACHINE_OF_RUN = machines_from_logs()


def arm_of(name: str) -> str:
    """The config name without its fold and fold-seed suffixes."""
    name = re.sub(r"_f\d+$", "", name)
    return re.sub(r"_s\d+$", "", name)


def curves(history: pd.DataFrame) -> pd.DataFrame:
    """Per-epoch means over the run's seeds, under the shared key names."""
    rows = []
    for column in history.columns:
        m = KEY.match(column)
        if not m or m.group(3) not in CURVES.values():
            continue
        fold, seed, metric = m.groups()
        epoch_col = f"fold{fold}_seed{seed}/epoch"
        if epoch_col not in history:
            continue
        pair = history[[epoch_col, column]].dropna()
        for epoch, value in zip(pair[epoch_col], pair[column], strict=True):
            rows.append({"epoch": int(round(epoch)), "metric": metric, "value": value})
    long = pd.DataFrame(rows)
    assert len(long), "no fold/seed curve keys in the run's history"
    mean = long.groupby(["epoch", "metric"])["value"].mean().unstack("metric")
    curve = pd.DataFrame(index=mean.index)
    for key, column in CURVES.items():
        if column in mean:
            curve[key] = mean[column] / 60.0 if column == "seconds" else mean[column]
    return curve


def label_and_relog(api: wandb.Api) -> dict[str, dict[str, object]]:
    """Group every finished mixture run by arm and log its shared-key curves."""
    groups: dict[str, dict[str, object]] = {}
    for run in api.runs(f"{ENTITY}/{PROJECT}", filters={"state": "finished"}):
        sweeps = [t for t in run.tags if SWEEP.match(t) and "smoke" not in t]
        if not sweeps:
            continue  # smokes carry no round tag, or a smoke tag
        sweep = sweeps[0]
        if "ex23/combinations/growth_auroc/served" not in run.summary:
            continue  # a run that died before scoring
        arm = arm_of(run.name)
        rnd = sweep.split("_")[0]
        if not run.config.get("curves_relogged"):
            history = run.history(samples=100000, pandas=True)
            curve = curves(history)
            live = wandb.init(
                entity=ENTITY,
                project=PROJECT,
                id=run.id,
                resume="must",
                dir=osp.join(
                    DATA_ROOT, "wandb-experiments", "040-inhibitor-synergy-wetlab"
                ),
                settings=wandb.Settings(silent=True),
            )
            live.define_metric(X)
            live.define_metric("curve/*", step_metric=X)
            for epoch, row in curve.iterrows():
                live.log(
                    {X: int(epoch)}
                    | {k: float(v) for k, v in row.items() if pd.notna(v)}
                )
            live.config.update({"curves_relogged": True}, allow_val_change=True)
            live.finish()
            run = api.run(f"{ENTITY}/{PROJECT}/{run.id}")
        run.group = arm
        run.config["arm"] = arm
        run.config["round"] = rnd
        run.config["host_train_"] = run.config.get("host_train")
        run.config["dose_"] = run.config.get("dose")
        run.config["fold"] = int(run.config.get("folds", [0])[0])
        run.config["fold_seed_"] = int(run.config.get("fold_seed", 0))
        run.config["view"] = VIEW_TAG
        # The same config at the same seed differs between machines by up to 0.23, so
        # the machine is a grouping key worth having. It is read from the slurm logs,
        # which print the run URL at start: resuming a run to re-log its curves
        # rewrites its W&B metadata (host, program, startedAt) with the machine that
        # resumed it, so the metadata is not a record of where it trained.
        run.config["machine"] = MACHINE_OF_RUN.get(run.id, "unknown")
        run.update()
        entry = groups.setdefault(
            arm,
            {"url": f"https://wandb.ai/{ENTITY}/{PROJECT}/groups/{arm}", "run_ids": []},
        )
        entry["run_ids"].append(run.id)  # type: ignore[union-attr]
        print(f"{arm:34s} {run.name:36s} {run.id}", flush=True)
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
            name="0 wet-lab scores per arm (median of the folds is the number the note "
            "reports; the model-free bar is in each title)",
            is_open=True,
            layout_settings=ws.SectionLayoutSettings(columns=2, rows=1),
            panels=[
                wr.BarPlot(
                    title=title,
                    metrics=[metric],
                    max_runs_to_show=MAX_SHOWN,
                    max_bars_to_show=MAX_SHOWN,
                    layout=wr.Layout(w=12, h=8),
                )
                for title, metric in FINAL.items()
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
    parser.add_argument(
        "--relabel-only",
        action="store_true",
        help="rewrite every run's group and config keys (machine, arm) and stop",
    )
    args = parser.parse_args()
    if args.view_only:
        print(populate_view())
        return
    if args.relabel_only:
        api = wandb.Api()
        n = 0
        for run in api.runs(f"{ENTITY}/{PROJECT}", filters={"config.view": VIEW_TAG}):
            run.config["machine"] = MACHINE_OF_RUN.get(run.id, "unknown")
            run.update()
            n += 1
        print(f"relabeled {n} runs; machines known for {len(MACHINE_OF_RUN)} run ids")
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
