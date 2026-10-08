# experiments/030-solid-growth-multi/scripts/wandb_view_030.py
# [[experiments.030-solid-growth-multi.scripts.wandb_view_030]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/030-solid-growth-multi/scripts/wandb_view_030
"""Label the 030 per-entry runs by arm and seed and curate a grouped W&B Charts view.

The 025 script (experiments/025-solid-growth/scripts/random_split_wandb_view.py) pointed
at the 030 project. Runs reach W&B by `wandb sync` from the IGB login node, which resets
API-written names and tags, so this runs AFTER every sync:

1. ``label_runs``: every run whose tags name a campaign config is renamed
   ``<arm>_seed<k>[_rank<r>]`` (rank 0 is the run carrying ``val/`` metrics), grouped
   under its arm, and given the config keys ``arm``, ``seed_``, ``split`` and ``rank0``
   the view groups on. Smoke runs carry no campaign config tag and are left alone.
2. ``populate_view``: the saved view ``VIEW_ID`` is overwritten with the sections below,
   grouped by arm, x axis epoch, no smoothing.

    python experiments/030-solid-growth-multi/scripts/wandb_view_030.py
"""

from __future__ import annotations

import wandb
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws

ENTITY = "zhao-group"
PROJECT = "torchcell_030-solid-growth-multi_equivariant_cell_graph_transformer"
VIEW_NAME = "030 per-entry dataset token: arms grouped, against the 025 S3 closure"
# Pinned after the first `save_as_new_view()`; None creates the view and prints its id.
VIEW_ID: str | None = "yja1oz2m2ml"
X = "epoch"

# Runs of jobs that never trained: 2413834 died in the sanity check, 2413837 trained one
# batch per epoch; neither was synced, listed here in case they ever are.
EXCLUDED_RUN_IDS: set[str] = set()
KEEP_STATES = {"finished", "running"}

# config tag -> arm label. The split is read from the run's split_R / split_Q tag and
# written to config["split"]; the R view filters on it, so Q arms never appear there.
ARMS = {
    "cgt_030_s3_r_tok_fit_000": "s3_holdout_table_token",
    "cgt_030_s3_r_tok_embfit_001": "s3_holdout_composite_token",
    "cgt_030_s3_r_tok_fit_002": "s3_table_token",
    "cgt_030_s3_r_tok_embfit_003": "s3_composite_token",
    "cgt_030_s3_q_tok_embfit_004": "s3q_composite_token",
}
# The Q split gets its own saved view: Q and R are different questions and are never
# compared in one panel. Pinned after the first save_as_new_view().
Q_VIEW_NAME = "030 query-pair-disjoint: closure with the token"
Q_VIEW_ID: str | None = None


def _split_of(run: wandb.apis.public.Run) -> str:
    return "Q" if "split_Q" in run.tags else "R"

# The Kuzmin screens are the validation and test sources (the pinned triples); the
# Costanzo tokens carry most of the training rows.
TOKENS = [
    "TmiKuzmin2018Dataset",
    "TmiKuzmin2020Dataset",
    "DmiKuzmin2018Dataset",
    "DmiKuzmin2020Dataset",
    "DmiCostanzo2016Dataset",
]


def _line(y: list[str], title: str, w: int = 6, h: int = 6) -> wr.LinePlot:
    return wr.LinePlot(
        x=X, y=y, title=title, title_x="epoch", layout=wr.Layout(w=w, h=h)
    )


# Sections in rank order. Each (title, panels); a panel is (y keys, title).
SECTIONS: list[tuple[str, list[tuple[list[str], str]]]] = [
    (
        "1 interaction on the held-out triples, Pearson across arms",
        [
            (["val/gene_interaction/Pearson"], "val interaction Pearson"),
            (
                ["train/gene_interaction/Pearson", "val/gene_interaction/Pearson"],
                "train and val interaction Pearson",
            ),
            (["train/gene_interaction/Pearson"], "train interaction Pearson"),
            (["val/gene_interaction/MSE"], "val interaction MSE"),
        ],
    ),
    (
        "2 fitness",
        [
            (["val/fitness/Pearson"], "val fitness Pearson"),
            (
                ["train/fitness/Pearson", "val/fitness/Pearson"],
                "train and val fitness Pearson",
            ),
            (["train/fitness/Pearson"], "train fitness Pearson"),
            (["train/fitness_loss", "val/fitness_loss"], "train and val fitness loss"),
        ],
    ),
    (
        "3 per label type: smf, dmf, tmf fitness and dmi, tmi interaction (val is triples only)",
        [
            (
                [
                    "train/fitness/smf/Pearson",
                    "train/fitness/dmf/Pearson",
                    "train/fitness/tmf/Pearson",
                    "val/fitness/tmf/Pearson",
                ],
                "fitness Pearson by type (train smf, dmf, tmf; val tmf)",
            ),
            (
                [
                    "train/gene_interaction/dmi/Pearson",
                    "train/gene_interaction/tmi/Pearson",
                    "val/gene_interaction/tmi/Pearson",
                ],
                "interaction Pearson by type (train dmi, tmi; val tmi)",
            ),
            (
                [
                    "train/fitness/smf/MSE",
                    "train/fitness/dmf/MSE",
                    "train/fitness/tmf/MSE",
                    "val/fitness/tmf/MSE",
                ],
                "fitness MSE by type",
            ),
            (
                [
                    "train/gene_interaction/dmi/MSE",
                    "train/gene_interaction/tmi/MSE",
                    "val/gene_interaction/tmi/MSE",
                ],
                "interaction MSE by type",
            ),
        ],
    ),
    (
        "4 per source token (interaction Pearson of the entry rows under each token)",
        [
            (
                [f"val/token/{t}/gene_interaction/Pearson" for t in TOKENS],
                "val interaction Pearson by token",
            ),
            (
                [f"train/token/{t}/gene_interaction/Pearson" for t in TOKENS],
                "train interaction Pearson by token",
            ),
            (
                [f"train/token/{t}/fitness/Pearson" for t in TOKENS],
                "train fitness Pearson by token",
            ),
            (
                [f"train/token/{t}/gene_interaction/n_entries" for t in TOKENS],
                "train interaction entry rows by token",
            ),
        ],
    ),
    (
        "4 per perturbation order (train; val and test are triples only)",
        [
            (
                [
                    "train/fitness/order1/Pearson",
                    "train/fitness/order2/Pearson",
                    "train/fitness/order3/Pearson",
                ],
                "train fitness Pearson by order (1, 2, 3)",
            ),
            (
                [
                    "train/gene_interaction/order2/Pearson",
                    "train/gene_interaction/order3/Pearson",
                    "val/gene_interaction/order3/Pearson",
                ],
                "interaction Pearson by order (train 2, train 3, val 3)",
            ),
            (
                [
                    "train/n_records/fitness/order1",
                    "train/n_records/fitness/order2",
                    "train/n_records/fitness/order3",
                ],
                "train records with fitness by order",
            ),
            (
                [
                    "train/n_records/gene_interaction/order2",
                    "train/n_records/gene_interaction/order3",
                ],
                "train records with interaction by order",
            ),
        ],
    ),
    (
        "5 losses",
        [
            (["train/loss"], "train loss"),
            (["train/point_loss", "val/point_loss"], "train and val point loss"),
            (
                ["train/graph_reg_loss", "val/graph_reg_loss"],
                "train and val graph penalty",
            ),
            (["train/dist_loss", "val/dist_loss"], "train and val distribution loss"),
        ],
    ),
    (
        "6 operator and gradient probe",
        [
            (
                ["train/cls_pert_strain_sd", "val/cls_pert_strain_sd"],
                "perturbed CLS across-strain sd",
            ),
            (
                [
                    "probe/grad_norm/point",
                    "probe/grad_norm/fitness",
                    "probe/grad_norm/graph_reg",
                ],
                "gradient norms by term",
            ),
            (
                ["probe/grad_ratio/graph_reg_to_point"],
                "graph penalty to point gradient ratio",
            ),
            (["val/residual_update_ratio"], "val residual update ratio"),
        ],
    ),
    (
        "7 bookkeeping",
        [
            (
                ["train/cuda_peak_allocated_gb", "val/cuda_peak_allocated_gb"],
                "peak GPU memory (GB)",
            ),
            (["learning_rate"], "learning rate"),
            (["arm/n_subset", "arm/n_holdout"], "records in the pool and held out"),
            (["arm/n_train_pinned"], "pinned train triples"),
        ],
    ),
]


def _arm_of(run: wandb.apis.public.Run) -> str | None:
    """The arm label. No epoch-budget suffix: a seed trained in pieces (the 100-epoch
    job, then checkpoint continuations with larger budgets) is ONE curve, and the
    grouped view draws it as one line only if every piece carries the same arm.
    """
    cfg = [t for t in run.tags if t in ARMS]
    if not cfg:
        return None
    return ARMS[cfg[0]]


def label_runs(api: wandb.Api) -> int:
    """Name and group every kept campaign run; write the keys the view groups on.

    Within an (arm, seed) the rank-0 runs (those carrying ``val/``) are ordered by
    creation time: the first is ``<arm>_seed<k>``, later ones are its continuations
    ``<arm>_seed<k>_cont<j>`` (each resumed from the previous piece's last checkpoint).
    Other ranks are ``_rank<i>`` in creation order; they log no validation metrics.
    """
    runs = list(api.runs(f"{ENTITY}/{PROJECT}"))
    by_arm_seed: dict[tuple[str, int], list] = {}
    for run in runs:
        if run.id in EXCLUDED_RUN_IDS or run.state not in KEEP_STATES:
            continue
        arm = _arm_of(run)
        if arm is None:
            continue
        by_arm_seed.setdefault((arm, int(run.config.get("seed", 42))), []).append(run)
    n = 0
    for (arm, seed), seed_runs in by_arm_seed.items():
        rank0_runs = sorted(
            (r for r in seed_runs if "val/gene_interaction/Pearson" in r.summary),
            key=lambda r: r.created_at,
        )
        other_runs = sorted(
            (r for r in seed_runs if "val/gene_interaction/Pearson" not in r.summary),
            key=lambda r: (r.created_at, r.id),
        )
        names = {}
        for j, run in enumerate(rank0_runs):
            names[run.id] = f"{arm}_seed{seed}" + ("" if j == 0 else f"_cont{j}")
        for i, run in enumerate(other_runs, start=1):
            names[run.id] = f"{arm}_seed{seed}_rank{i}"
        for run in seed_runs:
            rank0 = "val/gene_interaction/Pearson" in run.summary
            run.name = names[run.id]
            run.group = arm
            run.config["arm"] = arm
            run.config["seed_"] = seed
            run.config["split"] = _split_of(run)
            run.config["rank0"] = rank0
            run.config["continuation"] = (
                rank0_runs.index(run) if rank0 else None
            )
            run.update()
            n += 1
    return n


def populate_view() -> str:
    """Overwrite (or create) the saved Charts view grouped by arm."""
    sections = [
        ws.Section(
            name=name,
            is_open=True,
            layout_settings=ws.SectionLayoutSettings(columns=4, rows=1),
            panel_settings=ws.SectionPanelSettings(x_axis=X, smoothing_type="none"),
            panels=[_line(y, title) for y, title in panels],
        )
        for name, panels in SECTIONS
    ]
    settings = ws.WorkspaceSettings(
        x_axis=X, smoothing_type="none", max_runs=60, sort_panels_alphabetically=False
    )
    runset_settings = ws.RunsetSettings(
        filters=[ws.Config("split") == "R"],
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
        return view.url
    view = ws.Workspace.from_url(f"https://wandb.ai/{ENTITY}/{PROJECT}?nw={VIEW_ID}")
    view.name = VIEW_NAME
    view.sections = sections
    view.settings = settings
    view.runset_settings = runset_settings
    view.save()
    return view.url


def populate_q_view() -> str:
    """Overwrite (or create) the saved view of the query-pair-disjoint arms."""
    sections = [
        ws.Section(
            name=name,
            is_open=True,
            layout_settings=ws.SectionLayoutSettings(columns=4, rows=1),
            panel_settings=ws.SectionPanelSettings(x_axis=X, smoothing_type="none"),
            panels=[_line(y, title) for y, title in panels],
        )
        for name, panels in SECTIONS
    ]
    settings = ws.WorkspaceSettings(
        x_axis=X, smoothing_type="none", max_runs=60, sort_panels_alphabetically=False
    )
    runset_settings = ws.RunsetSettings(
        filters=[ws.Config("split") == "Q"],
        groupby=[ws.Config("arm")],
        order=[ws.Ordering(ws.Metric("Name"), ascending=True)],
    )
    if Q_VIEW_ID is None:
        view = ws.Workspace(
            entity=ENTITY,
            project=PROJECT,
            name=Q_VIEW_NAME,
            sections=sections,
            settings=settings,
            runset_settings=runset_settings,
        )
        view.save_as_new_view()
        print(f"NEW saved Q view: {view.url}\n  pin its nw= id into Q_VIEW_ID")
        return view.url
    view = ws.Workspace.from_url(f"https://wandb.ai/{ENTITY}/{PROJECT}?nw={Q_VIEW_ID}")
    view.name = Q_VIEW_NAME
    view.sections = sections
    view.settings = settings
    view.runset_settings = runset_settings
    view.save()
    return view.url


def main() -> None:
    api = wandb.Api()
    print(f"labeled {label_runs(api)} runs")
    print(populate_view())
    print(populate_q_view())


if __name__ == "__main__":
    main()
