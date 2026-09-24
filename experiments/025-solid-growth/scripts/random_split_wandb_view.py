# experiments/025-solid-growth/scripts/random_split_wandb_view.py
# [[experiments.025-solid-growth.scripts.s3_closure_readout]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/025-solid-growth/scripts/random_split_wandb_view
"""Label the random-split (R) runs by arm and seed and curate a grouped W&B Charts view.

The R-split counterpart of ``disjoint_embedding_wandb_view.py``: the S0 arms (joint
fitness, control, graph-regularization ladder, hard mask) and the S3 closure cell all
train on the 010 random split and validate on the pinned triples, so they belong in one
view grouped by arm with Pearson compared across the groups. This script

1. names every kept run ``<arm>_seed<k>`` (``_rank<r>`` for the non-rank-0 ranks of a
   4-GPU job), sets ``run.group = arm`` so each arm has a group page, and writes the
   config keys ``arm``, ``seed_``, ``split = "R"`` and ``rank0`` the runset groups on;
   failed and crashed runs and the abandoned partial of S3 seed 1 are left unlabeled
   and so fall outside the view's ``split == R`` filter;
2. overwrites a SAVED workspace view (the API cannot write the personal
   ``?nw=nwuser...`` view) with sections in rank order, x = epoch, and for every
   headline metric three panels: validation, training, and training and validation on
   ONE plot (the curation rule of the ``wandb-curate`` skill).

Rerun after every sync:

    python experiments/025-solid-growth/scripts/random_split_wandb_view.py
"""

from __future__ import annotations

import wandb
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws

ENTITY = "zhao-group"
PROJECT = "torchcell_025-solid-growth_equivariant_cell_graph_transformer"
VIEW_NAME = "025 random split: arms grouped, S3 closure vs S0"
# Pinned after the first `save_as_new_view()`; None creates the view and prints its id.
VIEW_ID: str | None = "vo1fa9efqdf"
X = "epoch"

# The abandoned partial of S3 seed 1 (job 2408888, restarted for per-order logging).
EXCLUDED_RUN_IDS = {"kj03xx8y", "0kaadgdu", "bekoxpor", "ztfcxu37"}
KEEP_STATES = {"finished", "running"}

# config tag -> arm label; ctrl_013 is split further by graph_reg_lambda (the
# regularization ladder ran under that one config with the weight swept). The
# epoch budget is appended from each run's own config (``trainer.max_epochs``),
# since one config runs under several budgets: fit_031 seed 1 at 130 epochs, seeds
# 2 and 3 at the 50-epoch cap set after seed 1 peaked at epoch 32.
ARMS = {
    "cgt_s3_r_kl_fit_031": "s3_closure_fitness1.0",
    "cgt_s3_r_kl_embfit_034": "s3_closure_composite_fitness1.0",
    "cgt_s4_r_kl_fit_039": "s4_random_doubles_fitness1.0",
    "cgt_s0_r_kl_embfit_035": "s0_composite_fitness1.0",
    "cgt_s0_r_kl_fit_014": "s0_fitness1.0",
    "cgt_s0_r_kl_fit_015": "s0_fitness0.1",
    "cgt_s0_r_kl_ctrl_013": "s0_control",
    "cgt_s0_r_mask_028": "s0_hardmask",
}
LAMBDA_ARMS = {0.0: "s0_lambda0", 0.01: "s0_lambda1e-2"}


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
        "3 per perturbation order (train; val and test are triples only)",
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
        "4 losses",
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
        "5 operator and gradient probe",
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
        "6 bookkeeping",
        [
            (
                ["train/cuda_peak_allocated_gb", "val/cuda_peak_allocated_gb"],
                "peak GPU memory (GB)",
            ),
            (["learning_rate"], "learning rate"),
            (["arm/n_subset"], "records in the pool"),
            (["arm/n_train_pinned"], "pinned train triples"),
        ],
    ),
]


def _arm_of(run: wandb.apis.public.Run) -> str | None:
    cfg = [t for t in run.tags if t in ARMS]
    if not cfg:
        return None
    arm = ARMS[cfg[0]]
    if cfg[0] == "cgt_s0_r_kl_ctrl_013":
        model = run.config.get("model") or {}
        lam = (model.get("graph_regularization") or {}).get("graph_reg_lambda")
        arm = LAMBDA_ARMS.get(float(lam) if lam is not None else -1.0, arm)
    max_epochs = (run.config.get("trainer") or {})["max_epochs"]
    return f"{arm}_{max_epochs}ep"


def label_runs(api: wandb.Api) -> int:
    """Name and group every kept R-split run; write the keys the view groups on."""
    runs = list(api.runs(f"{ENTITY}/{PROJECT}", filters={"tags": {"$in": ["split_R"]}}))
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
        ranked = sorted(
            seed_runs,
            key=lambda r: (
                0 if "val/gene_interaction/Pearson" in r.summary else 1,
                r.id,
            ),
        )
        for i, run in enumerate(ranked):
            rank0 = "val/gene_interaction/Pearson" in run.summary
            run.name = f"{arm}_seed{seed}" + ("" if rank0 else f"_rank{i}")
            run.group = arm
            run.config["arm"] = arm
            run.config["seed_"] = seed
            run.config["split"] = "R"
            run.config["rank0"] = rank0
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


def main() -> None:
    api = wandb.Api()
    print(f"labeled {label_runs(api)} runs")
    print(populate_view())


if __name__ == "__main__":
    main()
