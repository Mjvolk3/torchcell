# experiments/025-solid-growth/scripts/disjoint_embedding_wandb_view.py
# [[experiments.025-solid-growth.scripts.disjoint_embedding_readout]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/025-solid-growth/scripts/disjoint_embedding_wandb_view
"""Name the disjoint-split runs by arm and seed and build a grouped W&B Charts view.

Offline IGB runs sync under directory names (``run_compute-5-7-<job>_<hash>``) and each
4-GPU job is four runs of which only rank 0 carries validation metrics. This script (1)
renames every run of a listed config ``<arm>_seed<k>`` (``_rank<r>`` for the three
non-rank-0 runs) and writes top-level config keys ``arm``, ``seed``, ``split`` and
``rank0`` so the runs table can group on them, and (2) overwrites a SAVED workspace view
whose runset is grouped by ``arm`` with the main charts in rank order, x = epoch. The
workspace API cannot write the personal default view (``?nw=nwuser...``), so the saved
view's ``nw=`` id is pinned here and later runs overwrite it. Rerun after every sync.

    python experiments/025-solid-growth/scripts/disjoint_embedding_wandb_view.py
"""

from __future__ import annotations

import wandb
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws

ENTITY = "zhao-group"
PROJECT = "torchcell_025-solid-growth_equivariant_cell_graph_transformer"
VIEW_NAME = "025 disjoint split: arms grouped"
# Pinned after the first `save_as_new_view()`; None creates the view and prints its id.
VIEW_ID: str | None = "paezdq4q5ex"
X = "epoch"

# config tag -> arm label (the group key), budget
ARMS = {
    "cgt_s0_q_kl_ctrl_016": "learnable_table_30ep",
    "cgt_s0_q_kl_emb_017": "composite_30ep",
    "cgt_s0_q_kl_rand_018": "random_vector_30ep",
    "cgt_s0_q_kl_calm_020": "calm_30ep",
    "cgt_s0_q_kl_prot_021": "prott5_30ep",
    "cgt_s0_q_kl_emb_022": "composite_100ep",
    "cgt_s0_q_kl_calm_023": "calm_100ep",
    "cgt_s0_q_kl_prot_024": "prott5_100ep",
    "cgt_s0_q_kl_fudt_026": "flanks_100ep",
    "cgt_s0_q_kl_embfit_027": "composite_fitness_30ep",
    "cgt_s0_q_kl_004": "learnable_table_cosine_job1640",
}

SECTIONS: list[tuple[str, list[str]]] = [
    (
        "1 held-out query pairs",
        [
            "val/gene_interaction/Pearson",
            "val/point_loss",
            "val/gene_interaction/MSE",
            "val/fitness/Pearson",
        ],
    ),
    (
        "2 training side",
        [
            "train/gene_interaction/Pearson",
            "train/point_loss",
            "train/graph_reg_loss",
            "train/loss",
        ],
    ),
    (
        "3 operator and probe",
        [
            "val/cls_pert_strain_sd",
            "probe/grad_norm/point",
            "probe/grad_norm/graph_reg",
            "probe/grad_ratio/graph_reg_to_point",
        ],
    ),
    (
        "4 bookkeeping",
        ["train/cuda_peak_allocated_gb", "val/cuda_peak_allocated_gb", "arm/n_subset"],
    ),
]


def label_runs(api: wandb.Api) -> int:
    """Rename runs ``<arm>_seed<k>[_rank<r>]`` and write the grouping keys."""
    n = 0
    for cfg, arm in ARMS.items():
        runs = list(api.runs(f"{ENTITY}/{PROJECT}", filters={"tags": {"$in": [cfg]}}))
        # Rank 0 of a job is the run whose summary carries validation Pearson; the
        # other three ranks of the same seed are numbered after it.
        by_seed: dict[int, list] = {}
        for run in runs:
            by_seed.setdefault(int(run.config.get("seed", 42)), []).append(run)
        for seed, seed_runs in by_seed.items():
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
                run.config["arm"] = arm
                run.config["seed_"] = seed
                run.config["split"] = "Q"
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
            panels=[
                wr.LinePlot(
                    x=X, y=[m], title=m, title_x="epoch", layout=wr.Layout(w=6, h=6)
                )
                for m in metrics
            ],
        )
        for name, metrics in SECTIONS
    ]
    settings = ws.WorkspaceSettings(
        x_axis=X, smoothing_type="none", max_runs=60, sort_panels_alphabetically=False
    )
    runset_settings = ws.RunsetSettings(
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
