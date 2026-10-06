# experiments/019-simb-multimodal/scripts/wandb_v22_view.py
# [[experiments.019-simb-multimodal.scripts.wandb_v22_view]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/wandb_v22_view
"""Label and view the v22 fast-model round (W&B project torchcell_019_expr_v22).

The launcher (gh_expr_v22_fast.slurm) tags each run `[v22, <arm>, split<k>]`. This script

1. names each run `<arm>_s<k>`, sets its W&B group to the arm so
   `/groups/<arm>/workspace` shows that arm across split seeds, keeps the trainer's own
   group (the checkpoint directory) in config `ckpt_group`, and writes config keys `arm`,
   `split`, `arm_split`;
2. writes two saved Charts views with the ranked sections (validation Pearson, train
   against validation, prediction spread, loss, throughput): one grouped by arm (one line
   per arm, mean over split seeds with the min to max band) and one with a line per run.
   View ids are recorded in results/wandb_view_ids.json under `v22` and `v22/runs`.

Rerun after a new wave starts; it is idempotent.

    python experiments/019-simb-multimodal/scripts/wandb_v22_view.py
"""
from __future__ import annotations

import json
import os.path as osp
import re

import wandb
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws

ENTITY = "zhao-group"
PROJECT = "torchcell_019_expr_v22"
X = "epoch"
VIEW_REGISTRY = osp.join(
    osp.dirname(osp.dirname(osp.abspath(__file__))), "results", "wandb_view_ids.json"
)
CKPT_GROUP_RE = re.compile(r"[\w.-]+-\d+_[0-9a-f]{64}")

SECTIONS: list[tuple[str, list[list[str]]]] = [
    (
        "1 validation Pearson",
        [
            ["val/expression/pearson_per_feature"],
            ["val/expression/pearson_per_instance"],
            ["val/expression/spearman_per_feature"],
        ],
    ),
    (
        "2 train against validation",
        [
            ["traineval/expression/pearson_per_feature", "val/expression/pearson_per_feature"],
            ["traineval/expression/pearson_per_feature"],
            ["traineval/expression/pred_sd_ratio", "val/expression/pred_sd_ratio"],
        ],
    ),
    (
        "3 prediction spread and squared error",
        [["val/expression/pred_sd_ratio"], ["val/expression/nmse"], ["val/expression/mse"]],
    ),
    (
        "4 loss and calibration",
        [
            ["train/loss", "val/loss"],
            ["val/loss"],
            ["val/expression/calib/coverage_50", "val/expression/calib/coverage_80"],
        ],
    ),
    ("5 throughput", [["perf/epoch_seconds"], ["train/grad_norm"], ["train/grad_norm_clip_frac"]]),
]


def label_runs(api: wandb.Api) -> tuple[int, set[str], list[str]]:
    changed = 0
    present: set[str] = set()
    arms: set[str] = set()
    for run in api.runs(f"{ENTITY}/{PROJECT}"):
        arm = next(t for t in run.tags if t.startswith("F_"))
        split = int(next(t for t in run.tags if t.startswith("split")).removeprefix("split"))
        name = f"{arm}_s{split}"
        held = [run.config.get("ckpt_group"), run.group, str(run.name).removeprefix("run_")]
        dirs = {h for h in held if isinstance(h, str) and CKPT_GROUP_RE.fullmatch(h)}
        if len(dirs) != 1:
            raise ValueError(f"{run.id}: checkpoint directory not unique in {held}")
        changed += int(run.name != name or run.group != arm)
        run.name = name
        run.group = arm
        run.config["ckpt_group"] = dirs.pop()
        run.config["arm"] = arm
        run.config["split"] = split
        run.config["arm_split"] = name
        run.update()
        present.update(run.summary.keys())
        arms.add(arm)
    return changed, present, sorted(arms)


def populate_view(slot: str, name: str, groupby: str, present: set[str]) -> str:
    sections = []
    for sec_name, metrics in SECTIONS:
        panels = [[k for k in ys if k in present] for ys in metrics]
        panels = [ys for ys in panels if ys]
        if not panels:
            continue
        sections.append(
            ws.Section(
                name=sec_name,
                is_open=True,
                layout_settings=ws.SectionLayoutSettings(columns=3, rows=2),
                panel_settings=ws.SectionPanelSettings(x_axis=X, smoothing_type="none"),
                panels=[
                    wr.LinePlot(
                        x=X,
                        y=ys,
                        title=" | ".join(ys),
                        title_x="epoch",
                        layout=wr.Layout(w=8, h=6),
                        max_runs_to_show=100,
                    )
                    for ys in panels
                ],
            )
        )
    settings = ws.WorkspaceSettings(
        x_axis=X, smoothing_type="none", max_runs=100, sort_panels_alphabetically=False
    )
    runset_settings = ws.RunsetSettings(
        groupby=[ws.Config(groupby)],
        order=[ws.Ordering(ws.Metric("Name"), ascending=True)],
    )
    with open(VIEW_REGISTRY) as f:
        reg: dict[str, dict[str, str]] = json.load(f)
    view_id = reg.get(slot, {}).get("id")
    if view_id is None:
        view = ws.Workspace(
            entity=ENTITY,
            project=PROJECT,
            name=name,
            sections=sections,
            settings=settings,
            runset_settings=runset_settings,
        )
        view.save_as_new_view()
        new_id = re.search(r"[?&]nw=([\w-]+)", view.url)
        if new_id is None:
            raise ValueError(f"no view id in {view.url}")
        reg[slot] = {"id": new_id.group(1), "url": view.url, "name": name}
        with open(VIEW_REGISTRY, "w") as f:
            json.dump(reg, f, indent=2, sort_keys=True)
        return view.url
    view = ws.Workspace.from_url(f"https://wandb.ai/{ENTITY}/{PROJECT}?nw={view_id}")
    view.name = name
    view.sections = sections
    view.settings = settings
    view.runset_settings = runset_settings
    view.save()
    return view.url


def main() -> None:
    api = wandb.Api()
    changed, present, arms = label_runs(api)
    print(f"labeled runs: {changed} changed; arms {arms}")
    print("by arm (one line per arm, mean over split seeds)")
    print(populate_view("v22", "v22: by arm", "arm", present))
    print("every run")
    print(populate_view("v22/runs", "v22: every run", "arm_split", present))
    for arm in arms:
        print(f"https://wandb.ai/{ENTITY}/{PROJECT}/groups/{arm}/workspace")


if __name__ == "__main__":
    main()
