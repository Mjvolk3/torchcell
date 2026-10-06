# experiments/019-simb-multimodal/scripts/wandb_grid_bench_view.py
# [[experiments.019-simb-multimodal.scripts.wandb_grid_bench_view]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/wandb_grid_bench_view
"""Label and view the depth x operator grid (W&B project torchcell_019_grid_bench).

The grid launcher (gh_small_model_bench.slurm, cells 10 to 13, job 3274) tags each run
with its cell (`grid_w90_l<L>_p4_w3`) and nothing names the operator, which lives only in
the config (`model.perturbation_head.hadamard`, `.null_sink`,
`multitask.response_basis_rank`). This script

1. names each run `<operator>_L<layers>`, sets its W&B group to the operator so
   `/groups/<operator>/workspace` shows that operator at the four depths, keeps the
   trainer's own group (the checkpoint directory) in config `ckpt_group`, and writes
   config keys `arm`, `operator`, `depth`, `cell`;
2. writes two saved Charts views, one grouped by operator and one grouped by depth, with
   the ranked sections: validation Pearson, train side, prediction spread, loss,
   throughput. View ids are recorded in results/wandb_view_ids.json under `grid_bench`
   and `grid_bench/by_depth`.

    python experiments/019-simb-multimodal/scripts/wandb_grid_bench_view.py
"""
from __future__ import annotations

import json
import os.path as osp
import re

import wandb
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws

ENTITY = "zhao-group"
PROJECT = "torchcell_019_grid_bench"
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
        "3 prediction spread (launch gate 0.05)",
        [["val/expression/pred_sd_ratio"], ["val/expression/nmse"], ["val/expression/mse"]],
    ),
    ("4 loss", [["train/loss", "val/loss"], ["val/loss"], ["train/loss"]]),
    ("5 throughput", [["perf/epoch_seconds"], ["train/grad_norm"], ["train/grad_norm_clip_frac"]]),
]


def operator(config: dict) -> str:
    head = config["model"]["perturbation_head"]
    if head.get("hadamard", "off") != "off":
        return "S_hadam"
    if head.get("null_sink"):
        return "S_sink"
    if config.get("multitask", {}).get("response_basis_rank") not in (None, 0):
        return "S_basis64"
    return "S_ref"


def label_runs(api: wandb.Api) -> tuple[int, set[str]]:
    changed = 0
    present: set[str] = set()
    for run in api.runs(f"{ENTITY}/{PROJECT}"):
        cell = next(t for t in run.tags if t.startswith("grid_"))
        depth = int(re.search(r"_l(\d+)_", cell).group(1))
        op = operator(run.config)
        name = f"{op}_L{depth}"
        held = [run.config.get("ckpt_group"), run.group, str(run.name).removeprefix("run_")]
        dirs = {h for h in held if isinstance(h, str) and CKPT_GROUP_RE.fullmatch(h)}
        if len(dirs) != 1:
            raise ValueError(f"{run.id}: checkpoint directory not unique in {held}")
        changed += int(run.name != name or run.group != op)
        run.name = name
        run.group = op
        run.config["ckpt_group"] = dirs.pop()
        run.config["arm"] = name
        run.config["operator"] = op
        run.config["depth"] = depth
        run.config["cell"] = cell
        run.update()
        present.update(run.summary.keys())
    return changed, present


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
    changed, present = label_runs(api)
    print(f"labeled runs: {changed} changed")
    print("by operator (one line per run, grouped by operator)")
    print(populate_view("grid_bench", "grid: by operator", "operator", present))
    print("by depth")
    print(populate_view("grid_bench/by_depth", "grid: by depth", "depth", present))
    for op in ("S_ref", "S_sink", "S_hadam", "S_basis64"):
        print(f"https://wandb.ai/{ENTITY}/{PROJECT}/groups/{op}/workspace")


if __name__ == "__main__":
    main()
