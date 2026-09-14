# experiments/019-simb-multimodal/scripts/wandb_v13_report.py
# [[experiments.019-simb-multimodal.scripts.wandb_v13_report]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/wandb_v13_report
"""Name the v13 split-round runs, tag them for grouping, and build the comparison report.

The offline runs arrive on W&B named after their sync directory
(``run_compute-0-1-2397311_<hash>``), which says nothing about the arm, and the arm is
only recoverable from the tags. This script (1) renames every run in the project to
``<arm>_seed<k>`` and writes four top-level config keys the UI can group and filter on
(``arm``, ``split``, ``readout``, ``partition``), (2) writes a W&B report with one panel
grid per question (headline by arm, per split with both readouts, every validation
metric by arm, the train side, split against partition), and (3) overwrites the saved Charts
view with six sections in rank order of importance, grouped by arm with
epoch on the x axis. Rerun after every sync; it is idempotent.

Run from the repo root:
    python experiments/019-simb-multimodal/scripts/wandb_v13_report.py
"""

from __future__ import annotations

import re

import wandb
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws

ENTITY = "zhao-group"
PROJECT = "torchcell_019_expr_v13"
REPORT_TITLE = "v13 split round: H_ref and H_concat on four partitions and a 90/10 fold"
# The saved Charts view this script owns. The workspace API refuses the personal default
# view ("does not currently support user views"), so a saved view is the nearest thing:
# open the project, pick "v13 split round by arm" in the view dropdown. Created once with
# save_as_new_view(); every later run overwrites it in place.
VIEW_ID = "ywphnc96tfh"
X = "epoch"
PF = "val/expression/pearson_per_feature"
SPLITS = ["s0", "s0_90", "s1", "s2", "s3"]
SPLIT_LABEL = {
    "s0": "split 0, 80/10/10",
    "s0_90": "split 0, test folded into train (90/10)",
    "s1": "split 1",
    "s2": "split 2",
    "s3": "split 3",
}
VAL_METRICS = [
    PF,
    "val/expression/spearman_per_feature",
    "val/expression/pearson_per_instance",
    "val/expression/pred_sd_ratio",
    "val/loss",
    "val/expression/nmse",
    "val/expression/mse",
    "val/expression/pearson_per_feature@k1",
    "val/expression/pearson_per_feature@k2",
    "val/expression/pearson_per_feature@k3",
    "val/expression/calib/coverage_50",
    "val/expression/calib/coverage_80",
    "val/expression/calib/pit_ks",
    "val/mean/pearson_per_feature",
]
TRAIN_METRICS = [
    "traineval/expression/pearson_per_feature",
    "traineval/expression/pred_sd_ratio",
    "traineval/expression/nmse",
    "train/loss",
    "train/grad_norm",
    "train/grad_norm_clip_frac",
]


def label_runs(api: wandb.Api) -> int:
    n = 0
    for run in api.runs(f"{ENTITY}/{PROJECT}"):
        arms = [t for t in run.tags if t.startswith("V_")]
        if len(arms) != 1:
            raise ValueError(f"{run.id}: expected one V_* tag, got {arms}")
        arm = arms[0]
        m = re.fullmatch(r"V_(ref|concat)_(s\d+(?:_90)?)", arm)
        if m is None:
            raise ValueError(f"{run.id}: arm tag {arm} does not parse")
        readout, split = m.group(1), m.group(2)
        seed = int(run.config["seed"])
        name = f"{arm}_seed{seed}"
        changed = run.name != name or run.config.get("arm") != arm
        run.name = name
        run.config["arm"] = arm
        run.config["split"] = split
        run.config["readout"] = readout
        run.config["partition"] = "90/10" if split.endswith("_90") else "80/10/10"
        run.update()
        n += int(changed)
    return n


def line(title: str, y: str, groupby: str | None = "arm") -> wr.LinePlot:
    kw = {}
    if groupby is not None:
        kw = {
            "groupby": wr.Config(groupby),
            "groupby_aggfunc": "mean",
            "groupby_rangefunc": "minmax",
        }
    return wr.LinePlot(
        title=title,
        x=X,
        y=[y],
        title_x="epoch",
        smoothing_type="none",
        legend_position="east",
        layout=wr.Layout(w=12, h=8),
        **kw,
    )


def runset(name: str, filters: str | None = None) -> wr.Runset:
    return wr.Runset(entity=ENTITY, project=PROJECT, name=name, filters=filters or "")


def build_report() -> wr.Report:
    blocks: list = [
        wr.H1(REPORT_TITLE),
        wr.P(
            "24 runs, four per A40 card on the IGB gpu partition, 6,000 epochs "
            "(job 2397311, config cgt_expr_v13_split). H_ref is the v12 reference "
            "(E_full input, pinball, shared MLP readout); H_concat hands the readout "
            "[h_pert ; h_i ; c] instead of h_pert. Each card holds one split's two "
            "readouts for two seeds. Grouped lines are the mean over seeds with the "
            "min-max band. Nothing here is a result until the runs finish."
        ),
        wr.TableOfContents(),
        wr.H2("Headline: validation Pearson by arm"),
        wr.PanelGrid(
            runsets=[runset("all runs")],
            panels=[
                line("val pearson_per_feature, mean over seeds by arm", PF, "arm"),
                line("val pearson_per_feature, every run", PF, None),
                line(
                    "val pearson_per_feature by split (both readouts, all seeds)",
                    PF,
                    "split",
                ),
                line("val pearson_per_feature by readout (all splits)", PF, "readout"),
            ],
        ),
        wr.H2("Per split: H_ref against H_concat on the same partition"),
    ]
    for split in SPLITS:
        blocks.append(wr.H3(SPLIT_LABEL[split]))
        blocks.append(
            wr.PanelGrid(
                runsets=[runset(SPLIT_LABEL[split], f"Config('split') == '{split}'")],
                panels=[
                    line(f"{split}: val pearson by readout", PF, "readout"),
                    line(f"{split}: val pearson, every run", PF, None),
                    line(
                        f"{split}: pred_sd_ratio",
                        "val/expression/pred_sd_ratio",
                        "readout",
                    ),
                    line(f"{split}: val loss", "val/loss", "readout"),
                ],
            )
        )
    blocks.append(wr.H2("Every validation metric, by arm"))
    blocks.append(
        wr.PanelGrid(
            runsets=[runset("all runs")],
            panels=[line(m, m, "arm") for m in VAL_METRICS],
        )
    )
    blocks.append(wr.H2("Train side, by arm"))
    blocks.append(
        wr.PanelGrid(
            runsets=[runset("all runs")],
            panels=[line(m, m, "arm") for m in TRAIN_METRICS],
        )
    )
    blocks.append(wr.H2("Split 0: 90/10 against 80/10/10"))
    blocks.append(
        wr.PanelGrid(
            runsets=[runset("split 0 only", "Config('split') in ['s0', 's0_90']")],
            panels=[
                line("split 0: val pearson by partition", PF, "partition"),
                line("split 0: val pearson by arm", PF, "arm"),
                line(
                    "split 0: traineval pearson by partition",
                    "traineval/expression/pearson_per_feature",
                    "partition",
                ),
                line("split 0: val loss by partition", "val/loss", "partition"),
            ],
        )
    )
    return wr.Report(
        entity=ENTITY,
        project=PROJECT,
        title=REPORT_TITLE,
        description="Generated by experiments/019-simb-multimodal/scripts/wandb_v13_report.py",
        blocks=blocks,
        width="fluid",
    )


# The Charts tab, in rank order of importance. Section 1 is what decides the round;
# every later section is what to read when section 1 moves or fails to.
CHART_SECTIONS: list[tuple[str, list[str]]] = [
    (
        "1 headline: validation",
        [
            PF,
            "val/expression/spearman_per_feature",
            "val/expression/pearson_per_instance",
            "val/expression/pred_sd_ratio",
            "val/loss",
            "val/mean/pearson_per_feature",
        ],
    ),
    (
        "2 train side, the generalization gap",
        [
            "traineval/expression/pearson_per_feature",
            "traineval/expression/pred_sd_ratio",
            "traineval/expression/spearman_per_feature",
            "traineval/expression/pearson_per_instance",
            "traineval/loss",
            "traineval/expression/nmse",
        ],
    ),
    (
        "3 masked conditioning, revealed 0 / 10 / 100 / 1000 genes",
        [
            "val/expression/pearson_per_feature@k0",
            "val/expression/pearson_per_feature@k1",
            "val/expression/pearson_per_feature@k2",
            "val/expression/pearson_per_feature@k3",
            "val/mask/loss@k0",
            "val/mask/loss@k1",
            "val/mask/loss@k2",
            "val/mask/loss@k3",
        ],
    ),
    (
        "4 error and calibration",
        [
            "val/expression/nmse",
            "val/expression/mse",
            "val/expression/calib/coverage_50",
            "val/expression/calib/coverage_80",
            "val/expression/calib/pit_ks",
            "traineval/expression/mse",
        ],
    ),
    (
        "5 optimization",
        [
            "train/loss",
            "train/grad_norm",
            "train/grad_norm_clip_frac",
            "train/mask/loss@k0",
            "train/mask/loss@k3",
            "perf/epoch_seconds",
        ],
    ),
    (
        "6 bookkeeping",
        [
            "val/expression/n_scored_genes@k0",
            "val/mask/n_revealed@k1",
            "val/mask/n_revealed@k3",
            "trainer/global_step",
        ],
    ),
]


def populate_view() -> str:
    """Overwrite the saved Charts view with the ranked sections above."""
    view = ws.Workspace.from_url(f"https://wandb.ai/{ENTITY}/{PROJECT}?nw={VIEW_ID}")
    view.name = "v13 split round by arm"
    view.sections = [
        ws.Section(
            name=name,
            is_open=True,
            layout_settings=ws.SectionLayoutSettings(columns=3, rows=2),
            panel_settings=ws.SectionPanelSettings(x_axis=X, smoothing_type="none"),
            panels=[
                wr.LinePlot(
                    x=X, y=[m], title=m, title_x="epoch", layout=wr.Layout(w=8, h=6)
                )
                for m in metrics
            ],
        )
        for i, (name, metrics) in enumerate(CHART_SECTIONS)
    ]
    view.settings = ws.WorkspaceSettings(
        x_axis=X,
        smoothing_type="none",
        max_runs=24,
        sort_panels_alphabetically=False,
    )
    view.runset_settings = ws.RunsetSettings(
        groupby=[ws.Config("arm")],
        order=[ws.Ordering(ws.Metric("Name"), ascending=True)],
    )
    view.save()
    return view.url


def main() -> None:
    api = wandb.Api()
    n = label_runs(api)
    print(f"labeled runs: {n} changed")
    report = build_report()
    existing = [
        r for r in api.reports(f"{ENTITY}/{PROJECT}") if r.display_name == REPORT_TITLE
    ]
    if existing:
        live = wr.Report.from_url(
            f"https://wandb.ai/{ENTITY}/{PROJECT}/reports/x--{existing[0].id}"
        )
        live.blocks = report.blocks
        live.width = report.width
        live.save()
        print(f"report updated: {live.url}")
    else:
        report.save()
        print(f"report created: {report.url}")
    print(f"charts view: {populate_view()}")


if __name__ == "__main__":
    main()
