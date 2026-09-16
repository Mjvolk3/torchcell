# experiments/028-gene-essentiality/scripts/wandb_views.py
# [[experiments.028-gene-essentiality.scripts.wandb_views]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/wandb_views
"""Saved W&B workspace views and reports for the 028 gene-essentiality contrasts (copied from 027's script), built from the API.

Each contrast is a ``ViewSpec``: a tag filter that selects its runs, the config keys the
runs are grouped by, and the metrics its panels show. The script saves one workspace view
(the project page with those filters, grouping and panels applied) and one report (the
same panels, once grouped by the contrast key and once by seed) per contrast, and writes
their URLs to ``results/wandb_views.json``.

Panels are only built for metrics that at least one matched run has logged, so a spec can
name a metric that older runs lack without producing an empty chart.

Usage, from the worktree root::

    python experiments/028-gene-essentiality/scripts/wandb_views.py --contrast all   # views only; add --report for reports
    python experiments/028-gene-essentiality/scripts/wandb_views.py --contrast kdraw
    python experiments/028-gene-essentiality/scripts/wandb_views.py --name "my view" \
        --tags gh_wave8_samp --groupby flux_draws --metrics val_pearson,latent_sigma_p50

The ad-hoc form (``--name`` with ``--tags``) builds a one-off view and report from the
command line without editing this file. A contrast whose URLs are already recorded in
``wandb_views.json`` is updated in place (same links); ``--new`` forces fresh ones.
``--per-row`` sets the panels per row (default 6, narrow panels for an ultrawide).
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import re
from typing import Any

import wandb
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

ENTITY = "zhao-group"
PROJECT = "torchcell_028_gene_essentiality"
EXPERIMENT_ROOT = os.getenv(
    "EXPERIMENT_ROOT", osp.join(osp.dirname(osp.dirname(osp.dirname(__file__))))
)
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "results")


class ViewSpec(BaseModel):
    """One contrast: which runs, how they are grouped, what the panels show."""

    key: str = Field(description="Short slug used on the command line.")
    title: str = Field(description="Title of the saved view and the report.")
    description: str = Field(
        description="One paragraph shown at the top of the report."
    )
    tags_any: list[str] = Field(
        description="Runs carrying ANY of these W&B tags are selected."
    )
    groupby: list[str] = Field(
        description="Config keys the workspace groups by, outermost first."
    )
    curves: list[str] = Field(
        description="Per-epoch metrics drawn as line plots against the step."
    )
    scalars: list[str] = Field(
        default_factory=list,
        description="Summary metrics drawn as bar charts, one bar per group.",
    )
    drop_failed: bool = Field(
        default=True,
        description=(
            "Filter out runs whose state is failed or crashed. The four essentiality "
            "crashes (jobs 1990, 1991, 1997, 1998) carry the round's tags and no epochs; "
            "left in, they sit as empty rows in every group and pull the group mean of a "
            "bar chart toward nothing. Running runs stay in."
        ),
    )


SPECS: dict[str, ViewSpec] = {
    "w12": ViewSpec(
        key="w12",
        title="028 wave 12: split protocol, label source, weight decay and batch size on gene essentiality",
        description=(
            "Gene essentiality, pooled arm (no flux layer), width 12, one block, one-layer "
            "preprocessor, 80 epochs, seeds 2700 to 2702. Arms: w12base repeats the wave-11 "
            "protocol (released labels, seeded 80/10/10 draw over 5,461 records, 75-gene "
            "validation) in the 028 experiment as the port check; w12fcl reads the split "
            "file ess_fcl_val0.2_s0 (released labels, class-stratified 158-gene validation, "
            "unlabeled records dropped); w12gen reads ess_genome_val0.2_s0 (5,795 labeled "
            "genes: released labels plus SGD essential and viable single-mutant fitness "
            "records, 1,093-gene validation); w12fclwd0, w12fclwd1e-2 and w12fclwd1e-1 are "
            "w12fcl at AdamW weight decay 0, 1e-2 and 1e-1 against the base 1e-4; w12fclb32 "
            "is w12fcl at batch 32 against 128. Every cell is scored on the same 195 pinned "
            "held-out genes (31 essential). Read the validation curve's smoothness against "
            "the validation size first, then the pinned test AUROC with its 0.17-wide gene "
            "bootstrap in mind. Group by arm."
        ),
        tags_any=[
            "w12base",
            "w12fcl",
            "w12gen",
            "w12fclwd0",
            "w12fclwd1e-2",
            "w12fclwd1e-1",
            "w12fclb32",
        ],
        groupby=["arm_tag"],
        curves=[
            "val_pearson",
            "val_loss",
            "train_data_loss",
            "train_loss",
            "val_pred_sd_ratio",
            "grad_norm_mean",
            "grad_norm_max",
            "grad_norm_embedding_preprocessor_mean",
            "grad_norm_transformer_layers_mean",
            "grad_norm_essentiality_head_mean",
            "graph_reg_loss",
        ],
        scalars=[
            "test_auroc_pinned",
            "test_accuracy_pinned",
            "test_recall_essential_pinned",
            "select_val_pearson",
            "n_val",
        ],
    )
}


READOUT_KEY_BY_HEAD = {"mulleder19": "aa_readout", "isobutanol": "ib_readout"}
POOLED_NO_MODULE = "pooled_no_module"


def readout_group(run: Any) -> str:
    """The label a run is grouped under: its readout, or the no-module control."""
    cfg = run.config
    key = READOUT_KEY_BY_HEAD.get(str(cfg.get("head")), "bx_readout")
    readout = cfg.get(key)
    if isinstance(readout, str) and readout.startswith("context_"):
        return readout
    if cfg.get("arm") == "pooled":
        return POOLED_NO_MODULE
    return str(readout or cfg.get("arm") or "unknown")


def loss_arm(run: Any) -> str:
    """The loss a run trained under, as one label the workspace can group by.

    The config carries it as two fields, `crps` (bool) and `crps_sigma_fixed` (float or
    absent on runs that predate the flag), and W&B groups on one key at a time.
    """
    cfg = run.config
    if not cfg.get("crps"):
        return "mse"
    fixed = cfg.get("crps_sigma_fixed")
    if fixed is None:
        return "crps_learned_sigma"
    return f"crps_sigma_fixed_{float(fixed):.2f}"


def annotate_readout_group(runs: list[Any]) -> int:
    """Write `readout_group` and `loss_arm` onto every matched run where missing or stale."""
    changed = 0
    for run in runs:
        labels = {"readout_group": readout_group(run), "loss_arm": loss_arm(run)}
        stale = {k: v for k, v in labels.items() if run.config.get(k) != v}
        if stale:
            for k, v in stale.items():
                run.config[k] = v
            run.update()
            changed += 1
    return changed


#: Tags train_bx.py derives from the cell rather than from the launcher: the project
#: markers, the host, the head, the arm, the seed, the draw count and the `--wandb-group`
#: wave tag. Whatever tag is left after removing these is the launcher's `--tag`, the
#: arm-level label that names the run's W&B group.
HOSTS = ("gilahyper",)
#: A run whose budget is at most this many epochs is a launcher smoke, not a cell.
SMOKE_EPOCHS = 2


def arm_tag(run: Any) -> str:
    """The launcher's arm-level tag on a run, the name its W&B group takes.

    Falls back to the wave tag (`--wandb-group`) for runs launched without a `--tag`.
    """
    cfg = run.config
    derived = {
        "028",
        "essentiality",
        str(cfg.get("head")),
        str(cfg.get("arm")),
        f"seed{cfg.get('seed')}",
        f"k{cfg.get('flux_draws')}",
        *HOSTS,
    }
    wave = [t for t in run.tags if t.startswith("gh_")]
    rest = [
        t
        for t in run.tags
        if t not in derived and t not in wave and not re.fullmatch(r"(seed|k)\d+", t)
    ]
    if len(rest) == 1:
        return rest[0]
    assert len(rest) == 0 and len(wave) == 1, (
        f"run {run.id} has an ambiguous arm tag: {run.tags}"
    )
    return wave[0]


def regroup_by_arm(runs: list[Any]) -> dict[str, list[str]]:
    """Put every seed of one arm in one W&B group and name each run `<arm tag>_seed<k>`.

    The trainer wrote a per-job group until 2026-09-15, so each group page held one run.
    Failed and crashed runs keep their old group so the arm's page shows only runs that
    produced epochs. Returns the group name -> run ids it now holds.
    """
    groups: dict[str, list[str]] = {}
    for run in runs:
        if run.state in ("failed", "crashed"):
            continue
        tag = arm_tag(run)
        if int(run.config.get("epochs", 0)) <= SMOKE_EPOCHS:
            # A launcher smoke (the 2-epoch essentiality check, job 2004) carries the
            # arm's tag; it goes to its own group so the arm's page holds real budgets.
            tag = f"{tag}_smoke"
        name = f"{tag}_seed{run.config.get('seed')}"
        # `arm_tag` as a CONFIG key too, so a workspace can group on it: every 028 arm
        # shares `arm: pooled` in its config and differs only by the launcher tag.
        if (
            run.group != tag
            or run.display_name != name
            or run.config.get("arm_tag") != tag
        ):
            run.group = tag
            run.display_name = name
            run.config["arm_tag"] = tag
            run.update()
        groups.setdefault(tag, []).append(run.id)
    return groups


def group_url(group: str) -> str:
    return f"https://wandb.ai/{ENTITY}/{PROJECT}/groups/{group}"


def runset_filters(spec: ViewSpec) -> list[Any]:
    """The tag filter, plus the state filter that drops failed and crashed runs."""
    filters: list[Any] = [ws.Tags().isin(spec.tags_any)]
    if spec.drop_failed:
        filters.append(ws.Metric("State") != "failed")
        filters.append(ws.Metric("State") != "crashed")
    return filters


def matched_runs(api: wandb.Api, spec: ViewSpec) -> list[Any]:
    """The runs the spec's tag filter selects, fetched once to validate the metrics."""
    return list(
        api.runs(
            f"{ENTITY}/{PROJECT}",
            filters={"tags": {"$in": spec.tags_any}},
            per_page=200,
        )
    )


def present_metrics(runs: list[Any], wanted: list[str]) -> list[str]:
    """Keep the metrics at least one matched run has in its summary."""
    logged: set[str] = set()
    for run in runs:
        logged.update(k for k in run.summary.keys() if not k.startswith("_"))
    return [m for m in wanted if m in logged]


# PANEL GEOMETRY. The W&B panel grid is 24 columns wide; a panel's width is a column count
# and its height a row count. The default is six narrow panels per row (4 columns each,
# 5 rows tall): the views are read on a 38-inch ultrawide, where the previous 3-per-row
# layout stretched every chart to a third of the screen and pushed the grid off the bottom.
GRID_COLUMNS = 24
PANELS_PER_ROW = 6
PANEL_ROWS = 5


def panel_layout(i: int, offset_rows: int = 0) -> Any:
    w = GRID_COLUMNS // PANELS_PER_ROW
    return wr.Layout(
        x=(i % PANELS_PER_ROW) * w,
        y=(offset_rows + i // PANELS_PER_ROW) * PANEL_ROWS,
        w=w,
        h=PANEL_ROWS,
    )


def rows_for(n_panels: int) -> int:
    return (n_panels + PANELS_PER_ROW - 1) // PANELS_PER_ROW


#: Train/validation pairs drawn as a TRIPLE: the two overlaid, then each alone. The overlay
#: is where overfitting is read, because the gap between the curves is the quantity and a
#: gap cannot be seen across two panels with independent y axes. The isolated panels are
#: where the shape of each curve is read, because the overlay's axis is set by whichever
#: curve has the larger range and flattens the other. Convention requested 2026-09-15: a
#: W&B view of a training round carries both.
TRAIN_VAL_PAIRS = (("train_data_loss", "val_loss"), ("train_loss", "val_loss"))


def train_val_panels(present: list[str], groupby: str | None) -> list[Any]:
    """For each train/val pair both runs have logged: overlay, train alone, val alone."""
    panels: list[Any] = []
    index = 0
    for train_metric, val_metric in TRAIN_VAL_PAIRS:
        if train_metric not in present or val_metric not in present:
            continue
        for title, ys in (
            (f"{train_metric} vs {val_metric}", [train_metric, val_metric]),
            (train_metric, [train_metric]),
            (val_metric, [val_metric]),
        ):
            panels.append(
                wr.LinePlot(
                    title=title,
                    x="epoch",
                    y=list(ys),
                    groupby=groupby,
                    groupby_aggfunc="mean" if groupby else None,
                    groupby_rangefunc="minmax" if groupby else None,
                    smoothing_type="none",
                    legend_position="south",
                    layout=panel_layout(index),
                )
            )
            index += 1
    return panels


def line_panels(metrics: list[str], groupby: str | None) -> list[Any]:
    panels: list[Any] = []
    for i, metric in enumerate(metrics):
        panels.append(
            wr.LinePlot(
                title=metric,
                # Every row train_bx logs carries `epoch`, and the user reads curves by
                # epoch, never by W&B's step counter.
                x="epoch",
                y=[metric],
                groupby=groupby,
                groupby_aggfunc="mean" if groupby else None,
                groupby_rangefunc="minmax" if groupby else None,
                smoothing_type="none",
                legend_position="south",
                layout=panel_layout(i),
            )
        )
    return panels


def bar_panels(metrics: list[str], offset_rows: int) -> list[Any]:
    panels: list[Any] = []
    for i, metric in enumerate(metrics):
        panels.append(
            wr.BarPlot(
                title=metric,
                metrics=[metric],
                orientation="h",
                layout=panel_layout(i, offset_rows),
            )
        )
    return panels


def workspace_sections(curves: list[str], scalars: list[str]) -> list[Any]:
    """Ranked sections: the train/validation triples first, then the curves, then the bars.

    Each W&B section carries its own panel grid, so the layouts restart at zero per section
    rather than needing a running offset.
    """
    sections: list[Any] = []
    paired = train_val_panels(curves, groupby=None)
    if paired:
        sections.append(
            ws.Section(name="train vs validation", panels=paired, is_open=True)
        )
    sections.append(
        ws.Section(
            name="curves", panels=line_panels(curves, groupby=None), is_open=True
        )
    )
    sections.append(
        ws.Section(
            name="scalars", panels=bar_panels(scalars, offset_rows=0), is_open=True
        )
    )
    return sections


def workspace_runset(spec: ViewSpec) -> Any:
    return ws.RunsetSettings(
        filters=runset_filters(spec),
        groupby=[ws.Config(k) for k in spec.groupby],
        order=[ws.Ordering(ws.Config("seed"), ascending=True)],
    )


def build_workspace(
    spec: ViewSpec, curves: list[str], scalars: list[str], existing_url: str | None
) -> str:
    """Save the project workspace view and return its URL.

    With ``existing_url`` the saved view is updated in place so a link the user already
    has keeps working; without it a new view is created.
    """
    if existing_url:
        workspace = ws.Workspace.from_url(existing_url)
        workspace.name = spec.title
        workspace.sections = workspace_sections(curves, scalars)
        workspace.runset_settings = workspace_runset(spec)
        workspace.save()
        return str(workspace.url)
    workspace = ws.Workspace(
        entity=ENTITY,
        project=PROJECT,
        name=spec.title,
        sections=workspace_sections(curves, scalars),
        runset_settings=workspace_runset(spec),
    )
    workspace.save_as_new_view()
    return str(workspace.url)


def report_blocks(spec: ViewSpec, curves: list[str], scalars: list[str]) -> list[Any]:
    """One grid grouped by the contrast key and one by seed."""
    primary = spec.groupby[0]
    grids: list[Any] = []
    for label, group_key in (
        (f"grouped by {primary}", primary),
        ("grouped by seed", "seed"),
    ):
        runset = wr.Runset(
            entity=ENTITY,
            project=PROJECT,
            name=f"{spec.title} ({label})",
            filters=runset_filters(spec),
            groupby=[group_key],
        )
        grids.append(wr.H2(text=label))
        grids.append(
            wr.PanelGrid(
                runsets=[runset],
                panels=line_panels(curves, groupby=group_key)
                + bar_panels(scalars, offset_rows=rows_for(len(curves))),
            )
        )
    return [wr.H1(text=spec.title), wr.P(text=spec.description), *grids]


def build_report(
    spec: ViewSpec, curves: list[str], scalars: list[str], existing_url: str | None
) -> str:
    """Save the report and return its URL, updating in place when a URL is known."""
    if existing_url:
        report = wr.Report.from_url(existing_url)
        report.title = spec.title
        report.description = spec.description
        report.width = "fluid"
        report.blocks = report_blocks(spec, curves, scalars)
        report.save()
        return str(report.url)
    report = wr.Report(
        entity=ENTITY,
        project=PROJECT,
        title=spec.title,
        description=spec.description,
        width="fluid",
        blocks=report_blocks(spec, curves, scalars),
    )
    report.save()
    return str(report.url)


def adhoc_spec(args: argparse.Namespace) -> ViewSpec:
    return ViewSpec(
        key="adhoc",
        title=args.name,
        description=args.description or f"Ad-hoc view: tags {args.tags}.",
        tags_any=args.tags.split(","),
        groupby=args.groupby.split(",") if args.groupby else ["seed"],
        curves=args.metrics.split(",") if args.metrics else ["val_pearson"],
        scalars=args.scalars.split(",") if args.scalars else ["test_spearman_pinned"],
    )


def main() -> None:
    global PANELS_PER_ROW
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contrast", default=None, help=" | ".join([*SPECS, "all"]))
    parser.add_argument("--name", default=None, help="Ad-hoc view title (with --tags).")
    parser.add_argument(
        "--tags", default=None, help="Comma-separated tags, ANY matches."
    )
    parser.add_argument("--groupby", default=None, help="Comma-separated config keys.")
    parser.add_argument(
        "--metrics", default=None, help="Comma-separated curve metrics."
    )
    parser.add_argument("--scalars", default=None, help="Comma-separated bar metrics.")
    parser.add_argument("--description", default=None)
    parser.add_argument(
        "--report",
        action="store_true",
        help=(
            "Also build (or update) the paired report. Off by default since 2026-09-14: "
            "the saved workspace view, the project page with the filters, grouping and "
            "panels applied, is what gets reviewed; a report is opt-in."
        ),
    )
    parser.add_argument("--no-workspace", action="store_true")
    parser.add_argument(
        "--per-row",
        type=int,
        default=PANELS_PER_ROW,
        help="Panels per grid row (24-column grid; 6 -> narrow panels for an ultrawide).",
    )
    parser.add_argument(
        "--new",
        action="store_true",
        help="Create fresh views instead of updating the URLs recorded in wandb_views.json.",
    )
    args = parser.parse_args()
    assert GRID_COLUMNS % args.per_row == 0, "--per-row must divide 24"
    PANELS_PER_ROW = args.per_row

    if args.name:
        assert args.tags, "--name needs --tags"
        specs = [adhoc_spec(args)]
    else:
        assert args.contrast, "give --contrast or --name/--tags"
        keys = list(SPECS) if args.contrast == "all" else args.contrast.split(",")
        specs = [SPECS[k] for k in keys]

    api = wandb.Api(timeout=60)
    out_path = osp.join(RESULTS_DIR, "wandb_views.json")
    record: dict[str, Any] = {}
    if osp.exists(out_path):
        with open(out_path) as fh:
            record = json.load(fh)

    for spec in specs:
        runs = matched_runs(api, spec)
        relabeled = annotate_readout_group(runs)
        if relabeled:
            print(f"[{spec.key}] readout_group written on {relabeled} runs")
        groups = regroup_by_arm(runs)
        curves = present_metrics(runs, spec.curves)
        scalars = present_metrics(runs, spec.scalars)
        missing = [m for m in spec.curves + spec.scalars if m not in curves + scalars]
        entry: dict[str, Any] = {
            "title": spec.title,
            "n_runs": len(runs),
            "tags_any": spec.tags_any,
            "groupby": spec.groupby,
            "curves": curves,
            "scalars": scalars,
            "metrics_not_logged_by_any_run": missing,
            # One W&B group per arm: the page at each URL shows that arm's seeds together
            # with their aggregated curves, which is how the round is reviewed.
            "groups": {
                tag: {"url": group_url(tag), "run_ids": ids}
                for tag, ids in sorted(groups.items())
            },
        }
        prior = {} if args.new else record.get(spec.key, {})
        if not args.no_workspace:
            entry["workspace_url"] = build_workspace(
                spec, curves, scalars, prior.get("workspace_url")
            )
        if args.report:
            entry["report_url"] = build_report(
                spec, curves, scalars, prior.get("report_url")
            )
        elif prior.get("report_url"):
            entry["report_url"] = prior["report_url"]
        entry["panels_per_row"] = PANELS_PER_ROW
        record[spec.key] = entry
        print(
            f"[{spec.key}] {len(runs)} runs; {len(curves)} curves, {len(scalars)} bars"
        )
        for label in ("workspace_url", "report_url"):
            if label in entry:
                print(f"  {label}: {entry[label]}")
        for tag, block in entry["groups"].items():
            print(f"  group {tag} ({len(block['run_ids'])} runs): {block['url']}")
        if missing:
            print(f"  not logged by any matched run: {missing}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(record, fh, indent=2)
    print(out_path)


if __name__ == "__main__":
    main()
