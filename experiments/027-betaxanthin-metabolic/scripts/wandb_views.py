# experiments/027-betaxanthin-metabolic/scripts/wandb_views.py
# [[experiments.027-betaxanthin-metabolic.scripts.wandb_views]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/027-betaxanthin-metabolic/scripts/wandb_views
"""Saved W&B workspace views and reports for the 027 contrasts, built from the API.

Each contrast is a ``ViewSpec``: a tag filter that selects its runs, the config keys the
runs are grouped by, and the metrics its panels show. The script saves one workspace view
(the project page with those filters, grouping and panels applied) and one report (the
same panels, once grouped by the contrast key and once by seed) per contrast, and writes
their URLs to ``results/wandb_views.json``.

Panels are only built for metrics that at least one matched run has logged, so a spec can
name a metric that older runs lack without producing an empty chart.

Usage, from the worktree root::

    python experiments/027-betaxanthin-metabolic/scripts/wandb_views.py --contrast all
    python experiments/027-betaxanthin-metabolic/scripts/wandb_views.py --contrast kdraw
    python experiments/027-betaxanthin-metabolic/scripts/wandb_views.py --name "my view" \
        --tags gh_wave8_samp --groupby flux_draws --metrics val_pearson,latent_sigma_p50

The ad-hoc form (``--name`` with ``--tags``) builds a one-off view and report from the
command line without editing this file.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
from typing import Any

import wandb
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

ENTITY = "zhao-group"
PROJECT = "torchcell_027_bxfx"
EXPERIMENT_ROOT = os.getenv(
    "EXPERIMENT_ROOT", osp.join(osp.dirname(osp.dirname(osp.dirname(__file__))))
)
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic", "results")


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


SPECS: dict[str, ViewSpec] = {
    "kdraw": ViewSpec(
        key="kdraw",
        title="027 wave 8: K latent draws per forward pass",
        description=(
            "Betaxanthin, plain projection head, constraint scale 0, 80 epochs. Sampled "
            "arms at K = 1, 16 and 128 draws per genotype per forward pass, paired by seed "
            "against the deterministic wave-6 cells at seeds 2703 to 2705. Read the paired "
            "difference per seed, never the raw score: the deterministic column alone spans "
            "0.001 to 0.104 across seeds."
        ),
        tags_any=["gh_wave8_samp", "gh_wave6_proj"],
        groupby=["flux_draws", "flux_latent_stochastic"],
        curves=[
            "val_pearson",
            "val_pred_sd_ratio",
            "latent_sigma_p50",
            "flux_draw_pred_sd",
            "train_data_loss",
            "c_kl",
            "feas_flux_active_frac",
        ],
        scalars=["test_spearman_pinned", "select_val_pearson"],
    ),
    "densbal": ViewSpec(
        key="densbal",
        title="027 wave 8: turnover density and active-row balance on the boxforce head",
        description=(
            "Betaxanthin, flux-force boxforce channel (K = 100), constraint scale 0, 80 "
            "epochs. Density weight 0 (wave-7 control), 0.5 and 2.0, and the combined "
            "density 0.5 plus active-row balance 0.5 cell. The honest feasibility rule is "
            "the active-row balance max and p95 on the top 100 strains; the median rule "
            "passes trivially when most rows carry no flux."
        ),
        tags_any=["gh_wave8_dens", "gh_wave8_densbal", "gh_wave7_ff"],
        groupby=["flux_density_weight", "flux_balance_active_weight"],
        curves=[
            "val_pearson",
            "val_pred_sd_ratio",
            "c_density",
            "c_balance_active",
            "feas_active_row_frac",
            "feas_flux_active_frac",
            "feas_balance_active_p95",
            "feas_balance_active_max",
            "feas_thermo_violation_frac",
            "feas_box_violation_frac",
        ],
        scalars=[
            "test_spearman_pinned",
            "feas/pinned_test_top100/balance_max_ratio_median",
            "feas/pinned_test_top100/thermo_standard_state_violation_frac_eq_0",
        ],
    ),
    "aa": ViewSpec(
        key="aa",
        title="027 waves 7 and 8: amino-acid readouts on the anchored projection head",
        description=(
            "Mulleder amino acids (19 features, 564 pinned genes), 200 epochs. Readout "
            "arms: pooled control, turnover, whole-cell concentration mixture; the "
            "concentration arm on the shipped versus verified thermodynamic tables and on "
            "the assumed versus sourced compartment volumes. Score is the mean per-feature "
            "pinned test Spearman at the validation-selected epoch."
        ),
        tags_any=["gh_wave7_aa", "gh_wave8_aa"],
        groupby=["aa_readout", "thermo_source", "volume_table"],
        curves=[
            "val_pearson",
            "val_loss",
            "val_pred_sd_ratio",
            "train_data_loss",
            "train_loss",
            "latent_sigma_p50",
        ],
        scalars=[
            "test_mean_spearman_pinned",
            "conc/pinned_test/mean_spearman",
            "conc/pinned_test/frac_pred_at_ceiling",
            "select_val_pearson",
        ],
    ),
}


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


def line_panels(metrics: list[str], groupby: str | None) -> list[Any]:
    panels: list[Any] = []
    for i, metric in enumerate(metrics):
        panels.append(
            wr.LinePlot(
                title=metric,
                x="Step",
                y=[metric],
                groupby=groupby,
                groupby_aggfunc="mean" if groupby else None,
                groupby_rangefunc="minmax" if groupby else None,
                smoothing_type="none",
                legend_position="south",
                layout=wr.Layout(x=(i % 3) * 8, y=(i // 3) * 6, w=8, h=6),
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
                layout=wr.Layout(
                    x=(i % 3) * 8, y=offset_rows * 6 + (i // 3) * 6, w=8, h=6
                ),
            )
        )
    return panels


def build_workspace(spec: ViewSpec, curves: list[str], scalars: list[str]) -> str:
    """Save the project workspace view and return its URL."""
    workspace = ws.Workspace(
        entity=ENTITY,
        project=PROJECT,
        name=spec.title,
        sections=[
            ws.Section(
                name="curves", panels=line_panels(curves, groupby=None), is_open=True
            ),
            ws.Section(
                name="scalars", panels=bar_panels(scalars, offset_rows=0), is_open=True
            ),
        ],
        runset_settings=ws.RunsetSettings(
            filters=[ws.Tags().isin(spec.tags_any)],
            groupby=[ws.Config(k) for k in spec.groupby],
            order=[ws.Ordering(ws.Config("seed"), ascending=True)],
        ),
    )
    workspace.save_as_new_view()
    return str(workspace.url)


def build_report(spec: ViewSpec, curves: list[str], scalars: list[str]) -> str:
    """Save the report with one grid grouped by the contrast key and one by seed."""
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
            filters=[ws.Tags().isin(spec.tags_any)],
            groupby=[group_key],
        )
        grids.append(wr.H2(text=label))
        grids.append(
            wr.PanelGrid(
                runsets=[runset],
                panels=line_panels(curves, groupby=group_key)
                + bar_panels(scalars, offset_rows=(len(curves) + 2) // 3),
            )
        )
    report = wr.Report(
        entity=ENTITY,
        project=PROJECT,
        title=spec.title,
        description=spec.description,
        width="fluid",
        blocks=[wr.H1(text=spec.title), wr.P(text=spec.description), *grids],
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contrast", default=None, help="kdraw | densbal | aa | all")
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
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--no-workspace", action="store_true")
    args = parser.parse_args()

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
        }
        if not args.no_workspace:
            entry["workspace_url"] = build_workspace(spec, curves, scalars)
        if not args.no_report:
            entry["report_url"] = build_report(spec, curves, scalars)
        record[spec.key] = entry
        print(
            f"[{spec.key}] {len(runs)} runs; {len(curves)} curves, {len(scalars)} bars"
        )
        for label in ("workspace_url", "report_url"):
            if label in entry:
                print(f"  {label}: {entry[label]}")
        if missing:
            print(f"  not logged by any matched run: {missing}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(record, fh, indent=2)
    print(out_path)


if __name__ == "__main__":
    main()
