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

    python experiments/027-betaxanthin-metabolic/scripts/wandb_views.py --contrast all   # views only; add --report for reports
    python experiments/027-betaxanthin-metabolic/scripts/wandb_views.py --contrast kdraw
    python experiments/027-betaxanthin-metabolic/scripts/wandb_views.py --name "my view" \
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
            "the assumed versus sourced compartment volumes, and in the whole-cell mixture "
            "versus the cytosol alone. Score is the mean per-feature pinned test Spearman "
            "at the validation-selected epoch."
        ),
        tags_any=["gh_wave7_aa", "gh_wave8_aa"],
        groupby=["aa_readout", "thermo_source", "volume_table", "conc_space"],
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
    "aa_all": ViewSpec(
        key="aa_all",
        title="027: every amino-acid run, all waves",
        description=(
            "Every cell whose head is the 19 Mulleder amino acids, waves 6 to 8, selected "
            "by the head tag and every amino-acid wave tag so nothing is left out: pooled "
            "controls, the wave-6 flux heads, turnover and concentration readouts, the "
            "thermodynamic-table and volume-fraction ablations, and the K-draw cells. "
            "Grouped by readout first, then by the ablation keys and the draw count. "
            "Score is the mean per-feature pinned test Spearman at the validation-selected "
            "epoch; budgets differ (80 versus 200 epochs), so pair within a budget."
        ),
        # The head tag alone misses the 14 wave-6 and early wave-7 cells that predate
        # it; the wave tags plus the head tag select all 45 runs whose config head is
        # mulleder19 (checked against the config filter on 2026-09-13).
        tags_any=[
            "mulleder19",
            "gh_wave6_aa",
            "gh_wave7_aa",
            "gh_wave8_aa",
            "gh_wave8_aasamp",
        ],
        groupby=[
            "aa_readout",
            "arm",
            "epochs",
            "thermo_source",
            "volume_table",
            "conc_space",
            "flux_draws",
        ],
        curves=[
            "val_pearson",
            "val_loss",
            "val_pred_sd_ratio",
            "train_data_loss",
            "flux_draw_pred_sd",
            "latent_sigma_p50",
        ],
        scalars=[
            "test_mean_spearman_pinned",
            "conc/pinned_test/mean_spearman",
            "select_val_pearson",
        ],
    ),
    "aasamp": ViewSpec(
        key="aasamp",
        title="027 wave 8: K latent draws on the amino-acid concentration readout",
        description=(
            "Mulleder amino acids, concentration readout with the draw-dependent "
            "concentration (log_c_from_latent), verified thermodynamic tables, constraint "
            "scale 0.1, 200 epochs. K = 64 against K = 1 draws per genotype per forward "
            "pass, both sampled, paired by seed 2700 to 2702. Read the paired difference; "
            "the across-draw prediction sd and the latent sigma show whether the draws are "
            "still distinct at the selected epoch."
        ),
        tags_any=["gh_wave8_aasamp"],
        groupby=["flux_draws"],
        curves=[
            "val_pearson",
            "val_loss",
            "flux_draw_pred_sd",
            "latent_sigma_p50",
            "c_kl",
            "train_data_loss",
        ],
        scalars=[
            "test_mean_spearman_pinned",
            "conc/pinned_test/mean_spearman",
            "select_val_pearson",
        ],
    ),
    "bxread": ViewSpec(
        key="bxread",
        title="027 wave 8: betaxanthin readouts, precursor concentration vs scalar vs pooled",
        description=(
            "Betaxanthin (Cachera), 200 epochs, matched settings at each seed. Arms: "
            "precursor-concentration readout (the same concentration mechanism the "
            "amino acids use, read on tyrosine and its shikimate precursors), the 026 "
            "scalar flux head, and the pooled control with no metabolic module. Score is "
            "the pinned test Spearman at the validation-selected epoch."
        ),
        tags_any=["gh_wave8_bx"],
        groupby=["bx_readout"],
        curves=[
            "val_pearson",
            "val_loss",
            "val_pred_sd_ratio",
            "train_data_loss",
            "latent_sigma_p50",
            "feas_flux_active_frac",
        ],
        scalars=["test_spearman_pinned", "select_val_pearson"],
    ),
    "aa_moduleon": ViewSpec(
        key="aa_moduleon",
        title="027: is the metabolic module on for amino acids? pooled vs flux-coupled vs context-only",
        description=(
            "Mulleder amino acids, 200 epochs, seeds 2700 to 2702 (plus 2703 on the "
            "concentration arm). The rank-1 read of the 2026-09-14 plan: the flux-coupled "
            "concentration readout (flux_concentration, wave 7) against the context-only "
            "concentration readout with no flux layer (context_concentration, wave 8) and "
            "the pooled control, plus the no-dissipation twin of the concentration arm. "
            "The kill rule was |ctxconc - conc| under 0.010 on the mean per-feature pinned "
            "test Spearman; measured paired mean +0.0019 (jobs 1899/1930/1931), so the "
            "module is off for this readout. Group by readout, then arm."
        ),
        tags_any=["w7aa", "w8aa_ctxconc", "w8aa_nodiss"],
        groupby=["aa_readout", "arm"],
        curves=[
            "val_pearson",
            "val_pred_sd_ratio",
            "val_loss",
            "train_data_loss",
            "train_loss",
            "latent_sigma_p50",
        ],
        scalars=[
            "test_mean_spearman_pinned",
            "conc/pinned_test/mean_spearman",
            "select_val_pearson",
        ],
    ),
    "bxctx": ViewSpec(
        key="bxctx",
        title="027 wave 9: betaxanthin, precursor concentration vs its context-only control vs scalar vs pooled",
        description=(
            "Betaxanthin (Cachera), 200 epochs, matched settings at each seed. The wave-8 "
            "three-way (precursor-concentration readout, 026 scalar flux head, pooled) "
            "plus the wave-9 context-only precursor readout on the pooled backbone with no "
            "flux layer (bxctxprec). Rank-4 read of the 2026-09-14 plan: |bxctxprec - "
            "bxprec| above the 0.012 paired sd means the layer does the work on this "
            "target; seed 2700 reads 0.079 vs 0.140 (job 1932). Group by readout."
        ),
        tags_any=["gh_wave8_bx", "gh_wave9_bx"],
        groupby=["bx_readout", "arm"],
        curves=[
            "val_pearson",
            "val_pred_sd_ratio",
            "val_loss",
            "train_data_loss",
            "latent_sigma_p50",
            "feas_flux_active_frac",
        ],
        scalars=["test_spearman_pinned", "select_val_pearson"],
    ),
    "ib": ViewSpec(
        key="ib",
        title="027 wave 9: isobutanol (Lopez 2024 biosensor), pooled vs scalar flux vs precursor concentration",
        description=(
            "Isobutanol YKO biosensor screen (log10 fold change, 504 pinned Merzbacher "
            "test genes, the 224 validated re-screen genes masked). Arms: pooled control, "
            "scalar flux head on the isobutanol pathway with product column r_1866, and "
            "the precursor-concentration readout on the 2-isopropylmalate pool the Leu3p "
            "biosensor binds; the scale-0 precursor cell is the mechanism test for the "
            "flat seed-2700 cell (job 1928). The linear baseline to clear is 0.198 (kNN on "
            "chromatin-pathway embeddings, job 1912). Group by readout and constraint scale."
        ),
        tags_any=["gh_wave9_ib"],
        groupby=["ib_readout", "constraint_scale", "arm"],
        curves=[
            "val_pearson",
            "val_pred_sd_ratio",
            "val_loss",
            "train_data_loss",
            "latent_sigma_p50",
            "feas_flux_active_frac",
        ],
        scalars=["test_spearman_pinned", "select_val_pearson"],
    ),
    "warmstart": ViewSpec(
        key="warmstart",
        title="027 wave 9: the warm-started residual head against the boxforce controls (feasibility first)",
        description=(
            "Betaxanthin, 200 epochs. The warm-started head (v_pre = pFBA reference + "
            "alpha N z, nominal abundance path, the 47-reaction thermo exemption, the "
            "TFA-consistent concentration init, the normalized availability fold, a "
            "two-sigma kcat ceiling, active-row balance 0.5 on a frozen row set, parsimony "
            "1e-3) against the wave-7 and wave-8 boxforce cells. At initialization it reads "
            "active-row balance max 0.031 and p95 0.002 with 82 active rows against the "
            "pFBA reference's 82 (jobs 1937 and 1945); the honest rule is max and p95 "
            "under 1e-2 on the top 100 at the selected epoch. Read the feasibility curves "
            "before the score."
        ),
        tags_any=["gh_wave9_warmstart", "gh_wave8_dens", "gh_wave8_densbal", "gh_wave7_ff"],
        groupby=["flux_parameterization", "arm", "flux_balance_active_weight"],
        curves=[
            "feas_balance_active_max",
            "feas_balance_active_p95",
            "feas_flux_active_frac",
            "feas_active_row_frac",
            "feas_thermo_violation_frac",
            "feas_box_violation_frac",
            "c_balance_active",
            "c_parsimony",
            "val_pred_sd_ratio",
            "val_pearson",
            "train_data_loss",
            "latent_sigma_p50",
        ],
        scalars=[
            "test_spearman_pinned",
            "feas/pinned_test_top100/balance_max_ratio_median",
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
    return [
        ws.Section(
            name="curves", panels=line_panels(curves, groupby=None), is_open=True
        ),
        ws.Section(
            name="scalars", panels=bar_panels(scalars, offset_rows=0), is_open=True
        ),
    ]


def workspace_runset(spec: ViewSpec) -> Any:
    return ws.RunsetSettings(
        filters=[ws.Tags().isin(spec.tags_any)],
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
            filters=[ws.Tags().isin(spec.tags_any)],
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
    parser.add_argument(
        "--contrast",
        default=None,
        help="kdraw | densbal | aa | aa_all | aasamp | bxread | all",
    )
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
        if missing:
            print(f"  not logged by any matched run: {missing}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(record, fh, indent=2)
    print(out_path)


if __name__ == "__main__":
    main()
