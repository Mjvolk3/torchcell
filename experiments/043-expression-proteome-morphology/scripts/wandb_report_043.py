# experiments/043-expression-proteome-morphology/scripts/wandb_report_043.py
# [[experiments.043-expression-proteome-morphology.scripts.wandb_report_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/wandb_report_043
"""W&B Report with one runset per panel grid: loss curves together and separated.

A saved view (``wandb_view_043.py``) has one runset for every panel, so it cannot hold a
panel per arm restricted to that arm's runs, nor a pair restricted to the HL-Gauss cells.
A Report holds several ``PanelGrid`` blocks, each with its own ``Runset`` filtered by run
id, so it carries both. The layout is the curated one of the views (owner, 2026-10-24:
"we want loss curves together and separated"): 1 loss together, one grid per arm with
train/loss and val/loss on one axis, mean over the arm's runs with the min/max band; 2
loss separated, train/loss and val/loss of every run on their own axes (color = arm,
dashed = fine-tune), then the same pair for the HL-Gauss z-score cells only; 3 the six
validation metrics grouped by arm; 4 optimization; 5 bookkeeping. Grids are built from
the round's runs as W&B holds them now, so rerun after every sync.

The report is created on the first run and its id and URL are recorded in
``results/wandb_view_ids.json`` under ``<round>_report``; later runs overwrite the same
report in place. After saving, the script reads the report back, counts grids and
panels, and checks that every metric key it plots is logged by at least one run of the
grid that plots it.

    python experiments/043-expression-proteome-morphology/scripts/wandb_report_043.py --round overnight_2026_10_23
"""

from __future__ import annotations

import argparse
import json
import os.path as osp
import re

import wandb
import wandb_workspaces.expr as expr
import wandb_workspaces.reports.v2 as wr
from pydantic import BaseModel, ConfigDict
from wandb_view_043 import (
    BOOKKEEPING,
    HLGAUSS_CELL,
    LOSS_KEYS,
    ROUNDS,
    SMOOTHING_TYPE,
    SMOOTHING_WEIGHT,
    VAL_METRICS,
    Arm,
    collect_arms,
    series_styles,
)

ENTITY = "zhao-group"
PROJECT = "torchcell_043-expression-proteome-morphology"
VIEW_REGISTRY = osp.join(
    osp.dirname(osp.dirname(osp.abspath(__file__))), "results", "wandb_view_ids.json"
)
EPOCH = "epoch"
#: The optimizer step: the trainer logs at step=step, and W&B's "Step" is ``_step``.
STEP = "Step"
STEP_KEY = "_step"
MAX_RUNS_SHOWN = 1000
LEGEND = "${run:displayName}"
GRAD_NORM = "train/grad_norm"
#: Report titles; the report id (and so its URL stem) is kept in the registry.
TITLES = {
    "overnight_2026_10_23": "043 overnight round 2026-10-23: loss curves",
    "ghyper_sweep_2026_10_24": "043 GilaHyper sweep 2026-10-24: loss curves",
    "aurora_2026_10_24": "043 Aurora capacity grid 2026-10-24: loss curves",
}
PARTIAL = "a run that is not finished is partial and has no final result"


class PanelSpec(BaseModel):
    """One LinePlot: the keys on y, the x axis, and whether it is grouped by arm."""

    model_config = ConfigDict(frozen=True)

    y: tuple[str, ...]
    x: str  # EPOCH or STEP
    title: str
    log_y: bool = False
    #: mean per arm (config ``arm_label``) with the min/max band over its runs
    grouped: bool = False
    #: one line per run colored by its arm (``<run id>:<metric>`` keys)
    colors: dict[str, str] | None = None
    marks: dict[str, str] | None = None
    #: panel width in the 24-unit grid
    w: int = 12


class GridSpec(BaseModel):
    """One panel grid: its section, a heading, a one-sentence caveat, its runs, its
    panels.
    """

    model_config = ConfigDict(frozen=True)

    section: str
    heading: str
    caption: str
    run_ids: tuple[str, ...]
    panels: tuple[PanelSpec, ...]


class ReportSpec(BaseModel):
    """The whole report for one round."""

    model_config = ConfigDict(frozen=True)

    title: str
    intro: str
    grids: tuple[GridSpec, ...]


def _x_name(x: str) -> str:
    return "epoch" if x == EPOCH else "optimizer step"


def _arm_caption(a: Arm) -> str:
    """Runs, seeds, objective and epochs reached of one arm, in one sentence."""
    shown = [i for i, r in enumerate(a.run_ids) if r in a.loss_run_ids]
    seeds = ", ".join(str(a.seeds[i]) for i in shown)
    reached = [a.epochs_reached[i] for i in shown]
    states = {a.states[i] for i in shown}
    span = (
        f"epoch {reached[0]}"
        if len(set(reached)) == 1
        else f"epochs {min(reached)} to {max(reached)}"
    )
    tail = "" if states == {"finished"} else f"; {PARTIAL}"
    plural = "s" if len(shown) != 1 else ""
    return (
        f"{len(shown)} run{plural} (seed {seeds}), {a.objective}, reached {span}; "
        "train/loss and val/loss are the mean over the runs with the min/max band"
        f"{tail}."
    )


def _separated(arm_set: list[Arm], loss_y: tuple[str, ...]) -> tuple[PanelSpec, ...]:
    """train/loss and val/loss on their own axes, one line per run colored by arm."""
    out = []
    for k in loss_y:
        colors, marks = series_styles(arm_set, k)
        out.append(
            PanelSpec(
                y=(k,),
                x=EPOCH,
                title=f"{k} by epoch, every run (color = arm, dashed = fine-tune)",
                colors=colors,
                marks=marks,
            )
        )
    return tuple(out)


def build_spec(key: str) -> ReportSpec:
    """The report in the curated section order, from the round's runs in W&B."""
    arms = collect_arms(wandb.Api(timeout=120), ROUNDS[key])
    with_loss = [a for a in arms if a.loss_run_ids]
    present = set().union(*(a.keys for a in arms))
    loss_y = tuple(k for k in LOSS_KEYS if k in present)
    grids: list[GridSpec] = []
    for a in with_loss:
        y = tuple(k for k in LOSS_KEYS if k in a.keys)
        grids.append(
            GridSpec(
                section="1 Loss, together",
                heading=f"{a.label}: {' and '.join(y)} by epoch",
                caption=_arm_caption(a),
                run_ids=a.loss_run_ids,
                panels=(
                    PanelSpec(
                        y=y,
                        x=EPOCH,
                        title=(
                            f"{a.label}: {' and '.join(y)} by epoch, mean with the "
                            "min/max band"
                        ),
                        grouped=True,
                        w=24,
                    ),
                ),
            )
        )
    if with_loss:
        every = tuple(r for a in with_loss for r in a.loss_run_ids)
        grids.append(
            GridSpec(
                section="2 Loss, separated",
                heading="Every run: train/loss and val/loss on separate axes",
                caption=(
                    f"All {len(every)} runs of the round that logged a loss, one line per "
                    "run colored by arm (fine-tunes dashed in their stage 1 color); loss "
                    "families and epoch budgets differ, so read shapes here and levels in "
                    f"the HL-Gauss grid below; {PARTIAL}."
                ),
                run_ids=every,
                panels=_separated(with_loss, loss_y),
            )
        )
    cells = [a for a in with_loss if a.category == HLGAUSS_CELL]
    if cells:
        left_out = [a.label for a in with_loss if a.category != HLGAUSS_CELL]
        grids.append(
            GridSpec(
                section="2 Loss, separated",
                heading="HL-Gauss z-score cells only: train/loss and val/loss",
                caption=(
                    "Only the stage 1 cells trained with HL-Gauss on z-score targets, so "
                    "loss levels are comparable; left out (another objective, the alone "
                    "Kemmeren runs on thousands of epochs, or a Kemmeren fine-tune): "
                    f"{', '.join(left_out) if left_out else 'none'}; {PARTIAL}."
                ),
                run_ids=tuple(r for a in cells for r in a.loss_run_ids),
                panels=_separated(cells, loss_y),
            )
        )
    every_run = tuple(r for a in arms for r in a.run_ids)
    val = tuple(
        PanelSpec(y=(k,), x=EPOCH, title=f"{name} by epoch, mean per arm", grouped=True)
        for k, name in VAL_METRICS
        if k in present
    )
    if val:
        grids.append(
            GridSpec(
                section="3 Validation metrics",
                heading="Validation metrics by epoch, grouped by arm",
                caption=(
                    "Mean per arm with the min/max band over its runs; a maximum read off "
                    f"a curve is an upward-biased order statistic, not a score; {PARTIAL}."
                ),
                run_ids=every_run,
                panels=val,
            )
        )
    opt = []
    if GRAD_NORM in present:
        opt.append(
            PanelSpec(
                y=(GRAD_NORM,),
                x=STEP,
                title="train/grad_norm (pre-clip, log y) by optimizer step, mean per arm",
                log_y=True,
                grouped=True,
            )
        )
    if "train/lr" in present:
        opt.append(
            PanelSpec(
                y=("train/lr",),
                x=STEP,
                title="train/lr by optimizer step, mean per arm",
                grouped=True,
            )
        )
    if opt:
        grids.append(
            GridSpec(
                section="4 Optimization",
                heading="Gradient norm and learning rate by optimizer step",
                caption=(
                    "train/grad_norm before clipping on a log axis and the logged "
                    f"learning rate, grouped by arm; {PARTIAL}."
                ),
                run_ids=every_run,
                panels=tuple(opt),
            )
        )
    book = tuple(
        PanelSpec(y=tuple(ys), x=EPOCH, title=t, grouped=True)
        for t, ys in BOOKKEEPING
        if all(k in present for k in ys)
    )
    if book:
        grids.append(
            GridSpec(
                section="5 Bookkeeping",
                heading="Epoch time and budget",
                caption=(
                    "Seconds per epoch, steps per epoch and the fixed epoch budget, "
                    "grouped by arm."
                ),
                run_ids=every_run,
                panels=book,
            )
        )
    return ReportSpec(
        title=TITLES[key],
        intro=(
            f"Round {key}: {ROUNDS[key].intro} Hardware: {ROUNDS[key].hardware}. "
            "Sections: 1 train/loss and val/loss together, one grid per arm; 2 the same "
            "losses on separate axes, every run and then the HL-Gauss z-score cells only; "
            "3 validation metrics; 4 optimization; 5 bookkeeping. Smoothing is a "
            f"{SMOOTHING_WEIGHT}-point running average. Built by "
            "experiments/043-expression-proteome-morphology/scripts/wandb_report_043.py; "
            "rerun it after every sync."
        ),
        grids=tuple(grids),
    )


def _line_plot(p: PanelSpec, i: int) -> wr.LinePlot:
    """Panel ``i`` of its grid, two per row at w=12, one per row at w=24 (h=8)."""
    per_row = 24 // p.w
    return wr.LinePlot(
        title=p.title,
        x=p.x,
        y=list(p.y),
        title_x=_x_name(p.x),
        smoothing_type=SMOOTHING_TYPE,
        smoothing_factor=SMOOTHING_WEIGHT,
        log_y=p.log_y,
        max_runs_to_show=MAX_RUNS_SHOWN,
        **(
            {
                "groupby": wr.Config("arm_label"),
                "groupby_aggfunc": "mean",
                "groupby_rangefunc": "minmax",
            }
            if p.grouped
            else {"legend_template": LEGEND}
        ),
        line_colors=p.colors,
        line_marks=p.marks,
        layout=wr.Layout(x=p.w * (i % per_row), y=8 * (i // per_row), w=p.w, h=8),
    )


def build_blocks(spec: ReportSpec) -> list:
    blocks: list = [wr.MarkdownBlock(text=spec.intro)]
    section = None
    for grid in spec.grids:
        if grid.section != section:
            section = grid.section
            blocks.append(wr.H1(text=section))
        blocks.append(wr.H2(text=grid.heading))
        blocks.append(wr.MarkdownBlock(text=grid.caption))
        runset = wr.Runset(
            entity=ENTITY,
            project=PROJECT,
            name=grid.heading,
            filters=[expr.Metric("ID").isin(list(grid.run_ids))],
        )
        blocks.append(
            wr.PanelGrid(
                runsets=[runset],
                panels=[_line_plot(p, i) for i, p in enumerate(grid.panels)],
            )
        )
    return blocks


def _registry() -> dict:
    with open(VIEW_REGISTRY) as f:
        return json.load(f)


def save_report(key: str, spec: ReportSpec) -> str:
    """Create the report once; afterwards overwrite the same report id in place."""
    slot = f"{key}_report"
    reg = _registry()
    entry = reg.get(slot)
    if entry is None:
        report = wr.Report(
            entity=ENTITY,
            project=PROJECT,
            title=spec.title,
            width="fluid",
            blocks=build_blocks(spec),
        )
    else:
        report = wr.Report.from_url(entry["url"])
        if report.id != entry["id"]:
            raise ValueError(f"registry id {entry['id']} != loaded {report.id}")
        report.title = spec.title
        report.width = "fluid"
        report.blocks = build_blocks(spec)
    report.save()
    if entry is not None and report.id != entry["id"]:
        raise ValueError(f"report id changed on update: {entry['id']} -> {report.id}")
    reg[slot] = {"id": report.id, "url": report.url, "name": spec.title}
    with open(VIEW_REGISTRY, "w") as f:
        json.dump(reg, f, indent=2, sort_keys=True)
        f.write("\n")
    return report.url


def verify(url: str, spec: ReportSpec) -> None:
    """Read the report back, count grids and panels per section, and check every
    plotted key against the summaries and sampled history of the grid's runs.
    """
    loaded = wr.Report.from_url(url)
    grids = [b for b in loaded.blocks if isinstance(b, wr.PanelGrid)]
    n_panels = sum(len(g.panels) for g in grids)
    want_panels = sum(len(g.panels) for g in spec.grids)
    if len(grids) != len(spec.grids) or n_panels != want_panels:
        raise ValueError(
            f"expected {len(spec.grids)} grids / {want_panels} panels, "
            f"found {len(grids)} / {n_panels}"
        )
    per_section: dict[str, list[int]] = {}
    for g_spec, g in zip(spec.grids, grids, strict=True):
        if len(g.panels) != len(g_spec.panels):
            raise ValueError(f"{g_spec.heading}: panel count differs on read-back")
        per_section.setdefault(g_spec.section, []).append(len(g.panels))
    print(f"report id {loaded.id}: {len(grids)} panel grids, {n_panels} panels")
    for name, counts in per_section.items():
        print(f"  {name}: {len(counts)} grids, {sum(counts)} panels")
    api = wandb.Api()
    keys_by_run: dict[str, set[str]] = {}
    for grid in spec.grids:
        for rid in grid.run_ids:
            if rid not in keys_by_run:
                run = api.run(f"{ENTITY}/{PROJECT}/{rid}")
                keys = set(run.summary._json_dict)
                for row in run.history(samples=50, pandas=False):
                    keys.update(row)
                keys_by_run[rid] = keys
    missing = []
    for grid in spec.grids:
        keys = {k for p in grid.panels for k in p.y}
        keys |= {STEP_KEY if p.x == STEP else EPOCH for p in grid.panels}
        for k in sorted(keys):
            if not any(k in keys_by_run[r] for r in grid.run_ids):
                missing.append((grid.heading, k))
    if missing:
        raise ValueError(f"keys logged by no run of their grid: {missing}")
    print("every plotted key is logged by a run of its grid")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--round", required=True, choices=sorted(TITLES))
    args = parser.parse_args()
    spec = build_spec(args.round)
    url = save_report(args.round, spec)
    if re.search(r"/reports/", url) is None:
        raise ValueError(f"not a report URL: {url}")
    verify(url, spec)
    print(f"\n{url}")


if __name__ == "__main__":
    main()
