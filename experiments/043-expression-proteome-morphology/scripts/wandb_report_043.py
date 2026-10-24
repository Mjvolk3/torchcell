# experiments/043-expression-proteome-morphology/scripts/wandb_report_043.py
# [[experiments.043-expression-proteome-morphology.scripts.wandb_report_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/wandb_report_043
"""W&B Report with one runset per panel grid, for readable 043 loss curves.

A saved view (``wandb_view_043.py``) has one runset, so its total-loss panel puts every
run of a round on one axis: HL-Gauss (about 2 to 3) beside pinball (about 0.2), and the
alone_kemmeren runs (3,783 epochs) squash the 80-epoch and 20-epoch runs into the left
edge. A Report holds several ``PanelGrid`` blocks, each with its own ``Runset`` filtered
by run id, so each grid shows only a few comparable runs.

The report is created on the first run and its id and URL are recorded in
``results/wandb_view_ids.json`` under ``<round>_report``; later runs overwrite the same
report in place (URL unchanged). After saving, the script reads the report back, counts
grids and panels, and checks that every metric key it plots is logged by at least one run
of the grid that plots it.

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

ENTITY = "zhao-group"
PROJECT = "torchcell_043-expression-proteome-morphology"
VIEW_REGISTRY = osp.join(
    osp.dirname(osp.dirname(osp.abspath(__file__))), "results", "wandb_view_ids.json"
)
EPOCH = "epoch"
#: The optimizer step: the trainer logs at step=step, and W&B's "Step" is ``_step``.
STEP = "Step"
STEP_KEY = "_step"
MAX_RUNS_SHOWN = 20
LEGEND = "${run:displayName} ${metricName}"
LOSS_KEYS = ("train/loss", "val/loss")
KEMMEREN = "val/kemmeren/pearson_per_gene_median"
DOUBLES = "val/doubles/pearson_per_strain_centered_median"
MUENZNER = "val/muenzner/pearson_per_gene_median"
CAUDAL = "val/caudal/pearson_per_gene_median"
MESSNER = "val/messner/pearson_per_gene_median"
VAISHNAV = "val/vaishnav/pearson"
#: Panel-title names that say the statistic.
METRIC_TITLE = {
    "train/loss and val/loss": "train/loss and val/loss",
    KEMMEREN: "val/kemmeren per-gene Pearson median",
    DOUBLES: "val/doubles (Sameith) per-double centered Pearson median",
    MUENZNER: "val/muenzner per-gene Pearson median",
    CAUDAL: "val/caudal per-gene Pearson median",
    MESSNER: "val/messner per-gene Pearson median",
    VAISHNAV: "val/vaishnav Pearson",
}


class PanelSpec(BaseModel):
    """One LinePlot: the keys on y and the x axis."""

    model_config = ConfigDict(frozen=True)

    y: tuple[str, ...]
    x: str  # EPOCH or STEP
    title: str


class GridSpec(BaseModel):
    """One panel grid: a heading, a one-sentence caveat, its runs, its panels."""

    model_config = ConfigDict(frozen=True)

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


def _panel(y: tuple[str, ...], x: str) -> PanelSpec:
    name = "train/loss and val/loss" if y == LOSS_KEYS else METRIC_TITLE[y[0]]
    return PanelSpec(y=y, x=x, title=f"{name} by {_x_name(x)}")


def _loss_pair() -> tuple[PanelSpec, ...]:
    return (_panel(LOSS_KEYS, EPOCH), _panel(LOSS_KEYS, STEP))


HLGAUSS_S1 = ("gdqlsjpb", "oc881bqm")
PINBALL_S1 = ("05d5anh5",)
FINETUNES = ("jxlvzmmb", "zbsnrjqn", "fth871fz")
ALONE = ("18s6j2p2", "bc1awz4y")
JOINT_9 = ("86cc6tgp",)

REPORTS: dict[str, ReportSpec] = {
    "overnight_2026_10_23": ReportSpec(
        title="043 overnight round 2026-10-23: loss curves",
        intro=(
            "Round overnight_2026_10_23 (store v2), one runset per panel grid so each "
            "grid holds a few comparable runs on its own axes. No smoothing anywhere; "
            "curves are raw per-epoch values, and a maximum read off a curve is an "
            "upward-biased order statistic, not a score. Built by "
            "experiments/043-expression-proteome-morphology/scripts/wandb_report_043.py."
        ),
        grids=(
            GridSpec(
                heading="Stage 1, HL-Gauss: train/loss and val/loss by epoch",
                caption=(
                    "Stage 1 all-dataset HL-Gauss runs, seeds 0 (gdqlsjpb) and 1 "
                    "(oc881bqm), 80 epochs; HL-Gauss loss is a cross-entropy over bins, "
                    "so its scale is not comparable to the pinball grid below."
                ),
                run_ids=HLGAUSS_S1,
                panels=_loss_pair(),
            ),
            GridSpec(
                heading="Stage 1, pinball: train/loss and val/loss by epoch",
                caption=(
                    "Stage 1 all-dataset pinball run, seed 0 only (05d5anh5), 80 epochs, "
                    "on its own y axis because the loss family differs; one seed, so no "
                    "seed spread is shown."
                ),
                run_ids=PINBALL_S1,
                panels=_loss_pair(),
            ),
            GridSpec(
                heading=(
                    "Alone Kemmeren against stage 1 (HL-Gauss) at matched optimizer "
                    "steps"
                ),
                caption=(
                    "Alone Kemmeren runs (18s6j2p2 s0, bc1awz4y s1; 3,783 epochs) "
                    "against the stage 1 HL-Gauss runs on the optimizer-step axis, the "
                    "only axis on which they are comparable; the stage 1 loss covers every "
                    "dataset, not Kemmeren alone, so read the Kemmeren and doubles medians for the matched "
                    "comparison, not the loss levels."
                ),
                run_ids=ALONE + HLGAUSS_S1,
                panels=(
                    _panel(LOSS_KEYS, STEP),
                    _panel((KEMMEREN,), STEP),
                    _panel((DOUBLES,), STEP),
                ),
            ),
            GridSpec(
                heading="Kemmeren fine-tunes (20 epochs) from each stage 1",
                caption=(
                    "Kemmeren fine-tunes from stage 1 HL-Gauss (jxlvzmmb s0, zbsnrjqn s1) "
                    "and from stage 1 pinball (fth871fz s0), 20 epochs each; the pinball "
                    "fine-tune trains with pinball loss, so compare its loss shape, not "
                    "its level, with the HL-Gauss fine-tunes."
                ),
                run_ids=FINETUNES,
                panels=(
                    _panel(LOSS_KEYS, EPOCH),
                    _panel((KEMMEREN,), EPOCH),
                    _panel((DOUBLES,), EPOCH),
                ),
            ),
            GridSpec(
                heading="Stage 1 validation metrics by epoch",
                caption=(
                    "Stage 1 HL-Gauss (gdqlsjpb, oc881bqm), stage 1 pinball (05d5anh5) "
                    "and joint_9 (86cc6tgp, 15 epochs, deadline-capped, so partial) on "
                    "validation medians per epoch; joint_9 stopped early, so its curves "
                    "end at epoch 15 and are not a final result."
                ),
                run_ids=HLGAUSS_S1 + PINBALL_S1 + JOINT_9,
                panels=tuple(
                    _panel((k,), EPOCH)
                    for k in (KEMMEREN, DOUBLES, MUENZNER, CAUDAL, MESSNER, VAISHNAV)
                ),
            ),
        ),
    )
}


def _line_plot(p: PanelSpec) -> wr.LinePlot:
    return wr.LinePlot(
        title=p.title,
        x=p.x,
        y=list(p.y),
        title_x=_x_name(p.x),
        smoothing_type="none",
        max_runs_to_show=MAX_RUNS_SHOWN,
        legend_template=LEGEND,
        layout=wr.Layout(w=12, h=8),
    )


def build_blocks(spec: ReportSpec) -> list:
    blocks: list = [wr.MarkdownBlock(text=spec.intro)]
    for grid in spec.grids:
        blocks.append(wr.H2(text=grid.heading))
        blocks.append(wr.MarkdownBlock(text=grid.caption))
        runset = wr.Runset(
            entity=ENTITY,
            project=PROJECT,
            name=grid.heading,
            filters=[expr.Metric("ID").isin(list(grid.run_ids))],
        )
        blocks.append(
            wr.PanelGrid(runsets=[runset], panels=[_line_plot(p) for p in grid.panels])
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
    """Read the report back, count grids and panels, and check every plotted key."""
    loaded = wr.Report.from_url(url)
    grids = [b for b in loaded.blocks if isinstance(b, wr.PanelGrid)]
    n_panels = sum(len(g.panels) for g in grids)
    want_panels = sum(len(g.panels) for g in spec.grids)
    print(f"report id {loaded.id}: {len(grids)} panel grids, {n_panels} panels")
    if len(grids) != len(spec.grids) or n_panels != want_panels:
        raise ValueError(
            f"expected {len(spec.grids)} grids / {want_panels} panels, "
            f"found {len(grids)} / {n_panels}"
        )
    api = wandb.Api()
    keys_by_run: dict[str, set[str]] = {}
    for grid in spec.grids:
        for rid in grid.run_ids:
            if rid not in keys_by_run:
                run = api.run(f"{ENTITY}/{PROJECT}/{rid}")
                keys = set(run.summary._json_dict)
                hist = run.history(samples=50, pandas=False)
                for row in hist:
                    keys.update(row)
                keys_by_run[rid] = keys
    print("\n| grid | key | runs logging it | runs in grid |")
    print("|---|---|---|---|")
    missing = []
    for i, grid in enumerate(spec.grids, start=1):
        keys = {k for p in grid.panels for k in p.y}
        keys |= {STEP_KEY if p.x == STEP else EPOCH for p in grid.panels}
        for k in sorted(keys):
            have = [r for r in grid.run_ids if k in keys_by_run[r]]
            print(
                f"| {i} | {k} | {len(have)} ({', '.join(have)}) | {len(grid.run_ids)} |"
            )
            if not have:
                missing.append((i, k))
    if missing:
        raise ValueError(f"keys logged by no run of their grid: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--round", required=True, choices=sorted(REPORTS))
    args = parser.parse_args()
    spec = REPORTS[args.round]
    url = save_report(args.round, spec)
    if re.search(r"/reports/", url) is None:
        raise ValueError(f"not a report URL: {url}")
    verify(url, spec)
    print(f"\n{url}")


if __name__ == "__main__":
    main()
