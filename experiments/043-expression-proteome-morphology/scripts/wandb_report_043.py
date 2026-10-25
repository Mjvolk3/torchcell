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
of the grid that plots it. Every id must appear in ``ROUNDS`` of ``wandb_view_043.py``; an
id W&B does not hold yet (a pending cell) is skipped and printed, and a grid with no run
yet is left out, so rerun the script as cells start.

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
from wandb_view_043 import ROUNDS

ENTITY = "zhao-group"
PROJECT = "torchcell_043-expression-proteome-morphology"
VIEW_REGISTRY = osp.join(
    osp.dirname(osp.dirname(osp.abspath(__file__))), "results", "wandb_view_ids.json"
)
EPOCH = "epoch"
#: The optimizer step: the trainer logs at step=step, and W&B's "Step" is ``_step``.
STEP = "Step"
STEP_KEY = "_step"
MAX_RUNS_SHOWN = 30
LEGEND = "${run:displayName} ${metricName}"
LOSS_KEYS = ("train/loss", "val/loss")
KEMMEREN = "val/kemmeren/pearson_per_gene_median"
DOUBLES = "val/doubles/pearson_per_strain_centered_median"
MUENZNER = "val/muenzner/pearson_per_gene_median"
CAUDAL = "val/caudal/pearson_per_gene_median"
MESSNER = "val/messner/pearson_per_gene_median"
VAISHNAV = "val/vaishnav/pearson"
GRAD_NORM = "train/grad_norm"
#: Panel-title names that say the statistic.
METRIC_TITLE = {
    "train/loss and val/loss": "train/loss and val/loss",
    "val/loss": "val/loss",
    KEMMEREN: "val/kemmeren per-gene Pearson median",
    DOUBLES: "val/doubles (Sameith) per-double centered Pearson median",
    MUENZNER: "val/muenzner per-gene Pearson median",
    CAUDAL: "val/caudal per-gene Pearson median",
    MESSNER: "val/messner per-gene Pearson median",
    VAISHNAV: "val/vaishnav Pearson",
    GRAD_NORM: "train/grad_norm (pre-clip, log scale)",
}


class PanelSpec(BaseModel):
    """One LinePlot: the keys on y and the x axis."""

    model_config = ConfigDict(frozen=True)

    y: tuple[str, ...]
    x: str  # EPOCH or STEP
    title: str
    log_y: bool = False


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


def _panel(y: tuple[str, ...], x: str, log_y: bool = False) -> PanelSpec:
    name = "train/loss and val/loss" if y == LOSS_KEYS else METRIC_TITLE[y[0]]
    return PanelSpec(y=y, x=x, title=f"{name} by {_x_name(x)}", log_y=log_y)


def _loss_pair() -> tuple[PanelSpec, ...]:
    return (_panel(LOSS_KEYS, EPOCH), _panel(LOSS_KEYS, STEP))


METRICS = (KEMMEREN, DOUBLES, MUENZNER, CAUDAL, MESSNER, VAISHNAV)


def _metric_panels(x: str = EPOCH) -> tuple[PanelSpec, ...]:
    return tuple(_panel((k,), x) for k in METRICS)


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


GS = "gs1024_"
LAYER_PATTERNS = tuple(
    f"{GS}w1_{a}" for a in ("smss", "ssss", "mmmm", "msmsms", "smmss")
) + (f"{GS}w2_mmmm_cos",)  # MMMM with the cosine schedule (the divergence test)
GS_BASELINE = f"{GS}w1_smss"
GS_DIMS_OPT = (f"{GS}w2_h288",) + tuple(
    f"{GS}w3_{a}" for a in ("ffn2", "hh256", "lrhalf", "lrx2")
)
GS_STAGE1 = LAYER_PATTERNS + (f"{GS}w2_qn", f"{GS}w2_mle", f"{GS}w2_binq") + GS_DIMS_OPT
#: Objective-changing cells: each on its own grid, since their losses are on different
#: targets or a different loss family.
GS_OBJECTIVES = (
    ("qn", "quantile_normal target transform", f"{GS}w2_qn"),
    ("mle", "mle loss family", f"{GS}w2_mle"),
    ("binq", "binned_quantile loss family", f"{GS}w2_binq"),
)
PARTIAL = "runs still in flight are partial and have no final result"

REPORTS["ghyper_sweep_2026_10_24"] = ReportSpec(
    title="043 GilaHyper sweep 2026-10-24: loss curves",
    intro=(
        "Round ghyper_sweep_2026_10_24 (store v2, seed 0, 80 epochs, HL-Gauss unless the "
        "grid says otherwise), one runset per panel grid. No smoothing; "
        f"{PARTIAL}, and a maximum read off a curve is an upward-biased order "
        "statistic, not a score. Run ids not yet in W&B are left out until the script "
        "is rerun. Built by "
        "experiments/043-expression-proteome-morphology/scripts/wandb_report_043.py."
    ),
    grids=(
        GridSpec(
            heading=(
                "Layer pattern, HL-Gauss stage 1: train/loss and val/loss by epoch"
            ),
            caption=(
                "Trunk layer patterns SMSS (baseline), SSSS, MMMM, MSMSMS and SMMSS "
                f"under one HL-Gauss objective, so loss levels are comparable; {PARTIAL}."
            ),
            run_ids=LAYER_PATTERNS,
            panels=(_panel(LOSS_KEYS, EPOCH), _panel(("val/loss",), EPOCH)),
        ),
        GridSpec(
            heading="Gradient norm by optimizer step (divergence)",
            caption=(
                "train/grad_norm before clipping (clip 10), log scale, every stage 1 "
                "cell: MMMM and MSMSMS exceed 1e3 from epoch 71.5 and 72.4 (7,101 steps "
                f"per epoch) and diverge inside their scoring window; {PARTIAL}."
            ),
            run_ids=GS_STAGE1,
            panels=(_panel((GRAD_NORM,), STEP, log_y=True),),
        ),
        GridSpec(
            heading="Layer pattern: validation metrics by epoch",
            caption=(
                "The same five layer-pattern runs on the validation medians and the "
                f"Vaishnav Pearson per epoch; {PARTIAL}."
            ),
            run_ids=LAYER_PATTERNS,
            panels=_metric_panels(),
        ),
    )
    + tuple(
        GridSpec(
            heading=(
                f"Preprocessing and loss family (each its own objective): {tag}, {what}"
            ),
            caption=(
                f"{rid} alone, because its loss is on a different objective than every "
                f"other cell, so its loss level is comparable with nothing; {PARTIAL}."
            ),
            run_ids=(rid,),
            panels=(_panel(LOSS_KEYS, EPOCH),) + _metric_panels(),
        )
        for tag, what, rid in GS_OBJECTIVES
    )
    + (
        GridSpec(
            heading="Dims and optimization",
            caption=(
                "Hidden 288, trunk FFN 2x, head_hidden 256, lr 1.5e-4 and lr 6e-4 "
                "against the SMSS baseline, all HL-Gauss, so loss levels are "
                f"comparable; {PARTIAL}."
            ),
            run_ids=(GS_BASELINE,) + GS_DIMS_OPT,
            panels=(_panel(LOSS_KEYS, EPOCH),) + _metric_panels(),
        ),
        GridSpec(
            heading="Kemmeren fine-tunes (20 epochs)",
            caption=(
                "Every Kemmeren fine-tune (the _ft run of a stage 1 cell, 20 epochs) on "
                "loss and the Kemmeren and doubles medians; fine-tunes inherit their "
                "stage 1 objective, so compare loss levels only within one objective; "
                f"{PARTIAL}."
            ),
            run_ids=tuple(f"{r}_ft" for r in GS_STAGE1),
            panels=(
                _panel(LOSS_KEYS, EPOCH),
                _panel((KEMMEREN,), EPOCH),
                _panel((DOUBLES,), EPOCH),
            ),
        ),
    ),
)

AB = "a1024b_all_"
AURORA_ANCHORS = tuple(f"{AB}onto_hlgauss_equal_s{s}" for s in range(3)) + tuple(
    f"{AB}name_hlgauss_equal_s{s}" for s in range(4)
)
AURORA_CAPACITY = tuple(
    f"{AB}onto_hlgauss_equal_{c}_s{s}"
    for c, seeds in (("bs16", 3), ("bs32", 3), ("h288", 3), ("ffn2", 2))
    for s in range(seeds)
)
AURORA_ALONE = "a1024_alone_kemmeren_onto_hlgauss_equal_s0"

REPORTS["aurora_2026_10_24"] = ReportSpec(
    title="043 Aurora capacity grid 2026-10-24: loss curves",
    intro=(
        "Round aurora_2026_10_24 (store v2, micro-batch 1 resubmission a1024b_, 80 "
        "epochs), one runset per panel grid. No smoothing; "
        f"{PARTIAL}, and a maximum read off a curve is an upward-biased order "
        "statistic, not a score. Run ids not yet in W&B are left out until the script "
        "is rerun. Built by "
        "experiments/043-expression-proteome-morphology/scripts/wandb_report_043.py."
    ),
    grids=(
        GridSpec(
            heading="Ontology vs name anchors: train/loss and val/loss",
            caption=(
                "Ontology-conditioned anchors (all_onto, seeds 0-2) against name "
                "anchors (all_name, seeds 0-3), both HL-Gauss at batch 8, by epoch and "
                f"by optimizer step; {PARTIAL}."
            ),
            run_ids=AURORA_ANCHORS,
            panels=_loss_pair(),
        ),
        GridSpec(
            heading="Capacity axis (bs16, bs32, h288, ffn2): train/loss and val/loss",
            caption=(
                "Batch 16 and 32, hidden 288 and FFN 2x cells under ontology "
                "conditioning, by epoch and by optimizer step (bs16 and bs32 take fewer "
                f"steps per epoch, so the two axes differ); {PARTIAL}."
            ),
            run_ids=AURORA_CAPACITY,
            panels=_loss_pair(),
        ),
        GridSpec(
            heading="Pinball cell: train/loss and val/loss",
            caption=(
                "The pinball loss-family cell alone, on its own y axis because its "
                f"loss is not on the HL-Gauss scale; {PARTIAL}."
            ),
            run_ids=(f"{AB}onto_pinball_equal_s0",),
            panels=_loss_pair(),
        ),
        GridSpec(
            heading="MSE cell: train/loss and val/loss",
            caption=(
                "The mse loss-family cell alone, on its own y axis because its loss is "
                f"not on the HL-Gauss scale; {PARTIAL}."
            ),
            run_ids=(f"{AB}onto_mse_equal_s0",),
            panels=_loss_pair(),
        ),
        GridSpec(
            heading="Alone Kemmeren (ontology) by optimizer step",
            caption=(
                "The alone Kemmeren run under ontology conditioning on loss and the "
                "Kemmeren and doubles medians by optimizer step, the axis it is "
                f"step-matched on (571,100 steps); {PARTIAL}."
            ),
            run_ids=(AURORA_ALONE,),
            panels=(
                _panel(LOSS_KEYS, STEP),
                _panel((KEMMEREN,), STEP),
                _panel((DOUBLES,), STEP),
            ),
        ),
    ),
)


def resolve(key: str, spec: ReportSpec) -> ReportSpec:
    """Check every id against ``ROUNDS[key]``, drop ids W&B does not hold yet (printed),
    and drop a grid none of whose runs exist yet (printed).
    """
    wanted = {r for g in spec.grids for r in g.run_ids}
    stray = sorted(wanted - set(ROUNDS[key].run_ids))
    if stray:
        raise ValueError(f"ids not in ROUNDS[{key!r}]: {stray}")
    found = {
        r.id
        for r in wandb.Api().runs(
            f"{ENTITY}/{PROJECT}", filters={"name": {"$in": sorted(wanted)}}
        )
    }
    for rid in sorted(wanted - found):
        print(f"skipped (not in W&B yet): {rid}")
    grids = []
    for g in spec.grids:
        present = tuple(r for r in g.run_ids if r in found)
        if not present:
            print(f"skipped grid (no run yet): {g.heading}")
            continue
        grids.append(g.model_copy(update={"run_ids": present}))
    return spec.model_copy(update={"grids": tuple(grids)})


def _line_plot(p: PanelSpec, i: int) -> wr.LinePlot:
    """Panel ``i`` of its grid, placed two per row (w=12 of the 24-unit grid, h=8)."""
    return wr.LinePlot(
        title=p.title,
        x=p.x,
        y=list(p.y),
        title_x=_x_name(p.x),
        smoothing_type="none",
        log_y=p.log_y,
        max_runs_to_show=MAX_RUNS_SHOWN,
        legend_template=LEGEND,
        layout=wr.Layout(x=12 * (i % 2), y=8 * (i // 2), w=12, h=8),
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
    spec = resolve(args.round, REPORTS[args.round])
    url = save_report(args.round, spec)
    if re.search(r"/reports/", url) is None:
        raise ValueError(f"not a report URL: {url}")
    verify(url, spec)
    print(f"\n{url}")


if __name__ == "__main__":
    main()
