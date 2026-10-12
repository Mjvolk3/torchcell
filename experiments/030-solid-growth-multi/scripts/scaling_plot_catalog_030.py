# experiments/030-solid-growth-multi/scripts/scaling_plot_catalog_030.py
# [[experiments.030-solid-growth-multi.scripts.scaling_plot_catalog_030]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/030-solid-growth-multi/scripts/scaling_plot_catalog_030
"""The catalog of plots the 030 scaling grid can feed, as one typed spec.

Every plot of ``notes-tex/030-scaling-plots/`` is a ``PlotSpec`` here: its axes, the
series on it, the read it uses, the W&B keys and the runs it draws on, and the one
sentence each outcome would say. The spec writes the document's catalog table and the
json that ``notes/assets/drawio/030-scaling-plots.gen.py`` reads for its panel titles and
axis labels, so the table and the sketches name the same plots the same way.

The runs are the queued grid (``delta_submit_scaling_030.sh`` and the 025 pair):
5 sizes x 4 subsets at seed 1, seeds 2-3 at 1/64 and 1/16 for 45x2 and 180x8, a
learning-rate check at 45x2 on 1/4, two calibration jobs, and the 025 twin at 180x8 on
the four subsets with seeds 2-3 at the two smallest. The keys are those
``scaling_readout_030.py`` collects. Nothing here is a result; ``feeds`` says which rows
of the grid a plot needs and ``status`` says whether the queued grid provides them in
full, in part, or not at all.

    PYTHONPATH=$PWD python experiments/030-solid-growth-multi/scripts/scaling_plot_catalog_030.py

Writes ``notes-tex/030-scaling-plots/tables/t1-catalog.tex`` and
``experiments/030-solid-growth-multi/results/scaling_plot_catalog_030.json``.
"""

from __future__ import annotations

import json
import os
import os.path as osp
import sys
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from arm_030 import results_dir  # noqa: E402

SCRIPT = "experiments/030-solid-growth-multi/scripts/scaling_plot_catalog_030.py"


class PlotSpec(BaseModel):
    """One plot the grid can feed."""

    id: str
    family: Literal["surface", "compute", "diagnostic", "pair"]
    title: str
    x: str
    y: str
    series: str
    read: Literal["per-epoch", "selected", "matched", "both", "calibration"]
    keys: list[str]
    feeds: str
    status: Literal["full", "partial", "missing"]
    shows: str


CATALOG: list[PlotSpec] = [
    PlotSpec(
        id="S1",
        family="surface",
        title="Loss against D at every size",
        x="D, training records (log); 030 also in entry rows",
        y="val MSE, label units (log)",
        series="one curve per size, 5; per screen and pooled",
        read="both",
        keys=[
            "val/gene_interaction/MSE",
            "val/token/[screen]/gene_interaction/MSE",
            "compute/records_per_epoch",
        ],
        feeds="the 5 x 4 seed-1 grid; seeds 2-3 at the two smallest subsets for 45x2 and 180x8",
        status="full",
        shows="whether curves of different sizes stay parallel (one beta) or fan out (the data exponent depends on size); the 180x8 curve is the fitted one",
    ),
    PlotSpec(
        id="S2",
        family="surface",
        title="Loss against N at every subset",
        x="N, non-embedding parameters (log)",
        y="val MSE, label units (log)",
        series="one curve per subset, 4; per screen and pooled",
        read="both",
        keys=[
            "val/gene_interaction/MSE",
            "val/token/[screen]/gene_interaction/MSE",
            "compute/params_non_embedding",
        ],
        feeds="the 5 x 4 seed-1 grid",
        status="partial",
        shows="the plateau each subset sets, E + B D^-beta, and the size at which a curve meets it; at ten epochs on the full pool the top curve is a fixed-epoch slice",
    ),
    PlotSpec(
        id="S3",
        family="surface",
        title="The joint surface as a 5 x 4 heatmap",
        x="D, four subsets",
        y="N, five sizes",
        series="cell color = matched-read val MSE; a second heatmap for the selected read",
        read="both",
        keys=["val/gene_interaction/MSE", "compute/records_seen"],
        feeds="the 5 x 4 seed-1 grid",
        status="full",
        shows="the gradient of the loss in each direction at every cell: the direction of steeper descent is where the next unit of effort goes from that cell",
    ),
    PlotSpec(
        id="S4",
        family="surface",
        title="Fitted forms on the two axes",
        x="D or N (log)",
        y="val MSE minus the fitted E (log)",
        series="pure and floored fits with bootstrap bands; the held-out point ringed",
        read="both",
        keys=["[sweep]_fits.json"],
        feeds="S1's 180x8 column and S2's full-pool column",
        status="partial",
        shows="whether L - E is straight (the floored form holds) and whether the held-out point lands inside the band; on four D points the floored held-out test is undefined",
    ),
    PlotSpec(
        id="C1",
        family="compute",
        title="Learning curves against records seen",
        x="records seen (log)",
        y="val MSE, label units (log)",
        series="one line per point, colored by size, one panel per subset; B marked",
        read="per-epoch",
        keys=["val/gene_interaction/MSE", "compute/records_seen", "epoch"],
        feeds="every run; chains stitched by resumed_from",
        status="full",
        shows="where each point's selected epoch sits, where B cuts each curve, and which curves are still falling at their last epoch (unconverged)",
    ),
    PlotSpec(
        id="C2",
        family="compute",
        title="Loss against training FLOPs, the envelope",
        x="cumulative FLOPs (log)",
        y="val MSE, label units (log)",
        series="one trajectory per point, colored by size; the lower envelope drawn",
        read="per-epoch",
        keys=["val/gene_interaction/MSE", "compute/flops_cumulative"],
        feeds="every run on the full pool and on 1/4 (the large-data columns)",
        status="partial",
        shows="the compute-efficient frontier across sizes: which size reaches a given loss for the least compute; one budget line of an IsoFLOP sweep, not the sweep",
    ),
    PlotSpec(
        id="C3",
        family="compute",
        title="Loss against GPU hours",
        x="GPU hours (log)",
        y="val MSE, label units (log)",
        series="as C2, wall clock instead of FLOPs",
        read="per-epoch",
        keys=["val/gene_interaction/MSE", "compute/gpu_hours_cumulative"],
        feeds="every run",
        status="full",
        shows="the same envelope in the unit the queue charges; the ratio of C2 to C3 per size is the utilization each size achieved",
    ),
    PlotSpec(
        id="C4",
        family="compute",
        title="Cost per record and pace against size",
        x="N (log)",
        y="FLOPs per record (left); records per second (right)",
        series="five sizes, min-to-max bar on the FLOPs from the five measured steps",
        read="calibration",
        keys=[
            "compute/flops_per_record",
            "compute/flops_per_step_min",
            "compute/flops_per_step_max",
            "compute/records_seen",
            "compute/gpu_hours_cumulative",
        ],
        feeds="the two calibration jobs (180x8, 45x2) now; every grid run's first epoch after",
        status="partial",
        shows="whether pace follows the FLOPs (GPU-bound) or stays flat across sizes (loader-bound), which decides whether the small sizes could have trained longer",
    ),
    PlotSpec(
        id="D1",
        family="diagnostic",
        title="Bootstrap distribution of each exponent",
        x="exponent value",
        y="count of resamples",
        series="one histogram per axis, read and target; the 90% interval marked",
        read="both",
        keys=["[sweep]_fits.json"],
        feeds="300 resamples over runs of S1's and S2's fitted columns",
        status="full",
        shows="how wide the interval is; with four D points and mostly one seed the D interval will be wide, and a histogram says so more honestly than a point",
    ),
    PlotSpec(
        id="D2",
        family="diagnostic",
        title="Held-out prediction against observation",
        x="predicted loss at the held-out point",
        y="observed loss",
        series="one marker per fit (axis, read, target, form); the diagonal",
        read="both",
        keys=["[sweep]_fits.json"],
        feeds="every fit of S4",
        status="partial",
        shows="which fits extrapolate one step; markers off the diagonal are exponents that describe their points and nothing beyond them",
    ),
    PlotSpec(
        id="D3",
        family="diagnostic",
        title="Fitted E against the replicate floor",
        x="screen (2018, 2020, pooled)",
        y="loss, label units",
        series="fitted E with its interval; the floor bounds as a band; the mean-prediction loss as the top",
        read="both",
        keys=["[sweep]_fits.json", "triple_noise_ceiling_030.json"],
        feeds="S4's floored fits and the noise-ceiling file",
        status="full",
        shows="whether E sits below the floor (wrong fit), near it (noise-limited) or far above (model-limited), screen by screen",
    ),
    PlotSpec(
        id="D4",
        family="diagnostic",
        title="Seed spread against effect size",
        x="(size, subset) cell",
        y="val MSE, label units",
        series="three seeds per cell at 1/64 and 1/16 for 45x2 and 180x8; the step between adjacent cells beside it",
        read="selected",
        keys=["val/gene_interaction/MSE"],
        feeds="the eight seed-replicate cells plus their seed-1 neighbors",
        status="full",
        shows="whether a step along D or N is larger than the seed spread, the noise floor every exponent interval rests on",
    ),
    PlotSpec(
        id="D5",
        family="diagnostic",
        title="The learning-rate check",
        x="epoch",
        y="val MSE, label units (log)",
        series="three curves at 45x2 on 1/4: lr 1e-4, 2.5e-4, 1e-3",
        read="per-epoch",
        keys=["val/gene_interaction/MSE", "regression_task.optimizer.lr"],
        feeds="the two lr jobs and the seed-1 cell at 45x2 on 1/4",
        status="partial",
        shows="whether 2.5e-4 is near the optimum at the smallest size; it says nothing about 180x8, which runs at a batch four times smaller than it was tuned at",
    ),
    PlotSpec(
        id="D6",
        family="diagnostic",
        title="Per-screen Pearson beside the loss",
        x="D (log)",
        y="val Pearson, per screen and pooled",
        series="pooled with its 0.35 screen-offset floor marked; 2018 and 2020 per-token",
        read="selected",
        keys=[
            "val/gene_interaction/Pearson",
            "val/token/[screen]/gene_interaction/Pearson",
        ],
        feeds="the 180x8 column",
        status="full",
        shows="the headline metric readers expect, with the floor the two screen means put under the pooled number, so the per-screen loss curves and the Pearson tell one story",
    ),
    PlotSpec(
        id="P1",
        family="pair",
        title="The 030 minus 025 gap against D",
        x="D, training records (log)",
        y="030 minus 025 pooled val MSE, label units",
        series="seed mean and sd at 1/64 and 1/16; single points at 1/4; the full point hollow",
        read="both",
        keys=["val/gene_interaction/MSE"],
        feeds="the 180x8 column of both sweeps",
        status="partial",
        shows="what keeping every measurement and the token buy at each D; interpretable at the two converged subsets, a learning-speed gap at the full pool",
    ),
    PlotSpec(
        id="P2",
        family="pair",
        title="Paired learning curves",
        x="records seen (log)",
        y="val MSE, label units (log)",
        series="030 and 025 at 180x8, one panel per subset; B marked",
        read="per-epoch",
        keys=["val/gene_interaction/MSE", "compute/records_seen"],
        feeds="the 180x8 column of both sweeps",
        status="full",
        shows="whether the gap opens early and holds, or closes as 025 catches up, which is the difference between more supervision per genotype and a faster start",
    ),
]


def _esc(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("#", "\\#")
        .replace("^", "\\^{}")
        .replace("[", "{[}")
        .replace("]", "{]}")
    )


def write_table(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    counts = {
        s: sum(1 for p in CATALOG if p.status == s)
        for s in ("full", "partial", "missing")
    }
    lines = [
        f"%% GENERATED FILE -- do not hand-edit.\n%% SOURCE: {SCRIPT} (the PlotSpec catalog)",
        "\\begin{table}[htbp]",
        "\\centering\\footnotesize",
        f"\\caption{{The plot catalog: {len(CATALOG)} plots the queued grid can feed, "
        f"{counts['full']} in full and {counts['partial']} in part. Read names what the "
        "y value is at each point: per-epoch rows, the validation-selected epoch, the "
        "interpolation at the shared records-seen budget $B$, or the calibration jobs. "
        "Feeds names the rows of the grid the plot needs. Partial means the rows exist "
        "but a stated limit narrows the claim (ten-epoch full-pool points, one seed, "
        "four points on an axis).}",
        "\\label{tab:catalog}",
        "\\begin{tabular}{llp{0.30\\linewidth}p{0.16\\linewidth}p{0.26\\linewidth}l}",
        "\\toprule",
        "id & family & plot (x; y) & read & feeds & status \\\\",
        "\\midrule",
    ]
    for p in CATALOG:
        lines.append(
            f"{p.id} & {p.family} & \\textbf{{{_esc(p.title)}}}: {_esc(p.x)}; {_esc(p.y)} & "
            f"{p.read} & {_esc(p.feeds)} & {p.status} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    with open(osp.join(out_dir, "t1-catalog.tex"), "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    load_dotenv()
    repo = osp.dirname(os.environ["EXPERIMENT_ROOT"].rstrip("/"))
    write_table(osp.join(repo, "notes-tex/030-scaling-plots/tables"))
    with open(osp.join(results_dir(), "scaling_plot_catalog_030.json"), "w") as f:
        json.dump([p.model_dump() for p in CATALOG], f, indent=2)
    for p in CATALOG:
        print(f"  {p.id:<3} {p.status:<8} {p.title}")
    print(
        f"  {len(CATALOG)} plots; wrote tables/t1-catalog.tex and results/scaling_plot_catalog_030.json"
    )


if __name__ == "__main__":
    main()
