# experiments/025-solid-growth/scripts/graph_reg_round2_plan.py
# [[experiments.025-solid-growth.scripts.graph_reg_round2_plan]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/025-solid-growth/scripts/graph_reg_round2_plan
"""The second round of the graph-regularization study: arms, budget, and the planned figures.

Nothing here is measured. The script is the single source of the round-2 design so the
document's table, the budget and the figure wireframe come from one place and the
launcher can be written against the same arm list once the round is approved.

Every arm is ``cgt_s0_r_kl_ctrl_013`` (S0 triples, pinned 010 random split, learnable
table, AdamW 2.5e-4 constant, batch 256 over four GPUs) with the named keys changed;
30 epochs unless the arm says otherwise. Costs use the measured Delta rate on the staged
NVMe path, 27 min per epoch on four A40s (delta_cgt.slurm, job 22030924), so one
30-epoch run is about 14 h of wall clock and 4 x 14 = 56 GPU-hours, sized on a 24 h
clock; a 60-epoch run needs the 48 h clock.

Writes
  results/graph_reg_round2_plan.json
  notes-tex/025-graph-reg-sweep/tables/t4-round2.tex
  $ASSET_IMAGES_DIR/025-solid-growth/graph_reg_round2_mockup.{svg,png}   (wireframe, no data)
"""

from __future__ import annotations

import json
import os
import os.path as osp
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from matplotlib.axes import Axes
from pydantic import BaseModel

from torchcell.timestamp import timestamp
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    panel_label,
    savefig_true_size_svg,
)

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "025-solid-growth", "results")
TABLES_DIR = osp.join(
    osp.dirname(EXPERIMENT_ROOT), "notes-tex", "025-graph-reg-sweep", "tables"
)
IMG_DIR = osp.join(ASSET_IMAGES_DIR, "025-solid-growth")

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 6,
        "axes.labelsize": 6,
        "axes.titlesize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "svg.fonttype": "none",
        "axes.linewidth": 0.5,
    }
)
MIN_PER_EPOCH = 27.0
GPUS = 4
ORANGE, RED, PURPLE, YELLOW, BLUE, GRAY = PLOT_PALETTE[:6]


class Arm(BaseModel):
    """One planned arm."""

    round: str  # "1b" finishes the current figure; "2" the mechanism figure; "3" budget and representation
    group: str  # the question it answers
    name: str
    change: str  # the override or config on ctrl_013
    needs_code: str  # "" if the model already supports it
    seeds: int
    epochs: int = 30
    figure: str  # which planned figure and panel reads it

    @property
    def wall_h(self) -> float:
        """Wall-clock hours of one run at the measured Delta rate."""
        return self.epochs * MIN_PER_EPOCH / 60.0

    @property
    def gpu_h(self) -> float:
        """GPU-hours over the arm's seeds: four cards per run."""
        return self.wall_h * GPUS * self.seeds


ARMS: list[Arm] = [
    # Round 1b: what the current figure still needs. The random arm at 1e-3 ran the same
    # 30-epoch protocol as every other arm (30 logged epochs on both complete seeds); its
    # third seed finishes on its own and needs no submission.
    Arm(
        round="1b",
        group="completion",
        name="KL 1e-5, seed 2",
        change="graph_reg_lambda=1e-5, seed=2",
        needs_code="",
        seeds=1,
        figure="F1 a-e",
    ),
    Arm(
        round="1b",
        group="biology vs conditioning",
        name="random graphs, KL 1",
        change="rand_031, graph_reg_lambda=1",
        needs_code="",
        seeds=3,
        figure="F1 a-c, F1 f",
    ),
    Arm(
        round="1b",
        group="biology vs conditioning",
        name="random graphs, KL 0.1",
        change="rand_031, graph_reg_lambda=0.1",
        needs_code="",
        seeds=3,
        figure="F1 a-c",
    ),
    Arm(
        round="1b",
        group="ladder top",
        name="KL 10",
        change="graph_reg_lambda=10",
        needs_code="",
        seeds=3,
        figure="F1 a-e",
    ),
    Arm(
        round="1b",
        group="ladder top",
        name="KL 100",
        change="graph_reg_lambda=100",
        needs_code="",
        seeds=3,
        figure="F1 a-e",
    ),
    # Round 2: how the graphs enter (the new mechanism figure)
    Arm(
        round="2",
        group="direction",
        name="KL 1, symmetric targets",
        change="graph_regularization symmetrize=true",
        needs_code="symmetrize A for the directed graphs before row normalization",
        seeds=3,
        figure="F2 a-c",
    ),
    Arm(
        round="2",
        group="direction",
        name="mask, directed",
        change="attention_mask symmetric=false",
        needs_code="drop the transpose write in the mask builder",
        seeds=3,
        figure="F2 a-c",
    ),
    Arm(
        round="2",
        group="reach",
        name="mask, two-hop",
        change="attention_mask hops=2",
        needs_code="support A or A squared, self-loops kept",
        seeds=3,
        figure="F2 a-c",
    ),
    Arm(
        round="2",
        group="placement",
        name="mask, layers 1-2",
        change="attention_mask layers=[1,2]",
        needs_code="",
        seeds=3,
        figure="F2 a-f",
    ),
    Arm(
        round="2",
        group="placement",
        name="mask, layers 3-4",
        change="attention_mask layers=[3,4]",
        needs_code="",
        seeds=3,
        figure="F2 a-f",
    ),
    Arm(
        round="2",
        group="placement",
        name="mask, layers 1-4",
        change="attention_mask layers=[1,2,3,4]",
        needs_code="",
        seeds=3,
        figure="F2 a-f",
    ),
    Arm(
        round="2",
        group="placement",
        name="KL 1, layers 1-2",
        change="graph_reg_lambda=1, graph_reg_layer=[1,2]",
        needs_code="",
        seeds=3,
        figure="F2 a-f",
    ),
    Arm(
        round="2",
        group="placement",
        name="KL 1, layers 3-4",
        change="graph_reg_lambda=1, graph_reg_layer=[3,4]",
        needs_code="",
        seeds=3,
        figure="F2 a-f",
    ),
    Arm(
        round="2",
        group="placement",
        name="KL 1, layers 1-4",
        change="graph_reg_lambda=1, graph_reg_layer=[1,2,3,4]",
        needs_code="",
        seeds=3,
        figure="F2 a-f",
    ),
    # Round 3: budget, and whether the prior still helps on the representation we intend to use
    Arm(
        round="3",
        group="budget",
        name="no penalty, 60 epochs",
        change="graph_reg_lambda=0, max_epochs=60",
        needs_code="",
        seeds=3,
        epochs=60,
        figure="F3 a-b",
    ),
    Arm(
        round="3",
        group="budget",
        name="mask, 60 epochs",
        change="mask_028, max_epochs=60",
        needs_code="",
        seeds=3,
        epochs=60,
        figure="F3 a-b",
    ),
    Arm(
        round="3",
        group="budget",
        name="KL 1, 60 epochs",
        change="graph_reg_lambda=1, max_epochs=60",
        needs_code="",
        seeds=3,
        epochs=60,
        figure="F3 a-b",
    ),
    Arm(
        round="3",
        group="representation",
        name="composite embedding, no penalty",
        change="embfit_035 minus fitness head, graph_reg_lambda=0",
        needs_code="",
        seeds=3,
        figure="F3 e",
    ),
    Arm(
        round="3",
        group="representation",
        name="composite embedding, KL 1",
        change="embfit_035 minus fitness head, graph_reg_lambda=1",
        needs_code="",
        seeds=3,
        figure="F3 e",
    ),
]


def _tt(change: str) -> str:
    """Each space- or comma-separated token in monospace, so the column can break between tokens."""
    parts = re.split(r"([ ,]+)", change)
    return "".join(
        p_ if re.fullmatch(r"[ ,]+", p_) else "\\texttt{" + p_.replace("_", "\\_") + "}"
        for p_ in parts
        if p_
    )


def write_table(arms: list[Arm]) -> None:
    os.makedirs(TABLES_DIR, exist_ok=True)
    lines = [
        "%% SOURCE: experiments/025-solid-growth/scripts/graph_reg_round2_plan.py -- GENERATED, do not edit",
        "\\begin{tabular}{clp{2.6cm}p{3.6cm}p{3.0cm}rrr}",
        "\\toprule",
        "round & question & arm & change on \\texttt{ctrl\\_013} & code & seeds & epochs & GPU-h \\\\",
        "\\midrule",
    ]
    last_round, last_group = "", ""
    for a in arms:
        if a.round != last_round and last_round:
            lines.append("\\midrule")
        rd = a.round if a.round != last_round else ""
        g = a.group if (a.group != last_group or a.round != last_round) else ""
        last_round, last_group = a.round, a.group
        code = a.needs_code if a.needs_code else "none"
        lines.append(
            f"{rd} & {g} & {a.name} & {_tt(a.change)} & {code} & {a.seeds} & {a.epochs} & {a.gpu_h:.0f} \\\\"
        )
    total_runs = sum(a.seeds for a in arms)
    total_gpu = sum(a.gpu_h for a in arms)
    lines += [
        "\\midrule",
        f"total & & & & {total_runs} & & {total_gpu:,.0f} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
    ]
    with open(osp.join(TABLES_DIR, "t4-round2.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def _panel(
    ax: Axes, title: str, xlabel: str, ylabel: str, body: str, color: str = GRAY
) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(
        0.5,
        0.5,
        body,
        ha="center",
        va="center",
        fontsize=5,
        color=color,
        transform=ax.transAxes,
        wrap=True,
    )
    for s in ax.spines.values():
        s.set_linewidth(0.5)


def wireframe() -> None:
    """The three planned figures as labeled empty panels."""
    fig = plt.figure(figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(150)))
    gs = fig.add_gridspec(
        3, 6, left=0.05, right=0.99, bottom=0.05, top=0.93, hspace=0.9, wspace=0.6
    )
    # F1: the ladder, extended
    f1 = [fig.add_subplot(gs[0, i]) for i in range(6)]
    _panel(
        f1[0],
        "F1a  Pearson, three readings",
        "log10 λ, none, mask, rand",
        "held-out Pearson",
        "ladder to λ = 100;\nrandom at λ = 1 (and 0.1)\nreplaces random at 1e-3;\nstars per reading",
    )
    _panel(
        f1[1],
        "F1b  held-out loss",
        "log10 λ",
        "point loss",
        "epoch 29 and\nat min val loss",
    )
    _panel(
        f1[2],
        "F1c  edge recall",
        "log10 λ",
        "recall at degree",
        "vs own target;\nrandom arms also\nvs the biological\ngraphs (from ckpt)",
    )
    _panel(
        f1[3],
        "F1d  divergence",
        "log10 λ",
        "divergence",
        "floor at ~300:\ndoes 10, 100\nbreak it?",
    )
    _panel(
        f1[4],
        "F1e  gradient ratio",
        "probe epoch",
        "penalty / point",
        "adds 10, 100\nand random 1",
    )
    _panel(
        f1[5],
        "F1f  random control",
        "target",
        "Pearson, ep 29",
        "biological vs random\nvs none, AT λ = 1\n(n = 3 each)",
        color=PURPLE,
    )
    # F2: how the graphs enter
    f2 = [fig.add_subplot(gs[1, i]) for i in range(6)]
    cols = "mechanism (12 columns)"
    _panel(
        f2[0],
        "F2a  Pearson, three readings",
        cols,
        "held-out Pearson",
        "one column per mechanism;\nstars vs none",
        color=ORANGE,
    )
    _panel(
        f2[1],
        "F2b  held-out loss",
        "mechanism",
        "point loss",
        "epoch 29 and\nat min val loss",
    )
    _panel(
        f2[2],
        "F2c  edge recall",
        "mechanism",
        "recall at degree",
        "mask arms 1 by\nconstruction on their\nown support; score all\nvs one-hop biological",
    )
    _panel(
        f2[3],
        "F2d  val Pearson by epoch",
        "epoch",
        "Pearson",
        "none, mask L1, L3-4, L1-4,\nKL1 L1, L3-4, L1-4\n(the curves move here\nfrom F1 g)",
        color=BLUE,
    )
    _panel(f2[4], "F2e  train Pearson by epoch", "epoch", "Pearson", "same arms")
    _panel(f2[5], "F2f  val loss by epoch", "epoch", "point loss", "same arms")
    # F3: budget and the checkpoint panels
    f3 = [fig.add_subplot(gs[2, i]) for i in range(6)]
    _panel(
        f3[0],
        "F3a  60 epochs, val Pearson",
        "epoch (0-59)",
        "Pearson",
        "none, mask, KL 1;\ndoes KL 1 keep rising,\ndoes none keep falling",
        color=BLUE,
    )
    _panel(f3[1], "F3b  60 epochs, val loss", "epoch (0-59)", "point loss", "same arms")
    _panel(
        f3[2],
        "F3c  discovery",
        "top-k off-graph attention",
        "recall of unseen edges",
        "from best checkpoints:\nregulatory head vs\nTFLink-only edges;\nnull: degree-matched",
        color=RED,
    )
    _panel(
        f3[3],
        "F3d  overruling",
        "α_ij / (1/d_i), binned",
        "share with |ε| > 0.08",
        "from best checkpoints:\nkept vs down-weighted\nprior edges against\nCostanzo digenic ε",
        color=RED,
    )
    _panel(
        f3[4],
        "F3e  divergence, all arms",
        "arm",
        "divergence to biological Ã",
        "none, mask, random,\nsymmetric: from ckpt,\nfills the crosses of F1 d",
    )
    _panel(
        f3[5],
        "F3f  seed table",
        "",
        "",
        "per-run table of every\nround-2 run, W&B ids,\nas t2 today",
    )
    for a_, letter in zip(f1 + f2 + f3, "abcdefghijklmnopqr"):
        panel_label(a_, letter)
    fig.text(
        0.5,
        0.975,
        "PLANNED, no data: three figures of round 2. Row 1 revises Figure 2 of this document; rows 2 and 3 are new.",
        ha="center",
        fontsize=6,
        color=RED,
    )
    os.makedirs(IMG_DIR, exist_ok=True)
    savefig_true_size_svg(fig, osp.join(IMG_DIR, "graph_reg_round2_mockup.svg"))
    fig.savefig(osp.join(IMG_DIR, "graph_reg_round2_mockup.png"), dpi=300)
    savefig_true_size_svg(
        fig, osp.join(IMG_DIR, f"graph_reg_round2_mockup_{timestamp()}.svg")
    )
    plt.close(fig)


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plan = {
        "protocol": "cgt_s0_r_kl_ctrl_013, Delta gpuA40x4, 4 x A40 per run",
        "min_per_epoch": MIN_PER_EPOCH,
        "arms": [
            a.model_dump() | {"wall_h": round(a.wall_h, 1), "gpu_h": round(a.gpu_h)}
            for a in ARMS
        ],
        "total_runs": sum(a.seeds for a in ARMS),
        "runs_per_round": {
            rd: sum(a.seeds for a in ARMS if a.round == rd) for rd in ("1b", "2", "3")
        },
        "total_gpu_h": round(sum(a.gpu_h for a in ARMS)),
        "code_changes": sorted({a.needs_code for a in ARMS if a.needs_code}),
    }
    with open(osp.join(RESULTS_DIR, "graph_reg_round2_plan.json"), "w") as fh:
        json.dump(plan, fh, indent=1)
    write_table(ARMS)
    wireframe()
    print(
        f"{plan['total_runs']} runs, {plan['total_gpu_h']:,} GPU-h; code changes: {plan['code_changes']}"
    )


if __name__ == "__main__":
    main()
