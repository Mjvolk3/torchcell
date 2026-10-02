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
    config: str = "cgt_s0_r_kl_ctrl_013"  # Hydra config name the launcher loads
    overrides: list[str] = []  # Hydra overrides; new keys carry a leading +
    hours: int = 24  # Delta wall clock

    @property
    def slug(self) -> str:
        """Job-name stem: lowercase, words joined by hyphens, punctuation dropped."""
        return re.sub(r"[^a-z0-9]+", "-", self.name.lower()).strip("-")

    @property
    def wall_h(self) -> float:
        """Wall-clock hours of one run at the measured Delta rate."""
        return self.epochs * MIN_PER_EPOCH / 60.0

    @property
    def gpu_h(self) -> float:
        """GPU-hours over the arm's seeds: four cards per run."""
        return self.wall_h * GPUS * self.seeds


ARMS: list[Arm] = [
    # Round 1b finishes the current figure: prior vs mask at the ladder top, do the graphs
    # help at the lambda that separates (five seeds: each seed is a new rewiring and a new
    # initialization). Round 2: reach (2- and 3-hop, identical supports for mask and prior)
    # and direction. The reach masks are DIRECTED: undirected two hops of physical,
    # coexpression or experimental already covers 60 to 70 percent of all gene pairs
    # (graph_reg_khop_density.py), so a symmetric k-hop mask is no mask; the directed
    # one-hop mask is their one-hop anchor. Round 3: placement (optional), budget,
    # representation, width.
    Arm(
        round="1b",
        group="do graphs help",
        name="random graphs, KL 1",
        change="rand_031, graph_reg_lambda=1",
        needs_code="",
        seeds=5,
        epochs=30,
        figure="F1 a-c, f",
        config="cgt_s0_r_kl_rand_031",
        overrides=["model.graph_regularization.graph_reg_lambda=1"],
        hours=24,
    ),
    Arm(
        round="1b",
        group="do graphs help",
        name="random graphs, KL 0.1",
        change="rand_031, graph_reg_lambda=0.1",
        needs_code="",
        seeds=5,
        epochs=30,
        figure="F1 a-c, f",
        config="cgt_s0_r_kl_rand_031",
        overrides=["model.graph_regularization.graph_reg_lambda=0.1"],
        hours=24,
    ),
    Arm(
        round="1b",
        group="prior vs mask",
        name="KL 10",
        change="graph_reg_lambda=10",
        needs_code="",
        seeds=3,
        epochs=30,
        figure="F1 a-e",
        config="cgt_s0_r_kl_ctrl_013",
        overrides=["model.graph_regularization.graph_reg_lambda=10"],
        hours=24,
    ),
    Arm(
        round="1b",
        group="prior vs mask",
        name="KL 100",
        change="graph_reg_lambda=100",
        needs_code="",
        seeds=3,
        epochs=30,
        figure="F1 a-e",
        config="cgt_s0_r_kl_ctrl_013",
        overrides=["model.graph_regularization.graph_reg_lambda=100"],
        hours=24,
    ),
    Arm(
        round="1b",
        group="completion",
        name="KL 1e-5, seed 2",
        change="graph_reg_lambda=1e-5, seed=2",
        needs_code="",
        seeds=1,
        epochs=30,
        figure="F1 a-e",
        config="cgt_s0_r_kl_ctrl_013",
        overrides=["model.graph_regularization.graph_reg_lambda=1e-5"],
        hours=24,
    ),
    Arm(
        round="2",
        group="reach",
        name="mask, 2-hop, directed",
        change="attention_mask hops=2, symmetric=false",
        needs_code="k-hop support: reachability within k steps, self-loops kept",
        seeds=3,
        epochs=30,
        figure="F2 a-f",
        config="cgt_s0_r_mask_028",
        overrides=[
            "+model.attention_mask.hops=2",
            "+model.attention_mask.symmetric=false",
        ],
        hours=24,
    ),
    Arm(
        round="2",
        group="reach",
        name="mask, 3-hop, directed",
        change="attention_mask hops=3, symmetric=false",
        needs_code="same",
        seeds=3,
        epochs=30,
        figure="F2 a-f",
        config="cgt_s0_r_mask_028",
        overrides=[
            "+model.attention_mask.hops=3",
            "+model.attention_mask.symmetric=false",
        ],
        hours=24,
    ),
    Arm(
        round="2",
        group="reach",
        name="KL 1, 2-hop target",
        change="graph_regularization hops=2",
        needs_code="k-hop target: row-normalized k-hop indicator",
        seeds=3,
        epochs=30,
        figure="F2 a-f",
        config="cgt_s0_r_kl_ctrl_013",
        overrides=[
            "model.graph_regularization.graph_reg_lambda=1",
            "+model.graph_regularization.hops=2",
        ],
        hours=24,
    ),
    Arm(
        round="2",
        group="reach",
        name="KL 1, 3-hop target",
        change="graph_regularization hops=3",
        needs_code="same",
        seeds=3,
        epochs=30,
        figure="F2 a-f",
        config="cgt_s0_r_kl_ctrl_013",
        overrides=[
            "model.graph_regularization.graph_reg_lambda=1",
            "+model.graph_regularization.hops=3",
        ],
        hours=24,
    ),
    Arm(
        round="2",
        group="direction",
        name="KL 1, symmetric targets",
        change="graph_regularization symmetrize=true",
        needs_code="symmetrize A for the directed graphs before row normalization",
        seeds=3,
        epochs=30,
        figure="F2 a-c",
        config="cgt_s0_r_kl_ctrl_013",
        overrides=[
            "model.graph_regularization.graph_reg_lambda=1",
            "+model.graph_regularization.symmetrize=true",
        ],
        hours=24,
    ),
    Arm(
        round="2",
        group="direction",
        name="mask, directed",
        change="attention_mask symmetric=false",
        needs_code="drop the transpose write in the mask builder",
        seeds=3,
        epochs=30,
        figure="F2 a-c",
        config="cgt_s0_r_mask_028",
        overrides=["+model.attention_mask.symmetric=false"],
        hours=24,
    ),
    Arm(
        round="3",
        group="placement (optional)",
        name="mask, layers 1-2",
        change="attention_mask layers=[1,2]",
        needs_code="",
        seeds=3,
        epochs=30,
        figure="F3 a-c",
        config="cgt_s0_r_mask_028",
        overrides=["model.attention_mask.layers=[1,2]"],
        hours=24,
    ),
    Arm(
        round="3",
        group="placement (optional)",
        name="mask, layers 3-4",
        change="attention_mask layers=[3,4]",
        needs_code="",
        seeds=3,
        epochs=30,
        figure="F3 a-c",
        config="cgt_s0_r_mask_028",
        overrides=["model.attention_mask.layers=[3,4]"],
        hours=24,
    ),
    Arm(
        round="3",
        group="placement (optional)",
        name="mask, layers 1-4",
        change="attention_mask layers=[1,2,3,4]",
        needs_code="",
        seeds=3,
        epochs=30,
        figure="F3 a-c",
        config="cgt_s0_r_mask_028",
        overrides=["model.attention_mask.layers=[1,2,3,4]"],
        hours=24,
    ),
    Arm(
        round="3",
        group="placement (optional)",
        name="KL 1, layers 1-2",
        change="graph_reg_lambda=1, graph_reg_layer=[1,2]",
        needs_code="",
        seeds=3,
        epochs=30,
        figure="F3 a-c",
        config="cgt_s0_r_kl_ctrl_013",
        overrides=[
            "model.graph_regularization.graph_reg_lambda=1",
            "model.graph_regularization.graph_reg_layer=[1,2]",
        ],
        hours=24,
    ),
    Arm(
        round="3",
        group="placement (optional)",
        name="KL 1, layers 3-4",
        change="graph_reg_lambda=1, graph_reg_layer=[3,4]",
        needs_code="",
        seeds=3,
        epochs=30,
        figure="F3 a-c",
        config="cgt_s0_r_kl_ctrl_013",
        overrides=[
            "model.graph_regularization.graph_reg_lambda=1",
            "model.graph_regularization.graph_reg_layer=[3,4]",
        ],
        hours=24,
    ),
    Arm(
        round="3",
        group="placement (optional)",
        name="KL 1, layers 1-4",
        change="graph_reg_lambda=1, graph_reg_layer=[1,2,3,4]",
        needs_code="",
        seeds=3,
        epochs=30,
        figure="F3 a-c",
        config="cgt_s0_r_kl_ctrl_013",
        overrides=[
            "model.graph_regularization.graph_reg_lambda=1",
            "model.graph_regularization.graph_reg_layer=[1,2,3,4]",
        ],
        hours=24,
    ),
    Arm(
        round="3",
        group="budget",
        name="no penalty, 60 epochs",
        change="graph_reg_lambda=0, max_epochs=60",
        needs_code="",
        seeds=3,
        epochs=60,
        figure="F3 d-e",
        config="cgt_s0_r_kl_ctrl_013",
        overrides=[
            "model.graph_regularization.graph_reg_lambda=0",
            "trainer.max_epochs=60",
        ],
        hours=48,
    ),
    Arm(
        round="3",
        group="budget",
        name="mask, 60 epochs",
        change="mask_028, max_epochs=60",
        needs_code="",
        seeds=3,
        epochs=60,
        figure="F3 d-e",
        config="cgt_s0_r_mask_028",
        overrides=["trainer.max_epochs=60"],
        hours=48,
    ),
    Arm(
        round="3",
        group="budget",
        name="KL 1, 60 epochs",
        change="graph_reg_lambda=1, max_epochs=60",
        needs_code="",
        seeds=3,
        epochs=60,
        figure="F3 d-e",
        config="cgt_s0_r_kl_ctrl_013",
        overrides=[
            "model.graph_regularization.graph_reg_lambda=1",
            "trainer.max_epochs=60",
        ],
        hours=48,
    ),
    Arm(
        round="3",
        group="representation",
        name="composite embedding, no penalty",
        change="emb_040, graph_reg_lambda=0",
        needs_code="",
        seeds=3,
        epochs=30,
        figure="F3 f",
        config="cgt_s0_r_kl_emb_040",
        overrides=["model.graph_regularization.graph_reg_lambda=0"],
        hours=24,
    ),
    Arm(
        round="3",
        group="representation",
        name="composite embedding, KL 1",
        change="emb_040, graph_reg_lambda=1",
        needs_code="",
        seeds=3,
        epochs=30,
        figure="F3 f",
        config="cgt_s0_r_kl_emb_040",
        overrides=["model.graph_regularization.graph_reg_lambda=1"],
        hours=24,
    ),
    Arm(
        round="3",
        group="width",
        name="hidden 360, table, no penalty",
        change="hidden_channels=360, graph_reg_lambda=0",
        needs_code="",
        seeds=3,
        epochs=30,
        figure="F3 f",
        config="cgt_s0_r_kl_ctrl_013",
        overrides=[
            "model.hidden_channels=360",
            "model.graph_regularization.graph_reg_lambda=0",
        ],
        hours=48,
    ),
    Arm(
        round="3",
        group="width",
        name="hidden 360, table, KL 1",
        change="hidden_channels=360, graph_reg_lambda=1",
        needs_code="",
        seeds=3,
        epochs=30,
        figure="F3 f",
        config="cgt_s0_r_kl_ctrl_013",
        overrides=[
            "model.hidden_channels=360",
            "model.graph_regularization.graph_reg_lambda=1",
        ],
        hours=48,
    ),
    Arm(
        round="3",
        group="width",
        name="hidden 360, composite, no penalty",
        change="emb_w360_041, graph_reg_lambda=0",
        needs_code="preprocessor re-matched (h = 644)",
        seeds=3,
        epochs=30,
        figure="F3 f",
        config="cgt_s0_r_kl_emb_w360_041",
        overrides=["model.graph_regularization.graph_reg_lambda=0"],
        hours=48,
    ),
    Arm(
        round="3",
        group="width",
        name="hidden 360, composite, KL 1",
        change="emb_w360_041, graph_reg_lambda=1",
        needs_code="preprocessor re-matched (h = 644)",
        seeds=3,
        epochs=30,
        figure="F3 f",
        config="cgt_s0_r_kl_emb_w360_041",
        overrides=["model.graph_regularization.graph_reg_lambda=1"],
        hours=48,
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
    cols = "mechanism, reach and direction (9 columns)"
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
        "none, mask 1/2/3-hop,\nKL1 1/2/3-hop\n(the curves move here\nfrom F1 g)",
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
    _panel(
        f3[1],
        "F3b  placement curves",
        "epoch",
        "Pearson",
        "val Pearson by epoch\nfor the placement arms",
    )
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


SUBMITTER = osp.join(
    EXPERIMENT_ROOT, "025-solid-growth", "scripts", "delta_submit_round2.sh"
)
# Minutes between consecutive runs' earliest start times. The stagger spreads the 554 GB
# node-local stage-ins over a day without serializing the campaign: every job is eligible
# once its begin time passes and the queue schedules them in parallel. The first
# submission (2026-09-27) used six `after:<prev>+30` dependency lanes instead, which made
# each job eligible only 30 minutes after its predecessor STARTED, so a lane advanced one
# job per queue wait and 13 of 80 jobs finished in 4.5 days; the lanes were cleared with
# `scontrol update Dependency= StartTime=now+<k*20>minutes` on 2026-10-01. Dependencies
# are for one thing only: a run that outlives the wall limit and resumes from its
# checkpoint (afterok, with the W&B run resumed, not a new one).
STAGGER_MINUTES = 20


def write_submitter(arms: list[Arm]) -> None:
    """The Delta submitter, one sbatch per run, eligibility staggered, no dependencies.

    Runs are laid out in table order (round 1b first), and run k gets
    `--begin=now+<k * STAGGER_MINUTES>minutes`, so round 1b is eligible first and the
    stage-ins spread out while the queue still runs everything in parallel. The seed-2
    completion run has one seed; the others count from 1. ROUNDS filters (default all),
    DRY=1 prints only.
    """
    runs: list[tuple[Arm, int]] = [
        (a, sd) for a in arms for sd in ([2] if a.seeds == 1 else range(1, a.seeds + 1))
    ]
    lines = [
        "#!/bin/bash",
        "# experiments/025-solid-growth/scripts/delta_submit_round2.sh",
        "# [[experiments.025-solid-growth.scripts.graph_reg_round2_plan]]",
        "#",
        "# GENERATED by experiments/025-solid-growth/scripts/graph_reg_round2_plan.py -- do not edit.",
        "# Submits the next rounds of the graph-regularization study on Delta, one sbatch per",
        f"# run, NO dependencies, earliest start staggered {STAGGER_MINUTES} min apart in table order",
        "# (round 1b first) so the node-local stage-ins spread out while the queue runs the",
        "# jobs in parallel. RUN ON A DELTA LOGIN NODE FROM THE WORKTREE ROOT, after",
        "# delta_preflight_025.sh, from a worktree checked out at the commit that carries the",
        "# k-hop, symmetrize and directed-mask flags (never advance a worktree under a running job).",
        "# Read .claude/skills/submit-jobs/SKILL.md first.",
        "#",
        "#   DRY=1 bash experiments/025-solid-growth/scripts/delta_submit_round2.sh        # print only",
        "#   ROUNDS=1b bash experiments/025-solid-growth/scripts/delta_submit_round2.sh    # one round",
        "#         bash experiments/025-solid-growth/scripts/delta_submit_round2.sh        # everything",
        "set -euo pipefail",
        'ACCOUNT="${ACCOUNT:-bfjt-delta-gpu}"',
        'LAUNCHER="experiments/025-solid-growth/scripts/delta_cgt.slurm"',
        'ROUNDS="${ROUNDS:-1b 2 3}"',
        'DRY="${DRY:-0}"',
        f'STAGGER="${{STAGGER:-{STAGGER_MINUTES}}}"',
        '[[ -f "$LAUNCHER" ]] || { echo "run from the worktree root: $LAUNCHER not found" >&2; exit 2; }',
        "K=0",
        "submit() {  # submit <round> <job-name> <hours> <config> [overrides...]",
        '  local round="$1" name="$2" hours="$3" cfg="$4"; shift 4',
        '  [[ " $ROUNDS " == *" $round "* ]] || return 0',
        '  local cmd=(sbatch --parsable --account="$ACCOUNT" --time="${hours}:00:00" -J "$name"',
        '             --begin="now+$((K * STAGGER))minutes" "$LAUNCHER" "$cfg" "$@")',
        "  K=$((K + 1))",
        '  echo "${cmd[*]}"',
        '  if [[ "$DRY" != "1" ]]; then echo "  -> $("${cmd[@]}")"; fi',
        "}",
        "",
    ]
    for a, sd in runs:
        name = f"025-r{a.round}-{a.slug}-s{sd}"
        ov = " ".join(a.overrides + [f"+seed={sd}"])
        lines.append(f'submit {a.round} "{name}" {a.hours} {a.config} {ov}')
    lines += ["", 'echo "submitted $K jobs, last eligible in $(( (K - 1) * STAGGER )) min"']
    with open(SUBMITTER, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    os.chmod(SUBMITTER, 0o755)


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
        "submitter": SUBMITTER,
    }
    with open(osp.join(RESULTS_DIR, "graph_reg_round2_plan.json"), "w") as fh:
        json.dump(plan, fh, indent=1)
    write_table(ARMS)
    write_submitter(ARMS)
    wireframe()
    print(
        f"{plan['total_runs']} runs, {plan['total_gpu_h']:,} GPU-h; code changes: {plan['code_changes']}"
    )


if __name__ == "__main__":
    main()
