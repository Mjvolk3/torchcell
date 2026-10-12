# experiments/040-inhibitor-synergy-wetlab/scripts/r3_plots.py
# [[experiments.040-inhibitor-synergy-wetlab.scripts.r3_plots]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/040-inhibitor-synergy-wetlab/scripts/r3_plots
"""Every scored mixture arm on one sheet: the public score and the growth call.

Reads ``results/r3_readout_arms.csv`` and the per-run score files behind it
(``r3_readout.py``), and draws two panels, one per readout, with one row per arm:
every run as a point (a run is one seed on one fold on one machine) and the arm's
mean as a bar. Arms are sorted by their mean on the public score, so the sheet reads as
a ranking with its noise visible. The two machines' copies of a run are drawn in the
same color; the reference lines are ridge on the public side and Loewe on the growth
call. Rounds 1 and 2 (pool-and-last, 50 epochs) are included and labeled, so the
protocol change is visible on the same axis.

    python experiments/040-inhibitor-synergy-wetlab/scripts/r3_plots.py
"""

from __future__ import annotations

import glob
import os
import os.path as osp
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

from torchcell.timestamp import timestamp  # noqa: E402
from torchcell.utils import (  # noqa: E402
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    apply_paper_style,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()
apply_paper_style()
EXPERIMENT = osp.dirname(osp.dirname(osp.abspath(__file__)))
RESULTS = osp.join(EXPERIMENT, "results")
SLUG = "040-inhibitor-synergy-wetlab"
OUT = osp.join(os.environ["ASSET_IMAGES_DIR"], SLUG)
MACHINES = {"gilahyper": "mixture", "delta": "mixture-delta"}
CGT = PLOT_PALETTE[10]
RIDGE = PLOT_PALETTE[4]
LOEWE = PLOT_PALETTE[2]
GRAY = PLOT_PALETTE[5]
#: the two reference values the gate sheets carry (ladder ridge; Loewe from the fits)
REFERENCES = osp.join(RESULTS, "figure_gate_numbers.json")


def arm_of(name: str) -> str:
    return re.sub(r"_s\d+$", "", re.sub(r"_f\d+$", "", name))


def runs() -> pd.DataFrame:
    frames = []
    for machine, sub in MACHINES.items():
        for path in sorted(glob.glob(osp.join(RESULTS, sub, "*", "*_scores.csv"))):
            sweep = osp.basename(osp.dirname(path))
            if "smoke" in sweep:
                continue
            frames.append(pd.read_csv(path).assign(sweep=sweep, machine=machine))
    d = pd.concat(frames, ignore_index=True)
    d = d[d.member.str.startswith("seed")].copy()
    d["arm"] = d["name"].map(arm_of)
    d["round"] = d["sweep"].str.extract(r"^(r\d+(?:w\d+)?)_")[0]
    return d


def panel(
    ax: plt.Axes,
    d: pd.DataFrame,
    order: list[str],
    xlabel: str,
    ref: float,
    ref_label: str,
    ref_color: str,
) -> None:
    y = np.arange(len(order))[::-1]
    rng = np.random.default_rng(0)
    for yi, arm in zip(y, order):
        v = d[d.arm == arm].value.to_numpy()
        ax.barh(
            yi,
            v.mean(),
            height=0.7,
            color=CGT,
            edgecolor="black",
            linewidth=0.4,
            zorder=2,
        )
        ax.scatter(
            v,
            yi + rng.uniform(-0.22, 0.22, len(v)),
            s=4,
            color="black",
            alpha=0.55,
            linewidths=0,
            zorder=3,
        )
    ax.axvline(ref, color=ref_color, linewidth=0.9, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{a.replace('mix_', '')} ({int((d.arm == a).sum())})" for a in order],
        fontsize=4.8,
    )
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel(xlabel)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.grid(axis="x", which="both", linewidth=0.3, color="0.85", zorder=0)
    ax.tick_params(axis="x", which="minor", length=0)


def main() -> None:
    d = runs()
    import json

    ref = json.load(open(REFERENCES))
    gene = d[
        (d.task == "vanacloig")
        & (d.subset == "test_compounds")
        & (d.metric == "centered_spearman_median")
    ]
    growth = d[
        (d.task == "ex23")
        & (d.subset == "combinations")
        & (d.metric == "growth_auroc")
        & (d.call == "served")
    ]
    order = gene.groupby("arm").value.mean().sort_values(ascending=False).index.tolist()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(
            mm_to_in(PANEL_WIDTHS_MM["full"]),
            mm_to_in(max(120.0, 3.4 * len(order) + 20)),
        ),
        layout="constrained",
    )
    panel(
        axes[0],
        gene,
        order,
        "compound-cold centered Spearman (Vanacloig)",
        float(ref["ladderRidge"]),
        "nested ridge, FCFP4 counts",
        RIDGE,
    )
    panel(
        axes[1],
        growth,
        order,
        "growth AUROC over the 63 combinations (served call)",
        float(ref["loeweAuroc"]),
        "Loewe from the titration fits",
        LOEWE,
    )
    axes[1].set_yticklabels([])
    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="s",
                color=CGT,
                markeredgecolor="black",
                markeredgewidth=0.4,
                linestyle="",
                markersize=4,
                label="arm mean, cell graph transformer",
            ),
            Line2D(
                [],
                [],
                marker="o",
                color="black",
                alpha=0.55,
                linestyle="",
                markersize=2.5,
                label="one run (seed, fold, machine)",
            ),
            Line2D(
                [],
                [],
                color=RIDGE,
                linewidth=0.9,
                label="nested ridge on FCFP4 counts (public side)",
            ),
            Line2D(
                [],
                [],
                color=LOEWE,
                linewidth=0.9,
                label="Loewe from the titration fits (growth call)",
            ),
        ],
        loc="outside lower center",
        ncol=4,
        fontsize=4.8,
        frameon=False,
    )
    fig.suptitle(
        "every scored arm, rounds 1 to 4, both machines, sorted by the public score;"
        " (n) is the number of runs",
        fontsize=6,
    )
    for ax, ch in zip(axes, "ab"):
        ax.text(
            -0.02 if ch == "b" else -0.55,
            1.04,
            ch,
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
    os.makedirs(OUT, exist_ok=True)
    stem = osp.join(OUT, f"r3_arms_overview_{timestamp()}")
    fig.savefig(stem + ".png", dpi=300)
    savefig_true_size_svg(fig, stem + ".svg")
    print(stem + ".png")
    print(f"{len(order)} arms, {len(gene)} gene-level runs, {len(growth)} growth runs")


if __name__ == "__main__":
    main()
