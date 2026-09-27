# experiments/019-simb-multimodal/scripts/v19_joint_mockup.py
# [[experiments.019-simb-multimodal.scripts.v19_joint_mockup]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/v19_joint_mockup
"""Wireframe of the figure panels the joint proteome-and-expression rounds are expected to
produce, as they would sit in the paper's Figure 3 (one embedding, many phenotypes).

NOTHING HERE IS MEASURED. Every curve and point is sketched to show the SHAPE each panel
takes and the decision it carries; the axes, statistics, windows and margins are the ones
pre-registered in conf/cgt_expr_v19_joint_clean.yaml and in
notes-tex/019-simb-multimodal-expression/sections/5-joint-plan.tex. When the rounds land,
the readout scripts replace every sketch with the run data and this file is retired.

Panels
  a  expression head, K_expr against K_joint, one pair per partition (12), window mean
  b  proteome head, K_prot against K_joint, the same partitions, its own window
  c  the pre-registered test: paired differences with one-sided 95% bounds against 0
     (expression superiority) and against -0.015 (proteome non-inferiority); the
     permuted-label interaction beside them
  d  cross-modal conditioning (v20): held-out Pearson as a function of how much of the
     other modality is revealed, against the genotype-only floor and a shuffled-strain null
  e  validation curves of the two heads over training, with the per-head scoring windows
  f  the launch census: prediction spread at epoch 200 per run, the 0.05 gate, relaunches

Writes $ASSET_IMAGES_DIR/019-simb-multimodal/v19_joint_mockup.{svg,png} and a timestamped
svg. Import the svg true-size into notes/assets/drawio/Fig3-options.drawio.
"""

from __future__ import annotations

import os
import os.path as osp

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from matplotlib.axes import Axes
from matplotlib.ticker import MultipleLocator

from torchcell.timestamp import timestamp
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    panel_label,
    savefig_true_size_svg,
)

load_dotenv()
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
IMG_DIR = osp.join(ASSET_IMAGES_DIR, "019-simb-multimodal")

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 6,
        "axes.labelsize": 6,
        "axes.titlesize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 5,
        "svg.fonttype": "none",
        "axes.linewidth": 0.5,
    }
)
ORANGE, RED, PURPLE, YELLOW, BLUE, GRAY = PLOT_PALETTE[:6]
ARM = {"K_prot": ORANGE, "K_expr": RED, "K_joint": PURPLE, "K_perm": YELLOW}
N_PART = 12
RNG = np.random.default_rng(19)


def _box(ax: Axes) -> None:
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)
        s.set_color("black")


def _pearson_grid(ax: Axes, major: float = 0.1, minor: float = 0.05) -> None:
    ax.yaxis.set_major_locator(MultipleLocator(major))
    ax.yaxis.set_minor_locator(MultipleLocator(minor))
    ax.grid(True, axis="y", which="both", linewidth=0.3, color="#DDDDDD")
    ax.tick_params(axis="y", which="minor", length=0)


def _paired_panel(
    ax: Axes, ref: str, ref_level: float, joint_delta: float, spread: float, title: str
) -> None:
    """One pair per partition: the reference arm and the joint arm joined by a line."""
    part = np.arange(N_PART)
    base = ref_level + spread * np.sin(part / 2.1) + RNG.normal(0, 0.006, N_PART)
    joint = base + joint_delta + RNG.normal(0, 0.008, N_PART)
    for p in part:
        ax.plot([p - 0.15, p + 0.15], [base[p], joint[p]], color=GRAY, lw=0.5, zorder=1)
    ax.scatter(part - 0.15, base, s=9, color=ARM[ref], zorder=2, label=ref)
    ax.scatter(part + 0.15, joint, s=9, color=ARM["K_joint"], zorder=2, label="K_joint")
    ax.axhline(base.mean(), color=ARM[ref], lw=0.6, ls=":")
    ax.axhline(joint.mean(), color=ARM["K_joint"], lw=0.6, ls=":")
    ax.set_xticks(part)
    ax.set_xticklabels([str(p) for p in part])
    ax.set_xlabel("partition (split seed)")
    ax.set_ylabel("held-out Pearson, window mean")
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=True, edgecolor="black", fancybox=False)
    _pearson_grid(ax)
    _box(ax)


def panel_c(ax: Axes) -> None:
    """The pre-registered test as a forest plot."""
    rows = [
        ("expr:\njoint - expr", 0.012, 0.011, 0.0, PURPLE),
        ("prot:\njoint - prot", -0.004, 0.009, -0.015, ORANGE),
        ("expr:\nperm interaction", 0.009, 0.013, 0.0, YELLOW),
        ("prot:\nperm interaction", 0.001, 0.011, 0.0, YELLOW),
    ]
    ys = []
    for i, (label, mean, half, bound, color) in enumerate(rows):
        y = len(rows) - i
        ys.append((y, label))
        ax.plot([mean - half, mean + half], [y, y], color=color, lw=1.2)
        ax.scatter([mean], [y], s=14, color=color, zorder=3)
        ax.plot([bound, bound], [y - 0.3, y + 0.3], color="black", lw=0.6, ls="--")
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlim(-0.045, 0.045)
    ax.set_ylim(0.4, len(rows) + 0.6)
    ax.set_yticks([y for y, _ in ys])
    ax.set_yticklabels([lab for _, lab in ys], fontsize=5)
    ax.set_xlabel(
        "paired difference, window Pearson, 12 partitions\n"
        "bar: one-sided 95% bound; dashed: bound to clear"
    )
    ax.set_title("the test: superiority at 0, non-inferiority at -0.015")
    _box(ax)


def panel_d(ax: Axes) -> None:
    """Cross-modal conditioning: the v20 read."""
    k = np.array([0, 10, 100, 1000, 1850])
    x = np.arange(len(k))
    expr_from_prot = np.array([0.10, 0.12, 0.16, 0.20, 0.22])
    prot_from_expr = np.array([0.07, 0.09, 0.14, 0.19, 0.22])
    null = np.array([0.10, 0.10, 0.09, 0.10, 0.09])
    ax.plot(
        x,
        expr_from_prot,
        "-o",
        ms=3,
        color=RED,
        label="predict expression, proteome revealed",
    )
    ax.plot(
        x,
        prot_from_expr,
        "-s",
        ms=3,
        color=ORANGE,
        label="predict proteome, expression revealed",
    )
    ax.plot(
        x,
        null,
        "--",
        color=GRAY,
        lw=0.8,
        label="another strain's proteome revealed (null)",
    )
    ax.fill_between(
        x, expr_from_prot - 0.019, expr_from_prot + 0.019, color=RED, alpha=0.12, lw=0
    )
    ax.set_xticks(x)
    ax.set_xticklabels(["0", "10", "100", "1,000", "all"])
    ax.set_xlabel("features of the other modality revealed")
    ax.set_ylabel("held-out Pearson, same strains")
    ax.set_title("v20: given one modality, predict the other")
    ax.set_ylim(0.0, 0.3)
    ax.legend(loc="upper left", frameon=True, edgecolor="black", fancybox=False)
    _pearson_grid(ax)
    _box(ax)


def panel_e(ax: Axes) -> None:
    """Two heads, two windows."""
    ep = np.arange(0, 1201, 10)
    prot = 0.11 * (1 - np.exp(-ep / 90)) * np.exp(-ep / 2500) + RNG.normal(
        0, 0.003, ep.size
    )
    expr = 0.19 * (1 - np.exp(-ep / 450)) + RNG.normal(0, 0.003, ep.size)
    ax.axvspan(200, 400, color=ORANGE, alpha=0.15, lw=0)
    ax.axvspan(1000, 1200, color=RED, alpha=0.15, lw=0)
    ax.plot(ep, prot, color=ORANGE, lw=0.9, label="proteome head (K_joint)")
    ax.plot(ep, expr, color=RED, lw=0.9, label="expression head (K_joint)")
    ax.text(300, 0.215, "proteome window", ha="center", fontsize=5, color=ORANGE)
    ax.text(1100, 0.215, "expression window", ha="center", fontsize=5, color=RED)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation Pearson")
    ax.set_title("per-head windows: one peaks early, one is still rising")
    ax.set_ylim(0, 0.24)
    ax.legend(loc="lower right", frameon=True, edgecolor="black", fancybox=False)
    _pearson_grid(ax)
    _box(ax)


def panel_f(ax: Axes) -> None:
    """The launch census: spread at epoch 200 per run, by arm."""
    arms = ["K_prot", "K_expr", "K_joint", "K_perm"]
    for i, arm in enumerate(arms):
        vals = np.clip(RNG.normal(0.22, 0.07, N_PART), 0.06, 0.45)
        if arm == "K_expr":
            vals[3] = 0.004  # one never-launched run, relaunched
        x = i + RNG.uniform(-0.18, 0.18, N_PART)
        ax.scatter(x, vals, s=9, color=ARM[arm])
        if arm == "K_expr":
            ax.scatter(
                [x[3]], [vals[3]], s=30, facecolor="none", edgecolor="black", lw=0.6
            )
            ax.annotate(
                "relaunched, new init",
                (x[3], vals[3]),
                (x[3] + 0.45, 0.13),
                fontsize=5,
                arrowprops={"arrowstyle": "-", "lw": 0.5},
            )
    ax.axhline(0.05, color="black", lw=0.6, ls="--")
    ax.text(-0.45, 0.06, "gate 0.05", fontsize=5, ha="left", va="bottom")
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(arms)
    ax.set_ylabel("prediction spread at epoch 200")
    ax.set_title("launch census at epoch 200")
    ax.set_ylim(0, 0.5)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.grid(True, axis="y", linewidth=0.3, color="#DDDDDD")
    _box(ax)


def wireframe() -> None:
    fig, axes = plt.subplots(
        2, 3, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(112))
    )
    fig.subplots_adjust(
        left=0.05, right=0.99, bottom=0.1, top=0.9, wspace=0.42, hspace=0.7
    )
    a, b, c, d, e, f = axes.ravel()
    _paired_panel(
        a,
        "K_expr",
        0.15,
        +0.012,
        0.03,
        "expression head: joint against expression-only",
    )
    _paired_panel(
        b, "K_prot", 0.09, -0.004, 0.012, "proteome head: joint against proteome-only"
    )
    panel_c(c)
    panel_d(d)
    panel_e(e)
    panel_f(f)
    for ax, letter in zip((a, b, c, d, e, f), "abcdef"):
        panel_label(ax, letter)
    fig.text(
        0.5,
        0.965,
        "PLANNED, no data: the v19 deconfounded joint round (a to c, e, f) and the v20 "
        "conditioned cross-modal round (d), sketched at the shapes the pre-registration expects.",
        ha="center",
        fontsize=6,
        color=RED,
    )
    os.makedirs(IMG_DIR, exist_ok=True)
    savefig_true_size_svg(fig, osp.join(IMG_DIR, "v19_joint_mockup.svg"))
    fig.savefig(osp.join(IMG_DIR, "v19_joint_mockup.png"), dpi=300)
    savefig_true_size_svg(fig, osp.join(IMG_DIR, f"v19_joint_mockup_{timestamp()}.svg"))
    plt.close(fig)


def main() -> None:
    wireframe()
    print(osp.join(IMG_DIR, "v19_joint_mockup.svg"))


if __name__ == "__main__":
    main()
