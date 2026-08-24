# experiments/019-perturb-seq-costing/scripts/plot_poisson_primer.py
# [[experiments.019-perturb-seq-costing.scripts.plot_poisson_primer]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-perturb-seq-costing/scripts/plot_poisson_primer
"""A primer on Poisson sampling, contextualized to guide delivery.

Sec. 7.1 goes straight into a zero-truncated Poisson, and that is a big step for
a reader who has not met the distribution before. This figure is the step before
it: three panels that build the idea on the actual problem rather than on an
abstract example, so the arithmetic in the text lands on something already seen.

The pedagogical order is the one that matters, and it is not the order a
statistics course would use:

* (a) You cannot hand a cell two plasmids. You can only set an average, and the
  cells sort themselves into a distribution around it. That is the whole idea,
  and it is the part people find counterintuitive when they first meet it.
* (b) Selection deletes the zero class, which RAISES the mean of what survives.
  This is the zero-truncation, and it is why the lambda in the text is always
  lower than the plasmids-per-cell figure beside it.
* (c) The design consequence: k is a spread, not a setting. Turning the dial up
  to get more doubles also gets you triples and quadruples you did not ask for.

The same distribution governs droplet loading (Sec. 2.2), where the design goal
is the opposite -- keep lambda LOW so that two cells rarely share a droplet.
Panel (c) marks both regimes so the two uses are visibly the same mathematics.

Run:  python experiments/019-perturb-seq-costing/scripts/plot_poisson_primer.py
"""

from __future__ import annotations

import math
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

from design_equation import box, place_panel_letters, style
from scaling_analysis import lam_for_target_mean
from torchcell.utils import PANEL_WIDTHS_MM, mm_to_in, savefig_true_size_svg

load_dotenv()
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
OUT_DIR = osp.join(ASSET_IMAGES_DIR, "019-perturb-seq-costing")

# Palette slots: orange = what you keep, red = what is lost, purple = the action.
KEEP, LOST, ACTION, MUTED = "#D79B00", "#B85450", "#9673A6", "#666666"
FILL_KEEP, FILL_LOST = "#FFE6CC", "#F8CECC"

# The operating point the text uses for a 2-guide target.
TARGET_MEAN = 2.0
LAM = lam_for_target_mean(TARGET_MEAN)
MS = np.arange(0, 8)


def pois(m, lam: float) -> np.ndarray:
    return np.exp(-lam) * lam**m / np.array([math.factorial(int(i)) for i in np.atleast_1d(m)])


def panel_a(ax) -> None:
    """The raw process: you set an average, the cells spread around it."""
    p = pois(MS, LAM)
    colors = [FILL_LOST] + [FILL_KEEP] * (len(MS) - 1)
    edges = [LOST] + [KEEP] * (len(MS) - 1)
    ax.bar(MS, p, width=0.72, color=colors, edgecolor=edges, lw=0.6)
    ax.axvline(LAM, color=ACTION, lw=0.8, ls="--")
    ax.text(LAM + 0.15, 0.50, f"$\\lambda={LAM:.2f}$", fontsize=5,
            color=ACTION, va="top", ha="left")
    ax.text(0, p[0] + 0.015, "dies", fontsize=4.2, color=LOST, ha="center",
            va="bottom")
    ax.set_xlabel("Plasmids a cell takes up")
    ax.set_ylabel("Fraction of cells")
    ax.set_ylim(0, 0.54)
    ax.set_xticks(MS)
    ax.set_title("Uptake is random", loc="left", fontsize=6)
    box(ax)


def panel_b(ax) -> None:
    """Selection deletes the zero class, which raises the mean of the rest."""
    p = pois(MS, LAM)
    trunc = np.where(MS == 0, 0.0, p / (1.0 - p[0]))
    ax.bar(MS, p, width=0.72, color="none", edgecolor="#CCCCCC", lw=0.6,
           label="before selection")
    ax.bar(MS, trunc, width=0.72, color=FILL_KEEP, edgecolor=KEEP, lw=0.6,
           label="after selection")
    ax.axvline(LAM, color="#CCCCCC", lw=0.8, ls="--")
    ax.axvline(TARGET_MEAN, color=ACTION, lw=0.8, ls="--")
    ax.annotate("", xy=(TARGET_MEAN, 0.47), xytext=(LAM, 0.47),
                arrowprops=dict(arrowstyle="->", color=ACTION, lw=0.9))
    ax.text(7.45, 0.515, f"mean $\\to$ {TARGET_MEAN:.0f}",
            fontsize=5, color=ACTION, va="top", ha="right")
    ax.set_xlabel("Plasmids a surviving cell carries")
    ax.set_ylabel("Fraction of cells")
    ax.set_ylim(0, 0.54)
    ax.set_xticks(MS)
    ax.set_title("Selection lifts the mean", loc="left", fontsize=6)
    ax.legend(frameon=False, fontsize=4.2, loc="center right", handlelength=1.3,
              labelspacing=0.3, borderaxespad=0.4)
    box(ax)


def panel_c(ax) -> None:
    """The design consequence: k is a spread you steer, not a number you set."""
    lams = np.linspace(0.05, 6.0, 400)
    denom = 1.0 - np.exp(-lams)
    p1 = (np.exp(-lams) * lams) / denom
    ax.plot(lams, p1, color=KEEP, lw=1.1)
    ax.plot(lams, 1.0 - p1, color=ACTION, lw=1.1)
    ax.text(5.85, 0.66, "2 or more guides", fontsize=5, color=ACTION,
            ha="right", va="center")
    ax.text(5.85, 0.26, "exactly 1", fontsize=5, color=KEEP, ha="right")
    for lam, lab in ((lam_for_target_mean(2.0), "2"),
                     (lam_for_target_mean(3.0), "3"),
                     (lam_for_target_mean(5.0), "5")):
        ax.axvline(lam, color="#CCCCCC", lw=0.5, zorder=0)
        ax.text(lam, 0.055, lab, fontsize=4.2, color=MUTED, ha="center",
                va="bottom")
    ax.set_xlabel("Uptake rate $\\lambda$")
    ax.set_ylabel("Fraction of surviving cells")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0, 6.0)
    ax.set_title("$k$ is a spread", loc="left", fontsize=6)
    box(ax)


def main() -> None:
    style()
    os.makedirs(OUT_DIR, exist_ok=True)
    w = PANEL_WIDTHS_MM["wide"]
    fig, axes = plt.subplots(1, 3, figsize=(mm_to_in(w), mm_to_in(46)))
    panel_a(axes[0])
    panel_b(axes[1])
    panel_c(axes[2])
    fig.subplots_adjust(left=0.085, right=0.995, top=0.86, bottom=0.21,
                        wspace=0.52)
    place_panel_letters(fig, axes, "abc")
    savefig_true_size_svg(fig, osp.join(OUT_DIR, "poisson_primer.svg"))
    fig.savefig(osp.join(OUT_DIR, "poisson_primer.png"), dpi=300)
    plt.close(fig)

    p = pois(MS, LAM)
    print(f"lambda = {LAM:.4f} gives a post-selection mean of {TARGET_MEAN}")
    print(f"  cells taking up nothing: {p[0]*100:.1f}% (all die on selection)")
    trunc = p[1:] / (1 - p[0])
    for m, v in zip(MS[1:], trunc):
        print(f"  {m} plasmid(s): {v*100:5.1f}% of survivors")
    print(f"wrote {OUT_DIR}/poisson_primer.svg")


if __name__ == "__main__":
    main()
