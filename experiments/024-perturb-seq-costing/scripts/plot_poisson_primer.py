# experiments/024-perturb-seq-costing/scripts/plot_poisson_primer.py
# [[experiments.024-perturb-seq-costing.scripts.plot_poisson_primer]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/024-perturb-seq-costing/scripts/plot_poisson_primer
"""A primer on Poisson sampling, contextualized to guide delivery.

Sec. 4.1 introduces the distribution and Sec. 7.1 then goes straight into a
zero-truncated Poisson, which is a big step for a reader who has not met it
before. This figure is the step in between: three panels that build the idea on
the actual problem rather than on an abstract example, so the arithmetic in the
text lands on something already seen.

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

HOUSE STYLE, and this figure used to break it. Bar faces are the PLOT_PALETTE
LINE colors with black edges, the same as plot_compression.py and
plot_economics.py; the pale PLOT_PALETTE_FILL colors are the draw.io companion
and are never a bar face. An earlier version filled every bar with the pale
fill and drew the border in the line color, which read as a different figure
from the rest of the document while using the same palette.

Run:  python experiments/024-perturb-seq-costing/scripts/plot_poisson_primer.py
"""

from __future__ import annotations

import math
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

from design_equation import box, place_panel_letters, style
from figure_checks import assert_legible
from scaling_analysis import lam_for_target_mean
from torchcell.utils import PANEL_WIDTHS_MM, mm_to_in, savefig_true_size_svg

load_dotenv()
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
OUT_DIR = osp.join(ASSET_IMAGES_DIR, "024-perturb-seq-costing")

# Palette slots: orange = what you keep, red = what is lost, purple = the action.
# These are PLOT_PALETTE line colors and are used as bar FACES; the pale
# PLOT_PALETTE_FILL siblings appear only as the reference outline in panel (b).
KEEP, LOST, ACTION, MUTED = "#D79B00", "#B85450", "#9673A6", "#666666"
GHOST = "#B3B3B3"  # the before-selection reference outline in panel (b)

# The operating point the text uses for a 2-guide target.
TARGET_MEAN = 2.0
LAM = lam_for_target_mean(TARGET_MEAN)
MS = np.arange(0, 8)


# NO MATHTEXT WHERE A SPACE ADJOINS IT. savefig_true_size_svg writes
# svg.fonttype="none", so text stays as real <text> elements, and matplotlib
# positions each mathtext run separately: the space between a $...$ run and the
# plain text beside it is dropped by rsvg-convert on the way to PDF. It renders
# correctly in the PNG and wrongly in the document, which is the worst way for
# this to fail. "mean 1.59 -> 2.00" arrived as "mean1.59->2.00" and "$k$ is a
# spread" as "kis a spread". Unicode in a plain string keeps its spaces.
LAMBDA = "\u03bb"
ARROW = "\u2192"


def pois(m, lam: float) -> np.ndarray:
    return np.exp(-lam) * lam**m / np.array([math.factorial(int(i)) for i in np.atleast_1d(m)])


def panel_a(ax) -> None:
    """The raw process: you set an average, the cells spread around it."""
    p = pois(MS, LAM)
    colors = [LOST] + [KEEP] * (len(MS) - 1)
    ax.bar(MS, p, width=0.7, color=colors, edgecolor="black", lw=0.5)
    ax.axvline(LAM, color=ACTION, lw=0.8, ls="--", zorder=3)
    ax.text(LAM + 0.2, 0.545, f"{LAMBDA} = {LAM:.2f}", fontsize=5,
            color=ACTION, va="center", ha="left")
    # Label over the empty tail, elbow leader back to the zero bar. The elbow is
    # not decoration: the label is ~3.4 x-units wide at this panel width, so over
    # the zero bar it runs under the lambda rule, and any STRAIGHT leader from the
    # tail to a bar top of 0.204 passes below bar 1's top of 0.325. Going across
    # at 0.455 and then straight down the x=0 column is the only path that touches
    # nothing.
    ax.plot([2.90, 0, 0], [0.455, 0.455, p[0] + 0.014], color=LOST, lw=0.5,
            solid_joinstyle="miter", zorder=2)
    ax.text(3.05, 0.455, "take up nothing,\ndie on selection", fontsize=4.6,
            color=LOST, ha="left", va="center", linespacing=1.15)
    ax.set_xlabel("Plasmids a cell takes up")
    ax.set_ylabel("Fraction of cells")
    ax.set_ylim(0, 0.58)
    ax.set_xticks(MS)
    ax.set_title("Uptake is random", loc="left", fontsize=6)
    box(ax)


def panel_b(ax) -> None:
    """Selection deletes the zero class, which raises the mean of the rest."""
    p = pois(MS, LAM)
    trunc = np.where(MS == 0, 0.0, p / (1.0 - p[0]))
    ax.bar(MS, p, width=0.7, color="none", edgecolor=GHOST, lw=0.5,
           label="before selection")
    ax.bar(MS, trunc, width=0.7, color=KEEP, edgecolor="black", lw=0.5,
           label="after selection")
    # The two means, each labeled on its own rule. An arrow alone spanned only
    # 0.41 x-units on an axis running to 7.5 and rendered as a stray tick; the
    # arrow is kept for direction but the numbers carry the statement.
    ax.axvline(LAM, color=GHOST, lw=0.8, ls="--", zorder=3)
    ax.axvline(TARGET_MEAN, color=ACTION, lw=0.8, ls="--", zorder=3)
    ax.annotate("", xy=(TARGET_MEAN, 0.485), xytext=(LAM, 0.485),
                arrowprops=dict(arrowstyle="-|>", color=ACTION, lw=0.8,
                                mutation_scale=6))
    # Plain text with a unicode arrow, per the note above the pois() helper.
    ax.text(TARGET_MEAN + 0.35, 0.485,
            f"mean {LAM:.2f} {ARROW} {TARGET_MEAN:.2f}",
            fontsize=5, color=ACTION, va="center", ha="left")
    ax.set_xlabel("Plasmids a surviving cell carries")
    ax.set_ylabel("Fraction of cells")
    ax.set_ylim(0, 0.58)
    ax.set_xticks(MS)
    ax.set_title("Selection lifts the mean", loc="left", fontsize=6)
    # Lower right, not upper right: the top strip belongs to the mean
    # annotation, and the tail of the distribution leaves that corner empty.
    ax.legend(frameon=False, fontsize=4.6, loc="center right", handlelength=1.2,
              labelspacing=0.25, borderaxespad=0.3,
              bbox_to_anchor=(1.0, 0.42))
    box(ax)


def panel_c(ax) -> None:
    """The design consequence: k is a spread you steer, not a number you set."""
    lams = np.linspace(0.05, 6.0, 400)
    denom = 1.0 - np.exp(-lams)
    p1 = (np.exp(-lams) * lams) / denom
    ax.plot(lams, p1, color=KEEP, lw=1.1)
    ax.plot(lams, 1.0 - p1, color=ACTION, lw=1.1)
    ax.text(5.8, 0.70, "2 or more\nguides", fontsize=5, color=ACTION,
            ha="right", va="center")
    ax.text(5.8, 0.22, "exactly 1", fontsize=5, color=KEEP, ha="right",
            va="center")
    for lam, lab in ((lam_for_target_mean(2.0), "2"),
                     (lam_for_target_mean(3.0), "3"),
                     (lam_for_target_mean(5.0), "5")):
        ax.axvline(lam, color=GHOST, lw=0.5, zorder=0)
        ax.text(lam, 0.035, lab, fontsize=4.6, color=MUTED, ha="center",
                va="bottom")
    ax.set_xlabel(f"Uptake rate {LAMBDA}")
    ax.set_ylabel("Fraction of surviving cells")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0, 6.0)
    ax.set_title("k is a spread, not a setting", loc="left", fontsize=6)
    box(ax)


def main() -> None:
    style()
    os.makedirs(OUT_DIR, exist_ok=True)
    # Full page width, not "wide". At 118.9 mm each of the three panels was
    # 33 mm across, which is what pushed the panel (b) legend onto its bars and
    # squeezed panel (c)'s in-plot labels against the frame.
    w = PANEL_WIDTHS_MM["full"]
    fig, axes = plt.subplots(1, 3, figsize=(mm_to_in(w), mm_to_in(52)))
    panel_a(axes[0])
    panel_b(axes[1])
    panel_c(axes[2])
    # top=0.84 rather than 0.86: place_panel_letters clamps a letter to y=0.985
    # and sets it va="bottom", so at 8 pt the glyph ran off the canvas and the
    # (c) letter was cropped. The clamp is a backstop, not a layout -- the
    # layout has to reserve the room.
    fig.subplots_adjust(left=0.062, right=0.995, top=0.84, bottom=0.19,
                        wspace=0.40)
    place_panel_letters(fig, axes, "abc")

    # The other script that had no legibility call. The cropped letter above was
    # found and fixed by eye; check_inside_figure is what keeps it fixed.
    assert_legible(fig, axes=list(axes))

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
