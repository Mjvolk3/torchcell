# experiments/019-simb-multimodal/scripts/plot_pinball_vs_mse.py
# [[experiments.019-simb-multimodal.scripts.plot_pinball_vs_mse]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/plot_pinball_vs_mse
"""What the pinball loss actually DOES, next to MSE, and what the metric then reads.

THE CONFUSION THIS EXISTS TO REMOVE. "The metric reads 1 of 19" and "all 19 knots are
trained" are both true and describe different steps. TRAINING touches all nineteen: each
knot has its own pinball term, and every one of them sends gradient back through the SAME
shared trunk that produces the median. EVALUATION touches one: pearson_per_feature reads
DistHead.point(), which for a quantile head is the median knot alone. So the other 18 do
not sit idle -- they steer the shared weights, and the metric never inspects the result.

  (a) THE LOSS FUNCTIONS, exact. rho_tau(u) = max(tau*u, (tau-1)*u) at three tau against
      u^2. Pinball is piecewise LINEAR and ASYMMETRIC, which is what pins each knot at its
      own tau; MSE is symmetric and quadratic.
  (b) ONE GENE, illustrative values. The head emits 19 knots; the truth is one number. The
      vertical offset from each knot to the truth is that knot's residual u, which is the
      input to its own pinball term.
  (c) WHAT THE LOSS IS MADE OF, computed from panel (b). One bar per knot: its pinball
      contribution. The median bar is highlighted because it is the ONLY one the metric
      later reads, and it carries 8.1% of the total -- the remaining 91.9% is gradient
      spent on knots that never appear in the score.
  (d) CALIBRATION, measured. Nominal against achieved coverage for the incumbent at epoch
      10,000: coverage_80 = 0.578 against 0.80, coverage_50 = 0.326 against 0.50
      (sec:campaign). So the spread that 91.9% of the loss is buying is itself wrong.

Panels (b) and (c) use illustrative knot VALUES; the tau grid, the pinball formula, the
median-only readout and the arithmetic relating (b) to (c) are exact.

Run from the repo root:
    python experiments/019-simb-multimodal/scripts/plot_pinball_vs_mse.py
"""

from __future__ import annotations

import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from matplotlib.ticker import MultipleLocator
from scipy.stats import norm

load_dotenv()

from torchcell.utils import (  # noqa: E402
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    panel_label,
    savefig_true_size_svg,
)

# tau grid, verbatim from torchcell/losses/distributional.py
N_QUANTILES = 19
TAUS = np.linspace(0.05, 0.95, N_QUANTILES)
MEDIAN_IDX = int(np.argmin(np.abs(TAUS - 0.5)))
# One illustrative gene: predicted distribution and the single observed truth.
MU, SD, Y_TRUE = -0.35, 0.45, -0.62
# MEASURED, sec:campaign / the incumbent at epoch 10,000.
COVERAGE = {0.50: 0.326, 0.80: 0.578}

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 6,
        "axes.linewidth": 0.5,
        "svg.fonttype": "none",
        "axes.labelsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
    }
)


def _box(ax: plt.Axes) -> None:
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)


def main() -> None:
    knots = norm.ppf(TAUS, loc=MU, scale=SD)
    resid = Y_TRUE - knots
    rho = np.maximum(TAUS * resid, (TAUS - 1) * resid)
    med_share = rho[MEDIAN_IDX] / rho.sum()

    w = mm_to_in(PANEL_WIDTHS_MM["half"])
    fig, axes = plt.subplots(2, 2, figsize=(w * 2, mm_to_in(88)))

    # ---- (a) the loss functions ------------------------------------------------------
    ax = axes[0, 0]
    u = np.linspace(-2, 2, 601)
    for tau, color in ((0.05, PLOT_PALETTE[0]), (0.50, PLOT_PALETTE[1]),
                       (0.95, PLOT_PALETTE[2])):
        ax.plot(u, np.maximum(tau * u, (tau - 1) * u), color=color, lw=1.0,
                label=rf"pinball $\tau$={tau:.2f}")
    ax.plot(u, u**2, color=PLOT_PALETTE[5], lw=1.0, ls="--", label="MSE $u^2$")
    ax.axvline(0, color="black", lw=0.4, alpha=0.5)
    ax.set_xlabel(r"residual $u = y - \hat{y}$")
    ax.set_ylabel("loss")
    ax.set_ylim(0, 2)
    ax.legend(frameon=False, loc="upper center", handlelength=1.4, borderpad=0.2)
    ax.set_title(r"each $\tau$ has its own asymmetric loss", fontsize=6, pad=3)
    _box(ax)
    panel_label(ax, "a")

    # ---- (b) one gene: 19 knots and ONE truth ----------------------------------------
    ax = axes[0, 1]
    ax.scatter(TAUS, knots, s=7, color=PLOT_PALETTE[2], zorder=3, label="19 predicted knots")
    ax.scatter([TAUS[MEDIAN_IDX]], [knots[MEDIAN_IDX]], s=28, marker="D",
               facecolor=PLOT_PALETTE[1], edgecolor="black", lw=0.5, zorder=5,
               label=r"median knot ($\tau$=0.50)")
    ax.axhline(Y_TRUE, color="black", lw=0.8, ls="-", zorder=2, label="the one true value $y$")
    # the residual each knot is scored on
    ax.vlines(TAUS, np.minimum(knots, Y_TRUE), np.maximum(knots, Y_TRUE),
              color=PLOT_PALETTE[2], lw=0.5, alpha=0.45, zorder=1)
    ax.set_xlabel(r"quantile level $\tau$")
    ax.set_ylabel(r"predicted $\log_2$ ratio")
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.legend(frameon=False, loc="upper left", handlelength=1.2, borderpad=0.2,
              labelspacing=0.25)
    ax.set_title("each knot gets its own residual", fontsize=6, pad=3)
    _box(ax)
    panel_label(ax, "b")

    # ---- (c) what the loss is made of ------------------------------------------------
    ax = axes[1, 0]
    colors = [PLOT_PALETTE[2]] * N_QUANTILES
    colors[MEDIAN_IDX] = PLOT_PALETTE[1]
    ax.bar(TAUS, rho, width=0.035, color=colors, edgecolor="black", lw=0.4)
    ax.annotate(f"median knot: {med_share*100:.1f}% of the\nloss, and the only one scored",
                xy=(TAUS[MEDIAN_IDX], rho[MEDIAN_IDX]), xytext=(0.03, 0.108),
                fontsize=6, arrowprops=dict(arrowstyle="->", lw=0.6, color="black"))
    ax.set_xlabel(r"quantile level $\tau$")
    ax.set_ylabel(r"pinball contribution $\rho_\tau(u)$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.20)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.set_title(f"the other 18 carry {100-med_share*100:.1f}% of the gradient",
                 fontsize=6, pad=3)
    _box(ax)
    panel_label(ax, "c")

    # ---- (d) measured calibration ----------------------------------------------------
    ax = axes[1, 1]
    nominal = np.array(sorted(COVERAGE))
    actual = np.array([COVERAGE[n] for n in nominal])
    x = np.arange(len(nominal))
    ax.bar(x - 0.19, nominal, width=0.36, color=PLOT_PALETTE[5], edgecolor="black",
           lw=0.5, label="nominal")
    ax.bar(x + 0.19, actual, width=0.36, color=PLOT_PALETTE[1], edgecolor="black",
           lw=0.5, hatch="///", label="measured")
    for xi, a in zip(x, actual):
        ax.text(xi + 0.19, a + 0.02, f"{a:.3f}", ha="center", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(n*100)}%" for n in nominal])
    ax.set_xlabel("predictive interval")
    ax.set_ylabel("fraction of truths covered")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(axis="y", which="minor", length=0)
    ax.grid(axis="y", which="both", lw=0.3, alpha=0.35)
    ax.legend(frameon=False, loc="upper left", handlelength=1.0, borderpad=0.2)
    ax.set_title("and that spread is miscalibrated (measured)", fontsize=6, pad=3)
    _box(ax)
    panel_label(ax, "d")

    fig.tight_layout(pad=0.4)
    out_dir = os.environ["ASSET_IMAGES_DIR"]
    stem = osp.join(out_dir, "019-simb-multimodal", "pinball_vs_mse")
    os.makedirs(osp.dirname(stem), exist_ok=True)
    fig.savefig(f"{stem}.png", dpi=300)
    savefig_true_size_svg(fig, f"{stem}.svg")
    print(f"wrote {stem}.png")
    print(f"total pinball (mean over {N_QUANTILES}) = {rho.mean():.4f}")
    print(f"median knot share {med_share*100:.1f}%, other 18 = {100-med_share*100:.1f}%")


if __name__ == "__main__":
    main()
