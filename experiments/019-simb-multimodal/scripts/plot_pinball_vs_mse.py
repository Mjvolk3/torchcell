# experiments/019-simb-multimodal/scripts/plot_pinball_vs_mse.py
# [[experiments.019-simb-multimodal.scripts.plot_pinball_vs_mse]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/plot_pinball_vs_mse
"""What the quantile head + pinball loss looks like next to a point head + MSE.

Three panels, and the provenance of each differs -- stated per panel in the caption
because two are exact and one is measured:

  (a) THE LOSS FUNCTIONS, exact. Pinball rho_tau(u) = max(tau*u, (tau-1)*u) at three tau
      against squared error u^2, as a function of the residual u = y - yhat. Pinball is
      piecewise LINEAR and ASYMMETRIC (that asymmetry is what pins a knot at its own tau);
      MSE is symmetric and quadratic, so it grows far faster on a large residual.
  (b) WHAT EACH HEAD EMITS for one gene, illustrative. The point head emits ONE number.
      The quantile head emits 19, the knots at tau = 0.05..0.95 (torch.linspace(0.05,
      0.95, 19), DEFAULT_NUM_QUANTILES = 19). Only the MEDIAN knot (tau = 0.50) is read by
      pearson_per_feature via DistHead.point(), so 18 of 19 are trained against a quantity
      the metric never inspects. The knot VALUES here are drawn from a Gaussian for
      illustration; the tau grid and the median-only readout are exact.
  (c) CALIBRATION, measured. Nominal coverage against what the incumbent actually achieves
      at epoch 10,000: coverage_80 = 0.578 against 0.80 and coverage_50 = 0.326 against
      0.50 (sec:campaign). The intervals are far too narrow, while the point estimates are
      simultaneously OVER-dispersed at s/r = 1.92 -- two calibration failures in opposite
      directions, neither visible to a correlation metric.

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
    """All four spines, which is the house style rather than the despined look."""
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)


def main() -> None:
    w = mm_to_in(PANEL_WIDTHS_MM["third"])
    fig, axes = plt.subplots(1, 3, figsize=(w * 3, mm_to_in(46)))

    # ---- (a) the loss functions ------------------------------------------------------
    ax = axes[0]
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
    ax.legend(frameon=False, loc="upper center", handlelength=1.4)
    ax.set_title("asymmetric + linear vs symmetric + quadratic", fontsize=6, pad=3)
    _box(ax)
    panel_label(ax, "a")

    # ---- (b) what each head emits for ONE gene ---------------------------------------
    ax = axes[1]
    rng = np.random.default_rng(0)
    mu, sd = -0.35, 0.45
    from scipy.stats import norm

    knots = norm.ppf(TAUS, loc=mu, scale=sd)  # illustrative knot VALUES
    ax.scatter(knots, TAUS, s=6, color=PLOT_PALETTE[2], zorder=3,
               label=f"quantile head: {N_QUANTILES} knots")
    ax.scatter([knots[MEDIAN_IDX]], [TAUS[MEDIAN_IDX]], s=26, marker="D",
               facecolor=PLOT_PALETTE[1], edgecolor="black", lw=0.5, zorder=5,
               label=r"median knot ($\tau$=0.50), the only one read")
    ax.axvline(knots[MEDIAN_IDX], color=PLOT_PALETTE[1], lw=0.7, ls=":", zorder=2)
    # THE POINT HEAD HAS NO tau. Drawing it at tau=0.50 would both hide it under the
    # median diamond and imply it carries a quantile level, which is the confusion this
    # panel exists to remove. It gets its own strip below the tau axis instead.
    ax.axhline(-0.07, color="black", lw=0.4, alpha=0.35)
    ax.scatter([mu], [-0.13], s=26, marker="o", facecolor=PLOT_PALETTE[5],
               edgecolor="black", lw=0.5, zorder=4, clip_on=False,
               label=r"point head: 1 number, no $\tau$")
    ax.axvline(mu, color=PLOT_PALETTE[5], lw=0.7, ls="--", zorder=1, alpha=0.7)
    # The 80% interval the coverage diagnostic reads.
    lo, hi = knots[1], knots[-2]  # tau = 0.10 and 0.90
    ax.annotate("", xy=(lo, 0.06), xytext=(hi, 0.06),
                arrowprops=dict(arrowstyle="<->", lw=0.6, color="black"))
    ax.text((lo + hi) / 2, 0.11, r"80% interval", ha="center", fontsize=6)
    ax.set_xlabel(r"predicted $\log_2$ ratio for one gene")
    ax.set_ylabel(r"quantile level $\tau$")
    ax.set_ylim(-0.2, 1.05)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="y", which="minor", length=0)
    ax.grid(axis="y", which="both", lw=0.3, alpha=0.35)
    ax.legend(frameon=False, loc="upper left", handlelength=1.0, scatterpoints=1,
              borderpad=0.2, labelspacing=0.25)
    ax.set_title("the metric reads 1 of 19", fontsize=6, pad=3)
    _box(ax)
    panel_label(ax, "b")

    # ---- (c) measured calibration ----------------------------------------------------
    ax = axes[2]
    nominal = np.array(sorted(COVERAGE))
    actual = np.array([COVERAGE[n] for n in nominal])
    x = np.arange(len(nominal))
    ax.bar(x - 0.19, nominal, width=0.36, color=PLOT_PALETTE[5], edgecolor="black",
           lw=0.5, label="nominal")
    ax.bar(x + 0.19, actual, width=0.36, color=PLOT_PALETTE[1], edgecolor="black",
           lw=0.5, hatch="///", label="measured")
    for xi, (n, a) in enumerate(zip(nominal, actual)):
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
    ax.legend(frameon=False, loc="upper left", handlelength=1.0)
    ax.set_title("intervals too narrow (measured)", fontsize=6, pad=3)
    _box(ax)
    panel_label(ax, "c")

    fig.tight_layout(pad=0.4)
    out_dir = os.environ["ASSET_IMAGES_DIR"]
    stem = osp.join(out_dir, "019-simb-multimodal", "pinball_vs_mse")
    os.makedirs(osp.dirname(stem), exist_ok=True)
    fig.savefig(f"{stem}.png", dpi=300)
    savefig_true_size_svg(fig, f"{stem}.svg")
    print(f"wrote {stem}.png\nwrote {stem}.svg")
    print(f"tau grid ({N_QUANTILES}): {[round(t, 2) for t in TAUS]}")
    print(f"median knot index {MEDIAN_IDX} -> tau {TAUS[MEDIAN_IDX]:.2f}")


if __name__ == "__main__":
    main()
