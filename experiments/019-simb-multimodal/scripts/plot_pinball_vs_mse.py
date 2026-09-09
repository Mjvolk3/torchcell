# experiments/019-simb-multimodal/scripts/plot_pinball_vs_mse.py
# [[experiments.019-simb-multimodal.scripts.plot_pinball_vs_mse]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/plot_pinball_vs_mse
"""What the pinball loss actually DOES, next to MSE, how its knots are read back as a
distribution, and what the metric then reads.

THE CONFUSION THIS EXISTS TO REMOVE. "The metric reads 1 of 19" and "all 19 knots are
trained" are both true and describe different steps. TRAINING touches all nineteen: each
knot has its own pinball term, and every one of them sends gradient back through the SAME
shared trunk that produces the median. EVALUATION touches one: pearson_per_feature reads
DistHead.point(), which for a quantile head is the median knot alone. So the other 18 do
not sit idle -- they steer the shared weights, and the metric never inspects the result.

Panels a-e are SYNTHETIC (one made-up gene, or simulated genes); panel f is MEASURED.

  (a) THE LOSS FUNCTIONS, exact. rho_tau(u) = max(tau*u, (tau-1)*u) at three tau against
      u^2. Pinball is piecewise LINEAR and ASYMMETRIC, which is what pins each knot at its
      own tau; MSE is symmetric and quadratic.
  (b) ONE GENE, illustrative values. The head emits 19 knots (here the quantiles of a
      Gaussian with mean MU and SD); the observation is ONE number y = Y_TRUE (the "truth",
      the single measured log2 ratio for that gene in that strain). The vertical offset from
      each knot to y is that knot's residual u, which is the input to its own pinball term.
  (c) WHAT THE LOSS IS MADE OF, computed from panel (b). One bar per knot: its pinball
      contribution. The median bar is highlighted because it is the ONLY one the metric
      later reads, and it carries 8.1% of the total -- the remaining 91.9% is gradient
      spent on knots that never appear in the score.
  (d) READING THE KNOTS BACK AS A DISTRIBUTION, same gene. Plot each knot as the point
      (q_tau, tau): those are 19 points on the predictive CDF. torchcell's _quantile_pit
      joins them piecewise-linearly and clamps to 0 / 1 outside the knot range (no tail is
      invented past tau = 0.05 / 0.95). From this curve: the central 80% interval is
      [q_0.10, q_0.90], the 50% is [q_0.25, q_0.75], their WIDTH is the model's own noise
      estimate for that gene, and F_hat(y) is the PIT -- where the truth fell inside the
      predicted distribution. Sorting the knots first repairs quantile crossing.
  (e) WHAT PIT HISTOGRAMS DIAGNOSE, simulated. N_SIM genes with truth y ~ N(0, 1); the head
      predicts the quantiles of N(0, r) with predicted/true width ratio r in three cases:
      r = 1 (calibrated, flat histogram), r = R_FIT (the ratio back-solved from panel f:
      intervals too narrow, U-shaped, truths pile into the tails), r = 1/R_FIT (too wide,
      hump at 0.5). coverage_50 / coverage_80 for each case is printed in the legend; the
      r = R_FIT case reproduces the measured values in (f), which is the check that the
      Gaussian-scale-mismatch reading of (f) is self-consistent. The mass exactly at 0 and
      1 in the calibrated case is the tau grid's 5% tails, not miscalibration.
  (f) CALIBRATION, MEASURED, as a reliability curve: nominal central-interval mass alpha
      against the fraction of truths it covers, for the incumbent at epoch 10,000:
      coverage_80 = 0.578 against 0.80, coverage_50 = 0.326 against 0.50 (sec:campaign).
      Both points sit below the diagonal (overconfident). Under a Gaussian scale mismatch
      each back-solves to r = Phi^-1((1+cov)/2) / Phi^-1((1+alpha)/2); the two give 0.624
      and 0.627, so ONE ratio (0.63, intervals 1.6x too narrow) explains both. The curve is
      that fitted model, the diamonds are measured; the ratio is derived, not measured.

Run from the repo root:
    python experiments/019-simb-multimodal/scripts/plot_pinball_vs_mse.py
"""

from __future__ import annotations

import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator
from scipy.stats import norm

load_dotenv()

from torchcell.utils import (  # noqa: E402
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    PLOT_PALETTE_FILL,
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
# Simulation size for panel (e).
N_SIM = 200_000
SEED = 0

plt.rcParams.update(
    {
        "legend.frameon": True,
        "legend.fancybox": False,
        "legend.framealpha": 1.0,
        "legend.edgecolor": "black",
        "legend.facecolor": "white",
        "patch.linewidth": 0.5,
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


def quantile_pit(quantiles: np.ndarray, y: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """numpy mirror of torchcell.losses.distributional._quantile_pit.

    ``quantiles`` is ``[N, K]``, ``y`` is ``[N]``. Knots are sorted, the CDF is the
    piecewise-linear interpolant through ``(q_k, tau_k)``, clamped to 0 below the lowest
    knot and 1 above the highest.
    """
    q = np.sort(quantiles, axis=-1)
    pit = np.array([np.interp(yi, qi, taus) for yi, qi in zip(y, q)])
    pit = np.where(y < q[:, 0], 0.0, pit)
    pit = np.where(y > q[:, -1], 1.0, pit)
    return pit


def coverage(pit: np.ndarray, alpha: float) -> float:
    lo, hi = 0.5 * (1 - alpha), 0.5 * (1 + alpha)
    return float(np.mean((pit >= lo) & (pit <= hi)))


def main() -> None:
    knots = norm.ppf(TAUS, loc=MU, scale=SD)
    resid = Y_TRUE - knots
    rho = np.maximum(TAUS * resid, (TAUS - 1) * resid)
    med_share = rho[MEDIAN_IDX] / rho.sum()

    # Panel (f) back-solve, needed by (e) as well.
    nominal = np.array(sorted(COVERAGE))
    actual = np.array([COVERAGE[n] for n in nominal])
    ratios = norm.ppf(0.5 * (1 + actual)) / norm.ppf(0.5 * (1 + nominal))
    r_fit = float(ratios.mean())

    fig, axes = plt.subplots(
        2, 3, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(112))
    )

    # ---- (a) the loss functions ------------------------------------------------------
    ax = axes[0, 0]
    u = np.linspace(-2, 2, 601)
    for tau, color in (
        (0.05, PLOT_PALETTE[0]),
        (0.50, PLOT_PALETTE[1]),
        (0.95, PLOT_PALETTE[2]),
    ):
        ax.plot(
            u,
            np.maximum(tau * u, (tau - 1) * u),
            color=color,
            lw=1.0,
            label=rf"pinball $\tau$={tau:.2f}",
        )
    ax.plot(u, u**2, color=PLOT_PALETTE[5], lw=1.0, ls="--", label="MSE $u^2$")
    ax.axvline(0, color="black", lw=0.4, alpha=0.5)
    ax.set_xlabel("residual u = y \u2212 \u0177 (observed minus predicted)")
    ax.set_ylabel("loss")
    ax.set_ylim(0, 3.0)
    ax.legend(loc="upper center", handlelength=1.4, borderpad=0.3, labelspacing=0.25)
    ax.set_title(r"each $\tau$ has its own asymmetric loss", fontsize=6, pad=3)
    _box(ax)
    panel_label(ax, "a")

    # ---- (b) one gene: 19 knots and ONE truth ----------------------------------------
    ax = axes[0, 1]
    ax.scatter(
        TAUS, knots, s=7, color=PLOT_PALETTE[2], zorder=3, label="19 predicted knots"
    )
    ax.scatter(
        [TAUS[MEDIAN_IDX]],
        [knots[MEDIAN_IDX]],
        s=28,
        marker="D",
        facecolor=PLOT_PALETTE[1],
        edgecolor="black",
        lw=0.5,
        zorder=5,
        label=r"median knot ($\tau$=0.50)",
    )
    ax.axhline(
        Y_TRUE,
        color="black",
        lw=0.8,
        ls="-",
        zorder=2,
        label=f"the one observed value $y$={Y_TRUE:.2f}",
    )
    ax.vlines(
        TAUS,
        np.minimum(knots, Y_TRUE),
        np.maximum(knots, Y_TRUE),
        color=PLOT_PALETTE_FILL[2],
        lw=0.6,
        zorder=1,
    )
    ax.set_xlabel(r"quantile level $\tau$")
    ax.set_ylabel(r"predicted $\log_2$ ratio")
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.legend(loc="upper left", handlelength=1.2, borderpad=0.2, labelspacing=0.25)
    ax.set_title("each knot has its own residual (synthetic gene)", fontsize=6, pad=3)
    _box(ax)
    panel_label(ax, "b")

    # ---- (c) what the loss is made of ------------------------------------------------
    ax = axes[0, 2]
    colors = [PLOT_PALETTE[2]] * N_QUANTILES
    colors[MEDIAN_IDX] = PLOT_PALETTE[1]
    ax.bar(TAUS, rho, width=0.035, color=colors, edgecolor="black", lw=0.4)
    ax.annotate(
        f"median knot: {med_share * 100:.1f}% of the\nloss, and the only one scored",
        xy=(TAUS[MEDIAN_IDX], rho[MEDIAN_IDX]),
        xytext=(0.03, 0.19),
        va="top",
        fontsize=6,
        arrowprops=dict(arrowstyle="->", lw=0.6, color="black"),
    )
    ax.set_xlabel(r"quantile level $\tau$")
    ax.set_ylabel(r"pinball contribution $\rho_\tau(u)$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.20)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.set_title(
        f"the other 18 carry {100 - med_share * 100:.1f}% of the gradient",
        fontsize=6,
        pad=3,
    )
    _box(ax)
    panel_label(ax, "c")

    # ---- (d) the knots read back as a CDF: intervals and PIT -------------------------
    ax = axes[1, 0]
    q_sorted = np.sort(knots)
    xs = np.linspace(q_sorted[0] - 0.35, q_sorted[-1] + 0.35, 400)
    cdf = quantile_pit(np.tile(q_sorted, (xs.size, 1)), xs, TAUS)
    pit_y = float(quantile_pit(q_sorted[None, :], np.array([Y_TRUE]), TAUS)[0])
    q10, q25, q75, q90 = np.interp([0.10, 0.25, 0.75, 0.90], TAUS, q_sorted)
    ax.plot(xs, cdf, color="black", lw=0.8, zorder=3, label="F (tails clamped)")
    ax.scatter(
        q_sorted,
        TAUS,
        s=9,
        facecolor="white",
        edgecolor="black",
        lw=0.4,
        zorder=4,
        label="knots",
    )
    ax.scatter(
        [q_sorted[MEDIAN_IDX]],
        [0.5],
        s=28,
        marker="D",
        facecolor=PLOT_PALETTE[1],
        edgecolor="black",
        lw=0.5,
        zorder=5,
        label="median knot",
    )
    # interval brackets: drop lines from the CDF at their tau, then a bar under the curve
    for lo, hi, level, y_bar, color, lw in (
        (q10, q90, 0.10, 0.05, PLOT_PALETTE[2], 1.2),
        (q25, q75, 0.25, 0.11, PLOT_PALETTE[8], 1.2),
    ):
        # drop line from the lower knot only; the upper one would cross the legend
        ax.vlines([lo], y_bar, [level], color=color, lw=0.5, ls=":", zorder=2)
        ax.errorbar(
            [(lo + hi) / 2],
            [y_bar],
            xerr=[[(hi - lo) / 2]],
            fmt="none",
            ecolor=color,
            elinewidth=lw,
            capsize=2,
            capthick=lw,
            zorder=3,
            label=f"{int(round((1 - 2 * level) * 100))}%: width {hi - lo:.2f}",
        )
    ax.plot(
        [Y_TRUE, Y_TRUE, xs[0]],
        [0, pit_y, pit_y],
        color=PLOT_PALETTE[1],
        lw=0.8,
        ls="--",
        zorder=4,
    )
    ax.scatter(
        [Y_TRUE],
        [pit_y],
        s=14,
        color=PLOT_PALETTE[1],
        edgecolor="black",
        lw=0.4,
        zorder=6,
    )
    ax.text(
        xs[0] + 0.06,
        pit_y + 0.04,
        f"PIT = {pit_y:.2f}",
        ha="left",
        va="center",
        fontsize=6,
        color=PLOT_PALETTE[1],
    )
    ax.annotate(
        f"$y$ = {Y_TRUE:.2f}",
        xy=(Y_TRUE, 0.19),
        xytext=(4, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=6,
        color=PLOT_PALETTE[1],
    )
    ax.set_xlabel(r"$\log_2$ ratio")
    ax.set_ylabel("predictive CDF F")
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(axis="y", which="minor", length=0)
    ax.grid(axis="y", which="both", lw=0.3, alpha=0.35)
    ax.legend(loc="upper left", handlelength=1.4, borderpad=0.2, labelspacing=0.25)
    ax.set_title("knots as a CDF: width = noise, F(y) = PIT", fontsize=6, pad=3)
    _box(ax)
    panel_label(ax, "d")

    # ---- (e) PIT histograms for three width ratios, simulated ------------------------
    ax = axes[1, 1]
    rng = np.random.default_rng(SEED)
    y_sim = rng.standard_normal(N_SIM)
    cases = (
        (1.0, "black", "calibrated"),
        (r_fit, PLOT_PALETTE[1], "too narrow"),
        (1.0 / r_fit, PLOT_PALETTE[0], "too wide"),
    )
    bins = np.linspace(0, 1, 21)
    rows = []
    for ratio, color, name in cases:
        q_sim = np.tile(norm.ppf(TAUS, loc=0.0, scale=ratio), (N_SIM, 1))
        pit = quantile_pit(q_sim, y_sim, TAUS)
        c50, c80 = coverage(pit, 0.5), coverage(pit, 0.8)
        ax.hist(pit, bins=bins, density=True, histtype="step", color=color, lw=1.0)
        rows.append((ratio, color, name, c50, c80))
        print(f"(e) r={ratio:.3f} {name}: coverage_50={c50:.3f} coverage_80={c80:.3f}")
    # aligned columns (axes coords): handle | width ratio r | case | cov50 | cov80
    cols = (0.10, 0.27, 0.57, 0.76)
    y0, dy = 0.945, 0.075
    ax.add_patch(
        Rectangle(
            (0.02, y0 - 3.5 * dy - 0.01),
            0.90,
            3.5 * dy + 0.05,
            transform=ax.transAxes,
            facecolor="white",
            edgecolor="black",
            lw=0.5,
            zorder=5,
        )
    )
    for x, head in zip(cols, ("r", "case", "cov50", "cov80")):
        ax.text(
            x,
            y0,
            head,
            transform=ax.transAxes,
            fontsize=6,
            ha="left",
            va="center",
            fontweight="bold",
            zorder=6,
        )
    for i, (ratio, color, name, c50, c80) in enumerate(rows):
        y = y0 - (i + 1) * dy
        ax.plot(
            [0.04, 0.08], [y, y], transform=ax.transAxes, color=color, lw=1.0, zorder=6
        )
        for x, txt in zip(cols, (f"{ratio:.2f}", name, f"{c50:.2f}", f"{c80:.2f}")):
            ax.text(
                x,
                y,
                txt,
                transform=ax.transAxes,
                fontsize=6,
                ha="left",
                va="center",
                color=color,
                zorder=6,
            )
    ax.set_xlabel("PIT = F(y), one value per simulated gene")
    ax.set_ylabel("density (Uniform(0,1) = 1)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 5.0)
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(axis="x", which="minor", length=0)
    ax.grid(axis="x", which="both", lw=0.3, alpha=0.35)
    ax.set_title("simulated genes: truth N(0,1), head N(0,r)", fontsize=6, pad=3)
    _box(ax)
    panel_label(ax, "e")

    # ---- (f) measured calibration, as a reliability curve ----------------------------
    ax = axes[1, 2]
    alpha = np.linspace(0.001, 0.999, 400)
    implied = 2 * norm.cdf(r_fit * norm.ppf(0.5 * (1 + alpha))) - 1
    ax.fill_between(
        [0, 1],
        [0, 1],
        [0, 0],
        color=PLOT_PALETTE_FILL[1],
        lw=0,
        zorder=0,
        label="overconfident: too narrow",
    )
    ax.plot(
        [0, 1], [0, 1], color="black", lw=0.8, ls="--", label="calibrated", zorder=2
    )
    ax.plot(
        alpha,
        implied,
        color=PLOT_PALETTE[1],
        lw=1.0,
        zorder=3,
        label=f"fit, width ratio r = {r_fit:.2f}",
    )
    ax.vlines(nominal, actual, nominal, color=PLOT_PALETTE[1], lw=0.8, zorder=3)
    ax.scatter(
        nominal,
        actual,
        s=28,
        marker="D",
        facecolor=PLOT_PALETTE[1],
        edgecolor="black",
        lw=0.5,
        zorder=5,
        label="measured, epoch 10,000",
    )
    for n, a in zip(nominal, actual):
        ax.annotate(
            f"{a:.3f}",
            xy=(n, a),
            xytext=(5, -3),
            textcoords="offset points",
            fontsize=6,
            ha="left",
            va="center",
        )
    ax.set_xlabel(r"nominal central interval $\alpha$")
    ax.set_ylabel("fraction of truths covered")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MultipleLocator(0.2))
        axis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which="minor", length=0)
    ax.grid(which="both", lw=0.3, alpha=0.35)
    ax.legend(loc="upper left", handlelength=1.4, borderpad=0.2, labelspacing=0.25)
    ax.set_title(f"measured: intervals {1 / r_fit:.2f}x too narrow", fontsize=6, pad=3)
    _box(ax)
    panel_label(ax, "f")

    fig.tight_layout(pad=0.4)
    out_dir = os.environ["ASSET_IMAGES_DIR"]
    stem = osp.join(out_dir, "019-simb-multimodal", "pinball_vs_mse")
    os.makedirs(osp.dirname(stem), exist_ok=True)
    fig.savefig(f"{stem}.png", dpi=300)
    savefig_true_size_svg(fig, f"{stem}.svg")
    print(f"wrote {stem}.png")
    print(f"total pinball (mean over {N_QUANTILES}) = {rho.mean():.4f}")
    print(
        f"median knot share {med_share * 100:.1f}%, other 18 = {100 - med_share * 100:.1f}%"
    )
    print(
        f"(d) PIT of y = {pit_y:.3f}; 80% width {q90 - q10:.3f}; 50% width {q75 - q25:.3f}"
    )
    print(f"(f) back-solved width ratios {ratios.round(4)}, mean {r_fit:.4f}")


if __name__ == "__main__":
    main()
