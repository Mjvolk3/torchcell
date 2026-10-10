# experiments/041-scaling-laws/scripts/scaling_law_forms.py
# [[experiments.041-scaling-laws.scripts.scaling_law_forms]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/041-scaling-laws/scripts/scaling_law_forms
"""The two published scaling-law forms, drawn so they can be read and taught.

Nothing here is measured on a torchcell model. The panels draw the functional forms
of Kaplan et al. 2020 (arXiv:2001.08361) and Hoffmann et al. 2022 (arXiv:2203.15556)
at the constants those papers report, plus the Besiroglu et al. 2024 refit
(arXiv:2404.10102), so the shapes a real ablation would have to reproduce are on one
page before any run exists. Panel i is a SYNTHETIC fitting exercise: runs are drawn
from the Hoffmann form with multiplicative noise, the largest sizes are held out, and
a pure power law and a floored power law are fit on the rest and extrapolated, with a
bootstrap over runs. It demonstrates the fitting recipe, not a result.

Figure 1 (3 x 3, full width)
    a  Kaplan loss vs parameters, L = (N_c / N)^alpha_N, three exponents
    b  Kaplan loss vs data, L = (D_c / D)^alpha_D, three exponents
    c  the irreducible floor: E + A N^-alpha bends on log-log, L - E does not
    d  the joint form L(N, D) as a surface, with iso-compute lines and the
       compute-optimal frontier
    e  L vs N at fixed D: every curve plateaus at E + B D^-beta
    f  L vs D at fixed N: every curve plateaus at E + A N^-alpha
    g  IsoFLOP curves: L vs N at fixed C with D = C / (6 N), minima marked
    h  the compute-optimal allocation N*(C) and D*(C) under two fits
    i  fit on small runs, hold out the largest, extrapolate, bootstrap

GIFs (one per panel of Figure 1, one sweep each, for reading the forms rather than
for print; every frame's title carries the equation, the constants and their source,
the swept value, and a one-sentence takeaway; typeset by real LaTeX)
    scaling_gif_a_exponent_sweep.gif       Kaplan L(N) pinned at the center of the
                                           fitted range while alpha_N moves by +-0.02
    scaling_gif_b_data_exponent_sweep.gif  the same on the data axis, alpha_D
    scaling_gif_c_floor_sweep.gif          E rises under a fixed A N^-alpha: the bend
    scaling_gif_d_surface_sweep.gif        the joint surface with one budget line
                                           sliding across it
    scaling_gif_e_data_sweep.gif           L vs N while D grows: the plateau drops
    scaling_gif_f_params_sweep.gif         L vs D while N grows
    scaling_gif_g_compute_sweep.gif        the IsoFLOP curve while C grows; the
                                           minimum traces the frontier
    scaling_gif_h_allocation_sweep.gif     beta moves, a and b move with it
    scaling_gif_i_bootstrap.gif            one bootstrap resample and refit per frame

Writes
    $ASSET_IMAGES_DIR/041-scaling-laws/scaling_law_forms.{svg,png}
    $ASSET_IMAGES_DIR/041-scaling-laws/scaling_gif_<panel>_*.gif  (nine)
    experiments/041-scaling-laws/results/scaling_law_forms_summary.json
    notes-tex/modeling/scaling-laws/tables/t1-constants.tex

Run from the repo root:
    python experiments/041-scaling-laws/scripts/scaling_law_forms.py
    python experiments/041-scaling-laws/scripts/scaling_law_forms.py --no-gifs
"""

import argparse
import json
import os
import os.path as osp
import subprocess
import tempfile
from typing import Any

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from PIL import Image  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from torchcell.utils import (  # noqa: E402
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    apply_paper_style,
    mm_to_in,
    panel_label,
    savefig_true_size_svg,
)

load_dotenv()
FArray = NDArray[np.floating[Any]]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
REPO_ROOT = osp.dirname(EXPERIMENT_ROOT)
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "041-scaling-laws", "results")
IMAGES_DIR = osp.join(ASSET_IMAGES_DIR, "041-scaling-laws")
TABLES_DIR = osp.join(REPO_ROOT, "notes-tex", "modeling", "scaling-laws", "tables")

AMBER, BRICK, LILAC, WHEAT, STEEL, GRAY = PLOT_PALETTE[:6]
CHARCOAL = PLOT_PALETTE[11]
FLOPS_PER_PARAM_PER_EXAMPLE = 6.0  # transformer forward (2) + backward (4)


# --- the two forms, as typed records ---------------------------------------------
class KaplanFit(BaseModel):
    """L(N) = (N_c / N)^alpha_N and L(D) = (D_c / D)^alpha_D, each axis alone."""

    source: str
    N_c: float
    alpha_N: float
    D_c: float
    alpha_D: float

    def loss_N(self, N: FArray, alpha: float | None = None) -> FArray:
        """Loss against parameters, at the published exponent unless one is given."""
        a = self.alpha_N if alpha is None else alpha
        return (self.N_c / N) ** a

    def loss_D(self, D: FArray, alpha: float | None = None) -> FArray:
        """Loss against data, at the published exponent unless one is given."""
        a = self.alpha_D if alpha is None else alpha
        return (self.D_c / D) ** a


class ChinchillaFit(BaseModel):
    """L(N, D) = E + A / N^alpha + B / D^beta, with its compute-optimal allocation."""

    source: str
    E: float
    A: float
    B: float
    alpha: float
    beta: float

    def loss(self, N: FArray | float, D: FArray | float) -> FArray:
        """The joint form evaluated at (N, D)."""
        return (
            self.E
            + self.A * np.power(N, -self.alpha)
            + self.B * np.power(D, -self.beta)
        )

    @property
    def a(self) -> float:
        """Exponent of N* in C: N* grows as C^a."""
        return self.beta / (self.alpha + self.beta)

    @property
    def b(self) -> float:
        """Exponent of D* in C: D* grows as C^b, and a + b = 1."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def G(self) -> float:
        """Prefactor of the optimum, N* = G (C / 6)^(beta / (alpha + beta))."""
        return float(
            ((self.alpha * self.A) / (self.beta * self.B))
            ** (1.0 / (self.alpha + self.beta))
        )

    def N_opt(self, C: FArray | float) -> FArray:
        """Compute-optimal parameter count for a budget C under C = 6 N D."""
        return self.G * np.power(np.asarray(C) / FLOPS_PER_PARAM_PER_EXAMPLE, self.a)

    def D_opt(self, C: FArray | float) -> FArray:
        """Compute-optimal data for a budget C, the budget's remainder after N*."""
        D: FArray = np.asarray(C) / (FLOPS_PER_PARAM_PER_EXAMPLE * self.N_opt(C))
        return D

    def isoflop(self, N: FArray, C: float | FArray) -> FArray:
        """Loss along one budget line: N free, D = C / (6 N)."""
        return self.loss(N, C / (FLOPS_PER_PARAM_PER_EXAMPLE * N))


# Published constants. Every value is a fact from outside our holdings (none of the
# three papers is in the mirror yet) and the document flags them as such.
KAPLAN = KaplanFit(
    source="Kaplan et al. 2020, arXiv:2001.08361, Table 1 / eqs. 1.1 and 1.2",
    N_c=8.8e13,
    alpha_N=0.076,
    D_c=5.4e13,
    alpha_D=0.095,
)
HOFFMANN = ChinchillaFit(
    source="Hoffmann et al. 2022, arXiv:2203.15556, Approach 3 (eq. 10 fit)",
    E=1.69,
    A=406.4,
    B=410.7,
    alpha=0.34,
    beta=0.28,
)
BESIROGLU = ChinchillaFit(
    source="Besiroglu et al. 2024, arXiv:2404.10102, refit of the Approach 3 data",
    E=1.8172,
    A=482.01,
    B=2085.43,
    alpha=0.3478,
    beta=0.3658,
)


# --- the synthetic fitting exercise of panel i --------------------------------------
class SyntheticRuns(BaseModel):
    """Runs drawn from a known floored power law, so the recipe can be checked."""

    truth: ChinchillaFit
    sizes: list[float]
    seeds_per_size: int
    noise_sigma: float
    n_fit_sizes: int
    rng_seed: int


class FitResult(BaseModel):
    """One fitted form of panel i: its exponent, bootstrap interval, floor and the
    held-out predictions beside the observed held-out means.
    """

    form: str
    alpha: float
    alpha_ci90: tuple[float, float]
    E: float | None
    predicted_heldout: list[float]
    observed_heldout_mean: list[float]


SYNTH = SyntheticRuns(
    truth=HOFFMANN,
    sizes=list(np.logspace(6, 10, 7)),
    seeds_per_size=3,
    noise_sigma=0.02,
    n_fit_sizes=5,
    rng_seed=41,
)


def draw_runs(spec: SyntheticRuns) -> tuple[FArray, FArray]:
    rng = np.random.default_rng(spec.rng_seed)
    N = np.repeat(np.asarray(spec.sizes), spec.seeds_per_size)
    # Data is not limiting in this exercise: D -> infinity, so only E + A N^-alpha.
    truth = spec.truth.E + spec.truth.A * N ** (-spec.truth.alpha)
    L = truth * np.exp(rng.normal(0.0, spec.noise_sigma, size=N.shape))
    return N, L


def fit_power_law(N: FArray, L: FArray) -> tuple[float, float]:
    """Log L = c - alpha log N by least squares. Returns (A, alpha)."""
    slope, intercept = np.polyfit(np.log(N), np.log(L), 1)
    return float(np.exp(intercept)), float(-slope)


def fit_floored(N: FArray, L: FArray) -> tuple[float, float, float]:
    """L = E + A N^-alpha by nonlinear least squares in log space, best of a grid of
    starts, because the objective is not convex. Returns (E, A, alpha).
    """
    logN, logL = np.log(N), np.log(L)

    def objective(p: FArray) -> float:
        log_a, log_e, alpha = p
        pred = np.logaddexp(log_a - alpha * logN, np.full_like(logN, log_e))
        r = pred - logL
        return float(np.sum(r * r))

    starts = [
        np.array([la, le, al])
        for la in np.log([10.0, 100.0, 1000.0])
        for le in np.log([0.5, 1.0, 2.0])
        for al in [0.2, 0.35, 0.5]
    ]
    best = min(
        (
            minimize(
                objective,
                x0,
                method="L-BFGS-B",
                bounds=[(-5, 15), (-5, 5), (0.01, 2.0)],
            )
            for x0 in starts
        ),
        key=lambda r: r.fun,
    )
    log_a, log_e, alpha = best.x
    return float(np.exp(log_e)), float(np.exp(log_a)), float(alpha)


def ci90(samples: FArray) -> tuple[float, float]:
    """The 5th and 95th percentiles of a bootstrap sample."""
    lo, hi = np.percentile(samples, [5, 95])
    return float(lo), float(hi)


def bootstrap_fits(
    N: FArray, L: FArray, grid: FArray, n_boot: int, rng: np.random.Generator
) -> dict[str, FArray]:
    """Resample RUNS with replacement, refit both forms, return prediction bands and
    the sampled exponents.
    """
    pl_pred = np.empty((n_boot, grid.size))
    fl_pred = np.empty((n_boot, grid.size))
    pl_alpha = np.empty(n_boot)
    fl_alpha = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, N.size, size=N.size)
        A, al = fit_power_law(N[idx], L[idx])
        pl_pred[i] = A * grid ** (-al)
        pl_alpha[i] = al
        E, A2, al2 = fit_floored(N[idx], L[idx])
        fl_pred[i] = E + A2 * grid ** (-al2)
        fl_alpha[i] = al2
    return {
        "pl_pred": pl_pred,
        "fl_pred": fl_pred,
        "pl_alpha": pl_alpha,
        "fl_alpha": fl_alpha,
    }


# --- panel helpers -------------------------------------------------------------------
def box(ax: Axes) -> None:
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)
    ax.tick_params(width=0.5, length=2, pad=1.5)


def legend(ax: Axes, **kw: object) -> None:
    kw.setdefault("fontsize", 5)
    kw.setdefault("handlelength", 1.4)
    kw.setdefault("borderpad", 0.3)
    kw.setdefault("labelspacing", 0.25)
    ax.legend(**kw)


def plain_log_y(ax: Axes, ticks: list[float]) -> None:
    """A log y-axis spanning under a decade reads as '2 x 10^0' by default; label the
    ticks as plain numbers and drop the minor labels.
    """
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.yaxis.set_minor_formatter(NullFormatter())


def pow_label(v: float) -> str:
    e = int(np.round(np.log10(v)))
    return rf"$10^{{{e}}}$"


# --- figure 1 --------------------------------------------------------------------------
def panel_a(ax: Axes) -> None:
    N = np.logspace(5, 13, 200)
    for al, c in zip([0.05, KAPLAN.alpha_N, 0.15], [LILAC, AMBER, BRICK], strict=True):
        tag = " (published)" if al == KAPLAN.alpha_N else ""
        ax.loglog(
            N,
            KAPLAN.loss_N(N, al),
            color=c,
            lw=0.8,
            label=rf"$\alpha_N$ = {al:.3g}{tag}",
        )
    ax.set_xlabel("parameters N (log scale)")
    ax.set_ylabel("loss L (log scale)")
    ax.set_title(r"Kaplan: $L = (N_c / N)^{\alpha_N}$, data not limiting")
    legend(ax, loc="upper right")


def panel_b(ax: Axes) -> None:
    D = np.logspace(6, 13, 200)
    for al, c in zip([0.06, KAPLAN.alpha_D, 0.14], [LILAC, AMBER, BRICK], strict=True):
        tag = " (published)" if al == KAPLAN.alpha_D else ""
        ax.loglog(
            D,
            KAPLAN.loss_D(D, al),
            color=c,
            lw=0.8,
            label=rf"$\alpha_D$ = {al:.3g}{tag}",
        )
    ax.set_xlabel("training examples D (log scale)")
    ax.set_ylabel("loss L (log scale)")
    ax.set_title(r"Kaplan: $L = (D_c / D)^{\alpha_D}$, model not limiting")
    ax.set_ylim(top=40)
    legend(ax, loc="upper right")


def panel_c(ax: Axes) -> None:
    f = HOFFMANN
    N = np.logspace(6, 12, 200)
    term = f.A * N ** (-f.alpha)
    ax.loglog(N, term + f.E, color=AMBER, lw=0.8, label=r"$E + A\,N^{-\alpha}$ (bends)")
    ax.loglog(
        N,
        term,
        color=BRICK,
        lw=0.8,
        ls="--",
        label=r"$L - E = A\,N^{-\alpha}$ (straight)",
    )
    ax.axhline(f.E, color=GRAY, lw=0.6, ls=":")
    ax.text(8e11, f.E * 0.8, f"E = {f.E}", fontsize=5, color=GRAY, va="top", ha="right")
    ax.set_xlabel("parameters N (log scale)")
    ax.set_ylabel("loss L (log scale)")
    ax.set_title("the floor: curvature on log-log means E is near")
    legend(ax, loc="lower left")


def panel_d(ax: Axes, fig: Figure) -> None:
    f = HOFFMANN
    logN = np.linspace(6, 12, 240)
    logD = np.linspace(7, 13, 240)
    NN, DD = np.meshgrid(10**logN, 10**logD)
    L = f.loss(NN, DD)
    cmap = LinearSegmentedColormap.from_list("tc_gray", ["#FFFFFF", CHARCOAL])
    levels = np.linspace(1.7, 7.0, 16)
    cf = ax.contourf(logN, logD, L, levels=levels, cmap=cmap, extend="max")
    ax.contour(logN, logD, L, levels=levels, colors="white", linewidths=0.3)
    ax.set_xlim(6, 12)
    ax.set_ylim(7, 13)
    p0, p1 = ax.transData.transform([(6.0, 13.0), (7.0, 12.0)])
    rot = float(np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0])))
    for C in [1e18, 1e20, 1e22, 1e24]:
        k = np.log10(C / FLOPS_PER_PARAM_PER_EXAMPLE)
        ax.plot(logN, k - logN, color=AMBER, lw=0.7)
        # label just inside the edge the line LEAVES through, on its upper side:
        # the right edge when it reaches it, otherwise the bottom edge. The frontier
        # runs top-right, so the labels never meet it.
        if k - 11.9 >= 7.1:
            ax.text(
                11.9 - 0.1,
                k - 11.9 + 0.1,
                f"C = {pow_label(C)}",
                fontsize=5,
                color=AMBER,
                ha="right",
                va="bottom",
                rotation=rot,
                rotation_mode="anchor",
            )
        else:
            ax.text(
                k - 7.1 - 0.1,
                7.1 + 0.1,
                f"C = {pow_label(C)}",
                fontsize=5,
                color=AMBER,
                ha="right",
                va="bottom",
                rotation=rot,
                rotation_mode="anchor",
            )
    budgets = np.logspace(17, 26, 50)
    ax.plot(np.log10(f.N_opt(budgets)), np.log10(f.D_opt(budgets)), color=BRICK, lw=1.0)
    p0, p1 = ax.transData.transform(
        [
            (np.log10(f.N_opt(1e22)), np.log10(f.D_opt(1e22))),
            (np.log10(f.N_opt(1e24)), np.log10(f.D_opt(1e24))),
        ]
    )
    ax.text(
        float(np.log10(f.N_opt(1e21))) - 0.12,
        float(np.log10(f.D_opt(1e21))) + 0.12,
        "frontier",
        fontsize=5,
        color=BRICK,
        ha="center",
        va="bottom",
        rotation=float(np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))),
        rotation_mode="anchor",
    )
    ax.set_xlim(6, 12)
    ax.set_ylim(7, 13)
    ax.set_xlabel(r"$\log_{10}$ parameters N")
    ax.set_ylabel(r"$\log_{10}$ training examples D")
    ax.set_title(r"Chinchilla: $L(N, D) = E + A N^{-\alpha} + B D^{-\beta}$")
    cb = fig.colorbar(cf, ax=ax, fraction=0.045, pad=0.02)
    cb.ax.set_title("L", fontsize=6, pad=2)
    cb.set_ticks([2, 3, 4, 5, 6, 7])
    cb.ax.tick_params(labelsize=5, width=0.5, length=2)
    cb.outline.set_linewidth(0.5)  # type: ignore[operator]  # stub types outline as a Spine callable


def panel_e(ax: Axes) -> None:
    f = HOFFMANN
    N = np.logspace(6, 12, 200)
    for D, c in zip([1e8, 1e9, 1e10, 1e11], [AMBER, BRICK, LILAC, WHEAT], strict=True):
        ax.loglog(N, f.loss(N, D), color=c, lw=0.8, label=f"D = {pow_label(D)}")
        ax.axhline(f.E + f.B * D ** (-f.beta), color=c, lw=0.4, ls=":")
    ax.loglog(
        N, f.E + f.A * N ** (-f.alpha), color=STEEL, lw=0.8, label=r"D $\to \infty$"
    )
    ax.axhline(f.E, color=GRAY, lw=0.6, ls=":")
    ax.set_xlabel("parameters N (log scale)")
    ax.set_ylabel("loss L (log scale)")
    ax.set_title(r"L vs N at fixed D: plateau at $E + B D^{-\beta}$")
    ax.set_ylim(1.0, 8.0)
    plain_log_y(ax, [1, 2, 3, 4, 5, 6, 7])
    legend(ax, loc="lower center", ncol=3, columnspacing=0.8)


def panel_f(ax: Axes) -> None:
    f = HOFFMANN
    D = np.logspace(7, 13, 200)
    for N, c in zip([1e7, 1e8, 1e9, 1e10], [AMBER, BRICK, LILAC, WHEAT], strict=True):
        ax.loglog(D, f.loss(N, D), color=c, lw=0.8, label=f"N = {pow_label(N)}")
        ax.axhline(f.E + f.A * N ** (-f.alpha), color=c, lw=0.4, ls=":")
    ax.loglog(
        D, f.E + f.B * D ** (-f.beta), color=STEEL, lw=0.8, label=r"N $\to \infty$"
    )
    ax.axhline(f.E, color=GRAY, lw=0.6, ls=":")
    ax.set_xlabel("training examples D (log scale)")
    ax.set_ylabel("loss L (log scale)")
    ax.set_title(r"L vs D at fixed N: plateau at $E + A N^{-\alpha}$")
    ax.set_ylim(1.0, 8.0)
    plain_log_y(ax, [1, 2, 3, 4, 5, 6, 7])
    legend(ax, loc="lower center", ncol=3, columnspacing=0.8)


def panel_g(ax: Axes) -> dict[str, float]:
    f = HOFFMANN
    N = np.logspace(6.5, 11.5, 300)
    optima: dict[str, float] = {}
    for C, c in zip(
        [1e19, 1e20, 1e21, 1e22], [AMBER, BRICK, LILAC, WHEAT], strict=True
    ):
        L = f.isoflop(N, C)
        ax.loglog(N, L, color=c, lw=0.8, label=f"C = {pow_label(C)} FLOPs")
        n_star = float(f.N_opt(C))
        ax.plot(
            n_star,
            f.isoflop(np.array([n_star]), C),
            marker="o",
            ms=3,
            color=c,
            mec="black",
            mew=0.4,
        )
        optima[f"C={C:.0e}"] = n_star
    ax.plot(
        f.N_opt(np.logspace(18, 23, 40)),
        f.isoflop(f.N_opt(np.logspace(18, 23, 40)), np.logspace(18, 23, 40)),
        color=GRAY,
        lw=0.6,
        ls="--",
        label="minima trace the frontier",
    )
    ax.set_xlabel("parameters N (log scale), D = C / 6N")
    ax.set_ylabel("loss L (log scale)")
    ax.set_title("IsoFLOP curves: a U per budget, its minimum is N*")
    ax.set_ylim(2.0, 7.0)
    plain_log_y(ax, [2, 3, 4, 5, 6, 7])
    legend(ax, loc="upper left")
    return optima


def panel_h(ax: Axes) -> None:
    C = np.logspace(17, 25, 100)
    for f, ls, tag in [
        (HOFFMANN, "-", "Hoffmann 2022"),
        (BESIROGLU, "--", "Besiroglu 2024 refit"),
    ]:
        ax.loglog(
            C,
            f.N_opt(C),
            color=AMBER,
            lw=0.8,
            ls=ls,
            label=rf"$N^*$, slope $a$ = {f.a:.2f}, {tag}",
        )
        ax.loglog(
            C,
            f.D_opt(C),
            color=BRICK,
            lw=0.8,
            ls=ls,
            label=rf"$D^*$, slope $b$ = {f.b:.2f}, {tag}",
        )
    ax.set_xlabel("compute C (FLOPs, log scale)")
    ax.set_ylabel("optimal size N*, optimal data D* (log scale)")
    ax.set_title("allocation: how N* and D* grow with C (a + b = 1)")
    ax.set_ylim(1e7, 1e18)
    legend(ax, loc="upper right")


def panel_i(ax: Axes, rng: np.random.Generator) -> list[FitResult]:
    N, L = draw_runs(SYNTH)
    n_fit = SYNTH.n_fit_sizes * SYNTH.seeds_per_size
    Nf, Lf = N[:n_fit], L[:n_fit]
    Nh, Lh = N[n_fit:], L[n_fit:]
    grid = np.logspace(5.8, 10.3, 200)
    A_pl, al_pl = fit_power_law(Nf, Lf)
    E_fl, A_fl, al_fl = fit_floored(Nf, Lf)
    boot = bootstrap_fits(Nf, Lf, grid, n_boot=300, rng=rng)

    ax.axvspan(grid[0], Nf.max() * 1.6, color="#F2F2F2", lw=0, zorder=0)
    ax.text(grid[0] * 1.3, 1.56, "fitted sizes", fontsize=5, color=GRAY, va="bottom")
    ax.text(
        grid[-1] * 0.8,
        1.56,
        "held out",
        fontsize=5,
        color=GRAY,
        va="bottom",
        ha="right",
    )
    ax.fill_between(
        grid,
        *np.percentile(boot["pl_pred"], [5, 95], axis=0),
        color=LILAC,
        alpha=0.25,
        lw=0,
    )
    ax.fill_between(
        grid,
        *np.percentile(boot["fl_pred"], [5, 95], axis=0),
        color=WHEAT,
        alpha=0.35,
        lw=0,
    )
    ax.loglog(
        grid,
        A_pl * grid ** (-al_pl),
        color=LILAC,
        lw=0.8,
        label=rf"power law, $\alpha$ = {al_pl:.2f}",
    )
    ax.loglog(
        grid,
        E_fl + A_fl * grid ** (-al_fl),
        color=WHEAT,
        lw=0.8,
        label=rf"with floor, $\alpha$ = {al_fl:.2f}, E = {E_fl:.2f}",
    )
    truth = SYNTH.truth
    ax.loglog(
        grid,
        truth.E + truth.A * grid ** (-truth.alpha),
        color=CHARCOAL,
        lw=0.6,
        ls="--",
        label=rf"generating curve, $\alpha$ = {truth.alpha}, E = {truth.E}",
    )
    ax.plot(
        Nf,
        Lf,
        "o",
        ms=2.5,
        color=AMBER,
        mec="black",
        mew=0.3,
        label="fitted runs (3 seeds)",
    )
    ax.plot(Nh, Lh, "o", ms=2.5, mfc="white", mec=BRICK, mew=0.6, label="held-out runs")
    ax.set_xlabel("parameters N (log scale)")
    ax.set_ylabel("loss L (log scale)")
    ax.set_title("fit small, hold out large, extrapolate (synthetic)")
    ax.set_ylim(1.5, 7)
    plain_log_y(ax, [2, 3, 4, 5, 6, 7])
    legend(ax, loc="upper right")

    held_sizes = np.asarray(SYNTH.sizes[SYNTH.n_fit_sizes :])
    obs = [float(Lh[Nh == s].mean()) for s in held_sizes]
    return [
        FitResult(
            form="power_law",
            alpha=al_pl,
            alpha_ci90=ci90(boot["pl_alpha"]),
            E=None,
            predicted_heldout=[float(A_pl * s ** (-al_pl)) for s in held_sizes],
            observed_heldout_mean=obs,
        ),
        FitResult(
            form="floored_power_law",
            alpha=al_fl,
            alpha_ci90=ci90(boot["fl_alpha"]),
            E=E_fl,
            predicted_heldout=[float(E_fl + A_fl * s ** (-al_fl)) for s in held_sizes],
            observed_heldout_mean=obs,
        ),
    ]


def figure_1() -> dict[str, object]:
    fig, axes = plt.subplots(
        3, 3, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(150))
    )
    fig.subplots_adjust(
        left=0.055, right=0.985, bottom=0.05, top=0.955, wspace=0.34, hspace=0.46
    )
    ax = axes.ravel()
    rng = np.random.default_rng(SYNTH.rng_seed + 1)
    panel_a(ax[0])
    panel_b(ax[1])
    panel_c(ax[2])
    panel_d(ax[3], fig)
    panel_e(ax[4])
    panel_f(ax[5])
    optima = panel_g(ax[6])
    panel_h(ax[7])
    fits = panel_i(ax[8], rng)
    for a in ax:
        box(a)
    fig.canvas.draw()
    for a, letter in zip(ax, "abcdefghi", strict=True):
        panel_label(a, letter)
    stem = osp.join(IMAGES_DIR, "scaling_law_forms")
    savefig_true_size_svg(fig, stem + ".svg")
    fig.savefig(stem + ".png", dpi=300)
    plt.close(fig)
    print(f"wrote {stem}.svg and .png")
    return {
        "isoflop_optima_N": optima,
        "synthetic_fits": [f.model_dump() for f in fits],
    }


# --- GIFs ---------------------------------------------------------------------------------
# Every GIF is one sweep of one quantity. The title carries what a reader needs: line 1 is
# the equation and the constants it is drawn at, line 2 is the swept value and what it
# implies, in fixed-width numbers so nothing jumps. Legends name every line, dotted ones
# included, and hold the same entries in the same place on every frame. No text is drawn
# inside the axes, so nothing can be crossed by a line.
HOFF = (
    r"constants from Hoffmann 2022 (Approach 3): $E$ = 1.69, $A$ = 406.4, $B$ = 410.7, "
    r"$\alpha$ = 0.34, $\beta$ = 0.28"
)
JOINT = r"$L(N, D) = E + A\,N^{-\alpha} + B\,D^{-\beta}$"


def frames_to_gif(frames: list[FArray], path: str, duration_ms: int = 120) -> None:
    shapes = {f.shape for f in frames}
    if len(shapes) != 1:
        raise ValueError(f"frames of {path} differ in shape: {shapes}")
    # Hold the last frame so the loop reads as a sweep, then a pause, then a restart.
    frames = frames + [frames[-1]] * 8
    # Quantize to the GIF palette WITHOUT dithering. The default Floyd-Steinberg dither
    # scatters the antialiased edges of small text into speckle, which reads as
    # distortion; an adaptive 256-color palette with no dither keeps the glyph edges
    # smooth, since a plot is a few flat colors plus antialiasing ramps.
    images = [
        Image.fromarray(f).quantize(
            colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE
        )
        for f in frames
    ]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        # Lossless: trims each frame's palette to the colors it uses, which with the
        # palette above cuts the file to about 40%.
        optimize=True,
    )
    print(f"wrote {path} ({len(frames)} frames)")


GIF_DPI = 200
# The GIF frames are typeset by real LaTeX (Computer Modern, amsmath), not mathtext.
# Agg cannot rasterize usetex text without dvipng, which this machine lacks, so a frame
# is written as a PDF (the PDF backend reads the DVI itself) and rasterized with
# pdftoppm. type1cm.sty, which matplotlib's usetex preamble loads, is installed in the
# user texmf tree (~/texmf/tex/latex/type1cm/) because the system TeX Live omits it.
GIF_RC: dict[str, object] = {
    "text.usetex": True,
    # Latin Modern rather than the bluesky Computer Modern Type 1 fonts: same design,
    # heavier and better hinted at small sizes, so rasterized text stays even.
    "text.latex.preamble": r"\usepackage{lmodern}\usepackage{amsmath}\usepackage{amssymb}",
    "font.family": "serif",
    "font.size": 8.0,
    "axes.labelsize": 8.0,
    "axes.titlesize": 8.0,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.0,
    "mathtext.fontset": "cm",
}


def rasterize(fig: Figure) -> FArray:
    """One GIF frame: the figure through the PDF backend and pdftoppm, as RGB."""
    with tempfile.TemporaryDirectory() as tmp:
        pdf = osp.join(tmp, "frame.pdf")
        fig.savefig(pdf, format="pdf")
        subprocess.run(
            [
                "pdftoppm",
                "-r",
                str(GIF_DPI),
                "-png",
                "-singlefile",
                pdf,
                osp.join(tmp, "frame"),
            ],
            check=True,
        )
        frame: FArray = np.asarray(iio.imread(osp.join(tmp, "frame.png")))[
            ..., :3
        ].copy()
    return frame


def gif_figure(*lines: str, takeaway: str) -> tuple[Figure, Axes]:
    """A GIF frame: a wide canvas with a four-line title. Line 1 is the equation and
    what is held fixed, line 2 the constants and their source, line 3 the swept value
    and what it implies, in fixed-width numbers so the text does not jump, and the
    last line, in italics, is the one-sentence takeaway of the whole sweep.
    """
    fig, ax = plt.subplots(figsize=(mm_to_in(170), mm_to_in(105)), dpi=GIF_DPI)
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.10, top=0.80)
    title = "\n".join([*lines, r"\textit{" + takeaway + "}"])
    ax.set_title(title, fontsize=8, linespacing=1.6)
    box(ax)
    return fig, ax


def loss_axis(ax: Axes, lo: float, hi: float, ticks: list[float]) -> None:
    ax.set_ylim(lo, hi)
    plain_log_y(ax, ticks)
    ax.set_ylabel(r"loss $L$ (log scale)")


def gif_data_sweep(path: str) -> None:
    f = HOFFMANN
    N = np.logspace(6, 12, 200)
    frames = []
    for logD in np.linspace(7.5, 12.5, 36):
        D = 10**logD
        plateau = f.E + f.B * D ** (-f.beta)
        fig, ax = gif_figure(
            JOINT + r" against $N$ at a fixed $D$",
            HOFF,
            rf"$D = 10^{{{logD:4.1f}}}$ examples:  plateau $E + B D^{{-\beta}}$ = {plateau:4.2f};"
            r"  curve minus plateau is what more parameters can still buy",
            takeaway="Parameters stop paying at a plateau the data sets; only more data lowers it.",
        )
        ax.loglog(N, f.loss(N, D), color=AMBER, lw=1.2, label=r"$L(N, D)$ at this $D$")
        ax.loglog(
            N,
            f.E + f.A * N ** (-f.alpha),
            color=STEEL,
            lw=0.8,
            label=r"$L(N, \infty)$: data not limiting",
        )
        ax.axhline(
            plateau, color=AMBER, lw=0.6, ls=":", label=r"plateau $E + B D^{-\beta}$"
        )
        ax.axhline(f.E, color=GRAY, lw=0.6, ls=":", label=r"floor $E$ = 1.69")
        ax.set_xlim(1e6, 1e12)
        loss_axis(ax, 1.5, 8, [2, 3, 4, 5, 6, 7, 8])
        ax.set_xlabel(r"parameters $N$ (log scale)")
        legend(ax, loc="upper right")
        frames.append(rasterize(fig))
        plt.close(fig)
    frames_to_gif(frames, path)


def gif_params_sweep(path: str) -> None:
    f = HOFFMANN
    D = np.logspace(7, 13, 200)
    frames = []
    for logN in np.linspace(6.5, 11.5, 36):
        N = 10**logN
        plateau = f.E + f.A * N ** (-f.alpha)
        fig, ax = gif_figure(
            JOINT + r" against $D$ at a fixed $N$",
            HOFF,
            rf"$N = 10^{{{logN:4.1f}}}$ parameters:  plateau $E + A N^{{-\alpha}}$ = {plateau:4.2f};"
            r"  curve minus plateau is what more data can still buy",
            takeaway="Data stops paying at a plateau the model size sets; only a bigger model lowers it.",
        )
        ax.loglog(D, f.loss(N, D), color=BRICK, lw=1.2, label=r"$L(N, D)$ at this $N$")
        ax.loglog(
            D,
            f.E + f.B * D ** (-f.beta),
            color=STEEL,
            lw=0.8,
            label=r"$L(\infty, D)$: model not limiting",
        )
        ax.axhline(
            plateau, color=BRICK, lw=0.6, ls=":", label=r"plateau $E + A N^{-\alpha}$"
        )
        ax.axhline(f.E, color=GRAY, lw=0.6, ls=":", label=r"floor $E$ = 1.69")
        ax.set_xlim(1e7, 1e13)
        loss_axis(ax, 1.5, 8, [2, 3, 4, 5, 6, 7, 8])
        ax.set_xlabel(r"training examples $D$ (log scale)")
        legend(ax, loc="upper right")
        frames.append(rasterize(fig))
        plt.close(fig)
    frames_to_gif(frames, path)


def gif_compute_sweep(path: str) -> None:
    f = HOFFMANN
    N = np.logspace(6.5, 11.5, 300)
    frames = []
    trail_N: list[float] = []
    trail_L: list[float] = []
    for logC in np.linspace(18, 23, 36):
        C = 10**logC
        n_star = float(f.N_opt(C))
        d_star = float(f.D_opt(C))
        l_star = float(f.isoflop(np.array([n_star]), C)[0])
        trail_N.append(n_star)
        trail_L.append(l_star)
        fig, ax = gif_figure(
            r"IsoFLOP curve: "
            + JOINT
            + r" along one budget line $6ND = C$, so $D = C / 6N$",
            HOFF,
            rf"$C = 10^{{{logC:4.1f}}}$ FLOPs:  $N^* = 10^{{{np.log10(n_star):4.1f}}}$, "
            rf"$D^* = 10^{{{np.log10(d_star):4.1f}}}$, $L^*$ = {l_star:4.2f};"
            rf"  minimum at $N^* = G\,(C/6)^{{\beta/(\alpha+\beta)}}$, exponent {f.a:.2f}",
            takeaway="Every budget has one best model size, and the best sizes line up on the frontier.",
        )
        ax.loglog(
            N, f.isoflop(N, C), color=LILAC, lw=1.2, label=r"$L$ along the budget line"
        )
        ax.plot(
            trail_N,
            trail_L,
            color=GRAY,
            lw=0.6,
            ls="--",
            label="frontier traced by the minima",
        )
        ax.plot(
            n_star,
            l_star,
            "o",
            ms=4,
            color=LILAC,
            mec="black",
            mew=0.4,
            label=r"minimum, $N^*(C)$",
        )
        ax.set_xlim(N[0], N[-1])
        loss_axis(ax, 1.8, 8, [2, 3, 4, 5, 6, 7, 8])
        ax.set_xlabel(r"parameters $N$ (log scale), $D = C / 6N$")
        legend(ax, loc="upper left")
        frames.append(rasterize(fig))
        plt.close(fig)
    frames_to_gif(frames, path)


def gif_exponent_sweep(path: str) -> None:
    """Kaplan's pure power law pinned at the center of the fitted range while the
    exponent moves by +-0.02: the fitted runs stay within a few percent of every
    curve, the forecast at 10^12 moves by about a quarter each way. The pivot IS the
    pin: every curve is forced through the same loss at N = 10^7.5.
    """
    anchor = 10**7.5
    L_anchor = float(KAPLAN.loss_N(np.array([anchor]))[0])
    grid = np.logspace(6, 12, 200)
    N_fit = np.logspace(6, 9, 5)
    L_fit = KAPLAN.loss_N(N_fit)
    L12_pub = float(KAPLAN.loss_N(np.array([1e12]))[0])
    lo, hi = KAPLAN.alpha_N - 0.02, KAPLAN.alpha_N + 0.02
    frames = []
    for alpha in np.concatenate([np.linspace(lo, hi, 24), np.linspace(hi, lo, 24)]):
        curve = L_anchor * (grid / anchor) ** (-alpha)
        L12 = float(L_anchor * (1e12 / anchor) ** (-alpha))
        fig, ax = gif_figure(
            r"Kaplan: $L = (N_c / N)^{\alpha_N}$, straight on log-log axes with slope $-\alpha_N$;"
            r"  published $\alpha_N$ = 0.076, $N_c = 8.8 \times 10^{13}$",
            r"every curve is pinned through the same loss at $N = 10^{7.5}$ (center of the fitted range),"
            r" so the sweep pivots there",
            rf"$\alpha_N$ = {alpha:5.3f}:  forecast $L(10^{{12}})$ = {L12:4.2f} (published {L12_pub:4.2f});"
            rf"  the fitted runs move by at most {100 * (10 ** (1.5 * abs(alpha - KAPLAN.alpha_N)) - 1):3.0f}\%",
            takeaway="Fits that agree on the runs you have can disagree on the run you want: the exponent is the forecast.",
        )
        ax.axvspan(grid[0], 1.6e9, color="#F2F2F2", lw=0, label="fitted range (shaded)")
        ax.loglog(
            grid,
            KAPLAN.loss_N(grid),
            color=CHARCOAL,
            lw=0.6,
            ls="--",
            label=r"published $\alpha_N$ = 0.076",
        )
        ax.loglog(grid, curve, color=BRICK, lw=1.2, label=r"$\alpha_N$ in the sweep")
        ax.plot(
            N_fit,
            L_fit,
            "o",
            ms=3,
            color=AMBER,
            mec="black",
            mew=0.3,
            label="runs the fit sees",
        )
        ax.plot(
            1e12,
            L12,
            "o",
            ms=4,
            mfc="white",
            mec=BRICK,
            mew=0.6,
            label=r"forecast at $10^{12}$",
        )
        ax.set_xlim(1e6, 1e12)
        loss_axis(ax, 0.7, 7, [1, 2, 3, 4, 5, 6, 7])
        ax.set_xlabel(r"parameters $N$ (log scale)")
        legend(ax, loc="upper right")
        frames.append(rasterize(fig))
        plt.close(fig)
    frames_to_gif(frames, path)


def gif_data_exponent_sweep(path: str) -> None:
    """The data-axis twin of the exponent sweep, for panel b: Kaplan's L(D) pinned at
    the center of a fitted range of 10^7 to 10^10 examples while alpha_D moves by
    +-0.02, and the forecast at 10^13 read off.
    """
    anchor = 10**8.5
    L_anchor = float(KAPLAN.loss_D(np.array([anchor]))[0])
    grid = np.logspace(7, 13, 200)
    D_fit = np.logspace(7, 10, 5)
    L_fit = KAPLAN.loss_D(D_fit)
    L13_pub = float(KAPLAN.loss_D(np.array([1e13]))[0])
    lo, hi = KAPLAN.alpha_D - 0.02, KAPLAN.alpha_D + 0.02
    frames = []
    for alpha in np.concatenate([np.linspace(lo, hi, 24), np.linspace(hi, lo, 24)]):
        curve = L_anchor * (grid / anchor) ** (-alpha)
        L13 = float(L_anchor * (1e13 / anchor) ** (-alpha))
        fig, ax = gif_figure(
            r"Kaplan: $L = (D_c / D)^{\alpha_D}$, straight on log-log axes with slope $-\alpha_D$;"
            r"  published $\alpha_D$ = 0.095, $D_c = 5.4 \times 10^{13}$",
            r"every curve is pinned through the same loss at $D = 10^{8.5}$ (center of the fitted range),"
            r" so the sweep pivots there",
            rf"$\alpha_D$ = {alpha:5.3f}:  forecast $L(10^{{13}})$ = {L13:4.2f} (published {L13_pub:4.2f});"
            rf"  the fitted runs move by at most {100 * (10 ** (1.5 * abs(alpha - KAPLAN.alpha_D)) - 1):3.0f}\%",
            takeaway="The data axis forecasts the same way: a data-exponent shift no run can rule out moves the far forecast by a quarter.",
        )
        ax.axvspan(
            grid[0], 1.6e10, color="#F2F2F2", lw=0, label="fitted range (shaded)"
        )
        ax.loglog(
            grid,
            KAPLAN.loss_D(grid),
            color=CHARCOAL,
            lw=0.6,
            ls="--",
            label=r"published $\alpha_D$ = 0.095",
        )
        ax.loglog(grid, curve, color=BRICK, lw=1.2, label=r"$\alpha_D$ in the sweep")
        ax.plot(
            D_fit,
            L_fit,
            "o",
            ms=3,
            color=AMBER,
            mec="black",
            mew=0.3,
            label="runs the fit sees",
        )
        ax.plot(
            1e13,
            L13,
            "o",
            ms=4,
            mfc="white",
            mec=BRICK,
            mew=0.6,
            label=r"forecast at $10^{13}$",
        )
        ax.set_xlim(1e7, 1e13)
        loss_axis(ax, 0.7, 7, [1, 2, 3, 4, 5, 6, 7])
        ax.set_xlabel(r"training examples $D$ (log scale)")
        legend(ax, loc="upper right")
        frames.append(rasterize(fig))
        plt.close(fig)
    frames_to_gif(frames, path)


def gif_floor_sweep(path: str) -> None:
    """The floor E rises from 0 to 2.5 under a fixed A N^-alpha: on log-log axes the
    curve bends where it meets E, and L - E stays the same straight line.
    """
    f = HOFFMANN
    N = np.logspace(6, 12, 200)
    term = f.A * N ** (-f.alpha)
    frames = []
    for E in np.concatenate([np.linspace(0.0, 2.5, 26), np.linspace(2.5, 0.0, 26)]):
        bend = (f.A / max(E, 1e-9)) ** (1 / f.alpha) if E > 0 else np.inf
        where = rf"$10^{{{np.log10(bend):4.1f}}}$" if np.isfinite(bend) else "nowhere"
        fig, ax = gif_figure(
            r"$L = E + A\,N^{-\alpha}$ on log-log axes, data not limiting",
            r"$A$ = 406.4, $\alpha$ = 0.34 from Hoffmann 2022;  $L - E$ is the same straight line in every frame",
            rf"floor $E$ = {E:4.2f}:  the curve bends where $A N^{{-\alpha}} = E$, at $N$ = {where}",
            takeaway="A bend on log-log axes is the floor showing itself: fit $L - E$, not $L$.",
        )
        ax.loglog(
            N, term + E, color=AMBER, lw=1.2, label=r"$L = E + A N^{-\alpha}$ (bends)"
        )
        ax.loglog(
            N,
            term,
            color=BRICK,
            lw=0.8,
            ls="--",
            label=r"$L - E = A N^{-\alpha}$ (straight)",
        )
        if E > 0:
            ax.axhline(E, color=GRAY, lw=0.6, ls=":", label=r"floor $E$")
        else:
            ax.plot([], [], color=GRAY, lw=0.6, ls=":", label=r"floor $E$")
        ax.set_xlim(1e6, 1e12)
        ax.set_ylim(0.02, 10)
        ax.set_ylabel(r"loss $L$")
        ax.set_xlabel(r"parameters $N$ (log scale)")
        legend(ax, loc="lower left")
        frames.append(rasterize(fig))
        plt.close(fig)
    frames_to_gif(frames, path)


def gif_surface_sweep(path: str) -> None:
    """The joint surface with one budget line 6ND = C sliding across it; the optimum
    is where the line touches its lowest contour, and the optima trace the frontier.
    """
    f = HOFFMANN
    logN = np.linspace(6, 12, 240)
    logD = np.linspace(7, 13, 240)
    NN, DD = np.meshgrid(10**logN, 10**logD)
    L = f.loss(NN, DD)
    cmap = LinearSegmentedColormap.from_list("tc_gray", ["#FFFFFF", CHARCOAL])
    levels = np.linspace(1.7, 7.0, 16)
    budgets = np.logspace(17, 26, 50)
    frames = []
    for logC in np.linspace(18, 24, 36):
        C = 10**logC
        n_star, d_star = float(f.N_opt(C)), float(f.D_opt(C))
        fig, ax = gif_figure(
            JOINT + r" as a surface over $\log_{10} N$ and $\log_{10} D$",
            HOFF,
            rf"budget $C = 10^{{{logC:4.1f}}}$ FLOPs, the line $6ND = C$:  lowest contour at "
            rf"$N^* = 10^{{{np.log10(n_star):4.1f}}}$, $D^* = 10^{{{np.log10(d_star):4.1f}}}$, "
            rf"$L^*$ = {float(f.loss(n_star, d_star)):4.2f}",
            takeaway="A budget is a diagonal on the surface; the best split is where it touches the lowest contour.",
        )
        cf = ax.contourf(logN, logD, L, levels=levels, cmap=cmap, extend="max")
        ax.contour(logN, logD, L, levels=levels, colors="white", linewidths=0.3)
        k = np.log10(C / FLOPS_PER_PARAM_PER_EXAMPLE)
        ax.plot(logN, k - logN, color=AMBER, lw=1.2, label=r"budget line $6ND = C$")
        ax.plot(
            np.log10(f.N_opt(budgets)),
            np.log10(f.D_opt(budgets)),
            color=BRICK,
            lw=0.8,
            label="compute-optimal frontier",
        )
        ax.plot(
            np.log10(n_star),
            np.log10(d_star),
            "o",
            ms=4,
            color=AMBER,
            mec="black",
            mew=0.4,
            label=r"optimum $(N^*, D^*)$",
        )
        ax.set_xlim(6, 12)
        ax.set_ylim(7, 13)
        ax.set_xlabel(r"$\log_{10}$ parameters $N$")
        ax.set_ylabel(r"$\log_{10}$ training examples $D$")
        cb = fig.colorbar(cf, ax=ax, fraction=0.045, pad=0.02)
        cb.ax.set_title(r"$L$", fontsize=7, pad=2)
        cb.set_ticks([2, 3, 4, 5, 6, 7])
        cb.ax.tick_params(labelsize=5, width=0.5, length=2)
        cb.outline.set_linewidth(0.5)  # type: ignore[operator]
        legend(ax, loc="lower left")
        frames.append(rasterize(fig))
        plt.close(fig)
    frames_to_gif(frames, path)


def gif_bootstrap(path: str) -> None:
    """One bootstrap resample per frame: the fifteen fitted runs are redrawn with
    replacement, the floored form is refit, and the refits accumulate into the band
    that panel i shows at once. The interval on alpha narrows to its value as the
    resamples pile up.
    """
    N, L = draw_runs(SYNTH)
    n_fit = SYNTH.n_fit_sizes * SYNTH.seeds_per_size
    Nf, Lf = N[:n_fit], L[:n_fit]
    Nh, Lh = N[n_fit:], L[n_fit:]
    grid = np.logspace(5.8, 10.3, 200)
    E0, A0, al0 = fit_floored(Nf, Lf)
    rng = np.random.default_rng(SYNTH.rng_seed + 2)
    truth = SYNTH.truth
    previous: list[FArray] = []
    alphas: list[float] = []
    frames = []
    for i in range(40):
        idx = rng.integers(0, Nf.size, size=Nf.size)
        E, A, al = fit_floored(Nf[idx], Lf[idx])
        alphas.append(al)
        lo, hi = (
            np.percentile(alphas, [5, 95]) if len(alphas) >= 5 else (np.nan, np.nan)
        )
        interval = (
            f"[{lo:5.3f}, {hi:5.3f}]" if len(alphas) >= 5 else "(needs 5 resamples)"
        )
        fig, ax = gif_figure(
            r"bootstrap over runs: redraw the 15 fitted runs with replacement, refit $L = E + A\,N^{-\alpha}$",
            rf"synthetic runs from $\alpha$ = 0.34, $E$ = 1.69 with 2\% noise;  full-sample fit $\alpha$ = {al0:5.3f}",
            rf"resample {i + 1:2d}:  $\alpha$ = {al:5.3f}, $E$ = {E:4.2f};  90\% interval on $\alpha$ so far {interval}",
            takeaway="Resample the runs and refit: the spread of the refits is the error bar on the exponent.",
        )
        ax.axvspan(
            grid[0],
            Nf.max() * 1.6,
            color="#F2F2F2",
            lw=0,
            label="fitted range (shaded)",
        )
        for prev in previous:
            ax.loglog(grid, prev, color=WHEAT, lw=0.5, alpha=0.25)
        ax.plot([], [], color=WHEAT, lw=0.5, alpha=0.5, label="earlier refits")
        ax.loglog(grid, E + A * grid ** (-al), color=WHEAT, lw=1.2, label="this refit")
        ax.loglog(
            grid,
            truth.E + truth.A * grid ** (-truth.alpha),
            color=CHARCOAL,
            lw=0.6,
            ls="--",
            label="generating curve",
        )
        ax.plot(
            Nf, Lf, "o", ms=2.5, color=AMBER, mec="black", mew=0.3, label="fitted runs"
        )
        counts = np.bincount(idx, minlength=Nf.size)
        drawn = counts > 0
        ax.scatter(
            Nf[drawn],
            Lf[drawn],
            s=14 + 14 * counts[drawn],
            facecolors="none",
            edgecolors=BRICK,
            linewidths=0.6,
            label="drawn in this resample (size = times drawn)",
            zorder=5,
        )
        ax.plot(
            Nh, Lh, "o", ms=2.5, mfc="white", mec=GRAY, mew=0.6, label="held-out runs"
        )
        previous.append(E + A * grid ** (-al))
        ax.set_xlim(grid[0], grid[-1])
        loss_axis(ax, 1.5, 7, [2, 3, 4, 5, 6, 7])
        ax.set_xlabel(r"parameters $N$ (log scale)")
        legend(ax, loc="upper right")
        frames.append(rasterize(fig))
        plt.close(fig)
    frames_to_gif(frames, path, duration_ms=160)


def gif_allocation_sweep(path: str) -> None:
    """Beta moves with alpha fixed: the allocation exponents a = beta / (alpha + beta)
    and b = alpha / (alpha + beta) move with it, and the N* and D* lines tilt.
    """
    f = HOFFMANN
    C = np.logspace(17, 25, 100)
    frames = []
    for beta in np.concatenate([np.linspace(0.2, 0.4, 24), np.linspace(0.4, 0.2, 24)]):
        g = ChinchillaFit(source="sweep", E=f.E, A=f.A, B=f.B, alpha=f.alpha, beta=beta)
        fig, ax = gif_figure(
            r"compute-optimal allocation: $N^* = G\,(C/6)^{a}$, $D^* = C / 6N^*$, "
            r"$a = \beta / (\alpha + \beta)$, $b = \alpha / (\alpha + \beta)$, $a + b = 1$",
            rf"$\alpha$ = 0.34 fixed;  Hoffmann's $\beta$ = 0.28 gives $a$ = {f.a:4.2f}, $b$ = {f.b:4.2f} (dashed)",
            rf"$\beta$ = {beta:4.2f}:  $a$ = {g.a:4.2f}, $b$ = {g.b:4.2f};  a steeper data term tilts the budget toward parameters",
            takeaway="How a budget splits between size and data is set by the ratio of the two exponents, nothing else.",
        )
        ax.loglog(
            C,
            f.N_opt(C),
            color=AMBER,
            lw=0.6,
            ls="--",
            label=r"$N^*$ at Hoffmann $\beta$",
        )
        ax.loglog(
            C,
            f.D_opt(C),
            color=BRICK,
            lw=0.6,
            ls="--",
            label=r"$D^*$ at Hoffmann $\beta$",
        )
        ax.loglog(C, g.N_opt(C), color=AMBER, lw=1.2, label=r"$N^*$ in the sweep")
        ax.loglog(C, g.D_opt(C), color=BRICK, lw=1.2, label=r"$D^*$ in the sweep")
        ax.set_xlim(C[0], C[-1])
        ax.set_ylim(1e6, 1e17)
        ax.set_xlabel(r"compute $C$ (FLOPs, log scale)")
        ax.set_ylabel(r"optimal size $N^*$, optimal data $D^*$ (log scale)")
        legend(ax, loc="upper left")
        frames.append(rasterize(fig))
        plt.close(fig)
    frames_to_gif(frames, path)


# One GIF per panel of Figure 1, in panel order.
GIFS = {
    "scaling_gif_a_exponent_sweep.gif": gif_exponent_sweep,
    "scaling_gif_b_data_exponent_sweep.gif": gif_data_exponent_sweep,
    "scaling_gif_c_floor_sweep.gif": gif_floor_sweep,
    "scaling_gif_d_surface_sweep.gif": gif_surface_sweep,
    "scaling_gif_e_data_sweep.gif": gif_data_sweep,
    "scaling_gif_f_params_sweep.gif": gif_params_sweep,
    "scaling_gif_g_compute_sweep.gif": gif_compute_sweep,
    "scaling_gif_h_allocation_sweep.gif": gif_allocation_sweep,
    "scaling_gif_i_bootstrap.gif": gif_bootstrap,
}


# --- the constants table -----------------------------------------------------------------
def write_constants_table() -> str:
    os.makedirs(TABLES_DIR, exist_ok=True)
    path = osp.join(TABLES_DIR, "t1-constants.tex")
    rows = [
        (
            "Kaplan 2020",
            r"$N_c$",
            f"{KAPLAN.N_c:.1e}".replace("e+", r"\times 10^{") + "}",
            "parameters",
        ),
        ("Kaplan 2020", r"$\alpha_N$", f"{KAPLAN.alpha_N}", "--"),
        (
            "Kaplan 2020",
            r"$D_c$",
            f"{KAPLAN.D_c:.1e}".replace("e+", r"\times 10^{") + "}",
            "tokens",
        ),
        ("Kaplan 2020", r"$\alpha_D$", f"{KAPLAN.alpha_D}", "--"),
    ]
    for f, name in [(HOFFMANN, "Hoffmann 2022"), (BESIROGLU, "Besiroglu 2024")]:
        rows += [
            (name, "$E$", f"{f.E}", "loss"),
            (name, "$A$", f"{f.A}", r"loss $\cdot$ parameters$^{\alpha}$"),
            (name, "$B$", f"{f.B}", r"loss $\cdot$ tokens$^{\beta}$"),
            (name, r"$\alpha$", f"{f.alpha}", "--"),
            (name, r"$\beta$", f"{f.beta}", "--"),
            (name, "$a = \\beta / (\\alpha + \\beta)$", f"{f.a:.3f}", "derived"),
            (name, "$b = \\alpha / (\\alpha + \\beta)$", f"{f.b:.3f}", "derived"),
        ]
    lines = [
        "%% SOURCE: experiments/041-scaling-laws/scripts/scaling_law_forms.py (write_constants_table)",
        "%% Published constants typed from the papers named in the Fit records of that script;",
        "%% a and b are derived from alpha and beta there. Never hand-edit this file.",
        r"\begin{table}[htbp]",
        r"\centering\footnotesize",
        r"\caption{Published constants of the two forms, and the allocation exponents they imply. "
        r"Kaplan's $N_c$ and $D_c$ set the height of a pure power law and have no meaning on their own; "
        r"Hoffmann's and Besiroglu's rows are two fits of the same Chinchilla training runs, and the "
        r"difference between them is the point of panel h. Every value is from outside our holdings.}",
        r"\label{tab:constants}",
        r"\begin{tabular}{llrl}",
        r"\toprule",
        r"fit & symbol & value & units \\",
        r"\midrule",
    ]
    last = None
    for src, sym, val, unit in rows:
        if last is not None and src != last:
            lines.append(r"\addlinespace")
        lines.append(f"{src if src != last else ''} & {sym} & ${val}$ & {unit} \\\\")
        last = src
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    text = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"wrote {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--no-gifs", action="store_true", help="skip the four GIF sweeps"
    )
    args = parser.parse_args()
    apply_paper_style()
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    summary: dict[str, object] = {
        "fits": {
            "kaplan": KAPLAN.model_dump(),
            "hoffmann": HOFFMANN.model_dump()
            | {"a": HOFFMANN.a, "b": HOFFMANN.b, "G": HOFFMANN.G},
            "besiroglu": BESIROGLU.model_dump()
            | {"a": BESIROGLU.a, "b": BESIROGLU.b, "G": BESIROGLU.G},
        },
        "flops_per_param_per_example": FLOPS_PER_PARAM_PER_EXAMPLE,
        "synthetic_spec": SYNTH.model_dump(),
    }
    summary.update(figure_1())
    write_constants_table()
    if not args.no_gifs:
        with matplotlib.rc_context(GIF_RC):
            for name, draw in GIFS.items():
                draw(osp.join(IMAGES_DIR, name))
    out = osp.join(RESULTS_DIR, "scaling_law_forms_summary.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
