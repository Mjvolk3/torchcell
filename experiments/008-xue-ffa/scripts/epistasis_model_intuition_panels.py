# experiments/008-xue-ffa/scripts/epistasis_model_intuition_panels.py
# [[experiments.008-xue-ffa.scripts.epistasis_model_intuition_panels]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/008-xue-ffa/scripts/epistasis_model_intuition_panels
#
# The data panels of the epistasis-model figure: what the four nulls expect, how far the
# measured doubles sit from each of them, and what the replicate noise looks like. The
# schematic panels and the comparison table are native draw.io shapes and are written by
# epistasis_model_intuition_drawio.py, which embeds these three SVGs.
#
# WHY THESE THREE. The figure has to answer three questions a reader brings to a table of
# interaction scores.
#
#   a  What does each null expect? Drawn on one real double deletion, on a titer axis, so
#      the residual each model calls an interaction is a visible distance. The additive and
#      the multiplicative expectation differ here by exactly (1 - f_i)(1 - f_j), the product
#      of the two single-deletion losses, which is an identity and is stated as one.
#   b  Which null do the data follow? All 45 measured doubles against both expectations.
#      Neither null is a fit; the spread around each line IS the digenic interaction, and
#      the panel shows that the choice of null moves every point, not a few.
#   c  What noise model does the titer justify? Replicate standard deviation against strain
#      mean over every measured strain, on log axes, with the two assumptions the four
#      models make drawn as reference slopes: constant spread (slope 0), which is what OLS
#      on the linear scale would assume, and spread proportional to the mean (slope 1),
#      which is the Gamma family and is equivalent to constant spread on the log scale.
#
# Every number is computed here from the same normalized strain means the interaction
# scripts use, through free_fatty_acid_interactions.load_ffa_data and
# normalize_by_reference, so a change in normalization moves these panels with the rest.

import json
import os
import os.path as osp
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from free_fatty_acid_interactions import (  # noqa: E402
    load_ffa_data,
    normalize_by_reference,
    parse_genotype,
)

from torchcell.utils import (  # noqa: E402
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    apply_paper_style,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "008-xue-ffa/results")
IMAGES_DIR = osp.join(ASSET_IMAGES_DIR, "008-xue-ffa")
RAW_XLSX = osp.join(DATA_ROOT,
                    "data/torchcell/ffa_xue2025/raw/Supplementary Data 1_Raw titers.xlsx")

TOTAL = "Total Titer"

# One color per model, fixed across this figure and the model panel of Fig. 1
# (perspective_figure_panels.MODEL_COLORS carries the same four). The multiplicative model
# is blue and the additive one brick: as orange and brick the two nulls of panel b sat on
# one another at 5 pt marker size (author review, 2026.09.18), and those two are the pair
# every panel of the figure contrasts. Log-OLS takes the orange the multiplicative model
# gave up.
C_MULT = PLOT_PALETTE[4]
C_ADD = PLOT_PALETTE[1]
C_GLM = PLOT_PALETTE[2]
C_OLS = PLOT_PALETTE[0]
C_MEASURED = PLOT_PALETTE[5]


def strain_titers():
    """Normalized total-titer mean and replicate SD for every measured strain.

    Returns (means, sds), both keyed by the frozenset of deleted transcription factors, and
    the base strain as the empty set. The three chassis deletions are not in the key: every
    strain carries them, and the normalization divides them out.
    """
    raw, abbreviations, replicate_dict = load_ffa_data(RAW_XLSX)
    normalized, normalized_reps = normalize_by_reference(raw, replicate_dict)
    means, sds = {}, {}
    for _, row in normalized.iterrows():
        genotype = row.iloc[0]
        genes = parse_genotype(genotype, abbreviations)
        if genes is None:
            continue
        key = frozenset(genes)
        means[key] = float(row[TOTAL])
        reps = np.asarray(normalized_reps[genotype][TOTAL], dtype=float)
        sds[key] = float(np.std(reps, ddof=1))
    assert len(means) == 176, f"{len(means)} strains, expected 176"
    return means, sds


def pair_frame(means):
    """The 45 doubles with their two singles and both nulls' expectations."""
    singles = {next(iter(k)): v for k, v in means.items() if len(k) == 1}
    rows = []
    for key, f_ij in means.items():
        if len(key) != 2:
            continue
        a, b = sorted(key)
        rows.append({
            "pair": f"{a}-{b}", "f_i": singles[a], "f_j": singles[b], "f_ij": f_ij,
            "mult": singles[a] * singles[b], "add": singles[a] + singles[b] - 1.0,
        })
    out = pd.DataFrame(rows).sort_values("pair").reset_index(drop=True)
    assert len(out) == 45, f"{len(out)} doubles, expected 45"
    return out


def panel_expectations(ax, pairs):
    """One real double deletion on a titer axis, with what each null expects for it.

    The pair shown is the one whose two nulls are furthest apart, which is the pair where
    the choice of model matters most and is therefore the honest one to draw.
    """
    r = pairs.loc[(pairs["mult"] - pairs["add"]).abs().idxmax()]
    a, b = r["pair"].split("-")
    marks = [
        (1.0, C_MEASURED, "base strain"),
        (r["f_i"], C_MEASURED, f"$f$ {a}"),
        (r["f_j"], C_MEASURED, f"$f$ {b}"),
        (r["add"], C_ADD, "additive expects"),
        (r["mult"], C_MULT, "multiplicative expects"),
        (r["f_ij"], "#000000", "measured double"),
    ]
    for i, (x, color, label) in enumerate(marks):
        y = len(marks) - 1 - i
        ax.plot([0, x], [y, y], color=color, linewidth=0.7, zorder=2)
        ax.plot([x], [y], marker="o", markersize=3.0, color=color, markeredgecolor="black",
                markeredgewidth=0.25, zorder=3)
        # Clear of the marker rather than against it: at 0.02 the text touched the dot
        # it belongs to, which read as one mark (author review, 2026.09.18).
        ax.text(x + 0.055, y, f"{label} {x:.2f}", va="center", ha="left", fontsize=5)
    # The gap between the two nulls is exactly the product of the two singles' losses.
    # It is drawn on its own row between them so it crosses neither label.
    gap = r["mult"] - r["add"]
    ax.annotate("", xy=(r["add"], 1.5), xytext=(r["mult"], 1.5),
                arrowprops=dict(arrowstyle="<->", linewidth=0.5, color="black",
                                shrinkA=0, shrinkB=0))
    ax.text(r["mult"] + 0.065, 1.5, f"$(1-f_i)(1-f_j) = {gap:.2f}$",
            ha="left", va="center", fontsize=5)
    ax.set_yticks([])
    ax.set_xlim(0, 1.75)
    ax.set_ylim(-0.7, len(marks) - 0.4)
    ax.set_xlabel("titer (rel. base strain)")
    ax.set_title(f"the two nulls expect different things of {a} {b}", fontsize=6, pad=3)
    return {"pair": r["pair"], "null_gap": round(float(gap), 3)}


def panel_null_fit(ax, pairs):
    """All 45 doubles against both expectations, with the identity line.

    Distance from the line is the digenic interaction under that model, so the panel shows
    that the null is not a fit and that changing it moves every pair.
    """
    for col, color, label in ((("mult"), C_MULT, "multiplicative"),
                              (("add"), C_ADD, "additive")):
        ax.scatter(pairs[col], pairs["f_ij"], s=5, facecolor=color, edgecolor="black",
                   linewidth=0.2, zorder=3, label=label)
    vals = pairs[["mult", "add", "f_ij"]].to_numpy()
    pad = 0.06 * (vals.max() - vals.min())
    lo, hi = vals.min() - pad, vals.max() + pad
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.5, linestyle="--", zorder=2,
            label="expectation met")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("expected titer of the double")
    ax.set_ylabel("measured titer of the double")
    above = {c: int((pairs["f_ij"] > pairs[c]).sum()) for c in ("mult", "add")}
    med = {c: float((pairs["f_ij"] - pairs[c]).median()) for c in ("mult", "add")}
    # One line, and short enough to print inside the panel: a title wider than the panel
    # is not wrapped by matplotlib, it is cut off by the figure's own edge. The two
    # median residuals it used to carry are in the caption.
    ax.set_title(f"{above['mult']} and {above['add']} of 45 doubles beat the two nulls",
                 fontsize=6, pad=3)
    ax.legend(loc="lower right", frameon=False, handlelength=1.2, fontsize=5,
              labelspacing=0.25, borderaxespad=0.3, scatterpoints=1)
    return {"above": above, "median_residual": {k: round(v, 3) for k, v in med.items()}}


def panel_mean_variance(ax, means, sds):
    """Replicate spread against strain level, with the two noise assumptions as slopes.

    Fit is a least-squares line through the logs; its slope is what separates a constant
    spread (0) from a spread proportional to the mean (1), which is the Gamma family and is
    what makes a log link the natural one for a titer.
    """
    keys = [k for k in means if sds[k] > 0]
    x = np.array([means[k] for k in keys])
    y = np.array([sds[k] for k in keys])
    ax.scatter(x, y, s=4, facecolor=C_MEASURED, edgecolor="none", alpha=0.65, zorder=3)
    slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
    grid = np.linspace(x.min(), x.max(), 50)
    # The same symbol as panels a and b: the strain's titer is f, and the spread is the
    # spread of f. Naming it "strain mean" here and "slope 0" and "slope 1" in the key put
    # three vocabularies on one panel (author review, 2026.09.18). The two reference lines
    # are the two noise assumptions the four models divide over, and each is named by the
    # assumption and colored by the family that makes it.
    # A tilde, not \propto: Arial has no proportionality glyph and the SVG fell back to a
    # slash for it.
    ax.plot(grid, np.exp(intercept) * grid ** slope, color="#000000", linewidth=0.9,
            zorder=4, label=f"fit: SD ~ $f^{{\\,{slope:.2f}}}$")
    ref = float(np.median(y / x))
    ax.plot(grid, ref * grid, color=C_GLM, linewidth=0.7, linestyle=(0, (2, 1.5)),
            zorder=4, label="SD ~ $f$: the log-scale models")
    ax.plot(grid, np.full_like(grid, float(np.median(y))), color=C_MULT, linewidth=0.7,
            linestyle=(0, (1, 1.5)), zorder=4, label="SD constant: the linear-scale models")
    ax.set_xscale("log")
    ax.set_yscale("log")
    # Headroom for the key: at 38 mm the three rows of the key sat on the constant-spread
    # line and the points under it (author review, 2026.09.18). The top of the axis is
    # raised so the key sits above every point.
    ax.set_ylim(y.min() * 0.6, y.max() * 4.5)
    # Plain tick numbers. The default log formatter wrote 6 x 10^-1 and 2 x 10^0 for an
    # axis that spans half to twice the base strain.
    ax.set_xticks([0.5, 1.0, 2.0])
    ax.set_xticklabels(["0.5", "1", "2"])
    ax.set_yticks([0.01, 0.1])
    ax.set_yticklabels(["0.01", "0.1"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("$f$, strain mean titer (rel. base strain)")
    ax.set_ylabel("replicate SD of $f$")
    ax.set_title(f"spread grows with level across {len(keys)} strains", fontsize=6, pad=3)
    ax.legend(loc="upper left", frameon=False, handlelength=1.6, fontsize=5,
              labelspacing=0.25, borderaxespad=0.3, scatterpoints=1)
    return {"n_strains": len(keys), "log_log_slope": round(float(slope), 3)}


# Every expression in this figure is set as MATH, not as HTML subscripts. draw.io's
# <sub> is not typesetting: the variables come out upright rather than italic, the
# subscripts sit at the wrong size, and the exported PDF carried a serif fallback for the
# subscript runs, putting a fourth typeface in the figure. Rendering each expression here
# gives real math, in Arial, in the same 6 pt as the rest of the figure text, and the
# draw.io generator places the results as images.
EQUATIONS = {
    "f_i": r"$f_i$",
    "f_j": r"$f_j$",
    "mult": r"$f_i\,f_j$",
    "add": r"$f_i + f_j - 1$",
    "loss_i": r"$1 - f_i$",
    "loss_j": r"$1 - f_j$",
    "glm": r"$\exp(\alpha_i + \alpha_j)$",
    "gap": r"$(1 - f_i)(1 - f_j)$",
    # The classic picture in the figure's panel b: the interaction as the departure of
    # the measured double from what the singles predict.
    "f_ij": r"$f_{ij}$",
    "eps_def": r"$\varepsilon_{ij} = f_{ij} - f_i\,f_j$",
}


def emit_equations():
    """Each expression as a tight true-size SVG, with its size in millimetres.

    Rendered on its own transparent canvas and cropped to the ink, so the draw.io
    generator can center it in a cell without knowing anything about the expression. The
    sizes are written to a JSON beside the panels; a changed expression changes its box.
    """
    sizes = {}
    for name, tex in EQUATIONS.items():
        fig = plt.figure(figsize=(2.0, 0.3))
        t = fig.text(0, 0, tex, fontsize=6, ha="left", va="baseline")
        fig.canvas.draw()
        bb = t.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        pad = 0.01
        fig.set_size_inches(bb.width + 2 * pad, bb.height + 2 * pad)
        t.set_position((pad / (bb.width + 2 * pad),
                        pad / (bb.height + 2 * pad) - bb.y0 / (bb.height + 2 * pad)))
        t.set_transform(fig.transFigure)
        out = osp.join(IMAGES_DIR, f"mi_eq_{name}.svg")
        savefig_true_size_svg(fig, out, transparent=True)
        plt.close(fig)
        sizes[name] = [round((bb.width + 2 * pad) * 25.4, 3),
                       round((bb.height + 2 * pad) * 25.4, 3)]
    with open(osp.join(RESULTS_DIR, "epistasis_model_equation_sizes.json"), "w") as fh:
        json.dump(sizes, fh, indent=2)
    print(f"wrote {len(sizes)} equation SVGs")
    return sizes


# The four models as four level-set maps, drawn on the same square and in the same order
# as the table. Each entry is (key, color, title, surface, the scale its residual is
# measured on). Three of the four expect the same surface, which is the panel's first
# point; what separates them is the second column of the table, and it is drawn here as
# the SPACING of the contours, since a model's contours are evenly spaced on the scale it
# measures its residual on.
LEVEL_SET_MODELS = [
    ("multiplicative", C_MULT, "multiplicative", lambda a, b: a * b, "linear"),
    ("additive", C_ADD, "additive", lambda a, b: a + b - 1.0, "linear"),
    ("glm_log_link", C_GLM, "GLM log-link", lambda a, b: a * b, "log"),
    ("log_ols", C_OLS, "log-OLS", lambda a, b: a * b, "log"),
]
# The square runs to 1.5 on each axis, since a single deletion in this design can raise
# titer as well as lower it and the level sets above 1 are where the two surfaces part
# most (author review, 2026.09.18). Five contours each: evenly spaced in titer for the
# two linear-scale models, and a doubling apart for the two log-scale ones, which is what
# an equal residual means to a model fit on log titer.
SQUARE = 1.5
# The top level of each set still crosses the square as an arc, not a corner scrap: at
# 2.0 the additive surface touched only the corner point and its label sat on the 1.6.
LEVELS_LINEAR = [0.3, 0.6, 0.9, 1.2, 1.5]
LEVELS_LOG = [0.1, 0.2, 0.4, 0.8, 1.6]
# Where a contour's own value is written: on the diagonal of the square, which for both
# surfaces is the point of that contour closest to the origin, so the numbers run in a
# line up the panel instead of landing wherever a contour happens to leave it. A contour
# whose diagonal point is within 0.2 of the marked center is labeled on the steeper ray
# f_j = 1.8 f_i instead, so no number sits on the dot.
ON_DIAGONAL = {"multiplicative": np.sqrt, "additive": lambda v: (v + 1.0) / 2.0}
RAY = 1.8
ON_RAY = {"multiplicative": lambda v: np.sqrt(v / RAY),
          "additive": lambda v: (v + 1.0) / (1.0 + RAY)}


def label_point(kind, v, center=0.5, keep_clear=0.2):
    d = float(ON_DIAGONAL[kind](v))
    if abs(d - center) * np.sqrt(2.0) > keep_clear:
        return (d, d)
    x = float(ON_RAY[kind](v))
    return (x, RAY * x)


def panel_level_sets(axes, n=181):
    """What each model expects of a double, as level sets over the square of its singles.

    One axes per model, in the order the table lists them, so the four are read across and
    any one point on the square is read down. The center of the square is marked in every
    panel with the value that model expects there, which is the comparison in one number:
    0.25, 0.00, 0.25, 0.25.

    Two differences are visible and they are different kinds of thing. The additive
    surface is not the multiplicative one: its level sets are straight where theirs are
    hyperbolas, and the two surfaces differ by (1 - f_i)(1 - f_j), which is largest where
    both deletions are severe and is 0.25 at the center. The two log-scale models expect
    the same surface as the multiplicative one, and differ in where they put equal
    residuals: their contours are evenly spaced in the logarithm, so they crowd near 1
    and spread out where titer is low.
    """
    g = np.linspace(0.0, SQUARE, n)
    fi, fj = np.meshgrid(g, g)
    out = {}
    for ax, (key, color, title, surface, scale) in zip(axes, LEVEL_SET_MODELS):
        levels = LEVELS_LINEAR if scale == "linear" else LEVELS_LOG
        kind = "additive" if key == "additive" else "multiplicative"
        cs = ax.contour(fi, fj, surface(fi, fj), levels=levels, colors=color,
                        linewidths=0.6)
        ax.clabel(cs, cs.levels, inline=True, inline_spacing=1, fontsize=5,
                  fmt=lambda v: f"{v:g}",
                  manual=[label_point(kind, v) for v in levels])
        mid = float(surface(0.5, 0.5))
        ax.plot([0.5], [0.5], marker="o", markersize=2.2, color="black", zorder=5)
        # Bottom left and left-justified, so the four notes start on one vertical line and
        # read as one row across the panels; right-justified at bottom right they ended
        # at four different places (author review, 2026.09.18). The white ground covers
        # the tail of the lowest hyperbola, which hugs the axis there.
        ax.text(0.03, 0.03, f"expects {mid:.2f} at (0.5, 0.5)", transform=ax.transAxes,
                va="bottom", ha="left", fontsize=5, zorder=6,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
        ax.set_xlim(0, SQUARE)
        ax.set_ylim(0, SQUARE)
        ax.set_xticks([0, 0.5, 1.0, 1.5])
        ax.set_yticks([0, 0.5, 1.0, 1.5])
        ax.set_aspect("equal")
        ax.set_xlabel("$f_i$", labelpad=1)
        ax.set_title(f"{title}, residual on {scale} titer", fontsize=6, pad=3)
        out[key] = {"levels": levels, "at_half": round(mid, 3)}
    axes[0].set_ylabel("$f_j$", labelpad=1)
    out["gap_at_half"] = round(out["multiplicative"]["at_half"] - out["additive"]["at_half"], 3)
    return out


def emit(name, width_key, height_mm, draw, ncols=1):
    fig, axes = plt.subplots(
        1, ncols, figsize=(mm_to_in(PANEL_WIDTHS_MM[width_key]), mm_to_in(height_mm)))
    axes = np.atleast_1d(axes)
    info = draw(*axes) if ncols > 1 else draw(axes[0])
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
    fig.tight_layout(pad=0.4)
    png = osp.join(IMAGES_DIR, f"mi_panel_{name}.png")
    svg = osp.join(IMAGES_DIR, f"mi_panel_{name}.svg")
    fig.savefig(png, dpi=300)
    savefig_true_size_svg(fig, svg)
    plt.close(fig)
    print(f"wrote mi_panel_{name}  ({PANEL_WIDTHS_MM[width_key]} x {height_mm} mm)  {info}")


def main():
    apply_paper_style()
    os.makedirs(IMAGES_DIR, exist_ok=True)
    means, sds = strain_titers()
    pairs = pair_frame(means)
    pairs.to_csv(osp.join(RESULTS_DIR, "epistasis_model_intuition_pairs.csv"), index=False)
    # Heights: the two schematic panels that briefly sat above these moved to Fig. 1
    # (author review, 2026.09.18), so the three measured panels are back at 52 mm and the
    # level sets, which the review asked to be larger, at 40.
    emit("expectations", "third", 52.0, lambda ax: panel_expectations(ax, pairs))
    emit("null_fit", "third", 52.0, lambda ax: panel_null_fit(ax, pairs))
    emit("mean_variance", "third", 52.0,
         lambda ax: panel_mean_variance(ax, means, sds))
    emit("level_sets", "full", 40.0, lambda *axes: panel_level_sets(axes), ncols=4)
    emit_equations()


if __name__ == "__main__":
    main()
