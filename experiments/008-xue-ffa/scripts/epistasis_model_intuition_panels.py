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

# One color per model, fixed across this figure and Fig. 3's model panels.
C_MULT = PLOT_PALETTE[0]
C_ADD = PLOT_PALETTE[1]
C_GLM = PLOT_PALETTE[2]
C_OLS = PLOT_PALETTE[4]
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
        (r["f_ij"], PLOT_PALETTE[4], "measured double"),
    ]
    for i, (x, color, label) in enumerate(marks):
        y = len(marks) - 1 - i
        ax.plot([0, x], [y, y], color=color, linewidth=0.7, zorder=2)
        ax.plot([x], [y], marker="o", markersize=3.0, color=color, markeredgecolor="black",
                markeredgewidth=0.25, zorder=3)
        ax.text(x + 0.02, y, f"{label} {x:.2f}", va="center", ha="left", fontsize=5)
    # The gap between the two nulls is exactly the product of the two singles' losses.
    # It is drawn on its own row between them so it crosses neither label.
    gap = r["mult"] - r["add"]
    ax.annotate("", xy=(r["add"], 1.5), xytext=(r["mult"], 1.5),
                arrowprops=dict(arrowstyle="<->", linewidth=0.5, color="black",
                                shrinkA=0, shrinkB=0))
    ax.text(r["mult"] + 0.03, 1.5, f"$(1-f_i)(1-f_j) = {gap:.2f}$",
            ha="left", va="center", fontsize=5)
    ax.set_yticks([])
    ax.set_xlim(0, 1.75)
    ax.set_ylim(-0.7, len(marks) - 0.4)
    ax.set_xlabel("titer (rel. base strain)")
    ax.set_title(f"one double deletion, {a} and {b}:\nthe two nulls expect different things",
                 fontsize=6, pad=3)
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
    ax.set_title(f"{above['mult']} and {above['add']} of 45 doubles sit above the two\n"
                 f"expectations, by a median {med['mult']:+.2f} and {med['add']:+.2f}",
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
    ax.plot(grid, np.exp(intercept) * grid ** slope, color=C_GLM, linewidth=0.9, zorder=4,
            label=f"fitted slope {slope:.2f}")
    ref = float(np.median(y / x))
    ax.plot(grid, ref * grid, color=C_OLS, linewidth=0.7, linestyle=(0, (2, 1.5)),
            zorder=4, label="slope 1: spread scales with mean")
    ax.plot(grid, np.full_like(grid, float(np.median(y))), color=C_ADD, linewidth=0.7,
            linestyle=(0, (1, 1.5)), zorder=4, label="slope 0: constant spread")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("strain mean titer (rel. base strain)")
    ax.set_ylabel("replicate standard deviation")
    ax.set_title(f"spread grows with level across {len(keys)} strains,\n"
                 "which is what a log link assumes", fontsize=6, pad=3)
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


def panel_surfaces(ax, n=41):
    """The two expectations as surfaces over the unit square of single-deletion effects.

    This is the theory picture the rest of the figure is about. Both surfaces meet along
    the two edges where one deletion does nothing, because there the other deletion's
    effect is the combination's whatever the null; they separate in the interior, and the
    vertical gap between them is exactly (1 - f_i)(1 - f_j). The two log-scale models
    predict the SAME surface as the multiplicative one, which is the point the panel makes
    about where the four models do and do not differ: three share this expectation and
    differ in the scale their residual is measured on.
    """
    g = np.linspace(0.0, 1.0, n)
    fi, fj = np.meshgrid(g, g)
    # Additive first and more opaque, multiplicative over it and translucent: the
    # multiplicative surface is above the additive one everywhere on the square, so drawn
    # solid it would hide the thing the panel is comparing it with.
    ax.plot_surface(fi, fj, fi + fj - 1.0, color=C_ADD, alpha=0.85, linewidth=0,
                    antialiased=True, rstride=2, cstride=2)
    ax.plot_surface(fi, fj, fi * fj, color=C_MULT, alpha=0.55, linewidth=0,
                    antialiased=True, rstride=2, cstride=2)
    # The gap at the center of the square, drawn rather than described.
    x0 = 0.5
    ax.plot([x0, x0], [x0, x0], [x0 + x0 - 1.0, x0 * x0], color="black", linewidth=0.8,
            zorder=10)
    ax.text(x0, x0, (x0 * x0 + x0 + x0 - 1.0) / 2 + 0.06,
            f"  {x0 * x0 - (x0 + x0 - 1.0):.2f}", fontsize=5, zorder=11)
    ax.set_xlabel("$f_i$", labelpad=-8)
    ax.set_ylabel("$f_j$", labelpad=-8)
    ax.set_zlabel("")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_ticks([0, 0.5, 1])
        axis.set_tick_params(pad=-2, labelsize=5)
    ax.set_zlim(-1, 1)
    ax.set_zticks([-1, 0, 1])
    ax.view_init(elev=24, azim=-132)
    ax.set_box_aspect((1, 1, 0.78), zoom=1.04)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_alpha(0.0)
    ax.grid(True, linewidth=0.25)
    ax.set_title("expected $f_{ij}$ over the single-deletion square;\n"
                 "both log-scale models share the multiplicative surface",
                 fontsize=6, pad=-1)
    return {"gap_at_half": round(0.5 * 0.5 - (0.5 + 0.5 - 1.0), 3)}


def emit(name, width_key, height_mm, draw, projection=None):
    fig = plt.figure(figsize=(mm_to_in(PANEL_WIDTHS_MM[width_key]), mm_to_in(height_mm)))
    ax = fig.add_subplot(projection=projection)
    info = draw(ax)
    if projection is None:
        fig.tight_layout(pad=0.4)
    else:
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.07, top=0.91)
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
    emit("expectations", "third", 52.0, lambda ax: panel_expectations(ax, pairs))
    emit("null_fit", "third", 52.0, lambda ax: panel_null_fit(ax, pairs))
    emit("mean_variance", "third", 52.0,
         lambda ax: panel_mean_variance(ax, means, sds))
    emit("surfaces", "third", 52.0, panel_surfaces, projection="3d")
    emit_equations()


if __name__ == "__main__":
    main()
