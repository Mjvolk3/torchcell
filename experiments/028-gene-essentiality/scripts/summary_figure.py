# experiments/028-gene-essentiality/scripts/summary_figure.py
# [[experiments.028-gene-essentiality.scripts.summary_figure]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/summary_figure

"""The document's results on one page, nine panels, each from one committed result file.

Panels, in the order the document reaches them: the comparators computed rather than
quoted, the reproduction of Flux Cone Learning's own numbers, the fold ensembles along
the arc of rounds, the families and their post hoc averages, the genome-wide transfer,
the solve's argmax across solvers as a probe-transfer matrix, what the four stores carry,
the capacity box under the two turnover tiers, and the store at the database's scale.

Nothing is typed here: every number is read from ``results/``. Panels are added by
appending to ``PANELS``, so the figure grows without the layout being rewritten.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD \
        python experiments/028-gene-essentiality/scripts/summary_figure.py
"""

from __future__ import annotations

import json
import os
import os.path as osp
from typing import Any, Callable

import matplotlib
import numpy as np
from dotenv import load_dotenv

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

from torchcell.timestamp import timestamp  # noqa: E402
from torchcell.utils import (  # noqa: E402
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
RESULTS = osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "results")
IMAGES = osp.join(ASSET_IMAGES_DIR, "028-gene-essentiality")
#: Orange, red, purple and yellow are the primaries and are spent before blue and gray.
C = PLOT_PALETTE


def read(name: str) -> Any:
    with open(osp.join(RESULTS, name)) as fh:
        return json.load(fh)


def box(ax: Axes) -> None:
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    ax.tick_params(width=0.5, length=2)


def auroc_axis(ax: Axes, low: float = 0.4, high: float = 1.0) -> None:
    """A 0-to-1 metric axis: a gridline every 0.1, a label every 0.2."""
    ax.set_xlim(low, high)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(axis="x", which="minor", length=0)
    ax.grid(axis="x", which="both", lw=0.3, color="0.85", zorder=0)
    ax.set_axisbelow(True)


def hbars(
    ax: Axes,
    labels: list[str],
    values: list[float],
    colors: list[str],
    err: list[tuple[float, float]] | None = None,
) -> None:
    y = np.arange(len(labels))[::-1]
    xerr = None
    if err is not None:
        xerr = np.array(
            [[v - lo for v, (lo, _) in zip(values, err, strict=True)],
             [hi - v for v, (_, hi) in zip(values, err, strict=True)]]
        )
    ax.barh(y, values, height=0.68, color=colors, edgecolor="black", lw=0.4, zorder=3)
    if xerr is not None:
        ax.errorbar(
            values, y, xerr=xerr, fmt="none", ecolor="black", elinewidth=0.5, capsize=1.2, zorder=4
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.6, len(labels) - 0.4)


# --------------------------------------------------------------------------- panels


def panel_fba(ax: Axes) -> str:
    """The comparator FBA runs on four media, over the 1,121 labeled genes."""
    d = read("fba_essentiality_by_medium.json")
    label = {
        "shipped": "shipped bounds",
        "SM": "synthetic minimal",
        "SC": "synthetic complete",
        "YPD": "YPD approximation",
    }
    rows = []
    for r in d["rows"]:
        s = next(x for x in r["scores"] if x["gene_set"] == "all_labeled")
        rows.append((label[r["medium"]], s["auroc"], (s["auroc_ci_low"], s["auroc_ci_high"])))
    hbars(
        ax,
        [r[0] for r in rows],
        [r[1] for r in rows],
        [C[i % 4] for i in range(len(rows))],
        [r[2] for r in rows],
    )
    auroc_axis(ax, 0.4, 0.8)
    ax.set_xlabel("AUROC, 1,121 labeled genes")
    return "The growth ratio of a single-gene deletion, four media, gene bootstrap."


def panel_fcl(ax: Axes) -> str:
    """Flux Cone Learning's published numbers against our recomputation of them."""
    d = read("fcl_comparison.json")
    name = {
        "fcl_cv_5fold_mean": "FCL, folds inside training",
        "fcl_test_223": "FCL, 223 held out",
        "fba_all_1121": "FBA, 1,121 labeled",
    }
    rows = [r for r in d["reproduction_check"] if r["name"] in name]
    y = np.arange(len(rows))[::-1]
    ax.barh(
        y + 0.18, [r["published"] for r in rows], height=0.34, color=C[0],
        edgecolor="black", lw=0.4, label="published", zorder=3
    )
    ax.barh(
        y - 0.18, [r["recomputed"] for r in rows], height=0.34, color=C[1],
        edgecolor="black", lw=0.4, label="recomputed here", zorder=3
    )
    ax.set_yticks(y)
    ax.set_yticklabels([name[r["name"]] for r in rows])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    auroc_axis(ax, 0.4, 0.9)
    ax.set_xlabel("AUROC")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.16), ncol=2, frameon=False, handlelength=1.0)
    return "Each published value reproduces to the third decimal on its own gene set."


#: The arc of fold ensembles, round by round, with the label the document uses.
ENSEMBLE_ARC = [
    ("w14_cv_ensemble.json", "indicator"),
    ("w16_fbacv_ensemble.json", "+ FBA affine"),
    ("w23_chromprotcv_ensemble.json", "+ chromosome, ProtT5"),
    ("w28_cspcv_ensemble.json", "+ pathway"),
    ("w41_pathfluxcv_ensemble.json", "+ flux layer"),
    ("w43_pathoffcv_ensemble.json", "cached solve, no layer"),
    ("w50_pathlogfreecv_ensemble.json", "layer-free log1p"),
    ("w70_pathlogh48cv_ensemble.json", "layer-free, width 48"),
    ("w53_pathlogipmcv_ensemble.json", "interior-point store"),
    ("w55_pathlogpdlpcv_ensemble.json", "PDLP store"),
    ("w74_pathgopfbaaffcv_ensemble.json", "+ GO in the affine"),
    ("w82_pathlogfixedcv_ensemble.json", "pinned kernels"),
]


def panel_ensembles(ax: Axes) -> str:
    """The five-fold ensemble of each form, in the order the rounds reached them."""
    rows = []
    for name, label in ENSEMBLE_ARC:
        e = read(name)
        rows.append((label, e["ensemble_auroc"], (e["ensemble_ci_low"], e["ensemble_ci_high"])))
    hbars(
        ax,
        [r[0] for r in rows],
        [r[1] for r in rows],
        [C[0]] * len(rows),
        [r[2] for r in rows],
    )
    auroc_axis(ax, 0.6, 1.0)
    ax.set_xlabel("fold-ensemble AUROC, 195 genes")
    return "Five fold models per form, their standardized logits averaged."


def panel_families(ax: Axes) -> str:
    """Single families and the post hoc averages of them."""
    d = read("family_ensembles.json")
    singles = [c for c in d["combinations"] if len(c["families"]) == 1]
    multi = [c for c in d["combinations"] if len(c["families"]) > 1]
    singles = sorted(singles, key=lambda c: c["auroc"])[-4:]
    multi = sorted(multi, key=lambda c: c["auroc"])[-3:]
    short = {
        "pathway + ProtT5 + affine (round 22, 3 seeds)": "affine",
        "flux head, pathway + ProtT5 (rounds 23 and 25, 6 seeds)": "flux head",
        "flux head, ProtT5 (rounds 21 and 24, 6 seeds)": "flux head, ProtT5",
        "layer-free log1p vector + affine, pathway + ProtT5 (round 33, 6 seeds)": "layer-free",
        "layer-free log1p, interior-point store (round 36, 6 seeds)": "layer-free, ipm",
        "cached solve + flux head, pathway + ProtT5 (round 27, 6 seeds)": "cached solve",
        "GO convolution (400 epochs, 3 seeds)": "GO conv",
        "GO convolution (400 epochs, 6 seeds)": "GO conv, 6 seeds",
    }

    def name(c: dict[str, Any]) -> str:
        if len(c["families"]) == 1:
            return short.get(c["families"][0], c["families"][0])
        # In a combination the seed count is not what distinguishes the members, so the
        # labels drop to their shortest form and the caption carries the rest.
        parts = [short.get(f, f).replace("GO conv, 6 seeds", "GO").replace("GO conv", "GO")
                 for f in c["families"]]
        return " + ".join(parts)

    rows = singles + multi
    colors = [C[0]] * len(singles) + [C[2]] * len(multi)
    hbars(
        ax,
        [name(c) for c in rows],
        [c["auroc"] for c in rows],
        colors,
        [(c["ci_low"], c["ci_high"]) for c in rows],
    )
    auroc_axis(ax, 0.8, 1.0)
    ax.set_xlabel("AUROC, 195 genes")
    return (
        "The four best single families in orange, the three best post hoc averages of "
        "them in purple; an average over families is the document's highest number."
    )


def panel_genome(ax: Axes) -> str:
    """The same forms scored on the genome-wide held-out set."""
    d = read("genome_heldout_readout.json")
    keep = {
        "graph convolution, physical graph, degree and clustering": "graph, degree",
        "graph convolution on ProtT5 + FBA ratios": "graph, ProtT5 + ratios",
        "graph convolution, union of nine, ProtT5": "graph, union, ProtT5",
        "graph convolution on GO alone, 400 epochs": "graph, GO",
        "pathway multi-hot + ProtT5 + FBA affine (round 22 recipe)": "transformer, affine",
        "layer-free log1p vector + affine, pathway + ProtT5 (round 33 form)": "transformer, layer-free",
        "cached solve + flux head, pathway + ProtT5 (round 27 form)": "transformer, cached solve",
        "MLP on ProtT5 alone": "MLP, ProtT5",
    }
    rows = [
        (keep[a["label"]], a["pooled_auroc"])
        for a in d["arms"]
        if a["label"] in keep and a.get("pooled_auroc") is not None
    ]
    rows.sort(key=lambda r: r[1])
    colors = [C[5] if "MLP" in r[0] or "transformer" in r[0] else C[1] for r in rows]
    hbars(ax, [r[0] for r in rows], [r[1] for r in rows], colors)
    ax.axvline(0.5, color="black", lw=0.5, ls=(0, (2, 2)), zorder=5)
    auroc_axis(ax, 0.4, 0.8)
    ax.set_xlabel(f"AUROC, {d['n_test_genes']:,} held-out genes")
    return (
        "Red is message passing over a gene graph, gray is the transformer and the "
        "multilayer perceptron; the dashed line is chance."
    )


def panel_transfer(ax: Axes) -> str:
    """A probe fitted on one solver's flux, scored on another's."""
    d = read("solve_tolerance_transfer.json")
    names = sorted({p["fit"] for p in d["pairs"]})
    short = {
        "simplex_exact": "simplex",
        "simplex_exact2": "simplex 2",
        "ipm_nocross_1e-4": "ipm 4",
        "ipm_nocross_1e-6": "ipm 6",
        "ipm_nocross_1e-8": "ipm 8",
        "pdlp_1e-4": "PDLP 4",
        "pdlp_1e-6": "PDLP 6",
        "pdlp_1e-8": "PDLP 8",
    }
    m = np.full((len(names), len(names)), np.nan)
    for p in d["pairs"]:
        m[names.index(p["fit"]), names.index(p["score"])] = p["probe_auroc_test"]
    # pcolormesh rather than imshow: an image is raster in the PDF, and `make check`
    # wants a figure of this kind to be vector throughout.
    im = ax.pcolormesh(m, cmap="cividis", vmin=0.45, vmax=0.80, rasterized=False)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ticks = np.arange(len(names)) + 0.5
    ax.set_xticks(ticks)
    ax.set_xticklabels([short.get(n, n) for n in names], rotation=45, ha="right")
    ax.set_yticks(ticks)
    ax.set_yticklabels([short.get(n, n) for n in names])
    for i in range(len(names)):
        for j in range(len(names)):
            if not np.isnan(m[i, j]):
                ax.text(
                    j + 0.5, i + 0.5, f"{m[i, j]:.2f}", ha="center", va="center",
                    fontsize=4.6, color="white" if m[i, j] < 0.68 else "black",
                )
    ax.set_xlabel("scored on")
    ax.set_ylabel("fitted on")
    cax = ax.inset_axes((1.05, 0.0, 0.045, 1.0))
    cb = ax.figure.colorbar(im, cax=cax)
    cb.solids.set_rasterized(False)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(width=0.5, length=2)
    cb.set_label("probe AUROC", labelpad=2)
    return (
        "One growth program, seven solves. The diagonal is the within-solver number; a "
        "probe carried across solvers loses up to 0.35, and across tolerances of one "
        "solver loses nothing."
    )


def panel_stores(ax: Axes) -> str:
    """How far each store's flux sits from the pFBA store's, in the head's encoding."""
    d = read("store_stats.json")
    short = {
        "pFBA (two-stage, scipy HiGHS)": "pFBA",
        "dual simplex (vertex)": "dual simplex",
        "interior point, no crossover": "interior point",
        "PDLP, no crossover": "PDLP",
    }
    rows = [s for s in d["stores"] if s["store"] != "pFBA (two-stage, scipy HiGHS)"]
    y = np.arange(len(rows))[::-1]
    ax.barh(
        y + 0.18, [s["log1p_rel_l2_to_pfba_median"] for s in rows], height=0.34,
        color=C[0], edgecolor="black", lw=0.4, label="relative $L_2$", zorder=3
    )
    ax.barh(
        y - 0.18, [100 * s["frac_moved_gt_0_01_vs_pfba_mean"] for s in rows], height=0.34,
        color=C[3], edgecolor="black", lw=0.4, label="reactions moved (%)", zorder=3
    )
    ax.set_yticks(y)
    ax.set_yticklabels([short[s["store"]] for s in rows])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("distance from the pFBA store")
    ax.grid(axis="x", lw=0.3, color="0.85", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.16), ncol=1, frameon=False, handlelength=1.0)
    return (
        "The same optimum, four points. Every store carries one growth value and a "
        "different flux that attains it."
    )


def panel_kcat(ax: Axes) -> str:
    """Median growth against the capacity scale, under each turnover tier."""
    style = {"measured": (C[0], "-"), "predicted": (C[2], "--")}
    for tier, (color, ls) in style.items():
        d = read(f"capacity_calibration_{tier}.json")
        scales = [r["capacity_scale"] for r in d["rows"]]
        growth = [r["growth_median"] for r in d["rows"]]
        ax.plot(scales, growth, ls, color=color, lw=0.9, marker="o", ms=2.0, label=f"{tier} tier")
        first = next(r["capacity_scale"] for r in d["rows"] if r["feasible_frac"] >= 1.0)
        ax.axvline(first, color=color, lw=0.5, ls=(0, (1, 2)))
    ax.set_xscale("log")
    ax.set_xlabel("capacity scale")
    ax.set_ylabel("median growth, mmol/gDW/h")
    ax.grid(lw=0.3, color="0.85", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.16), ncol=2, frameon=False, handlelength=1.4)
    return (
        "The dotted rule on each curve is the smallest scale at which every genotype's "
        "program is feasible; the measured tier is the looser box at every scale."
    )


def panel_scale(ax: Axes) -> str:
    """What sizes the store over the whole solid-growth build."""
    k = read("store_key_count.json")
    b = read("solid_growth_store_SC_ipm.json")
    rows = [
        ("records in the build", k["n_records"]),
        ("read the wild-type row", k["n_records_empty_gem_subset"]),
        ("distinct GEM subsets", k["n_distinct_gem_subsets"]),
        ("programs solved (keys)", k["n_distinct_keys"]),
        ("lethal under the solve", b["n_lethal"]),
        ("uncertified, stored zero", b["n_uncertified_zero_flux"]),
    ]
    colors = [C[5], C[5], C[0], C[0], C[1], C[3]]
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[1] for r in rows], height=0.68, color=colors, edgecolor="black", lw=0.4, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xscale("log")
    ax.set_xlim(1e3, 5e7)
    ax.set_xlabel("count")
    ax.grid(axis="x", lw=0.3, color="0.85", zorder=0)
    ax.set_axisbelow(True)
    return (
        "The transform's size is the program count, not the record count: 3.5 percent "
        "of the records, an hour of processor time."
    )


#: Panel order, left to right and top to bottom. Append here to grow the figure.
PANELS: list[tuple[str, Callable[[Axes], str]]] = [
    ("Comparators, four media", panel_fba),
    ("Published numbers reproduced", panel_fcl),
    ("Fold ensembles along the rounds", panel_ensembles),
    ("Families and their averages", panel_families),
    ("Genome-wide transfer", panel_genome),
    ("The solve across solvers", panel_transfer),
    ("What the stores carry", panel_stores),
    ("The capacity box by tier", panel_kcat),
    ("The store at the build's scale", panel_scale),
]


def main() -> None:
    plt.rcParams.update(
        {"font.family": "Arial", "font.size": 6, "svg.fonttype": "none", "axes.linewidth": 0.5}
    )
    ncol = 3
    nrow = (len(PANELS) + ncol - 1) // ncol
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(52.0 * nrow)),
    )
    flat = np.atleast_1d(axes).ravel()
    captions: list[str] = []
    for ax, (title, fn) in zip(flat, PANELS, strict=False):
        note = fn(ax)
        box(ax)
        ax.set_title(title, fontsize=6, pad=3)
        captions.append(note)
    for ax in flat[len(PANELS):]:
        ax.set_visible(False)
    for i, ax in enumerate(flat[: len(PANELS)]):
        ax.text(
            -0.02,
            1.24 if ax.get_legend() else 1.08,
            "abcdefghijklmnop"[i],
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            va="bottom",
            ha="right",
        )
    fig.subplots_adjust(left=0.165, right=0.945, top=0.90, bottom=0.075, wspace=1.0, hspace=0.78)
    os.makedirs(IMAGES, exist_ok=True)
    stem = osp.join(IMAGES, f"summary_{timestamp()}")
    savefig_true_size_svg(fig, stem + ".svg")
    fig.savefig(stem + ".png", dpi=300)
    for letter, (title, _), note in zip("abcdefghi", PANELS, captions, strict=False):
        print(f"  \\textbf{{{letter}}}, {title}. {note}")
    print(stem + ".svg")


if __name__ == "__main__":
    main()
