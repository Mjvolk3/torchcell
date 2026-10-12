# experiments/028-gene-essentiality/scripts/essentiality_gate_sheet.py
# [[experiments.028-gene-essentiality.scripts.essentiality_gate_sheet]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/essentiality_gate_sheet
"""The gene-essentiality gate's Figure 1, re-laid out from the summary figure's readers.

The 028 summary sheet (``summary_figure.py``, goal 1) carries the right panels and the
wrong proportions for a gate: its repeated-splits panel fills a 72 mm box, the per-gene
panel runs the full width, neither carries a legend, and the labels call one family
"transformer" and "family 1" by turns. This sheet reuses every reader of
``summary_figure.py`` unchanged (so every number is the same committed result file) and
changes only the layout and the words:

- b (the ten repeated splits) at 52 mm, with its own legend above it;
- e (every essential held-out gene) at half width, with its own legend;
- every legend above its panel, nothing inside an axes but data;
- one name per model family everywhere: the cell graph transformer is CGT, the second
  family is GCN on the gene graphs, their average is "CGT + GCN average";
- the metabolism gate's status marks beside each panel letter (section status, figure
  status) and a status key at the foot, so a panel reads like a section.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD \\
        python experiments/028-gene-essentiality/scripts/essentiality_gate_sheet.py
"""

from __future__ import annotations

import importlib.util
import os
import os.path as osp

import matplotlib

matplotlib.use("Agg")
import gzip  # noqa: E402
import json  # noqa: E402
import textwrap  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Ellipse, FancyArrowPatch, Rectangle  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

HERE = osp.dirname(osp.abspath(__file__))
spec = importlib.util.spec_from_file_location("summary_figure", osp.join(HERE, "summary_figure.py"))
sf = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(sf)

WIDTH_MM = sf.WIDTH_MM
GRAY = sf.PLOT_PALETTE[5]

#: The rows the main-text sheet keeps, by their ``summary_figure.py`` label, and the name
#: a general reader sees. The study searched many module forms (cached solve, rule counts,
#: an LP inside training); the sheet shows one row per idea: the two published methods,
#: a hierarchy, a forest, a learned model with no graph, each family in its kept form,
#: and the average. CGT is the cell graph transformer reading sequence, pathway and, where
#: named, the metabolic solve (a pFBA solve done offline, its log(1+|v|) flux vector
#: through a zero-initialized affine on the logit); GCN is graph convolution over the nine
#: gene graphs (physical, regulatory, TFLink, six STRING) reading the GO annotation.
LABEL = {
    # b: the FCL held-out genes, best first as the leaderboard orders them
    "GO graph + layer-free solve": "average of CGT + solve and GCN",
    "transformer + layer-free solve": "CGT + metabolic solve",
    "graph convolution on GO": "GCN on gene graphs, GO input",
    "transformer + FBA ratios (reported)": "CGT + FBA growth ratio",
    "MLP, three inputs": "MLP, no graph: sequence, pathway, FBA",
    "random forest on GO + FBA ratios": "random forest: GO, FBA",
    "Flux Cone Learning, released": "Flux Cone Learning, published predictions",
    "DCell, published gene input": "DCell, published inputs",
    "FBA growth ratio, YPD": "FBA growth ratio, published predictions",
    # d: the genome-wide genes
    "physical degree, nothing trained": "physical degree, nothing trained",
    "graph convolution, ProtT5": "GCN on gene graphs, sequence input",
}
KEEP_B = [k for k in LABEL if k not in ("physical degree, nothing trained", "graph convolution, ProtT5")]
#: Panel d's gutter is 26 mm, so its rows take shorter names than panel b's.
LABEL_D = {
    "physical degree, nothing trained": "physical degree, untrained",
    "graph convolution, ProtT5": "GCN, sequence input",
    "graph convolution on GO": "GCN, GO input",
    "transformer + layer-free solve": "CGT + metabolic solve",
    "MLP, three inputs": "MLP: sequence, pathway, FBA",
}
KEEP_D = [
    "physical degree, nothing trained",
    "graph convolution, ProtT5",
    "graph convolution on GO",
    "transformer + layer-free solve",
    "MLP, three inputs",
]
CGT_NAME = "CGT + metabolic solve"
GCN_NAME = "GCN, GO input"
STATUS_SYMBOL = {"file": "●", "partial": "◐", "idea": "○"}
SECTION_SYMBOL = {"todo": ("✗", "#E53935"), "tent": ("■", "#E0A020"), "final": ("✓", "#43A047")}
SECTION_STATUS = "todo"
STATUS_MARKS = os.environ.get("GATE_STATUS_MARKS", "1") == "1"
#: The yeast cell drawing of Figure 1 (fig1_yeast_cells.py), the orange one.
CELL_PNG = osp.join(sf.IMAGES, "yeast_cell_orange.png")


def keep(bars, keys: list[str], names: dict[str, str]):
    """The named rows of a bar list, renamed, in the order the list gives them."""
    have = {b.label: b for b in bars}
    missing = [k for k in keys if k not in have]
    assert not missing, f"rows absent from the result readers: {missing}"
    out = [have[k] for k in bars_order(bars, keys)]
    for b in out:
        b.label = names[b.label]
    return out


def bars_order(bars, keys: list[str]) -> list[str]:
    return [b.label for b in bars if b.label in keys]


def letter_status(fig, x_mm: float, top_mm: float, ch: str, status: str) -> None:
    """A bold panel letter followed by the section-status and figure-status glyphs."""
    h_mm = fig.get_size_inches()[1] * 25.4
    x, y = x_mm / WIDTH_MM, 1.0 - (top_mm - 1.0) / h_mm
    fig.text(x, y, ch, fontsize=8, fontweight="bold", va="bottom", ha="left")
    if not STATUS_MARKS:
        return
    glyph, color = SECTION_SYMBOL[SECTION_STATUS]
    fig.text(x + 3.2 / WIDTH_MM, y + 0.3 / h_mm, glyph, fontsize=5, color=color, va="bottom", ha="left", family="DejaVu Sans")
    fig.text(x + 5.6 / WIDTH_MM, y + 0.3 / h_mm, STATUS_SYMBOL[status], fontsize=5, color=GRAY, va="bottom", ha="left", family="DejaVu Sans")


def panel_legend(fig, handles, x_mm: float, top_mm: float, ncol: int) -> None:
    """A legend whose lower-left corner sits just above a panel's top-left corner."""
    h_mm = fig.get_size_inches()[1] * 25.4
    fig.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(x_mm / WIDTH_MM, 1.0 - (top_mm - 0.5) / h_mm),
        ncol=ncol,
        frameon=False,
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=1.0,
        borderaxespad=0.0,
        fontsize=6,
    )


def marker(color: str, shape: str, label: str, hollow: bool = False) -> Line2D:
    return Line2D([], [], marker=shape, ls="", ms=3.2, mfc="white" if hollow else color,
                  mec="black" if hollow else color, mew=0.7, label=label)


def wrap_ticklabels(ax, width: int, fontsize: float) -> None:
    """Fold the y tick labels at ``width`` characters; a term name is never shortened."""
    ax.set_yticklabels(
        [textwrap.fill(t.get_text(), width) for t in ax.get_yticklabels()], fontsize=fontsize
    )


def arrow(ax, p0, p1) -> None:
    ax.annotate("", xy=p1, xytext=p0, arrowprops={"arrowstyle": "-|>", "lw": 0.8, "color": "black",
                                                 "shrinkA": 0, "shrinkB": 0, "mutation_scale": 8})


def cell(ax, x: float, y: float, w: float) -> None:
    """The Figure 1 yeast cell, centered at (x, y) in data units, ``w`` wide."""
    img = plt.imread(CELL_PNG)
    h = w * img.shape[0] / img.shape[1]
    ax.imshow(img, extent=(x - w / 2, x + w / 2, y - h / 2, y + h / 2), zorder=3, interpolation="lanczos")


def orf_row(ax, y: float, deleted: bool) -> None:
    """A genome segment with three open reading frames; the middle one deleted when asked."""
    ax.plot([1.0, 19.0], [y, y], color="black", lw=0.6, zorder=1)
    for i, (x0, w) in enumerate([(2.0, 4.5), (8.0, 5.0), (14.5, 3.5)]):
        middle = i == 1
        if middle and deleted:
            ax.add_patch(Rectangle((x0, y - 1.2), w, 2.4, facecolor="white", edgecolor="black", lw=0.5, ls=(0, (1.5, 1.5)), zorder=2))
            ax.text(x0 + w / 2, y, "Δ", fontsize=6, ha="center", va="center", zorder=3)
        else:
            ax.add_patch(Rectangle((x0, y - 1.2), w, 2.4, facecolor="#FFF2CC" if middle else "#F5F5F5", edgecolor="#D6B656" if middle else "black", lw=0.5, zorder=2))
            if middle:
                ax.text(x0 + w / 2, y, "ORF", fontsize=5, ha="center", va="center", zorder=3)


def panel_concept(ax) -> None:
    """The label: delete one ORF; the gene is essential when the deletion strain cannot grow."""
    ax.set_xlim(0, 54)
    ax.set_ylim(0, 44)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(1.0, 43.5, "genome", fontsize=6, ha="left", va="top")
    orf_row(ax, 40.0, deleted=False)
    arrow(ax, (10.5, 38.4), (10.5, 35.2))
    ax.text(12.5, 36.8, "delete one ORF", fontsize=6, ha="left", va="center")
    orf_row(ax, 33.0, deleted=True)
    arrow(ax, (10.5, 31.4), (10.5, 26.0))
    cell(ax, 10.5, 22.0, 9.0)
    ax.text(10.5, 18.2, "deletion strain\non YPD", fontsize=6, ha="center", va="top")
    arrow(ax, (15.5, 23.5), (28.5, 28.0))
    arrow(ax, (15.5, 20.5), (28.5, 11.5))
    cell(ax, 33.5, 28.5, 9.0)
    cell(ax, 42.5, 30.5, 6.0)
    ax.text(37.0, 24.0, "grows: nonessential", fontsize=6, ha="center", va="top")
    cell(ax, 33.5, 10.5, 9.0)
    ax.text(39.5, 10.5, "☠", fontsize=11, family="DejaVu Sans", ha="left", va="center", zorder=4)
    ax.text(37.0, 6.0, "no growth: essential", fontsize=6, ha="center", va="top")


def genome_split_counts() -> tuple[int, int]:
    """Train and validation gene counts of the split the genome-wide arms trained on."""
    split = osp.join(sf.RESULTS, "splits", sf.read("genome_heldout_readout.json")["split"])
    with gzip.open(split, "rt") as fh:
        d = json.load(fh)
    return len(d["train"]), len(d["val"])


def enrichment_split(ax_top, ax_bottom, set_label: str, models) -> None:
    """The enrichment panel as two stacked axes, one per gene class, same x axis."""
    rows = sf.enrichment_rows(set_label, models)
    for ax, cls in ((ax_top, sf.CLASSES[0]), (ax_bottom, sf.CLASSES[1])):
        sub = [r for r in rows if r["class"] == cls]
        y = np.arange(len(sub))[::-1].astype(float)
        for yi, row in zip(y, sub, strict=True):
            for k, (model, color) in enumerate(models):
                got = row["by_model"][model]
                if got is None:
                    continue
                effect, q = got
                ax.scatter(effect, yi + (0.18 if k == 0 else -0.18), s=9, marker="o" if k == 0 else "s",
                           facecolor=color if q < 0.1 else "white", edgecolor=color, lw=0.7, zorder=4)
        ax.axvline(0.5, color="black", lw=0.5, ls=(0, (2, 2)), zorder=2)
        ax.set_yticks(y)
        ax.set_yticklabels([textwrap.fill(r["name"], 38) for r in sub], fontsize=5, linespacing=0.9)
        ax.set_ylim(-0.6, len(sub) + 1.0)
        ax.text(0.97, 0.96, cls, transform=ax.transAxes, fontsize=5, style="italic", ha="right", va="top", zorder=5,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5})
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0.2, 1.0)
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))
        ax.tick_params(axis="x", which="minor", length=0)
        ax.grid(axis="x", which="both", lw=0.3, color="0.85", zorder=0)
        ax.set_axisbelow(True)
    ax_top.tick_params(axis="x", labelbottom=False)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "svg.fonttype": "none",
            "axes.linewidth": 0.5,
            "hatch.linewidth": 0.4,
            "axes.titlesize": 6,
            "legend.fontsize": 6,
        }
    )
    height = 140.0
    fig = sf.new_figure(height)
    fig.legend(
        handles=[
            sf.patch(sf.COMPARATOR, "published method, nothing learned here"),
            sf.patch(sf.NO_MODULE, "learned, no graph, no metabolic model"),
            sf.patch(sf.SOLVE, "CGT, cell graph transformer, reading the metabolic solve"),
            sf.patch(sf.GRAPH, "GCN, graph convolution over the nine gene graphs"),
            sf.patch(sf.AVERAGE, "average of the two, equal weights"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=3,
        frameon=False,
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=1.0,
        borderaxespad=0.2,
    )
    held = sf.read("fcl_comparison.json")
    n_held = next(s for s in held["scored"] if s["gene_set"] == "ours_scored" and s["seed"] is None)["n_genes"]
    # the grid: three columns whose axes share a right edge, two rows whose axes share
    # top and bottom; letters on one line, legends on the next, titles on the axes
    right = (56.0, 118.0, 177.0)
    lett = (0.0, 61.0, 123.0)
    H = 44.0
    LEG = 4.0  # a legend's baseline sits this far above its axes, clear of the title

    # ---------------------------------------------------------------- row 1
    top, ax_top = 12.0, 20.0
    ax = sf.place(fig, 2.0, ax_top, 54.0, H)
    panel_concept(ax)
    letter_status(fig, lett[0], top, "a", "file")

    bars = keep(sf.leaderboard(), KEEP_B, LABEL)
    ax = sf.place(fig, right[1] - 26.0, ax_top, 26.0, H)
    sf.draw_bars(ax, bars)
    ax.tick_params(axis="y", labelsize=5)
    sf.auroc_axis(ax, 0.5, 1.0)
    ax.set_xlabel(f"AUROC, FCL held-out {n_held} genes")
    ax.set_title(f"Held-out genes of the published method", pad=3, loc="right")
    letter_status(fig, lett[1], top, "b", "file")

    resplit = sf.read("resplit_readout.json")["splits"]
    order = sorted(range(len(resplit)), key=lambda i: resplit[i]["average"])
    panel_legend(fig, [marker(sf.SOLVE, "o", CGT_NAME), marker(sf.GRAPH, "s", GCN_NAME), marker(sf.AVERAGE, "D", "average")],
                 lett[2] + 8.0, ax_top - LEG, 3)
    ax = sf.place(fig, right[2] - 48.0, ax_top, 48.0, H)
    for key, color, shape in (("layer_free", sf.SOLVE, "o"), ("go_convolution", sf.GRAPH, "s"), ("average", sf.AVERAGE, "D")):
        ax.plot(range(len(order)), [resplit[i][key] for i in order], ls="-", lw=0.5, marker=shape,
                ms=3.0, color=color, mec="black", mew=0.4, zorder=3)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([str(resplit[i]["resplit"]) for i in order])
    ax.set_ylim(0.7, 1.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.tick_params(axis="y", which="minor", length=0)
    ax.grid(axis="y", which="both", lw=0.3, color="0.85", zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("repeated split, sorted by the average")
    ax.set_ylabel("held-out AUROC, 3 seeds per point")
    ax.set_title("Ten repeated splits, models fixed in advance", pad=3, loc="right")
    letter_status(fig, lett[2], top, "c", "file")

    # ---------------------------------------------------------------- row 2
    top, ax_top, H = 73.0, 81.0, 40.0
    n_train, n_val = genome_split_counts()
    gh = sf.read("genome_heldout_readout.json")
    bars = keep(sf.genome_transfer(), KEEP_D, LABEL_D)
    ax = sf.place(fig, right[0] - 26.0, ax_top, 26.0, H)
    sf.draw_bars(ax, bars)
    ax.tick_params(axis="y", labelsize=5)
    ax.axvline(0.5, color="black", lw=0.5, ls=(0, (2, 2)), zorder=5)
    sf.auroc_axis(ax, 0.4, 0.8)
    ax.set_xlabel(f"AUROC, {gh['n_test_genes']:,} genes outside\nthe metabolic model (dashed: chance)")
    ax.set_title(f"Trained on {n_train + n_val} metabolic genes, read on the rest", pad=3, loc="right")
    letter_status(fig, lett[0], top, "d", "file")

    genes = sorted(
        (r for r in sf.read("hard_genes.json")["genes"] if r["essential"] == 1),
        key=lambda r: -(r["misrank"][sf.HARD_ARMS[0][0]] + r["misrank"][sf.HARD_ARMS[1][0]]),
    )
    panel_legend(fig, [marker(sf.SOLVE, "o", CGT_NAME), marker(sf.GRAPH, "s", GCN_NAME)], lett[1] + 8.0, ax_top - LEG, 2)
    ax = sf.place(fig, right[1] - 50.0, ax_top, 50.0, H)
    x = np.arange(len(genes))
    for arm, color, shape in sf.HARD_ARMS:
        ax.plot(x, [r["misrank"][arm] for r in genes], ls="", marker=shape, ms=2.4, mfc=color, mec=color, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([r["name"] for r in genes], rotation=90, ha="center", va="top", fontsize=5)
    for tick, row in zip(ax.get_xticklabels(), genes, strict=True):
        if row["fba_ratio_ypd"] is not None and row["fba_ratio_ypd"] >= 0.999:
            tick.set_fontweight("bold")
    ax.set_xlim(-0.8, len(genes) - 0.2)
    ax.set_ylim(0.0, 0.8)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(axis="y", which="minor", length=0)
    ax.grid(axis="y", which="both", lw=0.3, color="0.85", zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylabel("share of nonessential\ngenes scored above")
    ax.set_xlabel("essential gene; bold: FBA calls its deletion viable")
    ax.set_title(f"The {len(genes)} essential held-out genes, worst first", pad=3, loc="right")
    letter_status(fig, lett[1], top, "e", "file")

    set_label, _, models = sf.ENRICH_SETS[0]
    panel_legend(fig, [marker(sf.SOLVE, "o", CGT_NAME), marker(sf.GRAPH, "s", GCN_NAME),
                       marker("white", "o", "hollow: q >= 0.1", hollow=True)], lett[2] + 8.0, ax_top - LEG, 3)
    gap = 3.0
    ax1 = sf.place(fig, right[2] - 21.0, ax_top, 21.0, (H - gap) / 2)
    ax2 = sf.place(fig, right[2] - 21.0, ax_top + (H + gap) / 2, 21.0, (H - gap) / 2)
    enrichment_split(ax1, ax2, set_label, models)
    ax2.set_xlabel("P(gene with the feature\nmisranked worse)")
    ax1.set_title("What the misranked share", pad=3, loc="right")
    letter_status(fig, lett[2], top, "f", "file")

    sf.save(fig, "essentiality_gate_sheet")


if __name__ == "__main__":
    main()
