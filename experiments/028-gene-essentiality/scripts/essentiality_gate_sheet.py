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

#: One name per family, and every learned bar says its graph and its input. CGT is the
#: cell graph transformer; GCN is graph convolution over the union of the nine gene graphs
#: (physical, regulatory, TFLink and six STRING channels; ``graph_baselines.py``), named
#: with the node input it reads: GO is the GO multi-hot, "3 inputs" is ProtT5 + pathway
#: multi-hot + FBA ratios, the transformer's three. Keys are the labels of
#: ``summary_figure.py``; a label absent here is an error, not a pass-through.
GCN9 = "GCN (9 graphs)"
THREE = "ProtT5+pathway+FBA"
LABEL = {
    # a: the released held-out genes
    "GO graph + solve + rule counts": f"avg: CGT solve+counts, {GCN9} GO",
    "GO graph + cached solve head": f"avg: CGT cached solve, {GCN9} GO",
    "GO graph + layer-free solve": f"avg: CGT layer-free, {GCN9} GO",
    "transformer + solve + rule counts": "CGT, solve + rule counts",
    "transformer + layer-free solve": "CGT, layer-free solve",
    "graph convolution on GO": f"{GCN9}, GO input",
    "transformer + cached solve head": "CGT, cached solve head",
    "transformer + learned-ceiling LP": "CGT, LP in the training loop",
    "transformer + FBA ratios (reported)": "CGT, FBA ratio affine (reported)",
    "graph convolution, three inputs": f"{GCN9}, {THREE}",
    "MLP, three inputs": f"MLP, {THREE}, no graph",
    "DCell hierarchy, GO membership input": "DCell hierarchy, GO membership",
    "random forest on GO + FBA ratios": "random forest, GO + FBA ratios",
    "Flux Cone Learning, released": "Flux Cone Learning, released",
    "DCell, published gene input": "DCell, published gene input",
    "FBA growth ratio, YPD": "FBA growth ratio, YPD",
    # c: the genome-wide held-out genes
    "graph convolution, degree only": "GCN (physical), degree+clustering",
    "graph convolution, ProtT5": f"{GCN9}, ProtT5",
    "physical degree, nothing trained": "physical degree, nothing trained",
    "transformer + FBA ratios": "CGT, FBA ratio affine",
}
STATUS_SYMBOL = {"file": "●", "partial": "◐", "idea": "○"}
STATUS_WORD = {"file": "on file", "partial": "partial", "idea": "idea only"}
SECTION_SYMBOL = {"todo": ("✗", "#E53935"), "tent": ("■", "#E0A020"), "final": ("✓", "#43A047")}
SECTION_WORD = {"todo": "not done, agents may edit freely", "tent": "author-reviewed", "final": "final"}
SECTION_STATUS = "todo"
STATUS_MARKS = os.environ.get("GATE_STATUS_MARKS", "1") == "1"


def rename(label: str) -> str:
    assert label in LABEL, f"no gate label for {label!r}"
    return LABEL[label]


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


def status_key(fig, top_mm: float) -> None:
    """The key to the marks beside each letter, one line at the top of the sheet."""
    if not STATUS_MARKS:
        return
    h_mm = fig.get_size_inches()[1] * 25.4
    y = 1.0 - top_mm / h_mm
    x = 0.0
    fig.text(x, y, "section status", fontsize=5, color=GRAY, ha="left", va="top", style="italic")
    x += 0.075
    for key in ("todo", "tent", "final"):
        glyph, color = SECTION_SYMBOL[key]
        fig.text(x, y, glyph, fontsize=5, color=color, ha="left", va="top", family="DejaVu Sans")
        fig.text(x + 0.012, y, SECTION_WORD[key], fontsize=5, color=GRAY, ha="left", va="top")
        x += 0.03 + 0.0052 * len(SECTION_WORD[key])
    x += 0.03
    fig.text(x, y, "figure status", fontsize=5, color=GRAY, ha="left", va="top", style="italic")
    x += 0.07
    for key in ("file", "partial", "idea"):
        fig.text(x, y, STATUS_SYMBOL[key], fontsize=5, color=GRAY, ha="left", va="top", family="DejaVu Sans")
        fig.text(x + 0.012, y, STATUS_WORD[key], fontsize=5, color=GRAY, ha="left", va="top")
        x += 0.065


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


def yeast_cell(ax, x: float, y: float, bud: bool, dead: bool) -> None:
    """A yeast cell in data units: an oval, a bud when it grows, a skull when it does not."""
    fill = "white" if not dead else "#F5F5F5"
    ax.add_patch(Ellipse((x, y), 7.0, 5.2, facecolor=fill, edgecolor="black", lw=0.6, zorder=3))
    ax.add_patch(Ellipse((x + 2.2, y + 1.0), 2.0, 1.6, facecolor="white", edgecolor="black", lw=0.4, zorder=2))
    if bud:
        ax.add_patch(Ellipse((x + 4.4, y + 2.4), 3.2, 2.6, facecolor="white", edgecolor="black", lw=0.6, zorder=3))
    if dead:
        ax.text(x + 6.3, y, "☠", fontsize=12, family="DejaVu Sans", ha="left", va="center", zorder=4)


def orf_row(ax, y: float, deleted: bool) -> None:
    """A genome segment with three open reading frames; the middle one deleted when asked."""
    ax.plot([1.0, 19.0], [y, y], color="black", lw=0.6, zorder=1)
    boxes = [(2.0, 4.5), (8.0, 5.0), (14.5, 3.5)]
    for i, (x0, w) in enumerate(boxes):
        middle = i == 1
        if middle and deleted:
            ax.add_patch(Rectangle((x0, y - 1.1), w, 2.2, facecolor="white", edgecolor="black", lw=0.5, ls=(0, (1.5, 1.5)), zorder=2))
            ax.text(x0 + w / 2, y, "Δ", fontsize=6, ha="center", va="center", zorder=3)
        else:
            ax.add_patch(Rectangle((x0, y - 1.1), w, 2.2, facecolor="#FFF2CC" if middle else "#F5F5F5", edgecolor="#D6B656" if middle else "black", lw=0.5, zorder=2))
            if middle:
                ax.text(x0 + w / 2, y, "ORF", fontsize=5, ha="center", va="center", zorder=3)


def panel_concept(ax) -> None:
    """The label: delete one ORF; the gene is essential when the deletion strain cannot grow."""
    ax.set_xlim(0, 44)
    ax.set_ylim(0, 44)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0.5, 43.5, "genome", fontsize=6, ha="left", va="top")
    orf_row(ax, 40.0, deleted=False)
    ax.add_patch(FancyArrowPatch((10.5, 37.8), (10.5, 35.0), arrowstyle="-|>", mutation_scale=6, lw=0.6, color="black"))
    ax.text(12.0, 36.4, "delete one ORF", fontsize=6, ha="left", va="center")
    orf_row(ax, 32.5, deleted=True)
    ax.add_patch(FancyArrowPatch((10.5, 30.3), (10.5, 26.5), arrowstyle="-|>", mutation_scale=6, lw=0.6, color="black"))
    ax.text(12.0, 28.6, "deletion strain on YPD", fontsize=6, ha="left", va="center")
    yeast_cell(ax, 10.5, 22.0, bud=False, dead=False)
    ax.add_patch(FancyArrowPatch((14.5, 22.0), (25.5, 22.0), arrowstyle="-|>", mutation_scale=6, lw=0.6, color="black"))
    ax.add_patch(FancyArrowPatch((14.0, 20.0), (25.5, 10.0), arrowstyle="-|>", mutation_scale=6, lw=0.6, color="black"))
    yeast_cell(ax, 30.0, 22.0, bud=True, dead=False)
    ax.text(30.0, 17.5, "grows: nonessential", fontsize=6, ha="center", va="top")
    yeast_cell(ax, 30.0, 8.5, bud=False, dead=True)
    ax.text(30.0, 4.0, "no growth: essential", fontsize=6, ha="center", va="top")


def genome_split_counts() -> tuple[int, int]:
    """Train and validation gene counts of the split the genome-wide arms trained on."""
    split = osp.join(sf.RESULTS, "splits", sf.read("genome_heldout_readout.json")["split"])
    with gzip.open(split, "rt") as fh:
        d = json.load(fh)
    return len(d["train"]), len(d["val"])


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
    height = 147.0
    fig = sf.new_figure(height)
    h_mm = height
    # the status key on the first line, the color key under it
    status_key(fig, 1.0)
    fig.legend(
        handles=[
            sf.patch(sf.COMPARATOR, "comparator, nothing learned here"),
            sf.patch(sf.NO_MODULE, "learned, no metabolic column, no graph"),
            sf.patch(sf.SOLVE, "CGT reading the metabolic solve"),
            sf.patch(sf.GRAPH, "GCN over the 9 gene graphs (physical, regulatory, TFLink, 6 STRING)"),
            sf.patch(sf.AVERAGE, "CGT + GCN average, equal weights"),
            sf.patch("white", "LP solved inside training", hatch="/////"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0 - 4.0 / h_mm),
        ncol=3,
        frameon=False,
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=1.0,
        borderaxespad=0.0,
    )
    held = sf.read("fcl_comparison.json")
    n_held = next(s for s in held["scored"] if s["gene_set"] == "ours_scored" and s["seed"] is None)["n_genes"]

    # ---------------------------------------------------------------- row 1
    top = 20.0
    # a: the label
    ax = sf.place(fig, 4.0, top + 4.0, 40.0, 40.0)
    panel_concept(ax)
    ax.set_title("Essential: the single deletion cannot grow", pad=3, loc="left")
    letter_status(fig, 0.0, top, "a", "file")

    # b: every method on the FCL held-out genes
    bars = sf.leaderboard()
    for b in bars:
        b.label = rename(b.label)
    ax = sf.place(fig, 92.0, top + 4.0, 36.0, 44.0)
    sf.draw_bars(ax, bars)
    sf.auroc_axis(ax, 0.5, 1.0)
    ax.set_xlabel(f"AUROC, FCL held-out {n_held} genes, fold ensembles")
    ax.set_title(f"Best method, FCL held-out {n_held}", pad=3, loc="left")
    letter_status(fig, 46.0, top, "b", "file")

    # c: the three fixed in advance on the ten repeated splits
    resplit = sf.read("resplit_readout.json")["splits"]
    order = sorted(range(len(resplit)), key=lambda i: resplit[i]["average"])
    cx, cw, ch = 140.0, 38.0, 36.0
    panel_legend(
        fig,
        [marker(sf.SOLVE, "o", "CGT layer-free"), marker(sf.GRAPH, "s", "GCN, GO input"),
         marker(sf.AVERAGE, "D", "average")],
        cx + 1.0, top - 1.0, 2,
    )
    ax = sf.place(fig, cx, top + 4.0, cw, ch)
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
    ax.set_xlabel("outer split, sorted by the average")
    ax.set_ylabel("held-out AUROC, 3 seeds per point")
    ax.set_title("Fixed in advance, ten resplits", pad=3, loc="left")
    letter_status(fig, 132.0, top, "c", "file")

    # ---------------------------------------------------------------- row 2
    top = 84.0
    # d: trained on the released labeled genes, read genome-wide
    n_train, n_val = genome_split_counts()
    gh = sf.read("genome_heldout_readout.json")
    bars = sf.genome_transfer()
    for b in bars:
        b.label = rename(b.label)
    ax = sf.place(fig, 30.0, top + 4.0, 26.0, 36.0)
    sf.draw_bars(ax, bars)
    wrap_ticklabels(ax, 24, 5)
    ax.axvline(0.5, color="black", lw=0.5, ls=(0, (2, 2)), zorder=5)
    sf.auroc_axis(ax, 0.4, 0.8)
    ax.set_xlabel(f"AUROC, {gh['n_test_genes']:,} genes\nthe FCL file never labeled")
    ax.set_title(f"Trained on {n_train + n_val} FCL-labeled genes,\nread on {gh['n_test_genes']:,} others (dashed: chance)", pad=3, loc="left")
    letter_status(fig, 0.0, top, "d", "file")

    # e: every essential held-out gene, each family's miss
    genes = sorted(
        (r for r in sf.read("hard_genes.json")["genes"] if r["essential"] == 1),
        key=lambda r: -(r["misrank"][sf.HARD_ARMS[0][0]] + r["misrank"][sf.HARD_ARMS[1][0]]),
    )
    ex, ew = 70.0, 56.0
    panel_legend(
        fig,
        [marker(sf.SOLVE, "o", "CGT layer-free"), marker(sf.GRAPH, "s", "GCN, GO input"),
         Line2D([], [], ls="", label="bold name: FBA ratio 1.0, FBA calls it viable")],
        ex + 2.0, top - 1.0, 2,
    )
    ax = sf.place(fig, ex, top + 4.0, ew, 24.0)
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
    ax.set_title(f"The {len(genes)} essential FCL held-out genes, worst first", pad=3, loc="left")
    letter_status(fig, 60.0, top, "e", "file")

    # f: what the misranked genes share
    set_label, _, models = sf.ENRICH_SETS[0]
    fx = 156.0
    panel_legend(
        fig,
        [marker(sf.SOLVE, "o", "CGT layer-free"), marker(sf.GRAPH, "s", "GCN, GO input"),
         marker("white", "o", "hollow: q >= 0.1", hollow=True)],
        fx - 18.0, top - 1.0, 2,
    )
    ax = sf.place(fig, fx, top + 4.0, 21.0, 46.0)
    sf.panel_enrichment(ax, set_label, models)
    # the class names move from the right of the axes to the groups' own rows
    for t in list(ax.texts):
        t.remove()
    rows_y = ax.get_yticks()
    gap = (rows_y.max() + rows_y.min()) / 2.0
    ax.text(0.98, rows_y.max() + 0.95, "essential, ranked low", transform=ax.get_yaxis_transform(), fontsize=5, ha="right", va="bottom", style="italic")
    ax.text(0.98, gap, "nonessential, ranked high", transform=ax.get_yaxis_transform(), fontsize=5, ha="right", va="center", style="italic")
    ax.set_ylim(rows_y.min() - 0.7, rows_y.max() + 2.0)
    wrap_ticklabels(ax, 30, 5)
    ax.set_xlabel("P(annotated gene\nmisranked worse)")
    ax.set_title(f"Misranked, held-out {n_held}", pad=3, loc="right")
    letter_status(fig, 130.0, top, "f", "file")

    sf.save(fig, "essentiality_gate_sheet")


if __name__ == "__main__":
    main()
