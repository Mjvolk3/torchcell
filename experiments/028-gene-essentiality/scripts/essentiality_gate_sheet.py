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
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

HERE = osp.dirname(osp.abspath(__file__))
spec = importlib.util.spec_from_file_location("summary_figure", osp.join(HERE, "summary_figure.py"))
sf = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(sf)

WIDTH_MM = sf.WIDTH_MM
GRAY = sf.PLOT_PALETTE[5]

#: The one name per family. Every bar label and legend entry passes through this map.
RENAME = {
    "transformer": "CGT",
    "graph convolution": "GCN",
    "GO graph": "GCN on GO",
}
STATUS_SYMBOL = {"file": "●", "partial": "◐", "idea": "○"}
STATUS_WORD = {"file": "on file", "partial": "partial", "idea": "idea only"}
SECTION_SYMBOL = {"todo": ("✗", "#E53935"), "tent": ("■", "#E0A020"), "final": ("✓", "#43A047")}
SECTION_WORD = {"todo": "not done, agents may edit freely", "tent": "author-reviewed", "final": "final"}
SECTION_STATUS = "todo"
STATUS_MARKS = os.environ.get("GATE_STATUS_MARKS", "1") == "1"


def rename(label: str) -> str:
    for old, new in RENAME.items():
        label = label.replace(old, new)
    return label


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


def status_key(fig) -> None:
    if not STATUS_MARKS:
        return
    x = 0.0
    fig.text(x, 0.004, "section status", fontsize=5, color=GRAY, ha="left", va="bottom", style="italic")
    x += 0.075
    for key in ("todo", "tent", "final"):
        glyph, color = SECTION_SYMBOL[key]
        fig.text(x, 0.004, glyph, fontsize=5, color=color, ha="left", va="bottom", family="DejaVu Sans")
        fig.text(x + 0.012, 0.004, SECTION_WORD[key], fontsize=5, color=GRAY, ha="left", va="bottom")
        x += 0.03 + 0.0052 * len(SECTION_WORD[key])
    x += 0.03
    fig.text(x, 0.004, "figure status", fontsize=5, color=GRAY, ha="left", va="bottom", style="italic")
    x += 0.07
    for key in ("file", "partial", "idea"):
        fig.text(x, 0.004, STATUS_SYMBOL[key], fontsize=5, color=GRAY, ha="left", va="bottom", family="DejaVu Sans")
        fig.text(x + 0.012, 0.004, STATUS_WORD[key], fontsize=5, color=GRAY, ha="left", va="bottom")
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
    height = 160.0
    fig = sf.new_figure(height)
    sf.top_legend(
        fig,
        [
            sf.patch(sf.COMPARATOR, "comparator, nothing learned here"),
            sf.patch(sf.NO_MODULE, "learned, no metabolic column, no graph"),
            sf.patch(sf.SOLVE, "CGT reading the metabolic solve"),
            sf.patch(sf.GRAPH, "GCN on the gene graphs"),
            sf.patch(sf.AVERAGE, "CGT + GCN average, equal weights"),
            sf.patch("white", "LP solved inside training", hatch="/////"),
        ],
        ncol=3,
    )
    left_label, right_block = 47.0, 100.0

    # a: every method on the released held-out genes
    top = 15.0
    bars = sf.leaderboard()
    for b in bars:
        b.label = rename(b.label)
    ax = sf.place(fig, left_label, top, 40.0, 44.0)
    sf.draw_bars(ax, bars)
    sf.auroc_axis(ax, 0.5, 1.0)
    ax.set_xlabel("AUROC, 195 held-out genes, fold ensembles")
    ax.set_title("Best method, released 195 genes", pad=3, loc="left")
    letter_status(fig, 0.0, top, "a", "file")

    # b: the three fixed in advance on the ten repeated splits, 52 mm wide
    resplit = sf.read("resplit_readout.json")["splits"]
    order = sorted(range(len(resplit)), key=lambda i: resplit[i]["average"])
    bx, bw, bh = right_block + 12.0, 52.0, 36.0
    panel_legend(
        fig,
        [marker(sf.SOLVE, "o", "CGT + solve"), marker(sf.GRAPH, "s", "GCN on GO"),
         marker(sf.AVERAGE, "D", "CGT + GCN average")],
        bx + 2.0, top - 1.0, 3,
    )
    ax = sf.place(fig, bx, top + 5.0, bw, bh)
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
    ax.set_xlabel("repeated outer split, sorted by the average")
    ax.set_ylabel("held-out AUROC")
    letter_status(fig, right_block, top, "b", "file")

    # c: what transfers off the metabolic genes
    top = 74.0
    bars = sf.genome_transfer()
    for b in bars:
        b.label = rename(b.label)
    ax = sf.place(fig, left_label, top, 40.0, 28.0)
    sf.draw_bars(ax, bars)
    ax.axvline(0.5, color="black", lw=0.5, ls=(0, (2, 2)), zorder=5)
    sf.auroc_axis(ax, 0.4, 0.8)
    ax.set_xlabel("AUROC, 4,674 genome-wide held-out genes")
    ax.set_title("What transfers off the metabolic genes", pad=3, loc="left")
    letter_status(fig, 0.0, top, "c", "file")

    # d: what the misranked genes share, with its legend above
    set_label, title, models = sf.ENRICH_SETS[0]
    dx = right_block + 44.0
    panel_legend(
        fig,
        [marker(sf.SOLVE, "o", "CGT + solve"), marker(sf.GRAPH, "s", "GCN on GO"),
         marker("white", "o", "hollow: q at or above 0.1", hollow=True)],
        right_block + 10.0, top - 1.0, 3,
    )
    ax = sf.place(fig, dx, top + 5.0, 20.0, 28.0)
    sf.panel_enrichment(ax, set_label, models)
    ax.set_title("Misranked genes, released 195", pad=3, loc="right")
    letter_status(fig, right_block, top, "d", "file")

    # e: every essential held-out gene, at half width, with its legend above
    top = 118.0
    genes = sorted(
        (r for r in sf.read("hard_genes.json")["genes"] if r["essential"] == 1),
        key=lambda r: -(r["misrank"][sf.HARD_ARMS[0][0]] + r["misrank"][sf.HARD_ARMS[1][0]]),
    )
    panel_legend(
        fig,
        [marker(sf.SOLVE, "o", "CGT + solve"), marker(sf.GRAPH, "s", "GCN on GO")],
        14.0, top - 1.0, 2,
    )
    ax = sf.place(fig, 14.0, top + 5.0, 74.0, 20.0)
    x = np.arange(len(genes))
    for arm, color, shape in sf.HARD_ARMS:
        ax.plot(x, [r["misrank"][arm] for r in genes], ls="", marker=shape, ms=2.6, mfc=color, mec=color, zorder=3)
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
    ax.set_ylabel("misrank")
    letter_status(fig, 0.0, top, "e", "file")

    status_key(fig)
    sf.save(fig, "essentiality_gate_sheet")


if __name__ == "__main__":
    main()
