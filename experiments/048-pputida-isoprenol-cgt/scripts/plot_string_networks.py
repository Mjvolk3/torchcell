# experiments/048-pputida-isoprenol-cgt/scripts/plot_string_networks.py
# [[experiments.048-pputida-isoprenol-cgt.scripts.plot_string_networks]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/048-pputida-isoprenol-cgt/scripts/plot_string_networks
"""The six STRING v12.5 channels over the campaign's CRISPRi target genes.

One panel per evidence channel, every panel on **one shared node layout**, so
the channels are comparable by eye: a gene sits in the same place in all six and
what changes between panels is only which edges exist.

The node set is the 121 KT2440 genes the campaign actually targeted, not all
5,729 in the cell graph. A 5,729-node layout is a hairball, and the targeted
subgraph is the part of the gene graph that carries the perturbation signal:
every CRISPRi guide in the 465 campaign records names one of these genes.

**Layout and drawing are deliberately thresholded differently.** The layout is
computed from the union of all six channels at every nonzero score, which is the
graph the model reads: 1,998 edges over the 121 genes, and all but one gene in a
single connected component, so the layout is one object rather than a scatter of
fragments. Edges are then drawn only above a stated score, because 1,998 edges
over 121 nodes is a solid block of ink. Spring weights are the channel score
rescaled to [0, 1]; passing raw STRING scores (up to 999) as spring weights
collapses the graph to a dot.

Node area is the number of designs a gene appears in, so the genes the campaign
returned to again and again (cyoD at 125 designs, mqo-I at 88) read as the hubs
they are. Node fill marks the 10 of 121 targets that catalyze no reaction in the
isoprenol GEM, which is exactly the set the metabolic module cannot see.

Outputs (PNG and true-size SVG) go to ``ASSET_IMAGES_DIR/048-pputida-isoprenol-cgt/``
and the per-channel edge counts to ``results/string_channel_edge_counts.csv``.
"""

from __future__ import annotations

import argparse
import math
import os
import os.path as osp

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from dotenv import load_dotenv
from matplotlib.lines import Line2D

from torchcell.graph.string_db import (
    STRING_SCORE_CHANNELS,
    StringTaxonMirror,
    string_channel_graphs,
)
from torchcell.metabolism.bigg_gem import BiggGEM
from torchcell.sequence import GeneSet
from torchcell.timestamp import timestamp
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    apply_paper_style,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()

ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
EXPERIMENT_DIR = osp.join(EXPERIMENT_ROOT, "048-pputida-isoprenol-cgt")
RESULTS = osp.join(EXPERIMENT_DIR, "results")
RECORD_TABLE = osp.join(RESULTS, "isoprenol_cgt_record_table.csv")

KT2440_TAXON = 160488
STRING_RELEASE = "12.5"
STRING_ROOT = "/bulk/torchcell-string"
ISOPRENOL_MODEL_ID = "pputida_isoprenol"

#: The channel score a drawn edge must exceed. STRING calls 400 medium
#: confidence and 700 high; the figure draws medium and up.
EDGE_SCORE_FLOOR = 400

#: Channel titles, since the raw keys are not display text. "cooccurence" is
#: STRING's own spelling of the column and stays as the source writes it.
CHANNEL_TITLES = {
    "neighborhood": "Neighborhood",
    "fusion": "Fusion",
    "cooccurence": "Co-occurrence",
    "coexpression": "Co-expression",
    "experimental": "Experimental",
    "database": "Database",
}

#: The genes to name in the figure: the campaign's most-used targets. Named by
#: their STRING preferred name where one exists, since PP_0815 says nothing and
#: cyoD says cytochrome o ubiquinol oxidase.
N_LABELED_GENES = 8

#: Minimum angular separation between two labeled genes, in degrees. Below this
#: the names collide on the rim.
MIN_LABEL_SEPARATION_DEG = 7.0


def target_genes() -> tuple[list[str], dict[str, int]]:
    """The campaign's CRISPRi target genes, and how many designs name each."""
    table = pd.read_csv(RECORD_TABLE)
    campaign = table[table.is_campaign]
    counts: dict[str, int] = {}
    for key in campaign["design_key"]:
        if key == "no_guide":
            continue
        for gene in str(key).split("|"):
            counts[gene] = counts.get(gene, 0) + 1
    return sorted(counts), counts


def shared_layout(genes: list[str]) -> dict[str, tuple[float, float]]:
    """One position per gene: a circle in genome order, clockwise from twelve.

    The position is a pure function of the locus tag, which is what makes this
    layout usable as a fixed frame. There is no seed, no optimizer, and no
    dependence on which edges happen to pass a threshold, so the six panels and
    any future panel over the same gene set are directly comparable, and a rerun
    reproduces the figure exactly.

    A force-directed layout was tried first and rejected: 121 genes carrying
    1,998 union edges collapse into a blob that fills a fifth of the panel while
    a few pendant genes stretch the frame, and the result is neither readable
    nor stable under a change of threshold.

    Genome order earns its place rather than being an arbitrary circle: ``PP_``
    tags are assigned along the chromosome, so a short chord is a pair of genes
    that sit near each other on the genome. The neighborhood channel scores
    conserved genomic adjacency, so it should and does draw short chords, while
    database and co-expression reach across the circle.
    """
    positions: dict[str, tuple[float, float]] = {}
    for index, gene in enumerate(genes):
        angle = math.pi / 2.0 - 2.0 * math.pi * index / len(genes)
        positions[gene] = (math.cos(angle), math.sin(angle))
    return positions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-floor", type=int, default=EDGE_SCORE_FLOOR)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-timestamp", action="store_true")
    args = parser.parse_args()

    apply_paper_style()
    genes, design_counts = target_genes()
    gene_set = GeneSet(genes)
    mirror = StringTaxonMirror(
        root=STRING_ROOT, taxon=KT2440_TAXON, release=STRING_RELEASE
    )
    # Two reads of the same mirror: everything for the layout, the thresholded
    # subset for the ink.
    full = string_channel_graphs(
        mirror, gene_set, channels=STRING_SCORE_CHANNELS, above_score=0
    )
    drawn = string_channel_graphs(
        mirror, gene_set, channels=STRING_SCORE_CHANNELS, above_score=args.score_floor
    )
    positions = shared_layout(genes)
    names = mirror.preferred_names()

    union = nx.Graph()
    for graph in drawn.values():
        union.add_edges_from(graph.edges())

    gem_genes = set(BiggGEM(ISOPRENOL_MODEL_ID).model.genes.list_attr("id"))
    off_model = [gene for gene in genes if gene not in gem_genes]

    sizes = {gene: 3.0 + 0.55 * design_counts[gene] for gene in genes}
    # Label the most-used targets, but skip one that sits within
    # MIN_LABEL_SEPARATION_DEG of a label already placed. The campaign's top
    # targets include the four-gene cyo operon, whose members are consecutive
    # locus tags and therefore consecutive points on the circle, so labeling all
    # of them prints four names on top of each other.
    labeled: list[str] = []
    step = 360.0 / len(genes)
    for gene in sorted(genes, key=lambda g: -design_counts[g]):
        index = genes.index(gene)
        if all(
            min(
                abs(index - genes.index(other)),
                len(genes) - abs(index - genes.index(other)),
            )
            * step
            >= MIN_LABEL_SEPARATION_DEG
            for other in labeled
        ):
            labeled.append(gene)
        if len(labeled) == N_LABELED_GENES:
            break

    rows = []
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(126.0)),
        constrained_layout=True,
    )
    for index, channel in enumerate(STRING_SCORE_CHANNELS):
        ax = axes.flat[index]
        graph = drawn[channel]
        color = PLOT_PALETTE[index]

        # The union of every channel as a faint backdrop, so each panel shows
        # what this channel contributes against what any channel knows.
        nx.draw_networkx_edges(
            union, positions, ax=ax, edge_color="#000000", alpha=0.055, width=0.25
        )
        weights = [float(d["weight"]) for _, _, d in graph.edges(data=True)]
        span = max(1000 - args.score_floor, 1)
        alphas = [
            0.30 + 0.60 * min((w - args.score_floor) / span, 1.0) for w in weights
        ]
        nx.draw_networkx_edges(
            graph,
            positions,
            ax=ax,
            edge_color=[matplotlib.colors.to_rgba(color, a) for a in alphas],
            width=0.45,
            connectionstyle="arc3,rad=0.18",
        )
        connected = [gene for gene in genes if gene in graph]
        # Every target is drawn in every panel, so a gene with no edge in this
        # channel reads as an unconnected point rather than vanishing. A gene
        # this channel does connect gets a black rim.
        rim = nx.Graph()
        rim.add_nodes_from(genes)
        nx.draw_networkx_nodes(
            rim,
            positions,
            nodelist=genes,
            node_size=[sizes[g] for g in genes],
            node_color=[
                PLOT_PALETTE[4] if g not in gem_genes else "#FFFFFF" for g in genes
            ],
            edgecolors=["#111111" if g in graph else "#BBBBBB" for g in genes],
            linewidths=[0.35 if g in graph else 0.2 for g in genes],
            ax=ax,
        )
        if index == 0:
            # On a circle a label belongs on the radius through its node, which
            # needs no collision solver and never covers the graph.
            for gene in labeled:
                x, y = positions[gene]
                angle = math.degrees(math.atan2(y, x))
                ax.text(
                    1.045 * x,
                    1.045 * y,
                    names.get(gene, gene),
                    fontsize=4.6,
                    fontstyle="italic",
                    ha="left" if x >= 0 else "right",
                    va="center",
                    rotation=angle if x >= 0 else angle + 180.0,
                    rotation_mode="anchor",
                )

        ax.set_title(
            f"{CHANNEL_TITLES[channel]}: {graph.number_of_edges()} edges, "
            f"{len(connected)} of {len(genes)} genes",
            fontsize=6.0,
            pad=2.0,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-1.34, 1.34)
        ax.set_ylim(-1.34, 1.34)
        ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
        rows.append(
            {
                "channel": channel,
                "score_floor_exclusive": args.score_floor,
                "edges_drawn": graph.number_of_edges(),
                "genes_with_a_drawn_edge": len(connected),
                "edges_all_scores": full[channel].number_of_edges(),
                "genes_total": len(genes),
            }
        )

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=3.2,
            markerfacecolor="#FFFFFF",
            markeredgecolor="#333333",
            markeredgewidth=0.25,
            label="catalyzes a reaction in the isoprenol GEM",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=3.2,
            markerfacecolor=PLOT_PALETTE[4],
            markeredgecolor="#333333",
            markeredgewidth=0.25,
            label=f"no GEM reaction ({len(off_model)} of {len(genes)} targets)",
        ),
        Line2D(
            [],
            [],
            color="#000000",
            alpha=0.25,
            linewidth=0.5,
            label="union of all six channels, any score",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=3,
        frameon=True,
        fontsize=5.0,
        handletextpad=0.5,
    )
    fig.suptitle(
        f"STRING v12.5 evidence channels over the {len(genes)} CRISPRi target genes of "
        f"P. putida KT2440 (taxon 160488)\ngenes on a circle in genome order, edges "
        f"drawn above {args.score_floor}; node area is the number of designs naming "
        f"the gene",
        fontsize=6.5,
    )

    out_dir = osp.join(ASSET_IMAGES_DIR, "048-pputida-isoprenol-cgt")
    os.makedirs(out_dir, exist_ok=True)
    stem = "string_channels_crispri_targets"
    if not args.no_timestamp:
        stem = f"{stem}_{timestamp()}"
    png = osp.join(out_dir, f"{stem}.png")
    svg = osp.join(out_dir, f"{stem}.svg")
    fig.savefig(png, dpi=400)
    savefig_true_size_svg(fig, svg)
    plt.close(fig)

    counts = pd.DataFrame(rows)
    counts_path = osp.join(RESULTS, "string_channel_edge_counts.csv")
    counts.to_csv(counts_path, index=False)
    print(counts.to_string(index=False))
    print(f"\n{len(genes) - len(off_model)} of {len(genes)} targets are in the GEM")
    print(png)
    print(svg)
    print(counts_path)


if __name__ == "__main__":
    main()
