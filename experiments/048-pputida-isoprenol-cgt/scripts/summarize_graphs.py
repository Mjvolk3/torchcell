# experiments/048-pputida-isoprenol-cgt/scripts/summarize_graphs.py
# [[experiments.048-pputida-isoprenol-cgt.scripts.summarize_graphs]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/048-pputida-isoprenol-cgt/scripts/summarize_graphs
"""One table of the graphs the P. putida cell graph is made of, for a slide.

Deliberately basic. One row per graph, what an edge means, how many nodes and
edges, and nothing else: a slide audience cannot read a table of fifteen
columns, and the detail already lives in the figures and the per-graph CSVs.

The numbers are read off the **built cell graph**, not recomputed from the
source files, so the table says what the model actually consumes rather than
what the inputs contain. The two differ in one way worth knowing: the STRING
channel graphs are restricted to the genome's 5,729 genes before they enter,
and the metabolic bipartite graph's reaction nodes are per catalytic unit, so a
reaction with an OR in its GPR contributes more than one node.

Writes ``results/cell_graph_summary.csv``, prints the same table as markdown to
paste into a deck, and renders a slide-sized PNG and SVG.
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from matplotlib.patches import Rectangle

from torchcell.timestamp import timestamp
from torchcell.utils import PLOT_PALETTE_FILL, apply_paper_style, mm_to_in

# The assembly is a sibling module, as it is for the trainer: the experiment's
# scripts import each other by name rather than through a package.
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from pputida_cgt_data import build_dataset  # noqa: E402

load_dotenv()

ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS = osp.join(EXPERIMENT_ROOT, "048-pputida-isoprenol-cgt", "results")

#: What an edge of each STRING channel asserts, in one short phrase. STRING's
#: own definitions, compressed to fit a slide row.
CHANNEL_MEANING = {
    "neighborhood": "genes stay adjacent across genomes",
    "fusion": "the two are one gene in another genome",
    "cooccurence": "the two appear and vanish together",
    "coexpression": "expression correlates across conditions",
    "experimental": "a published experiment links them",
    "database": "a curated pathway holds both",
}
CHANNEL_DISPLAY = {
    "neighborhood": "STRING neighborhood",
    "fusion": "STRING fusion",
    "cooccurence": "STRING co-occurrence",
    "coexpression": "STRING co-expression",
    "experimental": "STRING experimental",
    "database": "STRING database",
}

#: The metabolic edge stores, by the suffix the cell graph gives them.
METABOLIC_MEANING = {
    "rmr": "a reaction consumes or produces a metabolite",
    "gpr": "a gene catalyzes a reaction",
}

SLIDE_WIDTH_MM = 240.0
FS_SLIDE = 9.0
FS_SLIDE_HEADER = 9.5


def graph_rows(cell_graph: Any) -> list[dict[str, object]]:
    """One row per edge store of the built cell graph.

    Two counts here are not the store's own ``num_edges``, and both would
    mislead a slide if they were.

    Each STRING store holds every undirected edge once and then one self-loop
    per gene, so its ``num_edges`` is 5,729 higher than the number of gene
    pairs the channel actually asserts. The row reports the pairs and the
    self-loop count is stated once, below the table.

    The gene-to-reaction store is a hyperedge store whose ``num_edges`` is the
    number of hyperedges, which is the 3,611 reactions that have a GPR at all,
    not the 4,578 gene-to-reaction links its index holds. The row reports the
    links, since that is what an edge is.
    """
    order = {
        f"string12_5_{channel}": index for index, channel in enumerate(CHANNEL_DISPLAY)
    }
    rows: list[dict[str, object]] = []
    for edge_type in cell_graph.edge_types:
        source, relation, target = edge_type
        store = cell_graph[edge_type]
        channel = relation.split("_")[-1]
        if channel in CHANNEL_MEANING:
            index = store["edge_index"]
            self_loops = int((index[0] == index[1]).sum())
            rows.append(
                {
                    "sort": order[relation],
                    "graph": CHANNEL_DISPLAY[channel],
                    "edge_means": CHANNEL_MEANING[channel],
                    "nodes": f"{int(cell_graph[source].num_nodes):,} genes",
                    "edges": int(index.shape[1]) - self_loops,
                    "self_loops": self_loops,
                }
            )
        elif relation in METABOLIC_MEANING:
            index = store["hyperedge_index"]
            rows.append(
                {
                    "sort": len(order) + (0 if relation == "rmr" else 1),
                    "graph": (
                        "Metabolism, reaction to metabolite"
                        if relation == "rmr"
                        else "Metabolism, gene to reaction"
                    ),
                    "edge_means": METABOLIC_MEANING[relation],
                    "nodes": (
                        f"{int(cell_graph[source].num_nodes):,} {source}s, "
                        f"{int(cell_graph[target].num_nodes):,} {target}s"
                    ),
                    "edges": int(index.shape[1]),
                    "self_loops": 0,
                }
            )
        else:
            raise ValueError(
                f"edge store {edge_type} is not one this table knows how to "
                f"describe; add it to CHANNEL_MEANING or METABOLIC_MEANING "
                f"rather than letting it print with a placeholder"
            )
    rows.sort(key=lambda row: int(str(row["sort"])))
    for row in rows:
        del row["sort"]
    return rows


def render_slide_table(frame: pd.DataFrame, path_png: str, path_svg: str) -> None:
    """Render the table as a slide asset.

    Sized and set for a projected slide rather than for print: wider than a
    Nature column and at 9 pt, because a table nobody can read from the back of
    a room is not a table. The paper figures in this experiment keep the 5 to
    7 pt band; this one deliberately does not.
    """
    columns = ("graph", "edge_means", "nodes", "edges")
    headers = ("Graph", "An edge means", "Nodes", "Edges")
    widths = (0.26, 0.37, 0.24, 0.13)
    row_height_mm = 8.0
    height_mm = row_height_mm * (len(frame) + 1) + 6.0

    fig = plt.figure(figsize=(mm_to_in(SLIDE_WIDTH_MM), mm_to_in(height_mm)))
    ax = fig.add_axes((0.012, 0.02, 0.976, 0.96))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, float(len(frame) + 1))
    ax.axis("off")

    starts = [sum(widths[:index]) for index in range(len(widths))]
    top = float(len(frame) + 1)

    for index, (header, start, width) in enumerate(
        zip(headers, starts, widths, strict=True)
    ):
        right = index == len(headers) - 1
        ax.text(
            start + width - 0.01 if right else start + 0.008,
            top - 0.5,
            header,
            fontsize=FS_SLIDE_HEADER,
            fontweight="bold",
            ha="right" if right else "left",
            va="center",
        )
    ax.plot([0.0, 1.0], [top - 1.0, top - 1.0], color="#000000", linewidth=1.0)
    ax.plot([0.0, 1.0], [top, top], color="#000000", linewidth=1.0)

    for row_index, record in enumerate(frame.to_dict("records")):
        y = top - 1.5 - row_index
        if row_index % 2 == 1:
            ax.add_patch(
                Rectangle(
                    (0.0, y - 0.5),
                    1.0,
                    1.0,
                    facecolor=PLOT_PALETTE_FILL[5],
                    edgecolor="none",
                    zorder=0,
                )
            )
        for index, (column, start, width) in enumerate(
            zip(columns, starts, widths, strict=True)
        ):
            value = record[column]
            text = f"{value:,}" if isinstance(value, int) else str(value)
            right = index == len(columns) - 1
            ax.text(
                start + width - 0.01 if right else start + 0.008,
                y,
                text,
                fontsize=FS_SLIDE,
                ha="right" if right else "left",
                va="center",
                zorder=2,
            )
    ax.plot([0.0, 1.0], [0.0, 0.0], color="#000000", linewidth=1.0)

    fig.savefig(path_png, dpi=400)
    fig.savefig(path_svg)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-timestamp", action="store_true")
    args = parser.parse_args()

    apply_paper_style()
    dataset = build_dataset()
    cell_graph = dataset.cell_graph
    frame = pd.DataFrame(graph_rows(cell_graph))
    dataset.close_lmdb()

    csv_path = osp.join(RESULTS, "cell_graph_summary.csv")
    frame.to_csv(csv_path, index=False)

    out_dir = osp.join(ASSET_IMAGES_DIR, "048-pputida-isoprenol-cgt")
    os.makedirs(out_dir, exist_ok=True)
    stem = "cell_graph_summary_table"
    if not args.no_timestamp:
        stem = f"{stem}_{timestamp()}"
    png = osp.join(out_dir, f"{stem}.png")
    svg = osp.join(out_dir, f"{stem}.svg")
    render_slide_table(frame, png, svg)

    print("| Graph | An edge means | Nodes | Edges |")
    print("|---|---|---|---|")
    for record in frame.to_dict("records"):
        print(
            f"| {record['graph']} | {record['edge_means']} | "
            f"{record['nodes']} | {int(record['edges']):,} |"
        )
    print()
    loops = int(frame.self_loops.max())
    print(
        f"total edges: {int(frame.edges.sum()):,}; each STRING store also "
        f"carries one self-loop per gene ({loops:,})"
    )
    print(csv_path)
    print(png)
    print(svg)


if __name__ == "__main__":
    main()
