# experiments/database/scripts/plot_ontology_dag.py
# [[experiments.database.scripts.plot_ontology_dag]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/plot_ontology_dag
# Copied 2026-10-26 from the uncommitted worktree fig/si-database-panels (same path there)
# for manuscript SI figures; that copy is the origin of this file.
"""The torchcell experiment ontology drawn as a layered DAG, in the style of the DCell
Gene Ontology panel (experiments/006-kuzmin-tmi/scripts/dcell_model_go_stats.py).

The schema is read live through ``torchcell.paper.ontology_graph`` (the pydantic models,
never a copy), so a class added to the schema appears on the next run. Nodes are every
pydantic model and every controlled vocabulary; two edge kinds make it a DAG rather than
a tree: inheritance (parent to child, drawn solid) and composition (a class to a class one
of its fields holds, drawn dashed). A class that several classes hold (``Media`` under
every environment, ``ProvenanceGap`` under many) is where the drawing stops being a tree.

Strata are topological generations: a class sits one row below the deepest class that
points at it, so the experiment roots and the abstract perturbation roots sit at the top
and the vocabularies collect at the bottom. Within a stratum, classes are grouped by lane
(genotype, environment, experiment, phenotype, provenance, vocabulary) and ordered by the
mean x of their parents, so each lane's subtree stays together.

One record type is highlighted: ``FitnessExperiment`` and every class reachable from it,
which is the closure the knowledge-graph manifest fingerprints for the datasets that load
fitness, and therefore the set of classes whose change forces those datasets to rebuild.

The legend is a white framed box inside the plot at the lower right, where strata 9 to 11
hold no node right of the middle; the script measures the box against every mark under it
and stops if the box covers ink or comes within LEGEND_MIN_CLEAR_MM of it.

Outputs (true-size SVG + PNG + a CSV of the layout) under $ASSET_IMAGES_DIR/database/:
  ontology_dag.svg / .png      the panel, PANEL_WIDTHS_MM["full"] wide
  ontology_dag_layout.csv      class, lane, kind, stratum, x, highlighted

Run from the repo root:
    python experiments/database/scripts/plot_ontology_dag.py [--highlight FitnessExperiment]
"""

from __future__ import annotations

import argparse
import csv
import os
import os.path as osp

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from dotenv import load_dotenv
from matplotlib.collections import PathCollection
from matplotlib.path import Path
from matplotlib.lines import Line2D

from torchcell.paper.ontology_graph import LANE_ORDER, LANE_PALETTE_INDEX, build_ontology_graph
from torchcell.utils import PANEL_WIDTHS_MM, PLOT_PALETTE, mm_to_in, savefig_true_size_svg

load_dotenv()
IMG_DIR = osp.join(os.environ["ASSET_IMAGES_DIR"], "database")

# Set after the torchcell imports: torchcell.graph applies the repo mplstyle on import.
plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 6,
        "axes.labelsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "svg.fonttype": "none",
        "savefig.bbox": None,
    }
)

PANEL_W_MM = PANEL_WIDTHS_MM["full"]
# The plot box is 87.4 mm tall; the panel is that plus a hairline of white above and below.
# The legend sits inside the box (2026-10-25, it was a row under the axes on a 95 mm panel),
# so nothing is drawn below the bottom spine and the height follows the box.
AXES_H_MM = 87.4
MARGIN_TOP_MM = 0.475
MARGIN_BOTTOM_MM = 0.475
PANEL_H_MM = AXES_H_MM + MARGIN_TOP_MM + MARGIN_BOTTOM_MM
LEGEND_MIN_CLEAR_MM = 0.5  # frame to the nearest mark; 0.46 mm once the panel is set at 165 mm
LANE_LABEL = {
    "genotype": "Genotype",
    "environment": "Environment",
    "experiment": "Experiment",
    "phenotype": "Phenotype",
    "provenance": "Provenance",
    "enum": "Controlled vocabulary",
}
LANE_COLOR = {lane: PLOT_PALETTE[i] for lane, i in LANE_PALETTE_INDEX.items()}
HIGHLIGHT = "#000000"  # the closure of the highlighted record type: thick black border and black edges, as the companion figures mark a closure
EDGE_COLD = "#C8C8C8"


ROOT = "Dataset"
# What the served graph's ``dataset`` node holds (biocypher/config/torchcell_schema_config.yaml:
# ``experiment member of``, ``experiment reference of``, the publication): the unified root.
ROOT_CHILDREN = ("Experiment", "ExperimentReference", "Publication")


def schema_dag() -> nx.DiGraph:
    """Every schema class and vocabulary under one root, inheritance and composition edges.

    The synthetic root is the served graph's ``dataset`` node, which holds the experiments,
    their references and the publication; it plays the part GO:ROOT plays over the three
    namespaces. An abstract base nobody references directly (``GenePerturbation``: a
    genotype's field holds the union of its leaves, not the base) is hung under the class
    whose field names two or more of its descendants, drawn as a composition edge to the
    base. Classes still unreachable from the root (behavior mixins, vocabularies no field
    references) are dropped and counted in the panel text.
    """
    g = build_ontology_graph()
    G = nx.DiGraph()
    for c in g.classes.values():
        G.add_node(c.name, lane=c.lane, kind=c.kind)
        if c.parent:
            G.add_edge(c.parent, c.name, kind="inh")
    for e in g.composition_edges:
        if e.source == e.target or G.has_edge(e.source, e.target):
            continue
        G.add_edge(e.source, e.target, kind="comp")
    G.add_node(ROOT, lane="experiment", kind="root")
    for child in ROOT_CHILDREN:
        G.add_edge(ROOT, child, kind="comp")
    # The union rule: a field whose annotation names two or more classes that all descend
    # from one unreferenced base (``Genotype.perturbations`` over the GenePerturbation
    # leaves) holds that base, so the base hangs under the holding class.
    def top(name: str) -> str:
        """The highest non-mixin base above a class (mixins carry behavior, not kind)."""
        while g.classes[name].parent is not None and not g.classes[name].parent.endswith("Mixin"):
            name = g.classes[name].parent
        return name

    for cls in g.classes.values():
        for f in cls.own_fields:
            refs = [r for r in f.references if r in g.classes and g.classes[r].kind == "model"]
            if len(refs) < 2:
                continue
            by_base: dict[str, list[str]] = {}
            for r in refs:
                by_base.setdefault(top(r), []).append(r)
            for base, members in by_base.items():
                unheld = all(u.endswith("Mixin") for u in G.predecessors(base))  # only a mixin above it
                if (len(members) >= 2 and base not in refs and base != cls.name and unheld
                        and not base.endswith("Mixin") and not G.has_edge(cls.name, base)
                        and not nx.has_path(G, base, cls.name)):
                    G.add_edge(cls.name, base, kind="comp")
    if not nx.is_directed_acyclic_graph(G):
        raise SystemExit(f"schema graph has cycles: {list(nx.simple_cycles(G))[:5]}")
    reach = nx.descendants(G, ROOT) | {ROOT}
    dropped = sorted(set(G) - reach)
    G.remove_nodes_from(dropped)
    G.graph["dropped"] = dropped
    G.graph["ontology"] = g
    return G


def layout(G: nx.DiGraph) -> tuple[dict[str, tuple[float, int]], int]:
    """Positions in [0, 1] x {stratum}: strata are topological generations, x groups by
    lane then follows the mean x of the parents; large strata are spread evenly, small
    ones keep their barycenter and are pushed apart to a minimum gap."""
    lane_rank = {lane: i for i, lane in enumerate(LANE_ORDER)}
    pos: dict[str, tuple[float, int]] = {}
    gens = list(nx.topological_generations(G))
    for s, names in enumerate(gens):
        names = list(names)
        if s <= 1:  # the root, then what it holds, spread evenly like GO's namespace roots
            ordered = sorted(names, key=lambda n: (lane_rank[G.nodes[n]["lane"]], n))
            for i, n in enumerate(ordered):
                pos[n] = ((i + 0.5) / len(ordered), s)
            continue
        bary = {n: float(np.mean([pos[p][0] for p in G.predecessors(n)])) for n in names}
        ordered = sorted(names, key=lambda n: (lane_rank[G.nodes[n]["lane"]], bary[n], n))
        k = len(ordered)
        if k >= 30:
            for i, n in enumerate(ordered):
                pos[n] = ((i + 0.5) / k, s)
        else:
            xs = np.array([bary[n] for n in ordered])
            gap = 0.018
            for i in range(1, k):
                xs[i] = max(xs[i], xs[i - 1] + gap)
            xs = np.clip(xs - max(0.0, xs[-1] - (1 - gap / 2)), gap / 2, 1 - gap / 2)
            for n, x in zip(ordered, xs):
                pos[n] = (float(x), s)
    return pos, len(gens)


def inh_ancestors(G: nx.DiGraph, n: str) -> set[str]:
    """The inheritance chain above ``n`` (composition edges are not followed)."""
    out: set[str] = set()
    stack = [n]
    while stack:
        cur = stack.pop()
        for p in G.predecessors(cur):
            if G.edges[p, cur]["kind"] == "inh" and p not in out:
                out.add(p)
                stack.append(p)
    return out


def closure(G: nx.DiGraph, highlight: str) -> set[str]:
    """The classes a record of type ``highlight`` can serialize through: the type and its
    inheritance ancestors, the classes its fields hold, and for each held class its whole
    subtree plus its own inheritance parents. Fields are resolved per name along the
    inheritance chain, nearest declaration first, so ``FitnessExperiment.phenotype:
    FitnessPhenotype`` overrides ``Experiment.phenotype: Phenotype`` and the other
    phenotype types stay out, while the inherited ``environment: Environment`` field
    brings the environment subtree in."""
    g = G.graph["ontology"]
    chain = [highlight]
    while g.classes[chain[-1]].parent:
        chain.append(g.classes[chain[-1]].parent)
    fields: dict[str, list[str]] = {}
    for cls in chain:
        for f in g.classes[cls].own_fields:
            if not f.inherited:
                fields.setdefault(f.name, f.references)
    hot = {c for c in chain if c in G}
    for refs in fields.values():
        for t in refs:
            if t in G:
                hot |= {t} | nx.descendants(G, t) | inh_ancestors(G, t)
    return hot


# Classes of the companion figure (one fitness record and its closure) named on the plot,
# and on which side of the node the label sits.
LABELS = {"Genotype": "below", "Environment": "above"}


def legend_clearance_mm(fig: plt.Figure, ax: plt.Axes, leg, dpi: int = 600) -> tuple[float, tuple[float, float, float, float]]:
    """How far the legend frame is from the nearest mark of the plot, and where the frame is.

    The figure is laid out and rendered at ``dpi``: at the screen dpi Agg's hinted text
    heights run about 0.2 mm short per legend row, so the box measured there is shorter
    than the one the SVG carries; at 600 dpi it matches the export. The plot is rendered
    with the legend and the four spines hidden (the box is set a fixed pad off the spines on
    purpose); every pixel not white is a mark: nodes, edges, labels, the count line, tick
    labels. Returns the gap in mm from the frame's outer edge (0 when the box covers a mark)
    and the frame as (width, height, gap to the right spine, gap to the bottom spine) in mm."""
    dpi0 = fig.dpi
    fig.set_dpi(dpi)
    fig.canvas.draw()
    bb = leg.get_frame().get_window_extent()
    hidden = [leg, *ax.spines.values()]
    for a in hidden:
        a.set_visible(False)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).astype(float) / 255
    for a in hidden:
        a.set_visible(True)
    fig.set_dpi(dpi0)
    rgb = rgba[..., :3] * rgba[..., 3:] + (1 - rgba[..., 3:])  # onto white
    ink = (rgb < 0.96).any(axis=-1)
    half = leg.get_frame().get_linewidth() / 2 * dpi / 72
    rows = rgba.shape[0]
    x0, x1 = bb.x0 - half, bb.x1 + half
    y0, y1 = rows - bb.y1 - half, rows - bb.y0 + half  # image rows run down
    ys, xs = np.nonzero(ink)
    dx = np.maximum(np.maximum(x0 - xs, xs - x1), 0)
    dy = np.maximum(np.maximum(y0 - ys, ys - y1), 0)
    px_mm = 25.4 / dpi
    pos = ax.get_position()  # figure fractions, independent of dpi
    w_px, h_px = fig.get_figwidth() * dpi, fig.get_figheight() * dpi
    frame = (bb.width * px_mm, bb.height * px_mm, (pos.x1 * w_px - bb.x1) * px_mm, (bb.y0 - pos.y0 * h_px) * px_mm)
    return float(np.hypot(dx, dy).min() * px_mm), frame


def panel(G: nx.DiGraph, highlight: str) -> None:
    pos, n_strata = layout(G)
    hot_nodes = closure(G, highlight)
    hot_edges = {(u, v) for u, v in G.edges if u in hot_nodes and v in hot_nodes}

    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_W_MM), mm_to_in(PANEL_H_MM)))
    fig.subplots_adjust(left=0.045, right=0.995, bottom=MARGIN_BOTTOM_MM / PANEL_H_MM, top=1 - MARGIN_TOP_MM / PANEL_H_MM)

    def curve(u: str, v: str) -> Path:
        """A vertical S-curve between two nodes: leaves the parent straight down and
        arrives at the child straight down, as layered graph drawings route edges."""
        (xu, yu), (xv, yv) = pos[u], pos[v]
        ym = (yu + yv) / 2
        return Path([(xu, yu), (xu, ym), (xv, ym), (xv, yv)], [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])

    for kind, ls in (("inh", "solid"), ("comp", (0, (2, 1.5)))):
        cold = [curve(u, v) for u, v, d in G.edges(data=True) if d["kind"] == kind and (u, v) not in hot_edges]
        ax.add_collection(PathCollection(cold, facecolors="none", edgecolors=EDGE_COLD, linewidths=0.2, linestyles=ls, zorder=1))
        hot = [curve(u, v) for u, v, d in G.edges(data=True) if d["kind"] == kind and (u, v) in hot_edges]
        ax.add_collection(PathCollection(hot, facecolors="none", edgecolors=HIGHLIGHT, linewidths=0.4, linestyles=ls, alpha=0.9, zorder=3))

    for lane in LANE_ORDER:
        sel = [n for n in G if G.nodes[n]["lane"] == lane and n not in hot_nodes]
        if sel:
            xs, ys = zip(*(pos[n] for n in sel))
            ax.scatter(xs, ys, s=4, color=LANE_COLOR[lane], linewidths=0, zorder=2)
        sel = [n for n in G if G.nodes[n]["lane"] == lane and n in hot_nodes]
        if sel:
            xs, ys = zip(*(pos[n] for n in sel))
            ax.scatter(xs, ys, s=8, facecolor=LANE_COLOR[lane], edgecolor=HIGHLIGHT, linewidths=0.7, zorder=4)

    # Name the roots and the highlighted record type, on white boxes clear of the edges.
    label_box = dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none")
    x, s = pos[ROOT]
    ax.scatter([x], [s], s=12, color="#666666", linewidths=0, zorder=5)
    ax.text(x, s - 0.45, ROOT, ha="center", va="center", fontsize=6, bbox=label_box, zorder=6)
    for i, n in enumerate(ROOT_CHILDREN):
        x, s = pos[n]
        ax.text(x, s - 0.45, n, ha="center", va="center", fontsize=6, bbox=label_box, zorder=6)
    x, s = pos[highlight]
    ax.text(x, s + 0.45, highlight, ha="center", va="center", fontsize=6, fontweight="bold", color=HIGHLIGHT, bbox=label_box, zorder=6)
    for name, side in LABELS.items():
        x, s = pos[name]
        ax.text(x, s + (0.45 if side == "below" else -0.45), name, ha="left", va="center", fontsize=6,
                fontweight="bold", color=HIGHLIGHT, bbox=label_box, zorder=6)

    n_inh = sum(1 for *_, d in G.edges(data=True) if d["kind"] == "inh")
    n_comp = G.number_of_edges() - n_inh
    ax.text(
        0.008, 0.015,
        f"{G.number_of_nodes() - 1:,} classes under one root, {n_inh:,} inheritance + {n_comp:,} composition edges, "
        f"{n_strata} strata",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=6,
    )
    ax.set_xlim(-0.005, 1.005)
    ax.set_ylim(n_strata - 0.4, -0.9)
    ax.set_yticks(range(n_strata))
    ax.set_ylabel("Stratum")
    ax.set_xticks([])
    ax.tick_params(axis="y", width=0.5, length=2, pad=1.5)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    handles = [Line2D([], [], marker="o", ls="", ms=3, color=LANE_COLOR[l], label=LANE_LABEL[l]) for l in LANE_ORDER] + [
        Line2D([], [], color=EDGE_COLD, lw=0.8, label="Inheritance"),
        Line2D([], [], color=EDGE_COLD, lw=0.8, ls=(0, (2, 1.5)), label="Composition (a field holds the class)"),
        Line2D([], [], marker="o", ls="", ms=3.5, markerfacecolor="white", markeredgecolor=HIGHLIGHT, markeredgewidth=1.0, label=f"Closure of {highlight} (thick border)"),
    ]
    # Two columns: the six lanes (what a node's color means) on the left, the two edge kinds
    # and the closure border on the right; three empty slots pad the right column so the
    # column-major fill splits 6 | 3. Framed in the repo legend style (PAPER_RC in
    # torchcell.utils: white face, 0.5 pt black edge, square corners) at the lower right.
    blanks = [Line2D([], [], ls="", marker="", label=" ")] * 3
    leg = ax.legend(handles=handles + blanks, loc="lower right", ncol=2, frameon=True, fancybox=False,
                    facecolor="white", edgecolor="black", framealpha=1.0, handlelength=1.4, handletextpad=0.5,
                    borderpad=0.4, borderaxespad=0.5, labelspacing=0.25, columnspacing=1.2)
    leg.get_frame().set_linewidth(0.5)
    leg.set_zorder(10)
    clear_mm, (lw_mm, lh_mm, right_mm, bottom_mm) = legend_clearance_mm(fig, ax, leg)
    if clear_mm < LEGEND_MIN_CLEAR_MM:
        raise SystemExit(f"legend box is {clear_mm:.2f} mm from the nearest mark (< {LEGEND_MIN_CLEAR_MM} mm): move or re-column it")
    print(f"legend box {lw_mm:.1f} x {lh_mm:.1f} mm, {right_mm:.2f} mm from the right spine and {bottom_mm:.2f} mm above "
          f"the bottom spine; nearest mark {clear_mm:.2f} mm from its frame")

    os.makedirs(IMG_DIR, exist_ok=True)
    savefig_true_size_svg(fig, osp.join(IMG_DIR, "ontology_dag.svg"))
    fig.savefig(osp.join(IMG_DIR, "ontology_dag.png"), dpi=300)
    plt.close(fig)
    with open(osp.join(IMG_DIR, "ontology_dag_layout.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "lane", "kind", "stratum", "x", "highlighted"])
        for n in sorted(G):
            w.writerow([n, G.nodes[n]["lane"], G.nodes[n]["kind"], pos[n][1], f"{pos[n][0]:.4f}", int(n in hot_nodes)])
    print(f"{G.number_of_nodes() - 1} classes under {ROOT}, {n_inh} inheritance + {n_comp} composition edges, "
          f"{n_strata} strata; closure of {highlight}: {len(hot_nodes)} classes; dropped {G.graph['dropped']} "
          f"-> {IMG_DIR}/ontology_dag.svg")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--highlight", default="FitnessExperiment", help="record type whose closure is drawn with a thick black border")
    args = ap.parse_args()
    panel(schema_dag(), args.highlight)


if __name__ == "__main__":
    main()
