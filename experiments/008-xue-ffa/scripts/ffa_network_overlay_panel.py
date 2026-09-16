# experiments/008-xue-ffa/scripts/ffa_network_overlay_panel.py
# [[experiments.008-xue-ffa.scripts.ffa_network_overlay_panel]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/008-xue-ffa/scripts/ffa_network_overlay_panel
#
# The network overlay of the epistasis perspective, rebuilt as a publication panel: the
# ten deleted transcription factors with their significant trigenic interactions drawn
# among them, then the regulatory links into the fatty acid pathway genes, the reactions,
# the intermediates, and the measured species, left to right across a full-width panel.
#
# WHY A NEW SCRIPT rather than another version of create_ffa_multigraph_overlays.py. That
# script is a sweep: 762 near-identical renders at 14 x 18 in, portrait, with 12 pt labels
# and a four-line title, laid out from a stored spring layout. Fitting one of those into
# Nature's 180 x 170 mm box puts its labels at 4.4 pt (the floor is 5), runs the
# metabolite labels across the reaction column, and leaves it at a width that tiles with
# nothing. Every one of those is a layout property, and the layout is what has to change.
# This script keeps the sweep's data path (the same overlap table, the same BH-within-
# readout rule, the same regulatory graphs) and replaces only the drawing: positions are
# chosen here, in millimetres, on a 179 mm canvas, at 6 pt.
#
# WHAT IS DRAWN, and what is not.
#   - Trigenic interactions significant at a 5% FDR computed within the total-titer readout
#     under the multiplicative model, each as the three edges of its triangle. Edge width
#     scales with the number of interactions the edge carries, so the figure shows how many
#     of the 45 pairs are involved in a significant three-way interaction and how often.
#     Positive is drawn over negative. The old render restricted to triples connected in
#     the genetic interaction graph, a leftover of the enrichment sweep; the default here is
#     every significant triple, and --graph restores the restriction. The two unlabeled
#     rings above and below draw the same interactions one sign at a time, as triangles.
#   - Regulatory links from a deleted factor to a pathway gene, from the SGD regulatory
#     graph or TFLink, as dotted arrows.
#   - The 13 core pathway genes, their 64 reactions, and the metabolites those reactions
#     touch. Metabolites with no edge in the pathway subgraph (66 of 148) are dropped; they
#     were carried into the bipartite export by subsystem membership and connect to
#     nothing drawn. Reactions and intermediates are unlabeled; the caption says what the
#     columns are.
#   - Measured species are labeled once per species, with a bracket spanning the
#     compartment copies (peroxisome, ER membrane, lipid particle), which is what removes
#     the ~33 compartment-suffixed labels that drove the old figure's width and height.
#
# STYLE. torchcell palette, Arial 6 pt, true-size SVG at PANEL_WIDTHS_MM["full"]. Node
# fills are the pale draw.io companions with a black outline (lilac = deleted factor,
# wheat = pathway gene, sand = reaction, gray = intermediate, steel = measured species);
# the saturated line colors are reserved for the data, blue = positive and brick =
# negative, matching every other panel in the document. A network has no axes to box, so
# this panel deviates from the boxed-axes rule.

import argparse
import os
import os.path as osp

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.lines import Line2D
from matplotlib.patches import (Circle, FancyArrowPatch, FancyBboxPatch, Polygon,
                                Rectangle)
from statsmodels.stats.multitest import multipletests

from torchcell.graph.graph import SCerevisiaeGraph
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    PLOT_PALETTE_FILL,
    apply_paper_style,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()
EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")
DATA_ROOT = os.getenv("DATA_ROOT")
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "008-xue-ffa/results")
IMAGES_DIR = osp.join(ASSET_IMAGES_DIR, "008-xue-ffa")

TOTAL = "Total Titer"
TF_GENES = ["FKH1", "GCN5", "MED4", "OPI1", "RFX1", "RGR1", "RPD3", "SPT3", "YAP6", "TFC7"]

# Pathway genes top to bottom: synthesis (ACC1, FAS1, FAS2), elongation (ELO1-3),
# desaturation (OLE1), activation (FAA1-4), then the degradation and acylation steps
# whose deletion or product defines the chassis (POX1, SLC1). Every gene in the export
# must appear here; an unlisted gene is an error, not a gene to append.
GENE_ORDER = ["ACC1", "FAS1", "FAS2", "ELO1", "ELO2", "ELO3", "OLE1",
              "FAA1", "FAA2", "FAA3", "FAA4", "POX1", "SLC1"]

# Measured species, in the order they are stacked top to bottom, with the label each group
# of compartment copies carries. LPA and PA are defined in the caption.
SPECIES_LABELS = [
    ("myristate", "C14:0 myristate"),
    ("palmitate", "C16:0 palmitate"),
    ("palmitoleate", "C16:1 palmitoleate"),
    ("stearate", "C18:0 stearate"),
    ("oleate", "C18:1 oleate"),
    ("1-acyl-sn-glycerol 3-phosphate (16:0)", "LPA 16:0"),
    ("1-acyl-sn-glycerol 3-phosphate (16:1)", "LPA 16:1"),
    ("1-acyl-sn-glycerol 3-phosphate (18:0)", "LPA 18:0"),
    ("1-acyl-sn-glycerol 3-phosphate (18:1)", "LPA 18:1"),
    ("phosphatidate (1-16:0, 2-18:1)", "PA 16:0/18:1"),
    ("phosphatidate (1-16:1, 2-18:1)", "PA 16:1/18:1"),
    ("phosphatidate (1-18:0, 2-18:1)", "PA 18:0/18:1"),
    ("phosphatidate (1-18:1, 2-18:1)", "PA 18:1/18:1"),
]
COMPARTMENT_ORDER = {"p": 0, "erm": 1, "lp": 2}

# Colors: pale fills for objects, saturated lines for data, as in the other panels.
# Positive interactions are blue and negative brick (the document-wide sign encoding);
# the measured species take the amber object fill so no object shares a hue with a sign.
C_POS = PLOT_PALETTE[4]
C_NEG = PLOT_PALETTE[1]
FILL_TF = PLOT_PALETTE_FILL[2]
FILL_GENE = PLOT_PALETTE_FILL[3]
FILL_RXN = PLOT_PALETTE_FILL[15]
FILL_INTERMEDIATE = PLOT_PALETTE_FILL[11]
FILL_SPECIES = PLOT_PALETTE_FILL[0]
C_REGULATES = PLOT_PALETTE[5]
C_PATHWAY_EDGE = "#BBBBBB"

# Geometry, in millimetres on the panel canvas. Width is the strict full-panel width;
# height follows the content.
W_MM = PANEL_WIDTHS_MM["full"]
H_MM = 118.0
Y_TOP = 105.0  # top of the node columns
Y_BOT = 8.0  # bottom of the node columns
Y_HEADER = 112.0
X_TF = 31.0  # center of the factor circle
Y_TF = 66.0
R_TF = 16.5
X_GENE = 86.0
X_RXN = 111.0
X_INT = 131.0
X_SPECIES = 150.0
X_BRACKET = 153.0
X_LABEL = 154.5
TF_BOX = (8.0, 3.2)

# The two sign-split rings. The labeled ring carries both signs at once, which answers
# "is this pair in an interaction" but not "what shape does each sign make", because 45
# brick edges and 17 blue ones in one circle read as a single mass. Each sign therefore
# also gets its own ring, drawn at the SAME node angles and with no labels, so the two
# shapes are comparable at a glance: positive above the labeled ring, negative below.
# The labeled ring shrank from R 23 to 16.5 to make room, and its boxes with it.
#
# The rings draw TRIPLES, not pairs: one translucent triangle per significant interaction,
# at the same opacity in both rings, so the ink is proportional to how many interactions
# land there and the two signs are read on one scale. The labeled ring keeps the pairwise
# edge form, because a pair's edge is what the width encoding and the offset are for.
R_RING = 8.0
RING_NODE_R = 0.75
RING_TRI_ALPHA = 0.10
RING_TRI_EDGE_PT = 0.12
Y_RING_POS = 97.0
Y_RING_NEG = 35.0
# The sign arrow beside each ring, on the empty left margin at the ring's own height.
X_RING_ARROW = 19.0
RING_ARROW_HALF = 4.0
# Bottom of the legend frame, level with the lowest pathway gene box rather than with the
# canvas edge, which is what "flush with the bottom row" means here.
Y_LEGEND_BOT = 6.3
GENE_BOX = (9.5, 3.4)
RXN_SIDE = 1.1
INT_R = 0.6
SPECIES_R = 0.75


def is_true(val):
    if isinstance(val, bool):
        return val
    return str(val).lower() in ("true", "1", "yes")


def significant_triples(model, readout, graph):
    """Significant trigenic triples on one readout, BH within that readout.

    Reads the connected-topology overlap table the enrichment sweep wrote, which holds
    every triple's p-value once per readout plus a per-graph connectivity flag. With
    graph=None the flag is ignored and every triple significant on the readout is kept.
    """
    path = osp.join(RESULTS_DIR, "graph_enrichment", f"{model}_trigenic_graph_overlap.csv")
    df = pd.read_csv(path)
    df = df[(df["ffa_type"] == readout) & df["p_value"].notna()].copy()
    reject, _, _, _ = multipletests(df["p_value"], method="fdr_bh", alpha=0.05)
    df = df[reject]
    if graph is not None:
        df = df[df[f"{graph}_connected"] == True]  # noqa: E712
    triples = [tuple(str(g).replace(":", "_").split("_")) for g in df["gene_set"]]
    signs = list(np.sign(df["interaction_score"]).astype(int))
    return list(zip(triples, signs))


def pair_multiplicity(triples_with_sign):
    """Number of positive and negative interactions each unordered pair takes part in."""
    pos, neg = {}, {}
    for genes, sign in triples_with_sign:
        for i in range(3):
            for j in range(i + 1, 3):
                key = tuple(sorted((genes[i], genes[j])))
                d = pos if sign > 0 else neg
                d[key] = d.get(key, 0) + 1
    return pos, neg


def load_pathway():
    G = nx.read_graphml(osp.join(RESULTS_DIR, "ffa_bipartite_network.graphml"))
    core = [n for n in G if G.nodes[n].get("node_type") == "gene"
            and is_true(G.nodes[n].get("is_core"))]
    rxns = [n for n in G if G.nodes[n].get("node_type") == "reaction"]
    mets = [n for n in G if G.nodes[n].get("node_type") == "metabolite"]
    mets = [n for n in mets if G.degree(n) > 0]
    species = [n for n in mets if is_true(G.nodes[n].get("is_target_ffa"))]
    intermediates = [n for n in mets if n not in set(species)]
    return G, core, rxns, intermediates, species


def standard_names(genome, systematic):
    """Systematic id -> the standard name the annotation gives it."""
    std_to_ids = genome.feature_index["standard_to_ids"]
    inverse = {}
    for std, ids in std_to_ids.items():
        for fid in ids:
            inverse.setdefault(fid.upper(), std)
    return {s: inverse[s.upper()] for s in systematic}


def regulatory_edges(graph, tf_sys, gene_sys):
    """(tf, gene) pairs joined in the SGD regulatory graph or in TFLink."""
    edges = set()
    for g in (graph.G_regulatory.graph, graph.G_tflink.graph):
        for tf in tf_sys:
            for gene in gene_sys:
                if tf in g and gene in g and (g.has_edge(tf, gene) or g.has_edge(gene, tf)):
                    edges.add((tf, gene))
    return sorted(edges)


def barycenter_order(nodes, neighbors_y):
    """Order nodes by the mean y of their neighbors; a node with none keeps last place."""
    keyed = []
    for n in nodes:
        ys = neighbors_y(n)
        keyed.append((np.mean(ys) if ys else -1e9, n))
    keyed.sort(key=lambda t: -t[0])
    return [n for _, n in keyed]


def spread(n):
    return np.linspace(Y_TOP, Y_BOT, n) if n > 1 else np.array([(Y_TOP + Y_BOT) / 2])


PT_MM = 25.4 / 72.0  # millimetres per point
EDGE_GAP_MM = 0.15  # clear space between a pair's negative and positive edge


def edge_width_pt(k):
    """Line width of an interaction edge, in points, from the pair's multiplicity."""
    return 0.4 + 0.25 * (k - 1)


def ring_edge_width_pt(k):
    """Edge width in a sign-split ring: the same encoding, scaled to the smaller radius.

    Floored at 0.25 pt. Scaling alone would put a single interaction at 0.19 pt, under
    the hairline that print holds, and a line thinner than that either drops out or is
    rendered at whatever the device's minimum happens to be.
    """
    return max(0.25, edge_width_pt(k) * R_RING / R_TF)


def ring_positions(center_y, radius):
    """The ten factors on a circle of `radius` about (X_TF, `center_y`).

    One function for every ring in the figure, so the labeled ring and the two
    sign-split ones put each factor at the same angle and the shapes can be compared
    without the reader checking which node is which.
    """
    return {
        tf: (X_TF + radius * np.cos(ang), center_y - radius * np.sin(ang))
        for tf, ang in (
            (tf, -np.pi / 2 + 2 * np.pi * i / len(TF_GENES))
            for i, tf in enumerate(sorted(TF_GENES))
        )
    }


def ring_triangles(triples, ring_pos, sign):
    """The three ring positions of every significant triple carrying `sign`.

    One triangle per interaction, which is the unit the analysis tests. Drawn at low
    opacity so overlap accumulates: where many interactions share a corner of the ring the
    ink builds up, and where a factor takes part in none the ring stays clear.
    """
    return [[ring_pos[g] for g in genes]
            for genes, s in sorted(triples) if s == sign]


def interaction_segments(pos, neg_mult, pos_mult):
    """Every interaction edge as (sign, k, (x0, y0), (x1, y1)) in millimetres.

    A pair that takes part in both a negative and a positive interaction gets its two
    edges drawn side by side, offset perpendicular to the pair's axis by half the sum of
    their widths plus a small gap, so neither hides the other. Every one of the 45 pairs
    is in at least one negative interaction, so a positive edge always has a partner;
    a pair with only a negative edge stays on the axis. Endpoints are the node centers,
    which the node shapes then cover.
    """
    segments = []
    for (a, b), k_neg in sorted(neg_mult.items()):
        (xa, ya), (xb, yb) = pos[a], pos[b]
        k_pos = pos_mult.get((a, b), 0)
        if k_pos == 0:
            segments.append((-1, k_neg, (xa, ya), (xb, yb)))
            continue
        dx, dy = xb - xa, yb - ya
        norm = float(np.hypot(dx, dy))
        nx_, ny_ = -dy / norm, dx / norm
        d = (edge_width_pt(k_neg) + edge_width_pt(k_pos)) / 2 * PT_MM + EDGE_GAP_MM
        for sign, k, s in ((-1, k_neg, -0.5), (+1, k_pos, +0.5)):
            off = (s * d * nx_, s * d * ny_)
            segments.append((sign, k, (xa + off[0], ya + off[1]), (xb + off[0], yb + off[1])))
    for (a, b), k_pos in sorted(pos_mult.items()):
        if (a, b) not in neg_mult:
            segments.append((+1, k_pos, pos[a], pos[b]))
    # negative first so positive draws on top where the two still touch
    segments.sort(key=lambda s: s[0])
    return segments


def rounded_box(ax, x, y, w, h, fill, z):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0,rounding_size=0.9",
        facecolor=fill, edgecolor="black", linewidth=0.5, zorder=z))


def layout(model, readout, graph):
    """Everything the drawing needs: nodes, edges, and positions in millimetres.

    Shared by the matplotlib panel and the draw.io generator so the two renderings place
    every node at the same spot.
    """
    triples = significant_triples(model, readout, graph)
    pos_mult, neg_mult = pair_multiplicity(triples)
    n_pos = sum(1 for _, s in triples if s > 0)
    n_neg = sum(1 for _, s in triples if s < 0)
    print(f"{len(triples)} significant triples: {n_pos} positive, {n_neg} negative; "
          f"{len(pos_mult)} pairs positive, {len(neg_mult)} pairs negative")

    G, core, rxns, intermediates, species = load_pathway()

    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    graph_store = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )
    apply_paper_style()  # torchcell.graph applies its own style at import

    tf_sys = {tf: genome.alias_to_systematic[tf][0] for tf in TF_GENES}
    gene_std = standard_names(genome, core)
    missing = set(gene_std.values()) - set(GENE_ORDER)
    if missing:
        raise ValueError(f"pathway genes not in GENE_ORDER: {sorted(missing)}")
    core_sorted = sorted(core, key=lambda s: GENE_ORDER.index(gene_std[s]))
    reg = regulatory_edges(graph_store, list(tf_sys.values()), core)
    print(f"{len(reg)} regulatory edges from {len({t for t, _ in reg})} factors")

    # --- positions -------------------------------------------------------------------
    pos = dict(ring_positions(Y_TF, R_TF))
    ring_pos = {+1: ring_positions(Y_RING_POS, R_RING),
                -1: ring_positions(Y_RING_NEG, R_RING)}
    for gene, y in zip(core_sorted, spread(len(core_sorted))):
        pos[gene] = (X_GENE, y)

    gene_of_rxn = {r: [u for u, v, d in G.edges(data=True)
                       if d.get("edge_type") == "catalyzes" and v == r and u in pos]
                   for r in rxns}
    rxns_sorted = barycenter_order(rxns, lambda r: [pos[g][1] for g in gene_of_rxn[r]])
    for r, y in zip(rxns_sorted, spread(len(rxns_sorted))):
        pos[r] = (X_RXN, y)

    def rxn_neighbors(m):
        return [pos[r][1] for r in nx.all_neighbors(G, m) if r in pos]

    int_sorted = barycenter_order(intermediates, rxn_neighbors)
    for m, y in zip(int_sorted, spread(len(int_sorted))):
        pos[m] = (X_INT, y)

    name_of = {m: G.nodes[m].get("name") for m in species}
    comp_of = {m: G.nodes[m].get("compartment") for m in species}
    label_of = dict(SPECIES_LABELS)
    unknown = {name_of[m] for m in species} - set(label_of)
    if unknown:
        raise ValueError(f"measured species without a label: {sorted(unknown)}")
    order = {name: i for i, (name, _) in enumerate(SPECIES_LABELS)}
    species_sorted = sorted(species, key=lambda m: (order[name_of[m]],
                                                    COMPARTMENT_ORDER[comp_of[m]]))
    for m, y in zip(species_sorted, spread(len(species_sorted))):
        pos[m] = (X_SPECIES, y)

    return {
        "pos": pos, "ring_pos": ring_pos,
        "triples": triples, "pos_mult": pos_mult, "neg_mult": neg_mult,
        "n_pos": n_pos, "n_neg": n_neg, "G": G, "core_sorted": core_sorted,
        "rxns": rxns, "intermediates": intermediates, "species_sorted": species_sorted,
        "gene_std": gene_std, "tf_sys": tf_sys, "reg": reg, "name_of": name_of,
        "label_of": label_of,
    }


def draw(model, readout, graph, out_stem):
    apply_paper_style()
    L = layout(model, readout, graph)
    pos, pos_mult, neg_mult = L["pos"], L["pos_mult"], L["neg_mult"]
    n_pos, n_neg, G, core_sorted = L["n_pos"], L["n_neg"], L["G"], L["core_sorted"]
    rxns, intermediates, species_sorted = L["rxns"], L["intermediates"], L["species_sorted"]
    gene_std, tf_sys, reg, name_of, label_of = (L["gene_std"], L["tf_sys"], L["reg"],
                                                L["name_of"], L["label_of"])
    species = species_sorted

    # --- canvas ------------------------------------------------------------------------
    fig = plt.figure(figsize=(mm_to_in(W_MM), mm_to_in(H_MM)))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W_MM)
    ax.set_ylim(0, H_MM)
    ax.set_aspect("equal")
    ax.axis("off")

    # pathway edges, thin and light, under everything
    for u, v, d in G.edges(data=True):
        if u in pos and v in pos and d.get("edge_type") in ("catalyzes", "consumed_by",
                                                            "produces"):
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                    color=C_PATHWAY_EDGE, linewidth=0.3, solid_capstyle="round", zorder=1)

    # regulatory arrows from the factor boxes into the gene boxes
    sys_to_tf = {v: k for k, v in tf_sys.items()}
    for tf, gene in reg:
        x0, y0 = pos[sys_to_tf[tf]]
        x1, y1 = pos[gene]
        ax.add_patch(FancyArrowPatch(
            (x0 + TF_BOX[0] / 2, y0), (x1 - GENE_BOX[0] / 2, y1),
            arrowstyle="-|>", mutation_scale=4, color=C_REGULATES, linewidth=0.5,
            linestyle=(0, (0.8, 1.2)), shrinkA=0, shrinkB=0, zorder=2))

    # interaction edges: negative under positive, width by multiplicity, a pair's two
    # signs drawn side by side
    width = edge_width_pt
    for sign, k, (x0, y0), (x1, y1) in interaction_segments(pos, neg_mult, pos_mult):
        ax.plot([x0, x1], [y0, y1], color=C_POS if sign > 0 else C_NEG,
                linewidth=width(k), solid_capstyle="round", zorder=4 if sign > 0 else 3)

    # the two sign-split rings: one sign each, same angles, no labels, one translucent
    # triangle per interaction, with an arrow beside each ring giving its sign
    for sign, color in ((+1, C_POS), (-1, C_NEG)):
        rpos = L["ring_pos"][sign]
        for tri in ring_triangles(L["triples"], rpos, sign):
            ax.add_patch(Polygon(tri, closed=True, facecolor=color,
                                 alpha=RING_TRI_ALPHA, edgecolor=color,
                                 linewidth=RING_TRI_EDGE_PT, zorder=3))
        for tf in TF_GENES:
            ax.add_patch(Circle(rpos[tf], RING_NODE_R, facecolor=FILL_TF,
                                edgecolor="black", linewidth=0.3, zorder=5))
        y_c = Y_RING_POS if sign > 0 else Y_RING_NEG
        ax.add_patch(FancyArrowPatch(
            (X_RING_ARROW, y_c - sign * RING_ARROW_HALF),
            (X_RING_ARROW, y_c + sign * RING_ARROW_HALF),
            arrowstyle="-|>", mutation_scale=5, color=color, linewidth=0.9,
            shrinkA=0, shrinkB=0, zorder=5))

    # nodes
    for r in rxns:
        x, y = pos[r]
        ax.add_patch(Rectangle((x - RXN_SIDE / 2, y - RXN_SIDE / 2), RXN_SIDE, RXN_SIDE,
                               facecolor=FILL_RXN, edgecolor="black", linewidth=0.3,
                               zorder=5))
    for m in intermediates:
        ax.add_patch(Circle(pos[m], INT_R, facecolor=FILL_INTERMEDIATE, edgecolor="black",
                            linewidth=0.3, zorder=5))
    for m in species:
        ax.add_patch(Circle(pos[m], SPECIES_R, facecolor=FILL_SPECIES, edgecolor="black",
                            linewidth=0.4, zorder=6))
    for gene in core_sorted:
        x, y = pos[gene]
        rounded_box(ax, x, y, *GENE_BOX, FILL_GENE, 6)
        ax.text(x, y, gene_std[gene], ha="center", va="center", fontsize=6, zorder=7)
    for tf in TF_GENES:
        x, y = pos[tf]
        rounded_box(ax, x, y, *TF_BOX, FILL_TF, 6)
        ax.text(x, y, tf, ha="center", va="center", fontsize=6, zorder=7)

    # species labels, one per group of compartment copies, with a bracket
    i = 0
    while i < len(species_sorted):
        name = name_of[species_sorted[i]]
        j = i
        while j + 1 < len(species_sorted) and name_of[species_sorted[j + 1]] == name:
            j += 1
        y_hi = pos[species_sorted[i]][1]
        y_lo = pos[species_sorted[j]][1]
        ax.plot([X_BRACKET, X_BRACKET + 0.8, X_BRACKET + 0.8, X_BRACKET],
                [y_hi + 0.9, y_hi + 0.9, y_lo - 0.9, y_lo - 0.9],
                color="black", linewidth=0.4, zorder=6)
        ax.text(X_LABEL + 0.8, (y_hi + y_lo) / 2, label_of[name], ha="left",
                va="center", fontsize=6, zorder=7)
        i = j + 1

    # column headers
    for x, text in ((X_TF, "deleted transcription factors"),
                    (X_GENE, "pathway genes"),
                    (X_RXN, "reactions"),
                    (X_INT, "intermediates"),
                    (X_SPECIES + 8, "measured species")):
        ax.text(x, Y_HEADER, text, ha="center", va="center", fontsize=6)

    # legend, in the clear region under the factor circle
    k_max = max(list(neg_mult.values()) + list(pos_mult.values()))
    handles = [
        Line2D([], [], color=C_NEG, linewidth=width(1),
               label=f"negative trigenic interaction (n = {n_neg})"),
        Line2D([], [], color=C_POS, linewidth=width(1),
               label=f"positive trigenic interaction (n = {n_pos})"),
        Line2D([], [], color=C_NEG, linewidth=width(k_max),
               label=f"width: interactions per pair, {width(1)} pt at 1 to "
                     f"{width(k_max):.2f} pt at {k_max}"),
        Line2D([], [], color=C_REGULATES, linewidth=0.5, linestyle=(0, (0.8, 1.2)),
               label="factor regulates gene (SGD regulatory or TFLink)"),
        Line2D([], [], color=C_PATHWAY_EDGE, linewidth=0.5,
               label="catalyzes, consumes or produces"),
    ]
    # Bottom-left, its lower edge level with the lowest pathway gene box.
    ax.legend(handles=handles, loc="lower left",
              bbox_to_anchor=(2.0 / W_MM, Y_LEGEND_BOT / H_MM),
              fontsize=5, handlelength=2.2, borderpad=0.5, labelspacing=0.35)

    os.makedirs(IMAGES_DIR, exist_ok=True)
    png = osp.join(IMAGES_DIR, out_stem + ".png")
    svg = osp.join(IMAGES_DIR, out_stem + ".svg")
    fig.savefig(png, dpi=300)
    savefig_true_size_svg(fig, svg)
    plt.close(fig)
    print(f"wrote {png}\nwrote {svg}")
    return png, svg


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="multiplicative",
                    choices=["multiplicative", "additive", "log_ols", "glm_log_link"])
    ap.add_argument("--readout", default=TOTAL)
    ap.add_argument("--graph", default=None,
                    help="restrict to triples connected in this interaction graph, e.g. "
                         "genetic; default draws every significant triple")
    ap.add_argument("--out", default="panel_network_overlay",
                    help="output stem under notes/assets/images/008-xue-ffa/")
    args = ap.parse_args()
    draw(args.model, args.readout, args.graph, args.out)


if __name__ == "__main__":
    main()
