# experiments/048-pputida-isoprenol-cgt/scripts/plot_metabolic_network.py
# [[experiments.048-pputida-isoprenol-cgt.scripts.plot_metabolic_network]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/048-pputida-isoprenol-cgt/scripts/plot_metabolic_network
"""The metabolic module: the isoprenol route, and what the CRISPRi guides do to it.

Two panels, both drawn as the cell graph stores metabolism: a **bipartite**
graph of metabolite nodes and reaction nodes, never a metabolite-to-metabolite
cartoon. Positions are hand-set, so the diagram is stable across reruns and can
be read as a map.

**Panel a, the heterologous route.** The nine reactions that carry acetyl-CoA to
extracellular isoprenol, in order, each annotated with the genes its GPR names.
Five of them (mvaE, mvaS, MK, PMDHKQ, aphA) are the pIY670 plasmid's and are not
KT2440 loci, which is why they are constant across every strain in the campaign
and why the model never sees them as gene tokens. The rest are native.

**Panel b, the competition for acetyl-CoA.** The route's entry metabolite is
acetyl-CoA, which the model's GEM connects to 73 reactions. The campaign's
CRISPRi guides are not aimed at the route: they are aimed at the native
metabolism around it, and 45 of those 73 reactions are catalyzed by a gene some
guide knocks down. The panel draws each such gene once, sized by how many
acetyl-CoA reactions it carries and split by whether those reactions produce or
consume acetyl-CoA, which is the mechanism by which a guide against fatty acid
metabolism can move isoprenol titer at all.

Outputs (PNG and true-size SVG) go to ``ASSET_IMAGES_DIR/048-pputida-isoprenol-cgt/``
and the per-gene acetyl-CoA accounting to ``results/accoa_competition.csv``.
"""

from __future__ import annotations

import argparse
import math
import os
import os.path as osp
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")

import cobra
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from torchcell.graph.string_db import StringTaxonMirror
from torchcell.metabolism.bigg_gem import BiggGEM
from torchcell.timestamp import timestamp
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    PLOT_PALETTE_FILL,
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
STRING_ROOT = "/bulk/torchcell-string"
ISOPRENOL_MODEL_ID = "pputida_isoprenol"

#: The route, as (reaction id, substrate metabolite, product metabolite), in the
#: order carbon travels. Read off the deposited model, not invented: every pair
#: is an edge of ``BiggGEM("pputida_isoprenol").bipartite_graph``.
ROUTE: tuple[tuple[str, str, str], ...] = (
    ("ACCOAAT", "accoa_c", "aacoa_c"),
    ("HMGCOAS", "aacoa_c", "hmgcoa_c"),
    ("HMGCOAR", "hmgcoa_c", "mva_c"),
    ("MEVKIN", "mva_c", "5pmva_c"),
    ("PMD", "5pmva_c", "3mbut3e1yp_c"),
    ("APP", "3mbut3e1yp_c", "ip_c"),
    ("IPtpp", "ip_c", "ip_p"),
    ("IPtex", "ip_p", "ip_e"),
)
EXCHANGE = "EX_ip_e"

#: Display names for the route's metabolites. BiGG ids are not reading text and
#: four of the nine carry no ``name`` in the deposited model.
METABOLITE_LABELS = {
    "accoa_c": "acetyl-CoA",
    "aacoa_c": "acetoacetyl-CoA",
    "hmgcoa_c": "HMG-CoA",
    "mva_c": "mevalonate",
    "5pmva_c": "mevalonate-5-P",
    "3mbut3e1yp_c": "isopentenyl-P",
    "ip_c": "isoprenol (in)",
    "ip_p": "isoprenol (peri)",
    "ip_e": "isoprenol (out)",
}

#: The five pIY670 genes. Constant in every strain, so background rather than
#: signal, and absent from the gene graph because they are not KT2440 loci.
PLASMID_GENES = frozenset({"mvaE", "mvaS", "MK", "PMDHKQ", "aphA"})

ENTRY_METABOLITE = "accoa_c"


def target_genes() -> set[str]:
    """The campaign's CRISPRi target genes."""
    table = pd.read_csv(RECORD_TABLE)
    campaign = table[table.is_campaign]
    genes: set[str] = set()
    for key in campaign["design_key"]:
        if key != "no_guide":
            genes.update(str(key).split("|"))
    return genes


def route_genes(model: cobra.Model, reaction_id: str) -> list[str]:
    """The genes a reaction's GPR names, sorted with plasmid genes first."""
    reaction = model.reactions.get_by_id(reaction_id)
    genes = sorted(str(gene.id) for gene in reaction.genes)
    return sorted(genes, key=lambda g: (g not in PLASMID_GENES, g))


def accoa_competition(model: cobra.Model, targets: set[str]) -> pd.DataFrame:
    """Per gene, the acetyl-CoA reactions it catalyzes that a guide knocks down.

    ``produces`` and ``consumes`` count the reactions by the sign of the
    acetyl-CoA coefficient. A reversible reaction is counted by its written
    direction, which is what the stoichiometry in the cell graph carries.
    """
    metabolite = model.metabolites.get_by_id(ENTRY_METABOLITE)
    route_ids = {reaction for reaction, _, _ in ROUTE} | {EXCHANGE}
    produces: Counter[str] = Counter()
    consumes: Counter[str] = Counter()
    subsystems: defaultdict[str, set[str]] = defaultdict(set)
    reactions: defaultdict[str, list[str]] = defaultdict(list)
    for reaction in metabolite.reactions:
        if reaction.id in route_ids:
            continue
        for gene in {str(g.id) for g in reaction.genes} & targets:
            if reaction.get_coefficient(ENTRY_METABOLITE) < 0:
                consumes[gene] += 1
            else:
                produces[gene] += 1
            subsystems[gene].add(str(reaction.subsystem) or "unassigned")
            reactions[gene].append(str(reaction.id))
    frame = pd.DataFrame(
        [
            {
                "gene": gene,
                "produces": produces[gene],
                "consumes": consumes[gene],
                "n_reactions": produces[gene] + consumes[gene],
                "subsystems": "; ".join(sorted(subsystems[gene])),
                "reaction_ids": " ".join(sorted(reactions[gene])),
            }
            for gene in sorted(set(produces) | set(consumes))
        ]
    )
    return frame.sort_values(
        ["n_reactions", "gene"], ascending=[False, True]
    ).reset_index(drop=True)


#: Node marker areas in points squared, so a node's printed size does not
#: depend on the data limits. Drawing nodes as data-space patches under
#: ``aspect="equal"`` is what made the first version of this figure letterbox
#: itself into a quarter of its canvas and overlap its own arrows.
METABOLITE_AREA = 46.0
REACTION_AREA = 30.0
HUB_AREA = 82.0

#: Font sizes. Nature's figure band is 5 pt minimum and 7 pt maximum, with
#: 8 pt bold lowercase for panel letters only.
FS_NODE_LABEL = 5.6
FS_REACTION = 5.2
FS_GENE = 5.0
FS_TITLE = 6.0
FS_PANEL_LETTER = 8.0


def _marker_radius_pt(area: float) -> float:
    """Half the drawn width of a scatter marker, in points."""
    return math.sqrt(area / math.pi)


def draw_metabolite(
    ax: Axes, x: float, y: float, label: str, area: float = METABOLITE_AREA
) -> None:
    """A metabolite node: circle, label to its right at a fixed point offset."""
    ax.scatter(
        [x],
        [y],
        s=area,
        marker="o",
        facecolor=PLOT_PALETTE_FILL[4],
        edgecolor=PLOT_PALETTE[4],
        linewidth=0.5,
        zorder=3,
        clip_on=False,
    )
    if label:
        ax.annotate(
            label,
            (x, y),
            xytext=(_marker_radius_pt(area) + 2.5, 0),
            textcoords="offset points",
            fontsize=FS_NODE_LABEL,
            ha="left",
            va="center",
            zorder=4,
            annotation_clip=False,
        )


def draw_reaction(ax: Axes, x: float, y: float, label: str, genes: list[str]) -> None:
    """A reaction node: square, id to the left, the genes its GPR names below it.

    A plasmid gene is set apart by color because it is background in this
    campaign: present in every strain, and targeted by no guide.
    """
    ax.scatter(
        [x],
        [y],
        s=REACTION_AREA,
        marker="s",
        facecolor=PLOT_PALETTE_FILL[0],
        edgecolor=PLOT_PALETTE[0],
        linewidth=0.5,
        zorder=3,
        clip_on=False,
    )
    offset = -(math.sqrt(REACTION_AREA) / 2.0 + 2.5)
    ax.annotate(
        label,
        (x, y),
        xytext=(offset, 2.3 if genes else 0.0),
        textcoords="offset points",
        fontsize=FS_REACTION,
        ha="right",
        va="center",
        zorder=4,
        annotation_clip=False,
    )
    if genes:
        shown = ", ".join(genes[:3]) + (", ..." if len(genes) > 3 else "")
        plasmid = any(gene in PLASMID_GENES for gene in genes)
        ax.annotate(
            shown,
            (x, y),
            xytext=(offset, -3.4),
            textcoords="offset points",
            fontsize=FS_GENE,
            fontstyle="italic",
            color=PLOT_PALETTE[1] if plasmid else "#444444",
            ha="right",
            va="center",
            zorder=4,
            annotation_clip=False,
        )


def arrow(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    shrink_a: float,
    shrink_b: float,
    color: str = "#555555",
) -> None:
    """A directed edge between two nodes.

    ``shrink_a`` and ``shrink_b`` are in points and are the drawn radii of the
    nodes at each end, so the arrow starts and ends clear of them whatever the
    data limits are. That is the whole reason the nodes are markers rather than
    patches: the gap is set in print units, not in data units.
    """
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "linewidth": 0.5,
            "shrinkA": shrink_a + 1.2,
            "shrinkB": shrink_b + 1.2,
            "mutation_scale": 4.0,
        },
        annotation_clip=False,
        zorder=2,
    )


def panel_route(ax: Axes, model: cobra.Model) -> None:
    """Panel a: the route from acetyl-CoA to measured titer, top to bottom."""
    metabolites = [ROUTE[0][1]] + [product for _, _, product in ROUTE]
    x = 0.12
    met_r = _marker_radius_pt(METABOLITE_AREA)
    rxn_r = math.sqrt(REACTION_AREA) / 2.0

    for index, metabolite in enumerate(metabolites):
        draw_metabolite(ax, x, -float(index), METABOLITE_LABELS[metabolite])
    for index, (reaction, _, _) in enumerate(ROUTE):
        y = -(index + 0.5)
        draw_reaction(ax, x, y, reaction, route_genes(model, reaction))
        arrow(ax, (x, -float(index)), (x, y), met_r, rxn_r)
        arrow(ax, (x, y), (x, -float(index + 1)), rxn_r, met_r)

    # The exchange reaction, which is what a measured titer is.
    y_last = -float(len(metabolites) - 1)
    y_exchange = y_last - 0.55
    draw_reaction(ax, x, y_exchange, EXCHANGE, [])
    arrow(ax, (x, y_last), (x, y_exchange), met_r, rxn_r)
    ax.annotate(
        "measured titer",
        (x, y_exchange),
        xytext=(rxn_r + 2.5, 0),
        textcoords="offset points",
        fontsize=FS_GENE,
        color=PLOT_PALETTE[1],
        ha="left",
        va="center",
        annotation_clip=False,
    )
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(y_exchange - 0.3, 0.3)


def panel_competition(
    ax: Axes, frame: pd.DataFrame, names: dict[str, str], total: int
) -> None:
    """Panel b: the CRISPRi targets that carry acetyl-CoA reactions.

    One row per gene, every gene named, since there are only fourteen of them.
    Marker area is proportional to the reaction count, so pcaF-II's 33 of the 49
    read as 33 without covering the panel.

    The edge is an undirected statement, "this gene catalyzes n of acetyl-CoA's
    reactions", and its color says which way those reactions run. An arrowhead
    would assert a direction of flow that a gene-to-metabolite edge does not
    have.
    """
    rows = frame.to_dict("records")
    consumers = [row for row in rows if int(row["consumes"]) >= int(row["produces"])]
    producers = [row for row in rows if int(row["consumes"]) < int(row["produces"])]
    ordered = consumers + producers

    hub_x, hub_y = -0.62, 0.0
    draw_metabolite(ax, hub_x, hub_y, "", area=HUB_AREA)
    for offset, text, size, color, weight in (
        (
            -(_marker_radius_pt(HUB_AREA) + 4.0),
            "acetyl-CoA",
            FS_NODE_LABEL,
            "#000000",
            "bold",
        ),
        (
            -(_marker_radius_pt(HUB_AREA) + 11.0),
            f"{total} reactions",
            FS_GENE,
            "#555555",
            "normal",
        ),
        (
            -(_marker_radius_pt(HUB_AREA) + 17.5),
            f"{int(frame.n_reactions.sum())} knocked down",
            FS_GENE,
            "#555555",
            "normal",
        ),
    ):
        ax.annotate(
            text,
            (hub_x, hub_y),
            xytext=(0, offset),
            textcoords="offset points",
            fontsize=size,
            color=color,
            fontweight=weight,
            ha="center",
            va="center",
            annotation_clip=False,
            zorder=4,
        )

    top, bottom, x = 0.90, -0.90, 0.10
    step = (top - bottom) / max(len(ordered) - 1, 1)
    for index, row in enumerate(ordered):
        count = int(row["n_reactions"])
        y = top - index * step
        color = PLOT_PALETTE[1] if row in consumers else PLOT_PALETTE[2]
        area = 16.0 + 5.5 * count
        ax.plot(
            [hub_x, x],
            [hub_y, y],
            color=color,
            linewidth=0.4,
            alpha=0.75,
            solid_capstyle="round",
            zorder=2,
        )
        ax.scatter(
            [x],
            [y],
            s=area,
            marker="o",
            facecolor=color,
            edgecolor="#333333",
            linewidth=0.3,
            zorder=3,
            clip_on=False,
        )
        ax.annotate(
            f"{names.get(str(row['gene']), str(row['gene']))} ({count})",
            (x, y),
            xytext=(_marker_radius_pt(area) + 2.5, 0),
            textcoords="offset points",
            fontsize=FS_GENE,
            fontstyle="italic",
            ha="left",
            va="center",
            annotation_clip=False,
            zorder=4,
        )

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.05, 1.05)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-timestamp", action="store_true")
    args = parser.parse_args()

    apply_paper_style()
    gem = BiggGEM(ISOPRENOL_MODEL_ID)
    model = gem.model
    targets = target_genes()
    frame = accoa_competition(model, targets)
    names = StringTaxonMirror(
        root=STRING_ROOT, taxon=KT2440_TAXON, release="12.5"
    ).preferred_names()
    total_accoa = len(model.metabolites.get_by_id(ENTRY_METABOLITE).reactions)

    # Nature full width. Height is set by the shorter of the two constraints
    # that matter: the route's 18 nodes need a metabolite-to-reaction gap wider
    # than the two markers' drawn radii, which at these marker areas is about
    # 3.5 mm, and a figure much taller than it is wide reads as a column rather
    # than a figure. 110 mm gives roughly 9.5 mm per step, so the arrows are
    # visible and the proportions stay close to 3:2.
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(110.0)),
        gridspec_kw={"width_ratios": [0.92, 1.08]},
        constrained_layout=True,
    )
    panel_route(axes[0], model)
    panel_competition(axes[1], frame, names, total_accoa)

    titles = (
        "The isoprenol route, as the cell graph stores it",
        "The CRISPRi targets competing for acetyl-CoA",
    )
    for ax, letter, title in zip(axes, ("a", "b"), titles, strict=True):
        ax.text(
            -0.045,
            1.012,
            letter,
            transform=ax.transAxes,
            fontsize=FS_PANEL_LETTER,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
        ax.set_title(title, fontsize=FS_TITLE, loc="left", pad=3.0)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=3.2,
            markerfacecolor=PLOT_PALETTE_FILL[4],
            markeredgecolor=PLOT_PALETTE[4],
            markeredgewidth=0.5,
            label="metabolite node",
        ),
        Line2D(
            [],
            [],
            marker="s",
            linestyle="none",
            markersize=3.0,
            markerfacecolor=PLOT_PALETTE_FILL[0],
            markeredgecolor=PLOT_PALETTE[0],
            markeredgewidth=0.5,
            label="reaction node",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=3.2,
            markerfacecolor=PLOT_PALETTE[1],
            markeredgecolor="#333333",
            markeredgewidth=0.3,
            label="target gene, consumes acetyl-CoA",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=3.2,
            markerfacecolor=PLOT_PALETTE[2],
            markeredgecolor="#333333",
            markeredgewidth=0.3,
            label="target gene, produces acetyl-CoA",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=4,
        frameon=True,
        fontsize=5.0,
        handletextpad=0.5,
    )

    out_dir = osp.join(ASSET_IMAGES_DIR, "048-pputida-isoprenol-cgt")
    os.makedirs(out_dir, exist_ok=True)
    stem = "isoprenol_metabolic_module"
    if not args.no_timestamp:
        stem = f"{stem}_{timestamp()}"
    png = osp.join(out_dir, f"{stem}.png")
    svg = osp.join(out_dir, f"{stem}.svg")
    fig.savefig(png, dpi=400)
    savefig_true_size_svg(fig, svg)
    plt.close(fig)

    frame_path = osp.join(RESULTS, "accoa_competition.csv")
    frame.to_csv(frame_path, index=False)
    print(frame.drop(columns=["reaction_ids"]).to_string(index=False))
    print(png)
    print(svg)
    print(frame_path)


if __name__ == "__main__":
    main()
