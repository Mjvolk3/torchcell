# experiments/044-organism-similarity/scripts/07_figures.py
# [[experiments.044-organism-similarity.scripts.07_figures]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/044-organism-similarity/scripts/07_figures.py

"""Figures 1-5 of experiment 044 (organism similarity) and their captions.

Reads committed result tables (``results/``), the tier-free trees and Peter et al.
2018's sha256-pinned SNP distance matrix (Fig 4); writes PNG +
true-size SVG to ``ASSET_IMAGES_DIR/044-organism-similarity/`` (timestamped until
approved) and ``results/figure_captions.json``.

- Fig 1: the claim strip plot, protein divergence (100 - identity) of each pair, log
  axis, by pair category, with the within-human literature band.
- Fig 2: NCBI taxonomy cladogram + AAI, ortholog Jaccard and EC-4 Jaccard heatmaps.
- Fig 3: taxonomy cladogram vs the AAI neighbor-joining tree (tanglegram).
- Fig 4: (a) neighbor-joining tree over Peter et al. 2018's clades (mean between-clade
  SNP distance from the 1,011 x 1,011 matrix; the isolates with no clade are one more
  leaf), (b) within-clade pairwise SNP distances in the same rows. Writes
  ``results/clade_snp_distance.csv``, ``results/clades_nj.nwk`` and
  ``results/clade_distance_summary.csv``.
- Fig 5: ortholog fraction vs AAI per pair, one color per pair class.
- Fig 6 (the main figure): (a) protein divergence of everything measured from S288C, the
  isolates included, beside the same axis measured from human; (b) the ortholog fraction
  each organism shares with S288C. Writes ``results/s288c_centered.csv``.

Color is the organism group (plan rule). Fig 2's strip uses the plan's groups
(reference, established, emerging, CABBI, genus, mammal, outgroup); Figs 1 and 3 color
by the taxonomic group the claim is about (S. cerevisiae isolates, Saccharomyces,
other Saccharomycotina, non-Saccharomycotina fungus, mammals, bird, bacteria); Fig 1
draws a cross-group pair with the face of one member's color and the edge of the other.
Fig 5 colors by pair class (``PAIR_CLASSES``): a within-group class takes its group's
color, a cross-group class a palette color no group uses. Fig 4 uses one color.

Run from the repo root: ``PYTHONPATH=. ~/miniconda3/envs/torchcell/bin/python ...``.
"""

from __future__ import annotations

import gzip
import json
import os
import os.path as osp
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from Bio import Phylo  # noqa: E402
from Bio.Phylo.TreeConstruction import (  # noqa: E402
    DistanceMatrix,
    DistanceTreeConstructor,
)
from dotenv import load_dotenv  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from torchcell.timestamp import timestamp  # noqa: E402
from torchcell.utils import (  # noqa: E402
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    apply_paper_style,
    mm_to_in,
    savefig_true_size_svg,
)

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from orgsim_manifest import DATA_ROOT, sha256_file  # noqa: E402

load_dotenv()
EXP_DIR = osp.join(os.environ["EXPERIMENT_ROOT"], "044-organism-similarity")
RES = osp.join(EXP_DIR, "results")
OUT_DIR = osp.join(os.environ["ASSET_IMAGES_DIR"], "044-organism-similarity")
STAMP = timestamp()
FS = 6

#: Peter et al. 2018's SNP distance matrix (same pin as 05_trees.py).
PETER_MATRIX = osp.join(
    DATA_ROOT, "data/peter2018/1011DistanceMatrixBasedOnSNPs.tab.gz"
)
PETER_MATRIX_SHA256 = "140da4e5193584c01e60c554a2ba5075a542d925be540afe7c7a92b7377af928"
#: The group of isolates Table S1 gives no clade (Fig 4).
NO_CLADE = "no clade in Table S1"
#: Fig 4b draws a box for a group with at least this many pairs (5 isolates), points below.
MIN_BOX_PAIRS = 10

#: Plan organism groups -> palette (Fig 2 strip).
PLAN_GROUPS = [
    "reference",
    "established",
    "emerging",
    "cabbi",
    "genus",
    "mammal",
    "outgroup",
]
PLAN_COLOR = {g: PLOT_PALETTE[i] for i, g in enumerate(PLAN_GROUPS)}
PLAN_LABEL = {
    "reference": "Reference",
    "established": "Established host",
    "emerging": "Emerging host",
    "cabbi": "CABBI",
    "genus": "Saccharomyces",
    "mammal": "Mammal",
    "outgroup": "Outgroup (bird)",
}

#: Taxonomic groups -> palette (Figs 1, 3, 5).
TAXA = [
    "isolates",
    "saccharomyces",
    "saccharomycotina",
    "other_fungus",
    "mammal",
    "bird",
    "bacteria",
]
TAXON_COLOR = {
    "isolates": PLOT_PALETTE[0],
    "saccharomyces": PLOT_PALETTE[1],
    "saccharomycotina": PLOT_PALETTE[2],
    "other_fungus": PLOT_PALETTE[3],
    "mammal": PLOT_PALETTE[4],
    "bird": PLOT_PALETTE[5],
    "bacteria": PLOT_PALETTE[6],
}
TAXON_LABEL = {
    "isolates": "S. cerevisiae isolates",
    "saccharomyces": "Saccharomyces",
    "saccharomycotina": "Other Saccharomycotina",
    "other_fungus": "Basidiomycete yeast",
    "mammal": "Mammal",
    "bird": "Bird",
    "bacteria": "Bacteria",
}

#: Fig 1 pair categories, in axis order.
CATEGORIES = [
    ("within_scer", "Within S. cerevisiae\n(isolate vs S288C)"),
    ("within_genus", "Within\nSaccharomyces"),
    ("within_mammals", "Within\nmammals"),
    ("mammal_bird", "Mammal\nvs chicken"),
    ("within_saccharomycotina", "Within\nSaccharomycotina"),
    ("within_bacteria", "Within\nbacteria"),
    ("yeast_bacteria", "Saccharomycotina\nvs bacteria"),
]
LABELED = [
    ("hsap_grch38", "ptro", "human/chimp"),
    ("hsap_grch38", "mmul", "human/macaque"),
    ("hsap_grch38", "mmus_c57bl6j", "human/mouse"),
    ("hsap_grch38", "ggal", "human/chicken"),
    ("scer_s288c", "spar_cbs432", "S288C/S. paradoxus"),
    ("scer_s288c", "seub_fm1318", "S288C/S. eubayanus"),
    ("pkud_sd108", "pkud_cbs573", "SD108/CBS573"),
]

#: Fig 5 pair classes, legend order: Fig 1's six columns, then the cross-group classes.
PAIR_CLASSES = [
    ("within_genus", "Within Saccharomyces"),
    ("within_mammals", "Within mammals"),
    ("mammal_bird", "Mammal vs chicken"),
    ("within_saccharomycotina", "Within Saccharomycotina"),
    ("within_bacteria", "Within bacteria"),
    ("yeast_bacteria", "Saccharomycotina vs bacteria"),
    ("yeast_animal", "Saccharomycotina vs amniote"),
    ("animal_bacteria", "Amniote vs bacteria"),
    ("basidio_yeast", "R. toruloides vs Saccharomycotina"),
    ("basidio_animal", "R. toruloides vs amniote"),
    ("basidio_bacteria", "R. toruloides vs bacteria"),
]
#: A within-group class takes its group's Fig 1 color (a Fig 1 point whose face and edge
#: are both that color); a cross-group class takes a palette color no group uses.
PAIR_COLOR = {
    "within_genus": PLOT_PALETTE[1],
    "within_mammals": PLOT_PALETTE[4],
    "within_saccharomycotina": PLOT_PALETTE[2],
    "within_bacteria": PLOT_PALETTE[6],
    "mammal_bird": PLOT_PALETTE[7],
    "yeast_bacteria": PLOT_PALETTE[8],
    "yeast_animal": PLOT_PALETTE[11],
    "animal_bacteria": PLOT_PALETTE[10],
    "basidio_yeast": PLOT_PALETTE[12],
    "basidio_animal": PLOT_PALETTE[13],
    "basidio_bacteria": PLOT_PALETTE[14],
}

#: Fig 6: the human yardstick partners, nearest first.
HUMAN_YARDSTICK = [
    "ptro",
    "ggor",
    "pabe",
    "mmul",
    "mmus_c57bl6j",
    "rnor",
    "cgri_cho",
    "ggal",
]
#: Fig 6 columns (S288C partners by taxonomic group, then the human yardstick).
FIG6_COLUMNS = [
    ("isolates", "{n:,}\nisolates\n(training)"),
    ("saccharomyces", "Saccharo-\nmyces"),
    ("saccharomycotina", "Other\nSaccharo-\nmycotina"),
    ("rtor", "R. toru-\nloides"),
    ("animal", "Mammals,\nchicken"),
    ("bacteria", "Bacteria"),
    ("human_vs", "Human vs"),
]
FIG6_COLUMN = {
    "saccharomyces": "saccharomyces",
    "saccharomycotina": "saccharomycotina",
    "other_fungus": "rtor",
    "mammal": "animal",
    "bird": "animal",
    "bacteria": "bacteria",
}
FIG6_LABELS = {
    "spar_cbs432": "S. paradoxus",
    "seub_fm1318": "S. eubayanus",
    "pkud_sd108": "P. kudriavzevii SD108 (this study)",
    "pkud_cbs573": "P. kudriavzevii CBS573",
    "ecoli_mg1655": "E. coli",
    "pput_kt2440": "P. putida",
    "ptro": "chimp",
    "ggor": "gorilla",
    "pabe": "orangutan",
    "mmul": "macaque",
    "mmus_c57bl6j": "mouse",
    "rnor": "rat",
    "cgri_cho": "hamster",
    "ggal": "chicken",
}
#: Fig 6 S288C-side label positions: (x offset from the column, divergence %, alignment).
FIG6_LABEL_XY = {
    "S. paradoxus": (0.3, 4.0, "left"),
    "S. eubayanus": (0.3, 9.0, "left"),
    "P. kudriavzevii SD108 (this study)": (-0.45, 30.0, "left"),
    "P. kudriavzevii CBS573": (-0.45, 21.0, "left"),
    "E. coli": (0.3, 12.0, "right"),
    "P. putida": (0.3, 17.0, "right"),
}

#: Fig 1 label positions (x = category index, y = divergence %) and alignment.
FIG1_LABEL_XY = {
    "human/chimp": (2.4, 1.6, "left"),
    "human/macaque": (2.4, 4.0, "left"),
    "human/mouse": (2.4, 11.5, "left"),
    "human/chicken": (3.3, 21.0, "left"),
    "S288C/S. paradoxus": (0.72, 7.5, "right"),
    "S288C/S. eubayanus": (0.72, 17.0, "right"),
    "SD108/CBS573": (4.3, 0.44, "left"),
}

#: Fig 5 label positions (data coordinates) and alignment, placed by eye in empty space.
FIG5_LABEL_XY = {
    "SD108/CBS573": (99.5, 1.12, "right"),
    "S288C/S. paradoxus": (91.0, 1.06, "right"),
    "S288C/S. eubayanus": (78.0, 1.01, "right"),
    "human/chimp": (100.5, 0.64, "right"),
    "human/macaque": (89.0, 0.70, "center"),
    "human/mouse": (78.5, 0.64, "center"),
    "human/chicken": (71.0, 0.64, "right"),
}


def style() -> None:
    """Repo figure standard: Arial 6 pt, boxed axes, editable SVG text."""
    apply_paper_style()
    plt.rcParams.update(
        {
            "font.size": FS,
            "axes.labelsize": FS,
            "axes.titlesize": FS,
            "xtick.labelsize": FS,
            "ytick.labelsize": FS,
            "legend.fontsize": FS,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
            "svg.fonttype": "none",
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )


def save(fig: plt.Figure, name: str, pdf: bool = False) -> str:
    """PNG + true-size SVG (+ a PDF at the figure's size); returns the SVG path."""
    os.makedirs(OUT_DIR, exist_ok=True)
    base = osp.join(OUT_DIR, f"{name}_{STAMP}")
    fig.savefig(base + ".png", dpi=300)
    if pdf:
        fig.savefig(base + ".pdf", format="pdf")
    savefig_true_size_svg(fig, base + ".svg")
    plt.close(fig)
    return base + ".svg"


def load() -> dict:
    """Every result table the figures read."""
    lin = pd.read_csv(osp.join(RES, "organism_lineage.csv")).set_index("key")
    org = json.load(open(osp.join(RES, "organisms.json")))["organisms"]
    display = {o["key"]: o["display"] for o in org}
    taxon = {}
    for k, r in lin.iterrows():
        if r.is_saccharomyces:
            taxon[k] = "saccharomyces"
        elif r.is_saccharomycotina:
            taxon[k] = "saccharomycotina"
        elif r.is_fungi:
            taxon[k] = "other_fungus"
        elif r.is_mammalia:
            taxon[k] = "mammal"
        elif r.is_bacteria:
            taxon[k] = "bacteria"
        elif r["class"] == "Aves":
            taxon[k] = "bird"
        else:
            taxon[k] = "protist"
    pw = pd.read_csv(osp.join(RES, "pairwise_similarity.csv"))
    iso = pd.read_parquet(osp.join(RES, "isolate_protein_identity.parquet"))
    lit = json.load(open(osp.join(RES, "literature_values.json")))
    sc = pd.read_csv(osp.join(RES, "busco_aai.csv"))
    return {
        "lin": lin,
        "display": display,
        "taxon": taxon,
        "pw": pw,
        "iso": iso,
        "lit": lit,
        "sc": sc,
    }


def pair_category(i: str, j: str, lin: pd.DataFrame, taxon: dict[str, str]) -> str:
    """Pair class of an organism pair; every pair of two proteomes has exactly one.

    The first six are Fig 1's columns. The rest are the cross-group pairs Fig 1 does not
    plot, so Fig 5 can color all 528 pairs.
    """
    a, b = lin.loc[i], lin.loc[j]
    if a.is_saccharomyces and b.is_saccharomyces:
        return "within_genus"
    if a.is_mammalia and b.is_mammalia:
        return "within_mammals"
    if (a.is_mammalia and b["class"] == "Aves") or (
        b.is_mammalia and a["class"] == "Aves"
    ):
        return "mammal_bird"
    if a.is_saccharomycotina and b.is_saccharomycotina:
        return "within_saccharomycotina"
    if a.is_bacteria and b.is_bacteria:
        return "within_bacteria"
    pair = {taxon[i], taxon[j]}
    yeast = {"saccharomyces", "saccharomycotina"}
    animal = {"mammal", "bird"}
    if pair & yeast and "bacteria" in pair:
        return "yeast_bacteria"
    if pair & yeast and pair & animal:
        return "yeast_animal"
    if pair & animal and "bacteria" in pair:
        return "animal_bacteria"
    if "other_fungus" in pair:
        rest = pair - {"other_fungus"}
        if rest & yeast:
            return "basidio_yeast"
        if rest & animal:
            return "basidio_animal"
        if rest == {"bacteria"}:
            return "basidio_bacteria"
    raise ValueError(f"pair {i}/{j} ({pair}) has no class")


def pair_table(d: dict) -> pd.DataFrame:
    """Pairwise table with category and taxon colors."""
    pw = d["pw"].copy()
    pw["category"] = [
        pair_category(i, j, d["lin"], d["taxon"]) for i, j in zip(pw.org_i, pw.org_j)
    ]
    pw["taxon_i"] = pw.org_i.map(d["taxon"])
    pw["taxon_j"] = pw.org_j.map(d["taxon"])
    return pw


def fig1(d: dict) -> tuple[str, str, dict]:
    """The claim strip plot."""
    pw = pair_table(d)
    sc = d["sc"].set_index(["org_i", "org_j"])
    iso = d["iso"]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(92)))
    xs = {c: k for k, (c, _) in enumerate(CATEGORIES)}
    # within S. cerevisiae: every isolate vs S288C, mean protein identity over ORFs
    y = 100.0 * (1.0 - iso["mean_identity"].to_numpy())
    x = xs["within_scer"] + rng.uniform(-0.3, 0.3, len(y))
    ax.scatter(x, y, s=2, lw=0, color=TAXON_COLOR["isolates"], alpha=0.8, zorder=2)
    stats = {
        "within_scer": {
            "n": int(len(y)),
            "median_divergence_pct": float(np.median(y)),
            "min_divergence_pct": float(y.min()),
            "max_divergence_pct": float(y.max()),
        }
    }
    pos: dict[tuple[str, str], tuple[float, float]] = {}
    for cat, _ in CATEGORIES[1:]:
        sub = pw[pw.category == cat]
        yy = 100.0 - sub["aai_mean"].to_numpy()
        xx = xs[cat] + rng.uniform(-0.25, 0.25, len(sub))
        for (_, r), xv, yv in zip(sub.iterrows(), xx, yy):
            face = TAXON_COLOR[r.taxon_i]
            edge = TAXON_COLOR[r.taxon_j]
            ax.scatter(xv, yv, s=9, facecolor=face, edgecolor=edge, lw=0.8, zorder=3)
            pos[(r.org_i, r.org_j)] = (xv, yv)
            key = (r.org_i, r.org_j)
            if key in sc.index and not np.isnan(sc.loc[key, "aai_sc_mean"]):
                ax.scatter(
                    xv,
                    100.0 - sc.loc[key, "aai_sc_mean"],
                    s=18,
                    marker="D",
                    facecolor="none",
                    edgecolor="black",
                    lw=0.4,
                    zorder=3,
                )
        stats[cat] = {
            "n": int(len(sub)),
            "median_aai": float(sub["aai_mean"].median()),
            "min_aai": float(sub["aai_mean"].min()),
            "max_aai": float(sub["aai_mean"].max()),
        }
    for i, j, label in LABELED:
        m = ((pw.org_i == i) & (pw.org_j == j)) | ((pw.org_i == j) & (pw.org_j == i))
        r = pw[m].iloc[0]
        xv, yv = pos[(r.org_i, r.org_j)]
        tx, ty, ha = FIG1_LABEL_XY[label]
        ax.annotate(
            f"{label} {r.aai_mean:.1f}%",
            xy=(xv, yv),
            xytext=(tx, ty),
            textcoords="data",
            fontsize=FS,
            ha=ha,
            va="center",
            arrowprops={"arrowstyle": "-", "lw": 0.4, "color": "black"},
        )
    lo, hi = d["lit"]["protein_identity_vs_reference"]
    ax.axhspan(
        100 * (1 - hi),
        100 * (1 - lo),
        color=PLOT_PALETTE[5],
        alpha=0.25,
        lw=0,
        zorder=1,
    )
    ax.text(
        0.75,
        100 * (1 - hi) * 0.93,
        "one human genome vs GRCh38, peptide-altering sites (1000 Genomes 2015; "
        "literature value, not computed here)",
        ha="left",
        va="bottom",
        fontsize=FS,
    )
    ax.set_yscale("log")
    ax.set_ylim(0.03, 100)
    ax.set_yticks([0.1, 1, 10, 100])
    ax.set_yticklabels(["0.1", "1", "10", "100"])
    ax.set_ylabel("Protein divergence, 100 - identity (%)")
    ax.invert_yaxis()
    sec = ax.secondary_yaxis("right", functions=(lambda v: v, lambda v: v))
    sec.set_yticks([0.1, 1, 10, 50])
    sec.set_yticklabels(["99.9", "99", "90", "50"])
    sec.set_ylabel("Identity (%)")
    ax.set_xticks(range(len(CATEGORIES)))
    ax.set_xticklabels([lab for _, lab in CATEGORIES])
    ax.set_xlim(-0.6, len(CATEGORIES) - 0.4)
    ax.grid(axis="y", which="major", lw=0.3, color="#CCCCCC")
    # legend lists only the groups that color a point here (R. toruloides is in no column)
    plotted = pw[pw.category.isin([c for c, _ in CATEGORIES[1:]])]
    fig1_taxa = {"isolates"} | set(plotted.taxon_i) | set(plotted.taxon_j)
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            markersize=3,
            markerfacecolor=TAXON_COLOR[t],
            markeredgecolor=TAXON_COLOR[t],
            label=TAXON_LABEL[t],
        )
        for t in TAXA
        if t in fig1_taxa
    ] + [
        Line2D(
            [],
            [],
            marker="D",
            ls="",
            markersize=2.5,
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=0.4,
            label="single-copy Eukaryota OG AAI",
        ),
        Patch(
            facecolor=PLOT_PALETTE[5], alpha=0.25, label="within-human literature band"
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        frameon=False,
        ncol=5,
        handletextpad=0.3,
        columnspacing=0.8,
    )
    fig.subplots_adjust(left=0.07, right=0.93, bottom=0.25, top=0.97)
    path = save(fig, "fig1_claim_protein_divergence")
    ref = stats
    caption = (
        "Fig. 1 | Protein divergence within S. cerevisiae, within Saccharomyces and within "
        "mammals on one axis. Each point is a pair; the y axis is 100 minus protein "
        "identity (log scale, inverted; identity on the right). Within S. cerevisiae "
        f"(n = {ref['within_scer']['n']:,} isolates): the mean over S288C reference ORFs "
        "of each isolate's protein identity to S288C, from Peter 2018's reference-inferred "
        "alleles, with every gap column of a frameshifted or indel-bearing ORF charged "
        "(06_isolate_protein_identity.py); the per-isolate MEDIAN over ORFs, which "
        "frameshifted ORFs do not pull down, gives a median divergence of "
        f"{100 * (1 - iso['median_identity'].median()):.2f}%. Every other column: AAI, the mean "
        "percent identity over reciprocal best mmseqs2 hits of one-protein-per-gene "
        "proteomes (03_mmseqs_rbh.py); n pairs per column "
        + ", ".join(
            f"{lab.replace(chr(10), ' ')} {ref[c]['n']}" for c, lab in CATEGORIES[1:]
        )
        + ". Open diamonds: AAI restricted to Eukaryota-level eggNOG OGs that are single "
        "copy in both members, the proxy for Peris et al. 2023's BUSCO AAI. Point face "
        "and edge colors are the taxonomic groups of the two members. Gray band: one "
        "human genome differs from GRCh38 at 10,000 to 12,000 peptide-altering sites "
        "(1000 Genomes Project 2015, doi:10.1038/nature15393; literature value, not "
        "computed here), divided by this study's 11.7 million-residue human per-gene "
        "proteome."
    )
    return path, caption, stats


def leaf_order(tree: Phylo.BaseTree.Tree) -> list[str]:
    """Terminal names top to bottom."""
    return [t.name for t in tree.get_terminals()]


def draw_cladogram(ax, tree, flip: bool = False, colors: dict | None = None) -> dict:
    """Rectangular cladogram, leaves aligned at x = depth; returns {leaf: y}."""
    depths = tree.depths(unit_branch_lengths=True)
    max_d = max(depths.values())
    ys: dict = {}
    for k, t in enumerate(tree.get_terminals()):
        ys[t] = k

    def y_of(c):
        if c in ys:
            return ys[c]
        ys[c] = float(np.mean([y_of(ch) for ch in c.clades]))
        return ys[c]

    y_of(tree.root)

    def x_of(c):
        return max_d if c.is_terminal() else depths[c]

    sign = -1 if flip else 1
    for c in tree.find_clades():
        if c.is_terminal():
            continue
        xc = sign * x_of(c)
        yc = [ys[ch] for ch in c.clades]
        ax.plot([xc, xc], [min(yc), max(yc)], color="black", lw=0.5)
        for ch in c.clades:
            ax.plot([xc, sign * x_of(ch)], [ys[ch], ys[ch]], color="black", lw=0.5)
    ax.set_ylim(len(tree.get_terminals()) - 0.5, -0.5)
    ax.axis("off")
    return {t.name: ys[t] for t in tree.get_terminals()}


def untangle(tree: Phylo.BaseTree.Tree, target: dict[str, int]) -> None:
    """Rotate every node's children by the mean target position of their leaves, so a
    tanglegram's lines cross only where the two topologies disagree.
    """
    for clade in tree.find_clades(order="postorder"):
        if clade.clades:
            clade.clades.sort(
                key=lambda c: np.mean([target[t.name] for t in c.get_terminals()])
            )


def fig2(d: dict) -> tuple[str, str]:
    """Cladogram + three heatmaps."""
    tree = Phylo.read(osp.join(RES, "taxonomy_cladogram.nwk"), "newick")
    order = leaf_order(tree)
    pw = d["pw"]
    n = len(order)
    idx = {k: i for i, k in enumerate(order)}
    mats = {}
    for col in ("aai_mean", "ortholog_jaccard", "ec4_jaccard"):
        m = np.full((n, n), np.nan)
        for i, j, v in zip(pw.org_i, pw.org_j, pw[col]):
            m[idx[i], idx[j]] = m[idx[j], idx[i]] = v
        mats[col] = m
    W = PANEL_WIDTHS_MM["full"]
    fig = plt.figure(figsize=(mm_to_in(W), mm_to_in(110)))
    gs = fig.add_gridspec(
        1,
        6,
        width_ratios=[0.9, 0.12, 1.15, 1, 1, 1],
        wspace=0.14,
        left=0.01,
        right=0.985,
        top=0.89,
        bottom=0.08,
    )
    ax_t = fig.add_subplot(gs[0])
    draw_cladogram(ax_t, tree)
    ax_s = fig.add_subplot(gs[1])
    for k, key in enumerate(order):
        g = d["lin"].loc[key, "color_group"]
        ax_s.add_patch(plt.Rectangle((0, k - 0.5), 1, 1, color=PLAN_COLOR[g], lw=0))
    ax_s.set_xlim(0, 1)
    ax_s.set_ylim(n - 0.5, -0.5)
    ax_s.axis("off")
    ax_l = fig.add_subplot(gs[2])
    for k, key in enumerate(order):
        absent = "" if d["lin"].loc[key, "has_proteome"] else " (no proteome)"
        ax_l.text(
            0.02,
            k,
            d["display"][key] + absent,
            va="center",
            fontsize=FS,
            style="italic" if absent == "" else "normal",
        )
    ax_l.set_ylim(n - 0.5, -0.5)
    ax_l.axis("off")
    titles = {
        "aai_mean": ("AAI (%)", "Oranges", (30, 100), [40, 60, 80]),
        "ortholog_jaccard": ("Ortholog Jaccard", "Purples", (0, 1), [0, 0.5, 1]),
        "ec4_jaccard": ("EC-4 Jaccard", "Reds", (0, 1), [0, 0.5, 1]),
    }
    for k, col in enumerate(("aai_mean", "ortholog_jaccard", "ec4_jaccard")):
        ax = fig.add_subplot(gs[3 + k])
        title, cmap, (vmin, vmax), ticks = titles[col]
        cm = plt.get_cmap(cmap).copy()
        cm.set_bad("#E0E0E0")
        im = ax.imshow(
            np.ma.masked_invalid(mats[col]),
            cmap=cm,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.5)
        cax = ax.inset_axes([0.0, 1.03, 1.0, 0.025])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=ticks)
        cb.ax.xaxis.set_ticks_position("top")
        cb.ax.tick_params(labelsize=FS, length=2, width=0.4)
        cb.outline.set_linewidth(0.4)
        cb.set_label(title, fontsize=FS, labelpad=2)
        cb.ax.xaxis.set_label_position("top")
    handles = [Patch(color=PLAN_COLOR[g], label=PLAN_LABEL[g]) for g in PLAN_GROUPS]
    fig.legend(
        handles=handles,
        loc="lower left",
        ncol=7,
        frameon=False,
        fontsize=FS,
        bbox_to_anchor=(0.01, 0.0),
        handlelength=1,
        columnspacing=0.8,
    )
    path = save(fig, "fig2_cladogram_heatmaps")
    caption = (
        "Fig. 2 | The organisms torchcell serves or targets on three similarity axes, "
        "ordered by the NCBI taxonomy. Left: NCBI taxonomy cladogram (topology only; "
        "ete3 NCBITaxa, 05_trees.py) of the 34 reference organisms; the strip is the "
        "plan's first group per organism. Heatmaps, rows and columns in cladogram order: "
        "AAI (mean percent identity over reciprocal best hits, 03_mmseqs_rbh.py), "
        "ortholog Jaccard (RBH count over the union of the two per-gene proteomes) and "
        "EC-4 Jaccard (complete EC numbers from one eggNOG-mapper 2 run, eggNOG 5.0.2, "
        "04_emapper.py). Gray: no proteome (A. limacinum has no annotated assembly at "
        "NCBI) or the diagonal."
    )
    return path, caption


def fig3(d: dict) -> tuple[str, str]:
    """Cladogram vs AAI NJ tanglegram."""
    t1 = Phylo.read(osp.join(RES, "taxonomy_cladogram.nwk"), "newick")
    t2 = Phylo.read(osp.join(RES, "aai_nj.nwk"), "newick")
    keep = {t.name for t in t2.get_terminals()}
    for t in list(t1.get_terminals()):
        if t.name not in keep:
            t1.prune(t)
    left_y = {t.name: k for k, t in enumerate(t1.get_terminals())}
    untangle(t2, left_y)
    fig = plt.figure(figsize=(mm_to_in(PANEL_WIDTHS_MM["half"]), mm_to_in(95)))
    gs = fig.add_gridspec(
        1,
        4,
        width_ratios=[0.8, 1.05, 0.45, 0.8],
        wspace=0.0,
        left=0.02,
        right=0.98,
        top=0.95,
        bottom=0.02,
    )
    a1, al, a2, a3 = (fig.add_subplot(gs[k]) for k in range(4))
    y1 = draw_cladogram(a1, t1)
    y2 = draw_cladogram(a3, t2, flip=True)
    a1.set_title("NCBI taxonomy", fontsize=FS)
    a3.set_title("AAI neighbor joining", fontsize=FS)
    n = len(y1)
    for name, ya in y1.items():
        al.text(0.02, ya, d["display"][name], fontsize=FS, va="center", ha="left")
        c = TAXON_COLOR[d["taxon"][name]]
        a2.plot([0, 1], [ya, y2[name]], color=c, lw=0.6)
    for ax in (al, a2):
        ax.set_xlim(0, 1)
        ax.set_ylim(n - 0.5, -0.5)
        ax.axis("off")
    path = save(fig, "fig3_taxonomy_vs_aai_nj")
    caption = (
        "Fig. 3 | NCBI taxonomy and AAI agree on the large groups. Left: NCBI taxonomy "
        "cladogram (topology only) over the 33 organisms with a proteome. Right: "
        "neighbor-joining tree (Biopython) on d = -ln(AAI/100), rooted with the bacteria "
        "as outgroup, drawn as topology with children rotated to follow the left tree. Lines join the same organism and are "
        "colored by its taxonomic group (05_trees.py)."
    )
    return path, caption


def clade_distances() -> dict:
    """Clade-level SNP distances from Peter et al. 2018's 1,011 x 1,011 matrix.

    Every Table S1 clade is its own group, and the isolates with no clade (ATK, the one
    unmatched code, included) form one more, so the groups partition the 1,011 isolates.
    Between groups: the mean over every cross pair. Within a group: every pair of its
    members. Writes ``results/clade_snp_distance.csv`` (the group matrix, diagonal 0).
    """
    assert sha256_file(PETER_MATRIX) == PETER_MATRIX_SHA256
    with gzip.open(PETER_MATRIX, "rt") as fh:
        mat = pd.read_csv(fh, sep="\t", index_col=0)
    clades = pd.read_csv(osp.join(RES, "isolate_clades.csv"))
    assert list(clades["isolate"]) == list(mat.index), "clade table order differs"
    m = mat.to_numpy()
    lab = clades["clade"].fillna(NO_CLADE).to_numpy()
    groups = list(pd.Series(lab).value_counts().index)
    members = {g: np.where(lab == g)[0] for g in groups}
    between = pd.DataFrame(0.0, index=groups, columns=groups)
    for a in groups:
        for b in groups:
            if a != b:
                between.loc[a, b] = float(m[np.ix_(members[a], members[b])].mean())
    between.index.name = "clade"
    between.to_csv(osp.join(RES, "clade_snp_distance.csv"))
    within = {
        g: m[np.ix_(ix, ix)][np.triu_indices(len(ix), 1)] for g, ix in members.items()
    }
    iu = np.triu_indices(len(m), 1)
    return {
        "m": m,
        "members": members,
        "between": between,
        "within": within,
        "all_pairs": m[iu],
        "isolates": list(mat.index),
    }


def clade_tree(between: pd.DataFrame) -> Phylo.BaseTree.Tree:
    """Biopython NJ on the clade matrix, midpoint rooted, ladderized; writes the tree.

    NJ runs on positional ids (``c00`` ...), so the Newick writer never sees a clade name's
    spaces or periods during the build; the names are put back before writing.
    """
    names = list(between.index)
    ids = [f"c{k:02d}" for k in range(len(names))]
    lower = [
        [float(between.iloc[i, j]) for j in range(i)] + [0.0] for i in range(len(names))
    ]
    tree = DistanceTreeConstructor().nj(DistanceMatrix(ids, lower))
    tree.root_at_midpoint()
    tree.ladderize()
    back = dict(zip(ids, names))
    for t in tree.get_terminals():
        t.name = back[t.name]
    for c in tree.get_nonterminals():
        c.name = None
    Phylo.write(tree, osp.join(RES, "clades_nj.nwk"), "newick")
    return tree


def fig4(d: dict) -> tuple[str, str, dict]:
    """Clade-level isolate tree (a) and within-clade SNP distances (b)."""
    cd = clade_distances()
    tree = clade_tree(cd["between"])
    leaves = [t.name for t in tree.get_terminals()]
    n_iso = {g: len(ix) for g, ix in cd["members"].items()}
    div = pd.read_parquet(
        osp.join(
            EXP_DIR,
            "..",
            "018-natural-isolate-genomics",
            "results",
            "per_strain_divergence_summary.parquet",
        )
    )
    nearest = div.sort_values("mean_divergence").iloc[0]
    near_idx = [
        k
        for k, iso in enumerate(cd["isolates"])
        if iso.removeprefix("SACE_") == nearest.strain
    ]
    assert len(near_idx) == 1, near_idx
    near_clade = next(g for g, ix in cd["members"].items() if near_idx[0] in ix)
    overall = float(np.median(cd["all_pairs"]))
    # per-clade summary, in leaf order
    rows = []
    for g in leaves:
        others = cd["between"].loc[g].drop(g)
        nn = str(others.idxmin())
        cross = cd["m"][np.ix_(cd["members"][g], cd["members"][nn])].ravel()
        w = cd["within"][g]
        rows.append(
            {
                "clade": g,
                "n_isolates": n_iso[g],
                "n_within_pairs": int(len(w)),
                "median_within": float(np.median(w)) if len(w) else np.nan,
                "nearest_other_clade": nn,
                "mean_to_nearest_other_clade": float(others.min()),
                "median_to_nearest_other_clade": float(np.median(cross)),
                "contains_nearest_to_s288c": g == near_clade,
            }
        )
    summ = pd.DataFrame(rows)
    summ.to_csv(osp.join(RES, "clade_distance_summary.csv"), index=False)

    W = PANEL_WIDTHS_MM["full"]
    fig = plt.figure(figsize=(mm_to_in(W), mm_to_in(100)))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[0.8, 0.72, 1.2],
        wspace=0.0,
        left=0.02,
        right=0.98,
        top=0.93,
        bottom=0.12,
    )
    ax_t, ax_l, ax_b = (fig.add_subplot(gs[k]) for k in range(3))
    n = len(leaves)
    # (a) phylogram; NJ can return slightly negative branches, drawn at zero length
    depths = tree.depths()
    neg = [c.branch_length for c in tree.find_clades() if (c.branch_length or 0) < 0]
    ys = {t: k for k, t in enumerate(tree.get_terminals())}

    def y_of(c):
        if c in ys:
            return ys[c]
        ys[c] = float(np.mean([y_of(ch) for ch in c.clades]))
        return ys[c]

    y_of(tree.root)
    xpos = {}
    for c in tree.find_clades(order="preorder"):
        xpos[c] = depths[c]
    for c in tree.find_clades(order="preorder"):
        for ch in c.clades:
            xpos[ch] = max(xpos[ch], xpos[c])
    for c in tree.find_clades():
        if c.is_terminal():
            continue
        yc = [ys[ch] for ch in c.clades]
        ax_t.plot([xpos[c], xpos[c]], [min(yc), max(yc)], color="black", lw=0.5)
        for ch in c.clades:
            ax_t.plot([xpos[c], xpos[ch]], [ys[ch], ys[ch]], color="black", lw=0.5)
    xmax = max(xpos[t] for t in tree.get_terminals())
    for t in tree.get_terminals():
        ax_t.plot(
            [xpos[t], xmax * 1.04],
            [ys[t], ys[t]],
            color="#BBBBBB",
            lw=0.3,
            ls=(0, (1, 1.5)),
        )
        if t.name == near_clade:
            ax_t.scatter(
                xpos[t], ys[t], marker="o", s=10, color="black", zorder=4, lw=0
            )
    ax_t.set_xlim(0, xmax * 1.04)
    ax_t.set_ylim(n - 0.5, -0.5)
    ax_t.set_yticks([])
    for s in ("top", "right", "left"):
        ax_t.spines[s].set_visible(False)
    ax_t.spines["bottom"].set_linewidth(0.5)
    ax_t.set_xlabel("Branch length (% SNP distance)")
    # leaf labels
    for g in leaves:
        lab = f"{g} (n = {n_iso[g]:,})"
        if g == near_clade:
            lab = f"{g} (n = {n_iso[g]:,}, holds {nearest.strain})"
        ax_l.text(
            0.02,
            ys[next(t for t in tree.get_terminals() if t.name == g)],
            lab,
            va="center",
            ha="left",
            fontsize=FS,
            fontweight="bold" if g == near_clade else "normal",
        )
    ax_l.set_xlim(0, 1)
    ax_l.set_ylim(n - 0.5, -0.5)
    ax_l.axis("off")
    # (b) within-clade distances, same rows
    box_color = PLOT_PALETTE[0]
    rng = np.random.default_rng(0)
    box_rows, strip_rows = [], []
    for k, g in enumerate(leaves):
        w = cd["within"][g]
        if len(w) >= MIN_BOX_PAIRS:
            box_rows.append((k, w))
        elif len(w) > 0:
            strip_rows.append((k, w))
        else:
            ax_b.text(1.5e-4, k, "n = 1, no pair", va="center", ha="left", fontsize=FS)
    ax_b.boxplot(
        [w for _, w in box_rows],
        positions=[k for k, _ in box_rows],
        vert=False,
        widths=0.6,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": box_color, "edgecolor": "black", "linewidth": 0.5},
        medianprops={"color": "black", "linewidth": 0.8},
        whiskerprops={"color": "black", "linewidth": 0.5},
        capprops={"color": "black", "linewidth": 0.5},
    )
    for k, w in strip_rows:
        ax_b.scatter(
            w,
            k + rng.uniform(-0.15, 0.15, len(w)),
            s=6,
            facecolor=box_color,
            edgecolor="black",
            lw=0.3,
            zorder=3,
        )
    ax_b.axvline(overall, color=PLOT_PALETTE[5], lw=0.8, ls="--", zorder=1)
    ax_b.text(
        overall,
        -0.9,
        f"median of all {len(cd['all_pairs']):,} pairs, {overall:.2f}% ",
        ha="right",
        va="bottom",
        fontsize=FS,
        color=PLOT_PALETTE[5],
    )
    ax_b.set_xscale("log")
    ax_b.set_xlim(1e-4, 3)
    ax_b.set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1])
    ax_b.set_xticklabels(["0.0001", "0.001", "0.01", "0.1", "1"])
    ax_b.set_ylim(n - 0.5, -0.5)
    ax_b.set_yticks(range(n))
    ax_b.set_yticklabels([])
    ax_b.tick_params(axis="y", length=0)
    ax_b.grid(axis="x", which="major", lw=0.3, color="#DDDDDD")
    ax_b.set_axisbelow(True)
    ax_b.set_xlabel("Within-clade pairwise SNP distance (%)")
    for s in ax_b.spines.values():
        s.set_linewidth(0.5)
    for ax, letter in ((ax_t, "a"), (ax_b, "b")):
        ax.text(
            -0.02 if ax is ax_t else -0.01,
            1.035,
            letter,
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            va="bottom",
            ha="left",
        )
    path = save(fig, "fig4_clade_tree_within_distance")
    with_med = summ[summ.n_within_pairs > 0]
    named = with_med[with_med.clade != NO_CLADE]
    pooled_named = np.concatenate([cd["within"][g] for g in named.clade])
    stats = {
        "nearest_to_s288c": str(nearest.strain),
        "nearest_mean_divergence": float(nearest.mean_divergence),
        "nearest_clade": near_clade,
        "n_leaves": int(n),
        "n_named_clades": int((summ.clade != NO_CLADE).sum()),
        "n_isolates_no_clade": int(n_iso[NO_CLADE]),
        "n_isolates": int(len(cd["isolates"])),
        "n_all_pairs": int(len(cd["all_pairs"])),
        "median_all_pairs": overall,
        "n_box_clades": int(len(box_rows)),
        "n_strip_clades": int(len(strip_rows)),
        "min_box_pairs": MIN_BOX_PAIRS,
        "n_named_clades_with_pairs": int(len(named)),
        "median_of_clade_medians": float(named.median_within.median()),
        "min_clade_median": float(named.median_within.min()),
        "min_clade_median_clade": str(named.loc[named.median_within.idxmin(), "clade"]),
        "max_clade_median": float(named.median_within.max()),
        "max_clade_median_clade": str(named.loc[named.median_within.idxmax(), "clade"]),
        "pooled_within_named_median": float(np.median(pooled_named)),
        "n_pooled_within_named": int(len(pooled_named)),
        "n_named_below_overall": int((named.median_within < overall).sum()),
        "no_clade_median_within": float(
            summ.set_index("clade").loc[NO_CLADE, "median_within"]
        ),
        "n_negative_branches": int(len(neg)),
        "min_negative_branch": float(min(neg)) if neg else 0.0,
        "tree_height": float(xmax),
    }
    caption = (
        "Fig. 4 | Every named S. cerevisiae clade but one is tighter than the species as "
        "a whole. "
        f"(a) Neighbor-joining tree (Biopython) over the {stats['n_leaves']} groups of "
        f"Peter et al. 2018's {stats['n_isolates']:,} isolates: the "
        f"{stats['n_named_clades']} clades of their Supplementary Table 1 (Wine/European "
        f"subclades merged) and the {stats['n_isolates_no_clade']} isolates the table "
        "gives no clade; the distance between two groups is the mean SNP distance "
        "(percent non-identical bases, heterozygous differences half-weighted) over every "
        "pair of their members, midpoint rooted, branch lengths in percent. Bold leaf and "
        f"black dot: the clade holding {nearest.strain}, the isolate with the lowest mean "
        f"per-ORF nucleotide divergence from S288C in experiment 018 "
        f"({100 * nearest.mean_divergence:.4f}%); S288C itself is not in Peter's matrix. "
        f"(b) SNP distance between every pair of isolates within a group, same rows. "
        f"{stats['n_named_below_overall']} of the {stats['n_named_clades_with_pairs']} "
        "named clades with at least one pair have a median below the median over all "
        f"{stats['n_all_pairs']:,} isolate pairs ({overall:.2f}%, dashed gray line); "
        f"clade medians run from {stats['min_clade_median']:.4f}% "
        f"({stats['min_clade_median_clade']}) to {stats['max_clade_median']:.2f}% "
        f"({stats['max_clade_median_clade']}), median {stats['median_of_clade_medians']:.2f}%. "
        f"Boxes (groups with at least {MIN_BOX_PAIRS} pairs): median, quartiles and "
        "whiskers at one and a half times the interquartile range, outliers not drawn; points: every "
        "pair of a smaller group. One color for every group, since color carries no "
        "category here. Log axis."
    )
    return path, caption, stats


def fig5(d: dict) -> tuple[str, str, dict]:
    """Ortholog fraction vs AAI, one color per pair class."""
    pw = pair_table(d)
    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_WIDTHS_MM["half"]), mm_to_in(92)))
    n_class = pw["category"].value_counts()
    by_size = sorted(PAIR_CLASSES, key=lambda c: -int(n_class.get(c[0], 0)))
    for cls, _ in by_size:
        sub = pw[pw.category == cls]
        ax.scatter(
            sub.aai_mean,
            sub.ortholog_fraction,
            s=7,
            facecolor=PAIR_COLOR[cls],
            edgecolor="black",
            lw=0.3,
            zorder=3,
        )
    for i, j, label in LABELED:
        m = ((pw.org_i == i) & (pw.org_j == j)) | ((pw.org_i == j) & (pw.org_j == i))
        r = pw[m].iloc[0]
        tx, ty, ha = FIG5_LABEL_XY[label]
        ax.annotate(
            label,
            xy=(r.aai_mean, r.ortholog_fraction),
            xytext=(tx, ty),
            textcoords="data",
            fontsize=FS,
            ha=ha,
            va="center",
            arrowprops={"arrowstyle": "-", "lw": 0.3},
        )
    ax.set_xlabel("AAI over reciprocal best hits (%)")
    ax.set_ylabel("Ortholog fraction (reciprocal best hits / smaller proteome)")
    ax.set_ylim(0, 1.16)
    from matplotlib.ticker import MultipleLocator

    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(axis="y", which="minor", length=0)
    ax.grid(axis="y", which="both", lw=0.3, color="#DDDDDD")
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            markersize=3,
            markerfacecolor=PAIR_COLOR[cls],
            markeredgecolor="black",
            markeredgewidth=0.3,
            label=f"{lab} ({int(n_class.get(cls, 0))})",
        )
        for cls, lab in PAIR_CLASSES
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(-0.13, -0.11),
        frameon=False,
        fontsize=FS,
        ncol=2,
        handletextpad=0.2,
        columnspacing=0.6,
        title="Pair class (n pairs)",
        title_fontsize=FS,
        alignment="left",
    )
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.33, top=0.98)
    path = save(fig, "fig5_ortholog_fraction_vs_aai")
    stats = {
        "n_pairs": int(len(pw)),
        "n_per_class": {cls: int(n_class.get(cls, 0)) for cls, _ in PAIR_CLASSES},
    }
    caption = (
        "Fig. 5 | Shared orthologs fall with AAI. Each point"
        f" is one of the {len(pw)} pairs of the 33 "
        "proteomes: AAI, the mean percent identity over reciprocal best mmseqs2 hits, "
        "against the ortholog fraction, the number of reciprocal best hits over the gene "
        "count of the smaller per-gene proteome (03_mmseqs_rbh.py). Color is the pair "
        "class (n pairs in the legend; amniote is a mammal or the chicken); a "
        "within-group class takes that group's Fig. 1 color, and a cross-group class "
        "takes a color no group uses. Points are drawn largest class first; the "
        "cross-kingdom pairs at the lowest AAI overlap. "
        + ", ".join(f"{lab} {int(n_class.get(cls, 0))}" for cls, lab in PAIR_CLASSES)
        + "."
    )
    return path, caption, stats


def spread_log(ys: list[float], gap: float) -> list[float]:
    """Label y positions on a log axis, pushed apart to at least ``gap`` decades, in order."""
    order = np.argsort(ys)
    ly = np.log10(np.asarray(ys, dtype=float))
    out = ly.copy()
    for a, b in zip(order[:-1], order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    # recentre the pushed block on the original mean so labels stay near their points
    out += np.mean(ly - out)
    return list(10**out)


def s288c_centered(d: dict) -> pd.DataFrame:
    """Every S288C pair and the human yardstick pairs Fig 6 plots; writes
    ``results/s288c_centered.csv``.
    """
    pw = d["pw"]
    rows = []
    for ref, partners in (("scer_s288c", None), ("hsap_grch38", HUMAN_YARDSTICK)):
        m = (pw.org_i == ref) | (pw.org_j == ref)
        for _, r in pw[m].iterrows():
            other = r.org_j if r.org_i == ref else r.org_i
            if partners is not None and other not in partners:
                continue
            rows.append(
                {
                    "reference": ref,
                    "organism": other,
                    "display": d["display"][other],
                    "group": d["taxon"][other],
                    "column": FIG6_COLUMN[d["taxon"][other]]
                    if ref == "scer_s288c"
                    else "human_vs",
                    "aai_mean": float(r.aai_mean),
                    "aai_median": float(r.aai_median),
                    "ortholog_fraction": float(r.ortholog_fraction),
                    "n_rbh": int(r.n_rbh),
                }
            )
    out = pd.DataFrame(rows).sort_values(
        ["reference", "aai_mean"], ascending=[False, False]
    )
    out.to_csv(osp.join(RES, "s288c_centered.csv"), index=False)
    return out


def fig6(d: dict) -> tuple[str, str, dict]:
    """The main transfer figure: everything measured from S288C, the human yardstick, and
    the ortholog fraction each organism shares with S288C.
    """
    sc = s288c_centered(d)
    s = sc[sc.reference == "scer_s288c"]
    h = sc[sc.reference == "hsap_grch38"].set_index("organism").loc[HUMAN_YARDSTICK]
    iso = d["iso"]
    cols = [c for c, _ in FIG6_COLUMNS]
    xs = {c: k for k, c in enumerate(cols)}
    W = PANEL_WIDTHS_MM["full"]
    fig = plt.figure(figsize=(mm_to_in(W), mm_to_in(88)))
    # explicit placement: (a) about twice (b), with room between them for (a)'s identity
    # axis and (b)'s organism names
    ax = fig.add_axes([0.065, 0.16, 0.54, 0.70])
    ax_b = fig.add_axes([0.80, 0.16, 0.185, 0.70])
    rng = np.random.default_rng(0)
    # (a) isolates: two sub-strips, median over ORFs and mean over ORFs
    c_iso = TAXON_COLOR["isolates"]
    for stat, dx, marker in (
        ("median_identity", -0.2, "o"),
        ("mean_identity", 0.2, "^"),
    ):
        y = 100.0 * (1.0 - iso[stat].to_numpy())
        ax.scatter(
            xs["isolates"] + dx + rng.uniform(-0.13, 0.13, len(y)),
            y,
            s=2.5,
            marker=marker,
            lw=0,
            color=c_iso,
            alpha=0.8,
            zorder=2,
        )
    labels: dict[str, list] = {}

    def plot_column(sub: pd.DataFrame, col: str, x0: float) -> None:
        sub = sub.sort_values("aai_mean", ascending=False)
        n = len(sub)
        offs = np.linspace(-0.25, 0.25, n) if n > 1 else np.zeros(1)
        for (_, r), dx in zip(sub.iterrows(), offs):
            hi = r.organism == "pkud_sd108"
            color = TAXON_COLOR[r.group]
            ax.scatter(
                x0 + dx,
                100.0 - r.aai_mean,
                s=26 if hi else 9,
                facecolor=color,
                edgecolor="black",
                lw=0.8 if hi else 0.3,
                zorder=4 if hi else 3,
            )
            if r.organism in FIG6_LABELS and (
                col == "human_vs" or FIG6_LABELS[r.organism] in FIG6_LABEL_XY
            ):
                labels.setdefault(col, []).append(
                    (
                        x0 + dx,
                        100.0 - r.aai_mean,
                        FIG6_LABELS[r.organism],
                        r.aai_mean,
                        hi,
                    )
                )

    for col in cols[1:-1]:
        plot_column(s[s.column == col], col, xs[col])
    plot_column(h.reset_index(), "human_vs", xs["human_vs"])
    # S288C-side labels at hand-placed positions in the empty space above the points;
    # the human column's labels sit to its right, pushed apart on the log axis
    for col, items in labels.items():
        if col == "human_vs":
            items = sorted(items, key=lambda t: t[1])
            ly = spread_log([t[1] for t in items], 0.17)
            place = [(xs[col] + 0.36, yl, "left") for yl in ly]
        else:
            place = [
                (
                    xs[col] + FIG6_LABEL_XY[name][0],
                    FIG6_LABEL_XY[name][1],
                    FIG6_LABEL_XY[name][2],
                )
                for _, _, name, _, _ in items
            ]
        for (xv, yv, name, aai, hi), (tx, ty, ha) in zip(items, place):
            ax.annotate(
                f"{name} {aai:.1f}%",
                xy=(xv, yv),
                xytext=(tx, ty),
                textcoords="data",
                fontsize=FS,
                fontweight="bold" if hi else "normal",
                ha=ha,
                va="center",
                arrowprops={"arrowstyle": "-", "lw": 0.3, "color": "black"},
            )
    ax.axvline(xs["human_vs"] - 0.62, color="black", lw=0.5, ls="--")
    lo, hi_band = d["lit"]["protein_identity_vs_reference"]
    ax.axhspan(
        100 * (1 - hi_band),
        100 * (1 - lo),
        color=PLOT_PALETTE[5],
        alpha=0.25,
        lw=0,
        zorder=1,
    )
    ax.set_yscale("log")
    ax.set_ylim(0.03, 100)
    ax.set_yticks([0.1, 1, 10, 100])
    ax.set_yticklabels(["0.1", "1", "10", "100"])
    ax.set_ylabel("Protein divergence, 100 - identity (%)")
    ax.invert_yaxis()
    sec = ax.secondary_yaxis("right", functions=(lambda v: v, lambda v: v))
    sec.set_yticks([0.1, 1, 10, 50])
    sec.set_yticklabels(["99.9", "99", "90", "50"])
    sec.set_ylabel("Identity (%)", labelpad=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([lab.format(n=len(iso)) for _, lab in FIG6_COLUMNS])
    ax.set_xlim(-0.55, len(cols) + 0.75)
    ax.grid(axis="y", which="major", lw=0.3, color="#CCCCCC")
    ax.text(
        xs["saccharomyces"] - 0.5,
        100 * (1 - hi_band) * 0.85,
        "one human genome vs GRCh38 (literature)",
        fontsize=FS,
        va="bottom",
        ha="left",
    )
    ax.text(
        xs["human_vs"] - 0.55,
        0.035,
        "yardstick:\nfrom human",
        fontsize=FS,
        va="top",
        ha="left",
    )
    ax.text(xs["isolates"] - 0.5, 0.035, "from S288C", fontsize=FS, va="top", ha="left")
    # (b) ortholog fraction with S288C, same order top to bottom
    order = []
    for col in cols[1:-1]:
        order += list(
            s[s.column == col].sort_values("aai_mean", ascending=False).itertuples()
        )
    yb = np.arange(len(order))
    for k, r in enumerate(order):
        hi = r.organism == "pkud_sd108"
        ax_b.barh(
            k,
            r.ortholog_fraction,
            height=0.7,
            color=TAXON_COLOR[r.group],
            edgecolor="black",
            lw=0.8 if hi else 0.3,
            zorder=3,
        )
        if hi:
            ax_b.text(
                r.ortholog_fraction + 0.02,
                k,
                f"{r.ortholog_fraction:.3f}",
                va="center",
                ha="left",
                fontsize=FS,
                fontweight="bold",
            )
    ax_b.set_yticks(yb)
    ax_b.set_yticklabels([r.display for r in order])
    for lab, r in zip(ax_b.get_yticklabels(), order):
        if r.organism == "pkud_sd108":
            lab.set_fontweight("bold")
    ax_b.tick_params(axis="y", length=0, pad=1)
    ax_b.set_ylim(len(order) - 0.5, -0.5)
    ax_b.set_xlim(0, 1)
    from matplotlib.ticker import MultipleLocator

    ax_b.xaxis.set_major_locator(MultipleLocator(0.2))
    ax_b.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax_b.tick_params(axis="x", which="minor", length=0)
    ax_b.grid(axis="x", which="both", lw=0.3, color="#DDDDDD")
    ax_b.set_axisbelow(True)
    ax_b.set_xlabel(
        "Ortholog fraction with S288C\n(reciprocal best hits / smaller proteome)"
    )
    for a_ in (ax, ax_b):
        for sp in a_.spines.values():
            sp.set_linewidth(0.5)
    for a_, letter, dx in ((ax, "a", -0.08), (ax_b, "b", -1.0)):
        a_.text(
            dx,
            1.02,
            letter,
            transform=a_.transAxes,
            fontsize=8,
            fontweight="bold",
            va="bottom",
            ha="left",
        )
    handles = (
        [
            Line2D(
                [],
                [],
                marker="o",
                ls="",
                markersize=2.5,
                markerfacecolor=c_iso,
                markeredgecolor=c_iso,
                label="isolate, median over ORFs",
            ),
            Line2D(
                [],
                [],
                marker="^",
                ls="",
                markersize=2.5,
                markerfacecolor=c_iso,
                markeredgecolor=c_iso,
                label="isolate, mean over ORFs, frameshifts charged",
            ),
        ]
        + [
            Line2D(
                [],
                [],
                marker="o",
                ls="",
                markersize=3,
                markerfacecolor=TAXON_COLOR[t],
                markeredgecolor="black",
                markeredgewidth=0.3,
                label=TAXON_LABEL[t],
            )
            for t in (
                "saccharomyces",
                "saccharomycotina",
                "other_fungus",
                "mammal",
                "bird",
                "bacteria",
            )
        ]
        + [
            Patch(
                facecolor=PLOT_PALETTE[5],
                alpha=0.25,
                label="within-human literature band",
            )
        ]
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=5,
        frameon=False,
        fontsize=FS,
        handletextpad=0.3,
        columnspacing=1.0,
    )
    path = save(fig, "fig6_main_transfer_ladder", pdf=True)
    sd = s.set_index("organism").loc["pkud_sd108"]
    vert = {"mammal", "bird"}
    pw = d["pw"]
    vmask = pw.org_i.map(d["taxon"]).isin(vert) & pw.org_j.map(d["taxon"]).isin(vert)
    ymaj = s[s.column == "saccharomycotina"]
    stats = {
        "sd108_aai_mean": float(sd.aai_mean),
        "sd108_aai_median": float(sd.aai_median),
        "sd108_ortholog_fraction": float(sd.ortholog_fraction),
        "sd108_n_rbh": int(sd.n_rbh),
        "sd108_n_genes": int(
            pw[(pw.org_i == "scer_s288c") & (pw.org_j == "pkud_sd108")].n_j.iloc[0]
        ),
        "cbs573_aai_mean": float(
            s.set_index("organism").loc["pkud_cbs573", "aai_mean"]
        ),
        "min_vertebrate_aai": float(pw[vmask].aai_mean.min()),
        "n_vertebrate_pairs": int(vmask.sum()),
        "human_chicken_aai": float(h.loc["ggal", "aai_mean"]),
        "human_chimp_aai": float(h.loc["ptro", "aai_mean"]),
        "isolate_median_of_median_identity": float(
            100 * iso["median_identity"].median()
        ),
        "isolate_median_of_mean_identity": float(100 * iso["mean_identity"].median()),
        "isolate_min_mean_identity": float(100 * iso["mean_identity"].min()),
        "n_isolates": int(len(iso)),
        "n_s288c_partners": int(len(s)),
        "n_per_column": {c: int((s.column == c).sum()) for c in cols[1:-1]},
        "n_human_yardstick": int(len(h)),
        "saccharomycotina_aai_min": float(ymaj.aai_mean.min()),
        "saccharomycotina_aai_max": float(ymaj.aai_mean.max()),
        "saccharomycotina_of_min": float(ymaj.ortholog_fraction.min()),
        "saccharomycotina_of_max": float(ymaj.ortholog_fraction.max()),
        "animal_from_s288c_aai_min": float(s[s.column == "animal"].aai_mean.min()),
        "animal_from_s288c_aai_max": float(s[s.column == "animal"].aai_mean.max()),
        "bacteria_from_s288c_aai_min": float(s[s.column == "bacteria"].aai_mean.min()),
        "bacteria_from_s288c_aai_max": float(s[s.column == "bacteria"].aai_mean.max()),
        "bacteria_from_s288c_of_max": float(
            s[s.column == "bacteria"].ortholog_fraction.max()
        ),
        "spar_aai": float(s.set_index("organism").loc["spar_cbs432", "aai_mean"]),
        "seub_aai": float(s.set_index("organism").loc["seub_fm1318", "aai_mean"]),
    }
    st = stats
    caption = (
        "Fig. 6 | The training isolates reach human/chimp protein divergence from S288C; "
        "P. kudriavzevii SD108 lies beyond every vertebrate pair, with "
        f"{100 * st['sd108_ortholog_fraction']:.1f}% of its genes in one-to-one orthology "
        "with S. cerevisiae. (a) Protein divergence from S288C (100 minus identity, log "
        "scale, inverted; identity on the right), columns left to right by distance, "
        "colored by the partner's taxonomic group. The "
        f"{st['n_isolates']:,} isolates are drawn twice: each isolate's median over ORFs "
        f"of protein identity to S288C (circles; median over isolates "
        f"{st['isolate_median_of_median_identity']:.2f}%) and its mean over ORFs, which "
        "charges every gap column of a frameshifted or indel-bearing ORF (triangles; "
        f"median {st['isolate_median_of_mean_identity']:.2f}%, lowest "
        f"{st['isolate_min_mean_identity']:.2f}%, against human/chimp AAI "
        f"{st['human_chimp_aai']:.2f}%). SD108 (large point, black edge) is at "
        f"{st['sd108_aai_mean']:.1f}% AAI from S288C, below the lowest of the "
        f"{st['n_vertebrate_pairs']} vertebrate pairs ({st['min_vertebrate_aai']:.1f}%; "
        f"human/chicken {st['human_chicken_aai']:.1f}%); the other Saccharomycotina "
        f"hosts sit at {st['saccharomycotina_aai_min']:.1f} to "
        f"{st['saccharomycotina_aai_max']:.1f}%, the mammals and chicken at "
        f"{st['animal_from_s288c_aai_min']:.1f} to {st['animal_from_s288c_aai_max']:.1f}% "
        f"and the bacteria at {st['bacteria_from_s288c_aai_min']:.1f} to "
        f"{st['bacteria_from_s288c_aai_max']:.1f}%. Right of the dashed line, the same "
        "axis measured from human, as a yardstick (mammals blue, chicken gray). Gray "
        "band: one human genome vs GRCh38 at peptide-altering sites (1000 Genomes 2015, "
        "literature value). (b) Ortholog fraction with S288C, reciprocal best hits over "
        "the smaller per-gene proteome, same organisms and order as (a) without the "
        f"isolates. SD108 shares {st['sd108_n_rbh']:,} reciprocal best hits with S288C, "
        f"{100 * st['sd108_ortholog_fraction']:.1f}% of its {st['sd108_n_genes']:,} genes; "
        "the other Saccharomycotina range from "
        f"{st['saccharomycotina_of_min']:.3f} to {st['saccharomycotina_of_max']:.3f}, "
        f"no bacterium exceeds {st['bacteria_from_s288c_of_max']:.3f}. AAI: mean percent "
        "identity over reciprocal best mmseqs2 hits (03_mmseqs_rbh.py); isolate identity: "
        "06_isolate_protein_identity.py; table: results/s288c_centered.csv."
    )
    return path, caption, stats


def headline(d: dict) -> dict:
    """Every number the experiment note cites, in one file."""
    pw = pair_table(d)
    iso = d["iso"]
    sc = d["sc"].set_index(["org_i", "org_j"])

    def pair(i: str, j: str) -> pd.Series:
        m = ((pw.org_i == i) & (pw.org_j == j)) | ((pw.org_i == j) & (pw.org_j == i))
        return pw[m].iloc[0]

    out: dict = {
        "isolates_vs_s288c": {
            "n_isolates": int(len(iso)),
            "n_orfs_per_isolate": int(iso["n_orfs"].median()),
            "median_of_mean_identity": float(iso["mean_identity"].median()),
            "min_mean_identity": float(iso["mean_identity"].min()),
            "max_mean_identity": float(iso["mean_identity"].max()),
            "median_of_median_identity": float(iso["median_identity"].median()),
            "median_of_pooled_identity": float(iso["pooled_identity"].median()),
            "median_of_mean_identity_equal_length": float(
                iso["mean_identity_equal_length"].median()
            ),
        },
        "categories": {},
        "pairs": {},
        "literature_protein_identity_band": d["lit"]["protein_identity_vs_reference"],
    }
    for cat, _ in CATEGORIES[1:]:
        sub = pw[pw.category == cat]
        out["categories"][cat] = {
            "n_pairs": int(len(sub)),
            "aai_median": float(sub["aai_mean"].median()),
            "aai_min": float(sub["aai_mean"].min()),
            "aai_max": float(sub["aai_mean"].max()),
            "ortholog_fraction_median": float(sub["ortholog_fraction"].median()),
            "ec4_jaccard_median": float(sub["ec4_jaccard"].median()),
        }
    for i, j, label in LABELED:
        r = pair(i, j)
        key = (r.org_i, r.org_j)
        out["pairs"][label] = {
            "aai_mean": float(r.aai_mean),
            "aai_median": float(r.aai_median),
            "n_rbh": int(r.n_rbh),
            "ortholog_fraction": float(r.ortholog_fraction),
            "ec4_jaccard": float(r.ec4_jaccard),
            "aai_single_copy_og": float(sc.loc[key, "aai_sc_mean"]),
            "n_single_copy_og": int(sc.loc[key, "n_with_hit"]),
        }
    r = pair("scer_s288c", "ecoli_mg1655")
    out["pairs"]["S288C/E. coli"] = {
        "aai_mean": float(r.aai_mean),
        "n_rbh": int(r.n_rbh),
        "ortholog_fraction": float(r.ortholog_fraction),
    }
    both = d["sc"].merge(d["pw"], on=["org_i", "org_j"])
    both = both[both["n_with_hit"] >= 50]
    out["single_copy_vs_rbh_aai"] = {
        "n_pairs": int(len(both)),
        "pearson_r": float(np.corrcoef(both["aai_sc_mean"], both["aai_mean"])[0, 1]),
        "median_difference_sc_minus_rbh": float(
            (both["aai_sc_mean"] - both["aai_mean"]).median()
        ),
    }
    return out


def main() -> None:
    style()
    d = load()
    unknown = {k for k, v in d["taxon"].items() if v == "protist"}
    assert unknown <= {"alim"}, unknown
    caps: dict[str, str] = {}
    paths: dict[str, str] = {}
    p, c, s1 = fig1(d)
    paths["fig1"], caps["fig1_claim_protein_divergence"] = p, c
    p, c = fig2(d)
    paths["fig2"], caps["fig2_cladogram_heatmaps"] = p, c
    p, c = fig3(d)
    paths["fig3"], caps["fig3_taxonomy_vs_aai_nj"] = p, c
    p, c, s4 = fig4(d)
    paths["fig4"], caps["fig4_clade_tree_within_distance"] = p, c
    p, c, s5 = fig5(d)
    paths["fig5"], caps["fig5_ortholog_fraction_vs_aai"] = p, c
    p, c, s6 = fig6(d)
    paths["fig6"], caps["fig6_main_transfer_ladder"] = p, c
    with open(osp.join(RES, "figure_captions.json"), "w") as fh:
        json.dump(caps, fh, indent=1)
    with open(osp.join(RES, "headline_numbers.json"), "w") as fh:
        json.dump(headline(d), fh, indent=1)
    with open(osp.join(RES, "figure_stats.json"), "w") as fh:
        json.dump(
            {"fig1": s1, "fig4": s4, "fig5": s5, "fig6": s6, "paths": paths},
            fh,
            indent=1,
        )
    print(json.dumps({"paths": paths, "fig4": s4, "fig5": s5, "fig6": s6}, indent=1))


if __name__ == "__main__":
    main()
