# experiments/044-organism-similarity/scripts/07_figures.py
# [[experiments.044-organism-similarity.scripts.07_figures]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/044-organism-similarity/scripts/07_figures.py

"""Figures 1-5 of experiment 044 (organism similarity) and their captions.

Reads only committed result tables (``results/``) and the tier-free trees; writes PNG +
true-size SVG to ``ASSET_IMAGES_DIR/044-organism-similarity/`` (timestamped until
approved) and ``results/figure_captions.json``.

- Fig 1: the claim strip plot, protein divergence (100 - identity) of each pair, log
  axis, by pair category, with the within-human literature band.
- Fig 2: NCBI taxonomy cladogram + AAI, ortholog Jaccard and EC-4 Jaccard heatmaps.
- Fig 3: taxonomy cladogram vs the AAI neighbor-joining tree (tanglegram).
- Fig 4: the 1,011-isolate tree colored by Peter's clade.
- Fig 5: ortholog fraction vs AAI per pair.

Color is the organism group (plan rule). Fig 2's strip uses the plan's groups
(reference, established, emerging, CABBI, genus, mammal, outgroup); Figs 1, 3 and 5 color
by the taxonomic group the claim is about (S. cerevisiae isolates, Saccharomyces,
other Saccharomycotina, non-Saccharomycotina fungus, mammals, bird, bacteria); a
cross-group pair is drawn with the face of one member's color and the edge of the other.

Run from the repo root: ``PYTHONPATH=. ~/miniconda3/envs/torchcell/bin/python ...``.
"""

from __future__ import annotations

import json
import os
import os.path as osp

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from Bio import Phylo  # noqa: E402
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

load_dotenv()
EXP_DIR = osp.join(os.environ["EXPERIMENT_ROOT"], "044-organism-similarity")
RES = osp.join(EXP_DIR, "results")
OUT_DIR = osp.join(os.environ["ASSET_IMAGES_DIR"], "044-organism-similarity")
STAMP = timestamp()
FS = 6

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
    "human/chimp": (101.0, 0.70, "right"),
    "human/macaque": (91.0, 0.76, "center"),
    "human/mouse": (84.0, 0.64, "center"),
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


def save(fig: plt.Figure, name: str) -> str:
    """PNG + true-size SVG; returns the SVG path."""
    os.makedirs(OUT_DIR, exist_ok=True)
    base = osp.join(OUT_DIR, f"{name}_{STAMP}")
    fig.savefig(base + ".png", dpi=300)
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


def pair_category(i: str, j: str, lin: pd.DataFrame) -> str | None:
    """Fig 1 category of an organism pair (None when it is in none)."""
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
    if (a.is_saccharomycotina and b.is_bacteria) or (
        b.is_saccharomycotina and a.is_bacteria
    ):
        return "yeast_bacteria"
    return None


def pair_table(d: dict) -> pd.DataFrame:
    """Pairwise table with category and taxon colors."""
    pw = d["pw"].copy()
    pw["category"] = [pair_category(i, j, d["lin"]) for i, j in zip(pw.org_i, pw.org_j)]
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


def fig4(d: dict) -> tuple[str, str, dict]:
    """The 1,011-isolate tree colored by clade."""
    tree = Phylo.read(osp.join(RES, "isolates_nj.nwk"), "newick")
    tree.root_at_midpoint()
    tree.ladderize()
    clades = pd.read_csv(osp.join(RES, "isolate_clades.csv")).set_index("isolate")
    counts = clades["clade"].value_counts()
    top = list(counts.index[:17])
    color = {c: PLOT_PALETTE[k] for k, c in enumerate(top)}
    other = PLOT_PALETTE[17]
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
    fig = plt.figure(figsize=(mm_to_in(PANEL_WIDTHS_MM["half"]), mm_to_in(110)))
    ax = fig.add_axes([0.02, 0.24, 0.80, 0.74])
    depths = tree.depths()
    ys = {t: k for k, t in enumerate(tree.get_terminals())}

    def y_of(c):
        if c in ys:
            return ys[c]
        ys[c] = float(np.mean([y_of(ch) for ch in c.clades]))
        return ys[c]

    y_of(tree.root)
    segs_v, segs_h = [], []
    for c in tree.find_clades():
        if c.is_terminal():
            continue
        yc = [ys[ch] for ch in c.clades]
        segs_v.append([(depths[c], min(yc)), (depths[c], max(yc))])
        for ch in c.clades:
            segs_h.append([(depths[c], ys[ch]), (depths[ch], ys[ch])])
    from matplotlib.collections import LineCollection

    ax.add_collection(LineCollection(segs_v + segs_h, colors="black", linewidths=0.2))
    xmax = max(depths.values())
    strip_x = xmax * 1.03
    w = xmax * 0.06
    nterm = len(ys) if False else len(tree.get_terminals())
    for t in tree.get_terminals():
        code = t.name
        cl = clades.loc[code, "clade"] if code in clades.index else np.nan
        col = color.get(cl, other) if isinstance(cl, str) else "white"
        ax.add_patch(plt.Rectangle((strip_x, ys[t] - 0.5), w, 1, color=col, lw=0))
    near_code = [
        t
        for t in tree.get_terminals()
        if t.name.removeprefix("SACE_") == nearest.strain
    ]
    for t in near_code:
        ax.scatter(strip_x + w * 1.6, ys[t], marker="<", s=12, color="black", zorder=4)
        ax.text(
            strip_x + w * 2.2,
            ys[t],
            f"{t.name}: nearest\nto S288C",
            fontsize=FS,
            va="center",
        )
    ax.set_xlim(0, strip_x + w * 2)
    ax.set_ylim(nterm + 0.5, -0.5)
    ax.set_yticks([])
    ax.set_xlabel("SNP distance (% non-identical bases, Peter 2018 matrix)")
    for s in ax.spines.values():
        s.set_linewidth(0.5)
    handles = [Patch(color=color[c], label=c) for c in top] + [
        Patch(color=other, label=f"other clades ({len(counts) - len(top)})"),
        Patch(
            facecolor="white", edgecolor="black", lw=0.3, label="no clade in Table S1"
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower left",
        ncol=3,
        frameon=False,
        fontsize=5,
        bbox_to_anchor=(0.0, 0.0),
        handlelength=0.8,
        columnspacing=0.5,
        handletextpad=0.3,
    )
    path = save(fig, "fig4_isolate_tree")
    n_clade = int(clades["clade"].notna().sum())
    caption = (
        "Fig. 4 | The 1,011 S. cerevisiae isolates. rapidnj neighbor-joining tree on "
        "Peter et al. 2018's SNP distance matrix (percent non-identical bases, "
        "heterozygous differences half-weighted), midpoint rooted (05_trees.py). Strip: "
        f"clade from Peter's Supplementary Table 1 ({n_clade} of 1,011 isolates carry "
        f"one; the {len(top)} largest clades in color, the remaining "
        f"{len(counts) - len(top)} in one color; Wine/European subclades merged). "
        f"Arrow: {nearest.strain}, the isolate with the lowest mean per-ORF nucleotide "
        f"divergence from S288C in 018 ({100 * nearest.mean_divergence:.4f}%); S288C "
        "itself is not in Peter's matrix."
    )
    return (
        path,
        caption,
        {
            "nearest_to_s288c": str(nearest.strain),
            "nearest_mean_divergence": float(nearest.mean_divergence),
        },
    )


def fig5(d: dict) -> tuple[str, str]:
    """Ortholog fraction vs AAI."""
    pw = pair_table(d)
    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_WIDTHS_MM["half"]), mm_to_in(70)))
    for _, r in pw.iterrows():
        ax.scatter(
            r.aai_mean,
            r.ortholog_fraction,
            s=7,
            facecolor=TAXON_COLOR[r.taxon_i],
            edgecolor=TAXON_COLOR[r.taxon_j],
            lw=0.6,
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
    ax.set_ylabel("Ortholog fraction, n RBH / min(n_i, n_j)")
    ax.set_ylim(0, 1.16)
    from matplotlib.ticker import MultipleLocator

    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(axis="y", which="minor", length=0)
    ax.grid(axis="y", which="both", lw=0.3, color="#DDDDDD")
    present = sorted({t for t in pw.taxon_i} | {t for t in pw.taxon_j}, key=TAXA.index)
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
        for t in present
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=FS)
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.15, top=0.97)
    path = save(fig, "fig5_ortholog_fraction_vs_aai")
    caption = (
        f"Fig. 5 | Shared orthologs fall with AAI. Each point is one of the {len(pw)} "
        "pairs of the 33 proteomes: AAI (mean RBH percent identity) against the ortholog "
        "fraction, reciprocal best hits over the smaller per-gene proteome "
        "(03_mmseqs_rbh.py). Face and edge colors are the taxonomic groups of the two "
        "members."
    )
    return path, caption


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
    paths["fig4"], caps["fig4_isolate_tree"] = p, c
    p, c = fig5(d)
    paths["fig5"], caps["fig5_ortholog_fraction_vs_aai"] = p, c
    with open(osp.join(RES, "figure_captions.json"), "w") as fh:
        json.dump(caps, fh, indent=1)
    with open(osp.join(RES, "headline_numbers.json"), "w") as fh:
        json.dump(headline(d), fh, indent=1)
    with open(osp.join(RES, "figure_stats.json"), "w") as fh:
        json.dump({"fig1": s1, "fig4": s4, "paths": paths}, fh, indent=1)
    print(json.dumps({"paths": paths, "fig1": s1, "fig4": s4}, indent=1))


if __name__ == "__main__":
    main()
