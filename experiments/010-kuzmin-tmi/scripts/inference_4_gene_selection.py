# experiments/010-kuzmin-tmi/scripts/inference_4_gene_selection.py
# [[experiments.010-kuzmin-tmi.scripts.inference_4_gene_selection]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/inference_4_gene_selection
#
# The inference_4 gene roster: metabolism x regulation, grounded in measured screens.
#
# WHAT CHANGES FROM inference_1. That roster was scored on overlap across four source
# lists and covered 324 of the 1,161 yeast-GEM genes, 28 percent, with no constraint on
# how a triple mixed its sources. Two things go differently here.
#
#   1. The axes are named. A triple must carry 1 to 2 metabolic genes AND 1 to 2
#      regulators, so every prediction is about a regulator acting on metabolism rather
#      than about an unstructured triple. yeast-GEM 9.0.2 supplies metabolism; TFLink
#      plus the SGD regulatory graph supply regulators. The two sets overlap in 12
#      genes, so the stratification is close to a genuine partition.
#
#   2. Support is a screen count, not presence. The intent is that each triple contain
#      a gene the trigenic data actually constrains. "Appears in the Kuzmin data" does
#      not do that: an array gene appears under hundreds of query screens while a query
#      gene appears under one, and both satisfy presence. The one-screen genes are what
#      produced inference_1's positive tail, and removing them shrank the best predicted
#      effect by 9.6x. So the gate is DISTINCT QUERY SCREENS, and presence is reported
#      beside it to show how much weaker it is.
#
# FILTERS, reused verbatim from expand_gene_selection_inference_1.py so the two rosters
# are comparable: drop SGD-essential genes, drop any gene whose single-mutant fitness is
# below 0.9 in any of Kuzmin2018 / Kuzmin2020 / Costanzo2016, drop Q-prefix mitochondrial
# genes. Sick singles are dropped because a triple built on one cannot demonstrate a
# fitness gain, and essential genes cannot be deleted at all.
#
# Run from repo root:
#   ~/miniconda3/envs/torchcell/bin/python \
#     experiments/010-kuzmin-tmi/scripts/inference_4_gene_selection.py
#
# Outputs:
#   results/inference_4/gene_candidates.csv   every candidate with its annotations
#   results/inference_4/gene_list.txt         the surviving roster, one ORF per line
#   results/inference_4/sizing.json           exact triple counts under each constraint
#   $ASSET_IMAGES_DIR/010-kuzmin-tmi/inference_4_gene_selection.{png,svg}

import json
import math
import os
import os.path as osp
from collections import Counter
from itertools import combinations_with_replacement

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from torchcell.datasets.scerevisiae import SmfCostanzo2016Dataset
from torchcell.datasets.scerevisiae.kuzmin2018 import (
    SmfKuzmin2018Dataset,
    TmfKuzmin2018Dataset,
)
from torchcell.datasets.scerevisiae.kuzmin2020 import (
    SmfKuzmin2020Dataset,
    TmfKuzmin2020Dataset,
)
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]

RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi", "results", "inference_4")
IMAGES_DIR = osp.join(ASSET_IMAGES_DIR, "010-kuzmin-tmi")

MIN_SMF = 0.9          # the 006 fitness floor, applied across all three SMF sources
MIN_SCREENS = 50       # the support gate established by screen_diversity_audit.py
HO = "YDL227C"


def root(name: str) -> str:
    return osp.join(DATA_ROOT, "data/torchcell", name)


def set_plot_style():
    plt.rcParams.update(
        {
            "font.family": "Arial", "font.size": 6, "axes.labelsize": 6,
            "axes.titlesize": 6, "xtick.labelsize": 6, "ytick.labelsize": 6,
            "legend.fontsize": 5, "legend.title_fontsize": 5, "figure.titlesize": 6,
            "svg.fonttype": "none", "axes.linewidth": 0.5,
            "savefig.bbox": None, "savefig.pad_inches": 0.0,
        }
    )


def metabolic_genes() -> set[str]:
    from torchcell.metabolism.yeast_GEM import YeastGEM

    gem = YeastGEM(root=root("yeast-GEM"))
    return {str(g) for g in gem.gene_set}


def build_graph(genome):
    from torchcell.graph import SCerevisiaeGraph

    return SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )


def regulator_genes(graph) -> tuple[set[str], set[str]]:
    """TFLink regulators and SGD regulatory-graph regulators, each as source nodes."""
    tflink = {str(u) for u, _ in graph.G_tflink.graph.edges()}
    sgd_reg = {str(u) for u, _ in graph.G_regulatory.graph.edges()}
    return tflink, sgd_reg


def essential_genes(graph) -> set[str]:
    """SGD inviable-null genes, the same call expand_gene_selection_inference_1.py makes."""
    from torchcell.datasets.scerevisiae.sgd import GeneEssentialitySgdDataset

    ds = GeneEssentialitySgdDataset(
        root=root("gene_essentiality_sgd"), scerevisiae_graph=graph
    )
    out = set()
    for i in range(len(ds)):
        exp = ds[i]["experiment"]
        if exp["phenotype"]["is_essential"]:
            for p in exp["genotype"]["perturbations"]:
                out.add(p["systematic_gene_name"])
    return out


def kuzmin_support() -> pd.DataFrame:
    """Per gene: distinct query screens, and whether it appears as query or as array.

    A Kuzmin screen is one query double crossed against an array, so grouping a gene's
    trigenic records by their query double counts the independent contexts it was
    observed in. A query gene has one such context however many arrays it met.
    """
    rows = []
    for cls, rt in (
        (TmfKuzmin2018Dataset, "tmf_kuzmin2018"),
        (TmfKuzmin2020Dataset, "tmf_kuzmin2020"),
    ):
        df = cls(root=root(rt)).df
        df = df[df["Combined mutant type"] == "trigenic"]
        q1 = df["Query systematic name_1"].astype(str).to_numpy()
        q2 = df["Query systematic name_2"].astype(str).to_numpy()
        x = df["Array systematic name"].astype(str).to_numpy()
        lo = np.where(q1 < q2, q1, q2)
        hi = np.where(q1 < q2, q2, q1)
        screen = np.char.add(np.char.add(lo, "+"), hi)
        rows.append(pd.DataFrame({"gene": q1, "screen": screen, "role": "query"}))
        rows.append(pd.DataFrame({"gene": q2, "screen": screen, "role": "query"}))
        rows.append(pd.DataFrame({"gene": x, "screen": screen, "role": "array"}))
    long = pd.concat(rows, ignore_index=True)
    agg = long.groupby("gene").agg(
        distinct_screens=("screen", "nunique"),
        n_records=("screen", "size"),
    )
    agg["as_query"] = long[long["role"] == "query"].groupby("gene").size().reindex(agg.index).fillna(0).astype(int)
    agg["as_array"] = long[long["role"] == "array"].groupby("gene").size().reindex(agg.index).fillna(0).astype(int)
    return agg.reset_index()


def smf_table(genes: set[str]) -> pd.DataFrame:
    """Minimum single-mutant fitness across the three sources, the 006 filter's input."""
    frames = []
    ks = SmfKuzmin2020Dataset(root=root("smf_kuzmin2020")).df
    gene = np.where(ks["ORF1"] == HO, ks["ORF2"], ks["ORF1"])
    ks = ks.assign(gene=gene)
    ks = ks[ks["gene"].isin(genes) & (ks["Mutant type"] == "Single mutant")]
    frames.append(pd.DataFrame({"gene": ks["gene"], "smf": ks["Fitness"], "src": "K2020"}))

    k18 = SmfKuzmin2018Dataset(root=root("smf_kuzmin2018")).df
    col = "Query systematic name no ho"
    sub = k18[k18[col].isin(genes)]
    frames.append(pd.DataFrame(
        {"gene": sub[col], "smf": sub["Combined mutant fitness"], "src": "K2018"}))

    cs = SmfCostanzo2016Dataset(root=root("smf_costanzo2016")).df
    cs = cs[cs["Systematic gene name"].isin(genes)
            & (cs["Temperature"] == 30)
            & cs["perturbation_type"].str.contains("deletion")]
    frames.append(pd.DataFrame(
        {"gene": cs["Systematic gene name"], "smf": cs["Single mutant fitness"], "src": "C2016"}))

    long = pd.concat(frames, ignore_index=True).dropna(subset=["smf"])
    return long.groupby("gene")["smf"].agg(min_smf="min", mean_smf="mean",
                                           n_smf_sources="size").reset_index()


def count_triples(classes: dict[str, int], keep) -> int:
    """Exact number of 3-subsets whose class multiset satisfies `keep`.

    Enumerating C(n,3) directly is 4.6e8 rows at this roster size, and every constraint
    here is a function of class membership alone, so the count is closed-form over the
    multisets of classes instead.
    """
    names = list(classes)
    total = 0
    for combo in combinations_with_replacement(names, 3):
        if not keep(combo):
            continue
        counts = Counter(combo)
        ways = 1
        for cname, k in counts.items():
            ways *= math.comb(classes[cname], k)
        total += ways
    return total


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
    )
    genome_genes = {str(g) for g in genome.gene_set}
    print(f"genome gene set: {len(genome_genes):,}")

    graph = build_graph(genome)
    met = metabolic_genes() & genome_genes
    tflink, sgd_reg = regulator_genes(graph)
    tflink &= genome_genes
    sgd_reg &= genome_genes
    reg = tflink | sgd_reg
    print(f"yeast-GEM genes in genome:        {len(met):,}")
    print(f"TFLink regulators:                {len(tflink):,}")
    print(f"SGD regulatory-graph regulators:  {len(sgd_reg):,}")
    print(f"regulators, union:                {len(reg):,}")
    print(f"metabolic AND regulator:          {len(met & reg):,}")

    candidates = sorted(met | reg)
    print(f"candidate union:                  {len(candidates):,}")

    support = kuzmin_support()
    smf = smf_table(set(candidates))
    ess = essential_genes(graph)
    print(f"SGD essential genes:              {len(ess):,}")

    df = pd.DataFrame({"gene": candidates})
    df["is_metabolic"] = df["gene"].isin(met)
    df["is_tflink_regulator"] = df["gene"].isin(tflink)
    df["is_sgd_regulator"] = df["gene"].isin(sgd_reg)
    df["is_regulator"] = df["gene"].isin(reg)
    df = df.merge(support, on="gene", how="left")
    df[["distinct_screens", "n_records", "as_query", "as_array"]] = (
        df[["distinct_screens", "n_records", "as_query", "as_array"]].fillna(0).astype(int)
    )
    df = df.merge(smf, on="gene", how="left")

    df["is_essential"] = df["gene"].isin(ess)

    df["is_mito"] = df["gene"].str.startswith("Q")
    df["fails_fitness"] = df["min_smf"].isna() | (df["min_smf"] < MIN_SMF)
    df["passes_support"] = df["distinct_screens"] >= MIN_SCREENS
    df["present_in_kuzmin"] = df["n_records"] > 0
    df["keep"] = ~df["is_essential"] & ~df["is_mito"] & ~df["fails_fitness"]

    print(f"\ndropped essential:        {int(df['is_essential'].sum()):,}")
    print(f"dropped mitochondrial:    {int(df['is_mito'].sum()):,}")
    print(f"dropped smf < {MIN_SMF} or absent: {int(df['fails_fitness'].sum()):,}")
    roster = df[df["keep"]].copy()
    print(f"roster after filters:     {len(roster):,}")
    print(f"  metabolic:              {int(roster['is_metabolic'].sum()):,}")
    print(f"  regulator:              {int(roster['is_regulator'].sum()):,}")
    print(f"  both:                   {int((roster['is_metabolic'] & roster['is_regulator']).sum()):,}")
    print(f"  present in Kuzmin:      {int(roster['present_in_kuzmin'].sum()):,}")
    print(f"  >= {MIN_SCREENS} screens:         {int(roster['passes_support'].sum()):,}")

    df.to_csv(osp.join(RESULTS_DIR, "gene_candidates.csv"), index=False)
    with open(osp.join(RESULTS_DIR, "gene_list.txt"), "w") as f:
        for g in roster["gene"]:
            f.write(f"{g}\n")

    # ---- exact sizing under each constraint ------------------------------------
    def klass(row, support_col):
        role = ("B" if row["is_metabolic"] and row["is_regulator"]
                else "M" if row["is_metabolic"] else "R")
        return role + ("H" if row[support_col] else "L")

    sizing = {}
    for support_col, support_name in (
        ("passes_support", f">= {MIN_SCREENS} distinct screens"),
        ("present_in_kuzmin", "present in Kuzmin at all"),
    ):
        classes = Counter(roster.apply(lambda r: klass(r, support_col), axis=1))
        classes = dict(classes)

        def n_met(combo):
            return sum(c[0] in "MB" for c in combo)

        def n_reg(combo):
            return sum(c[0] in "RB" for c in combo)

        def has_support(combo):
            return any(c[1] == "H" for c in combo)

        variants = {
            "unconstrained": lambda c: True,
            "composition only (1-2 metabolic, 1-2 regulator)":
                lambda c: 1 <= n_met(c) <= 2 and 1 <= n_reg(c) <= 2,
            "support only (>=1 supported gene)": has_support,
            "composition AND support":
                lambda c: 1 <= n_met(c) <= 2 and 1 <= n_reg(c) <= 2 and has_support(c),
            "composition AND >=2 supported genes":
                lambda c: (1 <= n_met(c) <= 2 and 1 <= n_reg(c) <= 2
                           and sum(x[1] == "H" for x in c) >= 2),
            "composition AND all 3 supported":
                lambda c: (1 <= n_met(c) <= 2 and 1 <= n_reg(c) <= 2
                           and all(x[1] == "H" for x in c)),
        }
        sizing[support_name] = {
            "classes": classes,
            "counts": {k: count_triples(classes, v) for k, v in variants.items()},
        }

    print("\n=== exact triple counts ===")
    for support_name, block in sizing.items():
        print(f"\nsupport definition: {support_name}")
        print(f"  class sizes: {block['classes']}")
        for k, v in block["counts"].items():
            print(f"  {k:52s} {v:>15,}")

    # Throughput was measured at 1,505 triples/s on one GilaHyper GPU by
    # score_010_checkpoints_directly.py; four GPUs run four shards.
    rate = 1505.0
    headline = sizing[f">= {MIN_SCREENS} distinct screens"]["counts"]["composition AND support"]
    print(f"\nat {rate:,.0f} triples/s/GPU, the composition+support space takes "
          f"{headline / rate / 3600:.1f} GPU-hours, "
          f"{headline / rate / 3600 / 4:.1f} h on 4 GPUs, per checkpoint")

    summary = {
        "n_genome": len(genome_genes),
        "n_metabolic": len(met),
        "n_tflink_regulators": len(tflink),
        "n_sgd_regulators": len(sgd_reg),
        "n_regulators_union": len(reg),
        "n_both": len(met & reg),
        "n_candidates": len(candidates),
        "n_roster": int(len(roster)),
        "min_smf": MIN_SMF,
        "min_screens": MIN_SCREENS,
        "throughput_triples_per_s_per_gpu": rate,
        "sizing": sizing,
    }
    with open(osp.join(RESULTS_DIR, "sizing.json"), "w") as f:
        json.dump(summary, f, indent=2)

    plot(df, roster, sizing, osp.join(IMAGES_DIR, "inference_4_gene_selection"))
    print(f"\nwrote {RESULTS_DIR} and figures to {IMAGES_DIR}")


def plot(df, roster, sizing, out_stem):
    set_plot_style()
    fig, axes = plt.subplots(
        1, 3,
        figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(56.0)),
        gridspec_kw={"width_ratios": [1.0, 1.15, 1.0]},
    )

    ax = axes[0]
    labels = ["metabolic\nonly", "regulator\nonly", "both"]
    m_only = int((roster["is_metabolic"] & ~roster["is_regulator"]).sum())
    r_only = int((~roster["is_metabolic"] & roster["is_regulator"]).sum())
    both = int((roster["is_metabolic"] & roster["is_regulator"]).sum())
    vals = [m_only, r_only, both]
    sup = [
        int((roster["is_metabolic"] & ~roster["is_regulator"] & roster["passes_support"]).sum()),
        int((~roster["is_metabolic"] & roster["is_regulator"] & roster["passes_support"]).sum()),
        int((roster["is_metabolic"] & roster["is_regulator"] & roster["passes_support"]).sum()),
    ]
    xs = np.arange(3)
    ax.bar(xs, vals, 0.6, color="0.78", edgecolor="black", linewidth=0.4, zorder=3,
           label="in roster")
    ax.bar(xs, sup, 0.6, color=PLOT_PALETTE[0], edgecolor="black", linewidth=0.4,
           zorder=4, label="and screen-supported")
    for x, v, s in zip(xs, vals, sup):
        ax.text(x, v + max(vals) * 0.02, f"{v}", ha="center", va="bottom", fontsize=5)
        ax.text(x, s / 2, f"{s}", ha="center", va="center", fontsize=5, color="black")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Genes")
    ax.set_title("a  The roster splits cleanly", fontsize=6, loc="left", pad=3)
    ax.legend(loc="upper right", frameon=True, fontsize=5, handlelength=1.0,
              labelspacing=0.25, borderpad=0.3)

    # Presence is not support: the two gates differ by an order of magnitude.
    ax = axes[1]
    r = roster[roster["present_in_kuzmin"]]
    bins = np.logspace(0, np.log10(max(r["distinct_screens"].max(), 2)), 40)
    ax.hist(r["distinct_screens"], bins=bins, color=PLOT_PALETTE[2], edgecolor="black",
            linewidth=0.3, zorder=3)
    ax.axvline(50, color="black", linewidth=0.6, linestyle=":", zorder=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Distinct Kuzmin query screens")
    ax.set_ylabel("Genes")
    ax.set_title(
        "b  Presence is not support\n"
        f"{int(roster['present_in_kuzmin'].sum())} present, "
        f"{int(roster['passes_support'].sum())} above the gate",
        fontsize=6, loc="left", pad=3,
    )

    ax = axes[2]
    block = sizing[">= 50 distinct screens"]["counts"]
    keys = ["unconstrained", "composition only (1-2 metabolic, 1-2 regulator)",
            "support only (>=1 supported gene)", "composition AND support",
            "composition AND >=2 supported genes", "composition AND all 3 supported"]
    short = ["none", "composition", "support", "both", "both, 2 supported",
             "both, 3 supported"]
    vals = [block[k] for k in keys]
    ys = np.arange(len(vals))[::-1]
    ax.barh(ys, vals, 0.62, color=PLOT_PALETTE[1], edgecolor="black", linewidth=0.4,
            zorder=3)
    for y, v in zip(ys, vals):
        ax.text(v * 1.15, y, f"{v:,}", va="center", fontsize=5)
    ax.set_xscale("log")
    ax.set_yticks(ys)
    ax.set_yticklabels(short)
    ax.set_xlabel("Triples in the inference space")
    ax.set_xlim(right=max(vals) * 12)
    ax.set_title("c  What each constraint costs", fontsize=6, loc="left", pad=3)

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_color("black")
        ax.grid(axis="y", which="major", color="0.85", linewidth=0.3, zorder=0)
        ax.set_axisbelow(True)

    fig.suptitle(
        "The inference_4 roster: yeast-GEM metabolism crossed with TFLink and SGD "
        "regulators, filtered by the inference_1 essentiality and fitness rules.",
        fontsize=6, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(f"{out_stem}.png", dpi=300)
    savefig_true_size_svg(fig, f"{out_stem}.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
