# experiments/010-kuzmin-tmi/scripts/inference_4_panel_design.py
# [[experiments.010-kuzmin-tmi.scripts.inference_4_panel_design]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/inference_4_panel_design
#
# A 20-strain panel from inference_4, built around the question a strain engineer has.
#
# THE QUESTION. Flux-rechanneling deletions cost growth. If a third deletion, typically
# regulatory, restores that growth without restoring the competing flux, the triple
# carries a positive trigenic interaction and the strain is worth building. This selects
# a panel to test exactly that, under a hard budget of 20 CONSTRUCTED strains.
#
# THE ENGINEERING AXIS IS DERIVED, NOT HAND-PICKED. Marquez-Zavala 2026
# (DOI 10.1016/j.ymben.2026.03.017, mirrored as marquez-zavalaDatabase15000Strain2026,
# paper.md sha256 952805b4205164dbf43d350eacf48e1c02f9aaad8fe9bc5529d4da358e1d8c68) mined
# over 15,000 strain-design articles and reports that the frequently targeted genes
# concentrate in central carbon metabolism, naming four areas: the pyruvate node, the
# upper-glycolysis / pentose-phosphate branch point, TCA entry, and the fermentative
# pathways. Those four map onto yeast-GEM 9.0.2 subsystems, so ENGINEERING_SUBSYSTEMS
# below turns a prose claim into a reproducible gene set. Nothing is chosen by name.
#
# WHY tau CLOSURE SETS THE BUDGET. A trigenic score consumes seven fitnesses,
#     tau_abc = f_abc - f_ab f_c - f_ac f_b - f_bc f_a + 2 f_a f_b f_c,
# so every triple needs all three of its doubles and all three of its singles measured
# beside it. Strain cost of a design is therefore |genes| + |pairs| + |triples|, and the
# search maximizes the number of CLOSED tau under cost <= MAX_STRAINS rather than the
# number of triples built.
#
# WHAT THE MEASURED LOWER RUNGS BUY. Rearranging the same identity,
#     f_abc = tau_abc + (f_ab f_c + f_ac f_b + f_bc f_a - 2 f_a f_b f_c),
# so a predicted tau plus PUBLISHED singles and doubles gives a predicted triple
# FITNESS, in absolute units, against a multiplicative expectation built entirely from
# measurements. That converts an uncalibrated interaction score into the quantity the
# bench actually reads, and it is reported triple by triple.
#
# NOT A MEASUREMENT. Every tau here is a model output on an unlabeled space, and the
# model has not been refit on the query-pair-disjoint split. The singles, doubles and
# expectations ARE measurements, and are labeled by source per rung.
#
# Run from repo root:
#   ~/miniconda3/envs/torchcell/bin/python \
#     experiments/010-kuzmin-tmi/scripts/inference_4_panel_design.py
#
# Outputs, all under results/inference_4/:
#   engineering_axis_genes.csv    the derived axis, with subsystem membership
#   engineering_strata.csv        predicted tau by engineering content of the triple
#   panel20_candidates.csv        every triple inside the candidate shortlist
#   panel20_designs.csv           every feasible design under the strain budget
#   panel20_triples.csv           the chosen design, triple by triple, with the ladder
#   panel20_strains.csv           the 20 strains to construct
#   panel20_summary.json          counts quoted in prose
#   $ASSET_IMAGES_DIR/010-kuzmin-tmi/inference_4_panel_design.{png,svg}

import glob
import json
import os
import os.path as osp
from itertools import combinations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from dotenv import load_dotenv
from matplotlib.ticker import MultipleLocator
from pydantic import BaseModel, Field

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

BASE = osp.join(DATA_ROOT, "data/torchcell/experiments/010-kuzmin-tmi/inference_4")
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi", "results", "inference_4")
IMAGES_DIR = osp.join(ASSET_IMAGES_DIR, "010-kuzmin-tmi")

CHECKPOINTS = {"lzs9pcj3": ("M01", 0.4520), "yv4r30bi": ("M02", 0.4472),
               "c7671wgj": ("M03", 0.4619)}

# The four central-carbon areas Marquez-Zavala 2026 names, as yeast-GEM subsystems.
ENGINEERING_SUBSYSTEMS = {
    "Glycolysis / gluconeogenesis",
    "Pentose phosphate pathway",
    "Citrate cycle (TCA cycle)",
    "Pyruvate metabolism",
}

CUTS = [0.08, 0.12, 0.16, 0.20, 0.30]
CONSENSUS_CUT = 0.08      # every checkpoint must clear this for a triple to be a candidate
MAX_STRAINS = 20          # constructed genotypes, wild type excluded
SHORTLIST_N = 14
LABEL_SD = 0.06326
HO = "YDL227C"            # the ho-delta half of a Kuzmin single-mutant cross


class Strain(BaseModel):
    """One genotype to construct, and where its fitness comes from if already known."""

    genotype: tuple[str, ...]
    order: int = Field(ge=1, le=3)
    measured_fitness: float | None = None
    fitness_source: str = "to_measure"


class DesignTriple(BaseModel):
    """One triple of the panel with its predicted interaction and its measured ladder."""

    genes: tuple[str, str, str]
    arm: str
    worst: float
    mean: float
    per_checkpoint: dict[str, float]
    f_singles: dict[str, float | None] = {}
    f_doubles: dict[str, float | None] = {}
    f_expected: float | None = None
    f_triple_predicted: float | None = None


def set_plot_style():
    """Called inside plot(): constructing a dataset resets rcParams and savefig.bbox."""
    plt.rcParams.update(
        {
            "font.family": "Arial", "font.size": 6, "axes.labelsize": 6,
            "axes.titlesize": 6, "xtick.labelsize": 6, "ytick.labelsize": 6,
            "legend.fontsize": 5, "legend.title_fontsize": 5, "figure.titlesize": 6,
            "svg.fonttype": "none", "axes.linewidth": 0.5,
            "savefig.bbox": None, "savefig.pad_inches": 0.0,
        }
    )


def root(name: str) -> str:
    return osp.join(DATA_ROOT, "data/torchcell", name)


# --------------------------------------------------------------------------- inputs


def engineering_axis() -> dict[str, set[str]]:
    """Gene -> the named central-carbon subsystems it carries, from yeast-GEM 9.0.2."""
    from torchcell.metabolism.yeast_GEM import YeastGEM

    gem = YeastGEM(root=root("yeast-GEM"))
    present = {r.subsystem for r in gem.model.reactions if r.subsystem}
    missing = ENGINEERING_SUBSYSTEMS - present
    if missing:
        raise SystemExit(f"yeast-GEM has no subsystem named {sorted(missing)}")
    out: dict[str, set[str]] = {}
    for r in gem.model.reactions:
        if r.subsystem in ENGINEERING_SUBSYSTEMS:
            for g in r.genes:
                out.setdefault(g.id, set()).add(r.subsystem)
    return out


def common_names() -> dict[str, str]:
    """Systematic -> standard gene name, from the R64 annotation the resolver reads."""
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

    genome = SCerevisiaeGenome(osp.join(DATA_ROOT, "data/sgd/genome"), overwrite=False)
    out: dict[str, str] = {}
    for standard, ids in genome.feature_index["standard_to_ids"].items():
        for i in ids:
            out.setdefault(i, standard)
    return out


def load_checkpoint(tag: str) -> np.ndarray:
    """One checkpoint's predictions in dataset order, verified contiguous."""
    files = sorted(glob.glob(osp.join(BASE, "inferred", f"*{tag}*shard*.parquet")))
    if len(files) != 4:
        raise SystemExit(f"{tag}: expected 4 shard files, found {len(files)}")
    parts, cursor = [], 0
    for f in files:
        t = pq.read_table(f, columns=["index", "prediction"])
        i = t["index"].to_numpy()
        if i[0] != cursor or not np.all(np.diff(i) == 1):
            raise SystemExit(f"{tag}: shard {f} is not contiguous from {cursor}")
        cursor = int(i[-1]) + 1
        parts.append(t["prediction"].to_numpy())
    return np.concatenate(parts).astype(np.float32)


# ------------------------------------------------------------------ measured rungs


def single_lookup(genes: set[str]) -> dict[str, tuple[float, str]]:
    """Gene -> (fitness, source). Kuzmin's own screens first, Costanzo 2016 at 30 C after."""
    from torchcell.datasets.scerevisiae import SmfCostanzo2016Dataset
    from torchcell.datasets.scerevisiae.kuzmin2018 import SmfKuzmin2018Dataset
    from torchcell.datasets.scerevisiae.kuzmin2020 import SmfKuzmin2020Dataset

    out: dict[str, tuple[float, str]] = {}

    df = SmfKuzmin2020Dataset(root=root("smf_kuzmin2020")).df
    gene = np.where(df["ORF1"] == HO, df["ORF2"], df["ORF1"])
    df = df.assign(gene=gene)
    df = df[df["gene"].isin(genes) & (df["Mutant type"] == "Single mutant")]
    for g, block in df.groupby("gene"):
        f = float(block["Fitness"].mean())
        if np.isfinite(f):
            out.setdefault(str(g), (f, "SmfKuzmin2020"))

    k18 = SmfKuzmin2018Dataset(root=root("smf_kuzmin2018")).df
    col = "Query systematic name no ho"
    if col in k18.columns:
        for g, block in k18[k18[col].isin(genes)].groupby(col):
            f = float(block["Combined mutant fitness"].mean())
            if np.isfinite(f):
                out.setdefault(str(g), (f, "SmfKuzmin2018"))

    cs = SmfCostanzo2016Dataset(root=root("smf_costanzo2016")).df
    cs = cs[
        cs["Systematic gene name"].isin(genes)
        & (cs["Temperature"] == 30)
        & cs["perturbation_type"].str.contains("deletion")
    ]
    for g, block in cs.groupby("Systematic gene name"):
        f = float(block["Single mutant fitness"].mean())
        if np.isfinite(f):
            out.setdefault(str(g), (f, "SmfCostanzo2016_30C"))
    return out


def double_lookup(pairs: set[frozenset[str]]) -> dict[frozenset[str], tuple[float, float | None, str]]:
    """Pair -> (double-mutant fitness, published epsilon or None, source)."""
    from torchcell.datasets.scerevisiae.costanzo2016 import DmiCostanzo2016Dataset
    from torchcell.datasets.scerevisiae.kuzmin2018 import DmfKuzmin2018Dataset
    from torchcell.datasets.scerevisiae.kuzmin2020 import DmfKuzmin2020Dataset

    want_genes = {g for p in pairs for g in p}
    out: dict[frozenset[str], tuple[float, float | None, str]] = {}

    for cls, rt, fit_col, src in (
        (DmfKuzmin2020Dataset, "dmf_kuzmin2020", "Double/triple mutant fitness", "DmfKuzmin2020"),
        (DmfKuzmin2018Dataset, "dmf_kuzmin2018", "Combined mutant fitness", "DmfKuzmin2018"),
    ):
        df = cls(root=root(rt)).df
        q = df["Query systematic name no ho"].to_numpy().astype(str)
        a = df["Array systematic name"].to_numpy().astype(str)
        hit = np.isin(q, list(want_genes)) & np.isin(a, list(want_genes))
        if not hit.any():
            continue
        f = df[fit_col].to_numpy(dtype=float)[hit]
        for qi, ai, fi in zip(q[hit], a[hit], f):
            key = frozenset((qi, ai))
            if len(key) == 2 and key in pairs and np.isfinite(fi):
                out.setdefault(key, (float(fi), None, src))

    dmi = DmiCostanzo2016Dataset(root=root("dmi_costanzo2016")).df
    dmi = dmi[dmi["Temperature"] == 30]
    q = dmi["Query Systematic Name"].to_numpy().astype(str)
    a = dmi["Array Systematic Name"].to_numpy().astype(str)
    hit = np.isin(q, list(want_genes)) & np.isin(a, list(want_genes))
    if hit.any():
        f = dmi["Double mutant fitness"].to_numpy(dtype=float)[hit]
        e = dmi["Genetic interaction score (ε)"].to_numpy(dtype=float)[hit]
        agg: dict[frozenset[str], list[tuple[float, float]]] = {}
        for qi, ai, fi, ei in zip(q[hit], a[hit], f, e):
            key = frozenset((qi, ai))
            if len(key) == 2 and key in pairs and np.isfinite(fi):
                agg.setdefault(key, []).append((float(fi), float(ei)))
        for key, vals in agg.items():
            fm = float(np.mean([v[0] for v in vals]))
            em = float(np.nanmean([v[1] for v in vals]))
            out.setdefault(key, (fm, em, "DmiCostanzo2016_30C"))
    return out


# ------------------------------------------------------------------- design search


def strain_cost(triples: list[tuple[str, str, str]]) -> tuple[int, set[str], set[frozenset[str]]]:
    """Strains needed to close tau on every triple: singles + doubles + the triples."""
    genes: set[str] = set()
    pairs: set[frozenset[str]] = set()
    for t in triples:
        genes.update(t)
        for p in combinations(t, 2):
            pairs.add(frozenset(p))
    return len(genes) + len(pairs) + len(triples), genes, pairs


def search_designs(
    cand: pd.DataFrame, shortlist: list[str], chassis: tuple[str, str], max_strains: int
) -> pd.DataFrame:
    """Every design under the strain budget whose gene set contains the chassis pair.

    Only triples the inference space actually contains are eligible, so a design is
    always scoreable. Gene sets are capped at 5: six genes cost six singles before any
    pair is built, which cannot close more tau than five within this budget.
    """
    have = {
        frozenset((r.g1, r.g2, r.g3)): (r.worst, r.mean)
        for r in cand.itertuples()
    }
    others = [g for g in shortlist if g not in chassis]
    rows = []
    for extra_k in (2, 3):
        for extra in combinations(others, extra_k):
            gset = list(chassis) + list(extra)
            avail = [t for t in combinations(sorted(gset), 3) if frozenset(t) in have]
            if not avail:
                continue
            best: tuple | None = None
            for k in range(len(avail), 0, -1):
                if best is not None:
                    break
                for T in combinations(avail, k):
                    cost, genes, pairs = strain_cost(list(T))
                    if cost > max_strains:
                        continue
                    worsts = [have[frozenset(t)][0] for t in T]
                    score = float(np.median(worsts))
                    if best is None or score > best[0]:
                        best = (score, T, cost, len(genes), len(pairs))
            if best is None:
                continue
            score, T, cost, n_genes, n_pairs = best
            both = sum(1 for t in T if chassis[0] in t and chassis[1] in t)
            rows.append({
                "genes": "+".join(sorted(gset)),
                "n_genes_used": n_genes,
                "n_pairs": n_pairs,
                "n_triples": len(T),
                "strains": cost,
                "median_worst": score,
                "min_worst": float(min(have[frozenset(t)][0] for t in T)),
                "max_worst": float(max(have[frozenset(t)][0] for t in T)),
                "n_chassis_pair_triples": both,
                "triples": ";".join("+".join(t) for t in T),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit("no feasible design under the strain budget")
    return out.sort_values(
        ["n_triples", "n_chassis_pair_triples", "median_worst"], ascending=False
    ).reset_index(drop=True)


# --------------------------------------------------------------------------- main


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    print("deriving the engineering axis from yeast-GEM ...")
    axis = engineering_axis()
    names = common_names()

    roster = pd.read_csv(osp.join(RESULTS_DIR, "gene_candidates.csv"))
    roster = roster[roster["keep"]].reset_index(drop=True)
    keep = set(roster["gene"])
    eng = sorted(g for g in axis if g in keep)
    print(f"  {len(axis)} yeast-GEM genes in the four subsystems, {len(eng)} in the "
          f"{len(keep)}-gene roster")

    pd.DataFrame({
        "gene": eng,
        "name": [names.get(g, g) for g in eng],
        "subsystems": ["; ".join(sorted(axis[g])) for g in eng],
    }).to_csv(osp.join(RESULTS_DIR, "engineering_axis_genes.csv"), index=False)

    print("loading checkpoints ...")
    preds = {t: load_checkpoint(t) for t in CHECKPOINTS}
    stack = np.stack([preds[t] for t in CHECKPOINTS], axis=1)
    worst, mean = stack.min(axis=1), stack.mean(axis=1)

    idx = pq.read_table(
        osp.join(BASE, "triple_index.parquet"),
        columns=["index", "gene1", "gene2", "gene3", "n_supported"],
    )
    if idx.num_rows != len(mean):
        raise SystemExit("triple_index.parquet does not align with the predictions")
    g1 = idx["gene1"].to_numpy(zero_copy_only=False)
    g2 = idx["gene2"].to_numpy(zero_copy_only=False)
    g3 = idx["gene3"].to_numpy(zero_copy_only=False)
    n_sup = idx["n_supported"].to_numpy()

    eng_set = set(eng)
    n_eng = (np.isin(g1, eng).astype(np.int8) + np.isin(g2, eng).astype(np.int8)
             + np.isin(g3, eng).astype(np.int8))

    # How far into the positive tail the engineering-loaded triples reach.
    rows = []
    for k in sorted(set(n_eng.tolist())):
        m = n_eng == k
        row = {"n_engineering_genes": int(k), "triples": int(m.sum()),
               "share": float(m.mean()), "best_mean": float(mean[m].max()),
               "best_worst": float(worst[m].max())}
        for c in CUTS:
            row[f"consensus_{c}"] = int((stack[m] > c).all(axis=1).sum())
        rows.append(row)
    strata = pd.DataFrame(rows)
    strata.to_csv(osp.join(RESULTS_DIR, "engineering_strata.csv"), index=False)
    print("\n=== predicted tau by engineering content of the triple ===")
    print(strata.to_string(index=False))

    # Does the ranked head overlap what strain engineering actually deletes? Asked of
    # the SAME top 500 the ranking figure reports, so the two are directly comparable.
    top500 = pd.read_csv(osp.join(RESULTS_DIR, "top_triples.csv"))
    head_genes = pd.Series(
        np.concatenate([top500["gene1"], top500["gene2"], top500["gene3"]])
    ).value_counts()
    head_axis = [g for g in head_genes.index if g in eng_set]
    print("\n=== engineering-axis overlap with the ranked head ===")
    print(f"  {len(head_axis)} of the {len(head_genes)} genes carrying the top 500 are "
          f"on the axis ({len(head_axis) / len(head_genes):.1%}), against "
          f"{len(eng) / len(keep):.1%} of the roster")
    for g in head_axis:
        print(f"    {g} {names.get(g, g):<8} in {int(head_genes[g])} of the top 500")
    head_overlap = {
        "n_head_genes": int(len(head_genes)),
        "n_head_genes_on_axis": int(len(head_axis)),
        "axis_share_of_head_genes": float(len(head_axis) / len(head_genes)),
        "axis_share_of_roster": float(len(eng) / len(keep)),
        "head_axis_genes": [
            {"gene": g, "name": names.get(g, g), "top500_triples": int(head_genes[g])}
            for g in head_axis
        ],
    }

    # Candidate pool: every checkpoint positive, and at least two engineering genes.
    pool = (stack > CONSENSUS_CUT).all(axis=1) & (n_eng >= 2)
    print(f"\nconsensus > {CONSENSUS_CUT} with >= 2 engineering genes: {int(pool.sum())}")
    if pool.sum() == 0:
        raise SystemExit("no candidate triples: the engineering axis reaches no consensus tail")

    hits = pd.Series(np.concatenate([g1[pool], g2[pool], g3[pool]])).value_counts()
    shortlist = list(hits.index[:SHORTLIST_N])
    print("\ngenes carrying that pool:")
    for g, c in hits.head(SHORTLIST_N).items():
        print(f"  {g} {names.get(g, g):<8} {c:>4}  "
              f"{'engineering axis' if g in eng_set else 'regulator or other'}")

    # Every triple fully inside the shortlist, so the design search can score any of them.
    m = np.isin(g1, shortlist) & np.isin(g2, shortlist) & np.isin(g3, shortlist)
    ii = np.where(m)[0]
    cand = pd.DataFrame({
        "index": ii, "g1": g1[ii], "g2": g2[ii], "g3": g3[ii],
        **{CHECKPOINTS[t][0]: preds[t][ii] for t in CHECKPOINTS},
        "worst": worst[ii], "mean": mean[ii], "n_supported": n_sup[ii],
        "n_eng": np.isin(g1[ii], eng).astype(int) + np.isin(g2[ii], eng).astype(int)
                 + np.isin(g3[ii], eng).astype(int),
    })
    cand.to_csv(osp.join(RESULTS_DIR, "panel20_candidates.csv"), index=False)
    print(f"\ntriples of the space fully inside the shortlist: {len(cand)}")

    # The chassis pair: the two engineering-axis genes co-occurring most in the pool.
    pair_counts: dict[frozenset[str], int] = {}
    for a, b, c in zip(g1[pool], g2[pool], g3[pool]):
        for x, y in combinations((a, b, c), 2):
            if x in eng_set and y in eng_set:
                pair_counts[frozenset((x, y))] = pair_counts.get(frozenset((x, y)), 0) + 1
    # Ties on co-occurrence are broken by screen support, the same gate the roster used:
    # two pairs appearing in equally many pool triples are not equally constrained by the
    # trigenic data, and the better-screened pair is the one a panel should rest on.
    screens = dict(zip(roster["gene"], roster["distinct_screens"]))
    chassis_fs = max(
        pair_counts,
        key=lambda k: (pair_counts[k], sum(screens.get(g, 0) for g in k)),
    )
    chassis = tuple(sorted(chassis_fs))
    print(f"\nchassis pair: {chassis[0]} ({names.get(chassis[0])}) + "
          f"{chassis[1]} ({names.get(chassis[1])}), in {pair_counts[chassis_fs]} pool "
          f"triples, {sum(screens.get(g, 0) for g in chassis_fs)} screens between them")
    print("  competing pairs:")
    for k in sorted(pair_counts, key=lambda k: -pair_counts[k])[:5]:
        gg = sorted(k)
        print(f"    {names.get(gg[0], gg[0]):<6} {names.get(gg[1], gg[1]):<6} "
              f"pool {pair_counts[k]:>2}  screens {sum(screens.get(g, 0) for g in k):>4}")

    designs = search_designs(cand, shortlist, chassis, MAX_STRAINS)
    designs.to_csv(osp.join(RESULTS_DIR, "panel20_designs.csv"), index=False)
    print(f"\n=== feasible designs under {MAX_STRAINS} strains (top 10) ===")
    print(designs.head(10).drop(columns=["triples"]).to_string(index=False))

    chosen = designs.iloc[0]
    triples = [tuple(t.split("+")) for t in chosen["triples"].split(";")]
    cost, genes, pairs = strain_cost(triples)
    print(f"\nchosen: {len(triples)} closed tau on {cost} strains over {len(genes)} genes")

    # Measured lower rungs for the chosen design.
    print("\nlooking up measured singles and doubles ...")
    singles = single_lookup(genes)
    doubles = double_lookup(pairs)
    print(f"  singles resolved {len(singles)}/{len(genes)}, "
          f"doubles resolved {len(doubles)}/{len(pairs)}")

    lookup = {frozenset((r.g1, r.g2, r.g3)): r for r in cand.itertuples()}
    out_rows = []
    for t in triples:
        r = lookup[frozenset(t)]
        a, b, c = t
        fa, fb, fc = (singles.get(x, (np.nan, "missing"))[0] for x in (a, b, c))
        fab = doubles.get(frozenset((a, b)), (np.nan, None, "missing"))[0]
        fac = doubles.get(frozenset((a, c)), (np.nan, None, "missing"))[0]
        fbc = doubles.get(frozenset((b, c)), (np.nan, None, "missing"))[0]
        f_exp = fab * fc + fac * fb + fbc * fa - 2 * fa * fb * fc
        arm = "chassis pair" if chassis[0] in t and chassis[1] in t else "one chassis gene"
        # Validated before it is flattened, so a missing rung is an error rather than a
        # silent NaN propagated into a predicted fitness.
        DesignTriple(
            genes=(a, b, c), arm=arm, worst=float(r.worst), mean=float(r.mean),
            per_checkpoint={CHECKPOINTS[k][0]: float(getattr(r, CHECKPOINTS[k][0]))
                            for k in CHECKPOINTS},
            f_singles={x: (None if not np.isfinite(v) else float(v))
                       for x, v in zip(t, (fa, fb, fc))},
            f_doubles={"+".join(sorted(pp)): (None if not np.isfinite(v) else float(v))
                       for pp, v in zip(combinations(t, 2), (fab, fac, fbc))},
            f_expected=None if not np.isfinite(f_exp) else float(f_exp),
            f_triple_predicted=None if not np.isfinite(f_exp) else float(f_exp + r.worst),
        )
        out_rows.append({
            "triple": "+".join(t), "gene1": a, "gene2": b, "gene3": c,
            "name1": names.get(a, a), "name2": names.get(b, b), "name3": names.get(c, c),
            "arm": arm,
            "worst": r.worst, "mean": r.mean,
            **{CHECKPOINTS[k][0]: getattr(r, CHECKPOINTS[k][0]) for k in CHECKPOINTS},
            "n_supported": r.n_supported,
            "f_a": fa, "f_b": fb, "f_c": fc,
            "f_ab": fab, "f_ac": fac, "f_bc": fbc,
            "f_expected": f_exp,
            "f_triple_predicted_worst": f_exp + r.worst,
            "f_triple_predicted_mean": f_exp + r.mean,
            "rescue_over_worst_double": (f_exp + r.worst) - np.nanmin([fab, fac, fbc]),
        })
    panel = pd.DataFrame(out_rows).sort_values("worst", ascending=False)
    panel.to_csv(osp.join(RESULTS_DIR, "panel20_triples.csv"), index=False)
    print("\n=== the panel, triple by triple ===")
    print(panel[["triple", "name1", "name2", "name3", "arm", "worst", "mean",
                 "f_expected", "f_triple_predicted_worst"]]
          .to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    # The build list, validated rather than assembled as loose dicts.
    strains: list[Strain] = []
    for g in sorted(genes):
        f, src = singles.get(g, (None, "to_measure"))
        strains.append(Strain(genotype=(g,), order=1, measured_fitness=f,
                              fitness_source=src))
    eps_of: dict[str, float | None] = {}
    for p in sorted(pairs, key=lambda s: sorted(s)):
        gg = tuple(sorted(p))
        f, eps, src = doubles.get(p, (None, None, "to_measure"))
        strains.append(Strain(genotype=gg, order=2, measured_fitness=f,
                              fitness_source=src))
        eps_of["+".join(gg)] = eps
    for t in triples:
        strains.append(Strain(genotype=tuple(t), order=3, fitness_source="to_measure"))
    if len(strains) != cost:
        raise SystemExit(f"build list is {len(strains)} strains, cost said {cost}")
    strain_df = pd.DataFrame(
        [
            {
                "genotype": "+".join(s.genotype), "order": s.order,
                "name": "+".join(names.get(g, g) for g in s.genotype),
                "measured_fitness": s.measured_fitness,
                "published_epsilon": eps_of.get("+".join(s.genotype)),
                "fitness_source": s.fitness_source,
            }
            for s in strains
        ]
    )
    strain_df.to_csv(osp.join(RESULTS_DIR, "panel20_strains.csv"), index=False)
    print(f"\n=== {len(strain_df)} strains to construct ===")
    print(strain_df.to_string(index=False))

    summary = {
        "engineering_subsystems": sorted(ENGINEERING_SUBSYSTEMS),
        "n_engineering_axis_genes_in_gem": len(axis),
        "n_engineering_axis_genes_in_roster": len(eng),
        "n_roster_genes": len(keep),
        "strata": strata.to_dict(orient="records"),
        "head_overlap": head_overlap,
        "consensus_cut": CONSENSUS_CUT,
        "n_pool_triples": int(pool.sum()),
        "shortlist": [{"gene": g, "name": names.get(g, g),
                       "engineering_axis": g in eng_set, "pool_triples": int(hits[g])}
                      for g in shortlist],
        "chassis": [{"gene": g, "name": names.get(g, g),
                     "subsystems": sorted(axis.get(g, []))} for g in chassis],
        "chassis_pool_triples": int(pair_counts[chassis_fs]),
        "max_strains": MAX_STRAINS,
        "n_feasible_designs": int(len(designs)),
        "chosen": {
            "genes": chosen["genes"], "n_triples": int(chosen["n_triples"]),
            "strains": int(chosen["strains"]), "n_pairs": int(chosen["n_pairs"]),
            "median_worst": float(chosen["median_worst"]),
            "n_chassis_pair_triples": int(chosen["n_chassis_pair_triples"]),
        },
        "n_singles_already_measured": int(len(singles)),
        "n_doubles_already_measured": int(len(doubles)),
        "n_doubles_needed": int(len(pairs)),
        "label_sd": LABEL_SD,
        "panel": panel.to_dict(orient="records"),
    }
    with open(osp.join(RESULTS_DIR, "panel20_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    plot(strata, cand, hits, names, eng_set, designs, panel, strain_df,
         osp.join(IMAGES_DIR, "inference_4_panel_design"))
    print(f"\nwrote {RESULTS_DIR} and figures to {IMAGES_DIR}")


def _letter(ax, letter):
    ax.text(-0.16, 1.06, letter, transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="bottom", ha="left")


def plot(strata, cand, hits, names, eng_set, designs, panel, strain_df, out_stem):
    set_plot_style()
    fig, axes2 = plt.subplots(
        3, 2, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(168.0))
    )
    axes = axes2.ravel()

    # a: the engineering-loaded strata barely reach the consensus tail.
    ax = axes[0]
    xs = np.arange(len(CUTS))
    w = 0.26
    for j, row in strata.iterrows():
        vals = [max(row[f"consensus_{c}"], 0.6) for c in CUTS]
        ax.bar(xs + (j - 1) * w, vals, w, color=PLOT_PALETTE[int(j)],
               edgecolor="black", linewidth=0.3, zorder=3,
               label=f"{int(row['n_engineering_genes'])} axis gene(s)")
        for x, v, raw in zip(xs + (j - 1) * w, vals, [row[f"consensus_{c}"] for c in CUTS]):
            ax.text(x, v * 1.3, f"{int(raw):,}", ha="center", va="bottom",
                    fontsize=4.5, rotation=90)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"$>{c:+.2f}$" for c in CUTS])
    ax.set_ylabel("Triples, all three checkpoints above")
    ax.set_ylim(0.5, max(strata[f"consensus_{CUTS[0]}"]) * 40)
    ax.set_title("The engineering axis barely reaches the consensus tail",
                 fontsize=6, loc="left", pad=3)
    ax.legend(loc="upper right", frameon=True, fontsize=5, handlelength=1.0,
              labelspacing=0.2, borderpad=0.3)

    # b: who carries the candidate pool.
    ax = axes[1]
    top = hits.head(SHORTLIST_N)
    ys = np.arange(len(top))[::-1]
    cols = [PLOT_PALETTE[0] if g in eng_set else PLOT_PALETTE[4] for g in top.index]
    ax.barh(ys, top.to_numpy(), 0.62, color=cols, edgecolor="black",
            linewidth=0.4, zorder=3)
    for y, v in zip(ys, top.to_numpy()):
        ax.text(v + max(top) * 0.01, y, f"{int(v)}", va="center", fontsize=5)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{names.get(g, g)}" for g in top.index], fontsize=5)
    ax.set_xlim(0, float(top.max()) * 1.22)
    ax.set_xlabel("Candidate triples containing the gene")
    ax.set_title(f"Who carries the pool\nconsensus $>{CONSENSUS_CUT:+.2f}$ with two axis genes",
                 fontsize=6, loc="left", pad=3)
    ax.legend(handles=[
        plt.Line2D([], [], marker="s", linestyle="none", markersize=3,
                   color=PLOT_PALETTE[0], label="engineering axis"),
        plt.Line2D([], [], marker="s", linestyle="none", markersize=3,
                   color=PLOT_PALETTE[4], label="regulator or other"),
    ], loc="lower right", frameon=True, fontsize=5, handlelength=1.0,
        labelspacing=0.25, borderpad=0.3)

    # c: what the strain budget buys.
    ax = axes[2]
    ax.scatter(designs["strains"], designs["n_triples"], s=4,
               color=PLOT_PALETTE[5], linewidths=0, zorder=3, label="feasible design")
    best = designs.iloc[0]
    ax.scatter([best["strains"]], [best["n_triples"]], s=22, facecolor="none",
               edgecolor=PLOT_PALETTE[1], linewidths=0.8, zorder=5, label="chosen")
    ax.axvline(MAX_STRAINS, color="black", linewidth=0.5, linestyle="--", zorder=2)
    ax.set_xlabel("Strains to construct")
    ax.set_ylabel("Closed $\\tau$ values")
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.set_title(f"What the budget buys\nbudget {MAX_STRAINS} strains, "
                 f"{len(designs)} feasible designs", fontsize=6, loc="left", pad=3)
    ax.legend(loc="upper left", frameon=True, fontsize=5, handlelength=1.0,
              labelspacing=0.25, borderpad=0.3)

    # d: the panel's predicted interactions, by arm.
    ax = axes[3]
    p = panel.reset_index(drop=True)
    ys = np.arange(len(p))[::-1]
    arm_color = {"chassis pair": PLOT_PALETTE[0], "one chassis gene": PLOT_PALETTE[4]}
    ax.barh(ys, p["worst"], 0.6, color=[arm_color[a] for a in p["arm"]],
            edgecolor="black", linewidth=0.4, zorder=3)
    for y, row in zip(ys, p.itertuples()):
        lo = min(getattr(row, CHECKPOINTS[k][0]) for k in CHECKPOINTS)
        hi = max(getattr(row, CHECKPOINTS[k][0]) for k in CHECKPOINTS)
        ax.plot([lo, hi], [y, y], color="black", linewidth=0.5, zorder=4)
    ax.axvline(0.08, color="0.5", linewidth=0.5, linestyle=":", zorder=2)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r.name1} {r.name2} {r.name3}" for r in p.itertuples()],
                       fontsize=5)
    ax.set_xlabel("Predicted $\\tau$, worst of three checkpoints")
    ax.set_title("The panel\nbar is the worst checkpoint, line spans all three",
                 fontsize=6, loc="left", pad=3)
    ax.legend(handles=[
        plt.Line2D([], [], marker="s", linestyle="none", markersize=3,
                   color=arm_color[a], label=a) for a in ("chassis pair", "one chassis gene")
        if a in set(p["arm"])
    ], loc="lower right", frameon=True, fontsize=5, handlelength=1.0,
        labelspacing=0.25, borderpad=0.3)

    # e: the measured ladder under each triple, and where the prediction puts it.
    ax = axes[4]
    ys = np.arange(len(p))[::-1]
    ax.scatter(p["f_expected"], ys, s=10, marker="s", color="0.55",
               edgecolor="black", linewidths=0.3, zorder=4, label="multiplicative expectation")
    ax.scatter(p["f_triple_predicted_worst"], ys, s=12, marker="o",
               color=PLOT_PALETTE[0], edgecolor="black", linewidths=0.3, zorder=5,
               label="expectation $+$ predicted $\\tau$")
    for y, row in zip(ys, p.itertuples()):
        ax.plot([row.f_expected, row.f_triple_predicted_worst], [y, y],
                color=PLOT_PALETTE[0], linewidth=0.6, zorder=3)
        lo = np.nanmin([row.f_ab, row.f_ac, row.f_bc])
        ax.scatter([lo], [y], s=8, marker="|", color=PLOT_PALETTE[4], zorder=4)
    ax.axvline(1.0, color="black", linewidth=0.5, linestyle="--", zorder=2)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r.name1} {r.name2} {r.name3}" for r in p.itertuples()],
                       fontsize=5)
    ax.set_xlabel("Fitness")
    ax.set_title("What the panel predicts at the bench\n"
                 "dashed line is wild type; tick is the worst measured double",
                 fontsize=6, loc="left", pad=3)
    ax.legend(loc="lower right", frameon=True, fontsize=5, handlelength=1.0,
              labelspacing=0.25, borderpad=0.3)

    # f: every rung of the panel that is already published. Only the six triples are
    # unknown, so these 14 double as a plate-level calibration set.
    ax = axes[5]
    known = strain_df[strain_df["measured_fitness"].notna()].copy()
    known = known.sort_values(["order", "measured_fitness"])
    ys = np.arange(len(known))[::-1]
    cols = [PLOT_PALETTE[0] if o == 1 else PLOT_PALETTE[4] for o in known["order"]]
    ax.barh(ys, known["measured_fitness"], 0.6, color=cols, edgecolor="black",
            linewidth=0.4, zorder=3)
    for y, v in zip(ys, known["measured_fitness"]):
        ax.text(v + 0.006, y, f"{v:.3f}", va="center", fontsize=4.5)
    ax.axvline(1.0, color="black", linewidth=0.5, linestyle="--", zorder=2)
    ax.set_yticks(ys)
    ax.set_yticklabels(known["name"], fontsize=4.5)
    ax.set_xlim(0, float(known["measured_fitness"].max()) * 1.22)
    ax.set_xlabel("Published fitness")
    ax.set_title(
        f"{len(known)} of the {len(strain_df)} strains are already measured\n"
        f"only the {int((strain_df['order'] == 3).sum())} triples are unknown",
        fontsize=6, loc="left", pad=3,
    )
    ax.legend(handles=[
        plt.Line2D([], [], marker="s", linestyle="none", markersize=3,
                   color=PLOT_PALETTE[0], label="single"),
        plt.Line2D([], [], marker="s", linestyle="none", markersize=3,
                   color=PLOT_PALETTE[4], label="double"),
    ], loc="lower right", frameon=True, fontsize=5, handlelength=1.0,
        labelspacing=0.25, borderpad=0.3)

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_color("black")
        ax.grid(axis="y", which="major", color="0.85", linewidth=0.3, zorder=0)
        ax.set_axisbelow(True)
    for ax, letter in zip(axes, "abcdef"):
        _letter(ax, letter)

    fig.suptitle(
        "A 20-strain panel from inference_4. Interactions are predicted; singles, "
        "doubles and the multiplicative expectation are measured.",
        fontsize=6, y=0.997,
    )
    fig.tight_layout(rect=(0.01, 0, 1, 0.975))
    fig.savefig(f"{out_stem}.png", dpi=300)
    savefig_true_size_svg(fig, f"{out_stem}.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
