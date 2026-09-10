#!/usr/bin/env python
# experiments/010-kuzmin-tmi/scripts/inference_4_panel_overlap.py
# [[experiments.010-kuzmin-tmi.scripts.inference_4_panel_overlap]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/inference_4_panel_overlap
"""Is any panel strain, or any inference_4 triple, ALREADY in the 010 build?

THE WRINKLE. inference_4_generate_triples.py enumerates the roster and never removes
combinations the Kuzmin screen already measured, so "unmeasured space" was an
assumption rather than a fact. If a nominated triple sits in 010's TRAIN split its
prediction is recall, not prediction. If it sits in VAL or TEST the answer is already
known and the strain need not be built at all. Either way the panel's premise changes.

IDENTITY. A record's identity is the sorted, de-duplicated set of its perturbed gene
names, the same key transfer_010_tmi_splits.py uses to carry 010's split assignment
into 025. Split membership is read from 010's own cached index_seed_42.json, which is
the split the checkpoints were actually trained under.

FOUR LEVELS, because exact membership is the strictest and rarest of them and a triple
can leak without being present:

  exact triple   the gene set appears verbatim in 010, with its measured tau and split
  query pair     one of the triple's three pairs is a recurring Kuzmin QUERY double.
                 Kuzmin crosses a query double against an array of singles, so a
                 shared query pair is the axis on which the additive null falls from
                 0.400 to 0.127; a triple carrying one is interpolation.
  any pair       any of the three pairs co-occurs in any 010 record
  gene           how many of the three genes appear in any 010 record

Run from repo root:
  ~/miniconda3/envs/torchcell/bin/python \
    experiments/010-kuzmin-tmi/scripts/inference_4_panel_overlap.py

Outputs, under results/inference_4/:
  panel20_overlap.csv        every panel strain, its 010 status and split
  panel20_overlap.json       the counts quoted in prose
  $ASSET_IMAGES_DIR/010-kuzmin-tmi/inference_4_panel_overlap.{png,svg}
"""

import json
import os
import os.path as osp
from collections import Counter
from itertools import combinations

import matplotlib

matplotlib.use("Agg")

import lmdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from dotenv import load_dotenv
from tqdm import tqdm

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
BUILD_010 = osp.join(
    DATA_ROOT, "data/torchcell/experiments/010-kuzmin-tmi/001-small-build"
)
INFERENCE_4 = osp.join(
    DATA_ROOT, "data/torchcell/experiments/010-kuzmin-tmi/inference_4"
)
SPLITS = ("train", "val", "test")
# A Kuzmin query double recurs across the whole array; a pair seen a handful of times
# is an incidental co-occurrence. Five is the floor inference_1_training_overlap.py
# used to recover the 420 query pairs, and it is reused here unchanged.
QUERY_PAIR_MIN_RECURRENCE = 5


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


def load_010():
    """Identity -> (split, tau); plus pair counts and the gene vocabulary of the build."""
    with open(osp.join(BUILD_010, "data_module_cache/index_seed_42.json")) as f:
        index = json.load(f)
    split_of = {i: s for s in SPLITS for i in index[s]}
    print("010 split cache: " + ", ".join(f"{s} {len(index[s]):,}" for s in SPLITS))

    identity_split: dict[tuple[str, ...], str] = {}
    identity_tau: dict[tuple[str, ...], float] = {}
    pair_counts: Counter = Counter()
    genes_seen: set[str] = set()

    env = lmdb.open(osp.join(BUILD_010, "processed/lmdb"), readonly=True, lock=False,
                    readahead=False)
    with env.begin() as txn:
        n = txn.stat()["entries"]
        for i in tqdm(range(n), desc="reading the 010 build"):
            rec = json.loads(txn.get(str(i).encode()))
            exp = rec[0]["experiment"]
            ident = tuple(sorted({p["systematic_gene_name"]
                                  for p in exp["genotype"]["perturbations"]}))
            identity_split[ident] = split_of[i]
            phen = exp["phenotype"]
            identity_tau[ident] = phen.get("gene_interaction")
            genes_seen.update(ident)
            for p in combinations(ident, 2):
                pair_counts[p] += 1
    env.close()
    print(f"  {n:,} records, {len(identity_split):,} distinct identities, "
          f"{len(genes_seen):,} genes")
    query_pairs = {p for p, c in pair_counts.items() if c >= QUERY_PAIR_MIN_RECURRENCE}
    print(f"  {len(query_pairs):,} pairs recur at least "
          f"{QUERY_PAIR_MIN_RECURRENCE} times (the query doubles)")
    return identity_split, identity_tau, set(pair_counts), query_pairs, genes_seen


def space_overlap(identity_split: dict, code: dict[str, int]) -> dict:
    """How much of the whole 41.9M inference_4 space is already in the 010 build."""
    idx = pq.read_table(osp.join(INFERENCE_4, "triple_index.parquet"),
                        columns=["gene1", "gene2", "gene3"])
    g1 = idx["gene1"].to_numpy(zero_copy_only=False)
    g2 = idx["gene2"].to_numpy(zero_copy_only=False)
    g3 = idx["gene3"].to_numpy(zero_copy_only=False)
    n = len(g1)

    # Order-free int64 key over the shared vocabulary. A gene the 010 build never saw
    # gets -1, and any triple containing one cannot be a member, so it is excluded up
    # front rather than looked up.
    def enc(arr):
        return np.array([code.get(g, -1) for g in arr], dtype=np.int64)

    c1, c2, c3 = enc(g1), enc(g2), enc(g3)
    known = (c1 >= 0) & (c2 >= 0) & (c3 >= 0)
    stack = np.sort(np.stack([c1, c2, c3], axis=1), axis=1)
    N = np.int64(len(code) + 1)
    keys = stack[:, 0] * N * N + stack[:, 1] * N + stack[:, 2]

    member_keys = {}
    for ident, split in identity_split.items():
        cs = sorted(code[g] for g in ident)
        if len(cs) != 3:
            continue
        member_keys[cs[0] * int(N) * int(N) + cs[1] * int(N) + cs[2]] = split
    arr = np.fromiter(member_keys.keys(), dtype=np.int64, count=len(member_keys))
    hit = known & np.isin(keys, arr)
    n_hit = int(hit.sum())
    by_split = Counter(member_keys[int(k)] for k in keys[hit])
    print(f"\ninference_4 space: {n_hit:,} of {n:,} triples ({n_hit / n:.4%}) are "
          f"already in the 010 build")
    for s in SPLITS:
        print(f"    {s}: {by_split.get(s, 0):,}")
    return {"n_space": int(n), "n_in_010": n_hit,
            "share_in_010": float(n_hit / n),
            "by_split": {s: int(by_split.get(s, 0)) for s in SPLITS}}


def head_enrichment(identity_split: dict, identity_tau: dict, base_rate: float) -> dict:
    """How enriched the RANKED HEAD is for triples the model was trained on.

    The base rate over the whole space is tiny, so if the top of the ranking is not
    also tiny the ranking is partly reporting screen membership rather than novel
    structure. top_triples.csv is the same top 500 the ranking figure reports.
    """
    top = pd.read_csv(osp.join(RESULTS_DIR, "top_triples.csv"))
    rows = []
    for r in top.itertuples():
        ident = tuple(sorted((r.gene1, r.gene2, r.gene3)))
        rows.append({"rank": r.rank, "ensemble_mean": r.ensemble_mean,
                     "split_010": identity_split.get(ident, "not in 010"),
                     "measured_tau_010": identity_tau.get(ident)})
    hd = pd.DataFrame(rows)
    out = {"n_top": int(len(hd)), "base_rate": base_rate, "by_k": []}
    for k in (10, 50, 100, 500):
        sub = hd.head(k)
        n_in = int((sub["split_010"] != "not in 010").sum())
        out["by_k"].append({
            "k": k, "n_in_010": n_in, "share": float(n_in / k),
            "enrichment": float((n_in / k) / base_rate) if base_rate > 0 else None,
            "splits": {s: int((sub["split_010"] == s).sum()) for s in SPLITS},
        })
    print("\n=== is the ranked head enriched for triples the model trained on? ===")
    print(f"  base rate over the space: {base_rate:.4%}")
    for row in out["by_k"]:
        print(f"  top {row['k']:>4}: {row['n_in_010']:>3} in 010 "
              f"({row['share']:.1%}), enrichment {row['enrichment']:>8,.0f}x, "
              f"train {row['splits']['train']} val {row['splits']['val']} "
              f"test {row['splits']['test']}")
    # Where a measured value exists, how close is the prediction to it?
    seen = hd[hd["split_010"] != "not in 010"].dropna(subset=["measured_tau_010"])
    if len(seen):
        out["seen_in_head"] = [
            {"rank": int(r.rank), "split": r.split_010,
             "predicted_mean": float(r.ensemble_mean),
             "measured_tau": float(r.measured_tau_010)}
            for r in seen.itertuples()
        ]
        print(f"\n  {len(seen)} of the top 500 carry a measured tau:")
        for r in seen.head(15).itertuples():
            print(f"    rank {int(r.rank):>3} [{r.split_010:>5}] predicted "
                  f"{r.ensemble_mean:+.4f}   measured {r.measured_tau_010:+.4f}")
    return out


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    identity_split, identity_tau, any_pairs, query_pairs, genes_seen = load_010()
    code = {g: i for i, g in enumerate(sorted(genes_seen))}

    strains = pd.read_csv(osp.join(RESULTS_DIR, "panel20_strains.csv"))
    scored_path = osp.join(RESULTS_DIR, "panel20_scored.csv")
    scored = pd.read_csv(scored_path) if osp.exists(scored_path) else None

    rows = []
    for r in strains.itertuples():
        genes = tuple(sorted(r.genotype.split("+")))
        split = identity_split.get(genes)
        pairs = [tuple(sorted(p)) for p in combinations(genes, 2)]
        rows.append({
            "genotype": r.genotype, "name": r.name, "order": r.order,
            "in_010": split is not None,
            "split_010": split or "not in 010",
            "measured_tau_010": identity_tau.get(genes),
            "n_genes_in_010": sum(g in genes_seen for g in genes),
            "n_pairs_in_010": sum(p in any_pairs for p in pairs),
            "n_query_pairs": sum(p in query_pairs for p in pairs),
            "query_pairs": ";".join("+".join(p) for p in pairs if p in query_pairs),
        })
    df = pd.DataFrame(rows)
    if scored is not None:
        df = df.merge(scored[["genotype", "pred_mean", "pred_worst"]], on="genotype",
                      how="left")
    df.to_csv(osp.join(RESULTS_DIR, "panel20_overlap.csv"), index=False)

    print("\n=== panel strains against the 010 build ===")
    cols = ["name", "order", "in_010", "split_010", "n_genes_in_010",
            "n_pairs_in_010", "n_query_pairs"]
    print(df[cols].to_string(index=False))

    tri = df[df["order"] == 3]
    print(f"\ntriples in 010: {int(tri['in_010'].sum())} of {len(tri)}")
    for s in SPLITS:
        k = int((tri["split_010"] == s).sum())
        if k:
            print(f"    {s}: {k}")
    print(f"triples carrying a trained query double: "
          f"{int((tri['n_query_pairs'] > 0).sum())} of {len(tri)}")

    summary = {
        "query_pair_min_recurrence": QUERY_PAIR_MIN_RECURRENCE,
        "n_query_pairs": len(query_pairs),
        "n_010_identities": len(identity_split),
        "panel": {
            "n_strains": int(len(df)),
            "n_in_010": int(df["in_010"].sum()),
            "by_order": {
                str(int(o)): {
                    "n": int((df["order"] == o).sum()),
                    "n_in_010": int(df[df["order"] == o]["in_010"].sum()),
                    "splits": {
                        s: int((df[df["order"] == o]["split_010"] == s).sum())
                        for s in SPLITS
                    },
                    "n_with_query_pair": int(
                        (df[df["order"] == o]["n_query_pairs"] > 0).sum()
                    ),
                }
                for o in sorted(df["order"].unique())
            },
        },
        "space": space_overlap(identity_split, code),
    }
    summary["head"] = head_enrichment(
        identity_split, identity_tau, summary["space"]["share_in_010"]
    )
    with open(osp.join(RESULTS_DIR, "panel20_overlap.json"), "w") as f:
        json.dump(summary, f, indent=2)

    plot(df, summary, osp.join(IMAGES_DIR, "inference_4_panel_overlap"))
    print(f"\nwrote {RESULTS_DIR} and figures to {IMAGES_DIR}")


def _letter(ax, letter):
    ax.text(-0.17, 1.06, letter, transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="bottom", ha="left")


def plot(df, summary, out_stem):
    set_plot_style()
    fig, axes = plt.subplots(
        1, 3, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(62.0))
    )

    # a: the whole space, measured against unmeasured.
    ax = axes[0]
    sp = summary["space"]
    parts = [sp["n_space"] - sp["n_in_010"]] + [sp["by_split"][s] for s in SPLITS]
    labels = ["not in 010", "010 train", "010 val", "010 test"]
    cols = ["0.72", PLOT_PALETTE[0], PLOT_PALETTE[1], PLOT_PALETTE[2]]
    xs = np.arange(len(parts))
    ax.bar(xs, [max(v, 0.6) for v in parts], 0.62, color=cols, edgecolor="black",
           linewidth=0.4, zorder=3)
    for x, v in zip(xs, parts):
        ax.text(x, max(v, 0.6) * 1.35, f"{v:,}", ha="center", va="bottom", fontsize=4.6,
                rotation=90)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=5)
    ax.set_ylabel("Triples")
    ax.set_ylim(0.5, sp["n_space"] * 2000)
    ax.set_title(
        f"The space against the 010 build\n"
        f"{sp['n_in_010']:,} of {sp['n_space']:,} ({sp['share_in_010']:.3%}) "
        f"are already measured",
        fontsize=6, loc="left", pad=3,
    )

    # b: the ranked head is hundreds of times enriched for what the model trained on.
    ax = axes[1]
    hd = summary["head"]
    ks = [r["k"] for r in hd["by_k"]]
    xs = np.arange(len(ks))
    width = 0.34
    for j, (s, col) in enumerate((("train", PLOT_PALETTE[0]),
                                  ("test", PLOT_PALETTE[2]))):
        vals = [r["splits"][s] / r["k"] for r in hd["by_k"]]
        ax.bar(xs + (j - 0.5) * width, vals, width, color=col, edgecolor="black",
               linewidth=0.4, zorder=3, label=f"010 {s}")
    ax.axhline(hd["base_rate"], color="black", linewidth=0.6, linestyle="--", zorder=4)
    for x, r in zip(xs, hd["by_k"]):
        ax.text(x, r["share"] + 0.003, f"{r['enrichment']:,.0f}x", ha="center",
                va="bottom", fontsize=4.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"top {k}" for k in ks])
    ax.set_ylabel("Share already in the 010 build")
    ax.set_ylim(0, max(r["share"] for r in hd["by_k"]) * 1.45)
    ax.set_title(f"The ranked head is enriched for trained triples\n"
                 f"dashed line is the {hd['base_rate']:.4%} base rate",
                 fontsize=6, loc="left", pad=3)
    ax.legend(loc="upper left", frameon=True, fontsize=5, handlelength=1.0,
              labelspacing=0.25, borderpad=0.3)

    # c: where a measured value exists in the head, the prediction overshoots it.
    ax = axes[2]
    seen = pd.DataFrame(hd.get("seen_in_head", []))
    if len(seen):
        colmap = {"train": PLOT_PALETTE[0], "val": PLOT_PALETTE[1],
                  "test": PLOT_PALETTE[2]}
        for s, sub in seen.groupby("split"):
            ax.scatter(sub["measured_tau"], sub["predicted_mean"], s=16,
                       color=colmap[s], edgecolor="black", linewidths=0.4,
                       zorder=4, label=f"010 {s} ({len(sub)})")
        lim = np.array([
            min(seen["measured_tau"].min(), 0) - 0.1,
            max(seen["predicted_mean"].max(), seen["measured_tau"].max()) + 0.15,
        ])
        ax.plot(lim, lim, color="black", linewidth=0.5, linestyle="--", zorder=2)
        ax.axhline(0, color="0.7", linewidth=0.4, zorder=1)
        ax.axvline(0, color="0.7", linewidth=0.4, zorder=1)
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ratio = float((seen["measured_tau"] / seen["predicted_mean"]).mean())
        n_over = int((seen["predicted_mean"] > seen["measured_tau"]).sum())
        ax.set_title(
            f"Where the head has a measured $\\tau$\n"
            f"{n_over} of {len(seen)} over-predicted, {ratio:.2f}$\\times$ on average",
            fontsize=6, loc="left", pad=3,
        )
    ax.set_xlabel("Measured $\\tau$ in the 010 build")
    ax.set_ylabel("Predicted $\\tau$, ensemble mean")
    ax.legend(loc="upper left", frameon=True, fontsize=5, handlelength=1.0,
              labelspacing=0.25, borderpad=0.3)

    for ax in axes:
        for sp_ in ax.spines.values():
            sp_.set_visible(True)
            sp_.set_linewidth(0.5)
            sp_.set_color("black")
        ax.grid(axis="y", which="major", color="0.85", linewidth=0.3, zorder=0)
        ax.set_axisbelow(True)
    for ax, letter in zip(axes, "abc"):
        _letter(ax, letter)

    fig.suptitle(
        "Whether the panel and the inference_4 space were already measured in the "
        "010 build the checkpoints trained on.",
        fontsize=6, y=0.99,
    )
    fig.tight_layout(rect=(0.01, 0, 1, 0.93))
    fig.savefig(f"{out_stem}.png", dpi=300)
    savefig_true_size_svg(fig, f"{out_stem}.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
