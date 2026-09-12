# experiments/025-solid-growth/scripts/additive_baselines_025_panels.py
# [[experiments.025-solid-growth.scripts.additive_baselines_025_panels]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/025-solid-growth/scripts/additive_baselines_025_panels
"""Multi-panel figures and tables for notes-tex/025-additive-baselines.

The single-plot figures of ``additive_baselines_025.py`` served the review of the 010
report. The manuscript wants the same evidence condensed: one figure for the ladder under
both splits, one for every transformer run on the query-pair-disjoint split, and tables
the text can cite without retyping a number. Nothing is refit here. Every panel and every
table cell is read from result files that other committed scripts wrote, plus the per-epoch
validation histories of the transformer runs, fetched once from W&B and cached to results.

Reads
    results/additive_baselines_025.csv                 six baselines x two arms x splits
    results/additive_baselines_025_summary.json        arm R transformer references
    results/additive_baselines_025_arm_val_history.csv jobs 1598 and 1640, per epoch
    results/query_pair_disjoint_splits_025.json.gz     arm Q partition, pair_assignment
    results/pinned_splits_from_010_seed_42.json.gz     arm R partition
    010-kuzmin-tmi/results/query_pair_disjoint_cv.csv  five disjoint folds on the 010 build
    W&B                                                the other disjoint transformer runs

Writes
    $ASSET_IMAGES_DIR/025-solid-growth/additive_baselines_025_fig1_ladders.{svg,png}
    $ASSET_IMAGES_DIR/025-solid-growth/additive_baselines_025_fig2_disjoint_runs.{svg,png}
    results/additive_baselines_025_disjoint_runs_history.csv   (W&B cache; delete to refetch)
    results/additive_baselines_025_panels_summary.json         every number the prose uses
    notes-tex/025-additive-baselines/tables/t1-arms.tex
    notes-tex/025-additive-baselines/tables/t2-heldout.tex
    notes-tex/025-additive-baselines/tables/t3-disjoint-runs.tex
    notes-tex/025-additive-baselines/tables/t4-armq-genes.tex      what arm Q holds out, per part

Panel c of the second figure is a placeholder by design: it draws the arm Q test nulls and
an empty slot per planned replicate, so the layout the finished figure will have is fixed
now and the bars land in it when the runs are scored.

Run from the repo root:
    python experiments/025-solid-growth/scripts/additive_baselines_025_panels.py
"""

import gzip
import json
import os
import os.path as osp
from collections import Counter
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    apply_paper_style,
    mm_to_in,
    panel_label,
    savefig_true_size_svg,
)

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
REPO_ROOT = osp.dirname(EXPERIMENT_ROOT)
# 025 triple index -> its three systematic gene names, the same table the ladder script reads.
RECAP = osp.join(
    DATA_ROOT,
    "data/torchcell/experiments/025-solid-growth/recapitulation/recapitulation_per_triple.csv.gz",
)
QUERY_PAIR_MIN_COUNT = 5

RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "025-solid-growth", "results")
RESULTS_010 = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi", "results")
IMAGES_DIR = osp.join(ASSET_IMAGES_DIR, "025-solid-growth")
TABLES_DIR = osp.join(REPO_ROOT, "notes-tex", "025-additive-baselines", "tables")

WANDB_PROJECT = (
    "zhao-group/torchcell_025-solid-growth_equivariant_cell_graph_transformer"
)
VAL_KEY = "val/gene_interaction/Pearson"
ARM_HISTORY_CSV = "additive_baselines_025_arm_val_history.csv"
DISJOINT_HISTORY_CSV = "additive_baselines_025_disjoint_runs_history.csv"

MODELS = [
    "B0_train_mean",
    "B4_query_pair_only",
    "B3_hierarchical_mean",
    "B1_additive_gene",
    "B2_additive_plus_pair",
    "B5_gene_embedding_mlp",
]
LABELS = {
    "B0_train_mean": "B0 train\nmean",
    "B4_query_pair_only": "B4 query\npair only",
    "B3_hierarchical_mean": "B3 screen\nmean",
    "B1_additive_gene": "B1 additive\nridge",
    "B2_additive_plus_pair": "B2 additive\n+ pair",
    "B5_gene_embedding_mlp": "B5 MLP\n(3 seeds)",
}
SHORT = {
    "B0_train_mean": "B0 train mean",
    "B1_additive_gene": "B1 additive per-gene ridge",
    "B2_additive_plus_pair": "B2 additive plus pair ridge",
    "B3_hierarchical_mean": "B3 screen mean",
    "B4_query_pair_only": "B4 query pair only",
    "B5_gene_embedding_mlp": "B5 embedding MLP, 3 seeds",
}
CGT_010 = {
    "CGT_M01_lzs9pcj3": "CGT M01\n(010 test)",
    "CGT_M02_yv4r30bi": "CGT M02\n(010 test)",
    "CGT_M03_c7671wgj": "CGT M03\n(010 test)",
}
COLOR = {"R": PLOT_PALETTE[0], "Q": PLOT_PALETTE[1]}

# Every transformer run that has trained on the arm Q partition, in submission order. The
# first is the 010 configuration itself (cosine schedule, learnable gene table, mean
# readout). The three IGB runs come from the joint-fitness-head branch (PR #346, unlanded
# at the time of writing): a constant learning rate for 30 epochs, a perturbed-CLS
# readout, and a fixed sequence embedding in place of the learnable table. They are runs
# on the same split, not replicates of job 1640, and the table says so.
DISJOINT_RUNS = [
    {
        "key": "job_1640",
        "job": "GH 1640",
        "run": "327csnlk",
        "config": "cgt_s0_q_kl_004",
        "gene_input": "learnable table",
        "schedule": "cosine, 010 schedule",
        "readout": "mean",
        "budget": "600 epochs, cut at 12 h",
        "history": "arm",
    },
    {
        "key": "igb_2391132",
        "job": "IGB 2391132",
        "run": "s1vx2zgw",
        "config": "cgt_s0_q_kl_emb_017",
        "gene_input": "four-region sequence composite",
        "schedule": "constant $2.5 \\times 10^{-4}$",
        "readout": "perturbed CLS",
        "budget": "30 epochs, finished",
        "history": "wandb",
    },
    {
        "key": "igb_2391133",
        "job": "IGB 2391133",
        "run": "8aa08xx0",
        "config": "cgt_s0_q_kl_calm_020",
        "gene_input": "CaLM codon embedding",
        "schedule": "constant $2.5 \\times 10^{-4}$",
        "readout": "perturbed CLS",
        "budget": "30 epochs, finished",
        "history": "wandb",
    },
    {
        "key": "igb_2391134",
        "job": "IGB 2391134",
        "run": "pmkzwwzw",
        "config": "cgt_s0_q_kl_prot_021",
        "gene_input": "ProtT5 protein embedding",
        "schedule": "constant $2.5 \\times 10^{-4}$",
        "readout": "perturbed CLS",
        "budget": "30 epochs, finished",
        "history": "wandb",
    },
]
RUN_COLORS = [PLOT_PALETTE[1], PLOT_PALETTE[2], PLOT_PALETTE[3], PLOT_PALETTE[4]]
RUN_SHORT = ["GH 1640, table, cosine", "IGB 2391132, composite", "IGB 2391133, CaLM", "IGB 2391134, ProtT5"]

# The replicate set the disjoint comparison still needs: the 010 configuration on arm Q,
# three seeds, each scored on test at its best validation epoch, plus the one surviving
# checkpoint of job 1640. These are the empty slots of Fig 2c.
PLANNED_TEST_SLOTS = ["1640\nepoch 7", "seed 1", "seed 2", "seed 3"]

EKW = {"elinewidth": 0.5, "capthick": 0.5, "capsize": 1.5}
BAR = {"edgecolor": "black", "linewidth": 0.5}


def load_gz(name: str):
    with gzip.open(osp.join(RESULTS_DIR, name), "rt") as f:
        return json.load(f)


def style_metric_axis(ax, top: float) -> None:
    ax.set_ylim(0, top)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which="minor", length=0)
    ax.grid(axis="y", which="both", linewidth=0.3, color="0.85")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)


def fetch_disjoint_histories(arm_history: pd.DataFrame) -> pd.DataFrame:
    """Per-epoch validation Pearson for every run in DISJOINT_RUNS.

    Job 1640 is read from the arm history the ladder script already cached; the IGB runs
    are fetched once and cached beside it. Delete the csv to refetch.
    """
    frames = [
        arm_history[arm_history["arm"] == "Q"][["epoch", VAL_KEY]].assign(key="job_1640")
    ]
    path = osp.join(RESULTS_DIR, DISJOINT_HISTORY_CSV)
    wanted = [r for r in DISJOINT_RUNS if r["history"] == "wandb"]
    if osp.exists(path):
        cached = pd.read_csv(path)
    else:
        import wandb

        api = wandb.Api(timeout=120)
        parts = []
        for r in wanted:
            run = api.run(f"{WANDB_PROJECT}/{r['run']}")
            hist = pd.DataFrame(run.scan_history(keys=["epoch", VAL_KEY])).dropna()
            hist["key"] = r["key"]
            hist["run"] = r["run"]
            parts.append(hist)
        cached = pd.concat(parts, ignore_index=True)
        cached.to_csv(path, index=False)
        print(f"wrote {path}")
    missing = {r["key"] for r in wanted} - set(cached["key"])
    if missing:
        raise SystemExit(f"cache {path} lacks runs {sorted(missing)}; delete it to refetch")
    frames.append(cached[["epoch", VAL_KEY, "key"]])
    return pd.concat(frames, ignore_index=True)


def run_stats(history: pd.DataFrame, key: str) -> dict[str, object]:
    h = history[history["key"] == key].sort_values("epoch").reset_index(drop=True)
    i = int(h[VAL_KEY].idxmax())
    return {
        "n_epochs_logged": int(h["epoch"].nunique()),
        "best_epoch": int(h.loc[i, "epoch"]),
        "val_best": float(h.loc[i, VAL_KEY]),
        "last_epoch": int(h["epoch"].max()),
        "val_last": float(h.loc[h["epoch"].idxmax(), VAL_KEY]),
    }


def arm_sizes() -> dict[str, dict[str, object]]:
    q = load_gz("query_pair_disjoint_splits_025.json.gz")
    r = load_gz("pinned_splits_from_010_seed_42.json.gz")["pinned"]
    pairs = pd.Series(q["pair_assignment"]).value_counts()
    return {
        "R": {
            "records": {s: len(r[s]) for s in ("train", "val", "test")},
            "pairs": None,
            "n_recurring_pairs": int(len(q["pair_assignment"])),
        },
        "Q": {
            "records": {s: len(q["splits"][s]) for s in ("train", "val", "test")},
            "pairs": {s: int(pairs[s]) for s in ("train", "val", "test")},
            "n_recurring_pairs": int(len(q["pair_assignment"])),
        },
    }


def arm_q_gene_semantics() -> dict[str, object]:
    """What arm Q holds out: the query pair, and how much of each record's genes it has seen.

    Regroups the S0 records by query pair with the rule ``subset_definitions.query_pair_split``
    used (the most frequent recurring pair a record carries), checks the recurring pairs are
    arm Q's 420, and then counts, for the validation and test parts, the distinct array
    genes, how many of them the training part contains, how many query-pair genes it
    contains, and how many records have all three genes in training.
    """
    q = load_gz("query_pair_disjoint_splits_025.json.gz")
    pair_assignment: dict[str, str] = q["pair_assignment"]
    subset = np.array(sorted(load_gz("subset_S0_indices.json.gz")), dtype=np.int64)
    recap = pd.read_csv(RECAP, usecols=["idx_025", "gene_a", "gene_b", "gene_c"]).set_index("idx_025")
    genes = recap.loc[subset, ["gene_a", "gene_b", "gene_c"]].to_numpy()

    counts: Counter = Counter()
    for trip in genes:
        for p in combinations(sorted(trip), 2):
            counts[p] += 1
    recurring = {p for p, c in counts.items() if c >= QUERY_PAIR_MIN_COUNT}
    if {"+".join(p) for p in recurring} != set(pair_assignment):
        raise SystemExit("recurring pairs do not match arm Q's pair_assignment")

    query_pair = np.empty(len(subset), dtype=object)
    array_gene = np.empty(len(subset), dtype=object)
    for i, trip in enumerate(genes):
        rec = [p for p in combinations(sorted(trip), 2) if p in recurring]
        best = max(rec, key=lambda p: counts[p])
        query_pair[i] = best
        (array_gene[i],) = set(trip) - set(best)

    row_of = {int(r): i for i, r in enumerate(subset)}
    parts = {s: np.array([row_of[int(r)] for r in q["splits"][s]]) for s in ("train", "val", "test")}
    train_genes = set(genes[parts["train"]].ravel())
    out: dict[str, object] = {
        "n_distinct_pairs": len(counts),
        "n_recurring_pairs": len(recurring),
        "n_recurring_pair_instances": int(sum(counts[p] for p in recurring)),
        "parts": {},
    }
    for s in ("val", "test"):
        rows = parts[s]
        pairs = {query_pair[i] for i in rows}
        arr = {array_gene[i] for i in rows}
        qp_genes = {g for p in pairs for g in p}
        all_seen = np.array([all(g in train_genes for g in genes[i]) for i in rows])
        out["parts"][s] = {
            "records": int(rows.size),
            "query_pairs": len(pairs),
            "query_pairs_in_train": sum(1 for p in pairs if pair_assignment["+".join(p)] == "train"),
            "distinct_array_genes": len(arr),
            "array_genes_in_train": sum(1 for g in arr if g in train_genes),
            "query_pair_genes": len(qp_genes),
            "query_pair_genes_in_train": sum(1 for g in qp_genes if g in train_genes),
            "records_all_three_genes_in_train": int(all_seen.sum()),
            "frac_records_all_three_genes_in_train": float(all_seen.mean()),
        }
    return out


def test_table(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    test = df[df["split"] == "test"]
    return {
        arm: test[test["arm"] == arm].groupby("model")["pearson"].agg(["mean", "std", "count"])
        for arm in ("R", "Q")
    }


def val_nulls(df: pd.DataFrame, arm: str) -> dict[str, float]:
    val = df[(df["split"] == "val") & (df["arm"] == arm)]
    return {m: float(val[val["model"] == m]["pearson"].mean()) for m in MODELS}


# --- figure 1: the ladder under both splits -----------------------------------------


def ladder(ax, agg: pd.DataFrame, arm: str, extra: list[tuple[str, float, bool]], title: str):
    """Bars for the six baselines then the transformer entries of one arm.

    ``extra`` is (tick label, value, hatched) per transformer bar; hatched marks a
    validation maximum with no test score.
    """
    cats = [LABELS[m] for m in MODELS] + [e[0] for e in extra]
    x = np.arange(len(cats))
    vals = [agg.loc[m, "mean"] for m in MODELS]
    err = [0.0 if np.isnan(agg.loc[m, "std"]) else agg.loc[m, "std"] for m in MODELS]
    ax.bar(x[: len(MODELS)], vals, 0.7, yerr=err, error_kw=EKW, color=COLOR[arm], **BAR)
    for k, (_, v, hatched) in enumerate(extra):
        ax.bar(x[len(MODELS) + k], v, 0.7, color=COLOR[arm], hatch="///" if hatched else None, **BAR)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=45, ha="right", fontsize=5)
    ax.set_ylabel("Held-out Pearson r")
    ax.set_title(title, fontsize=6, pad=3)
    style_metric_axis(ax, 0.55)


def fold_spread(ax, cv: pd.DataFrame, agg_q: pd.DataFrame) -> dict[str, object]:
    """Five disjoint folds on the 010 build per model, with the arm Q single split marked."""
    models = ["B3_hierarchical_mean", "B1_additive_gene", "B2_additive_plus_pair", "B5_gene_embedding_mlp"]
    # B5 has three seeds per fold; average them so each fold is one point.
    per_fold = cv.groupby(["model", "fold"])["pearson"].mean().reset_index()
    out = {}
    x = np.arange(len(models))
    rng = np.random.default_rng(0)
    for i, m in enumerate(models):
        pts = per_fold[per_fold["model"] == m].sort_values("fold")["pearson"].to_numpy()
        ax.bar(i, pts.mean(), 0.7, yerr=pts.std(ddof=1), error_kw=EKW, color=PLOT_PALETTE[5], **BAR)
        jitter = rng.uniform(-0.18, 0.18, size=pts.size)
        ax.plot(i + jitter, pts, linestyle="none", marker="o", markersize=2.2,
                markerfacecolor="white", markeredgecolor="black", markeredgewidth=0.5, zorder=3)
        ax.plot(i, agg_q.loc[m, "mean"], linestyle="none", marker="D", markersize=3.2,
                markerfacecolor=COLOR["Q"], markeredgecolor="black", markeredgewidth=0.5, zorder=4)
        out[m] = {
            "fold_mean": float(pts.mean()),
            "fold_sd": float(pts.std(ddof=1)),
            "fold_min": float(pts.min()),
            "fold_max": float(pts.max()),
            "arm_q_test": float(agg_q.loc[m, "mean"]),
        }
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in models], rotation=45, ha="right", fontsize=5)
    ax.set_ylabel("Test Pearson r, held-out screens")
    ax.set_title("disjoint null across held-out screen sets", fontsize=6, pad=3)
    style_metric_axis(ax, 0.30)
    handles = [
        Patch(facecolor=PLOT_PALETTE[5], label="5 folds on the 010 build, mean ± sd", **BAR),
        Line2D([], [], linestyle="none", marker="o", markersize=2.2, markerfacecolor="white",
               markeredgecolor="black", markeredgewidth=0.5, label="one fold"),
        Line2D([], [], linestyle="none", marker="D", markersize=3.2, markerfacecolor=COLOR["Q"],
               markeredgecolor="black", markeredgewidth=0.5, label="arm Q, the one split the transformer trains on"),
    ]
    ax.legend(handles=handles, loc="upper center", ncol=1, handlelength=1.4, labelspacing=0.3, borderpad=0.4)
    return out


def val_curves(ax, df: pd.DataFrame, arm_history: pd.DataFrame, summary: dict) -> None:
    jobs = {"R": "GH 1598", "Q": "GH 1640"}
    for arm in ("R", "Q"):
        h = arm_history[arm_history["arm"] == arm].sort_values("epoch")
        ax.plot(h["epoch"], h[VAL_KEY], color=COLOR[arm], linewidth=0.9, label=f"CGT arm {arm}, {jobs[arm]}")
        ax.axhline(val_nulls(df, arm)["B1_additive_gene"], color=COLOR[arm], linewidth=0.7,
                   linestyle=(0, (4, 2)), label=f"B1 additive ridge, arm {arm} (val)")
        i = int(h[VAL_KEY].idxmax())
        ax.plot(h.loc[i, "epoch"], h.loc[i, VAL_KEY], marker="o", markersize=2.5, markerfacecolor="white",
                markeredgecolor=COLOR[arm], markeredgewidth=0.7, linestyle="none")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Pearson r")
    ax.set_xlim(left=0)
    ax.set_title("the 010 configuration under each split", fontsize=6, pad=3)
    style_metric_axis(ax, 0.55)
    # The band between the two arms is empty at every epoch.
    ax.legend(loc="center right", handlelength=1.6, labelspacing=0.3, borderpad=0.4)


def figure_1(df, agg, summary, arm_history, cv) -> dict[str, object]:
    apply_paper_style()
    fig, axes = plt.subplots(2, 2, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(128.0)))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    cgt010 = summary["transformer"]["R"]["010_checkpoints_test_pearson"]
    ref_r = summary["transformer"]["R"]["job_1598_replication"]
    ref_q = summary["transformer"]["Q"]["job_1640_disjoint"]
    ladder(ax_a, agg["R"], "R",
           [(CGT_010[k], cgt010[k], False) for k in CGT_010] + [("CGT GH 1598\n(val max)", ref_r["val_pearson_best_epoch"], True)],
           "arm R, random over records")
    handles = [
        Patch(facecolor="white", label="test score", **BAR),
        Patch(facecolor="white", hatch="///", label="validation max over epochs, no test score", **BAR),
    ]
    ax_a.legend(handles=handles, loc="upper left", handlelength=1.4, labelspacing=0.3, borderpad=0.4)
    ladder(ax_b, agg["Q"], "Q", [("CGT GH 1640\n(val max)", ref_q["val_pearson_best_epoch"], True)],
           "arm Q, query-pair disjoint")
    val_curves(ax_c, df, arm_history, summary)
    spread = fold_spread(ax_d, cv, agg["Q"])

    # rect leaves the top 4 percent free: the panel letters sit 12 pt above each axes box
    # and were clipped off the top row without it.
    fig.tight_layout(pad=0.4, w_pad=1.2, h_pad=1.6, rect=(0, 0, 1, 0.965))
    for ax, letter in zip((ax_a, ax_b, ax_c, ax_d), "abcd"):
        panel_label(ax, letter)
    stem = osp.join(IMAGES_DIR, "additive_baselines_025_fig1_ladders")
    fig.savefig(stem + ".png", dpi=300)
    savefig_true_size_svg(fig, stem + ".svg")
    print(f"wrote {stem}.svg")
    plt.close(fig)
    return spread


# --- figure 2: every transformer run on the disjoint split ---------------------------


def figure_2(df, agg, history: pd.DataFrame, stats: dict[str, dict]) -> None:
    apply_paper_style()
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(62.0)))
    nulls_val = val_nulls(df, "Q")
    b1_val, b5_val = nulls_val["B1_additive_gene"], nulls_val["B5_gene_embedding_mlp"]
    b1_test, b5_test = agg["Q"].loc["B1_additive_gene", "mean"], agg["Q"].loc["B5_gene_embedding_mlp", "mean"]
    top = 0.10 + max(stats[r["key"]]["val_best"] for r in DISJOINT_RUNS)
    top = float(np.ceil(top * 10) / 10)

    # a) validation per epoch
    for r, c, short in zip(DISJOINT_RUNS, RUN_COLORS, RUN_SHORT):
        h = history[history["key"] == r["key"]].sort_values("epoch")
        ax_a.plot(h["epoch"], h[VAL_KEY], color=c, linewidth=0.9, label=short)
        i = int(h[VAL_KEY].idxmax())
        ax_a.plot(h.loc[i, "epoch"], h.loc[i, VAL_KEY], marker="o", markersize=2.5, markerfacecolor="white",
                  markeredgecolor=c, markeredgewidth=0.7, linestyle="none")
    ax_a.axhline(b1_val, color="black", linewidth=0.7, linestyle=(0, (4, 2)), label="B1 additive ridge (val)")
    ax_a.axhline(b5_val, color=PLOT_PALETTE[5], linewidth=0.7, linestyle=(0, (1, 1.5)), label="B5 MLP, 3 seeds (val)")
    ax_a.set_xlabel("Epoch")
    ax_a.set_ylabel("Validation Pearson r")
    ax_a.set_xlim(left=0)
    ax_a.set_title("arm Q, every transformer run", fontsize=6, pad=3)
    style_metric_axis(ax_a, top)
    ax_a.yaxis.set_major_locator(MultipleLocator(0.1))
    ax_a.yaxis.set_minor_locator(MultipleLocator(0.05))
    # Legend in the band above every curve's maximum, which top leaves clear.
    ax_a.legend(loc="upper center", ncol=2, handlelength=1.6, labelspacing=0.3, borderpad=0.4, columnspacing=0.8)

    # b) best and last epoch per run, as a dumbbell
    x = np.arange(len(DISJOINT_RUNS))
    for i, (r, c) in enumerate(zip(DISJOINT_RUNS, RUN_COLORS)):
        s = stats[r["key"]]
        ax_b.plot([i, i], [s["val_last"], s["val_best"]], color=c, linewidth=1.2, zorder=2)
        ax_b.plot(i, s["val_best"], marker="o", markersize=4, markerfacecolor="white", markeredgecolor=c,
                  markeredgewidth=0.8, linestyle="none", zorder=3)
        ax_b.plot(i, s["val_last"], marker="o", markersize=4, markerfacecolor=c, markeredgecolor="black",
                  markeredgewidth=0.5, linestyle="none", zorder=3)
    ax_b.axhline(b1_val, color="black", linewidth=0.7, linestyle=(0, (4, 2)))
    ax_b.axhline(b5_val, color=PLOT_PALETTE[5], linewidth=0.7, linestyle=(0, (1, 1.5)))
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([s.replace(", ", "\n", 1) for s in RUN_SHORT], fontsize=5)
    ax_b.set_xlim(-0.6, len(DISJOINT_RUNS) - 0.4)
    ax_b.set_ylabel("Validation Pearson r")
    ax_b.set_title("best epoch (open) to last epoch (filled)", fontsize=6, pad=3)
    style_metric_axis(ax_b, top)
    ax_b.yaxis.set_major_locator(MultipleLocator(0.1))
    ax_b.yaxis.set_minor_locator(MultipleLocator(0.05))
    handles = [
        Line2D([], [], linestyle="none", marker="o", markersize=4, markerfacecolor="white", markeredgecolor="black",
               markeredgewidth=0.8, label="best validation epoch"),
        Line2D([], [], linestyle="none", marker="o", markersize=4, markerfacecolor="black", markeredgecolor="black",
               markeredgewidth=0.5, label="last logged epoch"),
    ]
    ax_b.legend(handles=handles, loc="upper center", handlelength=1.2, labelspacing=0.3, borderpad=0.4)

    # c) placeholder: the test comparison that has not been run
    ax_c.axhline(b1_test, color="black", linewidth=0.7, linestyle=(0, (4, 2)), label=f"B1 additive ridge (test) {b1_test:.3f}")
    ax_c.axhline(b5_test, color=PLOT_PALETTE[5], linewidth=0.7, linestyle=(0, (1, 1.5)), label=f"B5 MLP, 3 seeds (test) {b5_test:.3f}")
    ax_c.set_xticks(np.arange(len(PLANNED_TEST_SLOTS)))
    ax_c.set_xticklabels(PLANNED_TEST_SLOTS, fontsize=5)
    ax_c.set_xlim(-0.6, len(PLANNED_TEST_SLOTS) - 0.4)
    ax_c.set_ylabel("Test Pearson r")
    ax_c.set_title("arm Q test scores: not yet run", fontsize=6, pad=3)
    style_metric_axis(ax_c, top)
    ax_c.yaxis.set_major_locator(MultipleLocator(0.1))
    ax_c.yaxis.set_minor_locator(MultipleLocator(0.05))
    # The note sits in the band between the two null lines and the legend, so no line
    # crosses it; the white patch keeps the gridlines off the letters.
    ax_c.text(0.5, b1_test + 0.25 * (top - b1_test), "placeholder\ncgt_s0_q_kl_004, three seeds,\nscored on test at best val epoch;\njob 1640 epoch 7 checkpoint",
              transform=ax_c.get_yaxis_transform(), ha="center", va="center", fontsize=5.5, color=PLOT_PALETTE[5],
              bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0})
    ax_c.legend(loc="upper center", handlelength=1.6, labelspacing=0.3, borderpad=0.4)

    fig.tight_layout(pad=0.4, w_pad=1.2, rect=(0, 0, 1, 0.93))
    for ax, letter in zip((ax_a, ax_b, ax_c), "abc"):
        panel_label(ax, letter)
    stem = osp.join(IMAGES_DIR, "additive_baselines_025_fig2_disjoint_runs")
    fig.savefig(stem + ".png", dpi=300)
    savefig_true_size_svg(fig, stem + ".svg")
    print(f"wrote {stem}.svg")
    plt.close(fig)


# --- tables -----------------------------------------------------------------------


def write_table(name: str, body: str) -> None:
    os.makedirs(TABLES_DIR, exist_ok=True)
    path = osp.join(TABLES_DIR, name)
    with open(path, "w") as f:
        f.write(
            "%% GENERATED by experiments/025-solid-growth/scripts/additive_baselines_025_panels.py\n"
            "%% SOURCE: the result files named in that script's docstring. Do not edit by hand.\n"
        )
        f.write(body)
    print(f"wrote {path}")


def fmt(v: float) -> str:
    return f"{v:.3f}"


def fmt_int(v: int) -> str:
    return f"{v:,}".replace(",", "{,}")


def table_arms(sizes: dict) -> None:
    r, q = sizes["R"], sizes["Q"]
    rows = [
        ("Held-out unit", "record", "query pair"),
        ("Recurring query pairs", f"{r['n_recurring_pairs']}, shared by every part", f"{q['n_recurring_pairs']}, no pair in two parts"),
        ("Query pairs, train / val / test", "", " / ".join(str(q["pairs"][s]) for s in ("train", "val", "test"))),
        ("Records, train / val / test", " / ".join(fmt_int(r["records"][s]) for s in ("train", "val", "test")),
         " / ".join(fmt_int(q["records"][s]) for s in ("train", "val", "test"))),
        ("Split artifact", r"\file{pinned_splits_from_010_seed_42.json.gz}", r"\file{query_pair_disjoint_splits_025.json.gz}"),
        ("Transformer run", "GH 1598", "GH 1640, and Table~\\ref{tab:disjointruns}"),
    ]
    body = ["\\begin{tabular}{lll}", "\\toprule", " & Arm R & Arm Q \\\\", "\\midrule"]
    body += [f"{a} & {b} & {c} \\\\" for a, b, c in rows]
    body += ["\\bottomrule", "\\end{tabular}", ""]
    write_table("t1-arms.tex", "\n".join(body))


def table_heldout(agg: dict, cgt010: dict, ref_r: dict, ref_q: dict, d010: pd.DataFrame) -> None:
    t010 = d010[d010["split"] == "test"].groupby("model")["pearson"].agg(["mean", "std"])
    order = ["B0_train_mean", "B4_query_pair_only", "B3_hierarchical_mean", "B1_additive_gene",
             "B2_additive_plus_pair", "B5_gene_embedding_mlp"]

    def cell(a: pd.DataFrame, m: str) -> str:
        if a.loc[m, "count"] > 1:
            return f"${fmt(a.loc[m, 'mean'])} \\pm {fmt(a.loc[m, 'std'])}$"
        return fmt(a.loc[m, "mean"])

    def cell010(m: str) -> str:
        if m == "B5_gene_embedding_mlp":
            return f"${fmt(t010.loc[m, 'mean'])} \\pm {fmt(t010.loc[m, 'std'])}$"
        return fmt(t010.loc[m, "mean"])

    body = ["\\begin{tabular}{lrrr}", "\\toprule",
            " & \\multicolumn{2}{c}{Arm R, random over records} & Arm Q, query-pair disjoint \\\\",
            "Model & 010 build & 025 build & 025 build \\\\", "\\midrule"]
    for m in order:
        body.append(f"{SHORT[m]} & {cell010(m)} & {cell(agg['R'], m)} & {cell(agg['Q'], m)} \\\\")
    lo, hi = min(cgt010.values()), max(cgt010.values())
    body.append(f"CGT, three 010 checkpoints, test & {fmt(lo)} to {fmt(hi)} & & \\\\")
    body.append(f"CGT GH 1598, val max, epoch {ref_r['best_epoch']} of {ref_r['n_epochs_logged']} & & {fmt(ref_r['val_pearson_best_epoch'])} & \\\\")
    body.append(f"CGT GH 1640, val max, epoch {ref_q['best_epoch']} of {ref_q['n_epochs_logged']} & & & {fmt(ref_q['val_pearson_best_epoch'])} \\\\")
    body += ["\\bottomrule", "\\end{tabular}", ""]
    write_table("t2-heldout.tex", "\n".join(body))


def table_armq_genes(sem: dict) -> None:
    v, t = sem["parts"]["val"], sem["parts"]["test"]

    def seen(p: dict, k: str, tot: str) -> str:
        return f"{p[k]} of {p[tot]}"

    rows = [
        ("Records", fmt_int(v["records"]), fmt_int(t["records"])),
        ("Query pairs held out", str(v["query_pairs"]), str(t["query_pairs"])),
        ("Query pairs also in training", str(v["query_pairs_in_train"]), str(t["query_pairs_in_train"])),
        ("Distinct array genes", fmt_int(v["distinct_array_genes"]), fmt_int(t["distinct_array_genes"])),
        ("Array genes present in training", seen(v, "array_genes_in_train", "distinct_array_genes"),
         seen(t, "array_genes_in_train", "distinct_array_genes")),
        ("Query-pair genes present in training", seen(v, "query_pair_genes_in_train", "query_pair_genes"),
         seen(t, "query_pair_genes_in_train", "query_pair_genes")),
        ("Records with all three genes in training",
         f"{100 * v['frac_records_all_three_genes_in_train']:.1f}\\%",
         f"{100 * t['frac_records_all_three_genes_in_train']:.1f}\\%"),
    ]
    body = ["\\begin{tabular}{lrr}", "\\toprule", " & Validation & Test \\\\", "\\midrule"]
    body += [f"{a} & {b} & {c} \\\\" for a, b, c in rows]
    body += ["\\bottomrule", "\\end{tabular}", ""]
    write_table("t4-armq-genes.tex", "\n".join(body))


def table_disjoint_runs(stats: dict) -> None:
    # Fixed widths on the two free-text columns, so the row fits the 182 mm text block.
    body = ["\\begin{tabular}{ll>{\\raggedright\\arraybackslash}p{28mm}>{\\raggedright\\arraybackslash}p{38mm}rrr}", "\\toprule",
            "Run & Config & Gene input & Schedule, readout & Epochs & Val max (epoch) & Val last \\\\", "\\midrule"]
    for r in DISJOINT_RUNS:
        s = stats[r["key"]]
        config = r["config"].replace("_", "\\_")
        body.append(
            f"{r['job']} & \\texttt{{{config}}} & {r['gene_input']} & "
            f"{r['schedule']}, {r['readout']} & {s['n_epochs_logged']} & "
            f"{fmt(s['val_best'])} ({s['best_epoch']}) & {fmt(s['val_last'])} \\\\"
        )
    body += ["\\bottomrule", "\\end{tabular}", ""]
    write_table("t3-disjoint-runs.tex", "\n".join(body))


def main() -> None:
    os.makedirs(IMAGES_DIR, exist_ok=True)
    df = pd.read_csv(osp.join(RESULTS_DIR, "additive_baselines_025.csv"))
    with open(osp.join(RESULTS_DIR, "additive_baselines_025_summary.json")) as f:
        summary = json.load(f)
    arm_history = pd.read_csv(osp.join(RESULTS_DIR, ARM_HISTORY_CSV))
    cv = pd.read_csv(osp.join(RESULTS_010, "query_pair_disjoint_cv.csv"))
    d010 = pd.read_csv(osp.join(RESULTS_010, "additive_baseline_gene_interaction.csv"))
    agg = test_table(df)

    history = fetch_disjoint_histories(arm_history)
    stats = {r["key"]: run_stats(history, r["key"]) for r in DISJOINT_RUNS}
    for r in DISJOINT_RUNS:
        print(f"{r['job']:<12s} {r['config']:<22s} {stats[r['key']]}")

    spread = figure_1(df, agg, summary, arm_history, cv)
    figure_2(df, agg, history, stats)

    sizes = arm_sizes()
    cgt010 = summary["transformer"]["R"]["010_checkpoints_test_pearson"]
    ref_r = summary["transformer"]["R"]["job_1598_replication"]
    ref_q = summary["transformer"]["Q"]["job_1640_disjoint"]
    table_arms(sizes)
    table_heldout(agg, cgt010, ref_r, ref_q, d010)
    table_disjoint_runs(stats)
    sem = arm_q_gene_semantics()
    table_armq_genes(sem)
    print(json.dumps(sem, indent=1))

    out = {
        "arms": sizes,
        "arm_q_gene_semantics": sem,
        "test_pearson": {
            arm: {m: {"mean": float(a.loc[m, "mean"]), "sd": (None if np.isnan(a.loc[m, "std"]) else float(a.loc[m, "std"]))}
                  for m in MODELS}
            for arm, a in agg.items()
        },
        "val_nulls": {arm: val_nulls(df, arm) for arm in ("R", "Q")},
        "disjoint_fold_spread_010": spread,
        "disjoint_runs": [dict(r, **stats[r["key"]]) for r in DISJOINT_RUNS],
        "note": "validation maxima are upward-biased order statistics; no disjoint run has a test score",
    }
    path = osp.join(RESULTS_DIR, "additive_baselines_025_panels_summary.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
