# experiments/025-solid-growth/scripts/additive_baselines_025.py
# [[experiments.025-solid-growth.scripts.additive_baselines_025]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/025-solid-growth/scripts/additive_baselines_025
"""The 010 additive-null ladder, refit on the 025 build for both transformer arms.

The transformer arms of experiment 025 train on the 376,732 trigenic records (subset S0)
under two splits: R, 010's random-over-records partition carried across by genotype
identity (job 1598), and Q, a query-pair-disjoint partition in which no Kuzmin query
double appears in more than one part (job 1609). Every baseline here is fit on exactly
the record pool and split those runs read, from the same committed index artifacts the
trainer loads, so a transformer number from either arm sits next to a null on the same
data.

The six baselines are the ones of
``experiments/010-kuzmin-tmi/scripts/additive_baseline_gene_interaction.py`` and are
imported from it rather than copied: B0 train mean, B1 per-gene ridge, B2 ridge on genes
plus recurring pairs, B3 hierarchical empirical mean, B4 recurring pairs only, B5 an
embedding-sum MLP on B1's feature space over three seeds. Ridge alpha and B5 early
stopping select on validation; test is reported.

Inputs (read exactly as the trainer reads them):
    results/subset_S0_indices.json.gz                the S0 record pool
    results/pinned_splits_from_010_seed_42.json.gz   arm R split, key ``pinned``
    results/query_pair_disjoint_splits_025.json.gz   arm Q split, key ``splits``
    $DATA_ROOT/.../025-solid-growth/001-full-build/processed/label_df.parquet
    $DATA_ROOT/.../025-solid-growth/recapitulation/recapitulation_per_triple.csv.gz
        (025 triple index -> its three systematic gene names)

Transformer reference rows: the three 010 checkpoints' test metrics from the 010 CSV,
and job 1598's validation Pearson at its best epoch read from the run's wandb history
(no test evaluation of that checkpoint exists yet). Arm Q has no transformer number
until job 1609 finishes; the summary carries a placeholder, not a number.

Arm R on 025 against the same model on the 010 build is the end-to-end check that the
025 build and the pinned split reproduce 010: same labels, same records, same partition
should give the same ridge to numerical precision.

Run from the repo root (CPU is enough; the MLP is small):
    CUDA_VISIBLE_DEVICES="" python experiments/025-solid-growth/scripts/additive_baselines_025.py
    python experiments/025-solid-growth/scripts/additive_baselines_025.py --plot-only
"""

import argparse
import gzip
import importlib.util
import json
import os
import os.path as osp
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from dotenv import load_dotenv
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MultipleLocator

from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    apply_paper_style,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]

BUILD = osp.join(
    DATA_ROOT, "data/torchcell/experiments/025-solid-growth/001-full-build"
)
RECAP = osp.join(
    DATA_ROOT,
    "data/torchcell/experiments/025-solid-growth/recapitulation/recapitulation_per_triple.csv.gz",
)
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "025-solid-growth", "results")
RESULTS_010 = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi", "results")
IMAGES_DIR = osp.join(ASSET_IMAGES_DIR, "025-solid-growth")

# The shared baseline machinery lives with the 010 experiment; import it from there.
_spec = importlib.util.spec_from_file_location(
    "additive_baseline_gene_interaction",
    osp.join(
        EXPERIMENT_ROOT, "010-kuzmin-tmi/scripts/additive_baseline_gene_interaction.py"
    ),
)
assert _spec is not None and _spec.loader is not None
b010 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b010)

ARMS = {
    # arm -> (split artifact, key holding the three index lists, description)
    "R": ("pinned_splits_from_010_seed_42.json.gz", "pinned", "random over records"),
    "Q": ("query_pair_disjoint_splits_025.json.gz", "splits", "query-pair disjoint"),
}
SUBSET = "subset_S0_indices.json.gz"
MLP_SEEDS = [0, 1, 2]

# Job 1598 = cgt_s0_r_kl_000, the R-arm replication; rank-0 run of the DDP job.
WANDB_PROJECT = (
    "zhao-group/torchcell_025-solid-growth_equivariant_cell_graph_transformer"
)
JOB_1598_RUN = "0yw7moue"
JOB_1609 = "pending job 1609 (cgt_s0_q_kl_004, starts 2026-09-09 23:00)"

LABELS = {
    "B0_train_mean": "Train mean",
    "B4_query_pair_only": "Query pair\nonly",
    "B3_hierarchical_mean": "Hierarchical\nempirical mean",
    "B1_additive_gene": "Additive\n(per-gene ridge)",
    "B2_additive_plus_pair": "Additive\n+ gene-pair ridge",
    "B5_gene_embedding_mlp": "Nonlinear MLP\n(same features)",
}
CGT_010 = {
    "CGT_M01_lzs9pcj3": "CGT 010 M01",
    "CGT_M02_yv4r30bi": "CGT 010 M02",
    "CGT_M03_c7671wgj": "CGT 010 M03",
}


def load_gz(name: str):
    with gzip.open(osp.join(RESULTS_DIR, name), "rt") as f:
        return json.load(f)


def load_records() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """(record_ids sorted, y, row_genes (n,3) of gene column ids, gene_names)."""
    subset = np.array(sorted(load_gz(SUBSET)), dtype=np.int64)
    label_df = pd.read_parquet(
        osp.join(BUILD, "processed", "label_df.parquet"),
        columns=["index", "gene_interaction"],
    )
    label_df = label_df.set_index("index").loc[subset]
    y = label_df["gene_interaction"].to_numpy(dtype=np.float64)
    assert np.isfinite(y).all(), "a subset record carries no gene_interaction label"

    recap = pd.read_csv(RECAP, usecols=["idx_025", "gene_a", "gene_b", "gene_c"])
    recap = recap.set_index("idx_025")
    missing = np.setdiff1d(subset, recap.index.to_numpy())
    assert missing.size == 0, (
        f"{missing.size} subset records absent from the recap table"
    )
    genes3 = recap.loc[subset, ["gene_a", "gene_b", "gene_c"]].to_numpy()
    gene_names = sorted(set(genes3.ravel().tolist()))
    col = {g: j for j, g in enumerate(gene_names)}
    row_genes = np.vectorize(col.__getitem__)(genes3).astype(np.int32)
    assert (
        np.sort(row_genes, axis=1)[:, :-1] != np.sort(row_genes, axis=1)[:, 1:]
    ).all()
    return subset, y, row_genes, gene_names


def split_rows(record_ids: np.ndarray, arm: str) -> dict[str, np.ndarray]:
    fname, key, _ = ARMS[arm]
    parts = load_gz(fname)[key]
    id_to_row = {int(r): i for i, r in enumerate(record_ids)}
    out = {}
    for name in ("train", "val", "test"):
        rows = [id_to_row[int(r)] for r in parts[name] if int(r) in id_to_row]
        assert len(rows) == len(parts[name]), f"{arm}/{name}: pinned index outside S0"
        out[name] = np.array(sorted(rows), dtype=np.int64)
    n = sum(v.size for v in out.values())
    assert n == record_ids.size, f"{arm}: splits cover {n} of {record_ids.size} records"
    return out


def fit_arm(
    arm: str,
    record_ids: np.ndarray,
    y: np.ndarray,
    row_genes: np.ndarray,
    gene_names: list[str],
) -> list[dict[str, object]]:
    splits = split_rows(record_ids, arm)
    tr, va, te = splits["train"], splits["val"], splits["test"]
    n_genes = len(gene_names)
    print(
        f"\n=== arm {arm} ({ARMS[arm][2]}): train {tr.size} val {va.size} test {te.size}"
    )
    print(
        f"label sd train {y[tr].std(ddof=0):.6f} val {y[va].std(ddof=0):.6f} "
        f"test {y[te].std(ddof=0):.6f}"
    )

    pairs = b010.pair_keys(row_genes)
    train_pairs, train_counts = np.unique(pairs[tr].reshape(-1), return_counts=True)
    vocab = {int(p): j for j, p in enumerate(train_pairs[train_counts >= 5])}
    xg = b010.gene_matrix(row_genes, n_genes)
    xp = b010.pair_matrix(pairs, vocab)
    xgp = sp.hstack([xg, xp]).tocsr()
    print(f"B2 design: {n_genes} gene + {len(vocab)} recurring-pair columns")
    for name, idx in (("train", tr), ("val", va), ("test", te)):
        covered = np.asarray(xp[idx].sum(axis=1)).ravel() > 0
        print(
            f"  {name}: records carrying a recurring train pair = {covered.mean():.3%}"
        )

    rows: list[dict[str, object]] = []

    mean_pred = np.full(y.size, y[tr].mean())
    for name, idx in (("val", va), ("test", te)):
        rows.append(
            {"arm": arm, "model": "B0_train_mean", "alpha": None, "split": name}
            | b010.score(y[idx], mean_pred[idx])
        )

    for tag, design in (
        ("B1_additive_gene", xg),
        ("B2_additive_plus_pair", xgp),
        ("B4_query_pair_only", xp),
    ):
        best = None
        for alpha in b010.ALPHA_GRID:
            beta, b0 = b010.ridge_fit(design[tr], y[tr], alpha)
            pred = design @ beta + b0
            r = b010.score(y[va], pred[va])["pearson"]
            if best is None or r > best[0]:
                best = (r, alpha, pred)
        _, alpha, pred = best
        for name, idx in (("train", tr), ("val", va), ("test", te)):
            rows.append(
                {"arm": arm, "model": tag, "alpha": alpha, "split": name}
                | b010.score(y[idx], pred[idx])
            )
        print(
            f"  {tag:<22s} alpha {alpha:<6g} test pearson "
            f"{b010.score(y[te], pred[te])['pearson']:.4f}"
        )
        np.save(
            osp.join(RESULTS_DIR, f"additive_baselines_025_pred_{tag}_{arm}.npy"), pred
        )
        if tag == "B1_additive_gene":
            beta, b0 = b010.ridge_fit(xg[tr], y[tr], alpha)
            np.savez(
                osp.join(
                    RESULTS_DIR, f"additive_baselines_025_B1_coefficients_{arm}.npz"
                ),
                gene_names=np.array(gene_names),
                beta=beta,
                intercept=b0,
                alpha=alpha,
            )

    for name, idx in (("val", va), ("test", te)):
        pred = b010.hierarchical_mean(row_genes, pairs, y, tr, idx)
        rows.append(
            {"arm": arm, "model": "B3_hierarchical_mean", "alpha": None, "split": name}
            | b010.score(y[idx], pred)
        )
    print(
        f"  {'B3_hierarchical_mean':<22s}              test pearson "
        f"{rows[-1]['pearson']:.4f}"
    )

    for seed in MLP_SEEDS:
        pred, stop = b010.embedding_mlp(row_genes, y, tr, va, te, n_genes, seed=seed)
        for name, idx in (("train", tr), ("val", va), ("test", te)):
            rows.append(
                {
                    "arm": arm,
                    "model": "B5_gene_embedding_mlp",
                    "alpha": None,
                    "split": name,
                    "seed": seed,
                }
                | b010.score(y[idx], pred[idx])
            )
        np.save(
            osp.join(
                RESULTS_DIR,
                f"additive_baselines_025_pred_B5_gene_embedding_mlp_s{seed}_{arm}.npy",
            ),
            pred,
        )
        print(
            f"  B5 seed {seed} stop {stop} test pearson "
            f"{b010.score(y[te], pred[te])['pearson']:.4f}"
        )
    return rows


def job_1598_best_val() -> dict[str, object]:
    """Validation Pearson at the best epoch of the R-arm replication, from wandb."""
    import wandb

    run = wandb.Api(timeout=120).run(f"{WANDB_PROJECT}/{JOB_1598_RUN}")
    hist = pd.DataFrame(
        run.scan_history(keys=["epoch", "val/gene_interaction/Pearson"])
    )
    hist = hist.dropna()
    i = int(hist["val/gene_interaction/Pearson"].idxmax())
    return {
        "run": JOB_1598_RUN,
        "n_epochs_logged": int(hist["epoch"].nunique()),
        "best_epoch": int(hist.loc[i, "epoch"]),
        "val_pearson_best_epoch": float(hist.loc[i, "val/gene_interaction/Pearson"]),
        "note": "max over logged validation epochs, an upward-biased order statistic; "
        "no test evaluation of this checkpoint exists",
    }


def summarize(df: pd.DataFrame, ref_1598: dict[str, object]) -> dict[str, object]:
    test = df[df["split"] == "test"]
    out: dict[str, object] = {"arms": {}, "transformer": {}}
    for arm in ARMS:
        sub = (
            test[test["arm"] == arm]
            .groupby("model")["pearson"]
            .agg(["mean", "std", "count"])
        )
        out["arms"][arm] = {
            "split": ARMS[arm][0],
            "description": ARMS[arm][2],
            "test_pearson": {
                m: {
                    "mean": float(r["mean"]),
                    "sd": (None if np.isnan(r["std"]) else float(r["std"])),
                    "n": int(r["count"]),
                }
                for m, r in sub.iterrows()
            },
        }
    cgt010 = pd.read_csv(
        osp.join(RESULTS_010, "additive_baseline_gene_interaction.csv")
    )
    cgt010 = cgt010[cgt010["model"].str.startswith("CGT") & (cgt010["split"] == "test")]
    out["transformer"]["R"] = {
        "010_checkpoints_test_pearson": {
            r["model"]: float(r["pearson"]) for _, r in cgt010.iterrows()
        },
        "job_1598_replication": ref_1598,
    }
    out["transformer"]["Q"] = JOB_1609
    return out


def plot_ladder(df: pd.DataFrame, summary: dict[str, object]) -> None:
    apply_paper_style()
    test = df[df["split"] == "test"]
    agg = {
        arm: test[test["arm"] == arm].groupby("model")["pearson"].agg(["mean", "std"])
        for arm in ARMS
    }
    models = list(LABELS)
    cgt010 = summary["transformer"]["R"]["010_checkpoints_test_pearson"]
    ref = summary["transformer"]["R"]["job_1598_replication"]

    cats = models + list(CGT_010) + ["CGT_025_1598", "CGT_025_1609"]
    ticklabels = (
        [LABELS[m] for m in models]
        + list(CGT_010.values())
        + ["CGT 025 R\njob 1598", "CGT 025 Q\njob 1609"]
    )
    x = np.arange(len(cats))
    w = 0.38

    fig, ax = plt.subplots(
        figsize=(mm_to_in(PANEL_WIDTHS_MM["half_plus"]), mm_to_in(62.0))
    )
    r_vals = [agg["R"].loc[m, "mean"] for m in models]
    r_err = [np.nan_to_num(agg["R"].loc[m, "std"]) for m in models]
    q_vals = [agg["Q"].loc[m, "mean"] for m in models]
    q_err = [np.nan_to_num(agg["Q"].loc[m, "std"]) for m in models]
    ekw = {"elinewidth": 0.5, "capthick": 0.5, "capsize": 1.5}
    ax.bar(
        x[: len(models)] - w / 2,
        r_vals,
        w,
        yerr=r_err,
        error_kw=ekw,
        color=PLOT_PALETTE[0],
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        x[: len(models)] + w / 2,
        q_vals,
        w,
        yerr=q_err,
        error_kw=ekw,
        color=PLOT_PALETTE[1],
        edgecolor="black",
        linewidth=0.5,
    )
    # transformer, arm R only: the three 010 checkpoints (test) and the 025 replication
    for k, tag in enumerate(CGT_010):
        ax.bar(
            x[len(models) + k],
            cgt010[tag],
            w,
            color=PLOT_PALETTE[0],
            edgecolor="black",
            linewidth=0.5,
        )
    ax.bar(
        x[-2],
        ref["val_pearson_best_epoch"],
        w,
        color=PLOT_PALETTE[0],
        edgecolor="black",
        linewidth=0.5,
        hatch="///",
    )
    # The Q-arm transformer slot stays empty until job 1609 reports; the word stands
    # upright over its own tick, inside the axes, touching no spine or bar.
    ax.text(
        x[-1],
        0.015,
        "pending",
        ha="center",
        va="bottom",
        rotation=90,
        fontsize=6,
        color=PLOT_PALETTE[1],
    )

    ax.set_xticks(x)
    ax.set_xticklabels(ticklabels, rotation=45, ha="right")
    ax.set_ylabel("Held-out Pearson r")
    ax.set_ylim(0, 0.55)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which="minor", length=0)
    ax.grid(axis="y", which="both", linewidth=0.3, color="0.85")
    ax.set_axisbelow(True)
    handles = [
        Patch(
            facecolor=PLOT_PALETTE[0],
            edgecolor="black",
            linewidth=0.5,
            label="Arm R, random over records (test)",
        ),
        Patch(
            facecolor=PLOT_PALETTE[1],
            edgecolor="black",
            linewidth=0.5,
            label="Arm Q, query-pair disjoint (test)",
        ),
        Patch(
            facecolor=PLOT_PALETTE[0],
            edgecolor="black",
            linewidth=0.5,
            hatch="///",
            label="Val at best epoch, no test evaluation",
        ),
    ]
    # Upper left is clear: the first two models are at zero on both arms.
    ax.legend(
        handles=handles,
        loc="upper left",
        handlelength=1.4,
        borderpad=0.4,
        labelspacing=0.3,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
    fig.tight_layout(pad=0.4)
    stem = osp.join(IMAGES_DIR, "additive_baselines_025_ladder")
    fig.savefig(stem + ".png", dpi=300)
    savefig_true_size_svg(fig, stem + ".svg")
    print(f"wrote {stem}.svg")


def plot_parity(df: pd.DataFrame) -> None:
    """Arm R on 025 against the same model on the 010 build, one point per model."""
    apply_paper_style()
    r025 = (
        df[(df["split"] == "test") & (df["arm"] == "R")]
        .groupby("model")["pearson"]
        .mean()
    )
    d010 = pd.read_csv(osp.join(RESULTS_010, "additive_baseline_gene_interaction.csv"))
    r010 = d010[d010["split"] == "test"].groupby("model")["pearson"].mean()
    models = [m for m in LABELS if m in r025.index and m in r010.index]
    xs = np.array([r010[m] for m in models])
    ys = np.array([r025[m] for m in models])
    short = {
        "B0_train_mean": "B0",
        "B4_query_pair_only": "B4",
        "B3_hierarchical_mean": "B3",
        "B1_additive_gene": "B1",
        "B2_additive_plus_pair": "B2",
        "B5_gene_embedding_mlp": "B5",
    }

    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_WIDTHS_MM["third"]), mm_to_in(50.0)))
    ax.plot([0, 0.5], [0, 0.5], color="black", linewidth=0.5, linestyle="--", zorder=1)
    ax.scatter(
        xs, ys, s=14, color=PLOT_PALETTE[0], edgecolor="black", linewidth=0.4, zorder=3
    )
    # The points sit on one another (B3 and B4 coincide, B1 and B2 nearly), so the
    # per-model values go in an aligned-column table in the clear lower-right region
    # rather than as labels beside the points.
    order = [
        "B0_train_mean",
        "B1_additive_gene",
        "B2_additive_plus_pair",
        "B3_hierarchical_mean",
        "B4_query_pair_only",
        "B5_gene_embedding_mlp",
    ]
    # Box top stays below the identity line at its left edge (line is at y = x in
    # axes fraction here) and its bottom clear of the x axis.
    cols = (0.48, 0.63, 0.81)
    y0, dy = 0.36, 0.045
    ax.add_patch(
        Rectangle(
            (0.45, y0 - 6.5 * dy - 0.005),
            0.525,
            7.0 * dy + 0.03,
            transform=ax.transAxes,
            facecolor="white",
            edgecolor="black",
            linewidth=0.5,
            zorder=4,
        )
    )
    for xc, head in zip(cols, ("model", "010", "025")):
        ax.text(
            xc,
            y0,
            head,
            transform=ax.transAxes,
            fontsize=6,
            ha="left",
            va="center",
            fontweight="bold",
            zorder=5,
        )
    for i, m in enumerate(order):
        yy = y0 - (i + 1) * dy
        for xc, txt in zip(cols, (short[m], f"{r010[m]:.4f}", f"{r025[m]:.4f}")):
            ax.text(
                xc,
                yy,
                txt,
                transform=ax.transAxes,
                fontsize=6,
                ha="left",
                va="center",
                zorder=5,
            )
    ax.set_xlabel("010 build, test Pearson r")
    ax.set_ylabel("025 build, test Pearson r")
    ax.set_xlim(-0.02, 0.5)
    ax.set_ylim(-0.02, 0.5)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MultipleLocator(0.2))
        axis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which="minor", length=0)
    ax.grid(which="both", linewidth=0.3, color="0.85")
    ax.set_axisbelow(True)
    ax.set_title("arm R, same records and split", fontsize=6, pad=3)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
    fig.tight_layout(pad=0.4)
    stem = osp.join(IMAGES_DIR, "additive_baselines_025_vs_010")
    fig.savefig(stem + ".png", dpi=300)
    savefig_true_size_svg(fig, stem + ".svg")
    print(f"wrote {stem}.svg")
    for m, xi, yi in zip(models, xs, ys):
        print(f"  parity {m:<24s} 010 {xi:.6f}  025 {yi:.6f}  diff {yi - xi:+.2e}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--plot-only",
        action="store_true",
        help="redraw both figures from the CSV and summary; no refit",
    )
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    out_csv = osp.join(RESULTS_DIR, "additive_baselines_025.csv")
    out_json = osp.join(RESULTS_DIR, "additive_baselines_025_summary.json")

    if args.plot_only:
        df = pd.read_csv(out_csv)
        with open(out_json) as f:
            summary = json.load(f)
        plot_ladder(df, summary)
        plot_parity(df)
        return

    t0 = time.time()
    record_ids, y, row_genes, gene_names = load_records()
    print(
        f"S0 records {record_ids.size}  genes {len(gene_names)}  "
        f"label sd {y.std(ddof=0):.6f}"
    )
    rows: list[dict[str, object]] = []
    for arm in ARMS:
        rows.extend(fit_arm(arm, record_ids, y, row_genes, gene_names))
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")

    ref_1598 = job_1598_best_val()
    print(f"job 1598 {ref_1598}")
    summary = summarize(df, ref_1598)
    summary["n_records"] = int(record_ids.size)
    summary["n_genes"] = len(gene_names)
    summary["wall_time_s"] = round(time.time() - t0, 1)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out_json}")
    print(
        df[df["split"] == "test"]
        .groupby(["arm", "model"])["pearson"]
        .agg(["mean", "std"])
        .to_string()
    )
    plot_ladder(df, summary)
    plot_parity(df)
    print(f"wall time {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
