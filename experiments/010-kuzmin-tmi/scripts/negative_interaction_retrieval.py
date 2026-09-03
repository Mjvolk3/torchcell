# experiments/010-kuzmin-tmi/scripts/negative_interaction_retrieval.py
# [[experiments.010-kuzmin-tmi.scripts.negative_interaction_retrieval]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/negative_interaction_retrieval
"""Can the 010 model be used to find NEGATIVE trigenic interactions?

Every confident prediction the model makes about the construction panel is
negative: 17 of the 20 build-list triples have all three checkpoints agreeing on
sign and all 17 are negative, while the one positive is not agreed. That invites
using the model as a retriever of negative interactions rather than positive
ones.

The reason that needs testing rather than assuming is that the label is already
skewed negative. Of 376,732 records, 59.4 percent have tau below zero and the
mean is -0.0080. A squared-error fit shrinks toward the mean, so a model that had
learned nothing beyond the mean would still emit mostly negative predictions.
"Most predictions are negative" is therefore not evidence of anything.

What would be evidence is retrieval: among the K records a model calls most
negative, what fraction really are strong negatives, and does that beat both the
base rate and the additive null. That is what this script measures, as
precision at K and as enrichment over the base rate, on the held-out test split.

It runs the same measurement on the query-pair-disjoint split for the baselines,
which is where the random-split leak is removed. The transformer has not been
retrained on that split, so it has no row there.

Thresholds are on tau alone. Kuzmin classifies a negative trigenic interaction
using a significance test as well, and the build stores no p-value, so a
threshold here is a magnitude cut and not the published call.
"""

import json
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import average_precision_score

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

RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi", "results")
IMAGE_DIR = osp.join(ASSET_IMAGES_DIR, "010-kuzmin-tmi")
BUILD_DIR = osp.join(
    DATA_ROOT, "data/torchcell/experiments/010-kuzmin-tmi/001-small-build"
)

THRESHOLDS = [-0.05, -0.08, -0.10, -0.20]
K_GRID = [10, 30, 100, 300, 1000, 3000, 10000]
HEADLINE_TAU = -0.10


def labels() -> tuple[np.ndarray, np.ndarray]:
    """Test-split tau on the row order the saved prediction files use."""
    label_df = pd.read_parquet(osp.join(BUILD_DIR, "processed", "label_df.parquet"))
    record_ids = np.sort(label_df["index"].to_numpy())
    y_all = label_df.set_index("index").loc[record_ids, "gene_interaction"].to_numpy()
    rows = np.load(osp.join(RESULTS_DIR, "cgt_record_rows_test.npy"))
    return y_all, rows


def random_split_predictions(rows: np.ndarray) -> dict[str, np.ndarray]:
    preds: dict[str, np.ndarray] = {}
    for tag, fname in (
        ("B1 additive ridge", "additive_baseline_pred_B1_additive_gene.npy"),
        ("B5 nonlinear MLP", "additive_baseline_pred_B5_gene_embedding_mlp_s0.npy"),
    ):
        preds[tag] = np.load(osp.join(RESULTS_DIR, fname))[rows]
    for tag in ("M01_lzs9pcj3", "M02_yv4r30bi", "M03_c7671wgj"):
        preds[f"CGT {tag.split('_')[0]}"] = np.load(
            osp.join(RESULTS_DIR, f"cgt_predictions_{tag}_test.npy")
        )
    return preds


def disjoint_split_predictions() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """The same measurement where the query-pair leak is removed."""
    label_df = pd.read_parquet(osp.join(BUILD_DIR, "processed", "label_df.parquet"))
    record_ids = np.sort(label_df["index"].to_numpy())
    y_all = label_df.set_index("index").loc[record_ids, "gene_interaction"].to_numpy()
    id_to_row = {int(r): i for i, r in enumerate(record_ids)}
    with open(osp.join(RESULTS_DIR, "index_query_pair_disjoint_seed_42.json")) as f:
        split = json.load(f)
    te = np.array([id_to_row[int(r)] for r in split["test"]], dtype=np.int64)
    preds = {
        "B1 additive ridge": np.load(
            osp.join(
                RESULTS_DIR,
                "additive_baseline_pred_B1_additive_gene_query_pair_disjoint.npy",
            )
        )[te],
        "B5 nonlinear MLP": np.load(
            osp.join(
                RESULTS_DIR,
                "additive_baseline_pred_B5_gene_embedding_mlp_s0_query_pair_disjoint.npy",
            )
        )[te],
    }
    return y_all[te], preds


def retrieval(
    y: np.ndarray, preds: dict[str, np.ndarray], split_name: str
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for tau in THRESHOLDS:
        hit = y < tau
        base = float(hit.mean())
        print(
            f"\n[{split_name}] tau < {tau}: {int(hit.sum())} of {y.size} records, "
            f"base rate {base:.3%}"
        )
        for name, p in preds.items():
            order = np.argsort(p)  # most negative prediction first
            ap = float(average_precision_score(hit, -p))
            line = [f"  {name:<18} AP {ap:.4f}"]
            for k in K_GRID:
                if k > y.size:
                    continue
                prec = float(hit[order[:k]].mean())
                rows.append(
                    {
                        "split": split_name,
                        "threshold": tau,
                        "model": name,
                        "k": k,
                        "precision": prec,
                        "base_rate": base,
                        "enrichment": prec / base if base > 0 else np.nan,
                        "average_precision": ap,
                    }
                )
                if k in (100, 1000):
                    line.append(f"P@{k} {prec:.3f} ({prec / base:.1f}x)")
            print("  ".join(line))
    return pd.DataFrame(rows)


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    y_all, rows = labels()
    y = y_all[rows]
    print(f"random-split test: {y.size} records")
    frames = [retrieval(y, random_split_predictions(rows), "random")]

    y_d, preds_d = disjoint_split_predictions()
    print(f"\nquery-pair-disjoint test: {y_d.size} records")
    frames.append(retrieval(y_d, preds_d, "query_pair_disjoint"))

    out = pd.concat(frames, ignore_index=True)
    path = osp.join(RESULTS_DIR, "negative_interaction_retrieval.csv")
    out.to_csv(path, index=False)
    print(f"\nwrote {path}")
    plot(out)


def plot(out: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "axes.linewidth": 0.5,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(62.0)), sharey=True
    )
    for ax, split, title in (
        (axes[0], "random", "Random over records (the published split)"),
        (axes[1], "query_pair_disjoint", "Query-pair disjoint"),
    ):
        d = out[(out["split"] == split) & (out["threshold"] == HEADLINE_TAU)]
        if d.empty:
            continue
        for i, (name, g) in enumerate(d.groupby("model", sort=False)):
            g = g.sort_values("k")
            ax.plot(
                g["k"],
                g["precision"],
                marker="o",
                ms=2.5,
                linewidth=0.9,
                color=PLOT_PALETTE[i % 6],
                markeredgecolor="black",
                markeredgewidth=0.3,
                label=name,
            )
        base = float(d["base_rate"].iloc[0])
        ax.axhline(
            base,
            color="black",
            linewidth=0.7,
            linestyle="--",
            label=f"base rate {base:.1%}",
        )
        ax.set_xscale("log")
        ax.set_xlabel("K most negative predictions")
        ax.set_title(title, fontsize=6)
        for spine in ax.spines.values():
            spine.set_visible(True)
        ax.grid(which="both", linewidth=0.3, color="0.85")
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=5, loc="upper right")
    axes[0].set_ylabel(rf"Precision, fraction with true $\tau < {HEADLINE_TAU}$")
    fig.suptitle(
        "Retrieving strong negative trigenic interactions on held-out test data",
        fontsize=6.5,
    )
    fig.tight_layout()
    os.makedirs(IMAGE_DIR, exist_ok=True)
    stem = osp.join(IMAGE_DIR, "negative_interaction_retrieval")
    fig.savefig(stem + ".png", dpi=300)
    savefig_true_size_svg(fig, stem + ".svg")
    print(f"wrote {stem}.svg")


if __name__ == "__main__":
    main()
