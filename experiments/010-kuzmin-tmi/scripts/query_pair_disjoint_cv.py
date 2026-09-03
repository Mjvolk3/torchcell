# experiments/010-kuzmin-tmi/scripts/query_pair_disjoint_cv.py
# [[experiments.010-kuzmin-tmi.scripts.query_pair_disjoint_cv]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/query_pair_disjoint_cv
"""Query-pair-grouped cross-validation of the 010 additive nulls.

One query-pair-disjoint split is a noisy estimate. The build holds 420 Kuzmin
query doubles, so a 10 percent test share is 68 groups, and a single draw of 68
groups leaves enough between-fold variance that the same model scored 0.135 on
that split's validation part and 0.174 on its test part. Reporting either alone
would state a fold as a capability.

This script therefore runs K-fold cross-validation with the query pair as the
grouping unit: every fold holds out a disjoint set of query doubles, fits on the
rest, and the reported number is the mean and standard deviation across folds.
The same folds are used for every baseline, so the comparison between models is
paired.

The ridge penalty is selected inside each fold on an inner validation set of
held-out query pairs drawn from that fold's training groups, so no test group is
consulted during fitting. B5 early-stops on the same inner set.

Run ``query_pair_disjoint_split.py`` first for the single-split artifacts. This
script builds its own folds and does not read that split.
"""

import argparse
import json
import os
import os.path as osp
from collections import Counter

import numpy as np
import pandas as pd
import scipy.sparse as sp
from additive_baseline_gene_interaction import (
    ALPHA_GRID,
    RESULTS_DIR,
    embedding_mlp,
    gene_matrix,
    hierarchical_mean,
    load_records,
    pair_keys,
    pair_matrix,
    ridge_fit,
    score,
)
from dotenv import load_dotenv
from query_pair_disjoint_split import QUERY_PAIR_MIN_COUNT

load_dotenv()

SEED = 42
N_FOLDS = 5
INNER_VAL_FRACTION = 0.125  # of the training groups, so ~10 percent of records
MLP_SEEDS = [0, 1, 2]


def query_pair_of_record(pairs: np.ndarray) -> np.ndarray:
    """Each record's query pair, the most frequent of its three gene pairs."""
    keys, counts = np.unique(pairs.reshape(-1), return_counts=True)
    count_of = dict(zip(keys.tolist(), counts.tolist()))
    recurring = {int(k) for k, c in count_of.items() if c >= QUERY_PAIR_MIN_COUNT}
    query = np.empty(pairs.shape[0], dtype=np.int64)
    for i, row in enumerate(pairs):
        hits = [int(p) for p in row if int(p) in recurring]
        assert hits, f"record {i} carries no recurring pair"
        query[i] = min(hits, key=lambda k: (-count_of[k], k))
    return query


def grouped_folds(query: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    """Assign query pairs to folds, balancing record counts.

    Groups are placed largest first into whichever fold currently holds fewest
    records, which keeps fold sizes close despite group sizes spanning three
    orders of magnitude.
    """
    sizes = Counter(query.tolist())
    rng = np.random.default_rng(seed)
    groups = np.array(sorted(sizes.keys()), dtype=np.int64)
    rng.shuffle(groups)
    groups = sorted(groups.tolist(), key=lambda g: (-sizes[g], g))

    fold_groups: list[list[int]] = [[] for _ in range(n_folds)]
    fold_records = [0] * n_folds
    for g in groups:
        f = int(np.argmin(fold_records))
        fold_groups[f].append(g)
        fold_records[f] += sizes[g]
    return [np.array(sorted(g), dtype=np.int64) for g in fold_groups]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    ap.add_argument(
        "--skip-mlp",
        action="store_true",
        help="linear baselines only; B5 needs a GPU and dominates the runtime",
    )
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    row_genes, y, _, gene_names = load_records()
    row_genes = np.sort(row_genes, axis=1)
    n_genes = len(gene_names)
    pairs = pair_keys(row_genes)
    query = query_pair_of_record(pairs)
    folds = grouped_folds(query, args.folds, SEED)

    print(f"records {y.size}  genes {n_genes}  query pairs {len(set(query.tolist()))}")
    for f, g in enumerate(folds):
        n = int(np.isin(query, g).sum())
        print(f"  fold {f}: query pairs {g.size:>4d}  records {n:>7d} ({n / y.size:.2%})")

    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []

    for f, held in enumerate(folds):
        te = np.nonzero(np.isin(query, held))[0]
        rest_groups = np.array(
            sorted(set(query.tolist()) - set(held.tolist())), dtype=np.int64
        )
        shuffled = rest_groups.copy()
        rng.shuffle(shuffled)
        n_inner = max(1, int(round(INNER_VAL_FRACTION * shuffled.size)))
        inner_groups = shuffled[:n_inner]
        va = np.nonzero(np.isin(query, inner_groups))[0]
        tr = np.nonzero(
            np.isin(query, shuffled[n_inner:])
        )[0]
        assert not (set(query[tr].tolist()) & set(query[te].tolist()))
        assert not (set(query[va].tolist()) & set(query[te].tolist()))
        print(
            f"\nfold {f}: train {tr.size} val {va.size} test {te.size}  "
            f"test label sd {y[te].std(ddof=0):.6f}"
        )

        # B0
        rows.append(
            {"fold": f, "model": "B0_train_mean", "seed": None}
            | score(y[te], np.full(te.size, y[tr].mean()))
        )

        # Pair vocabulary is built on this fold's TRAIN records only, so a
        # held-out query pair can never receive a coefficient.
        train_pairs, train_counts = np.unique(
            pairs[tr].reshape(-1), return_counts=True
        )
        vocab = {int(p): j for j, p in enumerate(train_pairs[train_counts >= 5])}
        xg = gene_matrix(row_genes, n_genes)
        xp = pair_matrix(pairs, vocab)
        xgp = sp.hstack([xg, xp]).tocsr()

        for tag, design in (
            ("B1_additive_gene", xg),
            ("B2_additive_plus_pair", xgp),
            ("B4_query_pair_only", xp),
        ):
            best = None
            for alpha in ALPHA_GRID:
                beta, b0 = ridge_fit(design[tr], y[tr], alpha)
                pred = design @ beta + b0
                r = score(y[va], pred[va])["pearson"]
                if best is None or r > best[0]:
                    best = (r, alpha, pred)
            _, alpha, pred = best
            s = score(y[te], pred[te])
            print(f"  {tag:<22s} alpha {alpha:<8g} test pearson {s['pearson']:.4f}")
            rows.append({"fold": f, "model": tag, "seed": None, "alpha": alpha} | s)

        pred = hierarchical_mean(row_genes, pairs, y, tr, te)
        s = score(y[te], pred)
        print(f"  {'B3_hierarchical_mean':<22s} {'':<14s} test pearson {s['pearson']:.4f}")
        rows.append({"fold": f, "model": "B3_hierarchical_mean", "seed": None} | s)

        if not args.skip_mlp:
            for seed in MLP_SEEDS:
                pred, stop = embedding_mlp(
                    row_genes, y, tr, va, te, n_genes, seed=seed
                )
                s = score(y[te], pred[te])
                print(
                    f"  {'B5_gene_embedding_mlp':<22s} seed {seed} stop {stop:<3d} "
                    f"test pearson {s['pearson']:.4f}"
                )
                rows.append(
                    {"fold": f, "model": "B5_gene_embedding_mlp", "seed": seed} | s
                )

    df = pd.DataFrame(rows)
    out_csv = osp.join(RESULTS_DIR, "query_pair_disjoint_cv.csv")
    df.to_csv(out_csv, index=False)

    summary = (
        df.groupby("model")["pearson"]
        .agg(["mean", "std", "min", "max", "count"])
        .sort_values("mean")
    )
    print(f"\nwrote {out_csv}")
    print("\nheld-out Pearson across query-pair-disjoint folds")
    print(summary.to_string())

    summary_path = osp.join(RESULTS_DIR, "query_pair_disjoint_cv_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(
            {
                "n_folds": args.folds,
                "seed": SEED,
                "metric": "test Pearson, held-out query pairs",
                "models": {
                    m: {
                        "mean": float(r["mean"]),
                        "sd": float(r["std"]),
                        "min": float(r["min"]),
                        "max": float(r["max"]),
                        "n": int(r["count"]),
                    }
                    for m, r in summary.iterrows()
                },
            },
            fh,
            indent=2,
        )
    print(f"wrote {summary_path}")
    plot(df)


LABELS = {
    "B0_train_mean": "Train mean",
    "B4_query_pair_only": "Query pair\nonly",
    "B3_hierarchical_mean": "Hierarchical\nempirical mean",
    "B1_additive_gene": "Additive\n(per-gene ridge)",
    "B2_additive_plus_pair": "Additive\n+ gene-pair ridge",
    "B5_gene_embedding_mlp": "Nonlinear MLP\n(same features)",
}


def plot(df: pd.DataFrame) -> None:
    """Random split beside query-pair-disjoint folds, same models, same axis."""
    import matplotlib.pyplot as plt

    from torchcell.utils import (
        PANEL_WIDTHS_MM,
        PLOT_PALETTE,
        mm_to_in,
        savefig_true_size_svg,
    )

    random_csv = osp.join(RESULTS_DIR, "additive_baseline_gene_interaction.csv")
    rnd = pd.read_csv(random_csv)
    rnd = rnd[(rnd["split"] == "test") & rnd["model"].isin(LABELS)]
    rnd = rnd.groupby("model")["pearson"].mean()

    cv = df.groupby("model")["pearson"].agg(["mean", "std"])

    models = [m for m in LABELS if m in cv.index and m in rnd.index]
    x = np.arange(len(models))
    width = 0.38

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "axes.linewidth": 0.5,
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(
        figsize=(mm_to_in(PANEL_WIDTHS_MM["half_plus"]), mm_to_in(58.0))
    )
    ax.bar(
        x - width / 2,
        [rnd[m] for m in models],
        width,
        color=PLOT_PALETTE[5],
        edgecolor="black",
        linewidth=0.5,
        label="Random over records",
    )
    ax.bar(
        x + width / 2,
        [cv.loc[m, "mean"] for m in models],
        width,
        yerr=[np.nan_to_num(cv.loc[m, "std"]) for m in models],
        error_kw={"elinewidth": 0.5, "capthick": 0.5, "capsize": 1.5},
        color=PLOT_PALETTE[0],
        edgecolor="black",
        linewidth=0.5,
        label="Query-pair disjoint, 5 folds",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in models], rotation=45, ha="right")
    ax.set_ylabel("Held-out Pearson r")
    ax.set_ylim(0, 0.5)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.tick_params(which="minor", length=0)
    ax.grid(axis="y", which="both", linewidth=0.3, color="0.85")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    for spine in ax.spines.values():
        spine.set_visible(True)
    fig.tight_layout()

    stem = osp.join(
        os.environ["ASSET_IMAGES_DIR"],
        "010-kuzmin-tmi",
        "query_pair_disjoint_comparison",
    )
    os.makedirs(osp.dirname(stem), exist_ok=True)
    fig.savefig(stem + ".png", dpi=300)
    savefig_true_size_svg(fig, stem + ".svg")
    print(f"wrote {stem}.svg")


if __name__ == "__main__":
    main()
