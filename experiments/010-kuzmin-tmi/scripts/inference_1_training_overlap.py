# experiments/010-kuzmin-tmi/scripts/inference_1_training_overlap.py
# [[experiments.010-kuzmin-tmi.scripts.inference_1_training_overlap]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/inference_1_training_overlap

"""How much of what inference_1 nominates the model has already screened.

Kuzmin's trigenic screen crosses a query DOUBLE mutant against an array of
single mutants, so every training record is a query pair plus one array gene and
the same query pair recurs across hundreds or thousands of records. That
structure is exact on this build: 420 pairs occur five or more times and cover
376,733 pair-instances against 376,732 records, so each record carries exactly
one of them (``query_pair_disjoint_split.py`` establishes this).

It is also what the held-out numbers were mostly measuring. On the published
split, which is random over records, the additive null reaches 0.400 test
Pearson; once whole query pairs are held out it falls to 0.127 plus or minus
0.033. So whether a nominated triple sits on a query double the model trained on
is not a detail, it is most of the difference between interpolation and
extrapolation.

This asks that of the ``inference_1`` nominations, at four levels:

    query double    does the triple contain one of the 420 Kuzmin query pairs
    array gene      is the remaining gene one this screen ever measured
    any pair        how many of the triple's three pairs occur in ANY training
                    record, recurring or not
    gene            how many of the triple's three genes occur in any record

The reading direction is the tail. If the most extreme predictions are enriched
for triples carrying a trained query double, the ranking is partly reporting
screen identity; if they are depleted, the extremes are extrapolations and the
disjoint-split result is the relevant prior for them.

Outputs, under ``experiments/010-kuzmin-tmi/results/``:

    inference_1_overlap_summary.json     gene-level and space-level counts
    inference_1_overlap_by_k.csv         overlap against K, both tails
    inference_1_top_positive.csv         the highest predictions, ensembled
"""

import glob
import json
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from dotenv import load_dotenv

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

RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi", "results")
IMAGE_DIR = osp.join(ASSET_IMAGES_DIR, "010-kuzmin-tmi")
BUILD_DIR = osp.join(
    DATA_ROOT, "data/torchcell/experiments/010-kuzmin-tmi/001-small-build"
)
INFERENCE_DIR = osp.join(
    DATA_ROOT, "data/torchcell/experiments/010-kuzmin-tmi/inference_1/inferred"
)

CHECKPOINTS = {"M01_lzs9pcj3": "0.4520", "M02_yv4r30bi": "0.4472", "M03_c7671wgj": "0.4619"}
TAGS = list(CHECKPOINTS)

# A pair must recur at least this often to read as a Kuzmin query double. The
# count distribution is bimodal with nothing between 2 and 200, so any value in
# that gap gives the same 420 pairs.
QUERY_PAIR_MIN_COUNT = 5

K_GRID = [10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000]
N_TOP = 200


# ---------------------------------------------------------------------------
# Training structure
# ---------------------------------------------------------------------------


def training_triples() -> tuple[np.ndarray, list[str]]:
    """The 376,732 training triples as sorted gene-name columns."""
    label_df = pd.read_parquet(osp.join(BUILD_DIR, "processed", "label_df.parquet"))
    with open(osp.join(BUILD_DIR, "processed", "is_any_perturbed_gene_index.json")) as f:
        gene_index = json.load(f)

    record_ids = np.sort(label_df["index"].to_numpy())
    id_to_row = {int(r): i for i, r in enumerate(record_ids)}
    names = sorted(gene_index.keys())
    col_of = {g: j for j, g in enumerate(names)}

    rows = np.full((len(record_ids), 3), -1, dtype=np.int32)
    fill = np.zeros(len(record_ids), dtype=np.int8)
    for gene, ids in gene_index.items():
        col = col_of[gene]
        for rid in ids:
            row = id_to_row[int(rid)]
            rows[row, fill[row]] = col
            fill[row] += 1
    assert (fill == 3).all(), "every 010 record must carry exactly three perturbed genes"
    return np.sort(rows, axis=1), names


def pair_codes(triples: np.ndarray, base: int) -> np.ndarray:
    """Canonical unordered pair keys per row, shape (n, 3)."""
    a, b, c = (triples[:, j].astype(np.int64) for j in range(3))
    return np.stack([a * base + b, a * base + c, b * base + c], axis=1)


# ---------------------------------------------------------------------------
# Inference space
# ---------------------------------------------------------------------------


def prediction_path(pearson: str) -> str:
    matches = [
        f
        for f in glob.glob(osp.join(INFERENCE_DIR, f"*Pearson={pearson}*.parquet"))
        if not f.endswith((".rank0", ".rank1", ".rank2", ".rank3"))
    ]
    assert len(matches) == 1, f"expected one file for Pearson={pearson}, got {matches}"
    return matches[0]


def inference_space(name_to_col: dict[str, int]) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """Inference triples as training-gene columns, plus one column per checkpoint.

    A gene present in inference_1 but absent from any training record gets a
    fresh column, so the encoding stays total and "unseen gene" stays visible
    rather than being silently dropped.
    """
    preds: dict[str, np.ndarray] = {}
    triples: np.ndarray | None = None
    vocab = list(name_to_col)
    col_of = dict(name_to_col)
    index_ref: np.ndarray | None = None

    for tag, pearson in CHECKPOINTS.items():
        table = pq.read_table(
            prediction_path(pearson),
            columns=["index", "gene1", "gene2", "gene3", "prediction"],
        )
        idx = table["index"].to_numpy()
        if index_ref is None:
            index_ref = idx
        else:
            assert np.array_equal(idx, index_ref), f"{tag} row order differs"

        if triples is None:
            cols = []
            for col in ("gene1", "gene2", "gene3"):
                s = table[col].to_pandas()
                for g in s.unique():
                    if g not in col_of:
                        col_of[g] = len(vocab)
                        vocab.append(g)
                cols.append(s.map(col_of).to_numpy(dtype=np.int32))
            triples = np.sort(np.stack(cols, axis=1), axis=1)

        preds[tag] = table["prediction"].to_numpy().astype(np.float64)
        del table

    assert triples is not None
    return triples, pd.DataFrame(preds), vocab


# ---------------------------------------------------------------------------
# Overlap
# ---------------------------------------------------------------------------


def overlap_flags(
    inf_triples: np.ndarray,
    train_triples: np.ndarray,
    base: int,
) -> dict[str, np.ndarray]:
    """Per-inference-triple overlap against the training screen."""
    train_pairs = pair_codes(train_triples, base)

    keys, counts = np.unique(train_pairs.reshape(-1), return_counts=True)
    query_pairs = keys[counts >= QUERY_PAIR_MIN_COUNT]
    all_pairs = keys
    train_genes = np.unique(train_triples.reshape(-1))
    # Array genes: the gene left over once a record's query pair is removed.
    is_query = np.isin(train_pairs, query_pairs)
    # Column j of pair_codes omits gene column (2, 1, 0) respectively.
    omitted = np.array([2, 1, 0])
    array_gene = train_triples[np.arange(len(train_triples)), omitted[is_query.argmax(axis=1)]]
    array_genes = np.unique(array_gene)

    # Query genes: the two members of each of the 420 query doubles.
    query_genes = np.unique(
        np.stack([query_pairs // base, query_pairs % base]).reshape(-1)
    )

    inf_pairs = pair_codes(inf_triples, base)
    in_query = np.isin(inf_pairs, query_pairs)
    in_any = np.isin(inf_pairs, all_pairs)
    gene_seen = np.isin(inf_triples, train_genes)
    gene_is_array = np.isin(inf_triples, array_genes)
    gene_is_query = np.isin(inf_triples, query_genes)

    return {
        "has_query_double": in_query.any(axis=1),
        "n_query_doubles": in_query.sum(axis=1).astype(np.int8),
        "n_pairs_seen": in_any.sum(axis=1).astype(np.int8),
        "n_genes_seen": gene_seen.sum(axis=1).astype(np.int8),
        "n_genes_array": gene_is_array.sum(axis=1).astype(np.int8),
        "n_genes_query": gene_is_query.sum(axis=1).astype(np.int8),
        "_query_pairs": query_pairs,
        "_train_genes": train_genes,
        "_array_genes": array_genes,
        "_query_genes": query_genes,
    }


def overlap_by_k(
    ens: np.ndarray, flags: dict[str, np.ndarray]
) -> pd.DataFrame:
    """Overlap statistics for the K most extreme predictions, both tails."""
    order_neg = np.argsort(ens, kind="stable")
    order_pos = order_neg[::-1]
    rows: list[dict[str, object]] = []
    for tail, order in (("most negative", order_neg), ("most positive", order_pos)):
        for k in K_GRID:
            sel = order[:k]
            rows.append(
                {
                    "tail": tail,
                    "k": k,
                    "frac_with_query_double": float(flags["has_query_double"][sel].mean()),
                    "mean_pairs_seen": float(flags["n_pairs_seen"][sel].mean()),
                    "mean_genes_seen": float(flags["n_genes_seen"][sel].mean()),
                    "frac_all_three_genes_seen": float(
                        (flags["n_genes_seen"][sel] == 3).mean()
                    ),
                }
            )
    rows.append(
        {
            "tail": "whole space",
            "k": len(ens),
            "frac_with_query_double": float(flags["has_query_double"].mean()),
            "mean_pairs_seen": float(flags["n_pairs_seen"].mean()),
            "mean_genes_seen": float(flags["n_genes_seen"].mean()),
            "frac_all_three_genes_seen": float((flags["n_genes_seen"] == 3).mean()),
        }
    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("loading the training screen ...")
    train_triples, train_names = training_triples()
    name_to_col = {g: j for j, g in enumerate(train_names)}
    print(f"  {len(train_triples):,} records over {len(train_names):,} genes")

    print("loading inference_1 ...")
    inf_triples, preds, vocab = inference_space(name_to_col)
    ens = preds[TAGS].to_numpy().mean(axis=1)
    print(f"  {len(inf_triples):,} triples over {len(vocab) - len(train_names):,} "
          f"genes new to the training vocabulary")

    base = len(vocab) + 1
    flags = overlap_flags(inf_triples, train_triples, base)

    inf_genes = np.unique(inf_triples.reshape(-1))
    is_q = np.isin(inf_genes, flags["_query_genes"])
    is_a = np.isin(inf_genes, flags["_array_genes"])
    # How many of the 420 query doubles are even representable here. If both
    # members of a query double are not in the 526-gene space, no inference
    # triple can carry it, so a low triple-level overlap may be structural
    # rather than a property of the ranking.
    qp = flags["_query_pairs"]
    inf_gene_set = set(inf_genes.tolist())
    representable = int(
        sum(
            1
            for p in qp.tolist()
            if (p // base) in inf_gene_set and (p % base) in inf_gene_set
        )
    )
    summary = {
        "n_training_records": int(len(train_triples)),
        "n_training_genes": int(len(train_names)),
        "n_query_doubles": int(len(qp)),
        "n_query_genes": int(len(flags["_query_genes"])),
        "n_array_genes": int(len(flags["_array_genes"])),
        "n_inference_triples": int(len(inf_triples)),
        "n_inference_genes": int(len(inf_genes)),
        "inference_genes_seen_in_training": int(
            np.isin(inf_genes, flags["_train_genes"]).sum()
        ),
        "inference_genes_that_are_query_genes": int(is_q.sum()),
        "inference_genes_that_are_array_genes": int(is_a.sum()),
        "inference_genes_query_and_array": int((is_q & is_a).sum()),
        "inference_genes_array_only": int((is_a & ~is_q).sum()),
        "inference_genes_query_only": int((is_q & ~is_a).sum()),
        "inference_genes_unseen": int((~is_q & ~is_a).sum()),
        "query_doubles_representable_in_space": representable,
        "frac_triples_with_query_double": float(flags["has_query_double"].mean()),
        "n_triples_with_query_double": int(flags["has_query_double"].sum()),
        "frac_triples_all_three_genes_seen": float((flags["n_genes_seen"] == 3).mean()),
        "pairs_seen_histogram": {
            str(i): int((flags["n_pairs_seen"] == i).sum()) for i in range(4)
        },
        "genes_seen_histogram": {
            str(i): int((flags["n_genes_seen"] == i).sum()) for i in range(4)
        },
    }
    with open(osp.join(RESULTS_DIR, "inference_1_overlap_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\ngene-level and space-level overlap:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    by_k = overlap_by_k(ens, flags)
    by_k.to_csv(osp.join(RESULTS_DIR, "inference_1_overlap_by_k.csv"), index=False)
    print("\noverlap against K:")
    print(by_k.to_string(index=False))

    # The highest predictions, which the earlier review reported only in summary.
    names = np.array(vocab, dtype=object)
    order_pos = np.argsort(ens, kind="stable")[::-1][:N_TOP]
    vals = preds.iloc[order_pos][TAGS].to_numpy()
    top_pos = pd.DataFrame(
        {
            "rank": np.arange(1, N_TOP + 1),
            "triple": [" + ".join(names[r]) for r in inf_triples[order_pos]],
            "ensemble_mean": vals.mean(axis=1),
            "ensemble_sd": vals.std(axis=1, ddof=0),
            "ensemble_min": vals.min(axis=1),
            "ensemble_max": vals.max(axis=1),
            "sign_agree": np.isin((vals > 0).sum(axis=1), [0, 3]),
            "has_query_double": flags["has_query_double"][order_pos],
            "n_pairs_seen": flags["n_pairs_seen"][order_pos],
            "n_genes_seen": flags["n_genes_seen"][order_pos],
        }
    )
    for j, tag in enumerate(TAGS):
        top_pos[tag] = vals[:, j]
    top_pos.to_csv(osp.join(RESULTS_DIR, "inference_1_top_positive.csv"), index=False)
    print(f"\ntop {N_TOP} positive: {top_pos['ensemble_mean'].max():.4f} to "
          f"{top_pos['ensemble_mean'].min():.4f}, "
          f"{int(top_pos['has_query_double'].sum())} carry a trained query double")

    # How concentrated each tail is. The negative review found one gene in 110 of
    # its top 200, so the same count on the positive side says whether that is a
    # property of the model or of one locus.
    pos_genes = pd.Series(inf_triples[order_pos].reshape(-1)).map(lambda c: names[c])
    pos_counts = pos_genes.value_counts()
    summary["top_200_positive_distinct_genes"] = int(pos_counts.size)
    summary["top_200_positive_top_gene"] = str(pos_counts.index[0])
    summary["top_200_positive_top_gene_count"] = int(pos_counts.iloc[0])
    with open(osp.join(RESULTS_DIR, "inference_1_overlap_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  {pos_counts.size} distinct genes in the top {N_TOP}; "
          f"{pos_counts.index[0]} appears in {pos_counts.iloc[0]}")

    plot(ens, flags, by_k, top_pos, summary)


def plot(
    ens: np.ndarray,
    flags: dict[str, np.ndarray],
    by_k: pd.DataFrame,
    top_pos: pd.DataFrame,
    summary: dict[str, object],
) -> None:
    apply_paper_style()
    fig, axgrid = plt.subplots(
        2, 2, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(115.0))
    )
    axes = axgrid.ravel()

    # a. The highest predictions, which the negative-only review did not show.
    ax = axes[0]
    show = top_pos.head(20).iloc[::-1]
    y = np.arange(len(show))
    ax.barh(
        y,
        show["ensemble_mean"],
        color=PLOT_PALETTE[0],
        edgecolor="black",
        linewidth=0.4,
        height=0.72,
    )
    ax.errorbar(
        show["ensemble_mean"],
        y,
        xerr=[
            show["ensemble_mean"] - show["ensemble_min"],
            show["ensemble_max"] - show["ensemble_mean"],
        ],
        fmt="none",
        ecolor="black",
        elinewidth=0.5,
        capsize=1.2,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(show["triple"], fontsize=4)
    ax.set_xlabel(r"Ensemble predicted $\tau$")
    ax.set_title("Top 20 positive, bars span 3 checkpoints", fontsize=6)

    # b. Gene-level overlap with the screen's two roles.
    ax = axes[1]
    labels = [
        "query gene\n(in a query double)",
        "array gene only",
        "never perturbed\nin the build",
    ]
    counts = [
        int(summary["inference_genes_that_are_query_genes"]),
        int(summary["inference_genes_array_only"]),
        int(summary["inference_genes_unseen"]),
    ]
    ax.bar(
        np.arange(3),
        counts,
        color=[PLOT_PALETTE[0], PLOT_PALETTE[1], PLOT_PALETTE[2]],
        edgecolor="black",
        linewidth=0.4,
        width=0.62,
    )
    for i, c in enumerate(counts):
        ax.text(
            i,
            c,
            f"{c}\n{c / int(summary['n_inference_genes']):.0%}",
            ha="center",
            va="bottom",
            fontsize=5,
        )
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(labels, fontsize=5)
    ax.set_ylabel("Genes")
    ax.set_ylim(0, max(counts) * 1.28)
    ax.set_title(
        f"The {summary['n_inference_genes']} inference_1 genes by screen role", fontsize=6
    )

    # c. Does the extreme tail sit on pairs the screen already co-measured.
    ax = axes[2]
    base_pairs = float(by_k[by_k["tail"] == "whole space"]["mean_pairs_seen"].iloc[0])
    for i, tail in enumerate(("most negative", "most positive")):
        d = by_k[by_k["tail"] == tail].sort_values("k")
        ax.plot(
            d["k"],
            d["mean_pairs_seen"],
            marker="o",
            ms=2.5,
            linewidth=0.9,
            color=PLOT_PALETTE[i],
            markeredgecolor="black",
            markeredgewidth=0.3,
            label=f"K {tail}",
        )
    ax.axhline(
        base_pairs,
        color="black",
        linewidth=0.7,
        linestyle="--",
        label=f"whole space {base_pairs:.2f}",
    )
    ax.set_xscale("log")
    ax.set_ylim(0, 3)
    ax.set_xlabel("K most extreme predictions")
    ax.set_ylabel("Mean pairs already screened (of 3)")
    ax.set_title("Pair overlap rises sharply in the tail", fontsize=6)
    ax.legend(frameon=False, fontsize=4.5, loc="best")

    # d. Total overlap: how much of each triple the screen has already seen.
    ax = axes[3]
    order_neg = np.argsort(ens, kind="stable")
    groups = {
        "whole space": np.arange(len(ens)),
        "top 1,000 neg": order_neg[:1000],
        "top 1,000 pos": order_neg[::-1][:1000],
    }
    width = 0.26
    x = np.arange(4)
    for i, (label, sel) in enumerate(groups.items()):
        counts = np.array([(flags["n_pairs_seen"][sel] == v).mean() for v in range(4)])
        ax.bar(
            x + (i - 1) * width,
            counts,
            width=width,
            color=PLOT_PALETTE[i],
            edgecolor="black",
            linewidth=0.4,
            label=label,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in range(4)])
    ax.set_xlabel("Pairs of the triple already screened (of 3)")
    ax.set_ylabel("Fraction of triples")
    ax.set_title("Total pair overlap with training", fontsize=6)
    ax.legend(frameon=False, fontsize=4.5, loc="best")

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(True)
        ax.grid(which="both", linewidth=0.3, color="0.85")
        ax.set_axisbelow(True)

    fig.suptitle(
        "inference_1 nominations against the Kuzmin screen the model trained on",
        fontsize=6.5,
    )
    fig.tight_layout()
    os.makedirs(IMAGE_DIR, exist_ok=True)
    stem = osp.join(IMAGE_DIR, "inference_1_training_overlap")
    fig.savefig(stem + ".png", dpi=300)
    savefig_true_size_svg(fig, stem + ".svg")
    print(f"\nwrote {stem}.svg")


if __name__ == "__main__":
    main()
