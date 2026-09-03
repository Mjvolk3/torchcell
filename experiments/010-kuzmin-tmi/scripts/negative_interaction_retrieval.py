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

Thresholds follow the published call rather than a bare magnitude cut. Kuzmin
2018 defines a negative trigenic interaction as a CONJUNCTION, and the sidedness
is deliberate:

    negative trigenic:  tau < -0.08  AND  p < 0.05     (one-sided)
    negative digenic:   |eps| > 0.08 AND  p < 0.05     (two-sided)

quoting the SI: "we used an established interaction magnitude cut-off for
digenic interactions (p < 0.05, |epsilon| > 0.08) and trigenic interactions
(p < 0.05, tau < -0.08)". Kuzmin 2018 scored NEGATIVES ONLY, by design, on the
grounds that negative interactions carry a better signal-to-noise ratio. The
symmetric form |tau| > 0.08 appears only in Kuzmin 2020 and the 2021 protocol,
which do score positives. Baryshnikova 2010 adds a stringent tier, which for
digenic is sign-asymmetric at eps < -0.12 for negatives and eps > 0.16 for
positives.

An earlier version of this script used the magnitude cut alone. On this build
that is four times more permissive: 29,713 records have tau < -0.08 but only
5,675 also have p < 0.05.

One caveat about the stored p-value, from the protocol paper's output format:
it is the significance of the UNADJUSTED triple-mutant epsilon, computed at the
digenic scoring stage, not a significance test on tau itself. So it is usable as
the published filter and is not a statistic about tau.
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

# The published Kuzmin call, plus the stringent tier, plus a magnitude-only row
# so the cost of dropping the p-value conjunct stays visible.
P_VALUE_MAX = 0.05
CRITERIA: list[tuple[str, float, bool]] = [
    ("published tau<-0.08 & p<0.05", -0.08, True),
    ("stringent tau<-0.12 & p<0.05", -0.12, True),
    ("magnitude only tau<-0.08", -0.08, False),
    ("magnitude only tau<-0.20", -0.20, False),
]
K_GRID = [10, 30, 100, 300, 1000, 3000, 10000]
HEADLINE = "published tau<-0.08 & p<0.05"


def tau_and_pvalue() -> tuple[np.ndarray, np.ndarray]:
    """Tau and its p-value per record, in record-index order.

    label_df carries only gene_interaction, so the p-value is read from the LMDB,
    which stores gene_interaction_p_value on every phenotype.
    """
    import lmdb

    path = osp.join(BUILD_DIR, "processed", "lmdb")
    env = lmdb.open(path, readonly=True, lock=False, subdir=True)
    n = int(env.stat()["entries"])
    tau = np.empty(n)
    pval = np.empty(n)
    with env.begin() as txn:
        for key, value in txn.cursor():
            phen = json.loads(value.decode())[0]["experiment"]["phenotype"]
            i = int(key.decode())
            tau[i] = phen["gene_interaction"]
            pval[i] = phen["gene_interaction_p_value"]
    env.close()
    assert not np.isnan(pval).any(), "every record must carry a p-value"
    return tau, pval


def labels() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Test-split tau and p-value on the row order the prediction files use."""
    tau, pval = tau_and_pvalue()
    rows = np.load(osp.join(RESULTS_DIR, "cgt_record_rows_test.npy"))
    return tau, pval, rows


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


def disjoint_split_rows() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Test rows of the query-pair-disjoint split, plus the baseline predictions."""
    label_df = pd.read_parquet(osp.join(BUILD_DIR, "processed", "label_df.parquet"))
    record_ids = np.sort(label_df["index"].to_numpy())
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
    return te, preds


def retrieval(
    tau: np.ndarray, pval: np.ndarray, preds: dict[str, np.ndarray], split_name: str
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, cut, use_p in CRITERIA:
        hit = (tau < cut) & (pval < P_VALUE_MAX) if use_p else (tau < cut)
        base = float(hit.mean())
        print(
            f"\n[{split_name}] {label}: {int(hit.sum())} of {tau.size} records, "
            f"base rate {base:.3%}"
        )
        for name, p in preds.items():
            order = np.argsort(p)
            ap = float(average_precision_score(hit, -p))
            line = [f"  {name:<18} AP {ap:.4f}"]
            for k in K_GRID:
                if k > tau.size:
                    continue
                prec = float(hit[order[:k]].mean())
                rows.append(
                    {
                        "split": split_name,
                        "criterion": label,
                        "tau_cut": cut,
                        "uses_p_value": use_p,
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
    tau_all, pval_all, rows = labels()
    print(f"random-split test: {rows.size} records")
    frames = [
        retrieval(
            tau_all[rows], pval_all[rows], random_split_predictions(rows), "random"
        )
    ]

    te_d, preds_d = disjoint_split_rows()
    print(f"\nquery-pair-disjoint test: {te_d.size} records")
    frames.append(
        retrieval(tau_all[te_d], pval_all[te_d], preds_d, "query_pair_disjoint")
    )

    out = pd.concat(frames, ignore_index=True)
    path = osp.join(RESULTS_DIR, "negative_interaction_retrieval.csv")
    out.to_csv(path, index=False)
    print(f"\nwrote {path}")
    plot(out)


def plot(out: pd.DataFrame) -> None:
    apply_paper_style()
    fig, axes = plt.subplots(
        1, 2, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(62.0)), sharey=True
    )
    for ax, split, title in (
        (axes[0], "random", "Random over records (the published split)"),
        (axes[1], "query_pair_disjoint", "Query-pair disjoint"),
    ):
        d = out[(out["split"] == split) & (out["criterion"] == HEADLINE)]
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
    axes[0].set_ylabel(r"Precision: fraction that are true negative interactions")
    fig.suptitle(
        "Retrieving published negative trigenic interactions\n(Kuzmin 2018 call: $\\tau < -0.08$ and $p < 0.05$)",
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
