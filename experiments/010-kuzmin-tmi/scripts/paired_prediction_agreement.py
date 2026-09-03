# experiments/010-kuzmin-tmi/scripts/paired_prediction_agreement.py
# [[experiments.010-kuzmin-tmi.scripts.paired_prediction_agreement]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/paired_prediction_agreement

"""Record-by-record agreement between the nulls and the transformer on test data.

Comparing summary statistics only says two models score alike. Two models can
post the same Pearson while disagreeing on every record. The stronger comparison,
and the one Figure 1 of the DeWitt preprint makes, is between the predictions
themselves.

That needs per-record transformer predictions on the labeled split, which
``score_010_checkpoints_directly.py`` produces and validates against the
recorded metrics. This script joins them with the baseline predictions from
``additive_baseline_gene_interaction.py`` and reports, on the 37,673 test
records:

  * the full correlation matrix between every model's predictions,
  * how much of the transformer's advantage over the additive null is shared
    with it, via the correlation of their errors,
  * a paired bootstrap over records for the Pearson difference between the
    additive null and each checkpoint, so the gap in the results table gets an
    interval rather than being read as exact,
  * the top-K overlap of the records each model calls most extreme.
"""

import os
import os.path as osp
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import pearsonr, spearmanr

from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()

EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi", "results")

BASELINES = {
    "B1_additive": "additive_baseline_pred_B1_additive_gene.npy",
    "B2_additive_pair": "additive_baseline_pred_B2_additive_plus_pair.npy",
    "B4_query_pair": "additive_baseline_pred_B4_query_pair_only.npy",
    "B5_mlp": "additive_baseline_pred_B5_gene_embedding_mlp_s0.npy",
}
CHECKPOINTS = ["M01_lzs9pcj3", "M02_yv4r30bi", "M03_c7671wgj"]
TOP_K = [100, 1_000, 10_000]
N_BOOT = 2_000
RNG = np.random.default_rng(0)


def load_test() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Test labels and every model's test predictions, on a common record order."""
    import json

    data_root = os.environ["DATA_ROOT"]
    build = osp.join(
        data_root, "data/torchcell/experiments/010-kuzmin-tmi/001-small-build"
    )
    label_df = pd.read_parquet(osp.join(build, "processed", "label_df.parquet"))
    record_ids = np.sort(label_df["index"].to_numpy())
    y_all = (
        label_df.set_index("index").loc[record_ids, "gene_interaction"].to_numpy()
    )
    rows = np.load(osp.join(RESULTS_DIR, "cgt_record_rows_test.npy"))

    preds: dict[str, np.ndarray] = {}
    for tag, fname in BASELINES.items():
        # Baseline arrays are over every record; select the test rows.
        preds[tag] = np.load(osp.join(RESULTS_DIR, fname))[rows]
    for tag in CHECKPOINTS:
        preds[f"CGT_{tag.split('_')[0]}"] = np.load(
            osp.join(RESULTS_DIR, f"cgt_predictions_{tag}_test.npy")
        )
    del json
    return y_all[rows], preds


def paired_bootstrap(
    y: np.ndarray, a: np.ndarray, b: np.ndarray
) -> tuple[float, float, float]:
    """Percentile interval for r(y, b) - r(y, a), resampling records."""
    n = y.size
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = RNG.integers(0, n, n)
        diffs[i] = pearsonr(y[idx], b[idx])[0] - pearsonr(y[idx], a[idx])[0]
    point = float(pearsonr(y, b)[0] - pearsonr(y, a)[0])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return point, float(lo), float(hi)


def main() -> None:
    y, preds = load_test()
    names = list(preds)
    print(f"test records: {y.size}")
    for k in names:
        print(
            f"  {k:<18} pearson {pearsonr(y, preds[k])[0]:.4f}  "
            f"sd {preds[k].std():.6f}"
        )

    rows: list[dict[str, object]] = []

    print("\nprediction-to-prediction agreement on the test split")
    for a, b in combinations(names, 2):
        r = float(pearsonr(preds[a], preds[b])[0])
        rho = float(spearmanr(preds[a], preds[b])[0])
        print(f"  {a:<18} vs {b:<18} Pearson {r:.4f}  Spearman {rho:.4f}")
        rows.append({"pair": f"{a}|{b}", "quantity": "pred_pearson", "value": r})
        rows.append({"pair": f"{a}|{b}", "quantity": "pred_spearman", "value": rho})

    print("\nerror correlation, are the models wrong in the same places")
    err = {k: y - preds[k] for k in names}
    for a, b in combinations(names, 2):
        r = float(pearsonr(err[a], err[b])[0])
        rows.append({"pair": f"{a}|{b}", "quantity": "error_pearson", "value": r})
    for tag in [n for n in names if n.startswith("CGT")]:
        r = float(pearsonr(err["B1_additive"], err[tag])[0])
        print(f"  B1_additive vs {tag:<10} error Pearson {r:.4f}")

    print(f"\npaired bootstrap on the Pearson gap over B1, {N_BOOT} resamples")
    for tag in [n for n in names if n.startswith("CGT")] + ["B5_mlp"]:
        point, lo, hi = paired_bootstrap(y, preds["B1_additive"], preds[tag])
        print(f"  {tag:<10} delta r = {point:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
        rows.append({"pair": f"B1_additive|{tag}", "quantity": "delta_r", "value": point})
        rows.append({"pair": f"B1_additive|{tag}", "quantity": "delta_r_lo", "value": lo})
        rows.append({"pair": f"B1_additive|{tag}", "quantity": "delta_r_hi", "value": hi})

    # The raw prediction correlation between two checkpoints mixes the additive
    # part, which both reproduce, with the non-additive part, which is the thing
    # in question. Regressing each model's prediction on B1's and correlating the
    # residuals separates them: this is the correlation of the NON-ADDITIVE
    # content across two training runs.
    print("\nnon-additive content, prediction residualized on B1")
    b1 = preds["B1_additive"]
    b1c = b1 - b1.mean()

    def residualize(v: np.ndarray) -> np.ndarray:
        vc = v - v.mean()
        return vc - b1c * (float(vc @ b1c) / float(b1c @ b1c))

    nonadd = {k: residualize(preds[k]) for k in names if k != "B1_additive"}
    share = {
        k: float(nonadd[k].var() / preds[k].var()) for k in nonadd
    }
    for k in sorted(share):
        print(f"  {k:<18} share of prediction variance not explained by B1 {share[k]:.4f}")
        rows.append(
            {"pair": k, "quantity": "nonadditive_variance_share", "value": share[k]}
        )
    cgt = [n for n in names if n.startswith("CGT")]
    for a, b in combinations(cgt, 2):
        r = float(pearsonr(nonadd[a], nonadd[b])[0])
        print(f"  {a:<12} vs {b:<12} non-additive Pearson {r:.4f}")
        rows.append({"pair": f"{a}|{b}", "quantity": "nonadditive_pearson", "value": r})
    for a in cgt:
        r = float(pearsonr(nonadd[a], nonadd["B5_mlp"])[0])
        print(f"  {a:<12} vs {'B5_mlp':<12} non-additive Pearson {r:.4f}")
        rows.append(
            {"pair": f"{a}|B5_mlp", "quantity": "nonadditive_pearson", "value": r}
        )
    # Does the non-additive part carry signal? Correlate it with the part of the
    # label that B1 also fails to explain.
    y_res = residualize(y)
    for a in cgt + ["B5_mlp"]:
        r = float(pearsonr(nonadd[a], y_res)[0])
        print(f"  {a:<12} non-additive vs residual label  Pearson {r:.4f}")
        rows.append(
            {"pair": a, "quantity": "nonadditive_vs_label_residual", "value": r}
        )

    print("\ntop-K overlap on the most negative predicted interaction")
    order = {k: np.argsort(preds[k]) for k in names}
    # A sweep rather than three points: the tail is where a selection reads, and
    # the question is where agreement sets in, not whether it is low at K=100.
    for k in TOP_K:
        for a, b in combinations(["B1_additive", "B5_mlp", "CGT_M03"], 2):
            ov = np.intersect1d(order[a][:k], order[b][:k]).size / k
            print(f"  K={k:>6} {a:<12} vs {b:<12} {ov:.4f}")
            rows.append(
                {"pair": f"{a}|{b}", "quantity": f"bottom_{k}_overlap", "value": ov}
            )
    print("\ntop-K overlap between transformer training runs, swept")
    for k in (10, 30, 100, 300, 1_000, 3_000, 10_000, 30_000):
        for a, b in combinations(cgt, 2):
            ov = np.intersect1d(order[a][:k], order[b][:k]).size / k
            rows.append(
                {"pair": f"{a}|{b}", "quantity": f"bottom_{k}_overlap", "value": ov}
            )
        vals = [
            np.intersect1d(order[a][:k], order[b][:k]).size / k
            for a, b in combinations(cgt, 2)
        ]
        print(f"  K={k:>6} checkpoint pairs {min(vals):.3f} to {max(vals):.3f}")

    out = osp.join(RESULTS_DIR, "paired_prediction_agreement.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")
    plot(y, preds)


def plot(y: np.ndarray, preds: dict[str, np.ndarray]) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "axes.linewidth": 0.5,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(
        1, 3, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(58.0))
    )
    panels = [
        ("B1_additive", "CGT_M03", "Additive ridge", "Transformer M03"),
        ("B5_mlp", "CGT_M03", "Nonlinear MLP", "Transformer M03"),
        ("CGT_M01", "CGT_M03", "Transformer M01", "Transformer M03"),
    ]
    for ax, (a, b, la, lb) in zip(axes, panels):
        ax.hexbin(preds[a], preds[b], gridsize=60, bins="log", cmap="magma_r", linewidths=0)
        lo = float(min(preds[a].min(), preds[b].min()))
        hi = float(max(preds[a].max(), preds[b].max()))
        ax.plot([lo, hi], [lo, hi], color=PLOT_PALETTE[5], linewidth=0.5, linestyle="--")
        r = pearsonr(preds[a], preds[b])[0]
        ax.set_xlabel(la)
        ax.set_ylabel(lb)
        ax.set_title(f"r = {r:.3f}", fontsize=6, pad=2)
        for spine in ax.spines.values():
            spine.set_visible(True)
    fig.tight_layout()

    stem = osp.join(ASSET_IMAGES_DIR, "010-kuzmin-tmi", "paired_prediction_agreement")
    os.makedirs(osp.dirname(stem), exist_ok=True)
    fig.savefig(stem + ".png", dpi=300)
    savefig_true_size_svg(fig, stem + ".svg")
    print(f"wrote {stem}.svg")


if __name__ == "__main__":
    main()
