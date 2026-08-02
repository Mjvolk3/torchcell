# experiments/020-cachera-betaxanthin/scripts/analyze_merzbacher_baseline.py
# [[experiments.020-cachera-betaxanthin.merzbacher-comparison]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/020-cachera-betaxanthin/scripts/analyze_merzbacher_baseline
"""What Merzbacher 2025 actually achieved on betaxanthin, from their own shipped predictions.

The paper reports "69.8 % accuracy against a 67.2 % majority rate" and nothing else for this
task -- no correlation, no AUC, no confusion matrix. On a problem where 67 % of genes are one
class, accuracy alone cannot distinguish "learned something" from "predicted the majority".

Their Zenodo deposit settles it. `figures/fig4/fig4c_*.csv` holds PER-GENE, PER-FLUX-SAMPLE
predictions with class scores, and `figures/fig4/fig4b.csv` holds per-fold metrics including
an **MCC column that appears nowhere in the paper**. This script aggregates the sample-level
predictions to gene level the way their own `tools/knockout_voting.py` does -- majority vote
over a deletion's flux samples -- and reports the confusion matrix, the fraction of genes
called medium, and high-producer recall.

WHY THIS MATTERS FOR OUR COMPARISON. It fixes what "beating them" means. If their model calls
~95 % of genes medium, then matching their accuracy is not the goal and exceeding it slightly
is not a result; the informative axes are high-producer recall and rank correlation, which is
also what strain design needs. Their own MCC is the honest headline they had in hand.

Be fair when writing this up: they claim "promising accuracy", not a significant gain, and a
nonzero MCC means there IS signal. The criticism is that accuracy was the wrong metric to
report, not that the work is empty.

Reads the sha256-pinned mirror written by ``build_merzbacher_split.py``.
Writes ``results/merzbacher_baseline_analysis.json``.
"""

from __future__ import annotations

import json
import os
import os.path as osp
from collections import Counter
from glob import glob
from typing import Any

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
FIG4 = osp.join(
    DATA_ROOT, "data/merzbacher2025_fcl/deletionprediction-main/figures/fig4"
)
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "020-cachera-betaxanthin", "results")

#: Their class encoding, from yeast_training_production.py.
CLASS_NAMES = {0: "low", 1: "medium", 2: "high"}


def per_fold_metrics() -> dict[str, dict[str, float]]:
    """Their own per-fold metrics, averaged per model.

    `mcc` is the column the paper never reports and is the one that accounts for imbalance.
    """
    b = pd.read_csv(osp.join(FIG4, "fig4b.csv"))
    cols = ["overall_accuracy", "macro_weighted_f1", "mcc", "class_2_accuracy"]
    return {
        str(m): {c: float(v) for c, v in row.items()}
        for m, row in b.groupby("model")[cols].mean().iterrows()
    }


def gene_level_confusion(path: str) -> dict[str, Any]:
    """Aggregate one model's sample-level predictions to gene level by majority vote.

    Their features are one row per (deletion, flux sample), all sharing the deletion's label;
    `tools/knockout_voting.py` votes them into a single gene-level call, so scoring at the
    row level would not reproduce their reported numbers.
    """
    df = pd.read_csv(path)
    g = df.groupby("knockout_name").agg(
        true=("true_label", "first"),
        pred=("prediction", lambda s: Counter(s).most_common(1)[0][0]),
    )
    cm = {
        f"true_{CLASS_NAMES[t]}": {
            f"pred_{CLASS_NAMES[p]}": int(((g["true"] == t) & (g["pred"] == p)).sum())
            for p in (0, 1, 2)
        }
        for t in (0, 1, 2)
    }
    n = len(g)
    high = g[g["true"] == 2]
    return {
        "n_genes": n,
        "gene_level_accuracy": float((g["true"] == g["pred"]).mean()),
        # The number a majority-class predictor would score -- the only honest reference.
        "majority_class_rate": float((g["true"] == 1).mean()),
        "fraction_predicted_medium": float((g["pred"] == 1).mean()),
        "high_producers_true": int(len(high)),
        "high_producers_found": int((high["pred"] == 2).sum()),
        "high_producer_recall": float((high["pred"] == 2).mean()),
        "confusion_matrix": cm,
    }


def main() -> None:
    models = {
        osp.basename(p).replace("fig4c_", "").replace(".csv", ""): gene_level_confusion(
            p
        )
        for p in sorted(glob(osp.join(FIG4, "fig4c_*.csv")))
    }
    report = {
        "source": "Merzbacher 2025 Zenodo deposit, figures/fig4/ (sha256-pinned mirror)",
        "aggregation": "majority vote over each deletion's flux samples (their knockout_voting)",
        "per_fold_metrics_their_own": per_fold_metrics(),
        "gene_level": models,
        "reading": (
            "Accuracy on a 67%-majority problem cannot separate 'learned something' from "
            "'predicted the majority'. MCC can, and they computed it without reporting it. "
            "Note also that models with higher high-producer accuracy have LOWER overall "
            "accuracy -- the only way to call more highs is to stop calling everything "
            "medium -- so accuracy actively selects against the capability the task is about."
        ),
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = osp.join(RESULTS_DIR, "merzbacher_baseline_analysis.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 78)
    print("MERZBACHER BASELINE, from their own shipped predictions")
    print("=" * 78)
    print(f"{'model':46s} {'acc':>6s} {'MCC':>7s} {'high':>6s}")
    for m, v in sorted(report["per_fold_metrics_their_own"].items()):
        print(
            f"  {m:44s} {v['overall_accuracy']:6.3f} {v['mcc']:7.3f} "
            f"{v['class_2_accuracy']:6.3f}"
        )
    print()
    for m, v in models.items():
        print(
            f"{m}\n  genes={v['n_genes']}  acc={v['gene_level_accuracy']:.3f} "
            f"(majority {v['majority_class_rate']:.3f})  "
            f"predicted MEDIUM {v['fraction_predicted_medium']:.1%}  "
            f"high producers {v['high_producers_found']}/{v['high_producers_true']}"
        )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
