# experiments/020-cachera-betaxanthin/scripts/evaluate_merzbacher_head_to_head.py
# [[experiments.020-cachera-betaxanthin.merzbacher-comparison]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/020-cachera-betaxanthin/scripts/evaluate_merzbacher_head_to_head
"""Score OUR betaxanthin predictions against Merzbacher 2025, on their own test genes.

Consumes the per-gene test dumps written by `train_cgt_multitask._dump_test_predictions`
(enabled by `trainer.dump_test_predictions` on `delta_grid_betaxanthin`) and produces
`results/merzbacher_head_to_head.json`.

THE PROTOCOL WAS FIXED BEFORE WE HAD A MODEL, in
[[experiments.020-cachera-betaxanthin.merzbacher-comparison]], so it is not chosen after
seeing our number. This script is that protocol, executable:

1. GROUND TRUTH = THEIR RELEASED LABELS, never labels we re-derive. Their rule min-max scales
   production to [0,1] then cuts at 0.40 / 0.65, so the cuts depend on the observed extremes;
   applied to our (larger) copy of the same screen it gives 107/476/56 against their
   109/431/100 -- 81.2 % agreement, and we would call barely half as many high producers.
   Same rule, same screen, different classes. Re-deriving would compare different TASKS.
2. OUR PREDICTIONS ARE BINNED WITH THE SCALE FITTED ON THE TRAIN POOL ONLY. Fitting the
   min/max on the test genes leaks the class distribution, which on a 67 %-majority problem is
   most of the answer.
3. PRIMARY METRICS ARE IMBALANCE-IMMUNE: Spearman, and top-k enrichment on high producers.
   MCC is reported next to accuracy -- they computed MCC and did not report it.
4. Accuracy, per-class accuracy and high-producer recall are reported too, so the comparison
   also lands on their terms.
5. THE FRACTION OF GENES WE CALL MEDIUM IS REPORTED. Their best model calls 607/640 (94.8 %)
   medium. If ours is also ~95 %, we reproduced their failure mode rather than beat it, and
   the report has to say so -- which is why this number is computed rather than left to
   whoever reads the confusion matrix.

WHY IT AGGREGATES OVER RUNS. A single run's test number carries the same replicate noise as
its val number (sigma = 0.030 on this arm), so one dump is not a result. Dumps are grouped by
their `wandb_tags` setting label and reported as mean +- SEM over seeds, with the pooled
"all runs" row alongside. A head-to-head quoted from the best single dump would be the
max-over-96 statistic, i.e. about 2 sigma of pure selection.

    python experiments/020-cachera-betaxanthin/scripts/evaluate_merzbacher_head_to_head.py
    python .../evaluate_merzbacher_head_to_head.py --dumps /path/to/test-predictions
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import os.path as osp
import statistics as st
import sys
from typing import Any

import numpy as np
from dotenv import load_dotenv
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import confusion_matrix, matthews_corrcoef

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from build_merzbacher_split import (  # noqa: E402
    MERZBACHER_THRESHOLDS,
    load_merzbacher_split,
    read_cachera_raw,
)

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "020-cachera-betaxanthin", "results")
SPLIT_PATH = osp.join(RESULTS_DIR, "merzbacher_nested_split.json")
BASELINE_PATH = osp.join(RESULTS_DIR, "merzbacher_baseline_analysis.json")
#: Their best model at gene level, from their own shipped per-flux-sample predictions.
THEIR_BEST = "RandomForestClassifier_Resampled"


def setting_label(dump: dict[str, Any]) -> str:
    """The grid cell a dump came from, read off the tags the grid runner stamped."""
    tags = [t for t in dump.get("wandb_tags", []) if t.startswith("s") and "_lam" in t]
    return tags[0] if tags else "unlabelled"


def load_dumps(path: str) -> list[dict[str, Any]]:
    """Read every prediction dump under `path` that carries a betaxanthin head."""
    files = sorted(glob.glob(osp.join(path, "*.json"))) if osp.isdir(path) else [path]
    out = []
    for f in files:
        with open(f) as fh:
            payload = json.load(fh)
        if "betaxanthin" not in payload.get("predictions", {}):
            continue
        payload["_file"] = f
        out.append(payload)
    return out


def gene_predictions(dump: dict[str, Any]) -> dict[str, float]:
    """{systematic ORF -> predicted betaxanthin} for one run's test split.

    A record can carry several deletion genes only if the build ever emitted a multi-deletion
    strain; the Cachera screen is single deletions, so a record naming anything other than
    exactly one gene is a build defect and is RAISED rather than silently averaged away.
    """
    preds: dict[str, float] = {}
    for rec in dump["predictions"]["betaxanthin"]:
        genes = rec["genes"]
        if len(genes) != 1:
            raise ValueError(
                f"record {rec['record_index']} names {len(genes)} deletion genes "
                f"({genes[:5]}); the Cachera screen is single deletions"
            )
        preds[genes[0]] = float(rec["pred"][0])
    return preds


def scale_bins(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Their 3-class rule applied with an EXTERNALLY supplied scale (train pool).

    `merzbacher_bins` min-max scales whatever array it is given, which is right for
    reproducing their labels and wrong for scoring ours -- scaling our test predictions by
    their own min/max would let the test set set its own class boundaries.
    """
    t1, t2 = MERZBACHER_THRESHOLDS
    scaled = (values - lo) / (hi - lo)
    return np.where(scaled < t1, 0, np.where(scaled < t2, 1, 2))


def score(pred_class: np.ndarray, true_class: np.ndarray) -> dict[str, Any]:
    """Their metrics plus the two they had and did not report."""
    cm = confusion_matrix(true_class, pred_class, labels=[0, 1, 2])
    n_high = int((true_class == 2).sum())
    return {
        "n_genes": int(len(true_class)),
        "accuracy": float((pred_class == true_class).mean()),
        "majority_class_rate": float(
            np.bincount(true_class, minlength=3).max() / len(true_class)
        ),
        "mcc": float(matthews_corrcoef(true_class, pred_class)),
        "fraction_predicted_medium": float((pred_class == 1).mean()),
        "high_producers_true": n_high,
        "high_producers_found": int(cm[2, 2]),
        "high_producer_recall": float(cm[2, 2] / n_high) if n_high else None,
        "confusion_matrix": cm.tolist(),
    }


def topk_enrichment(pred: np.ndarray, true_class: np.ndarray, k: int) -> float:
    """Fraction of the top-k predicted genes that are TRUE high producers.

    The imbalance-immune reading of the task strain design actually cares about: if you can
    only build k strains, how many of them are worth building? Chance is the base rate.
    """
    order = np.argsort(-pred)[:k]
    return float((true_class[order] == 2).mean())


def evaluate_one(
    dump: dict[str, Any],
    truth: dict[str, int],
    lo: float,
    hi: float,
    raw: dict[str, float],
) -> dict[str, Any]:
    preds = gene_predictions(dump)
    # Intersect with `raw` as well as with their labels: the regression columns are scored
    # against the RAW screen value, and their test list contains at least one gene our screen
    # cannot supply (IPP1/YBR011C is essential, so no deletion strain exists and no screen
    # could have measured it). Dropping it here is reported by `n_scored`, not silent.
    genes = sorted(set(preds) & set(truth) & set(raw))
    p = np.array([preds[g] for g in genes], dtype=float)
    t = np.array([truth[g] for g in genes], dtype=int)
    obs = np.array([raw[g] for g in genes], dtype=float)
    out = score(scale_bins(p, lo, hi), t)
    out.update(
        {
            "file": osp.basename(dump["_file"]),
            "setting": setting_label(dump),
            "seed": dump.get("seed"),
            "dist": dump.get("dist"),
            "n_scored": len(genes),
            # REGRESSION -- what they tried and abandoned ("challenging with the limited
            # number of knockouts at the high and low ends"). Against the raw screen value,
            # not the bins, because the bins throw away the ordering WITHIN a class, which is
            # exactly the ordering strain design needs.
            "spearman_vs_raw": float(spearmanr(p, obs).statistic),
            "pearson_vs_raw": float(pearsonr(p, obs).statistic),
            "topk_50_high_precision": topk_enrichment(p, t, 50),
            "topk_100_high_precision": topk_enrichment(p, t, 100),
        }
    )
    return out


def aggregate(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, Any]:
    """Mean +- SEM over runs, for the metrics a single run cannot pin down."""
    out: dict[str, Any] = {"n_runs": len(rows)}
    for key in keys:
        vals = [r[key] for r in rows if r.get(key) is not None]
        if not vals:
            continue
        mean = st.mean(vals)
        sd = st.stdev(vals) if len(vals) > 1 else float("nan")
        out[key] = {
            "mean": mean,
            "sd": sd,
            "sem": sd / len(vals) ** 0.5 if len(vals) > 1 else None,
            "n": len(vals),
        }
    return out


AGG_KEYS = (
    "spearman_vs_raw",
    "pearson_vs_raw",
    "mcc",
    "accuracy",
    "high_producer_recall",
    "fraction_predicted_medium",
    "topk_50_high_precision",
    "topk_100_high_precision",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dumps",
        default=osp.join(DATA_ROOT, "test-predictions"),
        help="prediction-dump file or directory (default: $DATA_ROOT/test-predictions)",
    )
    ap.add_argument(
        "--out", default=osp.join(RESULTS_DIR, "merzbacher_head_to_head.json")
    )
    args = ap.parse_args()

    dumps = load_dumps(args.dumps)
    if not dumps:
        raise SystemExit(f"no betaxanthin prediction dumps under {args.dumps}")

    with open(SPLIT_PATH) as fh:
        split = json.load(fh)
    with open(BASELINE_PATH) as fh:
        baseline = json.load(fh)

    # Recipe step 1: ground truth is THEIR released label, keyed by their gene name.
    _, their_val = load_merzbacher_split()
    # `yeast_production_validation_split.csv` keys its gene column `knockout` (the TEST csv
    # uses `name`); the released class distribution is 109/431/100 -- assert it, because a
    # silently mis-keyed ground truth is the one error that would make every number below
    # look plausible and be wrong.
    truth = {str(r["knockout"]): int(r["label"]) for _, r in their_val.iterrows()}
    counts = {c: sum(1 for v in truth.values() if v == c) for c in (0, 1, 2)}
    assert counts == {0: 109, 1: 431, 2: 100}, (
        f"their released labels changed: {counts}"
    )

    raw, _ = read_cachera_raw()

    # Recipe step 2: the bin scale is fitted on the TRAIN POOL, never on the test genes.
    pool = [raw[g] for g in split["split"]["train_val_pool"] if g in raw]
    lo, hi = float(np.nanmin(pool)), float(np.nanmax(pool))

    rows = [evaluate_one(d, truth, lo, hi, raw) for d in dumps]
    by_setting: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_setting.setdefault(r["setting"], []).append(r)

    payload = {
        "protocol": "[[experiments.020-cachera-betaxanthin.merzbacher-comparison]]",
        "ground_truth": "Merzbacher released labels (yeast_production_validation_split.csv)",
        "bin_scale_fitted_on": {
            "split": "train_val_pool",
            "n_genes": len(pool),
            "min": lo,
            "max": hi,
            "thresholds": list(MERZBACHER_THRESHOLDS),
        },
        "their_baseline_gene_level": baseline["gene_level"].get(THEIR_BEST),
        "their_best_model": THEIR_BEST,
        "pooled": aggregate(rows, AGG_KEYS),
        "by_setting": {
            k: aggregate(v, AGG_KEYS) for k, v in sorted(by_setting.items())
        },
        "per_run": rows,
    }
    os.makedirs(osp.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)

    their = payload["their_baseline_gene_level"] or {}
    print(
        f"\nwrote {args.out}   ({len(rows)} runs, {rows[0]['n_scored']} genes scored)"
    )
    print(
        f"\n{'setting':<44s} {'n':>3s} {'spearman':>16s} {'MCC':>16s} "
        f"{'acc':>16s} {'high-recall':>16s}"
    )
    for name, agg in sorted(payload["by_setting"].items()):

        def cell(k: str) -> str:
            a = agg.get(k)
            if not a:
                return f"{'--':>16s}"
            sem = f"+-{a['sem']:.3f}" if a["sem"] is not None else ""
            return f"{a['mean']:.3f}{sem:>10s}"

        print(
            f"{name:<44s} {agg['n_runs']:>3d} {cell('spearman_vs_raw')} {cell('mcc')} "
            f"{cell('accuracy')} {cell('high_producer_recall')}"
        )
    # Their gene-level block carries no MCC -- they computed it per FOLD (fig4b.csv) and never
    # reported it; the note records 0.232 for this model. Printed from the per-fold table so
    # the comparison is against a number they actually produced.
    their_mcc = baseline["per_fold_metrics_their_own"].get(THEIR_BEST, {}).get("mcc")
    print(f"\nMERZBACHER {THEIR_BEST} (their own shipped predictions, gene level):")
    print(
        f"  accuracy {their.get('gene_level_accuracy'):.3f} vs majority rate "
        f"{their.get('majority_class_rate'):.3f} | MCC {their_mcc:.3f} | "
        f"high-producer recall {their.get('high_producer_recall'):.3f} | "
        f"fraction called medium {their.get('fraction_predicted_medium'):.3f}"
    )
    print(
        "  regression: NOT AVAILABLE to them -- they tried and abandoned it, so the "
        "spearman column above has no counterpart."
    )


if __name__ == "__main__":
    main()
