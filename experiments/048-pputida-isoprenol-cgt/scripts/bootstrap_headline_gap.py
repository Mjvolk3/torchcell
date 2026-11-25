# experiments/048-pputida-isoprenol-cgt/scripts/bootstrap_headline_gap.py
# [[experiments.048-pputida-isoprenol-cgt.scripts.bootstrap_headline_gap]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/048-pputida-isoprenol-cgt/scripts/bootstrap_headline_gap
"""Is the CGT's lead over the nulls on the held-out cycle larger than noise?

The trial's headline is that on the cycle split the CGT scores Spearman about
0.62 against boosted trees at 0.510, a gap near 0.11. The spread across seeds is
small, about 0.024, but that spread is the wrong uncertainty: the cycle split is
deterministic, so every seed scores the SAME 59 cycle-6 records and the seed
spread measures initialization variance alone. It says nothing about the test
set being 59 records drawn from one cycle of one campaign. For a Spearman near
0.62 at n = 59 the sampling standard error is roughly 0.08, which is most of the
gap, so the seed spread standing in for a sampling error would be the difference
between a result and a claim.

This resamples the RECORDS instead. Both models predicted the same 59 records in
the same order, so one bootstrap draw of record positions indexes the observed
vector, the model's predictions and each null's together, and the paired
difference is recomputed inside every draw. Pairing is what makes the test
sensitive: the two models succeed and fail on largely the same strains, so the
difference is far better determined than either score alone, and resampling them
jointly keeps that.

Reported per null: the paired gap, its percentile interval, and the share of
draws in which the CGT loses. That last number is the one to read, because it is
the probability the ordering reverses under resampling and needs no distributional
assumption at all.

Seeds are pooled by averaging each record's prediction across the available
seeds before resampling. That is deliberate: it reports the gap for the
configuration rather than for one initialization, and averaging predictions
across seeds is what an ensemble of this configuration would do anyway.

    python experiments/048-pputida-isoprenol-cgt/scripts/bootstrap_headline_gap.py \
        --tag headline --split cycle
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import os.path as osp

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import spearmanr

load_dotenv()

EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
TRIAL = osp.join(
    EXPERIMENT_ROOT, "048-pputida-isoprenol-cgt", "results", "low_data_trial"
)

NULLS = ("boosted_trees", "ridge")


def load_predictions(
    tag: str, split: str
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], list[int]]:
    """Pool the per-record predictions of every seed of one configuration.

    Raises:
        ValueError: If the reports disagree about which records the test split
            holds, or if a report predates prediction saving. Averaging
            predictions over seeds that scored different records would produce a
            number with no referent, so it is refused rather than aligned.
    """
    paths = sorted(glob.glob(osp.join(TRIAL, f"train_{split}_{tag}_seed*.json")))
    if not paths:
        raise ValueError(
            f"no reports match train_{split}_{tag}_seed*.json under {TRIAL}. "
            f"The bootstrap needs runs whose reports carry test_predicted, "
            f"which the trainer has saved since 2026-11-25; earlier runs have "
            f"metrics only and cannot be resampled."
        )
    observed: np.ndarray | None = None
    indices: list[int] | None = None
    model_rows: list[np.ndarray] = []
    null_rows: dict[str, list[np.ndarray]] = {name: [] for name in NULLS}
    for path in paths:
        with open(path) as handle:
            report = json.load(handle)
        if "test_predicted" not in report:
            raise ValueError(
                f"{osp.basename(path)} has no test_predicted, so it was written "
                f"before the trainer saved predictions. Re-run that cell rather "
                f"than dropping it, so every seed in the pool is the same "
                f"configuration."
            )
        if indices is None:
            indices = [int(i) for i in report["test_indices"]]
            observed = np.asarray(report["test_observed"], dtype=float)
        elif [int(i) for i in report["test_indices"]] != indices:
            raise ValueError(
                f"{osp.basename(path)} scored a different set of test records "
                f"than the earlier reports of the same tag. On the cycle split "
                f"the partition is deterministic, so this means the split "
                f"changed between runs and the seeds are not comparable."
            )
        model_rows.append(np.asarray(report["test_predicted"], dtype=float))
        for name in NULLS:
            null_rows[name].append(
                np.asarray(
                    report["lower_order_nulls"][name]["test_predictions"], dtype=float
                )
            )
    assert observed is not None and indices is not None
    model = np.mean(np.vstack(model_rows), axis=0)
    nulls = {name: np.mean(np.vstack(rows), axis=0) for name, rows in null_rows.items()}
    return observed, model, nulls, indices


def paired_bootstrap(
    observed: np.ndarray,
    model: np.ndarray,
    null: np.ndarray,
    draws: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Resample records and recompute the paired Spearman gap in each draw.

    A draw whose resampled observed values are constant has no defined rank
    correlation and is skipped rather than scored as zero. At n = 59 that is
    vanishingly rare, but counting it as a tie would bias the reversal rate
    toward the model.
    """
    n = len(observed)
    gaps: list[float] = []
    for _ in range(draws):
        pick = rng.integers(0, n, size=n)
        y = observed[pick]
        if np.unique(y).size < 2:
            continue
        a = spearmanr(y, model[pick]).statistic
        b = spearmanr(y, null[pick]).statistic
        if np.isfinite(a) and np.isfinite(b):
            gaps.append(a - b)
    array = np.asarray(gaps)
    point = float(
        spearmanr(observed, model).statistic - spearmanr(observed, null).statistic
    )
    return {
        "model_spearman": float(spearmanr(observed, model).statistic),
        "null_spearman": float(spearmanr(observed, null).statistic),
        "gap": point,
        "gap_mean": float(array.mean()),
        "ci_low": float(np.percentile(array, 2.5)),
        "ci_high": float(np.percentile(array, 97.5)),
        "reversal_rate": float((array <= 0.0).mean()),
        "draws_used": int(array.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="headline")
    parser.add_argument("--split", default="cycle", choices=("design", "cycle"))
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    observed, model, nulls, indices = load_predictions(args.tag, args.split)
    rng = np.random.default_rng(args.seed)
    rows = []
    for name, null in nulls.items():
        result = paired_bootstrap(observed, model, null, args.draws, rng)
        result["null"] = name
        rows.append(result)

    frame = pd.DataFrame(rows)[
        [
            "null",
            "model_spearman",
            "null_spearman",
            "gap",
            "ci_low",
            "ci_high",
            "reversal_rate",
            "draws_used",
        ]
    ]
    path = osp.join(TRIAL, f"bootstrap_gap_{args.split}_{args.tag}.csv")
    frame.to_csv(path, index=False)

    print(
        f"{args.split} split, tag {args.tag}, {len(indices)} test records, "
        f"predictions pooled over seeds"
    )
    print(frame.to_string(index=False))
    print(path)


if __name__ == "__main__":
    main()
