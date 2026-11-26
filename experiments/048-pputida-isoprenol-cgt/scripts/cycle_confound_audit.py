# experiments/048-pputida-isoprenol-cgt/scripts/cycle_confound_audit.py
# [[experiments.048-pputida-isoprenol-cgt.scripts.cycle_confound_audit]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/048-pputida-isoprenol-cgt/scripts/cycle_confound_audit
"""How much of the design split's rank score is the campaign's chronology?

Measured first, which is why this script exists: on the 44-record design-split
test set, ranking the strains by DBTL cycle number alone scores Spearman 0.807,
which beats the boosted trees (0.773) and every CGT cell trained so far. Titer
rose across the campaign, as a successful optimization campaign's titer should,
and the design split is random over designs rather than over cycles, so train
and test both span all seven cycles and "a later cycle means a higher titer" is
most of the available rank information.

The model is never handed the cycle. It does not need to be: each cycle used its
own sgRNA sets, so the cycle is recoverable from the guide identity the model
does get. A design-split Spearman is therefore not a clean measure of
biological ranking skill, and this script quantifies the part that is.

Two corrections are computed and neither is a substitute for the other:

stratified Spearman
    Spearman computed WITHIN each cycle and pooled as a size-weighted mean, so
    the between-cycle ordering contributes nothing. This is the honest
    design-split objective. It is noisier than the overall figure, because the
    44 test records split into seven groups, and the script reports each group's
    size so a reader can see how thin they are.
partial Spearman
    The rank correlation of titer and the ranker after both have had the cycle's
    rank regressed out of them. It uses every record rather than splitting them,
    so it is less noisy, but it assumes the cycle's effect is monotone in rank
    and linear once ranked.

The cycle split needs no correction: every one of its test records is cycle 6, so
the cycle has zero variance inside the test set and can carry no rank
information. That is the reason the cycle arm scores so much lower than the
design arm, and it makes the cycle arm the stronger evidence of the two.

Writes ``results/low_data_trial/cycle_confound_audit.csv`` and prints the table.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import rankdata, spearmanr

load_dotenv()

EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
EXPERIMENT = "048-pputida-isoprenol-cgt"
RESULTS = osp.join(EXPERIMENT_ROOT, EXPERIMENT, "results")
TRIAL = osp.join(RESULTS, "low_data_trial")


def stratified_spearman(
    frame: pd.DataFrame, column: str, by: str = "cycle"
) -> tuple[float, dict[int, int]]:
    """Size-weighted mean of the within-group Spearman.

    A group with fewer than three records, or one whose titers are all equal,
    has no defined rank correlation and is dropped from the mean rather than
    counted as zero: a zero would read as "no relationship measured here", and
    the truth is that nothing could be measured there at all. The returned sizes
    are the groups that DID contribute, so the reader can see what the number
    rests on.
    """
    total = 0.0
    weight = 0
    sizes: dict[int, int] = {}
    for group, part in frame.groupby(by):
        if len(part) < 3:
            continue
        if part["titer"].nunique() < 2 or part[column].nunique() < 2:
            continue
        statistic = spearmanr(part["titer"], part[column]).statistic
        if not np.isfinite(statistic):
            continue
        total += statistic * len(part)
        weight += len(part)
        sizes[int(group)] = len(part)
    return (total / weight if weight else float("nan")), sizes


def partial_spearman(frame: pd.DataFrame, column: str, by: str = "cycle") -> float:
    """Spearman of titer and the ranker with the grouping rank regressed out."""
    control = rankdata(frame[by].to_numpy(dtype=float))
    design = np.column_stack([np.ones_like(control), control])

    def residual(values: np.ndarray) -> np.ndarray:
        ranks = rankdata(values)
        coefficients, *_ = np.linalg.lstsq(design, ranks, rcond=None)
        return ranks - design @ coefficients

    left = residual(frame["titer"].to_numpy(dtype=float))
    right = residual(frame[column].to_numpy(dtype=float))
    if np.std(right) == 0.0 or np.std(left) == 0.0:
        return float("nan")
    return float(spearmanr(left, right).statistic)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="design", choices=("design", "cycle"))
    args = parser.parse_args()

    table = pd.read_csv(osp.join(RESULTS, "isoprenol_cgt_record_table.csv"))
    with open(osp.join(RESULTS, "isoprenol_cgt_splits.json")) as handle:
        splits = json.load(handle)
    test_indices = splits[args.split]["test"]
    test = table[table["index"].isin(test_indices)].copy()
    if len(test) != len(test_indices):
        raise ValueError(
            f"the committed record table covers {len(test)} of the "
            f"{len(test_indices)} {args.split}-split test indices, so the audit "
            f"would score a different set than the model was tested on"
        )

    rankers = {
        "cycle number": "cycle",
        "guide count": "n_guides",
        "control titer": "control_titer",
    }
    rows = []
    for label, column in rankers.items():
        overall = spearmanr(test["titer"], test[column]).statistic
        strat, sizes = stratified_spearman(test, column)
        rows.append(
            {
                "ranker": label,
                "test_spearman": float(overall),
                "within_cycle_spearman": strat,
                "partial_spearman": partial_spearman(test, column),
                "contributing_cycles": len(sizes),
                "contributing_records": sum(sizes.values()),
            }
        )

    frame = pd.DataFrame(rows)
    os.makedirs(TRIAL, exist_ok=True)
    path = osp.join(TRIAL, f"cycle_confound_audit_{args.split}.csv")
    frame.to_csv(path, index=False)

    counts = test.groupby("cycle").size().to_dict()
    print(f"{args.split} split, {len(test)} test records")
    print(f"records per cycle in test: {dict(sorted(counts.items()))}")
    print(frame.to_string(index=False))
    print(path)


if __name__ == "__main__":
    main()
