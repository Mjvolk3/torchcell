# experiments/031-env-chemgen-inhibitor-tolerance/scripts/reliability_reconciliation.py
# [[experiments.031-env-chemgen-inhibitor-tolerance.scripts.reliability_reconciliation]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/031-env-chemgen-inhibitor-tolerance/scripts/reliability_reconciliation
"""Do the served-SE reliability index and the raw replicate agreement measure the same
thing at different levels, or do they disagree?

``replicate_noise_and_ceilings.py`` reports two reliabilities per condition and they differ
in level by roughly a factor of two on Vanacloig (index 0.703 against a replicate Spearman
of 0.333). The two are not estimates of the same quantity. ``replicate_rho`` is the
agreement of ONE replicate with another, so it is the reliability of a SINGLE measurement.
The served response is the MEAN over that condition's replicates, and averaging k
measurements raises reliability by the Spearman-Brown factor

    rel_mean = k * rel_single / (1 + (k - 1) * rel_single)

so the index should sit ABOVE the replicate Spearman by exactly that much and no more. This
script lifts each condition's ``replicate_rho`` to the mean of its own ``n_samples_mean``
replicates and compares it with the measured index. Where the lifted value explains the
index, the served SE captures the noise the replicates show and either number is usable
once the target is named. Where the index still exceeds it, the served SE is blind to a
component of the noise, and the raw number is the one to quote.

Writes ``results/reliability_reconciliation.csv`` (per condition) and prints the per-dataset
medians with the residual.
"""

from __future__ import annotations

import os
import os.path as osp

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(
    EXPERIMENT_ROOT, "031-env-chemgen-inhibitor-tolerance", "results"
)
DATASETS = ["vanacloig2022", "hillenmeyer2008_hom", "hillenmeyer2008_het"]


def spearman_brown(rel_single: pd.Series, k: pd.Series) -> pd.Series:
    """Reliability of the mean of k measurements each of reliability rel_single."""
    return k * rel_single / (1.0 + (k - 1.0) * rel_single)


def main() -> None:
    frames = []
    for name in DATASETS:
        df = pd.read_csv(osp.join(RESULTS_DIR, f"condition_noise_{name}.csv"))
        df = df[df["replicate_rho"].notna() & df["reliability"].notna()].copy()
        df["dataset"] = name
        # the served response averages this many replicates; below 1 replicate there is
        # nothing to lift, and a non-positive single-measure reliability cannot be lifted
        df = df[(df["n_samples_mean"] >= 1) & (df["replicate_rho"] > 0)]
        df["rel_single_raw"] = df["replicate_rho"]
        df["rel_mean_predicted"] = spearman_brown(
            df["replicate_rho"], df["n_samples_mean"]
        )
        df["index_minus_predicted"] = df["reliability"] - df["rel_mean_predicted"]
        frames.append(
            df[
                [
                    "dataset",
                    "condition",
                    "n_samples_mean",
                    "n_replicates",
                    "rel_single_raw",
                    "rel_mean_predicted",
                    "reliability",
                    "index_minus_predicted",
                    "ceiling_r_truth",
                ]
            ]
        )
    out = pd.concat(frames, ignore_index=True)
    path = osp.join(RESULTS_DIR, "reliability_reconciliation.csv")
    out.to_csv(path, index=False)

    for name, block in out.groupby("dataset"):
        print(
            f"{name}: n={len(block)}  "
            f"raw single {block['rel_single_raw'].median():.3f}  "
            f"k {block['n_samples_mean'].median():.1f}  "
            f"Spearman-Brown predicted index {block['rel_mean_predicted'].median():.3f}  "
            f"measured index {block['reliability'].median():.3f}  "
            f"residual {block['index_minus_predicted'].median():+.3f}"
        )
    print()
    print(
        "A residual near zero means the served SE and the raw replicates measure the same "
        "noise. A large positive residual means the served SE is blind to part of it."
    )
    print(path)


if __name__ == "__main__":
    main()
