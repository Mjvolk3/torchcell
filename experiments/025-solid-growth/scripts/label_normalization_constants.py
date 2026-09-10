# experiments/025-solid-growth/scripts/label_normalization_constants.py
# [[experiments.025-solid-growth.scripts.label_normalization_constants]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/025-solid-growth/scripts/label_normalization_constants
"""The label-standardization constants each fit population gives on the 025 build.

The trainer z-scores ``gene_interaction`` once, before training, as
``(y - mean) / sd`` with the two constants computed over a population named by
``transforms.fit_on``: ``subset`` uses every record of the arm, which is what 010 did
over its 376,732 records, and ``train`` uses the arm's pinned training split alone.
The inverse transform is applied before any metric is computed, and Pearson is
invariant to an affine map of the labels, so the choice cannot move the reported
correlation; it moves the scale of the point loss relative to the Sinkhorn and graph
terms by the ratio of the two sds squared, and the offset by an amount the readout
bias absorbs.

This script writes the constants for every population the arms use, so the document
can quote them from a result file rather than from a training log:

    subset            all S0 triples                     (010's population, arm R/KL 000)
    train, R          010's training split on 025        (what ``fit_on: train`` gives arm R)
    train, Q          the query-pair-disjoint training split (arm Q/KL 004)
    val, Q / test, Q  the held-out parts of the Q split, for the record

Reads ``processed/label_df.parquet`` of the 025 build through its ``index`` column
and the committed split artifacts under ``experiments/025-solid-growth/results``.

Run from the repo root:
    python experiments/025-solid-growth/scripts/label_normalization_constants.py
"""

import gzip
import json
import os
import os.path as osp

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]

BUILD = osp.join(
    DATA_ROOT, "data/torchcell/experiments/025-solid-growth/001-full-build"
)
RESULTS = osp.join(EXPERIMENT_ROOT, "025-solid-growth", "results")
LABEL = "gene_interaction"


def load_gz(name: str):
    with gzip.open(osp.join(RESULTS, name), "rt") as f:
        return json.load(f)


def main() -> None:
    df = pd.read_parquet(
        osp.join(BUILD, "processed", "label_df.parquet"), columns=["index", LABEL]
    )
    subset = set(load_gz("subset_S0_indices.json.gz"))
    r_split = load_gz("pinned_splits_from_010_seed_42.json.gz")["pinned"]
    q_split = load_gz("query_pair_disjoint_splits_025.json.gz")["splits"]

    populations = {
        "subset": subset,
        "train_R": set(r_split["train"]) & subset,
        "val_R": set(r_split["val"]) & subset,
        "test_R": set(r_split["test"]) & subset,
        "train_Q": set(q_split["train"]) & subset,
        "val_Q": set(q_split["val"]) & subset,
        "test_Q": set(q_split["test"]) & subset,
    }
    out: dict[str, dict[str, float | int]] = {}
    print(f"{'population':<10} {'n':>8} {'mean':>14} {'sd':>12}")
    for name, ids in populations.items():
        values = df.loc[df["index"].isin(ids), LABEL].dropna().to_numpy()
        assert values.size == len(ids), f"{name}: {len(ids)} ids, {values.size} labels"
        # np.std with ddof=0, the estimator COOLabelNormalizationTransform uses.
        stats = {
            "n": int(values.size),
            "mean": float(np.mean(values)),
            "sd": float(np.std(values)),
        }
        out[name] = stats
        print(
            f"{name:<10} {stats['n']:>8,} {stats['mean']:>+14.9f} {stats['sd']:>12.9f}"
        )

    sd_sub, sd_r, sd_q = out["subset"]["sd"], out["train_R"]["sd"], out["train_Q"]["sd"]
    out["point_loss_scale_ratio"] = {
        "train_R_vs_subset": float((sd_sub / sd_r) ** 2),
        "train_Q_vs_subset": float((sd_sub / sd_q) ** 2),
    }
    print(
        "point-loss scale ratio (sd_subset / sd_train)^2: "
        f"R {out['point_loss_scale_ratio']['train_R_vs_subset']:.5f}, "
        f"Q {out['point_loss_scale_ratio']['train_Q_vs_subset']:.5f}"
    )
    path = osp.join(RESULTS, "label_normalization_constants.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
