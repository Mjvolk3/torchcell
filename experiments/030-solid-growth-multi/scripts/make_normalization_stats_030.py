# experiments/030-solid-growth-multi/scripts/make_normalization_stats_030.py
# [[experiments.030-solid-growth-multi.scripts.make_normalization_stats_030]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/030-solid-growth-multi/scripts/make_normalization_stats_030
r"""Fit the label normalizer's constants on the arm's training ENTRY rows and commit them.

On 025 the constants came from ``label_df`` over the training records. 030 keeps every
source entry per record and ``label_df`` holds one arbitrary entry per record (the last
written), so the population the per-entry loss trains on is the ENTRY table of
``closure_recompute_030.py`` (``closure/entries.parquet``, one row per stored entry of
every S3 record) restricted to the records that train under the arm's split: the pool
minus the essentiality holdout minus the pinned validation and test triples. The IGB
compute nodes hold neither that table nor the closure directory, so the constants are
fitted here once and read by every run through ``COOLabelNormalizationTransform``'s
``fit_stats``; the training script recomputes the training set and refuses a file whose
``train_index_sha256`` differs.

    PYTHONPATH=$PWD python experiments/030-solid-growth-multi/scripts/make_normalization_stats_030.py \\
        --config-name cgt_030_s3_r_tok_embfit_001

Writes ``results/<transforms.fit_stats>`` of the named config.
"""

from __future__ import annotations

import os
import os.path as osp
import sys

import hydra
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from arm_030 import (  # noqa: E402
    NormalizationStats030,
    index_sha256,
    resolve_arm,
    results_dir,
)

LABEL_OF_EXP_TYPE = {"fitness": "fitness", "gene interaction": "gene_interaction"}


@hydra.main(
    version_base=None,
    config_path=osp.join(osp.dirname(__file__), "../conf"),
    config_name="cgt_030_s3_r_tok_embfit_001",
)
def main(cfg: DictConfig) -> None:
    """Resolve the arm, select the training entry rows, write the statistics."""
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    raw = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(raw, dict)
    arm = resolve_arm(raw["subset"])
    train = arm.train_records()
    sha = index_sha256(train)
    print(f"train records: {len(train)}  sha256 {sha[:16]}", flush=True)

    build = osp.join(data_root, raw["dataset"]["root_rel"])
    entries_path = osp.join(
        data_root,
        "data/torchcell/experiments/030-solid-growth-multi/closure/entries.parquet",
    )
    entries = pd.read_parquet(entries_path, columns=["idx", "exp_type", "value"])
    entries["label"] = entries["exp_type"].map(LABEL_OF_EXP_TYPE)
    assert entries["label"].notna().all(), "an exp_type has no label"
    rows = entries[entries["idx"].isin(set(train))]
    covered = rows["idx"].nunique()
    assert covered == len(train), (
        f"entry table covers {covered} of {len(train)} training records"
    )

    labels = list(raw["transforms"]["forward_transform"]["normalization"])
    stats: dict[str, dict[str, float]] = {}
    n_rows: dict[str, int] = {}
    for label in labels:
        values = rows.loc[rows["label"] == label, "value"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        assert values.size, f"no training entry rows of {label}"
        stats[label] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "q25": float(np.percentile(values, 25)),
            "q75": float(np.percentile(values, 75)),
        }
        n_rows[label] = int(values.size)
        print(
            f"{label}: n={values.size} mean={stats[label]['mean']:.6f} sd={stats[label]['std']:.6f}"
        )

    out = NormalizationStats030(
        build=build,
        entries_parquet=entries_path,
        subset_name=arm.subset_name,
        split_file=arm.split_file,
        exclude_name=arm.exclude_name,
        unpinned_to_train=arm.unpinned_to_train,
        n_train_records=len(train),
        train_index_sha256=sha,
        n_rows=n_rows,
        stats=stats,
    )
    path = osp.join(results_dir(), raw["transforms"]["fit_stats"])
    with open(path, "w") as f:
        f.write(out.model_dump_json(indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
