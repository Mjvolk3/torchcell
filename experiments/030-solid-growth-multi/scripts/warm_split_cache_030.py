# experiments/030-solid-growth-multi/scripts/warm_split_cache_030.py
# [[experiments.030-solid-growth-multi.scripts.warm_split_cache_030]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/030-solid-growth-multi/scripts/warm_split_cache_030
r"""Write the data module's split cache for an arm, one file per seed, on CPU.

``CellDataModule`` caches its split under ``<build>/data_module_cache/index_seed_<seed>
<pin tag><subset tag>.json`` and checks for it with a bare ``osp.exists``: four DDP ranks
starting on IGB with no cache would each scan the 13.5M-record build and race to write
the same file. This script computes the cache here, for every seed the campaign will
run, under the arm's FINAL pool (S3 minus the essentiality holdout), pin and
unpinned-to-train rule, so the files ``sync_igb_030_build.sh`` mirrors are the ones the
launcher's preflight looks for. The split does not depend on the embeddings, so they
are not loaded.

    PYTHONPATH=$PWD python experiments/030-solid-growth-multi/scripts/warm_split_cache_030.py \\
        --config-name cgt_030_s3_r_tok_embfit_001 '+warm.seeds=[0,1,2]'
"""

from __future__ import annotations

import os
import os.path as osp
import sys

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from arm_030 import (  # noqa: E402
    EXPERIMENT,
    build_dataset,
    make_data_module,
    resolve_arm,
)


@hydra.main(
    version_base=None,
    config_path=osp.join(osp.dirname(__file__), "../conf"),
    config_name="cgt_030_s3_r_tok_embfit_001",
)
def main(cfg: DictConfig) -> None:
    """Build the dataset once, then the data module per seed."""
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    experiment_root = os.environ["EXPERIMENT_ROOT"]
    raw = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(raw, dict)
    seeds = [int(s) for s in raw["warm"]["seeds"]]
    arm = resolve_arm(raw["subset"])
    with open(
        osp.join(experiment_root, EXPERIMENT, "queries/001_multi_measurement.cql")
    ) as f:
        query = f.read()
    dataset, _, _, _ = build_dataset(
        data_root=data_root,
        dataset_root=osp.join(data_root, raw["dataset"]["root_rel"]),
        query=query,
        graph_names=list(raw["cell_dataset"]["graphs"]),
        node_embedding_names=[],
        phenotype_labels=list(raw["cell_dataset"]["phenotype_labels"]),
        dataset_vocabulary=arm.dataset_vocabulary,
    )
    print(
        f"dataset: {len(dataset)} records; pool {len(arm.pool)}; holdout {len(arm.excluded)}"
    )
    dm_cfg = dict(raw["data_module"]) | {"num_workers": 0, "persistent_workers": False}
    for seed in seeds:
        dm = make_data_module(
            dataset, arm, seed, dm_cfg, ["perturbation_indices", "phenotype_values"]
        )
        index_file, details_file = dm._cache_files()
        assert osp.exists(index_file) and osp.exists(details_file), index_file
        print(
            f"seed {seed}: train={len(dm.index.train)} val={len(dm.index.val)} "
            f"test={len(dm.index.test)} val_ess={len(dm.extra_val_datasets['val_ess'])}\n"
            f"  {index_file}\n  {details_file}"
        )
    print("finished: split cache warm")


if __name__ == "__main__":
    main()
