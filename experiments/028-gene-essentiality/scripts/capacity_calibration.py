# experiments/028-gene-essentiality/scripts/capacity_calibration.py
# [[experiments.028-gene-essentiality.scripts.capacity_calibration]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/capacity_calibration

r"""The smallest enzyme-capacity scale under which the growth LP admits growth.

Round 15 found the anchored layer's growth LP infeasible for every genotype: the
capacity ceilings, predicted $k_{cat}$ times the abundance head's nominal abundances,
put the median reaction's upper bound at 0.084 mmol/gDW/h. GECKO handles the same
mismatch with a saturation factor that scales the whole pool until the wild type grows
at its measured rate. This script does the calibration on the layer as built: it
rebuilds the anchored LP cell from a result file at initialization, takes one batch of
genotypes, and for a geometric grid of scales on the capacity ceilings solves the growth
LP through the layer's own forward pass, recording the fraction of genotypes whose
program is feasible and the median growth. The scale to run round 15c at is the smallest
under which every genotype is feasible and the median growth is at least half of what
the stoichiometry-only box gives the same genotypes (round 15b: 0.185 at initialization).

    PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES="" TC_LP_WORKERS=8 PYTHONPATH=$PWD \
        python experiments/028-gene-essentiality/scripts/capacity_calibration.py \
        --result w26_lpbio_h12l1p1_e200_s2700.json
"""

from __future__ import annotations

import argparse
import json
import os.path as osp
import sys

import numpy as np
import torch
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from train_essentiality import (  # noqa: E402
    NODE_EMBEDDING_NAMES,
    RESULTS_DIR,
    BxCell,
    build_dataset,
    build_model_for_cell,
    dataset_paths,
    load_config,
    load_split_file,
)

from torchcell.datamodules import CellDataModule  # noqa: E402

SCALES = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0)
REFERENCE_GROWTH = 0.185  # the stoichiometry-only box's median at initialization


class ScaleRow(BaseModel):
    """The growth LP through the layer at one capacity scale."""

    capacity_scale: float
    n_genotypes: int
    feasible_frac: float
    growth_median: float
    growth_min: float
    growth_max: float


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", default="w26_lpbio_h12l1p1_e200_s2700.json")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    with open(osp.join(RESULTS_DIR, args.result)) as fh:
        result = json.load(fh)
    cell = BxCell(**result["cell"])
    assert cell.split_file is not None
    cfg = load_config()
    NODE_EMBEDDING_NAMES[:] = list(cell.node_embeddings)
    dataset = build_dataset(cell.head)
    split = load_split_file(cell.split_file)
    pinned = {k: {int(i) for i in split[k]} for k in ("train", "val", "test")}
    device = torch.device("cpu")
    model, _, _, _ = build_model_for_cell(cell, cfg, dataset, device)
    layer = model.flux_layer
    assert layer is not None and layer.parameterization == "lp"
    dm = CellDataModule(
        dataset=dataset,
        cache_dir=osp.join(dataset_paths(cell.head)[0], "data_module_cache"),
        split_indices=["phenotype_label_index", "perturbation_count_index"],
        batch_size=args.batch_size,
        random_seed=cell.split_seed,
        num_workers=0,
        pin_memory=False,
        prefetch=False,
        follow_batch=["perturbation_indices", "deletion_indices", "phenotype_values"],
        pinned_split_indices=pinned,
        drop_unpinned=True,
    )
    dm.setup()
    batch = next(iter(dm.test_dataloader())).to(device)
    model.eval()
    rows: list[ScaleRow] = []
    chosen: float | None = None
    for scale in SCALES:
        layer.config.capacity_scale = scale
        with torch.no_grad():
            _, reps = model(dataset.cell_graph.to(device), batch)
        growth = reps["biomass_flux"].reshape(-1).cpu().numpy()
        feasible = growth > 0
        row = ScaleRow(
            capacity_scale=scale,
            n_genotypes=int(growth.shape[0]),
            feasible_frac=float(feasible.mean()),
            growth_median=float(np.median(growth)),
            growth_min=float(growth.min()),
            growth_max=float(growth.max()),
        )
        rows.append(row)
        print(row.model_dump_json(), flush=True)
        if (
            chosen is None
            and row.feasible_frac >= 0.9
            and row.growth_median >= 0.5 * REFERENCE_GROWTH
        ):
            chosen = scale
    out = {
        "result": args.result,
        "reference_growth": REFERENCE_GROWTH,
        "rule": (
            "the smallest scale with at least 90 percent of the batch feasible (the "
            "lethal deletions are infeasible by right) and a median growth of at least "
            "half the stoichiometry-only box's"
        ),
        "chosen_scale": chosen,
        "rows": [r.model_dump() for r in rows],
    }
    path = osp.join(RESULTS_DIR, "capacity_calibration.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"chosen scale: {chosen}\n{path}")


if __name__ == "__main__":
    main()
