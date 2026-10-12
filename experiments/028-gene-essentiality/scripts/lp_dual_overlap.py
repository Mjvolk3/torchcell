# experiments/028-gene-essentiality/scripts/lp_dual_overlap.py
# [[experiments.028-gene-essentiality.scripts.lp_dual_overlap]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/lp_dual_overlap

r"""Which bounds bind, and whether they are the same bounds for every genotype.

The forced column of the layer that solves does not fit the training labels through
its ceilings (028 rounds 18 and 19: training loss 1.25 at every learning rate and head
depth). The gradient it has is the dual of the binding bounds, so the question is what
the binding set looks like across genotypes. If the reactions at their ceiling are the
same reactions whatever gene is deleted, the abundance head can move one growth level
for every genotype at once and cannot move the ranking between genotypes, which is what
the labels ask for. This script rebuilds an elastic LP cell at initialization, solves a
batch of held-out genotypes through the layer's own forward pass, and reports:

- per genotype, the number of gene-owned reactions with a nonzero dual (the binding
  set), and its Jaccard overlap with every other genotype's;
- the union of binding sets against the number of gene-owned reactions, and how many
  reactions bind on every genotype;
- the same for the deleted gene's own reactions: whether the deletion's closed reactions
  ever carry a dual (they cannot, their bounds are zero by the deletion), which is the
  formal reason the deleted gene's identity reaches the value only through the
  polytope's shape and never through a learnable bound of its own.

    PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES="" TC_LP_WORKERS=16 PYTHONPATH=$PWD \
        python experiments/028-gene-essentiality/scripts/lp_dual_overlap.py \
        --result w32_lpelbio100_h12l1p1_e200_s2700.json --batch-size 64
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


class DualOverlap(BaseModel):
    """The binding sets of one batch of genotypes at one capacity scale."""

    capacity_scale: float
    n_genotypes: int
    n_gene_owned_reactions: int
    binding_per_genotype_median: float
    binding_per_genotype_min: int
    binding_per_genotype_max: int
    binding_union: int
    binding_on_every_genotype: int
    jaccard_median: float
    jaccard_min: float
    jaccard_p05: float
    #: Reactions in the union that bind on fewer than half the genotypes: the part of
    #: the binding set that is genotype-specific.
    binding_specific: int
    dual_magnitude_median: float


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", default="w32_lpelbio100_h12l1p1_e200_s2700.json")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--scales", default="30,100,1000")
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
    gpr = layer.has_gpr.cpu().numpy()
    rows: list[DualOverlap] = []
    for scale in (float(s) for s in args.scales.split(",")):
        layer.config.capacity_scale = scale
        with torch.no_grad():
            model(dataset.cell_graph.to(device), batch)
        last = layer.lp_solver.last
        assert last is not None
        dual = np.abs(last.d_ub[:, gpr]) + np.abs(last.d_lb[:, gpr])
        binding = dual > 0
        counts = binding.sum(axis=1)
        n = binding.shape[0]
        jac = []
        for i in range(n):
            for j in range(i + 1, n):
                inter = np.logical_and(binding[i], binding[j]).sum()
                union = np.logical_or(binding[i], binding[j]).sum()
                jac.append(inter / union if union else 1.0)
        jac_arr = np.array(jac)
        freq = binding.mean(axis=0)
        rows.append(
            DualOverlap(
                capacity_scale=scale,
                n_genotypes=int(n),
                n_gene_owned_reactions=int(gpr.sum()),
                binding_per_genotype_median=float(np.median(counts)),
                binding_per_genotype_min=int(counts.min()),
                binding_per_genotype_max=int(counts.max()),
                binding_union=int(binding.any(axis=0).sum()),
                binding_on_every_genotype=int(binding.all(axis=0).sum()),
                jaccard_median=float(np.median(jac_arr)),
                jaccard_min=float(jac_arr.min()),
                jaccard_p05=float(np.quantile(jac_arr, 0.05)),
                binding_specific=int(((freq > 0) & (freq < 0.5)).sum()),
                dual_magnitude_median=float(np.median(dual[binding])),
            )
        )
        print(rows[-1].model_dump_json(), flush=True)
    out = osp.join(RESULTS_DIR, "lp_dual_overlap.json")
    with open(out, "w") as fh:
        json.dump(
            {"result": args.result, "rows": [r.model_dump() for r in rows]}, fh, indent=2
        )
    print(out)


if __name__ == "__main__":
    main()
