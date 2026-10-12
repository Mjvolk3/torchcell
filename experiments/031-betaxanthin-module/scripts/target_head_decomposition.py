# experiments/031-betaxanthin-module/scripts/target_head_decomposition.py
# [[experiments.031-betaxanthin-module.scripts.target_head_decomposition]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/031-betaxanthin-module/scripts/target_head_decomposition

r"""How much of a store arm's prediction is the backbone, and how much the slot.

A store arm predicts ``h(z) + a(s)``: the backbone's context head over the cell graph plus
the affine (or the hidden slot) over the gene's store row. Round 203 (transformer dropout
0.3 on the pool through 256) read two of six seeds to a Pearson of 1.0000 against the
same seeds at dropout 0.1 on the pinned predictions, although their training curves
differ from the first epoch: a change inside the transformer left the prediction
untouched, which says the prediction is the slot. This script measures that directly.

For every result file named, the cell is rebuilt from its record exactly as the trainer
built it (``build_model_for_cell``), the checkpoint's weights are loaded, and the test
loader is run twice: once as trained, and once with the affine replaced by zero, which
leaves the backbone's head ``h(z)`` alone. The slot's contribution ``a(s)`` is the
difference. On the pinned genes the script reports the Spearman of each part against the
target, the spread of each part, and the Spearman between the two parts and between each
part and the sum. The rank of the sum is set by the part with the larger spread, and the
share the backbone adds is the difference between the sum's Spearman and the slot's alone.

The store and the ratio table are taken from the result file's ``flux_meta``, so the
environment the trainer reads at import is set from the record and nothing is guessed.

Writes ``results/target_head_decomposition_<name>.json`` in the study. Run from the repo
root on CPU::

    PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD \
        ~/miniconda3/envs/torchcell/bin/python \
        experiments/031-betaxanthin-module/scripts/target_head_decomposition.py \
        --study 031-betaxanthin-module --name w206_aapoolmlp256 \
        --results w206_bxdelaapoolmlp256_h12l1p1_e200_s2700.json ...
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
import time
from typing import Any

import numpy as np
import torch
from dotenv import load_dotenv
from pydantic import BaseModel
from scipy.stats import spearmanr

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]


class PartScores(BaseModel):
    """One part of the prediction (the sum, the backbone's head, or the slot) on the pin."""

    spearman_vs_target: float
    pearson_vs_target: float
    sd: float
    top50_enrichment: float


class CellDecomposition(BaseModel):
    """One cell's prediction split into the backbone's head and the store slot."""

    result: str
    checkpoint: str
    seed: int
    select_epoch: int
    n_pinned: int
    reported_spearman: float
    full: PartScores
    backbone_head: PartScores
    slot: PartScores
    spearman_head_vs_slot: float
    spearman_full_vs_slot: float
    spearman_full_vs_head: float
    sd_ratio_head_over_slot: float
    backbone_adds: float


def _flux_meta_paths(result: dict[str, Any]) -> tuple[str, str]:
    """The store npz and the ratio csv the cell was trained with, from its record."""
    text = json.dumps(result)
    stores = sorted(
        set(
            s
            for s in text.replace("\\/", "/").split('"')
            if "ess_solve_encodings_" in s and s.endswith(".npz")
        )
    )
    csvs = sorted(
        set(
            s
            for s in text.replace("\\/", "/").split('"')
            if "fba_ratio" in s and s.endswith(".csv")
        )
    )
    assert len(stores) == 1, f"the record names {len(stores)} stores: {stores}"
    assert len(csvs) == 1, f"the record names {len(csvs)} ratio tables: {csvs}"
    return stores[0], csvs[0]


def _top50(pred: np.ndarray, target: np.ndarray) -> float:
    """Top-50 enrichment of the true top-10-percent producers, the trainer's rule."""
    n = len(pred)
    top_true = target >= np.quantile(target, 0.9)
    top_pred = np.argsort(-pred)[:50]
    return (
        float(top_true[top_pred].mean() / top_true.mean()) if n >= 50 else float("nan")
    )


def _scores(pred: np.ndarray, target: np.ndarray) -> PartScores:
    return PartScores(
        spearman_vs_target=float(spearmanr(pred, target)[0]),
        pearson_vs_target=float(np.corrcoef(pred, target)[0, 1]),
        sd=float(np.std(pred)),
        top50_enrichment=_top50(pred, target),
    )


class _Zero(torch.nn.Module):
    """Stands in for the affine so the wrapper returns the backbone's head alone."""

    def forward(self, batch: Any, like: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(like)


def decompose(result_path: str, study_results: str) -> CellDecomposition:
    with open(result_path) as fh:
        result = json.load(fh)
    store, csv = _flux_meta_paths(result)
    os.environ["ESS_SOLVE_TABLE"] = store
    os.environ["ESS_FBA_CSV"] = csv
    sys.path.insert(0, osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "scripts"))
    import train_essentiality as te  # noqa: E402

    assert te.SOLVE_ENCODINGS_NPZ == store, (te.SOLVE_ENCODINGS_NPZ, store)
    from torchcell.datamodules import CellDataModule  # noqa: E402

    cell = te.BxCell(**result["cell"])
    cfg = te.load_config()
    te.NODE_EMBEDDING_NAMES[:] = list(cell.node_embeddings)
    ckpt_path = result["checkpoint_path"]
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert checkpoint["cell"]["seed"] == cell.seed, (
        checkpoint["cell"]["seed"],
        cell.seed,
    )
    assert checkpoint["select_epoch"] == result["select_epoch"], (
        checkpoint["select_epoch"],
        result["select_epoch"],
    )
    dataset = te.build_dataset(cell.head, deletion_only=cell.deletion_only_graph)
    assert cell.head != te.ESSENTIALITY_HEAD, "the target heads only"
    pinned, _ = te.resolve_pinned_test(dataset, cfg)
    device = torch.device("cpu")
    model, head_specs, main_head, _ = te.build_model_for_cell(
        cell, cfg, dataset, device
    )
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(checkpoint["parameters"][name].to(param.device))
    mean = checkpoint["target_mean"].to(device)
    sd = checkpoint["target_sd"].to(device)
    dm = CellDataModule(
        dataset=dataset,
        cache_dir=osp.join(te.dataset_paths(cell.head)[0], "data_module_cache"),
        split_indices=["phenotype_label_index", "perturbation_count_index"],
        batch_size=cell.batch_size,
        random_seed=cell.split_seed,
        num_workers=0,
        pin_memory=False,
        prefetch=False,
        follow_batch=["perturbation_indices", "deletion_indices", "phenotype_values"],
        pinned_test_indices=pinned,
    )
    dm.setup()
    dist_head = (
        te.make_dist_head(cell.dist).to(device) if cell.dist == "quantile" else None
    )
    assert hasattr(model, "fba_readout"), "not a store arm: no affine to remove"
    full, target = te._predict(
        model,
        dm.test_dataloader(),
        dataset,
        head_specs,
        main_head,
        mean,
        sd,
        cell.target_log10_molar,
        dist_head,
    )
    affine = model.fba_readout
    model.fba_readout = _Zero()
    head_only, target2 = te._predict(
        model,
        dm.test_dataloader(),
        dataset,
        head_specs,
        main_head,
        mean,
        sd,
        cell.target_log10_molar,
        dist_head,
    )
    model.fba_readout = affine
    assert np.array_equal(np.isfinite(target), np.isfinite(target2))
    mask = te.held_out_mask(dm, dataset, main_head)
    assert main_head in te.SCALAR_HEADS, (
        "one column; the amino acids need a per-column form"
    )
    obs = mask & np.isfinite(target[:, 0])
    f, h, t = full[obs, 0], head_only[obs, 0], target[obs, 0]
    s = f - h
    full_scores = _scores(f, t)
    reported = float(result["test"]["pinned"]["spearman"])
    assert abs(full_scores.spearman_vs_target - reported) < 2e-3, (
        f"rebuilt {full_scores.spearman_vs_target:.4f} against the reported {reported:.4f}"
    )
    slot_scores = _scores(s, t)
    return CellDecomposition(
        result=result_path,
        checkpoint=ckpt_path,
        seed=cell.seed,
        select_epoch=int(result["select_epoch"]),
        n_pinned=int(obs.sum()),
        reported_spearman=reported,
        full=full_scores,
        backbone_head=_scores(h, t),
        slot=slot_scores,
        spearman_head_vs_slot=float(spearmanr(h, s)[0]),
        spearman_full_vs_slot=float(spearmanr(f, s)[0]),
        spearman_full_vs_head=float(spearmanr(f, h)[0]),
        sd_ratio_head_over_slot=float(np.std(h) / np.std(s)),
        backbone_adds=full_scores.spearman_vs_target - slot_scores.spearman_vs_target,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", required=True)
    ap.add_argument("--name", required=True, help="the output file's stem")
    ap.add_argument("--results", nargs="+", required=True)
    args = ap.parse_args()
    assert os.environ.get("PYTHONHASHSEED") == "0", "PYTHONHASHSEED must be 0"
    started = time.time()
    results_dir = osp.join(EXPERIMENT_ROOT, args.study, "results")
    cells = [decompose(osp.join(results_dir, r), results_dir) for r in args.results]
    for c in cells:
        print(
            f"seed {c.seed} ep {c.select_epoch}: full {c.full.spearman_vs_target:+.4f} "
            f"head {c.backbone_head.spearman_vs_target:+.4f} slot {c.slot.spearman_vs_target:+.4f} "
            f"sd head/slot {c.sd_ratio_head_over_slot:.3f} head~slot {c.spearman_head_vs_slot:+.3f} "
            f"full~slot {c.spearman_full_vs_slot:+.4f} backbone adds {c.backbone_adds:+.4f}",
            flush=True,
        )
    keys = ("full", "backbone_head", "slot")
    summary = {
        f"mean_{k}_spearman": float(
            np.mean([getattr(c, k).spearman_vs_target for c in cells])
        )
        for k in keys
    }
    summary.update(
        mean_sd_ratio_head_over_slot=float(
            np.mean([c.sd_ratio_head_over_slot for c in cells])
        ),
        mean_spearman_full_vs_slot=float(
            np.mean([c.spearman_full_vs_slot for c in cells])
        ),
        mean_spearman_head_vs_slot=float(
            np.mean([c.spearman_head_vs_slot for c in cells])
        ),
        mean_backbone_adds=float(np.mean([c.backbone_adds for c in cells])),
        n_backbone_adds_positive=int(sum(c.backbone_adds > 0 for c in cells)),
        n_cells=len(cells),
    )
    out = {
        "generated_by": "experiments/031-betaxanthin-module/scripts/target_head_decomposition.py",
        "name": args.name,
        "statistic": (
            "pinned-test Spearman of the full prediction, of the backbone's context head "
            "alone (the affine replaced by zero) and of their difference, the slot; sd "
            "ratio and rank agreement between the parts"
        ),
        "summary": summary,
        "cells": [c.model_dump() for c in cells],
        "seconds": time.time() - started,
    }
    path = osp.join(results_dir, f"target_head_decomposition_{args.name}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(summary, indent=1))
    print(path)


if __name__ == "__main__":
    main()
