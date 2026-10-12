# experiments/028-gene-essentiality/scripts/flux_growth_probe.py
# [[experiments.028-gene-essentiality.scripts.flux_growth_probe]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/flux_growth_probe

r"""Does the flux layer's growth column rank essential genes before any training?

Rounds 2, 4 and 6 found the layer's growth column level with the pooled head and blind to
its own $k_{cat}$ table, while FBA's growth ratio, one linear program per deletion, lifts
every model by 0.05. This probe separates the two candidate causes. It rebuilds a flux
cell from a result file, deletes each labeled gene through the same batches training
used, and reads the growth column ``v_{r_2111}`` (a) at initialization, before the
backbone has seen a label, and (b) under the cell's trained weights. For each it scores
the negative growth column against the released label (AUROC on the train, validation and
held-out genes) and against FBA's ratio (Spearman per medium), and counts deletions whose
growth is at zero.

If the initialized column sits at chance and is uncorrelated with the solve, the layer as
built does not compute the quantity FBA computes, whatever the readout learns on top of
it; if it ranks, the readouts failed to keep what the physics gave them.

    python experiments/028-gene-essentiality/scripts/flux_growth_probe.py \
        --result w15_bio_s0_h12l1p1_e200_s2700.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os.path as osp
import sys
from typing import Any

import numpy as np
import torch
from dotenv import load_dotenv
from pydantic import BaseModel
from scipy.stats import spearmanr

load_dotenv()
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from train_essentiality import (  # noqa: E402
    BIOMASS_REACTION,
    CHECKPOINT_DIR,
    FBA_MEDIA,
    FBA_PER_GENE_CSV,
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


class GrowthScores(BaseModel):
    """The growth column scored against the label and against the solve."""

    weights: str
    n_deletions: int
    n_labeled: int
    zero_growth_frac: float
    growth_median: float
    growth_iqr: list[float]
    auroc_train: float
    auroc_val: float
    auroc_test: float
    n_train: int
    n_val: int
    n_test: int
    spearman_vs_fba: dict[str, float]


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    pos = label == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    from scipy.stats import rankdata

    ranks = rankdata(score)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def growth_by_gene(
    model: Any, loader: Any, dataset: Any, gene_ids: list[str]
) -> dict[str, float]:
    """``gene -> v_growth`` over a loader, keyed by each record's single deleted gene."""
    device = next(model.parameters()).device
    model.eval()
    out: dict[str, float] = {}
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, reps = model(dataset.cell_graph.to(device), batch)
            growth = reps["biomass_flux"].reshape(-1).cpu().numpy()
            gene = batch["gene"]
            indices = gene.deletion_indices.cpu().numpy()
            assignment = gene.deletion_indices_batch.cpu().numpy()
            counts = np.bincount(assignment, minlength=int(batch.num_graphs))
            for row in np.flatnonzero(counts == 1):
                (g,) = indices[assignment == row]
                out[gene_ids[int(g)]] = float(growth[row])
    return out


def score(
    weights: str,
    growth: dict[str, float],
    labels: dict[str, int],
    genes: dict[str, list[str]],
    fba: dict[str, list[float]],
) -> GrowthScores:
    labeled = [g for g in growth if g in labels]
    v = np.array([growth[g] for g in labeled])
    y = np.array([labels[g] for g in labeled])
    per_split = {}
    for k in ("train", "val", "test"):
        sel = np.array([g in set(genes[k]) for g in labeled])
        per_split[k] = (auroc(-v[sel], y[sel]), int(sel.sum()))
    with_fba = [g for g in labeled if g in fba]
    sp = {}
    for i, m in enumerate(FBA_MEDIA):
        r = np.array([fba[g][i] for g in with_fba])
        vv = np.array([growth[g] for g in with_fba])
        sp[m] = float(spearmanr(vv, r).statistic)
    return GrowthScores(
        weights=weights,
        n_deletions=len(growth),
        n_labeled=len(labeled),
        zero_growth_frac=float(np.mean(np.abs(v) < 1e-6)),
        growth_median=float(np.median(v)),
        growth_iqr=[float(np.percentile(v, 25)), float(np.percentile(v, 75))],
        auroc_train=per_split["train"][0],
        auroc_val=per_split["val"][0],
        auroc_test=per_split["test"][0],
        n_train=per_split["train"][1],
        n_val=per_split["val"][1],
        n_test=per_split["test"][1],
        spearman_vs_fba=sp,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result", required=True, help="a flux cell's result file name"
    )
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    with open(osp.join(RESULTS_DIR, args.result)) as fh:
        result = json.load(fh)
    cell = BxCell(**result["cell"])
    assert cell.split_file is not None
    cfg = load_config()
    NODE_EMBEDDING_NAMES[:] = list(cell.node_embeddings)
    dataset = build_dataset(cell.head)
    gene_ids = list(dataset.cell_graph["gene"].node_ids)
    split = load_split_file(cell.split_file)
    labels = {g: int(v) for g, v in split["labels"].items()}
    genes = {k: list(v) for k, v in split["genes"].items()}
    pinned = {k: {int(i) for i in split[k]} for k in ("train", "val", "test")}
    with open(FBA_PER_GENE_CSV, newline="") as fh:
        fba = {
            row["gene"]: [float(row[m]) for m in FBA_MEDIA]
            for row in csv.DictReader(fh)
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _, _ = build_model_for_cell(cell, cfg, dataset, device)
    assert model.flux_layer is not None, "the probe reads a flux arm's growth column."
    assert BIOMASS_REACTION in model.flux_layer.rxn_ids
    dm = CellDataModule(
        dataset=dataset,
        cache_dir=osp.join(dataset_paths(cell.head)[0], "data_module_cache"),
        split_indices=["phenotype_label_index", "perturbation_count_index"],
        batch_size=cell.batch_size,
        random_seed=cell.split_seed,
        num_workers=0,
        pin_memory=False,
        prefetch=False,
        follow_batch=["perturbation_indices", "deletion_indices", "phenotype_values"],
        pinned_split_indices=pinned,
        drop_unpinned=True,
    )
    dm.setup()
    loaders = [dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()]

    blocks: list[GrowthScores] = []
    growth_init: dict[str, float] = {}
    for loader in loaders:
        growth_init.update(growth_by_gene(model, loader, dataset, gene_ids))
    blocks.append(score("initialization", growth_init, labels, genes, fba))

    ckpt_path = args.checkpoint or result.get("checkpoint_path")
    if ckpt_path is None:
        ckpt_path = osp.join(CHECKPOINT_DIR, f"{osp.splitext(args.result)[0]}.pt")
    if osp.isfile(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.copy_(checkpoint["parameters"][name].to(param.device))
        growth_trained: dict[str, float] = {}
        for loader in loaders:
            growth_trained.update(growth_by_gene(model, loader, dataset, gene_ids))
        blocks.append(
            score(
                f"trained (epoch {checkpoint['select_epoch']})",
                growth_trained,
                labels,
                genes,
                fba,
            )
        )

    stem = osp.splitext(args.result)[0]
    out = {
        "result": args.result,
        "cell": result["cell"],
        "checkpoint": ckpt_path if osp.isfile(ckpt_path) else None,
        "score": "AUROC of the NEGATIVE growth column against the released label; "
        "Spearman of the growth column against FBA's knockout/wild-type ratio per medium",
        "blocks": [b.model_dump() for b in blocks],
    }
    path = osp.join(RESULTS_DIR, f"flux_growth_probe_{stem}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    for b in blocks:
        print(
            f"{b.weights:28s} n={b.n_labeled} zero-growth {b.zero_growth_frac:.3f} "
            f"median {b.growth_median:.4g} AUROC train {b.auroc_train:.3f} val "
            f"{b.auroc_val:.3f} test {b.auroc_test:.3f} | Spearman vs FBA "
            + " ".join(f"{m} {r:+.3f}" for m, r in b.spearman_vs_fba.items())
        )
    print(path)


if __name__ == "__main__":
    main()
