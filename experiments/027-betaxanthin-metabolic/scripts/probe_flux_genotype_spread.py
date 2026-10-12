# experiments/027-betaxanthin-metabolic/scripts/probe_flux_genotype_spread.py
# [[experiments.027-betaxanthin-metabolic.scripts.probe_flux_genotype_spread]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/027-betaxanthin-metabolic/scripts/probe_flux_genotype_spread
"""Does the flux move between genotypes at all, and can a readout on it express anything.

Every readout that goes through the flux layer collapses to a constant, and no readout on
the pooled context does. Measured at the selected epoch of each cell, the validation
prediction spread ratio is 0.37 to 0.46 for the three context-precursor cells and 0.0000 to
0.06 for every flux cell (wave 9 and wave 10 result files). A loss, a spread and a schedule
have each been blamed for that. This probe asks the prior question: how much does the layer's
own output vary across genotypes, before any head reads it.

Three quantities, all across genotypes on the pinned test split:

* **The flux vector.** Per reaction, the standard deviation across genotypes, divided by that
  reaction's mean absolute flux. A reaction whose flux is the same in every strain carries no
  genotype information no matter what reads it. Reported as a distribution over reactions and
  as the count above a relative threshold.
* **What the head actually reads.** The biomass column for the essentiality readout, the
  precursor mixture for the betaxanthin readouts. Its across-genotype standard deviation is
  the ceiling on the spread any affine of it can produce, so comparing it with the target's
  own standard deviation says whether the head is collapsed or starved.
* **The pooled context, as the control.** The same statistic on the backbone's pooled
  perturbed-gene vector, which the context-only readouts consume. If the context varies and
  the flux does not, the collapse is the layer's, not the trainer's.

Run from the worktree root (CPU is fine, it is one forward pass)::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/027-betaxanthin-metabolic/scripts/probe_flux_genotype_spread.py \
        --result wave9_ess_flux_e80_s2700.json --out probe_flux_spread_ess_flux_s2700.json
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys

import numpy as np
import torch
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
SCRIPTS = osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic", "scripts")
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic", "results")
CHECKPOINT_DIR = osp.join(DATA_ROOT, "checkpoints-027")
sys.path.insert(0, SCRIPTS)

from train_bx import (  # noqa: E402
    BxCell,
    build_dataset,
    build_model_for_cell,
    load_config,
    perturbed_gene_pool,
    resolve_pinned_test,
)

from torchcell.datamodules import CellDataModule  # noqa: E402

#: A reaction counts as carrying genotype information when its across-genotype standard
#: deviation is at least this fraction of its own mean absolute flux. 1e-3 is the same
#: relative scale the feasibility audit uses for a turning-over balance row.
RELATIVE_SPREAD_THRESHOLD = 1e-3


class SpreadStats(BaseModel):
    """Distribution of an across-genotype spread over reactions or context dimensions."""

    model_config = ConfigDict(extra="forbid")
    n: int
    median: float
    p90: float
    maximum: float
    n_above_threshold: int = Field(
        description="Entries whose across-genotype sd is at least 1e-3 of their own mean magnitude."
    )


class ProbeReport(BaseModel):
    """What one trained cell's flux, head input and prediction do across genotypes."""

    model_config = ConfigDict(extra="forbid")
    result_file: str
    checkpoint: str
    arm: str
    seed: int
    head: str
    select_epoch: int | None
    n_genotypes: int
    n_reactions: int
    flux_relative_spread: SpreadStats = Field(
        description="Per reaction, sd across genotypes over mean |flux|."
    )
    flux_absolute_sd: SpreadStats = Field(
        description="Per reaction, sd across genotypes."
    )
    flux_sd_over_scale: SpreadStats = Field(
        description=(
            "Per reaction, sd across genotypes divided by the layer's flux_scale. THIS is "
            "the headline: dividing instead by a reaction's own mean, as "
            "flux_relative_spread does, reports noise on near-zero fluxes as a huge spread "
            "(median 6.7 on a cell whose absolute spread is 2.5e-5 against a scale of 100)."
        )
    )
    flux_scale: float
    context_relative_spread: SpreadStats = Field(
        description="The same statistic on the pooled perturbed-gene context, the control."
    )
    head_input_sd: float = Field(
        description="Across-genotype sd of the scalar the head reads, in its own units."
    )
    head_input_mean_abs: float
    head_input_name: str
    prediction_sd: float = Field(
        description="Across-genotype sd of the head's own output."
    )
    target_sd: float = Field(
        description="Across-genotype sd of the standardized target, ~1."
    )
    prediction_sd_ratio: float


def _stats(values: np.ndarray, magnitude: np.ndarray | None = None) -> SpreadStats:
    finite = values[np.isfinite(values)]
    if magnitude is None:
        above = int((finite > 0).sum())
    else:
        scale = np.where(np.abs(magnitude) > 0, np.abs(magnitude), np.nan)
        rel = values / scale
        above = int(np.nansum(rel >= RELATIVE_SPREAD_THRESHOLD))
    return SpreadStats(
        n=int(finite.size),
        median=float(np.median(finite)) if finite.size else float("nan"),
        p90=float(np.quantile(finite, 0.9)) if finite.size else float("nan"),
        maximum=float(finite.max()) if finite.size else float("nan"),
        n_above_threshold=above,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result", required=True, help="Result file name under results/."
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out", required=True, help="Output file name under results/.")
    args = parser.parse_args()

    with open(osp.join(RESULTS_DIR, args.result)) as fh:
        result = json.load(fh)
    cell = BxCell(**result["cell"])
    cfg = load_config()
    ckpt_path = args.checkpoint or result.get("checkpoint_path")
    if ckpt_path is None:
        ckpt_path = osp.join(CHECKPOINT_DIR, f"{osp.splitext(args.result)[0]}.pt")
    assert osp.isfile(ckpt_path), f"no checkpoint at {ckpt_path}."
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    dataset = build_dataset()
    pinned, _ = resolve_pinned_test(dataset, cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _head_specs, main_head, _ = build_model_for_cell(cell, cfg, dataset, device)
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(checkpoint["parameters"][name].to(param.device))
    model.eval()

    dm = CellDataModule(
        dataset=dataset,
        cache_dir=osp.join(DATA_ROOT, cfg.data.dataset_subdir, "data_module_cache"),
        split_indices=["phenotype_label_index", "perturbation_count_index"],
        batch_size=cell.batch_size,
        random_seed=cell.split_seed,
        num_workers=cell.num_workers,
        pin_memory=False,
        prefetch=False,
        follow_batch=["perturbation_indices", "deletion_indices", "phenotype_values"],
        pinned_test_indices=pinned,
    )
    dm.setup()

    flux_rows: list[np.ndarray] = []
    context_rows: list[np.ndarray] = []
    head_rows: list[np.ndarray] = []
    pred_rows: list[np.ndarray] = []
    head_input_name = "none"
    with torch.no_grad():
        for batch in dm.test_dataloader():
            batch = batch.to(device)
            _predictions, reps = model(dataset.cell_graph.to(device), batch)
            if "flux" in reps:
                flux_rows.append(reps["flux"]["v"].detach().cpu().numpy())
            context = perturbed_gene_pool(
                reps["H_genes_pert"],
                batch["gene"].perturbation_indices,
                batch["gene"].perturbation_indices_batch,
            )
            context_rows.append(context.detach().cpu().numpy())
            for key in ("precursor_log10_c", "context_log_c", "growth"):
                if key in reps:
                    head_input_name = key
                    value = reps[key]
                    head_rows.append(
                        value.reshape(value.shape[0], -1).detach().cpu().numpy()
                    )
                    break
            out = reps["head_outputs"][main_head]
            pred_rows.append(
                out.reshape(out.shape[0], -1)[:, :1].detach().cpu().numpy()
            )

    flux = np.concatenate(flux_rows, axis=0) if flux_rows else np.zeros((0, 0))
    context = np.concatenate(context_rows, axis=0)
    preds = np.concatenate(pred_rows, axis=0)
    head_input = np.concatenate(head_rows, axis=0) if head_rows else preds

    flux_sd = flux.std(axis=0) if flux.size else np.zeros(0)
    flux_mag = np.abs(flux).mean(axis=0) if flux.size else np.zeros(0)
    context_sd = context.std(axis=0)
    context_mag = np.abs(context).mean(axis=0)
    scale = (
        float(model.flux_layer.config.flux_scale)
        if model.flux_layer is not None
        else float("nan")
    )
    report = ProbeReport(
        result_file=args.result,
        checkpoint=ckpt_path,
        arm=cell.arm,
        seed=cell.seed,
        head=main_head,
        select_epoch=result.get("select_epoch"),
        n_genotypes=int(preds.shape[0]),
        n_reactions=int(flux.shape[1]) if flux.size else 0,
        flux_relative_spread=_stats(
            np.divide(flux_sd, np.where(flux_mag > 0, flux_mag, np.nan)), None
        )
        if flux.size
        else _stats(np.zeros(0)),
        flux_absolute_sd=_stats(flux_sd, flux_mag)
        if flux.size
        else _stats(np.zeros(0)),
        flux_sd_over_scale=_stats(flux_sd / scale)
        if flux.size
        else _stats(np.zeros(0)),
        flux_scale=scale,
        context_relative_spread=_stats(
            np.divide(context_sd, np.where(context_mag > 0, context_mag, np.nan)), None
        ),
        head_input_sd=float(head_input.std(axis=0).mean()),
        head_input_mean_abs=float(np.abs(head_input).mean()),
        head_input_name=head_input_name,
        prediction_sd=float(preds.std()),
        target_sd=1.0,
        prediction_sd_ratio=float(preds.std()),
    )
    with open(osp.join(RESULTS_DIR, args.out), "w") as fh:
        json.dump(report.model_dump(), fh, indent=2)
    print(json.dumps(report.model_dump(), indent=2))
    print(osp.join(RESULTS_DIR, args.out))


if __name__ == "__main__":
    main()
