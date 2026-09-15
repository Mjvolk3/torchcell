# experiments/027-betaxanthin-metabolic/scripts/audit_stuck_balance_rows.py
# [[experiments.027-betaxanthin-metabolic.wave7-flux-force-head]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/027-betaxanthin-metabolic/scripts/audit_stuck_balance_rows
r"""Why do the rows that CARRY flux still read a balance ratio of 2.0?

THE QUESTION. A row of S at ratio 2.0 is a metabolite whose every flux-carrying reaction
pushes the same way: all production or all consumption, nothing on the other side. The
turnover-thresholded Genesis rule is decided by these rows, and the wave-8 density term
moved their COUNT (422 to 691 to 810 across density weight 0.0, 0.5, 2.0 at seed 2700)
without moving the ratio at all. Either the other side is available and the objective
simply does not use it, in which case a loss term can fix it, or the other side is boxed
to zero, in which case no loss term on the current bounds can.

WHAT IS CLASSIFIED. For every stuck row of every audited genotype, this takes each reaction
that touches the metabolite and asks whether it COULD carry the opposite-sign contribution
under the box the layer actually built, then attributes each blocked reaction to the first
stage of the box that already excluded that sign:

``gem_bound``
    the GEM's own bound, clamped to the flux scale, already forbids it. Irreversibility as
    the model ships it, and the medium, since exchange bounds are how the medium enters.
``availability``
    ``c_j`` is zero, so the deletion or the GPR closed the reaction. A knockout reaches the
    box this way and nothing downstream can reopen it.
``capacity``
    the enzyme bound ``sum_u kcat_u E_u`` collapsed that side.
``sign_clamp``
    the box survived all three and the DIRECTIONAL tightening zeroed the side, which is the
    only stage a thermodynamic gradient can move, since the tightening reads the layer's own
    delta_r_G.

A row is UNFIXABLE BY ANY LOSS TERM ON THE CURRENT BOUNDS when no reaction touching it can
supply the opposite sign after every stage. A row whose only blocked candidates fall under
``sign_clamp`` is recoverable in principle, because the concentrations that set delta_r_G
are learned.

Run from the repo root, on CPU::

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \\
        experiments/027-betaxanthin-metabolic/scripts/audit_stuck_balance_rows.py \\
        --result wave8_dens_d2p0_boxforce_k100_e80_s2700.json --genotypes 32

Writes ``results/stuck_rows_<stem>.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
from typing import Any

import numpy as np
import torch
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
SCRIPTS_DIR = osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic", "scripts")
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic", "results")
sys.path.insert(0, SCRIPTS_DIR)

from train_bx import (  # noqa: E402
    BxCell,
    build_dataset,
    build_model_for_cell,
    load_config,
    resolve_pinned_test,
)

from torchcell.datamodules import CellDataModule  # noqa: E402
from torchcell.metabolism.flux_layer import FluxLayer  # noqa: E402
from torchcell.timestamp import timestamp  # noqa: E402

#: A reaction carries flux above this, in mmol/gDW/h, matching the audit's
#: FLUX_ACTIVE_RELATIVE_THRESHOLD of 1e-6 * flux_scale.
FLUX_ACTIVE_RELATIVE_THRESHOLD = 1.0e-6
#: A row counts as stuck when its balance ratio is this close to 2.0, the largest value the
#: statistic takes, which happens exactly when every contribution shares a sign.
STUCK_RATIO_TOLERANCE = 1.0e-3
#: Turnover a row must clear to be in the rule's scope at all, as a fraction of flux_scale.
TURNOVER_THRESHOLD = 1.0e-3
#: A bound counts as open on a side when it exceeds this, in mmol/gDW/h.
OPEN_BOUND = 1.0e-9

CAUSES = ("gem_bound", "availability", "capacity", "sign_clamp", "open")


class RowCauseCounts(BaseModel):
    """How the stuck rows of one genotype are caused, counted rather than named."""

    model_config = ConfigDict(extra="forbid")

    n_independent_rows: int
    n_rows_above_turnover: int
    n_stuck_rows: int
    #: Rows with NO reaction able to supply the opposite sign after every stage of the box.
    n_unfixable_rows: int
    #: Of the unfixable rows, the stage that blocked the LEAST severely blocked candidate,
    #: which is the stage a remedy would have to reach.
    unfixable_by_cause: dict[str, int]
    #: Rows where a reaction COULD supply the opposite sign and the flux does not use it.
    n_fixable_by_objective: int
    #: Stuck rows with no other reaction at all in the model: a dead end in the GEM.
    n_rows_with_no_counterpart_reaction: int


class StuckRowReport(BaseModel):
    """The whole check, as written to disk."""

    model_config = ConfigDict(extra="forbid")

    built_at: str
    result_file: str
    checkpoint: str
    cell: dict[str, Any]
    n_genotypes: int
    genotype_selection: str
    flux_active_threshold: float
    turnover_threshold: float
    definitions: dict[str, str]
    per_genotype_mean: RowCauseCounts
    totals: RowCauseCounts
    fraction_unfixable_of_stuck: float = Field(
        description="Rows no loss term can balance on the current bounds, over stuck rows."
    )
    fraction_sign_clamp_of_unfixable: float
    top_metabolites: list[dict[str, Any]] = Field(
        description="The metabolites stuck in the most genotypes, with their cause."
    )


def _dense_s(layer: FluxLayer) -> torch.Tensor:
    return torch.sparse_coo_tensor(
        layer.s_indices, layer.s_values, (layer.n_metabolites, layer.n_reactions)
    ).to_dense()


def _stage_boxes(
    layer: FluxLayer, out: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, ...]:
    """The box at each stage, ``[B, r]`` each: raw, availability, capacity, final.

    Rebuilt from what ``forward`` returns rather than from a second forward pass, so the
    boxes are the ones the reported flux was actually clipped into: ``c_j`` and ``e_g`` come
    out of the layer, and the capacity is formed exactly as ``dynamic_box`` forms it.
    """
    c_j = out["c_j"]
    batch = c_j.shape[0]
    raw_lb = layer.lb.unsqueeze(0).expand(batch, -1).clone()
    raw_ub = layer.ub.unsqueeze(0).expand(batch, -1).clone()
    avail_lb, avail_ub = layer.lb.unsqueeze(0) * c_j, layer.ub.unsqueeze(0) * c_j
    cap_lb, cap_ub = avail_lb.clone(), avail_ub.clone()
    if layer.config.use_enzyme_capacity:
        e_u = layer.unit_abundance(out["e_g"])
        cap = c_j.new_zeros(batch, layer.n_reactions)
        cap.index_add_(1, layer.unit_reaction, layer.kcat_per_h.unsqueeze(0) * e_u)
        capped = torch.where(layer.has_gpr.unsqueeze(0), cap, avail_ub.abs())
        cap_ub = torch.minimum(avail_ub, capped)
        cap_lb = torch.maximum(avail_lb, -capped)
    return raw_lb, raw_ub, avail_lb, avail_ub, cap_lb, cap_ub, out["lb"], out["ub"]


def _classify(
    layer: FluxLayer, out: dict[str, torch.Tensor], s_dense: torch.Tensor
) -> tuple[list[RowCauseCounts], dict[int, dict[str, int]]]:
    """Per genotype: count the stuck rows and attribute each to the stage that blocks it."""
    v = out["v"]
    rows = layer.independent_rows
    residual = layer._s_matmul(v)[:, rows]
    omega = layer.turnover(v)[:, rows]
    eps = layer.config.balance_epsilon
    ratio = residual.abs() / (omega + eps)
    turnover_floor = TURNOVER_THRESHOLD * layer.config.flux_scale
    flux_floor = FLUX_ACTIVE_RELATIVE_THRESHOLD * layer.config.flux_scale
    raw_lb, raw_ub, av_lb, av_ub, cap_lb, cap_ub, fin_lb, fin_ub = _stage_boxes(
        layer, out
    )
    s_rows = s_dense[rows]  # [n_rows, r]
    per_genotype: list[RowCauseCounts] = []
    metabolite_tally: dict[int, dict[str, int]] = {}
    for b in range(v.shape[0]):
        stuck = (ratio[b] >= 2.0 - STUCK_RATIO_TOLERANCE) & (omega[b] > turnover_floor)
        counts = dict.fromkeys(CAUSES, 0)
        n_unfixable = 0
        n_fixable = 0
        n_no_counterpart = 0
        stuck_idx = torch.nonzero(stuck, as_tuple=False).flatten().tolist()
        for r in stuck_idx:
            coeffs = s_rows[r]
            touching = torch.nonzero(coeffs, as_tuple=False).flatten()
            carrying = touching[v[b, touching].abs() > flux_floor]
            if carrying.numel() == 0:
                continue
            # The sign the flux currently pushes; the row needs the opposite one.
            push = torch.sign((coeffs[carrying] * v[b, carrying]).sum())
            wanted = -push
            # A reaction supplies the wanted sign when its box admits a v with
            # sign(S_ij * v) == wanted, i.e. an open bound on the matching side.
            candidates = [int(j) for j in touching.tolist() if int(j) not in set()]
            best = None
            for j in candidates:
                need_positive_v = bool((coeffs[j] * wanted) > 0)

                def open_on(lb: torch.Tensor, ub: torch.Tensor, jj: int = j) -> bool:
                    return bool(
                        ub[b, jj] > OPEN_BOUND
                        if need_positive_v
                        else lb[b, jj] < -OPEN_BOUND
                    )

                if open_on(fin_lb, fin_ub):
                    best = "open"
                    break
                if not open_on(raw_lb, raw_ub):
                    stage = "gem_bound"
                elif not open_on(av_lb, av_ub):
                    stage = "availability"
                elif not open_on(cap_lb, cap_ub):
                    stage = "capacity"
                else:
                    stage = "sign_clamp"
                order = CAUSES.index(stage)
                if best is None or order > CAUSES.index(best):
                    best = stage
            if best is None:
                n_no_counterpart += 1
                n_unfixable += 1
                counts["gem_bound"] += 1
                continue
            if best == "open":
                n_fixable += 1
            else:
                n_unfixable += 1
                counts[best] += 1
                met = int(rows[r])
                tally = metabolite_tally.setdefault(met, dict.fromkeys(CAUSES, 0))
                tally[best] += 1
        per_genotype.append(
            RowCauseCounts(
                n_independent_rows=int(rows.numel()),
                n_rows_above_turnover=int((omega[b] > turnover_floor).sum()),
                n_stuck_rows=len(stuck_idx),
                n_unfixable_rows=n_unfixable,
                unfixable_by_cause=counts,
                n_fixable_by_objective=n_fixable,
                n_rows_with_no_counterpart_reaction=n_no_counterpart,
            )
        )
    return per_genotype, metabolite_tally


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--genotypes", type=int, default=32)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    with open(osp.join(RESULTS_DIR, args.result)) as handle:
        result = json.load(handle)
    cell = BxCell(**result["cell"])
    cfg = load_config()
    ckpt_path = args.checkpoint or result["checkpoint_path"]
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert checkpoint["cell"]["seed"] == cell.seed, (
        f"checkpoint is for seed {checkpoint['cell']['seed']}, result for {cell.seed}."
    )

    dataset = build_dataset()
    pinned, _ = resolve_pinned_test(dataset, cfg)
    device = torch.device("cpu")
    model, _, _, _ = build_model_for_cell(cell, cfg, dataset, device)
    assert model.flux_layer is not None
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
        num_workers=0,
        pin_memory=False,
        prefetch=False,
        follow_batch=["perturbation_indices", "deletion_indices", "phenotype_values"],
        pinned_test_indices=pinned,
    )
    dm.setup()

    # `nn.Module.__getattr__` is typed to return Tensor | Module, so the flux layer
    # arrives as a union; the assert is the narrowing and it also catches an arm that
    # has no flux layer at all, which has nothing to audit.
    layer = model.flux_layer
    assert isinstance(layer, FluxLayer)
    scale = float(layer.config.flux_scale)
    s_dense = _dense_s(layer)
    seen = 0
    per_genotype: list[RowCauseCounts] = []
    tally: dict[int, dict[str, int]] = {}
    cell_graph = dataset.cell_graph.to(device)
    with torch.no_grad():
        for batch in dm.all_dataloader():
            batch = batch.to(device)
            _, reps = model(cell_graph, batch)
            counts, batch_tally = _classify(layer, reps["flux"], s_dense)
            per_genotype.extend(counts)
            for met, sub in batch_tally.items():
                target = tally.setdefault(met, dict.fromkeys(CAUSES, 0))
                for k, n in sub.items():
                    target[k] += n
            seen += len(counts)
            if seen >= args.genotypes:
                break
    per_genotype = per_genotype[: args.genotypes]

    def mean_of(field: str) -> float:
        return float(np.mean([getattr(c, field) for c in per_genotype]))

    totals_causes = dict.fromkeys(CAUSES, 0)
    for c in per_genotype:
        for k, n in c.unfixable_by_cause.items():
            totals_causes[k] += n
    totals = RowCauseCounts(
        n_independent_rows=per_genotype[0].n_independent_rows,
        n_rows_above_turnover=sum(c.n_rows_above_turnover for c in per_genotype),
        n_stuck_rows=sum(c.n_stuck_rows for c in per_genotype),
        n_unfixable_rows=sum(c.n_unfixable_rows for c in per_genotype),
        unfixable_by_cause=totals_causes,
        n_fixable_by_objective=sum(c.n_fixable_by_objective for c in per_genotype),
        n_rows_with_no_counterpart_reaction=sum(
            c.n_rows_with_no_counterpart_reaction for c in per_genotype
        ),
    )
    mean_counts = RowCauseCounts(
        n_independent_rows=per_genotype[0].n_independent_rows,
        n_rows_above_turnover=int(round(mean_of("n_rows_above_turnover"))),
        n_stuck_rows=int(round(mean_of("n_stuck_rows"))),
        n_unfixable_rows=int(round(mean_of("n_unfixable_rows"))),
        unfixable_by_cause={
            k: int(round(v / len(per_genotype))) for k, v in totals_causes.items()
        },
        n_fixable_by_objective=int(round(mean_of("n_fixable_by_objective"))),
        n_rows_with_no_counterpart_reaction=int(
            round(mean_of("n_rows_with_no_counterpart_reaction"))
        ),
    )
    top = sorted(
        (
            {
                "metabolite": str(layer.met_ids[met]),
                "metabolite_index": met,
                "genotypes_stuck": sum(sub.values()),
                "causes": sub,
            }
            for met, sub in tally.items()
        ),
        key=lambda row: -int(str(row["genotypes_stuck"])),
    )[:25]

    report = StuckRowReport(
        built_at=timestamp(),
        result_file=args.result,
        checkpoint=ckpt_path,
        cell=result["cell"],
        n_genotypes=len(per_genotype),
        genotype_selection=(
            "the first genotypes of the all-split loader in dataset order, which is not "
            "the production ranking; the stuck rows are a property of the network and the "
            "box rather than of the prediction, and the per-genotype spread is reported"
        ),
        flux_active_threshold=FLUX_ACTIVE_RELATIVE_THRESHOLD * scale,
        turnover_threshold=TURNOVER_THRESHOLD * scale,
        definitions={
            "stuck_row": (
                "an independent row of S whose balance ratio is within 1e-3 of 2.0, its "
                "largest possible value, and whose turnover clears the rule's threshold"
            ),
            "unfixable": (
                "no reaction touching the metabolite can supply the opposite-sign "
                "contribution under the FINAL box, so no loss term moving v inside that "
                "box can balance the row"
            ),
            "sign_clamp": (
                "the side survived the GEM bound, the availability and the capacity and "
                "was zeroed by the directional tightening, which reads the layer's own "
                "delta_r_G and is therefore the one stage a thermodynamic gradient reaches"
            ),
        },
        per_genotype_mean=mean_counts,
        totals=totals,
        fraction_unfixable_of_stuck=(
            totals.n_unfixable_rows / totals.n_stuck_rows
            if totals.n_stuck_rows
            else float("nan")
        ),
        fraction_sign_clamp_of_unfixable=(
            totals_causes["sign_clamp"] / totals.n_unfixable_rows
            if totals.n_unfixable_rows
            else float("nan")
        ),
        top_metabolites=top,
    )
    out_name = args.out or f"stuck_rows_{osp.splitext(args.result)[0]}.json"
    out_path = osp.join(RESULTS_DIR, out_name)
    with open(out_path, "w") as handle:
        handle.write(report.model_dump_json(indent=2))

    print(f"genotypes audited: {report.n_genotypes}")
    print(f"stuck rows per genotype (mean): {mean_counts.n_stuck_rows}")
    print(f"  of which unfixable on the current bounds: {mean_counts.n_unfixable_rows}")
    print(f"  fixable by the objective: {mean_counts.n_fixable_by_objective}")
    print("unfixable by cause (totals):", totals_causes)
    print(f"fraction unfixable of stuck: {report.fraction_unfixable_of_stuck:.4f}")
    print(
        "fraction of unfixable blocked by the sign clamp: "
        f"{report.fraction_sign_clamp_of_unfixable:.4f}"
    )
    print("wrote", out_path)


if __name__ == "__main__":
    main()
