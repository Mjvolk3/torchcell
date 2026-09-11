# experiments/027-betaxanthin-metabolic/scripts/audit_top_strain_feasibility.py
# [[experiments.027-betaxanthin-metabolic.wave6]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/027-betaxanthin-metabolic/scripts/audit_top_strain_feasibility.py

r"""Re-open a finished cell and re-audit the feasibility of the strains it nominates.

WHY THIS EXISTS SEPARATELY FROM THE TRAINER. The Genesis "Metabolic Grounding" endpoint is a
property of the model's OUTPUT, not of its training curve: of the top 100 strains it ranks
highest, how many carry a flux vector that respects mass balance and the second law. The
trainer now writes that number at the selected epoch, but a number written once by the
process that produced it cannot be checked. This script rebuilds the model from the result
file's own cell record, loads the selected-epoch weights the trainer saved, and recomputes
the same block from scratch. Agreement is the check; disagreement is a finding.

THE SAMPLER. With ``--stochastic`` the flux head's reparameterized draw is switched on and
the layer is run ``--draws`` times per genotype. The fraction reported is then the fraction
of DRAWS that are feasible, not the feasibility of the mean flux, which is a different and
strictly easier question: a mean of infeasible vectors can look feasible. Dropout is held
in eval mode throughout so the spread measured is the flux posterior alone, which is the
convention ``experiments/026-metabolism-flux/scripts/flux_sampling_demo.py`` fixed.

The arm must have been trained with a stochastic head for the draws to mean anything;
``--stochastic`` on a deterministic arm switches on a ``log_std`` output that was never
supervised, and the script says so in its output rather than hiding it.

Run from the repo root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/027-betaxanthin-metabolic/scripts/audit_top_strain_feasibility.py \
        --result wave5_g7_flux_capacity_nocstr_s2704.json

Writes ``results/audit_<result stem>.json``.
"""

import argparse
import json
import os
import os.path as osp
import sys
from typing import Any, cast

import numpy as np
import torch
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

load_dotenv()

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from train_bx import (  # noqa: E402
    CHECKPOINT_DIR,
    FEASIBILITY_DEFINITIONS,
    HEAD,
    RESULTS_DIR,
    TOP_K_FEASIBLE,
    BxCell,
    _fractions,
    _per_sample_feasibility,
    _top_k_mask,
    build_dataset,
    build_feasibility_block,
    build_model_for_cell,
    code_provenance,
    collect_feasibility,
    criteria_flags,
    load_config,
    pinned_test_mask,
    resolve_pinned_test,
)

from torchcell.datamodules import CellDataModule  # noqa: E402

DATA_ROOT = cast(str, os.environ["DATA_ROOT"])


class DrawAudit(BaseModel):
    """Feasibility measured over stochastic draws rather than over the mean flux."""

    model_config = ConfigDict(extra="forbid")

    draws: int
    n_genotypes: int
    arm_was_trained_stochastic: bool
    note: str
    fraction_of_draws: dict[str, float] = Field(default_factory=dict)
    fraction_of_genotypes_all_draws_feasible: dict[str, float] = Field(
        default_factory=dict
    )


class AuditReport(BaseModel):
    """One re-audit of one finished cell."""

    model_config = ConfigDict(extra="forbid")

    result_file: str
    checkpoint: str
    cell: dict[str, Any]
    select_epoch: int
    recomputed_feasibility: dict[str, Any]
    agreement_with_trainer: dict[str, Any]
    draw_audit: DrawAudit | None = None
    code_provenance: dict[str, str]


def _enable_sampler(model: Any) -> None:
    """Take reparameterized draws without taking dropout noise with them."""
    layer = model.flux_layer
    layer.config.stochastic = True
    model.eval()
    layer.train()
    for module in layer.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result", required=True,
        help="Result file name under results/, e.g. wave5_g7_flux_capacity_nocstr_s2704.json",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Override the checkpoint path; by default it is taken from the result file.",
    )
    parser.add_argument(
        "--stochastic", action="store_true",
        help="Switch the flux head's sampler on and report the fraction of DRAWS feasible.",
    )
    parser.add_argument("--draws", type=int, default=16)
    parser.add_argument(
        "--draw-genotypes", type=int, default=TOP_K_FEASIBLE,
        help="How many top-ranked genotypes to draw for. The default is the Genesis top 100.",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    path = osp.join(RESULTS_DIR, args.result)
    with open(path) as fh:
        result = json.load(fh)
    cell = BxCell(**result["cell"])
    cfg = load_config()

    ckpt_path = args.checkpoint or result.get("checkpoint_path")
    if ckpt_path is None:
        ckpt_path = osp.join(CHECKPOINT_DIR, f"{osp.splitext(args.result)[0]}.pt")
    assert osp.isfile(ckpt_path), (
        f"no checkpoint at {ckpt_path}. A cell trained before the wave-6 checkpoint saving "
        "cannot be re-audited; it has to be re-run."
    )
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert checkpoint["cell"]["seed"] == cell.seed, (
        f"checkpoint is for seed {checkpoint['cell']['seed']} and the result file for "
        f"{cell.seed}."
    )

    dataset = build_dataset()
    pinned, _ = resolve_pinned_test(dataset, cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, head_specs, main_head, _ = build_model_for_cell(cell, cfg, dataset, device)
    assert model.flux_layer is not None, (
        f"arm {cell.arm!r} has no flux layer, so there is nothing to audit. The trainer "
        "records that as an unavailable feasibility block, which is the answer."
    )
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(checkpoint["parameters"][name].to(param.device))
    mean = checkpoint["target_mean"].to(device)
    sd = checkpoint["target_sd"].to(device)

    dm = CellDataModule(
        dataset=dataset,
        cache_dir=osp.join(DATA_ROOT, cfg.data.dataset_subdir, "data_module_cache"),
        split_indices=["phenotype_label_index", "perturbation_count_index"],
        batch_size=cell.batch_size,
        random_seed=cell.split_seed,
        num_workers=cell.num_workers,
        pin_memory=True,
        prefetch=False,
        follow_batch=["perturbation_indices", "phenotype_values"],
        pinned_test_indices=pinned,
    )
    dm.setup()

    per_test = collect_feasibility(
        model, dm.test_dataloader(), dataset, main_head, mean, sd
    )
    per_all = collect_feasibility(
        model, dm.all_dataloader(), dataset, main_head, mean, sd
    )
    block = build_feasibility_block(
        per_all, per_test, pinned_test_mask(dm, dataset), model.flux_layer, main_head
    )

    # AGREEMENT. The trainer computed the same block inside the training process; this one
    # rebuilt the model from the saved weights. The two must match to numerical noise, and
    # a mismatch means the checkpoint does not describe the scored model.
    agreement: dict[str, Any] = {"checked": False}
    trainer_block = result.get("feasibility")
    if trainer_block is not None and trainer_block.get("available"):
        deltas: dict[str, float] = {}
        for group, payload in block.groups.items():
            for name, value in payload.fractions.items():
                ref = trainer_block["groups"].get(group, {}).get("fractions", {}).get(name)
                if ref is not None:
                    deltas[f"{group}/{name}"] = float(value - ref)
        agreement = {
            "checked": True,
            "max_abs_delta": max((abs(v) for v in deltas.values()), default=0.0),
            "deltas": deltas,
        }

    draw_audit = None
    if args.stochastic:
        layer = model.flux_layer
        was_stochastic = bool(layer.config.stochastic)
        _enable_sampler(model)
        order_mask = _top_k_mask(
            per_all["prediction"],
            np.ones(per_all["prediction"].size, dtype=bool),
            args.draw_genotypes,
        )
        wanted = set(np.flatnonzero(order_mask).tolist())
        cell_graph = dataset.cell_graph.to(device)
        # Per genotype, one list of per-draw feasibility rows.
        stacked: dict[str, list[np.ndarray]] = {}
        with torch.no_grad():
            for _ in range(args.draws):
                cursor = 0
                keep: dict[str, list[np.ndarray]] = {}
                for batch in dm.all_dataloader():
                    size = int(batch.num_graphs)
                    rows = [i for i in range(size) if (cursor + i) in wanted]
                    cursor += size
                    if not rows:
                        continue
                    batch = batch.to(device)
                    _, reps = model(cell_graph, batch)
                    per = _per_sample_feasibility(layer, reps["flux"])
                    idx = torch.tensor(rows, device=device)
                    for key, value in per.items():
                        keep.setdefault(key, []).append(
                            value[idx].detach().cpu().numpy()
                        )
                for key, chunks in keep.items():
                    stacked.setdefault(key, []).append(np.concatenate(chunks))
        # [draws, genotypes] per key.
        per_draw = {k: np.stack(v, axis=0) for k, v in stacked.items()}
        flat = {k: v.reshape(-1) for k, v in per_draw.items()}
        n_geno = next(iter(per_draw.values())).shape[1]
        # The flags come back flat, so reshape them to [draws, genotypes] and require every
        # draw of a genotype to pass. Feasible on average and feasible on every draw are
        # different claims and both are reported.
        all_draws = {
            name: float(flags.reshape(args.draws, n_geno).all(axis=0).mean())
            for name, flags in criteria_flags(flat).items()
        }
        draw_audit = DrawAudit(
            draws=args.draws,
            n_genotypes=n_geno,
            arm_was_trained_stochastic=was_stochastic,
            note=(
                "fraction_of_draws counts every (genotype, draw) pair; "
                "fraction_of_genotypes_all_draws_feasible counts a genotype only when "
                "every one of its draws meets the criterion. The arm was trained with a "
                + ("stochastic" if was_stochastic else "DETERMINISTIC")
                + " flux head; on a deterministic arm the sampler reads a log_std output "
                "that was never supervised, so the spread is not a calibrated posterior."
            ),
            fraction_of_draws=_fractions(flat),
            fraction_of_genotypes_all_draws_feasible=all_draws,
        )

    report = AuditReport(
        result_file=path,
        checkpoint=ckpt_path,
        cell=cell.model_dump(),
        select_epoch=int(result["select_epoch"]),
        recomputed_feasibility=block.model_dump(),
        agreement_with_trainer=agreement,
        draw_audit=draw_audit,
        code_provenance=code_provenance(),
    )
    out = args.out or f"audit_{osp.splitext(args.result)[0]}.json"
    out_path = osp.join(RESULTS_DIR, out)
    with open(out_path, "w") as fh:
        fh.write(report.model_dump_json(indent=2))
    print(f"wrote {out_path}")
    key = "all_top100" if "all_top100" in block.groups else "all"
    print(f"recomputed feasibility[{key}]: {block.groups[key].fractions}")
    if agreement["checked"]:
        print(f"agreement with the trainer: max |delta| {agreement['max_abs_delta']:.3e}")
    if draw_audit is not None:
        print(f"draw audit ({draw_audit.draws} draws, {draw_audit.n_genotypes} genotypes)")
        print(f"  fraction of draws: {draw_audit.fraction_of_draws}")
        print(
            "  fraction of genotypes feasible on every draw: "
            f"{draw_audit.fraction_of_genotypes_all_draws_feasible}"
        )
    if FEASIBILITY_DEFINITIONS and main_head != HEAD:
        print(
            "note: this cell trained the amino-acid head, so there is no betaxanthin "
            "ranking and the top-100 groups are absent by design."
        )


if __name__ == "__main__":
    main()
