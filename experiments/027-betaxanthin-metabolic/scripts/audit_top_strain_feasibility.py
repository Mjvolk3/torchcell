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
from typing import Any

import numpy as np
import torch
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

load_dotenv()

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from train_bx import (  # noqa: E402
    BALANCE_TOLERANCES,
    BOTH_BALANCE_TOLERANCE,
    CHECKPOINT_DIR,
    FEASIBILITY_DEFINITIONS,
    HEAD,
    RESULTS_DIR,
    TOP_K_FEASIBLE,
    BxCell,
    FeasibilityStats,
    _fractions,
    _head_params,
    _per_sample_feasibility,
    _stats,
    _top_k_mask,
    build_dataset,
    build_feasibility_block,
    build_model_for_cell,
    code_provenance,
    criteria_flags,
    load_config,
    pinned_test_mask,
    resolve_pinned_test,
)

from torchcell.datamodules import CellDataModule  # noqa: E402

DATA_ROOT = os.environ["DATA_ROOT"]

#: A reaction counts as CARRYING FLUX when its magnitude exceeds this fraction of the
#: layer's working flux magnitude (``FluxLayerConfig.flux_scale``, 100 mmol/gDW/h), i.e.
#: 1e-4 mmol/gDW/h at the shipped scale. Below that the reaction is off, and a balance
#: statistic computed over a vector of off reactions is a statistic about zeros.
FLUX_ACTIVE_RELATIVE_THRESHOLD = 1e-6

#: Turnover thresholds for the THIRD balance rule, as fractions of the flux scale. A row of
#: S whose turnover omega_i is at the balance_epsilon floor has a ratio of 2 whenever the two
#: tiny fluxes touching it share a sign, so a plain max over all rows is decided by idle
#: metabolites and fails every head for a reason that has nothing to do with the flux the
#: model actually carries. The rule below evaluates the same ratio over the rows that
#: TURN OVER: omega_i > threshold * flux_scale. 1e-3 is the primary; the other two are the
#: sensitivity, reported beside it rather than chosen after seeing the answer.
BALANCE_TURNOVER_THRESHOLDS: tuple[float, ...] = (1e-2, 1e-3, 1e-4)
PRIMARY_TURNOVER_THRESHOLD = 1e-3


def _threshold_tag(threshold: float) -> str:
    """The suffix a thresholded statistic carries, e.g. ``1e-03``."""
    return f"{threshold:.0e}"


STRICT_DEFINITIONS: dict[str, str] = {
    "why": (
        "The trainer's Genesis criterion tests the MEDIAN relative balance residual over "
        "the independent rows of S. A flux vector that is zero on most reactions passes "
        "that test on the zeros alone: the median row of a mostly-off vector has no "
        "production to fail to consume. The max over the same rows is the honest form of "
        "'this vector balances', and the fraction of reactions carrying flux says whether "
        "there was anything to balance."
    ),
    "balance_max_ratio": (
        "max over the independent rows i of S of |(Sv)_i| / (omega_i + balance_epsilon), "
        "the same ratio the trainer takes the median of. Computed by "
        "train_bx._per_sample_feasibility, which every wave-6 cell already recorded the "
        "distribution of; only the CRITERION built on it is new."
    ),
    "flux_active_frac": (
        "fraction of the 4,131 reactions with |v| > 1e-6 * flux_scale = 1e-4 mmol/gDW/h."
    ),
    "genesis_median_rule": (
        "the trainer's criterion: balance_median_ratio <= 1e-2 AND no active reaction "
        "running uphill at standard state."
    ),
    "genesis_max_rule": (
        "the same criterion with balance_max_ratio <= 1e-2 in place of the median, so "
        "EVERY independent row must balance, not the typical one."
    ),
    "balance_active_max_ratio_<t>": (
        "max of the same ratio over the independent rows whose turnover omega_i exceeds "
        "t * flux_scale. The plain max is decided by rows at the balance_epsilon floor, "
        "where two tiny same-sign fluxes give a ratio of exactly 2, so it fails every head "
        "for a reason unrelated to the flux the model carries; restricting to rows that "
        "turn over is the rule that separates the heads."
    ),
    "balance_active_p95_ratio_<t>": (
        "the 95th percentile of the same ratio over the same rows, which is the max with "
        "the top 5 percent of rows removed and is what a handful of near-threshold rows "
        "cannot dominate."
    ),
    "balance_active_rows_<t>": (
        "how many independent rows clear the turnover threshold. A rule evaluated over "
        "three rows is not a statement about the flux vector, so the count is reported "
        "beside every thresholded ratio."
    ),
    "genesis_active_rule": (
        "the Genesis criterion with balance_active_max_ratio at the primary threshold "
        "(1e-3 * flux_scale) <= 1e-2 in place of the median or the plain max."
    ),
}


def _per_sample_extras(
    layer: Any, flux: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Per-sample scalars the trainer does not compute. ``[B]`` each.

    The balance ratio is rebuilt here from the layer's own ``_s_matmul`` and ``turnover``,
    the same two calls ``train_bx._per_sample_feasibility`` makes, so the thresholded
    statistics are the SAME ratio as the trainer's median and the plain max, restricted to a
    subset of the rows. A genotype with no row above the threshold gets NaN rather than a
    number, and the count says how many rows the statistic was taken over.
    """
    cfg = layer.config
    scale = float(cfg.flux_scale)
    v = flux["v"]
    residual = layer._s_matmul(v)[:, layer.independent_rows].abs()
    omega = layer.turnover(v)[:, layer.independent_rows]
    ratio = residual / (omega + cfg.balance_epsilon)
    out: dict[str, torch.Tensor] = {
        "flux_active_frac": (v.abs() > FLUX_ACTIVE_RELATIVE_THRESHOLD * scale)
        .float()
        .mean(dim=1)
    }
    for threshold in BALANCE_TURNOVER_THRESHOLDS:
        tag = _threshold_tag(threshold)
        active = omega > threshold * scale
        counts = active.sum(dim=1)
        masked = ratio.masked_fill(~active, float("-inf"))
        maximum = masked.max(dim=1).values
        # A quantile over a masked row needs the inactive entries gone, not set to -inf, so
        # it is taken per genotype. There is no vectorized masked quantile in torch.
        p95 = torch.full_like(maximum, float("nan"))
        for i in range(ratio.shape[0]):
            selected = ratio[i][active[i]]
            if selected.numel():
                p95[i] = torch.quantile(selected, 0.95)
        out[f"balance_active_max_ratio_{tag}"] = maximum.masked_fill(
            counts == 0, float("nan")
        )
        out[f"balance_active_p95_ratio_{tag}"] = p95
        out[f"balance_active_rows_{tag}"] = counts.to(ratio.dtype)
    return out


def collect_with_extras(
    model: Any,
    loader: Any,
    dataset: Any,
    head_name: str,
    mean: torch.Tensor,
    sd: torch.Tensor,
) -> dict[str, np.ndarray]:
    """One forward pass collecting the trainer's per-sample scalars AND the extras.

    A copy of ``train_bx.collect_feasibility`` with :func:`_per_sample_extras` folded into
    the same pass rather than a second one. It is a copy and not a wrapper because
    ``train_bx.py`` is owned by another work stream tonight and this script may not edit
    it; the trainer's own scalars still come from ``_per_sample_feasibility``, so the two
    cannot drift in their definitions.
    """
    device = next(model.parameters()).device
    layer = model.flux_layer
    assert layer is not None, "collect_with_extras called on an arm with no flux layer."
    was_training = model.training
    model.eval()
    parts: dict[str, list[np.ndarray]] = {}
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, reps = model(dataset.cell_graph.to(device), batch)
            per = _per_sample_feasibility(layer, reps["flux"])
            per.update(_per_sample_extras(layer, reps["flux"]))
            if head_name == HEAD:
                pred = _head_params(reps["head_outputs"][head_name])
                per["prediction"] = pred[:, 0] * sd[0] + mean[0]
            for key, value in per.items():
                parts.setdefault(key, []).append(value.detach().cpu().numpy())
    model.train(was_training)
    return {k: np.concatenate(v) for k, v in parts.items()}


def _nan_stats(x: np.ndarray) -> FeasibilityStats:
    """``_stats`` for a quantity that is NaN where no row cleared the threshold."""
    finite = x[np.isfinite(x)]
    if not finite.size:
        return FeasibilityStats(
            median=float("nan"), mean=float("nan"), p90=float("nan"), max=float("nan")
        )
    return FeasibilityStats(
        median=float(np.median(finite)),
        mean=float(np.mean(finite)),
        p90=float(np.percentile(finite, 90)),
        max=float(np.max(finite)),
    )


def strict_criteria_flags(per: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-genotype feasibility under the MAX and the turnover-thresholded balance rules.

    The thermodynamic half is unchanged in every rule, so the rows differ in exactly one
    thing: which statistic of the per-row balance residual has to clear 1 percent. A NaN
    ratio (no row above the turnover threshold) compares False, so a genotype whose flux
    vector turns nothing over is never counted as balanced.
    """
    out: dict[str, np.ndarray] = {}
    median = per["balance_median_ratio"]
    maximum = per["balance_max_ratio"]
    for tol in BALANCE_TOLERANCES:
        out[f"balance_max_ratio_le_{tol:.0e}"] = maximum <= tol
    active: dict[str, np.ndarray] = {}
    with np.errstate(invalid="ignore"):
        for threshold in BALANCE_TURNOVER_THRESHOLDS:
            tag = _threshold_tag(threshold)
            for statistic in ("max", "p95"):
                key = f"balance_active_{statistic}_ratio_{tag}"
                if key not in per:
                    continue
                values = per[key]
                active[f"{statistic}_{tag}"] = values
                for tol in BALANCE_TOLERANCES:
                    out[f"{key}_le_{tol:.0e}"] = (
                        np.nan_to_num(values, nan=np.inf) <= tol
                    )
    for key, label in (
        ("thermo_model_violation_frac", "thermo_model"),
        ("thermo_standard_state_violation_frac", "thermo_standard_state"),
    ):
        if key not in per:
            continue
        downhill = per[key] <= 0.0
        out[f"genesis_median_rule_{label}"] = (
            median <= BOTH_BALANCE_TOLERANCE
        ) & downhill
        out[f"genesis_max_rule_{label}"] = (
            maximum <= BOTH_BALANCE_TOLERANCE
        ) & downhill
        for name, values in active.items():
            out[f"genesis_active_{name}_rule_{label}"] = (
                np.nan_to_num(values, nan=np.inf) <= BOTH_BALANCE_TOLERANCE
            ) & downhill
    return out


class StrictGroup(BaseModel):
    """One group of genotypes under both balance rules."""

    model_config = ConfigDict(extra="forbid")

    n: int
    rank_by: str | None = None
    stats: dict[str, FeasibilityStats]
    fractions: dict[str, float]


class StrictFeasibility(BaseModel):
    """The max-based balance rule and the flux-carrying fraction, per group."""

    model_config = ConfigDict(extra="forbid")

    definitions: dict[str, str] = Field(
        default_factory=lambda: dict(STRICT_DEFINITIONS)
    )
    flux_active_threshold: float
    flux_scale: float
    n_independent_balance_rows: int
    n_reactions: int
    groups: dict[str, StrictGroup] = Field(default_factory=dict)


def _strict_group(
    per: dict[str, np.ndarray], sel: np.ndarray, rank_by: str | None
) -> StrictGroup:
    keys = ("balance_median_ratio", "balance_max_ratio", "flux_active_frac")
    thresholded = [
        f"balance_active_{statistic}_{_threshold_tag(t)}"
        for t in BALANCE_TURNOVER_THRESHOLDS
        for statistic in ("max_ratio", "p95_ratio", "rows")
    ]
    sub = {k: v[sel] for k, v in per.items() if k not in {"prediction", "observed"}}
    return StrictGroup(
        n=int(sel.sum()),
        rank_by=rank_by,
        stats={k: _stats(sub[k]) for k in keys}
        | {k: _nan_stats(sub[k]) for k in thresholded if k in sub},
        fractions={
            **{k: float(v.mean()) for k, v in criteria_flags(sub).items()},
            **{k: float(v.mean()) for k, v in strict_criteria_flags(sub).items()},
        },
    )


def build_strict_block(
    per_all: dict[str, np.ndarray],
    per_test: dict[str, np.ndarray],
    test_pinned: np.ndarray,
    layer: Any,
    head_name: str,
) -> StrictFeasibility:
    """The same four groups the trainer reports, under the median AND the max rule."""
    rank_note = "predicted betaxanthin, descending" if head_name == HEAD else None
    groups: dict[str, StrictGroup] = {
        "all": _strict_group(
            per_all, np.ones(per_all["balance_median_ratio"].size, bool), None
        ),
        "pinned_test": _strict_group(per_test, test_pinned, None),
    }
    if rank_note is not None:
        all_sel = np.ones(per_all["balance_median_ratio"].size, bool)
        groups["all_top100"] = _strict_group(
            per_all,
            _top_k_mask(per_all["prediction"], all_sel, TOP_K_FEASIBLE),
            rank_note,
        )
        groups["pinned_test_top100"] = _strict_group(
            per_test,
            _top_k_mask(per_test["prediction"], test_pinned, TOP_K_FEASIBLE),
            rank_note,
        )
    return StrictFeasibility(
        flux_active_threshold=(
            FLUX_ACTIVE_RELATIVE_THRESHOLD * float(layer.config.flux_scale)
        ),
        flux_scale=float(layer.config.flux_scale),
        n_independent_balance_rows=int(layer.independent_rows.numel()),
        n_reactions=int(layer.n_reactions),
        groups=groups,
    )


class DrawAudit(BaseModel):
    """Feasibility measured over stochastic draws rather than over the mean flux."""

    model_config = ConfigDict(extra="forbid")

    draws: int
    n_genotypes: int
    arm_was_trained_stochastic: bool
    #: How the audited genotypes were chosen. The Genesis question is about the strains the
    #: model NOMINATES, which only exists for the betaxanthin head; an amino-acid cell has
    #: no production ranking, so it is audited on a fixed prefix of the dataset instead and
    #: the difference is recorded rather than left for the reader to infer.
    genotype_selection: str
    note: str
    fraction_of_draws: dict[str, float] = Field(default_factory=dict)
    fraction_of_genotypes_all_draws_feasible: dict[str, float] = Field(
        default_factory=dict
    )
    #: The distribution of the two balance order statistics and of the flux-carrying
    #: fraction over every (genotype, draw) pair.
    draw_stats: dict[str, FeasibilityStats] = Field(default_factory=dict)


class AuditReport(BaseModel):
    """One re-audit of one finished cell."""

    model_config = ConfigDict(extra="forbid")

    result_file: str
    checkpoint: str
    cell: dict[str, Any]
    select_epoch: int
    recomputed_feasibility: dict[str, Any]
    #: The max-based balance rule and the flux-carrying fraction, which the trainer's block
    #: does not carry. Computed in the same forward pass as the recomputation above.
    strict_feasibility: StrictFeasibility
    agreement_with_trainer: dict[str, Any]
    draw_audit: DrawAudit | None = None
    code_provenance: dict[str, str]


def _enable_sampler(model: Any) -> None:
    """Take reparameterized draws without taking dropout noise with them.

    PRECONDITION, checked rather than assumed. On the ``box`` parameterization the flux MLP
    is built with ``z_out = 2 if cfg.stochastic else 1``, so a cell trained deterministically
    has no spread channel at all and setting ``config.stochastic`` afterwards indexes a
    dimension of size 1. Refused with a message instead of crashing in the draw loop, and
    NOT worked around: a spread that was never trained is not a posterior, so there is
    nothing honest to fall back to.
    """
    layer = model.flux_layer
    z_out_width = layer.flux_mlp[-1].out_features
    if layer.parameterization == "box" and z_out_width < 2:
        raise SystemExit(
            "--stochastic needs a flux head with a spread channel. This cell was trained "
            f"on the 'box' parameterization with a {z_out_width}-channel flux MLP, so no "
            "log_std exists to draw from and none was ever supervised. The 'projection' "
            "parameterization builds the spread head unconditionally and is what the draw "
            "audit is for; re-run this cell on that arm, or drop --stochastic to get the "
            "deterministic recomputation, which is what the trainer's own block reports."
        )
    layer.config.stochastic = True
    model.eval()
    layer.train()
    for module in layer.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        required=True,
        help="Result file name under results/, e.g. wave5_g7_flux_capacity_nocstr_s2704.json",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Override the checkpoint path; by default it is taken from the result file.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Switch the flux head's sampler on and report the fraction of DRAWS feasible.",
    )
    parser.add_argument("--draws", type=int, default=16)
    parser.add_argument(
        "--draw-genotypes",
        type=int,
        default=TOP_K_FEASIBLE,
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

    per_test = collect_with_extras(
        model, dm.test_dataloader(), dataset, main_head, mean, sd
    )
    per_all = collect_with_extras(
        model, dm.all_dataloader(), dataset, main_head, mean, sd
    )
    pinned_mask = pinned_test_mask(dm, dataset)
    # The trainer's block is rebuilt from the trainer's OWN scalars alone, so the
    # recomputation stays comparable to what the cell wrote. The extras go into their own
    # block rather than into this one.
    extra_keys = {"flux_active_frac"} | {
        key for key in per_all if key.startswith("balance_active_")
    }
    trainer_keys = {k: v for k, v in per_all.items() if k not in extra_keys}
    trainer_keys_test = {k: v for k, v in per_test.items() if k not in extra_keys}
    block = build_feasibility_block(
        trainer_keys, trainer_keys_test, pinned_mask, model.flux_layer, main_head
    )
    strict = build_strict_block(
        per_all, per_test, pinned_mask, model.flux_layer, main_head
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
                ref = (
                    trainer_block["groups"]
                    .get(group, {})
                    .get("fractions", {})
                    .get(name)
                )
                if ref is not None:
                    deltas[f"{group}/{name}"] = float(value - ref)
        agreement = {
            "checked": True,
            "max_abs_delta": max((abs(v) for v in deltas.values()), default=0.0),
            "deltas": deltas,
        }

    draw_audit = None
    if args.stochastic:
        layer: Any = model.flux_layer
        was_stochastic = bool(layer.config.stochastic)
        _enable_sampler(model)
        n_total = per_all["balance_median_ratio"].size
        if "prediction" in per_all:
            order_mask = _top_k_mask(
                per_all["prediction"], np.ones(n_total, dtype=bool), args.draw_genotypes
            )
            selection = (
                f"the {args.draw_genotypes} genotypes with the highest predicted "
                "betaxanthin, which is the Genesis question"
            )
        else:
            # No production ranking exists on the amino-acid head, so there is no "top"
            # to audit. A fixed prefix in dataset order is arbitrary but reproducible, and
            # saying which it is beats inventing a ranking out of a 19-column prediction.
            order_mask = np.zeros(n_total, dtype=bool)
            order_mask[: args.draw_genotypes] = True
            selection = (
                f"the first {args.draw_genotypes} genotypes in dataset order; this cell "
                "trained the amino-acid head, which defines no production ranking"
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
                    per.update(_per_sample_extras(layer, reps["flux"]))
                    idx = torch.tensor(rows, device=device)
                    for key, drawn in per.items():
                        keep.setdefault(key, []).append(
                            drawn[idx].detach().cpu().numpy()
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
        draw_flags = {**criteria_flags(flat), **strict_criteria_flags(flat)}
        all_draws = {
            name: float(flags.reshape(args.draws, n_geno).all(axis=0).mean())
            for name, flags in draw_flags.items()
        }
        draw_stats = {
            name: _stats(flat[name])
            for name in (
                "balance_median_ratio",
                "balance_max_ratio",
                "flux_active_frac",
            )
        }
        draw_audit = DrawAudit(
            draws=args.draws,
            n_genotypes=n_geno,
            arm_was_trained_stochastic=was_stochastic,
            genotype_selection=selection,
            note=(
                "fraction_of_draws counts every (genotype, draw) pair; "
                "fraction_of_genotypes_all_draws_feasible counts a genotype only when "
                "every one of its draws meets the criterion. The arm was trained with a "
                + ("stochastic" if was_stochastic else "DETERMINISTIC")
                + " flux head; on a deterministic arm the sampler reads a log_std output "
                "that was never supervised, so the spread is not a calibrated posterior."
            ),
            fraction_of_draws={
                **_fractions(flat),
                **{k: float(v.mean()) for k, v in strict_criteria_flags(flat).items()},
            },
            fraction_of_genotypes_all_draws_feasible=all_draws,
            draw_stats=draw_stats,
        )

    report = AuditReport(
        result_file=path,
        checkpoint=ckpt_path,
        cell=cell.model_dump(),
        select_epoch=int(result["select_epoch"]),
        recomputed_feasibility=block.model_dump(),
        strict_feasibility=strict,
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
    strict_group = strict.groups[key]
    print(f"strict feasibility[{key}], n={strict_group.n}")
    primary = _threshold_tag(PRIMARY_TURNOVER_THRESHOLD)
    for stat_name in (
        "balance_median_ratio",
        "balance_max_ratio",
        f"balance_active_max_ratio_{primary}",
        f"balance_active_p95_ratio_{primary}",
        f"balance_active_rows_{primary}",
        "flux_active_frac",
    ):
        stat = strict_group.stats[stat_name]
        print(
            f"  {stat_name:34s} median {stat.median:.4g} p90 {stat.p90:.4g} "
            f"max {stat.max:.4g}"
        )
    for flag_name, fraction in sorted(strict_group.fractions.items()):
        if flag_name.startswith(("genesis_", "balance_max_ratio_le_")):
            print(f"  {flag_name:52s} {fraction:.3f}")
    if agreement["checked"]:
        print(
            f"agreement with the trainer: max |delta| {agreement['max_abs_delta']:.3e}"
        )
    if draw_audit is not None:
        print(
            f"draw audit ({draw_audit.draws} draws, {draw_audit.n_genotypes} genotypes)"
        )
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
