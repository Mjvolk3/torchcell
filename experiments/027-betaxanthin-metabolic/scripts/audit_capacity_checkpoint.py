# experiments/027-betaxanthin-metabolic/scripts/audit_capacity_checkpoint.py
# [[experiments.026-metabolism-flux.kcat-uncertainty-2026-09-14]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/027-betaxanthin-metabolic/scripts/audit_capacity_checkpoint.py

r"""What the enzyme capacity bound actually IS at a trained checkpoint, term by term.

THE NUMBER THIS SETTLES. [[experiments.027-betaxanthin-metabolic.wave8-sampling]] reports,
for the amino-acid arms, a "median box width of 1.5e-7 mmol gDW-1 h-1, against a plain
projection box of order 1 to 10 in the same units", and attributes it to "the capacity bound
and its abundance prior (median 2.7e-6 mmol/gDW)". The parameter review could reproduce the
direction and not the magnitude: at that abundance the per-reaction CAPACITY is of order
0.045 mmol/gDW/h, not 1.5e-7, leaving a factor of about 3e5 unexplained. A number that is
quoted in notes and cannot be reproduced is not a result yet, and nothing downstream should
be designed on it.

THE DECOMPOSITION. The box is

.. math::
    v^\ell_j=\max(c_j\,\underline v_j,\,-\mathrm{cap}_j),\qquad
    v^u_j=\min(c_j\,\overline v_j,\,\mathrm{cap}_j),\qquad
    \mathrm{cap}_j=\sum_{u:\rho(u)=j}k_{cat,u}E_u

so the width is the GEM's own box times an availability factor :math:`c_j` times a capacity
factor, and each is measured separately here rather than inferred. The unit abundance
:math:`E_u` is a SOFTMIN over subunits, whose undershoot against the arithmetic minimum is
measured as its own ratio, because a power mean with a finite exponent sits BELOW the
minimum and a 17-subunit complex is where that bites.

THE CONTROL is the same forward pass with ``use_enzyme_capacity`` flipped off, which
isolates the bound's contribution to the width exactly rather than by subtraction of
modeled terms.

THE PROMISCUITY OVERCOUNT. The bound lets one gene's abundance serve every reaction it
catalyzes at full strength at once, which is what GECKO's shared enzyme pseudometabolite
prevents. The only thing coupling them is the protein budget, which is soft and sits at
scale 0.0 in the recipe that ships. The implied overcount per gene,
:math:`\sum_j|v_j|/(k_{cat}E_g)` over the units that gene belongs to, is measured here.

PROVENANCE CAVEAT, RECORDED RATHER THAN ASSUMED. A rebuilt layer resolves the k_cat table
from the live mirror, and the mirror has been regenerated mid-queue before. The resolved
table is therefore compared element-wise against the checkpoint's own ``kcat_per_h`` buffer
and the comparison is written into the report, so a drifted mirror is visible instead of
silently re-labelling the tiers.

Run from the worktree root; CPU is enough and is the default::

    PYTHONHASHSEED=0 PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/027-betaxanthin-metabolic/scripts/audit_capacity_checkpoint.py

Writes ``experiments/026-metabolism-flux/results/capacity_checkpoint_audit.json``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import os.path as osp
from typing import Any

import numpy as np
import torch
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from torchcell.metabolism.constraints import build_gem_tensors
from torchcell.metabolism.parameters import (
    KCAT_SIGMA_LADDER,
    KINETICS_MIRROR,
    PLAUSIBILITY_RESULT,
    VERIFIED_PAIRS_FEATURE,
    SubstrateMatch,
    load_predicted_unit_table,
    resolve_kcat_table,
)
from torchcell.metabolism.yeast_GEM import YeastGEM
from torchcell.models.cell_graph_transformer_metabolism import perturbed_gene_pool
from torchcell.timestamp import timestamp

load_dotenv()

EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
DATA_ROOT = os.environ["DATA_ROOT"]
EXP_DIR = osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic")
RESULTS_027 = osp.join(EXP_DIR, "results")
RESULTS_026 = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
OED_SCEREVISIAE = osp.join(
    DATA_ROOT, "data/enzyme_kinetics/open_enzyme_database/scerevisiae"
)
OED_ALL = osp.join(DATA_ROOT, "data/enzyme_kinetics/open_enzyme_database/all")

#: The quoted median box width this audit has to explain, mmol gDW-1 h-1.
QUOTED_MEDIAN_BOX_WIDTH = 1.5e-7

#: The cells audited by default, and what each is here for.
#:
#: * ``wave8_aa_conccyto_e200_s2700`` -- an amino-acid cell of the family the 1.5e-7 number
#:   was reported on, with the abundance head and the capacity bound both on.
#: * ``wave6_proj_flux_projection_s2700`` -- the plain projection, the "order 1 to 10" box
#:   the amino-acid box is contrasted against.
DEFAULT_CELLS = [
    "wave8_aa_conccyto_e200_s2700.json",
    "wave6_proj_flux_projection_s2700.json",
]


def _percentiles(x: np.ndarray, name: str) -> dict[str, float]:
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return {f"{name}_{k}": float("nan") for k in ("p10", "p50", "p90", "min", "max")}
    return {
        f"{name}_p10": float(np.percentile(finite, 10)),
        f"{name}_p50": float(np.median(finite)),
        f"{name}_p90": float(np.percentile(finite, 90)),
        f"{name}_min": float(finite.min()),
        f"{name}_max": float(finite.max()),
    }


class CapacityAudit(BaseModel):
    """One checkpoint's capacity bound, decomposed."""

    model_config = ConfigDict(extra="forbid")

    cell: str
    arm: str
    seed: int
    parameterization: str
    use_enzyme_capacity: bool
    use_abundance_head: bool
    use_protein_budget: bool
    p_avail_g_per_gdw: float
    constraint_final_scale: float
    n_genotypes: int
    n_reactions: int
    n_units: int
    box_matches_genotype_chain: bool = Field(
        description=(
            "True when the box the forward pass returned is exactly the one the "
            "availability-and-capacity chain predicts. False means the arm tightened it "
            "further, which is the arm's channel and not an error."
        )
    )

    quoted_median_box_width: float = QUOTED_MEDIAN_BOX_WIDTH
    box_width_all: dict[str, float]
    box_width_gpr_only: dict[str, float]
    box_width_no_capacity_control: dict[str, float]
    box_width_gem_base: dict[str, float]
    explains_quoted_within_factor_3: bool
    ratio_measured_over_quoted: float

    availability_factor: dict[str, float] = Field(
        description="c_j, the availability multiplier on the GEM box."
    )
    capacity_factor: dict[str, float] = Field(
        description="width with capacity over width with availability alone."
    )
    frac_c_j_below_one: float
    frac_c_j_zero: float

    capacity_mmol_per_gdw_per_h: dict[str, float]
    e_g_mmol_per_gdw: dict[str, float]
    e_u_mmol_per_gdw: dict[str, float]
    softmin_over_arithmetic_min: dict[str, float] = Field(
        description="E_u from the softmin divided by the arithmetic min over subunits."
    )
    softmin_undershoot_multigene: dict[str, float]

    feas_capacity_active_frac: float
    protein_used_g_per_gdw: float
    protein_used_over_p_avail: float

    kcat_table_matches_checkpoint: bool
    n_units_kcat_differs: int
    max_abs_log10_kcat_diff: float
    tier_counts: dict[str, int]

    width_by_tier: dict[str, dict[str, float]] = Field(default_factory=dict)
    width_by_genes_per_unit: dict[str, dict[str, float]] = Field(default_factory=dict)
    width_by_availability: dict[str, dict[str, float]] = Field(default_factory=dict)

    n_genes_in_more_than_one_reaction: int
    promiscuity_overcount: dict[str, float]
    frac_genes_overcount_gt_1: float


def _load_train_bx() -> Any:
    spec = importlib.util.spec_from_file_location(
        "train_bx_for_capacity_audit", osp.join(EXP_DIR, "scripts", "train_bx.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_reference_kcat(n_units: int, units, model_dir: str):
    """The k_cat table a rebuild resolves right now, for comparison with the checkpoint."""
    kinetics_dir = osp.join(DATA_ROOT, KINETICS_MIRROR)
    substrates = SubstrateMatch.from_verified_pairs(
        osp.join(kinetics_dir, VERIFIED_PAIRS_FEATURE)
    )
    predicted = load_predicted_unit_table(
        "k_cat",
        n_units,
        kinetics_dir,
        osp.join(EXPERIMENT_ROOT, PLAUSIBILITY_RESULT),
    )
    return resolve_kcat_table(
        units,
        model_dir,
        OED_SCEREVISIAE,
        predicted=predicted,
        substrates=substrates,
        homolog_mirror_dir=OED_ALL,
    )


def audit_cell(result_file: str, tbx: Any, dataset: Any, cfg: Any) -> CapacityAudit:
    """Rebuild one cell, run one forward pass, and take the bound apart."""
    device = torch.device("cpu")
    stem = osp.splitext(result_file)[0]
    ckpt_path = osp.join(
        os.getenv("TC_CHECKPOINT_DIR", osp.join(DATA_ROOT, "checkpoints-027")),
        f"{stem}.pt",
    )
    assert osp.isfile(ckpt_path), f"no checkpoint at {ckpt_path}"
    with open(osp.join(RESULTS_027, result_file)) as handle:
        result = json.load(handle)
    coverage = result["flux_meta"]["coverage"]

    cell = tbx.BxCell(**result["cell"])
    cell.flux_parameterization = str(coverage["parameterization_raw"])
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert checkpoint["cell"]["seed"] == cell.seed, (
        f"checkpoint seed {checkpoint['cell']['seed']} against result seed {cell.seed}"
    )

    model, _, _, _ = tbx.build_model_for_cell(cell, cfg, dataset, device)
    layer = model.flux_layer
    assert layer is not None, f"arm {cell.arm!r} has no flux layer."
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(checkpoint["parameters"][name].to(param.device))
    model.eval()

    pinned, _ = tbx.resolve_pinned_test(dataset, cfg)
    dm = tbx.CellDataModule(
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
    cell_graph = dataset.cell_graph.to(device)
    batch = next(iter(dm.all_dataloader())).to(device)

    with torch.no_grad():
        _, reps = model(cell_graph, batch)
        h_genes = reps["H_genes_pert"]
        pert = batch["gene"].perturbation_indices
        pert_batch = batch["gene"].perturbation_indices_batch
        v = reps["flux"]["v"]
        lb = reps["flux"]["lb"]
        ub = reps["flux"]["ub"]

        # THE GENOTYPE CHAIN, recomputed exactly as `forward` does, because the forward's
        # output dict carries neither the capacity nor the unit abundance.
        gamma = layer.gene_availability(h_genes, pert, pert_batch)
        c_j = layer.reaction_availability(gamma)
        use_head = bool(layer.config.use_abundance_head)
        e_g = layer.gene_abundance(h_genes, gamma) if use_head else None
        e_u = layer.unit_abundance(e_g) if e_g is not None else None
        lb_chain, ub_chain, capacity = layer.dynamic_box(c_j, e_u)
        # RECORDED, NOT ASSERTED. A channel that tightens the box to the direction the
        # reaction energy licenses returns a box the genotype chain alone does not predict,
        # and that is a property of the arm rather than an error. The width below is taken
        # from the forward pass, which is the box the model actually used; the decomposition
        # is taken from the chain, and the two are compared so a gap is visible.
        box_matches_forward = bool(
            torch.allclose(lb_chain, lb, atol=1e-8)
            and torch.allclose(ub_chain, ub, atol=1e-8)
        )

        # THE CONTROL. A config flag, flipped and restored; no code path changes.
        original = layer.config.use_enzyme_capacity
        layer.config.use_enzyme_capacity = False
        lb_nocap, ub_nocap, _ = layer.dynamic_box(c_j, e_u)
        layer.config.use_enzyme_capacity = original

    width = (ub - lb).cpu().numpy()
    width_nocap = (ub_nocap - lb_nocap).cpu().numpy()
    base_width = (layer.ub - layer.lb).cpu().numpy()
    has_gpr = layer.has_gpr.cpu().numpy().astype(bool)
    c_j_np = c_j.cpu().numpy()

    availability = np.where(base_width > 0, width_nocap / np.maximum(base_width, 1e-30), np.nan)
    capacity_factor = np.where(
        width_nocap > 0, width / np.maximum(width_nocap, 1e-30), np.nan
    )

    capacity_np = (
        capacity.cpu().numpy() if capacity is not None else np.full_like(width, np.nan)
    )
    e_g_np = e_g.cpu().numpy() if e_g is not None else np.array([np.nan])
    e_u_np = e_u.cpu().numpy() if e_u is not None else np.array([np.nan])

    # THE SOFTMIN'S OWN UNDERSHOOT. A power mean with exponent -beta sits below the
    # arithmetic minimum, and how far below is a property of the number of subunits.
    unit_gene_unit = layer.unit_gene_unit.cpu().numpy()
    unit_gene_gene = layer.unit_gene_gene.cpu().numpy()
    n_units = int(layer.n_units)
    genes_per_unit = np.bincount(unit_gene_unit, minlength=n_units)
    if e_g is not None:
        index = layer.unit_gene_unit.unsqueeze(0).expand(e_g.shape[0], -1)
        arithmetic_min_t = torch.full(
            (e_g.shape[0], n_units), float("inf"), dtype=e_g.dtype
        ).scatter_reduce(1, index, e_g[:, layer.unit_gene_gene], reduce="amin")
        arithmetic_min = arithmetic_min_t.cpu().numpy()
        ratio = e_u_np / np.where(arithmetic_min > 0, arithmetic_min, np.nan)
        multigene = genes_per_unit > 1
        softmin_ratio = _percentiles(ratio.reshape(-1), "softmin_ratio")
        softmin_multi = _percentiles(ratio[:, multigene].reshape(-1), "softmin_ratio")
    else:
        softmin_ratio = _percentiles(np.array([np.nan]), "softmin_ratio")
        softmin_multi = _percentiles(np.array([np.nan]), "softmin_ratio")

    # THE TIER OF EACH REACTION'S BOUND. A reaction's capacity is a sum over units, so the
    # tier that matters is the one supplying the most of it.
    source = YeastGEM()
    gem = build_gem_tensors(
        source.model, model_dir=source.model_dir, with_independent_rows=False
    )
    units = gem.catalytic_units
    reference = resolve_reference_kcat(units.n_units, units, source.model_dir)
    kcat_ckpt = (layer.kcat_per_h.cpu().numpy() / 3600.0).astype(np.float64)
    kcat_ref = reference.values.numpy().astype(np.float64)
    differs = ~np.isclose(kcat_ckpt, kcat_ref, rtol=1e-5, atol=0.0)
    max_diff = (
        float(np.abs(np.log10(kcat_ckpt[differs]) - np.log10(kcat_ref[differs])).max())
        if differs.any()
        else 0.0
    )
    tiers = np.array([KCAT_SIGMA_LADDER.tier_of(p) for p in reference.provenance])
    unit_reaction = units.unit_reaction.numpy()
    per_unit_capacity = (
        kcat_ckpt * 3600.0 * (e_u_np.mean(axis=0) if e_u is not None else 1.0e-3)
    )
    dominant_tier = np.full(len(gem.rxn_ids), "none", dtype=object)
    best = np.zeros(len(gem.rxn_ids))
    for u in range(units.n_units):
        j = int(unit_reaction[u])
        if per_unit_capacity[u] >= best[j]:
            best[j] = per_unit_capacity[u]
            dominant_tier[j] = tiers[u]
    max_genes = np.zeros(len(gem.rxn_ids), dtype=np.int64)
    np.maximum.at(max_genes, unit_reaction, genes_per_unit)

    median_width_per_reaction = np.median(width, axis=0)
    width_by_tier = {
        str(tier): _percentiles(
            median_width_per_reaction[dominant_tier == tier], "width"
        )
        for tier in ("measured", "homolog", "predicted", "default", "none")
        if (dominant_tier == tier).any()
    }
    bins = {"1": max_genes == 1, "2_4": (max_genes >= 2) & (max_genes <= 4), "ge5": max_genes >= 5}
    width_by_genes = {
        name: _percentiles(median_width_per_reaction[selector & has_gpr], "width")
        for name, selector in bins.items()
        if (selector & has_gpr).any()
    }
    c_j_median = np.median(c_j_np, axis=0)
    width_by_availability = {
        "c_j_lt_1": _percentiles(median_width_per_reaction[c_j_median < 1.0], "width"),
        "c_j_eq_1": _percentiles(median_width_per_reaction[c_j_median >= 1.0], "width"),
    }

    # THE PROMISCUITY OVERCOUNT, per gene: the enzyme the flux implies, against the
    # abundance the gene actually has.
    abs_v = np.abs(v.cpu().numpy())
    demand_per_unit = abs_v[:, unit_reaction] / np.maximum(kcat_ckpt * 3600.0, 1e-30)
    n_gem_genes = int(layer.n_gem_genes)
    implied_t = torch.zeros(abs_v.shape[0], n_gem_genes, dtype=torch.float64)
    implied_t.index_add_(
        1,
        layer.unit_gene_gene,
        torch.from_numpy(demand_per_unit[:, unit_gene_unit]).to(torch.float64),
    )
    implied = implied_t.numpy()
    if e_g is not None:
        overcount = implied / np.where(e_g_np > 0, e_g_np, np.nan)
    else:
        overcount = implied / 1.0e-3
    reactions_per_gene: dict[int, set[int]] = {}
    for u, g in zip(unit_gene_unit.tolist(), unit_gene_gene.tolist()):
        reactions_per_gene.setdefault(g, set()).add(int(unit_reaction[u]))
    n_promiscuous = sum(1 for s in reactions_per_gene.values() if len(s) > 1)

    diagnostics = reps["flux"]
    feas = float(diagnostics.get("feas_capacity_active_frac", torch.tensor(float("nan"))))
    if e_g is not None:
        protein = float((e_g * layer.mw_kda.unsqueeze(0)).sum(dim=1).mean())
    else:
        protein = float("nan")

    measured_median = float(np.median(width))
    ratio_to_quoted = measured_median / QUOTED_MEDIAN_BOX_WIDTH

    return CapacityAudit(
        cell=stem,
        arm=str(cell.arm),
        seed=int(cell.seed),
        parameterization=str(coverage["parameterization_raw"]),
        use_enzyme_capacity=bool(layer.config.use_enzyme_capacity),
        use_abundance_head=use_head,
        use_protein_budget=bool(layer.config.use_protein_budget),
        p_avail_g_per_gdw=float(layer.config.p_avail),
        constraint_final_scale=float(layer.config.constraint_final_scale),
        n_genotypes=int(width.shape[0]),
        n_reactions=int(width.shape[1]),
        n_units=n_units,
        box_matches_genotype_chain=box_matches_forward,
        box_width_all=_percentiles(width.reshape(-1), "width"),
        box_width_gpr_only=_percentiles(width[:, has_gpr].reshape(-1), "width"),
        box_width_no_capacity_control=_percentiles(width_nocap.reshape(-1), "width"),
        box_width_gem_base=_percentiles(base_width, "width"),
        explains_quoted_within_factor_3=bool(1 / 3 <= ratio_to_quoted <= 3),
        ratio_measured_over_quoted=ratio_to_quoted,
        availability_factor=_percentiles(availability, "factor"),
        capacity_factor=_percentiles(capacity_factor, "factor"),
        frac_c_j_below_one=float((c_j_np < 1.0 - 1e-9).mean()),
        frac_c_j_zero=float((c_j_np <= 1e-12).mean()),
        capacity_mmol_per_gdw_per_h=_percentiles(
            capacity_np[:, has_gpr].reshape(-1) if capacity is not None else capacity_np,
            "capacity",
        ),
        e_g_mmol_per_gdw=_percentiles(e_g_np.reshape(-1), "e_g"),
        e_u_mmol_per_gdw=_percentiles(e_u_np.reshape(-1), "e_u"),
        softmin_over_arithmetic_min=softmin_ratio,
        softmin_undershoot_multigene=softmin_multi,
        feas_capacity_active_frac=feas,
        protein_used_g_per_gdw=protein,
        protein_used_over_p_avail=protein / float(layer.config.p_avail),
        kcat_table_matches_checkpoint=bool(not differs.any()),
        n_units_kcat_differs=int(differs.sum()),
        max_abs_log10_kcat_diff=max_diff,
        tier_counts=reference.tier_counts(),
        width_by_tier=width_by_tier,
        width_by_genes_per_unit=width_by_genes,
        width_by_availability=width_by_availability,
        n_genes_in_more_than_one_reaction=n_promiscuous,
        promiscuity_overcount=_percentiles(overcount.reshape(-1), "overcount"),
        frac_genes_overcount_gt_1=float(np.nanmean(overcount > 1.0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", nargs="*", default=DEFAULT_CELLS)
    parser.add_argument("--out", default="capacity_checkpoint_audit.json")
    args = parser.parse_args()

    assert os.environ.get("PYTHONHASHSEED") == "0", (
        "set PYTHONHASHSEED=0. The k_cat resolver breaks ties through a hash-ordered "
        "container, so the reference table this audit compares the checkpoint against "
        "would otherwise differ from run to run on a handful of units."
    )

    tbx = _load_train_bx()
    cfg = tbx.load_config()
    dataset = tbx.build_dataset()

    audits: list[CapacityAudit] = []
    for name in args.cells:
        audit = audit_cell(name, tbx, dataset, cfg)
        audits.append(audit)
        print(
            f"\n=== {audit.cell} ({audit.arm}, seed {audit.seed}) ===\n"
            f"  box width p50 {audit.box_width_all['width_p50']:.4g} against the quoted "
            f"{QUOTED_MEDIAN_BOX_WIDTH:.1e} "
            f"(ratio {audit.ratio_measured_over_quoted:.3g}, "
            f"within a factor of 3: {audit.explains_quoted_within_factor_3})\n"
            f"  no-capacity control p50 {audit.box_width_no_capacity_control['width_p50']:.4g}; "
            f"GEM base p50 {audit.box_width_gem_base['width_p50']:.4g}\n"
            f"  availability factor p50 {audit.availability_factor['factor_p50']:.4g}; "
            f"capacity factor p50 {audit.capacity_factor['factor_p50']:.4g}\n"
            f"  E_g p50 {audit.e_g_mmol_per_gdw['e_g_p50']:.4g}; "
            f"E_u p50 {audit.e_u_mmol_per_gdw['e_u_p50']:.4g}; "
            f"capacity p50 {audit.capacity_mmol_per_gdw_per_h['capacity_p50']:.4g}\n"
            f"  softmin / arithmetic min p50 "
            f"{audit.softmin_over_arithmetic_min['softmin_ratio_p50']:.4g} "
            f"(multigene units {audit.softmin_undershoot_multigene['softmin_ratio_p50']:.4g})\n"
            f"  feas_capacity_active_frac {audit.feas_capacity_active_frac:.4g}; "
            f"protein/p_avail {audit.protein_used_over_p_avail:.4g}\n"
            f"  box matches the genotype chain: {audit.box_matches_genotype_chain}\n"
            f"  k_cat table matches the checkpoint: {audit.kcat_table_matches_checkpoint} "
            f"({audit.n_units_kcat_differs} units differ, max "
            f"{audit.max_abs_log10_kcat_diff:.3g} decades)\n"
            f"  promiscuity overcount p50 "
            f"{audit.promiscuity_overcount['overcount_p50']:.4g}, "
            f"fraction > 1: {audit.frac_genes_overcount_gt_1:.4g}, "
            f"{audit.n_genes_in_more_than_one_reaction} genes serve more than one reaction",
            flush=True,
        )

    payload = {
        "generated_at": timestamp(),
        "quoted_median_box_width": QUOTED_MEDIAN_BOX_WIDTH,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", ""),
        "cells": [a.model_dump() for a in audits],
    }
    out_path = osp.join(RESULTS_026, args.out)
    with open(out_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
