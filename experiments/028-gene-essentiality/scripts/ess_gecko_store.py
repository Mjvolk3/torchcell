# experiments/028-gene-essentiality/scripts/ess_gecko_store.py
# [[experiments.028-gene-essentiality.scripts.ess_gecko_store]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/ess_gecko_store

r"""GECKO-style enzyme-constrained parsimonious deletion flux stores, no proteome (round 114).

Round 113's ``ess_kcat_bounded_store.py`` capped every reaction by a kcat times a MEASURED
abundance. Here nothing is measured: the enzyme amounts are VARIABLES the program allocates
under one total-protein pool, which is GECKO's formulation (Sanchez 2017; GECKO 3, Chen
2024). One linear program per stage, in HiGHS through scipy, over

.. math::
    x = [p,\ n,\ e],\qquad p, n \in \mathbb{R}^{r}_{\ge 0},\ e \in \mathbb{R}^{n_e}_{\ge 0},

with :math:`v = p - n` the signed flux in the store's units (mmol/gDW/h, the GEM box
clamped to ``FLUX_SCALE`` and split EXACTLY as ``parsimonious_reference_flux`` splits it,
so a forced medium bound survives), and :math:`e_u` the usage (mmol/gDW) of catalytic unit
:math:`u`. Constraints:

* :math:`S p - S n = 0`;
* for every reaction :math:`j` whose units ALL carry a kcat and a molecular weight,
  :math:`p_j + n_j \le \sum_{u:\rho(u)=j} 3600\,k_{cat,u}\,e_u` (kcat in 1/s, so 3600 kcat
  is per hour; both directions draw on the same enzyme, which is the GECKO convention with
  one kcat per unit). Isozymes add; a complex is one unit whose kcat is the min over its
  subunits, the rule ``resolve_kcat_table`` already applies. A reaction with a unit lacking
  either number keeps only the GEM bound and is counted;
* the pool, :math:`\sum_u MW_u e_u \le P_{avail}`, with :math:`MW_u` in kDa = g/mmol the
  SUM of the SwissProt molecular weights of the unit's member genes (one copy of each
  subunit; no stoichiometry is recorded in the GEM), defined only when every member is in
  SwissProt, nothing defaulted.

Stage 1 maximizes growth. Stage 2 (parsimony, pFBA) holds growth at its stage-1 optimum
and minimizes :math:`\sum_j (p_j + n_j)`; the usages carry no cost in stage 2. The pool's
shadow price is the stage-1 dual of the pool row; an enzyme row "binds" when its stage-1
dual is nonzero. The pool usage a solve reports is the MINIMUM pool that carries the
stage-2 flux: because every unit belongs to exactly one reaction the enzyme rows are
separable, and the minimum is :math:`\sum_j |v_j| \min_{u:\rho(u)=j} MW_u/(3600 k_{cat,u})`
over the surviving units, in closed form. The stage-2 usage vector the solver happens to
return is degenerate (any usage above the minimum and under the pool is optimal), so it is
recorded for the wild type only and never compared across genotypes.

VARIANTS (``--variant``; the enzyme program always runs on the predicted kcat tier):

* ``pool125``: P_avail 0.125 g/gDW (``GECKO3_YEAST_PROTEIN_POOL``), kcat point values.
* ``pool250``: 0.25, the f P_tot product before saturation (``PROTEIN_POOL_SETTINGS[0.25]``).
* ``pool125s1``: 0.125, every kcat widened to its 1-sigma upper edge
  (``kcat_ceiling_sigma_k=1``).
* ``pool125sign`` / ``pool125sign30``: ``pool125`` plus thermodynamic DIRECTION constraints
  from the union-external standard reaction energies the 028 flux layer used
  (``resolve_reaction_delta_g(..., ThermoSource.UNION_EXTERNAL)``): a reaction with
  :math:`\Delta_r G'^\circ \le -T` loses its reverse part, one with :math:`\ge +T` its
  forward part, :math:`T` = 10 or 30 kJ/mol; :math:`|\Delta| < T` and unpriced reactions
  keep the GEM box. The reactions whose fixed direction conflicts with the plain pFBA
  wild-type flux are listed in the summary.
* ``pool125minenz``: ``pool125`` with stage 2 minimizing the enzyme MASS
  :math:`\sum_u MW_u e_u` (GECKO's minimal enzyme usage) instead of the total flux; the
  usages are then determinate and the pool usage is the LP's own.
* ``plainfba``: NO enzyme constraints and NO parsimony: the growth-maximizing LP alone,
  HiGHS dual simplex at default tolerances (``method="highs-ds"``), the solver's vertex
  as the vector.
* ``pfbalambda3`` / ``pfbalambda4``: NO enzyme constraints, parsimony INSIDE the objective
  as one stage, :math:`\max\ v_b - \lambda \sum_j (p_j + n_j)`, :math:`\lambda` = 1e-3 or
  1e-4 (1/h per mmol/gDW/h). The growth reached is recorded beside the two-stage pFBA
  growth of the plain store, per gene and for the wild type.

The pool is NEVER scaled: if the wild type reaches less than ``MIN_WILD_TYPE_FRACTION`` of
its unconstrained growth the store is still written and the fact is printed and recorded.
A deletion fixes :math:`e_u = 0` for every unit containing the gene and closes the
reactions ``knock_out_model_genes`` closes. Lethal (growth below ``LETHAL_GROWTH``) is the
zero row, as in the plain store.

Writes ``$DATA_ROOT/data/torchcell/reference_flux/ess_solve_encodings_YPD_gecko<variant>.npz``
(``genes``, ``v_deletion`` float32 in the model's signed reaction order, ``v_wild_type``,
``rxn_ids``, ``provenance`` = a plain ``StoreProvenance``, ``gecko_provenance`` = the
``GeckoStoreProvenance`` record, ``growth_ratio``, ``pool_usage`` (nan where the variant
has no pool)) and ``results/ess_gecko_store_<variant>.json``. Run from the repo root::

    PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD \
        ~/miniconda3/envs/torchcell/bin/python \
        experiments/028-gene-essentiality/scripts/ess_gecko_store.py \
        --variant pool125 --processes 24
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import os.path as osp
import sys
import time
from multiprocessing import Pool
from typing import Any, Literal

import numpy as np
import torch
from cobra.manipulation import knock_out_model_genes
from dotenv import load_dotenv
from pydantic import BaseModel
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, hstack, vstack

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "results")
TABLE_DIR = osp.join(DATA_ROOT, "data/torchcell/reference_flux")
PLAIN_STORE = osp.join(TABLE_DIR, "ess_solve_encodings_YPD_nuc.npz")
#: See the note at the first solve.
HIGHS_OPTIONS = {"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9}
OED_SCEREVISIAE = osp.join(
    DATA_ROOT, "data/enzyme_kinetics/open_enzyme_database/scerevisiae"
)
OED_ALL = osp.join(DATA_ROOT, "data/enzyme_kinetics/open_enzyme_database/all")

sys.path.insert(0, osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "scripts"))
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from ess_reference_flux_table import (  # noqa: E402
    FLUX_SCALE,
    MediaApplied,
    StoreProvenance,
    labeled_genes,
)
from train_flux import media_report  # noqa: E402

from torchcell.metabolism.constraints import build_gem_tensors  # noqa: E402
from torchcell.metabolism.flux_layer import (  # noqa: E402
    genotype_key,
    parsimonious_reference_flux,
)
from torchcell.metabolism.parameters import (  # noqa: E402
    KCAT_SIGMA_LADDER,
    KINETICS_MIRROR,
    PLAUSIBILITY_RESULT,
    PROTEIN_POOL_SETTINGS,
    THERMO_EQUILIBRATOR_MIRROR,
    VERIFIED_PAIRS_FEATURE,
    ParameterProvenance,
    ProteinPoolProvenance,
    SubstrateMatch,
    ThermoSource,
    load_predicted_unit_table,
    molecular_weight_table,
    resolve_kcat_table,
    resolve_reaction_delta_g,
)
from torchcell.metabolism.yeast_GEM import YeastGEM  # noqa: E402

Variant = Literal[
    "pool125",
    "pool250",
    "pool125s1",
    "pool125sign",
    "pool125sign30",
    "pool125minenz",
    "plainfba",
    "pfbalambda3",
    "pfbalambda4",
]
Stage2 = Literal["flux", "enzyme_mass"]
Solver = Literal["highs", "highs-ds"]


class VariantSpec(BaseModel):
    """What one ``--variant`` runs: pool, kcat widening, direction rule, objective."""

    model_config = {"extra": "forbid"}

    name: str
    p_avail: float | None
    sigma_k: float = 0.0
    sign_threshold_kj_per_mol: float | None = None
    stage2: Stage2 | None = "flux"
    lambda_parsimony: float | None = None
    solver: Solver = "highs"


VARIANT_SPECS: dict[str, VariantSpec] = {
    "pool125": VariantSpec(name="pool125", p_avail=0.125),
    "pool250": VariantSpec(name="pool250", p_avail=0.25),
    "pool125s1": VariantSpec(name="pool125s1", p_avail=0.125, sigma_k=1.0),
    "pool125sign": VariantSpec(
        name="pool125sign", p_avail=0.125, sign_threshold_kj_per_mol=10.0
    ),
    "pool125sign30": VariantSpec(
        name="pool125sign30", p_avail=0.125, sign_threshold_kj_per_mol=30.0
    ),
    "pool125minenz": VariantSpec(name="pool125minenz", p_avail=0.125, stage2="enzyme_mass"),
    "plainfba": VariantSpec(name="plainfba", p_avail=None, stage2=None, solver="highs-ds"),
    "pfbalambda3": VariantSpec(
        name="pfbalambda3", p_avail=None, stage2=None, lambda_parsimony=1e-3
    ),
    "pfbalambda4": VariantSpec(
        name="pfbalambda4", p_avail=None, stage2=None, lambda_parsimony=1e-4
    ),
}
VARIANTS: tuple[str, ...] = tuple(VARIANT_SPECS)
#: Below this fraction of the unconstrained growth the wild type is reported as not
#: reaching the pool's growth; the store is written regardless.
MIN_WILD_TYPE_FRACTION = 0.1
#: A deletion whose growth is below this is lethal and stored as the zero row.
LETHAL_GROWTH = 1e-6
#: Stage 2 holds growth at this fraction of the stage-1 optimum. At exactly 1.0 HiGHS's
#: simplex declares the floor infeasible at its 1e-7 feasibility tolerance for some pools
#: (measured on the pool250 and pool125s1 wild types), so the floor sits 1e-6 below the
#: optimum; the value is recorded as the store's ``growth_fraction``.
STAGE2_GROWTH_FRACTION = 1.0 - 1e-6
#: A stage-1 dual below this in magnitude is zero: the row does not bind.
DUAL_TOL = 1e-9
#: Genes whose pool usage differs from the wild type's by more than this fraction of the
#: pool are the ones whose allocation changed.
POOL_MOVED_TOL = 0.01
#: A wild-type flux below this in magnitude carries no direction.
FLUX_TOL = 1e-9
SPLIT_FILE = "ess_fcl_val0.2_s0.json.gz"
SECONDS_PER_HOUR = 3600.0


class ThermoApplied(BaseModel):
    """The standard reaction energies a direction rule read, and what it fixed."""

    model_config = {"extra": "forbid"}

    source: str
    source_paths: dict[str, str]
    sha256: dict[str, str]
    counts: dict[str, int]
    n_reactions_with_energy: int
    threshold_kj_per_mol: float
    n_fixed_forward: int
    n_fixed_reverse: int
    n_bound_conflicts: int
    bound_conflicts: list[str]
    n_conflicts_with_plain_wild_type_flux: int
    conflicts_with_plain_wild_type_flux: list[dict[str, Any]]


class GeckoStoreProvenance(StoreProvenance):
    """``StoreProvenance`` plus the enzyme-constrained program's record."""

    variant: str
    spec: VariantSpec
    pool: ProteinPoolProvenance | None
    thermo: ThermoApplied | None
    kcat_tier_counts: dict[str, int]
    kcat_sigma_k: float
    kcat_sigma_ladder_version: str
    n_units: int
    n_units_with_kcat: int
    n_units_with_mw: int
    n_units_with_kcat_and_mw: int
    n_units_in_program: int
    n_reactions_with_gpr: int
    n_reactions_enzyme_constrained: int
    n_reactions_with_gpr_kept_gem_bound: int
    n_variables: int
    n_equality_rows: int
    n_inequality_rows_stage1: int
    wild_type_growth_unconstrained: float
    wild_type_growth_constrained: float
    wild_type_growth_fraction: float
    wild_type_reaches_min_fraction: bool
    wild_type_pool_usage_fraction: float | None
    wild_type_pool_usage_fraction_stage2_lp: float | None
    wild_type_enzyme_mass_g_per_gdw: float | None
    wild_type_pool_dual_per_hour_per_g_gdw: float | None
    wild_type_pool_binding: bool | None
    n_enzyme_rows_binding_wild_type: int | None
    n_units_nonzero_usage_wild_type: int | None
    n_units_nonzero_usage_wild_type_stage2_lp: int | None
    mw_rule: str
    kcat_rule: str
    program: str
    pool_usage_rule: str


class GeneRow(BaseModel):
    gene: str
    growth: float
    growth_ratio: float
    plain_growth: float
    plain_growth_ratio: float
    lethal: bool
    pool_usage_fraction: float | None
    enzyme_mass_g_per_gdw: float | None
    n_units_nonzero_usage: int | None
    n_units_usage_changed_vs_wild_type: int | None
    allocation_mass_moved_fraction_of_pool: float | None
    pool_dual: float | None
    n_enzyme_rows_binding: int | None
    relative_l2_log1p_to_plain: float


class Program(BaseModel):
    """The program's fixed parts; only bounds change per genotype."""

    model_config = {"arbitrary_types_allowed": True}

    n_rxn: int
    n_e: int
    biomass: int
    a_eq: Any
    a_ub: Any
    b_ub: np.ndarray
    p_bounds: list[tuple[float, float]]
    n_bounds: list[tuple[float, float]]
    e_reaction: np.ndarray
    e_kcat_per_h: np.ndarray
    e_mw: np.ndarray
    e_unit: np.ndarray
    row_reaction: np.ndarray
    p_avail: float | None
    stage2: Stage2 | None
    lambda_parsimony: float | None
    solver: Solver


class Solve(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    v: np.ndarray
    e: np.ndarray
    growth: float
    pool_dual: float
    n_enzyme_rows_binding: int
    success: bool


# --------------------------------------------------------------------------------------
# worker state (inherited by fork; set before the Pool is created)
# --------------------------------------------------------------------------------------
_P: Program | None = None
_KNOCKED_RXN: dict[str, np.ndarray] = {}
_KNOCKED_E: dict[str, np.ndarray] = {}
_E_WT: np.ndarray | None = None


def split_box(lb: np.ndarray, ub: np.ndarray) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """The split box, exactly equal to the original one (``parsimonious_reference_flux``)."""
    p_bounds: list[tuple[float, float]] = []
    n_bounds: list[tuple[float, float]] = []
    for low, high in zip(lb.tolist(), ub.tolist(), strict=True):
        if low >= 0.0:
            p_bounds.append((low, high))
            n_bounds.append((0.0, 0.0))
        elif high <= 0.0:
            p_bounds.append((0.0, 0.0))
            n_bounds.append((-high, -low))
        else:
            p_bounds.append((0.0, high))
            n_bounds.append((0.0, -low))
    return p_bounds, n_bounds


def build_program(
    s_dense: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    biomass: int,
    e_unit: np.ndarray,
    e_reaction: np.ndarray,
    e_kcat_per_h: np.ndarray,
    e_mw: np.ndarray,
    spec: VariantSpec,
) -> Program:
    m, r = s_dense.shape
    n_e = int(e_unit.size)
    s = csr_matrix(s_dense)
    a_eq = hstack([s, -s, csr_matrix((m, n_e))], format="csr") if n_e else hstack(
        [s, -s], format="csr"
    )
    constrained = np.unique(e_reaction)
    if n_e:
        row_of = {int(j): k for k, j in enumerate(constrained.tolist())}
        n_rows = len(constrained)
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        for k, j in enumerate(constrained.tolist()):
            rows += [k, k]
            cols += [j, r + j]
            vals += [1.0, 1.0]
        # The usage columns are scaled to CAPACITY, e' = 3600 kcat e (mmol/gDW/h), so every
        # enzyme-row coefficient is -1 and the 3.6 to 2e9 range of 3600 kcat sits in the
        # pool row alone; the solution is mapped back to e = e' / (3600 kcat). Exact.
        for i in range(n_e):
            rows.append(row_of[int(e_reaction[i])])
            cols.append(2 * r + i)
            vals.append(-1.0)
        enzyme = coo_matrix((vals, (rows, cols)), shape=(n_rows, 2 * r + n_e)).tocsr()
        assert spec.p_avail is not None
        pool = coo_matrix(
            (
                (e_mw / e_kcat_per_h).astype(np.float64),
                (np.zeros(n_e, dtype=np.int64), 2 * r + np.arange(n_e)),
            ),
            shape=(1, 2 * r + n_e),
        ).tocsr()
        a_ub = vstack([enzyme, pool], format="csr")
        b_ub = np.concatenate([np.zeros(n_rows), [spec.p_avail]])
    else:
        a_ub = None
        b_ub = np.zeros(0)
    p_bounds, n_bounds = split_box(lb, ub)
    return Program(
        n_rxn=r,
        n_e=n_e,
        biomass=biomass,
        a_eq=a_eq,
        a_ub=a_ub,
        b_ub=b_ub,
        p_bounds=p_bounds,
        n_bounds=n_bounds,
        e_reaction=e_reaction,
        e_kcat_per_h=e_kcat_per_h,
        e_mw=e_mw,
        e_unit=e_unit,
        row_reaction=constrained,
        p_avail=spec.p_avail,
        stage2=spec.stage2,
        lambda_parsimony=spec.lambda_parsimony,
        solver=spec.solver,
    )


def _failed(prog: Program, pool_dual: float = 0.0, n_binding: int = 0) -> Solve:
    return Solve(
        v=np.zeros(prog.n_rxn),
        e=np.zeros(prog.n_e),
        growth=0.0,
        pool_dual=pool_dual,
        n_enzyme_rows_binding=n_binding,
        success=False,
    )


def solve_program(
    prog: Program, knocked_rxn: np.ndarray, knocked_e: np.ndarray
) -> Solve:
    """Stage 1 (max growth, optionally minus lambda x total flux) then the stage 2 the
    variant asks for (min total flux or min enzyme mass at that growth), one genotype."""
    r, n_e = prog.n_rxn, prog.n_e
    p_b = list(prog.p_bounds)
    n_b = list(prog.n_bounds)
    for j in knocked_rxn.tolist():
        p_b[j] = (0.0, 0.0)
        n_b[j] = (0.0, 0.0)
    e_b: list[tuple[float, float | None]] = [(0.0, None)] * n_e
    for i in knocked_e.tolist():
        e_b[i] = (0.0, 0.0)
    bounds = p_b + n_b + e_b
    c1 = np.zeros(2 * r + n_e)
    if prog.lambda_parsimony is not None:
        c1[: 2 * r] = prog.lambda_parsimony
    c1[prog.biomass] += -1.0
    c1[r + prog.biomass] += 1.0
    b_eq = np.zeros(prog.a_eq.shape[0])
    # MEASURED: the glutathione biomass demand is 1e-6 x growth, which at pool-limited
    # growth (0.067 /h) is 6.7e-8, under HiGHS's default primal feasibility tolerance of
    # 1e-7, so GSH1 and GSH2 deletions read viable under a pool while lethal on the plain
    # box. The tighter tolerance makes them lethal under the pool as well.
    first = linprog(
        c1,
        A_ub=prog.a_ub,
        b_ub=prog.b_ub if prog.a_ub is not None else None,
        A_eq=prog.a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method=prog.solver,
        options=HIGHS_OPTIONS,
    )
    if not first.success:
        return _failed(prog)
    x1 = np.asarray(first.x, dtype=np.float64)
    v1 = x1[:r] - x1[r : 2 * r]
    z_star = float(v1[prog.biomass])
    pool_dual, n_binding = 0.0, 0
    if prog.a_ub is not None:
        marginals = np.asarray(first.ineqlin.marginals, dtype=np.float64)
        # d(objective)/d(b_ub) is nonpositive for a minimization; the shadow price of the
        # pool for growth is its negative, in (1/h) per (g/gDW).
        pool_dual = float(-marginals[-1])
        n_binding = int((np.abs(marginals[:-1]) > DUAL_TOL).sum())
    if prog.stage2 is None:
        return Solve(
            v=v1,
            e=x1[2 * r :] / prog.e_kcat_per_h,
            growth=z_star,
            pool_dual=pool_dual,
            n_enzyme_rows_binding=n_binding,
            success=True,
        )
    growth_row = np.zeros(2 * r + n_e)
    growth_row[prog.biomass] = -1.0
    growth_row[r + prog.biomass] = 1.0
    growth_ub = csr_matrix(growth_row.reshape(1, -1))
    a_ub2 = growth_ub if prog.a_ub is None else vstack([prog.a_ub, growth_ub], format="csr")
    b_ub2 = np.concatenate([prog.b_ub, [-STAGE2_GROWTH_FRACTION * z_star]])
    c2 = np.zeros(2 * r + n_e)
    if prog.stage2 == "flux":
        c2[: 2 * r] = 1.0
    else:
        c2[2 * r :] = prog.e_mw / prog.e_kcat_per_h
    second = linprog(
        c2,
        A_ub=a_ub2,
        b_ub=b_ub2,
        A_eq=prog.a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method=prog.solver,
        options=HIGHS_OPTIONS,
    )
    if not second.success:
        return _failed(prog, pool_dual, n_binding)
    x2 = np.asarray(second.x, dtype=np.float64)
    v = x2[:r] - x2[r : 2 * r]
    return Solve(
        v=v,
        e=x2[2 * r :] / prog.e_kcat_per_h,
        growth=float(v[prog.biomass]),
        pool_dual=pool_dual,
        n_enzyme_rows_binding=n_binding,
        success=True,
    )


def minimum_pool_allocation(
    prog: Program, v: np.ndarray, knocked_e: np.ndarray
) -> tuple[float, np.ndarray]:
    """The least pool (g/gDW) that carries ``v``, and the usage vector realizing it.

    Every unit constrains exactly one reaction, so per reaction the cheapest unit
    (min MW/kcat among the surviving ones) takes the whole flux.
    """
    cost = prog.e_mw / prog.e_kcat_per_h
    alive = np.ones(prog.n_e, dtype=bool)
    alive[knocked_e] = False
    e = np.zeros(prog.n_e)
    for j in prog.row_reaction.tolist():
        flux = abs(float(v[j]))
        if flux <= FLUX_TOL:
            continue
        members = np.where((prog.e_reaction == j) & alive)[0]
        assert members.size > 0, (
            f"reaction {j} carries flux {flux} with every unit knocked out"
        )
        best = members[np.argmin(cost[members])]
        e[best] = flux / prog.e_kcat_per_h[best]
    return float(np.dot(prog.e_mw, e)), e


def usage_of(
    prog: Program, solve: Solve, knocked_e: np.ndarray
) -> tuple[float, int, np.ndarray]:
    """Enzyme mass (g/gDW), the number of units in use and the usage vector, by the
    variant's rule (the LP's own for enzyme_mass, the minimum allocation otherwise)."""
    if prog.n_e == 0:
        return float("nan"), 0, np.zeros(0)
    if prog.stage2 == "enzyme_mass":
        e = np.where(solve.e > 1e-12, solve.e, 0.0)
        return float(np.dot(prog.e_mw, e)), int((e > 0).sum()), e
    mass, e = minimum_pool_allocation(prog, solve.v, knocked_e)
    return mass, int((e > 0).sum()), e


def allocation_change(prog: Program, e: np.ndarray) -> tuple[int, float]:
    """How the usage differs from the wild type's: units whose usage moved, and the mass
    moved (sum |MW (e - e_wt)|) as a fraction of the pool."""
    assert _E_WT is not None
    if prog.n_e == 0:
        return 0, float("nan")
    diff = e - _E_WT
    n_changed = int((np.abs(diff) > 1e-12).sum())
    assert prog.p_avail is not None
    return n_changed, float(np.dot(prog.e_mw, np.abs(diff))) / prog.p_avail


def solve_gene(
    gene: str,
) -> tuple[str, np.ndarray, float, float, int, float, int, int, float]:
    assert _P is not None
    knocked_rxn = _KNOCKED_RXN[gene]
    knocked_e = _KNOCKED_E[gene]
    solve = solve_program(_P, knocked_rxn, knocked_e)
    if not solve.success or solve.growth < LETHAL_GROWTH:
        n_changed, moved = allocation_change(_P, np.zeros(_P.n_e))
        return (
            gene,
            np.zeros(_P.n_rxn, dtype=np.float32),
            float(solve.growth) if solve.success else 0.0,
            0.0 if _P.n_e else float("nan"),
            0,
            solve.pool_dual,
            solve.n_enzyme_rows_binding,
            n_changed,
            moved,
        )
    mass, n_used, e = usage_of(_P, solve, knocked_e)
    n_changed, moved = allocation_change(_P, e)
    return (
        gene,
        solve.v.astype(np.float32),
        solve.growth,
        mass,
        n_used,
        solve.pool_dual,
        solve.n_enzyme_rows_binding,
        n_changed,
        moved,
    )


def apply_sign_rule(
    lb: np.ndarray,
    ub: np.ndarray,
    delta: np.ndarray,
    mask: np.ndarray,
    threshold: float,
    rxn_ids: list[str],
    plain_wild_type: np.ndarray,
    table: Any,
) -> tuple[np.ndarray, np.ndarray, ThermoApplied]:
    """Fix the direction of every priced reaction whose |delta_r G'| reaches the threshold."""
    forward_only = mask & (delta <= -threshold)
    reverse_only = mask & (delta >= threshold)
    new_lb = np.where(forward_only, np.maximum(lb, 0.0), lb)
    new_ub = np.where(reverse_only, np.minimum(ub, 0.0), ub)
    bound_conflict = new_lb > new_ub
    wt_conflict = (forward_only & (plain_wild_type < -FLUX_TOL)) | (
        reverse_only & (plain_wild_type > FLUX_TOL)
    )
    conflicts = [
        {
            "reaction": rxn_ids[j],
            "delta_r_g0_kj_per_mol": float(delta[j]),
            "plain_wild_type_flux": float(plain_wild_type[j]),
            "gem_lb": float(lb[j]),
            "gem_ub": float(ub[j]),
        }
        for j in np.where(wt_conflict)[0].tolist()
    ]
    record = ThermoApplied(
        source=str(table.source),
        source_paths=dict(table.source_paths),
        sha256=dict(table.sha256),
        counts=dict(table.counts),
        n_reactions_with_energy=int(mask.sum()),
        threshold_kj_per_mol=threshold,
        n_fixed_forward=int(forward_only.sum()),
        n_fixed_reverse=int(reverse_only.sum()),
        n_bound_conflicts=int(bound_conflict.sum()),
        bound_conflicts=[rxn_ids[j] for j in np.where(bound_conflict)[0].tolist()],
        n_conflicts_with_plain_wild_type_flux=len(conflicts),
        conflicts_with_plain_wild_type_flux=conflicts,
    )
    return new_lb, new_ub, record


def main() -> None:
    global _P, _KNOCKED_RXN, _KNOCKED_E, _E_WT
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--processes", type=int, default=24)
    parser.add_argument(
        "--limit-genes",
        type=int,
        default=0,
        help="Smoke test: solve only the first N genes (0 = every labeled gene).",
    )
    parser.add_argument(
        "--out-dir",
        default=TABLE_DIR,
        help="Smoke test: write the npz here instead of the reference_flux directory.",
    )
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    args = parser.parse_args()
    spec = VARIANT_SPECS[args.variant]
    variant = spec.name
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("PYTHONHASHSEED must be 0, as for the plain store.")
    torch.set_num_threads(1)
    started = time.time()

    plain = np.load(PLAIN_STORE, allow_pickle=False)
    plain_provenance = StoreProvenance.model_validate_json(str(plain["provenance"]))
    genes = [str(g) for g in plain["genes"]]
    assert genes == labeled_genes(SPLIT_FILE), "plain store genes differ from the split"
    plain_v = plain["v_deletion"].astype(np.float32)
    plain_wt = plain["v_wild_type"].astype(np.float64)
    if args.limit_genes:
        genes = genes[: args.limit_genes]
        plain_v = plain_v[: args.limit_genes]
    plain_ratio = plain["gene_flux_stats"][
        :, list(plain["stat_names"]).index("deletion_growth_ratio")
    ].astype(np.float64)[: len(genes)]
    plain_growth = plain_ratio * plain_provenance.wild_type_growth

    source = YeastGEM()
    model = source.model
    media = media_report(model, "YPD")
    assert media["bounds_sha256"] == plain_provenance.media_report.bounds_sha256
    gem = build_gem_tensors(model, source.model_dir)
    rxn_ids = list(gem.rxn_ids)
    assert rxn_ids == [str(r) for r in plain["rxn_ids"]], "rxn_ids differ from the plain store"
    n_rxn = len(rxn_ids)
    biomass = int(gem.biomass_index)
    s_dense = (gem.s.to_dense() if gem.s.is_sparse else gem.s).double().numpy()
    base_lb = gem.lb.clamp(-FLUX_SCALE, FLUX_SCALE).double().numpy()
    base_ub = gem.ub.clamp(-FLUX_SCALE, FLUX_SCALE).double().numpy()
    gem_genes = {g.id for g in model.genes}
    assert all(g in gem_genes for g in genes)

    unconstrained = parsimonious_reference_flux(
        gem.s, torch.from_numpy(base_lb).float(), torch.from_numpy(base_ub).float(), biomass
    ).objective_value
    assert abs(unconstrained - plain_provenance.wild_type_growth) < 1e-6, (
        unconstrained,
        plain_provenance.wild_type_growth,
    )

    # -- thermodynamic direction rule -----------------------------------------------------
    thermo: ThermoApplied | None = None
    lb, ub = base_lb, base_ub
    if spec.sign_threshold_kj_per_mol is not None:
        table = resolve_reaction_delta_g(
            gem, osp.join(DATA_ROOT, THERMO_EQUILIBRATOR_MIRROR), ThermoSource.UNION_EXTERNAL
        )
        delta = table.delta_r_g0.numpy().astype(np.float64)
        mask = table.mask.numpy().astype(bool)
        lb, ub, thermo = apply_sign_rule(
            base_lb, base_ub, delta, mask, spec.sign_threshold_kj_per_mol, rxn_ids, plain_wt, table
        )
        print(
            f"thermo {thermo.source}: {thermo.n_reactions_with_energy}/{n_rxn} reactions priced "
            f"({thermo.counts}); T {thermo.threshold_kj_per_mol} kJ/mol fixes "
            f"{thermo.n_fixed_forward} forward, {thermo.n_fixed_reverse} reverse; "
            f"{thermo.n_bound_conflicts} bound conflicts (lb > ub); "
            f"{thermo.n_conflicts_with_plain_wild_type_flux} conflicts with the plain pFBA "
            "wild-type flux",
            flush=True,
        )

    # -- the enzyme parameters ------------------------------------------------------------
    units = gem.catalytic_units
    unit_reaction = units.unit_reaction.numpy()
    unit_of = units.unit_gene_index[0].numpy()
    gene_of = units.unit_gene_index[1].numpy()
    pool: ProteinPoolProvenance | None = None
    tier_counts: dict[str, int] = {}
    has_kcat = np.zeros(units.n_units, dtype=bool)
    has_mw = np.zeros(units.n_units, dtype=bool)
    usable = np.zeros(units.n_units, dtype=bool)
    n_units_rxn = np.zeros(n_rxn, dtype=np.int64)
    np.add.at(n_units_rxn, unit_reaction, 1)
    has_gpr = n_units_rxn > 0
    constrained_rxn = np.zeros(n_rxn, dtype=bool)
    e_unit = np.zeros(0, dtype=np.int64)
    e_reaction = np.zeros(0, dtype=np.int64)
    e_kcat_per_h = np.zeros(0)
    e_mw = np.zeros(0)
    if spec.p_avail is not None:
        pool = PROTEIN_POOL_SETTINGS[spec.p_avail]
        kinetics_dir = osp.join(DATA_ROOT, KINETICS_MIRROR)
        substrates = SubstrateMatch.from_verified_pairs(
            osp.join(kinetics_dir, VERIFIED_PAIRS_FEATURE)
        )
        predicted = load_predicted_unit_table(
            "k_cat", units.n_units, kinetics_dir, osp.join(EXPERIMENT_ROOT, PLAUSIBILITY_RESULT)
        )
        table_kcat = resolve_kcat_table(
            units,
            source.model_dir,
            OED_SCEREVISIAE,
            predicted=predicted,
            substrates=substrates,
            homolog_mirror_dir=OED_ALL,
            kcat_ceiling_sigma_k=spec.sigma_k,
        )
        tier_counts = table_kcat.tier_counts()
        kcat_per_s = table_kcat.values.numpy().astype(np.float64)
        has_kcat = table_kcat.known_mask.numpy().astype(bool)
        mw = molecular_weight_table(source.model_dir, units.gene_ids)
        gene_mw = mw.values.numpy().astype(np.float64)
        gene_has_mw = np.array(
            [p is ParameterProvenance.SWISSPROT for p in mw.provenance], dtype=bool
        )
        unit_mw = np.zeros(units.n_units)
        np.add.at(unit_mw, unit_of, gene_mw[gene_of])
        unit_n_missing_mw = np.zeros(units.n_units, dtype=np.int64)
        np.add.at(unit_n_missing_mw, unit_of, (~gene_has_mw[gene_of]).astype(np.int64))
        has_mw = unit_n_missing_mw == 0
        usable = has_kcat & has_mw
        n_usable_rxn = np.zeros(n_rxn, dtype=np.int64)
        np.add.at(n_usable_rxn, unit_reaction, usable.astype(np.int64))
        constrained_rxn = has_gpr & (n_usable_rxn == n_units_rxn)
        in_program = usable & constrained_rxn[unit_reaction]
        e_unit = np.where(in_program)[0]
        e_reaction = unit_reaction[e_unit]
        e_kcat_per_h = kcat_per_s[e_unit] * SECONDS_PER_HOUR
        e_mw = unit_mw[e_unit]
        print(
            f"kcat tiers {tier_counts}, sigma_k {spec.sigma_k}; units with kcat "
            f"{int(has_kcat.sum())}, with MW {int(has_mw.sum())}, both {int(usable.sum())}, "
            f"in program {len(e_unit)}; reactions with GPR {int(has_gpr.sum())}, "
            f"enzyme-constrained {int(constrained_rxn.sum())}; pool {pool.name} "
            f"{spec.p_avail} g/gDW",
            flush=True,
        )

    prog = build_program(
        s_dense, lb, ub, biomass, e_unit, e_reaction, e_kcat_per_h, e_mw, spec
    )
    n_variables = 2 * n_rxn + prog.n_e
    n_eq = int(prog.a_eq.shape[0])
    n_ub = 0 if prog.a_ub is None else int(prog.a_ub.shape[0])
    print(
        f"program: {n_variables} variables ({n_rxn} p, {n_rxn} n, {prog.n_e} e), "
        f"{n_eq} equality rows, {n_ub} inequality rows"
        + (f" ({n_ub - 1} enzyme + 1 pool)" if n_ub else "")
        + f"; stage 2 {spec.stage2}, lambda {spec.lambda_parsimony}, solver {spec.solver}; "
        f"{time.time() - started:.0f}s",
        flush=True,
    )
    _P = prog

    # -- the wild type ------------------------------------------------------------------
    no_rxn = np.zeros(0, dtype=np.int64)
    no_e = np.zeros(0, dtype=np.int64)
    wt = solve_program(prog, no_rxn, no_e)
    if not wt.success:
        # No store can be written from an infeasible wild type; the summary records what
        # the program would need relaxed and the run ends there.
        message = f"the wild-type program of variant {variant} did not solve" + (
            f"; the direction rule at {spec.sign_threshold_kj_per_mol} kJ/mol fixed "
            f"{thermo.n_fixed_forward + thermo.n_fixed_reverse} reactions, of which "
            f"{thermo.n_conflicts_with_plain_wild_type_flux} conflict with the plain pFBA "
            "wild-type flux (listed under thermo.conflicts_with_plain_wild_type_flux)"
            if thermo is not None
            else ""
        )
        print(message, flush=True)
        out = osp.join(args.results_dir, f"ess_gecko_store_{variant}.json")
        assert not osp.exists(out), f"{out} exists; outputs are never overwritten"
        with open(out, "w") as fh:
            json.dump(
                {
                    "store": None,
                    "variant": variant,
                    "spec": spec.model_dump(),
                    "wild_type_infeasible": True,
                    "message": message,
                    "n_variables": n_variables,
                    "n_equality_rows": n_eq,
                    "n_inequality_rows_stage1": n_ub,
                    "wild_type_growth_unconstrained": float(unconstrained),
                    "thermo": thermo.model_dump() if thermo is not None else None,
                    "seconds": time.time() - started,
                },
                fh,
                indent=2,
            )
        print(out, flush=True)
        return
    has_pool = prog.n_e > 0
    wt_mass, wt_n_used, wt_e = usage_of(prog, wt, no_e)
    _E_WT = wt_e
    wt_pool_fraction = wt_mass / spec.p_avail if has_pool else None
    wt_pool_fraction_lp = (
        float(np.dot(prog.e_mw, wt.e)) / spec.p_avail if has_pool else None
    )
    wt_fraction = wt.growth / unconstrained
    reaches = wt_fraction >= MIN_WILD_TYPE_FRACTION
    print(
        f"wild type: growth {wt.growth:.4f} ({wt_fraction:.3f} of unconstrained "
        f"{unconstrained:.4f})"
        + (
            f", pool used {wt_pool_fraction:.3f} (stage-2 LP usage {wt_pool_fraction_lp:.3f}), "
            f"pool dual {wt.pool_dual:.4f} /h per g/gDW, {wt.n_enzyme_rows_binding} enzyme "
            f"rows binding, {wt_n_used} units with nonzero usage, "
            f"{int((wt.e > 1e-12).sum())} in the stage-2 LP"
            if has_pool
            else ""
        ),
        flush=True,
    )
    if not reaches:
        print(
            f"WILD TYPE DOES NOT REACH {MIN_WILD_TYPE_FRACTION} OF ITS UNCONSTRAINED GROWTH "
            f"under variant {variant}: {wt.growth:.4f} of {unconstrained:.4f}. The store is "
            "written as-is; nothing is scaled.",
            flush=True,
        )

    # -- the deletions ------------------------------------------------------------------
    rxn_index = {r: i for i, r in enumerate(rxn_ids)}
    gene_index = {g: i for i, g in enumerate(units.gene_ids)}
    unit_index_in_program = np.full(units.n_units, -1, dtype=np.int64)
    unit_index_in_program[e_unit] = np.arange(len(e_unit))
    for gene in genes:
        with model:
            knocked = knock_out_model_genes(model, [gene])
            _KNOCKED_RXN[gene] = np.array(
                [rxn_index[r.id] for r in knocked], dtype=np.int64
            )
        gene_units = unit_of[gene_of == gene_index[gene]]
        program_units = unit_index_in_program[gene_units]
        _KNOCKED_E[gene] = np.unique(program_units[program_units >= 0])
    print(f"{len(genes)} genes, knockouts indexed, {time.time() - started:.0f}s", flush=True)

    with Pool(args.processes) as pool_:
        results = pool_.map(solve_gene, genes, chunksize=4)
    print(f"{len(genes)} solves, {time.time() - started:.0f}s", flush=True)
    order = {g: i for i, g in enumerate(genes)}
    v_deletion = np.zeros((len(genes), n_rxn), dtype=np.float32)
    growth = np.zeros(len(genes))
    mass = np.full(len(genes), np.nan)
    n_used = np.zeros(len(genes), dtype=np.int64)
    pool_dual = np.zeros(len(genes))
    n_binding = np.zeros(len(genes), dtype=np.int64)
    n_units_changed = np.zeros(len(genes), dtype=np.int64)
    mass_moved = np.full(len(genes), np.nan)
    for gene, v, g, m_, nu, dual, nb, nc, mm in results:
        i = order[gene]
        v_deletion[i] = v
        growth[i] = g
        mass[i] = m_
        n_used[i] = nu
        pool_dual[i] = dual
        n_binding[i] = nb
        n_units_changed[i] = nc
        mass_moved[i] = mm
    lethal = growth < LETHAL_GROWTH
    ratio = growth / wt.growth
    pool_usage = mass / spec.p_avail if has_pool else np.full(len(genes), np.nan)

    enc_new = np.log1p(np.abs(v_deletion.astype(np.float64)))
    enc_plain = np.log1p(np.abs(plain_v.astype(np.float64)))
    rel = np.linalg.norm(enc_new - enc_plain, axis=1) / np.maximum(
        np.linalg.norm(enc_plain, axis=1), 1e-12
    )
    rows = [
        GeneRow(
            gene=g,
            growth=float(growth[i]),
            growth_ratio=float(ratio[i]),
            plain_growth=float(plain_growth[i]),
            plain_growth_ratio=float(plain_ratio[i]),
            lethal=bool(lethal[i]),
            pool_usage_fraction=float(pool_usage[i]) if has_pool else None,
            enzyme_mass_g_per_gdw=float(mass[i]) if has_pool else None,
            n_units_nonzero_usage=int(n_used[i]) if has_pool else None,
            n_units_usage_changed_vs_wild_type=int(n_units_changed[i]) if has_pool else None,
            allocation_mass_moved_fraction_of_pool=float(mass_moved[i]) if has_pool else None,
            pool_dual=float(pool_dual[i]) if has_pool else None,
            n_enzyme_rows_binding=int(n_binding[i]) if has_pool else None,
            relative_l2_log1p_to_plain=float(rel[i]),
        )
        for i, g in enumerate(genes)
    ]

    flux_sha = hashlib.sha256(v_deletion.tobytes()).hexdigest()
    if has_pool:
        assert pool is not None
        method = (
            f"GECKO-style enzyme-constrained pFBA (variant {variant}): scipy HiGHS LPs "
            f"over [p, n, e] with S p - S n = 0, p_j + n_j <= sum_u 3600 kcat_u e_u on "
            f"{int(constrained_rxn.sum())} reactions, sum_u MW_u e_u <= {spec.p_avail} "
            f"g/gDW ({pool.name}), kcat sigma_k {spec.sigma_k}"
            + (
                f", directions fixed at |delta_r G'| >= {spec.sign_threshold_kj_per_mol} "
                "kJ/mol from the union-external standard reaction energies"
                if thermo is not None
                else ""
            )
            + f"; stage 1 max growth, stage 2 (growth held at {STAGE2_GROWTH_FRACTION!r} "
            "of the stage-1 optimum, the usage columns scaled to capacity 3600 kcat_u e_u) "
            + (
                "min sum(p + n) with the usages free"
                if spec.stage2 == "flux"
                else "min sum_u MW_u e_u"
            )
            + "; NO measured proteome; gene deleted by fixing its units' usages at 0 and "
            "closing the reactions cobra.manipulation.knock_out_model_genes closes"
        )
    elif spec.stage2 is None and spec.lambda_parsimony is None:
        method = (
            "growth-maximizing LP alone (no enzyme constraints, no parsimony) on the "
            "split GEM box, scipy linprog method highs-ds (HiGHS dual simplex, default "
            "tolerances), the solver's vertex as the vector; gene deleted through "
            "cobra.manipulation.knock_out_model_genes"
        )
    else:
        method = (
            f"one-stage penalized FBA: max v_b - {spec.lambda_parsimony:g} x sum(p + n) "
            "on the split GEM box (no enzyme constraints), scipy HiGHS; gene deleted "
            "through cobra.manipulation.knock_out_model_genes"
        )
    base = dict(
        gem=osp.basename(source.model_dir),
        growth_fraction=STAGE2_GROWTH_FRACTION if spec.stage2 is not None else 1.0,
        flux_scale=FLUX_SCALE,
        method=method + "; written by ess_gecko_store.py",
        solver=spec.solver,
        n_nonoptimal_zero_flux=0,
        medium="YPD",
        media_report=MediaApplied.model_validate(media),
        split_file=SPLIT_FILE,
        wild_type_growth=wt.growth,
        wild_type_key=genotype_key([]),
        n_sets=len(genes) + 1,
        n_lethal_zero_flux=int(lethal.sum()),
        n_genes_outside_the_gem=0,
        background_genes=[],
        python_hash_seed=os.environ.get("PYTHONHASHSEED"),
        flux_sha256=flux_sha,
    )
    provenance = StoreProvenance(**base)
    gecko_provenance = GeckoStoreProvenance(
        **base,
        variant=variant,
        spec=spec,
        pool=pool,
        thermo=thermo,
        kcat_tier_counts=tier_counts,
        kcat_sigma_k=spec.sigma_k,
        kcat_sigma_ladder_version=KCAT_SIGMA_LADDER.version,
        n_units=int(units.n_units),
        n_units_with_kcat=int(has_kcat.sum()),
        n_units_with_mw=int(has_mw.sum()),
        n_units_with_kcat_and_mw=int(usable.sum()),
        n_units_in_program=int(len(e_unit)),
        n_reactions_with_gpr=int(has_gpr.sum()),
        n_reactions_enzyme_constrained=int(constrained_rxn.sum()),
        n_reactions_with_gpr_kept_gem_bound=int((has_gpr & ~constrained_rxn).sum())
        if has_pool
        else 0,
        n_variables=n_variables,
        n_equality_rows=n_eq,
        n_inequality_rows_stage1=n_ub,
        wild_type_growth_unconstrained=float(unconstrained),
        wild_type_growth_constrained=wt.growth,
        wild_type_growth_fraction=float(wt_fraction),
        wild_type_reaches_min_fraction=bool(reaches),
        wild_type_pool_usage_fraction=wt_pool_fraction,
        wild_type_pool_usage_fraction_stage2_lp=wt_pool_fraction_lp,
        wild_type_enzyme_mass_g_per_gdw=wt_mass if has_pool else None,
        wild_type_pool_dual_per_hour_per_g_gdw=wt.pool_dual if has_pool else None,
        wild_type_pool_binding=bool(abs(wt.pool_dual) > DUAL_TOL) if has_pool else None,
        n_enzyme_rows_binding_wild_type=wt.n_enzyme_rows_binding if has_pool else None,
        n_units_nonzero_usage_wild_type=wt_n_used if has_pool else None,
        n_units_nonzero_usage_wild_type_stage2_lp=int((wt.e > 1e-12).sum())
        if has_pool
        else None,
        mw_rule=(
            "MW_u [kDa = g/mmol] = sum over the unit's member genes of the SwissProt MW "
            "(yeast-GEM data/databases/swissprot.tsv joined on the systematic ORF name), one "
            "copy per subunit (the GEM records no subunit stoichiometry); defined only when "
            "every member gene has a SwissProt MW, nothing defaulted"
        ),
        kcat_rule=(
            "resolve_kcat_table (measured, homolog, predicted-median tiers; a complex takes "
            "the min over its subunits); a unit resolved to the organism default has NO kcat "
            f"here; kcat_ceiling_sigma_k {spec.sigma_k}"
        ),
        program=(
            "variables [p, n, e] >= 0, v = p - n with the GEM box split exactly as "
            "parsimonious_reference_flux splits it; S p - S n = 0; for every reaction whose "
            "units all have a kcat and an MW, p_j + n_j - sum_{u: rho(u)=j} 3600 kcat_u e_u "
            "<= 0 (both directions draw on the same enzyme; isozymes add); sum_u MW_u e_u "
            "<= P_avail; stage 1 max (p_b - n_b) [minus lambda sum(p + n) for the lambda "
            "variants], stage 2 (if any) min sum(p + n) or min sum_u MW_u e_u subject to "
            f"p_b - n_b >= {STAGE2_GROWTH_FRACTION!r} z*; the usage columns enter the matrix "
            "as capacities e'_u = 3600 kcat_u e_u and are mapped back; an enzyme row binds when its stage-1 dual exceeds "
            f"{DUAL_TOL} in magnitude; the pool dual is minus the stage-1 marginal of the "
            "pool row, in (1/h) per (g/gDW); a direction rule fixes lb >= 0 where "
            "delta_r G' <= -T and ub <= 0 where delta_r G' >= +T"
        ),
        pool_usage_rule=(
            "for stage-2 = flux: the minimum pool that carries the stage-2 flux (per "
            "enzyme-constrained reaction the surviving unit with the smallest "
            "MW_u / (3600 kcat_u) takes |v_j|), the stage-2 LP's own usage vector being "
            "degenerate; for stage-2 = enzyme_mass: the LP's own sum_u MW_u e_u; reported "
            "as a fraction of P_avail; nan where the variant has no pool. The allocation "
            "change per genotype is the number of units whose usage differs from the "
            "wild type's under the same rule and the mass moved, sum_u MW_u |e_u - e_u^wt|, "
            "as a fraction of P_avail"
        ),
    )
    path = osp.join(args.out_dir, f"ess_solve_encodings_YPD_gecko{variant}.npz")
    assert not osp.exists(path), f"{path} exists; outputs are never overwritten"
    np.savez(
        path,
        genes=np.array(genes),
        v_deletion=v_deletion,
        v_wild_type=wt.v.astype(np.float32),
        rxn_ids=np.array(rxn_ids),
        provenance=np.array(provenance.model_dump_json()),
        gecko_provenance=np.array(gecko_provenance.model_dump_json()),
        growth_ratio=ratio.astype(np.float32),
        pool_usage=pool_usage.astype(np.float32),
    )
    moved_ratio = int((np.abs(ratio - plain_ratio) > 0.01).sum())
    pool_moved = (
        np.abs(pool_usage - wt_pool_fraction) > POOL_MOVED_TOL
        if has_pool
        else np.zeros(len(genes), dtype=bool)
    )
    quantile_grid = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
    summary: dict[str, Any] = {
        "store": path,
        "shape": list(v_deletion.shape),
        "variant": variant,
        "spec": spec.model_dump(),
        "pool_name": pool.name if pool is not None else None,
        "p_avail_g_per_gdw": spec.p_avail,
        "kcat_sigma_k": spec.sigma_k,
        "kcat_tier_counts": tier_counts,
        "thermo": thermo.model_dump() if thermo is not None else None,
        "n_variables": n_variables,
        "n_equality_rows": n_eq,
        "n_inequality_rows_stage1": n_ub,
        "n_units": int(units.n_units),
        "n_units_with_kcat": int(has_kcat.sum()),
        "n_units_with_mw": int(has_mw.sum()),
        "n_units_with_kcat_and_mw": int(usable.sum()),
        "n_units_in_program": int(len(e_unit)),
        "n_reactions_with_gpr": int(has_gpr.sum()),
        "n_reactions_enzyme_constrained": int(constrained_rxn.sum()),
        "wild_type_growth_unconstrained": float(unconstrained),
        "wild_type_growth_constrained": wt.growth,
        "wild_type_growth_fraction": float(wt_fraction),
        "wild_type_reaches_min_fraction": bool(reaches),
        "wild_type_pool_usage_fraction": wt_pool_fraction,
        "wild_type_pool_usage_fraction_stage2_lp": wt_pool_fraction_lp,
        "wild_type_enzyme_mass_g_per_gdw": wt_mass if has_pool else None,
        "wild_type_pool_dual_per_hour_per_g_gdw": wt.pool_dual if has_pool else None,
        "wild_type_pool_binding": bool(abs(wt.pool_dual) > DUAL_TOL) if has_pool else None,
        "n_enzyme_rows_binding_wild_type": wt.n_enzyme_rows_binding if has_pool else None,
        "n_units_nonzero_usage_wild_type": wt_n_used if has_pool else None,
        "n_units_nonzero_usage_wild_type_stage2_lp": int((wt.e > 1e-12).sum())
        if has_pool
        else None,
        "n_genes": len(genes),
        "n_lethal": int(lethal.sum()),
        "n_lethal_plain": int((plain_growth < LETHAL_GROWTH).sum()),
        "n_genes_ratio_moved_gt_0.01": moved_ratio,
        "n_genes_log1p_vector_moved": int((rel > 1e-6).sum()),
        "median_relative_l2_log1p_to_plain": float(np.median(rel)),
        "median_relative_l2_log1p_to_plain_viable": float(np.median(rel[~lethal]))
        if (~lethal).any()
        else None,
        "n_genes_pool_usage_moved_gt_0.01": int(pool_moved.sum()) if has_pool else None,
        "n_genes_pool_usage_moved_gt_0.01_viable": int((pool_moved & ~lethal).sum())
        if has_pool
        else None,
        "n_genes_pool_binding": int((np.abs(pool_dual) > DUAL_TOL).sum())
        if has_pool
        else None,
        "n_genes_allocation_moved_gt_0.01_of_pool": int(
            (mass_moved > POOL_MOVED_TOL).sum()
        )
        if has_pool
        else None,
        "n_genes_allocation_moved_gt_0.01_of_pool_viable": int(
            ((mass_moved > POOL_MOVED_TOL) & ~lethal).sum()
        )
        if has_pool
        else None,
        "n_genes_any_unit_usage_changed_viable": int(
            ((n_units_changed > 0) & ~lethal).sum()
        )
        if has_pool
        else None,
        "allocation_mass_moved_fraction_quantiles_viable": {
            str(q): float(v)
            for q, v in zip(quantile_grid, np.quantile(mass_moved[~lethal], quantile_grid))
        }
        if has_pool and (~lethal).any()
        else None,
        "pool_usage_fraction_quantiles_viable": {
            str(q): float(v)
            for q, v in zip(quantile_grid, np.quantile(pool_usage[~lethal], quantile_grid))
        }
        if has_pool and (~lethal).any()
        else None,
        "growth_minus_plain_growth_quantiles_viable": {
            str(q): float(v)
            for q, v in zip(
                quantile_grid, np.quantile((growth - plain_growth)[~lethal], quantile_grid)
            )
        }
        if (~lethal).any()
        else None,
        "seconds": time.time() - started,
        "processes": args.processes,
        "gecko_provenance": gecko_provenance.model_dump(),
        "rows": [r.model_dump() for r in rows],
    }
    out = osp.join(args.results_dir, f"ess_gecko_store_{variant}.json")
    assert not osp.exists(out), f"{out} exists; outputs are never overwritten"
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k not in ("rows", "gecko_provenance", "thermo")
            },
            indent=2,
        )
    )
    print(f"{path}\n{out}", flush=True)


if __name__ == "__main__":
    main()
