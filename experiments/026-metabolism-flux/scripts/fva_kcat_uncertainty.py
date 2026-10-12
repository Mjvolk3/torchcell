# experiments/026-metabolism-flux/scripts/fva_kcat_uncertainty.py
# [[experiments.026-metabolism-flux.kcat-uncertainty-2026-09-14]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/fva_kcat_uncertainty.py

r"""Propagate turnover-number uncertainty through the enzyme capacity bound into FVA.

THE QUESTION. The flux layer's capacity bound :math:`|v_j|\le\sum_u k_{cat,u}E_u` is built
from ONE point value per catalytic unit, and 2,831 of those 3,728 values are a median over
sequence predictors that disagree by a median 0.387 decades. The bound is presented as a
ceiling and is in fact a point estimate. This measures what that does to the feasible set:
whether the reactions whose interval is sensitive to the parameter are a small identifiable
minority, and how much of the spread is shared across units rather than per unit.

THE DRAW, AND WHY IT IS NOT INDEPENDENT ACROSS UNITS. Predictor error is not independent:
a predictor that reads low reads low everywhere. So

.. math::
    \log_{10}k_{cat,u}=\mu_u+\sigma_u\varepsilon_u,\qquad
    \varepsilon_u=\rho z_{\text{global}}+\sqrt{1-\rho^2}\,z_u

with one :math:`z_{\text{global}}` per draw, which gives
:math:`\operatorname{corr}(\varepsilon_u,\varepsilon_{u'})=\rho^2`. Drawing independently
(:math:`\rho=0`) maximizes the chance that some single unit becomes the bottleneck and is
therefore the WIDEST case, not the neutral one. :math:`\rho` is estimated here from the data
rather than assumed, by a two-way random-effects decomposition of the admitted predictors'
per-unit values into a per-predictor offset and a per-unit residual.

:math:`\mu_u` is the resolver's own value, unchanged. :math:`\sigma_u` is
:data:`~torchcell.metabolism.parameters.KCAT_SIGMA_LADDER`, whose four rungs are each
measured and re-asserted by ``kcat_sigma_table.py``.

THE SETTINGS, and each is a real disagreement rather than a knob:

* ``rho`` in ``{0, estimated, 0.9}`` -- independent, measured, near-common-mode.
* complex aggregation in ``{min, max, median}`` over a unit's subunits. We take the MIN
  ("a complex cannot turn over faster than its slowest subunit"); GECKO 3 takes "the highest
  predicted $k_{cat}$ across its subunits". Both are defensible readings of a per-subunit
  predictor, and the single reaction that sets wild-type growth under our bound is a
  17-subunit complex whose predictors disagree by 0.993 decades, so the convention is not
  academic. The aggregation is applied to the PREDICTED tier only, since the measured and
  homolog tiers resolve per unit inside the resolver and are not rebuilt here.
* abundance in ``{1e-3, 2.7e-6}`` mmol/gDW. These are the two paths the layer already has,
  and they differ by 370x, so "the enzyme capacity bound" names two constraints two and a
  half orders of magnitude apart depending on a boolean.

THE CONTROLS, both run by ``--control``:

1. **No capacity, SBML default bounds** must reproduce ``results/fva_wildtype.csv``, which
   was computed that way, with no cassette, at growth 0.085844 /h.
2. **A sigma = 0 draw** must reproduce the point-estimate capacity run exactly, twice over,
   which is what makes a difference between draws attributable to the draw.

THE MEDIUM IS REPORTED, NEVER ASSUMED. ``fva_wildtype.csv`` is on the SBML default bounds.
The same primary setting is also run on YPD-approx through
:func:`torchcell.metabolism.media.media_to_bounds`, so the two sit side by side and no
number is quoted without the medium it was computed on.

PRE-REGISTERED READS: (i) the fraction of reactions whose FVA width varies by more than 2x
across draws, split by EC class, substrate heavy-atom bin and genes per unit; (ii) the
distribution of wild-type growth; (iii) the reactions at their capacity bound at the FBA
optimum, with the frequency of each across draws; (iv) rho.

PRE-REGISTERED KILL RULE, applied verbatim: if the growth spread at the estimated rho is
under 10 % AND fewer than 5 % of reactions move their width by 2x, predictor uncertainty
does not reach the polytope for this GEM, and the distributional program is a solution to a
non-problem. Say so and stop.

E6 RIDES ALONG (``--mode pool``). The protein pool is swept over ``p_avail`` in
``{0.125, 0.25, 0.35}`` g/gDW crossed with a saturation factor in ``{0.5, 1.0}``, in two
enforcements that are NOT the same constraint and are reported apart: a GECKO-light LP pool
``sum_j c_j(v_j^+ + v_j^-) <= sigma p_avail`` with ``c_j`` the cheapest unit's
``MW/(k_cat 3600)``, which allocates protein optimally, and the equal-molar abundance the
flux layer's own prior uses, which does not. The LP is an upper bound on the growth a pool
permits and the equal-molar path a lower one.

``PYTHONHASHSEED=0`` IS REQUIRED, AND THIS IS MEASURED, NOT SUPERSTITION. Resolving the
k_cat table twice in two processes with a free hash seed gives 2 of 3,728 units a different
value, by up to 0.313 decades: the resolver breaks a tie between equally acceptable database
rows through a container whose iteration order depends on string hashing. Within one process
it is exactly reproducible. The sweep is sharded over jobs, so an unpinned seed would compare
settings that were run on slightly different tables, and the script refuses to start without
the pin rather than producing a comparison that quietly is not one.

Run from the worktree root; one setting per slurm job::

    PYTHONHASHSEED=0 PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/fva_kcat_uncertainty.py \
        --mode primary --draws 200 --control --out fva_kcat_primary_sbml
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from cobra.flux_analysis import flux_variability_analysis
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from torchcell.metabolism.constraints import build_gem_tensors
from torchcell.metabolism.media import FBA_MEDIA, media_to_bounds
from torchcell.metabolism.parameters import (
    KCAT_SIGMA_LADDER,
    KINETICS_MIRROR,
    PLAUSIBILITY_RESULT,
    PROTEIN_POOL_SETTINGS,
    VERIFIED_PAIRS_FEATURE,
    PredictorSelection,
    SubstrateMatch,
    load_predicted_unit_table,
    molecular_weight_table,
    resolve_kcat_table,
)
from torchcell.metabolism.yeast_GEM import YeastGEM

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
OED_SCEREVISIAE = osp.join(
    DATA_ROOT, "data/enzyme_kinetics/open_enzyme_database/scerevisiae"
)
OED_ALL = osp.join(DATA_ROOT, "data/enzyme_kinetics/open_enzyme_database/all")

#: The growth floor FVA is run at, as a fraction of that draw's own optimum. 0.9 is
#: ``fva_reference.py``'s convention and is kept so the widths are comparable to it.
FRACTION_OF_OPTIMUM = 0.9

#: The nominal abundance the layer assumes when the abundance head is off, mmol/gDW.
E_NOMINAL = 1.0e-3

#: The median abundance under the budget-consistent equal-molar prior the abundance head
#: starts from, mmol/gDW, as reported in [[experiments.027-betaxanthin-metabolic.wave8-sampling]].
#: Carried as a literal so the sweep is comparable to that note; the value this run's own
#: molecular-weight table implies is recomputed and recorded beside it.
E_BUDGET = 2.7e-6

#: The committed wild-type growth of ``results/fva_wildtype.csv``, SBML default bounds, no
#: cassette. Control 1 must land on this.
SHIPPED_WILDTYPE_GROWTH = 0.085844

AGGREGATIONS = ("min", "max", "median")
ABUNDANCES = {"e_bar_1e-3": E_NOMINAL, "budget_2.7e-6": E_BUDGET}


class RhoEstimate(BaseModel):
    r"""The shared component of predictor error, from a two-way random-effects fit.

    On the balanced ``P x U`` table of per-unit :math:`\log_{10}` values,
    :math:`y_{pu}=\mu_u+a_p+e_{pu}`, the ANOVA expectations are
    :math:`E[\mathrm{MS}_{pred}]=s^2+U\tau^2` and :math:`E[\mathrm{MS}_{err}]=s^2`, so
    :math:`\tau^2` is identified and :math:`\rho=\tau/\sqrt{\tau^2+s^2}`.

    **What it is a correlation OF.** The offsets are per PREDICTOR, so this number describes
    how correlated the predicted tier's error is across units. It is not an estimate of the
    correlation of a measurement error, and the measured and homolog rungs of the ladder are
    not evidence for it. With three admitted predictors :math:`\tau^2` carries two degrees of
    freedom, so the estimate is coarse; the seven-predictor value is reported beside it.
    """

    model_config = ConfigDict(extra="forbid")

    predictors: list[str]
    n_units: int
    tau_decades: float = Field(description="sd of the per-predictor offset, decades")
    residual_sd_decades: float = Field(description="sd of the per-unit residual, decades")
    rho: float = Field(description="tau / sqrt(tau^2 + s^2); corr between units is rho^2")
    per_predictor_offset: dict[str, float]


class DrawSummary(BaseModel):
    """One Monte Carlo draw, reduced to the numbers the reads need."""

    model_config = ConfigDict(extra="forbid")

    draw: int
    growth_per_h: float
    n_at_capacity_bound: int
    median_width: float
    solver_status: str


class SettingResult(BaseModel):
    """One (rho, aggregation, abundance, medium) cell of the sweep."""

    model_config = ConfigDict(extra="forbid")

    name: str
    rho: float
    rho_source: str
    aggregation: str
    abundance_name: str
    abundance_mmol_per_gdw: float
    medium: str
    reaction_set: str
    n_reactions: int
    n_draws: int
    point_growth_per_h: float
    growth_p05: float
    growth_p50: float
    growth_p95: float
    growth_min: float
    growth_max: float
    growth_relative_spread: float = Field(
        description="(p95 - p05) / p50 of the wild-type growth across draws."
    )
    frac_width_ratio_gt_2: float
    width_ratio_p50: float
    width_ratio_p90: float
    n_narrowed_at_point: int
    point_feasible: bool = True
    n_infeasible_draws: int = 0
    min_abundance_scale_for_feasibility: float | None = Field(
        default=None,
        description=(
            "When the point estimate leaves no feasible point, the smallest uniform "
            "multiplier on the abundance that restores one. It turns an infeasibility into "
            "a number: how far short of the GEM's own maintenance demand the setting is."
        ),
    )
    wall_time_s: float
    draws: list[DrawSummary]
    top_bottleneck_reactions: dict[str, float] = Field(
        default_factory=dict,
        description="reaction id -> fraction of draws it sat at its capacity bound",
    )


class ControlResult(BaseModel):
    """The two controls, as numbers rather than as an assertion that passed."""

    model_config = ConfigDict(extra="forbid")

    shipped_growth: float
    recomputed_growth_no_capacity: float
    max_abs_diff_minimum: float
    max_abs_diff_maximum: float
    n_reactions_compared: int
    reproduces_shipped: bool
    sigma_zero_matches_point_estimate: bool
    sigma_zero_repeat_identical: bool


class PoolSetting(BaseModel):
    """One (p_avail, saturation) cell of the E6 pool sweep."""

    model_config = ConfigDict(extra="forbid")

    p_avail_g_per_gdw: float
    saturation_sigma: float
    effective_pool_g_per_gdw: float
    provenance_name: str
    growth_lp_pool_per_h: float
    growth_lp_status: str
    growth_equimolar_per_h: float
    growth_equimolar_status: str
    equimolar_e_mmol_per_gdw: float
    equimolar_min_scale_for_feasibility: float | None = Field(
        default=None,
        description=(
            "Uniform multiplier on the equal-molar abundance that restores a feasible "
            "point, when the setting itself has none."
        ),
    )
    pool_used_g_per_gdw: float
    frac_reactions_capacity_narrowed_equimolar: float
    unconstrained_growth_per_h: float


class UncertaintyReport(BaseModel):
    """Everything one job of this script produced."""

    model_config = ConfigDict(extra="forbid")

    mode: str
    built_at: str
    medium: str
    fraction_of_optimum: float
    ladder_version: str
    tier_counts: dict[str, int]
    rho_admitted: RhoEstimate | None = None
    rho_all_predictors: RhoEstimate | None = None
    settings: list[SettingResult] = Field(default_factory=list)
    controls: ControlResult | None = None
    pool: list[PoolSetting] = Field(default_factory=list)
    kill_rule: dict[str, object] = Field(default_factory=dict)
    python_hash_seed: str = ""
    notes: str = ""


# --------------------------------------------------------------------------- #
# parameter side
# --------------------------------------------------------------------------- #


def _per_unit_log10(parquet_path: str, n_units: int, aggregation: str) -> np.ndarray:
    """One predictor's log10 k_cat per unit under a stated subunit aggregation.

    Over substrates the median, always, for the resolver's own reason: a reaction with
    several substrates gets one prediction each and none of them is the reaction's value.
    Over the unit's GENES the aggregation is the thing under test.
    """
    frame = pd.read_parquet(parquet_path, columns=["unit_id", "gene_id", "k_cat"])
    frame = frame[np.isfinite(frame["k_cat"]) & (frame["k_cat"] > 0)]
    per_gene = frame.groupby(["unit_id", "gene_id"], as_index=False)["k_cat"].median()
    grouped = per_gene.groupby("unit_id")["k_cat"]
    per_unit = {"min": grouped.min, "max": grouped.max, "median": grouped.median}[
        aggregation
    ]()
    values = np.full(n_units, np.nan)
    values[per_unit.index.to_numpy().astype(int)] = np.log10(per_unit.to_numpy())
    return values


def predictor_stack(
    n_units: int, kinetics_dir: str, predictors: list[str], aggregation: str
) -> np.ndarray:
    """``[P, n_units]`` of log10 values, one row per predictor."""
    return np.vstack(
        [
            _per_unit_log10(
                osp.join(kinetics_dir, name, "verified", "k_cat.parquet"),
                n_units,
                aggregation,
            )
            for name in predictors
        ]
    )


def estimate_rho(stack: np.ndarray, predictors: list[str]) -> RhoEstimate:
    """Two-way random-effects split of the predictor stack into offset and residual."""
    complete = np.isfinite(stack).all(axis=0)
    y = stack[:, complete]
    n_pred, n_unit = y.shape
    grand = y.mean()
    pred_mean = y.mean(axis=1)
    unit_mean = y.mean(axis=0)
    ms_pred = n_unit * ((pred_mean - grand) ** 2).sum() / (n_pred - 1)
    residual = y - pred_mean[:, None] - unit_mean[None, :] + grand
    ms_err = (residual**2).sum() / ((n_pred - 1) * (n_unit - 1))
    tau_sq = max((ms_pred - ms_err) / n_unit, 0.0)
    rho = float(np.sqrt(tau_sq / (tau_sq + ms_err))) if tau_sq + ms_err > 0 else 0.0
    return RhoEstimate(
        predictors=list(predictors),
        n_units=int(n_unit),
        tau_decades=float(np.sqrt(tau_sq)),
        residual_sd_decades=float(np.sqrt(ms_err)),
        rho=rho,
        per_predictor_offset={
            name: float(pred_mean[i] - grand) for i, name in enumerate(predictors)
        },
    )


class UnitParameters(BaseModel):
    """The resolved per-unit table this script draws from, plus what maps it to reactions."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    log10_mu: np.ndarray
    log10_sigma: np.ndarray
    unit_reaction: np.ndarray
    is_predicted: np.ndarray
    genes_per_unit: np.ndarray
    unit_mw_kda: np.ndarray
    tier_counts: dict[str, int]
    predicted_log10_sd: np.ndarray


def build_unit_parameters(source, gem, aggregation: str) -> UnitParameters:
    """Resolve the shipped table, then re-aggregate the predicted tier if asked.

    The measured and homolog tiers are NOT re-aggregated. They resolve per unit inside
    ``resolve_kcat_table`` against database rows, so rebuilding them under a different
    subunit rule would be a different question from the one the aggregation setting asks,
    which is what a per-subunit sequence PREDICTOR's disagreement means for a complex.
    """
    units = gem.catalytic_units
    kinetics_dir = osp.join(DATA_ROOT, KINETICS_MIRROR)
    substrates = SubstrateMatch.from_verified_pairs(
        osp.join(kinetics_dir, VERIFIED_PAIRS_FEATURE)
    )
    plausibility = osp.join(EXPERIMENT_ROOT, PLAUSIBILITY_RESULT)
    predicted = load_predicted_unit_table(
        "k_cat", units.n_units, kinetics_dir, plausibility
    )
    table = resolve_kcat_table(
        units,
        source.model_dir,
        OED_SCEREVISIAE,
        predicted=predicted,
        substrates=substrates,
        homolog_mirror_dir=OED_ALL,
    )
    assert predicted.log10_sd is not None
    sd = predicted.log10_sd.numpy().copy()
    mu = np.log10(table.values.numpy().astype(np.float64))
    is_predicted = np.array(
        [KCAT_SIGMA_LADDER.tier_of(p) == "predicted" for p in table.provenance]
    )
    if aggregation != "min":
        admitted = PredictorSelection.from_study("k_cat", plausibility).selected
        stack = predictor_stack(units.n_units, kinetics_dir, admitted, aggregation)
        covered = np.isfinite(stack).sum(axis=0)
        alt = np.full(units.n_units, np.nan)
        alt[covered > 0] = np.nanmedian(stack[:, covered > 0], axis=0)
        alt_sd = np.full(units.n_units, np.nan)
        alt_sd[covered >= 2] = np.nanstd(stack[:, covered >= 2], axis=0, ddof=1)
        move = is_predicted & np.isfinite(alt)
        mu[move] = alt[move]
        sd[move] = alt_sd[move]

    sigma = KCAT_SIGMA_LADDER.sigma(table.provenance, sd).numpy().astype(np.float64)
    genes = np.zeros(units.n_units, dtype=np.int64)
    np.add.at(genes, units.unit_gene_index[0].numpy(), 1)
    mw = molecular_weight_table(source.model_dir, units.gene_ids).values.numpy()
    unit_mw = np.zeros(units.n_units, dtype=np.float64)
    np.add.at(
        unit_mw,
        units.unit_gene_index[0].numpy(),
        mw[units.unit_gene_index[1].numpy()].astype(np.float64),
    )
    return UnitParameters(
        log10_mu=mu,
        log10_sigma=sigma,
        unit_reaction=units.unit_reaction.numpy(),
        is_predicted=is_predicted,
        genes_per_unit=genes,
        unit_mw_kda=unit_mw,
        tier_counts=table.tier_counts(),
        predicted_log10_sd=sd,
    )


# --------------------------------------------------------------------------- #
# model side
# --------------------------------------------------------------------------- #


def capacity_from_log10(
    log10_kcat: np.ndarray, params: UnitParameters, n_reactions: int, abundance: float
) -> np.ndarray:
    """``[n_reactions]`` capacity in mmol/gDW/h; zero where no catalytic unit exists.

    Isozymes add, which is the same ``index_add`` the layer does; a reaction with no GPR
    gets no bound at all, which is the layer's more conservative choice than GECKO's
    standard-k_cat fallback.
    """
    cap = np.zeros(n_reactions, dtype=np.float64)
    np.add.at(
        cap, params.unit_reaction, np.power(10.0, log10_kcat) * 3600.0 * abundance
    )
    return cap


def apply_capacity(
    model, rxn_ids, base_lb, base_ub, cap: np.ndarray, cap_index: np.ndarray
) -> list[str]:
    """Intersect the base box with the capacity ceiling, in place.

    Returns the reactions whose capacity sits BELOW their forced lower bound. That is not a
    tight box, it is an infeasible one, and the caller reports the draw as such rather than
    widening the ceiling to rescue it.
    """
    violating: list[str] = []
    for j in cap_index:
        lo = max(base_lb[j], -cap[j])
        hi = min(base_ub[j], cap[j])
        if lo > hi:
            violating.append(rxn_ids[j])
            continue
        model.reactions.get_by_id(rxn_ids[j]).bounds = (lo, hi)
    return violating


def reset_bounds(model, rxn_ids, base_lb, base_ub, cap_index: np.ndarray) -> None:
    """Put the capacity-carrying reactions back on the base box."""
    for j in cap_index:
        model.reactions.get_by_id(rxn_ids[j]).bounds = (base_lb[j], base_ub[j])


def draw_log10(
    params: UnitParameters, rho: float, rng: np.random.Generator
) -> np.ndarray:
    """One correlated draw of log10 k_cat over units."""
    z_global = rng.standard_normal()
    z_unit = rng.standard_normal(params.log10_mu.shape[0])
    eps = rho * z_global + np.sqrt(max(1.0 - rho**2, 0.0)) * z_unit
    return params.log10_mu + params.log10_sigma * eps


def run_fva(model, reaction_ids: list[str], processes: int) -> pd.DataFrame:
    return flux_variability_analysis(
        model,
        reaction_list=reaction_ids,
        fraction_of_optimum=FRACTION_OF_OPTIMUM,
        loopless=False,
        processes=processes,
    )


def narrowed_reactions(
    params: UnitParameters, gem, fva_reference: pd.DataFrame
) -> list[str]:
    """Reactions the point-estimate capacity narrows under EITHER abundance setting.

    The union rather than one of them, so every setting of the sweep is measured on the same
    reaction list and the widths are comparable across settings.
    """
    n_reactions = len(gem.rxn_ids)
    magnitude = fva_reference[["minimum", "maximum"]].abs().max(axis=1)
    keep: set[str] = set()
    for abundance in ABUNDANCES.values():
        cap = capacity_from_log10(params.log10_mu, params, n_reactions, abundance)
        for j, rid in enumerate(gem.rxn_ids):
            if cap[j] > 0.0 and rid in magnitude.index and cap[j] < magnitude[rid]:
                keep.add(rid)
    return sorted(keep)


def reaction_strata(gem, params: UnitParameters) -> pd.DataFrame:
    """EC class, genes per unit and substrate heavy-atom bin, one row per reaction."""
    pairs = pd.read_parquet(
        osp.join(DATA_ROOT, KINETICS_MIRROR, VERIFIED_PAIRS_FEATURE),
        columns=["unit_id", "reaction_id", "smiles", "reaction_ec"],
    )
    ec_class: dict[str, str] = {}
    for rid, block in pairs.groupby("reaction_id"):
        ec = str(block["reaction_ec"].iloc[0] or "")
        ec_class[str(rid)] = ec.split(";")[0].split(".")[0] if ec else "none"

    heavy: dict[str, float] = {}
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    cache: dict[str, int] = {}
    for rid, block in pairs.groupby("reaction_id"):
        best = 0
        for smiles in block["smiles"].fillna(""):
            if not smiles:
                continue
            if smiles not in cache:
                mol = Chem.MolFromSmiles(smiles)
                cache[smiles] = int(mol.GetNumHeavyAtoms()) if mol is not None else 0
            best = max(best, cache[smiles])
        heavy[str(rid)] = float(best) if best else float("nan")

    genes = np.zeros(len(gem.rxn_ids), dtype=np.int64)
    np.maximum.at(genes, params.unit_reaction, params.genes_per_unit)
    rows = []
    for j, rid in enumerate(gem.rxn_ids):
        atoms = heavy.get(rid, float("nan"))
        if not np.isfinite(atoms):
            atom_bin = "unknown"
        elif atoms < 12:
            atom_bin = "lt12"
        elif atoms < 30:
            atom_bin = "12_29"
        elif atoms <= 60:
            atom_bin = "30_60"
        else:
            atom_bin = "gt60"
        rows.append(
            {
                "reaction_id": rid,
                "ec_class": ec_class.get(rid, "none"),
                "genes_per_unit_max": int(genes[j]),
                "heavy_atom_bin": atom_bin,
            }
        )
    return pd.DataFrame(rows).set_index("reaction_id")


def minimum_feasible_abundance_scale(
    model,
    gem,
    params: UnitParameters,
    base_lb,
    base_ub,
    cap_index: np.ndarray,
    abundance: float,
    lo: float = 1.0,
    hi: float = 1.0e6,
    iterations: int = 40,
) -> float | None:
    """Smallest uniform multiplier on the abundance under which an optimum exists.

    Bisection in log space. Returns ``None`` when even ``hi`` leaves the model infeasible,
    which would mean the infeasibility is not the abundance's doing at all.
    """
    n_reactions = len(gem.rxn_ids)

    def feasible(scale: float) -> bool:
        reset_bounds(model, gem.rxn_ids, base_lb, base_ub, cap_index)
        cap = capacity_from_log10(
            params.log10_mu, params, n_reactions, abundance * scale
        )
        if apply_capacity(model, gem.rxn_ids, base_lb, base_ub, cap, cap_index):
            return False
        return model.optimize().status == "optimal"

    if not feasible(hi):
        reset_bounds(model, gem.rxn_ids, base_lb, base_ub, cap_index)
        return None
    log_lo, log_hi = np.log10(lo), np.log10(hi)
    for _ in range(iterations):
        mid = 0.5 * (log_lo + log_hi)
        if feasible(10.0**mid):
            log_hi = mid
        else:
            log_lo = mid
    reset_bounds(model, gem.rxn_ids, base_lb, base_ub, cap_index)
    return float(10.0**log_hi)


def run_setting(
    model,
    gem,
    params: UnitParameters,
    base_lb,
    base_ub,
    *,
    name: str,
    rho: float,
    rho_source: str,
    aggregation: str,
    abundance_name: str,
    medium: str,
    reaction_ids: list[str],
    reaction_set: str,
    n_draws: int,
    seed: int,
    processes: int,
    out_prefix: str,
    strata: pd.DataFrame,
) -> SettingResult:
    """One setting: the point estimate, then ``n_draws`` correlated draws, then the reads."""
    started = time.time()
    abundance = ABUNDANCES[abundance_name]
    n_reactions = len(gem.rxn_ids)
    rng = np.random.default_rng(seed)
    cap_index = np.flatnonzero(
        np.bincount(params.unit_reaction, minlength=n_reactions) > 0
    )

    reset_bounds(model, gem.rxn_ids, base_lb, base_ub, cap_index)
    cap_point = capacity_from_log10(params.log10_mu, params, n_reactions, abundance)
    violating = apply_capacity(
        model, gem.rxn_ids, base_lb, base_ub, cap_point, cap_index
    )
    assert not violating, (
        f"the POINT estimate puts {len(violating)} reactions below their forced lower "
        f"bound ({violating[:5]}); the capacity table and the GEM disagree."
    )
    point = model.optimize()
    if point.status != "optimal":
        scale = minimum_feasible_abundance_scale(
            model, gem, params, base_lb, base_ub, cap_index, abundance
        )
        print(
            f"[{name}] the POINT estimate is {point.status}: this abundance and this "
            f"k_cat table leave the GEM no feasible point. Smallest uniform abundance "
            f"multiplier that restores one: "
            f"{'none below 1e6' if scale is None else f'{scale:.4g}'}",
            flush=True,
        )
        reset_bounds(model, gem.rxn_ids, base_lb, base_ub, cap_index)
        return SettingResult(
            name=name,
            rho=rho,
            rho_source=rho_source,
            aggregation=aggregation,
            abundance_name=abundance_name,
            abundance_mmol_per_gdw=abundance,
            medium=medium,
            reaction_set=reaction_set,
            n_reactions=len(reaction_ids),
            n_draws=0,
            point_growth_per_h=float("nan"),
            growth_p05=float("nan"),
            growth_p50=float("nan"),
            growth_p95=float("nan"),
            growth_min=float("nan"),
            growth_max=float("nan"),
            growth_relative_spread=float("nan"),
            frac_width_ratio_gt_2=float("nan"),
            width_ratio_p50=float("nan"),
            width_ratio_p90=float("nan"),
            n_narrowed_at_point=0,
            point_feasible=False,
            min_abundance_scale_for_feasibility=scale,
            wall_time_s=time.time() - started,
            draws=[],
        )
    point_growth = float(point.objective_value)
    fva_point = run_fva(model, reaction_ids, processes)
    width_point = (fva_point["maximum"] - fva_point["minimum"]).to_numpy()
    magnitude = pd.Series(
        np.abs(np.vstack([fva_point["minimum"], fva_point["maximum"]])).max(axis=0),
        index=fva_point.index,
    )
    n_narrowed = int(
        sum(
            1
            for j, rid in enumerate(gem.rxn_ids)
            if cap_point[j] > 0 and rid in magnitude.index and cap_point[j] < magnitude[rid]
        )
    )

    widths = np.empty((n_draws, len(reaction_ids)), dtype=np.float64)
    bottleneck = np.zeros(len(gem.rxn_ids), dtype=np.int64)
    draws: list[DrawSummary] = []
    for d in range(n_draws):
        log10_kcat = draw_log10(params, rho, rng)
        reset_bounds(model, gem.rxn_ids, base_lb, base_ub, cap_index)
        cap = capacity_from_log10(log10_kcat, params, n_reactions, abundance)
        violating = apply_capacity(
            model, gem.rxn_ids, base_lb, base_ub, cap, cap_index
        )
        solution = model.optimize()
        status = "capacity_below_forced_lb" if violating else str(solution.status)
        growth = float(solution.objective_value) if not violating else float("nan")
        at_bound = 0
        if not violating and solution.status == "optimal":
            fluxes = solution.fluxes.to_numpy()
            active = (cap > 0) & (np.abs(fluxes) >= 0.99 * cap)
            bottleneck += active
            at_bound = int(active.sum())
        if not violating and solution.status == "optimal":
            fva = run_fva(model, reaction_ids, processes)
            widths[d] = (fva["maximum"] - fva["minimum"]).to_numpy()
        else:
            # A draw that leaves no feasible point has no interval to report. Recording nan
            # keeps it in the denominator of every read rather than quietly dropping it.
            widths[d] = np.nan
        draws.append(
            DrawSummary(
                draw=d,
                growth_per_h=growth,
                n_at_capacity_bound=at_bound,
                median_width=float(np.median(widths[d])),
                solver_status=status,
            )
        )
        if d % 10 == 0:
            print(
                f"[{name}] draw {d}/{n_draws} growth {growth:.5f} "
                f"median width {np.median(widths[d]):.4g} "
                f"({time.time() - started:.0f}s)",
                flush=True,
            )

    growth_values = np.array([d.growth_per_h for d in draws])
    finite_growth = growth_values[np.isfinite(growth_values)]
    n_infeasible = int(n_draws - finite_growth.size)
    assert finite_growth.size > 0, (
        f"[{name}] every one of {n_draws} draws was infeasible while the point estimate "
        "was not; the draw is not a perturbation of the point, it is a different model."
    )
    with np.errstate(invalid="ignore"):
        ratio = np.nanmax(widths, axis=0) / np.maximum(np.nanmin(widths, axis=0), 1e-12)
    moved = ratio > 2.0
    p05, p50, p95 = np.percentile(finite_growth, [5, 50, 95])
    relative = float((p95 - p05) / p50) if p50 > 0 else float("inf")

    np.savez_compressed(
        osp.join(RESULTS, f"{out_prefix}_widths.npz"),
        widths=widths.astype(np.float32),
        width_point=width_point.astype(np.float32),
        reaction_ids=np.array(reaction_ids),
        growth=growth_values,
        bottleneck_counts=bottleneck,
        all_reaction_ids=np.array(gem.rxn_ids),
    )

    stratified = strata.reindex(reaction_ids).copy()
    stratified["width_ratio"] = ratio
    stratified["moved_gt_2x"] = moved
    stratified["width_point"] = width_point
    stratified.to_csv(osp.join(RESULTS, f"{out_prefix}_per_reaction.csv"))
    for column in ("ec_class", "genes_per_unit_max", "heavy_atom_bin"):
        table = stratified.groupby(column)["moved_gt_2x"].agg(["mean", "size"])
        print(f"[{name}] moved >2x by {column}:\n{table.to_string()}", flush=True)

    order = np.argsort(-bottleneck)
    top = {
        gem.rxn_ids[j]: float(bottleneck[j] / n_draws)
        for j in order[:20]
        if bottleneck[j] > 0
    }
    reset_bounds(model, gem.rxn_ids, base_lb, base_ub, cap_index)
    return SettingResult(
        name=name,
        rho=rho,
        rho_source=rho_source,
        aggregation=aggregation,
        abundance_name=abundance_name,
        abundance_mmol_per_gdw=abundance,
        medium=medium,
        reaction_set=reaction_set,
        n_reactions=len(reaction_ids),
        n_draws=n_draws,
        point_growth_per_h=point_growth,
        growth_p05=float(p05),
        growth_p50=float(p50),
        growth_p95=float(p95),
        growth_min=float(finite_growth.min()),
        growth_max=float(finite_growth.max()),
        growth_relative_spread=relative,
        frac_width_ratio_gt_2=float(moved.mean()),
        width_ratio_p50=float(np.median(ratio)),
        width_ratio_p90=float(np.percentile(ratio, 90)),
        n_narrowed_at_point=n_narrowed,
        point_feasible=True,
        n_infeasible_draws=n_infeasible,
        wall_time_s=time.time() - started,
        draws=draws,
        top_bottleneck_reactions=top,
    )


def run_controls(
    model,
    gem,
    params: UnitParameters,
    base_lb,
    base_ub,
    fva_reference: pd.DataFrame,
    processes: int,
) -> ControlResult:
    """The two controls. Numbers, not assertions, so a near-miss is visible."""
    n_reactions = len(gem.rxn_ids)
    cap_index = np.flatnonzero(
        np.bincount(params.unit_reaction, minlength=n_reactions) > 0
    )
    reset_bounds(model, gem.rxn_ids, base_lb, base_ub, cap_index)
    solution = model.optimize()
    growth = float(solution.objective_value)
    fva = run_fva(model, list(gem.rxn_ids), processes)
    shared = fva.index.intersection(fva_reference.index)
    diff_min = float(
        (fva.loc[shared, "minimum"] - fva_reference.loc[shared, "minimum"]).abs().max()
    )
    diff_max = float(
        (fva.loc[shared, "maximum"] - fva_reference.loc[shared, "maximum"]).abs().max()
    )

    zero_sigma = UnitParameters(
        log10_mu=params.log10_mu,
        log10_sigma=np.zeros_like(params.log10_sigma),
        unit_reaction=params.unit_reaction,
        is_predicted=params.is_predicted,
        genes_per_unit=params.genes_per_unit,
        unit_mw_kda=params.unit_mw_kda,
        tier_counts=params.tier_counts,
        predicted_log10_sd=params.predicted_log10_sd,
    )
    point_cap = capacity_from_log10(
        params.log10_mu, params, n_reactions, E_NOMINAL
    )
    draw_a = capacity_from_log10(
        draw_log10(zero_sigma, 0.5, np.random.default_rng(1)),
        params,
        n_reactions,
        E_NOMINAL,
    )
    draw_b = capacity_from_log10(
        draw_log10(zero_sigma, 0.9, np.random.default_rng(7)),
        params,
        n_reactions,
        E_NOMINAL,
    )
    return ControlResult(
        shipped_growth=SHIPPED_WILDTYPE_GROWTH,
        recomputed_growth_no_capacity=growth,
        max_abs_diff_minimum=diff_min,
        max_abs_diff_maximum=diff_max,
        n_reactions_compared=int(len(shared)),
        reproduces_shipped=bool(
            abs(growth - SHIPPED_WILDTYPE_GROWTH) < 1e-5
            and diff_min < 1e-5
            and diff_max < 1e-5
        ),
        sigma_zero_matches_point_estimate=bool(
            np.allclose(draw_a, point_cap, rtol=0, atol=0)
        ),
        sigma_zero_repeat_identical=bool(np.array_equal(draw_a, draw_b)),
    )


def run_pool_sweep(
    model, gem, params: UnitParameters, base_lb, base_ub, model_dir: str
) -> list[PoolSetting]:
    r"""E6: the protein pool, enforced two ways that bracket the truth.

    The LP pool prices every flux against the cheapest enzyme that can carry it,
    :math:`\sum_j c_j(v_j^++v_j^-)\le\sigma p_{avail}` with
    :math:`c_j=\min_u MW_u/(k_{cat,u}\cdot3600)`, and lets the solver allocate protein
    optimally, so its growth is the MOST a pool permits. The equal-molar path gives every
    gene the same abundance, which is the flux layer's own prior, and allocates nothing, so
    its growth is a lower bound. Quoting either alone would overstate the case.
    """
    n_reactions = len(gem.rxn_ids)
    cap_index = np.flatnonzero(
        np.bincount(params.unit_reaction, minlength=n_reactions) > 0
    )
    reset_bounds(model, gem.rxn_ids, base_lb, base_ub, cap_index)
    unconstrained = float(model.optimize().objective_value)

    cost = np.full(n_reactions, np.inf)
    per_unit_cost = params.unit_mw_kda / (
        np.power(10.0, params.log10_mu) * 3600.0
    )
    np.minimum.at(cost, params.unit_reaction, per_unit_cost)
    mw_total_kda = float(
        molecular_weight_table(model_dir, gem.catalytic_units.gene_ids).values.sum()
    )

    out: list[PoolSetting] = []
    for p_avail, record in sorted(PROTEIN_POOL_SETTINGS.items()):
        for saturation in (0.5, 1.0):
            pool = p_avail * saturation
            reset_bounds(model, gem.rxn_ids, base_lb, base_ub, cap_index)
            terms = {}
            for j, rid in enumerate(gem.rxn_ids):
                if not np.isfinite(cost[j]) or cost[j] <= 0:
                    continue
                reaction = model.reactions.get_by_id(rid)
                terms[reaction.forward_variable] = cost[j]
                terms[reaction.reverse_variable] = cost[j]
            constraint = model.problem.Constraint(0, ub=pool, name="tc_protein_pool")
            model.add_cons_vars([constraint])
            model.solver.update()
            constraint.set_linear_coefficients(terms)
            solution = model.optimize()
            growth_lp = float(solution.objective_value)
            used = (
                float(sum(coef * var.primal for var, coef in terms.items()))
                if solution.status == "optimal"
                else float("nan")
            )
            model.remove_cons_vars([constraint])
            model.solver.update()

            e_equimolar = pool / mw_total_kda
            reset_bounds(model, gem.rxn_ids, base_lb, base_ub, cap_index)
            cap = capacity_from_log10(
                params.log10_mu, params, n_reactions, e_equimolar
            )
            violating = apply_capacity(
                model, gem.rxn_ids, base_lb, base_ub, cap, cap_index
            )
            assert not violating, (
                f"equal-molar E puts {len(violating)} reactions below a forced lower bound"
            )
            equimolar = model.optimize()
            growth_eq = (
                float(equimolar.objective_value)
                if equimolar.status == "optimal"
                else float("nan")
            )
            equimolar_scale = (
                None
                if equimolar.status == "optimal"
                else minimum_feasible_abundance_scale(
                    model,
                    gem,
                    params,
                    base_lb,
                    base_ub,
                    cap_index,
                    e_equimolar,
                )
            )
            has_cap = cap > 0
            narrowed = float(
                np.mean(
                    cap[has_cap]
                    < np.array(
                        [
                            max(abs(base_lb[j]), abs(base_ub[j]))
                            for j in np.flatnonzero(has_cap)
                        ]
                    )
                )
            )
            reset_bounds(model, gem.rxn_ids, base_lb, base_ub, cap_index)
            out.append(
                PoolSetting(
                    p_avail_g_per_gdw=p_avail,
                    saturation_sigma=saturation,
                    effective_pool_g_per_gdw=pool,
                    provenance_name=record.name,
                    growth_lp_pool_per_h=growth_lp,
                    growth_lp_status=str(solution.status),
                    growth_equimolar_per_h=growth_eq,
                    growth_equimolar_status=str(equimolar.status),
                    equimolar_e_mmol_per_gdw=e_equimolar,
                    equimolar_min_scale_for_feasibility=equimolar_scale,
                    pool_used_g_per_gdw=used,
                    frac_reactions_capacity_narrowed_equimolar=narrowed,
                    unconstrained_growth_per_h=unconstrained,
                )
            )
            print(
                f"[pool] p_avail {p_avail} sigma {saturation} -> LP growth {growth_lp:.5f} "
                f"({solution.status}), equimolar growth {growth_eq:.5f} "
                f"({equimolar.status}, min feasible scale "
                f"{'-' if equimolar_scale is None else f'{equimolar_scale:.3g}'}) "
                f"at E={e_equimolar:.3g}, "
                f"pool used {used:.4f} g/gDW (unconstrained {unconstrained:.5f})",
                flush=True,
            )
    return out


def sweep_settings(rho_estimated: float) -> list[dict]:
    """The 18 cells, in a fixed order so a shard index names the same cell every run."""
    cells = []
    for rho_name, rho in (
        ("rho0", 0.0),
        ("rhoest", rho_estimated),
        ("rho09", 0.9),
    ):
        for aggregation in AGGREGATIONS:
            for abundance in ABUNDANCES:
                cells.append(
                    {
                        "name": f"{rho_name}_{aggregation}_{abundance}",
                        "rho": rho,
                        "rho_source": (
                            "estimated" if rho_name == "rhoest" else "fixed"
                        ),
                        "aggregation": aggregation,
                        "abundance_name": abundance,
                    }
                )
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("primary", "sweep", "pool"), required=True)
    parser.add_argument("--medium", choices=("sbml_default", "YPD"), default="sbml_default")
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2700)
    parser.add_argument("--control", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    assert os.environ.get("PYTHONHASHSEED") == "0", (
        "set PYTHONHASHSEED=0. The k_cat resolver breaks ties through a hash-ordered "
        "container, so two processes with a free seed resolve 2 of 3,728 units differently "
        "by up to 0.313 decades, and the shards of this sweep would not be comparable."
    )
    os.makedirs(RESULTS, exist_ok=True)
    started = time.time()
    source = YeastGEM()
    model = source.model
    gem = build_gem_tensors(
        source.model, model_dir=source.model_dir, with_independent_rows=False
    )
    print(f"GEM loaded, {len(gem.rxn_ids)} reactions ({time.time()-started:.0f}s)", flush=True)

    if args.medium == "YPD":
        media_to_bounds(FBA_MEDIA["YPD"], model).apply(model)
    base_lb = np.array([r.lower_bound for r in model.reactions], dtype=np.float64)
    base_ub = np.array([r.upper_bound for r in model.reactions], dtype=np.float64)

    params = build_unit_parameters(source, gem, "min")
    print(f"tier counts {params.tier_counts}", flush=True)

    kinetics_dir = osp.join(DATA_ROOT, KINETICS_MIRROR)
    plausibility = osp.join(EXPERIMENT_ROOT, PLAUSIBILITY_RESULT)
    admitted = PredictorSelection.from_study("k_cat", plausibility).selected
    rho_admitted = estimate_rho(
        predictor_stack(
            gem.catalytic_units.n_units, kinetics_dir, admitted, "min"
        ),
        admitted,
    )
    everything = [
        name
        for name in (
            "ezkinetics",
            "eitlem",
            "unikp",
            "catpred",
            "dlkcat",
            "deepenzyme",
            "turnup",
        )
        if osp.exists(osp.join(kinetics_dir, name, "verified", "k_cat.parquet"))
    ]
    rho_all = estimate_rho(
        predictor_stack(gem.catalytic_units.n_units, kinetics_dir, everything, "min"),
        everything,
    )
    print(
        f"[rho] admitted {admitted}: tau {rho_admitted.tau_decades:.4f}, "
        f"residual {rho_admitted.residual_sd_decades:.4f}, rho {rho_admitted.rho:.4f} "
        f"(corr between units rho^2 = {rho_admitted.rho ** 2:.4f}) on "
        f"{rho_admitted.n_units} units",
        flush=True,
    )
    print(
        f"[rho] all {everything}: tau {rho_all.tau_decades:.4f}, "
        f"residual {rho_all.residual_sd_decades:.4f}, rho {rho_all.rho:.4f}",
        flush=True,
    )

    report = UncertaintyReport(
        mode=args.mode,
        built_at=datetime.now(UTC).isoformat(),
        medium=args.medium,
        fraction_of_optimum=FRACTION_OF_OPTIMUM,
        ladder_version=KCAT_SIGMA_LADDER.version,
        tier_counts=params.tier_counts,
        rho_admitted=rho_admitted,
        rho_all_predictors=rho_all,
        python_hash_seed=os.environ.get("PYTHONHASHSEED", ""),
    )

    if args.mode == "pool":
        report.pool = run_pool_sweep(
            model, gem, params, base_lb, base_ub, source.model_dir
        )
        report.notes = (
            "E6. The LP pool allocates protein optimally and the equal-molar path does "
            "not, so the two bracket the growth a pool permits. Neither is compared to a "
            "measured growth rate on a different medium."
        )
        path = osp.join(RESULTS, f"{args.out}.json")
        with open(path, "w") as handle:
            handle.write(report.model_dump_json(indent=2))
        print(f"wrote {path}", flush=True)
        return

    fva_reference = pd.read_csv(osp.join(RESULTS, "fva_wildtype.csv"), index_col=0)
    if args.control:
        assert args.medium == "sbml_default", (
            "control 1 compares against fva_wildtype.csv, which was computed on the SBML "
            "default bounds; running it on another medium would compare two media."
        )
        report.controls = run_controls(
            model, gem, params, base_lb, base_ub, fva_reference, args.processes
        )
        print(f"[control] {report.controls.model_dump_json(indent=2)}", flush=True)

    strata = reaction_strata(gem, params)
    if args.mode == "primary":
        cells = [
            {
                "name": f"rhoest_min_e_bar_1e-3_{args.medium}",
                "rho": rho_admitted.rho,
                "rho_source": "estimated",
                "aggregation": "min",
                "abundance_name": "e_bar_1e-3",
            }
        ]
        reaction_ids = list(gem.rxn_ids)
        reaction_set = "all"
    else:
        primary_name = "rhoest_min_e_bar_1e-3"
        everything_but_primary = [
            cell
            for cell in sweep_settings(rho_admitted.rho)
            if cell["name"] != primary_name
        ]
        cells = [
            cell
            for i, cell in enumerate(everything_but_primary)
            if i % args.n_shards == args.shard
        ]
        reaction_ids = narrowed_reactions(params, gem, fva_reference)
        reaction_set = "capacity_narrowed_union"
        print(
            f"[sweep] shard {args.shard}/{args.n_shards} runs "
            f"{[c['name'] for c in cells]} on {len(reaction_ids)} reactions",
            flush=True,
        )

    for index, cell in enumerate(cells):
        cell_params = (
            params
            if cell["aggregation"] == "min"
            else build_unit_parameters(source, gem, cell["aggregation"])
        )
        result = run_setting(
            model,
            gem,
            cell_params,
            base_lb,
            base_ub,
            name=cell["name"],
            rho=float(cell["rho"]),
            rho_source=str(cell["rho_source"]),
            aggregation=str(cell["aggregation"]),
            abundance_name=str(cell["abundance_name"]),
            medium=args.medium,
            reaction_ids=reaction_ids,
            reaction_set=reaction_set,
            n_draws=args.draws,
            seed=args.seed + index,
            processes=args.processes,
            out_prefix=f"{args.out}_{cell['name']}",
            strata=strata,
        )
        report.settings.append(result)
        print(
            f"[{result.name}] growth p05/p50/p95 {result.growth_p05:.5f}/"
            f"{result.growth_p50:.5f}/{result.growth_p95:.5f} "
            f"relative spread {result.growth_relative_spread:.4f}; "
            f"width ratio p50 {result.width_ratio_p50:.3f}, "
            f"moved >2x {result.frac_width_ratio_gt_2:.4f}",
            flush=True,
        )

    if args.mode == "primary" and report.settings:
        primary = report.settings[0]
        spread_small = primary.growth_relative_spread < 0.10
        few_moved = primary.frac_width_ratio_gt_2 < 0.05
        report.kill_rule = {
            "rule": (
                "kill if the growth spread at the estimated rho is under 10 percent AND "
                "fewer than 5 percent of reactions move their width by 2x"
            ),
            "growth_relative_spread": primary.growth_relative_spread,
            "frac_width_ratio_gt_2": primary.frac_width_ratio_gt_2,
            "growth_spread_under_10pct": spread_small,
            "moved_under_5pct": few_moved,
            "killed": bool(spread_small and few_moved),
        }
        print(f"[kill rule] {json.dumps(report.kill_rule, indent=2)}", flush=True)

    path = osp.join(RESULTS, f"{args.out}.json")
    with open(path, "w") as handle:
        handle.write(report.model_dump_json(indent=2))
    print(f"wrote {path} in {time.time()-started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
