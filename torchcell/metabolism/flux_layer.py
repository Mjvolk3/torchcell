# torchcell/metabolism/flux_layer
# [[torchcell.metabolism.flux_layer]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/metabolism/flux_layer
# Test file: tests/torchcell/metabolism/test_flux_layer.py

r"""A differentiable, enzyme-constrained, thermodynamically-feasible flux layer.

WHAT THIS REPLACES
------------------
Constraint-based modeling answers "which flux vectors are possible" with a linear (or,
once thermodynamics is added, a mixed-integer) program, solved once per genotype. Smith
et al. (2026), *Thermo-Flux*, is the current statement of that method: the second law
enters as :math:`\Delta_r G_j \nu_j < 0`, which is **bilinear**, and is implemented with
one **binary indicator per reaction** under a big-M pair, giving a MILP whose cost the
authors describe as scaling with "the number of possible direction combinations ...
exponentially with the number of reactions", budgeted at 24 h wall time per model on an
HPC cluster.

Two consequences follow, and together they are the reason this module exists.

**Per-genotype cost does not amortize.** Merzbacher et al. (2025) sample one flux cone per
single deletion by MCMC: 1,159 yeast deletions, 124 samples each, 4.43 GB. The pairwise
extension is :math:`\binom{1161}{2}=673{,}380` independent random walks, roughly 2.6 TB.
An amortized model pays the cost once at training time and answers a double deletion in
the same forward pass as a single.

**A program cannot learn.** An LP has no use for a phenotype measurement. This layer is
supervised by ~10^7 fitness records, 4,735 betaxanthin measurements and 4,678 amino-acid
profiles, and its flux is *conditioned* on them.

THE RELAXATION, TERM BY TERM
----------------------------
Every hard constraint is either enforced exactly by the parameterization or relaxed to a
smooth penalty. Nothing is enforced by a binary variable.

======================  ==========================================  =====================
constraint              Thermo-Flux / GECKO form                     here
======================  ==========================================  =====================
box + directionality    :math:`lb_j\le\nu_j\le ub_j`                 **exact**, sigmoid
capacity (enzyme)       :math:`\nu_j\le k_{cat}E`                    **exact**, in the box
mass balance            :math:`S\nu=0`                               soft, scale-relative
second law              :math:`\Delta_rG_j\nu_j<0` + binary          soft hinge, no binary
Gibbs dissipation       :math:`g_{diss}\le g_{lim}`                  soft hinge
protein budget          :math:`\sum MW_g E_g\le P_{avail}`           soft hinge
======================  ==========================================  =====================

The second-law relaxation is the load-bearing one:

.. math::
    C_{\mathrm{th}} = \frac{1}{|\mathcal{J}|}\sum_{j\in\mathcal{J}}
        \operatorname{relu}\!\big(\nu_j\,\Delta_r G_j + \epsilon\big),

with :math:`\mathcal{J}` excluding exchange reactions, the biomass reaction, and water
transport -- the exemptions Thermo-Flux's Box 2 specifies, because those processes cross
the system boundary or are lumped and so carry no meaningful reaction energy. The
:math:`\epsilon` is not decoration: the paper's own big-M pair admits
:math:`\nu_j=\Delta_r G_j=0` under either value of the binary, so the strict inequality of
its Eq. (10) is not actually realized, and a smooth version has to name its tolerance.

**Loop-freedom is free, and that is the whole trick.** Because :math:`\Delta_r G` is a
difference of potentials it sums to zero around any cycle, so no cycle can run downhill
everywhere; requiring :math:`\nu_j\Delta_j\le0` therefore forbids internal loops without a
single integer variable. That is exactly the property that makes loopless FBA a MILP.

SPENDING THE EXACTNESS BUDGET: BOX, NULL SPACE, PROJECTION
----------------------------------------------------------
``parameterization`` decides which constraint holds by construction.

``box``
    :math:`v=lb+(ub-lb)\,\sigma(z)`. Box, directionality and enzyme capacity are exact;
    :math:`Sv=0` is a scale-relative penalty.

``nullspace``
    :math:`v=\mathcal Nz` with :math:`S\mathcal N=0`. Mass balance is exact; the box is a
    penalty.

``projection``
    Both, alternately. Start at :math:`v_0=\mathcal Nz` and alternate the two projections
    :math:`P_{\mathrm{bal}}(v)=\mathcal N(\mathcal N^\top v)` (exact because
    :math:`\mathcal N` is orthonormal, so :math:`\mathcal N\mathcal N^\top` is the
    orthogonal projector onto :math:`\ker S`) and
    :math:`P_{\mathrm{box}}(v)=\mathrm{clip}(v,lb,ub)`. This is von Neumann's alternating
    projection between a subspace and a box, both convex, so the iterates converge to a
    point of the intersection; ``projection_method="dykstra"`` runs Dykstra's correction
    instead, which converges to the NEAREST such point to :math:`v_0`.

    **What "exact" means here is settled by which projection runs last, and that is a
    config choice, not a detail.** ``projection_end="balance"`` ends on
    :math:`P_{\mathrm{bal}}`, so :math:`Sv=0` holds to machine precision and the residual
    box violation is what the layer reports; ``projection_end="box"`` ends on the clip, so
    the box holds to machine precision and the residual is :math:`\lVert Sv\rVert`. The
    trade is real and neither end is free: :math:`K` finite steps do not reach the
    intersection, they only shrink whichever residual is left over. ``feasibility`` returns
    both per sample so the caller never has to infer which one is the loose end.

    Default ``balance``, because the Genesis filter that motivates the head asks first
    whether a nominated strain's flux is stoichiometrically consistent, and a residual box
    violation is a bound that is slightly overrun rather than a flux that comes from
    nowhere.

``fluxforce``
    The second law as a **smooth coupling** rather than as a sign clamp. For every priced,
    non-exempt reaction the direction and a magnitude factor come from the reaction energy
    through the flux-force relation (Beard and Qian; the flux-force efficacy of Noor et al.
    2014),

    .. math::
        v_j = m_j\,f(\Delta_rG_j),\qquad
        f(x) = \frac{1-e^{x/RT}}{1+e^{x/RT}} = -\tanh\!\Big(\frac{x}{2RT}\Big),\qquad
        m_j = \max(|v^\ell_j|,|v^u_j|)\,\sigma(z_j),

    so :math:`\operatorname{sign}v_j=-\operatorname{sign}\Delta_rG_j` by construction,
    :math:`|v_j|\le m_j`, a near-equilibrium reaction goes quiet instead of being pinned,
    and :math:`\partial v/\partial\ln c` is nonzero everywhere. Unpriced and exempt
    reactions keep the sigmoid box. The flux is then clipped to the genotype box and run
    through the same alternating projection, so mass balance is exact at
    ``projection_end='balance'`` and the bounds are exact at ``'box'``.

    **Why it exists, and it is a measured contrast.** ``thermo_box`` decides the sign with
    ``delta_g > eps`` inside a ``torch.where``, which is a step: no gradient reaches the
    learned concentrations through the box, so the model cannot move a pool to license a
    direction along that path, and the only thermodynamic gradient it has left is the soft
    hinge. In wave 6 every projection cell without the sign tightening sustained its escape
    from the betaxanthin mean plateau (6 of 6) and none with it did (0 of 6). The projection
    can still move a reaction across zero after the fact, which is the one second-law
    violation this channel produces, and ``feas_sign_flip_frac`` counts it.

``warmstart``
    A **residual** head. The proposal is a feasible reference flux plus a small step inside
    :math:`\ker S`,

    .. math::
        v_{\mathrm{pre}} = \mathrm{clip}(v_{\mathrm{ref}},v^\ell,v^u)
            + \alpha\,\lVert \mathrm{clip}(v_{\mathrm{ref}})\rVert\,
              \frac{\mathcal N z}{\lVert\mathcal N z\rVert},

    so the relative distance from the reference is exactly :math:`\alpha`, one learned
    scalar, and the direction is whatever the genotype asks for. The same clip and the same
    alternating projection follow, so the head differs from ``projection`` and ``fluxforce``
    in **where the alternation starts** and in nothing else.

    **Why, and it is measured rather than argued.** The alternation cannot repair a proposal
    that is an O(1) distance from the polytope at any K: from a uniform random point of the
    box, K from 20 to 5,000, von Neumann and Dykstra alike, the turnover-thresholded balance
    ratio stays at 2.0, the worst value the statistic takes
    (``experiments/026-metabolism-flux/results/probe_projrate.json``). From a proposal within
    1 percent of a feasible flux the EXISTING K = 100 alternation reaches 0.0012 on that same
    statistic (``probe_warmstart.json``). And the trained ``fluxforce_k100_boxforce`` head of
    job 1784 proposes at a relative distance of 0.990, i.e. the projection discards 99 percent
    of what the head emits
    (``probe_trained_proposal_distance_wave8_dens_d2p0_boxforce_k100_e80_s2700.json``, 32
    genotypes). The proposal, not the projection, is what fails.

    :math:`v_{\mathrm{ref}}` is a pFBA solution computed **inside the build step from the
    GEM's own tensors** by :func:`parsimonious_reference_flux`, never supplied by the caller.
    That is what keeps the head organism-independent: parsimonious flux balance is a property
    of a stoichiometric model, so a new GEM brings its own reference and no yeast constant
    enters. The reference is clipped into each genotype's ``dynamic_box``, so a deletion
    still zeroes its reactions exactly.

``latent_stochastic`` turns the last two into an amortized sampler: :math:`z=\mu+\sigma
\epsilon` with a :math:`\mathrm{KL}(\mathcal N(\mu,\sigma)\Vert\mathcal N(0,I))` term
returned as ``c_kl``, and :meth:`FluxLayer.sample` draws :math:`S` fluxes per genotype and
reports each draw's own feasibility. Every draw is projected, so the sampler's support is
the feasible set rather than a Gaussian around it: the 026 stochastic head, which drew
:math:`z` per reaction and projected nothing, produced draws with a median balance ratio of
1.98 and 30 percent thermodynamic violations
(``experiments/026-metabolism-flux/results/flux_sampling_demo.json``).

THERMODYNAMICS: FREE VS ANCHORED
--------------------------------
Following Thermo-Flux Eq. (4), the in-cell reaction energy decomposes into three terms,

.. math::
    \Delta_r G_j = \underbrace{\Delta_r G'^{\circ}_j}_{\text{tabulated}}
                 + \underbrace{RT\sum_{i\neq H^+} S_{ij}\ln c_i}_{\text{concentration}}
                 + \underbrace{\Delta_r G^{t}_j}_{\text{transport}},

and this module implements all three in ``ThermoMode.ANCHORED``:

* :math:`\Delta_r G'^{\circ}_j=\sum_i S_{ij}\Delta_f G'^{\circ}_i` from the GEM's shipped
  table (85.1 % of metabolites, 77.7 % of reactions on yeast-GEM 9.0.2, once BOTH missing
  value conventions are rejected -- see :func:`~torchcell.metabolism.constraints._read_delta_g_csv`);
* :math:`\ln c_i` **learned**, squashed into the physiological window
  :math:`[0.1\,\mu\mathrm{M},10\,\mathrm{mM}]` that Thermo-Flux uses by default, and
  conditioned on the genotype so a deletion can move a pool;
* :math:`\Delta_r G^{t}_j = -N_H RT\ln(10)\,\Delta\mathrm{pH} - Fq\,\Delta\Phi` from the
  compartment table, computed once and held as a buffer.

``ThermoMode.FREE`` drops the table and learns an unconstrained potential
:math:`\mu_i`. It needs no data and still kills loops, but :math:`\mu` is identified only
up to an affine map, so it is not an energy and cannot be checked against anything.
**Anchoring is what turns the thermodynamic term from a regularizer into a measurement**,
and the ablation between the two modes is the experiment that says whether the tabulated
energies carry information.

UNCERTAINTY, AND WHY IT IS A LATENT
-----------------------------------
Thermo-Flux carries correlated uncertainty as :math:`\Delta_rG^{\circ,\mathrm{error}}=Qm`
with :math:`m\sim\mathcal N(0,1)` and :math:`Q` the square-root covariance from
eQuilibrator, and its Box 3 *infers* :math:`m` by regression against measured
concentrations and fluxes. That inference is exactly what a latent variable does, and both
forms of it are implemented here, selected by ``thermo_uncertainty``.

``isotropic`` (**the default, and what every 026 and 027 run to date used**)
    :math:`m` is a learned **per-reaction** offset of dimension :math:`r`, added straight
    onto :math:`\Delta_rG_j`, with a :math:`\lVert m\rVert^2` prior. Every reaction gets its
    own free direction and the correlations between them are dropped.

``covariance``
    :math:`m` is a learned latent of dimension :math:`k` and the offset is
    :math:`Qm`, so the reactions can only move together in the directions the component
    contribution actually leaves uncertain. :math:`Q` is supplied by the caller as
    ``delta_g_sqrt_cov``; :func:`reaction_sqrt_covariance` builds it.

:math:`Q` **is available**, contrary to what this docstring said before 2026-09-10. Our
eQuilibrator recomputation writes ``formation_sigma.npz`` into
``$DATA_ROOT/data/torchcell/thermo_equilibrator``, holding the metabolite square-root
covariance :math:`\Sigma^{1/2}` of the component contribution as a dense
``2099 x 590`` factor over the species it could reach. The reaction-level factor is
:math:`Q=S^\top\Sigma^{1/2}`, a ``4131 x 590`` matrix, and
:math:`\sigma_j=\lVert Q_j\rVert_2` is exactly the per-reaction number
:class:`~torchcell.metabolism.parameters.ReactionThermoTable` already carries in ``sigma``
(``resolve_reaction_delta_g`` forms the same product and keeps only its row norms).
On yeast-GEM 9.0.2 with the union route, 1,755 of 4,131 reactions have EVERY participant
inside the factor; the rest carry no covariance and therefore no offset at all under the
``covariance`` path, because a reaction whose error model is unknown should not be given a
free parameter that looks like one.

Metabolites with no tabulated formation energy follow Thermo-Flux's own convention:
:math:`\Delta_f G^\circ = 0` with an uncertainty of :math:`\pm 3000` kJ/mol, which is
wide enough that any reaction touching one can take either sign. Implemented here as a
mask rather than as a number, which has the same effect and cannot be mistaken for data.

ORGANISM PORTABILITY
--------------------
Everything below is a function of ``GemTensors`` plus a small
:class:`CompartmentParameters` table and two scalars (:math:`P_{\mathrm{avail}}`,
:math:`g_{\mathrm{lim}}`). There is no yeast constant anywhere in this file. Porting to
another yeast or to a bacterium means supplying a different GEM, a compartment table, and
kinetic parameters -- which is the point, because the kinetic parameters for a new
organism come from a sequence-based predictor rather than from a literature curation
effort.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field

from torchcell.metabolism.constraints import (
    CONC_LOWER_M,
    CONC_UPPER_M,
    GemTensors,
    ThermoMode,
    null_space_basis,
)
from torchcell.metabolism.parameters import (
    EXPERIMENTAL_SOURCES,
    ParameterProvenance,
    ReactionThermoTable,
    ThermoSource,
)

#: Faraday constant, kJ/(mol V). Enters the transport term as F q dPhi.
FARADAY_KJ_PER_MOL_V = 96.485

#: Working flux magnitude, mmol/gDW/h. Genome-scale models use +/-1000 as "unbounded"
#: rather than as a physical claim; a sigmoid stretched over 2,000 units puts every real
#: flux inside a vanishing fraction of its range and saturates the gradient. Clamping the
#: box to a physiological magnitude is a modeling choice and is reported as one.
DEFAULT_FLUX_SCALE = 100.0


def _linear_quantile(x: torch.Tensor, q: float) -> torch.Tensor:
    r"""``torch.quantile(x, q)`` without its input-size cap, by sorting.

    ``torch.quantile`` refuses an input above 2**24 elements with
    ``RuntimeError: quantile() input tensor is too large``. That ceiling is reachable here:
    the active-row balance percentile is taken over ``[B, independent rows]``, which at
    batch 128 and 2,593 rows is 332k elements at one draw per genotype and 42.5M at 128 of
    them. The cap was hit by the wave-8 K = 128 benchmark before any job was submitted.

    This reproduces the DEFAULT 'linear' interpolation exactly, so every number an existing
    arm reported is unchanged: the quantile sits at position :math:`q(n-1)` of the sorted
    values and interpolates between the two neighbours it falls between.
    """
    flat = x.reshape(-1)
    n = flat.numel()
    assert n > 0, "the quantile of an empty tensor is not defined."
    ordered, _ = torch.sort(flat)
    position = q * (n - 1)
    lower = int(position)
    upper = min(lower + 1, n - 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _tier_counts(provenance: list[str] | None, n: int) -> dict[str, int]:
    """Split a provenance list into measured / predicted / default counts.

    ``None`` means the caller supplied no provenance, which is reported as ``unknown`` for
    the whole table rather than silently attributed to any tier.
    """
    if provenance is None:
        return {"measured": 0, "predicted": 0, "default": 0, "unknown": n}
    if len(provenance) != n:
        raise ValueError(
            f"provenance has {len(provenance)} entries against {n} catalytic units."
        )
    counts = {"measured": 0, "predicted": 0, "default": 0, "unknown": 0}
    for tag in provenance:
        if tag in {str(p) for p in EXPERIMENTAL_SOURCES}:
            counts["measured"] += 1
        elif tag == str(ParameterProvenance.ORGANISM_DEFAULT):
            counts["default"] += 1
        else:
            counts["predicted"] += 1
    return counts


class CompartmentParameters(BaseModel):
    """pH, ionic strength and membrane potential per compartment.

    Values for *S. cerevisiae* iMM904 as tabulated by Thermo-Flux (Smith et al. 2026,
    Table 1): cytosol pH 7.0, mitochondrion pH 7.4, extracellular pH 5.0; ionic strength
    0.25 M in all compartments; T = 298 K; membrane potential 60 mV across the plasma
    membrane and 160 mV across the mitochondrial membrane. They are defaults, not
    constants: another organism supplies its own, which is the entire organism-specific
    surface of the thermodynamic layer.
    """

    ph: dict[str, float] = Field(default_factory=lambda: {"c": 7.0, "m": 7.4, "e": 5.0})
    ionic_strength_m: dict[str, float] = Field(default_factory=dict)
    membrane_potential_mv: dict[str, float] = Field(
        default_factory=lambda: {"e": 60.0, "m": 160.0}
    )
    reference_compartment: str = "c"
    temperature_k: float = 303.15
    source: str = (
        "Smith et al. 2026, Thermo-Flux, Table 1 (iMM904 parameter set); "
        "membrane potentials in mV, pH dimensionless."
    )


class FluxLayerConfig(BaseModel):
    """Every switch the flux layer has, in one place so a run records what it ran.

    Kept as a pydantic model rather than kwargs so a config is serializable into the
    experiment's results JSON and an ablation arm is a diff between two of these.
    """

    model_config = ConfigDict(extra="forbid")

    hidden_dim: int = 32
    reaction_embed_dim: int = 16
    parameterization: str = Field(
        default="box",
        description=(
            "'box' -- v = lb + (ub-lb)*sigmoid(z); bounds exact, Sv=0 soft. "
            "'nullspace' -- v = N z with S N = 0; Sv=0 exact, bounds soft. "
            "'projection' -- alternate the two projections K times from v0 = N z, so both "
            "constraints hold to whatever K buys and the one named by `projection_end` "
            "holds exactly. "
            "'fluxforce' -- the priced reactions take v = m f(dG) with f the flux-force "
            "relation and m a nonnegative magnitude read off the genotype box, then the "
            "same clip and projection; the second law is smooth and differentiable in the "
            "concentrations rather than a step. "
            "'projection_k<N>' names the step count in the string itself, which is the "
            "only channel 026's build_flux_layer leaves open to an arm (it forwards this "
            "field and no other new one); 'projection' alone takes `projection_steps`."
        ),
    )
    projection_steps: int = Field(
        default=5,
        description=(
            "K, the number of box clips in the alternating projection. Each clip is "
            "followed by the balance projection except, under projection_end='box', the "
            "last. K=1 is one clip; the off-end residual falls with K and the cost is 2K "
            "matmuls of [B, r] against the [r, nullity] basis per forward."
        ),
    )
    projection_end: str = Field(
        default="balance",
        description=(
            "Which projection runs last, and therefore which constraint is EXACT rather "
            "than merely small. 'balance' -- S v = 0 to machine precision, box violation "
            "reported. 'box' -- lb <= v <= ub to machine precision, balance residual "
            "reported. Read as a claim about the output, not a tuning knob."
        ),
    )
    projection_method: str = Field(
        default="alternating",
        description=(
            "'alternating' -- von Neumann, a point OF the intersection. 'dykstra' -- the "
            "NEAREST point of the intersection to v0, at one extra pair of correction "
            "tensors. Both converge; only Dykstra's limit is the metric projection."
        ),
    )
    thermo_exempt_path: str | None = Field(
        default=None,
        description=(
            "JSON naming reactions to EXEMPT from the second law on top of Thermo-Flux's "
            "Box 2 set, written by "
            "`experiments/026-metabolism-flux/scripts/thermo_exempt_gate.py` and read by "
            "`read_thermo_exempt_list`. None is off and is what every run to date did.\n\n"
            "WHY A REACTION MAY BE EXEMPTED, and it is the same argument "
            "`thermo_box_margin_kj` already makes. A tabulated standard energy that no "
            "concentration in the physiological window can reconcile with a real flux is "
            "not a measurement of a direction; pricing it asserts one the data does not "
            "have. MEASURED (results/thermo_exempt_candidates.json, union_external, window "
            "1e-7 to 1e-2 M): with the flux fixed at a cobra FBA optimum the total L1 slack "
            "on the sign constraints is 2,641.5 kJ over 425 reactions, and removing the 47 "
            "the LP puts its violation on leaves 0.000 kJ over the remaining 378. The "
            "infeasibility is concentrated in those 47.\n\n"
            "PRECEDENT: yETFL applies no thermodynamic constraint at all to its 1,304 "
            "membrane reactions, and its thermodynamics did not change its essentiality "
            "predictions. Exempting 47 is more conservative than the published practice.\n\n"
            "The file's own route, window and table sha256 travel into `coverage_report` "
            "with the file's sha256, because an exemption list is a claim about a specific "
            "thermodynamic table and is meaningless beside a different one."
        ),
    )
    log_c_init_path: str | None = Field(
        default=None,
        description=(
            "JSON holding a log-concentration vector to INITIALIZE the anchored "
            "concentration at, written by "
            "`experiments/026-metabolism-flux/scripts/thermo_exempt_gate.py` as its LP's "
            "witness and read by `read_log_c_init`. None is off and is what every run to "
            "date did.\n\n"
            "WHY IT MATTERS, MEASURED. The directional box takes the sign of the layer's "
            "OWN delta_r_G, which is evaluated at its own learned concentrations. At "
            "initialization those are a zero-centered parameter plus a random linear head, "
            "so the box is tightened against a concentration vector that licenses nothing "
            "in particular: of the 495 reactions the pFBA reference carries, the tightening "
            "forbids the direction of 145 outright "
            "(`results/probe_warmstart_init.json`). The LP that produced the exemption list "
            "also produced the concentrations under which that flux IS sign-consistent, so "
            "starting there makes the box agree with the reference by construction.\n\n"
            "WHAT IT DOES, AND WHAT IT IS NOT. It sets `log_c_base` so the squashed "
            "concentration equals the vector, and zeroes `log_c_from_context` so the "
            "genotype term starts at no deviation rather than at a random one. Both are "
            "INITIAL VALUES of learned parameters: training is free to move the "
            "concentrations anywhere in the window, which is why "
            "`feas_thermo_violation_frac` is logged every epoch rather than assumed.\n\n"
            "It is a per-GEM quantity: the vector is solved from this GEM's stoichiometry, "
            "this thermodynamic route and this exemption list, all named in the file and "
            "recorded in `coverage_report`. Another GEM brings its own."
        ),
    )
    warm_start_reference: str = Field(
        default="pfba",
        description=(
            "How the 'warmstart' parameterization builds its reference flux. 'pfba' -- a "
            "parsimonious flux balance solution of the UNPERTURBED model, computed at "
            "construction by `parsimonious_reference_flux` from the GEM's own S, bounds and "
            "objective column. There is no other route and no caller-supplied vector: a "
            "reference passed in from outside would be a per-organism constant smuggled "
            "into a module that has none, and pFBA is a property of the stoichiometric "
            "model rather than of yeast. Read only when the parameterization is "
            "'warmstart'."
        ),
    )
    warm_start_init_rel: float = Field(
        default=0.01,
        gt=0.0,
        description=(
            "The initial value of alpha, the residual head's one learned scalar, which IS "
            "the relative distance of the proposal from the reference: "
            "||v_pre - v_ref|| / ||v_ref|| = alpha exactly, because the null-space direction "
            "is normalized before it is scaled. MEASURED motivation: at a proposal "
            "displacement of 0.008 the existing K = 100 alternation reaches a "
            "turnover-thresholded balance ratio of 0.00118 with 82 active rows, and at 0.60 "
            "it reads 1.80 (`experiments/026-metabolism-flux/results/probe_warmstart.json`). "
            "0.01 starts the head inside the regime that works and lets the data term buy "
            "its way out."
        ),
    )
    warm_start_growth_fraction: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description=(
            "The fraction of the maximum objective value the parsimonious reference must "
            "still reach. 1.0 is pFBA proper: minimize total flux subject to OPTIMAL "
            "growth. Below 1.0 trades growth for sparsity, which is a different reference "
            "and is recorded as one."
        ),
    )
    warm_start_cache_path: str | None = Field(
        default=None,
        description=(
            "Where the reference flux is cached. None recomputes it at every construction, "
            "which is two LP solves of a 2,806 x 4,131 model. The cache stores a digest of "
            "S, the bounds and the objective alongside the flux and recomputes when they "
            "disagree, so a cache cannot silently outlive the GEM it was built from."
        ),
    )
    latent_stochastic: bool = Field(
        default=False,
        description=(
            "For 'nullspace' and 'projection': draw z = mu + sigma * eps from a learned "
            "amortized Gaussian rather than emitting mu, and return "
            "KL(N(mu, sigma) || N(0, I)) as out['c_kl']. Off reproduces the deterministic "
            "head exactly, including the order of the multiply by flux_scale. Distinct "
            "from `stochastic`, which is the per-reaction draw of the 'box' head; an arm "
            "that sets `stochastic` on a latent parameterization gets this instead, since "
            "`stochastic` is a no-op there and `stochastic` is the only boolean 026's "
            "build_flux_layer forwards."
        ),
    )
    draws: int = Field(
        default=1,
        ge=1,
        description=(
            "How many latent draws :meth:`FluxLayer.forward` takes per genotype. At 1, the "
            "default, the forward pass is bit-for-bit what it has always been. Above 1 the "
            "genotype chain, the box and the reaction energies are computed ONCE and the "
            "batch is expanded to B*draws rows, each carrying its own z, so every "
            "constraint residual becomes the mean over draws as well as over the batch and "
            "the readout sees one flux per draw. The draw is taken in eval too, since a "
            "K-draw arm is scored on the average of its per-draw predictions the way Flux "
            "Cone Learning averages over its OptGPSampler samples. Requires a latent "
            "parameterization with the draw switched on, since K copies of mu are not "
            "samples; the constructor asserts it rather than silently returning copies. "
            "Read off `config` at call time, like `stochastic`, so no caller signature "
            "changes and `sample()` keeps owning the explicit-N audit path."
        ),
    )
    latent_log_std_init: float = Field(
        default=-3.0,
        description=(
            "Initial log sigma of the latent posterior, from a zero-weight linear head so "
            "every sample starts at the same spread. exp(-3) = 0.050 against a latent whose "
            "deterministic magnitude is order 1, so the draw starts as a perturbation of "
            "the mean rather than as noise that replaces it."
        ),
    )
    kl_weight: float = Field(
        default=1.0e-3,
        description=(
            "Weight on out['c_kl'] when the `weights` dict carries no 'kl' key, which is "
            "the case for every arm built by 026's build_flux_layer. It changes nothing "
            "unless `latent_stochastic` is set, because c_kl is exactly zero otherwise."
        ),
    )
    log_c_from_latent: bool = Field(
        default=False,
        description=(
            "Make the ANCHORED concentration a function of the DRAWN latent as well as "
            "of the genotype context: log c = lo + (hi - lo) sigmoid(base + W context + "
            "U z), with U a zero-initialized `nn.Linear(n_latent, n_metabolites)`, so each "
            "of the K draws carries its own concentration vector and the amino-acid "
            "concentration readout becomes a function of the sample. Without it "
            "`thermo_log_c` reads the context alone and a K-draw contrast on that readout "
            "is null by construction (measured, wave 8 Finding 5 in "
            "[[experiments.027-betaxanthin-metabolic.wave8-sampling]]). With it, the "
            "reaction energies, the second-law box, the hinge and the dissipation are all "
            "computed AFTER the draw, per draw, and the `thermo_*` extras carry B*draws "
            "rows. Zero init keeps the flag-on forward pass bit-identical to flag-off "
            "until U moves, so the latent path is learned rather than injected as noise. "
            "Off, the default, is bit-identical to every earlier arm. Requires "
            "`latent_stochastic` and a parameterization that has a latent "
            "('projection' or 'nullspace') under ThermoMode.ANCHORED; the constructor "
            "asserts all three."
        ),
    )
    thermo_box: bool = Field(
        default=False,
        description=(
            "Make the SECOND LAW exact by construction, as a per-sample directional box. "
            "For every reaction the thermodynamic table prices and Box 2 does not exempt, "
            "the sign of the layer's own delta_r_G decides the sign the flux is allowed: "
            "dG > +epsilon forces ub <- min(ub, 0), dG < -epsilon forces lb <- max(lb, 0), "
            "and |dG| <= epsilon leaves the box alone. A flux inside that box satisfies "
            "v dG <= 0 identically, so c_thermo becomes a diagnostic that should read ~0 "
            "rather than a penalty the data term can outbid. Requires a projection "
            "parameterization ending on the box, since only the clip enforces it. Off by "
            "default; reachable from an 026 arm as the 'tdg' token of the "
            "parameterization string."
        ),
    )
    thermo_box_margin_kj: float = Field(
        default=0.0,
        description=(
            "Only reactions whose |delta_r_G| exceeds this margin, in kJ/mol, get a "
            "directional bound; the rest keep both directions. 0.0 falls back to "
            "`thermo_epsilon` (1e-3 kJ/mol), which constrains essentially every priced "
            "reaction and is what the first tdg cells ran. MEASURED motivation (job 1725): "
            "at that setting 70.6 percent of reactions are pinned below 1e-6 against 11.9 "
            "percent for the untightened head, and the resulting head scores at chance. A "
            "reaction half a kJ/mol from equilibrium has a sign the table cannot resolve, "
            "so pinning it spends a degree of freedom on no thermodynamic content. "
            "Reachable from an arm as the 'tdgm<margin>' token."
        ),
    )
    thermo_uncertainty: str = Field(
        default="isotropic",
        description=(
            "'isotropic' -- a free per-reaction offset on delta_r_G with an L2 prior, what "
            "every 026 and 027 run to date used. 'covariance' -- Thermo-Flux's Qm, with m a "
            "learned latent of the factor's rank and a N(0, I) prior; requires the caller "
            "to pass `delta_g_sqrt_cov`, since a covariance path with no covariance would "
            "be the isotropic path wearing its name."
        ),
    )
    null_space_cache_path: str | None = Field(
        default=None,
        description=(
            "Where torchcell.metabolism.constraints.null_space_basis caches the basis when "
            "the layer has to build it itself, which happens only for 'projection' with no "
            "`null_space` argument. None recomputes: a dense SVD of the 2,806 x 4,131 "
            "stoichiometric matrix, measured at 2.6 s, paid once at construction."
        ),
    )
    thermo_mode: ThermoMode = ThermoMode.ANCHORED
    thermo_source: ThermoSource = Field(
        default=ThermoSource.SHIPPED,
        description=(
            "Which route supplies the standard reaction energies: the GEM's shipped "
            "metabolite table summed over S ('shipped', what every 026/027 run before "
            "2026-09-10 used), our eQuilibrator recomputation ('equilibrator'), or the merge "
            "of both plus the shipped reaction column as a last resort ('union'). The table "
            "itself is built by parameters.resolve_reaction_delta_g and passed in; this field "
            "records which one, and the layer refuses a table that disagrees with it."
        ),
    )
    conc_lower_m: float = Field(
        default=CONC_LOWER_M,
        gt=0.0,
        description=(
            "Floor of the physiological concentration window the ANCHORED log concentration "
            "is squashed into, in molar. Defaults to constraints.CONC_LOWER_M, which is "
            "Thermo-Flux's 0.1 uM, so every arm configured before 2026-09-11 is unchanged."
        ),
    )
    conc_upper_m: float = Field(
        default=CONC_UPPER_M,
        gt=0.0,
        description=(
            "Ceiling of that window, in molar; the default is Thermo-Flux's 10 mM. A "
            "measured concentration ABOVE the ceiling is unreachable by the learned "
            "concentration whatever it learns, and the failure is silent, since the sigmoid "
            "simply saturates. Mulleder 2016's intracellular amino acids run to 9.5e-2 M "
            "(glutamine), with 10.6 percent of all 88,882 measurements above 1e-2 M, so an "
            "arm that SUPERVISES the learned concentration against them has to raise this "
            "(experiments/027-betaxanthin-metabolic/scripts/mulleder_concentration_window.py, "
            "results/mulleder_concentration_window.json). Raising it also widens the "
            "concentration term RT S^T ln c, which is a real loosening of the "
            "thermodynamic anchor and is why it is a config field rather than a new default."
        ),
    )
    constraint_warmup_epochs: int = Field(
        default=0,
        description=(
            "Ramp every constraint weight linearly from 0 to `constraint_final_scale` over "
            "this many epochs. Measured motivation: under the recipe the pooled arm escapes "
            "the betaxanthin mean plateau on (job 1634), the flux arms sat at spread 0.000 "
            "for 40 epochs with a validation DATA loss of 1 to 5 against the pooled arm's "
            "0.49 (jobs 1636, 1637). 0 keeps the constraints on from the first step, which is "
            "what those runs did."
        ),
    )
    constraint_delay_epochs: int = Field(
        default=0,
        description=(
            "Hold the constraint scale at EXACTLY zero for this many epochs before the "
            "warmup ramp begins. Measured motivation (2026-09-10): with the head fix and a "
            "final scale of 0.1 the flux arm never left the mean plateau in 80 epochs (job "
            "1669), and the same cell at scale 0.0 escaped at epoch 33 (job 1673). Delaying "
            "the ramp past the escape is what lets the module be measured on a head that has "
            "already learned, rather than on one the constraints never allowed to start. 0 "
            "reproduces the plain warmup."
        ),
    )
    constraint_final_scale: float = Field(
        default=1.0,
        description=(
            "Multiplies the whole weighted constraint sum once warmup is over. Separate from "
            "`weights` so the RELATIVE balance between the physics terms, which was tuned "
            "against their magnitudes at initialization, is not disturbed by changing how "
            "hard the physics binds overall."
        ),
    )
    use_shipped_transport_delta_g: bool = Field(
        default=False,
        description=(
            "Deprecated alias for transport_delta_g_source='shipped', kept because the "
            "026 and 027 arms were configured with it. True is read as 'shipped' when "
            "transport_delta_g_source is left at its default."
        ),
    )
    transport_delta_g_source: str = Field(
        default="off",
        description=(
            "Where the transport term comes from. 'off' -- zero, which asserts that "
            "moving a proton across a pH gradient is free. 'shipped' -- recover it from "
            "the GEM's own reaction column on the reactions whose metabolite-sum energy "
            "is structurally zero. 'computed' -- Thermo-Flux Eq. (7) evaluated from the "
            "stoichiometry, compartments and charges by torchcell.metabolism.transport, "
            "read from transport_terms_path. Default 'off' is what every 026/027 arm ran."
        ),
    )
    transport_terms_path: str | None = Field(
        default=None,
        description=(
            "JSON written by torchcell.metabolism.transport.write_transport_table. "
            "Required by transport_delta_g_source='computed', which does not fall back "
            "to the shipped column: a run must record which route it took."
        ),
    )
    use_enzyme_capacity: bool = True
    use_protein_budget: bool = True
    use_abundance_head: bool = Field(
        default=False,
        description=(
            "Predict a per-gene enzyme abundance E_g (mmol/gDW) from the gene token and "
            "use it in the capacity bound, |v_j| <= sum_u kcat_u E_u, in place of the "
            "nominal constant E_bar. The protein budget then constrains E directly, "
            "sum_g MW_g E_g <= p_avail, the GECKO pool constraint, which is what pins the "
            "scale of an otherwise free multiplier. A deletion still zeroes E_g hard. The "
            "learned availability gate is off in this mode: E_g is the gate."
        ),
    )
    abundance_prior_budget_fraction: float = Field(
        default=0.5,
        description=(
            "With no per-gene prior supplied, every gene starts at the equal-molar "
            "abundance that fills this fraction of p_avail. Sets the scale of E at "
            "initialization so the bound is neither slack nor zero."
        ),
    )
    abundance_log_range: float = Field(
        default=3.0,
        description=(
            "E_g = E0_g exp(range * tanh(w.h_g)): the head can move a gene at most "
            "e^range (~20x) either side of its prior, so the budget rather than the "
            "head's reach bounds the total."
        ),
    )
    use_dissipation_limit: bool = Field(
        default=False,
        description=(
            "Penalize g_diss above g_diss_limit. OFF by default since 2026-09-08: the "
            "Niebel 2019 limit is an empirical plateau fitted to one organism's physiology, "
            "not a law, and Thermo-Flux carries it as an option outside its constraint set. "
            "g_diss is always computed and logged so the limit serves as a VALIDATION "
            "readout: a model that stays under it unasked has learned something, a model "
            "trained under it has been told the answer. Every 026/027 grid ran with this on."
        ),
    )
    density_weight: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Weight on the TURNOVER-DENSITY term, which penalizes independent rows of S "
            "that carry no throughput. 0.0 is off and is what every cell before 2026-09-12 "
            "ran.\n\n"
            "WHY IT EXISTS, MEASURED. The Genesis balance criterion is a statistic over the "
            "rows of S, and a flux vector that is mostly zero passes it on the zeros: the "
            "`fluxforce_k100_boxforce` head meets the median rule on 100 of 100 top strains "
            "(jobs 1767, 1770, 1771) with only 26.5 percent of reactions carrying flux "
            "above 1e-4 mmol/gDW/h, and under the turnover-thresholded rule it reads 2.0, "
            "the worst value the statistic takes. Emptying the network is the degenerate "
            "way to balance it, and nothing in the loss argued against it.\n\n"
            "NOT MULTIPLIED BY `constraint_scale`. The recipe these cells run holds that "
            "scale at exactly 0.0, so a term inside the ramp is identically zero and the "
            "cell would be a silent duplicate of the arm it is meant to test. This term is "
            "a countermeasure to a degenerate solution rather than a physical constraint "
            "being ramped in, and `constraint_loss` adds it outside the product."
        ),
    )
    balance_active_row_set: str = Field(
        default="dynamic",
        description=(
            "Which rows the active-row balance term is priced on.\n\n"
            "'dynamic' (the default, and what jobs 1795 and 1797 ran) recomputes the set "
            "every forward from the flux under test: a row is in the set while its turnover "
            "clears the threshold. That set is a function of the very flux being penalized, "
            "so a head can leave the penalty by killing the row. MEASURED on job 1795 "
            "(density 0.5, balance 0.5): the fraction of rows above the threshold fell from "
            "0.00501 to 0.00225 over the run while `c_density` did not move (0.9837 to "
            "0.9860) and the training p95 fell to 1.7549 over that remaining 0.2 percent. "
            "The p95 moved by emptying the set rather than by balancing it.\n\n"
            "'frozen' fixes the set at the FIRST forward pass and holds it, so killing a row "
            "keeps it in the penalty at a ratio of 2 and the dodge costs rather than pays. "
            "PROVENANCE HAZARD, stated because it is easy to miss: the trainer's checkpoint "
            "stores `named_parameters` and no buffers, so a rebuild for an audit re-freezes "
            "on ITS first batch rather than restoring the training set. "
            "`coverage_report` therefore records the frozen row count and a digest of the "
            "row indices, and an audit whose digest differs was scored on a different set."
        ),
    )
    balance_active_weight: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Weight on the ACTIVE-ROW balance term, the squared relative residual over "
            "only the independent rows of S whose turnover clears "
            "`density_turnover_threshold`. 0.0 is off and is what every cell before "
            "2026-09-12 ran.\n\n"
            "WHY IT IS SEPARATE FROM `c_balance`, MEASURED. Every 027 cell runs the physics "
            "ramp at 0.0 or 0.1, and `constraint_loss` multiplies the whole physics sum, "
            "`c_balance` included, by that scale: at 0.0 the balance penalty is switched "
            "OFF and the only thing balancing the flux is the projection. The structural "
            "audit of job 1784 found 96.4 percent of the rows sitting at the worst possible "
            "balance ratio had an OPEN counterpart reaction the flux simply did not use, "
            "with the sign clamp and the enzyme bound blocking none of them "
            "(`stuck_rows_wave8_dens_d2p0_boxforce_k100_e80_s2700.json`). Balance was never "
            "asked for. This term asks for it, outside the ramp, like the density term.\n\n"
            "WHY ONLY THE ACTIVE ROWS. The rule this is judged by is the thresholded one, "
            "and a penalty averaged over all 2,593 rows is dominated by idle rows that pass "
            "it for free. Restricting to the rows that turn over makes the trained term and "
            "the judging rule the same statistic on the same set. A head could in principle "
            "dodge it by pushing a row back below the threshold, which is exactly what the "
            "density term prices, so the two are meant to run together."
        ),
    )
    parsimony_outside_ramp_weight: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Weight on `c_parsimony` added OUTSIDE the constraint ramp, beside the density "
            "and active-row balance terms. 0.0 is off and is what every cell to date ran.\n\n"
            "WHY IT IS NOT THE `weights['parsimony']` ENTRY. That one is inside the product "
            "`constraint_loss` multiplies by `constraint_scale`, and every wave-7 and "
            "wave-8 cell runs that scale at exactly 0.0, so the parsimony term those cells "
            "report is identically zero. Parsimony has therefore NEVER been applied in this "
            "experiment, which matters because parsimony is what makes an FBA flux sparse: "
            "the pFBA reference carries flux on 10.3 percent of reactions "
            "(`results/probe_fba_rule.json`) against the boxforce head's 28 to 32 percent. "
            "A term that is switched off cannot be the reason the head is dense.\n\n"
            "It is a countermeasure to a specific degenerate solution rather than a physical "
            "constraint being ramped in, which is the same argument `density_weight` and "
            "`balance_active_weight` already make, and it is added the same way."
        ),
    )
    density_turnover_threshold: float = Field(
        default=1.0e-3,
        gt=0.0,
        description=(
            "The turnover a row must reach to count as active, as a FRACTION of "
            "`flux_scale`: 1e-3 is 0.1 mmol/gDW/h at the shipped scale. Deliberately the "
            "same constant the feasibility audit's thresholded balance rule uses "
            "(`BALANCE_TURNOVER_THRESHOLDS`, primary 1e-3), so the term the model is "
            "trained against and the criterion it is judged by are the same statistic."
        ),
    )
    stochastic: bool = Field(
        default=False,
        description=(
            "Draw z from a learned Gaussian instead of emitting it, making the layer an "
            "amortized flux sampler rather than a flux selector."
        ),
    )
    flux_scale: float = DEFAULT_FLUX_SCALE
    softmin_beta: float = Field(
        default=8.0,
        description=(
            "Complex softmin temperature. Too soft and a complex behaves like a mean, so "
            "a deletion stops being lethal; too hard and gradients reach one subunit only."
        ),
    )
    availability_fold: str = Field(
        default="softmin",
        description=(
            "How a complex's per-gene availabilities fold into the unit's. 'softmin' is "
            "the log-sum-exp softmin every run to date used. 'normalized_softmin' divides "
            "it by its own value on the all-ones vector, which is the only place the two "
            "differ.\n\n"
            "WHY IT MATTERS, MEASURED. The softmin is "
            "-log(sum_g exp(-beta gamma_g))/beta, and on an INTACT complex, every gate at "
            "1, that is 1 - log(n_u)/beta rather than 1: it charges a complex for having "
            "subunits. At the shipped beta of 8 a 17-subunit complex reads 0.646 on a "
            "wild type where nothing is deleted, and with untrained gates near 0.4 it "
            "reads 0.0447, which is what set ATP synthase's box ceiling to 4.47 against a "
            "reference flux of 5.95 AFTER the enzyme capacity had been raised past it "
            "(`results/probe_warmstart_init.json`). It is the second 'minimum over 17 "
            "subunits' rule stacked on the same reaction.\n\n"
            "WHY NORMALIZING RATHER THAN SHIFTING. The obvious alternative, adding "
            "log(n_u)/beta to make the fold a log-MEAN-exp, also returns 1 on the intact "
            "complex and destroys the deletion: one gate at 0 among 16 at 1 then returns "
            "0.354 instead of 0. Dividing by the all-ones value is a per-unit POSITIVE "
            "constant, so it preserves the ordering and the gradient everywhere, returns "
            "exactly 1 on the intact complex, and leaves a deleted subunit at 0, because "
            "the raw softmin goes non-positive the moment any gate is 0 and the existing "
            "clamp catches it."
        ),
    )
    thermo_epsilon: float = 1.0e-3
    balance_epsilon: float = 1.0e-6
    p_avail: float = Field(
        default=0.35,
        description="Metabolic protein budget, g protein / gDW. Organism-specific.",
    )
    g_diss_limit: float = Field(
        default=3700.0,
        description=(
            "Gibbs dissipation limit, J/gDW/h. 3700 for S. cerevisiae per Niebel et al. "
            "2019, the value Thermo-Flux uses for its iMM904 case study."
        ),
    )
    default_kcat_per_s: float = Field(
        default=13.7,
        description=(
            "Fallback turnover when no measured or predicted value exists. GECKO's own "
            "practice is a default from the organism's kcat distribution; this is a "
            "placeholder that MUST be reported with its coverage, never presented as a "
            "measurement."
        ),
    )
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "balance": 1.0,
            "thermo": 1.0,
            "capacity": 1.0,
            "budget": 1.0,
            "dissipation": 1.0,
            "parsimony": 1.0e-3,
            "thermo_prior": 1.0e-2,
        }
    )


#: ``projection``, plus optional suffixes naming the step count, the exact end, and the
#: method: ``projection_k20_box_dykstra``. The suffix form exists because 026's
#: ``build_flux_layer`` forwards ``parameterization`` and no other new config field, so the
#: string is the only channel an ARMS entry has for any of the three.
_PROJECTION_PATTERN = re.compile(
    r"^projection(?:_k(\d+))?(?:_(balance|box))?(?:_(tdg)(?:m(\d+))?)?(?:_(dykstra))?$"
)

#: ``fluxforce``, with the projection's step-count and exact-end suffixes plus one of its
#: own: ``fluxforce_k20_box``, ``fluxforce_k20_force``. ``force`` ends on a clip into the box
#: tightened to the direction the reaction energy licenses, so bounds and the second law are
#: both exact and the balance residual is the leftover. There is no ``tdg`` token: the sign
#: clamp as the WHOLE mechanism is what this channel replaces.
_FLUXFORCE_PATTERN = re.compile(
    r"^fluxforce(?:_k(\d+))?(?:_(balance|box|force|boxforce))?(?:_(dykstra))?$"
)

#: ``warmstart``, with the same three suffixes: ``warmstart_k100_boxforce`` is the residual
#: head whose box is tightened to the licensed direction before an alternation of 100 clips.
#: The step count, the exact end and the method are all read off the string for the same
#: reason the other two channels read them there: it is the only field 026's
#: ``build_flux_layer`` forwards, so an arm can name its whole channel and nothing else.
_WARMSTART_PATTERN = re.compile(
    r"^warmstart(?:_k(\d+))?(?:_(balance|box|force|boxforce))?(?:_(dykstra))?$"
)


class ParameterizationSpec(BaseModel):
    """The parsed form of ``FluxLayerConfig.parameterization``.

    ``kind`` is one of ``box``, ``nullspace``, ``projection``, ``fluxforce``. The three
    projection fields are meaningful only for the last two, and a ``None`` there means the
    string named nothing and the config's own value stands. Kept as a record rather than a
    tuple so a run's log can print what it parsed.
    """

    model_config = ConfigDict(extra="forbid")

    kind: str
    projection_steps: int
    projection_end: str | None = None
    projection_method: str | None = None
    thermo_box: bool = False
    thermo_box_margin_kj: float | None = None
    raw: str


def parse_parameterization(raw: str, default_steps: int) -> ParameterizationSpec:
    r"""Read ``box``, ``nullspace``, or ``projection[_k<N>][_balance|_box][_tdg][_dykstra]``.

    The suffixes, in order, name the step count, which projection runs LAST (and is
    therefore exact), whether the box is tightened by the second law (``tdg``), and the
    method. ``projection_k20_box_tdg`` is twenty alternations ending on a directional clip.

    ``fluxforce[_k<N>][_balance|_box]`` is the same projection wrapped around a different
    pre-projection flux: the direction and the magnitude factor come from the reaction
    energy through a smooth odd function rather than from a sign clamp on the box. It takes
    the step count and the exact end and nothing else.

    An unrecognized string is an error rather than a silent fall back to ``box``: a run
    whose parameterization was a typo must not report a number.
    """
    if raw in {"box", "nullspace"}:
        return ParameterizationSpec(kind=raw, projection_steps=default_steps, raw=raw)
    warm = _WARMSTART_PATTERN.match(raw)
    if warm is not None:
        warm_steps = default_steps if warm.group(1) is None else int(warm.group(1))
        if warm_steps < 1:
            raise ValueError(f"warmstart needs at least one step; got {warm_steps}.")
        return ParameterizationSpec(
            kind="warmstart",
            projection_steps=warm_steps,
            projection_end=warm.group(2),
            projection_method=warm.group(3),
            raw=raw,
        )
    force = _FLUXFORCE_PATTERN.match(raw)
    if force is not None:
        force_steps = default_steps if force.group(1) is None else int(force.group(1))
        if force_steps < 1:
            raise ValueError(f"fluxforce needs at least one step; got {force_steps}.")
        return ParameterizationSpec(
            kind="fluxforce",
            projection_steps=force_steps,
            projection_end=force.group(2),
            projection_method=force.group(3),
            raw=raw,
        )
    match = _PROJECTION_PATTERN.match(raw)
    if match is None:
        raise ValueError(
            f"parameterization={raw!r} is not one of 'box', 'nullspace', "
            "'projection[_k<N>][_balance|_box][_tdg][_dykstra]', "
            "'fluxforce[_k<N>][_balance|_box|_force|_boxforce][_dykstra]', or "
            "'warmstart[_k<N>][_balance|_box|_force|_boxforce][_dykstra]'."
        )
    steps = default_steps if match.group(1) is None else int(match.group(1))
    if steps < 1:
        raise ValueError(f"projection needs at least one step; got {steps}.")
    return ParameterizationSpec(
        kind="projection",
        projection_steps=steps,
        projection_end=match.group(2),
        projection_method=match.group(5),
        thermo_box=match.group(3) is not None,
        thermo_box_margin_kj=(
            None if match.group(4) is None else float(match.group(4))
        ),
        raw=raw,
    )


class ReactionSqrtCovariance(BaseModel):
    r"""Thermo-Flux's :math:`Q`, the reaction square-root covariance, with its provenance.

    ``q`` is :math:`S^\top\Sigma^{1/2}` where :math:`\Sigma^{1/2}` is the metabolite factor
    eQuilibrator's component contribution emits, so :math:`Qm` with
    :math:`m\sim\mathcal N(0,I)` reproduces the correlated standard-energy error and
    :math:`\lVert Q_j\rVert_2` is the per-reaction sigma
    :class:`~torchcell.metabolism.parameters.ReactionThermoTable` already reports.

    ``mask`` is True only where EVERY participant of the reaction is inside the factor. A
    partially covered reaction's norm is an underestimate rather than a conservative bound,
    so its row is zeroed and it takes no offset at all.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    q: torch.Tensor = Field(description="[r, k] kJ/mol per unit latent.")
    mask: torch.Tensor = Field(description="[r] bool, True on fully covered reactions.")
    n_components: int
    n_reactions_covered: int
    source_path: str
    sha256: str


def reaction_sqrt_covariance(
    thermo_dir: str, s: torch.Tensor, met_ids: list[str]
) -> ReactionSqrtCovariance:
    r"""Build :math:`Q=S^\top\Sigma^{1/2}` from the eQuilibrator formation factor.

    Reads the same ``formation_sigma.npz`` that
    :func:`~torchcell.metabolism.parameters.resolve_reaction_delta_g` reads, and forms the
    same product; that function keeps only the row norms, and this one keeps the matrix,
    which is what the ``Qm`` offset needs. The two are checked against each other in
    ``tests/torchcell/metabolism/test_flux_layer.py``.

    Args:
        thermo_dir: ``$DATA_ROOT/data/torchcell/thermo_equilibrator``.
        s: ``[m, r]`` stoichiometric matrix, sparse or dense.
        met_ids: The GEM's metabolite ids, in row order of ``s``.
    """
    import hashlib
    import os.path as osp

    import numpy as np

    path = osp.join(thermo_dir, "formation_sigma.npz")
    payload = np.load(path, allow_pickle=False)
    covered_ids = [str(x) for x in payload["met_ids"]]
    factor = payload["sigma"]
    position = {mid: i for i, mid in enumerate(covered_ids)}
    full = np.zeros((len(met_ids), factor.shape[1]))
    covered = np.zeros(len(met_ids), dtype=bool)
    for i, mid in enumerate(met_ids):
        j = position.get(mid)
        if j is not None:
            full[i] = factor[j]
            covered[i] = True
    dense = (s.to_dense() if s.is_sparse else s).numpy()
    projected = dense.T @ full
    participates = np.abs(dense) > 0
    complete = ~(participates & ~covered[:, None]).any(axis=0)
    complete &= participates.any(axis=0)
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return ReactionSqrtCovariance(
        q=torch.from_numpy(projected * complete[:, None]).float(),
        mask=torch.from_numpy(complete),
        n_components=int(factor.shape[1]),
        n_reactions_covered=int(complete.sum()),
        source_path=path,
        sha256=digest.hexdigest(),
    )


class ThermoExemptList(BaseModel):
    """Reactions exempted from the second law by an external, provenance-carrying file.

    The list is a claim about ONE thermodynamic table under ONE concentration window: a
    reaction that no concentration can make downhill under ``union_external`` may be
    perfectly satisfiable under ``equilibrator``. So the route, the window and the table's
    own hashes are carried beside the ids and are recorded by ``coverage_report``, and the
    file's sha256 with them, because two runs citing "the exemption list" have to be able to
    prove they cited the same one.
    """

    model_config = ConfigDict(extra="forbid")

    reaction_ids: list[str]
    route: str | None = None
    window_m: list[float] | None = None
    epsilon_kj: float | None = None
    method: str | None = None
    gate_met: bool | None = None
    thermo_table_sha256: dict[str, str] | None = None
    source_path: str
    sha256: str


def read_thermo_exempt_list(path: str) -> ThermoExemptList:
    """Read the JSON ``thermo_exempt_gate.py`` writes, with its provenance and its hash.

    Only the fields that describe WHAT was exempted and UNDER WHAT are read; the LP's
    intermediate results stay in the file. A file with no ``exempt_reaction_ids`` is an
    error rather than an empty list, because an exemption path that silently exempts
    nothing would report a route it did not take.
    """
    import json
    import os.path as osp

    if not osp.isfile(path):
        raise ValueError(
            f"thermo_exempt_path={path!r} does not exist. Build it with "
            "experiments/026-metabolism-flux/scripts/thermo_exempt_gate.py."
        )
    with open(path) as handle:
        raw = handle.read()
    payload = json.loads(raw)
    if "exempt_reaction_ids" not in payload:
        raise ValueError(
            f"{path} carries no 'exempt_reaction_ids'. It is not an exemption list."
        )
    provenance = payload.get("provenance", {})
    return ThermoExemptList(
        reaction_ids=[str(r) for r in payload["exempt_reaction_ids"]],
        route=provenance.get("route"),
        window_m=provenance.get("window_m"),
        epsilon_kj=provenance.get("epsilon_kj"),
        method=payload.get("selected_method"),
        gate_met=payload.get("gate_met"),
        thermo_table_sha256=provenance.get("thermo_table_sha256"),
        source_path=path,
        sha256=hashlib.sha256(raw.encode()).hexdigest(),
    )


class LogConcentrationInit(BaseModel):
    """A log-concentration vector to start the anchored concentration at, with its origin.

    The vector is only meaningful beside the route, the window and the exemption list it was
    solved under: it is the LP's answer to "which concentrations make THIS flux downhill
    under THIS table once THOSE reactions are exempted", and none of those three is implied
    by the numbers.
    """

    model_config = ConfigDict(extra="forbid")

    ln_c: list[float]
    met_ids: list[str]
    route: str | None = None
    window_m: list[float] | None = None
    exempt_file: str | None = None
    exempt_file_sha256: str | None = None
    n_constraints_satisfied: int | None = None
    residual_total_slack_kj: float | None = None
    source_path: str
    sha256: str


def read_log_c_init(path: str) -> LogConcentrationInit:
    """Read the concentration witness ``thermo_exempt_gate.py`` writes, with its hash."""
    import json
    import os.path as osp

    if not osp.isfile(path):
        raise ValueError(
            f"log_c_init_path={path!r} does not exist. Build it with "
            "experiments/026-metabolism-flux/scripts/thermo_exempt_gate.py, which writes "
            "it beside the exemption list."
        )
    with open(path) as handle:
        raw = handle.read()
    payload = json.loads(raw)
    for key in ("ln_c", "met_ids"):
        if key not in payload:
            raise ValueError(
                f"{path} carries no {key!r}. It is not a concentration init."
            )
    return LogConcentrationInit(
        ln_c=[float(x) for x in payload["ln_c"]],
        met_ids=[str(m) for m in payload["met_ids"]],
        route=payload.get("route"),
        window_m=payload.get("window_m"),
        exempt_file=payload.get("exempt_file"),
        exempt_file_sha256=payload.get("exempt_file_sha256"),
        n_constraints_satisfied=payload.get("n_constraints_satisfied"),
        residual_total_slack_kj=payload.get("residual_total_slack_kj"),
        source_path=path,
        sha256=hashlib.sha256(raw.encode()).hexdigest(),
    )


class ReferenceFluxRecord(BaseModel):
    """A parsimonious reference flux with the solve that produced it.

    Carried beside the buffer so ``coverage_report`` can say what the warm start starts
    from without re-solving: a head whose reference is a degenerate or zero-growth solution
    would train perfectly happily and report nothing about it.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    v: torch.Tensor = Field(description="[r] mmol/gDW/h.")
    objective_index: int
    objective_value: float
    growth_fraction: float
    total_absolute_flux: float
    n_reactions_carrying_flux: int
    digest: str
    from_cache: bool


def parsimonious_reference_flux(
    s: torch.Tensor,
    lb: torch.Tensor,
    ub: torch.Tensor,
    objective_index: int,
    growth_fraction: float = 1.0,
    cache_path: str | None = None,
) -> ReferenceFluxRecord:
    r"""Parsimonious flux balance, solved from the GEM's tensors alone.

    Two linear programs, which is the textbook pFBA:

    .. math::
        z^\star=\max\{v_{b}:\ Sv=0,\ v^\ell\le v\le v^u\},\qquad
        \min\Big\{\sum_j|v_j|:\ Sv=0,\ v^\ell\le v\le v^u,\ v_b\ge\gamma z^\star\Big\}.

    The absolute value is handled by splitting :math:`v=p-n` with :math:`p,n\ge0`, and the
    split **preserves the original box exactly** rather than relaxing it: a reaction with
    :math:`v^\ell\ge0` gets :math:`p\in[v^\ell,v^u]` and :math:`n=0`, one with
    :math:`v^u\le0` gets the mirror, and only a genuinely reversible reaction gets both
    halves. Writing it the lazy way, :math:`p\in[0,\max(v^u,0)]` and
    :math:`n\in[0,\max(-v^\ell,0)]`, silently drops every forced-flux bound the medium
    imposes.

    NO SOLVER OBJECT, NO COBRA MODEL, NO ORGANISM. The inputs are the stoichiometric matrix,
    the bounds the layer has already clamped to its own flux scale, and the index of the
    objective column. That is what makes the warm start portable: the reference is
    recomputed for whatever GEM the layer is built on.

    Args:
        s: ``[m, r]`` stoichiometric matrix, sparse or dense.
        lb: ``[r]`` lower bounds, as the layer holds them.
        ub: ``[r]`` upper bounds.
        objective_index: Column of the growth reaction.
        growth_fraction: The fraction of :math:`z^\star` the parsimonious solution must
            still reach. 1.0 is pFBA proper.
        cache_path: Optional ``.npz`` cache. It stores a digest of the inputs and is
            recomputed when they disagree, so it cannot outlive the GEM it was built from.

    Returns:
        The flux and the record of the solve.
    """
    import os
    import os.path as osp

    import numpy as np
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix, hstack

    dense = (s.to_dense() if s.is_sparse else s).double().numpy()
    lower = lb.double().numpy()
    upper = ub.double().numpy()
    digest = hashlib.sha256()
    for array in (dense, lower, upper, np.array([objective_index, growth_fraction])):
        digest.update(np.ascontiguousarray(array).tobytes())
    key = digest.hexdigest()

    if cache_path is not None and osp.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=False)
        if str(cached["digest"]) == key:
            flux = cached["v"].astype(np.float64)
            return ReferenceFluxRecord(
                v=torch.from_numpy(flux).float(),
                objective_index=objective_index,
                objective_value=float(flux[objective_index]),
                growth_fraction=growth_fraction,
                total_absolute_flux=float(np.abs(flux).sum()),
                n_reactions_carrying_flux=int((np.abs(flux) > 1e-9).sum()),
                digest=key,
                from_cache=True,
            )

    n_reactions = dense.shape[1]
    sparse_s = csr_matrix(dense)
    b_eq = np.zeros(dense.shape[0])
    objective = np.zeros(n_reactions)
    objective[objective_index] = -1.0
    first = linprog(
        objective,
        A_eq=sparse_s,
        b_eq=b_eq,
        bounds=list(zip(lower, upper, strict=True)),
        method="highs",
    )
    if not first.success:
        raise ValueError(
            "the reference growth LP did not solve, so there is no parsimonious reference "
            f"to warm start from: {first.message}. The bounds the layer holds, clamped to "
            "its flux scale, may not admit growth."
        )
    z_star = float(-first.fun)

    # The split box, exactly equal to the original one.
    p_bounds: list[tuple[float, float]] = []
    n_bounds: list[tuple[float, float]] = []
    for j in range(n_reactions):
        low, high = float(lower[j]), float(upper[j])
        if low >= 0.0:
            p_bounds.append((low, high))
            n_bounds.append((0.0, 0.0))
        elif high <= 0.0:
            p_bounds.append((0.0, 0.0))
            n_bounds.append((-high, -low))
        else:
            p_bounds.append((0.0, high))
            n_bounds.append((0.0, -low))
    a_eq = hstack([sparse_s, -sparse_s], format="csr")
    growth_row = np.zeros(2 * n_reactions)
    growth_row[objective_index] = -1.0
    growth_row[n_reactions + objective_index] = 1.0
    second = linprog(
        np.ones(2 * n_reactions),
        A_eq=a_eq,
        b_eq=b_eq,
        A_ub=csr_matrix(growth_row.reshape(1, -1)),
        b_ub=np.array([-growth_fraction * z_star]),
        bounds=p_bounds + n_bounds,
        method="highs",
    )
    if not second.success:
        raise ValueError(
            f"the parsimonious LP did not solve: {second.message}. The growth floor "
            f"{growth_fraction} x {z_star:.6g} may be unreachable at LP tolerance; lower "
            "warm_start_growth_fraction rather than relaxing the bounds."
        )
    flux = second.x[:n_reactions] - second.x[n_reactions:]
    if cache_path is not None:
        os.makedirs(osp.dirname(cache_path) or ".", exist_ok=True)
        np.savez(cache_path, v=flux, digest=np.array(key))
    return ReferenceFluxRecord(
        v=torch.from_numpy(flux).float(),
        objective_index=objective_index,
        objective_value=float(flux[objective_index]),
        growth_fraction=growth_fraction,
        total_absolute_flux=float(np.abs(flux).sum()),
        n_reactions_carrying_flux=int((np.abs(flux) > 1e-9).sum()),
        digest=key,
        from_cache=False,
    )


class FluxSample(BaseModel):
    """A batch of flux draws with each draw's own feasibility.

    A sampler that reports one averaged residual cannot answer the question the Genesis
    metabolic-grounding target asks, which is what FRACTION of the draws for a nominated
    strain are feasible. Every field except ``v`` is therefore ``[S, B]``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    v: torch.Tensor = Field(description="[S, B, r] flux draws.")
    balance_ratio_median: torch.Tensor = Field(description="[S, B]")
    balance_ratio_max: torch.Tensor = Field(description="[S, B]")
    box_violation_frac: torch.Tensor = Field(description="[S, B]")
    thermo_violation_frac: torch.Tensor = Field(description="[S, B]")
    thermo_uphill_frac: torch.Tensor = Field(description="[S, B]")
    n_samples: int
    parameterization: str
    projection_steps: int
    projection_end: str


def gene_index_map(
    model_gene_ids: list[str], gem_gene_ids: list[str]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map GEM gene positions onto the model's gene-token positions.

    The transformer's token index runs over the whole genome (6,607 ORFs); the GEM knows
    only its metabolic subset (1,161). Both directions are needed: the flux layer reads
    gene tokens at GEM positions, and a perturbation arrives as a token index that must be
    tested for GEM membership.

    Returns:
        ``(gem_to_model, model_to_gem)``. ``gem_to_model[k]`` is the token index of GEM
        gene ``k``, or ``-1`` if that gene is absent from the model's universe.
        ``model_to_gem[t]`` is the GEM index of token ``t``, or ``-1``.
    """
    pos = {g: i for i, g in enumerate(model_gene_ids)}
    gem_to_model = torch.tensor(
        [pos.get(g, -1) for g in gem_gene_ids], dtype=torch.long
    )
    model_to_gem = torch.full((len(model_gene_ids),), -1, dtype=torch.long)
    for k, t in enumerate(gem_to_model.tolist()):
        if t >= 0:
            model_to_gem[t] = k
    return gem_to_model, model_to_gem


class FluxLayer(nn.Module):
    r"""Gene tokens in, a feasible flux vector and its feasibility residuals out.

    The chain, and every arrow in it is differentiable:

    .. math::
        \gamma_g \longrightarrow c_u \longrightarrow c_j \longrightarrow
        [\,v^\ell_j,\ v^u_j\,] \longrightarrow v_j
        \longrightarrow \{C_{\mathrm{bal}},C_{\mathrm{th}},C_{\mathrm{cap}},
        C_{\mathrm{bud}},C_{\mathrm{diss}}\}

    A deletion sets :math:`\gamma_g=0` **hard**, from the perturbation rather than from a
    learned gate, because whether a deleted gene is absent is not something the model
    should be free to disagree with. A complex takes a softmin over its subunits (the
    scarcest limits); isozymes sum (capacities add). A reaction whose availability falls
    to zero has :math:`v^\ell_j=v^u_j=0` and therefore carries **exactly** zero flux -- the
    deletion bites through the parameterization, not through a penalty that could be
    traded off against the data term.
    """

    # BUFFER TYPE DECLARATIONS. `nn.Module.__getattr__` is typed to return
    # `Tensor | Module`, so every `register_buffer` read is a union to mypy and every
    # index, call or bitwise op on one is an error. Declaring the buffers as class-level
    # `Tensor` annotations is the documented way to tell the checker what they are; it
    # changes no runtime behavior, since `register_buffer` still owns the values.
    s_indices: torch.Tensor
    s_values: torch.Tensor
    lb: torch.Tensor
    ub: torch.Tensor
    unit_gene_unit: torch.Tensor
    unit_gene_gene: torch.Tensor
    unit_reaction: torch.Tensor
    has_gpr: torch.Tensor
    gem_to_model: torch.Tensor
    model_to_gem: torch.Tensor
    constitutive: torch.Tensor
    independent_rows: torch.Tensor
    thermo_exempt: torch.Tensor
    is_exchange: torch.Tensor
    delta_r_g0: torch.Tensor
    delta_r_g0_mask: torch.Tensor
    delta_r_g0_sigma: torch.Tensor
    delta_r_g0_sigma_mask: torch.Tensor
    met_g_mask: torch.Tensor
    delta_r_g_transport: torch.Tensor
    kcat_per_h: torch.Tensor
    km_mm: torch.Tensor
    mw_kda: torch.Tensor
    null_basis: torch.Tensor
    log_e0: torch.Tensor
    delta_g_q: torch.Tensor
    frozen_active_rows: torch.Tensor
    v_reference: torch.Tensor

    def __init__(
        self,
        gem: GemTensors,
        model_gene_ids: list[str],
        config: FluxLayerConfig | None = None,
        compartments: CompartmentParameters | None = None,
        kcat_per_s: torch.Tensor | None = None,
        molecular_weight_kda: torch.Tensor | None = None,
        null_space: torch.Tensor | None = None,
        constitutive_genes: list[str] | None = None,
        abundance_prior_mmol_per_gdw: torch.Tensor | None = None,
        reaction_thermo: ReactionThermoTable | None = None,
        kcat_provenance: list[str] | None = None,
        km_mm: torch.Tensor | None = None,
        km_provenance: list[str] | None = None,
        delta_g_sqrt_cov: ReactionSqrtCovariance | None = None,
    ) -> None:
        """Build the layer against one GEM.

        Args:
            gem: Constraint tensors from :func:`~torchcell.metabolism.constraints.build_gem_tensors`.
            model_gene_ids: The transformer's gene ids, in token-index order.
            config: Switches and weights.
            compartments: pH / potential table for the transport term.
            kcat_per_s: ``[n_units]`` turnover per catalytic unit, or None to use the
                configured default for every unit. Coverage must be reported by the caller.
            molecular_weight_kda: ``[n_gem_genes]`` gene-product molecular weight, or None
                to disable the protein budget.
            null_space: ``[r, nullity]`` basis of ``ker S``, required only when
                ``config.parameterization == 'nullspace'``. Built once by the caller with
                :func:`~torchcell.metabolism.constraints.null_space_basis`, since it costs
                a singular value decomposition of the whole stoichiometric matrix.
            constitutive_genes: GEM gene ids that are always present at full availability.
                Heterologous cassette genes belong here: they are not open reading frames of
                the host, so they own no gene token, and the availability chain would
                otherwise read their absence from the token universe as availability zero
                and switch off the very pathway that was added. There is no learned gate on
                them, because a knockout library cannot delete them.
            abundance_prior_mmol_per_gdw: ``[n_gem_genes]`` wild-type enzyme abundance per
                GEM gene, the prior the abundance head moves around; a measured absolute
                proteome belongs here. None gives every gene the equal-molar abundance that
                fills ``abundance_prior_budget_fraction`` of the budget. Read only when
                ``config.use_abundance_head`` is set.
            reaction_thermo: A merged standard reaction energy table from
                :func:`~torchcell.metabolism.parameters.resolve_reaction_delta_g`, carrying a
                per-reaction provenance flag and, where eQuilibrator reaches every
                participant, a propagated uncertainty. None keeps the historical behavior of
                summing the GEM's own shipped metabolite table over ``S``. Its ``source``
                must equal ``config.thermo_source``.
            kcat_provenance: ``[n_units]`` provenance strings from the
                :class:`~torchcell.metabolism.parameters.ParameterTable` the ``kcat_per_s``
                values came from, so the coverage report can split measured from predicted
                from default rather than reporting one boolean.
            km_mm: ``[n_units]`` affinity per catalytic unit. Stored and reported; nothing in
                :meth:`forward` reads it, since the capacity bound is saturation-free.
            km_provenance: The same, for ``km_mm``.
            delta_g_sqrt_cov: Thermo-Flux's :math:`Q`, from
                :func:`reaction_sqrt_covariance`. Required by, and read only by,
                ``config.thermo_uncertainty='covariance'``.
        """
        super().__init__()
        self.config = config or FluxLayerConfig()
        self.compartments = compartments or CompartmentParameters()
        cfg = self.config
        # The warmup multiplier, moved by `set_constraint_epoch`. It starts at the FINAL
        # scale rather than at zero so a trainer that never calls the setter runs the
        # constraints exactly as every run before 2026-09-10 did.
        self.constraint_scale = float(cfg.constraint_final_scale)

        # Cached by `unit_gene_count` on first use; a plain attribute so no arm's
        # state_dict gains a key for a tensor it does not read.
        self._unit_gene_count: torch.Tensor | None = None
        self.n_reactions = gem.n_reactions
        self.n_metabolites = gem.n_metabolites
        self.rxn_ids = gem.rxn_ids
        self.met_ids = gem.met_ids
        cu = gem.catalytic_units
        self.n_units = cu.n_units
        self.n_gem_genes = len(cu.gene_ids)

        # -- structural buffers (never learned) -------------------------------
        self.register_buffer("s_indices", gem.s.indices())
        self.register_buffer("s_values", gem.s.values())
        scale = cfg.flux_scale
        self.register_buffer("lb", gem.lb.clamp(min=-scale, max=scale))
        self.register_buffer("ub", gem.ub.clamp(min=-scale, max=scale))
        self.register_buffer("unit_gene_unit", cu.unit_gene_index[0])
        self.register_buffer("unit_gene_gene", cu.unit_gene_index[1])
        self.register_buffer("unit_reaction", cu.unit_reaction)
        has_gpr = torch.zeros(self.n_reactions, dtype=torch.bool)
        has_gpr[cu.unit_reaction] = True
        self.register_buffer("has_gpr", has_gpr)

        gem_to_model, model_to_gem = gene_index_map(model_gene_ids, cu.gene_ids)
        self.register_buffer("gem_to_model", gem_to_model)
        self.register_buffer("model_to_gem", model_to_gem)
        self.n_genes_mapped = int((gem_to_model >= 0).sum())

        constitutive = torch.zeros(self.n_gem_genes, dtype=torch.bool)
        named = set(constitutive_genes or [])
        if named:
            gene_pos = {g: i for i, g in enumerate(cu.gene_ids)}
            unknown = sorted(named - set(gene_pos))
            if unknown:
                raise ValueError(
                    f"constitutive_genes not present in the GEM: {unknown}. "
                    "Apply the pathway to the model before building its tensors."
                )
            for g in named:
                constitutive[gene_pos[g]] = True
        self.register_buffer("constitutive", constitutive)
        self.n_constitutive_genes = int(constitutive.sum())

        rows = (
            gem.independent_rows
            if gem.independent_rows is not None
            else torch.arange(self.n_metabolites)
        )
        self.register_buffer("independent_rows", rows)

        # -- second-law exemptions (Thermo-Flux Box 2) ------------------------
        exempt = torch.zeros(self.n_reactions, dtype=torch.bool)
        if gem.exchange_indices is not None:
            exempt[gem.exchange_indices] = True
        if gem.biomass_index is not None:
            exempt[gem.biomass_index] = True
        for j, rid in enumerate(gem.rxn_ids):
            if "H2O" in rid.upper() or "water" in rid.lower():
                exempt[j] = True
        # THE EXTERNAL EXEMPTION LIST, on top of Box 2. Applied here so that everything
        # downstream, the hinge, the directional tightening, the flux-force channel and the
        # feasibility report, sees one exempt set and cannot disagree about it.
        self.thermo_exempt_list: ThermoExemptList | None = None
        self.n_thermo_exempt_from_file = 0
        if cfg.thermo_exempt_path is not None:
            exempt = self._extend_exempt(exempt, cfg.thermo_exempt_path, gem.rxn_ids)
        self.register_buffer("thermo_exempt", exempt)
        self.n_thermo_reactions = int((~exempt).sum())

        exch = torch.zeros(self.n_reactions, dtype=torch.bool)
        if gem.exchange_indices is not None:
            exch[gem.exchange_indices] = True
        self.register_buffer("is_exchange", exch)

        # -- thermodynamics ---------------------------------------------------
        self.thermo_mode = cfg.thermo_mode
        self.reaction_thermo = reaction_thermo
        if reaction_thermo is not None:
            # THE MERGED TABLE. Supplied by the caller from
            # `torchcell.metabolism.parameters.resolve_reaction_delta_g`, which is where the
            # route (shipped / eQuilibrator / union) and the per-reaction provenance live.
            # The layer does not choose the route; it records the one it was handed and
            # refuses a table that disagrees with its own config.
            if reaction_thermo.source != cfg.thermo_source:
                raise ValueError(
                    f"config asks for thermo_source={cfg.thermo_source} but the supplied "
                    f"table was built as {reaction_thermo.source}. One of the two is a "
                    "typo, and a run must not record a route it did not use."
                )
            if reaction_thermo.delta_r_g0.numel() != self.n_reactions:
                raise ValueError(
                    f"reaction_thermo covers {reaction_thermo.delta_r_g0.numel()} reactions "
                    f"against this GEM's {self.n_reactions}."
                )
            self.register_buffer("delta_r_g0", reaction_thermo.delta_r_g0.clone())
            self.register_buffer("delta_r_g0_mask", reaction_thermo.mask.clone())
            self.register_buffer("delta_r_g0_sigma", reaction_thermo.sigma.clone())
            self.register_buffer(
                "delta_r_g0_sigma_mask", reaction_thermo.sigma_mask.clone()
            )
            assert gem.thermo is not None, (
                "a merged reaction table is built from the GEM's own metabolite table, so "
                "the GEM must carry one."
            )
            self.register_buffer("met_g_mask", gem.thermo.met_mask)
            self.rt = gem.thermo.temperature_k * 8.314462618e-3
        elif gem.thermo is not None:
            drg0, drg0_mask = gem.standard_reaction_delta_g()
            self.register_buffer("delta_r_g0", drg0)
            self.register_buffer("delta_r_g0_mask", drg0_mask)
            self.register_buffer("delta_r_g0_sigma", torch.zeros(self.n_reactions))
            self.register_buffer(
                "delta_r_g0_sigma_mask", torch.zeros(self.n_reactions, dtype=torch.bool)
            )
            self.register_buffer("met_g_mask", gem.thermo.met_mask)
            self.rt = gem.thermo.temperature_k * 8.314462618e-3
        else:
            self.register_buffer("delta_r_g0_sigma", torch.zeros(self.n_reactions))
            self.register_buffer(
                "delta_r_g0_sigma_mask", torch.zeros(self.n_reactions, dtype=torch.bool)
            )
            self.register_buffer("delta_r_g0", torch.zeros(self.n_reactions))
            self.register_buffer(
                "delta_r_g0_mask", torch.zeros(self.n_reactions, dtype=torch.bool)
            )
            self.register_buffer(
                "met_g_mask", torch.zeros(self.n_metabolites, dtype=torch.bool)
            )
            self.rt = self.compartments.temperature_k * 8.314462618e-3
        self.n_transport_terms_sourced = 0
        self.register_buffer("delta_r_g_transport", self._transport_term(gem))

        # -- the standard-energy uncertainty latent --------------------------
        if cfg.thermo_uncertainty not in {"isotropic", "covariance"}:
            raise ValueError(
                f"thermo_uncertainty={cfg.thermo_uncertainty!r} is not 'isotropic' or "
                "'covariance'."
            )
        self.thermo_uncertainty = cfg.thermo_uncertainty
        self.delta_g_sqrt_cov = delta_g_sqrt_cov
        if cfg.thermo_uncertainty == "covariance":
            if delta_g_sqrt_cov is None:
                raise ValueError(
                    "thermo_uncertainty='covariance' needs Q; build it with "
                    "torchcell.metabolism.flux_layer.reaction_sqrt_covariance. There is no "
                    "isotropic fall back, because a run must not record a correlated "
                    "uncertainty it did not use."
                )
            if delta_g_sqrt_cov.q.shape[0] != self.n_reactions:
                raise ValueError(
                    f"Q covers {delta_g_sqrt_cov.q.shape[0]} reactions against this GEM's "
                    f"{self.n_reactions}."
                )
            self.register_buffer("delta_g_q", delta_g_sqrt_cov.q.clone())
            self.n_delta_g_components = int(delta_g_sqrt_cov.n_components)
            self.n_reactions_cov_covered = int(delta_g_sqrt_cov.n_reactions_covered)
        else:
            self.register_buffer("delta_g_q", torch.zeros(self.n_reactions, 0))
            self.n_delta_g_components = 0
            self.n_reactions_cov_covered = 0

        # -- kinetics ---------------------------------------------------------
        if kcat_per_s is None:
            kcat = torch.full((self.n_units,), cfg.default_kcat_per_s)
            self.kcat_is_default = True
        else:
            kcat = kcat_per_s
            self.kcat_is_default = False
        # mmol/gDW/h per g enzyme: kcat [1/s] * 3600 s/h. The capacity bound is
        # |v_j| <= kcat_gj * E_g, so the units of E follow from this line.
        self.register_buffer("kcat_per_h", kcat * 3600.0)
        # THE THREE-WAY TIER COUNT. `kcat_is_default` is a single boolean and it says
        # nothing about a table that is 4 percent measured, 91 percent predicted and 5
        # percent default -- which is what the predicted tier makes it. The counts travel
        # with the layer so a run's coverage report can separate a measurement from a
        # prediction without re-resolving the table.
        self.kcat_tier_counts = _tier_counts(kcat_provenance, self.n_units)
        self.km_tier_counts = _tier_counts(km_provenance, self.n_units)
        if km_mm is None:
            self.register_buffer("km_mm", torch.zeros(self.n_units))
            self.km_is_present = False
        else:
            # STORED, NOT USED. The capacity bound in force is saturation-free, so K_M enters
            # nothing in `forward`; it is carried so the table's coverage is on the record
            # before a saturation term is written against it.
            self.register_buffer("km_mm", km_mm)
            self.km_is_present = True
        if molecular_weight_kda is None:
            self.register_buffer("mw_kda", torch.ones(self.n_gem_genes) * 40.0)
            self.mw_is_default = True
        else:
            self.register_buffer("mw_kda", molecular_weight_kda)
            self.mw_is_default = False

        # -- learned parts ----------------------------------------------------
        d = cfg.hidden_dim
        spec = parse_parameterization(cfg.parameterization, cfg.projection_steps)
        self.parameterization = spec.kind
        self.projection_steps = spec.projection_steps
        # The string wins over the config field when it names one, because the string is the
        # only channel an 026 ARMS entry has and a cell must be able to say which end it
        # made exact in the one field its launcher forwards.
        self.projection_end = spec.projection_end or cfg.projection_end
        self.projection_method = spec.projection_method or cfg.projection_method
        self.thermo_box = bool(spec.thermo_box or cfg.thermo_box)
        self.thermo_box_margin_kj = (
            cfg.thermo_box_margin_kj
            if spec.thermo_box_margin_kj is None
            else spec.thermo_box_margin_kj
        )
        if self.thermo_box and not (
            spec.kind == "projection" and self.projection_end == "box"
        ):
            raise ValueError(
                "thermo_box needs parameterization='projection...' ending on the box "
                f"(got {spec.kind!r} ending on {self.projection_end!r}). The directional "
                "box is enforced by the clip, so a head that ends on the balance "
                "projection would report the second law as exact while leaving it soft."
            )
        if spec.kind in {"projection", "fluxforce", "warmstart"}:
            allowed = (
                {"balance", "box", "force", "boxforce"}
                if spec.kind in {"fluxforce", "warmstart"}
                else {"balance", "box"}
            )
            if self.projection_end not in allowed:
                raise ValueError(
                    f"projection_end={self.projection_end!r} is not one of "
                    f"{sorted(allowed)}. 'force' ends on the direction clip and exists only "
                    "on the fluxforce channel, which is the only one that has a direction "
                    "to restore."
                )
            if self.projection_method not in {"alternating", "dykstra"}:
                raise ValueError(
                    f"projection_method={self.projection_method!r} is not 'alternating' "
                    "or 'dykstra'."
                )
        if spec.kind == "fluxforce" and self.thermo_mode is not ThermoMode.ANCHORED:
            raise ValueError(
                "parameterization='fluxforce' needs ThermoMode.ANCHORED (got "
                f"{self.thermo_mode}). The channel reads its direction and its magnitude "
                "factor off delta_r_G, so with no tabulated energies it would be the box "
                "head wearing a thermodynamic name."
            )
        if (
            spec.kind == "warmstart"
            and self.projection_end in {"force", "boxforce"}
            and self.thermo_mode is not ThermoMode.ANCHORED
        ):
            raise ValueError(
                f"projection_end={self.projection_end!r} tightens the box to the direction "
                "the reaction energy licenses, which needs ThermoMode.ANCHORED (got "
                f"{self.thermo_mode}). Use 'warmstart_k<N>_box' when there is no table."
            )
        if spec.kind in {"nullspace", "projection", "fluxforce", "warmstart"}:
            if null_space is None:
                if spec.kind == "nullspace":
                    raise ValueError(
                        "parameterization='nullspace' needs the basis; build it with "
                        "torchcell.metabolism.constraints.null_space_basis(gem.s)."
                    )
                # THE PROJECTION ARM BUILDS ITS OWN. 026's `build_flux_layer` computes the
                # basis only for 'nullspace', and it is not this module's to edit, so the
                # layer falls back to computing it rather than refusing to run. It is a
                # 2.6 s dense SVD, paid once; a caller that already has the cached basis
                # should still pass it.
                null_space = null_space_basis(
                    gem.s, cache_path=cfg.null_space_cache_path
                )
            self.register_buffer("null_basis", null_space)
            self.n_latent = int(null_space.shape[1])
            # THE FLUX-FORCE CHANNEL USES THE BASIS AND NOT THE LATENT. Its pre-projection
            # flux is per reaction, read off the reaction energies, so the null-space latent
            # and its posterior head would be parameters nothing reads.
            if spec.kind != "fluxforce":
                self.latent_mlp = nn.Sequential(
                    nn.Linear(d, d), nn.ReLU(), nn.Linear(d, self.n_latent)
                )
            # THE POSTERIOR SPREAD HEAD IS ALWAYS BUILT ON THE PROJECTION KIND, whether or
            # not the run samples, so a deterministically trained cell can be switched into
            # a sampler after its weights are loaded. `audit_top_strain_feasibility.py` does
            # exactly that, by setting `config.stochastic` on a layer that is already built,
            # and a head created lazily at that point would not be in the checkpoint. On
            # 'nullspace' the head is built only when asked for, so the existing arm's
            # parameter set and its consumption of the init RNG are untouched.
            if spec.kind in {"projection", "warmstart"} or cfg.latent_stochastic:
                # Zero weights and a constant bias, so every genotype starts at the same
                # spread and the posterior widens only where the data asks it to.
                self.latent_log_std = nn.Linear(d, self.n_latent)
                nn.init.zeros_(self.latent_log_std.weight)
                nn.init.constant_(self.latent_log_std.bias, cfg.latent_log_std_init)
            self.null_basis_orthonormality_error, self.null_basis_balance_error = (
                self._validate_null_basis()
            )
        else:
            self.n_latent = 0
            self.null_basis_orthonormality_error = 0.0
            self.null_basis_balance_error = 0.0

        # -- the warm start's reference flux, solved HERE and not passed in -----
        self.reference_flux: ReferenceFluxRecord | None = None
        if spec.kind == "warmstart":
            if cfg.warm_start_reference != "pfba":
                raise ValueError(
                    f"warm_start_reference={cfg.warm_start_reference!r} is not 'pfba', "
                    "which is the only route implemented. A caller-supplied reference is "
                    "deliberately not accepted: it would be an organism-specific constant "
                    "in a module that has none."
                )
            if gem.biomass_index is None:
                raise ValueError(
                    "the warm start needs the GEM's objective column to solve pFBA, and "
                    "this GEM declares none. build_gem_tensors detects it by objective "
                    "coefficient or takes biomass_id."
                )
            # THE BOUNDS ARE THE LAYER'S OWN, already clamped to flux_scale, so the
            # reference is inside the box the head will clip it into rather than being a
            # vector the parameterization cannot represent.
            self.reference_flux = parsimonious_reference_flux(
                gem.s,
                self.lb,
                self.ub,
                int(gem.biomass_index),
                growth_fraction=float(cfg.warm_start_growth_fraction),
                cache_path=cfg.warm_start_cache_path,
            )
            self.register_buffer("v_reference", self.reference_flux.v.clone())
            # ALPHA IS THE PROPOSAL'S RELATIVE DISTANCE, exactly, because the null-space
            # direction is normalized before it is scaled. Learned in log space so it
            # stays positive without a clamp, and so a multiplicative step in it is what
            # gradient descent takes.
            self.warm_log_alpha = nn.Parameter(
                torch.tensor(math.log(float(cfg.warm_start_init_rel)))
            )
        # NO BUFFER AND NO PARAMETER ON ANY OTHER KIND, deliberately: registering an empty
        # `v_reference` everywhere would change the state_dict of every arm already trained,
        # for a tensor none of them reads.

        if cfg.balance_active_row_set not in {"dynamic", "frozen"}:
            raise ValueError(
                f"balance_active_row_set={cfg.balance_active_row_set!r} is not 'dynamic' "
                "or 'frozen'."
            )
        # The frozen row set, empty until the first forward fills it. A buffer so it moves
        # with the module across devices; NOT restored by the 027 checkpoint, which stores
        # parameters only, and `coverage_report` records its digest for exactly that reason.
        self.register_buffer(
            "frozen_active_rows",
            torch.zeros(int(self.independent_rows.numel()), dtype=torch.bool),
        )
        self.frozen_active_rows_ready = False
        self.reaction_embedding = nn.Embedding(self.n_reactions, cfg.reaction_embed_dim)
        # The fluxforce head carries its spread in a head of its own, built unconditionally,
        # so the width of this one does not depend on `stochastic` there; every other kind
        # keeps the width it has always had, and with it its checkpoints and its init RNG.
        z_out = 1 if spec.kind == "fluxforce" else (2 if cfg.stochastic else 1)
        self.flux_mlp = nn.Sequential(
            nn.Linear(d + cfg.reaction_embed_dim, d), nn.ReLU(), nn.Linear(d, z_out)
        )
        if spec.kind == "fluxforce":
            # ALWAYS BUILT, for the same reason the projection's posterior head is: a cell
            # trained deterministically is switched into a sampler after its weights load,
            # and a head created at that point would not be in the checkpoint. Zero weights
            # and a constant bias, so every genotype starts at the same spread.
            self.flux_log_std = nn.Linear(d + cfg.reaction_embed_dim, 1)
            nn.init.zeros_(self.flux_log_std.weight)
            nn.init.constant_(self.flux_log_std.bias, cfg.latent_log_std_init)
        # A learned gate on non-deleted genes: dosage, alleles, over-expression. Deleted
        # genes never reach it.
        self.availability = nn.Linear(d, 1)
        # The protein node. One linear readout per gene token into a log-abundance
        # deviation from the prior; the prior carries the units and the budget carries the
        # scale, so the head learns only relative abundance.
        if cfg.use_abundance_head:
            self.abundance = nn.Linear(d, 1)
            if abundance_prior_mmol_per_gdw is None:
                # Equal molar abundance across GEM genes filling a fraction of the budget:
                # protein mass is sum_g MW_g[g/mmol] E_g[mmol/gDW], so E0 = frac p_avail /
                # sum_g MW_g. With yeast-GEM's 1,161 genes at ~40 kDa this is ~4e-6
                # mmol/gDW, against the 1e-3 the nominal E_bar path assumes.
                e0 = torch.full(
                    (self.n_gem_genes,),
                    cfg.abundance_prior_budget_fraction
                    * cfg.p_avail
                    / float(self.mw_kda.sum()),
                )
                self.abundance_prior_is_default = True
            else:
                e0 = abundance_prior_mmol_per_gdw.clone().float()
                self.abundance_prior_is_default = False
            self.register_buffer("log_e0", torch.log(e0.clamp(min=1e-12)))
        else:
            self.register_buffer("log_e0", torch.zeros(self.n_gem_genes))
            self.abundance_prior_is_default = True
        if self.thermo_mode is ThermoMode.FREE:
            self.mu_base = nn.Parameter(torch.zeros(self.n_metabolites))
            self.mu_from_context = nn.Linear(d, self.n_metabolites)
        elif self.thermo_mode is ThermoMode.ANCHORED:
            # log-concentration, squashed into the physiological window
            self.log_c_base = nn.Parameter(torch.zeros(self.n_metabolites))
            self.log_c_from_context = nn.Linear(d, self.n_metabolites)
            if self.thermo_uncertainty == "isotropic":
                # The Q m offset of Thermo-Flux Eq. (5) with Q dropped: one free direction
                # per reaction. What every 026 and 027 run to date fit.
                self.delta_g_offset = nn.Parameter(torch.zeros(self.n_reactions))
            else:
                # Q m proper. m lives in the factor's component space, so the reactions can
                # only move in the directions the component contribution left uncertain.
                self.delta_g_m = nn.Parameter(torch.zeros(self.n_delta_g_components))

        # K DRAWS NEEDS AN ACTUAL DRAW. Checked here rather than in the config validator
        # because `latent_stochastic` is a property of the BUILT layer: on the projection
        # kind either boolean turns the draw on, and on `nullspace` only one of them does.
        # Without this, a K = 16 arm would run 16 copies of mu, report a per-draw mean that
        # equals the deterministic prediction, and cost 16x the compute to do it.
        assert cfg.draws == 1 or self.latent_stochastic, (
            f"draws={cfg.draws} on a layer whose latent is deterministic "
            f"(parameterization {self.parameterization!r}, latent_stochastic "
            f"{cfg.latent_stochastic}, stochastic {cfg.stochastic}). K copies of mu are not "
            "K samples; set latent_stochastic on the arm."
        )

        # A DRAW-DEPENDENT CONCENTRATION NEEDS A DRAW, A LATENT AND A CONCENTRATION. The
        # flag reads z inside `reaction_potential`, so it is meaningless on a layer whose
        # latent is deterministic (every draw would carry the same concentration again), on
        # a parameterization with no null-space latent, and outside ANCHORED, where no
        # concentration is learned at all. A property of the BUILT layer, like `draws`
        # above, since `latent_stochastic` is one too.
        self.log_c_from_latent_enabled = bool(cfg.log_c_from_latent)
        if self.log_c_from_latent_enabled:
            assert self.parameterization in {"nullspace", "projection"}, (
                f"log_c_from_latent needs a null-space latent to read; parameterization "
                f"{self.parameterization!r} has none."
            )
            assert self.latent_stochastic, (
                "log_c_from_latent on a layer whose latent is deterministic "
                f"(parameterization {self.parameterization!r}, latent_stochastic "
                f"{cfg.latent_stochastic}, stochastic {cfg.stochastic}). A concentration "
                "read off mu is the context-only concentration again; set "
                "latent_stochastic on the arm."
            )
            assert self.thermo_mode is ThermoMode.ANCHORED, (
                f"log_c_from_latent needs ThermoMode.ANCHORED (got {self.thermo_mode}); "
                "no concentration is learned otherwise."
            )
            # ZERO WEIGHT, AND NO RNG CONSUMED. `nn.Linear` draws a Kaiming init before it
            # is zeroed, which would shift every parameter initialized after this layer
            # and make a flag-on cell differ from its flag-off pair at init for a reason
            # that is not the flag. The fork restores the generator state on exit, so the
            # two cells consume the init RNG identically and differ only once U learns.
            with torch.random.fork_rng(devices=[]):
                self.log_c_from_latent = nn.Linear(
                    self.n_latent, self.n_metabolites, bias=False
                )
            nn.init.zeros_(self.log_c_from_latent.weight)

        # -- where the anchored concentration STARTS -----------------------------
        # Last, because it writes into `log_c_base` and `log_c_from_context`, which the
        # ANCHORED branch above creates. A layer built with the field set therefore begins
        # at the witness rather than at a random deviation from it.
        self.log_c_init: LogConcentrationInit | None = None
        self.log_c_init_max_error = 0.0
        if cfg.log_c_init_path is not None:
            self.apply_log_c_init(cfg.log_c_init_path)

    # -- construction helpers -------------------------------------------------

    @property
    def latent_stochastic(self) -> bool:
        """Whether the latent is drawn rather than emitted, read at CALL time.

        A property rather than an ``__init__`` flag because a trained cell is switched into
        a sampler after its weights are loaded: ``audit_top_strain_feasibility.py`` sets
        ``config.stochastic`` on a built layer and then draws. On the projection kind either
        switch turns sampling on, since the spread head is built there unconditionally; on
        ``nullspace`` only ``latent_stochastic`` does, so ``stochastic`` stays the no-op it
        has always been for that arm.

        On ``fluxforce`` the draw is not on a null-space latent at all: it is on the
        per-reaction magnitude logit, through ``flux_log_std``. The name is kept because it
        is the switch every caller already reads.
        """
        if self.parameterization == "fluxforce":
            return bool(self.config.latent_stochastic or self.config.stochastic)
        if not hasattr(self, "latent_log_std"):
            return False
        if self.parameterization in {"projection", "warmstart"}:
            return bool(self.config.latent_stochastic or self.config.stochastic)
        return bool(self.config.latent_stochastic)

    def _extend_exempt(
        self, exempt: torch.Tensor, path: str, rxn_ids: list[str]
    ) -> torch.Tensor:
        """Fold an external exemption list into the Box 2 mask, and record what it was."""
        listed = read_thermo_exempt_list(path)
        position = {rid: j for j, rid in enumerate(rxn_ids)}
        unknown = sorted(set(listed.reaction_ids) - set(position))
        if unknown:
            raise ValueError(
                f"the exemption list at {path} names reactions this GEM does not have: "
                f"{unknown[:10]}. A list built against another model must not be applied "
                "by index or by silent omission."
            )
        added = 0
        for rid in listed.reaction_ids:
            j = position[rid]
            if not bool(exempt[j]):
                added += 1
            exempt[j] = True
        self.thermo_exempt_list = listed
        self.n_thermo_exempt_from_file = added
        return exempt

    def apply_log_c_init(self, path: str) -> float:
        r"""Start the anchored concentration at a vector that licenses a real flux.

        The layer squashes its concentration into the window,
        :math:`\log c = lo + (hi-lo)\,\sigma(\text{base} + W\,\text{context})`, so setting
        the concentration to a target :math:`\log c^\star` means inverting the sigmoid on
        the base and starting the context term at zero:

        .. math::
            \text{base} = \operatorname{logit}
                \frac{\log c^\star - lo}{hi - lo},\qquad W \leftarrow 0.

        BOTH HALVES ARE NEEDED AND BOTH ARE INITIAL VALUES. Leaving :math:`W` at its random
        initialization would add an arbitrary per-genotype deviation to the very vector
        being set, so the concentration the directional box sees would not be the witness.
        Zeroing it starts the genotype term at no deviation; the gradient reaches it from
        the first step, so the genotype dependence is learned rather than removed.

        The target is clamped a hair inside the window, since a value ON the boundary has an
        infinite logit, and the clamp distance is returned so a caller can see whether the
        witness sat at a bound.

        Returns:
            The largest absolute difference, in log-concentration units, between the
            realized initial concentration and the file's vector. Zero to floating point
            unless the witness sat on the window boundary.
        """
        if self.thermo_mode is not ThermoMode.ANCHORED:
            raise ValueError(
                f"log_c_init needs ThermoMode.ANCHORED (got {self.thermo_mode}); no "
                "concentration is learned otherwise."
            )
        record = read_log_c_init(path)
        if record.met_ids != list(self.met_ids):
            raise ValueError(
                f"the concentration init at {path} is indexed by a different metabolite "
                f"list ({len(record.met_ids)} entries against this GEM's "
                f"{len(self.met_ids)}). A vector applied by position to another model's "
                "metabolites is not a concentration, it is noise."
            )
        lo = math.log(self.config.conc_lower_m)
        hi = math.log(self.config.conc_upper_m)
        if record.window_m is not None:
            wanted = [math.log(record.window_m[0]), math.log(record.window_m[1])]
            if abs(wanted[0] - lo) > 1e-9 or abs(wanted[1] - hi) > 1e-9:
                raise ValueError(
                    f"the concentration init was solved in the window {record.window_m} M "
                    f"and this layer squashes into [{self.config.conc_lower_m:g}, "
                    f"{self.config.conc_upper_m:g}] M. A vector rescaled into a different "
                    "window is a different concentration."
                )
        target = torch.tensor(record.ln_c, dtype=torch.float32)
        span = hi - lo
        # A hair inside the window: the logit of 0 or 1 is infinite, and the witness sits ON
        # a bound for every metabolite the LP did not care about.
        margin = 1.0e-6
        fraction = ((target - lo) / span).clamp(min=margin, max=1.0 - margin)
        with torch.no_grad():
            self.log_c_base.copy_(torch.log(fraction) - torch.log1p(-fraction))
            nn.init.zeros_(self.log_c_from_context.weight)
            nn.init.zeros_(self.log_c_from_context.bias)
            realized = lo + span * torch.sigmoid(self.log_c_base)
            error = float((realized - target).abs().max())
        self.log_c_init = record
        self.log_c_init_max_error = error
        self.config.log_c_init_path = path
        return error

    def apply_thermo_exempt_list(self, path: str) -> int:
        """Switch the exemption list on for a layer that is already built.

        WHY THIS EXISTS AND IS NOT A SMELL. 026's ``build_flux_layer`` owns the
        construction of every layer these experiments train and forwards only the fields its
        own registry names, so a caller that wants a construction-time switch it does not
        forward has exactly two options: edit that registry, or set the switch on the built
        object. 027's trainer already takes the second for the density weight, the
        active-row balance weight and the concentration window. This is the same move for
        the exemption list, and unlike those it has to rebuild a buffer, which is why it is
        a method rather than an assignment.

        The exempt set is read by the hinge, by the directional tightening and by
        ``feasibility``, all at call time, so the change takes effect everywhere at once.

        Returns:
            How many reactions this call newly exempted.
        """
        exempt = self._extend_exempt(self.thermo_exempt.clone(), path, self.rxn_ids)
        self.thermo_exempt = exempt
        self.n_thermo_reactions = int((~exempt).sum())
        self.config.thermo_exempt_path = path
        return self.n_thermo_exempt_from_file

    def _validate_null_basis(self, probes: int = 8) -> tuple[float, float]:
        r"""Measure the two properties the projection rests on, rather than assuming them.

        :math:`P_{\mathrm{bal}}(v)=\mathcal N(\mathcal N^\top v)` is the orthogonal
        projector onto :math:`\ker S` only when :math:`\mathcal N^\top\mathcal N=I`. A basis
        that is merely a basis makes it an oblique map, which still lands in
        :math:`\ker S` but no longer contracts the distance to the box, so the alternating
        projection need not converge. Rather than form the ``1538 x 1538`` Gram matrix, this
        applies :math:`\mathcal N^\top\mathcal N` to a few random latents, which is the same
        assertion at 1/200th of the cost, and it also reads off
        :math:`\lVert S\mathcal Nz\rVert_\infty` so the balance claim is measured too.

        Returns:
            ``(orthonormality_error, balance_error)``, both in the units the caller should
            report: the first is dimensionless, the second the largest absolute row of
            :math:`S\mathcal Nz` relative to the largest absolute entry of
            :math:`\mathcal Nz`.
        """
        generator = torch.Generator(device="cpu").manual_seed(0)
        z = torch.randn(probes, self.n_latent, generator=generator)
        v = z @ self.null_basis.t()
        recovered = v @ self.null_basis
        ortho = float((recovered - z).abs().max())
        residual = float(self._s_matmul(v).abs().max())
        balance = residual / max(float(v.abs().max()), 1e-12)
        if ortho > 1e-3:
            raise ValueError(
                f"the supplied null-space basis is not orthonormal: N^T N z differs from z "
                f"by {ortho:.3e}. The balance projection N (N^T v) is only a projector for "
                "an orthonormal basis; constraints.null_space_basis returns one."
            )
        return ortho, balance

    def _thermo_tighten(
        self, lb: torch.Tensor, ub: torch.Tensor, delta_g: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Tighten the box so that every flux inside it runs downhill.

        The second law asks :math:`v_j\Delta_rG_j\le0`, which for a KNOWN sign of
        :math:`\Delta_rG_j` is just a sign restriction on :math:`v_j`, and a sign
        restriction is a bound. So it belongs in the box rather than in the objective:

        .. math::
            \Delta_rG_j>+\epsilon\ \Rightarrow\ v^u_j\leftarrow\min(v^u_j,0),\qquad
            \Delta_rG_j<-\epsilon\ \Rightarrow\ v^\ell_j\leftarrow\max(v^\ell_j,0),

        with :math:`|\Delta_rG_j|\le\epsilon` left alone, because near equilibrium the sign
        is not resolved and forcing one would be asserting a direction the data does not
        have. Reactions outside ``dg_mask`` or exempted by Thermo-Flux's Box 2 are untouched.

        **This is a per-sample box**, because :math:`\Delta_rG` carries the learned
        concentration term and therefore depends on the genotype. That is what makes it a
        constraint the model can satisfy rather than one it must fight: to license a flux in
        a direction, the model moves the concentrations inside the physiological window
        until the reaction energy has the sign it needs, which is exactly the degree of
        freedom Thermo-Flux gives its own solver.

        The tightening can empty a box, when a reaction carries a strictly positive lower
        bound and an energy that forbids forward flux. The second law wins there and the
        lower bound is dropped to meet the upper one, but the count is returned rather than
        absorbed, because a genotype with many emptied boxes is one whose medium or forced
        fluxes disagree with its thermodynamics.

        Returns:
            ``(lb, ub, emptied)`` with ``emptied`` a ``[B, r]`` bool of the boxes the
            tightening inverted.
        """
        active = (self.delta_r_g0_mask & (~self.thermo_exempt)).unsqueeze(0)
        # The margin, not the numerical epsilon, is what decides which reactions the box
        # constrains. A reaction 0.5 kJ/mol from equilibrium has a sign the table cannot
        # resolve, and pinning it costs a degree of freedom for no thermodynamic content.
        eps = max(self.config.thermo_epsilon, self.thermo_box_margin_kj)
        uphill = active & (delta_g > eps)
        downhill = active & (delta_g < -eps)
        zero = ub.new_zeros(())
        ub = torch.where(uphill, torch.minimum(ub, zero), ub)
        lb = torch.where(downhill, torch.maximum(lb, zero), lb)
        emptied = lb > ub
        return torch.minimum(lb, ub), ub, emptied

    def _project_balanced(self, v: torch.Tensor) -> torch.Tensor:
        r"""Orthogonal projection onto :math:`\ker S`, exact for an orthonormal basis."""
        return (v @ self.null_basis) @ self.null_basis.t()

    def _project_feasible(
        self, v: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor
    ) -> torch.Tensor:
        r"""Alternate the balance projection and the box clip, ``projection_steps`` times.

        ``K`` counts the CLIPS. Each is followed by the balance projection except, under
        ``projection_end='box'``, the last one, so the returned point is exactly in the box
        or exactly in :math:`\ker S` according to the config and the other residual is what
        :meth:`feasibility` reports.

        ``clamp`` with tensor bounds passes gradient to the input where the flux is interior
        and to ``lb``/``ub`` where it is clipped, so the genotype term reaches the flux
        through the box even when the latent term is saturated, which is the property the
        deletion needs.
        """
        steps = self.projection_steps
        # 'boxforce' ends on the clip like 'box' does; what differs is the box it clips
        # into, which the caller has already tightened to the licensed direction.
        end_on_box = self.projection_end in {"box", "boxforce"}
        if self.projection_method == "alternating":
            for k in range(steps):
                v = torch.minimum(torch.maximum(v, lb), ub)
                if not (end_on_box and k == steps - 1):
                    v = self._project_balanced(v)
            return v
        # DYKSTRA. The plain alternation reaches A INTERSECT B; Dykstra's correction terms
        # make the limit the point of the intersection NEAREST v0, which is the difference
        # between "a feasible flux" and "the feasible flux this genotype's latent asked for".
        p = torch.zeros_like(v)
        q = torch.zeros_like(v)
        for k in range(steps):
            y = torch.minimum(torch.maximum(v + p, lb), ub)
            p = v + p - y
            if end_on_box and k == steps - 1:
                return y
            w = self._project_balanced(y + q)
            q = y + q - w
            v = w
        return v

    def flux_force_efficacy(self, delta_g: torch.Tensor) -> torch.Tensor:
        r"""The flux-force relation, :math:`f(\Delta_rG)=-\tanh(\Delta_rG/2RT)`.

        For an elementary reaction the forward and reverse rates stand in the ratio
        :math:`v^+/v^-=e^{-\Delta_rG/RT}`, so the net flux is the forward rate times

        .. math::
            \frac{1-e^{\Delta_rG/RT}}{1+e^{\Delta_rG/RT}}
            = -\tanh\!\Big(\frac{\Delta_rG}{2RT}\Big),

        which is Beard and Qian's flux-force relation and the quantity Noor et al. (2014)
        call the flux-force efficacy. Three properties are what the channel is built on, and
        each is a property of this function rather than of a clamp around it:

        * **the sign is structural**, :math:`\operatorname{sign} f=-\operatorname{sign}
          \Delta_rG`, so a nonnegative magnitude times :math:`f` satisfies
          :math:`v\,\Delta_rG\le0` identically;
        * **near equilibrium it goes quiet** rather than being pinned, since
          :math:`f\to0` smoothly as :math:`\Delta_rG\to0`, which is what the directional box
          instead spends a degree of freedom on;
        * **it is differentiable everywhere**, :math:`f'=-\operatorname{sech}^2(x)/2RT<0`,
          so the learned concentrations receive gradient through the direction itself. The
          directional box decides the sign with ``delta_g > eps`` inside ``torch.where``,
          which is a step function and passes no gradient to :math:`\ln c` at all.

        Saturation is the honest limit: past roughly :math:`|\Delta_rG|>10RT` the derivative
        underflows and a reaction that far from equilibrium is effectively a sign again.
        """
        return -torch.tanh(delta_g / (2.0 * self.rt))

    def _fluxforce_flux(
        self,
        context: torch.Tensor,
        lb: torch.Tensor,
        ub: torch.Tensor,
        delta_g: torch.Tensor,
        dg_mask: torch.Tensor,
        draw: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""The pre-projection flux of the ``fluxforce`` channel, and its priced mask.

        .. math::
            v_j = m_j\,f(\Delta_rG_j),\qquad
            m_j = \max(|v^\ell_j|,|v^u_j|)\,\sigma(z_j)\ \ge 0

        on every reaction the thermodynamic table prices and Box 2 does not exempt, and the
        ordinary sigmoid box :math:`v_j=v^\ell_j+(v^u_j-v^\ell_j)\sigma(z_j)` on the rest,
        which is what an unpriced reaction has always been. The magnitude comes from the
        GENOTYPE box, so availability, the deletion and the enzyme-capacity bound all still
        bite exactly: :math:`c_j=0` gives :math:`v^\ell_j=v^u_j=0`, hence :math:`m_j=0` and a
        flux of exactly zero.

        The clip that follows is not decoration. An irreversible reaction with
        :math:`v^\ell_j=0` whose energy forbids the forward direction gets
        :math:`m_jf<0`, and clipping it to zero is the physically right answer: the reaction
        does not run. Only the box can say that, because :math:`f` is never exactly zero.

        Returns:
            ``(v_pre, priced)`` with ``v_pre`` ``[B, r]`` inside the box and satisfying the
            second law, and ``priced`` the ``[1, r]`` mask the force law was applied on.
        """
        b = context.shape[0]
        rxn_emb = self.reaction_embedding.weight.unsqueeze(0).expand(b, -1, -1)
        ctx = context.unsqueeze(1).expand(-1, self.n_reactions, -1)
        features = torch.cat([ctx, rxn_emb], dim=-1)
        z = self.flux_mlp(features)[..., 0]
        if draw:
            std = torch.exp(self.flux_log_std(features).squeeze(-1))
            z = z + torch.randn_like(z) * std
        fraction = torch.sigmoid(z)
        priced = (dg_mask & (~self.thermo_exempt)).unsqueeze(0)
        magnitude = torch.maximum(lb.abs(), ub.abs()) * fraction
        v_force = magnitude * self.flux_force_efficacy(delta_g)
        v_box = lb + (ub - lb) * fraction
        v_pre = torch.where(priced, v_force, v_box)
        return torch.minimum(torch.maximum(v_pre, lb), ub), priced

    def warm_alpha(self) -> torch.Tensor:
        r"""The residual head's one learned scalar, :math:`\alpha=e^{\log\alpha}>0`."""
        return torch.exp(self.warm_log_alpha)

    def _warmstart_flux(
        self, context: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor, draw: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        r"""The pre-projection flux of the ``warmstart`` channel, and its reference.

        .. math::
            v_{\mathrm{ref}}^{(g)}=\mathrm{clip}(v_{\mathrm{ref}},v^\ell_g,v^u_g),\qquad
            v_{\mathrm{pre}} = v_{\mathrm{ref}}^{(g)}
                + \alpha\,\lVert v_{\mathrm{ref}}^{(g)}\rVert\,
                  \frac{\mathcal N z}{\lVert\mathcal N z\rVert}.

        THREE PROPERTIES, AND EACH IS THE REASON FOR ONE LINE.

        **The genotype still bites exactly.** The reference is clipped into the box the
        genotype chain built, so a deleted gene's reaction has :math:`v^\ell=v^u=0` and its
        reference flux is exactly zero, the same hard zero every other channel gets. Nothing
        about the warm start is a prior the deletion has to argue with.

        **The residual is already balanced.** :math:`\mathcal Nz\in\ker S` and the reference
        satisfies :math:`Sv=0` up to the LP's tolerance, so the only thing that breaks mass
        balance before the projection runs is the CLIP, and at small :math:`\alpha` the clip
        is small. That is the whole mechanism.

        **Alpha means something.** Normalizing the direction before scaling makes
        :math:`\lVert v_{\mathrm{pre}}-v_{\mathrm{ref}}\rVert/\lVert v_{\mathrm{ref}}\rVert
        =\alpha` identically, so the one learned scalar is on the same axis as the
        ``rel`` of ``probe_warmstart.json`` and the trained-proposal distance of
        ``probe_trained_proposal_distance.py``. A magnitude folded into an unnormalized
        :math:`z` would be readable only through the norm of an MLP output.

        Returns:
            ``(v_pre, kl, latent_std, v_ref_clipped)``, the first ``[B, r]`` inside the box.
        """
        latent, kl, latent_std = self._latent(context, force_sample=draw)
        reference = torch.minimum(
            torch.maximum(self.v_reference.unsqueeze(0).expand_as(lb), lb), ub
        )
        direction = latent @ self.null_basis.t()
        direction = direction / direction.norm(dim=1, keepdim=True).clamp(min=1e-12)
        step = self.warm_alpha() * reference.norm(dim=1, keepdim=True)
        v_pre = reference + step * direction
        return (torch.minimum(torch.maximum(v_pre, lb), ub), kl, latent_std, reference)

    def _latent(
        self, context: torch.Tensor, force_sample: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        r"""The latent :math:`z`, its KL against :math:`\mathcal N(0,I)`, and its sd.

        Deterministic unless ``latent_stochastic``, and then :math:`z=\mu+\sigma\epsilon` in
        training or whenever ``force_sample`` is set, which is how :meth:`sample` draws in
        eval mode without changing the module's training flag.

        The KL is the per-sample MEAN over latent dimensions rather than the sum, so one
        weight serves a 1,538-dimensional yeast basis and a three-dimensional toy alike.

        The third return is the posterior sd itself, or ``None`` on a deterministic layer.
        It is returned rather than recomputed by the caller because a K-draw arm logs it
        every epoch: measured on the one 027 cell that ever trained it
        ([[experiments.027-betaxanthin-metabolic.wave8-sampling]]), the objective shrank the
        median sigma by 130x over 80 epochs while the KL, at an effective weight of 1e-4,
        did not resist, so an arm that samples without watching this number cannot tell a
        sampler from a point estimate.
        """
        mu = self.latent_mlp(context)
        if not self.latent_stochastic:
            return mu, mu.new_zeros(()), None
        log_std = self.latent_log_std(context)
        std = torch.exp(log_std)
        z = mu + torch.randn_like(mu) * std if (self.training or force_sample) else mu
        kl = 0.5 * (mu**2 + std**2 - 1.0 - 2.0 * log_std).mean(dim=1)
        return z, kl.mean(), std

    def _transport_term(self, gem: GemTensors) -> torch.Tensor:
        r"""Supply :math:`\Delta_r G^t_j`, from the GEM's own column or from the physics.

        Summing formation energies over a transport reaction cancels exactly, because the
        same chemical appears on both sides: :math:`\sum_i S_{ij}\Delta_f G'^{\circ}_i=0`
        whenever every compartment is priced at one pH. A transport reaction's entire
        standard energy IS its transport term, so leaving this at zero asserts that moving
        a proton across a two-unit pH gradient is free.

        ``transport_delta_g_source`` picks the route, and the default ``off`` is what every
        arm in
        [[experiments.026-metabolism-flux.enzyme-constrained-thermodynamic-flux-layer]] ran.

        ``shipped``
            Take the GEM's reaction column on exactly the reactions whose metabolite-sum
            energy is structurally zero. Of the 3,204 reactions carrying both values, 441
            have a recomputed energy of exactly 0 against a shipped value above 5, and all
            441 cross a compartment boundary; relaxing the floor to any nonzero value gives
            874 of the 1,099 degenerate reactions. **Read the units before using this on
            yeast-GEM 9**: that column is yETFL's, in kcal/mol, mislabeled kJ/mol
            (``results/thermo_yeast9_provenance.json``, slope 4.187 against our kJ/mol
            formation energies over 1,927 metabolites).

        ``computed``
            Thermo-Flux Eq. (7), :math:`-N_HRT\ln(10)\Delta\mathrm{pH}-Fq\Delta\Phi`,
            evaluated per reaction by :mod:`torchcell.metabolism.transport` from the
            stoichiometry, the compartment of each metabolite and its charge, and read back
            from ``transport_terms_path``. Nothing is recovered from any shipped column, so
            it carries to a GEM that ships no thermodynamics at all. Requires the path:
            silently reverting to the shipped column would make a run's own record wrong.
        """
        transport = torch.zeros(gem.n_reactions)
        source = self.config.transport_delta_g_source
        if source == "off" and self.config.use_shipped_transport_delta_g:
            source = "shipped"
        if source == "off":
            return transport
        if source == "shipped":
            if gem.thermo is None:
                raise ValueError(
                    "transport_delta_g_source='shipped' needs the GEM's own thermodynamic "
                    "table, and this GEM carries none."
                )
            recomputed, recomputed_mask = gem.standard_reaction_delta_g()
            degenerate = recomputed_mask & (recomputed.abs() < 1e-6)
            usable = degenerate & gem.thermo.rxn_mask
            transport[usable] = gem.thermo.rxn_delta_g[usable]
            self.n_transport_terms_sourced = int(usable.sum())
            return transport
        if source == "computed":
            if self.config.transport_terms_path is None:
                raise ValueError(
                    "transport_delta_g_source='computed' needs transport_terms_path; "
                    "build it with torchcell.metabolism.transport.write_transport_table."
                )
            from torchcell.metabolism.transport import TransportTable

            table = TransportTable.read(self.config.transport_terms_path)
            values, mask = table.as_tensor(gem.rxn_ids)
            transport = values
            self.n_transport_terms_sourced = int((mask & (values.abs() > 1e-9)).sum())
            return transport
        raise ValueError(
            f"transport_delta_g_source={source!r} is not one of 'off', 'shipped', "
            "'computed'."
        )

    # -- forward --------------------------------------------------------------

    def gene_availability(
        self,
        h_genes: torch.Tensor,
        perturbation_indices: torch.Tensor,
        perturbation_indices_batch: torch.Tensor,
    ) -> torch.Tensor:
        r"""Per-sample :math:`\gamma_g\in[0,1]` over the GEM's genes. ``[B, n_gem_genes]``.

        Deleted genes are set to zero **hard**, from the perturbation. Everything else is
        a learned sigmoid gate, which is what leaves room for dosage, hypomorphic alleles
        and over-expression later without changing this signature.

        Constitutive genes are pinned to one. A heterologous cassette gene has no gene
        token, so the token-universe mask would otherwise zero it, and zeroing it switches
        off the pathway it was added to provide in every sample.
        """
        valid = self.gem_to_model >= 0
        token_idx = self.gem_to_model.clamp(min=0)
        h_gem = h_genes[:, token_idx, :]  # [B, n_gem_genes, d]
        if self.config.use_abundance_head:
            # No learned gate: availability is a hard fact (present or deleted) and every
            # graded effect goes through the abundance instead.
            gamma = valid.float().unsqueeze(0).expand(h_genes.shape[0], -1).clone()
        else:
            gamma = torch.sigmoid(self.availability(h_gem)).squeeze(-1)
            gamma = gamma * valid.unsqueeze(0)
        if self.n_constitutive_genes:
            gamma = torch.where(
                self.constitutive.unsqueeze(0), torch.ones_like(gamma), gamma
            )

        # Hard-zero the deleted genes that the GEM knows about.
        gem_of_pert = self.model_to_gem[perturbation_indices]
        in_gem = gem_of_pert >= 0
        if in_gem.any():
            rows = perturbation_indices_batch[in_gem]
            cols = gem_of_pert[in_gem]
            keep = torch.ones_like(gamma)
            keep[rows, cols] = 0.0
            gamma = gamma * keep
        return gamma

    def unit_gene_count(self) -> torch.Tensor:
        r"""``[n_units]`` how many genes each catalytic unit has, cached on first use.

        A plain attribute rather than a registered buffer, deliberately: it is a function of
        the GPR the layer was already built from, and registering it would add a key to the
        ``state_dict`` of every arm already trained for a tensor those arms do not read.
        """
        if self._unit_gene_count is None:
            self._unit_gene_count = torch.bincount(
                self.unit_gene_unit, minlength=self.n_units
            ).to(self.lb.dtype)
        return self._unit_gene_count

    def reaction_availability(self, gamma: torch.Tensor) -> torch.Tensor:
        r"""Fold :math:`\gamma_g` up the GPR: softmin over complexes, sum over isozymes.

        .. math::
            c_u=-\tfrac1\beta\log\!\sum_{g\in C_u}e^{-\beta\gamma_g},\qquad
            c_j=\min\Big(1,\ \sum_{u:\,\rho(u)=j}c_u\Big)

        Reactions with no GPR keep availability 1: an unannotated reaction is not one that
        no gene catalyzes, it is one nobody has assigned a gene to, and setting it to zero
        would let a missing annotation delete a reaction.

        **The softmin does not return 1 on an intact complex**, and that is what
        ``availability_fold='normalized_softmin'`` corrects. Evaluated with every gate at 1
        the expression above is :math:`1-\log(n_u)/\beta`, so a complex is charged for
        having subunits: at the shipped :math:`\beta=8` a 17-subunit complex reads 0.646 on
        a wild type. Dividing by that same all-ones value makes the fold exact there,
        leaves the ordering and the gradient alone because the divisor is a positive
        per-unit constant, and keeps a deleted subunit at zero: one gate at 0 sends the raw
        softmin non-positive, which the clamp below already catches.
        """
        beta = self.config.softmin_beta
        b = gamma.shape[0]
        gathered = gamma[:, self.unit_gene_gene]  # [B, nnz]
        acc = gamma.new_zeros(b, self.n_units)
        acc.index_add_(1, self.unit_gene_unit, torch.exp(-beta * gathered))
        # Units with no genes would give log(0); the parser never emits them, but guard.
        c_u = -torch.log(acc.clamp(min=1e-30)) / beta
        if self.config.availability_fold == "normalized_softmin":
            # The all-ones value, 1 - log(n_u)/beta, which is what the softmin returns on an
            # intact complex. It is positive for every unit with fewer than exp(beta) genes,
            # which at the shipped beta is 2,981 against a largest unit of a few dozen; the
            # clamp is there so a pathological GPR cannot divide by zero rather than because
            # any real one does.
            denominator = (
                1.0 - torch.log(self.unit_gene_count().clamp(min=1.0)) / beta
            ).clamp(min=1e-6)
            c_u = c_u / denominator.unsqueeze(0)
        elif self.config.availability_fold != "softmin":
            raise ValueError(
                f"availability_fold={self.config.availability_fold!r} is not 'softmin' or "
                "'normalized_softmin'."
            )
        c_j = gamma.new_zeros(b, self.n_reactions)
        c_j.index_add_(1, self.unit_reaction, c_u.clamp(min=0.0))
        c_j = c_j.clamp(max=1.0)
        return torch.where(self.has_gpr.unsqueeze(0), c_j, torch.ones_like(c_j))

    def gene_abundance(
        self, h_genes: torch.Tensor, gamma: torch.Tensor
    ) -> torch.Tensor:
        r"""Per-gene enzyme abundance :math:`E_g` in mmol/gDW. ``[B, n_gem_genes]``.

        .. math::
            E_g = \gamma_g\, E^0_g \exp\big(r\,\tanh(w^{\!\top}h_g)\big)

        :math:`\gamma_g` here is the hard availability from :meth:`gene_availability`,
        so a deleted gene has exactly zero abundance and a constitutive gene sits at its
        prior. The tanh bounds the head's reach to :math:`e^{\pm r}` of the prior; the
        pool budget in :meth:`forward` is what bounds the total.
        """
        token_idx = self.gem_to_model.clamp(min=0)
        h_gem = h_genes[:, token_idx, :]
        dev = self.config.abundance_log_range * torch.tanh(
            self.abundance(h_gem).squeeze(-1)
        )
        if self.n_constitutive_genes:
            dev = torch.where(
                self.constitutive.unsqueeze(0), torch.zeros_like(dev), dev
            )
        return gamma * torch.exp(self.log_e0.unsqueeze(0) + dev)

    def unit_abundance(self, e_g: torch.Tensor) -> torch.Tensor:
        r"""Fold gene abundance up the GPR to the catalytic unit. ``[B, n_units]``.

        A complex is limited by its scarcest subunit, so the unit abundance is a smooth
        minimum over its genes, the power mean with exponent :math:`-\beta`:

        .. math::
            E_u = \Big(\sum_{g\in C_u} E_g^{-\beta}\Big)^{-1/\beta}

        Scale-free, so the same :math:`\beta` serves abundances spanning decades, and a
        subunit at zero sends the unit to zero, which is the deletion propagating.
        """
        beta = self.config.softmin_beta
        log_e = torch.log(e_g.clamp(min=1e-30))
        gathered = -beta * log_e[:, self.unit_gene_gene]
        acc = e_g.new_full((e_g.shape[0], self.n_units), -float("inf"))
        # logsumexp by scatter: acc_u = log sum_g exp(-beta log E_g)
        acc = torch.scatter_reduce(
            acc,
            1,
            self.unit_gene_unit.unsqueeze(0).expand(e_g.shape[0], -1),
            gathered,
            reduce="amax",
            include_self=True,
        )
        shifted = torch.exp(gathered - acc[:, self.unit_gene_unit])
        total = e_g.new_zeros(e_g.shape[0], self.n_units)
        total.index_add_(1, self.unit_gene_unit, shifted)
        log_pm = -(acc + torch.log(total.clamp(min=1e-30))) / beta
        return torch.exp(log_pm)

    def dynamic_box(
        self, c_j: torch.Tensor, e_u: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        r"""The genotype-dependent flux box :math:`[v^\ell_j,v^u_j]` and its capacity.

        Availability scales the box multiplicatively, so :math:`c_j=0` collapses it to the
        single point :math:`0`. Enzyme capacity enters here too rather than as a penalty:

        .. math::
            \bar v^u_j=\min\Big(v^u_j,\ \sum_{u:\rho(u)=j}k_{\mathrm{cat},u}E_u\Big)

        with :math:`E_u` the predicted unit abundance when the abundance head is on, and
        the nominal :math:`c_j\bar E` otherwise. Putting capacity in the box rather than
        the objective is what makes it **exact** at no cost, and it is why there is no
        ``C_box`` term in the loss. Returns the capacity too, so the caller can report
        how often the bound is the thing that limits the flux.
        """
        lb = self.lb.unsqueeze(0) * c_j
        ub = self.ub.unsqueeze(0) * c_j
        if not self.config.use_enzyme_capacity:
            return lb, ub, None
        cap = c_j.new_zeros(c_j.shape[0], self.n_reactions)
        if e_u is not None:
            per_unit = self.kcat_per_h.unsqueeze(0) * e_u
            cap.index_add_(1, self.unit_reaction, per_unit)
        else:
            # Enzyme available per unit is not predicted; the capacity ceiling uses a
            # nominal abundance so that kcat sets the SCALE of the bound. This is where a
            # measured or predicted proteome enters when use_abundance_head is set.
            per_unit = self.kcat_per_h * 1.0e-3
            cap.index_add_(
                1, self.unit_reaction, per_unit.unsqueeze(0).expand(c_j.shape[0], -1)
            )
            cap = cap * c_j
        capped = torch.where(self.has_gpr.unsqueeze(0), cap, ub.abs())
        ub = torch.minimum(ub, capped)
        lb = torch.maximum(lb, -capped)
        return lb, ub, capped

    def reaction_potential(
        self, context: torch.Tensor, latent: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        r"""Per-reaction :math:`\Delta_r G_j` and the mask of reactions it is valid on.

        In ``ANCHORED`` mode this is Thermo-Flux Eq. (4) term by term. In ``FREE`` mode it
        is :math:`\sum_i S_{ij}\mu_i` with :math:`\mu` unconstrained. In ``OFF`` mode the
        mask is empty and the caller skips the term.

        ``latent`` is the drawn :math:`z`, ``[B, n_latent]`` on the same rows as
        ``context``, and it is REQUIRED exactly when ``log_c_from_latent`` is set: the
        anchored concentration then reads
        :math:`\log c=lo+(hi-lo)\,\sigma(\text{base}+W\,\text{context}+U z)`, so a caller
        that passes ``B*K`` expanded rows gets one concentration per draw. Passing it on a
        layer without the flag, or omitting it on one with the flag, is asserted against
        rather than silently ignored, because either way the concentration would not be
        the one the flag names.
        """
        b = context.shape[0]
        extras: dict[str, torch.Tensor] = {}
        assert (latent is not None) == self.log_c_from_latent_enabled, (
            f"reaction_potential got latent={'a tensor' if latent is not None else None} "
            f"on a layer with log_c_from_latent={self.log_c_from_latent_enabled}; the "
            "drawn latent is read exactly when the flag is set."
        )
        if self.thermo_mode is ThermoMode.OFF:
            zeros = context.new_zeros(b, self.n_reactions)
            return zeros, torch.zeros_like(self.delta_r_g0_mask), extras

        if self.thermo_mode is ThermoMode.FREE:
            mu = self.mu_base.unsqueeze(0) + self.mu_from_context(context)
            delta = self._s_transpose_matmul(mu)
            mask = torch.ones_like(self.delta_r_g0_mask)
            extras["mu"] = mu
            return delta, mask, extras

        # ANCHORED
        lo, hi = (
            math.log(self.config.conc_lower_m),
            math.log(self.config.conc_upper_m),
        )
        raw = self.log_c_base.unsqueeze(0) + self.log_c_from_context(context)
        if latent is not None:
            raw = raw + self.log_c_from_latent(latent)
        log_c = lo + (hi - lo) * torch.sigmoid(raw)
        conc_term = self.rt * self._s_transpose_matmul(log_c)
        offset = (
            self.delta_g_offset
            if self.thermo_uncertainty == "isotropic"
            else self.delta_g_q @ self.delta_g_m
        )
        delta = (
            self.delta_r_g0.unsqueeze(0)
            + conc_term
            + self.delta_r_g_transport.unsqueeze(0)
            + offset.unsqueeze(0)
        )
        extras["log_c"] = log_c
        extras["conc_term"] = conc_term
        extras["delta_g_offset"] = offset
        return delta, self.delta_r_g0_mask, extras

    def _s_transpose_matmul(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, m] -> [B, r]`` via ``S^T x``, using the sparse structure directly."""
        s = torch.sparse_coo_tensor(
            self.s_indices, self.s_values, (self.n_metabolites, self.n_reactions)
        )
        out: torch.Tensor = torch.sparse.mm(s.t(), x.t()).t()
        return out

    def _s_matmul(self, v: torch.Tensor) -> torch.Tensor:
        """``[B, r] -> [B, m]`` via ``S v``."""
        s = torch.sparse_coo_tensor(
            self.s_indices, self.s_values, (self.n_metabolites, self.n_reactions)
        )
        out: torch.Tensor = torch.sparse.mm(s, v.t()).t()
        return out

    def turnover(self, v: torch.Tensor) -> torch.Tensor:
        r""":math:`\omega_i(v)=\tfrac12\sum_j|S_{ij}v_j|`, the throughput of metabolite i.

        Doubles as the mass-balance normalizer and as the precursor-pool readout. A
        residual of 0.01 is negligible on a metabolite carrying flux 10 and nonsense on one
        carrying 0.01, which is why the balance penalty is a ratio rather than a square.
        """
        s_abs = torch.sparse_coo_tensor(
            self.s_indices, self.s_values.abs(), (self.n_metabolites, self.n_reactions)
        )
        out: torch.Tensor = 0.5 * torch.sparse.mm(s_abs, v.abs().t()).t()
        return out

    def forward(
        self,
        h_genes: torch.Tensor,
        context: torch.Tensor,
        perturbation_indices: torch.Tensor,
        perturbation_indices_batch: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Predict a flux vector and every feasibility residual for it.

        Args:
            h_genes: ``[B, N, d]`` perturbed gene token representations.
            context: ``[B, d]`` whole-cell context (the perturbed-gene pool).
            perturbation_indices: Flat ``[total_pert]`` gene-token indices.
            perturbation_indices_batch: ``[total_pert]`` sample assignment.

        Returns:
            A dict with ``v`` ``[B, r]``, every constraint residual as a scalar, and the
            per-sample feasibility diagnostics. Residuals are returned rather than summed
            so the caller owns the weighting and every term can be logged separately.

            Under ``draws = K > 1`` every batch-axis entry carries ``B*K`` rows, grouped
            by genotype. The ``thermo_*`` extras of the anchored mode (``thermo_log_c``
            ``[., m]``, ``thermo_conc_term`` ``[., r]``, and ``thermo_delta_g_offset``
            ``[r]`` with no batch axis) are the one exception and depend on the flag:
            with ``log_c_from_latent`` OFF they are a function of the context alone and
            stay at ``B`` rows; with it ON they are computed after the draw and carry
            ``B*K`` rows, one concentration per draw, and so do ``delta_r_g`` and
            ``g_diss_per_sample``.
        """
        cfg = self.config
        b = h_genes.shape[0]
        gamma = self.gene_availability(
            h_genes, perturbation_indices, perturbation_indices_batch
        )
        c_j = self.reaction_availability(gamma)
        e_g: torch.Tensor | None = None
        e_u: torch.Tensor | None = None
        if cfg.use_abundance_head:
            e_g = self.gene_abundance(h_genes, gamma)
            e_u = self.unit_abundance(e_g)
        lb, ub, capacity = self.dynamic_box(c_j, e_u)

        # THE REACTION ENERGIES ARE NEEDED BEFORE THE FLUX when the second law is a box
        # rather than a penalty, so they are computed here for every arm and reused below.
        # Nothing about this call is stateful or random, so moving it earlier leaves every
        # existing arm's numbers and its consumption of the RNG exactly as they were.
        #
        # UNLESS THE CONCENTRATION READS THE DRAW. With `log_c_from_latent` the energies
        # are a function of z, which does not exist yet, so they are computed once per
        # draw after the expansion below, and the directional box with them.
        thermo_emptied: torch.Tensor | None = None
        delta_g: torch.Tensor
        dg_mask: torch.Tensor
        extras: dict[str, torch.Tensor]
        if not self.log_c_from_latent_enabled:
            delta_g, dg_mask, extras = self.reaction_potential(context)
            if self.thermo_box:
                lb, ub, thermo_emptied = self._thermo_tighten(lb, ub, delta_g)

        # K DRAWS PER GENOTYPE, as an expansion of the batch rather than a loop.
        #
        # Everything above this line is a deterministic function of the tokens, so it is
        # computed once and repeated; everything below it then runs on B*K rows with no
        # change of its own. That is what makes the constraint residuals the mean over draws
        # automatically: they are already means over the batch axis, and the draws are now
        # part of it. `repeat_interleave` groups the draws of one genotype contiguously, so
        # a consumer reshapes to [B, K, ...] with a plain view.
        #
        # EVERY BATCH-AXIS ENTRY OF THE RETURNED DICT CARRIES B*K ROWS, including the ones
        # the draw cannot change. `c_j`, `gamma` and `e_g` are genotype quantities and
        # repeating them stores identical rows, but a consumer that pairs one of them with
        # the flux needs them to line up: `_per_sample_feasibility` in 027's trainer rebuilds
        # the box from `c_j` and compares it against `v`, and at B rows against B*K that is
        # a shape error at the end of a run rather than at its start. Every mean over the
        # expanded rows equals the mean over the original ones, so no reported number moves.
        #
        # The `thermo_*` extras are the exception and stay at B rows when the
        # concentration reads the context alone: no draw touches them, and the trainer
        # refuses the readout that consumes them with K > 1 unless `log_c_from_latent` is
        # set. With that flag they are computed below, after the draw, and carry B*K rows.
        draws = int(cfg.draws)
        if draws > 1:
            context = context.repeat_interleave(draws, dim=0)
            lb = lb.repeat_interleave(draws, dim=0)
            ub = ub.repeat_interleave(draws, dim=0)
            if not self.log_c_from_latent_enabled:
                delta_g = delta_g.repeat_interleave(draws, dim=0)
            c_j = c_j.repeat_interleave(draws, dim=0)
            gamma = gamma.repeat_interleave(draws, dim=0)
            if e_g is not None:
                e_g = e_g.repeat_interleave(draws, dim=0)
            if capacity is not None:
                capacity = capacity.repeat_interleave(draws, dim=0)
            if thermo_emptied is not None:
                thermo_emptied = thermo_emptied.repeat_interleave(draws, dim=0)
            # Every use of `b` below this point sizes a per-FLUX tensor, so it follows the
            # expansion. `gamma` and `e_g` keep their B rows and their own diagnostics with
            # them.
            b = context.shape[0]

        kl = h_genes.new_zeros(())
        latent_std: torch.Tensor | None = None
        latent: torch.Tensor | None = None
        if self.log_c_from_latent_enabled:
            # THE DRAW FIRST, THEN THE ENERGIES. The concentration is a function of z, so
            # z is drawn on the expanded rows before anything thermodynamic exists, and the
            # reaction energies, the directional box and every second-law term below then
            # see one delta_r_G per draw. The flux and the concentration share the draw, so
            # the per-draw hinge prices sign consistency between the drawn v and the drawn
            # log c rather than between a drawn v and a concentration that never moved.
            # The RNG is consumed here instead of inside the parameterization branch, and
            # nothing between the two points draws, so the sample is the same one.
            latent, kl, latent_std = self._latent(context, force_sample=draws > 1)
            delta_g, dg_mask, extras = self.reaction_potential(context, latent=latent)
            if self.thermo_box:
                lb, ub, thermo_emptied = self._thermo_tighten(lb, ub, delta_g)
        sign_flip_frac: torch.Tensor | None = None
        v_pre_projection: torch.Tensor | None = None
        warm_stats: dict[str, torch.Tensor] | None = None
        if self.parameterization == "warmstart":
            # THE RESIDUAL HEAD. Everything here is the boxforce recipe with one thing
            # changed: where the alternation starts. The box is tightened to the licensed
            # direction FIRST, exactly as `boxforce` does, so every clip in the loop is
            # already the directional one; then the reference flux is clipped into that box
            # and displaced by alpha of its own norm inside ker S.
            box_lb, box_ub = lb, ub
            if self.projection_end == "boxforce":
                box_lb, box_ub, thermo_emptied = self._thermo_tighten(lb, ub, delta_g)
            v_pre_projection, kl, latent_std, reference = self._warmstart_flux(
                context, box_lb, box_ub, draw=draws > 1
            )
            v = self._project_feasible(v_pre_projection, box_lb, box_ub)
            if self.projection_end == "force":
                lb_force, ub_force, thermo_emptied = self._thermo_tighten(
                    lb, ub, delta_g
                )
                v = torch.minimum(torch.maximum(v, lb_force), ub_force)
            # WHAT THE PROJECTION STILL DISCARDS, LOGGED EVERY EPOCH. This is the statistic
            # the whole arm rests on: 0.990 on the trained boxforce head of job 1784, and
            # 0.008 on the proposal whose projection clears the turnover rule at 0.0012. An
            # arm that does not watch it cannot tell a warm start from a cold one.
            pre_norm = v_pre_projection.norm(dim=1).clamp(min=1e-12)
            warm_stats = {
                "feas_warm_proposal_rel": (
                    (v - v_pre_projection).norm(dim=1) / pre_norm
                ).mean(),
                "feas_warm_alpha": self.warm_alpha().detach(),
                "feas_warm_reference_clipped_frac": (
                    (self.v_reference.unsqueeze(0) != reference).float().mean()
                ),
                "feas_warm_reference_active_frac": (
                    (reference.abs() > 1.0e-6 * cfg.flux_scale).float().mean()
                ),
            }
            out_of_box = torch.relu(v - ub) + torch.relu(lb - v)
            box_penalty = (out_of_box / cfg.flux_scale).mean()
        elif self.parameterization == "fluxforce":
            box_lb, box_ub = lb, ub
            if self.projection_end == "boxforce":
                # THE DIRECTION IN THE BOX FROM THE START, so the alternation converges
                # against the set the flux has to end in rather than being overwritten by a
                # clip after it. Measured untrained on the real GEM, balance ratio median
                # after the projection: 0.193 at K = 20, 0.0218 at K = 50, 0.0000 at K = 100,
                # against the clip-last end's 0.403, 0.181 and 0.057, with the second law and
                # the box exact in both. The magnitude and the direction that reach the box
                # are still smooth functions of delta_r_G, so unlike `tdg` the concentrations
                # keep their gradient.
                box_lb, box_ub, thermo_emptied = self._thermo_tighten(lb, ub, delta_g)
            v_pre_projection, priced = self._fluxforce_flux(
                context,
                box_lb,
                box_ub,
                delta_g,
                dg_mask,
                draw=self.latent_stochastic and (self.training or draws > 1),
            )
            v = self._project_feasible(v_pre_projection, box_lb, box_ub)
            # WHAT THE PROJECTION COSTS, MEASURED. The pre-projection flux satisfies the
            # second law by construction; the balance projection is free to move a reaction
            # across zero, and when it does, that reaction now runs uphill. It is the one
            # second-law violation this channel can produce, so it is counted rather than
            # left to be inferred from the thermo fraction. Measured on job 1745 at seed
            # 2700: the flip fraction 0.2846 and the top-100 model violation median 0.2851
            # agree, which is what they must do if every violation is a flip.
            carrying = (v_pre_projection.abs() > 1.0e-6) & priced
            flipped = (v * v_pre_projection < 0) & carrying
            sign_flip_frac = flipped.float().sum() / carrying.float().sum().clamp(
                min=1.0
            )
            if self.projection_end == "force":
                # THE DIRECTION RESTORED LAST. The alternation ended on the balance
                # projection, so this clip is what the flux leaves through: the box
                # tightened to the sign the reaction energy licenses. Bounds and the second
                # law are then both exact and the balance residual is the reported leftover,
                # which is the opposite trade from the box end. It is NOT the `tdg` head:
                # the direction and the magnitude that enter this clip are already smooth
                # functions of delta_r_G, so the learned concentrations keep the gradient
                # path the sign clamp never had.
                lb_force, ub_force, thermo_emptied = self._thermo_tighten(
                    lb, ub, delta_g
                )
                v = torch.minimum(torch.maximum(v, lb_force), ub_force)
            out_of_box = torch.relu(v - ub) + torch.relu(lb - v)
            box_penalty = (out_of_box / cfg.flux_scale).mean()
        elif self.parameterization in {"nullspace", "projection"}:
            # Mass balance is exact here, for any latent: S (N z) = (S N) z = 0.
            #
            # A K-draw arm forces the draw in eval as well as in training, because its
            # prediction IS the average over draws and emitting mu K times would score a
            # different estimator than the one that was trained. At K = 1 the flag is False
            # and `_latent` behaves exactly as it always has. Under `log_c_from_latent`
            # the draw was already taken above, before the energies, and is reused here.
            if latent is None:
                latent, kl, latent_std = self._latent(context, force_sample=draws > 1)
            z_latent = latent * cfg.flux_scale
            v = z_latent @ self.null_basis.t()
            if self.parameterization == "projection":
                # ... and then the box is made exact too, alternately, so that whichever
                # end the config names holds to machine precision and the other residual
                # is a measured number rather than an unbounded penalty.
                v = self._project_feasible(v, lb, ub)
            # The box is what is soft now, and directionality with it.
            out_of_box = torch.relu(v - ub) + torch.relu(lb - v)
            box_penalty = (out_of_box / cfg.flux_scale).mean()
        else:
            rxn_emb = self.reaction_embedding.weight.unsqueeze(0).expand(b, -1, -1)
            ctx = context.unsqueeze(1).expand(-1, self.n_reactions, -1)
            z_out = self.flux_mlp(torch.cat([ctx, rxn_emb], dim=-1))
            if cfg.stochastic:
                mean, log_std = z_out[..., 0], z_out[..., 1].clamp(-6.0, 2.0)
                z = (
                    mean + torch.randn_like(mean) * log_std.exp()
                    if self.training
                    else mean
                )
            else:
                z = z_out[..., 0]
            # The box, exactly. Nothing downstream may violate it.
            v = lb + (ub - lb) * torch.sigmoid(z)
            box_penalty = v.new_zeros(())

        out: dict[str, torch.Tensor] = {"v": v, "c_j": c_j, "gamma": gamma}
        out["c_box"] = box_penalty
        out["c_kl"] = kl
        # THE NUMBER A SAMPLED ARM LIVES OR DIES BY, logged every epoch rather than
        # reconstructed from the KL afterwards. `c_kl` mixes the mu and the sigma halves, so
        # a KL that rises does not by itself say which one moved.
        out["latent_n_draws"] = v.new_tensor(float(draws))
        if latent_std is not None:
            out["latent_sigma_p50"] = latent_std.detach().median()
            out["latent_sigma_mean"] = latent_std.detach().mean()
        # The box, carried out so `feasibility` can price a violation per sample without
        # rebuilding the genotype chain. Two-dimensional, so the trainer's scalar log, which
        # takes only 0-dim entries, does not pick them up.
        out["lb"] = lb
        out["ub"] = ub
        if v_pre_projection is not None:
            # Two-dimensional, so the trainer's scalar log leaves it alone; an audit that
            # wants to price the projection's own cost reads it.
            out["v_pre_projection"] = v_pre_projection
        if warm_stats is not None:
            out.update(warm_stats)
        if sign_flip_frac is not None:
            out["feas_sign_flip_frac"] = sign_flip_frac
        out["feas_box_violation_frac"] = (
            ((v > ub + 1e-4) | (v < lb - 1e-4)).float().mean()
        )
        if capacity is not None:
            # How often the enzyme bound is the thing limiting a flux. A capacity that is
            # never active is a uniform rescaling of the box, not an enzyme constraint, and
            # the loss curve cannot tell the two apart; this can.
            gpr = self.has_gpr.unsqueeze(0).expand_as(v)
            at_bound = (v.abs() >= 0.95 * capacity) & gpr & (capacity > 0)
            out["feas_capacity_active_frac"] = (
                at_bound.float().sum() / gpr.float().sum().clamp(min=1)
            )
            out["capacity_median"] = capacity[gpr].median()
        if e_g is not None:
            out["e_g"] = e_g
            out["abundance_log_dev_mean"] = (
                (torch.log(e_g.clamp(min=1e-30)) - self.log_e0.unsqueeze(0))[gamma > 0]
                .abs()
                .mean()
            )

        # (a) mass balance, scale-relative, on independent rows only
        residual = self._s_matmul(v)[:, self.independent_rows]
        omega = self.turnover(v)[:, self.independent_rows]
        ratio = residual / (omega.detach() + cfg.balance_epsilon)
        out["c_balance"] = (ratio**2).mean()
        out["feas_balance_median"] = (
            residual.abs() / (omega + cfg.balance_epsilon)
        ).median()

        # (a2) TURNOVER DENSITY. A balance statistic over a mostly-zero flux is a statement
        # about the zeros, so a head that can empty the network has a degenerate way to
        # satisfy it. This term prices exactly that: a row whose turnover reaches the
        # threshold costs nothing, a row at zero costs one, and the cost is linear in
        # between, so the gradient pushes throughput onto dead rows rather than pushing every
        # flux up. It is reported for every arm and weighted only when asked.
        active_turnover = cfg.density_turnover_threshold * cfg.flux_scale
        out["c_density"] = torch.relu(1.0 - omega / active_turnover).mean()
        out["feas_active_row_frac"] = (omega > active_turnover).float().mean()
        out["feas_flux_active_frac"] = (
            (v.abs() > 1.0e-6 * cfg.flux_scale).float().mean()
        )

        # (a3) BALANCE ON THE ROWS THAT CARRY FLUX. `c_balance` above averages the squared
        # relative residual over every independent row, so idle rows, which pass for free,
        # dominate it; and at a constraint scale of 0.0 it is multiplied away entirely. This
        # term is the same ratio restricted to the rows the thresholded rule is evaluated
        # on, and `constraint_loss` adds it outside the ramp. `omega` is detached in the
        # denominator exactly as it is in `c_balance`, so the gradient reduces the RESIDUAL
        # rather than inflating the turnover under it.
        above_threshold = omega > active_turnover
        if cfg.balance_active_row_set == "frozen":
            if not self.frozen_active_rows_ready:
                # One batch at initialization decides the set, and it is held from there:
                # a row the head later kills stays in the penalty at a ratio of 2, so
                # emptying it costs instead of paying.
                with torch.no_grad():
                    self.frozen_active_rows = above_threshold.any(dim=0)
                self.frozen_active_rows_ready = True
            active_rows = self.frozen_active_rows.unsqueeze(0).expand_as(
                above_threshold
            )
        else:
            active_rows = above_threshold
        active_count = active_rows.sum(dim=1).clamp(min=1)
        active_sq = (ratio**2) * active_rows
        out["c_balance_active"] = (active_sq.sum(dim=1) / active_count).mean()
        active_ratio = (residual.abs() / (omega + cfg.balance_epsilon))[active_rows]
        if active_ratio.numel() > 0:
            out["feas_balance_active_max"] = active_ratio.max()
            out["feas_balance_active_p95"] = _linear_quantile(active_ratio, 0.95)
        else:
            out["feas_balance_active_max"] = v.new_zeros(())
            out["feas_balance_active_p95"] = v.new_zeros(())

        # (b) second law. A hinge, unless `thermo_box` already made it a bound, in which
        # case the same expression is a DIAGNOSTIC that should read ~0 and says so if not.
        out.update({f"thermo_{k}": t for k, t in extras.items()})
        if thermo_emptied is not None:
            out["feas_thermo_box_emptied_frac"] = thermo_emptied.float().mean()
        active = dg_mask & (~self.thermo_exempt)
        if active.any() and self.thermo_mode is not ThermoMode.OFF:
            va, ga = v[:, active], delta_g[:, active]
            drive = va * ga
            # NORMALIZED to a dimensionless fraction of the available driving force.
            # The raw hinge relu(v * dG) is in kJ mmol gDW^-1 h^-1 mol^-1 and, measured at
            # init on a real batch, is ~72 against a data loss of ~1 -- so an unnormalized
            # term does not regularize the model, it replaces it. Dividing by |v||dG|
            # makes each reaction contribute at most 1 and turns the term into "what
            # fraction of the driving force runs uphill", which is also the quantity worth
            # reporting.
            out["c_thermo"] = (
                torch.relu(drive + cfg.thermo_epsilon)
                / (va.abs() * ga.abs() + cfg.thermo_epsilon)
            ).mean()
            out["feas_thermo_violation_frac"] = (drive > 0).float().mean()
            out["delta_r_g"] = delta_g
        else:
            out["c_thermo"] = v.new_zeros(())
            out["feas_thermo_violation_frac"] = v.new_zeros(())

        # (c) Gibbs dissipation rate (Niebel 2019), summed over exchange reactions.
        #
        # ALWAYS COMPUTED UNDER ANCHORED THERMODYNAMICS, whether or not it is penalized.
        # Thermo-Flux takes the cellular Gibbs energy dissipation rate over the boundary
        # fluxes, g_diss = sum_{j in exchange} v_j delta_r_G_j, and Niebel et al. (2019)
        # report an upper limit near 3.7 kJ/gDW/h on it for S. cerevisiae at high growth
        # rate. Units here: mmol/gDW/h times kJ/mol is J/gDW/h, so that limit is 3,700 in
        # the numbers below, which is what `g_diss_limit` carries. SIGN CONVENTION, stated
        # because it is easy to read the wrong way: this is the SIGNED sum, so a state that
        # dissipates has a NEGATIVE value and the magnitude is the dissipation rate. The
        # optional penalty is `use_dissipation_limit`, off by default, and it prices the
        # signed excess above the limit exactly as every 026 and 027 arm measured it.
        if self.thermo_mode is ThermoMode.ANCHORED:
            g_diss_per_sample = (
                v[:, self.is_exchange] * delta_g[:, self.is_exchange]
            ).sum(dim=1)
            out["g_diss_per_sample"] = g_diss_per_sample
            out["thermo_g_diss_mean"] = g_diss_per_sample.mean()
            # The same number under the prefix 027's trainer filters its scalar log on
            # ('c_', 'feas_', 'g_diss', 'protein_used'), so an arm that does not penalize
            # the dissipation still records it every epoch.
            out["g_diss_mean"] = out["thermo_g_diss_mean"]
        if cfg.use_dissipation_limit and self.thermo_mode is ThermoMode.ANCHORED:
            g_diss = (v[:, self.is_exchange] * delta_g[:, self.is_exchange]).sum(dim=1)
            # kJ/mol * mmol/gDW/h = J/gDW/h, so no unit conversion is needed.
            out["g_diss"] = g_diss.mean()
            # A RATIO hinge, not a squared excess. At init the dissipation is ~2e5 times
            # the 3,700 J/gDW/h limit, so a squared excess divided by g_lim^2 evaluates to
            # ~4e4 and its gradient dwarfs every other term into NaN within one step.
            out["c_dissipation"] = torch.relu(g_diss / cfg.g_diss_limit - 1.0).mean()
        else:
            out["c_dissipation"] = v.new_zeros(())

        # (d) the protein budget
        if cfg.use_protein_budget and e_g is not None:
            # The pool constraint on the PREDICTED abundance, forward: sum_g MW_g E_g in
            # g/gDW against p_avail. This is what pins the scale of E, since without it
            # the head could inflate every abundance and make the capacity bound slack.
            protein = (e_g * self.mw_kda.unsqueeze(0)).sum(dim=1)
            out["protein_used"] = protein.mean()
            out["c_budget"] = torch.relu(protein / cfg.p_avail - 1.0).mean()
            out["feas_budget_ratio"] = (protein / cfg.p_avail).mean()
        elif cfg.use_protein_budget:
            # The legacy path: enzyme demand IMPLIED by the flux, priced against the
            # budget. The 1e-3 makes the mass 1,000x smaller than MW[g/mmol] x E[mmol/gDW]
            # says it is, so this term is a soft regularizer rather than the pool
            # constraint; it is kept as the arms were measured with it and is superseded
            # by the abundance-head path above.
            demand = torch.sqrt(v**2 + 1e-8)[:, self.unit_reaction] / self.kcat_per_h
            e_implied = v.new_zeros(b, self.n_gem_genes)
            e_implied.index_add_(1, self.unit_gene_gene, demand[:, self.unit_gene_unit])
            protein = (e_implied * self.mw_kda).sum(dim=1) * 1e-3
            out["protein_used"] = protein.mean()
            out["c_budget"] = torch.relu(protein / cfg.p_avail - 1.0).mean()
            out["feas_budget_ratio"] = (protein / cfg.p_avail).mean()
            out["e_g"] = e_implied
        else:
            out["c_budget"] = v.new_zeros(())

        # (e) parsimony: the main defence against a 1,538-dimensional null space
        out["c_parsimony"] = torch.sqrt(v**2 + 1e-8).mean() / cfg.flux_scale

        # (f) prior on the Thermo-Flux uncertainty latent. Under 'covariance' the prior is
        # N(0, I) on m itself, which is Thermo-Flux's own statement of it; under 'isotropic'
        # it is the same expression applied to the per-reaction offset, where it is a ridge
        # rather than a likelihood because the per-reaction errors are not independent.
        if self.thermo_mode is not ThermoMode.ANCHORED:
            out["c_thermo_prior"] = v.new_zeros(())
        elif self.thermo_uncertainty == "isotropic":
            out["c_thermo_prior"] = (self.delta_g_offset**2).mean()
        else:
            out["c_thermo_prior"] = (self.delta_g_m**2).mean()

        return out

    # -- feasibility, per sample ----------------------------------------------

    def _thermo_active(self) -> torch.Tensor:
        """Reactions the second law is applied to: priced, and not exempt."""
        return self.delta_r_g0_mask & (~self.thermo_exempt)

    def feasibility(
        self,
        out: dict[str, torch.Tensor],
        box_tol: float = 1.0e-4,
        flux_tol: float = 1.0e-6,
    ) -> dict[str, torch.Tensor]:
        r"""Per-sample feasibility, as ``[B]`` tensors. The canonical definitions.

        :meth:`forward` averages these into scalars for the training log; a filter over
        nominated strains needs them per genotype, because "95 percent of the top 100 do not
        violate mass balance" is a statement about individual strains and a batch mean can
        satisfy it while a third of the strains do not.

        Keys, and what each one is:

        ``balance_ratio_median`` / ``balance_ratio_max``
            :math:`|S_iv|/(\omega_i+\epsilon)` over the independent rows, where
            :math:`\omega` is the metabolite's turnover, so the residual is priced against
            the throughput it sits on rather than absolutely. The median is the typical
            metabolite and the max is the worst one; a filter should read the max.
            :meth:`forward`'s ``feas_balance_median`` is the median over the FLATTENED
            batch, which is not the mean of these, so the two are related but not equal.

        ``box_violation_frac``
            Fraction of reactions outside ``[lb, ub]`` by more than ``box_tol``, the same
            tolerance and definition ``feas_box_violation_frac`` uses.

        ``thermo_violation_frac``
            Fraction of second-law reactions with :math:`v_j\Delta_rG_j>0`, matching
            ``feas_thermo_violation_frac``. Counts a reaction at zero flux whose product is
            numerically positive.

        ``thermo_uphill_frac``
            The same, restricted to reactions actually CARRYING flux
            (:math:`|v_j|>` ``flux_tol``). This is the number a feasibility filter wants: a
            reaction at zero flux is not running uphill, it is not running.

        Zero-filled where the term does not apply, so the caller gets the same keys from
        every arm rather than having to test the config.
        """
        v = out["v"]
        lb, ub = out["lb"], out["ub"]
        eps = self.config.balance_epsilon
        residual = self._s_matmul(v)[:, self.independent_rows].abs()
        omega = self.turnover(v)[:, self.independent_rows]
        ratio = residual / (omega + eps)
        result = {
            "balance_ratio_median": ratio.median(dim=1).values,
            "balance_ratio_max": ratio.max(dim=1).values,
            "box_violation_frac": (
                ((v > ub + box_tol) | (v < lb - box_tol)).float().mean(dim=1)
            ),
        }
        active = self._thermo_active()
        if "delta_r_g" in out and bool(active.any()):
            drive = v[:, active] * out["delta_r_g"][:, active]
            carrying = v[:, active].abs() > flux_tol
            uphill = (drive > 0) & carrying
            result["thermo_violation_frac"] = (drive > 0).float().mean(dim=1)
            result["thermo_uphill_frac"] = uphill.float().sum(
                dim=1
            ) / carrying.float().sum(dim=1).clamp(min=1.0)
        else:
            result["thermo_violation_frac"] = v.new_zeros(v.shape[0])
            result["thermo_uphill_frac"] = v.new_zeros(v.shape[0])
        return result

    def sample(
        self,
        h_genes: torch.Tensor,
        context: torch.Tensor,
        perturbation_indices: torch.Tensor,
        perturbation_indices_batch: torch.Tensor,
        n_samples: int,
    ) -> FluxSample:
        """Draw ``n_samples`` fluxes per genotype and price each draw's own feasibility.

        The genotype chain, the box and the reaction energies are computed ONCE and shared
        across draws, because they are deterministic functions of the tokens: only the
        latent is resampled. The draw is taken whether or not the module is training, which
        is the point of calling this rather than :meth:`forward` in a loop.

        Requires a latent parameterization with ``latent_stochastic`` set. There is no
        deterministic fall back: a "sampler" that returned ``n_samples`` copies of one flux
        would report a feasibility fraction of 0 or 1 and look like a measurement.
        """
        if self.parameterization not in {
            "nullspace",
            "projection",
            "fluxforce",
            "warmstart",
        }:
            raise ValueError(
                f"sample() needs parameterization 'nullspace', 'projection', 'fluxforce' "
                f"or 'warmstart'; this layer is {self.parameterization!r}. The box head's "
                "draw is per reaction and carries no balance guarantee at all."
            )
        if not self.latent_stochastic:
            raise ValueError(
                "sample() needs latent_stochastic; this layer emits a single flux per "
                "genotype and repeating it is not a sample."
            )
        cfg = self.config
        gamma = self.gene_availability(
            h_genes, perturbation_indices, perturbation_indices_batch
        )
        c_j = self.reaction_availability(gamma)
        e_u = (
            self.unit_abundance(self.gene_abundance(h_genes, gamma))
            if cfg.use_abundance_head
            else None
        )
        lb, ub, _ = self.dynamic_box(c_j, e_u)
        # Under `log_c_from_latent` the energies are a function of the draw, so they are
        # computed inside the loop from each draw's own z; the mask is the anchored table's
        # and does not depend on it.
        delta_g: torch.Tensor
        dg_mask: torch.Tensor
        if self.log_c_from_latent_enabled:
            dg_mask = self.delta_r_g0_mask
        else:
            delta_g, dg_mask, _ = self.reaction_potential(context)
            if self.thermo_box:
                # The same directional box the deterministic head gets, so every DRAW is
                # thermodynamically feasible rather than only the mean.
                lb, ub, _ = self._thermo_tighten(lb, ub, delta_g)
        active = dg_mask & (~self.thermo_exempt)
        has_thermo = bool(active.any()) and self.thermo_mode is not ThermoMode.OFF

        draws: list[torch.Tensor] = []
        feas: list[dict[str, torch.Tensor]] = []
        for _ in range(n_samples):
            draw_lb, draw_ub = lb, ub
            if self.log_c_from_latent_enabled:
                latent, _, _ = self._latent(context, force_sample=True)
                delta_g, _, _ = self.reaction_potential(context, latent=latent)
                if self.thermo_box:
                    draw_lb, draw_ub, _ = self._thermo_tighten(lb, ub, delta_g)
                v = (latent * cfg.flux_scale) @ self.null_basis.t()
                if self.parameterization == "projection":
                    v = self._project_feasible(v, draw_lb, draw_ub)
            elif self.parameterization == "warmstart":
                # EVERY DRAW IS A DIFFERENT DIRECTION OUT OF THE SAME REFERENCE, which is
                # what makes the sampler's support a neighborhood of the polytope rather
                # than a Gaussian somewhere near it.
                box_lb, box_ub = lb, ub
                if self.projection_end == "boxforce":
                    box_lb, box_ub, _ = self._thermo_tighten(lb, ub, delta_g)
                v_pre, _, _, _ = self._warmstart_flux(
                    context, box_lb, box_ub, draw=True
                )
                v = self._project_feasible(v_pre, box_lb, box_ub)
                if self.projection_end == "force":
                    lb_force, ub_force, _ = self._thermo_tighten(lb, ub, delta_g)
                    v = torch.minimum(torch.maximum(v, lb_force), ub_force)
                draw_lb, draw_ub = box_lb, box_ub
            elif self.parameterization == "fluxforce":
                box_lb, box_ub = lb, ub
                if self.projection_end == "boxforce":
                    box_lb, box_ub, _ = self._thermo_tighten(lb, ub, delta_g)
                v_pre, _ = self._fluxforce_flux(
                    context, box_lb, box_ub, delta_g, dg_mask, draw=True
                )
                v = self._project_feasible(v_pre, box_lb, box_ub)
                if self.projection_end == "force":
                    # Every DRAW leaves through the direction clip too, so the sampler's
                    # support is the set the deterministic head lands in.
                    lb_force, ub_force, _ = self._thermo_tighten(lb, ub, delta_g)
                    v = torch.minimum(torch.maximum(v, lb_force), ub_force)
            else:
                latent, _, _ = self._latent(context, force_sample=True)
                v = (latent * cfg.flux_scale) @ self.null_basis.t()
                if self.parameterization == "projection":
                    v = self._project_feasible(v, lb, ub)
            piece: dict[str, torch.Tensor] = {"v": v, "lb": draw_lb, "ub": draw_ub}
            if has_thermo:
                piece["delta_r_g"] = delta_g
            draws.append(v)
            feas.append(self.feasibility(piece))
        stacked = {k: torch.stack([f[k] for f in feas]) for k in feas[0]}
        return FluxSample(
            v=torch.stack(draws),
            balance_ratio_median=stacked["balance_ratio_median"],
            balance_ratio_max=stacked["balance_ratio_max"],
            box_violation_frac=stacked["box_violation_frac"],
            thermo_violation_frac=stacked["thermo_violation_frac"],
            thermo_uphill_frac=stacked["thermo_uphill_frac"],
            n_samples=n_samples,
            parameterization=self.parameterization,
            projection_steps=self.projection_steps,
            projection_end=self.projection_end,
        )

    def set_constraint_epoch(self, epoch: int) -> float:
        r"""Set the constraint scale for one epoch and return it.

        Linear warmup, :math:`s(e)=\mathrm{scale}\cdot\min(1, e/W)`, so epoch 0 carries **no**
        constraint at all and epoch :math:`W` carries the full configured scale. The trainer
        calls this once per epoch; a layer whose trainer never calls it stays at the
        configured scale, which is the pre-warmup behavior.

        Why the schedule exists, and what has NOT been shown. Measured: on the betaxanthin
        cell the flux arms sat at a validation prediction spread of 0.000 for 40 epochs with
        their validation data loss between 1 and 5, while the same recipe without the module
        escaped the plateau by epoch 26 at a data loss of 0.49 (jobs 1636, 1637 against 1634).
        Hypothesis (untested): the constraint terms hold the head on the plateau, and letting
        the data term move first lets it escape. The warmup is the experiment that separates
        those, not evidence for either.
        """
        delay = self.config.constraint_delay_epochs
        warmup = self.config.constraint_warmup_epochs
        if epoch < delay:
            ramp = 0.0
        else:
            since = float(epoch - delay)
            ramp = 1.0 if warmup <= 0 else min(1.0, since / float(warmup))
        self.constraint_scale = float(self.config.constraint_final_scale) * ramp
        return self.constraint_scale

    def constraint_loss(self, out: dict[str, torch.Tensor]) -> torch.Tensor:
        """Weighted sum of the physics terms. Every term is already dimensionless.

        Normalization to dimensionless happens inside :meth:`forward` rather than here,
        because a term whose scale depends on the phenotype's units cannot be tuned
        against another one: that is the failure the 019 joint runs hit, where an
        un-normalized morphology MSE swamped expression.

        The whole sum is multiplied by :attr:`constraint_scale`, which
        :meth:`set_constraint_epoch` moves. Scaling the SUM rather than the individual
        weights keeps the relative balance between the physics terms, which was set against
        their magnitudes at initialization, fixed while the physics is ramped in.
        """
        w = self.config.weights
        total = (
            w["balance"] * out["c_balance"]
            + w["thermo"] * out["c_thermo"]
            + w["budget"] * out["c_budget"]
            + w["dissipation"] * out["c_dissipation"]
            + w["parsimony"] * out["c_parsimony"]
            + w["thermo_prior"] * out["c_thermo_prior"]
            + w.get("box", 1.0) * out["c_box"]
            # The latent KL, which is exactly zero unless `latent_stochastic` is set, so the
            # sum is unchanged for every arm that does not sample. The default comes from
            # `kl_weight` rather than from the dict because 026's build_flux_layer writes
            # the whole `weights` dict itself and an arm cannot add a key to it.
            + w.get("kl", self.config.kl_weight) * out["c_kl"]
        )
        # THE DENSITY TERM IS OUTSIDE THE RAMP, and that is a decision rather than an
        # oversight. Every cell of waves 6 and 7 runs `constraint_final_scale` at 0.0 or
        # 0.1, and at 0.0 a term inside the product is identically zero: the arm would
        # report a density weight it never applied. The term is not one of the physics
        # constraints being ramped in, it is what stops the head from satisfying the
        # balance criterion by emptying the network. At the default weight of 0.0 it adds
        # exactly zero, so every arm before wave 8 computes the same loss it always did.
        density = w.get("density", self.config.density_weight) * out["c_density"]
        # THE ACTIVE-ROW BALANCE TERM IS OUTSIDE THE RAMP FOR THE SAME REASON, and it is the
        # one the structural audit says is missing: the rows that carry flux and do not
        # balance almost all COULD balance, they were simply never asked to. Zero at the
        # default weight, so every arm before wave 8 computes the loss it always did.
        active_balance = (
            w.get("balance_active", self.config.balance_active_weight)
            * out["c_balance_active"]
        )
        # PARSIMONY OUTSIDE THE RAMP, for the third time and for the same reason. The
        # `weights['parsimony']` entry above is inside the product, and the recipe these
        # cells run holds `constraint_scale` at 0.0, so parsimony has never actually been
        # applied in this experiment. Zero at the default weight, so every arm before this
        # computes the loss it always did.
        parsimony = (
            w.get("parsimony_outside_ramp", self.config.parsimony_outside_ramp_weight)
            * out["c_parsimony"]
        )
        return total * self.constraint_scale + density + active_balance + parsimony

    def coverage_report(self) -> dict[str, Any]:
        """What fraction of the layer rests on data rather than on a default.

        This is not optional reporting. A capacity constraint built on a default
        ``kcat`` for every reaction is a uniform rescaling of the box, not an enzyme
        constraint, and the two are indistinguishable from the loss curve alone.
        """
        return {
            "n_reactions": self.n_reactions,
            "n_metabolites": self.n_metabolites,
            "n_catalytic_units": self.n_units,
            "n_gem_genes": self.n_gem_genes,
            "n_gem_genes_in_model_universe": self.n_genes_mapped,
            "n_constitutive_genes": self.n_constitutive_genes,
            "n_reactions_with_gpr": int(self.has_gpr.sum()),
            "n_independent_balance_rows": int(self.independent_rows.numel()),
            "n_reactions_second_law_applied": int(
                (self.delta_r_g0_mask & (~self.thermo_exempt)).sum()
            ),
            "n_reactions_second_law_exempt": int(self.thermo_exempt.sum()),
            # WHERE THE CONCENTRATION STARTS, which is what decides the directional box at
            # initialization. A run that does not record this cannot say whether its box
            # licensed a real flux or a random one.
            "log_c_init_file": (
                None
                if self.log_c_init is None
                else {
                    "path": self.log_c_init.source_path,
                    "sha256": self.log_c_init.sha256,
                    "route": self.log_c_init.route,
                    "window_m": self.log_c_init.window_m,
                    "exempt_file": self.log_c_init.exempt_file,
                    "exempt_file_sha256": self.log_c_init.exempt_file_sha256,
                    "n_constraints_satisfied": self.log_c_init.n_constraints_satisfied,
                    "residual_total_slack_kj": (
                        self.log_c_init.residual_total_slack_kj
                    ),
                    "max_realized_error_ln_c": self.log_c_init_max_error,
                }
            ),
            # THE EXEMPTION LIST, with what it is a claim ABOUT. A list built against one
            # thermodynamic route says nothing about another, so the route, the window and
            # the table's hashes travel with the file's own hash.
            "thermo_exempt_file": (
                None
                if self.thermo_exempt_list is None
                else {
                    "path": self.thermo_exempt_list.source_path,
                    "sha256": self.thermo_exempt_list.sha256,
                    "n_reactions_listed": len(self.thermo_exempt_list.reaction_ids),
                    "n_reactions_newly_exempted": self.n_thermo_exempt_from_file,
                    "route": self.thermo_exempt_list.route,
                    "window_m": self.thermo_exempt_list.window_m,
                    "epsilon_kj": self.thermo_exempt_list.epsilon_kj,
                    "method": self.thermo_exempt_list.method,
                    "gate_met": self.thermo_exempt_list.gate_met,
                    "thermo_table_sha256": self.thermo_exempt_list.thermo_table_sha256,
                }
            ),
            "n_reactions_delta_g_known": int(self.delta_r_g0_mask.sum()),
            "frac_reactions_delta_g_known": float(self.delta_r_g0_mask.float().mean()),
            "n_metabolites_delta_f_g_known": int(self.met_g_mask.sum()),
            "frac_metabolites_delta_f_g_known": float(self.met_g_mask.float().mean()),
            "kcat_is_default_for_all_units": self.kcat_is_default,
            # THE THREE-WAY SPLIT. `kcat_is_default_for_all_units` is False the moment one
            # unit carries a real value, which is why it says nothing useful about a table
            # that is 4 percent measured.
            "kcat_tier_counts": dict(self.kcat_tier_counts),
            "km_tier_counts": dict(self.km_tier_counts),
            "km_table_present": self.km_is_present,
            "thermo_source": str(self.config.thermo_source),
            "thermo_route_counts": (
                dict(self.reaction_thermo.counts)
                if self.reaction_thermo is not None
                else None
            ),
            "thermo_metabolite_counts": (
                dict(self.reaction_thermo.met_counts)
                if self.reaction_thermo is not None
                else None
            ),
            "n_reactions_delta_g_sigma_known": int(self.delta_r_g0_sigma_mask.sum()),
            "constraint_warmup_epochs": self.config.constraint_warmup_epochs,
            "constraint_final_scale": self.config.constraint_final_scale,
            "constraint_scale_now": self.constraint_scale,
            "mw_is_default_for_all_genes": self.mw_is_default,
            "use_abundance_head": self.config.use_abundance_head,
            "abundance_prior_is_default": self.abundance_prior_is_default,
            "abundance_prior_median_mmol_per_gdw": (
                float(torch.exp(self.log_e0).median())
                if self.config.use_abundance_head
                else None
            ),
            "n_transport_terms_sourced": self.n_transport_terms_sourced,
            "transport_term_is_zero": self.n_transport_terms_sourced == 0,
            "thermo_mode": str(self.thermo_mode),
            "parameterization": self.parameterization,
            "parameterization_raw": self.config.parameterization,
            "projection_steps": (
                self.projection_steps
                if self.parameterization in {"projection", "fluxforce", "warmstart"}
                else None
            ),
            "projection_end": (
                self.projection_end
                if self.parameterization in {"projection", "fluxforce", "warmstart"}
                else None
            ),
            "projection_method": (
                self.projection_method
                if self.parameterization in {"projection", "fluxforce", "warmstart"}
                else None
            ),
            # THE WARM START'S REFERENCE, so a run records what it started from. A pFBA
            # solution at zero growth, or one carrying flux on every reaction, would train
            # without complaint and mean something entirely different.
            "warm_start_reference": (
                None
                if self.reference_flux is None
                else {
                    "route": self.config.warm_start_reference,
                    "growth_fraction": self.reference_flux.growth_fraction,
                    "objective_index": self.reference_flux.objective_index,
                    "objective_value": self.reference_flux.objective_value,
                    "total_absolute_flux": self.reference_flux.total_absolute_flux,
                    "n_reactions_carrying_flux": (
                        self.reference_flux.n_reactions_carrying_flux
                    ),
                    "frac_reactions_carrying_flux": (
                        self.reference_flux.n_reactions_carrying_flux
                        / float(self.n_reactions)
                    ),
                    "digest": self.reference_flux.digest,
                    "from_cache": self.reference_flux.from_cache,
                    "init_rel": self.config.warm_start_init_rel,
                    "alpha_now": float(self.warm_alpha()),
                }
            ),
            # The temperature the flux-force relation is evaluated at, in kJ/mol, so a
            # reported efficacy can be reproduced from the number the run actually used.
            "rt_kj_per_mol": self.rt,
            "availability_fold": self.config.availability_fold,
            "balance_active_row_set": self.config.balance_active_row_set,
            "balance_active_weight": self.config.balance_active_weight,
            "density_weight": self.config.density_weight,
            "parsimony_outside_ramp_weight": (
                self.config.parsimony_outside_ramp_weight
            ),
            "density_turnover_threshold": self.config.density_turnover_threshold,
            # The frozen set, so an audit can tell whether it was scored on the training
            # one: the checkpoint carries parameters and no buffers, and a rebuild
            # re-freezes on its own first batch.
            "frozen_active_row_count": int(self.frozen_active_rows.sum()),
            "frozen_active_row_digest": hashlib.sha256(
                self.frozen_active_rows.numpy().tobytes()
            ).hexdigest()[:16],
            "n_latent": self.n_latent,
            # MEASURED at construction, not asserted: the projection is a projection only
            # if the basis is orthonormal, and it lands in ker S only to the precision the
            # basis was computed at.
            "null_basis_orthonormality_error": self.null_basis_orthonormality_error,
            "null_basis_balance_error": self.null_basis_balance_error,
            "thermo_box": self.thermo_box,
            "thermo_box_margin_kj": self.thermo_box_margin_kj,
            "latent_stochastic": self.latent_stochastic,
            "kl_weight": self.config.weights.get("kl", self.config.kl_weight),
            "thermo_uncertainty": self.thermo_uncertainty,
            "n_delta_g_components": self.n_delta_g_components,
            "n_reactions_covariance_covered": self.n_reactions_cov_covered,
            "delta_g_sqrt_cov_sha256": (
                None if self.delta_g_sqrt_cov is None else self.delta_g_sqrt_cov.sha256
            ),
        }
