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
concentrations and fluxes. That inference is exactly what a latent variable does. Here
:math:`m` is a learned per-reaction offset with a Gaussian prior, so the same quantity is
fit by gradient descent instead of by an MILP regression.

:math:`Q` itself is **not available**: eQuilibrator is not installed and the GEM ships
point values with no covariance, so the offset is currently isotropic rather than
correlated. That is a real gap and it is recorded as one rather than approximated. The
covariance would enter as a single matrix multiply and nothing else would change.

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

import math
from typing import Any

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field

from torchcell.metabolism.constraints import (
    CONC_LOWER_M,
    CONC_UPPER_M,
    GemTensors,
    ThermoMode,
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
            "The exactness budget has to be spent on one or the other, and which one "
            "matters more is what the two arms measure."
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
        """
        super().__init__()
        self.config = config or FluxLayerConfig()
        self.compartments = compartments or CompartmentParameters()
        cfg = self.config
        # The warmup multiplier, moved by `set_constraint_epoch`. It starts at the FINAL
        # scale rather than at zero so a trainer that never calls the setter runs the
        # constraints exactly as every run before 2026-09-10 did.
        self.constraint_scale = float(cfg.constraint_final_scale)

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
        self.parameterization = cfg.parameterization
        if cfg.parameterization == "nullspace":
            if null_space is None:
                raise ValueError(
                    "parameterization='nullspace' needs the basis; build it with "
                    "torchcell.metabolism.constraints.null_space_basis(gem.s)."
                )
            self.register_buffer("null_basis", null_space)
            self.n_latent = int(null_space.shape[1])
            self.latent_mlp = nn.Sequential(
                nn.Linear(d, d), nn.ReLU(), nn.Linear(d, self.n_latent)
            )
        else:
            self.n_latent = 0
        self.reaction_embedding = nn.Embedding(self.n_reactions, cfg.reaction_embed_dim)
        z_out = 2 if cfg.stochastic else 1
        self.flux_mlp = nn.Sequential(
            nn.Linear(d + cfg.reaction_embed_dim, d), nn.ReLU(), nn.Linear(d, z_out)
        )
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
            # the Q m offset of Thermo-Flux Eq. (5), isotropic because Q is unavailable
            self.delta_g_offset = nn.Parameter(torch.zeros(self.n_reactions))

    # -- construction helpers -------------------------------------------------

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

    def reaction_availability(self, gamma: torch.Tensor) -> torch.Tensor:
        r"""Fold :math:`\gamma_g` up the GPR: softmin over complexes, sum over isozymes.

        .. math::
            c_u=-\tfrac1\beta\log\!\sum_{g\in C_u}e^{-\beta\gamma_g},\qquad
            c_j=\min\Big(1,\ \sum_{u:\,\rho(u)=j}c_u\Big)

        Reactions with no GPR keep availability 1: an unannotated reaction is not one that
        no gene catalyzes, it is one nobody has assigned a gene to, and setting it to zero
        would let a missing annotation delete a reaction.
        """
        beta = self.config.softmin_beta
        b = gamma.shape[0]
        gathered = gamma[:, self.unit_gene_gene]  # [B, nnz]
        acc = gamma.new_zeros(b, self.n_units)
        acc.index_add_(1, self.unit_gene_unit, torch.exp(-beta * gathered))
        # Units with no genes would give log(0); the parser never emits them, but guard.
        c_u = -torch.log(acc.clamp(min=1e-30)) / beta
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
        self, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        r"""Per-reaction :math:`\Delta_r G_j` and the mask of reactions it is valid on.

        In ``ANCHORED`` mode this is Thermo-Flux Eq. (4) term by term. In ``FREE`` mode it
        is :math:`\sum_i S_{ij}\mu_i` with :math:`\mu` unconstrained. In ``OFF`` mode the
        mask is empty and the caller skips the term.
        """
        b = context.shape[0]
        extras: dict[str, torch.Tensor] = {}
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
        lo, hi = math.log(CONC_LOWER_M), math.log(CONC_UPPER_M)
        raw = self.log_c_base.unsqueeze(0) + self.log_c_from_context(context)
        log_c = lo + (hi - lo) * torch.sigmoid(raw)
        conc_term = self.rt * self._s_transpose_matmul(log_c)
        delta = (
            self.delta_r_g0.unsqueeze(0)
            + conc_term
            + self.delta_r_g_transport.unsqueeze(0)
            + self.delta_g_offset.unsqueeze(0)
        )
        extras["log_c"] = log_c
        extras["conc_term"] = conc_term
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

        if self.parameterization == "nullspace":
            # Mass balance is exact here, for any latent: S (N z) = (S N) z = 0.
            z_latent = self.latent_mlp(context) * cfg.flux_scale
            v = z_latent @ self.null_basis.t()
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

        # (b) second law, relaxed to a hinge; no binary variables
        delta_g, dg_mask, extras = self.reaction_potential(context)
        out.update({f"thermo_{k}": t for k, t in extras.items()})
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

        # (c) Gibbs dissipation rate (Niebel 2019), summed over exchange reactions
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

        # (f) prior on the Thermo-Flux uncertainty latent
        if self.thermo_mode is ThermoMode.ANCHORED:
            out["c_thermo_prior"] = (self.delta_g_offset**2).mean()
        else:
            out["c_thermo_prior"] = v.new_zeros(())

        return out

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
        )
        return total * self.constraint_scale

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
        }
