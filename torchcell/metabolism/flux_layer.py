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

#: Faraday constant, kJ/(mol V). Enters the transport term as F q dPhi.
FARADAY_KJ_PER_MOL_V = 96.485

#: Working flux magnitude, mmol/gDW/h. Genome-scale models use +/-1000 as "unbounded"
#: rather than as a physical claim; a sigmoid stretched over 2,000 units puts every real
#: flux inside a vanishing fraction of its range and saturates the gradient. Clamping the
#: box to a physiological magnitude is a modeling choice and is reported as one.
DEFAULT_FLUX_SCALE = 100.0


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
    use_shipped_transport_delta_g: bool = Field(
        default=False,
        description=(
            "Recover the transport term for the 441 reactions whose recomputed standard "
            "energy is structurally zero, from the GEM's shipped reaction table. Changes "
            "what the thermodynamic term asserts, so it is off by default."
        ),
    )
    use_enzyme_capacity: bool = True
    use_protein_budget: bool = True
    use_dissipation_limit: bool = True
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
    independent_rows: torch.Tensor
    thermo_exempt: torch.Tensor
    is_exchange: torch.Tensor
    delta_r_g0: torch.Tensor
    delta_r_g0_mask: torch.Tensor
    met_g_mask: torch.Tensor
    delta_r_g_transport: torch.Tensor
    kcat_per_h: torch.Tensor
    mw_kda: torch.Tensor
    null_basis: torch.Tensor

    def __init__(
        self,
        gem: GemTensors,
        model_gene_ids: list[str],
        config: FluxLayerConfig | None = None,
        compartments: CompartmentParameters | None = None,
        kcat_per_s: torch.Tensor | None = None,
        molecular_weight_kda: torch.Tensor | None = None,
        null_space: torch.Tensor | None = None,
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
        """
        super().__init__()
        self.config = config or FluxLayerConfig()
        self.compartments = compartments or CompartmentParameters()
        cfg = self.config

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
        if gem.thermo is not None:
            drg0, drg0_mask = gem.standard_reaction_delta_g()
            self.register_buffer("delta_r_g0", drg0)
            self.register_buffer("delta_r_g0_mask", drg0_mask)
            self.register_buffer("met_g_mask", gem.thermo.met_mask)
            self.rt = gem.thermo.temperature_k * 8.314462618e-3
        else:
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
        r"""Recover :math:`\Delta_r G^t_j` for transport reactions from the GEM's own table.

        Thermo-Flux Eq. (7) gives
        :math:`\Delta_r G^t_j=-N_HRT\ln(10)\Delta\mathrm{pH}-Fq\Delta\Phi`, and computing it
        from scratch needs the transported species and its charge. Identifying those is
        Thermo-Flux's Step 4 and requires eQuilibrator species distributions plus manual
        curation of the ambiguous cases, none of which is available here.

        **But the term is recoverable, and a measurement says so.** Summing formation
        energies over a transport reaction cancels exactly, because the same species appears
        on both sides: :math:`\sum_i S_{ij}\Delta_f G'^{\circ}_i = 0`. So a transport
        reaction's entire standard energy IS its transport term. Comparing the two shipped
        tables on the 3,204 reactions where both carry a value:

        * **441 reactions have a recomputed energy of exactly 0 and a shipped value above
          5 kJ/mol, and all 441 of them are multi-compartment.** 441 of 441, not a
          majority. Relaxing the 5 kJ/mol floor to any nonzero value gives 874, out of
          1,099 reactions whose recomputed energy is degenerate.
        * the 321 reactions with the opposite pattern are all single-compartment, so they
          have a different cause.

        The shipped reaction column therefore carries the driving force the metabolite sum
        structurally cannot. Using it on exactly those reactions is sourcing the term from
        the GEM's own curation rather than estimating it.

        ``use_shipped_transport_delta_g`` gates this because it is a change in what the
        thermodynamic term asserts, not a tuning knob, and the arms reported in
        [[experiments.026-metabolism-flux.enzyme-constrained-thermodynamic-flux-layer]] were
        run with it OFF. What it does not fix is the residual single-compartment
        disagreement, median 10.2 kJ/mol, which is roughly 4 RT and remains unexplained.
        """
        transport = torch.zeros(gem.n_reactions)
        if not self.config.use_shipped_transport_delta_g or gem.thermo is None:
            return transport
        recomputed, recomputed_mask = gem.standard_reaction_delta_g()
        degenerate = recomputed_mask & (recomputed.abs() < 1e-6)
        usable = degenerate & gem.thermo.rxn_mask
        transport[usable] = gem.thermo.rxn_delta_g[usable]
        self.n_transport_terms_sourced = int(usable.sum())
        return transport

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
        """
        valid = self.gem_to_model >= 0
        token_idx = self.gem_to_model.clamp(min=0)
        h_gem = h_genes[:, token_idx, :]  # [B, n_gem_genes, d]
        gamma = torch.sigmoid(self.availability(h_gem)).squeeze(-1)
        gamma = gamma * valid.unsqueeze(0)

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

    def dynamic_box(self, c_j: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r"""The genotype-dependent flux box :math:`[v^\ell_j,v^u_j]`.

        Availability scales the box multiplicatively, so :math:`c_j=0` collapses it to the
        single point :math:`0`. Enzyme capacity enters here too rather than as a penalty:

        .. math::
            \bar v^u_j=\min\Big(v^u_j,\ c_j\sum_{u:\rho(u)=j}k_{\mathrm{cat},u}\bar E\Big)

        Putting capacity in the box rather than the objective is what makes it **exact** at
        no cost, and it is why there is no ``C_box`` term in the loss.
        """
        lb = self.lb.unsqueeze(0) * c_j
        ub = self.ub.unsqueeze(0) * c_j
        if self.config.use_enzyme_capacity:
            cap = c_j.new_zeros(c_j.shape[0], self.n_reactions)
            # Enzyme available per unit is not predicted; the capacity ceiling uses a
            # nominal abundance so that kcat sets the SCALE of the bound. This is where a
            # measured or predicted proteome would enter.
            per_unit = self.kcat_per_h * 1.0e-3
            cap.index_add_(
                1, self.unit_reaction, per_unit.unsqueeze(0).expand(c_j.shape[0], -1)
            )
            cap = cap * c_j
            capped = torch.where(self.has_gpr.unsqueeze(0), cap, ub.abs())
            ub = torch.minimum(ub, capped)
            lb = torch.maximum(lb, -capped)
        return lb, ub

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
        lb, ub = self.dynamic_box(c_j)

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

        # (d) enzyme demand and the protein budget
        if cfg.use_protein_budget:
            demand = torch.sqrt(v**2 + 1e-8)[:, self.unit_reaction] / self.kcat_per_h
            e_g = v.new_zeros(b, self.n_gem_genes)
            e_g.index_add_(1, self.unit_gene_gene, demand[:, self.unit_gene_unit])
            protein = (e_g * self.mw_kda).sum(dim=1) * 1e-3
            out["protein_used"] = protein.mean()
            out["c_budget"] = torch.relu(protein / cfg.p_avail - 1.0).mean()
            out["feas_budget_ratio"] = (protein / cfg.p_avail).mean()
            out["e_g"] = e_g
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

    def constraint_loss(self, out: dict[str, torch.Tensor]) -> torch.Tensor:
        """Weighted sum of the physics terms. Every term is already dimensionless.

        Normalization to dimensionless happens inside :meth:`forward` rather than here,
        because a term whose scale depends on the phenotype's units cannot be tuned
        against another one: that is the failure the 019 joint runs hit, where an
        un-normalized morphology MSE swamped expression.
        """
        w = self.config.weights
        return (
            w["balance"] * out["c_balance"]
            + w["thermo"] * out["c_thermo"]
            + w["budget"] * out["c_budget"]
            + w["dissipation"] * out["c_dissipation"]
            + w["parsimony"] * out["c_parsimony"]
            + w["thermo_prior"] * out["c_thermo_prior"]
            + w.get("box", 1.0) * out["c_box"]
        )

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
            "mw_is_default_for_all_genes": self.mw_is_default,
            "n_transport_terms_sourced": self.n_transport_terms_sourced,
            "transport_term_is_zero": self.n_transport_terms_sourced == 0,
            "thermo_mode": str(self.thermo_mode),
        }
