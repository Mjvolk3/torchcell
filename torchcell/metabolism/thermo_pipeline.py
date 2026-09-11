# torchcell/metabolism/thermo_pipeline
# [[torchcell.metabolism.thermo_pipeline]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/metabolism/thermo_pipeline
# Test file: tests/torchcell/metabolism/test_thermo_pipeline.py

r"""Organism-agnostic standard reaction energies for a GEM, by route, with reasons.

WHAT THIS IS FOR
----------------
Every thermodynamic number the flux layer consumes has so far been produced by a chain of
yeast-GEM scripts under ``experiments/026-metabolism-flux/scripts/``. That chain works and
is the test case here, but it is not portable: it hardcodes yeast-GEM's file layout, its
compartment letters, its shipped ``model_metDeltaG.csv``, and a lipid-name grammar that
only yeast-GEM writes. A new organism, which is the point of the enzyme-constrained layer,
would have to reimplement all of it.

This module is that chain with the organism taken out. It takes

* a GEM, as any :mod:`cobra` model (SBML or YAML on disk, or an in-memory model), whose
  metabolites carry whatever identifiers they carry: InChI, SMILES, KEGG, MetaNetX, ChEBI,
  SEED, plus a formula and a charge;
* a compartment table of pH, ionic strength and pMg;
* an ordered list of RESOLVER stages, each of which prices whatever metabolites it can and
  says why it could not price the rest;

and it emits one :class:`ThermoPipelineResult`: a formation energy per metabolite with the
route that produced it, a :math:`\Delta_r G'^\circ` per reaction with a propagated sigma,
and, for every reaction it cannot price, a reason drawn from a closed enum plus the
metabolites that blocked it.

THE RULES THAT MAKE THE OUTPUT READABLE
---------------------------------------
1. **A reaction is priced only when EVERY participant carries a formation energy.** One
   unpriced participant makes the sum a number that looks like a free energy and is not
   one. This is the same predicate ``audit_reaction_thermo_coverage.py`` uses, restated
   once here so the two cannot drift.
2. **Every value carries its route, and routes are never averaged.** A reaction assembled
   from two routes is tagged ``mixed`` rather than attributed to either, because its
   uncertainty is not either route's.
3. **Every failure carries a reason from :class:`UnpricedReason`, not a blank.** The
   distinction between "no structure exists for this species" and "a structure exists and
   the group decomposer refused it" is the difference between work that is possible and
   work that is not, and a coverage percentage hides it.
4. **No stage may silently substitute a value.** A surrogate (an energy borrowed from a
   compound of the same composition) is a route of its own, is admitted only under an
   explicit spread bound, and carries that spread as its sigma.

WHAT IS ORGANISM-SPECIFIC AND WHERE IT LIVES
--------------------------------------------
Nothing in this module names a species, a compartment letter, a metabolite or a file. The
three things that ARE organism-specific enter as data or as a plugged-in stage:

* the compartment conditions, as :class:`CompartmentConditions`;
* the shipped thermodynamic table, if the GEM has one, through
  :class:`TableResolver`, which is handed a plain ``{met_id: value}`` mapping and the unit
  it is written in;
* a structure writer for species whose structure is implied by their NAME rather than by
  an accession, through the :class:`StructureWriter` protocol. yeast-GEM's lipid grammar
  (``phosphatidylethanolamine (1-16:1, 2-16:1)``) is one implementation of that protocol
  and lives in the experiment folder, not here.

Resolvers that need eQuilibrator import it inside their methods, so this module imports in
the plain torchcell environment where ``equilibrator_api`` is absent.
"""

from __future__ import annotations

import hashlib
import json
import os.path as osp
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:  # pragma: no cover - typing only
    import cobra

#: kJ per kcal, thermochemical. A shipped table declared in kcal is converted on read.
KJ_PER_KCAL = 4.184

#: Tokens that make an elemental formula a placeholder rather than a composition. ``R`` is
#: a variable side chain, ``X`` a generic halogen or an unspecified atom, ``*`` an
#: attachment point. A formula carrying one of these describes a FAMILY of molecules, so no
#: single structure and no single formation energy exists for it.
PLACEHOLDER_TOKENS = ("R", "X", "*")


class ThermoRoute(StrEnum):
    """How one formation energy was obtained. Ordered from most to least direct.

    ``CACHE_ACCESSION``      -- an identifier on the metabolite resolved to a compound that
                                already carries a component-contribution energy.
    ``CACHE_STRUCTURE``      -- a structure (InChI/SMILES) resolved to such a compound.
    ``STRUCTURE_WRITER``     -- a structure was WRITTEN from the metabolite's name by a
                                plugged-in writer, then priced.
    ``EXTERNAL_PKA``         -- a structure was priced through the external-pKa bridge,
                                because no licensed pKa tool was available.
    ``COMPOSITION_SURROGATE``-- borrowed from compounds of the same heavy-atom composition,
                                admitted only when their spread is within a stated bound.
    ``REPEAT_UNIT_SURROGATE``-- a polymer species whose formula is a repeat unit, priced as
                                its monomer minus water with a measured bond correction and
                                an inflated sigma. A surrogate, and labeled one.
    ``SHIPPED_TABLE``        -- the GEM's own metabolite table.
    ``LITERATURE_TABLE``     -- a measured and evaluated value from a pinned thermochemical
                                table, for species a group-contribution model cannot reach
                                at all (a bare ion has no groups and no training reaction).
    ``SHIPPED_REACTION``     -- the GEM's own REACTION table, which is not a formation
                                energy at all and can only fill a reaction directly.
    ``NONE``                 -- no route succeeded.
    """

    CACHE_ACCESSION = "cache_accession"
    CACHE_STRUCTURE = "cache_structure"
    STRUCTURE_WRITER = "structure_writer"
    EXTERNAL_PKA = "external_pka"
    LITERATURE_TABLE = "literature_table"
    COMPOSITION_SURROGATE = "composition_surrogate"
    REPEAT_UNIT_SURROGATE = "repeat_unit_surrogate"
    SHIPPED_TABLE = "shipped_table"
    SHIPPED_REACTION = "shipped_reaction_column"
    MIXED = "mixed"
    NONE = "none"


class UnpricedReason(StrEnum):
    """Why a metabolite or a reaction has no energy. Closed set, by design.

    A free-text reason is a reason nobody can count. These are the categories the coverage
    ceiling is reported over, and each one implies a different next step:

    ``PLACEHOLDER_FORMULA``        -- nothing is possible; the species is a family.
    ``PSEUDO_METABOLITE``          -- a modeling device (a biomass pool, a SLIME backbone,
                                      a difference formula), not a molecule.
    ``NO_STRUCTURE_AVAILABLE``     -- a definite formula but no structure in any namespace
                                      and no writer that can compose one; needs a structure.
    ``GROUP_DECOMPOSITION_FAILED`` -- a structure exists and the decomposer refused it;
                                      needs a decomposition, not a structure.
    ``NO_PKA_PREDICTION``          -- a structure exists and decomposes, and the pKa stage
                                      returned nothing for it.
    ``PLACEHOLDER_STRUCTURE``      -- a structure exists and carries an ATTACHMENT POINT,
                                      so the species is a residue inside a macromolecule
                                      rather than a molecule. Nothing is possible.
    ``SURROGATE_SPREAD_TOO_WIDE``  -- same-composition compounds disagree by more than the
                                      admitted bound, so borrowing is not defensible.
    ``NOT_ATTEMPTED``              -- no stage in this configuration claimed it.
    ``BLOCKED_BY_METABOLITE``      -- reactions only: at least one participant is unpriced.
    ``NO_PARTICIPANTS``            -- reactions only: an empty stoichiometry.
    """

    PLACEHOLDER_FORMULA = "placeholder_formula"
    PSEUDO_METABOLITE = "pseudo_metabolite"
    NO_STRUCTURE_AVAILABLE = "no_structure_available"
    GROUP_DECOMPOSITION_FAILED = "group_decomposition_failed"
    NO_PKA_PREDICTION = "no_pka_prediction"
    PLACEHOLDER_STRUCTURE = "placeholder_structure"
    SURROGATE_SPREAD_TOO_WIDE = "surrogate_spread_too_wide"
    NOT_ATTEMPTED = "not_attempted"
    BLOCKED_BY_METABOLITE = "blocked_by_metabolite"
    NO_PARTICIPANTS = "no_participants"


class CompartmentConditions(BaseModel):
    """pH, ionic strength and pMg per compartment, with a stated default.

    The defaults are eQuilibrator's, and they are DEFAULTS rather than constants: a
    compartment the table does not name is priced at them and is listed in
    :attr:`ThermoPipelineResult.compartments_at_default`, so a missing organelle shows up
    as a reported assumption instead of a silent one.
    """

    model_config = ConfigDict(extra="forbid")

    p_h: dict[str, float] = Field(default_factory=dict)
    ionic_strength_m: dict[str, float] = Field(default_factory=dict)
    p_mg: dict[str, float] = Field(default_factory=dict)
    default_p_h: float = 7.0
    default_ionic_strength_m: float = 0.25
    default_p_mg: float = 3.0
    temperature_k: float = 298.15
    source: str = Field(
        default="",
        description="Where these numbers come from. Empty means nobody said, which is a "
        "finding about the run rather than a formatting detail.",
    )

    def for_compartment(self, compartment: str) -> tuple[float, float, float]:
        """``(pH, ionic strength in M, pMg)`` for one compartment."""
        return (
            self.p_h.get(compartment, self.default_p_h),
            self.ionic_strength_m.get(compartment, self.default_ionic_strength_m),
            self.p_mg.get(compartment, self.default_p_mg),
        )

    def missing(self, compartments: list[str]) -> list[str]:
        """Compartments this table does not name, in sorted order."""
        return sorted({c for c in compartments if c not in self.p_h})


class MetaboliteIdentity(BaseModel):
    """Everything a resolver is allowed to look at for one metabolite.

    Built from a cobra metabolite by :func:`identities_from_model`. Deliberately a flat
    record rather than the cobra object: a resolver that reaches into the model can reach
    into anything, and this is the interface a new organism has to satisfy.
    """

    model_config = ConfigDict(extra="forbid")

    met_id: str
    name: str = ""
    compartment: str = ""
    formula: str = ""
    charge: int = 0
    #: Namespace -> identifier(s), lowercased namespace keys as cobra writes them
    #: (``kegg.compound``, ``metanetx.chemical``, ``chebi``, ``seed.compound``, ...).
    accessions: dict[str, list[str]] = Field(default_factory=dict)
    inchi: str | None = None
    inchi_key: str | None = None
    smiles: str | None = None

    @property
    def has_placeholder_formula(self) -> bool:
        """True when the formula names a family of molecules rather than one molecule."""
        return any(token in self.formula for token in PLACEHOLDER_TOKENS)

    @property
    def has_structure(self) -> bool:
        """True when an InChI or a SMILES is present."""
        return bool(self.inchi or self.smiles)


class MetabolitePrice(BaseModel):
    """One formation energy, or one stated failure.

    ``dgf_prime_kj_per_mol`` is the TRANSFORMED formation energy at the metabolite's own
    compartment conditions, which is the quantity that sums over a stoichiometric column.
    """

    model_config = ConfigDict(extra="forbid")

    met_id: str
    dgf_prime_kj_per_mol: float | None = None
    sigma_kj_per_mol: float | None = None
    route: ThermoRoute = ThermoRoute.NONE
    reason: UnpricedReason | None = None
    p_h: float | None = None
    ionic_strength_m: float | None = None
    p_mg: float | None = None
    temperature_k: float | None = None
    detail: str = ""
    #: Free-form, route-specific provenance: the accession that hit, the pKa list used, the
    #: predictor and its weight hashes, the compounds a surrogate was drawn from.
    provenance: dict[str, object] = Field(default_factory=dict)

    @property
    def priced(self) -> bool:
        """True when a route produced a value."""
        return self.dgf_prime_kj_per_mol is not None


class ReactionPrice(BaseModel):
    """One standard reaction energy, or one stated failure."""

    model_config = ConfigDict(extra="forbid")

    reaction_id: str
    delta_r_g0_kj_per_mol: float | None = None
    sigma_kj_per_mol: float | None = None
    route: ThermoRoute = ThermoRoute.NONE
    reason: UnpricedReason | None = None
    blocking_met_ids: list[str] = Field(default_factory=list)
    #: The reasons of the blocking metabolites, deduplicated. A reaction blocked only by
    #: placeholder-formula species can never be priced; one blocked by a decomposition
    #: failure can.
    blocking_reasons: list[UnpricedReason] = Field(default_factory=list)
    n_participants: int = 0


@runtime_checkable
class MetaboliteResolver(Protocol):
    """A stage that prices some metabolites and says why it could not price the rest.

    A resolver is handed only the metabolites still unpriced when it runs, so the order of
    :attr:`ThermoPipelineConfig.stage_names` IS the precedence policy. It must return one
    entry per metabolite it was handed: a priced one, or one carrying a
    :class:`UnpricedReason`. Returning nothing for a metabolite leaves it
    :attr:`UnpricedReason.NOT_ATTEMPTED`, which is a different statement from "this stage
    tried and failed".
    """

    name: str

    def resolve(
        self, metabolites: list[MetaboliteIdentity], conditions: CompartmentConditions
    ) -> list[MetabolitePrice]:
        """Price what this stage can."""
        ...


@runtime_checkable
class StructureWriter(Protocol):
    """A stage that composes a STRUCTURE for a metabolite from its name.

    This is the pluggable half of the organism-specific surface. yeast-GEM writes acyl
    composition into lipid names, so a writer can compose a SMILES and check the formula it
    produces against the one the GEM declares; another organism's model may write nothing
    of the kind, or write something else. A writer returns ``None`` for a name it does not
    recognize, and the pipeline then records
    :attr:`UnpricedReason.NO_STRUCTURE_AVAILABLE` rather than guessing.

    The formula check is the writer's own responsibility and is not optional: a written
    structure that does not reproduce the GEM's formula is a different molecule.
    """

    name: str

    def write(self, metabolite: MetaboliteIdentity) -> str | None:
        """A SMILES for this metabolite, or ``None`` if the name does not determine one."""
        ...


class TableResolver(BaseModel):
    """A resolver over a plain ``{met_id: value}`` table, e.g. a GEM's shipped column.

    ``units`` is stated rather than assumed, because a shipped table that declares kJ and
    holds kcal is not hypothetical: yeast-GEM 9 ships yETFL's kcal/mol values under a kJ/mol
    header, and reading them as kJ understates every energy by 4.184x. The conversion
    happens here, on read, so no downstream path can hold a kcal number and call it kJ.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = "shipped_table"
    values: dict[str, float]
    units: str = Field(default="kJ/mol", description="'kJ/mol' or 'kcal/mol'.")
    sigma_kj_per_mol: float | None = None
    source_path: str = ""
    sha256: str = ""
    route: ThermoRoute = ThermoRoute.SHIPPED_TABLE
    #: Per-metabolite override of the unit conversion, for a table that MIXES units. The
    #: value is the factor already applied by ``units``, which is divided back out. yeast-GEM
    #: needs this: 33 of its 3,210 reaction rows come from dGPredictor in kJ/mol while the
    #: rest come from yETFL in kcal/mol.
    already_kj_per_mol: list[str] = Field(default_factory=list)

    def resolve(
        self, metabolites: list[MetaboliteIdentity], conditions: CompartmentConditions
    ) -> list[MetabolitePrice]:
        """Price every metabolite the table has a row for."""
        if self.units not in ("kJ/mol", "kcal/mol"):
            raise ValueError(f"{self.name}: unknown units {self.units!r}")
        factor = KJ_PER_KCAL if self.units == "kcal/mol" else 1.0
        exempt = set(self.already_kj_per_mol)
        out: list[MetabolitePrice] = []
        for met in metabolites:
            raw = self.values.get(met.met_id)
            if raw is None or not np.isfinite(raw):
                out.append(
                    MetabolitePrice(
                        met_id=met.met_id,
                        reason=UnpricedReason.NOT_ATTEMPTED,
                        detail=f"{self.name} has no row for this metabolite",
                    )
                )
                continue
            scale = 1.0 if met.met_id in exempt else factor
            p_h, ionic, p_mg = conditions.for_compartment(met.compartment)
            out.append(
                MetabolitePrice(
                    met_id=met.met_id,
                    dgf_prime_kj_per_mol=float(raw) * scale,
                    sigma_kj_per_mol=self.sigma_kj_per_mol,
                    route=self.route,
                    p_h=p_h,
                    ionic_strength_m=ionic,
                    p_mg=p_mg,
                    temperature_k=conditions.temperature_k,
                    detail=f"{self.name} row, {self.units} x {scale:g}",
                    provenance={
                        "source_path": self.source_path,
                        "sha256": self.sha256,
                        "units_in_file": self.units,
                        "conversion_factor": scale,
                    },
                )
            )
        return out


class ThermoPipelineConfig(BaseModel):
    """Everything the pipeline needs that is not the GEM itself."""

    model_config = ConfigDict(extra="forbid")

    organism: str = Field(default="", description="Free text, for the record only.")
    gem_path: str = Field(
        default="", description="SBML/YAML path, or empty for in-memory."
    )
    gem_sha256: str = ""
    conditions: CompartmentConditions = Field(default_factory=CompartmentConditions)
    stage_names: list[str] = Field(
        default_factory=list,
        description="Resolver names in precedence order; filled in from the stages passed.",
    )
    #: Reaction ids the caller wants the flux-carrying fraction reported over, usually the
    #: reactions with nonzero FVA width. Empty means the fraction is not reported.
    flux_carrying_reaction_ids: list[str] = Field(default_factory=list)
    #: A last-resort reaction-level table, e.g. a GEM's own reaction column. It cannot be
    #: summed and cannot supply a sigma, so it is a separate route and never merged into the
    #: metabolite tally.
    reaction_table: dict[str, float] = Field(default_factory=dict)
    reaction_table_units: str = "kJ/mol"
    reaction_table_already_kj: list[str] = Field(default_factory=list)
    reaction_table_source: str = ""
    reaction_table_sha256: str = ""
    #: Metabolite ids to treat as modeling devices rather than molecules (biomass pools,
    #: SLIME backbones). Supplied by the caller, because what counts as a pseudo-metabolite
    #: is a property of the model, not of chemistry.
    pseudo_metabolite_ids: list[str] = Field(default_factory=list)


class RouteCounts(BaseModel):
    """How many entities each route produced, and how many failed for each reason."""

    by_route: dict[str, int] = Field(default_factory=dict)
    by_reason: dict[str, int] = Field(default_factory=dict)
    n_total: int = 0
    n_priced: int = 0

    @property
    def fraction(self) -> float:
        """Priced over total, 0 when the total is 0."""
        return self.n_priced / self.n_total if self.n_total else 0.0


class ThermoPipelineResult(BaseModel):
    """The reaction table, the metabolite table behind it, and every stated gap."""

    model_config = ConfigDict(extra="forbid")

    built_at: str
    config: ThermoPipelineConfig
    compartments: list[str]
    compartments_at_default: list[str]
    metabolites: list[MetabolitePrice]
    reactions: list[ReactionPrice]
    metabolite_counts: RouteCounts
    reaction_counts: RouteCounts
    flux_carrying: dict[str, float] = Field(default_factory=dict)
    #: met_id -> how many otherwise priceable reactions it alone blocks. The short list that
    #: turns a diffuse coverage gap into a work queue.
    blocking_metabolites: list[dict[str, object]] = Field(default_factory=list)
    notes: str = ""


def sha256_of(path: str) -> str:
    """sha256 of a file, the canonical identity of every artifact this pipeline reads."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _accessions_of(metabolite: cobra.Metabolite) -> dict[str, list[str]]:
    """The metabolite's annotation as ``namespace -> [id, ...]``.

    cobra stores an annotation value as a string or a list of strings depending on how many
    the SBML carried, and code that assumes one shape drops the other silently.
    """
    out: dict[str, list[str]] = {}
    for key, value in (metabolite.annotation or {}).items():
        if isinstance(value, (list, tuple, set)):
            out[str(key).lower()] = [str(v) for v in value]
        else:
            out[str(key).lower()] = [str(value)]
    return out


def identities_from_model(model: cobra.Model) -> list[MetaboliteIdentity]:
    """Every metabolite of a cobra model as a :class:`MetaboliteIdentity`.

    Structures are read from the annotation under the keys SBML models actually use
    (``inchi``, ``inchikey``, ``smiles``), each of which may be absent. Nothing is inferred
    from the metabolite id or name here: a writer that composes a structure from a name is a
    separate, opt-in stage.
    """
    out: list[MetaboliteIdentity] = []
    for met in model.metabolites:
        accessions = _accessions_of(met)

        def _first(*keys: str) -> str | None:
            for key in keys:
                values = accessions.get(key)
                if values:
                    return values[0]
            return None

        out.append(
            MetaboliteIdentity(
                met_id=met.id,
                name=met.name or "",
                compartment=met.compartment or "",
                formula=met.formula or "",
                charge=int(met.charge or 0),
                accessions=accessions,
                inchi=_first("inchi"),
                inchi_key=_first("inchikey", "inchi_key"),
                smiles=_first("smiles"),
            )
        )
    return out


def _classify_unattempted(met: MetaboliteIdentity, pseudo: set[str]) -> UnpricedReason:
    """The reason a metabolite no stage priced could not be priced.

    Ordered so the most final reason wins: a pseudo-metabolite is not a molecule at all, a
    placeholder formula is a family of molecules, and only then does the absence of a
    structure become the operative blocker.
    """
    if met.met_id in pseudo:
        return UnpricedReason.PSEUDO_METABOLITE
    if not met.formula:
        return UnpricedReason.PSEUDO_METABOLITE
    if met.has_placeholder_formula:
        return UnpricedReason.PLACEHOLDER_FORMULA
    if not met.has_structure:
        return UnpricedReason.NO_STRUCTURE_AVAILABLE
    return UnpricedReason.NOT_ATTEMPTED


def resolve_metabolites(
    identities: list[MetaboliteIdentity],
    stages: list[MetaboliteResolver],
    conditions: CompartmentConditions,
    pseudo_metabolite_ids: list[str] | None = None,
) -> list[MetabolitePrice]:
    """Run the stages in order, each on what is still unpriced, and keep the first hit.

    The precedence is the stage order and nothing else: a later stage never overwrites an
    earlier one, and a metabolite priced by stage 1 is not even offered to stage 2. That is
    what makes "database first, prediction second" a property of the configuration rather
    than of the code.

    A stage's stated failure is kept as the reason, so the final record says which stage
    tried last and why it stopped, rather than collapsing every gap into "unpriced".
    """
    pseudo = set(pseudo_metabolite_ids or [])
    priced: dict[str, MetabolitePrice] = {}
    reasons: dict[str, MetabolitePrice] = {}
    remaining = list(identities)
    for stage in stages:
        if not remaining:
            break
        answers = stage.resolve(remaining, conditions)
        returned = {a.met_id for a in answers}
        unknown = returned - {m.met_id for m in remaining}
        if unknown:
            raise ValueError(
                f"stage {stage.name!r} returned {len(unknown)} metabolites it was not "
                f"handed, e.g. {sorted(unknown)[:5]}"
            )
        for answer in answers:
            if answer.priced:
                priced[answer.met_id] = answer
            elif answer.reason is UnpricedReason.NOT_ATTEMPTED:
                # NOT_ATTEMPTED means "this stage has nothing to say about this
                # metabolite", so it must not erase what an earlier stage DID say. Without
                # this guard a diagnosis-only stage placed last wiped every specific reason
                # for the species it happened not to cover, and the ceiling then counted
                # them under the generic fallback.
                reasons.setdefault(answer.met_id, answer)
            elif answer.reason is not None:
                reasons[answer.met_id] = answer
        remaining = [m for m in remaining if m.met_id not in priced]

    out: list[MetabolitePrice] = []
    for met in identities:
        if met.met_id in priced:
            out.append(priced[met.met_id])
            continue
        stated = reasons.get(met.met_id)
        structural = _classify_unattempted(met, pseudo)
        # A structural fact about the MODEL outranks a stage's report about itself. A
        # biomass pool that a structure writer declined is not "no structure available",
        # it is not a molecule, and a coverage ceiling counted the first way would put it
        # on the work queue forever.
        if structural in (
            UnpricedReason.PSEUDO_METABOLITE,
            UnpricedReason.PLACEHOLDER_FORMULA,
        ):
            out.append(
                MetabolitePrice(
                    met_id=met.met_id,
                    reason=structural,
                    detail=stated.detail if stated is not None else "",
                )
            )
            continue
        if stated is not None and stated.reason is not UnpricedReason.NOT_ATTEMPTED:
            out.append(stated)
            continue
        out.append(
            MetabolitePrice(
                met_id=met.met_id,
                reason=structural,
                detail=(
                    stated.detail
                    if stated is not None
                    else "no configured stage priced this metabolite"
                ),
            )
        )
    return out


def _counts(
    routes: list[ThermoRoute], reasons: list[UnpricedReason | None], n_priced: int
) -> RouteCounts:
    by_route: dict[str, int] = {}
    for route in routes:
        by_route[str(route)] = by_route.get(str(route), 0) + 1
    by_reason: dict[str, int] = {}
    for reason in reasons:
        if reason is None:
            continue
        by_reason[str(reason)] = by_reason.get(str(reason), 0) + 1
    return RouteCounts(
        by_route=dict(sorted(by_route.items(), key=lambda kv: -kv[1])),
        by_reason=dict(sorted(by_reason.items(), key=lambda kv: -kv[1])),
        n_total=len(routes),
        n_priced=n_priced,
    )


def assemble_reactions(
    model: cobra.Model,
    prices: list[MetabolitePrice],
    reason_by_met: dict[str, UnpricedReason | None],
    config: ThermoPipelineConfig,
    formation_sigma: dict[str, float] | None = None,
) -> list[ReactionPrice]:
    r"""Sum the metabolite table over :math:`S`, one reaction at a time.

    A reaction is priced only when every participant is, so the loop is over reactions and
    the blocking participants are recorded per reaction rather than counted globally.

    The sigma is the quadrature sum of the participants' formation sigmas weighted by their
    stoichiometric coefficients. That treats the formation uncertainties as INDEPENDENT and
    is therefore an underestimate whenever they are not: component contribution's
    covariance is strongly correlated across compounds sharing groups, and the correlated
    version is :math:`\lVert S_j^\top \Sigma^{1/2} \rVert_2`. A caller holding the square
    root factor should propagate it and overwrite these values; a caller holding only
    per-metabolite sigmas gets this, and the result records which it was.
    """
    priced = {p.met_id: p for p in prices if p.priced}
    reaction_fill = dict(config.reaction_table)
    exempt = set(config.reaction_table_already_kj)
    factor = KJ_PER_KCAL if config.reaction_table_units == "kcal/mol" else 1.0

    out: list[ReactionPrice] = []
    for reaction in model.reactions:
        stoichiometry = {m.id: float(c) for m, c in reaction.metabolites.items()}
        if not stoichiometry:
            out.append(
                ReactionPrice(
                    reaction_id=reaction.id,
                    reason=UnpricedReason.NO_PARTICIPANTS,
                    n_participants=0,
                )
            )
            continue
        blocking = sorted(m for m in stoichiometry if m not in priced)
        if not blocking:
            # Every participant is in `priced`, and a priced entry has a float energy by
            # construction, so the cast is a statement about `MetabolitePrice.priced`
            # rather than an unchecked assumption.
            total = 0.0
            for met_id, coefficient in stoichiometry.items():
                energy = priced[met_id].dgf_prime_kj_per_mol
                assert energy is not None
                total += coefficient * energy
            routes = {priced[m].route for m in stoichiometry}
            sigma = None
            if formation_sigma is not None and all(
                m in formation_sigma for m in stoichiometry
            ):
                sigma = float(
                    np.sqrt(
                        sum(
                            (coefficient * formation_sigma[m]) ** 2
                            for m, coefficient in stoichiometry.items()
                        )
                    )
                )
            out.append(
                ReactionPrice(
                    reaction_id=reaction.id,
                    delta_r_g0_kj_per_mol=float(total),
                    sigma_kj_per_mol=sigma,
                    route=routes.pop() if len(routes) == 1 else ThermoRoute.MIXED,
                    n_participants=len(stoichiometry),
                )
            )
            continue
        raw = reaction_fill.get(reaction.id)
        if raw is not None and np.isfinite(raw):
            scale = 1.0 if reaction.id in exempt else factor
            out.append(
                ReactionPrice(
                    reaction_id=reaction.id,
                    delta_r_g0_kj_per_mol=float(raw) * scale,
                    route=ThermoRoute.SHIPPED_REACTION,
                    blocking_met_ids=blocking,
                    blocking_reasons=sorted(
                        {
                            r
                            for r in (reason_by_met.get(m) for m in blocking)
                            if r is not None
                        }
                    ),
                    n_participants=len(stoichiometry),
                )
            )
            continue
        out.append(
            ReactionPrice(
                reaction_id=reaction.id,
                reason=UnpricedReason.BLOCKED_BY_METABOLITE,
                blocking_met_ids=blocking,
                blocking_reasons=sorted(
                    {
                        r
                        for r in (reason_by_met.get(m) for m in blocking)
                        if r is not None
                    }
                ),
                n_participants=len(stoichiometry),
            )
        )
    return out


def run_thermo_pipeline(
    model: cobra.Model,
    config: ThermoPipelineConfig,
    stages: list[MetaboliteResolver],
    formation_sigma: dict[str, float] | None = None,
) -> ThermoPipelineResult:
    """The whole pipeline: identities, staged resolution, reaction assembly, accounting.

    Args:
        model: Any cobra model. yeast-GEM 9.0.2 is the test case; nothing here reads it.
        config: Conditions, the optional reaction-level fill, and what counts as a
            pseudo-metabolite in this model.
        stages: Resolvers in PRECEDENCE order.
        formation_sigma: Optional per-metabolite formation sigma in kJ/mol. Independent
            propagation; see :func:`assemble_reactions`.
    """
    from datetime import UTC, datetime

    identities = identities_from_model(model)
    config = config.model_copy(update={"stage_names": [s.name for s in stages]})
    prices = resolve_metabolites(
        identities, stages, config.conditions, config.pseudo_metabolite_ids
    )
    reason_by_met = {p.met_id: p.reason for p in prices}
    reactions = assemble_reactions(
        model, prices, reason_by_met, config, formation_sigma
    )

    compartments = sorted({m.compartment for m in identities if m.compartment})
    met_counts = _counts(
        [p.route for p in prices],
        [p.reason for p in prices],
        sum(1 for p in prices if p.priced),
    )
    rxn_counts = _counts(
        [r.route for r in reactions],
        [r.reason for r in reactions],
        sum(1 for r in reactions if r.delta_r_g0_kj_per_mol is not None),
    )

    flux_carrying: dict[str, float] = {}
    if config.flux_carrying_reaction_ids:
        wanted = set(config.flux_carrying_reaction_ids)
        subset = [r for r in reactions if r.reaction_id in wanted]
        n_priced = sum(1 for r in subset if r.delta_r_g0_kj_per_mol is not None)
        flux_carrying = {
            "n_flux_carrying": float(len(subset)),
            "n_priced": float(n_priced),
            "fraction": float(n_priced / len(subset)) if subset else 0.0,
        }

    blocked: dict[str, int] = {}
    for reaction in reactions:
        if reaction.delta_r_g0_kj_per_mol is not None:
            continue
        for met_id in reaction.blocking_met_ids:
            blocked[met_id] = blocked.get(met_id, 0) + 1
    by_id = {m.met_id: m for m in identities}
    blocking = [
        {
            "met_id": met_id,
            "name": by_id[met_id].name if met_id in by_id else "",
            "formula": by_id[met_id].formula if met_id in by_id else "",
            "reason": str(reason_by_met.get(met_id) or ""),
            "n_reactions_blocked": count,
        }
        for met_id, count in sorted(blocked.items(), key=lambda kv: -kv[1])
    ]

    return ThermoPipelineResult(
        built_at=datetime.now(UTC).isoformat(),
        config=config,
        compartments=compartments,
        compartments_at_default=config.conditions.missing(compartments),
        metabolites=prices,
        reactions=reactions,
        metabolite_counts=met_counts,
        reaction_counts=rxn_counts,
        flux_carrying=flux_carrying,
        blocking_metabolites=blocking,
        notes=(
            "A reaction is priced only when EVERY participant carries a formation energy. "
            "'mixed' marks a reaction assembled from more than one route, whose "
            "uncertainty is not any single route's. Every gap carries an UnpricedReason."
        ),
    )


def load_model(path: str) -> cobra.Model:
    """Read a GEM from SBML or YAML, dispatching on the extension.

    No format sniffing: an ``.xml`` that is not SBML should fail as an SBML parse error,
    not be silently retried as something else.
    """
    import cobra

    extension = osp.splitext(path)[1].lower()
    if extension in (".xml", ".sbml"):
        return cobra.io.read_sbml_model(path)
    if extension in (".yml", ".yaml"):
        return cobra.io.load_yaml_model(path)
    if extension == ".json":
        return cobra.io.load_json_model(path)
    if extension == ".mat":
        return cobra.io.load_matlab_model(path)
    raise ValueError(f"unsupported GEM format: {path}")


class FileTableResolver:
    """A stage over a per-metabolite table on disk, CSV or parquet.

    The generic half of a resolver: any organism with a formation-energy table keyed by
    metabolite id gets a stage without writing code. A structure writer or a live
    eQuilibrator call still needs one, which is what the :class:`MetaboliteResolver`
    protocol is for.
    """

    def __init__(
        self,
        name: str,
        path: str,
        value_column: str,
        route: ThermoRoute = ThermoRoute.CACHE_ACCESSION,
        sigma_column: str | None = None,
        met_id_column: str = "met_id",
        units: str = "kJ/mol",
    ) -> None:
        """Read the table once and index it by metabolite id."""
        import pandas as pd

        if units not in ("kJ/mol", "kcal/mol"):
            raise ValueError(f"{name}: unknown units {units!r}")
        frame = (
            pd.read_parquet(path)
            if osp.splitext(path)[1].lower() == ".parquet"
            else pd.read_csv(path)
        )
        missing = {met_id_column, value_column} - set(frame.columns)
        if missing:
            raise ValueError(f"{path} has no column {sorted(missing)}")
        frame = frame[frame[value_column].notna()]
        factor = KJ_PER_KCAL if units == "kcal/mol" else 1.0
        self.name = name
        self.path = path
        self.route = route
        self.units = units
        self.sha256 = sha256_of(path)
        self.values = {
            str(k): float(v) * factor
            for k, v in zip(frame[met_id_column], frame[value_column])
        }
        self.sigma = (
            {
                str(k): float(v)
                for k, v in zip(frame[met_id_column], frame[sigma_column])
                if v == v
            }
            if sigma_column and sigma_column in frame.columns
            else {}
        )

    def resolve(
        self, metabolites: list[MetaboliteIdentity], conditions: CompartmentConditions
    ) -> list[MetabolitePrice]:
        """Price every metabolite the table has a finite row for."""
        out: list[MetabolitePrice] = []
        for met in metabolites:
            value = self.values.get(met.met_id)
            if value is None:
                out.append(
                    MetabolitePrice(
                        met_id=met.met_id,
                        reason=UnpricedReason.NOT_ATTEMPTED,
                        detail=f"{self.name} has no row for this metabolite",
                    )
                )
                continue
            p_h, ionic, p_mg = conditions.for_compartment(met.compartment)
            out.append(
                MetabolitePrice(
                    met_id=met.met_id,
                    dgf_prime_kj_per_mol=value,
                    sigma_kj_per_mol=self.sigma.get(met.met_id),
                    route=self.route,
                    p_h=p_h,
                    ionic_strength_m=ionic,
                    p_mg=p_mg,
                    temperature_k=conditions.temperature_k,
                    detail=f"{self.name}, {self.units} on read",
                    provenance={"source_path": self.path, "sha256": self.sha256},
                )
            )
        return out


def debye_hueckel_kj(ionic_strength_m: float, temperature_k: float) -> float:
    """The extended Debye-Huckel coefficient in kJ/mol, eQuilibrator's form and constants.

    Restated here rather than imported because this module must import in an environment
    without eQuilibrator. ``fetch_ion_formation_energies.py`` ASSERTS that the transform
    built from this function equals
    ``equilibrator_cache.thermodynamic_constants._legendre_transform`` over a grid of
    charges, ionic strengths and temperatures, and records the deviation in the table file,
    so the copy is verified on every rebuild rather than trusted.
    """
    a1, a2, a3 = 9.20483e-3, 1.284668e-5, 4.95199e-8
    alpha = a1 * temperature_k - a2 * temperature_k**2 + a3 * temperature_k**3
    return float(alpha / (1.0 / ionic_strength_m**0.5 + 1.6))


class IonFormationResolver:
    r"""Price monatomic aqueous ions from a pinned thermochemical table, by element and charge.

    A bare ion is the one class of species a group-contribution model cannot reach even in
    principle: it has no groups to decompose and it does not appear in the training
    reactions. Its standard formation energy is instead measured and evaluated in the
    thermochemical literature, so it arrives as a TABLE with provenance rather than as a
    computation, and it is a route of its own
    (:attr:`ThermoRoute.LITERATURE_TABLE`) so no reader mistakes it for a prediction.

    Organism-agnostic: the key is (element symbol, charge), and a metabolite matches when
    its formula is a single element symbol. Nothing here names a metabolite or a model.

    **Two properties of these values that the caller has to know.**

    *They cancel where the reaction is balanced.* In a reaction elementally balanced in the
    ion's element, the ion's coefficients sum to zero and the absolute value drops out, as
    does the ionic-strength term whenever the compartments share an ionic strength. The
    values change no balanced reaction's energy; they matter only for reactions that are not
    elementally balanced, which in a GEM are the exchange, sink and biomass pseudo-reactions
    where a standard reaction energy has no physical meaning.

    *A GEM may declare the wrong charge.* yeast-GEM 9.0.2 declares potassium and sodium with
    charge 0, which is the metal rather than the aqueous ion, while declaring calcium, zinc,
    copper and cadmium correctly. Where the model's charge matches no tabulated species for
    that element and the element has exactly ONE tabulated ion, that ion is used and the
    disagreement is recorded per metabolite in ``provenance['charge_source']``; the
    alternative reading, a neutral metal atom in solution, is not a species. Where the
    element has several tabulated ions the metabolite is left unpriced, because then the
    model's annotation is the only thing that could pick one and it is the thing in doubt.
    """

    name = "ion_formation_table"

    def __init__(
        self,
        path: str,
        *,
        allow_charge_from_table: bool = True,
        gauge_shift_kj_per_mol: float = 0.0,
    ) -> None:
        """Read the pinned ion table and index it by element.

        ``gauge_shift_kj_per_mol`` adds a constant to every ion's value and exists for ONE
        purpose: proving the cancellation claim above by rerunning the pipeline with a
        different gauge and showing which reaction energies move. It is not a tuning knob
        and a production run leaves it at zero.
        """
        with open(path) as handle:
            payload = json.load(handle)
        self.path = path
        self.sha256 = sha256_of(path)
        self.citation = payload.get("citation", "")
        self.allow_charge_from_table = allow_charge_from_table
        self.gauge_shift_kj_per_mol = gauge_shift_kj_per_mol
        self.by_element: dict[str, list[dict[str, object]]] = {}
        for entry in payload["entries"]:
            self.by_element.setdefault(str(entry["element"]), []).append(entry)

    def _entry(self, element: str, charge: int) -> tuple[dict[str, object] | None, str]:
        candidates = self.by_element.get(element, [])
        for entry in candidates:
            if int(cast(int, entry["charge"])) == charge:
                return entry, "gem_annotation"
        if self.allow_charge_from_table and len(candidates) == 1:
            return candidates[0], "table_single_ion"
        return None, "no_entry"

    def resolve(
        self, metabolites: list[MetaboliteIdentity], conditions: CompartmentConditions
    ) -> list[MetabolitePrice]:
        """Price every metabolite whose formula is a single tabulated element."""
        out: list[MetabolitePrice] = []
        for met in metabolites:
            element = (met.formula or "").strip()
            if not element or not element.isalpha() or len(element) > 2:
                out.append(
                    MetabolitePrice(
                        met_id=met.met_id, reason=UnpricedReason.NOT_ATTEMPTED
                    )
                )
                continue
            entry, charge_source = self._entry(element, met.charge)
            if entry is None:
                out.append(
                    MetabolitePrice(
                        met_id=met.met_id,
                        reason=UnpricedReason.NOT_ATTEMPTED,
                        detail=(
                            f"no tabulated aqueous ion for {element} at charge "
                            f"{met.charge}"
                        ),
                    )
                )
                continue
            charge = int(cast(int, entry["charge"]))
            p_h, ionic, p_mg = conditions.for_compartment(met.compartment)
            temperature = conditions.temperature_k
            # Recomputed at the compartment's own temperature from the tabulated enthalpy
            # and reaction entropy rather than taken at the table's reporting temperature,
            # so the stage is condition-general.
            delta_f_h = float(cast(float, entry["delta_f_h_kj_per_mol"]))
            delta_r_s = float(cast(float, entry["delta_r_s_j_per_mol_k"]))
            dgf = delta_f_h - temperature * delta_r_s / 1000.0
            transform = (
                -debye_hueckel_kj(ionic, temperature) * charge**2 if ionic > 0 else 0.0
            )
            out.append(
                MetabolitePrice(
                    met_id=met.met_id,
                    dgf_prime_kj_per_mol=dgf + transform + self.gauge_shift_kj_per_mol,
                    route=ThermoRoute.LITERATURE_TABLE,
                    p_h=p_h,
                    ionic_strength_m=ionic,
                    p_mg=p_mg,
                    temperature_k=temperature,
                    detail=f"{entry['species']} via {charge_source}",
                    provenance={
                        "source_path": self.path,
                        "sha256": self.sha256,
                        "citation": self.citation,
                        "species": entry["species"],
                        "formation_reaction": entry["formation_reaction"],
                        "charge_source": charge_source,
                        "gem_declared_charge": met.charge,
                        "tabulated_charge": charge,
                        "delta_f_g_untransformed_kj_per_mol": dgf,
                        "debye_hueckel_term_kj_per_mol": transform,
                        "gauge_shift_kj_per_mol": self.gauge_shift_kj_per_mol,
                    },
                )
            )
        return out


def write_result(result: ThermoPipelineResult, path: str) -> None:
    """Serialize the result, including every per-entity record."""
    with open(path, "w") as handle:
        handle.write(result.model_dump_json(indent=1))


def read_result(path: str) -> ThermoPipelineResult:
    """Read back a serialized result."""
    with open(path) as handle:
        return ThermoPipelineResult.model_validate_json(handle.read())


def summarize(result: ThermoPipelineResult) -> str:
    """A short human-readable accounting, the thing a run should print."""
    lines = [
        f"organism: {result.config.organism or 'unstated'}",
        f"stages: {' -> '.join(result.config.stage_names) or 'none'}",
        (
            f"metabolites: {result.metabolite_counts.n_priced}"
            f"/{result.metabolite_counts.n_total}"
            f" ({result.metabolite_counts.fraction:.3%})"
        ),
        (
            f"reactions:   {result.reaction_counts.n_priced}"
            f"/{result.reaction_counts.n_total}"
            f" ({result.reaction_counts.fraction:.3%})"
        ),
    ]
    if result.flux_carrying:
        lines.append(
            f"flux carrying: {result.flux_carrying['n_priced']:.0f}"
            f"/{result.flux_carrying['n_flux_carrying']:.0f}"
            f" ({result.flux_carrying['fraction']:.3%})"
        )
    if result.compartments_at_default:
        lines.append(
            "compartments priced at the DEFAULT conditions: "
            + ", ".join(result.compartments_at_default)
        )
    lines.append("reaction routes: " + json.dumps(result.reaction_counts.by_route))
    lines.append("reaction gaps:   " + json.dumps(result.reaction_counts.by_reason))
    lines.append("metabolite gaps: " + json.dumps(result.metabolite_counts.by_reason))
    return "\n".join(lines)


# ===================================================================================== CLI
#
# A new organism with a formation-energy table needs no code: point this at the GEM, the
# table and a conditions file. A structure writer or a live eQuilibrator call still needs a
# stage, which is what a driver like `thermo_pipeline_yeast9.py` is for.


def _parse_table_spec(spec: str) -> dict[str, str]:
    """``path[:value_column[:sigma_column[:route[:units]]]]`` as a keyword dict."""
    parts = spec.split(":")
    keys = ("path", "value_column", "sigma_column", "route", "units")
    out = {k: v for k, v in zip(keys, parts) if v}
    out.setdefault("value_column", "dgf_prime_kj_per_mol")
    out.setdefault("route", ThermoRoute.CACHE_ACCESSION.value)
    out.setdefault("units", "kJ/mol")
    return out


def main(argv: list[str] | None = None) -> int:
    """Price a GEM's reactions from tables on disk, and report every gap by reason."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Standard reaction energies for a GEM."
    )
    parser.add_argument("--gem", required=True, help="SBML, YAML, JSON or MAT model")
    parser.add_argument(
        "--conditions",
        default=None,
        help="JSON of a CompartmentConditions; omitted means eQuilibrator's defaults",
    )
    parser.add_argument(
        "--metabolite-table",
        action="append",
        default=[],
        metavar="PATH[:VALUE_COL[:SIGMA_COL[:ROUTE[:UNITS]]]]",
        help=(
            "a per-metabolite formation-energy table, CSV or parquet, keyed by met_id. "
            "Repeat in PRECEDENCE order: the first table that prices a metabolite wins."
        ),
    )
    parser.add_argument(
        "--reaction-table",
        default=None,
        help="CSV of reaction_id,value used as a last resort where no metabolite route can",
    )
    parser.add_argument("--reaction-table-units", default="kJ/mol")
    parser.add_argument(
        "--reaction-table-already-kj",
        default=None,
        help="a file of reaction ids, one per line, exempt from the unit conversion",
    )
    parser.add_argument(
        "--flux-carrying",
        default=None,
        help="a file of reaction ids, one per line, to report the priced fraction over",
    )
    parser.add_argument(
        "--pseudo-metabolites",
        default=None,
        help="a file of metabolite ids, one per line, that are modeling devices",
    )
    parser.add_argument("--organism", default="")
    parser.add_argument("--out", required=True, help="where to write the result JSON")
    args = parser.parse_args(argv)

    conditions = CompartmentConditions()
    if args.conditions:
        with open(args.conditions) as handle:
            conditions = CompartmentConditions.model_validate_json(handle.read())

    def _ids(path: str | None) -> list[str]:
        if not path:
            return []
        with open(path) as handle:
            return [line.strip() for line in handle if line.strip()]

    reaction_values: dict[str, float] = {}
    if args.reaction_table:
        import csv as csv_module

        with open(args.reaction_table) as handle:
            for row in csv_module.reader(handle):
                if len(row) < 2:
                    continue
                try:
                    reaction_values[row[0]] = float(row[1])
                except ValueError:
                    continue  # the header line

    model = load_model(args.gem)
    config = ThermoPipelineConfig(
        organism=args.organism,
        gem_path=args.gem,
        gem_sha256=sha256_of(args.gem),
        conditions=conditions,
        flux_carrying_reaction_ids=_ids(args.flux_carrying),
        reaction_table=reaction_values,
        reaction_table_units=args.reaction_table_units,
        reaction_table_already_kj=_ids(args.reaction_table_already_kj),
        reaction_table_source=args.reaction_table or "",
        reaction_table_sha256=(
            sha256_of(args.reaction_table) if args.reaction_table else ""
        ),
        pseudo_metabolite_ids=_ids(args.pseudo_metabolites),
    )
    stages: list[MetaboliteResolver] = []
    for i, spec in enumerate(args.metabolite_table):
        fields = _parse_table_spec(spec)
        stages.append(
            FileTableResolver(
                name=f"table_{i}_{osp.basename(fields['path'])}",
                path=fields["path"],
                value_column=fields["value_column"],
                sigma_column=fields.get("sigma_column"),
                route=ThermoRoute(fields["route"]),
                units=fields["units"],
            )
        )

    result = run_thermo_pipeline(model, config, stages)
    write_result(result, args.out)
    print(summarize(result))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
