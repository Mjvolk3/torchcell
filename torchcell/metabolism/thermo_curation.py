# torchcell/metabolism/thermo_curation
# [[torchcell.metabolism.thermo_curation]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/metabolism/thermo_curation
# Test file: tests/torchcell/metabolism/test_thermo_curation.py

r"""Curated thermodynamic records, and the pipeline stage that prices from them.

WHAT THIS IS FOR
----------------
:mod:`torchcell.metabolism.thermo_pipeline` prices what an automatic route can reach. What
is left is not a coding problem: a species with no structure in any namespace needs someone
to go and find one, a protein-bound redox carrier has a midpoint potential rather than a
formation energy, and a GEM formula that is impossible needs a person to say what the right
one is. Each of those is a CURATION act, and a curation act with no provenance is a guess
with a number attached.

This module is the typed record of one such act, the loader that validates a directory of
them, and the stage that turns the admitted ones into prices. Nothing in it names an
organism, a metabolite or a file: a record names its own metabolite ids and its own
sources, and the stage reads records.

ADMISSION IS THE WHOLE POINT
----------------------------
A record carries an :class:`Admission`, and only ``measured`` and ``tabulated`` enter the
table a model trains against. ``computed`` and ``provisional`` records are loaded, validated
and REPORTED, under :attr:`~torchcell.metabolism.thermo_pipeline.UnpricedReason.RECORDED_NOT_ADMITTED`,
and they never reach a price. The reason to keep them at all is that "we looked, and what we
found is a computed value we do not trust enough to train on" is a finding, and deleting it
means the next person repeats the search. The reason not to admit them is that a computed
value is indistinguishable from a measured one once it is a float in a parquet.

THE FOUR METABOLITE ROUTES, AND THE TWO THAT ARE NOT METABOLITE ROUTES
----------------------------------------------------------------------
``structure`` and ``polymer_convention``
    A curated SMILES goes through the SAME structure route every other structure goes
    through: create the compound, check that the group decomposer accepts it, take
    component contribution's formation energy, transform it to the compartment's
    conditions. That route needs eQuilibrator, which does not live in the environment the
    GEM driver runs in, so it enters through the :class:`StructurePricer` protocol:
    :class:`EquilibratorStructurePricer` runs it, :class:`PricedStructureTable` replays
    what it wrote. Route ``curated_structure``.
``pka_ladder``
    A structure that already resolves and whose pKa ladder was the missing piece. Same
    route, with the ladder handed to the compound bridge. Route ``curated_pka``.

    **Where the ladder comes from, and what happens without one.** A curated ladder, from
    this record or from a ``pka_ladder`` record covering the same species. The pKa CASCADE
    that ``external_pka_apply_selection.py`` runs is not invoked here: it is a selected
    ensemble of predictors, and a predicted ladder on a curated structure would be a
    computed value entering by the back door of a measured record. A structure with
    ionizable centers and no curated ladder is therefore reported
    :attr:`~torchcell.metabolism.thermo_pipeline.UnpricedReason.NO_PKA_PREDICTION` rather
    than priced, and the remedy is one record field. A structure with NO ionizable center
    is a single microspecies and an empty ladder is its right answer, which is the same
    test ``external_pka_apply_selection.py price`` makes.
``formation_energy``
    A measured or tabulated formation energy, Legendre-transformed to each compartment's
    conditions by the same
    :func:`~torchcell.metabolism.thermo_pipeline.legendre_transform_kj` the ion table uses.
    Route ``curated_formation``.
``formula_correction``
    The GEM's formula or charge is wrong. The correction is applied to the identity BEFORE
    anything checks a composition against it (:func:`apply_formula_corrections`), and the
    override is recorded per metabolite so no reader is left comparing against a formula
    that is no longer the one in use. It prices only when the record also supplies a
    structure, in which case the route is ``curated_formula``.
``carrier_potential`` and ``reaction_energy``
    Neither is a formation energy. A midpoint potential is a property of a COUPLE, so the
    oxidized and reduced forms have no separate formation energy and never will; the
    reaction is priced instead, as :math:`\Delta_r G'^\circ=-nF\Delta E'^\circ` against the
    partner couple, by :class:`CuratedReactionPricer`. Both species stay unpriced under
    :attr:`~torchcell.metabolism.thermo_pipeline.UnpricedReason.PRICED_AT_REACTION_LEVEL`,
    which is a statement that they are priced elsewhere rather than a gap.
"""

from __future__ import annotations

import json
import os.path as osp
import re
from collections.abc import Callable
from enum import StrEnum
from glob import glob
from typing import Any, Literal, Protocol, cast, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from torchcell.metabolism.thermo_pipeline import (
    FARADAY_KJ_PER_MOL_PER_V,
    PLACEHOLDER_TOKENS,
    CompartmentConditions,
    MetaboliteIdentity,
    MetabolitePrice,
    ReactionPrice,
    ThermoRoute,
    UnpricedReason,
    element_counts,
    legendre_transform_kj,
)


class CurationKind(StrEnum):
    """What a curator produced. The record's ``value`` shape follows from this."""

    STRUCTURE = "structure"
    FORMATION_ENERGY = "formation_energy"
    REACTION_ENERGY = "reaction_energy"
    PKA_LADDER = "pka_ladder"
    CARRIER_POTENTIAL = "carrier_potential"
    POLYMER_CONVENTION = "polymer_convention"
    FORMULA_CORRECTION = "formula_correction"
    NOT_FOUND = "not_found"


class Admission(StrEnum):
    """How good the evidence is, which decides whether a value may be trained on.

    ``MEASURED``    -- an experimental value from a primary source.
    ``TABULATED``   -- an evaluated value from a standard compilation, or a structure from a
                       curated chemical database, which is a statement of fact about a
                       molecule rather than a prediction about it.
    ``COMPUTED``    -- produced by a model: a quantum calculation, a group contribution, a
                       predictor. Recorded, never admitted.
    ``PROVISIONAL`` -- a curator's best reading, explicitly not settled. Recorded, never
                       admitted.
    ``NONE``        -- nothing was found.
    """

    MEASURED = "measured"
    TABULATED = "tabulated"
    COMPUTED = "computed"
    PROVISIONAL = "provisional"
    NONE = "none"


#: The admissions that may reach the table a model reads. A binding decision, not a default:
#: a computed value is indistinguishable from a measured one once it is a float in a
#: parquet, so the separation has to happen before the parquet.
ADMITTED = frozenset({Admission.MEASURED, Admission.TABULATED})


class ProvenanceSource(BaseModel):
    """Where one fact came from, in the shape the repository's provenance rule requires.

    Every key is REQUIRED, because a source missing one of them is a source nobody can
    re-fetch or verify. Empty strings are permitted and are reported by
    :attr:`is_complete`: a curator who could not hash a rendered web page should say so by
    leaving the field empty rather than by omitting it, which is the difference between a
    known gap and an unnoticed one.
    """

    model_config = ConfigDict(extra="allow")

    source_url: str
    retrieval_method: str
    retrieval_command: str
    sha256: str
    retrieved_at: str
    verbatim_quote: str
    location: str

    @model_validator(mode="before")
    @classmethod
    def _null_is_empty(cls, data: Any) -> Any:
        """A stated null is a stated gap. An ABSENT key is still an error.

        A source that is a computation this run did has no URL to record, and writing null
        there is the honest thing; what may not happen is the key going missing, because
        then nobody can tell a gap from an oversight.
        """
        if not isinstance(data, dict):
            return data
        return {k: ("" if v is None else v) for k, v in data.items()}

    @property
    def is_complete(self) -> bool:
        """True when nothing that pins this source to a byte string is empty."""
        return bool(self.sha256 and self.verbatim_quote and self.retrieval_command)


class SearchLogEntry(BaseModel):
    """One search a curator ran, and what it returned.

    A negative result is evidence and is only evidence if the search is written down: "not
    in the SI" is worth nothing without the queries that were tried.
    """

    model_config = ConfigDict(extra="allow")

    query: str
    engine: str
    date: str
    outcome: str


class SmilesAttempt(BaseModel):
    """One spelling of a structure that was put to the group decomposer, and the answer.

    A representation that decomposes and one that does not can be the SAME molecule: the
    decomposer's group definitions match the pentavalent spelling of a nitro group and not
    the charge-separated one, and a curator who tried both has measured that rather than
    guessed it.
    """

    model_config = ConfigDict(extra="allow")

    smiles: str
    decomposes: bool | None = None
    error: str | None = None


class DecompositionCheck(BaseModel):
    """Whether component contribution accepts the curated structure, and what it said.

    This is what separates "a structure was missing" from "a structure exists and the
    decomposer refuses it": the first is closed by curation and the second is not.
    """

    model_config = ConfigDict(extra="allow")

    attempted: bool
    decomposes: bool | None = None
    error: str | None = None
    smiles_tried: list[SmilesAttempt] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _one_smiles_is_a_list(cls, data: Any) -> Any:
        """Accept a bare SMILES, or a list of them, where a list of attempts is wanted."""
        if not isinstance(data, dict):
            return data
        tried = data.get("smiles_tried")
        if tried is None:
            return data
        if isinstance(tried, str):
            tried = [tried]
        return {
            **data,
            "smiles_tried": [
                {"smiles": entry} if isinstance(entry, str) else entry
                for entry in tried
            ],
        }


class PricedConditions(BaseModel):
    """The conditions a curator's own number is reported at.

    ``p_h`` may be one number or one per metabolite, because a record covering several
    compartment copies of a species prices each at its own compartment's pH.
    """

    model_config = ConfigDict(extra="allow")

    p_h: float | dict[str, float] | None = None
    ionic_strength_m: float | None = None
    temperature_k: float | None = None
    p_mg: float | None = None

    def for_met(self, met_id: str) -> float | None:
        """The pH this record used for one metabolite, or ``None`` if it did not say."""
        if isinstance(self.p_h, dict):
            return self.p_h.get(met_id)
        return self.p_h


class CuratedPrice(BaseModel):
    """A transformed formation energy the curator computed, per metabolite.

    The escape hatch for a species the structure route cannot reach and a curator can: a
    polymer residue whose repeat unit carries attachment points has no decomposition, but
    the difference between a measured dimer and its monomer does have one. The conditions
    are stated and are CHECKED against the pipeline's before the value is used, because a
    transformed formation energy at another pH is a different number.
    """

    model_config = ConfigDict(extra="allow")

    dgf_prime_kj_per_mol: dict[str, float] = Field(default_factory=dict)
    sigma_kj_per_mol: float | None = None
    conditions: PricedConditions | None = None
    rule: str = ""


class SurrogateRouteInput(BaseModel):
    """What a polymer record hands the repeat-unit surrogate route, and its own answer.

    ``preferred_dimer_difference`` is the curator's value and outranks
    ``drop_in_for_existing_route``, which is what the existing route would produce off its
    generic glycosidic correction. Both are recorded, because the difference between them
    is the size of the correction's calibration error on this compound.
    """

    model_config = ConfigDict(extra="allow")

    route: str = ""
    preferred_dimer_difference: CuratedPrice | None = None
    drop_in_for_existing_route: dict[str, Any] = Field(default_factory=dict)


class PerMetaboliteStructure(BaseModel):
    """One structure inside a record that covers several metabolites separately.

    Two GEM species can share a curation act and not share a molecule:
    4-nitrophenyl phosphate and its hydrolysis product 4-nitrophenol are one record because
    one reaction blocks on both, and only one of them has a usable representation.
    """

    model_config = ConfigDict(extra="allow")

    met_id: str
    smiles: str | None = None
    inchi: str | None = None
    inchikey: str | None = None
    formula: str = ""
    charge: int | None = None
    representation_rule: str | None = None


class StructureValue(BaseModel):
    """A structure a curator found, with the microspecies it represents."""

    model_config = ConfigDict(extra="allow")

    smiles: str | None = None
    inchi: str | None = None
    inchikey: str | None = None
    formula: str = ""
    charge: int | None = None
    microspecies_note: str = ""
    representation_rule: str | None = None
    #: The structure's own pKa ladder, where the curator measured or quoted one. Without it
    #: the compound bridge cannot create a molecule that has an ionizable center.
    pkas: list[float] = Field(default_factory=list)
    #: One entry per metabolite where the record's metabolites are different molecules.
    per_metabolite: list[PerMetaboliteStructure] = Field(default_factory=list)
    #: A value the curator computed where the structure route cannot run.
    curated_dgf_prime: CuratedPrice | None = None

    @model_validator(mode="after")
    def _something_is_here(self) -> StructureValue:
        """A structure record has to carry a structure somewhere."""
        if self.smiles or self.per_metabolite or self.curated_dgf_prime:
            return self
        raise ValueError(
            "a structure record carries neither a smiles, a per_metabolite list nor a "
            "curated_dgf_prime"
        )


class FormationEnergyValue(BaseModel):
    r"""A formation energy, and the state it is reported in.

    ``standard_or_transformed`` is required and is the field that decides what the number
    means: :math:`\Delta_f G^\circ` of one microspecies, or :math:`\Delta_f G'^\circ` of the
    pseudoisomer group at a stated pH. Reading one as the other is a several hundred kJ/mol
    error on a multiply protonated species, so it is never inferred.
    """

    model_config = ConfigDict(extra="allow")

    dgf_kj_per_mol: float
    standard_or_transformed: Literal["standard", "transformed"]
    T_K: float
    pH: float | None = None
    ionic_strength_M: float | None = None
    pMg: float | None = None
    #: The charge and proton count of the species the value belongs to. Needed to move the
    #: value to another compartment's conditions; when absent the GEM's own declaration is
    #: used and the substitution is recorded.
    charge: int | None = None
    num_protons: float | None = None


class ReactionEnergyValue(BaseModel):
    """A reaction energy measured as a reaction energy, for one named reaction."""

    model_config = ConfigDict(extra="allow")

    reaction_id: str
    drg_kj_per_mol: float
    standard_or_transformed: Literal["standard", "transformed"]
    T_K: float
    pH: float | None = None
    ionic_strength_M: float | None = None
    pMg: float | None = None
    drh_kj_per_mol: float | None = None
    ionization_data: dict[str, Any] = Field(default_factory=dict)


class PkaRung(BaseModel):
    """One measured protonation constant, with the species it interconverts."""

    model_config = ConfigDict(extra="allow")

    pka: float
    acid_species: str = ""
    conjugate_base: str = ""
    sigma: float | None = None


class PkaLadderValue(BaseModel):
    """A pKa ladder, and the structure it belongs to.

    Two shapes, because a predicted ladder is a list of numbers while a MEASURED one is a
    list of constants each with its own species, ionic strength and method, and flattening
    the second into the first would throw the provenance away.
    """

    model_config = ConfigDict(extra="allow")

    pkas: list[float] = Field(default_factory=list)
    pka_ladder: list[PkaRung] = Field(default_factory=list)
    site_assignments: list[str] | dict[str, Any] = Field(default_factory=list)
    #: The structure the ladder was measured on. Absent means the GEM's own structure.
    smiles: str | None = None

    @property
    def ladder(self) -> list[float]:
        """The ladder as the compound bridge wants it: bare numbers, descending."""
        if self.pkas:
            return sorted((float(p) for p in self.pkas), reverse=True)
        return sorted((float(r.pka) for r in self.pka_ladder), reverse=True)


class PartnerCouple(BaseModel):
    """The other half of a redox reaction, named and with its own midpoint potential."""

    model_config = ConfigDict(extra="allow")

    couple: str = ""
    e_prime_mv: float


class CarrierPotentialValue(BaseModel):
    r"""A protein-bound redox carrier's midpoint potential, and its partner couple.

    ``oxidized_met_ids`` and ``reduced_met_ids`` are what fix the SIGN of
    :math:`-nF\Delta E'^\circ`, and a record that omits them is priced only where the
    direction follows from chemistry the GEM already states: a higher declared charge is the
    oxidized form, and at equal charge more hydrogen ATOMS is the reduced form, since a
    hydrogen atom at constant net charge is a proton plus an electron. Where neither
    applies the reaction is left unpriced and says why, rather than being priced with a sign
    someone guessed.
    """

    model_config = ConfigDict(extra="allow")

    e_prime_mv: float
    n_electrons: int
    T_K: float | None = None
    pH: float | None = None
    ionic_strength_M: float | None = None
    organism: str = ""
    protein: str = ""
    method: str = ""
    vs: str = ""
    #: How the midpoint potential moves with pH, in mV per pH unit, when the record
    #: measured it. Absent means no correction is applied and the record's own pH is
    #: carried in the provenance rather than a number being invented for another pH.
    e_prime_slope_mv_per_ph: float | None = None
    #: A bare potential in mV, or the named couple that carries one. NOT needed by the
    #: pricing route: the pseudo formation energies make the partner couple price itself
    #: through its own metabolites. Kept because records carry it and it is worth
    #: reporting.
    partner_couple_e_prime_mv: float | PartnerCouple | None = None
    partner_couple: str = ""
    oxidized_met_ids: list[str] = Field(default_factory=list)
    reduced_met_ids: list[str] = Field(default_factory=list)

    @property
    def partner_potential_mv(self) -> float | None:
        """The partner couple's midpoint potential, whichever shape it was written in."""
        if isinstance(self.partner_couple_e_prime_mv, PartnerCouple):
            return self.partner_couple_e_prime_mv.e_prime_mv
        return self.partner_couple_e_prime_mv

    @property
    def partner_name(self) -> str:
        """What the partner couple is called, if the record said."""
        if isinstance(self.partner_couple_e_prime_mv, PartnerCouple):
            return self.partner_couple_e_prime_mv.couple or self.partner_couple
        return self.partner_couple


class PolymerConventionValue(BaseModel):
    """How a polymer species is represented, and the repeat unit that stands for it.

    Two shapes, both real. A residue whose repeat unit carries attachment points has no
    decomposition and is priced by the curator's own dimer difference
    (:attr:`preferred_dimer_difference`); a species the GEM merely CALLS a polymer, such as
    a dolichol-linked oligosaccharide written with its lipid tail spelled out, is an
    ordinary molecule with an ordinary SMILES and goes through the structure route.
    """

    model_config = ConfigDict(extra="allow")

    repeat_unit_smiles: str = ""
    repeat_unit_formula: str = ""
    repeat_unit_charge: int | None = None
    linkage: str = ""
    end_groups: str = ""
    convention_note: str = ""
    #: A definite structure, where the convention resolves to one molecule.
    smiles: str | None = None
    inchi: str | None = None
    inchikey: str | None = None
    formula: str = ""
    charge: int | None = None
    microspecies_note: str = ""
    #: The structure's own pKa ladder, where the curator measured or quoted one.
    pkas: list[float] = Field(default_factory=list)
    #: The curator's own priced value, per metabolite, for a residue the decomposer refuses.
    preferred_dimer_difference: CuratedPrice | None = None
    curated_dgf_prime: CuratedPrice | None = None
    surrogate_route_input: SurrogateRouteInput | None = None

    @property
    def curated_price(self) -> CuratedPrice | None:
        """The curator's own value, whichever of the three places it was written in."""
        if self.preferred_dimer_difference is not None:
            return self.preferred_dimer_difference
        if self.curated_dgf_prime is not None:
            return self.curated_dgf_prime
        if self.surrogate_route_input is not None:
            return self.surrogate_route_input.preferred_dimer_difference
        return None

    @model_validator(mode="after")
    def _something_is_here(self) -> PolymerConventionValue:
        """A convention record has to carry a structure or a value."""
        if self.repeat_unit_smiles or self.smiles or self.curated_price is not None:
            return self
        raise ValueError(
            "a polymer_convention record carries neither a repeat unit, a structure nor a "
            "priced value"
        )


class FormulaCorrectionValue(BaseModel):
    """A GEM formula or charge that is wrong, and what it should be."""

    model_config = ConfigDict(extra="allow")

    proposed_formula: str
    proposed_charge: int | None = None
    derivation: str = ""
    #: A structure for the corrected species, when the curator found one. Without it the
    #: correction fixes the bookkeeping and prices nothing.
    smiles: str | None = None


class NotFoundValue(BaseModel):
    """Nothing was found. The searches are the content of the record."""

    model_config = ConfigDict(extra="allow")


CuratedValue = (
    StructureValue
    | FormationEnergyValue
    | ReactionEnergyValue
    | PkaLadderValue
    | CarrierPotentialValue
    | PolymerConventionValue
    | FormulaCorrectionValue
    | NotFoundValue
)

#: The value model each kind discriminates to. One mapping rather than a tagged union,
#: because the tag lives at the TOP level of the record while the fields it selects live
#: inside ``value``, and copying the tag into the value would change the record's shape on
#: every round trip.
VALUE_MODEL_BY_KIND: dict[CurationKind, type[BaseModel]] = {
    CurationKind.STRUCTURE: StructureValue,
    CurationKind.FORMATION_ENERGY: FormationEnergyValue,
    CurationKind.REACTION_ENERGY: ReactionEnergyValue,
    CurationKind.PKA_LADDER: PkaLadderValue,
    CurationKind.CARRIER_POTENTIAL: CarrierPotentialValue,
    CurationKind.POLYMER_CONVENTION: PolymerConventionValue,
    CurationKind.FORMULA_CORRECTION: FormulaCorrectionValue,
    CurationKind.NOT_FOUND: NotFoundValue,
}


class CuratedThermoRecord(BaseModel):
    """One curation act: what was looked for, what was found, and what may be done with it.

    Strict at the top level and tolerant inside ``value``. The top level is the contract
    between the curator and the pipeline, so an unexpected key there is a mistake worth
    raising on; ``value`` is where a curator writes down whatever the chemistry needed,
    and dropping an extra field there would throw away evidence.
    """

    model_config = ConfigDict(extra="forbid")

    slug: str
    met_ids: list[str]
    #: What the GEM says, echoed for the reader. One value, or one per metabolite where the
    #: record's metabolites are different species. Descriptive only: the pipeline reads the
    #: GEM itself, so a record cannot change a formula by echoing it wrong. Changing one is
    #: what ``kind=formula_correction`` is for.
    gem_name: str | dict[str, str] = ""
    gem_formula: str | dict[str, str] = ""
    gem_charge: int | dict[str, int] | None = None
    compartments: list[str] = Field(default_factory=list)
    blocked_reactions: list[str] = Field(default_factory=list)
    kind: CurationKind
    admission: Admission
    value: CuratedValue = Field(default_factory=NotFoundValue)
    sigma_kj_per_mol: float | None = None
    sources: list[ProvenanceSource] = Field(default_factory=list)
    searches: list[SearchLogEntry] = Field(default_factory=list)
    decomposition_check: DecompositionCheck | None = None
    predictor_check: dict[str, Any] | None = None
    computed_values_recorded_not_admitted: list[dict[str, Any]] = Field(
        default_factory=list
    )
    notes: str = ""

    @model_validator(mode="before")
    @classmethod
    def _value_follows_the_kind(cls, data: Any) -> Any:
        """Parse ``value`` with the model the record's own ``kind`` names."""
        if not isinstance(data, dict):
            return data
        kind = data.get("kind")
        if kind is None:
            return data
        model = VALUE_MODEL_BY_KIND[CurationKind(kind)]
        value = data.get("value")
        if value is None:
            value = {}
        if isinstance(value, BaseModel):
            value = value.model_dump()
        return {**data, "value": model.model_validate(value)}

    @model_validator(mode="after")
    def _admission_and_kind_agree(self) -> CuratedThermoRecord:
        """A record that found nothing may not be admitted, and vice versa."""
        expected = VALUE_MODEL_BY_KIND[self.kind]
        if type(self.value) is not expected:
            raise ValueError(
                f"{self.slug}: kind={self.kind} wants a {expected.__name__} value, got "
                f"{type(self.value).__name__}"
            )
        if self.kind is CurationKind.NOT_FOUND and self.admission is not Admission.NONE:
            raise ValueError(
                f"{self.slug}: kind=not_found cannot carry admission={self.admission}"
            )
        if self.kind is not CurationKind.NOT_FOUND and self.admission is Admission.NONE:
            raise ValueError(
                f"{self.slug}: admission=none needs kind=not_found; a value with no "
                "admission is a value nobody can act on"
            )
        if not self.met_ids:
            raise ValueError(f"{self.slug}: a record names no metabolite")
        return self

    @property
    def admitted(self) -> bool:
        """True when this record may enter the table a model reads."""
        return self.admission in ADMITTED


class CuratedRecordSet(BaseModel):
    """Every record in one directory, indexed by metabolite id.

    A metabolite may be named by more than one record (a structure and its pKa ladder are
    two acts), so the index is a list and the resolver picks by kind.
    """

    model_config = ConfigDict(extra="forbid")

    directory: str
    records: list[CuratedThermoRecord] = Field(default_factory=list)
    #: slug -> the file it was read from, so a report can point at the record it quotes.
    paths: dict[str, str] = Field(default_factory=dict)

    def for_met(self, met_id: str) -> list[CuratedThermoRecord]:
        """Every record naming this metabolite, in slug order."""
        return sorted(
            (r for r in self.records if met_id in r.met_ids), key=lambda r: r.slug
        )

    def of_kind(self, kind: CurationKind) -> list[CuratedThermoRecord]:
        """Every record of one kind, in slug order."""
        return sorted((r for r in self.records if r.kind is kind), key=lambda r: r.slug)


def load_curated_records(directory: str) -> CuratedRecordSet:
    """Read and validate every ``*.json`` in a directory. A bad record raises.

    No file is skipped and no error is swallowed: a record that does not validate is a
    curation act nobody can audit, and loading the other thirteen while dropping it
    silently would put a coverage number in a note that no one could reproduce. The file
    name must match the slug, so the report and the directory listing cannot disagree.
    """
    if not osp.isdir(directory):
        raise FileNotFoundError(f"{directory} is not a directory")
    out = CuratedRecordSet(directory=directory)
    for path in sorted(glob(osp.join(directory, "*.json"))):
        with open(path) as handle:
            payload = json.load(handle)
        try:
            record = CuratedThermoRecord.model_validate(payload)
        except Exception as exc:  # noqa: BLE001 - re-raised with the file that failed
            raise ValueError(f"{path} is not a valid curation record: {exc}") from exc
        stem = osp.splitext(osp.basename(path))[0]
        if record.slug != stem:
            raise ValueError(f"{path} holds slug {record.slug!r}, not {stem!r}")
        if record.slug in out.paths:
            raise ValueError(f"{record.slug} appears twice in {directory}")
        out.paths[record.slug] = path
        out.records.append(record)
    return out


class PricedStructure(BaseModel):
    """What the structure route did with one SMILES at one set of conditions."""

    model_config = ConfigDict(extra="forbid")

    smiles: str
    p_h: float
    ionic_strength_m: float
    temperature_k: float
    p_mg: float
    pkas: list[float] = Field(default_factory=list)
    dgf_prime_kj_per_mol: float | None = None
    sigma_kj_per_mol: float | None = None
    dgf_untransformed_kj_per_mol: float | None = None
    compound_id: int | None = None
    group_decomposition_ok: bool | None = None
    #: How many groups the decomposition actually assigned. Zero is the honest test.
    n_nonzero_groups: int | None = None
    #: Norm of the infinite-variance component of the estimate. Above the tolerance the
    #: structure decomposes into a group nobody trained on and the value is unusable.
    sigma_inf_norm: float | None = None
    #: ``priced``, ``failed_group_decomposition``, ``no_formation_energy``,
    #: ``compound_creation_failed`` or ``compound_creation_raised``. The same vocabulary
    #: ``external_pka_apply_selection.py price`` writes, so the two are comparable.
    outcome: str = ""
    detail: str = ""

    @property
    def priced(self) -> bool:
        """True when the route produced a value."""
        return self.dgf_prime_kj_per_mol is not None


def _conditions_key(
    smiles: str, p_h: float, ionic_strength_m: float, temperature_k: float, p_mg: float
) -> str:
    """The identity of one (structure, conditions) pricing. Rounded, because a float key
    that came from a JSON round trip is not the float that went in.
    """
    return (
        f"{smiles}|pH={p_h:.4f}|I={ionic_strength_m:.6f}"
        f"|T={temperature_k:.4f}|pMg={p_mg:.4f}"
    )


@runtime_checkable
class StructurePricer(Protocol):
    """The structure route, behind an interface, because it needs another environment.

    Component contribution lives in ``equilibrator-ext`` and cobra does not, so the GEM
    driver and the compound bridge cannot run in one process. The protocol is what lets the
    resolver be the same object in both: :class:`EquilibratorStructurePricer` runs the
    route, :class:`PricedStructureTable` replays the artifact it wrote.
    """

    name: str

    def price(
        self,
        smiles: str,
        pkas: list[float],
        p_h: float,
        ionic_strength_m: float,
        temperature_k: float,
        p_mg: float,
    ) -> PricedStructure:
        """Price one structure at one set of conditions."""
        ...


class PricedStructureTable:
    """A :class:`StructurePricer` that replays a JSON of :class:`PricedStructure` rows.

    A (structure, conditions) pair the table does not hold RAISES. There is no fallback to
    another condition set or to an untransformed value: a formation energy at the wrong pH
    is a wrong number, and a run that needs a pairing the artifact does not have needs the
    artifact rebuilt, which is a thing a person should be told rather than a thing a
    default should paper over.
    """

    name = "curated_structure_table"

    def __init__(self, path: str) -> None:
        """Read the priced-structure artifact and index it by structure and conditions."""
        with open(path) as handle:
            payload = json.load(handle)
        self.path = path
        self.rows: dict[str, PricedStructure] = {}
        for row in payload["records"]:
            priced = PricedStructure.model_validate(row)
            self.rows[
                _conditions_key(
                    priced.smiles,
                    priced.p_h,
                    priced.ionic_strength_m,
                    priced.temperature_k,
                    priced.p_mg,
                )
            ] = priced

    def price(
        self,
        smiles: str,
        pkas: list[float],
        p_h: float,
        ionic_strength_m: float,
        temperature_k: float,
        p_mg: float,
    ) -> PricedStructure:
        """The stored outcome for this structure at these conditions."""
        key = _conditions_key(smiles, p_h, ionic_strength_m, temperature_k, p_mg)
        if key not in self.rows:
            raise KeyError(
                f"{self.path} holds no pricing for {key}. Rebuild it with "
                "experiments/026-metabolism-flux/scripts/thermo_curation_price.py under "
                "the equilibrator-ext environment."
            )
        return self.rows[key]


class EquilibratorStructurePricer:
    """The real structure route: create the compound, decompose it, transform it.

    The same four steps ``thermo_structure_attempt.py`` and
    ``external_pka_apply_selection.py price`` take, in one place, so a curated structure is
    priced by the route every other structure is priced by rather than by a second
    implementation that could drift from it. ChemAxon is bypassed throughout: it is not
    licensed here, and the pKa ladder is supplied instead.

    Imports eQuilibrator inside :meth:`price`, so this module still imports in the plain
    torchcell environment where the compound bridge is absent.
    """

    name = "equilibrator_structure_pricer"

    def __init__(self, local_cache_path: str) -> None:
        """Hold the writable compound cache the created compounds are written into."""
        self.local_cache_path = local_cache_path
        self._cc: Any | None = None
        self._with_ladder: Any | None = None
        self._plain: Any | None = None

    def _load(self) -> None:
        from equilibrator_api import ComponentContribution
        from equilibrator_assets.local_compound_cache import LocalCompoundCache

        if self._cc is not None:
            return
        self._cc = ComponentContribution()
        # Two handles on ONE cache file. The guarded one refuses an empty ladder, which is
        # the right behavior for a molecule that HAS an ionizable center; the plain one is
        # used only where the ladder is empty because there is nothing to ionize.
        with_ladder = LocalCompoundCache(pkas_provided_externally=True)
        with_ladder.load_cache(self.local_cache_path)
        plain = LocalCompoundCache()
        plain.load_cache(self.local_cache_path)
        self._with_ladder = with_ladder
        self._plain = plain

    def price(
        self,
        smiles: str,
        pkas: list[float],
        p_h: float,
        ionic_strength_m: float,
        temperature_k: float,
        p_mg: float,
    ) -> PricedStructure:
        """Create the compound from this SMILES and price it at these conditions."""
        from equilibrator_api import Q_

        self._load()
        assert self._cc is not None and self._with_ladder is not None
        assert self._plain is not None
        out = PricedStructure(
            smiles=smiles,
            p_h=p_h,
            ionic_strength_m=ionic_strength_m,
            temperature_k=temperature_k,
            p_mg=p_mg,
            pkas=list(pkas),
        )
        # With a ladder, the guarded cache CREATES the compound from the structure. Without
        # one, the unguarded cache is read only (no ChemAxon license here), so it can look
        # an existing compound up and cannot make a new one. Looking up first is what
        # separates a species whose cache row already exists, where no ladder is needed at
        # all, from one that would have to be created, where the ladder is the missing
        # piece; asking the ladder question first mislabeled the former.
        cache = self._with_ladder if pkas else self._plain
        kwargs: dict[str, Any] = {"mol_format": "smiles", "bypass_chemaxon": True}
        if pkas:
            kwargs["specified_pkas"] = {smiles: list(pkas)}
        try:
            results = cache.get_compounds([smiles], **kwargs)
        except Exception as exc:  # noqa: BLE001 - the bridge raises many types
            out.outcome = "compound_creation_raised"
            out.detail = f"{type(exc).__name__}: {exc}"[:300]
            return out
        compound = results[0].compound if results else None
        if compound is None:
            method = str(results[0].method) if results else ""
            status = str(results[0].status) if results else ""
            centers = ionizable_centers(smiles) if not pkas else []
            if centers:
                out.outcome = "no_pka_ladder"
                out.detail = (
                    "no compound exists for this structure and the record supplies no pKa "
                    f"ladder, while the structure carries ionizable centers "
                    f"({', '.join(centers)}), so creating it would mean inventing its "
                    f"protonation states (method={method} status={status})"
                )
                return out
            out.outcome = (
                "compound_creation_read_only"
                if "read-only" in method
                else "compound_creation_failed"
            )
            out.detail = f"method={method} status={status}"
            return out
        out.compound_id = int(compound.id)
        # A group vector that EXISTS is not a decomposition. Two shapes of empty have both
        # been seen on cache rows: the empty list, and a vector of the right length that is
        # all zeros. Either means no group covered any atom, and component contribution
        # returns nothing for both, so the test is a nonzero COUNT rather than "not None".
        group_vector = getattr(compound, "group_vector", None)
        out.n_nonzero_groups = sum(1 for g in group_vector if g) if group_vector else 0
        out.group_decomposition_ok = out.n_nonzero_groups > 0
        if not out.group_decomposition_ok:
            out.outcome = "failed_group_decomposition"
            out.detail = "the compound's group vector is empty or all zero, so no group covers it"
            return out
        # The same estimability test the bulk route applies: a group that no training
        # compound carries lies in the null space of component contribution, and its
        # coefficient has infinite variance. The energy that comes back for such a compound
        # is a number with no information in it.
        _, _, sigma_inf = self._cc.predictor.preprocess.get_compound_prediction(
            compound
        )
        if sigma_inf is not None:
            import numpy as np

            out.sigma_inf_norm = float(np.linalg.norm(np.asarray(sigma_inf)))
            if out.sigma_inf_norm > SIGMA_INF_TOLERANCE:
                out.outcome = "group_in_null_space"
                out.detail = (
                    "the structure decomposes and one of its groups is in the component "
                    "contribution null space, so the estimate has infinite variance"
                )
                return out
        mu, sigma, _ = self._cc.standard_dg_formation(compound)
        if mu is None:
            out.outcome = "no_formation_energy"
            return out
        correction = compound.transform(
            Q_(p_h), Q_(f"{ionic_strength_m}M"), Q_(f"{temperature_k}K"), Q_(p_mg)
        )
        out.dgf_untransformed_kj_per_mol = float(mu)
        out.dgf_prime_kj_per_mol = float(mu) + float(correction.m_as("kJ/mol"))
        if sigma is not None:
            import numpy as np

            out.sigma_kj_per_mol = float(np.linalg.norm(np.asarray(sigma)))
        out.outcome = "priced"
        return out


#: What counts as a proton this molecule can lose or gain in the (0, 14) window. Copied
#: from ``external_pka_apply_selection.IONIZABLE_CENTER_SMARTS`` so the two paths ask the
#: same question of the same structures; an empty match is a positive statement that the
#: molecule is a single microspecies rather than a tool failure.
IONIZABLE_CENTER_SMARTS: tuple[str, ...] = (
    "[CX3](=O)[OX2H1,OX1-]",  # carboxylic acid or carboxylate
    "[PX4](=O)([OX2H1,OX1-])",  # phosphate
    "[SX4](=O)(=O)[OX2H1,OX1-]",  # sulfonate
    "[OX2H1,OX1-]c",  # phenol or phenolate
    "[NX3;!$(N[C,S]=[O,S,N]);!$(N=*);!$([N+](=O)[O-])]",  # amine
    "[NX4+]",  # quaternary or protonated amine
    "[SX2H]",  # thiol
    "[nX3H]",  # protonated aromatic nitrogen
    "[NX3][CX3](=[NX2])[NX3]",  # guanidine
)


def ionizable_centers(smiles: str) -> list[str]:
    """Which ionizable-center patterns a structure matches, by SMARTS.

    An empty result is a positive statement: the molecule has no proton to lose or gain in
    the window, so an empty pKa ladder is its answer and not a tool failure.
    """
    from rdkit import Chem, RDLogger

    # ``rdkit-stubs`` does not declare ``DisableLog``: it is re-exported from the compiled
    # rdBase module at runtime. Same accessor as ``kinetics_pipeline._disable_rdkit_logs``.
    disable = cast(Callable[[str], None], getattr(RDLogger, "DisableLog"))  # noqa: B009
    disable("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ["unparseable"]
    found = []
    for smarts in IONIZABLE_CENTER_SMARTS:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is not None and mol.HasSubstructMatch(pattern):
            found.append(smarts)
    return found


def decomposition_refused(record: CuratedThermoRecord) -> bool:
    """True when the curator MEASURED that this structure does not decompose.

    A measured decomposition failure is evidence, and it is stronger evidence than
    re-running the same decomposer through a compound cache that resolves by InChIKey and
    can return a stored empty group vector instead. When a record carries it, the structure
    route is not run at all and the reason is the one the curator recorded, with their error
    text.
    """
    check = record.decomposition_check
    return check is not None and check.attempted and check.decomposes is False


def structure_for(record: CuratedThermoRecord, met_id: str) -> tuple[str | None, str]:
    """The SMILES and the declared formula one record supplies for one metabolite."""
    value = record.value
    if isinstance(value, StructureValue):
        for entry in value.per_metabolite:
            if entry.met_id == met_id:
                return entry.smiles, entry.formula
        return value.smiles, value.formula
    if isinstance(value, PolymerConventionValue):
        if value.smiles:
            return value.smiles, value.formula
        return value.repeat_unit_smiles or None, value.repeat_unit_formula
    if isinstance(value, PkaLadderValue):
        return value.smiles, ""
    if isinstance(value, FormulaCorrectionValue):
        return value.smiles, value.proposed_formula
    return None, ""


def ladder_for(
    record: CuratedThermoRecord, records: CuratedRecordSet
) -> tuple[list[float], str]:
    """The curated pKa ladder that belongs with one record, and the slug it came from.

    A ladder is a record of its own, so a structure record and the ladder that completes it
    are joined here, on the metabolite they share.
    """
    if isinstance(record.value, PkaLadderValue):
        return record.value.ladder, record.slug
    own = getattr(record.value, "pkas", None)
    if own:
        return sorted((float(p) for p in own), reverse=True), record.slug
    for met_id in record.met_ids:
        for other in records.for_met(met_id):
            if other.kind is CurationKind.PKA_LADDER and other.admitted:
                assert isinstance(other.value, PkaLadderValue)
                return other.value.ladder, other.slug
    return [], ""


class PricingRequest(BaseModel):
    """One (structure, ladder, conditions) the structure route has to be run on.

    The handshake between the two environments. The GEM side knows which structures are
    needed and at which compartment conditions and cannot run component contribution; the
    compound-bridge side can run it and cannot read the GEM. Deriving the request list from
    the SAME resolver that will consume the answers is what keeps the two from drifting.
    """

    model_config = ConfigDict(extra="forbid")

    smiles: str
    pkas: list[float] = Field(default_factory=list)
    p_h: float
    ionic_strength_m: float
    temperature_k: float
    p_mg: float
    #: Who wants it, for the report only.
    slugs: list[str] = Field(default_factory=list)
    met_ids: list[str] = Field(default_factory=list)

    @property
    def key(self) -> str:
        """The identity this request will be answered under."""
        return _conditions_key(
            self.smiles, self.p_h, self.ionic_strength_m, self.temperature_k, self.p_mg
        )


def requests_digest(requests: list[PricingRequest]) -> str:
    """A content hash of a request SET, stable across runs.

    The file the requests are written to carries a build timestamp, so hashing the file
    would report every rerun as a change and the staleness gate would cry wolf until
    someone turned it off. This hashes what the answers actually depend on: the structures,
    the ladders and the conditions.
    """
    import hashlib

    payload = json.dumps(sorted(r.key + "|" + repr(r.pkas) for r in requests))
    return hashlib.sha256(payload.encode()).hexdigest()


def pricing_requests(
    records: CuratedRecordSet,
    identities: list[MetaboliteIdentity],
    conditions: CompartmentConditions,
) -> list[PricingRequest]:
    """Every structure route call an admitted record implies, deduplicated.

    A record covering three compartment copies of one species implies three calls, because
    the transform is per compartment; a record whose structure carries an attachment point
    implies none, because the structure route cannot be run on a residue.
    """
    by_key: dict[str, PricingRequest] = {}
    by_id = {m.met_id: m for m in identities}
    for record in records.records:
        if not record.admitted:
            continue
        for met_id in record.met_ids:
            met = by_id.get(met_id)
            if met is None:
                continue
            smiles, _ = structure_for(record, met_id)
            if smiles is None or ATTACHMENT_POINT in smiles:
                continue
            if decomposition_refused(record):
                continue
            pkas, _ = ladder_for(record, records)
            p_h, ionic, p_mg = conditions.for_compartment(met.compartment)
            request = PricingRequest(
                smiles=smiles,
                pkas=pkas,
                p_h=p_h,
                ionic_strength_m=ionic,
                temperature_k=conditions.temperature_k,
                p_mg=p_mg,
                slugs=[record.slug],
                met_ids=[met_id],
            )
            existing = by_key.get(request.key)
            if existing is None:
                by_key[request.key] = request
                continue
            if record.slug not in existing.slugs:
                existing.slugs.append(record.slug)
            if met_id not in existing.met_ids:
                existing.met_ids.append(met_id)
    return [by_key[k] for k in sorted(by_key)]


class FormulaOverride(BaseModel):
    """One GEM formula or charge a curated record replaced, and with what."""

    model_config = ConfigDict(extra="forbid")

    met_id: str
    slug: str
    gem_formula: str
    gem_charge: int
    corrected_formula: str
    corrected_charge: int
    derivation: str = ""


def apply_formula_corrections(
    identities: list[MetaboliteIdentity], records: CuratedRecordSet
) -> tuple[list[MetaboliteIdentity], list[FormulaOverride]]:
    """Rewrite the identities an admitted ``formula_correction`` record corrects.

    Run this before anything reads a formula: the composition check that decides whether a
    structure is the right molecule compares against the GEM's formula, and comparing
    against a formula the curator has shown to be impossible would reject the correct
    structure. Only ADMITTED corrections are applied; a provisional one is reported and
    changes nothing.

    Returns the new identity list and one :class:`FormulaOverride` per metabolite changed,
    so the run's record says the GEM was overridden rather than quietly disagreeing with it.
    """
    corrections: dict[str, CuratedThermoRecord] = {}
    for record in records.of_kind(CurationKind.FORMULA_CORRECTION):
        if not record.admitted:
            continue
        for met_id in record.met_ids:
            corrections[met_id] = record
    if not corrections:
        return list(identities), []
    out: list[MetaboliteIdentity] = []
    overrides: list[FormulaOverride] = []
    for identity in identities:
        correction = corrections.get(identity.met_id)
        if correction is None:
            out.append(identity)
            continue
        value = correction.value
        assert isinstance(value, FormulaCorrectionValue)
        charge = (
            identity.charge if value.proposed_charge is None else value.proposed_charge
        )
        out.append(
            identity.model_copy(
                update={"formula": value.proposed_formula, "charge": charge}
            )
        )
        overrides.append(
            FormulaOverride(
                met_id=identity.met_id,
                slug=correction.slug,
                gem_formula=identity.formula,
                gem_charge=identity.charge,
                corrected_formula=value.proposed_formula,
                corrected_charge=charge,
                derivation=value.derivation,
            )
        )
    return out, overrides


#: Above this norm the infinite-variance component of an estimate is nonzero, which means
#: a group in the decomposition lies in the null space of the component-contribution
#: training set. Copied as a constant so the compound route and the reaction route apply
#: one threshold.
SIGMA_INF_TOLERANCE = 1e-8

#: An attachment point in a structure. A species whose structure carries one is a residue
#: inside a macromolecule and has no formation energy of its own, so the structure route
#: must not be run on it. Same test and same reason as ``thermo_structure_attempt.py``.
ATTACHMENT_POINT = "*"

#: A charge written into a formula string: a trailing sign with optional digits AFTER it
#: (``C6H4NO3-``, ``C6H4NO3-2``) or the parenthesized form (``C42H36FeN4O16(8-)``). Digits
#: BEFORE a sign are deliberately not stripped: in ``C6H4NO3-`` the 3 is the oxygen count,
#: and a rule that read "3-" as the charge turned the anion of nitrophenol into a molecule
#: with one oxygen and rejected the curated structure as a different compound.
_CHARGE_SUFFIX = re.compile(r"(\(\d*[+-]\)|[+-]\d*)$")


def _strip_charge_suffix(formula: str) -> str:
    """``C6H4NO3-`` -> ``C6H4NO3``. A curator writes the charge into the formula string."""
    return _CHARGE_SUFFIX.sub("", (formula or "").strip())


def _heavy_composition(formula: str) -> dict[str, float] | None:
    """Heavy-atom composition, or ``None`` when the formula cannot bear one.

    Hydrogens are dropped because a structure is one microspecies, whose proton count
    depends on pH, while the GEM's depends on its own charge convention. Comparing them
    would reject every correct structure of an ionizable compound.
    """
    text = _strip_charge_suffix(formula)
    if not text or any(token in text for token in PLACEHOLDER_TOKENS):
        return None
    try:
        counts = element_counts(text)
    except ValueError:
        return None
    return {element: n for element, n in counts.items() if element != "H"}


class CuratedResolver:
    """The stage that prices metabolites from admitted curated records.

    Sits at the TOP of the precedence order: a curator looked at this species on purpose,
    which is stronger evidence than any automatic route, and a record that disagrees with
    an automatic value should win rather than be shadowed by it.

    Prices nothing it cannot justify. A structure whose heavy-atom composition disagrees
    with the GEM's (corrected) formula is a different molecule and is refused with a stated
    reason; a structure the group decomposer refuses keeps
    :attr:`~torchcell.metabolism.thermo_pipeline.UnpricedReason.GROUP_DECOMPOSITION_FAILED`;
    a record whose admission is ``computed`` or ``provisional`` is reported under
    :attr:`~torchcell.metabolism.thermo_pipeline.UnpricedReason.RECORDED_NOT_ADMITTED` and
    never priced.
    """

    name = "curated_records"

    #: Kinds whose species are priced at the REACTION level and therefore never carry a
    #: formation energy. A carrier is NOT one of them: it gets a pair of pseudo formation
    #: energies whose difference is the measured potential, which prices every reaction
    #: that conserves it without a partner couple.
    REACTION_LEVEL_KINDS = frozenset({CurationKind.REACTION_ENERGY})

    def __init__(
        self, records: CuratedRecordSet, structure_pricer: StructurePricer | None = None
    ) -> None:
        """Hold the records, and the pricer the structure routes need.

        Raises when the set holds an admitted structure and no pricer was given, rather
        than dropping those records: a run that silently prices thirteen of fourteen
        records is a run whose coverage number means nothing.
        """
        self.records = records
        self.structure_pricer = structure_pricer
        needs_pricer = [
            r.slug
            for r in records.records
            if r.admitted
            and any(
                smiles is not None and ATTACHMENT_POINT not in smiles
                for smiles in (self._structure_of(r, met_id)[0] for met_id in r.met_ids)
            )
        ]
        if needs_pricer and structure_pricer is None:
            raise ValueError(
                f"{len(needs_pricer)} admitted records need the structure route "
                f"({', '.join(needs_pricer)}) and no StructurePricer was given"
            )

    def _ladder_for(self, record: CuratedThermoRecord) -> tuple[list[float], str]:
        """The curated pKa ladder that belongs with this record, and where it came from."""
        return ladder_for(record, self.records)

    def _structure_of(
        self, record: CuratedThermoRecord, met_id: str
    ) -> tuple[str | None, str]:
        """The SMILES and the declared formula this record supplies for one metabolite."""
        return structure_for(record, met_id)

    def _curated_price_of(self, record: CuratedThermoRecord) -> CuratedPrice | None:
        """The curator's own priced value, whichever of the two fields carries it."""
        value = record.value
        if isinstance(value, PolymerConventionValue):
            return value.curated_price
        if isinstance(value, StructureValue):
            return value.curated_dgf_prime
        return None

    def _price_curated_value(
        self,
        record: CuratedThermoRecord,
        price: CuratedPrice,
        met: MetaboliteIdentity,
        conditions: CompartmentConditions,
    ) -> MetabolitePrice:
        """Take the curator's own transformed value, after checking its conditions.

        The conditions check is not a formality: the record states the pH its number was
        transformed at, and a formation energy at another compartment's pH is a different
        number by several kJ/mol per proton. A mismatch is refused rather than corrected,
        because correcting it would need the species' proton count, which is the thing a
        residue with attachment points does not have a definite one of.
        """
        value = price.dgf_prime_kj_per_mol.get(met.met_id)
        p_h, ionic, p_mg = conditions.for_compartment(met.compartment)
        if value is None:
            return MetabolitePrice(
                met_id=met.met_id,
                reason=UnpricedReason.NO_STRUCTURE_AVAILABLE,
                reason_is_final=True,
                detail=f"{record.slug}: the curated value names no price for this species",
                provenance={"slug": record.slug},
            )
        stated = price.conditions
        if stated is not None:
            stated_p_h = stated.for_met(met.met_id)
            mismatch = [
                f"pH {stated_p_h} against {p_h}"
                for _ in [0]
                if stated_p_h is not None and abs(stated_p_h - p_h) > 1e-6
            ]
            if (
                stated.ionic_strength_m is not None
                and abs(stated.ionic_strength_m - ionic) > 1e-9
            ):
                mismatch.append(f"I {stated.ionic_strength_m} against {ionic}")
            if (
                stated.temperature_k is not None
                and abs(stated.temperature_k - conditions.temperature_k) > 1e-6
            ):
                mismatch.append(
                    f"T {stated.temperature_k} against {conditions.temperature_k}"
                )
            if mismatch:
                return MetabolitePrice(
                    met_id=met.met_id,
                    reason=UnpricedReason.RECORDED_NOT_ADMITTED,
                    reason_is_final=True,
                    detail=(
                        f"{record.slug}: the curated value was transformed at "
                        + "; ".join(mismatch)
                        + ", and a transformed formation energy is not portable"
                    ),
                    provenance={"slug": record.slug},
                )
        return MetabolitePrice(
            met_id=met.met_id,
            dgf_prime_kj_per_mol=float(value),
            sigma_kj_per_mol=(
                price.sigma_kj_per_mol
                if price.sigma_kj_per_mol is not None
                else record.sigma_kj_per_mol
            ),
            route=self._route_of(record),
            p_h=p_h,
            ionic_strength_m=ionic,
            p_mg=p_mg,
            temperature_k=conditions.temperature_k,
            detail=f"{record.slug}: the curator's own value, {price.rule[:120]}",
            provenance={
                "slug": record.slug,
                "kind": str(record.kind),
                "admission": str(record.admission),
                "rule": price.rule,
                "sources": [s.source_url for s in record.sources],
                "source_sha256": [s.sha256 for s in record.sources],
            },
        )

    def _route_of(self, record: CuratedThermoRecord) -> ThermoRoute:
        """Which curated route this record's kind is."""
        if record.kind is CurationKind.PKA_LADDER:
            return ThermoRoute.CURATED_PKA
        if record.kind is CurationKind.FORMULA_CORRECTION:
            return ThermoRoute.CURATED_FORMULA
        return ThermoRoute.CURATED_STRUCTURE

    def _price_structure(
        self,
        record: CuratedThermoRecord,
        met: MetaboliteIdentity,
        conditions: CompartmentConditions,
    ) -> MetabolitePrice:
        """Run the structure route on this record's SMILES for one metabolite."""
        smiles, declared = self._structure_of(record, met.met_id)
        if smiles is None:
            # NOT_ATTEMPTED rather than a reason of this stage's own. A record that
            # supplies no structure has not learned anything new about WHY the species is
            # unpriced, and the reason the later diagnosis stages already state (a
            # decomposition that fails, a structure nobody has) is the true one. Claiming
            # "no structure available" here would downgrade a measured decomposition
            # failure into a retrieval problem and move the coverage ceiling with it.
            return MetabolitePrice(
                met_id=met.met_id,
                reason=UnpricedReason.NOT_ATTEMPTED,
                detail=(
                    f"{record.slug}: kind={record.kind} supplies no structure for this "
                    "species"
                ),
                provenance={"slug": record.slug},
            )
        # The measured decomposition failure is checked BEFORE the attachment point, and
        # the order matters: a polymer repeat unit carries attachment points AND is
        # something a dimer difference can price, which mannan demonstrates. Calling it
        # PLACEHOLDER_STRUCTURE would put it in the permanently closed class and assert
        # that nothing can ever price it, which the record next to it disproves.
        if not decomposition_refused(record) and ATTACHMENT_POINT in smiles:
            return MetabolitePrice(
                met_id=met.met_id,
                reason=UnpricedReason.PLACEHOLDER_STRUCTURE,
                reason_is_final=True,
                detail=(
                    f"{record.slug}: the curated structure carries an attachment point, so "
                    "it is a residue inside a macromolecule and the structure route cannot "
                    "price it; a curated value is what such a species needs"
                ),
                provenance={"slug": record.slug, "smiles": smiles},
            )
        if decomposition_refused(record):
            check = record.decomposition_check
            assert check is not None
            return MetabolitePrice(
                met_id=met.met_id,
                reason=UnpricedReason.GROUP_DECOMPOSITION_FAILED,
                reason_is_final=True,
                detail=(
                    f"{record.slug}: the curator ran the group decomposer on this "
                    f"structure and it refused it. {(check.error or '')[:200]}"
                ),
                provenance={
                    "slug": record.slug,
                    "smiles": smiles,
                    "decomposition_error": check.error,
                    "smiles_tried": [a.smiles for a in check.smiles_tried],
                },
            )
        wanted = _heavy_composition(met.formula)
        found = _heavy_composition(declared)
        if wanted is not None and found is not None and wanted != found:
            return MetabolitePrice(
                met_id=met.met_id,
                reason=UnpricedReason.NO_STRUCTURE_AVAILABLE,
                reason_is_final=True,
                detail=(
                    f"{record.slug}: the curated structure's composition {declared} is not "
                    f"the GEM's {met.formula}; pricing it would price a different molecule"
                ),
                provenance={"slug": record.slug, "curated_formula": declared},
            )
        pkas, ladder_slug = self._ladder_for(record)
        p_h, ionic, p_mg = conditions.for_compartment(met.compartment)
        assert self.structure_pricer is not None
        priced = self.structure_pricer.price(
            smiles, pkas, p_h, ionic, conditions.temperature_k, p_mg
        )
        provenance: dict[str, object] = {
            "slug": record.slug,
            "kind": str(record.kind),
            "admission": str(record.admission),
            "smiles": smiles,
            "pkas": pkas,
            "pka_ladder_slug": ladder_slug,
            "pricer": getattr(self.structure_pricer, "name", ""),
            "outcome": priced.outcome,
            "compound_id": priced.compound_id,
            "composition_checked": wanted is not None and found is not None,
            "sources": [s.source_url for s in record.sources],
            "source_sha256": [s.sha256 for s in record.sources],
        }
        if not priced.priced:
            # What the outcome means for the WORK QUEUE, which is what the reason is for.
            # A structure that reaches the decomposer and is refused, and one whose cache
            # row carries an empty group vector so component contribution returns nothing,
            # both need a decomposition; only a missing ladder needs a pKa route.
            reason = {
                "failed_group_decomposition": UnpricedReason.GROUP_DECOMPOSITION_FAILED,
                "no_formation_energy": UnpricedReason.GROUP_DECOMPOSITION_FAILED,
                "compound_creation_raised": UnpricedReason.GROUP_DECOMPOSITION_FAILED,
                "compound_creation_failed": UnpricedReason.GROUP_DECOMPOSITION_FAILED,
                "no_pka_ladder": UnpricedReason.NO_PKA_PREDICTION,
                "compound_creation_read_only": UnpricedReason.NO_PKA_PREDICTION,
                "group_in_null_space": UnpricedReason.GROUP_IN_NULL_SPACE,
            }[priced.outcome]
            return MetabolitePrice(
                met_id=met.met_id,
                reason=reason,
                reason_is_final=True,
                detail=(
                    f"{record.slug}: structure route said {priced.outcome}. "
                    f"{priced.detail[:200]}"
                ),
                provenance=provenance,
            )
        return MetabolitePrice(
            met_id=met.met_id,
            dgf_prime_kj_per_mol=priced.dgf_prime_kj_per_mol,
            sigma_kj_per_mol=(
                record.sigma_kj_per_mol
                if record.sigma_kj_per_mol is not None
                else priced.sigma_kj_per_mol
            ),
            route=self._route_of(record),
            p_h=p_h,
            ionic_strength_m=ionic,
            p_mg=p_mg,
            temperature_k=conditions.temperature_k,
            detail=f"{record.slug} via the curated structure route",
            provenance=provenance,
        )

    def _price_formation(
        self,
        record: CuratedThermoRecord,
        met: MetaboliteIdentity,
        conditions: CompartmentConditions,
    ) -> MetabolitePrice:
        """Move a curated formation energy to this metabolite's compartment conditions."""
        value = record.value
        assert isinstance(value, FormationEnergyValue)
        p_h, ionic, p_mg = conditions.for_compartment(met.compartment)
        charge = met.charge if value.charge is None else value.charge
        if value.num_protons is None:
            protons = _heavy_free_protons(met.formula)
        else:
            protons = value.num_protons
        if protons is None:
            return MetabolitePrice(
                met_id=met.met_id,
                reason=UnpricedReason.RECORDED_NOT_ADMITTED,
                reason_is_final=True,
                detail=(
                    f"{record.slug}: the value cannot be transformed, because neither the "
                    f"record nor the GEM formula {met.formula!r} gives a proton count"
                ),
                provenance={"slug": record.slug},
            )
        # Undo the transform the record was reported under, then apply this compartment's.
        # A value already reported as standard has nothing to undo.
        if value.standard_or_transformed == "transformed":
            if value.pH is None or value.ionic_strength_M is None:
                return MetabolitePrice(
                    met_id=met.met_id,
                    reason=UnpricedReason.RECORDED_NOT_ADMITTED,
                    reason_is_final=True,
                    detail=(
                        f"{record.slug}: a transformed value with no pH or ionic strength "
                        "cannot be moved to another compartment"
                    ),
                    provenance={"slug": record.slug},
                )
            back = legendre_transform_kj(
                charge=charge,
                num_protons=protons,
                p_h=value.pH,
                ionic_strength_m=value.ionic_strength_M,
                temperature_k=value.T_K,
            )
        else:
            back = 0.0
        standard = value.dgf_kj_per_mol - back
        forward = legendre_transform_kj(
            charge=charge,
            num_protons=protons,
            p_h=p_h,
            ionic_strength_m=ionic,
            temperature_k=conditions.temperature_k,
        )
        return MetabolitePrice(
            met_id=met.met_id,
            dgf_prime_kj_per_mol=standard + forward,
            sigma_kj_per_mol=record.sigma_kj_per_mol,
            route=ThermoRoute.CURATED_FORMATION,
            p_h=p_h,
            ionic_strength_m=ionic,
            p_mg=p_mg,
            temperature_k=conditions.temperature_k,
            detail=f"{record.slug}, {value.standard_or_transformed} value transformed",
            provenance={
                "slug": record.slug,
                "admission": str(record.admission),
                "reported_kj_per_mol": value.dgf_kj_per_mol,
                "reported_state": value.standard_or_transformed,
                "reported_p_h": value.pH,
                "reported_ionic_strength_m": value.ionic_strength_M,
                "reported_temperature_k": value.T_K,
                "standard_kj_per_mol": standard,
                "charge_used": charge,
                "num_protons_used": protons,
                "charge_source": "record" if value.charge is not None else "gem",
                "sources": [s.source_url for s in record.sources],
                "source_sha256": [s.sha256 for s in record.sources],
            },
        )

    def resolve(
        self, metabolites: list[MetaboliteIdentity], conditions: CompartmentConditions
    ) -> list[MetabolitePrice]:
        """Price every metabolite an admitted record covers, and report the rest."""
        offered = {m.met_id: m for m in metabolites}
        out: list[MetabolitePrice] = []
        for met in metabolites:
            covering = self.records.for_met(met.met_id)
            if not covering:
                out.append(
                    MetabolitePrice(
                        met_id=met.met_id, reason=UnpricedReason.NOT_ATTEMPTED
                    )
                )
                continue
            out.append(self._resolve_one(covering, met, conditions, offered))
        return out

    def _price_carrier(
        self,
        record: CuratedThermoRecord,
        met: MetaboliteIdentity,
        conditions: CompartmentConditions,
        offered: dict[str, MetaboliteIdentity],
    ) -> MetabolitePrice:
        r"""Give a redox carrier couple a pair of PSEUDO formation energies.

        A midpoint potential is a property of the couple, and the two forms have no
        separate formation energy. They do, however, have a well defined DIFFERENCE:
        :math:`\Delta_f G'(\mathrm{red})-\Delta_f G'(\mathrm{ox})=-nFE'^\circ`. Fixing the
        oxidized form at zero and the reduced form at :math:`-nFE'^\circ` makes the
        ordinary stoichiometric sum give the right answer for EVERY reaction in which the
        carrier is conserved, because the arbitrary gauge cancels there, and it needs no
        partner couple and no electron chemical potential: the partner is priced by its own
        metabolites like anything else.

        The gauge is arbitrary and its cancellation is what makes it safe, so a reaction
        that does NOT conserve the carrier must not be priced from these numbers.
        :func:`~torchcell.metabolism.thermo_pipeline.carrier_conservation_violations`
        catches those and the pipeline refuses them.

        ``E'^\circ`` is moved to the compartment's pH only when the record states a slope;
        otherwise the record's own pH is carried in the provenance and no correction is
        invented.
        """
        value = record.value
        assert isinstance(value, CarrierPotentialValue)
        partner = [
            offered[other]
            for other in record.met_ids
            if other != met.met_id
            and other in offered
            and offered[other].compartment == met.compartment
        ]
        if len(partner) != 1:
            return MetabolitePrice(
                met_id=met.met_id,
                reason=UnpricedReason.NOT_ATTEMPTED,
                detail=(
                    f"{record.slug}: the other half of the couple is not among the "
                    f"metabolites this stage was handed in compartment {met.compartment!r}"
                ),
                provenance={"slug": record.slug},
            )
        direction = carrier_direction(value, met, partner[0])
        if direction is None:
            return MetabolitePrice(
                met_id=met.met_id,
                reason=UnpricedReason.NOT_ATTEMPTED,
                detail=(
                    f"{record.slug}: neither the record nor the GEM says which of "
                    f"{met.met_id} and {partner[0].met_id} is the oxidized form, so the "
                    "sign of -nF E' cannot be fixed"
                ),
                provenance={"slug": record.slug},
            )
        p_h, ionic, p_mg = conditions.for_compartment(met.compartment)
        e_prime_mv = value.e_prime_mv
        slope = value.e_prime_slope_mv_per_ph
        if slope is not None and value.pH is not None:
            e_prime_mv = e_prime_mv + slope * (p_h - value.pH)
        is_reduced = met.met_id == direction.reduced_met_id
        energy = (
            -value.n_electrons * FARADAY_KJ_PER_MOL_PER_V * e_prime_mv / 1000.0
            if is_reduced
            else 0.0
        )
        return MetabolitePrice(
            met_id=met.met_id,
            dgf_prime_kj_per_mol=energy,
            sigma_kj_per_mol=record.sigma_kj_per_mol if is_reduced else 0.0,
            route=ThermoRoute.CARRIER_POTENTIAL,
            p_h=p_h,
            ionic_strength_m=ionic,
            p_mg=p_mg,
            temperature_k=conditions.temperature_k,
            detail=(
                f"{record.slug}: pseudo formation energy, "
                f"{'reduced' if is_reduced else 'oxidized'} form of the couple"
            ),
            provenance={
                "slug": record.slug,
                "carrier_group": record.slug,
                "admission": str(record.admission),
                "form": "reduced" if is_reduced else "oxidized",
                "gauge": "dGf'(oxidized) = 0; only the difference is physical",
                "e_prime_mv_reported": value.e_prime_mv,
                "e_prime_mv_used": e_prime_mv,
                "e_prime_slope_mv_per_ph": slope,
                "reported_p_h": value.pH,
                "compartment_p_h": p_h,
                "n_electrons": value.n_electrons,
                "reduced_met_id": direction.reduced_met_id,
                "oxidized_met_id": direction.oxidized_met_id,
                "direction_decided_by": direction.decided_by,
                "organism": value.organism,
                "protein": value.protein,
                "method": value.method,
                "vs": value.vs,
                "sources": [s.source_url for s in record.sources],
                "source_sha256": [s.sha256 for s in record.sources],
            },
        )

    def _resolve_one(
        self,
        covering: list[CuratedThermoRecord],
        met: MetaboliteIdentity,
        conditions: CompartmentConditions,
        offered: dict[str, MetaboliteIdentity],
    ) -> MetabolitePrice:
        """The best answer the records covering one metabolite support."""
        admitted = [r for r in covering if r.admitted]
        for record in admitted:
            if record.kind is CurationKind.CARRIER_POTENTIAL:
                return self._price_carrier(record, met, conditions, offered)
            if record.kind is not CurationKind.REACTION_ENERGY:
                continue
            return MetabolitePrice(
                met_id=met.met_id,
                reason=UnpricedReason.PRICED_AT_REACTION_LEVEL,
                reason_is_final=True,
                detail=(
                    f"{record.slug}: a {record.kind} record prices the reactions this "
                    "species appears in, so the species has no formation energy"
                ),
                provenance={"slug": record.slug, "kind": str(record.kind)},
            )
        for record in admitted:
            if record.kind is CurationKind.FORMATION_ENERGY:
                return self._price_formation(record, met, conditions)
        for record in admitted:
            price = self._curated_price_of(record)
            if price is not None and met.met_id in price.dgf_prime_kj_per_mol:
                return self._price_curated_value(record, price, met, conditions)
        for record in admitted:
            if self._structure_of(record, met.met_id)[0] is not None:
                return self._price_structure(record, met, conditions)
        for record in covering:
            if record.kind is CurationKind.NOT_FOUND:
                # A searched-and-found-nothing record must not become a reason of its own:
                # the honest reason is the structural one the pipeline already computes,
                # and this record is the evidence that it will not change.
                return MetabolitePrice(
                    met_id=met.met_id,
                    reason=UnpricedReason.NOT_ATTEMPTED,
                    detail=(
                        f"{record.slug}: {len(record.searches)} searches, nothing found"
                    ),
                    provenance={"slug": record.slug},
                )
        if admitted:
            # An ADMITTED record covers this species and none of its routes reached it: a
            # ladder with no structure to put it on, a per-metabolite entry whose structure
            # the curator could not find. That is not the "recorded but not admitted"
            # bucket, and calling it that would misreport an admitted record. The reason
            # stays with the diagnosis stages.
            record = admitted[0]
            return MetabolitePrice(
                met_id=met.met_id,
                reason=UnpricedReason.NOT_ATTEMPTED,
                detail=(
                    f"{record.slug}: admitted, and none of its routes reaches this "
                    "species"
                ),
                provenance={"slug": record.slug, "kind": str(record.kind)},
            )
        record = covering[0]
        return MetabolitePrice(
            met_id=met.met_id,
            reason=UnpricedReason.RECORDED_NOT_ADMITTED,
            reason_is_final=True,
            detail=(
                f"{record.slug}: kind={record.kind} admission={record.admission}, "
                "recorded and reported but not admitted to the priced table"
            ),
            provenance={
                "slug": record.slug,
                "kind": str(record.kind),
                "admission": str(record.admission),
                "sources": [s.source_url for s in record.sources],
            },
        )


def _heavy_free_protons(formula: str) -> float | None:
    """The hydrogen count of a formula, or ``None`` when it has no definite one."""
    text = _strip_charge_suffix(formula)
    if not text or any(token in text for token in PLACEHOLDER_TOKENS):
        return None
    try:
        counts = element_counts(text)
    except ValueError:
        return None
    return counts.get("H", 0.0)


_TOKEN = re.compile(r"([A-Z][a-z]?|\*)(\d*\.?\d*)")


def _formula_tokens(formula: str) -> dict[str, float] | None:
    """Every token of a formula INCLUDING a placeholder, or ``None`` if it does not parse.

    Not a composition: ``R4`` is counted as four of something unspecified. Only for
    comparing two formulas of the same species written the same way.
    """
    text = _strip_charge_suffix(formula)
    if not text:
        return None
    out: dict[str, float] = {}
    consumed = 0
    for element, count in _TOKEN.findall(text):
        consumed += len(element) + len(count)
        out[element] = out.get(element, 0.0) + (float(count) if count else 1.0)
    if consumed != len(text):
        return None
    return out


class CarrierDirection(BaseModel):
    """Which species of a carrier couple is reduced, and how that was decided."""

    model_config = ConfigDict(extra="forbid")

    reduced_met_id: str
    oxidized_met_id: str
    #: ``record``, ``charge`` or ``hydrogen_at_equal_charge``.
    decided_by: str


def carrier_direction(
    value: CarrierPotentialValue, first: MetaboliteIdentity, second: MetaboliteIdentity
) -> CarrierDirection | None:
    """Which of two species of one carrier is the reduced form, or ``None`` if unknowable.

    Three rules, in order, and each is chemistry rather than a naming convention:

    1. the record says so;
    2. the declared charges differ, and removing electrons is what raises a charge, so the
       LOWER charge is the reduced form;
    3. the charges are equal and the formulas differ in hydrogen only, so the extra
       hydrogen is a hydrogen ATOM, which is a proton plus an electron, and the species
       with MORE hydrogen is the reduced form.

    Where none applies the direction is unknown and this returns ``None``. That happens
    when a GEM gives both forms of a carrier the same charge and no formula, which is a
    modeling shortcut that erases the difference between them.
    """
    reduced = set(value.reduced_met_ids)
    oxidized = set(value.oxidized_met_ids)
    for a, b in ((first, second), (second, first)):
        if a.met_id in reduced and b.met_id in oxidized:
            return CarrierDirection(
                reduced_met_id=a.met_id, oxidized_met_id=b.met_id, decided_by="record"
            )
    if first.charge != second.charge:
        low, high = (first, second) if first.charge < second.charge else (second, first)
        return CarrierDirection(
            reduced_met_id=low.met_id, oxidized_met_id=high.met_id, decided_by="charge"
        )
    # A placeholder token is tolerated HERE and nowhere else: this comparison asks only
    # whether two formula strings differ in hydrogen alone, which is a question about the
    # two strings rather than about a composition, so ``C42H44FeN8O8S2R4`` against
    # ``C42H45FeN8O8S2R4`` is answerable even though neither has a composition.
    tokens = [_formula_tokens(first.formula), _formula_tokens(second.formula)]
    if tokens[0] is None or tokens[1] is None:
        return None
    counts = [tokens[0].pop("H", 0.0), tokens[1].pop("H", 0.0)]
    if counts[0] == counts[1] or tokens[0] != tokens[1]:
        return None
    more, less = (first, second) if counts[0] > counts[1] else (second, first)
    return CarrierDirection(
        reduced_met_id=more.met_id,
        oxidized_met_id=less.met_id,
        decided_by="hydrogen_at_equal_charge",
    )


class CuratedReactionPricer:
    r"""Price a reaction from a curated whole-reaction energy.

    Some reaction energies are measured as reaction energies and cannot be factored into
    formation energies. This prices those, reading a ``reaction_energy`` record that names
    the reaction.

    Redox carriers are NOT here. They were, priced as
    :math:`-nF\Delta E'^\circ` between two couples, which needed the partner couple's
    potential as a second field and closed nothing when a record carried only its own. The
    carrier is now a pair of pseudo formation energies at the METABOLITE level
    (:meth:`CuratedResolver._price_carrier`), so the ordinary stoichiometric sum prices
    every reaction that conserves it and the partner couple prices itself.
    """

    name = "curated_reaction_pricer"

    def __init__(
        self, records: CuratedRecordSet, identities: list[MetaboliteIdentity]
    ) -> None:
        """Index the admitted reaction-level records by metabolite and by reaction."""
        self.records = records
        self.identity: dict[str, MetaboliteIdentity] = {m.met_id: m for m in identities}
        self.energy_by_reaction: dict[str, CuratedThermoRecord] = {}
        for record in records.of_kind(CurationKind.REACTION_ENERGY):
            if not record.admitted:
                continue
            assert isinstance(record.value, ReactionEnergyValue)
            self.energy_by_reaction[record.value.reaction_id] = record
        #: Why a reaction a record touches was NOT priced, read by the closure report.
        self.refusals: dict[str, str] = {}

    def price(
        self,
        reaction_id: str,
        stoichiometry: dict[str, float],
        unpriced_met_ids: list[str],
        prices: dict[str, MetabolitePrice],
        conditions: CompartmentConditions,
    ) -> ReactionPrice | None:
        """Price this reaction from a curated record, or return ``None``."""
        record = self.energy_by_reaction.get(reaction_id)
        if record is None:
            return None
        return self._from_reaction_energy(record, reaction_id, conditions)

    def _from_reaction_energy(
        self,
        record: CuratedThermoRecord,
        reaction_id: str,
        conditions: CompartmentConditions,
    ) -> ReactionPrice:
        value = record.value
        assert isinstance(value, ReactionEnergyValue)
        return ReactionPrice(
            reaction_id=reaction_id,
            delta_r_g0_kj_per_mol=value.drg_kj_per_mol,
            sigma_kj_per_mol=record.sigma_kj_per_mol,
            route=ThermoRoute.CURATED_FORMATION,
            provenance={
                "slug": record.slug,
                "admission": str(record.admission),
                "reported_state": value.standard_or_transformed,
                "reported_p_h": value.pH,
                "reported_temperature_k": value.T_K,
                "pipeline_temperature_k": conditions.temperature_k,
                "sources": [s.source_url for s in record.sources],
            },
        )


class MoietyParticipant(BaseModel):
    """One participant of a conserved-moiety request, as the GEM declares it."""

    model_config = ConfigDict(extra="forbid")

    met_id: str
    coefficient: float
    name: str = ""
    formula: str = ""
    charge: int = 0
    compartment: str = ""
    p_h: float
    ionic_strength_m: float
    temperature_k: float
    p_mg: float
    #: A curated structure for this species, when a record supplies one.
    smiles: str | None = None
    #: The GEM's own accessions, for the compound bridge to try.
    accessions: dict[str, list[str]] = Field(default_factory=dict)


class ConservedMoietyRequest(BaseModel):
    """One reaction the conserved-moiety route should be attempted on."""

    model_config = ConfigDict(extra="forbid")

    reaction_id: str
    participants: list[MoietyParticipant] = Field(default_factory=list)
    #: The participants that block the ordinary route, for the report.
    blocking_met_ids: list[str] = Field(default_factory=list)


class PricedMoietyReaction(BaseModel):
    r"""What the conserved-moiety route did with one reaction.

    The route exists because a moiety with no group cover, the porphyrin macrocycle, sits
    unchanged on both sides of the reactions that modify its substituents. Decomposing
    every participant in the decomposer's LENIENT mode leaves that moiety unassigned; when
    the unassigned atoms match as a multiset across the two sides they cancel exactly, and
    the reaction is the GROUP DIFFERENCE of everything else, which component contribution
    can price even though it can price neither participant on its own.
    """

    model_config = ConfigDict(extra="forbid")

    reaction_id: str
    #: ``priced``, ``moiety_does_not_cancel``, ``group_in_null_space``,
    #: ``no_structure``, ``untrusted_structure`` or ``decomposition_failed``.
    outcome: str = ""
    delta_r_g0_kj_per_mol: float | None = None
    sigma_kj_per_mol: float | None = None
    #: Norm of the infinite-variance component. Zero is the estimability test.
    sigma_inf_norm: float | None = None
    #: The untransformed group-difference estimate, before the compartment transform.
    standard_kj_per_mol: float | None = None
    transform_kj_per_mol: float | None = None
    #: Element -> net count of the atoms no group covered. Empty is what lets the route run.
    unassigned_residual: dict[str, float] = Field(default_factory=dict)
    #: Groups in the difference vector, with how many training compounds carry each. A
    #: group at zero is why an inestimable difference is inestimable.
    difference_groups: list[dict[str, object]] = Field(default_factory=list)
    #: Group counts moved from a protonated form nobody trained on to its conjugate base.
    protonation_normalized: dict[str, float] = Field(default_factory=dict)
    #: How each participant entered: ``reactant_contribution``, ``group_contribution``,
    #: ``lenient_decomposition``, or the reason it did not.
    participants: dict[str, str] = Field(default_factory=dict)
    detail: str = ""

    @property
    def priced(self) -> bool:
        """True when the route produced a value."""
        return self.delta_r_g0_kj_per_mol is not None


class MoietyReactionTable:
    """A :class:`ReactionPricer` that replays what the conserved-moiety route computed.

    The route needs the group decomposer and the component-contribution parameters, which
    live in the other environment, so the GEM driver reads the artifact rather than the
    library. A reaction the artifact does not name is simply not priced by this stage; a
    reaction it names with an outcome other than ``priced`` is not priced either, and the
    reason travels with it into the report.
    """

    name = "conserved_moiety_table"

    def __init__(self, path: str) -> None:
        """Read the priced-moiety artifact and index it by reaction."""
        with open(path) as handle:
            payload = json.load(handle)
        self.path = path
        self.rows: dict[str, PricedMoietyReaction] = {
            row["reaction_id"]: PricedMoietyReaction.model_validate(row)
            for row in payload["records"]
        }
        self.refusals: dict[str, str] = {
            reaction_id: f"{row.outcome}: {row.detail}"
            for reaction_id, row in self.rows.items()
            if not row.priced
        }

    def price(
        self,
        reaction_id: str,
        stoichiometry: dict[str, float],
        unpriced_met_ids: list[str],
        prices: dict[str, MetabolitePrice],
        conditions: CompartmentConditions,
    ) -> ReactionPrice | None:
        """The stored conserved-moiety value for this reaction, if there is one."""
        row = self.rows.get(reaction_id)
        if row is None or not row.priced:
            return None
        return ReactionPrice(
            reaction_id=reaction_id,
            delta_r_g0_kj_per_mol=row.delta_r_g0_kj_per_mol,
            sigma_kj_per_mol=row.sigma_kj_per_mol,
            route=ThermoRoute.CONSERVED_MOIETY,
            provenance={
                "route_artifact": self.path,
                "standard_kj_per_mol": row.standard_kj_per_mol,
                "transform_kj_per_mol": row.transform_kj_per_mol,
                "sigma_inf_norm": row.sigma_inf_norm,
                "unassigned_residual": row.unassigned_residual,
                "protonation_normalized": row.protonation_normalized,
                "participants": row.participants,
                "detail": row.detail,
            },
        )


def moiety_requests(
    model_reactions: list[tuple[str, dict[str, float]]],
    identities: list[MetaboliteIdentity],
    prices: dict[str, MetabolitePrice],
    records: CuratedRecordSet,
    conditions: CompartmentConditions,
    reachable_reasons: frozenset[UnpricedReason],
    already_priced: frozenset[str] = frozenset(),
) -> list[ConservedMoietyRequest]:
    """Every unpriced reaction the conserved-moiety route could possibly reach.

    Possible means: every participant the ordinary route cannot price has a STRUCTURE to
    decompose, from a curated record or from one of the GEM's own accessions, and its
    reason is one a structure could still fix. A species with a placeholder formula has no
    structure and never will, so a reaction blocked by one is not requested.
    """
    by_id = {m.met_id: m for m in identities}
    structure_by_met: dict[str, str] = {}
    for record in records.records:
        for met_id in record.met_ids:
            smiles, _ = structure_for(record, met_id)
            if smiles is not None and ATTACHMENT_POINT not in smiles:
                structure_by_met[met_id] = smiles
    out: list[ConservedMoietyRequest] = []
    for reaction_id, stoichiometry in model_reactions:
        if reaction_id in already_priced:
            # A reaction the cancellation rule or the fill table already priced never
            # reaches a reaction pricer, so computing a second value for it would be work
            # nobody reads.
            continue
        blocking = sorted(
            m for m in stoichiometry if m not in prices or not prices[m].priced
        )
        if not blocking:
            continue
        reachable = all(
            (prices[m].reason in reachable_reasons if m in prices else False)
            and (m in structure_by_met or by_id[m].accessions)
            for m in blocking
        )
        if not reachable:
            continue
        participants = []
        for met_id, coefficient in sorted(stoichiometry.items()):
            met = by_id[met_id]
            p_h, ionic, p_mg = conditions.for_compartment(met.compartment)
            participants.append(
                MoietyParticipant(
                    met_id=met_id,
                    coefficient=coefficient,
                    name=met.name,
                    formula=met.formula,
                    charge=met.charge,
                    compartment=met.compartment,
                    p_h=p_h,
                    ionic_strength_m=ionic,
                    temperature_k=conditions.temperature_k,
                    p_mg=p_mg,
                    smiles=structure_by_met.get(met_id),
                    accessions=met.accessions,
                )
            )
        out.append(
            ConservedMoietyRequest(
                reaction_id=reaction_id,
                participants=participants,
                blocking_met_ids=blocking,
            )
        )
    return out
