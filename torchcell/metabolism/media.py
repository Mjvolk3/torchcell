# torchcell/metabolism/media.py
# [[torchcell.metabolism.media]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/metabolism/media.py
r"""Map an ontology ``Media`` object onto genome-scale-model exchange bounds.

This is the cobra adapter that ``[[torchcell.datamodels.media-components]]`` promises:
``Media`` records the wet-lab truth (real ingredients at real concentrations) and every
model convention (uptake magnitudes, exchange-reaction ids, the closed-by-default
convention) lives HERE, never in the data ontology.

Why the layer is written as a pure function of a GEM
----------------------------------------------------
A media layer that hardcodes ``r_1714`` is a yeast-GEM script, not a media layer: the
same recipe has to land on iIsor850, iMM904, or a Yarrowia model without an edit. So the
only inputs are (1) an ontology ``Media`` and (2) a ``cobra.Model``, and every exchange
reaction is FOUND by resolving the component's chemical identity against the identifiers
the model itself carries (``chebi`` / ``kegg.compound`` / ``metanetx.chemical`` /
``bigg.metabolite`` annotations) or against the metabolite's name. Nothing in the
resolution path knows what organism it is looking at, and the recipes below are lists of
chemicals rather than lists of reaction ids.

Two consequences of that choice are load-bearing:

- The name channel needs a chemical normalizer (``L-aspartic acid`` in a recipe vs
  ``L-aspartate`` in the model; ``thiamine hydrochloride`` vs ``thiamine``). Those
  rules are chemistry, they hold for any model, and they are listed explicitly in
  ``_NAME_SYNONYMS`` / ``_DISSOCIATION`` so a reader can audit every one.
- A salt is not one exchange. ``ammonium sulfate`` is an ammonium exchange AND a
  sulfate exchange, so a component resolves to a LIST of exchanges.

Coverage is a record, never a silent drop
-----------------------------------------
Every component of the input medium comes back in ``MediaBounds.resolutions`` with one
of three outcomes: ``resolved`` (it became exchange bounds), ``excluded_by_role`` (agar,
G418, peptone: the model has no way to represent it and we say so), or ``unresolved``
(we tried and failed, with the reason and the candidate names we tried recorded). A
component can never vanish between the ontology object and the bound vector.

The Suthers uptake magnitudes, and where our old code disagreed with the source
------------------------------------------------------------------------------
Suthers et al. 2020 (``suthersGenomescaleMetabolicReconstruction2020``, Metab Eng
Commun 11, e00148, doi 10.1016/j.mec.2020.e00148), sec. 2.5 "Modeling simulations":

    "During initial testing ... the carbon substrate uptake rate was set to a value 3.3
    mmol gDW-1 hr-1; we chose this value as a rough estimate for glucose uptake ... and
    arbitrarily applied it to each carbon substrate. ... For growth predictions involving
    rich media, supplementary compound uptake rates were set to 0.165 mmol gDW-1 hr-1
    (i.e., 5% of default substrate uptake rate of 3.3 mmol gDW-1 hr-1). ... The undefined
    composition of yeast extract in Yeast-Peptone-Dextrose (YPD) media was assumed to be
    that of YNB media plus 20 amino acids and D-glucose. ... Glucose uptake rate was set
    to 10.0 mmol gDW-1 hr-1 during OptKnock simulations."

The supplement rate is a FIXED ABSOLUTE 0.165 mmol/gDW/h anchored to the DEFAULT 3.3
carbon uptake. It is not 5% of whatever glucose bound the caller happens to set: the
paper's own OptKnock runs raise glucose to 10.0 and leave supplements at 0.165.

``experiments/007-kuzmin-tm/scripts/setup_media_conditions.py`` (lines 71 and 124) and
the iBioFoundry ``media_setup.py`` it was ported from both compute
``supplement_rate = glucose_rate * 0.05``, which at their ``glucose_rate=10.0`` gives
0.5, i.e. **3.03x the sourced value**. ``torchcell/datamodels/media.py`` repeats the
"5% of glucose" phrasing in prose. FOLLOW THE SOURCE: ``UptakePolicy.supplement_uptake``
is the absolute 0.165 and does not move when ``carbon_uptake`` moves. Constant-ratio
rescaling remains a defensible modeling choice, but it is a choice of ours, not a
sourced value, so it is not the default here.

``carbon_uptake`` defaults to 3.3 for the same reason: it is the value the 0.165 is
defined against, so the default policy is internally coherent with its own source. A
caller doing OptKnock-style work sets ``carbon_uptake=10.0`` and the supplement rate
correctly stays put.

Naming discipline: YPD is approximated, and the name says so
-----------------------------------------------------------
Real YPD contains peptone-derived peptides and yeast-extract lipids that a GEM has no
way to represent. The recipe below is therefore named ``YPD_APPROX`` and its medium is
labeled "YPD-approx (YNB + 20 amino acids; peptone NOT modeled)", carrying the Suthers
substitution quote. Peptone is never modeled, by us or by Suthers. The bound vector of
``YPD_APPROX`` is numerically identical to SC minus its two nucleobases, but it asserts
something different, so it gets its own name and its own provenance string.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

from pydantic import Field

from torchcell.datamodels.media import SC as _ONTOLOGY_SC
from torchcell.datamodels.media import YNB as _ONTOLOGY_YNB
from torchcell.datamodels.pydant import ModelStrict
from torchcell.datamodels.schema import (
    Compound,
    Concentration,
    ConcentrationUnit,
    Media,
    MediaComponent,
    MediaComponentRole,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps cobra out of import time
    import cobra

FLUX_UNIT = "mmol/gDW/h"

# --------------------------------------------------------------------------- #
# Uptake policy. Model convention, sourced to Suthers 2020 sec. 2.5 (see module
# docstring for the verbatim quote and for where our older code diverged).
# --------------------------------------------------------------------------- #

#: Citation anchor repeated into every bound's ``source`` string so a bound vector is
#: self-describing once serialized to JSON.
SUTHERS_2020 = "suthersGenomescaleMetabolicReconstruction2020"


class UptakePolicy(ModelStrict):
    """Magnitudes assigned to an uptake bound, by the component's functional role.

    Three tiers, because a GEM medium only ever distinguishes three things: the growth-
    limiting carbon source, a supplement that is present but not the limiting substrate,
    and a species assumed to be in excess (water, oxygen, bulk salts).

    ``supplement_uptake`` is deliberately independent of ``carbon_uptake``: Suthers'
    0.165 is an absolute rate anchored to the DEFAULT 3.3 carbon uptake, and their own
    OptKnock runs raise carbon to 10.0 without touching it.
    """

    carbon_uptake: float = Field(
        default=3.3,
        description="uptake magnitude for the carbon source, mmol/gDW/h; Suthers' "
        "default substrate rate, the value the 0.165 supplement rate is 5% of",
    )
    supplement_uptake: float = Field(
        default=0.165,
        description="uptake magnitude for amino acids / vitamins / nucleobases, "
        "mmol/gDW/h; Suthers' ABSOLUTE rich-media supplement rate, NOT 5% of "
        "whatever carbon_uptake is set to",
    )
    unlimited_uptake: float = Field(
        default=1000.0,
        description="uptake magnitude for species assumed in excess (water, oxygen, "
        "bulk salts, trace metals); the conventional cobra 'open' bound",
    )

    def source_for(self, role: MediaComponentRole) -> str:
        """Provenance string recorded on every bound this policy produces."""
        if role in _CARBON_ROLES:
            return (
                f"{SUTHERS_2020} sec2.5 default carbon substrate uptake "
                f"{self.carbon_uptake} {FLUX_UNIT}"
            )
        if role in _SUPPLEMENT_ROLES:
            return (
                f"{SUTHERS_2020} sec2.5 supplementary compound uptake "
                f"{self.supplement_uptake} {FLUX_UNIT} (absolute; 5% of the default "
                "3.3 carbon uptake, not rescaled with carbon_uptake)"
            )
        return (
            f"cobra convention: species assumed in excess, opened to "
            f"{self.unlimited_uptake} {FLUX_UNIT}"
        )

    def bound_for(self, role: MediaComponentRole) -> float:
        """Uptake magnitude (positive) for a component in this functional role."""
        if role in _CARBON_ROLES:
            return self.carbon_uptake
        if role in _SUPPLEMENT_ROLES:
            return self.supplement_uptake
        return self.unlimited_uptake


_CARBON_ROLES = frozenset({MediaComponentRole.carbon_source})
_SUPPLEMENT_ROLES = frozenset(
    {
        MediaComponentRole.amino_acid,
        MediaComponentRole.vitamin,
        MediaComponentRole.nucleobase,
    }
)
#: Roles a constraint-based model structurally cannot represent. Agar is a gel matrix,
#: not a nutrient; a selection agent acts through a genotype x medium mechanism the GEM
#: has no genes for; peptone and yeast extract are undefined digests. These are reported
#: as ``excluded_by_role`` rather than left to fail name resolution, so that "the model
#: cannot represent this" is distinguishable from "we could not find it".
_NOT_MODELED_ROLES = frozenset(
    {
        MediaComponentRole.gelling_agent,
        MediaComponentRole.selection_agent,
        MediaComponentRole.complex_ingredient,
    }
)

# --------------------------------------------------------------------------- #
# Chemical name normalization. Every rule here is chemistry, not yeast: an
# ontology recipe names the bottle on the bench ("thiamine hydrochloride") while a
# GEM names the species in solution ("thiamine"), and that gap is the same on any
# model. Kept as explicit tables so each mapping is auditable.
# --------------------------------------------------------------------------- #

#: Salt / hydrate / stereo qualifiers that a GEM never carries on the species name.
_SALT_QUALIFIERS = (
    "hydrochloride",
    "hydrate",
    "monohydrate",
    "dihydrate",
    "anhydrous",
    "sodium salt",
    "potassium salt",
    "calcium salt",
)

#: Bench name -> species name. Trivial synonyms only; each is a naming difference for
#: the SAME chemical, never a substitution of one chemical for another.
_NAME_SYNONYMS: dict[str, str] = {
    "niacin": "nicotinate",
    "nicotinic acid": "nicotinate",
    "vitamin b3": "nicotinate",
    # the vitamin is the (R) enantiomer; the bench sells its calcium salt
    "calcium pantothenate": "(r)-pantothenate",
    "pantothenate": "(r)-pantothenate",
    "pantothenic acid": "(r)-pantothenate",
    # British spelling, as several reconstructions carry it
    "sulfate": "sulphate",
    "sulfite": "sulphite",
    "vitamin b1": "thiamine",
    "vitamin b2": "riboflavin",
    "vitamin b6": "pyridoxine",
    "vitamin h": "biotin",
    "para-aminobenzoic acid": "4-aminobenzoate",
    "p-aminobenzoic acid": "4-aminobenzoate",
    "paba": "4-aminobenzoate",
    "inositol": "myo-inositol",
    "dextrose": "d-glucose",
    "glucose": "d-glucose",
    "water": "h2o",
    "proton": "h+",
    "hydrogen ion": "h+",
    "dioxygen": "oxygen",
    "o2": "oxygen",
    "ferrous iron": "iron(2+)",
    "iron(ii)": "iron(2+)",
    "copper(2+)": "cu2(+)",
    "manganese(2+)": "mn(2+)",
    "zinc(2+)": "zn(2+)",
    "magnesium(2+)": "mg(2+)",
    "calcium(2+)": "ca(2+)",
    "ammonium ion": "ammonium",
}

#: Salts that dissociate into more than one transportable ion. A component naming one of
#: these resolves to SEVERAL exchanges, which is why a resolution holds a list.
_DISSOCIATION: dict[str, tuple[str, ...]] = {
    "ammonium sulfate": ("ammonium", "sulfate"),
    "monosodium l-glutamate": ("sodium", "l-glutamate"),
    "monosodium glutamate": ("sodium", "l-glutamate"),
    "potassium phosphate": ("potassium", "phosphate"),
    "magnesium sulfate": ("mg(2+)", "sulfate"),
    "sodium chloride": ("sodium", "chloride"),
}


def _normalize(name: str) -> str:
    """Lowercase, collapse whitespace, drop punctuation a model never carries."""
    text = name.strip().lower()
    text = text.replace("’", "'")
    text = re.sub(r"\s+", " ", text)
    return text


def _candidate_names(name: str) -> list[str]:
    """Chemical-naming variants to try against a model's metabolite names, in order.

    Order matters: the most literal form is tried first so a model that does carry the
    bench name wins before any rewrite is applied. Each later candidate encodes one
    chemistry rule (drop the salt form, conjugate acid -> conjugate base, D/L prefix
    convention), and every rule generalizes across models.
    """
    base = _normalize(name)
    out: list[str] = [base]

    if base in _NAME_SYNONYMS:
        out.append(_NAME_SYNONYMS[base])

    stripped = base
    for qualifier in _SALT_QUALIFIERS:
        stripped = stripped.replace(qualifier, "").strip()
    stripped = re.sub(r"\s+", " ", stripped)
    if stripped and stripped != base:
        out.append(stripped)
        if stripped in _NAME_SYNONYMS:
            out.append(_NAME_SYNONYMS[stripped])

    for candidate in list(out):
        # conjugate acid -> conjugate base, the form a GEM stores at physiological pH
        if candidate.endswith("ic acid"):
            out.append(candidate[: -len("ic acid")] + "ate")
        if candidate.endswith(" acid"):
            out.append(candidate[: -len(" acid")] + "ate")
        # amino acids: recipes write "glycine", models often write "L-glycine"
        if not candidate.startswith(("l-", "d-")):
            out.append("l-" + candidate)

    seen: set[str] = set()
    ordered: list[str] = []
    for candidate in out:
        if candidate and candidate not in seen:
            seen.add(candidate)
            ordered.append(candidate)
    return ordered


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #

#: Identifier namespaces a model may annotate its metabolites with. Ordered by how
#: specific the identifier is, so the most chemically precise channel wins.
ANNOTATION_KEYS: tuple[str, ...] = (
    "chebi",
    "kegg.compound",
    "metanetx.chemical",
    "bigg.metabolite",
)


class ExchangeBound(ModelStrict):
    """One exchange reaction opened to an uptake magnitude, with its provenance.

    ``uptake_bound`` is a POSITIVE magnitude in ``unit``; applying it sets the exchange's
    ``lower_bound`` to its negation (cobra's sign convention for uptake). Keeping the
    stored value positive means a serialized bound vector reads as a medium composition
    rather than as a pile of negative numbers.
    """

    exchange_id: str = Field(description="exchange reaction id in the target model")
    metabolite_id: str = Field(description="extracellular metabolite id")
    metabolite_name: str = Field(description="metabolite name as the model carries it")
    uptake_bound: float = Field(description="positive uptake magnitude")
    unit: Literal["mmol/gDW/h"] = Field(
        default="mmol/gDW/h",
        description="flux unit; a bound is a RATE, never the recipe's concentration",
    )
    source: str = Field(description="why this magnitude, citation-anchored")


class ComponentResolution(ModelStrict):
    """What became of one ontology component on the way to a bound vector.

    Exactly one of the three outcomes, recorded for EVERY component of the input medium.
    """

    component_name: str
    role: MediaComponentRole
    outcome: Literal["resolved", "excluded_by_role", "unresolved"]
    exchange_ids: list[str] = Field(default_factory=list)
    match_channel: str | None = Field(
        default=None,
        description="how it matched: 'annotation:<namespace>' or 'name:<candidate>'",
    )
    reason: str | None = Field(
        default=None, description="why it was excluded or could not be resolved"
    )
    candidates_tried: list[str] = Field(
        default_factory=list,
        description="normalized name candidates tried; the audit trail of a failure",
    )


class MediaBounds(ModelStrict):
    """A medium expressed as exchange bounds for one specific model.

    Pure data: it holds no reference to the ``cobra.Model`` it was built against, only
    that model's id, so it serializes to JSON and can be diffed against another medium.
    """

    media_name: str
    model_id: str
    policy: UptakePolicy
    bounds: dict[str, ExchangeBound] = Field(
        default_factory=dict, description="exchange reaction id -> bound"
    )
    resolutions: list[ComponentResolution] = Field(default_factory=list)

    @property
    def n_components(self) -> int:
        """Components in the ontology object that produced this vector."""
        return len(self.resolutions)

    @property
    def n_resolved(self) -> int:
        """Components that became at least one exchange bound."""
        return sum(1 for r in self.resolutions if r.outcome == "resolved")

    @property
    def unresolved_names(self) -> list[str]:
        """Components we tried and failed to place on an exchange reaction."""
        return [r.component_name for r in self.resolutions if r.outcome == "unresolved"]

    @property
    def excluded_names(self) -> list[str]:
        """Components a constraint-based model structurally cannot represent."""
        return [
            r.component_name
            for r in self.resolutions
            if r.outcome == "excluded_by_role"
        ]

    def apply(self, model: cobra.Model) -> None:
        """Close every exchange on ``model``, then open exactly this medium.

        Mutates the model, so callers use cobra's ``with model:`` context to scope it.
        Closing first is what makes a medium a statement about the whole boundary rather
        than an additive patch on whatever the model shipped with.
        """
        for reaction in model.exchanges:
            reaction.lower_bound = 0.0
        for bound in self.bounds.values():
            model.reactions.get_by_id(
                bound.exchange_id
            ).lower_bound = -bound.uptake_bound


class MediaBoundsDiff(ModelStrict):
    """Exchange-level difference between two bound vectors on the same model."""

    left: str
    right: str
    only_in_left: dict[str, float] = Field(default_factory=dict)
    only_in_right: dict[str, float] = Field(default_factory=dict)
    differing: dict[str, tuple[float, float]] = Field(default_factory=dict)

    @property
    def n_differences(self) -> int:
        """Exchanges whose uptake bound is not identical across the two media."""
        return len(self.only_in_left) + len(self.only_in_right) + len(self.differing)


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #


class ExchangeIndex(ModelStrict):
    """Lookup tables built once from a model's exchange reactions.

    Built from what the MODEL declares (its annotations and its metabolite names), which
    is what keeps the mapping organism-agnostic: no yeast reaction id appears anywhere in
    this module, and the same index construction runs on any cobra model.
    """

    model_id: str
    by_annotation: dict[str, str] = Field(
        default_factory=dict, description="'<namespace>:<id>' -> exchange reaction id"
    )
    by_name: dict[str, str] = Field(
        default_factory=dict, description="normalized metabolite name -> exchange id"
    )
    metabolite_of: dict[str, tuple[str, str]] = Field(
        default_factory=dict, description="exchange id -> (metabolite id, name)"
    )


def build_exchange_index(model: cobra.Model) -> ExchangeIndex:
    """Index a model's exchange reactions by annotation identifier and by name.

    An exchange with more than one metabolite is skipped: it is not a boundary reaction
    for a single species, so no medium component can name it unambiguously.
    """
    by_annotation: dict[str, str] = {}
    by_name: dict[str, str] = {}
    metabolite_of: dict[str, tuple[str, str]] = {}

    for reaction in model.exchanges:
        metabolites = list(reaction.metabolites)
        if len(metabolites) != 1:
            continue
        metabolite = metabolites[0]
        metabolite_of[reaction.id] = (metabolite.id, metabolite.name)

        for namespace in ANNOTATION_KEYS:
            value = metabolite.annotation.get(namespace)
            if value is None:
                continue
            values = value if isinstance(value, list) else [value]
            for item in values:
                key = f"{namespace}:{_normalize(str(item))}"
                by_annotation.setdefault(key, reaction.id)

        by_name.setdefault(_normalize(metabolite.name), reaction.id)

    return ExchangeIndex(
        model_id=model.id,
        by_annotation=by_annotation,
        by_name=by_name,
        metabolite_of=metabolite_of,
    )


def _annotation_keys_for(compound: Compound) -> list[tuple[str, str]]:
    """Identifier lookups a compound's own cross-references support.

    ``Compound.chebi_id`` is the join key the media design note names. Every entry a
    compound carries is tried before any name matching, because an identifier is exact
    and a name is a guess dressed as a fact.
    """
    keys: list[tuple[str, str]] = []
    if compound.chebi_id is not None:
        keys.append(("chebi", f"chebi:{_normalize(compound.chebi_id)}"))
        # a model may store the bare numeric form rather than the CURIE
        keys.append(("chebi", f"chebi:{_normalize(compound.chebi_id.split(':')[-1])}"))
    return keys


def _resolve_one_name(
    name: str, index: ExchangeIndex
) -> tuple[str | None, str, list[str]]:
    """Best exchange for a chemical name: (exchange id, channel, candidates tried)."""
    candidates = _candidate_names(name)
    for candidate in candidates:
        exchange_id = index.by_name.get(candidate)
        if exchange_id is not None:
            return exchange_id, f"name:{candidate}", candidates
    return None, "", candidates


def resolve_component(
    component: MediaComponent, index: ExchangeIndex
) -> ComponentResolution:
    """Place one ontology component on zero or more exchange reactions.

    Order is annotation first, then dissociation into ions, then chemical name. A
    component whose role the model structurally cannot represent short-circuits to
    ``excluded_by_role`` before any lookup, so a failure to find agar is never reported
    as a resolution failure.
    """
    name = component.compound.name
    if component.role in _NOT_MODELED_ROLES:
        return ComponentResolution(
            component_name=name,
            role=component.role,
            outcome="excluded_by_role",
            reason=f"role '{component.role.value}' has no constraint-based "
            "representation (gel matrix, genotype-dependent selection agent, or "
            "undefined biological digest)",
        )

    for namespace, key in _annotation_keys_for(component.compound):
        exchange_id = index.by_annotation.get(key)
        if exchange_id is not None:
            return ComponentResolution(
                component_name=name,
                role=component.role,
                outcome="resolved",
                exchange_ids=[exchange_id],
                match_channel=f"annotation:{namespace}",
            )

    normalized = _normalize(name)
    if normalized in _DISSOCIATION:
        exchange_ids: list[str] = []
        tried: list[str] = []
        for ion in _DISSOCIATION[normalized]:
            exchange_id, _channel, candidates = _resolve_one_name(ion, index)
            tried.extend(candidates)
            if exchange_id is not None:
                exchange_ids.append(exchange_id)
        if exchange_ids:
            return ComponentResolution(
                component_name=name,
                role=component.role,
                outcome="resolved",
                exchange_ids=exchange_ids,
                match_channel=f"name:dissociation({normalized})",
                candidates_tried=tried,
            )
        return ComponentResolution(
            component_name=name,
            role=component.role,
            outcome="unresolved",
            reason="salt dissociates but no ion matched a model exchange",
            candidates_tried=tried,
        )

    exchange_id, channel, candidates = _resolve_one_name(name, index)
    if exchange_id is not None:
        return ComponentResolution(
            component_name=name,
            role=component.role,
            outcome="resolved",
            exchange_ids=[exchange_id],
            match_channel=channel,
            candidates_tried=candidates,
        )
    return ComponentResolution(
        component_name=name,
        role=component.role,
        outcome="unresolved",
        reason="no model exchange matched by annotation or by any chemical name variant"
        + (
            f"; component identity is '{component.definition.value}' so it names a "
            "mixture rather than a single species"
            if component.definition.value != "defined"
            else ""
        ),
        candidates_tried=candidates,
    )


def media_to_bounds(
    media: Media,
    model: cobra.Model,
    policy: UptakePolicy | None = None,
    index: ExchangeIndex | None = None,
) -> MediaBounds:
    """Map an ontology ``Media`` onto exchange bounds for ``model``.

    Pure: reads the model's structure and annotations, mutates nothing. When two
    components land on the same exchange (a shared ion), the LARGER uptake wins, because
    the medium contains both sources and availability is additive rather than limiting.
    """
    resolved_policy = policy if policy is not None else UptakePolicy()
    resolved_index = index if index is not None else build_exchange_index(model)

    bounds: dict[str, ExchangeBound] = {}
    resolutions: list[ComponentResolution] = []

    for component in media.components:
        resolution = resolve_component(component, resolved_index)
        resolutions.append(resolution)
        if resolution.outcome != "resolved":
            continue
        magnitude = resolved_policy.bound_for(component.role)
        source = resolved_policy.source_for(component.role)
        for exchange_id in resolution.exchange_ids:
            metabolite_id, metabolite_name = resolved_index.metabolite_of[exchange_id]
            existing = bounds.get(exchange_id)
            if existing is not None and existing.uptake_bound >= magnitude:
                continue
            bounds[exchange_id] = ExchangeBound(
                exchange_id=exchange_id,
                metabolite_id=metabolite_id,
                metabolite_name=metabolite_name,
                uptake_bound=magnitude,
                source=f"{source} [component: {component.compound.name}]",
            )

    return MediaBounds(
        media_name=media.name,
        model_id=model.id,
        policy=resolved_policy,
        bounds=bounds,
        resolutions=resolutions,
    )


def diff_bounds(left: MediaBounds, right: MediaBounds) -> MediaBoundsDiff:
    """Exchange-level difference between two bound vectors."""
    left_ids = set(left.bounds)
    right_ids = set(right.bounds)
    return MediaBoundsDiff(
        left=left.media_name,
        right=right.media_name,
        only_in_left={
            i: left.bounds[i].uptake_bound for i in sorted(left_ids - right_ids)
        },
        only_in_right={
            i: right.bounds[i].uptake_bound for i in sorted(right_ids - left_ids)
        },
        differing={
            i: (left.bounds[i].uptake_bound, right.bounds[i].uptake_bound)
            for i in sorted(left_ids & right_ids)
            if left.bounds[i].uptake_bound != right.bounds[i].uptake_bound
        },
    )


# --------------------------------------------------------------------------- #
# The four recipes our datasets need, as ontology ``Media`` objects.
#
# They are ``Media``, not bound dicts, so the audit exercises the real ontology ->
# bounds path rather than a shortcut, and so a recipe can be compared field by field
# against what a dataset loader emits.
# --------------------------------------------------------------------------- #


def _component(
    name: str,
    role: MediaComponentRole,
    note: str | None = None,
    concentration: Concentration | None = None,
) -> MediaComponent:
    """A name-only defined component, matching the ontology library's discipline.

    ChEBI / InChIKey cross-refs stay empty exactly as in
    ``torchcell/datamodels/media.py``: they are filled by a sourced resolver pass and
    never guessed here. The consequence is measured by the audit -- with the join key
    empty, resolution falls to the name channel on every component.
    """
    return MediaComponent(
        compound=Compound(name=name), role=role, note=note, concentration=concentration
    )


#: Species a wet-lab recipe never lists but a GEM cannot grow without: the solvent, the
#: atmosphere, the protons, and the salt/trace-metal content that the ontology hides
#: inside a ``composition_deferred`` "yeast nitrogen base" line. Declaring them here is
#: the honest place for them: they are model requirements, not bench ingredients, and
#: they are named as chemicals so the resolver finds them on any annotated model.
MINERAL_BASE: list[MediaComponent] = [
    _component("water", MediaComponentRole.other, "solvent; never a recipe line"),
    _component(
        "oxygen",
        MediaComponentRole.other,
        "aerobic culture; the Media schema "
        "has no aeration field, so this is a modeling assumption, not a record",
    ),
    _component("H+", MediaComponentRole.buffer, "proton exchange, cobra convention"),
    _component(
        "phosphate",
        MediaComponentRole.bulk_salt,
        "YNB salt, composition_deferred in the ontology",
    ),
    _component(
        "sulfate",
        MediaComponentRole.bulk_salt,
        "YNB salt, composition_deferred in the ontology",
    ),
    _component("sodium", MediaComponentRole.bulk_salt, "YNB salt"),
    _component("potassium", MediaComponentRole.bulk_salt, "YNB salt"),
    _component("chloride", MediaComponentRole.bulk_salt, "YNB salt"),
    _component("iron(2+)", MediaComponentRole.trace_element, "YNB trace metal"),
    _component("copper(2+)", MediaComponentRole.trace_element, "YNB trace metal"),
    _component("manganese(2+)", MediaComponentRole.trace_element, "YNB trace metal"),
    _component("zinc(2+)", MediaComponentRole.trace_element, "YNB trace metal"),
    _component("magnesium(2+)", MediaComponentRole.trace_element, "YNB trace metal"),
    _component("calcium(2+)", MediaComponentRole.trace_element, "YNB trace metal"),
]

_GLUCOSE = _component(
    "D-glucose",
    MediaComponentRole.carbon_source,
    "2% w/v at the bench; the bound comes from the uptake policy, not this number, "
    "because a concentration is not a flux",
    Concentration(value=2.0, unit=ConcentrationUnit.percent_w_v),
)
_AMMONIUM_SULFATE = _component(
    "ammonium sulfate",
    MediaComponentRole.nitrogen_source,
    "5 g/L; dissociates to ammonium + sulfate",
    Concentration(value=5.0, unit=ConcentrationUnit.g_per_l),
)

#: The nine YNB vitamins, taken from the ontology library rather than re-listed, so the
#: recipe and the data ontology cannot drift apart.
_YNB_VITAMINS: list[MediaComponent] = list(_ONTOLOGY_YNB.components)

#: The twenty amino acids, taken from the ontology SC definition for the same reason.
_SC_AMINO_ACIDS: list[MediaComponent] = [
    c for c in _ONTOLOGY_SC.components if c.role is MediaComponentRole.amino_acid
]

_URACIL = _component(
    "uracil", MediaComponentRole.nucleobase, "SC nucleobase supplement"
)
_ADENINE = _component(
    "adenine", MediaComponentRole.nucleobase, "SC nucleobase supplement"
)

_SUTHERS_YPD_QUOTE = (
    "The undefined composition of yeast extract in Yeast-Peptone-Dextrose (YPD) media "
    "was assumed to be that of YNB media plus 20 amino acids and D-glucose."
)

SM_FBA = Media(
    name="SM (synthetic minimal): YNB vitamins + ammonium sulfate + D-glucose",
    state="liquid",
    is_synthetic=True,
    base_medium="SD",
    components=[*MINERAL_BASE, _GLUCOSE, _AMMONIUM_SULFATE, *_YNB_VITAMINS],
)
"""Mulleder's SM: defined minimal medium, no amino acids, no nucleobases.

Modeling decision, stated rather than sourced: SM is read as YNB + glucose + ammonium
with NO amino-acid supplement. Suthers gives no SM recipe; the composition is the
standard Difco YNB formulation already cited in ``torchcell/datamodels/media.py``.
"""

SC_FBA = Media(
    name="SC (synthetic complete): SM + 20 amino acids + uracil + adenine",
    state="liquid",
    is_synthetic=True,
    base_medium="SC",
    components=[*SM_FBA.components, *_SC_AMINO_ACIDS, _URACIL, _ADENINE],
)
"""Cachera betaxanthin's medium.

The SC composition is NOT in Suthers, which uses SC throughout its Fig. 4 without ever
listing it. The recipe is the standard Difco/Sigma SC formulation cited in
``torchcell/datamodels/media.py``; only the uptake magnitudes are Suthers'.
"""

SC_URA_FBA = Media(
    name="SC-Ura (synthetic complete minus uracil): SM + 20 amino acids + adenine",
    state="liquid",
    is_synthetic=True,
    base_medium="SC",
    components=[c for c in SC_FBA.components if c.compound.name != "uracil"],
    dropouts=[Compound(name="uracil")],
)
"""Ozaydin beta-carotene's medium: SC with uracil withheld to select the URA3 plasmid."""

YPD_APPROX_FBA = Media(
    name="YPD-approx (YNB + 20 amino acids; peptone NOT modeled)",
    state="liquid",
    is_synthetic=True,
    base_medium="YPD_approx",
    components=[*SM_FBA.components, *_SC_AMINO_ACIDS],
)
"""Ohya morphology's medium, APPROXIMATED, and the name says which.

Real YPD is ``is_synthetic=False``: peptone-derived peptides and yeast-extract lipids
have no representation in a GEM. This object is ``is_synthetic=True`` because it is not
YPD, it is the defined stand-in Suthers substitutes for it:

    "The undefined composition of yeast extract in Yeast-Peptone-Dextrose (YPD) media
    was assumed to be that of YNB media plus 20 amino acids and D-glucose."

Peptone is never modeled, by us or by Suthers. The bound vector is identical to SC minus
its two nucleobases; the separate name exists because the two objects assert different
things about the bench.
"""

#: The four media our datasets actually need, keyed by the medium NAME the dataset
#: loaders emit, so an audit can join a loader's ``Media`` to the recipe it should use.
FBA_MEDIA: dict[str, Media] = {
    "SM": SM_FBA,
    "SC": SC_FBA,
    "SC-URA": SC_URA_FBA,
    "YPD": YPD_APPROX_FBA,
}
