# torchcell/datamodels/identity.py
# [[torchcell.datamodels.identity]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datamodels/identity
# Test file: tests/torchcell/datamodels/test_identity.py
"""Composition-based identity for the environment-side entities of the graph.

A medium, a temperature, an environment perturbation and an environment are
PERSISTENT ENTITIES: two datasets that state the same medium are stating the same
thing, and the aggregate the whole campaign is for ("every record measured on YPD",
"every record on SC plus 0.4 M NaCl") is only formable when both datasets land on
ONE node. Content-addressing the whole pydantic dump cannot do that, because the
dump also carries the dataset's own bench quote: ``provenance`` (``SourcedValue``
with a sha256 anchor), ``note``, ``defers_to``, ``provenance_gaps`` and the
free-text ``name``. Two papers stating YPD with 20 g/L agar then differ in every
byte that is not the medium, and the join fails at node identity.

These projections read only the COMPOSITION: what is in the medium, at what dose,
in what state; what was added, of what species, at what dose; what the temperature
was. Everything that records who said it, or how it was worded, is dropped. What
remains is hashed by ``identity_sha256`` over a canonical JSON encoding, so the
same composition from any source yields one id.

Three properties the projections hold to:

1. **Total.** Every field read exists on the current schema class (the
   ``*_IDENTITY_FIELDS`` tuples name them, and the tests check them against
   ``model_fields``), so a projection cannot silently skip a slot a record filled.
2. **Absence is part of identity.** A field that is ``None`` stays ``None`` in the
   projection rather than being omitted, so an environment whose temperature is a
   typed ``ProvenanceGap`` is a DIFFERENT environment from one at 30 C, not a
   coincidental match.
3. **Order is not identity.** Components, dropouts and perturbations are sorted by
   their own canonical encoding, so two loaders listing the same ingredients in a
   different order agree.

Two deliberate non-joins, both documented rather than papered over:

- **Units are not converted.** A temperature stated in Kelvin does not join one
  stated in Celsius. Converting would need a rounding tolerance, which is a policy
  choice this layer has no business making; every loader states Celsius today.
- **Structure strings are not identity.** ``smiles`` / ``inchi`` are not part of
  the compound precedence: a SMILES is toolkit-dependent, so two spellings of one
  molecule would not join anyway, and the InChIKey derived from a structure is the
  hash that does. A compound carrying only a structure string falls through to its
  normalized name, exactly as one carrying no identifier does.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from torchcell.datamodels.compound_identity import normalize_compound_name
from torchcell.datamodels.schema import (
    BiologicPerturbation,
    Compound,
    Concentration,
    Environment,
    EnvironmentPhysicalPerturbation,
    Media,
    MediaComponent,
    SmallMoleculePerturbation,
    Solvent,
    Temperature,
)

__all__ = [
    "BIOLOGIC_PERTURBATION_IDENTITY_FIELDS",
    "COMPOUND_IDENTITY_PRECEDENCE",
    "CONCENTRATION_IDENTITY_FIELDS",
    "ENVIRONMENT_IDENTITY_FIELDS",
    "MEDIA_COMPONENT_IDENTITY_FIELDS",
    "MEDIA_IDENTITY_FIELDS",
    "PHYSICAL_PERTURBATION_IDENTITY_FIELDS",
    "SMALL_MOLECULE_IDENTITY_FIELDS",
    "SOLVENT_IDENTITY_FIELDS",
    "TEMPERATURE_IDENTITY_FIELDS",
    "compound_identity_key",
    "environment_identity",
    "environment_perturbation_identity",
    "identity_sha256",
    "media_identity",
    "temperature_identity",
]

COMPOUND_IDENTITY_PRECEDENCE: tuple[str, ...] = ("inchikey", "chebi_id", "pubchem_cid")
"""Identifier fields tried in order before falling back to the normalized name.

InChIKey first (the canonical hash of the standard InChI, and ``Compound``'s own
primary key), then the ChEBI CURIE, then the PubChem CID. Each is a registry-stable
handle for ONE substance, so the first one present is a sufficient key and the rest
are redundant cross-references. ``smiles`` / ``inchi`` are deliberately absent (see
the module docstring).
"""

CONCENTRATION_IDENTITY_FIELDS: tuple[str, ...] = ("value", "unit", "basis")
SOLVENT_IDENTITY_FIELDS: tuple[str, ...] = ("name", "percent", "compound")
MEDIA_COMPONENT_IDENTITY_FIELDS: tuple[str, ...] = (
    "compound",
    "role",
    "concentration",
    "definition",
)
MEDIA_IDENTITY_FIELDS: tuple[str, ...] = (
    "state",
    "is_synthetic",
    "base_medium",
    "components",
    "dropouts",
)
TEMPERATURE_IDENTITY_FIELDS: tuple[str, ...] = ("value", "unit")
SMALL_MOLECULE_IDENTITY_FIELDS: tuple[str, ...] = (
    "perturbation_type",
    "compound",
    "concentration",
    "solvent",
)
PHYSICAL_PERTURBATION_IDENTITY_FIELDS: tuple[str, ...] = (
    "perturbation_type",
    "factor",
    "magnitude",
    "agent",
)
BIOLOGIC_PERTURBATION_IDENTITY_FIELDS: tuple[str, ...] = (
    "perturbation_type",
    "agent_class",
    "name",
    "uniprot_id",
    "sequence",
    "concentration",
)
ENVIRONMENT_IDENTITY_FIELDS: tuple[str, ...] = (
    "media",
    "temperature",
    "perturbations",
    "aerobicity",
    "duration_hours",
    "duration_generations",
)


def _canonical(obj: Any) -> str:
    """Canonical JSON encoding: sorted keys, no insignificant whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _sorted_identities(identities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order a list of projections by their canonical encoding, not by input order."""
    return sorted(identities, key=_canonical)


def compound_identity_key(compound: Compound) -> str:
    """The join key for a compound, as ``"<field>:<value>"``.

    Precedence: ``inchikey``, else ``chebi_id``, else ``pubchem_cid``, else the
    normalized lowercase ``name`` (``normalize_compound_name``, the same folding the
    environment collision check uses, so 'NaCl' and 'sodium chloride' agree even
    with no identifier on either side). The field name is part of the key, so a name
    that happens to be spelled like an InChIKey cannot collide with a real one.
    """
    for field in COMPOUND_IDENTITY_PRECEDENCE:
        value = getattr(compound, field)
        if value is not None:
            return f"{field}:{value}"
    return f"name:{normalize_compound_name(compound.name)}"


def _concentration_identity(
    concentration: Concentration | None,
) -> dict[str, Any] | None:
    """Project a dose: numeric value, typed unit, and how the dose was set."""
    if concentration is None:
        return None
    return {
        "value": concentration.value,
        "unit": None if concentration.unit is None else concentration.unit.value,
        "basis": None if concentration.basis is None else concentration.basis.value,
    }


def _solvent_identity(solvent: Solvent | None) -> dict[str, Any] | None:
    """Project a vehicle: its compound key (name-derived when untyped) and fraction."""
    if solvent is None:
        return None
    compound = solvent.compound
    key = (
        compound_identity_key(compound)
        if compound is not None
        else f"name:{normalize_compound_name(solvent.name)}"
    )
    return {"compound": key, "percent": solvent.percent}


def _media_component_identity(component: MediaComponent) -> dict[str, Any]:
    """Project one ingredient: which substance, in what role, how defined, how much."""
    return {
        "compound": compound_identity_key(component.compound),
        "role": component.role.value,
        "definition": component.definition.value,
        "concentration": _concentration_identity(component.concentration),
    }


def media_identity(media: Media) -> dict[str, Any]:
    """Project a medium onto its composition: state, synthetic flag, base, contents.

    ``name``, ``provenance``, and each component's ``provenance`` / ``note`` /
    ``defers_to`` are dropped: they record who stated the recipe, not what the recipe
    is. Components and dropouts are sorted, so listing order is not identity.
    """
    return {
        "state": media.state,
        "is_synthetic": media.is_synthetic,
        "base_medium": media.base_medium,
        "components": _sorted_identities(
            [_media_component_identity(component) for component in media.components]
        ),
        "dropouts": sorted(
            compound_identity_key(dropout) for dropout in media.dropouts
        ),
    }


def temperature_identity(temperature: Temperature) -> dict[str, Any]:
    """Project a temperature onto its value and typed unit (no unit conversion)."""
    return {"value": temperature.value, "unit": temperature.unit.value}


def environment_perturbation_identity(perturbation: Any) -> dict[str, Any]:
    """Project an environment perturbation onto its typed slots.

    The discriminator plus the slots of the concrete leaf: compound key, dose and
    vehicle for a small molecule; factor, magnitude and realizing agent for a
    physical factor; agent class, name, accession, sequence and dose for a biologic.
    ``description`` is dropped (it is a per-dataset restatement of the leaf's own
    default). The dose IS part of identity on both dosed leaves: two doses of one
    compound are two different edits to the environment.
    """
    if isinstance(perturbation, SmallMoleculePerturbation):
        return {
            "perturbation_type": perturbation.perturbation_type,
            "compound": compound_identity_key(perturbation.compound),
            "concentration": _concentration_identity(perturbation.concentration),
            "solvent": _solvent_identity(perturbation.solvent),
        }
    if isinstance(perturbation, EnvironmentPhysicalPerturbation):
        return {
            "perturbation_type": perturbation.perturbation_type,
            "factor": perturbation.factor.value,
            "magnitude": _concentration_identity(perturbation.magnitude),
            "agent": (
                None
                if perturbation.agent is None
                else compound_identity_key(perturbation.agent)
            ),
        }
    if isinstance(perturbation, BiologicPerturbation):
        return {
            "perturbation_type": perturbation.perturbation_type,
            "agent_class": perturbation.agent_class.value,
            "name": perturbation.name.strip().lower(),
            "uniprot_id": perturbation.uniprot_id,
            "sequence": perturbation.sequence,
            "concentration": _concentration_identity(perturbation.concentration),
        }
    raise TypeError(
        "no identity projection for "
        f"{type(perturbation).__name__}; an environment perturbation is one of "
        "SmallMoleculePerturbation, EnvironmentPhysicalPerturbation, "
        "BiologicPerturbation"
    )


def environment_identity(environment: Environment) -> dict[str, Any]:
    """Project an environment onto medium, temperature, edits, oxygen and duration.

    ``provenance_gaps`` is dropped, but what a gap MEANS is kept: the gapped field is
    ``None``, and a ``None`` temperature projects as ``None``, so a record that never
    carried a temperature does not merge with one measured at 30 C.
    """
    return {
        "media": media_identity(environment.media),
        "temperature": (
            None
            if environment.temperature is None
            else temperature_identity(environment.temperature)
        ),
        "perturbations": _sorted_identities(
            [
                environment_perturbation_identity(perturbation)
                for perturbation in environment.perturbations
            ]
        ),
        "aerobicity": environment.aerobicity,
        "duration_hours": environment.duration_hours,
        "duration_generations": environment.duration_generations,
    }


def identity_sha256(obj: Mapping[str, Any]) -> str:
    """Hash an identity projection: sha256 over its canonical JSON encoding."""
    return hashlib.sha256(_canonical(dict(obj)).encode("utf-8")).hexdigest()
