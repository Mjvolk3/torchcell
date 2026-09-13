# tests/torchcell/datamodels/test_identity.py
"""Identity by composition: the same medium stated by two papers is ONE node.

The environment-side entities used to be content-addressed on the whole pydantic
dump, which carries the stating dataset's ``name``, ``provenance``, ``note``,
``defers_to`` and ``provenance_gaps``. Two datasets on the same YPD plate then got
two media nodes and the campaign's join ("everything measured on YPD") could not
form. These tests pin the two halves of the fix: what the projection IGNORES (who
said it, how it was worded, what order it was listed in) and what it KEEPS (every
typed compositional slot, including a slot that is absent).
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from torchcell.datamodels import schema as s
from torchcell.datamodels.identity import (
    BIOLOGIC_PERTURBATION_IDENTITY_FIELDS,
    COMPOUND_IDENTITY_PRECEDENCE,
    CONCENTRATION_IDENTITY_FIELDS,
    ENVIRONMENT_IDENTITY_FIELDS,
    MEDIA_COMPONENT_IDENTITY_FIELDS,
    MEDIA_IDENTITY_FIELDS,
    PHYSICAL_PERTURBATION_IDENTITY_FIELDS,
    SMALL_MOLECULE_IDENTITY_FIELDS,
    SOLVENT_IDENTITY_FIELDS,
    TEMPERATURE_IDENTITY_FIELDS,
    compound_identity_key,
    environment_identity,
    environment_perturbation_identity,
    identity_sha256,
    media_identity,
    temperature_identity,
)
from torchcell.datamodels.media import MEDIA_LIBRARY, YPD, YPD_AGAR, YPD_LIQUID
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    SourcedValue,
)

_SHA = "0" * 64
_NACL_KEY = "FAPWRFPIFSIZLT-UHFFFAOYSA-M"


def _sourced(value: str, quote: str, citation_key: str) -> SourcedValue:
    return SourcedValue(
        value=value,
        quote=quote,
        provenance=Provenance(
            source_uri="paper.md", citation_key=citation_key, sha256=_SHA
        ),
    )


def _media_id(media: s.Media) -> str:
    return identity_sha256(media_identity(media))


def _glucose() -> s.Compound:
    return s.Compound(name="D-glucose", inchikey="WQZGKKKJIJFFOK-GASJEMHNSA-N")


def _plate(name: str, citation_key: str, quote: str, glucose_percent: float) -> s.Media:
    """One YP-glucose plate as a given paper would state it (its own quotes/notes)."""
    return s.Media(
        name=name,
        state="solid",
        is_synthetic=False,
        base_medium="YPD",
        components=[
            s.MediaComponent(
                compound=_glucose(),
                role=s.MediaComponentRole.carbon_source,
                concentration=s.Concentration(
                    value=glucose_percent, unit=s.ConcentrationUnit.percent_w_v
                ),
                provenance=[_sourced("20 g/L", quote, citation_key)],
                note=f"as stated by {citation_key}",
                defers_to=[citation_key],
            ),
            s.MediaComponent(
                compound=s.Compound(name="agar", chebi_id="CHEBI:2509"),
                role=s.MediaComponentRole.gelling_agent,
                concentration=s.Concentration(
                    value=2.0, unit=s.ConcentrationUnit.percent_w_v
                ),
                provenance=[_sourced("20 g/L agar", quote, citation_key)],
            ),
        ],
        provenance=[_sourced("recipe", quote, citation_key)],
    )


def _nacl(name: str, value: float, description: str) -> s.SmallMoleculePerturbation:
    return s.SmallMoleculePerturbation(
        description=description,
        compound=s.Compound(name=name, inchikey=_NACL_KEY),
        concentration=s.Concentration(value=value, unit=s.ConcentrationUnit.molar),
    )


# --------------------------------------------------------------------------- #
# Compound keys
# --------------------------------------------------------------------------- #
def test_compound_key_precedence_is_inchikey_then_chebi_then_cid_then_name() -> None:
    everything = s.Compound(
        name="sodium chloride",
        inchikey=_NACL_KEY,
        chebi_id="CHEBI:26710",
        pubchem_cid=5234,
    )
    assert compound_identity_key(everything) == f"inchikey:{_NACL_KEY}"
    assert (
        compound_identity_key(
            s.Compound(name="x", chebi_id="CHEBI:26710", pubchem_cid=5234)
        )
        == "chebi_id:CHEBI:26710"
    )
    assert compound_identity_key(s.Compound(name="x", pubchem_cid=5234)) == (
        "pubchem_cid:5234"
    )
    assert compound_identity_key(s.Compound(name="Ethanol")) == "name:ethanol"


def test_a_structure_string_alone_falls_through_to_the_name() -> None:
    """SMILES / InChI are not in the precedence: a toolkit spelling joins nothing."""
    assert "smiles" not in COMPOUND_IDENTITY_PRECEDENCE
    assert "inchi" not in COMPOUND_IDENTITY_PRECEDENCE
    compound = s.Compound(name="Ethanol", smiles="CCO")
    assert compound_identity_key(compound) == "name:ethanol"


def test_the_name_fallback_folds_documented_synonyms() -> None:
    assert compound_identity_key(s.Compound(name="NaCl")) == "name:sodium chloride"
    assert compound_identity_key(s.Compound(name=" EtOH ")) == "name:ethanol"


# --------------------------------------------------------------------------- #
# Media
# --------------------------------------------------------------------------- #
def test_the_same_medium_from_two_papers_is_one_media_node() -> None:
    """Different name, different quotes, different notes; identical composition."""
    mota = _plate("YPD (solid, 2% agar)", "mota2024", "20 g/L agar", 2.0)
    bloom = _plate("YPAD plates", "bloom2019", "2% agar plates", 2.0)
    assert mota.name != bloom.name
    assert mota.model_dump() != bloom.model_dump()
    assert _media_id(mota) == _media_id(bloom)


def test_every_library_medium_survives_a_rename_and_a_requote() -> None:
    for key, media in sorted(MEDIA_LIBRARY.items()):
        restated = media.model_copy(
            update={
                "name": f"{media.name} (as restated elsewhere)",
                "provenance": [
                    _sourced("recipe", "a different verbatim quote", "other")
                ],
            }
        )
        assert _media_id(restated) == _media_id(media), key


def test_a_concentration_change_changes_the_media_id() -> None:
    two_percent = _plate("YPD", "mota2024", "20 g/L", 2.0)
    four_percent = _plate("YPD", "mota2024", "20 g/L", 4.0)
    assert _media_id(two_percent) != _media_id(four_percent)


def test_a_state_change_changes_the_media_id() -> None:
    plate = _plate("YPD", "mota2024", "20 g/L", 2.0)
    culture = plate.model_copy(update={"state": "liquid"})
    assert _media_id(plate) != _media_id(culture)


def test_component_order_is_not_media_identity() -> None:
    plate = _plate("YPD", "mota2024", "20 g/L", 2.0)
    reversed_plate = plate.model_copy(update={"components": plate.components[::-1]})
    assert _media_id(reversed_plate) == _media_id(plate)


def test_a_dropout_is_part_of_media_identity() -> None:
    plate = _plate("YPD", "mota2024", "20 g/L", 2.0)
    minus_uracil = plate.model_copy(
        update={"dropouts": [s.Compound(name="uracil", chebi_id="CHEBI:17568")]}
    )
    assert _media_id(minus_uracil) != _media_id(plate)


def test_the_ypd_family_is_separated_by_composition_only() -> None:
    """What separates the YPD root from the plate is the agar row, not the quotes."""
    root = media_identity(YPD)
    plate = media_identity(YPD_AGAR)
    assert {k: v for k, v in root.items() if k != "components"} == {
        k: v for k, v in plate.items() if k != "components"
    }
    extra = [c for c in plate["components"] if c not in root["components"]]
    assert all(c["role"] == "gelling_agent" for c in extra), extra


def test_liquid_and_plate_ypd_are_different_media_nodes() -> None:
    assert _media_id(YPD_LIQUID) != _media_id(YPD_AGAR)


# --------------------------------------------------------------------------- #
# Environment perturbations
# --------------------------------------------------------------------------- #
def test_two_names_for_one_inchikey_are_one_perturbation_node() -> None:
    abbreviated = _nacl("NaCl", 0.4, "0.4 M NaCl stress")
    spelled_out = _nacl("sodium chloride", 0.4, "salt stress, 0.4 M")
    assert abbreviated.model_dump() != spelled_out.model_dump()
    assert identity_sha256(
        environment_perturbation_identity(abbreviated)
    ) == identity_sha256(environment_perturbation_identity(spelled_out))


def test_a_dose_change_changes_the_perturbation_id() -> None:
    assert identity_sha256(
        environment_perturbation_identity(_nacl("NaCl", 0.4, "x"))
    ) != identity_sha256(environment_perturbation_identity(_nacl("NaCl", 1.0, "x")))


def test_the_solvent_is_part_of_the_perturbation_id() -> None:
    plain = _nacl("NaCl", 0.4, "x")
    in_dmso = plain.model_copy(update={"solvent": s.Solvent(name="DMSO", percent=1.0)})
    assert identity_sha256(environment_perturbation_identity(plain)) != identity_sha256(
        environment_perturbation_identity(in_dmso)
    )


def test_a_physical_factor_keys_on_factor_magnitude_and_agent() -> None:
    ph45 = s.EnvironmentPhysicalPerturbation(
        factor=s.PhysicalFactor.ph,
        magnitude=s.Concentration(value=4.5, unit=s.ConcentrationUnit.ph),
    )
    ph35 = ph45.model_copy(
        update={
            "magnitude": s.Concentration(value=3.5, unit=s.ConcentrationUnit.ph),
            "description": "a differently worded pH edit",
        }
    )
    identity = environment_perturbation_identity(ph45)
    assert identity["factor"] == "pH"
    assert identity["agent"] is None
    assert identity_sha256(identity) != identity_sha256(
        environment_perturbation_identity(ph35)
    )


def test_a_biologic_keys_on_its_agent_and_its_dose() -> None:
    defensin = s.BiologicPerturbation(
        agent_class=s.BiologicAgentClass.peptide,
        name="plant defensin DmAMP1",
        uniprot_id="P0C8Y4",
        concentration=s.Concentration(value=5.0, unit=s.ConcentrationUnit.ug_per_ml),
    )
    louder = defensin.model_copy(
        update={
            "concentration": s.Concentration(
                value=50.0, unit=s.ConcentrationUnit.ug_per_ml
            )
        }
    )
    assert identity_sha256(
        environment_perturbation_identity(defensin)
    ) != identity_sha256(environment_perturbation_identity(louder))


def test_an_unprojectable_perturbation_raises_rather_than_falling_back() -> None:
    base = s.EnvironmentPerturbation(perturbation_type="mystery", description="?")
    with pytest.raises(TypeError):
        environment_perturbation_identity(base)


# --------------------------------------------------------------------------- #
# Temperature and environment
# --------------------------------------------------------------------------- #
def test_temperature_keys_on_value_and_unit() -> None:
    assert temperature_identity(s.Temperature(value=30.0)) == {
        "value": 30.0,
        "unit": "Celsius",
    }
    assert identity_sha256(
        temperature_identity(s.Temperature(value=30.0))
    ) != identity_sha256(temperature_identity(s.Temperature(value=26.0)))


def test_a_gapped_temperature_is_not_the_same_environment_as_a_measured_one() -> None:
    media = _plate("YPD", "mota2024", "20 g/L", 2.0)
    gapped = s.Environment(
        media=media,
        temperature=None,
        provenance_gaps=[
            ProvenanceGap(
                field="temperature", reason=ProvenanceGapReason.not_carried_by_curation
            )
        ],
    )
    measured = s.Environment(media=media, temperature=s.Temperature(value=30.0))
    assert environment_identity(gapped)["temperature"] is None
    assert identity_sha256(environment_identity(gapped)) != identity_sha256(
        environment_identity(measured)
    )


def test_provenance_gaps_are_not_themselves_part_of_environment_identity() -> None:
    media = _plate("YPD", "mota2024", "20 g/L", 2.0)
    silent = s.Environment(media=media, temperature=None)
    documented = s.Environment(
        media=media,
        temperature=None,
        provenance_gaps=[
            ProvenanceGap(
                field="temperature", reason=ProvenanceGapReason.not_reported_by_primary
            )
        ],
    )
    assert identity_sha256(environment_identity(silent)) == identity_sha256(
        environment_identity(documented)
    )


def test_perturbation_order_is_not_environment_identity() -> None:
    media = _plate("YPD", "mota2024", "20 g/L", 2.0)
    salt = _nacl("NaCl", 0.4, "salt")
    peroxide = s.SmallMoleculePerturbation(
        compound=s.Compound(
            name="hydrogen peroxide", inchikey="MHAJPDPJQMAIIY-UHFFFAOYSA-N"
        ),
        concentration=s.Concentration(value=2.0, unit=s.ConcentrationUnit.millimolar),
    )
    forward = s.Environment(media=media, perturbations=[salt, peroxide])
    backward = s.Environment(media=media, perturbations=[peroxide, salt])
    assert identity_sha256(environment_identity(forward)) == identity_sha256(
        environment_identity(backward)
    )


def test_duration_and_aerobicity_are_part_of_environment_identity() -> None:
    media = _plate("YPD", "mota2024", "20 g/L", 2.0)
    five = s.Environment(media=media, duration_generations=5.0)
    twenty = s.Environment(media=media, duration_generations=20.0)
    anaerobic = s.Environment(media=media, aerobicity="anaerobic")
    base = s.Environment(media=media)
    ids = {
        identity_sha256(environment_identity(environment))
        for environment in (five, twenty, anaerobic, base)
    }
    assert len(ids) == 4


# --------------------------------------------------------------------------- #
# Totality: the projections read real fields, and an absent field stays absent
# --------------------------------------------------------------------------- #
_FIELDS_READ: dict[type[BaseModel], tuple[str, ...]] = {
    s.Compound: COMPOUND_IDENTITY_PRECEDENCE + ("name",),
    s.Concentration: CONCENTRATION_IDENTITY_FIELDS,
    s.Solvent: SOLVENT_IDENTITY_FIELDS,
    s.MediaComponent: MEDIA_COMPONENT_IDENTITY_FIELDS,
    s.Media: MEDIA_IDENTITY_FIELDS,
    s.Temperature: TEMPERATURE_IDENTITY_FIELDS,
    s.SmallMoleculePerturbation: SMALL_MOLECULE_IDENTITY_FIELDS,
    s.EnvironmentPhysicalPerturbation: PHYSICAL_PERTURBATION_IDENTITY_FIELDS,
    s.BiologicPerturbation: BIOLOGIC_PERTURBATION_IDENTITY_FIELDS,
    s.Environment: ENVIRONMENT_IDENTITY_FIELDS,
}


def test_every_projected_field_exists_on_its_schema_class() -> None:
    for model, fields in _FIELDS_READ.items():
        for field in fields:
            assert field in model.model_fields, f"{model.__name__}.{field}"


def test_the_projection_keys_are_the_declared_slots() -> None:
    """Solvent is the one exception: its ``name`` folds INTO the compound key."""
    plate = _plate("YPD", "mota2024", "20 g/L", 2.0)
    assert set(media_identity(plate)) == set(MEDIA_IDENTITY_FIELDS)
    assert set(temperature_identity(s.Temperature(value=30.0))) == set(
        TEMPERATURE_IDENTITY_FIELDS
    )
    assert set(environment_identity(s.Environment(media=plate))) == set(
        ENVIRONMENT_IDENTITY_FIELDS
    )
    assert set(environment_perturbation_identity(_nacl("NaCl", 0.4, "x"))) == set(
        SMALL_MOLECULE_IDENTITY_FIELDS
    )
    component = media_identity(plate)["components"][0]
    assert set(component) == set(MEDIA_COMPONENT_IDENTITY_FIELDS)
    assert set(component["concentration"]) == set(CONCENTRATION_IDENTITY_FIELDS)


def test_an_absent_slot_stays_none_rather_than_disappearing() -> None:
    unamounted = s.Media(
        name="YNB, amount pending",
        state="liquid",
        is_synthetic=True,
        components=[
            s.MediaComponent(
                compound=s.Compound(name="biotin"), role=s.MediaComponentRole.vitamin
            )
        ],
    )
    identity = media_identity(unamounted)
    assert identity["base_medium"] is None
    assert identity["components"][0]["concentration"] is None
    environment = environment_identity(s.Environment(media=unamounted))
    assert environment["temperature"] is None
    assert environment["duration_hours"] is None
    assert environment["duration_generations"] is None


def test_identity_sha256_is_independent_of_key_insertion_order() -> None:
    assert identity_sha256({"a": 1, "b": [2, 3]}) == identity_sha256(
        {"b": [2, 3], "a": 1}
    )
