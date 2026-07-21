# tests/torchcell/datamodels/test_ui1_assay_biologic_compound_gap
# [[tests.torchcell.datamodels.test_ui1_assay_biologic_compound_gap]]
"""UI-1 schema foundation: AssayType, BiologicPerturbation, Compound gap affordance.

Covers the three additive, non-breaking additions from
``[[plan.env-schema-assay-compound-biologic.2026.07.20]]``. Enforcement (a
structure-or-gap validator on Compound, assay_type population) is UI-2/UI-3, so a
Compound with no structure and no gap MUST still construct here.
"""

import pytest
from pydantic import TypeAdapter

from torchcell.datamodels.schema import (
    AssayType,
    BiologicAgentClass,
    BiologicPerturbation,
    Compound,
    Concentration,
    ConcentrationUnit,
    EnvironmentPerturbationType,
    EnvironmentResponsePhenotype,
    MeasurementType,
    SmallMoleculePerturbation,
)
from torchcell.verification.sourced import ProvenanceGap, ProvenanceGapReason


def _erp(**kw: object) -> EnvironmentResponsePhenotype:
    """A minimal valid numeric env-response phenotype (z_score needs a numeric value)."""
    base: dict[str, object] = dict(
        measurement_type=MeasurementType.z_score, environment_response=1.0
    )
    base.update(kw)
    return EnvironmentResponsePhenotype(**base)  # type: ignore[arg-type]


# --- AssayType ------------------------------------------------------------- #
def test_assay_type_round_trips() -> None:
    for member in AssayType:
        assert AssayType(member.value) is member
        assert str(member.value) == member.value


def test_assay_type_default_none_and_set_value() -> None:
    assert _erp().assay_type is None
    erp = _erp(assay_type=AssayType.pooled_competitive_growth_barcode)
    assert erp.assay_type is AssayType.pooled_competitive_growth_barcode
    # survives a model_dump / re-validate round-trip
    assert EnvironmentResponsePhenotype(**erp.model_dump()).assay_type is (
        AssayType.pooled_competitive_growth_barcode
    )


def test_provenance_gap_on_assay_type_validates_when_none() -> None:
    # The inherited ProvenanceGapMixin already resolves the new field; no new validator.
    erp = _erp(
        assay_type=None,
        provenance_gaps=[
            ProvenanceGap(
                field="assay_type", reason=ProvenanceGapReason.not_reported_by_primary
            )
        ],
    )
    assert erp.assay_type is None
    assert erp.provenance_gaps[0].field == "assay_type"


def test_provenance_gap_on_assay_type_rejects_a_stored_value() -> None:
    # Honesty invariant: a gapped field must be None (cannot store AND declare missing).
    with pytest.raises(ValueError):
        _erp(
            assay_type=AssayType.spot_dilution,
            provenance_gaps=[
                ProvenanceGap(
                    field="assay_type",
                    reason=ProvenanceGapReason.not_reported_by_primary,
                )
            ],
        )


# --- BiologicPerturbation -------------------------------------------------- #
def test_biologic_perturbation_round_trips_and_small_molecule_unregressed() -> None:
    adapter: TypeAdapter[EnvironmentPerturbationType] = TypeAdapter(
        EnvironmentPerturbationType
    )
    bio = BiologicPerturbation(
        agent_class=BiologicAgentClass.peptide,
        name="plant defensin DmAMP1",
        concentration=Concentration(value=4.0, unit=ConcentrationUnit.millimolar),
    )
    back = adapter.validate_python(bio.model_dump())
    assert type(back) is BiologicPerturbation
    assert back.perturbation_type == "biologic"
    back_json = adapter.validate_json(bio.model_dump_json())
    assert type(back_json) is BiologicPerturbation
    # widening the union must not regress the existing leaves
    sm = SmallMoleculePerturbation(
        compound=Compound(name="isobutanol"),
        concentration=Concentration(value=1.0, unit=ConcentrationUnit.millimolar),
    )
    assert type(adapter.validate_python(sm.model_dump())) is SmallMoleculePerturbation


# --- Compound gap affordance (no enforcement this unit) -------------------- #
def test_compound_accepts_inchikey_gap_when_none() -> None:
    c = Compound(
        name="furfural",
        provenance_gaps=[
            ProvenanceGap(
                field="inchikey", reason=ProvenanceGapReason.not_reported_by_primary
            )
        ],
    )
    assert c.inchikey is None
    assert c.provenance_gaps[0].field == "inchikey"


def test_compound_with_no_structure_and_no_gap_still_constructs() -> None:
    # Proves UI-1 added the AFFORDANCE only -- no structure-or-gap enforcement (that is UI-2).
    c = Compound(name="acetic acid")
    assert c.inchikey is None
    assert c.provenance_gaps == []
