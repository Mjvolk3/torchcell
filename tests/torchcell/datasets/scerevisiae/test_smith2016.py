# tests/torchcell/datasets/scerevisiae/test_smith2016.py
"""Smith 2016 loader: the vendor-code drop, the shared medium, the dose parsing.

The retention and encoding tests are pure (no build tree, no network). The mirror audit
binds each module-level ``SourcedValue`` to its verbatim quote and skips when the
``torchcell-library`` mirror is not mounted.
"""

from __future__ import annotations

import os
import os.path as osp
from typing import cast

import pytest

from torchcell.datamodels.compound_identity import resolve_compound_identity
from torchcell.datamodels.media import MEDIA_LIBRARY, SC_URA
from torchcell.datamodels.schema import (
    AssayType,
    ConcentrationUnit,
    EnvironmentResponsePhenotype,
    MeasurementType,
    SampleUnit,
    SmallMoleculePerturbation,
    UncertaintyType,
)
from torchcell.datasets.scerevisiae import smith2016 as s
from torchcell.verification.sourced import SourcedValue, audit_sourced_value

#: Every non-DMSO public compound the screen dosed, and every vendor catalog code.
PUBLIC_DRUGS = [
    "Fluconazole",
    "Aureobasidin-A",
    "Cerulenin",
    "Rapamycin",
    "Doxorubicin",
    "Cantharidin",
    "NSC-180973",
    "Daunorubicin",
    "DMSO",
]
VENDOR_CODES = [
    "1181-0519",
    "4130-1276",
    "9121982",
    "9125678",
    "7312221",
    "ST016598",
    "CBF-666774",
    "6630449",
    "0KPI-0099",
    "9150499",
]


@pytest.mark.parametrize("drug", PUBLIC_DRUGS)
def test_public_drugs_carry_a_structure_identifier(drug: str) -> None:
    resolution = resolve_compound_identity(name=drug)
    assert resolution.identified, f"{drug} lost its identity"
    assert resolution.inchikey is not None


def test_the_released_tamoxifen_alias_is_what_keeps_nsc_180973() -> None:
    resolution = resolve_compound_identity(name="NSC-180973")
    assert resolution.name == "tamoxifen"
    assert resolution.inchikey == "NKANXQFJJICGDU-QPLCGJKRSA-N"


@pytest.mark.parametrize("code", VENDOR_CODES)
def test_vendor_catalog_codes_are_terminal_and_drop(code: str) -> None:
    resolution = resolve_compound_identity(name=code)
    assert not resolution.identified
    assert resolution.status.value == "PROPRIETARY"


def test_the_medium_is_the_shared_sc_ura_object() -> None:
    assert SC_URA in MEDIA_LIBRARY.values()
    assert MEDIA_LIBRARY["SC_URA"] is SC_URA
    assert [dropout.name for dropout in SC_URA.dropouts] == ["uracil"]


def test_dmso_control_dose_is_the_sourced_one_percent() -> None:
    concentration = s.parse_concentration(s.DMSO, 0.01)
    assert concentration.value == 1.0
    assert concentration.unit is ConcentrationUnit.percent_v_v
    assert s.DMSO_PERCENT.value == 1.0


@pytest.mark.parametrize(
    ("raw", "value", "unit"),
    [
        ("20 uM", 20.0, ConcentrationUnit.micromolar),
        ("2.2 nM", 2.2, ConcentrationUnit.nanomolar),
        ("1 mM", 1.0, ConcentrationUnit.millimolar),
    ],
)
def test_released_doses_parse_into_typed_units(
    raw: str, value: float, unit: ConcentrationUnit
) -> None:
    concentration = s.parse_concentration("Fluconazole", raw)
    assert concentration.value == value
    assert concentration.unit is unit


def test_an_unparseable_dose_raises_instead_of_being_guessed() -> None:
    with pytest.raises(ValueError, match="unparseable concentration"):
        s.parse_concentration("Fluconazole", "a lot")


def test_exposure_duration_is_recorded_in_generations_with_a_gap_for_hours() -> None:
    assert s.DURATION_GENERATIONS.value == 20.0
    assert "20 culture doublings" in s.DURATION_GENERATIONS.quote
    assert s.DURATION_HOURS_GAP.field == "duration_hours"
    assert s.DURATION_HOURS_GAP.reason.value == "not_reported_by_primary"


def test_the_assay_is_a_pooled_barcode_competition() -> None:
    assert s.ASSAY.value is AssayType.pooled_competitive_growth_barcode


def test_the_replicate_structure_is_recorded_outside_n_samples() -> None:
    """n_samples stays 1 because var(A) is ALREADY the combined estimate's variance."""
    assert s.REPLICATE_STRUCTURE.value == {"1% DMSO": 8, "20 uM fluconazole": 3}
    assert "combined" in s.REPLICATE_STRUCTURE.quote


def test_the_environment_is_sc_ura_with_one_typed_drug_edit() -> None:
    dataset = s.CrispriChemgenSmith2016Dataset.__new__(s.CrispriChemgenSmith2016Dataset)
    environment = s.CrispriChemgenSmith2016Dataset._environment(
        dataset, "Fluconazole", "20 uM"
    )
    assert environment.media is SC_URA
    assert environment.duration_generations == 20.0
    assert environment.duration_hours is None
    assert [gap.field for gap in environment.provenance_gaps] == ["duration_hours"]
    perturbation = cast(SmallMoleculePerturbation, environment.perturbations[0])
    assert len(environment.perturbations) == 1
    assert perturbation.compound.name == "fluconazole"
    assert perturbation.compound.inchikey is not None
    assert perturbation.solvent is None


def test_reference_is_the_uninduced_zero_baseline() -> None:
    dataset = s.CrispriChemgenSmith2016Dataset.__new__(s.CrispriChemgenSmith2016Dataset)
    environment = s.CrispriChemgenSmith2016Dataset._environment(dataset, s.DMSO, 0.01)
    dataset.name = "crispri_chemgen_smith2016"
    reference = s.CrispriChemgenSmith2016Dataset._reference(dataset, environment)
    phenotype = reference.phenotype_reference
    assert phenotype.environment_response == 0.0
    assert phenotype.assay_type is AssayType.pooled_competitive_growth_barcode


def test_uncertainty_is_the_variance_of_a_single_released_estimate() -> None:
    """SE = sqrt(var(A)/1): n_samples must stay 1 or the SE shrinks by sqrt(8)."""
    phenotype = EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.log2_ratio,
        assay_type=AssayType.pooled_competitive_growth_barcode,
        environment_response=-1.5,
        environment_response_uncertainty=0.25,
        environment_response_uncertainty_type=UncertaintyType.variance,
        n_samples=1,
        sample_unit=SampleUnit.pooled,
        units=s.UNITS,
    )
    assert phenotype.environment_response_se == pytest.approx(0.5)


def _library_root() -> str | None:
    data_root = os.environ.get("DATA_ROOT")
    if data_root is None:
        return None
    root = osp.join(data_root, "torchcell-library")
    return root if osp.isdir(root) else None


def test_every_sourced_value_is_backed_by_its_verbatim_quote() -> None:
    root = _library_root()
    if root is None:
        pytest.skip("torchcell-library mirror not mounted")
    values = [value for value in vars(s).values() if isinstance(value, SourcedValue)]
    assert values, "the loader must carry module-level SourcedValues"
    for value in values:
        result = audit_sourced_value(value, root)
        assert result.passed, f"{value.value!r}: {result.message}"
