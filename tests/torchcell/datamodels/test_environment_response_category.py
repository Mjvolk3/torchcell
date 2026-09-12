# tests/torchcell/datamodels/test_environment_response_category.py
"""The typed qualitative axis of ``EnvironmentResponsePhenotype``.

A categorical screen used to write its call into a free-text ``category``, so each
screen's private vocabulary ("tolerant", "++", "wild type") was its own bucket and no
two screens joined. ``ResponseCategory`` is the shared axis, ``category_label`` keeps
the source's own word so the mapping stays auditable, and ``screen_id`` separates two
screening runs of the same compound at the same dose.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from torchcell.datamodels.schema import (
    CATEGORICAL_MEASUREMENT_TYPES,
    AssayType,
    EnvironmentResponsePhenotype,
    MeasurementType,
    ResponseCategory,
)


def test_response_category_covers_the_ordered_ladder_and_the_ungraded_calls() -> None:
    """The vocabulary is closed: a screen adds a mapping, never a new string."""
    assert {c.value for c in ResponseCategory} == {
        "enhanced",
        "no_change",
        "mildly_reduced",
        "reduced",
        "severely_reduced",
        "sensitive",
        "resistant",
        "not_determined",
    }


def test_no_two_response_categories_alias_one_value() -> None:
    """``a = "x"`` then ``b = "x"`` makes b an ALIAS, collapsing two terms silently."""
    assert len(list(ResponseCategory)) == len({c.value for c in ResponseCategory})


def test_a_categorical_record_takes_a_typed_call_and_keeps_the_source_word() -> None:
    phenotype = EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.categorical,
        assay_type=AssayType.spot_dilution,
        category=ResponseCategory.sensitive,
        category_label="sensitive",
    )
    assert phenotype.category is ResponseCategory.sensitive
    assert phenotype.category_label == "sensitive"
    assert phenotype.environment_response is None


def test_an_ordinal_record_carries_its_rank_alongside_the_typed_grade() -> None:
    """Smith 2006 scores a clear zone 4/3/2/1; the rank is ordered, not a quantity."""
    phenotype = EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.ordinal,
        assay_type=AssayType.halo_zone,
        environment_response=1.0,
        category=ResponseCategory.severely_reduced,
        category_label="1",
        units="clear-zone size: 4=larger than wild type ... 1=small or not detectable",
    )
    assert phenotype.category is ResponseCategory.severely_reduced
    assert phenotype.environment_response == 1.0


def test_ordinal_is_treated_as_a_call_not_a_quantity() -> None:
    """An ordinal reference may state the grade with no number, exactly as categorical."""
    assert MeasurementType.ordinal in CATEGORICAL_MEASUREMENT_TYPES
    phenotype = EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.ordinal,
        category=ResponseCategory.no_change,
        category_label="3",
    )
    assert phenotype.environment_response is None


@pytest.mark.parametrize("measurement_type", sorted(CATEGORICAL_MEASUREMENT_TYPES))
def test_a_call_readout_without_a_category_is_rejected(
    measurement_type: MeasurementType,
) -> None:
    with pytest.raises(ValidationError, match="requires `category`"):
        EnvironmentResponsePhenotype(measurement_type=measurement_type)


def test_a_numeric_readout_still_requires_its_number() -> None:
    with pytest.raises(ValidationError, match="numeric environment_response"):
        EnvironmentResponsePhenotype(measurement_type=MeasurementType.z_score)


def test_an_off_vocabulary_category_is_rejected_rather_than_stored() -> None:
    """A screen's private word must be MAPPED; storing it raw is the old failure."""
    with pytest.raises(ValidationError):
        EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.categorical,
            category="minor_to_moderate_growth_inhibition",  # type: ignore[arg-type]
        )


def test_a_verbatim_label_without_a_typed_call_is_rejected() -> None:
    with pytest.raises(ValidationError, match="category_label requires `category`"):
        EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.z_score,
            environment_response=-1.2,
            category_label="++",
        )


def test_screen_id_separates_two_runs_of_the_same_condition() -> None:
    """Same strain, same compound, same dose, two screens: two records, not one."""

    def one(screen_id: str | None) -> EnvironmentResponsePhenotype:
        return EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.sensitivity_score,
            assay_type=AssayType.pooled_competitive_growth_barcode,
            environment_response=2.5,
            screen_id=screen_id,
        )

    first, second = one("STUDY-0001"), one("STUDY-0002")
    assert first.screen_id != second.screen_id
    assert first.model_dump() != second.model_dump()
    assert one(None).screen_id is None


def test_the_category_axis_is_serialized_as_its_string_value() -> None:
    """The graph projects the value, so a round trip must not depend on the enum."""
    phenotype = EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.categorical,
        category=ResponseCategory.resistant,
        category_label="tolerant",
    )
    dumped = phenotype.model_dump()
    assert dumped["category"] == "resistant"
    assert EnvironmentResponsePhenotype(**dumped) == phenotype
