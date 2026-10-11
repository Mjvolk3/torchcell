# tests/torchcell/datamodels/test_life_cycle_phenotype.py
# [[tests.torchcell.datamodels.test_life_cycle_phenotype]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datamodels/test_life_cycle_phenotype.py
"""The life-cycle phenotype family (De Chiara 2022): key grammar, bounds, assay coupling."""

from __future__ import annotations

import math
from typing import Any

import pytest
from pydantic import ValidationError

from torchcell.datamodels import schema as s
from torchcell.datamodels.schema import (
    LifeCycleAssay,
    LifeCycleMeasurementType,
    LifeCyclePhenotype,
    LifeCycleTimeUnit,
    life_cycle_key,
    parse_life_cycle_key,
)

SPO = LifeCycleMeasurementType.sporulation_efficiency
DYAD = LifeCycleMeasurementType.dyad_fraction
CLS = LifeCycleMeasurementType.stationary_phase_viability
H = LifeCycleTimeUnit.hour
D = LifeCycleTimeUnit.day


def _phenotype(**kw: Any) -> LifeCyclePhenotype:
    fields: dict[str, Any] = {
        "life_cycle": {"sporulation_efficiency@24h": 22.0},
        "assay": LifeCycleAssay.microscopy_ascus_count,
    }
    fields.update(kw)
    return LifeCyclePhenotype(**fields)


def test_key_round_trips_and_whole_times_have_one_spelling() -> None:
    assert life_cycle_key(SPO, 24.0, H) == "sporulation_efficiency@24h"
    assert life_cycle_key(CLS, 35, D) == "stationary_phase_viability@35d"
    assert life_cycle_key(DYAD, 1.5, H) == "dyad_fraction@1.5h"
    assert parse_life_cycle_key("stationary_phase_viability@7d") == (CLS, 7.0, D)
    assert parse_life_cycle_key("dyad_fraction@1.5h") == (DYAD, 1.5, H)


@pytest.mark.parametrize(
    ("key", "message"),
    [
        ("sporulation_efficiency@24.0h", "is not canonical"),
        ("sporulation_efficiency24h", "is not '<measurement type>@<time><h|d>'"),
        ("sporulation_efficiency@24min", "is not '<measurement type>@<time><h|d>'"),
        ("germination@24h", "'germination' is not a valid LifeCycleMeasurementType"),
    ],
)
def test_parse_refuses_a_malformed_or_unknown_key(key: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_life_cycle_key(key)


def test_life_cycle_key_refuses_a_negative_or_infinite_time() -> None:
    with pytest.raises(ValueError, match="finite and >= 0, got -1"):
        life_cycle_key(SPO, -1.0, H)
    with pytest.raises(ValueError, match="finite and >= 0, got inf"):
        life_cycle_key(SPO, math.inf, H)


def test_values_are_stored_sorted_and_verbatim() -> None:
    phenotype = _phenotype(
        life_cycle={
            "sporulation_efficiency@72h": 51.0,
            "dyad_fraction@72h": 15.68627451,
            "sporulation_efficiency@24h": 22.0,
        },
        cells_per_sample=100,
        cells_per_sample_is_minimum=True,
    )
    assert list(phenotype.life_cycle.items()) == [
        ("dyad_fraction@72h", 15.68627451),
        ("sporulation_efficiency@24h", 22.0),
        ("sporulation_efficiency@72h", 51.0),
    ]
    assert phenotype.label_name == "life_cycle"
    assert phenotype.graph_level == "global"
    assert repr(phenotype) == (
        "LifeCyclePhenotype(assay=microscopy_ascus_count, keys=['dyad_fraction@72h', "
        "'sporulation_efficiency@24h', 'sporulation_efficiency@72h'])"
    )


def test_a_part_over_its_whole_cannot_exceed_100_but_viability_can() -> None:
    with pytest.raises(ValidationError, match="cannot exceed 100"):
        _phenotype(life_cycle={"sporulation_efficiency@24h": 100.5})
    with pytest.raises(ValidationError, match="cannot exceed 100"):
        _phenotype(life_cycle={"dyad_fraction@72h": 101.0})
    viability = _phenotype(
        life_cycle={"stationary_phase_viability@7d": 114.5283589},
        assay=LifeCycleAssay.flow_cytometry_viability_stain,
    )
    assert viability.life_cycle["stationary_phase_viability@7d"] == 114.5283589


@pytest.mark.parametrize("bad", [-0.1, math.nan, math.inf])
def test_a_value_is_a_finite_non_negative_percentage(bad: float) -> None:
    with pytest.raises(ValidationError, match="finite percentage >= 0"):
        _phenotype(life_cycle={"sporulation_efficiency@24h": bad})


def test_a_value_must_be_a_number_and_the_map_non_empty() -> None:
    with pytest.raises(ValidationError, match="is not a number"):
        _phenotype(life_cycle={"sporulation_efficiency@24h": True})
    with pytest.raises(ValidationError, match="life_cycle cannot be empty"):
        _phenotype(life_cycle={})


@pytest.mark.parametrize(
    ("assay", "key"),
    [
        (LifeCycleAssay.flow_cytometry_scatter, "dyad_fraction@24h"),
        (LifeCycleAssay.flow_cytometry_viability_stain, "sporulation_efficiency@8d"),
        (LifeCycleAssay.microscopy_ascus_count, "stationary_phase_viability@7d"),
    ],
)
def test_an_assay_reports_only_the_measurements_it_produces(
    assay: LifeCycleAssay, key: str
) -> None:
    with pytest.raises(ValidationError, match=f"{assay.value} does not measure"):
        _phenotype(life_cycle={key: 1.0}, assay=assay)


def test_sample_count_and_unit_are_paired_and_counts_positive() -> None:
    with pytest.raises(ValidationError, match="set together or not at all"):
        _phenotype(n_samples=1)
    with pytest.raises(ValidationError, match="n_samples must be >= 1, got 0"):
        _phenotype(n_samples=0, sample_unit=s.SampleUnit.biological_replicate)
    with pytest.raises(ValidationError, match="cells_per_sample must be >= 1, got 0"):
        _phenotype(cells_per_sample=0)
    with pytest.raises(ValidationError, match="requires cells_per_sample"):
        _phenotype(cells_per_sample_is_minimum=True)
    stated = _phenotype(n_samples=1, sample_unit=s.SampleUnit.biological_replicate)
    assert (stated.n_samples, stated.sample_unit) == (1, "biological_replicate")


def test_the_family_is_registered_in_the_unions_and_maps() -> None:
    assert s.EXPERIMENT_TYPE_MAP["life_cycle"] is s.LifeCycleExperiment
    assert (
        s.EXPERIMENT_REFERENCE_TYPE_MAP["life_cycle"] is s.LifeCycleExperimentReference
    )
    assert s.LifeCycleExperiment.model_fields["environment"].annotation is (
        s.CultureEnvironment
    )
    assert s.LifeCycleExperimentReference.model_fields[
        "environment_reference"
    ].annotation is (s.CultureEnvironment)
    assert s.LIFE_CYCLE_BOUNDED_PERCENT == frozenset({SPO, DYAD})
    assert s.ConcentrationUnit.mg_nitrogen_per_l.value == "mg_N/L"
    assert s.MeasurementType.cell_yield in s.ABSOLUTE_MEASUREMENT_TYPES


def test_a_timed_pre_culture_states_its_duration() -> None:
    """De Chiara's 24 h YPD and 48 h YPA steps are timed cultures, not overnight ones."""
    step = s.PreCulture(source=s.PreCultureSource.timed_culture, duration_hours=48.0)
    assert (step.source.value, step.duration_hours) == ("timed_culture", 48.0)
    assert len({m.value for m in s.PreCultureSource}) == len(s.PreCultureSource) == 5


def test_the_sporulation_and_starvation_media_are_library_members() -> None:
    from torchcell.datamodels.media import CARBON_FREE_MEDIA, MEDIA_LIBRARY

    kac = MEDIA_LIBRARY["SPORULATION_KAC"]
    assert [(c.compound.name, c.concentration) for c in kac.components] == [
        (
            "potassium acetate",
            s.Concentration(value=2.0, unit=s.ConcentrationUnit.percent_w_v),
        )
    ]
    assert [g.field for g in kac.components[0].compound.provenance_gaps] == ["inchikey"]
    assert (kac.state, kac.base_medium) == ("liquid", "SPORULATION_KAC")
    pbs = MEDIA_LIBRARY["PBS"]
    assert (pbs.components, pbs.base_medium) == ([], "PBS")
    assert "PBS" in CARBON_FREE_MEDIA
