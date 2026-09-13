# tests/torchcell/datasets/scerevisiae/test_hillenmeyer2008.py
"""Unit tests for the Hillenmeyer 2008 header parser, drop rules and raw-mirror map.

Every test runs on synthetic headers shaped like the released ones, so nothing here
touches the 41 MB matrices or the built LMDB.
"""

from __future__ import annotations

from typing import Any

import pytest

from torchcell.datamodels.media import (
    HILLENMEYER_DROPOUT_MEDIA,
    SC,
    SD,
    YP_GLYCEROL_LIQUID,
    YPD_LIQUID,
)
from torchcell.datamodels.schema import (
    ConcentrationUnit,
    EnvironmentPhysicalPerturbation,
    PhysicalFactor,
    SmallMoleculePerturbation,
)
from torchcell.datasets.scerevisiae.hillenmeyer2008 import (
    DROP_HEAT_SHOCK_CYCLE,
    DROP_UNIDENTIFIABLE,
    DROP_UNNAMED_MEDIA_AGENT,
    MATRICES,
    SUSPICIOUS_BATCHES,
    WAYBACK_TIMESTAMPS,
    build_environment,
    canonical_concentration,
    parse_columns,
    parse_condition,
    raw_relpaths,
    wayback_url,
)

BLANK = ("", "", "")


def _parse(cond1: str, conc1: str = "", unit1: str = "", *cond2: str) -> Any:
    second = cond2 if cond2 else BLANK
    return parse_condition(cond1, conc1, unit1, second[0], second[1], second[2])


# --------------------------------------------------------------------------- #
# Dose canonicalization (the sorbitol 1.5 M / 1.5e+06 uM duplicate)
# --------------------------------------------------------------------------- #
def test_molar_spellings_of_one_dose_canonicalize_identically() -> None:
    molar = canonical_concentration("1.5", "m")
    micro = canonical_concentration("1.5e+06", "um")
    assert (molar.value, molar.unit) == (1.5, ConcentrationUnit.molar)
    assert (micro.value, micro.unit) == (molar.value, molar.unit)


@pytest.mark.parametrize(
    ("conc", "unit", "value", "expected"),
    [
        ("1000", "um", 1.0, ConcentrationUnit.millimolar),
        ("1", "mm", 1.0, ConcentrationUnit.millimolar),
        ("2000", "um", 2.0, ConcentrationUnit.millimolar),
        ("6.9", "um", 6.9, ConcentrationUnit.micromolar),
        ("0.5", "um", 500.0, ConcentrationUnit.nanomolar),
        ("15000", "um", 15.0, ConcentrationUnit.millimolar),
    ],
)
def test_molar_family_picks_the_largest_unit_at_or_above_one(
    conc: str, unit: str, value: float, expected: ConcentrationUnit
) -> None:
    dose = canonical_concentration(conc, unit)
    assert (dose.value, dose.unit) == (value, expected)


def test_non_molar_units_are_left_alone_and_a_missing_unit_is_a_fixed_dose() -> None:
    ugml = canonical_concentration("1", "ug/ml")
    assert (ugml.value, ugml.unit) == (1.0, ConcentrationUnit.ug_per_ml)
    assert canonical_concentration("", "").basis is not None
    assert canonical_concentration("", "").value is None


def test_an_unknown_unit_raises_rather_than_defaulting() -> None:
    with pytest.raises(ValueError, match="unrecognized concentration unit"):
        canonical_concentration("5", "furlongs")


# --------------------------------------------------------------------------- #
# Condition classification
# --------------------------------------------------------------------------- #
def test_a_second_condition_is_appended_in_every_branch_not_only_small_molecules() -> (
    None
):
    """``pH7.5 + FK506`` must not be stored as a plain pH stress."""
    parse = _parse("pH7.5", "", "", "FK506", "1", "ug/ml")
    assert parse.drop_reason is None
    kinds = [p.perturbation_type for p in parse.perturbations]
    assert kinds == ["environment_physical", "small_molecule"]
    physical = parse.perturbations[0]
    assert isinstance(physical, EnvironmentPhysicalPerturbation)
    assert physical.factor is PhysicalFactor.ph
    assert physical.magnitude is not None and physical.magnitude.value == 7.5
    drug = parse.perturbations[1]
    assert isinstance(drug, SmallMoleculePerturbation)
    assert drug.compound.inchikey is not None
    assert drug.concentration.value == 1.0


def test_plain_ph_stays_a_single_physical_edit() -> None:
    parse = _parse("pH8")
    assert [p.perturbation_type for p in parse.perturbations] == [
        "environment_physical"
    ]


def test_irradiated_splits_three_ways() -> None:
    """Bare UV, and the two photo-activated crosslinkers, are different conditions."""
    only = _parse("no drug irradiated")
    assert [p.factor for p in only.perturbations] == [PhysicalFactor.radiation]

    for label, conc, unit in (
        ("angelicin irradiated", "62.5", "um"),
        ("psoralen irradiated", "0.5", "um"),
    ):
        parse = _parse(label, conc, unit)
        assert parse.drop_reason is None
        assert [p.perturbation_type for p in parse.perturbations] == [
            "small_molecule",
            "environment_physical",
        ]
        compound = parse.perturbations[0]
        assert isinstance(compound, SmallMoleculePerturbation)
        assert compound.compound.inchikey is not None
        assert compound.concentration.value is not None


def test_media_swaps_map_onto_the_shared_library_objects() -> None:
    assert _parse("minimal media").media == SD
    assert _parse("synthetic complete").media == SC
    assert _parse("YP glycerol").media == YP_GLYCEROL_LIQUID
    assert _parse("benomyl", "6.9", "um").media == YPD_LIQUID


def test_nutrient_dropouts_are_derived_media_not_perturbations() -> None:
    parse = _parse("tryptophan dropout")
    assert parse.perturbations == []
    assert parse.media == HILLENMEYER_DROPOUT_MEDIA["tryptophan dropout"]
    assert parse.media.base_medium == "SC"
    assert any(c.name == "L-tryptophan" for c in parse.media.dropouts)


def test_the_vitamin_dropout_control_is_plain_sc_not_a_vitamin_dropout() -> None:
    assert _parse("vitamin drop-out control media").media == SC


def test_temperature_is_a_scalar_not_a_perturbation() -> None:
    parse = _parse("37 degrees C")
    assert parse.perturbations == []
    assert parse.temperature_c == 37.0
    assert parse.media == YPD_LIQUID


def test_compounds_resolve_to_the_tables_canonical_name() -> None:
    """The source label is the lookup key; the canonical name is what is stored."""
    nacl = _parse("NaCl", "1", "m").perturbations[0]
    assert isinstance(nacl, SmallMoleculePerturbation)
    assert nacl.compound.name == "sodium chloride"
    assert nacl.compound.inchikey is not None


# --------------------------------------------------------------------------- #
# Drop rules
# --------------------------------------------------------------------------- #
def test_heat_shock_cycle_is_dropped_rather_than_stored_as_a_steady_temperature() -> (
    None
):
    parse = _parse("37c, 45c")
    assert parse.drop_reason == DROP_HEAT_SHOCK_CYCLE
    assert parse.temperature_c is None


@pytest.mark.parametrize(
    "label",
    [
        "chemical diversity labs 14a",
        "tyrphostin",
        "amphotericin",
        "FeCl4",
        "DMSO 1%",
        "methotrexate, 30ul total up/dn",
        "ptp2",
    ],
)
def test_a_compound_with_no_structure_identifier_drops_its_records(label: str) -> None:
    parse = _parse(label, "100", "um")
    assert parse.drop_reason == DROP_UNIDENTIFIABLE
    assert parse.drop_detail is not None and label in parse.drop_detail


def test_an_unidentifiable_second_compound_also_drops_the_column() -> None:
    parse = _parse("fluconazole", "50", "um", "amphotericin", "10", "um")
    assert parse.drop_reason == DROP_UNIDENTIFIABLE


def test_a_dose_on_a_media_swap_with_no_named_agent_drops() -> None:
    parse = _parse("minimal media", "400", "um")
    assert parse.drop_reason == DROP_UNNAMED_MEDIA_AGENT
    assert parse.drop_detail == "minimal media: 400 um"


def test_an_undosed_media_swap_is_kept() -> None:
    assert _parse("minimal media").drop_reason is None


# --------------------------------------------------------------------------- #
# Environment assembly
# --------------------------------------------------------------------------- #
def test_generation_sign_is_a_protocol_flag_so_duration_stores_the_magnitude() -> None:
    parse = _parse("benomyl", "6.9", "um")
    assert build_environment(parse, "-5gen").duration_generations == 5.0
    assert build_environment(parse, "5gen").duration_generations == 5.0
    assert build_environment(parse, "0gen").duration_generations == 0.0


def test_an_unstated_temperature_is_a_typed_gap_not_a_defaulted_thirty() -> None:
    environment = build_environment(_parse("benomyl", "6.9", "um"), "20gen")
    assert environment.temperature is None
    gaps = {gap.field: gap for gap in environment.provenance_gaps}
    assert "temperature" in gaps
    assert gaps["temperature"].reason.value == "deferred_pending_source_review"
    assert gaps["temperature"].resolve_with is not None
    assert "pierce" in str(gaps["temperature"].resolve_with.citation_key).lower()


def test_a_stated_temperature_carries_no_gap() -> None:
    environment = build_environment(_parse("37 degrees C"), "5gen")
    assert environment.temperature is not None
    assert environment.temperature.value == 37.0
    assert environment.provenance_gaps == []


# --------------------------------------------------------------------------- #
# Column grouping
# --------------------------------------------------------------------------- #
def _header(*columns: str) -> list[str]:
    return ["Orf", *columns]


def test_arrays_of_one_condition_scored_on_different_control_sets_stay_distinct() -> (
    None
):
    header = _header(
        "01_04_24_02:benomyl:6.9:um::::20gen:het_04_01_2:old scanner",
        "01_04_24_03:benomyl:6.9:um::::20gen:het_04_01_2:old scanner",
        "04_03_17_01:benomyl:6.9:um::::20gen:het_06_03:new scanner",
    )
    control_sets = {
        "01_04_24_02": "het_04_01_2::old scanner::20::tag3::YPD::dmso::0",
        "01_04_24_03": "het_04_01_2::old scanner::20::tag3::YPD::dmso::0",
        "04_03_17_01": "het_06_03::new scanner::20::tag3::YPD::dmso::0",
    }
    specs = parse_columns(header, control_sets)
    assert len({s.group_key for s in specs}) == 2


def test_the_signed_generation_count_survives_on_the_control_set() -> None:
    """``-5gen`` and ``5gen`` share a duration magnitude and must not merge."""
    header = _header(
        "03_06_05_01:benomyl:6.9:um::::-5gen:het_09_02:old scanner",
        "03_06_05_04:benomyl:6.9:um::::5gen:het_09_02:old scanner",
    )
    control_sets = {
        "03_06_05_01": "het_09_02::old scanner::-5::tag3::YPD::dmso::0",
        "03_06_05_04": "het_09_02::old scanner::5::tag3::YPD::dmso::0",
    }
    specs = parse_columns(header, control_sets)
    assert [s.environment.duration_generations for s in specs] == [5.0, 5.0]
    assert len({s.group_key for s in specs}) == 2


def test_an_array_with_no_control_set_raises() -> None:
    header = _header("01_04_24_02:benomyl:6.9:um::::20gen:het_04_01_2:old scanner")
    with pytest.raises(ValueError, match="no control set"):
        parse_columns(header, {})


def test_two_spellings_of_one_dose_group_together() -> None:
    header = _header(
        "01_12_11_02:sorbitol:1.5:m::::15gen:hom_05_01:old scanner",
        "01_11_28_02:sorbitol:1.5e+06:um::::15gen:hom_05_01:old scanner",
    )
    control_sets = dict.fromkeys(
        ["01_12_11_02", "01_11_28_02"], "hom_05_01::old scanner::15::tag3::YPD::dmso::0"
    )
    specs = parse_columns(header, control_sets)
    assert len({s.group_key for s in specs}) == 1


# --------------------------------------------------------------------------- #
# Raw mirror + constants
# --------------------------------------------------------------------------- #
def test_every_consumed_file_has_an_archived_url() -> None:
    consumed = {
        name
        for spec in MATRICES.values()
        for name in (spec.filename, spec.keyfile, spec.controls_file)
    }
    assert consumed <= set(WAYBACK_TIMESTAMPS)
    for name in WAYBACK_TIMESTAMPS:
        assert raw_relpaths()[name] == f"data/{name}"
        assert wayback_url(name).startswith("http://web.archive.org/web/")
        assert wayback_url(name).endswith(f"/{name}")


def test_the_som_names_ten_suspicious_construction_batches() -> None:
    assert len(SUSPICIOUS_BATCHES) == 10
    assert len(set(SUSPICIOUS_BATCHES)) == 10
    assert all(batch.startswith("chr") for batch in SUSPICIOUS_BATCHES)
