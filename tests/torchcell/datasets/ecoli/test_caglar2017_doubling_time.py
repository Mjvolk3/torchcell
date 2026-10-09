# tests/torchcell/datasets/ecoli/test_caglar2017_doubling_time.py
# [[tests.torchcell.datasets.ecoli.test_caglar2017_doubling_time]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/ecoli/test_caglar2017_doubling_time.py
"""Caglar 2017 Table S5 loader (``torchcell.datasets.ecoli.caglar2017_doubling_time``).

The synthetic tests write a Table S5 with the released header and the Methods' replicate
counts, plus a Table S1 carrying the base condition's own doubling-time fit repeated
across samples, and drive the two readers, the environment builder and both phenotype
builders. Nothing here reads ``$DATA_ROOT``: the readers take a path.

The ``@pytest.mark.data`` tests read the real mirror and pin the numbers the dendron note
and issue #776 state: 55 rows over 19 conditions with 2 replicates for exactly gluconate
and lactate; the interval asymmetric in 55 of 55 rows with exactly one non-bracketing
row (``Glycerol.tab`` replicate 1, released 95p ``-1027.769034``); the 9 base-condition
records; and Table S1's reference row at 53.679955 min.
"""

from __future__ import annotations

import json
import os.path as osp
from pathlib import Path

import pandas as pd
import pytest

import torchcell.datasets.ecoli.caglar2017 as base
import torchcell.datasets.ecoli.caglar2017_doubling_time as dt
from torchcell.datamodels.media import DAVIS_MINIMAL, DM500
from torchcell.datamodels.schema import (
    ABSOLUTE_MEASUREMENT_TYPES,
    MeasurementType,
    SampleUnit,
)

#: One synthetic value per (condition, replicate), chosen so the 19 conditions and the
#: Methods' replicate counts are reproduced exactly; the Glycerol replicate-1 row carries
#: a NEGATIVE upper limit, as the release does.
NEGATIVE_UPPER = -1027.769034


def _s5_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for condition in dt.CONDITIONS:
        n = 2 if condition in dt.TWO_REPLICATE_CONDITIONS else 3
        for replicate in range(1, n + 1):
            value = 60.0 + replicate
            upper = (
                NEGATIVE_UPPER
                if condition == "Glycerol.tab" and replicate == 1
                else value + 9.0
            )
            rows.append(
                {
                    dt.COL_CONDITION: condition,
                    dt.COL_REPLICATE: replicate,
                    dt.COL_MINUTES: value,
                    dt.COL_LOWER: value - 7.0,
                    dt.COL_UPPER: upper,
                    dt.COL_R_SQUARED: 0.99,
                }
            )
    return rows


def _write_s5(tmp_path: Path) -> Path:
    path = tmp_path / "srep45303-s6.csv"
    pd.DataFrame(_s5_rows()).to_csv(path, index=False)
    return path


def _write_s1(tmp_path: Path, *, repeats: int = 3, tuples: int = 1) -> Path:
    """Table S1 with the base condition's fit repeated across ``repeats`` sample rows."""
    experiment, carbon, mg_mm, na_mm = dt.CONDITIONS[dt.REFERENCE_CONDITION]
    rows = []
    for index in range(repeats):
        rows.append(
            {
                dt.S1_COL_EXPERIMENT: experiment,
                dt.S1_COL_CARBON: carbon,
                dt.S1_COL_MG: mg_mm,
                dt.S1_COL_NA: na_mm,
                dt.S1_COL_MINUTES: 53.679955 + (index if index < tuples - 1 else 0),
                dt.S1_COL_LOWER: 49.100298,
                dt.S1_COL_UPPER: 59.201792,
                dt.S1_COL_R_SQUARED: 0.98493,
            }
        )
    # one unrelated condition, so the join is a real selection rather than the whole file
    rows.append(
        {
            dt.S1_COL_EXPERIMENT: "NaCl_stress",
            dt.S1_COL_CARBON: "glucose",
            dt.S1_COL_MG: 0.8,
            dt.S1_COL_NA: 300.0,
            dt.S1_COL_MINUTES: 94.552067,
            dt.S1_COL_LOWER: 80.493097,
            dt.S1_COL_UPPER: 114.561433,
            dt.S1_COL_R_SQUARED: 0.859135,
        }
    )
    path = tmp_path / "srep45303-s2.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_read_table_s5_types_every_row_and_asserts_the_released_shape(
    tmp_path: Path,
) -> None:
    rows = dt.read_table_s5(_write_s5(tmp_path))
    assert len(rows) == dt.EXPECTED_RECORDS
    assert len({row.condition for row in rows}) == dt.EXPECTED_CONDITIONS
    counts = {
        name: sum(1 for row in rows if row.condition == name)
        for name in {row.condition for row in rows}
    }
    assert sorted(name for name, n in counts.items() if n == 2) == sorted(
        dt.TWO_REPLICATE_CONDITIONS
    )
    assert set(counts.values()) == {2, 3}


def test_read_table_s5_refuses_a_wrong_replicate_count(tmp_path: Path) -> None:
    rows = [
        r
        for r in _s5_rows()
        if not (r[dt.COL_CONDITION] == "Glucose.tab" and r[dt.COL_REPLICATE] == 3)
    ]
    rows.append({**rows[-1], dt.COL_CONDITION: "Lactate.tab", dt.COL_REPLICATE: 3})
    path = tmp_path / "s5.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(RuntimeError, match="replicates, the Methods state"):
        dt.read_table_s5(path)


def test_read_table_s5_refuses_an_unknown_condition(tmp_path: Path) -> None:
    rows = _s5_rows()
    rows[0] = {**rows[0], dt.COL_CONDITION: "Fructose.tab"}
    path = tmp_path / "s5.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(RuntimeError, match="conditions differ from CONDITIONS"):
        dt.read_table_s5(path)


def test_brackets_value_is_false_only_for_the_negative_upper_row(
    tmp_path: Path,
) -> None:
    rows = dt.read_table_s5(_write_s5(tmp_path))
    bad = [row for row in rows if not row.brackets_value]
    assert len(bad) == dt.EXPECTED_NON_BRACKETING
    assert (bad[0].condition, bad[0].replicate, bad[0].upper_95) == (
        "Glycerol.tab",
        1,
        NEGATIVE_UPPER,
    )


def test_read_reference_row_requires_one_condition_level_fit(tmp_path: Path) -> None:
    row = dt.read_reference_row(_write_s1(tmp_path))
    assert (row.minutes, row.lower_95, row.upper_95) == (
        53.679955,
        49.100298,
        59.201792,
    )
    assert row.sample_rows == 3
    assert row.experiment == "glucose_time_course"


def test_read_reference_row_refuses_two_distinct_fits(tmp_path: Path) -> None:
    with pytest.raises(
        RuntimeError, match="distinct\n?.*doubling-time tuples|distinct"
    ):
        dt.read_reference_row(_write_s1(tmp_path, repeats=3, tuples=3))


def test_build_environment_edits_match_the_condition() -> None:
    base_env = dt.build_environment("Glucose.tab")
    assert base_env.perturbations == []
    assert base_env.media.name == DM500.name
    temperature = base_env.temperature
    assert temperature is not None
    assert temperature.value == float(base.CULTURE_CONDITIONS.value)

    carbon_env = dt.build_environment("Lactate.tab")
    assert carbon_env.media.name == DAVIS_MINIMAL.name
    assert len(carbon_env.perturbations) == 1

    salt_env = dt.build_environment("NaCl_300_mM.tab")
    assert salt_env.media.name == DM500.name
    assert len(salt_env.perturbations) == 1

    low_mg_env = dt.build_environment("MgSO4-2_000.005_mM.tab")
    assert len(low_mg_env.perturbations) == 1


def test_exactly_the_three_base_conditions_carry_no_environmental_edit() -> None:
    unedited = sorted(
        name for name in dt.CONDITIONS if not dt.build_environment(name).perturbations
    )
    assert unedited == sorted(dt.BASE_CONDITIONS)


def test_phenotype_stores_both_limits_the_level_and_the_replicate_id(
    tmp_path: Path,
) -> None:
    row = next(
        r
        for r in dt.read_table_s5(_write_s5(tmp_path))
        if r.condition == "Glycerol.tab" and r.replicate == 1
    )
    phenotype = dt.phenotype(row)
    assert phenotype.measurement_type is MeasurementType.growth_rate
    assert phenotype.measurement_type in ABSOLUTE_MEASUREMENT_TYPES
    assert phenotype.environment_response == row.minutes
    assert phenotype.environment_response_lower == row.lower_95
    assert phenotype.environment_response_upper == NEGATIVE_UPPER
    assert phenotype.confidence_level == dt.CONFIDENCE_LEVEL
    assert phenotype.replicate_id == "1"
    assert phenotype.screen_id == "glycerol_time_course"
    assert phenotype.n_samples == 1
    assert phenotype.sample_unit is SampleUnit.biological_replicate
    # The asymmetric interval does not reduce to an SE, so the SE is a typed absence
    # rather than one side of the interval divided by 1.96.
    assert phenotype.environment_response_se is None
    assert phenotype.gapped_fields() == {
        "environment_response_se",
        "environment_response_uncertainty",
    }


def test_reference_phenotype_states_the_base_condition_value_not_zero(
    tmp_path: Path,
) -> None:
    reference = dt.reference_phenotype(dt.read_reference_row(_write_s1(tmp_path)))
    assert reference.environment_response == 53.679955
    assert reference.environment_response_lower == 49.100298
    assert reference.environment_response_upper == 59.201792
    assert reference.confidence_level == dt.CONFIDENCE_LEVEL
    assert reference.n_samples is None
    assert "n_samples" in reference.gapped_fields()


def test_released_summary_recomputes_the_declared_counts(tmp_path: Path) -> None:
    rows = dt.read_table_s5(_write_s5(tmp_path))
    assert dt.released_summary(rows) == {
        "records": dt.EXPECTED_RECORDS,
        "conditions": dt.EXPECTED_CONDITIONS,
        "non_bracketing": dt.EXPECTED_NON_BRACKETING,
    }


def test_every_condition_maps_to_a_table_s1_experiment() -> None:
    """``screen_id`` is Table S1's own ``experiment`` label, and it is what separates
    the four conditions that collide on the environment alone.
    """
    screens = {name: spec[0] for name, spec in dt.CONDITIONS.items()}
    assert screens["MgSO4_000.080_mM.tab"] == "MgSO4_stress_high"
    assert screens["MgSO4-2_000.080_mM.tab"] == "MgSO4_stress_low"
    assert len(set(screens.values())) == 7
    collide = [
        name for name in dt.CONDITIONS if not dt.build_environment(name).perturbations
    ]
    assert len({screens[name] for name in collide}) == len(collide)


# --------------------------------------------------------------------------- #
# Against the real mirror
# --------------------------------------------------------------------------- #
def _mirror(table: str) -> str:
    data_root = base._data_root()
    return str(base.raw_mirror_dir(data_root) / base.si_table_relpath(table))


@pytest.mark.data
def test_real_table_s5_is_55_rows_over_19_conditions() -> None:
    rows = dt.read_table_s5(_mirror("S5"))
    assert dt.released_summary(rows) == {
        "records": 55,
        "conditions": 19,
        "non_bracketing": 1,
    }


@pytest.mark.data
def test_real_interval_is_asymmetric_in_every_row() -> None:
    rows = dt.read_table_s5(_mirror("S5"))
    symmetric = [
        row
        for row in rows
        if abs((row.upper_95 - row.minutes) - (row.minutes - row.lower_95)) < 1e-9
    ]
    assert symmetric == []
    worst = min(rows, key=lambda row: row.upper_95)
    assert (worst.condition, worst.replicate) == ("Glycerol.tab", 1)
    assert worst.upper_95 == NEGATIVE_UPPER
    assert worst.minutes == 80.95212424
    assert worst.lower_95 == 38.94241445


@pytest.mark.data
def test_real_reference_row_is_table_s1s_base_condition_fit() -> None:
    row = dt.read_reference_row(_mirror("S1"))
    assert row.minutes == 53.67995457
    assert row.lower_95 <= row.minutes <= row.upper_95
    assert row.sample_rows == 36


@pytest.mark.data
def test_real_store_holds_55_records_and_its_declared_ledgers() -> None:
    root = osp.join(base._data_root(), dt.DATASET_ROOT_REL)
    accounting = json.loads(
        Path(root, "preprocess", "build_accounting.json").read_text()
    )
    assert accounting["kept_records"] == dt.EXPECTED_RECORDS
    assert accounting["unperturbed_records"] == dt.EXPECTED_UNPERTURBED
    assert sorted(accounting["unperturbed_conditions"]) == sorted(dt.BASE_CONDITIONS)
    assert accounting["non_bracketing_records"] == dt.EXPECTED_NON_BRACKETING
    not_stored = json.loads(Path(root, "preprocess", "not_stored.json").read_text())
    assert not_stored["issue"] == 776
    assert [item["n_values"] for item in not_stored["items"]] == [55, 1]
