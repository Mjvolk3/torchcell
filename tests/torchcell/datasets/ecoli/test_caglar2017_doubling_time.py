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

import hashlib
import json
import os.path as osp
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import torchcell.datasets.ecoli.caglar2017 as base
import torchcell.datasets.ecoli.caglar2017_doubling_time as dt
from torchcell.datamodels.media import DAVIS_MINIMAL, DM500
from torchcell.datamodels.schema import (
    ABSOLUTE_MEASUREMENT_TYPES,
    AssemblyReferenceGenome,
    BacterialEnvironmentResponseExperiment,
    BacterialEnvironmentResponseExperimentReference,
    MeasurementType,
    SampleUnit,
)
from torchcell.verification.report import Level

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


def test_read_table_s5_refuses_a_row_count_other_than_the_released_one(
    tmp_path: Path,
) -> None:
    rows = _s5_rows()[:-1]
    path = tmp_path / "s5.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(RuntimeError, match="Table S5 has 54 rows, expected 55"):
        dt.read_table_s5(path)


def test_read_table_s5_refuses_a_repeated_condition_replicate_key(
    tmp_path: Path,
) -> None:
    """Replicate 1 twice keeps the condition's row count, so only the key check
    catches it; the replicate id is what keeps the records L1-distinct.
    """
    rows = _s5_rows()
    index = next(
        i
        for i, row in enumerate(rows)
        if row[dt.COL_CONDITION] == "Glucose.tab" and row[dt.COL_REPLICATE] == 2
    )
    rows[index] = {**rows[index], dt.COL_REPLICATE: 1}
    path = tmp_path / "s5.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(RuntimeError, match=r"repeats a \(condition, replicate\) key"):
        dt.read_table_s5(path)


def test_read_reference_row_refuses_a_non_bracketing_reference_interval(
    tmp_path: Path,
) -> None:
    """The record-level limits are stored verbatim whatever side they land on, but the
    REFERENCE is a single number the dataset asserts as its scale, so a reference
    interval that does not bracket its value stops the build.
    """
    experiment, carbon, mg_mm, na_mm = dt.CONDITIONS[dt.REFERENCE_CONDITION]
    path = tmp_path / "s1_bad.csv"
    pd.DataFrame(
        [
            {
                dt.S1_COL_EXPERIMENT: experiment,
                dt.S1_COL_CARBON: carbon,
                dt.S1_COL_MG: mg_mm,
                dt.S1_COL_NA: na_mm,
                dt.S1_COL_MINUTES: 53.679955,
                dt.S1_COL_LOWER: 49.100298,
                dt.S1_COL_UPPER: -1027.769034,
                dt.S1_COL_R_SQUARED: 0.98493,
            }
        ]
    ).to_csv(path, index=False)
    with pytest.raises(
        RuntimeError, match="reference interval does not bracket its value"
    ):
        dt.read_reference_row(path)


# --------------------------------------------------------------------------- #
# The loader end to end over a synthetic mirror
# --------------------------------------------------------------------------- #
#: The REL606 pair the genomes tier deposits, which ``l4_assembly_pin`` re-derives.
REL606_PIN = AssemblyReferenceGenome(
    species="Escherichia coli",
    strain="REL606",
    assembly_set="ecoli_B_REL606_ASM1798v1",
    assembly_accession="GCA_000017985.1",
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _staged_files(tmp_path: Path) -> dict[str, bytes]:
    """The mirror's seven files. Tables S5 and S1 are the synthetic release the readers
    above are driven with; the other four tables and the GenPept batch are placeholders
    this module never reads, present because the first deposit writes a manifest of every
    file it knows and Table S3's first column is what names the batch. Table S8
    (``-s9``) is one of those placeholders: #770's fold-change loader pinned it in the
    shared ``SI_TABLES``, so every file that map names has to be staged here.
    """
    return {
        "data/srep45303-s2.csv": _write_s1(tmp_path).read_bytes(),
        "data/srep45303-s3.csv": b",MURI_016\nECB_00001,4.6\n",
        "data/srep45303-s4.csv": b",MURI_016\nYP_1.1,0.9\n",
        "data/srep45303-s5.csv": b'"","Branch"\n"1","OAA from PEP"\n',
        "data/srep45303-s6.csv": _write_s5(tmp_path).read_bytes(),
        "data/srep45303-s9.csv": b"id,dataType,fullFileName\nYP_1.1,protein,x\n",
        "ncbi_protein/yp_batch_00.gp": b"LOCUS       YP_1\n//\n",
    }


@pytest.fixture
def mirror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The synthetic raw mirror deposited under a temporary ``DATA_ROOT``.

    The pins are repointed at the staged bytes, so ``download()`` and
    ``verify_raw_files`` run for real against this mirror rather than being stubbed.
    ``SI_TABLES`` is patched in BOTH modules: this one imported the name, so the two
    bindings are separate.
    """
    files = _staged_files(tmp_path)
    staging = tmp_path / "staging"
    for rel, data in files.items():
        (staging / rel).parent.mkdir(parents=True, exist_ok=True)
        (staging / rel).write_bytes(data)
    pinned = {
        table: (obj, _sha(files[f"data/{obj}"]), description)
        for table, (obj, _, description) in base.SI_TABLES.items()
    }
    monkeypatch.setattr(base, "SI_TABLES", pinned)
    monkeypatch.setattr(dt, "SI_TABLES", pinned)
    monkeypatch.setattr(
        base, "YP_BATCH_SHA256", (_sha(files["ncbi_protein/yp_batch_00.gp"]),)
    )
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "dr"))
    return base.deposit_raw_mirror(source_dir=staging)


@pytest.fixture
def built(
    tmp_path: Path, mirror: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[dt.DoublingTimeCaglar2017Dataset]:
    """The 55-record store, built from the synthetic mirror into a temporary root."""
    monkeypatch.setattr(dt, "assembly_reference", lambda strain, **_: REL606_PIN)
    dataset = dt.DoublingTimeCaglar2017Dataset(
        root=str(tmp_path / "doubling_time_caglar2017")
    )
    yield dataset
    dataset.close_lmdb()


def _record(dataset: dt.DoublingTimeCaglar2017Dataset, index: int) -> dict[str, Any]:
    item = dataset.get_single_item(index)
    assert item is not None
    return dict(item)


def _records(dataset: dt.DoublingTimeCaglar2017Dataset) -> list[dict[str, Any]]:
    return [_record(dataset, index) for index in range(len(dataset))]


def test_the_build_links_both_pinned_tables_and_writes_one_record_per_curve(
    built: dt.DoublingTimeCaglar2017Dataset,
) -> None:
    assert sorted(osp.basename(p) for p in built.raw_paths) == [
        "srep45303-s2.csv",
        "srep45303-s6.csv",
    ]
    assert len(built) == dt.EXPECTED_RECORDS
    records = _records(built)
    # One record per (condition, replicate). The condition is split across the two
    # axes the record carries it on: screen_id is Table S1's experiment, and the
    # environment separates the conditions within a series.
    assert Counter(rec["experiment"]["phenotype"]["screen_id"] for rec in records) == {
        "MgSO4_stress_high": 18,
        "MgSO4_stress_low": 15,
        "NaCl_stress": 12,
        "glucose_time_course": 3,
        "glycerol_time_course": 3,
        "gluconate_growth": 2,
        "lactate_growth": 2,
    }
    assert Counter(
        rec["experiment"]["phenotype"]["replicate_id"] for rec in records
    ) == {"1": 19, "2": 19, "3": 17}
    assert {rec["publication"]["doi"] for rec in records} == {base.PAPER_DOI}
    assert {len(rec["experiment"]["genotype"]["perturbations"]) for rec in records} == {
        0
    }


def test_the_stored_record_carries_the_released_number_and_both_limits(
    built: dt.DoublingTimeCaglar2017Dataset,
) -> None:
    """The non-bracketing row goes in verbatim: its 95% upper limit is NEGATIVE."""
    phenotypes = [rec["experiment"]["phenotype"] for rec in _records(built)]
    negative = [
        p for p in phenotypes if p["environment_response_upper"] == NEGATIVE_UPPER
    ]
    assert len(negative) == dt.EXPECTED_NON_BRACKETING
    stored = negative[0]
    assert stored["measurement_type"] == MeasurementType.growth_rate.value
    assert stored["environment_response"] == 61.0
    assert stored["environment_response_lower"] == 54.0
    assert stored["confidence_level"] == dt.CONFIDENCE_LEVEL
    assert (stored["replicate_id"], stored["screen_id"]) == (
        "1",
        "glycerol_time_course",
    )
    assert stored["units"] == dt.UNITS
    reference = next(
        rec["reference"]["phenotype_reference"]
        for rec in _records(built)
        if rec["experiment"]["phenotype"]["environment_response_upper"]
        == NEGATIVE_UPPER
    )
    assert reference["environment_response"] == 53.679955


def test_the_build_ledgers_state_every_declared_count(
    built: dt.DoublingTimeCaglar2017Dataset,
) -> None:
    out = Path(built.preprocess_dir)
    accounting = json.loads((out / "build_accounting.json").read_text())
    assert (accounting["source_rows"], accounting["kept_records"]) == (
        dt.EXPECTED_RECORDS,
        dt.EXPECTED_RECORDS,
    )
    assert accounting["conditions"] == dt.EXPECTED_CONDITIONS
    assert sorted(
        name for name, n in accounting["replicate_counts"].items() if n == 2
    ) == sorted(dt.TWO_REPLICATE_CONDITIONS)
    assert accounting["unperturbed_records"] == dt.EXPECTED_UNPERTURBED
    assert sorted(accounting["unperturbed_conditions"]) == sorted(dt.BASE_CONDITIONS)
    assert accounting["non_bracketing_records"] == dt.EXPECTED_NON_BRACKETING
    assert [
        (row["condition"], row["replicate"], row["upper_95"])
        for row in accounting["non_bracketing_rows"]
    ] == [("Glycerol.tab", 1, NEGATIVE_UPPER)]
    assert accounting["reference_minutes"] == 53.679955
    assert sum(accounting["screen_ids"].values()) == dt.EXPECTED_RECORDS

    dropped = json.loads((out / "dropped_records.json").read_text())
    assert (dropped["dropped_records"], dropped["rules"]) == (0, [])

    not_stored = json.loads((out / "not_stored.json").read_text())
    assert not_stored["issue"] == 776
    assert [item["n_values"] for item in not_stored["items"]] == [
        dt.EXPECTED_RECORDS,
        1,
    ]
    assert len(not_stored["items"][0]["values"]) == dt.EXPECTED_RECORDS

    spread = json.loads((out / "base_condition_spread.json").read_text())
    assert sorted(spread["mean_minutes_per_base_condition"]) == sorted(
        dt.BASE_CONDITIONS
    )
    assert spread["reference_minutes_table_s1"] == 53.679955

    sourced = json.loads((out / "sourced_values.json").read_text())
    assert sorted(sourced) == sorted(dt.SOURCED_VALUES)

    table = pd.read_csv(out / "doubling_times.csv")
    assert len(table) == dt.EXPECTED_RECORDS
    assert int((~table["brackets_value"]).sum()) == dt.EXPECTED_NON_BRACKETING


def test_verify_build_passes_the_absolute_gate_and_writes_its_report(
    built: dt.DoublingTimeCaglar2017Dataset,
) -> None:
    report = dt.verify_build(built.root)

    assert report.passed
    assert report.dataset_name == "doubling_time_caglar2017"
    assert report.provenance.citation_key == base.CITATION_KEY
    assert report.provenance.sha256 == base.SI_TABLES["S5"][1]
    by_name = {result.name: result for result in report.results}
    # The replicate_id in the L1 key is what makes 55 per-replicate rows 55 records
    # rather than 19 with 36 duplicates.
    assert by_name["pair_uniqueness"].details == {
        "n_pairs": dt.EXPECTED_RECORDS,
        "n_duplicated": 0,
    }
    interval = by_name["interval_orientation"].details
    assert (interval["n_intervals"], interval["n_non_bracketing"]) == (
        dt.EXPECTED_RECORDS,
        dt.EXPECTED_NON_BRACKETING,
    )
    assert interval["expected_non_bracketing"] == dt.EXPECTED_NON_BRACKETING
    perturbed = by_name["environment_perturbed"].details
    assert (perturbed["n_records"], perturbed["n_missing"]) == (
        dt.EXPECTED_RECORDS,
        dt.EXPECTED_UNPERTURBED,
    )
    assert by_name["reference_zero"].details["rule"] == "absolute_reference"
    pin = by_name["assembly_pin_resolves"]
    assert pin.level is Level.L4
    assert pin.details == {
        "pins": [(REL606_PIN.assembly_set, REL606_PIN.assembly_accession)],
        "expected": (REL606_PIN.assembly_set, REL606_PIN.assembly_accession),
        "n_records": dt.EXPECTED_RECORDS,
    }
    written = json.loads(
        Path(built.root, "preprocess", "verification_report.json").read_text()
    )
    assert [r["name"] for r in written["results"]] == [
        result.name for result in report.results
    ]


def test_l4_assembly_pin_fails_on_a_record_pinning_another_assembly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dt, "assembly_reference", lambda strain, **_: REL606_PIN)
    other = ("ecoli_K12_MG1655_ASM584v2", "GCA_000005845.2")
    records = [
        {
            "reference": {
                "genome_reference": {
                    "assembly_set": other[0],
                    "assembly_accession": other[1],
                }
            }
        }
    ]
    result = dt.l4_assembly_pin(records)
    assert not result.passed
    assert result.details["pins"] == [other]
    assert result.details["expected"] == (
        REL606_PIN.assembly_set,
        REL606_PIN.assembly_accession,
    )
    assert "the deposited assembly report names" in result.message


def test_the_two_base_class_hooks_this_loader_does_not_use(
    built: dt.DoublingTimeCaglar2017Dataset,
) -> None:
    """``process()`` builds the records inline, so ``preprocess_raw`` passes its frame
    through and ``create_experiment`` refuses rather than half-working.
    """
    frame = pd.DataFrame({"a": [1]})
    assert built.preprocess_raw(frame) is frame
    with pytest.raises(NotImplementedError, match="builds records in process"):
        built.create_experiment()


def test_the_loader_declares_the_two_schema_classes_it_writes() -> None:
    dataset = dt.DoublingTimeCaglar2017Dataset.__new__(dt.DoublingTimeCaglar2017Dataset)
    assert dataset.experiment_class is BacterialEnvironmentResponseExperiment
    assert dataset.reference_class is BacterialEnvironmentResponseExperimentReference
    assert dt.DoublingTimeCaglar2017Dataset.has_gene_perturbations is False


def test_the_build_refuses_a_declared_unperturbed_count_the_table_does_not_hold(
    tmp_path: Path, mirror: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 9 base-condition records are an oracle, not a side effect: if the release
    moved, the build stops instead of storing a different number of them.
    """
    monkeypatch.setattr(dt, "assembly_reference", lambda strain, **_: REL606_PIN)
    monkeypatch.setattr(dt, "EXPECTED_UNPERTURBED", 8)
    with pytest.raises(
        RuntimeError,
        match="9 records carry no environmental edit, the module declares 8",
    ):
        dt.DoublingTimeCaglar2017Dataset(root=str(tmp_path / "wrong_unperturbed"))


def test_the_build_refuses_a_declared_non_bracketing_count_the_table_does_not_hold(
    tmp_path: Path, mirror: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dt, "assembly_reference", lambda strain, **_: REL606_PIN)
    monkeypatch.setattr(dt, "EXPECTED_NON_BRACKETING", 0)
    with pytest.raises(
        RuntimeError,
        match="1 released intervals do not bracket their value, the module declares 0",
    ):
        dt.DoublingTimeCaglar2017Dataset(root=str(tmp_path / "wrong_non_bracketing"))


def test_main_builds_the_dev_tree_store_then_verifies_it(
    mirror: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(dt, "assembly_reference", lambda strain, **_: REL606_PIN)

    assert dt.main(["build"]) == 0
    build_out = capsys.readouterr().out
    assert f"len = {dt.EXPECTED_RECORDS}" in build_out
    assert f'"kept_records": {dt.EXPECTED_RECORDS}' in build_out
    assert Path(tmp_path, "dr", dt.DATASET_ROOT_REL, "processed", "lmdb").is_dir()

    assert dt.main(["verify"]) == 0
    assert "doubling_time_caglar2017: PASS" in capsys.readouterr().out


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
