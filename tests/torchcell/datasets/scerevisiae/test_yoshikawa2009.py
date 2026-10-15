# tests/torchcell/datasets/scerevisiae/test_yoshikawa2009.py
# [[tests.torchcell.datasets.scerevisiae.test_yoshikawa2009]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_yoshikawa2009.py
"""Yoshikawa 2009 loader: the release reader, the strain names, the deposit and one build.

``process()`` reads a legacy BIFF workbook and no library in the env WRITES one, so the
reader runs on a stub ``xlrd`` book and the build stubs ``_read_release`` with parsed
rows; everything downstream runs for real, including the compound table and the
currency check (on small pin tables and a duck-typed genome).

Synthetic build (seven released rows; each condition's rates are distinct numbers).
Every strain is stored as released, under its ORF cell (upper-cased) and its name cell,
with no genome lookup (#1045): the renamed YFL035C-B, the retired YAR037W, the merged
YAR044W and the lower-case YAR043C-a are kept as the release names them.

Records: 7 strains x 4 conditions x 2 cultures = 56, in release-row order, each strain
nonstress (cultures 1, 2), 5% ethanol (1, 2), 8% ethanol (1, 2), 1 M NaCl (1, 2). Source
cells: 7 x 4 x 2 = 56; nothing refused.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import pytest
import xlrd

from torchcell.data import ManifestPinMismatchError
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import YPD_LIQUID
from torchcell.datamodels.schema import (
    AssayType,
    BarcodedKanMxDeletionPerturbation,
    ConcentrationUnit,
    CultureEnvironment,
    EukaryoticStrainBackground,
    Genotype,
    MatingType,
    MeasurementType,
    PreCultureSource,
    ResponseCategory,
    SampleUnit,
    SmallMoleculePerturbation,
    StrainEnvironmentResponseExperiment,
    StrainEnvironmentResponseExperimentReference,
    StrainReferenceGenome,
    UncertaintyType,
)
from torchcell.datasets.scerevisiae import yoshikawa2009 as y
from torchcell.verification.report import Provenance

_PROV = Provenance(source_uri="sgd", sha256="0" * 64)

#: (ORF cell, name cell) of the seven synthetic release rows.
ORFS: list[tuple[str, str]] = [
    ("YAL001C", "TFC3"),
    ("YAL002W", "VPS8"),
    ("YFL034C-A", "RPL22B"),
    ("YFL035C-B", "YFL035C-B"),
    ("YAR037W", "YAR037W"),
    ("YAR044W", "OSH1"),
    ("YAR043C-a", "YAR043C-a"),
]
CALLS = {"non-stress": 1, "5% ethanol": 2, "8% ethanol": 0, "1M NaCl": 2}


def _row(index: int, orf: str, name: str) -> y.ReleaseRow:
    """Rates 0.1 * (index + 1) + 0.01 * condition + 0.001 * culture, distinct per cell."""
    return y.ReleaseRow(
        orf=orf,
        name=name,
        conditions={
            label: y.ConditionCells(
                rates=(
                    0.1 * (index + 1) + 0.01 * k + 0.001,
                    0.1 * (index + 1) + 0.01 * k + 0.002,
                ),
                call=CALLS[label],
            )
            for k, label in enumerate(y.RELEASE_CONDITIONS)
        },
    )


ROWS = [_row(i, orf, name) for i, (orf, name) in enumerate(ORFS)]


def _build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """A real scratch build of the synthetic release (56 records)."""
    cls = y.EnvChemgenYoshikawa2009Dataset
    monkeypatch.setattr(cls, "_read_release", lambda self: list(ROWS))
    monkeypatch.setattr(y, "check_release", lambda rows: None)
    raw = tmp_path / "yoshikawa" / "raw"
    raw.mkdir(parents=True)
    (raw / y.TABLE_S1.filename).write_bytes(b"placeholder: the reader is stubbed")
    monkeypatch.setattr(y, "verify_raw_files", lambda raw_dir, pins: None)
    return cls(root=str(tmp_path / "yoshikawa"))


@pytest.fixture
def built(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    dataset = _build(tmp_path, monkeypatch)
    yield dataset
    dataset.close_lmdb()


def _exp(dataset: Any, idx: int) -> StrainEnvironmentResponseExperiment:
    return StrainEnvironmentResponseExperiment.model_validate(
        dataset[idx]["experiment"]
    )


def _deletion(dataset: Any, idx: int) -> BarcodedKanMxDeletionPerturbation:
    genotype = _exp(dataset, idx).genotype
    assert isinstance(genotype, Genotype)
    (perturbation,) = genotype.perturbations
    assert isinstance(perturbation, BarcodedKanMxDeletionPerturbation)
    return perturbation


def _ref(dataset: Any, idx: int) -> StrainEnvironmentResponseExperimentReference:
    return StrainEnvironmentResponseExperimentReference.model_validate(
        dict(dataset[idx]["reference"])
    )


# --------------------------------------------------------------------------- #
# The build
# --------------------------------------------------------------------------- #
def test_drop_log_accounts_every_released_cell_as_a_record(built: Any) -> None:
    """Every released cell is a record, the 5% ethanol columns included (#1045)."""
    assert len(built) == 56
    log = json.loads(Path(built.preprocess_dir, "dropped_records.json").read_text())
    assert (
        log["source_records"],
        log["kept_records"],
        log["dropped_records"],
        log["unperturbed_records"],
        log["rules"],
    ) == (56, 56, 0, 14, [])
    assert not Path(built.preprocess_dir, "systematic_name_ledger.json").exists()


def test_each_record_is_one_culture_with_the_strains_call(built: Any) -> None:
    """YAL001C (row 0): nonstress cultures 1 and 2, then 5% ethanol, 8% ethanol and
    1 M NaCl.
    """
    expected = [
        (0.101, "1st experiment", ResponseCategory.no_change, "1: no-change"),
        (0.102, "2nd experiment", ResponseCategory.no_change, "1: no-change"),
        (0.111, "1st experiment", ResponseCategory.resistant, "2: tolerant"),
        (0.112, "2nd experiment", ResponseCategory.resistant, "2: tolerant"),
        (0.121, "1st experiment", ResponseCategory.sensitive, "0: sensitive"),
        (0.122, "2nd experiment", ResponseCategory.sensitive, "0: sensitive"),
        (0.131, "1st experiment", ResponseCategory.resistant, "2: tolerant"),
        (0.132, "2nd experiment", ResponseCategory.resistant, "2: tolerant"),
    ]
    for idx, (rate, replicate, category, label) in enumerate(expected):
        phenotype = _exp(built, idx).phenotype
        assert math.isclose(phenotype.environment_response or 0.0, rate)
        assert (
            phenotype.replicate_id,
            phenotype.category,
            phenotype.category_label,
        ) == (replicate, category, label)
        assert phenotype.measurement_type is MeasurementType.growth_rate
        assert phenotype.assay_type is AssayType.liquid_od_growth
        assert (phenotype.n_samples, phenotype.sample_unit) == (
            1,
            SampleUnit.biological_replicate,
        )
        assert phenotype.environment_response_se is None
        assert {g.field for g in phenotype.provenance_gaps} == {
            "environment_response_uncertainty",
            "environment_response_se",
        }


def test_every_strain_is_a_kanmx4_deletion_named_as_released(built: Any) -> None:
    """Each strain's first record (every 8th) carries its released ORF cell, upper-cased,
    and its released name cell; no ConstructedOrf, no lookup (#1045). YFL034C-A and the
    renamed YFL035C-B stay two strains under two names.
    """
    stored = [
        (d.systematic_gene_name, d.perturbed_gene_name, d.constructed_orf)
        for d in (_deletion(built, 8 * i) for i in range(len(ORFS)))
    ]
    assert stored == [
        ("YAL001C", "TFC3", None),
        ("YAL002W", "VPS8", None),
        ("YFL034C-A", "RPL22B", None),
        ("YFL035C-B", "YFL035C-B", None),
        ("YAR037W", "YAR037W", None),
        ("YAR044W", "OSH1", None),
        ("YAR043C-A", "YAR043C-a", None),
    ]
    first = _deletion(built, 0)
    assert (first.collection, first.cassette) == ("Open Biosystems", "kanMX4")
    assert {g.field for g in first.provenance_gaps} == {"barcode", "downtag_barcode"}
    assert math.isclose(_exp(built, 24).phenotype.environment_response or 0.0, 0.401)


def test_two_released_orfs_that_coincide_upper_cased_stop_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cls = y.EnvChemgenYoshikawa2009Dataset
    rows = [_row(0, "YAL043C-a", "x"), _row(1, "YAL043C-A", "y")]
    monkeypatch.setattr(cls, "_read_release", lambda self: rows)
    monkeypatch.setattr(y, "check_release", lambda rows: None)
    monkeypatch.setattr(y, "verify_raw_files", lambda raw_dir, pins: None)
    raw = tmp_path / "y" / "raw"
    raw.mkdir(parents=True)
    (raw / y.TABLE_S1.filename).write_bytes(b"x")
    with pytest.raises(RuntimeError, match="coincide once upper-cased"):
        cls(root=str(tmp_path / "y"))


def test_environments_carry_ypd_the_stress_and_the_static_culture(built: Any) -> None:
    nonstress = _exp(built, 0).environment
    ethanol_5 = _exp(built, 2).environment
    ethanol = _exp(built, 4).environment
    nacl = _exp(built, 6).environment
    for env in (nonstress, ethanol_5, ethanol, nacl):
        assert isinstance(env, CultureEnvironment)
        assert env.media.name == YPD_LIQUID.name
        assert env.temperature is not None and env.temperature.value == 30.0
        assert env.aerobicity == "aerobic"
        assert env.culture_format is not None
        assert (
            env.culture_format.vessel,
            env.culture_format.working_volume_ul,
            env.culture_format.shaking_rpm,
        ) == ("96-well microtiter plate", 127.0, 0.0)
        assert [g.field for g in env.culture_format.provenance_gaps] == [
            "inoculum_od600"
        ]
        assert env.pre_culture is not None
        assert (env.pre_culture.source, env.pre_culture.duration_hours) == (
            PreCultureSource.timed_culture,
            24.0,
        )
    assert nonstress.perturbations == [] and nonstress.duration_hours == 9.0
    edits = []
    for env in (ethanol, nacl):
        assert env.duration_hours == 11.0
        (edit,) = env.perturbations
        assert isinstance(edit, SmallMoleculePerturbation)
        assert edit.solvent is None
        assert [g.field for g in edit.provenance_gaps] == ["solvent"]
        edits.append(
            (edit.compound.inchikey, edit.concentration.value, edit.concentration.unit)
        )
    assert edits == [
        (resolved_compound("ethanol").inchikey, 8.0, ConcentrationUnit.percent_v_v),
        (resolved_compound("sodium chloride").inchikey, 1.0, ConcentrationUnit.molar),
    ]
    # the 5% ethanol columns: the release's own dose, basis-free, and no read time
    assert ethanol_5.duration_hours is None
    assert [(g.field, g.reason.value, g.note) for g in ethanol_5.provenance_gaps] == [
        ("duration_hours", "not_reported_by_primary", y.FIVE_PERCENT_ETHANOL_NOTE)
    ]
    (edit,) = ethanol_5.perturbations
    assert isinstance(edit, SmallMoleculePerturbation)
    assert (
        edit.compound.inchikey,
        edit.concentration.value,
        edit.concentration.unit,
    ) == (resolved_compound("ethanol").inchikey, 5.0, ConcentrationUnit.percent)


def test_one_reference_per_condition_from_table_1(built: Any) -> None:
    got = []
    for idx in (0, 4, 6):
        ref = _ref(built, idx)
        phenotype = ref.phenotype_reference
        assert ref.environment_reference == _exp(built, idx).environment
        assert phenotype.measurement_type is MeasurementType.growth_rate
        assert phenotype.environment_response_uncertainty_type is (
            UncertaintyType.sample_sd
        )
        assert (phenotype.n_samples, phenotype.sample_unit) == (
            720,
            SampleUnit.biological_replicate,
        )
        got.append(
            (
                phenotype.environment_response,
                phenotype.environment_response_uncertainty,
                phenotype.environment_response_se,
            )
        )
    for (avg, sd, se), (want_avg, want_sd) in zip(
        got, [(0.452, 0.0044), (0.178, 0.0056), (0.203, 0.0076)], strict=True
    ):
        assert math.isclose(avg or 0.0, want_avg)
        assert math.isclose(sd or 0.0, want_sd)
        assert math.isclose(se or 0.0, want_sd / math.sqrt(720))
    # the 5% ethanol records read the nonstress standard strain (Table 1 has no 5% row)
    five = _ref(built, 2)
    assert five.environment_reference == _exp(built, 0).environment
    assert five.phenotype_reference.environment_response == 0.452
    assert five.phenotype_reference.units == y.FIVE_PERCENT_REFERENCE_UNITS
    assert _ref(built, 0).phenotype_reference.units == y.REFERENCE_UNITS
    genome = _ref(built, 0).genome_reference
    assert isinstance(genome, StrainReferenceGenome)
    assert isinstance(genome.background, EukaryoticStrainBackground)
    assert genome.strain == "BY4742"
    assert genome.background.mating_type is MatingType.alpha
    assert sorted(a.allele_name for a in genome.background.alleles) == [
        "his3Δ1",
        "leu2Δ0",
        "lys2Δ0",
        "ura3Δ0",
    ]
    assert genome.background.provenance is None
    assert [g.field for g in genome.background.provenance_gaps] == ["provenance"]


def test_release_fidelity_matches_the_build_and_names_a_changed_cell(
    built: Any,
) -> None:
    records = [built[i] for i in range(len(built))]
    passed = y.release_fidelity(records, ROWS)
    assert passed.passed, passed.message
    assert passed.details["n_records"] == passed.details["n_expected"] == 56
    changed = list(ROWS)
    cells = changed[1].conditions["1M NaCl"]
    changed[1] = changed[1].model_copy(
        update={
            "conditions": {
                **changed[1].conditions,
                "1M NaCl": cells.model_copy(update={"rates": (9.0, cells.rates[1])}),
            }
        }
    )
    failed = y.release_fidelity(records, changed)
    assert not failed.passed
    assert failed.details["n_mismatched"] == 1
    renamed = list(ROWS)
    renamed[1] = renamed[1].model_copy(update={"name": "vps8"})
    moved = y.release_fidelity(records, renamed)
    assert not moved.passed and moved.details["n_mismatched"] == 8


def test_inline_construction_hooks_are_inert(built: Any) -> None:
    assert built.preprocess_raw("frame") == "frame"
    with pytest.raises(NotImplementedError):
        built.create_experiment()
    assert built.raw_file_names == ["FYR_456_sm_tableS1.xls"]
    assert built.experiment_class is StrainEnvironmentResponseExperiment
    assert built.reference_class is StrainEnvironmentResponseExperimentReference


# --------------------------------------------------------------------------- #
# The release reader (on a stub xlrd book)
# --------------------------------------------------------------------------- #
class _Cell:
    def __init__(self, value: Any) -> None:
        self.value = value
        if value == "":
            self.ctype = xlrd.XL_CELL_EMPTY
        elif isinstance(value, str):
            self.ctype = xlrd.XL_CELL_TEXT
        else:
            self.ctype = xlrd.XL_CELL_NUMBER


class _Sheet:
    def __init__(self, rows: list[list[Any]]) -> None:
        self._rows = rows
        self.nrows = len(rows)
        self.ncols = len(rows[0])

    def cell(self, r: int, c: int) -> _Cell:
        return _Cell(self._rows[r][c])

    def cell_value(self, r: int, c: int) -> Any:
        return self._rows[r][c]


class _Book:
    def __init__(self, rows: list[list[Any]], sheets: list[str] | None = None) -> None:
        self._sheet = _Sheet(rows)
        self._names = ["data"] if sheets is None else sheets

    def sheet_names(self) -> list[str]:
        return self._names

    def sheet_by_index(self, index: int) -> _Sheet:
        return self._sheet


def _header() -> list[list[Any]]:
    conditions, columns = y._expected_header()
    return [conditions, columns]


def _body(orf: str = "YAL001C", call: float = 1.0) -> list[Any]:
    cells: list[Any] = [orf, "TFC3"]
    for k in range(4):
        cells.extend([0.4 + 0.01 * k, 0.41 + 0.01 * k, call])
    return cells


def _read(
    monkeypatch: pytest.MonkeyPatch,
    rows: list[list[Any]],
    sheets: list[str] | None = None,
) -> list[y.ReleaseRow]:
    monkeypatch.setattr(xlrd, "open_workbook", lambda path: _Book(rows, sheets))
    return y.read_release("stub.xls")


def test_reader_parses_the_header_names_rates_and_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (row,) = _read(monkeypatch, [*_header(), _body(call=2.0)])
    assert (row.orf, row.name) == ("YAL001C", "TFC3")
    assert list(row.conditions) == list(y.RELEASE_CONDITIONS)
    cells = row.conditions["8% ethanol"]
    assert cells.rates == pytest.approx((0.42, 0.43))
    assert cells.call == 2


@pytest.mark.parametrize(
    ("rows", "sheets", "message"),
    [
        ([*_header(), _body()], ["data", "other"], "sheets are"),
        ([[c for c in r[:13]] for r in [*_header(), _body()]], None, "13 columns"),
        ([["x", *_header()[0][1:]], _header()[1], _body()], None, "header differs"),
        ([*_header(), [1.0, *_body()[1:]]], None, "name cells are not text"),
        ([*_header(), [*_body()[:3], "n/a", *_body()[4:]]], None, "not a number"),
        ([*_header(), [*_body()[:2], math.inf, *_body()[3:]]], None, "non-finite"),
        ([*_header(), _body(call=3.0)], None, "call 3.0"),
        ([*_header(), _body(), _body()], None, "lists an ORF twice"),
    ],
    ids=[
        "two-sheets",
        "columns",
        "header",
        "name-not-text",
        "rate-not-number",
        "non-finite",
        "bad-call",
        "duplicate",
    ],
)
def test_reader_refuses_what_the_release_does_not_write(
    monkeypatch: pytest.MonkeyPatch,
    rows: list[list[Any]],
    sheets: list[str] | None,
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        _read(monkeypatch, rows, sheets)


def _release_with_table2_counts() -> list[y.ReleaseRow]:
    """4,729 rows whose kept-condition calls reproduce Table 2 exactly."""
    counts = y.TABLE1_COUNTS.value
    rows = []
    for i in range(4729):
        conditions = {}
        for spec in y.CONDITIONS:
            if spec.table2_label is None:
                conditions[spec.label] = y.ConditionCells(rates=(0.1, 0.1), call=1)
                continue
            sensitive, no_change, _ = counts[spec.table2_label]
            call = 0 if i < sensitive else 1 if i < sensitive + no_change else 2
            conditions[spec.label] = y.ConditionCells(rates=(0.1, 0.1), call=call)
        rows.append(y.ReleaseRow(orf=f"Y{i}", name=f"Y{i}", conditions=conditions))
    return rows


def test_check_release_holds_the_strain_count_and_table_2() -> None:
    rows = _release_with_table2_counts()
    y.check_release(rows)
    with pytest.raises(RuntimeError, match="4728 strains"):
        y.check_release(rows[:-1])
    flipped = list(rows)
    cells = flipped[0].conditions["8% ethanol"]
    flipped[0] = flipped[0].model_copy(
        update={
            "conditions": {
                **flipped[0].conditions,
                "8% ethanol": cells.model_copy(update={"call": 1}),
            }
        }
    )
    with pytest.raises(RuntimeError, match="8% ethanol calls"):
        y.check_release(flipped)


def test_condition_label_reads_the_one_compound_and_its_dose() -> None:
    """Ethanol is added at two doses, so the label is read off compound and dose."""
    envs = {spec.label: y.environment(spec).model_dump() for spec in y.CONDITIONS}
    assert {label: y.condition_label(env) for label, env in envs.items()} == {
        "non-stress": "non-stress",
        "5% ethanol": "5% ethanol",
        "8% ethanol": "8% ethanol",
        "1M NaCl": "1M NaCl",
    }
    two = dict(envs["1M NaCl"])
    two["perturbations"] = envs["1M NaCl"]["perturbations"] * 2
    with pytest.raises(RuntimeError, match="2 edits"):
        y.condition_label(two)


def test_table_1_rows_parse_the_printed_sd() -> None:
    rows = (y.TABLE1_NONSTRESS, y.TABLE1_ETHANOL, y.TABLE1_NACL)
    assert [(t.average, t.sensitive_below, t.tolerant_above) for t in rows] == [
        (0.452, 0.437, 0.466),
        (0.178, 0.160, 0.196),
        (0.203, 0.178, 0.229),
    ]
    assert [t.sd for t in rows] == pytest.approx([0.0044, 0.0056, 0.0076])
    # every released condition is stored; 5% ethanol reads the nonstress row
    assert [spec.label for spec in y.CONDITIONS] == list(y.RELEASE_CONDITIONS)
    assert [(s.key, s.reference_key, s.table2_label) for s in y.CONDITIONS] == [
        ("nonstress", "nonstress", "nonstress"),
        ("ethanol_5", "nonstress", None),
        ("ethanol_8", "ethanol_8", "8% ethanol"),
        ("nacl_1m", "nacl_1m", "1M NaCl"),
    ]


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def _synthetic_deposit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[y.RawFile, y.RawFile]:
    """Two synthetic SI files under tmp DATA_ROOT, pinned on the module."""
    si = tmp_path / "torchcell-raw" / y.CITATION_KEY / "si"
    si.mkdir(parents=True)
    files = []
    for name, payload in (
        (y.TABLE_S1.filename, b"publisher bytes"),
        (y.TABLE_S1_YEASTPHENOME.filename, b"yp-data bytes"),
    ):
        (si / name).write_bytes(payload)
        files.append(
            y.RawFile(
                filename=name,
                sha256=hashlib.sha256(payload).hexdigest(),
                bytes=len(payload),
            )
        )
    monkeypatch.setattr(y, "TABLE_S1", files[0])
    monkeypatch.setattr(y, "TABLE_S1_YEASTPHENOME", files[1])
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    return files[0], files[1]


def test_manifest_records_the_manual_and_the_scripted_retrieval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    xls, xlsx = _synthetic_deposit(tmp_path, monkeypatch)
    out = y.write_raw_manifest()
    manifest = y.load_manifest()
    assert out == tmp_path / "torchcell-raw" / y.CITATION_KEY / "manifest.json"
    assert (manifest.citation_key, manifest.doi, manifest.zotero_item_key) == (
        y.CITATION_KEY,
        y.DOI,
        "DCF6RHD6",
    )
    publisher, yeastphenome = manifest.files
    assert (publisher.path, publisher.role, publisher.sha256) == (
        "si/FYR_456_sm_tableS1.xls",
        "raw_data",
        xls.sha256,
    )
    assert publisher.retrieval is not None
    assert publisher.retrieval.method.value == "manual_browser"
    assert publisher.retrieval.params["deposit_record"] == "si/DEPOSIT.md"
    assert (yeastphenome.path, yeastphenome.role) == (
        "si/FYR_456_sm_tableS1.xlsx",
        "si_data",
    )
    assert yeastphenome.retrieval is not None
    assert yeastphenome.retrieval.params == {"url": y.YP_DATA_URL}
    assert y.YP_DATA_COMMIT in y.YP_DATA_URL
    assert y.manifest_sha256(manifest, xlsx.raw_rel) == xlsx.sha256
    with pytest.raises(KeyError, match="not in the raw-mirror manifest"):
        y.manifest_sha256(manifest, "si/other.xls")


def test_manifest_refuses_a_missing_or_different_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    xls, _ = _synthetic_deposit(tmp_path, monkeypatch)
    path = tmp_path / "torchcell-raw" / y.CITATION_KEY / xls.raw_rel
    path.write_bytes(b"other bytes")
    with pytest.raises(RuntimeError, match="is not the pinned"):
        y.write_raw_manifest()
    monkeypatch.setattr(
        y,
        "TABLE_S1",
        xls.model_copy(update={"sha256": hashlib.sha256(b"other bytes").hexdigest()}),
    )
    with pytest.raises(RuntimeError, match="is not the pinned 15 bytes"):
        y.write_raw_manifest()
    path.unlink()
    with pytest.raises(RuntimeError, match="raw mirror is missing"):
        y.write_raw_manifest()
    assert not (path.parent.parent / "manifest.json").exists()


def test_download_links_the_publisher_file_and_refuses_a_bad_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    xls, _ = _synthetic_deposit(tmp_path, monkeypatch)
    dataset = y.EnvChemgenYoshikawa2009Dataset.__new__(y.EnvChemgenYoshikawa2009Dataset)
    raw_dir = tmp_path / "store" / "raw"
    monkeypatch.setattr(
        y.EnvChemgenYoshikawa2009Dataset, "raw_dir", property(lambda self: str(raw_dir))
    )
    with pytest.raises(RuntimeError, match="run write_raw_manifest"):
        dataset.download()
    y.write_raw_manifest()
    dataset.download()
    assert (raw_dir / xls.filename).read_bytes() == b"publisher bytes"
    monkeypatch.setattr(y, "TABLE_S1", xls.model_copy(update={"sha256": "f" * 64}))
    with pytest.raises(ManifestPinMismatchError):
        dataset.download()


# --------------------------------------------------------------------------- #
# Verification entry point (its collaborators stubbed)
# --------------------------------------------------------------------------- #
def test_verify_build_adds_release_fidelity_to_the_family_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from torchcell.verification import runners
    from torchcell.verification.report import VerificationReport

    calls: list[tuple[str, str]] = []

    def entry(name: str, data_root: str) -> VerificationReport:
        calls.append((name, data_root))
        return VerificationReport(dataset_name=name, provenance=_PROV)

    monkeypatch.setattr(runners, "verify_environment_response_entry", entry)
    monkeypatch.setattr(runners, "load_records", lambda root: [])
    monkeypatch.setattr(y, "read_release", lambda path: list(ROWS))
    report = y.verify_build(str(tmp_path / "store"), data_root=str(tmp_path))
    assert calls == [("env_chemgen_yoshikawa2009", str(tmp_path))]
    (fidelity,) = report.results
    assert fidelity.name == "release_fidelity"
    assert not fidelity.passed  # an empty store holds none of the 56 released cells
    assert fidelity.details["n_expected"] == 56


def test_yeastphenome_screens_do_not_serve_this_paper() -> None:
    from torchcell.datasets.scerevisiae import yeastphenome

    assert y.PMID not in {screen["pmid"] for screen in yeastphenome.SCREENS}


# --------------------------------------------------------------------------- #
# Sourcing audits (``--data``; skip when the mirrors are absent)
# --------------------------------------------------------------------------- #
def _data_root() -> Path | None:
    root = os.environ.get("DATA_ROOT")
    return None if root is None else Path(root)


def _library_root() -> Path | None:
    root = _data_root()
    if root is None:
        return None
    library = root / "torchcell-library"
    return library if (library / y.CITATION_KEY / y.PAPER_MD).exists() else None


@pytest.mark.data
@pytest.mark.skipif(_library_root() is None, reason="literature mirror not present")
def test_every_sourced_value_is_a_verbatim_quote_in_its_pinned_mirror() -> None:
    root = _library_root()
    assert root is not None
    for value in y.SOURCED_VALUES:
        payload = value.source_path(root).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == value.provenance.sha256
        assert value.quote in payload.decode("utf-8"), value.value


@pytest.mark.data
@pytest.mark.skipif(
    _data_root() is None
    or not (_data_root() or Path())
    .joinpath(y.RAW_DIR_REL, y.TABLE_S1.raw_rel)
    .exists(),
    reason="raw mirror not present",
)
def test_the_deposited_release_reads_and_reproduces_table_2() -> None:
    root = _data_root()
    assert root is not None
    path = root / y.RAW_DIR_REL / y.TABLE_S1.raw_rel
    assert hashlib.sha256(path.read_bytes()).hexdigest() == y.TABLE_S1.sha256
    rows = y.read_release(path)
    y.check_release(rows)
    assert y.call_counts(rows, "5% ethanol") == (893, 3728, 108)
    assert sorted(r.orf for r in rows if r.orf != r.orf.upper()) == [
        "YAL043C-a",
        "YAR043C-a",
    ]
