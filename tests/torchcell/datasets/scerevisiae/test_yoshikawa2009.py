# tests/torchcell/datasets/scerevisiae/test_yoshikawa2009.py
# [[tests.torchcell.datasets.scerevisiae.test_yoshikawa2009]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_yoshikawa2009.py
"""Yoshikawa 2009 loader: the release reader, the strain names, the deposit and one build.

``process()`` reads a legacy BIFF workbook and no library in the env WRITES one, so the
reader runs on a stub ``xlrd`` book and the build stubs ``_read_release`` with parsed
rows; everything downstream runs for real, including the compound table and the
currency check (on small pin tables and a duck-typed genome).

Synthetic build (seven released rows; each condition's rates are distinct numbers):

    YAL001C    current                               kept
    YAL002W    current                               kept
    YFL034C-A  current                               kept
    YFL035C-B  renamed -> YFL034C-A (a pinned rename) kept, its released name recorded
    YAR037W    retired                               refused
    YAR044W    merged into YAR042W                   refused
    YDL134C-A  renamed -> YDL133C-A, unsourced       refused

Records: 4 strains x 3 conditions x 2 cultures = 24, in release-row order, each strain
nonstress (cultures 1, 2), 8% ethanol (1, 2), 1 M NaCl (1, 2). Source cells: 7 x 4 x 2 =
56; refused: the 5% ethanol columns (7 x 2 = 14) and 3 strains x 6 = 18.
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
    Genotype,
    MatingType,
    MeasurementType,
    OrfHistoryRelation,
    PreCultureSource,
    ResponseCategory,
    SampleUnit,
    SmallMoleculePerturbation,
    StrainEnvironmentResponseExperiment,
    StrainEnvironmentResponseExperimentReference,
    UncertaintyType,
)
from torchcell.datamodels.strain_background import R64_GFF
from torchcell.datasets.gene_alias_resolution import (
    AliasResolutionRule,
    GeneNameRefusal,
    GeneNameRefusalReason,
    GeneNameRefused,
    PinnedAliasResolution,
    pin_table,
)
from torchcell.datasets.scerevisiae import yoshikawa2009 as y
from torchcell.datasets.systematic_name_currency import (
    PinnedSystematicNameRefusal,
    refusal_table,
)
from torchcell.sequence.genome.base import GeneNameResolution, GeneNameStatus
from torchcell.verification.report import Provenance

_PROV = Provenance(source_uri="sgd", sha256="0" * 64)
CURRENT = GeneNameStatus.CURRENT
RENAMED = GeneNameStatus.RENAMED
RETIRED = GeneNameStatus.RETIRED

_RESOLUTIONS: dict[str, tuple[GeneNameStatus, str | None]] = {
    "YAL001C": (CURRENT, "YAL001C"),
    "YAL002W": (CURRENT, "YAL002W"),
    "YFL034C-A": (CURRENT, "YFL034C-A"),
    "YAR042W": (CURRENT, "YAR042W"),
    "YDL133C-A": (CURRENT, "YDL133C-A"),
    "YFL035C-B": (RENAMED, "YFL034C-A"),
    "YAR037W": (RETIRED, "YAR037W"),
    "YAR044W": (RENAMED, "YAR042W"),
    "YDL134C-A": (RENAMED, "YDL133C-A"),
    "YZZ001W": (RENAMED, "YAL002W"),
    "YZZ002W": (RENAMED, "YAL002W"),
}


class _Genome:
    """The lookups the currency check reads, table-driven."""

    def __init__(
        self, resolutions: dict[str, tuple[GeneNameStatus, str | None]] = _RESOLUTIONS
    ) -> None:
        self._resolutions = resolutions
        self.gene_set = {n for n, (st, _) in resolutions.items() if st is CURRENT}
        self.alias_to_systematic: dict[str, list[str]] = {
            n: [orf]
            for n, (st, orf) in resolutions.items()
            if st is RENAMED and orf is not None
        }
        self.feature_index: dict[str, Any] = {"standard_to_ids": {}}

    def resolve_gene_name(self, name: str) -> GeneNameResolution:
        status, orf = self._resolutions[name]
        return GeneNameResolution(input_name=name, status=status, systematic_name=orf)


def _rename(alias: str, orf: str) -> PinnedAliasResolution:
    return PinnedAliasResolution(
        alias=alias,
        systematic_name=orf,
        rule=AliasResolutionRule.SGD_SYSTEMATIC_RENAME,
        provenance=_PROV,
        quote=f"{alias} used to be the systematic name; changed to {orf}.",
    )


RENAMES = pin_table([_rename("YFL035C-B", "YFL034C-A")])
REFUSALS = refusal_table(
    [
        PinnedSystematicNameRefusal(
            name="YAR044W",
            reason=GeneNameRefusalReason.MERGED_INTO_LOCUS,
            locus="YAR042W",
            provenance=_PROV,
            quote="YAR044W was merged into YAR042W.",
            basis="merged",
        ),
        PinnedSystematicNameRefusal(
            name="YAR037W",
            reason=GeneNameRefusalReason.RETIRED_SYSTEMATIC_NAME,
            locus=None,
            provenance=_PROV,
            quote=None,
            basis="absent",
        ),
    ]
)
UNSOURCED = {
    "YDL134C-A": GeneNameRefusal(
        name="YDL134C-A",
        reason=GeneNameRefusalReason.NOT_CURRENT_UNPINNED,
        candidates=["YDL133C-A"],
        detail="history lists the name only",
    )
}
ORFS = [
    "YAL001C",
    "YAL002W",
    "YFL034C-A",
    "YFL035C-B",
    "YAR037W",
    "YAR044W",
    "YDL134C-A",
]
CALLS = {"non-stress": 1, "5% ethanol": 2, "8% ethanol": 0, "1M NaCl": 2}


def _row(index: int, orf: str) -> y.ReleaseRow:
    """Rates 0.1 * (index + 1) + 0.01 * condition + 0.001 * culture, distinct per cell."""
    return y.ReleaseRow(
        orf=orf,
        name=orf,
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


ROWS = [_row(i, orf) for i, orf in enumerate(ORFS)]


def _resolve(orfs: list[str], genome: Any = None) -> y.StrainResolution:
    return y.resolve_strains(
        orfs,
        _Genome() if genome is None else genome,
        renames=RENAMES,
        refusals=REFUSALS,
        unsourced=UNSOURCED,
        rows_per_orf=6,
    )


def _build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """A real scratch build of the synthetic release (24 records)."""
    cls = y.EnvChemgenYoshikawa2009Dataset
    monkeypatch.setattr(cls, "_read_release", lambda self: list(ROWS))
    monkeypatch.setattr(cls, "_resolve", lambda self, orfs: _resolve(orfs))
    monkeypatch.setattr(cls, "_common_names", lambda self: {"YAL001C": "TFC3"})
    monkeypatch.setattr(y, "check_release", lambda rows: None)
    raw = tmp_path / "yoshikawa" / "raw"
    raw.mkdir(parents=True)
    (raw / y.TABLE_S1.filename).write_bytes(b"placeholder: the reader is stubbed")
    monkeypatch.setattr(y, "verify_raw_files", lambda raw_dir, pins: None)
    return cls(root=str(tmp_path / "yoshikawa"), genome=None)


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
def test_drop_log_counts_the_condition_then_the_strains(built: Any) -> None:
    assert len(built) == 24
    log = json.loads(Path(built.preprocess_dir, "dropped_records.json").read_text())
    assert (
        log["source_records"],
        log["kept_records"],
        log["dropped_records"],
        log["unperturbed_records"],
    ) == (56, 24, 32, 8)
    assert [
        (r["rule"], r["scope"], r["n_records"], r["items"]) for r in log["rules"]
    ] == [
        ("condition_not_described_by_primary", "condition", 14, ["5% ethanol"]),
        ("released_name_retired", "strain", 6, ["YAR037W"]),
        ("released_name_merged_into_locus", "strain", 6, ["YAR044W"]),
        ("released_name_change_unsourced", "strain", 6, ["YDL134C-A"]),
    ]


def test_systematic_name_ledger_names_the_rename_and_the_three_refusals(
    built: Any,
) -> None:
    ledger = json.loads(
        Path(built.preprocess_dir, "systematic_name_ledger.json").read_text()
    )
    assert (ledger["n_rows_in"], ledger["n_rows_refused"], ledger["n_rows_out"]) == (
        42,
        18,
        24,
    )
    assert [
        (o["released_name"], o["stored_name"], o["n_rows"]) for o in ledger["renamed"]
    ] == [("YFL035C-B", "YFL034C-A", 6)]
    assert [
        (o["released_name"], o["reason"], o["locus"]) for o in ledger["refused"]
    ] == [
        ("YAR037W", "retired_systematic_name", None),
        ("YAR044W", "merged_into_locus", "YAR042W"),
        ("YDL134C-A", "not_current_unpinned", "YDL133C-A"),
    ]


def test_each_record_is_one_culture_with_the_strains_call(built: Any) -> None:
    """YAL001C (row 0): nonstress cultures 1 and 2, then 8% ethanol, then 1 M NaCl."""
    expected = [
        (0.101, "1st experiment", ResponseCategory.no_change, "1: no-change"),
        (0.102, "2nd experiment", ResponseCategory.no_change, "1: no-change"),
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


def test_the_strain_is_a_kanmx4_deletion_with_its_released_name_when_renamed(
    built: Any,
) -> None:
    first = _deletion(built, 0)
    assert (first.systematic_gene_name, first.perturbed_gene_name) == (
        "YAL001C",
        "TFC3",
    )
    assert (first.collection, first.cassette, first.constructed_orf) == (
        "Open Biosystems",
        "kanMX4",
        None,
    )
    assert {g.field for g in first.provenance_gaps} == {"barcode", "downtag_barcode"}
    direct = _deletion(built, 12)
    renamed = _deletion(built, 18)
    assert direct.systematic_gene_name == renamed.systematic_gene_name == "YFL034C-A"
    assert direct.constructed_orf is None
    assert renamed.constructed_orf is not None
    assert (
        renamed.constructed_orf.source_systematic_name,
        renamed.constructed_orf.relation,
        renamed.constructed_orf.deleted_span,
    ) == ("YFL035C-B", OrfHistoryRelation.alias, None)
    assert [g.field for g in renamed.constructed_orf.provenance_gaps] == [
        "deleted_span"
    ]
    assert math.isclose(_exp(built, 18).phenotype.environment_response or 0.0, 0.401)


def test_environments_carry_ypd_the_stress_and_the_static_culture(built: Any) -> None:
    nonstress = _exp(built, 0).environment
    ethanol = _exp(built, 2).environment
    nacl = _exp(built, 4).environment
    for env in (nonstress, ethanol, nacl):
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


def test_one_reference_per_condition_from_table_1(built: Any) -> None:
    got = []
    for idx in (0, 2, 4):
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
    genome = _ref(built, 0).genome_reference
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
    kept = _resolve(ORFS).kept
    passed = y.release_fidelity(records, ROWS, kept)
    assert passed.passed, passed.message
    assert passed.details["n_records"] == passed.details["n_expected"] == 24
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
    failed = y.release_fidelity(records, changed, kept)
    assert not failed.passed
    assert failed.details["n_mismatched"] == 1
    moved = y.release_fidelity(records, ROWS, {**kept, "YAL002W": "YAL001C"})
    assert not moved.passed and moved.details["n_mismatched"] == 6


def test_a_build_without_a_genome_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cls = y.EnvChemgenYoshikawa2009Dataset
    monkeypatch.setattr(cls, "_read_release", lambda self: list(ROWS))
    monkeypatch.setattr(y, "check_release", lambda rows: None)
    monkeypatch.setattr(y, "verify_raw_files", lambda raw_dir, pins: None)
    raw = tmp_path / "y" / "raw"
    raw.mkdir(parents=True)
    (raw / y.TABLE_S1.filename).write_bytes(b"x")
    with pytest.raises(RuntimeError, match="requires a genome"):
        cls(root=str(tmp_path / "y"), genome=None)
    dataset = cls.__new__(cls)
    dataset.genome = None
    with pytest.raises(RuntimeError, match="requires a genome"):
        dataset._common_names()


def test_inline_construction_hooks_are_inert(built: Any) -> None:
    assert built.preprocess_raw("frame") == "frame"
    with pytest.raises(NotImplementedError):
        built.create_experiment()
    assert built.raw_file_names == ["FYR_456_sm_tableS1.xls"]
    assert built.experiment_class is StrainEnvironmentResponseExperiment
    assert built.reference_class is StrainEnvironmentResponseExperimentReference


# --------------------------------------------------------------------------- #
# Strain names
# --------------------------------------------------------------------------- #
def test_resolve_keeps_live_and_renamed_and_refuses_the_rest() -> None:
    got = _resolve(ORFS)
    assert got.kept == {
        "YAL001C": "YAL001C",
        "YAL002W": "YAL002W",
        "YFL034C-A": "YFL034C-A",
        "YFL035C-B": "YFL034C-A",
    }
    assert got.refused == {
        "YAR037W": GeneNameRefusalReason.RETIRED_SYSTEMATIC_NAME,
        "YAR044W": GeneNameRefusalReason.MERGED_INTO_LOCUS,
        "YDL134C-A": GeneNameRefusalReason.NOT_CURRENT_UNPINNED,
    }
    assert got.shared_genes == {"YFL034C-A": ["YFL034C-A", "YFL035C-B"]}
    assert (got.released_name("YFL035C-B"), got.released_name("YAL001C")) == (
        "YFL035C-B",
        None,
    )


def test_a_lower_case_suffix_is_judged_upper_cased_and_recorded_verbatim() -> None:
    got = _resolve(["yfl035c-B"])
    assert got.kept == {"yfl035c-B": "YFL034C-A"}
    assert got.released_name("yfl035c-B") == "yfl035c-B"


def test_two_renamed_names_on_one_gene_stop_the_build() -> None:
    renames = pin_table([_rename("YZZ001W", "YAL002W")])
    genome = _Genome()
    with pytest.raises(RuntimeError, match="two renamed released ORFs"):
        y.resolve_strains(
            ["YZZ001W", "YZZ002W"],
            genome,
            renames={**renames, **pin_table([_rename("YZZ002W", "YAL002W")])},
            refusals={},
            unsourced={},
        )


def test_an_unsourced_change_the_genome_contradicts_stops_the_build() -> None:
    resolutions = {**_RESOLUTIONS, "YDL134C-A": (RETIRED, "YDL134C-A")}
    with pytest.raises(GeneNameRefused, match="pin_rule_contradicted"):
        _resolve(["YDL134C-A"], _Genome(resolutions))


def test_an_unpinned_non_current_name_stops_the_build() -> None:
    with pytest.raises(GeneNameRefused, match="not_current_unpinned"):
        _resolve(["YZZ001W"])


def test_the_pin_tables_extend_the_sga_tables_by_the_names_only_this_release_has() -> (
    None
):
    assert len(y.RENAMES) == 16 + 5 and len(y.REFUSALS) == 35 + 7
    assert sorted(p.alias for p in y.YOSHIKAWA_RENAMES) == [
        "YHR039C-B",
        "YML010W-A",
        "YML013C-A",
        "YML095C-A",
        "YMR158C-B",
    ]
    assert sorted((r.name, r.reason.value, r.locus) for r in y.YOSHIKAWA_REFUSALS) == [
        ("YAR043C-A", "retired_systematic_name", None),
        ("YCL003W", "merged_into_locus", "YCL004W"),
        ("YDL061W", "retired_systematic_name", None),
        ("YGR272C", "merged_into_locus", "YGR271C-A"),
        ("YHR034W", "retired_systematic_name", None),
        ("YHR096W", "retired_systematic_name", None),
        ("YKL147W", "retired_systematic_name", None),
    ]
    assert list(y.UNSOURCED_NAME_CHANGES) == ["YDL134C-A"]
    assert not set(y.UNSOURCED_NAME_CHANGES) & (set(y.RENAMES) | set(y.REFUSALS))


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
            sensitive, no_change, _ = counts[spec.table2_label]
            call = 0 if i < sensitive else 1 if i < sensitive + no_change else 2
            conditions[spec.label] = y.ConditionCells(rates=(0.1, 0.1), call=call)
        conditions["5% ethanol"] = y.ConditionCells(rates=(0.1, 0.1), call=1)
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


def test_condition_label_reads_the_one_compound() -> None:
    envs = {spec.label: y.environment(spec).model_dump() for spec in y.CONDITIONS}
    assert {label: y.condition_label(env) for label, env in envs.items()} == {
        "non-stress": "non-stress",
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
    assert y.REFUSED_CONDITIONS.keys() == {"5% ethanol"}
    assert [spec.label for spec in y.CONDITIONS] == [
        label for label in y.RELEASE_CONDITIONS if label not in y.REFUSED_CONDITIONS
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
    monkeypatch.setattr(y, "SCerevisiaeGenome", lambda **kwargs: _Genome())
    real = y.resolve_strains
    monkeypatch.setattr(
        y,
        "resolve_strains",
        lambda orfs, genome: real(
            orfs, genome, renames=RENAMES, refusals=REFUSALS, unsourced=UNSOURCED
        ),
    )
    report = y.verify_build(str(tmp_path / "store"), data_root=str(tmp_path))
    assert calls == [("env_chemgen_yoshikawa2009", str(tmp_path))]
    (fidelity,) = report.results
    assert fidelity.name == "release_fidelity"
    assert not fidelity.passed  # an empty store holds none of the 24 kept cells
    assert fidelity.details["n_expected"] == 24


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


_GFF = (
    "torchcell-genomes/sgd_S288C_R64-4-1_20230830/"
    "saccharomyces_cerevisiae_R64-4-1_20230830.gff"
)


@pytest.mark.data
@pytest.mark.skipif(
    _data_root() is None or not (_data_root() or Path()).joinpath(_GFF).exists(),
    reason="R64-4-1 genomes tier not present",
)
def test_every_new_pin_matches_its_sgd_history_bytes() -> None:
    root = _data_root()
    assert root is not None
    gff = (root / _GFF).read_bytes()
    assert hashlib.sha256(gff).hexdigest() == R64_GFF.sha256
    pins: list[PinnedAliasResolution | PinnedSystematicNameRefusal] = [
        *y.YOSHIKAWA_RENAMES,
        *y.YOSHIKAWA_REFUSALS,
    ]
    for pin in pins:
        if pin.provenance == R64_GFF:
            assert isinstance(pin, PinnedSystematicNameRefusal)
            assert pin.quote is None
            assert f"ID={pin.name};" not in gff.decode("utf-8")
            continue
        raw = (root / pin.provenance.source_uri).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == pin.provenance.sha256
        notes = [h["note"] for h in json.loads(raw)["locus"]["history"]]
        assert pin.quote is not None
        assert any(pin.quote in note for note in notes), pin.quote
    unsourced = root / "data/sgd/genome/genes/YDL133C-A.json"
    assert hashlib.sha256(unsourced.read_bytes()).hexdigest() in (
        y.UNSOURCED_NAME_CHANGES["YDL134C-A"].detail
    )
    notes = [h["note"] for h in json.loads(unsourced.read_bytes())["locus"]["history"]]
    assert [n for n in notes if "YDL134C-A" in n] == ["<b>Name:</b> YDL134C-A"]


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
