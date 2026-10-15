# tests/torchcell/datasets/ecoli/test_baba2006_growth.py
# [[tests.torchcell.datasets.ecoli.test_baba2006_growth]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/ecoli/test_baba2006_growth.py
"""Baba 2006 Table 3's OD600 arm: the readings, the released identity, the gapped parent.

Hermetic tests exercise the cell parser and the record builders; ``data`` tests read the
deposited ``msb4100050-s5.xls`` and the built dev store.
"""

from __future__ import annotations

import json
import os
import os.path as osp
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from torchcell.data.experiment_dataset import file_sha256
from torchcell.datamodels.schema import (
    DeletionStrainPerturbation,
    EnvironmentResponsePhenotype,
    MeasurementType,
    ReferenceGenome,
)
from torchcell.datasets.ecoli import baba2006 as bb
from torchcell.datasets.ecoli import baba2006_growth as m
from torchcell.verification.sourced import ProvenanceGap, ProvenanceGapReason

ROW = m.GrowthRow(
    sheet_row=7,
    eck="ECK0614",
    gene="dcuC",
    jw_id="JW0613",
    b_number="b0621",
    readings={"LB_22hr": 0.773, "MOPS_24hr": 0.407, "MOPS_48hr": 0.473},
    non_numeric={},
)


def test_reading_keeps_numbers_and_numeric_text_and_names_the_rest() -> None:
    assert m._reading(0.848) == 0.848
    assert m._reading("0.623") == 0.623
    assert m._reading("N.A.") is None
    assert m._reading("NT") == "NT"


def test_keyable_reads_the_released_b_number_only() -> None:
    assert m.keyable(ROW)
    assert not m.keyable(ROW.model_copy(update={"b_number": "none"}))


def test_genotype_stores_the_released_identity_without_a_crosswalk() -> None:
    [perturbation] = m.genotype(ROW).perturbations
    assert isinstance(perturbation, DeletionStrainPerturbation)
    assert perturbation.systematic_gene_name == "b0621"
    assert perturbation.gene_namespace == "ecoli_k12_mg1655_bnumber"
    assert perturbation.perturbed_gene_name == "dcuC"
    assert perturbation.identifier_mapping is None
    assert perturbation.construction is not None
    assert perturbation.construction.strain_accession == "JW0613"
    assert perturbation.collection == "Keio collection"


def test_phenotype_is_an_absolute_optical_density_with_its_gaps() -> None:
    phenotype = m.phenotype(0.773)
    assert phenotype.measurement_type is MeasurementType.optical_density
    assert phenotype.environment_response == 0.773
    assert phenotype.gapped_fields() == {
        "environment_response_uncertainty",
        "environment_response_se",
        "n_samples",
        "sample_unit",
    }


def test_reference_reading_is_a_typed_gap() -> None:
    reference = m.reference_phenotype()
    assert reference.environment_response is None
    assert "environment_response" in reference.gapped_fields()


def test_a_numeric_response_of_none_without_a_gap_is_refused() -> None:
    with pytest.raises(ValueError, match="requires a numeric environment_response"):
        EnvironmentResponsePhenotype(measurement_type=MeasurementType.optical_density)
    gap = ProvenanceGap(
        field="environment_response", reason=ProvenanceGapReason.not_reported_by_primary
    )
    gapped = EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.optical_density, provenance_gaps=[gap]
    )
    assert gapped.environment_response is None


@pytest.mark.parametrize(
    ("condition", "hours", "base"),
    [
        ("LB_22hr", 22.0, "LB"),
        ("MOPS_24hr", 24.0, "MOPS_MINIMAL"),
        ("MOPS_48hr", 48.0, "MOPS_MINIMAL"),
    ],
)
def test_environment_states_medium_time_and_temperature(
    condition: m.ConditionName, hours: float, base: str
) -> None:
    environment = m.environment(condition)
    assert environment.duration_hours == hours
    assert environment.media.base_medium == base
    assert environment.temperature is not None
    assert environment.temperature.value == 37.0
    names = [c.compound.name for c in environment.media.components]
    assert "kanamycin" in names
    parent = m.environment(condition, kanamycin=False)
    assert "kanamycin" not in [c.compound.name for c in parent.media.components]


def test_mops_medium_states_the_glucose_and_phosphate_amounts() -> None:
    media = m.mops_medium(kanamycin=False)
    amounts = {
        c.compound.name: (c.concentration.value if c.concentration else None)
        for c in media.components
    }
    assert amounts == {"D-glucose": 0.4, "phosphate": 2.0}


def _data_root() -> str:
    return os.environ.get("DATA_ROOT", "")


@pytest.mark.data
def test_table_3_counts_are_the_pinned_ones() -> None:
    rows = m.read_growth_rows(
        bb.raw_mirror_dir(_data_root()) / bb.mirror_relpath(m.TABLE_S3)
    )
    keyed = [r for r in rows if m.keyable(r)]
    assert len(rows) == 3985
    assert len(keyed) == 3912
    assert sum(len(r.readings) for r in keyed) == m.EXPECTED_RECORDS
    assert Counter(n for r in keyed for n in r.readings) == {
        "LB_22hr": 3912,
        "MOPS_24hr": 3912,
        "MOPS_48hr": 3911,
    }
    assert {r.jw_id: r.non_numeric for r in rows if r.non_numeric} == {
        "JW5920": {"MOPS_48hr": "NT"}
    }
    shared = Counter(r.b_number for r in keyed)
    assert sorted(b for b, n in shared.items() if n > 1) == ["b0621", "b2094", "b2218"]


@pytest.mark.data
def test_every_paper_quote_is_verbatim_in_the_raw_mirror() -> None:
    from torchcell.verification.sourced import audit_sourced_value

    root = Path(_data_root()) / "torchcell-raw"
    assert all(audit_sourced_value(v, root).passed for v in m.SOURCED_VALUES.values())


@pytest.mark.data
def test_dev_store_verifies() -> None:
    root = osp.join(_data_root(), m.DATASET_ROOT_REL)
    if not osp.isdir(osp.join(root, "processed")):
        pytest.skip("growth_yield_baba2006 dev store not built")
    report = m.verify_build(root, _data_root())
    assert report.passed, report.summary()


# --------------------------------------------------------------------------- #
# The hermetic build over a fake Table 3
# --------------------------------------------------------------------------- #
class FakeSheet:
    """The members of ``xlrd.sheet.Sheet`` the loader reads."""

    def __init__(self, rows: list[list[Any]]) -> None:
        """Hold the rows."""
        self.rows = rows
        self.nrows = len(rows)

    def cell_value(self, r: int, c: int) -> Any:
        """One cell; past a short row's end reads as xlrd's empty string."""
        row = self.rows[r]
        return row[c] if c < len(row) else ""


class FakeBook:
    """A one-sheet workbook."""

    def __init__(self, sheet: FakeSheet) -> None:
        """Hold the sheet."""
        self.sheet = sheet

    def sheet_by_index(self, i: int) -> FakeSheet:
        """The only sheet."""
        return self.sheet


#: (eck, gene, jw, b, LB, MOPS 24 h, MOPS 48 h)
TABLE: list[tuple[str, str, str, str, Any, Any, Any]] = [
    ("ECK0001", "thrL", "JW4367", "b0001", 0.848, 0.355, 0.542),
    ("ECK0614", "dcuC", "JW0613", "b0621", 0.773, 0.407, 0.473),
    ("ECK0614", "dcuC", "JW0616", "b0621", 0.545, 0.342, 0.435),
    ("ECK2211", "rcsC", "JW5920", "b2218", "0.623", "0.327", "NT"),
    ("none", "none", "JW0032", "none", 0.829, 0.415, 0.468),
    ("ECK0002", "thrA", "JW0001", "b0002", "N.A.", "N.A.", "N.A."),
]


def table_rows() -> list[list[Any]]:
    """Title, the header cells both loaders check, and one row per entry."""
    header1: list[Any] = [""] * 20
    header3: list[Any] = [""] * 20
    for (r, c), text in {**bb.TABLE_S3_HEADER, **m.GROWTH_HEADER}.items():
        (header1 if r == 1 else header3)[c] = text
    rows: list[list[Any]] = [[bb.TABLE_S3_TITLE], header1, [""] * 20, header3]
    for eck, gene, jw, b, lb, m24, m48 in TABLE:
        rows.append([eck, gene, jw] + [""] * 5 + [b] + [""] * 8 + [lb, m24, m48])
    return rows


@pytest.fixture
def fake_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A data root whose raw mirror holds a fake Table 3, served through xlrd."""
    import xlrd

    data_root = tmp_path / "data_root"
    table = bb.raw_mirror_dir(str(data_root)) / bb.mirror_relpath(m.TABLE_S3)
    table.parent.mkdir(parents=True)
    table.write_bytes(b"xls placeholder")
    book = FakeBook(FakeSheet(table_rows()))
    monkeypatch.setattr(xlrd, "open_workbook", lambda path: book)
    monkeypatch.setattr(m, "EXPECTED_ROWS_WITH_READING", 5)
    monkeypatch.setattr(m, "EXPECTED_NO_IDENTIFIER_ROWS", 1)
    monkeypatch.setattr(m, "EXPECTED_RECORDS", 11)
    monkeypatch.setattr(m, "EXPECTED_UNPERTURBED", 7)
    monkeypatch.setattr(m, "TABLE_S3_SHA256", file_sha256(table))
    monkeypatch.setattr(
        bb, "manifest_sha256", lambda manifest, relpath: file_sha256(table)
    )
    monkeypatch.setattr(bb, "load_manifest", lambda data_root: None)
    monkeypatch.setattr(m, "assembly_reference", lambda strain: REFERENCE)
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    return data_root


REFERENCE = ReferenceGenome(
    species="Escherichia coli",
    strain="BW25113",
    assembly_set="ecoli_K12_BW25113_ASM75055v1",
    assembly_accession="GCA_000750555.1",
)


def test_read_growth_rows_keeps_every_released_cell(fake_table: Path) -> None:
    rows = m.read_growth_rows(
        bb.raw_mirror_dir(str(fake_table)) / bb.mirror_relpath(m.TABLE_S3)
    )
    assert [r.jw_id for r in rows] == ["JW4367", "JW0613", "JW0616", "JW5920", "JW0032"]
    rcsc = rows[3]
    assert rcsc.readings == {"LB_22hr": 0.623, "MOPS_24hr": 0.327}
    assert rcsc.non_numeric == {"MOPS_48hr": "NT"}


def test_read_growth_rows_refuses_a_moved_header(
    fake_table: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import xlrd

    rows = table_rows()
    rows[3][18] = "MOPS_24h"
    monkeypatch.setattr(xlrd, "open_workbook", lambda path: FakeBook(FakeSheet(rows)))
    with pytest.raises(bb.TableFormatError, match=r"\(3, 18\)"):
        m.read_growth_rows("t3.xls")


def test_hermetic_build_and_verify(
    fake_table: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from torchcell.verification import runners
    from torchcell.verification.report import Level, LevelResult

    root = fake_table / m.DATASET_ROOT_REL
    dataset = m.GrowthYieldBaba2006Dataset(root=str(root))
    assert len(dataset) == 11
    ledger = json.loads((root / "preprocess" / "not_stored.json").read_text())
    assert ledger["stored_per_condition"] == {
        "LB_22hr": 4,
        "MOPS_24hr": 4,
        "MOPS_48hr": 3,
    }
    assert ledger["not_stored"][0]["jw_ids"] == ["JW0032"]

    monkeypatch.setattr(
        runners, "_bacterial_gene_set", lambda assembly, base: {"b0001", "b0621"}
    )
    monkeypatch.setattr(
        m,
        "audit_sourced_value",
        lambda value, root: LevelResult(
            level=Level.L3, name="provenance_audit", passed=True, message="stub"
        ),
    )
    report = m.verify_build(str(root), str(fake_table))
    assert report.passed, report.summary()
    l4 = next(r for r in report.results if r.level == Level.L4)
    assert l4.details["missing"] == ["b2218"]
    assert m.main(["verify"]) == 0


def test_build_refuses_a_count_that_is_not_pinned(
    fake_table: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(m, "EXPECTED_ROWS_WITH_READING", 6)
    with pytest.raises(RuntimeError, match="5 rows carry a reading"):
        m.GrowthYieldBaba2006Dataset(root=str(fake_table / m.DATASET_ROOT_REL))
