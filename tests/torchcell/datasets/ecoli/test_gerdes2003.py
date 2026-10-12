# tests/torchcell/datasets/ecoli/test_gerdes2003.py
# [[tests.torchcell.datasets.ecoli.test_gerdes2003]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/ecoli/test_gerdes2003.py
"""The Gerdes 2003 genetic-footprinting loader (``torchcell.datasets.ecoli.gerdes2003``).

Synthetic tests (run everywhere) serve Table S1 through a fake ``xlrd.open_workbook`` (no
``.xls`` writer is installed) and use the real ``EcoliK12MG1655Genome`` over a synthetic
MG1655 assembly (network refused). The synthetic Table S1:

    row  gene   call  Blattner ID  outcome
    1    thrL   E     b0001        -> b0001, True
    2    thrA   N     b0002        -> b0002, False (insertions "101, 130")
    3    thrW   N*    b0003        -> b0003, False (one insertion, read as 69.0)
    4    yaaP   X     b0004        -> dropped: not covered
    5    proB   N     b0099        -> b0005, False, a locus_tag_synonym of b0005
    6    proC   E     b0006        -> b0006, True (it IS the locus of its collision group)
    7    frag   N     b0098        -> dropped: a synonym of b0006, which row 6 holds
    8    insZ   ?     b0007        -> dropped: ambiguous
    9    hokC   N     (empty)      -> dropped: no Blattner ID
    10   insA   E     b0010        -> dropped: b0010 on two rows
    11   insA2  N     b0010        -> dropped: b0010 on two rows
    12   dead   N     b0500        -> dropped: retired

Eight of nine distinct single Blattner ids resolve (0.889 < 0.98), so the success path
stores every row under its released identifier (no genome is read in the build); the
build. Five records.

Data-gated tests (``@pytest.mark.data``) read the real raw mirror and the built dev-tree
LMDB under ``$DATA_ROOT`` (never building it).
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import os.path as osp
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import xlrd

import torchcell.datasets.ecoli.gerdes2003 as m
from tests.torchcell.sequence.genome._bacterial_fixtures import (
    MG1655_GAF,
    SyntheticLocus,
    forbid_network,
    serve_tier,
    write_assembly,
)
from torchcell.data import ManifestPinMismatchError
from torchcell.datamodels.schema import (
    AssemblyReferenceGenome,
    BacterialReleasedTransposonPerturbation,
    ComponentDefinition,
    ConcentrationUnit,
    EssentialityEvidenceClass,
    EssentialityEvidenceType,
    EssentialityStatementRole,
    FootprintingCall,
    Genotype,
    MediaComponentRole,
    TransposonInsertionPerturbation,
)
from torchcell.literature.manifest import ROLE_RAW_DATA, ArtifactRecord, Manifest
from torchcell.sequence.genome.ecoli.k12 import MG1655_ASSEMBLY, EcoliK12MG1655Genome
from torchcell.verification.report import Provenance, VerificationReport
from torchcell.verification.sourced import (
    ProvenanceGapReason,
    SourcedValue,
    audit_sourced_value,
)


class FakeSheet:
    """The four members of ``xlrd.sheet.Sheet`` the loader reads."""

    def __init__(self, rows: list[list[Any]]) -> None:
        """Hold the rows; ``nrows`` / ``ncols`` as xlrd reports them."""
        self.rows = rows
        self.nrows = len(rows)
        self.ncols = max(len(r) for r in rows)

    def cell_value(self, r: int, c: int) -> Any:
        """One cell; a short row reads as xlrd's empty string past its end."""
        row = self.rows[r]
        return row[c] if c < len(row) else ""


class FakeBook:
    """A workbook with named sheets."""

    def __init__(self, sheet: FakeSheet, names: list[str]) -> None:
        """Hold the sheets and their names."""
        self.sheet = sheet
        self.names = names

    def sheet_by_index(self, i: int) -> FakeSheet:
        """The ``i``-th sheet."""
        assert i == 0
        return self.sheet

    def sheet_names(self) -> list[str]:
        """The sheet names."""
        return self.names


BOOKS: dict[str, FakeBook] = {}


@pytest.fixture(autouse=True)
def fake_xlrd(monkeypatch: pytest.MonkeyPatch) -> None:
    """``xlrd.open_workbook`` serves the registered fake books by resolved path."""
    BOOKS.clear()

    def open_workbook(path: str) -> FakeBook:
        return BOOKS[str(Path(path).resolve())]

    monkeypatch.setattr(xlrd, "open_workbook", open_workbook)


#: (ergo, gene, call, length, insertions, assertion error, Blattner ID)
ROWS: list[tuple[str, str, str, float, Any, str, str]] = [
    ("REC00001", "thrL", "E", 21.0, "-", "< 0.1", "b0001"),
    ("REC00002", "thrA", "N", 820.0, "101, 130", "", "b0002"),
    ("REC00003", "thrW", "N*", 310.0, 69.0, "", "b0003"),
    ("REC00004", "yaaP", "X", 98.0, "-", "", "b0004"),
    ("REC00005", "proB", "N", 258.0, "23", "", "b0099"),
    ("REC00006", "proC", "E", 269.0, "-", "< 0.001", "b0006"),
    ("REC00007", "frag", "N", 50.0, "12", "", "b0098"),
    ("REC00008", "insZ", "?", 72.0, "-", "", "b0007"),
    ("REC00009", "hokC ", "N", 69.0, "40", "", ""),
    ("REC00010", "insA", "E", 91.0, "-", "< 0.2", "b0010"),
    ("REC00011", "insA2", "N", 91.0, "30", "", "b0010"),
    ("REC00012", "dead", "N", 100.0, "50", "", "b0500"),
]
SYNTHETIC_CALLS = {"?": 1, "E": 3, "N": 6, "N*": 1, "X": 1}


def table_rows(
    rows: list[tuple[str, str, str, float, Any, str, str]] = ROWS,
) -> list[list[Any]]:
    """The group header, the column header, and one data row per entry."""
    out: list[list[Any]] = [
        [""] * 12,
        ["ORF start"] + [""] * 11,
        list(m.TABLE_S1_HEADER),
    ]
    for i, (ergo, gene, call, length, ins, err, b) in enumerate(rows):
        out.append(
            [
                float(100 * i),
                ergo,
                gene,
                call,
                length,
                ins,
                err,
                "P0",
                "role",
                b,
                "UNC",
                0.5,
            ]
        )
    return out


def register(path: Path, rows: list[list[Any]], names: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"xls placeholder")
    BOOKS[str(path.resolve())] = FakeBook(FakeSheet(rows), names or [m.TABLE_S1_SHEET])


def _locus(
    tag: str, start: int, symbol: str, *syn: str, pseudo: bool = False
) -> SyntheticLocus:
    return SyntheticLocus(
        tag=tag,
        parts=((start, start + 8),),
        strand="+",
        symbol=symbol,
        synonyms=syn,
        pseudo=pseudo,
        product=None if pseudo else f"protein {symbol}",
        protein_id=None if pseudo else f"AAC{start:05d}.1",
        protein=None if pseudo else "MK",
    )


MG_LOCI = [
    _locus("b0001", 1, "thrL", "ECK0001"),
    _locus("b0002", 12, "thrA", "ECK0002"),
    _locus("b0003", 24, "thrW", "ECK0003"),
    _locus("b0004", 36, "yaaP", "ECK0004", pseudo=True),
    _locus("b0005", 48, "proB", "ECK0005", "b0099"),
    _locus("b0006", 60, "proC", "ECK0006", "b0098"),
    _locus("b0007", 72, "insZ", "ECK0007", pseudo=True),
]
UNIVERSE = {locus.tag for locus in MG_LOCI}
REFERENCE = AssemblyReferenceGenome(
    species="Escherichia coli",
    strain="MG1655",
    assembly_set="ecoli_K12_MG1655_ASM584v2",
    assembly_accession="GCA_000005845.2",
)


@pytest.fixture
def mg1655(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EcoliK12MG1655Genome:
    """The real MG1655 class over the synthetic assembly; network refused."""
    files = write_assembly(tmp_path / "tier", MG1655_ASSEMBLY, MG_LOCI, MG1655_GAF)
    forbid_network(monkeypatch)
    serve_tier(monkeypatch, files)
    return EcoliK12MG1655Genome(genome_root=str(tmp_path / "mg1655"), overwrite=True)


@pytest.fixture
def synthetic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A dataset root whose raw/ holds the fake Table S1; pins checked by presence."""
    root = tmp_path / m.DATASET_ROOT_REL
    register(root / "raw" / m.TABLE_S1, table_rows())

    def presence(raw_dir: str, expected: Mapping[str, str]) -> None:
        missing = [f for f in expected if not osp.exists(osp.join(raw_dir, f))]
        if missing:
            raise FileNotFoundError(f"pinned raw files absent: {missing}")

    monkeypatch.setattr(m, "verify_raw_files", presence)
    monkeypatch.setattr(m, "TABLE_CALLS", SYNTHETIC_CALLS)
    monkeypatch.setattr(m, "assembly_reference", lambda strain, **_: REFERENCE)
    return root


@pytest.fixture
def built(
    synthetic: Path, monkeypatch: pytest.MonkeyPatch
) -> m.GeneEssentialityGerdes2003Dataset:
    monkeypatch.setattr(m, "EXPECTED_RECORDS", 12)
    monkeypatch.setattr(m, "EXPECTED_B_NUMBER_RECORDS", 9)
    monkeypatch.setattr(m, "EXPECTED_ERGO_RECORDS", 3)
    return m.GeneEssentialityGerdes2003Dataset(root=str(synthetic))


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def test_read_table_s1_reads_every_row(tmp_path: Path) -> None:
    register(tmp_path / "t.xls", table_rows())
    rows = m.read_table_s1(tmp_path / "t.xls")
    assert [(r.row, r.call, r.insertions_aa, r.blattner_id) for r in rows][:3] == [
        (1, "E", (), "b0001"),
        (2, "N", (101, 130), "b0002"),
        (3, "N*", (69,), "b0003"),
    ]
    assert rows[8].gene == "hokC" and rows[0].assertion_error == "< 0.1"
    assert m.call_histogram(rows) == SYNTHETIC_CALLS
    assert [r.is_essential for r in rows[:4]] == [True, False, False, None]
    assert rows[7].is_essential is None


def _mutated(row: int, col: int, value: Any) -> list[list[Any]]:
    rows = table_rows()
    rows[row][col] = value
    return rows


@pytest.mark.parametrize(
    ("rows", "names", "message"),
    [
        (table_rows(), ["Sheet1"], "sheets"),
        (_mutated(2, 8, "functional role"), None, "header"),
        (_mutated(3, 3, "e"), None, "call 'e'"),
        (_mutated(3, 4, 20.5), None, "length"),
        (_mutated(3, 9, " b0001"), None, "Blattner ID"),
        (_mutated(3, 5, 2.5), None, "insertion 2.5"),
        (_mutated(3, 5, "1, x"), None, "insertions"),
        (_mutated(3, 5, None), None, "insertions"),
    ],
)
def test_read_table_s1_refuses_an_unexpected_table(
    tmp_path: Path, rows: list[list[Any]], names: list[str] | None, message: str
) -> None:
    register(tmp_path / "t.xls", rows, names)
    with pytest.raises(m.TableFormatError, match=message):
        m.read_table_s1(tmp_path / "t.xls")


# --------------------------------------------------------------------------- #
# Identifiers as released, and the ledger
# --------------------------------------------------------------------------- #
def test_identify_rows_keys_unique_b_numbers_and_the_rest_by_ergo_id(
    tmp_path: Path,
) -> None:
    register(tmp_path / "t.xls", table_rows())
    rows = m.read_table_s1(tmp_path / "t.xls")
    identities = m.identify_rows(rows)
    assert [(i.gerdes.gene, i.kind, i.identifier) for i in identities] == [
        ("thrL", "MG1655 b-number", "b0001"),
        ("thrA", "MG1655 b-number", "b0002"),
        ("thrW", "MG1655 b-number", "b0003"),
        ("yaaP", "MG1655 b-number", "b0004"),
        ("proB", "MG1655 b-number", "b0099"),
        ("proC", "MG1655 b-number", "b0006"),
        ("frag", "MG1655 b-number", "b0098"),
        ("insZ", "MG1655 b-number", "b0007"),
        ("hokC", "ERGO id", "REC00009"),
        ("insA", "ERGO id", "REC00010"),
        ("insA2", "ERGO id", "REC00011"),
        ("dead", "MG1655 b-number", "b0500"),
    ]
    log_ = m.build_drop_log("gerdes", identities)
    assert (
        log_.source_rows,
        log_.kept_records,
        log_.b_number_records,
        log_.ergo_records,
        log_.dropped_records,
        log_.rules,
    ) == (12, 12, 9, 3, 0, [])
    assert log_.kept_calls == SYNTHETIC_CALLS
    assert any("REC00010 insA (E): Blattner ID 'b0010'" in n for n in log_.notes)


def test_identify_rows_refuses_two_rows_on_one_ergo_id(tmp_path: Path) -> None:
    register(tmp_path / "t.xls", table_rows([ROWS[8], ROWS[8]]))
    rows = m.read_table_s1(tmp_path / "t.xls")
    with pytest.raises(m.DuplicateRecordError, match="REC00009"):
        m.identify_rows(rows)


def test_a_malformed_b_number_keys_by_ergo_id(tmp_path: Path) -> None:
    register(
        tmp_path / "t.xls",
        table_rows([("REC00018", "nhaA", "N", 100.0, "12", "", "b0018.1")]),
    )
    (identity,) = m.identify_rows(m.read_table_s1(tmp_path / "t.xls"))
    assert (identity.kind, identity.identifier) == ("ERGO id", "REC00018")


# --------------------------------------------------------------------------- #
# Medium, environment, records
# --------------------------------------------------------------------------- #
def test_fermentor_medium_states_every_component() -> None:
    selection = m.fermentor_medium(kanamycin=True)
    parent = m.fermentor_medium(kanamycin=False)
    assert (selection.state, selection.base_medium, selection.is_synthetic) == (
        "liquid",
        "LB",
        False,
    )
    assert (
        len(selection.components)
        == 2 + len(m.COMPONENTS) + 1
        == len(parent.components) + 1
    )
    tryptone, extract = selection.components[:2]
    assert (tryptone.compound.name, tryptone.definition) == (
        "tryptone",
        ComponentDefinition.intrinsically_undefined,
    )
    assert extract.concentration is not None and extract.concentration.value == 5.0
    kan = selection.components[-1]
    assert kan.role is MediaComponentRole.selection_agent
    assert kan.concentration is not None
    assert (kan.concentration.value, kan.concentration.unit) == (
        10.0,
        ConcentrationUnit.ug_per_ml,
    )
    vitamins = [c for c in selection.components if c.role is MediaComponentRole.vitamin]
    assert len(vitamins) == 9
    env = m.environment()
    assert (env.duration_hours, env.duration_generations, env.aerobicity) == (
        12.0,
        23.0,
        "aerobic",
    )
    assert env.temperature is not None and env.temperature.value == 37.0
    assert m.environment(kanamycin=False).media == parent


def test_build_experiment_and_reference(tmp_path: Path) -> None:
    register(tmp_path / "t.xls", table_rows([ROWS[4], ROWS[5], ROWS[3], ROWS[8]]))
    proB, proC, yaaP, hokC = m.identify_rows(m.read_table_s1(tmp_path / "t.xls"))
    env = m.environment()
    a = m.build_experiment("gerdes", proB, env)
    b = m.build_experiment("gerdes", proC, env)
    assert isinstance(a.genotype, Genotype) and isinstance(b.genotype, Genotype)
    (pa,), (pb,) = a.genotype.perturbations, b.genotype.perturbations
    assert isinstance(pa, TransposonInsertionPerturbation)
    assert isinstance(pb, TransposonInsertionPerturbation)
    # the released b0099 is stored verbatim; the annotation's synonym route is not read
    assert (pa.systematic_gene_name, pa.identifier_mapping) == ("b0099", None)
    assert (pb.identifier_mapping, pb.transposon, pb.insertion_position) == (
        None,
        "EZ::TN<KAN-2>",
        None,
    )
    assert (a.phenotype.is_essential, b.phenotype.is_essential) == (False, True)
    assert a.phenotype.provenance_gaps == []
    # #924: the call letter, the insertions and the assertion error as released
    ea, eb = a.phenotype.evidence, b.phenotype.evidence
    assert (ea.footprinting_call, ea.footprint_insertions_aa, ea.assertion_error) == (
        FootprintingCall.non_essential,
        [23],
        None,
    )
    assert (eb.footprinting_call, eb.footprint_insertions_aa) == (
        FootprintingCall.essential,
        [],
    )
    assert (eb.assertion_error, eb.assertion_error_is_upper_bound) == (0.001, True)
    assert eb.evidence_type is EssentialityEvidenceType.genetic_footprinting
    assert eb.evidence_class is EssentialityEvidenceClass.insertion_depletion
    assert eb.scoring_rule is m.SOURCED_VALUES["scoring_rule"]
    assert eb.call_definition is m.SOURCED_VALUES["call_definition"]
    assert eb.statements_of(EssentialityStatementRole.caveat) == [
        m.SOURCED_VALUES["calls_uncorrected"]
    ]
    assert eb.statements_of(EssentialityStatementRole.statistic_definition) == [
        m.SOURCED_VALUES["assertion_error"],
        m.SOURCED_VALUES["insertion_coordinates"],
    ]

    x = m.build_experiment("gerdes", yaaP, env)
    assert x.phenotype.is_essential is None
    assert x.phenotype.evidence.footprinting_call is FootprintingCall.not_covered
    assert [g.field for g in x.phenotype.provenance_gaps] == ["is_essential"]
    assert "not covered" in str(x.phenotype.provenance_gaps[0].note)

    h = m.build_experiment("gerdes", hokC, env)
    assert isinstance(h.genotype, Genotype)
    (ph,) = h.genotype.perturbations
    assert isinstance(ph, BacterialReleasedTransposonPerturbation)
    assert (
        ph.systematic_gene_name,
        ph.perturbed_gene_name,
        ph.strain_label,
        ph.identifier_kind,
        ph.transposon,
    ) == ("REC00009", "hokC", "REC00009", "ERGO id", "EZ::TN<KAN-2>")
    reference = m.build_reference("gerdes", REFERENCE)
    assert reference.environment_reference == m.environment(kanamycin=False)
    assert reference.phenotype_reference.is_essential is False
    ref = reference.phenotype_reference.evidence
    assert ref.footprinting_call is None and ref.statements == eb.statements


@pytest.mark.parametrize(
    ("text", "value", "bound"),
    [
        ("< 0.05", 0.05, True),
        ("< 0.0001", 0.0001, True),
        ("0.306647", 0.306647, False),
        ("", None, None),
        ("n.d.", None, None),
    ],
)
def test_read_assertion_error_reads_the_four_released_forms(
    text: str, value: float | None, bound: bool | None
) -> None:
    cell = m.read_assertion_error(text, "row 1")
    assert (cell.value, cell.is_upper_bound, cell.released) == (value, bound, text)


@pytest.mark.parametrize("text", ["<0.05", "~0.1", "1.5", "< 2", "abc"])
def test_read_assertion_error_refuses_any_other_form(text: str) -> None:
    with pytest.raises(m.TableFormatError, match="assertion error"):
        m.read_assertion_error(text, "row 1")


def test_an_e_call_without_an_assertion_error_carries_a_typed_gap() -> None:
    def row(call: str, error: str) -> m.GerdesRow:
        return m.GerdesRow(
            row=1,
            ergo_id="REC1",
            gene="g",
            call=call,  # type: ignore[arg-type]
            length_aa=100,
            insertions_aa=(),
            assertion_error=error,
            blattner_id="b0001",
        )

    for error, note in (
        ("n.d.", "Table S1 reads 'n.d.'"),
        ("", "cell is empty in Table S1"),
    ):
        evidence = m.evidence(row("E", error))
        (gap,) = evidence.provenance_gaps
        assert (gap.field, gap.reason) == (
            "assertion_error",
            ProvenanceGapReason.not_reported_by_primary,
        )
        assert gap.note is not None and note in gap.note
    # an N call has no assertion error to gap; the one N row that carries one keeps it
    assert m.evidence(row("N", "")).provenance_gaps == []
    released = m.evidence(row("N", "< 0.05"))
    assert (released.assertion_error, released.assertion_error_is_upper_bound) == (
        0.05,
        True,
    )


def test_perturbation_field_gaps_name_real_leaf_fields() -> None:
    fields = set(TransposonInsertionPerturbation.model_fields)
    assert [g.field for g in m.PERTURBATION_FIELD_GAPS] == [
        "insertion_position",
        "insertion_strand",
    ]
    assert {g.field for g in m.PERTURBATION_FIELD_GAPS} <= fields


# --------------------------------------------------------------------------- #
# The hermetic build and its verification
# --------------------------------------------------------------------------- #
def _locus_of(record: Mapping[str, Any]) -> str:
    return str(
        record["experiment"]["genotype"]["perturbations"][0]["systematic_gene_name"]
    )


def test_build_twelve_records_and_the_ledgers(
    built: m.GeneEssentialityGerdes2003Dataset,
) -> None:
    assert len(built) == 12
    records = [built[i] for i in range(12)]
    assert [
        (_locus_of(r), r["experiment"]["phenotype"]["is_essential"]) for r in records
    ] == [
        ("b0001", True),
        ("b0002", False),
        ("b0003", False),
        ("b0004", None),
        ("b0099", False),
        ("b0006", True),
        ("b0098", False),
        ("b0007", None),
        ("REC00009", False),
        ("REC00010", True),
        ("REC00011", False),
        ("b0500", False),
    ]
    assert [
        r["experiment"]["genotype"]["perturbations"][0]["perturbation_type"]
        for r in records[7:11]
    ] == [
        "transposon_insertion",
        "bacterial_released_transposon",
        "bacterial_released_transposon",
        "bacterial_released_transposon",
    ]
    ambiguous = records[7]["experiment"]["phenotype"]
    assert ambiguous["evidence"]["footprinting_call"] == "?"
    assert [g["field"] for g in ambiguous["provenance_gaps"]] == ["is_essential"]
    preprocess = Path(built.preprocess_dir)
    drops = json.loads((preprocess / "dropped_records.json").read_text())
    assert (
        drops["kept_records"],
        drops["b_number_records"],
        drops["ergo_records"],
        drops["dropped_records"],
        drops["rules"],
    ) == (12, 9, 3, 0, [])
    assert drops["kept_calls"] == SYNTHETIC_CALLS
    gaps = json.loads((preprocess / "perturbation_field_gaps.json").read_text())
    assert [g["field"] for g in gaps] == ["insertion_position", "insertion_strand"]
    assert not (preprocess / "identifier_reconciliation.json").exists()
    import pandas as pd

    calls = pd.read_csv(preprocess / "calls.csv", keep_default_na=False)
    assert calls.loc[1, "insertions_aa"] == "101 130"
    assert calls["record"].tolist() == list(range(12))
    assert calls.loc[9, "identifier_kind"] == "ERGO id"
    assert calls.loc[9, "blattner_id"] == "b0010"
    assert calls.loc[3, "call"] == "X" and calls.loc[3, "is_essential"] == ""


def test_build_refuses_other_counts(
    synthetic: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(m, "TABLE_CALLS", {**SYNTHETIC_CALLS, "E": 4})
    with pytest.raises(m.CallCountError, match="pinned"):
        m.GeneEssentialityGerdes2003Dataset(root=str(synthetic))
    monkeypatch.setattr(m, "TABLE_CALLS", SYNTHETIC_CALLS)
    monkeypatch.setattr(m, "EXPECTED_RECORDS", 12)
    monkeypatch.setattr(m, "EXPECTED_B_NUMBER_RECORDS", 10)
    monkeypatch.setattr(m, "EXPECTED_ERGO_RECORDS", 2)
    with pytest.raises(m.CallCountError, match="the module declares"):
        m.GeneEssentialityGerdes2003Dataset(root=str(synthetic))


def test_unused_hooks_and_classes(built: m.GeneEssentialityGerdes2003Dataset) -> None:
    import pandas as pd

    frame = pd.DataFrame({"a": [1]})
    assert built.preprocess_raw(frame) is frame
    with pytest.raises(NotImplementedError):
        built.create_experiment()
    assert built.experiment_class.__name__ == "BacterialGeneEssentialityExperiment"
    assert (
        built.reference_class.__name__ == "BacterialGeneEssentialityExperimentReference"
    )
    assert built.raw_file_names == [m.TABLE_S1]


def _report(
    built: m.GeneEssentialityGerdes2003Dataset,
    mg1655: EcoliK12MG1655Genome,
    records: list[dict[str, Any]] | None = None,
) -> VerificationReport:
    from torchcell.verification.runners import load_records

    rows = m.read_table_s1(Path(built.raw_dir) / m.TABLE_S1)
    return m.verify_records(
        records if records is not None else load_records(built.root),
        rows=rows,
        universe=UNIVERSE,
        expected_count=12,
    )


def test_verify_records_passes_and_fails_on_tampering(
    built: m.GeneEssentialityGerdes2003Dataset, mg1655: EcoliK12MG1655Genome
) -> None:
    from torchcell.verification.runners import load_records

    report = _report(built, mg1655)
    assert report.passed, report.summary()
    assert [r.name for r in report.results][:6] == [
        "structural",
        "count",
        "one_record_per_released_identifier",
        "calls_match_table_s1",
        "table_s1_call_histogram_is_pinned",
        "reference_is_the_viable_parent_without_kanamycin",
    ]
    finding = report.results[-1]
    assert finding.name == "gene_containment_mg1655_locus_tags" and finding.passed
    # b0098, b0099 and b0500 are no gene row of the synthetic annotation
    assert finding.details["missing_examples"] == ["b0098", "b0099", "b0500"]
    assert finding.details["n_ergo_records"] == 3
    records = load_records(built.root)
    records[0]["experiment"]["phenotype"]["is_essential"] = False
    records[1]["experiment"]["genotype"]["perturbations"][0]["systematic_gene_name"] = (
        "b0004"
    )
    records[2]["reference"]["phenotype_reference"]["is_essential"] = True
    records[3]["experiment"]["phenotype"]["evidence"]["footprinting_call"] = "N"
    records[9]["experiment"]["genotype"]["perturbations"][0]["systematic_gene_name"] = (
        "b0010"
    )
    records.append(records[4])
    failed = {
        r.name: r for r in _report(built, mg1655, records).results if not r.passed
    }
    assert set(failed) >= {
        "count",
        "one_record_per_released_identifier",
        "calls_match_table_s1",
        "reference_is_the_viable_parent_without_kanamycin",
    }
    # 1 (flipped call) + 3 (thrA's record under yaaP's b0004: call, gap and evidence)
    # + 1 (the X letter replaced on the evidence) + 1 (the released-leaf record under a
    # b-number no ERGO row has)
    assert failed["calls_match_table_s1"].details["n_problems"] == 6


def test_verify_records_fails_on_tampered_evidence(
    built: m.GeneEssentialityGerdes2003Dataset, mg1655: EcoliK12MG1655Genome
) -> None:
    """#924: a record whose insertions are not its row's fails L2, and a reference
    whose statements are not the sourced ones fails L3.
    """
    from torchcell.verification.runners import load_records

    records = load_records(built.root)
    records[0]["experiment"]["phenotype"]["evidence"]["footprint_insertions_aa"] = [999]
    # the reference is interned (one object shared by every record): copy it first
    records[1] = copy.deepcopy(records[1])
    records[1]["reference"]["phenotype_reference"]["evidence"]["statements"] = []
    failed = {
        r.name: r for r in _report(built, mg1655, records).results if not r.passed
    }
    assert failed["calls_match_table_s1"].details["n_problems"] == 1
    assert "evidence" in failed["calls_match_table_s1"].details["problems"][0]
    assert failed["evidence_is_genetic_footprinting"].details["n_wrong"] == 1


def test_run_verification_and_cli(
    built: m.GeneEssentialityGerdes2003Dataset,
    mg1655: EcoliK12MG1655Genome,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import torchcell.verification.runners as runners

    text = tmp_path / "torchcell-raw" / m.CITATION_KEY / m.GENETIC_TEXT_REL
    text.parent.mkdir(parents=True)
    text.write_text("Escherichia coli strain MG1655 was used.")
    value = SourcedValue(
        value="MG1655",
        quote="strain MG1655",
        provenance=Provenance(
            source_uri=m.GENETIC_TEXT_REL,
            citation_key=m.CITATION_KEY,
            sha256=hashlib.sha256(text.read_bytes()).hexdigest(),
        ),
    )
    monkeypatch.setattr(m, "SOURCED_VALUES", {"strain": value})
    monkeypatch.setattr(runners, "_gene_set_for_reference", lambda ref, base: UNIVERSE)
    report = m.run_verification(str(tmp_path))
    assert report.passed, report.summary()
    assert report.results[-1].name == "provenance_audit"
    written = json.loads(
        (Path(built.root) / "preprocess" / "verification_report.json").read_text()
    )
    assert written["dataset_name"] == "gene_essentiality_gerdes2003"

    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    assert m.main(["build"]) == 0
    assert capsys.readouterr().out.strip().endswith("len = 12")
    assert m.main(["verify"]) == 0
    # an empty gene universe changes the L4 finding's counts and nothing else: it is
    # reported under the verdict, never gating it (owner, 2026-10-12)
    capsys.readouterr()
    monkeypatch.setattr(runners, "_gene_set_for_reference", lambda ref, base: set())
    assert m.main(["verify"]) == 0
    out = capsys.readouterr().out
    assert "gene_essentiality_gerdes2003: PASS" in out
    assert "L4 gene_containment_mg1655_locus_tags: 0 of 9 released b-numbers" in out


def test_download_links_the_mirror_and_checks_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "root"
    path = data_root / m.RAW_DIR_REL / m.TABLE_S1_REL
    path.parent.mkdir(parents=True)
    path.write_bytes(b"table")
    pin = hashlib.sha256(b"table").hexdigest()
    record = ArtifactRecord(
        path=m.TABLE_S1_REL, role=ROLE_RAW_DATA, bytes=5, sha256=pin, source="s"
    )
    manifest = Manifest(
        citation_key=m.CITATION_KEY,
        files=[record],
        created_at=datetime.now(UTC).isoformat(),
    )
    (data_root / m.RAW_DIR_REL / "manifest.json").write_text(manifest.model_dump_json())
    monkeypatch.setattr(m, "TABLE_S1_SHA256", pin)
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    dataset = object.__new__(m.GeneEssentialityGerdes2003Dataset)
    monkeypatch.setattr(
        m.GeneEssentialityGerdes2003Dataset,
        "raw_dir",
        str(tmp_path / "raw"),
        raising=False,
    )
    dataset.download()
    assert os.listdir(tmp_path / "raw") == [m.TABLE_S1]
    with pytest.raises(KeyError, match="not in the raw-mirror manifest"):
        m.manifest_sha256(manifest, "data/absent.xls")
    monkeypatch.setattr(m, "TABLE_S1_SHA256", "0" * 64)
    with pytest.raises(ManifestPinMismatchError):
        dataset.download()


# --------------------------------------------------------------------------- #
# Real data (``--data``): the mirror and the built dev-tree LMDB, never rebuilt
# --------------------------------------------------------------------------- #
#: Read off Table S1 by hand: thrA N; mog E; dnaK E despite inserts at codons 9 and 581
#: (the gene's ends); accD N* (the page's own example); ylbE released as b0519, a
#: synonym of b4572 on the pinned MG1655 annotation.
HAND_CHECKED: dict[str, bool] = {
    "b0002": False,
    "b0009": True,
    "b0014": True,
    "b2316": False,
    "b0519": False,
}


@pytest.mark.data
def test_real_mirror_matches_the_module_pins() -> None:
    from torchcell.data.experiment_dataset import file_sha256

    root = os.environ["DATA_ROOT"]
    manifest = m.load_manifest(root)
    for relpath, sha256 in (
        (m.TABLE_S1_REL, m.TABLE_S1_SHA256),
        (m.TABLE_PAGE_REL, m.TABLE_PAGE_SHA256),
        (m.GENETIC_TEXT_REL, m.GENETIC_TEXT_SHA256),
        (m.ASSERTION_TEXT_REL, m.ASSERTION_TEXT_SHA256),
        (m.PUBMED_XML_REL, m.PUBMED_XML_SHA256),
    ):
        assert m.manifest_sha256(manifest, relpath) == sha256
        assert file_sha256(m.raw_mirror_dir(root) / relpath) == sha256


@pytest.mark.data
@pytest.mark.parametrize("key", sorted(m.SOURCED_VALUES))
def test_real_sourced_values_are_verbatim(key: str) -> None:
    result = audit_sourced_value(
        m.SOURCED_VALUES[key], Path(os.environ["DATA_ROOT"]) / "torchcell-raw"
    )
    assert result.passed, result.message


@pytest.mark.data
def test_real_build_counts_and_hand_checked_rows() -> None:
    """4,291 records = every Table S1 row: 4,208 under a released b-number, 83 under
    an ERGO id (64 with no Blattner ID, b0018.1, 18 rows of 9 repeated b-numbers).
    """
    from torchcell.verification.runners import load_records

    root = osp.join(os.environ["DATA_ROOT"], m.DATASET_ROOT_REL)
    drops = json.loads(Path(root, "preprocess", "dropped_records.json").read_text())
    assert (
        drops["source_rows"],
        drops["kept_records"],
        drops["b_number_records"],
        drops["ergo_records"],
        drops["dropped_records"],
        drops["rules"],
    ) == (4291, 4291, 4208, 83, 0, [])
    assert drops["kept_calls"] == {"?": 221, "E": 619, "N": 3114, "N*": 6, "X": 331}
    records = load_records(root)
    calls = {
        _locus_of(r): r["experiment"]["phenotype"]["is_essential"] for r in records
    }
    assert {k: calls[k] for k in HAND_CHECKED} == HAND_CHECKED
    assert sum(1 for v in calls.values() if v is None) == 331 + 221
    # ylbE is released as b0519, which the annotation carries as a synonym of b4572:
    # the record stores b0519 verbatim and no mapping
    (ylbe,) = [r for r in records if _locus_of(r) == "b0519"]
    assert ylbe["experiment"]["genotype"]["perturbations"][0]["identifier_mapping"] is (
        None
    )
    assert (
        sum(
            1
            for r in records
            if r["experiment"]["genotype"]["perturbations"][0]["perturbation_type"]
            == "bacterial_released_transposon"
        )
        == 83
    )
