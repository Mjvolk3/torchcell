# tests/torchcell/datasets/ecoli/test_baba2006.py
# [[tests.torchcell.datasets.ecoli.test_baba2006]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/ecoli/test_baba2006.py
"""The Baba 2006 Keio construction-outcome loader (``torchcell.datasets.ecoli.baba2006``).

Synthetic tests (run everywhere) serve the three released ``.xls`` tables through a fake
``xlrd.open_workbook`` (no ``.xls`` writer is installed); the build reads no genome, and
the verifier's L4 finding reads the real ``EcoliK12BW25113Genome`` over a synthetic
BW25113 assembly whose loci carry JW ids as ``gene_synonym`` (network refused). The
synthetic Table 3:

    row  gene   JW      verified/tested  outcome
    1    thrL   JW4367  8/8   obtained       -> record 0, False
    2    thrA   JW0001  0/8   not obtained   -> record 1, True
    3    hokC   JW5002  1/8   not obtained   -> record 2, True (one colony, as secM)
    4    yaaP   JW0004  4/4   obtained       -> record 3, False
    5    proB   JW0006  2/8   obtained       -> record 4, False (shares a locus with JW0007)
    6    proB   JW0007  3/8   obtained       -> record 5, False
    7    insX   JW9999  5/8   obtained       -> record 6, False (no locus carries JW9999)
    8    insL   JW0015  N.A.  not targeted   -> ledgered under ``not_targeted``

Seven records; the L4 finding counts JW9999 unplaced and JW0006/JW0007 on one locus.

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

import torchcell.datasets.ecoli.baba2006 as m
from tests.torchcell.sequence.genome._bacterial_fixtures import (
    SyntheticLocus,
    forbid_network,
    serve_tier,
    write_assembly,
)
from torchcell.data import ManifestPinMismatchError
from torchcell.datamodels.schema import (
    BacterialReleasedStrainPerturbation,
    EssentialityEvidenceClass,
    EssentialityEvidenceType,
    EssentialityStatementRole,
    GeneEssentialityExperiment,
    Genotype,
    MediaComponentRole,
    ReferenceGenome,
)
from torchcell.literature.manifest import ROLE_RAW_DATA, ArtifactRecord, Manifest
from torchcell.sequence.genome.ecoli.k12 import BW25113_ASSEMBLY, EcoliK12BW25113Genome
from torchcell.verification.report import Provenance, VerificationReport
from torchcell.verification.sourced import SourcedValue, audit_sourced_value


# --------------------------------------------------------------------------- #
# Fake .xls workbooks
# --------------------------------------------------------------------------- #
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
    """A one-or-more sheet workbook."""

    def __init__(self, sheets: list[FakeSheet], names: list[str] | None = None) -> None:
        """Hold the sheets and their names."""
        self.sheets = sheets
        self.nsheets = len(sheets)
        self.names = names or [f"Sheet{i}" for i in range(len(sheets))]

    def sheet_by_index(self, i: int) -> FakeSheet:
        """The ``i``-th sheet."""
        return self.sheets[i]

    def sheet_names(self) -> list[str]:
        """The sheet names."""
        return self.names


BOOKS: dict[str, FakeBook] = {}


@pytest.fixture(autouse=True)
def fake_xlrd(monkeypatch: pytest.MonkeyPatch) -> dict[str, FakeBook]:
    """``xlrd.open_workbook`` serves the registered fake books by resolved path."""
    BOOKS.clear()

    def open_workbook(path: str) -> FakeBook:
        return BOOKS[str(Path(path).resolve())]

    monkeypatch.setattr(xlrd, "open_workbook", open_workbook)
    return BOOKS


def _register(path: Path, book: FakeBook) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"xls placeholder")
    BOOKS[str(path.resolve())] = book


NA = m.NOT_TARGETED
#: (eck, gene, jw, b, verified, tested)
KEIO: list[tuple[str, str, str, str, Any, Any]] = [
    ("ECK0001", "thrL", "JW4367", "b0001", 8.0, 8.0),
    ("ECK0002", "thrA", "JW0001", "b0002", 0.0, 8.0),
    ("ECK0003", "hokC", "JW5002", "b4412", 1.0, 8.0),
    ("ECK0004", "yaaP", "JW0004", "b0004", 4.0, 4.0),
    ("ECK0005", "proB", "JW0006", "b0005", 2.0, 8.0),
    ("ECK0005", "proB", "JW0007", "b0005", 3.0, 8.0),
    ("ECK0099", "insX", "JW9999", "none", 5.0, 8.0),
    ("ECK0015", "insL", "JW0015", "b0016", NA, NA),
]
TARGETED_JW = [k[2] for k in KEIO if k[5] != NA]
NOT_OBTAINED_JW = ["JW0001", "JW5002"]
SYNTHETIC_COUNTS = {"targeted": 7, "obtained": 5, "not_obtained": 2}


def table_s3_rows(
    keio: list[tuple[str, str, str, str, Any, Any]] = KEIO,
) -> list[list[Any]]:
    """Title, the header rows the loader checks, and one data row per entry."""
    header1 = [""] * 17
    for (r, c), text in m.TABLE_S3_HEADER.items():
        if r == 1:
            header1[c] = text
    header3 = [""] * 17
    header3[2], header3[8] = "JW id", "b number"
    rows: list[list[Any]] = [[m.TABLE_S3_TITLE], header1, [""] * 17, header3]
    for eck, gene, jw, b, verified, tested in keio:
        percent = NA if tested == NA else 100.0 * verified / tested
        row: list[Any] = [eck, gene, jw] + [""] * 5 + [b] + [""] * 5
        rows.append(row + [verified, tested, percent])
    return rows


def jw_table(title: str, header_row: int, jw_ids: list[str]) -> FakeBook:
    rows: list[list[Any]] = [[title]] + [["", "", ""] for _ in range(header_row)]
    rows[header_row][2] = "JW id" if header_row == 2 else "JW ID"
    rows.extend(["", "", jw] for jw in jw_ids)
    return FakeBook([FakeSheet(rows)])


def write_raw(
    raw: Path, keio: list[tuple[str, str, str, str, Any, Any]] = KEIO
) -> None:
    _register(raw / m.TABLE_S3, FakeBook([FakeSheet(table_s3_rows(keio))]))
    _register(raw / m.TABLE_S2, jw_table(m.TABLE_S2_TITLE, 2, TARGETED_JW))
    _register(raw / m.TABLE_S6, jw_table(m.TABLE_S6_TITLE, 3, NOT_OBTAINED_JW))


# --------------------------------------------------------------------------- #
# The synthetic genome and the hermetic build
# --------------------------------------------------------------------------- #
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
        protein_id=None if pseudo else f"AIN{start:05d}.1",
        protein=None if pseudo else "MK",
    )


BW_LOCI = [
    _locus("BW25113_0001", 1, "thrL", "ECK0001", "JW4367"),
    _locus("BW25113_0002", 12, "thrA", "ECK0002", "JW0001"),
    _locus("BW25113_4412", 30, "hokC", "ECK0003", "JW5002"),
    _locus("BW25113_0004", 44, "yaaP", "ECK0004", "JW0004", pseudo=True),
    _locus("BW25113_0005", 55, "proB", "ECK0005", "JW0006", "JW0007"),
]
UNIVERSE = {locus.tag for locus in BW_LOCI}
REFERENCE = ReferenceGenome(
    species="Escherichia coli",
    strain="BW25113",
    assembly_set="ecoli_K12_BW25113_ASM75055v1",
    assembly_accession="GCA_000750555.1",
)


@pytest.fixture
def bw25113(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EcoliK12BW25113Genome:
    """The real BW25113 class over the synthetic assembly; network refused."""
    files = write_assembly(tmp_path / "tier", BW25113_ASSEMBLY, BW_LOCI)
    forbid_network(monkeypatch)
    serve_tier(monkeypatch, files)
    root = tmp_path / "bw25113"
    root.mkdir()
    return EcoliK12BW25113Genome(genome_root=str(root), overwrite=False)


@pytest.fixture
def synthetic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A dataset root whose raw/ holds the three fake tables; pins checked by presence."""
    root = tmp_path / m.DATASET_ROOT_REL
    write_raw(root / "raw")
    pins: list[Mapping[str, str]] = []

    def presence(raw_dir: str, expected: Mapping[str, str]) -> None:
        missing = [f for f in expected if not osp.exists(osp.join(raw_dir, f))]
        if missing:
            raise FileNotFoundError(f"pinned raw files absent: {missing}")
        pins.append(dict(expected))

    monkeypatch.setattr(m, "verify_raw_files", presence)
    monkeypatch.setattr(m, "EXPECTED_COUNTS", SYNTHETIC_COUNTS)
    monkeypatch.setattr(m, "assembly_reference", lambda strain, **_: REFERENCE)
    return root


@pytest.fixture
def built(
    synthetic: Path, monkeypatch: pytest.MonkeyPatch
) -> m.GeneEssentialityBaba2006Dataset:
    monkeypatch.setattr(m, "EXPECTED_RECORDS", len(TARGETED_JW))
    return m.GeneEssentialityBaba2006Dataset(root=str(synthetic))


def _row(
    jw: str, verified: int | None, tested: int | None, gene: str = "g"
) -> m.KeioRow:
    return m.KeioRow(
        row=1,
        eck="ECK0001",
        gene=gene,
        jw_id=jw,
        b_number="b0001",
        verified=verified,
        tested=tested,
        percent=None
        if tested is None or verified is None
        else 100.0 * verified / tested,
    )


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def test_read_table_s3_reads_targeted_and_untargeted_rows(tmp_path: Path) -> None:
    write_raw(tmp_path)
    rows = m.read_table_s3(tmp_path / m.TABLE_S3)
    assert [(r.row, r.jw_id, r.verified, r.tested, r.targeted) for r in rows] == [
        (1, "JW4367", 8, 8, True),
        (2, "JW0001", 0, 8, True),
        (3, "JW5002", 1, 8, True),
        (4, "JW0004", 4, 4, True),
        (5, "JW0006", 2, 8, True),
        (6, "JW0007", 3, 8, True),
        (7, "JW9999", 5, 8, True),
        (8, "JW0015", None, None, False),
    ]
    assert rows[2].percent == 12.5
    assert [r.obtained for r in rows if r.targeted] == [
        True,
        False,
        False,
        True,
        True,
        True,
        True,
    ]
    with pytest.raises(ValueError, match="was not targeted"):
        _ = rows[7].obtained


def _mutated(row: int, col: int, value: Any) -> list[list[Any]]:
    rows = table_s3_rows()
    rows[row][col] = value
    return rows


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (_mutated(0, 0, "Supplementary Table 9"), "Table 3 title"),
        (_mutated(1, 14, "5. checked"), r"cell \(1, 14\)"),
        (_mutated(4, 2, " JW4367"), "JW id"),
        (_mutated(4, 14, 9.0), "9 of 8 verified"),
        (_mutated(4, 15, 6.0), "8 of 6 verified"),
        (_mutated(4, 14, 7.5), "not a non-negative whole number"),
        (_mutated(4, 16, 50.0), "percent 50.0"),
        (table_s3_rows([*KEIO, KEIO[0]]), "repeats JW ids"),
    ],
)
def test_read_table_s3_refuses_an_unexpected_table(
    tmp_path: Path, rows: list[list[Any]], message: str
) -> None:
    _register(tmp_path / "t3.xls", FakeBook([FakeSheet(rows)]))
    with pytest.raises(m.TableFormatError, match=message):
        m.read_table_s3(tmp_path / "t3.xls")


def test_sheet_refuses_a_second_sheet(tmp_path: Path) -> None:
    sheet = FakeSheet(table_s3_rows())
    _register(tmp_path / "t3.xls", FakeBook([sheet, sheet]))
    with pytest.raises(m.TableFormatError, match="2 sheets"):
        m.read_table_s3(tmp_path / "t3.xls")


def test_read_jw_ids_checks_title_and_header(tmp_path: Path) -> None:
    write_raw(tmp_path)
    assert (
        m.read_jw_ids(tmp_path / m.TABLE_S6, title=m.TABLE_S6_TITLE, header_row=3)
        == NOT_OBTAINED_JW
    )
    with pytest.raises(m.TableFormatError, match="title"):
        m.read_jw_ids(tmp_path / m.TABLE_S6, title=m.TABLE_S2_TITLE, header_row=3)
    with pytest.raises(m.TableFormatError, match="JW header"):
        m.read_jw_ids(tmp_path / m.TABLE_S6, title=m.TABLE_S6_TITLE, header_row=2)


def test_scoring_check_and_require_scoring(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    write_raw(tmp_path)
    rows = m.read_table_s3(tmp_path / m.TABLE_S3)
    check = m.scoring_check(rows, TARGETED_JW, NOT_OBTAINED_JW)
    assert check.counts == SYNTHETIC_COUNTS
    assert (check.not_targeted, check.not_obtained_with_one_verified) == (
        1,
        ["hokC JW5002"],
    )
    assert check.targeted_equals_table_s2 and check.not_obtained_equals_table_s6
    monkeypatch.setattr(m, "EXPECTED_COUNTS", SYNTHETIC_COUNTS)
    m.require_scoring(check)
    with pytest.raises(m.ScoringRuleError, match="the unobtained are not Table 6"):
        m.require_scoring(m.scoring_check(rows, TARGETED_JW, ["JW0001"]))
    monkeypatch.setattr(m, "EXPECTED_COUNTS", {**SYNTHETIC_COUNTS, "obtained": 4})
    with pytest.raises(m.ScoringRuleError, match="the paper states"):
        m.require_scoring(check)


# --------------------------------------------------------------------------- #
# Environments and records
# --------------------------------------------------------------------------- #
def test_plate_media_differ_only_in_the_kanamycin() -> None:
    selection = m.plate_medium(kanamycin=True)
    parent = m.plate_medium(kanamycin=False)
    assert (selection.state, selection.base_medium, selection.is_synthetic) == (
        "solid",
        "LB",
        False,
    )
    names = [c.compound.name for c in selection.components]
    assert names[-1] == "kanamycin" and len(names) == len(parent.components) + 1
    kan = selection.components[-1]
    assert (kan.role, kan.concentration.value if kan.concentration else None) == (
        MediaComponentRole.selection_agent,
        30.0,
    )
    agar = [c for c in parent.components if c.role is MediaComponentRole.gelling_agent]
    assert len(agar) == 1 and agar[0].concentration is None
    env = m.environment()
    assert (env.temperature.value if env.temperature else None, env.aerobicity) == (
        37.0,
        "aerobic",
    )
    assert [g.field for g in env.provenance_gaps] == ["duration_hours"]
    assert m.environment(kanamycin=False).media == parent


def test_build_experiment_stores_the_released_jw_id_and_the_call() -> None:
    env = m.environment()
    obtained = m.build_experiment("baba", _row("JW0001", 6, 8, gene="thrA"), env)
    missing = m.build_experiment("baba", _row("JW0001", 0, 8, gene="thrA"), env)
    assert isinstance(obtained, GeneEssentialityExperiment)
    for record, essential, collection in (
        (obtained, False, "Keio collection"),
        (missing, True, None),
    ):
        assert isinstance(record.genotype, Genotype)
        (p,) = record.genotype.perturbations
        assert isinstance(p, BacterialReleasedStrainPerturbation)
        assert record.phenotype.is_essential is essential
        assert (p.collection, p.cassette, p.systematic_gene_name, p.strain_label) == (
            collection,
            "kanamycin cassette",
            "JW0001",
            "JW0001",
        )
        assert (p.perturbed_gene_name, p.identifier_kind, p.released_kind) == (
            "thrA",
            "Keio JW id",
            "deletion",
        )
    # an N.A. row has no construction outcome: it is a ledger row, never a record
    untargeted = _row("JW0015", None, None, "none")
    assert m.drop_rule_of(untargeted) == "not_targeted"
    assert m.drop_rule_of(_row("JW0001", 0, 8, gene="thrA")) is None
    with pytest.raises(ValueError, match="was not targeted"):
        m.build_experiment("baba", untargeted, env)


def test_evidence_is_the_construction_outcome_with_the_rows_pcr_counts() -> None:
    """#924: the paper's rule, its 'candidates' definition, its caveat on population
    methods and the count definitions, with the row's counts on a targeted record only.
    """
    record = m.build_experiment("baba", _row("JW0001", 1, 8, "thrA"), m.environment())
    evidence = record.phenotype.evidence
    assert evidence.evidence_type is EssentialityEvidenceType.construction_outcome
    assert evidence.evidence_class is EssentialityEvidenceClass.mutant_recovery
    assert (
        evidence.colonies_verified,
        evidence.colonies_tested,
        evidence.percent_verified,
    ) == (1, 8, 12.5)
    assert evidence.scoring_rule is m.SOURCED_VALUES["scoring_rule"]
    assert evidence.scoring_rule.value == 2
    assert evidence.call_definition is m.SOURCED_VALUES["call_definition"]
    assert "candidates for essential genes" in evidence.call_definition.quote
    assert evidence.statements_of(EssentialityStatementRole.caveat) == [
        m.SOURCED_VALUES["population_methods_caveat"]
    ]
    assert evidence.statements_of(EssentialityStatementRole.statistic_definition) == [
        m.SOURCED_VALUES["colonies_tested"],
        m.SOURCED_VALUES["percent_column"],
    ]
    reference = m.build_reference("baba", REFERENCE).phenotype_reference.evidence
    assert reference.colonies_tested is None
    assert reference.statements == evidence.statements


def test_build_reference_is_the_viable_parent_without_kanamycin() -> None:
    reference = m.build_reference("baba", REFERENCE)
    assert reference.phenotype_reference.is_essential is False
    assert reference.environment_reference == m.environment(kanamycin=False)
    assert reference.genome_reference == REFERENCE


# --------------------------------------------------------------------------- #
# The hermetic build
# --------------------------------------------------------------------------- #
def _locus_of(record: Mapping[str, Any]) -> str:
    return str(
        record["experiment"]["genotype"]["perturbations"][0]["systematic_gene_name"]
    )


def test_build_seven_records_and_the_ledgers(
    built: m.GeneEssentialityBaba2006Dataset,
) -> None:
    assert len(built) == 7
    records = [built[i] for i in range(7)]
    assert [
        (_locus_of(r), r["experiment"]["phenotype"]["is_essential"]) for r in records
    ] == [
        ("JW4367", False),
        ("JW0001", True),
        ("JW5002", True),
        ("JW0004", False),
        ("JW0006", False),
        ("JW0007", False),
        ("JW9999", False),
    ]
    insx = records[6]["experiment"]["genotype"]["perturbations"][0]
    assert (insx["perturbed_gene_name"], insx["strain_label"]) == ("insX", "JW9999")
    assert all(r["experiment"]["phenotype"]["provenance_gaps"] == [] for r in records)
    assert records[0]["experiment"]["phenotype"]["evidence"]["colonies_tested"] == 8
    preprocess = Path(built.preprocess_dir)
    drops = json.loads((preprocess / "dropped_records.json").read_text())
    assert (drops["source_rows"], drops["kept_records"], drops["dropped_records"]) == (
        8,
        7,
        1,
    )
    assert (drops["kept_essential"], drops["kept_non_essential"]) == (2, 5)
    assert [
        (rule["rule"], rule["n_records"], rule["items"]) for rule in drops["rules"]
    ] == [("not_targeted", 1, ["insL JW0015"])]
    check = json.loads((preprocess / "scoring_check.json").read_text())
    assert (check["obtained"], check["not_obtained"], check["not_targeted"]) == (
        5,
        2,
        1,
    )
    assert not (preprocess / "identifier_reconciliation.json").exists()
    import pandas as pd

    calls = pd.read_csv(preprocess / "calls.csv")
    assert calls["record"].tolist()[:7] == list(range(7))
    assert pd.isna(calls.loc[7, "record"]) and calls.loc[7, "drop_rule"] == (
        "not_targeted"
    )
    assert pd.isna(calls.loc[7, "obtained"]) and pd.isna(calls.loc[7, "is_essential"])
    assert calls["drop_rule"].isna().sum() == 7
    assert calls["b_number"].tolist()[6] == "none"


def test_build_refuses_a_release_whose_scoring_moved(
    synthetic: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_raw(
        synthetic / "raw",
        [(e, g, j, b, (0.0 if j == "JW4367" else v), t) for e, g, j, b, v, t in KEIO],
    )
    with pytest.raises(m.ScoringRuleError):
        m.GeneEssentialityBaba2006Dataset(root=str(synthetic))


def test_build_refuses_a_row_count_other_than_the_declared_one(
    synthetic: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(m, "EXPECTED_RECORDS", len(KEIO) + 1)
    with pytest.raises(m.TableFormatError, match="the module declares"):
        m.GeneEssentialityBaba2006Dataset(root=str(synthetic))


def test_unused_hooks_and_classes(built: m.GeneEssentialityBaba2006Dataset) -> None:
    import pandas as pd

    frame = pd.DataFrame({"a": [1]})
    assert built.preprocess_raw(frame) is frame
    with pytest.raises(NotImplementedError):
        built.create_experiment()
    assert built.experiment_class.__name__ == "GeneEssentialityExperiment"
    assert built.reference_class.__name__ == "GeneEssentialityExperimentReference"
    assert built.raw_file_names == [m.TABLE_S2, m.TABLE_S3, m.TABLE_S6]


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def _report(
    built: m.GeneEssentialityBaba2006Dataset,
    bw25113: EcoliK12BW25113Genome,
    records: list[dict[str, Any]] | None = None,
) -> VerificationReport:
    from torchcell.verification.runners import load_records

    raw = Path(built.raw_dir)
    rows = m.read_table_s3(raw / m.TABLE_S3)
    check = m.scoring_check(rows, TARGETED_JW, NOT_OBTAINED_JW)
    return m.verify_records(
        records if records is not None else load_records(built.root),
        rows=rows,
        check=check,
        genome=bw25113,
        expected_count=7,
    )


def test_verify_records_passes_the_hermetic_build(
    built: m.GeneEssentialityBaba2006Dataset, bw25113: EcoliK12BW25113Genome
) -> None:
    report = _report(built, bw25113)
    assert report.passed, report.summary()
    assert [r.name for r in report.results][:6] == [
        "structural",
        "count",
        "one_record_per_jw_id",
        "calls_match_table_s3",
        "scoring_rule_reproduces_the_paper_and_table_6",
        "reference_is_the_viable_parent_without_kanamycin",
    ]
    finding = report.results[-1]
    assert finding.name == "gene_containment_bw25113_jw_synonyms" and finding.passed
    # JW9999 is in no annotation (JW0015 is a ledger row, so not counted); JW0006 and
    # JW0007 share BW25113_0005
    assert finding.details["unplaced"] == ["JW9999"]
    assert finding.details["on_shared_locus"] == ["JW0006", "JW0007"]


def test_verify_records_fails_on_tampered_records(
    built: m.GeneEssentialityBaba2006Dataset, bw25113: EcoliK12BW25113Genome
) -> None:
    from torchcell.verification.runners import load_records

    records = load_records(built.root)
    records[0]["experiment"]["phenotype"]["is_essential"] = True
    records[1]["experiment"]["genotype"]["perturbations"][0]["collection"] = (
        "Keio collection"
    )
    records[2]["experiment"]["genotype"]["perturbations"][0]["systematic_gene_name"] = (
        "JW0004"
    )
    records[3] = copy.deepcopy(records[3])
    records[3]["reference"]["phenotype_reference"]["is_essential"] = True
    records[6]["experiment"]["genotype"]["perturbations"][0]["systematic_gene_name"] = (
        "JW0015"
    )
    records.append(records[0])
    failed = {
        r.name: r for r in _report(built, bw25113, records).results if not r.passed
    }
    assert set(failed) >= {
        "count",
        "one_record_per_jw_id",
        "calls_match_table_s3",
        "reference_is_the_viable_parent_without_kanamycin",
    }
    # 1 (flipped call) + 1 (collection on an unobtained deletion) + 3 (the JW id of
    # another row: its call, its PCR counts and its collection) + 1 (a record under the
    # N.A. row's JW id) + 1 (the appended copy of the flipped record)
    assert failed["calls_match_table_s3"].details["n_problems"] == 7


def test_verify_records_fails_on_tampered_evidence(
    built: m.GeneEssentialityBaba2006Dataset, bw25113: EcoliK12BW25113Genome
) -> None:
    """#924: a record whose PCR counts are not its row's fails L2, and a reference whose
    statements are not the sourced ones fails L3.
    """
    from torchcell.verification.runners import load_records

    records = load_records(built.root)
    records[0]["experiment"]["phenotype"]["evidence"]["colonies_tested"] = 4
    # the reference is interned (one object shared by every record): copy it first
    records[1] = copy.deepcopy(records[1])
    records[1]["reference"]["phenotype_reference"]["evidence"]["statements"] = []
    failed = {
        r.name: r for r in _report(built, bw25113, records).results if not r.passed
    }
    assert failed["calls_match_table_s3"].details["n_problems"] == 1
    assert "evidence counts" in failed["calls_match_table_s3"].details["problems"][0]
    assert failed["evidence_is_construction_outcome"].details["n_wrong"] == 1


def test_run_verification_and_cli(
    built: m.GeneEssentialityBaba2006Dataset,
    bw25113: EcoliK12BW25113Genome,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """End to end on the hermetic build, with one synthetic sourced value audited
    against a synthetic raw-mirror text.
    """
    text = tmp_path / "torchcell-raw" / m.CITATION_KEY / m.PAPER_TEXT_REL
    text.parent.mkdir(parents=True)
    text.write_text("E. coli K-12 strain BW25113 was used.")
    value = SourcedValue(
        value="BW25113",
        quote="E. coli K-12 strain BW25113",
        provenance=Provenance(
            source_uri=m.PAPER_TEXT_REL,
            citation_key=m.CITATION_KEY,
            sha256=hashlib.sha256(text.read_bytes()).hexdigest(),
        ),
    )
    monkeypatch.setattr(m, "SOURCED_VALUES", {"strain": value})
    monkeypatch.setattr(m, "bacterial_genome", lambda *a, **k: bw25113)
    report = m.run_verification(str(tmp_path))
    assert report.passed, report.summary()
    assert report.results[-1].name == "provenance_audit"
    written = json.loads(
        (Path(built.root) / "preprocess" / "verification_report.json").read_text()
    )
    assert written["dataset_name"] == "gene_essentiality_baba2006"

    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    assert m.main(["build"]) == 0
    assert capsys.readouterr().out.strip().endswith("len = 7")
    assert m.main(["verify"]) == 0
    # the L4 row is a genome-reference finding: it reads the annotation in the verifier
    # only, passes whatever it counts, and is listed under the verdict as [rp]
    capsys.readouterr()
    assert m.main(["verify"]) == 0
    out = capsys.readouterr().out
    assert "gene_essentiality_baba2006: PASS" in out
    assert "L4 gene_containment_bw25113_jw_synonyms:" in out


def test_download_links_the_mirror_and_checks_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "root"
    mirror = data_root / m.RAW_DIR_REL
    pins: dict[str, str] = {}
    records = []
    for name in m.DATA_SHA256:
        path = mirror / m.mirror_relpath(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        pins[name] = hashlib.sha256(name.encode()).hexdigest()
        records.append(
            ArtifactRecord(
                path=m.mirror_relpath(name),
                role=ROLE_RAW_DATA,
                bytes=len(name),
                sha256=pins[name],
                source="s",
            )
        )
    manifest = Manifest(
        citation_key=m.CITATION_KEY,
        files=records,
        created_at=datetime.now(UTC).isoformat(),
    )
    (mirror / "manifest.json").write_text(manifest.model_dump_json())
    monkeypatch.setattr(m, "DATA_SHA256", pins)
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    dataset = object.__new__(m.GeneEssentialityBaba2006Dataset)
    monkeypatch.setattr(
        m.GeneEssentialityBaba2006Dataset,
        "raw_dir",
        str(tmp_path / "raw"),
        raising=False,
    )
    dataset.download()
    assert sorted(os.listdir(tmp_path / "raw")) == sorted(pins)
    assert m.load_manifest(str(data_root)).citation_key == m.CITATION_KEY
    with pytest.raises(KeyError, match="not in the raw-mirror manifest"):
        m.manifest_sha256(manifest, "data/absent.xls")
    monkeypatch.setattr(m, "DATA_SHA256", {**pins, m.TABLE_S3: "0" * 64})
    with pytest.raises(ManifestPinMismatchError):
        dataset.download()


# --------------------------------------------------------------------------- #
# Real data (``--data``): the mirror and the built dev-tree LMDB, never rebuilt
# --------------------------------------------------------------------------- #
#: Read off Table 3 by hand: thrA 6 of 8 (obtained); ribF, lspA, folC, mreB 0 of 8;
#: secM 1 of 8 (the paper's exception, in Table 6); dedD 6 of 8, the gene the paper
#: deleted with a redesigned primer once its overlap with folC was avoided.
HAND_CHECKED: dict[str, bool] = {
    "JW0001": False,
    "JW0023": True,
    "JW0025": True,
    "JW5007": True,
    "JW5378": False,
    "JW2312": True,
    "JW3220": True,
}


@pytest.mark.data
def test_real_mirror_matches_the_module_pins() -> None:
    from torchcell.data.experiment_dataset import file_sha256

    root = os.environ["DATA_ROOT"]
    manifest = m.load_manifest(root)
    for name, sha256 in m.DATA_SHA256.items():
        assert m.manifest_sha256(manifest, m.mirror_relpath(name)) == sha256
        assert file_sha256(m.raw_mirror_dir(root) / m.mirror_relpath(name)) == sha256
    assert m.manifest_sha256(manifest, m.PAPER_TEXT_REL) == m.PAPER_TEXT_SHA256


@pytest.mark.data
@pytest.mark.parametrize("key", sorted(m.SOURCED_VALUES))
def test_real_sourced_values_are_verbatim(key: str) -> None:
    result = audit_sourced_value(
        m.SOURCED_VALUES[key], Path(os.environ["DATA_ROOT"]) / "torchcell-raw"
    )
    assert result.passed, result.message


@pytest.mark.data
def test_real_build_counts_and_hand_checked_rows() -> None:
    """4,288 records = every targeted Table 3 row: 3,985 obtained, 303 not; the 90
    not-targeted rows are ledgered under ``not_targeted``.
    """
    from torchcell.verification.runners import load_records

    root = osp.join(os.environ["DATA_ROOT"], m.DATASET_ROOT_REL)
    drops = json.loads(Path(root, "preprocess", "dropped_records.json").read_text())
    assert (
        drops["source_rows"],
        drops["kept_records"],
        drops["kept_essential"],
        drops["kept_non_essential"],
        drops["dropped_records"],
        [(rule["rule"], rule["n_records"]) for rule in drops["rules"]],
    ) == (4378, 4288, 303, 3985, 90, [("not_targeted", 90)])
    records = load_records(root)
    calls = {
        _locus_of(r): r["experiment"]["phenotype"]["is_essential"] for r in records
    }
    assert {k: calls[k] for k in HAND_CHECKED} == HAND_CHECKED
    assert sum(1 for v in calls.values() if v is None) == 0
    assert "JW4296.5" in calls
