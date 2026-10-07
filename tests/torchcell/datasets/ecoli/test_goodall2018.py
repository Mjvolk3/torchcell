# tests/torchcell/datasets/ecoli/test_goodall2018.py
# [[tests.torchcell.datasets.ecoli.test_goodall2018]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/ecoli/test_goodall2018.py
"""The Goodall 2018 TraDIS essentiality loader (``torchcell.datasets.ecoli.goodall2018``).

Synthetic tests (run everywhere) build every input in ``tmp_path``. The hermetic build
uses the real ``EcoliK12BW25113Genome`` over the synthetic BW25113 assembly of
``tests/torchcell/sequence/genome/_bacterial_fixtures.py`` (thrL BW25113_0001, thrA
BW25113_0002, hokC BW25113_4412, pseudogene yaaP BW25113_0004, proB BW25113_0005, and
ECK0005 a synonym of two loci), served through a stubbed ``resolve`` with the network
refused, and replaces the loader's ``verify_raw_files`` with a presence check (the
synthetic tables cannot carry the real pins; the pins are asserted by the refusal test
and the data-gated tests). The two synthetic tables:

    row  gene     Table S1 (TL)              Table S4 (LB)
    1    thrL     essential  (-20)           essential  (-30)
    2    thrA     non_essential (15)         essential  (-4)
    3    hokC     unclear (1)    -> dropped  non_essential (9)
    4    yaaP     essential (-5), pseudogene unclear (2)    -> dropped
    5    ECK0005  non_essential, ambiguous -> dropped in both
    6    proB     non_essential (12)         unclear (3.59) -> dropped; the exact log2(12)
                                             cut would call it non-essential

Five of six distinct names resolve (0.833 < 0.99), so the success path lowers
``MIN_RESOLVED_FRACTION`` to 0.8 and a separate test shows the default stops the build.
Seven records: TL thrL, thrA, yaaP, proB; LB thrL, thrA, hokC.

Data-gated tests (``@pytest.mark.data``) read the real raw mirror and the built dev-tree
LMDB under ``$DATA_ROOT`` (they never build it): the manifest pins, the provenance audit
of every sourced value, the counts, hand-checked rows (read off ``si2.xlsx`` and
``si7.xlsx`` with openpyxl), and the L0-L4 verifier.
"""

from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import openpyxl
import pytest

import torchcell.datasets.ecoli.goodall2018 as m
from tests.torchcell.sequence.genome._bacterial_fixtures import (
    BW25113_LOCI,
    forbid_network,
    serve_tier,
    write_assembly,
)
from torchcell.data import ManifestPinMismatchError, RawSha256MismatchError
from torchcell.datamodels.media import LB
from torchcell.datamodels.schema import (
    AssemblyReferenceGenome,
    BacterialGeneEssentialityExperiment,
    BacterialGeneEssentialityExperimentReference,
    MediaComponentRole,
    TransposonInsertionPerturbation,
)
from torchcell.datasets.bacteria_common import LocusTagResolutionError
from torchcell.literature.manifest import RetrievalMethod, RetrievalRecord
from torchcell.sequence.genome.base import GeneNameStatus
from torchcell.sequence.genome.ecoli.k12 import BW25113_ASSEMBLY, EcoliK12BW25113Genome
from torchcell.verification.report import Provenance, VerificationReport
from torchcell.verification.sourced import SourcedValue, audit_sourced_value

Row = tuple[Any, ...]
TL_ROWS: list[Row] = [
    ("thrL", 0.0, -20.0, True, False, False),
    ("thrA", 0.3, 15.0, False, True, False),
    ("hokC", 0.06, 1.0, False, False, True),
    ("yaaP", 0.01, -5.0, True, False, False),
    ("ECK0005", 0.2, 10.0, False, True, False),
    ("proB", 0.25, 12.0, False, True, False),
]
LB_ROWS: list[Row] = [
    ("thrL", 0, -30.0, True, False, False),
    ("thrA", 0.02, -4.0, True, False, False),
    ("hokC", 0.2, 9.0, False, True, False),
    ("yaaP", 0.05, 2.0, False, False, True),
    ("ECK0005", 0.2, 10.0, False, True, False),
    ("proB", 0.03, 3.59, False, False, True),
]
SYNTHETIC_TL_COUNTS = {"essential": 2, "non_essential": 3, "unclear": 1}
REFERENCE = AssemblyReferenceGenome(
    species="Escherichia coli",
    strain="BW25113",
    assembly_set="ecoli_K12_BW25113_ASM75055v1",
    assembly_accession="GCA_000750555.1",
)
TL_SPEC = m.CONDITIONS_BY_KEY["TL"]
LB_SPEC = m.CONDITIONS_BY_KEY["LB"]


def _write_table(
    path: Path,
    title: str,
    rows: list[Row],
    *,
    header: tuple[str, ...] = m.TABLE_COLUMNS,
    extra_sheet: bool = False,
) -> None:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.append([title])
    sheet.append(list(header))
    for row in rows:
        sheet.append(list(row))
    if extra_sheet:
        workbook.create_sheet("second")
    workbook.save(path)


def _write_raw(raw: Path, tl: list[Row] = TL_ROWS, lb: list[Row] = LB_ROWS) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    _write_table(raw / m.TABLE_S1, TL_SPEC.title, tl)
    _write_table(raw / m.TABLE_S4, LB_SPEC.title, lb)


def _call(
    gene: str,
    call: m.EssentialityCall,
    *,
    row: int = 1,
    condition: m.ConditionKey = "TL",
    ratio: float = -10.0,
) -> m.GeneCall:
    return m.GeneCall(
        condition=condition,
        row=row,
        gene=gene,
        insertion_index=0.1,
        log_likelihood_ratio=ratio,
        call=call,
    )


@pytest.fixture
def bw25113(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EcoliK12BW25113Genome:
    """The real BW25113 class over the synthetic assembly; network refused."""
    files = write_assembly(tmp_path / "tier", BW25113_ASSEMBLY, BW25113_LOCI)
    forbid_network(monkeypatch)
    serve_tier(monkeypatch, files)
    root = tmp_path / "bw25113"
    root.mkdir()
    return EcoliK12BW25113Genome(genome_root=str(root), overwrite=False)


@pytest.fixture
def presence_only_pins(monkeypatch: pytest.MonkeyPatch) -> list[Mapping[str, str]]:
    """Replace the build-time byte check with a presence check that records the pins."""
    calls: list[Mapping[str, str]] = []

    def record(raw_dir: str, pins: Mapping[str, str]) -> None:
        missing = [f for f in pins if not osp.exists(osp.join(raw_dir, f))]
        if missing:
            raise FileNotFoundError(f"pinned raw files absent: {missing}")
        calls.append(dict(pins))

    monkeypatch.setattr(m, "verify_raw_files", record)
    return calls


@pytest.fixture
def synthetic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bw25113: EcoliK12BW25113Genome,
    presence_only_pins: list[Mapping[str, str]],
) -> Path:
    """A dataset root at ``<tmp>/data/torchcell/...`` whose raw/ holds both tables."""
    root = tmp_path / m.DATASET_ROOT_REL
    _write_raw(root / "raw")
    monkeypatch.setattr(m, "EXPECTED_TL_COUNTS", SYNTHETIC_TL_COUNTS)
    monkeypatch.setattr(m, "assembly_reference", lambda strain, **_: REFERENCE)
    return root


@pytest.fixture
def built(
    synthetic: Path, monkeypatch: pytest.MonkeyPatch, bw25113: EcoliK12BW25113Genome
) -> m.GeneEssentialityGoodall2018Dataset:
    monkeypatch.setattr(m, "MIN_RESOLVED_FRACTION", 0.8)
    return m.GeneEssentialityGoodall2018Dataset(
        root=str(synthetic), ecoli_genome=bw25113
    )


# --------------------------------------------------------------------------- #
# Parsing the released tables
# --------------------------------------------------------------------------- #
def test_read_calls_reads_every_row(tmp_path: Path) -> None:
    _write_raw(tmp_path)
    calls = m.read_calls(tmp_path / m.TABLE_S4, LB_SPEC)
    assert [(c.row, c.gene, c.call) for c in calls] == [
        (1, "thrL", "essential"),
        (2, "thrA", "essential"),
        (3, "hokC", "non_essential"),
        (4, "yaaP", "unclear"),
        (5, "ECK0005", "non_essential"),
        (6, "proB", "unclear"),
    ]
    assert calls[0] == m.GeneCall(
        condition="LB",
        row=1,
        gene="thrL",
        insertion_index=0.0,
        log_likelihood_ratio=-30.0,
        call="essential",
    )


@pytest.mark.parametrize(
    ("title", "header", "rows", "message"),
    [
        ("Table S9", m.TABLE_COLUMNS, TL_ROWS, "title 'Table S9'"),
        (TL_SPEC.title, ("Gene", "x", "y", "a", "b", "c"), TL_ROWS, "header"),
        (
            TL_SPEC.title,
            m.TABLE_COLUMNS,
            [("thrL", 0.1, -9.0, True, True, False)],
            "data row 1: call flags",
        ),
        (
            TL_SPEC.title,
            m.TABLE_COLUMNS,
            [("thrL", 0.1, -9.0, "TRUE", False, False)],
            "data row 1: call flags",
        ),
        (
            TL_SPEC.title,
            m.TABLE_COLUMNS,
            [("thrL", "n/a", -9.0, True, False, False)],
            "'n/a' is not a number",
        ),
        (
            TL_SPEC.title,
            m.TABLE_COLUMNS,
            [("thrL", -0.1, -9.0, True, False, False)],
            "insertion index -0.1 < 0",
        ),
        (
            TL_SPEC.title,
            m.TABLE_COLUMNS,
            [(" thrL", 0.1, -9.0, True, False, False)],
            "gene ' thrL'",
        ),
        (
            TL_SPEC.title,
            m.TABLE_COLUMNS,
            [("thrL", 0.1, -9.0, True, False, False, "extra")],
            "header",
        ),
    ],
)
def test_read_calls_refuses_an_unexpected_table(
    tmp_path: Path, title: str, header: tuple[str, ...], rows: list[Row], message: str
) -> None:
    path = tmp_path / m.TABLE_S1
    _write_table(path, title, rows, header=header)
    with pytest.raises(m.TableFormatError, match=message.replace("(", r"\(")):
        m.read_calls(path, TL_SPEC)


def test_read_calls_refuses_a_second_sheet(tmp_path: Path) -> None:
    path = tmp_path / m.TABLE_S1
    _write_table(path, TL_SPEC.title, TL_ROWS, extra_sheet=True)
    with pytest.raises(m.TableFormatError, match="si2.xlsx: 2 sheets"):
        m.read_calls(path, TL_SPEC)


@pytest.mark.parametrize(
    ("value", "message"),
    [(True, "True is not a number"), (float("inf"), "inf is not finite")],
)
def test_number_refuses_bools_and_non_finite_values(value: Any, message: str) -> None:
    with pytest.raises(m.TableFormatError, match=f"here: {message}"):
        m._number(value, "here")


def test_call_from_ratio_is_strict_at_the_threshold() -> None:
    assert m.call_from_ratio(-3.61, 3.6) == "essential"
    assert m.call_from_ratio(-3.6, 3.6) == "unclear"
    assert m.call_from_ratio(3.6, 3.6) == "unclear"
    assert m.call_from_ratio(3.61, 3.6) == "non_essential"


def test_call_rule_check_lists_disagreements_and_the_log2_12_movers() -> None:
    calls = [
        _call("a", "essential", ratio=-5.0),
        _call("b", "unclear", row=2, ratio=3.59),
        _call("c", "essential", row=3, ratio=5.0),
    ]
    check = m.call_rule_check(calls, "TL", 3.6)
    assert (check.n_rows, check.n_agree) == (3, 2)
    assert check.disagreements == ["c (row 3): ratio 5.0 released essential"]
    assert check.moved_at_exact_log2_12 == ["b (row 2): ratio 3.59"]


def test_call_counts_lists_every_call() -> None:
    assert m.call_counts([_call("a", "essential")]) == {
        "essential": 1,
        "non_essential": 0,
        "unclear": 0,
    }


# --------------------------------------------------------------------------- #
# Identifiers and the ledger
# --------------------------------------------------------------------------- #
def test_resolve_calls_keeps_loci_and_names_every_drop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bw25113: EcoliK12BW25113Genome
) -> None:
    monkeypatch.setattr(m, "MIN_RESOLVED_FRACTION", 0.8)
    _write_raw(tmp_path)
    calls = m.read_calls(tmp_path / m.TABLE_S1, TL_SPEC)
    outcomes, report = m.resolve_calls(bw25113, calls, label="synthetic")
    assert [(o.call.gene, o.locus_tag, o.symbol, o.drop_rule) for o in outcomes] == [
        ("thrL", "BW25113_0001", "thrL", None),
        ("thrA", "BW25113_0002", "thrA", None),
        ("hokC", "BW25113_4412", "hokC", "unclear_call_not_representable"),
        ("yaaP", "BW25113_0004", "yaaP", None),
        ("ECK0005", None, None, "symbol_not_on_one_bw25113_locus"),
        ("proB", "BW25113_0005", "proB", None),
    ]
    assert outcomes[4].reason == "ambiguous over BW25113_0005, BW25113_0008"
    assert report.status_histogram[GeneNameStatus.NON_GENE_FEATURE] == 1
    rules = m.drop_rules(outcomes)
    assert [(r.rule, r.items) for r in rules] == [
        (
            "symbol_not_on_one_bw25113_locus",
            ["TL row 5 ECK0005: ambiguous over BW25113_0005, BW25113_0008"],
        ),
        ("unclear_call_not_representable", ["TL row 3 hokC (BW25113_4412): ratio 1.0"]),
    ]


def test_resolve_calls_refuses_two_kept_rows_on_one_locus(
    monkeypatch: pytest.MonkeyPatch, bw25113: EcoliK12BW25113Genome
) -> None:
    monkeypatch.setattr(m, "MIN_RESOLVED_FRACTION", 0.5)
    calls = [_call("thrL", "essential"), _call("thrL", "essential", row=2)]
    with pytest.raises(
        m.DuplicateRecordError, match="TL: two kept rows on BW25113_0001"
    ):
        m.resolve_calls(bw25113, calls, label="synthetic")


def test_unplaced_reasons_cover_collision_and_retirement(
    monkeypatch: pytest.MonkeyPatch, bw25113: EcoliK12BW25113Genome
) -> None:
    """``thrL`` and ``ECK0001`` both resolve to BW25113_0001 (kept as given); ``yaaQ``
    is retired; all three are dropped with their reason.
    """
    monkeypatch.setattr(m, "MIN_RESOLVED_FRACTION", 0.0)
    calls = [
        _call("thrL", "essential"),
        _call("ECK0001", "essential", row=2),
        _call("yaaQ", "essential", row=3),
    ]
    outcomes, _ = m.resolve_calls(bw25113, calls, label="synthetic")
    assert [(o.drop_rule, o.reason) for o in outcomes] == [
        (
            "symbol_not_on_one_bw25113_locus",
            "kept as given on collision with another resolving name",
        ),
        (
            "symbol_not_on_one_bw25113_locus",
            "kept as given on collision with another resolving name",
        ),
        ("symbol_not_on_one_bw25113_locus", "retired"),
    ]


def test_build_drop_log_refuses_an_unaccounted_drop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcome = m.RowOutcome(
        call=_call("a", "essential"),
        locus_tag=None,
        symbol=None,
        drop_rule="symbol_not_on_one_bw25113_locus",
        reason="retired",
    )
    log = m.build_drop_log("x", [outcome])
    assert (log.source_records, log.kept_records, log.dropped_records) == (1, 0, 1)
    monkeypatch.setattr(m, "drop_rules", lambda outcomes: [])
    with pytest.raises(RuntimeError, match="do not account for every dropped row"):
        m.build_drop_log("x", [outcome])


def test_check_replicon_refuses_another_chromosome(
    monkeypatch: pytest.MonkeyPatch, bw25113: EcoliK12BW25113Genome
) -> None:
    m.check_replicon(bw25113)
    monkeypatch.setattr(m, "PAPER_REPLICON", "NC_000913.3")
    with pytest.raises(ValueError, match=r"sit on \['CP009273.1'\]"):
        m.check_replicon(bw25113)


def test_canonical_symbol_falls_back_to_the_tag(bw25113: EcoliK12BW25113Genome) -> None:
    """A locus without a symbol keeps its tag; a symbol that resolves elsewhere too."""
    assert m.canonical_symbol(bw25113, "BW25113_0002") == "thrA"
    loci = bw25113.genbank.loci
    original = loci["BW25113_0002"]
    loci["BW25113_0002"] = original.model_copy(update={"symbol": None})
    try:
        assert m.canonical_symbol(bw25113, "BW25113_0002") == "BW25113_0002"
        loci["BW25113_0002"] = original.model_copy(update={"symbol": "thrL"})
        assert m.canonical_symbol(bw25113, "BW25113_0002") == "BW25113_0002"
    finally:
        loci["BW25113_0002"] = original


# --------------------------------------------------------------------------- #
# Environments and records
# --------------------------------------------------------------------------- #
def test_selection_medium_derives_from_lb_with_unstated_agar_and_drug() -> None:
    medium = m.selection_medium()
    assert medium.base_medium == "LB"
    assert medium.state == "solid"
    assert medium.components[: len(LB.components)] == LB.components
    agar, drug = medium.components[len(LB.components) :]
    assert (agar.compound.name, agar.role, agar.concentration) == (
        "agar",
        MediaComponentRole.gelling_agent,
        None,
    )
    assert (drug.compound.name, drug.role, drug.concentration) == (
        "chloramphenicol",
        MediaComponentRole.selection_agent,
        None,
    )
    assert medium.provenance == [m.SOURCED_VALUES["selection_medium"]]
    assert m.SELECTION_MEDIUM_STATEMENT is m.SOURCED_VALUES["selection_medium"]


def test_environments_carry_their_sourced_values_and_gaps() -> None:
    tl = m.environment("TL")
    assert tl.temperature is None and tl.duration_hours is None
    assert tl.gapped_fields() == {"temperature", "duration_hours"}
    lb = m.environment("LB")
    assert lb.media == LB
    assert lb.temperature is not None and lb.temperature.value == 37.0
    assert lb.aerobicity == "aerobic"
    assert lb.gapped_fields() == {"duration_generations"}


def test_perturbation_field_gaps_name_real_leaf_fields() -> None:
    fields = {gap.field for gap in m.PERTURBATION_FIELD_GAPS}
    assert fields == {"insertion_position", "insertion_strand"}
    assert fields <= set(TransposonInsertionPerturbation.model_fields)


def _kept(gene: str, tag: str, call: m.EssentialityCall) -> m.RowOutcome:
    return m.RowOutcome(
        call=_call(gene, call), locus_tag=tag, symbol=gene, drop_rule=None, reason=None
    )


def test_build_experiment_is_one_gene_level_insertion() -> None:
    experiment = m.build_experiment(
        "GeneEssentialityGoodall2018Dataset",
        _kept("thrA", "BW25113_0002", "essential"),
        m.environment("LB"),
    )
    dumped = experiment.model_dump()
    assert dumped["experiment_type"] == "bacterial_gene_essentiality"
    assert dumped["phenotype"]["is_essential"] is True
    (perturbation,) = dumped["genotype"]["perturbations"]
    assert {
        k: perturbation[k]
        for k in (
            "systematic_gene_name",
            "perturbed_gene_name",
            "perturbation_type",
            "gene_namespace",
            "state",
            "mechanism_so_id",
            "transposon",
            "barcode",
            "insertion_position",
            "insertion_strand",
            "library_pool",
        )
    } == {
        "systematic_gene_name": "BW25113_0002",
        "perturbed_gene_name": "thrA",
        "perturbation_type": "transposon_insertion",
        "gene_namespace": "ecoli_k12_bw25113_locus_tag",
        "state": "absent",
        "mechanism_so_id": "SO:0001218",
        "transposon": "mini-Tn5",
        "barcode": None,
        "insertion_position": None,
        "insertion_strand": None,
        "library_pool": None,
    }
    non_essential = m.build_experiment(
        "x", _kept("thrA", "BW25113_0002", "non_essential"), m.environment("LB")
    )
    assert non_essential.phenotype.is_essential is False


def test_build_experiment_refuses_a_dropped_row() -> None:
    dropped = m.RowOutcome(
        call=_call("hokC", "unclear", row=3),
        locus_tag="BW25113_4412",
        symbol="hokC",
        drop_rule="unclear_call_not_representable",
        reason="ratio 1.0",
    )
    with pytest.raises(ValueError, match="row 3 was dropped"):
        m.build_experiment("x", dropped, m.environment("TL"))


def test_build_reference_keeps_the_assembly_pin_and_a_viable_parent() -> None:
    dumped = m.build_reference("x", REFERENCE, m.environment("TL")).model_dump()
    assert dumped["genome_reference"]["assembly_set"] == "ecoli_K12_BW25113_ASM75055v1"
    assert dumped["genome_reference"]["assembly_accession"] == "GCA_000750555.1"
    assert dumped["phenotype_reference"]["is_essential"] is False
    assert dumped["environment_reference"] == m.environment("TL").model_dump()


# --------------------------------------------------------------------------- #
# Hermetic build
# --------------------------------------------------------------------------- #
def _gene(record: Mapping[str, Any]) -> str:
    return str(
        record["experiment"]["genotype"]["perturbations"][0]["perturbed_gene_name"]
    )


def test_build_seven_records_with_two_references_and_ledgers(
    built: m.GeneEssentialityGoodall2018Dataset,
    synthetic: Path,
    presence_only_pins: list[Mapping[str, str]],
) -> None:
    assert presence_only_pins == [m.DATA_SHA256]
    assert len(built) == 7
    records = [built[i] for i in range(7)]
    tl_env = m.environment("TL").model_dump()
    lb_env = m.environment("LB").model_dump()
    observed = [
        (
            "TL" if r["experiment"]["environment"] == tl_env else "LB",
            _gene(r),
            r["experiment"]["phenotype"]["is_essential"],
        )
        for r in records
    ]
    assert observed == [
        ("TL", "thrL", True),
        ("TL", "thrA", False),
        ("TL", "yaaP", True),
        ("TL", "proB", False),
        ("LB", "thrL", True),
        ("LB", "thrA", True),
        ("LB", "hokC", False),
    ]
    assert records[2]["experiment"]["genotype"]["perturbations"][0][
        "systematic_gene_name"
    ] == ("BW25113_0004")
    assert records[0]["reference"]["environment_reference"] == tl_env
    assert records[6]["reference"]["environment_reference"] == lb_env
    assert records[0]["reference"]["genome_reference"] == REFERENCE.model_dump()
    assert records[0]["publication"] == m.PUBLICATION.model_dump()

    preprocess = synthetic / "preprocess"
    drops = json.loads((preprocess / "dropped_records.json").read_text())
    assert (
        drops["source_records"],
        drops["kept_records"],
        drops["dropped_records"],
    ) == (12, 7, 5)
    assert [(c["condition"], c["kept_records"]) for c in drops["conditions"]] == [
        ("TL", 4),
        ("LB", 3),
    ]
    assert [(r["rule"], r["n_records"]) for r in drops["rules"]] == [
        ("symbol_not_on_one_bw25113_locus", 2),
        ("unclear_call_not_representable", 3),
    ]
    rule = json.loads((preprocess / "call_rule.json").read_text())
    assert rule[1]["moved_at_exact_log2_12"] == ["proB (row 6): ratio 3.59"]
    gaps = json.loads((preprocess / "perturbation_field_gaps.json").read_text())
    assert [g["field"] for g in gaps] == ["insertion_position", "insertion_strand"]
    ledger = json.loads((preprocess / "identifier_reconciliation.json").read_text())
    assert ledger["min_resolved_fraction"] == 0.8
    assert ledger["reconciliation"]["ambiguous_kept"] == {
        "ECK0005": ["BW25113_0005", "BW25113_0008"]
    }
    calls = (preprocess / "calls.csv").read_text().splitlines()
    assert calls[0] == (
        "condition,row,gene,insertion_index,log_likelihood_ratio,call,locus_tag,"
        "symbol,drop_rule,record"
    )
    assert calls[1] == "TL,1,thrL,0.0,-20.0,essential,BW25113_0001,thrL,,0"
    assert calls[3] == (
        "TL,3,hokC,0.06,1.0,unclear,BW25113_4412,hokC,unclear_call_not_representable,"
    )
    assert calls[12] == (
        "LB,6,proB,0.03,3.59,unclear,BW25113_0005,proB,unclear_call_not_representable,"
    )


def test_build_stops_below_the_resolution_threshold(
    synthetic: Path, bw25113: EcoliK12BW25113Genome
) -> None:
    with pytest.raises(LocusTagResolutionError, match=r"5 of 6 names \(0\.833\)"):
        m.GeneEssentialityGoodall2018Dataset(root=str(synthetic), ecoli_genome=bw25113)
    assert not (synthetic / "processed" / "lmdb").exists()


def test_build_refuses_a_genome_of_another_strain(
    synthetic: Path, monkeypatch: pytest.MonkeyPatch, bw25113: EcoliK12BW25113Genome
) -> None:
    monkeypatch.setattr(
        m, "BACTERIAL_ASSEMBLY_SETS", {"BW25113": "ecoli_K12_MG1655_ASM584v2"}
    )
    with pytest.raises(ValueError, match="needs the ecoli_K12_MG1655_ASM584v2 genome"):
        m.GeneEssentialityGoodall2018Dataset(root=str(synthetic), ecoli_genome=bw25113)


def test_build_loads_the_default_genome_on_a_direct_run(
    synthetic: Path, monkeypatch: pytest.MonkeyPatch, bw25113: EcoliK12BW25113Genome
) -> None:
    monkeypatch.setattr(m, "MIN_RESOLVED_FRACTION", 0.8)
    asked: list[tuple[str, str]] = []

    def default(host: str, strain: str) -> EcoliK12BW25113Genome:
        asked.append((host, strain))
        return bw25113

    monkeypatch.setattr(m, "bacterial_genome", default)
    dataset = m.GeneEssentialityGoodall2018Dataset(root=str(synthetic))
    assert asked == [("ecoli", "BW25113")]
    assert len(dataset) == 7


def test_build_refuses_a_call_off_the_threshold(
    synthetic: Path, bw25113: EcoliK12BW25113Genome
) -> None:
    rows = [*TL_ROWS[:-1], ("proB", 0.25, 3.0, False, True, False)]
    _write_raw(synthetic / "raw", tl=rows)
    with pytest.raises(m.CallRuleError, match="si2.xlsx: 1 calls disagree"):
        m.GeneEssentialityGoodall2018Dataset(root=str(synthetic), ecoli_genome=bw25113)


def test_build_refuses_table_s1_counts_other_than_the_paper(
    synthetic: Path, monkeypatch: pytest.MonkeyPatch, bw25113: EcoliK12BW25113Genome
) -> None:
    monkeypatch.setattr(m, "EXPECTED_TL_COUNTS", m.SOURCED_VALUES["tl_counts"].value)
    with pytest.raises(m.CallCountError, match="the paper states"):
        m.GeneEssentialityGoodall2018Dataset(root=str(synthetic), ecoli_genome=bw25113)


def test_process_verifies_the_real_pins(
    synthetic: Path, monkeypatch: pytest.MonkeyPatch, bw25113: EcoliK12BW25113Genome
) -> None:
    """With the real check back, the synthetic Table S1 is refused by its pin."""
    from torchcell.data import verify_raw_files

    monkeypatch.setattr(m, "verify_raw_files", verify_raw_files)
    with pytest.raises(RawSha256MismatchError, match=m.DATA_SHA256[m.TABLE_S1]):
        m.GeneEssentialityGoodall2018Dataset(root=str(synthetic), ecoli_genome=bw25113)


def test_unused_hooks_are_inert(built: m.GeneEssentialityGoodall2018Dataset) -> None:
    assert built.experiment_class is BacterialGeneEssentialityExperiment
    assert built.reference_class is BacterialGeneEssentialityExperimentReference
    assert built.raw_file_names == [m.TABLE_S1, m.TABLE_S4]
    sentinel = object()
    assert built.preprocess_raw(sentinel) is sentinel  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError):
        built.create_experiment()


# --------------------------------------------------------------------------- #
# Verification on the hermetic build
# --------------------------------------------------------------------------- #
UNIVERSE = {locus.tag for locus in BW25113_LOCI}


def _hermetic_report(
    built: m.GeneEssentialityGoodall2018Dataset,
    bw25113: EcoliK12BW25113Genome,
    records: list[dict[str, Any]] | None = None,
) -> VerificationReport:
    from torchcell.verification.runners import load_records

    raw = Path(built.raw_dir)
    raw_calls: dict[m.ConditionKey, list[m.GeneCall]] = {
        spec.key: m.read_calls(raw / spec.table, spec) for spec in m.CONDITIONS
    }
    return m.verify_records(
        records if records is not None else load_records(built.root),
        raw_calls=raw_calls,
        resolve=bw25113.resolve_gene_name,
        universe=UNIVERSE,
        expected_count=7,
    )


def test_verify_records_passes_the_hermetic_build(
    built: m.GeneEssentialityGoodall2018Dataset, bw25113: EcoliK12BW25113Genome
) -> None:
    report = _hermetic_report(built, bw25113)
    assert report.passed, report.summary()
    names = [r.name for r in report.results]
    assert names[:6] == [
        "structural",
        "count",
        "one_record_per_gene_and_condition",
        "calls_match_released_tables",
        "call_threshold_convention",
        "table_s1_counts_equal_the_paper",
    ]
    assert "media_membership" in names
    assert names[-1] == "gene_containment_bw25113_locus_tags"


def test_verify_records_fails_on_tampered_records(
    built: m.GeneEssentialityGoodall2018Dataset, bw25113: EcoliK12BW25113Genome
) -> None:
    """A flipped call, a record in no condition's environment, a duplicate, a locus off
    the universe and a symbol on no table row each fail their level.
    """
    from torchcell.verification.runners import load_records

    records = load_records(built.root)
    tl_thrl = records[0]
    assert _gene(tl_thrl) == "thrL"
    flipped = {**tl_thrl, "experiment": {**tl_thrl["experiment"]}}
    flipped["experiment"]["phenotype"] = {
        **tl_thrl["experiment"]["phenotype"],
        "is_essential": False,
    }
    stray = {**records[1], "experiment": {**records[1]["experiment"]}}
    stray["experiment"]["environment"] = {
        **records[1]["experiment"]["environment"],
        "aerobicity": "anaerobic",
    }
    off = {**records[2], "experiment": {**records[2]["experiment"]}}
    perturbation = {
        **records[2]["experiment"]["genotype"]["perturbations"][0],
        "systematic_gene_name": "BW25113_9999",
        "perturbed_gene_name": "yaaX",
    }
    off["experiment"]["genotype"] = {"perturbations": [perturbation]}
    tampered = [flipped, stray, off, records[3], records[3]]
    report = _hermetic_report(built, bw25113, tampered)
    failed = {r.name for r in report.results if not r.passed}
    assert {
        "count",
        "one_record_per_gene_and_condition",
        "calls_match_released_tables",
        "gene_containment_bw25113_locus_tags",
    } <= failed
    (l2,) = [r for r in report.results if r.name == "calls_match_released_tables"]
    problems = l2.details["problems"]
    assert any("stored False, table essential" in p for p in problems)
    assert any("None thrA: 0 table rows" in p for p in problems)
    assert any("yaaX: 0 table rows" in p for p in problems)


def test_verify_records_flags_a_locus_the_row_does_not_resolve_to(
    built: m.GeneEssentialityGoodall2018Dataset, bw25113: EcoliK12BW25113Genome
) -> None:
    from torchcell.verification.runners import load_records

    records = load_records(built.root)
    record = records[0]
    moved = {**record, "experiment": {**record["experiment"]}}
    perturbation = {
        **record["experiment"]["genotype"]["perturbations"][0],
        "systematic_gene_name": "BW25113_0008",
    }
    moved["experiment"]["genotype"] = {"perturbations": [perturbation]}
    report = _hermetic_report(built, bw25113, [moved, *records[1:]])
    (l2,) = [r for r in report.results if r.name == "calls_match_released_tables"]
    assert not l2.passed
    assert l2.details["problems"] == [
        f"TL {_gene(record)}: resolves renamed BW25113_0001, stored BW25113_0008"
    ]


def test_run_verification_writes_the_report(
    built: m.GeneEssentialityGoodall2018Dataset,
    bw25113: EcoliK12BW25113Genome,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end on the hermetic build: the genome and the gene universe are the
    synthetic ones, and one synthetic sourced value is audited against a synthetic
    library file.
    """
    import torchcell.verification.runners as runners

    paper = tmp_path / "torchcell-library" / m.CITATION_KEY / "paper.md"
    paper.parent.mkdir(parents=True)
    paper.write_text("We used E. coli K-12 strain BW25113.")
    value = SourcedValue(
        value="BW25113",
        quote="E. coli K-12 strain BW25113",
        provenance=Provenance(
            source_uri="paper.md",
            citation_key=m.CITATION_KEY,
            sha256=hashlib.sha256(paper.read_bytes()).hexdigest(),
        ),
    )
    monkeypatch.setattr(m, "SOURCED_VALUES", {"strain": value})
    monkeypatch.setattr(m, "bacterial_genome", lambda *a, **k: bw25113)
    monkeypatch.setattr(runners, "_gene_set_for_reference", lambda ref, base: UNIVERSE)
    report = m.run_verification(str(tmp_path))
    assert report.passed, report.summary()
    assert report.results[-1].name == "provenance_audit"
    written = json.loads(
        (Path(built.root) / "preprocess" / "verification_report.json").read_text()
    )
    assert written["dataset_name"] == "gene_essentiality_goodall2018"


# --------------------------------------------------------------------------- #
# Raw mirror and CLI
# --------------------------------------------------------------------------- #
def _synthetic_raw_files(
    tmp_path: Path,
) -> tuple[tuple[m.RawFile, ...], dict[str, Path]]:
    sources: dict[str, Path] = {}
    files: list[m.RawFile] = []
    for name, payload in (("a.xlsx", b"table a"), ("b.xlsx", b"table b")):
        path = tmp_path / "download" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        sha = hashlib.sha256(payload).hexdigest()
        url = f"https://pmc-oa-opendata.s3.amazonaws.com/PMC1.1/{name}"
        files.append(
            m.RawFile(
                name=name,
                sha256=sha,
                bytes=len(payload),
                description=name,
                retrieval=RetrievalRecord(
                    method=RetrievalMethod.pmc_cloud,
                    source_url=url,
                    retriever="torchcell.literature.retrieve.pmc_cloud_object",
                    params={"key": f"PMC1.1/{name}"},
                    sha256=sha,
                    retrieved_at="2026-10-07",
                ),
            )
        )
        sources[name] = path
    return tuple(files), sources


@pytest.fixture
def two_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[tuple[m.RawFile, ...], dict[str, Path]]:
    files, sources = _synthetic_raw_files(tmp_path)
    monkeypatch.setattr(m, "RAW_FILES", files)
    monkeypatch.setattr(m, "DATA_SHA256", {f.name: f.sha256 for f in files})
    return files, sources


def test_deposit_is_idempotent_and_refuses_a_differing_file(
    tmp_path: Path, two_files: tuple[tuple[m.RawFile, ...], dict[str, Path]]
) -> None:
    files, sources = two_files
    data_root = str(tmp_path / "root")
    root = m.deposit_raw_mirror(sources=sources, data_root=data_root)
    assert root == Path(data_root) / m.RAW_DIR_REL
    m.deposit_raw_mirror(sources=sources, data_root=data_root)
    manifest = m.load_manifest(data_root)
    assert [
        (f.path, f.sha256, f.retrieval.method if f.retrieval else None)
        for f in manifest.files
    ] == [
        ("data/a.xlsx", files[0].sha256, RetrievalMethod.pmc_cloud),
        ("data/b.xlsx", files[1].sha256, RetrievalMethod.pmc_cloud),
    ]
    assert manifest.si_expected == list(m.NOT_MIRRORED)
    assert m.manifest_sha256(manifest, "data/b.xlsx") == files[1].sha256
    with pytest.raises(KeyError, match="data/c.xlsx is not in the raw-mirror manifest"):
        m.manifest_sha256(manifest, "data/c.xlsx")

    (root / "data" / "a.xlsx").write_bytes(b"edited")
    with pytest.raises(RuntimeError, match="exists with a different sha256; refusing"):
        m.deposit_raw_mirror(sources=sources, data_root=data_root)
    (root / "data" / "a.xlsx").write_bytes(b"table a")
    sources["b.xlsx"].write_bytes(b"other")
    with pytest.raises(RuntimeError, match="b.xlsx sha256 mismatch"):
        m.deposit_raw_mirror(sources=sources, data_root=data_root)
    with pytest.raises(KeyError, match="no source given for"):
        m.deposit_raw_mirror(sources={}, data_root=data_root)


def test_download_links_the_mirror_and_checks_the_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    two_files: tuple[tuple[m.RawFile, ...], dict[str, Path]],
) -> None:
    files, sources = two_files
    data_root = tmp_path / "root"
    m.deposit_raw_mirror(sources=sources, data_root=str(data_root))
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    dataset = object.__new__(m.GeneEssentialityGoodall2018Dataset)
    monkeypatch.setattr(
        m.GeneEssentialityGoodall2018Dataset,
        "raw_dir",
        str(tmp_path / "raw"),
        raising=False,
    )
    dataset.download()
    linked = tmp_path / "raw" / "a.xlsx"
    assert linked.is_symlink()
    assert os.readlink(linked) == str(data_root / m.RAW_DIR_REL / "data" / "a.xlsx")

    (data_root / m.RAW_DIR_REL / "data" / "b.xlsx").unlink()
    with pytest.raises(RuntimeError, match="required raw artifact missing"):
        dataset.download()
    manifest_path = data_root / m.RAW_DIR_REL / "manifest.json"
    manifest_path.write_text(
        manifest_path.read_text().replace(files[0].sha256, "0" * 64)
    )
    with pytest.raises(ManifestPinMismatchError, match="data/a.xlsx"):
        dataset.download()


def test_retrieve_runs_the_recorded_retriever_and_verifies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    two_files: tuple[tuple[m.RawFile, ...], dict[str, Path]],
) -> None:
    calls: list[RetrievalRecord] = []

    def fake(record: RetrievalRecord) -> bytes:
        calls.append(record)
        return b"table a" if record.params["key"].endswith("a.xlsx") else b"wrong"

    monkeypatch.setattr(m, "run_retriever", fake)
    out = m.retrieve_raw_files(tmp_path / "fetched", ["a.xlsx"])
    assert out == {"a.xlsx": tmp_path / "fetched" / "a.xlsx"}
    assert [c.params["key"] for c in calls] == ["PMC1.1/a.xlsx"]
    with pytest.raises(RawSha256MismatchError):
        m.retrieve_raw_files(tmp_path / "fetched", ["b.xlsx"])
    assert not (tmp_path / "fetched" / "b.xlsx").exists()


def test_every_consumed_file_is_pinned_once() -> None:
    assert [f.name for f in m.RAW_FILES] == ["si2.xlsx", "si7.xlsx"]
    assert all(
        len(f.sha256) == 64 and f.sha256 == f.retrieval.sha256 for f in m.RAW_FILES
    )
    assert {f.retrieval.method for f in m.RAW_FILES} == {RetrievalMethod.pmc_cloud}
    assert [f.retrieval.params["key"] for f in m.RAW_FILES] == [
        "PMC5821084.1/mbo001183726st1.xlsx",
        "PMC5821084.1/mbo001183726st4.xlsx",
    ]


def test_cli_deposits_from_the_library_or_a_retrieval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    two_files: tuple[tuple[m.RawFile, ...], dict[str, Path]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, sources = two_files
    data_root = tmp_path / "root"
    si = data_root / m.LIBRARY_DIR_REL / "si"
    si.mkdir(parents=True)
    for name, path in sources.items():
        (si / name).write_bytes(path.read_bytes())
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    assert m.main(["deposit"]) == 0
    assert capsys.readouterr().out.strip() == str(data_root / m.RAW_DIR_REL)

    retrieved: list[str] = []

    def fake_retrieve(dest: str) -> dict[str, Path]:
        retrieved.append(dest)
        return dict(sources)

    monkeypatch.setattr(m, "retrieve_raw_files", fake_retrieve)
    assert m.main(["deposit", "--retrieve-into", str(tmp_path / "fetched")]) == 0
    assert retrieved == [str(tmp_path / "fetched")]


def test_cli_build_and_verify(
    built: m.GeneEssentialityGoodall2018Dataset,
    bw25113: EcoliK12BW25113Genome,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import torchcell.verification.runners as runners

    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    assert m.main(["build"]) == 0
    assert capsys.readouterr().out.strip().endswith("len = 7")

    monkeypatch.setattr(m, "SOURCED_VALUES", {})
    monkeypatch.setattr(m, "bacterial_genome", lambda *a, **k: bw25113)
    monkeypatch.setattr(runners, "_gene_set_for_reference", lambda ref, base: UNIVERSE)
    assert m.main(["verify"]) == 0
    assert "gene_essentiality_goodall2018: PASS" in capsys.readouterr().out
    monkeypatch.setattr(runners, "_gene_set_for_reference", lambda ref, base: set())
    assert m.main(["verify"]) == 1


# --------------------------------------------------------------------------- #
# Real data (``--data``): the mirror and the built dev-tree LMDB, never rebuilt
# --------------------------------------------------------------------------- #
def _real_data_root() -> str:
    return os.environ["DATA_ROOT"]


def _built_root() -> str:
    return osp.join(_real_data_root(), m.DATASET_ROOT_REL)


@pytest.mark.data
def test_real_mirror_matches_the_module_pins() -> None:
    from torchcell.data.experiment_dataset import file_sha256

    manifest = m.load_manifest(_real_data_root())
    for raw in m.RAW_FILES:
        assert m.manifest_sha256(manifest, raw.mirror_relpath) == raw.sha256
        path = m.raw_mirror_dir(_real_data_root()) / raw.mirror_relpath
        assert path.stat().st_size == raw.bytes
        assert file_sha256(path) == raw.sha256


@pytest.mark.data
@pytest.mark.parametrize("key", sorted(m.SOURCED_VALUES))
def test_real_sourced_values_are_verbatim(key: str) -> None:
    value = m.SOURCED_VALUES[key]
    result = audit_sourced_value(value, Path(_real_data_root()) / "torchcell-library")
    assert result.passed, result.message


@pytest.mark.data
def test_real_tables_follow_the_printed_threshold_and_the_paper_counts() -> None:
    raw = m.raw_mirror_dir(_real_data_root()) / "data"
    tl = m.read_calls(raw / m.TABLE_S1, TL_SPEC)
    lb = m.read_calls(raw / m.TABLE_S4, LB_SPEC)
    assert m.call_counts(tl) == {
        "essential": 358,
        "non_essential": 3793,
        "unclear": 162,
    }
    assert m.call_counts(lb) == {
        "essential": 356,
        "non_essential": 3802,
        "unclear": 155,
    }
    cases: list[tuple[list[m.GeneCall], m.ConditionKey, str]] = [
        (tl, "TL", "ydhW"),
        (lb, "LB", "grxD"),
    ]
    for calls, key, moved in cases:
        check = m.call_rule_check(calls, key, 3.6)
        assert check.disagreements == []
        assert [entry.split(" ")[0] for entry in check.moved_at_exact_log2_12] == [
            moved
        ]


@pytest.mark.data
def test_real_build_counts_and_hand_checked_rows() -> None:
    """8,203 records = 8,626 rows - 106 multi-copy symbols - 317 unclear calls. degS
    (row 3176 of both tables; ratio 9.88 in S1, -5.74 in S4) is the paper's
    conditionally essential example: non-essential as plated, essential after LB.
    """
    from torchcell.verification.runners import load_records

    drops = json.loads(
        Path(_built_root(), "preprocess", "dropped_records.json").read_text()
    )
    assert (drops["source_records"], drops["kept_records"]) == (8626, 8203)
    assert [r["n_records"] for r in drops["rules"]] == [106, 317]
    records = load_records(_built_root())
    assert len(records) == 8203
    tl_env = m.environment("TL").model_dump()
    observed = {
        (
            "TL" if r["experiment"]["environment"] == tl_env else "LB",
            r["experiment"]["genotype"]["perturbations"][0]["systematic_gene_name"],
        ): r["experiment"]["phenotype"]["is_essential"]
        for r in records
    }
    assert {key: observed[key] for key in HAND_CHECKED} == HAND_CHECKED


@pytest.mark.data
def test_real_build_passes_l0_to_l4() -> None:
    report = m.run_verification(_real_data_root())
    assert report.passed, report.summary()


#: Read off ``si2.xlsx`` / ``si7.xlsx`` (openpyxl) and the BW25113 GenBank symbols:
#: degS BW25113_3235, dnaA BW25113_3702, ttcC BW25113_4638 (a pseudogene), ftsK
#: BW25113_0890 (non-essential as plated, unclear and dropped after LB).
HAND_CHECKED: dict[tuple[str, str], bool] = {
    ("TL", "BW25113_3235"): False,
    ("LB", "BW25113_3235"): True,
    ("TL", "BW25113_3702"): True,
    ("LB", "BW25113_3702"): True,
    ("TL", "BW25113_4638"): True,
    ("LB", "BW25113_4638"): True,
    ("TL", "BW25113_0890"): False,
}
