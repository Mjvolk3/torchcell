# tests/torchcell/datasets/ecoli/test_wang2024.py
# [[tests.torchcell.datasets.ecoli.test_wang2024]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/ecoli/test_wang2024.py
"""The Wang 2024 rifampicin Tn-seq loader (``torchcell.datasets.ecoli.wang2024``).

Synthetic tests (run everywhere) write a six-sheet Table S2 in ``tmp_path``. The
hermetic build uses the real ``EcoliK12MG1655Genome`` over the synthetic MG1655 assembly
of ``tests/torchcell/sequence/genome/_bacterial_fixtures.py`` with the network refused,
and ``verify_raw_files`` replaced with a presence check (synthetic bytes cannot carry the
real pin; the pin is asserted by the data-gated tests).

The five synthetic genes, identical input columns on every sheet:

    #Orf    annotation            input     post-treatment   fate
    b0001   thrL, current         reads     reads            kept, log2FC per sheet
    b0002   thrA, current         0         0                dropped: no reads
    b0004   yaaP, pseudogene      reads     reads            kept
    b0005   proB, current         0         reads            kept: reads after
    b0099   on no locus           reads     reads            dropped: identifier

Four of the five names are locus tags (0.8 < 0.98), so the build fixture lowers
``MIN_RESOLVED_FRACTION`` and a separate test shows the default stops the build.
3 kept genes x 6 sheets = 18 records.

Data-gated tests (``--data``) read the real raw mirror and the built dev-tree LMDB under
``$DATA_ROOT`` (they never build it).
"""

from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import openpyxl
import pandas as pd
import pytest

import torchcell.datasets.ecoli.wang2024 as m
from tests.torchcell.sequence.genome._bacterial_fixtures import (
    MG1655_GAF,
    MG1655_LOCI,
    forbid_network,
    serve_tier,
    write_assembly,
)
from torchcell.datamodels.media import LB
from torchcell.datamodels.schema import (
    AssayType,
    AssemblyReferenceGenome,
    BacterialEnvironmentResponseExperiment,
    BacterialEnvironmentResponseExperimentReference,
    ConcentrationUnit,
    DoseBasis,
    MeasurementType,
    SampleUnit,
    TransposonInsertionPerturbation,
)
from torchcell.datasets.bacteria_common import LocusTagResolutionError
from torchcell.datasets.dataset_registry import dataset_registry
from torchcell.literature.manifest import RetrievalMethod, RetrievalRecord
from torchcell.sequence.genome.ecoli.k12 import MG1655_ASSEMBLY, EcoliK12MG1655Genome
from torchcell.verification.environment_response import _condition_signature
from torchcell.verification.sourced import ProvenanceGapReason, audit_sourced_value

REFERENCE = AssemblyReferenceGenome(
    species="Escherichia coli",
    strain="MG1655",
    assembly_set="ecoli_K12_MG1655_ASM584v2",
    assembly_accession="GCA_000005845.2",
)
SHEETS = [spec.sheet for spec in m.CONDITIONS]
#: ``(#Orf, Name, Sites, Mean Ctrl, Sum Ctrl, Mean Exp, Sum Exp)``; log2FC per sheet below.
GENES: tuple[tuple[str, str, int, float, float, float, float], ...] = (
    ("b0001", "thrL", 58, 219.4, 25454.9, 177.7, 20617.55),
    ("b0002", "thrA", 12, 0.0, 0.0, 0.0, 0.0),
    ("b0004", "yaaP", 30, 10.0, 600.0, 20.0, 1200.0),
    ("b0005", "proB", 20, 0.0, 0.0, 4.0, 160.0),
    ("b0099", "ghostG", 40, 50.0, 4000.0, 25.0, 2000.0),
)
KEPT = ("b0001", "b0004", "b0005")
SYNTHETIC_RECORDS = len(KEPT) * len(m.CONDITIONS)


def _log2fc(orf: str, sheet_index: int) -> float:
    """The synthetic log2FC of a gene on one sheet (0 for the no-read gene)."""
    if orf == "b0002":
        return 0.0
    return round(-1.5 + 0.5 * sheet_index + (0.25 if orf == "b0004" else 0.0), 2)


def _row(gene: tuple[Any, ...], sheet_index: int) -> list[Any]:
    orf, name, sites, mean_ctrl, sum_ctrl, mean_exp, sum_exp = gene
    return [
        orf,
        name,
        sites,
        mean_ctrl,
        mean_exp,
        _log2fc(orf, sheet_index),
        sum_ctrl,
        sum_exp,
        round(mean_exp - mean_ctrl, 1),
        0.5,
        1.0,
    ]


def _write_table(
    path: Path,
    *,
    sheets: Sequence[str] | None = None,
    preamble: Sequence[Any] | None = None,
    header: Sequence[str] | None = None,
    rows: Mapping[str, list[list[Any]]] | None = None,
) -> None:
    """A synthetic Table S2; every keyword overrides one released feature."""
    book = openpyxl.Workbook()
    first = book.active
    assert first is not None
    book.remove(first)
    for index, sheet_name in enumerate(sheets or SHEETS):
        sheet = book.create_sheet(sheet_name)
        if index == 0:
            for line in m.FIRST_SHEET_PREAMBLE if preamble is None else preamble:
                sheet.append([line])
        sheet.append(list(m.HEADER if header is None else header))
        data = (
            rows[sheet_name]
            if rows is not None and sheet_name in rows
            else [_row(gene, index) for gene in GENES]
        )
        for row in data:
            sheet.append(row)
    book.save(path)


def _frames(path: Path) -> dict[str, pd.DataFrame]:
    return m.read_table_s2(path)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def mg1655(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EcoliK12MG1655Genome:
    """The real MG1655 class over the synthetic assembly."""
    files = write_assembly(tmp_path / "tier", MG1655_ASSEMBLY, MG1655_LOCI, MG1655_GAF)
    forbid_network(monkeypatch)
    serve_tier(monkeypatch, files)
    root = tmp_path / "mg1655"
    root.mkdir()
    return EcoliK12MG1655Genome(genome_root=str(root), overwrite=False)


@pytest.fixture
def table(tmp_path: Path) -> Path:
    path = tmp_path / "s2" / m.DATA_FILE
    path.parent.mkdir(parents=True)
    _write_table(path)
    return path


@pytest.fixture
def presence_only_pins(monkeypatch: pytest.MonkeyPatch) -> list[Mapping[str, str]]:
    """Replace the build-time byte check with a presence check that records the pins."""
    calls: list[Mapping[str, str]] = []

    def record(raw: str, pins: Mapping[str, str]) -> None:
        missing = [f for f in pins if not osp.exists(osp.join(raw, f))]
        if missing:
            raise FileNotFoundError(f"pinned raw files absent: {missing}")
        calls.append(dict(pins))

    monkeypatch.setattr(m, "verify_raw_files", record)
    return calls


@pytest.fixture
def synthetic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    presence_only_pins: list[Mapping[str, str]],
) -> Path:
    """A dataset root whose ``raw/`` holds the synthetic Table S2."""
    root = tmp_path / m.DATASET_ROOT_REL
    (root / "raw").mkdir(parents=True)
    _write_table(root / "raw" / m.DATA_FILE)
    monkeypatch.setattr(m, "reference_genome", lambda *a, **k: REFERENCE)
    return root


@pytest.fixture
def built(
    synthetic: Path, monkeypatch: pytest.MonkeyPatch, mg1655: EcoliK12MG1655Genome
) -> m.EnvChemgenWang2024Dataset:
    monkeypatch.setattr(m, "MIN_RESOLVED_FRACTION", 0.75)
    return m.EnvChemgenWang2024Dataset(root=str(synthetic), ecoli_genome=mg1655)


# --------------------------------------------------------------------------- #
# The six conditions
# --------------------------------------------------------------------------- #
def test_the_six_conditions_are_the_released_sheets_in_order() -> None:
    assert [(c.sheet, c.dose_mg_per_l, c.hours) for c in m.CONDITIONS] == [
        ("0.25xMIC-1hour", 2.0, 1.0),
        ("0.25xMIC-3hours", 2.0, 3.0),
        ("4xMIC-1hour", 32.0, 1.0),
        ("4xMIC-3hours", 32.0, 3.0),
        ("20xMIC-1hour", 160.0, 1.0),
        ("20xMIC-3hours", 160.0, 3.0),
    ]
    assert [c.screen_id for c in m.CONDITIONS] == SHEETS
    assert m.CONDITIONS_BY_SHEET["4xMIC-3hours"] is m.CONDITIONS[3]


def test_each_dose_names_its_mic_multiple_and_the_mic() -> None:
    assert [c.mic_multiple for c in m.CONDITIONS] == ["0.25x"] * 2 + ["4x"] * 2 + [
        "20x"
    ] * 2
    assert m.CONDITIONS[0].dose_description == "rifampicin at 0.25x MIC (MIC 8 mg/L)"
    assert m.CONDITIONS[5].dose_description == "rifampicin at 20x MIC (MIC 8 mg/L)"
    # the absolute dose is the multiple of the sourced MIC
    for spec in m.CONDITIONS:
        multiple = float(spec.mic_multiple.removesuffix("x"))
        assert spec.dose_mg_per_l == multiple * m.MIC_MG_PER_L


def test_the_condition_table_refuses_an_unsourced_dose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doses = m.SOURCED_VALUES["doses_mg_per_l"]
    monkeypatch.setitem(
        m.SOURCED_VALUES, "doses_mg_per_l", doses.model_copy(update={"value": (2.0,)})
    )
    with pytest.raises(ValueError, match="is not a sourced"):
        m._conditions()


def test_the_condition_table_refuses_a_sheet_naming_another_multiple(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    multiples = m.SOURCED_VALUES["mic_multiples"]
    monkeypatch.setitem(
        m.SOURCED_VALUES,
        "mic_multiples",
        multiples.model_copy(update={"value": {2.0: "1x", 32.0: "4x", 160.0: "20x"}}),
    )
    with pytest.raises(ValueError, match="does not name 1x MIC"):
        m._conditions()


def test_rifampicin_is_the_absolute_dose_set_by_the_mic() -> None:
    drug = m.rifampicin(m.CONDITIONS[2])
    assert drug.compound.name == "rifampicin"
    assert drug.concentration.value == 32.0
    assert drug.concentration.unit is ConcentrationUnit.ug_per_ml
    assert drug.concentration.basis is DoseBasis.MIC
    assert drug.description == "rifampicin at 4x MIC (MIC 8 mg/L)"


def test_each_environment_is_lb_with_one_drug_for_the_exposure_time() -> None:
    for spec in m.CONDITIONS:
        env = m.environment(spec)
        assert env.media == LB
        assert env.temperature is not None and env.temperature.value == 37.0
        assert env.aerobicity == "aerobic"
        assert env.duration_hours == spec.hours
        assert [p.concentration.value for p in env.perturbations] == [  # type: ignore[union-attr]
            spec.dose_mg_per_l
        ]


def test_the_six_conditions_have_distinct_signatures() -> None:
    signatures = {
        json.dumps(
            _condition_signature(
                m.build_experiment(
                    "ds", "b0001", "thrL", 0.0, spec, m.environment(spec)
                ).model_dump(mode="json")
            ),
            sort_keys=True,
            default=str,
        )
        for spec in m.CONDITIONS
    }
    assert len(signatures) == 6


# --------------------------------------------------------------------------- #
# Genotype and phenotype
# --------------------------------------------------------------------------- #
def test_the_genotype_is_one_gene_level_tn5_insertion() -> None:
    genotype = m.insertion_genotype("b0001", "thrL")
    (pert,) = genotype.perturbations
    assert isinstance(pert, TransposonInsertionPerturbation)
    assert (pert.systematic_gene_name, pert.perturbed_gene_name) == ("b0001", "thrL")
    assert pert.transposon == "Tn5"
    assert (pert.barcode, pert.insertion_position, pert.insertion_strand) == (
        None,
        None,
        None,
    )
    assert pert.gene_namespace == m.MG1655_NAMESPACE


def test_the_three_absent_perturbation_fields_are_typed_gaps() -> None:
    assert [gap.field for gap in m.PERTURBATION_FIELD_GAPS] == [
        "barcode",
        "insertion_position",
        "insertion_strand",
    ]
    assert {gap.reason for gap in m.PERTURBATION_FIELD_GAPS} == {
        ProvenanceGapReason.not_reported_by_primary
    }


def test_the_phenotype_is_a_log2_ratio_over_two_replicates() -> None:
    phenotype = m.phenotype(-2.22, m.CONDITIONS[1])
    assert phenotype.measurement_type is MeasurementType.log2_ratio
    assert phenotype.assay_type is AssayType.other
    assert phenotype.environment_response == -2.22
    assert (phenotype.n_samples, phenotype.sample_unit) == (
        2,
        SampleUnit.biological_replicate,
    )
    assert phenotype.screen_id == "0.25xMIC-3hours"
    assert phenotype.units == m.UNITS
    assert [gap.field for gap in phenotype.provenance_gaps] == [
        "environment_response_uncertainty",
        "environment_response_se",
    ]
    assert phenotype.environment_response_se is None


def test_the_reference_phenotype_scores_zero_in_the_same_condition() -> None:
    reference = m.reference_phenotype(m.CONDITIONS[4])
    assert reference.environment_response == 0.0
    assert reference.screen_id == "20xMIC-1hour"
    assert reference.units == m.UNITS_REFERENCE


def test_build_experiment_and_reference_validate_as_a_pair() -> None:
    spec = m.CONDITIONS[0]
    env = m.environment(spec)
    experiment = m.build_experiment("ds", "b0001", "thrL", 1.5, spec, env)
    reference = m.build_reference("ds", REFERENCE, spec, env)
    assert isinstance(experiment, BacterialEnvironmentResponseExperiment)
    assert isinstance(reference, BacterialEnvironmentResponseExperimentReference)
    assert experiment.phenotype.environment_response == 1.5
    assert reference.environment_reference == experiment.environment


# --------------------------------------------------------------------------- #
# Reading Table S2
# --------------------------------------------------------------------------- #
def test_read_table_s2_reads_every_sheet(table: Path) -> None:
    frames = _frames(table)
    assert list(frames) == SHEETS
    first = frames[SHEETS[0]]
    assert first.shape == (5, 11)
    assert first["#Orf"].tolist() == [g[0] for g in GENES]
    assert first.loc[0, "log2FC"] == -1.5
    assert frames[SHEETS[5]].loc[0, "log2FC"] == 1.0


def test_numeric_refuses_text_booleans_and_infinities() -> None:
    assert m._numeric(3, "x") == 3.0
    for value in ("NA", True, float("inf")):
        with pytest.raises(m.SheetFormatError):
            m._numeric(value, "x")


def test_read_table_s2_refuses_a_changed_sheet_list(tmp_path: Path) -> None:
    path = tmp_path / "t.xlsx"
    _write_table(path, sheets=SHEETS[:5])
    with pytest.raises(m.SheetFormatError, match="sheets"):
        m.read_table_s2(path)


def test_read_table_s2_refuses_a_changed_preamble(tmp_path: Path) -> None:
    path = tmp_path / "t.xlsx"
    _write_table(path, preamble=["Table S2", "a", "b", "c", None])
    with pytest.raises(m.SheetFormatError, match="preamble"):
        m.read_table_s2(path)


def test_read_table_s2_refuses_a_changed_header(tmp_path: Path) -> None:
    path = tmp_path / "t.xlsx"
    _write_table(path, header=[*m.HEADER[:-1], "q-value"])
    with pytest.raises(m.SheetFormatError, match="header"):
        m.read_table_s2(path)


def test_read_table_s2_refuses_an_id_that_is_not_a_b_number(tmp_path: Path) -> None:
    path = tmp_path / "t.xlsx"
    rows = [_row(gene, 1) for gene in GENES]
    rows[0][0] = "thrL"
    _write_table(path, rows={SHEETS[1]: rows})
    with pytest.raises(m.SheetFormatError, match="is not a b-number"):
        m.read_table_s2(path)


def test_read_table_s2_refuses_a_text_value_cell(tmp_path: Path) -> None:
    path = tmp_path / "t.xlsx"
    rows = [_row(gene, 2) for gene in GENES]
    rows[3][9] = "n/a"
    _write_table(path, rows={SHEETS[2]: rows})
    with pytest.raises(m.SheetFormatError, match="p-value"):
        m.read_table_s2(path)


def test_check_sheets_align_returns_the_shared_ids(table: Path) -> None:
    assert m.check_sheets_align(_frames(table)) == [g[0] for g in GENES]


def test_check_sheets_align_refuses_a_different_input_column(tmp_path: Path) -> None:
    path = tmp_path / "t.xlsx"
    rows = [_row(gene, 3) for gene in GENES]
    rows[0][3] = 1.0
    _write_table(path, rows={SHEETS[3]: rows})
    with pytest.raises(m.SheetFormatError, match="'Mean Ctrl' differs"):
        m.check_sheets_align(m.read_table_s2(path))


def test_check_sheets_align_refuses_a_repeated_id(tmp_path: Path) -> None:
    path = tmp_path / "t.xlsx"
    rows = {
        sheet: [_row(gene, i) for gene in (*GENES, GENES[0])]
        for i, sheet in enumerate(SHEETS)
    }
    _write_table(path, rows=rows)
    with pytest.raises(m.SheetFormatError, match="repeats"):
        m.check_sheets_align(m.read_table_s2(path))


# --------------------------------------------------------------------------- #
# Identifiers and the retention ledger
# --------------------------------------------------------------------------- #
def test_resolve_b_numbers_keeps_only_locus_tags(
    mg1655: EcoliK12MG1655Genome, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(m, "MIN_RESOLVED_FRACTION", 0.75)
    kept, ledger = m.resolve_b_numbers(mg1655, [g[0] for g in GENES], label="t")
    assert kept == {"b0001": "thrL", "b0002": "thrA", "b0004": "yaaP", "b0005": "proB"}
    assert list(ledger.not_a_locus_tag) == ["b0099"]
    assert ledger.n_locus_tags == 4
    assert ledger.min_resolved_fraction == 0.75


def test_resolve_b_numbers_stops_below_the_default_threshold(
    mg1655: EcoliK12MG1655Genome,
) -> None:
    with pytest.raises(LocusTagResolutionError):
        m.resolve_b_numbers(mg1655, [g[0] for g in GENES], label="t")


def test_canonical_symbol_falls_back_to_the_tag_without_a_symbol(
    mg1655: EcoliK12MG1655Genome, monkeypatch: pytest.MonkeyPatch
) -> None:
    locus = mg1655.genbank.loci["b0001"]
    monkeypatch.setitem(
        mg1655.genbank.loci, "b0001", locus.model_copy(update={"symbol": None})
    )
    assert m.canonical_symbol(mg1655, "b0001") == "b0001"
    assert m.canonical_symbol(mg1655, "b0005") == "proB"


def test_build_drop_log_accounts_for_every_cell(table: Path) -> None:
    frames = _frames(table)
    b_numbers = m.check_sheets_align(frames)
    kept = {"b0001": "thrL", "b0002": "thrA", "b0004": "yaaP", "b0005": "proB"}
    log = m.build_drop_log("ds", frames, b_numbers, kept)
    assert (log.source_records, log.kept_records, log.dropped_records) == (30, 18, 12)
    assert {r.rule: r.n_records for r in log.rules} == {
        m.RULE_NO_READS: 6,
        m.RULE_IDENTIFIER: 6,
    }
    assert log.rules[1].items == ["b0099"]
    assert log.rules[0].items[0] == "b0002 0.25xMIC-1hour"
    first = log.conditions[0]
    assert (first.no_reads, first.dropped_identifier, first.no_input_reads_kept) == (
        1,
        1,
        1,
    )
    # sheet 0: thrL -1.5, yaaP -1.25, proB -1.5; sheet 3: 0.0, 0.25, 0.0
    assert (first.n_positive, first.n_negative, first.n_zero) == (0, 3, 0)
    assert (log.conditions[3].n_positive, log.conditions[3].n_zero) == (1, 2)


def test_build_drop_log_refuses_rules_that_miss_a_cell(
    table: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = _frames(table)
    rule = m.DropRule
    monkeypatch.setattr(m, "DropRule", lambda **k: rule(**{**k, "n_records": 0}))
    with pytest.raises(RuntimeError, match="do not account"):
        m.build_drop_log("ds", frames, m.check_sheets_align(frames), {"b0001": "x"})


def test_stored_cells_skip_dropped_genes_and_no_read_cells(table: Path) -> None:
    frames = _frames(table)
    kept = {"b0001": "thrL", "b0002": "thrA", "b0004": "yaaP", "b0005": "proB"}
    cells = list(m.stored_cells(frames, kept))
    assert len(cells) == SYNTHETIC_RECORDS
    assert cells[0] == ("b0001", "thrL", m.CONDITIONS[0], -1.5)
    assert {c[0] for c in cells} == set(KEPT)
    assert [c[2].sheet for c in cells[:3]] == [SHEETS[0]] * 3


# --------------------------------------------------------------------------- #
# The hermetic build
# --------------------------------------------------------------------------- #
def test_the_build_stores_one_record_per_kept_cell(
    built: m.EnvChemgenWang2024Dataset, presence_only_pins: list[Mapping[str, str]]
) -> None:
    assert len(built) == SYNTHETIC_RECORDS
    assert presence_only_pins == [m.DATA_SHA256]


def test_the_built_records_carry_the_released_values(
    built: m.EnvChemgenWang2024Dataset,
) -> None:
    records = [built[i] for i in range(len(built))]
    first = records[0]["experiment"]
    assert first["genotype"]["perturbations"][0]["systematic_gene_name"] == "b0001"
    assert first["phenotype"]["environment_response"] == -1.5
    assert first["phenotype"]["screen_id"] == "0.25xMIC-1hour"
    assert first["environment"]["duration_hours"] == 1.0
    values = [r["experiment"]["phenotype"]["environment_response"] for r in records]
    assert values[-3:] == [1.0, 1.25, 1.0]
    references = {r["reference"]["phenotype_reference"]["screen_id"] for r in records}
    assert references == set(SHEETS)
    assert {
        r["reference"]["phenotype_reference"]["environment_response"] for r in records
    } == {0.0}


def test_the_build_writes_every_ledger(built: m.EnvChemgenWang2024Dataset) -> None:
    out = Path(built.preprocess_dir)
    log = json.loads((out / "dropped_records.json").read_text())
    assert log["kept_records"] == SYNTHETIC_RECORDS
    identifiers = json.loads((out / "identifier_reconciliation.json").read_text())
    assert list(identifiers["not_a_locus_tag"]) == ["b0099"]
    gaps = json.loads((out / "perturbation_field_gaps.json").read_text())
    assert [gap["field"] for gap in gaps] == [
        "barcode",
        "insertion_position",
        "insertion_strand",
    ]


def test_the_build_refuses_a_genome_of_another_strain(
    synthetic: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Other:
        ASSEMBLY_SET = "ecoli_K12_BW25113_ASM75055v1"

    with pytest.raises(ValueError, match="needs the ecoli_K12_MG1655_ASM584v2 genome"):
        m.EnvChemgenWang2024Dataset(
            root=str(synthetic),
            ecoli_genome=Other(),  # type: ignore[arg-type]
        )


def test_the_dataset_is_registered_and_declares_its_classes() -> None:
    assert dataset_registry["EnvChemgenWang2024Dataset"] is m.EnvChemgenWang2024Dataset
    dataset = m.EnvChemgenWang2024Dataset.__new__(m.EnvChemgenWang2024Dataset)
    assert dataset.experiment_class is BacterialEnvironmentResponseExperiment
    assert dataset.reference_class is BacterialEnvironmentResponseExperimentReference
    assert dataset.raw_file_names == [m.DATA_FILE]
    frame = pd.DataFrame({"a": [1]})
    assert dataset.preprocess_raw(frame) is frame
    with pytest.raises(NotImplementedError):
        dataset.create_experiment()


def test_verify_build_runs_the_gate_on_the_built_store(
    built: m.EnvChemgenWang2024Dataset,
    mg1655: EcoliK12MG1655Genome,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from torchcell.verification.report import Level, LevelResult

    audited: list[tuple[str, Path]] = []

    def audit(value: Any, mirror: Path) -> LevelResult:
        audited.append((value.quote, mirror))
        return LevelResult(level=Level.L0, name="sourced", passed=True, message="")

    monkeypatch.setattr(m, "audit_sourced_value", audit)
    report = m.verify_build(
        built.root,
        genome=mg1655,
        data_root=str(tmp_path),
        expected_count=SYNTHETIC_RECORDS,
    )
    rows = {result.name: result for result in report.results}
    assert rows["pair_uniqueness"].passed
    assert len(audited) == len(m.SOURCED_VALUES)
    assert {mirror for _, mirror in audited} == {tmp_path / "torchcell-raw"}
    assert osp.exists(osp.join(built.root, "preprocess", "verification_report.json"))


# --------------------------------------------------------------------------- #
# The raw mirror
# --------------------------------------------------------------------------- #
def test_every_retrieved_file_records_a_re_runnable_pmc_retrieval() -> None:
    assert [raw.name for raw in m.RETRIEVED_FILES] == [
        m.DATA_FILE,
        m.PAPER_PDF_FILE,
        m.PAPER_TEXT_FILE,
    ]
    for raw in m.RETRIEVED_FILES:
        assert raw.retrieval.method is RetrievalMethod.pmc_cloud
        assert raw.retrieval.params == {"key": f"PMC10782999.1/{raw.name}"}
        assert raw.retrieval.sha256 == raw.sha256
        assert raw.source_url == (
            f"https://pmc-oa-opendata.s3.amazonaws.com/PMC10782999.1/{raw.name}"
        )
    assert m.RAW_FILES[-1].derived and m.RAW_FILES[-1].relpath == m.LEGENDS_REL


def test_extract_sheet_legends_renders_the_title_and_legend_rows(table: Path) -> None:
    text = m.extract_sheet_legends(table)
    assert text.splitlines() == [
        "# sheet: 0.25xMIC-1hour",
        *[str(line) for line in m.FIRST_SHEET_PREAMBLE[:4]],
    ]
    record = m.legends_processing("abc")
    assert record.input_sha256 == ["abc"]
    assert record.params == {"sheets": ["0.25xMIC-1hour"], "rows": [1, 2, 3, 4]}


def _patched_raw_files(
    monkeypatch: pytest.MonkeyPatch, table: Path, tmp_path: Path
) -> dict[str, str | Path]:
    """Sources with synthetic bytes, and RAW_FILES re-pinned to those bytes."""
    sources: dict[str, str | Path] = {m.DATA_FILE: table}
    for name in (m.PAPER_PDF_FILE, m.PAPER_TEXT_FILE):
        path = tmp_path / "src" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        sources[name] = path
    legend = m.extract_sheet_legends(table)
    pins = {name: m._sha256(path) for name, path in sources.items()}
    pins[m.LEGENDS_FILE] = hashlib.sha256(legend.encode()).hexdigest()
    files = tuple(
        raw.model_copy(update={"sha256": pins[raw.name]}) for raw in m.RAW_FILES
    )
    monkeypatch.setattr(m, "RAW_FILES", files)
    monkeypatch.setattr(m, "RETRIEVED_FILES", tuple(f for f in files if not f.derived))
    return sources


def test_deposit_raw_mirror_writes_the_manifest(
    tmp_path: Path, table: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _patched_raw_files(monkeypatch, table, tmp_path)
    data_root = str(tmp_path / "root")
    root = m.deposit_raw_mirror(sources=sources, data_root=data_root)
    manifest = m.load_manifest(data_root)
    assert manifest.citation_key == m.CITATION_KEY
    assert [record.path for record in manifest.files] == [
        raw.relpath for raw in m.RAW_FILES
    ]
    legends = manifest.files[-1]
    assert legends.processing is not None and legends.retrieval is None
    assert legends.processing.input_sha256 == [m._sha256(table)]
    assert (root / m.LEGENDS_REL).read_text() == m.extract_sheet_legends(table)
    assert list(manifest.si_expected) == list(m.NOT_MIRRORED)
    # idempotent: a second deposit over the same bytes succeeds
    m.deposit_raw_mirror(sources=sources, data_root=data_root)
    assert m.manifest_sha256(manifest, f"data/{m.DATA_FILE}") == m._sha256(table)


def test_deposit_raw_mirror_refuses_a_changed_mirror_file(
    tmp_path: Path, table: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _patched_raw_files(monkeypatch, table, tmp_path)
    data_root = str(tmp_path / "root")
    root = m.deposit_raw_mirror(sources=sources, data_root=data_root)
    (root / "paper" / m.PAPER_PDF_FILE).write_bytes(b"edited")
    with pytest.raises(RuntimeError, match="exists with a different sha256"):
        m.deposit_raw_mirror(sources=sources, data_root=data_root)
    (root / "paper" / m.PAPER_PDF_FILE).write_bytes(m.PAPER_PDF_FILE.encode())
    (root / m.LEGENDS_REL).write_text("edited")
    with pytest.raises(RuntimeError, match="exists with a different sha256"):
        m.deposit_raw_mirror(sources=sources, data_root=data_root)


def test_deposit_raw_mirror_refuses_a_missing_source(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="no source given"):
        m.deposit_raw_mirror(sources={}, data_root=str(tmp_path))


def test_deposit_raw_mirror_refuses_a_legend_text_that_is_not_the_pinned_one(
    tmp_path: Path, table: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _patched_raw_files(monkeypatch, table, tmp_path)
    files = tuple(
        raw.model_copy(update={"sha256": "0" * 64}) if raw.derived else raw
        for raw in m.RAW_FILES
    )
    monkeypatch.setattr(m, "RAW_FILES", files)
    with pytest.raises(RuntimeError, match="rendered legends sha256"):
        m.deposit_raw_mirror(sources=sources, data_root=str(tmp_path / "r"))


def test_deposit_raw_mirror_refuses_a_retrieved_file_hash_mismatch(
    tmp_path: Path, table: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _patched_raw_files(monkeypatch, table, tmp_path)
    Path(sources[m.PAPER_TEXT_FILE]).write_bytes(b"other")
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        m.deposit_raw_mirror(sources=sources, data_root=str(tmp_path / "root"))


def test_manifest_sha256_refuses_an_unknown_path() -> None:
    from torchcell.literature.manifest import Manifest

    manifest = Manifest(citation_key=m.CITATION_KEY, doi=m.PAPER_DOI, title="t")
    with pytest.raises(KeyError, match="not in the raw-mirror manifest"):
        m.manifest_sha256(manifest, f"data/{m.DATA_FILE}")


def test_retrieve_raw_files_runs_each_recorded_retriever(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    served: list[str] = []

    def fake(record: RetrievalRecord) -> bytes:
        served.append(record.params["key"])
        return b"bytes"

    monkeypatch.setattr(m, "run_retriever", fake)
    monkeypatch.setattr(m, "write_verified", lambda data, path, sha, url: None)
    out = m.retrieve_raw_files(tmp_path)
    assert list(out) == [m.DATA_FILE, m.PAPER_PDF_FILE, m.PAPER_TEXT_FILE]
    assert served == [f"PMC10782999.1/{name}" for name in out]


def test_download_links_the_pinned_mirror_file(
    tmp_path: Path, table: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _patched_raw_files(monkeypatch, table, tmp_path)
    data_root = tmp_path / "root"
    m.deposit_raw_mirror(sources=sources, data_root=str(data_root))
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.setattr(m, "RAW_FILES_BY_NAME", {f.name: f for f in m.RAW_FILES})
    dataset = m.EnvChemgenWang2024Dataset.__new__(m.EnvChemgenWang2024Dataset)
    dataset.root = str(tmp_path / "ds")
    dataset.download()
    link = Path(dataset.raw_dir) / m.DATA_FILE
    assert link.is_symlink() and link.resolve() == (
        data_root / m.RAW_DIR_REL / "data" / m.DATA_FILE
    )
    (data_root / m.RAW_DIR_REL / "data" / m.DATA_FILE).unlink()
    link.unlink()
    with pytest.raises(RuntimeError, match="missing from mirror"):
        dataset.download()


def test_the_mirror_directory_hangs_off_data_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATA_ROOT", "/tmp/root")
    assert m.raw_mirror_dir() == Path(f"/tmp/root/torchcell-raw/{m.CITATION_KEY}")


def test_main_dispatches_each_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    calls: list[Any] = []
    monkeypatch.setattr(m, "retrieve_raw_files", lambda d: calls.append(("r", d)))
    monkeypatch.setattr(
        m, "deposit_raw_mirror", lambda **k: calls.append(("d", sorted(k["sources"])))
    )

    class FakeDataset:
        def __init__(self, root: str) -> None:
            calls.append(("b", root))

        def __len__(self) -> int:
            return 7

    class FakeReport:
        passed = False

        def summary(self) -> str:
            return "summary"

    monkeypatch.setattr(m, "EnvChemgenWang2024Dataset", FakeDataset)
    monkeypatch.setattr(m, "verify_build", lambda root, data_root: FakeReport())
    assert m.main(["deposit", "--download-dir", str(tmp_path), "--retrieve"]) == 0
    assert m.main(["build"]) == 0
    assert m.main(["verify"]) == 1
    assert calls == [
        ("r", tmp_path),
        ("d", sorted([m.DATA_FILE, m.PAPER_PDF_FILE, m.PAPER_TEXT_FILE])),
        ("b", osp.join(str(tmp_path), m.DATASET_ROOT_REL)),
    ]
    assert "len = 7" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Sourced values
# --------------------------------------------------------------------------- #
def test_every_sourced_value_is_anchored_in_the_raw_mirror() -> None:
    for key, value in m.SOURCED_VALUES.items():
        assert value.provenance.citation_key == m.CITATION_KEY, key
        assert (value.provenance.source_uri, value.provenance.sha256) in (
            (m.PAPER_TEXT_REL, m.PAPER_TEXT_SHA256),
            (m.LEGENDS_REL, m.LEGENDS_SHA256),
        ), key
        assert value.quote.strip(), key


def test_the_derived_constants_come_from_the_sourced_values() -> None:
    assert (m.TEMPERATURE_C, m.AEROBICITY, m.TRANSPOSON) == (37.0, "aerobic", "Tn5")
    assert (m.MIC_MG_PER_L, m.N_REPLICATES) == (8.0, 2)
    assert m.PUBLICATION.pubmed_id == "38054714"
    assert m.PUBLICATION.doi == "10.1128/spectrum.02895-23"


# --------------------------------------------------------------------------- #
# Data-gated: the real mirror and the built dev store
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    return os.environ["DATA_ROOT"]


def _mirror() -> Path:
    root = Path(_data_root()) / m.RAW_DIR_REL
    if not (root / "manifest.json").exists():
        pytest.skip("the Wang 2024 raw mirror is not deposited")
    return root


def _built_root() -> str:
    root = osp.join(_data_root(), m.DATASET_ROOT_REL)
    if not osp.isdir(osp.join(root, "processed", "lmdb")):
        pytest.skip("the Wang 2024 dev-tree LMDB is not built")
    return root


@pytest.mark.data
def test_the_real_mirror_matches_every_pin() -> None:
    root = _mirror()
    manifest = m.load_manifest(_data_root())
    for raw in m.RAW_FILES:
        assert m.manifest_sha256(manifest, raw.relpath) == raw.sha256
        assert m._sha256(root / raw.relpath) == raw.sha256


@pytest.mark.data
def test_every_sourced_value_is_backed_by_its_verbatim_quote() -> None:
    _mirror()
    mirror = Path(_data_root()) / "torchcell-raw"
    for key, value in m.SOURCED_VALUES.items():
        result = audit_sourced_value(value, mirror)
        assert result.passed, f"{key}: {result.message}"


@pytest.mark.data
def test_the_released_table_has_the_measured_shape() -> None:
    frames = m.read_table_s2(_mirror() / "data" / m.DATA_FILE)
    assert [frame.shape for frame in frames.values()] == [(4419, 11)] * 6
    b_numbers = m.check_sheets_align(frames)
    assert b_numbers[:2] == ["b0001", "b0002"]
    first = frames["0.25xMIC-1hour"].set_index("#Orf")
    assert first.at["b0001", "log2FC"] == -0.38 or first.at["b0001", "Name"] == "thrL"
    assert frames["4xMIC-1hour"].set_index("#Orf").at["b0053", "log2FC"] == 4.32


@pytest.mark.data
def test_the_built_store_holds_the_measured_counts() -> None:
    log = json.loads(
        Path(_built_root(), "preprocess", "dropped_records.json").read_text()
    )
    assert (log["source_loci"], log["kept_loci"]) == (4419, 4414)
    assert log["source_records"] == 26514
    assert log["kept_records"] == m.EXPECTED_RECORDS == 24763
    assert {rule["rule"]: rule["n_records"] for rule in log["rules"]} == {
        m.RULE_NO_READS: 1721,
        m.RULE_IDENTIFIER: 30,
    }
    assert [c["no_reads"] for c in log["conditions"]] == [287, 294, 285, 277, 286, 292]
