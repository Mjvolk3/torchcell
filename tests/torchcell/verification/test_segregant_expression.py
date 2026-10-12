# tests/torchcell/verification/test_segregant_expression.py
# [[tests.torchcell.verification.test_segregant_expression]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/verification/test_segregant_expression.py
"""The segregant single-cell expression verifier on the synthetic Boocock 2025 store.

The store is the four-record build of ``_boocock2025_release.py`` (pooled segregants S1
and S2, de novo cells LD:TTTA-1 and LD:TTTC-1). Each failure test edits ONE thing (a
stored count, a block level, a panel's MATa parent, the reference base, the expected
count) and asserts that exactly the rule owning it fails.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.torchcell.datasets.scerevisiae._boocock2025_release import (
    _PANELS,
    _build,
    _genome,
)
from torchcell.datasets.scerevisiae import boocock2025 as m
from torchcell.datasets.scerevisiae import nguessan2025
from torchcell.verification import runners
from torchcell.verification.report import (
    Level,
    LevelResult,
    Provenance,
    VerificationReport,
)
from torchcell.verification.segregant_expression import (
    panel_of,
    verify_segregant_expression_streaming,
)
from torchcell.verification.sourced import SourcedValue

_INDEX = {
    "AAA": "GENOMES_ASSEMBLED/AAA_6.re.fa",
    "AAS": "GENOMES_ASSEMBLED/AAS_3.re.fa",
    "ACD": "GENOMES_ASSEMBLED/ACD_7.re.fa",
}
_PROVENANCE = Provenance(source_uri="synthetic", citation_key="synthetic")


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """The built synthetic store, its records and a one-quote library."""
    dataset = _build(tmp_path, monkeypatch)
    library = tmp_path / "library"
    (library / "synthetic").mkdir(parents=True)
    paper = library / "synthetic" / "paper.md"
    paper.write_text("cells were grown at 30 C in minimal YNB")
    quote = SourcedValue(
        value=30.0,
        provenance=Provenance(
            source_uri="paper.md",
            citation_key="synthetic",
            sha256=hashlib.sha256(paper.read_bytes()).hexdigest(),
        ),
        quote="grown at 30 C",
    )
    return {
        "root": dataset.root,
        "records": list(runners.stream_records(dataset.root)),
        "library": library,
        "sourced": [quote],
        "gene_set": set(dataset.gene_set),
    }


def _verify(
    store: dict[str, Any], records: list[dict[str, Any]] | None = None, **overrides: Any
) -> VerificationReport:
    kwargs: dict[str, Any] = dict(
        dataset_name="boocock2025",
        provenance=_PROVENANCE,
        expected_count=4,
        raw_dir=Path(store["root"]) / "raw",
        library_root=store["library"],
        assembly_index=_INDEX,
        genome=_genome(),
        sgd_genes={"YAL001C", "YAL002W", "YBL001C", "YCL001W", "YDL001W"},
        gene_set=store["gene_set"],
        panels=dict(_PANELS),
        sourced=store["sourced"],
        marker_sample=10,
        expected_pooled_segregants=2,
        check_paper_numbers=False,
    )
    kwargs.update(overrides)
    return verify_segregant_expression_streaming(
        records if records is not None else store["records"], **kwargs
    )


def _by_name(report: VerificationReport) -> dict[str, Any]:
    return {r.name: r for r in report.results}


def test_the_synthetic_store_passes_every_level(store: dict[str, Any]) -> None:
    report = _verify(store)
    results = _by_name(report)
    assert report.passed, report.summary()
    assert {r.level for r in report.results} == {
        Level.L0,
        Level.L1,
        Level.L2,
        Level.L3,
        Level.L4,
    }
    assert results["panel_counts"].details["observed"] == {"P": 2, "D": 2}
    assert results["mosaic_round_trip"].details["blocks_per_record_by_panel"] == {
        "P": {"min": 2, "median": 2.5, "mean": 2.5, "max": 3},
        "D": {"min": 3, "median": 3.5, "mean": 3.5, "max": 4},
    }
    # d1 at chrI 100: 0.99 stored as 1.0, the largest grid error of the four cells.
    assert results["mosaic_round_trip"].details["worst_posterior_error"] == (
        pytest.approx(0.02)
    )
    assert results["mat_orientation"].details["observed"] == {
        "P": 0.0,
        "D": pytest.approx(0.6),
    }
    assert results["marker_reference"].details == {
        "n_checked": 7,
        "n_match": 7,
        "ratio": 1.0,
    }
    assert results["gene_containment_sgd"].details["outside_reference"] == []


def test_a_changed_count_fails_value_fidelity(store: dict[str, Any]) -> None:
    records = copy.deepcopy(store["records"])
    target = next(
        r for r in records if r["experiment"]["genotype"]["segregant_id"] == "S1"
    )
    target["experiment"]["phenotype"]["umi_count"] = {"YAL002W": 4}
    results = _by_name(_verify(store, records))
    assert not results["value_fidelity"].passed
    assert results["value_fidelity"].details["failures"] == [
        "('P', 'S1'): umi_count differs from the release"
    ]
    assert not results["reference_pseudobulk"].passed
    assert results["mosaic_round_trip"].passed


def test_the_summed_cells_and_the_marker_gap_are_checked(store: dict[str, Any]) -> None:
    """#1016: a record names the cells it sums and gaps the markers it never counted."""

    def tampered(edit: Any) -> dict[str, Any]:
        records = copy.deepcopy(store["records"])
        target = next(
            r for r in records if r["experiment"]["genotype"]["segregant_id"] == "S1"
        )
        edit(target["experiment"]["phenotype"])
        return _by_name(_verify(store, records))

    results = tampered(
        lambda p: p.update(cell_barcodes=["TTTT-9"] + p["cell_barcodes"][1:])
    )
    assert results["structural"].passed
    assert results["value_fidelity"].details["failures"] == [
        "('P', 'S1'): cell_barcodes differs from the release"
    ]
    results = tampered(lambda p: p.update(provenance_gaps=[]))
    assert not results["structural"].passed
    assert results["structural"].details["n_records"] == 4
    assert len(results["structural"].details["failures"]) == 1


def test_a_level_off_the_grid_fails_the_round_trip(store: dict[str, Any]) -> None:
    records = copy.deepcopy(store["records"])
    target = next(
        r for r in records if r["experiment"]["genotype"]["segregant_id"] == "LD:TTTA-1"
    )
    target["experiment"]["genotype"]["blocks"][1]["posterior"] = 0.8
    results = _by_name(_verify(store, records))
    assert not results["mosaic_round_trip"].passed
    assert results["mosaic_round_trip"].details["failures"] == [
        "LD:TTTA-1: posterior off by 0.1700"
    ]
    assert results["value_fidelity"].passed


def test_a_flipped_parent_fails_the_round_trip(store: dict[str, Any]) -> None:
    records = copy.deepcopy(store["records"])
    target = next(
        r for r in records if r["experiment"]["genotype"]["segregant_id"] == "S2"
    )
    target["experiment"]["genotype"]["blocks"][0]["parent"] = 1
    target["experiment"]["genotype"]["blocks"][1]["parent"] = 2
    results = _by_name(_verify(store, records))
    assert results["mosaic_round_trip"].details["failures"] == ["S2: calls differ"]


def test_the_wrong_mata_parent_fails_the_orientation_rule(
    store: dict[str, Any],
) -> None:
    panels = dict(_PANELS)
    panels["D"] = panels["D"].model_copy(update={"mata_parent": 1})
    results = _by_name(_verify(store, panels=panels))
    assert not results["mat_orientation"].passed
    assert results["mat_orientation"].details["wrong"] == ["D"]


def test_a_reference_base_that_differs_fails_marker_reference(
    store: dict[str, Any],
) -> None:
    genome = _genome()
    genome.fasta_dna = {
        "chrI": SimpleNamespace(seq="N" * 300),
        "chrIII": SimpleNamespace(seq="N" * 150),
    }
    results = _by_name(_verify(store, genome=genome))
    assert not results["marker_reference"].passed
    assert results["marker_reference"].details["ratio"] == 0.0


def test_count_and_missing_quote_are_reported(store: dict[str, Any]) -> None:
    quote = store["sourced"][0].model_copy(update={"quote": "grown at 37 C"})
    results = _by_name(_verify(store, expected_count=5, sourced=[quote]))
    assert results["count"].message == "4 records (expected 5)"
    assert not results["sourced_values"].passed
    assert results["structural"].passed


def test_paper_number_rules_compare_the_release_to_table_s4(
    store: dict[str, Any],
) -> None:
    results = _by_name(_verify(store, check_paper_numbers=True))
    assert results["table_s4"].passed
    assert results["table_s4"].details["observed"] == {
        "P": (3, 3, 4.0),
        "D": (2, 2, 4.5),
    }
    pooled = results["pooled_panel"]
    assert not pooled.passed
    assert (pooled.details["n_segregants"], pooled.details["over_ten"]) == (2, 0)
    assert pooled.details["n_shared_markers"] == 2


def test_panel_of_reads_the_library_prefix() -> None:
    library_panel = {"LD": "D"}
    assert panel_of({"segregant_id": "LD:TTTA-1"}, library_panel) == "D"
    assert panel_of({"segregant_id": "A01_01"}, library_panel) == "Ap"


def test_the_registry_entry_names_the_dev_tree_root_and_count() -> None:
    spec = runners.SEGREGANT_EXPRESSION_DATASETS["boocock2025"]
    assert spec["root"] == "data/torchcell/boocock2025"
    assert spec["expected_count"] == m.EXPECTED_RECORDS == 79516
    assert spec["provenance"].sha256 == m.PAPER_MD_SHA256


def test_run_segregant_expression_writes_the_report_verify_build_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Each entry is gated by the ``verify_build`` of the module it names.

    N'Guessan 2025's held-out spec is added to the registry here so the dispatch is
    exercised on two modules.
    """
    monkeypatch.setitem(
        runners.SEGREGANT_EXPRESSION_DATASETS,
        "nguessan2025",
        nguessan2025.VERIFICATION_SPEC,
    )
    seen: list[tuple[str, str, str]] = []
    reports: dict[str, VerificationReport] = {}
    for name, module in (("boocock2025", m), ("nguessan2025", nguessan2025)):
        report = VerificationReport(dataset_name=name, provenance=_PROVENANCE)
        report.add(
            LevelResult(level=Level.L0, name="structural", passed=True, message="ok")
        )
        reports[name] = report

        def fake_verify_build(
            dataset_root: str, *, data_root: str, name: str = name
        ) -> VerificationReport:
            seen.append((name, dataset_root, data_root))
            return reports[name]

        monkeypatch.setattr(module, "verify_build", fake_verify_build)
    assert runners.run_segregant_expression(str(tmp_path)) is True
    roots = {n: tmp_path / "data" / "torchcell" / n for n in reports}
    assert seen == [(n, str(roots[n]), str(tmp_path)) for n in reports]
    for name, root in roots.items():
        written = root / "preprocess" / "verification_report.json"
        assert (
            VerificationReport.model_validate_json(written.read_text())
            == (reports[name])
        )
    out = capsys.readouterr().out
    assert "boocock2025: PASS" in out
    assert "nguessan2025: PASS" in out


def test_run_segregant_expression_fails_when_one_store_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing N'Guessan report makes the family fail although Boocock passes."""
    monkeypatch.setitem(
        runners.SEGREGANT_EXPRESSION_DATASETS,
        "nguessan2025",
        nguessan2025.VERIFICATION_SPEC,
    )
    for name, module, passed in (
        ("boocock2025", m, True),
        ("nguessan2025", nguessan2025, False),
    ):
        report = VerificationReport(dataset_name=name, provenance=_PROVENANCE)
        report.add(
            LevelResult(
                level=Level.L0, name="structural", passed=passed, message="synthetic"
            )
        )
        monkeypatch.setattr(
            module,
            "verify_build",
            lambda dataset_root, *, data_root, report=report: report,
        )
    assert runners.run_segregant_expression(str(tmp_path)) is False


def test_every_entry_names_its_own_loader_module() -> None:
    """Boocock is registered; N'Guessan's held-out spec already names its module."""
    assert {
        name: spec["verify_build_module"]
        for name, spec in runners.SEGREGANT_EXPRESSION_DATASETS.items()
    } == {"boocock2025": "torchcell.datasets.scerevisiae.boocock2025"}
    assert nguessan2025.VERIFICATION_SPEC["verify_build_module"] == (
        nguessan2025.__name__
    )
