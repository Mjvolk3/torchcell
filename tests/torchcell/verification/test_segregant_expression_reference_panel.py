# tests/torchcell/verification/test_segregant_expression_reference_panel.py
# [[tests.torchcell.verification.test_segregant_expression_reference_panel]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/verification/test_segregant_expression_reference_panel.py
"""The reference-panel segregant expression verifier on the synthetic N'Guessan release.

The release and its three records are described in
``tests/torchcell/datasets/scerevisiae/_nguessan2025_release.py``. Each failure test
corrupts one thing in the streamed records (or the inputs) and asserts the one rule
that must catch it.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.torchcell.datasets.scerevisiae import _nguessan2025_release as rel
from torchcell.datasets.scerevisiae import nguessan2025 as m
from torchcell.verification import runners
from torchcell.verification import segregant_expression_reference_panel as v
from torchcell.verification.report import Provenance, VerificationReport
from torchcell.verification.sourced import SourcedValue

_INDEX = {"AAA": "GENOMES_ASSEMBLED/AAA_6.re.fa"}


@pytest.fixture
def built(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> m.Nguessan2025Dataset:
    return rel.build(tmp_path, monkeypatch)


def _quote(tmp_path: Path) -> tuple[Path, SourcedValue]:
    library = tmp_path / "library"
    (library / "K").mkdir(parents=True)
    paper = library / "K" / "paper.md"
    paper.write_text("a verbatim sentence.\n")
    return library, SourcedValue(
        value=1,
        provenance=Provenance(
            source_uri="paper.md",
            citation_key="K",
            sha256=hashlib.sha256(paper.read_bytes()).hexdigest(),
        ),
        quote="verbatim sentence",
    )


def _verify(
    dataset: m.Nguessan2025Dataset,
    tmp_path: Path,
    records: list[dict[str, Any]] | None = None,
    **overrides: Any,
) -> VerificationReport:
    library, quote = _quote(tmp_path)
    genome = overrides.pop("genome", rel.genome())
    kwargs: dict[str, Any] = {
        "dataset_name": "nguessan2025",
        "provenance": Provenance(source_uri="synthetic", citation_key="K"),
        "expected_count": 3,
        "raw_dir": dataset.raw_dir,
        "library_root": library,
        "raw_root": library,
        "assembly_index": _INDEX,
        "genome": genome,
        "sgd_genes": set(rel.GENES),
        "gene_set": {"YAL001C", "YCL001W"},
        "gene_midpoints": v.gene_midpoints_from_genome(genome, rel.GENES),
        "sourced": [quote],
        "release_sourced": [quote],
        "check_paper_numbers": False,
    }
    kwargs.update(overrides)
    stream = records if records is not None else runners.stream_records(dataset.root)
    return v.verify_reference_panel_expression_streaming(stream, **kwargs)


def _by_name(report: VerificationReport) -> dict[str, Any]:
    return {r.name: r for r in report.results}


def _records(dataset: m.Nguessan2025Dataset) -> list[dict[str, Any]]:
    return copy.deepcopy(list(runners.stream_records(dataset.root)))


def test_every_rule_passes_on_the_synthetic_build_but_the_powerless_cis_test(
    built: m.Nguessan2025Dataset, tmp_path: Path
) -> None:
    results = _by_name(_verify(built, tmp_path))
    failing = sorted(name for name, r in results.items() if not r.passed)
    # three strains cannot power a correlation test; the rule reports, it never passes
    assert failing == ["cis_concordance"]
    cis = results["cis_concordance"].details
    assert (cis["n_strains"], cis["n_tested_genes"]) == (3, 2)
    assert results["mosaic_round_trip"].details["blocks_per_record"] == {
        "min": 2,
        "median": 3.0,
        "mean": 3.0,
        "max": 4,
    }
    assert results["mosaic_round_trip"].details["worst_posterior_error"] == (
        pytest.approx(0.02)
    )
    mat = results["mat_fixation"].details
    assert (mat["a_specific_umis"], mat["alpha_specific_umis"]) == (5, 0)
    assert mat["mean_rm_at_mat"] == pytest.approx(0.99)
    assert "release_numbers" not in results


def test_value_fidelity_catches_a_changed_count(
    built: m.Nguessan2025Dataset, tmp_path: Path
) -> None:
    records = _records(built)
    phenotype = records[0]["experiment"]["phenotype"]
    phenotype["umi_count"]["YAL002W"] = 4
    phenotype["total_umis"] = 6  # still a valid phenotype, so it reaches L2
    phenotype["cell_total_umis"] = [6]
    results = _by_name(_verify(built, tmp_path, records))
    assert results["structural"].passed
    assert results["value_fidelity"].details["failures"] == [
        "10: umi_count differs from the release",
        "10: total_umis differs from the release",
        "10: cell_total_umis differs from the release",
    ]
    assert not results["reference_pseudobulk"].passed


def test_value_fidelity_catches_a_strain_outside_the_assignment(
    built: m.Nguessan2025Dataset, tmp_path: Path
) -> None:
    records = _records(built)
    records[2]["experiment"]["genotype"]["segregant_id"] = "20"
    results = _by_name(_verify(built, tmp_path, records))
    assert results["value_fidelity"].details["failures"] == [
        "20: not an assigned batch-1 strain"
    ]


def test_value_fidelity_catches_a_stage_partition_without_its_gap(
    built: m.Nguessan2025Dataset, tmp_path: Path
) -> None:
    records = _records(built)
    records[1]["experiment"]["phenotype"]["provenance_gaps"] = []
    report = _verify(built, tmp_path, records)
    results = _by_name(report)
    # the unrecorded gap is refused at L0 (schema) before the L2 comparison runs
    assert not results["structural"].passed
    assert results["structural"].details["failures"][0]["index"] == 1


def test_mosaic_round_trip_catches_a_flipped_block(
    built: m.Nguessan2025Dataset, tmp_path: Path
) -> None:
    records = _records(built)
    records[2]["experiment"]["genotype"]["blocks"][1]["parent"] = 1
    records[2]["experiment"]["genotype"]["blocks"][0]["parent"] = 2
    results = _by_name(_verify(built, tmp_path, records))
    assert results["mosaic_round_trip"].details["failures"] == [
        "12: called parent differs"
    ]


def test_mosaic_round_trip_catches_a_posterior_off_the_grid_bound(
    built: m.Nguessan2025Dataset, tmp_path: Path
) -> None:
    records = _records(built)
    records[0]["experiment"]["genotype"]["blocks"][2]["posterior"] = 0.7
    results = _by_name(_verify(built, tmp_path, records))
    assert results["mosaic_round_trip"].details["failures"] == [
        "10: posterior off by 0.1000"
    ]


def test_mosaic_round_trip_catches_a_misaligned_block(
    built: m.Nguessan2025Dataset, tmp_path: Path
) -> None:
    records = _records(built)
    records[1]["experiment"]["genotype"]["blocks"][0]["n_markers"] = 1
    records[1]["experiment"]["genotype"]["blocks"][0]["end"] = 100
    results = _by_name(_verify(built, tmp_path, records))
    (failure,) = results["mosaic_round_trip"].details["failures"]
    assert failure.startswith("11: block ")


def test_uniqueness_and_count_catch_a_repeated_record(
    built: m.Nguessan2025Dataset, tmp_path: Path
) -> None:
    records = _records(built)
    records.append(copy.deepcopy(records[0]))
    results = _by_name(_verify(built, tmp_path, records))
    assert not results["count"].passed
    assert results["segregant_uniqueness"].details["n_duplicates"] == 1


def test_reference_pseudobulk_catches_a_record_with_another_reference(
    built: m.Nguessan2025Dataset, tmp_path: Path
) -> None:
    records = _records(built)
    # the interned reference is one shared object; give record 2 its own copy
    records[2]["reference"] = copy.deepcopy(records[2]["reference"])
    records[2]["reference"]["phenotype_reference"]["total_umis"] = 18
    results = _by_name(_verify(built, tmp_path, records))
    assert not results["reference_pseudobulk"].passed
    assert (
        results["reference_pseudobulk"].details["records_naming_another_reference"] == 1
    )


def test_structural_catches_a_wrong_experiment_type(
    built: m.Nguessan2025Dataset, tmp_path: Path
) -> None:
    records = _records(built)
    records[0]["experiment"]["experiment_type"] = "segregant_growth"
    results = _by_name(_verify(built, tmp_path, records))
    assert not results["structural"].passed


def test_release_numbers_compare_with_the_paper(
    built: m.Nguessan2025Dataset, tmp_path: Path
) -> None:
    result = _by_name(_verify(built, tmp_path, check_paper_numbers=True))[
        "release_numbers"
    ]
    assert not result.passed
    assert result.details["observed"] == {
        "barcodes": 5,
        "median_umi": 5.0,
        "median_genes": 2.0,
        "cells_per_strain": 1.33,
        "batch1_strains": 3,
    }
    assert result.details["expected"]["median_umi"] == 5158.0


def test_sourced_values_audit_both_mirrors(
    built: m.Nguessan2025Dataset, tmp_path: Path
) -> None:
    library, quote = _quote(tmp_path / "audit")
    bad = quote.model_copy(update={"quote": "not in the file"})
    result = _by_name(
        _verify(
            built,
            tmp_path,
            library_root=library,
            raw_root=library,
            sourced=[quote],
            release_sourced=[bad],
        )
    )["sourced_values"]
    assert not result.passed
    assert result.message.startswith("1/2 quotes present")


def test_parents_pinned_catches_a_wrong_member(
    built: m.Nguessan2025Dataset, tmp_path: Path
) -> None:
    result = _by_name(_verify(built, tmp_path, assembly_index={"AAA": "other.fa"}))[
        "parents_pinned"
    ]
    assert not result.passed
    assert result.details["problems"] == [
        "RM11-1a: AAA is not member GENOMES_ASSEMBLED/AAA_6.re.fa"
    ]


def test_parents_pinned_catches_by_off_the_reference(
    built: m.Nguessan2025Dataset, tmp_path: Path
) -> None:
    records = _records(built)
    records[0]["experiment"]["genotype"]["parent_1"]["assembly_ref"]["path"] = "x.fa"
    result = _by_name(_verify(built, tmp_path, records))["parents_pinned"]
    assert result.details["problems"] == ["BY4741: not the S288C reference sequence"]


def test_marker_reference_catches_a_ref_allele_off_the_s288c_base(
    built: m.Nguessan2025Dataset, tmp_path: Path
) -> None:
    genome = rel.StubGenome()
    genome.fasta_dna = {
        "chrI": rel.StubGenome._Seq("T" * 300),
        "chrIII": rel.StubGenome._Seq("T" * 150),
    }
    result = _by_name(_verify(built, tmp_path, genome=genome))["marker_reference"]
    assert not result.passed
    assert (result.details["n_checked"], result.details["n_match"]) == (4, 0)


def test_gene_containment_catches_a_gene_outside_the_reference(
    built: m.Nguessan2025Dataset, tmp_path: Path
) -> None:
    result = _by_name(
        _verify(built, tmp_path, sgd_genes={"YAL001C", "YAL002W", "YCL001W"})
    )["gene_containment_sgd"]
    assert not result.passed
    assert result.details["outside_reference"] == ["YFL026W"]


def test_mat_fixation_catches_alpha_expression(
    built: m.Nguessan2025Dataset, tmp_path: Path
) -> None:
    records = _records(built)
    reference = records[0]["reference"]["phenotype_reference"]
    reference["umi_count"]["YPL187W"] = 1
    results = _by_name(_verify(built, tmp_path, [records[0]]))
    assert not results["mat_fixation"].passed
    assert results["mat_fixation"].details["alpha_specific_umis"] == 1


def test_cis_statistics_separate_a_planted_locus_effect_from_none() -> None:
    rng = np.random.default_rng(1)
    genotype = (rng.random((400, 30)) > 0.5).astype(float)
    planted = genotype * 2.0 + rng.normal(size=(400, 30))
    stats = v.cis_statistics(planted, genotype, permutations=20, seed=0)
    assert stats["observed_rate"] == 1.0
    assert stats["null_rate"] == pytest.approx(0.01, abs=0.005)
    noise = rng.normal(size=(400, 30))
    null = v.cis_statistics(noise, genotype, permutations=20, seed=0)
    assert null["observed_rate"] < 5 * (1 - v.CIS_NULL_QUANTILE)


def test_cis_concordance_needs_a_testable_gene(
    built: m.Nguessan2025Dataset, tmp_path: Path
) -> None:
    result = _by_name(_verify(built, tmp_path, gene_midpoints={}))["cis_concordance"]
    assert not result.passed
    assert result.message == "no gene could be tested"


def test_gene_midpoints_skip_genes_the_genome_does_not_annotate() -> None:
    midpoints = v.gene_midpoints_from_genome(rel.genome(), ["YAL001C", "YZZ999W"])
    assert midpoints == {"YAL001C": ("chrI", 150)}
