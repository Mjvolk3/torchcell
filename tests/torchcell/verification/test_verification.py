"""Tests for the L0-L4 verification framework (report + levels)."""

import math
from collections.abc import Callable
from typing import Any

from pydantic import TypeAdapter

from torchcell.verification import (
    Level,
    LevelResult,
    Provenance,
    VerificationReport,
    l0_structural,
    l1_completeness,
    l1_count,
    l2_cross_method,
    l2_value_fidelity,
    l3_convention,
    l4_cross_source,
    sha256_file,
)


# --------------------------------------------------------------------------- #
# Report / provenance models
# --------------------------------------------------------------------------- #
def test_level_ordering_is_a_gate():
    assert Level.L0 < Level.L1 < Level.L2 < Level.L3 < Level.L4


def test_empty_report_is_not_passed():
    report = VerificationReport(
        dataset_name="empty", provenance=Provenance(source_uri="file://x")
    )
    assert report.passed is False


def test_report_passed_iff_all_levels_pass():
    report = VerificationReport(
        dataset_name="d", provenance=Provenance(source_uri="file://x")
    )
    report.add(LevelResult(level=Level.L0, name="a", passed=True, message="ok"))
    assert report.passed is True
    report.add(LevelResult(level=Level.L1, name="b", passed=False, message="no"))
    assert report.passed is False
    assert report.levels_covered == {Level.L0, Level.L1}
    assert "FAIL" in report.summary()


def test_sha256_file(tmp_path):
    p = tmp_path / "artifact.txt"
    p.write_bytes(b"hello world")
    # Known sha256 of "hello world".
    assert (
        sha256_file(p)
        == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    )


# --------------------------------------------------------------------------- #
# L0 structural -- run against the real schema union
# --------------------------------------------------------------------------- #
def _fitness_record():
    return {
        "experiment_type": "fitness",
        "dataset_name": "d",
        "genotype": {
            "perturbations": [
                {
                    "systematic_gene_name": "YAL001C",
                    "perturbed_gene_name": "TFC3",
                    "strain_id": "D1",
                    "perturbation_type": "deletion",
                    "deletion_type": "KanMX",
                }
            ]
        },
        "environment": {
            "media": {"name": "YEPD", "state": "solid"},
            "temperature": {"value": 30.0},
        },
        "phenotype": {
            "graph_level": "global",
            "label_name": "fitness",
            "label_statistic_name": "fitness_se",
            "fitness": 0.85,
            "fitness_se": 0.05,
            "n_samples": 4,
        },
    }


def test_l0_structural_passes_on_valid_records():
    from torchcell.datamodels.schema import ExperimentType

    validate: Callable[[Any], object] = TypeAdapter(ExperimentType).validate_python
    result = l0_structural([_fitness_record()], validate)
    assert result.passed
    assert result.level is Level.L0
    assert result.details["n_records"] == 1


def test_l0_structural_reports_bad_records():
    from torchcell.datamodels.schema import ExperimentType

    validate: Callable[[Any], object] = TypeAdapter(ExperimentType).validate_python
    bad = _fitness_record()
    bad["phenotype"]["fitness"] = float("nan")  # fitness cannot be NaN
    result = l0_structural([bad], validate)
    assert not result.passed
    assert result.details["n_failures"] == 1


# --------------------------------------------------------------------------- #
# L1 completeness / count
# --------------------------------------------------------------------------- #
def test_l1_completeness_flags_missing():
    ok = l1_completeness(["a", "b"], ["a", "b"])
    assert ok.passed
    missing = l1_completeness(["a"], ["a", "b"])
    assert not missing.passed
    assert "b" in missing.details["missing"]


def test_l1_completeness_extra_allowed_by_default():
    assert l1_completeness(["a", "b", "c"], ["a", "b"]).passed
    assert not l1_completeness(["a", "b", "c"], ["a", "b"], allow_extra=False).passed


def test_l1_count():
    assert l1_count(4718, 4718).passed
    assert not l1_count(4717, 4718).passed


# --------------------------------------------------------------------------- #
# L2 value fidelity / cross-method
# --------------------------------------------------------------------------- #
def test_l2_value_fidelity_range_and_nan():
    assert l2_value_fidelity([0.1, 0.9], minimum=0.0, maximum=1.0).passed
    assert not l2_value_fidelity([0.1, 1.5], maximum=1.0).passed
    assert not l2_value_fidelity([float("nan")]).passed
    assert l2_value_fidelity([float("nan")], allow_nan=True).passed
    assert not l2_value_fidelity([float("inf")]).passed
    assert not l2_value_fidelity([True]).passed  # bool is not a real measurement


def test_l2_cross_method():
    assert l2_cross_method([1.0, 2.0], [1.0, 2.0 + 1e-9]).passed
    bad = l2_cross_method([1.0, 2.0], [1.0, 2.5])
    assert not bad.passed
    assert bad.details["n_disagreements"] == 1
    assert not l2_cross_method([1.0], [1.0, 2.0]).passed  # length mismatch


# --------------------------------------------------------------------------- #
# L3 semantic / L4 cross-source
# --------------------------------------------------------------------------- #
def test_l3_convention():
    assert l3_convention("log2_orientation", True).passed
    assert not l3_convention("log2_orientation", False, detail="sign inverted").passed


def test_l4_cross_source():
    ok = l4_cross_source([("YAL001C", 0.5, 0.5)])
    assert ok.passed
    bad = l4_cross_source([("YAL001C", 0.5, 0.9)])
    assert not bad.passed
    assert bad.details["n_disagreements"] == 1


# --------------------------------------------------------------------------- #
# End-to-end assembly
# --------------------------------------------------------------------------- #
def test_report_assembly_end_to_end():
    from torchcell.datamodels.schema import ExperimentType

    validate: Callable[[Any], object] = TypeAdapter(ExperimentType).validate_python
    report = VerificationReport(
        dataset_name="fitness_demo",
        provenance=Provenance(
            source_uri="https://doi.org/10.1234/x",
            citation_key="authorTitleYear",
            sha256="deadbeef",
            method="pdftotext -layout",
            page="Table 1",
        ),
    )
    report.add(l0_structural([_fitness_record()], validate))
    report.add(l1_count(1, 1))
    report.add(l2_value_fidelity([0.85], minimum=0.0))
    report.add(l3_convention("fitness_ratio", True, detail="fitness = ko/wt"))
    assert report.passed
    assert report.levels_covered == {Level.L0, Level.L1, Level.L2, Level.L3}
    assert not math.isnan(0.0)  # sanity
