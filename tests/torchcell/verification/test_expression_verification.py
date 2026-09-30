# tests/torchcell/verification/test_expression_verification.py
"""Unit tests for the WS5 expression dataset verifier (synthetic, CI-safe).

Builds records from the real pydantic models (so they are schema-valid by
construction) and checks that a correct dataset passes every level and that each
failure mode -- sign inversion, non-zero reference, wrong count, dropped gene --
is caught by the level it belongs to.

2026.09.30 (Phase 17): every level's verdict and exact message on the three-mutant good
table (4 genes x 3 records = 12 log2 values, SEs and replicate counts; deleted-gene log2
-3.0, -2.5, -4.0, so the median is -3.000 and all three are negative), then the exact
message of each failure mode. The orientation rule is pinned at its boundary: deleted-gene
values -1.0 and +1.0 have median 0.000, which is NOT < 0 and fails. A deleted gene
missing from the platform map is counted as absent, a perturbation with no systematic
name is not counted at all, and a table where no deleted gene is on the map fails with
its own message. A NaN SE is allowed (single replicate) while a replicate count of 0
fails ``n_replicates_ge_1``.
"""

from __future__ import annotations

import math
from typing import Any

from torchcell.datamodels.schema import (
    Environment,
    Genotype,
    Media,
    MicroarrayExpressionExperiment,
    MicroarrayExpressionExperimentReference,
    MicroarrayExpressionPhenotype,
    ReferenceGenome,
    SgaKanMxDeletionPerturbation,
    Temperature,
)
from torchcell.verification.expression import (
    measured_gene_universe,
    verify_expression_dataset,
)
from torchcell.verification.report import Level, Provenance

GENES = ["YAL001C", "YAL002W", "YAL003W", "YBR001C"]
PROV = Provenance(source_uri="test://synthetic", citation_key="test2025")


def _record(
    deleted_gene: str,
    log2: dict[str, float],
    *,
    ref_log2_nonzero: bool = False,
    drop_gene: str | None = None,
) -> dict[str, Any]:
    """A schema-valid {experiment, reference} record for one deletion mutant."""
    genes = [g for g in GENES if g != drop_gene]
    expr = {g: 100.0 for g in genes}
    log2_map = {g: log2.get(g, 0.1) for g in genes}
    se_map = {g: 0.05 for g in genes}
    var_map = {g: 0.0025 for g in genes}
    n_map = {g: 4 for g in genes}

    phenotype = MicroarrayExpressionPhenotype(
        expression=expr,
        expression_log2_ratio=log2_map,
        expression_log2_ratio_se=se_map,
        expression_log2_ratio_variance=var_map,
        n_replicates=n_map,
    )
    ref_value = 0.5 if ref_log2_nonzero else 0.0
    reference_phenotype = MicroarrayExpressionPhenotype(
        expression=expr,
        expression_log2_ratio={g: ref_value for g in genes},
        expression_log2_ratio_se=None,
        expression_log2_ratio_variance=None,
        n_replicates={g: 1 for g in genes},
    )
    environment = Environment(
        media=Media(name="SC", state="liquid", is_synthetic=True),
        temperature=Temperature(value=30),
    )
    experiment = MicroarrayExpressionExperiment(
        dataset_name="test",
        genotype=Genotype(
            perturbations=[
                SgaKanMxDeletionPerturbation(
                    systematic_gene_name=deleted_gene,
                    perturbed_gene_name=deleted_gene,
                    strain_id=f"KanMX_{deleted_gene}",
                )
            ]
        ),
        environment=environment,
        phenotype=phenotype,
    )
    reference = MicroarrayExpressionExperimentReference(
        dataset_name="test",
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="BY4742"
        ),
        environment_reference=environment.model_copy(),
        phenotype_reference=reference_phenotype,
    )
    return {"experiment": experiment.model_dump(), "reference": reference.model_dump()}


def _good_records() -> list[dict[str, Any]]:
    # Each deleted gene is strongly down-regulated (negative log2 at that gene).
    return [
        _record("YAL001C", {"YAL001C": -3.0, "YAL002W": 0.2}),
        _record("YAL002W", {"YAL002W": -2.5, "YAL003W": -0.1}),
        _record("YAL003W", {"YAL003W": -4.0, "YBR001C": 0.3}),
    ]


def test_good_dataset_passes_all_levels():
    report = verify_expression_dataset(
        _good_records(), dataset_name="good", provenance=PROV, expected_count=3
    )
    assert report.passed, report.summary()
    # L0-L3 all covered.
    assert {Level.L0, Level.L1, Level.L2, Level.L3} <= report.levels_covered


def test_sign_inversion_fails_orientation():
    # Deleted genes stored as strongly POSITIVE => log2(reference/sample) inversion.
    records = [
        _record("YAL001C", {"YAL001C": 3.0}),
        _record("YAL002W", {"YAL002W": 2.5}),
        _record("YAL003W", {"YAL003W": 4.0}),
    ]
    report = verify_expression_dataset(
        records, dataset_name="inverted", provenance=PROV, expected_count=3
    )
    assert not report.passed
    orient = [r for r in report.results if r.name == "deletion_downregulates"]
    assert orient and not orient[0].passed


def test_nonzero_reference_fails():
    records = [
        _record("YAL001C", {"YAL001C": -3.0}, ref_log2_nonzero=True),
        _record("YAL002W", {"YAL002W": -2.5}),
        _record("YAL003W", {"YAL003W": -4.0}),
    ]
    report = verify_expression_dataset(
        records, dataset_name="badref", provenance=PROV, expected_count=3
    )
    assert not report.passed
    ref = [r for r in report.results if r.name == "reference_log2_zero"]
    assert ref and not ref[0].passed


def test_wrong_count_fails():
    report = verify_expression_dataset(
        _good_records(), dataset_name="miscount", provenance=PROV, expected_count=99
    )
    count = [r for r in report.results if r.level is Level.L1 and r.name == "count"]
    assert count and not count[0].passed


def test_dropped_gene_fails_completeness():
    records = _good_records()
    # One record silently drops a measured gene.
    records.append(_record("YBR001C", {"YBR001C": -3.0}, drop_gene="YAL001C"))
    report = verify_expression_dataset(
        records, dataset_name="dropped", provenance=PROV, expected_count=4
    )
    comp = [r for r in report.results if r.name == "gene_completeness"]
    assert comp and not comp[0].passed


def test_measured_gene_universe():
    assert measured_gene_universe(_good_records()) == set(GENES)


# --- Phase 17: exact messages per level and per failure mode ------------------ #
def _result(records: list[dict[str, Any]], name: str) -> Any:
    report = verify_expression_dataset(
        records, dataset_name="t", provenance=PROV, expected_count=len(records)
    )
    return next(r for r in report.results if r.name == name)


def test_good_dataset_exact_results_in_order() -> None:
    """Eight results in the order the verifier adds them, each with its exact message."""
    report = verify_expression_dataset(
        _good_records(), dataset_name="good", provenance=PROV, expected_count=3
    )
    assert [(r.level, r.name, r.passed, r.message) for r in report.results] == [
        (Level.L0, "structural", True, "3 records validated"),
        (Level.L1, "count", True, "observed 3, expected 3"),
        (
            Level.L1,
            "gene_completeness",
            True,
            "all 3 records measure the full 4-gene universe",
        ),
        (Level.L2, "value_fidelity", True, "12 values checked"),
        (Level.L2, "se_nonnegative", True, "12 values checked"),
        (Level.L2, "n_replicates_ge_1", True, "12 values checked"),
        (
            Level.L3,
            "reference_log2_zero",
            True,
            "reference log2(sample/ref) == 0 for all 12 values",
        ),
        (
            Level.L3,
            "deletion_downregulates",
            True,
            "median deleted-gene log2=-3.000 (<0 => correct orientation); "
            "frac_neg=1.000 over 3 deleted genes (0 deleted genes absent from the "
            "platform map)",
        ),
    ]


def test_failure_messages_for_reference_count_and_dropped_gene() -> None:
    records = _good_records()
    records[1] = _record("YAL002W", {"YAL002W": -2.5}, ref_log2_nonzero=True)
    assert _result(records, "reference_log2_zero").message == (
        "reference log2 not identically zero: max|value|=0.5"
    )

    report = verify_expression_dataset(
        _good_records(), dataset_name="t", provenance=PROV, expected_count=99
    )
    count = next(r for r in report.results if r.name == "count")
    assert count.message == "observed 3, expected 99"

    records = _good_records() + [
        _record("YBR001C", {"YBR001C": -3.0}, drop_gene="YAL001C")
    ]
    completeness = _result(records, "gene_completeness")
    assert completeness.message == "1/4 records missing genes vs the universe"
    assert completeness.details["short"] == [{"index": 3, "n_missing": 1}]


def test_orientation_median_of_exactly_zero_fails() -> None:
    """-1.0 and +1.0 average to a median of 0.0, which the strict ``< 0`` rejects."""
    records = [
        _record("YAL001C", {"YAL001C": -1.0}),
        _record("YAL002W", {"YAL002W": 1.0}),
    ]
    result = _result(records, "deletion_downregulates")
    assert result.passed is False
    assert result.message == (
        "median deleted-gene log2=0.000 (<0 => correct orientation); frac_neg=0.500 "
        "over 2 deleted genes (0 deleted genes absent from the platform map)"
    )


def test_orientation_counts_off_platform_deletions_and_skips_unnamed_ones() -> None:
    """An off-map deletion is counted as absent; a nameless perturbation is not counted."""
    unnamed = _record("YAL001C", {"YAL001C": -3.0})
    unnamed["experiment"]["genotype"]["perturbations"][0]["systematic_gene_name"] = None
    records = _good_records() + [_record("YDR001C", {}), unnamed]
    result = _result(records, "deletion_downregulates")
    assert result.passed is True
    assert result.message == (
        "median deleted-gene log2=-3.000 (<0 => correct orientation); frac_neg=1.000 "
        "over 3 deleted genes (1 deleted genes absent from the platform map)"
    )


def test_orientation_fails_when_no_deleted_gene_is_on_the_map() -> None:
    result = _result([_record("YDR001C", {})], "deletion_downregulates")
    assert result.passed is False
    assert result.message == "no deleted genes were present in any expression map"


def test_nan_se_is_allowed_and_zero_replicates_fail() -> None:
    """A NaN SE passes (``allow_nan=True``); ``n_replicates`` 0 is ``< 1.0``."""
    records = _good_records()
    records[0]["experiment"]["phenotype"]["expression_log2_ratio_se"]["YAL001C"] = (
        math.nan
    )
    assert _result(records, "se_nonnegative").message == "12 values checked"

    records = _good_records()
    records[2]["experiment"]["phenotype"]["n_replicates"]["YBR001C"] = 0
    replicates = _result(records, "n_replicates_ge_1")
    assert replicates.passed is False
    assert replicates.message == "1/12 values invalid"
    assert replicates.details["bad"] == [{"index": 11, "value": 0.0, "reason": "< 1.0"}]
