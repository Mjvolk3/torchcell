# tests/torchcell/verification/test_expression_verification.py
"""Unit tests for the WS5 expression dataset verifier (synthetic, CI-safe).

Builds records from the real pydantic models (so they are schema-valid by
construction) and checks that a correct dataset passes every level and that each
failure mode -- sign inversion, non-zero reference, wrong count, dropped gene --
is caught by the level it belongs to.
"""

from __future__ import annotations

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
