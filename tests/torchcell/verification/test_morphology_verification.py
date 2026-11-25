# tests/torchcell/verification/test_morphology_verification.py
"""Unit tests for the WS6 Ohya CalMorph morphology verifier (synthetic, CI-safe).

Builds records from the real pydantic models (schema-valid by construction, with the
FULL 281 base + 220 CV vocabulary so L1 completeness passes) and checks that a correct
dataset passes every level and that each failure mode -- dropped parameter, negative
CV, non-finite value, wrong count, under-populated reference -- is caught by the level
it belongs to.
"""

from __future__ import annotations

from typing import Any

from torchcell.datamodels.calmorph_labels import CALMORPH_LABELS, CALMORPH_STATISTICS
from torchcell.datamodels.schema import (
    CalMorphExperiment,
    CalMorphExperimentReference,
    CalMorphPhenotype,
    Environment,
    Genotype,
    KanMxDeletionPerturbation,
    Media,
    ReferenceGenome,
    Temperature,
)
from torchcell.sequence.genome.registry import (
    SGD_S288C_R64,
    SGD_S288C_R64_GENBANK_ACCESSION,
)
from torchcell.verification.morphology import (
    perturbed_gene_set,
    verify_morphology_dataset,
)
from torchcell.verification.report import Level, Provenance

PROV = Provenance(source_uri="test://synthetic", citation_key="test2005")
GENES = ["YAL002W", "YAL004W", "YAL005C"]


def _full_base(fill: float = 1.0) -> dict[str, float]:
    return {k: fill for k in CALMORPH_LABELS}


def _full_cv(fill: float = 0.5) -> dict[str, float]:
    return {k: fill for k in CALMORPH_STATISTICS}


def _record(
    gene: str,
    *,
    base: dict[str, float] | None = None,
    cv: dict[str, float] | None = None,
    ref_base: dict[str, float] | None = None,
    ref_cv: dict[str, float] | None = None,
    replicate_id: str | None = None,
) -> dict[str, Any]:
    """A schema-valid CalMorph {experiment, reference} record for one deletion mutant."""
    phenotype = CalMorphPhenotype(
        calmorph=base if base is not None else _full_base(),
        calmorph_coefficient_of_variation=cv if cv is not None else _full_cv(),
        replicate_id=replicate_id,
    )
    reference_phenotype = CalMorphPhenotype(
        calmorph=ref_base if ref_base is not None else _full_base(),
        calmorph_coefficient_of_variation=ref_cv if ref_cv is not None else _full_cv(),
    )
    environment = Environment(
        media=Media(name="YEPD", state="solid", is_synthetic=False),
        temperature=Temperature(value=30),
    )
    experiment = CalMorphExperiment(
        dataset_name="test",
        genotype=Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=gene, perturbed_gene_name=gene
                )
            ]
        ),
        environment=environment,
        phenotype=phenotype,
    )
    reference = CalMorphExperimentReference(
        dataset_name="test",
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae",
            strain="BY4741",
            assembly_set=SGD_S288C_R64,
            assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
        ),
        environment_reference=environment.model_copy(),
        phenotype_reference=reference_phenotype,
    )
    return {"experiment": experiment.model_dump(), "reference": reference.model_dump()}


def _good_records() -> list[dict[str, Any]]:
    return [_record(g) for g in GENES]


def test_good_dataset_passes_all_levels():
    report = verify_morphology_dataset(
        _good_records(),
        dataset_name="good",
        provenance=PROV,
        expected_count=3,
        expected_replicate_records=0,
    )
    assert report.passed, report.summary()
    assert {Level.L0, Level.L1, Level.L2, Level.L3} <= report.levels_covered


def test_dropped_parameter_fails_completeness():
    partial = _full_base()
    partial.pop(next(iter(CALMORPH_LABELS)))  # drop one base parameter
    records = _good_records()
    records.append(_record("YAL007C", base=partial))
    report = verify_morphology_dataset(
        records,
        dataset_name="dropped",
        provenance=PROV,
        expected_count=4,
        expected_replicate_records=0,
    )
    comp = [r for r in report.results if r.name == "calmorph_completeness"]
    assert comp and not comp[0].passed


def test_negative_cv_fails():
    bad_cv = _full_cv()
    bad_cv[next(iter(CALMORPH_STATISTICS))] = -0.5  # CV cannot be negative
    records = _good_records()
    records.append(_record("YAL007C", cv=bad_cv))
    report = verify_morphology_dataset(
        records,
        dataset_name="negcv",
        provenance=PROV,
        expected_count=4,
        expected_replicate_records=0,
    )
    cv = [r for r in report.results if r.name == "cv_nonnegative"]
    assert cv and not cv[0].passed


def test_non_finite_base_value_fails():
    # inf passes the schema's NaN-only validator but must be caught by L2 finiteness.
    inf_base = _full_base()
    inf_base[next(iter(CALMORPH_LABELS))] = float("inf")
    records = _good_records()
    records.append(_record("YAL007C", base=inf_base))
    report = verify_morphology_dataset(
        records,
        dataset_name="inf",
        provenance=PROV,
        expected_count=4,
        expected_replicate_records=0,
    )
    vf = [
        r for r in report.results if r.level is Level.L2 and r.name == "value_fidelity"
    ]
    assert vf and not vf[0].passed


def test_wrong_count_fails():
    report = verify_morphology_dataset(
        _good_records(),
        dataset_name="miscount",
        provenance=PROV,
        expected_count=99,
        expected_replicate_records=0,
    )
    count = [r for r in report.results if r.level is Level.L1 and r.name == "count"]
    assert count and not count[0].passed


def test_underpopulated_reference_fails():
    partial_ref = _full_base()
    partial_ref.pop(next(iter(CALMORPH_LABELS)))
    records = _good_records()
    records.append(_record("YAL007C", ref_base=partial_ref))
    report = verify_morphology_dataset(
        records,
        dataset_name="badref",
        provenance=PROV,
        expected_count=4,
        expected_replicate_records=0,
    )
    ref = [r for r in report.results if r.name == "reference_populated"]
    assert ref and not ref[0].passed


def test_perturbed_gene_set():
    assert perturbed_gene_set(_good_records()) == set(GENES)


def _rule(report: Any, name: str) -> Any:
    return next(r for r in report.results if r.name == name)


def test_wild_type_replicates_of_one_strain_are_distinct_records() -> None:
    """#1178: three mutants plus two wild-type sample records of one genotype, told
    apart by their replicate_id, pass uniqueness and the declared replicate count.
    """
    records = _good_records() + [
        _record("YOR202W", replicate_id="04his3-1"),
        _record("YOR202W", replicate_id="04his3-2"),
    ]
    report = verify_morphology_dataset(
        records,
        dataset_name="replicates",
        provenance=PROV,
        expected_count=5,
        expected_replicate_records=2,
    )
    assert report.passed, report.summary()
    assert _rule(report, "record_uniqueness").details == {
        "n_keys": 5,
        "n_duplicated": 0,
        "n_genotypes": 4,
    }
    assert _rule(report, "replicate_records").details == {
        "observed": 2,
        "expected": 2,
        "n_distinct": 2,
    }


def test_a_strain_stored_twice_fails_uniqueness() -> None:
    """Two records of one genotype and environment with no replicate_id are one key."""
    records = _good_records() + [_record(GENES[0])]
    report = verify_morphology_dataset(
        records,
        dataset_name="dup",
        provenance=PROV,
        expected_count=4,
        expected_replicate_records=0,
    )
    rule = _rule(report, "record_uniqueness")
    assert not rule.passed
    assert rule.details["n_duplicated"] == 1


def test_replicate_records_fail_on_a_wrong_count_or_a_repeated_id() -> None:
    """The count must equal the declared one and the ids must be distinct."""
    repeated = _good_records() + [
        _record("YOR202W", replicate_id="w1"),
        _record("YOR202W", replicate_id="w1"),
    ]
    report = verify_morphology_dataset(
        repeated,
        dataset_name="rep",
        provenance=PROV,
        expected_count=5,
        expected_replicate_records=2,
    )
    assert not _rule(report, "replicate_records").passed
    assert _rule(report, "replicate_records").details["n_distinct"] == 1
    assert not _rule(report, "record_uniqueness").passed
    short = verify_morphology_dataset(
        _good_records(),
        dataset_name="short",
        provenance=PROV,
        expected_count=3,
        expected_replicate_records=1,
    )
    assert _rule(short, "replicate_records").details == {
        "observed": 0,
        "expected": 1,
        "n_distinct": 0,
    }
    assert not short.passed
