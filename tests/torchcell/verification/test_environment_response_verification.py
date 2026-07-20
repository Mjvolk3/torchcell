# tests/torchcell/verification/test_environment_response_verification.py
"""Unit tests for the WS15 environment-response verifier + the ProvenanceGap pass.

Closes a real coverage gap: no env-response verification test existed. Also exercises
the ProvenanceGap affordance end-to-end -- the Phenotype-level honesty invariants (L0)
and the informational L1 ``provenance_gaps`` census emitted by both the eager and
streaming verifiers (a documented gap is a PASS, and its deferred fields form a worklist).
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from torchcell.datamodels.schema import (
    Compound,
    Concentration,
    DoseBasis,
    Environment,
    EnvironmentResponseExperiment,
    EnvironmentResponseExperimentReference,
    EnvironmentResponsePhenotype,
    Genotype,
    KanMxDeletionPerturbation,
    MeasurementType,
    Media,
    ReferenceGenome,
    SampleUnit,
    SmallMoleculePerturbation,
    Temperature,
)
from torchcell.verification import Provenance, ProvenanceGap, ProvenanceGapReason
from torchcell.verification.environment_response import (
    environment_response_gene_set,
    verify_environment_response_dataset,
    verify_environment_response_dataset_streaming,
)
from torchcell.verification.report import Level

PROV = Provenance(
    source_uri="test://synthetic", citation_key="turcoGlobalAnalysisYeast2023"
)
GENES = ["YAL001C", "YBR085W", "YJR155W"]
SGD = set(GENES)


def _env() -> Environment:
    return Environment(
        media=Media(name="SD", state="liquid", is_synthetic=True),
        temperature=Temperature(value=30),
        perturbations=[
            SmallMoleculePerturbation(
                compound=Compound(name="hydroquinone"),
                concentration=Concentration(basis=DoseBasis.IC30),
            )
        ],
    )


def _phenotype(
    value: float, *, gaps: list[ProvenanceGap] | None = None
) -> EnvironmentResponsePhenotype:
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.log2_ratio,
        environment_response=value,
        units="log2(treatment/control)",
        provenance_gaps=gaps or [],
    )


def _record(
    gene: str, value: float, *, gaps: list[ProvenanceGap] | None = None
) -> dict[str, Any]:
    env = _env()
    exp = EnvironmentResponseExperiment(
        dataset_name="test",
        genotype=Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=gene, perturbed_gene_name=gene
                )
            ]
        ),
        environment=env,
        phenotype=_phenotype(value, gaps=gaps),
    )
    ref = EnvironmentResponseExperimentReference(
        dataset_name="test",
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="BY4741"
        ),
        environment_reference=env.model_copy(),
        phenotype_reference=_phenotype(0.0),
    )
    return {"experiment": exp.model_dump(), "reference": ref.model_dump()}


def _good_records() -> list[dict[str, Any]]:
    return [_record(g, v) for g, v in zip(GENES, [-1.2, 0.8, -0.3])]


# --- verifier core ----------------------------------------------------------- #
def test_good_dataset_passes_all_levels():
    report = verify_environment_response_dataset(
        _good_records(), dataset_name="good", provenance=PROV, expected_count=3
    )
    assert report.passed, report.summary()
    assert {Level.L0, Level.L1, Level.L2, Level.L3} <= report.levels_covered


def test_duplicate_pair_fails_uniqueness():
    records = _good_records()
    records.append(_record("YAL001C", -1.2))  # same strain + condition -> duplicate
    report = verify_environment_response_dataset(
        records, dataset_name="dup", provenance=PROV, expected_count=4
    )
    u = [r for r in report.results if r.name == "pair_uniqueness"]
    assert u and not u[0].passed


def test_gene_set_helper():
    assert environment_response_gene_set(_good_records()) == SGD


# --- ProvenanceGap: schema-level honesty invariants (L0) --------------------- #
def test_gapped_field_must_be_none():
    """Cannot both store a value and declare it missing."""
    with pytest.raises(ValidationError):
        EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.log2_ratio,
            environment_response=-1.0,
            n_samples=3,
            sample_unit=SampleUnit.biological_replicate,
            provenance_gaps=[
                ProvenanceGap(
                    field="n_samples",
                    reason=ProvenanceGapReason.not_reported_by_primary,
                )
            ],
        )


def test_gap_field_must_exist():
    with pytest.raises(ValidationError):
        EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.log2_ratio,
            environment_response=-1.0,
            provenance_gaps=[
                ProvenanceGap(
                    field="not_a_real_field",
                    reason=ProvenanceGapReason.not_reported_by_primary,
                )
            ],
        )


def test_valid_gap_leaves_field_none():
    p = _phenotype(
        -1.0,
        gaps=[
            ProvenanceGap(
                field="n_samples",
                reason=ProvenanceGapReason.deferred_pending_source_review,
                looked_in=PROV,
            )
        ],
    )
    assert p.n_samples is None
    assert len(p.provenance_gaps) == 1


# --- ProvenanceGap: verifier L1 census pass (eager + streaming) -------------- #
def _gapped_records() -> list[dict[str, Any]]:
    """One record fully sourced, two with deferred n_samples gaps (the worklist)."""
    gap = [
        ProvenanceGap(
            field="n_samples",
            reason=ProvenanceGapReason.deferred_pending_source_review,
            looked_in=PROV,
        )
    ]
    return [
        _record(GENES[0], -1.2),
        _record(GENES[1], 0.8, gaps=gap),
        _record(GENES[2], -0.3, gaps=gap),
    ]


def test_gap_pass_is_emitted_and_passes_eager():
    report = verify_environment_response_dataset(
        _gapped_records(), dataset_name="gaps", provenance=PROV, expected_count=3
    )
    assert report.passed, report.summary()  # documented gaps do NOT fail the build
    g = [r for r in report.results if r.name == "provenance_gaps"]
    assert g and g[0].passed
    assert g[0].details["n_gaps"] == 2
    assert g[0].details["n_records_with_gaps"] == 2
    assert g[0].details["worklist_fields"] == ["n_samples"]


def test_gap_pass_is_emitted_streaming():
    report = verify_environment_response_dataset_streaming(
        iter(_gapped_records()),
        dataset_name="gaps-stream",
        provenance=PROV,
        expected_count=3,
        sgd_genes=SGD,
    )
    assert report.passed, report.summary()
    g = [r for r in report.results if r.name == "provenance_gaps"]
    assert g and g[0].passed
    assert g[0].details["n_gaps"] == 2
    assert g[0].details["worklist_fields"] == ["n_samples"]


def test_no_gaps_reports_fully_sourced():
    report = verify_environment_response_dataset(
        _good_records(), dataset_name="clean", provenance=PROV, expected_count=3
    )
    g = [r for r in report.results if r.name == "provenance_gaps"]
    assert g and g[0].passed and g[0].details["n_gaps"] == 0


# --- Environment-level ProvenanceGap (temperature not carried by curation) ----- #
def test_environment_temperature_is_optional_and_gappable():
    """A curation layer (YeastPhenome) may not carry temperature -> typed absence."""
    env = Environment(
        media=Media(name="YPD", state="liquid", is_synthetic=False),
        temperature=None,
        provenance_gaps=[
            ProvenanceGap(
                field="temperature",
                reason=ProvenanceGapReason.not_carried_by_curation,
                looked_in=PROV,
            )
        ],
    )
    assert env.temperature is None
    assert env.provenance_gaps[0].field == "temperature"


def test_environment_gap_honesty_invariant():
    """A gapped temperature must be None -- cannot both set it and declare it missing."""
    with pytest.raises(ValidationError):
        Environment(
            media=Media(name="YPD", state="liquid", is_synthetic=False),
            temperature=Temperature(value=30),
            provenance_gaps=[
                ProvenanceGap(
                    field="temperature",
                    reason=ProvenanceGapReason.not_carried_by_curation,
                )
            ],
        )


def _temp_gapped_record(gene: str, value: float) -> dict[str, Any]:
    """A record whose environment carries a temperature ProvenanceGap (temp=None)."""
    env = Environment(
        media=Media(name="YPD", state="liquid", is_synthetic=False),
        temperature=None,
        provenance_gaps=[
            ProvenanceGap(
                field="temperature",
                reason=ProvenanceGapReason.not_carried_by_curation,
                looked_in=PROV,
            )
        ],
        perturbations=[
            SmallMoleculePerturbation(
                compound=Compound(name="quinine"),
                concentration=Concentration(basis=DoseBasis.IC30),
            )
        ],
    )
    exp = EnvironmentResponseExperiment(
        dataset_name="test",
        genotype=Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=gene, perturbed_gene_name=gene
                )
            ]
        ),
        environment=env,
        phenotype=_phenotype(value),
    )
    ref = EnvironmentResponseExperimentReference(
        dataset_name="test",
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="BY4743", ploidy="diploid"
        ),
        environment_reference=env.model_copy(),
        phenotype_reference=_phenotype(0.0),
    )
    return {"experiment": exp.model_dump(), "reference": ref.model_dump()}


def test_environment_gap_counted_in_census_eager_and_streaming():
    """The L1 census aggregates BOTH phenotype and environment gaps; a documented
    temperature gap passes and appears in by_field.
    """
    recs = [_temp_gapped_record(g, v) for g, v in zip(GENES, [-1.2, 0.8, -0.3])]
    eager = verify_environment_response_dataset(
        recs, dataset_name="tgap", provenance=PROV, expected_count=3
    )
    stream = verify_environment_response_dataset_streaming(
        iter(recs),
        dataset_name="tgap-s",
        provenance=PROV,
        expected_count=3,
        sgd_genes=SGD,
    )
    for report in (eager, stream):
        assert report.passed, report.summary()  # documented env gap does NOT fail
        g = [r for r in report.results if r.name == "provenance_gaps"][0]
        assert g.passed
        assert g.details["by_field"]["temperature"] == 3
